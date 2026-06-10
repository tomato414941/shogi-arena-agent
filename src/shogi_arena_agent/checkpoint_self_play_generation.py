from __future__ import annotations

import json
import multiprocessing as mp
import queue
import random
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from statistics import mean
from time import perf_counter

import cshogi
import shogi

from shogi_arena_agent.board_backend import ShogiBoard, board_is_black_turn, board_turn_name, copy_board, legal_move_usis
from shogi_arena_agent.checkpoint_self_play_evaluator import (
    CentralPolicyValueEvaluator,
    ProcessCentralPolicyValueEvaluator,
    ProcessQueuedPolicyValueEvaluator,
)
from shogi_arena_agent.mcts_config import (
    MctsConfig,
    MoveSelectionConfig,
    visit_sampling_move_selection_config,
)
from shogi_arena_agent.mcts_evaluator import MovePriors, PolicyValueEvaluator
from shogi_arena_agent.mcts_performance import MctsMovePerformance, MultiPositionMctsPerformance, leaf_eval_batch_metrics
from shogi_arena_agent.mcts_tree import position_ply
from shogi_arena_agent.model_policy import ShogiMoveChoiceCheckpointEvaluator
from shogi_arena_agent.shogi_game import (
    ShogiActorSpec,
    ShogiDecisionTelemetry,
    ShogiGameRecord,
    ShogiTransitionRecord,
    position_command,
)
from shogi_arena_agent.shogi_generation import records_summary
from shogi_arena_agent.start_positions import StartPosition, startpos
from shogi_arena_agent.usi import RESIGN_MOVE, UsiPosition, board_from_position

GenerationProgressCallback = Callable[[dict[str, object]], None]
ShogiGameRecordCallback = Callable[[ShogiGameRecord], None]
_SelfPlayMove = int | str

_ACTION_PLANE_DIRECTION_DELTAS = (-10, -9, -8, -1, 1, 8, 9, 10)
_ACTION_PLANE_KNIGHT_DELTAS = (-19, -17)
_ACTION_PLANE_MOVE_TYPE_DELTAS = _ACTION_PLANE_DIRECTION_DELTAS + _ACTION_PLANE_KNIGHT_DELTAS
_ACTION_PLANE_MOVE_TYPE_OFFSET_BY_DELTA = {
    delta: index for index, delta in enumerate(_ACTION_PLANE_MOVE_TYPE_DELTAS)
}
_ACTION_PLANE_PROMOTE_MOVE_TYPE_OFFSET = len(_ACTION_PLANE_MOVE_TYPE_DELTAS)
_ACTION_PLANE_DROP_MOVE_TYPE_OFFSET = _ACTION_PLANE_PROMOTE_MOVE_TYPE_OFFSET + len(_ACTION_PLANE_MOVE_TYPE_DELTAS)
_ACTION_PLANE_MOVE_TYPE_COUNT = _ACTION_PLANE_DROP_MOVE_TYPE_OFFSET + 7


@dataclass(frozen=True)
class CheckpointSelfPlayConfig:
    checkpoint: str
    games: int
    concurrent_games_per_process: int
    max_plies: int
    mcts_simulations: int
    nn_leaf_eval_batch_limit: int
    device: str
    board_backend: str
    checkpoint_id: str | None = None
    inference_precision: str = "fp32"
    compile_model: bool = False
    move_selection: MoveSelectionConfig | None = None
    self_play_worker_processes: int = 1
    central_evaluator_batch_size_limit: int | None = None
    central_evaluator_flush_timeout_sec: float = 0.002
    progress_every_plies: int = 0
    start_positions: tuple[StartPosition, ...] = ()


@dataclass(frozen=True)
class CheckpointSelfPlayGenerationResult:
    records: tuple[ShogiGameRecord, ...]
    central_evaluator_performance: dict[str, object]


@dataclass(frozen=True)
class _WorkerRecordMessage:
    record_index: int
    record: ShogiGameRecord


@dataclass(frozen=True)
class _WorkerProgressMessage:
    worker_id: int
    payload: dict[str, object]


@dataclass(frozen=True)
class _WorkerCompleteMessage:
    worker_id: int


@dataclass(frozen=True)
class _WorkerErrorMessage:
    worker_id: int
    error_message: str


def run_checkpoint_self_play_generation(
    config: CheckpointSelfPlayConfig,
    *,
    checkpoint_evaluator_cls: type[ShogiMoveChoiceCheckpointEvaluator] = ShogiMoveChoiceCheckpointEvaluator,
    record_callback: ShogiGameRecordCallback | None = None,
    progress_callback: GenerationProgressCallback | None = None,
) -> CheckpointSelfPlayGenerationResult:
    _validate_config(config)
    move_selection = config.move_selection or visit_sampling_move_selection_config(seed=None)
    actor = _checkpoint_self_play_actor(config, move_selection)
    checkpoint_evaluator = checkpoint_evaluator_cls.from_checkpoint(
        config.checkpoint,
        device=config.device,
        precision=config.inference_precision,
        compile_model=config.compile_model,
    )
    use_action_indices = _can_use_self_play_action_indices(config, checkpoint_evaluator)
    central_batch_limit = config.central_evaluator_batch_size_limit or config.nn_leaf_eval_batch_limit
    if config.self_play_worker_processes == 1:
        with CentralPolicyValueEvaluator(
            checkpoint_evaluator,
            batch_size_limit=central_batch_limit,
            flush_timeout_sec=config.central_evaluator_flush_timeout_sec,
        ) as central_evaluator:
            selector = _checkpoint_self_play_selector(
                config,
                move_selection,
                evaluator=central_evaluator.client(),
                use_action_indices=use_action_indices,
            )
            records = _generate_checkpoint_self_play_games_with_selector(
                config,
                actor=actor,
                selector=selector,
                record_callback=record_callback,
                progress_callback=progress_callback,
            )
            central_evaluator_performance = central_evaluator.performance_summary()
    else:
        records, central_evaluator_performance = _generate_checkpoint_self_play_games_with_process_workers(
            config,
            actor=actor,
            move_selection=move_selection,
            evaluate_positions=checkpoint_evaluator,
            use_action_indices=use_action_indices,
            record_callback=record_callback,
            progress_callback=progress_callback,
        )
    return CheckpointSelfPlayGenerationResult(
        records=records,
        central_evaluator_performance=central_evaluator_performance,
    )


def _generate_checkpoint_self_play_games_with_process_workers(
    config: CheckpointSelfPlayConfig,
    *,
    actor: ShogiActorSpec,
    move_selection: MoveSelectionConfig,
    evaluate_positions: Callable[[Sequence[tuple[str, tuple[str, ...]]]], list[tuple[MovePriors, float]]],
    use_action_indices: bool,
    record_callback: ShogiGameRecordCallback | None,
    progress_callback: GenerationProgressCallback | None,
) -> tuple[tuple[ShogiGameRecord, ...], dict[str, object]]:
    context = mp.get_context("spawn")
    request_queue = context.Queue()
    event_queue = context.Queue()
    records_by_index: dict[int, ShogiGameRecord] = {}
    response_queues = {worker_id: context.Queue() for worker_id in range(config.self_play_worker_processes)}
    worker_counts = _worker_game_counts(config.games, config.self_play_worker_processes)
    start_index = 0
    processes: list[mp.Process] = []
    worker_ids: list[int] = []
    for worker_index, game_count in enumerate(worker_counts):
        if game_count <= 0:
            continue
        worker_start_index = start_index
        start_index += game_count
        worker_config = _worker_config(config, game_count=game_count, start_index=worker_start_index)
        process = context.Process(
            target=_run_checkpoint_self_play_worker_process,
            kwargs={
                "worker_id": worker_index,
                "start_index": worker_start_index,
                "config": worker_config,
                "actor": actor,
                "move_selection": _worker_move_selection(move_selection, worker_index),
                "use_action_indices": use_action_indices,
                "request_queue": request_queue,
                "response_queue": response_queues[worker_index],
                "event_queue": event_queue,
            },
            name=f"checkpoint-self-play-worker-{worker_index}",
        )
        processes.append(process)
        worker_ids.append(worker_index)

    central_batch_limit = config.central_evaluator_batch_size_limit or config.nn_leaf_eval_batch_limit
    with ProcessCentralPolicyValueEvaluator(
        evaluate_positions,
        request_queue=request_queue,
        response_queues=response_queues,
        batch_size_limit=central_batch_limit,
        flush_timeout_sec=config.central_evaluator_flush_timeout_sec,
    ) as process_central_evaluator:
        for process in processes:
            process.start()
        _collect_worker_events(
            event_queue,
            processes=processes,
            worker_ids=worker_ids,
            records_by_index=records_by_index,
            record_callback=record_callback,
            progress_callback=progress_callback,
        )
        central_evaluator_performance = process_central_evaluator.performance_summary()
    for process in processes:
        process.join()
        if process.exitcode not in (0, None):
            raise RuntimeError(f"checkpoint self-play worker {process.name} exited with code {process.exitcode}")
    missing_indexes = [index for index in range(config.games) if index not in records_by_index]
    if missing_indexes:
        raise RuntimeError(f"checkpoint self-play workers did not return records for indexes: {missing_indexes}")
    return tuple(records_by_index[index] for index in range(config.games)), central_evaluator_performance


def _run_checkpoint_self_play_worker_process(
    *,
    worker_id: int,
    start_index: int,
    config: CheckpointSelfPlayConfig,
    actor: ShogiActorSpec,
    move_selection: MoveSelectionConfig,
    use_action_indices: bool,
    request_queue: object,
    response_queue: object,
    event_queue: object,
) -> None:
    try:
        selector = _checkpoint_self_play_selector(
            config,
            move_selection,
            evaluator=ProcessQueuedPolicyValueEvaluator(
                worker_id=worker_id,
                request_queue=request_queue,
                response_queue=response_queue,
            ),
            use_action_indices=use_action_indices,
        )
        next_record_index = start_index

        def write_record(record: ShogiGameRecord) -> None:
            nonlocal next_record_index
            event_queue.put(_WorkerRecordMessage(next_record_index, record))
            next_record_index += 1

        def write_progress(payload: dict[str, object]) -> None:
            event_queue.put(_WorkerProgressMessage(worker_id, payload))

        _generate_checkpoint_self_play_games_with_selector(
            config,
            actor=actor,
            selector=selector,
            record_callback=write_record,
            progress_callback=write_progress,
        )
        event_queue.put(_WorkerCompleteMessage(worker_id))
    except BaseException:
        event_queue.put(_WorkerErrorMessage(worker_id, traceback.format_exc()))


def _collect_worker_events(
    event_queue: object,
    *,
    processes: Sequence[mp.Process],
    worker_ids: Sequence[int],
    records_by_index: dict[int, ShogiGameRecord],
    record_callback: ShogiGameRecordCallback | None,
    progress_callback: GenerationProgressCallback | None,
) -> None:
    remaining_workers = set(worker_ids)
    while remaining_workers:
        try:
            message = event_queue.get(timeout=0.5)
        except queue.Empty:
            exited_workers = [
                (worker_id, process.exitcode)
                for worker_id, process in zip(worker_ids, processes, strict=True)
                if worker_id in remaining_workers and process.exitcode is not None
            ]
            if exited_workers and len(exited_workers) == len(remaining_workers):
                raise RuntimeError(f"checkpoint self-play workers exited before completion: {exited_workers}")
            for worker_id, exitcode in exited_workers:
                if exitcode not in (0, None):
                    raise RuntimeError(f"checkpoint self-play worker {worker_id} exited with code {exitcode}")
            continue
        if isinstance(message, _WorkerRecordMessage):
            records_by_index[message.record_index] = message.record
            if record_callback is not None:
                record_callback(message.record)
            continue
        if isinstance(message, _WorkerProgressMessage):
            if progress_callback is not None:
                progress_callback(message.payload)
            continue
        if isinstance(message, _WorkerCompleteMessage):
            remaining_workers.discard(message.worker_id)
            continue
        if isinstance(message, _WorkerErrorMessage):
            for process in processes:
                if process.is_alive():
                    process.terminate()
            raise RuntimeError(f"checkpoint self-play worker {message.worker_id} failed: {message.error_message}")
        raise RuntimeError(f"unknown checkpoint self-play worker message: {message!r}")


def _generate_checkpoint_self_play_games_with_selector(
    config: CheckpointSelfPlayConfig,
    *,
    actor: ShogiActorSpec,
    selector: "_CheckpointSelfPlayMctsExecutor",
    record_callback: ShogiGameRecordCallback | None,
    progress_callback: GenerationProgressCallback | None,
) -> tuple[ShogiGameRecord, ...]:
    games = [
        _ActiveCheckpointSelfPlayGame(
            actor=actor,
            board_backend=config.board_backend,
            start_position=_start_position_for_game(config, index),
        )
        for index in range(config.games)
    ]
    remaining = set(range(config.games))
    started_at = perf_counter()
    for ply in range(config.max_plies):
        if not remaining:
            break
        black_indexes = [index for index in sorted(remaining) if board_is_black_turn(games[index].board)]
        white_indexes = [index for index in sorted(remaining) if not board_is_black_turn(games[index].board)]
        for indexes in (black_indexes, white_indexes):
            for offset in range(0, len(indexes), config.concurrent_games_per_process):
                batch_indexes = indexes[offset : offset + config.concurrent_games_per_process]
                positions = [UsiPosition(games[index].position_command()) for index in batch_indexes]
                results = selector.select_moves(positions)
                multi_position_search_performance = _performance_payload(selector.last_multi_position_search_performance)
                for game_index, result in zip(batch_indexes, results, strict=True):
                    telemetry = ShogiDecisionTelemetry(
                        move_performance=_performance_payload(result.performance),
                        multi_position_search_performance=multi_position_search_performance,
                        search_evidence=result.search_evidence,
                    )
                    if game_index in remaining and games[game_index].apply_move(result.move, telemetry):
                        remaining.remove(game_index)
                        if record_callback is not None:
                            record_callback(games[game_index].to_record())
        if config.progress_every_plies and (ply + 1) % config.progress_every_plies == 0:
            _emit_progress(
                _progress_payload(games, remaining=remaining, ply=ply + 1, elapsed_sec=perf_counter() - started_at),
                progress_callback=progress_callback,
            )
    records = tuple(game.to_record() for game in games)
    if record_callback is not None:
        completed_indexes = set(range(config.games)) - remaining
        for index, record in enumerate(records):
            if index not in completed_indexes:
                record_callback(record)
    return records


def summarize_checkpoint_self_play_records(
    records: tuple[ShogiGameRecord, ...],
    *,
    wall_time_sec: float | None = None,
) -> dict[str, object]:
    return records_summary(records, wall_time_sec=wall_time_sec)


@dataclass(frozen=True)
class _CheckpointSelfPlayMctsMoveResult:
    move: str
    policy_targets: dict[str, float] | None
    search_evidence: dict[str, object] | None
    performance: MctsMovePerformance


@dataclass(slots=True)
class _SelfPlayMctsNode:
    prior: float
    move: _SelfPlayMove = ""
    visit_count: int = 0
    value_sum: float = 0.0
    pending: bool = False
    child_moves: tuple[_SelfPlayMove, ...] = ()
    child_usis: tuple[str, ...] = ()
    child_priors: tuple[float, ...] = ()
    child_visit_counts: list[int] = field(default_factory=list)
    child_value_sums: list[float] = field(default_factory=list)
    child_nodes: list["_SelfPlayMctsNode | None"] = field(default_factory=list)

    @property
    def value_mean(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        return bool(self.child_moves)


@dataclass(frozen=True)
class _SelfPlaySelectedSimulation:
    nodes: list[_SelfPlayMctsNode]
    edge_indices: list[int]
    board: ShogiBoard
    node: _SelfPlayMctsNode


@dataclass(frozen=True)
class _SelfPlayLegalMoves:
    moves: tuple[_SelfPlayMove, ...]
    usis: tuple[str, ...]
    action_indices: tuple[int, ...] | None = None

    def __bool__(self) -> bool:
        return bool(self.moves)

    def evaluation_request(self, board: ShogiBoard):
        if self.action_indices is None:
            return (board, self.usis)
        return (board, self.usis, self.action_indices)


@dataclass(frozen=True)
class _SelfPlayLegalMoveBuild:
    legal_moves: _SelfPlayLegalMoves
    phase_times: dict[str, float]


@dataclass(frozen=True)
class _SelfPlayPendingSimulation:
    nodes: list[_SelfPlayMctsNode]
    edge_indices: list[int]
    board: ShogiBoard
    node: _SelfPlayMctsNode
    legal_moves: _SelfPlayLegalMoves


class _CheckpointSelfPlayMctsExecutor:
    """Self-play-only MCTS executor.

    This executor is intentionally separate from the general multi-position
    executor so checkpoint self-play can evolve toward a central evaluator
    service without changing ordinary match or CSA paths.
    """

    def __init__(
        self,
        evaluator: PolicyValueEvaluator,
        *,
        config: MctsConfig,
        move_selection: MoveSelectionConfig,
        use_action_indices: bool = False,
    ) -> None:
        self.evaluator = evaluator
        self.config = config
        self.move_selection = move_selection
        self.use_action_indices = use_action_indices
        self._rng = random.Random(self.move_selection.seed)
        self.last_multi_position_search_performance: MultiPositionMctsPerformance | None = None

    def select_moves(self, positions: Sequence[UsiPosition]) -> list[_CheckpointSelfPlayMctsMoveResult]:
        started_at = perf_counter()
        search_stats = _SelfPlayMctsSearchStats(
            position_count=len(positions),
            leaf_eval_batch_size_limit=self.config.nn_leaf_eval_batch_limit,
        )
        states = [
            _SelfPlayMctsSearchState.from_position(
                position,
                self.config.simulation_count,
                board_backend=self.config.board_backend,
                move_selection=self.move_selection,
                rng=self._rng,
                leaf_eval_batch_size_limit=self.config.nn_leaf_eval_batch_limit,
                use_action_indices=self.use_action_indices,
            )
            for position in positions
        ]
        for state in states:
            search_stats.add_phase_times(state.phase_wall_time_sec)
        active_states = [state for state in states if state.legal_moves]
        if active_states:
            self._expand_roots(active_states, search_stats)
        while any(state.remaining_simulations > 0 for state in active_states):
            pending, made_progress = self._collect_pending_leaf_evaluations(active_states, search_stats)
            if pending:
                self._evaluate_pending(pending, search_stats)
            elif not made_progress:
                break
        self.last_multi_position_search_performance = search_stats.to_performance(started_at)
        return [state.to_result() for state in states]

    def _collect_pending_leaf_evaluations(
        self,
        active_states: Sequence["_SelfPlayMctsSearchState"],
        search_stats: "_SelfPlayMctsSearchStats",
    ) -> tuple[list[tuple["_SelfPlayMctsSearchState", _SelfPlayPendingSimulation]], bool]:
        pending: list[tuple[_SelfPlayMctsSearchState, _SelfPlayPendingSimulation]] = []
        selected_count_by_state_id: dict[int, int] = {}
        made_progress = False
        while len(pending) < self.config.nn_leaf_eval_batch_limit:
            round_made_progress = False
            for state in active_states:
                state_id = id(state)
                if selected_count_by_state_id.get(state_id, 0) >= state.remaining_simulations:
                    continue
                simulation, progressed = self._select_leaf_for_evaluation(state, search_stats)
                made_progress = made_progress or progressed
                round_made_progress = round_made_progress or progressed
                if simulation is None:
                    continue
                selected_count_by_state_id[state_id] = selected_count_by_state_id.get(state_id, 0) + 1
                pending.append((state, simulation))
                if len(pending) >= self.config.nn_leaf_eval_batch_limit:
                    break
            if not round_made_progress:
                break
        return pending, made_progress

    def _select_leaf_for_evaluation(
        self,
        state: "_SelfPlayMctsSearchState",
        search_stats: "_SelfPlayMctsSearchStats",
    ) -> tuple[_SelfPlayPendingSimulation | None, bool]:
        board_copy_started_at = perf_counter()
        board = copy_board(state.board)
        self._record_phase_time(state, search_stats, "board_copy", perf_counter() - board_copy_started_at)

        selection_started_at = perf_counter()
        simulation = _select_pending_simulation(state.root, board, c_puct=self.config.c_puct)
        self._record_phase_time(state, search_stats, "selection", perf_counter() - selection_started_at)
        if simulation is None:
            if not _self_play_node_has_pending_descendant(state.root):
                state.remaining_simulations = 0
            return None, False
        if simulation.board.is_game_over():
            self._complete_simulation(
                state,
                search_stats,
                simulation.nodes,
                simulation.edge_indices,
                value=-1.0,
            )
            return None, True

        legal_moves_build = _self_play_legal_moves_for_board(
            simulation.board,
            use_action_indices=self.use_action_indices,
            include_usis=not self.use_action_indices,
        )
        legal_moves = legal_moves_build.legal_moves
        for phase_name, elapsed in legal_moves_build.phase_times.items():
            self._record_phase_time(state, search_stats, phase_name, elapsed)
        if not legal_moves:
            self._complete_simulation(
                state,
                search_stats,
                simulation.nodes,
                simulation.edge_indices,
                value=-1.0,
            )
            return None, True

        simulation.node.pending = True
        return _SelfPlayPendingSimulation(
            nodes=simulation.nodes,
            edge_indices=simulation.edge_indices,
            board=simulation.board,
            node=simulation.node,
            legal_moves=legal_moves,
        ), True

    def _complete_simulation(
        self,
        state: "_SelfPlayMctsSearchState",
        search_stats: "_SelfPlayMctsSearchStats",
        nodes: list[_SelfPlayMctsNode],
        edge_indices: list[int],
        *,
        value: float,
    ) -> None:
        backup_started_at = perf_counter()
        _backpropagate_path(nodes, edge_indices, value)
        self._record_phase_time(state, search_stats, "backup", perf_counter() - backup_started_at)
        state.completed_simulations += 1
        search_stats.completed_simulations += 1
        state.remaining_simulations -= 1

    @staticmethod
    def _record_phase_time(
        state: "_SelfPlayMctsSearchState",
        search_stats: "_SelfPlayMctsSearchStats",
        name: str,
        elapsed: float,
    ) -> None:
        state.add_phase_time(name, elapsed)
        search_stats.add_phase_time(name, elapsed)

    def _expand_roots(self, states: Sequence["_SelfPlayMctsSearchState"], search_stats: "_SelfPlayMctsSearchStats") -> None:
        started_at = perf_counter()
        evaluations = self.evaluator.evaluate_batch(
            tuple(state.legal_moves.evaluation_request(state.board) for state in states)
        )
        elapsed = perf_counter() - started_at
        search_stats.model_call_count += 1
        search_stats.model_wall_time_sec += elapsed
        if len(evaluations) != len(states):
            raise ValueError("batch evaluator must return one evaluation per request")
        for state, (priors, _value) in zip(states, evaluations, strict=True):
            expand_started_at = perf_counter()
            _expand_node_with_evaluation(state.root, state.legal_moves, priors)
            self._record_phase_time(state, search_stats, "expand", perf_counter() - expand_started_at)
            state.model_call_count += 1
            state.model_wall_time_sec += elapsed

    def _evaluate_pending(
        self,
        pending: Sequence[tuple["_SelfPlayMctsSearchState", _SelfPlayPendingSimulation]],
        search_stats: "_SelfPlayMctsSearchStats",
    ) -> None:
        batch_build_started_at = perf_counter()
        requests = tuple(simulation.legal_moves.evaluation_request(simulation.board) for _state, simulation in pending)
        batch_build_elapsed = perf_counter() - batch_build_started_at
        search_stats.add_phase_time("batch_build", batch_build_elapsed)
        for state in _unique_pending_states(pending):
            state.add_phase_time("batch_build", batch_build_elapsed)

        started_at = perf_counter()
        evaluations = self.evaluator.evaluate_batch(requests)
        elapsed = perf_counter() - started_at
        search_stats.model_call_count += 1
        search_stats.model_wall_time_sec += elapsed
        search_stats.add_leaf_eval_batch_size(len(pending))
        if len(evaluations) != len(pending):
            raise ValueError("batch evaluator must return one evaluation per request")
        updated_state_ids: set[int] = set()
        for (state, simulation), (priors, value) in zip(pending, evaluations, strict=True):
            state_id = id(state)
            if state_id not in updated_state_ids:
                state.leaf_eval_batch_sizes.append(len(pending))
                state.model_call_count += 1
                state.model_wall_time_sec += elapsed
                updated_state_ids.add(state_id)
            simulation.node.pending = False
            expand_started_at = perf_counter()
            _expand_node_with_evaluation(simulation.node, simulation.legal_moves, priors)
            self._record_phase_time(state, search_stats, "expand", perf_counter() - expand_started_at)
            self._complete_simulation(
                state,
                search_stats,
                simulation.nodes,
                simulation.edge_indices,
                value=max(-1.0, min(1.0, float(value))),
            )


@dataclass
class _SelfPlayMctsSearchState:
    board: ShogiBoard
    legal_moves: _SelfPlayLegalMoves
    root: _SelfPlayMctsNode
    started_at: float
    remaining_simulations: int
    ply: int
    leaf_eval_batch_size_limit: int
    completed_simulations: int = 0
    model_call_count: int = 0
    model_wall_time_sec: float = 0.0
    leaf_eval_batch_sizes: list[int] = field(default_factory=list)
    phase_wall_time_sec: dict[str, float] = field(default_factory=dict)
    move_selection: MoveSelectionConfig | None = None
    rng: random.Random = field(default_factory=random.Random)

    @classmethod
    def from_position(
        cls,
        position: UsiPosition,
        simulation_count: int,
        *,
        board_backend: str,
        move_selection: MoveSelectionConfig,
        rng: random.Random,
        leaf_eval_batch_size_limit: int,
        use_action_indices: bool,
    ) -> "_SelfPlayMctsSearchState":
        position_started_at = perf_counter()
        board = board_from_position(position, backend=board_backend)
        position_elapsed = perf_counter() - position_started_at
        legal_moves_build = _self_play_legal_moves_for_board(
            board,
            use_action_indices=use_action_indices,
            include_usis=True,
        )
        legal_moves = legal_moves_build.legal_moves
        state = cls(
            board=board,
            legal_moves=legal_moves,
            root=_SelfPlayMctsNode(prior=1.0),
            started_at=perf_counter(),
            remaining_simulations=simulation_count,
            ply=position_ply(position),
            leaf_eval_batch_size_limit=leaf_eval_batch_size_limit,
            move_selection=move_selection,
            rng=rng,
        )
        state.add_phase_time("position_parse", position_elapsed)
        for phase_name, elapsed in legal_moves_build.phase_times.items():
            state.add_phase_time(phase_name, elapsed)
        return state

    def add_phase_time(self, name: str, elapsed: float) -> None:
        self.phase_wall_time_sec[name] = self.phase_wall_time_sec.get(name, 0.0) + elapsed

    def to_result(self) -> _CheckpointSelfPlayMctsMoveResult:
        if not self.legal_moves:
            return _CheckpointSelfPlayMctsMoveResult(
                move=RESIGN_MOVE,
                policy_targets=None,
                search_evidence=None,
                performance=_mcts_move_performance_since(
                    self.started_at,
                    model_call_count=self.model_call_count,
                    model_wall_time_sec=self.model_wall_time_sec,
                    output_count=0,
                    leaf_eval_batch_sizes=self.leaf_eval_batch_sizes,
                    leaf_eval_batch_size_limit=self.leaf_eval_batch_size_limit,
                    phase_wall_time_sec=self.phase_wall_time_sec,
                ),
            )
        if self.move_selection is None:
            raise ValueError("move_selection is required")
        return _CheckpointSelfPlayMctsMoveResult(
            move=_select_self_play_final_move_at_ply(self.root, self.ply, self.move_selection, self.rng),
            policy_targets=_self_play_visit_count_policy_targets(self.root),
            search_evidence={
                "mcts_root_child_visit_counts": _self_play_root_child_visit_counts(self.root),
                "mcts_root_mean_value": self.root.value_mean,
            },
            performance=_mcts_move_performance_since(
                self.started_at,
                model_call_count=self.model_call_count,
                model_wall_time_sec=self.model_wall_time_sec,
                output_count=self.completed_simulations,
                leaf_eval_batch_sizes=self.leaf_eval_batch_sizes,
                leaf_eval_batch_size_limit=self.leaf_eval_batch_size_limit,
                phase_wall_time_sec=self.phase_wall_time_sec,
            ),
        )


@dataclass
class _SelfPlayMctsSearchStats:
    position_count: int
    leaf_eval_batch_size_limit: int
    completed_simulations: int = 0
    model_call_count: int = 0
    model_wall_time_sec: float = 0.0
    leaf_eval_batch_sizes: list[int] = field(default_factory=list)
    phase_wall_time_sec: dict[str, float] = field(default_factory=dict)

    def add_phase_time(self, name: str, elapsed: float) -> None:
        self.phase_wall_time_sec[name] = self.phase_wall_time_sec.get(name, 0.0) + elapsed

    def add_phase_times(self, phase_times: dict[str, float]) -> None:
        for name, elapsed in phase_times.items():
            self.add_phase_time(name, elapsed)

    def add_leaf_eval_batch_size(self, size: int) -> None:
        self.leaf_eval_batch_sizes.append(size)

    def to_performance(self, started_at: float) -> MultiPositionMctsPerformance:
        request_wall_time_sec = perf_counter() - started_at
        non_model_wall_time_sec = max(0.0, request_wall_time_sec - self.model_wall_time_sec)
        output_per_sec = self.completed_simulations / request_wall_time_sec if request_wall_time_sec > 0 else 0.0
        phase_times = dict(sorted(self.phase_wall_time_sec.items()))
        phase_times["unattributed"] = max(0.0, non_model_wall_time_sec - sum(phase_times.values()))
        return MultiPositionMctsPerformance(
            request_wall_time_sec=request_wall_time_sec,
            position_count=self.position_count,
            completed_simulations=self.completed_simulations,
            model_call_count=self.model_call_count,
            model_wall_time_sec=self.model_wall_time_sec,
            non_model_wall_time_sec=non_model_wall_time_sec,
            output_per_sec=output_per_sec,
            **leaf_eval_batch_metrics(
                self.leaf_eval_batch_sizes,
                batch_size_limit=self.leaf_eval_batch_size_limit,
            ),
            phase_wall_time_sec=phase_times,
        )


class _ActiveCheckpointSelfPlayGame:
    def __init__(
        self,
        *,
        actor: ShogiActorSpec,
        board_backend: str,
        start_position: StartPosition | None = None,
    ) -> None:
        self.start_position = start_position or startpos()
        self.board = board_from_position(self.start_position.usi_position, backend=board_backend)
        self.actor = actor
        self.initial_position_sfen = self.board.sfen()
        self.transitions: list[ShogiTransitionRecord] = []
        self.end_reason = "max_plies"
        self.winner: str | None = None

    @property
    def moves(self) -> tuple[str, ...]:
        return tuple(transition.action_usi for transition in self.transitions)

    def position_command(self) -> str:
        return position_command(self.moves, start_position=self.start_position)

    def apply_move(self, move: str, telemetry: ShogiDecisionTelemetry | None) -> bool:
        side = board_turn_name(self.board)
        legal_moves = legal_move_usis(self.board)
        position_sfen = self.board.sfen()
        if move == RESIGN_MOVE or move not in legal_moves:
            self.winner = "white" if board_is_black_turn(self.board) else "black"
            self.end_reason = "resign" if move == RESIGN_MOVE else "illegal_move"
            self._finalize_rewards()
            return True
        self.board.push_usi(move)
        done = self.board.is_game_over()
        self.winner = "black" if done and not board_is_black_turn(self.board) else "white" if done else None
        self.transitions.append(
            ShogiTransitionRecord(
                ply=len(self.transitions),
                side=side,
                position_sfen=position_sfen,
                legal_moves=legal_moves,
                action_usi=move,
                next_position_sfen=self.board.sfen(),
                reward=_transition_reward(side=side, winner=self.winner, done=done),
                done=done,
                decision_telemetry=telemetry,
            )
        )
        if done:
            self.end_reason = "game_over"
            return True
        return False

    def to_record(self) -> ShogiGameRecord:
        if self.end_reason == "max_plies":
            self._finalize_rewards()
        return ShogiGameRecord(
            black_actor=self.actor,
            white_actor=self.actor,
            initial_position_sfen=self.initial_position_sfen,
            transitions=tuple(self.transitions),
            end_reason=self.end_reason,
            winner=self.winner,
        )

    def _finalize_rewards(self) -> None:
        self.transitions = [
            ShogiTransitionRecord(
                ply=transition.ply,
                side=transition.side,
                position_sfen=transition.position_sfen,
                legal_moves=transition.legal_moves,
                action_usi=transition.action_usi,
                next_position_sfen=transition.next_position_sfen,
                reward=_transition_reward(side=transition.side, winner=self.winner, done=True),
                done=True,
                decision_usi_info_lines=transition.decision_usi_info_lines,
                decision_telemetry=transition.decision_telemetry,
            )
            for transition in self.transitions
        ]


def _checkpoint_self_play_selector(
    config: CheckpointSelfPlayConfig,
    move_selection: MoveSelectionConfig,
    *,
    evaluator: PolicyValueEvaluator,
    use_action_indices: bool = False,
) -> "_CheckpointSelfPlayMctsExecutor":
    return _CheckpointSelfPlayMctsExecutor(
        evaluator=evaluator,
        config=MctsConfig(
            simulation_count=config.mcts_simulations,
            nn_leaf_eval_batch_limit=config.nn_leaf_eval_batch_limit,
            board_backend=config.board_backend,
            root_reuse=False,
        ),
        move_selection=move_selection,
        use_action_indices=use_action_indices,
    )


def _checkpoint_self_play_actor(config: CheckpointSelfPlayConfig, move_selection: MoveSelectionConfig) -> ShogiActorSpec:
    return ShogiActorSpec(
        kind="checkpoint_self_play",
        name="checkpoint",
        settings={
            "checkpoint": config.checkpoint,
            "checkpoint_id": config.checkpoint_id,
            "checkpoint_path": config.checkpoint,
            "self_play_worker_processes": config.self_play_worker_processes,
            "move_selection_profile": "visit-sampling",
            "move_selection_temperature": move_selection.temperature,
            "move_selection_temperature_plies": move_selection.temperature_plies,
            "move_selector": "mcts",
            "mcts_simulations_per_move": config.mcts_simulations,
            "nn_leaf_eval_batch_limit": config.nn_leaf_eval_batch_limit,
            "central_evaluator_batch_size_limit": config.central_evaluator_batch_size_limit
            or config.nn_leaf_eval_batch_limit,
            "central_evaluator_flush_timeout_sec": config.central_evaluator_flush_timeout_sec,
            "move_time_limit_sec": None,
            "root_reuse": False,
            "device": config.device,
            "inference_precision": config.inference_precision,
            "compile_model": config.compile_model,
            "concurrent_games_per_process": config.concurrent_games_per_process,
            "board_backend": config.board_backend,
        },
    )


def _validate_config(config: CheckpointSelfPlayConfig) -> None:
    if config.games <= 0:
        raise ValueError("games must be positive")
    if config.concurrent_games_per_process <= 0:
        raise ValueError("concurrent_games_per_process must be positive")
    if config.max_plies <= 0:
        raise ValueError("max_plies must be positive")
    if config.mcts_simulations <= 0:
        raise ValueError("mcts_simulations must be positive")
    if config.nn_leaf_eval_batch_limit <= 0:
        raise ValueError("nn_leaf_eval_batch_limit must be positive")
    if config.self_play_worker_processes <= 0:
        raise ValueError("self_play_worker_processes must be positive")
    if config.central_evaluator_batch_size_limit is not None and config.central_evaluator_batch_size_limit <= 0:
        raise ValueError("central_evaluator_batch_size_limit must be positive")
    if config.central_evaluator_flush_timeout_sec < 0.0:
        raise ValueError("central_evaluator_flush_timeout_sec must be non-negative")
    if config.inference_precision not in {"fp32", "bf16"}:
        raise ValueError("inference_precision must be fp32 or bf16")
    if config.progress_every_plies < 0:
        raise ValueError("progress_every_plies must be non-negative")
    if config.start_positions and len(config.start_positions) != config.games:
        raise ValueError("start_positions must be empty or contain one start position per generated game")


def _worker_game_counts(games: int, worker_processes: int) -> list[int]:
    return [
        games // worker_processes + (1 if index < games % worker_processes else 0)
        for index in range(worker_processes)
    ]


def _worker_config(config: CheckpointSelfPlayConfig, *, game_count: int, start_index: int) -> CheckpointSelfPlayConfig:
    start_positions = ()
    if config.start_positions:
        start_positions = config.start_positions[start_index : start_index + game_count]
    return CheckpointSelfPlayConfig(
        checkpoint=config.checkpoint,
        checkpoint_id=config.checkpoint_id,
        games=game_count,
        concurrent_games_per_process=config.concurrent_games_per_process,
        max_plies=config.max_plies,
        mcts_simulations=config.mcts_simulations,
        nn_leaf_eval_batch_limit=config.nn_leaf_eval_batch_limit,
        device=config.device,
        board_backend=config.board_backend,
        move_selection=config.move_selection,
        inference_precision=config.inference_precision,
        compile_model=config.compile_model,
        self_play_worker_processes=1,
        central_evaluator_batch_size_limit=config.central_evaluator_batch_size_limit,
        central_evaluator_flush_timeout_sec=config.central_evaluator_flush_timeout_sec,
        progress_every_plies=config.progress_every_plies,
        start_positions=start_positions,
    )


def _worker_move_selection(move_selection: MoveSelectionConfig, worker_index: int) -> MoveSelectionConfig:
    if move_selection.seed is None:
        return move_selection
    return MoveSelectionConfig(
        mode=move_selection.mode,
        temperature=move_selection.temperature,
        temperature_plies=move_selection.temperature_plies,
        seed=move_selection.seed + worker_index,
    )


def _start_position_for_game(config: CheckpointSelfPlayConfig, game_index: int) -> StartPosition | None:
    if not config.start_positions:
        return None
    return config.start_positions[game_index]


def _performance_payload(performance: object | None) -> dict[str, object] | None:
    if performance is None:
        return None
    from dataclasses import asdict

    return asdict(performance)


def _progress_payload(
    games: list[_ActiveCheckpointSelfPlayGame],
    *,
    remaining: set[int],
    ply: int,
    elapsed_sec: float,
) -> dict[str, object]:
    completed = len(games) - len(remaining)
    active_plies = [len(games[index].transitions) for index in remaining]
    return {
        "ply": ply,
        "elapsed_sec": elapsed_sec,
        "completed_games": completed,
        "remaining_games": len(remaining),
        "active_plies_avg": mean(active_plies) if active_plies else 0.0,
        "active_plies_max": max(active_plies) if active_plies else 0,
    }


def _emit_progress(payload: dict[str, object], *, progress_callback: GenerationProgressCallback | None) -> None:
    if progress_callback is not None:
        progress_callback(payload)
        return
    print("progress " + json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)


def _can_use_self_play_action_indices(config: CheckpointSelfPlayConfig, evaluator: object) -> bool:
    return config.board_backend == "cshogi" and bool(getattr(evaluator, "accepts_action_indices", False))


def _self_play_legal_moves_for_board(
    board: ShogiBoard,
    *,
    use_action_indices: bool,
    include_usis: bool,
) -> _SelfPlayLegalMoveBuild:
    started_at = perf_counter()
    if use_action_indices and isinstance(board, cshogi.Board):
        enumerate_started_at = perf_counter()
        moves = tuple(board.legal_moves)
        enumerate_elapsed = perf_counter() - enumerate_started_at
        if not moves:
            total_elapsed = perf_counter() - started_at
            return _SelfPlayLegalMoveBuild(
                _SelfPlayLegalMoves((), (), ()),
                {
                    "legal_moves": total_elapsed,
                    "legal_moves_enumerate": enumerate_elapsed,
                },
            )
        usi_started_at = perf_counter()
        usis = tuple(cshogi.move_to_usi(move) for move in moves) if include_usis else ()
        usi_elapsed = perf_counter() - usi_started_at
        action_index_started_at = perf_counter()
        action_indices = tuple(_cached_cshogi_action_plane_policy_action_index(move, board.turn) for move in moves)
        action_index_elapsed = perf_counter() - action_index_started_at
        total_elapsed = perf_counter() - started_at
        return _SelfPlayLegalMoveBuild(
            _SelfPlayLegalMoves(moves=moves, usis=usis, action_indices=action_indices),
            {
                "legal_moves": total_elapsed,
                "legal_moves_enumerate": enumerate_elapsed,
                "legal_moves_usi": usi_elapsed,
                "legal_moves_action_index": action_index_elapsed,
            },
        )
    usi_started_at = perf_counter()
    usis = legal_move_usis(board)
    usi_elapsed = perf_counter() - usi_started_at
    total_elapsed = perf_counter() - started_at
    return _SelfPlayLegalMoveBuild(
        _SelfPlayLegalMoves(moves=usis, usis=usis, action_indices=None),
        {
            "legal_moves": total_elapsed,
            "legal_moves_usi": usi_elapsed,
        },
    )


def _push_self_play_move(board: ShogiBoard, move: _SelfPlayMove) -> None:
    if isinstance(board, cshogi.Board) and isinstance(move, int):
        board.push(move)
        return
    board.push_usi(str(move))


def _cshogi_action_plane_policy_action_index(move: int, *, turn: int) -> int:
    to_square = _cshogi_square_to_absolute_square(cshogi.move_to(move))
    relative_to_square = _side_to_move_relative_square(to_square, turn)
    move_type = _cshogi_action_plane_policy_move_type(move, to_square=to_square, turn=turn)
    return relative_to_square * _ACTION_PLANE_MOVE_TYPE_COUNT + move_type


@lru_cache(maxsize=65536)
def _cached_cshogi_action_plane_policy_action_index(move: int, turn: int) -> int:
    return _cshogi_action_plane_policy_action_index(move, turn=turn)


def _cshogi_action_plane_policy_move_type(move: int, *, to_square: int, turn: int) -> int:
    if cshogi.move_is_drop(move):
        return _ACTION_PLANE_DROP_MOVE_TYPE_OFFSET + cshogi.move_drop_hand_piece(move)
    from_square = _cshogi_square_to_absolute_square(cshogi.move_from(move))
    relative_from_square = _side_to_move_relative_square(from_square, turn)
    relative_to_square = _side_to_move_relative_square(to_square, turn)
    delta = relative_to_square - relative_from_square
    if delta not in _ACTION_PLANE_KNIGHT_DELTAS:
        delta = _action_plane_direction_delta(relative_from_square, relative_to_square)
    offset = _ACTION_PLANE_PROMOTE_MOVE_TYPE_OFFSET if cshogi.move_is_promotion(move) else 0
    return offset + _ACTION_PLANE_MOVE_TYPE_OFFSET_BY_DELTA[delta]


def _action_plane_direction_delta(relative_from_square: int, relative_to_square: int) -> int:
    from_rank, from_file = divmod(relative_from_square, 9)
    to_rank, to_file = divmod(relative_to_square, 9)
    rank_delta = 0 if to_rank == from_rank else 1 if to_rank > from_rank else -1
    file_delta = 0 if to_file == from_file else 1 if to_file > from_file else -1
    return rank_delta * 9 + file_delta


def _cshogi_square_to_absolute_square(square: int) -> int:
    return (square % 9) * 9 + (8 - square // 9)


def _side_to_move_relative_square(square: int, turn: int) -> int:
    if turn == shogi.BLACK:
        return square
    return 80 - square


def _select_pending_simulation(
    root: _SelfPlayMctsNode,
    board: ShogiBoard,
    *,
    c_puct: float,
) -> _SelfPlaySelectedSimulation | None:
    node = root
    nodes = [node]
    edge_indices: list[int] = []
    while node.child_moves:
        selected_index = _select_self_play_puct_child_index(node, c_puct=c_puct)
        if selected_index is None:
            return None
        child = node.child_nodes[selected_index]
        if child is None:
            child = _SelfPlayMctsNode(
                move=node.child_moves[selected_index],
                prior=node.child_priors[selected_index],
            )
            node.child_nodes[selected_index] = child
        edge_indices.append(selected_index)
        node = child
        _push_self_play_move(board, node.move)
        nodes.append(node)
    return _SelfPlaySelectedSimulation(
        nodes=nodes,
        edge_indices=edge_indices,
        board=board,
        node=node,
    )


def _expand_node_with_evaluation(
    node: _SelfPlayMctsNode,
    legal_moves: _SelfPlayLegalMoves,
    priors: MovePriors,
) -> None:
    prior_values = _aligned_self_play_priors(legal_moves, priors)
    node.child_moves = legal_moves.moves
    node.child_usis = legal_moves.usis
    node.child_priors = prior_values
    node.child_visit_counts = [0] * len(legal_moves.moves)
    node.child_value_sums = [0.0] * len(legal_moves.moves)
    node.child_nodes = [None] * len(legal_moves.moves)


def _aligned_self_play_priors(legal_moves: _SelfPlayLegalMoves, priors: MovePriors) -> tuple[float, ...]:
    if isinstance(priors, Mapping):
        if len(legal_moves.usis) != len(legal_moves.moves):
            raise ValueError("mapping priors require USI legal moves")
        prior_values = [max(0.0, float(priors.get(move, 0.0))) for move in legal_moves.usis]
        total = sum(prior_values)
        if total <= 0.0:
            uniform = 1.0 / len(legal_moves.moves)
            return tuple(uniform for _move in legal_moves.moves)
        inverse_total = 1.0 / total
        return tuple(prior * inverse_total for prior in prior_values)
    if len(priors) != len(legal_moves.moves):
        raise ValueError("aligned move priors must match legal move count")
    return tuple(float(prior) for prior in priors)


def _backpropagate_path(
    nodes: list[_SelfPlayMctsNode],
    edge_indices: list[int],
    value: float,
) -> None:
    for visited_node in reversed(nodes):
        visited_node.visit_count += 1
        visited_node.value_sum += value
        value = -value
    for edge_offset, index in enumerate(edge_indices):
        parent = nodes[edge_offset]
        node = nodes[edge_offset + 1]
        parent.child_visit_counts[index] = node.visit_count
        parent.child_value_sums[index] = node.value_sum


def _select_self_play_puct_child_index(
    node: _SelfPlayMctsNode,
    *,
    c_puct: float,
) -> int | None:
    parent_sqrt = max(1, node.visit_count) ** 0.5
    best_index: int | None = None
    best_score: float | None = None
    best_move_key = ""
    child_moves = node.child_moves
    child_nodes = node.child_nodes
    child_visit_counts = node.child_visit_counts
    child_value_sums = node.child_value_sums
    child_priors = node.child_priors
    exploration_scale = c_puct * parent_sqrt
    for index in range(len(child_moves)):
        child = child_nodes[index]
        if child is not None and child.pending:
            continue
        child_visit_count = child_visit_counts[index]
        child_value_mean = child_value_sums[index] / child_visit_count if child_visit_count else 0.0
        score = -child_value_mean + exploration_scale * child_priors[index] / (1 + child_visit_count)
        move_key = str(child_moves[index])
        if best_score is None or score > best_score or (score == best_score and move_key > best_move_key):
            best_index = index
            best_score = score
            best_move_key = move_key
    return best_index


def _self_play_node_has_pending_descendant(node: _SelfPlayMctsNode) -> bool:
    if node.pending:
        return True
    return any(child is not None and _self_play_node_has_pending_descendant(child) for child in node.child_nodes)


def _unique_pending_states(
    pending: Sequence[tuple["_SelfPlayMctsSearchState", _SelfPlayPendingSimulation]],
) -> tuple["_SelfPlayMctsSearchState", ...]:
    states: list[_SelfPlayMctsSearchState] = []
    seen_state_ids: set[int] = set()
    for state, _simulation in pending:
        state_id = id(state)
        if state_id in seen_state_ids:
            continue
        seen_state_ids.add(state_id)
        states.append(state)
    return tuple(states)


def _select_self_play_final_move_at_ply(
    root: _SelfPlayMctsNode,
    ply: int,
    config: MoveSelectionConfig,
    rng: random.Random,
) -> str:
    if config.mode == "visit_sample":
        if config.temperature is None or config.temperature_plies is None:
            raise ValueError("visit_sample selection requires temperature and temperature_plies")
        if ply < config.temperature_plies:
            return _sample_self_play_visit_count_move(root, temperature=config.temperature, rng=rng)
        return _deterministic_self_play_final_move(root)
    if config.mode == "max_visit":
        return _deterministic_self_play_final_move(root)
    raise ValueError(f"unsupported final move selection mode: {config.mode}")


def _deterministic_self_play_final_move(root: _SelfPlayMctsNode) -> str:
    selected_index = max(
        range(len(root.child_moves)),
        key=lambda index: (
            root.child_visit_counts[index],
            -_self_play_child_value_mean(root, index),
            _self_play_child_usi(root, index),
        ),
    )
    return _self_play_child_usi(root, selected_index)


def _self_play_child_usi(root: _SelfPlayMctsNode, index: int) -> str:
    if root.child_usis:
        return root.child_usis[index]
    move = root.child_moves[index]
    if isinstance(move, str):
        return move
    return cshogi.move_to_usi(move)


def _sample_self_play_visit_count_move(root: _SelfPlayMctsNode, *, temperature: float, rng: random.Random) -> str:
    weights = [max(0, visit_count) ** (1.0 / temperature) for visit_count in root.child_visit_counts]
    total = sum(weights)
    if total <= 0:
        return _self_play_child_usi(root, rng.randrange(len(root.child_moves)))
    threshold = rng.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if cumulative >= threshold:
            return _self_play_child_usi(root, index)
    return _self_play_child_usi(root, len(root.child_moves) - 1)


def _self_play_visit_count_policy_targets(root: _SelfPlayMctsNode) -> dict[str, float]:
    total = sum(root.child_visit_counts)
    if total <= 0:
        return _self_play_normalized_priors(root)
    return {
        _self_play_child_usi(root, index): visit_count / total
        for index, visit_count in enumerate(root.child_visit_counts)
    }


def _self_play_normalized_priors(root: _SelfPlayMctsNode) -> dict[str, float]:
    total = sum(max(0.0, prior) for prior in root.child_priors)
    if total <= 0.0:
        uniform = 1.0 / len(root.child_moves)
        return {_self_play_child_usi(root, index): uniform for index in range(len(root.child_moves))}
    inverse_total = 1.0 / total
    return {
        _self_play_child_usi(root, index): max(0.0, prior) * inverse_total
        for index, prior in enumerate(root.child_priors)
    }


def _self_play_root_child_visit_counts(root: _SelfPlayMctsNode) -> dict[str, int]:
    return {
        _self_play_child_usi(root, index): visit_count
        for index, visit_count in enumerate(root.child_visit_counts)
    }


def _self_play_child_value_mean(node: _SelfPlayMctsNode, index: int) -> float:
    visit_count = node.child_visit_counts[index]
    if visit_count == 0:
        return 0.0
    return node.child_value_sums[index] / visit_count


def _mcts_move_performance_since(
    started_at: float,
    *,
    model_call_count: int,
    model_wall_time_sec: float,
    output_count: int,
    leaf_eval_batch_sizes: Sequence[int],
    leaf_eval_batch_size_limit: int,
    phase_wall_time_sec: dict[str, float],
) -> MctsMovePerformance:
    request_wall_time_sec = perf_counter() - started_at
    non_model_wall_time_sec = max(0.0, request_wall_time_sec - model_wall_time_sec)
    phase_times = dict(sorted(phase_wall_time_sec.items()))
    phase_times["unattributed_wait"] = max(0.0, non_model_wall_time_sec - sum(phase_times.values()))
    output_per_sec = output_count / request_wall_time_sec if request_wall_time_sec > 0.0 else 0.0
    return MctsMovePerformance(
        request_wall_time_sec=request_wall_time_sec,
        model_call_count=model_call_count,
        model_wall_time_sec=model_wall_time_sec,
        non_model_wall_time_sec=non_model_wall_time_sec,
        output_count=output_count,
        output_per_sec=output_per_sec,
        **leaf_eval_batch_metrics(
            leaf_eval_batch_sizes,
            batch_size_limit=leaf_eval_batch_size_limit,
        ),
        phase_wall_time_sec=phase_times,
    )


def _transition_reward(*, side: str, winner: str | None, done: bool) -> float:
    if not done or winner is None:
        return 0.0
    return 1.0 if side == winner else -1.0
