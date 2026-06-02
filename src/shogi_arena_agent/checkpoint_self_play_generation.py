from __future__ import annotations

import json
import random
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from statistics import mean
from time import perf_counter

from shogi_arena_agent.board_backend import ShogiBoard, board_is_black_turn, board_turn_name, copy_board, legal_move_usis
from shogi_arena_agent.checkpoint_self_play_evaluator import CentralPolicyValueEvaluator
from shogi_arena_agent.mcts_config import (
    MctsConfig,
    MoveSelectionConfig,
    visit_sampling_move_selection_config,
)
from shogi_arena_agent.mcts_evaluator import PolicyValueEvaluator
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
    move_selection: MoveSelectionConfig | None = None
    self_play_worker_threads: int = 1
    central_evaluator_batch_size_limit: int | None = None
    central_evaluator_flush_timeout_sec: float = 0.002
    progress_every_plies: int = 0
    start_positions: tuple[StartPosition, ...] = ()


@dataclass(frozen=True)
class CheckpointSelfPlayGenerationResult:
    records: tuple[ShogiGameRecord, ...]
    central_evaluator_performance: dict[str, object]


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
    checkpoint_evaluator = checkpoint_evaluator_cls.from_checkpoint(config.checkpoint, device=config.device)
    central_batch_limit = config.central_evaluator_batch_size_limit or config.nn_leaf_eval_batch_limit
    with CentralPolicyValueEvaluator(
        checkpoint_evaluator.evaluate_positions,
        batch_size_limit=central_batch_limit,
        flush_timeout_sec=config.central_evaluator_flush_timeout_sec,
    ) as central_evaluator:
        records = _generate_checkpoint_self_play_games_with_central_evaluator(
            config,
            actor=actor,
            move_selection=move_selection,
            central_evaluator=central_evaluator,
            record_callback=record_callback,
            progress_callback=progress_callback,
        )
        central_evaluator_performance = central_evaluator.performance_summary()
    return CheckpointSelfPlayGenerationResult(
        records=records,
        central_evaluator_performance=central_evaluator_performance,
    )


def _generate_checkpoint_self_play_games_with_central_evaluator(
    config: CheckpointSelfPlayConfig,
    *,
    actor: ShogiActorSpec,
    move_selection: MoveSelectionConfig,
    central_evaluator: CentralPolicyValueEvaluator,
    record_callback: ShogiGameRecordCallback | None,
    progress_callback: GenerationProgressCallback | None,
) -> tuple[ShogiGameRecord, ...]:
    if config.self_play_worker_threads == 1:
        selector = _checkpoint_self_play_selector(
            config,
            move_selection,
            evaluator=central_evaluator.client(),
        )
        return _generate_checkpoint_self_play_games_with_selector(
            config,
            actor=actor,
            selector=selector,
            record_callback=record_callback,
            progress_callback=progress_callback,
        )

    records_by_index: dict[int, ShogiGameRecord] = {}
    records_lock = threading.Lock()
    callback_lock = threading.Lock()
    errors: list[BaseException] = []
    worker_counts = _worker_game_counts(config.games, config.self_play_worker_threads)
    start_index = 0
    threads: list[threading.Thread] = []
    for worker_index, game_count in enumerate(worker_counts):
        if game_count <= 0:
            continue
        worker_start_index = start_index
        start_index += game_count
        worker_config = _worker_config(config, game_count=game_count, start_index=worker_start_index)

        def run_worker(
            *,
            local_worker_index: int = worker_index,
            local_start_index: int = worker_start_index,
            local_config: CheckpointSelfPlayConfig = worker_config,
        ) -> None:
            try:
                selector = _checkpoint_self_play_selector(
                    local_config,
                    _worker_move_selection(move_selection, local_worker_index),
                    evaluator=central_evaluator.client(),
                )
                worker_records = _generate_checkpoint_self_play_games_with_selector(
                    local_config,
                    actor=actor,
                    selector=selector,
                    record_callback=_locked_record_callback(record_callback, callback_lock),
                    progress_callback=_locked_progress_callback(progress_callback, callback_lock),
                )
                with records_lock:
                    for local_index, record in enumerate(worker_records):
                        records_by_index[local_start_index + local_index] = record
            except BaseException as error:
                with records_lock:
                    errors.append(error)

        thread = threading.Thread(target=run_worker, name=f"checkpoint-self-play-worker-{worker_index}", daemon=True)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return tuple(records_by_index[index] for index in range(config.games))


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
class _SelfPlayMctsChild:
    move: str
    node: "_SelfPlayMctsNode"


@dataclass(slots=True)
class _SelfPlayMctsNode:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    pending: bool = False
    children: list[_SelfPlayMctsChild] = field(default_factory=list)

    @property
    def value_mean(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass(frozen=True)
class _SelfPlaySelectedSimulation:
    path: list[_SelfPlayMctsNode]
    board: ShogiBoard
    node: _SelfPlayMctsNode


@dataclass(frozen=True)
class _SelfPlayPendingSimulation:
    path: list[_SelfPlayMctsNode]
    board: ShogiBoard
    legal_moves: tuple[str, ...]


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
    ) -> None:
        self.evaluator = evaluator
        self.config = config
        self.move_selection = move_selection
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
        made_progress = False
        for state in active_states:
            if state.remaining_simulations <= 0:
                continue
            simulation, progressed = self._select_leaf_for_evaluation(state, search_stats)
            made_progress = made_progress or progressed
            if simulation is None:
                continue
            pending.append((state, simulation))
            if len(pending) >= self.config.nn_leaf_eval_batch_limit:
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
            state.remaining_simulations = 0
            return None, False
        if simulation.board.is_game_over():
            self._complete_simulation(state, search_stats, simulation.path, value=-1.0)
            return None, True

        legal_moves_started_at = perf_counter()
        legal_moves = legal_move_usis(simulation.board)
        self._record_phase_time(state, search_stats, "legal_moves", perf_counter() - legal_moves_started_at)
        if not legal_moves:
            self._complete_simulation(state, search_stats, simulation.path, value=-1.0)
            return None, True

        simulation.node.pending = True
        return _SelfPlayPendingSimulation(path=simulation.path, board=simulation.board, legal_moves=legal_moves), True

    def _complete_simulation(
        self,
        state: "_SelfPlayMctsSearchState",
        search_stats: "_SelfPlayMctsSearchStats",
        path: list[_SelfPlayMctsNode],
        *,
        value: float,
    ) -> None:
        backup_started_at = perf_counter()
        _backpropagate_path(path, value)
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
        evaluations = self.evaluator.evaluate_batch(tuple((state.board, state.legal_moves) for state in states))
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
        requests = tuple((simulation.board, simulation.legal_moves) for _state, simulation in pending)
        batch_build_elapsed = perf_counter() - batch_build_started_at
        search_stats.add_phase_time("batch_build", batch_build_elapsed)
        for state, _simulation in pending:
            state.add_phase_time("batch_build", batch_build_elapsed)

        started_at = perf_counter()
        evaluations = self.evaluator.evaluate_batch(requests)
        elapsed = perf_counter() - started_at
        search_stats.model_call_count += 1
        search_stats.model_wall_time_sec += elapsed
        search_stats.add_leaf_eval_batch_size(len(pending))
        if len(evaluations) != len(pending):
            raise ValueError("batch evaluator must return one evaluation per request")
        for (state, simulation), (priors, value) in zip(pending, evaluations, strict=True):
            state.leaf_eval_batch_sizes.append(len(pending))
            simulation.path[-1].pending = False
            expand_started_at = perf_counter()
            _expand_node_with_evaluation(simulation.path[-1], simulation.legal_moves, priors)
            self._record_phase_time(state, search_stats, "expand", perf_counter() - expand_started_at)
            self._complete_simulation(state, search_stats, simulation.path, value=max(-1.0, min(1.0, float(value))))
            state.model_call_count += 1
            state.model_wall_time_sec += elapsed


@dataclass
class _SelfPlayMctsSearchState:
    board: ShogiBoard
    legal_moves: tuple[str, ...]
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
    ) -> "_SelfPlayMctsSearchState":
        position_started_at = perf_counter()
        board = board_from_position(position, backend=board_backend)
        position_elapsed = perf_counter() - position_started_at
        legal_moves_started_at = perf_counter()
        legal_moves = legal_move_usis(board)
        legal_moves_elapsed = perf_counter() - legal_moves_started_at
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
        state.add_phase_time("legal_moves", legal_moves_elapsed)
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
    )


def _checkpoint_self_play_actor(config: CheckpointSelfPlayConfig, move_selection: MoveSelectionConfig) -> ShogiActorSpec:
    return ShogiActorSpec(
        kind="checkpoint_self_play",
        name="checkpoint",
        settings={
            "checkpoint": config.checkpoint,
            "checkpoint_id": config.checkpoint_id,
            "checkpoint_path": config.checkpoint,
            "self_play_worker_threads": config.self_play_worker_threads,
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
    if config.self_play_worker_threads <= 0:
        raise ValueError("self_play_worker_threads must be positive")
    if config.central_evaluator_batch_size_limit is not None and config.central_evaluator_batch_size_limit <= 0:
        raise ValueError("central_evaluator_batch_size_limit must be positive")
    if config.central_evaluator_flush_timeout_sec < 0.0:
        raise ValueError("central_evaluator_flush_timeout_sec must be non-negative")
    if config.progress_every_plies < 0:
        raise ValueError("progress_every_plies must be non-negative")
    if config.start_positions and len(config.start_positions) != config.games:
        raise ValueError("start_positions must be empty or contain one start position per generated game")


def _worker_game_counts(games: int, worker_threads: int) -> list[int]:
    return [games // worker_threads + (1 if index < games % worker_threads else 0) for index in range(worker_threads)]


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
        self_play_worker_threads=1,
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


def _locked_record_callback(
    callback: ShogiGameRecordCallback | None,
    lock: threading.Lock,
) -> ShogiGameRecordCallback | None:
    if callback is None:
        return None

    def locked(record: ShogiGameRecord) -> None:
        with lock:
            callback(record)

    return locked


def _locked_progress_callback(
    callback: GenerationProgressCallback | None,
    lock: threading.Lock,
) -> GenerationProgressCallback | None:
    if callback is None:
        return None

    def locked(payload: dict[str, object]) -> None:
        with lock:
            callback(payload)

    return locked


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


def _select_pending_simulation(
    root: _SelfPlayMctsNode,
    board: ShogiBoard,
    *,
    c_puct: float,
) -> _SelfPlaySelectedSimulation | None:
    node = root
    path = [node]
    while node.children:
        selected = _select_self_play_puct_child(node, c_puct=c_puct)
        if selected is None:
            return None
        child = selected
        move = child.move
        node = child.node
        board.push_usi(move)
        path.append(node)
    return _SelfPlaySelectedSimulation(path=path, board=board, node=node)


def _expand_node_with_evaluation(
    node: _SelfPlayMctsNode,
    legal_moves: tuple[str, ...],
    priors: dict[str, float],
) -> None:
    prior_values = [max(0.0, float(priors.get(move, 0.0))) for move in legal_moves]
    total = sum(prior_values)
    if total <= 0.0:
        uniform = 1.0 / len(legal_moves)
        node.children = [_SelfPlayMctsChild(move=move, node=_SelfPlayMctsNode(prior=uniform)) for move in legal_moves]
        return
    inverse_total = 1.0 / total
    node.children = [
        _SelfPlayMctsChild(move=move, node=_SelfPlayMctsNode(prior=prior * inverse_total))
        for move, prior in zip(legal_moves, prior_values, strict=True)
    ]


def _backpropagate_path(path: list[_SelfPlayMctsNode], value: float) -> None:
    for visited_node in reversed(path):
        visited_node.visit_count += 1
        visited_node.value_sum += value
        value = -value


def _select_self_play_puct_child(
    node: _SelfPlayMctsNode,
    *,
    c_puct: float,
) -> _SelfPlayMctsChild | None:
    parent_sqrt = max(1, node.visit_count) ** 0.5
    best: _SelfPlayMctsChild | None = None
    best_score: tuple[float, str] | None = None
    for child in node.children:
        child_node = child.node
        if child_node.pending:
            continue
        child_visit_count = child_node.visit_count
        child_value_mean = child_node.value_sum / child_visit_count if child_visit_count else 0.0
        exploration = c_puct * child_node.prior * parent_sqrt / (1 + child_visit_count)
        score = (-child_value_mean + exploration, child.move)
        if best_score is None or score > best_score:
            best = child
            best_score = score
    return best


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
    return max(root.children, key=lambda child: (child.node.visit_count, -child.node.value_mean, child.move)).move


def _sample_self_play_visit_count_move(root: _SelfPlayMctsNode, *, temperature: float, rng: random.Random) -> str:
    weights = [max(0, child.node.visit_count) ** (1.0 / temperature) for child in root.children]
    total = sum(weights)
    if total <= 0:
        return rng.choice(root.children).move
    threshold = rng.random() * total
    cumulative = 0.0
    for child, weight in zip(root.children, weights, strict=True):
        cumulative += weight
        if cumulative >= threshold:
            return child.move
    return root.children[-1].move


def _self_play_visit_count_policy_targets(root: _SelfPlayMctsNode) -> dict[str, float]:
    total = sum(child.node.visit_count for child in root.children)
    if total <= 0:
        return _self_play_normalized_priors(root)
    return {child.move: child.node.visit_count / total for child in root.children}


def _self_play_normalized_priors(root: _SelfPlayMctsNode) -> dict[str, float]:
    total = sum(max(0.0, child.node.prior) for child in root.children)
    if total <= 0.0:
        uniform = 1.0 / len(root.children)
        return {child.move: uniform for child in root.children}
    inverse_total = 1.0 / total
    return {child.move: max(0.0, child.node.prior) * inverse_total for child in root.children}


def _self_play_root_child_visit_counts(root: _SelfPlayMctsNode) -> dict[str, int]:
    return {child.move: child.node.visit_count for child in root.children}


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
