from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from time import perf_counter

from shogi_arena_agent.board_backend import ShogiBoard, copy_board, legal_move_usis
from shogi_arena_agent.mcts_move_selector import (
    MctsConfig,
    MctsMovePerformance,
    MoveSelectionConfig,
    PolicyValueEvaluator,
    UniformPolicyValueEvaluator,
    evaluation_move_selection_config,
)
from shogi_arena_agent.mcts_tree import (
    MctsNode,
    PendingSimulation,
    SelectedSimulation,
    normalize_priors,
    position_ply,
    select_final_move_at_ply,
    visit_count_policy_targets,
)
from shogi_arena_agent.usi import RESIGN_MOVE, UsiPosition, board_from_position


@dataclass(frozen=True)
class MctsMoveResult:
    move: str
    policy_targets: dict[str, float] | None
    performance: MctsMovePerformance


@dataclass(frozen=True)
class MctsBatchPerformance:
    request_wall_time_sec: float
    position_count: int
    completed_simulations: int
    model_call_count: int
    model_wall_time_sec: float
    non_model_wall_time_sec: float
    output_per_sec: float
    actual_nn_leaf_eval_batch_size_avg: float = 0.0
    actual_nn_leaf_eval_batch_size_max: int = 0
    actual_nn_leaf_eval_batch_count: int = 0
    phase_wall_time_sec: dict[str, float] = field(default_factory=dict)


class MctsBatchSearchExecutor:
    """Run temporary MCTS searches for multiple positions in one NN evaluation flow.

    This executor does not own per-game search sessions or persistent roots.
    """

    def __init__(
        self,
        evaluator: PolicyValueEvaluator | None = None,
        *,
        config: MctsConfig | None = None,
        move_selection: MoveSelectionConfig | None = None,
    ) -> None:
        self.evaluator = evaluator or UniformPolicyValueEvaluator()
        self.config = config or MctsConfig()
        self.move_selection = move_selection or evaluation_move_selection_config()
        self._rng = random.Random(self.move_selection.seed)
        self.last_batch_performance: MctsBatchPerformance | None = None

    def select_moves(self, positions: Sequence[UsiPosition]) -> list[MctsMoveResult]:
        started_at = perf_counter()
        batch_stats = _BatchSearchStats(position_count=len(positions))
        states = [
            _BatchedSearchState.from_position(
                position,
                self.config.simulation_count,
                board_backend=self.config.board_backend,
                move_selection=self.move_selection,
                rng=self._rng,
            )
            for position in positions
        ]
        for state in states:
            batch_stats.add_phase_times(state.phase_wall_time_sec)
        active_states = [state for state in states if state.legal_moves]
        if active_states:
            self._expand_roots(active_states, batch_stats)
        while any(state.remaining_simulations > 0 for state in active_states):
            pending: list[tuple[_BatchedSearchState, PendingSimulation]] = []
            made_progress = False
            for state in active_states:
                if state.remaining_simulations <= 0:
                    continue
                board_copy_started_at = perf_counter()
                board = copy_board(state.board)
                board_copy_elapsed = perf_counter() - board_copy_started_at
                state.add_phase_time("board_copy", board_copy_elapsed)
                batch_stats.add_phase_time("board_copy", board_copy_elapsed)

                selection_started_at = perf_counter()
                simulation = _select_pending_simulation(state.root, board, c_puct=self.config.c_puct)
                selection_elapsed = perf_counter() - selection_started_at
                state.add_phase_time("selection", selection_elapsed)
                batch_stats.add_phase_time("selection", selection_elapsed)
                if simulation is None:
                    state.remaining_simulations = 0
                    continue
                if simulation.board.is_game_over():
                    backup_started_at = perf_counter()
                    _backpropagate_path(simulation.path, -1.0)
                    backup_elapsed = perf_counter() - backup_started_at
                    state.add_phase_time("backup", backup_elapsed)
                    batch_stats.add_phase_time("backup", backup_elapsed)
                    state.completed_simulations += 1
                    batch_stats.completed_simulations += 1
                    state.remaining_simulations -= 1
                    made_progress = True
                    continue
                legal_moves_started_at = perf_counter()
                legal_moves = legal_move_usis(simulation.board)
                legal_moves_elapsed = perf_counter() - legal_moves_started_at
                state.add_phase_time("legal_moves", legal_moves_elapsed)
                batch_stats.add_phase_time("legal_moves", legal_moves_elapsed)
                if not legal_moves:
                    backup_started_at = perf_counter()
                    _backpropagate_path(simulation.path, -1.0)
                    backup_elapsed = perf_counter() - backup_started_at
                    state.add_phase_time("backup", backup_elapsed)
                    batch_stats.add_phase_time("backup", backup_elapsed)
                    state.completed_simulations += 1
                    batch_stats.completed_simulations += 1
                    state.remaining_simulations -= 1
                    made_progress = True
                    continue
                simulation.node.pending = True
                pending.append((state, PendingSimulation(path=simulation.path, board=simulation.board, legal_moves=legal_moves)))
                made_progress = True
                if len(pending) >= self.config.evaluation_batch_size:
                    break
            if pending:
                self._evaluate_pending(pending, batch_stats)
            elif not made_progress:
                break
        self.last_batch_performance = batch_stats.to_performance(started_at)
        return [state.to_result() for state in states]

    def _expand_roots(self, states: Sequence["_BatchedSearchState"], batch_stats: "_BatchSearchStats") -> None:
        started_at = perf_counter()
        evaluations = self.evaluator.evaluate_batch(tuple((state.board, state.legal_moves) for state in states))
        elapsed = perf_counter() - started_at
        batch_stats.model_call_count += 1
        batch_stats.model_wall_time_sec += elapsed
        if len(evaluations) != len(states):
            raise ValueError("batch evaluator must return one evaluation per request")
        for state, (priors, _value) in zip(states, evaluations, strict=True):
            expand_started_at = perf_counter()
            _expand_node_with_evaluation(state.root, state.legal_moves, priors)
            expand_elapsed = perf_counter() - expand_started_at
            state.add_phase_time("expand", expand_elapsed)
            batch_stats.add_phase_time("expand", expand_elapsed)
            state.model_call_count += 1
            state.model_wall_time_sec += elapsed

    def _evaluate_pending(
        self,
        pending: Sequence[tuple["_BatchedSearchState", PendingSimulation]],
        batch_stats: "_BatchSearchStats",
    ) -> None:
        batch_build_started_at = perf_counter()
        requests = tuple((simulation.board, simulation.legal_moves) for _state, simulation in pending)
        batch_build_elapsed = perf_counter() - batch_build_started_at
        batch_stats.add_phase_time("batch_build", batch_build_elapsed)
        for state, _simulation in pending:
            state.add_phase_time("batch_build", batch_build_elapsed)

        started_at = perf_counter()
        evaluations = self.evaluator.evaluate_batch(requests)
        elapsed = perf_counter() - started_at
        batch_stats.model_call_count += 1
        batch_stats.model_wall_time_sec += elapsed
        if len(evaluations) != len(pending):
            raise ValueError("batch evaluator must return one evaluation per request")
        for (state, simulation), (priors, value) in zip(pending, evaluations, strict=True):
            simulation.path[-1].pending = False
            expand_started_at = perf_counter()
            _expand_node_with_evaluation(simulation.path[-1], simulation.legal_moves, priors)
            expand_elapsed = perf_counter() - expand_started_at
            state.add_phase_time("expand", expand_elapsed)
            batch_stats.add_phase_time("expand", expand_elapsed)
            backup_started_at = perf_counter()
            _backpropagate_path(simulation.path, max(-1.0, min(1.0, float(value))))
            backup_elapsed = perf_counter() - backup_started_at
            state.add_phase_time("backup", backup_elapsed)
            batch_stats.add_phase_time("backup", backup_elapsed)
            state.completed_simulations += 1
            batch_stats.completed_simulations += 1
            state.remaining_simulations -= 1
            state.model_call_count += 1
            state.model_wall_time_sec += elapsed


@dataclass
class _BatchedSearchState:
    board: ShogiBoard
    legal_moves: tuple[str, ...]
    root: MctsNode
    started_at: float
    remaining_simulations: int
    ply: int
    completed_simulations: int = 0
    model_call_count: int = 0
    model_wall_time_sec: float = 0.0
    phase_wall_time_sec: dict[str, float] = field(default_factory=dict)
    move_selection: MoveSelectionConfig = field(default_factory=evaluation_move_selection_config)
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
    ) -> "_BatchedSearchState":
        position_started_at = perf_counter()
        board = board_from_position(position, backend=board_backend)
        position_elapsed = perf_counter() - position_started_at
        legal_moves_started_at = perf_counter()
        legal_moves = legal_move_usis(board)
        legal_moves_elapsed = perf_counter() - legal_moves_started_at
        state = cls(
            board=board,
            legal_moves=legal_moves,
            root=MctsNode(prior=1.0),
            started_at=perf_counter(),
            remaining_simulations=simulation_count,
            ply=position_ply(position),
            move_selection=move_selection,
            rng=rng,
        )
        state.add_phase_time("position_parse", position_elapsed)
        state.add_phase_time("legal_moves", legal_moves_elapsed)
        return state

    def add_phase_time(self, name: str, elapsed: float) -> None:
        self.phase_wall_time_sec[name] = self.phase_wall_time_sec.get(name, 0.0) + elapsed

    def to_result(self) -> MctsMoveResult:
        if not self.legal_moves:
            return MctsMoveResult(
                move=RESIGN_MOVE,
                policy_targets=None,
                performance=_batched_performance_since(
                    self.started_at,
                    model_call_count=self.model_call_count,
                    model_wall_time_sec=self.model_wall_time_sec,
                    output_count=0,
                    phase_wall_time_sec=self.phase_wall_time_sec,
                ),
            )
        return MctsMoveResult(
            move=select_final_move_at_ply(self.root, self.ply, self.move_selection, self.rng),
            policy_targets=visit_count_policy_targets(self.root),
            performance=_batched_performance_since(
                self.started_at,
                model_call_count=self.model_call_count,
                model_wall_time_sec=self.model_wall_time_sec,
                output_count=self.completed_simulations,
                phase_wall_time_sec=self.phase_wall_time_sec,
            ),
        )


@dataclass
class _BatchSearchStats:
    position_count: int
    completed_simulations: int = 0
    model_call_count: int = 0
    model_wall_time_sec: float = 0.0
    phase_wall_time_sec: dict[str, float] = field(default_factory=dict)

    def add_phase_time(self, name: str, elapsed: float) -> None:
        self.phase_wall_time_sec[name] = self.phase_wall_time_sec.get(name, 0.0) + elapsed

    def add_phase_times(self, phase_times: dict[str, float]) -> None:
        for name, elapsed in phase_times.items():
            self.add_phase_time(name, elapsed)

    def to_performance(self, started_at: float) -> MctsBatchPerformance:
        request_wall_time_sec = perf_counter() - started_at
        non_model_wall_time_sec = max(0.0, request_wall_time_sec - self.model_wall_time_sec)
        output_per_sec = self.completed_simulations / request_wall_time_sec if request_wall_time_sec > 0 else 0.0
        phase_times = dict(sorted(self.phase_wall_time_sec.items()))
        phase_times["unattributed"] = max(0.0, non_model_wall_time_sec - sum(phase_times.values()))
        return MctsBatchPerformance(
            request_wall_time_sec=request_wall_time_sec,
            position_count=self.position_count,
            completed_simulations=self.completed_simulations,
            model_call_count=self.model_call_count,
            model_wall_time_sec=self.model_wall_time_sec,
            non_model_wall_time_sec=non_model_wall_time_sec,
            output_per_sec=output_per_sec,
            phase_wall_time_sec=phase_times,
        )


def _select_pending_simulation(root: MctsNode, board: ShogiBoard, *, c_puct: float) -> SelectedSimulation | None:
    node = root
    path = [node]
    while node.children:
        selected = _select_child_node(node, c_puct=c_puct)
        if selected is None:
            return None
        move, node = selected
        board.push_usi(move)
        path.append(node)
    return SelectedSimulation(path=path, board=board, node=node)


def _select_child_node(node: MctsNode, *, c_puct: float) -> tuple[str, MctsNode] | None:
    parent_visits = max(1, node.visit_count)

    def score(item: tuple[str, MctsNode]) -> tuple[float, str]:
        move, child = item
        exploration = c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
        return -child.value_mean + exploration, move

    candidates = [item for item in node.children.items() if not item[1].pending]
    if not candidates:
        return None
    return max(candidates, key=score)


def _expand_node_with_evaluation(node: MctsNode, legal_moves: tuple[str, ...], priors: dict[str, float]) -> None:
    normalized_priors = normalize_priors(legal_moves, priors)
    node.children = {move: MctsNode(prior=normalized_priors[move]) for move in legal_moves}


def _backpropagate_path(path: list[MctsNode], value: float) -> None:
    for visited_node in reversed(path):
        visited_node.visit_count += 1
        visited_node.value_sum += value
        value = -value


def _batched_performance_since(
    started_at: float,
    *,
    model_call_count: int,
    model_wall_time_sec: float,
    output_count: int,
    phase_wall_time_sec: dict[str, float],
) -> MctsMovePerformance:
    request_wall_time_sec = perf_counter() - started_at
    non_model_wall_time_sec = max(0.0, request_wall_time_sec - model_wall_time_sec)
    phase_times = dict(sorted(phase_wall_time_sec.items()))
    phase_times["unattributed_wait"] = max(0.0, non_model_wall_time_sec - sum(phase_times.values()))
    output_per_sec = output_count / request_wall_time_sec if request_wall_time_sec > 0 else 0.0
    return MctsMovePerformance(
        request_wall_time_sec=request_wall_time_sec,
        model_call_count=model_call_count,
        model_wall_time_sec=model_wall_time_sec,
        non_model_wall_time_sec=non_model_wall_time_sec,
        output_count=output_count,
        output_per_sec=output_per_sec,
        phase_wall_time_sec=phase_times,
    )
