from __future__ import annotations

import math
import random
from collections.abc import Sequence
from time import perf_counter
from dataclasses import dataclass, field
from typing import Protocol

from shogi_arena_agent.board_backend import ShogiBoard, copy_board, legal_move_usis, validate_board_backend
from shogi_arena_agent.usi import RESIGN_MOVE, UsiPosition, board_from_position


class PolicyValueEvaluator(Protocol):
    def evaluate_batch(
        self,
        requests: Sequence[tuple[ShogiBoard, tuple[str, ...]]],
    ) -> list[tuple[dict[str, float], float]]:
        """Return move priors and values from the side-to-move perspective."""


@dataclass(frozen=True)
class MctsConfig:
    simulation_count: int = 32
    c_puct: float = 1.5
    evaluation_batch_size: int = 1
    move_time_limit_sec: float | None = None
    board_backend: str = "python-shogi"

    def __post_init__(self) -> None:
        if self.simulation_count <= 0:
            raise ValueError("simulation_count must be positive")
        if self.c_puct <= 0.0:
            raise ValueError("c_puct must be positive")
        if self.evaluation_batch_size <= 0:
            raise ValueError("evaluation_batch_size must be positive")
        if self.move_time_limit_sec is not None and self.move_time_limit_sec < 0.0:
            raise ValueError("move_time_limit_sec must be non-negative")
        validate_board_backend(self.board_backend)


@dataclass(frozen=True)
class MoveSelectionConfig:
    mode: str = "deterministic"
    temperature: float = 1.0
    temperature_plies: int = 0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"deterministic", "visit_sample"}:
            raise ValueError("mode must be deterministic or visit_sample")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.temperature_plies < 0:
            raise ValueError("temperature_plies must be non-negative")


def evaluation_move_selection_config() -> MoveSelectionConfig:
    return MoveSelectionConfig(mode="deterministic")


def self_play_move_selection_config(*, seed: int | None = None) -> MoveSelectionConfig:
    return MoveSelectionConfig(mode="visit_sample", temperature=1.0, temperature_plies=40, seed=seed)


@dataclass(frozen=True)
class MctsMovePerformance:
    request_wall_time_sec: float
    model_call_count: int
    model_wall_time_sec: float
    non_model_wall_time_sec: float
    output_count: int
    output_per_sec: float
    actual_nn_leaf_eval_batch_size_avg: float = 0.0
    actual_nn_leaf_eval_batch_size_max: int = 0
    actual_nn_leaf_eval_batch_count: int = 0
    phase_wall_time_sec: dict[str, float] = field(default_factory=dict)


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


class UniformPolicyValueEvaluator:
    def evaluate_batch(
        self,
        requests: Sequence[tuple[ShogiBoard, tuple[str, ...]]],
    ) -> list[tuple[dict[str, float], float]]:
        evaluations: list[tuple[dict[str, float], float]] = []
        for _board, legal_moves in requests:
            if not legal_moves:
                evaluations.append(({}, -1.0))
                continue
            prior = 1.0 / len(legal_moves)
            evaluations.append(({move: prior for move in legal_moves}, 0.0))
        return evaluations


class MctsPolicy:
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
        self.last_policy_targets: dict[str, float] | None = None
        self.last_performance: MctsMovePerformance | None = None
        self._model_call_count = 0
        self._model_wall_time_sec = 0.0
        self._leaf_eval_batch_sizes: list[int] = []

    def select_move(self, position: UsiPosition) -> str:
        started_at = perf_counter()
        self._model_call_count = 0
        self._model_wall_time_sec = 0.0
        self._leaf_eval_batch_sizes = []
        board = board_from_position(position, backend=self.config.board_backend)
        legal_moves = legal_move_usis(board)
        if not legal_moves:
            self.last_policy_targets = None
            self.last_performance = self._performance_since(started_at, output_count=0)
            return RESIGN_MOVE

        root = _Node(prior=1.0)
        self._expand(root, board)
        completed_simulations = 0
        deadline = None
        if self.config.move_time_limit_sec is not None:
            deadline = started_at + self.config.move_time_limit_sec
        while completed_simulations < self.config.simulation_count:
            if deadline is not None and perf_counter() >= deadline:
                break
            completed_simulations += self._run_simulation_batch(
                root,
                board,
                max_count=min(self.config.evaluation_batch_size, self.config.simulation_count - completed_simulations),
            )

        self.last_policy_targets = _visit_count_policy_targets(root)
        self.last_performance = self._performance_since(started_at, output_count=completed_simulations)
        return _select_final_move(root, position, self.move_selection, self._rng)

    def _run_simulation_batch(self, root: _Node, board: ShogiBoard, *, max_count: int) -> int:
        pending: list[_PendingSimulation] = []
        completed = 0
        for _ in range(max_count):
            simulation = self._select_simulation(root, copy_board(board))
            if simulation is None:
                break
            if simulation.board.is_game_over():
                self._backpropagate(simulation.path, -1.0)
                completed += 1
                continue
            legal_moves = legal_move_usis(simulation.board)
            if not legal_moves:
                self._backpropagate(simulation.path, -1.0)
                completed += 1
                continue
            simulation.node.pending = True
            pending.append(_PendingSimulation(path=simulation.path, board=simulation.board, legal_moves=legal_moves))

        if pending:
            self._leaf_eval_batch_sizes.append(len(pending))
            started_at = perf_counter()
            evaluations = self.evaluator.evaluate_batch(
                tuple((simulation.board, simulation.legal_moves) for simulation in pending)
            )
            self._model_call_count += 1
            self._model_wall_time_sec += perf_counter() - started_at
            if len(evaluations) != len(pending):
                raise ValueError("batch evaluator must return one evaluation per request")
            for simulation, (priors, value) in zip(pending, evaluations, strict=True):
                simulation.path[-1].pending = False
                self._expand_with_evaluation(simulation.path[-1], simulation.legal_moves, priors)
                self._backpropagate(simulation.path, max(-1.0, min(1.0, float(value))))
            completed += len(pending)
        return completed

    def _select_simulation(self, root: _Node, board: ShogiBoard) -> _SelectedSimulation | None:
        node = root
        path = [node]
        while node.children:
            selected = self._select_child(node)
            if selected is None:
                return None
            move, node = selected
            board.push_usi(move)
            path.append(node)
        return _SelectedSimulation(path=path, board=board, node=node)

    def _backpropagate(self, path: list[_Node], value: float) -> None:
        for visited_node in reversed(path):
            visited_node.visit_count += 1
            visited_node.value_sum += value
            value = -value

    def _expand(self, node: _Node, board: ShogiBoard) -> float:
        legal_moves = legal_move_usis(board)
        if not legal_moves:
            return -1.0

        started_at = perf_counter()
        priors, value = self.evaluator.evaluate_batch(((board, legal_moves),))[0]
        self._model_call_count += 1
        self._model_wall_time_sec += perf_counter() - started_at
        self._expand_with_evaluation(node, legal_moves, priors)
        return max(-1.0, min(1.0, float(value)))

    def _expand_with_evaluation(self, node: _Node, legal_moves: tuple[str, ...], priors: dict[str, float]) -> None:
        normalized_priors = _normalize_priors(legal_moves, priors)
        node.children = {move: _Node(prior=normalized_priors[move]) for move in legal_moves}

    def _select_child(self, node: _Node) -> tuple[str, _Node] | None:
        parent_visits = max(1, node.visit_count)

        def score(item: tuple[str, _Node]) -> tuple[float, str]:
            move, child = item
            exploration = self.config.c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
            return -child.value_mean + exploration, move

        candidates = [item for item in node.children.items() if not item[1].pending]
        if not candidates:
            return None
        return max(candidates, key=score)

    def _performance_since(self, started_at: float, *, output_count: int) -> MctsMovePerformance:
        request_wall_time_sec = perf_counter() - started_at
        non_model_wall_time_sec = max(0.0, request_wall_time_sec - self._model_wall_time_sec)
        output_per_sec = output_count / request_wall_time_sec if request_wall_time_sec > 0 else 0.0
        leaf_batch_size_avg = (
            sum(self._leaf_eval_batch_sizes) / len(self._leaf_eval_batch_sizes) if self._leaf_eval_batch_sizes else 0.0
        )
        return MctsMovePerformance(
            request_wall_time_sec=request_wall_time_sec,
            model_call_count=self._model_call_count,
            model_wall_time_sec=self._model_wall_time_sec,
            non_model_wall_time_sec=non_model_wall_time_sec,
            output_count=output_count,
            output_per_sec=output_per_sec,
            actual_nn_leaf_eval_batch_size_avg=leaf_batch_size_avg,
            actual_nn_leaf_eval_batch_size_max=max(self._leaf_eval_batch_sizes, default=0),
            actual_nn_leaf_eval_batch_count=len(self._leaf_eval_batch_sizes),
        )


class BatchedMctsMoveSelector:
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
            pending: list[tuple[_BatchedSearchState, _PendingSimulation]] = []
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
                pending.append((state, _PendingSimulation(path=simulation.path, board=simulation.board, legal_moves=legal_moves)))
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
        pending: Sequence[tuple["_BatchedSearchState", _PendingSimulation]],
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
class _Node:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    pending: bool = False
    children: dict[str, "_Node"] = field(default_factory=dict)

    @property
    def value_mean(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass
class _SelectedSimulation:
    path: list[_Node]
    board: ShogiBoard
    node: _Node


@dataclass
class _PendingSimulation:
    path: list[_Node]
    board: ShogiBoard
    legal_moves: tuple[str, ...]


@dataclass
class _BatchedSearchState:
    board: ShogiBoard
    legal_moves: tuple[str, ...]
    root: _Node
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
            root=_Node(prior=1.0),
            started_at=perf_counter(),
            remaining_simulations=simulation_count,
            ply=_position_ply(position),
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
            move=_select_final_move_at_ply(self.root, self.ply, self.move_selection, self.rng),
            policy_targets=_visit_count_policy_targets(self.root),
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


def _select_pending_simulation(root: _Node, board: ShogiBoard, *, c_puct: float) -> _SelectedSimulation | None:
    node = root
    path = [node]
    while node.children:
        selected = _select_child_node(node, c_puct=c_puct)
        if selected is None:
            return None
        move, node = selected
        board.push_usi(move)
        path.append(node)
    return _SelectedSimulation(path=path, board=board, node=node)


def _select_child_node(node: _Node, *, c_puct: float) -> tuple[str, _Node] | None:
    parent_visits = max(1, node.visit_count)

    def score(item: tuple[str, _Node]) -> tuple[float, str]:
        move, child = item
        exploration = c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
        return -child.value_mean + exploration, move

    candidates = [item for item in node.children.items() if not item[1].pending]
    if not candidates:
        return None
    return max(candidates, key=score)


def _select_final_move(
    root: _Node,
    position: UsiPosition,
    config: MoveSelectionConfig,
    rng: random.Random,
) -> str:
    return _select_final_move_at_ply(root, _position_ply(position), config, rng)


def _select_final_move_at_ply(root: _Node, ply: int, config: MoveSelectionConfig, rng: random.Random) -> str:
    if config.mode == "visit_sample" and ply < config.temperature_plies:
        return _sample_visit_count_move(root, temperature=config.temperature, rng=rng)
    return _deterministic_final_move(root)


def _deterministic_final_move(root: _Node) -> str:
    return max(root.children.items(), key=lambda item: (item[1].visit_count, -item[1].value_mean, item[0]))[0]


def _sample_visit_count_move(root: _Node, *, temperature: float, rng: random.Random) -> str:
    moves = tuple(root.children)
    weights = [max(0, root.children[move].visit_count) ** (1.0 / temperature) for move in moves]
    total = sum(weights)
    if total <= 0:
        return rng.choice(moves)
    threshold = rng.random() * total
    cumulative = 0.0
    for move, weight in zip(moves, weights, strict=True):
        cumulative += weight
        if cumulative >= threshold:
            return move
    return moves[-1]


def _position_ply(position: UsiPosition) -> int:
    words = position.command.split()
    return len(words[words.index("moves") + 1 :]) if "moves" in words else 0


def _expand_node_with_evaluation(node: _Node, legal_moves: tuple[str, ...], priors: dict[str, float]) -> None:
    normalized_priors = _normalize_priors(legal_moves, priors)
    node.children = {move: _Node(prior=normalized_priors[move]) for move in legal_moves}


def _backpropagate_path(path: list[_Node], value: float) -> None:
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


def _normalize_priors(legal_moves: tuple[str, ...], priors: dict[str, float]) -> dict[str, float]:
    positive_priors = {move: max(0.0, float(priors.get(move, 0.0))) for move in legal_moves}
    total = sum(positive_priors.values())
    if total <= 0.0:
        uniform = 1.0 / len(legal_moves)
        return {move: uniform for move in legal_moves}
    return {move: prior / total for move, prior in positive_priors.items()}


def _visit_count_policy_targets(root: _Node) -> dict[str, float]:
    total = sum(child.visit_count for child in root.children.values())
    if total <= 0:
        return _normalize_priors(tuple(root.children), {move: child.prior for move, child in root.children.items()})
    return {move: child.visit_count / total for move, child in root.children.items()}
