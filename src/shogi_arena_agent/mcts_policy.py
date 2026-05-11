from __future__ import annotations

import math
import copy
from collections.abc import Sequence
from time import perf_counter
from dataclasses import dataclass, field
from typing import Protocol

import shogi

from shogi_arena_agent.usi import RESIGN_MOVE, UsiPosition, board_from_position


class PolicyValueEvaluator(Protocol):
    def evaluate_batch(
        self,
        requests: Sequence[tuple[shogi.Board, tuple[str, ...]]],
    ) -> list[tuple[dict[str, float], float]]:
        """Return move priors and values from the side-to-move perspective."""


@dataclass(frozen=True)
class MctsConfig:
    simulation_count: int = 32
    c_puct: float = 1.5
    evaluation_batch_size: int = 1
    move_time_limit_sec: float | None = None

    def __post_init__(self) -> None:
        if self.simulation_count <= 0:
            raise ValueError("simulation_count must be positive")
        if self.c_puct <= 0.0:
            raise ValueError("c_puct must be positive")
        if self.evaluation_batch_size <= 0:
            raise ValueError("evaluation_batch_size must be positive")
        if self.move_time_limit_sec is not None and self.move_time_limit_sec < 0.0:
            raise ValueError("move_time_limit_sec must be non-negative")


@dataclass(frozen=True)
class MctsMovePerformance:
    request_wall_time_sec: float
    model_call_count: int
    model_wall_time_sec: float
    non_model_wall_time_sec: float
    output_count: int
    output_per_sec: float


@dataclass(frozen=True)
class MctsMoveResult:
    move: str
    policy_targets: dict[str, float] | None
    performance: MctsMovePerformance


class UniformPolicyValueEvaluator:
    def evaluate_batch(
        self,
        requests: Sequence[tuple[shogi.Board, tuple[str, ...]]],
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
    ) -> None:
        self.evaluator = evaluator or UniformPolicyValueEvaluator()
        self.config = config or MctsConfig()
        self.last_policy_targets: dict[str, float] | None = None
        self.last_performance: MctsMovePerformance | None = None
        self._model_call_count = 0
        self._model_wall_time_sec = 0.0

    def select_move(self, position: UsiPosition) -> str:
        started_at = perf_counter()
        self._model_call_count = 0
        self._model_wall_time_sec = 0.0
        board = board_from_position(position)
        legal_moves = _legal_move_usis(board)
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
        return max(root.children.items(), key=lambda item: (item[1].visit_count, -item[1].value_mean, item[0]))[0]

    def _run_simulation_batch(self, root: _Node, board: shogi.Board, *, max_count: int) -> int:
        pending: list[_PendingSimulation] = []
        completed = 0
        for _ in range(max_count):
            simulation = self._select_simulation(root, copy.deepcopy(board))
            if simulation is None:
                break
            if simulation.board.is_game_over():
                self._backpropagate(simulation.path, -1.0)
                completed += 1
                continue
            legal_moves = _legal_move_usis(simulation.board)
            if not legal_moves:
                self._backpropagate(simulation.path, -1.0)
                completed += 1
                continue
            simulation.node.pending = True
            pending.append(_PendingSimulation(path=simulation.path, board=simulation.board, legal_moves=legal_moves))

        if pending:
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

    def _select_simulation(self, root: _Node, board: shogi.Board) -> _SelectedSimulation | None:
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

    def _expand(self, node: _Node, board: shogi.Board) -> float:
        legal_moves = _legal_move_usis(board)
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
        return MctsMovePerformance(
            request_wall_time_sec=request_wall_time_sec,
            model_call_count=self._model_call_count,
            model_wall_time_sec=self._model_wall_time_sec,
            non_model_wall_time_sec=non_model_wall_time_sec,
            output_count=output_count,
            output_per_sec=output_per_sec,
        )


class BatchedMctsMoveSelector:
    def __init__(
        self,
        evaluator: PolicyValueEvaluator | None = None,
        *,
        config: MctsConfig | None = None,
    ) -> None:
        self.evaluator = evaluator or UniformPolicyValueEvaluator()
        self.config = config or MctsConfig()

    def select_moves(self, positions: Sequence[UsiPosition]) -> list[MctsMoveResult]:
        states = [_BatchedSearchState.from_position(position, self.config.simulation_count) for position in positions]
        active_states = [state for state in states if state.legal_moves]
        if active_states:
            self._expand_roots(active_states)
        while any(state.remaining_simulations > 0 for state in active_states):
            pending: list[tuple[_BatchedSearchState, _PendingSimulation]] = []
            made_progress = False
            for state in active_states:
                if state.remaining_simulations <= 0:
                    continue
                simulation = _select_pending_simulation(state.root, copy.deepcopy(state.board), c_puct=self.config.c_puct)
                if simulation is None:
                    state.remaining_simulations = 0
                    continue
                if simulation.board.is_game_over():
                    _backpropagate_path(simulation.path, -1.0)
                    state.completed_simulations += 1
                    state.remaining_simulations -= 1
                    made_progress = True
                    continue
                legal_moves = _legal_move_usis(simulation.board)
                if not legal_moves:
                    _backpropagate_path(simulation.path, -1.0)
                    state.completed_simulations += 1
                    state.remaining_simulations -= 1
                    made_progress = True
                    continue
                simulation.node.pending = True
                pending.append((state, _PendingSimulation(path=simulation.path, board=simulation.board, legal_moves=legal_moves)))
                made_progress = True
                if len(pending) >= self.config.evaluation_batch_size:
                    break
            if pending:
                self._evaluate_pending(pending)
            elif not made_progress:
                break
        return [state.to_result() for state in states]

    def _expand_roots(self, states: Sequence["_BatchedSearchState"]) -> None:
        started_at = perf_counter()
        evaluations = self.evaluator.evaluate_batch(tuple((state.board, state.legal_moves) for state in states))
        elapsed = perf_counter() - started_at
        if len(evaluations) != len(states):
            raise ValueError("batch evaluator must return one evaluation per request")
        for state, (priors, _value) in zip(states, evaluations, strict=True):
            _expand_node_with_evaluation(state.root, state.legal_moves, priors)
            state.model_call_count += 1
            state.model_wall_time_sec += elapsed

    def _evaluate_pending(self, pending: Sequence[tuple["_BatchedSearchState", _PendingSimulation]]) -> None:
        started_at = perf_counter()
        evaluations = self.evaluator.evaluate_batch(tuple((simulation.board, simulation.legal_moves) for _state, simulation in pending))
        elapsed = perf_counter() - started_at
        if len(evaluations) != len(pending):
            raise ValueError("batch evaluator must return one evaluation per request")
        for (state, simulation), (priors, value) in zip(pending, evaluations, strict=True):
            simulation.path[-1].pending = False
            _expand_node_with_evaluation(simulation.path[-1], simulation.legal_moves, priors)
            _backpropagate_path(simulation.path, max(-1.0, min(1.0, float(value))))
            state.completed_simulations += 1
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
    board: shogi.Board
    node: _Node


@dataclass
class _PendingSimulation:
    path: list[_Node]
    board: shogi.Board
    legal_moves: tuple[str, ...]


@dataclass
class _BatchedSearchState:
    board: shogi.Board
    legal_moves: tuple[str, ...]
    root: _Node
    started_at: float
    remaining_simulations: int
    completed_simulations: int = 0
    model_call_count: int = 0
    model_wall_time_sec: float = 0.0

    @classmethod
    def from_position(cls, position: UsiPosition, simulation_count: int) -> "_BatchedSearchState":
        board = board_from_position(position)
        return cls(
            board=board,
            legal_moves=_legal_move_usis(board),
            root=_Node(prior=1.0),
            started_at=perf_counter(),
            remaining_simulations=simulation_count,
        )

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
                ),
            )
        return MctsMoveResult(
            move=max(self.root.children.items(), key=lambda item: (item[1].visit_count, -item[1].value_mean, item[0]))[0],
            policy_targets=_visit_count_policy_targets(self.root),
            performance=_batched_performance_since(
                self.started_at,
                model_call_count=self.model_call_count,
                model_wall_time_sec=self.model_wall_time_sec,
                output_count=self.completed_simulations,
            ),
        )


def _legal_move_usis(board: shogi.Board) -> tuple[str, ...]:
    return tuple(sorted(move.usi() for move in board.legal_moves))


def _select_pending_simulation(root: _Node, board: shogi.Board, *, c_puct: float) -> _SelectedSimulation | None:
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
) -> MctsMovePerformance:
    request_wall_time_sec = perf_counter() - started_at
    non_model_wall_time_sec = max(0.0, request_wall_time_sec - model_wall_time_sec)
    output_per_sec = output_count / request_wall_time_sec if request_wall_time_sec > 0 else 0.0
    return MctsMovePerformance(
        request_wall_time_sec=request_wall_time_sec,
        model_call_count=model_call_count,
        model_wall_time_sec=model_wall_time_sec,
        non_model_wall_time_sec=non_model_wall_time_sec,
        output_count=output_count,
        output_per_sec=output_per_sec,
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
