from __future__ import annotations

import math
import random
from collections.abc import Sequence
from time import perf_counter
from dataclasses import dataclass, field
from typing import Protocol

from shogi_arena_agent.board_backend import ShogiBoard, copy_board, legal_move_usis, validate_board_backend
from shogi_arena_agent.mcts_tree import (
    MctsNode,
    PendingSimulation,
    SelectedSimulation,
    normalize_priors,
    position_moves,
    select_final_move,
    visit_count_policy_targets,
)
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
    root_reuse: bool = False

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


class MctsSearchSession:
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
        self.last_policy_targets: dict[str, float] | None = None
        self.last_performance: MctsMovePerformance | None = None
        self._model_call_count = 0
        self._model_wall_time_sec = 0.0
        self._leaf_eval_batch_sizes: list[int] = []
        self._root: MctsNode | None = None
        self._root_moves: tuple[str, ...] | None = None

    def select_move(self, position: UsiPosition) -> str:
        started_at = perf_counter()
        self._model_call_count = 0
        self._model_wall_time_sec = 0.0
        self._leaf_eval_batch_sizes = []
        board = board_from_position(position, backend=self.config.board_backend)
        position_moves_tuple = position_moves(position)
        legal_moves = legal_move_usis(board)
        if not legal_moves:
            self._discard_root()
            self.last_policy_targets = None
            self.last_performance = self._performance_since(started_at, output_count=0)
            return RESIGN_MOVE

        root = self._root_for_position(position_moves_tuple)
        if not root.children:
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

        self.last_policy_targets = visit_count_policy_targets(root)
        self.last_performance = self._performance_since(started_at, output_count=completed_simulations)
        selected_move = select_final_move(root, position, self.move_selection, self._rng)
        self._store_root(root, position_moves_tuple)
        return selected_move

    def _root_for_position(self, position_moves: tuple[str, ...]) -> MctsNode:
        if not self.config.root_reuse or self._root is None or self._root_moves is None:
            return MctsNode(prior=1.0)
        if len(position_moves) < len(self._root_moves) or position_moves[: len(self._root_moves)] != self._root_moves:
            self._discard_root()
            return MctsNode(prior=1.0)
        root = self._root
        for move in position_moves[len(self._root_moves) :]:
            child = root.children.get(move)
            if child is None:
                self._discard_root()
                return MctsNode(prior=1.0)
            root = child
        root.pending = False
        return root

    def _store_root(self, root: MctsNode, position_moves: tuple[str, ...]) -> None:
        if not self.config.root_reuse:
            self._discard_root()
            return
        self._root = root
        self._root_moves = position_moves

    def _discard_root(self) -> None:
        self._root = None
        self._root_moves = None

    def _run_simulation_batch(self, root: MctsNode, board: ShogiBoard, *, max_count: int) -> int:
        pending: list[PendingSimulation] = []
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
            pending.append(PendingSimulation(path=simulation.path, board=simulation.board, legal_moves=legal_moves))

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

    def _select_simulation(self, root: MctsNode, board: ShogiBoard) -> SelectedSimulation | None:
        node = root
        path = [node]
        while node.children:
            selected = self._select_child(node)
            if selected is None:
                return None
            move, node = selected
            board.push_usi(move)
            path.append(node)
        return SelectedSimulation(path=path, board=board, node=node)

    def _backpropagate(self, path: list[MctsNode], value: float) -> None:
        for visited_node in reversed(path):
            visited_node.visit_count += 1
            visited_node.value_sum += value
            value = -value

    def _expand(self, node: MctsNode, board: ShogiBoard) -> float:
        legal_moves = legal_move_usis(board)
        if not legal_moves:
            return -1.0

        started_at = perf_counter()
        priors, value = self.evaluator.evaluate_batch(((board, legal_moves),))[0]
        self._model_call_count += 1
        self._model_wall_time_sec += perf_counter() - started_at
        self._expand_with_evaluation(node, legal_moves, priors)
        return max(-1.0, min(1.0, float(value)))

    def _expand_with_evaluation(self, node: MctsNode, legal_moves: tuple[str, ...], priors: dict[str, float]) -> None:
        normalized_priors = normalize_priors(legal_moves, priors)
        node.children = {move: MctsNode(prior=normalized_priors[move]) for move in legal_moves}

    def _select_child(self, node: MctsNode) -> tuple[str, MctsNode] | None:
        parent_visits = max(1, node.visit_count)

        def score(item: tuple[str, MctsNode]) -> tuple[float, str]:
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


class MctsMoveSelector:
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
        self._default_session = self.new_session()
        self.last_policy_targets: dict[str, float] | None = None
        self.last_performance: MctsMovePerformance | None = None

    def new_session(self) -> MctsSearchSession:
        return MctsSearchSession(
            self.evaluator,
            config=self.config,
            move_selection=self.move_selection,
        )

    def select_move(self, position: UsiPosition) -> str:
        move = self._default_session.select_move(position)
        self.last_policy_targets = self._default_session.last_policy_targets
        self.last_performance = self._default_session.last_performance
        return move

