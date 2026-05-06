from __future__ import annotations

import math
import copy
from dataclasses import dataclass, field
from typing import Protocol

import shogi

from shogi_arena_agent.usi import RESIGN_MOVE, UsiPosition, board_from_position


class PolicyValueEvaluator(Protocol):
    def evaluate(self, board: shogi.Board, legal_moves: tuple[str, ...]) -> tuple[dict[str, float], float]:
        """Return move priors and a value from the side-to-move perspective."""


@dataclass(frozen=True)
class MctsConfig:
    simulation_count: int = 32
    c_puct: float = 1.5

    def __post_init__(self) -> None:
        if self.simulation_count <= 0:
            raise ValueError("simulation_count must be positive")
        if self.c_puct <= 0.0:
            raise ValueError("c_puct must be positive")


class UniformPolicyValueEvaluator:
    def evaluate(self, board: shogi.Board, legal_moves: tuple[str, ...]) -> tuple[dict[str, float], float]:
        if not legal_moves:
            return {}, -1.0
        prior = 1.0 / len(legal_moves)
        return {move: prior for move in legal_moves}, 0.0


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

    def select_move(self, position: UsiPosition) -> str:
        board = board_from_position(position)
        legal_moves = _legal_move_usis(board)
        if not legal_moves:
            self.last_policy_targets = None
            return RESIGN_MOVE

        root = _Node(prior=1.0)
        self._expand(root, board)
        for _ in range(self.config.simulation_count):
            self._run_simulation(root, copy.deepcopy(board))

        self.last_policy_targets = _visit_count_policy_targets(root)
        return max(root.children.items(), key=lambda item: (item[1].visit_count, item[1].value_mean, item[0]))[0]

    def _run_simulation(self, root: _Node, board: shogi.Board) -> None:
        node = root
        path = [node]
        while node.children:
            move, node = self._select_child(node)
            board.push_usi(move)
            path.append(node)

        if board.is_game_over():
            value = -1.0
        else:
            value = self._expand(node, board)

        for visited_node in reversed(path):
            visited_node.visit_count += 1
            visited_node.value_sum += value
            value = -value

    def _expand(self, node: _Node, board: shogi.Board) -> float:
        legal_moves = _legal_move_usis(board)
        if not legal_moves:
            return -1.0

        priors, value = self.evaluator.evaluate(board, legal_moves)
        normalized_priors = _normalize_priors(legal_moves, priors)
        node.children = {move: _Node(prior=normalized_priors[move]) for move in legal_moves}
        return max(-1.0, min(1.0, float(value)))

    def _select_child(self, node: _Node) -> tuple[str, _Node]:
        parent_visits = max(1, node.visit_count)

        def score(item: tuple[str, _Node]) -> tuple[float, str]:
            move, child = item
            exploration = self.config.c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
            return -child.value_mean + exploration, move

        return max(node.children.items(), key=score)


@dataclass
class _Node:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[str, "_Node"] = field(default_factory=dict)

    @property
    def value_mean(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def _legal_move_usis(board: shogi.Board) -> tuple[str, ...]:
    return tuple(sorted(move.usi() for move in board.legal_moves))


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
