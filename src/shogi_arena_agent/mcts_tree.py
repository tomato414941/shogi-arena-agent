from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol

from shogi_arena_agent.board_backend import ShogiBoard
from shogi_arena_agent.usi import UsiPosition


class MoveSelectionLike(Protocol):
    mode: str
    temperature: float
    temperature_plies: int


@dataclass(slots=True)
class MctsNode:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    pending: bool = False
    children: dict[str, "MctsNode"] = field(default_factory=dict)

    @property
    def value_mean(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass
class SelectedSimulation:
    path: list[MctsNode]
    board: ShogiBoard
    node: MctsNode


@dataclass
class PendingSimulation:
    path: list[MctsNode]
    board: ShogiBoard
    legal_moves: tuple[str, ...]


def select_final_move(
    root: MctsNode,
    position: UsiPosition,
    config: MoveSelectionLike,
    rng: random.Random,
) -> str:
    return select_final_move_at_ply(root, position_ply(position), config, rng)


def select_final_move_at_ply(root: MctsNode, ply: int, config: MoveSelectionLike, rng: random.Random) -> str:
    if config.mode == "visit_sample" and ply < config.temperature_plies:
        return _sample_visit_count_move(root, temperature=config.temperature, rng=rng)
    return deterministic_final_move(root)


def deterministic_final_move(root: MctsNode) -> str:
    return max(root.children.items(), key=lambda item: (item[1].visit_count, -item[1].value_mean, item[0]))[0]


def position_ply(position: UsiPosition) -> int:
    return len(position_moves(position))


def position_moves(position: UsiPosition) -> tuple[str, ...]:
    words = position.command.split()
    return tuple(words[words.index("moves") + 1 :]) if "moves" in words else ()


def normalize_priors(legal_moves: tuple[str, ...], priors: dict[str, float]) -> dict[str, float]:
    positive_priors = {move: max(0.0, float(priors.get(move, 0.0))) for move in legal_moves}
    total = sum(positive_priors.values())
    if total <= 0.0:
        uniform = 1.0 / len(legal_moves)
        return {move: uniform for move in legal_moves}
    return {move: prior / total for move, prior in positive_priors.items()}


def expanded_children(legal_moves: tuple[str, ...], priors: dict[str, float]) -> dict[str, MctsNode]:
    prior_values = tuple(max(0.0, float(priors.get(move, 0.0))) for move in legal_moves)
    total = sum(prior_values)
    if total <= 0.0:
        uniform = 1.0 / len(legal_moves)
        return {move: MctsNode(prior=uniform) for move in legal_moves}
    return {move: MctsNode(prior=prior / total) for move, prior in zip(legal_moves, prior_values, strict=True)}


def visit_count_policy_targets(root: MctsNode) -> dict[str, float]:
    total = sum(child.visit_count for child in root.children.values())
    if total <= 0:
        return normalize_priors(tuple(root.children), {move: child.prior for move, child in root.children.items()})
    return {move: child.visit_count / total for move, child in root.children.items()}


def select_puct_child(node: MctsNode, *, c_puct: float) -> tuple[str, MctsNode] | None:
    parent_sqrt = math.sqrt(max(1, node.visit_count))
    best: tuple[str, MctsNode] | None = None
    best_score: tuple[float, str] | None = None
    for move, child in node.children.items():
        if child.pending:
            continue
        exploration = c_puct * child.prior * parent_sqrt / (1 + child.visit_count)
        score = (-child.value_mean + exploration, move)
        if best_score is None or score > best_score:
            best = (move, child)
            best_score = score
    return best


def _sample_visit_count_move(root: MctsNode, *, temperature: float, rng: random.Random) -> str:
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
