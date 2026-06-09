from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from shogi_arena_agent.board_backend import ShogiBoard

MovePriors = Mapping[str, float] | Sequence[float]


class PolicyValueEvaluator(Protocol):
    def evaluate_batch(
        self,
        requests: Sequence[tuple[ShogiBoard, tuple[str, ...]]],
    ) -> list[tuple[MovePriors, float]]:
        """Return move priors and values from the side-to-move perspective."""


class UniformPolicyValueEvaluator:
    def evaluate_batch(
        self,
        requests: Sequence[tuple[ShogiBoard, tuple[str, ...]]],
    ) -> list[tuple[MovePriors, float]]:
        evaluations: list[tuple[MovePriors, float]] = []
        for _board, legal_moves in requests:
            if not legal_moves:
                evaluations.append(({}, -1.0))
                continue
            prior = 1.0 / len(legal_moves)
            evaluations.append(({move: prior for move in legal_moves}, 0.0))
        return evaluations
