from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from shogi_arena_agent.board_backend import ShogiBoard

MovePriors = Mapping[str, float] | Sequence[float]
PolicyValueEvaluationRequest = tuple[ShogiBoard, tuple[str, ...]] | tuple[
    ShogiBoard,
    tuple[str, ...],
    tuple[int, ...] | None,
]


def aligned_prior_values(
    legal_moves: Sequence[str],
    priors: MovePriors,
    *,
    expected_count: int | None = None,
    clamp_negative: bool = True,
) -> tuple[float, ...]:
    count = len(legal_moves) if expected_count is None else expected_count
    if isinstance(priors, Mapping):
        if len(legal_moves) != count:
            raise ValueError("mapping priors require USI legal moves")
        values = tuple(float(priors.get(move, 0.0)) for move in legal_moves)
    else:
        if len(priors) != count:
            raise ValueError("aligned move priors must match legal move count")
        values = tuple(float(prior) for prior in priors)
    if clamp_negative:
        return tuple(max(0.0, value) for value in values)
    return values


def normalized_prior_values(
    legal_moves: Sequence[str],
    priors: MovePriors,
    *,
    expected_count: int | None = None,
) -> tuple[float, ...]:
    values = aligned_prior_values(legal_moves, priors, expected_count=expected_count)
    if not values:
        return ()
    total = sum(values)
    if total <= 0.0:
        uniform = 1.0 / len(values)
        return tuple(uniform for _value in values)
    inverse_total = 1.0 / total
    return tuple(value * inverse_total for value in values)


def normalized_prior_dict(legal_moves: tuple[str, ...], priors: MovePriors) -> dict[str, float]:
    values = normalized_prior_values(legal_moves, priors)
    return {move: value for move, value in zip(legal_moves, values, strict=True)}


class PolicyValueEvaluator(Protocol):
    def evaluate_batch(
        self,
        requests: Sequence[PolicyValueEvaluationRequest],
    ) -> list[tuple[MovePriors, float]]:
        """Return move priors and values from the side-to-move perspective."""


class UniformPolicyValueEvaluator:
    def evaluate_batch(
        self,
        requests: Sequence[PolicyValueEvaluationRequest],
    ) -> list[tuple[MovePriors, float]]:
        evaluations: list[tuple[MovePriors, float]] = []
        for request in requests:
            legal_moves = request[1]
            if not legal_moves:
                evaluations.append(({}, -1.0))
                continue
            prior = 1.0 / len(legal_moves)
            evaluations.append(({move: prior for move in legal_moves}, 0.0))
        return evaluations
