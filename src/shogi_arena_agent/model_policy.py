from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from shogi_arena_agent.board_backend import ShogiBoard, legal_move_usis
from shogi_arena_agent.usi import RESIGN_MOVE, UsiPosition, board_from_position


MoveRanker = Callable[[str, tuple[str, ...]], Sequence[float]]
PositionEvaluation = tuple[dict[str, float], float]
PositionEvaluationRequest = tuple[str, tuple[str, ...]]


class RankedMovePolicy:
    def __init__(self, rank_moves: MoveRanker, *, board_backend: str = "python-shogi") -> None:
        self.rank_moves = rank_moves
        self.board_backend = board_backend

    def select_move(self, position: UsiPosition) -> str:
        board = board_from_position(position, backend=self.board_backend)
        legal_moves = legal_move_usis(board)
        if not legal_moves:
            return RESIGN_MOVE

        scores = tuple(float(score) for score in self.rank_moves(board.sfen(), legal_moves))
        if len(scores) != len(legal_moves):
            raise ValueError("ranker must return one score per legal move")
        best_index = max(range(len(legal_moves)), key=lambda index: scores[index])
        return legal_moves[best_index]


class DirectMovePolicy:
    def __init__(self, evaluator: ShogiMoveChoiceCheckpointEvaluator, *, board_backend: str = "python-shogi") -> None:
        self.evaluator = evaluator
        self.board_backend = board_backend

    def select_move(self, position: UsiPosition) -> str:
        board = board_from_position(position, backend=self.board_backend)
        legal_moves = legal_move_usis(board)
        if not legal_moves:
            return RESIGN_MOVE
        priors, _value = self.evaluator.evaluate_batch(((board, legal_moves),))[0]
        return max(legal_moves, key=lambda move: priors.get(move, 0.0))


class ShogiMoveChoiceCheckpointPolicy(DirectMovePolicy):
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        board_backend: str = "python-shogi",
    ) -> ShogiMoveChoiceCheckpointPolicy:
        return cls(ShogiMoveChoiceCheckpointEvaluator.from_checkpoint(checkpoint_path, device=device), board_backend=board_backend)


class ShogiMoveChoiceCheckpointEvaluator:
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, *, device: str = "cpu") -> ShogiMoveChoiceCheckpointEvaluator:
        try:
            from intrep.problems.shogi_policy_value.inference import ShogiPolicyValueCheckpointEvaluator
        except ImportError as error:
            raise RuntimeError(
                "intelligence-representation and torch are required to use shogi move choice checkpoints"
            ) from error

        evaluator = ShogiPolicyValueCheckpointEvaluator.from_checkpoint(checkpoint_path, device=device)
        return cls(evaluator)

    def __init__(
        self,
        evaluate_positions: Callable[[Sequence[PositionEvaluationRequest]], list[PositionEvaluation]] | object,
    ) -> None:
        self._backend = evaluate_positions
        self.last_performance: dict[str, float] = {}

    def evaluate(self, board: ShogiBoard, legal_moves: tuple[str, ...]) -> tuple[dict[str, float], float]:
        return self.evaluate_batch(((board, legal_moves),))[0]

    def evaluate_batch(self, requests: Sequence[tuple[ShogiBoard, tuple[str, ...]]]) -> list[PositionEvaluation]:
        position_requests = [(board.sfen(), legal_moves) for board, legal_moves in requests]
        return self.evaluate_positions(position_requests)

    def __call__(self, requests: Sequence[PositionEvaluationRequest]) -> list[PositionEvaluation]:
        return self.evaluate_positions(requests)

    def evaluate_positions(self, requests: Sequence[PositionEvaluationRequest]) -> list[PositionEvaluation]:
        if hasattr(self._backend, "evaluate_batch"):
            evaluations = self._backend.evaluate_batch(requests)  # type: ignore[union-attr]
        elif callable(self._backend):
            evaluations = self._backend(requests)
        else:
            raise TypeError("checkpoint evaluator backend must be callable or expose evaluate_batch")
        self.last_performance = dict(getattr(self._backend, "last_performance", {}))
        return evaluations
