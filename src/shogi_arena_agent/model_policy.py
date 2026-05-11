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
            import torch
            from intrep.problems.shogi_policy_value.checkpoint import load_shogi_policy_value_checkpoint
            from intrep.worlds.shogi.move_encoding import shogi_candidate_move_features
            from intrep.worlds.shogi.position_encoding import shogi_position_token_ids_from_sfen
        except ImportError as error:
            raise RuntimeError(
                "intelligence-representation and torch are required to use shogi move choice checkpoints"
            ) from error

        model = load_shogi_policy_value_checkpoint(checkpoint_path, device=device)
        torch_device = torch.device(device)

        def evaluate_batch(requests: Sequence[PositionEvaluationRequest]) -> list[PositionEvaluation]:
            if not requests:
                return []
            max_choice_count = max(len(legal_moves) for _position_sfen, legal_moves in requests)
            position_token_ids = torch.stack(
                [shogi_position_token_ids_from_sfen(position_sfen) for position_sfen, _legal_moves in requests]
            ).to(torch_device)
            candidate_move_features = torch.stack(
                [
                    shogi_candidate_move_features(
                        legal_moves,
                        max_choice_count=max_choice_count,
                    )
                    for _position_sfen, legal_moves in requests
                ]
            ).to(torch_device)
            candidate_mask = torch.zeros((len(requests), max_choice_count), dtype=torch.bool, device=torch_device)
            for index, (_position_sfen, legal_moves) in enumerate(requests):
                candidate_mask[index, : len(legal_moves)] = True

            with torch.no_grad():
                if hasattr(model, "forward_policy_value"):
                    logits, values = model.forward_policy_value(position_token_ids, candidate_move_features, candidate_mask)
                else:
                    logits = model(position_token_ids, candidate_move_features, candidate_mask)
                    values = model.predict_value(position_token_ids) if hasattr(model, "predict_value") else None

            evaluations: list[PositionEvaluation] = []
            for index, (_position_sfen, legal_moves) in enumerate(requests):
                move_logits = logits[index, : len(legal_moves)]
                probabilities = torch.softmax(move_logits, dim=0).detach().cpu().tolist()
                prior = {move: float(probabilities[move_index]) for move_index, move in enumerate(legal_moves)}
                value = 0.0 if values is None else float(values[index].detach().cpu().item())
                evaluations.append((prior, value))
            return evaluations

        return cls(evaluate_batch)

    def __init__(
        self,
        evaluate_positions: Callable[[Sequence[PositionEvaluationRequest]], list[PositionEvaluation]],
    ) -> None:
        self.evaluate_positions = evaluate_positions

    def evaluate(self, board: ShogiBoard, legal_moves: tuple[str, ...]) -> tuple[dict[str, float], float]:
        return self.evaluate_batch(((board, legal_moves),))[0]

    def evaluate_batch(self, requests: Sequence[tuple[ShogiBoard, tuple[str, ...]]]) -> list[PositionEvaluation]:
        position_requests = [(board.sfen(), legal_moves) for board, legal_moves in requests]
        return self.evaluate_positions(position_requests)
