from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import shogi

from shogi_arena_agent.usi import RESIGN_MOVE, UsiPosition, board_from_position


MoveRanker = Callable[[str, tuple[str, ...]], Sequence[float]]


class RankedMovePolicy:
    def __init__(self, rank_moves: MoveRanker) -> None:
        self.rank_moves = rank_moves

    def select_move(self, position: UsiPosition) -> str:
        board = board_from_position(position)
        legal_moves = tuple(sorted(move.usi() for move in board.legal_moves))
        if not legal_moves:
            return RESIGN_MOVE

        scores = tuple(float(score) for score in self.rank_moves(board.sfen(), legal_moves))
        if len(scores) != len(legal_moves):
            raise ValueError("ranker must return one score per legal move")
        best_index = max(range(len(legal_moves)), key=lambda index: scores[index])
        return legal_moves[best_index]


class ShogiMoveChoiceCheckpointPolicy(RankedMovePolicy):
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, *, device: str = "cpu") -> ShogiMoveChoiceCheckpointPolicy:
        try:
            import torch
            from intrep.shogi_move_choice_checkpoint import load_shogi_move_choice_checkpoint
            from intrep.shogi_move_encoding import shogi_candidate_move_features
            from intrep.shogi_position_encoding import shogi_position_token_ids_from_sfen
        except ImportError as error:
            raise RuntimeError(
                "intelligence-representation and torch are required to use shogi move choice checkpoints"
            ) from error

        model = load_shogi_move_choice_checkpoint(checkpoint_path, device=device)
        torch_device = torch.device(device)

        def rank_moves(position_sfen: str, legal_moves: tuple[str, ...]) -> Sequence[float]:
            position_token_ids = shogi_position_token_ids_from_sfen(position_sfen).unsqueeze(0).to(torch_device)
            candidate_move_features = shogi_candidate_move_features(
                legal_moves,
                max_choice_count=len(legal_moves),
            ).unsqueeze(0).to(torch_device)
            candidate_mask = torch.ones((1, len(legal_moves)), dtype=torch.bool, device=torch_device)
            with torch.no_grad():
                logits = model(position_token_ids, candidate_move_features, candidate_mask)
            return logits.squeeze(0).detach().cpu().tolist()

        return cls(rank_moves)


class ShogiMoveChoiceCheckpointEvaluator:
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, *, device: str = "cpu") -> ShogiMoveChoiceCheckpointEvaluator:
        try:
            import torch
            from intrep.shogi_move_choice_checkpoint import load_shogi_move_choice_checkpoint
            from intrep.shogi_move_encoding import shogi_candidate_move_features
            from intrep.shogi_position_encoding import shogi_position_token_ids_from_sfen
        except ImportError as error:
            raise RuntimeError(
                "intelligence-representation and torch are required to use shogi move choice checkpoints"
            ) from error

        model = load_shogi_move_choice_checkpoint(checkpoint_path, device=device)
        torch_device = torch.device(device)

        def evaluate(position_sfen: str, legal_moves: tuple[str, ...]) -> tuple[dict[str, float], float]:
            position_token_ids = shogi_position_token_ids_from_sfen(position_sfen).unsqueeze(0).to(torch_device)
            candidate_move_features = shogi_candidate_move_features(
                legal_moves,
                max_choice_count=len(legal_moves),
            ).unsqueeze(0).to(torch_device)
            candidate_mask = torch.ones((1, len(legal_moves)), dtype=torch.bool, device=torch_device)
            with torch.no_grad():
                logits = model(position_token_ids, candidate_move_features, candidate_mask).squeeze(0)
                value = model.predict_value(position_token_ids).squeeze(0) if hasattr(model, "predict_value") else None
                probabilities = torch.softmax(logits, dim=0).detach().cpu().tolist()
            prior = {move: float(probabilities[index]) for index, move in enumerate(legal_moves)}
            return prior, 0.0 if value is None else float(value.detach().cpu().item())

        return cls(evaluate)

    def __init__(self, evaluate_position: Callable[[str, tuple[str, ...]], tuple[dict[str, float], float]]) -> None:
        self.evaluate_position = evaluate_position

    def evaluate(self, board: shogi.Board, legal_moves: tuple[str, ...]) -> tuple[dict[str, float], float]:
        return self.evaluate_position(board.sfen(), legal_moves)
