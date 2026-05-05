import unittest

import shogi

from shogi_arena_agent.shogi_game import play_shogi_game
from shogi_arena_agent.mcts_policy import MctsConfig, MctsPolicy, PolicyValueEvaluator
from shogi_arena_agent.usi import UsiEngine, UsiPosition, board_from_position


class PriorOnlyEvaluator:
    def __init__(self, preferred_move: str) -> None:
        self.preferred_move = preferred_move

    def evaluate(self, board: shogi.Board, legal_moves: tuple[str, ...]) -> tuple[dict[str, float], float]:
        return {move: 1.0 if move == self.preferred_move else 0.0 for move in legal_moves}, 0.0


class CaptureValueEvaluator:
    def evaluate(self, board: shogi.Board, legal_moves: tuple[str, ...]) -> tuple[dict[str, float], float]:
        if "8h2b+" in legal_moves:
            return {move: 1.0 if move == "8h2b+" else 0.1 for move in legal_moves}, 0.0
        return {move: 1.0 for move in legal_moves}, _captured_piece_value(board)


class MctsPolicyTest(unittest.TestCase):
    def test_returns_legal_move(self) -> None:
        move = MctsPolicy(config=MctsConfig(simulation_count=4)).select_move(UsiPosition())

        board = shogi.Board()
        self.assertIn(move, {legal_move.usi() for legal_move in board.legal_moves})

    def test_policy_prior_guides_search(self) -> None:
        position = UsiPosition()
        legal_moves = tuple(sorted(move.usi() for move in board_from_position(position).legal_moves))
        preferred_move = "7g7f"
        self.assertIn(preferred_move, legal_moves)

        move = MctsPolicy(
            PriorOnlyEvaluator(preferred_move),
            config=MctsConfig(simulation_count=8),
        ).select_move(position)

        self.assertEqual(move, preferred_move)

    def test_value_guides_search_after_expansion(self) -> None:
        position = UsiPosition(command="position startpos moves 7g7f 3c3d")

        move = MctsPolicy(
            CaptureValueEvaluator(),
            config=MctsConfig(simulation_count=32, c_puct=1.5),
        ).select_move(position)

        self.assertEqual(move, "8h2b+")

    def test_can_play_shogi_game(self) -> None:
        player = UsiEngine(
            policy=MctsPolicy(
                PriorOnlyEvaluator("7g7f"),
                config=MctsConfig(simulation_count=4),
            )
        )

        result = play_shogi_game(black=player, white=UsiEngine(), max_plies=4)

        self.assertEqual(result.end_reason, "max_plies")
        self.assertEqual(len(result.plies), 4)

    def test_rejects_invalid_config(self) -> None:
        with self.assertRaises(ValueError):
            MctsConfig(simulation_count=0)
        with self.assertRaises(ValueError):
            MctsConfig(c_puct=0.0)


def _captured_piece_value(board: shogi.Board) -> float:
    black_hand_count = sum(board.pieces_in_hand[shogi.BLACK].values())
    white_hand_count = sum(board.pieces_in_hand[shogi.WHITE].values())
    if black_hand_count > white_hand_count:
        return 1.0 if board.turn == shogi.BLACK else -1.0
    if white_hand_count > black_hand_count:
        return 1.0 if board.turn == shogi.WHITE else -1.0
    return 0.0


if __name__ == "__main__":
    unittest.main()
