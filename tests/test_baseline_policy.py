import unittest

from shogi_arena_agent.baseline_policy import DeterministicLegalMovePolicy
from shogi_arena_agent.usi import UsiPosition, board_from_position


class DeterministicLegalMovePolicyTest(unittest.TestCase):
    def test_returns_legal_move(self) -> None:
        position = UsiPosition(command="position startpos moves 7g7f")
        move = DeterministicLegalMovePolicy().select_move(position)
        legal_moves = {legal_move.usi() for legal_move in board_from_position(position).legal_moves}

        self.assertIn(move, legal_moves)


if __name__ == "__main__":
    unittest.main()
