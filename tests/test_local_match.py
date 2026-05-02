import unittest

from shogi_arena_agent.local_match import play_local_match, position_command
from shogi_arena_agent.usi import UsiEngine, UsiPosition


class IllegalPolicy:
    def select_move(self, position: UsiPosition) -> str:
        return "7g7f"


class LocalMatchTest(unittest.TestCase):
    def test_position_command_without_moves(self) -> None:
        self.assertEqual(position_command(()), "position startpos")

    def test_position_command_with_moves(self) -> None:
        self.assertEqual(position_command(("7g7f", "3c3d")), "position startpos moves 7g7f 3c3d")

    def test_default_engines_play_legal_moves_until_max_plies(self) -> None:
        result = play_local_match(max_plies=6)

        self.assertEqual(result.end_reason, "max_plies")
        self.assertEqual(len(result.moves), 6)

    def test_match_stops_on_illegal_move(self) -> None:
        result = play_local_match(
            black=UsiEngine(policy=IllegalPolicy()),
            white=UsiEngine(policy=IllegalPolicy()),
            max_plies=4,
        )

        self.assertEqual(result.end_reason, "illegal_move")
        self.assertEqual(result.moves, ("7g7f",))


if __name__ == "__main__":
    unittest.main()
