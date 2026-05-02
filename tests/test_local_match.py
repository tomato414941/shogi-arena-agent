import unittest
import tempfile
from pathlib import Path

from shogi_arena_agent.local_match import load_match_result, play_local_match, position_command, save_match_result
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
        self.assertNotEqual(result.final_sfen, "")

    def test_match_stops_on_illegal_move(self) -> None:
        result = play_local_match(
            black=UsiEngine(policy=IllegalPolicy()),
            white=UsiEngine(policy=IllegalPolicy()),
            max_plies=4,
        )

        self.assertEqual(result.end_reason, "illegal_move")
        self.assertEqual(result.moves, ("7g7f",))

    def test_match_result_round_trip(self) -> None:
        result = play_local_match(max_plies=2)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "match.json"
            save_match_result(result, path)
            loaded = load_match_result(path)

        self.assertEqual(loaded, result)


if __name__ == "__main__":
    unittest.main()
