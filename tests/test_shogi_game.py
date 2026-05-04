import unittest
import tempfile
from pathlib import Path

from shogi_arena_agent.shogi_game import (
    PlayerSpec,
    load_shogi_game_records_jsonl,
    play_shogi_game,
    position_command,
    save_shogi_game_records_jsonl,
)
from shogi_arena_agent.usi import RESIGN_MOVE, UsiEngine, UsiPosition


class IllegalPolicy:
    def select_move(self, position: UsiPosition) -> str:
        return "7g7f"


class ResignPolicy:
    def select_move(self, position: UsiPosition) -> str:
        return RESIGN_MOVE


class ShogiGameTest(unittest.TestCase):
    def test_position_command_without_moves(self) -> None:
        self.assertEqual(position_command(()), "position startpos")

    def test_position_command_with_moves(self) -> None:
        self.assertEqual(position_command(("7g7f", "3c3d")), "position startpos moves 7g7f 3c3d")

    def test_default_engines_play_legal_moves_until_max_plies(self) -> None:
        result = play_shogi_game(max_plies=6)

        self.assertEqual(result.end_reason, "max_plies")
        self.assertEqual(len(result.moves), 6)
        self.assertEqual(result.black_player.name, "black")
        self.assertEqual(result.white_player.name, "white")
        self.assertIsNone(result.winner)

    def test_records_explicit_player_specs(self) -> None:
        result = play_shogi_game(
            black_player=PlayerSpec(kind="checkpoint", name="model-a", settings={"checkpoint": "a.pt"}),
            white_player=PlayerSpec(kind="yaneuraou", name="yaneuraou", settings={"go_command": "go nodes 10"}),
            max_plies=2,
        )

        self.assertEqual(result.black_player.kind, "checkpoint")
        self.assertEqual(result.black_player.settings["checkpoint"], "a.pt")
        self.assertEqual(result.white_player.kind, "yaneuraou")
        self.assertEqual(result.white_player.settings["go_command"], "go nodes 10")

    def test_game_stops_on_illegal_move(self) -> None:
        result = play_shogi_game(
            black=UsiEngine(policy=IllegalPolicy()),
            white=UsiEngine(policy=IllegalPolicy()),
            max_plies=4,
        )

        self.assertEqual(result.end_reason, "illegal_move")
        self.assertEqual(result.moves, ("7g7f",))
        self.assertEqual(result.winner, "black")

    def test_game_stops_on_resign(self) -> None:
        result = play_shogi_game(
            black=UsiEngine(policy=ResignPolicy()),
            white=UsiEngine(),
            max_plies=4,
        )

        self.assertEqual(result.end_reason, "resign")
        self.assertEqual(result.moves, ())
        self.assertEqual(result.winner, "white")

    def test_shogi_game_records_jsonl_round_trip(self) -> None:
        results = (play_shogi_game(max_plies=1), play_shogi_game(max_plies=2))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            save_shogi_game_records_jsonl(results, path)
            loaded = load_shogi_game_records_jsonl(path)

        self.assertEqual(loaded, results)


if __name__ == "__main__":
    unittest.main()
