import unittest
import tempfile
import json
from pathlib import Path

from shogi_arena_agent.shogi_game import (
    ShogiActorSpec,
    load_shogi_game_records_jsonl,
    play_shogi_game,
    position_command,
    save_shogi_game_records_jsonl,
    shogi_game_record_to_json,
)
from shogi_arena_agent.usi import RESIGN_MOVE, UsiEngine, UsiPosition
from shogi_arena_agent.usi_process import UsiGoResult


class IllegalPolicy:
    def select_move(self, position: UsiPosition) -> str:
        return "7g7f"


class ResignPolicy:
    def select_move(self, position: UsiPosition) -> str:
        return RESIGN_MOVE


class InfoLinePlayer:
    def position(self, command: str) -> None:
        pass

    def go(self) -> UsiGoResult:
        return UsiGoResult(
            bestmove="7g7f",
            info_lines=("info multipv 1 score cp 100 pv 7g7f",),
        )


class ShogiGameTest(unittest.TestCase):
    def test_position_command_without_moves(self) -> None:
        self.assertEqual(position_command(()), "position startpos")

    def test_position_command_with_moves(self) -> None:
        self.assertEqual(position_command(("7g7f", "3c3d")), "position startpos moves 7g7f 3c3d")

    def test_default_engines_play_legal_moves_until_max_plies(self) -> None:
        result = play_shogi_game(max_plies=6)

        self.assertEqual(result.end_reason, "max_plies")
        self.assertEqual(len(result.transitions), 6)
        self.assertEqual(result.black_actor.name, "black")
        self.assertEqual(result.white_actor.name, "white")
        self.assertIsNone(result.winner)
        self.assertTrue(result.initial_position_sfen)
        self.assertTrue(result.transitions[0].position_sfen)
        self.assertTrue(result.transitions[0].legal_moves)
        self.assertEqual(result.transitions[0].action_usi, "1g1f")
        self.assertTrue(result.transitions[0].next_position_sfen)
        self.assertEqual(result.transitions[-1].done, True)
        self.assertEqual(result.transitions[-1].reward, 0.0)

    def test_records_explicit_actor_specs(self) -> None:
        result = play_shogi_game(
            black_actor=ShogiActorSpec(kind="checkpoint", name="model-a", settings={"checkpoint": "a.pt"}),
            white_actor=ShogiActorSpec(kind="yaneuraou", name="yaneuraou", settings={"go_command": "go nodes 10"}),
            max_plies=2,
        )

        self.assertEqual(result.black_actor.kind, "checkpoint")
        self.assertEqual(result.black_actor.settings["checkpoint"], "a.pt")
        self.assertEqual(result.white_actor.kind, "yaneuraou")
        self.assertEqual(result.white_actor.settings["go_command"], "go nodes 10")

    def test_records_raw_usi_info_lines(self) -> None:
        result = play_shogi_game(black=InfoLinePlayer(), white=UsiEngine(), max_plies=1)

        self.assertEqual(
            result.transitions[0].usi_info_lines,
            ("info multipv 1 score cp 100 pv 7g7f",),
        )

    def test_game_stops_on_illegal_move(self) -> None:
        result = play_shogi_game(
            black=UsiEngine(policy=IllegalPolicy()),
            white=UsiEngine(policy=IllegalPolicy()),
            max_plies=4,
        )

        self.assertEqual(result.end_reason, "illegal_move")
        self.assertEqual(tuple(transition.action_usi for transition in result.transitions), ("7g7f",))
        self.assertTrue(result.transitions[-1].done)
        self.assertEqual(result.winner, "black")

    def test_game_stops_on_resign(self) -> None:
        result = play_shogi_game(
            black=UsiEngine(policy=ResignPolicy()),
            white=UsiEngine(),
            max_plies=4,
        )

        self.assertEqual(result.end_reason, "resign")
        self.assertEqual(result.transitions, ())
        self.assertEqual(result.winner, "white")

    def test_shogi_game_records_jsonl_round_trip(self) -> None:
        results = (play_shogi_game(max_plies=1), play_shogi_game(max_plies=2))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            save_shogi_game_records_jsonl(results, path)
            loaded = load_shogi_game_records_jsonl(path)

        self.assertEqual(loaded, results)

    def test_shogi_game_record_json_uses_transitions_as_source_of_truth(self) -> None:
        results = (
            play_shogi_game(max_plies=1),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            save_shogi_game_records_jsonl(results, path)
            payload = path.read_text(encoding="utf-8")

        self.assertIn('"transitions"', payload)
        self.assertIn('"action_usi"', payload)
        self.assertIn('"legal_moves"', payload)
        self.assertNotIn('"moves"', payload)

    def test_game_record_json_does_not_store_teacher_targets(self) -> None:
        result = play_shogi_game(max_plies=1)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            payload = shogi_game_record_to_json(result)
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

            loaded = load_shogi_game_records_jsonl(path)

        self.assertEqual(loaded, (result,))
        self.assertNotIn("policy_targets", payload["transitions"][0])


if __name__ == "__main__":
    unittest.main()
