import unittest
import tempfile
import json
from pathlib import Path

from shogi_arena_agent.shogi_game import (
    ShogiActorSpec,
    ShogiDecisionTelemetry,
    ShogiGameRecord,
    ShogiTransitionRecord,
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


class SessionPolicy:
    def __init__(self, move: str) -> None:
        self.move = move
        self.positions: list[UsiPosition] = []

    def select_move(self, position: UsiPosition) -> str:
        self.positions.append(position)
        return self.move


class SessionPolicyFactory:
    def __init__(self, move: str) -> None:
        self.move = move
        self.sessions: list[SessionPolicy] = []

    def new_session(self) -> SessionPolicy:
        session = SessionPolicy(self.move)
        self.sessions.append(session)
        return session

    def select_move(self, position: UsiPosition) -> str:
        raise AssertionError("play_shogi_game should use a per-game session")


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
            white_actor=ShogiActorSpec(kind="usi_engine", name="usi-engine", settings={"go_command": "go nodes 10"}),
            max_plies=2,
        )

        self.assertEqual(result.black_actor.kind, "checkpoint")
        self.assertEqual(result.black_actor.settings["checkpoint"], "a.pt")
        self.assertEqual(result.white_actor.kind, "usi_engine")
        self.assertEqual(result.white_actor.settings["go_command"], "go nodes 10")

    def test_records_raw_decision_usi_info_lines(self) -> None:
        result = play_shogi_game(black=InfoLinePlayer(), white=UsiEngine(), max_plies=1)

        self.assertEqual(
            result.transitions[0].decision_usi_info_lines,
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

    def test_game_record_json_round_trips_decision_telemetry(self) -> None:
        result = ShogiGameRecord(
            black_actor=ShogiActorSpec(kind="checkpoint", name="black", settings={}),
            white_actor=ShogiActorSpec(kind="checkpoint", name="white", settings={}),
            initial_position_sfen="start",
            transitions=(
                ShogiTransitionRecord(
                    ply=0,
                    side="black",
                    position_sfen="before",
                    legal_moves=("7g7f",),
                    action_usi="7g7f",
                    next_position_sfen="after",
                    reward=0.0,
                    done=True,
                    decision_telemetry=ShogiDecisionTelemetry(
                        move_performance={"request_wall_time_sec": 0.4},
                        batch_performance={"position_count": 4},
                    ),
                ),
            ),
            end_reason="max_plies",
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            save_shogi_game_records_jsonl((result,), path)
            loaded = load_shogi_game_records_jsonl(path)

        self.assertEqual(loaded, (result,))

    def test_loads_legacy_performance_info_lines_as_decision_telemetry(self) -> None:
        payload = shogi_game_record_to_json(play_shogi_game(max_plies=1))
        transition = payload["transitions"][0]
        assert isinstance(transition, dict)
        transition["decision_usi_info_lines"] = [
            "info depth 1 nodes 1 pv 7g7f",
            'info string intrep_performance {"request_wall_time_sec": 0.4}',
            'info string intrep_batch_performance {"position_count": 4}',
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            loaded = load_shogi_game_records_jsonl(path)

        loaded_transition = loaded[0].transitions[0]
        self.assertEqual(loaded_transition.decision_usi_info_lines, ("info depth 1 nodes 1 pv 7g7f",))
        self.assertIsNotNone(loaded_transition.decision_telemetry)
        assert loaded_transition.decision_telemetry is not None
        self.assertEqual(loaded_transition.decision_telemetry.move_performance, {"request_wall_time_sec": 0.4})
        self.assertEqual(loaded_transition.decision_telemetry.batch_performance, {"position_count": 4})

    def test_play_shogi_game_starts_policy_session_per_game(self) -> None:
        black_policy = SessionPolicyFactory("7g7f")
        white_policy = SessionPolicyFactory("3c3d")

        play_shogi_game(
            black=UsiEngine(policy=black_policy),
            white=UsiEngine(policy=white_policy),
            max_plies=2,
        )

        self.assertEqual(len(black_policy.sessions), 1)
        self.assertEqual(len(white_policy.sessions), 1)
        self.assertEqual(black_policy.sessions[-1].positions, [UsiPosition()])
        self.assertEqual(white_policy.sessions[-1].positions, [UsiPosition(command="position startpos moves 7g7f")])


if __name__ == "__main__":
    unittest.main()
