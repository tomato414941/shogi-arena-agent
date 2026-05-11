from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from shogi_arena_agent.shogi_game import (
    ShogiActorSpec,
    ShogiGameRecord,
    ShogiTransitionRecord,
    load_shogi_game_records_jsonl,
)
from shogi_arena_agent.usi import UsiEngine


class EvaluateShogiPlayersScriptTest(unittest.TestCase):
    def test_evaluates_players_on_both_sides(self) -> None:
        module = _load_script_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "games.jsonl"
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                module.main(
                    [
                        "--player-kind",
                        "deterministic_legal",
                        "--opponent-kind",
                        "deterministic_legal",
                        "--games",
                        "2",
                        "--max-plies",
                        "2",
                        "--out",
                        str(output_path),
                    ]
                )

            records = load_shogi_game_records_jsonl(output_path)
            summary = json.loads(stdout.getvalue())

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].black_actor.name, "player")
        self.assertEqual(records[0].white_actor.name, "opponent")
        self.assertEqual(records[1].black_actor.name, "opponent")
        self.assertEqual(records[1].white_actor.name, "player")
        self.assertEqual(summary["game_count"], 2)
        self.assertEqual(summary["black_game_count"], 1)
        self.assertEqual(summary["white_game_count"], 1)

    def test_summarizes_in_process_mcts_performance(self) -> None:
        module = _load_script_module()
        record = ShogiGameRecord(
            black_actor=ShogiActorSpec(kind="test", name="black", settings={}),
            white_actor=ShogiActorSpec(kind="test", name="white", settings={}),
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
                    done=False,
                    decision_usi_info_lines=(
                        'info string intrep_performance {"model_call_count": 2, "model_wall_time_sec": 0.1, '
                        '"non_model_wall_time_sec": 0.2, "output_count": 4, "output_per_sec": 10.0, '
                        '"phase_wall_time_sec": {"legal_moves": 0.05}, "request_wall_time_sec": 0.4}',
                    ),
                ),
            ),
            end_reason="max_plies",
        )

        performance = module._performance_summary([record])

        self.assertIsNotNone(performance)
        assert performance is not None
        self.assertEqual(performance["request_count"], 1)
        self.assertEqual(performance["request_wall_time_sec_avg"], 0.4)
        self.assertEqual(performance["model_call_count_avg"], 2.0)

    def test_external_players_are_reused_across_games(self) -> None:
        module = _load_script_module()

        class FakeUsiProcess:
            enter_count = 0

            def __init__(self, **_kwargs: object) -> None:
                self.engine = UsiEngine()

            def __enter__(self) -> "FakeUsiProcess":
                type(self).enter_count += 1
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def position(self, command: str) -> None:
                self.engine.handle_line(command)

            def go(self) -> str:
                return self.engine.policy.select_move(self.engine.position)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "games.jsonl"

            with patch("shogi_arena_agent.player_cli.UsiProcess", FakeUsiProcess), contextlib.redirect_stdout(io.StringIO()):
                module.main(
                    [
                        "--player-kind",
                        "yaneuraou",
                        "--player-yaneuraou-command",
                        "engine",
                        "--opponent-kind",
                        "yaneuraou",
                        "--opponent-yaneuraou-command",
                        "engine",
                        "--games",
                        "4",
                        "--max-plies",
                        "2",
                        "--out",
                        str(output_path),
                    ]
                )

        self.assertEqual(FakeUsiProcess.enter_count, 2)


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_shogi_players.py"
    spec = importlib.util.spec_from_file_location("evaluate_shogi_players", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
