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

from shogi_arena_agent.deterministic_legal_policy import DeterministicLegalMovePolicy
from shogi_arena_agent.shogi_game import load_shogi_game_records_jsonl
from shogi_arena_agent.usi import UsiEngine


class GenerateShogiGamesScriptTest(unittest.TestCase):
    def test_checkpoint_fixed_side_writes_game_records(self) -> None:
        module = _load_script_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "games.jsonl"
            stdout = io.StringIO()

            with patch(
                "shogi_arena_agent.player_cli._load_checkpoint_policy",
                return_value=DeterministicLegalMovePolicy(),
            ), contextlib.redirect_stdout(stdout):
                module.main(
                    [
                        "--black-kind",
                        "checkpoint",
                        "--black-checkpoint",
                        "black.pt",
                        "--white-kind",
                        "checkpoint",
                        "--white-checkpoint",
                        "white.pt",
                        "--games",
                        "1",
                        "--max-plies",
                        "2",
                        "--out",
                        str(output_path),
                    ]
                )

            records = load_shogi_game_records_jsonl(output_path)
            summary = json.loads(stdout.getvalue())

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].black_actor.settings["checkpoint"], "black.pt")
        self.assertEqual(records[0].white_actor.settings["checkpoint"], "white.pt")
        self.assertEqual(records[0].black_actor.settings["evaluation_batch_size"], 1)
        self.assertEqual(summary["game_count"], 1)

    def test_records_checkpoint_evaluation_batch_size(self) -> None:
        module = _load_script_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "games.jsonl"

            with patch(
                "shogi_arena_agent.player_cli._load_checkpoint_policy",
                return_value=DeterministicLegalMovePolicy(),
            ), contextlib.redirect_stdout(io.StringIO()):
                module.main(
                    [
                        "--black-kind",
                        "checkpoint",
                        "--black-checkpoint",
                        "black.pt",
                        "--black-checkpoint-evaluation-batch-size",
                        "8",
                        "--white-kind",
                        "deterministic_legal",
                        "--games",
                        "1",
                        "--max-plies",
                        "1",
                        "--out",
                        str(output_path),
                    ]
                )

            records = load_shogi_game_records_jsonl(output_path)

        self.assertEqual(records[0].black_actor.settings["evaluation_batch_size"], 8)

    def test_yaneuraou_requires_command(self) -> None:
        module = _load_script_module()

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            module.main(
                [
                    "--black-kind",
                    "yaneuraou",
                    "--white-kind",
                    "deterministic_legal",
                    "--out",
                    "games.jsonl",
                ]
            )

    def test_checkpoint_players_are_loaded_once(self) -> None:
        module = _load_script_module()
        calls: list[str] = []

        def fake_load_policy(checkpoint: str, **_kwargs: object) -> DeterministicLegalMovePolicy:
            calls.append(checkpoint)
            return DeterministicLegalMovePolicy()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "games.jsonl"

            with patch("shogi_arena_agent.player_cli._load_checkpoint_policy", side_effect=fake_load_policy):
                with contextlib.redirect_stdout(io.StringIO()):
                    module.main(
                        [
                            "--black-kind",
                            "checkpoint",
                            "--black-checkpoint",
                            "black.pt",
                            "--white-kind",
                            "checkpoint",
                            "--white-checkpoint",
                            "white.pt",
                            "--games",
                            "3",
                            "--max-plies",
                            "1",
                            "--out",
                            str(output_path),
                        ]
                    )

        self.assertEqual(calls, ["black.pt", "white.pt"])

    def test_deterministic_legal_writes_game_records(self) -> None:
        module = _load_script_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "games.jsonl"
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                module.main(
                    [
                        "--black-kind",
                        "deterministic_legal",
                        "--white-kind",
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
        self.assertEqual(records[0].black_actor.kind, "deterministic_legal")
        self.assertEqual(records[0].white_actor.kind, "deterministic_legal")
        self.assertEqual(summary["game_count"], 2)

    def test_yaneuraou_fixed_side_writes_game_records(self) -> None:
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
            stdout = io.StringIO()

            with patch("shogi_arena_agent.player_cli.UsiProcess", FakeUsiProcess), contextlib.redirect_stdout(stdout):
                module.main(
                    [
                        "--black-kind",
                        "yaneuraou",
                        "--black-yaneuraou-command",
                        "engine",
                        "--white-kind",
                        "yaneuraou",
                        "--white-yaneuraou-command",
                        "engine",
                        "--games",
                        "3",
                        "--max-plies",
                        "2",
                        "--out",
                        str(output_path),
                    ]
                )

            records = load_shogi_game_records_jsonl(output_path)
            summary = json.loads(stdout.getvalue())

        self.assertEqual(len(records), 3)
        self.assertEqual(records[0].black_actor.kind, "yaneuraou")
        self.assertEqual(records[0].white_actor.kind, "yaneuraou")
        self.assertEqual(summary["game_count"], 3)
        self.assertEqual(FakeUsiProcess.enter_count, 2)


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_shogi_games.py"
    spec = importlib.util.spec_from_file_location("generate_shogi_games", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
