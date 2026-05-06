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


class GenerateShogiGamesScriptTest(unittest.TestCase):
    def test_self_play_writes_game_records(self) -> None:
        module = _load_script_module()
        original_load_policy = module._load_policy
        module._load_policy = lambda _checkpoint, _args: DeterministicLegalMovePolicy()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "games.jsonl"
                stdout = io.StringIO()

                with contextlib.redirect_stdout(stdout):
                    module.main(
                        [
                            "--checkpoint",
                            "black.pt",
                            "--white-checkpoint",
                            "white.pt",
                            "--matchup",
                            "checkpoint-self",
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
        finally:
            module._load_policy = original_load_policy

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].black_actor.settings["checkpoint"], "black.pt")
        self.assertEqual(records[0].white_actor.settings["checkpoint"], "white.pt")
        self.assertEqual(summary["game_count"], 1)

    def test_checkpoint_yaneuraou_requires_command(self) -> None:
        module = _load_script_module()

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            module.main(["--checkpoint", "model.pt", "--matchup", "checkpoint-yaneuraou", "--out", "games.jsonl"])

    def test_checkpoint_deterministic_legal_writes_game_records(self) -> None:
        module = _load_script_module()
        original_load_policy = module._load_policy
        module._load_policy = lambda _checkpoint, _args: DeterministicLegalMovePolicy()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "games.jsonl"
                stdout = io.StringIO()

                with contextlib.redirect_stdout(stdout):
                    module.main(
                        [
                            "--checkpoint",
                            "model.pt",
                            "--matchup",
                            "checkpoint-deterministic-legal",
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
        finally:
            module._load_policy = original_load_policy

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].black_actor.kind, "checkpoint")
        self.assertEqual(records[0].white_actor.kind, "deterministic_legal")
        self.assertEqual(records[1].black_actor.kind, "deterministic_legal")
        self.assertEqual(records[1].white_actor.kind, "checkpoint")
        self.assertEqual(summary["game_count"], 2)

    def test_self_play_loads_each_checkpoint_once(self) -> None:
        module = _load_script_module()
        calls: list[str] = []
        original_load_policy = module._load_policy

        def fake_load_policy(checkpoint: str, _args: object) -> DeterministicLegalMovePolicy:
            calls.append(checkpoint)
            return DeterministicLegalMovePolicy()

        module._load_policy = fake_load_policy
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "games.jsonl"

                with contextlib.redirect_stdout(io.StringIO()):
                    module.main(
                        [
                            "--checkpoint",
                            "black.pt",
                            "--white-checkpoint",
                            "white.pt",
                            "--matchup",
                            "checkpoint-self",
                            "--games",
                            "3",
                            "--max-plies",
                            "1",
                            "--out",
                            str(output_path),
                        ]
                    )
        finally:
            module._load_policy = original_load_policy

        self.assertEqual(calls, ["black.pt", "white.pt"])

    def test_yaneuraou_self_writes_game_records(self) -> None:
        module = _load_script_module()

        class FakeUsiProcess:
            def __init__(self, **_kwargs: object) -> None:
                self.moves = iter(("7g7f", "3c3d"))

            def __enter__(self) -> "FakeUsiProcess":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def position(self, _command: str) -> None:
                return None

            def go(self) -> str:
                return next(self.moves, "2g2f")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "games.jsonl"
            stdout = io.StringIO()

            with patch.object(module, "UsiProcess", FakeUsiProcess), contextlib.redirect_stdout(stdout):
                module.main(
                    [
                        "--matchup",
                        "yaneuraou-self",
                        "--yaneuraou",
                        "engine",
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
        self.assertEqual(records[0].black_actor.kind, "yaneuraou")
        self.assertEqual(records[0].white_actor.kind, "yaneuraou")
        self.assertEqual(summary["game_count"], 1)


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
