from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

from shogi_arena_agent.baseline_policy import DeterministicLegalMovePolicy
from shogi_arena_agent.shogi_game import load_shogi_game_records_jsonl


class GenerateCheckpointGamesScriptTest(unittest.TestCase):
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
                            "--opponent",
                            "self",
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
        self.assertEqual(records[0].black_player.settings["checkpoint"], "black.pt")
        self.assertEqual(records[0].white_player.settings["checkpoint"], "white.pt")
        self.assertEqual(summary["game_count"], 1)

    def test_yaneuraou_opponent_requires_command(self) -> None:
        module = _load_script_module()

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            module.main(["--checkpoint", "model.pt", "--opponent", "yaneuraou", "--out", "games.jsonl"])


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_checkpoint_games.py"
    spec = importlib.util.spec_from_file_location("generate_checkpoint_games", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
