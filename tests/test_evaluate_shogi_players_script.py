from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

from shogi_arena_agent.shogi_game import load_shogi_game_records_jsonl


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
