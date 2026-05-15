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

from intrep.worlds.shogi.engine_analysis import load_shogi_engine_analysis_jsonl
from intrep.worlds.shogi.game_record import (
    ShogiActorSpec,
    ShogiGameRecord,
    ShogiTransitionRecord,
    write_shogi_game_records_jsonl,
)
from shogi_arena_agent.usi_process import UsiGoResult


class AnalyzeShogiPositionsScriptTest(unittest.TestCase):
    def test_writes_engine_analysis_jsonl(self) -> None:
        module = _load_script_module()

        class FakeUsiProcess:
            def __init__(self, **_kwargs: object) -> None:
                self.options: list[tuple[str, object]] = []
                self.position_commands: list[str] = []

            def __enter__(self) -> "FakeUsiProcess":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def setoption(self, *, name: str, value: object) -> None:
                self.options.append((name, value))

            def position(self, command: str) -> None:
                self.position_commands.append(command)

            def go(self) -> UsiGoResult:
                return UsiGoResult(
                    bestmove="7g7f",
                    info_lines=(
                        "info multipv 1 score cp 100 pv 7g7f",
                        "info multipv 2 score cp 0 pv 2g2f",
                    ),
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.jsonl"
            output_path = Path(temp_dir) / "analysis.jsonl"
            write_shogi_game_records_jsonl(input_path, [_record()])
            stdout = io.StringIO()

            with patch.object(module, "UsiProcess", FakeUsiProcess), contextlib.redirect_stdout(stdout):
                module.main(
                    [
                        "--input",
                        str(input_path),
                        "--out",
                        str(output_path),
                        "--usi-command",
                        "engine",
                        "--usi-name",
                        "analysis-engine",
                        "--usi-go-command",
                        "go nodes 30",
                        "--multipv",
                        "2",
                    ]
                )

            analyses = load_shogi_engine_analysis_jsonl(output_path)
            summary = json.loads(stdout.getvalue())

        self.assertEqual(summary["analysis_count"], 1)
        self.assertEqual(summary["position_count"], 1)
        self.assertEqual(analyses[0].engine.name, "analysis-engine")
        self.assertEqual(analyses[0].engine.settings["multipv"], 2)
        self.assertEqual(analyses[0].usi_info_lines[0], "info multipv 1 score cp 100 pv 7g7f")
        self.assertEqual(analyses[0].usi_info_lines[1], "info multipv 2 score cp 0 pv 2g2f")


def _record() -> ShogiGameRecord:
    actor = ShogiActorSpec(kind="checkpoint", name="model", settings={})
    return ShogiGameRecord(
        black_actor=actor,
        white_actor=actor,
        initial_position_sfen="lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        transitions=(
            ShogiTransitionRecord(
                ply=0,
                side="black",
                position_sfen="lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
                legal_moves=("7g7f", "2g2f"),
                action_usi="7g7f",
                next_position_sfen="lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2",
                reward=0.0,
                done=True,
            ),
        ),
        winner=None,
        end_reason="max_plies",
    )


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_shogi_positions_with_usi_engine.py"
    spec = importlib.util.spec_from_file_location("analyze_shogi_positions_with_usi_engine", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
