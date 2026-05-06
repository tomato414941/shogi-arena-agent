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
    load_shogi_game_records_jsonl,
    save_shogi_game_records_jsonl,
)
from shogi_arena_agent.usi_process import UsiGoResult


class AnnotateShogiRecordsScriptTest(unittest.TestCase):
    def test_annotates_records_with_raw_multipv_info(self) -> None:
        module = _load_script_module()

        class FakeUsiProcess:
            def __init__(self, **_kwargs: object) -> None:
                self.options: list[tuple[str, object]] = []

            def __enter__(self) -> "FakeUsiProcess":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def setoption(self, *, name: str, value: object) -> None:
                self.options.append((name, value))

            def position(self, _command: str) -> None:
                return None

            def go(self) -> UsiGoResult:
                return UsiGoResult(
                    bestmove="7g7f",
                    info_lines=(
                        "info multipv 1 score cp 100 pv 7g7f",
                        "info multipv 2 score cp 0 pv 2g2f",
                        "info multipv 3 score cp -100 pv 1a1b",
                    ),
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.jsonl"
            output_path = Path(temp_dir) / "output.jsonl"
            save_shogi_game_records_jsonl((_record(),), input_path)
            stdout = io.StringIO()

            with patch.object(module, "UsiProcess", FakeUsiProcess), contextlib.redirect_stdout(stdout):
                module.main(
                    [
                        "--input",
                        str(input_path),
                        "--out",
                        str(output_path),
                        "--yaneuraou",
                        "engine",
                        "--engine-go-command",
                        "go nodes 30",
                        "--multipv",
                        "3",
                    ]
                )

            records = load_shogi_game_records_jsonl(output_path)
            summary = json.loads(stdout.getvalue())

        transition = records[0].transitions[0]
        self.assertEqual(summary["annotated_count"], 1)
        self.assertEqual(summary["annotated_ratio"], 1.0)
        self.assertEqual(transition.usi_info_lines[0], "info multipv 1 score cp 100 pv 7g7f")
        self.assertEqual(transition.usi_info_lines[1], "info multipv 2 score cp 0 pv 2g2f")


def _record() -> ShogiGameRecord:
    return ShogiGameRecord(
        black_actor=ShogiActorSpec(kind="checkpoint", name="model", settings={}),
        white_actor=ShogiActorSpec(kind="yaneuraou", name="teacher", settings={}),
        initial_position_sfen="lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        transitions=(
            module_transition(),
        ),
        end_reason="max_plies",
        winner=None,
    )


def module_transition():
    from shogi_arena_agent.shogi_game import ShogiTransitionRecord

    return ShogiTransitionRecord(
        ply=0,
        side="black",
        position_sfen="lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        legal_moves=("7g7f", "2g2f"),
        action_usi="7g7f",
        next_position_sfen="lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2",
        reward=0.0,
        done=True,
    )


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "annotate_shogi_records_with_yaneuraou_multipv.py"
    spec = importlib.util.spec_from_file_location("annotate_shogi_records_with_yaneuraou_multipv", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
