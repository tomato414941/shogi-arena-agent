from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

from shogi_arena_agent.shogi_game import (
    ShogiActorSpec,
    ShogiDecisionTelemetry,
    ShogiGameRecord,
    ShogiTransitionRecord,
    save_shogi_game_records_jsonl,
)


class AnalyzeShogiTeacherQualityScriptTest(unittest.TestCase):
    def test_summarizes_mcts_teacher_quality(self) -> None:
        module = _load_script_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "games.jsonl"
            save_shogi_game_records_jsonl([_record()], input_path)
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                module.main(["--input", str(input_path)])

            summary = json.loads(stdout.getvalue())

        self.assertEqual(summary["schema_version"], "shogi_arena_agent.teacher_quality.v1")
        self.assertEqual(summary["game_count"], 1)
        self.assertEqual(summary["transition_count"], 2)
        self.assertEqual(summary["mcts_transition_count"], 2)
        self.assertEqual(summary["prior_transition_count"], 1)
        self.assertEqual(summary["end_reasons"], {"max_plies": 1})
        self.assertEqual(summary["average_plies"], 2.0)
        self.assertEqual(summary["chosen_visit_top1_rate"], 0.5)
        self.assertEqual(summary["policy_mcts_top1_disagreement_rate"], 1.0)
        self.assertEqual(summary["visit_top1_share"]["count"], 2)
        self.assertAlmostEqual(summary["visit_top1_share"]["mean"], 0.75)
        self.assertEqual(summary["root_mean_value"]["count"], 2)


def _record() -> ShogiGameRecord:
    actor = ShogiActorSpec(kind="checkpoint", name="model", settings={})
    return ShogiGameRecord(
        black_actor=actor,
        white_actor=actor,
        initial_position_sfen="startpos",
        transitions=(
            ShogiTransitionRecord(
                ply=0,
                side="black",
                position_sfen="startpos",
                legal_moves=("7g7f", "2g2f"),
                action_usi="7g7f",
                next_position_sfen="after 7g7f",
                reward=0.0,
                done=False,
                decision_telemetry=ShogiDecisionTelemetry(
                    search_evidence={
                        "mcts_root_child_priors": {"7g7f": 0.2, "2g2f": 0.8},
                        "mcts_root_child_visit_counts": {"7g7f": 6, "2g2f": 2},
                        "mcts_root_mean_value": 0.25,
                    }
                ),
            ),
            ShogiTransitionRecord(
                ply=1,
                side="white",
                position_sfen="after 7g7f",
                legal_moves=("3c3d", "8c8d"),
                action_usi="8c8d",
                next_position_sfen="after 8c8d",
                reward=0.0,
                done=True,
                decision_telemetry=ShogiDecisionTelemetry(
                    search_evidence={
                        "mcts_root_child_visit_counts": {"3c3d": 3, "8c8d": 1},
                        "mcts_root_mean_value": -0.5,
                    }
                ),
            ),
        ),
        end_reason="max_plies",
        winner=None,
    )


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_shogi_teacher_quality.py"
    spec = importlib.util.spec_from_file_location("analyze_shogi_teacher_quality", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
