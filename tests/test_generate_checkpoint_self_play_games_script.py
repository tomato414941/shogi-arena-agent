from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shogi_arena_agent.shogi_game import load_shogi_game_records_jsonl


class GenerateCheckpointSelfPlayGamesScriptTest(unittest.TestCase):
    def test_writes_checkpoint_self_play_records_without_player_cli(self) -> None:
        module = _load_script_module()
        batch_sizes: list[int] = []
        loaded_checkpoints: list[str] = []

        class FakeEvaluator:
            @classmethod
            def from_checkpoint(cls, checkpoint: str, **_kwargs: object) -> "FakeEvaluator":
                loaded_checkpoints.append(checkpoint)
                self = cls()
                self.checkpoint = checkpoint
                return self

            def evaluate_positions(self, requests):
                batch_sizes.append(len(requests))
                return [({move: 1.0 for move in legal_moves}, 0.0) for _sfen, legal_moves in requests]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "games.jsonl"
            stdout = io.StringIO()

            with (
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(io.StringIO()),
                patch.object(module, "ShogiMoveChoiceCheckpointEvaluator", FakeEvaluator),
            ):
                module.main(
                    [
                        "--checkpoint",
                        "model.pt",
                        "--checkpoint-id",
                        "model-entry",
                        "--games",
                        "4",
                        "--self-play-worker-threads",
                        "2",
                        "--concurrent-games-per-process",
                        "4",
                        "--mcts-simulations",
                        "2",
                        "--mcts-nn-leaf-eval-batch-limit",
                        "8",
                        "--central-evaluator-batch-size-limit",
                        "16",
                        "--central-evaluator-flush-timeout-sec",
                        "0.05",
                        "--max-plies",
                        "2",
                        "--out",
                        str(output_path),
                    ]
                )

            records = load_shogi_game_records_jsonl(output_path)
            summary = json.loads(stdout.getvalue())

        self.assertEqual(len(records), 4)
        self.assertEqual(loaded_checkpoints, ["model.pt"])
        self.assertIn(4, batch_sizes)
        self.assertEqual(records[0].black_actor.kind, "checkpoint_self_play")
        self.assertEqual(records[0].white_actor.kind, "checkpoint_self_play")
        self.assertEqual(records[0].black_actor.settings["checkpoint"], "model.pt")
        self.assertEqual(records[0].black_actor.settings["checkpoint_id"], "model-entry")
        self.assertEqual(records[0].black_actor.settings["move_selection_profile"], "visit-sampling")
        self.assertEqual(records[0].black_actor.settings["self_play_worker_threads"], 2)
        self.assertEqual(records[0].black_actor.settings["mcts_simulations_per_move"], 2)
        self.assertEqual(records[0].black_actor.settings["nn_leaf_eval_batch_limit"], 8)
        self.assertEqual(records[0].black_actor.settings["central_evaluator_batch_size_limit"], 16)
        self.assertEqual(records[0].black_actor.settings["central_evaluator_flush_timeout_sec"], 0.05)
        self.assertIsNotNone(records[0].transitions[0].decision_telemetry)
        assert records[0].transitions[0].decision_telemetry is not None
        self.assertIsNotNone(records[0].transitions[0].decision_telemetry.search_evidence)
        assert records[0].transitions[0].decision_telemetry.search_evidence is not None
        self.assertIn("mcts_root_child_visit_counts", records[0].transitions[0].decision_telemetry.search_evidence)
        self.assertIn("mcts_root_mean_value", records[0].transitions[0].decision_telemetry.search_evidence)
        self.assertEqual(summary["game_count"], 4)
        self.assertIn("multi_position_search_performance", summary)
        self.assertIn("central_evaluator_performance", summary)
        self.assertGreater(summary["central_evaluator_performance"]["model_call_count"], 0)
        self.assertGreaterEqual(summary["central_evaluator_performance"]["actual_nn_leaf_eval_batch_size_max"], 4)


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_checkpoint_self_play_games.py"
    spec = importlib.util.spec_from_file_location("generate_checkpoint_self_play_games", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
