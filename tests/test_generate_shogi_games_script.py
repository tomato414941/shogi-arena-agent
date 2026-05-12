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
        self.assertEqual(records[0].white_actor.settings["evaluation_batch_size"], 1)
        self.assertIsNone(records[0].black_actor.settings["move_time_limit_sec"])
        self.assertIsNone(records[0].white_actor.settings["move_time_limit_sec"])
        self.assertEqual(summary["game_count"], 1)

    def test_records_checkpoint_mcts_settings_for_both_sides(self) -> None:
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
                        "--black-checkpoint-move-time-limit-sec",
                        "8.5",
                        "--white-kind",
                        "checkpoint",
                        "--white-checkpoint",
                        "white.pt",
                        "--white-checkpoint-evaluation-batch-size",
                        "16",
                        "--white-checkpoint-move-time-limit-sec",
                        "9.0",
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
        self.assertEqual(records[0].white_actor.settings["evaluation_batch_size"], 16)
        self.assertEqual(records[0].black_actor.settings["move_time_limit_sec"], 8.5)
        self.assertEqual(records[0].white_actor.settings["move_time_limit_sec"], 9.0)

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

    def test_parallel_checkpoint_mcts_batches_games(self) -> None:
        module = _load_script_module()
        batch_sizes: list[int] = []

        class FakeEvaluator:
            @classmethod
            def from_checkpoint(cls, *_args: object, **_kwargs: object) -> "FakeEvaluator":
                return cls()

            def evaluate_batch(self, requests):
                batch_sizes.append(len(requests))
                return [({move: 1.0 for move in legal_moves}, 0.0) for _board, legal_moves in requests]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "games.jsonl"
            stdout = io.StringIO()

            with patch.object(module, "ShogiMoveChoiceCheckpointEvaluator", FakeEvaluator), contextlib.redirect_stdout(stdout):
                module.main(
                    [
                        "--black-kind",
                        "checkpoint",
                        "--black-checkpoint",
                        "black.pt",
                        "--black-checkpoint-simulations",
                        "2",
                        "--black-checkpoint-evaluation-batch-size",
                        "8",
                        "--white-kind",
                        "checkpoint",
                        "--white-checkpoint",
                        "white.pt",
                        "--white-checkpoint-simulations",
                        "2",
                        "--white-checkpoint-evaluation-batch-size",
                        "8",
                        "--games",
                        "4",
                        "--parallel-games",
                        "4",
                        "--max-plies",
                        "2",
                        "--out",
                        str(output_path),
                    ]
                )

            records = load_shogi_game_records_jsonl(output_path)
            summary = json.loads(stdout.getvalue())

        self.assertEqual(len(records), 4)
        self.assertIn(4, batch_sizes)
        self.assertEqual(records[0].black_actor.settings["parallel_games"], 4)
        self.assertTrue(
            any(line.startswith("info string intrep_batch_performance ") for line in records[0].transitions[0].decision_usi_info_lines)
        )
        self.assertIn("generation_wall_time_sec", summary)
        self.assertIn("inference_performance", summary)
        self.assertIn("batch_performance", summary)
        self.assertIn("phase_wall_time_sec_total", summary["batch_performance"])

    def test_parallel_checkpoint_mcts_can_print_progress(self) -> None:
        module = _load_script_module()

        class FakeEvaluator:
            @classmethod
            def from_checkpoint(cls, *_args: object, **_kwargs: object) -> "FakeEvaluator":
                return cls()

            def evaluate_batch(self, requests):
                return [({move: 1.0 for move in legal_moves}, 0.0) for _board, legal_moves in requests]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "games.jsonl"
            stderr = io.StringIO()

            with (
                patch.object(module, "ShogiMoveChoiceCheckpointEvaluator", FakeEvaluator),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(stderr),
            ):
                module.main(
                    [
                        "--black-kind",
                        "checkpoint",
                        "--black-checkpoint",
                        "black.pt",
                        "--black-checkpoint-simulations",
                        "1",
                        "--white-kind",
                        "checkpoint",
                        "--white-checkpoint",
                        "white.pt",
                        "--white-checkpoint-simulations",
                        "1",
                        "--games",
                        "2",
                        "--parallel-games",
                        "2",
                        "--progress-every-plies",
                        "1",
                        "--max-plies",
                        "1",
                        "--out",
                        str(output_path),
                    ]
                )

        self.assertIn("progress ", stderr.getvalue())

    def test_parallel_checkpoint_mcts_accepts_cshogi_backend(self) -> None:
        module = _load_script_module()

        class FakeEvaluator:
            @classmethod
            def from_checkpoint(cls, *_args: object, **_kwargs: object) -> "FakeEvaluator":
                return cls()

            def evaluate_batch(self, requests):
                return [({move: 1.0 for move in legal_moves}, 0.0) for _board, legal_moves in requests]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "games.jsonl"

            with patch.object(module, "ShogiMoveChoiceCheckpointEvaluator", FakeEvaluator), contextlib.redirect_stdout(io.StringIO()):
                module.main(
                    [
                        "--black-kind",
                        "checkpoint",
                        "--black-checkpoint",
                        "black.pt",
                        "--black-checkpoint-simulations",
                        "2",
                        "--white-kind",
                        "checkpoint",
                        "--white-checkpoint",
                        "white.pt",
                        "--white-checkpoint-simulations",
                        "2",
                        "--games",
                        "2",
                        "--parallel-games",
                        "2",
                        "--board-backend",
                        "cshogi",
                        "--max-plies",
                        "1",
                        "--out",
                        str(output_path),
                    ]
                )

            records = load_shogi_game_records_jsonl(output_path)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].black_actor.settings["board_backend"], "cshogi")

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
