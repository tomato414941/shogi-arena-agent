import argparse
import unittest
from unittest.mock import patch

from shogi_arena_agent.__main__ import build_engine, parse_args
from shogi_arena_agent.model_policy import ShogiMoveChoiceCheckpointPolicy
from shogi_arena_agent.mcts_move_selector import MctsMoveSelector
from shogi_arena_agent.usi import UsiEngine


class MainTest(unittest.TestCase):
    def test_parse_args_defaults_to_deterministic_legal(self) -> None:
        args = parse_args([])

        self.assertIsNone(args.checkpoint)
        self.assertEqual(args.move_selector, "direct")
        self.assertEqual(args.mcts_simulations, 16)
        self.assertEqual(args.mcts_evaluation_batch_size, 1)
        self.assertIsNone(args.mcts_move_time_limit_sec)
        self.assertFalse(args.mcts_root_reuse)
        self.assertEqual(args.board_backend, "python-shogi")
        self.assertEqual(args.device, "cpu")

    def test_build_engine_defaults_to_usi_engine(self) -> None:
        engine = build_engine(argparse.Namespace(checkpoint=None, device="cpu"))

        self.assertIsInstance(engine, UsiEngine)
        self.assertEqual(engine.handle_line("go btime 0 wtime 0"), ["bestmove 1g1f"])

    def test_build_engine_supports_mcts_move_selector(self) -> None:
        args = parse_args(
            [
                "--checkpoint",
                "checkpoint.pt",
                "--move-selector",
                "mcts",
                "--mcts-simulations",
                "32",
                "--mcts-evaluation-batch-size",
                "8",
                "--mcts-move-time-limit-sec",
                "9.0",
                "--mcts-root-reuse",
                "--board-backend",
                "cshogi",
                "--device",
                "cuda",
            ]
        )

        with patch("shogi_arena_agent.model_policy.ShogiMoveChoiceCheckpointEvaluator.from_checkpoint") as from_checkpoint:
            from_checkpoint.return_value.evaluate_batch.return_value = [({"7g7f": 1.0}, 0.0)]
            engine = build_engine(args)

        self.assertIsInstance(engine.policy, MctsMoveSelector)
        self.assertEqual(engine.policy.config.simulation_count, 32)
        self.assertEqual(engine.policy.config.evaluation_batch_size, 8)
        self.assertEqual(engine.policy.config.move_time_limit_sec, 9.0)
        self.assertTrue(engine.policy.config.root_reuse)
        self.assertEqual(engine.policy.config.board_backend, "cshogi")
        from_checkpoint.assert_called_once_with("checkpoint.pt", device="cuda")

    def test_build_engine_passes_board_backend_to_direct_move_selector(self) -> None:
        args = parse_args(
            [
                "--checkpoint",
                "checkpoint.pt",
                "--board-backend",
                "cshogi",
                "--device",
                "cuda",
            ]
        )

        with patch("shogi_arena_agent.model_policy.ShogiMoveChoiceCheckpointEvaluator.from_checkpoint") as from_checkpoint:
            policy = build_engine(args).policy

        self.assertIsInstance(policy, ShogiMoveChoiceCheckpointPolicy)
        self.assertEqual(policy.board_backend, "cshogi")
        from_checkpoint.assert_called_once_with("checkpoint.pt", device="cuda")


if __name__ == "__main__":
    unittest.main()
