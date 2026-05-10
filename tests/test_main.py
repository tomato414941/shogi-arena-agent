import argparse
import unittest
from unittest.mock import patch

from shogi_arena_agent.__main__ import build_engine, parse_args
from shogi_arena_agent.mcts_policy import MctsPolicy
from shogi_arena_agent.usi import UsiEngine


class MainTest(unittest.TestCase):
    def test_parse_args_defaults_to_deterministic_legal(self) -> None:
        args = parse_args([])

        self.assertIsNone(args.checkpoint)
        self.assertEqual(args.checkpoint_policy, "direct")
        self.assertEqual(args.checkpoint_simulations, 16)
        self.assertEqual(args.checkpoint_evaluation_batch_size, 1)
        self.assertEqual(args.device, "cpu")

    def test_build_engine_defaults_to_usi_engine(self) -> None:
        engine = build_engine(argparse.Namespace(checkpoint=None, device="cpu"))

        self.assertIsInstance(engine, UsiEngine)
        self.assertEqual(engine.handle_line("go btime 0 wtime 0"), ["bestmove 1g1f"])

    def test_build_engine_supports_mcts_checkpoint_policy(self) -> None:
        args = parse_args(
            [
                "--checkpoint",
                "checkpoint.pt",
                "--checkpoint-policy",
                "mcts",
                "--checkpoint-simulations",
                "32",
                "--checkpoint-evaluation-batch-size",
                "8",
                "--device",
                "cuda",
            ]
        )

        with patch("shogi_arena_agent.model_policy.ShogiMoveChoiceCheckpointEvaluator.from_checkpoint") as from_checkpoint:
            from_checkpoint.return_value.evaluate_batch.return_value = [({"7g7f": 1.0}, 0.0)]
            engine = build_engine(args)

        self.assertIsInstance(engine.policy, MctsPolicy)
        self.assertEqual(engine.policy.config.simulation_count, 32)
        self.assertEqual(engine.policy.config.evaluation_batch_size, 8)
        from_checkpoint.assert_called_once_with("checkpoint.pt", device="cuda")


if __name__ == "__main__":
    unittest.main()
