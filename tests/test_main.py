import argparse
import unittest

from shogi_arena_agent.__main__ import build_engine, parse_args
from shogi_arena_agent.usi import UsiEngine


class MainTest(unittest.TestCase):
    def test_parse_args_defaults_to_baseline(self) -> None:
        args = parse_args([])

        self.assertIsNone(args.checkpoint)
        self.assertEqual(args.device, "cpu")

    def test_build_engine_defaults_to_usi_engine(self) -> None:
        engine = build_engine(argparse.Namespace(checkpoint=None, device="cpu"))

        self.assertIsInstance(engine, UsiEngine)
        self.assertEqual(engine.handle_line("go btime 0 wtime 0"), ["bestmove 1g1f"])


if __name__ == "__main__":
    unittest.main()
