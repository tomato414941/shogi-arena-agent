from __future__ import annotations

import argparse
import sys

from shogi_arena_agent.usi import UsiEngine


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shogi-arena-agent as a USI engine.")
    parser.add_argument("--checkpoint", help="Path to a shogi move-choice checkpoint.")
    parser.add_argument("--device", default="cpu", help="Torch device used with --checkpoint.")
    return parser.parse_args(argv)


def build_engine(args: argparse.Namespace) -> UsiEngine:
    if args.checkpoint is None:
        return UsiEngine()

    from shogi_arena_agent.model_policy import ShogiMoveChoiceCheckpointPolicy

    policy = ShogiMoveChoiceCheckpointPolicy.from_checkpoint(args.checkpoint, device=args.device)
    return UsiEngine(policy=policy)


def main(argv: list[str] | None = None) -> None:
    engine = build_engine(parse_args(argv))
    for line in sys.stdin:
        if line.strip() == "quit":
            break
        for response in engine.handle_line(line):
            print(response, flush=True)


if __name__ == "__main__":
    main()
