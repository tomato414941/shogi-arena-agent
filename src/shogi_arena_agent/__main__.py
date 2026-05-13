from __future__ import annotations

import argparse
import sys

from shogi_arena_agent.usi import BOARD_BACKENDS, UsiEngine


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shogi-arena-agent as a USI engine.")
    parser.add_argument("--checkpoint", help="Path to a shogi move-choice checkpoint.")
    parser.add_argument("--move-selector", choices=("direct", "mcts"), default="direct")
    parser.add_argument("--mcts-simulations", type=int, default=16)
    parser.add_argument("--mcts-evaluation-batch-size", type=int, default=1)
    parser.add_argument("--mcts-move-time-limit-sec", type=float)
    parser.add_argument("--mcts-root-reuse", action="store_true")
    parser.add_argument("--board-backend", choices=BOARD_BACKENDS, default="python-shogi")
    parser.add_argument("--device", default="cpu", help="Torch device used with --checkpoint.")
    return parser.parse_args(argv)


def build_engine(args: argparse.Namespace) -> UsiEngine:
    if args.checkpoint is None:
        return UsiEngine()

    from shogi_arena_agent.mcts_move_selector import MctsConfig, MctsMoveSelector
    from shogi_arena_agent.model_policy import ShogiMoveChoiceCheckpointEvaluator, ShogiMoveChoiceCheckpointPolicy

    if args.move_selector == "direct":
        policy = ShogiMoveChoiceCheckpointPolicy.from_checkpoint(
            args.checkpoint,
            device=args.device,
            board_backend=args.board_backend,
        )
    else:
        evaluator = ShogiMoveChoiceCheckpointEvaluator.from_checkpoint(args.checkpoint, device=args.device)
        policy = MctsMoveSelector(
            evaluator=evaluator,
            config=MctsConfig(
                simulation_count=args.mcts_simulations,
                evaluation_batch_size=args.mcts_evaluation_batch_size,
                move_time_limit_sec=args.mcts_move_time_limit_sec,
                board_backend=args.board_backend,
                root_reuse=args.mcts_root_reuse,
            ),
        )
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
