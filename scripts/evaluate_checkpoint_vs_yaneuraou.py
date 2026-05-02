from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from shogi_arena_agent.match_evaluation import evaluate_player_against_usi_engine
from shogi_arena_agent.mcts_policy import MctsConfig, MctsPolicy
from shogi_arena_agent.model_policy import ShogiMoveChoiceCheckpointEvaluator, ShogiMoveChoiceCheckpointPolicy
from shogi_arena_agent.usi import UsiEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a shogi checkpoint against YaneuraOu.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--yaneuraou", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy", choices=("direct", "mcts"), default="mcts")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--engine-go-command", default="go nodes 1")
    parser.add_argument("--read-timeout-seconds", type=float, default=10.0)
    args = parser.parse_args()

    player = UsiEngine(name=f"checkpoint-{args.policy}", policy=_load_policy(args))
    evaluation = evaluate_player_against_usi_engine(
        player,
        [args.yaneuraou],
        game_count=args.games,
        max_plies=args.max_plies,
        engine_go_command=args.engine_go_command,
        read_timeout_seconds=args.read_timeout_seconds,
    )
    output = {
        "checkpoint": args.checkpoint,
        "yaneuraou": args.yaneuraou,
        "policy": args.policy,
        "games": args.games,
        "max_plies": args.max_plies,
        "simulations": args.simulations if args.policy == "mcts" else None,
        "engine_go_command": args.engine_go_command,
        "evaluation": asdict(evaluation),
    }
    _write_json(output, Path(args.out))


def _load_policy(args: argparse.Namespace) -> Any:
    if args.policy == "direct":
        return ShogiMoveChoiceCheckpointPolicy.from_checkpoint(args.checkpoint)
    evaluator = ShogiMoveChoiceCheckpointEvaluator.from_checkpoint(args.checkpoint)
    return MctsPolicy(evaluator=evaluator, config=MctsConfig(simulation_count=args.simulations))


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
