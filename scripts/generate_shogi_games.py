from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from shogi_arena_agent.match_evaluation import evaluate_player_against_usi_engine
from shogi_arena_agent.mcts_policy import MctsConfig, MctsPolicy
from shogi_arena_agent.model_policy import ShogiMoveChoiceCheckpointEvaluator, ShogiMoveChoiceCheckpointPolicy
from shogi_arena_agent.shogi_game import ShogiActorSpec, ShogiGameRecord, play_shogi_game, save_shogi_game_records_jsonl
from shogi_arena_agent.usi import UsiEngine
from shogi_arena_agent.usi_process import UsiProcess


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play shogi games and write raw game logs.")
    parser.add_argument("--matchup", choices=("checkpoint-yaneuraou", "checkpoint-self", "yaneuraou-self"), default="checkpoint-yaneuraou")
    parser.add_argument("--checkpoint")
    parser.add_argument("--yaneuraou")
    parser.add_argument("--white-checkpoint", help="Checkpoint for white in self-play. Defaults to --checkpoint.")
    parser.add_argument("--out", required=True, help="Path to write one ShogiGameRecord JSON object per line.")
    parser.add_argument("--policy", choices=("direct", "mcts"), default="mcts")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--engine-go-command", default="go nodes 1")
    parser.add_argument("--read-timeout-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)

    if args.matchup == "checkpoint-self":
        if not args.checkpoint:
            parser.error("--checkpoint is required when --matchup checkpoint-self")
        records = _play_self_games(args)
        save_shogi_game_records_jsonl(records, Path(args.out))
        print(json.dumps(_records_summary(records), indent=2))
        return

    if args.matchup == "yaneuraou-self":
        if not args.yaneuraou:
            parser.error("--yaneuraou is required when --matchup yaneuraou-self")
        records = _play_yaneuraou_self_games(args)
        save_shogi_game_records_jsonl(records, Path(args.out))
        print(json.dumps(_records_summary(records), indent=2))
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required when --matchup checkpoint-yaneuraou")
    if not args.yaneuraou:
        parser.error("--yaneuraou is required when --matchup checkpoint-yaneuraou")

    player = UsiEngine(name=f"checkpoint-{args.policy}", policy=_load_policy(args.checkpoint, args))
    evaluation = evaluate_player_against_usi_engine(
        player,
        [args.yaneuraou],
        game_count=args.games,
        max_plies=args.max_plies,
        engine_go_command=args.engine_go_command,
        read_timeout_seconds=args.read_timeout_seconds,
        player_actor=ShogiActorSpec(
            kind="checkpoint",
            name=f"checkpoint-{args.policy}",
            settings=_checkpoint_settings(args.checkpoint, args),
        ),
        engine_actor=ShogiActorSpec(
            kind="yaneuraou",
            name="yaneuraou",
            settings={
                "command": args.yaneuraou,
                "go_command": args.engine_go_command,
            },
        ),
    )
    save_shogi_game_records_jsonl(evaluation.results, Path(args.out))
    print(json.dumps(_evaluation_summary(evaluation), indent=2))


def _play_self_games(args: argparse.Namespace) -> tuple[ShogiGameRecord, ...]:
    white_checkpoint = args.white_checkpoint or args.checkpoint
    black_policy = _load_policy(args.checkpoint, args)
    white_policy = _load_policy(white_checkpoint, args)
    records: list[ShogiGameRecord] = []
    black_spec = ShogiActorSpec(
        kind="checkpoint",
        name=f"checkpoint-{args.policy}-black",
        settings=_checkpoint_settings(args.checkpoint, args),
    )
    white_spec = ShogiActorSpec(
        kind="checkpoint",
        name=f"checkpoint-{args.policy}-white",
        settings=_checkpoint_settings(white_checkpoint, args),
    )
    for _game_index in range(args.games):
        records.append(
            play_shogi_game(
                black=UsiEngine(name=black_spec.name, policy=black_policy),
                white=UsiEngine(name=white_spec.name, policy=white_policy),
                black_actor=black_spec,
                white_actor=white_spec,
                max_plies=args.max_plies,
            )
        )
    return tuple(records)


def _play_yaneuraou_self_games(args: argparse.Namespace) -> tuple[ShogiGameRecord, ...]:
    records: list[ShogiGameRecord] = []
    black_spec = _yaneuraou_actor_spec(args, name="yaneuraou-black")
    white_spec = _yaneuraou_actor_spec(args, name="yaneuraou-white")
    for _game_index in range(args.games):
        with UsiProcess(
            command=[args.yaneuraou],
            go_command=args.engine_go_command,
            read_timeout_seconds=args.read_timeout_seconds,
        ) as black_engine, UsiProcess(
            command=[args.yaneuraou],
            go_command=args.engine_go_command,
            read_timeout_seconds=args.read_timeout_seconds,
        ) as white_engine:
            records.append(
                play_shogi_game(
                    black=black_engine,
                    white=white_engine,
                    black_actor=black_spec,
                    white_actor=white_spec,
                    max_plies=args.max_plies,
                )
            )
    return tuple(records)


def _load_policy(checkpoint: str, args: argparse.Namespace) -> Any:
    if args.policy == "direct":
        return ShogiMoveChoiceCheckpointPolicy.from_checkpoint(checkpoint)
    evaluator = ShogiMoveChoiceCheckpointEvaluator.from_checkpoint(checkpoint)
    return MctsPolicy(evaluator=evaluator, config=MctsConfig(simulation_count=args.simulations))


def _checkpoint_settings(checkpoint: str, args: argparse.Namespace) -> dict[str, str | int | None]:
    return {
        "checkpoint": checkpoint,
        "policy": args.policy,
        "simulations": args.simulations if args.policy == "mcts" else None,
    }


def _yaneuraou_actor_spec(args: argparse.Namespace, *, name: str) -> ShogiActorSpec:
    return ShogiActorSpec(
        kind="yaneuraou",
        name=name,
        settings={
            "command": args.yaneuraou,
            "go_command": args.engine_go_command,
            "read_timeout_seconds": args.read_timeout_seconds,
        },
    )


def _evaluation_summary(evaluation: Any) -> dict[str, Any]:
    data = asdict(evaluation)
    data.pop("results")
    return data


def _records_summary(records: tuple[ShogiGameRecord, ...]) -> dict[str, Any]:
    end_reasons = Counter(record.end_reason for record in records)
    return {
        "game_count": len(records),
        "end_reasons": dict(end_reasons),
        "average_plies": sum(len(record.transitions) for record in records) / len(records) if records else 0.0,
        "black_wins": sum(1 for record in records if record.winner == "black"),
        "white_wins": sum(1 for record in records if record.winner == "white"),
        "draws": sum(1 for record in records if record.winner is None),
    }


if __name__ == "__main__":
    main()
