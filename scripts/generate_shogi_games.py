from __future__ import annotations

import argparse
import json
from collections import Counter
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any

from shogi_arena_agent.player_cli import BuiltPlayer, add_player_arguments, build_static_player, player_context, validate_player_arguments
from shogi_arena_agent.shogi_game import ShogiGameRecord, play_shogi_game, save_shogi_game_records_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play fixed-side shogi games and write raw game logs.")
    add_player_arguments(parser, "black")
    add_player_arguments(parser, "white")
    parser.add_argument("--out", required=True, help="Path to write one ShogiGameRecord JSON object per line.")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=80)
    args = parser.parse_args(argv)

    validate_player_arguments(parser, args, "black")
    validate_player_arguments(parser, args, "white")
    if args.games <= 0:
        parser.error("--games must be positive")

    records = _play_games(args)
    save_shogi_game_records_jsonl(records, Path(args.out))
    print(json.dumps(_records_summary(records), indent=2))


def _play_games(args: argparse.Namespace) -> tuple[ShogiGameRecord, ...]:
    records: list[ShogiGameRecord] = []
    black_static = build_static_player(args, "black", name="black")
    white_static = build_static_player(args, "white", name="white")
    with ExitStack() as stack:
        black = stack.enter_context(_player_context(args, "black", name="black", static_player=black_static))
        white = stack.enter_context(_player_context(args, "white", name="white", static_player=white_static))
        for _game_index in range(args.games):
            records.append(
                play_shogi_game(
                    black=black.player,
                    white=white.player,
                    black_actor=black.actor,
                    white_actor=white.actor,
                    max_plies=args.max_plies,
                )
            )
    return tuple(records)


def _player_context(
    args: argparse.Namespace,
    prefix: str,
    *,
    name: str,
    static_player: BuiltPlayer | None,
):
    if static_player is not None:
        return nullcontext(static_player)
    return player_context(args, prefix, name=name)


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
