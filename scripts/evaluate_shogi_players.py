from __future__ import annotations

import argparse
import json
from contextlib import ExitStack, nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

from shogi_arena_agent.match_evaluation import summarize_match_results
from shogi_arena_agent.player_cli import BuiltPlayer, add_player_arguments, build_static_player, player_context, validate_player_arguments
from shogi_arena_agent.shogi_game import ShogiGameRecord, play_shogi_game, save_shogi_game_records_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate two shogi players with alternating sides.")
    add_player_arguments(parser, "player")
    add_player_arguments(parser, "opponent")
    parser.add_argument("--out", required=True, help="Path to write one ShogiGameRecord JSON object per line.")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=80)
    args = parser.parse_args(argv)

    validate_player_arguments(parser, args, "player")
    validate_player_arguments(parser, args, "opponent")
    if args.games <= 0:
        parser.error("--games must be positive")

    results, player_sides = _evaluate(args)
    evaluation = summarize_match_results(results, player_sides)
    save_shogi_game_records_jsonl(evaluation.results, Path(args.out))
    print(json.dumps(_evaluation_summary(evaluation), indent=2))


def _evaluate(args: argparse.Namespace) -> tuple[list[ShogiGameRecord], list[str]]:
    results: list[ShogiGameRecord] = []
    player_sides: list[str] = []
    player_static = build_static_player(args, "player", name="player")
    opponent_static = build_static_player(args, "opponent", name="opponent")
    with ExitStack() as stack:
        player = stack.enter_context(_player_context(args, "player", name="player", static_player=player_static))
        opponent = stack.enter_context(_player_context(args, "opponent", name="opponent", static_player=opponent_static))
        for game_index in range(args.games):
            if game_index % 2 == 0:
                results.append(
                    play_shogi_game(
                        black=player.player,
                        white=opponent.player,
                        black_actor=player.actor,
                        white_actor=opponent.actor,
                        max_plies=args.max_plies,
                    )
                )
                player_sides.append("black")
            else:
                results.append(
                    play_shogi_game(
                        black=opponent.player,
                        white=player.player,
                        black_actor=opponent.actor,
                        white_actor=player.actor,
                        max_plies=args.max_plies,
                    )
                )
                player_sides.append("white")
    return results, player_sides


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


def _evaluation_summary(evaluation: Any) -> dict[str, Any]:
    data = asdict(evaluation)
    data.pop("results")
    return data


if __name__ == "__main__":
    main()
