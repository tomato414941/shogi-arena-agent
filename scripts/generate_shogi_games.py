from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from shogi_arena_agent.model_policy import ShogiMoveChoiceCheckpointEvaluator
from shogi_arena_agent.player_cli import add_player_arguments, validate_player_arguments
from shogi_arena_agent.shogi_game import save_shogi_game_records_jsonl
from shogi_arena_agent.shogi_generation import (
    ShogiGenerationConfig,
    ShogiPlayerGenerationConfig,
    generate_shogi_games,
    records_summary,
)
from shogi_arena_agent.usi import BOARD_BACKENDS

STANDARD_MAX_PLIES = 320
DEFAULT_MAX_PLIES = 320


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play fixed-side shogi games and write raw game logs.")
    add_player_arguments(parser, "black")
    add_player_arguments(parser, "white")
    parser.add_argument("--out", required=True, help="Path to write one ShogiGameRecord JSON object per line.")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--parallel-games", type=int, default=1)
    parser.add_argument("--progress-every-plies", type=int, default=0)
    # Computer-shogi self-play should not end as a short artificial draw; use
    # the WCSC-style 320-ply cap as the default and warn on shorter overrides.
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    parser.add_argument("--board-backend", choices=BOARD_BACKENDS, default="python-shogi")
    args = parser.parse_args(argv)

    validate_player_arguments(parser, args, "black")
    validate_player_arguments(parser, args, "white")
    if args.games <= 0:
        parser.error("--games must be positive")
    if args.parallel_games <= 0:
        parser.error("--parallel-games must be positive")
    if args.progress_every_plies < 0:
        parser.error("--progress-every-plies must be non-negative")
    _warn_short_max_plies(args.max_plies)

    started_at = perf_counter()
    records = generate_shogi_games(
        _generation_config_from_args(args),
        checkpoint_evaluator_cls=ShogiMoveChoiceCheckpointEvaluator,
    )
    save_shogi_game_records_jsonl(records, Path(args.out))
    print(json.dumps(records_summary(records, wall_time_sec=perf_counter() - started_at), indent=2))


def _generation_config_from_args(args: argparse.Namespace) -> ShogiGenerationConfig:
    return ShogiGenerationConfig(
        black=_player_config_from_args(args, "black"),
        white=_player_config_from_args(args, "white"),
        games=args.games,
        parallel_games=args.parallel_games,
        max_plies=args.max_plies,
        board_backend=args.board_backend,
        progress_every_plies=args.progress_every_plies,
    )


def _player_config_from_args(args: argparse.Namespace, prefix: str) -> ShogiPlayerGenerationConfig:
    return ShogiPlayerGenerationConfig(
        kind=getattr(args, f"{prefix}_kind"),
        checkpoint=getattr(args, f"{prefix}_checkpoint"),
        checkpoint_profile=getattr(args, f"{prefix}_checkpoint_profile"),
        checkpoint_policy=getattr(args, f"{prefix}_checkpoint_policy"),
        checkpoint_simulations=getattr(args, f"{prefix}_checkpoint_simulations"),
        checkpoint_evaluation_batch_size=getattr(args, f"{prefix}_checkpoint_evaluation_batch_size"),
        checkpoint_move_time_limit_sec=getattr(args, f"{prefix}_checkpoint_move_time_limit_sec"),
        checkpoint_device=getattr(args, f"{prefix}_checkpoint_device"),
        checkpoint_board_backend=getattr(args, f"{prefix}_checkpoint_board_backend"),
        yaneuraou_command=getattr(args, f"{prefix}_yaneuraou_command"),
        yaneuraou_go_command=getattr(args, f"{prefix}_yaneuraou_go_command"),
        yaneuraou_read_timeout_seconds=getattr(args, f"{prefix}_yaneuraou_read_timeout_seconds"),
        yaneuraou_policy_target_multipv=getattr(args, f"{prefix}_yaneuraou_policy_target_multipv"),
        yaneuraou_policy_target_temperature_cp=getattr(args, f"{prefix}_yaneuraou_policy_target_temperature_cp"),
    )


def _warn_short_max_plies(max_plies: int) -> None:
    if max_plies < STANDARD_MAX_PLIES:
        print(
            f"warning: --max-plies {max_plies} is below the computer-shogi standard cap "
            f"of {STANDARD_MAX_PLIES}; this can create artificial max_plies draws.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
