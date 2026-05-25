from __future__ import annotations

import argparse
import os

from shogi_arena_agent.csa_player import new_python_shogi_csa_protocol, run_csa_player
from shogi_arena_agent.player_cli import (
    add_player_arguments,
    player_context,
    player_spec_from_args,
    validate_player_arguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a shogi player on a CSA-compatible server such as Floodgate."
    )
    parser.add_argument("--host", default="wdoor.c.u-tokyo.ac.jp")
    parser.add_argument("--port", type=int, default=4081)
    parser.add_argument("--username", default=os.environ.get("CSA_USERNAME"))
    parser.add_argument("--password", default=os.environ.get("CSA_PASSWORD"))
    parser.add_argument("--games", type=int, default=1)
    add_player_arguments(parser, "player")
    args = parser.parse_args()
    if not args.username:
        parser.error("--username or CSA_USERNAME is required")
    if not args.password:
        parser.error("--password or CSA_PASSWORD is required")
    validate_player_arguments(parser, args, "player")
    return args


def main() -> None:
    args = parse_args()
    spec = player_spec_from_args(
        args,
        "player",
        default_move_selection_profile="visit-sampling",
        default_move_selection_temperature=0.0,
        default_move_selection_temperature_plies=0,
    )
    with player_context(spec, name=args.username) as built:
        results = run_csa_player(
            protocol=new_python_shogi_csa_protocol(),
            player=built.player,
            host=args.host,
            port=args.port,
            username=args.username,
            password=args.password,
            games=args.games,
        )
    for result in results:
        print(
            f"game={result.game_count} moves_played={result.moves_played} end_message={result.end_message}",
            flush=True,
        )


if __name__ == "__main__":
    main()
