from __future__ import annotations

import argparse
import os
from typing import Any

import shogi

from shogi_arena_agent.csa_player import (
    CsaProtocol,
    new_python_shogi_csa_protocol,
    run_csa_player,
)
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
    parser.add_argument("--game-name")
    parser.add_argument("--game-side", choices=("black", "white"))
    add_player_arguments(parser, "player")
    args = parser.parse_args()
    if not args.username:
        parser.error("--username or CSA_USERNAME is required")
    if not args.password:
        parser.error("--password or CSA_PASSWORD is required")
    if args.game_name and not args.game_side:
        parser.error("--game-side is required when --game-name is set")
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
            protocol=_LoggingCsaProtocol(
                new_python_shogi_csa_protocol(
                    game_command=_game_command(args.game_name, args.game_side)
                )
            ),
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


def _game_command(game_name: str | None, game_side: str | None) -> str | None:
    if game_name is None:
        return None
    side = {"black": "+", "white": "-"}.get(game_side or "")
    if side is None:
        raise ValueError("--game-side is required when --game-name is set")
    return f"%%GAME {game_name} {side}"


class _LoggingCsaProtocol:
    def __init__(self, protocol: CsaProtocol) -> None:
        self.protocol = protocol

    def open(self, host: str, port: int = 0) -> object:
        print(f"csa_open host={host} port={port}", flush=True)
        result = self.protocol.open(host, port)
        print("csa_opened", flush=True)
        return result

    def login_ex(self, username: str, password: str) -> object:
        print(f"csa_login username={username}", flush=True)
        result = self.protocol.login_ex(username, password)
        print("csa_logged_in", flush=True)
        return result

    def wait_match(self, block: bool = True) -> dict[str, object] | None:
        print("csa_wait_match", flush=True)
        match = self.protocol.wait_match(block=block)
        print(f"csa_match_received={match is not None}", flush=True)
        return match

    def agree(self) -> object:
        print("csa_agree", flush=True)
        result = self.protocol.agree()
        print("csa_agreed", flush=True)
        return result

    def wait_server_message(
        self,
        board: shogi.Board,
        block: bool = True,
    ) -> tuple[int | None, str | None, int | None, int | None] | None:
        message = self.protocol.wait_server_message(board, block=block)
        _color, move_usi, _time, end_message = message or (None, None, None, None)
        if move_usi is not None:
            print(f"csa_server_move move={move_usi}", flush=True)
        if end_message is not None:
            print(f"csa_server_end message={end_message}", flush=True)
        return message

    def move(self, piece_type: int, color: int, move: shogi.Move) -> object:
        print(f"csa_send_move move={move.usi()}", flush=True)
        return self.protocol.move(piece_type, color, move)

    def resign(self) -> object:
        print("csa_resign", flush=True)
        return self.protocol.resign()

    def logout(self) -> object:
        print("csa_logout", flush=True)
        return self.protocol.logout()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.protocol, name)


if __name__ == "__main__":
    main()
