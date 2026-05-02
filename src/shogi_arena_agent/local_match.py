from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import shogi

from shogi_arena_agent.usi import RESIGN_MOVE, UsiEngine


@dataclass(frozen=True)
class LocalMatchResult:
    moves: tuple[str, ...]
    end_reason: str


class LocalPlayer(Protocol):
    def position(self, command: str) -> None:
        """Set the current USI position."""

    def go(self) -> str:
        """Return a USI bestmove."""


class InProcessPlayer:
    def __init__(self, engine: UsiEngine) -> None:
        self.engine = engine

    def position(self, command: str) -> None:
        self.engine.handle_line(command)

    def go(self) -> str:
        response = self.engine.handle_line("go btime 0 wtime 0")
        if len(response) != 1 or not response[0].startswith("bestmove "):
            return RESIGN_MOVE
        return response[0].removeprefix("bestmove ")


def position_command(moves: tuple[str, ...]) -> str:
    if not moves:
        return "position startpos"
    return "position startpos moves " + " ".join(moves)


def play_local_match(
    *,
    black: LocalPlayer | UsiEngine | None = None,
    white: LocalPlayer | UsiEngine | None = None,
    max_plies: int = 32,
) -> LocalMatchResult:
    board = shogi.Board()
    players = [
        _as_player(black or UsiEngine(name="black")),
        _as_player(white or UsiEngine(name="white")),
    ]
    moves: list[str] = []

    for ply in range(max_plies):
        player = players[ply % 2]
        player.position(position_command(tuple(moves)))
        move = player.go()
        if move == RESIGN_MOVE:
            return LocalMatchResult(moves=tuple(moves), end_reason="resign")

        legal_moves = {legal_move.usi() for legal_move in board.legal_moves}
        if move not in legal_moves:
            return LocalMatchResult(moves=tuple(moves), end_reason="illegal_move")

        board.push_usi(move)
        moves.append(move)
        if board.is_game_over():
            return LocalMatchResult(moves=tuple(moves), end_reason="game_over")

    return LocalMatchResult(moves=tuple(moves), end_reason="max_plies")


def _as_player(player: LocalPlayer | UsiEngine) -> LocalPlayer:
    if isinstance(player, UsiEngine):
        return InProcessPlayer(player)
    return player
