from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import shogi

from shogi_arena_agent.usi import RESIGN_MOVE, UsiEngine


@dataclass(frozen=True)
class LocalMatchResult:
    moves: tuple[str, ...]
    end_reason: str
    final_sfen: str
    winner: str | None = None


def save_match_result(result: LocalMatchResult, path: str | Path) -> None:
    data = {
        "moves": list(result.moves),
        "end_reason": result.end_reason,
        "final_sfen": result.final_sfen,
        "winner": result.winner,
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_match_result(path: str | Path) -> LocalMatchResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return LocalMatchResult(
        moves=tuple(data["moves"]),
        end_reason=data["end_reason"],
        final_sfen=data["final_sfen"],
        winner=data.get("winner"),
    )


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
            winner = "white" if board.turn == shogi.BLACK else "black"
            return LocalMatchResult(moves=tuple(moves), end_reason="resign", final_sfen=board.sfen(), winner=winner)

        legal_moves = {legal_move.usi() for legal_move in board.legal_moves}
        if move not in legal_moves:
            winner = "white" if board.turn == shogi.BLACK else "black"
            return LocalMatchResult(moves=tuple(moves), end_reason="illegal_move", final_sfen=board.sfen(), winner=winner)

        board.push_usi(move)
        moves.append(move)
        if board.is_game_over():
            winner = "black" if board.turn == shogi.WHITE else "white"
            return LocalMatchResult(moves=tuple(moves), end_reason="game_over", final_sfen=board.sfen(), winner=winner)

    return LocalMatchResult(moves=tuple(moves), end_reason="max_plies", final_sfen=board.sfen())


def _as_player(player: LocalPlayer | UsiEngine) -> LocalPlayer:
    if isinstance(player, UsiEngine):
        return InProcessPlayer(player)
    return player
