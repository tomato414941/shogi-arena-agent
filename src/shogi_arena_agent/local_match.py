from __future__ import annotations

from dataclasses import dataclass

import shogi

from shogi_arena_agent.usi import RESIGN_MOVE, UsiEngine


@dataclass(frozen=True)
class LocalMatchResult:
    moves: tuple[str, ...]
    end_reason: str


def position_command(moves: tuple[str, ...]) -> str:
    if not moves:
        return "position startpos"
    return "position startpos moves " + " ".join(moves)


def play_local_match(
    *,
    black: UsiEngine | None = None,
    white: UsiEngine | None = None,
    max_plies: int = 32,
) -> LocalMatchResult:
    board = shogi.Board()
    engines = [black or UsiEngine(name="black"), white or UsiEngine(name="white")]
    moves: list[str] = []

    for ply in range(max_plies):
        engine = engines[ply % 2]
        engine.handle_line(position_command(tuple(moves)))
        response = engine.handle_line("go btime 0 wtime 0")
        if len(response) != 1 or not response[0].startswith("bestmove "):
            return LocalMatchResult(moves=tuple(moves), end_reason="invalid_response")

        move = response[0].removeprefix("bestmove ")
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
