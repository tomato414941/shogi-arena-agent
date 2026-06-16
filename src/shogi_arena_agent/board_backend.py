from __future__ import annotations

import copy
from dataclasses import dataclass

import cshogi
import shogi

from shogi_arena_agent.usi import BOARD_BACKENDS, UsiPosition, board_from_position

ShogiBoard = shogi.Board | cshogi.Board


@dataclass(frozen=True)
class ShogiTerminalStatus:
    end_reason: str
    winner: str | None


def new_board(*, backend: str = "python-shogi") -> ShogiBoard:
    return board_from_position(UsiPosition(), backend=backend)


def board_turn_name(board: ShogiBoard) -> str:
    return "black" if board.turn == shogi.BLACK else "white"


def board_is_black_turn(board: ShogiBoard) -> bool:
    return board.turn == shogi.BLACK


def copy_board(board: ShogiBoard) -> ShogiBoard:
    if isinstance(board, cshogi.Board):
        return board.copy()
    return copy.deepcopy(board)


def legal_move_usis(board: ShogiBoard) -> tuple[str, ...]:
    if isinstance(board, cshogi.Board):
        return tuple(sorted(cshogi.move_to_usi(move) for move in board.legal_moves))
    return tuple(sorted(move.usi() for move in board.legal_moves))


def terminal_status(board: ShogiBoard) -> ShogiTerminalStatus | None:
    if board.is_game_over():
        return ShogiTerminalStatus(end_reason="game_over", winner=opponent_side(board_turn_name(board)))
    if isinstance(board, cshogi.Board):
        repetition_status = board.is_draw()
        if repetition_status != cshogi.NOT_REPETITION:
            return _cshogi_repetition_status(board, repetition_status)
        if board.is_nyugyoku():
            return ShogiTerminalStatus(end_reason="nyugyoku_win", winner=board_turn_name(board))
        return None
    if board.is_fourfold_repetition():
        return ShogiTerminalStatus(end_reason="repetition_draw", winner=None)
    return None


def opponent_side(side: str) -> str:
    if side == "black":
        return "white"
    if side == "white":
        return "black"
    raise ValueError(f"unsupported side: {side}")


def validate_board_backend(backend: str) -> None:
    if backend not in BOARD_BACKENDS:
        raise ValueError(f"unsupported board backend: {backend}")


def _cshogi_repetition_status(board: cshogi.Board, repetition_status: int) -> ShogiTerminalStatus:
    side_to_move = board_turn_name(board)
    previous_side = opponent_side(side_to_move)
    if repetition_status == cshogi.REPETITION_DRAW:
        return ShogiTerminalStatus(end_reason="repetition_draw", winner=None)
    if repetition_status == cshogi.REPETITION_WIN:
        return ShogiTerminalStatus(end_reason="repetition_win", winner=side_to_move)
    if repetition_status == cshogi.REPETITION_LOSE:
        return ShogiTerminalStatus(end_reason="repetition_loss", winner=previous_side)
    if repetition_status == cshogi.REPETITION_SUPERIOR:
        return ShogiTerminalStatus(end_reason="repetition_superior", winner=side_to_move)
    if repetition_status == cshogi.REPETITION_INFERIOR:
        return ShogiTerminalStatus(end_reason="repetition_inferior", winner=previous_side)
    raise ValueError(f"unsupported cshogi repetition status: {repetition_status}")
