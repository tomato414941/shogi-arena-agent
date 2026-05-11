from __future__ import annotations

import cshogi
import copy
import shogi

from shogi_arena_agent.usi import BOARD_BACKENDS, UsiPosition, board_from_position

ShogiBoard = shogi.Board | cshogi.Board


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


def validate_board_backend(backend: str) -> None:
    if backend not in BOARD_BACKENDS:
        raise ValueError(f"unsupported board backend: {backend}")
