from __future__ import annotations

from shogi_arena_agent.board_backend import legal_move_usis
from shogi_arena_agent.usi import RESIGN_MOVE, UsiPosition, board_from_position


class DeterministicLegalMovePolicy:
    def select_move(self, position: UsiPosition) -> str:
        board = board_from_position(position)
        moves = legal_move_usis(board)
        if not moves:
            return RESIGN_MOVE
        return moves[0]
