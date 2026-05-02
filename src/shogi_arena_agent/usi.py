from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import shogi

RESIGN_MOVE = "resign"


@dataclass(frozen=True)
class UsiPosition:
    command: str = "position startpos"


class MovePolicy(Protocol):
    def select_move(self, position: UsiPosition) -> str:
        """Return a USI move string such as 7g7f."""


def board_from_position(position: UsiPosition) -> shogi.Board:
    words = position.command.split()
    if words[:2] == ["position", "startpos"]:
        board = shogi.Board()
        moves = words[words.index("moves") + 1 :] if "moves" in words else []
    elif words[:2] == ["position", "sfen"]:
        moves_index = words.index("moves") if "moves" in words else len(words)
        sfen = " ".join(words[2:moves_index])
        board = shogi.Board(sfen)
        moves = words[moves_index + 1 :] if moves_index < len(words) else []
    else:
        board = shogi.Board()
        moves = []

    for move in moves:
        board.push_usi(move)
    return board


class LegalMovePolicy:
    def select_move(self, position: UsiPosition) -> str:
        board = board_from_position(position)
        moves = sorted(move.usi() for move in board.legal_moves)
        if not moves:
            return RESIGN_MOVE
        return moves[0]


class UsiEngine:
    def __init__(self, *, name: str = "shogi-arena-agent", author: str = "intrep", policy: MovePolicy | None = None) -> None:
        self.name = name
        self.author = author
        self.policy = policy or LegalMovePolicy()
        self.position = UsiPosition()

    def handle_line(self, line: str) -> list[str]:
        line = line.strip()
        if line == "usi":
            return [
                f"id name {self.name}",
                f"id author {self.author}",
                "usiok",
            ]
        if line == "isready":
            return ["readyok"]
        if line.startswith("position"):
            self.position = UsiPosition(command=line)
            return []
        if line.startswith("go"):
            return [f"bestmove {self.policy.select_move(self.position)}"]
        if line == "usinewgame":
            self.position = UsiPosition()
            return []
        if line == "quit":
            return []
        return []


def run_usi_loop(input_lines: Iterable[str], *, engine: UsiEngine | None = None) -> list[str]:
    engine = engine or UsiEngine()
    output: list[str] = []
    for line in input_lines:
        if line.strip() == "quit":
            break
        output.extend(engine.handle_line(line))
    return output
