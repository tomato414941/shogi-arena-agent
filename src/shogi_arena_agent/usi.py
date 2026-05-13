from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import cshogi
import shogi

RESIGN_MOVE = "resign"
BOARD_BACKENDS = ("python-shogi", "cshogi")


@dataclass(frozen=True)
class UsiPosition:
    command: str = "position startpos"


class MovePolicy(Protocol):
    def select_move(self, position: UsiPosition) -> str:
        """Return a USI move string such as 7g7f."""


class SessionMovePolicyFactory(Protocol):
    def new_session(self) -> MovePolicy:
        """Return a move policy state for one game."""


def board_from_position(position: UsiPosition, *, backend: str = "python-shogi") -> shogi.Board | cshogi.Board:
    if backend not in BOARD_BACKENDS:
        raise ValueError(f"unsupported board backend: {backend}")
    words = position.command.split()
    if words[:2] == ["position", "startpos"]:
        board = _new_board(backend)
        moves = words[words.index("moves") + 1 :] if "moves" in words else []
    elif words[:2] == ["position", "sfen"]:
        moves_index = words.index("moves") if "moves" in words else len(words)
        sfen = " ".join(words[2:moves_index])
        board = _new_board(backend, sfen=sfen)
        moves = words[moves_index + 1 :] if moves_index < len(words) else []
    else:
        board = _new_board(backend)
        moves = []

    for move in moves:
        board.push_usi(move)
    return board


def _new_board(backend: str, *, sfen: str | None = None) -> shogi.Board | cshogi.Board:
    if backend == "python-shogi":
        return shogi.Board(sfen) if sfen is not None else shogi.Board()
    board = cshogi.Board()
    if sfen is not None:
        board.set_sfen(sfen)
    return board


class UsiEngine:
    def __init__(self, *, name: str = "shogi-arena-agent", author: str = "intrep", policy: MovePolicy | None = None) -> None:
        from shogi_arena_agent.deterministic_legal_policy import DeterministicLegalMovePolicy

        self.name = name
        self.author = author
        self.policy = policy or DeterministicLegalMovePolicy()
        self._active_policy: MovePolicy | None = None
        self.position = UsiPosition()

    @property
    def active_policy(self) -> MovePolicy:
        if self._active_policy is None:
            self._active_policy = _new_policy_session(self.policy)
        return self._active_policy

    def new_game(self) -> None:
        self.position = UsiPosition()
        self._active_policy = _new_policy_session(self.policy)

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
            return [f"bestmove {self.active_policy.select_move(self.position)}"]
        if line == "usinewgame":
            self.new_game()
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


def _new_policy_session(policy: MovePolicy) -> MovePolicy:
    new_session = getattr(policy, "new_session", None)
    if callable(new_session):
        return new_session()
    return policy
