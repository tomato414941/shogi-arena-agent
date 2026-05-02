from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


STARTPOS_FALLBACK_MOVE = "7g7f"


@dataclass(frozen=True)
class UsiPosition:
    command: str = "position startpos"


class MovePolicy(Protocol):
    def select_move(self, position: UsiPosition) -> str:
        """Return a USI move string such as 7g7f."""


class FallbackPolicy:
    def select_move(self, position: UsiPosition) -> str:
        return STARTPOS_FALLBACK_MOVE


class UsiEngine:
    def __init__(self, *, name: str = "shogi-arena-agent", author: str = "intrep", policy: MovePolicy | None = None) -> None:
        self.name = name
        self.author = author
        self.policy = policy or FallbackPolicy()
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
