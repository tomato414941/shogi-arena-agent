from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import shogi

from shogi_arena_agent.usi import RESIGN_MOVE, UsiEngine


@dataclass(frozen=True)
class PlayerSpec:
    kind: str
    name: str
    settings: dict[str, str | int | float | bool | None]


@dataclass(frozen=True)
class LocalMatchResult:
    black_player: PlayerSpec
    white_player: PlayerSpec
    moves: tuple[str, ...]
    end_reason: str
    winner: str | None = None


def save_match_result(result: LocalMatchResult, path: str | Path) -> None:
    data = {
        "moves": list(result.moves),
        "end_reason": result.end_reason,
        "winner": result.winner,
        "black_player": _player_spec_to_json(result.black_player),
        "white_player": _player_spec_to_json(result.white_player),
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_match_result(path: str | Path) -> LocalMatchResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return LocalMatchResult(
        black_player=_player_spec_from_json(data["black_player"]),
        white_player=_player_spec_from_json(data["white_player"]),
        moves=tuple(data["moves"]),
        end_reason=data["end_reason"],
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
    black_player: PlayerSpec | None = None,
    white_player: PlayerSpec | None = None,
    max_plies: int = 32,
) -> LocalMatchResult:
    board = shogi.Board()
    black_engine = black or UsiEngine(name="black")
    white_engine = white or UsiEngine(name="white")
    players = [
        _as_player(black_engine),
        _as_player(white_engine),
    ]
    black_spec = black_player or _default_player_spec(black_engine, side="black")
    white_spec = white_player or _default_player_spec(white_engine, side="white")
    moves: list[str] = []

    for ply in range(max_plies):
        player = players[ply % 2]
        player.position(position_command(tuple(moves)))
        move = player.go()
        if move == RESIGN_MOVE:
            winner = "white" if board.turn == shogi.BLACK else "black"
            return LocalMatchResult(
                black_player=black_spec,
                white_player=white_spec,
                moves=tuple(moves),
                end_reason="resign",
                winner=winner,
            )

        legal_moves = {legal_move.usi() for legal_move in board.legal_moves}
        if move not in legal_moves:
            winner = "white" if board.turn == shogi.BLACK else "black"
            return LocalMatchResult(
                black_player=black_spec,
                white_player=white_spec,
                moves=tuple(moves),
                end_reason="illegal_move",
                winner=winner,
            )

        board.push_usi(move)
        moves.append(move)
        if board.is_game_over():
            winner = "black" if board.turn == shogi.WHITE else "white"
            return LocalMatchResult(
                black_player=black_spec,
                white_player=white_spec,
                moves=tuple(moves),
                end_reason="game_over",
                winner=winner,
            )

    return LocalMatchResult(
        black_player=black_spec,
        white_player=white_spec,
        moves=tuple(moves),
        end_reason="max_plies",
    )


def _as_player(player: LocalPlayer | UsiEngine) -> LocalPlayer:
    if isinstance(player, UsiEngine):
        return InProcessPlayer(player)
    return player


def _default_player_spec(player: LocalPlayer | UsiEngine, *, side: str) -> PlayerSpec:
    if isinstance(player, UsiEngine):
        return PlayerSpec(kind="usi_engine", name=player.name, settings={})
    return PlayerSpec(kind="local_player", name=side, settings={})


def _player_spec_to_json(spec: PlayerSpec) -> dict[str, object]:
    return {
        "kind": spec.kind,
        "name": spec.name,
        "settings": spec.settings,
    }


def _player_spec_from_json(data: dict[str, object]) -> PlayerSpec:
    settings = data.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("player settings must be an object")
    return PlayerSpec(
        kind=str(data["kind"]),
        name=str(data["name"]),
        settings=cast(dict[str, str | int | float | bool | None], settings),
    )
