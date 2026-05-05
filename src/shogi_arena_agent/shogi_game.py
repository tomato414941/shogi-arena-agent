from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import shogi

from shogi_arena_agent.usi import RESIGN_MOVE, UsiEngine
from shogi_arena_agent.usi_process import UsiGoResult


@dataclass(frozen=True)
class PlayerSpec:
    kind: str
    name: str
    settings: dict[str, str | int | float | bool | None]


@dataclass(frozen=True)
class ShogiGamePlyRecord:
    side: str
    position: str
    bestmove: str
    ponder: str | None = None
    usi_info_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShogiGameRecord:
    black_player: PlayerSpec
    white_player: PlayerSpec
    plies: tuple[ShogiGamePlyRecord, ...]
    end_reason: str
    winner: str | None = None


def save_shogi_game_records_jsonl(results: Iterable[ShogiGameRecord], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(shogi_game_record_to_json(result), sort_keys=True) for result in results]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_shogi_game_records_jsonl(path: str | Path) -> tuple[ShogiGameRecord, ...]:
    results: list[ShogiGameRecord] = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                results.append(shogi_game_record_from_json(json.loads(line)))
    return tuple(results)


def shogi_game_record_to_json(result: ShogiGameRecord) -> dict[str, object]:
    return {
        "plies": [_ply_record_to_json(ply) for ply in result.plies],
        "end_reason": result.end_reason,
        "winner": result.winner,
        "black_player": _player_spec_to_json(result.black_player),
        "white_player": _player_spec_to_json(result.white_player),
    }


def shogi_game_record_from_json(data: dict[str, object]) -> ShogiGameRecord:
    return ShogiGameRecord(
        black_player=_player_spec_from_json(_object_dict(data["black_player"])),
        white_player=_player_spec_from_json(_object_dict(data["white_player"])),
        plies=tuple(_ply_record_from_json(_object_dict(ply)) for ply in cast(list[object], data["plies"])),
        end_reason=str(data["end_reason"]),
        winner=None if data.get("winner") is None else str(data["winner"]),
    )


class ShogiPlayer(Protocol):
    def position(self, command: str) -> None:
        """Set the current USI position."""

    def go(self) -> str | UsiGoResult:
        """Return a USI bestmove or full go result."""


class InProcessShogiPlayer:
    def __init__(self, engine: UsiEngine) -> None:
        self.engine = engine

    def position(self, command: str) -> None:
        self.engine.handle_line(command)

    def go(self) -> UsiGoResult:
        response = self.engine.handle_line("go btime 0 wtime 0")
        if len(response) != 1 or not response[0].startswith("bestmove "):
            return UsiGoResult(bestmove=RESIGN_MOVE)
        return _go_result_from_bestmove_line(response[0])


def position_command(moves: tuple[str, ...]) -> str:
    if not moves:
        return "position startpos"
    return "position startpos moves " + " ".join(moves)


def play_shogi_game(
    *,
    black: ShogiPlayer | UsiEngine | None = None,
    white: ShogiPlayer | UsiEngine | None = None,
    black_player: PlayerSpec | None = None,
    white_player: PlayerSpec | None = None,
    max_plies: int = 32,
) -> ShogiGameRecord:
    board = shogi.Board()
    black_engine = black or UsiEngine(name="black")
    white_engine = white or UsiEngine(name="white")
    players = [
        _as_player(black_engine),
        _as_player(white_engine),
    ]
    black_spec = black_player or _default_player_spec(black_engine, side="black")
    white_spec = white_player or _default_player_spec(white_engine, side="white")
    plies: list[ShogiGamePlyRecord] = []

    for ply in range(max_plies):
        player = players[ply % 2]
        moves = tuple(record.bestmove for record in plies)
        position = position_command(moves)
        side = "black" if board.turn == shogi.BLACK else "white"
        player.position(position)
        go_result = _coerce_go_result(player.go())
        move = go_result.bestmove
        if move == RESIGN_MOVE:
            winner = "white" if board.turn == shogi.BLACK else "black"
            return ShogiGameRecord(
                black_player=black_spec,
                white_player=white_spec,
                plies=tuple(plies),
                end_reason="resign",
                winner=winner,
            )

        legal_moves = {legal_move.usi() for legal_move in board.legal_moves}
        if move not in legal_moves:
            winner = "white" if board.turn == shogi.BLACK else "black"
            return ShogiGameRecord(
                black_player=black_spec,
                white_player=white_spec,
                plies=tuple(plies),
                end_reason="illegal_move",
                winner=winner,
            )

        plies.append(
            ShogiGamePlyRecord(
                side=side,
                position=position,
                bestmove=move,
                ponder=go_result.ponder,
                usi_info_lines=go_result.info_lines,
            )
        )
        board.push_usi(move)
        if board.is_game_over():
            winner = "black" if board.turn == shogi.WHITE else "white"
            return ShogiGameRecord(
                black_player=black_spec,
                white_player=white_spec,
                plies=tuple(plies),
                end_reason="game_over",
                winner=winner,
            )

    return ShogiGameRecord(
        black_player=black_spec,
        white_player=white_spec,
        plies=tuple(plies),
        end_reason="max_plies",
    )


def _as_player(player: ShogiPlayer | UsiEngine) -> ShogiPlayer:
    if isinstance(player, UsiEngine):
        return InProcessShogiPlayer(player)
    return player


def _default_player_spec(player: ShogiPlayer | UsiEngine, *, side: str) -> PlayerSpec:
    if isinstance(player, UsiEngine):
        return PlayerSpec(kind="usi_engine", name=player.name, settings={})
    return PlayerSpec(kind="shogi_player", name=side, settings={})


def _player_spec_to_json(spec: PlayerSpec) -> dict[str, object]:
    return {
        "kind": spec.kind,
        "name": spec.name,
        "settings": spec.settings,
    }


def _ply_record_to_json(record: ShogiGamePlyRecord) -> dict[str, object]:
    return {
        "side": record.side,
        "position": record.position,
        "bestmove": record.bestmove,
        "ponder": record.ponder,
        "usi_info_lines": list(record.usi_info_lines),
    }


def _ply_record_from_json(data: dict[str, object]) -> ShogiGamePlyRecord:
    return ShogiGamePlyRecord(
        side=str(data["side"]),
        position=str(data["position"]),
        bestmove=str(data["bestmove"]),
        ponder=None if data.get("ponder") is None else str(data["ponder"]),
        usi_info_lines=tuple(str(line) for line in cast(list[object], data.get("usi_info_lines", []))),
    )


def _coerce_go_result(result: str | UsiGoResult) -> UsiGoResult:
    if isinstance(result, UsiGoResult):
        return result
    return UsiGoResult(bestmove=result)


def _go_result_from_bestmove_line(line: str) -> UsiGoResult:
    words = line.removeprefix("bestmove ").split()
    ponder = words[2] if len(words) >= 3 and words[1] == "ponder" else None
    return UsiGoResult(bestmove=words[0], ponder=ponder)


def _player_spec_from_json(data: dict[str, object]) -> PlayerSpec:
    settings = data.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("player settings must be an object")
    return PlayerSpec(
        kind=str(data["kind"]),
        name=str(data["name"]),
        settings=cast(dict[str, str | int | float | bool | None], settings),
    )


def _object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected object")
    return value
