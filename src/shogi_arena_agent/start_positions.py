from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from shogi_arena_agent.board_backend import legal_move_usis
from shogi_arena_agent.usi import UsiPosition, board_from_position


@dataclass(frozen=True)
class StartPosition:
    id: str
    usi_position: UsiPosition
    sfen: str
    opening_moves: tuple[str, ...] = ()
    source: str = "explicit"


def startpos() -> StartPosition:
    board = board_from_position(UsiPosition())
    return StartPosition(
        id="startpos",
        usi_position=UsiPosition(),
        sfen=board.sfen(),
        opening_moves=(),
        source="startpos",
    )


def random_legal_opening_start_positions(
    *,
    count: int,
    opening_plies: int,
    seed: int,
    board_backend: str = "python-shogi",
) -> tuple[StartPosition, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    if opening_plies < 0:
        raise ValueError("opening_plies must be non-negative")
    rng = random.Random(seed)
    return tuple(
        _random_legal_opening_start_position(
            index=index,
            opening_plies=opening_plies,
            rng=rng,
            board_backend=board_backend,
        )
        for index in range(count)
    )


def save_start_positions_jsonl(start_positions: Iterable[StartPosition], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(start_position_to_json(position), sort_keys=True) for position in start_positions]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def start_position_to_json(position: StartPosition) -> dict[str, object]:
    return {
        "id": position.id,
        "usi_position": position.usi_position.command,
        "sfen": position.sfen,
        "opening_moves": list(position.opening_moves),
        "source": position.source,
    }


def _random_legal_opening_start_position(
    *,
    index: int,
    opening_plies: int,
    rng: random.Random,
    board_backend: str,
) -> StartPosition:
    opening_moves: list[str] = []
    board = board_from_position(UsiPosition(), backend=board_backend)
    for _ply in range(opening_plies):
        legal_moves = legal_move_usis(board)
        if not legal_moves:
            break
        move = rng.choice(legal_moves)
        board.push_usi(move)
        opening_moves.append(move)
        if board.is_game_over():
            break
    usi_position = UsiPosition(command=_position_command_from_opening_moves(tuple(opening_moves)))
    return StartPosition(
        id=f"random-legal-opening-{index:06d}",
        usi_position=usi_position,
        sfen=board.sfen(),
        opening_moves=tuple(opening_moves),
        source="random-legal-opening",
    )


def _position_command_from_opening_moves(opening_moves: tuple[str, ...]) -> str:
    if not opening_moves:
        return "position startpos"
    return "position startpos moves " + " ".join(opening_moves)
