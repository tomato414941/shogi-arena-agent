from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Protocol, cast

from shogi_arena_agent.board_backend import board_is_black_turn, board_turn_name, legal_move_usis, new_board
from shogi_arena_agent.usi import RESIGN_MOVE, UsiEngine
from shogi_arena_agent.usi_process import UsiGoResult


@dataclass(frozen=True)
class ShogiActorSpec:
    kind: str
    name: str
    settings: dict[str, str | int | float | bool | None]


@dataclass(frozen=True)
class ShogiTransitionRecord:
    ply: int
    side: str
    position_sfen: str
    legal_moves: tuple[str, ...]
    action_usi: str
    next_position_sfen: str
    reward: float
    done: bool
    decision_usi_info_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShogiGameRecord:
    black_actor: ShogiActorSpec
    white_actor: ShogiActorSpec
    initial_position_sfen: str
    transitions: tuple[ShogiTransitionRecord, ...]
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
        "black_actor": _actor_spec_to_json(result.black_actor),
        "white_actor": _actor_spec_to_json(result.white_actor),
        "initial_position_sfen": result.initial_position_sfen,
        "transitions": [_transition_record_to_json(transition) for transition in result.transitions],
        "end_reason": result.end_reason,
        "winner": result.winner,
    }


def shogi_game_record_from_json(data: dict[str, object]) -> ShogiGameRecord:
    return ShogiGameRecord(
        black_actor=_actor_spec_from_json(_object_dict(data["black_actor"])),
        white_actor=_actor_spec_from_json(_object_dict(data["white_actor"])),
        initial_position_sfen=str(data["initial_position_sfen"]),
        transitions=tuple(
            _transition_record_from_json(_object_dict(transition))
            for transition in cast(list[object], data["transitions"])
        ),
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
        move = self.engine.active_policy.select_move(self.engine.position)
        return UsiGoResult(bestmove=move, info_lines=_policy_info_lines(self.engine.active_policy))


def position_command(moves: tuple[str, ...]) -> str:
    if not moves:
        return "position startpos"
    return "position startpos moves " + " ".join(moves)


def play_shogi_game(
    *,
    black: ShogiPlayer | UsiEngine | None = None,
    white: ShogiPlayer | UsiEngine | None = None,
    black_actor: ShogiActorSpec | None = None,
    white_actor: ShogiActorSpec | None = None,
    max_plies: int = 32,
    board_backend: str = "python-shogi",
) -> ShogiGameRecord:
    board = new_board(backend=board_backend)
    initial_position_sfen = board.sfen()
    black_engine = black or UsiEngine(name="black")
    white_engine = white or UsiEngine(name="white")
    if isinstance(black_engine, UsiEngine):
        black_engine.new_game()
    if isinstance(white_engine, UsiEngine):
        white_engine.new_game()
    players = [
        _as_player(black_engine),
        _as_player(white_engine),
    ]
    black_spec = black_actor or _default_actor_spec(black_engine, side="black")
    white_spec = white_actor or _default_actor_spec(white_engine, side="white")
    transitions: list[ShogiTransitionRecord] = []

    for ply in range(max_plies):
        player = players[ply % 2]
        moves = tuple(record.action_usi for record in transitions)
        position = position_command(moves)
        side = board_turn_name(board)
        position_sfen = board.sfen()
        legal_moves = legal_move_usis(board)
        player.position(position)
        go_result = _coerce_go_result(player.go())
        move = go_result.bestmove
        if move == RESIGN_MOVE:
            winner = "white" if board_is_black_turn(board) else "black"
            return ShogiGameRecord(
                black_actor=black_spec,
                white_actor=white_spec,
                initial_position_sfen=initial_position_sfen,
                transitions=tuple(_finalize_transitions(transitions, winner=winner)),
                end_reason="resign",
                winner=winner,
            )

        if move not in legal_moves:
            winner = "white" if board_is_black_turn(board) else "black"
            return ShogiGameRecord(
                black_actor=black_spec,
                white_actor=white_spec,
                initial_position_sfen=initial_position_sfen,
                transitions=tuple(_finalize_transitions(transitions, winner=winner)),
                end_reason="illegal_move",
                winner=winner,
            )

        board.push_usi(move)
        done = board.is_game_over()
        winner = "black" if done and not board_is_black_turn(board) else "white" if done else None
        transitions.append(
            ShogiTransitionRecord(
                ply=ply,
                side=side,
                position_sfen=position_sfen,
                legal_moves=legal_moves,
                action_usi=move,
                next_position_sfen=board.sfen(),
                reward=_transition_reward(side=side, winner=winner, done=done),
                done=done,
                decision_usi_info_lines=go_result.info_lines,
            )
        )
        if done:
            return ShogiGameRecord(
                black_actor=black_spec,
                white_actor=white_spec,
                initial_position_sfen=initial_position_sfen,
                transitions=tuple(transitions),
                end_reason="game_over",
                winner=winner,
            )

    return ShogiGameRecord(
        black_actor=black_spec,
        white_actor=white_spec,
        initial_position_sfen=initial_position_sfen,
        transitions=tuple(_finalize_transitions(transitions, winner=None)),
        end_reason="max_plies",
    )


def _as_player(player: ShogiPlayer | UsiEngine) -> ShogiPlayer:
    if isinstance(player, UsiEngine):
        return InProcessShogiPlayer(player)
    return player


def _policy_info_lines(policy: object) -> tuple[str, ...]:
    performance = getattr(policy, "last_performance", None)
    if performance is None or not is_dataclass(performance):
        return ()
    return ("info string intrep_performance " + json.dumps(asdict(performance), sort_keys=True),)


def _default_actor_spec(player: ShogiPlayer | UsiEngine, *, side: str) -> ShogiActorSpec:
    if isinstance(player, UsiEngine):
        return ShogiActorSpec(kind="usi_engine", name=player.name, settings={})
    return ShogiActorSpec(kind="shogi_player", name=side, settings={})


def _actor_spec_to_json(spec: ShogiActorSpec) -> dict[str, object]:
    return {
        "kind": spec.kind,
        "name": spec.name,
        "settings": spec.settings,
    }


def _transition_record_to_json(record: ShogiTransitionRecord) -> dict[str, object]:
    return {
        "ply": record.ply,
        "side": record.side,
        "position_sfen": record.position_sfen,
        "legal_moves": list(record.legal_moves),
        "action_usi": record.action_usi,
        "next_position_sfen": record.next_position_sfen,
        "reward": record.reward,
        "done": record.done,
        "decision_usi_info_lines": list(record.decision_usi_info_lines),
    }


def _transition_record_from_json(data: dict[str, object]) -> ShogiTransitionRecord:
    return ShogiTransitionRecord(
        ply=int(data["ply"]),
        side=str(data["side"]),
        position_sfen=str(data["position_sfen"]),
        legal_moves=tuple(str(move) for move in cast(list[object], data["legal_moves"])),
        action_usi=str(data["action_usi"]),
        next_position_sfen=str(data["next_position_sfen"]),
        reward=float(data["reward"]),
        done=bool(data["done"]),
        decision_usi_info_lines=tuple(str(line) for line in cast(list[object], data.get("decision_usi_info_lines", []))),
    )


def _finalize_transitions(
    transitions: list[ShogiTransitionRecord],
    *,
    winner: str | None,
) -> list[ShogiTransitionRecord]:
    if not transitions:
        return []
    finalized = list(transitions)
    last = finalized[-1]
    finalized[-1] = ShogiTransitionRecord(
        ply=last.ply,
        side=last.side,
        position_sfen=last.position_sfen,
        legal_moves=last.legal_moves,
        action_usi=last.action_usi,
        next_position_sfen=last.next_position_sfen,
        reward=_transition_reward(side=last.side, winner=winner, done=True),
        done=True,
        decision_usi_info_lines=last.decision_usi_info_lines,
    )
    return finalized


def _transition_reward(*, side: str, winner: str | None, done: bool) -> float:
    if not done or winner is None:
        return 0.0
    return 1.0 if side == winner else -1.0


def _coerce_go_result(result: str | UsiGoResult) -> UsiGoResult:
    if isinstance(result, UsiGoResult):
        return result
    return UsiGoResult(bestmove=result)


def _actor_spec_from_json(data: dict[str, object]) -> ShogiActorSpec:
    settings = data.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("actor settings must be an object")
    return ShogiActorSpec(
        kind=str(data["kind"]),
        name=str(data["name"]),
        settings=cast(dict[str, str | int | float | bool | None], settings),
    )


def _object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected object")
    return value
