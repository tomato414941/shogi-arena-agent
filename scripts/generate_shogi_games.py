from __future__ import annotations

import argparse
import json
import shogi
from collections import Counter
from contextlib import ExitStack, nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

from shogi_arena_agent.mcts_policy import BatchedMctsMoveSelector, MctsConfig
from shogi_arena_agent.model_policy import ShogiMoveChoiceCheckpointEvaluator
from shogi_arena_agent.player_cli import BuiltPlayer, add_player_arguments, build_static_player, player_context, validate_player_arguments
from shogi_arena_agent.shogi_game import (
    ShogiActorSpec,
    ShogiGameRecord,
    ShogiTransitionRecord,
    play_shogi_game,
    position_command,
    save_shogi_game_records_jsonl,
)
from shogi_arena_agent.usi import UsiPosition


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play fixed-side shogi games and write raw game logs.")
    add_player_arguments(parser, "black")
    add_player_arguments(parser, "white")
    parser.add_argument("--out", required=True, help="Path to write one ShogiGameRecord JSON object per line.")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--parallel-games", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=80)
    args = parser.parse_args(argv)

    validate_player_arguments(parser, args, "black")
    validate_player_arguments(parser, args, "white")
    if args.games <= 0:
        parser.error("--games must be positive")
    if args.parallel_games <= 0:
        parser.error("--parallel-games must be positive")

    records = _play_games(args)
    save_shogi_game_records_jsonl(records, Path(args.out))
    print(json.dumps(_records_summary(records), indent=2))


def _play_games(args: argparse.Namespace) -> tuple[ShogiGameRecord, ...]:
    if args.parallel_games > 1:
        return _play_batched_checkpoint_mcts_games(args)
    records: list[ShogiGameRecord] = []
    black_static = build_static_player(args, "black", name="black")
    white_static = build_static_player(args, "white", name="white")
    with ExitStack() as stack:
        black = stack.enter_context(_player_context(args, "black", name="black", static_player=black_static))
        white = stack.enter_context(_player_context(args, "white", name="white", static_player=white_static))
        for _game_index in range(args.games):
            records.append(
                play_shogi_game(
                    black=black.player,
                    white=white.player,
                    black_actor=black.actor,
                    white_actor=white.actor,
                    max_plies=args.max_plies,
                )
            )
    return tuple(records)


def _play_batched_checkpoint_mcts_games(args: argparse.Namespace) -> tuple[ShogiGameRecord, ...]:
    _validate_batched_checkpoint_mcts_args(args)
    black_actor = _checkpoint_actor(args, "black", name="black")
    white_actor = _checkpoint_actor(args, "white", name="white")
    black_selector = _checkpoint_selector(args, "black")
    white_selector = _checkpoint_selector(args, "white")
    games = [_ActiveBatchedGame(black_actor=black_actor, white_actor=white_actor) for _ in range(args.games)]
    remaining = set(range(args.games))
    for _ply in range(args.max_plies):
        if not remaining:
            break
        black_indexes = [index for index in sorted(remaining) if games[index].board.turn == shogi.BLACK]
        white_indexes = [index for index in sorted(remaining) if games[index].board.turn == shogi.WHITE]
        for indexes, selector in ((black_indexes, black_selector), (white_indexes, white_selector)):
            for offset in range(0, len(indexes), args.parallel_games):
                batch_indexes = indexes[offset : offset + args.parallel_games]
                positions = [UsiPosition(position_command(games[index].moves)) for index in batch_indexes]
                results = selector.select_moves(positions)
                batch_info_lines = _batch_performance_info_lines(selector.last_batch_performance)
                for game_index, result in zip(batch_indexes, results, strict=True):
                    info_lines = _performance_info_lines(result.performance) + batch_info_lines
                    if game_index in remaining and games[game_index].apply_move(result.move, info_lines):
                        remaining.remove(game_index)
    return tuple(game.to_record() for game in games)


def _player_context(
    args: argparse.Namespace,
    prefix: str,
    *,
    name: str,
    static_player: BuiltPlayer | None,
):
    if static_player is not None:
        return nullcontext(static_player)
    return player_context(args, prefix, name=name)


class _ActiveBatchedGame:
    def __init__(self, *, black_actor: ShogiActorSpec, white_actor: ShogiActorSpec) -> None:
        self.board = shogi.Board()
        self.black_actor = black_actor
        self.white_actor = white_actor
        self.initial_position_sfen = self.board.sfen()
        self.transitions: list[ShogiTransitionRecord] = []
        self.end_reason = "max_plies"
        self.winner: str | None = None

    @property
    def moves(self) -> tuple[str, ...]:
        return tuple(transition.action_usi for transition in self.transitions)

    def apply_move(self, move: str, info_lines: tuple[str, ...]) -> bool:
        side = "black" if self.board.turn == shogi.BLACK else "white"
        legal_moves = tuple(sorted(legal_move.usi() for legal_move in self.board.legal_moves))
        position_sfen = self.board.sfen()
        if move == "resign" or move not in legal_moves:
            self.winner = "white" if self.board.turn == shogi.BLACK else "black"
            self.end_reason = "resign" if move == "resign" else "illegal_move"
            self._finalize_rewards()
            return True
        self.board.push_usi(move)
        done = self.board.is_game_over()
        self.winner = "black" if done and self.board.turn == shogi.WHITE else "white" if done else None
        self.transitions.append(
            ShogiTransitionRecord(
                ply=len(self.transitions),
                side=side,
                position_sfen=position_sfen,
                legal_moves=legal_moves,
                action_usi=move,
                next_position_sfen=self.board.sfen(),
                reward=_transition_reward(side=side, winner=self.winner, done=done),
                done=done,
                decision_usi_info_lines=info_lines,
            )
        )
        if done:
            self.end_reason = "game_over"
            return True
        return False

    def to_record(self) -> ShogiGameRecord:
        if self.end_reason == "max_plies":
            self._finalize_rewards()
        return ShogiGameRecord(
            black_actor=self.black_actor,
            white_actor=self.white_actor,
            initial_position_sfen=self.initial_position_sfen,
            transitions=tuple(self.transitions),
            end_reason=self.end_reason,
            winner=self.winner,
        )

    def _finalize_rewards(self) -> None:
        self.transitions = [
            ShogiTransitionRecord(
                ply=transition.ply,
                side=transition.side,
                position_sfen=transition.position_sfen,
                legal_moves=transition.legal_moves,
                action_usi=transition.action_usi,
                next_position_sfen=transition.next_position_sfen,
                reward=_transition_reward(side=transition.side, winner=self.winner, done=True),
                done=True,
                decision_usi_info_lines=transition.decision_usi_info_lines,
            )
            for transition in self.transitions
        ]


def _validate_batched_checkpoint_mcts_args(args: argparse.Namespace) -> None:
    for prefix in ("black", "white"):
        if getattr(args, f"{prefix}_kind") != "checkpoint":
            raise SystemExit("--parallel-games currently supports checkpoint-vs-checkpoint generation only")
        if getattr(args, f"{prefix}_checkpoint_policy") != "mcts":
            raise SystemExit("--parallel-games currently supports checkpoint MCTS players only")
        if getattr(args, f"{prefix}_checkpoint_move_time_limit_sec") is not None:
            raise SystemExit("--parallel-games does not support move time limits yet")


def _checkpoint_selector(args: argparse.Namespace, prefix: str) -> BatchedMctsMoveSelector:
    evaluator = ShogiMoveChoiceCheckpointEvaluator.from_checkpoint(
        getattr(args, f"{prefix}_checkpoint"),
        device=getattr(args, f"{prefix}_checkpoint_device"),
    )
    return BatchedMctsMoveSelector(
        evaluator=evaluator,
        config=MctsConfig(
            simulation_count=getattr(args, f"{prefix}_checkpoint_simulations"),
            evaluation_batch_size=getattr(args, f"{prefix}_checkpoint_evaluation_batch_size"),
        ),
    )


def _checkpoint_actor(args: argparse.Namespace, prefix: str, *, name: str) -> ShogiActorSpec:
    return ShogiActorSpec(
        kind="checkpoint",
        name=name,
        settings={
            "checkpoint": getattr(args, f"{prefix}_checkpoint"),
            "policy": getattr(args, f"{prefix}_checkpoint_policy"),
            "simulations": getattr(args, f"{prefix}_checkpoint_simulations"),
            "evaluation_batch_size": getattr(args, f"{prefix}_checkpoint_evaluation_batch_size"),
            "move_time_limit_sec": getattr(args, f"{prefix}_checkpoint_move_time_limit_sec"),
            "device": getattr(args, f"{prefix}_checkpoint_device"),
            "parallel_games": args.parallel_games,
        },
    )


def _performance_info_lines(performance: object) -> tuple[str, ...]:
    return ("info string intrep_performance " + json.dumps(asdict(performance), sort_keys=True),)


def _batch_performance_info_lines(performance: object | None) -> tuple[str, ...]:
    if performance is None:
        return ()
    return ("info string intrep_batch_performance " + json.dumps(asdict(performance), sort_keys=True),)


def _transition_reward(*, side: str, winner: str | None, done: bool) -> float:
    if not done or winner is None:
        return 0.0
    return 1.0 if side == winner else -1.0


def _records_summary(records: tuple[ShogiGameRecord, ...]) -> dict[str, Any]:
    end_reasons = Counter(record.end_reason for record in records)
    return {
        "game_count": len(records),
        "end_reasons": dict(end_reasons),
        "average_plies": sum(len(record.transitions) for record in records) / len(records) if records else 0.0,
        "black_wins": sum(1 for record in records if record.winner == "black"),
        "white_wins": sum(1 for record in records if record.winner == "white"),
        "draws": sum(1 for record in records if record.winner is None),
    }


if __name__ == "__main__":
    main()
