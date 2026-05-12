from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from contextlib import ExitStack, nullcontext
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from shogi_arena_agent.board_backend import board_is_black_turn, board_turn_name, legal_move_usis, new_board
from shogi_arena_agent.mcts_policy import (
    BatchedMctsMoveSelector,
    MctsConfig,
    evaluation_move_selection_config,
    self_play_move_selection_config,
)
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
from shogi_arena_agent.usi import BOARD_BACKENDS, UsiPosition

STANDARD_MAX_PLIES = 320
DEFAULT_MAX_PLIES = 320


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play fixed-side shogi games and write raw game logs.")
    add_player_arguments(parser, "black")
    add_player_arguments(parser, "white")
    parser.add_argument("--out", required=True, help="Path to write one ShogiGameRecord JSON object per line.")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--parallel-games", type=int, default=1)
    parser.add_argument("--progress-every-plies", type=int, default=0)
    # Computer-shogi self-play should not end as a short artificial draw; use
    # the WCSC-style 320-ply cap as the default and warn on shorter overrides.
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    parser.add_argument("--board-backend", choices=BOARD_BACKENDS, default="python-shogi")
    args = parser.parse_args(argv)

    validate_player_arguments(parser, args, "black")
    validate_player_arguments(parser, args, "white")
    if args.games <= 0:
        parser.error("--games must be positive")
    if args.parallel_games <= 0:
        parser.error("--parallel-games must be positive")
    if args.progress_every_plies < 0:
        parser.error("--progress-every-plies must be non-negative")
    _warn_short_max_plies(args.max_plies)

    started_at = perf_counter()
    records = _play_games(args)
    save_shogi_game_records_jsonl(records, Path(args.out))
    print(json.dumps(_records_summary(records, wall_time_sec=perf_counter() - started_at), indent=2))


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
                    board_backend=args.board_backend,
                )
            )
    return tuple(records)


def _play_batched_checkpoint_mcts_games(args: argparse.Namespace) -> tuple[ShogiGameRecord, ...]:
    _validate_batched_checkpoint_mcts_args(args)
    black_actor = _checkpoint_actor(args, "black", name="black")
    white_actor = _checkpoint_actor(args, "white", name="white")
    black_selector = _checkpoint_selector(args, "black")
    white_selector = _checkpoint_selector(args, "white")
    games = [
        _ActiveBatchedGame(black_actor=black_actor, white_actor=white_actor, board_backend=args.board_backend)
        for _ in range(args.games)
    ]
    remaining = set(range(args.games))
    started_at = perf_counter()
    for ply in range(args.max_plies):
        if not remaining:
            break
        black_indexes = [index for index in sorted(remaining) if board_is_black_turn(games[index].board)]
        white_indexes = [index for index in sorted(remaining) if not board_is_black_turn(games[index].board)]
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
        if args.progress_every_plies and (ply + 1) % args.progress_every_plies == 0:
            _print_progress(games, remaining=remaining, ply=ply + 1, elapsed_sec=perf_counter() - started_at)
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
    def __init__(self, *, black_actor: ShogiActorSpec, white_actor: ShogiActorSpec, board_backend: str) -> None:
        self.board = new_board(backend=board_backend)
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
        side = board_turn_name(self.board)
        legal_moves = legal_move_usis(self.board)
        position_sfen = self.board.sfen()
        if move == "resign" or move not in legal_moves:
            self.winner = "white" if board_is_black_turn(self.board) else "black"
            self.end_reason = "resign" if move == "resign" else "illegal_move"
            self._finalize_rewards()
            return True
        self.board.push_usi(move)
        done = self.board.is_game_over()
        self.winner = "black" if done and not board_is_black_turn(self.board) else "white" if done else None
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
            board_backend=args.board_backend,
        ),
        move_selection=_move_selection_config(getattr(args, f"{prefix}_checkpoint_profile")),
    )


def _checkpoint_actor(args: argparse.Namespace, prefix: str, *, name: str) -> ShogiActorSpec:
    return ShogiActorSpec(
        kind="checkpoint",
        name=name,
        settings={
            "checkpoint": getattr(args, f"{prefix}_checkpoint"),
            "profile": getattr(args, f"{prefix}_checkpoint_profile"),
            "policy": getattr(args, f"{prefix}_checkpoint_policy"),
            "simulations": getattr(args, f"{prefix}_checkpoint_simulations"),
            "evaluation_batch_size": getattr(args, f"{prefix}_checkpoint_evaluation_batch_size"),
            "move_time_limit_sec": getattr(args, f"{prefix}_checkpoint_move_time_limit_sec"),
            "device": getattr(args, f"{prefix}_checkpoint_device"),
            "parallel_games": args.parallel_games,
            "board_backend": args.board_backend,
        },
    )


def _move_selection_config(profile: str):
    if profile == "self-play":
        return self_play_move_selection_config()
    return evaluation_move_selection_config()


def _performance_info_lines(performance: object) -> tuple[str, ...]:
    return ("info string intrep_performance " + json.dumps(asdict(performance), sort_keys=True),)


def _batch_performance_info_lines(performance: object | None) -> tuple[str, ...]:
    if performance is None:
        return ()
    return ("info string intrep_batch_performance " + json.dumps(asdict(performance), sort_keys=True),)


def _print_progress(
    games: list[_ActiveBatchedGame],
    *,
    remaining: set[int],
    ply: int,
    elapsed_sec: float,
) -> None:
    completed = len(games) - len(remaining)
    active_plies = [len(games[index].transitions) for index in remaining]
    payload = {
        "ply": ply,
        "elapsed_sec": elapsed_sec,
        "completed_games": completed,
        "remaining_games": len(remaining),
        "active_plies_avg": mean(active_plies) if active_plies else 0.0,
        "active_plies_max": max(active_plies) if active_plies else 0,
    }
    print("progress " + json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)


def _transition_reward(*, side: str, winner: str | None, done: bool) -> float:
    if not done or winner is None:
        return 0.0
    return 1.0 if side == winner else -1.0


def _records_summary(records: tuple[ShogiGameRecord, ...], *, wall_time_sec: float | None = None) -> dict[str, Any]:
    end_reasons = Counter(record.end_reason for record in records)
    summary: dict[str, Any] = {
        "game_count": len(records),
        "end_reasons": dict(end_reasons),
        "average_plies": sum(len(record.transitions) for record in records) / len(records) if records else 0.0,
        "black_wins": sum(1 for record in records if record.winner == "black"),
        "white_wins": sum(1 for record in records if record.winner == "white"),
        "draws": sum(1 for record in records if record.winner is None),
    }
    if wall_time_sec is not None:
        summary["generation_wall_time_sec"] = wall_time_sec
        total_plies = sum(len(record.transitions) for record in records)
        summary["plies_per_sec"] = total_plies / wall_time_sec if wall_time_sec > 0.0 else 0.0
    inference_performance = _performance_summary(records, prefix="info string intrep_performance ")
    if inference_performance is not None:
        summary["inference_performance"] = inference_performance
    batch_performance = _performance_summary(records, prefix="info string intrep_batch_performance ")
    if batch_performance is not None:
        summary["batch_performance"] = batch_performance
    return summary


def _performance_summary(records: tuple[ShogiGameRecord, ...], *, prefix: str) -> dict[str, Any] | None:
    samples = [
        sample
        for record in records
        for transition in record.transitions
        for sample in _transition_performance_samples(transition.decision_usi_info_lines, prefix=prefix)
    ]
    if not samples:
        return None
    summary: dict[str, Any] = {"sample_count": len(samples)}
    for key in (
        "request_wall_time_sec",
        "model_call_count",
        "model_wall_time_sec",
        "non_model_wall_time_sec",
        "output_count",
        "output_per_sec",
        "position_count",
        "completed_simulations",
    ):
        values = [sample[key] for sample in samples if isinstance(sample.get(key), int | float)]
        if values:
            summary[f"{key}_avg"] = mean(values)
            summary[f"{key}_max"] = max(values)
    phase_totals: dict[str, float] = {}
    for sample in samples:
        phase_times = sample.get("phase_wall_time_sec")
        if not isinstance(phase_times, dict):
            continue
        for name, elapsed in phase_times.items():
            if isinstance(elapsed, int | float):
                phase_totals[name] = phase_totals.get(name, 0.0) + float(elapsed)
    if phase_totals:
        summary["phase_wall_time_sec_total"] = dict(sorted(phase_totals.items()))
        summary["phase_wall_time_sec_avg"] = {
            name: elapsed / len(samples) for name, elapsed in sorted(phase_totals.items())
        }
    return summary


def _transition_performance_samples(info_lines: tuple[str, ...], *, prefix: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line in info_lines:
        if line.startswith(prefix):
            payload = json.loads(line[len(prefix) :])
            if isinstance(payload, dict):
                samples.append(payload)
    return samples


def _warn_short_max_plies(max_plies: int) -> None:
    if max_plies < STANDARD_MAX_PLIES:
        print(
            f"warning: --max-plies {max_plies} is below the computer-shogi standard cap "
            f"of {STANDARD_MAX_PLIES}; this can create artificial max_plies draws.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
