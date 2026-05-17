from __future__ import annotations

import argparse
import json
import subprocess
import sys
from contextlib import ExitStack, nullcontext
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from shogi_arena_agent.match_evaluation import summarize_match_results
from shogi_arena_agent.player_cli import BuiltPlayer, add_player_arguments, build_static_player, player_context, validate_player_arguments
from shogi_arena_agent.shogi_game import (
    ShogiGameRecord,
    load_shogi_game_records_jsonl,
    play_shogi_game,
    save_shogi_game_records_jsonl,
)

STANDARD_MAX_PLIES = 320
DEFAULT_MAX_PLIES = 320


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate two shogi players with alternating sides.")
    add_player_arguments(parser, "player-a")
    add_player_arguments(parser, "player-b")
    parser.add_argument("--out", required=True, help="Path to write one ShogiGameRecord JSON object per line.")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--match-worker-processes", type=int, default=1)
    parser.add_argument("--progress-every-games", type=int, default=0)
    parser.add_argument("--start-game-index", type=int, default=0, help=argparse.SUPPRESS)
    # Computer-shogi evaluation should not end as a short artificial draw; use
    # the WCSC-style 320-ply cap as the default and warn on shorter overrides.
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    args = parser.parse_args(argv)

    validate_player_arguments(parser, args, "player-a")
    validate_player_arguments(parser, args, "player-b")
    if args.games <= 0:
        parser.error("--games must be positive")
    if args.match_worker_processes <= 0:
        parser.error("--match-worker-processes must be positive")
    if args.progress_every_games < 0:
        parser.error("--progress-every-games must be non-negative")
    _warn_short_max_plies(args.max_plies)

    if args.match_worker_processes > 1:
        _run_sharded_match(args)
        return

    results, player_a_sides = _evaluate(args)
    evaluation = summarize_match_results(results, player_a_sides)
    save_shogi_game_records_jsonl(evaluation.results, Path(args.out))
    print(json.dumps(_evaluation_summary(evaluation), indent=2))


def _evaluate(args: argparse.Namespace) -> tuple[list[ShogiGameRecord], list[str]]:
    results: list[ShogiGameRecord] = []
    player_a_sides: list[str] = []
    player_a_static = build_static_player(args, "player-a", name="player_a")
    player_b_static = build_static_player(args, "player-b", name="player_b")
    with ExitStack() as stack:
        player_a = stack.enter_context(_player_context(args, "player-a", name="player_a", static_player=player_a_static))
        player_b = stack.enter_context(_player_context(args, "player-b", name="player_b", static_player=player_b_static))
        started_at = perf_counter()
        for offset in range(args.games):
            game_index = args.start_game_index + offset
            if game_index % 2 == 0:
                results.append(
                    play_shogi_game(
                        black=player_a.player,
                        white=player_b.player,
                        black_actor=player_a.actor,
                        white_actor=player_b.actor,
                        max_plies=args.max_plies,
                    )
                )
                player_a_sides.append("black")
            else:
                results.append(
                    play_shogi_game(
                        black=player_b.player,
                        white=player_a.player,
                        black_actor=player_b.actor,
                        white_actor=player_a.actor,
                        max_plies=args.max_plies,
                    )
                )
                player_a_sides.append("white")
            if args.progress_every_games and (offset + 1) % args.progress_every_games == 0:
                _print_match_progress(
                    results,
                    player_a_sides,
                    completed_games=offset + 1,
                    total_games=args.games,
                    elapsed_sec=perf_counter() - started_at,
                )
    return results, player_a_sides


def _run_sharded_match(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []
    shard_paths: list[Path] = []
    start_index = args.start_game_index
    for shard_index, shard_games in enumerate(_shard_game_counts(args.games, args.match_worker_processes)):
        if shard_games <= 0:
            continue
        shard_out = out.with_name(f"{out.stem}.shard-{shard_index:03d}{out.suffix}")
        shard_paths.append(shard_out)
        commands.append(
            _shard_command(
                args,
                shard_games=shard_games,
                shard_out=shard_out,
                start_game_index=start_index,
            )
        )
        start_index += shard_games

    processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None, text=True) for command in commands]
    failures: list[str] = []
    for command, process in zip(commands, processes, strict=True):
        stdout, _stderr = process.communicate()
        if process.returncode != 0:
            failures.append(f"{command!r} exited with {process.returncode}: {stdout}")
    if failures:
        raise SystemExit("; ".join(failures))

    results: list[ShogiGameRecord] = []
    player_a_sides: list[str] = []
    start_index = args.start_game_index
    for shard_path in shard_paths:
        records = list(load_shogi_game_records_jsonl(shard_path))
        results.extend(records)
        player_a_sides.extend(_player_a_sides(start_index=start_index, games=len(records)))
        start_index += len(records)
    save_shogi_game_records_jsonl(results, out)
    evaluation = summarize_match_results(results, player_a_sides)
    summary = _evaluation_summary(evaluation)
    summary["match_worker_processes"] = args.match_worker_processes
    print(json.dumps(summary, indent=2))


def _shard_game_counts(games: int, worker_processes: int) -> list[int]:
    return [games // worker_processes + (1 if index < games % worker_processes else 0) for index in range(worker_processes)]


def _shard_command(
    args: argparse.Namespace,
    *,
    shard_games: int,
    shard_out: Path,
    start_game_index: int,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        *_player_command_args(args, "player-a"),
        *_player_command_args(args, "player-b"),
        "--out",
        str(shard_out),
        "--games",
        str(shard_games),
        "--match-worker-processes",
        "1",
        "--progress-every-games",
        str(args.progress_every_games),
        "--start-game-index",
        str(start_game_index),
        "--max-plies",
        str(args.max_plies),
    ]


def _player_command_args(args: argparse.Namespace, prefix: str) -> list[str]:
    command = [
        f"--{prefix}-kind",
        getattr(args, f"{prefix.replace('-', '_')}_kind"),
    ]
    for name in (
        "checkpoint",
        "checkpoint_id",
        "move_selection_profile",
        "move_selector",
        "mcts_simulations",
        "mcts_evaluation_batch_size",
        "mcts_move_time_limit_sec",
        "mcts_root_reuse",
        "device",
        "board_backend",
        "usi_command",
        "usi_option",
        "usi_go_command",
        "usi_read_timeout_seconds",
        "usi_policy_target_multipv",
        "usi_policy_target_temperature_cp",
    ):
        value = getattr(args, f"{prefix.replace('-', '_')}_{name}")
        if value is None:
            continue
        flag = f"--{prefix}-{name.replace('_', '-')}"
        if isinstance(value, list | tuple):
            for item in value:
                command.extend([flag, str(item)])
            continue
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        command.extend([flag, str(value)])
    return command


def _player_a_sides(*, start_index: int, games: int) -> list[str]:
    return ["black" if (start_index + offset) % 2 == 0 else "white" for offset in range(games)]


def _print_match_progress(
    results: list[ShogiGameRecord],
    player_a_sides: list[str],
    *,
    completed_games: int,
    total_games: int,
    elapsed_sec: float,
) -> None:
    evaluation = summarize_match_results(results, player_a_sides)
    payload = {
        "completed_games": completed_games,
        "total_games": total_games,
        "elapsed_sec": elapsed_sec,
        "player_a_wins": evaluation.player_a_wins,
        "player_a_losses": evaluation.player_a_losses,
        "draws": evaluation.draws,
    }
    print("progress " + json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)


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


def _evaluation_summary(evaluation: Any) -> dict[str, Any]:
    data = asdict(evaluation)
    performance = _performance_summary(evaluation.results)
    if performance is not None:
        data["inference_performance"] = performance
    data.pop("results")
    return data


def _performance_summary(results: list[ShogiGameRecord]) -> dict[str, Any] | None:
    samples = [
        transition.decision_telemetry.move_performance
        for result in results
        for transition in result.transitions
        if transition.decision_telemetry is not None and transition.decision_telemetry.move_performance is not None
    ]
    if not samples:
        return None
    request_times = [sample["request_wall_time_sec"] for sample in samples]
    model_times = [sample["model_wall_time_sec"] for sample in samples]
    non_model_times = [sample["non_model_wall_time_sec"] for sample in samples]
    output_counts = [sample["output_count"] for sample in samples]
    output_rates = [sample["output_per_sec"] for sample in samples]
    model_call_counts = [sample["model_call_count"] for sample in samples]
    summary = {
        "request_count": len(samples),
        "request_wall_time_sec_avg": mean(request_times),
        "request_wall_time_sec_p95": _percentile(request_times, 0.95),
        "request_wall_time_sec_max": max(request_times),
        "model_call_count_avg": mean(model_call_counts),
        "model_wall_time_sec_avg": mean(model_times),
        "non_model_wall_time_sec_avg": mean(non_model_times),
        "output_count_avg": mean(output_counts),
        "output_per_sec_avg": mean(output_rates),
    }
    _add_actual_leaf_eval_batch_summary(summary, samples)
    return summary


def _add_actual_leaf_eval_batch_summary(summary: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    avg_values = [
        sample["actual_nn_leaf_eval_batch_size_avg"]
        for sample in samples
        if isinstance(sample.get("actual_nn_leaf_eval_batch_size_avg"), int | float)
    ]
    max_values = [
        sample["actual_nn_leaf_eval_batch_size_max"]
        for sample in samples
        if isinstance(sample.get("actual_nn_leaf_eval_batch_size_max"), int | float)
    ]
    count_values = [
        sample["actual_nn_leaf_eval_batch_count"]
        for sample in samples
        if isinstance(sample.get("actual_nn_leaf_eval_batch_count"), int | float)
    ]
    fill_ratio_values = [
        sample["actual_nn_leaf_eval_batch_size_fill_ratio_avg"]
        for sample in samples
        if isinstance(sample.get("actual_nn_leaf_eval_batch_size_fill_ratio_avg"), int | float)
    ]
    histogram: dict[int, int] = {}
    for sample in samples:
        sample_histogram = sample.get("actual_nn_leaf_eval_batch_size_histogram")
        if not isinstance(sample_histogram, dict):
            continue
        for size, count in sample_histogram.items():
            if not isinstance(count, int):
                continue
            histogram[int(size)] = histogram.get(int(size), 0) + count
    if not avg_values or not max_values:
        return
    summary["actual_nn_leaf_eval_batch_size_avg"] = mean(avg_values)
    summary["actual_nn_leaf_eval_batch_size_max"] = max(max_values)
    if count_values:
        summary["actual_nn_leaf_eval_batch_count_avg"] = mean(count_values)
        summary["actual_nn_leaf_eval_batch_count_max"] = max(count_values)
    if fill_ratio_values:
        summary["actual_nn_leaf_eval_batch_size_fill_ratio_avg"] = mean(fill_ratio_values)
    if histogram:
        summary["actual_nn_leaf_eval_batch_size_histogram"] = dict(sorted(histogram.items()))


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, int(round(fraction * (len(sorted_values) - 1)))))
    return sorted_values[index]


def _warn_short_max_plies(max_plies: int) -> None:
    if max_plies < STANDARD_MAX_PLIES:
        print(
            f"warning: --max-plies {max_plies} is below the computer-shogi standard cap "
            f"of {STANDARD_MAX_PLIES}; this can create artificial max_plies draws.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
