from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

from shogi_arena_agent.player_cli import PlayerSpec, add_player_arguments, player_spec_from_args, validate_player_arguments
from shogi_arena_agent.player_match_runner import (
    PlayerMatchRunConfig,
    ShardedPlayerMatchConfig,
    print_progress,
    run_player_match,
    run_sharded_player_match,
)
from shogi_arena_agent.shogi_game import (
    ShogiGameRecord,
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

    config = _match_config_from_args(args)
    if args.match_worker_processes > 1:
        evaluation = run_sharded_player_match(
            ShardedPlayerMatchConfig(
                match=config,
                worker_processes=args.match_worker_processes,
                out=Path(args.out),
                script_path=Path(__file__).resolve(),
            )
        )
        summary = _evaluation_summary(evaluation)
        summary["match_worker_processes"] = args.match_worker_processes
        print(json.dumps(summary, indent=2))
        return

    evaluation = run_player_match(config, progress_callback=print_progress)
    save_shogi_game_records_jsonl(evaluation.results, Path(args.out))
    print(json.dumps(_evaluation_summary(evaluation), indent=2))


def _match_config_from_args(args: argparse.Namespace) -> PlayerMatchRunConfig:
    return PlayerMatchRunConfig(
        player_a=_player_config_from_args(args, "player-a"),
        player_b=_player_config_from_args(args, "player-b"),
        games=args.games,
        max_plies=args.max_plies,
        progress_every_games=args.progress_every_games,
        start_game_index=args.start_game_index,
    )


def _player_config_from_args(args: argparse.Namespace, prefix: str) -> PlayerSpec:
    return player_spec_from_args(args, prefix)


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
