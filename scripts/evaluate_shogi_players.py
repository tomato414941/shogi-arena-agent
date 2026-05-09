from __future__ import annotations

import argparse
import json
from contextlib import ExitStack, nullcontext
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

from shogi_arena_agent.match_evaluation import summarize_match_results
from shogi_arena_agent.player_cli import BuiltPlayer, add_player_arguments, build_static_player, player_context, validate_player_arguments
from shogi_arena_agent.shogi_game import ShogiGameRecord, play_shogi_game, save_shogi_game_records_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate two shogi players with alternating sides.")
    add_player_arguments(parser, "player")
    add_player_arguments(parser, "opponent")
    parser.add_argument("--out", required=True, help="Path to write one ShogiGameRecord JSON object per line.")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=80)
    args = parser.parse_args(argv)

    validate_player_arguments(parser, args, "player")
    validate_player_arguments(parser, args, "opponent")
    if args.games <= 0:
        parser.error("--games must be positive")

    results, player_sides = _evaluate(args)
    evaluation = summarize_match_results(results, player_sides)
    save_shogi_game_records_jsonl(evaluation.results, Path(args.out))
    print(json.dumps(_evaluation_summary(evaluation), indent=2))


def _evaluate(args: argparse.Namespace) -> tuple[list[ShogiGameRecord], list[str]]:
    results: list[ShogiGameRecord] = []
    player_sides: list[str] = []
    player_static = build_static_player(args, "player", name="player")
    opponent_static = build_static_player(args, "opponent", name="opponent")
    with ExitStack() as stack:
        player = stack.enter_context(_player_context(args, "player", name="player", static_player=player_static))
        opponent = stack.enter_context(_player_context(args, "opponent", name="opponent", static_player=opponent_static))
        for game_index in range(args.games):
            if game_index % 2 == 0:
                results.append(
                    play_shogi_game(
                        black=player.player,
                        white=opponent.player,
                        black_actor=player.actor,
                        white_actor=opponent.actor,
                        max_plies=args.max_plies,
                    )
                )
                player_sides.append("black")
            else:
                results.append(
                    play_shogi_game(
                        black=opponent.player,
                        white=player.player,
                        black_actor=opponent.actor,
                        white_actor=player.actor,
                        max_plies=args.max_plies,
                    )
                )
                player_sides.append("white")
    return results, player_sides


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
        sample
        for result in results
        for transition in result.transitions
        for sample in _transition_performance_samples(transition.decision_usi_info_lines)
    ]
    if not samples:
        return None
    request_times = [sample["request_wall_time_sec"] for sample in samples]
    model_times = [sample["model_wall_time_sec"] for sample in samples]
    non_model_times = [sample["non_model_wall_time_sec"] for sample in samples]
    output_counts = [sample["output_count"] for sample in samples]
    output_rates = [sample["output_per_sec"] for sample in samples]
    model_call_counts = [sample["model_call_count"] for sample in samples]
    return {
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


def _transition_performance_samples(info_lines: tuple[str, ...]) -> list[dict[str, float]]:
    prefix = "info string intrep_performance "
    samples: list[dict[str, float]] = []
    for line in info_lines:
        if line.startswith(prefix):
            samples.append({key: float(value) for key, value in json.loads(line[len(prefix) :]).items()})
    return samples


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, int(round(fraction * (len(sorted_values) - 1)))))
    return sorted_values[index]


if __name__ == "__main__":
    main()
