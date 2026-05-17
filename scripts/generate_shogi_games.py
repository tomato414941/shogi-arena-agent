from __future__ import annotations

import argparse
import json
import sys
import subprocess
from pathlib import Path
from time import perf_counter

from shogi_arena_agent.model_policy import ShogiMoveChoiceCheckpointEvaluator
from shogi_arena_agent.player_cli import PlayerSpec, add_player_arguments, player_spec_from_args, validate_player_arguments
from shogi_arena_agent.shogi_game import save_shogi_game_records_jsonl
from shogi_arena_agent.shogi_generation import (
    ShogiGenerationConfig,
    generate_shogi_games,
    records_summary,
)
from shogi_arena_agent.usi import BOARD_BACKENDS

STANDARD_MAX_PLIES = 320
DEFAULT_MAX_PLIES = 320


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play fixed-side shogi games and write raw game logs.")
    add_player_arguments(parser, "black")
    add_player_arguments(parser, "white")
    parser.add_argument("--out", required=True, help="Path to write one ShogiGameRecord JSON object per line.")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--concurrent-games-per-process", type=int, default=1)
    parser.add_argument("--generation-worker-processes", type=int, default=1)
    parser.add_argument("--seed", type=int)
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
    if args.concurrent_games_per_process <= 0:
        parser.error("--concurrent-games-per-process must be positive")
    if args.generation_worker_processes <= 0:
        parser.error("--generation-worker-processes must be positive")
    if args.progress_every_plies < 0:
        parser.error("--progress-every-plies must be non-negative")
    _warn_short_max_plies(args.max_plies)

    if args.generation_worker_processes > 1:
        _run_sharded_generation(args)
        return

    started_at = perf_counter()
    records = generate_shogi_games(
        _generation_config_from_args(args),
        checkpoint_evaluator_cls=ShogiMoveChoiceCheckpointEvaluator,
    )
    save_shogi_game_records_jsonl(records, Path(args.out))
    print(json.dumps(records_summary(records, wall_time_sec=perf_counter() - started_at), indent=2))


def _generation_config_from_args(args: argparse.Namespace) -> ShogiGenerationConfig:
    return ShogiGenerationConfig(
        black=_player_config_from_args(args, "black"),
        white=_player_config_from_args(args, "white"),
        games=args.games,
        concurrent_games_per_process=args.concurrent_games_per_process,
        max_plies=args.max_plies,
        board_backend=args.board_backend,
        progress_every_plies=args.progress_every_plies,
    )


def _player_config_from_args(args: argparse.Namespace, prefix: str) -> PlayerSpec:
    return player_spec_from_args(args, prefix, seed=args.seed)


def _warn_short_max_plies(max_plies: int) -> None:
    if max_plies < STANDARD_MAX_PLIES:
        print(
            f"warning: --max-plies {max_plies} is below the computer-shogi standard cap "
            f"of {STANDARD_MAX_PLIES}; this can create artificial max_plies draws.",
            file=sys.stderr,
        )


def _run_sharded_generation(args: argparse.Namespace) -> None:
    started_at = perf_counter()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shard_paths: list[Path] = []
    commands: list[list[str]] = []
    summaries: list[dict[str, object]] = []
    for shard_index, shard_games in enumerate(_shard_game_counts(args.games, args.generation_worker_processes)):
        if shard_games <= 0:
            continue
        shard_out = out.with_name(f"{out.stem}.shard-{shard_index:03d}{out.suffix}")
        shard_paths.append(shard_out)
        command = _shard_command(args, shard_index=shard_index, shard_games=shard_games, shard_out=shard_out)
        commands.append(command)
    processes = [
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None, text=True)
        for command in commands
    ]
    failures: list[str] = []
    for command, process in zip(commands, processes, strict=True):
        stdout, _stderr = process.communicate()
        if process.returncode != 0:
            failures.append(f"{command!r} exited with {process.returncode}")
            continue
        summaries.append(json.loads(stdout))
    if failures:
        raise SystemExit("; ".join(failures))
    with out.open("w", encoding="utf-8") as merged:
        for shard_path in shard_paths:
            merged.write(shard_path.read_text(encoding="utf-8"))
    print(json.dumps(_aggregate_shard_summaries(summaries, wall_time_sec=perf_counter() - started_at, args=args), indent=2))


def _shard_game_counts(games: int, worker_processes: int) -> list[int]:
    return [games // worker_processes + (1 if index < games % worker_processes else 0) for index in range(worker_processes)]


def _shard_command(
    args: argparse.Namespace,
    *,
    shard_index: int,
    shard_games: int,
    shard_out: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *_player_command_args(args, "black"),
        *_player_command_args(args, "white"),
        "--out",
        str(shard_out),
        "--games",
        str(shard_games),
        "--concurrent-games-per-process",
        str(args.concurrent_games_per_process),
        "--generation-worker-processes",
        "1",
        "--progress-every-plies",
        str(args.progress_every_plies),
        "--max-plies",
        str(args.max_plies),
        "--board-backend",
        args.board_backend,
    ]
    if args.seed is not None:
        command.extend(["--seed", str(args.seed + shard_index)])
    return command


def _player_command_args(args: argparse.Namespace, prefix: str) -> list[str]:
    command = [
        f"--{prefix}-kind",
        getattr(args, f"{prefix}_kind"),
    ]
    for name in (
        "checkpoint",
        "checkpoint_id",
        "move_selection_profile",
        "move_selection_temperature",
        "move_selection_temperature_plies",
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
        value = getattr(args, f"{prefix}_{name}")
        if value is None:
            continue
        if isinstance(value, list | tuple):
            for item in value:
                command.extend([f"--{prefix}-{name.replace('_', '-')}", str(item)])
            continue
        if isinstance(value, bool):
            if value:
                command.append(f"--{prefix}-{name.replace('_', '-')}")
            continue
        command.extend([f"--{prefix}-{name.replace('_', '-')}", str(value)])
    return command


def _aggregate_shard_summaries(
    summaries: list[dict[str, object]],
    *,
    wall_time_sec: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    total_games = sum(int(summary.get("game_count", 0)) for summary in summaries)
    total_plies = sum(float(summary.get("average_plies", 0.0)) * int(summary.get("game_count", 0)) for summary in summaries)
    end_reasons: dict[str, int] = {}
    for summary in summaries:
        for reason, count in dict(summary.get("end_reasons", {})).items():
            end_reasons[str(reason)] = end_reasons.get(str(reason), 0) + int(count)
    aggregate = {
        "game_count": total_games,
        "end_reasons": end_reasons,
        "average_plies": total_plies / total_games if total_games else 0.0,
        "black_wins": sum(int(summary.get("black_wins", 0)) for summary in summaries),
        "white_wins": sum(int(summary.get("white_wins", 0)) for summary in summaries),
        "draws": sum(int(summary.get("draws", 0)) for summary in summaries),
        "generation_wall_time_sec": wall_time_sec,
        "plies_per_sec": total_plies / wall_time_sec if wall_time_sec > 0.0 else 0.0,
        "generation_worker_processes": args.generation_worker_processes,
        "concurrent_games_per_process": args.concurrent_games_per_process,
        "seed": args.seed,
        "shards": summaries,
    }
    for name in ("inference_performance", "batch_performance"):
        performance = _aggregate_performance_summaries(
            [summary[name] for summary in summaries if isinstance(summary.get(name), dict)]
        )
        if performance is not None:
            aggregate[name] = performance
    return aggregate


def _aggregate_performance_summaries(summaries: list[object]) -> dict[str, object] | None:
    typed = [summary for summary in summaries if isinstance(summary, dict)]
    sample_count = sum(int(summary.get("sample_count", 0)) for summary in typed)
    if sample_count == 0:
        return None
    aggregate: dict[str, object] = {"sample_count": sample_count}
    numeric_keys = {
        key
        for summary in typed
        for key, value in summary.items()
        if isinstance(value, int | float) and key != "sample_count"
    }
    for key in sorted(numeric_keys):
        values = [(int(summary.get("sample_count", 0)), float(summary[key])) for summary in typed if isinstance(summary.get(key), int | float)]
        if not values:
            continue
        if key.endswith("_max"):
            aggregate[key] = max(value for _count, value in values)
        elif key.endswith("_avg"):
            total_weight = sum(count for count, _value in values)
            aggregate[key] = sum(count * value for count, value in values) / total_weight if total_weight else 0.0
    phase_totals: dict[str, float] = {}
    for summary in typed:
        phase_total = summary.get("phase_wall_time_sec_total")
        if not isinstance(phase_total, dict):
            continue
        for phase, elapsed in phase_total.items():
            if isinstance(elapsed, int | float):
                phase_totals[str(phase)] = phase_totals.get(str(phase), 0.0) + float(elapsed)
    if phase_totals:
        aggregate["phase_wall_time_sec_total"] = dict(sorted(phase_totals.items()))
        aggregate["phase_wall_time_sec_avg"] = {
            phase: elapsed / sample_count for phase, elapsed in sorted(phase_totals.items())
        }
    return aggregate


if __name__ == "__main__":
    main()
