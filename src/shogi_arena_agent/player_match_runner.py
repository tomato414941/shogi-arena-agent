from __future__ import annotations

import json
import subprocess
import sys
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

from shogi_arena_agent.match_evaluation import MatchEvaluation, summarize_match_results
from shogi_arena_agent.player_cli import (
    BuiltPlayer,
    CheckpointPolicyPlayerSpec,
    DeterministicLegalPlayerSpec,
    ExternalEnginePlayerSpec,
    PlayerSpec,
    build_static_player,
    player_context,
)
from shogi_arena_agent.shogi_game import (
    ShogiGameRecord,
    load_shogi_game_records_jsonl,
    play_shogi_game,
    save_shogi_game_records_jsonl,
)


@dataclass(frozen=True)
class PlayerMatchRunConfig:
    player_a: PlayerSpec
    player_b: PlayerSpec
    games: int
    max_plies: int
    progress_every_games: int = 0
    start_game_index: int = 0


@dataclass(frozen=True)
class ShardedPlayerMatchConfig:
    match: PlayerMatchRunConfig
    worker_processes: int
    out: Path
    script_path: Path


ProgressCallback = Callable[[dict[str, object]], None]


def run_player_match(config: PlayerMatchRunConfig, *, progress_callback: ProgressCallback | None = None) -> MatchEvaluation:
    results: list[ShogiGameRecord] = []
    player_a_sides: list[str] = []
    player_a_static = build_static_player(config.player_a, name="player_a")
    player_b_static = build_static_player(config.player_b, name="player_b")
    with ExitStack() as stack:
        player_a = stack.enter_context(_player_context(config.player_a, name="player_a", static_player=player_a_static))
        player_b = stack.enter_context(_player_context(config.player_b, name="player_b", static_player=player_b_static))
        started_at = perf_counter()
        for offset in range(config.games):
            game_index = config.start_game_index + offset
            if game_index % 2 == 0:
                results.append(
                    play_shogi_game(
                        black=player_a.player,
                        white=player_b.player,
                        black_actor=player_a.actor,
                        white_actor=player_b.actor,
                        max_plies=config.max_plies,
                    )
                )
            else:
                results.append(
                    play_shogi_game(
                        black=player_b.player,
                        white=player_a.player,
                        black_actor=player_b.actor,
                        white_actor=player_a.actor,
                        max_plies=config.max_plies,
                    )
                )
            player_a_sides.append(_player_a_side(game_index))
            if config.progress_every_games and (offset + 1) % config.progress_every_games == 0:
                evaluation = summarize_match_results(results, player_a_sides)
                _emit_progress(
                    evaluation,
                    completed_games=offset + 1,
                    total_games=config.games,
                    elapsed_sec=perf_counter() - started_at,
                    progress_callback=progress_callback,
                )
    return summarize_match_results(results, player_a_sides)


def run_sharded_player_match(config: ShardedPlayerMatchConfig) -> MatchEvaluation:
    config.out.parent.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []
    shard_paths: list[Path] = []
    start_index = config.match.start_game_index
    for shard_index, shard_games in enumerate(_shard_game_counts(config.match.games, config.worker_processes)):
        if shard_games <= 0:
            continue
        shard_out = config.out.with_name(f"{config.out.stem}.shard-{shard_index:03d}{config.out.suffix}")
        shard_paths.append(shard_out)
        commands.append(
            _shard_command(
                config.match,
                script_path=config.script_path,
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
        raise RuntimeError("; ".join(failures))

    results: list[ShogiGameRecord] = []
    player_a_sides: list[str] = []
    start_index = config.match.start_game_index
    for shard_path in shard_paths:
        records = list(load_shogi_game_records_jsonl(shard_path))
        results.extend(records)
        player_a_sides.extend(_player_a_sides(start_index=start_index, games=len(records)))
        start_index += len(records)
    save_shogi_game_records_jsonl(results, config.out)
    return summarize_match_results(results, player_a_sides)


def print_progress(payload: dict[str, object]) -> None:
    print("progress " + json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)


def _player_context(
    spec: PlayerSpec,
    *,
    name: str,
    static_player: BuiltPlayer | None,
):
    if static_player is not None:
        return nullcontext(static_player)
    return player_context(spec, name=name)


def _shard_game_counts(games: int, worker_processes: int) -> list[int]:
    return [games // worker_processes + (1 if index < games % worker_processes else 0) for index in range(worker_processes)]


def _shard_command(
    config: PlayerMatchRunConfig,
    *,
    script_path: Path,
    shard_games: int,
    shard_out: Path,
    start_game_index: int,
) -> list[str]:
    return [
        sys.executable,
        str(script_path),
        *_player_command_args(config.player_a, "player-a"),
        *_player_command_args(config.player_b, "player-b"),
        "--out",
        str(shard_out),
        "--games",
        str(shard_games),
        "--match-worker-processes",
        "1",
        "--progress-every-games",
        str(config.progress_every_games),
        "--start-game-index",
        str(start_game_index),
        "--max-plies",
        str(config.max_plies),
    ]


def _player_command_args(spec: PlayerSpec, prefix: str) -> list[str]:
    if isinstance(spec, CheckpointPolicyPlayerSpec):
        return _command_args(
            prefix,
            "checkpoint",
            {
                "checkpoint": spec.checkpoint,
                "checkpoint_id": spec.checkpoint_id,
                "move_selection_profile": spec.move_selection_profile,
                "move_selector": spec.move_selector,
                "mcts_simulations": spec.mcts_simulations,
                "mcts_evaluation_batch_size": spec.mcts_evaluation_batch_size,
                "mcts_move_time_limit_sec": spec.mcts_move_time_limit_sec,
                "mcts_root_reuse": spec.mcts_root_reuse,
                "device": spec.device,
                "board_backend": spec.board_backend,
            },
        )
    if isinstance(spec, ExternalEnginePlayerSpec):
        return _command_args(
            prefix,
            "usi_engine",
            {
                "usi_command": spec.command,
                "usi_option": spec.usi_option,
                "usi_go_command": spec.usi_go_command,
                "usi_read_timeout_seconds": spec.usi_read_timeout_seconds,
                "usi_policy_target_multipv": spec.usi_policy_target_multipv,
                "usi_policy_target_temperature_cp": spec.usi_policy_target_temperature_cp,
            },
        )
    if isinstance(spec, DeterministicLegalPlayerSpec):
        return [f"--{prefix}-kind", "deterministic_legal"]
    raise TypeError(f"unsupported player spec: {type(spec).__name__}")


def _command_args(prefix: str, kind: str, fields: dict[str, object]) -> list[str]:
    command = [f"--{prefix}-kind", kind]
    for name, value in fields.items():
        if value is None:
            continue
        flag = f"--{prefix}-{name.replace('_', '-')}"
        if isinstance(value, tuple):
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
    return [_player_a_side(start_index + offset) for offset in range(games)]


def _player_a_side(game_index: int) -> str:
    return "black" if game_index % 2 == 0 else "white"


def _emit_progress(
    evaluation: MatchEvaluation,
    *,
    completed_games: int,
    total_games: int,
    elapsed_sec: float,
    progress_callback: ProgressCallback | None,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        {
            "completed_games": completed_games,
            "total_games": total_games,
            "elapsed_sec": elapsed_sec,
            "player_a_wins": evaluation.player_a_wins,
            "player_a_losses": evaluation.player_a_losses,
            "draws": evaluation.draws,
        }
    )
