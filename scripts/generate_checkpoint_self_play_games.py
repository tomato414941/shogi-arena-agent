from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from shogi_arena_agent.checkpoint_self_play_generation import (
    CheckpointSelfPlayConfig,
    run_checkpoint_self_play_generation,
    summarize_checkpoint_self_play_records,
)
from shogi_arena_agent.generated_game_artifacts import GeneratedGameArtifacts
from shogi_arena_agent.mcts_config import visit_sampling_move_selection_config
from shogi_arena_agent.model_policy import ShogiMoveChoiceCheckpointEvaluator
from shogi_arena_agent.usi import BOARD_BACKENDS

STANDARD_MAX_PLIES = 320
DEFAULT_MAX_PLIES = 320
DEFAULT_MOVE_SELECTION_TEMPERATURE = 1.0
DEFAULT_MOVE_SELECTION_TEMPERATURE_PLIES = 40


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate checkpoint self-play shogi game records.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--out", required=True)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--self-play-worker-processes", type=int, default=1)
    parser.add_argument("--concurrent-games-per-process", type=int, default=1)
    parser.add_argument("--mcts-simulations", type=int, default=128)
    parser.add_argument("--mcts-nn-leaf-eval-batch-limit", type=int, default=64)
    parser.add_argument("--central-evaluator-batch-size-limit", type=int)
    parser.add_argument("--central-evaluator-flush-timeout-sec", type=float, default=0.002)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--inference-precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--board-backend", choices=BOARD_BACKENDS, default="python-shogi")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--move-selection-temperature", type=float, default=DEFAULT_MOVE_SELECTION_TEMPERATURE)
    parser.add_argument("--move-selection-temperature-plies", type=int, default=DEFAULT_MOVE_SELECTION_TEMPERATURE_PLIES)
    parser.add_argument("--progress-every-plies", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    args = parser.parse_args(argv)

    _validate_args(parser, args)

    started_at = perf_counter()
    config = CheckpointSelfPlayConfig(
        checkpoint=args.checkpoint,
        checkpoint_id=args.checkpoint_id,
        games=args.games,
        self_play_worker_processes=args.self_play_worker_processes,
        concurrent_games_per_process=args.concurrent_games_per_process,
        max_plies=args.max_plies,
        mcts_simulations=args.mcts_simulations,
        nn_leaf_eval_batch_limit=args.mcts_nn_leaf_eval_batch_limit,
        central_evaluator_batch_size_limit=args.central_evaluator_batch_size_limit,
        central_evaluator_flush_timeout_sec=args.central_evaluator_flush_timeout_sec,
        device=args.device,
        inference_precision=args.inference_precision,
        compile_model=args.compile_model,
        board_backend=args.board_backend,
        move_selection=visit_sampling_move_selection_config(
            seed=args.seed,
            temperature=args.move_selection_temperature,
            temperature_plies=args.move_selection_temperature_plies,
        ),
        progress_every_plies=args.progress_every_plies,
    )
    with GeneratedGameArtifacts(Path(args.out)) as artifacts:
        result = run_checkpoint_self_play_generation(
            config,
            checkpoint_evaluator_cls=ShogiMoveChoiceCheckpointEvaluator,
            record_callback=artifacts.write_record,
            progress_callback=artifacts.write_progress,
        )
    summary = summarize_checkpoint_self_play_records(result.records, wall_time_sec=perf_counter() - started_at)
    summary["central_evaluator_performance"] = result.central_evaluator_performance
    print(json.dumps(summary, indent=2))


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.games <= 0:
        parser.error("--games must be positive")
    if args.self_play_worker_processes <= 0:
        parser.error("--self-play-worker-processes must be positive")
    if args.concurrent_games_per_process <= 0:
        parser.error("--concurrent-games-per-process must be positive")
    if args.mcts_simulations <= 0:
        parser.error("--mcts-simulations must be positive")
    if args.mcts_nn_leaf_eval_batch_limit <= 0:
        parser.error("--mcts-nn-leaf-eval-batch-limit must be positive")
    if args.central_evaluator_batch_size_limit is not None and args.central_evaluator_batch_size_limit <= 0:
        parser.error("--central-evaluator-batch-size-limit must be positive")
    if args.central_evaluator_flush_timeout_sec < 0.0:
        parser.error("--central-evaluator-flush-timeout-sec must be non-negative")
    if args.progress_every_plies < 0:
        parser.error("--progress-every-plies must be non-negative")
    if args.max_plies <= 0:
        parser.error("--max-plies must be positive")
    if args.move_selection_temperature <= 0.0:
        parser.error("--move-selection-temperature must be positive")
    if args.move_selection_temperature_plies < 0:
        parser.error("--move-selection-temperature-plies must be non-negative")
    if args.max_plies < STANDARD_MAX_PLIES:
        print(
            f"warning: --max-plies {args.max_plies} is below the computer-shogi standard cap "
            f"of {STANDARD_MAX_PLIES}; this can create artificial max_plies draws.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
