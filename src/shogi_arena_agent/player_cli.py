from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from shogi_arena_agent.deterministic_legal_policy import DeterministicLegalMovePolicy
from shogi_arena_agent.mcts_config import (
    MctsConfig,
    evaluation_move_selection_config,
    self_play_move_selection_config,
)
from shogi_arena_agent.mcts_move_selector import MctsMoveSelector
from shogi_arena_agent.model_policy import ShogiMoveChoiceCheckpointEvaluator, ShogiMoveChoiceCheckpointPolicy
from shogi_arena_agent.usi import BOARD_BACKENDS
from shogi_arena_agent.multipv_policy import (
    MultiPVPolicyTargetConfig,
    MultiPVPolicyTargetPlayer,
    configure_multipv_player_options,
)
from shogi_arena_agent.shogi_game import ShogiActorSpec, ShogiPlayer
from shogi_arena_agent.usi import UsiEngine
from shogi_arena_agent.usi_process import UsiProcess

PLAYER_KINDS = ("checkpoint", "usi_engine", "deterministic_legal")
MOVE_SELECTION_PROFILES = ("evaluation", "self-play")


@dataclass(frozen=True)
class BuiltPlayer:
    player: ShogiPlayer | UsiEngine
    actor: ShogiActorSpec


@dataclass(frozen=True)
class CheckpointPolicyPlayerSpec:
    checkpoint: str
    checkpoint_id: str | None = None
    move_selection_profile: str = "evaluation"
    move_selector: str = "mcts"
    mcts_simulations: int = 16
    mcts_evaluation_batch_size: int = 1
    mcts_move_time_limit_sec: float | None = None
    mcts_root_reuse: bool = False
    device: str = "cpu"
    board_backend: str = "python-shogi"
    seed: int | None = None


@dataclass(frozen=True)
class ExternalEnginePlayerSpec:
    command: str
    usi_option: tuple[str, ...] = ()
    usi_go_command: str = "go nodes 1"
    usi_read_timeout_seconds: float = 10.0
    usi_policy_target_multipv: int | None = None
    usi_policy_target_temperature_cp: float = 100.0


@dataclass(frozen=True)
class DeterministicLegalPlayerSpec:
    pass


PlayerSpec = CheckpointPolicyPlayerSpec | ExternalEnginePlayerSpec | DeterministicLegalPlayerSpec


def add_player_arguments(parser: argparse.ArgumentParser, prefix: str) -> None:
    parser.add_argument(f"--{prefix}-kind", choices=PLAYER_KINDS, required=True)
    parser.add_argument(f"--{prefix}-checkpoint")
    parser.add_argument(f"--{prefix}-checkpoint-id")
    parser.add_argument(f"--{prefix}-move-selection-profile", choices=MOVE_SELECTION_PROFILES, default="evaluation")
    parser.add_argument(f"--{prefix}-move-selector", choices=("direct", "mcts"), default="mcts")
    parser.add_argument(f"--{prefix}-mcts-simulations", type=int, default=16)
    parser.add_argument(f"--{prefix}-mcts-evaluation-batch-size", type=int, default=1)
    parser.add_argument(f"--{prefix}-mcts-move-time-limit-sec", type=float)
    parser.add_argument(f"--{prefix}-mcts-root-reuse", action="store_true")
    parser.add_argument(f"--{prefix}-device", default="cpu")
    parser.add_argument(f"--{prefix}-board-backend", choices=BOARD_BACKENDS, default="python-shogi")
    parser.add_argument(f"--{prefix}-usi-command")
    parser.add_argument(f"--{prefix}-usi-option", action="append", default=[])
    parser.add_argument(f"--{prefix}-usi-go-command", default="go nodes 1")
    parser.add_argument(f"--{prefix}-usi-read-timeout-seconds", type=float, default=10.0)
    parser.add_argument(f"--{prefix}-usi-policy-target-multipv", type=int)
    parser.add_argument(f"--{prefix}-usi-policy-target-temperature-cp", type=float, default=100.0)


def validate_player_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace, prefix: str) -> None:
    kind = _arg(args, prefix, "kind")
    if kind == "checkpoint" and not _arg(args, prefix, "checkpoint"):
        parser.error(f"--{prefix}-checkpoint is required when --{prefix}-kind checkpoint")
    if kind == "usi_engine" and not _arg(args, prefix, "usi_command"):
        parser.error(f"--{prefix}-usi-command is required when --{prefix}-kind usi_engine")


def player_spec_from_args(args: argparse.Namespace, prefix: str, *, seed: int | None = None) -> PlayerSpec:
    kind = _arg(args, prefix, "kind")
    if kind == "checkpoint":
        checkpoint = _arg(args, prefix, "checkpoint")
        if checkpoint is None:
            raise ValueError(f"{prefix} checkpoint player requires a checkpoint")
        return CheckpointPolicyPlayerSpec(
            checkpoint=checkpoint,
            checkpoint_id=_arg(args, prefix, "checkpoint_id"),
            move_selection_profile=_arg(args, prefix, "move_selection_profile"),
            move_selector=_arg(args, prefix, "move_selector"),
            mcts_simulations=_arg(args, prefix, "mcts_simulations"),
            mcts_evaluation_batch_size=_arg(args, prefix, "mcts_evaluation_batch_size"),
            mcts_move_time_limit_sec=_arg(args, prefix, "mcts_move_time_limit_sec"),
            mcts_root_reuse=_arg(args, prefix, "mcts_root_reuse"),
            device=_arg(args, prefix, "device"),
            board_backend=_arg(args, prefix, "board_backend"),
            seed=seed,
        )
    if kind == "deterministic_legal":
        return DeterministicLegalPlayerSpec()
    command = _arg(args, prefix, "usi_command")
    if command is None:
        raise ValueError(f"{prefix} USI engine player requires a command")
    return ExternalEnginePlayerSpec(
        command=command,
        usi_option=tuple(_arg(args, prefix, "usi_option")),
        usi_go_command=_arg(args, prefix, "usi_go_command"),
        usi_read_timeout_seconds=_arg(args, prefix, "usi_read_timeout_seconds"),
        usi_policy_target_multipv=_arg(args, prefix, "usi_policy_target_multipv"),
        usi_policy_target_temperature_cp=_arg(args, prefix, "usi_policy_target_temperature_cp"),
    )


def build_static_player(spec: PlayerSpec, *, name: str) -> BuiltPlayer | None:
    if isinstance(spec, CheckpointPolicyPlayerSpec):
        mcts_config = MctsConfig(
            simulation_count=spec.mcts_simulations,
            evaluation_batch_size=spec.mcts_evaluation_batch_size,
            move_time_limit_sec=spec.mcts_move_time_limit_sec,
            board_backend=spec.board_backend,
            root_reuse=spec.mcts_root_reuse,
        )
        policy = _load_move_selector(
            spec.checkpoint,
            move_selector=spec.move_selector,
            config=mcts_config,
            profile=spec.move_selection_profile,
            device=spec.device,
            board_backend=spec.board_backend,
            seed=spec.seed,
        )
        return BuiltPlayer(
            player=UsiEngine(name=name, policy=policy),
            actor=ShogiActorSpec(
                kind="checkpoint",
                name=name,
                settings={
                    "checkpoint": spec.checkpoint,
                    "checkpoint_id": spec.checkpoint_id,
                    "checkpoint_path": spec.checkpoint,
                    "move_selection_profile": spec.move_selection_profile,
                    "move_selector": spec.move_selector,
                    "mcts_simulations_per_move": mcts_config.simulation_count if spec.move_selector == "mcts" else None,
                    "nn_leaf_eval_batch_limit": mcts_config.evaluation_batch_size if spec.move_selector == "mcts" else None,
                    "simulations": mcts_config.simulation_count if spec.move_selector == "mcts" else None,
                    "evaluation_batch_size": mcts_config.evaluation_batch_size if spec.move_selector == "mcts" else None,
                    "move_time_limit_sec": mcts_config.move_time_limit_sec if spec.move_selector == "mcts" else None,
                    "root_reuse": mcts_config.root_reuse if spec.move_selector == "mcts" else None,
                    "device": spec.device,
                    "board_backend": spec.board_backend,
                    "seed": spec.seed,
                },
            ),
        )
    if isinstance(spec, DeterministicLegalPlayerSpec):
        return BuiltPlayer(
            player=UsiEngine(name=name, policy=DeterministicLegalMovePolicy()),
            actor=ShogiActorSpec(kind="deterministic_legal", name=name, settings={}),
        )
    return None


@contextmanager
def player_context(spec: PlayerSpec, *, name: str) -> Iterator[BuiltPlayer]:
    static_player = build_static_player(spec, name=name)
    if static_player is not None:
        yield static_player
        return

    if not isinstance(spec, ExternalEnginePlayerSpec):
        raise TypeError(f"unsupported player spec: {type(spec).__name__}")
    options = _parse_usi_options(list(spec.usi_option))
    actor = ShogiActorSpec(
        kind="usi_engine",
        name=name,
        settings={
            "command": spec.command,
            "usi_options_json": json.dumps(options, sort_keys=True),
            "go_command": spec.usi_go_command,
            "read_timeout_seconds": spec.usi_read_timeout_seconds,
            "policy_target_multipv": spec.usi_policy_target_multipv,
            "policy_target_temperature_cp": spec.usi_policy_target_temperature_cp,
        },
    )
    with UsiProcess(
        command=[spec.command],
        options=options,
        go_command=spec.usi_go_command,
        read_timeout_seconds=spec.usi_read_timeout_seconds,
    ) as engine:
        player: ShogiPlayer = engine
        if spec.usi_policy_target_multipv is not None:
            config = MultiPVPolicyTargetConfig(
                multipv=spec.usi_policy_target_multipv,
                temperature_cp=spec.usi_policy_target_temperature_cp,
            )
            configure_multipv_player_options(engine, config)
            player = MultiPVPolicyTargetPlayer(engine, config)
        yield BuiltPlayer(player=player, actor=actor)


def _load_move_selector(
    checkpoint: str,
    *,
    move_selector: str,
    config: MctsConfig,
    profile: str,
    device: str,
    board_backend: str,
    seed: int | None = None,
) -> Any:
    if move_selector == "direct":
        return ShogiMoveChoiceCheckpointPolicy.from_checkpoint(checkpoint, device=device, board_backend=board_backend)
    evaluator = ShogiMoveChoiceCheckpointEvaluator.from_checkpoint(checkpoint, device=device)
    return MctsMoveSelector(evaluator=evaluator, config=config, move_selection=_move_selection_config(profile, seed=seed))


def _move_selection_config(profile: str, *, seed: int | None = None):
    if profile == "self-play":
        return self_play_move_selection_config(seed=seed)
    return evaluation_move_selection_config()


def _arg(args: argparse.Namespace, prefix: str, name: str) -> Any:
    return getattr(args, f"{prefix.replace('-', '_')}_{name}")


def _parse_usi_options(values: list[str] | None) -> dict[str, str]:
    options: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("USI option must be NAME=VALUE")
        name, option_value = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("USI option name must not be empty")
        options[name] = option_value.strip()
    return options
