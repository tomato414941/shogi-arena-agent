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


def build_static_player(args: argparse.Namespace, prefix: str, *, name: str) -> BuiltPlayer | None:
    kind = _arg(args, prefix, "kind")
    if kind == "checkpoint":
        checkpoint = _arg(args, prefix, "checkpoint")
        checkpoint_id = _arg(args, prefix, "checkpoint_id")
        profile = _arg(args, prefix, "move_selection_profile")
        move_selector = _arg(args, prefix, "move_selector")
        simulations = _arg(args, prefix, "mcts_simulations")
        evaluation_batch_size = _arg(args, prefix, "mcts_evaluation_batch_size")
        move_time_limit_sec = _arg(args, prefix, "mcts_move_time_limit_sec")
        root_reuse = _arg(args, prefix, "mcts_root_reuse")
        device = _arg(args, prefix, "device")
        board_backend = _arg(args, prefix, "board_backend")
        mcts_config = MctsConfig(
            simulation_count=simulations,
            evaluation_batch_size=evaluation_batch_size,
            move_time_limit_sec=move_time_limit_sec,
            board_backend=board_backend,
            root_reuse=root_reuse,
        )
        policy = _load_move_selector(
            checkpoint,
            move_selector=move_selector,
            config=mcts_config,
            profile=profile,
            device=device,
            board_backend=board_backend,
        )
        return BuiltPlayer(
            player=UsiEngine(name=name, policy=policy),
            actor=ShogiActorSpec(
                kind="checkpoint",
                name=name,
                settings={
                    "checkpoint": checkpoint,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_path": checkpoint,
                    "move_selection_profile": profile,
                    "move_selector": move_selector,
                    "mcts_simulations_per_move": mcts_config.simulation_count if move_selector == "mcts" else None,
                    "nn_leaf_eval_batch_limit": mcts_config.evaluation_batch_size if move_selector == "mcts" else None,
                    "simulations": mcts_config.simulation_count if move_selector == "mcts" else None,
                    "evaluation_batch_size": mcts_config.evaluation_batch_size if move_selector == "mcts" else None,
                    "move_time_limit_sec": mcts_config.move_time_limit_sec if move_selector == "mcts" else None,
                    "root_reuse": mcts_config.root_reuse if move_selector == "mcts" else None,
                    "device": device,
                    "board_backend": board_backend,
                },
            ),
        )
    if kind == "deterministic_legal":
        return BuiltPlayer(
            player=UsiEngine(name=name, policy=DeterministicLegalMovePolicy()),
            actor=ShogiActorSpec(kind="deterministic_legal", name=name, settings={}),
        )
    return None


@contextmanager
def player_context(args: argparse.Namespace, prefix: str, *, name: str) -> Iterator[BuiltPlayer]:
    static_player = build_static_player(args, prefix, name=name)
    if static_player is not None:
        yield static_player
        return

    command = _arg(args, prefix, "usi_command")
    options = _parse_usi_options(_arg(args, prefix, "usi_option"))
    go_command = _arg(args, prefix, "usi_go_command")
    read_timeout_seconds = _arg(args, prefix, "usi_read_timeout_seconds")
    multipv = _arg(args, prefix, "usi_policy_target_multipv")
    temperature_cp = _arg(args, prefix, "usi_policy_target_temperature_cp")
    actor = ShogiActorSpec(
        kind="usi_engine",
        name=name,
        settings={
            "command": command,
            "usi_options_json": json.dumps(options, sort_keys=True),
            "go_command": go_command,
            "read_timeout_seconds": read_timeout_seconds,
            "policy_target_multipv": multipv,
            "policy_target_temperature_cp": temperature_cp,
        },
    )
    with UsiProcess(
        command=[command],
        options=options,
        go_command=go_command,
        read_timeout_seconds=read_timeout_seconds,
    ) as engine:
        player: ShogiPlayer = engine
        if multipv is not None:
            config = MultiPVPolicyTargetConfig(multipv=multipv, temperature_cp=temperature_cp)
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
) -> Any:
    if move_selector == "direct":
        return ShogiMoveChoiceCheckpointPolicy.from_checkpoint(checkpoint, device=device, board_backend=board_backend)
    evaluator = ShogiMoveChoiceCheckpointEvaluator.from_checkpoint(checkpoint, device=device)
    return MctsMoveSelector(evaluator=evaluator, config=config, move_selection=_move_selection_config(profile))


def _move_selection_config(profile: str):
    if profile == "self-play":
        return self_play_move_selection_config()
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
