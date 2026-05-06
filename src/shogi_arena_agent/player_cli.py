from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from shogi_arena_agent.deterministic_legal_policy import DeterministicLegalMovePolicy
from shogi_arena_agent.mcts_policy import MctsConfig, MctsPolicy
from shogi_arena_agent.model_policy import ShogiMoveChoiceCheckpointEvaluator, ShogiMoveChoiceCheckpointPolicy
from shogi_arena_agent.multipv_policy import (
    MultiPVPolicyTargetConfig,
    MultiPVPolicyTargetPlayer,
    configure_multipv_player_options,
)
from shogi_arena_agent.shogi_game import ShogiActorSpec, ShogiPlayer
from shogi_arena_agent.usi import UsiEngine
from shogi_arena_agent.usi_process import UsiProcess

PLAYER_KINDS = ("checkpoint", "yaneuraou", "deterministic_legal")


@dataclass(frozen=True)
class BuiltPlayer:
    player: ShogiPlayer | UsiEngine
    actor: ShogiActorSpec


def add_player_arguments(parser: argparse.ArgumentParser, prefix: str) -> None:
    parser.add_argument(f"--{prefix}-kind", choices=PLAYER_KINDS, required=True)
    parser.add_argument(f"--{prefix}-checkpoint")
    parser.add_argument(f"--{prefix}-checkpoint-policy", choices=("direct", "mcts"), default="mcts")
    parser.add_argument(f"--{prefix}-checkpoint-simulations", type=int, default=16)
    parser.add_argument(f"--{prefix}-yaneuraou-command")
    parser.add_argument(f"--{prefix}-yaneuraou-go-command", default="go nodes 1")
    parser.add_argument(f"--{prefix}-yaneuraou-read-timeout-seconds", type=float, default=10.0)
    parser.add_argument(f"--{prefix}-yaneuraou-policy-target-multipv", type=int)
    parser.add_argument(f"--{prefix}-yaneuraou-policy-target-temperature-cp", type=float, default=100.0)


def validate_player_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace, prefix: str) -> None:
    kind = _arg(args, prefix, "kind")
    if kind == "checkpoint" and not _arg(args, prefix, "checkpoint"):
        parser.error(f"--{prefix}-checkpoint is required when --{prefix}-kind checkpoint")
    if kind == "yaneuraou" and not _arg(args, prefix, "yaneuraou_command"):
        parser.error(f"--{prefix}-yaneuraou-command is required when --{prefix}-kind yaneuraou")


def build_static_player(args: argparse.Namespace, prefix: str, *, name: str) -> BuiltPlayer | None:
    kind = _arg(args, prefix, "kind")
    if kind == "checkpoint":
        checkpoint = _arg(args, prefix, "checkpoint")
        policy_kind = _arg(args, prefix, "checkpoint_policy")
        simulations = _arg(args, prefix, "checkpoint_simulations")
        policy = _load_checkpoint_policy(checkpoint, policy_kind=policy_kind, simulations=simulations)
        return BuiltPlayer(
            player=UsiEngine(name=name, policy=policy),
            actor=ShogiActorSpec(
                kind="checkpoint",
                name=name,
                settings={
                    "checkpoint": checkpoint,
                    "policy": policy_kind,
                    "simulations": simulations if policy_kind == "mcts" else None,
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

    command = _arg(args, prefix, "yaneuraou_command")
    go_command = _arg(args, prefix, "yaneuraou_go_command")
    read_timeout_seconds = _arg(args, prefix, "yaneuraou_read_timeout_seconds")
    multipv = _arg(args, prefix, "yaneuraou_policy_target_multipv")
    temperature_cp = _arg(args, prefix, "yaneuraou_policy_target_temperature_cp")
    actor = ShogiActorSpec(
        kind="yaneuraou",
        name=name,
        settings={
            "command": command,
            "go_command": go_command,
            "read_timeout_seconds": read_timeout_seconds,
            "policy_target_multipv": multipv,
            "policy_target_temperature_cp": temperature_cp,
        },
    )
    with UsiProcess(
        command=[command],
        go_command=go_command,
        read_timeout_seconds=read_timeout_seconds,
    ) as engine:
        player: ShogiPlayer = engine
        if multipv is not None:
            config = MultiPVPolicyTargetConfig(multipv=multipv, temperature_cp=temperature_cp)
            configure_multipv_player_options(engine, config)
            player = MultiPVPolicyTargetPlayer(engine, config)
        yield BuiltPlayer(player=player, actor=actor)


def _load_checkpoint_policy(checkpoint: str, *, policy_kind: str, simulations: int) -> Any:
    if policy_kind == "direct":
        return ShogiMoveChoiceCheckpointPolicy.from_checkpoint(checkpoint)
    evaluator = ShogiMoveChoiceCheckpointEvaluator.from_checkpoint(checkpoint)
    return MctsPolicy(evaluator=evaluator, config=MctsConfig(simulation_count=simulations))


def _arg(args: argparse.Namespace, prefix: str, name: str) -> Any:
    return getattr(args, f"{prefix}_{name}")
