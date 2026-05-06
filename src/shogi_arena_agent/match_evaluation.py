from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from shogi_arena_agent.shogi_game import ShogiPlayer, ShogiGameRecord, ShogiActorSpec, play_shogi_game
from shogi_arena_agent.multipv_policy import (
    MultiPVPolicyTargetConfig,
    MultiPVPolicyTargetPlayer,
    configure_multipv_player_options,
)
from shogi_arena_agent.usi import UsiEngine
from shogi_arena_agent.usi_process import UsiProcess


@dataclass(frozen=True)
class MatchEvaluation:
    game_count: int
    black_game_count: int
    white_game_count: int
    end_reasons: dict[str, int]
    player_wins: int
    player_losses: int
    draws: int
    player_black_wins: int
    player_black_losses: int
    player_white_wins: int
    player_white_losses: int
    average_plies: float
    illegal_move_count: int
    results: tuple[ShogiGameRecord, ...]


def evaluate_player_against_deterministic_legal(
    player: ShogiPlayer | UsiEngine,
    *,
    game_count: int = 2,
    max_plies: int = 64,
    player_actor: ShogiActorSpec | None = None,
) -> MatchEvaluation:
    if game_count <= 0:
        raise ValueError("game_count must be positive")

    results: list[ShogiGameRecord] = []
    player_sides: list[str] = []
    player_actor = player_actor or _shogi_actor_spec(player, name="player")
    for game_index in range(game_count):
        if game_index % 2 == 0:
            result = play_shogi_game(
                black=player,
                white=UsiEngine(name="deterministic-legal-white"),
                black_actor=player_actor,
                white_actor=ShogiActorSpec(kind="deterministic_legal", name="deterministic-legal-white", settings={}),
                max_plies=max_plies,
            )
            player_sides.append("black")
        else:
            result = play_shogi_game(
                black=UsiEngine(name="deterministic-legal-black"),
                white=player,
                black_actor=ShogiActorSpec(kind="deterministic_legal", name="deterministic-legal-black", settings={}),
                white_actor=player_actor,
                max_plies=max_plies,
            )
            player_sides.append("white")
        results.append(result)

    return summarize_match_results(results, player_sides)


def evaluate_player_against_usi_engine(
    player: ShogiPlayer | UsiEngine,
    engine_command: Sequence[str],
    *,
    game_count: int = 2,
    max_plies: int = 64,
    engine_go_command: str = "go btime 0 wtime 0",
    read_timeout_seconds: float = 5.0,
    engine_policy_target_multipv: int | None = None,
    engine_policy_target_temperature_cp: float = 100.0,
    player_actor: ShogiActorSpec | None = None,
    engine_actor: ShogiActorSpec | None = None,
) -> MatchEvaluation:
    if game_count <= 0:
        raise ValueError("game_count must be positive")

    results: list[ShogiGameRecord] = []
    player_sides: list[str] = []
    player_actor = player_actor or _shogi_actor_spec(player, name="player")
    multipv_config = (
        MultiPVPolicyTargetConfig(
            multipv=engine_policy_target_multipv,
            temperature_cp=engine_policy_target_temperature_cp,
        )
        if engine_policy_target_multipv is not None
        else None
    )
    external_actor = engine_actor or ShogiActorSpec(
        kind="usi_process",
        name="external",
        settings={
            "command": " ".join(engine_command),
            "go_command": engine_go_command,
            "read_timeout_seconds": read_timeout_seconds,
            "policy_target_multipv": engine_policy_target_multipv,
            "policy_target_temperature_cp": engine_policy_target_temperature_cp,
        },
    )
    for game_index in range(game_count):
        with UsiProcess(
            command=engine_command,
            go_command=engine_go_command,
            read_timeout_seconds=read_timeout_seconds,
        ) as external_engine:
            external_player: ShogiPlayer = external_engine
            if multipv_config is not None:
                configure_multipv_player_options(external_engine, multipv_config)
                external_player = MultiPVPolicyTargetPlayer(external_engine, multipv_config)
            if game_index % 2 == 0:
                result = play_shogi_game(
                    black=player,
                    white=external_player,
                    black_actor=player_actor,
                    white_actor=external_actor,
                    max_plies=max_plies,
                )
                player_sides.append("black")
            else:
                result = play_shogi_game(
                    black=external_player,
                    white=player,
                    black_actor=external_actor,
                    white_actor=player_actor,
                    max_plies=max_plies,
                )
                player_sides.append("white")
            results.append(result)

    return summarize_match_results(results, player_sides)


def summarize_match_results(results: list[ShogiGameRecord], player_sides: list[str]) -> MatchEvaluation:
    game_count = len(results)
    end_reasons = Counter(result.end_reason for result in results)
    player_wins = sum(1 for result, side in zip(results, player_sides) if result.winner == side)
    player_losses = sum(1 for result, side in zip(results, player_sides) if result.winner is not None and result.winner != side)
    draws = sum(1 for result in results if result.winner is None)
    player_black_wins = sum(
        1 for result, side in zip(results, player_sides) if side == "black" and result.winner == "black"
    )
    player_black_losses = sum(
        1 for result, side in zip(results, player_sides) if side == "black" and result.winner == "white"
    )
    player_white_wins = sum(
        1 for result, side in zip(results, player_sides) if side == "white" and result.winner == "white"
    )
    player_white_losses = sum(
        1 for result, side in zip(results, player_sides) if side == "white" and result.winner == "black"
    )
    return MatchEvaluation(
        game_count=game_count,
        black_game_count=(game_count + 1) // 2,
        white_game_count=game_count // 2,
        end_reasons=dict(end_reasons),
        player_wins=player_wins,
        player_losses=player_losses,
        draws=draws,
        player_black_wins=player_black_wins,
        player_black_losses=player_black_losses,
        player_white_wins=player_white_wins,
        player_white_losses=player_white_losses,
        average_plies=sum(len(result.transitions) for result in results) / game_count,
        illegal_move_count=end_reasons.get("illegal_move", 0),
        results=tuple(results),
    )


def _shogi_actor_spec(player: ShogiPlayer | UsiEngine, *, name: str) -> ShogiActorSpec:
    if isinstance(player, UsiEngine):
        return ShogiActorSpec(kind="usi_engine", name=player.name, settings={})
    return ShogiActorSpec(kind="shogi_player", name=name, settings={})
