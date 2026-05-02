from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from shogi_arena_agent.local_match import LocalPlayer, LocalMatchResult, play_local_match
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
    results: tuple[LocalMatchResult, ...]


def evaluate_player_against_baseline(
    player: LocalPlayer | UsiEngine,
    *,
    game_count: int = 2,
    max_plies: int = 64,
) -> MatchEvaluation:
    if game_count <= 0:
        raise ValueError("game_count must be positive")

    results: list[LocalMatchResult] = []
    player_sides: list[str] = []
    for game_index in range(game_count):
        if game_index % 2 == 0:
            result = play_local_match(black=player, white=UsiEngine(name="baseline-white"), max_plies=max_plies)
            player_sides.append("black")
        else:
            result = play_local_match(black=UsiEngine(name="baseline-black"), white=player, max_plies=max_plies)
            player_sides.append("white")
        results.append(result)

    return _summarize_match_results(results, player_sides)


def evaluate_player_against_usi_engine(
    player: LocalPlayer | UsiEngine,
    engine_command: Sequence[str],
    *,
    game_count: int = 2,
    max_plies: int = 64,
    engine_go_command: str = "go btime 0 wtime 0",
    read_timeout_seconds: float = 5.0,
) -> MatchEvaluation:
    if game_count <= 0:
        raise ValueError("game_count must be positive")

    results: list[LocalMatchResult] = []
    player_sides: list[str] = []
    for game_index in range(game_count):
        with UsiProcess(
            command=engine_command,
            go_command=engine_go_command,
            read_timeout_seconds=read_timeout_seconds,
        ) as external_engine:
            if game_index % 2 == 0:
                result = play_local_match(black=player, white=external_engine, max_plies=max_plies)
                player_sides.append("black")
            else:
                result = play_local_match(black=external_engine, white=player, max_plies=max_plies)
                player_sides.append("white")
            results.append(result)

    return _summarize_match_results(results, player_sides)


def _summarize_match_results(results: list[LocalMatchResult], player_sides: list[str]) -> MatchEvaluation:
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
        average_plies=sum(len(result.moves) for result in results) / game_count,
        illegal_move_count=end_reasons.get("illegal_move", 0),
        results=tuple(results),
    )
