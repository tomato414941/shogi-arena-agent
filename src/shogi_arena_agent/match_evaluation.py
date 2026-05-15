from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from shogi_arena_agent.shogi_game import ShogiGameRecord


@dataclass(frozen=True)
class MatchEvaluation:
    game_count: int
    player_a_black_game_count: int
    player_a_white_game_count: int
    end_reasons: dict[str, int]
    player_a_wins: int
    player_a_losses: int
    player_b_wins: int
    player_b_losses: int
    draws: int
    player_a_black_wins: int
    player_a_black_losses: int
    player_a_white_wins: int
    player_a_white_losses: int
    average_plies: float
    illegal_move_count: int
    results: tuple[ShogiGameRecord, ...]


def summarize_match_results(results: list[ShogiGameRecord], player_a_sides: list[str]) -> MatchEvaluation:
    game_count = len(results)
    end_reasons = Counter(result.end_reason for result in results)
    player_a_wins = sum(1 for result, side in zip(results, player_a_sides) if result.winner == side)
    player_a_losses = sum(
        1 for result, side in zip(results, player_a_sides) if result.winner is not None and result.winner != side
    )
    draws = sum(1 for result in results if result.winner is None)
    player_a_black_wins = sum(
        1 for result, side in zip(results, player_a_sides) if side == "black" and result.winner == "black"
    )
    player_a_black_losses = sum(
        1 for result, side in zip(results, player_a_sides) if side == "black" and result.winner == "white"
    )
    player_a_white_wins = sum(
        1 for result, side in zip(results, player_a_sides) if side == "white" and result.winner == "white"
    )
    player_a_white_losses = sum(
        1 for result, side in zip(results, player_a_sides) if side == "white" and result.winner == "black"
    )
    return MatchEvaluation(
        game_count=game_count,
        player_a_black_game_count=(game_count + 1) // 2,
        player_a_white_game_count=game_count // 2,
        end_reasons=dict(end_reasons),
        player_a_wins=player_a_wins,
        player_a_losses=player_a_losses,
        player_b_wins=player_a_losses,
        player_b_losses=player_a_wins,
        draws=draws,
        player_a_black_wins=player_a_black_wins,
        player_a_black_losses=player_a_black_losses,
        player_a_white_wins=player_a_white_wins,
        player_a_white_losses=player_a_white_losses,
        average_plies=sum(len(result.transitions) for result in results) / game_count,
        illegal_move_count=end_reasons.get("illegal_move", 0),
        results=tuple(results),
    )
