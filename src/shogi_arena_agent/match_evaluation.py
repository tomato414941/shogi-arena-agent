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
    for game_index in range(game_count):
        if game_index % 2 == 0:
            result = play_local_match(black=player, white=UsiEngine(name="baseline-white"), max_plies=max_plies)
        else:
            result = play_local_match(black=UsiEngine(name="baseline-black"), white=player, max_plies=max_plies)
        results.append(result)

    return _summarize_match_results(results, game_count)


def evaluate_player_against_usi_engine(
    player: LocalPlayer | UsiEngine,
    engine_command: Sequence[str],
    *,
    game_count: int = 2,
    max_plies: int = 64,
    read_timeout_seconds: float = 5.0,
) -> MatchEvaluation:
    if game_count <= 0:
        raise ValueError("game_count must be positive")

    results: list[LocalMatchResult] = []
    for game_index in range(game_count):
        with UsiProcess(command=engine_command, read_timeout_seconds=read_timeout_seconds) as external_engine:
            if game_index % 2 == 0:
                result = play_local_match(black=player, white=external_engine, max_plies=max_plies)
            else:
                result = play_local_match(black=external_engine, white=player, max_plies=max_plies)
            results.append(result)

    return _summarize_match_results(results, game_count)


def _summarize_match_results(results: list[LocalMatchResult], game_count: int) -> MatchEvaluation:
    end_reasons = Counter(result.end_reason for result in results)
    return MatchEvaluation(
        game_count=game_count,
        black_game_count=(game_count + 1) // 2,
        white_game_count=game_count // 2,
        end_reasons=dict(end_reasons),
        average_plies=sum(len(result.moves) for result in results) / game_count,
        illegal_move_count=end_reasons.get("illegal_move", 0),
        results=tuple(results),
    )
