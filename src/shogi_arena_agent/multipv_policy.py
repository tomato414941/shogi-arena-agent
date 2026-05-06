from __future__ import annotations

from dataclasses import dataclass
from math import exp
from collections.abc import Sequence

from shogi_arena_agent.usi_process import UsiGoResult


@dataclass(frozen=True)
class MultiPVPolicyTargetConfig:
    multipv: int
    temperature_cp: float = 100.0
    mate_cp: float = 100000.0

    def __post_init__(self) -> None:
        if self.multipv <= 0:
            raise ValueError("multipv must be positive")
        if self.temperature_cp <= 0.0:
            raise ValueError("temperature_cp must be positive")


@dataclass(frozen=True)
class MultiPVInfo:
    multipv: int
    score_cp: float
    move: str


class MultiPVPolicyTargetPlayer:
    def __init__(self, player, config: MultiPVPolicyTargetConfig) -> None:
        self.player = player
        self.config = config

    def position(self, command: str) -> None:
        self.player.position(command)

    def go(self) -> UsiGoResult:
        result = self.player.go()
        if isinstance(result, str):
            return UsiGoResult(bestmove=result)
        return UsiGoResult(
            bestmove=result.bestmove,
            ponder=result.ponder,
            info_lines=result.info_lines,
            policy_targets=policy_targets_from_multipv_info(result.info_lines, self.config),
        )


def configure_multipv_player_options(player, config: MultiPVPolicyTargetConfig) -> None:
    if hasattr(player, "setoption"):
        player.setoption(name="MultiPV", value=config.multipv)


def policy_targets_from_multipv_info(
    info_lines: Sequence[str],
    config: MultiPVPolicyTargetConfig,
) -> dict[str, float] | None:
    scored_moves: dict[str, float] = {}
    for line in info_lines:
        parsed = parse_multipv_info_line(line, mate_cp=config.mate_cp)
        if parsed is None:
            continue
        if parsed.multipv <= config.multipv:
            scored_moves[parsed.move] = parsed.score_cp
    if not scored_moves:
        return None
    max_score = max(scored_moves.values())
    weights = {
        move: exp((score_cp - max_score) / config.temperature_cp)
        for move, score_cp in scored_moves.items()
    }
    total = sum(weights.values())
    return {move: weight / total for move, weight in weights.items()}


def parse_multipv_info_line(line: str, *, mate_cp: float = 100000.0) -> MultiPVInfo | None:
    words = line.split()
    if not words or words[0] != "info":
        return None
    try:
        score_index = words.index("score")
        pv_index = words.index("pv")
    except ValueError:
        return None
    if pv_index + 1 >= len(words) or score_index + 2 >= len(words):
        return None
    multipv = 1
    if "multipv" in words:
        multipv_index = words.index("multipv")
        if multipv_index + 1 >= len(words):
            return None
        multipv = int(words[multipv_index + 1])
    score_kind = words[score_index + 1]
    score_value = float(words[score_index + 2])
    if score_kind == "cp":
        score_cp = score_value
    elif score_kind == "mate":
        score_cp = mate_cp if score_value > 0 else -mate_cp
    else:
        return None
    return MultiPVInfo(multipv=multipv, score_cp=score_cp, move=words[pv_index + 1])
