from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from shogi_arena_agent.shogi_game import ShogiGameRecord, ShogiTransitionRecord, load_shogi_game_records_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze MCTS teacher quality evidence in shogi game records.")
    parser.add_argument("--input", action="append", required=True, type=Path, help="Input ShogiGameRecord JSONL path.")
    args = parser.parse_args(argv)

    records = []
    for path in args.input:
        records.extend(load_shogi_game_records_jsonl(path))
    print(json.dumps(analyze_teacher_quality(tuple(records), input_files=tuple(args.input)), indent=2, sort_keys=True))


def analyze_teacher_quality(
    records: tuple[ShogiGameRecord, ...],
    *,
    input_files: tuple[Path, ...] = (),
) -> dict[str, Any]:
    transition_count = 0
    mcts_transition_count = 0
    prior_transition_count = 0
    chosen_visit_top1_matches: list[float] = []
    policy_mcts_top1_disagreements: list[float] = []
    visit_top1_shares: list[float] = []
    visit_entropies: list[float] = []
    visit_normalized_entropies: list[float] = []
    prior_top1_shares: list[float] = []
    root_mean_values: list[float] = []
    end_reasons: dict[str, int] = {}

    for record in records:
        end_reasons[record.end_reason] = end_reasons.get(record.end_reason, 0) + 1
        for transition in record.transitions:
            transition_count += 1
            evidence = _search_evidence(transition)
            if evidence is None:
                continue
            visits = _positive_number_dict(evidence.get("mcts_root_child_visit_counts"))
            if visits:
                mcts_transition_count += 1
                visit_top1 = _top_key(visits)
                chosen_visit_top1_matches.append(1.0 if transition.action_usi == visit_top1 else 0.0)
                visit_top1_shares.append(max(visits.values()) / sum(visits.values()))
                entropy = _entropy(visits.values())
                visit_entropies.append(entropy)
                visit_normalized_entropies.append(_normalized_entropy(entropy, len(visits)))
            priors = _positive_number_dict(evidence.get("mcts_root_child_priors"))
            if priors:
                prior_transition_count += 1
                prior_top1_shares.append(max(priors.values()) / sum(priors.values()))
                if visits:
                    policy_mcts_top1_disagreements.append(1.0 if _top_key(priors) != _top_key(visits) else 0.0)
            root_mean_value = evidence.get("mcts_root_mean_value")
            if _is_number(root_mean_value):
                root_mean_values.append(float(root_mean_value))

    return {
        "schema_version": "shogi_arena_agent.teacher_quality.v1",
        "input_files": [str(path) for path in input_files],
        "game_count": len(records),
        "transition_count": transition_count,
        "mcts_transition_count": mcts_transition_count,
        "prior_transition_count": prior_transition_count,
        "end_reasons": dict(sorted(end_reasons.items())),
        "average_plies": transition_count / len(records) if records else 0.0,
        "chosen_visit_top1_rate": _mean_or_none(chosen_visit_top1_matches),
        "policy_mcts_top1_disagreement_rate": _mean_or_none(policy_mcts_top1_disagreements),
        "visit_top1_share": _summary(visit_top1_shares),
        "visit_entropy": _summary(visit_entropies),
        "visit_normalized_entropy": _summary(visit_normalized_entropies),
        "prior_top1_share": _summary(prior_top1_shares),
        "root_mean_value": _summary(root_mean_values),
    }


def _search_evidence(transition: ShogiTransitionRecord) -> dict[str, object] | None:
    telemetry = transition.decision_telemetry
    if telemetry is None or telemetry.search_evidence is None:
        return None
    return telemetry.search_evidence


def _positive_number_dict(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        if _is_number(item) and float(item) > 0.0:
            result[str(key)] = float(item)
    return result


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))


def _top_key(values: dict[str, float]) -> str:
    return max(values, key=lambda key: (values[key], key))


def _entropy(values: Any) -> float:
    total = sum(values)
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for value in values:
        probability = value / total
        if probability > 0.0:
            entropy -= probability * math.log(probability)
    return entropy


def _normalized_entropy(entropy: float, count: int) -> float:
    if count <= 1:
        return 0.0
    return entropy / math.log(count)


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "min": min(values),
        "max": max(values),
    }


if __name__ == "__main__":
    main()
