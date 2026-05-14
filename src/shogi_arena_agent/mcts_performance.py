from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MctsMovePerformance:
    request_wall_time_sec: float
    model_call_count: int
    model_wall_time_sec: float
    non_model_wall_time_sec: float
    output_count: int
    output_per_sec: float
    actual_nn_leaf_eval_batch_size_avg: float = 0.0
    actual_nn_leaf_eval_batch_size_max: int = 0
    actual_nn_leaf_eval_batch_count: int = 0
    actual_nn_leaf_eval_batch_size_histogram: dict[int, int] = field(default_factory=dict)
    actual_nn_leaf_eval_batch_size_fill_ratio_avg: float = 0.0
    phase_wall_time_sec: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MctsBatchPerformance:
    request_wall_time_sec: float
    position_count: int
    completed_simulations: int
    model_call_count: int
    model_wall_time_sec: float
    non_model_wall_time_sec: float
    output_per_sec: float
    actual_nn_leaf_eval_batch_size_avg: float = 0.0
    actual_nn_leaf_eval_batch_size_max: int = 0
    actual_nn_leaf_eval_batch_count: int = 0
    actual_nn_leaf_eval_batch_size_histogram: dict[int, int] = field(default_factory=dict)
    actual_nn_leaf_eval_batch_size_fill_ratio_avg: float = 0.0
    phase_wall_time_sec: dict[str, float] = field(default_factory=dict)


def leaf_eval_batch_metrics(batch_sizes: Sequence[int], *, batch_size_limit: int) -> dict[str, object]:
    if not batch_sizes:
        return {
            "actual_nn_leaf_eval_batch_size_avg": 0.0,
            "actual_nn_leaf_eval_batch_size_max": 0,
            "actual_nn_leaf_eval_batch_count": 0,
            "actual_nn_leaf_eval_batch_size_histogram": {},
            "actual_nn_leaf_eval_batch_size_fill_ratio_avg": 0.0,
        }
    histogram: dict[int, int] = {}
    for size in batch_sizes:
        histogram[size] = histogram.get(size, 0) + 1
    size_avg = sum(batch_sizes) / len(batch_sizes)
    return {
        "actual_nn_leaf_eval_batch_size_avg": size_avg,
        "actual_nn_leaf_eval_batch_size_max": max(batch_sizes),
        "actual_nn_leaf_eval_batch_count": len(batch_sizes),
        "actual_nn_leaf_eval_batch_size_histogram": dict(sorted(histogram.items())),
        "actual_nn_leaf_eval_batch_size_fill_ratio_avg": size_avg / batch_size_limit if batch_size_limit > 0 else 0.0,
    }
