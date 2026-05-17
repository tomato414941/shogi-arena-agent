from __future__ import annotations

from dataclasses import dataclass

from shogi_arena_agent.board_backend import validate_board_backend


@dataclass(frozen=True)
class MctsConfig:
    simulation_count: int = 32
    c_puct: float = 1.5
    evaluation_batch_size: int = 1
    move_time_limit_sec: float | None = None
    board_backend: str = "python-shogi"
    root_reuse: bool = False

    def __post_init__(self) -> None:
        if self.simulation_count <= 0:
            raise ValueError("simulation_count must be positive")
        if self.c_puct <= 0.0:
            raise ValueError("c_puct must be positive")
        if self.evaluation_batch_size <= 0:
            raise ValueError("evaluation_batch_size must be positive")
        if self.move_time_limit_sec is not None and self.move_time_limit_sec < 0.0:
            raise ValueError("move_time_limit_sec must be non-negative")
        validate_board_backend(self.board_backend)


@dataclass(frozen=True)
class MoveSelectionConfig:
    mode: str = "visit_sample"
    temperature: float = 1.0
    temperature_plies: int = 40
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.mode != "visit_sample":
            raise ValueError("mode must be visit_sample")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.temperature_plies < 0:
            raise ValueError("temperature_plies must be non-negative")


def visit_sampling_move_selection_config(
    *,
    seed: int | None = None,
    temperature: float = 1.0,
    temperature_plies: int = 40,
) -> MoveSelectionConfig:
    return MoveSelectionConfig(
        mode="visit_sample",
        temperature=temperature,
        temperature_plies=temperature_plies,
        seed=seed,
    )
