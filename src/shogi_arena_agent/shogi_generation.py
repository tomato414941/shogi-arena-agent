from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from contextlib import ExitStack, nullcontext
from dataclasses import asdict, dataclass
from statistics import mean
from time import perf_counter
from typing import Any

from shogi_arena_agent.board_backend import board_is_black_turn, board_turn_name, legal_move_usis
from shogi_arena_agent.multi_position_mcts_search_executor import MultiPositionMctsSearchExecutor
from shogi_arena_agent.mcts_config import (
    MctsConfig,
    visit_sampling_move_selection_config,
)
from shogi_arena_agent.model_policy import ShogiMoveChoiceCheckpointEvaluator
from shogi_arena_agent.player_cli import (
    BuiltPlayer,
    CheckpointPolicyPlayerSpec,
    PlayerSpec,
    build_static_player,
    player_context,
)
from shogi_arena_agent.shogi_game import (
    ShogiActorSpec,
    ShogiDecisionTelemetry,
    ShogiGameRecord,
    ShogiTransitionRecord,
    play_shogi_game,
    position_command,
)
from shogi_arena_agent.start_positions import StartPosition, startpos
from shogi_arena_agent.usi import UsiPosition, board_from_position

GenerationProgressCallback = Callable[[dict[str, Any]], None]
ShogiGameRecordCallback = Callable[[ShogiGameRecord], None]


@dataclass(frozen=True)
class ShogiGenerationConfig:
    black: PlayerSpec
    white: PlayerSpec
    games: int
    concurrent_games_per_process: int
    max_plies: int
    board_backend: str
    start_positions: tuple[StartPosition, ...] = ()
    progress_every_plies: int = 0


def generate_shogi_games(
    config: ShogiGenerationConfig,
    *,
    checkpoint_evaluator_cls: type[ShogiMoveChoiceCheckpointEvaluator] = ShogiMoveChoiceCheckpointEvaluator,
    record_callback: ShogiGameRecordCallback | None = None,
    progress_callback: GenerationProgressCallback | None = None,
) -> tuple[ShogiGameRecord, ...]:
    _validate_start_position_count(config)
    if config.concurrent_games_per_process > 1:
        return _play_multi_position_checkpoint_mcts_games(
            config,
            checkpoint_evaluator_cls=checkpoint_evaluator_cls,
            record_callback=record_callback,
            progress_callback=progress_callback,
        )
    records: list[ShogiGameRecord] = []
    black_static = build_static_player(config.black, name="black")
    white_static = build_static_player(config.white, name="white")
    with ExitStack() as stack:
        black = stack.enter_context(_player_context(config.black, name="black", static_player=black_static))
        white = stack.enter_context(_player_context(config.white, name="white", static_player=white_static))
        for _game_index in range(config.games):
            record = play_shogi_game(
                black=black.player,
                white=white.player,
                black_actor=black.actor,
                white_actor=white.actor,
                max_plies=config.max_plies,
                board_backend=config.board_backend,
                start_position=_start_position_for_game(config, _game_index),
            )
            records.append(record)
            if record_callback is not None:
                record_callback(record)
    return tuple(records)


def records_summary(records: tuple[ShogiGameRecord, ...], *, wall_time_sec: float | None = None) -> dict[str, Any]:
    end_reasons = Counter(record.end_reason for record in records)
    game_count = len(records)
    max_plies_draw_count = end_reasons.get("max_plies", 0)
    game_over_count = end_reasons.get("game_over", 0)
    summary: dict[str, Any] = {
        "game_count": game_count,
        "end_reasons": dict(end_reasons),
        "average_plies": sum(len(record.transitions) for record in records) / game_count if game_count else 0.0,
        "black_wins": sum(1 for record in records if record.winner == "black"),
        "white_wins": sum(1 for record in records if record.winner == "white"),
        "draws": sum(1 for record in records if record.winner is None),
        "max_plies_draw_count": max_plies_draw_count,
        "max_plies_draw_rate": max_plies_draw_count / game_count if game_count else 0.0,
        "game_over_count": game_over_count,
        "game_over_rate": game_over_count / game_count if game_count else 0.0,
    }
    if wall_time_sec is not None:
        summary["generation_wall_time_sec"] = wall_time_sec
        total_plies = sum(len(record.transitions) for record in records)
        summary["plies_per_sec"] = total_plies / wall_time_sec if wall_time_sec > 0.0 else 0.0
    inference_performance = _performance_summary(
        transition.decision_telemetry.move_performance
        for record in records
        for transition in record.transitions
        if transition.decision_telemetry is not None and transition.decision_telemetry.move_performance is not None
    )
    if inference_performance is not None:
        summary["inference_performance"] = inference_performance
    multi_position_search_performance = _performance_summary(
        transition.decision_telemetry.multi_position_search_performance
        for record in records
        for transition in record.transitions
        if transition.decision_telemetry is not None and transition.decision_telemetry.multi_position_search_performance is not None
    )
    if multi_position_search_performance is not None:
        summary["multi_position_search_performance"] = multi_position_search_performance
    return summary


def _play_multi_position_checkpoint_mcts_games(
    config: ShogiGenerationConfig,
    *,
    checkpoint_evaluator_cls: type[ShogiMoveChoiceCheckpointEvaluator],
    record_callback: ShogiGameRecordCallback | None,
    progress_callback: GenerationProgressCallback | None,
) -> tuple[ShogiGameRecord, ...]:
    _validate_multi_position_checkpoint_mcts_config(config)
    black_actor = _checkpoint_actor(
        config.black,
        name="black",
        concurrent_games_per_process=config.concurrent_games_per_process,
        board_backend=config.board_backend,
    )
    white_actor = _checkpoint_actor(
        config.white,
        name="white",
        concurrent_games_per_process=config.concurrent_games_per_process,
        board_backend=config.board_backend,
    )
    black_selector = _checkpoint_selector(
        config.black,
        board_backend=config.board_backend,
        evaluator_cls=checkpoint_evaluator_cls,
    )
    white_selector = _checkpoint_selector(
        config.white,
        board_backend=config.board_backend,
        evaluator_cls=checkpoint_evaluator_cls,
    )
    games = [
        _ActiveGeneratedGame(
            black_actor=black_actor,
            white_actor=white_actor,
            board_backend=config.board_backend,
            start_position=_start_position_for_game(config, index),
        )
        for index in range(config.games)
    ]
    remaining = set(range(config.games))
    started_at = perf_counter()
    for ply in range(config.max_plies):
        if not remaining:
            break
        black_indexes = [index for index in sorted(remaining) if board_is_black_turn(games[index].board)]
        white_indexes = [index for index in sorted(remaining) if not board_is_black_turn(games[index].board)]
        for indexes, selector in ((black_indexes, black_selector), (white_indexes, white_selector)):
            for offset in range(0, len(indexes), config.concurrent_games_per_process):
                batch_indexes = indexes[offset : offset + config.concurrent_games_per_process]
                positions = [UsiPosition(games[index].position_command()) for index in batch_indexes]
                results = selector.select_moves(positions)
                multi_position_search_performance = _performance_payload(selector.last_multi_position_search_performance)
                for game_index, result in zip(batch_indexes, results, strict=True):
                    telemetry = ShogiDecisionTelemetry(
                        move_performance=_performance_payload(result.performance),
                        multi_position_search_performance=multi_position_search_performance,
                        search_evidence=result.search_evidence,
                    )
                    if game_index in remaining and games[game_index].apply_move(result.move, telemetry):
                        remaining.remove(game_index)
                        if record_callback is not None:
                            record_callback(games[game_index].to_record())
        if config.progress_every_plies and (ply + 1) % config.progress_every_plies == 0:
            _emit_progress(
                _progress_payload(games, remaining=remaining, ply=ply + 1, elapsed_sec=perf_counter() - started_at),
                progress_callback=progress_callback,
            )
    records = tuple(game.to_record() for game in games)
    if record_callback is not None:
        completed_indexes = set(range(config.games)) - remaining
        for index, record in enumerate(records):
            if index not in completed_indexes:
                record_callback(record)
    return records


def _player_context(
    spec: PlayerSpec,
    *,
    name: str,
    static_player: BuiltPlayer | None,
):
    if static_player is not None:
        return nullcontext(static_player)
    return player_context(spec, name=name)


def _validate_start_position_count(config: ShogiGenerationConfig) -> None:
    if config.start_positions and len(config.start_positions) != config.games:
        raise ValueError("start_positions must be empty or contain one start position per generated game")


def _start_position_for_game(config: ShogiGenerationConfig, game_index: int) -> StartPosition | None:
    if not config.start_positions:
        return None
    return config.start_positions[game_index]


class _ActiveGeneratedGame:
    def __init__(
        self,
        *,
        black_actor: ShogiActorSpec,
        white_actor: ShogiActorSpec,
        board_backend: str,
        start_position: StartPosition | None = None,
    ) -> None:
        self.start_position = start_position or startpos()
        self.board = board_from_position(self.start_position.usi_position, backend=board_backend)
        self.black_actor = black_actor
        self.white_actor = white_actor
        self.initial_position_sfen = self.board.sfen()
        self.transitions: list[ShogiTransitionRecord] = []
        self.end_reason = "max_plies"
        self.winner: str | None = None

    @property
    def moves(self) -> tuple[str, ...]:
        return tuple(transition.action_usi for transition in self.transitions)

    def position_command(self) -> str:
        return position_command(self.moves, start_position=self.start_position)

    def apply_move(self, move: str, telemetry: ShogiDecisionTelemetry | None) -> bool:
        side = board_turn_name(self.board)
        legal_moves = legal_move_usis(self.board)
        position_sfen = self.board.sfen()
        if move == "resign" or move not in legal_moves:
            self.winner = "white" if board_is_black_turn(self.board) else "black"
            self.end_reason = "resign" if move == "resign" else "illegal_move"
            self._finalize_rewards()
            return True
        self.board.push_usi(move)
        done = self.board.is_game_over()
        self.winner = "black" if done and not board_is_black_turn(self.board) else "white" if done else None
        self.transitions.append(
            ShogiTransitionRecord(
                ply=len(self.transitions),
                side=side,
                position_sfen=position_sfen,
                legal_moves=legal_moves,
                action_usi=move,
                next_position_sfen=self.board.sfen(),
                reward=_transition_reward(side=side, winner=self.winner, done=done),
                done=done,
                decision_telemetry=telemetry,
            )
        )
        if done:
            self.end_reason = "game_over"
            return True
        return False

    def to_record(self) -> ShogiGameRecord:
        if self.end_reason == "max_plies":
            self._finalize_rewards()
        return ShogiGameRecord(
            black_actor=self.black_actor,
            white_actor=self.white_actor,
            initial_position_sfen=self.initial_position_sfen,
            transitions=tuple(self.transitions),
            end_reason=self.end_reason,
            winner=self.winner,
        )

    def _finalize_rewards(self) -> None:
        self.transitions = [
            ShogiTransitionRecord(
                ply=transition.ply,
                side=transition.side,
                position_sfen=transition.position_sfen,
                legal_moves=transition.legal_moves,
                action_usi=transition.action_usi,
                next_position_sfen=transition.next_position_sfen,
                reward=_transition_reward(side=transition.side, winner=self.winner, done=True),
                done=True,
                decision_usi_info_lines=transition.decision_usi_info_lines,
                decision_telemetry=transition.decision_telemetry,
            )
            for transition in self.transitions
        ]


def _validate_multi_position_checkpoint_mcts_config(config: ShogiGenerationConfig) -> None:
    for player in (config.black, config.white):
        if not isinstance(player, CheckpointPolicyPlayerSpec):
            raise SystemExit("--concurrent-games-per-process currently supports checkpoint-vs-checkpoint generation only")
        if player.move_selector != "mcts":
            raise SystemExit("--concurrent-games-per-process currently supports checkpoint MCTS players only")
        if player.mcts_move_time_limit_sec is not None:
            raise SystemExit("--concurrent-games-per-process does not support move time limits yet")
        if player.mcts_root_reuse:
            raise SystemExit(
                "--concurrent-games-per-process uses MultiPositionMctsSearchExecutor, which does not maintain per-game search sessions"
            )


def _checkpoint_selector(
    player: CheckpointPolicyPlayerSpec,
    *,
    board_backend: str,
    evaluator_cls: type[ShogiMoveChoiceCheckpointEvaluator],
) -> MultiPositionMctsSearchExecutor:
    evaluator = evaluator_cls.from_checkpoint(
        player.checkpoint,
        device=player.device,
    )
    return MultiPositionMctsSearchExecutor(
        evaluator=evaluator,
        config=MctsConfig(
            simulation_count=player.mcts_simulations,
            nn_leaf_eval_batch_limit=player.mcts_nn_leaf_eval_batch_limit,
            board_backend=board_backend,
            root_reuse=player.mcts_root_reuse,
        ),
        move_selection=_move_selection_config(
            player.move_selection_profile,
            seed=player.seed,
            temperature=player.move_selection_temperature,
            temperature_plies=player.move_selection_temperature_plies,
        ),
    )


def _checkpoint_actor(
    player: CheckpointPolicyPlayerSpec,
    *,
    name: str,
    concurrent_games_per_process: int,
    board_backend: str,
) -> ShogiActorSpec:
    return ShogiActorSpec(
        kind="checkpoint",
        name=name,
        settings={
            "checkpoint": player.checkpoint,
            "checkpoint_id": player.checkpoint_id,
            "checkpoint_path": player.checkpoint,
            "move_selection_profile": player.move_selection_profile,
            "move_selection_temperature": player.move_selection_temperature,
            "move_selection_temperature_plies": player.move_selection_temperature_plies,
            "move_selector": player.move_selector,
            "mcts_simulations_per_move": player.mcts_simulations,
            "nn_leaf_eval_batch_limit": player.mcts_nn_leaf_eval_batch_limit,
            "move_time_limit_sec": player.mcts_move_time_limit_sec,
            "root_reuse": player.mcts_root_reuse,
            "device": player.device,
            "concurrent_games_per_process": concurrent_games_per_process,
            "board_backend": board_backend,
            "seed": player.seed,
        },
    )


def _move_selection_config(
    profile: str,
    *,
    seed: int | None = None,
    temperature: float | None = None,
    temperature_plies: int | None = None,
):
    if profile == "visit-sampling":
        kwargs: dict[str, object] = {"seed": seed}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if temperature_plies is not None:
            kwargs["temperature_plies"] = temperature_plies
        return visit_sampling_move_selection_config(**kwargs)
    raise ValueError(f"unsupported move selection profile: {profile}")


def _performance_payload(performance: object | None) -> dict[str, object] | None:
    if performance is None:
        return None
    return asdict(performance)


def _progress_payload(
    games: list[_ActiveGeneratedGame],
    *,
    remaining: set[int],
    ply: int,
    elapsed_sec: float,
) -> dict[str, object]:
    completed = len(games) - len(remaining)
    active_plies = [len(games[index].transitions) for index in remaining]
    return {
        "ply": ply,
        "elapsed_sec": elapsed_sec,
        "completed_games": completed,
        "remaining_games": len(remaining),
        "active_plies_avg": mean(active_plies) if active_plies else 0.0,
        "active_plies_max": max(active_plies) if active_plies else 0,
    }


def _emit_progress(payload: dict[str, object], *, progress_callback: GenerationProgressCallback | None) -> None:
    if progress_callback is not None:
        progress_callback(payload)
        return
    print("progress " + json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)


def _transition_reward(*, side: str, winner: str | None, done: bool) -> float:
    if not done or winner is None:
        return 0.0
    return 1.0 if side == winner else -1.0


def _performance_summary(samples: Iterable[dict[str, object] | None]) -> dict[str, Any] | None:
    sample_list = [sample for sample in samples if sample is not None]
    if not sample_list:
        return None
    summary: dict[str, Any] = {"sample_count": len(sample_list)}
    for key in (
        "request_wall_time_sec",
        "model_call_count",
        "model_wall_time_sec",
        "non_model_wall_time_sec",
        "output_count",
        "output_per_sec",
        "position_count",
        "completed_simulations",
    ):
        values = [sample[key] for sample in sample_list if isinstance(sample.get(key), int | float)]
        if values:
            summary[f"{key}_avg"] = mean(values)
            summary[f"{key}_max"] = max(values)
    _add_actual_leaf_eval_batch_summary(summary, sample_list)
    phase_totals: dict[str, float] = {}
    for sample in sample_list:
        phase_times = sample.get("phase_wall_time_sec")
        if not isinstance(phase_times, dict):
            continue
        for name, elapsed in phase_times.items():
            if isinstance(elapsed, int | float):
                phase_totals[name] = phase_totals.get(name, 0.0) + float(elapsed)
    if phase_totals:
        summary["phase_wall_time_sec_total"] = dict(sorted(phase_totals.items()))
        summary["phase_wall_time_sec_avg"] = {
            name: elapsed / len(sample_list) for name, elapsed in sorted(phase_totals.items())
        }
    return summary


def _add_actual_leaf_eval_batch_summary(summary: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    avg_values = [
        sample["actual_nn_leaf_eval_batch_size_avg"]
        for sample in samples
        if isinstance(sample.get("actual_nn_leaf_eval_batch_size_avg"), int | float)
    ]
    max_values = [
        sample["actual_nn_leaf_eval_batch_size_max"]
        for sample in samples
        if isinstance(sample.get("actual_nn_leaf_eval_batch_size_max"), int | float)
    ]
    count_values = [
        sample["actual_nn_leaf_eval_batch_count"]
        for sample in samples
        if isinstance(sample.get("actual_nn_leaf_eval_batch_count"), int | float)
    ]
    fill_ratio_values = [
        sample["actual_nn_leaf_eval_batch_size_fill_ratio_avg"]
        for sample in samples
        if isinstance(sample.get("actual_nn_leaf_eval_batch_size_fill_ratio_avg"), int | float)
    ]
    histogram: dict[int, int] = {}
    for sample in samples:
        sample_histogram = sample.get("actual_nn_leaf_eval_batch_size_histogram")
        if not isinstance(sample_histogram, dict):
            continue
        for size, count in sample_histogram.items():
            if not isinstance(count, int):
                continue
            histogram[int(size)] = histogram.get(int(size), 0) + count
    if not avg_values or not max_values:
        return
    summary["actual_nn_leaf_eval_batch_size_avg"] = mean(avg_values)
    summary["actual_nn_leaf_eval_batch_size_max"] = max(max_values)
    if count_values:
        summary["actual_nn_leaf_eval_batch_count_avg"] = mean(count_values)
        summary["actual_nn_leaf_eval_batch_count_max"] = max(count_values)
    if fill_ratio_values:
        summary["actual_nn_leaf_eval_batch_size_fill_ratio_avg"] = mean(fill_ratio_values)
    if histogram:
        summary["actual_nn_leaf_eval_batch_size_histogram"] = dict(sorted(histogram.items()))
