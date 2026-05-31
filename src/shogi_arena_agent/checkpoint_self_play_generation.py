from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean
from time import perf_counter

from shogi_arena_agent.board_backend import board_is_black_turn, board_turn_name, legal_move_usis
from shogi_arena_agent.mcts_config import (
    MctsConfig,
    MoveSelectionConfig,
    visit_sampling_move_selection_config,
)
from shogi_arena_agent.model_policy import ShogiMoveChoiceCheckpointEvaluator
from shogi_arena_agent.multi_position_mcts_search_executor import MultiPositionMctsSearchExecutor
from shogi_arena_agent.shogi_game import (
    ShogiActorSpec,
    ShogiDecisionTelemetry,
    ShogiGameRecord,
    ShogiTransitionRecord,
    position_command,
)
from shogi_arena_agent.shogi_generation import records_summary
from shogi_arena_agent.start_positions import StartPosition, startpos
from shogi_arena_agent.usi import RESIGN_MOVE, UsiPosition, board_from_position

GenerationProgressCallback = Callable[[dict[str, object]], None]
ShogiGameRecordCallback = Callable[[ShogiGameRecord], None]


@dataclass(frozen=True)
class CheckpointSelfPlayConfig:
    checkpoint: str
    games: int
    concurrent_games_per_process: int
    max_plies: int
    mcts_simulations: int
    nn_leaf_eval_batch_limit: int
    device: str
    board_backend: str
    checkpoint_id: str | None = None
    move_selection: MoveSelectionConfig | None = None
    progress_every_plies: int = 0
    start_positions: tuple[StartPosition, ...] = ()


def generate_checkpoint_self_play_games(
    config: CheckpointSelfPlayConfig,
    *,
    checkpoint_evaluator_cls: type[ShogiMoveChoiceCheckpointEvaluator] = ShogiMoveChoiceCheckpointEvaluator,
    record_callback: ShogiGameRecordCallback | None = None,
    progress_callback: GenerationProgressCallback | None = None,
) -> tuple[ShogiGameRecord, ...]:
    _validate_config(config)
    move_selection = config.move_selection or visit_sampling_move_selection_config(seed=None)
    actor = _checkpoint_self_play_actor(config, move_selection)
    selector = _checkpoint_self_play_selector(config, move_selection, checkpoint_evaluator_cls=checkpoint_evaluator_cls)
    games = [
        _ActiveCheckpointSelfPlayGame(
            actor=actor,
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
        for indexes in (black_indexes, white_indexes):
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


def summarize_checkpoint_self_play_records(
    records: tuple[ShogiGameRecord, ...],
    *,
    wall_time_sec: float | None = None,
) -> dict[str, object]:
    return records_summary(records, wall_time_sec=wall_time_sec)


class _ActiveCheckpointSelfPlayGame:
    def __init__(
        self,
        *,
        actor: ShogiActorSpec,
        board_backend: str,
        start_position: StartPosition | None = None,
    ) -> None:
        self.start_position = start_position or startpos()
        self.board = board_from_position(self.start_position.usi_position, backend=board_backend)
        self.actor = actor
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
        if move == RESIGN_MOVE or move not in legal_moves:
            self.winner = "white" if board_is_black_turn(self.board) else "black"
            self.end_reason = "resign" if move == RESIGN_MOVE else "illegal_move"
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
            black_actor=self.actor,
            white_actor=self.actor,
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


def _checkpoint_self_play_selector(
    config: CheckpointSelfPlayConfig,
    move_selection: MoveSelectionConfig,
    *,
    checkpoint_evaluator_cls: type[ShogiMoveChoiceCheckpointEvaluator],
) -> MultiPositionMctsSearchExecutor:
    evaluator = checkpoint_evaluator_cls.from_checkpoint(config.checkpoint, device=config.device)
    return MultiPositionMctsSearchExecutor(
        evaluator=evaluator,
        config=MctsConfig(
            simulation_count=config.mcts_simulations,
            nn_leaf_eval_batch_limit=config.nn_leaf_eval_batch_limit,
            board_backend=config.board_backend,
            root_reuse=False,
        ),
        move_selection=move_selection,
    )


def _checkpoint_self_play_actor(config: CheckpointSelfPlayConfig, move_selection: MoveSelectionConfig) -> ShogiActorSpec:
    return ShogiActorSpec(
        kind="checkpoint_self_play",
        name="checkpoint",
        settings={
            "checkpoint": config.checkpoint,
            "checkpoint_id": config.checkpoint_id,
            "checkpoint_path": config.checkpoint,
            "move_selection_profile": "visit-sampling",
            "move_selection_temperature": move_selection.temperature,
            "move_selection_temperature_plies": move_selection.temperature_plies,
            "move_selector": "mcts",
            "mcts_simulations_per_move": config.mcts_simulations,
            "nn_leaf_eval_batch_limit": config.nn_leaf_eval_batch_limit,
            "move_time_limit_sec": None,
            "root_reuse": False,
            "device": config.device,
            "concurrent_games_per_process": config.concurrent_games_per_process,
            "board_backend": config.board_backend,
        },
    )


def _validate_config(config: CheckpointSelfPlayConfig) -> None:
    if config.games <= 0:
        raise ValueError("games must be positive")
    if config.concurrent_games_per_process <= 0:
        raise ValueError("concurrent_games_per_process must be positive")
    if config.max_plies <= 0:
        raise ValueError("max_plies must be positive")
    if config.mcts_simulations <= 0:
        raise ValueError("mcts_simulations must be positive")
    if config.nn_leaf_eval_batch_limit <= 0:
        raise ValueError("nn_leaf_eval_batch_limit must be positive")
    if config.progress_every_plies < 0:
        raise ValueError("progress_every_plies must be non-negative")
    if config.start_positions and len(config.start_positions) != config.games:
        raise ValueError("start_positions must be empty or contain one start position per generated game")


def _start_position_for_game(config: CheckpointSelfPlayConfig, game_index: int) -> StartPosition | None:
    if not config.start_positions:
        return None
    return config.start_positions[game_index]


def _performance_payload(performance: object | None) -> dict[str, object] | None:
    if performance is None:
        return None
    from dataclasses import asdict

    return asdict(performance)


def _progress_payload(
    games: list[_ActiveCheckpointSelfPlayGame],
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
