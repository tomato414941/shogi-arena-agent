import unittest

from collections.abc import Sequence

import shogi

from shogi_arena_agent.shogi_game import play_shogi_game
from shogi_arena_agent.mcts_policy import (
    BatchedMctsMoveSelector,
    MctsConfig,
    MctsPolicy,
    PolicyValueEvaluator,
    self_play_move_selection_config,
)
from shogi_arena_agent.usi import UsiEngine, UsiPosition, board_from_position


class PriorOnlyEvaluator:
    def __init__(self, preferred_move: str) -> None:
        self.preferred_move = preferred_move

    def evaluate_batch(
        self,
        requests: Sequence[tuple[shogi.Board, tuple[str, ...]]],
    ) -> list[tuple[dict[str, float], float]]:
        return [
            ({move: 1.0 if move == self.preferred_move else 0.0 for move in legal_moves}, 0.0)
            for _board, legal_moves in requests
        ]


class CaptureValueEvaluator:
    def evaluate_batch(
        self,
        requests: Sequence[tuple[shogi.Board, tuple[str, ...]]],
    ) -> list[tuple[dict[str, float], float]]:
        evaluations: list[tuple[dict[str, float], float]] = []
        for board, legal_moves in requests:
            if "8h2b+" in legal_moves:
                evaluations.append(({move: 1.0 if move == "8h2b+" else 0.1 for move in legal_moves}, 0.0))
                continue
            evaluations.append(({move: 1.0 for move in legal_moves}, _captured_piece_value(board)))
        return evaluations


class FinalSelectionValueEvaluator:
    def evaluate_batch(
        self,
        requests: Sequence[tuple[shogi.Board, tuple[str, ...]]],
    ) -> list[tuple[dict[str, float], float]]:
        evaluations: list[tuple[dict[str, float], float]] = []
        for board, legal_moves in requests:
            if board.move_number == 1:
                evaluations.append(({"7g7f": 1.0, "2g2f": 1.0}, 0.0))
            elif board.move_stack[-1].usi() == "7g7f":
                evaluations.append(({move: 1.0 for move in legal_moves}, -0.5))
            elif board.move_stack[-1].usi() == "2g2f":
                evaluations.append(({move: 1.0 for move in legal_moves}, 0.5))
            else:
                evaluations.append(({move: 1.0 for move in legal_moves}, 0.0))
        return evaluations


class BatchCountingEvaluator:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def evaluate_batch(
        self,
        requests: Sequence[tuple[shogi.Board, tuple[str, ...]]],
    ) -> list[tuple[dict[str, float], float]]:
        self.batch_sizes.append(len(requests))
        return [({move: 1.0 for move in legal_moves}, 0.0) for _board, legal_moves in requests]


class MctsPolicyTest(unittest.TestCase):
    def test_returns_legal_move(self) -> None:
        move = MctsPolicy(config=MctsConfig(simulation_count=4)).select_move(UsiPosition())

        board = shogi.Board()
        self.assertIn(move, {legal_move.usi() for legal_move in board.legal_moves})

    def test_policy_prior_guides_search(self) -> None:
        position = UsiPosition()
        legal_moves = tuple(sorted(move.usi() for move in board_from_position(position).legal_moves))
        preferred_move = "7g7f"
        self.assertIn(preferred_move, legal_moves)

        move = MctsPolicy(
            PriorOnlyEvaluator(preferred_move),
            config=MctsConfig(simulation_count=8),
        ).select_move(position)

        self.assertEqual(move, preferred_move)

    def test_records_visit_count_policy_targets(self) -> None:
        policy = MctsPolicy(config=MctsConfig(simulation_count=8))

        policy.select_move(UsiPosition())

        self.assertIsNotNone(policy.last_policy_targets)
        self.assertAlmostEqual(sum(policy.last_policy_targets.values()), 1.0)
        self.assertIn("7g7f", policy.last_policy_targets)

    def test_records_move_performance(self) -> None:
        policy = MctsPolicy(config=MctsConfig(simulation_count=4))

        policy.select_move(UsiPosition())

        self.assertIsNotNone(policy.last_performance)
        assert policy.last_performance is not None
        self.assertEqual(policy.last_performance.output_count, 4)
        self.assertGreater(policy.last_performance.request_wall_time_sec, 0.0)
        self.assertGreater(policy.last_performance.model_call_count, 0)
        self.assertGreaterEqual(policy.last_performance.model_wall_time_sec, 0.0)
        self.assertGreaterEqual(policy.last_performance.non_model_wall_time_sec, 0.0)

    def test_batches_leaf_evaluations(self) -> None:
        evaluator = BatchCountingEvaluator()
        policy = MctsPolicy(evaluator, config=MctsConfig(simulation_count=8, evaluation_batch_size=4))

        policy.select_move(UsiPosition())

        self.assertIsNotNone(policy.last_performance)
        assert policy.last_performance is not None
        self.assertEqual(policy.last_performance.output_count, 8)
        self.assertLess(policy.last_performance.model_call_count, 9)
        self.assertIn(4, evaluator.batch_sizes)

    def test_self_play_selection_can_sample_different_root_moves(self) -> None:
        selector = BatchedMctsMoveSelector(
            config=MctsConfig(simulation_count=8, evaluation_batch_size=8),
            move_selection=self_play_move_selection_config(seed=1),
        )

        results = selector.select_moves([UsiPosition(), UsiPosition(), UsiPosition(), UsiPosition()])

        self.assertGreater(len({result.move for result in results}), 1)

    def test_batched_selector_batches_across_positions(self) -> None:
        evaluator = BatchCountingEvaluator()
        selector = BatchedMctsMoveSelector(evaluator, config=MctsConfig(simulation_count=4, evaluation_batch_size=8))

        results = selector.select_moves(
            [
                UsiPosition(),
                UsiPosition(command="position startpos moves 7g7f 3c3d"),
            ]
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.move != "resign" for result in results))
        self.assertIn(2, evaluator.batch_sizes)

    def test_batched_selector_records_phase_timings(self) -> None:
        selector = BatchedMctsMoveSelector(config=MctsConfig(simulation_count=4, evaluation_batch_size=8))

        result = selector.select_moves([UsiPosition()])[0]

        phases = result.performance.phase_wall_time_sec
        self.assertGreater(phases["legal_moves"], 0.0)
        self.assertGreater(phases["board_copy"], 0.0)
        self.assertGreater(phases["selection"], 0.0)
        self.assertGreater(phases["expand"], 0.0)
        self.assertGreater(phases["backup"], 0.0)

    def test_batched_selector_records_batch_performance(self) -> None:
        selector = BatchedMctsMoveSelector(config=MctsConfig(simulation_count=4, evaluation_batch_size=8))

        selector.select_moves([UsiPosition(), UsiPosition()])

        performance = selector.last_batch_performance
        self.assertIsNotNone(performance)
        assert performance is not None
        self.assertEqual(performance.position_count, 2)
        self.assertEqual(performance.completed_simulations, 8)
        self.assertGreater(performance.request_wall_time_sec, 0.0)
        self.assertGreater(performance.model_call_count, 0)
        self.assertGreater(performance.phase_wall_time_sec["legal_moves"], 0.0)

    def test_batched_selector_supports_cshogi_backend(self) -> None:
        selector = BatchedMctsMoveSelector(config=MctsConfig(simulation_count=4, evaluation_batch_size=8, board_backend="cshogi"))

        results = selector.select_moves([UsiPosition(), UsiPosition(command="position startpos moves 7g7f 3c3d")])

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.move != "resign" for result in results))
        self.assertIsNotNone(selector.last_batch_performance)

    def test_move_time_limit_can_stop_before_simulation_limit(self) -> None:
        policy = MctsPolicy(config=MctsConfig(simulation_count=8, move_time_limit_sec=0.0))

        policy.select_move(UsiPosition())

        self.assertIsNotNone(policy.last_performance)
        assert policy.last_performance is not None
        self.assertEqual(policy.last_performance.output_count, 0)

    def test_value_guides_search_after_expansion(self) -> None:
        position = UsiPosition(command="position startpos moves 7g7f 3c3d")

        move = MctsPolicy(
            CaptureValueEvaluator(),
            config=MctsConfig(simulation_count=32, c_puct=1.5),
        ).select_move(position)

        self.assertEqual(move, "8h2b+")

    def test_final_selection_uses_root_player_value(self) -> None:
        move = MctsPolicy(
            FinalSelectionValueEvaluator(),
            config=MctsConfig(simulation_count=2, c_puct=1.5),
        ).select_move(UsiPosition())

        self.assertEqual(move, "7g7f")

    def test_can_play_shogi_game(self) -> None:
        player = UsiEngine(
            policy=MctsPolicy(
                PriorOnlyEvaluator("7g7f"),
                config=MctsConfig(simulation_count=4),
            )
        )

        result = play_shogi_game(black=player, white=UsiEngine(), max_plies=4)

        self.assertEqual(result.end_reason, "max_plies")
        self.assertEqual(len(result.transitions), 4)

    def test_rejects_invalid_config(self) -> None:
        with self.assertRaises(ValueError):
            MctsConfig(simulation_count=0)
        with self.assertRaises(ValueError):
            MctsConfig(c_puct=0.0)
        with self.assertRaises(ValueError):
            MctsConfig(evaluation_batch_size=0)
        with self.assertRaises(ValueError):
            MctsConfig(move_time_limit_sec=-1.0)


def _captured_piece_value(board: shogi.Board) -> float:
    black_hand_count = sum(board.pieces_in_hand[shogi.BLACK].values())
    white_hand_count = sum(board.pieces_in_hand[shogi.WHITE].values())
    if black_hand_count > white_hand_count:
        return 1.0 if board.turn == shogi.BLACK else -1.0
    if white_hand_count > black_hand_count:
        return 1.0 if board.turn == shogi.WHITE else -1.0
    return 0.0


if __name__ == "__main__":
    unittest.main()
