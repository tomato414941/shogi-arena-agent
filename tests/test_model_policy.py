import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

import torch

from shogi_arena_agent.mcts_config import MctsConfig
from shogi_arena_agent.mcts_move_selector import MctsMoveSelector
from shogi_arena_agent.shogi_game import play_shogi_game
from shogi_arena_agent.model_policy import DirectMovePolicy, RankedMovePolicy, ShogiMoveChoiceCheckpointEvaluator
from shogi_arena_agent.usi import UsiEngine, UsiPosition, board_from_position


class RankedMovePolicyTest(unittest.TestCase):
    def test_returns_highest_scored_legal_move(self) -> None:
        position = UsiPosition(command="position startpos")
        legal_moves = tuple(sorted(move.usi() for move in board_from_position(position).legal_moves))
        preferred_move = legal_moves[-1]

        def rank_moves(position_sfen: str, candidate_moves: tuple[str, ...]) -> list[float]:
            self.assertTrue(position_sfen)
            return [1.0 if move == preferred_move else 0.0 for move in candidate_moves]

        move = RankedMovePolicy(rank_moves).select_move(position)

        self.assertEqual(move, preferred_move)

    def test_ranked_policy_can_play_shogi_game(self) -> None:
        def rank_first(_position_sfen: str, candidate_moves: tuple[str, ...]) -> list[float]:
            return [float(len(candidate_moves) - index) for index, _move in enumerate(candidate_moves)]

        result = play_shogi_game(
            black=UsiEngine(policy=RankedMovePolicy(rank_first)),
            white=UsiEngine(),
            max_plies=4,
        )

        self.assertEqual(result.end_reason, "max_plies")
        self.assertEqual(len(result.transitions), 4)

    def test_checkpoint_evaluator_wraps_position_callable(self) -> None:
        def evaluate_positions(
            requests: Sequence[tuple[str, tuple[str, ...]]],
        ) -> list[tuple[dict[str, float], float]]:
            return [({move: 1.0 for move in candidate_moves}, 0.25) for _position_sfen, candidate_moves in requests]

        board = board_from_position(UsiPosition(command="position startpos"))
        legal_moves = tuple(sorted(move.usi() for move in board.legal_moves))
        evaluator = ShogiMoveChoiceCheckpointEvaluator(evaluate_positions)

        priors, value = evaluator.evaluate(board, legal_moves)

        self.assertEqual(set(priors), set(legal_moves))
        self.assertEqual(value, 0.25)

    def test_checkpoint_evaluator_can_evaluate_batch(self) -> None:
        calls: list[tuple[str, tuple[str, ...]]] = []

        def evaluate_positions(
            requests: Sequence[tuple[str, tuple[str, ...]]],
        ) -> list[tuple[dict[str, float], float]]:
            calls.extend(requests)
            return [
                ({move: float(index + move_index) for move_index, move in enumerate(candidate_moves)}, float(index))
                for index, (_position_sfen, candidate_moves) in enumerate(requests)
            ]

        first_board = board_from_position(UsiPosition(command="position startpos"))
        second_board = board_from_position(UsiPosition(command="position startpos moves 7g7f 3c3d"))
        first_moves = tuple(sorted(move.usi() for move in first_board.legal_moves))
        second_moves = tuple(sorted(move.usi() for move in second_board.legal_moves))
        evaluator = ShogiMoveChoiceCheckpointEvaluator(evaluate_positions)

        evaluations = evaluator.evaluate_batch(((first_board, first_moves), (second_board, second_moves)))

        self.assertEqual(len(evaluations), 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(set(evaluations[0][0]), set(first_moves))
        self.assertEqual(set(evaluations[1][0]), set(second_moves))
        self.assertEqual(evaluations[0][1], 0.0)
        self.assertEqual(evaluations[1][1], 1.0)

    def test_direct_policy_uses_evaluator(self) -> None:
        def evaluate_positions(
            requests: Sequence[tuple[str, tuple[str, ...]]],
        ) -> list[tuple[dict[str, float], float]]:
            return [
                ({move: 1.0 if move == "7g7f" else 0.0 for move in candidate_moves}, 0.5)
                for _position_sfen, candidate_moves in requests
            ]

        evaluator = ShogiMoveChoiceCheckpointEvaluator(evaluate_positions)

        move = DirectMovePolicy(evaluator).select_move(UsiPosition())

        self.assertEqual(move, "7g7f")

    def test_policy_plane_checkpoint_evaluator_feeds_mcts_priors(self) -> None:
        from intrep.problems.shogi_policy_value.checkpoint import save_shogi_policy_value_model_checkpoint
        from intrep.problems.shogi_policy_value.model import SHOGI_POLICY_VALUE_MODEL_POLICY_PLANE_SHARED_TRANSFORMER
        from intrep.problems.shogi_policy_value.training import ShogiPolicyValueTrainingConfig, build_shogi_policy_value_model
        from intrep.worlds.shogi.policy_plane import shogi_policy_plane_action_index

        position = UsiPosition()
        board = board_from_position(position)
        legal_moves = tuple(sorted(move.usi() for move in board.legal_moves))
        preferred_move = "7g7f"
        action_index = shogi_policy_plane_action_index(preferred_move, turn=board.turn)
        config = ShogiPolicyValueTrainingConfig(
            model=SHOGI_POLICY_VALUE_MODEL_POLICY_PLANE_SHARED_TRANSFORMER,
            embedding_dim=8,
            hidden_dim=16,
            num_heads=2,
            num_layers=1,
        )
        model = build_shogi_policy_value_model(config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.policy_head.scorer[-1].bias[action_index] = 8.0

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "policy-plane.pt"
            save_shogi_policy_value_model_checkpoint(checkpoint_path, model, config)
            evaluator = ShogiMoveChoiceCheckpointEvaluator.from_checkpoint(checkpoint_path)

        priors, value = evaluator.evaluate(board, legal_moves)
        selected = MctsMoveSelector(evaluator, config=MctsConfig(simulation_count=8)).select_move(position)

        self.assertEqual(max(priors, key=priors.get), preferred_move)
        self.assertAlmostEqual(sum(priors.values()), 1.0, places=6)
        self.assertEqual(value, 0.0)
        self.assertEqual(selected, preferred_move)


if __name__ == "__main__":
    unittest.main()
