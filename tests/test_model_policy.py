import unittest

from collections.abc import Sequence

from shogi_arena_agent.shogi_game import play_shogi_game
from shogi_arena_agent.model_policy import RankedMovePolicy, ShogiMoveChoiceCheckpointEvaluator
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
        def evaluate_position(position_sfen: str, candidate_moves: tuple[str, ...]) -> tuple[dict[str, float], float]:
            self.assertTrue(position_sfen)
            return {move: 1.0 for move in candidate_moves}, 0.25

        board = board_from_position(UsiPosition(command="position startpos"))
        legal_moves = tuple(sorted(move.usi() for move in board.legal_moves))
        evaluator = ShogiMoveChoiceCheckpointEvaluator(evaluate_position)

        priors, value = evaluator.evaluate(board, legal_moves)

        self.assertEqual(set(priors), set(legal_moves))
        self.assertEqual(value, 0.25)

    def test_checkpoint_evaluator_can_evaluate_many_with_batch_callable(self) -> None:
        calls: list[tuple[str, tuple[str, ...]]] = []

        def evaluate_position(position_sfen: str, candidate_moves: tuple[str, ...]) -> tuple[dict[str, float], float]:
            raise AssertionError("single evaluator should not be used")

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
        evaluator = ShogiMoveChoiceCheckpointEvaluator(evaluate_position, evaluate_positions)

        evaluations = evaluator.evaluate_many(((first_board, first_moves), (second_board, second_moves)))

        self.assertEqual(len(evaluations), 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(set(evaluations[0][0]), set(first_moves))
        self.assertEqual(set(evaluations[1][0]), set(second_moves))
        self.assertEqual(evaluations[0][1], 0.0)
        self.assertEqual(evaluations[1][1], 1.0)

    def test_checkpoint_evaluator_evaluate_many_falls_back_to_single_callable(self) -> None:
        call_count = 0

        def evaluate_position(position_sfen: str, candidate_moves: tuple[str, ...]) -> tuple[dict[str, float], float]:
            nonlocal call_count
            call_count += 1
            self.assertTrue(position_sfen)
            return {move: 1.0 for move in candidate_moves}, 0.5

        first_board = board_from_position(UsiPosition(command="position startpos"))
        second_board = board_from_position(UsiPosition(command="position startpos moves 7g7f 3c3d"))
        first_moves = tuple(sorted(move.usi() for move in first_board.legal_moves))
        second_moves = tuple(sorted(move.usi() for move in second_board.legal_moves))
        evaluator = ShogiMoveChoiceCheckpointEvaluator(evaluate_position)

        evaluations = evaluator.evaluate_many(((first_board, first_moves), (second_board, second_moves)))

        self.assertEqual(call_count, 2)
        self.assertEqual(len(evaluations), 2)
        self.assertEqual(evaluations[0][1], 0.5)
        self.assertEqual(evaluations[1][1], 0.5)


if __name__ == "__main__":
    unittest.main()
