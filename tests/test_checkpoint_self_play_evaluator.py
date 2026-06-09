import threading
import unittest
from collections.abc import Sequence

import shogi

from shogi_arena_agent.checkpoint_self_play_evaluator import (
    CentralPolicyValueEvaluator,
    PositionEvaluation,
    PositionEvaluationRequest,
)


class RecordingPositionEvaluator:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.last_performance: dict[str, float] = {}

    def __call__(self, requests: Sequence[PositionEvaluationRequest]) -> list[PositionEvaluation]:
        self.batch_sizes.append(len(requests))
        self.last_performance = {"model_forward_sec": 0.5, "output_decode_sec": 0.25}
        return [({move: 1.0 for move in legal_moves}, 0.25) for _sfen, legal_moves in requests]


class CentralPolicyValueEvaluatorTest(unittest.TestCase):
    def test_batches_requests_from_multiple_clients(self) -> None:
        backend = RecordingPositionEvaluator()
        board = shogi.Board()
        legal_moves = tuple(move.usi() for move in board.legal_moves)
        barrier = threading.Barrier(2)
        results: list[PositionEvaluation] = []

        with CentralPolicyValueEvaluator(backend, batch_size_limit=8, flush_timeout_sec=0.05) as central:
            clients = [central.client(), central.client()]

            def evaluate(client_index: int) -> None:
                barrier.wait()
                results.extend(clients[client_index].evaluate_batch(((board, legal_moves),)))

            threads = [threading.Thread(target=evaluate, args=(index,)) for index in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(central.model_call_count, 1)
            self.assertEqual(central.actual_batch_sizes, [2])
            performance = central.performance_summary()
            self.assertEqual(performance["backend_model_forward_sec"], 0.5)
            self.assertEqual(performance["backend_output_decode_sec"], 0.25)

        self.assertEqual(backend.batch_sizes, [2])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(value == 0.25 for _priors, value in results))

    def test_preserves_result_order_within_client_batch(self) -> None:
        def evaluate_positions(requests: Sequence[PositionEvaluationRequest]) -> list[PositionEvaluation]:
            evaluations: list[PositionEvaluation] = []
            for _sfen, legal_moves in requests:
                preferred_move = legal_moves[-1]
                evaluations.append(({move: 1.0 if move == preferred_move else 0.0 for move in legal_moves}, 0.0))
            return evaluations

        first_board = shogi.Board()
        second_board = shogi.Board()
        second_board.push_usi("7g7f")
        first_moves = tuple(move.usi() for move in first_board.legal_moves)
        second_moves = tuple(move.usi() for move in second_board.legal_moves)

        with CentralPolicyValueEvaluator(evaluate_positions, batch_size_limit=8) as central:
            client = central.client()
            evaluations = client.evaluate_batch(((first_board, first_moves), (second_board, second_moves)))

        self.assertEqual(evaluations[0][0][first_moves[-1]], 1.0)
        self.assertEqual(evaluations[1][0][second_moves[-1]], 1.0)

    def test_forwards_optional_action_indices(self) -> None:
        seen_requests: list[PositionEvaluationRequest] = []

        def evaluate_positions(requests: Sequence[PositionEvaluationRequest]) -> list[PositionEvaluation]:
            seen_requests.extend(requests)
            evaluations: list[PositionEvaluation] = []
            for request in requests:
                action_indices = request[2] if len(request) == 3 else None
                prior_count = len(action_indices) if action_indices is not None else len(request[1])
                evaluations.append(([1.0 / prior_count] * prior_count, 0.0))
            return evaluations

        board = shogi.Board()
        legal_moves = tuple(move.usi() for move in board.legal_moves)
        action_indices = tuple(range(len(legal_moves)))

        with CentralPolicyValueEvaluator(evaluate_positions, batch_size_limit=8) as central:
            client = central.client()
            evaluations = client.evaluate_batch(((board, legal_moves, action_indices),))

        self.assertEqual(seen_requests, [(board.sfen(), legal_moves, action_indices)])
        self.assertEqual(len(evaluations[0][0]), len(action_indices))

    def test_propagates_backend_failure_to_waiting_client(self) -> None:
        def evaluate_positions(_requests: Sequence[PositionEvaluationRequest]) -> list[PositionEvaluation]:
            raise RuntimeError("backend failed")

        board = shogi.Board()
        legal_moves = tuple(move.usi() for move in board.legal_moves)

        with CentralPolicyValueEvaluator(evaluate_positions, batch_size_limit=8) as central:
            client = central.client()
            with self.assertRaisesRegex(RuntimeError, "backend failed"):
                client.evaluate_batch(((board, legal_moves),))
            with self.assertRaisesRegex(RuntimeError, "backend failed"):
                client.evaluate_batch(((board, legal_moves),))

    def test_propagates_backend_shape_failure_to_waiting_client(self) -> None:
        def evaluate_positions(_requests: Sequence[PositionEvaluationRequest]) -> list[PositionEvaluation]:
            return []

        board = shogi.Board()
        legal_moves = tuple(move.usi() for move in board.legal_moves)

        with CentralPolicyValueEvaluator(evaluate_positions, batch_size_limit=8) as central:
            client = central.client()
            with self.assertRaisesRegex(ValueError, "one evaluation per request"):
                client.evaluate_batch(((board, legal_moves),))


if __name__ == "__main__":
    unittest.main()
