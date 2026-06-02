from __future__ import annotations

import itertools
import queue
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter

from shogi_arena_agent.board_backend import ShogiBoard
from shogi_arena_agent.mcts_performance import leaf_eval_batch_metrics

PositionEvaluation = tuple[dict[str, float], float]
PositionEvaluationRequest = tuple[str, tuple[str, ...]]
EvaluatePositions = Callable[[Sequence[PositionEvaluationRequest]], list[PositionEvaluation]]


@dataclass(frozen=True)
class EvaluationRequest:
    request_id: int
    position_sfen: str
    legal_moves: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationResult:
    request_id: int
    priors: dict[str, float]
    value: float


class CentralPolicyValueEvaluator:
    """Batch policy/value requests for one central evaluator backend."""

    def __init__(
        self,
        evaluate_positions: EvaluatePositions,
        *,
        batch_size_limit: int,
        flush_timeout_sec: float = 0.002,
    ) -> None:
        if batch_size_limit <= 0:
            raise ValueError("batch_size_limit must be positive")
        if flush_timeout_sec < 0.0:
            raise ValueError("flush_timeout_sec must be non-negative")
        self.evaluate_positions = evaluate_positions
        self.batch_size_limit = batch_size_limit
        self.flush_timeout_sec = flush_timeout_sec
        self.actual_batch_sizes: list[int] = []
        self.model_call_count = 0
        self.model_wall_time_sec = 0.0
        self._request_queue: queue.Queue[tuple[EvaluationRequest, queue.Queue[EvaluationResult]]] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="central-policy-value-evaluator", daemon=True)
        self._started = False

    def __enter__(self) -> CentralPolicyValueEvaluator:
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def close(self) -> None:
        if not self._started:
            return
        self._stop.set()
        self._thread.join()

    def client(self) -> "QueuedPolicyValueEvaluator":
        if not self._started:
            raise RuntimeError("central evaluator must be started before creating clients")
        return QueuedPolicyValueEvaluator(self._request_queue)

    def performance_summary(self) -> dict[str, object]:
        return {
            "model_call_count": self.model_call_count,
            "model_wall_time_sec": self.model_wall_time_sec,
            **leaf_eval_batch_metrics(
                self.actual_batch_sizes,
                batch_size_limit=self.batch_size_limit,
            ),
        }

    def _run(self) -> None:
        while not self._stop.is_set() or not self._request_queue.empty():
            batch = self._read_batch()
            if not batch:
                continue
            requests = tuple(item[0] for item in batch)
            responses = tuple(item[1] for item in batch)
            started_at = perf_counter()
            evaluations = self.evaluate_positions(tuple((request.position_sfen, request.legal_moves) for request in requests))
            elapsed = perf_counter() - started_at
            self.model_call_count += 1
            self.model_wall_time_sec += elapsed
            self.actual_batch_sizes.append(len(requests))
            if len(evaluations) != len(requests):
                raise ValueError("central evaluator backend must return one evaluation per request")
            for request, response_queue, (priors, value) in zip(requests, responses, evaluations, strict=True):
                response_queue.put(EvaluationResult(request.request_id, priors, float(value)))

    def _read_batch(self) -> list[tuple[EvaluationRequest, queue.Queue[EvaluationResult]]]:
        try:
            first = self._request_queue.get(timeout=self.flush_timeout_sec if not self._stop.is_set() else 0.0)
        except queue.Empty:
            return []
        batch = [first]
        deadline = perf_counter() + self.flush_timeout_sec
        while len(batch) < self.batch_size_limit:
            timeout = max(0.0, deadline - perf_counter())
            try:
                batch.append(self._request_queue.get(timeout=timeout))
            except queue.Empty:
                break
        return batch


class QueuedPolicyValueEvaluator:
    """MCTS-side evaluator client that does not own the model backend."""

    def __init__(self, request_queue: queue.Queue[tuple[EvaluationRequest, queue.Queue[EvaluationResult]]]) -> None:
        self._request_queue = request_queue
        self._request_ids = itertools.count()

    def evaluate_batch(
        self,
        requests: Sequence[tuple[ShogiBoard, tuple[str, ...]]],
    ) -> list[PositionEvaluation]:
        if not requests:
            return []
        response_queue: queue.Queue[EvaluationResult] = queue.Queue()
        request_ids: list[int] = []
        for board, legal_moves in requests:
            request_id = next(self._request_ids)
            request_ids.append(request_id)
            self._request_queue.put((EvaluationRequest(request_id, board.sfen(), tuple(legal_moves)), response_queue))
        remaining = set(request_ids)
        results: dict[int, EvaluationResult] = {}
        while remaining:
            result = response_queue.get()
            if result.request_id not in remaining:
                continue
            results[result.request_id] = result
            remaining.remove(result.request_id)
        return [(results[request_id].priors, results[request_id].value) for request_id in request_ids]
