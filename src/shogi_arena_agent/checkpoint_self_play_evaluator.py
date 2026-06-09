from __future__ import annotations

import itertools
import queue
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from time import perf_counter

from shogi_arena_agent.board_backend import ShogiBoard
from shogi_arena_agent.mcts_evaluator import MovePriors
from shogi_arena_agent.mcts_performance import leaf_eval_batch_metrics

PositionEvaluation = tuple[MovePriors, float]
PositionEvaluationRequest = tuple[str, tuple[str, ...]]
EvaluatePositions = Callable[[Sequence[PositionEvaluationRequest]], list[PositionEvaluation]]


@dataclass(frozen=True)
class EvaluationRequest:
    request_id: int
    position_sfen: str
    legal_moves: tuple[str, ...]
    enqueued_at: float


@dataclass(frozen=True)
class EvaluationResult:
    request_id: int
    priors: MovePriors
    value: float


@dataclass(frozen=True)
class _EvaluationResponse:
    request_id: int
    result: PositionEvaluation | None = None
    error: BaseException | None = None


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
        self.request_queue_wait_seconds: list[float] = []
        self.model_call_count = 0
        self.model_wall_time_sec = 0.0
        self.backend_performance_seconds: dict[str, float] = {}
        self.batch_first_wait_sec = 0.0
        self.batch_fill_wait_sec = 0.0
        self.response_send_wall_time_sec = 0.0
        self._request_queue: queue.Queue[tuple[EvaluationRequest, queue.Queue[_EvaluationResponse]]] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="central-policy-value-evaluator", daemon=True)
        self._started = False
        self._failure: BaseException | None = None

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
        return QueuedPolicyValueEvaluator(self._request_queue, self._current_failure)

    def performance_summary(self) -> dict[str, object]:
        return {
            "request_count": sum(self.actual_batch_sizes),
            "model_call_count": self.model_call_count,
            "model_wall_time_sec": self.model_wall_time_sec,
            **{
                f"backend_{key}": value
                for key, value in sorted(self.backend_performance_seconds.items())
            },
            "batch_first_wait_sec": self.batch_first_wait_sec,
            "batch_fill_wait_sec": self.batch_fill_wait_sec,
            "response_send_wall_time_sec": self.response_send_wall_time_sec,
            "request_queue_wait_sec_avg": (
                sum(self.request_queue_wait_seconds) / len(self.request_queue_wait_seconds)
                if self.request_queue_wait_seconds
                else 0.0
            ),
            "request_queue_wait_sec_max": max(self.request_queue_wait_seconds)
            if self.request_queue_wait_seconds
            else 0.0,
            **leaf_eval_batch_metrics(
                self.actual_batch_sizes,
                batch_size_limit=self.batch_size_limit,
            ),
        }

    def _current_failure(self) -> BaseException | None:
        return self._failure

    def _run(self) -> None:
        while not self._stop.is_set() or not self._request_queue.empty():
            batch = self._read_batch()
            if not batch:
                continue
            requests = tuple(item[0] for item in batch)
            responses = tuple(item[1] for item in batch)
            if self._failure is not None:
                self._send_error_response(requests, responses, self._failure)
                continue
            try:
                ready_at = perf_counter()
                self.request_queue_wait_seconds.extend(ready_at - request.enqueued_at for request in requests)
                started_at = perf_counter()
                evaluations = _evaluate_positions_backend(
                    self.evaluate_positions,
                    tuple((request.position_sfen, request.legal_moves) for request in requests),
                )
                elapsed = perf_counter() - started_at
                self.model_call_count += 1
                self.model_wall_time_sec += elapsed
                _add_backend_performance(self.backend_performance_seconds, self.evaluate_positions)
                self.actual_batch_sizes.append(len(requests))
                if len(evaluations) != len(requests):
                    raise ValueError("central evaluator backend must return one evaluation per request")
            except BaseException as error:
                self._failure = error
                self._send_error_response(requests, responses, error)
                continue
            response_started_at = perf_counter()
            for request, response_queue, evaluation in zip(requests, responses, evaluations, strict=True):
                response_queue.put(_EvaluationResponse(request.request_id, result=evaluation))
            self.response_send_wall_time_sec += perf_counter() - response_started_at

    @staticmethod
    def _send_error_response(
        requests: Sequence[EvaluationRequest],
        responses: Sequence[queue.Queue[_EvaluationResponse]],
        error: BaseException,
    ) -> None:
        for request, response_queue in zip(requests, responses, strict=True):
            response_queue.put(_EvaluationResponse(request.request_id, error=error))

    def _read_batch(self) -> list[tuple[EvaluationRequest, queue.Queue[_EvaluationResponse]]]:
        wait_started_at = perf_counter()
        try:
            first = self._request_queue.get(timeout=self.flush_timeout_sec if not self._stop.is_set() else 0.0)
        except queue.Empty:
            self.batch_first_wait_sec += perf_counter() - wait_started_at
            return []
        self.batch_first_wait_sec += perf_counter() - wait_started_at
        batch = [first]
        deadline = perf_counter() + self.flush_timeout_sec
        fill_started_at = perf_counter()
        while len(batch) < self.batch_size_limit:
            timeout = max(0.0, deadline - perf_counter())
            try:
                batch.append(self._request_queue.get(timeout=timeout))
            except queue.Empty:
                break
        self.batch_fill_wait_sec += perf_counter() - fill_started_at
        return batch


class QueuedPolicyValueEvaluator:
    """MCTS-side evaluator client that does not own the model backend."""

    def __init__(
        self,
        request_queue: queue.Queue[tuple[EvaluationRequest, queue.Queue[_EvaluationResponse]]],
        failure: Callable[[], BaseException | None],
    ) -> None:
        self._request_queue = request_queue
        self._failure = failure
        self._request_ids = itertools.count()

    def evaluate_batch(
        self,
        requests: Sequence[tuple[ShogiBoard, tuple[str, ...]]],
    ) -> list[PositionEvaluation]:
        if not requests:
            return []
        failure = self._failure()
        if failure is not None:
            raise failure
        response_queue: queue.Queue[_EvaluationResponse] = queue.Queue()
        request_ids: list[int] = []
        for board, legal_moves in requests:
            request_id = next(self._request_ids)
            request_ids.append(request_id)
            self._request_queue.put(
                (EvaluationRequest(request_id, board.sfen(), tuple(legal_moves), perf_counter()), response_queue)
            )
        remaining = set(request_ids)
        results: dict[int, PositionEvaluation] = {}
        while remaining:
            response = response_queue.get()
            if response.request_id not in remaining:
                continue
            if response.error is not None:
                raise response.error
            if response.result is None:
                raise RuntimeError("central evaluator returned an empty response")
            results[response.request_id] = response.result
            remaining.remove(response.request_id)
        return [(results[request_id][0], float(results[request_id][1])) for request_id in request_ids]


@dataclass(frozen=True)
class ProcessEvaluationRequest:
    worker_id: int
    request_id: int
    position_sfen: str
    legal_moves: tuple[str, ...]
    enqueued_at: float


@dataclass(frozen=True)
class ProcessEvaluationResponse:
    request_id: int
    result: PositionEvaluation | None = None
    error_message: str | None = None


class ProcessCentralPolicyValueEvaluator:
    """Batch policy/value requests from worker processes in one parent-owned evaluator."""

    def __init__(
        self,
        evaluate_positions: EvaluatePositions,
        *,
        request_queue: Any,
        response_queues: Mapping[int, Any],
        batch_size_limit: int,
        flush_timeout_sec: float = 0.002,
    ) -> None:
        if batch_size_limit <= 0:
            raise ValueError("batch_size_limit must be positive")
        if flush_timeout_sec < 0.0:
            raise ValueError("flush_timeout_sec must be non-negative")
        if not response_queues:
            raise ValueError("response_queues must not be empty")
        self.evaluate_positions = evaluate_positions
        self.request_queue = request_queue
        self.response_queues = dict(response_queues)
        self.batch_size_limit = batch_size_limit
        self.flush_timeout_sec = flush_timeout_sec
        self.actual_batch_sizes: list[int] = []
        self.request_queue_wait_seconds: list[float] = []
        self.model_call_count = 0
        self.model_wall_time_sec = 0.0
        self.backend_performance_seconds: dict[str, float] = {}
        self.batch_first_wait_sec = 0.0
        self.batch_fill_wait_sec = 0.0
        self.response_send_wall_time_sec = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="process-central-policy-value-evaluator", daemon=True)
        self._started = False
        self._failure_message: str | None = None

    def __enter__(self) -> "ProcessCentralPolicyValueEvaluator":
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

    def performance_summary(self) -> dict[str, object]:
        return {
            "request_count": sum(self.actual_batch_sizes),
            "model_call_count": self.model_call_count,
            "model_wall_time_sec": self.model_wall_time_sec,
            **{
                f"backend_{key}": value
                for key, value in sorted(self.backend_performance_seconds.items())
            },
            "batch_first_wait_sec": self.batch_first_wait_sec,
            "batch_fill_wait_sec": self.batch_fill_wait_sec,
            "response_send_wall_time_sec": self.response_send_wall_time_sec,
            "request_queue_wait_sec_avg": (
                sum(self.request_queue_wait_seconds) / len(self.request_queue_wait_seconds)
                if self.request_queue_wait_seconds
                else 0.0
            ),
            "request_queue_wait_sec_max": max(self.request_queue_wait_seconds)
            if self.request_queue_wait_seconds
            else 0.0,
            **leaf_eval_batch_metrics(
                self.actual_batch_sizes,
                batch_size_limit=self.batch_size_limit,
            ),
        }

    def _run(self) -> None:
        while not self._stop.is_set() or not self.request_queue.empty():
            batch = self._read_batch()
            if not batch:
                continue
            if self._failure_message is not None:
                self._send_error_response(batch, self._failure_message)
                continue
            try:
                ready_at = perf_counter()
                self.request_queue_wait_seconds.extend(ready_at - request.enqueued_at for request in batch)
                started_at = perf_counter()
                evaluations = _evaluate_positions_backend(
                    self.evaluate_positions,
                    tuple((request.position_sfen, request.legal_moves) for request in batch),
                )
                elapsed = perf_counter() - started_at
                self.model_call_count += 1
                self.model_wall_time_sec += elapsed
                _add_backend_performance(self.backend_performance_seconds, self.evaluate_positions)
                self.actual_batch_sizes.append(len(batch))
                if len(evaluations) != len(batch):
                    raise ValueError("central evaluator backend must return one evaluation per request")
            except BaseException as error:
                self._failure_message = str(error)
                self._send_error_response(batch, self._failure_message)
                continue
            response_started_at = perf_counter()
            for request, evaluation in zip(batch, evaluations, strict=True):
                self.response_queues[request.worker_id].put(
                    ProcessEvaluationResponse(request.request_id, result=evaluation)
                )
            self.response_send_wall_time_sec += perf_counter() - response_started_at

    def _send_error_response(self, requests: Sequence[ProcessEvaluationRequest], error_message: str) -> None:
        for request in requests:
            self.response_queues[request.worker_id].put(
                ProcessEvaluationResponse(request.request_id, error_message=error_message)
            )

    def _read_batch(self) -> list[ProcessEvaluationRequest]:
        wait_started_at = perf_counter()
        try:
            first = self.request_queue.get(timeout=self.flush_timeout_sec if not self._stop.is_set() else 0.0)
        except queue.Empty:
            self.batch_first_wait_sec += perf_counter() - wait_started_at
            return []
        self.batch_first_wait_sec += perf_counter() - wait_started_at
        batch = [first]
        deadline = perf_counter() + self.flush_timeout_sec
        fill_started_at = perf_counter()
        while len(batch) < self.batch_size_limit:
            timeout = max(0.0, deadline - perf_counter())
            try:
                batch.append(self.request_queue.get(timeout=timeout))
            except queue.Empty:
                break
        self.batch_fill_wait_sec += perf_counter() - fill_started_at
        return batch


class ProcessQueuedPolicyValueEvaluator:
    """Worker-process evaluator client that sends requests to a parent evaluator."""

    def __init__(
        self,
        *,
        worker_id: int,
        request_queue: Any,
        response_queue: Any,
    ) -> None:
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._request_ids = itertools.count()

    def evaluate_batch(
        self,
        requests: Sequence[tuple[ShogiBoard, tuple[str, ...]]],
    ) -> list[PositionEvaluation]:
        if not requests:
            return []
        request_ids: list[int] = []
        for board, legal_moves in requests:
            request_id = next(self._request_ids)
            request_ids.append(request_id)
            self.request_queue.put(
                ProcessEvaluationRequest(
                    worker_id=self.worker_id,
                    request_id=request_id,
                    position_sfen=board.sfen(),
                    legal_moves=tuple(legal_moves),
                    enqueued_at=perf_counter(),
                )
            )
        remaining = set(request_ids)
        results: dict[int, PositionEvaluation] = {}
        while remaining:
            response: ProcessEvaluationResponse = self.response_queue.get()
            if response.request_id not in remaining:
                continue
            if response.error_message is not None:
                raise RuntimeError(response.error_message)
            if response.result is None:
                raise RuntimeError("central evaluator returned an empty response")
            results[response.request_id] = response.result
            remaining.remove(response.request_id)
        return [(results[request_id][0], float(results[request_id][1])) for request_id in request_ids]


def _evaluate_positions_backend(
    backend: object,
    requests: Sequence[PositionEvaluationRequest],
) -> list[PositionEvaluation]:
    if hasattr(backend, "evaluate_positions"):
        return backend.evaluate_positions(requests)  # type: ignore[union-attr]
    if hasattr(backend, "evaluate_batch"):
        return backend.evaluate_batch(requests)  # type: ignore[union-attr]
    if callable(backend):
        return backend(requests)
    raise TypeError("central evaluator backend must be callable or expose evaluate_batch/evaluate_positions")


def _add_backend_performance(target: dict[str, float], evaluate_positions: object) -> None:
    performance = getattr(evaluate_positions, "last_performance", None)
    if not isinstance(performance, Mapping):
        return
    for key, value in performance.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if key == "request_count":
            continue
        target[key] = target.get(key, 0.0) + float(value)
