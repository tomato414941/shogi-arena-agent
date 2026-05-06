from __future__ import annotations

import subprocess
import sys
import queue
import threading
from dataclasses import dataclass
from math import exp
from collections.abc import Sequence
from types import TracebackType


@dataclass(frozen=True)
class UsiGoResult:
    bestmove: str
    ponder: str | None = None
    info_lines: tuple[str, ...] = ()
    policy_targets: dict[str, float] | None = None


class UsiProcess:
    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        go_command: str = "go btime 0 wtime 0",
        read_timeout_seconds: float = 5.0,
        policy_target_multipv: int | None = None,
        policy_target_temperature_cp: float = 100.0,
        policy_target_mate_cp: float = 100000.0,
    ) -> None:
        self.command = list(command or [sys.executable, "-m", "shogi_arena_agent"])
        if not go_command.startswith("go"):
            raise ValueError("go_command must start with 'go'")
        if policy_target_multipv is not None and policy_target_multipv <= 0:
            raise ValueError("policy_target_multipv must be positive")
        if policy_target_temperature_cp <= 0.0:
            raise ValueError("policy_target_temperature_cp must be positive")
        self.go_command = go_command
        self.read_timeout_seconds = read_timeout_seconds
        self.policy_target_multipv = policy_target_multipv
        self.policy_target_temperature_cp = policy_target_temperature_cp
        self.policy_target_mate_cp = policy_target_mate_cp
        self.process: subprocess.Popen[str] | None = None
        self._stdout_lines: queue.Queue[str] = queue.Queue()
        self._stdout_thread: threading.Thread | None = None

    def __enter__(self) -> UsiProcess:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def start(self) -> None:
        if self.process is not None:
            return
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stdout_thread.start()
        self._send("usi")
        self._read_until("usiok")
        if self.policy_target_multipv is not None:
            self._send(f"setoption name MultiPV value {self.policy_target_multipv}")
        self._send("isready")
        self._read_until("readyok")

    def position(self, command: str) -> None:
        self._send(command)

    def go(self) -> UsiGoResult:
        self._send(self.go_command)
        return self._read_bestmove()

    def close(self) -> None:
        if self.process is None:
            return
        process = self.process
        if self.process.poll() is None:
            self._send("quit")
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        self.process = None
        self._stdout_thread = None

    def _send(self, line: str) -> None:
        process = self._running_process()
        if process.stdin is None:
            raise RuntimeError("USI process stdin is not available")
        process.stdin.write(line + "\n")
        process.stdin.flush()

    def _read_until(self, expected: str) -> list[str]:
        lines: list[str] = []
        while True:
            line = self._read_line()
            lines.append(line)
            if line == expected:
                return lines

    def _read_bestmove(self) -> UsiGoResult:
        info_lines: list[str] = []
        while True:
            line = self._read_line()
            if line.startswith("info "):
                info_lines.append(line)
                continue
            if line.startswith("bestmove "):
                words = line.removeprefix("bestmove ").split()
                ponder = words[2] if len(words) >= 3 and words[1] == "ponder" else None
                policy_targets = self._policy_targets_from_info_lines(info_lines)
                return UsiGoResult(
                    bestmove=words[0],
                    ponder=ponder,
                    info_lines=tuple(info_lines),
                    policy_targets=policy_targets,
                )

    def _read_line(self) -> str:
        process = self._running_process()
        try:
            line = self._stdout_lines.get(timeout=self.read_timeout_seconds)
        except queue.Empty:
            raise TimeoutError(f"USI process did not respond within {self.read_timeout_seconds} seconds")
        if line == "":
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise RuntimeError(f"USI process exited unexpectedly: {stderr.strip()}")
        return line.strip()

    def _read_stdout(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            self._stdout_lines.put(line)
        self._stdout_lines.put("")

    def _running_process(self) -> subprocess.Popen[str]:
        if self.process is None:
            raise RuntimeError("USI process has not started")
        if self.process.poll() is not None:
            raise RuntimeError(f"USI process exited with code {self.process.returncode}")
        return self.process

    def _policy_targets_from_info_lines(self, info_lines: Sequence[str]) -> dict[str, float] | None:
        if self.policy_target_multipv is None:
            return None
        scored_moves: dict[str, float] = {}
        for line in info_lines:
            parsed = _parse_multipv_info_line(line, mate_cp=self.policy_target_mate_cp)
            if parsed is None:
                continue
            multipv, score_cp, move = parsed
            if multipv <= self.policy_target_multipv:
                scored_moves[move] = score_cp
        if not scored_moves:
            return None
        max_score = max(scored_moves.values())
        weights = {
            move: exp((score_cp - max_score) / self.policy_target_temperature_cp)
            for move, score_cp in scored_moves.items()
        }
        total = sum(weights.values())
        return {move: weight / total for move, weight in weights.items()}


def _parse_multipv_info_line(line: str, *, mate_cp: float) -> tuple[int, float, str] | None:
    words = line.split()
    if not words or words[0] != "info":
        return None
    try:
        score_index = words.index("score")
        pv_index = words.index("pv")
    except ValueError:
        return None
    if pv_index + 1 >= len(words) or score_index + 2 >= len(words):
        return None
    multipv = 1
    if "multipv" in words:
        multipv_index = words.index("multipv")
        if multipv_index + 1 >= len(words):
            return None
        multipv = int(words[multipv_index + 1])
    score_kind = words[score_index + 1]
    score_value = float(words[score_index + 2])
    if score_kind == "cp":
        score_cp = score_value
    elif score_kind == "mate":
        score_cp = mate_cp if score_value > 0 else -mate_cp
    else:
        return None
    return multipv, score_cp, words[pv_index + 1]
