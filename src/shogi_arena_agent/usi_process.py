from __future__ import annotations

import subprocess
import sys
import queue
import threading
from collections.abc import Sequence
from types import TracebackType


class UsiProcess:
    def __init__(self, command: Sequence[str] | None = None, *, read_timeout_seconds: float = 5.0) -> None:
        self.command = list(command or [sys.executable, "-m", "shogi_arena_agent"])
        self.read_timeout_seconds = read_timeout_seconds
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
        self._send("isready")
        self._read_until("readyok")

    def position(self, command: str) -> None:
        self._send(command)

    def go(self) -> str:
        self._send("go btime 0 wtime 0")
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

    def _read_bestmove(self) -> str:
        while True:
            line = self._read_line()
            if line.startswith("bestmove "):
                return line.removeprefix("bestmove ").split()[0]

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
