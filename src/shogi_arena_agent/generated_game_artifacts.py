from __future__ import annotations

import json
import sys
from pathlib import Path

from shogi_arena_agent.shogi_game import ShogiGameRecord, shogi_game_record_to_json


class GeneratedGameArtifacts:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.events_out = out.with_name(f"{out.stem}.events.jsonl")
        self.progress_out = out.with_name(f"{out.stem}.progress.json")
        self._records_written = 0

    def __enter__(self) -> "GeneratedGameArtifacts":
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text("", encoding="utf-8")
        self.events_out.write_text("", encoding="utf-8")
        self.progress_out.write_text(
            json.dumps({"completed_games": 0, "events_path": self.events_out.name}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.write_event({"event": "generation_started", "games_path": self.out.name})
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def write_record(self, record: ShogiGameRecord) -> None:
        with self.out.open("a", encoding="utf-8") as file:
            file.write(json.dumps(shogi_game_record_to_json(record), sort_keys=True) + "\n")
        self._records_written += 1
        self.write_event(
            {
                "event": "game_finished",
                "completed_games": self._records_written,
                "plies": len(record.transitions),
                "end_reason": record.end_reason,
                "winner": record.winner,
            }
        )
        self._write_progress(
            {
                "completed_games": self._records_written,
                "last_game_plies": len(record.transitions),
                "last_game_end_reason": record.end_reason,
                "last_game_winner": record.winner,
            }
        )

    def write_progress(self, payload: dict[str, object]) -> None:
        self.write_event({"event": "progress", **payload})
        self._write_progress({"completed_games": self._records_written, **payload})
        print("progress " + json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)

    def write_event(self, payload: dict[str, object]) -> None:
        with self.events_out.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, sort_keys=True) + "\n")

    def _write_progress(self, payload: dict[str, object]) -> None:
        self.progress_out.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
