from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize shogi match sweep outputs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    summary = summarize_sweep(args.run_dir)
    output = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(output, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")


def summarize_sweep(run_dir: Path) -> dict[str, Any]:
    cases = []
    for summary_path in sorted(run_dir.glob("mcts-*/summary.json")):
        case_dir = summary_path.parent
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        games_path = case_dir / "games.jsonl"
        cases.append(
            {
                "case": case_dir.name,
                "games_path": str(games_path),
                "summary_path": str(summary_path),
                "game_count": summary.get("game_count"),
                "player_wins": _first_present(summary, "player_a_wins", "player_wins"),
                "player_losses": _first_present(summary, "player_a_losses", "player_losses"),
                "draws": summary.get("draws"),
                "average_plies": summary.get("average_plies"),
                "illegal_move_count": summary.get("illegal_move_count"),
                "inference_performance": _compact_performance(summary.get("inference_performance")),
            }
        )
    return {
        "schema_version": "shogi_arena_agent.match_sweep_summary.v1",
        "run_dir": str(run_dir),
        "cases": cases,
    }


def _first_present(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _compact_performance(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    keys = (
        "request_count",
        "request_wall_time_sec_avg",
        "request_wall_time_sec_p95",
        "request_wall_time_sec_max",
        "model_call_count_avg",
        "model_wall_time_sec_avg",
        "non_model_wall_time_sec_avg",
        "actual_nn_leaf_eval_batch_size_avg",
        "actual_nn_leaf_eval_batch_size_max",
        "actual_nn_leaf_eval_batch_size_fill_ratio_avg",
        "output_count_avg",
        "output_per_sec_avg",
    )
    return {key: value.get(key) for key in keys if key in value}


if __name__ == "__main__":
    main()
