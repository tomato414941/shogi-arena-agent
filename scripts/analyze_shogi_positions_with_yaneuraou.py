from __future__ import annotations

import argparse
import json
from pathlib import Path

from intrep.worlds.shogi.engine_analysis import (
    analyze_shogi_position_with_usi_session,
    shogi_analysis_positions_from_game_records,
    write_shogi_engine_analysis_jsonl,
)
from intrep.worlds.shogi.game_record import ShogiActorSpec, load_shogi_game_records_jsonl
from shogi_arena_agent.multipv_policy import MultiPVPolicyTargetConfig, configure_multipv_player_options
from shogi_arena_agent.usi_process import UsiProcess


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze shogi positions with YaneuraOu and write engine analysis JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--yaneuraou", required=True)
    parser.add_argument("--engine-go-command", default="go nodes 30")
    parser.add_argument("--multipv", type=int, default=3)
    parser.add_argument("--read-timeout-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)

    result = analyze_shogi_positions_with_yaneuraou(
        input_path=args.input,
        output_path=args.out,
        yaneuraou=args.yaneuraou,
        go_command=args.engine_go_command,
        multipv=args.multipv,
        read_timeout_seconds=args.read_timeout_seconds,
    )
    print(json.dumps(result, indent=2))


def analyze_shogi_positions_with_yaneuraou(
    *,
    input_path: Path,
    output_path: Path,
    yaneuraou: str,
    go_command: str,
    multipv: int,
    read_timeout_seconds: float = 10.0,
) -> dict[str, object]:
    records = load_shogi_game_records_jsonl(input_path)
    positions = shogi_analysis_positions_from_game_records(records)
    engine = ShogiActorSpec(
        kind="usi_engine",
        name="yaneuraou",
        settings={
            "command": yaneuraou,
            "go_command": go_command,
            "multipv": multipv,
        },
    )
    analyses = []

    with UsiProcess(
        command=[yaneuraou],
        go_command=go_command,
        read_timeout_seconds=read_timeout_seconds,
    ) as session:
        configure_multipv_player_options(session, MultiPVPolicyTargetConfig(multipv=multipv))
        for position in positions:
            analyses.append(analyze_shogi_position_with_usi_session(position, engine=engine, session=session))

    write_shogi_engine_analysis_jsonl(output_path, analyses)
    return {
        "input": str(input_path),
        "out": str(output_path),
        "game_count": len(records),
        "position_count": len(positions),
        "analysis_count": len(analyses),
        "engine": "yaneuraou",
        "engine_go_command": go_command,
        "multipv": multipv,
    }


if __name__ == "__main__":
    main()
