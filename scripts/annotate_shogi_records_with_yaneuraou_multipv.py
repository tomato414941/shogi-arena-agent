from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from shogi_arena_agent.multipv_policy import (
    MultiPVPolicyTargetConfig,
    MultiPVPolicyTargetPlayer,
    configure_multipv_player_options,
)
from shogi_arena_agent.shogi_game import (
    ShogiGameRecord,
    ShogiTransitionRecord,
    load_shogi_game_records_jsonl,
    save_shogi_game_records_jsonl,
)
from shogi_arena_agent.usi_process import UsiProcess


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Annotate shogi game records with YaneuraOu MultiPV policy targets.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--yaneuraou", required=True)
    parser.add_argument("--engine-go-command", default="go nodes 30")
    parser.add_argument("--multipv", type=int, default=3)
    parser.add_argument("--temperature-cp", type=float, default=100.0)
    parser.add_argument("--read-timeout-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)

    result = annotate_shogi_records_with_yaneuraou_multipv(
        input_path=args.input,
        output_path=args.out,
        yaneuraou=args.yaneuraou,
        go_command=args.engine_go_command,
        multipv=args.multipv,
        temperature_cp=args.temperature_cp,
        read_timeout_seconds=args.read_timeout_seconds,
    )
    print(json.dumps(result, indent=2))


def annotate_shogi_records_with_yaneuraou_multipv(
    *,
    input_path: Path,
    output_path: Path,
    yaneuraou: str,
    go_command: str,
    multipv: int,
    temperature_cp: float,
    read_timeout_seconds: float = 10.0,
) -> dict[str, object]:
    records = load_shogi_game_records_jsonl(input_path)
    config = MultiPVPolicyTargetConfig(multipv=multipv, temperature_cp=temperature_cp)
    annotated_records: list[ShogiGameRecord] = []
    transition_count = 0
    annotated_count = 0

    with UsiProcess(
        command=[yaneuraou],
        go_command=go_command,
        read_timeout_seconds=read_timeout_seconds,
    ) as engine:
        configure_multipv_player_options(engine, config)
        teacher = MultiPVPolicyTargetPlayer(engine, config)
        for record in records:
            annotated_transitions: list[ShogiTransitionRecord] = []
            for transition in record.transitions:
                transition_count += 1
                teacher.position(f"position sfen {transition.position_sfen}")
                go_result = teacher.go()
                if go_result.info_lines:
                    annotated_count += 1
                annotated_transitions.append(
                    replace(
                        transition,
                        usi_info_lines=go_result.info_lines,
                    )
                )
            annotated_records.append(replace(record, transitions=tuple(annotated_transitions)))

    save_shogi_game_records_jsonl(tuple(annotated_records), output_path)
    return {
        "input": str(input_path),
        "out": str(output_path),
        "game_count": len(annotated_records),
        "transition_count": transition_count,
        "annotated_count": annotated_count,
        "annotated_ratio": annotated_count / transition_count if transition_count else 0.0,
        "teacher": "yaneuraou",
        "engine_go_command": go_command,
        "multipv": multipv,
        "temperature_cp": temperature_cp,
    }

if __name__ == "__main__":
    main()
