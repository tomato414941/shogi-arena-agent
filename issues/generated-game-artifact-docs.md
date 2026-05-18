# Generated Game Artifact Docs

Status: open
Priority: low

## Problem

Generated-game runs now produce multiple artifacts:

- game records JSONL
- events JSONL
- progress JSON
- final summary JSON printed by the CLI

Their roles are not yet documented in `shogi-arena-agent`.

Without a short contract, future code may blur:

- durable source records
- operational event logs
- latest progress snapshots
- final aggregate summaries

## Desired Shape

Add a compact runtime artifact contract to the repository docs.

It should state:

- `games.jsonl` is the durable generated game-record output.
- `events.jsonl` is append-only operational telemetry.
- `progress.json` is a latest-state convenience snapshot.
- the final summary is an aggregate report, not the source record.

## Close Condition

- The generated-game artifact roles are documented in one concise place.
- The doc avoids run-specific examples and does not duplicate experiment logs.
