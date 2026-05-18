# Generated Game Artifact Writer Boundary

Status: open
Priority: medium

## Problem

`scripts/generate_shogi_games.py` now writes durable generation artifacts:

- game records JSONL
- progress JSON
- events JSONL

The behavior is useful, but the writer currently lives as a private helper
inside the CLI script. That makes the CLI own too much artifact behavior and
makes it harder for other runtime entrypoints to reuse the same output contract.

## Desired Shape

Move the generated-game artifact writer into `src/shogi_arena_agent`.

The CLI should only:

- parse arguments
- build player/generation config
- call the generation runtime
- print the final summary

The reusable runtime writer should own:

- append-only game-record JSONL writes
- append-only event JSONL writes
- latest-progress JSON writes
- file naming around one output path

## Close Condition

- Artifact-writing code no longer lives in `scripts/generate_shogi_games.py`.
- The CLI uses a reusable writer from `src/shogi_arena_agent`.
- Existing generated-game CLI behavior remains covered by tests.
