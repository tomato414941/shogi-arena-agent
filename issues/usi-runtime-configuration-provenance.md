# USI Runtime Configuration Provenance

Status: open. Priority: low.

## Issue

The `python -m shogi_arena_agent` USI entrypoint can now run checkpoint-backed
actors with runtime search settings such as:

```text
--checkpoint-policy mcts
--checkpoint-simulations 4096
--checkpoint-evaluation-batch-size 64
--device cuda
```

For local game generation, `generate_shogi_games.py` records these settings in
`ShogiActorSpec.settings`. For external arena play through a plain USI process,
the engine only returns USI responses such as `bestmove ...`; the runtime
configuration is not automatically attached to any later game record.

This is acceptable for playing a game, but weak for later analysis or learning
from external arena logs.

## Current Position

Do not add a new logging or manifest path just for the bare USI entrypoint.

Treat this as an integration concern for the first real external arena path
that persists games, such as Lishogi Bot, Floodgate, or another tournament
runner.

## Candidate Direction

When an external arena integration writes durable game records or run outputs,
record a small actor/run manifest that includes:

- checkpoint path or durable checkpoint identity
- checkpoint policy or search strategy
- MCTS simulation count
- evaluation batch size
- device
- arena integration name
- repository revision, if available

Keep this separate from raw USI protocol responses. The protocol engine should
remain simple, while the arena runner owns provenance for persisted outputs.

## Acceptance Criteria

This issue can close when the first persisted external arena integration either:

- writes runtime configuration provenance alongside its game records, or
- explicitly decides that the external arena's own logs are not used for
  learning or model evaluation in this project phase.
