# Generated Game Shard Progress Aggregation

Status: open
Priority: low

## Problem

Sharded generated-game runs now write per-shard durable artifacts:

- `games.shard-XXX.jsonl`
- `games.shard-XXX.events.jsonl`
- `games.shard-XXX.progress.json`

This is enough to diagnose whether workers are alive and whether games are
finishing, but there is no parent-level progress file that aggregates all
shards.

For long runs, a human or monitor still has to inspect several shard progress
files to answer simple questions:

- how many games are complete in total?
- which shard is slowest?
- are any shards stalled?

## Desired Shape

Expose a parent-level progress view for sharded generation without duplicating
game records or summaries.

The parent view should be derived from shard progress files and should not
become a second source of truth for generated records.

## Close Condition

- A sharded run exposes one parent progress artifact with total completed games
  and per-shard status.
- The parent progress artifact is clearly derived from shard progress files.
- Tests cover the parent progress artifact for a multi-worker run.
