# Generated Game Event Schema

Status: open
Priority: low

## Problem

Generated-game events are currently JSON objects written to `events.jsonl`.
They are intentionally simple, but the schema is implicit:

- `generation_started`
- `progress`
- `game_finished`

As more event kinds are added, ad hoc dictionaries can make event consumers
fragile.

## Desired Shape

Keep the event stream simple, but make event construction explicit enough that
field names and required fields are not scattered through call sites.

This does not need a large framework. A small module-level set of helper
functions or dataclasses is enough if the schema starts to grow.

## Close Condition

- Event kinds and required fields are represented in one place.
- Tests cover the emitted fields for current event kinds.
- The event schema stays small and runtime-focused.
