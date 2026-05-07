# External Player Lifetime

Status: open.

## Issue

The current game-generation and evaluation scripts open external player
contexts inside the per-game loop. For YaneuraOu or other USI engines, this
means the engine process may be started and stopped once per game.

This is inefficient for hundreds or thousands of games and adds avoidable
operational noise to experiments. Process startup cost, startup failures, and
engine initialization variance can become part of a run even though they are
not part of the player policy being evaluated.

## Current State

Static in-process players such as checkpoint and deterministic players are
built once before the game loop. External players are opened through
`player_context()` inside the game loop.

## Candidate Direction

Use run-scoped player contexts when a player can safely be reused across games.
Each game should still send a fresh `position startpos` command through
`play_shogi_game`, so the player state is reset by protocol command rather than
process restart.

Keep a simple fallback path only if a specific engine proves unsafe to reuse.
Do not add broad lifecycle modes until there is a concrete need.

## Acceptance Criteria

- Generating or evaluating multiple games with the same external player does
  not restart that player once per game by default.
- Tests verify external player contexts are entered once for a multi-game run.
- Game records still preserve the same actor settings.
