# Match Evaluation Responsibility

Status: open.

## Issue

`match_evaluation.py` still contains older high-level evaluation helpers that
know about USI process construction, MultiPV wrapping, actor settings, and
alternating sides. The current scripts now compose players through
`player_cli.py` and mostly use `match_evaluation.py` for result summarization.

This creates two paths that can drift: one path for current CLI player
composition, and another older path inside library helpers.

## Current State

`scripts/evaluate_shogi_players.py` builds both players via `player_cli.py`,
plays games, and calls `summarize_match_results()`.

`match_evaluation.py` also exposes `evaluate_player_against_usi_engine()` and
`evaluate_player_against_deterministic_legal()`, which duplicate player
construction and opponent-specific evaluation policy.

## Candidate Direction

Make one place responsible for player construction. Prefer keeping
`match_evaluation.py` focused on summarizing already-played records unless a
library-level evaluation API is still actively needed.

If high-level library evaluation remains useful, route it through the same
player-building path as the CLI rather than maintaining a second composition
model.

## Acceptance Criteria

- There is a single active path for composing checkpoint, USI-engine, and
  deterministic players.
- Match summary logic remains reusable and tested.
- Obsolete evaluation helpers are removed or rewritten to delegate to the
  active player-composition path.
