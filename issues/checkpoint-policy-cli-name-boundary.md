# Checkpoint Policy CLI Name Boundary

Status: open. Priority: low.

## Issue

`--*-checkpoint-policy` currently selects the move-selection strategy used with
a checkpoint, such as `direct` or `mcts`.

The name can be confused with other policy concepts:

- policy heads
- policy targets
- learned policy outputs
- direct move policies
- MCTS search policies
- actor settings recorded as `policy`

This is manageable while the only strategies are `direct` and `mcts`, but it may
become unclear before adding more checkpoint-backed strategies such as batched
MCTS, time-control MCTS, or beam search.

## Desired Direction

Before adding more checkpoint-backed move-selection strategies, decide whether
to keep `--*-checkpoint-policy` or introduce a clearer alias such as:

- `--*-checkpoint-search`
- `--*-checkpoint-move-strategy`

Prefer preserving active command compatibility unless the rename happens before
the CLI is widely used.

## Non-Goals

- changing `ShogiGameRecord` schema immediately
- renaming policy heads or policy targets
- adding new search strategies
- changing existing run artifacts

## Acceptance Criteria

This issue can close when checkpoint-backed move-selection naming is either:

- documented as intentionally remaining `checkpoint-policy`, or
- renamed or aliased to a clearer term before additional strategies make the
  distinction harder to recover.
