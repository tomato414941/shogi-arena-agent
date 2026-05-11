# Batched MCTS Expansion Overhead

Status: open. Priority: medium.

## Issue

After switching batched self-play from `python-shogi` to `cshogi`, legal move
generation is no longer the dominant CPU-side cost. The largest remaining
non-model phase in the measured RunPod Online Replay smoke is MCTS node
expansion.

Current expansion creates Python `_Node` objects and a Python dictionary entry
for each legal move:

```text
node.children = {move: _Node(prior=normalized_priors[move]) for move in legal_moves}
```

## Evidence

RunPod Online Replay smoke with `cshogi`:

```text
checkpoint: d256
games: 16
parallel-games: 16
max-plies: 20
simulations: 128
evaluation-batch-size: 64
max-steps: 5
device: cuda
```

Aggregate non-model phase share:

```text
expand: 51.84%
selection: 33.41%
legal_moves: 6.70%
board_copy: 3.00%
```

This makes expansion the largest remaining non-model phase after the `cshogi`
backend change.

## Current Position

Do not optimize this before confirming that Online Replay learning behavior is
stable at a small real training scale.

This is now a real optimization candidate, but it is not blocking correctness.

## Candidate Directions

- reduce per-node Python object allocation,
- store child statistics in compact arrays instead of one `_Node` object per
  legal move,
- avoid rebuilding normalized prior dictionaries where possible,
- profile expansion separately on representative midgame positions.

## Acceptance Criteria

This issue can close when either:

- expansion is no longer a top non-model phase in a RunPod cshogi self-play
  profile, or
- the project explicitly accepts the current expansion overhead for this phase.
