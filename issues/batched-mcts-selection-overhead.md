# Batched MCTS Selection Overhead

Status: open. Priority: medium.

## Issue

After switching batched self-play to the `cshogi` board backend, MCTS selection
is one of the largest remaining CPU-side costs.

Selection currently scans Python dictionary entries and computes a PUCT score
for candidate children on each descent:

```text
candidates = [item for item in node.children.items() if not item[1].pending]
return max(candidates, key=score)
```

This is simple and correct, but it does repeated Python-level work inside the
inner MCTS loop.

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

This makes selection the second largest remaining non-model phase after
expansion.

## Current Position

Do not replace the selection implementation until there is enough training
signal to justify deeper MCTS runtime work.

The current implementation is readable and adequate for correctness. The next
step should be profiling at the target training scale before changing the data
structure.

## Candidate Directions

- avoid allocating a temporary `candidates` list on every selection,
- keep child statistics in arrays to make score computation cheaper,
- cache parent visit terms within a selection step,
- evaluate whether pending-child filtering can be represented without scanning
  every child.

## Acceptance Criteria

This issue can close when either:

- selection is no longer a top non-model phase in a RunPod cshogi self-play
  profile, or
- the project explicitly accepts the current selection overhead for this phase.
