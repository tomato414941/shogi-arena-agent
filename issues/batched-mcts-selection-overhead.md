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

2026-05-13 current-code RunPod self-play profile with `cshogi`:

```text
checkpoint: d256
GPU: RTX 4000 Ada Generation
Pod: 6 vCPU / 31 GiB community
max-plies: 320
MCTS simulations per move: 16
NN leaf eval batch limit: 32
device: cuda
```

Measured non-model phase share across worker-scaling cases:

```text
worker=1: expand 63.81%, selection 23.38%, legal_moves 8.38%, board_copy 3.17%
worker=2: expand 64.08%, selection 22.75%, legal_moves 8.45%, board_copy 3.53%
worker=4: expand 62.76%, selection 23.94%, legal_moves 8.31%, board_copy 3.64%
worker=6: expand 65.81%, selection 21.63%, legal_moves 7.73%, board_copy 3.52%
```

Selection remains the second largest measured non-model phase after expansion.

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
