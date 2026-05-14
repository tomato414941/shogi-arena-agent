# Batched MCTS Expansion Overhead

Status: open. Priority: medium.

## Issue

After switching batched self-play from `python-shogi` to `cshogi`, legal move
generation is no longer the dominant CPU-side cost. The largest remaining
measured non-model phase is MCTS node expansion.

Current expansion creates Python `MctsNode` objects and a Python dictionary entry
for each legal move:

```text
node.children = {move: MctsNode(prior=normalized_priors[move]) for move in legal_moves}
```

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

Expansion remains the largest measured non-model phase after the `cshogi`
backend change and after the `MctsBatchSearchExecutor` split.

## Current Position

Do not optimize this before confirming that Online Replay learning behavior is
stable at a small real training scale.

This is now a real optimization candidate, but it is not blocking correctness.

2026-05-14: a small hot-path cleanup removed the intermediate normalized-prior
dictionary during expansion. This issue remains open until a follow-up profile
shows whether that materially changes expansion share or throughput.

## Candidate Directions

- reduce per-node Python object allocation,
- store child statistics in compact arrays instead of one `MctsNode` object per
  legal move,
- avoid rebuilding normalized prior dictionaries where possible,
- profile expansion separately on representative midgame positions.

## Acceptance Criteria

This issue can close when either:

- expansion is no longer a top non-model phase in a RunPod cshogi self-play
  profile, or
- the project explicitly accepts the current expansion overhead for this phase.
