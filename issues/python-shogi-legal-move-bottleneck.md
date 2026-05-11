# Python-Shogi Legal Move Bottleneck

Status: closed.

## Issue

RunPod Online Replay smoke runs showed that batched checkpoint self-play was
not limited by GPU forward time alone. In the `python-shogi` backend, the
largest CPU-side cost was legal move generation inside MCTS.

This affected the main shogi self-play path used by
`intelligence-representation` Online Replay.

## Evidence

Initial local profiling with `UniformPolicyValueEvaluator` showed
`_legal_move_usis` dominated CPU time. The expensive calls were inside
`python-shogi` legal move generation, especially the pawn-drop and self-check
legality path.

RunPod d256 aggregate timing before the backend change:

```text
checkpoint: d256
games: 16
parallel-games: 16
max-plies: 20
simulations: 128
evaluation-batch-size: 64
```

With `python-shogi`:

```text
wall time: 139.524s
legal_moves total: 64.652s
legal_moves share of non-model time: 72.02%
```

With `cshogi`:

```text
wall time: 54.476s
legal_moves total: 0.643s
legal_moves share of non-model time: 6.55%
```

The same RunPod comparison used one RTX 5090 Pod and ran the two backends
sequentially.

## Implemented

`cshogi` was added as a supported board backend.

The hot self-play path can now select the board backend:

```text
scripts/generate_shogi_games.py --board-backend {python-shogi,cshogi}
```

The default shogi generation backend in `intelligence-representation` is now
`cshogi` for generated-data training and Online Replay.

Compatibility checks compare `python-shogi` and `cshogi` legal move sets across
representative positions, including multi-ply positions, promotion/capture
paths, bishop drops, and SFEN positions with moves.

## Verification

Local tests:

```text
uv run python -m unittest discover -s tests
```

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

Result:

```text
status: passed
remote command wall time: 79.004s
generated records: 16
decisions: 320
replay examples: 240
training steps: 5
actor settings include board_backend: cshogi
```

Aggregate batch timing:

```text
request avg: 2.536890s
model avg: 2.060601s
non-model avg: 0.476289s
output/sec: 828.967 simulations/sec
```

## Current Position

The original legal-move-generation CPU bottleneck is resolved for the main
batched self-play path.

The next visible non-model costs after switching to `cshogi` are Python-side
MCTS tree work, mainly expansion and selection. That is a separate optimization
topic and should not keep this issue open.

## Acceptance Criteria

This issue is closed because:

- the dominant CPU bottleneck was identified,
- a faster board backend was added,
- Online Replay now uses `cshogi` by default from the model-training project,
- RunPod comparison showed a material wall-clock improvement, and
- RunPod Online Replay smoke passed with `board_backend: cshogi`.
