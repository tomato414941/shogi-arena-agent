# Checkpoint Self-Play Central GPU Evaluator

Status: open
Priority: high

## Problem

Checkpoint self-play generation currently reuses the general generated-game
path. That path is useful for ordinary game generation, USI-engine games, and
small checks, but it is not shaped around high-throughput checkpoint self-play.

With sharded generation, each worker process owns its own checkpoint evaluator
and sends small batches to the GPU independently. The multi-position MCTS path
also collects at most one leaf evaluation per active position per iteration.
As a result, `nn_leaf_eval_batch_limit=64` does not mean the GPU receives
batches near 64; the effective batch size is usually bounded by the active
positions in one worker-side chunk.

Recent measurements show this directly:

- `w8_c8_s128_b64_g128_a40`: actual NN leaf eval batch avg about 6.73, max 8
- `w8_c16_s128_b64_g128_a40`: actual NN leaf eval batch avg about 11.49, max 16

Increasing per-worker concurrency filled batches somewhat better, but did not
materially improve generated plies/sec. The implementation structure fragments
GPU work before it reaches the model.

## Desired Shape

Create a checkpoint self-play generation path whose only job is fast
checkpoint-vs-checkpoint data generation.

The desired runtime shape is:

```text
CPU search workers
  -> submit leaf evaluation requests
  -> receive priors/value
  -> continue MCTS backup and move selection

central GPU evaluator
  -> owns one checkpoint model instance
  -> batches requests from all workers
  -> flushes by batch size or short timeout
  -> returns results by request id
```

The central evaluator should know how to evaluate checkpoint policy/value
requests. It should not own MCTS tree state. Search workers should own board and
tree state. They should not own GPU model instances.

## Scope

In scope:

- checkpoint-vs-checkpoint self-play generation
- fixed-simulation MCTS
- GPU batched policy/value inference
- generated game records with MCTS visit-count policy targets
- generated game records with MCTS root mean value targets
- throughput telemetry for requested batch size and actual batch size

Out of scope:

- CSA / Floodgate / water-gate play
- USI-engine games
- arbitrary player abstraction
- time-limit search
- root reuse
- training loop changes
- tensor cache construction

## Design Notes

This should not be implemented by making the existing general player abstraction
more generic. The problem is the opposite: high-throughput self-play needs fewer
use cases in the hot path.

The existing `generate_shogi_games.py` path can remain the general and
correctness-oriented path. The new path should be allowed to reject unsupported
player kinds and search modes explicitly.

## Acceptance Criteria

- A checkpoint-vs-checkpoint self-play run can use one central GPU evaluator
  across multiple CPU search workers.
- The central evaluator owns the checkpoint model once per GPU process.
- Worker-side MCTS does not instantiate checkpoint models.
- Telemetry records requested batch limit, actual evaluator batch sizes, model
  call count, model wall time, and generated plies/sec.
- A RunPod measurement compares the new path against the current sharded
  `generate_shogi_games.py` path under a comparable MCTS128 workload.
