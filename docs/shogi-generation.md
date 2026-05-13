# Shogi Generation

This document describes the shogi game generation runtime owned by
`shogi-arena-agent`.

The main entrypoint is:

```sh
uv run --extra model python scripts/generate_shogi_games.py
```

## Parallelism

Self-play generation has two separate parallelism settings:

| Setting | Meaning |
| --- | --- |
| `--concurrent-games-per-process` | Number of games advanced together inside one Python generator process. |
| `--generation-worker-processes` | Number of Python generator processes launched by the parent generator. |

The approximate active-game count is:

```text
concurrent_games_per_process * generation_worker_processes
```

Use `--concurrent-games-per-process` to batch neural-network leaf evaluation
within one process. Use `--generation-worker-processes` when one generator
process is near one CPU core and the machine has additional CPU cores available.

Each worker process writes a shard JSONL file. The parent process merges those
shards into the requested `--out` JSONL and prints an aggregate summary.

## Example

```sh
uv run --extra model python scripts/generate_shogi_games.py \
  --black-kind checkpoint \
  --black-checkpoint /path/to/checkpoint.pt \
  --black-move-selector mcts \
  --black-move-selection-profile self-play \
  --black-mcts-simulations 16 \
  --black-mcts-evaluation-batch-size 32 \
  --black-device cuda \
  --white-kind checkpoint \
  --white-checkpoint /path/to/checkpoint.pt \
  --white-move-selector mcts \
  --white-move-selection-profile self-play \
  --white-mcts-simulations 16 \
  --white-mcts-evaluation-batch-size 32 \
  --white-device cuda \
  --games 32 \
  --concurrent-games-per-process 8 \
  --generation-worker-processes 4 \
  --seed 7 \
  --board-backend cshogi \
  --max-plies 320 \
  --out runs/shogi/self-play.jsonl
```

With these settings, 32 total games are split across 4 worker processes, and
each process advances up to 8 active games at a time.
