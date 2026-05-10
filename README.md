# shogi-arena-agent

Runtime project for turning shogi models and USI engines into arena-ready
shogi actors.

This repository is intentionally separate from `intelligence-representation`.
The model research repository should stay focused on training, checkpoints, and
portable inference boundaries. This repository owns shogi-specific runtime
concerns such as USI, local engine matches, Lishogi Bot integration, and future
arena deployment. It may use exported checkpoints from
`intelligence-representation`, but model training and benchmark records stay in
the model repository.

## Scope

In scope:

- USI engine wrapper
- local shogi arena runs
- Lishogi Bot bridge
- optional Floodgate or tournament integration
- loading exported model checkpoints from `intelligence-representation`
- runtime search strategies for checkpoint-backed actors
- actor specs and game records for local evaluation

Out of scope:

- model architecture research
- training recipes
- model-specific benchmark records
- RunPod training operations
- general multimodal representation design
- changes that make `intelligence-representation` depend on this repository

## Dependency Direction

```text
shogi-arena-agent
  may depend on intelligence-representation

intelligence-representation
  must not depend on shogi-arena-agent
```

## Initial Milestones

1. Define the minimal inference boundary for a shogi position.
2. Load a local placeholder policy and return a legal move.
3. Wrap that policy behind a USI-compatible process.
4. Run local games against a deterministic legal-move engine.
5. Evaluate Lishogi Bot integration.

## Local USI Smoke

```sh
printf 'usi\nisready\nposition startpos\ngo btime 0 wtime 0\nquit\n' \
  | uv run python -m shogi_arena_agent
```

Example response:

```text
id name shogi-arena-agent
id author intrep
usiok
readyok
bestmove <legal-usi-move>
```

The default deterministic legal-move policy is intentionally simple: it reconstructs the current
board with `python-shogi` and returns one deterministic legal move.

## Checkpoint Policy

`ShogiMoveChoiceCheckpointPolicy` loads a shogi move-choice checkpoint exported
by `intelligence-representation` and ranks the current legal moves. The import is
lazy, so the default USI deterministic legal-move policy does not require the research repository or
PyTorch.

Install the optional model dependencies only when running checkpoint-backed
policies:

```sh
uv run --extra model python -c "import torch, intrep"
```

Run the checkpoint as a USI engine:

```sh
uv run --extra model python -m shogi_arena_agent \
  --checkpoint /path/to/checkpoint.pt \
  --device cuda \
  --checkpoint-policy mcts \
  --checkpoint-simulations 4096 \
  --checkpoint-evaluation-batch-size 64 \
  --checkpoint-move-time-limit-sec 9.0
```

```python
from shogi_arena_agent.model_policy import ShogiMoveChoiceCheckpointPolicy
from shogi_arena_agent.usi import UsiEngine

policy = ShogiMoveChoiceCheckpointPolicy.from_checkpoint("shogi.pt")
engine = UsiEngine(policy=policy)
```

## Shogi Game Smoke

```sh
uv run python - <<'PY'
from shogi_arena_agent.shogi_game import play_shogi_game

print(play_shogi_game(max_plies=8))
PY
```

This runs two `UsiEngine` instances against each other and verifies each
returned `bestmove` before applying it to the board.

To exercise the real process boundary:

```sh
uv run python - <<'PY'
from shogi_arena_agent.shogi_game import play_shogi_game
from shogi_arena_agent.usi_process import UsiProcess

with UsiProcess() as black, UsiProcess() as white:
    print(play_shogi_game(black=black, white=white, max_plies=8))
PY
```

`UsiProcess(command=[...])` can also wrap an external USI engine command. A
read timeout is applied so a non-responsive engine does not hang the runner.

## External Engine Evaluation

Any USI-compatible engine can be used as the opponent process. For example,
after preparing a YaneuraOu executable:

```sh
uv run python scripts/evaluate_shogi_players.py \
  --player-kind deterministic_legal \
  --opponent-kind yaneuraou \
  --opponent-yaneuraou-command /path/to/YaneuraOu \
  --opponent-yaneuraou-go-command "go nodes 1" \
  --games 2 \
  --max-plies 64 \
  --out runs/shogi/evaluation.jsonl
```

Checkpoint-backed games can be written as raw game log JSONL:

```sh
uv run --extra model python scripts/generate_shogi_games.py \
  --black-kind checkpoint \
  --black-checkpoint /path/to/checkpoint.pt \
  --black-checkpoint-policy mcts \
  --black-checkpoint-simulations 16 \
  --white-kind yaneuraou \
  --white-yaneuraou-command /path/to/YaneuraOu \
  --white-yaneuraou-go-command "go nodes 1" \
  --games 2 \
  --max-plies 80 \
  --out runs/shogi/games.jsonl
```

Checkpoint self-play uses the same record format:

```sh
uv run --extra model python scripts/generate_shogi_games.py \
  --black-kind checkpoint \
  --black-checkpoint /path/to/checkpoint.pt \
  --black-checkpoint-policy direct \
  --white-kind checkpoint \
  --white-checkpoint /path/to/checkpoint.pt \
  --white-checkpoint-policy direct \
  --games 2 \
  --max-plies 80 \
  --out runs/shogi/self-play.jsonl
```

YaneuraOu self-play also writes the same record format:

```sh
uv run --extra model python scripts/generate_shogi_games.py \
  --black-kind yaneuraou \
  --black-yaneuraou-command /path/to/YaneuraOu \
  --black-yaneuraou-go-command "go nodes 1" \
  --white-kind yaneuraou \
  --white-yaneuraou-command /path/to/YaneuraOu \
  --white-yaneuraou-go-command "go nodes 1" \
  --games 2 \
  --max-plies 80 \
  --out runs/shogi/yaneuraou-self.jsonl
```

Local smoke tests can use a material-evaluation YaneuraOu build.

RunPod checkpoint-vs-YaneuraOu evaluation has a repository entrypoint:

```sh
CHECKPOINT=../intelligence-representation/models/d256-h1024-heads8-l6-shogi/checkpoint.pt \
MCTS_SIMULATIONS=4096 \
MCTS_BATCH_SIZE=64 \
MCTS_MOVE_TIME_LIMIT_SEC=9.0 \
GAMES=1 \
MAX_PLIES=80 \
./scripts/runpod_evaluate_checkpoint_vs_yaneuraou.sh
```

This script uses the shared `../runpod-job-runner/scripts/run_job.py` helper
for pod lifecycle management, then runs the shogi arena evaluation in this
repository. Results are copied back under `runs/shogi/`.

## Game Log Smoke

```sh
uv run python - <<'PY'
from pathlib import Path

from shogi_arena_agent.shogi_game import play_shogi_game, save_shogi_game_records_jsonl

result = play_shogi_game(max_plies=8)
save_shogi_game_records_jsonl((result,), Path("games.jsonl"))
print(result)
PY
```
