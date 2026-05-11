#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ARENA_ROOT=$(pwd)
INTREP_ROOT=${INTREP_ROOT:-"$ARENA_ROOT/../intelligence-representation"}
RUNPOD_RUNNER_ROOT=${RUNPOD_RUNNER_ROOT:-"$ARENA_ROOT/../runpod-job-runner"}
RUNPOD_JOB=${RUNPOD_JOB:-"$RUNPOD_RUNNER_ROOT/scripts/run_job.py"}

CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to an intelligence-representation checkpoint path}
OUTPUT_DIR=${OUTPUT_DIR:-runs/shogi/runpod-checkpoint-vs-yaneuraou}

GPU_TYPE=${GPU_TYPE:-NVIDIA GeForce RTX 5090}
MCTS_SIMULATIONS=${MCTS_SIMULATIONS:-4096}
MCTS_BATCH_SIZE=${MCTS_BATCH_SIZE:-64}
MCTS_MOVE_TIME_LIMIT_SEC=${MCTS_MOVE_TIME_LIMIT_SEC:-9.0}
GAMES=${GAMES:-1}
MAX_PLIES=${MAX_PLIES:-320}
YANEURAOU_GO_COMMAND=${YANEURAOU_GO_COMMAND:-go nodes 1}
YANEURAOU_READ_TIMEOUT_SECONDS=${YANEURAOU_READ_TIMEOUT_SECONDS:-30}

ARENA_REPOSITORY_URL=${ARENA_REPOSITORY_URL:-$(git config --get remote.origin.url)}
ARENA_REF=${ARENA_REF:-main}

if [[ -z "$ARENA_REPOSITORY_URL" ]]; then
  echo "ARENA_REPOSITORY_URL is required when origin remote is unset" >&2
  exit 1
fi

if [[ ! -f "$RUNPOD_JOB" ]]; then
  echo "RunPod runner not found: $RUNPOD_JOB" >&2
  exit 1
fi

CHECKPOINT_ABS=$(realpath "$CHECKPOINT")
INTREP_ABS=$(realpath "$INTREP_ROOT")
case "$CHECKPOINT_ABS" in
  "$INTREP_ABS"/*) CHECKPOINT_REL=${CHECKPOINT_ABS#"$INTREP_ABS"/} ;;
  *)
    echo "CHECKPOINT must be under INTREP_ROOT so the intrep runner can sync it: $CHECKPOINT_ABS" >&2
    exit 1
    ;;
esac

REMOTE_CHECKPOINT="/root/intrep/$CHECKPOINT_REL"
REMOTE_OUTPUT="/root/intrep/$OUTPUT_DIR"
REMOTE_SUMMARY="$REMOTE_OUTPUT/summary.json"
REMOTE_GAMES="$REMOTE_OUTPUT/games.jsonl"
REMOTE_CUDA="$REMOTE_OUTPUT/cuda.txt"

python3 "$RUNPOD_JOB" \
  --repo-root "$INTREP_ROOT" \
  --allow-existing-pods \
  --name shogi-arena-eval \
  --gpu-type "$GPU_TYPE" \
  --container-disk-size 30 \
  --volume-size 20 \
  --remote-dir /root/intrep \
  --sync src \
  --sync tests \
  --sync pyproject.toml \
  --sync uv.lock \
  --sync README.md \
  --sync AGENTS.md \
  --sync scripts/setup_runpod.sh \
  --sync "$CHECKPOINT_REL" \
  --setup-command 'cd "$REMOTE_DIR"; bash scripts/setup_runpod.sh' \
  --remote "set -euo pipefail
apt-get update >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y git build-essential >/dev/null
cd /root
rm -rf shogi-arena-agent YaneuraOu
GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch '$ARENA_REF' '$ARENA_REPOSITORY_URL' shogi-arena-agent
cd /root/shogi-arena-agent
/root/intrep/.venv/bin/python -m pip install -e . --no-deps
cd /root
GIT_TERMINAL_PROMPT=0 git clone --depth 1 https://github.com/yaneurao/YaneuraOu.git YaneuraOu
cd /root/YaneuraOu/source
make -s -f Makefile -j\"\$(nproc)\" normal TARGET_CPU=AVX2 YANEURAOU_EDITION=YANEURAOU_ENGINE_MATERIAL COMPILER=g++ TARGET=YaneuraOu-runpod
mkdir -p '$REMOTE_OUTPUT'
cd /root/shogi-arena-agent
/root/intrep/.venv/bin/python scripts/evaluate_shogi_players.py \\
  --player-kind checkpoint \\
  --player-checkpoint '$REMOTE_CHECKPOINT' \\
  --player-checkpoint-policy mcts \\
  --player-checkpoint-simulations '$MCTS_SIMULATIONS' \\
  --player-checkpoint-evaluation-batch-size '$MCTS_BATCH_SIZE' \\
  --player-checkpoint-move-time-limit-sec '$MCTS_MOVE_TIME_LIMIT_SEC' \\
  --player-checkpoint-device cuda \\
  --opponent-kind yaneuraou \\
  --opponent-yaneuraou-command /root/YaneuraOu/source/YaneuraOu-runpod \\
  --opponent-yaneuraou-go-command '$YANEURAOU_GO_COMMAND' \\
  --opponent-yaneuraou-read-timeout-seconds '$YANEURAOU_READ_TIMEOUT_SECONDS' \\
  --games '$GAMES' \\
  --max-plies '$MAX_PLIES' \\
  --out '$REMOTE_GAMES' \\
  | tee '$REMOTE_SUMMARY'
/root/intrep/.venv/bin/python - <<'PY' > '$REMOTE_CUDA'
import torch
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0))
PY
" \
  --output "$OUTPUT_DIR" \
  --timings-output "$OUTPUT_DIR/runpod_timings.json" \
  "$@"
