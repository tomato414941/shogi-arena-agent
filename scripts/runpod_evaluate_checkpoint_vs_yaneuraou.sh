#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ARENA_ROOT=$(pwd)
INTREP_ROOT=${INTREP_ROOT:-"$ARENA_ROOT/../intelligence-representation"}
RUNPOD_RUNNER_ROOT=${RUNPOD_RUNNER_ROOT:-"$ARENA_ROOT/../runpod-job-runner"}
RUNPOD_JOB=${RUNPOD_JOB:-"$RUNPOD_RUNNER_ROOT/scripts/run_job.py"}

CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to an intelligence-representation checkpoint path}
OUTPUT_DIR=${OUTPUT_DIR:-runs/shogi/runpod-checkpoint-vs-yaneuraou}

GPU_TYPE=${GPU_TYPE:-NVIDIA RTX 4000 Ada Generation}
DATA_CENTER_IDS=${DATA_CENTER_IDS:-}
MCTS_SIMULATIONS=${MCTS_SIMULATIONS:-4096}
MCTS_BATCH_SIZE=${MCTS_BATCH_SIZE:-64}
MCTS_MOVE_TIME_LIMIT_SEC=${MCTS_MOVE_TIME_LIMIT_SEC:-9.0}
BOARD_BACKEND=${BOARD_BACKEND:-cshogi}
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
REMOTE_GPU_SAMPLES="$REMOTE_OUTPUT/gpu_samples.csv"
REMOTE_GPU_SUMMARY="$REMOTE_OUTPUT/gpu_summary.json"

python3 "$RUNPOD_JOB" \
  --repo-root "$INTREP_ROOT" \
  --allow-existing-pods \
  --name shogi-arena-eval \
  --template-id runpod-torch-v280 \
  --gpu-type "$GPU_TYPE" \
  ${DATA_CENTER_IDS:+--data-center-ids "$DATA_CENTER_IDS"} \
  --container-disk-size 30 \
  --volume-size 0 \
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
/root/intrep/.venv/bin/python -m pip install -e .
cd /root
GIT_TERMINAL_PROMPT=0 git clone --depth 1 https://github.com/yaneurao/YaneuraOu.git YaneuraOu
cd /root/YaneuraOu/source
make -s -f Makefile -j\"\$(nproc)\" normal TARGET_CPU=AVX2 YANEURAOU_EDITION=YANEURAOU_ENGINE_MATERIAL COMPILER=g++ TARGET=YaneuraOu-runpod
mkdir -p '$REMOTE_OUTPUT'
cd /root/shogi-arena-agent
/root/intrep/.venv/bin/python - <<'PY' > '$REMOTE_GPU_SAMPLES' &
from __future__ import annotations

import csv
import subprocess
import sys
import time


def cpu_totals() -> tuple[int, int]:
    parts = open('/proc/stat', encoding='utf-8').readline().split()[1:]
    values = [int(part) for part in parts]
    idle = values[3] + values[4]
    return sum(values), idle


def gpu_sample() -> tuple[float, float, float, float]:
    output = subprocess.check_output(
        [
            'nvidia-smi',
            '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw',
            '--format=csv,noheader,nounits',
        ],
        text=True,
    ).strip()
    gpu_util, memory_used, memory_total, power_draw = [float(part.strip()) for part in output.split(',')]
    return gpu_util, memory_used, memory_total, power_draw


writer = csv.writer(sys.stdout)
writer.writerow(
    [
        'timestamp_ms',
        'cpu_util_percent',
        'gpu_util_percent',
        'memory_used_mib',
        'memory_total_mib',
        'power_draw_w',
    ]
)
sys.stdout.flush()
previous_total, previous_idle = cpu_totals()
while True:
    time.sleep(1)
    total, idle = cpu_totals()
    total_delta = total - previous_total
    idle_delta = idle - previous_idle
    previous_total, previous_idle = total, idle
    cpu_util = 0.0 if total_delta <= 0 else 100.0 * (1.0 - idle_delta / total_delta)
    try:
        gpu_util, memory_used, memory_total, power_draw = gpu_sample()
    except Exception:
        gpu_util, memory_used, memory_total, power_draw = 0.0, 0.0, 0.0, 0.0
    writer.writerow(
        [
            int(time.time() * 1000),
            round(cpu_util, 3),
            gpu_util,
            memory_used,
            memory_total,
            power_draw,
        ]
    )
    sys.stdout.flush()
PY
GPU_SAMPLER_PID=\$!
cleanup_gpu_sampler() {
  kill \"\$GPU_SAMPLER_PID\" >/dev/null 2>&1 || true
  wait \"\$GPU_SAMPLER_PID\" >/dev/null 2>&1 || true
}
trap cleanup_gpu_sampler EXIT
/root/intrep/.venv/bin/python scripts/evaluate_shogi_players.py \\
  --player-a-kind checkpoint \\
  --player-a-checkpoint '$REMOTE_CHECKPOINT' \\
  --player-a-move-selector mcts \\
  --player-a-mcts-simulations '$MCTS_SIMULATIONS' \\
  --player-a-mcts-evaluation-batch-size '$MCTS_BATCH_SIZE' \\
  --player-a-mcts-move-time-limit-sec '$MCTS_MOVE_TIME_LIMIT_SEC' \\
  --player-a-device cuda \\
  --player-a-board-backend '$BOARD_BACKEND' \\
  --player-b-kind usi_engine \\
  --player-b-usi-command /root/YaneuraOu/source/YaneuraOu-runpod \\
  --player-b-usi-go-command '$YANEURAOU_GO_COMMAND' \\
  --player-b-usi-read-timeout-seconds '$YANEURAOU_READ_TIMEOUT_SECONDS' \\
  --games '$GAMES' \\
  --max-plies '$MAX_PLIES' \\
  --out '$REMOTE_GAMES' \\
  | tee '$REMOTE_SUMMARY'
cleanup_gpu_sampler
trap - EXIT
/root/intrep/.venv/bin/python - <<'PY' > '$REMOTE_GPU_SUMMARY'
from __future__ import annotations

import csv
import json
from pathlib import Path

path = Path('$REMOTE_GPU_SAMPLES')
samples = []
with path.open(encoding='utf-8') as file:
    for row in csv.DictReader(file):
        try:
            samples.append(
                {
                    'cpu_util_percent': float(row['cpu_util_percent'].strip()),
                    'gpu_util_percent': float(row['gpu_util_percent'].strip()),
                    'memory_used_mib': float(row['memory_used_mib'].strip()),
                    'memory_total_mib': float(row['memory_total_mib'].strip()),
                    'power_draw_w': float(row['power_draw_w'].strip()),
                }
            )
        except (KeyError, ValueError):
            continue

summary = {'sample_count': len(samples)}
if samples:
    for key in ('cpu_util_percent', 'gpu_util_percent', 'memory_used_mib', 'power_draw_w'):
        values = [sample[key] for sample in samples]
        summary[f'{key}_avg'] = sum(values) / len(values)
        summary[f'{key}_max'] = max(values)
    summary['memory_total_mib'] = max(sample['memory_total_mib'] for sample in samples)

print(json.dumps(summary, indent=2, sort_keys=True))
PY
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
