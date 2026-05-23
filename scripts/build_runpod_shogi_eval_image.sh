#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_IMAGE=${BASE_IMAGE:?set BASE_IMAGE to the RunPod GPU base image}
IMAGE=${IMAGE:?set IMAGE to the output image name}
PUSH=${PUSH:-0}

docker build \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -f containers/runpod-shogi-eval/Dockerfile \
  -t "$IMAGE" \
  .

if [[ "$PUSH" == "1" ]]; then
  docker push "$IMAGE"
fi
