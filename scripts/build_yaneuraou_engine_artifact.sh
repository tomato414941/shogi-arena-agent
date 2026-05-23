#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ARTIFACT_NAME=${ARTIFACT_NAME:-yaneuraou-engine}
OUTPUT=${OUTPUT:-"artifacts/$ARTIFACT_NAME.tar.zst"}
WORK_DIR=${WORK_DIR:-"runs/engine-artifact-build/$ARTIFACT_NAME"}
YANEURAOU_REPOSITORY_URL=${YANEURAOU_REPOSITORY_URL:-https://github.com/yaneurao/YaneuraOu.git}
YANEURAOU_REF=${YANEURAOU_REF:-master}
YANEURAOU_EDITION=${YANEURAOU_EDITION:-YANEURAOU_ENGINE_MATERIAL}
YANEURAOU_EVAL_ARCHIVE_URL=${YANEURAOU_EVAL_ARCHIVE_URL:-}
YANEURAOU_EVAL_ARCHIVE_SHA256=${YANEURAOU_EVAL_ARCHIVE_SHA256:-}
YANEURAOU_TARGET_CPU=${YANEURAOU_TARGET_CPU:-AVX2}
YANEURAOU_COMPILER=${YANEURAOU_COMPILER:-g++}
YANEURAOU_TARGET=${YANEURAOU_TARGET:-YaneuraOu-runpod}

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR/source" "$WORK_DIR/artifact/bin"

GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch "$YANEURAOU_REF" "$YANEURAOU_REPOSITORY_URL" "$WORK_DIR/source/YaneuraOu"
make -s -C "$WORK_DIR/source/YaneuraOu/source" -f Makefile -j"$(nproc)" normal \
  TARGET_CPU="$YANEURAOU_TARGET_CPU" \
  YANEURAOU_EDITION="$YANEURAOU_EDITION" \
  COMPILER="$YANEURAOU_COMPILER" \
  TARGET="$YANEURAOU_TARGET"

cp "$WORK_DIR/source/YaneuraOu/source/$YANEURAOU_TARGET" "$WORK_DIR/artifact/bin/YaneuraOu-runpod"

if [[ -n "$YANEURAOU_EVAL_ARCHIVE_URL" ]]; then
  mkdir -p "$WORK_DIR/artifact/eval"
  curl -L --fail --retry 3 "$YANEURAOU_EVAL_ARCHIVE_URL" -o "$WORK_DIR/eval-archive"
  if [[ -n "$YANEURAOU_EVAL_ARCHIVE_SHA256" ]]; then
    echo "$YANEURAOU_EVAL_ARCHIVE_SHA256  $WORK_DIR/eval-archive" | sha256sum -c -
  fi
  if 7z l "$WORK_DIR/eval-archive" >/dev/null 2>&1; then
    7z x -y "$WORK_DIR/eval-archive" -o"$WORK_DIR/artifact/eval" >/dev/null
  else
    unzip -q "$WORK_DIR/eval-archive" -d "$WORK_DIR/artifact/eval"
  fi
  if [[ ! -f "$WORK_DIR/artifact/eval/nn.bin" ]]; then
    nnue_file=$(find "$WORK_DIR/artifact/eval" -type f \( -name 'nn.bin' -o -name '*.nnue' -o -name '*.bin' \) | head -n 1)
    if [[ -z "$nnue_file" ]]; then
      echo "NNUE eval archive did not contain nn.bin, *.nnue, or *.bin" >&2
      exit 1
    fi
    cp "$nnue_file" "$WORK_DIR/artifact/eval/nn.bin"
  fi
fi

cat > "$WORK_DIR/artifact/manifest.json" <<JSON
{
  "artifact_name": "$ARTIFACT_NAME",
  "engine": "YaneuraOu",
  "repository_url": "$YANEURAOU_REPOSITORY_URL",
  "ref": "$YANEURAOU_REF",
  "edition": "$YANEURAOU_EDITION",
  "target_cpu": "$YANEURAOU_TARGET_CPU",
  "binary": "bin/YaneuraOu-runpod",
  "eval_dir": "eval"
}
JSON

mkdir -p "$(dirname "$OUTPUT")"
tar -I zstd -cf "$OUTPUT" -C "$WORK_DIR/artifact" .
sha256sum "$OUTPUT" | tee "$OUTPUT.sha256"
