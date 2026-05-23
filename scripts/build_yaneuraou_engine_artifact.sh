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

SOURCE_COMMIT=$(git -C "$WORK_DIR/source/YaneuraOu" rev-parse HEAD)
BINARY_SHA256=$(sha256sum "$WORK_DIR/artifact/bin/YaneuraOu-runpod" | awk '{print $1}')
EVAL_ARCHIVE_SHA256_ACTUAL=
NNUE_SHA256=
if [[ -n "$YANEURAOU_EVAL_ARCHIVE_URL" ]]; then
  mkdir -p "$WORK_DIR/artifact/eval"
  curl -L --fail --retry 3 "$YANEURAOU_EVAL_ARCHIVE_URL" -o "$WORK_DIR/eval-archive"
  EVAL_ARCHIVE_SHA256_ACTUAL=$(sha256sum "$WORK_DIR/eval-archive" | awk '{print $1}')
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
  NNUE_SHA256=$(sha256sum "$WORK_DIR/artifact/eval/nn.bin" | awk '{print $1}')
fi

ARTIFACT_CREATED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ARTIFACT_NAME="$ARTIFACT_NAME" \
ARTIFACT_CREATED_AT="$ARTIFACT_CREATED_AT" \
YANEURAOU_REPOSITORY_URL="$YANEURAOU_REPOSITORY_URL" \
YANEURAOU_REF="$YANEURAOU_REF" \
SOURCE_COMMIT="$SOURCE_COMMIT" \
YANEURAOU_EDITION="$YANEURAOU_EDITION" \
YANEURAOU_TARGET_CPU="$YANEURAOU_TARGET_CPU" \
BINARY_SHA256="$BINARY_SHA256" \
YANEURAOU_EVAL_ARCHIVE_URL="$YANEURAOU_EVAL_ARCHIVE_URL" \
EVAL_ARCHIVE_SHA256="$EVAL_ARCHIVE_SHA256_ACTUAL" \
NNUE_SHA256="$NNUE_SHA256" \
python3 - <<'PY' > "$WORK_DIR/artifact/manifest.json"
import json
import os


def optional(value: str) -> str | None:
    return value or None


manifest = {
    "artifact_name": os.environ["ARTIFACT_NAME"],
    "created_at": os.environ["ARTIFACT_CREATED_AT"],
    "engine": "YaneuraOu",
    "repository_url": os.environ["YANEURAOU_REPOSITORY_URL"],
    "ref": os.environ["YANEURAOU_REF"],
    "source_commit": os.environ["SOURCE_COMMIT"],
    "edition": os.environ["YANEURAOU_EDITION"],
    "target_cpu": os.environ["YANEURAOU_TARGET_CPU"],
    "binary": "bin/YaneuraOu-runpod",
    "binary_sha256": os.environ["BINARY_SHA256"],
    "eval_dir": "eval",
    "eval_archive_url": optional(os.environ["YANEURAOU_EVAL_ARCHIVE_URL"]),
    "eval_archive_sha256": optional(os.environ["EVAL_ARCHIVE_SHA256"]),
    "nnue_sha256": optional(os.environ["NNUE_SHA256"]),
}
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

mkdir -p "$(dirname "$OUTPUT")"
tar -I zstd -cf "$OUTPUT" -C "$WORK_DIR/artifact" .
sha256sum "$OUTPUT" | tee "$OUTPUT.sha256"
