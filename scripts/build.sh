#!/usr/bin/env bash
# Build Gufo with the deepseek41 (DeepSeek V4.1 Flash / Engram) extension.
#
# Default GUFO_REF pins gufo-org PR #966 head (TextProvider::claim_file) —
# the hook the extension needs until that lands on main. Repin to main once
# merged. See gufo-org/gufo issue #997 for the tracking thread.
set -euo pipefail

GUFO_REPO=${GUFO_REPO:-https://github.com/gufo-org/gufo.git}
GUFO_REF=${GUFO_REF:-refs/pull/966/head}
GUFO_DIR=${GUFO_DIR:-"$(cd "$(dirname "$0")/.." && pwd)/build-gufo"}
EXT_DIR=$(cd "$(dirname "$0")/.." && pwd)
GUFO_EXTENSIONS_LIST=${GUFO_EXTENSIONS_LIST:-deepseek41}
PRESET=${PRESET:-release}

if [[ ! -d "$GUFO_DIR/.git" ]]; then
  git clone --quiet "$GUFO_REPO" "$GUFO_DIR"
fi
git -C "$GUFO_DIR" fetch --quiet "$GUFO_REPO" "$GUFO_REF"
git -C "$GUFO_DIR" checkout --quiet FETCH_HEAD

# Explicit mode (-DGUFO_EXTENSIONS) alone excludes the in-tree models from
# the shipped set; never edit extensions.lock.
cd "$GUFO_DIR"

cmake --preset "$PRESET" \
  -DGUFO_EXTENSION_DIRS="$EXT_DIR" \
  -DGUFO_EXTENSIONS="$GUFO_EXTENSIONS_LIST"
cmake --build --preset "$PRESET" --target gufo --parallel "${JOBS:-4}"

echo
"$GUFO_DIR/build/$PRESET/gufo" diagnose --section extensions || true
echo
echo "Try: $GUFO_DIR/build/$PRESET/gufo chat --model <v41 shard1>.gguf"
