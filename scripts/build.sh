#!/usr/bin/env bash
# Build Gufo with the deepseek41 (DeepSeek V4.1 Flash / Engram) extension.
#
# The default GUFO_REF carries TextProvider::claim_file (the extension-side
# fallback hook, PR to gufo main). Once the claim hook AND native deepseek41
# support are unnecessary (i.e. only the claim hook is needed), repin to
# whatever gufo revision contains it.
set -euo pipefail

GUFO_REPO=${GUFO_REPO:-https://github.com/Farenheith/gufo.git}
GUFO_REF=${GUFO_REF:-feat/rejected-gguf-extension-fallback}
GUFO_DIR=${GUFO_DIR:-"$(cd "$(dirname "$0")/.." && pwd)/build-gufo"}
EXT_DIR=$(cd "$(dirname "$0")/.." && pwd)
GUFO_EXTENSIONS_LIST=${GUFO_EXTENSIONS_LIST:-deepseek41}
PRESET=${PRESET:-release}

if [[ -d "$GUFO_DIR/.git" ]]; then
  git -C "$GUFO_DIR" fetch --quiet "$GUFO_REPO" "$GUFO_REF"
  git -C "$GUFO_DIR" checkout --quiet FETCH_HEAD
else
  git clone --quiet "$GUFO_REPO" "$GUFO_DIR"
  git -C "$GUFO_DIR" checkout --quiet "$GUFO_REF"
fi

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
