#!/usr/bin/env bash
# DeepSeek-V4.1-Flash via gufo + out-of-tree deepseek41 extension (TEST slot).
# llama-swap slot "deepseek41" -> this script; gufo binary is the extension
# build (GUFO_EXTENSIONS=deepseek41). Tracking: gufo#997.
set -euo pipefail
export GUFO_DEEPSEEK_ROCM_NAN_PROBE=1
export DS4_ENGRAM_DEBUG=1
export DS4_DEBUG_STATE=1
# export DS4_GOLDEN0=1
# export GUFO_DEEPSEEK_ROCM_DECODE_STAGE_PROFILE=1
export GUFO_DS4_NO_COMP_ATTN=1
# export DS4_ENGRAM_NO_KV=1
# export DS4_ENGRAM_NO_GATE=1
# export GUFO_DS4_PREFILL_AS_DECODE=1

# export DS4_IQ_ZERO=1

# GUFO_DS4_PREFILL_AS_DECODE=1 # normal batch path test head73
# GUFO_DS4_NO_COMP_ATTN=1 # full-stack test head70
# DS4_ENGRAM_NO_KV=1 # full-stack test head70
# DS4_ENGRAM_NO_GATE=1 # full-stack test head70
export DS4_GOLDEN0=1
export DS4_H_DUMP=1
exec /home/tosol/projects/personal/deepseek41-gufo-extension/build-gufo/build/release/gufo \
  serve llm \
  --model /home/tosol/.lmstudio/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF/IQ2_XXS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-00001-of-00006.gguf \
  --served-model-name deepseek41 \
  --sessions 1 \
  --context 65536 \
  "$@"
