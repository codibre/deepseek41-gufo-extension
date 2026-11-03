# deepseek41-gufo-extension

Out-of-tree [gufo](https://github.com/gufo-org/gufo) model extension for
**DeepSeek V4.1 Flash** (`general.architecture == "deepseek41"`), starting
with the GSQ-RCO Coder quantizations
([ISTA-DASLab](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF)).

gufo core has no `deepseek41` model, and its strict GGUF loader cannot open
these containers anyway (metadata-only shard 1 of a 6-way split, IQ2_XXS
fused 3-D expert slabs). This extension claims every `deepseek41` artifact
via `TextProvider::claim_file` and owns loading end to end.

## Status

* ✅ **Phase A** — manifest, claim probe, standalone `Ds41Reader`
  (split-aware GGUF reader, no quant validation), inspect harness.
  Verified against the real IQ2_XXS coder container.
* 🔜 **Phase B+** — vendored V4 engine with GGUF-driven config, Engram
  module, source-layer sharing + candidate blocks, YaRN×16, HIP kernels.
  Full plan with verified metadata: [`docs/PORT-NOTES.md`](docs/PORT-NOTES.md).

Requires a gufo core carrying the `claim_file` hook
(`Farenheith/gufo` branch `feat/rejected-gguf-extension-fallback`).

## Build

```bash
scripts/build.sh            # clones gufo (claim branch) into build-gufo, builds
# reuse an existing tree:
GUFO_DIR=../qwen38-flash-next-gsq-gufo-extension/build-gufo scripts/build.sh
```

Inspect a container (no runtime, no full model load):

```bash
BG=build-gufo
g++ -std=c++20 -O1 -I"$BG/include" -I"$BG" -Isrc \
    src/model/gguf/ds41_reader.cpp scripts/inspect.cpp -o /tmp/ds41_inspect
/tmp/ds41_inspect <dir>/IQ2_XXS/...-00001-of-00006.gguf          # headers
/tmp/ds41_inspect <dir>/IQ2_XXS/...-00001-of-00006.gguf --map    # + tensor ptrs
```

## Layout

```
extension.json/cmake        manifest + targets (links gufo_ext_runtime only)
src/model/gguf/ds41_reader  split-aware reader (fork of qwen38 GsqReader)
src/model/probe|provider    claim_file registration (key gguf:deepseek41)
docs/PORT-NOTES.md          verified port spec (phases A–G)
docs/model-metadata.txt     full GGUF KV dump of the target file
reference/deepseek-inference  official MIT reference (read-only, not built)
scripts/inspect.cpp         standalone container harness
```

## Model files (not in git)

IQ2_XXS coder: weights 72.1 GB + 110.6 GB Engram table (shard 6, kept
mmap'd; see docs for the host-budget notes). Point gufo at shard 1.
