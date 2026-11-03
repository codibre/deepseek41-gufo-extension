# DeepSeek V4.1 Flash → gufo extension: port spec

Target: any `general.architecture == "deepseek41"` GGUF. Verified against
`ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF` (IQ2_XXS coder);
full KV dump in `docs/model-metadata.txt`.

References:
- Upstream llama.cpp `src/models/deepseek41.cpp` + `deepseek4.cpp`
  (blobless clone branch `upstream-master` in `~/projects/personal/llama.cpp`;
  extracted copies were in `~/.hermes/cache/scratch/dsv4/`).
- gufo native `src/models/deepseek_v4` + `deepseek_v4_flash` (arch
  `deepseek4`) — same DSA/indexer/hyper-connection/compressed-KV engine.
- gufo/qwen4exp native PLE n-gram table (also the qwen38 GSQ extension) —
  the lazy hashed-table read machinery Engram needs already exists there.
- **Official DeepSeek reference** (PyTorch, data-center kernels; MIT):
  `huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash` `inference/` — a local
  copy is vendored under `reference/deepseek-inference/` (model.py,
  engram.py, kernel.py, generate.py, config.json). `model.py::Engram` is
  the exact Engram semantics; `SharedAttentionRuntime` documents the
  kv_source/index_source/candidate_source sharing; `kernel.py` has the
  TileLang sinkhorn/sparse-attn. Per DeepSeek the full checkpoint is
  552B main + 196B engram, 40 layers, DSpark MTP (optional speculative).

## Container layout (verified 2026-11-03)

* `IQ2_XXS/…-00001-of-00006.gguf` … shard 6; a sibling
  `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` is a **symlink** to shard 6.
  Entry point = shard 1. `split.count=6`, `split.tensors.count=1043`.
* Shard 1 is metadata-only (`n_tensors == 0`) carrying all 77 KV pairs;
  shards 2..5 list+store their own tensors; shard 6 stores the single
  `per_layer_token_embd.weight` (IQ4_NL). Offsets are relative to each
  shard's own data region. `Ds41Reader` handles it (metadata-only shard 1
  was the one relaxation over the qwen38 GsqReader, plus typed array
  getters).
* Small tensors (norms, biases, gates) are F32 — real dtype, not padding.
* Expert slabs are 3-D IQ2_XXS (`ffn_gate_exps`, `ffn_down_exps`,
  `[*, 5120, 192]`): core strict loader rejects 3-D non-Q8K quantized →
  claim path mandatory.
* Sizes: shard1 5.7 MB; shards 2–4 ≈ 19.78 GB each; shard 5 ≈ 12.75 GB;
  engram shard 110.6 GB. IQ2_XXS total ≈ 72.1 GB + 110.6 GB engram.

## Architecture (from docs/model-metadata.txt)

40 layers, n_embd 5120, 64 heads, **1 KV head**, head_dim 512 (+64 rope),
q_lora_rank 1280 (MLA-style), output_lora_rank 1024 / 8 output groups.
RoPE **YaRN factor 16** (orig 65536 → `context_length` 1048576),
freq_base 10000, beta_fast 32, beta_slow 1. `sliding_window` 128.
MoE: 192 routed experts **top-6** + 1 shared, ff 2304, sigmoid gating
(`expert_gating_func=4`), weights scale 1.5 + norm, swiglu clamp 10 for
routed and shared experts (per-layer arrays, all 10.0 here).
`hyper_connection.count = 4`, sinkhorn 20 iters, eps 1e-6 → hc_mult 4; the
V4.1 attention/MoE residual is the V4 hc stream — gufo's ds4 engine has it.
DSA indexer: 32 heads × 128, **top_k 512**, used at layers
`indexer.source_layers = [2,8,14,20,24,28,32,36]` only (others dense
sliding+full); compressed-KV producers `kv_source_layers = [2,8,14,20]`;
`compress_ratios` per layer (0/1/2), `compress_rope_freq_base` 160000.
Candidate blocks: source layer 20, block 8, top 2048 blocks, pin of the
partially filled block (`cand_pin` +1e30, upstream lines 276+).

### Engram / PLE (the new subsystem)

Metadata: `ple.layers=[1,14]`, `ple.ngram_size=4`, `ple.heads_per_ngram=8`
→ `n_cols = 3*8 = 24` heads per module, `embedding_length_per_layer_input
= 256`, 8 `ple.layer_multipliers` (u64), 48 `ple.head_offsets`, 48
`ple.head_vocab_sizes` (~16.0M primes), `engram.pad_token_id=2`,
`engram.token_map` (129280 u32 — **pruned vocabulary** class ids).

Per token (upstream `engram_hash`, deepseek41.cpp:240): map token→class
through `token_map` (out of range → DEAD); build `toks[0..3]` = current +
3 prev tokens (`get_prev_tokens`, blocked at sequence start/DEAD → pad
class); `rolling = toks[0]*mult[0]`, then `rolling ^= toks[o]*mult[o]` for
o=1..3; for each of 24 head columns: `row = rolling % primes[col] +
offs[col]` (module e occupies cols/rows offset per module — all modules
share one table). 24 int32 row ids per token per engram module.

Table: `per_layer_token_embd.weight` IQ4_NL `[256, n_rows]` with
`n_rows = max(head_offsets[c]+head_vocab_sizes[c])` (≈ 768M rows, 110.6 GB,
= shard 6). Lazily read: upstream `llama_prefetch_rows` batches one
readahead per ubatch (deepseek41.cpp:315) — mmap + madvise, never bulk-load.

Module (per engram layer `il`, e = `engram_index(il)`; upstream
`build_engram` :372): gather rows `[6144, nt]` → `kv = engram_kv[il] @ rows`
(`engram_kv` `[6144, 5120*(hc+1)=25600]`); key = first hc·n_embd, value =
last n_embd; `dot = sum((rmsnorm(x) * rmsnorm(key)) * gate_w)/sqrt(5120)`;
`gate = sigmoid(copysign(sqrt(max(|dot|,1e-6)), dot))` (×`engram_keep`
mask: embedding/image tokens get 0); `x += value ⊗ gate` on the hc
residual stream. Applied before attention at layers 1 and 14.

`token_map` present (pruned container): look up by *file* token id; an
unpruned container still ships a map (verify when a full model lands).

## Port phases

* **A — claim + reader + scaffold (done).** `Ds41Reader` fork, header-only
  claim on `general.architecture == "deepseek41"`, manifest key
  `gguf:deepseek41`, `scripts/inspect.cpp` harness. Needs the core
  `claim_file` hook (Farenheith/gufo `feat/rejected-gguf-extension-fallback`).
* **B — vendor engine.** Copy gufo `src/models/deepseek_v4` (+ `_flash`
  deltas) to `src/engine/`, include rewrite per the qwen38
  `scripts/rewrite_includes.py` recipe, namespace `gufo::models::deepseek41`,
  link `gufo_ext_runtime` only. Tensor access via `Ds41Reader` (claimed
  files hand us an empty core reader by contract).
* **C — GGUF-driven config.** Replace DS4_* equality checks with the keys
  dumped in docs/model-metadata.txt (40 layers, 192 experts top-6, ff 2304,
  o_lora 1024/8 groups, indexer top-k 512, YaRN×16 1M ctx, per-layer
  swiglu clamp arrays, hc count 4 — hc stays, it is NOT removed).
* **D — graph deltas.** (1) Engram module above: host hash → int32 row
  ids tensor input (`engram_ids`, per module) + `engram_keep`; gather from
  Ds41Reader-mapped IQ4_NL rows (reuse the qwen4exp PLE gather/lazy path);
  per-layer `engram_kv`/`engram_gate` weights (F32 here). (2) kv_source /
  indexer source-layer sharing tables ([2,8,14,20] producers; 8 consumer
  layers). (3) Candidate-block top-2048 selection from layer 20 scores
  incl. `cand_pin` of the partial block. (4) YaRN scaling wiring.
* **E — tokenizer/chat.** `tokenizer.ggml.pre = "joyai-llm"` — check gufo's
  pre-tokenizer table has it, else vendor the regex (qwen38 did this via
  vendored `family/text/tokenizer.cpp`). chat_template 18.4 KB
  ("Unsloth's DeepSeek-V4.1 template"); no add_bos key → default handling.
* **F — kernels (HIP).** MoE GEMM paths parameterized for 192×2304 IQ2_XXS
  fused slabs (qwen38 wave64 MoE covers 256×2560 fused IQ2_XXS — very
  likely reusable); IQ4_NL row gather for engram; DSA indexer kernels
  exist (gufo ds4) but must accept top_k 512/block 8 + candidate masks.
* **G — validation.** Golden compare vs llama.cpp (master has full
  deepseek41 support) same file/prompt/seed; then `gufo serve` window +
  llama-swap slot; A/B vs rejection on native main (must still reject).

## Host budget (189 GB RAM / 96 GB VRAM on this box)

Everything stays file-backed mmap; engram random 24-row-per-token reads are
served from page cache on first pass, then evictable. Weights 72.1 GB +
engram 110.6 GB > 189 GB RAM total, so paging is by design — llama.cpp runs
this file the same way. First load warms through disk; keep llama-swap
`--keep-kv`/ttl settings in mind when wiring the slot.

## Non-goals (first cut)

Vision (`deepseek41_vl_*` keys, `general.file_type=19` mismatch — tensor
types are authoritative); speculative/MTP draft support; CUDA path.
