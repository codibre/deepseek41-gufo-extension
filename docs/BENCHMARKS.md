# Benchmarks: V4.1-Flash (-1017, GSQ Coder) vs Qwen3.8-Flash-Next

Collected 2026-11-03 from lab cards + independent trackers; harnesses differ,
treat deltas < ~3 pts as noise. "Coder" = this repo's target file.

## Full-model baselines (lab-reported)

| Benchmark | V4.1-Flash-1017 | V4.1-Flash | Qwen3.8-Flash-Next |
|---|---|---|---|
| DeepSWE 1.1 (agentic coding) | 77.0 | 74.2 | 58.7 |
| NL2Repo-Bench (repo-level gen) | 68.2 | 65.4 | 48.1 |
| HLE no-tools | 42.6 | 36.8 | 35.9 |
| Agent's Last Exam pass@1 | 35.4 | 31.8 | 24.3 |
| GPQA Diamond | — | 90.9 | 91.7 |
| LiveCodeBench v6 | — | 89.71* | 91.9 |
| SWE-bench Pro | — | — | 62.5 |
| Arena Code Elo | — | 1614 | 1637 |

\* from the ISTA GSQ card. Sources: Qwen HF card, aireleasetracker.com,
benchlm.ai, respan.ai, anotherwrapper, llm-stats.

## GSQ-RCO Coder (pruned+quantized — the file this extension targets)

At reasoning effort 50 (ISTA card): SWE-bench Verified 77.00 (base 81.20,
94.8% retained); LiveCodeBench v6 88.57 (base 89.71, 98.7% retained).
Half of 384 routed experts removed per layer; engram tables NOT pruned.
Card explicitly warns: capability directed at code/agentic/long-context/
vision; general-purpose use → unpruned release.

## Read

- Agentic, repo-level, long-horizon coding: V4.1 lineage leads Flash-Next
  by a wide margin (DeepSWE +15..18, NL2Repo +17..20); the Coder keeps most
  of that (its SWE-bench Verified 77 vs full 81; no public DeepSWE for the
  Coder yet).
- Single-shot competitive coding / science / arena-style preference:
  Flash-Next ≥ V4.1 base (LCB 91.9 vs 89.7; GPQA +0.8; Arena Code Elo
  +23), and the Coder sits below base (LCB 88.6) → Flash-Next likely wins
  these head-to-head.
- Non-code domains: untested for the Coder; card says assume degradation.
- Cost side (this box): unified-memory APU, ~120 GB usable as VRAM.
  Coder = 72.1 GB weights → fully resident in the pool; KV at 1M ctx is
  ~1-2 GB; the 110.6 GB engram stays disk-mmap by design (24 short rows
  per token, served from page cache + NVMe). No CPU-paging cliff — decode
  speed should be in the same class as the resident Flash-Next slot.

## Cheap A/B once feasible

1. llama.cpp (master, deepseek41 support) serve the Coder file; same
   Hermes prompt set as the advanced slot; compare quality + tok/s.
2. After extension phase G: golden-prompt diff gufo vs llama.cpp, then
   swap into a llama-swap window for real agentic trials.
