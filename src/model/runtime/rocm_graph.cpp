#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include "../kernels/rocm/resident_api.h"
#include <gufo/ext/host.hpp>
#include <gufo/ext/platform/fs.hpp>
#include "dspark_internal.h"
#include "dspark_policy.h"
#include "model_data_internal.h"
#include "native_internal.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DS4_SPEC_PREFIX_SLOTS 4u

static bool g_rope_probe_done = false;
static bool g_woa_probe_done = false;

static float layer_rope_freq_base(uint32_t layer) {
    return ds4_layer_compress_ratio(layer) != 0 && DS4_COMPRESS_ROPE_FREQ_BASE > 0.0f
        ? DS4_COMPRESS_ROPE_FREQ_BASE
        : DS4_ROPE_FREQ_BASE;
}

static float layer_rope_freq_scale(uint32_t layer) {
    if (ds4_layer_compress_ratio(layer) == 0 || DS4_ROPE_SCALE_FACTOR <= 0.0f) {
        return 1.0f;
    }
    return 1.0f / DS4_ROPE_SCALE_FACTOR;
}

static uint32_t ds4_default_prefill_cap_for_prompt(int prompt_len) {
    if (prompt_len <= 0) return 1;
    uint32_t capacity = (uint32_t)prompt_len;

    /* Long prompts run in 4,096-token chunks. An 8K capacity regressed the 8K
     * prompt (201.80 against 210.21 tok/s) and was noise in a 16K replay, so
     * this is the policy rather than a tunable. */
    if (prompt_len > 4096) {
        capacity = 4096u;
    }

    if (capacity == 0) capacity = 1;
    if (capacity > (uint32_t)prompt_len) capacity = (uint32_t)prompt_len;
    return capacity;
}

/* =========================================================================
 * ROCm Release Graph State.
 * =========================================================================
 *
 * The release ROCm executor owns one fixed set of tensors for single-token
 * decode and another for batched prefill.  The structure is DS4-specific:
 * tensor names follow the model stages rather than generic graph nodes.
 */

struct ds4_rocm_graph {
    /* One-token decode tensors.  These stay allocated for the life of a
     * session; a generated token enters as an embedding in cur_hc and leaves as
     * logits after all 43 layers update their raw/compressed/indexer caches. */
    ds4_gpu_tensor *cur_hc;
    ds4_gpu_tensor *flat_hc;
    ds4_gpu_tensor *hc_mix;
    /* Raw (pre-sigmoid) mix row of the last executed FFN sublayer: the
     * V4.1 output head collapses with the last FFN's mix, but hc_mix is
     * overwritten by every attention sublayer, so snapshot it here. */
    ds4_gpu_tensor *hc_mix_ffn_raw;
    ds4_gpu_tensor *hc_split;
    /* Attention's split lives in a sibling buffer: the stream collapse
     * for each sublayer is weighted by the PREVIOUS sublayer's pre (official
     * Block.forward carries ffn_pre -> next attention, attn_pre -> this FFN),
     * so both splits must be alive at the same time. */
    ds4_gpu_tensor *hc_split_a;
    ds4_gpu_tensor *hc_pre_onehot;
    ds4_gpu_tensor *hc_pre;
    ds4_gpu_tensor *hc_post;
    ds4_gpu_tensor *hc_comb;
    ds4_gpu_tensor *attn_cur;
    ds4_gpu_tensor *attn_norm;
    /* Q8 forms of the normalized rows, written by their producer for the
     * quantized projections that read them. */
    ds4_gpu_tensor *attn_norm_q8;
    ds4_gpu_tensor *qr;
    ds4_gpu_tensor *qr_norm;
    ds4_gpu_tensor *qr_norm_q8;
    ds4_gpu_tensor *q;
    ds4_gpu_tensor *kv_raw;
    ds4_gpu_tensor *kv;

    /* Persistent KV state.  Raw KV is a sliding-window ring per layer.  Ratio-4
     * layers also keep an indexer-compressed cache; ratio-128 layers keep only
     * the attention-compressed cache.  The small state tensors are compressor
     * frontiers for the next compressed row, so they must be snapshotted with
     * the row counters whenever a checkpoint is saved or partially rewound. */
    ds4_gpu_tensor *layer_raw_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_comp_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_comp_cache_f16[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_index_comp_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_index_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_index_state_score[DS4_MAX_LAYER];

    uint32_t layer_n_comp[DS4_MAX_LAYER];
    uint32_t layer_n_index_comp[DS4_MAX_LAYER];
    uint32_t raw_cap;
    /* Maximum compressed-row capacity across layers.  Shared work buffers use
     * this worst-case size because ratio-4 indexer layers can still reach it. */
    uint32_t comp_cap;
    /* Persistent compressed caches are per layer, so size them from the actual
     * layer compression ratio instead of pessimistically using the ratio-4 cap
     * for every ratio-128 layer. */
    uint32_t layer_comp_cap[DS4_MAX_LAYER];
    bool managed_kv_cache;
    /* Per-layer work tensors.  They are reused in place by every layer instead
     * of allocating a generic graph arena.  This is why the code is verbose but
     * predictable: each pointer names an actual DS4 stage. */
    ds4_gpu_tensor *comp_kv_cur;
    ds4_gpu_tensor *comp_sc_cur;
    /* Decode's indexer compressor projections, kept apart from comp_*_cur so
     * both compressors of a layer update in one launch. */
    ds4_gpu_tensor *index_kv_cur;
    ds4_gpu_tensor *index_sc_cur;
    ds4_gpu_tensor *indexer_q;
    ds4_gpu_tensor *indexer_weights;
    ds4_gpu_tensor* indexer_scores;
    ds4_gpu_tensor *comp_selected;
    /* V4.1 source sharing.  comp_latent stages the pre-RoPE pooled latent of
     * the emit step so the index key can be derived before rotation.
     * shared_topk carries one published top-k row from the indexer layer to
     * every grouped reader of the same decode step; cand_mask holds the
     * layer-20 two-level candidate mask consumed by later indexer layers. */
    ds4_gpu_tensor *comp_latent;
    ds4_gpu_tensor *shared_topk;
    ds4_gpu_tensor *cand_mask;
    uint32_t shared_topk_n;
    uint32_t shared_topk_kv;   /* compressed-cache owner the row indexes */
    uint32_t cand_mask_n;
    bool cand_mask_valid;
    /* V4.1 prefill source sharing: per-token candidate masks for a whole
     * chunk (n_tokens x n_comp f32) and the published chunk-wide top-k.
     * pf_sel_owner is the compressed-cache owner the published rows index
     * (UINT32_MAX = nothing published for this chunk). */
    ds4_gpu_tensor *pf_cand_mask;
    uint64_t pf_cand_mask_cap;
    uint32_t pf_cand_n;
    bool pf_cand_valid;
    uint32_t pf_sel_owner;
    uint32_t pf_ratio;
    ds4_gpu_tensor *heads;
    ds4_gpu_tensor *heads_q8;
    ds4_gpu_tensor *attn_low;
    ds4_gpu_tensor *attn_low_q8;
    ds4_gpu_tensor *attn_out;
    ds4_gpu_tensor *after_attn_hc;
    ds4_gpu_tensor *ffn_cur;
    ds4_gpu_tensor *ffn_norm;
    ds4_gpu_tensor *ffn_norm_q8;
    ds4_gpu_tensor *ffn_norm_q8k;
    ds4_gpu_tensor *shared_gate;
    ds4_gpu_tensor *shared_up;
    ds4_gpu_tensor *shared_mid;
    ds4_gpu_tensor *shared_mid_q8;
    ds4_gpu_tensor *shared_out;
    ds4_gpu_tensor *router_logits;
    ds4_gpu_tensor *router_probs;
    ds4_gpu_tensor *router_selected;
    ds4_gpu_tensor *router_weights;
    ds4_gpu_tensor *routed_gate;
    ds4_gpu_tensor *routed_up;
    ds4_gpu_tensor *routed_mid;
    ds4_gpu_tensor *routed_down;
    ds4_gpu_tensor *routed_out;
    ds4_gpu_tensor *after_ffn_hc;
    ds4_gpu_tensor *output_pre;
    ds4_gpu_tensor *output_weights;
    ds4_gpu_tensor *output_embd;
    ds4_gpu_tensor *output_norm;
    ds4_gpu_tensor *logits;

    uint32_t prefill_cap;
    uint32_t raw_window;

    /* Batched prefill tensors.  Prefill is layer-major: a chunk of prompt
     * tokens moves through layer 0, then layer 1, and so on, updating the same
     * persistent caches used by decode.  Keeping this separate from decode
     * avoids a slow loop of one-token graph steps for long prompts. */
    ds4_gpu_tensor *prefill_tokens;
    ds4_gpu_tensor* batch_stage_scratch;
    ds4_gpu_tensor *batch_cur_hc = nullptr;
    // Engram (D3): per-module staged head ids + gathered rows/kv scratch.
    ds4_gpu_tensor *engram_ids[DS4_PLE_MAX_LAYERS] = {};
    ds4_gpu_tensor *engram_rows[DS4_PLE_MAX_LAYERS] = {};
    ds4_gpu_tensor *engram_kv[DS4_PLE_MAX_LAYERS] = {};
    std::vector<int32_t> token_hist;
    ds4_gpu_tensor *batch_next_hc;
    ds4_gpu_tensor* batch_flat_hc;
    ds4_gpu_tensor *batch_hc_mix;
    ds4_gpu_tensor *batch_hc_split;
    ds4_gpu_tensor *batch_hc_split_a;
    ds4_gpu_tensor *batch_pre_onehot;
    ds4_gpu_tensor *batch_attn_cur;
    ds4_gpu_tensor *batch_attn_norm;
    ds4_gpu_tensor *batch_qr;
    ds4_gpu_tensor *batch_qr_norm;
    ds4_gpu_tensor *batch_q;
    ds4_gpu_tensor *batch_kv_raw;
    ds4_gpu_tensor *batch_kv;
    ds4_gpu_tensor *batch_comp_kv;
    ds4_gpu_tensor *batch_comp_sc;
    ds4_gpu_tensor *batch_indexer_q;
    ds4_gpu_tensor *batch_indexer_weights;
    ds4_gpu_tensor *batch_heads;
    ds4_gpu_tensor *batch_attn_low;
    ds4_gpu_tensor *batch_attn_out;
    ds4_gpu_tensor *batch_group_tmp;
    ds4_gpu_tensor *batch_low_tmp;
    ds4_gpu_tensor *batch_after_attn_hc;
    ds4_gpu_tensor *batch_ffn_cur;
    ds4_gpu_tensor *batch_ffn_norm;
    ds4_gpu_tensor *batch_shared_gate;
    ds4_gpu_tensor *batch_shared_up;
    ds4_gpu_tensor *batch_shared_mid;
    ds4_gpu_tensor *batch_shared_out;
    ds4_gpu_tensor *batch_router_logits;
    ds4_gpu_tensor *batch_router_probs;
    ds4_gpu_tensor *batch_router_selected;
    ds4_gpu_tensor *batch_router_weights;
    ds4_gpu_tensor *batch_routed_gate;
    ds4_gpu_tensor *batch_routed_up;
    ds4_gpu_tensor *batch_routed_mid;
    ds4_gpu_tensor *batch_routed_down;
    ds4_gpu_tensor *batch_routed_out;
    ds4_gpu_tensor* batch_dspark_verify_cur_hc;
    ds4_gpu_tensor* batch_dspark_verify_after_attn_hc;
    uint32_t batch_dspark_verify_rows_cap;
    bool borrows_batch_workspace;
    ds4_rocm_graph* batch_workspace_owner;
    uint32_t batch_workspace_borrowers;
    // Keep the largest scratch arena cached. Older arenas remain alive only
    // while sessions still borrow them and are pruned on session creation.
    ds4_rocm_graph* next_batch_workspace;

    /* =====================================================================
     * DSpark speculative verification state.
     * =====================================================================
     *
     * A verification block runs the drafted suffix through the batched layer
     * path in one command stream, then reads back only the target's top token
     * per row.  Everything here is allocated on first use so a session that
     * never enables DSpark keeps exactly the residency it had before.
     */
    uint32_t spec_rows_cap;
    ds4_gpu_tensor *spec_logits;
    ds4_gpu_tensor *spec_row_tops;
    ds4_gpu_tensor *spec_frontier_logits;

    /*
     * Rollback frontier.
     *
     * Raw KV rows and compressed cache rows are append-only: a rejected block
     * leaves stale rows that the next write overwrites, and the row counters
     * say how much is live.  The compressor frontier tensors are the exception
     * because they are updated in place, so they are the only cache class a
     * rejected block has to restore.
     */
    ds4_gpu_tensor *spec_saved_attn_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_saved_attn_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_saved_index_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_saved_index_state_score[DS4_MAX_LAYER];
    uint32_t spec_saved_n_comp[DS4_MAX_LAYER];
    uint32_t spec_saved_n_index_comp[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix_attn_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix_attn_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix_index_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix_index_state_score[DS4_MAX_LAYER];
    uint32_t spec_prefix_n_comp[DS4_SPEC_PREFIX_SLOTS][DS4_MAX_LAYER];
    uint32_t spec_prefix_n_index_comp[DS4_SPEC_PREFIX_SLOTS][DS4_MAX_LAYER];
    bool spec_frontier_valid;
    bool spec_capture_prefixes;

    /* =====================================================================
     * DSpark drafting state.
     * =====================================================================
     *
     * The drafter needs two things the target does not keep: the target's
     * hidden states at the sampled layers, and one KV ring per DSpark stage
     * holding those hidden states projected into that stage's key/value space.
     * Block work tensors are borrowed from the batch_* set because drafting,
     * prefill, and verification never overlap within a session.
     */
    const ds4_dspark_model *dspark;
    uint32_t dspark_cache_cap;
    uint32_t dspark_capture_rows_cap;
    bool dspark_capture_enabled;
    /* Absolute position one past the last injected context row. */
    uint32_t dspark_context_len;
    ds4_gpu_tensor *dspark_kv_cache[DS4_DSPARK_MAX_STAGES];
    ds4_gpu_tensor *dspark_hc_mean;
    ds4_gpu_tensor *dspark_hc_mean_rows;
    ds4_gpu_tensor *dspark_features;
    ds4_gpu_tensor *dspark_features_batch;
    ds4_gpu_tensor *dspark_fused;
    /* Fused feature of the last committed position. It becomes row 0 of every
     * stage's input, which is how the drafter sees the target's current state
     * on the query side rather than only as injected history. */
    ds4_gpu_tensor *dspark_markov_key;
    ds4_gpu_tensor* dspark_candidates;
    ds4_gpu_tensor* dspark_candidate_scratch;
    ds4_gpu_tensor *dspark_markov_index;
    /* Bitmask of sampled target layers captured for the current position. */
    uint32_t dspark_capture_mask;
    uint32_t dspark_capture_batch_mask;
    uint32_t dspark_capture_batch_start;
    uint32_t dspark_capture_batch_tokens;
};

typedef struct ds4_rocm_graph ds4_gpu_graph;

/* DSpark feature capture, defined with the drafting code below. Both are no-ops
 * until a support model is attached, so the target paths only pay a branch. */
static bool rocm_graph_dspark_capture_decode_layer(ds4_gpu_graph *g, uint32_t il);
static bool rocm_graph_dspark_capture_rows(ds4_gpu_graph* g, uint32_t il,
                                           const ds4_gpu_tensor* hc,
                                           uint32_t n_rows);
static bool rocm_graph_dspark_capture_batch_layer(ds4_gpu_graph *g,
                                                  uint32_t il,
                                                  uint32_t start,
                                                  uint32_t n_tokens);
static bool rocm_graph_dspark_inject(ds4_gpu_graph *g,
                                     uint32_t pos0,
                                     uint32_t n_rows);
static bool rocm_graph_dspark_capture_complete(const ds4_gpu_graph *g);
static bool rocm_graph_capture_prefix_attn_state(ds4_gpu_graph *g,
                                                 uint32_t il,
                                                 uint32_t slot);
static bool rocm_graph_capture_prefix_index_state(ds4_gpu_graph *g,
                                                  uint32_t il,
                                                  uint32_t slot);

/* Release every ROCm tensor owned by the whole-model graph runtime. */
static void rocm_graph_spec_free(ds4_gpu_graph *g) {
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->spec_saved_attn_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_saved_attn_state_score[il]);
        ds4_gpu_tensor_free(g->spec_saved_index_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_saved_index_state_score[il]);
        ds4_gpu_tensor_free(g->spec_prefix_attn_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_prefix_attn_state_score[il]);
        ds4_gpu_tensor_free(g->spec_prefix_index_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_prefix_index_state_score[il]);
        g->spec_saved_attn_state_kv[il] = NULL;
        g->spec_saved_attn_state_score[il] = NULL;
        g->spec_saved_index_state_kv[il] = NULL;
        g->spec_saved_index_state_score[il] = NULL;
        g->spec_prefix_attn_state_kv[il] = NULL;
        g->spec_prefix_attn_state_score[il] = NULL;
        g->spec_prefix_index_state_kv[il] = NULL;
        g->spec_prefix_index_state_score[il] = NULL;
    }
    ds4_gpu_tensor_free(g->spec_row_tops);
    ds4_gpu_tensor_free(g->spec_logits);
    ds4_gpu_tensor_free(g->spec_frontier_logits);
    g->spec_row_tops = NULL;
    g->spec_logits = NULL;
    g->spec_frontier_logits = NULL;
    g->spec_rows_cap = 0;
    g->spec_frontier_valid = false;
    g->spec_capture_prefixes = false;
}

static void rocm_graph_dspark_free(ds4_gpu_graph *g) {
    for (uint32_t stage = 0; stage < DS4_DSPARK_MAX_STAGES; stage++) {
        ds4_gpu_tensor_free(g->dspark_kv_cache[stage]);
        g->dspark_kv_cache[stage] = NULL;
    }
    ds4_gpu_tensor_free(g->dspark_markov_index);
    ds4_gpu_tensor_free(g->dspark_markov_key);
    ds4_gpu_tensor_free(g->dspark_candidates);
    ds4_gpu_tensor_free(g->dspark_candidate_scratch);
    ds4_gpu_tensor_free(g->dspark_fused);
    ds4_gpu_tensor_free(g->dspark_features_batch);
    ds4_gpu_tensor_free(g->dspark_features);
    ds4_gpu_tensor_free(g->dspark_hc_mean_rows);
    ds4_gpu_tensor_free(g->dspark_hc_mean);
    g->dspark_markov_index = NULL;
    g->dspark_markov_key = NULL;
    g->dspark_candidates = nullptr;
    g->dspark_candidate_scratch = nullptr;
    g->dspark_fused = NULL;
    g->dspark_features_batch = NULL;
    g->dspark_features = NULL;
    g->dspark_hc_mean_rows = NULL;
    g->dspark_hc_mean = NULL;
    g->dspark = NULL;
    g->dspark_cache_cap = 0;
    g->dspark_capture_rows_cap = 0;
    g->dspark_capture_enabled = false;
    g->dspark_context_len = 0;
    g->dspark_capture_mask = 0;
    g->dspark_capture_batch_mask = 0;
    g->dspark_capture_batch_start = 0;
    g->dspark_capture_batch_tokens = 0;
}

/*
 * Prompt capture can be as wide as a full prefill chunk, while decode and
 * verification need only block_size + 1 rows. Keep the large buffers only
 * while the owning session is actively prefilling so idle HTTP sessions do
 * not each pin hundreds of MiB of unified memory.
 */
static bool rocm_graph_dspark_resize_capture(ds4_gpu_graph *g,
                                             uint32_t rows_cap) {
    if (!g || !g->dspark || !g->dspark_capture_enabled) return true;
    if (rows_cap == 0 || rows_cap > g->prefill_cap) return false;
    if (rows_cap == g->dspark_capture_rows_cap && g->dspark_features &&
        g->dspark_fused) {
      return true;
    }

    const uint32_t feature_width =
        g->dspark->n_target_layers * DS4_N_EMBD;
    ds4_gpu_tensor *features = ds4_gpu_tensor_alloc(
            (uint64_t)rows_cap * feature_width * sizeof(float));
    ds4_gpu_tensor *fused = ds4_gpu_tensor_alloc(
            (uint64_t)rows_cap * DS4_N_EMBD * sizeof(float));
    if (!features || !fused) {
        ds4_gpu_tensor_free(fused);
        ds4_gpu_tensor_free(features);
        return false;
    }

    ds4_gpu_tensor_free(g->dspark_fused);
    ds4_gpu_tensor_free(g->dspark_features);
    g->dspark_features = features;
    g->dspark_fused = fused;
    g->dspark_capture_rows_cap = rows_cap;
    g->dspark_capture_mask = 0;
    g->dspark_capture_batch_mask = 0;
    g->dspark_capture_batch_start = 0;
    g->dspark_capture_batch_tokens = 0;
    return true;
}

static void rocm_graph_bind_batch_workspace(
        ds4_gpu_graph *g,
        const ds4_gpu_graph *workspace) {
    g->prefill_tokens = workspace->prefill_tokens;
    g->batch_stage_scratch = workspace->batch_stage_scratch;
    g->batch_cur_hc = workspace->batch_cur_hc;
    g->batch_next_hc = workspace->batch_next_hc;
    g->batch_flat_hc = workspace->batch_flat_hc;
    g->batch_hc_mix = workspace->batch_hc_mix;
    g->batch_hc_split = workspace->batch_hc_split;
    g->batch_attn_cur = workspace->batch_attn_cur;
    g->batch_attn_norm = workspace->batch_attn_norm;
    g->batch_qr = workspace->batch_qr;
    g->batch_qr_norm = workspace->batch_qr_norm;
    g->batch_q = workspace->batch_q;
    g->batch_kv_raw = workspace->batch_kv_raw;
    g->batch_kv = workspace->batch_kv;
    g->batch_comp_kv = workspace->batch_comp_kv;
    g->batch_comp_sc = workspace->batch_comp_sc;
    g->batch_indexer_q = workspace->batch_indexer_q;
    g->batch_indexer_weights = workspace->batch_indexer_weights;
    g->batch_heads = workspace->batch_heads;
    g->batch_attn_low = workspace->batch_attn_low;
    g->batch_attn_out = workspace->batch_attn_out;
    g->batch_group_tmp = workspace->batch_group_tmp;
    g->batch_low_tmp = workspace->batch_low_tmp;
    g->batch_after_attn_hc = workspace->batch_after_attn_hc;
    g->batch_ffn_cur = workspace->batch_ffn_cur;
    g->batch_ffn_norm = workspace->batch_ffn_norm;
    g->batch_shared_gate = workspace->batch_shared_gate;
    g->batch_shared_up = workspace->batch_shared_up;
    g->batch_shared_mid = workspace->batch_shared_mid;
    g->batch_shared_out = workspace->batch_shared_out;
    g->batch_router_logits = workspace->batch_router_logits;
    g->batch_router_probs = workspace->batch_router_probs;
    g->batch_router_selected = workspace->batch_router_selected;
    g->batch_router_weights = workspace->batch_router_weights;
    g->batch_routed_gate = workspace->batch_routed_gate;
    g->batch_routed_up = workspace->batch_routed_up;
    g->batch_routed_mid = workspace->batch_routed_mid;
    g->batch_routed_down = workspace->batch_routed_down;
    g->batch_routed_out = workspace->batch_routed_out;
    g->batch_dspark_verify_cur_hc = workspace->batch_dspark_verify_cur_hc;
    g->batch_dspark_verify_after_attn_hc =
        workspace->batch_dspark_verify_after_attn_hc;
    g->batch_dspark_verify_rows_cap = workspace->batch_dspark_verify_rows_cap;
    g->borrows_batch_workspace = true;
}

static void rocm_graph_free_batch_workspace(ds4_gpu_graph *g) {
  ds4_gpu_tensor_free(g->batch_dspark_verify_after_attn_hc);
  ds4_gpu_tensor_free(g->batch_dspark_verify_cur_hc);
  ds4_gpu_tensor_free(g->batch_routed_out);
  ds4_gpu_tensor_free(g->batch_routed_down);
  ds4_gpu_tensor_free(g->batch_routed_mid);
  ds4_gpu_tensor_free(g->batch_routed_up);
  ds4_gpu_tensor_free(g->batch_routed_gate);
  ds4_gpu_tensor_free(g->batch_router_weights);
  ds4_gpu_tensor_free(g->batch_router_selected);
  ds4_gpu_tensor_free(g->batch_router_probs);
  ds4_gpu_tensor_free(g->batch_router_logits);
  ds4_gpu_tensor_free(g->batch_shared_out);
  ds4_gpu_tensor_free(g->batch_shared_mid);
  ds4_gpu_tensor_free(g->batch_shared_up);
  ds4_gpu_tensor_free(g->batch_shared_gate);
  ds4_gpu_tensor_free(g->batch_ffn_norm);
  ds4_gpu_tensor_free(g->batch_ffn_cur);
  ds4_gpu_tensor_free(g->batch_after_attn_hc);
  ds4_gpu_tensor_free(g->batch_low_tmp);
  ds4_gpu_tensor_free(g->batch_group_tmp);
  ds4_gpu_tensor_free(g->batch_attn_out);
  ds4_gpu_tensor_free(g->batch_attn_low);
  ds4_gpu_tensor_free(g->batch_heads);
  ds4_gpu_tensor_free(g->batch_indexer_weights);
  ds4_gpu_tensor_free(g->batch_indexer_q);
  ds4_gpu_tensor_free(g->batch_comp_sc);
  ds4_gpu_tensor_free(g->batch_comp_kv);
  ds4_gpu_tensor_free(g->batch_kv);
  ds4_gpu_tensor_free(g->batch_kv_raw);
  ds4_gpu_tensor_free(g->batch_q);
  ds4_gpu_tensor_free(g->batch_qr_norm);
  ds4_gpu_tensor_free(g->batch_qr);
  ds4_gpu_tensor_free(g->batch_attn_norm);
  ds4_gpu_tensor_free(g->batch_attn_cur);
  ds4_gpu_tensor_free(g->batch_pre_onehot);
  ds4_gpu_tensor_free(g->batch_hc_split_a);
  ds4_gpu_tensor_free(g->batch_hc_split);
  ds4_gpu_tensor_free(g->batch_hc_mix);
  ds4_gpu_tensor_free(g->batch_flat_hc);
  ds4_gpu_tensor_free(g->batch_next_hc);
  ds4_gpu_tensor_free(g->batch_cur_hc);
  ds4_gpu_tensor_free(g->prefill_tokens);
    for (uint32_t e = 0; e < DS4_PLE_MAX_LAYERS; ++e) {
      ds4_gpu_tensor_free(g->engram_ids[e]);
      g->engram_ids[e] = nullptr;
      ds4_gpu_tensor_free(g->engram_rows[e]);
      g->engram_rows[e] = nullptr;
      ds4_gpu_tensor_free(g->engram_kv[e]);
      g->engram_kv[e] = nullptr;
    }
  ds4_gpu_tensor_free(g->batch_stage_scratch);
}

static void rocm_graph_free(ds4_gpu_graph *g) {
    rocm_graph_dspark_free(g);
    rocm_graph_spec_free(g);
    if (!g->borrows_batch_workspace) {
        rocm_graph_free_batch_workspace(g);
    }
    ds4_gpu_tensor_free(g->logits);
    ds4_gpu_tensor_free(g->output_norm);
    ds4_gpu_tensor_free(g->output_embd);
    ds4_gpu_tensor_free(g->output_weights);
    ds4_gpu_tensor_free(g->output_pre);
    ds4_gpu_tensor_free(g->after_ffn_hc);
    ds4_gpu_tensor_free(g->routed_out);
    ds4_gpu_tensor_free(g->routed_down);
    ds4_gpu_tensor_free(g->routed_mid);
    ds4_gpu_tensor_free(g->routed_up);
    ds4_gpu_tensor_free(g->routed_gate);
    ds4_gpu_tensor_free(g->router_weights);
    ds4_gpu_tensor_free(g->router_selected);
    ds4_gpu_tensor_free(g->router_probs);
    ds4_gpu_tensor_free(g->router_logits);
    ds4_gpu_tensor_free(g->shared_out);
    ds4_gpu_tensor_free(g->shared_mid_q8);
    ds4_gpu_tensor_free(g->shared_mid);
    ds4_gpu_tensor_free(g->shared_up);
    ds4_gpu_tensor_free(g->shared_gate);
    ds4_gpu_tensor_free(g->ffn_norm_q8k);
    ds4_gpu_tensor_free(g->ffn_norm_q8);
    ds4_gpu_tensor_free(g->ffn_norm);
    ds4_gpu_tensor_free(g->ffn_cur);
    ds4_gpu_tensor_free(g->after_attn_hc);
    ds4_gpu_tensor_free(g->attn_out);
    ds4_gpu_tensor_free(g->attn_low_q8);
    ds4_gpu_tensor_free(g->attn_low);
    ds4_gpu_tensor_free(g->heads_q8);
    ds4_gpu_tensor_free(g->heads);
    ds4_gpu_tensor_free(g->index_sc_cur);
    ds4_gpu_tensor_free(g->index_kv_cur);
    ds4_gpu_tensor_free(g->comp_sc_cur);
    ds4_gpu_tensor_free(g->comp_kv_cur);
    ds4_gpu_tensor_free(g->comp_selected);
    ds4_gpu_tensor_free(g->comp_latent);
    ds4_gpu_tensor_free(g->shared_topk);
    ds4_gpu_tensor_free(g->cand_mask);
    ds4_gpu_tensor_free(g->pf_cand_mask);
    ds4_gpu_tensor_free(g->indexer_scores);
    ds4_gpu_tensor_free(g->indexer_weights);
    ds4_gpu_tensor_free(g->indexer_q);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_raw_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_comp_cache[il]);
        ds4_gpu_tensor_free(g->layer_attn_comp_cache_f16[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_state_kv[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_state_score[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_comp_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_state_kv[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_state_score[il]);
    }
    ds4_gpu_tensor_free(g->kv);
    ds4_gpu_tensor_free(g->kv_raw);
    ds4_gpu_tensor_free(g->q);
    ds4_gpu_tensor_free(g->qr_norm_q8);
    ds4_gpu_tensor_free(g->qr_norm);
    ds4_gpu_tensor_free(g->qr);
    ds4_gpu_tensor_free(g->attn_norm_q8);
    ds4_gpu_tensor_free(g->attn_norm);
    ds4_gpu_tensor_free(g->attn_cur);
    ds4_gpu_tensor_free(g->hc_comb);
    ds4_gpu_tensor_free(g->hc_post);
    ds4_gpu_tensor_free(g->hc_pre);
    ds4_gpu_tensor_free(g->hc_split_a);
    ds4_gpu_tensor_free(g->hc_pre_onehot);
    ds4_gpu_tensor_free(g->hc_split);
    ds4_gpu_tensor_free(g->hc_mix);
    ds4_gpu_tensor_free(g->hc_mix_ffn_raw);
    ds4_gpu_tensor_free(g->flat_hc);
    ds4_gpu_tensor_free(g->cur_hc);
    memset(g, 0, sizeof(*g));
}

static bool rocm_tensor_fill_f32(ds4_gpu_tensor *t, float v, uint64_t n) {
    return ds4_gpu_tensor_fill_f32(t, v, n) != 0;
}

static uint64_t rocm_graph_kv_cache_bytes_for_context(uint32_t ctx_size, uint32_t raw_cap) {
    uint64_t bytes = (uint64_t)DS4_N_LAYER *
                     raw_cap *
                     DS4_N_HEAD_DIM *
                     sizeof(float);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t comp_cap = (uint64_t)(ctx_size / ratio + 2u);
        bytes += comp_cap * DS4_N_HEAD_DIM * sizeof(float);
        if (ratio == 4) {
            bytes += comp_cap * DS4_N_HEAD_DIM * sizeof(uint16_t);
            bytes += comp_cap * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
        } else if (ds4_layer_is_kv_source(il)) {
            /* V4.1: only the owner of a shared group stores index keys. */
            bytes += comp_cap * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
        }
    }
    return bytes;
}

static uint64_t rocm_graph_context_bytes_for_kv_policy(
        uint32_t  ctx_size,
        uint32_t  raw_cap,
        uint32_t  prefill_cap,
        uint64_t *kv_cache_bytes_out) {
    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    uint64_t comp_cap = (uint64_t)(ctx_size / min_ratio + 2u);
    if (comp_cap < 2u) comp_cap = 2u;
    const uint64_t kv_cache_bytes = rocm_graph_kv_cache_bytes_for_context(ctx_size, raw_cap);
    if (kv_cache_bytes_out) *kv_cache_bytes_out = kv_cache_bytes;
    return kv_cache_bytes + comp_cap * prefill_cap * sizeof(float);
}

static ds4_gpu_tensor *rocm_graph_alloc_kv_cache_tensor(bool managed, uint64_t bytes) {
    return managed ? ds4_gpu_tensor_alloc_managed(bytes) : ds4_gpu_tensor_alloc(bytes);
}

static bool rocm_graph_reserve_compressed_rows(ds4_gpu_graph *g, uint32_t il,
                                               uint32_t required) {
  if (required > g->layer_comp_cap[il]) {
    fprintf(stderr, "ds4: reserve comp rows: required=%u > cap=%u layer=%u\n",
            required, g->layer_comp_cap[il], il);
    return false;
  }
  const uint64_t row_bytes = DS4_N_HEAD_DIM * sizeof(float);
  const uint64_t capacity =
      ds4_gpu_tensor_bytes(g->layer_attn_comp_cache[il]) / row_bytes;
  if (required <= capacity)
    return true;

  const uint64_t next_capacity = std::min<uint64_t>(
      g->layer_comp_cap[il], std::max<uint64_t>(required, capacity * 2));
  using Tensor =
      std::unique_ptr<ds4_gpu_tensor, decltype(&ds4_gpu_tensor_free)>;
  Tensor attention(rocm_graph_alloc_kv_cache_tensor(g->managed_kv_cache,
                                                    next_capacity * row_bytes),
                   ds4_gpu_tensor_free);
  Tensor mirror(nullptr, ds4_gpu_tensor_free);
  Tensor indexer(nullptr, ds4_gpu_tensor_free);
  const uint32_t grow_ratio = ds4_layer_compress_ratio(il);
  const bool indexed = grow_ratio == 4;
  /* V4.1 keeps one index cache per shared group; it grows with the owner's
   * attention cache (readers never store rows). */
  const bool v41_index = grow_ratio != 4 && ds4_layer_is_kv_source(il) &&
                         ds4_layer_is_topk_source(il) &&
                         g->layer_index_comp_cache[il] != NULL;
  if (indexed) {
    mirror.reset(rocm_graph_alloc_kv_cache_tensor(
        g->managed_kv_cache,
        next_capacity * DS4_N_HEAD_DIM * sizeof(uint16_t)));
    indexer.reset(rocm_graph_alloc_kv_cache_tensor(
        g->managed_kv_cache,
        next_capacity * DS4_N_INDEXER_HEAD_DIM * sizeof(float)));
  } else if (v41_index) {
    indexer.reset(rocm_graph_alloc_kv_cache_tensor(
        g->managed_kv_cache,
        next_capacity * DS4_N_INDEXER_HEAD_DIM * sizeof(float)));
  }
  if (!attention || (indexed && (!mirror || !indexer)) ||
      (v41_index && !indexer)) {
    fprintf(stderr,
            "ds4: reserve comp rows: alloc failed layer=%u next=%llu "
            "attention=%p mirror=%p indexer=%p\n",
            il, (unsigned long long)next_capacity, (void *)attention.get(),
            (void *)mirror.get(), (void *)indexer.get());
    return false;
  }
  const auto copy_rows = [](ds4_gpu_tensor* to, const ds4_gpu_tensor* from,
                            uint64_t bytes) {
    return bytes == 0 || ds4_gpu_tensor_copy(to, 0, from, 0, bytes) != 0;
  };
  if (!copy_rows(attention.get(), g->layer_attn_comp_cache[il],
                 static_cast<uint64_t>(g->layer_n_comp[il]) * row_bytes) ||
      (indexed && (!copy_rows(mirror.get(), g->layer_attn_comp_cache_f16[il],
                              static_cast<uint64_t>(g->layer_n_comp[il]) *
                                  DS4_N_HEAD_DIM * sizeof(uint16_t)) ||
                   !copy_rows(indexer.get(), g->layer_index_comp_cache[il],
                              static_cast<uint64_t>(g->layer_n_index_comp[il]) *
                                  DS4_N_INDEXER_HEAD_DIM * sizeof(float)))) ||
      (v41_index && g->layer_index_comp_cache[il] &&
       !copy_rows(indexer.get(), g->layer_index_comp_cache[il],
                  static_cast<uint64_t>(g->layer_n_index_comp[il]) *
                      DS4_N_INDEXER_HEAD_DIM * sizeof(float)))) {
    fprintf(stderr, "ds4: reserve comp rows: copy failed layer=%u req=%u\n",
            il, required);
    return false;
  }
  // All replacements and byte-exact copies succeed before releasing any old
  // cache. These caches have no persistent views; snapshots store live rows.
  ds4_gpu_tensor_free(g->layer_attn_comp_cache[il]);
  ds4_gpu_tensor_free(g->layer_attn_comp_cache_f16[il]);
  ds4_gpu_tensor_free(g->layer_index_comp_cache[il]);
  g->layer_attn_comp_cache[il] = attention.release();
  g->layer_attn_comp_cache_f16[il] = mirror.release();
  g->layer_index_comp_cache[il] = indexer.release();
  return true;
}

static bool rocm_graph_reserve_compressed_position(ds4_gpu_graph* g,
                                                   uint32_t il, uint64_t end) {
  const uint32_t ratio = ds4_layer_compress_ratio(il);
  if (ratio == 0)
    return true;
  return rocm_graph_reserve_compressed_rows(
      g, il,
      static_cast<uint32_t>(
          std::min<uint64_t>(g->layer_comp_cap[il], end / ratio + 2)));
}

/* =========================================================================
 * ROCm Release Graph Allocation.
 * ========================================================================= */

/* Allocate the ROCm graph state for a chosen raw-cache capacity.  The model
 * weights are not copied here; tensors reference the mapped GGUF. */
static bool rocm_graph_alloc_raw_cap(
        ds4_gpu_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer,
        uint32_t                raw_cap,
        uint32_t                ctx_size,
        uint32_t                prefill_cap,
        const ds4_gpu_graph    *batch_workspace) {
    memset(g, 0, sizeof(*g));
    if (raw_cap == 0) raw_cap = 1;
    if (ctx_size == 0) ctx_size = raw_cap;
    if (prefill_cap == 0) prefill_cap = 1;
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > ctx_size) raw_window = ctx_size;
    if (raw_window == 0) raw_window = 1;
    if (raw_cap < raw_window) raw_cap = raw_window;
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;
    g->raw_cap = raw_cap;
    g->raw_window = raw_window;
    g->prefill_cap = prefill_cap;
    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    g->comp_cap = ctx_size / min_ratio + 2u;
    if (g->comp_cap < 2u) g->comp_cap = 2u;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) {
            g->layer_comp_cap[il] = 0;
        } else {
            g->layer_comp_cap[il] = ctx_size / ratio + 2u;
            if (g->layer_comp_cap[il] < 2u) g->layer_comp_cap[il] = 2u;
        }
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const uint64_t group_dim = (uint64_t)DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP);
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t routed_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t vocab_dim = weights->output->dim[1];
    const uint64_t comp_width_max = 2ull * (DS4_N_HEAD_DIM > DS4_N_INDEXER_HEAD_DIM
        ? DS4_N_HEAD_DIM
        : DS4_N_INDEXER_HEAD_DIM);
    const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
    const uint64_t pc = prefill_cap;
    uint64_t kv_cache_bytes = 0;
    const uint64_t context_bytes =
        rocm_graph_context_bytes_for_kv_policy(ctx_size, raw_cap, prefill_cap, &kv_cache_bytes);
    const bool managed_kv_cache =
        ds4_gpu_should_use_managed_kv_cache(kv_cache_bytes, context_bytes) != 0;
    g->managed_kv_cache = managed_kv_cache;
    if (managed_kv_cache) {
        /*
         * Device allocations are fastest, but very large contexts can exhaust
         * Strix Halo unified memory once resident weights and driver allocations
         * are present. Managed memory preserves a demand-paged fallback for this
         * long-lived cache class.
         */
        gufo::server::Logger::LogFormatted(
            gufo::server::LogLevel::kInfo, "ds4",
            "ROCm using managed KV cache for ctx=%u "
            "(maximum KV %.2f GiB, context estimate %.2f GiB); "
            "this may degrade performance but is needed for very large "
            "contexts",
            ctx_size, (double)kv_cache_bytes / 1073741824.0,
            (double)context_bytes / 1073741824.0);
    }

    g->cur_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->flat_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->hc_mix = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    g->hc_mix_ffn_raw = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    g->hc_split = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    g->hc_split_a = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    g->hc_pre_onehot = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    if (g->hc_pre_onehot) ds4_gpu_hc_pre_onehot(g->hc_pre_onehot, 1u);
    g->hc_pre = ds4_gpu_tensor_view(g->hc_split, 0, (uint64_t)DS4_N_HC * sizeof(float));
    g->hc_post = ds4_gpu_tensor_view(g->hc_split,
                                       (uint64_t)DS4_N_HC * sizeof(float),
                                       (uint64_t)DS4_N_HC * sizeof(float));
    g->hc_comb = ds4_gpu_tensor_view(g->hc_split,
                                       2ull * DS4_N_HC * sizeof(float),
                                       (uint64_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    g->attn_cur = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->attn_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->attn_norm_q8 = ds4_gpu_tensor_alloc(ds4_gpu_q8_0_act_bytes(DS4_N_EMBD));
    g->qr = ds4_gpu_tensor_alloc(q_rank * sizeof(float));
    g->qr_norm = ds4_gpu_tensor_alloc(q_rank * sizeof(float));
    g->qr_norm_q8 = ds4_gpu_tensor_alloc(ds4_gpu_q8_0_act_bytes(q_rank));
    g->q = ds4_gpu_tensor_alloc(q_dim * sizeof(float));
    g->kv_raw = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    g->kv = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    bool state_init_ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->layer_raw_cache[il] = rocm_graph_alloc_kv_cache_tensor(
                managed_kv_cache,
                (uint64_t)raw_cap * DS4_N_HEAD_DIM * sizeof(float));
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0) {
          // Reserve the first prompt chunk plus one raw window of decode
          // headroom. The configured limits stay independent of storage.
          const uint64_t initial_comp_cap = std::min<uint64_t>(
              g->layer_comp_cap[il], (pc + DS4_N_SWA) / ratio + 2);
          const uint32_t coff = ratio == 4 ? 2u : 1u;
          const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
          const uint64_t attn_rows = (uint64_t)coff * ratio;
          g->layer_attn_comp_cache[il] = rocm_graph_alloc_kv_cache_tensor(
              managed_kv_cache,
              initial_comp_cap * DS4_N_HEAD_DIM * sizeof(float));
          if (ratio == 4) {
            g->layer_attn_comp_cache_f16[il] = rocm_graph_alloc_kv_cache_tensor(
                managed_kv_cache,
                initial_comp_cap * DS4_N_HEAD_DIM * sizeof(uint16_t));
          }
            g->layer_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            g->layer_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            if (g->layer_attn_state_kv[il]) {
                state_init_ok = state_init_ok &&
                                rocm_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows);
            }
            if (g->layer_attn_state_score[il]) {
                state_init_ok = state_init_ok &&
                                rocm_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF, attn_width * attn_rows);
            }

            if (ratio == 4 ||
                (ds4_layer_is_kv_source(il) &&
                 ds4_layer_is_topk_source(il))) {
                /* V3 indexed layers store their own index keys; V4.1 stores
                 * one index cache per shared group (on the owner layer). */
                const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
                const uint64_t index_rows = (uint64_t)coff * ratio;
                g->layer_index_comp_cache[il] =
                    rocm_graph_alloc_kv_cache_tensor(
                        managed_kv_cache, initial_comp_cap *
                                              DS4_N_INDEXER_HEAD_DIM *
                                              sizeof(float));
                g->layer_index_state_kv[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                g->layer_index_state_score[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                if (g->layer_index_state_kv[il]) {
                    state_init_ok = state_init_ok &&
                                    rocm_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows);
                }
                if (g->layer_index_state_score[il]) {
                    state_init_ok = state_init_ok &&
                                    rocm_tensor_fill_f32(g->layer_index_state_score[il], DS4_NEG_INF, index_width * index_rows);
                }
            }
        }
    }
    g->comp_kv_cur = ds4_gpu_tensor_alloc(comp_width_max * sizeof(float));
    g->comp_sc_cur = ds4_gpu_tensor_alloc(comp_width_max * sizeof(float));
    g->index_kv_cur = ds4_gpu_tensor_alloc(2ull * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
    g->index_sc_cur = ds4_gpu_tensor_alloc(2ull * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
    g->indexer_q = ds4_gpu_tensor_alloc(indexer_q_dim * sizeof(float));
    g->indexer_weights = ds4_gpu_tensor_alloc((uint64_t)DS4_N_INDEXER_HEAD * sizeof(float));
    // Cover the first prefill chunk and single-token decoding at any depth.
    // Later chunks grow this disposable scratch to the actual score shape.
    // A large configured window must not reserve a full-window prefill matrix.
    const uint64_t initial_score_count = std::max<uint64_t>(
        g->comp_cap, std::min<uint64_t>(g->comp_cap, pc / min_ratio + 2) * pc);
    g->indexer_scores =
        ds4_gpu_tensor_alloc(initial_score_count * sizeof(float));
    g->comp_selected = ds4_gpu_tensor_alloc((uint64_t)(DS4_N_INDEXER_TOP_K ? DS4_N_INDEXER_TOP_K : 1u) *
                                              pc * sizeof(uint32_t));
    /* V4.1 source-sharing staging (tiny; single-row decode semantics). */
    g->comp_latent = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    g->shared_topk = ds4_gpu_tensor_alloc((uint64_t)(DS4_N_INDEXER_TOP_K ? DS4_N_INDEXER_TOP_K : 1u) *
                                              sizeof(uint32_t));
    g->cand_mask = ds4_gpu_tensor_alloc((uint64_t)(g->comp_cap ? g->comp_cap : 2u) * sizeof(float));
    g->pf_cand_mask = nullptr;
    g->pf_cand_mask_cap = 0;
    g->pf_cand_n = 0;
    g->pf_cand_valid = false;
    g->pf_sel_owner = UINT32_MAX;
    g->pf_ratio = 1;
    g->heads = ds4_gpu_tensor_alloc(q_dim * sizeof(float));
    g->heads_q8 = ds4_gpu_tensor_alloc(ds4_gpu_q8_0_act_bytes(q_dim));
    g->attn_low = ds4_gpu_tensor_alloc(low_dim * sizeof(float));
    g->attn_low_q8 = ds4_gpu_tensor_alloc(ds4_gpu_q8_0_act_bytes(low_dim));
    g->attn_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->after_attn_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->ffn_cur = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->ffn_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->ffn_norm_q8 = ds4_gpu_tensor_alloc(ds4_gpu_q8_0_act_bytes(DS4_N_EMBD));
    g->ffn_norm_q8k = ds4_gpu_tensor_alloc(ds4_gpu_q8_K_act_bytes(DS4_N_EMBD));
    g->shared_gate = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_up = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_mid = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_mid_q8 = ds4_gpu_tensor_alloc(ds4_gpu_q8_0_act_bytes(shared_dim));
    g->shared_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->router_logits = ds4_gpu_tensor_alloc(DS4_N_EXPERT * sizeof(float));
    g->router_probs = ds4_gpu_tensor_alloc(DS4_N_EXPERT * sizeof(float));
    g->router_selected = ds4_gpu_tensor_alloc(DS4_N_EXPERT_USED * sizeof(int));
    g->router_weights = ds4_gpu_tensor_alloc(DS4_N_EXPERT_USED * sizeof(float));
    g->routed_gate = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_up = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_mid = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_down = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
    g->routed_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->after_ffn_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->output_pre = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->output_weights = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->output_embd = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->output_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->logits = ds4_gpu_tensor_alloc(vocab_dim * sizeof(float));
    if (batch_workspace) {
      if (batch_workspace->prefill_cap < prefill_cap) {
        rocm_graph_free(g);
        return false;
      }
        rocm_graph_bind_batch_workspace(g, batch_workspace);
    } else {
      // These stages run in order, including across grouped requests:
      // HC normalization -> attention Q/heads -> routed MoE -> shared expert.
      // Only routed down survives into the shared-expert stage. Keep it
      // beyond both sets of FFN intermediates, then reuse the other ranges.
      const uint64_t q_bytes = pc * q_dim * sizeof(float);
      const uint64_t hc_bytes = pc * hc_dim * sizeof(float);
      const uint64_t routed_mid_bytes =
          pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float);
      const uint64_t routed_down_bytes =
          pc * DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float);
      const uint64_t shared_mid_bytes = pc * shared_dim * sizeof(float);
      const uint64_t shared_out_bytes = pc * DS4_N_EMBD * sizeof(float);
      const uint64_t down_offset = std::max(
          3 * routed_mid_bytes, 3 * shared_mid_bytes + shared_out_bytes);
      g->batch_stage_scratch = ds4_gpu_tensor_alloc(
          std::max({2 * q_bytes, hc_bytes, down_offset + routed_down_bytes}));
      g->prefill_tokens = ds4_gpu_tensor_alloc(pc * sizeof(int32_t));
      g->batch_cur_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
      if (g_ds4_ple.present) {
        const uint32_t eg_cols = g_ds4_ple.cols_per_module;
        const uint64_t eg_in = (uint64_t)eg_cols * DS4_PLE_EMBD;
        for (uint32_t e = 0; e < g_ds4_ple.n_modules && e < DS4_PLE_MAX_LAYERS;
             ++e) {
          g->engram_ids[e] = ds4_gpu_tensor_alloc(pc * eg_cols * sizeof(int32_t));
          g->engram_rows[e] = ds4_gpu_tensor_alloc(pc * eg_in * sizeof(float));
          g->engram_kv[e] = ds4_gpu_tensor_alloc(
              pc * (DS4_N_HC + 1u) * DS4_N_EMBD * sizeof(float));
          if (!g->engram_ids[e] || !g->engram_rows[e] || !g->engram_kv[e])
            fprintf(stderr, "ds4: engram scratch alloc failed module %u\n", e);
        }
      }
      g->batch_next_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
      g->batch_flat_hc =
          ds4_gpu_tensor_view(g->batch_stage_scratch, 0, hc_bytes);
      g->batch_hc_mix = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
      g->batch_hc_split = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
      g->batch_hc_split_a = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
      g->batch_pre_onehot = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
      if (g->batch_pre_onehot) ds4_gpu_hc_pre_onehot(g->batch_pre_onehot, pc);
      g->batch_attn_cur = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
      g->batch_attn_norm =
          ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
      g->batch_qr = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
      g->batch_qr_norm = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
      g->batch_q = ds4_gpu_tensor_view(g->batch_stage_scratch, 0, q_bytes);
      g->batch_kv_raw =
          ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
      g->batch_kv = ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
      g->batch_comp_kv =
          ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
      g->batch_comp_sc =
          ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
      g->batch_indexer_q =
          ds4_gpu_tensor_alloc(pc * indexer_q_dim * sizeof(float));
      g->batch_indexer_weights =
          ds4_gpu_tensor_alloc(pc * DS4_N_INDEXER_HEAD * sizeof(float));
      // Indexer scoring may borrow this range for F16 keys and queries before
      // attention writes its heads. It is disjoint from live attention Q.
      g->batch_heads =
          ds4_gpu_tensor_view(g->batch_stage_scratch, q_bytes, q_bytes);
      g->batch_attn_low = ds4_gpu_tensor_alloc(pc * low_dim * sizeof(float));
      g->batch_attn_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
      g->batch_group_tmp = ds4_gpu_tensor_alloc(pc * group_dim * sizeof(float));
      g->batch_low_tmp =
          ds4_gpu_tensor_alloc(pc * DS4_N_LORA_O * sizeof(float));
      g->batch_after_attn_hc =
          ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
      g->batch_ffn_cur = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
      g->batch_ffn_norm = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
      g->batch_shared_gate =
          ds4_gpu_tensor_view(g->batch_stage_scratch, 0, shared_mid_bytes);
      g->batch_shared_up = ds4_gpu_tensor_view(
          g->batch_stage_scratch, shared_mid_bytes, shared_mid_bytes);
      g->batch_shared_mid = ds4_gpu_tensor_view(
          g->batch_stage_scratch, 2 * shared_mid_bytes, shared_mid_bytes);
      g->batch_shared_out = ds4_gpu_tensor_view(
          g->batch_stage_scratch, 3 * shared_mid_bytes, shared_out_bytes);
      g->batch_router_logits =
          ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
      g->batch_router_probs =
          ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
      g->batch_router_selected =
          ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(int));
      g->batch_router_weights =
          ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(float));
      g->batch_routed_gate =
          ds4_gpu_tensor_view(g->batch_stage_scratch, 0, routed_mid_bytes);
      g->batch_routed_up = ds4_gpu_tensor_view(
          g->batch_stage_scratch, routed_mid_bytes, routed_mid_bytes);
      g->batch_routed_mid = ds4_gpu_tensor_view(
          g->batch_stage_scratch, 2 * routed_mid_bytes, routed_mid_bytes);
      g->batch_routed_down = ds4_gpu_tensor_view(
          g->batch_stage_scratch, down_offset, routed_down_bytes);
      g->batch_routed_out =
          ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    }

    bool layer_cache_ok = true;
    for (uint32_t il = 0; layer_cache_ok && il < DS4_N_LAYER; il++) {
        layer_cache_ok = g->layer_raw_cache[il] != NULL;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (layer_cache_ok && ratio != 0) {
            layer_cache_ok = g->layer_attn_comp_cache[il] != NULL &&
                             g->layer_attn_state_kv[il] != NULL &&
                             g->layer_attn_state_score[il] != NULL;
        }
        if (layer_cache_ok && ratio == 4) {
            layer_cache_ok = g->layer_attn_comp_cache_f16[il] != NULL &&
                             g->layer_index_comp_cache[il] != NULL &&
                             g->layer_index_state_kv[il] != NULL &&
                             g->layer_index_state_score[il] != NULL;
        }
    }

    const bool ok =
        state_init_ok && layer_cache_ok && g->cur_hc && g->flat_hc &&
        g->hc_mix && g->hc_split && g->hc_pre && g->hc_post && g->hc_comb &&
        g->attn_cur && g->attn_norm && g->attn_norm_q8 && g->qr &&
        g->qr_norm && g->qr_norm_q8 && g->q &&
        g->kv_raw && g->kv && g->comp_kv_cur && g->comp_sc_cur &&
        g->index_kv_cur && g->index_sc_cur &&
        g->indexer_q && g->indexer_weights && g->indexer_scores &&
        g->comp_selected && g->heads && g->heads_q8 && g->attn_low &&
        g->attn_low_q8 &&
        g->attn_out &&
        g->after_attn_hc && g->ffn_cur && g->ffn_norm && g->ffn_norm_q8 &&
        g->ffn_norm_q8k && g->shared_gate &&
        g->shared_up && g->shared_mid && g->shared_mid_q8 && g->shared_out &&
        g->router_logits &&
        g->router_probs && g->router_selected && g->router_weights &&
        g->routed_gate && g->routed_up && g->routed_mid && g->routed_down &&
        g->routed_out && g->after_ffn_hc && g->output_pre &&
        g->output_weights && g->output_embd && g->output_norm && g->logits &&
        g->prefill_tokens && g->batch_stage_scratch && g->batch_cur_hc &&
        g->batch_next_hc && g->batch_flat_hc && g->batch_hc_mix &&
        g->batch_hc_split && g->batch_attn_cur && g->batch_attn_norm &&
        g->batch_qr && g->batch_qr_norm && g->batch_q && g->batch_kv_raw &&
        g->batch_kv && g->batch_comp_kv && g->batch_comp_sc &&
        g->batch_indexer_q && g->batch_indexer_weights && g->batch_heads &&
        g->batch_attn_low && g->batch_attn_out && g->batch_group_tmp &&
        g->batch_low_tmp && g->batch_after_attn_hc && g->batch_ffn_cur &&
        g->batch_ffn_norm && g->batch_shared_gate && g->batch_shared_up &&
        g->batch_shared_mid && g->batch_shared_out && g->batch_router_logits &&
        g->batch_router_probs && g->batch_router_selected &&
        g->batch_router_weights && g->batch_routed_gate && g->batch_routed_up &&
        g->batch_routed_mid && g->batch_routed_down && g->batch_routed_out;
    if (!ok) rocm_graph_free(g);
    return ok;
}

static uint32_t rocm_graph_raw_span_for_batch(
        const ds4_gpu_graph *g,
        uint32_t               pos0,
        uint32_t               n_tokens) {
    if (!g || g->raw_cap == 0 || n_tokens == 0) return 0;

    const uint32_t window = g->raw_window ? g->raw_window : DS4_N_SWA;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    uint64_t needed = (uint64_t)n_tokens;
    if (window != 0) {
        needed += n_tokens == 1 ? (uint64_t)window - 1u : (uint64_t)window;
    }
    uint64_t available = (uint64_t)last_pos + 1u;
    if (needed > available) needed = available;
    if (needed > g->raw_cap) needed = g->raw_cap;
    return (uint32_t)needed;
}

/* A/B diagnostic: drop every compressed-KV contribution from attention so
 * the model runs pure sliding-window.  Used to isolate whether the
 * constant-token loop originates in the compressed path or the core stack. */
static bool ds4_no_comp_attn_env(void) {
  static const int on = getenv("GUFO_DS4_NO_COMP_ATTN") ? 1 : 0;
  return on == 1;
}

/* A/B gate for the V4.1 sparse-prefill source-sharing selection (f9b6f03).
 * GUFO_DS4_PF_SEL=0 skips the prefill indexer publisher and the reader
 * match, so the batched prefill FA falls back to dense compressed attention
 * over all owner rows (pre-selection behavior).  Default: enabled. */
static bool ds4_pf_sel_env(void) {
  static const bool off = getenv("GUFO_DS4_PF_SEL") &&
                          atoi(getenv("GUFO_DS4_PF_SEL")) == 0;
  return !off;
}

static uint32_t rocm_graph_raw_start_for_span(
        const ds4_gpu_graph *g,
        uint32_t               last_pos,
        uint32_t               n_raw) {
    if (!g || g->raw_cap == 0 || n_raw == 0) return 0;
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    return first_raw_pos % g->raw_cap;
}

/* =========================================================================
 * ROCm Decode Helpers.
 * =========================================================================
 *
 * The generation path uses the qualified fused helpers below.
 */

static bool rocm_graph_decode_kv_store(ds4_gpu_tensor* kv,
                                       ds4_gpu_tensor* raw_cache,
                                       uint32_t raw_cap, uint32_t raw_row) {
  return ds4_gpu_kv_fp8_store_raw_tensor(kv, raw_cache, raw_cap, raw_row,
                                         DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
}

/* V4.1 weights arrive BF16; the load step converts them to device F16, so
 * graph validation accepts both spellings and reads the F16 copy. */
static bool ds4_graph_f16_weight_type(uint32_t type) {
    return type == DS4_TENSOR_F16 || type == DS4_TENSOR_BF16;
}
static bool rocm_graph_matmul_hc_fn_tensor(
        ds4_gpu_tensor *out, const ds4_model *model, const ds4_tensor *w,
        uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint64_t n_tok);

/*
 * DeepSeek V4.1 emits compressed-KV rows only on kv_source (producer) layers;
 * every other compressed layer (consumer) reads the nearest producer's cache.
 * APE tables and the compressor gate projection are absent in V4.1 for
 * ratio-1 producers: absent weights pass offset 0 (the kernel no-APE
 * sentinel) or are skipped.  Weight matmuls dispatch on the tensor's stored
 * type so BF16 weights are read as-is and widened in-kernel.
 */
static inline uint32_t rocm_graph_comp_src(uint32_t il) {
    return ds4_layer_kv_source(il);
}
static inline uint64_t rocm_graph_opt_w_off(const ds4_tensor *t) {
    return t ? t->abs_offset : 0ull;
}
static inline uint32_t rocm_graph_opt_w_type(const ds4_tensor *t) {
    return t ? t->type : 0u;
}
static bool rocm_graph_matmul_weight_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!out || !w || !x) return false;
    if (w->type == DS4_TENSOR_F16)
        return ds4_gpu_matmul_f16_tensor(out, model->map, model->size,
                                         w->abs_offset, in_dim, out_dim, x,
                                         n_tok) != 0;
    if (w->type == DS4_TENSOR_BF16)
        return ds4_gpu_matmul_bf16_tensor(out, model->map, model->size,
                                          w->abs_offset, in_dim, out_dim, x,
                                          n_tok) != 0;
    if (w->type == DS4_TENSOR_F32)
        return ds4_gpu_matmul_f32_tensor(out, model->map, model->size,
                                         w->abs_offset, in_dim, out_dim, x,
                                         n_tok) != 0;
    fprintf(stderr, "ds4: unsupported projection weight type %s\n",
            ds4_tensor_type_name(w->type));
    return false;
}

static uint32_t rocm_graph_attn_comp_cache_is_f16(void) {
    return 0;
}

static bool rocm_graph_use_attn_comp_mirror(
        const ds4_gpu_graph *g,
        uint32_t             il,
        uint32_t             n_tokens) {
    return n_tokens >= 128u &&
           g->layer_attn_comp_cache_f16[rocm_graph_comp_src(il)] != nullptr;
}

static ds4_gpu_tensor *rocm_graph_prefill_attn_comp_cache(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       n_tokens) {
    const uint32_t comp_src = rocm_graph_comp_src(il);
    return rocm_graph_use_attn_comp_mirror(g, il, n_tokens)
        ? g->layer_attn_comp_cache_f16[comp_src]
        : g->layer_attn_comp_cache[comp_src];
}

static uint32_t rocm_graph_prefill_attn_comp_cache_is_f16(
        const ds4_gpu_graph *g,
        uint32_t             il,
        uint32_t             n_tokens) {
    return rocm_graph_use_attn_comp_mirror(g, il, n_tokens) ? 1u : 0u;
}

static ds4_gpu_tensor *rocm_graph_attn_comp_update_target(
        ds4_gpu_graph *g,
        uint32_t       il) {
    return g->layer_attn_comp_cache[il];
}

static uint32_t rocm_graph_attn_comp_update_row(uint32_t row) {
    return row;
}

static ds4_gpu_tensor *rocm_graph_attn_comp_row_view(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       row) {
    return ds4_gpu_tensor_view(g->layer_attn_comp_cache[il],
                               (uint64_t)row * DS4_N_HEAD_DIM * sizeof(float),
                               (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
}

static ds4_gpu_tensor *rocm_graph_attn_comp_mirror_row_view(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       row) {
    if (!g->layer_attn_comp_cache_f16[il]) return nullptr;
    return ds4_gpu_tensor_view(
        g->layer_attn_comp_cache_f16[il],
        (uint64_t)row * DS4_N_HEAD_DIM * sizeof(uint16_t),
        (uint64_t)DS4_N_HEAD_DIM * sizeof(uint16_t));
}

static ds4_gpu_tensor *rocm_graph_attn_comp_prefill_target(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    const uint32_t view_rows = rows ? rows : 1u;
    return ds4_gpu_tensor_view(g->layer_attn_comp_cache[il],
                               (uint64_t)first_row * DS4_N_HEAD_DIM * sizeof(float),
                               (uint64_t)view_rows * DS4_N_HEAD_DIM * sizeof(float));
}

static ds4_gpu_tensor *rocm_graph_attn_comp_prefill_mirror(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    if (!g->layer_attn_comp_cache_f16[il]) return nullptr;
    const uint32_t view_rows = rows ? rows : 1u;
    return ds4_gpu_tensor_view(
        g->layer_attn_comp_cache_f16[il],
        (uint64_t)first_row * DS4_N_HEAD_DIM * sizeof(uint16_t),
        (uint64_t)view_rows * DS4_N_HEAD_DIM * sizeof(uint16_t));
}

static void rocm_graph_attn_comp_prefill_target_free(ds4_gpu_tensor *t) {
    ds4_gpu_tensor_free(t);
}

static bool rocm_graph_quantize_attn_comp_row(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       row,
        ds4_gpu_tensor *comp_row_f32) {
    ds4_gpu_tensor *mirror_row =
        rocm_graph_attn_comp_mirror_row_view(g, il, row);
    const bool ok = mirror_row
        ? ds4_gpu_dsv4_fp8_kv_quantize_mirror_f16_tensor(
              comp_row_f32, mirror_row, 1u, DS4_N_HEAD_DIM, DS4_N_ROT) != 0
        : ds4_gpu_dsv4_fp8_kv_quantize_tensor(
              comp_row_f32, 1u, DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
    ds4_gpu_tensor_free(mirror_row);
    return ok;
}

static bool rocm_graph_rebuild_attn_comp_mirror(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       rows) {
    if (!g->layer_attn_comp_cache_f16[il] || rows == 0u) return true;
    return ds4_gpu_tensor_convert_f32_to_f16(
               g->layer_attn_comp_cache_f16[il],
               g->layer_attn_comp_cache[il],
               (uint64_t)rows * DS4_N_HEAD_DIM) != 0;
}

/* Encode one DS4 decode layer on ROCm.  This is the release single-token
 * layer path; diagnostics reuse it so they compare exactly what generation
 * runs. */
static bool rocm_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0);
static bool rocm_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0);
static bool rocm_graph_matmul_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

static bool rocm_graph_reserve_indexer_scores(ds4_gpu_graph* g, uint32_t n_comp,
                                              uint32_t n_tokens) {
  if (n_comp == 0 || n_comp > g->comp_cap || n_tokens == 0 ||
      n_tokens > g->prefill_cap) {
    return false;
  }
  const uint64_t required =
      static_cast<uint64_t>(n_comp) * n_tokens * sizeof(float);
  const uint64_t capacity = ds4_gpu_tensor_bytes(g->indexer_scores);
  if (required <= capacity)
    return true;
  const uint64_t maximum =
      static_cast<uint64_t>(g->comp_cap) * g->prefill_cap * sizeof(float);
  auto* replacement =
      ds4_gpu_tensor_alloc(std::min(maximum, std::max(required, capacity * 2)));
  if (replacement == nullptr)
    return false;
  // Scores are overwritten before use; KV state and snapshots never refer
  // to this allocation. Geometric growth avoids reallocating each chunk.
  ds4_gpu_tensor_free(g->indexer_scores);
  g->indexer_scores = replacement;
  return true;
}

/* V4.1 prefill: grow the per-token candidate mask scratch (n_tokens x
 * n_comp f32) and the pre-RoPE latent staging for a whole chunk. */
static bool rocm_graph_reserve_pf_cand_mask(ds4_gpu_graph *g, uint64_t elems) {
    if (elems == 0) return false;
    if (g->pf_cand_mask && g->pf_cand_mask_cap >= elems * sizeof(float))
        return true;
    ds4_gpu_tensor *replacement = ds4_gpu_tensor_alloc(elems * sizeof(float));
    if (!replacement) return false;
    ds4_gpu_tensor_free(g->pf_cand_mask);
    g->pf_cand_mask = replacement;
    g->pf_cand_mask_cap = elems * sizeof(float);
    return true;
}

static bool rocm_graph_reserve_comp_latent(ds4_gpu_graph *g, uint64_t elems) {
    if (elems == 0) return false;
    if (g->comp_latent &&
        ds4_gpu_tensor_bytes(g->comp_latent) >= elems * sizeof(float))
        return true;
    ds4_gpu_tensor *replacement = ds4_gpu_tensor_alloc(elems * sizeof(float));
    if (!replacement) return false;
    ds4_gpu_tensor_free(g->comp_latent);
    g->comp_latent = replacement;
    return true;
}

/* V4.1 prefill owner: build n_new shared index-K rows from the staged
 * pre-RoPE latent (attn_k -> k_norm -> stride rope -> QAT), mirroring the
 * decode owner block.  Row i carries logical position
 * pos0 + ratio - 1 + i*ratio (window-end position of pooled row i), so it
 * lines up exactly with the rope'd compressed KV rows of this chunk. */
static bool rocm_graph_v41_owner_index_rows(ds4_gpu_graph *g,
                                            const ds4_model *model,
                                            const ds4_layer_weights *layer,
                                            uint32_t il, uint32_t pos0,
                                            uint32_t ratio, uint32_t n_new,
                                            bool compressed, float freq_base,
                                            float freq_scale, float ext_factor,
                                            float attn_factor) {
    if (n_new == 0u) return true;
    if (!layer->indexer_attn_k || !layer->indexer_k_norm ||
        !g->layer_index_comp_cache[il] || g->comp_latent == nullptr) {
        fprintf(stderr,
                "ds4: ROCm V4.1 prefill index-k weights/cache missing at "
                "layer %u\n", il);
        return false;
    }
    if (g->layer_n_index_comp[il] + n_new > g->layer_comp_cap[il]) {
        fprintf(stderr,
                "ds4: ROCm V4.1 prefill index cache capacity exceeded at "
                "layer %u\n", il);
        return false;
    }
    if (!rocm_graph_reserve_comp_latent(g, (uint64_t)n_new * DS4_N_HEAD_DIM))
        return false;
    ds4_gpu_tensor *krow = ds4_gpu_tensor_view(
        g->layer_index_comp_cache[il],
        (uint64_t)g->layer_n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM *
            sizeof(float),
        (uint64_t)n_new * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *latent = ds4_gpu_tensor_view(
        g->comp_latent, 0, (uint64_t)n_new * DS4_N_HEAD_DIM * sizeof(float));
    bool ok = krow != nullptr && latent != nullptr;
    if (ok)
        ok = rocm_graph_matmul_weight_tensor(
                 krow, model, layer->indexer_attn_k, DS4_N_HEAD_DIM,
                 DS4_N_INDEXER_HEAD_DIM, latent, n_new);
    if (ok)
        ok = ds4_gpu_rms_norm_weight_rows_tensor(
                 krow, krow, model->map, model->size,
                 layer->indexer_k_norm->abs_offset, DS4_N_INDEXER_HEAD_DIM,
                 n_new, DS4_RMS_EPS) != 0;
    if (ok)
        ok = ds4_gpu_rope_tail_stride_tensor(
                 krow, n_new, 1u, DS4_N_INDEXER_HEAD_DIM, DS4_N_ROT,
                 pos0 + ratio - 1u, ratio,
                 compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, false,
                 freq_base, freq_scale, ext_factor, attn_factor,
                 DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok)
        ok = ds4_gpu_dsv4_indexer_qat_tensor(krow, n_new,
                                             DS4_N_INDEXER_HEAD_DIM) != 0;
    if (krow) ds4_gpu_tensor_free(krow);
    if (latent) ds4_gpu_tensor_free(latent);
    if (ok) g->layer_n_index_comp[il] += n_new;
    return ok;
}

static bool rocm_graph_matmul_v41_dense_tensor(
        ds4_gpu_tensor *out, const ds4_model *model, const ds4_tensor *w,
        uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint64_t n_tok);

// ---------------------------------------------------------------------------
// Engram (D3): n-gram conditioned memory applied at the Engram PLE layers
// (artifact: blk.1 / blk.14).  Host side keeps the token history, computes the
// two module head-id rows exactly as llama.cpp b11870 deepseek41.cpp does, and
// stages them into per-graph input buffers; the captured kernels gather the
// IQ1_S table rows (host-resident, read AS IS — never uploaded), project them
// through the Q5_K engram_kv matrix and fold value*gate back into the HC
// stream with the sign-sqrt gate.
// ---------------------------------------------------------------------------
static int32_t rocm_engram_class_of(const ds4_ple_state &p, int t) {
  if (t < 0 || (size_t)t >= p.token_map.size())
    return (int32_t)DS4_PLE_DEAD;
  return p.token_map[(size_t)t];
}

static void rocm_engram_hash_rows(const std::vector<int32_t> &hist,
                                  uint32_t start, uint32_t n, uint32_t e,
                                  std::vector<int32_t> &out) {
  const ds4_ple_state &p = g_ds4_ple;
  const uint32_t cols = p.cols_per_module;
  const uint32_t ng = p.ngram;
  out.resize((size_t)n * cols);
  for (size_t i = 0; i < n; ++i) {
    const uint32_t pos = start + (uint32_t)i;
    // Mirror llama.cpp engram_hash: rolling folds one n-gram order per step
    // and each order's head group uses the rolling accumulated so far, all in
    // uint64 arithmetic (multipliers are 64-bit).
    uint64_t toks[8];
    bool blocked = false;
    int32_t c0 = rocm_engram_class_of(p, hist[pos]);
    blocked = c0 == (int32_t)DS4_PLE_DEAD;
    toks[0] = (uint64_t)(uint32_t)(blocked ? p.pad_class : c0);
    for (uint32_t s = 1; s < ng && s < 8u; ++s) {
      int32_t c = pos < s ? (int32_t)DS4_PLE_DEAD
                          : rocm_engram_class_of(p, hist[pos - s]);
      blocked = blocked || c == (int32_t)DS4_PLE_DEAD;
      toks[s] = (uint64_t)(uint32_t)(blocked ? p.pad_class : c);
    }
    // Mirror llama.cpp engram_hash: rolling folds one n-gram order per step
    // and each order's head group uses the rolling accumulated so far, all in
    // uint64 arithmetic (multipliers are 64-bit). Verified against the
    // official DeepSeek-V4.1 inference/engram.py, which folds products[...,i]
    // BEFORE taking the modulo for that order.
    uint64_t rolling = toks[0] * p.multipliers[e * p.ngram + 0u];
    for (uint32_t o = 1; o < ng && o < 8u; ++o) {
      rolling ^= toks[o] * p.multipliers[e * p.ngram + o];
      for (uint32_t h = 0; h < p.heads_per_ngram; ++h) {
        const uint32_t col = (o - 1u) * p.heads_per_ngram + h;
        if (col >= cols)
          continue;
        out[i * cols + col] =
            (int32_t)(rolling % (uint64_t)p.primes[e * cols + col]) +
            p.offsets[e * cols + col];
      }
    }
  }
}

// IQ4_NL code values of this GGML fork (ggml-common.h kvalues_iq4nl).
static const int8_t g_ds4_iq4nl_vals[16] = {-127, -104, -83, -65, -49, -35,
                                            -22,  -10,  1,   13,
                                            25,   38,   53,  69,
                                            89,   113};

static inline float ds4_h2f(uint16_t v) {
  const uint32_t sign = (uint32_t)(v >> 15) & 1u;
  const uint32_t exp = (uint32_t)(v >> 10) & 0x1fu;
  const uint32_t man = (uint32_t)v & 0x3ffu;
  uint32_t f;
  if (exp == 0u)
    return (sign ? -1.0f : 1.0f) * (float)man * 5.96046448e-8f;
  if (exp == 31u)
    return man ? NAN : (sign ? -INFINITY : INFINITY);
  f = (sign << 31) | ((exp + 112u) << 23) | (man << 13);
  float out;
  __builtin_memcpy(&out, &f, 4);
  return out;
}

// The Engram table is a ~110 GB IQ4_NL file region — far too big for any
// device mapping (the driver refuses: "SVM mapping failed, exceeds resident
// system memory limit"). Rows are random-access anyway, so gather them on
// the host straight from the mmap (AS IS, no copies kept) and upload only
// the few MB the current step needs.
static bool rocm_engram_gather_rows_host(const void *map,
                                         const std::vector<int32_t> &ids,
                                         std::vector<float> &out) {
  const uint8_t *tbl = (const uint8_t *)map + g_ds4_engram_table_off;
  out.assign((size_t)ids.size() * 256u, 0.0f);
  for (size_t r = 0; r < ids.size(); ++r) {
    const uint8_t *p = tbl + (size_t)ids[r] * 144u;
    float *d = out.data() + r * 256u;
    for (int b = 0; b < 8; ++b) {
      uint16_t hb;
      __builtin_memcpy(&hb, p + b * 18, 2);
      const float sc = ds4_h2f(hb);
      for (int q = 0; q < 16; ++q) {
        const uint8_t by = p[b * 18 + 2 + q];
        d[b * 32 + q] = sc * (float)g_ds4_iq4nl_vals[by & 15u];
        d[b * 32 + 16 + q] = sc * (float)g_ds4_iq4nl_vals[by >> 4];
      }
    }
  }
  return true;
}

// Update the token history for tokens[pos..pos+n), stage head ids and the
// host-gathered n-gram rows for every module.
static bool rocm_graph_engram_prepare(ds4_gpu_graph *g, const void *map,
                                      const int32_t *toks, uint32_t n,
                                      uint32_t pos) {
  if (!g_ds4_ple.present || !g->engram_ids[0])
    return true;
  if (g->token_hist.size() > pos)
    g->token_hist.resize(pos);
  while (g->token_hist.size() < pos)
    g->token_hist.push_back(0);
  g->token_hist.insert(g->token_hist.end(), toks, toks + n);
  for (uint32_t e = 0; e < g_ds4_ple.n_modules && e < DS4_PLE_MAX_LAYERS; ++e) {
    if (!g->engram_ids[e])
      continue;
    const char *only = getenv("DS4_ENGRAM_ONLY_MOD");
    if (only && (int)e != atoi(only))
      continue;
    std::vector<int32_t> ids;
    rocm_engram_hash_rows(g->token_hist, pos, n, e, ids);
    const int64_t table_rows = (int64_t)(g_ds4_engram_table_bytes / 144u);
    for (int32_t id : ids) {
      if (id < 0 || (int64_t)id >= table_rows) {
        fprintf(stderr, "ds4: engram id %d out of table rows %lld module %u\n",
                id, (long long)table_rows, e);
        return false;
      }
    }
    if (getenv("DS4_ENGRAM_DEBUG")) {
      fprintf(stderr, "ds4: engram ids mod=%u n=%u first=", e, n);
      for (uint32_t i = 0; i < ids.size(); ++i)
        fprintf(stderr, "%d ", ids[i]);
      int32_t mn = ids[0], mx = ids[0];
      for (int32_t id : ids) { if (id < mn) mn = id; if (id > mx) mx = id; }
      fprintf(stderr, "| min=%d max=%d\n", mn, mx);
    }
    std::vector<float> rows;
    if (!rocm_engram_gather_rows_host(map, ids, rows))
      return false;
    if (!ds4_gpu_tensor_write(g->engram_rows[e], 0, rows.data(),
                              rows.size() * sizeof(float))) {
      fprintf(stderr, "ds4: engram rows upload failed module=%u n=%u\n", e, n);
      return false;
    }
  }
  return true;
}

// Gather + project + gate for the Engram module owned by layer `il`.
/* Dump an entire tensor to file for layer-0 golden HC comparison. */
static void ds4_dbg_dump(const ds4_gpu_tensor *t, const char *path,
                         uint64_t count) {
  if (!getenv("DS4_GOLDEN0") || !t) return;
  ds4_gpu_synchronize();
  std::vector<float> buf(count);
  if (!ds4_gpu_tensor_read(t, 0, buf.data(), count * sizeof(float))) {
    fprintf(stderr, "ds4: g0 dump fail %s\n", path);
    return;
  }
  FILE *fp = fopen(path, "wb");
  if (!fp) return;
  fwrite(buf.data(), 4, count, fp);
  fclose(fp);
}

static bool rocm_graph_engram_at_layer(ds4_gpu_graph *g, const ds4_model *model,
                                       const ds4_layer_weights *layer,
                                       uint32_t il, ds4_gpu_tensor *hc,
                                       uint32_t nt) {
  if (!g_ds4_ple.present)
    return true;
  const int e = ds4_ple_module_for_layer(il);
  {
    const char *only = getenv("DS4_ENGRAM_ONLY_MOD");
    if (only && e >= 0 && e != atoi(only))
      return true;
  }
  if (e < 0 || !g->engram_rows[e] || !g->engram_kv[e] || nt == 0 ||
      nt > g->prefill_cap)
    return e < 0;
  const ds4_tensor *ekv = layer->engram_kv;
  const ds4_tensor *eg = layer->engram_gate;
  if (!ekv || !eg)
    return false;
  const uint32_t cols = g_ds4_ple.cols_per_module;
  const uint32_t in_dim = cols * DS4_PLE_EMBD;
  const uint32_t out_dim = (DS4_N_HC + 1u) * DS4_N_EMBD;
  const int no_kv = getenv("DS4_ENGRAM_NO_KV") != NULL;
  const int no_gate = getenv("DS4_ENGRAM_NO_GATE") != NULL;
  static int logged[DS4_PLE_MAX_LAYERS] = {0};
  const uint64_t kv_row_stride =
      ekv->dim[0] ? ekv->bytes / ekv->dim[0] : (uint64_t)in_dim * sizeof(int8_t);
  if (!logged[e]) {
    logged[e] = 1;
    fprintf(stderr,
            "ds4: engram mod=%u layer=%u table_off=%llu tbl[type=%u bytes=%llu] "
            "kv[type=%u dim=%llu,%llu bytes=%llu row_stride=%llu] "
            "gate[bytes=%llu] in=%u out=%u cap_ids=%llu\n",
            e, il, (unsigned long long)g_ds4_engram_table_off, ekv->type,
            (unsigned long long)g_ds4_engram_table_bytes, (unsigned long long)ekv->type,
            (unsigned long long)ekv->dim[0], (unsigned long long)ekv->dim[1],
            (unsigned long long)ekv->bytes, (unsigned long long)kv_row_stride,
            (unsigned long long)eg->bytes, in_dim, out_dim,
            (unsigned long long)ds4_gpu_tensor_bytes(g->engram_ids[e]));
  }
  ds4_gpu_tensor *v_rows = ds4_gpu_tensor_view(
      g->engram_rows[e], 0, (uint64_t)nt * in_dim * sizeof(float));
  ds4_gpu_tensor *v_kv = ds4_gpu_tensor_view(
      g->engram_kv[e], 0, (uint64_t)nt * out_dim * sizeof(float));
  bool ok = true;
  if (!v_rows || !v_kv)
    return false;
  const char *kv_p = (const char *)ds4_gpu_model_range_ptr(
      model->map, ekv->abs_offset, ekv->bytes, "engram kv");
  const char *gate_p = (const char *)ds4_gpu_model_range_ptr(
      model->map, eg->abs_offset, eg->bytes, "engram gate");
  if (!kv_p || !gate_p) {
    fprintf(stderr, "ds4: engram range resolve failed mod=%u\n", e);
    return false;
  }
  // Rows were host-gathered by rocm_graph_engram_prepare for this step.
  static uint64_t g_eg_dumped = 0;  // bitmask: capture only the first (prefill) call per layer
  static uint32_t g_eg_seq[64] = {0};
  const char* const eg_sfx = getenv("DS4_ENGRAM_DUMP_ALL") ? "_req" : "";
  char eg_seqstr[16] = "";
  const bool eg_dump = getenv("DS4_ENGRAM_DEBUG") && il < 64 &&
                       (getenv("DS4_ENGRAM_DUMP_ALL") || !((g_eg_dumped >> il) & 1u));
  if (eg_dump) {
    g_eg_dumped |= 1ull << il;
    if (eg_sfx[0]) {
      if (il < 64) g_eg_seq[il]++;
      snprintf(eg_seqstr, sizeof eg_seqstr, "%u", g_eg_seq[il]);
    }
  }
  if (eg_dump) {
    char nm[128];
    snprintf(nm, sizeof nm, "/tmp/eg_rows_L%u%s%s.bin", il, eg_sfx, eg_seqstr);
    ds4_dbg_dump(v_rows, nm, (uint64_t)nt * in_dim);
    snprintf(nm, sizeof nm, "/tmp/eg_pre_L%u%s%s.bin", il, eg_sfx, eg_seqstr);
    ds4_dbg_dump(hc, nm, (uint64_t)nt * (uint64_t)DS4_N_HC * DS4_N_EMBD);
    snprintf(nm, sizeof nm, "/tmp/eg_gatew_L%u.bin", il);
    FILE *fp = fopen(nm, "wb");
    if (fp) {
      fwrite(gate_p, 4, (size_t)DS4_N_HC * DS4_N_EMBD, fp);
      fclose(fp);
    }
  }
  if (!no_kv)
    ok = ds4_gpu_engram_kv(v_rows, kv_p, v_kv, nt, out_dim,
                           in_dim, in_dim / 256u) != 0 &&
         ds4_gpu_probe_sync("engram kv") != 0;
  if (eg_dump && !no_kv) {
    char nm[128];
    snprintf(nm, sizeof nm, "/tmp/eg_kv_L%u%s%s.bin", il, eg_sfx, eg_seqstr);
    ds4_dbg_dump(v_kv, nm, (uint64_t)nt * out_dim);
  }
  if (ok && !no_gate)
    ok = ds4_gpu_engram_gate(hc, v_kv, gate_p, nt, DS4_N_EMBD,
                             DS4_N_HC, (uint64_t)DS4_N_HC * DS4_N_EMBD,
                             out_dim, DS4_RMS_EPS) != 0 &&
         ds4_gpu_probe_sync("engram gate") != 0;
  if (eg_dump) {
    char nm[128];
    snprintf(nm, sizeof nm, "/tmp/eg_post_L%u%s%s.bin", il, eg_sfx, eg_seqstr);
    ds4_dbg_dump(hc, nm, (uint64_t)nt * (uint64_t)DS4_N_HC * DS4_N_EMBD);
  }
  ds4_gpu_tensor_free(v_rows);
  ds4_gpu_tensor_free(v_kv);
  return ok;
}

/* Layer-0 golden bisection: dump first values of decode attention nodes. */
static uint32_t g_ds4_g0_pos = 0u;
static void ds4_dbg_vals(const ds4_gpu_tensor *t, const char *tag,
                         uint64_t off, uint32_t n) {
  float buf[6];
  if (!getenv("DS4_GOLDEN0") || !t || n > 6u) return;
  ds4_gpu_synchronize();
  if (!ds4_gpu_readback_f32(t, off, n, buf)) {
    fprintf(stderr, "ds4: g0[%s] readback fail\n", tag);
    return;
  }
  fprintf(stderr, "ds4: g0[%s@%u]", tag, g_ds4_g0_pos);
  for (uint32_t i = 0; i < n; i++) fprintf(stderr, " %.7f", buf[i]);
  fprintf(stderr, "\n");
}


/* DS4_RMS_PROBE=1: per-layer row statistics to bisect where the residual
 * stream diverges. Appends rms/min/max/argmax lines to /tmp/ds4_rms.log. */
static void ds4_probe_stats(const ds4_gpu_tensor *t, uint64_t off_elems,
                            uint64_t count, const char *tag, uint32_t il) {
  static const bool on = getenv("DS4_RMS_PROBE") != NULL;
  if (!on || !t || count == 0) return;
  ds4_gpu_synchronize();
  std::vector<float> buf(count);
  if (!ds4_gpu_tensor_read(t, off_elems * sizeof(float), buf.data(),
                           count * sizeof(float))) {
    return;
  }
  double ss = 0.0, mx = -1e300, mn = 1e300;
  uint64_t am = 0, nan = 0;
  for (uint64_t i = 0; i < count; i++) {
    float v = buf[i];
    if (!(v == v)) { nan++; continue; }
    ss += (double)v * (double)v;
    if (v > mx) { mx = v; am = i; }
    if (v < mn) mn = v;
  }
  const uint64_t good = count - nan;
  FILE *fp = fopen("/tmp/ds4_rms.log", "a");
  if (!fp) return;
  fprintf(fp, "%-12s L%02u n=%llu rms=%.6g min=%.4g max=%.4g argmax=%llu nan=%llu\n",
          tag, il, (unsigned long long)count,
          good ? sqrt(ss / (double)good) : 0.0,
          nan == count ? 0.0 : mn, nan == count ? 0.0 : mx,
          (unsigned long long)am, (unsigned long long)nan);
  fclose(fp);
}

static bool rocm_graph_encode_decode_layer(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos,
        ds4_gpu_tensor       *raw_cache,
        uint32_t                raw_cap,
        uint32_t                raw_row,
        uint32_t                n_raw,
        int                     token) {
  if (!rocm_graph_reserve_compressed_position(g, il,
                                              static_cast<uint64_t>(pos) + 1))
    return false;
  const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
  const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
  if (!rocm_graph_engram_at_layer(g, model, layer, il, g->cur_hc, 1))
    return false;
  ds4_probe_stats(g->cur_hc, 0, (uint64_t)DS4_N_HC * DS4_N_EMBD, "dec_in", il);
  const uint64_t q_rank = layer->attn_q_a->dim[1];
  const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
  const uint32_t n_groups = DS4_N_OUT_GROUP;
  const uint32_t group_heads = DS4_N_HEAD / n_groups;
  const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
  const uint32_t rank = DS4_N_LORA_O;
  const uint32_t shared_dim = (uint32_t)layer->ffn_gate_shexp->dim[1];
  const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
  const uint64_t expert_mid_dim = layer->ffn_gate_exps->dim[1];
  const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
  const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
  const bool compressed = ds4_layer_compress_ratio(il) != 0;
  const float freq_base = layer_rope_freq_base(il);
  const float freq_scale = layer_rope_freq_scale(il);
  const float ext_factor =
      compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
  float attn_factor = 1.0f;
  if (ext_factor != 0.0f && freq_scale > 0.0f) {
    attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
  }

    bool ok = true;
    const bool decode_stage_profile = getenv("GUFO_DEEPSEEK_ROCM_DECODE_STAGE_PROFILE") != NULL;
    double decode_stage_t0 = decode_stage_profile ? ds4_now_seconds() : 0.0;
#define GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE(name) do { \
        if (ok && decode_stage_profile) { \
            ok = rocm_graph_layer_stage_profile_boundary("decode", (name), il, pos, 1, &decode_stage_t0); \
        } \
        if (!ok) fprintf(stderr, "ds4: decode stage FAILED: %s layer=%u pos=%u\n", \
                         (name), il, pos); \
    } while (0)
    if ((il == 0u || il == 1u) && pos <= 1u) {
      char tag[48];
      snprintf(tag, sizeof tag, "l%u_attn_norm", il);
      ds4_dbg_vals(g->attn_norm, tag, 0, 6u);
      ds4_dbg_vals(g->q, il == 0u ? "q" : "l1_q", 0, 6u);
      ds4_dbg_vals(g->kv, il == 0u ? "kv" : "l1_kv", 0, 6u);
    }
    if (il == 1u && pos <= 1u) {
      ds4_dbg_dump(g->cur_hc, "/tmp/l1_x.bin", 20480u);
      ds4_dbg_dump(g->hc_post, "/tmp/l1_post.bin", 4u);
    }
    if (il == 1u && pos == 0u) ds4_dbg_dump(g->cur_hc, "/tmp/g0_x_l1.bin", hc_dim);
    if (ok) ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->cur_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (!ok) fprintf(stderr, "ds4: decode pre-macro rms failed layer=%u\n", il);
    if (ok) ok = rocm_graph_matmul_plain_tensor(g->hc_mix, model, layer->hc_attn_fn,
                                                 hc_dim, mix_hc, g->flat_hc, 1);
    if (!ok) fprintf(stderr, "ds4: decode pre-macro hc_mix matmul failed layer=%u type=%u\n",
                     il, layer->hc_attn_fn ? layer->hc_attn_fn->type : 999u);
    if (ok) {
      ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(
               g->attn_cur, g->attn_norm, g->attn_norm_q8, NULL, g->hc_split_a,
               g->hc_mix, g->cur_hc,
               model->map, model->size, layer->hc_attn_scale->abs_offset,
               layer->hc_attn_base->abs_offset, layer->attn_norm->abs_offset,
               DS4_N_EMBD, DS4_N_HC, DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS,
               DS4_RMS_EPS,
               (il == 0u ? g->hc_pre_onehot : g->hc_split), 24u) != 0;
      if (!ok) fprintf(stderr, "ds4: decode pre-macro hc_split fused failed layer=%u\n", il);
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("attn_hc_pre");
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("attn_norm");
    if (ok && layer->attn_q_a->type == DS4_TENSOR_Q8_1) {
        /* V4.1 dense tier: Q8_1 low-rank projections, decomposed ops. */
        ok = ds4_gpu_matmul_q8_1_tensor(g->qr, model->map, model->size,
                                        layer->attn_q_a->abs_offset, DS4_N_EMBD,
                                        q_rank, g->attn_norm, 1u) != 0;
        if (ok)
            ok = ds4_gpu_matmul_q8_1_tensor(
                     g->kv_raw, model->map, model->size,
                     layer->attn_kv->abs_offset, DS4_N_EMBD, DS4_N_HEAD_DIM,
                     g->attn_norm, 1u) != 0;
        if (ok)
            ok = ds4_gpu_rms_norm_weight_rows_tensor(
                     g->qr_norm, g->qr, model->map, model->size,
                     layer->attn_q_a_norm->abs_offset, (uint32_t)q_rank, 1u,
                     DS4_RMS_EPS) != 0;
        if (ok)
            ok = ds4_gpu_rms_norm_weight_rows_tensor(
                     g->kv, g->kv_raw, model->map, model->size,
                     layer->attn_kv_a_norm->abs_offset, DS4_N_HEAD_DIM, 1u,
                     DS4_RMS_EPS) != 0;
        if (ok)
            ok = ds4_gpu_rope_tail_tensor(
                     g->kv, 1u, 1u, DS4_N_HEAD_DIM, DS4_N_ROT, pos,
                     compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, false,
                     freq_base, freq_scale, ext_factor, attn_factor,
                     DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
        if (ok)
            ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(g->kv, 1u, DS4_N_HEAD_DIM,
                                                     DS4_N_ROT) != 0;
        if (ok)
            ok = ds4_gpu_store_raw_kv_batch_tensor(raw_cache, g->kv, raw_cap,
                                                   raw_row, 1u,
                                                   DS4_N_HEAD_DIM) != 0;
    } else if (ok)
      ok = ds4_gpu_matmul_q8_0_shared_input_tensor(
               g->qr, layer->attn_q_a->abs_offset, q_rank, g->kv_raw,
               layer->attn_kv->abs_offset, DS4_N_HEAD_DIM, model->map,
               model->size, DS4_N_EMBD, g->attn_norm_q8) != 0;
    if (ok && layer->attn_q_a->type != DS4_TENSOR_Q8_1)
      ok = ds4_gpu_dsv4_qkv_rms_norm_decode_tensor(
               g->qr_norm, g->qr_norm_q8, g->qr, model->map, model->size,
               layer->attn_q_a_norm->abs_offset, (uint32_t)q_rank, g->kv,
               g->kv_raw, layer->attn_kv_a_norm->abs_offset, DS4_N_HEAD_DIM,
               raw_cache, raw_cap, raw_row, DS4_N_ROT, pos,
               compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, freq_base,
               freq_scale, ext_factor, attn_factor, DS4_ROPE_YARN_BETA_FAST,
               DS4_ROPE_YARN_BETA_SLOW, DS4_RMS_EPS) != 0;
    if (ok && (layer->attn_q_b->type == DS4_TENSOR_Q8_1 ||
               layer->attn_q_b->type == DS4_TENSOR_Q5_K))
      ok = rocm_graph_matmul_v41_dense_tensor(g->q, model, layer->attn_q_b,
                                              q_rank, q_dim, g->qr_norm, 1u);
    else if (ok)
      ok = ds4_gpu_matmul_q8_0_preq_tensor(g->q, model->map, model->size,
                                           layer->attn_q_b->abs_offset, q_rank,
                                           q_dim, g->qr_norm_q8) != 0;
    if (ok)
      ok = ds4_gpu_head_rms_norm_rope_tensor(
               g->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos,
               compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, false, freq_base,
               freq_scale, ext_factor, attn_factor, DS4_ROPE_YARN_BETA_FAST,
               DS4_ROPE_YARN_BETA_SLOW, DS4_RMS_EPS) != 0;
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("q_path");
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("kv_path");
    if (il == 0u && pos <= 1u) {
      ds4_dbg_vals(g->attn_norm, "attn_norm", 0, 6u);
      ds4_dbg_vals(g->qr, "qr", 0, 6u);
      ds4_dbg_vals(g->qr_norm, "qr_norm", 0, 6u);
      ds4_dbg_vals(g->q, "q", 0, 6u);
      ds4_dbg_vals(g->kv, "kv", 0, 6u);
    }

    uint32_t n_comp = 0;
    ds4_gpu_tensor *comp_cache = NULL;
    ds4_gpu_tensor *comp_selected = NULL;
    uint32_t n_selected = 0;
    double decode_index_stage_t0 = 0.0;
    const bool decode_index_stage_profile = getenv("GUFO_DEEPSEEK_ROCM_INDEXER_STAGE_PROFILE") != NULL;
    if (ok && compressed) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const bool emit = ((pos + 1u) % ratio) == 0u;
        const bool ds4_comp_producer = ds4_layer_is_kv_source(il);
        if (ds4_comp_producer &&
            (!layer->attn_compressor_kv || !layer->attn_compressor_norm ||
             layer->attn_compressor_kv->dim[0] != DS4_N_EMBD ||
             layer->attn_compressor_kv->dim[1] != comp_width ||
             (layer->attn_compressor_gate &&
              (layer->attn_compressor_gate->dim[0] != DS4_N_EMBD ||
               layer->attn_compressor_gate->dim[1] != comp_width)))) {
          fprintf(stderr,
                  "ds4: ROCm graph compressor weights unexpected at layer %u\n",
                  il);
          ok = false;
        }
        if (ok && emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
            fprintf(stderr, "ds4: ROCm graph compressed KV cache capacity exceeded at layer %u\n", il);
            ok = false;
        }
        if (ok && ds4_comp_producer) {
          ok = rocm_graph_matmul_weight_tensor(g->comp_kv_cur, model,
                   layer->attn_compressor_kv, DS4_N_EMBD, comp_width,
                   g->attn_norm, 1);
          if (ok && layer->attn_compressor_gate)
            ok = rocm_graph_matmul_weight_tensor(g->comp_sc_cur, model,
                     layer->attn_compressor_gate, DS4_N_EMBD, comp_width,
                     g->attn_norm, 1);
          else if (ok)
            ok = rocm_tensor_fill_f32(g->comp_sc_cur, 0.0f, comp_width);
        }
        const bool indexed = ratio == 4;
        /* V4.1 source-sharing artifact signature: two-level candidate
         * metadata exists only in V4.1 GGUFs.  Indexed layers there are the
         * top-k publishers, not ratio-4 layers. */
        uint32_t cand_layer = 0, cand_blocks = 0, cand_block_size = 0;
        ds4_cand_params(&cand_layer, &cand_blocks, &cand_block_size);
        const bool v41 = cand_block_size != 0u && cand_blocks != 0u;
        const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
        if (ok && indexed) {
            if (!layer->indexer_compressor_kv ||
                !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape ||
                !layer->indexer_compressor_norm ||
                !ds4_graph_f16_weight_type(layer->indexer_compressor_kv->type) ||
                !ds4_graph_f16_weight_type(layer->indexer_compressor_gate->type) ||
                layer->indexer_compressor_kv->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_gate->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_kv->dim[1] != index_width ||
                layer->indexer_compressor_gate->dim[1] != index_width) {
              fprintf(stderr,
                      "ds4: ROCm graph indexer compressor expects paired F16 "
                      "projections\n");
              ok = false;
            }
            if (ok && emit && g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
                fprintf(stderr, "ds4: ROCm graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            if (ok &&
                layer->indexer_compressor_kv->type == DS4_TENSOR_F16)
              ok = ds4_gpu_matmul_f16_pair_tensor(
                       g->index_kv_cur, g->index_sc_cur, model->map, model->size,
                       layer->indexer_compressor_kv->abs_offset,
                       layer->indexer_compressor_gate->abs_offset, DS4_N_EMBD,
                       index_width, g->attn_norm, 1) != 0;
            else if (ok)
              ok = rocm_graph_matmul_hc_fn_tensor(
                       g->index_kv_cur, model, layer->indexer_compressor_kv,
                       DS4_N_EMBD, index_width, g->attn_norm, 1) &&
                   rocm_graph_matmul_hc_fn_tensor(
                       g->index_sc_cur, model, layer->indexer_compressor_gate,
                       DS4_N_EMBD, index_width, g->attn_norm, 1);
        }
        if (ok && ds4_comp_producer &&
            (layer->attn_compressor_norm->type != DS4_TENSOR_F32 ||
             (indexed && layer->indexer_compressor_norm->type != DS4_TENSOR_F32))) {
            fprintf(stderr, "ds4: ROCm graph compressor expects F32 norms\n");
            ok = false;
        }
        if (ok && ds4_comp_producer) {
            const ds4_gpu_compressor_step steps[2] = {
                {g->comp_kv_cur, g->comp_sc_cur, g->layer_attn_state_kv[il],
                 g->layer_attn_state_score[il], rocm_graph_attn_comp_update_target(g, il),
                 g->layer_attn_comp_cache_f16[il],
                 rocm_graph_attn_comp_update_row(g->layer_n_comp[il]),
                 rocm_graph_opt_w_off(layer->attn_compressor_ape), rocm_graph_opt_w_type(layer->attn_compressor_ape),
                 layer->attn_compressor_norm->abs_offset, DS4_N_HEAD_DIM, false,
                 /*V4.1 index-k staging: pre-RoPE pooled latent (NULL-able). */
                 (v41 && ds4_layer_is_kv_source(il)) ? g->comp_latent : NULL},
                {g->index_kv_cur, g->index_sc_cur, g->layer_index_state_kv[il],
                 g->layer_index_state_score[il], g->layer_index_comp_cache[il], NULL,
                 g->layer_n_index_comp[il],
                 indexed ? layer->indexer_compressor_ape->abs_offset : 0,
                 indexed ? layer->indexer_compressor_ape->type : 0,
                 indexed ? layer->indexer_compressor_norm->abs_offset : 0,
                 DS4_N_INDEXER_HEAD_DIM, true},
            };
            ok = ds4_gpu_compressor_decode_tensor(
                     steps, indexed ? 2u : 1u, model->map, model->size, ratio, pos,
                     DS4_N_ROT, (uint32_t)DS4_ROPE_ORIG_CTX, freq_base, freq_scale,
                     ext_factor, attn_factor, DS4_ROPE_YARN_BETA_FAST,
                     DS4_ROPE_YARN_BETA_SLOW, DS4_RMS_EPS) != 0;
        }
        if (ok && emit && ds4_comp_producer) {
            g->layer_n_comp[il]++;
            if (indexed) g->layer_n_index_comp[il]++;
        }

        /* V4.1 index key: derived from the pooled latent BEFORE RoPE
         * (reference Indexer: attn_k -> k_norm -> rope tail -> QAT), stored
         * as one row of the owner layer's shared index cache. */
        if (ok && v41 && emit && ds4_layer_is_kv_source(il) &&
            ds4_layer_is_topk_source(il)) {
            if (!layer->indexer_attn_k || !layer->indexer_k_norm ||
                layer->indexer_attn_k->dim[0] != DS4_N_HEAD_DIM ||
                layer->indexer_attn_k->dim[1] != DS4_N_INDEXER_HEAD_DIM) {
                fprintf(stderr,
                        "ds4: ROCm graph V4.1 index-k weights missing at layer %u\n",
                        il);
                ok = false;
            }
            if (ok && g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
                fprintf(stderr,
                        "ds4: ROCm graph V4.1 index cache capacity exceeded at layer %u\n",
                        il);
                ok = false;
            }
            ds4_gpu_tensor *krow = NULL;
            if (ok)
                krow = ds4_gpu_tensor_view(
                    g->layer_index_comp_cache[il],
                    (uint64_t)g->layer_n_index_comp[il] *
                        DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                    (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float));
            if (ok && !krow) ok = false;
            if (ok)
                ok = rocm_graph_matmul_weight_tensor(
                         krow, model, layer->indexer_attn_k, DS4_N_HEAD_DIM,
                         DS4_N_INDEXER_HEAD_DIM, g->comp_latent, 1);
            if (ok)
                ok = ds4_gpu_rms_norm_weight_rows_tensor(
                         krow, krow, model->map, model->size,
                         layer->indexer_k_norm->abs_offset,
                         DS4_N_INDEXER_HEAD_DIM, 1, DS4_RMS_EPS) != 0;
            if (ok)
                ok = ds4_gpu_rope_tail_tensor(
                         krow, 1, 1, DS4_N_INDEXER_HEAD_DIM, DS4_N_ROT,
                         pos + 1u - ratio,
                         compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, false,
                         freq_base, freq_scale, ext_factor, attn_factor,
                         DS4_ROPE_YARN_BETA_FAST,
                         DS4_ROPE_YARN_BETA_SLOW) != 0;
            if (ok)
                ok = ds4_gpu_dsv4_indexer_qat_tensor(
                         krow, 1, DS4_N_INDEXER_HEAD_DIM) != 0;
            ds4_gpu_tensor_free(krow);
            if (ok) g->layer_n_index_comp[il]++;
        }

        if (ok && indexed) {
            if (ok && g->layer_n_comp[il] > DS4_N_INDEXER_TOP_K &&
                g->layer_n_index_comp[il] > DS4_N_INDEXER_TOP_K) {
              const uint64_t indexer_q_dim =
                  (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
              if (!layer->indexer_attn_q_b ||
                  !ds4_graph_f16_weight_type(layer->indexer_attn_q_b->type) ||
                  layer->indexer_attn_q_b->dim[0] != q_rank ||
                  layer->indexer_attn_q_b->dim[1] != indexer_q_dim) {
                fprintf(stderr,
                        "ds4: ROCm graph indexer q projection expects F16 "
                        "weights\n");
                ok = false;
              }
              if (ok && (!layer->indexer_proj ||
                         !ds4_graph_f16_weight_type(layer->indexer_proj->type) ||
                         layer->indexer_proj->dim[0] != DS4_N_EMBD ||
                         layer->indexer_proj->dim[1] != DS4_N_INDEXER_HEAD)) {
                fprintf(stderr,
                        "ds4: ROCm graph indexer weight projection expects F16 "
                        "weights\n");
                ok = false;
              }
              if (ok)
                ok = rocm_graph_matmul_hc_fn_tensor(
                         g->indexer_q, model, layer->indexer_attn_q_b,
                         q_rank, indexer_q_dim, g->qr_norm, 1);
              if (ok)
                ok = ds4_gpu_dsv4_indexer_rope_qat_tensor(
                         g->indexer_q, DS4_N_INDEXER_HEAD,
                         DS4_N_INDEXER_HEAD_DIM, DS4_N_ROT, pos,
                         compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, freq_base,
                         freq_scale, ext_factor, attn_factor,
                         DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
              if (ok)
                ok = rocm_graph_matmul_hc_fn_tensor(
                         g->indexer_weights, model, layer->indexer_proj,
                         DS4_N_EMBD, DS4_N_INDEXER_HEAD, g->attn_norm, 1);
              const float index_scale =
                  1.0f /
                  sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
              if (ok && decode_index_stage_profile) {
                ok = rocm_graph_indexer_stage_profile_boundary(
                    NULL, il, pos, 1, g->layer_n_index_comp[il],
                    &decode_index_stage_t0);
              }
              if (ok)
                ok = rocm_graph_reserve_indexer_scores(
                         g, g->layer_n_index_comp[il], 1) &&
                     ds4_gpu_indexer_score_one_tensor(
                         g->indexer_scores, g->indexer_q, g->indexer_weights,
                         g->layer_index_comp_cache[il],
                         g->layer_n_index_comp[il], DS4_N_INDEXER_HEAD,
                         DS4_N_INDEXER_HEAD_DIM, index_scale) != 0;
              if (ok && decode_index_stage_profile) {
                ok = rocm_graph_indexer_stage_profile_boundary(
                    "decode_score", il, pos, 1, g->layer_n_index_comp[il],
                    &decode_index_stage_t0);
              }
              if (ok)
                ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                 g->indexer_scores,
                                                 g->layer_n_index_comp[il], 1,
                                                 DS4_N_INDEXER_TOP_K) != 0;
              if (ok && decode_index_stage_profile) {
                ok = rocm_graph_indexer_stage_profile_boundary(
                    "decode_topk", il, pos, 1, g->layer_n_index_comp[il],
                    &decode_index_stage_t0);
              }
              /* Decode used to materialize a dense compressed-row mask and
               * call the generic gathered FlashAttention wrapper below.
               * That wrapper scans every compressed row and rejects long
               * contexts once raw+compressed rows exceed 8192.  Ratio-4 DS4
               * attention is sparse after indexer top-k, so use the private
               * indexed attention kernel instead: it scans only SWA raw rows
               * plus the selected compressed rows, matching prefill and
               * avoiding the long-context decode failure. */
              if (ok) {
                comp_selected = g->comp_selected;
                /*
                 * Contract: the indexer top-k is fixed by the model config
                 * and must remain the full 512 rows.  Do not reduce this for
                 * throughput benchmarks.
                 *
                 * Why: the indexer is not just an implementation detail.  It
                 * decides which compressed memory rows are visible to the
                 * attention kernel.  If we keep only 128/256 rows, the later
                 * indexed-attention math may be perfectly computed, but it is
                 * computed over the wrong candidate set: rows ranked 257-512
                 * are removed before softmax/PV can use them.  Those rows may
                 * carry weak-but-necessary evidence for retrieval, name/number
                 * recall, or long-context disambiguation.  The error is
                 * therefore semantic/algorithmic, not the acceptable kind of
                 * local numerical drift caused by a different reduction order
                 * or Tensor/NAX precision.
                 *
                 * Short prompt tests, first-token agreement, or even a small
                 * official-vector set can miss this because many prompts do
                 * not need the tail of the 512 selected compressed rows.  The
                 * failure appears only when the model needs information that
                 * fell below the reduced cutoff.  Optimizations belong inside
                 * the score/top-k/attention implementation while preserving
                 * DS4_N_INDEXER_TOP_K.
                 */
                n_selected = DS4_N_INDEXER_TOP_K < g->layer_n_index_comp[il]
                                 ? DS4_N_INDEXER_TOP_K
                                 : g->layer_n_index_comp[il];
              }
            }
        }

        /* V4.1 indexer publisher: one top-k per step, computed by the group's
         * index source over the OWNER's shared index cache, then published
         * for every grouped reader (reference kv_source/topk_source lists). */
        if (ok && v41 && ratio != 0 && ds4_layer_is_topk_source(il)) {
            const uint32_t idx_kv = ds4_layer_kv_source(il);
            const uint32_t n_idx = g->layer_n_index_comp[idx_kv];
            const uint64_t indexer_q_dim =
                (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_attn_q_b || !layer->indexer_proj ||
                layer->indexer_attn_q_b->dim[0] != q_rank ||
                layer->indexer_attn_q_b->dim[1] != indexer_q_dim ||
                layer->indexer_proj->dim[0] != DS4_N_EMBD ||
                layer->indexer_proj->dim[1] != DS4_N_INDEXER_HEAD) {
                fprintf(stderr,
                        "ds4: ROCm graph V4.1 indexer projections missing/mismatched at layer %u\n",
                        il);
                ok = false;
            }
            if (ok && n_idx > 0u && g->layer_index_comp_cache[idx_kv] == NULL) {
                fprintf(stderr,
                        "ds4: ROCm graph V4.1 shared index cache absent for layer %u (owner %u)\n",
                        il, idx_kv);
                ok = false;
            }
            if (ok && n_idx > 0u) {
                ok = rocm_graph_matmul_hc_fn_tensor(
                         g->indexer_q, model, layer->indexer_attn_q_b,
                         q_rank, indexer_q_dim, g->qr_norm, 1);
                if (ok)
                    ok = ds4_gpu_dsv4_indexer_rope_qat_tensor(
                             g->indexer_q, DS4_N_INDEXER_HEAD,
                             DS4_N_INDEXER_HEAD_DIM, DS4_N_ROT, pos,
                             compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                             freq_base, freq_scale, ext_factor, attn_factor,
                             DS4_ROPE_YARN_BETA_FAST,
                             DS4_ROPE_YARN_BETA_SLOW) != 0;
                if (ok)
                    ok = rocm_graph_matmul_hc_fn_tensor(
                             g->indexer_weights, model, layer->indexer_proj,
                             DS4_N_EMBD, DS4_N_INDEXER_HEAD, g->attn_norm, 1);
                const float index_scale =
                    1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM *
                                         DS4_N_INDEXER_HEAD));
                if (ok)
                    ok = rocm_graph_reserve_indexer_scores(g, n_idx, 1) &&
                         ds4_gpu_indexer_score_one_tensor(
                             g->indexer_scores, g->indexer_q,
                             g->indexer_weights,
                             g->layer_index_comp_cache[idx_kv], n_idx,
                             DS4_N_INDEXER_HEAD, DS4_N_INDEXER_HEAD_DIM,
                             index_scale) != 0;
                /* Two-level selection: publish candidates at the candidate
                 * source; every later publisher restricts its top-k to that
                 * candidate set (reference candidate_source_layer). */
                if (ok && il == cand_layer) {
                    ok = ds4_gpu_indexer_candidate_mask_tensor(
                             g->indexer_scores, n_idx, cand_block_size,
                             cand_blocks, g->cand_mask) != 0;
                    if (ok) {
                        g->cand_mask_n = n_idx;
                        g->cand_mask_valid = true;
                    }
                }
                if (ok && g->cand_mask_valid && g->cand_mask_n >= n_idx)
                    ok = ds4_gpu_indexer_apply_mask_tensor(
                             g->indexer_scores, n_idx, g->cand_mask) != 0;
                if (ok) {
                    /* Reference selects min(top_k, available) rows; the top-k
                     * kernel rejects top_k > n_comp outright. */
                    const uint32_t top_k_sel =
                        n_idx < DS4_N_INDEXER_TOP_K ? n_idx : DS4_N_INDEXER_TOP_K;
                    ok = ds4_gpu_indexer_topk_tensor(
                             g->comp_selected, g->indexer_scores, n_idx, 1,
                             top_k_sel) != 0;
                    if (ok) {
                        g->shared_topk_n = top_k_sel;
                        g->shared_topk_kv = idx_kv;
                        ok = ds4_gpu_tensor_copy(
                                 g->shared_topk, 0, g->comp_selected, 0,
                                 (uint64_t)g->shared_topk_n *
                                     sizeof(uint32_t)) != 0;
                    }
                }
                if (ok) {
                    comp_selected = g->shared_topk;
                    n_selected = g->shared_topk_n;
                }
            }
        }

        /* Grouped readers consume the top-k published earlier this step by
         * their index source; nothing published yet (or an owner switch)
         * leaves them on the dense fallback. */
        if (ok && v41 && ratio != 0 && !ds4_layer_is_topk_source(il) &&
            g->shared_topk_n > 0u &&
            g->shared_topk_kv == ds4_layer_kv_source(il)) {
            comp_selected = g->shared_topk;
            n_selected = g->shared_topk_n;
        }

        n_comp = g->layer_n_comp[rocm_graph_comp_src(il)];
        comp_cache = g->layer_attn_comp_cache[rocm_graph_comp_src(il)];
        if (getenv("DS4_NO_COMP")) n_comp = 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("compressor_indexer");

    if (ok && ds4_no_comp_attn_env()) {
        n_comp = 0;
        comp_cache = NULL;
        comp_selected = NULL;
        n_selected = 0;
    }

    if (ok) {
        const uint32_t raw_start = rocm_graph_raw_start_for_span(g, pos, n_raw);
        if (n_comp != 0 && comp_selected != NULL && n_selected != 0) {
            ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                    g->heads,
                    model->map,
                    model->size,
                    layer->attn_sinks->abs_offset,
                    g->q,
                    raw_cache,
                    comp_cache,
                    rocm_graph_attn_comp_cache_is_f16(),
                    comp_selected,
                    1,
                    pos,
                    n_raw,
                    raw_cap,
                    raw_start,
                    n_comp,
                    n_selected,
                    g->raw_window,
                    ds4_layer_compress_ratio(il),
                    DS4_N_HEAD,
                    DS4_N_HEAD_DIM) != 0;
            if (ok && decode_index_stage_profile) {
                ok = rocm_graph_indexer_stage_profile_boundary("decode_attention",
                                                                il,
                                                                pos,
                                                                1,
                                                                n_comp,
                                                                &decode_index_stage_t0);
            }
        } else {
            ok = ds4_gpu_attention_decode_heads_tensor(g->heads,
                                                         model->map, model->size,
                                                         layer->attn_sinks->abs_offset,
                                                         g->q, raw_cache, n_raw,
                                                         raw_cap,
                                                         raw_start,
                                                         n_comp ? comp_cache : NULL,
                                                         rocm_graph_attn_comp_cache_is_f16(),
                                                         n_comp,
                                                         NULL,
                                                         0,
                                                         DS4_N_HEAD, DS4_N_HEAD_DIM) != 0;
        }
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("attention");
    if (il == 0u && pos <= 1u) ds4_dbg_vals(g->heads, "heads", 0, 6u);
    if (ok)
      ok = ds4_gpu_rope_tail_quantize_q8_0_tensor(
               g->heads, g->heads_q8, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos,
               compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, true, freq_base,
               freq_scale, ext_factor, attn_factor, DS4_ROPE_YARN_BETA_FAST,
               DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok && layer->attn_output_a->type == DS4_TENSOR_Q5_K) {
        /* V4.1 dense tier: grouped Q5_K wo_a + plain Q5_K wo_b + hc split. */
        ok = ds4_gpu_attention_output_a_q5_k_tensor(
                 g->attn_low, model->map, model->size,
                 layer->attn_output_a->abs_offset, group_dim, rank, n_groups,
                 g->heads, 1u) != 0;
        if (ok)
            ok = ds4_gpu_matmul_q5_k_tensor(
                     g->attn_out, model->map, model->size,
                     layer->attn_output_b->abs_offset,
                     (uint64_t)n_groups * rank, DS4_N_EMBD, g->attn_low,
                     1u) != 0;
        if (ok)
            ok = ds4_gpu_hc_expand_split_tensor(
                     g->after_attn_hc, g->attn_out, g->cur_hc, g->hc_split_a,
                     DS4_N_EMBD, DS4_N_HC) != 0;
    } else if (ok) {
      ok = ds4_gpu_attention_output_low_q8_tensor(
               g->attn_low, model->map, model->size,
               layer->attn_output_a->abs_offset, group_dim, rank, n_groups,
               g->heads, g->heads_q8, g->attn_low_q8) != 0;
      if (ok) {
        ok = ds4_gpu_matmul_q8_0_hc_expand_tensor(
                 g->after_attn_hc, g->attn_out, model->map, model->size,
                 layer->attn_output_b->abs_offset, (uint64_t)n_groups * rank,
                 DS4_N_EMBD, g->attn_low, g->attn_low_q8, g->cur_hc, g->hc_split_a,
                 DS4_N_EMBD, DS4_N_HC) != 0;
      }
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("attn_output");
    if (il == 0u && pos <= 1u) ds4_dbg_vals(g->attn_out, "attn_out", 0, 6u);

    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("attn_hc_post");
    if (ok && il == 0u && pos <= 1u) {
      ds4_dbg_dump(g->hc_split_a, "/tmp/g0_coef_a.bin", 24u);
      ds4_dbg_dump(g->after_attn_hc, "/tmp/g0_x2.bin", hc_dim);
      ds4_dbg_dump(g->attn_out, "/tmp/g0_a.bin", DS4_N_EMBD);
      ds4_dbg_dump(g->ffn_cur, "/tmp/g0_x3.bin", DS4_N_EMBD);
    }
    if (ok) ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->after_attn_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = rocm_graph_matmul_plain_tensor(g->hc_mix, model, layer->hc_ffn_fn,
                                                 hc_dim, mix_hc, g->flat_hc, 1);
    if (ok) {
      ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(
               g->ffn_cur, g->ffn_norm, g->ffn_norm_q8, g->ffn_norm_q8k,
               g->hc_split, g->hc_mix, g->after_attn_hc, model->map, model->size,
               layer->hc_ffn_scale->abs_offset, layer->hc_ffn_base->abs_offset,
               layer->ffn_norm->abs_offset, DS4_N_EMBD, DS4_N_HC,
               DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS, DS4_RMS_EPS,
               g->hc_split_a, 24u) != 0;
    }
    /* Keep the raw FFN mix for the V4.1 output head (see hc_mix_ffn_raw). */
    if (ok)
      ok = ds4_gpu_tensor_copy(g->hc_mix_ffn_raw, 0, g->hc_mix, 0,
                               DS4_N_HC * sizeof(float)) != 0;
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("ffn_hc_pre");
    if (ok && il == 0u && pos <= 1u) ds4_dbg_dump(g->hc_split, "/tmp/g0_coef_f.bin", 24u);
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("ffn_norm");
    const uint64_t gate_row_bytes = ds4_routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = ds4_routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
    if (ok) ok = rocm_graph_matmul_plain_tensor(g->router_logits, model, layer->ffn_gate_inp,
                                                 DS4_N_EMBD, DS4_N_EXPERT, g->ffn_norm, 1);
    if (ok) ok = ds4_gpu_router_select_tensor(g->router_selected, g->router_weights, g->router_probs,
                                                model->map, model->size,
                                                layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                (uint32_t)token,
                                                DS4_N_EXPERT,
                                                DS4_N_EXPERT_USED,
                                                DS4_EXPERT_WEIGHT_SCALE,
                                                0,
                                                0,
                                                layer->ffn_exp_probs_b != NULL,
                                                layer->ffn_gate_tid2eid != NULL,
                                                g->router_logits) != 0;
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("router");
    if (il == 0u && pos <= 1u) {
      ds4_dbg_vals(g->after_attn_hc, "after_attn", 0, 6u);
      ds4_dbg_vals(g->ffn_norm, "ffn_norm", 0, 6u);
      ds4_dbg_vals(g->router_logits, "router", 0, 12u);
      int sel[8] = {0};
      float sw[8] = {0};
      if (ds4_gpu_tensor_read(g->router_selected, 0, sel, 6 * sizeof(int)) &&
          ds4_gpu_tensor_read(g->router_weights, 0, sw, 6 * sizeof(float)))
        fprintf(stderr, "ds4: g0[sel] %d %d %d %d %d %d w %f %f %f %f %f %f\n",
                sel[0], sel[1], sel[2], sel[3], sel[4], sel[5],
                (double)sw[0], (double)sw[1], (double)sw[2], (double)sw[3],
                (double)sw[4], (double)sw[5]);
      static FILE *binfp = fopen("/tmp/g0_ffn_norm.bin", "wb");
      static std::vector<float> normbuf(DS4_N_EMBD);
      if (binfp &&
          ds4_gpu_tensor_read(g->ffn_norm, 0, normbuf.data(),
                              DS4_N_EMBD * sizeof(float))) {
        fwrite(normbuf.data(), 4, DS4_N_EMBD, binfp);
        fflush(binfp);
      }
    }
    if (ok)
      ok = ds4_gpu_routed_moe_one_tensor(
               g->routed_out, g->routed_gate, g->routed_up, g->routed_mid,
               g->routed_down, model->map, model->size,
               layer->ffn_gate_exps->abs_offset, layer->ffn_up_exps->abs_offset,
               layer->ffn_down_exps->abs_offset, layer->ffn_gate_exps->type,
               layer->ffn_down_exps->type, gate_expert_bytes, gate_row_bytes,
               down_expert_bytes, down_row_bytes, (uint32_t)expert_in_dim,
               (uint32_t)down_in_dim, (uint32_t)routed_out_dim,
               g->router_selected, g->router_weights, DS4_N_EXPERT,
               DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, g->ffn_norm,
               g->ffn_norm_q8k) != 0;
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("routed_moe");
    if (il == 0u && pos <= 1u) ds4_dbg_vals(g->routed_mid, "mid", 0, 6u);
    if (ok) {
      if (layer->ffn_gate_shexp->type == DS4_TENSOR_Q5_K) {
        ok = ds4_gpu_shared_gate_up_swiglu_q5_k_tensor(
                 g->shared_mid, model->map, model->size,
                 layer->ffn_gate_shexp->abs_offset,
                 layer->ffn_up_shexp->abs_offset, DS4_N_EMBD, shared_dim,
                 g->ffn_norm, DS4_SWIGLU_CLAMP_EXP, 1u) != 0;
      } else {
        ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                 g->shared_gate, g->shared_up, g->shared_mid, g->shared_mid_q8,
                 model->map, model->size, layer->ffn_gate_shexp->abs_offset,
                 layer->ffn_up_shexp->abs_offset, DS4_N_EMBD, shared_dim,
                 g->ffn_norm, g->ffn_norm_q8, DS4_SWIGLU_CLAMP_EXP, 1u) != 0;
      }
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("shared_gate_up");
    if (ok && layer->ffn_down_shexp->type == DS4_TENSOR_Q5_K) {
      ok = ds4_gpu_shared_down_q5_k_tensor(
               g->shared_out, model->map, model->size,
               layer->ffn_down_shexp->abs_offset, shared_dim, DS4_N_EMBD,
               g->shared_mid, 1u) != 0;
      if (ok)
        ok = ds4_gpu_hc_expand_add_split_tensor(
                 g->after_ffn_hc, g->routed_out, g->shared_out,
                 g->after_attn_hc, g->hc_split, DS4_N_EMBD, DS4_N_HC) != 0;
    } else if (ok) {
      ok = ds4_gpu_shared_down_hc_expand_q8_0_tensor(
               g->after_ffn_hc, g->shared_out, model->map, model->size,
               layer->ffn_down_shexp->abs_offset, shared_dim, DS4_N_EMBD,
               g->shared_mid, g->shared_mid_q8, g->routed_out, g->after_attn_hc, g->hc_split,
               DS4_N_EMBD, DS4_N_HC) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("shared_down");
    if (il == 0u && pos <= 1u) {
      ds4_dbg_vals(g->routed_out, "routed_out", 0, 6u);
      ds4_dbg_vals(g->shared_out, "shared_out", 0, 6u);
      ds4_dbg_vals(g->after_ffn_hc, "after_ffn_s0", 0, 6u);
      ds4_dbg_vals(g->after_ffn_hc, "after_ffn_s1", (uint64_t)DS4_N_EMBD, 6u);
    }

    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("ffn_hc_post");
#undef GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE
    return ok;
}

/* Encode the final HC collapse, output norm, and vocab projection on ROCm. */
/* Journald rate-limits bursts; AD prefill floods stage lines. Write the
 * definitive head-argmax evidence to a file that survives suppression. */
static void ds4_head_probe_file(const char *tag, ds4_gpu_tensor *logits,
                                uint64_t nv, uint32_t pos) {
    if (!logits || !nv) return;
    std::vector<float> lg(nv);
    FILE *f = fopen("/tmp/ds4_head_probe.log", "a");
    if (!f) return;
    if (!ds4_gpu_readback_f32(logits, 0, nv, lg.data())) {
        fprintf(f, "%s pos=%u READBACK_FAIL\n", tag, pos);
    } else {
        uint64_t best = 0;
        for (uint64_t i = 1; i < nv; i++) if (lg[i] > lg[best]) best = i;
        fprintf(f, "%s pos=%u argmax=%llu %.4f vals:", tag, pos,
                (unsigned long long)best, lg[best]);
        for (uint64_t i = 0; i < 6 && i < nv; i++) fprintf(f, " %.5f", lg[i]);
        fputc('\n', f);
    }
    fclose(f);
}

static bool rocm_graph_encode_output_head(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint64_t               vocab_dim,
        ds4_gpu_tensor        *mix_pre) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    /* V4.1 ships no output_hc_* head: the streams collapse with the last
     * layer's FFN pre-mix (llama.cpp deepseek41.cpp "no hc_head: collapse
     * with the last FFN's pre-mix").  mix_pre is that layer's raw mix row
     * (first DS4_N_HC entries); the affine/sigmoid/eps math below is the
     * same one build_hc_mixes applies. */
    const bool v41_collapse = weights->output_hc_fn == NULL;
    const auto &last_layer = weights->layer[DS4_N_LAYER - 1];
    ds4_gpu_tensor *pre = NULL;
    uint64_t scale_off = 0;
    uint64_t base_off = 0;
    bool ok = true;
    if (v41_collapse) {
        if (!mix_pre || !last_layer.hc_ffn_scale || !last_layer.hc_ffn_base) {
            fprintf(stderr, "ds4: V4.1 output collapse missing FFN mix\n");
            return false;
        }
        pre = mix_pre;
        scale_off = last_layer.hc_ffn_scale->abs_offset;
        base_off = last_layer.hc_ffn_base->abs_offset;
    } else {
        ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->cur_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
        if (ok)
          ok = ds4_gpu_matmul_f16_tensor(g->output_pre, model->map, model->size,
                                         weights->output_hc_fn->abs_offset, hc_dim,
                                         DS4_N_HC, g->flat_hc, 1) != 0;
        pre = g->output_pre;
        scale_off = weights->output_hc_scale->abs_offset;
        base_off = weights->output_hc_base->abs_offset;
    }
    if (ok)
      ok = ds4_gpu_output_hc_weights_tensor(
               g->output_weights, pre, model->map, model->size,
               scale_off, base_off, DS4_N_HC, DS4_HC_EPS) != 0;
    if (!ok)
        fprintf(stderr,
                "ds4: output head step failed: hc_weights (model_size=%llu "
                "scale_off=%llu base_off=%llu)\n",
                (unsigned long long)model->size, (unsigned long long)scale_off,
                (unsigned long long)base_off);
    if (ok && !ds4_gpu_probe_sync("hc_weights")) ok = false;
    if (ok)
      ok = ds4_gpu_hc_weighted_sum_tensor(g->output_embd, g->cur_hc,
                                          g->output_weights, DS4_N_EMBD,
                                          DS4_N_HC) != 0;
    if (!ok) fprintf(stderr, "ds4: output head step failed: weighted_sum\n");
    if (ok && !ds4_gpu_probe_sync("weighted_sum")) ok = false;
    if (ok)
      ok = ds4_gpu_rms_norm_weight_tensor(
               g->output_norm, g->output_embd, model->map, model->size,
               weights->output_norm->abs_offset, DS4_N_EMBD, DS4_RMS_EPS) != 0;
    ds4_probe_stats(g->output_norm, 0, DS4_N_EMBD, "final_norm", 99u);
    if (!ok) fprintf(stderr, "ds4: output head step failed: final_norm type=%u off=%llu\n",
                     weights->output_norm->type,
                     (unsigned long long)weights->output_norm->abs_offset);
    if (ok)
      ok = weights->output->type == DS4_TENSOR_Q5_K
               ? ds4_gpu_matmul_q5_k_tensor(g->logits, model->map, model->size,
                                            weights->output->abs_offset,
                                            DS4_N_EMBD, vocab_dim,
                                            g->output_norm, 1u) != 0
               : ds4_gpu_matmul_q8_0_tensor(g->logits, model->map, model->size,
                                            weights->output->abs_offset,
                                            DS4_N_EMBD, vocab_dim,
                                            g->output_norm, 1) != 0;
    if (!ok) fprintf(stderr, "ds4: output head step failed: lm_head type=%u vocab=%llu\n",
                     weights->output->type, (unsigned long long)vocab_dim);
    if (ok) ds4_probe_stats(g->logits, 0, vocab_dim, "logits", 99u);
    return ok;
}

/* Shared dense projection helpers used by decode and batched prefill. */
static bool rocm_graph_matmul_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (w->type == DS4_TENSOR_F16) {
        return ds4_gpu_matmul_f16_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    if (w->type == DS4_TENSOR_BF16) {
        /* Weights stay BF16 in memory; matmul_bf16_kernel widens in-register. */
        return ds4_gpu_matmul_bf16_tensor(out, model->map, model->size,
                                            w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    if (w->type == DS4_TENSOR_F32) {
        return ds4_gpu_matmul_f32_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    if (w->type == DS4_TENSOR_Q8_0) {
        /* The target's dense projections are F16 or F32; the DSpark support
         * model ships a Q8_0 router projection. */
        return ds4_gpu_matmul_q8_0_tensor(out, model->map, model->size,
                                            w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    fprintf(stderr, "ds4: ROCm plain matmul does not support %s\n", ds4_tensor_type_name(w->type));
    return false;
}

/* Route a plain GEMM whose weights may be F16, F32 or BF16 by stored tensor
 * type.  V4.1 ships hc_*_fn projections as F32 while earlier artifacts shipped
 * F16; reading F32 weight bytes as F16 yields garbage, so dispatch on type. */
static bool rocm_graph_matmul_hc_fn_tensor(
        ds4_gpu_tensor *out, const ds4_model *model, const ds4_tensor *w,
        uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (w->type == DS4_TENSOR_F16) {
        return ds4_gpu_matmul_f16_tensor(out, model->map, model->size,
                                         w->abs_offset, in_dim, out_dim, x,
                                         n_tok) != 0;
    }
    if (w->type == DS4_TENSOR_F32) {
        return ds4_gpu_matmul_f32_tensor(out, model->map, model->size,
                                         w->abs_offset, in_dim, out_dim, x,
                                         n_tok) != 0;
    }
    if (w->type == DS4_TENSOR_BF16) {
        return ds4_gpu_matmul_bf16_tensor(out, model->map, model->size,
                                          w->abs_offset, in_dim, out_dim, x,
                                          n_tok) != 0;
    }
    fprintf(stderr, "ds4: hc-fn matmul unsupported type=%u\n", w->type);
    return false;
}

/* Debug helper: scan one intermediate prefill tensor for non-finite values
 * (layer 0 only; GUFO_DEEPSEEK_ROCM_NAN_PROBE gates it). */
static void rocm_graph_probe_layer_tensor(const ds4_gpu_tensor *t,
                                          uint64_t elems,
                                          const char *stage, uint32_t il) {
    static const bool on =
        getenv("GUFO_DEEPSEEK_ROCM_NAN_PROBE") != NULL;
    if (!on || !t || il != 0) return;
    char tag[128];
    snprintf(tag, sizeof(tag), "l0 %s", stage);
    unsigned long long c = 0, fi = 0;
    ds4_gpu_probe_nan(t, elems, tag, &c, &fi);
}

static bool rocm_graph_matmul_q8_0_named_tensor(
        const char             *module,
        uint32_t                il,
        uint32_t                pos0,
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    (void)module;
    (void)il;
    (void)pos0;
    const bool ok = ds4_gpu_matmul_q8_0_tensor(out,
                                                 model->map,
                                                 model->size,
                                                 w->abs_offset,
                                                 in_dim,
                                                 out_dim,
                                                 x,
                                                 n_tok) != 0;
    return ok;
}

/* V4.1 dense projections ship Q8_1 (low-rank q/kv) and Q5_K (expansions).
 * Route by stored type; other types keep the historical q8_0 path. */
static bool rocm_graph_matmul_v41_dense_tensor(
        ds4_gpu_tensor *out, const ds4_model *model, const ds4_tensor *w,
        uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (n_tok > 65535u) return false;
    if (w->type == DS4_TENSOR_Q8_1)
        return ds4_gpu_matmul_q8_1_tensor(out, model->map, model->size,
                                          w->abs_offset, in_dim, out_dim, x,
                                          (uint32_t)n_tok) != 0;
    if (w->type == DS4_TENSOR_Q5_K)
        return ds4_gpu_matmul_q5_k_tensor(out, model->map, model->size,
                                          w->abs_offset, in_dim, out_dim, x,
                                          (uint32_t)n_tok) != 0;
    return ds4_gpu_matmul_q8_0_tensor(out, model->map, model->size,
                                      w->abs_offset, in_dim, out_dim, x,
                                      n_tok) != 0;
}

/* =========================================================================
 * ROCm Release Decode and Prefill.
 * =========================================================================
 *
 * Everything below is the user-facing ROCm backend.
 */

/* Encode a full single-token decode step on ROCm.  This is the generation
 * hot path: update caches, run all layers, then produce logits. */
static bool rocm_graph_encode_token_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        bool                   need_logits) {
    if (g->raw_cap == 0) {
        fprintf(stderr, "ds4: ROCm graph raw KV cache is not allocated\n");
        return false;
    }
    const uint32_t raw_row = pos % g->raw_cap;
    const uint32_t n_raw = rocm_graph_raw_span_for_batch(g, pos, 1);

    bool ok = ds4_gpu_embed_token_hc_tensor(g->cur_hc,
                                              model->map,
                                              model->size,
                                              weights->token_embd->abs_offset,
                                              (uint32_t)weights->token_embd->dim[1],
                                              (uint32_t)token,
                                              DS4_N_EMBD,
                                              DS4_N_HC,
                                              weights->token_embd->type) != 0;

    /* One published top-k and one candidate mask live per decode step. */
    g->shared_topk_n = 0;
    g->shared_topk_kv = UINT32_MAX;
    g->cand_mask_valid = false;
    g->cand_mask_n = 0;

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        ok = rocm_graph_encode_decode_layer(g,
                                             model,
                                             &weights->layer[il],
                                             il,
                                             pos,
                                             g->layer_raw_cache[il],
                                             g->raw_cap,
                                             raw_row,
                                             n_raw,
                                             token);
        ds4_gpu_tensor *tmp = g->cur_hc;
        g->cur_hc = g->after_ffn_hc;
        g->after_ffn_hc = tmp;
        if (ok && getenv("DS4_H_DUMP")) {
            char hpath[64];
            snprintf(hpath, sizeof hpath, "/tmp/h68_L%u_p%u.bin", il, pos);
            ds4_dbg_dump(g->cur_hc, hpath, 4u * DS4_N_EMBD);
        }
        if (ok && il + 1u == DS4_N_LAYER && getenv("DS4_GOLDEN0"))
            ds4_dbg_vals(g->hc_split, "ffn_split_last24", 0, 24u);
        /* Capture after the swap so cur_hc holds this layer's output. */
        if (ok) ok = rocm_graph_dspark_capture_decode_layer(g, il);
    }

    if (ok && need_logits) {
        fprintf(stderr, "ds4: raw_swa head ENTER\n");
        /* hc_mix is reused by every layer's attention sublayer, so at this
         * point it holds layer-0 attention mix, not the last FFN's.  Use the
         * raw-mix snapshot taken right after the last FFN's mix matmul;
         * ds4_gpu_output_hc_weights_tensor re-applies affine+sigmoid on it,
         * matching the batch head which passes the raw mix row. */
        ds4_gpu_tensor *mix_pre =
            ds4_gpu_tensor_view(g->hc_mix_ffn_raw, 0, DS4_N_HC * sizeof(float));
        ok = mix_pre != NULL;
        if (ok)
            ok = rocm_graph_encode_output_head(g, model, weights,
                                               weights->output->dim[1], mix_pre);
        ds4_gpu_tensor_free(mix_pre);
        ds4_head_probe_file("rawswa", g->logits,
                            weights->output->dim[1], pos);
        if (ok && getenv("DS4_GOLDEN0")) {
          ds4_dbg_vals(g->logits, "logits", 0, 6u);
          const uint64_t nv = weights->output->dim[1];
          std::vector<float> lg(nv);
          if (ds4_gpu_readback_f32(g->logits, 0, nv, lg.data())) {
            uint64_t best = 0;
            for (uint64_t i = 1; i < nv; i++)
              if (lg[i] > lg[best]) best = i;
            fprintf(stderr, "ds4: g0[argmax] tok=%llu logit=%.4f\n",
                    (unsigned long long)best, lg[best]);
          }
        }
    }
    return ok;
}

static ds4_gpu_tensor *rocm_graph_tensor_row_view(
        ds4_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values) {
    return ds4_gpu_tensor_view(base,
                                 (uint64_t)row * row_values * sizeof(float),
                                 row_values * sizeof(float));
}

/* Upload prompt token ids for kernels that need token-aware hash routing. */
static bool rocm_graph_upload_prompt_tokens(
        ds4_gpu_tensor *out_tokens,
        const token_vec  *prompt,
        uint32_t          pos0,
        uint32_t          n_tokens) {
    if (!out_tokens || pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) {
        return false;
    }

    auto *tokens = static_cast<int32_t *>(
        ds4_xmalloc(static_cast<size_t>(n_tokens) * sizeof(int32_t)));
    for (uint32_t i = 0; i < n_tokens; i++) tokens[i] = prompt->v[pos0 + i];

    const bool ok = ds4_gpu_tensor_write(out_tokens,
                                           0,
                                           tokens,
                                           (uint64_t)n_tokens * sizeof(tokens[0])) != 0;
    free(tokens);
    return ok;
}

/* Rebuild ratio-4 compressor state after chunked prefill so a following decode
 * token sees the same rolling compression window. */
static bool rocm_graph_refresh_ratio4_compressor_state(
        ds4_gpu_graph  *g,
        const ds4_model  *model,
        ds4_gpu_tensor *state_kv,
        ds4_gpu_tensor *state_score,
        const ds4_tensor *kv_weight,
        const ds4_tensor *score_weight,
        const ds4_tensor *ape,
        uint32_t          head_dim,
        uint32_t          width,
        uint32_t          pos0,
        uint32_t          n_tokens) {
    if (n_tokens < 4) {
        return true;
    }
    if (!g || !model || !state_kv || !state_score || !kv_weight ||
        !score_weight || !ape || head_dim == 0 || width == 0) {
      return false;
    }

    /*
     * The recurrent ratio-4 state is intentionally rebuilt from the last
     * four tokens using the small-batch projection kernel. The full-chunk
     * projection is already available, but it uses the matrix-matrix path;
     * mixing those two accumulation orders changes a few FP8 rounding
     * decisions in later chunks.
     */
    ds4_gpu_tensor *tail_hc = ds4_gpu_tensor_view(
            g->batch_attn_norm,
            (uint64_t)(n_tokens - 4u) * DS4_N_EMBD * sizeof(float),
            4ull * DS4_N_EMBD * sizeof(float));
    bool ok = tail_hc != NULL;
    if (ok) {
        ok = rocm_graph_matmul_hc_fn_tensor(g->batch_comp_kv,
                                                         model, kv_weight,
                                         DS4_N_EMBD,
                                         width,
                                         tail_hc,
                                         4) != 0;
    }
    if (ok) {
        ok = rocm_graph_matmul_hc_fn_tensor(g->batch_comp_sc,
                                                         model, score_weight,
                                         DS4_N_EMBD,
                                         width,
                                         tail_hc,
                                         4) != 0;
    }
    if (ok) {
        ok = ds4_gpu_compressor_prefill_state_ratio4_tensor(state_kv,
                                                              state_score,
                                                              g->batch_comp_kv,
                                                              g->batch_comp_sc,
                                                              model->map,
                                                              model->size,
                                                              ape->abs_offset,
                                                              ape->type,
                                                              head_dim,
                                                              pos0 + n_tokens - 4u) != 0;
    }
    ds4_gpu_tensor_free(tail_hc);
    return ok;
}

/* Seed the batched HC state from token ids: every HC stream starts as the same
 * 4096-wide embedding.  The model-owned ROCm kernel handles every batch size;
 * model data never dequantizes or uploads embeddings through a host fallback. */
static bool rocm_graph_upload_prompt_embeddings_hc(
        ds4_gpu_tensor   *out_hc,
        ds4_gpu_tensor   *tokens,
        const ds4_model    *model,
        const ds4_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens) {
    if (pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) return false;
    if (!tokens || n_tokens == 0) return false;
    return ds4_gpu_embed_tokens_hc_tensor(out_hc,
                                          tokens,
                                          model->map,
                                          model->size,
                                          weights->token_embd->abs_offset,
                                          (uint32_t)weights->token_embd->dim[1],
                                          n_tokens,
                                          DS4_N_EMBD,
                                          DS4_N_HC,
                                          weights->token_embd->type) != 0;
}

/* Encode the batched prefill attention half for one layer.  It mirrors the CPU
 * layer-major path: HC pre/norm, Q/KV, cache/compression, prefix attention. */
static bool rocm_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = ds4_now_seconds();
    if (stage != NULL) {
        fprintf(stderr,
                    "ds4: ROCm indexer stage layer=%u pos=%u tokens=%u comp=%u %s=%.3f ms\n",
                il,
                pos0,
                n_tokens,
                n_comp,
                stage,
                (now - *stage_t0) * 1000.0);
    }
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

/* Optional prefill stage profiler. It intentionally ends the current ROCm
 * command buffer and waits, so the printed number includes encoding plus GPU
 * execution for the stage just emitted. This is disabled by default because it
 * adds synchronization points and changes scheduling. */
static bool rocm_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = ds4_now_seconds();
    fprintf(stderr,
            "ds4: ROCm layer stage part=%s layer=%u pos=%u tokens=%u %s=%.3f ms\n",
            part,
            il,
            pos0,
            n_tokens,
            stage,
            (now - *stage_t0) * 1000.0);
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

static bool rocm_graph_q_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = ds4_now_seconds();
    fprintf(stderr,
            "ds4: ROCm Q path stage layer=%u pos=%u tokens=%u %s=%.3f ms\n",
            il,
            pos0,
            n_tokens,
            stage,
            (now - *stage_t0) * 1000.0);
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

/*
 * Stateful middle of one decode attention row.
 *
 * Multi-session decode batches share the dense HC/Q/KV projections, but each
 * row owns a different raw/compressed cache and absolute position. This helper
 * keeps that stateful middle identical to ordinary decode while accepting row
 * views produced by the shared front end.
 */
static bool rocm_graph_encode_session_attention_core(
        ds4_gpu_graph       *g,
        const ds4_model     *model,
        const ds4_layer_weights *layer,
        uint32_t             il,
        uint32_t             pos,
        const ds4_gpu_tensor *attn_norm,
        const ds4_gpu_tensor *qr_norm,
        ds4_gpu_tensor       *q,
        ds4_gpu_tensor       *kv,
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *precomputed_comp_kv,
        const ds4_gpu_tensor *precomputed_comp_sc) {
  if (!g || !model || !layer || !attn_norm || !qr_norm || !q || !kv || !heads ||
      g->raw_cap == 0 ||
      ((precomputed_comp_kv == nullptr) != (precomputed_comp_sc == nullptr))) {
    return false;
  }
  if (!rocm_graph_reserve_compressed_position(g, il,
                                              static_cast<uint64_t>(pos) + 1))
    return false;

  const uint64_t q_rank = layer->attn_q_a->dim[1];
  const bool compressed = ds4_layer_compress_ratio(il) != 0;
  const float freq_base = layer_rope_freq_base(il);
  const float freq_scale = layer_rope_freq_scale(il);
  const float ext_factor =
      compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
  float attn_factor = 1.0f;
  if (ext_factor != 0.0f && freq_scale > 0.0f) {
    attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
  }

    const uint32_t raw_row = pos % g->raw_cap;
    const uint32_t n_raw = rocm_graph_raw_span_for_batch(g, pos, 1);
    ds4_gpu_tensor *raw_cache = g->layer_raw_cache[il];
    bool ok = ds4_gpu_rope_tail_tensor(
                  q,
                  1,
                  DS4_N_HEAD,
                  DS4_N_HEAD_DIM,
                  DS4_N_ROT,
                  pos,
                  compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                  false,
                  freq_base,
                  freq_scale,
                  ext_factor,
                  attn_factor,
                  DS4_ROPE_YARN_BETA_FAST,
                  DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        ok = ds4_gpu_rope_tail_tensor(
                 kv,
                 1,
                 DS4_N_HEAD_KV,
                 DS4_N_HEAD_DIM,
                 DS4_N_ROT,
                 pos,
                 compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                 false,
                 freq_base,
                 freq_scale,
                 ext_factor,
                 attn_factor,
                 DS4_ROPE_YARN_BETA_FAST,
                 DS4_ROPE_YARN_BETA_SLOW) != 0;
    }
    if (ok) {
        ok = rocm_graph_decode_kv_store(kv, raw_cache, g->raw_cap, raw_row);
    }

    uint32_t n_comp = 0;
    ds4_gpu_tensor *comp_cache = NULL;
    ds4_gpu_tensor *comp_selected = NULL;
    uint32_t n_selected = 0;
    if (ok && compressed) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const bool emit = ((pos + 1u) % ratio) == 0u;
        const bool ds4_comp_producer = ds4_layer_is_kv_source(il);
        if (ds4_comp_producer &&
            (!layer->attn_compressor_kv || !layer->attn_compressor_norm ||
             layer->attn_compressor_kv->dim[0] != DS4_N_EMBD ||
             layer->attn_compressor_kv->dim[1] != comp_width ||
             (layer->attn_compressor_gate &&
              (layer->attn_compressor_gate->dim[0] != DS4_N_EMBD ||
               layer->attn_compressor_gate->dim[1] != comp_width)))) {
          fprintf(stderr,
                  "ds4: ROCm graph compressor weights unexpected at layer %u\n",
                  il);
          ok = false;
        }
        if (ok && emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
            fprintf(stderr,
                    "ds4: ROCm graph compressed KV cache capacity exceeded "
                    "at layer %u\n",
                    il);
            ok = false;
        }
        if (ok && !precomputed_comp_kv && ds4_comp_producer) {
            ok = rocm_graph_matmul_weight_tensor(g->comp_kv_cur, model,
                     layer->attn_compressor_kv, DS4_N_EMBD, comp_width,
                     attn_norm, 1);
            if (ok && layer->attn_compressor_gate)
                ok = rocm_graph_matmul_weight_tensor(g->comp_sc_cur, model,
                         layer->attn_compressor_gate, DS4_N_EMBD, comp_width,
                         attn_norm, 1);
            else if (ok)
                ok = rocm_tensor_fill_f32(g->comp_sc_cur, 0.0f, comp_width);
        }
        const ds4_gpu_tensor *attn_comp_kv =
            precomputed_comp_kv ? precomputed_comp_kv : g->comp_kv_cur;
        const ds4_gpu_tensor *attn_comp_sc =
            precomputed_comp_sc ? precomputed_comp_sc : g->comp_sc_cur;
        const uint32_t comp_row = g->layer_n_comp[il];
        if (ok && ds4_comp_producer) {
            ok = ds4_gpu_compressor_update_tensor(
                     attn_comp_kv,
                     attn_comp_sc,
                     g->layer_attn_state_kv[il],
                     g->layer_attn_state_score[il],
                     rocm_graph_attn_comp_update_target(g, il),
                     model->map,
                     model->size,
                     rocm_graph_opt_w_off(layer->attn_compressor_ape),
                     rocm_graph_opt_w_type(layer->attn_compressor_ape),
                     layer->attn_compressor_norm->abs_offset,
                     layer->attn_compressor_norm->type,
                     DS4_N_HEAD_DIM,
                     ratio,
                     pos,
                     rocm_graph_attn_comp_update_row(comp_row),
                     DS4_N_ROT,
                     compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                     freq_base,
                     freq_scale,
                     ext_factor,
                     attn_factor,
                     DS4_ROPE_YARN_BETA_FAST,
                     DS4_ROPE_YARN_BETA_SLOW,
                     DS4_RMS_EPS,
                     false,
                     true,
                     false) != 0;
        }
        if (ok && emit) {
            ds4_gpu_tensor *comp_row_view =
                rocm_graph_attn_comp_row_view(g, il, comp_row);
            if (!comp_row_view) {
                ok = false;
            } else {
                ok = rocm_graph_quantize_attn_comp_row(
                    g, il, comp_row, comp_row_view);
                ds4_gpu_tensor_free(comp_row_view);
            }
        }
        if (ok && emit && ds4_comp_producer) {
            g->layer_n_comp[il]++;
        }

        if (ok && ratio == 4) {
            const uint32_t index_width =
                coff * DS4_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv ||
                !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape ||
                !layer->indexer_compressor_norm ||
                !ds4_graph_f16_weight_type(layer->indexer_compressor_kv->type) ||
                !ds4_graph_f16_weight_type(layer->indexer_compressor_gate->type) ||
                layer->indexer_compressor_kv->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_gate->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_kv->dim[1] != index_width ||
                layer->indexer_compressor_gate->dim[1] != index_width) {
              fprintf(stderr,
                      "ds4: ROCm graph indexer compressor expects paired "
                      "F16 compressor projections\n");
              ok = false;
            }
            if (ok && emit &&
                g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
              fprintf(stderr,
                      "ds4: ROCm graph indexer compressed KV cache capacity "
                      "exceeded at layer %u\n",
                      il);
              ok = false;
            }
            if (ok) {
                ok = ds4_gpu_matmul_f16_pair_tensor(
                         g->comp_kv_cur,
                         g->comp_sc_cur,
                         model->map,
                         model->size,
                         layer->indexer_compressor_kv->abs_offset,
                         layer->indexer_compressor_gate->abs_offset,
                         DS4_N_EMBD,
                         index_width,
                         attn_norm,
                         1) != 0;
            }
            const uint32_t index_row = g->layer_n_index_comp[il];
            if (ok) {
                ok = ds4_gpu_compressor_update_tensor(
                         g->comp_kv_cur,
                         g->comp_sc_cur,
                         g->layer_index_state_kv[il],
                         g->layer_index_state_score[il],
                         g->layer_index_comp_cache[il],
                         model->map,
                         model->size,
                         layer->indexer_compressor_ape->abs_offset,
                         layer->indexer_compressor_ape->type,
                         layer->indexer_compressor_norm->abs_offset,
                         layer->indexer_compressor_norm->type,
                         DS4_N_INDEXER_HEAD_DIM,
                         ratio,
                         pos,
                         index_row,
                         DS4_N_ROT,
                         compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                         freq_base,
                         freq_scale,
                         ext_factor,
                         attn_factor,
                         DS4_ROPE_YARN_BETA_FAST,
                         DS4_ROPE_YARN_BETA_SLOW,
                         DS4_RMS_EPS,
                         false,
                         true,
                         false) != 0;
            }
            if (ok && emit) {
                ds4_gpu_tensor *index_row_view = ds4_gpu_tensor_view(
                    g->layer_index_comp_cache[il],
                    (uint64_t)index_row * DS4_N_INDEXER_HEAD_DIM *
                        sizeof(float),
                    (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                if (!index_row_view) {
                    ok = false;
                } else {
                    ok = ds4_gpu_dsv4_indexer_qat_tensor(
                             index_row_view,
                             1,
                             DS4_N_INDEXER_HEAD_DIM) != 0;
                    ds4_gpu_tensor_free(index_row_view);
                }
            }
            if (ok && emit) {
                g->layer_n_index_comp[il]++;
            }
            if (ok && g->layer_n_comp[il] > DS4_N_INDEXER_TOP_K &&
                g->layer_n_index_comp[il] > DS4_N_INDEXER_TOP_K) {
              const uint64_t indexer_q_dim =
                  (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
              if (!layer->indexer_attn_q_b ||
                  !ds4_graph_f16_weight_type(layer->indexer_attn_q_b->type) ||
                  layer->indexer_attn_q_b->dim[0] != q_rank ||
                  layer->indexer_attn_q_b->dim[1] != indexer_q_dim ||
                  !layer->indexer_proj ||
                  !ds4_graph_f16_weight_type(layer->indexer_proj->type) ||
                  layer->indexer_proj->dim[0] != DS4_N_EMBD ||
                  layer->indexer_proj->dim[1] != DS4_N_INDEXER_HEAD) {
                fprintf(stderr,
                        "ds4: ROCm graph indexer projections have "
                        "unexpected shapes\n");
                ok = false;
              }
              if (ok) {
                ok = rocm_graph_matmul_hc_fn_tensor(
                         g->indexer_q, model, layer->indexer_attn_q_b,
                         q_rank, indexer_q_dim, qr_norm, 1);
              }
              if (ok) {
                ok = ds4_gpu_rope_tail_tensor(
                         g->indexer_q, 1, DS4_N_INDEXER_HEAD,
                         DS4_N_INDEXER_HEAD_DIM, DS4_N_ROT, pos,
                         compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, false,
                         freq_base, freq_scale, ext_factor, attn_factor,
                         DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
              }
              if (ok) {
                ok = ds4_gpu_dsv4_indexer_qat_tensor(
                         g->indexer_q, DS4_N_INDEXER_HEAD,
                         DS4_N_INDEXER_HEAD_DIM) != 0;
              }
              if (ok) {
                ok = rocm_graph_matmul_hc_fn_tensor(
                         g->indexer_weights, model, layer->indexer_proj,
                         DS4_N_EMBD, DS4_N_INDEXER_HEAD, attn_norm, 1);
              }
              const float index_scale =
                  1.0f /
                  sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
              if (ok) {
                ok = rocm_graph_reserve_indexer_scores(
                         g, g->layer_n_index_comp[il], 1) &&
                     ds4_gpu_indexer_score_one_tensor(
                         g->indexer_scores, g->indexer_q, g->indexer_weights,
                         g->layer_index_comp_cache[il],
                         g->layer_n_index_comp[il], DS4_N_INDEXER_HEAD,
                         DS4_N_INDEXER_HEAD_DIM, index_scale) != 0;
              }
              if (ok) {
                ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                 g->indexer_scores,
                                                 g->layer_n_index_comp[il], 1,
                                                 DS4_N_INDEXER_TOP_K) != 0;
              }
              if (ok) {
                comp_selected = g->comp_selected;
                n_selected = DS4_N_INDEXER_TOP_K < g->layer_n_index_comp[il]
                                 ? DS4_N_INDEXER_TOP_K
                                 : g->layer_n_index_comp[il];
              }
            }
        }

        n_comp = g->layer_n_comp[rocm_graph_comp_src(il)];
        comp_cache = g->layer_attn_comp_cache[rocm_graph_comp_src(il)];
        if (getenv("DS4_NO_COMP")) n_comp = 0;
    }

    if (ok) {
        const uint32_t raw_start =
            rocm_graph_raw_start_for_span(g, pos, n_raw);
        if (n_comp != 0 && comp_selected != NULL && n_selected != 0) {
          ok =
              ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                  heads, model->map, model->size, layer->attn_sinks->abs_offset,
                  q, raw_cache, g->layer_attn_comp_cache[il],
                  rocm_graph_attn_comp_cache_is_f16(), comp_selected, 1, pos,
                  n_raw, g->raw_cap, raw_start, n_comp, n_selected,
                  g->raw_window, ds4_layer_compress_ratio(il), DS4_N_HEAD,
                  DS4_N_HEAD_DIM) != 0;
        } else {
          ok = ds4_gpu_attention_decode_heads_tensor(
                   heads, model->map, model->size,
                   layer->attn_sinks->abs_offset, q, raw_cache, n_raw,
                   g->raw_cap, raw_start, n_comp ? comp_cache : NULL,
                   rocm_graph_attn_comp_cache_is_f16(), n_comp, NULL, 0,
                   DS4_N_HEAD, DS4_N_HEAD_DIM) != 0;
        }
    }
    if (ok) {
        ok = ds4_gpu_rope_tail_tensor(
                 heads,
                 1,
                 DS4_N_HEAD,
                 DS4_N_HEAD_DIM,
                 DS4_N_ROT,
                 pos,
                 compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                 true,
                 freq_base,
                 freq_scale,
                 ext_factor,
                 attn_factor,
                 DS4_ROPE_YARN_BETA_FAST,
                 DS4_ROPE_YARN_BETA_SLOW) != 0;
    }
    return ok;
}

static bool rocm_graph_encode_layer_attention_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens,
        bool                    front_ready = false,
        bool                    front_only = false,
        bool                    project_qb = true,
        bool                    project_q_a_kv = true,
        bool                    compressor_projection_ready = false,
        bool                    indexer_query_projection_ready = false,
        bool                    project_output = true) {
    if (getenv("DS4_CTRACE") && il <= 1u) {
        static long seen[256]; static int ns = 0;
        long k = ((long)n_tokens << 20) ^ ((long)pos0 << 8) ^ il;
        bool dup = false;
        for (int z = 0; z < ns; ++z) if (seen[z] == k) dup = true;
        if (!dup && ns < 256) { seen[ns++] = k;
            fprintf(stderr, "ds4: [ctrace] il=%u pos0=%u n_tokens=%u\n", il, pos0, n_tokens); }
    }
  if (n_tokens == 0 || n_tokens > g->prefill_cap ||
      (front_ready && front_only)) {
    return false;
  }
  if (!rocm_graph_reserve_compressed_position(
          g, il, static_cast<uint64_t>(pos0) + n_tokens))
    return false;
  if (!rocm_graph_engram_at_layer(g, model, layer, il, g->batch_cur_hc,
                                  n_tokens))
    return false;
  ds4_probe_stats(g->batch_cur_hc,
                  (uint64_t)(n_tokens - 1u) * DS4_N_EMBD, DS4_N_EMBD,
                  "in_stream", il);

  /* Any published F16 activation mirror belongs to the previous layer, whose
   * buffers this layer reuses. Drop it before anything can overwrite them. */
  ds4_gpu_clear_f16_input();

  const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
  const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
  const uint64_t q_rank = layer->attn_q_a->dim[1];
  const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
  const uint32_t n_groups = DS4_N_OUT_GROUP;
  const uint32_t group_heads = DS4_N_HEAD / n_groups;
  const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
  const uint32_t rank = DS4_N_LORA_O;
  const uint32_t ratio = ds4_layer_compress_ratio(il);
  const bool compressed = ratio != 0;
  const bool zero_prefix = pos0 == 0;
  const bool index_stage_profile =
      getenv("GUFO_DEEPSEEK_ROCM_INDEXER_STAGE_PROFILE") != NULL;
  const bool layer_stage_profile =
      getenv("GUFO_DEEPSEEK_ROCM_LAYER_STAGE_PROFILE") != NULL;
  const bool q_stage_profile =
      getenv("GUFO_DEEPSEEK_ROCM_Q_STAGE_PROFILE") != NULL;
  const bool hb_dump_pre = getenv("DS4_H_DUMP") != NULL;
  double layer_stage_t0 = layer_stage_profile ? ds4_now_seconds() : 0.0;
  double q_stage_t0 = q_stage_profile ? ds4_now_seconds() : 0.0;
#define GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE(name) do { \
        if (ok && layer_stage_profile) { \
            ok = rocm_graph_layer_stage_profile_boundary("attn", (name), il, pos0, n_tokens, &layer_stage_t0); \
        } \
    } while (0)
#define GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE(name) do { \
        if (ok && q_stage_profile) { \
            ok = rocm_graph_q_stage_profile_boundary((name), il, pos0, n_tokens, &q_stage_t0); \
        } \
    } while (0)
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    auto *comp_counts = compressed
                            ? static_cast<uint32_t *>(
                                  ds4_xcalloc(n_tokens, sizeof(uint32_t)))
                            : nullptr;
    auto* index_counts =
        ratio == 4
            ? static_cast<uint32_t*>(ds4_xcalloc(n_tokens, sizeof(uint32_t)))
            : nullptr;
    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_a_view = ds4_gpu_tensor_view(
            g->batch_hc_split_a, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    /* Official Block.forward: attention collapses the streams with the PRE
     * carried from the previous layer's FFN split (identity one-hot at the
     * first layer), not with its own attn pre. */
    ds4_gpu_tensor *hc_pre_view = (il == 0u)
        ? ds4_gpu_tensor_view(g->batch_pre_onehot, 0,
                              (uint64_t)n_tokens * mix_hc * sizeof(float))
        : hc_split_view;
    ds4_gpu_tensor *attn_cur_view = ds4_gpu_tensor_view(
            g->batch_attn_cur, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *after_attn_hc_view = ds4_gpu_tensor_view(
            g->batch_after_attn_hc, 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && hc_split_a_view &&
              hc_pre_view && attn_cur_view && after_attn_hc_view;
    if (!front_ready) {
      bool hc_pre_fused = false;
      if (ok && n_tokens >= 128u &&
          layer->hc_attn_fn->type == DS4_TENSOR_F16) {
        hc_pre_fused = true;
        ok =
            ds4_gpu_hc_norm_mix_split_weighted_sum_tensor(
                attn_cur_view, hc_mix_view, hc_split_a_view, g->batch_cur_hc,
                model->map, model->size, layer->hc_attn_fn->abs_offset,
                layer->hc_attn_scale->abs_offset,
                layer->hc_attn_base->abs_offset, DS4_N_EMBD, DS4_N_HC, n_tokens,
                DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS, DS4_RMS_EPS,
                hc_pre_view, 24u) != 0;
      }
      if (ok && !hc_pre_fused)
        ok = ds4_gpu_rms_norm_plain_rows_tensor(
                 g->batch_flat_hc, g->batch_cur_hc, (uint32_t)hc_dim, n_tokens,
                 DS4_RMS_EPS) != 0;
      rocm_graph_probe_layer_tensor(g->batch_flat_hc,
                                    (uint64_t)n_tokens * hc_dim,
                                    "hc rmsflat", il);
      if (ok && !hc_pre_fused)
        ok = rocm_graph_matmul_hc_fn_tensor(hc_mix_view, model,
                                       layer->hc_attn_fn, hc_dim,
                                       mix_hc, g->batch_flat_hc, n_tokens);
      rocm_graph_probe_layer_tensor(hc_mix_view,
                                    (uint64_t)n_tokens * mix_hc,
                                    "hc mix", il);
      if (il == 0) {
        /* Host-side truth: scan the same buffers and recompute row 0. */
        ds4_gpu_synchronize();
        std::vector<float> ob((size_t)n_tokens * mix_hc);
        std::vector<float> xr((size_t)hc_dim);
        const float *wb = (const float *)((const char *)model->map +
                                          layer->hc_attn_fn->abs_offset);
        bool okr = ds4_gpu_readback_f32(hc_mix_view, 0, ob.size(), ob.data()) &&
                   ds4_gpu_readback_f32(g->batch_flat_hc, 0, xr.size(),
                                        xr.data());
        uint64_t bad = 0, first = 0;
        for (uint64_t i = 0; i < ob.size(); i++)
            if (!(ob[i] == ob[i])) { if (!bad++) first = i; }
        double s = 0.0, xn = 0.0, wn = 0.0;
        for (uint64_t i = 0; i < hc_dim; i++) {
            s += (double)wb[i] * (double)xr[i];
            xn += (double)(xr[i] == xr[i]);
            wn += (double)(wb[i] == wb[i]);
        }
        fprintf(stderr,
                "ds4: hc_pre HOST okr=%d bad=%llu first=%llu out0=%g "
                "recompute0=%g xfinite=%g/%llu wfinite=%g/%llu\n",
                (int)okr, (unsigned long long)bad, (unsigned long long)first,
                (double)ob[0], s, xn, (unsigned long long)hc_dim, wn,
                (unsigned long long)hc_dim);
      }
      if (!hc_pre_fused) {
        if (ok)
          ok = ds4_gpu_hc_split_weighted_sum_tensor(
                   attn_cur_view, hc_split_a_view, hc_mix_view, g->batch_cur_hc,
                   model->map, model->size, layer->hc_attn_scale->abs_offset,
                   layer->hc_attn_base->abs_offset, DS4_N_EMBD, DS4_N_HC,
                   DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS,
                   hc_pre_view, 24u) != 0;
      }
      GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("hc_pre");
      if (ok)
        ok = ds4_gpu_rms_norm_weight_rows_tensor(
                 g->batch_attn_norm, g->batch_attn_cur, model->map, model->size,
                 layer->attn_norm->abs_offset, DS4_N_EMBD, n_tokens,
                 DS4_RMS_EPS) != 0;
      /* Eight projections in a ratio-4 layer read these rows, and each F16
       * route among them was converting them again. One mirror serves all of
       * them; the conversion is row-local, so every consumer sees the bytes it
       * would have produced itself. Declining is not an error. */
      if (ok && n_tokens >= 128u) { /* DS4_ROCM_WIDE_PREFILL_ROWS */
        (void)ds4_gpu_publish_f16_input_tensor(g->batch_attn_norm,
                                                (uint64_t)n_tokens * DS4_N_EMBD);
      }
    GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("norm");
    rocm_graph_probe_layer_tensor(g->batch_attn_cur,
                                  (uint64_t)n_tokens * DS4_N_EMBD,
                                  "attn cur", il);
    rocm_graph_probe_layer_tensor(g->batch_attn_norm,
                                  (uint64_t)n_tokens * DS4_N_EMBD,
                                  "attn norm", il);
    if (project_q_a_kv) {
    GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE("pre_q");
    if (ok)
      ok = rocm_graph_matmul_v41_dense_tensor(
          g->batch_qr, model, layer->attn_q_a, DS4_N_EMBD, q_rank,
          g->batch_attn_norm, n_tokens);
    GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE("q_a");
    {
      if (ok)
        ok = rocm_graph_matmul_v41_dense_tensor(
            g->batch_kv_raw, model, layer->attn_kv, DS4_N_EMBD,
            DS4_N_HEAD_DIM, g->batch_attn_norm, n_tokens);
      if (ok)
        ok =
            ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
                g->batch_qr_norm, g->batch_qr, model->map, model->size,
                layer->attn_q_a_norm->abs_offset, (uint32_t)q_rank, g->batch_kv,
                g->batch_kv_raw, layer->attn_kv_a_norm->abs_offset,
                DS4_N_HEAD_DIM, n_tokens, DS4_RMS_EPS) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE("q_a_norm");
    }
    if (ok && project_qb) {
        ok = rocm_graph_matmul_v41_dense_tensor(g->batch_q, model,
                                                layer->attn_q_b, q_rank, q_dim,
                                                g->batch_qr_norm, n_tokens);
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE("q_b");
    }
    if (front_only) {
        ds4_gpu_tensor_free(after_attn_hc_view);
        ds4_gpu_tensor_free(attn_cur_view);
        ds4_gpu_tensor_free(hc_split_view);
        ds4_gpu_tensor_free(hc_split_a_view);
        if (hc_pre_view != hc_split_view) ds4_gpu_tensor_free(hc_pre_view);
        ds4_gpu_tensor_free(hc_mix_view);
        free(index_counts);
        free(comp_counts);
        return ok;
    }
    /* One pass over the query rows instead of two. The fused kernel stages each
     * row in LDS, so the norm's write-back is no longer read again by the rope
     * pass; it is bit-identical and falls back when the row does not fit. */
    bool q_norm_rope_fused =
        ok && n_tokens > 6u &&
        ds4_gpu_head_rms_norm_rope_tail_tensor(
            g->batch_q, n_tokens, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos0,
            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, false, freq_base,
            freq_scale, ext_factor, attn_factor, DS4_ROPE_YARN_BETA_FAST,
            DS4_ROPE_YARN_BETA_SLOW, DS4_RMS_EPS) != 0;
    if (ok && !q_norm_rope_fused)
      ok = ds4_gpu_head_rms_norm_tensor(g->batch_q, n_tokens, DS4_N_HEAD,
                                        DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
    GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE("head_norm");
    if (ok && !q_norm_rope_fused)
      ok = ds4_gpu_rope_tail_tensor(
               g->batch_q, n_tokens, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT,
               pos0, compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, false,
               freq_base, freq_scale, ext_factor, attn_factor,
               DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE("rope");
    GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("q_path");
    rocm_graph_probe_layer_tensor(g->batch_q,
                                  (uint64_t)n_tokens * DS4_N_HEAD *
                                      DS4_N_HEAD_DIM,
                                  "q roped", il);
    rocm_graph_probe_layer_tensor(g->batch_qr_norm,
                                  (uint64_t)n_tokens * DS4_N_LORA_Q,
                                  "qr norm", il);

    if (ok)
      ok = ds4_gpu_rope_tail_tensor(
               g->batch_kv, n_tokens, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT,
               pos0, compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, false,
               freq_base, freq_scale, ext_factor, attn_factor,
               DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok)
      ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(g->batch_kv, n_tokens,
                                               DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
    GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("kv_path");
    rocm_graph_probe_layer_tensor(g->batch_kv,
                                  (uint64_t)n_tokens * DS4_N_HEAD_DIM,
                                  "kv", il);
    if (ok && getenv("DS4_H_DUMP") && il == 0u) {
        const uint32_t lr = n_tokens - 1u;
        struct { ds4_gpu_tensor *t; uint64_t st; const char *nm; } hbi[] = {
            {g->batch_qr_norm, 1280ull, "qrn"},
            {g->batch_q, 32768ull, "q"},
            {g->batch_kv, (uint64_t)DS4_N_HEAD_DIM, "kv"}};
        for (auto &e : hbi) {
            if (!e.t) continue;
            char hbp[64];
            snprintf(hbp, sizeof hbp, "/tmp/hb_%s_L00.bin", e.nm);
            ds4_gpu_tensor *v =
                ds4_gpu_tensor_view(e.t, (uint64_t)lr * e.st * 4u, e.st * 4u);
            if (v) ds4_dbg_dump(v, hbp, (uint32_t)e.st);
        }
    }
    /*
     * Static graph order is q, kv, cpy_k(raw SWA), then attention. For a
     * zero-prefix batch it is safe to store the whole batch at once: attention
     * reads the contiguous batch KV, and the ring only has to end with the last
     * SWA rows for later chunks/decode. For nonzero chunks the physical ring is
     * sized to hold the current chunk plus the previous SWA window, while the
     * attention mask still enforces the 128-token logical window.
     */
    if (ok && zero_prefix) ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                                    g->batch_kv,
                                                                    g->raw_cap,
                                                                    pos0,
                                                                    n_tokens,
                                                                    DS4_N_HEAD_DIM) != 0;
    const bool raw_batch_attention = zero_prefix && ratio == 0;
    bool batch_attention_done = false;

    if (ok && raw_batch_attention) {
        ok = ds4_gpu_attention_prefill_raw_heads_tensor(g->batch_heads,
                                                          model->map,
                                                          model->size,
                                                          layer->attn_sinks->abs_offset,
                                                          g->batch_q,
                                                          g->batch_kv,
                                                          n_tokens,
                                                          g->raw_window,
                                                          DS4_N_HEAD,
                                                          DS4_N_HEAD_DIM) != 0;
        if (ok) batch_attention_done = true;
    } else if (ok && !zero_prefix && ratio == 0 && n_tokens <= g->raw_cap) {
        /*
         * The ubatch path stores the whole batch in the SWA cache, then runs
         * one batched attention kernel with an absolute-position causal/window
         * mask.  This avoids mixing prefill with the different single-token
         * attention path.
         */
        const uint32_t n_raw = rocm_graph_raw_span_for_batch(g, pos0, n_tokens);
        /* Nonzero prompt chunks read the SWA cache as a ring.  FlashAttention
         * receives a linearized window starting at raw_start, not physical row
         * zero; otherwise wrapped chunks silently miss recent raw keys. */
        const uint32_t raw_start = rocm_graph_raw_start_for_span(g,
                                                                  pos0 + n_tokens - 1u,
                                                                  n_raw);
        ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                 g->batch_kv,
                                                 g->raw_cap,
                                                 pos0,
                                                 n_tokens,
                                                 DS4_N_HEAD_DIM) != 0;
        if (ok) {
          ok = ds4_gpu_attention_decode_raw_batch_heads_tensor(
                   g->batch_heads, model->map, model->size,
                   layer->attn_sinks->abs_offset, g->batch_q,
                   g->layer_raw_cache[il], n_tokens, pos0, n_raw, g->raw_cap,
                   raw_start, g->raw_window, DS4_N_HEAD, DS4_N_HEAD_DIM) != 0;
        }
        if (ok) batch_attention_done = true;
    } else if (ok && ratio != 0) {
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const uint32_t comp_src = rocm_graph_comp_src(il);
        const bool ds4_comp_producer = ds4_layer_is_kv_source(il);
        /* V4.1 source sharing (prefill mirror of the decode path). */
        uint32_t pf_cand_layer = 0, pf_cand_blocks = 0, pf_cand_block_size = 0;
        ds4_cand_params(&pf_cand_layer, &pf_cand_blocks, &pf_cand_block_size);
        const bool pf_v41 =
            pf_cand_block_size != 0u && pf_cand_blocks != 0u && !zero_prefix;
        uint32_t pf_chunk_rows = 0;   /* compressed rows this chunk pools */
        ds4_gpu_tensor *pf_latent_view = nullptr;
        const bool have_attn_comp = layer->attn_compressor_kv && layer->attn_compressor_norm;
        if (ok && ds4_comp_producer && !have_attn_comp) {
            fprintf(stderr,
                    "ds4: ROCm layer-major prefill needs attention compressor "
                    "weights at layer %u\n", il);
            ok = false;
        }
        if (ok && ds4_comp_producer && !compressor_projection_ready) {
            ok = rocm_graph_matmul_weight_tensor(g->batch_comp_kv, model,
                                                 layer->attn_compressor_kv,
                                                 DS4_N_EMBD, comp_width,
                                                 g->batch_attn_norm, n_tokens);
            if (ok && layer->attn_compressor_gate) {
                ok = rocm_graph_matmul_weight_tensor(g->batch_comp_sc, model,
                                                     layer->attn_compressor_gate,
                                                     DS4_N_EMBD, comp_width,
                                                     g->batch_attn_norm, n_tokens);
            } else if (ok) {
                /* No gate projection (ratio-1 producers): softmax over a
                 * single state row is 1.0, so a zero score stream is exact. */
                ok = rocm_tensor_fill_f32(g->batch_comp_sc, 0.0f,
                                          (uint64_t)n_tokens * comp_width);
            }
        }
        uint32_t n_comp = g->layer_n_comp[comp_src];
        if (getenv("DS4_NO_COMP")) n_comp = 0;
        if (zero_prefix) {
            n_comp = n_tokens / ratio;
            if (ok && ds4_comp_producer && n_comp > g->layer_comp_cap[il]) {
                fprintf(stderr, "ds4: ROCm layer-major compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            ds4_gpu_tensor *attn_comp_target = nullptr;
            ds4_gpu_tensor *attn_comp_mirror = nullptr;
            if (ok && ds4_comp_producer) {
                attn_comp_target = rocm_graph_attn_comp_prefill_target(g, il, 0, n_comp);
                if (ratio == 4) {
                    attn_comp_mirror =
                        rocm_graph_attn_comp_prefill_mirror(g, il, 0, n_comp);
                }
                ok = attn_comp_target != nullptr &&
                     (ratio != 4 || attn_comp_mirror != nullptr) &&
                     ds4_gpu_compressor_prefill_tensor(attn_comp_target,
                                                         g->layer_attn_state_kv[il],
                                                         g->layer_attn_state_score[il],
                                                         g->batch_comp_kv,
                                                         g->batch_comp_sc,
                                                         model->map,
                                                         model->size,
                                                         rocm_graph_opt_w_off(layer->attn_compressor_ape),
                                                         rocm_graph_opt_w_type(layer->attn_compressor_ape),
                                                         layer->attn_compressor_norm->abs_offset,
                                                         layer->attn_compressor_norm->type,
                                                         DS4_N_HEAD_DIM,
                                                         ratio,
                                                         pos0,
                                                         n_tokens,
                                                         DS4_N_ROT,
                                                         compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                         true,
                                                         freq_base,
                                                         freq_scale,
                                                         ext_factor,
                                                         attn_factor,
                                                         DS4_ROPE_YARN_BETA_FAST,
                                                         DS4_ROPE_YARN_BETA_SLOW,
                                                         DS4_RMS_EPS,
                                                         attn_comp_mirror,
                                                         NULL) != 0;
                if (ok && ratio == 4) {
                    ok = rocm_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_attn_state_kv[il],
                                                                     g->layer_attn_state_score[il],
                                                                     layer->attn_compressor_kv,
                                                                     layer->attn_compressor_gate,
                                                                     layer->attn_compressor_ape,
                                                                     DS4_N_HEAD_DIM,
                                                                     comp_width,
                                                                     pos0,
                                                                     n_tokens);
                }
            }
            if (ok) {
                if (ds4_comp_producer)
                    g->layer_n_comp[il] = n_comp;
                for (uint32_t t = 0; t < n_tokens; t++) {
                    comp_counts[t] = (pos0 + t + 1u) / ratio;
                }
                if (n_comp != 0) {
                }
            }
            rocm_graph_attn_comp_prefill_target_free(attn_comp_mirror);
            rocm_graph_attn_comp_prefill_target_free(attn_comp_target);
        } else {
            // Verification may reject an aligned block partway through.
            // The bulk compressor only saves its final state, so use the
            // row loop when rollback needs intermediate prefix snapshots.
            const bool aligned_chunk =
                !g->spec_capture_prefixes &&
                (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
            if (aligned_chunk) {
                const uint32_t comp_before = g->layer_n_comp[il];
                const uint32_t comp_chunk = n_tokens / ratio;
                if (comp_before + comp_chunk > g->layer_comp_cap[il]) {
                    fprintf(stderr, "ds4: ROCm graph compressed KV cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                ds4_gpu_tensor *attn_comp_target = ok && ds4_comp_producer
                    ? rocm_graph_attn_comp_prefill_target(
                          g, il, comp_before, comp_chunk)
                    : nullptr;
                ds4_gpu_tensor *attn_comp_mirror = ok && ds4_comp_producer && ratio == 4
                    ? rocm_graph_attn_comp_prefill_mirror(
                          g, il, comp_before, comp_chunk)
                    : nullptr;
                if (ok && ds4_comp_producer &&
                    (!attn_comp_target || (ratio == 4 && !attn_comp_mirror))) {
                  ok = false;
                }
                /* V4.1: stage the whole chunk's pre-RoPE pooled latent. */
                if (ok && pf_v41 && ds4_comp_producer && comp_chunk != 0u) {
                    if (!rocm_graph_reserve_comp_latent(
                            g, (uint64_t)comp_chunk * DS4_N_HEAD_DIM))
                        ok = false;
                    else
                        pf_latent_view = ds4_gpu_tensor_view(
                            g->comp_latent, 0,
                            (uint64_t)comp_chunk * DS4_N_HEAD_DIM *
                                sizeof(float));
                    if (ok && !pf_latent_view) ok = false;
                }
                if (ok && ds4_comp_producer && ratio == 4) {
                    ok = ds4_gpu_compressor_prefill_replay_tensor(
                            attn_comp_target,
                            g->layer_attn_state_kv[il],
                            g->layer_attn_state_score[il],
                            g->batch_comp_kv,
                            g->batch_comp_sc,
                            model->map,
                            model->size,
                            rocm_graph_opt_w_off(layer->attn_compressor_ape),
                            rocm_graph_opt_w_type(layer->attn_compressor_ape),
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            DS4_N_HEAD_DIM,
                            ratio,
                            pos0,
                            n_tokens,
                            DS4_N_ROT,
                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                            true,
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            DS4_ROPE_YARN_BETA_FAST,
                            DS4_ROPE_YARN_BETA_SLOW,
                            DS4_RMS_EPS,
                            attn_comp_mirror,
                            pf_latent_view) != 0;
                } else if (ok && ds4_comp_producer) {
                    ok = ds4_gpu_compressor_prefill_tensor(
                            attn_comp_target,
                            g->layer_attn_state_kv[il],
                            g->layer_attn_state_score[il],
                            g->batch_comp_kv,
                            g->batch_comp_sc,
                            model->map,
                            model->size,
                            rocm_graph_opt_w_off(layer->attn_compressor_ape),
                            rocm_graph_opt_w_type(layer->attn_compressor_ape),
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            DS4_N_HEAD_DIM,
                            ratio,
                            pos0,
                            n_tokens,
                            DS4_N_ROT,
                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                            true,
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            DS4_ROPE_YARN_BETA_FAST,
                            DS4_ROPE_YARN_BETA_SLOW,
                            DS4_RMS_EPS,
                            attn_comp_mirror,
                            pf_latent_view) != 0;
                }
                if (ok && ratio == 4) {
                    ok = rocm_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_attn_state_kv[il],
                                                                     g->layer_attn_state_score[il],
                                                                     layer->attn_compressor_kv,
                                                                     layer->attn_compressor_gate,
                                                                     layer->attn_compressor_ape,
                                                                     DS4_N_HEAD_DIM,
                                                                     comp_width,
                                                                     pos0,
                                                                     n_tokens);
                }
                if (ok) {
                    if (ds4_comp_producer) {
                        g->layer_n_comp[il] = comp_before + comp_chunk;
                        /* V4.1: the pooled rows exist; derive this chunk's
                         * shared index-K rows from the staged latent. */
                        if (pf_v41 && ds4_layer_is_topk_source(il))
                            ok = rocm_graph_v41_owner_index_rows(
                                     g, model, layer, il, pos0, ratio,
                                     comp_chunk, compressed, freq_base,
                                     freq_scale, ext_factor, attn_factor);
                        pf_chunk_rows = ok ? comp_chunk : 0u;
                    }
                    if (comp_counts) {
                        for (uint32_t t = 0; t < n_tokens; t++) {
                            comp_counts[t] = (pos0 + t + 1u) / ratio;
                        }
                    }
                }
                if (pf_latent_view) {
                    ds4_gpu_tensor_free(pf_latent_view);
                    pf_latent_view = nullptr;
                }
                rocm_graph_attn_comp_prefill_target_free(attn_comp_mirror);
                rocm_graph_attn_comp_prefill_target_free(attn_comp_target);
            } else if (!ds4_comp_producer) {
                if (comp_counts) {
                    for (uint32_t t = 0; t < n_tokens; t++)
                        comp_counts[t] = (pos0 + t + 1u) / ratio;
                }
            } else {
              for (uint32_t t = 0; ok && t < n_tokens;) {
                const uint32_t whole =
                    g->spec_capture_prefixes || (pos0 + t) % ratio != 0u
                        ? 0u
                        : (n_tokens - t) / ratio * ratio;
                if (whole != 0u) {
                  const uint32_t comp_row = g->layer_n_comp[il];
                  const uint32_t n_new = whole / ratio;
                  if (comp_row + n_new > g->layer_comp_cap[il]) {
                    fprintf(stderr,
                            "ds4: ROCm graph compressed KV cache capacity "
                            "exceeded at layer %u\n",
                            il);
                    ok = false;
                    break;
                  }
                  ds4_gpu_tensor* kv_view = ds4_gpu_tensor_view(
                      g->batch_comp_kv, (uint64_t)t * comp_width * sizeof(float),
                      (uint64_t)whole * comp_width * sizeof(float));
                  ds4_gpu_tensor* sc_view = ds4_gpu_tensor_view(
                      g->batch_comp_sc, (uint64_t)t * comp_width * sizeof(float),
                      (uint64_t)whole * comp_width * sizeof(float));
                  ds4_gpu_tensor* target =
                      rocm_graph_attn_comp_prefill_target(g, il, comp_row, n_new);
                  ds4_gpu_tensor* mirror =
                      rocm_graph_attn_comp_prefill_mirror(g, il, comp_row, n_new);
                  /* V4.1: stage this bulk run's pre-RoPE latent. */
                  ds4_gpu_tensor* pf_run_latent = nullptr;
                  if (ok && pf_v41 && n_new != 0u) {
                      if (!rocm_graph_reserve_comp_latent(
                              g, (uint64_t)n_new * DS4_N_HEAD_DIM))
                          ok = false;
                      else
                          pf_run_latent = ds4_gpu_tensor_view(
                              g->comp_latent, 0,
                              (uint64_t)n_new * DS4_N_HEAD_DIM *
                                  sizeof(float));
                      if (ok && !pf_run_latent) ok = false;
                  }
                  ok = kv_view && sc_view && target &&
                       ds4_gpu_compressor_prefill_replay_tensor(
                           target, g->layer_attn_state_kv[il],
                           g->layer_attn_state_score[il], kv_view, sc_view,
                           model->map, model->size,
                           rocm_graph_opt_w_off(layer->attn_compressor_ape),
                           rocm_graph_opt_w_type(layer->attn_compressor_ape),
                           layer->attn_compressor_norm->abs_offset,
                           layer->attn_compressor_norm->type, DS4_N_HEAD_DIM,
                           ratio, pos0 + t, whole, DS4_N_ROT,
                           compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, true,
                           freq_base, freq_scale, ext_factor, attn_factor,
                           DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW,
                           DS4_RMS_EPS, mirror, pf_run_latent) != 0;
                  if (ok && pf_v41 && ds4_layer_is_topk_source(il))
                    ok = rocm_graph_v41_owner_index_rows(
                             g, model, layer, il, pos0 + t, ratio, n_new,
                             compressed, freq_base, freq_scale, ext_factor,
                             attn_factor);
                  if (pf_run_latent) ds4_gpu_tensor_free(pf_run_latent);
                  if (ok) g->layer_n_comp[il] = comp_row + n_new;
                  if (comp_counts) {
                    for (uint32_t i = 0; i < whole; ++i)
                      comp_counts[t + i] = comp_row + (i + 1u) / ratio;
                  }
                  rocm_graph_attn_comp_prefill_target_free(mirror);
                  rocm_graph_attn_comp_prefill_target_free(target);
                  ds4_gpu_tensor_free(sc_view);
                  ds4_gpu_tensor_free(kv_view);
                  t += whole;
                  continue;
                }
                // A store span never crosses a pool/shift boundary.
                // Verification retains every intermediate state for
                // rejected-prefix rollback.
                const uint32_t rows =
                    g->spec_capture_prefixes
                        ? 1u
                        : std::min(n_tokens - t, ratio - (pos0 + t) % ratio);
                const uint32_t pos = pos0 + t + rows - 1u;
                const bool emit = ((pos + 1u) % ratio) == 0u;
                if (emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
                  fprintf(stderr,
                          "ds4: ROCm graph compressed KV cache capacity "
                          "exceeded at layer %u\n",
                          il);
                  ok = false;
                  break;
                }
                ds4_gpu_tensor* kv_view = ds4_gpu_tensor_view(
                    g->batch_comp_kv, (uint64_t)t * comp_width * sizeof(float),
                    (uint64_t)rows * comp_width * sizeof(float));
                ds4_gpu_tensor* sc_view = ds4_gpu_tensor_view(
                    g->batch_comp_sc, (uint64_t)t * comp_width * sizeof(float),
                    (uint64_t)rows * comp_width * sizeof(float));
                const uint32_t comp_row = g->layer_n_comp[il];
                ok = kv_view && sc_view;
                if (ok && rows > 1u) {
                  ok = ds4_gpu_compressor_store_batch_tensor(
                           kv_view, sc_view, g->layer_attn_state_kv[il],
                           g->layer_attn_state_score[il], model->map,
                           model->size, rocm_graph_opt_w_off(layer->attn_compressor_ape),
                           rocm_graph_opt_w_type(layer->attn_compressor_ape), DS4_N_HEAD_DIM,
                           ratio, pos0 + t, rows) != 0;
                }
                ok =
                    ok &&
                    ds4_gpu_compressor_update_tensor(
                        kv_view, sc_view, g->layer_attn_state_kv[il],
                        g->layer_attn_state_score[il],
                        rocm_graph_attn_comp_update_target(g, il), model->map,
                        model->size, rocm_graph_opt_w_off(layer->attn_compressor_ape),
                        rocm_graph_opt_w_type(layer->attn_compressor_ape),
                        layer->attn_compressor_norm->abs_offset,
                        layer->attn_compressor_norm->type, DS4_N_HEAD_DIM,
                        ratio, pos, rocm_graph_attn_comp_update_row(comp_row),
                        DS4_N_ROT, compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                        freq_base, freq_scale, ext_factor, attn_factor,
                        DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW,
                        DS4_RMS_EPS, rows > 1u, false, false) != 0;
                if (ok && emit) {
                  ds4_gpu_tensor* comp_row_view =
                      rocm_graph_attn_comp_row_view(g, il, comp_row);
                  ok = comp_row_view && rocm_graph_quantize_attn_comp_row(
                                            g, il, comp_row, comp_row_view);
                  ds4_gpu_tensor_free(comp_row_view);
                }
                if (ok && emit)
                  g->layer_n_comp[il]++;
                if (comp_counts) {
                  for (uint32_t row = t; row < t + rows; ++row)
                    comp_counts[row] =
                        comp_row + uint32_t(emit && row + 1u == t + rows);
                }
                if (ok && g->spec_capture_prefixes && t + 1u < n_tokens &&
                    t < DS4_SPEC_PREFIX_SLOTS) {
                  ok = rocm_graph_capture_prefix_attn_state(g, il, t);
                }
                ds4_gpu_tensor_free(sc_view);
                ds4_gpu_tensor_free(kv_view);
                t += rows;
              }
            }
            n_comp = g->layer_n_comp[comp_src];
        }
        GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("compressor");

        /* ----------------------------------------------------------------
         * V4.1 indexer publisher (prefill mirror of decode): every
         * top-k source layer scores its chunk against the OWNER's shared
         * index cache and publishes one per-token top-k for this chunk;
         * readers whose kv_source matches the publisher reuse it
         * (reference kv_source/topk_source lists).  The candidate mask
         * published at candidate_source_layer restricts every later
         * publisher, exactly like the decode two-level selection.
         * ---------------------------------------------------------------- */
        bool pf_sel = false;
        if (ok && pf_v41 && ds4_pf_sel_env() && ds4_layer_is_topk_source(il)) {
            const uint32_t pf_idx_kv = ds4_layer_kv_source(il);
            const uint32_t pf_n_idx = g->layer_n_index_comp[pf_idx_kv];
            g->pf_sel_owner = UINT32_MAX;
            if (n_comp == 0) n_comp = pf_n_idx;
            if (pf_n_idx > (uint32_t)DS4_N_INDEXER_TOP_K) {
                const uint64_t pf_q_dim =
                    (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
                ok = ok && layer->indexer_attn_q_b && layer->indexer_proj &&
                     g->layer_index_comp_cache[pf_idx_kv] != nullptr;
                if (ok)
                    ok = rocm_graph_matmul_hc_fn_tensor(
                             g->batch_indexer_q, model, layer->indexer_attn_q_b,
                             q_rank, pf_q_dim, g->batch_qr_norm, n_tokens);
                if (ok)
                    ok = ds4_gpu_rope_tail_tensor(
                             g->batch_indexer_q, n_tokens, DS4_N_INDEXER_HEAD,
                             DS4_N_INDEXER_HEAD_DIM, DS4_N_ROT, pos0,
                             compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                             false, freq_base, freq_scale, ext_factor,
                             attn_factor, DS4_ROPE_YARN_BETA_FAST,
                             DS4_ROPE_YARN_BETA_SLOW) != 0;
                if (ok)
                    ok = ds4_gpu_dsv4_indexer_qat_tensor(
                             g->batch_indexer_q,
                             (uint64_t)n_tokens * DS4_N_INDEXER_HEAD,
                             DS4_N_INDEXER_HEAD_DIM) != 0;
                if (ok)
                    ok = rocm_graph_matmul_hc_fn_tensor(
                             g->batch_indexer_weights, model,
                             layer->indexer_proj, DS4_N_EMBD,
                             DS4_N_INDEXER_HEAD, g->batch_attn_norm,
                             n_tokens);
                const float pf_iscale =
                    1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM *
                                         DS4_N_INDEXER_HEAD));
                if (ok)
                    ok = rocm_graph_reserve_indexer_scores(g, pf_n_idx,
                                                           n_tokens) &&
                         ds4_gpu_indexer_scores_decode_batch_tensor(
                             g->indexer_scores, g->batch_indexer_q,
                             g->batch_indexer_weights,
                             g->layer_index_comp_cache[pf_idx_kv], pf_n_idx,
                             n_tokens, pos0, DS4_N_INDEXER_HEAD,
                             DS4_N_INDEXER_HEAD_DIM,
                             ds4_layer_compress_ratio(pf_idx_kv), pf_iscale,
                             g->batch_heads) != 0;
                if (ok && il == pf_cand_layer) {
                    ok = rocm_graph_reserve_pf_cand_mask(
                             g, (uint64_t)n_tokens * pf_n_idx) &&
                         ds4_gpu_indexer_candidate_mask_batch_tensor(
                             g->indexer_scores, pf_n_idx, n_tokens, pos0,
                             ds4_layer_compress_ratio(pf_idx_kv),
                             pf_cand_block_size, pf_cand_blocks,
                             g->pf_cand_mask) != 0;
                    if (ok) {
                        g->pf_cand_n = pf_n_idx;
                        g->pf_cand_valid = true;
                    }
                }
                if (ok && g->pf_cand_valid && g->pf_cand_n >= pf_n_idx)
                    ok = ds4_gpu_indexer_apply_mask_batch_tensor(
                             g->indexer_scores, pf_n_idx, n_tokens,
                             g->pf_cand_mask) != 0;
                if (ok)
                    ok = ds4_gpu_indexer_topk_tensor(
                             g->comp_selected, g->indexer_scores, pf_n_idx,
                             n_tokens, DS4_N_INDEXER_TOP_K) != 0;
                if (ok)
                    ok = ds4_gpu_indexer_topk_drop_neginf_tensor(
                             g->comp_selected, g->indexer_scores, pf_n_idx,
                             n_tokens, DS4_N_INDEXER_TOP_K) != 0;
                if (ok) {
                    g->pf_sel_owner = pf_idx_kv;
                    g->pf_ratio = ds4_layer_compress_ratio(pf_idx_kv);
                    pf_sel = true;
                }
            }
        }
        if (ok && pf_v41 && ds4_pf_sel_env() &&
            !ds4_layer_is_topk_source(il) &&
            g->pf_sel_owner != UINT32_MAX &&
            g->pf_sel_owner == ds4_layer_kv_source(il))
            pf_sel = true;

        if (ok && ratio == 4) {
            const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv ||
                !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape ||
                !layer->indexer_compressor_norm || !layer->indexer_attn_q_b ||
                !layer->indexer_proj) {
              fprintf(stderr,
                      "ds4: ROCm layer-major prefill needs indexer weights\n");
              ok = false;
            }
            if (ok) {
                ok = rocm_graph_matmul_hc_fn_tensor(g->batch_comp_kv,
                                                 model,
                                                 layer->indexer_compressor_kv,
                                                 DS4_N_EMBD,
                                                 index_width,
                                                 g->batch_attn_norm,
                                                 n_tokens);
                if (ok) ok = rocm_graph_matmul_hc_fn_tensor(g->batch_comp_sc,
                                                         model,
                                                         layer->indexer_compressor_gate,
                                                         DS4_N_EMBD,
                                                         index_width,
                                                         g->batch_attn_norm,
                                                         n_tokens);
            }
            if (ok && !indexer_query_projection_ready) {
                ok = rocm_graph_matmul_hc_fn_tensor(g->batch_indexer_q,
                                                     model,
                                                     layer->indexer_attn_q_b,
                                                     q_rank,
                                                     (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                                     g->batch_qr_norm,
                                                     n_tokens) != 0;
            }
            if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_indexer_q,
                                                    n_tokens,
                                                    DS4_N_INDEXER_HEAD,
                                                    DS4_N_INDEXER_HEAD_DIM,
                                                    DS4_N_ROT,
                                                    pos0,
                                                    compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                    false,
                                                    freq_base,
                                                    freq_scale,
                                                    ext_factor,
                                                    attn_factor,
                                                    DS4_ROPE_YARN_BETA_FAST,
                                                    DS4_ROPE_YARN_BETA_SLOW) != 0;
            if (ok) ok = ds4_gpu_dsv4_indexer_qat_tensor(g->batch_indexer_q,
                                                          n_tokens * DS4_N_INDEXER_HEAD,
                                                          DS4_N_INDEXER_HEAD_DIM) != 0;
            if (ok && !indexer_query_projection_ready) {
                ok = rocm_graph_matmul_hc_fn_tensor(g->batch_indexer_weights,
                                                     model,
                                                     layer->indexer_proj,
                                                     DS4_N_EMBD,
                                                     DS4_N_INDEXER_HEAD,
                                                     g->batch_attn_norm,
                                                     n_tokens) != 0;
            }
            if (zero_prefix) {
                if (ok && n_comp > g->layer_comp_cap[il]) {
                    fprintf(stderr, "ds4: ROCm layer-major indexer cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                if (ok) {
                    ok = ds4_gpu_compressor_prefill_tensor(g->layer_index_comp_cache[il],
                                                             g->layer_index_state_kv[il],
                                                             g->layer_index_state_score[il],
                                                             g->batch_comp_kv,
                                                             g->batch_comp_sc,
                                                             model->map,
                                                             model->size,
                                                             layer->indexer_compressor_ape->abs_offset,
                                                             layer->indexer_compressor_ape->type,
                                                             layer->indexer_compressor_norm->abs_offset,
                                                             layer->indexer_compressor_norm->type,
                                                             DS4_N_INDEXER_HEAD_DIM,
                                                             ratio,
                                                             pos0,
                                                             n_tokens,
                                                             DS4_N_ROT,
                                                             compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                             false,
                                                             freq_base,
                                                             freq_scale,
                                                             ext_factor,
                                                             attn_factor,
                                                             DS4_ROPE_YARN_BETA_FAST,
                                                             DS4_ROPE_YARN_BETA_SLOW,
                                                             DS4_RMS_EPS,
                                                             nullptr,
                                                              nullptr) != 0;
                }
                if (ok && n_comp != 0) {
                    ok = ds4_gpu_dsv4_indexer_qat_tensor(g->layer_index_comp_cache[il],
                                                          n_comp,
                                                          DS4_N_INDEXER_HEAD_DIM) != 0;
                }
                if (ok) {
                    ok = rocm_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_index_state_kv[il],
                                                                     g->layer_index_state_score[il],
                                                                     layer->indexer_compressor_kv,
                                                                     layer->indexer_compressor_gate,
                                                                     layer->indexer_compressor_ape,
                                                                     DS4_N_INDEXER_HEAD_DIM,
                                                                     index_width,
                                                                     pos0,
                                                                     n_tokens);
                }
                if (ok) {
                    g->layer_n_index_comp[il] = n_comp;
                    for (uint32_t t = 0; t < n_tokens; t++) {
                        index_counts[t] = (pos0 + t + 1u) / ratio;
                    }
                    if (n_comp != 0) {
                    }
                }
            } else {
                const bool aligned_chunk =
                    !g->spec_capture_prefixes &&
                    (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
                if (aligned_chunk) {
                    const uint32_t index_before = g->layer_n_index_comp[il];
                    const uint32_t index_chunk = n_tokens / ratio;
                    if (index_before + index_chunk > g->layer_comp_cap[il]) {
                        fprintf(stderr, "ds4: ROCm graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                        ok = false;
                    }
                    ds4_gpu_tensor *index_view = NULL;
                    if (ok) {
                        index_view = ds4_gpu_tensor_view(
                                g->layer_index_comp_cache[il],
                                (uint64_t)index_before * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                (uint64_t)index_chunk * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                        ok = index_view != NULL;
                    }
                    if (ok) {
                        ok = ds4_gpu_compressor_prefill_replay_tensor(
                                index_view,
                                g->layer_index_state_kv[il],
                                g->layer_index_state_score[il],
                                g->batch_comp_kv,
                                g->batch_comp_sc,
                                model->map,
                                model->size,
                                layer->indexer_compressor_ape->abs_offset,
                                layer->indexer_compressor_ape->type,
                                layer->indexer_compressor_norm->abs_offset,
                                layer->indexer_compressor_norm->type,
                                DS4_N_INDEXER_HEAD_DIM,
                                ratio,
                                pos0,
                                n_tokens,
                                DS4_N_ROT,
                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                false,
                                freq_base,
                                freq_scale,
                                ext_factor,
                                attn_factor,
                                DS4_ROPE_YARN_BETA_FAST,
                                DS4_ROPE_YARN_BETA_SLOW,
                                DS4_RMS_EPS,
                                nullptr,
                                                              nullptr) != 0;
                    }
                    if (ok && index_chunk != 0) {
                        ok = ds4_gpu_dsv4_indexer_qat_tensor(index_view,
                                                              index_chunk,
                                                              DS4_N_INDEXER_HEAD_DIM) != 0;
                    }
                    if (ok) {
                        ok = rocm_graph_refresh_ratio4_compressor_state(g,
                                                                         model,
                                                                         g->layer_index_state_kv[il],
                                                                         g->layer_index_state_score[il],
                                                                         layer->indexer_compressor_kv,
                                                                         layer->indexer_compressor_gate,
                                                                         layer->indexer_compressor_ape,
                                                                         DS4_N_INDEXER_HEAD_DIM,
                                                                         index_width,
                                                                         pos0,
                                                                         n_tokens);
                    }
                    if (ok) {
                        g->layer_n_index_comp[il] = index_before + index_chunk;
                        if (index_counts) {
                            for (uint32_t t = 0; t < n_tokens; t++) {
                                index_counts[t] = (pos0 + t + 1u) / ratio;
                            }
                        }
                    }
                    ds4_gpu_tensor_free(index_view);
                } else {
                  for (uint32_t t = 0; ok && t < n_tokens;) {
                    const uint32_t whole =
                        g->spec_capture_prefixes || (pos0 + t) % ratio != 0u
                            ? 0u
                            : (n_tokens - t) / ratio * ratio;
                    if (whole != 0u) {
                      const uint32_t index_row = g->layer_n_index_comp[il];
                      const uint32_t n_new = whole / ratio;
                      if (index_row + n_new > g->layer_comp_cap[il]) {
                        fprintf(stderr,
                                "ds4: ROCm graph indexer compressed KV cache "
                                "capacity exceeded at layer %u\n",
                                il);
                        ok = false;
                        break;
                      }
                      ds4_gpu_tensor* kv_view = ds4_gpu_tensor_view(
                          g->batch_comp_kv,
                          (uint64_t)t * index_width * sizeof(float),
                          (uint64_t)whole * index_width * sizeof(float));
                      ds4_gpu_tensor* sc_view = ds4_gpu_tensor_view(
                          g->batch_comp_sc,
                          (uint64_t)t * index_width * sizeof(float),
                          (uint64_t)whole * index_width * sizeof(float));
                      ds4_gpu_tensor* index_view = ds4_gpu_tensor_view(
                          g->layer_index_comp_cache[il],
                          (uint64_t)index_row * DS4_N_INDEXER_HEAD_DIM *
                              sizeof(float),
                          (uint64_t)n_new * DS4_N_INDEXER_HEAD_DIM *
                              sizeof(float));
                      ok = kv_view && sc_view && index_view &&
                           ds4_gpu_compressor_prefill_replay_tensor(
                               index_view, g->layer_index_state_kv[il],
                               g->layer_index_state_score[il], kv_view,
                               sc_view, model->map, model->size,
                               layer->indexer_compressor_ape->abs_offset,
                               layer->indexer_compressor_ape->type,
                               layer->indexer_compressor_norm->abs_offset,
                               layer->indexer_compressor_norm->type,
                               DS4_N_INDEXER_HEAD_DIM, ratio, pos0 + t, whole,
                               DS4_N_ROT,
                               compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                               false, freq_base, freq_scale, ext_factor,
                               attn_factor, DS4_ROPE_YARN_BETA_FAST,
                               DS4_ROPE_YARN_BETA_SLOW, DS4_RMS_EPS,
                               nullptr,
                                                               nullptr) != 0 &&
                           ds4_gpu_dsv4_indexer_qat_tensor(
                               index_view, n_new, DS4_N_INDEXER_HEAD_DIM) != 0;
                      if (ok) g->layer_n_index_comp[il] = index_row + n_new;
                      if (index_counts) {
                        for (uint32_t i = 0; i < whole; ++i)
                          index_counts[t + i] = index_row + (i + 1u) / ratio;
                      }
                      ds4_gpu_tensor_free(index_view);
                      ds4_gpu_tensor_free(sc_view);
                      ds4_gpu_tensor_free(kv_view);
                      t += whole;
                      continue;
                    }
                    // A store span never crosses a pool/shift boundary.
                    // Verification retains every intermediate state for
                    // rejected-prefix rollback.
                    const uint32_t rows =
                        g->spec_capture_prefixes
                            ? 1u
                            : std::min(n_tokens - t,
                                       ratio - (pos0 + t) % ratio);
                    const uint32_t pos = pos0 + t + rows - 1u;
                    const bool emit = ((pos + 1u) % ratio) == 0u;
                    if (emit &&
                        g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
                      fprintf(stderr,
                              "ds4: ROCm graph indexer compressed KV cache "
                              "capacity exceeded at layer %u\n",
                              il);
                      ok = false;
                      break;
                    }
                    ds4_gpu_tensor* kv_view = ds4_gpu_tensor_view(
                        g->batch_comp_kv,
                        (uint64_t)t * index_width * sizeof(float),
                        (uint64_t)rows * index_width * sizeof(float));
                    ds4_gpu_tensor* sc_view = ds4_gpu_tensor_view(
                        g->batch_comp_sc,
                        (uint64_t)t * index_width * sizeof(float),
                        (uint64_t)rows * index_width * sizeof(float));
                    const uint32_t index_row = g->layer_n_index_comp[il];
                    ok = kv_view && sc_view;
                    if (ok && rows > 1u) {
                      ok = ds4_gpu_compressor_store_batch_tensor(
                               kv_view, sc_view, g->layer_index_state_kv[il],
                               g->layer_index_state_score[il], model->map,
                               model->size,
                               layer->indexer_compressor_ape->abs_offset,
                               layer->indexer_compressor_ape->type,
                               DS4_N_INDEXER_HEAD_DIM, ratio, pos0 + t,
                               rows) != 0;
                    }
                    ok = ok &&
                         ds4_gpu_compressor_update_tensor(
                             kv_view, sc_view, g->layer_index_state_kv[il],
                             g->layer_index_state_score[il],
                             g->layer_index_comp_cache[il], model->map,
                             model->size,
                             layer->indexer_compressor_ape->abs_offset,
                             layer->indexer_compressor_ape->type,
                             layer->indexer_compressor_norm->abs_offset,
                             layer->indexer_compressor_norm->type,
                             DS4_N_INDEXER_HEAD_DIM, ratio, pos, index_row,
                             DS4_N_ROT,
                             compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                             freq_base, freq_scale, ext_factor, attn_factor,
                             DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW,
                             DS4_RMS_EPS, rows > 1u, false, false) != 0;
                    if (ok && emit) {
                      ds4_gpu_tensor* index_row_view = ds4_gpu_tensor_view(
                          g->layer_index_comp_cache[il],
                          (uint64_t)index_row * DS4_N_INDEXER_HEAD_DIM *
                              sizeof(float),
                          (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                      if (!index_row_view) {
                        ok = false;
                      } else {
                        ok =
                            ds4_gpu_dsv4_indexer_qat_tensor(
                                index_row_view, 1, DS4_N_INDEXER_HEAD_DIM) != 0;
                        ds4_gpu_tensor_free(index_row_view);
                      }
                    }
                    if (ok && emit)
                      g->layer_n_index_comp[il]++;
                    if (index_counts) {
                      for (uint32_t row = t; row < t + rows; ++row)
                        index_counts[row] =
                            index_row + uint32_t(emit && row + 1u == t + rows);
                    }
                    if (ok && g->spec_capture_prefixes && t + 1u < n_tokens &&
                        t < DS4_SPEC_PREFIX_SLOTS) {
                      ok = rocm_graph_capture_prefix_index_state(g, il, t);
                    }
                    ds4_gpu_tensor_free(sc_view);
                    ds4_gpu_tensor_free(kv_view);
                    t += rows;
                  }
                }
            }
        }
        if (ok && ds4_no_comp_attn_env()) n_comp = 0;
        if (ratio == 4) GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("indexer_setup");

        // A verifier block may cross the first sparse-indexer boundary. Rows
        // before it still use scalar's dense compressed attention order.
        const bool crosses_indexer_boundary =
            g->spec_capture_prefixes && ratio == 4 &&
            n_comp > DS4_N_INDEXER_TOP_K &&
            (pos0 + 1u) / ratio <= DS4_N_INDEXER_TOP_K;
        if (ok && !zero_prefix && n_tokens <= g->raw_cap &&
            !crosses_indexer_boundary) {
            const uint32_t n_raw = rocm_graph_raw_span_for_batch(g, pos0, n_tokens);
            /* See the raw-only branch above: batched mixed attention also
             * consumes a logical raw window, linearized out of the ring. */
            const uint32_t raw_start =
                rocm_graph_raw_start_for_span(g, pos0 + n_tokens - 1u, n_raw);
            bool use_indexed_comp = false;
            uint32_t pf_ratio_local = ratio;
            double index_stage_t0 = 0.0;

            ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                     g->batch_kv,
                                                     g->raw_cap,
                                                     pos0,
                                                     n_tokens,
                                                     DS4_N_HEAD_DIM) != 0;
            if (ok && ratio == 4 && n_comp > DS4_N_INDEXER_TOP_K) {
                const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                if (index_stage_profile) {
                    ok = rocm_graph_indexer_stage_profile_boundary(NULL,
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                ok = ok &&
                     rocm_graph_reserve_indexer_scores(g, n_comp, n_tokens) &&
                     ds4_gpu_indexer_scores_decode_batch_tensor(
                         g->indexer_scores, g->batch_indexer_q,
                         g->batch_indexer_weights,
                         g->layer_index_comp_cache[il], n_comp, n_tokens, pos0,
                         DS4_N_INDEXER_HEAD, DS4_N_INDEXER_HEAD_DIM, ratio,
                         index_scale, g->batch_heads) != 0;
                if (ok && index_stage_profile) {
                    ok = rocm_graph_indexer_stage_profile_boundary("score",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                if (ok) {
                  ok = ds4_gpu_indexer_topk_tensor(
                           g->comp_selected, g->indexer_scores, n_comp,
                           n_tokens, DS4_N_INDEXER_TOP_K) != 0;
                  if (ok && index_stage_profile) {
                    ok = rocm_graph_indexer_stage_profile_boundary(
                        "topk", il, pos0, n_tokens, n_comp, &index_stage_t0);
                  }
                }
                if (ok) {
                    use_indexed_comp = true;
                }
            }
            /* V4.1: a published chunk top-k (indexed against the owner's
             * shared index cache) switches the batch FA to the indexed
             * gather over the owner's compressed cache. */
            if (ok && pf_v41 && !use_indexed_comp && pf_sel && n_comp != 0) {
                use_indexed_comp = true;
                pf_ratio_local = g->pf_ratio;
            }
            if (ok) {
                if (use_indexed_comp) {
                    ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(g->batch_heads,
                                                                              model->map,
                                                                              model->size,
                                                                              layer->attn_sinks->abs_offset,
                                                                              g->batch_q,
                                                                              g->layer_raw_cache[il],
                                                                              rocm_graph_prefill_attn_comp_cache(
                                                                                  g, il, n_tokens),
                                                                              rocm_graph_prefill_attn_comp_cache_is_f16(
                                                                                  g, il, n_tokens),
                                                                              g->comp_selected,
                                                                              n_tokens,
                                                                              pos0,
                                                                              n_raw,
                                                                              g->raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              DS4_N_INDEXER_TOP_K,
                                                                              g->raw_window,
                                                                              pf_ratio_local,
                                                                              DS4_N_HEAD,
                                                                              DS4_N_HEAD_DIM) != 0;
                    if (ok && index_stage_profile) {
                        ok = rocm_graph_indexer_stage_profile_boundary("attention",
                                                                        il,
                                                                        pos0,
                                                                        n_tokens,
                                                                        n_comp,
                                                                        &index_stage_t0);
                    }
                } else {
                  ok = ds4_gpu_attention_decode_mixed_batch_heads_tensor(
                           g->batch_heads, model->map, model->size,
                           layer->attn_sinks->abs_offset, g->batch_q,
                           g->layer_raw_cache[il],
                           pf_v41 ? rocm_graph_prefill_attn_comp_cache(g, il,
                                                                       n_tokens)
                                  : g->layer_attn_comp_cache[il],
                           pf_v41 ? rocm_graph_prefill_attn_comp_cache_is_f16(
                                        g, il, n_tokens)
                                  : rocm_graph_attn_comp_cache_is_f16(),
                           nullptr, 0,
                           n_tokens, pos0, n_raw, g->raw_cap, raw_start, n_comp,
                           g->raw_window, ratio, DS4_N_HEAD,
                           DS4_N_HEAD_DIM) != 0;
                }
                if (getenv("DS4_LTRACE") && (il == 0u || il == 1u)) {
                    static long seen[64]; static int ns = 0;
                    long k = ((long)n_tokens << 24) ^ ((long)n_comp << 12) ^ (long)il;
                    bool dup = false;
                    for (int z = 0; z < ns; ++z) if (seen[z] == k) dup = true;
                    if (!dup && ns < 64) { seen[ns++] = k;
                        fprintf(stderr, "ds4: [ltr] base il=%u n_tok=%u n_comp=%u pos0=%u raw_start=%u n_raw=%u pf_v41=%d\n",
                                il, n_tokens, n_comp, pos0, raw_start, n_raw, (int)pf_v41); }
                }
            }
            if (ok) batch_attention_done = true;
        }

        if (getenv("DS4_LTRACE") && (il == 0u || il == 1u)) {
            static long seen2[64]; static int ns2 = 0;
            long k = ((long)n_tokens << 20) ^ ((long)n_comp << 8) ^ ((long)il << 1) ^ (zero_prefix?1:0);
            bool dup = false;
            for (int z = 0; z < ns2; ++z) if (seen2[z] == k) dup = true;
            if (!dup && ns2 < 64) { seen2[ns2++] = k;
                fprintf(stderr, "ds4: [ltr] route il=%u n_tok=%u n_comp=%u zero_prefix=%d topk_needed=%d\n",
                        il, n_tokens, n_comp, (int)zero_prefix, (int)(ratio == 4 && n_comp > DS4_N_INDEXER_TOP_K)); }
        }
        const bool topk_prefill_needed = ratio == 4 && n_comp > DS4_N_INDEXER_TOP_K;
        if (ok && zero_prefix && topk_prefill_needed && n_comp != 0) {
            const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
            double index_stage_t0 = 0.0;
            if (index_stage_profile) {
                ok = rocm_graph_indexer_stage_profile_boundary(NULL,
                                                                il,
                                                                pos0,
                                                                n_tokens,
                                                                n_comp,
                                                                &index_stage_t0);
            }
            ok = ok && rocm_graph_reserve_indexer_scores(g, n_comp, n_tokens) &&
                 ds4_gpu_indexer_scores_prefill_tensor(
                     g->indexer_scores, g->batch_indexer_q,
                     g->batch_indexer_weights, g->layer_index_comp_cache[il],
                     n_comp, n_tokens, DS4_N_INDEXER_HEAD,
                     DS4_N_INDEXER_HEAD_DIM, ratio, index_scale,
                     g->batch_heads) != 0;
            if (ok && index_stage_profile) {
                ok = rocm_graph_indexer_stage_profile_boundary("score",
                                                                il,
                                                                pos0,
                                                                n_tokens,
                                                                n_comp,
                                                                &index_stage_t0);
            }
            if (ok) {
              ok = ds4_gpu_indexer_topk_tensor(
                       g->comp_selected, g->indexer_scores, n_comp, n_tokens,
                       DS4_N_INDEXER_TOP_K) != 0;
              if (ok && index_stage_profile) {
                ok = rocm_graph_indexer_stage_profile_boundary(
                    "topk", il, pos0, n_tokens, n_comp, &index_stage_t0);
              }
            }
            if (ok) {
                ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(g->batch_heads,
                                                                          model->map,
                                                                          model->size,
                                                                          layer->attn_sinks->abs_offset,
                                                                          g->batch_q,
                                                                          g->layer_raw_cache[il],
                                                                          rocm_graph_prefill_attn_comp_cache(
                                                                              g, il, n_tokens),
                                                                          rocm_graph_prefill_attn_comp_cache_is_f16(
                                                                              g, il, n_tokens),
                                                                          g->comp_selected,
                                                                          n_tokens,
                                                                          pos0,
                                                                          n_tokens,
                                                                          g->raw_cap,
                                                                          0,
                                                                          n_comp,
                                                                          DS4_N_INDEXER_TOP_K,
                                                                          g->raw_window,
                                                                          ratio,
                                                                          DS4_N_HEAD,
                                                                          DS4_N_HEAD_DIM) != 0;
                if (ok && index_stage_profile) {
                    ok = rocm_graph_indexer_stage_profile_boundary("attention",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
            }
            if (ok) batch_attention_done = true;
        }
        if (ok && zero_prefix && !topk_prefill_needed && n_comp != 0) {
            ok = ds4_gpu_attention_prefill_static_mixed_heads_tensor(g->batch_heads,
                                                                       model->map,
                                                                       model->size,
                                                                       layer->attn_sinks->abs_offset,
                                                                       g->batch_q,
                                                                       g->batch_kv,
                                                                       g->layer_attn_comp_cache[il],
                                                                       rocm_graph_attn_comp_cache_is_f16(),
                                                                       n_tokens,
                                                                       n_comp,
                                                                       g->raw_window,
                                                                       ratio,
                                                                       DS4_N_HEAD,
                                                                       DS4_N_HEAD_DIM) != 0;
            if (ok) batch_attention_done = true;
        }
    }

    if (ok && !raw_batch_attention && !batch_attention_done) {
        uint32_t raw_prefix_tokens = 0;
        if (zero_prefix && ratio != 0 && n_tokens <= g->raw_cap && comp_counts != NULL) {
            while (raw_prefix_tokens < n_tokens && comp_counts[raw_prefix_tokens] == 0u) {
                raw_prefix_tokens++;
            }
        }

        if (raw_prefix_tokens != 0) {
            ok = ds4_gpu_attention_prefill_raw_heads_tensor(g->batch_heads,
                                                              model->map,
                                                              model->size,
                                                              layer->attn_sinks->abs_offset,
                                                              g->batch_q,
                                                              g->batch_kv,
                                                              raw_prefix_tokens,
                                                              g->raw_window,
                                                              DS4_N_HEAD,
                                                              DS4_N_HEAD_DIM) != 0;
        }
        if (raw_prefix_tokens < n_tokens) {
            for (uint32_t t = raw_prefix_tokens; ok && t < n_tokens; t++) {
                const uint32_t pos = pos0 + t;
                const uint32_t n_raw = rocm_graph_raw_span_for_batch(g, pos, 1);
                const uint32_t raw_start = rocm_graph_raw_start_for_span(g, pos, n_raw);
                const uint32_t cur_comp =
                    (comp_counts && !ds4_no_comp_attn_env()) ? comp_counts[t] : 0u;
                const uint32_t cur_index = index_counts ? index_counts[t] : 0u;
                uint32_t n_selected = 0;

                if (ratio == 4 && cur_comp > DS4_N_INDEXER_TOP_K) {
                    const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                    ds4_gpu_tensor *indexer_q_view = rocm_graph_tensor_row_view(
                            g->batch_indexer_q, t, (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM);
                    ds4_gpu_tensor *indexer_w_view = rocm_graph_tensor_row_view(
                            g->batch_indexer_weights, t, DS4_N_INDEXER_HEAD);
                    ok = indexer_q_view && indexer_w_view &&
                         rocm_graph_reserve_indexer_scores(g, cur_index, 1) &&
                         ds4_gpu_indexer_score_one_tensor(
                             g->indexer_scores, indexer_q_view, indexer_w_view,
                             g->layer_index_comp_cache[il], cur_index,
                             DS4_N_INDEXER_HEAD, DS4_N_INDEXER_HEAD_DIM,
                             index_scale) != 0 &&
                         ds4_gpu_indexer_topk_tensor(
                             g->comp_selected, g->indexer_scores, cur_index, 1,
                             DS4_N_INDEXER_TOP_K) != 0;
                    ds4_gpu_tensor_free(indexer_w_view);
                    ds4_gpu_tensor_free(indexer_q_view);
                    if (ok) {
                      n_selected = DS4_N_INDEXER_TOP_K < cur_index
                                       ? DS4_N_INDEXER_TOP_K
                                       : cur_index;
                    }
                }

                ds4_gpu_tensor *q_view = rocm_graph_tensor_row_view(g->batch_q, t, q_dim);
                ds4_gpu_tensor *kv_cache_view = rocm_graph_tensor_row_view(g->batch_kv, t, DS4_N_HEAD_DIM);
                ds4_gpu_tensor *heads_view = rocm_graph_tensor_row_view(g->batch_heads, t, q_dim);
                ok = ok && q_view && kv_cache_view && heads_view;
                if (ok && !zero_prefix) {
                    ok = ds4_gpu_store_raw_kv_tensor(g->layer_raw_cache[il],
                                                       kv_cache_view,
                                                       g->raw_cap,
                                                       pos % g->raw_cap,
                                                       DS4_N_HEAD_DIM) != 0;
                }
                if (ok && n_selected != 0) {
                  if (getenv("DS4_FA_DBG")) fprintf(stderr,
                      "ds4fa: L%u t%u RAW n_raw=%u w=%u rs=%u | IDX comp=%u sel=%u ratio=%u\n",
                      il, t, n_raw, g->raw_window, raw_start, cur_comp, n_selected, ratio);
                  ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                           heads_view, model->map, model->size,
                           layer->attn_sinks->abs_offset, q_view,
                           g->layer_raw_cache[il], g->layer_attn_comp_cache[il],
                           rocm_graph_attn_comp_cache_is_f16(),
                           g->comp_selected, 1, pos, n_raw, g->raw_cap,
                           raw_start, cur_comp, n_selected, g->raw_window,
                           ratio, DS4_N_HEAD, DS4_N_HEAD_DIM) != 0;
                } else if (ok) {
                  if (getenv("DS4_FA_DBG") && t % 8u == 7u) fprintf(stderr,
                      "ds4fa: L%u t%u RAW n_raw=%u w=%u rs=%u | NOP comp=%u\n",
                      il, t, n_raw, g->raw_window, raw_start, cur_comp);
                  ok = ds4_gpu_attention_decode_heads_tensor(
                           heads_view, model->map, model->size,
                           layer->attn_sinks->abs_offset, q_view,
                           g->layer_raw_cache[il], n_raw, g->raw_cap, raw_start,
                           cur_comp ? g->layer_attn_comp_cache[il] : NULL,
                           rocm_graph_attn_comp_cache_is_f16(), cur_comp,
                           nullptr, 0, DS4_N_HEAD, DS4_N_HEAD_DIM) != 0;
                }
                ds4_gpu_tensor_free(heads_view);
                ds4_gpu_tensor_free(kv_cache_view);
                ds4_gpu_tensor_free(q_view);
            }
        }
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("attention");
    if (hb_dump_pre) {
        /* full per-layer q/kv dumps AFTER all kv transforms: the exact
         * values the attention kernel consumed (kv is roped + fp8 here). */
        struct { ds4_gpu_tensor *t; uint64_t w; const char *nm; } qkd[] = {
            {g->batch_q, (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM, "aq"},
            {g->batch_kv, (uint64_t)DS4_N_HEAD_DIM, "akv"},
        };
        for (auto &e : qkd) {
            if (!e.t) continue;
            ds4_gpu_tensor *v = ds4_gpu_tensor_view(
                e.t, 0, (uint64_t)n_tokens * e.w * 4ull);
            if (v) { char fn[80]; snprintf(fn, sizeof fn, "/tmp/hb_all_%s_%02u.bin", e.nm, il);
                     ds4_dbg_dump(v, fn, (uint32_t)(n_tokens * e.w)); }
            ds4_gpu_tensor_free(v);
        }
    }
    ds4_probe_stats(g->batch_heads,
                    (uint64_t)(n_tokens - 1u) * DS4_N_HEAD * DS4_N_HEAD_DIM,
                    DS4_N_HEAD * DS4_N_HEAD_DIM, "fa_heads", il);
    rocm_graph_probe_layer_tensor(g->batch_heads,
                                  (uint64_t)n_tokens * DS4_N_HEAD *
                                      DS4_N_HEAD_DIM,
                                  "heads", il);
    if (ok && getenv("DS4_H_DUMP") && il == 0u) {
        char hbp[64];
        ds4_gpu_tensor *v = ds4_gpu_tensor_view(
            g->batch_heads,
            (uint64_t)(n_tokens - 1u) * 32768ull * 4u, 32768ull * 4u);
        snprintf(hbp, sizeof hbp, "/tmp/hb_ho_L00.bin");
        if (v) ds4_dbg_dump(v, hbp, 32768u);
    }
    if (ok && getenv("DS4_H_DUMP")) {
        char hbp[64];
        ds4_gpu_tensor *v = ds4_gpu_tensor_view(
            g->batch_heads, 0, (uint64_t)n_tokens * 32768ull * 4u);
        snprintf(hbp, sizeof hbp, "/tmp/hb_all_ho_%02u.bin", il);
        if (v) ds4_dbg_dump(v, hbp, (uint32_t)(n_tokens * 32768u));
        ds4_gpu_tensor_free(v);
    }

    if (ok && !project_output) {
        ok = ds4_gpu_rope_tail_tensor(
                 g->batch_heads, n_tokens, DS4_N_HEAD, DS4_N_HEAD_DIM,
                 DS4_N_ROT, pos0,
                 compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, true,
                 freq_base, freq_scale, ext_factor, attn_factor,
                 DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
        GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("inv_rope");
        ds4_gpu_tensor_free(after_attn_hc_view);
        ds4_gpu_tensor_free(attn_cur_view);
        ds4_gpu_tensor_free(hc_split_view);
        ds4_gpu_tensor_free(hc_split_a_view);
        if (hc_pre_view != hc_split_view) ds4_gpu_tensor_free(hc_pre_view);
        ds4_gpu_tensor_free(hc_mix_view);
        free(index_counts);
        free(comp_counts);
        return ok;
    }

    /*
     * A one-row batch is used by the multi-session route only for the
     * cache-owning attention middle. Keep its output projection on the release
     * decode fusion: the generic batch projection is sized for prompt rows and
     * costs about 2 ms per layer at W=1, dwarfing the attention itself.
     */
    bool output_hc_fused = false;
    const bool v41_wo = layer->attn_output_a->type == DS4_TENSOR_Q5_K;
    if (ok && n_tokens == 1u && !v41_wo) {
        ok = ds4_gpu_rope_tail_tensor(g->batch_heads,
                                     1,
                                     DS4_N_HEAD,
                                     DS4_N_HEAD_DIM,
                                     DS4_N_ROT,
                                     pos0,
                                     compressed
                                         ? (uint32_t)DS4_ROPE_ORIG_CTX
                                         : 0,
                                     true,
                                     freq_base,
                                     freq_scale,
                                     ext_factor,
                                     attn_factor,
                                     DS4_ROPE_YARN_BETA_FAST,
                                     DS4_ROPE_YARN_BETA_SLOW) != 0;
        if (ok) {
            ok = ds4_gpu_attention_output_low_q8_tensor(
                     g->attn_low,
                     model->map,
                     model->size,
                     layer->attn_output_a->abs_offset,
                     group_dim,
                     rank,
                     n_groups,
                     g->batch_heads,
                     NULL,
                     NULL) != 0;
        }
        if (ok) {
            ok = ds4_gpu_matmul_q8_0_hc_expand_tensor(
                     after_attn_hc_view,
                     g->attn_out,
                     model->map,
                     model->size,
                     layer->attn_output_b->abs_offset,
                     (uint64_t)n_groups * rank,
                     DS4_N_EMBD,
                     g->attn_low,
                     NULL,
                     g->batch_cur_hc,
                     hc_split_view,
                     DS4_N_EMBD,
                     DS4_N_HC) != 0;
        }
        output_hc_fused = ok;
    }

    /* The output projection can fold the inverse rotated tail into its F16 group
     * pack; nothing else reads the roped F32 heads. It declines when the shape
     * or width does not qualify, in which case the separate pass runs. */
    bool inv_rope_fused = false;
    if (ok && !output_hc_fused && !v41_wo) {
        inv_rope_fused = ds4_gpu_attention_output_q8_batch_inv_rope_tensor(
                             g->batch_attn_out,
                             g->batch_attn_low,
                             g->batch_group_tmp,
                             g->batch_low_tmp,
                             model->map,
                             model->size,
                             layer->attn_output_a->abs_offset,
                             layer->attn_output_b->abs_offset,
                             group_dim,
                             rank,
                             n_groups,
                             DS4_N_EMBD,
                             g->batch_heads,
                             n_tokens,
                             DS4_N_HEAD_DIM,
                             DS4_N_ROT,
                             pos0,
                             compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                             freq_base,
                             freq_scale,
                             ext_factor,
                             attn_factor,
                             DS4_ROPE_YARN_BETA_FAST,
                             DS4_ROPE_YARN_BETA_SLOW) != 0;
    }
    if (ok && !output_hc_fused && !inv_rope_fused) {
      ok = ds4_gpu_rope_tail_tensor(
               g->batch_heads, n_tokens, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT,
               pos0, compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, true,
               freq_base, freq_scale, ext_factor, attn_factor,
               DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("inv_rope");
    if (getenv("DS4_H_DUMP") && il == 0u) {
        FILE *rf = fopen("/tmp/rope_dbg.log", "a");
        if (rf) {
            fprintf(rf,
                    "[rope-dbg] pos0=%u compressed=%d freq_base=%f freq_scale=%f "
                    "ext_factor=%f attn_factor=%f n_tokens=%u\n",
                    pos0, (int)compressed, freq_base, freq_scale, ext_factor,
                    attn_factor, n_tokens);
            fclose(rf);
        }
    }
    if (getenv("DS4_H_DUMP") && il == 0u && !g_rope_probe_done) {
        g_rope_probe_done = true;
        /* Oracle: rotate 4 synthetic tokens x 1 head (dim 512, rot 64) with
         * pos0=0 stride 1 (positions 0..3); compare against a host replica. */
        ds4_gpu_tensor *pt = ds4_gpu_tensor_alloc_managed(4u * 512u * 4u);
        if (pt) {
            float *probe = (float *)ds4_gpu_tensor_device_pointer(pt);
            for (uint32_t i = 0; i < 4u * 512u; i++) probe[i] = 0.0f;
            for (uint32_t t = 0; t < 4u; t++) {
                for (uint32_t p = 0; p < 32u; p++) {
                    probe[t * 512u + 448u + 2u * p] = 1.0f;
                    probe[t * 512u + 448u + 2u * p + 1u] = 0.5f + 0.05f * (float)p;
                }
            }
            ds4_gpu_rope_tail_tensor(pt, 4u, 1u, 512u, 64u, 0u, 0u, true,
                                     10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            ds4_dbg_dump(pt, const_cast<char *>("/tmp/rope_probe_L00.bin"),
                         4u * 512u);
            /* Deliberately NOT freed: pt lives in the bump arena; freeing it
             * would hipFree the whole arena and dangle every weight tensor. */
        }
    }
    if (getenv("DS4_H_DUMP") && il == 0u) {
        char hbp[64];
        ds4_gpu_tensor *v = ds4_gpu_tensor_view(
            g->batch_heads,
            (uint64_t)(n_tokens - 1u) * 32768ull * 4u, 32768ull * 4u);
        snprintf(hbp, sizeof hbp, "/tmp/hb_hr_L00.bin");
        if (v) ds4_dbg_dump(v, hbp, 32768u);
    }
    if (ok && !output_hc_fused && !inv_rope_fused) {
      ok = ds4_gpu_attention_output_q8_batch_tensor(
               g->batch_attn_out, g->batch_attn_low, g->batch_group_tmp,
               g->batch_low_tmp, model->map, model->size,
               layer->attn_output_a->abs_offset,
               layer->attn_output_b->abs_offset, group_dim, rank, n_groups,
               DS4_N_EMBD, g->batch_heads, n_tokens) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("output_proj");
    if (ok && v41_wo) {
      if (getenv("DS4_H_DUMP") && il == 0u) {
          char hbp[64];
          ds4_gpu_tensor *v = ds4_gpu_tensor_view(
              g->batch_heads,
              (uint64_t)(n_tokens - 1u) * 32768ull * 4u, 32768ull * 4u);
          snprintf(hbp, sizeof hbp, "/tmp/hb_h2_L00.bin");
          if (v) ds4_dbg_dump(v, hbp, 32768u);
          /* Byte-exact check of the weight image the kernel actually reads
           * (first 32 KB of attn_output_a) against the GGUF file bytes. */
          const char *wa_dbg = (const char *)ds4_gpu_model_range_ptr(
              model->map, layer->attn_output_a->abs_offset, 32768u,
              "dbg_output_a_q5k");
          if (wa_dbg) {
              FILE *wf = fopen("/tmp/hb_wa_L00.bin", "wb");
              if (wf) { fwrite(wa_dbg, 1, 32768u, wf); fclose(wf); }
          } else {
              fprintf(stderr, "ds4: wo_a runtime range lookup FAILED\n");
          }
      }
      /* Micro-probe: drive the real output_a kernel with a synthetic heads
       * tensor and dump in/out + the byte-exact registered weight image, so
       * the host can separate a wrong weight slice from a wrong contraction.
       * x[t,g,d] = ((d*13 + t*7 + g) % 29 - 14) * 0.25 */
      if (ok && getenv("DS4_H_DUMP") && getenv("DS4_GOLDEN0") &&
        getenv("GUFO_DS4_WOAP") && il == 0u &&
          !g_woa_probe_done) {
          g_woa_probe_done = true;
          const uint32_t pw_toks = 4u, pw_groups = n_groups;
          const uint32_t pw_gdim = (uint32_t)group_dim;
          const uint32_t pw_rank = (uint32_t)rank;
          ds4_gpu_tensor *ph = ds4_gpu_tensor_alloc_managed(
              (uint64_t)pw_toks * pw_groups * pw_gdim * 4u);
          ds4_gpu_tensor *pl = ds4_gpu_tensor_alloc_managed(
              (uint64_t)pw_toks * pw_groups * pw_rank * 4u);
          if (ph && pl &&
              ds4_gpu_attention_output_a_q5_k_tensor(
                  pl, model->map, model->size,
                  layer->attn_output_a->abs_offset, group_dim, rank, n_groups,
                  ph, pw_toks)) {
              ds4_gpu_synchronize();
              float *hp = (float *)ds4_gpu_tensor_device_pointer(ph);
              for (uint32_t t = 0; t < pw_toks; ++t)
                  for (uint32_t g = 0; g < pw_groups; ++g) {
                      float *row = hp + ((uint64_t)t * pw_groups + g) * pw_gdim;
                      for (uint32_t d = 0; d < pw_gdim; ++d)
                          row[d] =
                              (float)(((int)(d * 13u + t * 7u + g) % 29) - 14) *
                              0.25f;
                  }
              ds4_gpu_attention_output_a_q5_k_tensor(
                  pl, model->map, model->size,
                  layer->attn_output_a->abs_offset, group_dim, rank, n_groups,
                  ph, pw_toks);
              ds4_gpu_synchronize();
              FILE *pf = fopen("/tmp/gw_heads.bin", "wb");
              if (pf) {
                  fwrite(hp, 4u, (size_t)pw_toks * pw_groups * pw_gdim, pf);
                  fclose(pf);
              }
              const float *lp = (const float *)ds4_gpu_tensor_device_pointer(pl);
              pf = fopen("/tmp/gw_low.bin", "wb");
              if (pf) {
                  fwrite(lp, 4u, (size_t)pw_toks * pw_groups * pw_rank, pf);
                  fclose(pf);
              }
              const uint64_t wbytes = (uint64_t)pw_groups * pw_rank *
                                      (pw_gdim / 256ull) * 176ull;
              const char *wimg = (const char *)ds4_gpu_model_range_ptr(
                  model->map, layer->attn_output_a->abs_offset, wbytes,
                  "dbg_output_a_probe");
              if (wimg) {
                  pf = fopen("/tmp/gw_map.bin", "wb");
                  if (pf) {
                      fwrite(wimg, 1, (size_t)wbytes, pf);
                      fclose(pf);
                  }
              } else {
                  fprintf(stderr, "ds4: wo_a probe range lookup FAILED\n");
              }
              fprintf(stderr, "ds4: wo_a micro-probe written (toks=%u rank=%u "
                              "groups=%u gdim=%u)\n",
                      pw_toks, pw_rank, pw_groups, pw_gdim);
          }
      }
      if (getenv("DS4_H_DUMP") && il == 0u && g->batch_heads) {
          char hbp[80];
          snprintf(hbp, sizeof hbp, "/tmp/hb_all_hr_%02u.bin", il);
          ds4_gpu_tensor *v = ds4_gpu_tensor_view(g->batch_heads, 0,
              (uint64_t)n_tokens * 32768ull * 4ull);
          if (v) { ds4_dbg_dump(v, hbp, (uint32_t)(n_tokens * 32768ull));
                   ds4_gpu_tensor_free(v); }
      }
      if (getenv("DS4_H_DUMP") && il == 0u && g->batch_attn_low) {
          char hbp[80];
          snprintf(hbp, sizeof hbp, "/tmp/hb_all_low_%02u.bin", il);
          ds4_gpu_tensor *v = ds4_gpu_tensor_view(g->batch_attn_low, 0,
              (uint64_t)n_tokens * 8192ull * 4ull);
          if (v) { ds4_dbg_dump(v, hbp, (uint32_t)(n_tokens * 8192ull));
                   ds4_gpu_tensor_free(v); }
      }
      /* V4.1: grouped Q5_K wo_a (concat per group, no sum) + Q5_K wo_b. */
      ok = ds4_gpu_attention_output_a_q5_k_tensor(
               g->batch_attn_low, model->map, model->size,
               layer->attn_output_a->abs_offset, group_dim, rank, n_groups,
               g->batch_heads, (uint32_t)n_tokens) != 0;
      rocm_graph_probe_layer_tensor(g->batch_attn_low,
                                    (uint64_t)n_tokens * n_groups * rank,
                                    "attn low", il);
      if (getenv("DS4_H_DUMP") && il == 0u) {
          char hbp[64];
          ds4_gpu_tensor *v = ds4_gpu_tensor_view(
              g->batch_attn_low,
              (uint64_t)(n_tokens - 1u) * 8192ull * 4u, 8192ull * 4u);
          snprintf(hbp, sizeof hbp, "/tmp/hb_ol_L00.bin");
          if (v) ds4_dbg_dump(v, hbp, 8192u);
      }
      if (ok)
        ok = ds4_gpu_matmul_q5_k_tensor(
                 g->batch_attn_out, model->map, model->size,
                 layer->attn_output_b->abs_offset, (uint64_t)n_groups * rank,
                 DS4_N_EMBD, g->batch_attn_low, (uint32_t)n_tokens) != 0;
      rocm_graph_probe_layer_tensor(g->batch_attn_out,
                                    (uint64_t)n_tokens * DS4_N_EMBD,
                                    "attn out", il);
      if (getenv("DS4_H_DUMP") && il == 0u) {
          char hbp[64];
          ds4_gpu_tensor *v = ds4_gpu_tensor_view(
              g->batch_attn_out,
              (uint64_t)(n_tokens - 1u) * DS4_N_EMBD * 4u,
              (uint64_t)DS4_N_EMBD * 4u);
          snprintf(hbp, sizeof hbp, "/tmp/hb_ob_L00.bin");
          if (v) ds4_dbg_dump(v, hbp, (uint64_t)DS4_N_EMBD);
      }
    }
    if (ok && !output_hc_fused) {
        ok = ds4_gpu_hc_expand_split_tensor(after_attn_hc_view,
                                                  g->batch_attn_out,
                                                  g->batch_cur_hc,
                                                  hc_split_a_view,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    }
    if (getenv("DS4_H_DUMP") && g->batch_after_attn_hc) {
        char hbp[80];
        snprintf(hbp, sizeof hbp, "/tmp/hb_all_xa_%02u.bin", il);
        ds4_gpu_tensor *v = ds4_gpu_tensor_view(g->batch_after_attn_hc, 0,
            (uint64_t)n_tokens * 4ull * DS4_N_EMBD * 4ull);
        if (v) { ds4_dbg_dump(v, hbp, (uint32_t)(n_tokens * 4ull * DS4_N_EMBD));
                 ds4_gpu_tensor_free(v); }
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("hc_post");
    ds4_gpu_tensor_free(after_attn_hc_view);
    ds4_gpu_tensor_free(attn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
    free(index_counts);
    free(comp_counts);
#undef GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE
#undef GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE
    return ok;
}

/*
 * Multi-session decode attention.
 *
 * HC/Q/KV and output projections are position independent, so run them once
 * over all active rows. Cache mutation, compressor recurrence, indexer lookup,
 * and attention itself remain on each session graph.
 */
static bool rocm_graph_encode_sessions_attention_batch(
        ds4_gpu_graph             *g,
        const ds4_model           *model,
        const ds4_layer_weights   *layer,
        uint32_t                   il,
        const ds4_rocm_batch_item *items,
        uint32_t                   n_tokens) {
  if (!g || !model || !layer || !items || n_tokens < 2u ||
      n_tokens > g->prefill_cap) {
    return false;
  }
    for (uint32_t row = 0; row < n_tokens; ++row) {
        if (!items[row].graph || items[row].graph->raw_cap == 0) {
            return false;
        }
    }

    ds4_gpu_clear_f16_input();
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc =
        2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim =
        (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t n_groups = DS4_N_OUT_GROUP;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = DS4_N_LORA_O;
    const bool stage_profile =
        getenv("GUFO_DEEPSEEK_ROCM_LAYER_STAGE_PROFILE") != NULL;
    double stage_t0 = stage_profile ? ds4_now_seconds() : 0.0;
#define GUFO_DEEPSEEK_ROCM_PROFILE_SESSION_ATTN(name) do { \
        if (ok && stage_profile) { \
            ok = rocm_graph_layer_stage_profile_boundary( \
                "session_attn", (name), il, items[0].position, n_tokens, \
                &stage_t0); \
        } \
    } while (0)

    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
        g->batch_hc_mix,
        0,
        (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
        g->batch_hc_split,
        0,
        (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_a_view = ds4_gpu_tensor_view(
        g->batch_hc_split_a,
        0,
        (uint64_t)n_tokens * mix_hc * sizeof(float));
    /* See rocm_graph_encode_layer_attention_batch: the collapse PRE is the
     * previous layer's FFN split, identity one-hot at layer 0. */
    ds4_gpu_tensor *hc_pre_view = (il == 0u)
        ? ds4_gpu_tensor_view(g->batch_pre_onehot, 0,
                              (uint64_t)n_tokens * mix_hc * sizeof(float))
        : hc_split_view;
    ds4_gpu_tensor *attn_cur_view = ds4_gpu_tensor_view(
        g->batch_attn_cur,
        0,
        (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *after_attn_hc_view = ds4_gpu_tensor_view(
        g->batch_after_attn_hc,
        0,
        (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && attn_cur_view &&
              after_attn_hc_view;
    if (ok) {
        ok = ds4_gpu_rms_norm_plain_rows_tensor(
                 g->batch_flat_hc,
                 g->batch_cur_hc,
                 (uint32_t)hc_dim,
                 n_tokens,
                 DS4_RMS_EPS) != 0;
    }
    if (ok) {
        ok = rocm_graph_matmul_hc_fn_tensor(
                 hc_mix_view,
                 model,
                 layer->hc_attn_fn,
                 hc_dim,
                 mix_hc,
                 g->batch_flat_hc,
                 n_tokens) != 0;
    }
    if (ok) {
        /*
         * Use the release decode fusion for every row. The kernel maps one
         * independent block to each session, so widening the grid preserves
         * the one-row Sinkhorn, weighted-sum, and RMS reduction order exactly.
         */
        ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(
                 attn_cur_view,
                 g->batch_attn_norm,
                 NULL,
                 NULL,
                 hc_split_a_view,
                 hc_mix_view,
                 g->batch_cur_hc,
                 model->map,
                 model->size,
                 layer->hc_attn_scale->abs_offset,
                 layer->hc_attn_base->abs_offset,
                 layer->attn_norm->abs_offset,
                 DS4_N_EMBD,
                 DS4_N_HC,
                 DS4_N_HC_SINKHORN_ITER,
                 DS4_HC_EPS,
                 DS4_RMS_EPS,
                 hc_pre_view, 24u) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_SESSION_ATTN("hc_pre");

    if (ok) {
        ok = rocm_graph_matmul_v41_dense_tensor(g->batch_qr, model,
                                                layer->attn_q_a, DS4_N_EMBD,
                                                q_rank, g->batch_attn_norm,
                                                n_tokens);
    }
    if (ok) {
        ok = rocm_graph_matmul_v41_dense_tensor(g->batch_kv_raw, model,
                                                layer->attn_kv, DS4_N_EMBD,
                                                DS4_N_HEAD_DIM,
                                                g->batch_attn_norm, n_tokens);
    }
    if (ok) {
        ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
                 g->batch_qr_norm,
                 g->batch_qr,
                 model->map,
                 model->size,
                 layer->attn_q_a_norm->abs_offset,
                 (uint32_t)q_rank,
                 g->batch_kv,
                 g->batch_kv_raw,
                 layer->attn_kv_a_norm->abs_offset,
                 DS4_N_HEAD_DIM,
                 n_tokens,
                 DS4_RMS_EPS) != 0;
    }
    if (ok) {
        ok = rocm_graph_matmul_v41_dense_tensor(g->batch_q, model,
                                                layer->attn_q_b, q_rank, q_dim,
                                                g->batch_qr_norm, n_tokens);
    }
    if (ok) {
        ok = ds4_gpu_head_rms_norm_tensor(
                 g->batch_q,
                 n_tokens,
                 DS4_N_HEAD,
                 DS4_N_HEAD_DIM,
                 DS4_RMS_EPS) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_SESSION_ATTN("qkv");

    /*
     * The compressor projections use the same ordered per-row reduction as
     * serial decode while sharing both F16 weight streams across C2-C8.
     */
    const bool use_compressor_batch = ds4_layer_compress_ratio(il) != 0u &&
                                      ds4_layer_is_kv_source(il);
    uint32_t compressor_width = 0u;
    if (ok && use_compressor_batch) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const uint32_t coff = ratio == 4u ? 2u : 1u;
        compressor_width = coff * DS4_N_HEAD_DIM;
        if (!layer->attn_compressor_kv || !layer->attn_compressor_gate ||
            !ds4_graph_f16_weight_type(layer->attn_compressor_kv->type) ||
            !ds4_graph_f16_weight_type(rocm_graph_opt_w_type(layer->attn_compressor_gate)) ||
            layer->attn_compressor_kv->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_gate->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_kv->dim[1] != compressor_width ||
            layer->attn_compressor_gate->dim[1] != compressor_width) {
          ok = false;
        } else if (layer->attn_compressor_kv->type == DS4_TENSOR_F16) {
          ok = ds4_gpu_matmul_f16_pair_narrow_tensor(
                   g->batch_comp_kv, g->batch_comp_sc, model->map, model->size,
                   layer->attn_compressor_kv->abs_offset,
                   rocm_graph_opt_w_off(layer->attn_compressor_gate), DS4_N_EMBD,
                   compressor_width, g->batch_attn_norm, n_tokens) != 0;
        } else {
          ok = rocm_graph_matmul_hc_fn_tensor(
                   g->batch_comp_kv, model, layer->attn_compressor_kv,
                   DS4_N_EMBD, compressor_width, g->batch_attn_norm,
                   n_tokens);
          if (ok && layer->attn_compressor_gate)
            ok = rocm_graph_matmul_hc_fn_tensor(
                     g->batch_comp_sc, model, layer->attn_compressor_gate,
                     DS4_N_EMBD, compressor_width, g->batch_attn_norm,
                     n_tokens);
        }
    }
    for (uint32_t row = 0; ok && row < n_tokens; ++row) {
        ds4_gpu_tensor *attn_norm = rocm_graph_tensor_row_view(
            g->batch_attn_norm, row, DS4_N_EMBD);
        ds4_gpu_tensor *qr_norm = rocm_graph_tensor_row_view(
            g->batch_qr_norm, row, q_rank);
        ds4_gpu_tensor *q =
            rocm_graph_tensor_row_view(g->batch_q, row, q_dim);
        ds4_gpu_tensor *kv = rocm_graph_tensor_row_view(
            g->batch_kv, row, DS4_N_HEAD_DIM);
        ds4_gpu_tensor *heads =
            rocm_graph_tensor_row_view(g->batch_heads, row, q_dim);
        ds4_gpu_tensor *comp_kv = use_compressor_batch
            ? rocm_graph_tensor_row_view(
                  g->batch_comp_kv, row, compressor_width)
            : nullptr;
        ds4_gpu_tensor *comp_sc = use_compressor_batch
            ? rocm_graph_tensor_row_view(
                  g->batch_comp_sc, row, compressor_width)
            : nullptr;
        ok = attn_norm && qr_norm && q && kv && heads &&
             (!use_compressor_batch || (comp_kv && comp_sc));
        if (ok) {
            ok = rocm_graph_encode_session_attention_core(
                items[row].graph,
                model,
                layer,
                il,
                items[row].position,
                attn_norm,
                qr_norm,
                q,
                kv,
                heads,
                comp_kv,
                comp_sc);
        }
        ds4_gpu_tensor_free(comp_sc);
        ds4_gpu_tensor_free(comp_kv);
        ds4_gpu_tensor_free(heads);
        ds4_gpu_tensor_free(kv);
        ds4_gpu_tensor_free(q);
        ds4_gpu_tensor_free(qr_norm);
        ds4_gpu_tensor_free(attn_norm);
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_SESSION_ATTN("stateful_core");

    if (layer->attn_output_a->type == DS4_TENSOR_Q5_K) {
        if (ok)
            ok = ds4_gpu_attention_output_a_q5_k_tensor(
                     g->batch_attn_low, model->map, model->size,
                     layer->attn_output_a->abs_offset, group_dim, rank,
                     n_groups, g->batch_heads, (uint32_t)n_tokens) != 0;
        if (ok)
            ok = ds4_gpu_matmul_q5_k_tensor(
                     g->batch_attn_out, model->map, model->size,
                     layer->attn_output_b->abs_offset,
                     (uint64_t)n_groups * rank, DS4_N_EMBD, g->batch_attn_low,
                     (uint32_t)n_tokens) != 0;
    } else if (ok) {
        ok = ds4_gpu_attention_output_q8_batch_tensor(
                 g->batch_attn_out,
                 g->batch_attn_low,
                 g->batch_group_tmp,
                 g->batch_low_tmp,
                 model->map,
                 model->size,
                 layer->attn_output_a->abs_offset,
                 layer->attn_output_b->abs_offset,
                 group_dim,
                 rank,
                 n_groups,
                 DS4_N_EMBD,
                 g->batch_heads,
                 n_tokens) != 0;
    }
    if (getenv("DS4_H_DUMP") && g->batch_attn_out) {
        char hbp[80];
        snprintf(hbp, sizeof hbp, "/tmp/hb_all_ay_%02u.bin", il);
        ds4_gpu_tensor *v = ds4_gpu_tensor_view(g->batch_attn_out, 0,
            (uint64_t)n_tokens * DS4_N_EMBD * 4ull);
        if (v) { ds4_dbg_dump(v, hbp, (uint32_t)(n_tokens * DS4_N_EMBD));
                 ds4_gpu_tensor_free(v); }
    }
    if (ok) {
        ok = ds4_gpu_hc_expand_split_tensor(
                 after_attn_hc_view,
                 g->batch_attn_out,
                 g->batch_cur_hc,
                 hc_split_a_view,
                 DS4_N_EMBD,
                 DS4_N_HC) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_SESSION_ATTN("output");

    if (getenv("DS4_H_DUMP") && g->batch_after_attn_hc) {
        char hbp[80];
        snprintf(hbp, sizeof hbp, "/tmp/hb_all_xa_%02u.bin", il);
        ds4_gpu_tensor *v = ds4_gpu_tensor_view(g->batch_after_attn_hc, 0,
            (uint64_t)n_tokens * 4ull * DS4_N_EMBD * 4ull);
        if (v) { ds4_dbg_dump(v, hbp, (uint32_t)(n_tokens * 4ull * DS4_N_EMBD));
                 ds4_gpu_tensor_free(v); }
    }
    ds4_gpu_tensor_free(after_attn_hc_view);
    ds4_gpu_tensor_free(attn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_split_a_view);
    if (hc_pre_view != hc_split_view) ds4_gpu_tensor_free(hc_pre_view);
    ds4_gpu_tensor_free(hc_mix_view);
#undef GUFO_DEEPSEEK_ROCM_PROFILE_SESSION_ATTN
    return ok;
}

/* Row statistics for one workspace tensor row (residual-branch bisect). */
static void ds4_dbg_vec(const ds4_gpu_tensor *t, uint64_t off_elems,
                        uint32_t n, const char *tag) {
  if (!getenv("DS4_DEBUG_STATE") || !t) return;
  static float buf[DS4_N_EMBD];
  if (n > DS4_N_EMBD) n = DS4_N_EMBD;
  if (!ds4_gpu_readback_f32(t, off_elems, n, buf)) { fprintf(stderr, "ds4: vec[%s] readback fail\n", tag); return; }
  double s = 0.0; float mx = 0.f; uint32_t big = 0, nan = 0;
  for (uint32_t i = 0; i < n; i++) {
    const float a = fabsf(buf[i]);
    if (!(buf[i] == buf[i])) { nan++; continue; }
    if (a > mx) mx = a;
    if (a > 100.f) big++;
    s += (double)buf[i] * (double)buf[i];
  }
  fprintf(stderr, "ds4: vec[%s] rms=%.4f max=%.3f n>100=%u nan=%u\n",
          tag, sqrt(s / n), mx, big, nan);
}

/* Encode the batched prefill FFN half: HC pre/norm, shared expert, routed
 * experts, sum, and HC post. */
static bool rocm_graph_encode_layer_ffn_batch(
    ds4_gpu_graph* g, const ds4_model* model, const ds4_layer_weights* layer,
    uint32_t il, uint32_t pos0, uint32_t n_tokens,
    const ds4_gpu_tensor* tokens = nullptr,
    const uint32_t* router_group_offsets = nullptr,
    uint32_t router_group_count = 0) {
  if (n_tokens == 0 || n_tokens > g->prefill_cap)
    return false;
  if (getenv("DS4_DEBUG_STATE"))
    fprintf(stderr, "ds4: qtt L%02u rt=%u ep=%u guE=%u/%u dnE=%u guS=%u/%u dnS=%u\n",
            il,
            layer->ffn_gate_inp ? layer->ffn_gate_inp->type : 999,
            layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->type : 999,
            layer->ffn_gate_exps ? layer->ffn_gate_exps->type : 999,
            layer->ffn_up_exps ? layer->ffn_up_exps->type : 999,
            layer->ffn_down_exps ? layer->ffn_down_exps->type : 999,
            layer->ffn_gate_shexp ? layer->ffn_gate_shexp->type : 999,
            layer->ffn_up_shexp ? layer->ffn_up_shexp->type : 999,
            layer->ffn_down_shexp ? layer->ffn_down_shexp->type : 999);
  if (!tokens)
    tokens = g->prefill_tokens;
  if ((router_group_offsets != nullptr) != (router_group_count != 0u) ||
      router_group_count > 8u) {
    return false;
  }
  std::array<uint32_t, 9> group_offsets{};
  const uint32_t group_count = router_group_count;
  if (router_group_offsets != nullptr) {
    if (router_group_offsets[0] != 0u ||
        router_group_offsets[router_group_count] != n_tokens) {
      return false;
    }
    for (uint32_t group = 0; group <= router_group_count; ++group) {
      group_offsets[group] = router_group_offsets[group];
      if (group != 0u && group_offsets[group] <= group_offsets[group - 1u]) {
        return false;
      }
    }
  }

  const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
  const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
  const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
  const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
  const uint64_t expert_mid_dim = layer->ffn_gate_exps->dim[1];
  const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
  const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
  const uint64_t gate_row_bytes =
      ds4_routed_expert_row_bytes(layer->ffn_gate_exps);
  const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
  const uint64_t down_row_bytes =
      ds4_routed_expert_row_bytes(layer->ffn_down_exps);
  const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
  const bool layer_stage_profile =
      getenv("GUFO_DEEPSEEK_ROCM_LAYER_STAGE_PROFILE") != NULL;
  double layer_stage_t0 = layer_stage_profile ? ds4_now_seconds() : 0.0;
#define GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE(name) do { \
        if (ok && layer_stage_profile) { \
            ok = rocm_graph_layer_stage_profile_boundary("ffn", (name), il, pos0, n_tokens, &layer_stage_t0); \
        } \
    } while (0)

    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    /* Official Block.forward: the FFN collapses with THIS layer's attention
     * pre (written by the attention half into batch_hc_split_a). */
    ds4_gpu_tensor *hc_split_a_view = ds4_gpu_tensor_view(
            g->batch_hc_split_a, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *ffn_cur_view = ds4_gpu_tensor_view(
            g->batch_ffn_cur, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *next_hc_view = ds4_gpu_tensor_view(
            g->batch_next_hc, 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && ffn_cur_view && next_hc_view;
    bool hc_pre_fused = false;
    if (ok && n_tokens >= 128u &&
        layer->hc_ffn_fn->type == DS4_TENSOR_F16) {
      hc_pre_fused = true;
      ok = ds4_gpu_hc_norm_mix_split_weighted_sum_tensor(
               ffn_cur_view, hc_mix_view, hc_split_view, g->batch_after_attn_hc,
               model->map, model->size, layer->hc_ffn_fn->abs_offset,
               layer->hc_ffn_scale->abs_offset, layer->hc_ffn_base->abs_offset,
               DS4_N_EMBD, DS4_N_HC, n_tokens, DS4_N_HC_SINKHORN_ITER,
               DS4_HC_EPS, DS4_RMS_EPS,
               hc_split_a_view, 24u) != 0;
    }
    if (ok && !hc_pre_fused)
      ok = ds4_gpu_rms_norm_plain_rows_tensor(
               g->batch_flat_hc, g->batch_after_attn_hc, (uint32_t)hc_dim,
               n_tokens, DS4_RMS_EPS) != 0;
    if (ok && !hc_pre_fused) {
      ok = rocm_graph_matmul_hc_fn_tensor(hc_mix_view, model,
                                     layer->hc_ffn_fn, hc_dim,
                                     mix_hc, g->batch_flat_hc, n_tokens);
    }
    if (ok && !hc_pre_fused) {
        /*
         * The decode fusion is row-independent, so a W2-W8 grid is exact to
         * launching the C1 kernel once per session and also avoids a separate
         * normalization pass.
         */
        ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(
                 ffn_cur_view,
                 g->batch_ffn_norm,
                 NULL,
                 NULL,
                 hc_split_view,
                 hc_mix_view,
                 g->batch_after_attn_hc,
                 model->map,
                 model->size,
                 layer->hc_ffn_scale->abs_offset,
                 layer->hc_ffn_base->abs_offset,
                 layer->ffn_norm->abs_offset,
                 DS4_N_EMBD,
                 DS4_N_HC,
                 DS4_N_HC_SINKHORN_ITER,
                 DS4_HC_EPS,
                 DS4_RMS_EPS,
                 hc_split_a_view, 24u) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("hc_pre");
    rocm_graph_probe_layer_tensor(ffn_cur_view,
                                  (uint64_t)n_tokens * DS4_N_EMBD,
                                  "ffn cur", il);
    if (ok && hc_pre_fused)
      ok = ds4_gpu_rms_norm_weight_rows_tensor(
               g->batch_ffn_norm, g->batch_ffn_cur, model->map, model->size,
               layer->ffn_norm->abs_offset, DS4_N_EMBD, n_tokens,
               DS4_RMS_EPS) != 0;
    GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("norm");
    if (ok && group_count != 0u) {
      const uint64_t logits_row_bytes = (uint64_t)DS4_N_EXPERT * sizeof(float);
      const uint64_t selected_row_bytes =
          (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t);
      const uint64_t weights_row_bytes =
          (uint64_t)DS4_N_EXPERT_USED * sizeof(float);
      for (uint32_t group = 0; ok && group < group_count; ++group) {
        const uint32_t row0 = group_offsets[group];
        const uint32_t group_rows =
            group_offsets[group + 1u] - row0;
        ds4_gpu_tensor* logits_view = ds4_gpu_tensor_view(
            g->batch_router_logits, (uint64_t)row0 * logits_row_bytes,
            (uint64_t)group_rows * logits_row_bytes);
        ds4_gpu_tensor* probs_view = ds4_gpu_tensor_view(
            g->batch_router_probs, (uint64_t)row0 * logits_row_bytes,
            (uint64_t)group_rows * logits_row_bytes);
        ds4_gpu_tensor* selected_view = ds4_gpu_tensor_view(
            g->batch_router_selected, (uint64_t)row0 * selected_row_bytes,
            (uint64_t)group_rows * selected_row_bytes);
        ds4_gpu_tensor* weights_view = ds4_gpu_tensor_view(
            g->batch_router_weights, (uint64_t)row0 * weights_row_bytes,
            (uint64_t)group_rows * weights_row_bytes);
        ds4_gpu_tensor* norm_view = ds4_gpu_tensor_view(
            g->batch_ffn_norm, (uint64_t)row0 * DS4_N_EMBD * sizeof(float),
            (uint64_t)group_rows * DS4_N_EMBD * sizeof(float));
        ds4_gpu_tensor* tokens_view =
            ds4_gpu_tensor_view(tokens, (uint64_t)row0 * sizeof(int32_t),
                                (uint64_t)group_rows * sizeof(int32_t));
        ok = logits_view && probs_view && selected_view && weights_view &&
             norm_view && tokens_view;
        if (ok) {
          ok = rocm_graph_matmul_plain_tensor(
              logits_view, model, layer->ffn_gate_inp, DS4_N_EMBD, DS4_N_EXPERT,
              norm_view, group_rows);
        }
        if (ok) {
          ok = ds4_gpu_router_select_batch_tensor(
                   selected_view, weights_view, probs_view, model->map,
                   model->size,
                   layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset
                                          : 0,
                   layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset
                                           : 0,
                   layer->ffn_gate_tid2eid
                       ? static_cast<uint32_t>(layer->ffn_gate_tid2eid->dim[1])
                       : 0,
                   0, 0, layer->ffn_exp_probs_b != nullptr,
                   layer->ffn_gate_tid2eid != nullptr, logits_view, tokens_view,
                   DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_EXPERT_WEIGHT_SCALE,
                   group_rows) != 0;
        }
        ds4_gpu_tensor_free(tokens_view);
        ds4_gpu_tensor_free(norm_view);
        ds4_gpu_tensor_free(weights_view);
        ds4_gpu_tensor_free(selected_view);
        ds4_gpu_tensor_free(probs_view);
        ds4_gpu_tensor_free(logits_view);
      }
    } else {
      if (ok) {
        ok = rocm_graph_matmul_plain_tensor(
            g->batch_router_logits, model, layer->ffn_gate_inp, DS4_N_EMBD,
            DS4_N_EXPERT, g->batch_ffn_norm, n_tokens);
      }
      if (ok) {
        ok =
            ds4_gpu_router_select_batch_tensor(
                g->batch_router_selected, g->batch_router_weights,
                g->batch_router_probs, model->map, model->size,
                layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset
                                        : 0,
                layer->ffn_gate_tid2eid
                    ? static_cast<uint32_t>(layer->ffn_gate_tid2eid->dim[1])
                    : 0,
                0, 0, layer->ffn_exp_probs_b != nullptr,
                layer->ffn_gate_tid2eid != nullptr, g->batch_router_logits,
                tokens, DS4_N_EXPERT, DS4_N_EXPERT_USED,
                DS4_EXPERT_WEIGHT_SCALE, n_tokens) != 0;
      }
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("router");
    if (getenv("DS4_H_DUMP")) {
        ds4_gpu_tensor *fv = ds4_gpu_tensor_view(
            g->batch_ffn_norm, 0, (uint64_t)n_tokens * DS4_N_EMBD * 4u);
        if (fv) { char fn[80]; snprintf(fn, sizeof fn, "/tmp/hb_all_fn_%02u.bin", il);
                  ds4_dbg_dump(fv, fn, (uint32_t)(n_tokens * DS4_N_EMBD)); }
        if (fv) ds4_gpu_tensor_free(fv);
    }
    rocm_graph_probe_layer_tensor(g->batch_ffn_norm,
                                  (uint64_t)n_tokens * DS4_N_EMBD,
                                  "ffn norm", il);
    rocm_graph_probe_layer_tensor(g->batch_router_logits,
                                  (uint64_t)n_tokens * DS4_N_EXPERT,
                                  "router logits", il);

    /* Ask the routed MoE to leave its per-expert F16 rows unsummed so the
     * hyper-connection expansion below can fold the 6-way sum in, saving one
     * round trip through batch_routed_out. Advisory: only the F16 down route can
     * do it, so the outcome is read back after the call. */
    if (ok) ds4_gpu_set_routed_defer_sum(n_tokens >= 128u ? 1 : 0);
    if (ok) {
      ok = ds4_gpu_routed_moe_batch_tensor(
               g->batch_routed_out, g->batch_routed_gate, g->batch_routed_up,
               g->batch_routed_mid, g->batch_routed_down, model->map,
               model->size, layer->ffn_gate_exps->abs_offset,
               layer->ffn_up_exps->abs_offset, layer->ffn_down_exps->abs_offset,
               layer->ffn_gate_exps->type, layer->ffn_down_exps->type,
               gate_expert_bytes, gate_row_bytes, down_expert_bytes,
               down_row_bytes, static_cast<uint32_t>(expert_in_dim),
               static_cast<uint32_t>(down_in_dim),
               static_cast<uint32_t>(routed_out_dim), g->batch_router_selected,
               g->batch_router_weights, DS4_N_EXPERT, DS4_N_EXPERT_USED,
               DS4_SWIGLU_CLAMP_EXP, g->batch_ffn_norm, n_tokens) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("routed_moe");
    if (getenv("DS4_H_DUMP")) {
        ds4_gpu_tensor *rv = ds4_gpu_tensor_view(
            g->batch_router_logits, 0, (uint64_t)n_tokens * DS4_N_EXPERT * 4u);
        if (rv) { char fn[80]; snprintf(fn, sizeof fn, "/tmp/hb_all_rlg_%02u.bin", il);
                  ds4_dbg_dump(rv, fn, (uint32_t)(n_tokens * DS4_N_EXPERT)); }
        if (rv) ds4_gpu_tensor_free(rv);
        rv = ds4_gpu_tensor_view(g->batch_router_selected, 0,
                                 (uint64_t)n_tokens * DS4_N_EXPERT_USED * 4u);
        if (rv) { char fn[80]; snprintf(fn, sizeof fn, "/tmp/hb_all_rsel_%02u.bin", il);
                  ds4_dbg_dump(rv, fn, (uint32_t)(n_tokens * DS4_N_EXPERT_USED)); }
        if (rv) ds4_gpu_tensor_free(rv);
        rv = ds4_gpu_tensor_view(g->batch_router_weights, 0,
                                 (uint64_t)n_tokens * DS4_N_EXPERT_USED * 4u);
        if (rv) { char fn[80]; snprintf(fn, sizeof fn, "/tmp/hb_all_rw_%02u.bin", il);
                  ds4_dbg_dump(rv, fn, (uint32_t)(n_tokens * DS4_N_EXPERT_USED)); }
        if (rv) ds4_gpu_tensor_free(rv);
        rv = ds4_gpu_tensor_view(g->batch_routed_out, 0,
                                 (uint64_t)n_tokens * routed_out_dim * 4u);
        if (rv) { char fn[80]; snprintf(fn, sizeof fn, "/tmp/hb_all_ro_%02u.bin", il);
                  ds4_dbg_dump(rv, fn, (uint32_t)(n_tokens * routed_out_dim)); }
        if (rv) ds4_gpu_tensor_free(rv);
    }
    rocm_graph_probe_layer_tensor(g->batch_routed_out,
                                  (uint64_t)n_tokens * routed_out_dim,
                                  "routed out", il);
    if (il == 0 && getenv("GUFO_DEEPSEEK_ROCM_NAN_PROBE")) {
        ds4_gpu_synchronize();
        float h[4096];
        const char *gs = (const char *)model->map +
                         layer->ffn_gate_exps->abs_offset;
        int gf = 0;
        for (uint64_t e = 0; e < 512; e++) {
            float f;
            memcpy(&f, gs + e * 4, 4);
            if (isfinite(f)) gf++;
        }
        int rc = ds4_gpu_readback_f32(g->batch_routed_mid, 0, 4096, h);
        int bad = 0;
        for (int i = 0; i < 4096; i++)
            if (!isfinite(h[i])) bad++;
        fprintf(stderr,
                "ds4: moe HOST midbad=%d/4096 rc=%d mid0=%f sel0=%u sel63=%u "
                "w0=%f gtypes=%u/%u\n",
                bad, rc, (double)h[0], 0u, 0u,
                layer->ffn_gate_exps->type, layer->ffn_down_exps->type);
    }
    if (ok && layer->ffn_gate_shexp->type == DS4_TENSOR_Q5_K) {
      ok = ds4_gpu_shared_gate_up_swiglu_q5_k_tensor(
               g->batch_shared_mid, model->map, model->size,
               layer->ffn_gate_shexp->abs_offset,
               layer->ffn_up_shexp->abs_offset, DS4_N_EMBD, shared_dim,
               g->batch_ffn_norm, DS4_SWIGLU_CLAMP_EXP, n_tokens) != 0;
      GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("shared_gate_up");
      if (ok)
        ok = ds4_gpu_shared_down_q5_k_tensor(
                 g->batch_shared_out, model->map, model->size,
                 layer->ffn_down_shexp->abs_offset, shared_dim, DS4_N_EMBD,
                 g->batch_shared_mid, n_tokens) != 0;
    } else if (ok && n_tokens <= 136u && (n_tokens <= 16u || group_count != 0u)) {
      ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
               g->batch_shared_gate, g->batch_shared_up, g->batch_shared_mid, NULL,
               model->map, model->size, layer->ffn_gate_shexp->abs_offset,
               layer->ffn_up_shexp->abs_offset, DS4_N_EMBD, shared_dim,
               g->batch_ffn_norm, NULL, DS4_SWIGLU_CLAMP_EXP, n_tokens) != 0;
    } else {
      if (ok)
        ok = rocm_graph_matmul_q8_0_named_tensor(
            "shared_gate", il, pos0, g->batch_shared_gate, model,
            layer->ffn_gate_shexp, DS4_N_EMBD, shared_dim, g->batch_ffn_norm,
            n_tokens);
      if (ok)
        ok = rocm_graph_matmul_q8_0_named_tensor(
            "shared_up", il, pos0, g->batch_shared_up, model,
            layer->ffn_up_shexp, DS4_N_EMBD, shared_dim, g->batch_ffn_norm,
            n_tokens);
      GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("shared_gate_up");
      if (ok)
        ok = ds4_gpu_swiglu_tensor(g->batch_shared_mid, g->batch_shared_gate,
                                   g->batch_shared_up,
                                   (uint32_t)((uint64_t)n_tokens * shared_dim),
                                   DS4_SWIGLU_CLAMP_EXP, 1.0f) != 0;
    }
    if (ok && layer->ffn_down_shexp->type != DS4_TENSOR_Q5_K)
      ok = rocm_graph_matmul_q8_0_named_tensor("shared_down",
                                                      il,
                                                      pos0,
                                                      g->batch_shared_out,
                                                      model,
                                                      layer->ffn_down_shexp,
                                                      shared_dim,
                                                      DS4_N_EMBD,
                                                      g->batch_shared_mid,
                                                      n_tokens);
    GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("shared_down");
    rocm_graph_probe_layer_tensor(g->batch_shared_out,
                                  (uint64_t)n_tokens * DS4_N_EMBD,
                                  "shared out", il);
    if (getenv("DS4_DEBUG_STATE") && il >= 9u && il <= 13u) {
      const uint64_t last = (uint64_t)(n_tokens - 1u) * DS4_N_EMBD;
      fprintf(stderr, "ds4: vec L%02u routed: ", il);
      ds4_dbg_vec(g->batch_routed_out, last, DS4_N_EMBD, "ro");
      fprintf(stderr, "ds4: vec L%02u shared: ", il);
      ds4_dbg_vec(g->batch_shared_out, last, DS4_N_EMBD, "sh");
    }

    const bool routed_sum_deferred = ds4_gpu_routed_sum_deferred() != 0;
    ds4_gpu_set_routed_defer_sum(0);
    if (ok && routed_sum_deferred) {
        ok = ds4_gpu_hc_expand_add_split_moesum_tensor(next_hc_view,
                                                         g->batch_routed_down,
                                                         g->batch_shared_out,
                                                         g->batch_after_attn_hc,
                                                         hc_split_view,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC,
                                                         DS4_N_EXPERT_USED,
                                                         n_tokens) != 0;
        if (!ok) {
          fprintf(stderr,
                  "ds4: fused routed sum expansion rejected the shape\n");
        }
    } else if (ok) {
        ok = ds4_gpu_hc_expand_add_split_tensor(next_hc_view,
                                                  g->batch_routed_out,
                                                  g->batch_shared_out,
                                                  g->batch_after_attn_hc,
                                                  hc_split_view,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("hc_post");
    rocm_graph_probe_layer_tensor(next_hc_view,
                                  (uint64_t)n_tokens * 4ull * DS4_N_EMBD,
                                  "next hc", il);
    ds4_gpu_tensor_free(next_hc_view);
    ds4_gpu_tensor_free(ffn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_split_a_view);
    ds4_gpu_tensor_free(hc_mix_view);
#undef GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE
    return ok;
}

/* Encode one complete layer for prefill by chaining attention and FFN batches. */
static void ds4_dbg_layer_state(uint32_t il, uint32_t nt, const char *tag,
                                const ds4_gpu_tensor *t) {
  if (!getenv("DS4_DEBUG_STATE") ||
      (il != 0u && il != 1u && il != 14u && il != 19u && il != 39u) ||
      !t || nt == 0u)
    return;
  ds4_gpu_synchronize();
  fprintf(stderr, "ds4: state[%s] L%u last", tag, il);
  for (uint32_t hc = 0; hc < DS4_N_HC; ++hc) {
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    const uint64_t off = ((uint64_t)(nt - 1u) * DS4_N_HC + hc) * DS4_N_EMBD;
    if (ds4_gpu_readback_f32(t, off, 4, v))
      fprintf(stderr, " hc%u:%.5f,%.5f,%.5f,%.5f", hc, v[0], v[1], v[2], v[3]);
  }
  fprintf(stderr, "\n");
}

/* Per-layer stream RMS for the last token: growth-curve bisect for the
 * residual explosion (healthy models keep this in single/double digits). */
static void ds4_dbg_layer_rms(uint32_t il, uint32_t nt,
                              const ds4_gpu_tensor *t, const char *sfx) {
  if (!getenv("DS4_DEBUG_STATE") || !t || nt == 0u) return;
  static float buf[DS4_N_EMBD];
  fprintf(stderr, "ds4: rms L%02u%s", il, sfx);
  for (uint32_t hc = 0; hc < DS4_N_HC; ++hc) {
    const uint64_t off = ((uint64_t)(nt - 1u) * DS4_N_HC + hc) * DS4_N_EMBD;
    if (!ds4_gpu_readback_f32(t, off, DS4_N_EMBD, buf)) { fprintf(stderr, " ?"); continue; }
    double s = 0.0;
    uint32_t bad = 0;
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) {
      if (!(buf[i] == buf[i])) bad++;
      s += (double)buf[i] * (double)buf[i];
    }
    fprintf(stderr, " %u:%.3f%s", hc, sqrt(s / DS4_N_EMBD), bad ? "!NaN" : "");
  }
  fprintf(stderr, "\n");
}

/* Distribution probe over a host-side logits row: identifies whether the
 * constant-token loop is a saturated argmax (broken projection/mix) or a
 * flat distribution (missing signal). */
static void ds4_dbg_logits_h(const float *h, const char *tag, uint32_t pos) {
  if (!getenv("DS4_DEBUG_STATE") || !h) return;
  int a = -1, b = -1, big = 0;
  float av = -1e30f, bv = -1e30f;
  for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
    const float v = h[i];
    if (v > 10.0f) big++;
    if (v > av) { bv = av; b = a; av = v; a = (int)i; }
    else if (v > bv) { bv = v; b = (int)i; }
  }
  fprintf(stderr, "ds4: logits[%s p=%u] top1=%d(%.3f) top2=%d(%.3f) n>10=%d\n",
          tag, pos, a, av, b, bv, big);
}

static bool rocm_graph_encode_layer_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    ds4_dbg_layer_state(il, n_tokens, "in ", g->batch_cur_hc);
    ds4_dbg_layer_rms(il, n_tokens, g->batch_cur_hc, "");
    const bool hb_dump = getenv("DS4_H_DUMP") != NULL;
    auto hb_row = [&](ds4_gpu_tensor *t, uint64_t stride_f32, const char *nm) {
        if (!hb_dump || !t) return;
        char hbp[80];
        snprintf(hbp, sizeof hbp, "/tmp/hb_%s_L%02u.bin", nm, il);
        ds4_gpu_tensor *v = ds4_gpu_tensor_view(
            t, (uint64_t)(n_tokens - 1u) * stride_f32 * 4u, stride_f32 * 4u);
        if (v) ds4_dbg_dump(v, hbp, (uint32_t)stride_f32);
    };
    hb_row(g->batch_cur_hc, 4ull * DS4_N_EMBD, "in");
    if (hb_dump) {
        ds4_gpu_tensor *iv = ds4_gpu_tensor_view(
            g->batch_cur_hc, 0, (uint64_t)n_tokens * 4ull * DS4_N_EMBD * 4ull);
        if (iv) { char fn[80]; snprintf(fn, sizeof fn, "/tmp/hb_all_in_%02u.bin", il);
                  ds4_dbg_dump(iv, fn, (uint32_t)(n_tokens * 4u * DS4_N_EMBD)); }
        ds4_gpu_tensor_free(iv);
    }
    if (hb_dump && il == 0u && g->prefill_tokens) {
        ds4_gpu_tensor *tv = ds4_gpu_tensor_view(
            g->prefill_tokens, 0, (uint64_t)n_tokens * 4u);
        if (tv) ds4_dbg_dump(tv, "/tmp/hb_tokens.bin", (uint32_t)n_tokens);
    }
    bool ok = rocm_graph_encode_layer_attention_batch(g, model, layer, il, pos0, n_tokens);
    if (!ok) fprintf(stderr, "ds4: prefill layer %u attention failed\n", il);
    rocm_graph_probe_layer_tensor(g->batch_after_attn_hc,
                                  (uint64_t)n_tokens * 4ull * DS4_N_EMBD,
                                  "after attn", il);
    ds4_dbg_layer_state(il, n_tokens, "attn", g->batch_after_attn_hc);
    if (il >= 9u && il <= 13u)
        ds4_dbg_layer_rms(il, n_tokens, g->batch_after_attn_hc, "a");
    hb_row(g->batch_after_attn_hc, 4ull * DS4_N_EMBD, "x2");
    hb_row(g->batch_attn_out, DS4_N_EMBD, "ay");
    if (hb_dump && il == 0u) {
        ds4_gpu_tensor *cv = ds4_gpu_tensor_view(
            g->batch_hc_split_a, 0, (uint64_t)n_tokens * 24u * 4u);
        if (cv) ds4_dbg_dump(cv, "/tmp/hb_ca_full_L00.bin", (uint32_t)(n_tokens * 24u));
        ds4_gpu_tensor_free(cv);
    } else {
        hb_row(g->batch_hc_split_a, 24u, "ca");
    }
    const bool ok_ffn = ok && rocm_graph_encode_layer_ffn_batch(g, model, layer, il, pos0, n_tokens);
    if (ok && !ok_ffn) fprintf(stderr, "ds4: prefill layer %u ffn failed\n", il);
    ok = ok_ffn;
    ds4_dbg_layer_state(il, n_tokens, "out", g->batch_next_hc);
    hb_row(g->batch_ffn_cur, DS4_N_EMBD, "yf");
    hb_row(g->batch_routed_out, DS4_N_EMBD, "ro");
    hb_row(g->batch_shared_out, DS4_N_EMBD, "sh");
    hb_row(g->batch_hc_split, 24u, "cf");
    if (hb_dump) {
        ds4_gpu_tensor *fv = ds4_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_tokens * 24u * 4u);
        if (fv) { char fn[80]; snprintf(fn, sizeof fn, "/tmp/hb_all_cf_%02u.bin", il);
                  ds4_dbg_dump(fv, fn, (uint32_t)(n_tokens * 24u)); }
        ds4_gpu_tensor_free(fv);
    }
    if (ok) {
        ds4_gpu_tensor *tmp = g->batch_cur_hc;
        g->batch_cur_hc = g->batch_next_hc;
        g->batch_next_hc = tmp;
    }
    hb_row(g->batch_cur_hc, 4ull * DS4_N_EMBD, "out");
    /* Capture after the swap so batch_cur_hc holds this layer's output. */
    if (ok) ok = rocm_graph_dspark_capture_batch_layer(g, il, pos0, n_tokens);
    return ok;
}

/* Execute one ROCm decode token and read back logits. */
static bool rocm_graph_eval_token_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        float                 *logits) {
    const bool profile = getenv("GUFO_DEEPSEEK_ROCM_GRAPH_TOKEN_PROFILE") != NULL;
    const double t0 = profile ? ds4_now_seconds() : 0.0;

    const int32_t engram_token = (int32_t)token;
    if (!rocm_graph_engram_prepare(g, model->map, &engram_token, 1, pos))
      return false;
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = rocm_graph_encode_token_raw_swa(g, model, weights, token, pos, logits != NULL);
    const double t_encoded = profile ? ds4_now_seconds() : 0.0;
    if (ok) ok = ds4_gpu_end_commands() != 0;
    const double t_done = profile ? ds4_now_seconds() : 0.0;

    if (ok && logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        if (ok) ds4_dbg_logits_h(logits, "decode", pos);
    }
    const double t_read = profile ? ds4_now_seconds() : 0.0;
    if (profile) {
        fprintf(stderr,
                "ds4: ROCm graph token pos=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms logits=%d\n",
                pos,
                (t_encoded - t0) * 1000.0,
                (t_done - t_encoded) * 1000.0,
                (t_read - t_done) * 1000.0,
                (t_read - t0) * 1000.0,
                logits != NULL);
    }
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: ROCm synchronize after graph eval failure also failed\n");
        }
    }
    return ok;
}

/* Greedy verifier helper.  Speculative decoding only needs the target model's
 * top token after most accepted draft rows; the full vocabulary row is needed
 * once, for the final committed state that normal sampling will continue from.
 * Keeping intermediate rows device-resident avoids turning verification into a
 * sequence of large CPU readbacks. */
static bool rocm_graph_reset_prefill_state(ds4_gpu_graph *g) {
    memset(g->layer_n_comp, 0, sizeof(g->layer_n_comp));
    memset(g->layer_n_index_comp, 0, sizeof(g->layer_n_index_comp));
    g->dspark_context_len = 0;
    g->dspark_capture_mask = 0;
    g->dspark_capture_batch_mask = 0;
    g->dspark_capture_batch_start = 0;
    g->dspark_capture_batch_tokens = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
        const uint64_t attn_rows = (uint64_t)coff * ratio;
        if (!rocm_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows)) return false;
        if (!rocm_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF, attn_width * attn_rows)) return false;
        if (ratio == 4) {
            const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
            const uint64_t index_rows = (uint64_t)coff * ratio;
            if (!rocm_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows)) return false;
            if (!rocm_tensor_fill_f32(g->layer_index_state_score[il], DS4_NEG_INF, index_width * index_rows)) return false;
        }
    }
    return true;
}

/* Execute ROCm prefill in layer-major order so intermediate activations stay
 * on the GPU and cache state is built exactly once. */
static bool rocm_graph_prefill_layer_major(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;
    if (start > (uint32_t)prompt->len) return false;
    if (n_tokens > (uint32_t)prompt->len - start) return false;

    // Grow before prefill dispatch and leave one raw window for decoding.
    // Normal decode only grows when its actual rows exceed the allocation.
    for (uint32_t il = 0; il < DS4_N_LAYER; ++il) {
      if (!rocm_graph_reserve_compressed_position(
              g, il, static_cast<uint64_t>(start) + n_tokens + DS4_N_SWA))
        return false;
    }

    bool ok = rocm_graph_upload_prompt_tokens(g->prefill_tokens, prompt, start, n_tokens);
    if (!ok) return false;

    const bool profile =
        getenv("GUFO_DEEPSEEK_ROCM_GRAPH_PREFILL_PROFILE") != NULL;
    const double t0 = profile ? ds4_now_seconds() : 0.0;

    ok = rocm_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                 g->prefill_tokens,
                                                 model,
                                                 weights,
                                                 prompt,
                                                 start,
                                                 n_tokens);
    if (ok) {
      std::vector<int32_t> engram_toks(n_tokens);
      for (uint32_t i = 0; i < n_tokens; ++i)
        engram_toks[i] = (int32_t)prompt->v[start + i];
      if (getenv("DS4_ENGRAM_DEBUG")) {
        fprintf(stderr, "ds4: prompt ids n=%u start=%u:", n_tokens, start);
        for (uint32_t i = 0; i < n_tokens && i < 32u; ++i)
          fprintf(stderr, " %d", engram_toks[i]);
        fprintf(stderr, "\n");
      }
      ok = rocm_graph_engram_prepare(g, model->map, engram_toks.data(), n_tokens, start);
    }
    if (ok)
      ok = ds4_gpu_begin_commands() != 0;
    /* V4.1: one published prefill top-k and one chunk candidate mask live
     * per prefill call (chunk). */
    g->pf_sel_owner = UINT32_MAX;
    g->pf_cand_valid = false;
    g->pf_cand_n = 0;
    if (ok && logits && getenv("GUFO_DEEPSEEK_ROCM_NAN_PROBE")) {
        unsigned long long c = 0, fi = ~0ull;
        ds4_gpu_probe_nan(g->batch_cur_hc, (uint64_t)n_tokens * 4ull * DS4_N_EMBD,
                          "input embd hc", &c, &fi);
    }
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
      // Keep feature capture and the HC swap in the common layer encoder.
      ok = rocm_graph_encode_layer_batch(g, model, &weights->layer[il], il,
                                         start, n_tokens);
      {
        static int nan_probe = -1;
        if (nan_probe < 0)
            nan_probe = getenv("GUFO_DEEPSEEK_ROCM_NAN_PROBE") != NULL;
        if (nan_probe && ok && logits) {
            char tag[64];
            snprintf(tag, sizeof(tag), "prefill hc layer=%u", il);
            unsigned long long c = 0, fi = ~0ull;
            ds4_gpu_probe_nan(g->batch_cur_hc,
                              (uint64_t)n_tokens * 4ull * DS4_N_EMBD, tag,
                              &c, &fi);
        }
      }
      if (show_progress) {
        fprintf(stderr, "ds4: gpu prefill layer %u/%u\r", il + 1,
                (uint32_t)DS4_N_LAYER);
        fflush(stderr);
      }
    }
    if (show_progress) fputc('\n', stderr);

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint32_t output_row = n_tokens - 1u;
    ds4_gpu_tensor *saved_cur = g->cur_hc;
    ds4_gpu_tensor *last_hc = NULL;
    if (ok && logits) {
      last_hc = rocm_graph_tensor_row_view(g->batch_cur_hc, output_row, hc_dim);
      ok = last_hc != NULL;
    }
    if (ok && logits) {
        g->cur_hc = last_hc;
        /* batch_hc_mix is laid out as raw FFN mix rows of DS4_N_HC floats
         * (see hc_mix_view above the layers), so the head takes row
         * output_row*DS4_N_HC.  The previous offset (row*24) pointed into the
         * following tokens' rows, not comb, and fed the head a garbage mix. */
        ds4_gpu_tensor *mix_pre = ds4_gpu_tensor_view(
            g->batch_hc_mix,
            (uint64_t)output_row * DS4_N_HC * sizeof(float),
            DS4_N_HC * sizeof(float));
        ok = mix_pre != NULL;
        if (ok)
            ok = rocm_graph_encode_output_head(g, model, weights,
                                               weights->output->dim[1], mix_pre);
        ds4_head_probe_file("batch5654", g->logits,
                            weights->output->dim[1], (uint32_t)output_row);
        ds4_gpu_tensor_free(mix_pre);
        g->cur_hc = saved_cur;
    }

    const double t_encoded = profile ? ds4_now_seconds() : 0.0;
    if (ok)
      ok = ds4_gpu_end_commands() != 0;
    const double t_done = profile ? ds4_now_seconds() : 0.0;
    if (last_hc) ds4_gpu_tensor_free(last_hc);
    if (!ok) {
      if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr,
                "ds4: ROCm synchronize after prefill failure also failed\n");
      }
      return false;
    }

    const double t_before_read = profile ? ds4_now_seconds() : 0.0;
    if (logits) {
        if (getenv("GUFO_DEEPSEEK_ROCM_NAN_PROBE")) {
            unsigned long long c = 0, fi = ~0ull;
            ds4_gpu_probe_nan(g->logits, DS4_N_VOCAB, "logits prefill", &c,
                              &fi);
        }
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        if (ok) ds4_dbg_logits_h(logits, "prefill", start + n_tokens - 1u);
    }
    if (profile) {
        const double t_read = ds4_now_seconds();
        const double encode_s = t_encoded - t0;
        const double execute_s = t_done - t_encoded;
        fprintf(stderr,
                "ds4: gpu graph prefill total tokens=%u encode=%.3f ms "
                "execute=%.3f ms read=%.3f ms total=%.3f ms\n",
                n_tokens, encode_s * 1000.0, execute_s * 1000.0,
                (t_read - t_before_read) * 1000.0, (t_read - t0) * 1000.0);
    }
    return ok;
}

static bool rocm_graph_prefill_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress) {
    if (n_tokens <= 0 || n_tokens > prompt->len) return false;
    if ((uint32_t)n_tokens > g->prefill_cap) return false;
    const uint32_t rows = (uint32_t)n_tokens;
    if (!rocm_graph_dspark_resize_capture(g, rows)) return false;
    bool ok = rocm_graph_prefill_layer_major(g,
                                             model,
                                             weights,
                                             prompt,
                                             0,
                                             rows,
                                             logits,
                                             show_progress);
    if (ok && g->dspark && g->dspark_capture_enabled) {
        ok = rocm_graph_dspark_capture_complete(g) &&
             rocm_graph_dspark_inject(g, 0, rows);
    }
    if (g->dspark && g->dspark_capture_enabled) {
        const uint32_t steady_rows = g->dspark->block_size + 1u;
        if (!rocm_graph_dspark_resize_capture(g, steady_rows)) ok = false;
    }
    return ok;
}

/* Prefill a contiguous token range in fixed-size chunks.
 *
 * The common case starts at token zero, but server sessions also use this to
 * extend an existing KV cache with a long suffix.  Resumed chunks are aligned
 * to the same absolute prefill-cap boundaries used by a cold full prompt, so
 * compression windows and row finalization follow the same schedule after the
 * cached prefix.
 */
static bool rocm_graph_prefill_chunked_range(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress) {
    if (n_tokens == 0 || g->prefill_cap == 0) return false;
    if (start > (uint32_t)prompt->len) return false;
    if (n_tokens > (uint32_t)prompt->len - start) return false;

    uint32_t chunk_cap = g->prefill_cap;
    if (start != 0 && chunk_cap > g->raw_cap) chunk_cap = g->raw_cap;
    if (chunk_cap == 0) return false;

    const bool profile = getenv("GUFO_DEEPSEEK_ROCM_GRAPH_PREFILL_PROFILE") != NULL;
    const double t0 = profile ? ds4_now_seconds() : 0.0;
    const uint32_t end = start + n_tokens;

    for (uint32_t pos0 = start; pos0 < end; ) {
        const uint32_t remaining = end - pos0;
        uint32_t local_cap = chunk_cap;
        if (start != 0 && g->prefill_cap != 0) {
            const uint32_t mod = pos0 % g->prefill_cap;
            if (mod != 0) {
                const uint32_t to_boundary = g->prefill_cap - mod;
                if (to_boundary < local_cap) local_cap = to_boundary;
            }
        }
        const uint32_t chunk = remaining < local_cap ? remaining : local_cap;
        const uint32_t chunk_end = pos0 + chunk;
        float *chunk_logits = chunk_end == end ? logits : nullptr;
        bool ok = (!g->dspark_capture_enabled ||
                   rocm_graph_dspark_resize_capture(g, chunk)) &&
                  rocm_graph_prefill_layer_major(g,
                                                 model,
                                                 weights,
                                                 prompt,
                                                 pos0,
                                                 chunk,
                                                 chunk_logits,
                                                 show_progress);
        if (ok && g->dspark && g->dspark_capture_enabled) {
            ok = rocm_graph_dspark_capture_complete(g) &&
                 rocm_graph_dspark_inject(g, pos0, chunk);
        }
        if (!ok) {
            if (g->dspark && g->dspark_capture_enabled) {
                (void)rocm_graph_dspark_resize_capture(
                    g, g->dspark->block_size + 1u);
            }
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: ROCm synchronize after chunked prefill failure also failed\n");
            }
            return false;
        }
        pos0 = chunk_end;
    }
    if (g->dspark && g->dspark_capture_enabled &&
        !rocm_graph_dspark_resize_capture(g, g->dspark->block_size + 1u)) {
      return false;
    }
    if (show_progress) fputc('\n', stderr);
    if (profile) {
        const double t_read = ds4_now_seconds();
        fprintf(stderr,
                "ds4: gpu chunked prefill start=%u tokens=%u chunk=%u total=%.3f ms\n",
                start,
                n_tokens,
                chunk_cap,
                (t_read - t0) * 1000.0);
    }
    return true;
}

/* Long prompts are prefetched in fixed-size chunks.  Chunks bound transient
 * attention buffers while preserving the same final KV/cache state. */
static bool rocm_graph_prefill_chunked(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress) {
    if (n_tokens <= 0) return false;
    return rocm_graph_prefill_chunked_range(g,
                                             model,
                                             weights,
                                             prompt,
                                             0,
                                             (uint32_t)n_tokens,
                                             logits,
                                             show_progress);
}

/* =========================================================================
 * DSpark Speculative Verification.
 * =========================================================================
 *
 * Speculative decoding only pays off if the target can score a whole drafted
 * suffix for roughly the cost of one token.  DS4 decode is dominated by
 * streaming weights: the dense attention projections, shared expert, and LM
 * head are read once per step regardless of how many rows are in flight, and
 * only the routed experts grow with the row count.  Verification therefore
 * reuses the batched layer path rather than looping the one-token decode path.
 *
 * The verifier writes into the same persistent caches as prefill, so a rejected
 * block must be undone.  That is cheap here: raw and compressed rows are
 * append-only and get overwritten, so only the row counters and the in-place
 * compressor frontier need saving.
 */

static uint32_t rocm_graph_spec_rows_cap(const ds4_gpu_graph *g) {
    return g ? g->spec_rows_cap : 0u;
}

static bool rocm_graph_spec_prepare(ds4_gpu_graph *g,
                                    const ds4_weights *weights,
                                    uint32_t rows_cap) {
    if (!g || !weights || rows_cap == 0 || rows_cap > DS4_SPEC_MAX_ROWS) return false;
    if (rows_cap > g->prefill_cap) return false;
    if (g->spec_rows_cap >= rows_cap && g->spec_logits && g->spec_row_tops &&
        g->spec_frontier_logits) {
      return true;
    }

    rocm_graph_spec_free(g);
    const uint64_t vocab_dim = weights->output->dim[1];
    g->spec_logits =
        ds4_gpu_tensor_alloc((uint64_t)rows_cap * vocab_dim * sizeof(float));
    g->spec_row_tops =
        ds4_gpu_tensor_alloc((uint64_t)rows_cap * sizeof(int32_t));
    g->spec_frontier_logits = ds4_gpu_tensor_alloc(
        (uint64_t)(rows_cap < 8u ? rows_cap : 8u) *
        vocab_dim * sizeof(float));
    bool ok = g->spec_logits != NULL && g->spec_row_tops != NULL &&
              g->spec_frontier_logits != NULL;

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t attn_bytes =
            (uint64_t)coff * DS4_N_HEAD_DIM * coff * ratio * sizeof(float);
        g->spec_saved_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_bytes);
        g->spec_saved_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_bytes);
        g->spec_prefix_attn_state_kv[il] =
            ds4_gpu_tensor_alloc(DS4_SPEC_PREFIX_SLOTS * attn_bytes);
        g->spec_prefix_attn_state_score[il] =
            ds4_gpu_tensor_alloc(DS4_SPEC_PREFIX_SLOTS * attn_bytes);
        ok = g->spec_saved_attn_state_kv[il] != NULL &&
             g->spec_saved_attn_state_score[il] != NULL &&
             g->spec_prefix_attn_state_kv[il] != NULL &&
             g->spec_prefix_attn_state_score[il] != NULL;
        if (ok && ratio == 4) {
            const uint64_t index_bytes =
                (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio * sizeof(float);
            g->spec_saved_index_state_kv[il] = ds4_gpu_tensor_alloc(index_bytes);
            g->spec_saved_index_state_score[il] = ds4_gpu_tensor_alloc(index_bytes);
            g->spec_prefix_index_state_kv[il] =
                ds4_gpu_tensor_alloc(DS4_SPEC_PREFIX_SLOTS * index_bytes);
            g->spec_prefix_index_state_score[il] =
                ds4_gpu_tensor_alloc(DS4_SPEC_PREFIX_SLOTS * index_bytes);
            ok = g->spec_saved_index_state_kv[il] != NULL &&
                 g->spec_saved_index_state_score[il] != NULL &&
                 g->spec_prefix_index_state_kv[il] != NULL &&
                 g->spec_prefix_index_state_score[il] != NULL;
        }
    }

    if (!ok) {
        fprintf(stderr, "ds4: ROCm failed to allocate DSpark verification state\n");
        rocm_graph_spec_free(g);
        return false;
    }
    g->spec_rows_cap = rows_cap;
    return true;
}

static bool rocm_graph_capture_prefix_attn_state(ds4_gpu_graph *g,
                                                 uint32_t il,
                                                 uint32_t slot) {
    if (!g->spec_capture_prefixes) return true;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    if (ratio == 0) return true;
    if (slot >= DS4_SPEC_PREFIX_SLOTS || !g->spec_prefix_attn_state_kv[il] ||
        !g->spec_prefix_attn_state_score[il]) {
      return false;
    }
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    const uint64_t bytes =
        (uint64_t)coff * DS4_N_HEAD_DIM * coff * ratio * sizeof(float);
    const uint64_t offset = (uint64_t)slot * bytes;
    g->spec_prefix_n_comp[slot][il] = g->layer_n_comp[il];
    return ds4_gpu_tensor_copy(g->spec_prefix_attn_state_kv[il],
                               offset,
                               g->layer_attn_state_kv[il],
                               0,
                               bytes) != 0 &&
           ds4_gpu_tensor_copy(g->spec_prefix_attn_state_score[il],
                               offset,
                               g->layer_attn_state_score[il],
                               0,
                               bytes) != 0;
}

static bool rocm_graph_capture_prefix_index_state(ds4_gpu_graph *g,
                                                  uint32_t il,
                                                  uint32_t slot) {
    if (!g->spec_capture_prefixes) return true;
    if (ds4_layer_compress_ratio(il) != 4) return true;
    if (slot >= DS4_SPEC_PREFIX_SLOTS || !g->spec_prefix_index_state_kv[il] ||
        !g->spec_prefix_index_state_score[il]) {
      return false;
    }
    constexpr uint32_t kCoff = 2u;
    constexpr uint32_t kRatio = 4u;
    const uint64_t bytes =
        (uint64_t)kCoff * DS4_N_INDEXER_HEAD_DIM * kCoff * kRatio *
        sizeof(float);
    const uint64_t offset = (uint64_t)slot * bytes;
    g->spec_prefix_n_index_comp[slot][il] =
        g->layer_n_index_comp[il];
    return ds4_gpu_tensor_copy(g->spec_prefix_index_state_kv[il],
                               offset,
                               g->layer_index_state_kv[il],
                               0,
                               bytes) != 0 &&
           ds4_gpu_tensor_copy(g->spec_prefix_index_state_score[il],
                               offset,
                               g->layer_index_state_score[il],
                               0,
                               bytes) != 0;
}

/*
 * Attach a DSpark support model to this session's graph.
 *
 * Everything here is allocated only when a drafter is attached. The steady
 * capture buffers cover one verification block; prompt prefill grows them
 * temporarily and shrinks them again after seeding the support KV ring.
 */
static bool rocm_graph_dspark_attach(ds4_gpu_graph *g,
                                     const ds4_weights *weights,
                                     const ds4_dspark_model *dspark) {
    if (!g || !weights || !dspark) return false;
    if (g->dspark == dspark) return true;
    if (dspark->n_stages == 0 || dspark->n_stages > DS4_DSPARK_MAX_STAGES) return false;
    if (dspark->block_size == 0 || dspark->block_size > DS4_DSPARK_MAX_BLOCK) return false;

    rocm_graph_dspark_free(g);
    /* The verifier's row storage doubles as the draft block's logits. */
    if (!rocm_graph_spec_prepare(g, weights, dspark->block_size + 1u)) return false;

    g->dspark = dspark;
    g->dspark_cache_cap = g->raw_cap;
    g->dspark_capture_rows_cap = dspark->block_size + 1u;
    g->dspark_capture_enabled = true;
    g->dspark_context_len = 0;
    const uint32_t feature_width = dspark->n_target_layers * DS4_N_EMBD;

    bool ok = true;
    for (uint32_t stage = 0; ok && stage < dspark->n_stages; stage++) {
        g->dspark_kv_cache[stage] = ds4_gpu_tensor_alloc(
                (uint64_t)g->dspark_cache_cap * DS4_N_HEAD_DIM * sizeof(float));
        ok = g->dspark_kv_cache[stage] != NULL;
    }
    if (ok) {
        g->dspark_features = ds4_gpu_tensor_alloc(
                (uint64_t)g->dspark_capture_rows_cap * feature_width *
                    sizeof(float));
        g->dspark_fused = ds4_gpu_tensor_alloc(
            (uint64_t)g->dspark_capture_rows_cap * DS4_N_EMBD * sizeof(float));
        g->dspark_markov_key = ds4_gpu_tensor_alloc(
                8u * sizeof(unsigned long long));
        g->dspark_markov_index =
                ds4_gpu_tensor_alloc(8u * sizeof(int32_t));
        ok = g->dspark_features && g->dspark_fused && g->dspark_markov_key &&
             g->dspark_markov_index;
    }
    if (!ok) {
        fprintf(stderr, "ds4: ROCm failed to allocate DSpark drafting state\n");
        rocm_graph_dspark_free(g);
        return false;
    }
    gufo::server::Logger::LogFormatted(
        gufo::server::LogLevel::kInfo, "ds4",
        "DSpark drafting attached stages=%u block=%u ring=%u rows "
        "features=%.1f MiB",
        dspark->n_stages,
        dspark->block_size,
        g->dspark_cache_cap,
        (double)((uint64_t)g->dspark_capture_rows_cap * feature_width *
                 sizeof(float)) /
            1048576.0);
    return true;
}

static bool rocm_graph_spec_copy_frontier(ds4_gpu_graph *g, bool save) {
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t attn_bytes =
            (uint64_t)coff * DS4_N_HEAD_DIM * coff * ratio * sizeof(float);
        ds4_gpu_tensor *live_kv = g->layer_attn_state_kv[il];
        ds4_gpu_tensor *live_score = g->layer_attn_state_score[il];
        ds4_gpu_tensor *saved_kv = g->spec_saved_attn_state_kv[il];
        ds4_gpu_tensor *saved_score = g->spec_saved_attn_state_score[il];
        if (!live_kv || !live_score || !saved_kv || !saved_score) return false;
        ds4_gpu_tensor *dst_kv = save ? saved_kv : live_kv;
        ds4_gpu_tensor *dst_score = save ? saved_score : live_score;
        const ds4_gpu_tensor *src_kv = save ? live_kv : saved_kv;
        const ds4_gpu_tensor *src_score = save ? live_score : saved_score;
        if (ds4_gpu_tensor_copy(dst_kv, 0, src_kv, 0, attn_bytes) == 0) return false;
        if (ds4_gpu_tensor_copy(dst_score, 0, src_score, 0, attn_bytes) == 0) return false;
        if (ratio != 4) continue;
        const uint64_t index_bytes =
            (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio * sizeof(float);
        ds4_gpu_tensor *live_index_kv = g->layer_index_state_kv[il];
        ds4_gpu_tensor *live_index_score = g->layer_index_state_score[il];
        ds4_gpu_tensor *saved_index_kv = g->spec_saved_index_state_kv[il];
        ds4_gpu_tensor *saved_index_score = g->spec_saved_index_state_score[il];
        if (!live_index_kv || !live_index_score || !saved_index_kv ||
            !saved_index_score) {
          return false;
        }
        if (ds4_gpu_tensor_copy(save ? saved_index_kv : live_index_kv,
                                0,
                                save ? live_index_kv : saved_index_kv,
                                0,
                                index_bytes) == 0) {
            return false;
        }
        if (ds4_gpu_tensor_copy(save ? saved_index_score : live_index_score,
                                0,
                                save ? live_index_score : saved_index_score,
                                0,
                                index_bytes) == 0) {
            return false;
        }
    }
    return true;
}

static bool rocm_graph_spec_frontier_save(ds4_gpu_graph *g) {
    if (!g || g->spec_rows_cap == 0) return false;
    memcpy(g->spec_saved_n_comp, g->layer_n_comp, sizeof(g->spec_saved_n_comp));
    memcpy(g->spec_saved_n_index_comp,
           g->layer_n_index_comp,
           sizeof(g->spec_saved_n_index_comp));
    g->spec_frontier_valid = rocm_graph_spec_copy_frontier(g, true);
    return g->spec_frontier_valid;
}

static bool rocm_graph_spec_frontier_restore(ds4_gpu_graph *g) {
    if (!g || !g->spec_frontier_valid) return false;
    memcpy(g->layer_n_comp, g->spec_saved_n_comp, sizeof(g->layer_n_comp));
    memcpy(g->layer_n_index_comp,
           g->spec_saved_n_index_comp,
           sizeof(g->layer_n_index_comp));
    return rocm_graph_spec_copy_frontier(g, false);
}

static bool rocm_graph_spec_frontier_commit_prefix(ds4_gpu_graph *g,
                                                   uint32_t prefix_len) {
    if (!g || prefix_len == 0 || prefix_len > DS4_SPEC_PREFIX_SLOTS) {
        return false;
    }
    const uint32_t slot = prefix_len - 1u;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t attn_bytes =
            (uint64_t)coff * DS4_N_HEAD_DIM * coff * ratio * sizeof(float);
        const uint64_t attn_offset = (uint64_t)slot * attn_bytes;
        g->layer_n_comp[il] = g->spec_prefix_n_comp[slot][il];
        if (ds4_gpu_tensor_copy(g->layer_attn_state_kv[il], 0,
                                g->spec_prefix_attn_state_kv[il], attn_offset,
                                attn_bytes) == 0 ||
            ds4_gpu_tensor_copy(g->layer_attn_state_score[il], 0,
                                g->spec_prefix_attn_state_score[il],
                                attn_offset, attn_bytes) == 0) {
          return false;
        }
        if (ratio != 4) continue;
        const uint64_t index_bytes =
            (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio *
            sizeof(float);
        const uint64_t index_offset = (uint64_t)slot * index_bytes;
        g->layer_n_index_comp[il] =
            g->spec_prefix_n_index_comp[slot][il];
        if (ds4_gpu_tensor_copy(g->layer_index_state_kv[il], 0,
                                g->spec_prefix_index_state_kv[il], index_offset,
                                index_bytes) == 0 ||
            ds4_gpu_tensor_copy(g->layer_index_state_score[il], 0,
                                g->spec_prefix_index_state_score[il],
                                index_offset, index_bytes) == 0) {
          return false;
        }
    }
    return true;
}

/* Batched LM head: one logits row per verification row.  This mirrors
 * rocm_graph_encode_output_head exactly; only the row count differs, and the
 * HC helpers derive their row count from the destination tensor size. */
static bool rocm_graph_encode_output_head_batch(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint32_t               n_tokens,
        uint64_t               vocab_dim) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;
    if (n_tokens > g->spec_rows_cap) return false;
    if (!weights->output_hc_fn) {
        fprintf(stderr, "ds4: spec output head requires output_hc weights\n");
        return false;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_gpu_tensor *output_pre =
        ds4_gpu_tensor_view(g->batch_hc_mix, 0,
                            (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    ds4_gpu_tensor *output_weights =
        ds4_gpu_tensor_view(g->batch_hc_split, 0,
                            (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    ds4_gpu_tensor *output_embd =
        ds4_gpu_tensor_view(g->batch_ffn_cur, 0,
                            (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *output_norm =
        ds4_gpu_tensor_view(g->batch_ffn_norm, 0,
                            (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));

    bool ok = output_pre && output_weights && output_embd && output_norm;
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                     g->batch_cur_hc,
                                                     (uint32_t)hc_dim,
                                                     n_tokens,
                                                     DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(output_pre,
                                             model->map,
                                             model->size,
                                             weights->output_hc_fn->abs_offset,
                                             hc_dim,
                                             DS4_N_HC,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(output_weights,
                                                   output_pre,
                                                   model->map,
                                                   model->size,
                                                   weights->output_hc_scale->abs_offset,
                                                   weights->output_hc_base->abs_offset,
                                                   DS4_N_HC,
                                                   DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(output_embd,
                                                  g->batch_cur_hc,
                                                  output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(output_norm,
                                                       output_embd,
                                                       model->map,
                                                       model->size,
                                                       weights->output_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->spec_logits,
                                              model->map,
                                              model->size,
                                              weights->output->abs_offset,
                                              DS4_N_EMBD,
                                              vocab_dim,
                                              output_norm,
                                              n_tokens) != 0;

    ds4_gpu_tensor_free(output_norm);
    ds4_gpu_tensor_free(output_embd);
    ds4_gpu_tensor_free(output_weights);
    ds4_gpu_tensor_free(output_pre);
    return ok;
}

/*
 * Advance independent sessions through one layer-synchronous narrow-row pass.
 *
 * Dense HC/Q/KV projections, attention output, FFN/MoE, and the LM head share
 * the coordinator's batch arena. The stateful attention middle remains
 * session-owned because each row has independent caches and may have a
 * different absolute position. C=1 never enters this function.
 */
static bool rocm_graph_eval_sessions_batch(
        const ds4_model       *model,
        const ds4_weights     *weights,
        const ds4_rocm_batch_item *items,
        size_t                 item_count) {
    if (!model || !weights || !items || item_count < 2u || item_count > 8u) {
        return false;
    }

    ds4_gpu_graph *coordinator = items[0].graph;
    if (!coordinator || item_count > coordinator->prefill_cap) return false;
    for (size_t row = 0; row < item_count; ++row) {
      if (!items[row].graph || !items[row].logits ||
          items[row].graph->raw_cap == 0) {
        return false;
      }
    }
    if (!rocm_graph_spec_prepare(
            coordinator, weights, static_cast<uint32_t>(item_count))) {
        return false;
    }
    for (size_t row = 0; row < item_count; ++row) {
      ds4_rocm_graph_dspark_capture_reset(items[row].graph);
    }

    int32_t tokens[8] = {};
    for (size_t row = 0; row < item_count; ++row) {
        tokens[row] = items[row].token;
    }
    if (ds4_gpu_tensor_write(coordinator->prefill_tokens,
                             0,
                             tokens,
                             item_count * sizeof(tokens[0])) == 0) {
        return false;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t hc_row_bytes = hc_dim * sizeof(float);
    ds4_gpu_set_small_batch_mode(1);
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = ds4_gpu_embed_tokens_hc_tensor(
                 coordinator->batch_cur_hc,
                 coordinator->prefill_tokens,
                 model->map,
                 model->size,
                 weights->token_embd->abs_offset,
                 (uint32_t)weights->token_embd->dim[1],
                 static_cast<uint32_t>(item_count),
                 DS4_N_EMBD,
                 DS4_N_HC,
                 weights->token_embd->type) != 0;
    }

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; ++il) {
        const ds4_layer_weights *layer = &weights->layer[il];
        ok = rocm_graph_encode_sessions_attention_batch(
                coordinator,
                model,
                layer,
                il,
                items,
                static_cast<uint32_t>(item_count));
        if (ok) {
            ok = rocm_graph_encode_layer_ffn_batch(
                    coordinator,
                    model,
                    layer,
                    il,
                    items[0].position,
                    static_cast<uint32_t>(item_count));
        }
        if (ok) {
            ds4_gpu_tensor *tmp = coordinator->batch_cur_hc;
            coordinator->batch_cur_hc = coordinator->batch_next_hc;
            coordinator->batch_next_hc = tmp;
        }
        for (size_t row = 0; ok && row < item_count; ++row) {
          ds4_gpu_tensor* row_hc = ds4_gpu_tensor_view(
              coordinator->batch_cur_hc, row * hc_row_bytes, hc_row_bytes);
          ok = row_hc != nullptr;
          if (ok) {
            ok = rocm_graph_dspark_capture_rows(items[row].graph, il, row_hc,
                                                1u);
          }
          ds4_gpu_tensor_free(row_hc);
        }
    }

    if (ok) {
        ok = rocm_graph_encode_output_head_batch(
                coordinator,
                model,
                weights,
                static_cast<uint32_t>(item_count),
                weights->output->dim[1]);
    }
    for (size_t row = 0; ok && row < item_count; ++row) {
        ok = ds4_gpu_tensor_copy(items[row].graph->cur_hc,
                                 0,
                                 coordinator->batch_cur_hc,
                                 row * hc_row_bytes,
                                 hc_row_bytes) != 0;
    }
    if (ok) {
        ok = ds4_gpu_end_commands() != 0;
    } else {
        (void)ds4_gpu_synchronize();
    }
    ds4_gpu_set_small_batch_mode(0);
    if (!ok) return false;

    const uint64_t logits_row_bytes =
        weights->output->dim[1] * sizeof(float);
    for (size_t row = 0; row < item_count; ++row) {
        if (ds4_gpu_tensor_read(coordinator->spec_logits,
                                row * logits_row_bytes,
                                items[row].logits,
                                logits_row_bytes) == 0) {
            return false;
        }
    }
    return true;
}

/*
 * Score `n_tokens` candidate tokens starting at `start` in one batched pass.
 *
 * On success `row_tops[i]` is the target's greedy continuation after candidate
 * row i, which is what the caller compares against candidate i+1.  The caller
 * owns the accept decision and must call rocm_graph_spec_rollback() when it
 * commits fewer rows than it verified.
 */
static bool rocm_graph_verify_suffix(ds4_gpu_graph* g, const ds4_model* model,
                                     const ds4_weights* weights,
                                     const token_vec* tokens, uint32_t start,
                                     uint32_t n_tokens, int32_t* row_tops,
                                     bool run_output_head = true) {
  if (!g || !model || !weights || !tokens || n_tokens == 0)
    return false;
  if (n_tokens > g->prefill_cap || n_tokens > g->spec_rows_cap)
    return false;
  if (!g->spec_logits || !g->spec_row_tops)
    return false;
  if (start > (uint32_t)tokens->len ||
      n_tokens > (uint32_t)tokens->len - start) {
    return false;
  }
  if (run_output_head && !row_tops)
    return false;

  if (!rocm_graph_upload_prompt_tokens(g->prefill_tokens, tokens, start,
                                       n_tokens)) {
    return false;
  }
  if (!rocm_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                              g->prefill_tokens, model, weights,
                                              tokens, start, n_tokens)) {
    return false;
  }
  {
    std::vector<int32_t> engram_toks(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i)
      engram_toks[i] = (int32_t)tokens->v[start + i];
    if (getenv("DS4_ENGRAM_DEBUG")) {
      fprintf(stderr, "ds4: prompt ids n=%u:", n_tokens);
      for (uint32_t i = 0; i < n_tokens && i < 32u; ++i)
        fprintf(stderr, " %d", engram_toks[i]);
      fprintf(stderr, "\n");
    }
    if (!rocm_graph_engram_prepare(g, model->map, engram_toks.data(), n_tokens, start))
      return false;
  }

  const uint64_t vocab_dim = weights->output->dim[1];
  /*
   * Verification-block kernel selection is scoped to this pass. Prefill and
   * decode keep the routes they were tuned and baselined with, so attaching a
   * DSpark drafter cannot move their numbers.
   */
  const bool saved_capture_prefixes = g->spec_capture_prefixes;
  g->spec_capture_prefixes =
      n_tokens > 1u && n_tokens <= DS4_SPEC_PREFIX_SLOTS + 2u;
  ds4_gpu_set_small_batch_mode(1);
  bool ok = ds4_gpu_begin_commands() != 0;
  for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
    ok = rocm_graph_encode_layer_batch(g, model, &weights->layer[il], il, start,
                                       n_tokens);
  }
  if (ok && run_output_head) {
    ok = rocm_graph_encode_output_head_batch(g, model, weights, n_tokens,
                                             vocab_dim);
  }
  if (ok && run_output_head) {
    ok = ds4_gpu_spec_row_argmax_tensor(g->spec_row_tops, g->spec_logits,
                                        (uint32_t)vocab_dim, n_tokens) != 0;
  }
  if (ok) {
    ok = ds4_gpu_end_commands() != 0;
  } else {
    (void)ds4_gpu_synchronize();
  }
  ds4_gpu_set_small_batch_mode(0);
  g->spec_capture_prefixes = saved_capture_prefixes;
  if (!ok)
    return false;
  if (!run_output_head)
    return true;

  return ds4_gpu_tensor_read(g->spec_row_tops, 0, row_tops,
                             (uint64_t)n_tokens * sizeof(row_tops[0])) != 0;
}

static bool rocm_graph_distribute_verify_results(
    ds4_gpu_graph* coordinator, const ds4_rocm_verify_item* items,
    size_t item_count, const std::array<uint32_t, 9>& row_offsets,
    const std::vector<int32_t>& flat_tops, uint64_t vocab_dim) {
  if (!coordinator || !items || item_count == 0u ||
      flat_tops.size() < row_offsets[item_count]) {
    return false;
  }

  bool gather_frontiers = coordinator->spec_frontier_logits != nullptr;
  for (size_t index = 0; index < item_count; ++index) {
    const ds4_rocm_verify_item& item = items[index];
    memcpy(item.row_tops, flat_tops.data() + row_offsets[index],
           (size_t)item.n_tokens * sizeof(item.row_tops[0]));
    gather_frontiers =
        gather_frontiers && item.frontier_logits != nullptr &&
        item.logical_n_tokens != 0u &&
        item.logical_n_tokens <= item.n_tokens;
  }

  const uint64_t logits_row_bytes = vocab_dim * sizeof(float);
  if (gather_frontiers) {
    // Sampled items decide acceptance on the CPU, so they take every verified
    // row instead of the argmax-selected frontier.
    for (size_t index = 0; index < item_count; ++index) {
      const ds4_rocm_verify_item& item = items[index];
      if (item.row_logits == nullptr) continue;
      if (ds4_gpu_tensor_read(coordinator->spec_logits,
                              (uint64_t)row_offsets[index] * logits_row_bytes,
                              item.row_logits,
                              (uint64_t)item.n_tokens * logits_row_bytes) == 0) {
        return false;
      }
    }
    bool ok = ds4_gpu_begin_commands() != 0;
    for (size_t index = 0; ok && index < item_count; ++index) {
      const ds4_rocm_verify_item& item = items[index];
      if (item.row_logits != nullptr) continue;
      const uint32_t accepted = ds4_dspark_greedy_accept(
          item.row_tops, item.tokens->v + item.start, item.logical_n_tokens,
          item.stop_token);
      const uint32_t source_row =
          row_offsets[index] + accepted - 1u;
      ok = ds4_gpu_tensor_copy(
               coordinator->spec_frontier_logits,
               (uint64_t)index * logits_row_bytes,
               coordinator->spec_logits,
               (uint64_t)source_row * logits_row_bytes,
               logits_row_bytes) != 0;
    }
    if (ok) {
      ok = ds4_gpu_end_commands() != 0;
    } else {
      (void)ds4_gpu_synchronize();
    }
    if (!ok) return false;

    std::vector<float> frontier_logits(item_count * vocab_dim);
    if (ds4_gpu_tensor_read(
            coordinator->spec_frontier_logits, 0, frontier_logits.data(),
            (uint64_t)item_count * logits_row_bytes) == 0) {
      return false;
    }
    for (size_t index = 0; index < item_count; ++index) {
      if (items[index].row_logits != nullptr) continue;
      memcpy(items[index].frontier_logits,
             frontier_logits.data() + index * vocab_dim,
             (size_t)logits_row_bytes);
    }
    return true;
  }

  for (size_t index = 0; index < item_count; ++index) {
    if (index != 0u &&
        ds4_gpu_tensor_copy(
            items[index].graph->spec_logits, 0, coordinator->spec_logits,
            (uint64_t)row_offsets[index] * logits_row_bytes,
            (uint64_t)items[index].n_tokens * logits_row_bytes) == 0) {
      return false;
    }
  }
  return true;
}

/*
 * Verify one independently sized draft block per session.
 *
 * Attention remains session-local because it mutates independent raw,
 * compressed, and indexer caches. The much larger FFN/MoE and output-head
 * weights are shared across every verification row, so flatten the blocks and
 * stream those weights once per layer.
 */
static bool rocm_graph_verify_sessions_batch(ds4_engine* engine,
                                             const ds4_rocm_verify_item* items,
                                             size_t item_count) {
  if (!engine || !engine->model || !engine->weights || !items ||
      item_count < 2u || item_count > 8u) {
    return false;
  }

  ds4_gpu_graph* coordinator = items[0].graph;
  if (!coordinator || !coordinator->batch_dspark_verify_cur_hc ||
      !coordinator->batch_dspark_verify_after_attn_hc) {
    return false;
  }
  std::array<uint32_t, 9> row_offsets{};
  bool saved_capture_prefixes[8] = {};
  for (size_t index = 0; index < item_count; ++index) {
    const ds4_rocm_verify_item& item = items[index];
    if (!item.graph || !item.tokens || !item.row_tops || item.n_tokens == 0u ||
        item.logical_n_tokens == 0u || item.logical_n_tokens > item.n_tokens ||
        item.start > static_cast<uint32_t>(item.tokens->len) ||
        item.n_tokens > static_cast<uint32_t>(item.tokens->len) - item.start) {
      return false;
    }
    row_offsets[index + 1u] = row_offsets[index] + item.n_tokens;
    saved_capture_prefixes[index] = item.graph->spec_capture_prefixes;
  }
  const uint32_t total_rows = row_offsets[item_count];
  if (total_rows > coordinator->prefill_cap ||
      total_rows > coordinator->batch_dspark_verify_rows_cap) {
    return false;
  }

  std::vector<int32_t> flat_tokens(total_rows);
  std::vector<int32_t> flat_tops(total_rows);

  size_t prepared_count = 0;
  for (; prepared_count < item_count; ++prepared_count) {
    const ds4_rocm_verify_item& item = items[prepared_count];
    const uint32_t rows_cap =
        prepared_count == 0u ? total_rows : item.n_tokens;
    if (!rocm_graph_spec_prepare(item.graph, engine->weights, rows_cap) ||
        !rocm_graph_spec_frontier_save(item.graph)) {
      break;
    }
    item.graph->spec_capture_prefixes =
        item.n_tokens > 1u &&
        item.n_tokens <= DS4_SPEC_PREFIX_SLOTS + 2u;
    ds4_rocm_graph_dspark_capture_reset(item.graph);
    for (uint32_t row = 0; row < item.n_tokens; ++row) {
      flat_tokens[row_offsets[prepared_count] + row] =
          item.tokens->v[item.start + row];
    }
  }
  if (prepared_count != item_count) {
    for (size_t index = 0; index < prepared_count; ++index) {
      items[index].graph->spec_capture_prefixes = saved_capture_prefixes[index];
    }
    return false;
  }

  const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
  const uint64_t hc_row_bytes = hc_dim * sizeof(float);

  bool attn_group_pairs_compatible = (item_count & 1u) == 0u;
  for (size_t index = 0;
       attn_group_pairs_compatible && index < item_count; index += 2u) {
    attn_group_pairs_compatible =
        items[index].n_tokens != 0u &&
        items[index].n_tokens <= 6u &&
        items[index + 1u].n_tokens != 0u &&
        items[index + 1u].n_tokens <= 6u;
  }
  const bool combine_attn_front = attn_group_pairs_compatible;
  const bool exact_attn_output_rows =
      (total_rows >= 4u && total_rows <= 8u) ||
      total_rows == 10u || total_rows == 12u || total_rows == 16u ||
      total_rows == 18u || total_rows == 24u ||
      total_rows == 32u;
  const bool combine_attn_output = combine_attn_front && exact_attn_output_rows;
  bool ok = true;
  if (ok) {
    ok = ds4_gpu_tensor_write(
             coordinator->prefill_tokens, 0, flat_tokens.data(),
             (uint64_t)total_rows * sizeof(flat_tokens[0])) != 0;
  }
  if (ok) {
    ok = ds4_gpu_embed_tokens_hc_tensor(
             coordinator->batch_dspark_verify_cur_hc,
             coordinator->prefill_tokens, engine->model->map,
             engine->model->size, engine->weights->token_embd->abs_offset,
             static_cast<uint32_t>(engine->weights->token_embd->dim[1]),
             total_rows, DS4_N_EMBD, DS4_N_HC,
             engine->weights->token_embd->type) != 0;
  }

  ds4_gpu_set_small_batch_mode(1);
  if (ok)
    ok = ds4_gpu_begin_commands() != 0;

  for (uint32_t il = 0; ok && il < DS4_N_LAYER; ++il) {
    const ds4_layer_weights* layer = &engine->weights->layer[il];
    const uint64_t mix_hc =
        2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim =
        (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const uint32_t comp_width =
        (ratio == 4u ? 2u : 1u) * DS4_N_HEAD_DIM;
    const uint64_t indexer_q_dim =
        (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
    const bool combine_attn_f16 = combine_attn_front && ratio != 0u;
    const bool combine_indexer_f16 =
        combine_attn_f16 && ratio == 4u;
    auto with_attention_slice =
        [&](ds4_gpu_graph* graph, uint32_t row0, uint32_t rows,
            auto&& operation) {
          // Use the coordinator's projected rows even when sessions borrow
          // different arenas. Views also confine compressor and indexer
          // scratch writes to each request's possibly ragged slice.
          std::array<ds4_gpu_tensor**, 17> slots{{
              &graph->batch_cur_hc,
              &graph->batch_flat_hc,
              &graph->batch_hc_mix,
              &graph->batch_hc_split,
              &graph->batch_attn_cur,
              &graph->batch_attn_norm,
              &graph->batch_qr,
              &graph->batch_qr_norm,
              &graph->batch_q,
              &graph->batch_kv_raw,
              &graph->batch_kv,
              &graph->batch_after_attn_hc,
              &graph->batch_indexer_q,
              &graph->batch_indexer_weights,
              &graph->batch_heads,
              &graph->batch_comp_kv,
              &graph->batch_comp_sc,
          }};
          std::array<ds4_gpu_tensor*, 17> bases{{
              coordinator->batch_cur_hc,
              coordinator->batch_flat_hc,
              coordinator->batch_hc_mix,
              coordinator->batch_hc_split,
              coordinator->batch_attn_cur,
              coordinator->batch_attn_norm,
              coordinator->batch_qr,
              coordinator->batch_qr_norm,
              coordinator->batch_q,
              coordinator->batch_kv_raw,
              coordinator->batch_kv,
              coordinator->batch_after_attn_hc,
              coordinator->batch_indexer_q,
              coordinator->batch_indexer_weights,
              coordinator->batch_heads,
              coordinator->batch_comp_kv,
              coordinator->batch_comp_sc,
          }};
          const std::array<uint64_t, 17> row_bytes{{
              hc_row_bytes,
              hc_row_bytes,
              mix_hc * sizeof(float),
              mix_hc * sizeof(float),
              (uint64_t)DS4_N_EMBD * sizeof(float),
              (uint64_t)DS4_N_EMBD * sizeof(float),
              q_rank * sizeof(float),
              q_rank * sizeof(float),
              q_dim * sizeof(float),
              (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
              (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
              hc_row_bytes,
              indexer_q_dim * sizeof(float),
              (uint64_t)DS4_N_INDEXER_HEAD * sizeof(float),
              q_dim * sizeof(float),
              (uint64_t)comp_width * sizeof(float),
              (uint64_t)comp_width * sizeof(float),
          }};
          std::array<ds4_gpu_tensor*, 17> saved{};
          std::array<ds4_gpu_tensor*, 17> views{};
          bool slice_ok = true;
          for (size_t field = 0; field < slots.size(); ++field) {
            saved[field] = *slots[field];
            views[field] = ds4_gpu_tensor_view(
                bases[field], (uint64_t)row0 * row_bytes[field],
                (uint64_t)rows * row_bytes[field]);
            slice_ok = slice_ok && views[field] != nullptr;
          }
          if (slice_ok) {
            for (size_t field = 0; field < slots.size(); ++field) {
              *slots[field] = views[field];
            }
            slice_ok = operation();
            for (size_t field = 0; field < slots.size(); ++field) {
              *slots[field] = saved[field];
            }
          }
          for (ds4_gpu_tensor* view : views) {
            ds4_gpu_tensor_free(view);
          }
          return slice_ok;
        };
    if (combine_attn_front) {
      for (size_t index = 0; ok && index < item_count; ++index) {
        ds4_gpu_graph* graph = items[index].graph;
        const uint32_t row0 = row_offsets[index];
        const uint32_t rows = items[index].n_tokens;
        ok = ds4_gpu_tensor_copy(
                 coordinator->batch_cur_hc,
                 (uint64_t)row0 * hc_row_bytes,
                 coordinator->batch_dspark_verify_cur_hc,
                 (uint64_t)row0 * hc_row_bytes,
                 (uint64_t)rows * hc_row_bytes) != 0;
        if (ok) {
          ok = with_attention_slice(graph, row0, rows, [&]() {
            return rocm_graph_encode_layer_attention_batch(
                graph, engine->model, layer, il, items[index].start, rows,
                false, true, !combine_attn_front, !combine_attn_front);
          });
        }
      }
      if (ok && combine_attn_front) {
        ok = ds4_gpu_matmul_q8_0_group_pairs_tensor(
                 coordinator->batch_qr, engine->model->map,
                 engine->model->size, layer->attn_q_a->abs_offset,
                 DS4_N_EMBD, q_rank, coordinator->batch_attn_norm,
                 total_rows, row_offsets.data(),
                 static_cast<uint32_t>(item_count)) != 0;
      }
      if (ok && combine_attn_front) {
        ok = ds4_gpu_matmul_q8_0_group_pairs_tensor(
                 coordinator->batch_kv_raw, engine->model->map,
                 engine->model->size, layer->attn_kv->abs_offset,
                 DS4_N_EMBD, DS4_N_HEAD_DIM,
                 coordinator->batch_attn_norm, total_rows,
                 row_offsets.data(), static_cast<uint32_t>(item_count)) != 0;
      }
      if (ok && combine_attn_front) {
        ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
                 coordinator->batch_qr_norm, coordinator->batch_qr,
                 engine->model->map, engine->model->size,
                 layer->attn_q_a_norm->abs_offset,
                 static_cast<uint32_t>(q_rank), coordinator->batch_kv,
                 coordinator->batch_kv_raw,
                 layer->attn_kv_a_norm->abs_offset, DS4_N_HEAD_DIM,
                 total_rows, DS4_RMS_EPS) != 0;
      }

      if (ok && combine_attn_front) {
        ok = ds4_gpu_matmul_q8_0_group_pairs_tensor(
                 coordinator->batch_q, engine->model->map,
                 engine->model->size, layer->attn_q_b->abs_offset,
                 q_rank, q_dim, coordinator->batch_qr_norm, total_rows,
                 row_offsets.data(), static_cast<uint32_t>(item_count)) != 0;
      }

      if (ok && combine_attn_f16) {
        ok = ds4_gpu_matmul_f16_group_pairs_tensor(
                 coordinator->batch_comp_kv, engine->model->map,
                 engine->model->size,
                 layer->attn_compressor_kv->abs_offset, DS4_N_EMBD,
                 comp_width, coordinator->batch_attn_norm, total_rows,
                 row_offsets.data(),
                 static_cast<uint32_t>(item_count)) != 0;
      }
      if (ok && combine_attn_f16) {
        ok = ds4_gpu_matmul_f16_group_pairs_tensor(
                 coordinator->batch_comp_sc, engine->model->map,
                 engine->model->size,
                 rocm_graph_opt_w_off(layer->attn_compressor_gate), DS4_N_EMBD,
                 comp_width, coordinator->batch_attn_norm, total_rows,
                 row_offsets.data(),
                 static_cast<uint32_t>(item_count)) != 0;
      }
      if (ok && combine_indexer_f16) {
        ok = ds4_gpu_matmul_f16_group_pairs_tensor(
                 coordinator->batch_indexer_q, engine->model->map,
                 engine->model->size,
                 layer->indexer_attn_q_b->abs_offset, q_rank,
                 indexer_q_dim, coordinator->batch_qr_norm, total_rows,
                 row_offsets.data(),
                 static_cast<uint32_t>(item_count)) != 0;
      }
      if (ok && combine_indexer_f16) {
        ok = ds4_gpu_matmul_f16_group_pairs_tensor(
                 coordinator->batch_indexer_weights, engine->model->map,
                 engine->model->size, layer->indexer_proj->abs_offset,
                 DS4_N_EMBD, DS4_N_INDEXER_HEAD,
                 coordinator->batch_attn_norm, total_rows,
                 row_offsets.data(),
                 static_cast<uint32_t>(item_count)) != 0;
      }
    }
    for (size_t index = 0; ok && index < item_count; ++index) {
      ds4_gpu_graph* graph = items[index].graph;
      const uint64_t block_offset =
          (uint64_t)row_offsets[index] * hc_row_bytes;
      const uint64_t block_bytes =
          (uint64_t)items[index].n_tokens * hc_row_bytes;
      if (!combine_attn_front) {
        ok = ds4_gpu_tensor_copy(graph->batch_cur_hc, 0,
                                 coordinator->batch_dspark_verify_cur_hc,
                                 block_offset, block_bytes) != 0;
      }
      if (ok) {
        if (combine_attn_front) {
          ok = with_attention_slice(
              graph, row_offsets[index], items[index].n_tokens, [&]() {
                return rocm_graph_encode_layer_attention_batch(
                    graph, engine->model, layer, il, items[index].start,
                    items[index].n_tokens, true, false, true, true,
                    combine_attn_f16, combine_indexer_f16,
                    !combine_attn_output);
              });
        } else {
          ok = rocm_graph_encode_layer_attention_batch(
              graph, engine->model, layer, il, items[index].start,
              items[index].n_tokens);
        }
      }
      if (ok && !combine_attn_output) {
        ok = ds4_gpu_tensor_copy(
                 coordinator->batch_dspark_verify_after_attn_hc, block_offset,
                 combine_attn_front ? coordinator->batch_after_attn_hc
                                    : graph->batch_after_attn_hc,
                 combine_attn_front ? block_offset : 0, block_bytes) != 0;
      }
    }
    if (ok && combine_attn_output) {
      ok = ds4_gpu_attention_output_q8_exact_batch_tensor(
               coordinator->batch_attn_out, coordinator->batch_attn_low,
               engine->model->map, engine->model->size,
               layer->attn_output_a->abs_offset,
               layer->attn_output_b->abs_offset,
               DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP),
               DS4_N_LORA_O, DS4_N_OUT_GROUP, DS4_N_EMBD,
               coordinator->batch_heads, total_rows) != 0;
    }
    if (ok && combine_attn_output) {
      ds4_gpu_tensor* attn_out = ds4_gpu_tensor_view(
          coordinator->batch_attn_out, 0,
          (uint64_t)total_rows * DS4_N_EMBD * sizeof(float));
      ds4_gpu_tensor* cur_hc = ds4_gpu_tensor_view(
          coordinator->batch_cur_hc, 0,
          (uint64_t)total_rows * hc_row_bytes);
      ds4_gpu_tensor* hc_split = ds4_gpu_tensor_view(
          coordinator->batch_hc_split, 0,
          (uint64_t)total_rows * mix_hc * sizeof(float));
      ds4_gpu_tensor* after_attn_hc = ds4_gpu_tensor_view(
          coordinator->batch_dspark_verify_after_attn_hc, 0,
          (uint64_t)total_rows * hc_row_bytes);
      ok = attn_out && cur_hc && hc_split && after_attn_hc &&
           ds4_gpu_hc_expand_split_tensor(
               after_attn_hc, attn_out, cur_hc, hc_split, DS4_N_EMBD,
               DS4_N_HC) != 0;
      ds4_gpu_tensor_free(after_attn_hc);
      ds4_gpu_tensor_free(hc_split);
      ds4_gpu_tensor_free(cur_hc);
      ds4_gpu_tensor_free(attn_out);
    }
    if (ok) {
      ok = ds4_gpu_tensor_copy(coordinator->batch_after_attn_hc, 0,
                               coordinator->batch_dspark_verify_after_attn_hc,
                               0, (uint64_t)total_rows * hc_row_bytes) != 0;
    }
    if (ok) {
      ok = rocm_graph_encode_layer_ffn_batch(
          coordinator, engine->model, layer, il, items[0].start, total_rows,
          coordinator->prefill_tokens, row_offsets.data(),
          static_cast<uint32_t>(item_count));
    }
    for (size_t index = 0; ok && index < item_count; ++index) {
      const uint64_t block_offset =
          (uint64_t)row_offsets[index] * hc_row_bytes;
      ds4_gpu_tensor* block_hc =
          ds4_gpu_tensor_view(coordinator->batch_next_hc, block_offset,
                              (uint64_t)items[index].n_tokens * hc_row_bytes);
      ok = block_hc != NULL;
      if (ok) {
        ok = rocm_graph_dspark_capture_rows(items[index].graph, il, block_hc,
                                            items[index].n_tokens);
      }
      ds4_gpu_tensor_free(block_hc);
    }
    if (ok) {
      ok = ds4_gpu_tensor_copy(coordinator->batch_dspark_verify_cur_hc, 0,
                               coordinator->batch_next_hc, 0,
                               (uint64_t)total_rows * hc_row_bytes) != 0;
    }
  }
  if (ok) {
    ok = ds4_gpu_tensor_copy(coordinator->batch_cur_hc, 0,
                             coordinator->batch_dspark_verify_cur_hc, 0,
                             (uint64_t)total_rows * hc_row_bytes) != 0;
  }
  if (ok) {
    ok = rocm_graph_encode_output_head_batch(coordinator, engine->model,
                                             engine->weights, total_rows,
                                             engine->weights->output->dim[1]);
  }
  if (ok) {
    ok = ds4_gpu_spec_row_argmax_tensor(
             coordinator->spec_row_tops, coordinator->spec_logits,
             static_cast<uint32_t>(engine->weights->output->dim[1]),
             total_rows) != 0;
  }

  if (ok) {
    ok = ds4_gpu_end_commands() != 0;
  } else {
    (void)ds4_gpu_synchronize();
  }
  ds4_gpu_set_small_batch_mode(0);
  for (size_t index = 0; index < item_count; ++index) {
    items[index].graph->spec_capture_prefixes = saved_capture_prefixes[index];
  }
  if (!ok)
    return false;

  if (ds4_gpu_tensor_read(coordinator->spec_row_tops, 0, flat_tops.data(),
                          (uint64_t)total_rows * sizeof(flat_tops[0])) == 0) {
    return false;
  }
  {
    return rocm_graph_distribute_verify_results(
        coordinator, items, item_count, row_offsets, flat_tops,
        engine->weights->output->dim[1]);
  }
}

static bool rocm_graph_read_spec_logits_row(const ds4_gpu_graph *g,
                                            uint32_t row,
                                            float *logits) {
    if (!g || !g->spec_logits || !logits || row >= g->spec_rows_cap) return false;
    const uint64_t row_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    return ds4_gpu_tensor_read(g->spec_logits,
                                (uint64_t)row * row_bytes,
                                logits,
                                row_bytes) != 0;
}

/* =========================================================================
 * DSpark Drafting.
 * =========================================================================
 *
 * One draft pass proposes a whole block. The target's hidden states at the
 * sampled layers are projected into each stage's key/value space and appended to
 * that stage's ring, so the three-block drafter tracks the forty-three-block
 * target without re-running it. The block itself enters as
 * [last_committed_token, MASK, ...] and every block row attends to the injected
 * ring *and* to all other block rows, which is what lets one forward pass
 * produce `block_size` candidates instead of `block_size` sequential steps.
 */


static uint32_t rocm_graph_dspark_feature_row_stride(const ds4_gpu_graph *g) {
    return g->dspark ? g->dspark->n_target_layers * DS4_N_EMBD : 0u;
}

/* Feature slot for a target layer, or -1 when this layer is not sampled. */
static int rocm_graph_dspark_slot(const ds4_gpu_graph *g, uint32_t il) {
    if (!g->dspark) return -1;
    for (uint32_t i = 0; i < g->dspark->n_target_layers; i++) {
        if (g->dspark->target_layer_ids[i] == il) return (int)i;
    }
    return -1;
}

static uint32_t rocm_graph_dspark_complete_mask(const ds4_gpu_graph *g) {
    if (!g->dspark || g->dspark->n_target_layers >= 32u) return 0u;
    return (1u << g->dspark->n_target_layers) - 1u;
}

static bool rocm_graph_dspark_capture_rows(ds4_gpu_graph *g,
                                           uint32_t il,
                                           const ds4_gpu_tensor *hc,
                                           uint32_t n_rows) {
    int slot = rocm_graph_dspark_slot(g, il);
    if (slot < 0) return true;
    if (!g->dspark_features || n_rows == 0 ||
        n_rows > g->dspark_capture_rows_cap) {
      return false;
    }
    const uint32_t stride = rocm_graph_dspark_feature_row_stride(g);
    if (!ds4_gpu_dspark_capture_features_tensor(g->dspark_features,
                                                hc,
                                                stride,
                                                (uint32_t)slot * DS4_N_EMBD,
                                                DS4_N_EMBD,
                                                DS4_N_HC,
                                                n_rows)) {
        return false;
    }
    g->dspark_capture_mask |= 1u << (uint32_t)slot;
    return true;
}

/* Called from the single-token decode path once layer `il` has updated cur_hc. */
static bool rocm_graph_dspark_capture_decode_layer(ds4_gpu_graph *g, uint32_t il) {
    if (!g->dspark || !g->dspark_capture_enabled) return true;
    return rocm_graph_dspark_capture_rows(g, il, g->cur_hc, 1u);
}

/* Called from the batched path (prefill and verification) per layer. */
static bool rocm_graph_dspark_capture_batch_layer(ds4_gpu_graph *g,
                                                  uint32_t il,
                                                  uint32_t start,
                                                  uint32_t n_tokens) {
    if (!g->dspark || !g->dspark_capture_enabled) return true;
    const int slot = rocm_graph_dspark_slot(g, il);
    if (slot < 0) return true;
    /* A different (start, count) means a new batch has begun - a verification
     * block after a prompt chunk, say - so start a fresh capture window rather
     * than treating it as an inconsistency. */
    if (g->dspark_capture_batch_mask == 0u ||
        g->dspark_capture_batch_start != start ||
        g->dspark_capture_batch_tokens != n_tokens) {
      g->dspark_capture_batch_mask = 0u;
      g->dspark_capture_mask = 0u;
      g->dspark_capture_batch_start = start;
      g->dspark_capture_batch_tokens = n_tokens;
    }
    if (!rocm_graph_dspark_capture_rows(g, il, g->batch_cur_hc, n_tokens)) return false;
    g->dspark_capture_batch_mask |= 1u << (uint32_t)slot;
    return true;
}

static bool rocm_graph_dspark_capture_complete(const ds4_gpu_graph *g) {
    if (!g || !g->dspark_capture_enabled) return false;
    const uint32_t want = rocm_graph_dspark_complete_mask(g);
    return want != 0u && g->dspark_capture_mask == want;
}

/*
 * Project captured target features into every stage's ring at absolute positions
 * [pos0, pos0 + n_rows). Row r of dspark_features must hold position pos0 + r.
 */
static bool rocm_graph_dspark_inject(ds4_gpu_graph *g,
                                     uint32_t pos0,
                                     uint32_t n_rows) {
    const ds4_dspark_model *d = g->dspark;
    if (!d || n_rows == 0 || n_rows > g->dspark_capture_rows_cap) return false;
    if (!g->dspark_features || !g->dspark_fused) return false;
    const ds4_model *sm = d->model;
    const ds4_dspark_stage_weights *first = &d->stage[0];
    const uint32_t feature_width = rocm_graph_dspark_feature_row_stride(g);

    ds4_gpu_tensor *fused = ds4_gpu_tensor_view(
            g->dspark_fused, 0, (uint64_t)n_rows * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *kv_raw = ds4_gpu_tensor_view(
            g->batch_kv_raw, 0, (uint64_t)n_rows * DS4_N_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *kv = ds4_gpu_tensor_view(
            g->batch_kv, 0, (uint64_t)n_rows * DS4_N_HEAD_DIM * sizeof(float));
    bool ok = fused && kv_raw && kv;

    /* Fuse the sampled layers once; every stage reads the same fused feature. */
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(fused,
                                             sm->map,
                                             sm->size,
                                             first->main_proj->abs_offset,
                                             feature_width,
                                             DS4_N_EMBD,
                                             g->dspark_features,
                                             n_rows) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->dspark_fused,
                                                      fused,
                                                      sm->map,
                                                      sm->size,
                                                      first->main_norm->abs_offset,
                                                      DS4_N_EMBD,
                                                      n_rows,
                                                      DS4_RMS_EPS) != 0;

    for (uint32_t stage = 0; ok && stage < d->n_stages; stage++) {
        const ds4_layer_weights *layer = &d->stage[stage].block;
        ok = ds4_gpu_matmul_q8_0_tensor(kv_raw,
                                         sm->map,
                                         sm->size,
                                         layer->attn_kv->abs_offset,
                                         DS4_N_EMBD,
                                         DS4_N_HEAD_DIM,
                                         fused,
                                         n_rows) != 0;
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(kv,
                                                          kv_raw,
                                                          sm->map,
                                                          sm->size,
                                                          layer->attn_kv_a_norm->abs_offset,
                                                          DS4_N_HEAD_DIM,
                                                          n_rows,
                                                          DS4_RMS_EPS) != 0;
        if (ok) ok = ds4_gpu_rope_tail_tensor(kv,
                                               n_rows,
                                               DS4_N_HEAD_KV,
                                               DS4_N_HEAD_DIM,
                                               DS4_N_ROT,
                                               pos0,
                                               0,
                                               false,
                                               DS4_ROPE_FREQ_BASE,
                                               1.0f,
                                               0.0f,
                                               1.0f,
                                               DS4_ROPE_YARN_BETA_FAST,
                                               DS4_ROPE_YARN_BETA_SLOW) != 0;
        if (ok) ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv,
                                                          n_rows,
                                                          DS4_N_HEAD_DIM,
                                                          DS4_N_ROT) != 0;
        if (ok) ok = ds4_gpu_store_raw_kv_batch_tensor(g->dspark_kv_cache[stage],
                                                        kv,
                                                        g->dspark_cache_cap,
                                                        pos0,
                                                        n_rows,
                                                        DS4_N_HEAD_DIM) != 0;
    }

    ds4_gpu_tensor_free(kv);
    ds4_gpu_tensor_free(kv_raw);
    ds4_gpu_tensor_free(fused);
    /*
     * Context coverage is contiguous from position zero. A later injection
     * cannot fill a gap left while support capture was disabled.
     */
    if (ok && pos0 <= g->dspark_context_len &&
        pos0 + n_rows > g->dspark_context_len) {
      g->dspark_context_len = pos0 + n_rows;
    }
    return ok;
}

/*
 * One DSpark stage over the draft block: the DS4 block with the causal window
 * replaced by non-causal attention over the injected ring plus the block
 * itself. The FFN half is shared with the target's batched path, which is safe
 * because that half is driven entirely by the supplied weights.
 */
static bool rocm_graph_encode_dspark_stage_attention(
    ds4_gpu_graph* g, uint32_t stage, uint32_t pos0, uint32_t n_rows,
    bool front_ready = false, bool front_only = false, bool project_qb = true,
    bool project_q_a_kv = true) {
  const ds4_dspark_model* d = g->dspark;
  if (!d || stage >= d->n_stages || n_rows == 0u)
    return false;
  const uint32_t draft_rows = n_rows;
  const auto span =
      ds4_dspark_visible_span(pos0, draft_rows, g->dspark_cache_cap);
  if (span.count == 0u)
    return false;
  const ds4_model* sm = d->model;
  const ds4_layer_weights* layer = &d->stage[stage].block;
  const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
  const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
  const uint64_t q_rank = layer->attn_q_a->dim[1];
  const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
  const uint32_t group_heads = DS4_N_HEAD / DS4_N_OUT_GROUP;
  const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;

  ds4_gpu_tensor* hc_mix_view = ds4_gpu_tensor_view(
      g->batch_hc_mix, 0, (uint64_t)n_rows * mix_hc * sizeof(float));
  ds4_gpu_tensor* hc_split_view = ds4_gpu_tensor_view(
      g->batch_hc_split, 0, (uint64_t)n_rows * mix_hc * sizeof(float));
  ds4_gpu_tensor* attn_cur_view = ds4_gpu_tensor_view(
      g->batch_attn_cur, 0, (uint64_t)n_rows * DS4_N_EMBD * sizeof(float));
  ds4_gpu_tensor* after_attn_hc_view = ds4_gpu_tensor_view(
      g->batch_after_attn_hc, 0, (uint64_t)draft_rows * hc_dim * sizeof(float));
  bool ok = hc_mix_view && hc_split_view && attn_cur_view && after_attn_hc_view;

  if (ok && !front_ready) {
    ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc, g->batch_cur_hc,
                                            (uint32_t)hc_dim, n_rows,
                                            DS4_RMS_EPS) != 0;
  }
  if (ok && !front_ready) {
    ok = ds4_gpu_matmul_f16_tensor(hc_mix_view, sm->map, sm->size,
                                   layer->hc_attn_fn->abs_offset, hc_dim,
                                   mix_hc, g->batch_flat_hc, n_rows) != 0;
  }
  if (ok && !front_ready) {
    ok = ds4_gpu_hc_split_weighted_sum_tensor(
             attn_cur_view, hc_split_view, hc_mix_view, g->batch_cur_hc,
             sm->map, sm->size, layer->hc_attn_scale->abs_offset,
             layer->hc_attn_base->abs_offset, DS4_N_EMBD, DS4_N_HC,
             DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS, NULL, 0u) != 0;
  }
  if (ok && !front_ready) {
    ok =
        ds4_gpu_rms_norm_weight_rows_tensor(
            g->batch_attn_norm, g->batch_attn_cur, sm->map, sm->size,
            layer->attn_norm->abs_offset, DS4_N_EMBD, n_rows, DS4_RMS_EPS) != 0;
  }

  if (ok && project_q_a_kv) {
    ok = ds4_gpu_matmul_q8_0_tensor(
             g->batch_qr, sm->map, sm->size, layer->attn_q_a->abs_offset,
             DS4_N_EMBD, q_rank, g->batch_attn_norm, draft_rows) != 0;
  }
  if (ok && project_q_a_kv) {
    ok = ds4_gpu_rms_norm_weight_rows_tensor(
             g->batch_qr_norm, g->batch_qr, sm->map, sm->size,
             layer->attn_q_a_norm->abs_offset, (uint32_t)q_rank, draft_rows,
             DS4_RMS_EPS) != 0;
  }
  if (ok && project_qb) {
    ok = ds4_gpu_matmul_q8_0_tensor(g->batch_q, sm->map, sm->size,
                                    layer->attn_q_b->abs_offset, q_rank, q_dim,
                                    g->batch_qr_norm, draft_rows) != 0;
  }
  if (front_only) {
    ds4_gpu_tensor_free(after_attn_hc_view);
    ds4_gpu_tensor_free(attn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
    return ok;
  }
  if (ok) {
    ok = ds4_gpu_head_rms_norm_tensor(g->batch_q, draft_rows, DS4_N_HEAD,
                                      DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
  }
  if (ok) {
    ok = ds4_gpu_rope_tail_tensor(
             g->batch_q, draft_rows, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT,
             pos0, 0, false, DS4_ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f,
             DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
  }

  if (ok && project_q_a_kv) {
    ok = ds4_gpu_matmul_q8_0_tensor(
             g->batch_kv_raw, sm->map, sm->size, layer->attn_kv->abs_offset,
             DS4_N_EMBD, DS4_N_HEAD_DIM, g->batch_attn_norm, n_rows) != 0;
  }
  if (ok && project_q_a_kv) {
    ok = ds4_gpu_rms_norm_weight_rows_tensor(
             g->batch_kv, g->batch_kv_raw, sm->map, sm->size,
             layer->attn_kv_a_norm->abs_offset, DS4_N_HEAD_DIM, n_rows,
             DS4_RMS_EPS) != 0;
  }
  if (ok) {
    ok = ds4_gpu_rope_tail_tensor(
             g->batch_kv, draft_rows, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT,
             pos0, 0, false, DS4_ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f,
             DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
  }
  if (ok) {
    ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(g->batch_kv, n_rows,
                                             DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
  }
  if (ok) {
    ok = ds4_gpu_store_raw_kv_batch_tensor(g->dspark_kv_cache[stage],
                                           g->batch_kv, g->dspark_cache_cap,
                                           pos0, n_rows, DS4_N_HEAD_DIM) != 0;
  }

  if (ok) {
    ok = ds4_gpu_attention_noncausal_raw_batch_heads_tensor(
             g->batch_heads, sm->map, sm->size, layer->attn_sinks->abs_offset,
             g->batch_q, g->dspark_kv_cache[stage], draft_rows, span.count,
             g->dspark_cache_cap, span.start, DS4_N_HEAD, DS4_N_HEAD_DIM) != 0;
  }
  if (ok) {
    ok = ds4_gpu_rope_tail_tensor(
             g->batch_heads, draft_rows, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT,
             pos0, 0, true, DS4_ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f,
             DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
  }
  if (ok) {
    ok = ds4_gpu_attention_output_q8_batch_tensor(
             g->batch_attn_out, g->batch_attn_low, g->batch_group_tmp,
             g->batch_low_tmp, sm->map, sm->size,
             layer->attn_output_a->abs_offset, layer->attn_output_b->abs_offset,
             group_dim, DS4_N_LORA_O, DS4_N_OUT_GROUP, DS4_N_EMBD,
             g->batch_heads, draft_rows) != 0;
  }
  if (ok) {
    ok = ds4_gpu_hc_expand_split_tensor(after_attn_hc_view, g->batch_attn_out,
                                        g->batch_cur_hc, hc_split_view,
                                        DS4_N_EMBD, DS4_N_HC) != 0;
  }

  ds4_gpu_tensor_free(after_attn_hc_view);
  ds4_gpu_tensor_free(attn_cur_view);
  ds4_gpu_tensor_free(hc_split_view);
  ds4_gpu_tensor_free(hc_mix_view);
  return ok;
}

static bool rocm_graph_encode_dspark_stage(ds4_gpu_graph* g, uint32_t stage,
                                           uint32_t pos0, uint32_t n_rows) {
  if (!rocm_graph_encode_dspark_stage_attention(g, stage, pos0, n_rows)) {
    return false;
  }
  const ds4_dspark_model* d = g->dspark;
  const uint32_t draft_rows = n_rows;
  const ds4_model* sm = d->model;
  const ds4_layer_weights* layer = &d->stage[stage].block;
  const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
  if (!rocm_graph_encode_layer_ffn_batch(g, sm, layer, stage, pos0,
                                         draft_rows)) {
    return false;
  }

  return ds4_gpu_tensor_copy(g->batch_cur_hc, 0, g->batch_next_hc, 0,
                             (uint64_t)draft_rows * hc_dim * sizeof(float)) !=
         0;
}

/* Final support hidden rows, consumed by the tied target LM head. */
static bool rocm_graph_encode_dspark_final_hidden(ds4_gpu_graph* g,
                                                  uint32_t n_rows,
                                                  ds4_gpu_tensor* hidden_out,
                                                  uint32_t hidden_row_offset) {
  const ds4_dspark_model* d = g->dspark;
  if (!d || !hidden_out || n_rows == 0 || n_rows > g->spec_rows_cap) {
    return false;
  }
  const ds4_model* sm = d->model;
  const ds4_dspark_stage_weights* last = &d->stage[d->n_stages - 1u];
  const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;

  /* Only the draft block feeds the final head. */
  ds4_gpu_tensor* rows_hc = ds4_gpu_tensor_view(
      g->batch_cur_hc, 0, (uint64_t)n_rows * hc_dim * sizeof(float));
  if (!rows_hc)
    return false;

  ds4_gpu_tensor* head_pre = ds4_gpu_tensor_view(
      g->batch_hc_mix, 0, (uint64_t)n_rows * DS4_N_HC * sizeof(float));
  ds4_gpu_tensor* head_weights = ds4_gpu_tensor_view(
      g->batch_hc_split, 0, (uint64_t)n_rows * DS4_N_HC * sizeof(float));
  ds4_gpu_tensor* head_embd = ds4_gpu_tensor_view(
      g->batch_ffn_cur, 0, (uint64_t)n_rows * DS4_N_EMBD * sizeof(float));
  ds4_gpu_tensor* head_norm = ds4_gpu_tensor_view(
      hidden_out, (uint64_t)hidden_row_offset * DS4_N_EMBD * sizeof(float),
      (uint64_t)n_rows * DS4_N_EMBD * sizeof(float));
  bool ok = head_pre && head_weights && head_embd && head_norm;

  if (ok)
    ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc, rows_hc,
                                            (uint32_t)hc_dim, n_rows,
                                            DS4_RMS_EPS) != 0;
  if (ok)
    ok = ds4_gpu_matmul_f16_tensor(head_pre, sm->map, sm->size,
                                   last->hc_head_fn->abs_offset, hc_dim,
                                   DS4_N_HC, g->batch_flat_hc, n_rows) != 0;
  if (ok)
    ok = ds4_gpu_output_hc_weights_tensor(
             head_weights, head_pre, sm->map, sm->size,
             last->hc_head_scale->abs_offset, last->hc_head_base->abs_offset,
             DS4_N_HC, DS4_HC_EPS) != 0;
  if (ok)
    ok = ds4_gpu_hc_weighted_sum_tensor(head_embd, rows_hc, head_weights,
                                        DS4_N_EMBD, DS4_N_HC) != 0;
  if (ok)
    ok = ds4_gpu_rms_norm_weight_rows_tensor(
             head_norm, head_embd, sm->map, sm->size, last->norm->abs_offset,
             DS4_N_EMBD, n_rows, DS4_RMS_EPS) != 0;

  ds4_gpu_tensor_free(head_norm);
  ds4_gpu_tensor_free(head_embd);
  ds4_gpu_tensor_free(head_weights);
  ds4_gpu_tensor_free(head_pre);
  ds4_gpu_tensor_free(rows_hc);
  return ok;
}

static bool rocm_graph_encode_dspark_logits(
        ds4_gpu_graph *g,
        const ds4_model *target_model,
        const ds4_weights *target_weights,
        const ds4_gpu_tensor *hidden_rows,
        uint32_t n_rows) {
  if (!g || !target_model || !target_weights || !hidden_rows || n_rows == 0u ||
      n_rows > g->spec_rows_cap) {
    return false;
  }
    const uint64_t vocab_dim = target_weights->output->dim[1];
    return ds4_gpu_matmul_q8_0_tensor(g->spec_logits,
                                      target_model->map,
                                      target_model->size,
                                      target_weights->output->abs_offset,
                                      DS4_N_EMBD,
                                      vocab_dim,
                                      hidden_rows,
                                      n_rows) != 0;
}

/*
 * Trace the block's trajectory with the Markov path selector.
 *
 * Selecting each position by its own argmax lets the block drift: the best
 * token at position t given the block's shared context is not the best token
 * given what position t-1 actually emitted. The rank-256 bilinear term restores
 * that coupling. Positions resolve in order because each conditions the next.
 * Greedy reads back the chosen index; sampled cohorts read a compact shortlist
 * and confidence for each request.
 */
static bool rocm_graph_dspark_select_sampled(
    ds4_gpu_graph* g, const ds4_rocm_dspark_draft_item* items, size_t count,
    uint32_t rows_per_item) {
  constexpr uint32_t blocks = (DS4_N_VOCAB + 255u) / 256u;
  if (!g->dspark_candidates)
    g->dspark_candidates =
        ds4_gpu_tensor_alloc(8u * sizeof(ds4_dspark_candidates));
  if (!g->dspark_candidate_scratch)
    g->dspark_candidate_scratch = ds4_gpu_tensor_alloc(
        8u * blocks * DS4_DSPARK_CANDIDATES * sizeof(uint64_t));
  if (!g->dspark_candidates || !g->dspark_candidate_scratch)
    return false;
  const auto* d = g->dspark;
  const auto& last = d->stage[d->n_stages - 1];
  std::array<int32_t, 8> previous{};
  std::array<ds4_dspark_candidates, 8> candidates{};
  std::array<ds4_dspark_confidence_policy, 8> policies{};
  std::array<bool, 8> stopped{};
  uint32_t max_rows = 0;
  for (size_t i = 0; i < count; ++i) {
    previous[i] = items[i].target_next_token;
    if (previous[i] < 0 || (uint32_t)previous[i] >= DS4_N_VOCAB)
      return false;
    policies[i] = {
        .depth = items[i].position,
        .concurrency = items[i].concurrency,
        .maximum = std::min(items[i].max_draft_tokens, rows_per_item)};
    *items[i].n_tokens = 0;
    max_rows =
        std::max(max_rows, std::min(items[i].max_draft_tokens, rows_per_item));
  }
  for (uint32_t row = 0; row < max_rows; ++row) {
    if (!ds4_gpu_tensor_write(g->prefill_tokens, 0, previous.data(),
                              count * sizeof(previous[0])))
      return false;
    const uint64_t span_rows = (count - 1) * rows_per_item + 1;
    auto* logits = ds4_gpu_tensor_view(
        g->spec_logits, (uint64_t)row * DS4_N_VOCAB * sizeof(float),
        span_rows * DS4_N_VOCAB * sizeof(float));
    // DS4 confidence uses hc_head(x), before norm; this buffer survives the
    // tied LM-head projection and holds independent rows for every request.
    auto* hidden = ds4_gpu_tensor_view(
        g->batch_ffn_cur, (uint64_t)row * DS4_N_EMBD * sizeof(float),
        span_rows * DS4_N_EMBD * sizeof(float));
    const bool ok =
        logits && hidden &&
        ds4_gpu_dspark_candidates_tensor(
            g->dspark_candidates, g->dspark_candidate_scratch, logits,
            (uint64_t)rows_per_item * DS4_N_VOCAB, hidden,
            (uint64_t)rows_per_item * DS4_N_EMBD, g->prefill_tokens,
            d->model->map, d->model->size, last.markov_w1->abs_offset,
            last.markov_w2->abs_offset, last.confidence->abs_offset,
            DS4_N_VOCAB, d->markov_rank, DS4_N_EMBD, count);
    ds4_gpu_tensor_free(hidden);
    ds4_gpu_tensor_free(logits);
    if (!ok || !ds4_gpu_tensor_read(g->dspark_candidates, 0, candidates.data(),
                                    count * sizeof(candidates[0])))
      return false;
    for (size_t i = 0; i < count; ++i) {
      if (stopped[i] || row >= items[i].max_draft_tokens)
        continue;
      const auto* sampler = items[i].sampler;
      if (sampler && sampler->propose &&
          !policies[i].Include(candidates[i].confidence)) {
        stopped[i] = true;
        continue;
      }
      const int token =
          sampler && sampler->propose
              ? sampler->propose(sampler->ctx, row, &candidates[i])
              : candidates[i].ids[0];
      if (token < 0 || (uint32_t)token >= DS4_N_VOCAB)
        return false;
      items[i].tokens[row] = token;
      previous[i] = token;
      *items[i].n_tokens = row + 1;
    }
    bool more = false;
    for (size_t i = 0; i < count; ++i)
      more |= !stopped[i] && row + 1 < items[i].max_draft_tokens;
    if (!more)
      break;
  }
  return true;
}

static bool rocm_graph_dspark_select_block(ds4_gpu_graph* g,
                                           const ds4_gpu_tensor* logits_rows,
                                           int target_next_token,
                                           uint32_t n_rows,
                                           int32_t* tokens_out) {
  const ds4_dspark_model* d = g->dspark;
  if (!d || !logits_rows || n_rows == 0 || !tokens_out) {
    return false;
  }
  if (!g->dspark_markov_index || !g->dspark_markov_key)
    return false;
  const ds4_model* sm = d->model;
  const ds4_dspark_stage_weights* last = &d->stage[d->n_stages - 1u];
  if (target_next_token < 0 || (uint32_t)target_next_token >= DS4_N_VOCAB ||
      !ds4_gpu_dspark_markov_chain_tensor(
          g->dspark_markov_index, g->dspark_markov_key, logits_rows, sm->map,
          sm->size, last->markov_w1->abs_offset, last->markov_w2->abs_offset,
          DS4_N_VOCAB, d->markov_rank, (uint32_t)target_next_token, n_rows) ||
      !ds4_gpu_tensor_read(g->dspark_markov_index, 0, tokens_out,
                           (uint64_t)n_rows * sizeof(tokens_out[0])))
    return false;
  return std::all_of(tokens_out, tokens_out + n_rows, [](int32_t token) {
    return token >= 0 && (uint32_t)token < DS4_N_VOCAB;
  });
}

static bool rocm_graph_dspark_select_block_batch(
    ds4_gpu_graph* coordinator, const ds4_rocm_dspark_draft_item* items,
    size_t item_count, const ds4_gpu_tensor* logits_rows,
    uint32_t rows_per_item) {
  if (!coordinator || !coordinator->dspark || !items || !logits_rows ||
      rows_per_item == 0u ||
      (item_count != 2u && item_count != 4u && item_count != 6u &&
       item_count != 8u) ||
      !coordinator->dspark_markov_index || !coordinator->dspark_markov_key) {
    return false;
  }
  const ds4_dspark_model* d = coordinator->dspark;
  const ds4_model* sm = d->model;
  const ds4_dspark_stage_weights* last =
      &d->stage[d->n_stages - 1u];
  if (!last->markov_w1 || !last->markov_w2) {
    return false;
  }

  std::array<int32_t, 8> previous{};
  std::array<int32_t, 8> selected{};
  std::array<bool, 8> active{};
  std::array<uint32_t, 8> row_limits{};
  std::array<uint32_t, 8> selected_rows{};
  for (size_t index = 0; index < item_count; ++index) {
    if (!items[index].tokens || !items[index].n_tokens ||
        items[index].max_draft_tokens == 0u ||
        items[index].target_next_token < 0 ||
        (uint32_t)items[index].target_next_token >= DS4_N_VOCAB) {
      return false;
    }
    previous[index] = items[index].target_next_token;
    row_limits[index] =
        std::min(items[index].max_draft_tokens, rows_per_item);
    active[index] = true;
  }

  const uint64_t vocab_row_bytes =
      (uint64_t)DS4_N_VOCAB * sizeof(float);
  const uint64_t logits_row_stride =
      (uint64_t)rows_per_item * DS4_N_VOCAB;
  for (uint32_t row = 0u; row < rows_per_item; ++row) {
    if (ds4_gpu_tensor_write(
            coordinator->prefill_tokens, 0u, previous.data(),
            (uint64_t)item_count * sizeof(previous[0])) == 0) {
      return false;
    }

    const uint64_t logits_bytes =
        ((uint64_t)(item_count - 1u) * rows_per_item + 1u) *
        vocab_row_bytes;
    ds4_gpu_tensor* logits_view = ds4_gpu_tensor_view(
        logits_rows, (uint64_t)row * vocab_row_bytes, logits_bytes);
    if (!logits_view) {
      return false;
    }
    const bool ok =
        ds4_gpu_dspark_markov_argmax_batch_tensor(
            coordinator->dspark_markov_index,
            coordinator->dspark_markov_key, logits_view,
            coordinator->prefill_tokens, sm->map, sm->size,
            last->markov_w1->abs_offset, last->markov_w2->abs_offset,
            DS4_N_VOCAB, d->markov_rank,
            static_cast<uint32_t>(item_count), logits_row_stride) != 0;
    ds4_gpu_tensor_free(logits_view);
    if (!ok || ds4_gpu_tensor_read(
                   coordinator->dspark_markov_index, 0u, selected.data(),
                   (uint64_t)item_count * sizeof(selected[0])) == 0) {
      return false;
    }

    for (size_t index = 0; index < item_count; ++index) {
      if (selected[index] < 0 || (uint32_t)selected[index] >= DS4_N_VOCAB) {
        return false;
      }
      if (!active[index] || row >= row_limits[index]) {
        active[index] = false;
        continue;
      }

      items[index].tokens[row] = selected[index];
      previous[index] = selected[index];
      selected_rows[index] = row + 1u;
    }
  }
  for (size_t index = 0; index < item_count; ++index) {
    *items[index].n_tokens = selected_rows[index];
  }
  return true;
}

/*
 * Propose the tokens following the target-selected token at position pos0.
 *
 * `target_next_token` is the target's selected next token; it seeds both the
 * block's first embedding slot and the Markov chain. Requires that the target
 * has already captured its sampled hidden states for position pos0 - 1 and that
 * those have been injected into the rings.
 */
static bool rocm_graph_dspark_prepare_hidden(
    ds4_gpu_graph* g, const ds4_model* target_model,
    const ds4_weights* target_weights, int target_next_token, uint32_t pos0,
    ds4_gpu_tensor* hidden_out, uint32_t hidden_row_offset) {
  const ds4_dspark_model* d = g->dspark;
  if (!d)
    return false;
  const uint32_t block = d->block_size;
  const uint32_t rows = block;
  if (block == 0 || block > g->spec_rows_cap || rows > g->prefill_cap)
    return false;
  if (pos0 == 0 || g->dspark_context_len < pos0)
    return false;
  if (rows > g->dspark_cache_cap)
    return false;

  // The target's selected next token seeds draft row zero and the Markov head.
  int32_t block_tokens[DS4_DSPARK_MAX_BLOCK];
  block_tokens[0] = target_next_token;
  for (uint32_t i = 1; i < rows; i++) {
    block_tokens[i] = (int32_t)d->noise_token_id;
  }
  if (ds4_gpu_tensor_write(g->prefill_tokens, 0, block_tokens,
                           (uint64_t)block * sizeof(block_tokens[0])) == 0) {
    return false;
  }

  bool ok = ds4_gpu_begin_commands() != 0;
  /* Tied target embeddings, expanded across the hyper-connection streams. */
  if (ok)
    ok = ds4_gpu_embed_tokens_hc_tensor(
             g->batch_cur_hc, g->prefill_tokens, target_model->map,
             target_model->size, target_weights->token_embd->abs_offset,
             DS4_N_VOCAB, rows, DS4_N_EMBD, DS4_N_HC,
             target_weights->token_embd->type) != 0;
  for (uint32_t stage = 0; ok && stage < d->n_stages; stage++) {
    ok = rocm_graph_encode_dspark_stage(g, stage, pos0, rows);
  }
  if (ok) {
    ok = rocm_graph_encode_dspark_final_hidden(g, block, hidden_out,
                                               hidden_row_offset);
  }
  if (ok) {
    ok = ds4_gpu_end_commands() != 0;
  } else {
    (void)ds4_gpu_synchronize();
  }
  return ok;
}

/*
 * Run one support-model block for every active request.
 *
 * All sessions intentionally borrow one large batch scratch arena. Keep each
 * request's persistent draft rows in the dedicated flattened buffers and stage
 * only the request currently running its stateful attention into shared
 * scratch. The support FFN/MoE is stateless, so it can consume all flattened
 * rows in one pass while retaining one router group per request.
 */
static bool rocm_graph_dspark_prepare_hidden_batch(
    ds4_engine* engine, const ds4_rocm_dspark_draft_item* items,
    size_t item_count, uint32_t block, ds4_gpu_tensor* hidden_out) {
  if (!engine || !engine->model || !engine->weights || !items ||
      item_count < 2u || item_count > 8u || !hidden_out) {
    return false;
  }
  ds4_gpu_graph* coordinator = items[0].graph;
  const ds4_dspark_model* d = coordinator ? coordinator->dspark : nullptr;
  if (!coordinator || !d || !coordinator->batch_dspark_verify_cur_hc ||
      !coordinator->batch_dspark_verify_after_attn_hc) {
    return false;
  }
  const uint32_t rows = block;
  const uint32_t total_draft_rows = static_cast<uint32_t>(item_count) * block;
  if (block == 0u || block > d->block_size ||
      total_draft_rows > coordinator->prefill_cap ||
      total_draft_rows > coordinator->batch_dspark_verify_rows_cap ||
      total_draft_rows > DS4_SPEC_MAX_ROWS) {
    return false;
  }

  std::vector<int32_t> flat_tokens(total_draft_rows);
  for (size_t index = 0; index < item_count; ++index) {
    const ds4_rocm_dspark_draft_item& item = items[index];
    ds4_gpu_graph* graph = item.graph;
    if (!graph || graph->dspark != d || item.position == 0u ||
        graph->dspark_context_len < item.position ||
        rows > graph->prefill_cap || rows > graph->dspark_cache_cap) {
      return false;
    }
    const uint32_t row0 = static_cast<uint32_t>(index) * block;
    flat_tokens[row0] = item.target_next_token;
    for (uint32_t row = 1u; row < block; ++row) {
      flat_tokens[row0 + row] =
          static_cast<int32_t>(d->noise_token_id);
    }
  }
  if (ds4_gpu_tensor_write(
          coordinator->prefill_tokens, 0, flat_tokens.data(),
          static_cast<uint64_t>(total_draft_rows) *
              sizeof(flat_tokens[0])) == 0) {
    return false;
  }

  const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
  const uint64_t hc_row_bytes = hc_dim * sizeof(float);
  const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
  const bool combine_attn_q8 =
      item_count >= 4u && (item_count & 1u) == 0u && rows <= 6u;
  std::array<uint32_t, 9> front_offsets{};
  for (size_t index = 0; index <= item_count; ++index) {
    front_offsets[index] = static_cast<uint32_t>(index) * rows;
  }

  auto with_front_slice =
      [&](ds4_gpu_graph* graph, uint32_t row0, auto&& operation) {
        std::array<ds4_gpu_tensor**, 6> slots{{
            &graph->batch_cur_hc,
            &graph->batch_flat_hc,
            &graph->batch_hc_mix,
            &graph->batch_hc_split,
            &graph->batch_attn_cur,
            &graph->batch_attn_norm,
        }};
        std::array<ds4_gpu_tensor*, 6> bases{{
            coordinator->batch_cur_hc,
            coordinator->batch_flat_hc,
            coordinator->batch_hc_mix,
            coordinator->batch_hc_split,
            coordinator->batch_attn_cur,
            coordinator->batch_attn_norm,
        }};
        const std::array<uint64_t, 6> row_bytes{{
            hc_row_bytes,
            hc_row_bytes,
            mix_hc * sizeof(float),
            mix_hc * sizeof(float),
            (uint64_t)DS4_N_EMBD * sizeof(float),
            (uint64_t)DS4_N_EMBD * sizeof(float),
        }};
        std::array<ds4_gpu_tensor*, 6> saved{};
        std::array<ds4_gpu_tensor*, 6> views{};
        bool slice_ok = true;
        for (size_t field = 0; field < slots.size(); ++field) {
          saved[field] = *slots[field];
          views[field] = ds4_gpu_tensor_view(
              bases[field], (uint64_t)row0 * row_bytes[field],
              (uint64_t)rows * row_bytes[field]);
          slice_ok = slice_ok && views[field] != nullptr;
        }
        if (slice_ok) {
          for (size_t field = 0; field < slots.size(); ++field) {
            *slots[field] = views[field];
          }
          slice_ok = operation();
          for (size_t field = 0; field < slots.size(); ++field) {
            *slots[field] = saved[field];
          }
        }
        for (ds4_gpu_tensor* view : views) {
          ds4_gpu_tensor_free(view);
        }
        return slice_ok;
      };

  auto with_projected_slice =
      [&](ds4_gpu_graph* graph, uint32_t front_row0, auto&& operation) {
        ds4_gpu_tensor* saved_cur_hc = graph->batch_cur_hc;
        ds4_gpu_tensor* saved_hc_split = graph->batch_hc_split;
        ds4_gpu_tensor* saved_attn_norm = graph->batch_attn_norm;
        ds4_gpu_tensor* saved_q = graph->batch_q;
        ds4_gpu_tensor* saved_kv = graph->batch_kv;
        ds4_gpu_tensor* cur_hc = ds4_gpu_tensor_view(
            coordinator->batch_cur_hc,
            (uint64_t)front_row0 * hc_row_bytes,
            (uint64_t)rows * hc_row_bytes);
        ds4_gpu_tensor* hc_split = ds4_gpu_tensor_view(
            coordinator->batch_hc_split,
            (uint64_t)front_row0 * mix_hc * sizeof(float),
            (uint64_t)rows * mix_hc * sizeof(float));
        ds4_gpu_tensor* attn_norm = ds4_gpu_tensor_view(
            coordinator->batch_attn_norm,
            (uint64_t)front_row0 * DS4_N_EMBD * sizeof(float),
            (uint64_t)rows * DS4_N_EMBD * sizeof(float));
        ds4_gpu_tensor* q = ds4_gpu_tensor_view(
            coordinator->batch_q,
            (uint64_t)front_row0 * DS4_N_HEAD * DS4_N_HEAD_DIM * sizeof(float),
            (uint64_t)block * DS4_N_HEAD * DS4_N_HEAD_DIM * sizeof(float));
        ds4_gpu_tensor* kv = ds4_gpu_tensor_view(
            coordinator->batch_kv,
            (uint64_t)front_row0 * DS4_N_HEAD_DIM * sizeof(float),
            (uint64_t)rows * DS4_N_HEAD_DIM * sizeof(float));
        bool slice_ok =
            cur_hc && hc_split && attn_norm && q && kv;
        if (slice_ok) {
          graph->batch_cur_hc = cur_hc;
          graph->batch_hc_split = hc_split;
          graph->batch_attn_norm = attn_norm;
          graph->batch_q = q;
          graph->batch_kv = kv;
          slice_ok = operation();
          graph->batch_cur_hc = saved_cur_hc;
          graph->batch_hc_split = saved_hc_split;
          graph->batch_attn_norm = saved_attn_norm;
          graph->batch_q = saved_q;
          graph->batch_kv = saved_kv;
        }
        ds4_gpu_tensor_free(kv);
        ds4_gpu_tensor_free(q);
        ds4_gpu_tensor_free(attn_norm);
        ds4_gpu_tensor_free(hc_split);
        ds4_gpu_tensor_free(cur_hc);
        return slice_ok;
      };

  bool ok = ds4_gpu_begin_commands() != 0;
  if (ok) {
    ok = ds4_gpu_embed_tokens_hc_tensor(
             coordinator->batch_dspark_verify_cur_hc,
             coordinator->prefill_tokens, engine->model->map,
             engine->model->size,
             engine->weights->token_embd->abs_offset, DS4_N_VOCAB,
             total_draft_rows, DS4_N_EMBD, DS4_N_HC,
             engine->weights->token_embd->type) != 0;
  }

  for (uint32_t stage = 0; ok && stage < d->n_stages; ++stage) {
    const ds4_layer_weights* layer = &d->stage[stage].block;
    if (combine_attn_q8) {
      const uint64_t q_rank = layer->attn_q_a->dim[1];
      const uint64_t q_dim =
          (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
      for (size_t index = 0; ok && index < item_count; ++index) {
        ds4_gpu_graph* graph = items[index].graph;
        const uint32_t front_row0 = front_offsets[index];
        const uint64_t block_offset =
            static_cast<uint64_t>(index) * block * hc_row_bytes;
        ok = ds4_gpu_tensor_copy(
                 coordinator->batch_cur_hc, (uint64_t)front_row0 * hc_row_bytes,
                 coordinator->batch_dspark_verify_cur_hc, block_offset,
                 static_cast<uint64_t>(block) * hc_row_bytes) != 0;
        if (ok) {
          ok = with_front_slice(graph, front_row0, [&]() {
            return rocm_graph_encode_dspark_stage_attention(
                graph, stage, items[index].position, rows, false, true,
                false, false);
          });
        }
      }
      if (ok) {
        ok = ds4_gpu_matmul_q8_0_group_pairs_tensor(
                 coordinator->batch_qr, d->model->map, d->model->size,
                 layer->attn_q_a->abs_offset, DS4_N_EMBD, q_rank,
                 coordinator->batch_attn_norm, total_draft_rows,
                 front_offsets.data(), static_cast<uint32_t>(item_count)) != 0;
      }
      if (ok) {
        ok = ds4_gpu_matmul_q8_0_group_pairs_tensor(
                 coordinator->batch_kv_raw, d->model->map, d->model->size,
                 layer->attn_kv->abs_offset, DS4_N_EMBD, DS4_N_HEAD_DIM,
                 coordinator->batch_attn_norm, total_draft_rows,
                 front_offsets.data(), static_cast<uint32_t>(item_count)) != 0;
      }
      if (ok) {
        ok =
            ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
                coordinator->batch_qr_norm, coordinator->batch_qr,
                d->model->map, d->model->size, layer->attn_q_a_norm->abs_offset,
                static_cast<uint32_t>(q_rank), coordinator->batch_kv,
                coordinator->batch_kv_raw, layer->attn_kv_a_norm->abs_offset,
                DS4_N_HEAD_DIM, total_draft_rows, DS4_RMS_EPS) != 0;
      }
      if (ok) {
        ok = ds4_gpu_matmul_q8_0_group_pairs_tensor(
                 coordinator->batch_q, d->model->map, d->model->size,
                 layer->attn_q_b->abs_offset, q_rank, q_dim,
                 coordinator->batch_qr_norm, total_draft_rows,
                 front_offsets.data(), static_cast<uint32_t>(item_count)) != 0;
      }
      for (size_t index = 0; ok && index < item_count; ++index) {
        ds4_gpu_graph* graph = items[index].graph;
        const uint32_t front_row0 = front_offsets[index];
        const uint64_t block_offset =
            static_cast<uint64_t>(index) * block * hc_row_bytes;
        ok = with_projected_slice(graph, front_row0, [&]() {
          return rocm_graph_encode_dspark_stage_attention(
              graph, stage, items[index].position, rows, true, false,
              false, false);
        });
        if (ok) {
          ok = ds4_gpu_tensor_copy(
                   coordinator->batch_dspark_verify_after_attn_hc,
                   block_offset, graph->batch_after_attn_hc, 0,
                   static_cast<uint64_t>(block) * hc_row_bytes) != 0;
        }
      }
    } else {
      for (size_t index = 0; ok && index < item_count; ++index) {
        ds4_gpu_graph* graph = items[index].graph;
        const uint64_t block_offset =
            static_cast<uint64_t>(index) * block * hc_row_bytes;
        ok =
            ds4_gpu_tensor_copy(
                graph->batch_cur_hc, 0, coordinator->batch_dspark_verify_cur_hc,
                block_offset, static_cast<uint64_t>(block) * hc_row_bytes) != 0;
        if (ok) {
          ok = rocm_graph_encode_dspark_stage_attention(
              graph, stage, items[index].position, rows);
        }
        if (ok) {
          ok = ds4_gpu_tensor_copy(
                   coordinator->batch_dspark_verify_after_attn_hc,
                   block_offset, graph->batch_after_attn_hc, 0,
                   static_cast<uint64_t>(block) * hc_row_bytes) != 0;
        }
      }
    }
    if (ok) {
      ok = ds4_gpu_tensor_copy(
               coordinator->batch_after_attn_hc, 0,
               coordinator->batch_dspark_verify_after_attn_hc, 0,
               static_cast<uint64_t>(total_draft_rows) * hc_row_bytes) != 0;
    }
    if (ok) {
      ok = rocm_graph_encode_layer_ffn_batch(
          coordinator, d->model, layer, stage, items[0].position,
          total_draft_rows, coordinator->prefill_tokens);
    }
    if (ok) {
      ok = ds4_gpu_tensor_copy(
               coordinator->batch_dspark_verify_cur_hc, 0,
               coordinator->batch_next_hc, 0,
               static_cast<uint64_t>(total_draft_rows) * hc_row_bytes) != 0;
    }
  }

  if (ok) {
    ok = ds4_gpu_tensor_copy(
             coordinator->batch_cur_hc, 0,
             coordinator->batch_dspark_verify_cur_hc, 0,
             static_cast<uint64_t>(total_draft_rows) * hc_row_bytes) != 0;
  }
  if (ok) {
    ok = rocm_graph_encode_dspark_final_hidden(coordinator, total_draft_rows,
                                               hidden_out, 0u);
  }
  if (ok) {
    ok = ds4_gpu_end_commands() != 0;
  } else {
    (void)ds4_gpu_synchronize();
  }
  return ok;
}

static bool rocm_graph_dspark_draft(ds4_gpu_graph* g,
                                    const ds4_model* target_model,
                                    const ds4_weights* target_weights,
                                    int target_next_token, uint32_t pos0,
                                    int32_t* tokens_out, uint32_t* n_out,
                                    const ds4_dspark_sampler* sampler,
                                    uint32_t max_tokens, size_t concurrency) {
  if (!g || !tokens_out || !n_out)
    return false;
  *n_out = 0;
  const ds4_dspark_model* d = g->dspark;
  if (!d)
    return false;
  const uint32_t block = d->block_size;
  const uint32_t selected_rows =
      max_tokens ? std::min(block, max_tokens) : block;

  /* Mode 2 keeps generic narrow-batch routing available to the support
   * model without enabling verifier-only arithmetic candidates. */
  ds4_gpu_set_small_batch_mode(2);
  bool ok = rocm_graph_dspark_prepare_hidden(g, target_model, target_weights,
                                             target_next_token, pos0,
                                             g->batch_ffn_norm, 0u);
  if (!ok) {
    ds4_gpu_set_small_batch_mode(0);
    return false;
  }
  ok = ds4_gpu_begin_commands() != 0;
  if (ok) {
    // Non-causal support attention still needs the complete block. Its final
    // vocabulary projection and Markov selection only need the requested
    // prefix; later positions cannot influence earlier selections.
    ok = rocm_graph_encode_dspark_logits(g, target_model, target_weights,
                                         g->batch_ffn_norm, selected_rows);
  }
  if (ok) {
    ok = ds4_gpu_end_commands() != 0;
  } else {
    (void)ds4_gpu_synchronize();
  }
  if (!ok) {
    ds4_gpu_set_small_batch_mode(0);
    return false;
  }

  const ds4_rocm_dspark_draft_item item{.graph = g,
                                        .target_next_token = target_next_token,
                                        .position = pos0,
                                        .max_draft_tokens = max_tokens,
                                        .tokens = tokens_out,
                                        .n_tokens = n_out,
                                        .sampler = sampler,
                                        .concurrency = concurrency};
  const bool sampled = sampler && sampler->propose;
  const bool selected =
      sampled
          ? rocm_graph_dspark_select_sampled(g, &item, 1, block)
          : rocm_graph_dspark_select_block(g, g->spec_logits, target_next_token,
                                           selected_rows, tokens_out);
  ds4_gpu_set_small_batch_mode(0);
  if (!selected) {
    return false;
  }
  if (!sampled)
    *n_out = selected_rows;
  return true;
}

/*
 * Draft each request with its session-local support attention and KV state,
 * then stream the tied target vocabulary projection once across all hidden
 * rows. This preserves the support model's arithmetic through final norm and
 * changes only the row count of the shared Q8 output projection.
 */
static bool rocm_graph_dspark_draft_head_batch(
    ds4_engine* engine, const ds4_rocm_dspark_draft_item* items,
    size_t item_count) {
  if (!engine || !engine->model || !engine->weights || !items ||
      item_count < 2u || item_count > 8u) {
    return false;
  }
  ds4_gpu_graph* coordinator = items[0].graph;
  if (!coordinator || !coordinator->dspark ||
      !coordinator->batch_dspark_verify_after_attn_hc) {
    return false;
  }
  const uint32_t block = coordinator->dspark->block_size;
  uint32_t requested_block = 0u;
  for (size_t index = 0; index < item_count; ++index) {
    requested_block = std::max(
        requested_block,
        std::min(block, items[index].max_draft_tokens));
  }
  const uint32_t total_rows =
      static_cast<uint32_t>(item_count) * requested_block;
  if (requested_block == 0u || total_rows > coordinator->prefill_cap ||
      total_rows > coordinator->batch_dspark_verify_rows_cap ||
      total_rows > DS4_SPEC_MAX_ROWS ||
      !rocm_graph_spec_prepare(coordinator, engine->weights, total_rows)) {
    return false;
  }

  bool ok = true;
  for (size_t index = 0; index < item_count; ++index) {
    const ds4_rocm_dspark_draft_item& item = items[index];
    if (!item.graph || item.graph->dspark != coordinator->dspark ||
        !item.tokens || !item.n_tokens) {
      return false;
    }
    *item.n_tokens = 0u;
  }

  ds4_gpu_set_small_batch_mode(2);

  ok = rocm_graph_dspark_prepare_hidden_batch(
      engine, items, item_count, requested_block,
      coordinator->batch_dspark_verify_after_attn_hc);

  if (ok)
    ok = ds4_gpu_begin_commands() != 0;
  if (ok) {
    ok = rocm_graph_encode_dspark_logits(
        coordinator, engine->model, engine->weights,
        coordinator->batch_dspark_verify_after_attn_hc, total_rows);
  }
  if (ok) {
    ok = ds4_gpu_end_commands() != 0;
  } else {
    (void)ds4_gpu_synchronize();
  }
  if (!ok) {
    ds4_gpu_set_small_batch_mode(0);
    return false;
  }

  const bool sampled = std::any_of(
      items, items + item_count,
      [](const auto& item) { return item.sampler && item.sampler->propose; });
  ok = sampled ? rocm_graph_dspark_select_sampled(coordinator, items,
                                                  item_count, requested_block)
               : rocm_graph_dspark_select_block_batch(
                     coordinator, items, item_count, coordinator->spec_logits,
                     requested_block);

  ds4_gpu_set_small_batch_mode(0);
  return ok;
}

/* Pick a raw SWA cache size for ROCm.  During batched prefill it must cover
 * the previous window plus the current ubatch. */
static uint32_t rocm_graph_raw_cap_for_context(int ctx_size, uint32_t prefill_cap) {
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > (uint32_t)ctx_size) raw_window = (uint32_t)ctx_size;
    if (raw_window == 0) raw_window = 1;

    /*
     * During batched prefill the SWA cache must hold the current ubatch plus
     * the previous logical window. The cache is padded to a 256-row multiple
     * so the physical row order and FlashAttention block grouping match the
     * model path we compare against.
     */
    uint64_t wanted = (uint64_t)raw_window + prefill_cap;
    if (wanted > (uint32_t)ctx_size) wanted = (uint32_t)ctx_size;
    if (wanted == 0) wanted = 1;
    wanted = ds4_align_up(wanted, 256u);
    if (wanted > 8192u) wanted = 8192u;
    uint32_t raw_cap = (uint32_t)wanted;
    if (raw_cap < raw_window) raw_cap = raw_window;

    return raw_cap;
}

/* Choose the prefill ubatch size.  Whole-batch is fastest for normal prompts;
 * long prompts default to 4096-token chunks. */
static uint32_t rocm_graph_prefill_cap_for_prompt(int prompt_len) {
    return ds4_default_prefill_cap_for_prompt(prompt_len);
}

/* Extend shared prefixes with batched prefill once the suffix is large enough
 * to amortize batch setup. */
static uint32_t rocm_graph_resume_prefill_min_tokens(void) {
    return 4u;
}

uint32_t ds4_rocm_graph_prefill_capacity(int context_size) {
    return rocm_graph_prefill_cap_for_prompt(context_size);
}

uint32_t ds4_rocm_graph_resume_prefill_min_tokens(void) {
    return rocm_graph_resume_prefill_min_tokens();
}

ds4_rocm_graph *ds4_rocm_graph_create(ds4_engine *engine,
                                      int context_size,
                                      uint32_t prefill_capacity) {
  if (!engine || !engine->weights || context_size <= 0 ||
      prefill_capacity == 0) {
    return NULL;
  }

  // The head is the largest arena. Drop idle older arenas, and drop an idle
  // head before growing it, so increasing session capacities do not retain
  // every previous allocation. Active borrowers keep their original buffers.
  for (auto** link = &engine->batch_workspace; *link;) {
    auto* workspace = *link;
    const bool reuse_head = workspace == engine->batch_workspace &&
                            workspace->prefill_cap >= prefill_capacity;
    if (workspace->batch_workspace_borrowers != 0 || reuse_head) {
      link = &workspace->next_batch_workspace;
    } else {
      *link = workspace->next_batch_workspace;
      rocm_graph_free(workspace);
      free(workspace);
    }
  }
    auto *graph =
        static_cast<ds4_rocm_graph *>(ds4_xcalloc(1, sizeof(ds4_rocm_graph)));
    const uint32_t raw_capacity =
        rocm_graph_raw_cap_for_context(context_size, prefill_capacity);
    ds4_rocm_graph* batch_workspace = engine->batch_workspace;
    while (batch_workspace && batch_workspace->prefill_cap < prefill_capacity) {
      batch_workspace = batch_workspace->next_batch_workspace;
    }
    if (!rocm_graph_alloc_raw_cap(
            graph, engine->weights, &engine->weights->layer[0], raw_capacity,
            (uint32_t)context_size, prefill_capacity, batch_workspace)) {
      free(graph);
      return NULL;
    }
    if (!batch_workspace) {
      if (engine->dspark) {
        const uint32_t verify_rows_cap = 8u * (engine->dspark->block_size + 1u);
        const uint64_t verify_hc_elems =
            (uint64_t)verify_rows_cap * DS4_N_HC * DS4_N_EMBD;
        graph->batch_dspark_verify_cur_hc =
            ds4_gpu_tensor_alloc(verify_hc_elems * sizeof(float));
        graph->batch_dspark_verify_after_attn_hc =
            ds4_gpu_tensor_alloc(verify_hc_elems * sizeof(float));
        if (!graph->batch_dspark_verify_cur_hc ||
            !graph->batch_dspark_verify_after_attn_hc) {
          rocm_graph_free(graph);
          free(graph);
          return NULL;
        }
        graph->batch_dspark_verify_rows_cap = verify_rows_cap;
      }
      auto* workspace =
          static_cast<ds4_rocm_graph*>(ds4_xcalloc(1, sizeof(ds4_rocm_graph)));
      workspace->prefill_cap = graph->prefill_cap;
      rocm_graph_bind_batch_workspace(workspace, graph);
      workspace->borrows_batch_workspace = false;
      graph->borrows_batch_workspace = true;
      workspace->next_batch_workspace = engine->batch_workspace;
      engine->batch_workspace = workspace;
      batch_workspace = workspace;
      gufo::server::Logger::LogFormatted(
          gufo::server::LogLevel::kInfo, "ds4",
          "sessions share one %u-row batch workspace", graph->prefill_cap);
    }
    graph->batch_workspace_owner = batch_workspace;
    ++batch_workspace->batch_workspace_borrowers;
    return graph;
}

void ds4_rocm_graph_destroy(ds4_rocm_graph *graph) {
  while (graph) {
    ds4_rocm_graph* next = graph->next_batch_workspace;
    if (graph->batch_workspace_owner)
      --graph->batch_workspace_owner->batch_workspace_borrowers;
    rocm_graph_free(graph);
    free(graph);
    graph = next;
  }
}

bool ds4_rocm_graph_reset(ds4_rocm_graph *graph) {
    return graph && rocm_graph_reset_prefill_state(graph);
}

/* V4.1 interim correctness driver: run prefill as sequential decode steps.
 * The batch prefill path does not mirror the decode source-sharing yet
 * (owner index caches, published top-k, candidate masks), so feeding the
 * prompt through the verified single-token engine keeps semantics exact at
 * O(n) decode-step cost.  Gated by GUFO_DS4_PREFILL_AS_DECODE. */
static bool rocm_graph_prefill_tokens_as_decode(
        ds4_gpu_graph *g, const ds4_model *model, const ds4_weights *weights,
        const token_vec *prompt, uint32_t start, uint32_t n_tokens,
        float *logits) {
    const uint32_t steady_rows =
        g->dspark ? g->dspark->block_size + 1u : 1u;
    if (!rocm_graph_dspark_resize_capture(g, steady_rows)) return false;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        if (!rocm_graph_encode_token_raw_swa(g, model, weights,
                                             prompt->v[start + i],
                                             start + i,
                                             i + 1u == n_tokens)) {
            fprintf(stderr, "ds4: prefill-as-decode failed at pos %u\n",
                    start + i);
            return false;
        }
    }
    if (!logits) return true;
    fprintf(stderr, "ds4: prefillAD END n=%u\n", n_tokens);
    bool ok = ds4_gpu_tensor_read(g->logits, 0, logits,
                                  (uint64_t)DS4_N_VOCAB * sizeof(float));
    if (ok) ds4_dbg_logits_h(logits, "prefillAD", start + n_tokens - 1u);
    return ok;
}

bool ds4_rocm_graph_prefill(ds4_rocm_graph *graph,
                            ds4_engine *engine,
                            const ds4_tokens *prompt,
                            float *logits) {
    if (!graph || !engine || !prompt || prompt->len <= 0) return false;
    if (getenv("GUFO_DS4_PREFILL_AS_DECODE"))
        return rocm_graph_prefill_tokens_as_decode(
            graph, engine->model, engine->weights, prompt, 0,
            (uint32_t)prompt->len, logits);
    if (graph->prefill_cap < (uint32_t)prompt->len) {
        return rocm_graph_prefill_chunked(graph,
                                           engine->model,
                                           engine->weights,
                                           prompt,
                                           prompt->len,
                                           logits,
                                           false);
    }
    return rocm_graph_prefill_raw_swa(graph,
                                       engine->model,
                                       engine->weights,
                                       prompt,
                                       prompt->len,
                                       logits,
                                       false);
}

bool ds4_rocm_graph_prefill_range(ds4_rocm_graph *graph,
                                  ds4_engine *engine,
                                  const ds4_tokens *prompt,
                                  uint32_t start,
                                  uint32_t token_count,
                                                          float *logits) {
                                  if (getenv("GUFO_DS4_PREFILL_AS_DECODE"))
                                      return graph && engine && prompt &&
                                             rocm_graph_prefill_tokens_as_decode(
                                                 graph, engine->model, engine->weights, prompt, start,
                                                 token_count, logits);
                                  return graph && engine && prompt &&
           rocm_graph_prefill_chunked_range(graph,
                                             engine->model,
                                             engine->weights,
                                             prompt,
                                             start,
                                             token_count,
                                             logits,
                                             false);
}

bool ds4_rocm_graph_eval(ds4_rocm_graph *graph,
                         ds4_engine *engine,
                         int token,
                         uint32_t position,
                         float *logits) {
    return graph && engine &&
           rocm_graph_eval_token_raw_swa(graph,
                                          engine->model,
                                          engine->weights,
                                          token,
                                          position,
                                          logits);
}

bool ds4_rocm_graph_eval_batch(ds4_engine *engine,
                               const ds4_rocm_batch_item *items,
                               size_t item_count) {
    return engine && rocm_graph_eval_sessions_batch(engine->model,
                                                    engine->weights,
                                                    items,
                                                    item_count);
}

bool ds4_rocm_graph_spec_prepare(ds4_rocm_graph *graph,
                                 ds4_engine *engine,
                                 uint32_t rows_cap) {
    return graph && engine &&
           rocm_graph_spec_prepare(graph, engine->weights, rows_cap);
}

uint32_t ds4_rocm_graph_spec_rows_cap(const ds4_rocm_graph *graph) {
    return rocm_graph_spec_rows_cap(graph);
}

bool ds4_rocm_graph_spec_frontier_save(ds4_rocm_graph *graph) {
    return rocm_graph_spec_frontier_save(graph);
}

bool ds4_rocm_graph_spec_frontier_restore(ds4_rocm_graph *graph) {
    return rocm_graph_spec_frontier_restore(graph);
}

bool ds4_rocm_graph_spec_frontier_commit_prefix(ds4_rocm_graph *graph,
                                                uint32_t prefix_len) {
    return rocm_graph_spec_frontier_commit_prefix(graph, prefix_len);
}

bool ds4_rocm_graph_verify_suffix(ds4_rocm_graph *graph,
                                  ds4_engine *engine,
                                  const ds4_tokens *tokens,
                                  uint32_t start,
                                  uint32_t n_tokens,
                                  int32_t *row_tops) {
    return graph && engine &&
           rocm_graph_verify_suffix(graph,
                                     engine->model,
                                     engine->weights,
                                     tokens,
                                     start,
                                     n_tokens,
                                     row_tops);
}

bool ds4_rocm_graph_verify_batch(ds4_engine* engine,
                                 const ds4_rocm_verify_item* items,
                                 size_t item_count) {
  return rocm_graph_verify_sessions_batch(engine, items, item_count);
}

bool ds4_rocm_graph_read_spec_logits_row(const ds4_rocm_graph *graph,
                                         uint32_t row,
                                         float *logits) {
    return rocm_graph_read_spec_logits_row(graph, row, logits);
}

bool ds4_rocm_graph_dspark_attach(ds4_rocm_graph *graph,
                                  ds4_engine *engine,
                                  const ds4_dspark_model *dspark) {
    return graph && engine &&
           rocm_graph_dspark_attach(graph, engine->weights, dspark);
}

uint32_t ds4_rocm_graph_dspark_block_size(const ds4_rocm_graph *graph) {
    return graph && graph->dspark ? graph->dspark->block_size : 0u;
}

void ds4_rocm_graph_dspark_set_capture_enabled(ds4_rocm_graph *graph,
                                               bool enabled) {
    if (!graph || !graph->dspark) return;
    graph->dspark_capture_enabled = enabled;
    if (!enabled) {
        graph->dspark_context_len = 0u;
        graph->dspark_capture_mask = 0u;
        graph->dspark_capture_batch_mask = 0u;
        graph->dspark_capture_batch_start = 0u;
        graph->dspark_capture_batch_tokens = 0u;
    }
}

bool ds4_rocm_graph_dspark_capture_ready(const ds4_rocm_graph *graph) {
    return graph && rocm_graph_dspark_capture_complete(graph);
}

void ds4_rocm_graph_dspark_capture_reset(ds4_rocm_graph *graph) {
    if (!graph) return;
    graph->dspark_capture_mask = 0u;
    graph->dspark_capture_batch_mask = 0u;
    graph->dspark_capture_batch_start = 0u;
    graph->dspark_capture_batch_tokens = 0u;
}

uint32_t ds4_rocm_graph_dspark_batch_capture_rows(const ds4_rocm_graph *graph,
                                                 uint32_t *start) {
    if (!graph || !graph->dspark) return 0u;
    const uint32_t want = rocm_graph_dspark_complete_mask(graph);
    if (want == 0u || graph->dspark_capture_batch_mask != want) return 0u;
    if (start) *start = graph->dspark_capture_batch_start;
    return graph->dspark_capture_batch_tokens;
}

bool ds4_rocm_graph_dspark_inject(ds4_rocm_graph *graph,
                                  uint32_t pos0,
                                  uint32_t n_rows) {
    if (!graph) return false;
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = rocm_graph_dspark_inject(graph, pos0, n_rows);
    if (ok) {
        ok = ds4_gpu_end_commands() != 0;
    } else {
        (void)ds4_gpu_synchronize();
    }
    return ok;
}

uint32_t ds4_rocm_graph_dspark_context_len(const ds4_rocm_graph *graph) {
    return graph ? graph->dspark_context_len : 0u;
}

void ds4_rocm_graph_dspark_truncate_context(ds4_rocm_graph *graph, uint32_t length) {
    if (!graph) return;
    if (length < graph->dspark_context_len) graph->dspark_context_len = length;
}

bool ds4_rocm_graph_dspark_draft(ds4_rocm_graph* graph, ds4_engine* engine,
                                 int target_next_token, uint32_t pos0,
                                 int32_t* tokens_out, uint32_t* n_out,
                                 const ds4_dspark_sampler* sampler,
                                 uint32_t max_tokens, size_t concurrency) {
  return graph && engine &&
         rocm_graph_dspark_draft(graph, engine->model, engine->weights,
                                 target_next_token, pos0, tokens_out, n_out,
                                 sampler, max_tokens, concurrency);
}

bool ds4_rocm_graph_dspark_draft_head_batch(
    ds4_engine* engine, const ds4_rocm_dspark_draft_item* items,
    size_t item_count) {
  return rocm_graph_dspark_draft_head_batch(engine, items, item_count);
}

/* =========================================================================
 * Session Snapshot Payloads.
 * =========================================================================
 *
 * The server disk cache stores a high-level file header, then delegates the
 * graph-specific payload below to the engine.  This payload is intentionally
 * not mmaped: restoring a checkpoint copies bytes back into the already
 * allocated ROCm tensors, preserving the same live graph buffers used by
 * normal prefill/decode.  The raw SWA cache is serialized as the last logical
 * window only; suffix prefill writes its own raw rows before attention.  The
 * compressed caches are serialized up to their live row counts because sparse
 * attention may select rows from the whole prefix.
 *
 * The payload is model-specific rather than self-describing.  The fixed header
 * records enough shape information to reject a file written for a different
 * DS4 runtime, then the body writes: checkpoint tokens, last logits, per-layer
 * compressed row counts, raw SWA rows in logical order, compressed attention
 * rows, and the compressor/indexer frontiers. DSpark additionally persists its
 * injected support KV and deterministic request policy.
 * Temporary captures are unnecessary once support coverage reaches the
 * checkpoint.
 */

#define DS4_SESSION_PAYLOAD_MAGIC UINT32_C(0x34565344) /* "DSV4" */
#define DS4_SESSION_PAYLOAD_U32_FIELDS (19u + DS4_DSPARK_MAX_TARGET_LAYERS)
#define DS4_SESSION_IO_CHUNK (8u * 1024u * 1024u)

static constexpr std::array<uint64_t ds4_dspark_request_state::*, 12>
    dspark_snapshot_counters{
        &ds4_dspark_request_state::drafted,
        &ds4_dspark_request_state::accepted,
        &ds4_dspark_request_state::support_drafted,
        &ds4_dspark_request_state::support_accepted,
        &ds4_dspark_request_state::positional_accepted,
        &ds4_dspark_request_state::anchors,
        &ds4_dspark_request_state::full_blocks,
        &ds4_dspark_request_state::steps,
        &ds4_dspark_request_state::skipped,
        &ds4_dspark_request_state::policy_start_drafted,
        &ds4_dspark_request_state::policy_start_accepted,
        &ds4_dspark_request_state::policy_start_steps,
    };

static void payload_set_err(char *err, size_t errlen, const char *msg) {
    if (errlen != 0) snprintf(err, errlen, "%s", msg);
}

static void payload_put_u32(uint8_t out[4], uint32_t v) {
    out[0] = (uint8_t)(v);
    out[1] = (uint8_t)(v >> 8);
    out[2] = (uint8_t)(v >> 16);
    out[3] = (uint8_t)(v >> 24);
}

static uint32_t payload_get_u32(const uint8_t in[4]) {
    return (uint32_t)in[0] |
           ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}

static int payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes, char *err, size_t errlen) {
    const auto *p = static_cast<const uint8_t *>(ptr);
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fwrite(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to write session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    return 0;
}

static int payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes, uint64_t *remaining, char *err, size_t errlen) {
    if (remaining && *remaining < bytes) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    const uint64_t original = bytes;
    auto *p = static_cast<uint8_t *>(ptr);
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fread(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to read session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    if (remaining) *remaining -= original;
    return 0;
}

static int payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen) {
    uint8_t b[4];
    payload_put_u32(b, v);
    return payload_write_bytes(fp, b, sizeof(b), err, errlen);
}

static int payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining, char *err, size_t errlen) {
    uint8_t b[4];
    if (remaining && *remaining < sizeof(b)) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    if (fread(b, 1, sizeof(b), fp) != sizeof(b)) {
        payload_set_err(err, errlen, "failed to read session payload");
        return 1;
    }
    if (remaining) *remaining -= sizeof(b);
    *v = payload_get_u32(b);
    return 0;
}

static uint64_t layer_attn_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * DS4_N_HEAD_DIM * coff * ratio * sizeof(float);
}

static int payload_write_dspark_state(FILE* fp,
                                      const ds4_dspark_request_state& state,
                                      char* err, size_t errlen) {
  for (const auto field : dspark_snapshot_counters) {
    const uint64_t value = state.*field;
    if (payload_write_u32(fp, static_cast<uint32_t>(value), err, errlen) ||
        payload_write_u32(fp, static_cast<uint32_t>(value >> 32), err, errlen))
      return 1;
  }
  for (const uint32_t value :
       {state.skip_remaining, state.concurrent_width,
        uint32_t(state.plain_only), uint32_t(state.force_plain_request),
        state.policy_concurrency, state.probe_cycles, state.probe_emitted,
        state.probe_cost}) {
    if (payload_write_u32(fp, value, err, errlen))
      return 1;
  }
  return 0;
}

static int payload_read_dspark_state(FILE* fp, ds4_dspark_request_state& state,
                                     uint64_t* remaining, char* err,
                                     size_t errlen) {
  for (const auto field : dspark_snapshot_counters) {
    uint32_t low = 0, high = 0;
    if (payload_read_u32(fp, &low, remaining, err, errlen) ||
        payload_read_u32(fp, &high, remaining, err, errlen))
      return 1;
    state.*field = uint64_t(low) | (uint64_t(high) << 32);
  }
  uint32_t flags[8]{};
  for (auto& value : flags) {
    if (payload_read_u32(fp, &value, remaining, err, errlen))
      return 1;
  }
  if (flags[0] > DS4_DSPARK_RETRY_TOKENS || flags[1] == 0 ||
      flags[1] > DS4_DSPARK_MAX_BLOCK || flags[2] > 1 || flags[3] > flags[2] ||
      state.accepted > state.drafted ||
      state.support_accepted > state.support_drafted ||
      state.positional_accepted > state.support_drafted ||
      state.full_blocks > state.steps || state.anchors != state.steps ||
      state.policy_start_drafted > state.support_drafted ||
      state.policy_start_accepted > state.support_accepted ||
      state.policy_start_steps > state.steps || flags[4] < 1 || flags[4] > 8 ||
      flags[5] > 3 || flags[6] > flags[5] * (DS4_DSPARK_MAX_BLOCK + 1u) ||
      flags[7] > flags[5] * 1000u) {
    payload_set_err(err, errlen, "invalid DSpark controller state");
    return 1;
  }
  state.skip_remaining = flags[0];
  state.concurrent_width = flags[1];
  state.plain_only = flags[2] != 0;
  state.force_plain_request = flags[3] != 0;
  state.policy_concurrency = flags[4];
  state.probe_cycles = flags[5];
  state.probe_emitted = flags[6];
  state.probe_cost = flags[7];
  return 0;
}

static uint32_t session_dspark_live_rows(const ds4_gpu_graph* graph) {
  return std::min(graph->dspark_context_len, DS4_N_SWA);
}

static uint64_t layer_index_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio * sizeof(float);
}

/* Only the last logical sliding-window rows are needed from the raw cache.
 * The physical ROCm tensor is a ring sized for ubatches, but after restore
 * the next suffix chunk will write its own raw rows before any attention read.
 * Compressed rows are different: sparse attention can select any row from the
 * prefix, so those are persisted up to their live row counts. */
static uint32_t session_raw_live_rows(const ds4_gpu_graph *g, uint32_t checkpoint_len) {
    uint32_t rows = g->raw_window ? g->raw_window : DS4_N_SWA;
    if (rows > g->raw_cap) rows = g->raw_cap;
    if (rows > checkpoint_len) rows = checkpoint_len;
    return rows;
}

/* Return the exact engine-owned payload size, excluding the server's KVC file
 * header and observability text.  This is deliberately based on live row counts
 * rather than capacities so the disk cache scales with saved tokens, not with
 * the maximum context size used to allocate the graph. */
static uint64_t session_payload_live_tensor_bytes(const ds4_gpu_graph *g, uint32_t checkpoint_len) {
    uint64_t bytes = 0;
    const uint32_t raw_live = session_raw_live_rows(g, checkpoint_len);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        bytes += (uint64_t)raw_live * DS4_N_HEAD_DIM * sizeof(float);
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        bytes += (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM * sizeof(float);
        bytes += layer_attn_state_bytes(ratio);
        bytes += layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)g->layer_n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
            bytes += layer_index_state_bytes(ratio);
            bytes += layer_index_state_bytes(ratio);
        }
    }
    return bytes;
}

/* Accelerator tensors are copied through a fixed-size CPU buffer.  We do not mmap the
 * cache file and we do not allocate a second graph-sized blob just to serialize
 * it; both would be poor fits for this very large model. */
static int payload_write_tensor_span(FILE *fp, const ds4_gpu_tensor *tensor,
                                     uint64_t offset, uint64_t bytes,
                                     uint8_t *buf, size_t cap, char *err, size_t errlen) {
  if (!tensor || offset > ds4_gpu_tensor_bytes(tensor) ||
      bytes > ds4_gpu_tensor_bytes(tensor) - offset) {
    payload_set_err(err, errlen, "session tensor is smaller than the payload");
    return 1;
  }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (ds4_gpu_tensor_snapshot_read(tensor, offset + done, buf, n) == 0) {
          payload_set_err(err, errlen,
                          "failed to read accelerator session tensor");
          return 1;
        }
        if (payload_write_bytes(fp, buf, n, err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}

static int payload_read_tensor_span(FILE *fp, ds4_gpu_tensor *tensor,
                                    uint64_t offset, uint64_t bytes,
                                    uint8_t *buf, size_t cap, uint64_t *remaining,
                                    char *err, size_t errlen) {
  if (!tensor || offset > ds4_gpu_tensor_bytes(tensor) ||
      bytes > ds4_gpu_tensor_bytes(tensor) - offset) {
    payload_set_err(err, errlen, "session tensor is smaller than the payload");
    return 1;
  }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (payload_read_bytes(fp, buf, n, remaining, err, errlen) != 0) return 1;
        if (ds4_gpu_tensor_write(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to restore accelerator session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}

// A logical ring window has at most two contiguous spans. Transfer each span
// in bulk; copying thousands of individual rows dominates long-prefix reuse.
static int payload_ring_tensor(FILE* fp, ds4_gpu_tensor* tensor,
                               uint32_t capacity, uint32_t context,
                               uint32_t live, bool writing, uint8_t* buffer,
                               uint64_t* remaining, char* err, size_t errlen) {
  const uint32_t first = (context - live) % capacity;
  const uint32_t before_wrap = std::min(live, capacity - first);
  constexpr uint64_t row_bytes = DS4_N_HEAD_DIM * sizeof(float);
  const auto transfer = [&](uint64_t offset, uint64_t bytes) {
    return writing
               ? payload_write_tensor_span(fp, tensor, offset, bytes, buffer,
                                           DS4_SESSION_IO_CHUNK, err, errlen)
               : payload_read_tensor_span(fp, tensor, offset, bytes, buffer,
                                          DS4_SESSION_IO_CHUNK, remaining, err,
                                          errlen);
  };
  if (transfer(first * row_bytes, before_wrap * row_bytes))
    return 1;
  return live > before_wrap ? transfer(0, (live - before_wrap) * row_bytes) : 0;
}

// Only the semantic target window is visible to the next draft. Physical
// slots used for old context or uncommitted draft rows are not continuation
// state.
static int payload_dspark_tensors(FILE* fp, const ds4_gpu_graph* graph,
                                  uint32_t context, bool writing,
                                  uint8_t* buffer, uint64_t* remaining,
                                  char* err, size_t errlen) {
  if (!graph->dspark || context == 0)
    return 0;
  const uint32_t live = std::min(context, DS4_N_SWA);
  for (uint32_t stage = 0; stage < graph->dspark->n_stages; ++stage) {
    if (payload_ring_tensor(fp, graph->dspark_kv_cache[stage],
                            graph->dspark_cache_cap, context, live, writing,
                            buffer, remaining, err, errlen))
      return 1;
  }
  return 0;
}

uint64_t ds4_rocm_graph_snapshot_bytes(const ds4_rocm_graph *graph,
                                       const ds4_tokens *checkpoint) {
    if (!graph || !checkpoint || checkpoint->len <= 0) return 0;
    if (graph->dspark && graph->dspark_capture_enabled &&
        graph->dspark_context_len != static_cast<uint32_t>(checkpoint->len)) {
      return 0;  // A partial capture is not a complete continuation.
    }
    uint64_t bytes = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    bytes += dspark_snapshot_counters.size() * sizeof(uint64_t) +
             8 * sizeof(uint32_t);
    if (graph->dspark && graph->dspark_context_len != 0) {
      bytes += uint64_t(graph->dspark->n_stages) *
               session_dspark_live_rows(graph) * DS4_N_HEAD_DIM * sizeof(float);
    }
    bytes += (uint64_t)checkpoint->len * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_VOCAB * sizeof(float);
    bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
    bytes += session_payload_live_tensor_bytes(graph, (uint32_t)checkpoint->len);
    return bytes;
}

static int rocm_graph_save_payload(const ds4_rocm_graph* graph,
                                   const ds4_tokens* checkpoint,
                                   const float* logits,
                                   uint32_t prefill_capacity, int context_size,
                                   const ds4_dspark_request_state* state,
                                   FILE* fp, char* err, size_t errlen) {
  if (!graph || !checkpoint || checkpoint->len <= 0 || !logits || !fp) {
    payload_set_err(err, errlen, "invalid graph snapshot save");
    return 1;
  }
  // Successful session operations have already completed device writes.
  // The capture worker holds this session frozen while other sessions run.

  const uint32_t raw_live =
      session_raw_live_rows(graph, (uint32_t)checkpoint->len);
  /* Header fields:
   *   0 magic, 1 version, 2 ctx, 3 prefill chunk, 4 raw cap,
   *   5 raw window, 6 compressed cap, 7 token count,
   *   8 layers, 9 raw head dim, 10 indexer head dim, 11 vocab,
   *   12 live raw rows serialized below.
   */
  uint32_t header[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
      DS4_SESSION_PAYLOAD_MAGIC,
      DS4_SESSION_PAYLOAD_VERSION,
      (uint32_t)context_size,
      prefill_capacity,
      graph->raw_cap,
      graph->raw_window,
      graph->comp_cap,
      (uint32_t)checkpoint->len,
      DS4_N_LAYER,
      DS4_N_HEAD_DIM,
      DS4_N_INDEXER_HEAD_DIM,
      DS4_N_VOCAB,
      raw_live,
      graph->dspark ? graph->dspark->n_stages : 0u,
      graph->dspark ? graph->dspark->block_size : 0u,
      graph->dspark ? graph->dspark_cache_cap : 0u,
      graph->dspark_context_len,
      graph->dspark && graph->dspark_capture_enabled ? 1u : 0u,
      graph->dspark ? graph->dspark->n_target_layers : 0u,
  };
  if (graph->dspark) {
    std::copy_n(graph->dspark->target_layer_ids, graph->dspark->n_target_layers,
                header + 19);
  }
  for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
    if (payload_write_u32(fp, header[i], err, errlen) != 0)
      return 1;
  }
  if (payload_write_dspark_state(fp, *state, err, errlen))
    return 1;
  for (int i = 0; i < checkpoint->len; i++) {
    if (payload_write_u32(fp, (uint32_t)checkpoint->v[i], err, errlen) != 0) {
      return 1;
    }
  }
  if (payload_write_bytes(fp, logits, (uint64_t)DS4_N_VOCAB * sizeof(float),
                          err, errlen) != 0) {
    return 1;
  }
  for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
    if (payload_write_u32(fp, graph->layer_n_comp[il], err, errlen) != 0) {
      return 1;
    }
  }
  for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
    if (payload_write_u32(fp, graph->layer_n_index_comp[il], err, errlen) !=
        0) {
      return 1;
    }
  }

  auto* buf = static_cast<uint8_t*>(ds4_xmalloc(DS4_SESSION_IO_CHUNK));
  int rc = 0;
  for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
    /* Write the raw ring in logical position order.  The file does not care
     * where the rows happened to live physically in the source graph. */
    rc = payload_ring_tensor(fp, graph->layer_raw_cache[il], graph->raw_cap,
                             static_cast<uint32_t>(checkpoint->len), raw_live,
                             true, buf, nullptr, err, errlen);
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    if (rc != 0 || ratio == 0)
      continue;
    /* Compressed rows are append-only from row zero, so the live prefix is
     * contiguous.  The two compressor state tensors hold the partial window
     * that will become the next compressed row. */
    rc = payload_write_tensor_span(
        fp, graph->layer_attn_comp_cache[il], 0,
        (uint64_t)graph->layer_n_comp[il] * DS4_N_HEAD_DIM * sizeof(float), buf,
        DS4_SESSION_IO_CHUNK, err, errlen);
    if (rc == 0)
      rc = payload_write_tensor_span(fp, graph->layer_attn_state_kv[il], 0,
                                     layer_attn_state_bytes(ratio), buf,
                                     DS4_SESSION_IO_CHUNK, err, errlen);
    if (rc == 0)
      rc = payload_write_tensor_span(fp, graph->layer_attn_state_score[il], 0,
                                     layer_attn_state_bytes(ratio), buf,
                                     DS4_SESSION_IO_CHUNK, err, errlen);
    if (rc == 0 && ratio == 4) {
      rc = payload_write_tensor_span(fp, graph->layer_index_comp_cache[il], 0,
                                     (uint64_t)graph->layer_n_index_comp[il] *
                                         DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                     buf, DS4_SESSION_IO_CHUNK, err, errlen);
      if (rc == 0)
        rc = payload_write_tensor_span(fp, graph->layer_index_state_kv[il], 0,
                                       layer_index_state_bytes(ratio), buf,
                                       DS4_SESSION_IO_CHUNK, err, errlen);
      if (rc == 0)
        rc = payload_write_tensor_span(fp, graph->layer_index_state_score[il],
                                       0, layer_index_state_bytes(ratio), buf,
                                       DS4_SESSION_IO_CHUNK, err, errlen);
    }
  }
  if (rc == 0)
    rc = payload_dspark_tensors(fp, graph, graph->dspark_context_len, true, buf,
                                nullptr, err, errlen);
  free(buf);
  return rc;
}

static int rocm_graph_load_payload(ds4_rocm_graph* graph,
                                   ds4_tokens* checkpoint, float* logits,
                                   uint32_t prefill_capacity, int context_size,
                                   ds4_dspark_request_state* state, FILE* fp,
                                   uint64_t payload_bytes, char* err,
                                   size_t errlen) {
  if (!graph || !checkpoint || !logits || !fp) {
    payload_set_err(err, errlen, "invalid graph snapshot load");
    return 1;
  }
  uint64_t remaining = payload_bytes;
  uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
  for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
    if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0)
      return 1;
  }
  if (h[0] != DS4_SESSION_PAYLOAD_MAGIC ||
      h[1] != DS4_SESSION_PAYLOAD_VERSION) {
    payload_set_err(err, errlen, "unsupported session payload version");
    return 1;
  }
  const uint32_t saved_ctx = h[2];
  const uint32_t saved_prefill_cap = h[3];
  const uint32_t saved_raw_cap = h[4];
  const uint32_t saved_raw_window = h[5];
  const uint32_t saved_comp_cap = h[6];
  const uint32_t saved_tokens = h[7];
  const uint32_t saved_raw_live = h[12];
  if (saved_ctx > (uint32_t)context_size ||
      saved_tokens >= (uint32_t)context_size) {
    payload_set_err(err, errlen, "KV checkpoint does not fit current context");
    return 1;
  }
  if (h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_DIM ||
      h[10] != DS4_N_INDEXER_HEAD_DIM || h[11] != DS4_N_VOCAB) {
    payload_set_err(err, errlen,
                    "KV checkpoint was written for a different DS4 layout");
    return 1;
  }
  /* prefill_cap is scratch scheduling capacity, not durable KV layout.
   * Old checkpoints remain valid as long as the raw KV window matches. */
  (void)saved_prefill_cap;
  (void)prefill_capacity;
  if (saved_raw_window != graph->raw_window) {
    payload_set_err(
        err, errlen,
        "KV checkpoint graph chunk layout does not match current runtime");
    return 1;
  }
  /* The raw rows in the file are logical rows.  We can restore them into any
   * current ring with enough capacity, but the saved live count must be exactly
   * the last window implied by the saved token count. */
  const uint32_t expected_raw_live =
      saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
  if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
      saved_raw_live > saved_raw_cap || saved_raw_live > graph->raw_cap) {
    payload_set_err(
        err, errlen,
        "KV checkpoint raw ring layout does not match current context");
    return 1;
  }
  if (saved_comp_cap > graph->comp_cap) {
    payload_set_err(
        err, errlen,
        "KV checkpoint compressed cache is larger than current context");
    return 1;
  }

  const uint32_t stages = graph->dspark ? graph->dspark->n_stages : 0u;
  const uint32_t block = graph->dspark ? graph->dspark->block_size : 0u;
  const uint32_t targets = graph->dspark ? graph->dspark->n_target_layers : 0u;
  if (saved_tokens == 0 || h[13] != stages || h[14] != block ||
      h[15] != (graph->dspark ? graph->dspark_cache_cap : 0u) ||
      h[16] > saved_tokens || h[17] > 1u || h[18] != targets ||
      (h[17] != 0u && h[16] != saved_tokens) || (h[17] == 0u && h[16] != 0u)) {
    payload_set_err(err, errlen,
                    "incompatible DSpark snapshot layout or coverage");
    return 1;
  }
  for (uint32_t i = 0; i < DS4_DSPARK_MAX_TARGET_LAYERS; ++i) {
    const uint32_t layer =
        i < targets ? graph->dspark->target_layer_ids[i] : 0u;
    if (h[19 + i] != layer) {
      payload_set_err(err, errlen, "incompatible DSpark target feature layout");
      return 1;
    }
  }
  ds4_dspark_request_state restored_state{};
  if (payload_read_dspark_state(fp, restored_state, &remaining, err, errlen))
    return 1;
  if (graph->dspark && ((h[17] != 0u) == restored_state.force_plain_request)) {
    payload_set_err(err, errlen,
                    "inconsistent DSpark capture and request state");
    return 1;
  }

  token_vec new_checkpoint = {0};
  for (uint32_t i = 0; i < saved_tokens; i++) {
    uint32_t tok = 0;
    if (payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
      ds4_tokens_free(&new_checkpoint);
      return 1;
    }
    if (tok >= DS4_N_VOCAB) {
      ds4_tokens_free(&new_checkpoint);
      payload_set_err(err, errlen, "invalid checkpoint token ID");
      return 1;
    }
    ds4_tokens_push(&new_checkpoint, (int)tok);
  }
  if (payload_read_bytes(fp, logits, (uint64_t)DS4_N_VOCAB * sizeof(float),
                         &remaining, err, errlen) != 0) {
    ds4_tokens_free(&new_checkpoint);
    return 1;
  }
  if (!std::all_of(logits, logits + DS4_N_VOCAB,
                   [](float value) { return std::isfinite(value); })) {
    ds4_tokens_free(&new_checkpoint);
    payload_set_err(err, errlen, "checkpoint contains non-finite logits");
    return 1;
  }
  uint32_t n_comp[DS4_MAX_LAYER];
  uint32_t n_index_comp[DS4_MAX_LAYER];
  for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
    if (payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0) {
      ds4_tokens_free(&new_checkpoint);
      return 1;
    }
    if (n_comp[il] > saved_comp_cap || n_comp[il] > graph->layer_comp_cap[il]) {
      ds4_tokens_free(&new_checkpoint);
      payload_set_err(err, errlen,
                      "KV checkpoint has invalid compressed row count");
      return 1;
    }
  }
  for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
    if (payload_read_u32(fp, &n_index_comp[il], &remaining, err, errlen) != 0) {
      ds4_tokens_free(&new_checkpoint);
      return 1;
    }
    if (n_index_comp[il] > saved_comp_cap ||
        n_index_comp[il] > graph->layer_comp_cap[il]) {
      ds4_tokens_free(&new_checkpoint);
      payload_set_err(err, errlen,
                      "KV checkpoint has invalid indexer row count");
      return 1;
    }
  }

  if (ds4_gpu_synchronize() == 0) {
    ds4_tokens_free(&new_checkpoint);
    payload_set_err(err, errlen,
                    "failed to synchronize accelerator before KV restore");
    return 1;
  }
  for (uint32_t il = 0; il < DS4_N_LAYER; ++il) {
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    if (ratio == 0)
      continue;
    const uint64_t required =
        static_cast<uint64_t>(std::max(n_comp[il], n_index_comp[il])) +
        (DS4_N_SWA + ratio - 1) / ratio + 2;
    if (!rocm_graph_reserve_compressed_rows(
            graph, il,
            static_cast<uint32_t>(
                std::min<uint64_t>(graph->layer_comp_cap[il], required)))) {
      ds4_tokens_free(&new_checkpoint);
      payload_set_err(err, errlen,
                      "failed to reserve compressed cache for KV restore");
      return 1;
    }
  }
  auto* buf = static_cast<uint8_t*>(ds4_xmalloc(DS4_SESSION_IO_CHUNK));
  int rc = 0;
  for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
    /* Rebuild the physical raw ring expected by the current graph.  This is
     * why the file stores rows in logical order instead of dumping bytes from
     * the old ring layout. */
    rc = payload_ring_tensor(fp, graph->layer_raw_cache[il], graph->raw_cap,
                             saved_tokens, saved_raw_live, false, buf,
                             &remaining, err, errlen);
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    if (rc != 0 || ratio == 0)
      continue;
    rc = payload_read_tensor_span(
        fp, graph->layer_attn_comp_cache[il], 0,
        (uint64_t)n_comp[il] * DS4_N_HEAD_DIM * sizeof(float), buf,
        DS4_SESSION_IO_CHUNK, &remaining, err, errlen);
    if (rc == 0 && ratio == 4 &&
        !rocm_graph_rebuild_attn_comp_mirror(graph, il, n_comp[il])) {
      payload_set_err(err, errlen,
                      "failed to rebuild FP16 attention cache mirror");
      rc = 1;
    }
    if (rc == 0)
      rc = payload_read_tensor_span(
          fp, graph->layer_attn_state_kv[il], 0, layer_attn_state_bytes(ratio),
          buf, DS4_SESSION_IO_CHUNK, &remaining, err, errlen);
    if (rc == 0)
      rc = payload_read_tensor_span(fp, graph->layer_attn_state_score[il], 0,
                                    layer_attn_state_bytes(ratio), buf,
                                    DS4_SESSION_IO_CHUNK, &remaining, err,
                                    errlen);
    if (rc == 0 && ratio == 4) {
      rc = payload_read_tensor_span(
          fp, graph->layer_index_comp_cache[il], 0,
          (uint64_t)n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
          buf, DS4_SESSION_IO_CHUNK, &remaining, err, errlen);
      if (rc == 0)
        rc = payload_read_tensor_span(fp, graph->layer_index_state_kv[il], 0,
                                      layer_index_state_bytes(ratio), buf,
                                      DS4_SESSION_IO_CHUNK, &remaining, err,
                                      errlen);
      if (rc == 0)
        rc = payload_read_tensor_span(fp, graph->layer_index_state_score[il], 0,
                                      layer_index_state_bytes(ratio), buf,
                                      DS4_SESSION_IO_CHUNK, &remaining, err,
                                      errlen);
    }
  }
  if (rc == 0)
    rc = payload_dspark_tensors(fp, graph, h[16], false, buf, &remaining, err,
                                errlen);
  free(buf);
  if (rc != 0) {
    ds4_tokens_free(&new_checkpoint);
    return 1;
  }
  if (remaining != 0) {
    ds4_tokens_free(&new_checkpoint);
    payload_set_err(err, errlen, "KV checkpoint has trailing payload bytes");
    return 1;
  }
  if (ds4_gpu_synchronize() == 0) {
    ds4_tokens_free(&new_checkpoint);
    payload_set_err(err, errlen,
                    "failed to synchronize accelerator after KV restore");
    return 1;
  }

  ds4_tokens_free(checkpoint);
  *checkpoint = new_checkpoint;
  for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
    graph->layer_n_comp[il] = n_comp[il];
    graph->layer_n_index_comp[il] = n_index_comp[il];
  }
  *state = restored_state;
  graph->dspark_capture_enabled = h[17] != 0u;
  graph->dspark_context_len = h[16];
  graph->dspark_capture_mask = 0;
  graph->dspark_capture_batch_mask = 0;
  graph->dspark_capture_batch_start = 0;
  graph->dspark_capture_batch_tokens = 0;
  return 0;
}

void* ds4_rocm_graph_continuation_tensor(const ds4_rocm_graph* graph,
                                         ds4_continuation_tensor tensor,
                                         uint32_t index) {
  const ds4_gpu_tensor* selected = nullptr;
  switch (tensor) {
    case ds4_continuation_tensor::kRaw:
      selected = graph->layer_raw_cache[index];
      break;
    case ds4_continuation_tensor::kAttentionRows:
      selected = graph->layer_attn_comp_cache[index];
      break;
    case ds4_continuation_tensor::kAttentionValues:
      selected = graph->layer_attn_state_kv[index];
      break;
    case ds4_continuation_tensor::kAttentionScores:
      selected = graph->layer_attn_state_score[index];
      break;
    case ds4_continuation_tensor::kIndexerRows:
      selected = graph->layer_index_comp_cache[index];
      break;
    case ds4_continuation_tensor::kIndexerValues:
      selected = graph->layer_index_state_kv[index];
      break;
    case ds4_continuation_tensor::kIndexerScores:
      selected = graph->layer_index_state_score[index];
      break;
    case ds4_continuation_tensor::kDraftWindow:
      selected = graph->dspark_kv_cache[index];
      break;
  }
  return ds4_gpu_tensor_device_pointer(selected);
}

ds4_continuation_ring ds4_rocm_graph_raw_ring(const ds4_rocm_graph* graph,
                                              uint32_t position) {
  return {graph->raw_cap, session_raw_live_rows(graph, position)};
}

ds4_continuation_ring ds4_rocm_graph_draft_ring(const ds4_rocm_graph* graph,
                                                uint32_t position) {
  return {graph->dspark_cache_cap, std::min(position, DS4_N_SWA)};
}

bool ds4_rocm_graph_draft_capture(const ds4_rocm_graph* graph) {
  return graph->dspark && graph->dspark_capture_enabled;
}

bool ds4_rocm_graph_rows_at(const ds4_rocm_graph* graph, uint32_t position) {
  for (uint32_t il = 0; il < DS4_N_LAYER; ++il) {
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    if (ratio != 0 &&
        (graph->layer_n_comp[il] != position / ratio ||
         (ratio == 4 && graph->layer_n_index_comp[il] != position / ratio)))
      return false;
  }
  return true;
}

// The same headroom as a snapshot restore, so the next chunk does not grow.
bool ds4_rocm_graph_reserve_rows(ds4_rocm_graph* graph, uint32_t position) {
  if (ds4_gpu_synchronize() == 0)
    return false;
  for (uint32_t il = 0; il < DS4_N_LAYER; ++il) {
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    if (ratio == 0)
      continue;
    const uint64_t required =
        static_cast<uint64_t>(position / ratio) +
        (DS4_N_SWA + ratio - 1) / ratio + 2;
    if (position / ratio > graph->layer_comp_cap[il] ||
        !rocm_graph_reserve_compressed_rows(
            graph, il,
            static_cast<uint32_t>(
                std::min<uint64_t>(graph->layer_comp_cap[il], required))))
      return false;
  }
  return true;
}

bool ds4_rocm_graph_commit_rows(ds4_rocm_graph* graph, uint32_t position,
                                uint32_t draft_tokens, bool draft_capture) {
  for (uint32_t il = 0; il < DS4_N_LAYER; ++il) {
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    graph->layer_n_comp[il] = ratio == 0 ? 0 : position / ratio;
    graph->layer_n_index_comp[il] = ratio == 4 ? position / ratio : 0;
    if (ratio == 4 && !rocm_graph_rebuild_attn_comp_mirror(
                          graph, il, graph->layer_n_comp[il]))
      return false;
  }
  if (ds4_gpu_synchronize() == 0)
    return false;
  graph->dspark_capture_enabled = draft_capture;
  graph->dspark_context_len = draft_tokens;
  graph->dspark_capture_mask = 0;
  graph->dspark_capture_batch_mask = 0;
  graph->dspark_capture_batch_start = 0;
  graph->dspark_capture_batch_tokens = 0;
  return true;
}

int ds4_rocm_graph_save_snapshot(const ds4_rocm_graph* graph,
                                 const ds4_tokens* checkpoint,
                                 const float* logits, uint32_t prefill_capacity,
                                 int context_size,
                                 const ds4_dspark_request_state* state,
                                 ds4_session_snapshot* snap, char* err,
                                 size_t errlen) {
  if (!graph || !checkpoint || !logits || !snap) {
    payload_set_err(err, errlen, "invalid graph snapshot save");
    return 1;
  }
  const uint64_t bytes = ds4_rocm_graph_snapshot_bytes(graph, checkpoint);
  if (bytes == 0) {
    payload_set_err(err, errlen, "session has no valid checkpoint to snapshot");
    return 1;
  }
  if (bytes >= (uint64_t)SIZE_MAX) {
    payload_set_err(err, errlen,
                    "session snapshot is too large for this platform");
    return 1;
  }
  // fmemopen appends a terminator on close, including in binary mode.
  // Keep that byte outside the serialized payload.
  if (snap->cap < bytes + 1u) {
    auto* p = static_cast<uint8_t*>(
        realloc(snap->ptr, static_cast<size_t>(bytes + 1u)));
    if (!p) {
      payload_set_err(err, errlen,
                      "out of memory while allocating session snapshot");
      return 1;
    }
    snap->ptr = p;
    snap->cap = bytes + 1u;
  }

  FILE* fp = gufo::platform::OpenMemoryWriteStream(
      snap->ptr, static_cast<size_t>(bytes + 1u));
  if (!fp) {
    payload_set_err(err, errlen,
                    "failed to open memory stream for session snapshot");
    return 1;
  }
  const int rc =
      rocm_graph_save_payload(graph, checkpoint, logits, prefill_capacity,
                              context_size, state, fp, err, errlen);
  if (!gufo::platform::FinishMemoryWriteStream(fp, snap->ptr,
                                               static_cast<size_t>(bytes)) &&
      rc == 0) {
    payload_set_err(err, errlen, "failed to finalize memory session snapshot");
    return 1;
  }
  if (rc != 0)
    return 1;
  snap->len = bytes;
  return 0;
}

int ds4_rocm_graph_load_snapshot(ds4_rocm_graph* graph, ds4_tokens* checkpoint,
                                 float* logits, uint32_t prefill_capacity,
                                 int context_size,
                                 ds4_dspark_request_state* state,
                                 const ds4_session_snapshot* snap, char* err,
                                 size_t errlen) {
  if (!graph || !checkpoint || !logits || !snap || !snap->ptr ||
      snap->len == 0) {
    payload_set_err(err, errlen, "invalid graph snapshot load");
    return 1;
  }
    if (snap->len > (uint64_t)SIZE_MAX) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }

    FILE *fp = gufo::platform::OpenMemoryReadStream(snap->ptr, (size_t)snap->len);
    if (!fp) {
        payload_set_err(err, errlen, "failed to open memory stream for session snapshot restore");
        return 1;
    }
    const int rc = rocm_graph_load_payload(graph, checkpoint, logits,
                                           prefill_capacity, context_size,
                                           state, fp, snap->len, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close memory session snapshot");
        return 1;
    }
    return rc;
}
