/* Generic IQ-family routed-MoE route (correctness tier).
 *
 * GSQ artifacts pick the expert quant type per tensor (iq2_xxs, iq2_xs and
 * iq1_m all appear across layers), which the fixed q4k/iq2/q2k plan paths
 * cannot express.  This route reads the weight tensors AS IS — no dequant
 * copies, no activation requantization — and multiplies them straight against
 * the fp32 activations after an in-register per-group dequantization.  The
 * group/scale/sign unpacking mirrors dequantize_row_iq2_xxs / _iq2_xs /
 * _iq1_m from the reference quant implementation, over the same GGML block
 * tables the mmq tier already ships (iq2xxs_grid, iq2xs_grid, iq1s_grid_gpu,
 * ksigns_iq2xs, kmask_iq2xs), so the math matches the file format by
 * construction.
 */
#include "../mmq/ggml-common.h"

#define DS4_MOE_IQ_IQ2_XXS 16u
#define DS4_MOE_IQ_IQ2_XS 17u
#define DS4_MOE_IQ_IQ1_M 29u

static __device__ __forceinline__ float ds4_iq_warp_sum(float v) {
    const MASK_T mask = static_cast<MASK_T>(0xffffffffffffffffull);
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

/* Dequantize 8 elements (group g of a 256-element block) into v[]. */
static __device__ __forceinline__ void ds4_iq_deq8(uint32_t type,
                                                   const char *blk, int g,
                                                   float *v) {
    if (type == DS4_MOE_IQ_IQ2_XXS) {
        const block_iq2_xxs *b = (const block_iq2_xxs *)blk;
        const float d = __half2float(b->d);
        const int ib = g >> 2, l = g & 3;
        /* Mirror dequantize_row_iq2_xxs: each 32-element block is two
         * CONSECUTIVE little-endian words — aux0 holds the four grid-code
         * bytes aux8[l], aux1 holds the scale nibble (bits 28..31) plus the
         * four 7-bit sign indices (bits 7*l). */
        const uint32_t aux0 = ((const uint32_t *)b->qs)[2 * ib];
        const uint32_t aux1 = ((const uint32_t *)b->qs)[2 * ib + 1];
        const float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
        const uint64_t grid = iq2xxs_grid[(aux0 >> (8 * l)) & 0xffu];
        const uint8_t signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 0x7fu];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float e = (float)((grid >> (8 * j)) & 0xffu);
            v[j] = db * e * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
        }
    } else if (type == DS4_MOE_IQ_IQ2_XS) {
        const block_iq2_xs *b = (const block_iq2_xs *)blk;
        const float d = __half2float(b->d);
        const int ib = g >> 2, l = g & 3;
        /* Mirror dequantize_row_iq2_xs: qs[4*ib+l] = grid index (low 9 bits)
         * | sign-table index (bits 9..15); scales[ib] carries TWO 4-bit
         * sub-block scales — low nibble for l<2, high nibble for l>=2. */
        const uint16_t code = b->qs[4 * ib + l];
        const uint8_t scb = b->scales[ib];
        const float db =
            d * (0.5f + (float)(l < 2 ? (scb & 0xfu) : (scb >> 4))) * 0.25f;
        const uint64_t grid = iq2xs_grid[code & 511u];
        const uint8_t signs = ksigns_iq2xs[code >> 9];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float e = (float)((grid >> (8 * j)) & 0xffu);
            v[j] = db * e * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
        }
    } else {
        const block_iq1_m *b = (const block_iq1_m *)blk;
        const uint16_t *sc = (const uint16_t *)b->scales;
        iq1m_scale_t s;
        s.u16 = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                           ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
        const float d = __half2float(s.f16);
        const int ib = g >> 2, l = g & 3;
        const float dl = d * (2.0f * (float)((sc[ib >> 1] >> (6 * (ib & 1) + ((l < 2) ? 0 : 3))) & 0x7) + 1.0f);
        const uint8_t qhb = b->qh[2 * ib + (l >> 1)];
        const uint32_t qhl = (uint32_t)qhb >> (4 * (l & 1));
        const uint32_t idx = (uint32_t)b->qs[4 * ib + l] | ((qhl & 7u) << 8);
        const float delta = -1.0f + IQ1M_DELTA - (float)(qhl & 8u) * (2.0f * IQ1M_DELTA / 8.0f);
        const uint32_t grid = iq1s_grid_gpu[idx];
        /* gpu grid word packs 8 values: bytes 0..3 hold elements 0..3 in the
         * low nibble and elements 4..7 in the high nibble (values are 0..2). */
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t byte = (grid >> (8 * j)) & 0xffu;
            v[j] = dl * (float)(byte & 0xfu) + dl * delta;
            v[j + 4] = dl * (float)(byte >> 4) + dl * delta;
        }
    }
}

/* gate/up rows for every (token, slot) pair; swiglu and router weight baked
 * into mid.  grid = (ceil(mid_dim/8), pairs), block = 256 (8 warps).
 * Each warp owns one mid row; lane g dequantizes group g of each k block. */
__global__ static void moe_gate_up_mid_iq_kernel(
        float *__restrict__ mid_out, const char *__restrict__ gate_base,
        const char *__restrict__ up_base, const float *__restrict__ x,
        const int32_t *__restrict__ selected, const float *__restrict__ weights,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint32_t gate_type,
        uint32_t in_dim, uint32_t expert_mid_dim, uint32_t n_expert,
        float clamp) {
    const uint32_t pair = blockIdx.y;
    const uint32_t tok = pair / n_expert;
    const uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint64_t slab = (uint64_t)(uint32_t)expert_i * gate_expert_bytes;
    const uint32_t k_blocks = in_dim / QK_K;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const int lane = threadIdx.x & 31;
    for (uint32_t r = 0; r < rows_per_block; ++r) {
        const uint32_t row = blockIdx.x * rows_per_block + r;
        if (row >= expert_mid_dim) return;
        const char *gp = gate_base + slab + (uint64_t)row * gate_row_bytes;
        const char *up = up_base + slab + (uint64_t)row * gate_row_bytes;
        const float *xrow = x + (uint64_t)tok * in_dim;
        /* Quant blocks are far smaller than QK_K bytes (IQ1_M = 56, IQ2_XS
         * = 42, IQ2_XXS = 34): walk the row by real block size, not by 256. */
        const size_t blk_bytes = (size_t)(gate_row_bytes / k_blocks);
        float sumg = 0.f, sumu = 0.f;
        for (uint32_t kb = 0; kb < k_blocks; ++kb) {
            float wq[8], xv[8];
            const int base = (int)(kb * QK_K + lane * 8);
            ds4_iq_deq8(gate_type, gp + kb * blk_bytes, lane, wq);
#pragma unroll
            for (int j = 0; j < 8; ++j) xv[j] = xrow[base + j];
#pragma unroll
            for (int j = 0; j < 8; ++j) sumg = fmaf(wq[j], xv[j], sumg);
            ds4_iq_deq8(gate_type, up + kb * blk_bytes, lane, wq);
#pragma unroll
            for (int j = 0; j < 8; ++j) sumu = fmaf(wq[j], xv[j], sumu);
        }
        sumg = ds4_iq_warp_sum(sumg);
        sumu = ds4_iq_warp_sum(sumu);
        if (lane == 0) {
            float gate = sumg, upv = sumu;
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (upv > clamp) upv = clamp;
                if (upv < -clamp) upv = -clamp;
            }
            const float silu = gate / (1.0f + expf(-gate));
            mid_out[(uint64_t)pair * expert_mid_dim + row] =
                silu * upv * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

/* down projection summed over the slots, per token.
 * grid = (ceil(out_dim/8), n_tokens), block = 256 (8 warps). */
__global__ static void moe_down_sum_iq_kernel(
        float *__restrict__ out, const char *__restrict__ down_base,
        const float *__restrict__ mid, const int32_t *__restrict__ selected,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t down_type, uint32_t mid_dim, uint32_t out_dim,
        uint32_t n_expert) {
    const uint32_t tok = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t col = blockIdx.x * 8u + warp;
    if (col >= out_dim) return;
    const uint32_t k_blocks = mid_dim / QK_K;
    float acc = 0.f;
    for (uint32_t slot = 0; slot < n_expert; ++slot) {
        int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
        if (expert_i < 0) continue;
        const char *dp = down_base +
                         (uint64_t)(uint32_t)expert_i * down_expert_bytes +
                         (uint64_t)col * down_row_bytes;
        const float *mrow = mid + ((uint64_t)tok * n_expert + slot) * mid_dim;
        const size_t blk_bytes = (size_t)(down_row_bytes / k_blocks);
        float part = 0.f;
        for (uint32_t kb = 0; kb < k_blocks; ++kb) {
            float wq[8], mv[8];
            const int base = (int)(kb * QK_K + lane * 8);
            ds4_iq_deq8(down_type, dp + kb * blk_bytes, lane, wq);
#pragma unroll
            for (int j = 0; j < 8; ++j) mv[j] = mrow[base + j];
#pragma unroll
            for (int j = 0; j < 8; ++j) part = fmaf(wq[j], mv[j], part);
        }
        acc += ds4_iq_warp_sum(part);
    }
    if (lane == 0) out[(uint64_t)tok * out_dim + col] = acc;
}
