// ds4_rocm_engram.hip.hpp — V4.1 Engram conditioned memory (D3).
//
// One Engram module = PLE n-gram embedding table (IQ4_NL rows, host-resident,
// gathered on demand — never uploaded) + engram_kv projection (Q5_K,
// [ (hc+1)*n_embd, hc*n_embd ]) + engram_gate (F32, [hc*n_embd]).
//
// Semantics mirror llama.cpp b11870 src/models/deepseek41.cpp::build_engram:
//   rows   = table[ids[i,c]]                      (per (token, head) row, 256 f32)
//   kv     = rows @ engram_kv^T                   ((hc+1)*n_embd per token)
//   out[h] += value * sigmoid(copysign(sqrt(max(|dot_h|,1e-6)), dot_h))
//   dot_h  = (1/sqrt(n_embd)) * < rms_norm(x_h), rms_norm(key_h) * gate_h >
// rms uses the artifact eps (deepseek41.attention.layer_norm_rms_epsilon).
// Weights are read AS-IS from the mmap; no converted copies exist.
#ifndef DS4_ROCM_ENGRAM_HIP_INCLUDED
#define DS4_ROCM_ENGRAM_HIP_INCLUDED

#include "../mmq/ggml-common.h"

namespace {

// ids [nt*n_cols] int32 (GGUF token ids); table = mmap base of
// per_layer_token_embd (IQ4_NL rows: 8 blocks x 18 B = 144 B per 256-elem
// row; QK4_NL = 32). rows [nt*n_cols*256] f32.
__global__ void ds4_engram_gather_kernel(const int32_t *__restrict__ ids,
                                         const char *__restrict__ table,
                                         float *__restrict__ rows,
                                         uint32_t n_rows) {
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    if (warp >= n_rows)
        return;
    const int32_t id = ids[warp];
    // One warp covers the row's 8 IQ4_NL sub-blocks; 4 lanes per sub-block,
    // each lane 4 nibble bytes -> 8 floats (low nibble -> elem m, high ->
    // elem m + QK4_NL/2, per dequantize_row_iq4_nl).
    const block_iq4_nl *blk =
        (const block_iq4_nl *)(table +
                               (uint64_t)id * (8u * sizeof(block_iq4_nl))) +
        (lane >> 2);
    const float d = __half2float(blk->d);
    const uint32_t j = (lane & 3u) * 4u;
    float *olo = rows + (uint64_t)warp * QK_K + (uint64_t)(lane >> 2) * QK4_NL +
                 (lane & 3u) * 4u;
    float *ohi = olo + QK4_NL / 2;
#pragma unroll
    for (uint32_t k = 0; k < 4u; ++k) {
        const uint8_t byte = blk->qs[j + k];
        olo[k] = d * (float)kvalues_iq4nl[byte & 0xf];
        ohi[k] = d * (float)kvalues_iq4nl[byte >> 4];
    }
}

// kv[tok][orow] = dot(rows[tok][6144], engram_kv[orow]) — one warp per output
// row (gridDim.y = token), the 24 Q5_K sub-blocks split across lanes 0..23.
__global__ void ds4_engram_kv_kernel(const float *__restrict__ rows,
                                     const block_q5_K *__restrict__ kw,
                                     float *__restrict__ kv,
                                     uint32_t out_dim, uint32_t in_dim,
                                     uint32_t k_blocks) {
    const uint32_t orow = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (orow >= out_dim)
        return;
    const uint32_t lane = threadIdx.x & 31u;
    float acc = 0.0f;
    if (lane < k_blocks)
        acc = ds4_q5k_dot_block(kw + ((uint64_t)orow * k_blocks) + lane,
                                rows + (uint64_t)blockIdx.y * in_dim +
                                    (uint64_t)lane * QK_K);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(static_cast<MASK_T>(~MASK_T(0)), acc, off);
    if (lane == 0u)
        kv[(uint64_t)blockIdx.y * out_dim + orow] = acc;
}

// Block per token: compute the hc sign-sqrt gates and add value*gate in place.
__global__ void ds4_engram_gate_kernel(float *__restrict__ hc,
                                       const float *__restrict__ kv,
                                       const float *__restrict__ egate,
                                       uint32_t n_embd, uint32_t hc_mult,
                                       uint64_t hc_stride, uint64_t kv_stride,
                                       float eps) {
    float *x = hc + (uint64_t)blockIdx.x * hc_stride;
    const float *k = kv + (uint64_t)blockIdx.x * kv_stride;
    const float *val = k + (uint64_t)hc_mult * n_embd;
    const uint32_t tid = threadIdx.x;
    constexpr uint32_t NT = 256u;
    __shared__ float sg[4];

    float a1[4] = {0, 0, 0, 0}, a2[4] = {0, 0, 0, 0}, a3[4] = {0, 0, 0, 0};
    for (uint32_t d = tid; d < n_embd; d += NT) {
        for (uint32_t h = 0; h < hc_mult; ++h) {
            const float xv = x[(uint64_t)h * n_embd + d];
            const float kvv = k[(uint64_t)h * n_embd + d];
            a1[h] += xv * xv;
            a2[h] += kvv * kvv;
            a3[h] += xv * kvv * egate[(uint64_t)h * n_embd + d];
        }
    }
    // Warp-reduce each h, keep per-warp lane-0 partials in shared memory, then
    // sum the 8 warp partials per h and broadcast the final gate to the block.
    constexpr uint32_t NW = NT / 32u;
    __shared__ float w1[NW][4], w2[NW][4], w3[NW][4];
    const uint32_t wid = tid >> 5u;
#pragma unroll
    for (uint32_t h = 0; h < 4; ++h) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            a1[h] += __shfl_down_sync(static_cast<MASK_T>(~MASK_T(0)), a1[h], off);
            a2[h] += __shfl_down_sync(static_cast<MASK_T>(~MASK_T(0)), a2[h], off);
            a3[h] += __shfl_down_sync(static_cast<MASK_T>(~MASK_T(0)), a3[h], off);
        }
        if ((tid & 31u) == 0u) {
            w1[wid][h] = a1[h];
            w2[wid][h] = a2[h];
            w3[wid][h] = a3[h];
        }
    }
    __syncthreads();
    if (tid == 0u) {
        const float inv = 1.0f / sqrtf((float)n_embd);
        for (uint32_t h = 0; h < hc_mult; ++h) {
            float t1 = 0.f, t2 = 0.f, t3 = 0.f;
#pragma unroll
            for (uint32_t w = 0; w < NW; ++w) {
                t1 += w1[w][h];
                t2 += w2[w][h];
                t3 += w3[w][h];
            }
            const float rs1 = rsqrtf(t1 * inv * inv + eps);
            const float rs2 = rsqrtf(t2 * inv * inv + eps);
            const float dot = t3 * rs1 * rs2 * inv;
            const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
            sg[h] = 1.0f / (1.0f + expf(-copysignf(mag, dot)));
        }
    }
    __syncthreads();
    for (uint32_t d = tid; d < n_embd; d += NT) {
        const float vv = val[d];
        for (uint32_t h = 0; h < hc_mult; ++h)
            x[(uint64_t)h * n_embd + d] += vv * sg[h];
    }
}

} // namespace

int ds4_gpu_engram_gather(const void *table_ptr, const ds4_gpu_tensor *ids,
                          ds4_gpu_tensor *rows, uint32_t n_rows) {
    const unsigned threads = 256u; // 8 warps: 8 rows/block
    const unsigned blocks = (n_rows + 7u) / 8u;
    ds4_engram_gather_kernel<<<blocks, threads>>>(
        (const int32_t *)ids->ptr, (const char *)table_ptr,
        (float *)rows->ptr, n_rows);
    return hipGetLastError() == hipSuccess ? 1 : 0;
}

int ds4_gpu_engram_kv(const ds4_gpu_tensor *rows, const void *kv_weight_ptr,
                      ds4_gpu_tensor *kv, uint32_t nt, uint32_t out_dim,
                      uint32_t in_dim, uint32_t k_blocks) {
    const unsigned warps = 8u;
    const unsigned blocks = (out_dim + warps - 1u) / warps;
    dim3 grid(blocks, nt);
    ds4_engram_kv_kernel<<<grid, warps * 32u>>>(
        (const float *)rows->ptr, (const block_q5_K *)kv_weight_ptr,
        (float *)kv->ptr, out_dim, in_dim, k_blocks);
    return hipGetLastError() == hipSuccess ? 1 : 0;
}

int ds4_gpu_engram_gate(ds4_gpu_tensor *hc, const ds4_gpu_tensor *kv,
                        const void *egate_ptr, uint32_t nt, uint32_t n_embd,
                        uint32_t hc_mult, uint64_t hc_stride,
                        uint64_t kv_stride, float eps) {
    ds4_engram_gate_kernel<<<nt, 256u>>>(
        (float *)hc->ptr, (const float *)kv->ptr,
        (const float *)egate_ptr, n_embd, hc_mult, hc_stride, kv_stride, eps);
    return hipGetLastError() == hipSuccess ? 1 : 0;
}

#endif // DS4_ROCM_ENGRAM_HIP_INCLUDED
