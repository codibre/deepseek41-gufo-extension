extern "C" int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight) {
    if (!hip_tensor_has_f32(out, n) || !hip_tensor_has_f32(gate, n) || !hip_tensor_has_f32(up, n)) return 0;
    if (n == 0u) return 1;
    swiglu_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, n, clamp, weight);
    return hip_ok(hipGetLastError(), "swiglu launch");
}
extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
    ds4_gpu_tensor* gate, ds4_gpu_tensor* up, ds4_gpu_tensor* mid,
    ds4_gpu_tensor* mid_q8,
    const void* model_map, uint64_t model_size, uint64_t gate_offset,
    uint64_t up_offset, uint64_t in_dim, uint64_t out_dim,
    const ds4_gpu_tensor* x, const ds4_gpu_tensor* xq8, float clamp,
    uint32_t n_tok) {
  if (!gate || !up || !mid || !model_map || !x || in_dim == 0u ||
      out_dim == 0u || n_tok == 0u || out_dim > UINT32_MAX / n_tok ||
      in_dim > UINT32_MAX || out_dim > UINT32_MAX) {
    return 0;
  }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t x_bytes = 0;
    uint64_t out_bytes = 0;
    if (!hip_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !hip_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        !hip_u64_mul3_checked(in_dim, n_tok, sizeof(float), &x_bytes) ||
        !hip_u64_mul3_checked(out_dim, n_tok, sizeof(float), &out_bytes) ||
        !hip_tensor_has_bytes(x, x_bytes) ||
        !hip_tensor_has_bytes(gate, out_bytes) ||
        !hip_tensor_has_bytes(up, out_bytes) ||
        !hip_tensor_has_bytes(mid, out_bytes)) {
      return 0;
    }

    if (!ds4_gpu_matmul_q8_0_pair_tensor(gate, up, model_map, model_size,
                                         gate_offset, up_offset, in_dim,
                                         out_dim, out_dim, x, xq8, n_tok)) {
      return 0;
    }
    if (!mid_q8) {
      return ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)(out_dim * n_tok),
                                   clamp, 1.0f);
    }
    int8_t* q = NULL;
    float* q_scale = NULL;
    if (n_tok != 1u || out_dim % 256u != 0u ||
        !hip_q8_0_act(mid_q8, 1u, out_dim / 32u, &q, &q_scale)) {
      return 0;
    }
    swiglu_quantize_q8_0_kernel<<<(uint32_t)(out_dim / 256u), 256>>>(
        (float*)mid->ptr, q, q_scale, (const float*)gate->ptr,
        (const float*)up->ptr, clamp, 1.0f);
    return hip_ok(hipGetLastError(), "swiglu quantize launch");
}

/* ---------------------------------------------------------------------------
 * V4.1 GSQ stores the shared FFN experts as Q5_K.  The kernels below dot the
 * activations against the Q5_K blocks as they are stored in the GGUF — the
 * weights are never dequantized into a device-side copy — and fuse the SwiGLU
 * so the shared path keeps the same tensors-in, fp32-out contract as the
 * Q8_0 tier used by V4 Flash. */
#include "../mmq/ggml-common.h"

__device__ static inline void ds4_q5k_scale_min(int j, const uint8_t *q,
                                                uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

__device__ static float ds4_q5k_dot_block(const block_q5_K *b, const float *x) {
    const uint8_t *ql = b->qs;
    const uint8_t *qh = b->qh;
    const float d = __half2float(b->data.d);
    const float mn = __half2float(b->data.dmin);
    float acc = 0.0f;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, m;
        ds4_q5k_scale_min(is, b->scales, &sc, &m);
        const float d1 = d * (float)sc, m1 = mn * (float)m;
        ds4_q5k_scale_min(is + 1, b->scales, &sc, &m);
        const float d2 = d * (float)sc, m2 = mn * (float)m;
        for (int l = 0; l < 32; ++l)
            acc += (d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1) * x[j + l];
        for (int l = 0; l < 32; ++l)
            acc += (d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2) * x[j + 32 + l];
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
    return acc;
}

/* One warp per (token, output row); swiglu(gate_row, up_row) lands in mid. */
__global__ static void shared_gate_up_swiglu_q5_k_kernel(
        float *mid, const char *gate_w, const char *up_w, const float *x,
        uint64_t row_bytes, uint32_t in_blocks, uint32_t in_dim,
        uint32_t out_dim, float clamp) {
    const uint32_t tok = blockIdx.z;
    const uint32_t col = blockIdx.y;
    const block_q5_K *g =
        (const block_q5_K *)(gate_w + (uint64_t)col * row_bytes);
    const block_q5_K *u =
        (const block_q5_K *)(up_w + (uint64_t)col * row_bytes);
    const float *xr = x + (uint64_t)tok * in_dim;
    float ag = 0.0f, au = 0.0f;
    for (uint32_t c = threadIdx.x; c < in_blocks; c += blockDim.x) {
        ag += ds4_q5k_dot_block(g + c, xr + (uint64_t)c * QK_K);
        au += ds4_q5k_dot_block(u + c, xr + (uint64_t)c * QK_K);
    }
    const MASK_T mask = static_cast<MASK_T>(0xffffffffffffffffull);
    for (int o = 16; o > 0; o >>= 1) {
        ag += __shfl_down_sync(mask, ag, o);
        au += __shfl_down_sync(mask, au, o);
    }
    if ((threadIdx.x & 31u) == 0u)
        mid[(uint64_t)tok * out_dim + col] = swiglu_value(ag, au, clamp, 1.0f);
}

__global__ static void shared_down_q5_k_kernel(
        float *out, const char *w, const float *mid, uint64_t row_bytes,
        uint32_t in_blocks, uint32_t in_dim, uint32_t out_dim) {
    const uint32_t tok = blockIdx.z;
    const uint32_t col = blockIdx.y;
    const block_q5_K *b = (const block_q5_K *)(w + (uint64_t)col * row_bytes);
    const float *xr = mid + (uint64_t)tok * in_dim;
    float acc = 0.0f;
    for (uint32_t c = threadIdx.x; c < in_blocks; c += blockDim.x)
        acc += ds4_q5k_dot_block(b + c, xr + (uint64_t)c * QK_K);
    const MASK_T mask = static_cast<MASK_T>(0xffffffffffffffffull);
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(mask, acc, o);
    if ((threadIdx.x & 31u) == 0u)
        out[(uint64_t)tok * out_dim + col] = acc;
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q5_k_tensor(
    ds4_gpu_tensor *mid, const void *model_map, uint64_t model_size,
    uint64_t gate_offset, uint64_t up_offset, uint64_t in_dim,
    uint64_t out_dim, const ds4_gpu_tensor *x, float clamp, uint32_t n_tok) {
    if (!mid || !model_map || !x || in_dim == 0u || out_dim == 0u ||
        n_tok == 0u || (in_dim % QK_K) != 0u || in_dim > UINT32_MAX ||
        out_dim > UINT32_MAX || n_tok > 4096u) {
        return 0;
    }
    const uint64_t in_blocks = in_dim / QK_K;
    const uint64_t row_bytes = in_blocks * sizeof(block_q5_K);
    uint64_t weight_bytes = 0;
    if (!hip_u64_mul_checked(out_dim, row_bytes, &weight_bytes)) return 0;
    const char *gate_w =
        hip_model_range_ptr(model_map, gate_offset, weight_bytes, "shared_gate_q5k");
    const char *up_w =
        hip_model_range_ptr(model_map, up_offset, weight_bytes, "shared_up_q5k");
    if (!gate_w || !up_w) return 0;
    if (!hip_tensor_has_f32(mid, (uint64_t)n_tok * out_dim) ||
        !hip_tensor_has_f32(x, (uint64_t)n_tok * in_dim)) {
        return 0;
    }
    dim3 grid(1u, (unsigned)out_dim, n_tok);
    shared_gate_up_swiglu_q5_k_kernel<<<grid, 32u, 0, (hipStream_t)0>>>(
        (float *)mid->ptr, gate_w, up_w, (const float *)x->ptr, row_bytes,
        (uint32_t)in_blocks, (uint32_t)in_dim, (uint32_t)out_dim, clamp);
    return hip_ok(hipGetLastError(), "shared_gate_up_q5k launch");
}

extern "C" int ds4_gpu_shared_down_q5_k_tensor(
    ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
    uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
    const ds4_gpu_tensor *mid, uint32_t n_tok) {
    if (!out || !model_map || !mid || in_dim == 0u || out_dim == 0u ||
        n_tok == 0u || (in_dim % QK_K) != 0u || in_dim > UINT32_MAX ||
        out_dim > UINT32_MAX || n_tok > 4096u) {
        return 0;
    }
    const uint64_t in_blocks = in_dim / QK_K;
    const uint64_t row_bytes = in_blocks * sizeof(block_q5_K);
    uint64_t weight_bytes = 0;
    if (!hip_u64_mul_checked(out_dim, row_bytes, &weight_bytes)) return 0;
    const char *w =
        hip_model_range_ptr(model_map, weight_offset, weight_bytes, "shared_down_q5k");
    if (!w) return 0;
    if (!hip_tensor_has_f32(out, (uint64_t)n_tok * out_dim) ||
        !hip_tensor_has_f32(mid, (uint64_t)n_tok * in_dim)) {
        return 0;
    }
    dim3 grid(1u, (unsigned)out_dim, n_tok);
    shared_down_q5_k_kernel<<<grid, 32u, 0, (hipStream_t)0>>>(
        (float *)out->ptr, w, (const float *)mid->ptr, row_bytes,
        (uint32_t)in_blocks, (uint32_t)in_dim, (uint32_t)out_dim);
    return hip_ok(hipGetLastError(), "shared_down_q5k launch");
}

/* Generic Q5_K row GEMV: out[t, o] = x[t, :] . W[o, :] with W stored as Q5_K
 * blocks (used for the V4.1 lm_head, which GSQ quantizes to Q5_K).  Weights
 * are read as-is; unpacking happens in registers. */
__global__ static void matmul_q5_k_rows_kernel(
        float *out, const char *w, const float *x, uint64_t row_bytes,
        uint32_t in_blocks, uint32_t in_dim, uint32_t out_dim) {
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t tok = blockIdx.y;
    if (row >= out_dim) return;
    const block_q5_K *b = (const block_q5_K *)(w + (uint64_t)row * row_bytes);
    const float *xr = x + (uint64_t)tok * in_dim;
    float acc = 0.0f;
    for (uint32_t c = threadIdx.x; c < in_blocks; c += blockDim.x)
        acc += ds4_q5k_dot_block(b + c, xr + (uint64_t)c * QK_K);
    const MASK_T mask = static_cast<MASK_T>(0xffffffffffffffffull);
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(mask, acc, o);
    if (threadIdx.x == 0u)
        out[(uint64_t)tok * out_dim + row] = acc;
}

extern "C" int ds4_gpu_matmul_q5_k_tensor(
    ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
    uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
    const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (!out || !model_map || !x || in_dim == 0u || out_dim == 0u ||
        n_tok == 0u || (in_dim % QK_K) != 0u || in_dim > UINT32_MAX ||
        out_dim > UINT32_MAX || n_tok > 4096u) {
        return 0;
    }
    const uint64_t in_blocks = in_dim / QK_K;
    const uint64_t row_bytes = in_blocks * sizeof(block_q5_K);
    uint64_t weight_bytes = 0;
    if (!hip_u64_mul_checked(out_dim, row_bytes, &weight_bytes)) return 0;
    const char *w =
        hip_model_range_ptr(model_map, weight_offset, weight_bytes, "matmul_q5k");
    if (!w) return 0;
    if (!hip_tensor_has_f32(out, (uint64_t)n_tok * out_dim) ||
        !hip_tensor_has_f32(x, (uint64_t)n_tok * in_dim)) {
        return 0;
    }
    const dim3 block(32u, 8u);
    const dim3 grid((unsigned)((out_dim + 7u) / 8u), n_tok);
    matmul_q5_k_rows_kernel<<<grid, block, 0, (hipStream_t)0>>>(
        (float *)out->ptr, w, (const float *)x->ptr, row_bytes,
        (uint32_t)in_blocks, (uint32_t)in_dim, (uint32_t)out_dim);
    return hip_ok(hipGetLastError(), "matmul_q5k launch");
}
