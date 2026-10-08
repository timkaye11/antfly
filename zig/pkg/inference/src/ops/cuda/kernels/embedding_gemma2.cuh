// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// EmbeddingGemma 2's encoder deliberately has its own kernels.  Decoder
// attention is causal and applies a sqrt(head_dim) scale; neither property is
// valid here.  Values are BF16 in memory and all reductions remain FP32.

__device__ __forceinline__ float eg2_warp_sum(float x) {
    for (int d = 16; d; d >>= 1) x += __shfl_down_sync(0xffffffffu, x, d);
    return x;
}

__device__ __forceinline__ float eg2_warp_max(float x) {
    for (int d = 16; d; d >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffffu, x, d));
    return x;
}

// Gemma4/EmbeddingGemma2 audio local attention. Q/K/V/relative inputs and
// output are FP32; one warp owns a query/head and the causal context is tiny
// (24 slots, at most 12 valid keys), so scores and output stay in registers.
extern "C" __global__ void termite_gemma4_audio_local_attention_f32(
    float* out, const float* q, const float* k, const float* v,
    const float* rel, const float* q_scales, const float* valid,
    unsigned int rows, unsigned int hidden, unsigned int heads,
    unsigned int head_dim, unsigned int chunk, unsigned int context_left,
    unsigned int context, float k_scale, float logit_cap, float invalid_value) {
    unsigned int lane = threadIdx.x;
    unsigned int query = blockIdx.x, head = blockIdx.y;
    if (query >= rows || head >= heads || lane >= 32u) return;
    size_t out_base = (size_t)query * hidden + (size_t)head * head_dim;
    if (valid[query] == 0.0f) {
        for (unsigned int d = lane; d < head_dim; d += 32u) out[out_base + d] = 0.0f;
        return;
    }
    float scores[24];
    #pragma unroll
    for (unsigned int i = 0; i < 24; ++i) scores[i] = invalid_value;
    unsigned int block_start = (query / chunk) * chunk;
    unsigned int q_off = query - block_start;
    unsigned int past = context_left - 1u;
    float max_score = -INFINITY;
    for (unsigned int c = 0; c < context; ++c) {
        int rel_idx = (int)c - (int)q_off;
        int key_idx = (int)block_start + (int)c - (int)past;
        // Official sliding mask uses distance < (context_left - 1), so a
        // distance exactly equal to `past` is excluded.
        if (rel_idx < 0 || rel_idx >= (int)context_left || key_idx < 0 ||
            key_idx >= (int)rows || valid[key_idx] == 0.0f || key_idx > (int)query ||
            query - (unsigned int)key_idx >= past) continue;
        float score = 0.0f;
        size_t qb = (size_t)query * hidden + (size_t)head * head_dim;
        size_t kb = (size_t)key_idx * hidden + (size_t)head * head_dim;
        size_t rb = (size_t)rel_idx * hidden + (size_t)head * head_dim;
        for (unsigned int d = lane; d < head_dim; d += 32u) {
            float qs = q[qb + d] * q_scales[d];
            score += qs * (k[kb + d] * k_scale + rel[rb + d]);
        }
        score = eg2_warp_sum(score);
        score = __shfl_sync(0xffffffffu, score, 0);
        score = tanhf(score / logit_cap) * logit_cap;
        scores[c] = score;
        max_score = fmaxf(max_score, score);
    }
    float denominator = 0.0f;
    #pragma unroll
    for (unsigned int c = 0; c < 24; ++c) {
        if (c < context) {
            scores[c] = expf(scores[c] - max_score);
            denominator += scores[c];
        }
    }
    for (unsigned int d = lane; d < head_dim; d += 32u) {
        float value = 0.0f;
        for (unsigned int c = 0; c < context; ++c) {
            int key_idx = (int)block_start + (int)c - (int)past;
            if (key_idx < 0 || key_idx >= (int)rows) continue;
            value += (scores[c] / denominator) * v[(size_t)key_idx * hidden + (size_t)head * head_dim + d];
        }
        out[out_base + d] = denominator > 0.0f ? value : 0.0f;
    }
}

extern "C" __global__ void termite_embedding_gemma2_lookup_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* table, const long long* ids,
    unsigned int rows, unsigned int vocab, unsigned int hidden, float scale) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)rows * hidden;
    if (i >= n) return;
    unsigned int row = (unsigned int)(i / hidden), col = (unsigned int)(i % hidden);
    long long id = ids[row];
    out[i] = (id >= 0 && (unsigned long long)id < vocab)
        ? __float2bfloat16_rn(__bfloat162float(table[(size_t)id * hidden + col]) * scale)
        : __float2bfloat16_rn(0.0f);
}

extern "C" __global__ void termite_embedding_gemma2_rms_norm_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* input, const __nv_bfloat16* weight,
    unsigned int rows, unsigned int width, float eps) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < width; i += blockDim.x) {
        float x = __bfloat162float(input[(size_t)row * width + i]); sum += x * x;
    }
    __shared__ float warp_sum[8], inv;
    sum = eg2_warp_sum(sum);
    if ((threadIdx.x & 31u) == 0) warp_sum[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float x = threadIdx.x < (blockDim.x >> 5) ? warp_sum[threadIdx.x] : 0.0f;
        x = eg2_warp_sum(x);
        if (threadIdx.x == 0) inv = rsqrtf(x / (float)width + eps);
    }
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < width; i += blockDim.x) {
        float x = __bfloat162float(input[(size_t)row * width + i]);
        // EmbeddingGemma 2 uses direct RMS weights (no implicit +1).
        out[(size_t)row * width + i] = __float2bfloat16_rn(x * inv * (weight ? __bfloat162float(weight[i]) : 1.0f));
    }
}

extern "C" __global__ void termite_embedding_gemma2_rope_bf16(
    __nv_bfloat16* q, __nv_bfloat16* k, const long long* mask,
    unsigned int batch, unsigned int seq, unsigned int q_heads,
    unsigned int kv_heads, unsigned int dim, float theta, float rope_scale) {
    unsigned int pairs = dim >> 1;
    unsigned int pair = blockIdx.y * blockDim.x + threadIdx.x;
    unsigned int bs = blockIdx.x;
    unsigned int head = blockIdx.z;
    if (pair >= pairs || bs >= batch * seq || head >= q_heads) return;
    unsigned int b = bs / seq, pos = bs % seq;
    if (mask && mask[(size_t)b * seq + pos] == 0) return;
    float angle = ((float)pos / rope_scale) * powf(theta, -(2.0f * pair) / (float)dim);
    float s, c; sincosf(angle, &s, &c); s=__bfloat162float(__float2bfloat16_rn(s)); c=__bfloat162float(__float2bfloat16_rn(c));
    size_t qi = (((size_t)b * seq + pos) * q_heads + head) * dim + pair;
    float q0 = __bfloat162float(q[qi]), q1 = __bfloat162float(q[qi + pairs]);
    q[qi] = __float2bfloat16_rn(q0 * c - q1 * s); q[qi + pairs] = __float2bfloat16_rn(q1 * c + q0 * s);
    if (head < kv_heads) {
        size_t ki = (((size_t)b * seq + pos) * kv_heads + head) * dim + pair;
        float k0 = __bfloat162float(k[ki]), k1 = __bfloat162float(k[ki + pairs]);
        k[ki] = __float2bfloat16_rn(k0 * c - k1 * s); k[ki + pairs] = __float2bfloat16_rn(k1 * c + k0 * s);
    }
}

// Exact bounded-memory route. One warp owns a query/head and performs online
// softmax entirely in registers, so no SxS tensor or block-wide barrier is
// created. Each lane owns up to 16 values (head_dim <= 512).
// local_radius==0 denotes full bidirectional attention.
extern "C" __global__ void termite_embedding_gemma2_attention_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, const long long* mask, unsigned int batch,
    unsigned int seq, unsigned int q_heads, unsigned int kv_heads,
    unsigned int dim, unsigned int local_radius, unsigned int query_offset) {
    unsigned int lane = threadIdx.x;
    unsigned int query = query_offset + blockIdx.x, qh = blockIdx.y, b = blockIdx.z;
    if (b >= batch || query >= seq || qh >= q_heads) return;
    size_t out_base = (((size_t)b * seq + query) * q_heads + qh) * dim;
    if (mask && mask[(size_t)b * seq + query] == 0) {
        for (unsigned int d = lane; d < dim; d += 32u) out[out_base + d] = __float2bfloat16_rn(0.0f);
        return;
    }
    unsigned int kvh = qh * kv_heads / q_heads;
    unsigned int begin = local_radius && query > local_radius ? query - local_radius : 0;
    unsigned int end = local_radius ? min(seq, query + local_radius + 1u) : seq;
    float accum[16];
    #pragma unroll
    for (unsigned int n = 0; n < 16; ++n) accum[n] = 0.0f;
    float max_score = -INFINITY, denominator = 0.0f;
    size_t qb = (((size_t)b * seq + query) * q_heads + qh) * dim;
    for (unsigned int j = begin; j < end; ++j) {
        if (mask && mask[(size_t)b * seq + j] == 0) continue;
        size_t kb = (((size_t)b * seq + j) * kv_heads + kvh) * dim;
        float dot = 0.0f;
        for (unsigned int d = lane; d < dim; d += 32u)
            dot += __bfloat162float(q[qb + d]) * __bfloat162float(k[kb + d]);
        dot = eg2_warp_sum(dot);
        dot = __shfl_sync(0xffffffffu, dot, 0);
        float next_max = fmaxf(max_score, dot);
        float prior_scale = isfinite(max_score) ? expf(max_score - next_max) : 0.0f;
        float probability = expf(dot - next_max);
        denominator = denominator * prior_scale + probability;
        max_score = next_max;
        #pragma unroll
        for (unsigned int n = 0; n < 16; ++n) {
            unsigned int d = lane + n * 32u;
            if (d < dim) accum[n] = accum[n] * prior_scale + probability * __bfloat162float(v[kb + d]);
        }
    }
    #pragma unroll
    for (unsigned int n = 0; n < 16; ++n) {
        unsigned int d = lane + n * 32u;
        if (d < dim) out[out_base + d] = __float2bfloat16_rn(denominator > 0.0f ? accum[n] / denominator : 0.0f);
    }
}


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
extern "C" __global__ void termite_embedding_gemma2_attention_local_flash_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, const long long* mask, unsigned batch,
    unsigned seq, unsigned q_heads, unsigned kv_heads, unsigned dim,
    unsigned local_radius) {
    using namespace nvcuda;
    constexpr unsigned QT = 64, KT = 16, MMA = 16, THREADS = 512, WARPS = 16, QK_WARPS = 4;
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const unsigned qh = blockIdx.x;
    const unsigned qt = blockIdx.y * QT;
    const unsigned b = blockIdx.z;
    if (blockDim.x != THREADS || b >= batch || qh >= q_heads || qt >= seq ||
        (dim != 256 && dim != 512) || (kv_heads != 1 && kv_heads != 2)) return;
    const unsigned valid_rows = min(QT, seq - qt);
    const unsigned kvh = qh * kv_heads / q_heads;

    extern __shared__ unsigned char storage[];
    __nv_bfloat16* kvs = reinterpret_cast<__nv_bfloat16*>(storage);
    // Four dimension partials for every query share storage with the sixteen
    // warp-private PV merge tiles; both require 4096 FP32 values.
    float* warp_scratch = reinterpret_cast<float*>(kvs + KT * dim);
    __nv_bfloat16* probs = reinterpret_cast<__nv_bfloat16*>(warp_scratch + QK_WARPS * QT * KT);
    float* running_max = reinterpret_cast<float*>(probs + QT * KT);
    float* running_sum = running_max + QT;
    float* alpha = running_sum + QT;

    if (tid < QT) {
        running_max[tid] = -INFINITY;
        running_sum[tid] = 0.0f;
    }
    __syncthreads();

    // Four query subtiles; HD256 uses one output fragment per warp and HD512
    // uses two.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> out_frag[4][2];
    #pragma unroll
    for (unsigned qsub = 0; qsub < 4; ++qsub)
        for (unsigned f = 0; f < 2; ++f) wmma::fill_fragment(out_frag[qsub][f], 0.0f);
    const unsigned fragments = (dim / 16) / WARPS;

    unsigned first = 0, last = seq;
    if (local_radius) {
        first = qt > local_radius ? qt - local_radius : 0;
        last = min(seq, qt + valid_rows + local_radius);
    }
    first = (first / KT) * KT;
    last = ((last + KT - 1) / KT) * KT;
    for (unsigned kt = first; kt < last; kt += KT) {
        for (unsigned i = tid; i < KT * dim; i += THREADS) {
            const unsigned row = i / dim, d = i - row * dim, key = kt + row;
            const bool valid = key < seq && (!mask || mask[(size_t)b * seq + key]);
            kvs[i] = valid ? k[(((size_t)b * seq + key) * kv_heads + kvh) * dim + d]
                           : __float2bfloat16_rn(0.0f);
        }
        __syncthreads();

        const unsigned qsub = (warp / QK_WARPS) * MMA;
        const unsigned partial = warp % QK_WARPS;
        const unsigned slice = dim / QK_WARPS;
        const unsigned begin_d = partial * slice;
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sf;
            wmma::fill_fragment(sf, 0.0f);
            for (unsigned d = begin_d; d < begin_d + slice; d += MMA) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16,
                    __nv_bfloat16, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 16,
                    __nv_bfloat16, wmma::col_major> bf;
                const __nv_bfloat16* q_tile = q +
                    (((size_t)b * seq + qt + qsub) * q_heads + qh) * dim;
                wmma::load_matrix_sync(af, q_tile + d, q_heads * dim);
                wmma::load_matrix_sync(bf, kvs + d, dim);
                wmma::mma_sync(sf, af, bf, sf);
            }
            wmma::store_matrix_sync(
                warp_scratch + partial * QT * KT + qsub * KT, sf, KT,
                wmma::mem_row_major);
        }
        __syncthreads();

        for (unsigned score_base = 0; score_base < QT * KT; score_base += THREADS) {
            const unsigned score_index = score_base + tid;
            const unsigned row = score_index / KT, col = score_index - row * KT;
            const unsigned query = qt + row, key = kt + col;
            float score = 0.0f;
            #pragma unroll
            for (unsigned w = 0; w < QK_WARPS; ++w)
                score += warp_scratch[(w * QT + row) * KT + col];
            const bool visible = row < valid_rows && query < seq && key < seq &&
                (!mask || (mask[(size_t)b * seq + query] && mask[(size_t)b * seq + key])) &&
                (!local_radius || (key <= query + local_radius && query <= key + local_radius));
            score = visible ? score : -INFINITY;
            float tile_max = score;
            #pragma unroll
            for (unsigned offset = KT / 2; offset != 0; offset >>= 1)
                tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, offset, KT));
            const float old_max = running_max[row];
            const float next_max = fmaxf(old_max, tile_max);
            const float a = isfinite(old_max) ? expf(old_max - next_max) : 0.0f;
            const float p = isfinite(score) ? expf(score - next_max) : 0.0f;
            probs[row * KT + col] = __float2bfloat16_rn(p);
            float tile_sum = p;
            #pragma unroll
            for (unsigned offset = KT / 2; offset != 0; offset >>= 1)
                tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, offset, KT);
            if (col == 0) {
                alpha[row] = a;
                running_sum[row] = running_sum[row] * a + tile_sum;
                running_max[row] = next_max;
            }
        }
        __syncthreads();
        // QK partials are dead; reuse their shared allocation for V.
        for (unsigned i = tid; i < KT * dim; i += THREADS) {
            const unsigned row = i / dim, d = i - row * dim, key = kt + row;
            const bool valid = key < seq && (!mask || mask[(size_t)b * seq + key]);
            kvs[i] = valid ? v[(((size_t)b * seq + key) * kv_heads + kvh) * dim + d]
                           : __float2bfloat16_rn(0.0f);
        }
        __syncthreads();
        for (unsigned qsub = 0; qsub < 4; ++qsub) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16,
                __nv_bfloat16, wmma::row_major> pf;
            wmma::load_matrix_sync(pf, probs + qsub * MMA * KT, KT);
            for (unsigned f = 0; f < fragments; ++f) {
                const unsigned ot = warp + f * WARPS;
                wmma::fragment<wmma::matrix_b, 16, 16, 16,
                    __nv_bfloat16, wmma::row_major> vf;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> tf;
                wmma::load_matrix_sync(vf, kvs + ot * MMA, dim);
                wmma::fill_fragment(tf, 0.0f);
                wmma::mma_sync(tf, pf, vf, tf);
                #pragma unroll
                for (unsigned i = 0; i < out_frag[qsub][f].num_elements; ++i) {
                    // SM80+ WMMA m16n16 accumulator ownership: each lane owns
                    // two columns in rows lane/4 and lane/4+8. Applying alpha
                    // here avoids a store/scale/reload of every output tile.
                    const unsigned fragment_row = (lane >> 2) + ((i & 2u) ? 8u : 0u);
                    out_frag[qsub][f].x[i] *= alpha[qsub * MMA + fragment_row];
                    out_frag[qsub][f].x[i] += tf.x[i];
                }
            }
        }
        __syncthreads();
    }

    for (unsigned qsub = 0; qsub < 4; ++qsub) {
        for (unsigned f = 0; f < fragments; ++f) {
            const unsigned ot = warp + f * WARPS;
            float* scratch = warp_scratch + warp * MMA * KT;
            wmma::store_matrix_sync(scratch, out_frag[qsub][f], KT, wmma::mem_row_major);
            __syncwarp();
            for (unsigned i = lane; i < MMA * KT; i += 32) {
                const unsigned row = qsub * MMA + i / KT, col = i % KT;
                if (row < valid_rows && ot * MMA + col < dim) {
                    const float denom = running_sum[row];
                    out[(((size_t)b * seq + qt + row) * q_heads + qh) * dim + ot * MMA + col] =
                        __float2bfloat16_rn(denom > 0.0f ? scratch[i] / denom : 0.0f);
                }
            }
        }
    }
}
#else
extern "C" __global__ void termite_embedding_gemma2_attention_local_flash_bf16(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const long long*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned) {}
#endif


extern "C" __global__ void termite_embedding_gemma2_attention_softmax_bf16(
    __nv_bfloat16* probs, const float* scores, const long long* mask,
    unsigned int sequence, unsigned int query_base, unsigned int rows,
    unsigned int batch_index, unsigned int tile_capacity) {
    unsigned int row = blockIdx.x % rows;
    size_t matrix_offset = (size_t)(blockIdx.x / rows) * tile_capacity * sequence;
    unsigned int lane = threadIdx.x, query = query_base + row;
    bool query_valid = !mask || mask[(size_t)batch_index * sequence + query] != 0;
    extern __shared__ float reduce[];
    float local_max = -INFINITY;
    for (unsigned int key = lane; key < sequence; key += blockDim.x)
        if (query_valid && (!mask || mask[(size_t)batch_index * sequence + key]))
            local_max = fmaxf(local_max, scores[matrix_offset + (size_t)row * sequence + key]);
    reduce[lane] = local_max; __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (lane < stride) reduce[lane] = fmaxf(reduce[lane], reduce[lane + stride]);
        __syncthreads();
    }
    float row_max = reduce[0], local_sum = 0.0f;
    for (unsigned int key = lane; key < sequence; key += blockDim.x) {
        bool visible = query_valid && (!mask || mask[(size_t)batch_index * sequence + key]);
        float value = visible ? expf(scores[matrix_offset + (size_t)row * sequence + key] - row_max) : 0.0f;
        probs[matrix_offset + (size_t)row * sequence + key] = __float2bfloat16_rn(value);
        local_sum += value;
    }
    reduce[lane] = local_sum; __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (lane < stride) reduce[lane] += reduce[lane + stride];
        __syncthreads();
    }
    float inverse = reduce[0] > 0.0f ? 1.0f / reduce[0] : 0.0f;
    for (unsigned int key = lane; key < sequence; key += blockDim.x)
        probs[matrix_offset + (size_t)row * sequence + key] = __float2bfloat16_rn(
            __bfloat162float(probs[matrix_offset + (size_t)row * sequence + key]) * inverse);
}

extern "C" __global__ void termite_embedding_gemma2_matmul_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* input, const __nv_bfloat16* weight,
    unsigned int rows, unsigned int in_dim, unsigned int out_dim) {
    unsigned int row = blockIdx.x;
    unsigned int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= rows || col >= out_dim) return;
    float sum = 0.0f;
    for (unsigned int k = 0; k < in_dim; ++k)
        sum += __bfloat162float(input[(size_t)row * in_dim + k]) * __bfloat162float(weight[(size_t)col * in_dim + k]);
    out[(size_t)row * out_dim + col] = __float2bfloat16_rn(sum);
}

extern "C" __global__ void termite_embedding_gemma2_gelu_ple_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* ple, unsigned int count) {
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=count)return;
    float x=__bfloat162float(gate[i]); float g=0.5f*x*(1.0f+tanhf(0.7978845608028654f*(x+0.044715f*x*x*x))); g=__bfloat162float(__float2bfloat16_rn(g));
    out[i]=__float2bfloat16_rn(g*__bfloat162float(ple[i]));
}

extern "C" __global__ void termite_embedding_gemma2_gelu_mul_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up,
    unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float x = __bfloat162float(gate[i]);
    float g = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); g=__bfloat162float(__float2bfloat16_rn(g));
    out[i] = __float2bfloat16_rn(g * __bfloat162float(up[i]));
}

extern "C" __global__ void termite_embedding_gemma2_scale_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* input, unsigned int count, float scale) {
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)out[i]=__float2bfloat16_rn(__bfloat162float(input[i])*scale);
}

extern "C" __global__ void termite_embedding_gemma2_bf16_to_f32(
    float* out, const __nv_bfloat16* input, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = __bfloat162float(input[i]);
}

extern "C" __global__ void termite_embedding_gemma2_scale_device_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* input, const __nv_bfloat16* scale, unsigned int count) {
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)out[i]=__float2bfloat16_rn(__bfloat162float(input[i])*__bfloat162float(scale[0]));
}

extern "C" __global__ void termite_embedding_gemma2_residual_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* residual, const __nv_bfloat16* branch,
    const __nv_bfloat16* ple, unsigned int count, float branch_scale, float ple_scale) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float x = __bfloat162float(residual[i]) + branch_scale * __bfloat162float(branch[i]);
    if (ple) x += ple_scale * __bfloat162float(ple[i]);
    out[i] = __float2bfloat16_rn(x);
}

extern "C" __global__ void termite_embedding_gemma2_mean_l2_f32(
    float* out, const __nv_bfloat16* projected, const long long* mask,
    unsigned int batch, unsigned int seq, unsigned int width) {
    unsigned int b = blockIdx.x;
    if (b >= batch) return;
    __shared__ float norm_parts[8], inv_norm;
    float norm = 0.0f;
    for (unsigned int d = threadIdx.x; d < width; d += blockDim.x) {
        float sum = 0.0f, count = 0.0f;
        for (unsigned int s = 0; s < seq; ++s) if (!mask || mask[(size_t)b * seq + s]) {
            sum += __bfloat162float(projected[((size_t)b * seq + s) * width + d]); count += 1.0f;
        }
        float x = count > 0.0f ? sum / count : 0.0f;
        out[(size_t)b * width + d] = x; norm += x * x;
    }
    norm = eg2_warp_sum(norm);
    if ((threadIdx.x & 31u) == 0) norm_parts[threadIdx.x >> 5] = norm;
    __syncthreads();
    if (threadIdx.x < 32) {
        float x = threadIdx.x < (blockDim.x >> 5) ? norm_parts[threadIdx.x] : 0.0f;
        x = eg2_warp_sum(x);
        if (threadIdx.x == 0) inv_norm = x > 0.0f ? rsqrtf(x) : 0.0f;
    }
    __syncthreads();
    for (unsigned int d = threadIdx.x; d < width; d += blockDim.x)
        out[(size_t)b * width + d] *= inv_norm;
}

extern "C" __global__ void termite_gemma4_audio_clamp_f32(
    float* out, const float* input, unsigned int count,
    float lower, float upper, unsigned int has_lower, unsigned int has_upper) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float value = input[i];
    // Ordered comparisons preserve NaN, matching torch.clamp. Infinities
    // clamp normally when the corresponding finite bound is present.
    if (has_lower && value < lower) value = lower;
    if (has_upper && value > upper) value = upper;
    out[i] = value;
}

extern "C" __global__ void termite_gemma4_audio_glu_rows_f32(
    float* out, const float* input, unsigned int rows, unsigned int dim) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = (size_t)rows * dim;
    if ((size_t)i >= count) return;
    const unsigned int row = i / dim;
    const unsigned int column = i - row * dim;
    const size_t base = (size_t)row * dim * 2;
    const float gate = input[base + dim + column];
    out[i] = input[base + column] / (1.0f + expf(-gate));
}

extern "C" __global__ void termite_gemma4_audio_depthwise_causal_conv1d_f32(
    float* out, const float* input, const float* weight,
    unsigned int rows, unsigned int dim, unsigned int kernel_size) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = (size_t)rows * dim;
    if ((size_t)i >= count) return;
    const unsigned int time = i / dim;
    const unsigned int channel = i - time * dim;
    const size_t left_pad = (size_t)kernel_size - 1;
    float sum = 0.0f;
    for (unsigned int kernel = 0; kernel < kernel_size; ++kernel) {
        const size_t padded_time = (size_t)time + kernel;
        if (padded_time < left_pad) continue;
        const size_t source_time = padded_time - left_pad;
        sum += input[(size_t)source_time * dim + channel] *
            weight[(size_t)kernel * dim + channel];
    }
    out[i] = sum;
}
