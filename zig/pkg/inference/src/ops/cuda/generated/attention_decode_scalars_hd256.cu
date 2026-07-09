// Copyright 2026 Antfly, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Dev-only generated attention kernel from graph/quant_kernel_compiler.zig.
// plan_id=cuda/attention/decode_1x/hd256/device_scalars
// kernel_id=antfly_gqa_attention_decode_scalars_hd256_f32_v1
// production_baseline=termite_gqa_attention_decode_scalars_f32
// production_enabled=false
// Runtime opt-in: ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_DECODE=1.

#include <math.h>

extern "C" __global__ void antfly_gqa_attention_decode_scalars_hd256_f32_v1(
    float* dst,
    const float* q,
    const float* k,
    const float* v,
    const unsigned char* attn_or_mask,
    const float* bias,
    unsigned int batch,
    unsigned int q_seq_len,
    unsigned int kv_seq_len,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int query_position_offset,
    unsigned int kv_position_offset,
    unsigned int sliding_window,
    unsigned int total_sequence_len,
    unsigned int mask_len,
    unsigned int bias_mode,
    const unsigned int* decode_scalars
) {
    if (decode_scalars != 0) {
        kv_position_offset = decode_scalars[4];
        query_position_offset = decode_scalars[1];
        kv_seq_len = decode_scalars[2];
        total_sequence_len = decode_scalars[3];
    }

    __shared__ float warp_sums[4];
    __shared__ float shared_max_score;
    __shared__ float shared_denom;
    __shared__ float shared_alpha;
    __shared__ float shared_beta;
    unsigned int lane = threadIdx.x;
    unsigned int block = blockIdx.x;
    unsigned int total_blocks = batch * q_seq_len * num_heads;
    if (block >= total_blocks || q_seq_len != 1u || head_dim != 256u || blockDim.x != 128u || num_kv_heads == 0u || (num_heads % num_kv_heads) != 0u) return;

    unsigned int head = block % num_heads;
    unsigned int tmp = block / num_heads;
    unsigned int qi = tmp % q_seq_len;
    unsigned int b = tmp / q_seq_len;
    unsigned int heads_per_group = num_heads / num_kv_heads;
    unsigned int kv_head = head / heads_per_group;
    unsigned int q_hidden = num_heads * head_dim;
    unsigned int kv_hidden = num_kv_heads * head_dim;
    unsigned int query_pos = query_position_offset + qi;
    unsigned int q_base = (b * q_seq_len + qi) * q_hidden + head * head_dim;
    float scale = rsqrtf((float)head_dim);
    unsigned int warp = lane >> 5;
    unsigned int warp_lane = lane & 31u;
    if (lane == 0u) {
        shared_max_score = -3.402823466e+38f;
        shared_denom = 0.0f;
    }
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (unsigned int ki = 0; ki < kv_seq_len; ++ki) {
        unsigned int key_pos = kv_position_offset + ki;
        unsigned int mask_idx = query_pos * total_sequence_len + key_pos;
        bool future_allowed = attn_or_mask != 0 && mask_idx < mask_len && attn_or_mask[mask_idx] != 0u;
        bool future_blocked = key_pos > query_pos && !future_allowed;
        bool past_blocked = key_pos > query_pos || (sliding_window != 0u && (query_pos - key_pos) >= sliding_window);
        bool valid = !(future_blocked || past_blocked);
        unsigned int k_base = (b * kv_seq_len + ki) * kv_hidden + kv_head * head_dim;
        float dot = valid ? q[q_base + lane] * k[k_base + lane] + q[q_base + lane + 128u] * k[k_base + lane + 128u] : 0.0f;
        for (unsigned int offset = 16u; offset > 0u; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        if (warp_lane == 0u) warp_sums[warp] = dot;
        __syncthreads();
        if (warp == 0u) {
            float block_dot = warp_lane < 4u ? warp_sums[warp_lane] : 0.0f;
            for (unsigned int offset = 16u; offset > 0u; offset >>= 1) {
                block_dot += __shfl_down_sync(0xffffffffu, block_dot, offset);
            }
            if (warp_lane == 0u) {
                float score = valid ? block_dot * scale : -3.402823466e+38f;
                if (valid && bias_mode == 1u) score += bias[(head * q_seq_len + qi) * kv_seq_len + ki];
                if (valid && bias_mode == 2u) score += bias[((b * num_heads + head) * q_seq_len + qi) * kv_seq_len + ki];
                float next_max = fmaxf(shared_max_score, score);
                float alpha = shared_denom > 0.0f ? expf(shared_max_score - next_max) : 0.0f;
                float beta = valid ? expf(score - next_max) : 0.0f;
                shared_denom = shared_denom * alpha + beta;
                shared_max_score = next_max;
                shared_alpha = alpha;
                shared_beta = beta;
            }
        }
        __syncthreads();
        unsigned int v_idx = (b * kv_seq_len + ki) * kv_hidden + kv_head * head_dim + lane;
        acc0 = acc0 * shared_alpha + shared_beta * v[v_idx];
        acc1 = acc1 * shared_alpha + shared_beta * v[v_idx + 128u];
    }

    unsigned int out_idx = (b * q_seq_len + qi) * q_hidden + head * head_dim + lane;
    dst[out_idx] = shared_denom > 0.0f ? acc0 / shared_denom : 0.0f;
    dst[out_idx + 128u] = shared_denom > 0.0f ? acc1 / shared_denom : 0.0f;
}
