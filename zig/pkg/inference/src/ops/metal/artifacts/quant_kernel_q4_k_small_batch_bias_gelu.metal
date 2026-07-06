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

// Generated Metal candidate artifact from graph/quant_kernel_compiler.zig.
// plan_id=metal/q4_k/rows_2_8/bias_gelu/small_batch
// kernel_id=antfly_q4_k_small_batch_bias_gelu_msl_v1
// production_baseline=metal_handwritten_quant_matmul
// production_enabled=true
// Promoted after sequential Metal runtime evidence cleared correctness,
// route, provider-route, and speedup gates.

#include <metal_stdlib>
using namespace metal;

static inline float antfly_half_le_to_float(const device uchar *p) {
    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    return (float)as_type<half>(bits);
}

static inline float antfly_gelu(float x) {
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + fast::tanh(inner));
}

static inline void antfly_q4_k_unpack_scale_min(
    const device uchar *scales,
    int sub,
    thread float &scale,
    thread float &min_v
) {
    if (sub < 4) {
        scale = (float)(scales[sub] & 63u);
        min_v = (float)(scales[sub + 4] & 63u);
        return;
    }
    scale = (float)((scales[sub + 4] & 0x0fu) | ((scales[sub - 4] >> 6) << 4));
    min_v = (float)((scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4));
}

static inline float antfly_q4_k_dequant_lane(const device uchar *block, int lane) {
    const device uchar *d = block;
    const device uchar *dmin = block + 2;
    const device uchar *scales = block + 4;
    const device uchar *qs = block + 16;
    const int sub = lane >> 5;
    const int q_index = (sub >> 1) * 32 + (lane & 31);
    const uchar packed = qs[q_index];
    const uchar q = (sub & 1) == 0 ? (packed & 0x0fu) : (packed >> 4);
    float raw_scale = 0.0f;
    float raw_min = 0.0f;
    antfly_q4_k_unpack_scale_min(scales, sub, raw_scale, raw_min);
    return antfly_half_le_to_float(d) * raw_scale * (float)q - antfly_half_le_to_float(dmin) * raw_min;
}

kernel void antfly_q4_k_small_batch_bias_gelu_msl_v1(
    const device float *input [[buffer(0)]],
    const device uchar *weight_q4_k [[buffer(1)]],
    const device float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant int &rows [[buffer(4)]],
    constant int &in_dim [[buffer(5)]],
    constant int &out_dim [[buffer(6)]],
    uint3 thread_pos [[thread_position_in_threadgroup]],
    uint3 group_pos [[threadgroup_position_in_grid]]
) {
    const uint tid = thread_pos.x;
    const int col = (int)group_pos.x;
    const int row = (int)group_pos.y;
    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 255) != 0) return;

    float acc = 0.0f;
    const int block_count = in_dim >> 8;
    if (tid < 64) {
        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
            const device uchar *block = weight_q4_k + ((col * block_count + block_idx) * 144);
            const int base = block_idx << 8;
            for (int lane = (int)tid; lane < 256; lane += 64) {
                acc += input[row * in_dim + base + lane] * antfly_q4_k_dequant_lane(block, lane);
            }
        }
    }

    threadgroup float partial[64];
    if (tid < 64) partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[row * out_dim + col] = antfly_gelu(partial[0] + bias[col]);
}
