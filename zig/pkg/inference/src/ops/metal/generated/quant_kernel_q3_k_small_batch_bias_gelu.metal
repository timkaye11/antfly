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

// Dev-only generated Metal candidate from graph/quant_kernel_compiler.zig.
// plan_id=metal/q3_k/rows_2_8/bias_gelu/small_batch
// kernel_id=antfly_q3_k_small_batch_bias_gelu_msl_v1
// production_baseline=metal_handwritten_quant_matmul
// production_enabled=false
// General MSL lowering smoke for descriptor-driven quant matmul epilogues.
// Production Metal dispatch stays on native handwritten MSL until this
// candidate clears correctness and benchmark gates.

#include <metal_stdlib>
using namespace metal;

static inline float antfly_half_le_to_float(const device uchar *p) {
    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    return (float)as_type<half>(bits);
}

static inline float antfly_gelu(float x) {
    return 0.5f * x * (1.0f + fast::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

static inline int antfly_q3_k_raw_scale(const device uchar *scale_data, uint sub) {
    const uint i = sub & 3u;
    uint low = 0u;
    uint high = 0u;
    if (sub < 4u) {
        low = (uint)(scale_data[i] & 0x0Fu);
        high = (uint)(scale_data[8 + i] & 0x03u);
    } else if (sub < 8u) {
        low = (uint)(scale_data[4 + i] & 0x0Fu);
        high = (uint)((scale_data[8 + i] >> 2) & 0x03u);
    } else if (sub < 12u) {
        low = (uint)((scale_data[i] >> 4) & 0x0Fu);
        high = (uint)((scale_data[8 + i] >> 4) & 0x03u);
    } else {
        low = (uint)((scale_data[4 + i] >> 4) & 0x0Fu);
        high = (uint)((scale_data[8 + i] >> 6) & 0x03u);
    }
    return (int)(low | (high << 4)) - 32;
}

static inline float antfly_q3_k_dequant_lane(const device uchar *block, int lane) {
    const uint sub = (uint)lane >> 4;
    const uint i = (uint)lane & 15u;
    const uint chunk = sub >> 3;
    const uint group = (sub & 7u) >> 1;
    const uint l_base = (sub & 1u) << 4;
    const uint l = l_base + i;
    const uint q_base = chunk << 5;
    const uint shift = group << 1;
    const uint hm_bit = (chunk << 2) + group;
    const int low2 = (int)(((uint)block[32 + q_base + l] >> shift) & 0x03u);
    const int high1 = (int)(((uint)block[l] >> hm_bit) & 0x01u);
    const int q = low2 + high1 * 4 - 4;
    const float scale = antfly_half_le_to_float(block + 108) * (float)antfly_q3_k_raw_scale(block + 96, sub);
    return scale * (float)q;
}

kernel void antfly_q3_k_small_batch_bias_gelu_msl_v1(
    const device float *input [[buffer(0)]],
    const device uchar *weight_q3_k [[buffer(1)]],
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
    if (tid < 32) {
        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
            const device uchar *block = weight_q3_k + ((col * block_count + block_idx) * 110);
            for (int lane = (int)tid; lane < 256; lane += 32) {
                acc += input[row * in_dim + (block_idx << 8) + lane] * antfly_q3_k_dequant_lane(block, lane);
            }
        }
    }

    acc = simd_sum(acc);
    if (tid == 0) output[row * out_dim + col] = antfly_gelu(acc + bias[col]);
}
