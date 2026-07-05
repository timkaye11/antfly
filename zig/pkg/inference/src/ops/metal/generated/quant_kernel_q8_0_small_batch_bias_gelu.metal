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

// Generated Metal production artifact from graph/quant_kernel_compiler.zig.
// plan_id=metal/q8_0/rows_2_8/bias_gelu/small_batch
// kernel_id=antfly_q8_0_small_batch_bias_gelu_msl_v1
// production_baseline=metal_handwritten_quant_matmul
// production_enabled=true
// Promoted after Metal runtime correctness and sequential speedup gates.

#include <metal_stdlib>
using namespace metal;

static inline float antfly_half_le_to_float(const device uchar *p) {
    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    return (float)as_type<half>(bits);
}

static inline float antfly_q8_0_dequant_lane(const device uchar *block, int lane) {
    const float d = antfly_half_le_to_float(block);
    const int q = (int)as_type<char>(block[2 + lane]);
    return d * (float)q;
}

static inline float antfly_gelu(float x) {
    return 0.5f * x * (1.0f + fast::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

kernel void antfly_q8_0_small_batch_bias_gelu_msl_v1(
    const device float *input [[buffer(0)]],
    const device uchar *weight_q8_0 [[buffer(1)]],
    const device float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant int &rows [[buffer(4)]],
    constant int &in_dim [[buffer(5)]],
    constant int &out_dim [[buffer(6)]],
    uint3 thread_pos [[thread_position_in_threadgroup]],
    uint3 group_pos [[threadgroup_position_in_grid]]
) {
    const uint tid = thread_pos.x;
    const int col0 = (int)(group_pos.x << 1);
    const int col1 = col0 + 1;
    const int row = (int)group_pos.y;
    if (row >= rows || rows < 2 || rows > 8 || col0 >= out_dim || (in_dim & 31) != 0) return;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const int block_count = in_dim >> 5;
    const int lane = (int)tid;
    const device float *row_input = input + row * in_dim;
    const device uchar *col0_weight = weight_q8_0 + col0 * block_count * 34;
    const bool has_col1 = col1 < out_dim;
    const device uchar *col1_weight = has_col1 ? weight_q8_0 + col1 * block_count * 34 : col0_weight;
    for (int block_idx = 0; block_idx < block_count; ++block_idx) {
        const float x = row_input[(block_idx << 5) + lane];
        const device uchar *block0 = col0_weight + block_idx * 34;
        acc0 += x * antfly_q8_0_dequant_lane(block0, lane);
        if (has_col1) {
            const device uchar *block1 = col1_weight + block_idx * 34;
            acc1 += x * antfly_q8_0_dequant_lane(block1, lane);
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (tid == 0) {
        output[row * out_dim + col0] = antfly_gelu(acc0 + bias[col0]);
        if (has_col1) output[row * out_dim + col1] = antfly_gelu(acc1 + bias[col1]);
    }
}
