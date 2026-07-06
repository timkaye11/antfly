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

// Generated Metal artifact source from graph/quant_kernel_compiler.zig.
// plan_id=metal/q8_1/rows_2_8/none/small_batch
// kernel_id=antfly_q8_1_small_batch_msl_v1
// production_baseline=metal_handwritten_quant_matmul
// production_enabled=true
// Promoted after sequential Metal runtime evidence cleared correctness,
// route, and speedup gates.

#include <metal_stdlib>
using namespace metal;

static inline float antfly_half_le_to_float(const device uchar *p) {
    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    return (float)as_type<half>(bits);
}

static inline float antfly_q8_1_dequant_lane(const device uchar *block, int lane) {
    const float d = antfly_half_le_to_float(block);
    const int q = (int)as_type<char>(block[4 + lane]);
    return d * (float)q;
}

kernel void antfly_q8_1_small_batch_msl_v1(
    const device float *input [[buffer(0)]],
    const device uchar *weight_q8_1 [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant int &rows [[buffer(3)]],
    constant int &in_dim [[buffer(4)]],
    constant int &out_dim [[buffer(5)]],
    uint3 thread_pos [[thread_position_in_threadgroup]],
    uint3 group_pos [[threadgroup_position_in_grid]]
) {
    const uint tid = thread_pos.x;
    const int col = (int)group_pos.x;
    const int row = (int)group_pos.y;
    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 31) != 0) return;

    float acc = 0.0f;
    const int block_count = in_dim >> 5;
    if (tid < 32) {
        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
            const device uchar *block = weight_q8_1 + ((col * block_count + block_idx) * 36);
            const int lane = (int)tid;
            acc += input[row * in_dim + (block_idx << 5) + lane] * antfly_q8_1_dequant_lane(block, lane);
        }
    }

    acc = simd_sum(acc);
    if (tid == 0) output[row * out_dim + col] = acc;
}
