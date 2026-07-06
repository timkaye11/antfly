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
// plan_id=metal/q2_k/rows_2_8/none/small_batch
// kernel_id=antfly_q2_k_small_batch_msl_v1
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

static inline float antfly_q2_k_dequant_lane(const device uchar *block, int lane) {
    const uint sub = (uint)lane >> 4;
    const uint i = (uint)lane & 15u;
    const uchar scale_byte = block[sub];
    const float dsc = antfly_half_le_to_float(block + 16) * (float)(scale_byte & 0x0Fu);
    const float dmn = antfly_half_le_to_float(block + 18) * (float)(scale_byte >> 4);
    const uint chunk = sub >> 3;
    const uint group = (sub & 7u) >> 1;
    const uint l_base = (sub & 1u) << 4;
    const uint q_base = chunk << 5;
    const uint shift = group << 1;
    const uint q = ((uint)block[20 + q_base + l_base + i] >> shift) & 0x03u;
    return dsc * (float)q - dmn;
}

kernel void antfly_q2_k_small_batch_msl_v1(
    const device float *input [[buffer(0)]],
    const device uchar *weight_q2_k [[buffer(1)]],
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
    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 255) != 0) return;

    float acc = 0.0f;
    const int block_count = in_dim >> 8;
    if (tid < 32) {
        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
            const device uchar *block = weight_q2_k + ((col * block_count + block_idx) * 84);
            for (int lane = (int)tid; lane < 256; lane += 32) {
                acc += input[row * in_dim + (block_idx << 8) + lane] * antfly_q2_k_dequant_lane(block, lane);
            }
        }
    }

    acc = simd_sum(acc);
    if (tid == 0) output[row * out_dim + col] = acc;
}
