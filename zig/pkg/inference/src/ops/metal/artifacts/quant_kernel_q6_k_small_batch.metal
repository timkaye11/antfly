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
// plan_id=metal/q6_k/rows_2_8/none/small_batch
// kernel_id=antfly_q6_k_small_batch_msl_v1
// production_baseline=metal_handwritten_quant_matmul
// production_enabled=true
// Production Metal dispatch uses this checked-in artifact after
// correctness and benchmark gates.

#include <metal_stdlib>
using namespace metal;

static inline float antfly_half_le_to_float(const device uchar *p) {
    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    return (float)as_type<half>(bits);
}

static inline float antfly_q6_k_dequant_lane(const device uchar *block, int lane) {
    const device uchar *ql = block;
    const device uchar *qh = block + 128;
    const device uchar *scales = block + 192;
    const device uchar *d = block + 208;
    const int sub = lane >> 4;
    const int i = lane & 15;
    const int half_idx = sub >> 3;
    const int group = (sub & 7) >> 1;
    const int l = ((sub & 1) << 4) + i;
    const int ql_off = half_idx * 64 + (group & 1) * 32;
    const int qh_off = half_idx * 32;
    const int qh_shift = group * 2;
    const int nibble_shift = (group >> 1) * 4;
    const int low4 = (int)((ql[ql_off + l] >> nibble_shift) & 0x0fu);
    const int high2 = (int)((qh[qh_off + l] >> qh_shift) & 0x03u);
    const int q = (low4 | (high2 << 4)) - 32;
    const int scale_u = (int)scales[sub];
    const int scale = scale_u >= 128 ? scale_u - 256 : scale_u;
    return antfly_half_le_to_float(d) * (float)scale * (float)q;
}

kernel void antfly_q6_k_small_batch_msl_v1(
    const device float *input [[buffer(0)]],
    const device uchar *weight_q6_k [[buffer(1)]],
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
    if (tid < 128) {
        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
            const device uchar *block = weight_q6_k + ((col * block_count + block_idx) * 210);
            const int base = block_idx << 8;
            for (int lane = (int)tid; lane < 256; lane += 128) {
                acc += input[row * in_dim + base + lane] * antfly_q6_k_dequant_lane(block, lane);
            }
        }
    }

    threadgroup float partial[128];
    if (tid < 128) partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[row * out_dim + col] = partial[0];
}
