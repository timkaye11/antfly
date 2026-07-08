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
// plan_id=metal/q6_k/rows_2_8/bias/small_batch
// kernel_id=antfly_q6_k_small_batch_bias_msl_v1
// production_baseline=metal_handwritten_quant_matmul
// production_enabled=true
// Promoted after sequential Metal runtime evidence cleared correctness,
// route, provider-route, and speedup gates.

#include <metal_stdlib>
using namespace metal;

inline float antfly_qk_half_le_to_float(device const uchar *p) { ushort bits = (ushort(p[0]) | (ushort(p[1]) << 8)); return float(as_type<half>(bits)); }
inline float antfly_q6_k_dequant_lane(device const uchar *block, int lane) {
    device const uchar *ql = block; device const uchar *qh = block + 128; device const uchar *scales = block + 192; int sub = lane >> 4; int i = lane & 15; int half_idx = sub >> 3; int group = (sub & 7) >> 1; int l = ((sub & 1) << 4) + i;
    int ql_off = half_idx * 64 + (group & 1) * 32; int qh_off = half_idx * 32; int qh_shift = group * 2; int nibble_shift = (group >> 1) * 4; int low4 = int((ql[ql_off + l] >> nibble_shift) & 0x0fu); int high2 = int((qh[qh_off + l] >> qh_shift) & 0x03u);
    int scale_u = int(scales[sub]); int scale = scale_u >= 128 ? scale_u - 256 : scale_u; return antfly_qk_half_le_to_float(block + 208) * float(scale) * float((low4 | (high2 << 4)) - 32);
}
kernel void antfly_q6_k_small_batch_bias_msl_v1(device const float *input [[buffer(0)]], device const uchar *weight_q6_k [[buffer(1)]], device const float *bias [[buffer(2)]], device float *output [[buffer(3)]], constant int &rows [[buffer(4)]], constant int &in_dim [[buffer(5)]], constant int &out_dim [[buffer(6)]], uint3 thread_pos [[thread_position_in_threadgroup]], uint3 group_pos [[threadgroup_position_in_grid]], ushort lane_id [[thread_index_in_simdgroup]], ushort simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    uint tid = thread_pos.x; int col = int(group_pos.x); int row = int(group_pos.y); if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 255) != 0) return; float acc = 0.0f; int block_count = in_dim >> 8;
    for (int block_idx = 0; block_idx < block_count; ++block_idx) { device const uchar *block = weight_q6_k + ((col * block_count + block_idx) * 210); int base = block_idx << 8; for (int lane = int(tid); lane < 256; lane += 256) acc += input[row * in_dim + base + lane] * antfly_q6_k_dequant_lane(block, lane); }
    threadgroup float partial[32]; acc = simd_sum(acc); if (lane_id == 0u) partial[simdgroup_id] = acc; if (simdgroup_id == 0u && lane_id >= 8u) partial[lane_id] = 0.0f; threadgroup_barrier(mem_flags::mem_threadgroup); float total = simd_sum(partial[lane_id]); if (lane_id == 0u && simdgroup_id == 0u) output[row * out_dim + col] = total + bias[col];
}
