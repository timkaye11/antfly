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
// plan_id=metal/q4_k/rows_2_8/bias/small_batch
// kernel_id=antfly_q4_k_small_batch_bias_msl_v1
// production_baseline=metal_handwritten_quant_matmul
// production_enabled=true
// Promoted after sequential Metal runtime evidence cleared correctness,
// route, provider-route, and speedup gates.

#include <metal_stdlib>
using namespace metal;

inline float antfly_q4_k_half_le_to_float(device const uchar *p) { ushort bits = (ushort(p[0]) | (ushort(p[1]) << 8)); return float(as_type<half>(bits)); }
inline void antfly_q4_k_unpack_scale_min(device const uchar *scales, int sub, thread float &scale, thread float &min_v) {
    if (sub < 4) { scale = float(scales[sub] & 63u); min_v = float(scales[sub + 4] & 63u); return; }
    scale = float((scales[sub + 4] & 0x0fu) | ((scales[sub - 4] >> 6) << 4)); min_v = float((scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4));
}
inline float antfly_q4_k_dequant_lane(device const uchar *block, int lane) {
    device const uchar *scales = block + 4; device const uchar *qs = block + 16; int sub = lane >> 5; int q_index = (sub >> 1) * 32 + (lane & 31);
    uchar packed = qs[q_index]; int q = (sub & 1) == 0 ? int(packed & 0x0fu) : int(packed >> 4);
    float raw_scale = 0.0f; float raw_min = 0.0f; antfly_q4_k_unpack_scale_min(scales, sub, raw_scale, raw_min);
    return antfly_q4_k_half_le_to_float(block) * raw_scale * float(q) - antfly_q4_k_half_le_to_float(block + 2) * raw_min;
}
kernel void antfly_q4_k_small_batch_bias_msl_v1(device const float *input [[buffer(0)]], device const uchar *weight_q4_k [[buffer(1)]], device const float *bias [[buffer(2)]], device float *output [[buffer(3)]], constant int &rows [[buffer(4)]], constant int &in_dim [[buffer(5)]], constant int &out_dim [[buffer(6)]], uint3 thread_pos [[thread_position_in_threadgroup]], uint3 group_pos [[threadgroup_position_in_grid]]) {
    uint tid = thread_pos.x; int col = int(group_pos.x); int row = int(group_pos.y); if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 255) != 0) return; threadgroup float partial[64]; float acc = 0.0f; int block_count = in_dim >> 8;
    if (tid < 64) { for (int block_idx = 0; block_idx < block_count; ++block_idx) { device const uchar *block = weight_q4_k + ((col * block_count + block_idx) * 144); int base = block_idx << 8; for (int lane = int(tid); lane < 256; lane += 64) acc += input[row * in_dim + base + lane] * antfly_q4_k_dequant_lane(block, lane); } }
    if (tid < 64) partial[tid] = acc; threadgroup_barrier(mem_flags::mem_threadgroup); for (uint stride = 32; stride > 0; stride >>= 1) { if (tid < stride) partial[tid] += partial[tid + stride]; threadgroup_barrier(mem_flags::mem_threadgroup); } if (tid == 0) output[row * out_dim + col] = partial[0] + bias[col];
}
