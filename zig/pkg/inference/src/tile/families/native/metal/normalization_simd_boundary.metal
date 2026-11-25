// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
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

// native-program:entry termite_apply_rms_norm_1x_reduce_simd_tree
// native-program:indent 11
kernel void termite_apply_rms_norm_1x_reduce_simd_tree(device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]], device float *output [[buffer(2)]], constant termite_metal_apply_layer_norm_params &p [[buffer(3)]], threadgroup float *shmem [[threadgroup(0)]], uint tid [[thread_index_in_threadgroup]], uint tpg [[threads_per_threadgroup]]) {
    if (p.hidden_size == 0) return;
    float mean_square = 0.0f;
    for (uint i = tid; i < p.hidden_size; i += tpg) { float x = input[i]; mean_square += x * x; }
    shmem[tid] = mean_square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    termite_rms_tree_sum_simd(shmem, tid, tpg);
    float inv_rms = rsqrt(shmem[0] / float(p.hidden_size) + p.eps);
    for (uint i = tid; i < p.hidden_size; i += tpg) output[i] = input[i] * inv_rms * weight[i];
}
template<uint FixedWidth = 0u> inline float antfly_row_rms_simd_inverse(device const float *input, uint base, uint width, float eps, uint lane) {
    uint dim = FixedWidth ? FixedWidth : width;
    float sum = 0.0f;
    for(uint d=lane;d<dim;d+=32u) { float x=input[base+d]; sum+=x*x; }
    return rsqrt(simd_sum(sum)/float(dim)+eps);
}
template<uint H> inline void termite_lightonocr3_norm_cached_body(device const float4 *x, device const float4 *w, device float4 *y, constant termite_metal_apply_row_norm_params &p, threadgroup float *partial, uint tid, uint sg, uint lane, uint row) {
    if (row >= p.rows) return;
    constexpr uint count = H / (4 * 128);
    float4 stored[count];
    float sum = 0.0f;
    #pragma unroll
    for (uint i = 0; i < count; i++) {
        uint col = tid + i * 128;
        stored[i] = x[row * (H / 4) + col];
        sum += dot(stored[i], stored[i]);
    }
    sum = simd_sum(sum);
    if (lane == 0) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = rsqrt(simd_sum(lane < 4 ? partial[lane] : 0.0f) / float(H) + p.eps);
    #pragma unroll
    for (uint i = 0; i < count; i++) {
        uint col = tid + i * 128;
        y[row * (H / 4) + col] = stored[i] * (scale * w[col]);
    }
}
// native-program:entry termite_apply_rms_norm_add_1x_simd_tree
// native-program:indent 11
kernel void termite_apply_rms_norm_add_1x_simd_tree(device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]], device const float *residual [[buffer(2)]], device float *output [[buffer(3)]], constant termite_metal_apply_layer_norm_params &p [[buffer(4)]], threadgroup float *shmem [[threadgroup(0)]], uint tid [[thread_index_in_threadgroup]], uint tpg [[threads_per_threadgroup]]) {
    if (p.hidden_size == 0) return;
    float mean_square = 0.0f;
    for (uint i = tid; i < p.hidden_size; i += tpg) { float x = input[i]; mean_square += x * x; }
    shmem[tid] = mean_square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    termite_rms_tree_sum_simd(shmem, tid, tpg);
    mean_square = shmem[0] / float(p.hidden_size);
    float inv_rms = rsqrt(mean_square + p.eps);
    for (uint i = tid; i < p.hidden_size; i += tpg) output[i] = input[i] * inv_rms * weight[i] + residual[i];
}
// native-program:entry termite_dual_rms_residual_1x_simd_tree
// native-program:indent 11
kernel void termite_dual_rms_residual_1x_simd_tree(device const float *input [[buffer(0)]], device const float *post_weight [[buffer(1)]], device const float *residual [[buffer(2)]], device const float *pre_weight [[buffer(3)]], device float *residual_output [[buffer(4)]], device float *normalized_output [[buffer(5)]], constant termite_metal_apply_layer_norm_params &p [[buffer(6)]], threadgroup float *shmem [[threadgroup(0)]], uint tid [[thread_index_in_threadgroup]], uint tpg [[threads_per_threadgroup]]) {
    if (p.hidden_size == 0) return;
    float mean_square = 0.0f;
    for (uint i = tid; i < p.hidden_size; i += tpg) { float x = input[i]; mean_square += x * x; }
    shmem[tid] = mean_square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    termite_rms_tree_sum_simd(shmem, tid, tpg);
    float inv_rms = rsqrt(shmem[0] / float(p.hidden_size) + p.eps);
    float residual_mean_square = 0.0f;
    for (uint i = tid; i < p.hidden_size; i += tpg) { float x = input[i] * inv_rms * post_weight[i] + residual[i]; residual_output[i] = x; residual_mean_square += x * x; }
// native-program:entry termite_apply_rms_norm_add_scale_1x_simd_tree
// native-program:indent 11
    threadgroup_barrier(mem_flags::mem_threadgroup);
    shmem[tid] = residual_mean_square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    termite_rms_tree_sum_simd(shmem, tid, tpg);
    float residual_inv_rms = rsqrt(shmem[0] / float(p.hidden_size) + p.eps);
    for (uint i = tid; i < p.hidden_size; i += tpg) normalized_output[i] = residual_output[i] * residual_inv_rms * pre_weight[i];
}
kernel void termite_apply_rms_norm_add_scale_1x_simd_tree(device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]], device const float *residual [[buffer(2)]], device float *output [[buffer(3)]], constant termite_metal_apply_layer_norm_scale_params &p [[buffer(4)]], threadgroup float *shmem [[threadgroup(0)]], uint tid [[thread_index_in_threadgroup]], uint tpg [[threads_per_threadgroup]]) {
    if (p.hidden_size == 0) return;
    float mean_square = 0.0f;
    for (uint i = tid; i < p.hidden_size; i += tpg) { float x = input[i]; mean_square += x * x; }
    shmem[tid] = mean_square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    termite_rms_tree_sum_simd(shmem, tid, tpg);
    mean_square = shmem[0] / float(p.hidden_size);
    float inv_rms = rsqrt(mean_square + p.eps);
    for (uint i = tid; i < p.hidden_size; i += tpg) output[i] = (input[i] * inv_rms * weight[i] + residual[i]) * p.scale;
}
