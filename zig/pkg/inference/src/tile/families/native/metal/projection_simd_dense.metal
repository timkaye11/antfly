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

// native-program:entry termite_apply_linear_f16_1x_reduce_simd
// native-program:indent 11
kernel void termite_apply_linear_f16_1x_reduce_simd(device const float *input [[buffer(0)]], device const half *weight [[buffer(1)]], device const float *bias [[buffer(2)]], device float *output [[buffer(3)]], constant termite_metal_apply_linear_params &p [[buffer(4)]], uint lane [[thread_index_in_simdgroup]], uint group [[threadgroup_position_in_grid]]) {
    if (group >= p.out_dim) return;
    float acc = lane == 0u ? bias[group] : 0.0f;
    uint row_base = group * p.in_dim;
    for (uint i = lane; i < p.in_dim; i += 32u) acc += input[i] * float(weight[row_base + i]);
    acc += simd_shuffle_down(acc, 16u); acc += simd_shuffle_down(acc, 8u);
    acc += simd_shuffle_down(acc, 4u); acc += simd_shuffle_down(acc, 2u); acc += simd_shuffle_down(acc, 1u);
    if (lane == 0u) output[group] = acc;
}
