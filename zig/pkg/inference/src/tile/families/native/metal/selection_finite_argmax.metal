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

// native-program:entry termite_argmax_logits_finite_1x
// native-program:indent 11
inline float termite_qualification_expm1_f32(float input) {
    // Private finite [-0.511, 20] input from the f32 tanh branches below.
    const float ln2_hi = 6.9313812256e-01f;
    const float ln2_lo = 9.0580006145e-06f;
    const float invln2 = 1.4426950216e+00f;
    const float q1 = -3.3333212137e-2f;
    const float q2 = 1.5807170421e-3f;
    float x = input;
    uint bits = as_type<uint>(x);
    uint magnitude = bits & 0x7fffffffu;
    bool negative = (bits >> 31u) != 0u;
    int k = 0;
    float correction = 0.0f;
    if (magnitude > 0x3eb17218u) {
        float high, low;
        if (magnitude < 0x3f851592u) {
            if (!negative) { high = x - ln2_hi; low = ln2_lo; k = 1; }
            else { high = x + ln2_hi; low = -ln2_lo; k = -1; }
        } else {
            float rounded = invln2 * x;
            rounded += negative ? -0.5f : 0.5f;
            k = int(rounded);
            float scale = float(k);
            high = x - scale * ln2_hi;
            low = scale * ln2_lo;
        }
        x = high - low;
        correction = (high - x) - low;
    } else if (magnitude < 0x33000000u) { return x; }
    float half_x = 0.5f * x;
    float half_x_squared = x * half_x;
    float r = 1.0f + half_x_squared * (q1 + half_x_squared * q2);
    float denominator = 3.0f - r * half_x;
    float error = half_x_squared * ((r - denominator) / (6.0f - x * denominator));
    if (k == 0) return x - (x * error - half_x_squared);
    error = x * (error - correction) - correction;
    error -= half_x_squared;
    if (k == -1) return 0.5f * (x - error) - 0.5f;
    if (k == 1) return x < -0.25f ? -2.0f * (error - (x + 0.5f)) : 1.0f + 2.0f * (x - error);
    float two_to_k = as_type<float>(uint(127 + k) << 23u);
    float two_to_minus_k = as_type<float>(uint(127 - k) << 23u);
    if (k < 23) return (x - error + (1.0f - two_to_minus_k)) * two_to_k;
    return (x - (error + two_to_minus_k) + 1.0f) * two_to_k;
}
inline float termite_qualification_final_softcap(float value, float softcap) {
    if (softcap <= 0.0f) return value;
    float scaled = value / softcap;
    uint bits = as_type<uint>(scaled);
    uint magnitude = bits & 0x7fffffffu;
    float absolute = as_type<float>(magnitude);
    float result;
    if (magnitude > 0x3f0c9f54u) {
        if (magnitude > 0x41200000u) result = 1.0f;
        else { float e = termite_qualification_expm1_f32(2.0f * absolute); result = 1.0f - 2.0f / (e + 2.0f); }
    } else if (magnitude > 0x3e82c578u) {
        float e = termite_qualification_expm1_f32(2.0f * absolute);
        result = e / (e + 2.0f);
    } else if (magnitude >= 0x00800000u) {
        float e = termite_qualification_expm1_f32(-2.0f * absolute);
        result = -e / (e + 2.0f);
    } else result = absolute;
    return ((bits >> 31u) != 0u ? -result : result) * softcap;
}
kernel void termite_argmax_logits_finite_1x(device const float *logits [[buffer(0)]], device uint *output [[buffer(1)]], constant uint &out_dim [[buffer(2)]], constant float &softcap [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid != 0u) return;
    float best = -INFINITY;
    uint best_idx = 0xffffffffu;
    for (uint i = 0u; i < out_dim; ++i) {
        float value = logits[i];
        if ((as_type<uint>(value) & 0x7f800000u) == 0x7f800000u) { output[0] = 0xffffffffu; return; }
        value = termite_qualification_final_softcap(value, softcap);
        if (value > best || (value == best && i < best_idx)) { best = value; best_idx = i; }
    }
    output[0] = best_idx;
}
// native-program:entry termite_argmax_logits_finite_partials
// native-program:indent 11
kernel void termite_argmax_logits_finite_partials(device const float *logits [[buffer(0)]], device float *partial_values [[buffer(1)]], device uint *partial_ids [[buffer(2)]], constant uint &out_dim [[buffer(3)]], constant float &softcap [[buffer(4)]], threadgroup float *sh_vals [[threadgroup(0)]], threadgroup uint *sh_ids [[threadgroup(1)]], threadgroup uint *sh_bad [[threadgroup(2)]], ushort tid [[thread_index_in_threadgroup]], ushort tg_size [[threads_per_threadgroup]], uint block [[threadgroup_position_in_grid]]) {
    uint start = block * 1024u;
    uint count = min(1024u, out_dim - start);
    float best = -INFINITY;
    uint best_idx = 0xffffffffu;
    uint bad = 0u;
    for (uint offset = uint(tid); offset < count; offset += uint(tg_size)) {
        uint i = start + offset;
        float value = logits[i];
        if ((as_type<uint>(value) & 0x7f800000u) == 0x7f800000u) { bad = 1u; continue; }
        value = termite_qualification_final_softcap(value, softcap);
        if (value > best || (value == best && i < best_idx)) { best = value; best_idx = i; }
    }
    sh_vals[tid] = best; sh_ids[tid] = best_idx; sh_bad[tid] = bad;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = uint(tg_size) >> 1u; stride > 0u; stride >>= 1u) {
        if (uint(tid) < stride) {
            float other = sh_vals[uint(tid) + stride];
            uint other_idx = sh_ids[uint(tid) + stride];
            sh_bad[tid] |= sh_bad[uint(tid) + stride];
            if (other > sh_vals[tid] || (other == sh_vals[tid] && other_idx < sh_ids[tid])) { sh_vals[tid] = other; sh_ids[tid] = other_idx; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) { partial_values[block] = sh_bad[0] != 0u ? as_type<float>(0x7fc00000u) : sh_vals[0]; partial_ids[block] = sh_bad[0] != 0u ? 0xffffffffu : sh_ids[0]; }
}
// native-program:entry termite_argmax_logits_finite_reduce
// native-program:indent 11
kernel void termite_argmax_logits_finite_reduce(device const float *partial_values [[buffer(0)]], device const uint *partial_ids [[buffer(1)]], device uint *output [[buffer(2)]], constant uint &partial_count [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid != 0u) return;
    float best = -INFINITY;
    uint best_idx = 0xffffffffu;
    bool found = false;
    for (uint i = 0u; i < partial_count; ++i) {
        uint idx = partial_ids[i];
        float value = partial_values[i];
        uint bits = as_type<uint>(value);
        if ((bits & 0x7fffffffu) > 0x7f800000u) { output[0] = 0xffffffffu; return; }
        if (idx == 0xffffffffu) continue;
        found = true;
        if (value > best || (value == best && idx < best_idx)) { best = value; best_idx = idx; }
    }
    output[0] = found ? best_idx : 0xffffffffu;
}
inline bool termite_argmax_suppressed_token(uint token_id, device const int *suppress_ids, uint suppress_count) {
    for (uint i = 0u; i < suppress_count; ++i) {
        int raw = suppress_ids[i];
        if (raw >= 0 && uint(raw) == token_id) return true;
    }
    return false;
}
// native-program:entry termite_argmax_logits_finite_suppress_partials
// native-program:indent 11
kernel void termite_argmax_logits_finite_suppress_partials(device const float *logits [[buffer(0)]], device const int *suppress_ids [[buffer(1)]], device float *partial_values [[buffer(2)]], device uint *partial_ids [[buffer(3)]], constant termite_metal_argmax_suppress_params &p [[buffer(4)]], constant float &softcap [[buffer(5)]], threadgroup float *sh_vals [[threadgroup(0)]], threadgroup uint *sh_ids [[threadgroup(1)]], threadgroup uint *sh_bad [[threadgroup(2)]], ushort tid [[thread_index_in_threadgroup]], ushort tg_size [[threads_per_threadgroup]], uint block [[threadgroup_position_in_grid]]) {
    const uint block_size = 1024u;
    uint start = block * block_size;
    uint count = min(block_size, p.out_dim - start);
    float best = -INFINITY;
    uint best_idx = 0xffffffffu;
    uint bad = 0u;
    for (uint offset = uint(tid); offset < count; offset += uint(tg_size)) {
        uint i = start + offset;
        float value = logits[i];
        if ((as_type<uint>(value) & 0x7f800000u) == 0x7f800000u) { bad = 1u; continue; }
        value = termite_qualification_final_softcap(value, softcap);
        if (termite_argmax_suppressed_token(i, suppress_ids, p.suppress_count)) continue;
        if (value > best || (value == best && i < best_idx)) { best = value; best_idx = i; }
    }
    sh_vals[tid] = best; sh_ids[tid] = best_idx; sh_bad[tid] = bad;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = uint(tg_size) >> 1u; stride > 0u; stride >>= 1u) {
        if (uint(tid) < stride) {
            float other = sh_vals[uint(tid) + stride];
            uint other_idx = sh_ids[uint(tid) + stride];
            sh_bad[tid] |= sh_bad[uint(tid) + stride];
            if (other > sh_vals[tid] || (other == sh_vals[tid] && other_idx < sh_ids[tid])) { sh_vals[tid] = other; sh_ids[tid] = other_idx; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) { partial_values[block] = sh_bad[0] != 0u ? as_type<float>(0x7fc00000u) : sh_vals[0]; partial_ids[block] = sh_bad[0] != 0u ? 0xffffffffu : sh_ids[0]; }
}
inline void termite_whisper_lse_push(thread float &m, thread float &s, float v) {
