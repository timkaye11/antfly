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

//! Descriptor-driven renderer for generated Metal small-batch quant matmul
//! kernels. A kernel is `render(kernel_id, decoder, schedule, epilogue)`:
//! one canonical skeleton parametrized by the launch schedule (threads/cols/
//! reduction), with the per-format dequant math supplied as a leaf MSL fragment
//! (`FormatDecoder.lane_decode_msl`).
//!
//! This produces "v2" kernels. They are NOT byte-identical to the frozen v1
//! bodies — the skeleton is normalized (redundant `if (tid < N)` guards dropped
//! in favor of the strided lane loop's natural bounds; a single shared gelu /
//! half-load vocabulary; consistent `partial[]` placement). Every rendered
//! kernel earns production through the existing evidence gates against its v1
//! baseline, so semantic parity is proven by benchmark + CPU conformance, not
//! by construction.

const std = @import("std");
const compiler = @import("quant_kernel_compiler.zig");
const quant_matmul = @import("quant_matmul.zig");

const Epilogue = compiler.Epilogue;
const KernelSchedule = compiler.KernelSchedule;
const ReductionKind = compiler.ReductionKind;

/// A shared MSL vocabulary helper (dedup of the per-format v1 copies).
pub const HelperFragment = struct {
    name: []const u8,
    msl: []const u8,
};

/// Per-format dequant descriptor. `lane_decode_msl` is a leaf fragment with the
/// fixed contract `inline float <lane_decode_fn>(device const uchar *block,
/// int lane)` — a pure function of the block bytes and the lane index, validated
/// against the CPU reference by conformance tests. `helpers` are the shared
/// vocabulary fragments the fragment references, in dependency order.
pub const FormatDecoder = struct {
    format: quant_matmul.Format,
    weight_param: []const u8, // e.g. "weight_q4_0"
    lane_decode_fn: []const u8, // e.g. "antfly_q4_0_dequant_lane_v2"
    lane_decode_msl: []const u8,
    helpers: []const HelperFragment,
};

// ---- Shared vocabulary helpers -------------------------------------------

pub const helper_qk_half_le_to_float = HelperFragment{
    .name = "antfly_qk_half_le_to_float",
    .msl = "inline float antfly_qk_half_le_to_float(device const uchar *p) { ushort bits = (ushort(p[0]) | (ushort(p[1]) << 8)); return float(as_type<half>(bits)); }",
};

pub const helper_qk_f32_le_to_float = HelperFragment{
    .name = "antfly_qk_f32_le_to_float",
    .msl = "inline float antfly_qk_f32_le_to_float(device const uchar *p) { uint bits = uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24); return as_type<float>(bits); }",
};

pub const helper_qk_u32_le = HelperFragment{
    .name = "antfly_qk_u32_le",
    .msl = "inline uint antfly_qk_u32_le(device const uchar *p) { return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24); }",
};

pub const helper_qk_gelu = HelperFragment{
    .name = "antfly_qk_gelu",
    .msl = "inline float antfly_qk_gelu(float x) { float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x); return 0.5f * x * (1.0f + fast::tanh(inner)); }",
};

pub const helper_qk_unpack_scale_min_6bit = HelperFragment{
    .name = "antfly_qk_unpack_scale_min_6bit",
    .msl =
    \\inline void antfly_qk_unpack_scale_min_6bit(device const uchar *scales, int sub, thread float &scale, thread float &min_v) {
    \\    if (sub < 4) { scale = float(scales[sub] & 63u); min_v = float(scales[sub + 4] & 63u); return; }
    \\    scale = float((scales[sub + 4] & 0x0fu) | ((scales[sub - 4] >> 6) << 4)); min_v = float((scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4));
    \\}
    ,
};

pub const helper_q3_k_raw_scale = HelperFragment{
    .name = "antfly_q3_k_raw_scale",
    .msl =
    \\inline int antfly_q3_k_raw_scale(device const uchar *scale_data, uint sub) {
    \\    uint i = sub & 3u; uint low = 0u; uint high = 0u;
    \\    if (sub < 4u) { low = uint(scale_data[i] & 0x0fu); high = uint(scale_data[8u + i] & 0x03u); }
    \\    else if (sub < 8u) { low = uint(scale_data[4u + i] & 0x0fu); high = uint((scale_data[8u + i] >> 2) & 0x03u); }
    \\    else if (sub < 12u) { low = uint((scale_data[i] >> 4) & 0x0fu); high = uint((scale_data[8u + i] >> 4) & 0x03u); }
    \\    else { low = uint((scale_data[4u + i] >> 4) & 0x0fu); high = uint((scale_data[8u + i] >> 6) & 0x03u); }
    \\    return int(low | (high << 4)) - 32;
    \\}
    ,
};

// ---- Per-format decoders (v2 fragments) ----------------------------------
// These are the v1 dequant helpers with the lane fn renamed `_v2`, per-format
// `half_le_to_float`/`u32_le` calls repointed at the shared `antfly_qk_*`
// vocabulary, and q8_0 inlined so it no longer needs the external
// `termite_q8_0_block_scale` helper.

pub const decoder_q4_0 = FormatDecoder{
    .format = .q4_0,
    .weight_param = "weight_q4_0",
    .lane_decode_fn = "antfly_q4_0_dequant_lane_v2",
    .lane_decode_msl = "inline float antfly_q4_0_dequant_lane_v2(device const uchar *block, int lane) { float d = antfly_qk_half_le_to_float(block); int packed_index = lane & 15; uchar packed = block[2 + packed_index]; int q = lane < 16 ? int(packed & 0x0fu) - 8 : int(packed >> 4) - 8; return d * float(q); }",
    .helpers = &.{helper_qk_half_le_to_float},
};

pub const decoder_q4_1 = FormatDecoder{
    .format = .q4_1,
    .weight_param = "weight_q4_1",
    .lane_decode_fn = "antfly_q4_1_dequant_lane_v2",
    .lane_decode_msl = "inline float antfly_q4_1_dequant_lane_v2(device const uchar *block, int lane) { float d = antfly_qk_half_le_to_float(block); float m = antfly_qk_half_le_to_float(block + 2); int packed_index = lane & 15; uchar packed = block[4 + packed_index]; int q = lane < 16 ? int(packed & 0x0fu) : int(packed >> 4); return d * float(q) + m; }",
    .helpers = &.{helper_qk_half_le_to_float},
};

pub const decoder_q5_0 = FormatDecoder{
    .format = .q5_0,
    .weight_param = "weight_q5_0",
    .lane_decode_fn = "antfly_q5_0_dequant_lane_v2",
    .lane_decode_msl = "inline float antfly_q5_0_dequant_lane_v2(device const uchar *block, int lane) { float d = antfly_qk_half_le_to_float(block); uint qh = antfly_qk_u32_le(block + 2); int packed_index = lane & 15; uchar packed = block[6 + packed_index]; int low4 = lane < 16 ? int(packed & 0x0fu) : int(packed >> 4); int high = int((qh >> uint(lane)) & 1u); return d * float((low4 | (high << 4)) - 16); }",
    .helpers = &.{ helper_qk_half_le_to_float, helper_qk_u32_le },
};

pub const decoder_q5_1 = FormatDecoder{
    .format = .q5_1,
    .weight_param = "weight_q5_1",
    .lane_decode_fn = "antfly_q5_1_dequant_lane_v2",
    .lane_decode_msl = "inline float antfly_q5_1_dequant_lane_v2(device const uchar *block, int lane) { float d = antfly_qk_half_le_to_float(block); float m = antfly_qk_half_le_to_float(block + 2); uint qh = antfly_qk_u32_le(block + 4); int packed_index = lane & 15; uchar packed = block[8 + packed_index]; int low4 = lane < 16 ? int(packed & 0x0fu) : int(packed >> 4); int high = int((qh >> uint(lane)) & 1u); return d * float(low4 | (high << 4)) + m; }",
    .helpers = &.{ helper_qk_half_le_to_float, helper_qk_u32_le },
};

pub const decoder_q8_0 = FormatDecoder{
    .format = .q8_0,
    .weight_param = "weight_q8_0",
    .lane_decode_fn = "antfly_q8_0_dequant_lane_v2",
    .lane_decode_msl = "inline float antfly_q8_0_dequant_lane_v2(device const uchar *block, int lane) { float d = antfly_qk_half_le_to_float(block); int q = int(as_type<char>(block[2 + lane])); return d * float(q); }",
    .helpers = &.{helper_qk_half_le_to_float},
};

pub const decoder_q8_1 = FormatDecoder{
    .format = .q8_1,
    .weight_param = "weight_q8_1",
    .lane_decode_fn = "antfly_q8_1_dequant_lane_v2",
    .lane_decode_msl = "inline float antfly_q8_1_dequant_lane_v2(device const uchar *block, int lane) { float d = antfly_qk_half_le_to_float(block); int q = int(as_type<char>(block[4 + lane])); return d * float(q); }",
    .helpers = &.{helper_qk_half_le_to_float},
};

pub const decoder_q8_k = FormatDecoder{
    .format = .q8_k,
    .weight_param = "weight_q8_k",
    .lane_decode_fn = "antfly_q8_k_dequant_lane_v2",
    .lane_decode_msl = "inline float antfly_q8_k_dequant_lane_v2(device const uchar *block, int lane) { float d = antfly_qk_f32_le_to_float(block); int q = int(as_type<char>(block[4 + lane])); return d * float(q); }",
    .helpers = &.{helper_qk_f32_le_to_float},
};

pub const decoder_q2_k = FormatDecoder{
    .format = .q2_k,
    .weight_param = "weight_q2_k",
    .lane_decode_fn = "antfly_q2_k_dequant_lane_v2",
    .lane_decode_msl =
    \\inline float antfly_q2_k_dequant_lane_v2(device const uchar *block, int lane) {
    \\    uint sub = uint(lane) >> 4; uint i = uint(lane) & 15u; uchar scale_byte = block[sub]; float dsc = antfly_qk_half_le_to_float(block + 16) * float(scale_byte & 0x0fu); float dmn = antfly_qk_half_le_to_float(block + 18) * float(scale_byte >> 4);
    \\    uint chunk = sub >> 3; uint group = (sub & 7u) >> 1; uint l_base = (sub & 1u) << 4; uint q_base = chunk << 5; uint shift = group << 1; uint q = (uint(block[20u + q_base + l_base + i]) >> shift) & 0x03u;
    \\    return dsc * float(q) - dmn;
    \\}
    ,
    .helpers = &.{helper_qk_half_le_to_float},
};

pub const decoder_q3_k = FormatDecoder{
    .format = .q3_k,
    .weight_param = "weight_q3_k",
    .lane_decode_fn = "antfly_q3_k_dequant_lane_v2",
    .lane_decode_msl =
    \\inline float antfly_q3_k_dequant_lane_v2(device const uchar *block, int lane) {
    \\    uint sub = uint(lane) >> 4; uint i = uint(lane) & 15u; uint chunk = sub >> 3; uint group = (sub & 7u) >> 1; uint l = ((sub & 1u) << 4) + i; uint q_base = chunk << 5; uint shift = group << 1; uint hm_bit = (chunk << 2) + group;
    \\    int low2 = int((uint(block[32u + q_base + l]) >> shift) & 0x03u); int high1 = int((uint(block[l]) >> hm_bit) & 0x01u); int q = low2 + high1 * 4 - 4;
    \\    return antfly_qk_half_le_to_float(block + 108) * float(antfly_q3_k_raw_scale(block + 96, sub)) * float(q);
    \\}
    ,
    .helpers = &.{ helper_qk_half_le_to_float, helper_q3_k_raw_scale },
};

pub const decoder_q4_k = FormatDecoder{
    .format = .q4_k,
    .weight_param = "weight_q4_k",
    .lane_decode_fn = "antfly_q4_k_dequant_lane_v2",
    .lane_decode_msl =
    \\inline float antfly_q4_k_dequant_lane_v2(device const uchar *block, int lane) {
    \\    device const uchar *scales = block + 4; device const uchar *qs = block + 16; int sub = lane >> 5; int q_index = (sub >> 1) * 32 + (lane & 31);
    \\    uchar packed = qs[q_index]; int q = (sub & 1) == 0 ? int(packed & 0x0fu) : int(packed >> 4);
    \\    float raw_scale = 0.0f; float raw_min = 0.0f; antfly_qk_unpack_scale_min_6bit(scales, sub, raw_scale, raw_min);
    \\    return antfly_qk_half_le_to_float(block) * raw_scale * float(q) - antfly_qk_half_le_to_float(block + 2) * raw_min;
    \\}
    ,
    .helpers = &.{ helper_qk_half_le_to_float, helper_qk_unpack_scale_min_6bit },
};

pub const decoder_q5_k = FormatDecoder{
    .format = .q5_k,
    .weight_param = "weight_q5_k",
    .lane_decode_fn = "antfly_q5_k_dequant_lane_v2",
    .lane_decode_msl =
    \\inline float antfly_q5_k_dequant_lane_v2(device const uchar *block, int lane) {
    \\    device const uchar *scales = block + 4; device const uchar *qh = block + 16; device const uchar *ql = block + 48; int sub = lane >> 5; int i = lane & 31; int q_index = (sub >> 1) * 32 + i;
    \\    uchar packed = ql[q_index]; int low = (sub & 1) == 0 ? int(packed & 0x0fu) : int(packed >> 4); int high = int((qh[i] >> sub) & 1u); int q = low + high * 16;
    \\    float raw_scale = 0.0f; float raw_min = 0.0f; antfly_qk_unpack_scale_min_6bit(scales, sub, raw_scale, raw_min);
    \\    return antfly_qk_half_le_to_float(block) * raw_scale * float(q) - antfly_qk_half_le_to_float(block + 2) * raw_min;
    \\}
    ,
    .helpers = &.{ helper_qk_half_le_to_float, helper_qk_unpack_scale_min_6bit },
};

pub const decoder_q6_k = FormatDecoder{
    .format = .q6_k,
    .weight_param = "weight_q6_k",
    .lane_decode_fn = "antfly_q6_k_dequant_lane_v2",
    .lane_decode_msl =
    \\inline float antfly_q6_k_dequant_lane_v2(device const uchar *block, int lane) {
    \\    device const uchar *ql = block; device const uchar *qh = block + 128; device const uchar *scales = block + 192; int sub = lane >> 4; int i = lane & 15; int half_idx = sub >> 3; int group = (sub & 7) >> 1; int l = ((sub & 1) << 4) + i;
    \\    int ql_off = half_idx * 64 + (group & 1) * 32; int qh_off = half_idx * 32; int qh_shift = group * 2; int nibble_shift = (group >> 1) * 4; int low4 = int((ql[ql_off + l] >> nibble_shift) & 0x0fu); int high2 = int((qh[qh_off + l] >> qh_shift) & 0x03u);
    \\    int scale_u = int(scales[sub]); int scale = scale_u >= 128 ? scale_u - 256 : scale_u; return antfly_qk_half_le_to_float(block + 208) * float(scale) * float((low4 | (high2 << 4)) - 32);
    \\}
    ,
    .helpers = &.{helper_qk_half_le_to_float},
};

pub const all_decoders = [_]FormatDecoder{
    decoder_q4_0, decoder_q4_1, decoder_q5_0,  decoder_q5_1,
    decoder_q8_0, decoder_q8_1, decoder_q8_k,  decoder_q2_k,
    decoder_q3_k, decoder_q4_k, decoder_q5_k,  decoder_q6_k,
};

pub fn decoderFor(format: quant_matmul.Format) ?FormatDecoder {
    for (all_decoders) |d| {
        if (d.format == format) return d;
    }
    return null;
}

// ---- Renderer ------------------------------------------------------------

fn appendFmt(allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(u8), comptime fmt: []const u8, args: anytype) !void {
    const chunk = try std.fmt.allocPrint(allocator, fmt, args);
    defer allocator.free(chunk);
    try out.appendSlice(allocator, chunk);
}

fn log2Usize(value: usize) u6 {
    return @intCast(std.math.log2_int(usize, value));
}

/// Renders the full standalone MSL kernel: vocabulary helpers + dequant
/// fragment + (for gelu) the shared gelu helper + the kernel body.
pub fn renderKernel(
    allocator: std.mem.Allocator,
    kernel_id: []const u8,
    decoder: FormatDecoder,
    schedule: KernelSchedule,
    epilogue: Epilogue,
) ![]u8 {
    const block_values = decoder.format.valuesPerBlock() orelse return error.MissingBlockValues;
    const block_bytes = decoder.format.bytesPerBlock() orelse return error.MissingBlockBytes;
    try schedule.validate(block_values);
    try validateEpilogueSchedule(schedule, epilogue);

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    // Helper vocabulary (deduped by name across decoder + gelu).
    var emitted_names = EmittedNames{};
    for (decoder.helpers) |helper| try emitHelper(allocator, &out, &emitted_names, helper);
    if (epilogue == .bias_gelu) try emitHelper(allocator, &out, &emitted_names, helper_qk_gelu);
    try out.appendSlice(allocator, decoder.lane_decode_msl);
    try out.append(allocator, '\n');

    try renderBody(allocator, &out, kernel_id, decoder, schedule, epilogue, block_values, block_bytes);
    return out.toOwnedSlice(allocator);
}

/// One generated Metal kernel to emit into the shared runtime region: its
/// stable kernel name plus the descriptor/schedule/epilogue it renders from.
pub const RegionKernel = struct {
    kernel_id: []const u8,
    decoder: FormatDecoder,
    schedule: KernelSchedule,
    epilogue: Epilogue,
};

/// Renders the shared runtime region (all routes in one Metal compilation unit):
/// the union of vocabulary helpers + per-format dequant fragments (each emitted
/// exactly once, in first-use order), followed by every kernel body. This is the
/// single source of truth for both the `.metal` files and the metal_kernels.m
/// runtime region — the renderer, not any frozen constant.
pub fn renderRuntimeRegion(
    allocator: std.mem.Allocator,
    kernels: []const RegionKernel,
) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    // Pass 1: emit the deduped helper vocabulary + dequant fragments once each,
    // in the order they are first referenced across the routes.
    var emitted_names = EmittedNames{};
    for (kernels) |k| {
        try k.schedule.validate(k.decoder.format.valuesPerBlock() orelse return error.MissingBlockValues);
        try validateEpilogueSchedule(k.schedule, k.epilogue);
        for (k.decoder.helpers) |helper| try emitHelper(allocator, &out, &emitted_names, helper);
        if (k.epilogue == .bias_gelu) try emitHelper(allocator, &out, &emitted_names, helper_qk_gelu);
        try emitHelper(allocator, &out, &emitted_names, .{ .name = k.decoder.lane_decode_fn, .msl = k.decoder.lane_decode_msl });
    }

    // Pass 2: emit every kernel body.
    for (kernels) |k| {
        const block_values = k.decoder.format.valuesPerBlock() orelse return error.MissingBlockValues;
        const block_bytes = k.decoder.format.bytesPerBlock() orelse return error.MissingBlockBytes;
        try renderBody(allocator, &out, k.kernel_id, k.decoder, k.schedule, k.epilogue, block_values, block_bytes);
    }
    return out.toOwnedSlice(allocator);
}

/// Tracks the helper names already emitted so the shared vocabulary is written
/// exactly once. Backed by a fixed buffer (no allocation) so `renderKernel` and
/// `renderRuntimeRegion` stay usable at comptime — the checked-in `.metal`
/// sources render this way (a heap `ArrayList([]const u8)` would need pointer
/// alignment via `@intFromPtr`, which comptime cannot evaluate). The region has
/// ~18 distinct helper names across all routes; 64 leaves ample headroom.
const EmittedNames = struct {
    names: [64][]const u8 = undefined,
    len: usize = 0,

    fn contains(self: *const EmittedNames, name: []const u8) bool {
        for (self.names[0..self.len]) |n| {
            if (std.mem.eql(u8, n, name)) return true;
        }
        return false;
    }

    fn add(self: *EmittedNames, name: []const u8) void {
        std.debug.assert(self.len < self.names.len);
        self.names[self.len] = name;
        self.len += 1;
    }
};

fn emitHelper(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    emitted: *EmittedNames,
    helper: HelperFragment,
) !void {
    if (emitted.contains(helper.name)) return;
    emitted.add(helper.name);
    try out.appendSlice(allocator, helper.msl);
    try out.append(allocator, '\n');
}

fn validateEpilogueSchedule(schedule: KernelSchedule, epilogue: Epilogue) !void {
    switch (epilogue) {
        .none, .bias, .bias_gelu, .relu => {},
        else => return error.UnsupportedEpilogue,
    }
    // Two-column kernels only exist on the single-simdgroup reduction.
    if (schedule.cols_per_threadgroup == 2 and schedule.reduction != .simd_sum) {
        return error.TwoColRequiresSimdSum;
    }
    // bias_gelu two-column is supported (q8_0); relu is single-column only in
    // the current route set but the skeleton handles it either way.
}

fn renderBody(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    kernel_id: []const u8,
    decoder: FormatDecoder,
    schedule: KernelSchedule,
    epilogue: Epilogue,
    block_values: usize,
    block_bytes: usize,
) !void {
    const threads = schedule.threads_per_threadgroup;
    const mask = block_values - 1;
    const shift = log2Usize(block_values);
    const has_bias = epilogue == .bias or epilogue == .bias_gelu;
    const two_col = schedule.cols_per_threadgroup == 2;

    // Signature. Buffer layout: input=0, weight=1, [bias=2], output, rows,
    // in_dim, out_dim; bias shifts output and the scalars up by one slot.
    const bias_param = if (has_bias) "device const float *bias [[buffer(2)]], " else "";
    const out_idx: u8 = if (has_bias) 3 else 2;
    const simd_params = if (schedule.reduction == .hybrid_simd)
        ", ushort lane_id [[thread_index_in_simdgroup]], ushort simdgroup_id [[simdgroup_index_in_threadgroup]]"
    else
        "";
    try appendFmt(allocator, out,
        "kernel void {s}(device const float *input [[buffer(0)]], device const uchar *{s} [[buffer(1)]], {s}device float *output [[buffer({d})]], constant int &rows [[buffer({d})]], constant int &in_dim [[buffer({d})]], constant int &out_dim [[buffer({d})]], uint3 thread_pos [[thread_position_in_threadgroup]], uint3 group_pos [[threadgroup_position_in_grid]]{s}) {{\n",
        .{ kernel_id, decoder.weight_param, bias_param, out_idx, out_idx + 1, out_idx + 2, out_idx + 3, simd_params });

    if (two_col) {
        try renderTwoColBody(allocator, out, decoder, schedule, epilogue, mask, shift, block_values, block_bytes);
    } else {
        try renderSingleColBody(allocator, out, decoder, schedule, epilogue, threads, mask, shift, block_values, block_bytes);
    }
    try out.appendSlice(allocator, "}\n");
}

fn writeExpr(allocator: std.mem.Allocator, acc: []const u8, col: []const u8, epilogue: Epilogue) ![]u8 {
    return switch (epilogue) {
        .none => allocator.dupe(u8, acc),
        .relu => std.fmt.allocPrint(allocator, "max({s}, 0.0f)", .{acc}),
        .bias => std.fmt.allocPrint(allocator, "{s} + bias[{s}]", .{ acc, col }),
        .bias_gelu => std.fmt.allocPrint(allocator, "antfly_qk_gelu({s} + bias[{s}])", .{ acc, col }),
        else => error.UnsupportedEpilogue,
    };
}

fn renderSingleColBody(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    decoder: FormatDecoder,
    schedule: KernelSchedule,
    epilogue: Epilogue,
    threads: u16,
    mask: usize,
    shift: u6,
    block_values: usize,
    block_bytes: usize,
) !void {
    try appendFmt(allocator, out,
        "    uint tid = thread_pos.x; int col = int(group_pos.x); int row = int(group_pos.y); if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & {d}) != 0) return; float acc = 0.0f; int block_count = in_dim >> {d};\n",
        .{ mask, shift });
    // Accumulation: strided lane loop covers block_values with `threads` threads.
    try appendFmt(allocator, out,
        "    for (int block_idx = 0; block_idx < block_count; ++block_idx) {{ device const uchar *block = {s} + ((col * block_count + block_idx) * {d}); int base = block_idx << {d}; for (int lane = int(tid); lane < {d}; lane += {d}) acc += input[row * in_dim + base + lane] * {s}(block, lane); }}\n",
        .{ decoder.weight_param, block_bytes, shift, block_values, threads, decoder.lane_decode_fn });
    try renderReductionAndWrite(allocator, out, schedule, epilogue, threads);
}

fn renderReductionAndWrite(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    schedule: KernelSchedule,
    epilogue: Epilogue,
    threads: u16,
) !void {
    const write = try writeExpr(allocator, reductionResultName(schedule.reduction), "col", epilogue);
    defer allocator.free(write);
    switch (schedule.reduction) {
        .simd_sum => try appendFmt(allocator, out,
            "    acc = simd_sum(acc); if (tid == 0) output[row * out_dim + col] = {s};\n",
            .{write}),
        .threadgroup_tree => try appendFmt(allocator, out,
            "    threadgroup float partial[{d}]; partial[tid] = acc; threadgroup_barrier(mem_flags::mem_threadgroup); for (uint stride = {d}u; stride > 0u; stride >>= 1) {{ if (tid < stride) partial[tid] += partial[tid + stride]; threadgroup_barrier(mem_flags::mem_threadgroup); }} if (tid == 0) output[row * out_dim + col] = {s};\n",
            .{ threads, threads / 2, write }),
        .hybrid_simd => try appendFmt(allocator, out,
            "    threadgroup float partial[32]; acc = simd_sum(acc); if (lane_id == 0u) partial[simdgroup_id] = acc; if (simdgroup_id == 0u && lane_id >= {d}u) partial[lane_id] = 0.0f; threadgroup_barrier(mem_flags::mem_threadgroup); float total = simd_sum(partial[lane_id]); if (lane_id == 0u && simdgroup_id == 0u) output[row * out_dim + col] = {s};\n",
            .{ threads / 32, write }),
    }
}

fn reductionResultName(reduction: ReductionKind) []const u8 {
    return switch (reduction) {
        .simd_sum => "acc",
        .threadgroup_tree => "partial[0]",
        .hybrid_simd => "total",
    };
}

fn renderTwoColBody(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    decoder: FormatDecoder,
    schedule: KernelSchedule,
    epilogue: Epilogue,
    mask: usize,
    shift: u6,
    block_values: usize,
    block_bytes: usize,
) !void {
    _ = schedule; // two-col is always simd_sum (validated)
    try appendFmt(allocator, out,
        "    uint tid = thread_pos.x; int col0 = int(group_pos.x << 1); int col1 = col0 + 1; int row = int(group_pos.y); if (row >= rows || rows < 2 || rows > 8 || col0 >= out_dim || (in_dim & {d}) != 0) return; float acc0 = 0.0f; float acc1 = 0.0f; int block_count = in_dim >> {d};\n",
        .{ mask, shift });
    try appendFmt(allocator, out,
        "    device const float *row_input = input + row * in_dim; device const uchar *col0_weight = {s} + col0 * block_count * {d}; bool has_col1 = col1 < out_dim; device const uchar *col1_weight = has_col1 ? {s} + col1 * block_count * {d} : col0_weight;\n",
        .{ decoder.weight_param, block_bytes, decoder.weight_param, block_bytes });
    try appendFmt(allocator, out,
        "    for (int block_idx = 0; block_idx < block_count; ++block_idx) {{ device const uchar *block0 = col0_weight + block_idx * {d}; device const uchar *block1 = col1_weight + block_idx * {d}; int base = block_idx << {d}; for (int lane = int(tid); lane < {d}; lane += {d}) {{ float x = row_input[base + lane]; acc0 += x * {s}(block0, lane); if (has_col1) acc1 += x * {s}(block1, lane); }} }}\n",
        .{ block_bytes, block_bytes, shift, block_values, @as(u16, 32), decoder.lane_decode_fn, decoder.lane_decode_fn });
    const w0 = try writeExpr(allocator, "acc0", "col0", epilogue);
    defer allocator.free(w0);
    const w1 = try writeExpr(allocator, "acc1", "col1", epilogue);
    defer allocator.free(w1);
    try appendFmt(allocator, out,
        "    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1); if (tid == 0) {{ output[row * out_dim + col0] = {s}; if (has_col1) output[row * out_dim + col1] = {s}; }}\n",
        .{ w0, w1 });
}

// ---- Tests ---------------------------------------------------------------

test "metal renderer renders every production route and passes structure checks" {
    const allocator = std.testing.allocator;
    for (compiler.metal_production_schedules) |entry| {
        const decoder = decoderFor(entry.format) orelse return error.MissingDecoder;
        const kernel_id = try std.fmt.allocPrint(allocator, "antfly_{s}_small_batch{s}_msl_v2", .{
            @tagName(entry.format),
            epilogueSuffixForTest(entry.epilogue),
        });
        defer allocator.free(kernel_id);
        const source = try renderKernel(allocator, kernel_id, decoder, entry.schedule, entry.epilogue);
        defer allocator.free(source);

        // Kernel entrypoint present.
        try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, kernel_id));
        // Balanced braces.
        var depth: i32 = 0;
        for (source) |c| {
            if (c == '{') depth += 1 else if (c == '}') depth -= 1;
        }
        try std.testing.expectEqual(@as(i32, 0), depth);
        // Dequant fragment and its call present.
        try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, decoder.lane_decode_fn));
        // Reduction primitive present.
        switch (entry.schedule.reduction) {
            .simd_sum => try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, "simd_sum(")),
            .threadgroup_tree => try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, "partial[tid] += partial[tid + stride]")),
            .hybrid_simd => {
                try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, "simd_sum(partial[lane_id])"));
                try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, "simdgroup_id"));
            },
        }
        // hybrid params only when hybrid.
        const has_simd_params = std.mem.containsAtLeast(u8, source, 1, "lane_id [[thread_index_in_simdgroup]]");
        try std.testing.expectEqual(entry.schedule.reduction == .hybrid_simd, has_simd_params);
        // Two-column marker matches schedule.
        const has_two_col = std.mem.containsAtLeast(u8, source, 1, "group_pos.x << 1");
        try std.testing.expectEqual(entry.schedule.cols_per_threadgroup == 2, has_two_col);
        // Epilogue write.
        switch (entry.epilogue) {
            .none => {},
            .bias => try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, "+ bias[")),
            .bias_gelu => try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, "antfly_qk_gelu(")),
            .relu => try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, "max(")),
            else => return error.UnsupportedEpilogueForTest,
        }
        // No leftover per-format gelu/half duplication: v2 uses shared vocabulary.
        try std.testing.expect(!std.mem.containsAtLeast(u8, source, 1, "termite_q8_0_block_scale"));
    }
}

test "metal renderer validate rejects invalid schedules" {
    // simd_sum needs exactly 32 threads.
    try std.testing.expectError(error.SimdSumNeeds32Threads, (KernelSchedule{ .threads_per_threadgroup = 64, .cols_per_threadgroup = 1, .reduction = .simd_sum }).validate(32));
    // tree/hybrid need >= 64 threads.
    try std.testing.expectError(error.MultiSimdgroupNeeds64Threads, (KernelSchedule{ .threads_per_threadgroup = 32, .cols_per_threadgroup = 1, .reduction = .threadgroup_tree }).validate(256));
    // two-col requires simd_sum.
    const allocator = std.testing.allocator;
    const bad = renderKernel(allocator, "antfly_q4_k_x", decoder_q4_k, .{ .threads_per_threadgroup = 64, .cols_per_threadgroup = 2, .reduction = .threadgroup_tree }, .none);
    try std.testing.expectError(error.TwoColRequiresSimdSum, bad);
}

fn epilogueSuffixForTest(epilogue: Epilogue) []const u8 {
    return switch (epilogue) {
        .none => "",
        .bias => "_bias",
        .bias_gelu => "_bias_gelu",
        .relu => "_relu",
        else => "_x",
    };
}
