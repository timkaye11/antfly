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

//! Raw paged-KV CUDA decode-attention differential harness.
//!
//! The production bundle supplies the handwritten TurboQuant decode kernel.
//! Independently compiled generated cubins supply the score-prework candidate.
//! Keeping the candidate outside the runtime bundle prevents a parity or
//! performance experiment from changing production dispatch.

const std = @import("std");
const build_options = @import("build_options");
const cuda_artifact = @import("ops/cuda/artifact.zig");
const cuda_buffer = @import("ops/cuda/buffer.zig");
const cuda_context = @import("ops/cuda/context.zig");
const cuda_driver = @import("ops/cuda/driver.zig");
const cuda_renderer = @import("graph/quant_kernel_cuda_renderer.zig");

const Pattern = enum {
    random,
    near_tie,
    cancellation,

    fn label(self: Pattern) []const u8 {
        return switch (self) {
            .random => "random",
            .near_tie => "near-tie",
            .cancellation => "cancellation",
        };
    }
};

const PatternSelection = enum {
    all,
    random,
    near_tie,
    cancellation,

    fn count(self: PatternSelection) usize {
        return if (self == .all) 3 else 1;
    }

    fn at(self: PatternSelection, index: usize) Pattern {
        return switch (self) {
            .all => switch (index) {
                0 => .random,
                1 => .near_tie,
                2 => .cancellation,
                else => unreachable,
            },
            .random => .random,
            .near_tie => .near_tie,
            .cancellation => .cancellation,
        };
    }
};

const KeyFormat = enum(u32) {
    polar4 = 0,
    f16 = 2,

    fn label(self: KeyFormat) []const u8 {
        return switch (self) {
            .polar4 => "polar4",
            .f16 => "f16",
        };
    }
};

const KeyFormatSelection = enum {
    all,
    polar4,
    f16,

    fn count(self: KeyFormatSelection) usize {
        return if (self == .all) 2 else 1;
    }

    fn at(self: KeyFormatSelection, index: usize) KeyFormat {
        return switch (self) {
            .all => if (index == 0) .polar4 else .f16,
            .polar4 => .polar4,
            .f16 => .f16,
        };
    }
};

const Config = struct {
    head_dim: ?u16 = null,
    kv_len: usize = 1024,
    query_position: ?usize = null,
    sliding_window: u32 = 0,
    page_size: usize = 16,
    num_heads: usize = 16,
    num_kv_heads: usize = 1,
    pattern: PatternSelection = .all,
    key_format: KeyFormatSelection = .all,
    seed: u64 = 0x579a_e418_6d2c_3f01,
    iterations: usize = 0,
    candidate_hd256: ?[]const u8 = null,
    candidate_hd512: ?[]const u8 = null,
    json: bool = false,
    help: bool = false,
};

const DiffStats = struct {
    element_count: usize = 0,
    bitwise_mismatch_count: usize = 0,
    non_finite_mismatch_count: usize = 0,
    max_abs: f32 = 0,
    max_rel: f32 = 0,
    max_ulp: u64 = 0,
    first_mismatch_index: ?usize = null,
};

const CaseResult = struct {
    head_dim: u16,
    kv_len: usize,
    query_position: usize,
    sliding_window: u32,
    page_size: usize,
    block_count: usize,
    num_heads: usize,
    num_kv_heads: usize,
    pattern: Pattern,
    key_format: KeyFormat,
    diff: DiffStats,
    baseline_total_us: u64 = 0,
    candidate_total_us: u64 = 0,
    iterations: usize = 0,

    fn passes(self: CaseResult) bool {
        return self.diff.bitwise_mismatch_count == 0 and self.diff.non_finite_mismatch_count == 0;
    }
};

const Module = struct {
    module: cuda_driver.CUmodule = null,

    fn loadEmbedded(ctx: *cuda_context.CudaContext) !Module {
        return loadImage(ctx, cuda_artifact.image.ptr);
    }

    fn loadFile(ctx: *cuda_context.CudaContext, io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Module {
        const image = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024 * 1024));
        defer allocator.free(image);
        return loadImage(ctx, image.ptr);
    }

    fn loadImage(ctx: *cuda_context.CudaContext, image: *const anyopaque) !Module {
        try ctx.makeCurrent();
        var module: cuda_driver.CUmodule = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleLoadDataEx(&module, image, 0, null, null));
        errdefer {
            if (module != null) _ = ctx.driver.fns.cuModuleUnload(module);
        }
        return .{ .module = module };
    }

    fn deinit(self: *Module, ctx: *cuda_context.CudaContext) void {
        if (self.module != null) {
            ctx.makeCurrent() catch {};
            _ = ctx.driver.fns.cuModuleUnload(self.module);
            self.module = null;
        }
    }
};

const CandidateFunctions = struct {
    score: cuda_driver.CUfunction,
    consume: cuda_driver.CUfunction,
};

const LaunchArgs = struct {
    kv_seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    query_position_offset: u32,
    sliding_window: u32,
    total_sequence_len: u32,
    key_row_bytes: u32,
    value_row_bytes: u32,
    block_count: u32,
    page_size_tokens: u32,
    format: u32,
    physical_token_capacity: u32,
    score_capacity: u32,
};

fn checkedMul(a: usize, b: usize) !usize {
    const result = @mulWithOverflow(a, b);
    if (result[1] != 0) return error.InvalidArgument;
    return result[0];
}

fn ceilDiv(value: usize, divisor: usize) usize {
    return value / divisor + @intFromBool(value % divisor != 0);
}

fn parsePattern(value: []const u8) !PatternSelection {
    if (std.mem.eql(u8, value, "all")) return .all;
    if (std.mem.eql(u8, value, "random")) return .random;
    if (std.mem.eql(u8, value, "near-tie")) return .near_tie;
    if (std.mem.eql(u8, value, "cancellation")) return .cancellation;
    return error.InvalidPattern;
}

fn parseKeyFormat(value: []const u8) !KeyFormatSelection {
    if (std.mem.eql(u8, value, "all")) return .all;
    if (std.mem.eql(u8, value, "polar4")) return .polar4;
    if (std.mem.eql(u8, value, "f16")) return .f16;
    return error.InvalidKeyFormat;
}

fn parseConfig(args: []const []const u8) !Config {
    var cfg = Config{};
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--head-dim")) {
            index += 1;
            if (index >= args.len) return error.MissingValue;
            cfg.head_dim = if (std.mem.eql(u8, args[index], "all")) null else try std.fmt.parseInt(u16, args[index], 10);
        } else if (std.mem.eql(u8, arg, "--kv-len")) {
            index += 1;
            if (index >= args.len) return error.MissingValue;
            cfg.kv_len = try std.fmt.parseInt(usize, args[index], 10);
        } else if (std.mem.eql(u8, arg, "--query-position")) {
            index += 1;
            if (index >= args.len) return error.MissingValue;
            cfg.query_position = try std.fmt.parseInt(usize, args[index], 10);
        } else if (std.mem.eql(u8, arg, "--sliding-window")) {
            index += 1;
            if (index >= args.len) return error.MissingValue;
            cfg.sliding_window = try std.fmt.parseInt(u32, args[index], 10);
        } else if (std.mem.eql(u8, arg, "--page-size")) {
            index += 1;
            if (index >= args.len) return error.MissingValue;
            cfg.page_size = try std.fmt.parseInt(usize, args[index], 10);
        } else if (std.mem.eql(u8, arg, "--heads")) {
            index += 1;
            if (index >= args.len) return error.MissingValue;
            cfg.num_heads = try std.fmt.parseInt(usize, args[index], 10);
        } else if (std.mem.eql(u8, arg, "--kv-heads")) {
            index += 1;
            if (index >= args.len) return error.MissingValue;
            cfg.num_kv_heads = try std.fmt.parseInt(usize, args[index], 10);
        } else if (std.mem.eql(u8, arg, "--pattern")) {
            index += 1;
            if (index >= args.len) return error.MissingValue;
            cfg.pattern = try parsePattern(args[index]);
        } else if (std.mem.eql(u8, arg, "--key-format")) {
            index += 1;
            if (index >= args.len) return error.MissingValue;
            cfg.key_format = try parseKeyFormat(args[index]);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            index += 1;
            if (index >= args.len) return error.MissingValue;
            cfg.seed = try std.fmt.parseInt(u64, args[index], 0);
        } else if (std.mem.eql(u8, arg, "--iterations")) {
            index += 1;
            if (index >= args.len) return error.MissingValue;
            cfg.iterations = try std.fmt.parseInt(usize, args[index], 10);
        } else if (std.mem.eql(u8, arg, "--candidate-hd256")) {
            index += 1;
            if (index >= args.len) return error.MissingValue;
            cfg.candidate_hd256 = args[index];
        } else if (std.mem.eql(u8, arg, "--candidate-hd512")) {
            index += 1;
            if (index >= args.len) return error.MissingValue;
            cfg.candidate_hd512 = args[index];
        } else if (std.mem.eql(u8, arg, "--json")) {
            cfg.json = true;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            cfg.help = true;
        } else {
            return error.UnknownArgument;
        }
    }
    try validateConfig(cfg);
    return cfg;
}

fn validateConfig(cfg: Config) !void {
    if (cfg.help) return;
    if (cfg.head_dim) |head_dim| if (head_dim != 256 and head_dim != 512) return error.InvalidHeadDim;
    if (cfg.kv_len == 0 or cfg.kv_len > cuda_renderer.generated_attention_score_prework_max_kv_tokens) return error.InvalidKvLength;
    if (cfg.query_position) |position| if (position >= cfg.kv_len) return error.InvalidQueryPosition;
    if (cfg.page_size == 0 or cfg.page_size > cfg.kv_len) return error.InvalidPageSize;
    if (cfg.num_heads == 0 or cfg.num_heads > cuda_renderer.generated_attention_workspace_max_query_heads) return error.InvalidHeadCount;
    if (cfg.num_kv_heads == 0 or cfg.num_heads % cfg.num_kv_heads != 0) return error.InvalidKvHeadCount;
    if (cfg.num_heads / cfg.num_kv_heads > cuda_renderer.generated_attention_query_heads_per_kv_head) return error.InvalidGqaRatio;
    if (cfg.iterations > 1_000_000) return error.InvalidIterations;
    if ((cfg.head_dim == null or cfg.head_dim.? == 256) and cfg.candidate_hd256 == null) return error.MissingHd256Candidate;
    if ((cfg.head_dim == null or cfg.head_dim.? == 512) and cfg.candidate_hd512 == null) return error.MissingHd512Candidate;
}

fn nextU32(state: *u64) u32 {
    var value = state.*;
    value ^= value << 7;
    value ^= value >> 9;
    value ^= value << 8;
    state.* = value;
    return @truncate(value);
}

fn nextSignedUnit(state: *u64) f32 {
    const raw: f32 = @floatFromInt(nextU32(state) & 0xffff);
    return raw / 32767.5 - 1.0;
}

fn fillInputs(q: []f32, k: []f32, v: []f32, head_dim: usize, kv_len: usize, num_heads: usize, num_kv_heads: usize, pattern: Pattern, seed: u64) void {
    var state = seed ^ (@as(u64, @intFromEnum(pattern)) *% 0x9e37_79b9_7f4a_7c15);
    switch (pattern) {
        .random => {
            for (q) |*value| value.* = nextSignedUnit(&state) * 0.125;
            for (k) |*value| value.* = nextSignedUnit(&state) * 0.125;
            for (v) |*value| value.* = nextSignedUnit(&state) * 0.75;
        },
        .near_tie => {
            for (0..num_heads) |head| for (0..head_dim) |d| {
                const sign: f32 = if (((head + d) & 1) == 0) 1.0 else -1.0;
                q[head * head_dim + d] = sign * (0.03125 + @as(f32, @floatFromInt((head + d) % 5)) * 0.00003125);
            };
            for (0..kv_len) |token| for (0..num_kv_heads) |kv_head| for (0..head_dim) |d| {
                const sign: f32 = if (((kv_head + d) & 1) == 0) 1.0 else -1.0;
                const perturb = @as(f32, @floatFromInt((token * 17 + d * 3) % 11)) * 0.00000025;
                const index = (token * num_kv_heads + kv_head) * head_dim + d;
                k[index] = sign * (0.03125 + perturb);
                const parity: f32 = if (((token + d) & 1) == 0) 1.0 else -1.0;
                v[index] = parity * (0.25 + @as(f32, @floatFromInt(token % 29)) * 0.0005);
            };
        },
        .cancellation => {
            for (0..num_heads) |head| for (0..head_dim) |d| {
                const sign: f32 = if ((d & 1) == 0) 1.0 else -1.0;
                const perturb = @as(f32, @floatFromInt((head * 13 + d) % 7)) * 0.00001;
                q[head * head_dim + d] = sign * (0.125 + perturb);
            };
            for (0..kv_len) |token| for (0..num_kv_heads) |kv_head| for (0..head_dim) |d| {
                const sign: f32 = if ((d & 1) == 0) 1.0 else -1.0;
                const token_sign: f32 = if ((token & 1) == 0) 1.0 else -1.0;
                const perturb = @as(f32, @floatFromInt((token * 19 + kv_head * 7 + d) % 13)) * 0.00001;
                const index = (token * num_kv_heads + kv_head) * head_dim + d;
                k[index] = sign * (token_sign * 0.125 + perturb);
                v[index] = token_sign * (0.5 + @as(f32, @floatFromInt(d % 17)) * 0.0001);
            };
        },
    }
}

fn encodePolar4(value: f32) u8 {
    const clipped = std.math.clamp(value, -1.0, 1.0);
    const scaled = @round((clipped + 1.0) * 7.5);
    return @intFromFloat(std.math.clamp(scaled, 0.0, 15.0));
}

fn physicalToken(logical_token: usize, block_table: []const u32, page_size: usize) usize {
    const logical_block = logical_token / page_size;
    return @as(usize, block_table[logical_block]) * page_size + logical_token % page_size;
}

fn fillReversedBlockTable(block_table: []u32) void {
    for (block_table, 0..) |*physical_block, logical_block| {
        physical_block.* = @intCast(block_table.len - logical_block - 1);
    }
}

fn packPagedKeys(dst: []u8, logical: []const f32, block_table: []const u32, page_size: usize, kv_width: usize, row_bytes: usize, format: KeyFormat) void {
    @memset(dst, 0);
    const logical_tokens = logical.len / kv_width;
    for (0..logical_tokens) |token| {
        const physical = physicalToken(token, block_table, page_size);
        const row = dst[physical * row_bytes ..][0..row_bytes];
        const values = logical[token * kv_width ..][0..kv_width];
        switch (format) {
            .polar4 => for (0..row_bytes) |byte_index| {
                const value_index = byte_index * 2;
                const lo = encodePolar4(values[value_index]);
                const hi = if (value_index + 1 < values.len) encodePolar4(values[value_index + 1]) else 0;
                row[byte_index] = lo | (hi << 4);
            },
            .f16 => for (values, 0..) |value, value_index| {
                const half: f16 = @floatCast(value);
                const bits: u16 = @bitCast(half);
                row[value_index * 2] = @truncate(bits);
                row[value_index * 2 + 1] = @truncate(bits >> 8);
            },
        }
    }
}

fn packPagedValues(dst: []f32, logical: []const f32, block_table: []const u32, page_size: usize, kv_width: usize) void {
    @memset(dst, 0.0);
    const logical_tokens = logical.len / kv_width;
    for (0..logical_tokens) |token| {
        const physical = physicalToken(token, block_table, page_size);
        @memcpy(dst[physical * kv_width ..][0..kv_width], logical[token * kv_width ..][0..kv_width]);
    }
}

fn orderedF32Bits(value: f32) u32 {
    const bits: u32 = @bitCast(value);
    if ((bits & 0x7fff_ffff) == 0) return 0x8000_0000;
    return if ((bits & 0x8000_0000) != 0) ~bits else bits | 0x8000_0000;
}

fn ulpDistance(a: f32, b: f32) u64 {
    const ordered_a: i64 = @intCast(orderedF32Bits(a));
    const ordered_b: i64 = @intCast(orderedF32Bits(b));
    return @intCast(if (ordered_a >= ordered_b) ordered_a - ordered_b else ordered_b - ordered_a);
}

fn compareOutputs(reference: []const f32, candidate: []const f32) !DiffStats {
    if (reference.len != candidate.len) return error.LengthMismatch;
    var stats = DiffStats{ .element_count = reference.len };
    for (reference, candidate, 0..) |expected, actual, index| {
        const expected_bits: u32 = @bitCast(expected);
        const actual_bits: u32 = @bitCast(actual);
        if (expected_bits != actual_bits) {
            stats.bitwise_mismatch_count += 1;
            if (stats.first_mismatch_index == null) stats.first_mismatch_index = index;
        }
        if (!std.math.isFinite(expected) or !std.math.isFinite(actual)) {
            if (expected_bits != actual_bits) stats.non_finite_mismatch_count += 1;
            continue;
        }
        const abs_error = @abs(expected - actual);
        stats.max_abs = @max(stats.max_abs, abs_error);
        stats.max_rel = @max(stats.max_rel, abs_error / @max(@abs(expected), 1e-30));
        stats.max_ulp = @max(stats.max_ulp, ulpDistance(expected, actual));
    }
    return stats;
}

fn loadFunction(ctx: *cuda_context.CudaContext, module: cuda_driver.CUmodule, name: []const u8) !cuda_driver.CUfunction {
    var name_buffer: [160]u8 = undefined;
    const name_z = try std.fmt.bufPrintZ(&name_buffer, "{s}", .{name});
    var function: cuda_driver.CUfunction = null;
    try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&function, module, name_z));
    return function orelse error.CudaKernelUnavailable;
}

fn candidatePlan(head_dim: u16) cuda_renderer.AttentionRenderPlan {
    return cuda_renderer.attentionPlanFor(if (head_dim == 256)
        .gqa_decode_score_prework_hd256_f32
    else
        .gqa_decode_score_prework_hd512_f32);
}

fn launch(ctx: *cuda_context.CudaContext, function: cuda_driver.CUfunction, grid: u32, block: u32, params: []?*anyopaque) !void {
    try ctx.makeCurrent();
    try ctx.driver.check(ctx.driver.fns.cuLaunchKernel(function, grid, 1, 1, block, 1, 1, 0, ctx.stream, params.ptr, null));
    ctx.noteKernelLaunch();
}

fn launchBaseline(
    ctx: *cuda_context.CudaContext,
    function: cuda_driver.CUfunction,
    dst: cuda_buffer.DeviceBuffer,
    q: cuda_buffer.DeviceBuffer,
    k: cuda_buffer.DeviceBuffer,
    v: cuda_buffer.DeviceBuffer,
    block_table: cuda_buffer.DeviceBuffer,
    decode_scalars: cuda_buffer.DeviceBuffer,
    args: LaunchArgs,
) !void {
    var dst_ptr = dst.ptr;
    var q_ptr = q.ptr;
    var k_ptr = k.ptr;
    var v_ptr = v.ptr;
    var block_table_ptr = block_table.ptr;
    var null_ptr: cuda_driver.CUdeviceptr = 0;
    var batch: u32 = 1;
    var q_seq_len: u32 = 1;
    var kv_seq_len = args.kv_seq_len;
    var num_heads = args.num_heads;
    var num_kv_heads = args.num_kv_heads;
    var head_dim = args.head_dim;
    var query_position_offset = args.query_position_offset;
    var kv_position_offset: u32 = 0;
    var sliding_window = args.sliding_window;
    var total_sequence_len = args.total_sequence_len;
    var mask_len: u32 = 0;
    var bias_mode: u32 = 0;
    var key_row_bytes = args.key_row_bytes;
    var base_key_row_bytes = args.key_row_bytes;
    var value_row_bytes = args.value_row_bytes;
    var block_count = args.block_count;
    var page_size_tokens = args.page_size_tokens;
    var format = args.format;
    var value_format: u32 = 0;
    var physical_token_capacity = args.physical_token_capacity;
    var decode_scalars_ptr = decode_scalars.ptr;
    var params = [_]?*anyopaque{
        @ptrCast(&dst_ptr),            @ptrCast(&q_ptr),          @ptrCast(&k_ptr),              @ptrCast(&v_ptr),     @ptrCast(&block_table_ptr), @ptrCast(&null_ptr),                @ptrCast(&null_ptr),
        @ptrCast(&batch),              @ptrCast(&q_seq_len),      @ptrCast(&kv_seq_len),         @ptrCast(&num_heads), @ptrCast(&num_kv_heads),    @ptrCast(&head_dim),                @ptrCast(&query_position_offset),
        @ptrCast(&kv_position_offset), @ptrCast(&sliding_window), @ptrCast(&total_sequence_len), @ptrCast(&mask_len),  @ptrCast(&bias_mode),       @ptrCast(&key_row_bytes),           @ptrCast(&base_key_row_bytes),
        @ptrCast(&value_row_bytes),    @ptrCast(&block_count),    @ptrCast(&page_size_tokens),   @ptrCast(&format),    @ptrCast(&value_format),    @ptrCast(&physical_token_capacity), @ptrCast(&decode_scalars_ptr),
    };
    try launch(ctx, function, args.num_heads, args.head_dim, params[0..]);
}

fn launchCandidate(
    ctx: *cuda_context.CudaContext,
    functions: CandidateFunctions,
    dst: cuda_buffer.DeviceBuffer,
    scores: cuda_buffer.DeviceBuffer,
    q: cuda_buffer.DeviceBuffer,
    k: cuda_buffer.DeviceBuffer,
    v: cuda_buffer.DeviceBuffer,
    block_table: cuda_buffer.DeviceBuffer,
    decode_scalars: cuda_buffer.DeviceBuffer,
    args: LaunchArgs,
) !void {
    var scores_ptr = scores.ptr;
    var dst_ptr = dst.ptr;
    var q_ptr = q.ptr;
    var k_ptr = k.ptr;
    var v_ptr = v.ptr;
    var block_table_ptr = block_table.ptr;
    var batch: u32 = 1;
    var q_seq_len: u32 = 1;
    var kv_seq_len = args.kv_seq_len;
    var num_heads = args.num_heads;
    var num_kv_heads = args.num_kv_heads;
    var head_dim = args.head_dim;
    var query_position_offset = args.query_position_offset;
    var kv_position_offset: u32 = 0;
    var sliding_window = args.sliding_window;
    var total_sequence_len = args.total_sequence_len;
    var key_row_bytes = args.key_row_bytes;
    var base_key_row_bytes = args.key_row_bytes;
    var value_row_bytes = args.value_row_bytes;
    var block_count = args.block_count;
    var page_size_tokens = args.page_size_tokens;
    var format = args.format;
    var value_format: u32 = 0;
    var physical_token_capacity = args.physical_token_capacity;
    var score_capacity = args.score_capacity;
    const chunk_count_value: u32 = cuda_renderer.generated_attention_score_prework_chunks;
    var chunk_count = chunk_count_value;
    var chunk_size: u32 = @intCast(ceilDiv(args.score_capacity, chunk_count_value));
    var decode_scalars_ptr = decode_scalars.ptr;
    var score_params = [_]?*anyopaque{
        @ptrCast(&scores_ptr),     @ptrCast(&q_ptr),              @ptrCast(&k_ptr),                 @ptrCast(&block_table_ptr),
        @ptrCast(&batch),          @ptrCast(&q_seq_len),          @ptrCast(&kv_seq_len),            @ptrCast(&num_heads),
        @ptrCast(&num_kv_heads),   @ptrCast(&head_dim),           @ptrCast(&query_position_offset), @ptrCast(&kv_position_offset),
        @ptrCast(&sliding_window), @ptrCast(&total_sequence_len), @ptrCast(&key_row_bytes),         @ptrCast(&base_key_row_bytes),
        @ptrCast(&block_count),    @ptrCast(&page_size_tokens),   @ptrCast(&format),                @ptrCast(&physical_token_capacity),
        @ptrCast(&score_capacity), @ptrCast(&chunk_size),         @ptrCast(&chunk_count),           @ptrCast(&decode_scalars_ptr),
    };
    var consume_params = [_]?*anyopaque{
        @ptrCast(&dst_ptr),            @ptrCast(&scores_ptr),         @ptrCast(&v_ptr),                   @ptrCast(&block_table_ptr),
        @ptrCast(&batch),              @ptrCast(&q_seq_len),          @ptrCast(&kv_seq_len),              @ptrCast(&num_heads),
        @ptrCast(&num_kv_heads),       @ptrCast(&head_dim),           @ptrCast(&query_position_offset),   @ptrCast(&kv_position_offset),
        @ptrCast(&sliding_window),     @ptrCast(&total_sequence_len), @ptrCast(&value_row_bytes),         @ptrCast(&block_count),
        @ptrCast(&page_size_tokens),   @ptrCast(&value_format),       @ptrCast(&physical_token_capacity), @ptrCast(&score_capacity),
        @ptrCast(&decode_scalars_ptr),
    };
    const score_grid = try checkedMul(args.num_heads, chunk_count_value);
    try launch(ctx, functions.score, @intCast(score_grid), args.head_dim, score_params[0..]);
    try launch(ctx, functions.consume, args.num_heads, args.head_dim, consume_params[0..]);
}

fn timeBaseline(ctx: *cuda_context.CudaContext, iterations: usize, function: cuda_driver.CUfunction, dst: cuda_buffer.DeviceBuffer, q: cuda_buffer.DeviceBuffer, k: cuda_buffer.DeviceBuffer, v: cuda_buffer.DeviceBuffer, table: cuda_buffer.DeviceBuffer, scalars: cuda_buffer.DeviceBuffer, args: LaunchArgs) !u64 {
    if (iterations == 0) return 0;
    const pair = try ctx.beginProfileEventPair();
    for (0..iterations) |_| try launchBaseline(ctx, function, dst, q, k, v, table, scalars, args);
    return ctx.endProfileEventPairUs(pair);
}

fn timeCandidate(ctx: *cuda_context.CudaContext, iterations: usize, functions: CandidateFunctions, dst: cuda_buffer.DeviceBuffer, scores: cuda_buffer.DeviceBuffer, q: cuda_buffer.DeviceBuffer, k: cuda_buffer.DeviceBuffer, v: cuda_buffer.DeviceBuffer, table: cuda_buffer.DeviceBuffer, scalars: cuda_buffer.DeviceBuffer, args: LaunchArgs) !u64 {
    if (iterations == 0) return 0;
    const pair = try ctx.beginProfileEventPair();
    for (0..iterations) |_| try launchCandidate(ctx, functions, dst, scores, q, k, v, table, scalars, args);
    return ctx.endProfileEventPairUs(pair);
}

fn runCase(
    allocator: std.mem.Allocator,
    ctx: *cuda_context.CudaContext,
    baseline_function: cuda_driver.CUfunction,
    candidate_functions: CandidateFunctions,
    cfg: Config,
    head_dim: u16,
    pattern: Pattern,
    key_format: KeyFormat,
) !CaseResult {
    const dim: usize = head_dim;
    const q_count = try checkedMul(cfg.num_heads, dim);
    const kv_width = try checkedMul(cfg.num_kv_heads, dim);
    const logical_kv_count = try checkedMul(cfg.kv_len, kv_width);
    const block_count = ceilDiv(cfg.kv_len, cfg.page_size);
    const physical_capacity = try checkedMul(block_count, cfg.page_size);
    const key_row_bytes = switch (key_format) {
        .polar4 => ceilDiv(kv_width, 2),
        .f16 => try checkedMul(kv_width, 2),
    };
    const value_row_bytes = try checkedMul(kv_width, @sizeOf(f32));
    const key_bytes = try checkedMul(physical_capacity, key_row_bytes);
    const physical_value_count = try checkedMul(physical_capacity, kv_width);
    const score_capacity: usize = cuda_renderer.generated_attention_score_prework_max_kv_tokens;
    const score_count = try checkedMul(cfg.num_heads, score_capacity);

    const host_q = try allocator.alloc(f32, q_count);
    defer allocator.free(host_q);
    const logical_k = try allocator.alloc(f32, logical_kv_count);
    defer allocator.free(logical_k);
    const logical_v = try allocator.alloc(f32, logical_kv_count);
    defer allocator.free(logical_v);
    const paged_k = try allocator.alloc(u8, key_bytes);
    defer allocator.free(paged_k);
    const paged_v = try allocator.alloc(f32, physical_value_count);
    defer allocator.free(paged_v);
    const block_table = try allocator.alloc(u32, block_count);
    defer allocator.free(block_table);
    const host_baseline = try allocator.alloc(f32, q_count);
    defer allocator.free(host_baseline);
    const host_candidate = try allocator.alloc(f32, q_count);
    defer allocator.free(host_candidate);

    fillInputs(host_q, logical_k, logical_v, dim, cfg.kv_len, cfg.num_heads, cfg.num_kv_heads, pattern, cfg.seed);
    fillReversedBlockTable(block_table);
    packPagedKeys(paged_k, logical_k, block_table, cfg.page_size, kv_width, key_row_bytes, key_format);
    packPagedValues(paged_v, logical_v, block_table, cfg.page_size, kv_width);

    var d_q = try cuda_buffer.DeviceBuffer.alloc(ctx, host_q.len * @sizeOf(f32));
    defer d_q.free(ctx);
    var d_k = try cuda_buffer.DeviceBuffer.alloc(ctx, paged_k.len);
    defer d_k.free(ctx);
    var d_v = try cuda_buffer.DeviceBuffer.alloc(ctx, paged_v.len * @sizeOf(f32));
    defer d_v.free(ctx);
    var d_table = try cuda_buffer.DeviceBuffer.alloc(ctx, block_table.len * @sizeOf(u32));
    defer d_table.free(ctx);
    var d_scalars = try cuda_buffer.DeviceBuffer.alloc(ctx, 5 * @sizeOf(u32));
    defer d_scalars.free(ctx);
    var d_baseline = try cuda_buffer.DeviceBuffer.alloc(ctx, host_baseline.len * @sizeOf(f32));
    defer d_baseline.free(ctx);
    var d_candidate = try cuda_buffer.DeviceBuffer.alloc(ctx, host_candidate.len * @sizeOf(f32));
    defer d_candidate.free(ctx);
    var d_scores = try cuda_buffer.DeviceBuffer.alloc(ctx, score_count * @sizeOf(f32));
    defer d_scores.free(ctx);

    const query_position: u32 = @intCast(cfg.query_position orelse cfg.kv_len - 1);
    const kv_len_u32: u32 = @intCast(cfg.kv_len);
    const decode_scalars = [_]u32{ query_position, query_position, kv_len_u32, kv_len_u32, 0 };
    try d_q.copyFromHost(ctx, std.mem.sliceAsBytes(host_q));
    try d_k.copyFromHost(ctx, paged_k);
    try d_v.copyFromHost(ctx, std.mem.sliceAsBytes(paged_v));
    try d_table.copyFromHost(ctx, std.mem.sliceAsBytes(block_table));
    try d_scalars.copyFromHost(ctx, std.mem.sliceAsBytes(&decode_scalars));

    const args = LaunchArgs{
        .kv_seq_len = kv_len_u32,
        .num_heads = @intCast(cfg.num_heads),
        .num_kv_heads = @intCast(cfg.num_kv_heads),
        .head_dim = head_dim,
        .query_position_offset = query_position,
        .sliding_window = cfg.sliding_window,
        .total_sequence_len = kv_len_u32,
        .key_row_bytes = @intCast(key_row_bytes),
        .value_row_bytes = @intCast(value_row_bytes),
        .block_count = @intCast(block_count),
        .page_size_tokens = @intCast(cfg.page_size),
        .format = @intFromEnum(key_format),
        .physical_token_capacity = @intCast(physical_capacity),
        .score_capacity = @intCast(score_capacity),
    };

    try launchBaseline(ctx, baseline_function, d_baseline, d_q, d_k, d_v, d_table, d_scalars, args);
    try launchCandidate(ctx, candidate_functions, d_candidate, d_scores, d_q, d_k, d_v, d_table, d_scalars, args);
    try d_baseline.copyToHost(ctx, std.mem.sliceAsBytes(host_baseline));
    try d_candidate.copyToHost(ctx, std.mem.sliceAsBytes(host_candidate));
    try ctx.synchronize();

    for (0..3) |_| {
        try launchBaseline(ctx, baseline_function, d_baseline, d_q, d_k, d_v, d_table, d_scalars, args);
        try launchCandidate(ctx, candidate_functions, d_candidate, d_scores, d_q, d_k, d_v, d_table, d_scalars, args);
    }
    try ctx.synchronize();
    const baseline_total_us = try timeBaseline(ctx, cfg.iterations, baseline_function, d_baseline, d_q, d_k, d_v, d_table, d_scalars, args);
    const candidate_total_us = try timeCandidate(ctx, cfg.iterations, candidate_functions, d_candidate, d_scores, d_q, d_k, d_v, d_table, d_scalars, args);

    return .{
        .head_dim = head_dim,
        .kv_len = cfg.kv_len,
        .query_position = query_position,
        .sliding_window = cfg.sliding_window,
        .page_size = cfg.page_size,
        .block_count = block_count,
        .num_heads = cfg.num_heads,
        .num_kv_heads = cfg.num_kv_heads,
        .pattern = pattern,
        .key_format = key_format,
        .diff = try compareOutputs(host_baseline, host_candidate),
        .baseline_total_us = baseline_total_us,
        .candidate_total_us = candidate_total_us,
        .iterations = cfg.iterations,
    };
}

fn writeResultHuman(writer: anytype, result: CaseResult) !void {
    try writer.print(
        "head_dim={d} kv_len={d} query_position={d} sliding_window={d} page_size={d} blocks={d} heads={d} kv_heads={d} key_format={s} pattern={s} status={s}\n",
        .{ result.head_dim, result.kv_len, result.query_position, result.sliding_window, result.page_size, result.block_count, result.num_heads, result.num_kv_heads, result.key_format.label(), result.pattern.label(), if (result.passes()) "pass" else "fail" },
    );
    try writer.print("  generated_vs_production: elements={d} bitwise_mismatches={d} nonfinite_mismatches={d} max_abs={e:.7} max_rel={e:.7} max_ulp={d}", .{
        result.diff.element_count,
        result.diff.bitwise_mismatch_count,
        result.diff.non_finite_mismatch_count,
        result.diff.max_abs,
        result.diff.max_rel,
        result.diff.max_ulp,
    });
    if (result.diff.first_mismatch_index) |index| try writer.print(" first_mismatch={d}", .{index});
    try writer.writeByte('\n');
    if (result.iterations != 0) {
        const baseline_us = @as(f64, @floatFromInt(result.baseline_total_us)) / @as(f64, @floatFromInt(result.iterations));
        const candidate_us = @as(f64, @floatFromInt(result.candidate_total_us)) / @as(f64, @floatFromInt(result.iterations));
        try writer.print("  raw_launch_timing: iterations={d} baseline_us={d:.3} candidate_us={d:.3} speedup={d:.4}x\n", .{
            result.iterations,
            baseline_us,
            candidate_us,
            if (candidate_us > 0) baseline_us / candidate_us else 0,
        });
    }
}

fn writeResultsJson(writer: anytype, results: []const CaseResult, device_name: []const u8) !void {
    var pass = true;
    for (results) |result| if (!result.passes()) {
        pass = false;
        break;
    };
    try writer.print("{{\"schema\":\"antfly.cuda_paged_attention_diff.v1\",\"device\":\"{s}\",\"pass\":{s},\"results\":[", .{ device_name, if (pass) "true" else "false" });
    for (results, 0..) |result, index| {
        if (index != 0) try writer.writeByte(',');
        try writer.print(
            "{{\"head_dim\":{d},\"kv_len\":{d},\"query_position\":{d},\"sliding_window\":{d},\"page_size\":{d},\"block_count\":{d},\"num_heads\":{d},\"num_kv_heads\":{d},\"key_format\":\"{s}\",\"pattern\":\"{s}\",\"pass\":{s},\"bitwise_mismatches\":{d},\"nonfinite_mismatches\":{d},\"max_abs\":{e:.9},\"max_rel\":{e:.9},\"max_ulp\":{d},\"iterations\":{d},\"baseline_total_us\":{d},\"candidate_total_us\":{d}}}",
            .{ result.head_dim, result.kv_len, result.query_position, result.sliding_window, result.page_size, result.block_count, result.num_heads, result.num_kv_heads, result.key_format.label(), result.pattern.label(), if (result.passes()) "true" else "false", result.diff.bitwise_mismatch_count, result.diff.non_finite_mismatch_count, result.diff.max_abs, result.diff.max_rel, result.diff.max_ulp, result.iterations, result.baseline_total_us, result.candidate_total_us },
        );
    }
    try writer.writeAll("]}\n");
}

fn writeUsage(writer: anytype) !void {
    try writer.writeAll(
        \\Usage: zig build quant-kernel-cuda-paged-attention-diff -Dcuda=true -Dcuda-artifacts=sm89 -- [options]
        \\  --candidate-hd256 PATH    Generated HD256 score-prework cubin
        \\  --candidate-hd512 PATH    Generated HD512 score-prework cubin
        \\  --head-dim 256|512|all    Default: all
        \\  --kv-len N                 Default: 1024; maximum: 4096
        \\  --query-position N         Default: kv_len - 1
        \\  --sliding-window N         Default: 0 (disabled)
        \\  --page-size N              Default: 16; physical page order is reversed
        \\  --heads N --kv-heads N     Default: 16 and 1
        \\  --pattern all|random|near-tie|cancellation
        \\  --key-format all|polar4|f16
        \\  --seed N                   Decimal or 0x-prefixed deterministic seed
        \\  --iterations N             Raw launch timings; default: 0
        \\  --json                     Emit antfly.cuda_paged_attention_diff.v1 JSON
        \\
    );
}

fn parseConfigFromInit(init: std.process.Init) !Config {
    var args_iter = std.process.Args.Iterator.init(init.minimal.args);
    _ = args_iter.next();
    var args: [40][]const u8 = undefined;
    var len: usize = 0;
    while (args_iter.next()) |arg| {
        if (len == args.len) return error.TooManyArguments;
        args[len] = arg;
        len += 1;
    }
    return parseConfig(args[0..len]);
}

pub fn main(init: std.process.Init) !void {
    const cfg = try parseConfigFromInit(init);
    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    if (cfg.help) {
        try writeUsage(stdout);
        try stdout.flush();
        return;
    }
    if (!build_options.enable_cuda) return error.CudaDisabled;

    var ctx = try cuda_context.CudaContext.initDefault();
    defer ctx.deinit();
    var baseline_module = try Module.loadEmbedded(&ctx);
    defer baseline_module.deinit(&ctx);
    const baseline_function = try loadFunction(&ctx, baseline_module.module, "termite_gqa_attention_decode_turboquant_fast_f32");

    const max_results = 2 * 3 * 2;
    var results: [max_results]CaseResult = undefined;
    var result_len: usize = 0;
    const head_dim_count: usize = if (cfg.head_dim == null) 2 else 1;
    for (0..head_dim_count) |head_index| {
        const head_dim: u16 = cfg.head_dim orelse if (head_index == 0) 256 else 512;
        const path = if (head_dim == 256) cfg.candidate_hd256.? else cfg.candidate_hd512.?;
        var candidate_module = try Module.loadFile(&ctx, init.io, std.heap.c_allocator, path);
        defer candidate_module.deinit(&ctx);
        const plan = candidatePlan(head_dim);
        const functions = CandidateFunctions{
            .score = try loadFunction(&ctx, candidate_module.module, plan.kernel_id),
            .consume = try loadFunction(&ctx, candidate_module.module, plan.reduction_kernel_id),
        };
        for (0..cfg.pattern.count()) |pattern_index| {
            for (0..cfg.key_format.count()) |format_index| {
                results[result_len] = try runCase(
                    std.heap.c_allocator,
                    &ctx,
                    baseline_function,
                    functions,
                    cfg,
                    head_dim,
                    cfg.pattern.at(pattern_index),
                    cfg.key_format.at(format_index),
                );
                result_len += 1;
            }
        }
    }

    const output = results[0..result_len];
    if (cfg.json) {
        try writeResultsJson(stdout, output, ctx.info.nameSlice());
    } else {
        try stdout.print("CUDA paged attention differential: device={s} cc={d}.{d} baseline=termite_gqa_attention_decode_turboquant_fast_f32 candidate=generated_score_prework page_order=reversed\n", .{
            ctx.info.nameSlice(),
            ctx.info.compute_major,
            ctx.info.compute_minor,
        });
        for (output) |result| try writeResultHuman(stdout, result);
    }
    try stdout.flush();
    for (output) |result| if (!result.passes()) return error.PagedAttentionDifferentialExceeded;
}

test "paged attention diff parses the candidate and paged case surface" {
    const cfg = try parseConfig(&.{
        "--candidate-hd256", "hd256.cubin", "--head-dim",   "256", "--kv-len", "512",  "--query-position", "511",
        "--sliding-window",  "128",         "--page-size",  "32",  "--heads",  "8",    "--kv-heads",       "1",
        "--pattern",         "near-tie",    "--key-format", "f16", "--seed",   "0x42", "--iterations",     "100",
        "--json",
    });
    try std.testing.expectEqual(@as(?u16, 256), cfg.head_dim);
    try std.testing.expectEqual(@as(usize, 512), cfg.kv_len);
    try std.testing.expectEqual(@as(?usize, 511), cfg.query_position);
    try std.testing.expectEqual(@as(u32, 128), cfg.sliding_window);
    try std.testing.expectEqual(@as(usize, 32), cfg.page_size);
    try std.testing.expectEqual(PatternSelection.near_tie, cfg.pattern);
    try std.testing.expectEqual(KeyFormatSelection.f16, cfg.key_format);
    try std.testing.expectEqual(@as(usize, 100), cfg.iterations);
    try std.testing.expect(cfg.json);
}

test "paged attention diff rejects invalid topology and missing cubins" {
    try std.testing.expectError(error.MissingHd256Candidate, parseConfig(&.{ "--head-dim", "256" }));
    try std.testing.expectError(error.MissingHd512Candidate, parseConfig(&.{ "--head-dim", "512", "--candidate-hd256", "unused" }));
    try std.testing.expectError(error.InvalidKvLength, parseConfig(&.{ "--head-dim", "256", "--candidate-hd256", "x", "--kv-len", "4097" }));
    try std.testing.expectError(error.InvalidPageSize, parseConfig(&.{ "--head-dim", "256", "--candidate-hd256", "x", "--kv-len", "16", "--page-size", "32" }));
    try std.testing.expectError(error.InvalidGqaRatio, parseConfig(&.{ "--head-dim", "256", "--candidate-hd256", "x", "--heads", "32", "--kv-heads", "1" }));
    try std.testing.expectError(error.InvalidKeyFormat, parseConfig(&.{ "--head-dim", "256", "--candidate-hd256", "x", "--key-format", "packed3" }));
}

test "paged attention diff reverses physical pages and packs both key formats" {
    var table = [_]u32{ 0, 0, 0 };
    fillReversedBlockTable(&table);
    try std.testing.expectEqualSlices(u32, &.{ 2, 1, 0 }, &table);
    try std.testing.expectEqual(@as(usize, 9), physicalToken(1, &table, 4));
    try std.testing.expectEqual(@as(usize, 5), physicalToken(5, &table, 4));

    const logical = [_]f32{ -1.0, 1.0, 0.0, 0.5 };
    var polar = [_]u8{0} ** 2;
    const identity = [_]u32{0};
    packPagedKeys(&polar, &logical, &identity, 1, 4, 2, .polar4);
    try std.testing.expectEqual(@as(u8, 0xf0), polar[0]);
    try std.testing.expectEqual(@as(u8, 0xb8), polar[1]);

    var half = [_]u8{0} ** 8;
    packPagedKeys(&half, &logical, &identity, 1, 4, 8, .f16);
    const first_bits: u16 = @bitCast(@as(f16, -1.0));
    try std.testing.expectEqual(@as(u8, @truncate(first_bits)), half[0]);
    try std.testing.expectEqual(@as(u8, @truncate(first_bits >> 8)), half[1]);
}

test "paged attention diff reports exact output differences" {
    const reference = [_]f32{ 0.0, 1.0, -2.0, 4.0 };
    const candidate = [_]f32{ -0.0, 1.0000001, -2.0, 3.5 };
    const stats = try compareOutputs(&reference, &candidate);
    try std.testing.expectEqual(@as(usize, 3), stats.bitwise_mismatch_count);
    try std.testing.expectEqual(@as(usize, 0), stats.first_mismatch_index.?);
    try std.testing.expect(stats.max_abs >= 0.5);
    try std.testing.expect(stats.max_ulp > 0);
}
