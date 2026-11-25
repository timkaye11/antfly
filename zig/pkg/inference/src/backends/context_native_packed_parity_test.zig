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

//! Cross-ABI qualification for rounded native candidates. The flat whole-head
//! validator and the packed serving kernels intentionally have independent
//! indexing and reduction implementations; this fixture feeds both the same
//! candidate/query population and checks the actual ABI2 descriptor route.
const std = @import("std");
const build_options = @import("build_options");
const native = @import("context_attention_match.zig");
const attention = @import("context_kernels.zig");
const views = @import("../runtime/kv/context_view.zig");

const kv_heads = 2;
const source_rows = 517;
const query_rows = 3;
const compact_logical_rows = 512;
const compact_rows = 64;
const candidate_rows = 69;
const bias_rows_per_head = compact_rows;

fn compactMultiplicity(head: usize, compact_row: usize) usize {
    return if (head == 0) 8 else if (compact_row % 2 == 0) 4 else 12;
}

fn candidateRow(head: usize, logical: usize) usize {
    if (logical == 0) return 0; // exact prefix
    if (logical <= compact_logical_rows) {
        const zero_based = logical - 1;
        if (head == 0) return 1 + zero_based / 8;
        const pair = zero_based / 16;
        return 1 + pair * 2 + @intFromBool(zero_based % 16 >= 4);
    }
    if (logical <= compact_logical_rows + 2) return 1 + compact_rows + logical - compact_logical_rows - 1; // exact recent
    return 1 + compact_rows + 2 + logical - compact_logical_rows - 3; // exact continuation
}

fn storedKey(head: usize, row: usize, column: usize) f16 {
    const raw = @as(f32, @floatFromInt((row * 13 + head * 7 + column * 3) % 23)) - 11;
    return @floatCast(raw / 64);
}

fn storedValue(head: usize, row: usize, column: usize) f16 {
    const raw = @as(f32, @floatFromInt((row * 17 + head * 11 + column * 5) % 29)) - 14;
    return @floatCast(raw / 16);
}

fn queryValue(consumer: usize, query_head: usize, column: usize) f32 {
    const raw = @as(f32, @floatFromInt((consumer * 19 + query_head * 7 + column * 5) % 31)) - 15;
    return raw / 16;
}

fn fillConsumerQueries(comptime dim: usize, values: []f32, consumer: usize, query_heads: usize) void {
    std.debug.assert(values.len == query_rows * query_heads * dim);
    for (0..query_rows) |query_row| {
        for (0..query_heads) |query_head| {
            for (0..dim) |column|
                values[(query_row * query_heads + query_head) * dim + column] = queryValue(consumer + query_row * 3, query_head, column);
        }
    }
}

fn reference(comptime dim: usize, query: []const f32, head: usize, logical_rows: usize, output: []f64) void {
    std.debug.assert(query.len == dim and output.len == dim);
    var logits: [source_rows]f64 = undefined;
    var maximum = -std.math.inf(f64);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));
    for (0..logical_rows) |logical| {
        const row = candidateRow(head, logical);
        var dot: f64 = 0;
        for (0..dim) |column|
            dot += @as(f64, query[column]) * @as(f64, storedKey(head, row, column));
        logits[logical] = dot * @as(f64, scale);
        maximum = @max(maximum, logits[logical]);
    }
    @memset(output, 0);
    var denominator: f64 = 0;
    for (logits[0..logical_rows], 0..) |logit, logical| {
        const weight = @exp(logit - maximum);
        denominator += weight;
        const row = candidateRow(head, logical);
        for (output, 0..) |*value, column|
            value.* += weight * @as(f64, storedValue(head, row, column));
    }
    for (output) |*value| value.* /= denominator;
}

fn expectConsumerReference(comptime dim: usize, queries: []const f32, output: []const f32, query_heads: usize, tolerance: f64) !void {
    const groups = query_heads / kv_heads;
    var expected: [dim]f64 = undefined;
    for (0..query_rows) |query_row| for (0..query_heads) |query_head| {
        const vector = query_row * query_heads + query_head;
        const head = query_head / groups;
        reference(dim, queries[vector * dim ..][0..dim], head, source_rows - query_rows + query_row + 1, &expected);
        for (expected, output[vector * dim ..][0..dim]) |want, got|
            try std.testing.expectApproxEqAbs(want, @as(f64, got), tolerance);
    };
}

fn expectOutputsClose(expected: []const f32, actual: []const f32, tolerance: f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |want, got| try std.testing.expectApproxEqAbs(want, got, tolerance);
}

fn fixture(comptime dim: usize) !void {
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));

    // The flat validator operates one physical head at a time, exactly as the
    // native publication transaction does after assembling a complete head.
    var validation = try native.Kernels.initTracked(null);
    defer validation.deinit();
    var source_keys = try validation.core.buffer(source_rows * dim * @sizeOf(f16));
    defer source_keys.deinit();
    var source_values = try validation.core.buffer(source_rows * dim * @sizeOf(f16));
    defer source_values.deinit();
    var candidate_keys = try validation.core.buffer(candidate_rows * dim * @sizeOf(f16));
    defer candidate_keys.deinit();
    var candidate_values = try validation.core.buffer(candidate_rows * dim * @sizeOf(f16));
    defer candidate_values.deinit();
    var dense_bias = try validation.core.buffer(candidate_rows * @sizeOf(f32));
    defer dense_bias.deinit();
    var head_queries = try validation.core.buffer(5 * dim * @sizeOf(f32));
    defer head_queries.deinit();

    // Actual packed serving buffers are head-major ABI2 arenas. Descriptor
    // buffers are uploaded explicitly, matching PackedContextSession framing.
    var serving = try attention.Kernels.initFlash();
    defer serving.deinit();
    var packed_keys = try serving.core.buffer(kv_heads * candidate_rows * dim * @sizeOf(f16));
    defer packed_keys.deinit();
    var packed_values = try serving.core.buffer(kv_heads * candidate_rows * dim * @sizeOf(f16));
    defer packed_values.deinit();
    var packed_bias = try serving.core.buffer(kv_heads * bias_rows_per_head * @sizeOf(f32));
    defer packed_bias.deinit();
    var consumer_a_queries = try serving.core.buffer(query_rows * 4 * dim * @sizeOf(f32));
    defer consumer_a_queries.deinit();
    var consumer_b_queries = try serving.core.buffer(query_rows * 6 * dim * @sizeOf(f32));
    defer consumer_b_queries.deinit();
    var consumer_a_serial = try serving.core.buffer(query_rows * 4 * dim * @sizeOf(f32));
    defer consumer_a_serial.deinit();
    var consumer_b_serial = try serving.core.buffer(query_rows * 6 * dim * @sizeOf(f32));
    defer consumer_b_serial.deinit();
    var route_output = try serving.core.buffer(query_rows * 6 * dim * @sizeOf(f32));
    defer route_output.deinit();

    const spans = [_]views.Span{
        .{ .physical_start = 0, .row_count = 1, .logical_start = 0, .logical_count = 1 },
        .{ .physical_start = 1, .row_count = compact_rows, .logical_start = 1, .logical_count = compact_logical_rows, .kind = .compacted, .bias_start = 0 },
        .{ .physical_start = 65, .row_count = 2, .logical_start = 513, .logical_count = 2 },
        .{ .physical_start = 67, .row_count = 2, .logical_start = 515, .logical_count = 2 },
        .{ .physical_start = 0, .row_count = 1, .logical_start = 0, .logical_count = 1 },
        .{ .physical_start = 1, .row_count = compact_rows, .logical_start = 1, .logical_count = compact_logical_rows, .kind = .compacted, .bias_start = 0 },
        .{ .physical_start = 65, .row_count = 2, .logical_start = 513, .logical_count = 2 },
        .{ .physical_start = 67, .row_count = 2, .logical_start = 515, .logical_count = 2 },
    };
    const heads = [_]views.Head{
        .{ .kv_start = 0, .capacity_rows = candidate_rows, .bias_start = 0, .bias_rows = bias_rows_per_head, .span_start = 0, .span_count = 4 },
        .{ .kv_start = candidate_rows, .capacity_rows = candidate_rows, .bias_start = bias_rows_per_head, .bias_rows = bias_rows_per_head, .span_start = 4, .span_count = 4 },
    };
    const view = views.View{
        .version = views.abi_version,
        .epoch = 7,
        .logical_tokens = source_rows,
        .capacity_rows = kv_heads * candidate_rows,
        .bias_rows = kv_heads * bias_rows_per_head,
        .kv_heads = kv_heads,
        .head_dim = @intCast(dim),
        .spans = &spans,
        .heads = &heads,
    };
    try std.testing.expectEqual(@as(u32, candidate_rows), try view.validate());
    var descriptor_spans = try serving.core.buffer(spans.len * @sizeOf(views.Span));
    defer descriptor_spans.deinit();
    var descriptor_heads = try serving.core.buffer(heads.len * @sizeOf(views.Head));
    defer descriptor_heads.deinit();
    @memcpy(descriptor_spans.bytes[0 .. spans.len * @sizeOf(views.Span)], std.mem.sliceAsBytes(spans[0..]));
    @memcpy(descriptor_heads.bytes[0 .. heads.len * @sizeOf(views.Head)], std.mem.sliceAsBytes(heads[0..]));

    const packed_k = std.mem.bytesAsSlice(f16, @as([]align(2) u8, @alignCast(packed_keys.bytes[0..packed_keys.logical_bytes])));
    const packed_v = std.mem.bytesAsSlice(f16, @as([]align(2) u8, @alignCast(packed_values.bytes[0..packed_values.logical_bytes])));
    for (0..kv_heads) |head| for (0..candidate_rows) |row| for (0..dim) |column| {
        const offset = (head * candidate_rows + row) * dim + column;
        packed_k[offset] = storedKey(head, row, column);
        packed_v[offset] = storedValue(head, row, column);
    };
    for (0..kv_heads) |head| {
        for (0..compact_rows) |row|
            packed_bias.floats()[head * bias_rows_per_head + row] = @log(@as(f32, @floatFromInt(compactMultiplicity(head, row))));
    }
    fillConsumerQueries(dim, consumer_a_queries.floats(), 0, 4);
    fillConsumerQueries(dim, consumer_b_queries.floats(), 1, 6);

    const validation_request = native.HeadRequest{
        .source_rows = source_rows,
        .candidate_rows = candidate_rows,
        .head_dim = @intCast(dim),
        .validation_rows = 5,
        .validation_group_rows = 1,
        .scale = scale,
        .max_mass_error = 0.001,
        .max_output_rmse = 0.001,
        .max_consumer_output_rmse = 0.001,
        .staging_budget_bytes = 64 * 1024 * 1024,
    };
    for (0..kv_heads) |head| {
        const src_k = std.mem.bytesAsSlice(f16, @as([]align(2) u8, @alignCast(source_keys.bytes[0..source_keys.logical_bytes])));
        const src_v = std.mem.bytesAsSlice(f16, @as([]align(2) u8, @alignCast(source_values.bytes[0..source_values.logical_bytes])));
        const candidate_k = std.mem.bytesAsSlice(f16, @as([]align(2) u8, @alignCast(candidate_keys.bytes[0..candidate_keys.logical_bytes])));
        const candidate_v = std.mem.bytesAsSlice(f16, @as([]align(2) u8, @alignCast(candidate_values.bytes[0..candidate_values.logical_bytes])));
        for (0..candidate_rows) |row| for (0..dim) |column| {
            candidate_k[row * dim + column] = storedKey(head, row, column);
            candidate_v[row * dim + column] = storedValue(head, row, column);
        };
        for (0..source_rows) |logical| for (0..dim) |column| {
            const row = candidateRow(head, logical);
            src_k[logical * dim + column] = storedKey(head, row, column);
            src_v[logical * dim + column] = storedValue(head, row, column);
        };
        @memset(dense_bias.floats(), 0);
        for (0..compact_rows) |row|
            dense_bias.floats()[1 + row] = @log(@as(f32, @floatFromInt(compactMultiplicity(head, row))));
        const q = head_queries.floats();
        const a_first = head * 2;
        const b_first = head * 3;
        const a_last = (query_rows - 1) * 4;
        const b_last = (query_rows - 1) * 6;
        @memcpy(q[0 * dim ..][0..dim], consumer_a_queries.floats()[(a_last + a_first + 0) * dim ..][0..dim]);
        @memcpy(q[1 * dim ..][0..dim], consumer_a_queries.floats()[(a_last + a_first + 1) * dim ..][0..dim]);
        @memcpy(q[2 * dim ..][0..dim], consumer_b_queries.floats()[(b_last + b_first + 0) * dim ..][0..dim]);
        @memcpy(q[3 * dim ..][0..dim], consumer_b_queries.floats()[(b_last + b_first + 1) * dim ..][0..dim]);
        @memcpy(q[4 * dim ..][0..dim], consumer_b_queries.floats()[(b_last + b_first + 2) * dim ..][0..dim]);
        const metrics = try validation.validateHead(validation_request, source_keys.ref(), source_values.ref(), head_queries.ref(), candidate_keys.ref(), candidate_values.ref(), dense_bias.ref());
        try std.testing.expectEqual(@as(u32, 5), metrics.consumer_groups);
        try std.testing.expect(metrics.max_mass_error <= 2e-5);
        try std.testing.expect(metrics.output_rmse <= 2e-5);
        try std.testing.expect(metrics.max_consumer_output_rmse <= 2e-5);
    }

    const descriptors = attention.DescriptorStorage{ .spans = descriptor_spans.ref(), .heads = descriptor_heads.ref() };
    const common = attention.Request{
        .view = view,
        .current_epoch = view.epoch,
        .query_start = source_rows - query_rows,
        .query_rows = query_rows,
        .query_heads = 4,
        .scale = scale,
        .descriptors = descriptors,
    };
    try serving.attention(common, consumer_a_queries.ref(), packed_keys.ref(), packed_values.ref(), packed_bias.ref(), consumer_a_serial.ref());
    var consumer_b_request = common;
    consumer_b_request.query_heads = 6;
    try serving.attention(consumer_b_request, consumer_b_queries.ref(), packed_keys.ref(), packed_values.ref(), packed_bias.ref(), consumer_b_serial.ref());
    try expectConsumerReference(dim, consumer_a_queries.floats(), consumer_a_serial.floats(), 4, 5e-5);
    try expectConsumerReference(dim, consumer_b_queries.floats(), consumer_b_serial.floats(), 6, 5e-5);

    // A deliberately admitted three-way decode exercises the merge kernel;
    // the three-row flash dispatch crosses causal exact-row boundaries.
    inline for (.{ .{ 4, &consumer_a_queries, &consumer_a_serial }, .{ 6, &consumer_b_queries, &consumer_b_serial } }) |case| {
        const query_heads: u32 = case[0];
        const queries = case[1];
        const serial = case[2];
        const split_count: u32 = 3;
        try std.testing.expect(split_count > 1);
        var scratch = try serving.core.buffer(try attention.splitDecodeScratchBytes(query_heads, @intCast(dim), split_count));
        defer scratch.deinit();
        var request = common;
        request.query_heads = query_heads;
        request.query_start = source_rows - 1;
        request.query_rows = 1;
        request.split_decode = .{ .scratch = scratch.ref(), .split_count = split_count };
        const vector_elements = @as(usize, query_heads) * dim;
        const last_row_offset = (query_rows - 1) * vector_elements * @sizeOf(f32);
        const decode_queries = try queries.ref().slice(last_row_offset, vector_elements * @sizeOf(f32));
        const decode_output = try route_output.ref().slice(0, vector_elements * @sizeOf(f32));
        try serving.attention(request, decode_queries, packed_keys.ref(), packed_values.ref(), packed_bias.ref(), decode_output);
        try expectOutputsClose(serial.floats()[(query_rows - 1) * vector_elements ..], route_output.floats()[0..vector_elements], 8e-5);
        request.split_decode = null;
        request.flash_prefill = true;
        request.query_start = source_rows - query_rows;
        request.query_rows = query_rows;
        const flash_output = try route_output.ref().slice(0, query_rows * vector_elements * @sizeOf(f32));
        try serving.attention(request, queries.ref(), packed_keys.ref(), packed_values.ref(), packed_bias.ref(), flash_output);
        try expectOutputsClose(serial.floats(), route_output.floats()[0 .. query_rows * vector_elements], 0.001);
    }
}

test "context native packed parity validates rounded multihead candidate through actual serving ABI" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    try fixture(256);
    try fixture(512);
}

// The former split/flash benchmark's independent oracle and physical-layout
// checks live here without timing loops or performance evidence output.
const SplitFlashParity = struct {
    const shared = @import("metal_generated_kernels.zig");
    const Geometry = enum { global_exact, global_compacted, local_wrapped };
    const Case = struct { dim: u32, window: u32, logical: u32, geometry: Geometry };

    fn digest(bytes: []const u8) [32]u8 {
        var result: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &result, .{});
        return result;
    }

    fn sample(i: usize, salt: u32) f32 {
        var x: u32 = @intCast(i);
        x +%= salt *% 0x9e3779b9;
        x ^= x >> 16;
        x *%= 0x7feb352d;
        x ^= x >> 15;
        return (@as(f32, @floatFromInt(x % 2001)) - 1000) / 2048;
    }

    fn oracle(view: views.View, q: []const f32, k: []const f16, v: []const f16, mass: []const f32, position: u32, window: u32, scale: f32, softcap: f32, out: []f64) !void {
        const q_heads: usize = 8;
        const dim: usize = view.head_dim;
        @memset(out, 0);
        for (0..q_heads) |qh| {
            const kh = qh / (q_heads / view.kv_heads);
            const head = view.heads[kh];
            var maximum = -std.math.inf(f64);
            var denominator: f64 = 0;
            var logits = try std.testing.allocator.alloc(f64, head.capacity_rows);
            defer std.testing.allocator.free(logits);
            var offsets = try std.testing.allocator.alloc(usize, head.capacity_rows);
            defer std.testing.allocator.free(offsets);
            var count: usize = 0;
            for (view.spans[head.span_start..][0..head.span_count]) |span| for (0..span.row_count) |r| {
                const key_pos = span.logical_start + r;
                if (span.kind == .exact and (key_pos > position or (window != 0 and position -| key_pos >= window))) continue;
                const off = (@as(usize, head.kv_start + span.physical_start + @as(u32, @intCast(r)))) * dim;
                var score: f64 = 0;
                for (0..dim) |d| score += @as(f64, q[qh * dim + d]) * @as(f64, k[off + d]);
                score *= scale;
                if (softcap > 0) score = softcap * std.math.tanh(score / softcap);
                if (span.kind == .compacted) score += mass[head.bias_start + span.bias_start + r];
                logits[count] = score;
                offsets[count] = off;
                maximum = @max(maximum, score);
                count += 1;
            };
            for (logits[0..count]) |*score| {
                score.* = @exp(score.* - maximum);
                denominator += score.*;
            }
            for (logits[0..count], offsets[0..count]) |weight, off| {
                for (0..dim) |d| out[qh * dim + d] += weight * @as(f64, v[off + d]);
            }
            for (out[qh * dim ..][0..dim]) |*value| value.* /= denominator;
        }
    }

    const Descriptor = struct {
        spans: [32]views.Span = undefined,
        span_count: usize,
        heads: [2]views.Head,
        capacity: u32,
        bias_rows: u32,

        fn view(self: *const Descriptor, case: Case) views.View {
            return .{ .version = views.abi_version, .epoch = 7, .logical_tokens = case.logical, .capacity_rows = self.capacity, .bias_rows = self.bias_rows, .kv_heads = 2, .head_dim = case.dim, .heads = &self.heads, .spans = self.spans[0..self.span_count] };
        }
    };

    fn buildDescriptor(case: Case) !Descriptor {
        const compacted = case.geometry == .global_compacted;
        const local = case.geometry == .local_wrapped;
        if (compacted) {
            if (case.logical != 8128) return error.InvalidContextView;
            var result = Descriptor{ .span_count = 32, .heads = undefined, .capacity = 0, .bias_rows = 0 };
            const retained = [_]u32{ 128, 160 };
            var span_cursor: u32 = 0;
            var kv_cursor: u32 = 0;
            var bias_cursor: u32 = 0;
            for (0..2) |head_index| {
                const rows = retained[head_index];
                const head_span_start = span_cursor;
                var physical: u32 = 0;
                var bias: u32 = 0;
                for (0..15) |block| {
                    result.spans[span_cursor] = .{
                        .physical_start = physical,
                        .row_count = rows,
                        .logical_start = @intCast(block * 512),
                        .logical_count = 512,
                        .kind = .compacted,
                        .bias_start = bias,
                    };
                    span_cursor += 1;
                    physical += rows;
                    bias += rows;
                }
                result.spans[span_cursor] = .{ .physical_start = physical, .row_count = 448, .logical_start = 15 * 512, .logical_count = 448 };
                span_cursor += 1;
                physical += 448;
                result.heads[head_index] = .{ .kv_start = kv_cursor, .capacity_rows = physical, .bias_start = bias_cursor, .bias_rows = bias, .span_start = head_span_start, .span_count = 16 };
                kv_cursor += physical;
                bias_cursor += bias;
            }
            result.capacity = kv_cursor;
            result.bias_rows = bias_cursor;
            _ = try result.view(case).validate();
            return result;
        }

        const exact_rows: u32 = if (local) 256 else 4128;
        const retained: u32 = if (local) 256 else 4000;
        const gap = [_]u32{ 3, 5 };
        var result = Descriptor{ .span_count = 4, .heads = undefined, .capacity = 0, .bias_rows = 0 };
        const closed: u32 = if (local) case.logical - 512 else case.logical - exact_rows;
        var kv_cursor: u32 = 0;
        for (0..2) |head_index| {
            const span_start: usize = head_index * 2;
            const capacity = retained + gap[head_index] + exact_rows;
            result.spans[span_start] = .{ .physical_start = 0, .row_count = retained, .logical_start = if (local) closed else 0, .logical_count = retained };
            result.spans[span_start + 1] = .{ .physical_start = retained + gap[head_index], .row_count = exact_rows, .logical_start = closed, .logical_count = exact_rows };
            if (local) {
                result.spans[span_start].physical_start = retained + gap[head_index];
                result.spans[span_start + 1] = .{ .physical_start = 0, .row_count = 256, .logical_start = closed + 256, .logical_count = 256 };
            } else {
                result.spans[span_start].logical_count = 4000;
                result.spans[span_start + 1].logical_start = 4000;
            }
            result.heads[head_index] = .{ .kv_start = kv_cursor, .capacity_rows = capacity, .bias_start = 0, .bias_rows = 0, .span_start = @intCast(span_start), .span_count = 2 };
            kv_cursor += capacity;
        }
        result.capacity = kv_cursor;
        _ = try result.view(case).validate();
        return result;
    }

    fn check(case: Case) !void {
        var kernels = try attention.Kernels.initFlash();
        defer kernels.deinit();
        const descriptor = try buildDescriptor(case);
        const view = descriptor.view(case);
        const spans = view.spans;
        const heads = descriptor.heads;
        const capacity = descriptor.capacity;
        const bias0 = heads[0].bias_rows;
        const bias1 = heads[1].bias_rows;
        _ = try view.validate();
        const dim: usize = case.dim;
        var query = try shared.Buffer.init(kernels.core.runtime, 8 * dim * 4);
        defer query.deinit();
        var keys = try shared.Buffer.init(kernels.core.runtime, @as(usize, capacity) * dim * 2);
        defer keys.deinit();
        var values = try shared.Buffer.init(kernels.core.runtime, @as(usize, capacity) * dim * 2);
        defer values.deinit();
        var mass = try shared.Buffer.init(kernels.core.runtime, @max(1, @as(usize, bias0 + bias1)) * 4);
        defer mass.deinit();
        var split_out = try shared.Buffer.init(kernels.core.runtime, 8 * dim * 4);
        defer split_out.deinit();
        var flash_out = try shared.Buffer.init(kernels.core.runtime, 8 * dim * 4);
        defer flash_out.deinit();
        var span_buf = try shared.Buffer.init(kernels.core.runtime, spans.len * @sizeOf(views.Span));
        defer span_buf.deinit();
        var head_buf = try shared.Buffer.init(kernels.core.runtime, @sizeOf(@TypeOf(heads)));
        defer head_buf.deinit();
        @memcpy(span_buf.bytes[0 .. spans.len * @sizeOf(views.Span)], std.mem.sliceAsBytes(spans));
        @memcpy(head_buf.bytes[0..@sizeOf(@TypeOf(heads))], std.mem.asBytes(&heads));
        for (query.floats(), 0..) |*x, i| x.* = sample(i, 1);
        const kh = std.mem.bytesAsSlice(f16, @as([]align(2) u8, @alignCast(keys.bytes[0..keys.logical_bytes])));
        const vh = std.mem.bytesAsSlice(f16, @as([]align(2) u8, @alignCast(values.bytes[0..values.logical_bytes])));
        for (kh, 0..) |*x, i| x.* = @floatCast(sample(i, 2));
        for (vh, 0..) |*x, i| x.* = @floatCast(sample(i, 3));
        for (mass.floats(), 0..) |*x, i| x.* = @log(1.25 + @abs(sample(i, 4)));
        const plan = try attention.splitDecodePlan(capacity, 8, case.dim);
        var scratch = try shared.Buffer.init(kernels.core.runtime, plan.scratch_bytes);
        defer scratch.deinit();
        const scale: f32 = 1 / @sqrt(@as(f32, @floatFromInt(case.dim)));
        const descriptors: attention.DescriptorStorage = .{ .spans = span_buf.ref(), .heads = head_buf.ref() };
        const request = attention.Request{ .view = view, .current_epoch = 7, .query_start = case.logical - 1, .query_rows = 1, .query_heads = 8, .sliding_window = case.window, .scale = scale, .logit_softcap = 0.75, .split_decode = .{ .scratch = scratch.ref(), .split_count = plan.split_count }, .descriptors = descriptors };
        const before_hashes = [_][32]u8{ digest(query.bytes), digest(keys.bytes), digest(values.bytes), digest(mass.bytes), digest(head_buf.bytes), digest(span_buf.bytes) };
        const params = shared.Params{ .op = views.abi_version, .n = 1, .dim = case.dim, .groups = 8, .position = case.logical - 1, .stride = 2, .mode = case.window, .index = @intCast(spans.len), .epsilon = 0.75, .scale = scale };
        const inputs = [_]shared.Ref{ query.ref(), keys.ref(), values.ref(), mass.ref(), span_buf.ref(), kernels.status.ref(), head_buf.ref() };
        @memset(kernels.status.bytes[0..4], 0);
        try kernels.attention(request, query.ref(), keys.ref(), values.ref(), mass.ref(), split_out.ref());
        try std.testing.expectEqual(@as(u32, 0), std.mem.bytesToValue(u32, kernels.status.bytes[0..4]));
        @memset(kernels.status.bytes[0..4], 0);
        _ = try kernels.flash_prefill.?.dispatch(&kernels.core, params, &inputs, flash_out.ref());
        try std.testing.expectEqual(@as(u32, 0), std.mem.bytesToValue(u32, kernels.status.bytes[0..4]));
        const expected = try std.testing.allocator.alloc(f64, 8 * dim);
        defer std.testing.allocator.free(expected);
        try oracle(view, query.floats(), kh, vh, mass.floats(), case.logical - 1, case.window, scale, 0.75, expected);
        var max_split: f64 = 0;
        var max_flash: f64 = 0;
        for (expected, split_out.floats(), flash_out.floats()) |truth, s, f| {
            try std.testing.expect(std.math.isFinite(s) and std.math.isFinite(f));
            max_split = @max(max_split, @abs(truth - s));
            max_flash = @max(max_flash, @abs(truth - f));
        }
        try std.testing.expect(max_split <= 0.003);
        try std.testing.expect(max_flash <= 0.003);
        const after_hashes = [_][32]u8{ digest(query.bytes), digest(keys.bytes), digest(values.bytes), digest(mass.bytes), digest(head_buf.bytes), digest(span_buf.bytes) };
        try std.testing.expectEqualSlices([32]u8, &before_hashes, &after_hashes);
    }
};

test "context native packed descriptor validates fifteen blocks and exact tail" {
    const case = SplitFlashParity.Case{ .dim = 512, .window = 0, .logical = 8128, .geometry = .global_compacted };
    const descriptor = try SplitFlashParity.buildDescriptor(case);
    const view = descriptor.view(case);
    try std.testing.expectEqual(@as(u32, 2848), try view.validate());
    try std.testing.expectEqual(@as(usize, 32), view.spans.len);
    try std.testing.expectEqual(@as(u32, 15 * 128), view.heads[0].bias_rows);
    try std.testing.expectEqual(@as(u32, 15 * 160), view.heads[1].bias_rows);
    for (0..2) |head_index| {
        const head_spans = view.headSpans(head_index);
        for (head_spans[0..15], 0..) |span, block| {
            try std.testing.expectEqual(views.SpanKind.compacted, span.kind);
            try std.testing.expectEqual(@as(u32, @intCast(block * 512)), span.logical_start);
            try std.testing.expectEqual(@as(u32, 512), span.logical_count);
            if (block != 0) try std.testing.expectEqual(head_spans[block - 1].physical_start + head_spans[block - 1].row_count, span.physical_start);
        }
        const tail = head_spans[15];
        try std.testing.expectEqual(views.SpanKind.exact, tail.kind);
        try std.testing.expectEqual(@as(u32, 7680), tail.logical_start);
        try std.testing.expectEqual(@as(u32, 448), tail.logical_count);
        try std.testing.expectEqual(head_spans[14].physical_start + head_spans[14].row_count, tail.physical_start);
    }
}

test "context native packed split and flash match independent reference" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    try SplitFlashParity.check(.{ .dim = 512, .window = 0, .logical = 8128, .geometry = .global_exact });
    try SplitFlashParity.check(.{ .dim = 512, .window = 0, .logical = 8128, .geometry = .global_compacted });
    try SplitFlashParity.check(.{ .dim = 256, .window = 512, .logical = 8128, .geometry = .local_wrapped });
}
