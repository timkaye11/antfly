const std = @import("std");
const platform = @import("antfly_platform");
const attention = @import("context_kernels.zig");
const shared = @import("metal_generated_kernels.zig");
const views = @import("../runtime/kv/context_view.zig");

const Geometry = enum { global_exact, global_compacted, local_wrapped };
const Case = struct { label: []const u8, dim: u32, window: u32, logical: u32, geometry: Geometry };

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

test "synthetic global compacted descriptor validates fifteen blocks and exact tail" {
    const case = Case{ .label = "descriptor_only", .dim = 512, .window = 0, .logical = 8128, .geometry = .global_compacted };
    const descriptor = try buildDescriptor(case);
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

fn runCase(case: Case) !void {
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
    var sample_index: usize = 0;
    while (sample_index < 10) : (sample_index += 1) {
        const flash_first = sample_index % 2 == 1;
        var split_gpu: u64 = 0;
        var split_wait: u64 = 0;
        var split_wall: u64 = 0;
        var flash_timing: shared.PairTiming = undefined;
        for (0..2) |order| {
            const do_flash = (order == 0) == flash_first;
            @memset(kernels.status.bytes[0..4], 0);
            if (do_flash) flash_timing = try kernels.flash_prefill.?.dispatch(&kernels.core, params, &inputs, flash_out.ref()) else {
                const before = kernels.core.pair_metrics;
                try kernels.attention(request, query.ref(), keys.ref(), values.ref(), mass.ref(), split_out.ref());
                split_gpu = kernels.core.pair_metrics.gpu_nanos - before.gpu_nanos;
                split_wait = kernels.core.pair_metrics.wait_nanos - before.wait_nanos;
                split_wall = kernels.core.pair_metrics.wall_nanos - before.wall_nanos;
            }
            try std.testing.expectEqual(@as(u32, 0), std.mem.bytesToValue(u32, kernels.status.bytes[0..4]));
        }
        if (sample_index != 0) std.debug.print("synthetic_q1_split_flash_sample {{\"label\":\"{s}\",\"sample\":{d},\"split_pairs\":1,\"split_kernel_dispatches\":2,\"split_gpu_ns\":{d},\"split_wait_ns\":{d},\"split_wall_ns\":{d},\"flash_kernel_dispatches\":1,\"flash_gpu_ns\":{d},\"flash_wait_ns\":{d},\"flash_wall_ns\":{d}}}\n", .{ case.label, sample_index - 1, split_gpu, split_wait, split_wall, flash_timing.gpu_nanos, flash_timing.wait_nanos, flash_timing.wall_nanos });
    }
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
    const descriptor_hex = std.fmt.bytesToHex(digest(std.mem.sliceAsBytes(spans)), .lower);
    std.debug.print("synthetic_q1_split_flash_summary {{\"label\":\"{s}\",\"descriptor_sha256\":\"{s}\",\"samples\":9,\"split_max_abs\":{d},\"flash_max_abs\":{d}}}\n", .{ case.label, &descriptor_hex, max_split, max_flash });
}

test "synthetic E4 q1 same-buffer split versus flash diagnostic" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    if (!platform.env.getenvBool("ANTFLY_METAL_SYNTHETIC_PACKED_Q1_SPLIT_FLASH_BENCH")) return error.SkipZigTest;
    try runCase(.{ .label = "synthetic_global_exact_8128_d512", .dim = 512, .window = 0, .logical = 8128, .geometry = .global_exact });
    try runCase(.{ .label = "synthetic_global_compacted_8128_d512", .dim = 512, .window = 0, .logical = 8128, .geometry = .global_compacted });
    try runCase(.{ .label = "synthetic_local_wrapped_512_d256", .dim = 256, .window = 512, .logical = 8128, .geometry = .local_wrapped });
}
