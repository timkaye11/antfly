const builtin = @import("builtin");
const std = @import("std");
const solver = @import("attention_match.zig");

extern "c" fn @"cblas_dgemm$NEWLAPACK"(c_int, c_int, c_int, c_int, c_int, c_int, f64, [*]const f64, c_int, [*]const f64, c_int, f64, [*]f64, c_int) void;

pub const policy = "accelerate_fp64_probability_scan_v1|tile=64|gemm=f64|softcap_bias_softmax=scalar_f64|publication=f32";
pub const tile_rows: usize = 64;

pub const Context = struct {
    gemm_fn: ?GemmFn = null,
    test_owner: ?*anyopaque = null,
    calls: usize = 0,
    workspace_peak_bytes: usize = 0,
};

pub const GemmFn = *const fn (*Context, c_int, c_int, c_int, f64, [*]const f64, c_int, [*]const f64, c_int, f64, [*]f64, c_int) void;
pub const SinkFn = *const fn (?*anyopaque, usize, []const f32) anyerror!void;
pub const Sink = struct { context: ?*anyopaque = null, row_fn: SinkFn };
pub const Backend = struct {
    context: ?*anyopaque,
    scan_fn: *const fn (?*anyopaque, std.mem.Allocator, solver.Head, []const f32, usize, usize, solver.Control, Sink) anyerror!void,

    pub fn run(self: @This(), allocator: std.mem.Allocator, head: solver.Head, queries: []const f32, external_live_bytes: usize, combined_cap: usize, control: solver.Control, sink: Sink) !void {
        return self.scan_fn(self.context, allocator, head, queries, external_live_bytes, combined_cap, control, sink);
    }
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn checkedInt(value: usize) !c_int {
    if (value == 0 or value > std.math.maxInt(c_int)) return error.InvalidAttentionShape;
    return @intCast(value);
}

pub fn workspaceBytes(rows: usize, dim: usize, query_rows: usize) !usize {
    _ = try checkedInt(rows);
    _ = try checkedInt(dim);
    if (query_rows == 0) return error.InvalidAttentionShape;
    const tile = @min(query_rows, tile_rows);
    const q = try mul(tile, dim);
    const logits = try mul(tile, rows);
    return add(try mul(try add(q, logits), @sizeOf(f64)), try mul(logits, @sizeOf(f32)));
}

pub fn scan(
    context: *Context,
    allocator: std.mem.Allocator,
    head: solver.Head,
    queries: []const f32,
    external_live_bytes: usize,
    combined_cap: usize,
    control: solver.Control,
    sink: Sink,
) !void {
    try control.check();
    try head.validate();
    if (queries.len == 0 or queries.len % head.dim != 0 or external_live_bytes > combined_cap) return error.InvalidAttentionShape;
    const query_rows = queries.len / head.dim;
    const workspace = try workspaceBytes(head.rows, head.dim, query_rows);
    if (workspace > combined_cap - external_live_bytes) return error.CompactionWorkspaceLimit;
    const max_tile = @min(query_rows, tile_rows);
    const query64 = try allocator.alloc(f64, try mul(max_tile, head.dim));
    defer allocator.free(query64);
    const logits64 = try allocator.alloc(f64, try mul(max_tile, head.rows));
    defer allocator.free(logits64);
    const probabilities32 = try allocator.alloc(f32, try mul(max_tile, head.rows));
    defer allocator.free(probabilities32);

    var begin: usize = 0;
    while (begin < query_rows) : (begin += max_tile) {
        try control.check();
        const count = @min(max_tile, query_rows - begin);
        const q64 = query64[0 .. count * head.dim];
        const logits = logits64[0 .. count * head.rows];
        for (q64, queries[begin * head.dim ..][0 .. count * head.dim]) |*out, value| {
            if (!std.math.isFinite(value)) return error.NonFiniteAttention;
            out.* = value;
        }
        const nq = try checkedInt(count);
        const nr = try checkedInt(head.rows);
        const nd = try checkedInt(head.dim);
        if (context.gemm_fn) |gemm|
            gemm(context, nq, nr, nd, head.scale, q64.ptr, nd, head.keys.ptr, nd, 0, logits.ptr, nr)
        else if (comptime builtin.os.tag == .macos)
            @"cblas_dgemm$NEWLAPACK"(101, 111, 112, nq, nr, nd, head.scale, q64.ptr, nd, head.keys.ptr, nd, 0, logits.ptr, nr)
        else
            return error.UnsupportedOperation;
        try control.check();
        for (0..count) |local_row| {
            const row = logits[local_row * head.rows ..][0..head.rows];
            var maximum = -std.math.inf(f64);
            for (row, 0..) |*value, source_row| {
                var x = value.*;
                if (head.softcap > 0) x = head.softcap * std.math.tanh(x / head.softcap);
                x += if (head.bias) |bias| bias[source_row] else 0;
                if (!std.math.isFinite(x)) return error.NonFiniteAttention;
                value.* = x;
                maximum = @max(maximum, x);
            }
            var sum: f64 = 0;
            for (row) |*value| {
                value.* = @exp(value.* - maximum);
                sum += value.*;
            }
            if (!std.math.isFinite(sum) or sum <= 0) return error.NonFiniteAttention;
            const out = probabilities32[local_row * head.rows ..][0..head.rows];
            for (row, out) |value, *rounded| rounded.* = @floatCast(value / sum);
            try sink.row_fn(sink.context, begin + local_row, out);
        }
    }
    context.calls = std.math.add(usize, context.calls, 1) catch return error.InvalidAttentionShape;
    context.workspace_peak_bytes = @max(context.workspace_peak_bytes, workspace);
}

fn scanErased(raw: ?*anyopaque, allocator: std.mem.Allocator, head: solver.Head, queries: []const f32, external_live_bytes: usize, combined_cap: usize, control: solver.Control, sink: Sink) !void {
    const context: *Context = @ptrCast(@alignCast(raw orelse return error.InvalidAttentionShape));
    return scan(context, allocator, head, queries, external_live_bytes, combined_cap, control, sink);
}

pub fn backend(context: *Context) Backend {
    return .{ .context = context, .scan_fn = scanErased };
}

const Fixture = struct {
    rows: [2][3]f32 = undefined,
    seen: usize = 0,
    fn consume(raw: ?*anyopaque, row: usize, probabilities: []const f32) !void {
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        if (row != self.seen or probabilities.len != 3) return error.TestUnexpectedResult;
        @memcpy(self.rows[row][0..], probabilities);
        self.seen += 1;
    }
};

const Mock = struct {
    fn gemm(_: *Context, m: c_int, n: c_int, k: c_int, alpha: f64, a: [*]const f64, lda: c_int, b: [*]const f64, ldb: c_int, beta: f64, c: [*]f64, ldc: c_int) void {
        for (0..@intCast(m)) |i| for (0..@intCast(n)) |j| {
            var sum: f64 = 0;
            for (0..@intCast(k)) |x| sum += a[i * @as(usize, @intCast(lda)) + x] * b[j * @as(usize, @intCast(ldb)) + x];
            const at = i * @as(usize, @intCast(ldc)) + j;
            c[at] = alpha * sum + if (beta == 0) 0 else beta * c[at];
        };
    }
};

fn fixture(allocator: std.mem.Allocator, cap: usize, control: solver.Control) !void {
    const head: solver.Head = .{ .keys = &.{ 0.25, -0.5, 1.25, 0.75, -1.5, 0.125 }, .values = &.{ 0, 0, 0, 0, 0, 0 }, .bias = &.{ 0.1, -0.2, 0.05 }, .rows = 3, .dim = 2, .scale = 0.625, .softcap = 1.75 };
    const queries = [_]f32{ 0.3, -1.1, 2, 0.125 };
    var context: Context = .{ .gemm_fn = Mock.gemm };
    var actual: Fixture = .{};
    try scan(&context, allocator, head, &queries, 0, cap, control, .{ .context = &actual, .row_fn = Fixture.consume });
    try std.testing.expectEqual(@as(usize, 2), actual.seen);
    var expected: [3]f64 = undefined;
    var ignored: [2]f64 = undefined;
    for (0..2) |row| {
        const query64 = [_]f64{ queries[row * 2], queries[row * 2 + 1] };
        _ = try solver.attend(head, &query64, &expected, &ignored);
        for (expected, actual.rows[row]) |want, got| try std.testing.expectApproxEqAbs(@as(f32, @floatCast(want)), got, 2e-6);
    }
}

test "accelerate probability scan matches scalar rounded probabilities with softcap and bias" {
    try fixture(std.testing.allocator, 1 << 20, .{});
}

test "accelerate probability scan rejects cap cancellation and allocation failures atomically" {
    try std.testing.expectError(error.CompactionWorkspaceLimit, fixture(std.testing.failing_allocator, 1, .{}));
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, fixture(std.testing.allocator, 1 << 20, .{ .check_fn = Cancel.check }));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, fixture, .{ @as(usize, 1 << 20), solver.Control{} });
}
