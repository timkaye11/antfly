// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Default-off, value-phase-only adapter for chronological FP64 TSQR.
//! Attention, selection, heldout validation, and their Metal backend remain
//! outside this contract.
const std = @import("std");
const solver = @import("attention_match.zig");
const diagnostic = @import("attention_match_joint_chunk_mass_diagnostic.zig");
const tsqr = @import("attention_match_fp64_tsqr_value_fit.zig");

pub const policy = "e2_dense512_bounded_fp64_tsqr_value_fit_v2";
pub const max_queries: usize = 65_536;
pub const max_selected_rows: usize = 128;
pub const sealed_batch_rows: usize = 64;

pub const Context = struct {
    calls: usize = 0,
    last_queries: usize = 0,
    last_selected_rows: usize = 0,
    last_dim: usize = 0,
    last_workspace_bytes: usize = 0,
    workspace_peak_bytes: usize = 0,
    cancellation_checks: usize = 0,

    fn checkedShape(q: usize, m: usize, d: usize, ridge: f64) !void {
        if (q == 0 or q > max_queries or m == 0 or m > max_selected_rows or d == 0 or
            !std.math.isFinite(ridge) or ridge < 0) return error.InvalidFitConfig;
    }

    fn workspace(raw: ?*anyopaque, q: usize, m: usize, d: usize, ridge: f64) !usize {
        _ = raw orelse return error.InvalidCallbackContext;
        try checkedShape(q, m, d, ridge);
        return (try tsqr.workspaceBytes(m, d, sealed_batch_rows, 0)).combined_bytes;
    }

    const SliceSource = struct {
        a: []const f64,
        y: []const f64,
        q: usize,
        m: usize,
        d: usize,
        owner: *Context,

        fn fill(raw: ?*anyopaque, first: usize, count: usize, a: []f64, y: []f64, control: solver.Control) !void {
            const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
            if (first > self.q or count > self.q - first or a.len != count * self.m or y.len != count * self.d)
                return error.InvalidAttentionShape;
            self.owner.cancellation_checks += 1;
            try control.check();
            @memcpy(a, self.a[first * self.m ..][0 .. count * self.m]);
            @memcpy(y, self.y[first * self.d ..][0 .. count * self.d]);
        }
    };

    fn fit(raw: ?*anyopaque, allocator: std.mem.Allocator, a: []const f64, y: []const f64, q: usize, m: usize, d: usize, ridge: f64, control: solver.Control, out: []f64) !void {
        const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
        try checkedShape(q, m, d, ridge);
        if (a.len != (std.math.mul(usize, q, m) catch return error.InvalidAttentionShape) or
            y.len != (std.math.mul(usize, q, d) catch return error.InvalidAttentionShape) or
            out.len != (std.math.mul(usize, m, d) catch return error.InvalidAttentionShape))
            return error.InvalidAttentionShape;
        const workspace_bytes = try workspace(self, q, m, d, ridge);
        var source: SliceSource = .{ .a = a, .y = y, .q = q, .m = m, .d = d, .owner = self };
        try tsqr.fit(allocator, q, m, d, ridge, sealed_batch_rows, 0, workspace_bytes, .{
            .context = &source,
            .fill_fn = SliceSource.fill,
        }, control, out);
        self.calls += 1;
        self.last_queries = q;
        self.last_selected_rows = m;
        self.last_dim = d;
        self.last_workspace_bytes = workspace_bytes;
        self.workspace_peak_bytes = @max(self.workspace_peak_bytes, workspace_bytes);
    }

    pub fn valueBackend(self: *@This()) solver.ValueBackend {
        return .{ .context = self, .recompute_fixed_original_tiled = true, .workspace_fn = workspace, .fit_fn = fit };
    }
};

test "context E2 value backend preserves attention backend and replaces dense phase bound" {
    const q: usize = 30_720;
    const m: usize = 128;
    const d: usize = 512;
    const external: usize = 753_439_600;
    const cap: usize = 1 << 30;
    var context: Context = .{};
    const Attention = struct {
        var calls: usize = 0;
        fn workspace(_: ?*anyopaque, n: usize, selected: usize, queries: usize, dim: usize) !usize {
            calls += 1;
            if (n != 32_512 or selected != 1 or queries != 64 or dim != 512) return error.InvalidFitConfig;
            return 0;
        }
    };
    const cfg: solver.Config = .{
        .target_rows = m,
        .backend = .{ .workspace_fn = Attention.workspace },
        .value_backend = context.valueBackend(),
    };
    const head: solver.Head = .{ .keys = &.{}, .values = &.{}, .bias = null, .rows = 32_512, .dim = d, .scale = 1 };
    const plan = try diagnostic.workspaceBytesShape(head, q, 1, m, m, cfg, false, .identity_chunk_coefficients_v18, false);
    // The uncached phase retains its original output and attention tile.
    try std.testing.expectEqual(@as(usize, 301_286_456), plan.phase2_bytes);
    const cached_plan = try diagnostic.workspaceBytesShape(head, q, 1, m, m, cfg, true, .identity_chunk_coefficients_v18, false);
    try std.testing.expectEqual(@as(usize, 158_549_048), cached_plan.phase2_bytes);
    try std.testing.expectEqual(@as(usize, 913_829_800), try std.math.add(usize, external, cached_plan.required));
    try std.testing.expectEqual(@as(usize, 1_841_152), try solver.valueWorkspaceBytes(q, m, d, cfg.ridge, cfg));
    try std.testing.expectEqual(@as(usize, 1_056_567_208), try std.math.add(usize, external, plan.required));
    try std.testing.expect(external + plan.required <= cap);
    try std.testing.expectError(error.CompactionWorkspaceLimit, admit(external, plan.required, external + plan.required - 1));
    try std.testing.expectEqual(@as(usize, 1), Attention.calls);
    try std.testing.expectError(error.InvalidFitConfig, context.valueBackend().workspace_fn(&context, max_queries + 1, m, d, cfg.ridge));
    _ = try context.valueBackend().workspace_fn(&context, max_queries, m, d, cfg.ridge);
}

fn admit(external: usize, workspace_bytes: usize, cap: usize) !void {
    const total = std.math.add(usize, external, workspace_bytes) catch return error.CompactionWorkspaceLimit;
    if (total > cap) return error.CompactionWorkspaceLimit;
}

test "context E2 TSQR value backend agrees with dense oracle and late cancel is atomic" {
    const q: usize = 7;
    const m: usize = 3;
    const d: usize = 2;
    const a = [_]f64{ 1, 0, 2, 0, 1, 1, 1, 1, 0, 2, -1, 1, -1, 2, 0.5, 0.25, 1.5, -0.5, 1.25, 0.75, 2 };
    const y = [_]f64{ 2, -1, 1, 3, 4, 2, -2, 1, 3, 0.5, 1.5, -3, 2.5, 1 };
    var context: Context = .{};
    var actual: [m * d]f64 = undefined;
    var expected: [m * d]f64 = undefined;
    var cfg: solver.Config = .{ .target_rows = m, .ridge = 1e-8, .max_workspace_bytes = 1 << 20 };
    cfg.value_backend = context.valueBackend();
    try solver.fitValuesWithBackendForTest(std.testing.allocator, &a, &y, q, m, d, cfg.ridge, cfg, &actual);
    try solver.fitValuesReferenceForTest(std.testing.allocator, &a, &y, q, m, d, cfg.ridge, .{}, &expected);
    for (expected, actual) |want, got| try std.testing.expectApproxEqRel(want, got, 5e-12);
    try std.testing.expectEqual(@as(usize, 1), context.calls);
    try std.testing.expect(context.workspace_peak_bytes > 0);
    try std.testing.expect(context.cancellation_checks > 0);

    const required = try solver.valueWorkspaceBytes(q, m, d, cfg.ridge, cfg);
    cfg.max_workspace_bytes = required - 1;
    var refused: [m * d]f64 = @splat(-8);
    try std.testing.expectError(error.CompactionWorkspaceLimit, solver.fitValuesWithBackendForTest(std.testing.allocator, &a, &y, q, m, d, cfg.ridge, cfg, &refused));
    for (refused) |value| try std.testing.expectEqual(@as(f64, -8), value);
    try std.testing.expectEqual(@as(usize, 1), context.calls);

    const LateCancel = struct {
        checks: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.checks += 1;
            if (self.checks >= 12) return error.Cancelled;
        }
    };
    var cancel: LateCancel = .{};
    var unpublished: [m * d]f64 = @splat(-9);
    cfg.max_workspace_bytes = 1 << 20;
    cfg.control = .{ .context = &cancel, .check_fn = LateCancel.check };
    try std.testing.expectError(error.Cancelled, solver.fitValuesWithBackendForTest(std.testing.allocator, &a, &y, q, m, d, cfg.ridge, cfg, &unpublished));
    for (unpublished) |value| try std.testing.expectEqual(@as(f64, -9), value);
    try std.testing.expectEqual(@as(usize, 1), context.calls);
    try std.testing.expect(context.cancellation_checks > 1);
}
