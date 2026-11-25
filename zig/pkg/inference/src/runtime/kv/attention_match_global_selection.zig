// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Raw OMP selection in full-original denominator units for cumulative admission.
const std = @import("std");
const solver = @import("attention_match.zig");

pub const Selection = solver.MassSelectionResult;
pub const selection_policy = solver.mass_selection_policy;

fn checkedAdd(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}
fn checkedMul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}

pub fn selectFeaturesRaw(allocator: std.mem.Allocator, probabilities: []const f64, rho: []const f64, rows: usize, cfg: solver.Config) !Selection {
    try solver.validateConfigDiagnostic(cfg, rows);
    if (rho.len == 0 or probabilities.len != try checkedMul(rho.len, rows) or cfg.target_rows >= rows or cfg.selection != .residual_omp or cfg.mass_solver != .active_set) return error.InvalidFitConfig;
    const q = rho.len;
    const caller_bytes = try checkedAdd(try checkedMul(q, @sizeOf(f64)), try checkedMul(cfg.target_rows, @sizeOf(usize) + @sizeOf(f64)));
    const nested_bytes = try solver.massSelectionWorkspaceBytes(rows, cfg.target_rows, q);
    if (try checkedAdd(caller_bytes, nested_bytes) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    var nested = cfg;
    nested.max_workspace_bytes -= caller_bytes;
    return solver.selectMassTargetPolicyDiagnostic(allocator, probabilities, rho, rows, nested, .raw);
}

pub fn selectStreaming(allocator: std.mem.Allocator, original: solver.Head, chunk_start: usize, chunk_rows: usize, train: []const f64, cfg: solver.Config) !Selection {
    try original.validate();
    try solver.validateConfigDiagnostic(cfg, original.rows);
    if (chunk_rows == 0 or chunk_start > original.rows or chunk_rows > original.rows - chunk_start or train.len == 0 or train.len % original.dim != 0 or cfg.target_rows >= chunk_rows) return error.InvalidFitConfig;
    const q = train.len / original.dim;
    const matrix_count = try checkedMul(q, chunk_rows);
    const batch_count = try checkedMul(@as(usize, 64), original.rows);
    const retained_count = try checkedAdd(try checkedAdd(matrix_count, q), batch_count);
    const retained_bytes = try checkedMul(retained_count, @sizeOf(f64));
    const nested_bytes = try solver.massSelectionWorkspaceBytes(chunk_rows, cfg.target_rows, q);
    const attend_bytes = if (cfg.backend) |backend| try backend.workspace_fn(backend.context, original.rows, 1, @min(q, 64), original.dim) else 0;
    if (try checkedAdd(retained_bytes, @max(nested_bytes, attend_bytes)) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    try cfg.control.check();
    const matrix = try allocator.alloc(f64, matrix_count);
    defer allocator.free(matrix);
    const rho = try allocator.alloc(f64, q);
    defer allocator.free(rho);
    const batch = try allocator.alloc(f64, batch_count);
    defer allocator.free(batch);
    var offset: usize = 0;
    while (offset < q) {
        try cfg.control.check();
        const take = @min(@as(usize, 64), q - offset);
        try solver.attendBatch(allocator, original, train[offset * original.dim ..][0 .. take * original.dim], batch[0 .. take * original.rows], null, null, cfg);
        for (0..take) |local_query| {
            const source = batch[local_query * original.rows + chunk_start ..][0..chunk_rows];
            const destination = matrix[(offset + local_query) * chunk_rows ..][0..chunk_rows];
            @memcpy(destination, source);
            var sum: f64 = 0;
            for (source) |value| {
                if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
                sum += value;
            }
            if (!std.math.isFinite(sum) or sum < 0) return error.NonFiniteAttention;
            rho[offset + local_query] = sum;
        }
        offset += take;
    }
    var nested = cfg;
    nested.max_workspace_bytes -= retained_bytes;
    return selectFeaturesRaw(allocator, matrix, rho, chunk_rows, nested);
}

test "cumulative runtime global denominator raw selection differs from local chunk normalization" {
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    // Query zero assigns only one percent of whole-head mass to this chunk;
    // local normalization nevertheless gives it the same selection weight.
    var global = try selectFeaturesRaw(std.testing.allocator, &.{ 0.0099, 0.0001, 0.45, 0.55 }, &.{ 0.01, 1 }, 2, cfg);
    defer global.deinit();
    try std.testing.expectEqualSlices(usize, &.{1}, global.indices);
    var local = try selectFeaturesRaw(std.testing.allocator, &.{ 0.99, 0.01, 0.45, 0.55 }, &.{ 1, 1 }, 2, cfg);
    defer local.deinit();
    try std.testing.expectEqualSlices(usize, &.{0}, local.indices);
}

test "production raw selector is bit-identical to diagnostic raw policy" {
    const diagnostic = @import("attention_match_global_selection_diagnostic.zig");
    const probabilities = [_]f64{ 0.12, 0.28, 0.60, 0.55, 0.35, 0.10, 0.20, 0.70, 0.10 };
    const rho = [_]f64{ 1, 1, 1 };
    const cfg: solver.Config = .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    var production = try selectFeaturesRaw(std.testing.allocator, &probabilities, &rho, 3, cfg);
    defer production.deinit();
    var reference = try diagnostic.selectFeaturesNamedPolicy(std.testing.allocator, &probabilities, &rho, 3, cfg, .raw);
    defer reference.deinit();
    try std.testing.expectEqualSlices(usize, reference.indices, production.indices);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(reference.weights), std.mem.sliceAsBytes(production.weights));
    try std.testing.expectEqual(reference.query_rows_consumed, production.query_rows_consumed);
}

test "cumulative runtime global denominator selection rejects zero progress and fails closed on cap cancellation and allocation" {
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    try std.testing.expectError(error.InsufficientSelectionProgress, selectFeaturesRaw(std.testing.allocator, &.{ 0, 0, 0, 0 }, &.{ 0, 0 }, 2, cfg));
    var capped = cfg;
    capped.max_workspace_bytes = 1;
    try std.testing.expectError(error.CompactionWorkspaceLimit, selectFeaturesRaw(std.testing.allocator, &.{ 0.1, 0.9, 0.9, 0.1 }, &.{ 1, 1 }, 2, capped));
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    var cancelled = cfg;
    cancelled.control.check_fn = Cancel.check;
    try std.testing.expectError(error.Cancelled, selectFeaturesRaw(std.testing.allocator, &.{ 0.1, 0.9, 0.9, 0.1 }, &.{ 1, 1 }, 2, cancelled));
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, selectFeaturesRaw(failing.allocator(), &.{ 0.1, 0.9, 0.9, 0.1 }, &.{ 1, 1 }, 2, cfg));
}

test "cumulative runtime global denominator streaming matches monolithic across batch tail and fails before callback" {
    const q: usize = 65;
    const train = try std.testing.allocator.alloc(f64, q);
    defer std.testing.allocator.free(train);
    for (train, 0..) |*value, i| value.* = @as(f64, @floatFromInt(i)) / 64;
    const head: solver.Head = .{ .keys = &.{ 0.2, -0.4, 0.7 }, .values = &.{ 1, 2, 3 }, .bias = &.{ 0.1, -0.2, 0.3 }, .rows = 3, .dim = 1, .scale = 0.75, .softcap = 1.5 };
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    var streamed = try selectStreaming(std.testing.allocator, head, 0, 2, train, cfg);
    defer streamed.deinit();
    const full = try std.testing.allocator.alloc(f64, q * 3);
    defer std.testing.allocator.free(full);
    try solver.attendBatch(std.testing.allocator, head, train, full, null, null, cfg);
    const matrix = try std.testing.allocator.alloc(f64, q * 2);
    defer std.testing.allocator.free(matrix);
    const rho = try std.testing.allocator.alloc(f64, q);
    defer std.testing.allocator.free(rho);
    for (0..q) |i| {
        matrix[i * 2] = full[i * 3];
        matrix[i * 2 + 1] = full[i * 3 + 1];
        rho[i] = matrix[i * 2] + matrix[i * 2 + 1];
    }
    var monolithic = try selectFeaturesRaw(std.testing.allocator, matrix, rho, 2, cfg);
    defer monolithic.deinit();
    try std.testing.expectEqualSlices(usize, monolithic.indices, streamed.indices);
    const Blocked = struct {
        var calls: usize = 0;
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 1 << 20;
        }
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, _: []f64, _: ?[]f64, _: ?[]f64, _: solver.Control) !void {
            calls += 1;
        }
    };
    Blocked.calls = 0;
    var capped = cfg;
    capped.backend = .{ .workspace_fn = Blocked.workspace, .attend_fn = Blocked.attend };
    try std.testing.expectError(error.CompactionWorkspaceLimit, selectStreaming(std.testing.allocator, head, 0, 2, train, capped));
    try std.testing.expectEqual(@as(usize, 0), Blocked.calls);
    const Between = struct {
        var armed: bool = false;
        var calls: usize = 0;
        var processed: usize = 0;
        fn check(_: ?*anyopaque) !void {
            if (armed) return error.Cancelled;
        }
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, h: solver.Head, qs: []const f64, p: []f64, _: ?[]f64, _: ?[]f64, _: solver.Control) !void {
            for (0..qs.len / h.dim) |i| _ = try solver.attend(h, qs[i * h.dim ..][0..h.dim], p[i * h.rows ..][0..h.rows], null);
            calls += 1;
            processed += qs.len / h.dim;
            armed = true;
        }
    };
    Between.armed = false;
    Between.calls = 0;
    Between.processed = 0;
    var cancelled = cfg;
    cancelled.control.check_fn = Between.check;
    cancelled.backend = .{ .workspace_fn = Between.workspace, .attend_fn = Between.attend };
    try std.testing.expectError(error.Cancelled, selectStreaming(std.testing.allocator, head, 0, 2, train, cancelled));
    try std.testing.expectEqual(@as(usize, 1), Between.calls);
    try std.testing.expectEqual(@as(usize, 64), Between.processed);
}

fn streamingAllocationFailure(allocator: std.mem.Allocator) !void {
    var train: [65]f64 = undefined;
    for (&train, 0..) |*value, i| value.* = @as(f64, @floatFromInt(i)) / 64;
    const head: solver.Head = .{ .keys = &.{ 0.2, -0.4, 0.7 }, .values = &.{ 1, 2, 3 }, .bias = &.{ 0.1, -0.2, 0.3 }, .rows = 3, .dim = 1, .scale = 0.75, .softcap = 1.5 };
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    var result = try selectStreaming(allocator, head, 0, 2, &train, cfg);
    result.deinit();
}

test "cumulative runtime global denominator streaming unwinds every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, streamingAllocationFailure, .{});
}

test "cumulative runtime global denominator selected rows refit without exact seeds end to end" {
    const global = @import("attention_match_global_residual.zig");
    const source: solver.Head = .{ .keys = &.{ 0, 0, 0 }, .values = &.{ 2, 4, 8 }, .bias = &.{ 0, 0, 0 }, .rows = 3, .dim = 1, .scale = 1 };
    const train = [_]f64{ 0, 0.5, 1 };
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    var selected = try selectStreaming(std.testing.allocator, source, 0, 2, &train, cfg);
    defer selected.deinit();
    try std.testing.expectEqualSlices(usize, &.{0}, selected.indices);
    var fitted = try global.fitOriginalContextChunk(std.testing.allocator, source, 0, 2, &.{}, selected.indices, &train, cfg);
    defer fitted.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 2), fitted.weights[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 3), fitted.values[0], 1e-6);
    try std.testing.expectApproxEqAbs(@log(@as(f64, 2)), fitted.bias[0], 1e-6);
    const keys = [_]f64{ source.keys[selected.indices[0]], source.keys[2] };
    const values = [_]f64{ fitted.values[0], source.values[2] };
    const biases = [_]f64{ fitted.bias[0], source.bias.?[2] };
    const compact: solver.Head = .{ .keys = &keys, .values = &values, .bias = &biases, .rows = 2, .dim = 1, .scale = 1 };
    const metrics = try solver.validationMetrics(std.testing.allocator, source, compact, &.{0.7}, .{ .target_rows = 2 });
    try std.testing.expect(metrics.max_relative_mass_error < 1e-6);
    try std.testing.expect(metrics.relative_output_rmse < 1e-6);
}
