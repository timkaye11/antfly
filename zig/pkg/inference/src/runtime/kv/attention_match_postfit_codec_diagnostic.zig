// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! No-credit post-fit codec diagnosis. The caller authenticates the original
//! whole-fit input, fitted F16 checkpoint and TRAIN/heldout query bindings.
//! This module never fits geometry and never publishes the mutated clone.
const std = @import("std");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const solver = @import("attention_match.zig");

pub const Encoding = enum { fp8, rotated4 };

/// Reconstruct the candidate-row envelope used by the initial adaptive codec
/// attempt. The authenticated source must be the initial exact contiguous
/// layout. Consecutive candidate rows may share one coverage end: they form a
/// fitted source group whose source extent is `(prior_end, coverage_end]`.
/// Shrinkage is proven only when that complete candidate group has fewer rows
/// than its source group. The returned envelope includes every candidate group
/// from the first shrink through the last shrink, including unchanged islands.
pub fn fittedEnvelope(original: checkpoint.Head, fitted: checkpoint.Head) !?[2]usize {
    const original_dim: usize = original.dim;
    const fitted_dim: usize = fitted.dim;
    const original_elements = std.math.mul(usize, original.positions.len, original_dim) catch return error.InvalidAttentionShape;
    const fitted_elements = std.math.mul(usize, fitted.positions.len, fitted_dim) catch return error.InvalidAttentionShape;
    if (original.dim == 0 or original.dim != fitted.dim or fitted.coverage_ends.len == 0 or
        original.positions.len != original.coverage_ends.len or original.bias.len != original.positions.len or
        original.keys.len != original_elements or original.values.len != original.keys.len or
        fitted.positions.len != fitted.coverage_ends.len or fitted.bias.len != fitted.positions.len or
        fitted.keys.len != fitted_elements or fitted.values.len != fitted.keys.len)
        return error.InvalidAttentionShape;
    for (original.positions, original.coverage_ends, original.bias, 0..) |position, coverage_end, bias, i| {
        if (position != i or coverage_end != i + 1 or bias != 0) return error.InvalidAttentionShape;
    }
    var prior_source_end: usize = 0;
    var candidate_i: usize = 0;
    var first: ?usize = null;
    var end: usize = 0;
    while (candidate_i < fitted.coverage_ends.len) {
        const group_first = candidate_i;
        const source_end = fitted.coverage_ends[candidate_i];
        if (source_end <= prior_source_end or source_end > original.positions.len) return error.InvalidAttentionShape;
        while (candidate_i < fitted.coverage_ends.len and fitted.coverage_ends[candidate_i] == source_end) : (candidate_i += 1) {
            const position = fitted.positions[candidate_i];
            if (position < prior_source_end or position >= source_end or
                (candidate_i > group_first and position <= fitted.positions[candidate_i - 1]))
                return error.InvalidAttentionShape;
        }
        const candidate_rows = candidate_i - group_first;
        const source_rows = source_end - prior_source_end;
        if (candidate_rows > source_rows) return error.InvalidAttentionShape;
        if (candidate_rows < source_rows) {
            if (first == null) first = group_first;
            end = candidate_i;
        }
        prior_source_end = source_end;
    }
    return if (first) |begin| .{ begin, end } else null;
}

fn converted(a: std.mem.Allocator, values: []const f32) ![]f64 {
    const out = try a.alloc(f64, values.len);
    for (out, values) |*dst, src| dst.* = src;
    return out;
}

fn rejectedBackendWorkspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
    return 0;
}

/// Mutates only an owned clone of `fitted`. The exact scalar FP64 validation
/// limits remain 0.05/0.05 through the supplied config.
pub fn diagnoseHead(a: std.mem.Allocator, original: checkpoint.Head, fitted: checkpoint.Head, train: []const f64, heldout: []const f64, scale: f64, softcap: f64, encoding: Encoding, rotated4_risk_retention: engine.Rotated4RiskRetention, config: solver.Config) !engine.CodecDiagnostic {
    if (config.backend != null or config.max_relative_mass_error != 0.05 or config.max_relative_output_rmse != 0.05)
        return error.InvalidFitConfig;
    const envelope = (try fittedEnvelope(original, fitted)) orelse return .{};
    var owned = try checkpoint.cloneHead(a, fitted);
    defer {
        a.free(@constCast(owned.positions));
        a.free(@constCast(owned.coverage_ends));
        a.free(@constCast(owned.keys));
        a.free(@constCast(owned.values));
        a.free(@constCast(owned.bias));
        owned.compressed.deinit(a);
    }
    const original_keys = try converted(a, original.keys);
    defer a.free(original_keys);
    const original_values = try converted(a, original.values);
    defer a.free(original_values);
    const original_bias = try converted(a, original.bias);
    defer a.free(original_bias);
    const fitted_keys = try converted(a, fitted.keys);
    defer a.free(fitted_keys);
    const fitted_values = try converted(a, fitted.values);
    defer a.free(fitted_values);
    const fitted_bias = try converted(a, fitted.bias);
    defer a.free(fitted_bias);
    const source: solver.Head = .{ .keys = original_keys, .values = original_values, .bias = original_bias, .rows = original.positions.len, .dim = @intCast(original.dim), .scale = scale, .softcap = softcap };
    const candidate: solver.Head = .{ .keys = fitted_keys, .values = fitted_values, .bias = fitted_bias, .rows = fitted.positions.len, .dim = @intCast(fitted.dim), .scale = scale, .softcap = softcap };
    var diagnostic: engine.CodecDiagnostic = .{};
    const options: engine.Options = .{
        .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
        .storage_precision = .f16,
        .validation = .whole_head,
        .independent_fp64_publication = true,
        .fp8_checkpoint = encoding == .fp8,
        .rotated4_checkpoint = encoding == .rotated4,
        .rotated4_risk_retention = rotated4_risk_retention,
        .codec_diagnostic = &diagnostic,
        .fit = config,
    };
    try engine.authorizeAdaptiveCodec(a, source, candidate, &owned, envelope[0], envelope[1], train, heldout, options);
    return diagnostic;
}

test "attention_match postfit codec envelope reconstructs fitted islands and excludes suffix" {
    const original: checkpoint.Head = .{ .dim = 1, .positions = &.{ 0, 1, 2, 3, 4, 5 }, .coverage_ends = &.{ 1, 2, 3, 4, 5, 6 }, .keys = &.{ 0, 0, 0, 0, 0, 0 }, .values = &.{ 0, 0, 0, 0, 0, 0 }, .bias = &.{ 0, 0, 0, 0, 0, 0 } };
    const fitted: checkpoint.Head = .{ .dim = 1, .positions = &.{ 0, 2, 3, 5 }, .coverage_ends = &.{ 1, 3, 4, 6 }, .keys = &.{ 0, 0, 0, 0 }, .values = &.{ 0, 0, 0, 0 }, .bias = &.{ 0, 0, 0, 0 } };
    try std.testing.expectEqual([2]usize{ 1, 4 }, (try fittedEnvelope(original, fitted)).?);
}

test "context attention_match postfit codec groups repeated coverage ends and requires proven shrinkage" {
    const original: checkpoint.Head = .{ .dim = 1, .positions = &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, .coverage_ends = &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, .keys = &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, .values = &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, .bias = &.{ 0, 0, 0, 0, 0, 0, 0, 0 } };
    // [0,3) and [6,8) are unchanged complete groups. [3,6) shrinks
    // from three authenticated source rows to two fitted support rows.
    const fitted: checkpoint.Head = .{ .dim = 1, .positions = &.{ 0, 1, 2, 3, 5, 6, 7 }, .coverage_ends = &.{ 3, 3, 3, 6, 6, 8, 8 }, .keys = &.{ 0, 0, 0, 0, 0, 0, 0 }, .values = &.{ 0, 0, 0, 0, 0, 0, 0 }, .bias = &.{ 0, 0, 0, 0, 0, 0, 0 } };
    try std.testing.expectEqual([2]usize{ 3, 5 }, (try fittedEnvelope(original, fitted)).?);

    const unchanged: checkpoint.Head = .{ .dim = 1, .positions = &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, .coverage_ends = &.{ 3, 3, 3, 6, 6, 6, 8, 8 }, .keys = &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, .values = &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, .bias = &.{ 0, 0, 0, 0, 0, 0, 0, 0 } };
    try std.testing.expectEqual(@as(?[2]usize, null), try fittedEnvelope(original, unchanged));

    const gapped_original: checkpoint.Head = .{ .dim = 1, .positions = &.{ 0, 2 }, .coverage_ends = &.{ 1, 3 }, .keys = &.{ 0, 0 }, .values = &.{ 0, 0 }, .bias = &.{ 0, 0 } };
    try std.testing.expectError(error.InvalidAttentionShape, fittedEnvelope(gapped_original, fitted));
}

test "context attention_match postfit codec diagnostic rejects broadened or backend validation" {
    const empty: checkpoint.Head = .{};
    var config: solver.Config = .{
        .target_rows = 1,
        .max_relative_mass_error = 1.0,
        .max_relative_output_rmse = 0.05,
    };
    try std.testing.expectError(error.InvalidFitConfig, diagnoseHead(std.testing.allocator, empty, empty, &.{}, &.{}, 1, 0, .rotated4, .quarter_v1, config));
    config.max_relative_mass_error = 0.05;
    config.max_relative_output_rmse = 1.0;
    try std.testing.expectError(error.InvalidFitConfig, diagnoseHead(std.testing.allocator, empty, empty, &.{}, &.{}, 1, 0, .fp8, .quarter_v1, config));
    config.max_relative_output_rmse = 0.05;
    config.backend = .{ .workspace_fn = rejectedBackendWorkspace };
    try std.testing.expectError(error.InvalidFitConfig, diagnoseHead(std.testing.allocator, empty, empty, &.{}, &.{}, 1, 0, .fp8, .quarter_v1, config));
}
