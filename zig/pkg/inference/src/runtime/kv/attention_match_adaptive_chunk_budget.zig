// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pure training-seed-aware chunk target allocation. This module does not
//! select seeds, inspect heldout queries, fit values, or authorize quality.
const std = @import("std");

/// Redistributes the ordinary `ceil(source_rows / divisor)` targets without
/// changing their sum. Every chunk retains every mandatory training seed plus
/// at least one fitted residual row, and no chunk becomes an identity copy.
///
/// All validation completes before `out_targets` is written. Infeasible seed
/// pressure is retriable at a smaller divisor; seeds are never truncated.
pub fn adaptiveChunkTargets(
    source_rows: []const usize,
    mandatory_seed_counts: []const usize,
    divisor: usize,
    out_targets: []usize,
) !usize {
    if (divisor == 0 or source_rows.len == 0 or
        mandatory_seed_counts.len != source_rows.len or
        out_targets.len != source_rows.len)
        return error.InvalidFitConfig;

    var uniform_total: usize = 0;
    var required_total: usize = 0;
    var upper_total: usize = 0;
    for (source_rows, mandatory_seed_counts) |rows, seeds| {
        if (rows == 0 or seeds > rows) return error.InvalidFitConfig;
        const uniform = @divTrunc(rows, divisor) + @intFromBool(rows % divisor != 0);
        const required = try std.math.add(usize, seeds, 1);
        // `required == rows` would leave no actual residual compaction. The
        // exact-plus-residual fitter requires residual_target < residual_rows.
        if (required >= rows) return error.CompactionFitRejected;
        uniform_total = try std.math.add(usize, uniform_total, uniform);
        required_total = try std.math.add(usize, required_total, required);
        upper_total = try std.math.add(usize, upper_total, rows - 1);
    }
    if (required_total > uniform_total) return error.CompactionFitRejected;
    if (uniform_total > upper_total) return error.CompactionFitRejected;

    // No failure is possible below this point, so rejected inputs leave the
    // caller's output untouched.
    var borrowed: usize = 0;
    for (source_rows, mandatory_seed_counts, out_targets) |rows, seeds, *target| {
        const uniform = @divTrunc(rows, divisor) + @intFromBool(rows % divisor != 0);
        const required = seeds + 1;
        target.* = @max(uniform, required);
        borrowed += target.* - uniform;
    }

    // Return borrowed rows by repeatedly choosing the donor with the largest
    // remaining slack. Lower chunk index is the stable tie break.
    while (borrowed != 0) : (borrowed -= 1) {
        var donor: ?usize = null;
        var largest_slack: usize = 0;
        for (out_targets, mandatory_seed_counts, 0..) |target, seeds, chunk| {
            const required = seeds + 1;
            const slack = target - required;
            if (slack > largest_slack) {
                donor = chunk;
                largest_slack = slack;
            }
        }
        std.debug.assert(donor != null and largest_slack != 0);
        out_targets[donor.?] -= 1;
    }

    var actual_total: usize = 0;
    for (source_rows, mandatory_seed_counts, out_targets) |rows, seeds, target| {
        std.debug.assert(target >= seeds + 1 and target < rows);
        actual_total += target;
    }
    std.debug.assert(actual_total == uniform_total);
    return uniform_total;
}

test "attention match adaptive chunk targets preserve total and deterministic donor ties" {
    var unchanged: [2]usize = undefined;
    try std.testing.expectEqual(@as(usize, 8), try adaptiveChunkTargets(&.{ 8, 8 }, &.{ 1, 1 }, 2, &unchanged));
    try std.testing.expectEqualSlices(usize, &.{ 4, 4 }, &unchanged);

    var borrowed: [3]usize = undefined;
    try std.testing.expectEqual(@as(usize, 12), try adaptiveChunkTargets(&.{ 8, 8, 8 }, &.{ 5, 1, 1 }, 2, &borrowed));
    // Chunk zero needs six rows. Equal donor slack first selects lower chunk
    // one, then chunk two has the larger remaining slack.
    try std.testing.expectEqualSlices(usize, &.{ 6, 3, 3 }, &borrowed);
    var repeated: [3]usize = undefined;
    _ = try adaptiveChunkTargets(&.{ 8, 8, 8 }, &.{ 5, 1, 1 }, 2, &repeated);
    try std.testing.expectEqualSlices(usize, &borrowed, &repeated);
}

test "attention match adaptive chunk targets use ceiling for a partial final chunk" {
    var targets: [2]usize = undefined;
    try std.testing.expectEqual(@as(usize, 7), try adaptiveChunkTargets(&.{ 8, 5 }, &.{ 1, 1 }, 2, &targets));
    try std.testing.expectEqualSlices(usize, &.{ 4, 3 }, &targets);
}

test "attention match adaptive chunk targets reject pressure without partial output" {
    var targets = [_]usize{ 0xaaaa, 0xbbbb };
    try std.testing.expectError(error.CompactionFitRejected, adaptiveChunkTargets(&.{ 8, 8 }, &.{ 6, 6 }, 2, &targets));
    try std.testing.expectEqualSlices(usize, &.{ 0xaaaa, 0xbbbb }, &targets);
    try std.testing.expectError(error.CompactionFitRejected, adaptiveChunkTargets(&.{8}, &.{7}, 2, targets[0..1]));
    try std.testing.expectEqualSlices(usize, &.{ 0xaaaa, 0xbbbb }, &targets);
}

test "attention match adaptive chunk targets reject invalid shapes and divisor" {
    var targets: [2]usize = undefined;
    try std.testing.expectError(error.InvalidFitConfig, adaptiveChunkTargets(&.{ 8, 8 }, &.{ 1, 1 }, 0, &targets));
    try std.testing.expectError(error.InvalidFitConfig, adaptiveChunkTargets(&.{8}, &.{ 1, 1 }, 2, targets[0..1]));
    try std.testing.expectError(error.InvalidFitConfig, adaptiveChunkTargets(&.{0}, &.{0}, 2, targets[0..1]));
    try std.testing.expectError(error.InvalidFitConfig, adaptiveChunkTargets(&.{8}, &.{9}, 2, targets[0..1]));
}
