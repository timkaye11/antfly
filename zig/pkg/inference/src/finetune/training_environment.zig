// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Stable process-environment identity for checkpointed Gemma training.
//! Capture once at run admission; environment ordering is not semantic.
const std = @import("std");
const policy = @import("gemma4_preference_environment.zig");

pub const Snapshot = struct {
    allocator: std.mem.Allocator,
    assignments: [][]const u8,
    sha256: [32]u8,

    pub fn deinit(self: *Snapshot) void {
        for (self.assignments) |entry| self.allocator.free(entry);
        self.allocator.free(self.assignments);
    }
};

pub fn capture(allocator: std.mem.Allocator) !Snapshot {
    if (comptime @import("builtin").os.tag == .windows or !@import("builtin").link_libc)
        return error.TrainingEnvironmentCaptureUnsupported;
    var entries: std.ArrayList([]const u8) = .empty;
    defer entries.deinit(allocator);
    var index: usize = 0;
    while (std.c.environ[index]) |entry| : (index += 1) {
        try entries.append(allocator, std.mem.span(entry));
    }
    return fromAssignments(allocator, entries.items);
}

fn fromAssignments(allocator: std.mem.Allocator, entries: []const []const u8) !Snapshot {
    var selected: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (selected.items) |entry| allocator.free(entry);
        selected.deinit(allocator);
    }
    for (entries) |entry| {
        const separator = std.mem.indexOfScalar(u8, entry, '=') orelse continue;
        if (!policy.nameInScope(entry[0..separator])) continue;
        const copy = try allocator.dupe(u8, entry);
        errdefer allocator.free(copy);
        try selected.append(allocator, copy);
    }
    std.mem.sort([]const u8, selected.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lessThan);
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update("antfly-gemma-training-environment/v1\x00");
    // Admission-policy edits invalidate old identities even with no override.
    hasher.update(policy.source);
    for (selected.items) |entry| {
        hasher.update("\x00");
        hasher.update(entry);
    }
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    return .{ .allocator = allocator, .assignments = try selected.toOwnedSlice(allocator), .sha256 = digest };
}

test "gemma4 environment checkpoint identity binds numerics regardless of environment order" {
    const allocator = std.testing.allocator;
    var first = try fromAssignments(allocator, &.{ "TERMITE_GEMMA4_SPARSE_LOSS_CHUNK_ROWS=128", "TERMITE_METAL_DISABLE_GEMMA_GQA_ATTENTION_FUSION=1", "PATH=/a" });
    defer first.deinit();
    var reordered = try fromAssignments(allocator, &.{ "PATH=/b", "TERMITE_METAL_DISABLE_GEMMA_GQA_ATTENTION_FUSION=1", "TERMITE_GEMMA4_SPARSE_LOSS_CHUNK_ROWS=128" });
    defer reordered.deinit();
    try std.testing.expectEqualSlices(u8, &first.sha256, &reordered.sha256);
    var changed = try fromAssignments(allocator, &.{ "TERMITE_GEMMA4_SPARSE_LOSS_CHUNK_ROWS=256", "TERMITE_METAL_DISABLE_GEMMA_GQA_ATTENTION_FUSION=1" });
    defer changed.deinit();
    try std.testing.expect(!std.mem.eql(u8, &first.sha256, &changed.sha256));
    var absent = try fromAssignments(allocator, &.{"TERMITE_GEMMA4_SPARSE_LOSS_CHUNK_ROWS=128"});
    defer absent.deinit();
    try std.testing.expect(!std.mem.eql(u8, &first.sha256, &absent.sha256));
}
