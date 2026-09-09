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

//! Filesystem-aware path isolation for fine-tuning inputs and outputs.
//!
//! `std.fs.path.resolve` removes lexical `.` and `..` components but does not
//! resolve symbolic links. Mutable outputs commonly do not exist at preflight,
//! so canonicalize their deepest existing ancestor and retain the unresolved
//! suffix. Comparing these requested canonical paths catches aliases through
//! symlinked parents before any artifact is created. Preserve traversal order:
//! resolving `symlink/..` lexically changes its filesystem destination.

const std = @import("std");
const builtin = @import("builtin");

/// Canonicalize every existing ancestor while retaining a not-yet-created
/// suffix. Missing suffixes containing `.` or `..`, and dangling symbolic
/// links, are rejected because their destination cannot be established.
/// The returned path is owned by `allocator`.
pub fn resolveRequestedPath(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) ![]u8 {
    if (path.len == 0) return error.InvalidRequestedPath;

    const absolute = if (std.fs.path.isAbsolute(path))
        try allocator.dupe(u8, path)
    else blk: {
        const canonical_cwd = try std.Io.Dir.cwd().realPathFileAlloc(io, ".", allocator);
        defer allocator.free(canonical_cwd);
        break :blk try std.fs.path.join(allocator, &.{ canonical_cwd, path });
    };
    defer allocator.free(absolute);

    var existing_candidate: []const u8 = absolute;
    while (true) {
        const canonical = std.Io.Dir.realPathFileAbsoluteAlloc(io, existing_candidate, allocator) catch |err| switch (err) {
            error.FileNotFound => {
                // Only ordinary missing components may be retained. A parent
                // traversal must be resolved by the filesystem, and a dangling
                // symlink must not be mistaken for a new output name.
                const component = std.fs.path.basename(existing_candidate);
                if (std.mem.eql(u8, component, ".") or std.mem.eql(u8, component, "..")) {
                    return error.InvalidRequestedPath;
                }
                // A trailing separator forces directory traversal even with
                // no-follow stat, so remove it when checking the final entry.
                var entry_path = existing_candidate;
                while (entry_path.len > 1 and std.fs.path.isSep(entry_path[entry_path.len - 1])) {
                    entry_path = entry_path[0 .. entry_path.len - 1];
                }
                const stat = std.Io.Dir.cwd().statFile(io, entry_path, .{ .follow_symlinks = false }) catch |stat_err| switch (stat_err) {
                    error.FileNotFound => null,
                    else => return stat_err,
                };
                if (stat) |existing| {
                    if (existing.kind == .sym_link) return error.InvalidRequestedPath;
                }
                const parent = std.fs.path.dirname(existing_candidate) orelse return err;
                if (std.mem.eql(u8, parent, existing_candidate)) return err;
                existing_candidate = parent;
                continue;
            },
            else => return err,
        };
        defer allocator.free(canonical);

        if (existing_candidate.len == absolute.len) {
            return allocator.dupe(u8, canonical);
        }
        var unresolved_suffix = absolute[existing_candidate.len..];
        while (unresolved_suffix.len > 0 and std.fs.path.isSep(unresolved_suffix[0])) {
            unresolved_suffix = unresolved_suffix[1..];
        }
        // The retained suffix now contains only ordinary missing components.
        // Lexical normalization is safe here and makes repeated separators
        // compare equal when two outputs name the same not-yet-created path.
        return std.fs.path.resolve(allocator, &.{ canonical, unresolved_suffix });
    }
}

pub fn sameOrWithin(parent: []const u8, path: []const u8) bool {
    if (std.mem.eql(u8, parent, path)) return true;
    if (parent.len == 0 or path.len <= parent.len or !std.mem.startsWith(u8, path, parent)) return false;
    if (std.fs.path.isSep(parent[parent.len - 1])) return true;
    return std.fs.path.isSep(path[parent.len]);
}

pub fn pathsOverlap(a: []const u8, b: []const u8) bool {
    return sameOrWithin(a, b) or sameOrWithin(b, a);
}

test "requested paths canonicalize symlinked ancestors with missing suffixes" {
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.createDir(io, "immutable-model", .default_dir);
    try tmp.dir.symLink(io, "immutable-model", "output-alias", .{});

    const root = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    const model = try std.fs.path.join(allocator, &.{ root, "immutable-model" });
    defer allocator.free(model);
    const alias_output = try std.fs.path.join(allocator, &.{ root, "output-alias", "new", "report.json" });
    defer allocator.free(alias_output);

    const canonical_model = try resolveRequestedPath(allocator, io, model);
    defer allocator.free(canonical_model);
    const canonical_output = try resolveRequestedPath(allocator, io, alias_output);
    defer allocator.free(canonical_output);

    const expected_output = try std.fs.path.join(allocator, &.{ canonical_model, "new", "report.json" });
    defer allocator.free(expected_output);
    try std.testing.expectEqualStrings(expected_output, canonical_output);
    try std.testing.expect(pathsOverlap(canonical_model, canonical_output));
}

test "path overlap observes component boundaries" {
    try std.testing.expect(pathsOverlap("/models/gemma4", "/models/gemma4/reports/run.json"));
    try std.testing.expect(!pathsOverlap("/models/gemma4", "/models/gemma4-backup"));
}

test "gemma4 requested paths resolve symlinks before parent traversal" {
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "model/subdir");
    try tmp.dir.symLink(io, "model/subdir", "alias", .{});
    try tmp.dir.writeFile(io, .{ .sub_path = "model/existing", .data = "BASE" });

    const absolute_root = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(absolute_root);
    const relative_root = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer allocator.free(relative_root);
    for ([_][]const u8{ absolute_root, relative_root }) |root| {
        for ([_][2][]const u8{
            .{ "existing", "existing" },
            .{ "new/checkpoint", "new/checkpoint" },
            .{ "new//checkpoint/", "new/checkpoint" },
        }) |paths| {
            const requested = try std.fs.path.join(allocator, &.{ root, "alias", "..", paths[0] });
            defer allocator.free(requested);
            const resolved = try resolveRequestedPath(allocator, io, requested);
            defer allocator.free(resolved);
            const expected = try std.fs.path.join(allocator, &.{ absolute_root, "model", paths[1] });
            defer allocator.free(expected);
            try std.testing.expectEqualStrings(expected, resolved);
        }
    }
}

test "gemma4 requested paths reject unresolved traversal and dangling symlinks" {
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.symLink(io, "missing-target", "dangling", .{});
    const root = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    for ([_][]const u8{
        "missing/../checkpoint",
        "missing/./checkpoint",
        "dangling",
        "dangling/",
        "dangling//",
        "dangling/checkpoint",
        "dangling/../checkpoint",
    }) |suffix| {
        const requested = try std.fs.path.join(allocator, &.{ root, suffix });
        defer allocator.free(requested);
        try std.testing.expectError(error.InvalidRequestedPath, resolveRequestedPath(allocator, io, requested));
    }
}

test "requested relative paths become canonical absolute paths" {
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest;

    const resolved = try resolveRequestedPath(
        std.testing.allocator,
        std.testing.io,
        ".zig-cache/path-isolation/not-created.json",
    );
    defer std.testing.allocator.free(resolved);
    try std.testing.expect(std.fs.path.isAbsolute(resolved));
    try std.testing.expect(std.mem.endsWith(u8, resolved, "/.zig-cache/path-isolation/not-created.json"));
}
