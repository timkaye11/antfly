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
//! symlinked parents before any artifact is created.

const std = @import("std");
const builtin = @import("builtin");

/// Canonicalize every existing ancestor while retaining a not-yet-created
/// suffix. The returned path is owned by `allocator`.
pub fn resolveRequestedPath(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) ![]u8 {
    if (path.len == 0) return error.InvalidRequestedPath;

    const absolute = if (std.fs.path.isAbsolute(path))
        try std.fs.path.resolve(allocator, &.{path})
    else blk: {
        const canonical_cwd = try std.Io.Dir.cwd().realPathFileAlloc(io, ".", allocator);
        defer allocator.free(canonical_cwd);
        break :blk try std.fs.path.resolve(allocator, &.{ canonical_cwd, path });
    };
    defer allocator.free(absolute);

    var existing_candidate: []const u8 = absolute;
    while (true) {
        const canonical = std.Io.Dir.realPathFileAbsoluteAlloc(io, existing_candidate, allocator) catch |err| switch (err) {
            error.FileNotFound => {
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
        const unresolved_suffix = std.mem.trimStart(
            u8,
            absolute[existing_candidate.len..],
            &.{ '/', '\\' },
        );
        return std.fs.path.join(allocator, &.{ canonical, unresolved_suffix });
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
