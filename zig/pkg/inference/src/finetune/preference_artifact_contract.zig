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

//! Canonical input/output isolation for preference-training artifacts.
//!
//! The recipe layer enumerates its domain-specific paths. This module owns the
//! filesystem policy: resolve symlinked existing ancestors, retain unresolved
//! output suffixes, and reject unsafe output/output or input/output
//! containment. A job's artifact root is a namespace container, so it may
//! contain that same job's recipe and dataset inputs as strict descendants.

const std = @import("std");
const compat = @import("../io/compat.zig");

pub const Role = enum { input, output };

pub const Entry = struct {
    recipe_index: ?usize,
    role: Role,
    artifact_root: bool = false,
    label: []const u8,
    path: []u8,
};

pub fn deinitEntries(allocator: std.mem.Allocator, entries: *std.ArrayList(Entry)) void {
    for (entries.items) |entry| allocator.free(entry.path);
    entries.deinit(allocator);
}

pub fn append(
    allocator: std.mem.Allocator,
    io: std.Io,
    entries: *std.ArrayList(Entry),
    recipe_index: ?usize,
    role: Role,
    artifact_root: bool,
    label: []const u8,
    maybe_path: ?[]const u8,
) !void {
    const path = maybe_path orelse return;
    const canonical = try resolveOutputPath(allocator, io, path);
    errdefer allocator.free(canonical);
    try entries.append(allocator, .{
        .recipe_index = recipe_index,
        .role = role,
        .artifact_root = artifact_root,
        .label = label,
        .path = canonical,
    });
}

/// Canonicalize every existing ancestor while retaining a not-yet-created
/// suffix. This catches aliases through symlinked output parents without
/// requiring the output itself to exist.
pub fn resolveOutputPath(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    const absolute = try std.fs.path.resolve(allocator, &.{path});
    defer allocator.free(absolute);

    var existing_candidate: []const u8 = absolute;
    while (true) {
        const canonical = compat.cwd().realPathFileAlloc(io, existing_candidate, allocator) catch |err| switch (err) {
            error.FileNotFound => {
                existing_candidate = std.fs.path.dirname(existing_candidate) orelse return err;
                continue;
            },
            else => return err,
        };
        defer allocator.free(canonical);
        if (existing_candidate.len == absolute.len) return allocator.dupe(u8, canonical);
        const unresolved_suffix = std.mem.trimStart(u8, absolute[existing_candidate.len..], &.{ '/', '\\' });
        return std.fs.path.join(allocator, &.{ canonical, unresolved_suffix });
    }
}

pub fn validate(entries: []const Entry) !void {
    for (entries, 0..) |entry, idx| {
        if (entry.role != .output) continue;
        for (entries[0..idx]) |prior| {
            if (!pathsOverlap(prior.path, entry.path)) continue;
            if (allowsSameJobArtifactRootContainment(prior, entry)) continue;
            if (prior.role == .input) return error.PreferenceInputOutputConflict;
            return error.PreferenceArtifactConflict;
        }
        for (entries[idx + 1 ..]) |later| {
            if (later.role == .input and
                pathsOverlap(entry.path, later.path) and
                !allowsSameJobArtifactRootContainment(entry, later))
            {
                return error.PreferenceInputOutputConflict;
            }
        }
    }
}

fn allowsSameJobArtifactRootContainment(a: Entry, b: Entry) bool {
    if (a.recipe_index == null or b.recipe_index == null or a.recipe_index.? != b.recipe_index.?) {
        return false;
    }
    const root = if (a.role == .output and a.artifact_root)
        a
    else if (b.role == .output and b.artifact_root)
        b
    else
        return false;
    const other = if (a.role == .output and a.artifact_root) b else a;
    return !std.mem.eql(u8, root.path, other.path) and pathIsSameOrWithin(root.path, other.path);
}

fn pathsOverlap(a: []const u8, b: []const u8) bool {
    return pathIsSameOrWithin(a, b) or pathIsSameOrWithin(b, a);
}

fn pathIsSameOrWithin(parent: []const u8, path: []const u8) bool {
    if (std.mem.eql(u8, parent, path)) return true;
    if (parent.len == 0 or path.len <= parent.len or !std.mem.startsWith(u8, path, parent)) return false;
    if (std.fs.path.isSep(parent[parent.len - 1])) return true;
    return std.fs.path.isSep(path[parent.len]);
}

test "canonical contract permits same-job root containment and rejects unsafe overlaps" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    {
        var entries: std.ArrayList(Entry) = .empty;
        defer deinitEntries(allocator, &entries);
        try append(allocator, io, &entries, 0, .output, true, "root", "/tmp/preference-contract/job");
        try append(allocator, io, &entries, 0, .output, false, "report", "/tmp/preference-contract/job/report.json");
        try append(allocator, io, &entries, 0, .input, false, "recipe", "/tmp/preference-contract/job/recipe.json");
        try append(allocator, io, &entries, 0, .input, false, "dataset", "/tmp/preference-contract/job/data.jsonl");
        try validate(entries.items);
    }
    {
        var entries: std.ArrayList(Entry) = .empty;
        defer deinitEntries(allocator, &entries);
        try append(allocator, io, &entries, 0, .output, true, "root", "/tmp/preference-contract/job");
        try append(allocator, io, &entries, 0, .output, false, "report", "/tmp/preference-contract/job");
        try std.testing.expectError(error.PreferenceArtifactConflict, validate(entries.items));
    }
    {
        var entries: std.ArrayList(Entry) = .empty;
        defer deinitEntries(allocator, &entries);
        try append(allocator, io, &entries, 0, .output, true, "root", "/tmp/preference-contract/job");
        try append(allocator, io, &entries, 1, .input, false, "dataset", "/tmp/preference-contract/job/data.jsonl");
        try std.testing.expectError(error.PreferenceInputOutputConflict, validate(entries.items));
    }
}
