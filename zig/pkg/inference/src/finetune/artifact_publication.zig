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

//! Crash-safe directory publication for fine-tuning artifacts.
//!
//! Callers write a complete artifact into `staging_dir`, then publish it with
//! a no-replace rename. A failed writer or late destination collision leaves
//! an existing artifact untouched and removes the owned staging directory.

const std = @import("std");
const builtin = @import("builtin");

var publication_nonce: std.atomic.Value(u64) = .init(0);

/// Publish one immutable file through a synced sibling temporary. The final
/// path is never truncated or replaced, including when another writer wins a
/// late race.
pub fn writeFileImmutable(
    allocator: std.mem.Allocator,
    io: std.Io,
    final_path: []const u8,
    data: []const u8,
) !void {
    if (final_path.len == 0 or std.mem.eql(u8, final_path, ".") or std.mem.eql(u8, final_path, "/")) {
        return error.InvalidArtifactOutputPath;
    }
    if (try pathExists(io, final_path)) return error.Gemma4RunOutputAlreadyExists;
    const parent = std.fs.path.dirname(final_path) orelse ".";
    try std.Io.Dir.cwd().createDirPath(io, parent);

    const nonce = publication_nonce.fetchAdd(1, .monotonic);
    const temporary_leaf = try std.fmt.allocPrint(
        allocator,
        ".{s}.finetune-staging-{d}-{d}",
        .{ std.fs.path.basename(final_path), std.posix.system.getpid(), nonce },
    );
    defer allocator.free(temporary_leaf);
    const temporary_path = try std.fs.path.join(allocator, &.{ parent, temporary_leaf });
    defer allocator.free(temporary_path);
    errdefer std.Io.Dir.cwd().deleteFile(io, temporary_path) catch {};

    var file = try std.Io.Dir.cwd().createFile(io, temporary_path, .{ .truncate = false, .exclusive = true });
    var open = true;
    errdefer if (open) file.close(io);
    try file.writeStreamingAll(io, data);
    try file.sync(io);
    file.close(io);
    open = false;

    try renameNoReplace(allocator, io, temporary_path, final_path);
    try syncParentDirectory(io, final_path);
}

/// Replace a mutable state file through a fully written and synced sibling
/// temporary. Readers observe either the previous complete generation or the
/// new complete generation, never a truncated JSON/checkpoint sidecar.
pub fn writeFileAtomicReplace(
    allocator: std.mem.Allocator,
    io: std.Io,
    final_path: []const u8,
    data: []const u8,
) !void {
    if (final_path.len == 0 or std.mem.eql(u8, final_path, ".") or std.mem.eql(u8, final_path, "/")) {
        return error.InvalidArtifactOutputPath;
    }
    const parent = std.fs.path.dirname(final_path) orelse ".";
    try std.Io.Dir.cwd().createDirPath(io, parent);

    const nonce = publication_nonce.fetchAdd(1, .monotonic);
    const temporary_leaf = try std.fmt.allocPrint(
        allocator,
        ".{s}.finetune-replace-{d}-{d}",
        .{ std.fs.path.basename(final_path), std.posix.system.getpid(), nonce },
    );
    defer allocator.free(temporary_leaf);
    const temporary_path = try std.fs.path.join(allocator, &.{ parent, temporary_leaf });
    defer allocator.free(temporary_path);
    errdefer std.Io.Dir.cwd().deleteFile(io, temporary_path) catch {};

    var file = try std.Io.Dir.cwd().createFile(io, temporary_path, .{ .truncate = false, .exclusive = true });
    var open = true;
    errdefer if (open) file.close(io);
    try file.writeStreamingAll(io, data);
    try file.sync(io);
    file.close(io);
    open = false;

    try std.Io.Dir.rename(std.Io.Dir.cwd(), temporary_path, std.Io.Dir.cwd(), final_path, io);
    try syncParentDirectory(io, final_path);
}

pub const ImmutableDirectoryPublication = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    final_dir: []const u8,
    staging_dir: []u8,
    owns_staging: bool = false,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        final_dir: []const u8,
    ) !ImmutableDirectoryPublication {
        if (final_dir.len == 0 or std.mem.eql(u8, final_dir, ".") or std.mem.eql(u8, final_dir, "/")) {
            return error.InvalidArtifactOutputPath;
        }
        if (try pathExists(io, final_dir)) return error.Gemma4RunOutputAlreadyExists;

        const nonce = publication_nonce.fetchAdd(1, .monotonic);
        const staging_leaf = try std.fmt.allocPrint(
            allocator,
            ".{s}.finetune-staging-{d}-{d}",
            .{ std.fs.path.basename(final_dir), std.posix.system.getpid(), nonce },
        );
        defer allocator.free(staging_leaf);
        return .{
            .allocator = allocator,
            .io = io,
            .final_dir = final_dir,
            .staging_dir = try std.fs.path.join(allocator, &.{ std.fs.path.dirname(final_dir) orelse ".", staging_leaf }),
        };
    }

    pub fn deinit(self: *ImmutableDirectoryPublication) void {
        if (self.owns_staging) {
            std.Io.Dir.cwd().deleteTree(self.io, self.staging_dir) catch {};
        }
        self.allocator.free(self.staging_dir);
        self.* = undefined;
    }

    /// Create and claim the private sibling directory. Use this when the
    /// caller itself writes the directory contents.
    pub fn createStaging(self: *ImmutableDirectoryPublication) !void {
        std.debug.assert(!self.owns_staging);
        const parent_dir = std.fs.path.dirname(self.final_dir) orelse ".";
        try std.Io.Dir.cwd().createDirPath(self.io, parent_dir);
        try std.Io.Dir.cwd().createDir(self.io, self.staging_dir, .default_dir);
        self.owns_staging = true;
    }

    /// Claim a staging directory created atomically by a nested artifact
    /// writer. The directory must exist before it is claimed.
    pub fn claimStaging(self: *ImmutableDirectoryPublication) void {
        std.debug.assert(!self.owns_staging);
        self.owns_staging = true;
    }

    pub fn publish(self: *ImmutableDirectoryPublication) !void {
        std.debug.assert(self.owns_staging);
        // The rename is the visibility commit. Persist every regular file and
        // nested directory first so a reported successful publication cannot
        // point at metadata whose payload was still only in the page cache.
        try syncDirectoryTree(self.allocator, self.io, self.staging_dir);
        try renameNoReplace(self.allocator, self.io, self.staging_dir, self.final_dir);
        self.owns_staging = false;
        try syncParentDirectory(self.io, self.final_dir);
    }
};

fn syncDirectoryTree(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !void {
    var dir = if (std.fs.path.isAbsolute(path))
        try std.Io.Dir.openDirAbsolute(io, path, .{ .iterate = true })
    else
        try std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
    defer dir.close(io);

    var iterator = dir.iterate();
    while (try iterator.next(io)) |entry| switch (entry.kind) {
        .file => {
            var file = try dir.openFile(io, entry.name, .{});
            defer file.close(io);
            try file.sync(io);
        },
        .directory => {
            const child = try std.fs.path.join(allocator, &.{ path, entry.name });
            defer allocator.free(child);
            try syncDirectoryTree(allocator, io, child);
        },
        // A fine-tuning artifact must be self-contained. Following symlinks
        // here would make durability and provenance depend on mutable paths
        // outside the transaction.
        else => return error.UnsupportedArtifactEntry,
    };
    try syncDirectoryHandle(&dir);
}

fn renameNoReplace(
    allocator: std.mem.Allocator,
    io: std.Io,
    source: []const u8,
    destination: []const u8,
) !void {
    if (try pathExists(io, destination)) return error.Gemma4RunOutputAlreadyExists;
    if (comptime builtin.os.tag == .macos) {
        const Darwin = struct {
            extern "c" fn renameatx_np(
                old_dir_fd: c_int,
                old_path: [*:0]const u8,
                new_dir_fd: c_int,
                new_path: [*:0]const u8,
                flags: c_uint,
            ) c_int;
        };
        const source_z = try allocator.dupeZ(u8, source);
        defer allocator.free(source_z);
        const destination_z = try allocator.dupeZ(u8, destination);
        defer allocator.free(destination_z);
        const rename_excl: c_uint = 0x0000_0004;
        if (Darwin.renameatx_np(std.posix.AT.FDCWD, source_z.ptr, std.posix.AT.FDCWD, destination_z.ptr, rename_excl) != 0) {
            if (try pathExists(io, destination)) return error.Gemma4RunOutputAlreadyExists;
            return error.Gemma4RunPublishFailed;
        }
        return;
    }
    std.Io.Dir.renamePreserve(std.Io.Dir.cwd(), source, std.Io.Dir.cwd(), destination, io) catch |err| switch (err) {
        error.PathAlreadyExists => return error.Gemma4RunOutputAlreadyExists,
        else => return err,
    };
}

fn syncParentDirectory(io: std.Io, path: []const u8) !void {
    if (builtin.os.tag == .windows or builtin.os.tag == .wasi or builtin.os.tag == .freestanding) return;
    const parent = std.fs.path.dirname(path) orelse if (std.fs.path.isAbsolute(path)) "/" else ".";
    var dir = if (std.fs.path.isAbsolute(parent))
        try std.Io.Dir.openDirAbsolute(io, parent, .{ .iterate = true })
    else
        try std.Io.Dir.cwd().openDir(io, parent, .{ .iterate = true });
    defer dir.close(io);
    try syncDirectoryHandle(&dir);
}

fn syncDirectoryHandle(dir: *const std.Io.Dir) !void {
    if (comptime builtin.os.tag == .windows or builtin.os.tag == .wasi or builtin.os.tag == .freestanding) return;
    while (true) switch (std.posix.errno(std.posix.system.fsync(dir.handle))) {
        .SUCCESS => return,
        .INTR => continue,
        .INVAL => return,
        .BADF => return error.InvalidFileDescriptor,
        .IO => return error.InputOutput,
        .NOSPC => return error.NoSpaceLeft,
        .DQUOT => return error.DiskQuota,
        else => |err| return std.posix.unexpectedErrno(err),
    };
}

fn pathExists(io: std.Io, path: []const u8) !bool {
    std.Io.Dir.cwd().access(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    return true;
}

test "immutable file publication syncs and never replaces an existing artifact" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(root);
    const path = try std.fs.path.join(allocator, &.{ root, "prepared.json" });
    defer allocator.free(path);
    try writeFileImmutable(allocator, io, path, "complete-v1");
    try std.testing.expectError(
        error.Gemma4RunOutputAlreadyExists,
        writeFileImmutable(allocator, io, path, "replacement"),
    );
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(32));
    defer allocator.free(bytes);
    try std.testing.expectEqualStrings("complete-v1", bytes);
}

test "mutable file publication atomically replaces a complete generation" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(std.testing.io, ".", std.testing.allocator);
    defer std.testing.allocator.free(root);
    const path = try std.fs.path.join(std.testing.allocator, &.{ root, "state.json" });
    defer std.testing.allocator.free(path);

    try writeFileAtomicReplace(std.testing.allocator, std.testing.io, path, "{\"generation\":1}");
    try writeFileAtomicReplace(std.testing.allocator, std.testing.io, path, "{\"generation\":2}");
    const rendered = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, std.testing.allocator, .limited(1024));
    defer std.testing.allocator.free(rendered);
    try std.testing.expectEqualStrings("{\"generation\":2}", rendered);
}
