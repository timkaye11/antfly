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
        if (try pathExists(self.io, self.final_dir)) return error.Gemma4RunOutputAlreadyExists;
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
            const staging_z = try self.allocator.dupeZ(u8, self.staging_dir);
            defer self.allocator.free(staging_z);
            const final_z = try self.allocator.dupeZ(u8, self.final_dir);
            defer self.allocator.free(final_z);
            const rename_excl: c_uint = 0x0000_0004;
            if (Darwin.renameatx_np(std.posix.AT.FDCWD, staging_z.ptr, std.posix.AT.FDCWD, final_z.ptr, rename_excl) != 0) {
                if (try pathExists(self.io, self.final_dir)) return error.Gemma4RunOutputAlreadyExists;
                return error.Gemma4RunPublishFailed;
            }
            self.owns_staging = false;
            return;
        }
        std.Io.Dir.renamePreserve(std.Io.Dir.cwd(), self.staging_dir, std.Io.Dir.cwd(), self.final_dir, self.io) catch |err| switch (err) {
            error.PathAlreadyExists => return error.Gemma4RunOutputAlreadyExists,
            else => return err,
        };
        self.owns_staging = false;
    }
};

fn pathExists(io: std.Io, path: []const u8) !bool {
    std.Io.Dir.cwd().access(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    return true;
}
