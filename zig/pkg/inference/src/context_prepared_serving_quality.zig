// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
const std = @import("std");
const closure_mod = @import("server/native_kv_prepared_quality_closure.zig");
const sidecar = @import("server/native_kv_verified_sidecar.zig");
const checkpoint = @import("runtime/kv/attention_match_checkpoint.zig");
const teacher = @import("backends/context_teacher_forced.zig");

pub const schema = "antfly.prepared-serving-quality.input.v1";
const max_private_input_bytes: usize = 512 * 1024 * 1024;
pub const File = struct { path: []const u8, bytes: usize, sha256: []const u8 };
pub const Input = struct {
    schema: []const u8,
    model: []const u8,
    case_id: []const u8,
    case_manifest_sha256: []const u8,
    prompt_tokens_sha256: []const u8,
    prompt_count: usize,
    frame: File,
    source: File,
    closure: File,
};

fn parseDigest(raw: []const u8) ![32]u8 {
    if (raw.len != 64) return error.InvalidPreparedServingQualityInput;
    var result: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&result, raw) catch return error.InvalidPreparedServingQualityInput;
    if (std.mem.allEqual(u8, &result, 0)) return error.InvalidPreparedServingQualityInput;
    return result;
}

fn validPath(path: []const u8) bool {
    if (path.len == 0 or std.fs.path.isAbsolute(path)) return false;
    var parts = std.mem.splitScalar(u8, path, '/');
    while (parts.next()) |part| if (part.len == 0 or std.mem.eql(u8, part, ".") or std.mem.eql(u8, part, "..")) return false;
    return true;
}

fn readBound(io: std.Io, allocator: std.mem.Allocator, root: std.Io.Dir, file: File, maximum: usize) ![]u8 {
    if (!validPath(file.path) or file.bytes == 0 or file.bytes > maximum) return error.InvalidPreparedServingQualityInput;
    var opened = try root.openFile(io, file.path, .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false, .resolve_beneath = true });
    defer opened.close(io);
    const stat = try opened.stat(io);
    if (stat.kind != .file or stat.size != @as(u64, @intCast(file.bytes))) return error.PreparedServingQualityInputChanged;
    const bytes = try allocator.alloc(u8, file.bytes);
    errdefer allocator.free(bytes);
    if (try opened.readPositionalAll(io, bytes, 0) != bytes.len) return error.PreparedServingQualityInputChanged;
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    if (!std.mem.eql(u8, &digest, &try parseDigest(file.sha256))) return error.PreparedServingQualityInputChanged;
    return bytes;
}

pub const Loaded = struct {
    allocator: std.mem.Allocator,
    frame: []u8,
    source: []u8,
    closure_bytes: []u8,
    prepared: closure_mod.Prepared,

    pub fn deinit(self: *Loaded) void {
        self.prepared.deinit();
        self.allocator.free(self.closure_bytes);
        self.allocator.free(self.source);
        self.allocator.free(self.frame);
        self.* = undefined;
    }
};

pub fn load(io: std.Io, allocator: std.mem.Allocator, root: std.Io.Dir, input: Input, expected_identity: checkpoint.Identity, expected_producer: sidecar.ExpectedProducer, prompt: []const i64) !Loaded {
    if (!std.mem.eql(u8, input.schema, schema) or input.model.len == 0 or input.case_id.len == 0 or input.prompt_count != prompt.len or
        !std.mem.eql(u8, &try parseDigest(input.prompt_tokens_sha256), &try teacher.operatorPromptTokenDigest(&.{}, prompt)) or
        std.mem.allEqual(u8, &try parseDigest(input.case_manifest_sha256), 0))
        return error.InvalidPreparedServingQualityInput;
    const total_bytes = std.math.add(usize, input.frame.bytes, input.source.bytes) catch return error.InvalidPreparedServingQualityInput;
    const complete_bytes = std.math.add(usize, total_bytes, input.closure.bytes) catch return error.InvalidPreparedServingQualityInput;
    if (complete_bytes > max_private_input_bytes) return error.InvalidPreparedServingQualityInput;
    const frame = try readBound(io, allocator, root, input.frame, checkpoint.serving_limits.max_file_bytes + 1024 * 1024);
    errdefer allocator.free(frame);
    const closure_bytes = try readBound(io, allocator, root, input.closure, 4 * 1024 * 1024);
    errdefer allocator.free(closure_bytes);
    const source = try readBound(io, allocator, root, input.source, checkpoint.serving_limits.max_file_bytes);
    errdefer allocator.free(source);
    const prepared = try closure_mod.verifyPrepared(allocator, frame, source, closure_bytes, expected_identity, expected_producer, try teacher.operatorPromptTokenDigest(&.{}, prompt));
    return .{ .allocator = allocator, .frame = frame, .source = source, .closure_bytes = closure_bytes, .prepared = prepared };
}

test "context prepared serving input paths are relative and immutable" {
    try std.testing.expect(validPath("prepared/f16.afatt002"));
    inline for (.{ "", "/absolute", "../escape", "a//b" }) |path| try std.testing.expect(!validPath(path));
}
