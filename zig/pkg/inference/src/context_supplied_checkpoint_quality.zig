// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Test-only authenticated adapter for an already-fitted F16-v3 checkpoint.
//! Parsing and every provenance check complete before the caller creates a
//! private Metal owner or submits GPU work.
const std = @import("std");
const checkpoint = @import("runtime/kv/attention_match_checkpoint.zig");
const quality = @import("runtime/kv/metal_checkpoint_quality.zig");
const teacher = @import("backends/context_teacher_forced.zig");

pub const schema = "antfly.supplied-checkpoint-quality.input.v1";

pub const File = struct { path: []const u8, bytes: usize, sha256: []const u8 };
pub const CandidateClosure = struct {
    schema: []const u8,
    diagnostic_only: bool,
    quality_credit: bool,
    production_qualified: bool,
    release_eligible: bool,
    candidate_ready_for_owner_validation: bool,
    numerical_gate_pass: bool,
    global_savings_gate_pass: bool,
    fresh_independent_corpus_required: bool,
    input_manifest_sha256: []const u8,
    fixed_source65_manifest_sha256: []const u8,
    identity: checkpoint.Identity,
    config_sha256: []const u8,
    source_freeze_sha256: []const u8,
    binary_sha256: []const u8,
    exporter_sha256: []const u8,
    prompt_tokens_sha256: []const u8,
    logical_tokens: usize,
    source_checkpoint_sha256: []const u8,
    candidate_checkpoint: File,
    source_global_bytes: usize,
    candidate_global_bytes: usize,
    global_savings_gate: f64,
    completed_chunks: usize,
    validated_heads: usize,
    preserved_heads: usize,
    max_head_mass_error: f64,
    max_head_output_rmse: f64,
};

pub const Input = struct {
    schema: []const u8,
    model: []const u8,
    case_id: []const u8,
    case_manifest_sha256: []const u8,
    prompt_tokens_sha256: []const u8,
    prompt_count: usize,
    closure: File,
    checkpoint: File,
};

pub fn parseDigest(raw: []const u8) ![32]u8 {
    if (raw.len != 64) return error.InvalidSuppliedCheckpointQualityInput;
    var result: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&result, raw) catch return error.InvalidSuppliedCheckpointQualityInput;
    if (std.mem.allEqual(u8, &result, 0)) return error.InvalidSuppliedCheckpointQualityInput;
    return result;
}

fn digest(raw: []const u8) [32]u8 {
    var result: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(raw, &result, .{});
    return result;
}

fn validPath(path: []const u8) bool {
    if (path.len == 0 or std.fs.path.isAbsolute(path)) return false;
    var parts = std.mem.splitScalar(u8, path, '/');
    while (parts.next()) |part| if (part.len == 0 or std.mem.eql(u8, part, ".") or std.mem.eql(u8, part, "..")) return false;
    return true;
}

fn readBound(io: std.Io, allocator: std.mem.Allocator, root: std.Io.Dir, file: File, maximum: usize) ![]u8 {
    if (!validPath(file.path) or file.bytes == 0 or file.bytes > maximum) return error.InvalidSuppliedCheckpointQualityInput;
    const expected = try parseDigest(file.sha256);
    var opened = try root.openFile(io, file.path, .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false, .resolve_beneath = true });
    defer opened.close(io);
    const stat = try opened.stat(io);
    if (stat.kind != .file or stat.size != file.bytes) return error.SuppliedCheckpointChanged;
    const bytes = try allocator.alloc(u8, file.bytes);
    errdefer allocator.free(bytes);
    if (try opened.readPositionalAll(io, bytes, 0) != bytes.len or !std.mem.eql(u8, &digest(bytes), &expected)) return error.SuppliedCheckpointChanged;
    return bytes;
}

pub const Loaded = struct {
    allocator: std.mem.Allocator,
    closure_bytes: []u8,
    checkpoint_bytes: []u8,
    closure: std.json.Parsed(CandidateClosure),

    pub fn deinit(self: *Loaded) void {
        self.closure.deinit();
        self.allocator.free(self.checkpoint_bytes);
        self.allocator.free(self.closure_bytes);
        self.* = undefined;
    }

    pub fn candidate(self: *const Loaded, expected: checkpoint.Identity, prompt: []const i64) !quality.SuppliedCandidate {
        const value = self.closure.value;
        const result: quality.SuppliedCandidate = .{
            .wire = self.checkpoint_bytes,
            .wire_sha256 = try parseDigest(value.candidate_checkpoint.sha256),
            .closure_sha256 = digest(self.closure_bytes),
            .source_checkpoint_sha256 = try parseDigest(value.source_checkpoint_sha256),
            .prompt_tokens_sha256 = try parseDigest(value.prompt_tokens_sha256),
            .identity = value.identity,
            .logical_tokens = value.logical_tokens,
            .source_global_bytes = value.source_global_bytes,
            .candidate_global_bytes = value.candidate_global_bytes,
            .completed_chunks = value.completed_chunks,
            .validated_heads = value.validated_heads,
            .preserved_heads = value.preserved_heads,
            .max_head_mass_error = value.max_head_mass_error,
            .max_head_output_rmse = value.max_head_output_rmse,
            .global_savings_gate = value.global_savings_gate,
            .closure_global_savings_gate_pass = value.global_savings_gate_pass,
        };
        try result.validate(expected, prompt);
        return result;
    }
};

pub fn load(io: std.Io, allocator: std.mem.Allocator, root: std.Io.Dir, input: Input, expected: checkpoint.Identity, prompt: []const i64) !Loaded {
    if (!std.mem.eql(u8, input.schema, schema) or input.model.len == 0 or input.case_id.len == 0 or input.case_id.len > 256 or
        input.prompt_count != prompt.len or !std.mem.eql(u8, &try parseDigest(input.prompt_tokens_sha256), &try teacher.promptTokenDigest(prompt)) or
        std.mem.allEqual(u8, &try parseDigest(input.case_manifest_sha256), 0))
        return error.InvalidSuppliedCheckpointQualityInput;
    const closure_bytes = try readBound(io, allocator, root, input.closure, 4 * 1024 * 1024);
    errdefer allocator.free(closure_bytes);
    var closure = try std.json.parseFromSlice(CandidateClosure, allocator, closure_bytes, .{ .allocate = .alloc_always });
    errdefer closure.deinit();
    const value = closure.value;
    if (!std.mem.eql(u8, value.schema, "antfly.supplied-checkpoint-quality.candidate-closure.v1") or !value.diagnostic_only or value.quality_credit or
        value.production_qualified or value.release_eligible or !value.candidate_ready_for_owner_validation or !value.numerical_gate_pass or
        !value.global_savings_gate_pass or !value.fresh_independent_corpus_required or !value.identity.eql(expected) or
        !std.mem.eql(u8, value.prompt_tokens_sha256, input.prompt_tokens_sha256) or value.logical_tokens != prompt.len or
        !std.mem.eql(u8, value.candidate_checkpoint.path, input.checkpoint.path) or value.candidate_checkpoint.bytes != input.checkpoint.bytes or
        !std.mem.eql(u8, value.candidate_checkpoint.sha256, input.checkpoint.sha256))
        return error.InvalidSuppliedCheckpointQualityInput;
    const checkpoint_bytes = try readBound(io, allocator, root, input.checkpoint, checkpoint.serving_limits.max_file_bytes);
    errdefer allocator.free(checkpoint_bytes);
    var decoded = try checkpoint.decode(allocator, checkpoint_bytes, expected, checkpoint.serving_limits);
    defer decoded.deinit();
    if (decoded.state.logical_length != prompt.len or decoded.state.tokens.len != prompt.len) return error.SuppliedCheckpointBlueprintMismatch;
    for (decoded.state.tokens, prompt) |actual, wanted| if (wanted < 0 or actual != @as(u32, @intCast(wanted))) return error.SuppliedCheckpointBlueprintMismatch;
    const result: Loaded = .{ .allocator = allocator, .closure_bytes = closure_bytes, .checkpoint_bytes = checkpoint_bytes, .closure = closure };
    _ = try result.candidate(expected, prompt);
    return result;
}

test "supplied checkpoint loader round trips typed identity and authenticated tokens" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    // Invalid UTF-8 digest bytes force the JSON byte-array form used by the
    // captured checkpoint closure rather than the serializer's string form.
    const identity: checkpoint.Identity = .{
        .weights = @splat(0xff),
        .tokenizer = @splat(0xfe),
        .template = @splat(0xfd),
        .layout = @splat(0xfc),
    };
    const prompt = [_]i64{ 1, 2 };
    const tokens = [_]u32{ 1, 2 };
    const positions = [_]u64{ 0, 1 };
    const coverage = [_]u64{ 1, 2 };
    const keys = [_]f32{ 0.25, -0.5 };
    const values = [_]f32{ 1.5, -2.0 };
    const bias = [_]f32{ 0, 0 };
    const heads = [_]checkpoint.Head{.{ .dim = 1, .positions = &positions, .coverage_ends = &coverage, .keys = &keys, .values = &values, .bias = &bias }};
    const donors = [_]checkpoint.Donor{.{ .heads = &heads }};
    const state: checkpoint.State = .{
        .identity = identity,
        .logical_length = prompt.len,
        .tokens = &tokens,
        .logits = &.{ 0, 0, 1 },
        .layer_donors = &.{0},
        .donors = &donors,
    };
    const wire = try checkpoint.encodeWithEncoding(allocator, state, checkpoint.serving_limits, .f16, .{});
    defer allocator.free(wire);
    var wire_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(wire, &wire_digest, .{});
    const wire_hex = std.fmt.bytesToHex(wire_digest, .lower);
    const prompt_digest = try teacher.promptTokenDigest(&prompt);
    const prompt_hex = std.fmt.bytesToHex(prompt_digest, .lower);
    const nonzero_hex = std.fmt.bytesToHex(@as([32]u8, @splat(9)), .lower);

    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    try temporary.dir.writeFile(io, .{ .sub_path = "candidate.f16-v3", .data = wire });
    const closure_value = .{
        .schema = "antfly.supplied-checkpoint-quality.candidate-closure.v1",
        .diagnostic_only = true,
        .quality_credit = false,
        .production_qualified = false,
        .release_eligible = false,
        .candidate_ready_for_owner_validation = true,
        .numerical_gate_pass = true,
        .global_savings_gate_pass = true,
        .fresh_independent_corpus_required = true,
        .input_manifest_sha256 = nonzero_hex,
        .fixed_source65_manifest_sha256 = nonzero_hex,
        .identity = identity,
        .config_sha256 = nonzero_hex,
        .source_freeze_sha256 = nonzero_hex,
        .binary_sha256 = nonzero_hex,
        .exporter_sha256 = nonzero_hex,
        .prompt_tokens_sha256 = prompt_hex,
        .logical_tokens = prompt.len,
        .source_checkpoint_sha256 = nonzero_hex,
        .candidate_checkpoint = File{ .path = "candidate.f16-v3", .bytes = wire.len, .sha256 = &wire_hex },
        .source_global_bytes = 300,
        .candidate_global_bytes = 100,
        .global_savings_gate = 1.5,
        .completed_chunks = 1,
        .validated_heads = 1,
        .preserved_heads = 0,
        .max_head_mass_error = 0.01,
        .max_head_output_rmse = 0.01,
    };
    const closure_bytes = try std.json.Stringify.valueAlloc(allocator, closure_value, .{});
    defer allocator.free(closure_bytes);
    try temporary.dir.writeFile(io, .{ .sub_path = "candidate-closure.json", .data = closure_bytes });
    var closure_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(closure_bytes, &closure_digest, .{});
    const closure_hex = std.fmt.bytesToHex(closure_digest, .lower);
    const input: Input = .{
        .schema = schema,
        .model = "fixture",
        .case_id = "fixture-case",
        .case_manifest_sha256 = &nonzero_hex,
        .prompt_tokens_sha256 = &prompt_hex,
        .prompt_count = prompt.len,
        .closure = .{ .path = "candidate-closure.json", .bytes = closure_bytes.len, .sha256 = &closure_hex },
        .checkpoint = .{ .path = "candidate.f16-v3", .bytes = wire.len, .sha256 = &wire_hex },
    };

    var loaded = try load(io, allocator, temporary.dir, input, identity, &prompt);
    defer loaded.deinit();
    try std.testing.expect(loaded.closure.value.identity.eql(identity));
    const candidate = try loaded.candidate(identity, &prompt);
    try std.testing.expect(candidate.identity.eql(identity));
    try std.testing.expectEqualSlices(u8, wire, candidate.wire);

    const missing = [_]i64{1};
    try std.testing.expectError(error.InvalidSuppliedCheckpointQualityInput, load(io, allocator, temporary.dir, input, identity, &missing));
    const swapped = [_]i64{ 2, 1 };
    try std.testing.expectError(error.InvalidSuppliedCheckpointQualityInput, load(io, allocator, temporary.dir, input, identity, &swapped));

    // Replacing the typed 32-byte array with its 64-character hex spelling
    // reproduces the real metadata interop failure and must not parse.
    const marker = "\"weights\":[";
    const start = std.mem.indexOf(u8, closure_bytes, marker) orelse return error.ExpectedIdentityArray;
    const array_start = start + marker.len - 1;
    const array_end = std.mem.indexOfScalarPos(u8, closure_bytes, array_start, ']') orelse return error.ExpectedIdentityArray;
    const identity_hex = std.fmt.bytesToHex(identity.weights, .lower);
    const bad_identity_bytes = try std.mem.concat(allocator, u8, &.{ closure_bytes[0..array_start], "\"", &identity_hex, "\"", closure_bytes[array_end + 1 ..] });
    defer allocator.free(bad_identity_bytes);
    try temporary.dir.writeFile(io, .{ .sub_path = "bad-identity.json", .data = bad_identity_bytes });
    var bad_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bad_identity_bytes, &bad_digest, .{});
    const bad_hex = std.fmt.bytesToHex(bad_digest, .lower);
    var bad_input = input;
    bad_input.closure = .{ .path = "bad-identity.json", .bytes = bad_identity_bytes.len, .sha256 = &bad_hex };
    try std.testing.expectError(error.LengthMismatch, load(io, allocator, temporary.dir, bad_input, identity, &prompt));

    var failing = std.testing.FailingAllocator.init(allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, load(io, failing.allocator(), temporary.dir, input, identity, &prompt));
    try std.testing.expectError(error.InvalidSuppliedCheckpointQualityInput, parseDigest("bad"));
}
