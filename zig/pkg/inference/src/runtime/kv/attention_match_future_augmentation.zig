// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Exact future-decode TRAIN256 export. This contract is deliberately distinct
//! from generic training augmentation: only backend-owned generated frames from
//! an unchanged exact source may authorize the nested future query payload.
const std = @import("std");
const checkpoint = @import("attention_match_checkpoint.zig");
const attention_match = @import("attention_match.zig");
const engine = @import("attention_match_engine.zig");
const queries = @import("attention_match_queries.zig");
const augmentation = @import("attention_match_training_augmentation.zig");
const self_study = @import("attention_match_self_study_training.zig");
const future_training = @import("attention_match_future_decode_training.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const schema_v1 = "antfly.attention-match-future-augmentation.v1";
pub const schema_v2 = "antfly.attention-match-future-augmentation.v2";
pub const schema = schema_v1;
pub const maximum_bytes: usize = 128 * 1024 * 1024;

pub const FutureTrainingCaptureEvidence = struct {
    generated_ids: [256]i64,
    generated_ids_count: u16,
    generated_ids_sha256: [32]u8,
    source_before: self_study.SourceBoundary,
    source_after: self_study.SourceBoundary,
    early_eos_observations: u16,
    backend_owned_full_logits_calls: u16,
    backend_owned_full_logits_ns: u64,
    backend_owned_frame_successes: u16,
    generic_last_logits_calls: u16,
    backend_owned_full_logits_selected: bool,
    gpu_greedy_calls: u16 = 0,
    gpu_greedy_ns: u64 = 0,
    gpu_greedy_frame_successes: u16 = 0,
    gpu_greedy_selected: bool = false,
    primary_rows: u16 = 256,
    historical_rows: u16 = 256,
    future_rows: u16 = 256,
    heldout_rows: u16 = 64,
    heldout_sha256_before_fit: [32]u8 = @splat(0),

    pub fn validate(self: @This(), authority: future_training.Authority) !void {
        var ids_hash = Sha256.init(.{});
        for (self.generated_ids) |id| ids_hash.update(&std.mem.toBytes(std.mem.nativeToLittle(i64, id)));
        var source_hash = Sha256.init(.{});
        inline for (.{ self.source_before.identity.weights, self.source_before.identity.tokenizer, self.source_before.identity.template, self.source_before.identity.layout }) |value| source_hash.update(&value);
        source_hash.update(&std.mem.toBytes(std.mem.nativeToLittle(u64, self.source_before.sequence_id)));
        source_hash.update(&std.mem.toBytes(std.mem.nativeToLittle(u64, @intCast(self.source_before.logical_tokens))));
        const actual_ids_sha256 = ids_hash.finalResult();
        const actual_source_identity_sha256 = source_hash.finalResult();
        const domains = [_][32]u8{ authority.primary_sha256, authority.historical_sha256, authority.future_sha256, authority.heldout_sha256 };
        for (domains, 0..) |value, index| {
            if (std.mem.allEqual(u8, &value, 0)) return error.InvalidNativeQualityPublication;
            for (domains[0..index]) |prior| if (std.mem.eql(u8, &value, &prior)) return error.InvalidNativeQualityPublication;
        }
        const full_logits_lane = self.backend_owned_full_logits_selected and !self.gpu_greedy_selected and
            self.backend_owned_full_logits_calls == 256 and self.backend_owned_frame_successes == 256 and
            self.backend_owned_full_logits_ns > 0 and self.gpu_greedy_calls == 0 and
            self.gpu_greedy_frame_successes == 0 and self.gpu_greedy_ns == 0;
        const gpu_greedy_lane = self.gpu_greedy_selected and !self.backend_owned_full_logits_selected and
            self.gpu_greedy_calls == 256 and self.gpu_greedy_frame_successes == 256 and
            self.gpu_greedy_ns > 0 and self.backend_owned_full_logits_calls == 0 and
            self.backend_owned_frame_successes == 0 and self.backend_owned_full_logits_ns == 0;
        if (self.generated_ids_count != 256 or !std.mem.eql(u8, &actual_ids_sha256, &self.generated_ids_sha256) or
            !std.mem.eql(u8, &self.generated_ids_sha256, &authority.generated_ids_sha256) or
            !std.mem.eql(u8, &actual_source_identity_sha256, &authority.source_identity_sha256) or
            !std.meta.eql(self.source_before, self.source_after) or self.source_before.kv_revision != authority.source_revision or
            !std.mem.eql(u8, &self.source_before.tokens_sha256, &authority.source_tokens_sha256) or
            self.early_eos_observations != authority.early_eos_observations or
            (!full_logits_lane and !gpu_greedy_lane) or self.generic_last_logits_calls != 0 or
            self.primary_rows != 256 or self.historical_rows != 256 or self.future_rows != 256 or self.heldout_rows != 64 or
            !std.mem.eql(u8, &self.heldout_sha256_before_fit, &authority.heldout_sha256))
            return error.InvalidNativeQualityPublication;
    }
};

pub const Binding = struct {
    output_dir: []const u8,
    primary_parent_manifest_sha256: [32]u8,
    identity: checkpoint.Identity,
    source_freeze_sha256: [32]u8,
    binary_sha256: [32]u8,
    exporter_sha256: [32]u8,
};
pub const View = struct {
    capture: FutureTrainingCaptureEvidence,
    authority: future_training.Authority,
    batches: []const engine.ReferenceBatch,
    consumers: []const queries.Consumer,
    source_length: usize,
    control: ?attention_match.Control = null,
    historical_study_tokens_sha256: [32]u8 = @splat(0),
};
pub const Result = struct { manifest_sha256: [32]u8, nested_manifest_sha256: [32]u8, required_bytes: usize };
const Payload = struct { name: []const u8, bytes: usize, count: usize, sha256: [32]u8 };
const Manifest = struct {
    schema: []const u8,
    diagnostic_only: bool,
    quality_credit: bool,
    training_only: bool,
    heldout_present: bool,
    samples_per_consumer: u16,
    primary_parent_manifest_sha256: [64]u8,
    source_freeze_sha256: [64]u8,
    binary_sha256: [64]u8,
    exporter_sha256: [64]u8,
    authority_sha256: [64]u8,
    generation_contract_sha256: [64]u8,
    generated_ids_sha256: [64]u8,
    source_before: self_study.SourceBoundary,
    source_after: self_study.SourceBoundary,
    early_eos_observations: u16,
    backend_owned_full_logits_calls: u16,
    backend_owned_full_logits_ns: u64,
    backend_owned_frame_successes: u16,
    generic_last_logits_calls: u16,
    backend_owned_full_logits_selected: bool,
    gpu_greedy_calls: u16 = 0,
    gpu_greedy_ns: u64 = 0,
    gpu_greedy_frame_successes: u16 = 0,
    gpu_greedy_selected: bool = false,
    primary_rows: u16,
    historical_rows: u16,
    future_rows: u16,
    heldout_rows: u16,
    heldout_sha256_before_fit: [64]u8,
    nested_training_path: []const u8,
    nested_training_manifest_sha256: [64]u8,
    payloads: []const Payload,
};

fn validate(binding: Binding, view: View) !void {
    if (binding.output_dir.len == 0 or !std.fs.path.isAbsolute(binding.output_dir) or view.source_length == 0 or view.batches.len == 0) return error.InvalidFutureAugmentation;
    try view.capture.validate(view.authority);
    if (!std.meta.eql(binding.identity, view.capture.source_before.identity)) return error.InvalidFutureAugmentation;
    for (view.batches) |batch| if (batch.train_positions.len != future_training.training_rows or batch.train.len == 0 or batch.train.len % future_training.training_rows != 0 or batch.heldout_positions.len != future_training.heldout_rows or batch.independent_train_split != future_training.primary_rows or batch.independent_train_second_split != future_training.primary_rows + future_training.historical_rows) return error.InvalidFutureAugmentation;
}

pub fn write(allocator: std.mem.Allocator, io: std.Io, binding: Binding, view: View) !Result {
    try validate(binding, view);
    const future_batches = try allocator.alloc(engine.ReferenceBatch, view.batches.len);
    defer allocator.free(future_batches);
    for (view.batches, future_batches) |batch, *future| {
        const dim = batch.train.len / future_training.training_rows;
        const begin = (future_training.primary_rows + future_training.historical_rows) * dim;
        future.* = .{ .consumer_layer = batch.consumer_layer, .query_head = batch.query_head, .train = batch.train[begin..], .heldout = &.{}, .train_positions = batch.train_positions[future_training.primary_rows + future_training.historical_rows ..], .heldout_positions = &.{} };
    }
    try std.Io.Dir.createDirAbsolute(io, binding.output_dir, std.Io.File.Permissions.fromMode(0o700));
    const nested_dir = try std.fs.path.join(allocator, &.{ binding.output_dir, "training" });
    defer allocator.free(nested_dir);
    const nested = try augmentation.write(allocator, io, .{ .output_dir = nested_dir, .parent_manifest_sha256 = binding.primary_parent_manifest_sha256, .identity = binding.identity, .source_freeze_sha256 = binding.source_freeze_sha256, .binary_sha256 = binding.binary_sha256, .exporter_sha256 = binding.exporter_sha256, .parent_study_tokens_sha256 = view.authority.source_tokens_sha256, .study_tokens_sha256 = view.capture.generated_ids_sha256, .samples_per_stratum = 256, .max_bytes = maximum_bytes - 64 * 1024 }, .{ .batches = future_batches, .consumers = view.consumers, .samples_per_stratum = 256, .parent_study_tokens_sha256 = view.authority.source_tokens_sha256, .study_tokens_sha256 = view.capture.generated_ids_sha256, .source_length = view.source_length, .repeat_length = 256 });
    var dir = try std.Io.Dir.openDirAbsolute(io, binding.output_dir, .{ .follow_symlinks = false });
    defer dir.close(io);
    const id_bytes = std.mem.sliceAsBytes(&view.capture.generated_ids);
    var ids_file = try dir.createFile(io, "future.generated_ids.i64", .{ .exclusive = true, .truncate = false });
    defer ids_file.close(io);
    try ids_file.writeStreamingAll(io, id_bytes);
    try ids_file.sync(io);
    const payloads = [_]Payload{.{ .name = "future.generated_ids.i64", .bytes = id_bytes.len, .count = 256, .sha256 = view.capture.generated_ids_sha256 }};
    // Default-off remains the exact V1 JSON contract. Only the complete GPU
    // lane publishes V2 and its four additional counters.
    const body = if (view.capture.gpu_greedy_selected)
        try std.json.Stringify.valueAlloc(allocator, .{ .schema = schema_v2, .diagnostic_only = true, .quality_credit = false, .training_only = true, .heldout_present = false, .samples_per_consumer = @as(u16, 256), .primary_parent_manifest_sha256 = std.fmt.bytesToHex(binding.primary_parent_manifest_sha256, .lower), .source_freeze_sha256 = std.fmt.bytesToHex(binding.source_freeze_sha256, .lower), .binary_sha256 = std.fmt.bytesToHex(binding.binary_sha256, .lower), .exporter_sha256 = std.fmt.bytesToHex(binding.exporter_sha256, .lower), .authority_sha256 = std.fmt.bytesToHex(view.authority.digest(), .lower), .generation_contract_sha256 = std.fmt.bytesToHex(view.authority.generation_contract_sha256, .lower), .generated_ids_sha256 = std.fmt.bytesToHex(view.capture.generated_ids_sha256, .lower), .source_before = view.capture.source_before, .source_after = view.capture.source_after, .early_eos_observations = view.capture.early_eos_observations, .backend_owned_full_logits_calls = view.capture.backend_owned_full_logits_calls, .backend_owned_full_logits_ns = view.capture.backend_owned_full_logits_ns, .backend_owned_frame_successes = view.capture.backend_owned_frame_successes, .generic_last_logits_calls = view.capture.generic_last_logits_calls, .backend_owned_full_logits_selected = view.capture.backend_owned_full_logits_selected, .gpu_greedy_calls = view.capture.gpu_greedy_calls, .gpu_greedy_ns = view.capture.gpu_greedy_ns, .gpu_greedy_frame_successes = view.capture.gpu_greedy_frame_successes, .gpu_greedy_selected = view.capture.gpu_greedy_selected, .primary_rows = view.capture.primary_rows, .historical_rows = view.capture.historical_rows, .future_rows = view.capture.future_rows, .heldout_rows = view.capture.heldout_rows, .heldout_sha256_before_fit = std.fmt.bytesToHex(view.capture.heldout_sha256_before_fit, .lower), .nested_training_path = "training", .nested_training_manifest_sha256 = std.fmt.bytesToHex(nested.manifest_sha256, .lower), .payloads = &payloads }, .{})
    else
        try std.json.Stringify.valueAlloc(allocator, .{ .schema = schema_v1, .diagnostic_only = true, .quality_credit = false, .training_only = true, .heldout_present = false, .samples_per_consumer = @as(u16, 256), .primary_parent_manifest_sha256 = std.fmt.bytesToHex(binding.primary_parent_manifest_sha256, .lower), .source_freeze_sha256 = std.fmt.bytesToHex(binding.source_freeze_sha256, .lower), .binary_sha256 = std.fmt.bytesToHex(binding.binary_sha256, .lower), .exporter_sha256 = std.fmt.bytesToHex(binding.exporter_sha256, .lower), .authority_sha256 = std.fmt.bytesToHex(view.authority.digest(), .lower), .generation_contract_sha256 = std.fmt.bytesToHex(view.authority.generation_contract_sha256, .lower), .generated_ids_sha256 = std.fmt.bytesToHex(view.capture.generated_ids_sha256, .lower), .source_before = view.capture.source_before, .source_after = view.capture.source_after, .early_eos_observations = view.capture.early_eos_observations, .backend_owned_full_logits_calls = view.capture.backend_owned_full_logits_calls, .backend_owned_full_logits_ns = view.capture.backend_owned_full_logits_ns, .backend_owned_frame_successes = view.capture.backend_owned_frame_successes, .generic_last_logits_calls = view.capture.generic_last_logits_calls, .backend_owned_full_logits_selected = view.capture.backend_owned_full_logits_selected, .primary_rows = view.capture.primary_rows, .historical_rows = view.capture.historical_rows, .future_rows = view.capture.future_rows, .heldout_rows = view.capture.heldout_rows, .heldout_sha256_before_fit = std.fmt.bytesToHex(view.capture.heldout_sha256_before_fit, .lower), .nested_training_path = "training", .nested_training_manifest_sha256 = std.fmt.bytesToHex(nested.manifest_sha256, .lower), .payloads = &payloads }, .{});
    defer allocator.free(body);
    if (body.len + 1 > 64 * 1024) return error.FutureAugmentationLimit;
    var hash = Sha256.init(.{});
    hash.update(body);
    hash.update("\n");
    var file = try dir.createFile(io, "manifest.json.part", .{ .exclusive = true, .truncate = false });
    defer file.close(io);
    try file.writeStreamingAll(io, body);
    try file.writeStreamingAll(io, "\n");
    try file.sync(io);
    try std.Io.Dir.rename(dir, "manifest.json.part", dir, "manifest.json", io);
    return .{ .manifest_sha256 = hash.finalResult(), .nested_manifest_sha256 = nested.manifest_sha256, .required_bytes = nested.required_bytes + body.len + 1 + id_bytes.len };
}

pub const ReadBinding = struct {
    output_dir: []const u8,
    manifest_sha256: [32]u8,
    primary_parent_manifest_sha256: [32]u8,
    identity: checkpoint.Identity,
    source_freeze_sha256: [32]u8,
    binary_sha256: [32]u8,
    exporter_sha256: [32]u8,
    authority: future_training.Authority,
};

fn hex(value: [32]u8) [64]u8 {
    return std.fmt.bytesToHex(value, .lower);
}

fn futureDomainDigest(batches: []const engine.ReferenceBatch) ![32]u8 {
    var hash = Sha256.init(.{});
    if (batches.len == 0) return error.InvalidFutureAugmentation;
    for (batches) |batch| {
        if (batch.train_positions.len != future_training.future_rows or batch.train.len == 0 or batch.train.len % future_training.future_rows != 0 or batch.heldout.len != 0 or batch.heldout_positions.len != 0) return error.InvalidFutureAugmentation;
        hash.update(std.mem.sliceAsBytes(batch.train));
        hash.update(std.mem.sliceAsBytes(batch.train_positions));
    }
    return hash.finalResult();
}

pub fn read(allocator: std.mem.Allocator, io: std.Io, binding: ReadBinding) !augmentation.Owned {
    if (binding.output_dir.len == 0 or !std.fs.path.isAbsolute(binding.output_dir)) return error.InvalidFutureAugmentation;
    var dir = try std.Io.Dir.openDirAbsolute(io, binding.output_dir, .{ .follow_symlinks = false });
    defer dir.close(io);
    const bytes = try dir.readFileAlloc(io, "manifest.json", allocator, .limited(64 * 1024));
    defer allocator.free(bytes);
    var actual: [32]u8 = undefined;
    Sha256.hash(bytes, &actual, .{});
    if (!std.mem.eql(u8, &actual, &binding.manifest_sha256)) return error.InvalidFutureAugmentation;
    const parsed = try std.json.parseFromSlice(Manifest, allocator, bytes, .{ .ignore_unknown_fields = false });
    defer parsed.deinit();
    const m = parsed.value;
    const is_v1 = std.mem.eql(u8, m.schema, schema_v1);
    const is_v2 = std.mem.eql(u8, m.schema, schema_v2);
    const full_logits_lane = m.backend_owned_full_logits_selected and !m.gpu_greedy_selected and
        m.backend_owned_full_logits_calls == 256 and m.backend_owned_frame_successes == 256 and
        m.backend_owned_full_logits_ns > 0 and m.gpu_greedy_calls == 0 and
        m.gpu_greedy_frame_successes == 0 and m.gpu_greedy_ns == 0;
    const gpu_greedy_lane = m.gpu_greedy_selected and !m.backend_owned_full_logits_selected and
        m.gpu_greedy_calls == 256 and m.gpu_greedy_frame_successes == 256 and m.gpu_greedy_ns > 0 and
        m.backend_owned_full_logits_calls == 0 and m.backend_owned_frame_successes == 0 and m.backend_owned_full_logits_ns == 0;
    if ((!is_v1 and !is_v2) or (is_v1 and !full_logits_lane) or (is_v2 and !gpu_greedy_lane) or
        !m.diagnostic_only or m.quality_credit or !m.training_only or m.heldout_present or m.samples_per_consumer != 256 or
        !std.mem.eql(u8, &m.primary_parent_manifest_sha256, &hex(binding.primary_parent_manifest_sha256)) or !std.mem.eql(u8, &m.source_freeze_sha256, &hex(binding.source_freeze_sha256)) or
        !std.mem.eql(u8, &m.binary_sha256, &hex(binding.binary_sha256)) or !std.mem.eql(u8, &m.exporter_sha256, &hex(binding.exporter_sha256)) or
        !std.mem.eql(u8, &m.authority_sha256, &hex(binding.authority.digest())) or !std.mem.eql(u8, &m.generation_contract_sha256, &hex(binding.authority.generation_contract_sha256)) or
        !std.mem.eql(u8, &m.generated_ids_sha256, &hex(binding.authority.generated_ids_sha256)) or !std.meta.eql(m.source_before, m.source_after) or
        !std.meta.eql(m.source_before.identity, binding.identity) or m.source_before.kv_revision != binding.authority.source_revision or
        !std.mem.eql(u8, &m.source_before.tokens_sha256, &binding.authority.source_tokens_sha256) or m.early_eos_observations != binding.authority.early_eos_observations or
        m.generic_last_logits_calls != 0 or
        m.primary_rows != 256 or m.historical_rows != 256 or m.future_rows != 256 or m.heldout_rows != 64 or !std.mem.eql(u8, &m.heldout_sha256_before_fit, &hex(binding.authority.heldout_sha256)) or !std.mem.eql(u8, m.nested_training_path, "training") or m.payloads.len != 1)
        return error.InvalidFutureAugmentation;
    const row = m.payloads[0];
    if (!std.mem.eql(u8, row.name, "future.generated_ids.i64") or row.count != 256 or row.bytes != 256 * @sizeOf(i64) or !std.mem.eql(u8, &row.sha256, &binding.authority.generated_ids_sha256)) return error.InvalidFutureAugmentation;
    const ids = try allocator.alloc(i64, 256);
    defer allocator.free(ids);
    var id_file = try dir.openFile(io, row.name, .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false });
    defer id_file.close(io);
    const id_bytes = std.mem.sliceAsBytes(ids);
    if (try id_file.readPositionalAll(io, id_bytes, 0) != id_bytes.len) return error.InvalidFutureAugmentation;
    var extra: [1]u8 = undefined;
    if (try id_file.readPositionalAll(io, &extra, id_bytes.len) != 0) return error.InvalidFutureAugmentation;
    var id_sha: [32]u8 = undefined;
    Sha256.hash(id_bytes, &id_sha, .{});
    if (!std.mem.eql(u8, &id_sha, &row.sha256)) return error.InvalidFutureAugmentation;
    const nested_dir = try std.fs.path.join(allocator, &.{ binding.output_dir, "training" });
    defer allocator.free(nested_dir);
    const nested_path = try std.fs.path.join(allocator, &.{ nested_dir, "manifest.json" });
    defer allocator.free(nested_path);
    const nested_bytes = try std.Io.Dir.cwd().readFileAlloc(io, nested_path, allocator, .limited(64 * 1024));
    defer allocator.free(nested_bytes);
    var nested_sha: [32]u8 = undefined;
    Sha256.hash(nested_bytes, &nested_sha, .{});
    if (!std.mem.eql(u8, &m.nested_training_manifest_sha256, &hex(nested_sha))) return error.InvalidFutureAugmentation;
    var owned = try augmentation.read(allocator, io, .{ .output_dir = nested_dir, .parent_manifest_sha256 = binding.primary_parent_manifest_sha256, .identity = binding.identity, .source_freeze_sha256 = binding.source_freeze_sha256, .binary_sha256 = binding.binary_sha256, .exporter_sha256 = binding.exporter_sha256, .parent_study_tokens_sha256 = binding.authority.source_tokens_sha256, .study_tokens_sha256 = binding.authority.generated_ids_sha256, .samples_per_stratum = 256, .max_bytes = maximum_bytes - 64 * 1024 });
    errdefer owned.deinit();
    for (owned.batches) |batch| if (batch.train_positions.len != 256 or batch.heldout.len != 0 or batch.heldout_positions.len != 0) return error.InvalidFutureAugmentation;
    const reopened_future_sha256 = try futureDomainDigest(owned.batches);
    if (!std.mem.eql(u8, &reopened_future_sha256, &binding.authority.future_sha256)) return error.InvalidFutureAugmentation;
    return owned;
}

/// A self-contained pre-fit bundle. The primary TRAIN256 and heldout64 remain
/// in the authenticated whole-fit parent; the two nested payloads are the
/// exact selected historical and generated-future TRAIN256 domains. The outer
/// manifest is published last and binds all four byte domains.
pub const complete_schema = "antfly.attention-match-complete-future-mix.v2";
pub const CompleteBinding = Binding;
pub const CompleteView = View;
pub const CompleteResult = struct {
    manifest_sha256: [32]u8,
    historical_manifest_sha256: [32]u8,
    future_manifest_sha256: [32]u8,
    required_bytes: usize,
};
const CompleteManifest = struct {
    schema: []const u8,
    diagnostic_only: bool,
    quality_credit: bool,
    training_only: bool,
    primary_rows: u16,
    historical_rows: u16,
    future_rows: u16,
    heldout_rows: u16,
    source_length: usize,
    historical_repeat_length: usize,
    future_repeat_length: usize,
    primary_parent_manifest_sha256: [64]u8,
    source_freeze_sha256: [64]u8,
    binary_sha256: [64]u8,
    exporter_sha256: [64]u8,
    authority_sha256: [64]u8,
    primary_sha256: [64]u8,
    historical_sha256: [64]u8,
    historical_study_tokens_sha256: [64]u8,
    future_sha256: [64]u8,
    heldout_sha256: [64]u8,
    historical_path: []const u8,
    historical_manifest_sha256: [64]u8,
    future_path: []const u8,
    future_manifest_sha256: [64]u8,
};

fn domainDigest(batches: []const engine.ReferenceBatch, comptime domain: enum { primary, historical, future, heldout }) ![32]u8 {
    var hash = Sha256.init(.{});
    if (batches.len == 0) return error.InvalidFutureAugmentation;
    for (batches) |batch| {
        if (batch.train_positions.len != future_training.training_rows or batch.independent_train_split != future_training.primary_rows or batch.independent_train_second_split != future_training.primary_rows + future_training.historical_rows or batch.train.len == 0 or batch.train.len % future_training.training_rows != 0 or batch.heldout_positions.len != future_training.heldout_rows) return error.InvalidFutureAugmentation;
        const dim = batch.train.len / future_training.training_rows;
        const first = future_training.primary_rows * dim;
        const second = (future_training.primary_rows + future_training.historical_rows) * dim;
        switch (domain) {
            .primary => {
                hash.update(std.mem.sliceAsBytes(batch.train[0..first]));
                hash.update(std.mem.sliceAsBytes(batch.train_positions[0..future_training.primary_rows]));
            },
            .historical => {
                hash.update(std.mem.sliceAsBytes(batch.train[first..second]));
                hash.update(std.mem.sliceAsBytes(batch.train_positions[future_training.primary_rows .. future_training.primary_rows + future_training.historical_rows]));
            },
            .future => {
                hash.update(std.mem.sliceAsBytes(batch.train[second..]));
                hash.update(std.mem.sliceAsBytes(batch.train_positions[future_training.primary_rows + future_training.historical_rows ..]));
            },
            .heldout => {
                hash.update(std.mem.sliceAsBytes(batch.heldout));
                hash.update(std.mem.sliceAsBytes(batch.heldout_positions));
            },
        }
    }
    return hash.finalResult();
}

fn slicedDomain(allocator: std.mem.Allocator, batches: []const engine.ReferenceBatch, historical: bool) ![]engine.ReferenceBatch {
    const out = try allocator.alloc(engine.ReferenceBatch, batches.len);
    errdefer allocator.free(out);
    for (batches, out) |batch, *dst| {
        const dim = batch.train.len / future_training.training_rows;
        const first = future_training.primary_rows * dim;
        const second = (future_training.primary_rows + future_training.historical_rows) * dim;
        dst.* = if (historical)
            .{ .consumer_layer = batch.consumer_layer, .query_head = batch.query_head, .train = batch.train[first..second], .heldout = &.{}, .train_positions = batch.train_positions[future_training.primary_rows .. future_training.primary_rows + future_training.historical_rows], .heldout_positions = &.{} }
        else
            .{ .consumer_layer = batch.consumer_layer, .query_head = batch.query_head, .train = batch.train[second..], .heldout = &.{}, .train_positions = batch.train_positions[future_training.primary_rows + future_training.historical_rows ..], .heldout_positions = &.{} };
    }
    return out;
}

pub fn writeComplete(allocator: std.mem.Allocator, io: std.Io, binding: CompleteBinding, view: CompleteView) !CompleteResult {
    try validate(binding, view);
    if (std.mem.allEqual(u8, &view.historical_study_tokens_sha256, 0) or std.mem.eql(u8, &view.historical_study_tokens_sha256, &view.authority.source_tokens_sha256) or std.mem.eql(u8, &view.historical_study_tokens_sha256, &view.authority.generated_ids_sha256)) return error.InvalidFutureAugmentation;
    if (view.control) |control| try control.check();
    const primary_digest = try domainDigest(view.batches, .primary);
    const historical_digest = try domainDigest(view.batches, .historical);
    const future_digest = try domainDigest(view.batches, .future);
    const heldout_digest = try domainDigest(view.batches, .heldout);
    if (!std.mem.eql(u8, &primary_digest, &view.authority.primary_sha256) or
        !std.mem.eql(u8, &historical_digest, &view.authority.historical_sha256) or
        !std.mem.eql(u8, &future_digest, &view.authority.future_sha256) or
        !std.mem.eql(u8, &heldout_digest, &view.authority.heldout_sha256)) return error.InvalidFutureAugmentation;
    try std.Io.Dir.createDirAbsolute(io, binding.output_dir, std.Io.File.Permissions.fromMode(0o700));
    var committed = false;
    errdefer if (!committed) std.Io.Dir.cwd().deleteTree(io, binding.output_dir) catch {};
    const historical_dir = try std.fs.path.join(allocator, &.{ binding.output_dir, "historical" });
    defer allocator.free(historical_dir);
    const future_dir = try std.fs.path.join(allocator, &.{ binding.output_dir, "future" });
    defer allocator.free(future_dir);
    const historical = try slicedDomain(allocator, view.batches, true);
    defer allocator.free(historical);
    const common = .{ .parent_manifest_sha256 = binding.primary_parent_manifest_sha256, .identity = binding.identity, .source_freeze_sha256 = binding.source_freeze_sha256, .binary_sha256 = binding.binary_sha256, .exporter_sha256 = binding.exporter_sha256, .parent_study_tokens_sha256 = view.authority.source_tokens_sha256, .samples_per_stratum = @as(u16, 256), .max_bytes = maximum_bytes / 2 };
    if (view.control) |control| try control.check();
    const historical_result = try augmentation.write(allocator, io, .{ .output_dir = historical_dir, .parent_manifest_sha256 = common.parent_manifest_sha256, .identity = common.identity, .source_freeze_sha256 = common.source_freeze_sha256, .binary_sha256 = common.binary_sha256, .exporter_sha256 = common.exporter_sha256, .parent_study_tokens_sha256 = common.parent_study_tokens_sha256, .study_tokens_sha256 = view.historical_study_tokens_sha256, .samples_per_stratum = common.samples_per_stratum, .max_bytes = common.max_bytes }, .{ .batches = historical, .consumers = view.consumers, .samples_per_stratum = 256, .parent_study_tokens_sha256 = view.authority.source_tokens_sha256, .study_tokens_sha256 = view.historical_study_tokens_sha256, .source_length = view.source_length, .repeat_length = 512 });
    if (view.control) |control| try control.check();
    // Reuse the strict future-only producer as the nested future member so the
    // complete bundle carries authenticated raw IDs, frame counters, source
    // before/after, heldout-before-fit and generation contract evidence.
    const future_result = try write(allocator, io, .{ .output_dir = future_dir, .primary_parent_manifest_sha256 = binding.primary_parent_manifest_sha256, .identity = binding.identity, .source_freeze_sha256 = binding.source_freeze_sha256, .binary_sha256 = binding.binary_sha256, .exporter_sha256 = binding.exporter_sha256 }, view);
    if (view.control) |control| try control.check();
    const body = try std.json.Stringify.valueAlloc(allocator, CompleteManifest{ .schema = complete_schema, .diagnostic_only = true, .quality_credit = false, .training_only = true, .primary_rows = 256, .historical_rows = 256, .future_rows = 256, .heldout_rows = 64, .source_length = view.source_length, .historical_repeat_length = 512, .future_repeat_length = 256, .primary_parent_manifest_sha256 = hex(binding.primary_parent_manifest_sha256), .source_freeze_sha256 = hex(binding.source_freeze_sha256), .binary_sha256 = hex(binding.binary_sha256), .exporter_sha256 = hex(binding.exporter_sha256), .authority_sha256 = hex(view.authority.digest()), .primary_sha256 = hex(view.authority.primary_sha256), .historical_sha256 = hex(view.authority.historical_sha256), .historical_study_tokens_sha256 = hex(view.historical_study_tokens_sha256), .future_sha256 = hex(view.authority.future_sha256), .heldout_sha256 = hex(view.authority.heldout_sha256), .historical_path = "historical", .historical_manifest_sha256 = hex(historical_result.manifest_sha256), .future_path = "future", .future_manifest_sha256 = hex(future_result.manifest_sha256) }, .{});
    defer allocator.free(body);
    var hash = Sha256.init(.{});
    hash.update(body);
    hash.update("\n");
    var dir = try std.Io.Dir.openDirAbsolute(io, binding.output_dir, .{ .follow_symlinks = false });
    defer dir.close(io);
    var file = try dir.createFile(io, "manifest.json.part", .{ .exclusive = true, .truncate = false });
    defer file.close(io);
    try file.writeStreamingAll(io, body);
    try file.writeStreamingAll(io, "\n");
    try file.sync(io);
    try std.Io.Dir.rename(dir, "manifest.json.part", dir, "manifest.json", io);
    committed = true;
    return .{ .manifest_sha256 = hash.finalResult(), .historical_manifest_sha256 = historical_result.manifest_sha256, .future_manifest_sha256 = future_result.manifest_sha256, .required_bytes = historical_result.required_bytes + future_result.required_bytes + body.len + 1 };
}

pub const CompleteReadBinding = struct {
    output_dir: []const u8,
    manifest_sha256: [32]u8,
    primary_parent_manifest_sha256: [32]u8,
    identity: checkpoint.Identity,
    source_freeze_sha256: [32]u8,
    binary_sha256: [32]u8,
    exporter_sha256: [32]u8,
    authority: future_training.Authority,
    historical_study_tokens_sha256: [32]u8,
    primary_batches: []const engine.ReferenceBatch,
};
pub const CompleteOwned = struct {
    historical: augmentation.Owned,
    future: augmentation.Owned,
    historical_released: bool = false,

    /// Releases the authenticated historical payload only after a caller has
    /// materialized and frozen its replacement. The future owner remains live.
    /// This is one-shot so error unwinding cannot double-free the payload.
    pub fn releaseHistorical(self: *@This()) !void {
        if (self.historical_released) return error.InvalidFutureAugmentation;
        self.historical.deinit();
        self.historical_released = true;
    }

    pub fn deinit(self: *@This()) void {
        if (!self.historical_released) self.historical.deinit();
        self.future.deinit();
        self.* = undefined;
    }
};

fn emptyOwnedForReleaseTest(allocator: std.mem.Allocator) !augmentation.Owned {
    return .{
        .allocator = allocator,
        .manifest_bytes = try allocator.alloc(u8, 0),
        .batches = try allocator.alloc(engine.ReferenceBatch, 0),
        .training = try allocator.alloc(f32, 0),
        .positions = try allocator.alloc(u64, 0),
        .consumer_layers = try allocator.alloc(u32, 0),
        .query_heads = try allocator.alloc(u32, 0),
        .counts = try allocator.alloc(u32, 0),
        .dimensions = try allocator.alloc(u32, 0),
        .source_length = 1,
        .repeat_length = 1,
    };
}

test "complete future owner releases obsolete historical exactly once" {
    var owned: CompleteOwned = .{ .historical = try emptyOwnedForReleaseTest(std.testing.allocator), .future = try emptyOwnedForReleaseTest(std.testing.allocator) };
    defer owned.deinit();
    try owned.releaseHistorical();
    try std.testing.expect(owned.historical_released);
    try std.testing.expectError(error.InvalidFutureAugmentation, owned.releaseHistorical());
}

fn validateHistoricalSelectedPositions(batches: []const engine.ReferenceBatch, source_length: usize) !void {
    if (batches.len == 0) return error.InvalidFutureAugmentation;
    const start: u64 = std.math.cast(u64, source_length) orelse return error.InvalidFutureAugmentation;
    const end = std.math.add(u64, start, 512) catch return error.InvalidFutureAugmentation;
    for (batches) |batch| {
        if (batch.train_positions.len != future_training.historical_rows) return error.InvalidFutureAugmentation;
        for (batch.train_positions, 0..) |position, row| {
            if (position < start or position >= end or (row != 0 and batch.train_positions[row - 1] >= position))
                return error.InvalidFutureAugmentation;
        }
    }
}

fn primaryDomainsDigest(batches: []const engine.ReferenceBatch, heldout: bool) ![32]u8 {
    var hash = Sha256.init(.{});
    if (batches.len == 0) return error.InvalidFutureAugmentation;
    for (batches) |batch| {
        if (batch.train_positions.len != future_training.primary_rows or batch.heldout_positions.len != future_training.heldout_rows or batch.train.len == 0 or batch.train.len % future_training.primary_rows != 0) return error.InvalidFutureAugmentation;
        if (heldout) {
            hash.update(std.mem.sliceAsBytes(batch.heldout));
            hash.update(std.mem.sliceAsBytes(batch.heldout_positions));
        } else {
            hash.update(std.mem.sliceAsBytes(batch.train));
            hash.update(std.mem.sliceAsBytes(batch.train_positions));
        }
    }
    return hash.finalResult();
}

pub fn readComplete(allocator: std.mem.Allocator, io: std.Io, binding: CompleteReadBinding) !CompleteOwned {
    var dir = try std.Io.Dir.openDirAbsolute(io, binding.output_dir, .{ .follow_symlinks = false });
    defer dir.close(io);
    const bytes = try dir.readFileAlloc(io, "manifest.json", allocator, .limited(64 * 1024));
    defer allocator.free(bytes);
    var actual: [32]u8 = undefined;
    Sha256.hash(bytes, &actual, .{});
    if (!std.mem.eql(u8, &actual, &binding.manifest_sha256)) return error.InvalidFutureAugmentation;
    const parsed = try std.json.parseFromSlice(CompleteManifest, allocator, bytes, .{ .ignore_unknown_fields = false });
    defer parsed.deinit();
    const m = parsed.value;
    if (!std.mem.eql(u8, m.schema, complete_schema) or !m.diagnostic_only or m.quality_credit or !m.training_only or m.primary_rows != 256 or m.historical_rows != 256 or m.future_rows != 256 or m.heldout_rows != 64 or m.source_length == 0 or m.historical_repeat_length != 512 or m.future_repeat_length != 256 or
        !std.mem.eql(u8, &m.primary_parent_manifest_sha256, &hex(binding.primary_parent_manifest_sha256)) or !std.mem.eql(u8, &m.source_freeze_sha256, &hex(binding.source_freeze_sha256)) or !std.mem.eql(u8, &m.binary_sha256, &hex(binding.binary_sha256)) or !std.mem.eql(u8, &m.exporter_sha256, &hex(binding.exporter_sha256)) or !std.mem.eql(u8, &m.authority_sha256, &hex(binding.authority.digest())) or
        !std.mem.eql(u8, &m.primary_sha256, &hex(binding.authority.primary_sha256)) or !std.mem.eql(u8, &m.historical_sha256, &hex(binding.authority.historical_sha256)) or !std.mem.eql(u8, &m.historical_study_tokens_sha256, &hex(binding.historical_study_tokens_sha256)) or !std.mem.eql(u8, &m.future_sha256, &hex(binding.authority.future_sha256)) or !std.mem.eql(u8, &m.heldout_sha256, &hex(binding.authority.heldout_sha256)) or !std.mem.eql(u8, m.historical_path, "historical") or !std.mem.eql(u8, m.future_path, "future")) return error.InvalidFutureAugmentation;
    const primary_digest = try primaryDomainsDigest(binding.primary_batches, false);
    const heldout_digest = try primaryDomainsDigest(binding.primary_batches, true);
    if (!std.mem.eql(u8, &primary_digest, &binding.authority.primary_sha256) or !std.mem.eql(u8, &heldout_digest, &binding.authority.heldout_sha256)) return error.InvalidFutureAugmentation;
    const historical_dir = try std.fs.path.join(allocator, &.{ binding.output_dir, "historical" });
    defer allocator.free(historical_dir);
    const future_dir = try std.fs.path.join(allocator, &.{ binding.output_dir, "future" });
    defer allocator.free(future_dir);
    var historical = try augmentation.read(allocator, io, .{ .output_dir = historical_dir, .parent_manifest_sha256 = binding.primary_parent_manifest_sha256, .identity = binding.identity, .source_freeze_sha256 = binding.source_freeze_sha256, .binary_sha256 = binding.binary_sha256, .exporter_sha256 = binding.exporter_sha256, .parent_study_tokens_sha256 = binding.authority.source_tokens_sha256, .study_tokens_sha256 = binding.historical_study_tokens_sha256, .samples_per_stratum = 256, .max_bytes = maximum_bytes / 2 });
    errdefer historical.deinit();
    if (historical.source_length != m.source_length or historical.repeat_length != m.historical_repeat_length) return error.InvalidFutureAugmentation;
    // The authenticated TRAIN policy selects 256 rows from the 512-position
    // historical domain by normalized-Q k-center. It is not uniform sampling.
    // Authority/domain digests bind exact bytes; this check binds role,
    // chronology, uniqueness, and the complete historical position domain.
    try validateHistoricalSelectedPositions(historical.batches, m.source_length);
    var future = try read(allocator, io, .{ .output_dir = future_dir, .manifest_sha256 = blk: {
        var nested: [32]u8 = undefined;
        _ = std.fmt.hexToBytes(&nested, &m.future_manifest_sha256) catch return error.InvalidFutureAugmentation;
        break :blk nested;
    }, .primary_parent_manifest_sha256 = binding.primary_parent_manifest_sha256, .identity = binding.identity, .source_freeze_sha256 = binding.source_freeze_sha256, .binary_sha256 = binding.binary_sha256, .exporter_sha256 = binding.exporter_sha256, .authority = binding.authority });
    errdefer future.deinit();
    if (future.source_length != m.source_length or future.repeat_length != m.future_repeat_length) return error.InvalidFutureAugmentation;
    const historical_digest = try futureDomainDigest(historical.batches);
    const future_digest = try futureDomainDigest(future.batches);
    const historical_manifest_path = try std.fs.path.join(allocator, &.{ historical_dir, "manifest.json" });
    defer allocator.free(historical_manifest_path);
    const historical_manifest = try std.Io.Dir.cwd().readFileAlloc(io, historical_manifest_path, allocator, .limited(64 * 1024));
    defer allocator.free(historical_manifest);
    var historical_manifest_digest: [32]u8 = undefined;
    Sha256.hash(historical_manifest, &historical_manifest_digest, .{});
    const future_manifest_path = try std.fs.path.join(allocator, &.{ future_dir, "manifest.json" });
    defer allocator.free(future_manifest_path);
    const future_manifest = try std.Io.Dir.cwd().readFileAlloc(io, future_manifest_path, allocator, .limited(64 * 1024));
    defer allocator.free(future_manifest);
    var future_manifest_digest: [32]u8 = undefined;
    Sha256.hash(future_manifest, &future_manifest_digest, .{});
    const historical_manifest_hex = hex(historical_manifest_digest);
    const future_manifest_hex = hex(future_manifest_digest);
    if (!std.mem.eql(u8, &historical_digest, &binding.authority.historical_sha256) or !std.mem.eql(u8, &future_digest, &binding.authority.future_sha256) or !std.mem.eql(u8, &m.historical_manifest_sha256, &historical_manifest_hex) or !std.mem.eql(u8, &m.future_manifest_sha256, &future_manifest_hex)) return error.InvalidFutureAugmentation;
    return .{ .historical = historical, .future = future };
}

test "future capture rejects relabel, mutation, incomplete frames, counts, and heldout drift" {
    var ids: [256]i64 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i + 17);
    var ids_hash = Sha256.init(.{});
    for (ids) |id| ids_hash.update(&std.mem.toBytes(std.mem.nativeToLittle(i64, id)));
    const ids_sha = ids_hash.finalResult();
    const identity: checkpoint.Identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    const boundary: self_study.SourceBoundary = .{ .identity = identity, .sequence_id = 5, .logical_tokens = 7936, .kv_revision = 11, .tokens_sha256 = @splat(6) };
    var source_hash = Sha256.init(.{});
    inline for (.{ identity.weights, identity.tokenizer, identity.template, identity.layout }) |value| source_hash.update(&value);
    source_hash.update(&std.mem.toBytes(std.mem.nativeToLittle(u64, boundary.sequence_id)));
    source_hash.update(&std.mem.toBytes(std.mem.nativeToLittle(u64, boundary.logical_tokens)));
    const authority: future_training.Authority = .{ .source_identity_sha256 = source_hash.finalResult(), .source_tokens_sha256 = boundary.tokens_sha256, .source_revision = boundary.kv_revision, .generation_contract_sha256 = @splat(7), .generated_ids_sha256 = ids_sha, .primary_sha256 = @splat(8), .historical_sha256 = @splat(9), .future_sha256 = @splat(10), .heldout_sha256 = @splat(11), .early_eos_observations = 2 };
    var capture: FutureTrainingCaptureEvidence = .{ .generated_ids = ids, .generated_ids_count = 256, .generated_ids_sha256 = ids_sha, .source_before = boundary, .source_after = boundary, .early_eos_observations = 2, .backend_owned_full_logits_calls = 256, .backend_owned_full_logits_ns = 1, .backend_owned_frame_successes = 256, .generic_last_logits_calls = 0, .backend_owned_full_logits_selected = true, .heldout_sha256_before_fit = authority.heldout_sha256 };
    try capture.validate(authority);
    capture.generated_ids[9] += 1;
    try std.testing.expectError(error.InvalidNativeQualityPublication, capture.validate(authority));
    capture.generated_ids = ids;
    capture.source_after.kv_revision += 1;
    try std.testing.expectError(error.InvalidNativeQualityPublication, capture.validate(authority));
    capture.source_after = boundary;
    capture.backend_owned_frame_successes = 255;
    try std.testing.expectError(error.InvalidNativeQualityPublication, capture.validate(authority));
    capture.backend_owned_frame_successes = 256;
    capture.future_rows = 255;
    try std.testing.expectError(error.InvalidNativeQualityPublication, capture.validate(authority));
    capture.future_rows = 256;
    capture.heldout_sha256_before_fit = @splat(12);
    try std.testing.expectError(error.InvalidNativeQualityPublication, capture.validate(authority));
    var relabeled = authority;
    relabeled.future_sha256 = relabeled.historical_sha256;
    capture.heldout_sha256_before_fit = authority.heldout_sha256;
    try std.testing.expectError(error.InvalidNativeQualityPublication, capture.validate(relabeled));
}

test "future nested payload digest binds values then positions and rejects substitution" {
    var values: [256]f32 = undefined;
    var positions: [256]u64 = undefined;
    for (&values, &positions, 0..) |*value, *position, i| {
        value.* = @floatFromInt(i);
        position.* = @intCast(7936 + i);
    }
    const batch: engine.ReferenceBatch = .{ .consumer_layer = 14, .query_head = 0, .train = &values, .heldout = &.{}, .train_positions = &positions, .heldout_positions = &.{} };
    const expected = try futureDomainDigest(&.{batch});
    var changed_values = values;
    changed_values[127] += 1;
    var changed = batch;
    changed.train = &changed_values;
    const changed_values_digest = try futureDomainDigest(&.{changed});
    try std.testing.expect(!std.mem.eql(u8, &expected, &changed_values_digest));
    changed = batch;
    var changed_positions = positions;
    changed_positions[127] += 1;
    changed.train_positions = &changed_positions;
    const changed_positions_digest = try futureDomainDigest(&.{changed});
    try std.testing.expect(!std.mem.eql(u8, &expected, &changed_positions_digest));
}

test "context complete future mix binds primary historical future and heldout bytes independently" {
    var train: [768]f32 = undefined;
    var train_positions: [768]u64 = undefined;
    var heldout: [64]f32 = undefined;
    var heldout_positions: [64]u64 = undefined;
    for (&train, &train_positions, 0..) |*value, *position, i| {
        value.* = @floatFromInt(i + 1);
        position.* = @intCast(i + 1);
    }
    for (&heldout, &heldout_positions, 0..) |*value, *position, i| {
        value.* = @floatFromInt(i + 900);
        position.* = @intCast(i + 900);
    }
    const batch: engine.ReferenceBatch = .{ .consumer_layer = 23, .query_head = 0, .train = &train, .heldout = &heldout, .train_positions = &train_positions, .heldout_positions = &heldout_positions, .independent_train_split = 256, .independent_train_second_split = 512 };
    const primary = try domainDigest(&.{batch}, .primary);
    const historical = try domainDigest(&.{batch}, .historical);
    const future = try domainDigest(&.{batch}, .future);
    const held = try domainDigest(&.{batch}, .heldout);
    try std.testing.expect(!std.mem.eql(u8, &primary, &historical));
    try std.testing.expect(!std.mem.eql(u8, &historical, &future));
    try std.testing.expect(!std.mem.eql(u8, &future, &held));
    var changed = train;
    changed[511] += 1;
    var mutated = batch;
    mutated.train = &changed;
    const changed_historical = try domainDigest(&.{mutated}, .historical);
    const unchanged_future = try domainDigest(&.{mutated}, .future);
    try std.testing.expect(!std.mem.eql(u8, &historical, &changed_historical));
    try std.testing.expect(std.mem.eql(u8, &future, &unchanged_future));
}

fn completeRoundTripFixture(a: std.mem.Allocator, io: std.Io, output_dir: []const u8, control: ?attention_match.Control, verify_corruption: bool) !void {
    errdefer std.Io.Dir.cwd().deleteTree(io, output_dir) catch {};
    var train: [768]f32 = undefined;
    var train_positions: [768]u64 = undefined;
    var heldout: [64]f32 = undefined;
    var heldout_positions: [64]u64 = undefined;
    for (&train, &train_positions, 0..) |*value, *position, i| {
        value.* = @floatFromInt(i + 1);
        position.* = if (i < 256)
            @intCast(i + 1)
        else if (i < 512)
            7936 + (i - 256) * 511 / 255
        else
            8448 + i - 512;
    }
    for (&heldout, &heldout_positions, 0..) |*value, *position, i| {
        value.* = @floatFromInt(i + 900);
        position.* = @intCast(i + 900);
    }
    const batch: engine.ReferenceBatch = .{ .consumer_layer = 23, .query_head = 0, .train = &train, .heldout = &heldout, .train_positions = &train_positions, .heldout_positions = &heldout_positions, .independent_train_split = 256, .independent_train_second_split = 512 };
    const primary_batch: engine.ReferenceBatch = .{ .consumer_layer = 23, .query_head = 0, .train = train[0..256], .heldout = &heldout, .train_positions = train_positions[0..256], .heldout_positions = &heldout_positions };
    const consumer: queries.Consumer = .{ .layer = 23, .donor = 23, .query_heads = 1, .kv_heads = 1, .dim = 1 };
    var ids: [256]i64 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i + 17);
    var ids_hash = Sha256.init(.{});
    for (ids) |id| ids_hash.update(&std.mem.toBytes(std.mem.nativeToLittle(i64, id)));
    const identity: checkpoint.Identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    const boundary: self_study.SourceBoundary = .{ .identity = identity, .sequence_id = 5, .logical_tokens = 7936, .kv_revision = 11, .tokens_sha256 = @splat(6) };
    var source_hash = Sha256.init(.{});
    inline for (.{ identity.weights, identity.tokenizer, identity.template, identity.layout }) |value| source_hash.update(&value);
    source_hash.update(&std.mem.toBytes(std.mem.nativeToLittle(u64, boundary.sequence_id)));
    source_hash.update(&std.mem.toBytes(std.mem.nativeToLittle(u64, boundary.logical_tokens)));
    const authority: future_training.Authority = .{ .source_identity_sha256 = source_hash.finalResult(), .source_tokens_sha256 = boundary.tokens_sha256, .source_revision = boundary.kv_revision, .generation_contract_sha256 = future_training.policyDigest(), .generated_ids_sha256 = ids_hash.finalResult(), .primary_sha256 = try domainDigest(&.{batch}, .primary), .historical_sha256 = try domainDigest(&.{batch}, .historical), .future_sha256 = try domainDigest(&.{batch}, .future), .heldout_sha256 = try domainDigest(&.{batch}, .heldout), .early_eos_observations = 2 };
    const capture: FutureTrainingCaptureEvidence = .{ .generated_ids = ids, .generated_ids_count = 256, .generated_ids_sha256 = authority.generated_ids_sha256, .source_before = boundary, .source_after = boundary, .early_eos_observations = 2, .backend_owned_full_logits_calls = 256, .backend_owned_full_logits_ns = 1, .backend_owned_frame_successes = 256, .generic_last_logits_calls = 0, .backend_owned_full_logits_selected = true, .heldout_sha256_before_fit = authority.heldout_sha256 };
    const binding: CompleteBinding = .{ .output_dir = output_dir, .primary_parent_manifest_sha256 = @splat(13), .identity = identity, .source_freeze_sha256 = @splat(14), .binary_sha256 = @splat(15), .exporter_sha256 = @splat(16) };
    const historical_study_tokens_sha256: [32]u8 = @splat(12);
    const written = try writeComplete(a, io, binding, .{ .capture = capture, .authority = authority, .batches = &.{batch}, .consumers = &.{consumer}, .source_length = 7936, .control = control, .historical_study_tokens_sha256 = historical_study_tokens_sha256 });
    var restored = try readComplete(a, io, .{ .output_dir = output_dir, .manifest_sha256 = written.manifest_sha256, .primary_parent_manifest_sha256 = binding.primary_parent_manifest_sha256, .identity = identity, .source_freeze_sha256 = binding.source_freeze_sha256, .binary_sha256 = binding.binary_sha256, .exporter_sha256 = binding.exporter_sha256, .authority = authority, .historical_study_tokens_sha256 = historical_study_tokens_sha256, .primary_batches = &.{primary_batch} });
    defer restored.deinit();
    try std.testing.expectEqualSlices(f32, train[256..512], restored.historical.batches[0].train);
    try std.testing.expectEqualSlices(f32, train[512..768], restored.future.batches[0].train);
    try std.testing.expectEqual(@as(usize, 512), restored.historical.repeat_length);
    if (verify_corruption) {
        var out_of_bounds_positions: [256]u64 = undefined;
        @memcpy(&out_of_bounds_positions, train_positions[256..512]);
        out_of_bounds_positions[255] += 1;
        var out_of_bounds_historical = restored.historical.batches[0];
        out_of_bounds_historical.train_positions = &out_of_bounds_positions;
        try std.testing.expectError(error.InvalidFutureAugmentation, validateHistoricalSelectedPositions(&.{out_of_bounds_historical}, 7936));
        var duplicate_positions = out_of_bounds_positions;
        @memcpy(&duplicate_positions, train_positions[256..512]);
        duplicate_positions[127] = duplicate_positions[126];
        out_of_bounds_historical.train_positions = &duplicate_positions;
        try std.testing.expectError(error.InvalidFutureAugmentation, validateHistoricalSelectedPositions(&.{out_of_bounds_historical}, 7936));
        var short_primary = primary_batch;
        short_primary.train_positions = primary_batch.train_positions[0..255];
        try std.testing.expectError(error.InvalidFutureAugmentation, readComplete(a, io, .{ .output_dir = output_dir, .manifest_sha256 = written.manifest_sha256, .primary_parent_manifest_sha256 = binding.primary_parent_manifest_sha256, .identity = identity, .source_freeze_sha256 = binding.source_freeze_sha256, .binary_sha256 = binding.binary_sha256, .exporter_sha256 = binding.exporter_sha256, .authority = authority, .historical_study_tokens_sha256 = historical_study_tokens_sha256, .primary_batches = &.{short_primary} }));
        var mutated_primary = primary_batch;
        var mutated_positions: [256]u64 = undefined;
        @memcpy(&mutated_positions, train_positions[0..256]);
        mutated_positions[255] = 99_999;
        mutated_primary.train_positions = &mutated_positions;
        try std.testing.expectError(error.InvalidFutureAugmentation, readComplete(a, io, .{ .output_dir = output_dir, .manifest_sha256 = written.manifest_sha256, .primary_parent_manifest_sha256 = binding.primary_parent_manifest_sha256, .identity = identity, .source_freeze_sha256 = binding.source_freeze_sha256, .binary_sha256 = binding.binary_sha256, .exporter_sha256 = binding.exporter_sha256, .authority = authority, .historical_study_tokens_sha256 = historical_study_tokens_sha256, .primary_batches = &.{mutated_primary} }));
    }
    std.Io.Dir.cwd().deleteTree(io, output_dir) catch {};
}

test "context complete future mix filesystem roundtrip allocation unwind and cancellation cleanup" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const outer_len = try tmp.dir.realPath(io, &path_buffer);
    const output_dir = try std.fs.path.join(a, &.{ path_buffer[0..outer_len], "complete" });
    defer a.free(output_dir);
    try completeRoundTripFixture(a, io, output_dir, null, true);
    try std.testing.checkAllAllocationFailures(a, completeRoundTripFixture, .{ io, output_dir, null, false });
    const cancelled_dir = try std.fs.path.join(a, &.{ path_buffer[0..outer_len], "cancelled" });
    defer a.free(cancelled_dir);
    const cancelled = attention_match.Control{ .check_fn = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    }.check };
    try std.testing.expectError(error.Cancelled, completeRoundTripFixture(a, io, cancelled_dir, cancelled, false));
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.openDirAbsolute(io, cancelled_dir, .{}));
}
