// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic-only provenance and policy for generated self-study queries.
//! The query payload itself is written by attention_match_training_augmentation;
//! this outer contract authenticates how its training-only rows were obtained.
const std = @import("std");

/// The fixed private-study loop uses this same selector at every step.
pub fn greedyToken(logits: []const f32, vocab_size: usize) !i64 {
    if (vocab_size == 0 or logits.len != vocab_size) return error.InvalidStudyLogits;
    var best: usize = 0;
    for (logits, 0..) |value, index| {
        if (!std.math.isFinite(value)) return error.NonFiniteStudyLogits;
        if (value > logits[best]) best = index;
    }
    return std.math.cast(i64, best) orelse error.InvalidStudyLogits;
}

const augmentation = @import("attention_match_training_augmentation.zig");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const queries = @import("attention_match_queries.zig");
const study_templates = @import("../../backends/context_study_templates.zig");

pub const schema = "antfly.attention-match-self-study-training.v2";
pub const maximum_bytes: usize = 128 * 1024 * 1024;
pub const fit_workspace_bytes: usize = 1024 * 1024 * 1024;
pub const generated_tokens: u32 = 256;

pub const SourceBoundary = struct {
    identity: checkpoint.Identity,
    sequence_id: u64,
    logical_tokens: usize,
    kv_revision: u64,
    tokens_sha256: [32]u8,
};

pub const Study = struct {
    kind: enum { fit, gate },
    prompt_tokens: []const i64,
    prompt_tokens_sha256: [32]u8,
    rendered_suffix_sha256: [32]u8,
    generation_contract_sha256: [32]u8,
    generated_ids: []const i64,
    generated_ids_sha256: [32]u8,
    complete_repeat_sha256: [32]u8,
};

pub const Binding = struct {
    output_dir: []const u8,
    /// Immutable primary whole-fit parent used by this exact capture.
    primary_parent_manifest_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    binary_sha256: [32]u8,
    exporter_sha256: [32]u8,
    max_bytes: usize = maximum_bytes,
    max_fit_workspace_bytes: usize = fit_workspace_bytes,
};

pub const View = struct {
    before: SourceBoundary,
    after: SourceBoundary,
    study: Study,
    batches: []const engine.ReferenceBatch,
    consumers: []const queries.Consumer,
};

pub const Plan = struct {
    repeat_tokens: usize,
    batch_count: usize,
    query_bytes: usize,
};

pub const Result = struct { manifest_sha256: [32]u8, query_bytes: usize };
const Payload = struct { name: []const u8, bytes: usize, count: usize, sha256: [32]u8 };

pub fn validatePreparedStudyContract(template_bytes: []const u8, generation_contract_bytes: []const u8) !void {
    if (!std.mem.eql(u8, template_bytes, study_templates.template(.fit_training_v1)) or
        !std.mem.eql(u8, generation_contract_bytes, study_templates.generation_contract))
        return error.InvalidSelfStudyBinding;
    try study_templates.validate(.fit_training_v1, study_templates.authority(.fit_training_v1));
}

pub fn hashTokens(tokens: []const i64) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (tokens) |token| {
        var encoded: [8]u8 = undefined;
        std.mem.writeInt(i64, &encoded, token, .little);
        h.update(&encoded);
    }
    return h.finalResult();
}

fn sameBoundary(a: SourceBoundary, b: SourceBoundary) bool {
    return std.meta.eql(a.identity, b.identity) and a.sequence_id == b.sequence_id and
        a.logical_tokens == b.logical_tokens and a.kv_revision == b.kv_revision and
        std.mem.eql(u8, &a.tokens_sha256, &b.tokens_sha256);
}

pub fn validateExplicitRepeatBinding(enabled: bool, representative: bool, study_pass: u8, has_hash: bool) !void {
    if (!enabled) {
        if (has_hash) return error.InvalidSelfStudyBinding;
        return;
    }
    if (representative or study_pass != 0 or !has_hash) return error.InvalidSelfStudyBinding;
}

/// This validates authorization, immutability and cardinality before any
/// self-study artifact is committed. It intentionally does not validate or
/// replace the immutable parent's heldout rows.
pub fn validate(binding: Binding, view: View) !Plan {
    if (binding.output_dir.len == 0 or !std.fs.path.isAbsolute(binding.output_dir) or
        binding.max_bytes == 0 or binding.max_bytes > maximum_bytes or
        binding.max_fit_workspace_bytes != fit_workspace_bytes)
        return error.InvalidSelfStudyBinding;
    if (view.study.kind != .fit) return error.SelfStudyGateForbidden;
    if (!sameBoundary(view.before, view.after)) return error.SelfStudySourceMutated;
    if (view.before.logical_tokens < 2 or view.study.prompt_tokens.len == 0 or
        view.study.generated_ids.len != generated_tokens or view.study.prompt_tokens.len + generated_tokens < 320 or view.batches.len == 0 or
        view.consumers.len == 0)
        return error.InvalidSelfStudyCapture;
    const prompt_hash = hashTokens(view.study.prompt_tokens);
    const generated_hash = hashTokens(view.study.generated_ids);
    if (!std.mem.eql(u8, &prompt_hash, &view.study.prompt_tokens_sha256) or
        !std.mem.eql(u8, &generated_hash, &view.study.generated_ids_sha256))
        return error.InvalidSelfStudyProvenance;
    var complete = std.crypto.hash.sha2.Sha256.init(.{});
    for (view.study.prompt_tokens) |token| {
        var encoded: [8]u8 = undefined;
        std.mem.writeInt(i64, &encoded, token, .little);
        complete.update(&encoded);
    }
    for (view.study.generated_ids) |token| {
        var encoded: [8]u8 = undefined;
        std.mem.writeInt(i64, &encoded, token, .little);
        complete.update(&encoded);
    }
    if (!std.mem.eql(u8, &complete.finalResult(), &view.study.complete_repeat_sha256))
        return error.InvalidSelfStudyProvenance;

    var expected_batches: usize = 0;
    for (view.consumers, 0..) |consumer, index| {
        if (consumer.query_heads == 0 or consumer.query_heads > 256 or consumer.kv_heads == 0 or
            consumer.query_heads % consumer.kv_heads != 0 or consumer.dim == 0 or consumer.dim > 4096)
            return error.InvalidSelfStudyCapture;
        for (view.consumers[0..index]) |prior| if (prior.layer == consumer.layer) return error.InvalidSelfStudyCapture;
        expected_batches = try std.math.add(usize, expected_batches, consumer.query_heads);
    }
    if (view.batches.len != expected_batches) return error.InvalidSelfStudyCapture;
    var query_bytes: usize = 0;
    for (view.batches, 0..) |batch, index| {
        for (view.batches[0..index]) |prior| if (prior.consumer_layer == batch.consumer_layer and prior.query_head == batch.query_head) return error.InvalidSelfStudyCapture;
        const consumer = found: {
            for (view.consumers) |candidate| if (candidate.layer == batch.consumer_layer) break :found candidate;
            return error.InvalidSelfStudyCapture;
        };
        if (batch.consumer_layer != consumer.layer or batch.query_head >= consumer.query_heads or
            batch.heldout.len == 0 or batch.heldout_positions.len == 0 or
            batch.train_positions.len == 0 or
            batch.train.len % batch.train_positions.len != 0 or
            batch.heldout.len % batch.heldout_positions.len != 0)
            return error.InvalidSelfStudyCapture;
        const dim = batch.train.len / batch.train_positions.len;
        if (dim != consumer.dim or batch.heldout.len / batch.heldout_positions.len != dim)
            return error.InvalidSelfStudyCapture;
        query_bytes = try std.math.add(usize, query_bytes, try std.math.mul(usize, batch.train.len + batch.heldout.len, @sizeOf(f32)));
        query_bytes = try std.math.add(usize, query_bytes, try std.math.mul(usize, batch.train_positions.len + batch.heldout_positions.len, @sizeOf(u64)));
    }
    if (query_bytes > binding.max_bytes) return error.SelfStudyOutputLimit;
    return .{
        .repeat_tokens = view.study.prompt_tokens.len + view.study.generated_ids.len,
        .batch_count = view.batches.len,
        .query_bytes = query_bytes,
    };
}

/// Reuses the existing loader/writer contract: train and heldout observations
/// from the generated response are merged chronologically into TRAIN only.
/// The immutable source65 heldout remains the sole authorization domain.
pub fn mergeTraining(allocator: std.mem.Allocator, binding: Binding, view: View) !augmentation.OwnedTraining {
    _ = try validate(binding, view);
    return augmentation.mergeCollectorBatches(allocator, view.batches);
}

/// Commit the small outer provenance manifest after the existing nested
/// training augmentation has been committed and hashed. `output_dir` already
/// exists and contains only `training/`; manifest creation is exclusive.
pub fn writeProvenance(
    allocator: std.mem.Allocator,
    io: std.Io,
    binding: Binding,
    view: View,
    nested_manifest_sha256: [32]u8,
    nested_required_bytes: usize,
) !Result {
    const p = try validate(binding, view);
    var dir = try std.Io.Dir.openDirAbsolute(io, binding.output_dir, .{ .follow_symlinks = false });
    defer dir.close(io);
    var complete = try allocator.alloc(i64, view.study.prompt_tokens.len + view.study.generated_ids.len);
    defer allocator.free(complete);
    @memcpy(complete[0..view.study.prompt_tokens.len], view.study.prompt_tokens);
    @memcpy(complete[view.study.prompt_tokens.len..], view.study.generated_ids);
    const payload_values = .{ view.study.prompt_tokens, view.study.generated_ids, @as([]const i64, complete) };
    const payload_names = .{ "study.prompt_tokens.i64", "study.generated_ids.i64", "study.complete_repeat.i64" };
    var payloads: [3]Payload = undefined;
    inline for (payload_names, payload_values, 0..) |name, values, index| {
        const bytes = std.mem.sliceAsBytes(values);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
        var file = try dir.createFile(io, name, .{ .exclusive = true, .truncate = false });
        defer file.close(io);
        try file.writeStreamingAll(io, bytes);
        try file.sync(io);
        payloads[index] = .{ .name = name, .bytes = bytes.len, .count = values.len, .sha256 = digest };
    }
    const body = try std.json.Stringify.valueAlloc(allocator, .{
        .schema = schema,
        .diagnostic_only = true,
        .quality_credit = false,
        .publication_claim = false,
        .training_only = true,
        .heldout_present = false,
        .study_kind = "fit",
        .template_variant = "fit_training_v1",
        .template_sha256 = std.fmt.bytesToHex(study_templates.authority(.fit_training_v1).template_sha256, .lower),
        .generated_tokens = generated_tokens,
        .maximum_bytes = binding.max_bytes,
        .max_fit_workspace_bytes = binding.max_fit_workspace_bytes,
        .primary_parent_manifest_sha256 = std.fmt.bytesToHex(binding.primary_parent_manifest_sha256, .lower),
        .source_freeze_sha256 = std.fmt.bytesToHex(binding.source_freeze_sha256, .lower),
        .binary_sha256 = std.fmt.bytesToHex(binding.binary_sha256, .lower),
        .exporter_sha256 = std.fmt.bytesToHex(binding.exporter_sha256, .lower),
        .prompt_tokens_sha256 = std.fmt.bytesToHex(view.study.prompt_tokens_sha256, .lower),
        .rendered_suffix_sha256 = std.fmt.bytesToHex(view.study.rendered_suffix_sha256, .lower),
        .generation_contract_sha256 = std.fmt.bytesToHex(view.study.generation_contract_sha256, .lower),
        .generated_ids_sha256 = std.fmt.bytesToHex(view.study.generated_ids_sha256, .lower),
        .complete_repeat_sha256 = std.fmt.bytesToHex(view.study.complete_repeat_sha256, .lower),
        .original_kv_before = view.before,
        .original_kv_after = view.after,
        .repeat_tokens = p.repeat_tokens,
        .batch_count = p.batch_count,
        .query_bytes = p.query_bytes,
        .nested_training_path = "training",
        .nested_training_manifest_sha256 = std.fmt.bytesToHex(nested_manifest_sha256, .lower),
        .payloads = payloads,
    }, .{});
    defer allocator.free(body);
    const payload_bytes = std.math.mul(usize, complete.len + view.study.prompt_tokens.len + view.study.generated_ids.len, @sizeOf(i64)) catch return error.SelfStudyOutputLimit;
    const total = std.math.add(usize, nested_required_bytes, body.len + 1) catch return error.SelfStudyOutputLimit;
    if (body.len + 1 > 64 * 1024 or
        (std.math.add(usize, total, payload_bytes) catch return error.SelfStudyOutputLimit) > binding.max_bytes)
        return error.SelfStudyOutputLimit;
    var digest = std.crypto.hash.sha2.Sha256.init(.{});
    digest.update(body);
    digest.update("\n");
    var file = try dir.createFile(io, "manifest.json.part", .{ .exclusive = true, .truncate = false });
    defer file.close(io);
    try file.writeStreamingAll(io, body);
    try file.writeStreamingAll(io, "\n");
    try file.sync(io);
    try std.Io.Dir.rename(dir, "manifest.json.part", dir, "manifest.json", io);
    return .{ .manifest_sha256 = digest.finalResult(), .query_bytes = p.query_bytes };
}

test "self-study policy constants retain bounded diagnostic contract" {
    try std.testing.expectEqual(@as(usize, 128 * 1024 * 1024), maximum_bytes);
    try std.testing.expectEqual(@as(usize, 1024 * 1024 * 1024), fit_workspace_bytes);
    try std.testing.expectEqual(@as(u32, 256), generated_tokens);
}

test "self-study rejects gate before observing payload rows" {
    const zero: [32]u8 = @splat(0);
    const boundary: SourceBoundary = .{ .identity = .{ .weights = zero, .tokenizer = zero, .template = zero, .layout = zero }, .sequence_id = 1, .logical_tokens = 2, .kv_revision = 3, .tokens_sha256 = zero };
    try std.testing.expectError(error.SelfStudyGateForbidden, validate(.{ .output_dir = "/exclusive", .primary_parent_manifest_sha256 = zero, .source_freeze_sha256 = zero, .binary_sha256 = zero, .exporter_sha256 = zero }, .{ .before = boundary, .after = boundary, .study = .{ .kind = .gate, .prompt_tokens = &.{1}, .prompt_tokens_sha256 = hashTokens(&.{1}), .rendered_suffix_sha256 = zero, .generation_contract_sha256 = zero, .generated_ids = &.{}, .generated_ids_sha256 = zero, .complete_repeat_sha256 = zero }, .batches = &.{}, .consumers = &.{} }));
}

test "self-study rejects original KV mutation before publication" {
    const zero: [32]u8 = @splat(0);
    const boundary: SourceBoundary = .{ .identity = .{ .weights = zero, .tokenizer = zero, .template = zero, .layout = zero }, .sequence_id = 1, .logical_tokens = 2, .kv_revision = 3, .tokens_sha256 = zero };
    var changed = boundary;
    changed.kv_revision += 1;
    try std.testing.expectError(error.SelfStudySourceMutated, validate(.{ .output_dir = "/exclusive", .primary_parent_manifest_sha256 = zero, .source_freeze_sha256 = zero, .binary_sha256 = zero, .exporter_sha256 = zero }, .{ .before = boundary, .after = changed, .study = .{ .kind = .fit, .prompt_tokens = &.{1}, .prompt_tokens_sha256 = hashTokens(&.{1}), .rendered_suffix_sha256 = zero, .generation_contract_sha256 = zero, .generated_ids = &.{}, .generated_ids_sha256 = zero, .complete_repeat_sha256 = zero }, .batches = &.{}, .consumers = &.{} }));
}

test "self-study explicit repeat is hash bound and mutually exclusive" {
    try validateExplicitRepeatBinding(true, false, 0, true);
    try validateExplicitRepeatBinding(false, false, 0, false);
    try std.testing.expectError(error.InvalidSelfStudyBinding, validateExplicitRepeatBinding(true, true, 0, true));
    try std.testing.expectError(error.InvalidSelfStudyBinding, validateExplicitRepeatBinding(true, false, 1, true));
    try std.testing.expectError(error.InvalidSelfStudyBinding, validateExplicitRepeatBinding(true, false, 0, false));
    try std.testing.expectError(error.InvalidSelfStudyBinding, validateExplicitRepeatBinding(false, false, 0, true));
}

test "self-study prepared contract accepts only versioned training authority" {
    try validatePreparedStudyContract(study_templates.template(.fit_training_v1), study_templates.generation_contract);
    try std.testing.expectError(error.InvalidSelfStudyBinding, validatePreparedStudyContract(study_templates.template(.fit), study_templates.generation_contract));
    try std.testing.expectError(error.InvalidSelfStudyBinding, validatePreparedStudyContract(study_templates.template(.gate), study_templates.generation_contract));
    try std.testing.expectError(error.InvalidSelfStudyBinding, validatePreparedStudyContract(study_templates.template(.fit_training_v1), "wrong generation contract"));
}

test "self-study production greedy selector is finite deterministic and vocabulary bound" {
    try std.testing.expectEqual(@as(i64, 1), try greedyToken(&.{ 0, 2, 2 }, 3));
    try std.testing.expectEqual(@as(i64, 0), try greedyToken(&.{ 0, 0 }, 2));
    try std.testing.expectError(error.InvalidStudyLogits, greedyToken(&.{}, 0));
    try std.testing.expectError(error.InvalidStudyLogits, greedyToken(&.{ 0, 2 }, 3));
    try std.testing.expectError(error.NonFiniteStudyLogits, greedyToken(&.{ 0, std.math.nan(f32) }, 2));
    try std.testing.expectError(error.NonFiniteStudyLogits, greedyToken(&.{ std.math.inf(f32), 0 }, 2));
}

test "self-study positive merge and provenance round trip authenticates emitted arrays" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &root_buffer);
    const output_dir = try std.fs.path.join(allocator, &.{ root_buffer[0..root_len], "self-study" });
    defer allocator.free(output_dir);
    const nested_dir = try std.fs.path.join(allocator, &.{ output_dir, "training" });
    defer allocator.free(nested_dir);
    try std.Io.Dir.createDirAbsolute(io, output_dir, std.Io.File.Permissions.fromMode(0o700));

    var train: [256 * 2]f32 = undefined;
    var heldout: [64 * 2]f32 = undefined;
    var train_positions: [256]u64 = undefined;
    var heldout_positions: [64]u64 = undefined;
    for (&train_positions, 0..) |*position, i| {
        position.* = @intCast(7936 + i + i / 4);
        train[i * 2] = @floatFromInt(i);
        train[i * 2 + 1] = -@as(f32, @floatFromInt(i));
    }
    for (&heldout_positions, 0..) |*position, i| {
        position.* = @intCast(7936 + i * 5 + 4);
        heldout[i * 2] = @floatFromInt(i + 256);
        heldout[i * 2 + 1] = -@as(f32, @floatFromInt(i + 256));
    }
    const base_batch: engine.ReferenceBatch = .{
        .consumer_layer = 5,
        .query_head = 0,
        .train = &train,
        .heldout = &heldout,
        .train_positions = &train_positions,
        .heldout_positions = &heldout_positions,
    };
    var batches = [_]engine.ReferenceBatch{ base_batch, base_batch };
    batches[1].query_head = 1;
    const consumers = [_]queries.Consumer{.{ .layer = 5, .donor = 0, .query_heads = 2, .kv_heads = 1, .dim = 2 }};
    var prompt: [64]i64 = undefined;
    for (&prompt, 0..) |*token, index| token.* = @intCast(7 + index);
    var generated: [generated_tokens]i64 = undefined;
    for (&generated, 0..) |*token, i| token.* = @intCast(1000 + i);
    const prompt_hash = hashTokens(&prompt);
    const generated_hash = hashTokens(&generated);
    var complete: [prompt.len + generated.len]i64 = undefined;
    @memcpy(complete[0..prompt.len], &prompt);
    @memcpy(complete[prompt.len..], &generated);
    const complete_hash = hashTokens(&complete);
    const zero: [32]u8 = @splat(0);
    const boundary: SourceBoundary = .{
        .identity = .{ .weights = zero, .tokenizer = zero, .template = zero, .layout = zero },
        .sequence_id = 9,
        .logical_tokens = 7936,
        .kv_revision = 17,
        .tokens_sha256 = @as([32]u8, @splat(1)),
    };
    const binding: Binding = .{
        .output_dir = output_dir,
        .primary_parent_manifest_sha256 = @as([32]u8, @splat(2)),
        .source_freeze_sha256 = @as([32]u8, @splat(3)),
        .binary_sha256 = @as([32]u8, @splat(4)),
        .exporter_sha256 = @as([32]u8, @splat(5)),
    };
    try std.testing.expect(!std.mem.eql(u8, &boundary.tokens_sha256, &prompt_hash));
    const view: View = .{
        .before = boundary,
        .after = boundary,
        .study = .{
            .kind = .fit,
            .prompt_tokens = &prompt,
            .prompt_tokens_sha256 = prompt_hash,
            .rendered_suffix_sha256 = @as([32]u8, @splat(6)),
            .generation_contract_sha256 = @as([32]u8, @splat(7)),
            .generated_ids = &generated,
            .generated_ids_sha256 = generated_hash,
            .complete_repeat_sha256 = complete_hash,
        },
        .batches = &batches,
        .consumers = &consumers,
    };
    var merged = try mergeTraining(allocator, binding, view);
    defer merged.deinit();
    try std.testing.expectEqual(@as(usize, 320), merged.batches[0].train_positions.len);
    try std.testing.expectEqual(@as(usize, 2), merged.batches.len);
    try std.testing.expectEqual(@as(usize, 320), merged.batches[1].train_positions.len);
    try std.testing.expectEqual(@as(u64, 7940), merged.batches[0].train_positions[4]);
    var bad_batches = batches;
    bad_batches[1].query_head = 0;
    var bad_shape = view;
    bad_shape.batches = &bad_batches;
    try std.testing.expectError(error.InvalidSelfStudyCapture, validate(binding, bad_shape));
    bad_shape.batches = batches[0..1];
    try std.testing.expectError(error.InvalidSelfStudyCapture, validate(binding, bad_shape));
    const nested = try augmentation.write(allocator, io, .{
        .output_dir = nested_dir,
        .parent_manifest_sha256 = @as([32]u8, @splat(8)),
        .identity = boundary.identity,
        .source_freeze_sha256 = binding.source_freeze_sha256,
        .binary_sha256 = binding.binary_sha256,
        .exporter_sha256 = binding.exporter_sha256,
        .parent_study_tokens_sha256 = @as([32]u8, @splat(9)),
        .study_tokens_sha256 = complete_hash,
    }, .{
        .batches = merged.batches,
        .consumers = &consumers,
        .parent_study_tokens_sha256 = @as([32]u8, @splat(9)),
        .study_tokens_sha256 = complete_hash,
        .source_length = 7936,
        .repeat_length = prompt.len + generated.len,
    });
    const outer = try writeProvenance(allocator, io, binding, view, nested.manifest_sha256, nested.required_bytes);
    try std.testing.expect(outer.query_bytes > 0);

    const manifest_path = try std.fs.path.join(allocator, &.{ output_dir, "manifest.json" });
    defer allocator.free(manifest_path);
    const manifest_bytes = try std.Io.Dir.cwd().readFileAlloc(io, manifest_path, allocator, .limited(64 * 1024));
    defer allocator.free(manifest_bytes);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, manifest_bytes, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings(schema, parsed.value.object.get("schema").?.string);
    try std.testing.expectEqual(@as(i64, generated_tokens), parsed.value.object.get("generated_tokens").?.integer);
    try std.testing.expectEqual(@as(usize, 3), parsed.value.object.get("payloads").?.array.items.len);
    const expected_hashes = [_][32]u8{ prompt_hash, generated_hash, complete_hash };
    for ([_][]const u8{ "study.prompt_tokens.i64", "study.generated_ids.i64", "study.complete_repeat.i64" }, expected_hashes) |name, expected_hash| {
        const path = try std.fs.path.join(allocator, &.{ output_dir, name });
        defer allocator.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024));
        defer allocator.free(bytes);
        var actual: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &actual, .{});
        try std.testing.expectEqualSlices(u8, &expected_hash, &actual);
    }

    var mutated = generated;
    mutated[17] += 1;
    var bad_view = view;
    bad_view.study.generated_ids = &mutated;
    try std.testing.expectError(error.InvalidSelfStudyProvenance, validate(binding, bad_view));
}
