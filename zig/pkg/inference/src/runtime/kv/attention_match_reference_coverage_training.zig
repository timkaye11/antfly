// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic provenance for source-only reference-coverage TRAIN capture.
//! This is deliberately distinct from generated self-study provenance.
const std = @import("std");
const augmentation = @import("attention_match_training_augmentation.zig");
const coverage = @import("attention_match_reference_coverage.zig");
const engine = @import("attention_match_engine.zig");
const queries = @import("attention_match_queries.zig");
const self_study = @import("attention_match_self_study_training.zig");
const Control = @import("attention_match.zig").Control;

pub const schema = "antfly.attention-match-reference-coverage-training.v1";
pub const maximum_bytes: usize = 128 * 1024 * 1024;
pub const fit_workspace_bytes: usize = 1024 * 1024 * 1024;

pub const SourceBoundary = self_study.SourceBoundary;
pub const TrainingPolicy = enum { coverage320, dense512 };

pub const Binding = struct {
    output_dir: []const u8,
    retained_primary_manifest_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    binary_sha256: [32]u8,
    exporter_sha256: [32]u8,
    max_bytes: usize = maximum_bytes,
    max_fit_workspace_bytes: usize = fit_workspace_bytes,
};

pub const View = struct {
    before: SourceBoundary,
    after: SourceBoundary,
    source_tokens: []const i64,
    selection: *const coverage.Selection,
    batches: []const engine.ReferenceBatch,
    consumers: []const queries.Consumer,
    policy: TrainingPolicy = .coverage320,
};

pub const Plan = struct { batch_count: usize, query_bytes: usize, selected_token_class_count: usize };
pub const Result = struct { manifest_sha256: [32]u8, query_bytes: usize };
const Payload = struct { name: []const u8, bytes: usize, count: usize, sha256: [32]u8 };

fn hashI64(values: []const i64) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    for (values) |value| {
        var encoded: [8]u8 = undefined;
        std.mem.writeInt(i64, &encoded, value, .little);
        hash.update(&encoded);
    }
    return hash.finalResult();
}

fn sameBoundary(a: SourceBoundary, b: SourceBoundary) bool {
    return std.meta.eql(a, b);
}

fn selectedClassCount(selection: *const coverage.Selection) usize {
    var count: usize = 0;
    for (selection.offsets, 0..) |offset, index| {
        const token = selection.repeat_tokens[offset];
        var seen = false;
        for (selection.offsets[0..index]) |prior_offset| if (selection.repeat_tokens[prior_offset] == token) {
            seen = true;
            break;
        };
        if (!seen) count += 1;
    }
    return count;
}

pub fn validate(allocator: std.mem.Allocator, binding: Binding, view: View, control: Control) !Plan {
    const source_digest = hashI64(view.source_tokens);
    if (binding.output_dir.len == 0 or !std.fs.path.isAbsolute(binding.output_dir) or binding.max_bytes == 0 or
        binding.max_bytes > maximum_bytes or binding.max_fit_workspace_bytes != fit_workspace_bytes or
        !sameBoundary(view.before, view.after) or view.before.logical_tokens != view.source_tokens.len or
        !std.mem.eql(u8, &view.before.tokens_sha256, &source_digest) or view.batches.len == 0 or view.consumers.len == 0)
        return error.InvalidReferenceCoverageBinding;
    switch (view.policy) {
        .coverage320 => {
            try coverage.verify(allocator, view.source_tokens, view.selection.reference_tokens, view.selection.repeat_tokens, view.selection.offsets, control);
            if (view.selection.offsets.len != coverage.sample_count or view.selection.positions.len != coverage.sample_count or
                !std.mem.eql(u8, &view.selection.sampler_policy_sha256, &coverage.policyDigest())) return error.InvalidReferenceCoverageBinding;
        },
        .dense512 => {
            try coverage.verifyDense512(allocator, view.source_tokens, view.selection.reference_tokens, view.selection.repeat_tokens, view.selection.offsets, control);
            if (view.selection.offsets.len != coverage.source_study_tokens or view.selection.positions.len != coverage.source_study_tokens or
                !std.mem.eql(u8, &view.selection.sampler_policy_sha256, &coverage.dense512PolicyDigest())) return error.InvalidReferenceCoverageBinding;
        },
    }
    if (view.selection.reference_tokens.len != coverage.source_study_tokens or view.selection.repeat_tokens.len != coverage.source_study_tokens)
        return error.InvalidReferenceCoverageBinding;
    const selected_classes = selectedClassCount(view.selection);
    if (selected_classes != view.selection.selected_token_class_count or
        view.selection.all_classes_covered != (view.selection.source_token_class_count == view.selection.selected_token_class_count))
        return error.ReferenceCoverageClassMismatch;

    var expected_batches: usize = 0;
    const expected_samples: usize = switch (view.policy) {
        .coverage320 => coverage.sample_count,
        .dense512 => coverage.dense512_sample_count,
    };
    for (view.consumers, 0..) |consumer, index| {
        if (consumer.query_heads == 0 or consumer.kv_heads == 0 or consumer.query_heads % consumer.kv_heads != 0 or consumer.dim == 0) return error.InvalidReferenceCoverageBinding;
        for (view.consumers[0..index]) |prior| if (prior.layer == consumer.layer) return error.InvalidReferenceCoverageBinding;
        expected_batches = try std.math.add(usize, expected_batches, consumer.query_heads);
    }
    if (expected_batches != view.batches.len) return error.InvalidReferenceCoverageBinding;
    var query_bytes: usize = 0;
    for (view.batches, 0..) |batch, index| {
        try control.check();
        for (view.batches[0..index]) |prior| if (prior.consumer_layer == batch.consumer_layer and prior.query_head == batch.query_head) return error.InvalidReferenceCoverageBinding;
        const consumer = found: {
            for (view.consumers) |candidate| if (candidate.layer == batch.consumer_layer) break :found candidate;
            return error.InvalidReferenceCoverageBinding;
        };
        if (batch.query_head >= consumer.query_heads or batch.train_positions.len + batch.heldout_positions.len != expected_samples or
            batch.train.len % @max(batch.train_positions.len, 1) != 0 or batch.heldout.len % @max(batch.heldout_positions.len, 1) != 0)
            return error.InvalidReferenceCoverageBinding;
        const dim = if (batch.train_positions.len != 0) batch.train.len / batch.train_positions.len else batch.heldout.len / batch.heldout_positions.len;
        if (dim != consumer.dim or (batch.heldout_positions.len != 0 and batch.heldout.len / batch.heldout_positions.len != dim)) return error.InvalidReferenceCoverageBinding;
        query_bytes = try std.math.add(usize, query_bytes, try std.math.mul(usize, batch.train.len + batch.heldout.len, @sizeOf(f32)));
        query_bytes = try std.math.add(usize, query_bytes, try std.math.mul(usize, batch.train_positions.len + batch.heldout_positions.len, @sizeOf(u64)));
    }
    if (query_bytes > binding.max_bytes) return error.ReferenceCoverageOutputLimit;
    return .{ .batch_count = view.batches.len, .query_bytes = query_bytes, .selected_token_class_count = selected_classes };
}

pub fn mergeTraining(allocator: std.mem.Allocator, binding: Binding, view: View, control: Control) !augmentation.OwnedTraining {
    _ = try validate(allocator, binding, view, control);
    return switch (view.policy) {
        .coverage320 => augmentation.mergeCollectorBatches(allocator, view.batches),
        .dense512 => augmentation.mergeReferenceCoverageDense512(allocator, view.batches),
    };
}

/// The only writer authorization for a 512-row nested augmentation.
pub fn writeNestedDense512(allocator: std.mem.Allocator, io: std.Io, binding: augmentation.Binding, view: augmentation.View) !augmentation.Result {
    if (binding.samples_per_stratum != coverage.source_study_tokens or view.samples_per_stratum != coverage.source_study_tokens)
        return error.InvalidReferenceCoverageBinding;
    return augmentation.writeReferenceCoverageDense512(allocator, io, binding, view);
}

pub fn writeProvenance(allocator: std.mem.Allocator, io: std.Io, binding: Binding, view: View, nested_manifest_sha256: [32]u8, nested_required_bytes: usize, control: Control) !Result {
    const plan = try validate(allocator, binding, view, control);
    var dir = try std.Io.Dir.openDirAbsolute(io, binding.output_dir, .{ .follow_symlinks = false });
    defer dir.close(io);
    const offsets_u64 = try allocator.alloc(u64, view.selection.offsets.len);
    defer allocator.free(offsets_u64);
    for (view.selection.offsets, offsets_u64) |offset, *out| out.* = offset;
    const payload_names = .{ "coverage.source_tokens.i64", "coverage.primary_reference_tokens.i64", "coverage.rotated_training_tokens.i64", "coverage.selected_offsets.u64" };
    const payload_bytes = .{ std.mem.sliceAsBytes(view.source_tokens), std.mem.sliceAsBytes(view.selection.reference_tokens), std.mem.sliceAsBytes(view.selection.repeat_tokens), std.mem.sliceAsBytes(offsets_u64) };
    const payload_counts = .{ view.source_tokens.len, view.selection.reference_tokens.len, view.selection.repeat_tokens.len, view.selection.offsets.len };
    var payloads: [4]Payload = undefined;
    inline for (payload_names, payload_bytes, payload_counts, 0..) |name, bytes, count, index| {
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
        var file = try dir.createFile(io, name, .{ .exclusive = true, .truncate = false });
        defer file.close(io);
        try file.writeStreamingAll(io, bytes);
        try file.sync(io);
        payloads[index] = .{ .name = name, .bytes = bytes.len, .count = count, .sha256 = digest };
    }
    const body = try std.json.Stringify.valueAlloc(allocator, .{
        .schema = schema,
        .diagnostic_only = true,
        .quality_credit = false,
        .publication_claim = false,
        .training_only = true,
        .heldout_present = false,
        .study_kind = switch (view.policy) {
            .coverage320 => coverage.policy,
            .dense512 => coverage.dense512_policy,
        },
        .sampler_policy_sha256 = std.fmt.bytesToHex(view.selection.sampler_policy_sha256, .lower),
        .rotation_tokens = coverage.rotation_tokens,
        .source_tokens_sha256 = std.fmt.bytesToHex(view.before.tokens_sha256, .lower),
        .primary_reference_tokens_sha256 = std.fmt.bytesToHex(view.selection.reference_sha256, .lower),
        .rotated_training_tokens_sha256 = std.fmt.bytesToHex(view.selection.repeat_sha256, .lower),
        .selected_offsets_sha256 = std.fmt.bytesToHex(view.selection.offsets_sha256, .lower),
        .selection_digest = std.fmt.bytesToHex(view.selection.digest, .lower),
        .selected_offset_count = view.selection.offsets.len,
        .source_token_class_count = view.selection.source_token_class_count,
        .selected_token_class_count = plan.selected_token_class_count,
        .all_classes_covered = view.selection.all_classes_covered,
        .maximum_bytes = binding.max_bytes,
        .max_fit_workspace_bytes = binding.max_fit_workspace_bytes,
        .retained_primary_manifest_sha256 = std.fmt.bytesToHex(binding.retained_primary_manifest_sha256, .lower),
        .source_freeze_sha256 = std.fmt.bytesToHex(binding.source_freeze_sha256, .lower),
        .binary_sha256 = std.fmt.bytesToHex(binding.binary_sha256, .lower),
        .exporter_sha256 = std.fmt.bytesToHex(binding.exporter_sha256, .lower),
        .source_before = view.before,
        .source_after = view.after,
        .batch_count = plan.batch_count,
        .query_bytes = plan.query_bytes,
        .nested_training_path = "training",
        .nested_training_manifest_sha256 = std.fmt.bytesToHex(nested_manifest_sha256, .lower),
        .payloads = payloads,
    }, .{});
    defer allocator.free(body);
    var total = std.math.add(usize, nested_required_bytes, body.len + 1) catch return error.ReferenceCoverageOutputLimit;
    for (payloads) |payload| total = std.math.add(usize, total, payload.bytes) catch return error.ReferenceCoverageOutputLimit;
    if (body.len + 1 > 64 * 1024 or total > binding.max_bytes) return error.ReferenceCoverageOutputLimit;
    var digest = std.crypto.hash.sha2.Sha256.init(.{});
    digest.update(body);
    digest.update("\n");
    var file = try dir.createFile(io, "manifest.json.part", .{ .exclusive = true, .truncate = false });
    defer file.close(io);
    try file.writeStreamingAll(io, body);
    try file.writeStreamingAll(io, "\n");
    try file.sync(io);
    try std.Io.Dir.rename(dir, "manifest.json.part", dir, "manifest.json", io);
    return .{ .manifest_sha256 = digest.finalResult(), .query_bytes = plan.query_bytes };
}

test "selected class evidence counts only classes represented by offsets" {
    var selection: coverage.Selection = .{
        .allocator = std.testing.allocator,
        .reference_tokens = @constCast(&[_]i64{}),
        .repeat_tokens = @constCast(&[_]i64{ 7, 8, 7, 9 }),
        .offsets = @constCast(&[_]u32{ 0, 1, 2 }),
        .positions = @constCast(&[_]u64{}),
        .source_token_class_count = 3,
        .selected_token_class_count = 2,
        .all_classes_covered = false,
        .reference_sha256 = @splat(0),
        .repeat_sha256 = @splat(0),
        .offsets_sha256 = @splat(0),
        .sampler_policy_sha256 = @splat(0),
        .digest = @splat(0),
    };
    _ = &selection;
    try std.testing.expectEqual(@as(usize, 2), selectedClassCount(&selection));
}
