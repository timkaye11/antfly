// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Strict diagnostic reader for independent study-pass-1 TRAIN portfolios.
//! `read` preserves the sealed 32-row contract; `readForSamples` admits only
//! explicit 32 or 128. The primary whole-fit input remains separately owned.
const std = @import("std");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const augmentation = @import("attention_match_training_augmentation.zig");
const study_templates = @import("../../backends/context_study_templates.zig");
const self_study = @import("attention_match_self_study_training.zig");
const reference_coverage = @import("attention_match_reference_coverage.zig");
const reference_training = @import("attention_match_reference_coverage_training.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const Binding = struct {
    output_dir: []const u8,
    parent_manifest_sha256: [32]u8,
    identity: checkpoint.Identity,
    source_freeze_sha256: [32]u8,
    binary_sha256: [32]u8,
    exporter_sha256: [32]u8,
    parent_study_tokens_sha256: [32]u8,
    study_tokens_sha256: [32]u8,
};

/// The legacy 320-row shape is accepted only through this stronger outer
/// provenance binding. It cannot be requested through `readForSamples`.
pub const SelfStudyBinding = struct {
    output_dir: []const u8,
    outer_manifest_sha256: [32]u8,
    retained_primary_manifest_sha256: [32]u8,
    fresh_primary_manifest_sha256: [32]u8,
    identity: checkpoint.Identity,
    source_freeze_sha256: [32]u8,
    binary_sha256: [32]u8,
    exporter_sha256: [32]u8,
    parent_study_tokens_sha256: [32]u8,
    study_tokens_sha256: [32]u8,
    rendered_suffix_sha256: [32]u8,
};

pub const ReferenceCoverageBinding = struct {
    output_dir: []const u8,
    outer_manifest_sha256: [32]u8,
    retained_primary_manifest_sha256: [32]u8,
    fresh_primary_manifest_sha256: [32]u8,
    identity: checkpoint.Identity,
    source_freeze_sha256: [32]u8,
    binary_sha256: [32]u8,
    exporter_sha256: [32]u8,
    parent_study_tokens_sha256: [32]u8,
    study_tokens_sha256: [32]u8,
};

const Boundary = struct {
    identity: checkpoint.Identity,
    sequence_id: u64,
    logical_tokens: usize,
    kv_revision: u64,
    tokens_sha256: [32]u8,
};
const StudyPayload = struct { name: []const u8, bytes: usize, count: usize, sha256: [32]u8 };
const SelfStudyManifest = struct {
    schema: []const u8,
    diagnostic_only: bool,
    quality_credit: bool,
    publication_claim: bool,
    training_only: bool,
    heldout_present: bool,
    study_kind: []const u8,
    template_variant: []const u8,
    template_sha256: [64]u8,
    generated_tokens: u32,
    maximum_bytes: usize,
    max_fit_workspace_bytes: usize,
    primary_parent_manifest_sha256: [64]u8,
    source_freeze_sha256: [64]u8,
    binary_sha256: [64]u8,
    exporter_sha256: [64]u8,
    prompt_tokens_sha256: [64]u8,
    rendered_suffix_sha256: [64]u8,
    generation_contract_sha256: [64]u8,
    generated_ids_sha256: [64]u8,
    complete_repeat_sha256: [64]u8,
    original_kv_before: Boundary,
    original_kv_after: Boundary,
    repeat_tokens: usize,
    batch_count: usize,
    query_bytes: usize,
    nested_training_path: []const u8,
    nested_training_manifest_sha256: [64]u8,
    payloads: []const StudyPayload,
};
const ReferenceCoverageManifest = struct {
    schema: []const u8,
    diagnostic_only: bool,
    quality_credit: bool,
    publication_claim: bool,
    training_only: bool,
    heldout_present: bool,
    study_kind: []const u8,
    sampler_policy_sha256: [64]u8,
    rotation_tokens: usize,
    source_tokens_sha256: [64]u8,
    primary_reference_tokens_sha256: [64]u8,
    rotated_training_tokens_sha256: [64]u8,
    selected_offsets_sha256: [64]u8,
    selection_digest: [64]u8,
    selected_offset_count: usize,
    source_token_class_count: usize,
    selected_token_class_count: usize,
    all_classes_covered: bool,
    maximum_bytes: usize,
    max_fit_workspace_bytes: usize,
    retained_primary_manifest_sha256: [64]u8,
    source_freeze_sha256: [64]u8,
    binary_sha256: [64]u8,
    exporter_sha256: [64]u8,
    source_before: Boundary,
    source_after: Boundary,
    batch_count: usize,
    query_bytes: usize,
    nested_training_path: []const u8,
    nested_training_manifest_sha256: [64]u8,
    payloads: []const StudyPayload,
};

fn hashFile(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, row: StudyPayload) ![]T {
    if (row.bytes != try std.math.mul(usize, row.count, @sizeOf(T))) return error.InvalidSelfStudyProvenance;
    const values = try allocator.alloc(T, row.count);
    errdefer allocator.free(values);
    var file = try dir.openFile(io, row.name, .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false });
    defer file.close(io);
    const bytes = std.mem.sliceAsBytes(values);
    if (try file.readPositionalAll(io, bytes, 0) != bytes.len) return error.InvalidSelfStudyProvenance;
    var extra: [1]u8 = undefined;
    if (try file.readPositionalAll(io, &extra, bytes.len) != 0) return error.InvalidSelfStudyProvenance;
    var digest: [32]u8 = undefined;
    Sha256.hash(bytes, &digest, .{});
    if (!std.mem.eql(u8, &digest, &row.sha256)) return error.InvalidSelfStudyProvenance;
    return values;
}

fn payload(rows: []const StudyPayload, name: []const u8) !StudyPayload {
    var found: ?StudyPayload = null;
    for (rows) |row| if (std.mem.eql(u8, row.name, name)) {
        if (found != null) return error.InvalidSelfStudyProvenance;
        found = row;
    };
    return found orelse error.InvalidSelfStudyProvenance;
}

fn sameBoundary(a: Boundary, b: Boundary, identity: checkpoint.Identity) bool {
    return std.meta.eql(a, b) and std.meta.eql(a.identity, identity) and a.logical_tokens > 0;
}

fn exactTemplateAuthority(encoded: *const [64]u8) bool {
    const expected = std.fmt.bytesToHex(study_templates.authority(.fit_training_v1).template_sha256, .lower);
    return std.mem.eql(u8, encoded, &expected);
}

fn hexDigest(text: []const u8) ![32]u8 {
    if (text.len != 64) return error.InvalidSelfStudyProvenance;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, text) catch return error.InvalidSelfStudyProvenance;
    return out;
}

pub fn readSelfStudy320(allocator: std.mem.Allocator, io: std.Io, binding: SelfStudyBinding) !augmentation.Owned {
    if (binding.output_dir.len == 0 or !std.fs.path.isAbsolute(binding.output_dir)) return error.InvalidSelfStudyBinding;
    var outer = try std.Io.Dir.openDirAbsolute(io, binding.output_dir, .{ .follow_symlinks = false });
    defer outer.close(io);
    const manifest_bytes = try outer.readFileAlloc(io, "manifest.json", allocator, .limited(64 * 1024));
    defer allocator.free(manifest_bytes);
    var manifest_digest: [32]u8 = undefined;
    Sha256.hash(manifest_bytes, &manifest_digest, .{});
    if (!std.mem.eql(u8, &manifest_digest, &binding.outer_manifest_sha256)) return error.InvalidSelfStudyBinding;
    const parsed = try std.json.parseFromSlice(SelfStudyManifest, allocator, manifest_bytes, .{ .ignore_unknown_fields = false });
    defer parsed.deinit();
    const m = parsed.value;
    const authority = study_templates.authority(.fit_training_v1);
    const retained_parent = std.fmt.bytesToHex(binding.retained_primary_manifest_sha256, .lower);
    const freeze = std.fmt.bytesToHex(binding.source_freeze_sha256, .lower);
    const binary = std.fmt.bytesToHex(binding.binary_sha256, .lower);
    const exporter = std.fmt.bytesToHex(binding.exporter_sha256, .lower);
    const suffix = std.fmt.bytesToHex(binding.rendered_suffix_sha256, .lower);
    const generation = std.fmt.bytesToHex(authority.generation_contract_sha256, .lower);
    const complete = std.fmt.bytesToHex(binding.study_tokens_sha256, .lower);
    if (!std.mem.eql(u8, m.schema, self_study.schema) or !m.diagnostic_only or m.quality_credit or
        m.publication_claim or !m.training_only or m.heldout_present or
        !std.mem.eql(u8, m.study_kind, "fit") or !std.mem.eql(u8, m.template_variant, "fit_training_v1") or
        !exactTemplateAuthority(&m.template_sha256) or
        m.generated_tokens != study_templates.response_tokens or m.maximum_bytes != self_study.maximum_bytes or
        m.max_fit_workspace_bytes != self_study.fit_workspace_bytes or m.payloads.len != 3 or
        !std.mem.eql(u8, &m.primary_parent_manifest_sha256, &retained_parent) or
        !std.mem.eql(u8, &m.source_freeze_sha256, &freeze) or !std.mem.eql(u8, &m.binary_sha256, &binary) or
        !std.mem.eql(u8, &m.exporter_sha256, &exporter) or !std.mem.eql(u8, &m.rendered_suffix_sha256, &suffix) or
        !std.mem.eql(u8, &m.generation_contract_sha256, &generation) or
        !std.mem.eql(u8, &m.complete_repeat_sha256, &complete) or
        !sameBoundary(m.original_kv_before, m.original_kv_after, binding.identity) or
        m.repeat_tokens < augmentation.legacy_samples_per_stratum or m.batch_count == 0 or m.query_bytes == 0 or
        !std.mem.eql(u8, m.nested_training_path, "training")) return error.InvalidSelfStudyBinding;

    const prompt_row = try payload(m.payloads, "study.prompt_tokens.i64");
    const generated_row = try payload(m.payloads, "study.generated_ids.i64");
    const repeat_row = try payload(m.payloads, "study.complete_repeat.i64");
    const prompt = try hashFile(i64, allocator, io, outer, prompt_row);
    defer allocator.free(prompt);
    const generated = try hashFile(i64, allocator, io, outer, generated_row);
    defer allocator.free(generated);
    const repeat = try hashFile(i64, allocator, io, outer, repeat_row);
    defer allocator.free(repeat);
    if (generated.len != study_templates.response_tokens or repeat.len != prompt.len + generated.len or
        repeat.len != m.repeat_tokens or !std.mem.eql(i64, repeat[0..prompt.len], prompt) or
        !std.mem.eql(i64, repeat[prompt.len..], generated) or
        !std.mem.eql(u8, &prompt_row.sha256, &try hexDigest(&m.prompt_tokens_sha256)) or
        !std.mem.eql(u8, &generated_row.sha256, &try hexDigest(&m.generated_ids_sha256)) or
        !std.mem.eql(u8, &repeat_row.sha256, &binding.study_tokens_sha256)) return error.InvalidSelfStudyProvenance;

    const nested_dir = try std.fs.path.join(allocator, &.{ binding.output_dir, "training" });
    defer allocator.free(nested_dir);
    var nested_manifest = try std.Io.Dir.openDirAbsolute(io, nested_dir, .{ .follow_symlinks = false });
    defer nested_manifest.close(io);
    const nested_bytes = try nested_manifest.readFileAlloc(io, "manifest.json", allocator, .limited(64 * 1024));
    defer allocator.free(nested_bytes);
    var nested_digest: [32]u8 = undefined;
    Sha256.hash(nested_bytes, &nested_digest, .{});
    const nested_hex = std.fmt.bytesToHex(nested_digest, .lower);
    if (!std.mem.eql(u8, &m.nested_training_manifest_sha256, &nested_hex)) return error.InvalidSelfStudyBinding;
    var owned = try augmentation.read(allocator, io, .{
        .output_dir = nested_dir,
        .parent_manifest_sha256 = binding.fresh_primary_manifest_sha256,
        .identity = binding.identity,
        .source_freeze_sha256 = binding.source_freeze_sha256,
        .binary_sha256 = binding.binary_sha256,
        .exporter_sha256 = binding.exporter_sha256,
        .parent_study_tokens_sha256 = binding.parent_study_tokens_sha256,
        .study_tokens_sha256 = binding.study_tokens_sha256,
        .samples_per_stratum = augmentation.legacy_samples_per_stratum,
        .max_bytes = self_study.maximum_bytes - 64 * 1024,
    });
    errdefer owned.deinit();
    if (owned.batches.len != m.batch_count or owned.repeat_length != m.repeat_tokens or owned.source_length != m.original_kv_before.logical_tokens)
        return error.InvalidSelfStudyBinding;
    for (owned.batches) |batch| {
        if (batch.train_positions.len != augmentation.legacy_samples_per_stratum or batch.heldout.len != 0 or batch.heldout_positions.len != 0)
            return error.InvalidSelfStudyBinding;
    }
    return owned;
}

fn readReferenceCoverage(allocator: std.mem.Allocator, io: std.Io, binding: ReferenceCoverageBinding, dense512: bool) !augmentation.Owned {
    if (binding.output_dir.len == 0 or !std.fs.path.isAbsolute(binding.output_dir)) return error.InvalidReferenceCoverageBinding;
    var outer = try std.Io.Dir.openDirAbsolute(io, binding.output_dir, .{ .follow_symlinks = false });
    defer outer.close(io);
    const manifest_bytes = try outer.readFileAlloc(io, "manifest.json", allocator, .limited(64 * 1024));
    defer allocator.free(manifest_bytes);
    var manifest_digest: [32]u8 = undefined;
    Sha256.hash(manifest_bytes, &manifest_digest, .{});
    if (!std.mem.eql(u8, &manifest_digest, &binding.outer_manifest_sha256)) return error.InvalidReferenceCoverageBinding;
    const parsed = try std.json.parseFromSlice(ReferenceCoverageManifest, allocator, manifest_bytes, .{ .ignore_unknown_fields = false });
    defer parsed.deinit();
    const m = parsed.value;
    const retained = std.fmt.bytesToHex(binding.retained_primary_manifest_sha256, .lower);
    const freeze = std.fmt.bytesToHex(binding.source_freeze_sha256, .lower);
    const binary = std.fmt.bytesToHex(binding.binary_sha256, .lower);
    const exporter = std.fmt.bytesToHex(binding.exporter_sha256, .lower);
    const policy_digest = std.fmt.bytesToHex(if (dense512) reference_coverage.dense512PolicyDigest() else reference_coverage.policyDigest(), .lower);
    const policy = if (dense512) reference_coverage.dense512_policy else reference_coverage.policy;
    const samples: u16 = if (dense512) augmentation.dense_reference_samples_per_stratum else augmentation.legacy_samples_per_stratum;
    if (!std.mem.eql(u8, m.schema, reference_training.schema) or !m.diagnostic_only or m.quality_credit or m.publication_claim or
        !m.training_only or m.heldout_present or !std.mem.eql(u8, m.study_kind, policy) or
        !std.mem.eql(u8, &m.sampler_policy_sha256, &policy_digest) or m.rotation_tokens != reference_coverage.rotation_tokens or
        m.selected_offset_count != samples or m.source_token_class_count == 0 or m.selected_token_class_count == 0 or
        m.all_classes_covered != (m.source_token_class_count == m.selected_token_class_count) or m.maximum_bytes != reference_training.maximum_bytes or
        m.max_fit_workspace_bytes != reference_training.fit_workspace_bytes or !std.mem.eql(u8, &m.retained_primary_manifest_sha256, &retained) or
        !std.mem.eql(u8, &m.source_freeze_sha256, &freeze) or !std.mem.eql(u8, &m.binary_sha256, &binary) or
        !std.mem.eql(u8, &m.exporter_sha256, &exporter) or !sameBoundary(m.source_before, m.source_after, binding.identity) or
        m.batch_count == 0 or m.query_bytes == 0 or !std.mem.eql(u8, m.nested_training_path, "training") or m.payloads.len != 4)
        return error.InvalidReferenceCoverageBinding;

    const source_row = try payload(m.payloads, "coverage.source_tokens.i64");
    const reference_row = try payload(m.payloads, "coverage.primary_reference_tokens.i64");
    const repeat_row = try payload(m.payloads, "coverage.rotated_training_tokens.i64");
    const offsets_row = try payload(m.payloads, "coverage.selected_offsets.u64");
    const source = try hashFile(i64, allocator, io, outer, source_row);
    defer allocator.free(source);
    const reference = try hashFile(i64, allocator, io, outer, reference_row);
    defer allocator.free(reference);
    const repeat = try hashFile(i64, allocator, io, outer, repeat_row);
    defer allocator.free(repeat);
    const offsets_u64 = try hashFile(u64, allocator, io, outer, offsets_row);
    defer allocator.free(offsets_u64);
    const offsets = try allocator.alloc(u32, offsets_u64.len);
    defer allocator.free(offsets);
    for (offsets_u64, offsets) |offset, *out| out.* = std.math.cast(u32, offset) orelse return error.InvalidReferenceCoverageProvenance;
    var expected = if (dense512) try reference_coverage.selectDense512(allocator, source, reference_coverage.source_study_tokens, .{}) else try reference_coverage.select(allocator, source, reference_coverage.source_study_tokens, reference_coverage.sample_count, .{});
    defer expected.deinit();
    if (!std.mem.eql(i64, reference, expected.reference_tokens) or !std.mem.eql(i64, repeat, expected.repeat_tokens) or
        !std.mem.eql(u32, offsets, expected.offsets) or expected.source_token_class_count != m.source_token_class_count or
        expected.selected_token_class_count != m.selected_token_class_count or expected.all_classes_covered != m.all_classes_covered or
        !std.mem.eql(u8, &source_row.sha256, &try hexDigest(&m.source_tokens_sha256)) or
        !std.mem.eql(u8, &expected.reference_sha256, &try hexDigest(&m.primary_reference_tokens_sha256)) or
        !std.mem.eql(u8, &expected.repeat_sha256, &try hexDigest(&m.rotated_training_tokens_sha256)) or
        !std.mem.eql(u8, &expected.offsets_sha256, &try hexDigest(&m.selected_offsets_sha256)) or
        !std.mem.eql(u8, &expected.digest, &try hexDigest(&m.selection_digest)) or source.len != m.source_before.logical_tokens)
        return error.InvalidReferenceCoverageProvenance;

    const nested_dir = try std.fs.path.join(allocator, &.{ binding.output_dir, "training" });
    defer allocator.free(nested_dir);
    var nested_manifest = try std.Io.Dir.openDirAbsolute(io, nested_dir, .{ .follow_symlinks = false });
    defer nested_manifest.close(io);
    const nested_bytes = try nested_manifest.readFileAlloc(io, "manifest.json", allocator, .limited(64 * 1024));
    defer allocator.free(nested_bytes);
    var nested_digest: [32]u8 = undefined;
    Sha256.hash(nested_bytes, &nested_digest, .{});
    const nested_hex = std.fmt.bytesToHex(nested_digest, .lower);
    if (!std.mem.eql(u8, &m.nested_training_manifest_sha256, &nested_hex)) return error.InvalidReferenceCoverageBinding;
    const nested_binding: augmentation.Binding = .{
        .output_dir = nested_dir,
        .parent_manifest_sha256 = binding.fresh_primary_manifest_sha256,
        .identity = binding.identity,
        .source_freeze_sha256 = binding.source_freeze_sha256,
        .binary_sha256 = binding.binary_sha256,
        .exporter_sha256 = binding.exporter_sha256,
        .parent_study_tokens_sha256 = binding.parent_study_tokens_sha256,
        .study_tokens_sha256 = binding.study_tokens_sha256,
        .samples_per_stratum = samples,
        .max_bytes = reference_training.maximum_bytes - 64 * 1024,
    };
    var owned = if (dense512) try augmentation.readReferenceCoverageDense512(allocator, io, nested_binding) else try augmentation.read(allocator, io, nested_binding);
    errdefer owned.deinit();
    if (owned.batches.len != m.batch_count or owned.repeat_length != reference_coverage.source_study_tokens or
        owned.source_length != m.source_before.logical_tokens) return error.InvalidReferenceCoverageBinding;
    for (owned.batches) |batch| if (batch.train_positions.len != samples or batch.heldout.len != 0 or batch.heldout_positions.len != 0)
        return error.InvalidReferenceCoverageBinding;
    return owned;
}

pub fn readReferenceCoverage320(allocator: std.mem.Allocator, io: std.Io, binding: ReferenceCoverageBinding) !augmentation.Owned {
    return readReferenceCoverage(allocator, io, binding, false);
}

pub fn readReferenceCoverage512(allocator: std.mem.Allocator, io: std.Io, binding: ReferenceCoverageBinding) !augmentation.Owned {
    return readReferenceCoverage(allocator, io, binding, true);
}

fn validateBatches(batches: []const engine.ReferenceBatch, expected_samples: u16) !void {
    if (expected_samples != augmentation.portfolio_samples_per_stratum and expected_samples != augmentation.extended_portfolio_samples_per_stratum and expected_samples != augmentation.independent_domain_samples_per_stratum) return error.InvalidTrainingAugmentation;
    const expected: usize = expected_samples;
    for (batches) |batch| {
        if (batch.train_positions.len != expected or
            batch.train.len == 0 or batch.train.len % expected != 0 or
            batch.heldout.len != 0 or batch.heldout_positions.len != 0)
            return error.InvalidTrainingAugmentation;
    }
}

pub fn read(allocator: std.mem.Allocator, io: std.Io, binding: Binding) !augmentation.Owned {
    return readForSamples(allocator, io, binding, augmentation.portfolio_samples_per_stratum);
}

pub fn readForSamples(allocator: std.mem.Allocator, io: std.Io, binding: Binding, expected_samples: u16) !augmentation.Owned {
    if (expected_samples != augmentation.portfolio_samples_per_stratum and expected_samples != augmentation.extended_portfolio_samples_per_stratum and expected_samples != augmentation.independent_domain_samples_per_stratum) return error.InvalidTrainingAugmentation;
    var owned = try augmentation.read(allocator, io, .{
        .output_dir = binding.output_dir,
        .parent_manifest_sha256 = binding.parent_manifest_sha256,
        .identity = binding.identity,
        .source_freeze_sha256 = binding.source_freeze_sha256,
        .binary_sha256 = binding.binary_sha256,
        .exporter_sha256 = binding.exporter_sha256,
        .parent_study_tokens_sha256 = binding.parent_study_tokens_sha256,
        .study_tokens_sha256 = binding.study_tokens_sha256,
        .samples_per_stratum = expected_samples,
    });
    errdefer owned.deinit();
    try validateBatches(owned.batches, expected_samples);
    return owned;
}

test "train32 reader accepts exact TRAIN boundary and rejects heldout or 31 rows" {
    var train: [32]f32 = @splat(1);
    var positions: [32]u64 = undefined;
    for (&positions, 0..) |*position, i| position.* = @intCast(i + 1);
    const valid = [_]engine.ReferenceBatch{.{
        .consumer_layer = 0,
        .query_head = 0,
        .train = &train,
        .heldout = &.{},
        .train_positions = &positions,
        .heldout_positions = &.{},
    }};
    try validateBatches(&valid, augmentation.portfolio_samples_per_stratum);

    var heldout_values = valid;
    heldout_values[0].heldout = &.{1};
    try std.testing.expectError(error.InvalidTrainingAugmentation, validateBatches(&heldout_values, augmentation.portfolio_samples_per_stratum));
    var heldout_positions = valid;
    heldout_positions[0].heldout_positions = &.{33};
    try std.testing.expectError(error.InvalidTrainingAugmentation, validateBatches(&heldout_positions, augmentation.portfolio_samples_per_stratum));
    var short = valid;
    short[0].train = train[0..31];
    short[0].train_positions = positions[0..31];
    try std.testing.expectError(error.InvalidTrainingAugmentation, validateBatches(&short, augmentation.portfolio_samples_per_stratum));
}

test "portfolio reader accepts exact 128 and rejects heldout wrong count and unsupported sample request" {
    var train: [128]f32 = @splat(1);
    var positions: [128]u64 = undefined;
    for (&positions, 0..) |*position, i| position.* = @intCast(i + 1);
    var batches = [_]engine.ReferenceBatch{.{ .consumer_layer = 0, .query_head = 0, .train = &train, .heldout = &.{}, .train_positions = &positions, .heldout_positions = &.{} }};
    try validateBatches(&batches, augmentation.extended_portfolio_samples_per_stratum);
    batches[0].heldout_positions = &.{129};
    try std.testing.expectError(error.InvalidTrainingAugmentation, validateBatches(&batches, augmentation.extended_portfolio_samples_per_stratum));
    batches[0].heldout_positions = &.{};
    batches[0].train = train[0..127];
    batches[0].train_positions = positions[0..127];
    try std.testing.expectError(error.InvalidTrainingAugmentation, validateBatches(&batches, augmentation.extended_portfolio_samples_per_stratum));
    try std.testing.expectError(error.InvalidTrainingAugmentation, validateBatches(&batches, 64));
    try std.testing.expectError(error.InvalidTrainingAugmentation, validateBatches(&batches, augmentation.legacy_samples_per_stratum));
}

const FixtureSplit = struct {
    allocator: std.mem.Allocator,
    train: []f32,
    heldout: []f32,
    train_positions: []u64,
    heldout_positions: []u64,
    fn deinit(self: *FixtureSplit) void {
        self.allocator.free(self.train);
        self.allocator.free(self.heldout);
        self.allocator.free(self.train_positions);
        self.allocator.free(self.heldout_positions);
        self.* = undefined;
    }
};

fn splitFixture(allocator: std.mem.Allocator, values: []const f32, positions: []const u64, dim: usize) !FixtureSplit {
    const expected_values = std.math.mul(usize, positions.len, dim) catch return error.InvalidTrainingAugmentation;
    if (dim == 0 or positions.len == 0 or values.len != expected_values) return error.InvalidTrainingAugmentation;
    const heldout_rows = (positions.len - 1) / 5 + 1;
    const train_rows = positions.len - heldout_rows;
    const train = try allocator.alloc(f32, train_rows * dim);
    errdefer allocator.free(train);
    const heldout = try allocator.alloc(f32, heldout_rows * dim);
    errdefer allocator.free(heldout);
    const train_positions = try allocator.alloc(u64, train_rows);
    errdefer allocator.free(train_positions);
    const heldout_positions = try allocator.alloc(u64, heldout_rows);
    errdefer allocator.free(heldout_positions);
    var ti: usize = 0;
    var hi: usize = 0;
    for (positions, 0..) |position, index| {
        if (index % 5 == 0) {
            heldout_positions[hi] = position;
            @memcpy(heldout[hi * dim ..][0..dim], values[index * dim ..][0..dim]);
            hi += 1;
        } else {
            train_positions[ti] = position;
            @memcpy(train[ti * dim ..][0..dim], values[index * dim ..][0..dim]);
            ti += 1;
        }
    }
    if (ti != train_rows or hi != heldout_rows) return error.InvalidTrainingAugmentation;
    return .{ .allocator = allocator, .train = train, .heldout = heldout, .train_positions = train_positions, .heldout_positions = heldout_positions };
}

test "reference coverage filesystem fixture split rejects empty input" {
    try std.testing.expectError(error.InvalidTrainingAugmentation, splitFixture(std.testing.allocator, &.{}, &.{}, 2));
}

test "reference coverage writer reader filesystem round trip binds parents descriptors classes and digests" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &root_buffer);
    const output_dir = try std.fs.path.join(a, &.{ root_buffer[0..root_len], "coverage" });
    defer a.free(output_dir);
    const nested_dir = try std.fs.path.join(a, &.{ output_dir, "training" });
    defer a.free(nested_dir);
    try std.Io.Dir.createDirAbsolute(io, output_dir, std.Io.File.Permissions.fromMode(0o700));

    var source: [768]i64 = undefined;
    for (&source, 0..) |*token, index| token.* = @intCast(index % 266);
    var selection = try reference_coverage.select(a, &source, 512, 320, .{});
    defer selection.deinit();
    var values: [320 * 2]f32 = undefined;
    for (&values, 0..) |*value, index| value.* = @floatFromInt(index % 97);
    var split = try splitFixture(a, &values, selection.positions, 2);
    defer split.deinit();
    const batch: engine.ReferenceBatch = .{ .consumer_layer = 3, .query_head = 0, .train = split.train, .heldout = split.heldout, .train_positions = split.train_positions, .heldout_positions = split.heldout_positions };
    const consumers = [_]@import("attention_match_queries.zig").Consumer{.{ .layer = 3, .donor = 0, .query_heads = 1, .kv_heads = 1, .dim = 2 }};
    const identity: checkpoint.Identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    var source_hash = std.crypto.hash.sha2.Sha256.init(.{});
    for (source) |token| {
        var encoded: [8]u8 = undefined;
        std.mem.writeInt(i64, &encoded, token, .little);
        source_hash.update(&encoded);
    }
    const boundary: reference_training.SourceBoundary = .{ .identity = identity, .sequence_id = 9, .logical_tokens = source.len, .kv_revision = 17, .tokens_sha256 = source_hash.finalResult() };
    const retained_parent: [32]u8 = @splat(5);
    const fresh_parent: [32]u8 = @splat(6);
    const freeze: [32]u8 = @splat(7);
    const binary: [32]u8 = @splat(8);
    const exporter: [32]u8 = @splat(9);
    const parent_tokens: [32]u8 = @splat(10);
    const view: reference_training.View = .{ .before = boundary, .after = boundary, .source_tokens = &source, .selection = &selection, .batches = &.{batch}, .consumers = &consumers };
    const writer_binding: reference_training.Binding = .{ .output_dir = output_dir, .retained_primary_manifest_sha256 = retained_parent, .source_freeze_sha256 = freeze, .binary_sha256 = binary, .exporter_sha256 = exporter };
    var merged = try reference_training.mergeTraining(a, writer_binding, view, .{});
    defer merged.deinit();
    const nested = try augmentation.write(a, io, .{ .output_dir = nested_dir, .parent_manifest_sha256 = fresh_parent, .identity = identity, .source_freeze_sha256 = freeze, .binary_sha256 = binary, .exporter_sha256 = exporter, .parent_study_tokens_sha256 = parent_tokens, .study_tokens_sha256 = selection.repeat_sha256, .samples_per_stratum = 320, .max_bytes = reference_training.maximum_bytes - 64 * 1024 }, .{ .batches = merged.batches, .consumers = &consumers, .source_length = source.len, .repeat_length = 512, .samples_per_stratum = 320, .parent_study_tokens_sha256 = parent_tokens, .study_tokens_sha256 = selection.repeat_sha256 });
    const outer = try reference_training.writeProvenance(a, io, writer_binding, view, nested.manifest_sha256, nested.required_bytes, .{});
    var binding: ReferenceCoverageBinding = .{ .output_dir = output_dir, .outer_manifest_sha256 = outer.manifest_sha256, .retained_primary_manifest_sha256 = retained_parent, .fresh_primary_manifest_sha256 = fresh_parent, .identity = identity, .source_freeze_sha256 = freeze, .binary_sha256 = binary, .exporter_sha256 = exporter, .parent_study_tokens_sha256 = parent_tokens, .study_tokens_sha256 = selection.repeat_sha256 };
    var restored = try readReferenceCoverage320(a, io, binding);
    defer restored.deinit();
    try std.testing.expectEqual(@as(usize, 1), restored.batches.len);
    try std.testing.expectEqualSlices(f32, &values, restored.batches[0].train);
    try std.testing.expectEqualSlices(u64, selection.positions, restored.batches[0].train_positions);
    var wrong_parent = binding;
    wrong_parent.fresh_primary_manifest_sha256 = @splat(11);
    try std.testing.expectError(error.InvalidTrainingAugmentationBinding, readReferenceCoverage320(a, io, wrong_parent));

    var dir = try std.Io.Dir.openDirAbsolute(io, output_dir, .{ .follow_symlinks = false });
    defer dir.close(io);
    const original_manifest = try dir.readFileAlloc(io, "manifest.json", a, .limited(64 * 1024));
    defer a.free(original_manifest);
    const class_marker = "\"source_token_class_count\":266";
    const class_at = std.mem.indexOf(u8, original_manifest, class_marker) orelse return error.TestUnexpectedResult;
    const mutated_manifest = try a.dupe(u8, original_manifest);
    defer a.free(mutated_manifest);
    mutated_manifest[class_at + class_marker.len - 1] = '7';
    var manifest_file = try dir.openFile(io, "manifest.json", .{ .mode = .read_write, .allow_directory = false, .follow_symlinks = false });
    defer manifest_file.close(io);
    try manifest_file.writePositionalAll(io, mutated_manifest, 0);
    try manifest_file.sync(io);
    std.crypto.hash.sha2.Sha256.hash(mutated_manifest, &binding.outer_manifest_sha256, .{});
    try std.testing.expectError(error.InvalidReferenceCoverageBinding, readReferenceCoverage320(a, io, binding));
    try manifest_file.writePositionalAll(io, original_manifest, 0);
    try manifest_file.sync(io);

    const digest_hex = std.fmt.bytesToHex(selection.digest, .lower);
    const digest_at = std.mem.indexOf(u8, original_manifest, &digest_hex) orelse return error.TestUnexpectedResult;
    @memcpy(mutated_manifest, original_manifest);
    mutated_manifest[digest_at] = if (mutated_manifest[digest_at] == '0') '1' else '0';
    try manifest_file.writePositionalAll(io, mutated_manifest, 0);
    try manifest_file.sync(io);
    std.crypto.hash.sha2.Sha256.hash(mutated_manifest, &binding.outer_manifest_sha256, .{});
    try std.testing.expectError(error.InvalidReferenceCoverageProvenance, readReferenceCoverage320(a, io, binding));
    try manifest_file.writePositionalAll(io, original_manifest, 0);
    try manifest_file.sync(io);
    std.crypto.hash.sha2.Sha256.hash(original_manifest, &binding.outer_manifest_sha256, .{});

    var offsets_file = try dir.openFile(io, "coverage.selected_offsets.u64", .{ .mode = .read_write, .allow_directory = false, .follow_symlinks = false });
    defer offsets_file.close(io);
    var first: [1]u8 = undefined;
    _ = try offsets_file.readPositionalAll(io, &first, 0);
    first[0] +%= 1;
    try offsets_file.writePositionalAll(io, &first, 0);
    try offsets_file.sync(io);
    try std.testing.expectError(error.InvalidSelfStudyProvenance, readReferenceCoverage320(a, io, binding));
}

test "self-study reader binds exact compiled training template digest" {
    var encoded = std.fmt.bytesToHex(study_templates.authority(.fit_training_v1).template_sha256, .lower);
    try std.testing.expect(exactTemplateAuthority(&encoded));
    encoded[0] = if (encoded[0] == '0') '1' else '0';
    try std.testing.expect(!exactTemplateAuthority(&encoded));
}
test "reference coverage dense512 writer reader filesystem round trip binds parents descriptors classes and digests" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &root_buffer);
    const output_dir = try std.fs.path.join(a, &.{ root_buffer[0..root_len], "coverage" });
    defer a.free(output_dir);
    const nested_dir = try std.fs.path.join(a, &.{ output_dir, "training" });
    defer a.free(nested_dir);
    try std.Io.Dir.createDirAbsolute(io, output_dir, std.Io.File.Permissions.fromMode(0o700));

    var source: [768]i64 = undefined;
    for (&source, 0..) |*token, index| token.* = @intCast(index % 266);
    var selection = try reference_coverage.selectDense512(a, &source, 512, .{});
    defer selection.deinit();
    var values: [512 * 2]f32 = undefined;
    for (&values, 0..) |*value, index| value.* = @floatFromInt(index % 97);
    var split = try splitFixture(a, &values, selection.positions, 2);
    defer split.deinit();
    const batch: engine.ReferenceBatch = .{ .consumer_layer = 3, .query_head = 0, .train = split.train, .heldout = split.heldout, .train_positions = split.train_positions, .heldout_positions = split.heldout_positions };
    const consumers = [_]@import("attention_match_queries.zig").Consumer{.{ .layer = 3, .donor = 0, .query_heads = 1, .kv_heads = 1, .dim = 2 }};
    const identity: checkpoint.Identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    var source_hash = std.crypto.hash.sha2.Sha256.init(.{});
    for (source) |token| {
        var encoded: [8]u8 = undefined;
        std.mem.writeInt(i64, &encoded, token, .little);
        source_hash.update(&encoded);
    }
    const boundary: reference_training.SourceBoundary = .{ .identity = identity, .sequence_id = 9, .logical_tokens = source.len, .kv_revision = 17, .tokens_sha256 = source_hash.finalResult() };
    const retained_parent: [32]u8 = @splat(5);
    const fresh_parent: [32]u8 = @splat(6);
    const freeze: [32]u8 = @splat(7);
    const binary: [32]u8 = @splat(8);
    const exporter: [32]u8 = @splat(9);
    const parent_tokens: [32]u8 = @splat(10);
    const view: reference_training.View = .{ .before = boundary, .after = boundary, .source_tokens = &source, .selection = &selection, .batches = &.{batch}, .consumers = &consumers, .policy = .dense512 };
    const writer_binding: reference_training.Binding = .{ .output_dir = output_dir, .retained_primary_manifest_sha256 = retained_parent, .source_freeze_sha256 = freeze, .binary_sha256 = binary, .exporter_sha256 = exporter };
    var merged = try reference_training.mergeTraining(a, writer_binding, view, .{});
    defer merged.deinit();
    const nested = try reference_training.writeNestedDense512(a, io, .{ .output_dir = nested_dir, .parent_manifest_sha256 = fresh_parent, .identity = identity, .source_freeze_sha256 = freeze, .binary_sha256 = binary, .exporter_sha256 = exporter, .parent_study_tokens_sha256 = parent_tokens, .study_tokens_sha256 = selection.repeat_sha256, .samples_per_stratum = 512, .max_bytes = reference_training.maximum_bytes - 64 * 1024 }, .{ .batches = merged.batches, .consumers = &consumers, .source_length = source.len, .repeat_length = 512, .samples_per_stratum = 512, .parent_study_tokens_sha256 = parent_tokens, .study_tokens_sha256 = selection.repeat_sha256 });
    const outer = try reference_training.writeProvenance(a, io, writer_binding, view, nested.manifest_sha256, nested.required_bytes, .{});
    var binding: ReferenceCoverageBinding = .{ .output_dir = output_dir, .outer_manifest_sha256 = outer.manifest_sha256, .retained_primary_manifest_sha256 = retained_parent, .fresh_primary_manifest_sha256 = fresh_parent, .identity = identity, .source_freeze_sha256 = freeze, .binary_sha256 = binary, .exporter_sha256 = exporter, .parent_study_tokens_sha256 = parent_tokens, .study_tokens_sha256 = selection.repeat_sha256 };
    var restored = try readReferenceCoverage512(a, io, binding);
    defer restored.deinit();
    try std.testing.expectEqual(@as(usize, 1), restored.batches.len);
    try std.testing.expectEqualSlices(f32, &values, restored.batches[0].train);
    try std.testing.expectEqualSlices(u64, selection.positions, restored.batches[0].train_positions);
    var wrong_parent = binding;
    wrong_parent.fresh_primary_manifest_sha256 = @splat(11);
    try std.testing.expectError(error.InvalidTrainingAugmentationBinding, readReferenceCoverage512(a, io, wrong_parent));

    var dir = try std.Io.Dir.openDirAbsolute(io, output_dir, .{ .follow_symlinks = false });
    defer dir.close(io);
    const original_manifest = try dir.readFileAlloc(io, "manifest.json", a, .limited(64 * 1024));
    defer a.free(original_manifest);
    const class_marker = "\"source_token_class_count\":266";
    const class_at = std.mem.indexOf(u8, original_manifest, class_marker) orelse return error.TestUnexpectedResult;
    const mutated_manifest = try a.dupe(u8, original_manifest);
    defer a.free(mutated_manifest);
    mutated_manifest[class_at + class_marker.len - 1] = '7';
    var manifest_file = try dir.openFile(io, "manifest.json", .{ .mode = .read_write, .allow_directory = false, .follow_symlinks = false });
    defer manifest_file.close(io);
    try manifest_file.writePositionalAll(io, mutated_manifest, 0);
    try manifest_file.sync(io);
    std.crypto.hash.sha2.Sha256.hash(mutated_manifest, &binding.outer_manifest_sha256, .{});
    try std.testing.expectError(error.InvalidReferenceCoverageBinding, readReferenceCoverage512(a, io, binding));
    try manifest_file.writePositionalAll(io, original_manifest, 0);
    try manifest_file.sync(io);

    const digest_hex = std.fmt.bytesToHex(selection.digest, .lower);
    const digest_at = std.mem.indexOf(u8, original_manifest, &digest_hex) orelse return error.TestUnexpectedResult;
    @memcpy(mutated_manifest, original_manifest);
    mutated_manifest[digest_at] = if (mutated_manifest[digest_at] == '0') '1' else '0';
    try manifest_file.writePositionalAll(io, mutated_manifest, 0);
    try manifest_file.sync(io);
    std.crypto.hash.sha2.Sha256.hash(mutated_manifest, &binding.outer_manifest_sha256, .{});
    try std.testing.expectError(error.InvalidReferenceCoverageProvenance, readReferenceCoverage512(a, io, binding));
    try manifest_file.writePositionalAll(io, original_manifest, 0);
    try manifest_file.sync(io);
    std.crypto.hash.sha2.Sha256.hash(original_manifest, &binding.outer_manifest_sha256, .{});

    var offsets_file = try dir.openFile(io, "coverage.selected_offsets.u64", .{ .mode = .read_write, .allow_directory = false, .follow_symlinks = false });
    defer offsets_file.close(io);
    var first: [1]u8 = undefined;
    _ = try offsets_file.readPositionalAll(io, &first, 0);
    first[0] +%= 1;
    try offsets_file.writePositionalAll(io, &first, 0);
    try offsets_file.sync(io);
    try std.testing.expectError(error.InvalidSelfStudyProvenance, readReferenceCoverage512(a, io, binding));
}
