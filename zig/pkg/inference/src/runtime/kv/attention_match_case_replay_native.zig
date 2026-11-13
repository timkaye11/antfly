// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Development replay with optional independent TRAIN32 and bounded Gram geometry.
//! Source export identity and executing worker identity are separate bindings.
const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const platform = @import("antfly_platform");
const checkpoint = @import("attention_match_checkpoint.zig");
const whole = @import("attention_match_whole_fit_input.zig");
const replay = @import("attention_match_joint_chunk_mass_replay.zig");
const global_selection = @import("attention_match_global_selection_diagnostic.zig");
const adaptive = @import("attention_match_joint_adaptive_geometry.zig");
const gram = @import("attention_match_joint_gram_geometry.zig");
const gram_delta = @import("attention_match_gram_delta_accelerate.zig");
const engine = @import("attention_match_engine.zig");
const augmentation = @import("attention_match_training_augmentation.zig");
const train32 = @import("attention_match_train32_augmentation_reader.zig");
const future_augmentation = @import("attention_match_future_augmentation.zig");
const future_training = @import("attention_match_future_decode_training.zig");
const reference_coverage = @import("attention_match_reference_coverage.zig");
const gram_backend = @import("../../ops/metal_attention_match_train_gram.zig");
const training_source = @import("attention_match_joint_training_source.zig");
const local_weight_cap = @import("attention_match_local_weight_cap.zig");
const prepared_input = @import("attention_match_joint_prepared_input.zig");
const bounded_run = @import("attention_match_joint_bounded_run.zig");
const bounded_bridge = @import("attention_match_joint_bounded_tiled_replay.zig");
const tsqr_value = @import("attention_match_e2_tsqr_value_backend.zig");
const train_snapshot = @import("attention_match_joint_train_snapshot.zig");
const train_snapshot_v2 = @import("attention_match_joint_train_snapshot_v2.zig");
const postfit_codec = @import("attention_match_postfit_codec_diagnostic.zig");
const validation_accelerate = @import("../../ops/metal_attention_match_validation.zig");
const physical_budget = @import("attention_match_physical_budget.zig");

const ComponentExperiment = enum { disabled, retained_case1_v1 };
const TrainingRoute = enum { legacy, bounded_exact_f32_streamed_gram_tsqr_v1 };

const default_workspace_bytes = training_source.default_combined_bytes;
const diagnostic_large_workspace_bytes = training_source.supported_hard_maximum_combined_bytes;

fn supportedWorkspaceLimit(bytes: usize) bool {
    return bytes == default_workspace_bytes or bytes == diagnostic_large_workspace_bytes;
}

test "diagnostic workspace limit admits only sealed 512 MiB and 1 GiB profiles" {
    try std.testing.expect(supportedWorkspaceLimit(default_workspace_bytes));
    try std.testing.expect(supportedWorkspaceLimit(diagnostic_large_workspace_bytes));
    try std.testing.expect(!supportedWorkspaceLimit(0));
    try std.testing.expect(!supportedWorkspaceLimit(diagnostic_large_workspace_bytes + 1));
}
const AugmentationPlan = struct {
    input_dir: []const u8,
    manifest_sha256: []const u8,
    parent_study_tokens_sha256: []const u8,
    study_tokens_sha256: []const u8,
    samples_per_consumer: u16,
    self_study_rendered_suffix_sha256: ?[]const u8 = null,
    retained_primary_manifest_path: ?[]const u8 = null,
    retained_primary_manifest_sha256: ?[]const u8 = null,
    training_policy: ?[]const u8 = null,
};
const IndependentAugmentations = struct {
    dense: AugmentationPlan,
    future: FutureAugmentationPlan,
};
const FutureAugmentationPlan = struct {
    input_dir: []const u8,
    manifest_sha256: []const u8,
    source_identity_sha256: []const u8,
    source_tokens_sha256: []const u8,
    source_revision: u64,
    generation_contract_sha256: []const u8,
    generated_ids_sha256: []const u8,
    primary_sha256: []const u8,
    historical_sha256: []const u8,
    future_sha256: []const u8,
    heldout_sha256: []const u8,
    early_eos_observations: u16,
    historical_study_tokens_sha256: ?[]const u8 = null,
};
const CompleteFutureMixPlan = FutureAugmentationPlan;
const HistoricalTrainingOverridePlan = struct {
    input_dir: []const u8,
    manifest_sha256: []const u8,
    policy_sha256: []const u8,
    dense_input_dir: []const u8,
    dense_outer_manifest_sha256: []const u8,
    dense_nested_manifest_sha256: []const u8,
    dense_parent_study_tokens_sha256: []const u8,
    dense_study_tokens_sha256: []const u8,
    v4_proof_sha256: []const u8,
    policy_manifest_sha256: []const u8,
};
const Plan = struct {
    schema: []const u8,
    input_dir: []const u8,
    result_path: []const u8,
    input_manifest_sha256: []const u8,
    identity: struct { weights: []const u8, tokenizer: []const u8, template: []const u8, layout: []const u8 },
    config_sha256: []const u8,
    source_freeze_sha256: []const u8,
    binary_sha256: []const u8,
    exporter_sha256: []const u8,
    worker_source_freeze_sha256: []const u8,
    worker_binary_sha256: []const u8,
    worker_receipt_sha256: []const u8,
    worker_source_manifest_sha256: []const u8,
    donor: usize,
    head: usize,
    query_samples: u16,
    query_batch_rows: usize,
    training_route: TrainingRoute = .legacy,
    training_capture_experiment: bool = false,
    training_capture_backend: bool = false,
    flat_batch_policy: replay.FlatBatchPolicy = .established,
    training_reduction_experiment: bool = false,
    training_reduction_policy: replay.TrainingReductionPolicy = .scalar_reference,
    training_reduction_query_batch_rows: usize = 64,
    branch_chunk_rows: usize,
    max_workspace_bytes: usize,
    geometry_policy: []const u8,
    geometry_storage: replay.GeometryStorage,
    geometry_budget_percent: u8 = 100,
    fallback_geometry_budget_percent: u8 = 0,
    fallback_authorize: ?bool = null,
    gram_memoize_original_delta: bool = false,
    gram_delta_accelerated: bool = false,
    gram_delta_tile_rows: usize = 8,
    joint_fixed_original_memo: bool = false,
    train_budget_exchange: bool = false,
    final_refit_strategy: replay.FinalRefitStrategy = .identity_reuse,
    observe_phase_timing: bool = false,
    bounded_page_route: bool = false,
    bounded_delta_accelerated: bool = false,
    bounded_tile_queries: usize = 8,
    bounded_builder_transient_bytes: usize = 0,
    augmentation: ?AugmentationPlan = null,
    independent_augmentations: ?IndependentAugmentations = null,
    complete_future_mix: ?CompleteFutureMixPlan = null,
    historical_training_override: ?HistoricalTrainingOverridePlan = null,
    selection_score: ?global_selection.SelectionScore = null,
    selection_policy: ?global_selection.SelectionPolicy = null,
    observe_direct_train: bool,
    observe_validation_detail: bool,
    probability_storage: training_source.ProbabilityStorage,
    max_local_weight: ?f64,
    train_snapshot_output_dir: ?[]const u8 = null,
    diagnostic_only: bool,
    quality_credit: bool,
    production_qualified: bool,
    component_experiment: ComponentExperiment = .disabled,
    geometry_chooser_policy: replay.GeometryChooserPolicy = .legacy_full_scan,
    validation_policy: engine.Options.IndependentValidationPolicy = .scalar_reference,
    postfit_codec_checkpoint_path: ?[]const u8 = null,
    postfit_codec_checkpoint_sha256: ?[]const u8 = null,
    postfit_codec_encoding: ?postfit_codec.Encoding = null,
    postfit_codec_rotated4_risk_retention: engine.Rotated4RiskRetention = .quarter_v1,
};

const FallbackAuthorization = struct {
    allow: bool,
    calls: usize = 0,

    fn authorize(raw: ?*anyopaque) bool {
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        self.calls += 1;
        return self.allow;
    }
};

test "native fallback authorization callback counts fixed allow and deny decisions" {
    var allow: FallbackAuthorization = .{ .allow = true };
    try std.testing.expect(FallbackAuthorization.authorize(&allow));
    try std.testing.expectEqual(@as(usize, 1), allow.calls);
    var deny: FallbackAuthorization = .{ .allow = false };
    try std.testing.expect(!FallbackAuthorization.authorize(&deny));
    try std.testing.expectEqual(@as(usize, 1), deny.calls);
}

fn componentPolicyDigest(experiment: ComponentExperiment, geometry: replay.GeometryChooserPolicy, validation: engine.Options.IndependentValidationPolicy, geometry_budget_percent: u8) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(if (geometry_budget_percent == 100) "antfly.whole-fit-component-policy.v1\x00" else "antfly.whole-fit-component-policy.v2\x00");
    h.update(@tagName(experiment));
    h.update("\x00");
    h.update(@tagName(geometry));
    h.update("\x00");
    h.update(@tagName(validation));
    if (geometry_budget_percent != 100) {
        h.update("\x00");
        h.update(&.{geometry_budget_percent});
    }
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

fn trainingReductionPolicyDigest(policy: replay.TrainingReductionPolicy, query_batch_rows: usize) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("antfly.cached-training-reduction-policy.v1\x00");
    h.update(@tagName(policy));
    var le: [8]u8 = undefined;
    std.mem.writeInt(u64, &le, query_batch_rows, .little);
    h.update(&le);
    return h.finalResult();
}

fn flatBatchTupleDigest(policy: replay.FlatBatchPolicy, capture_rows: usize, gram_rows: usize, reduction_rows: usize) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("antfly.complete-future-mix-flat-batch-tuple.v1\x00");
    h.update(@tagName(policy));
    var le: [8]u8 = undefined;
    for ([_]usize{ capture_rows, gram_rows, reduction_rows }) |value| {
        std.mem.writeInt(u64, &le, value, .little);
        h.update(&le);
    }
    return h.finalResult();
}

fn completeFlatBatchTupleValid(policy: replay.FlatBatchPolicy, capture_rows: usize, gram_rows: usize, reduction_rows: usize) bool {
    const rows: usize = if (policy == .diagnostic_capture_gram_256_fallback_64_v1) 256 else 64;
    return capture_rows == rows and gram_rows == rows and reduction_rows == rows;
}

fn integratedPrivateBudgetValid(complete_future_mix: bool, percent: u8) bool {
    return percent == 91 or (complete_future_mix and percent == 95);
}

test "integrated private budget 95 is scoped to complete future mix" {
    try std.testing.expect(integratedPrivateBudgetValid(false, 91));
    try std.testing.expect(integratedPrivateBudgetValid(true, 91));
    try std.testing.expect(integratedPrivateBudgetValid(true, 95));
    try std.testing.expect(!integratedPrivateBudgetValid(false, 95));
    try std.testing.expect(!integratedPrivateBudgetValid(true, 94));
}

test "complete future flat-batch tuples admit established64 and diagnostic256 only" {
    try std.testing.expect(completeFlatBatchTupleValid(.established, 64, 64, 64));
    try std.testing.expect(completeFlatBatchTupleValid(.diagnostic_capture_gram_256_fallback_64_v1, 256, 256, 256));
    try std.testing.expect(!completeFlatBatchTupleValid(.established, 64, 256, 64));
    try std.testing.expect(!completeFlatBatchTupleValid(.diagnostic_capture_gram_256_fallback_64_v1, 256, 64, 256));
    try std.testing.expect(!completeFlatBatchTupleValid(.diagnostic_capture_gram_256_fallback_64_v1, 64, 256, 256));
}

const SnapshotContext = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    authority: train_snapshot.Authority,
    manifest_digest: ?[32]u8 = null,
    fn writeSnapshot(raw: ?*anyopaque, original: @import("attention_match.zig").Head, cached: *const training_source.Source, plan: *const replay.AdaptivePlan, control: @import("attention_match.zig").Control) ![32]u8 {
        const self: *SnapshotContext = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
        const result = try train_snapshot.write(self.allocator, self.io, self.authority, original, cached, plan, control);
        self.manifest_digest = result.manifest_sha256;
        return result.manifest_sha256;
    }
};
const SnapshotV2Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    authority: train_snapshot_v2.Authority,
    manifest_digest: ?[32]u8 = null,
    fn writeSnapshot(raw: ?*anyopaque, domain_runs: []const replay.TrainSnapshotDomainRun, original: @import("attention_match.zig").Head, cached: *const training_source.Source, plan: *const replay.AdaptivePlan, control: @import("attention_match.zig").Control) ![32]u8 {
        const self: *SnapshotV2Context = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
        const result = try train_snapshot_v2.write(self.allocator, self.io, self.authority, domain_runs, original, cached, plan, control);
        self.manifest_digest = result.manifest_sha256;
        return result.manifest_sha256;
    }
};
fn snapshotDomainRuns(allocator: std.mem.Allocator, input: *const whole.OwnedInput, augmented: []const engine.ReferenceBatch, donor: usize, head: usize) ![]replay.TrainSnapshotDomainRun {
    var runs = std.ArrayList(replay.TrainSnapshotDomainRun).empty;
    errdefer runs.deinit(allocator);
    var at: usize = 0;
    for (input.batches) |batch| {
        if (batch.consumer_layer >= input.checkpoint_owned.state.layer_donors.len or input.checkpoint_owned.state.layer_donors[batch.consumer_layer] != donor) continue;
        const group = input.layout.query_heads_by_layer[batch.consumer_layer] / input.checkpoint_owned.state.donors[donor].heads.len;
        if (group == 0 or batch.query_head / group != head) continue;
        const count = batch.train_positions.len;
        if (runs.items.len == 0) try runs.append(allocator, .{ .domain = .primary, .start = at, .count = count }) else runs.items[0].count += count;
        at += count;
    }
    for (augmented, 0..) |batch, index| {
        if (batch.consumer_layer >= input.checkpoint_owned.state.layer_donors.len or input.checkpoint_owned.state.layer_donors[batch.consumer_layer] != donor) continue;
        const group = input.layout.query_heads_by_layer[batch.consumer_layer] / input.checkpoint_owned.state.donors[donor].heads.len;
        if (group == 0 or batch.query_head / group != head) continue;
        const count = batch.train_positions.len;
        try runs.append(allocator, .{ .domain = if ((index & 1) == 0) .historical else .future, .start = at, .count = count });
        at += count;
    }
    if (runs.items.len != 33 or runs.items[0].domain != .primary or runs.items[0].count != 4096 or at != 12288) return error.InvalidWholeFitInput;
    var historical: usize = 0;
    var future: usize = 0;
    for (runs.items[1..], 0..) |run, index| {
        if (run.count != 256 or run.domain != @as(replay.TrainSnapshotDomain, if ((index & 1) == 0) .historical else .future)) return error.InvalidWholeFitInput;
        if (run.domain == .historical) historical += run.count else future += run.count;
    }
    if (historical != 4096 or future != 4096) return error.InvalidWholeFitInput;
    return runs.toOwnedSlice(allocator);
}
fn decodeDigest(value: []const u8) ![32]u8 {
    if (value.len != 64) return error.InvalidWholeFitInput;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, value) catch return error.InvalidWholeFitInput;
    return out;
}

const QueryOrigin = struct { consumer_layer: u32, query_head: u32, position: u64 };

fn heldoutOrigin(input: *const whole.OwnedInput, donor: usize, head: usize, query_index: usize) !QueryOrigin {
    const state = input.checkpoint_owned.state;
    const dim = state.donors[donor].heads[head].dim;
    var remaining = query_index;
    for (input.batches) |batch| {
        if (state.layer_donors[batch.consumer_layer] != donor) continue;
        const group = input.layout.query_heads_by_layer[batch.consumer_layer] / state.donors[donor].heads.len;
        if (group == 0 or batch.query_head / group != head) continue;
        const count = batch.heldout.len / dim;
        if (remaining < count) return .{ .consumer_layer = batch.consumer_layer, .query_head = batch.query_head, .position = batch.heldout_positions[remaining] };
        remaining -= count;
    }
    return error.InvalidWholeFitInput;
}

fn timingScope(p: Plan) []const u8 {
    if (p.observe_direct_train and p.observe_validation_detail) return "fitting_plus_direct_train_and_heldout_detail_observation";
    if (p.observe_direct_train) return "fitting_plus_direct_train_observation";
    if (p.observe_validation_detail) return "fitting_plus_heldout_detail_observation";
    return "fitting_only";
}

fn expectedGeometryPolicy(storage: replay.GeometryStorage, exchange: bool) []const u8 {
    if (exchange) return gram.exchange_policy;
    return switch (storage) {
        .dense => adaptive.signed_joint_policy,
        .per_query_gram => gram.policy,
    };
}

fn expectedBudgetPolicy(storage: replay.GeometryStorage, exchange: bool, percent: u8) ?[]const u8 {
    if (percent == 100) return expectedGeometryPolicy(storage, exchange);
    if (percent == 95 and storage == .per_query_gram and !exchange) return gram.budget95_policy;
    if (percent >= 50 and percent <= 94 and storage == .per_query_gram and !exchange) return gram.budget_percent_policy;
    return null;
}

fn supportedTrainingSourceQueryBatchRows(rows: usize) bool {
    return rows == 16 or rows == 64;
}

test "private flat-cache query batch policy keeps 16 default and permits 64 only" {
    try std.testing.expect(supportedTrainingSourceQueryBatchRows(16));
    try std.testing.expect(supportedTrainingSourceQueryBatchRows(64));
    try std.testing.expect(!supportedTrainingSourceQueryBatchRows(0));
    try std.testing.expect(!supportedTrainingSourceQueryBatchRows(32));
    try std.testing.expect(!supportedTrainingSourceQueryBatchRows(65));
}

test "budget95 policy is explicit Gram-only and default policy is unchanged" {
    try std.testing.expectEqualStrings(gram.policy, expectedBudgetPolicy(.per_query_gram, false, 100).?);
    try std.testing.expectEqualStrings(gram.budget95_policy, expectedBudgetPolicy(.per_query_gram, false, 95).?);
    try std.testing.expect(expectedBudgetPolicy(.dense, false, 95) == null);
    try std.testing.expect(expectedBudgetPolicy(.per_query_gram, true, 95) == null);
}

test "generalized budget policy admits only sealed v16 range" {
    inline for (.{ @as(u8, 50), 51, 94 }) |percent|
        try std.testing.expectEqualStrings(gram.budget_percent_policy, expectedBudgetPolicy(.per_query_gram, false, percent).?);
    inline for (.{ @as(u8, 0), 49, 96, 99 }) |percent|
        try std.testing.expect(expectedBudgetPolicy(.per_query_gram, false, percent) == null);
    try std.testing.expect(expectedBudgetPolicy(.dense, false, 94) == null);
    try std.testing.expect(expectedBudgetPolicy(.per_query_gram, true, 94) == null);
}

fn observationShapeValid(rejection: []const u8, has_mass: bool, has_output: bool, has_direct_train: bool, has_heldout_detail: bool, observe_direct_train: bool, observe_validation_detail: bool) bool {
    const completed = has_mass and has_output and
        (std.mem.eql(u8, rejection, "none") or std.mem.eql(u8, rejection, "heldout_gate"));
    return has_direct_train == (observe_direct_train and completed) and
        has_heldout_detail == (observe_validation_detail and completed);
}

fn heldoutGateEvidenceValid(branches: []const replay.Branch, recorded: usize, bounded_page_route: bool) !bool {
    var observed: usize = 0;
    var unavailable: usize = 0;
    for (branches) |branch| {
        const has_mass = branch.mass_error != null;
        const has_output = branch.output_rmse != null;
        if (has_mass != has_output) return false;
        if (has_mass) observed = try std.math.add(usize, observed, 1);
        if (std.mem.eql(u8, branch.rejection, "geometry_budget_infeasible")) unavailable = try std.math.add(usize, unavailable, 1);
    }
    return recorded == observed and (!bounded_page_route or observed != 0 or unavailable == branches.len);
}

test "bounded all-infeasible geometry has no heldout gate but mixed zero-gate evidence fails" {
    const empty: []f64 = &.{};
    const unavailable = replay.Branch{ .divisor = 4, .candidate_rows = 8, .mass_error = null, .output_rmse = null, .rejection = "geometry_budget_infeasible", .alpha = empty };
    const unavailable_branches = [_]replay.Branch{ unavailable, unavailable, unavailable };
    try std.testing.expect(try heldoutGateEvidenceValid(&unavailable_branches, 0, true));
    const failed = replay.Branch{ .divisor = 3, .candidate_rows = 8, .mass_error = null, .output_rmse = null, .rejection = "joint_mass_or_value_fit", .alpha = empty };
    const mixed = [_]replay.Branch{ unavailable, failed };
    try std.testing.expect(!try heldoutGateEvidenceValid(&mixed, 0, true));
}

fn rejectExactHeldoutCopies(primary: []const engine.ReferenceBatch, augmented: []const engine.ReferenceBatch) !void {
    for (augmented) |candidate| {
        const reference = found: {
            for (primary) |batch| if (batch.consumer_layer == candidate.consumer_layer and batch.query_head == candidate.query_head)
                break :found batch;
            return error.InvalidTrainingAugmentation;
        };
        if (reference.heldout_positions.len == 0 or reference.heldout.len % reference.heldout_positions.len != 0 or
            candidate.train_positions.len == 0 or candidate.train.len % candidate.train_positions.len != 0)
            return error.InvalidTrainingAugmentation;
        const dim = reference.heldout.len / reference.heldout_positions.len;
        if (candidate.train.len / candidate.train_positions.len != dim) return error.InvalidTrainingAugmentation;
        for (0..candidate.train_positions.len) |train_index| {
            const train_row = candidate.train[train_index * dim ..][0..dim];
            for (0..reference.heldout_positions.len) |heldout_index| {
                if (std.mem.eql(u8, std.mem.sliceAsBytes(train_row), std.mem.sliceAsBytes(reference.heldout[heldout_index * dim ..][0..dim]))) return error.SelfStudyHeldoutCopy;
            }
        }
    }
}

const IndependentDomain = struct {
    owned: augmentation.Owned,
    manifest_sha256: [32]u8,
};

fn readIndependent256(
    allocator: std.mem.Allocator,
    io: std.Io,
    binding: AugmentationPlan,
    parent_manifest_sha256: [32]u8,
    identity: checkpoint.Identity,
    source_freeze_sha256: [32]u8,
    binary_sha256: [32]u8,
    exporter_sha256: [32]u8,
) !IndependentDomain {
    if (binding.samples_per_consumer != augmentation.independent_domain_samples_per_stratum or
        binding.training_policy != null or binding.self_study_rendered_suffix_sha256 != null or
        binding.retained_primary_manifest_path != null or binding.retained_primary_manifest_sha256 != null or
        !std.fs.path.isAbsolute(binding.input_dir))
        return error.InvalidTrainingAugmentation;
    const path = try std.fs.path.join(allocator, &.{ binding.input_dir, "manifest.json" });
    defer allocator.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024));
    defer allocator.free(bytes);
    var actual: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &actual, .{});
    if (!std.mem.eql(u8, &actual, &try digest(binding.manifest_sha256))) return error.InvalidTrainingAugmentation;
    var owned = try train32.readForSamples(allocator, io, .{
        .output_dir = binding.input_dir,
        .parent_manifest_sha256 = parent_manifest_sha256,
        .identity = identity,
        .source_freeze_sha256 = source_freeze_sha256,
        .binary_sha256 = binary_sha256,
        .exporter_sha256 = exporter_sha256,
        .parent_study_tokens_sha256 = try digest(binding.parent_study_tokens_sha256),
        .study_tokens_sha256 = try digest(binding.study_tokens_sha256),
    }, augmentation.independent_domain_samples_per_stratum);
    errdefer owned.deinit();
    return .{ .owned = owned, .manifest_sha256 = actual };
}

fn readFuture256(allocator: std.mem.Allocator, io: std.Io, binding: FutureAugmentationPlan, parent_manifest_sha256: [32]u8, identity: checkpoint.Identity, source_freeze_sha256: [32]u8, binary_sha256: [32]u8, exporter_sha256: [32]u8) !IndependentDomain {
    if (!std.fs.path.isAbsolute(binding.input_dir)) return error.InvalidTrainingAugmentation;
    const authority: future_training.Authority = .{ .source_identity_sha256 = try digest(binding.source_identity_sha256), .source_tokens_sha256 = try digest(binding.source_tokens_sha256), .source_revision = binding.source_revision, .generation_contract_sha256 = try digest(binding.generation_contract_sha256), .generated_ids_sha256 = try digest(binding.generated_ids_sha256), .primary_sha256 = try digest(binding.primary_sha256), .historical_sha256 = try digest(binding.historical_sha256), .future_sha256 = try digest(binding.future_sha256), .heldout_sha256 = try digest(binding.heldout_sha256), .early_eos_observations = binding.early_eos_observations };
    var owned = try future_augmentation.read(allocator, io, .{ .output_dir = binding.input_dir, .manifest_sha256 = try digest(binding.manifest_sha256), .primary_parent_manifest_sha256 = parent_manifest_sha256, .identity = identity, .source_freeze_sha256 = source_freeze_sha256, .binary_sha256 = binary_sha256, .exporter_sha256 = exporter_sha256, .authority = authority });
    errdefer owned.deinit();
    return .{ .owned = owned, .manifest_sha256 = try digest(binding.manifest_sha256) };
}

fn validateIndependentBatchTriplet(primary: engine.ReferenceBatch, dense: engine.ReferenceBatch, future: engine.ReferenceBatch) !void {
    if (primary.train_positions.len != 256 or primary.heldout_positions.len != 64 or
        dense.consumer_layer != primary.consumer_layer or dense.query_head != primary.query_head or
        future.consumer_layer != primary.consumer_layer or future.query_head != primary.query_head or
        dense.train_positions.len != 256 or future.train_positions.len != 256 or
        dense.train.len != primary.train.len or future.train.len != primary.train.len or
        dense.heldout.len != 0 or dense.heldout_positions.len != 0 or future.heldout.len != 0 or future.heldout_positions.len != 0)
        return error.InvalidTrainingAugmentation;
}

test "independent dense and future domain shape mutations fail closed" {
    const values: [256]f32 = @splat(1);
    const heldout: [64]f32 = @splat(2);
    const positions: [256]u64 = @splat(1);
    const heldout_positions: [64]u64 = @splat(2);
    const primary: engine.ReferenceBatch = .{ .consumer_layer = 5, .query_head = 7, .train = &values, .heldout = &heldout, .train_positions = &positions, .heldout_positions = &heldout_positions };
    const domain: engine.ReferenceBatch = .{ .consumer_layer = 5, .query_head = 7, .train = &values, .heldout = &.{}, .train_positions = &positions, .heldout_positions = &.{} };
    try validateIndependentBatchTriplet(primary, domain, domain);
    var wrong_consumer = domain;
    wrong_consumer.query_head = 6;
    try std.testing.expectError(error.InvalidTrainingAugmentation, validateIndependentBatchTriplet(primary, wrong_consumer, domain));
    var short = domain;
    short.train_positions = positions[0..255];
    try std.testing.expectError(error.InvalidTrainingAugmentation, validateIndependentBatchTriplet(primary, domain, short));
    var leaked_heldout = domain;
    leaked_heldout.heldout = &.{3};
    try std.testing.expectError(error.InvalidTrainingAugmentation, validateIndependentBatchTriplet(primary, leaked_heldout, domain));
}

fn selectedQueryHeadConsumers(layer_donors: []const u32, query_heads_by_layer: []const u32, donor: usize, donor_head_count: usize, head: usize) !usize {
    if (layer_donors.len != query_heads_by_layer.len or donor_head_count == 0 or head >= donor_head_count) return error.InvalidTrainingAugmentation;
    var count: usize = 0;
    for (layer_donors, query_heads_by_layer) |layer_donor, query_heads| {
        if (layer_donor != donor) continue;
        if (query_heads == 0 or query_heads % donor_head_count != 0) return error.InvalidTrainingAugmentation;
        count = try std.math.add(usize, count, query_heads / donor_head_count);
    }
    if (count == 0) return error.InvalidTrainingAugmentation;
    return count;
}

test "selected query-head consumers derive GQA counts and reject incomplete routing" {
    try std.testing.expectEqual(@as(usize, 2), try selectedQueryHeadConsumers(&.{ 7, 2 }, &.{ 6, 3 }, 7, 3, 1));
    try std.testing.expectEqual(@as(usize, 3), try selectedQueryHeadConsumers(&.{7}, &.{6}, 7, 2, 0));
    try std.testing.expectError(error.InvalidTrainingAugmentation, selectedQueryHeadConsumers(&.{7}, &.{ 6, 6 }, 7, 2, 0));
    try std.testing.expectError(error.InvalidTrainingAugmentation, selectedQueryHeadConsumers(&.{7}, &.{5}, 7, 2, 0));
    try std.testing.expectError(error.InvalidTrainingAugmentation, selectedQueryHeadConsumers(&.{2}, &.{3}, 7, 2, 0));
    try std.testing.expectError(error.InvalidTrainingAugmentation, selectedQueryHeadConsumers(&.{7}, &.{6}, 7, 2, 2));
}

test "self-study copy guard compares query bytes not overlapping position namespaces" {
    const primary = [_]engine.ReferenceBatch{.{ .consumer_layer = 5, .query_head = 1, .train = &.{}, .heldout = &.{ 1, 2 }, .train_positions = &.{}, .heldout_positions = &.{99} }};
    const distinct = [_]engine.ReferenceBatch{.{ .consumer_layer = 5, .query_head = 1, .train = &.{ 1, 3 }, .heldout = &.{}, .train_positions = &.{99}, .heldout_positions = &.{} }};
    try rejectExactHeldoutCopies(&primary, &distinct);
    const copied = [_]engine.ReferenceBatch{.{ .consumer_layer = 5, .query_head = 1, .train = &.{ 1, 2 }, .heldout = &.{}, .train_positions = &.{100}, .heldout_positions = &.{} }};
    try std.testing.expectError(error.SelfStudyHeldoutCopy, rejectExactHeldoutCopies(&primary, &copied));
    const signed_zero_primary = [_]engine.ReferenceBatch{.{ .consumer_layer = 5, .query_head = 1, .train = &.{}, .heldout = &.{ 0.0, 2 }, .train_positions = &.{}, .heldout_positions = &.{99} }};
    const signed_zero_distinct_bytes = [_]engine.ReferenceBatch{.{ .consumer_layer = 5, .query_head = 1, .train = &.{ -0.0, 2 }, .heldout = &.{}, .train_positions = &.{100}, .heldout_positions = &.{} }};
    try rejectExactHeldoutCopies(&signed_zero_primary, &signed_zero_distinct_bytes);
}

test "observer evidence is exact for completed and rejected branches" {
    try std.testing.expect(observationShapeValid("heldout_gate", true, true, true, true, true, true));
    try std.testing.expect(observationShapeValid("selection_or_mass_fit", false, false, false, false, true, true));
    try std.testing.expect(observationShapeValid("none", true, true, false, false, false, false));
    try std.testing.expect(!observationShapeValid("heldout_gate", true, true, false, true, true, true));
    try std.testing.expect(!observationShapeValid("none", true, true, true, false, false, false));
}

fn digest(text: []const u8) ![32]u8 {
    if (text.len != 64) return error.InvalidWholeFitInput;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, text) catch return error.InvalidWholeFitInput;
    if (std.mem.allEqual(u8, &out, 0)) return error.InvalidWholeFitInput;
    return out;
}

const equivalence_fields = [_][]const u8{
    "schema",             "encoding",             "checkpoint_encoding",          "identity",       "config_sha256", "query_samples", "method",
    "fit",                "source_has_metadata",  "original_global_device_bytes", "logical_length", "batch_count",   "layer_count",   "donor_count",
    "layout_scale_count", "layout_softcap_count", "payloads",
};

fn wholeFitEquivalent(allocator: std.mem.Allocator, retained_bytes: []const u8, fresh_bytes: []const u8) !bool {
    const retained = try std.json.parseFromSlice(std.json.Value, allocator, retained_bytes, .{});
    defer retained.deinit();
    const fresh = try std.json.parseFromSlice(std.json.Value, allocator, fresh_bytes, .{});
    defer fresh.deinit();
    if (retained.value != .object or fresh.value != .object) return false;
    inline for (equivalence_fields) |field| {
        const a = retained.value.object.get(field) orelse return false;
        const b = fresh.value.object.get(field) orelse return false;
        const a_text = try std.json.Stringify.valueAlloc(allocator, a, .{});
        defer allocator.free(a_text);
        const b_text = try std.json.Stringify.valueAlloc(allocator, b, .{});
        defer allocator.free(b_text);
        if (!std.mem.eql(u8, a_text, b_text)) return false;
    }
    return true;
}

test "historical and fresh primary manifests may differ in provenance but require exact payload closure" {
    const a = std.testing.allocator;
    const retained =
        \\{"schema":"s","encoding":"e","checkpoint_encoding":"f16","identity":{"weights":"w"},"config_sha256":"c","query_samples":320,"method":"top1","fit":{},"source_has_metadata":true,"original_global_device_bytes":1,"logical_length":2,"batch_count":1,"layer_count":1,"donor_count":1,"layout_scale_count":1,"layout_softcap_count":1,"payloads":[{"name":"p","bytes":4,"count":1,"sha256":[1]}],"source_freeze_sha256":"old"}
    ;
    const fresh =
        \\{"schema":"s","encoding":"e","checkpoint_encoding":"f16","identity":{"weights":"w"},"config_sha256":"c","query_samples":320,"method":"top1","fit":{},"source_has_metadata":true,"original_global_device_bytes":1,"logical_length":2,"batch_count":1,"layer_count":1,"donor_count":1,"layout_scale_count":1,"layout_softcap_count":1,"payloads":[{"name":"p","bytes":4,"count":1,"sha256":[1]}],"source_freeze_sha256":"new"}
    ;
    try std.testing.expect(try wholeFitEquivalent(a, retained, fresh));
    const changed =
        \\{"schema":"s","encoding":"e","checkpoint_encoding":"f16","identity":{"weights":"w"},"config_sha256":"c","query_samples":320,"method":"top1","fit":{},"source_has_metadata":true,"original_global_device_bytes":1,"logical_length":2,"batch_count":1,"layer_count":1,"donor_count":1,"layout_scale_count":1,"layout_softcap_count":1,"payloads":[{"name":"p","bytes":4,"count":1,"sha256":[2]}],"source_freeze_sha256":"new"}
    ;
    try std.testing.expect(!(try wholeFitEquivalent(a, retained, changed)));
}

fn hashU64(hasher: *std.crypto.hash.sha2.Sha256, value: u64) void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, value, .little);
    hasher.update(&bytes);
}

fn nonzeroDigest(value: [32]u8) bool {
    for (value) |byte| if (byte != 0) return true;
    return false;
}

fn headDigest(head: @import("attention_match.zig").Head) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    hashU64(&h, head.rows);
    hashU64(&h, head.dim);
    hashU64(&h, @bitCast(head.scale));
    hashU64(&h, @bitCast(head.softcap));
    for (head.keys) |value| hashU64(&h, @bitCast(value));
    for (head.values) |value| hashU64(&h, @bitCast(value));
    if (head.bias) |bias| for (bias) |value| hashU64(&h, @bitCast(value));
    return h.finalResult();
}

fn mapDigest(source_indices: []const usize, coverage: []const usize) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    hashU64(&h, source_indices.len);
    for (source_indices) |value| hashU64(&h, value);
    hashU64(&h, coverage.len);
    for (coverage) |value| hashU64(&h, value);
    return h.finalResult();
}

fn executableDigest(a: std.mem.Allocator, io: std.Io) ![32]u8 {
    const path = try std.process.executablePathAlloc(io, a);
    defer a.free(path);
    const file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false });
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.InvalidWholeFitInput;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var buffer: [64 * 1024]u8 = undefined;
    var offset: u64 = 0;
    while (offset < stat.size) {
        const count: usize = @intCast(@min(stat.size - offset, buffer.len));
        const n = try file.readPositionalAll(io, buffer[0..count], offset);
        if (n == 0) return error.InvalidWholeFitInput;
        h.update(buffer[0..n]);
        offset += n;
    }
    return h.finalResult();
}

const Discard = struct {
    pub fn writeAll(_: *@This(), _: []const u8) !void {}
};

fn sourceDigest(state: checkpoint.State) ![32]u8 {
    var sink: Discard = .{};
    return (try checkpoint.exportEncoded(state, checkpoint.serving_limits, .f32, &sink, .{})).file_sha256;
}

test "attention match consumed native head replay" {
    if (comptime builtin.os.tag != .macos or !build_options.enable_metal) return error.SkipZigTest;
    const path = platform.env.getenv("ANTFLY_METAL_CASE_HEAD_REPLAY_PLAN") orelse return error.SkipZigTest;
    if (!std.fs.path.isAbsolute(path)) return error.InvalidWholeFitInput;
    const a = std.testing.allocator;
    const io = std.testing.io;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 * 1024));
    defer a.free(bytes);
    const parsed = try std.json.parseFromSlice(Plan, a, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false });
    defer parsed.deinit();
    const p = parsed.value;
    try local_weight_cap.validate(p.max_local_weight);
    const budget_policy = expectedBudgetPolicy(p.geometry_storage, p.train_budget_exchange, p.geometry_budget_percent) orelse return error.InvalidWholeFitInput;
    const generalized_budget = p.geometry_budget_percent >= 50 and p.geometry_budget_percent <= 94;
    const component_ab = p.component_experiment == .retained_case1_v1;
    const postfit_diagnostic = p.postfit_codec_checkpoint_path != null or p.postfit_codec_checkpoint_sha256 != null or p.postfit_codec_encoding != null;
    const high_fanout_bounded = p.training_route == .bounded_exact_f32_streamed_gram_tsqr_v1;
    const complete_future_mix = p.complete_future_mix != null;
    const historical_override = p.historical_training_override != null;
    const selection_score = p.selection_score orelse .raw;
    const selection_policy: global_selection.SelectionPolicy = p.selection_policy orelse switch (selection_score) {
        .raw => .raw,
        .normalized => .normalized,
    };
    const normalized_selection = selection_policy == .normalized;
    const peak_augmented_selection = selection_policy == .train_distinct_key_argmax_augmented_raw_v1;
    const peak_augmented_x16_selection = selection_policy == .train_distinct_key_argmax_augmented_raw_x16_v1;
    const peak_x16_centered_ridge_selection = selection_policy == .train_peak_x16_centered_mass_ridge_1pct_v1;
    const peak_support_centered_ridge_selection = selection_policy == .train_peak_support_centered_ridge_1pct_v1;
    const diagnostic_peak_selection = peak_augmented_selection or peak_augmented_x16_selection or peak_x16_centered_ridge_selection or peak_support_centered_ridge_selection;
    const independent_domains = p.independent_augmentations != null or complete_future_mix;
    const expected_policy = if (component_ab)
        switch (p.geometry_chooser_policy) {
            .legacy_full_scan => gram.policy,
            .incremental_gram_v1 => if (p.geometry_budget_percent == 100) gram.incremental_policy else gram.incremental_budget_percent_policy,
        }
    else if (p.bounded_page_route)
        if (high_fanout_bounded) bounded_bridge.budget_percent_policy else bounded_bridge.policy
    else
        budget_policy;
    const reduction_experiment = p.training_reduction_experiment;
    const reduction_accelerated = p.training_reduction_policy == .accelerate_fp64_cached_v1;
    const diagnostic_flat_batch = p.flat_batch_policy == .diagnostic_capture_gram_256_fallback_64_v1;
    const integrated_private = reduction_experiment and p.training_capture_experiment and p.training_capture_backend and
        reduction_accelerated and p.geometry_chooser_policy == .incremental_gram_v1 and integratedPrivateBudgetValid(complete_future_mix, p.geometry_budget_percent) and
        p.validation_policy == .accelerate_fp64_summary_v1;
    const typed_future_snapshot = p.train_snapshot_output_dir != null and complete_future_mix and integrated_private and
        (p.query_batch_rows == 64 or p.query_batch_rows == 256) and (p.gram_delta_tile_rows == 64 or p.gram_delta_tile_rows == 256);
    const fallback_diagnostic = p.fallback_geometry_budget_percent != 0 or p.fallback_authorize != null;
    const expected_schema = if (fallback_diagnostic) "antfly.consumed-native-head-replay-plan.v28" else if (historical_override) "antfly.consumed-native-head-replay-plan.v30" else if (peak_support_centered_ridge_selection) "antfly.consumed-native-head-replay-plan.v29" else if (peak_x16_centered_ridge_selection) "antfly.consumed-native-head-replay-plan.v27" else if (peak_augmented_x16_selection) "antfly.consumed-native-head-replay-plan.v26" else if (peak_augmented_selection) "antfly.consumed-native-head-replay-plan.v25" else if (normalized_selection) "antfly.consumed-native-head-replay-plan.v24" else if (complete_future_mix) "antfly.consumed-native-head-replay-plan.v23" else if (high_fanout_bounded) "antfly.consumed-native-head-replay-plan.v22" else if (postfit_diagnostic) "antfly.consumed-native-head-replay-plan.v21" else if (reduction_experiment) "antfly.consumed-native-head-replay-plan.v21" else if (p.training_capture_experiment) "antfly.consumed-native-head-replay-plan.v20" else if (component_ab) "antfly.consumed-native-head-replay-plan.v19" else if (p.bounded_delta_accelerated) "antfly.consumed-native-head-replay-plan.v18" else if (generalized_budget) "antfly.consumed-native-head-replay-plan.v16" else if (p.gram_delta_accelerated) "antfly.consumed-native-head-replay-plan.v14" else if (p.geometry_budget_percent == 95) "antfly.consumed-native-head-replay-plan.v12" else if (p.train_snapshot_output_dir != null) "antfly.consumed-native-head-replay-plan.v10" else if (p.final_refit_strategy != .identity_reuse) "antfly.consumed-native-head-replay-plan.v8" else if (p.bounded_page_route) "antfly.consumed-native-head-replay-plan.v7" else "antfly.consumed-native-head-replay-plan.v6";
    if (!std.mem.eql(u8, p.schema, expected_schema) or
        !std.fs.path.isAbsolute(p.input_dir) or !std.fs.path.isAbsolute(p.result_path) or
        !p.diagnostic_only or p.quality_credit or p.production_qualified or p.query_samples != 320 or
        (!supportedTrainingSourceQueryBatchRows(p.query_batch_rows) and (!complete_future_mix or p.query_batch_rows != 256)) or p.branch_chunk_rows != 256 or
        (normalized_selection and (!complete_future_mix or p.bounded_page_route or p.train_snapshot_output_dir != null)) or
        (p.selection_policy != null and !diagnostic_peak_selection) or
        (diagnostic_peak_selection and (!complete_future_mix or p.selection_score != null or p.bounded_page_route or p.train_snapshot_output_dir != null)) or
        (fallback_diagnostic and (!complete_future_mix or selection_policy != .raw or p.geometry_budget_percent != 91 or p.fallback_geometry_budget_percent != 95 or p.fallback_authorize == null or p.bounded_page_route or p.train_snapshot_output_dir != null or !integrated_private or p.donor != 23 or p.head > 1)) or
        (historical_override and (!complete_future_mix or selection_policy != .raw or p.selection_score != null or p.selection_policy != null or p.bounded_page_route or p.train_snapshot_output_dir != null or fallback_diagnostic)) or
        !supportedWorkspaceLimit(p.max_workspace_bytes) or
        !std.mem.eql(u8, p.geometry_policy, expected_policy) or
        ((p.gram_memoize_original_delta or p.train_budget_exchange) and p.geometry_storage != .per_query_gram) or
        (p.gram_delta_accelerated and (p.geometry_storage != .per_query_gram or p.bounded_page_route or (p.train_snapshot_output_dir != null and !typed_future_snapshot) or p.gram_memoize_original_delta or p.train_budget_exchange or p.final_refit_strategy != .identity_reuse or p.max_local_weight != null or p.probability_storage != .exact_f32_widened or (p.gram_delta_tile_rows != 8 and p.gram_delta_tile_rows != 64 and (!complete_future_mix or p.gram_delta_tile_rows != 256)))) or
        (generalized_budget and p.gram_delta_accelerated and !component_ab) or
        (p.bounded_delta_accelerated and (!p.bounded_page_route or p.bounded_tile_queries != 64 or p.probability_storage != .exact_f32_widened or p.geometry_budget_percent != 100)) or
        (!p.gram_delta_accelerated and p.gram_delta_tile_rows != 8) or
        (p.geometry_budget_percent != 100 and (!gram.validBudgetPercent(p.geometry_budget_percent) or (p.bounded_page_route and !high_fanout_bounded) or (p.train_snapshot_output_dir != null and !typed_future_snapshot) or p.train_budget_exchange or p.final_refit_strategy != .identity_reuse or p.max_local_weight != null)) or
        (p.final_refit_strategy != .identity_reuse and (p.bounded_page_route or p.train_budget_exchange or p.max_local_weight != null)) or
        (p.bounded_page_route and (p.geometry_storage != .dense or p.probability_storage != @as(training_source.ProbabilityStorage, if (p.bounded_delta_accelerated or high_fanout_bounded) .exact_f32_widened else .f64) or p.gram_memoize_original_delta or p.joint_fixed_original_memo or p.train_budget_exchange or p.max_local_weight != null or
            p.observe_direct_train or p.observe_phase_timing or p.bounded_tile_queries == 0 or p.bounded_builder_transient_bytes == 0)) or
        (p.independent_augmentations != null and complete_future_mix) or
        (high_fanout_bounded and (!p.bounded_page_route or p.bounded_delta_accelerated or p.geometry_budget_percent < 91 or p.geometry_budget_percent > 95 or p.geometry_chooser_policy != .legacy_full_scan or p.query_batch_rows != 64 or p.bounded_tile_queries != 64 or p.bounded_builder_transient_bytes != 256 * 1024 * 1024 or p.max_workspace_bytes != diagnostic_large_workspace_bytes or p.augmentation != null or !independent_domains or p.donor != 14 or p.head != 0 or !std.mem.eql(u8, p.identity.layout, "b61587d2700c7e3d1c1b014ea0e0a7f7a47bd93ea1237972e88084d5683511e4"))) or
        (!high_fanout_bounded and (p.training_route != .legacy or p.independent_augmentations != null)) or
        (!p.bounded_page_route and (p.bounded_tile_queries != 8 or p.bounded_builder_transient_bytes != 0)) or
        (p.train_snapshot_output_dir != null and (p.bounded_page_route or (p.joint_fixed_original_memo and !typed_future_snapshot) or p.probability_storage != .exact_f32_widened)) or
        (component_ab and (!builtin.is_test or p.geometry_storage != .per_query_gram or (p.geometry_budget_percent != 100 and p.geometry_budget_percent != 91 and p.geometry_budget_percent != 95) or p.probability_storage != .exact_f32_widened or (p.query_batch_rows != 64 and !diagnostic_flat_batch) or !p.joint_fixed_original_memo or p.bounded_page_route or (p.gram_delta_accelerated and !p.training_capture_experiment and !reduction_experiment) or p.gram_memoize_original_delta or p.train_budget_exchange or p.final_refit_strategy != .identity_reuse or p.max_local_weight != null or !p.observe_phase_timing)) or
        (diagnostic_flat_batch and (!complete_future_mix or p.query_batch_rows != 256 or p.gram_delta_tile_rows != 256 or p.training_reduction_query_batch_rows != 256 or !p.training_capture_experiment or !p.training_capture_backend or !p.gram_delta_accelerated or !reduction_accelerated)) or
        (!diagnostic_flat_batch and complete_future_mix and (p.query_batch_rows != 64 or p.gram_delta_tile_rows != 64 or p.training_reduction_query_batch_rows != 64)) or
        (p.training_capture_backend and !p.training_capture_experiment) or
        (p.training_capture_experiment and (!component_ab or !p.gram_delta_accelerated or p.gram_delta_tile_rows != @as(usize, if (diagnostic_flat_batch) 256 else 64) or p.observe_validation_detail or (!integrated_private and (p.geometry_chooser_policy != .legacy_full_scan or p.validation_policy != .scalar_reference)))) or
        (!component_ab and (p.geometry_chooser_policy != .legacy_full_scan or p.validation_policy != .scalar_reference)) or
        p.training_reduction_query_batch_rows == 0 or
        (!reduction_experiment and p.training_reduction_policy != .scalar_reference) or
        (reduction_experiment and (!component_ab or !p.gram_delta_accelerated or p.gram_delta_tile_rows != @as(usize, if (diagnostic_flat_batch) 256 else 64) or p.observe_validation_detail or (!integrated_private and (p.training_capture_experiment or p.training_capture_backend or p.geometry_budget_percent != 100 or p.geometry_chooser_policy != .legacy_full_scan or p.validation_policy != .scalar_reference))))) return error.InvalidWholeFitInput;
    if (postfit_diagnostic and (component_ab or reduction_experiment or p.training_capture_experiment or !builtin.is_test or p.postfit_codec_checkpoint_path == null or !std.fs.path.isAbsolute(p.postfit_codec_checkpoint_path.?) or p.postfit_codec_checkpoint_sha256 == null or p.postfit_codec_encoding == null or (p.postfit_codec_encoding.? == .fp8 and p.postfit_codec_rotated4_risk_retention != .quarter_v1) or p.augmentation == null or p.augmentation.?.samples_per_consumer != 512 or p.augmentation.?.training_policy == null or !std.mem.eql(u8, p.augmentation.?.training_policy.?, reference_coverage.dense512_policy) or p.max_workspace_bytes != diagnostic_large_workspace_bytes or p.query_batch_rows != 64 or p.geometry_budget_percent != 100 or p.geometry_storage != .per_query_gram or p.probability_storage != .exact_f32_widened or !p.joint_fixed_original_memo or p.bounded_page_route or p.bounded_delta_accelerated or p.gram_delta_accelerated or p.gram_memoize_original_delta or p.train_budget_exchange or p.final_refit_strategy != .identity_reuse or p.max_local_weight != null or p.train_snapshot_output_dir != null or p.observe_direct_train)) return error.InvalidWholeFitInput;
    _ = try digest(p.worker_source_freeze_sha256);
    _ = try digest(p.worker_receipt_sha256);
    _ = try digest(p.worker_source_manifest_sha256);
    if (postfit_diagnostic) _ = try digest(p.postfit_codec_checkpoint_sha256.?);
    const actual_binary = try executableDigest(a, io);
    if (!std.mem.eql(u8, &actual_binary, &try digest(p.worker_binary_sha256))) return error.InvalidWholeFitInput;
    var plan_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &plan_digest, .{});
    const manifest_path = try std.fs.path.join(a, &.{ p.input_dir, "manifest.json" });
    defer a.free(manifest_path);
    const manifest = try std.Io.Dir.cwd().readFileAlloc(io, manifest_path, a, .limited(whole.manifest_reservation));
    defer a.free(manifest);
    var manifest_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(manifest, &manifest_digest, .{});
    if (!std.mem.eql(u8, &manifest_digest, &try digest(p.input_manifest_sha256))) return error.InvalidWholeFitInput;
    const identity: checkpoint.Identity = .{ .weights = try digest(p.identity.weights), .tokenizer = try digest(p.identity.tokenizer), .template = try digest(p.identity.template), .layout = try digest(p.identity.layout) };
    var input = try whole.read(a, io, .{ .output_dir = p.input_dir, .identity = identity, .config_sha256 = try digest(p.config_sha256), .source_freeze_sha256 = try digest(p.source_freeze_sha256), .binary_sha256 = try digest(p.binary_sha256), .exporter_sha256 = try digest(p.exporter_sha256), .limits = checkpoint.serving_limits });
    defer input.deinit();
    if (input.query_samples != p.query_samples or input.method != .top1) return error.InvalidWholeFitInput;
    var augmented_owned: ?augmentation.Owned = null;
    defer if (augmented_owned) |*owned| owned.deinit();
    var augmentation_digest: ?[32]u8 = null;
    var dense_domain: ?IndependentDomain = null;
    defer if (dense_domain) |*domain| domain.owned.deinit();
    var override_evidence: ?struct { manifest: [32]u8, policy: [32]u8, derived: [32]u8, eligible: [32]u8, selection: [32]u8, dense_outer: [32]u8, dense_nested: [32]u8, v4_proof: [32]u8, policy_manifest: [32]u8, mix: [32]u8 } = null;
    var future_domain: ?IndependentDomain = null;
    defer if (future_domain) |*domain| domain.owned.deinit();
    var independent_batches: ?[]engine.ReferenceBatch = null;
    defer if (independent_batches) |batches| a.free(batches);
    if (p.augmentation) |binding| {
        if (!std.fs.path.isAbsolute(binding.input_dir)) return error.InvalidTrainingAugmentation;
        if (binding.samples_per_consumer != 32 and binding.samples_per_consumer != 128 and binding.samples_per_consumer != 320 and binding.samples_per_consumer != 512) return error.InvalidTrainingAugmentation;
        const augmentation_path = try std.fs.path.join(a, &.{ binding.input_dir, "manifest.json" });
        defer a.free(augmentation_path);
        const augmentation_manifest = try std.Io.Dir.cwd().readFileAlloc(io, augmentation_path, a, .limited(64 * 1024));
        defer a.free(augmentation_manifest);
        var actual_digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(augmentation_manifest, &actual_digest, .{});
        if (!std.mem.eql(u8, &actual_digest, &try digest(binding.manifest_sha256))) return error.InvalidTrainingAugmentation;
        if (binding.samples_per_consumer == 320 or binding.samples_per_consumer == 512) {
            const retained_path = binding.retained_primary_manifest_path orelse return error.InvalidSelfStudyBinding;
            const retained_sha_text = binding.retained_primary_manifest_sha256 orelse return error.InvalidSelfStudyBinding;
            if (!std.fs.path.isAbsolute(retained_path)) return error.InvalidSelfStudyBinding;
            const retained_manifest = try std.Io.Dir.cwd().readFileAlloc(io, retained_path, a, .limited(whole.manifest_reservation));
            defer a.free(retained_manifest);
            var retained_digest: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(retained_manifest, &retained_digest, .{});
            if (!std.mem.eql(u8, &retained_digest, &try digest(retained_sha_text)) or
                !try wholeFitEquivalent(a, retained_manifest, manifest)) return error.InvalidSelfStudyBinding;
            if (binding.training_policy) |policy| {
                const dense512 = binding.samples_per_consumer == 512;
                if (!std.mem.eql(u8, policy, if (dense512) reference_coverage.dense512_policy else reference_coverage.policy) or binding.self_study_rendered_suffix_sha256 != null) return error.InvalidReferenceCoverageBinding;
                const coverage_binding: train32.ReferenceCoverageBinding = .{
                    .output_dir = binding.input_dir,
                    .outer_manifest_sha256 = actual_digest,
                    .retained_primary_manifest_sha256 = retained_digest,
                    .fresh_primary_manifest_sha256 = manifest_digest,
                    .identity = identity,
                    .source_freeze_sha256 = try digest(p.source_freeze_sha256),
                    .binary_sha256 = try digest(p.binary_sha256),
                    .exporter_sha256 = try digest(p.exporter_sha256),
                    .parent_study_tokens_sha256 = try digest(binding.parent_study_tokens_sha256),
                    .study_tokens_sha256 = try digest(binding.study_tokens_sha256),
                };
                augmented_owned = if (dense512) try train32.readReferenceCoverage512(a, io, coverage_binding) else try train32.readReferenceCoverage320(a, io, coverage_binding);
            } else {
                if (binding.samples_per_consumer != 320) return error.InvalidSelfStudyBinding;
                const rendered = binding.self_study_rendered_suffix_sha256 orelse return error.InvalidSelfStudyBinding;
                augmented_owned = try train32.readSelfStudy320(a, io, .{
                    .output_dir = binding.input_dir,
                    .outer_manifest_sha256 = actual_digest,
                    .retained_primary_manifest_sha256 = retained_digest,
                    .fresh_primary_manifest_sha256 = manifest_digest,
                    .identity = identity,
                    .source_freeze_sha256 = try digest(p.source_freeze_sha256),
                    .binary_sha256 = try digest(p.binary_sha256),
                    .exporter_sha256 = try digest(p.exporter_sha256),
                    .parent_study_tokens_sha256 = try digest(binding.parent_study_tokens_sha256),
                    .study_tokens_sha256 = try digest(binding.study_tokens_sha256),
                    .rendered_suffix_sha256 = try digest(rendered),
                });
            }
        } else {
            if (binding.training_policy != null or binding.self_study_rendered_suffix_sha256 != null or binding.retained_primary_manifest_path != null or
                binding.retained_primary_manifest_sha256 != null) return error.InvalidTrainingAugmentation;
            augmented_owned = try train32.readForSamples(a, io, .{
                .output_dir = binding.input_dir,
                .parent_manifest_sha256 = manifest_digest,
                .identity = identity,
                .source_freeze_sha256 = try digest(p.source_freeze_sha256),
                .binary_sha256 = try digest(p.binary_sha256),
                .exporter_sha256 = try digest(p.exporter_sha256),
                .parent_study_tokens_sha256 = try digest(binding.parent_study_tokens_sha256),
                .study_tokens_sha256 = try digest(binding.study_tokens_sha256),
            }, binding.samples_per_consumer);
        }
        if (augmented_owned.?.source_length != input.checkpoint_owned.state.logical_length or
            ((binding.samples_per_consumer == 320 or binding.samples_per_consumer == 512) and binding.training_policy != null and augmented_owned.?.repeat_length != reference_coverage.source_study_tokens) or
            (binding.samples_per_consumer == 320 and binding.training_policy == null and augmented_owned.?.repeat_length < 320) or
            (binding.samples_per_consumer != 320 and augmented_owned.?.repeat_length != 512))
            return error.InvalidTrainingAugmentation;
        if (binding.samples_per_consumer == 320 or binding.samples_per_consumer == 512) try rejectExactHeldoutCopies(input.batches, augmented_owned.?.batches);
        augmentation_digest = actual_digest;
    }
    if (p.independent_augmentations) |bindings| {
        dense_domain = try readIndependent256(a, io, bindings.dense, manifest_digest, identity, try digest(p.source_freeze_sha256), try digest(p.binary_sha256), try digest(p.exporter_sha256));
        future_domain = try readFuture256(a, io, bindings.future, manifest_digest, identity, try digest(p.source_freeze_sha256), try digest(p.binary_sha256), try digest(p.exporter_sha256));
        if (std.mem.eql(u8, &dense_domain.?.manifest_sha256, &future_domain.?.manifest_sha256) or
            std.mem.eql(u8, bindings.dense.input_dir, bindings.future.input_dir) or
            std.mem.eql(u8, bindings.dense.study_tokens_sha256, bindings.future.generated_ids_sha256) or
            dense_domain.?.owned.source_length != input.checkpoint_owned.state.logical_length or
            future_domain.?.owned.source_length != input.checkpoint_owned.state.logical_length or
            dense_domain.?.owned.batches.len != input.batches.len or future_domain.?.owned.batches.len != input.batches.len)
            return error.InvalidTrainingAugmentation;
        try rejectExactHeldoutCopies(input.batches, dense_domain.?.owned.batches);
        try rejectExactHeldoutCopies(input.batches, future_domain.?.owned.batches);
        const combined = try a.alloc(engine.ReferenceBatch, try std.math.mul(usize, input.batches.len, 2));
        independent_batches = combined;
        for (input.batches, dense_domain.?.owned.batches, future_domain.?.owned.batches, 0..) |primary, dense, future, index| {
            try validateIndependentBatchTriplet(primary, dense, future);
            combined[index * 2] = dense;
            combined[index * 2 + 1] = future;
        }
    } else if (p.complete_future_mix) |binding| {
        if (!std.fs.path.isAbsolute(binding.input_dir)) return error.InvalidTrainingAugmentation;
        const authority: future_training.Authority = .{ .source_identity_sha256 = try digest(binding.source_identity_sha256), .source_tokens_sha256 = try digest(binding.source_tokens_sha256), .source_revision = binding.source_revision, .generation_contract_sha256 = try digest(binding.generation_contract_sha256), .generated_ids_sha256 = try digest(binding.generated_ids_sha256), .primary_sha256 = try digest(binding.primary_sha256), .historical_sha256 = try digest(binding.historical_sha256), .future_sha256 = try digest(binding.future_sha256), .heldout_sha256 = try digest(binding.heldout_sha256), .early_eos_observations = binding.early_eos_observations };
        var complete = try future_augmentation.readComplete(a, io, .{ .output_dir = binding.input_dir, .manifest_sha256 = try digest(binding.manifest_sha256), .primary_parent_manifest_sha256 = manifest_digest, .identity = identity, .source_freeze_sha256 = try digest(p.source_freeze_sha256), .binary_sha256 = try digest(p.binary_sha256), .exporter_sha256 = try digest(p.exporter_sha256), .authority = authority, .historical_study_tokens_sha256 = try digest(binding.historical_study_tokens_sha256 orelse return error.InvalidTrainingAugmentation), .primary_batches = input.batches });
        dense_domain = .{ .owned = complete.historical, .manifest_sha256 = try digest(binding.manifest_sha256) };
        future_domain = .{ .owned = complete.future, .manifest_sha256 = try digest(binding.manifest_sha256) };
        complete = undefined;
        if (p.historical_training_override) |override| {
            var replacement = try augmentation.readHistoricalOverride(a, io, .{ .output_dir = override.input_dir, .manifest_sha256 = try digest(override.manifest_sha256), .policy_sha256 = try digest(override.policy_sha256), .parent_manifest_sha256 = manifest_digest, .complete_manifest_sha256 = try digest(binding.manifest_sha256), .identity = identity, .config_sha256 = try digest(p.config_sha256), .source_freeze_sha256 = try digest(p.source_freeze_sha256), .binary_sha256 = try digest(p.binary_sha256), .exporter_sha256 = try digest(p.exporter_sha256), .base_historical_sha256 = authority.historical_sha256, .primary_sha256 = authority.primary_sha256, .future_sha256 = authority.future_sha256, .heldout_sha256 = authority.heldout_sha256, .dense_output_dir = override.dense_input_dir, .dense_outer_manifest_sha256 = try digest(override.dense_outer_manifest_sha256), .dense_nested_manifest_sha256 = try digest(override.dense_nested_manifest_sha256), .dense_parent_study_tokens_sha256 = try digest(override.dense_parent_study_tokens_sha256), .dense_study_tokens_sha256 = try digest(override.dense_study_tokens_sha256), .v4_proof_sha256 = try digest(override.v4_proof_sha256), .policy_manifest_sha256 = try digest(override.policy_manifest_sha256) }, input.batches);
            dense_domain.?.owned.deinit();
            dense_domain.?.owned = replacement.owned;
            replacement.owned = undefined;
            var mix_hash = std.crypto.hash.sha2.Sha256.init(.{});
            mix_hash.update("antfly.complete-future-historical-override.v1\x00");
            mix_hash.update(&try digest(binding.manifest_sha256));
            mix_hash.update(&try digest(override.manifest_sha256));
            mix_hash.update(&try digest(override.policy_sha256));
            mix_hash.update(&replacement.derived_historical_sha256);
            mix_hash.update(&authority.primary_sha256);
            mix_hash.update(&authority.future_sha256);
            mix_hash.update(&authority.heldout_sha256);
            override_evidence = .{ .manifest = try digest(override.manifest_sha256), .policy = try digest(override.policy_sha256), .derived = replacement.derived_historical_sha256, .eligible = replacement.eligible_indices_sha256, .selection = replacement.selection_indices_sha256, .dense_outer = replacement.dense_outer_manifest_sha256, .dense_nested = replacement.dense_nested_manifest_sha256, .v4_proof = replacement.v4_proof_sha256, .policy_manifest = replacement.policy_manifest_sha256, .mix = mix_hash.finalResult() };
        }
        if (dense_domain.?.owned.source_length != input.checkpoint_owned.state.logical_length or future_domain.?.owned.source_length != input.checkpoint_owned.state.logical_length or dense_domain.?.owned.batches.len != input.batches.len or future_domain.?.owned.batches.len != input.batches.len) return error.InvalidTrainingAugmentation;
        try rejectExactHeldoutCopies(input.batches, dense_domain.?.owned.batches);
        try rejectExactHeldoutCopies(input.batches, future_domain.?.owned.batches);
        const combined = try a.alloc(engine.ReferenceBatch, try std.math.mul(usize, input.batches.len, 2));
        independent_batches = combined;
        for (input.batches, dense_domain.?.owned.batches, future_domain.?.owned.batches, 0..) |primary, dense, future, index| {
            try validateIndependentBatchTriplet(primary, dense, future);
            combined[index * 2] = dense;
            combined[index * 2 + 1] = future;
        }
    }
    const augmented: []const engine.ReferenceBatch = if (independent_batches) |batches| batches else if (augmented_owned) |*owned| owned.batches else &.{};
    for (input.batches) |batch| if (batch.train_positions.len != 256 or batch.heldout_positions.len != 64)
        return error.InvalidWholeFitInput;
    if (p.augmentation != null or independent_domains) {
        if (augmented.len != input.batches.len * @as(usize, if (independent_domains) 2 else 1)) return error.InvalidTrainingAugmentation;
        for (augmented) |batch| {
            var matched = false;
            for (input.batches) |primary| {
                if (batch.consumer_layer != primary.consumer_layer or batch.query_head != primary.query_head) continue;
                const primary_dim = primary.train.len / primary.train_positions.len;
                const expected_samples: usize = if (independent_domains) 256 else p.augmentation.?.samples_per_consumer;
                if (batch.train.len != try std.math.mul(usize, expected_samples, primary_dim)) return error.InvalidTrainingAugmentation;
                matched = true;
                break;
            }
            if (!matched) return error.InvalidTrainingAugmentation;
        }
    }
    const state = input.checkpoint_owned.state;
    if (p.donor >= state.donors.len or p.head >= state.donors[p.donor].heads.len or state.donors[p.donor].window != 0) return error.InvalidWholeFitInput;
    const dim = state.donors[p.donor].heads[p.head].dim;
    var train_elements: usize = 0;
    var primary_train_elements: usize = 0;
    var augmentation_train_elements: usize = 0;
    var dense_train_elements: usize = 0;
    var future_train_elements: usize = 0;
    var heldout_elements: usize = 0;
    for ([_][]const engine.ReferenceBatch{ input.batches, augmented }, 0..) |batches, domain| for (batches, 0..) |batch, batch_index| {
        if (batch.consumer_layer >= state.layer_donors.len or batch.query_head >= input.layout.query_heads_by_layer[batch.consumer_layer]) return error.InvalidWholeFitInput;
        if (state.layer_donors[batch.consumer_layer] != p.donor) continue;
        const group = input.layout.query_heads_by_layer[batch.consumer_layer] / state.donors[p.donor].heads.len;
        if (group == 0) return error.InvalidWholeFitInput;
        if (batch.query_head / group != p.head) continue;
        if (batch.train.len % dim != 0 or batch.heldout.len % dim != 0) return error.InvalidWholeFitInput;
        train_elements = try std.math.add(usize, train_elements, batch.train.len);
        if (domain == 0) {
            primary_train_elements = try std.math.add(usize, primary_train_elements, batch.train.len);
        } else {
            augmentation_train_elements = try std.math.add(usize, augmentation_train_elements, batch.train.len);
            if (independent_domains) {
                if (batch_index % 2 == 0)
                    dense_train_elements = try std.math.add(usize, dense_train_elements, batch.train.len)
                else
                    future_train_elements = try std.math.add(usize, future_train_elements, batch.train.len);
            }
        }
        heldout_elements = try std.math.add(usize, heldout_elements, batch.heldout.len);
    };
    if (train_elements == 0 or heldout_elements == 0 or train_elements % dim != 0 or heldout_elements % dim != 0) return error.InvalidWholeFitInput;
    if (p.augmentation != null and augmentation_train_elements == 0)
        return error.InvalidTrainingAugmentation;
    if (complete_future_mix) {
        const consumers = try selectedQueryHeadConsumers(state.layer_donors, input.layout.query_heads_by_layer, p.donor, state.donors[p.donor].heads.len, p.head);
        const per_domain = try std.math.mul(usize, consumers, 256);
        const expected_heldout = try std.math.mul(usize, consumers, 64);
        const expected_train = try std.math.mul(usize, per_domain, 3);
        if (primary_train_elements / dim != per_domain or dense_train_elements / dim != per_domain or future_train_elements / dim != per_domain or heldout_elements / dim != expected_heldout or train_elements / dim != expected_train) return error.InvalidTrainingAugmentation;
    } else if (independent_domains and
        (primary_train_elements / dim != 10_240 or dense_train_elements / dim != 10_240 or future_train_elements / dim != 10_240 or heldout_elements / dim != 2_560 or train_elements / dim != 30_720))
        return error.InvalidTrainingAugmentation;
    const source_before = try sourceDigest(input.checkpoint_owned.state);
    if (postfit_diagnostic) {
        const checkpoint_path = p.postfit_codec_checkpoint_path.?;
        const checkpoint_bytes = try std.Io.Dir.cwd().readFileAlloc(io, checkpoint_path, a, .limited(checkpoint.serving_limits.max_file_bytes));
        defer a.free(checkpoint_bytes);
        var checkpoint_digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(checkpoint_bytes, &checkpoint_digest, .{});
        if (!std.mem.eql(u8, &checkpoint_digest, &try digest(p.postfit_codec_checkpoint_sha256.?))) return error.InvalidWholeFitInput;
        var fitted_owned = try checkpoint.decode(a, checkpoint_bytes, identity, checkpoint.serving_limits);
        defer fitted_owned.deinit();
        const fitted_state = fitted_owned.state;
        if (fitted_state.logical_length != state.logical_length or !std.mem.eql(u32, fitted_state.tokens, state.tokens) or
            !std.mem.eql(u32, fitted_state.layer_donors, state.layer_donors) or fitted_state.donors.len != state.donors.len or
            p.donor >= fitted_state.donors.len or p.head >= fitted_state.donors[p.donor].heads.len or
            fitted_state.donors[p.donor].layer != state.donors[p.donor].layer or fitted_state.donors[p.donor].window != 0)
            return error.InvalidWholeFitInput;
        const original_head = state.donors[p.donor].heads[p.head];
        const fitted_head = fitted_state.donors[p.donor].heads[p.head];
        var train = try a.alloc(f64, train_elements);
        defer a.free(train);
        var heldout = try a.alloc(f64, heldout_elements);
        defer a.free(heldout);
        var train_at: usize = 0;
        var heldout_at: usize = 0;
        for ([_][]const engine.ReferenceBatch{ input.batches, augmented }) |batches| for (batches) |batch| {
            if (state.layer_donors[batch.consumer_layer] != p.donor) continue;
            const group = input.layout.query_heads_by_layer[batch.consumer_layer] / state.donors[p.donor].heads.len;
            if (batch.query_head / group != p.head) continue;
            for (batch.train) |value| {
                train[train_at] = value;
                train_at += 1;
            }
            for (batch.heldout) |value| {
                heldout[heldout_at] = value;
                heldout_at += 1;
            }
        };
        if (train_at != train.len or heldout_at != heldout.len) return error.InvalidWholeFitInput;
        const envelope = postfit_codec.fittedEnvelope(original_head, fitted_head) catch null;
        var diagnostic: engine.CodecDiagnostic = .{};
        const reconstructable = envelope != null;
        if (reconstructable) diagnostic = try postfit_codec.diagnoseHead(a, original_head, fitted_head, train, heldout, input.layout.scales_by_donor[p.donor], input.layout.softcaps_by_donor[p.donor], p.postfit_codec_encoding.?, p.postfit_codec_rotated4_risk_retention, .{
            .target_rows = 1,
            .max_workspace_bytes = p.max_workspace_bytes,
            .max_relative_mass_error = 0.05,
            .max_relative_output_rmse = 0.05,
        });
        const source_after = try sourceDigest(input.checkpoint_owned.state);
        const fitted_after = try sourceDigest(fitted_owned.state);
        if (!std.mem.eql(u8, &source_before, &source_after)) return error.InvalidWholeFitInput;
        const body = try std.json.Stringify.valueAlloc(a, .{
            .schema = "antfly.postfit-codec-diagnostic-result.v2",
            .plan_sha256 = std.fmt.bytesToHex(plan_digest, .lower),
            .checkpoint_path = checkpoint_path,
            .checkpoint_sha256 = std.fmt.bytesToHex(checkpoint_digest, .lower),
            .checkpoint_state_sha256 = std.fmt.bytesToHex(fitted_after, .lower),
            .input_manifest_sha256 = std.fmt.bytesToHex(manifest_digest, .lower),
            .augmentation_manifest_sha256 = if (augmentation_digest) |value| std.fmt.bytesToHex(value, .lower) else null,
            .worker_source_freeze_sha256 = p.worker_source_freeze_sha256,
            .worker_binary_sha256 = p.worker_binary_sha256,
            .worker_receipt_sha256 = p.worker_receipt_sha256,
            .historical_export_source_freeze_sha256 = p.source_freeze_sha256,
            .historical_export_binary_sha256 = p.binary_sha256,
            .historical_exporter_sha256 = p.exporter_sha256,
            .donor = p.donor,
            .head = p.head,
            .encoding = p.postfit_codec_encoding.?,
            .rotated4_risk_retention_requested = p.postfit_codec_rotated4_risk_retention,
            .rotated4_risk_retention_actual = if (reconstructable) diagnostic.rotated4_risk_retention else p.postfit_codec_rotated4_risk_retention,
            .codec_eligible_rows = diagnostic.eligible_rows,
            .codec_retained_exact_rows = diagnostic.retained_exact_rows,
            .codec_packed_rows = diagnostic.packed_rows,
            .envelope_reconstructable = reconstructable,
            .outcome = if (reconstructable) @tagName(diagnostic.outcome) else "unreconstructable",
            .first = diagnostic.first,
            .end = diagnostic.end,
            .quality_evaluated = diagnostic.metrics.quality_evaluated,
            .max_relative_mass_error = diagnostic.metrics.max_relative_mass_error,
            .relative_output_rmse = diagnostic.metrics.relative_output_rmse,
            .max_relative_mass_error_limit = 0.05,
            .relative_output_rmse_limit = 0.05,
            .validation_policy = "scalar_fp64_independent_v1",
            .train_queries = train.len / dim,
            .heldout_queries = heldout.len / dim,
            .source_unchanged = true,
            .source_before_sha256 = std.fmt.bytesToHex(source_before, .lower),
            .source_after_sha256 = std.fmt.bytesToHex(source_after, .lower),
            .refit_performed = false,
            .published = false,
            .diagnostic_only = true,
            .quality_credit = false,
            .production_qualified = false,
        }, .{});
        defer a.free(body);
        var result_file = try std.Io.Dir.createFileAbsolute(io, p.result_path, .{ .exclusive = true, .truncate = false });
        defer result_file.close(io);
        try result_file.writeStreamingAll(io, body);
        try result_file.writeStreamingAll(io, "\n");
        try result_file.sync(io);
        return;
    }
    const Fitter = @import("../../ops/metal_attention_match_fit.zig").Fitter;
    var fitter = try Fitter.init(a, null, p.max_workspace_bytes);
    var live = true;
    defer if (live) fitter.deinit();
    try fitter.setQueryBatchRows(p.query_batch_rows);
    try fitter.setPhaseProfiling(complete_future_mix);
    if (fitter.queryBatchRows() != p.query_batch_rows) return error.InvalidWholeFitInput;
    var gram_context: gram_backend.Context = .{};
    var validation_context: validation_accelerate.Context = .{};
    var phase_timing: replay.PhaseTiming = .{};
    var snapshot_context: SnapshotContext = .{ .allocator = a, .io = io, .authority = .{ .output_dir = p.train_snapshot_output_dir orelse "/unused", .source_sha256 = source_before, .config_sha256 = try decodeDigest(p.config_sha256), .model_sha256 = try decodeDigest(p.identity.weights), .source_freeze_sha256 = try decodeDigest(p.worker_source_freeze_sha256), .binary_sha256 = try decodeDigest(p.worker_binary_sha256) } };
    const snapshot_v2_runs = if (typed_future_snapshot) try snapshotDomainRuns(a, &input, augmented, p.donor, p.head) else &.{};
    defer if (snapshot_v2_runs.len != 0) a.free(snapshot_v2_runs);
    const snapshot_mix = p.complete_future_mix;
    var snapshot_v2_context: SnapshotV2Context = .{ .allocator = a, .io = io, .authority = .{ .output_dir = p.train_snapshot_output_dir orelse "/unused", .source_sha256 = source_before, .config_sha256 = try decodeDigest(p.config_sha256), .model_sha256 = try decodeDigest(p.identity.weights), .source_freeze_sha256 = try decodeDigest(p.worker_source_freeze_sha256), .binary_sha256 = try decodeDigest(p.worker_binary_sha256), .producer_input_manifest_sha256 = try decodeDigest(p.input_manifest_sha256), .complete_future_mix_manifest_sha256 = try decodeDigest(if (snapshot_mix) |m| m.manifest_sha256 else p.input_manifest_sha256), .complete_future_mix_policy_sha256 = try decodeDigest(if (snapshot_mix) |m| m.generation_contract_sha256 else p.config_sha256), .consumer_source_manifest_sha256 = try decodeDigest(p.worker_source_manifest_sha256), .consumer_binary_sha256 = try decodeDigest(p.worker_binary_sha256) } };
    var bounded_evidence: ?bounded_run.Evidence = null;
    var tsqr_context: tsqr_value.Context = .{};
    var fallback_authorization: FallbackAuthorization = .{ .allow = p.fallback_authorize orelse false };
    const started = platform.time.monotonicNs();
    var result = if (p.bounded_page_route) blk: {
        var prepared = try prepared_input.prepareInput(a, &input, augmented, fitter.backend(), p.donor, p.head, p.branch_chunk_rows, p.max_workspace_bytes, .{});
        defer prepared.deinit();
        if (high_fanout_bounded) prepared.cfg.value_backend = tsqr_context.valueBackend();
        const acceleration: ?bounded_run.GeometryAcceleration = if (p.bounded_delta_accelerated)
            .{ .backend = gram_delta.accelerateBackend() orelse return error.AcceleratedDeltaUnsupported }
        else
            null;
        const output = if (high_fanout_bounded)
            try bounded_run.runPreparedViewBoundedTimedAcceleratedBudget(a, bounded_run.PreparedView.fromOwned(&prepared), &fitter, p.bounded_tile_queries, p.bounded_builder_transient_bytes, p.geometry_budget_percent, p.observe_validation_detail, null, acceleration)
        else
            try bounded_run.runPreparedViewBoundedTimedAccelerated(a, bounded_run.PreparedView.fromOwned(&prepared), &fitter, p.bounded_tile_queries, p.bounded_builder_transient_bytes, p.observe_validation_detail, null, acceleration);
        if (high_fanout_bounded and (output.evidence.train_queries != 30_720 or output.evidence.train_queries != train_elements / dim or output.evidence.geometry_budget_percent != p.geometry_budget_percent or
            !std.mem.eql(u8, output.evidence.training_route, "bounded_exact_f32_streamed_gram_tsqr_v1") or
            !std.mem.eql(u8, output.evidence.raw_selection_policy, @import("attention_match_streamed_gram_selection.zig").policy) or
            output.evidence.raw_selection_calls == 0 or output.evidence.raw_selection_probability_values == 0 or
            output.evidence.raw_selection_workspace_peak_bytes == 0 or tsqr_context.calls == 0 or tsqr_context.cancellation_checks == 0))
            return error.InvalidWholeFitInput;
        bounded_evidence = output.evidence;
        break :blk output.result;
    } else replay.runInputWithOptions(a, &input, augmented, fitter.backend(), p.donor, p.head, p.branch_chunk_rows, p.max_workspace_bytes, .{ .phase_timing = if (p.observe_phase_timing) &phase_timing else null, .training_source = true, .selection_score = selection_score, .selection_policy = p.selection_policy, .mass_strategy = .identity_chunk_coefficients_v18, .final_refit_strategy = p.final_refit_strategy, .observe_direct_train = p.observe_direct_train, .observe_validation_detail = p.observe_validation_detail, .probability_storage = p.probability_storage, .training_source_query_batch_rows = p.query_batch_rows, .training_capture_backend = if (p.training_capture_backend) fitter.trainingCaptureBackend() else null, .flat_batch_policy = p.flat_batch_policy, .max_local_weight = p.max_local_weight, .geometry_storage = p.geometry_storage, .geometry_budget_percent = p.geometry_budget_percent, .fallback_geometry_budget_percent = p.fallback_geometry_budget_percent, .fallback_authorize_context = if (fallback_diagnostic) &fallback_authorization else null, .fallback_authorize_fn = if (fallback_diagnostic) FallbackAuthorization.authorize else null, .geometry_chooser_policy = p.geometry_chooser_policy, .validation_backend = switch (p.validation_policy) {
        .scalar_reference => null,
        .accelerate_fp64_summary_v1 => validation_accelerate.backend(&validation_context),
    }, .training_reduction_policy = p.training_reduction_policy, .training_reduction_query_batch_rows = p.training_reduction_query_batch_rows, .gram_tile_rows = p.gram_delta_tile_rows, .gram_backend = if (p.geometry_storage == .per_query_gram) gram_backend.backend(&gram_context) else null, .gram_delta_backend = if (p.gram_delta_accelerated) gram_delta.accelerateBackend() orelse return error.AcceleratedDeltaUnsupported else null, .gram_delta_require_accelerated = p.gram_delta_accelerated, .gram_memoize_original_delta = p.gram_memoize_original_delta, .joint_fixed_original_memo = p.joint_fixed_original_memo, .train_budget_exchange = p.train_budget_exchange, .train_snapshot_context = if (typed_future_snapshot) &snapshot_v2_context else if (p.train_snapshot_output_dir != null) &snapshot_context else null, .train_snapshot_fn = if (p.train_snapshot_output_dir != null and !typed_future_snapshot) SnapshotContext.writeSnapshot else null, .train_snapshot_v2_domain_runs = snapshot_v2_runs, .train_snapshot_v2_fn = if (typed_future_snapshot) SnapshotV2Context.writeSnapshot else null }) catch |err| {
        if (fallback_diagnostic and !p.fallback_authorize.? and err == error.CompactionPhysicalBudgetUnsupported) {
            const after = try sourceDigest(input.checkpoint_owned.state);
            if (fallback_authorization.calls != 1 or !std.mem.eql(u8, &source_before, &after)) return error.InvalidWholeFitInput;
            const capture = fitter.trainingCaptureCounters();
            const dispatches = fitter.dispatchCount();
            const attention_calls = fitter.attentionCallCount();
            const solves = fitter.deinitAndTakeValueSolves();
            live = false;
            const failure_body = try std.json.Stringify.valueAlloc(a, .{ .schema = "antfly.consumed-native-head-replay-result.v28", .status = "expected_failure_fallback_authorization_denied", .expected_error = "CompactionPhysicalBudgetUnsupported", .plan_sha256 = std.fmt.bytesToHex(plan_digest, .lower), .donor = p.donor, .head = p.head, .primary_geometry_budget_percent = p.geometry_budget_percent, .fallback_geometry_budget_percent = p.fallback_geometry_budget_percent, .fallback_authorized = false, .fallback_authorization_calls = fallback_authorization.calls, .fitter_dispatches = dispatches, .fitter_attention_calls = attention_calls, .fitter_value_solves = solves, .fitter_capture = capture, .source_unchanged = true, .source_before_sha256 = std.fmt.bytesToHex(source_before, .lower), .source_after_sha256 = std.fmt.bytesToHex(after, .lower), .diagnostic_only = true, .quality_credit = false, .production_qualified = false }, .{});
            defer a.free(failure_body);
            var failure_file = try std.Io.Dir.createFileAbsolute(io, p.result_path, .{ .exclusive = true, .truncate = false });
            defer failure_file.close(io);
            try failure_file.writeStreamingAll(io, failure_body);
            try failure_file.writeStreamingAll(io, "\n");
            try failure_file.sync(io);
            return;
        }
        const snapshot_digest = if (typed_future_snapshot) snapshot_v2_context.manifest_digest else snapshot_context.manifest_digest;
        if (err != error.TrainSnapshotExported or p.train_snapshot_output_dir == null or snapshot_digest == null) return err;
        const after = try sourceDigest(input.checkpoint_owned.state);
        if (!std.mem.eql(u8, &source_before, &after)) return error.InvalidWholeFitInput;
        const body = try std.json.Stringify.valueAlloc(a, .{ .schema = "antfly.consumed-native-head-replay-result.v10", .plan_sha256 = std.fmt.bytesToHex(plan_digest, .lower), .train_snapshot_output_dir = p.train_snapshot_output_dir.?, .train_snapshot_manifest_sha256 = std.fmt.bytesToHex(snapshot_digest.?, .lower), .train_snapshot_training_only = true, .heldout_evaluated = false, .source_unchanged = true, .source_before_sha256 = std.fmt.bytesToHex(source_before, .lower), .source_after_sha256 = std.fmt.bytesToHex(after, .lower), .train_queries = train_elements / dim, .geometry_policy = p.geometry_policy, .geometry_storage = p.geometry_storage, .diagnostic_only = true, .quality_credit = false, .production_qualified = false }, .{});
        defer a.free(body);
        var stop_file = try std.Io.Dir.createFileAbsolute(io, p.result_path, .{ .exclusive = true, .truncate = false });
        defer stop_file.close(io);
        try stop_file.writeStreamingAll(io, body);
        try stop_file.writeStreamingAll(io, "\n");
        try stop_file.sync(io);
        return;
    };
    defer result.deinit();
    const elapsed = platform.time.monotonicNs() -| started;
    const source_after = try sourceDigest(input.checkpoint_owned.state);
    if (!std.mem.eql(u8, &source_before, &source_after) or !std.mem.eql(u8, result.geometry_policy, p.geometry_policy) or result.geometry_storage != p.geometry_storage or result.geometry_workspace_bound_bytes == 0 or result.geometry_workspace_bound_bytes > p.max_workspace_bytes) return error.InvalidWholeFitInput;
    const expected_component_geometry = switch (p.geometry_chooser_policy) {
        .legacy_full_scan => gram.policy,
        .incremental_gram_v1 => if (p.geometry_budget_percent == 100) gram.incremental_policy else gram.incremental_budget_percent_policy,
    };
    if (component_ab and !std.mem.eql(u8, result.geometry_policy, expected_component_geometry)) return error.InvalidWholeFitInput;
    if (component_ab and result.incremental_gram_chooser_calls != @as(usize, if (p.geometry_chooser_policy == .incremental_gram_v1) 3 else 0)) return error.InvalidWholeFitInput;
    const expected_geometry_budget_percent: u8 = if (fallback_diagnostic) p.fallback_geometry_budget_percent else p.geometry_budget_percent;
    if (result.geometry_budget_percent != expected_geometry_budget_percent) return error.InvalidWholeFitInput;
    if (!p.bounded_page_route) {
        const train_queries = train_elements / dim;
        const expected_attention_calls = std.math.divCeil(usize, train_queries, p.query_batch_rows) catch return error.InvalidWholeFitInput;
        if (result.training_source.counters.query_batch_rows != p.query_batch_rows or result.training_source.counters.attention_calls != expected_attention_calls) return error.InvalidWholeFitInput;
    }
    const expected_gram_attempts: usize = if (fallback_diagnostic) 2 else 1;
    if (gram_context.successful_calls != (if (!p.bounded_page_route and p.geometry_storage == .per_query_gram) try std.math.mul(usize, train_elements / dim, expected_gram_attempts) else @as(usize, 0))) return error.InvalidWholeFitInput;
    if (fallback_diagnostic and (fallback_authorization.calls != 1 or !result.fallback_reuse.fallback_attempted or result.fallback_reuse.live_training_sources != 1 or result.fallback_reuse.scored_bank_builds != 1 or result.fallback_reuse.geometry_attempts != 2 or result.fallback_reuse.final_validation_attempts != 2 or result.fallback_reuse.authorization_calls != 1 or !result.fallback_reuse.primary_rejected_result_destroyed or result.fallback_reuse.scored_packets == 0 or result.fallback_reuse.reselections_avoided != result.fallback_reuse.scored_packets or result.fallback_reuse.value_fit_dispatches_avoided != result.fallback_reuse.scored_packets)) return error.InvalidWholeFitInput;
    if (!p.bounded_page_route and (result.training_source.counters.probability_storage != p.probability_storage or result.weight_cap.configured != p.max_local_weight)) return error.InvalidWholeFitInput;
    if (p.training_capture_backend) {
        const expected_tiles = std.math.divCeil(usize, train_elements / dim, p.query_batch_rows) catch return error.InvalidWholeFitInput;
        if (result.training_source.counters.capture_begins != 1 or result.training_source.counters.capture_ends != 1 or result.training_source.counters.capture_tiles != expected_tiles) return error.InvalidWholeFitInput;
    } else if (result.training_source.counters.capture_begins != 0 or result.training_source.counters.capture_tiles != 0 or result.training_source.counters.capture_ends != 0) return error.InvalidWholeFitInput;
    if (result.training_reduction.policy != p.training_reduction_policy or
        (reduction_accelerated and (!result.training_reduction.accelerated or result.training_reduction.local_original_calls == 0 or result.training_reduction.rounded_harm_calls == 0 or result.training_reduction.tiles == 0 or result.training_reduction.backend_calls != try std.math.mul(usize, result.training_reduction.tiles, 2) or result.training_reduction.workspace_bytes_peak == 0)) or
        (!reduction_accelerated and (result.training_reduction.accelerated or result.training_reduction.local_original_calls != 0 or result.training_reduction.rounded_harm_calls != 0 or result.training_reduction.tiles != 0 or result.training_reduction.backend_calls != 0 or result.training_reduction.workspace_bytes_peak != 0)))
        return error.InvalidWholeFitInput;
    if (result.final_refit.strategy != p.final_refit_strategy or
        (p.final_refit_strategy == .identity_reuse and (result.final_refit.attempts != 0 or result.final_refit.completions != 0)) or
        (p.final_refit_strategy == .global_active_nnls_v16 and (result.final_refit.attempts == 0 or result.final_refit.completions == 0 or result.final_refit.completions > result.final_refit.attempts)))
        return error.InvalidWholeFitInput;
    if (!nonzeroDigest(result.frozen_support_bank_digest)) return error.InvalidWholeFitInput;
    if (!result.scored_candidates_reused or result.scored_packets_reused == 0 or result.reselections_avoided != result.scored_packets_reused or
        (p.final_refit_strategy == .identity_reuse) != (result.value_fit_dispatches_avoided > 0))
        return error.InvalidWholeFitInput;
    if (p.bounded_page_route and (bounded_evidence == null or bounded_evidence.?.provider_calls == 0 or bounded_evidence.?.provider_rows == 0 or bounded_evidence.?.workspace_bound_bytes == 0 or bounded_evidence.?.workspace_bound_bytes > p.max_workspace_bytes)) return error.InvalidWholeFitInput;
    if (!try heldoutGateEvidenceValid(result.branches, result.heldout_gate_calls, p.bounded_page_route)) return error.InvalidWholeFitInput;
    if (component_ab and ((p.validation_policy == .accelerate_fp64_summary_v1) != (validation_context.successfulCalls() > 0))) return error.InvalidWholeFitInput;
    if (bounded_evidence) |evidence| {
        if (evidence.geometry_delta_accelerated != p.bounded_delta_accelerated or evidence.geometry_delta_fallbacks != 0)
            return error.InvalidWholeFitInput;
        if (p.bounded_delta_accelerated) {
            const expected_bounded_delta_calls = std.math.mul(usize, evidence.geometry_delta_tiles, 2) catch return error.InvalidWholeFitInput;
            if (evidence.geometry_delta_tiles == 0 or evidence.geometry_delta_backend_calls != expected_bounded_delta_calls or
                evidence.geometry_delta_packed_bytes_peak == 0 or evidence.geometry_delta_staged_bytes_peak == 0)
                return error.InvalidWholeFitInput;
        } else if (evidence.geometry_delta_tiles != 0 or evidence.geometry_delta_backend_calls != 0 or
            evidence.geometry_delta_packed_bytes_peak != 0 or evidence.geometry_delta_staged_bytes_peak != 0)
            return error.InvalidWholeFitInput;
    } else if (p.bounded_delta_accelerated) return error.InvalidWholeFitInput;
    if ((p.train_snapshot_output_dir != null) != (result.train_snapshot_manifest_digest != null)) return error.InvalidWholeFitInput;
    const fit_middle = state.donors[p.donor].heads[p.head].positions.len - input.fit.prefix_rows - input.fit.recent_rows;
    const range_count = std.math.divCeil(usize, fit_middle, p.branch_chunk_rows) catch return error.InvalidWholeFitInput;
    const delta_tiles_per_range = std.math.divCeil(usize, train_elements / dim, p.gram_delta_tile_rows) catch return error.InvalidWholeFitInput;
    const expected_delta_tiles = try std.math.mul(usize, try std.math.mul(usize, delta_tiles_per_range, range_count), 3);
    const expected_delta_calls = try std.math.mul(usize, expected_delta_tiles, 2);
    if (result.gram_delta.requested != p.gram_delta_accelerated or result.gram_delta.accelerated != p.gram_delta_accelerated or
        result.gram_delta.tile_rows != (if (p.gram_delta_accelerated) p.gram_delta_tile_rows else 0) or
        result.gram_delta.tiles != (if (p.gram_delta_accelerated) expected_delta_tiles else 0) or
        result.gram_delta.backend_calls != (if (p.gram_delta_accelerated) expected_delta_calls else 0) or result.gram_delta.fallbacks != 0 or
        (p.gram_delta_accelerated and (result.gram_delta.packed_bytes_peak == 0 or result.gram_delta.staged_bytes_peak == 0 or result.gram_delta.page_bytes_peak == 0))) return error.InvalidWholeFitInput;
    const memo_tiles = std.math.divCeil(usize, train_elements / dim, 8) catch return error.InvalidWholeFitInput;
    const expected_memo_misses = try std.math.mul(usize, memo_tiles, range_count);
    const expected_memo_hits = try std.math.mul(usize, expected_memo_misses, 2);
    const memo_row_values = try std.math.add(usize, dim, 1);
    const expected_memo_bytes = try std.math.mul(usize, @min(@as(usize, 8), train_elements / dim), try std.math.mul(usize, memo_row_values, @sizeOf(f64)));
    if (result.gram_memo.enabled != p.gram_memoize_original_delta or
        result.gram_memo.bytes != (if (p.gram_memoize_original_delta) expected_memo_bytes else 0) or
        result.gram_memo.hits != (if (p.gram_memoize_original_delta) expected_memo_hits else 0) or
        result.gram_memo.misses != (if (p.gram_memoize_original_delta) expected_memo_misses else 0)) return error.InvalidWholeFitInput;
    const expected_joint_bytes = try std.math.mul(usize, train_elements / dim, try std.math.mul(usize, memo_row_values, @sizeOf(f64)));
    if (result.joint_fixed_memo.enabled != p.joint_fixed_original_memo or
        result.joint_fixed_memo.bytes != (if (p.joint_fixed_original_memo) expected_joint_bytes else 0) or
        result.joint_fixed_memo.builds != (if (p.joint_fixed_original_memo) range_count else 0) or
        result.joint_fixed_memo.hits != (if (p.joint_fixed_original_memo) try std.math.mul(usize, range_count, 3) else 0)) return error.InvalidWholeFitInput;
    if (p.observe_phase_timing and (phase_timing.cache_prepare_ns == 0 or phase_timing.local_original_rho_y_ns == 0 or
        phase_timing.selection_freeze_ns == 0 or phase_timing.joint_value_fit_ns == 0 or
        phase_timing.rounded_harm_ns == 0 or phase_timing.chooser_ns == 0 or phase_timing.final_heldout_ns == 0 or
        (p.geometry_storage == .per_query_gram and phase_timing.gram_construction_ns == 0))) return error.InvalidWholeFitInput;
    const expected_scored_candidates = try std.math.mul(usize, std.math.divCeil(usize, fit_middle, p.branch_chunk_rows) catch return error.InvalidWholeFitInput, 3);
    if (result.weight_cap.scored_candidates != (if (p.max_local_weight == null) 0 else expected_scored_candidates)) return error.InvalidWholeFitInput;
    if (!std.math.isFinite(result.weight_cap.max_before) or !std.math.isFinite(result.weight_cap.max_after) or
        result.weight_cap.max_before < 0 or result.weight_cap.max_after < 0 or result.weight_cap.max_before < result.weight_cap.max_after) return error.InvalidWholeFitInput;
    if (p.max_local_weight) |cap| {
        if (result.weight_cap.max_after > cap or (result.weight_cap.clamped_weights > 0) != (result.weight_cap.max_before > cap)) return error.InvalidWholeFitInput;
    } else if (result.weight_cap.clamped_weights != 0 or result.weight_cap.max_before != 0 or result.weight_cap.max_after != 0) return error.InvalidWholeFitInput;
    var origins: [3]?QueryOrigin = @splat(null);
    if (result.branches.len > origins.len) return error.InvalidWholeFitInput;
    for (result.branches, 0..) |branch, index| {
        const nominal_cap = std.math.divCeil(usize, fit_middle, branch.divisor) catch return error.InvalidWholeFitInput;
        const expected_cap = if (result.geometry_budget_percent != 100 and branch.divisor == 2)
            (std.math.mul(usize, nominal_cap, result.geometry_budget_percent) catch return error.InvalidWholeFitInput) / 100
        else
            nominal_cap;
        const unavailable = std.mem.eql(u8, branch.rejection, "geometry_budget_infeasible");
        if (branch.geometry_middle_cap_rows != expected_cap or
            (unavailable and (branch.geometry_middle_rows <= expected_cap or branch.geometry_targets.len != range_count or branch.mass_error != null or branch.output_rmse != null)) or
            (!unavailable and branch.geometry_middle_rows > expected_cap)) return error.InvalidWholeFitInput;
        if (!observationShapeValid(branch.rejection, branch.mass_error != null, branch.output_rmse != null, branch.direct_train_metrics != null, branch.heldout_detail != null, p.observe_direct_train, p.observe_validation_detail)) return error.InvalidWholeFitInput;
        if (branch.heldout_detail) |detail| {
            origins[index] = try heldoutOrigin(&input, p.donor, p.head, detail.validation.worst_query_index);
        }
    }
    const fitter_dispatches = fitter.dispatchCount();
    const fitter_attention_calls = fitter.attentionCallCount();
    const fitter_capture = fitter.trainingCaptureCounters();
    const expected_capture_allocations: usize = if (p.training_capture_backend) 5 else 0;
    const expected_capture_frees: usize = if (p.training_capture_backend) 5 else 0;
    if (fitter_dispatches == 0 or fitter_attention_calls == 0 or
        fitter_capture.allocations != expected_capture_allocations or
        fitter_capture.frees != expected_capture_frees)
        return error.InvalidWholeFitInput;
    const solves = fitter.deinitAndTakeValueSolves();
    live = false;
    if ((high_fanout_bounded and (solves != 0 or tsqr_context.calls == 0)) or
        (!high_fanout_bounded and solves == 0) or elapsed == 0 or result.training_source.outcome != .ready) return error.InvalidWholeFitInput;
    const high_fanout_result_schema = "antfly.consumed-native-head-replay-result.v22";
    const base_body = try std.json.Stringify.valueAlloc(a, .{
        .schema = if (fallback_diagnostic) "antfly.consumed-native-head-replay-result.v28" else if (historical_override) "antfly.consumed-native-head-replay-result.v30" else if (peak_support_centered_ridge_selection) "antfly.consumed-native-head-replay-result.v29" else if (peak_x16_centered_ridge_selection) "antfly.consumed-native-head-replay-result.v27" else if (peak_augmented_x16_selection) "antfly.consumed-native-head-replay-result.v26" else if (peak_augmented_selection) "antfly.consumed-native-head-replay-result.v25" else if (normalized_selection) "antfly.consumed-native-head-replay-result.v24" else if (high_fanout_bounded) high_fanout_result_schema else if (reduction_experiment) "antfly.consumed-native-head-replay-result.v21" else if (p.training_capture_experiment) "antfly.consumed-native-head-replay-result.v20" else if (component_ab) "antfly.consumed-native-head-replay-result.v19" else if (p.bounded_delta_accelerated) "antfly.consumed-native-head-replay-result.v18" else if (generalized_budget) "antfly.consumed-native-head-replay-result.v16" else if (p.gram_delta_accelerated) "antfly.consumed-native-head-replay-result.v14" else if (p.geometry_budget_percent == 95) "antfly.consumed-native-head-replay-result.v12" else if (p.train_snapshot_output_dir != null) "antfly.consumed-native-head-replay-result.v10" else if (p.final_refit_strategy != .identity_reuse) "antfly.consumed-native-head-replay-result.v8" else if (p.bounded_page_route) "antfly.consumed-native-head-replay-result.v7" else "antfly.consumed-native-head-replay-result.v6",
        .fallback_diagnostic = fallback_diagnostic,
        .primary_geometry_budget_percent = p.geometry_budget_percent,
        .fallback_geometry_budget_percent = p.fallback_geometry_budget_percent,
        .fallback_authorized = p.fallback_authorize,
        .fallback_authorization_calls = fallback_authorization.calls,
        .fallback_reuse = result.fallback_reuse,
        .selection_score = @tagName(selection_policy),
        .selection_policy = "antfly.global-mass-selection-score.v1",
        .selection_policy_sha256 = std.fmt.bytesToHex(global_selection.policyDigest(selection_policy), .lower),
        .selection_peak_row_multiplicity = @as(usize, if (peak_augmented_x16_selection or peak_x16_centered_ridge_selection) 16 else if (peak_augmented_selection) 1 else 0),
        .selection_fixed_support_mass_refit = peak_x16_centered_ridge_selection or peak_support_centered_ridge_selection,
        .selection_mass_ridge_ratio = @as(?f64, if (peak_x16_centered_ridge_selection or peak_support_centered_ridge_selection) 0.01 else null),
        .selection_mass_ridge_center = @as(?f64, if (peak_x16_centered_ridge_selection or peak_support_centered_ridge_selection) 1.0 else null),
        .training_route = @tagName(p.training_route),
        .training_route_policy = if (high_fanout_bounded) tsqr_value.policy else null,
        .raw_selection_policy = if (bounded_evidence) |evidence| evidence.raw_selection_policy else null,
        .raw_selection_calls = if (bounded_evidence) |evidence| evidence.raw_selection_calls else 0,
        .raw_selection_probability_values = if (bounded_evidence) |evidence| evidence.raw_selection_probability_values else 0,
        .raw_selection_workspace_peak_bytes = if (bounded_evidence) |evidence| evidence.raw_selection_workspace_peak_bytes else 0,
        .raw_selection_provenance_sha256 = if (bounded_evidence) |evidence| std.fmt.bytesToHex(evidence.raw_selection_provenance_sha256, .lower) else null,
        .tsqr_value_calls = tsqr_context.calls,
        .tsqr_value_workspace_peak_bytes = tsqr_context.workspace_peak_bytes,
        .plan_sha256 = std.fmt.bytesToHex(plan_digest, .lower),
        .input_manifest_sha256 = std.fmt.bytesToHex(manifest_digest, .lower),
        .worker_source_freeze_sha256 = p.worker_source_freeze_sha256,
        .worker_binary_sha256 = p.worker_binary_sha256,
        .worker_receipt_sha256 = p.worker_receipt_sha256,
        .worker_source_manifest_sha256 = p.worker_source_manifest_sha256,
        .executing_binary_verified = true,
        .worker_source_binding_scope = "source236_overlay_context_worker_build_v1",
        .source_before_sha256 = std.fmt.bytesToHex(source_before, .lower),
        .source_after_sha256 = std.fmt.bytesToHex(source_after, .lower),
        .source_unchanged = true,
        .export_source_freeze_sha256 = p.source_freeze_sha256,
        .export_binary_sha256 = p.binary_sha256,
        .exporter_sha256 = p.exporter_sha256,
        .identity = p.identity,
        .config_sha256 = p.config_sha256,
        .query_samples = p.query_samples,
        .workspace_limit_bytes = p.max_workspace_bytes,
        .head_dim = dim,
        .probability_storage = result.training_source.counters.probability_storage,
        .max_local_weight = result.weight_cap.configured,
        .weight_cap_policy = if (p.max_local_weight == null) "none" else local_weight_cap.policy,
        .weight_cap_scope = "all_scored_candidates_before_joint_value_fit",
        .weight_cap_scored_candidates = result.weight_cap.scored_candidates,
        .local_weights_clamped = result.weight_cap.clamped_weights,
        .max_local_weight_before = result.weight_cap.max_before,
        .max_local_weight_after = result.weight_cap.max_after,
        .train_queries = train_elements / dim,
        .primary_train_queries = primary_train_elements / dim,
        .augmentation_train_queries = augmentation_train_elements / dim,
        .dense_train_queries = dense_train_elements / dim,
        .future_train_queries = future_train_elements / dim,
        .dense_augmentation_manifest_sha256 = if (dense_domain) |domain| std.fmt.bytesToHex(domain.manifest_sha256, .lower) else null,
        .future_augmentation_manifest_sha256 = if (future_domain) |domain| std.fmt.bytesToHex(domain.manifest_sha256, .lower) else null,
        .independent_train_splits = if (independent_domains) [_]usize{ 256, 512 } else null,
        .heldout_queries = heldout_elements / dim,
        .query_batch_rows = p.query_batch_rows,
        .training_capture_experiment = p.training_capture_experiment,
        .training_capture_backend = p.training_capture_backend,
        .flat_batch_policy = p.flat_batch_policy,
        .flat_batch_tuple_sha256 = std.fmt.bytesToHex(flatBatchTupleDigest(p.flat_batch_policy, p.query_batch_rows, p.gram_delta_tile_rows, p.training_reduction_query_batch_rows), .lower),
        .training_reduction_experiment = p.training_reduction_experiment,
        .training_reduction_policy = p.training_reduction_policy,
        .training_reduction_query_batch_rows = p.training_reduction_query_batch_rows,
        .training_reduction_policy_sha256 = std.fmt.bytesToHex(trainingReductionPolicyDigest(p.training_reduction_policy, p.training_reduction_query_batch_rows), .lower),
        .training_reduction = result.training_reduction,
        .fitter_dispatches = fitter_dispatches,
        .fitter_attention_calls = fitter_attention_calls,
        .fitter_phase_counters = fitter.phaseCounters(),
        .training_capture_allocations = fitter_capture.allocations,
        .training_capture_frees = fitter_capture.frees,
        .geometry_policy = result.geometry_policy,
        .component_experiment = p.component_experiment,
        .geometry_chooser_policy = p.geometry_chooser_policy,
        .validation_policy = p.validation_policy,
        .component_policy_sha256 = std.fmt.bytesToHex(componentPolicyDigest(p.component_experiment, p.geometry_chooser_policy, p.validation_policy, p.geometry_budget_percent), .lower),
        .incremental_gram_chooser_calls = result.incremental_gram_chooser_calls,
        .fp64_validation_summary_calls = validation_context.successfulCalls(),
        .fp64_validation_dgemm_calls = try std.math.mul(usize, validation_context.successfulCalls(), 2),
        .geometry_storage = result.geometry_storage,
        .geometry_budget_percent = result.geometry_budget_percent,
        .gram_delta_accelerated = p.gram_delta_accelerated,
        .gram_delta_tile_rows = p.gram_delta_tile_rows,
        .gram_delta_policy = if (p.gram_delta_accelerated) gram_delta.policy else "scalar_ordered_reference_v1",
        .gram_delta = result.gram_delta,
        .geometry_workspace_bound_bytes = result.geometry_workspace_bound_bytes,
        .bounded_page_route = p.bounded_page_route,
        .bounded_delta_accelerated = p.bounded_delta_accelerated,
        .bounded_delta_tile_rows = if (p.bounded_delta_accelerated) p.bounded_tile_queries else 0,
        .bounded_delta_policy = if (p.bounded_delta_accelerated) "packed_fp64_accelerate_bounded_geometry_delta_v1" else "scalar_ordered_reference_v1",
        .bounded_delta_tiles = if (bounded_evidence) |e| e.geometry_delta_tiles else 0,
        .bounded_delta_backend_calls = if (bounded_evidence) |e| e.geometry_delta_backend_calls else 0,
        .bounded_delta_packed_bytes_peak = if (bounded_evidence) |e| e.geometry_delta_packed_bytes_peak else 0,
        .bounded_delta_staged_bytes_peak = if (bounded_evidence) |e| e.geometry_delta_staged_bytes_peak else 0,
        .bounded_delta_fallbacks = if (bounded_evidence) |e| e.geometry_delta_fallbacks else 0,
        .provider_page_calls = if (bounded_evidence) |e| e.provider_calls else 0,
        .provider_page_rows = if (bounded_evidence) |e| e.provider_rows else 0,
        .bounded_workspace_bytes = if (bounded_evidence) |e| e.workspace_bound_bytes else 0,
        .probability_matrix_bytes = 0,
        .gram_representation_bytes = 0,
        .heldout_gates = result.heldout_gate_calls,
        .train_snapshot_output_dir = p.train_snapshot_output_dir,
        .train_snapshot_manifest_sha256 = if (result.train_snapshot_manifest_digest) |snapshot_digest| std.fmt.bytesToHex(snapshot_digest, .lower) else null,
        .train_snapshot_training_only = p.train_snapshot_output_dir != null,
        .geometry_backend = if (p.geometry_storage == .per_query_gram) "accelerate_fp64_packed_gram_v1" else "none",
        .geometry_backend_calls = gram_context.successful_calls,
        .gram_memoize_original_delta = result.gram_memo.enabled,
        .gram_memo_bytes = result.gram_memo.bytes,
        .gram_memo_hits = result.gram_memo.hits,
        .gram_memo_misses = result.gram_memo.misses,
        .joint_fixed_original_memo = result.joint_fixed_memo.enabled,
        .joint_fixed_original_memo_bytes = result.joint_fixed_memo.bytes,
        .joint_fixed_original_memo_builds = result.joint_fixed_memo.builds,
        .joint_fixed_original_memo_hits = result.joint_fixed_memo.hits,
        .train_budget_exchange = p.train_budget_exchange,
        .final_refit = result.final_refit,
        .scored_candidates_reused = result.scored_candidates_reused,
        .scored_packets_reused = result.scored_packets_reused,
        .reselections_avoided = result.reselections_avoided,
        .value_fit_dispatches_avoided = result.value_fit_dispatches_avoided,
        .observe_phase_timing = p.observe_phase_timing,
        .phase_timing = if (p.observe_phase_timing) phase_timing else null,
        .augmentation_used = p.augmentation != null or independent_domains,
        .augmentation_policy = if (p.augmentation) |binding| binding.training_policy orelse if (binding.samples_per_consumer == 320) "generated_self_study_v2" else "study_pass1_portfolio" else "none",
        .augmentation_manifest_sha256 = if (augmentation_digest) |value| std.fmt.bytesToHex(value, .lower) else null,
        .augmentation_query_samples = if (p.augmentation) |binding| @as(usize, binding.samples_per_consumer) else 0,
        .observe_direct_train = p.observe_direct_train,
        .observe_validation_detail = p.observe_validation_detail,
        .timing_scope = timingScope(p),
        .heldout_worst_query_origins = origins[0..result.branches.len],
        .donor = p.donor,
        .head = p.head,
        .accepted = result.accepted,
        .chosen_divisor = result.chosen_divisor,
        .rows_before = input.checkpoint_owned.state.donors[p.donor].heads[p.head].positions.len,
        .rows_after = result.head.rows,
        .source_head_physical_bytes = try physical_budget.ordinaryHeadBytes(input.checkpoint_owned.state.donors[p.donor].heads[p.head].positions.len, dim),
        .candidate_head_physical_bytes = if (result.accepted) try physical_budget.candidateHeadBytes(result.head.rows, dim) else try physical_budget.ordinaryHeadBytes(input.checkpoint_owned.state.donors[p.donor].heads[p.head].positions.len, dim),
        .final_metrics_valid = result.final_metrics_valid,
        .final_mass_error = if (result.final_metrics_valid) result.final_mass_error else null,
        .final_output_rmse = if (result.final_metrics_valid) result.final_output_rmse else null,
        .branches = result.branches,
        .training_source = result.training_source,
        .head_sha256 = std.fmt.bytesToHex(headDigest(result.head), .lower),
        .selection_map_sha256 = std.fmt.bytesToHex(mapDigest(result.source_indices, result.coverage_source_indices), .lower),
        .frozen_support_bank_sha256 = std.fmt.bytesToHex(result.frozen_support_bank_digest, .lower),
        .accelerated_value_solves = solves,
        .elapsed_ns = elapsed,
        .diagnostic_only = true,
        .quality_credit = false,
        .production_qualified = false,
        .publication_claim = false,
    }, .{});
    defer a.free(base_body);
    var body = base_body;
    if (override_evidence) |e| {
        const suffix = try std.fmt.allocPrint(a, ",\"historical_training_override_policy\":\"normalized_q_cosine_kcenter_v1\",\"historical_training_override_manifest_sha256\":\"{s}\",\"historical_training_override_policy_sha256\":\"{s}\",\"historical_training_override_domain_sha256\":\"{s}\",\"historical_training_override_eligible_indices_sha256\":\"{s}\",\"historical_training_override_selection_indices_sha256\":\"{s}\",\"historical_training_override_dense_outer_manifest_sha256\":\"{s}\",\"historical_training_override_dense_nested_manifest_sha256\":\"{s}\",\"historical_training_override_v4_proof_sha256\":\"{s}\",\"historical_training_override_policy_manifest_sha256\":\"{s}\",\"actual_training_mix_policy_sha256\":\"{s}\"}}", .{ std.fmt.bytesToHex(e.manifest, .lower), std.fmt.bytesToHex(e.policy, .lower), std.fmt.bytesToHex(e.derived, .lower), std.fmt.bytesToHex(e.eligible, .lower), std.fmt.bytesToHex(e.selection, .lower), std.fmt.bytesToHex(e.dense_outer, .lower), std.fmt.bytesToHex(e.dense_nested, .lower), std.fmt.bytesToHex(e.v4_proof, .lower), std.fmt.bytesToHex(e.policy_manifest, .lower), std.fmt.bytesToHex(e.mix, .lower) });
        defer a.free(suffix);
        body = try std.mem.concat(a, u8, &.{ base_body[0 .. base_body.len - 1], suffix });
    }
    defer if (body.ptr != base_body.ptr) a.free(body);
    var file = try std.Io.Dir.createFileAbsolute(io, p.result_path, .{ .exclusive = true, .truncate = false });
    defer file.close(io);
    try file.writeStreamingAll(io, body);
    try file.writeStreamingAll(io, "\n");
    try file.sync(io);
}

test "native geometry policy seals TRAIN exchange only on explicit flag" {
    try std.testing.expectEqualStrings(adaptive.signed_joint_policy, expectedGeometryPolicy(.dense, false));
    try std.testing.expectEqualStrings(gram.policy, expectedGeometryPolicy(.per_query_gram, false));
    try std.testing.expectEqualStrings(gram.exchange_policy, expectedGeometryPolicy(.per_query_gram, true));
}
