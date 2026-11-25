// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const platform = @import("antfly_platform");
const checkpoint = @import("attention_match_checkpoint.zig");
const whole = @import("attention_match_whole_fit_input.zig");
const augmentation = @import("attention_match_training_augmentation.zig");
const replay = @import("attention_match_joint_chunk_mass_replay.zig");
const adaptive = @import("attention_match_joint_adaptive_geometry.zig");

const workspace_bytes: usize = 512 * 1024 * 1024;

const Plan = struct {
    schema: []const u8,
    input_dir: []const u8,
    augmentation_dir: []const u8,
    identity: struct { weights: []const u8, tokenizer: []const u8, template: []const u8, layout: []const u8 },
    config_sha256: []const u8,
    source_freeze_sha256: []const u8,
    binary_sha256: []const u8,
    exporter_sha256: []const u8,
    input_manifest_sha256: []const u8,
    augmentation_manifest_sha256: []const u8,
    fixed_source65_manifest_sha256: []const u8,
    parent_study_tokens_sha256: []const u8,
    study_tokens_sha256: []const u8,
    method: []const u8,
    query_samples: u16,
    diagnostic_donor: usize,
    diagnostic_head: usize,
    result_path: []const u8,
    max_workspace_bytes: usize,
    branch_ladder: [3]usize,
    coefficient_policy: []const u8,
    branch_chunk_rows: usize,
    value_policy: []const u8,
    mass_gate: f64,
    output_gate: f64,
    heldout_evaluations_per_branch: usize,
    heldout_controls_fit: bool,
    training_source_policy: []const u8 = "legacy",
    mass_strategy: []const u8 = "global_active_nnls_v16",
    retain_training_diagnostics: bool = false,
    query_batch_rows: usize = 16,
    flat_training_cache: bool = false,
    source_authority: []const u8 = "source72",
    geometry_policy: []const u8 = adaptive.signed_joint_policy,
    geometry_budget_policy: []const u8 = "ceil_eligible_divisor_cap_positive_gain_v1",
    training_score_policy: []const u8 = "cache_gpu_probability_v1_heuristic",
    heldout_validation_policy: []const u8 = "backend_null_fp64_v1",
    identity_rows_exact: bool = true,
    diagnostic_only: bool = true,
    quality_credit: bool = false,
    production_qualified: bool = false,
};

fn parseDigest(text: []const u8) ![32]u8 {
    if (text.len != 64) return error.InvalidWholeFitInput;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, text) catch return error.InvalidWholeFitInput;
    return out;
}

fn policyShape(plan: Plan) bool {
    const v18 = std.mem.eql(u8, plan.schema, "antfly.attention-match-joint-chunk-mass-plan.v18") and std.mem.eql(u8, plan.mass_strategy, "identity_chunk_coefficients_v18") and !plan.retain_training_diagnostics and std.mem.eql(u8, plan.coefficient_policy, "identity_chunk_coefficients_v18");
    const cache_pair = plan.flat_training_cache and std.mem.eql(u8, plan.training_source_policy, "flat_exact_tier_a_mandatory") and plan.query_batch_rows == 16;
    return v18 and cache_pair and std.mem.eql(u8, plan.geometry_policy, adaptive.signed_joint_policy) and std.mem.eql(u8, plan.geometry_budget_policy, "ceil_eligible_divisor_cap_positive_gain_v1") and std.mem.eql(u8, plan.training_score_policy, "cache_gpu_probability_v1_heuristic") and std.mem.eql(u8, plan.heldout_validation_policy, "backend_null_fp64_v1") and plan.identity_rows_exact and
        std.mem.eql(u8, plan.source_authority, "source72") and plan.diagnostic_only and !plan.quality_credit and !plan.production_qualified and
        std.fs.path.isAbsolute(plan.input_dir) and std.fs.path.isAbsolute(plan.augmentation_dir) and std.fs.path.isAbsolute(plan.result_path) and
        std.mem.eql(u8, plan.method, "top1") and plan.query_samples == 512 and plan.diagnostic_donor == 5 and plan.diagnostic_head == 1 and
        plan.max_workspace_bytes == workspace_bytes and std.mem.eql(usize, &plan.branch_ladder, &.{ 4, 3, 2 }) and
        plan.branch_chunk_rows == 256 and
        std.mem.eql(u8, plan.value_policy, "fixed_corrected_weights_local_value_ls_v1") and
        plan.mass_gate == 0.05 and plan.output_gate == 0.05 and plan.heldout_evaluations_per_branch == 1 and !plan.heldout_controls_fit;
}

fn hashU64(hasher: *std.crypto.hash.sha2.Sha256, value: u64) void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, value, .little);
    hasher.update(&bytes);
}

fn finalHeadDigest(head: @import("attention_match.zig").Head) [32]u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hashU64(&hasher, @intCast(head.rows));
    hashU64(&hasher, @intCast(head.dim));
    hashU64(&hasher, @bitCast(head.scale));
    hashU64(&hasher, @bitCast(head.softcap));
    for (head.keys) |value| hashU64(&hasher, @bitCast(value));
    for (head.values) |value| hashU64(&hasher, @bitCast(value));
    if (head.bias) |bias| for (bias) |value| hashU64(&hasher, @bitCast(value));
    return hasher.finalResult();
}

test "joint chunk mass source72 augmented replay" {
    if (comptime builtin.os.tag != .macos or !build_options.enable_metal) return error.SkipZigTest;
    const plan_path = platform.env.getenv("ANTFLY_METAL_JOINT_CHUNK_MASS_PLAN") orelse return error.SkipZigTest;
    if (!std.fs.path.isAbsolute(plan_path)) return error.InvalidWholeFitInput;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, plan_path, allocator, .limited(1024 * 1024));
    defer allocator.free(bytes);
    const parsed = try std.json.parseFromSlice(Plan, allocator, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false });
    defer parsed.deinit();
    const plan = parsed.value;
    if (!policyShape(plan)) return error.InvalidWholeFitInput;
    var plan_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &plan_digest, .{});
    const manifest_path = try std.fs.path.join(allocator, &.{ plan.input_dir, "manifest.json" });
    defer allocator.free(manifest_path);
    const manifest = try std.Io.Dir.cwd().readFileAlloc(io, manifest_path, allocator, .limited(whole.manifest_reservation));
    defer allocator.free(manifest);
    var manifest_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(manifest, &manifest_digest, .{});
    if (!std.mem.eql(u8, &manifest_digest, &(try parseDigest(plan.input_manifest_sha256)))) return error.InvalidWholeFitInput;
    const augmentation_path = try std.fs.path.join(allocator, &.{ plan.augmentation_dir, "manifest.json" });
    defer allocator.free(augmentation_path);
    const augmentation_manifest = try std.Io.Dir.cwd().readFileAlloc(io, augmentation_path, allocator, .limited(64 * 1024));
    defer allocator.free(augmentation_manifest);
    var augmentation_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(augmentation_manifest, &augmentation_digest, .{});
    if (!std.mem.eql(u8, &augmentation_digest, &(try parseDigest(plan.augmentation_manifest_sha256)))) return error.InvalidWholeFitInput;
    const identity: checkpoint.Identity = .{ .weights = try parseDigest(plan.identity.weights), .tokenizer = try parseDigest(plan.identity.tokenizer), .template = try parseDigest(plan.identity.template), .layout = try parseDigest(plan.identity.layout) };
    var input = try whole.read(allocator, io, .{ .output_dir = plan.input_dir, .identity = identity, .config_sha256 = try parseDigest(plan.config_sha256), .source_freeze_sha256 = try parseDigest(plan.source_freeze_sha256), .binary_sha256 = try parseDigest(plan.binary_sha256), .exporter_sha256 = try parseDigest(plan.exporter_sha256), .limits = checkpoint.serving_limits });
    defer input.deinit();
    var augmented = try augmentation.read(allocator, io, .{ .output_dir = plan.augmentation_dir, .parent_manifest_sha256 = manifest_digest, .identity = identity, .source_freeze_sha256 = try parseDigest(plan.source_freeze_sha256), .binary_sha256 = try parseDigest(plan.binary_sha256), .exporter_sha256 = try parseDigest(plan.exporter_sha256), .parent_study_tokens_sha256 = try parseDigest(plan.parent_study_tokens_sha256), .study_tokens_sha256 = try parseDigest(plan.study_tokens_sha256) });
    defer augmented.deinit();
    if (input.query_samples != 512 or input.method != .top1) return error.InvalidWholeFitInput;
    const Fitter = @import("../../ops/metal_attention_match_fit.zig").Fitter;
    var fitter = try Fitter.init(allocator, null, workspace_bytes);
    var live = true;
    defer if (live) fitter.deinit();
    try fitter.setQueryBatchRows(plan.query_batch_rows);
    if (fitter.queryBatchRows() != plan.query_batch_rows) return error.InvalidWholeFitInput;
    const started = platform.time.monotonicNs();
    const cached = true;
    var result = try replay.runInputV18(allocator, &input, augmented.batches, fitter.backend(), plan.diagnostic_donor, plan.diagnostic_head, plan.branch_chunk_rows, workspace_bytes);
    defer result.deinit();
    if (!std.mem.eql(u8, plan.geometry_policy, result.geometry_policy)) return error.InvalidWholeFitInput;
    const actual_query_batch_rows = fitter.queryBatchRows();
    const final_head_sha256 = std.fmt.bytesToHex(finalHeadDigest(result.head), .lower);
    const accelerated_value_solves = fitter.deinitAndTakeValueSolves();
    live = false;
    const elapsed_ns = platform.time.monotonicNs() -| started;
    if (accelerated_value_solves == 0 or elapsed_ns == 0) return error.InvalidWholeFitInput;
    const json = try std.json.Stringify.valueAlloc(allocator, .{ .schema = "antfly.attention-match-joint-chunk-mass-result.v18", .authenticated = true, .plan_sha256 = std.fmt.bytesToHex(plan_digest, .lower), .input_manifest_sha256 = std.fmt.bytesToHex(manifest_digest, .lower), .augmentation_manifest_sha256 = std.fmt.bytesToHex(augmentation_digest, .lower), .fixed_source65_manifest_sha256 = plan.fixed_source65_manifest_sha256, .identity = plan.identity, .config_sha256 = plan.config_sha256, .parent_study_tokens_sha256 = plan.parent_study_tokens_sha256, .study_tokens_sha256 = plan.study_tokens_sha256, .source_freeze_sha256 = plan.source_freeze_sha256, .binary_sha256 = plan.binary_sha256, .exporter_sha256 = plan.exporter_sha256, .method = plan.method, .query_samples = plan.query_samples, .diagnostic_donor = plan.diagnostic_donor, .diagnostic_head = plan.diagnostic_head, .max_workspace_bytes = plan.max_workspace_bytes, .branch_ladder = plan.branch_ladder, .branch_chunk_rows = plan.branch_chunk_rows, .coefficient_policy = plan.coefficient_policy, .value_policy = plan.value_policy, .heldout_evaluations_per_branch = plan.heldout_evaluations_per_branch, .heldout_controls_fit = false, .mass_strategy = plan.mass_strategy, .training_source_policy = plan.training_source_policy, .geometry_policy = result.geometry_policy, .geometry_budget_policy = plan.geometry_budget_policy, .training_score_policy = plan.training_score_policy, .heldout_validation_policy = plan.heldout_validation_policy, .identity_rows_exact = plan.identity_rows_exact, .query_batch_rows = actual_query_batch_rows, .flat_training_cache_requested = cached, .flat_training_cache = result.training_source.outcome == .ready, .source_authority = plan.source_authority, .retain_training_diagnostics = plan.retain_training_diagnostics, .training_source = result.training_source, .final_head_sha256 = final_head_sha256, .accepted = result.accepted, .chosen_divisor = result.chosen_divisor, .rows_before = input.checkpoint_owned.state.donors[plan.diagnostic_donor].heads[plan.diagnostic_head].positions.len, .rows_after = result.head.rows, .final_metrics_valid = result.final_metrics_valid, .final_mass_error = if (result.final_metrics_valid) result.final_mass_error else null, .final_output_rmse = if (result.final_metrics_valid) result.final_output_rmse else null, .branches = result.branches, .accelerated_value_solves = accelerated_value_solves, .elapsed_ns = elapsed_ns, .diagnostic_only = true, .quality_credit = false, .production_qualified = false, .publication_claim = false }, .{});
    defer allocator.free(json);
    var file = try std.Io.Dir.createFileAbsolute(io, plan.result_path, .{ .exclusive = true, .truncate = false });
    defer file.close(io);
    try file.writeStreamingAll(io, json);
    try file.writeStreamingAll(io, "\n");
    try file.sync(io);
}

test "adaptive identity v18 policy shape is exact" {
    const digest = "0000000000000000000000000000000000000000000000000000000000000000";
    const good: Plan = .{ .schema = "antfly.attention-match-joint-chunk-mass-plan.v18", .input_dir = "/input", .augmentation_dir = "/augmentation", .identity = .{ .weights = digest, .tokenizer = digest, .template = digest, .layout = digest }, .config_sha256 = digest, .source_freeze_sha256 = digest, .binary_sha256 = digest, .exporter_sha256 = digest, .input_manifest_sha256 = digest, .augmentation_manifest_sha256 = digest, .fixed_source65_manifest_sha256 = digest, .parent_study_tokens_sha256 = digest, .study_tokens_sha256 = digest, .method = "top1", .query_samples = 512, .diagnostic_donor = 5, .diagnostic_head = 1, .result_path = "/result", .max_workspace_bytes = workspace_bytes, .branch_ladder = .{ 4, 3, 2 }, .branch_chunk_rows = 256, .coefficient_policy = "identity_chunk_coefficients_v18", .value_policy = "fixed_corrected_weights_local_value_ls_v1", .mass_gate = 0.05, .output_gate = 0.05, .heldout_evaluations_per_branch = 1, .heldout_controls_fit = false, .training_source_policy = "flat_exact_tier_a_mandatory", .mass_strategy = "identity_chunk_coefficients_v18", .query_batch_rows = 16, .flat_training_cache = true };
    try std.testing.expect(policyShape(good));
    var bad = good;
    bad.heldout_evaluations_per_branch = 2;
    try std.testing.expect(!policyShape(bad));
}
