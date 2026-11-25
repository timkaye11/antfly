// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
const std = @import("std");
const platform = @import("antfly_platform");
const replay = @import("runtime/kv/attention_match_production768_offline_replay.zig");
const checkpoint = @import("runtime/kv/attention_match_checkpoint.zig");
const future_training = @import("runtime/kv/attention_match_future_decode_training.zig");
const train_cv = @import("runtime/kv/attention_match_train_domain_cv_split.zig");
const train_challenge = @import("runtime/kv/attention_match_production768_train_challenge_rewrite.zig");

const Ref = struct { path: []const u8, sha256: [32]u8 };
const HistoricalCoverageJson = struct {
    dense_dir: []const u8,
    dense_outer_manifest: Ref,
    dense_nested_manifest: Ref,
    dense_reference_parent_manifest: Ref,
    dense_reference_parent_manifest_sha256: [32]u8,
    dense_parent_study_tokens_sha256: [32]u8,
    dense_study_tokens_sha256: [32]u8,
    maximum_workspace_bytes: usize = 128 * 1024 * 1024,
};
const BindingJson = struct {
    raw_selection_solver: replay.RawSelectionSolver = .householder,
    raw_selection_solver_explicit: bool = false,
    raw_selection_policy_sha256: ?[]const u8 = null,
    encoding: checkpoint.Encoding = .f16,
    rotated4_risk_retention: replay.Rotated4RiskRetention = .quarter_v1,
    train_challenge_historical_selection: bool = false,
    hybrid_coverage_anchor: bool = false,
    train_challenge_mass_only: bool = false,
    phase_profiling: bool = false,
    serial_pilot_fp64_value_backend: bool = false,
    serial_full_fp64_value_backend: bool = false,
    serial_full_fp64_control: bool = false,
    historical_coverage_probability_backend: replay.HistoricalCoverageProbabilityBackend = .scalar,
    train_primary_policy_comparison: bool = false,
    train_primary_policy_cross_validation: bool = false,
    train_exact_reservation_policy: replay.TrainExactReservationPolicy = .disabled,
    critical_core_policy: replay.CriticalCorePolicy = .train_probability_gte_0_025_no_top1_v1,
    postfit_policy: replay.PostfitPolicy = .disabled,
    rejected_detail_diagnostic: bool = false,
    train_probability_census: bool = false,
    performance_comparison_policy: replay.PerformanceComparisonPolicy = .legacy,
    historical_coverage: ?HistoricalCoverageJson = null,
    schema: []const u8,
    whole_fit_dir: []const u8,
    complete_future_dir: []const u8,
    whole_manifest: Ref,
    complete_manifest: Ref,
    producer_source_freeze: Ref,
    producer_binary: Ref,
    producer_profile: Ref,
    integrity_closure: Ref,
    identity: checkpoint.Identity,
    config_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    binary_sha256: [32]u8,
    exporter_sha256: [32]u8,
    complete_manifest_sha256: [32]u8,
    parent_manifest_sha256: [32]u8,
    historical_study_tokens_sha256: [32]u8,
    complete_authority_sha256: [32]u8,
    authority: @FieldType(replay.CompleteAuthority, "authority"),
    runtime_chunk_rows: usize,
    export_metadata_chunk_rows: usize,
    max_transient_host_bytes: usize,
    query_reservation_bytes: usize,
    diagnostic_only: bool,
    qualification_credit: bool,
    production_qualified: bool,
};

const range_shared_policy_sha256 = "5b3321f3dd61e13c915896c637b6f334239c1281dc2a04f0b9f00777ec62d5fb";

fn rawSelectionTupleValid(solver_value: replay.RawSelectionSolver, explicit: bool, policy_sha256: ?[]const u8, v16: bool) bool {
    if (!v16) return !explicit and solver_value == .householder and policy_sha256 == null;
    if (!explicit) return false;
    return switch (solver_value) {
        .householder => policy_sha256 == null,
        .range_shared_fp64_gram_original_kkt_v1 => if (policy_sha256) |value| std.mem.eql(u8, value, range_shared_policy_sha256) else false,
        .fp64_gram_advisory => false,
    };
}

test "production768 V15 defaults remain strict Householder" {
    try std.testing.expect(rawSelectionTupleValid(.householder, false, null, false));
    try std.testing.expect(!rawSelectionTupleValid(.range_shared_fp64_gram_original_kkt_v1, false, null, false));
}

test "production768 V16 accepts only sealed explicit raw solver tuples" {
    try std.testing.expect(rawSelectionTupleValid(.householder, true, null, true));
    try std.testing.expect(rawSelectionTupleValid(.range_shared_fp64_gram_original_kkt_v1, true, range_shared_policy_sha256, true));
    try std.testing.expect(!rawSelectionTupleValid(.range_shared_fp64_gram_original_kkt_v1, true, null, true));
    try std.testing.expect(!rawSelectionTupleValid(.range_shared_fp64_gram_original_kkt_v1, true, "00", true));
    try std.testing.expect(!rawSelectionTupleValid(.fp64_gram_advisory, true, null, true));
}

fn hashFile(path: []const u8) ![32]u8 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, std.testing.allocator, .limited(512 * 1024 * 1024));
    defer std.testing.allocator.free(bytes);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return out;
}

fn authenticate(ref: Ref) !void {
    const actual = try hashFile(ref.path);
    if (!std.fs.path.isAbsolute(ref.path) or !std.mem.eql(u8, &actual, &ref.sha256)) return error.InvalidOfflineReplayBinding;
}

fn valueDiagnosticValid(phase: bool, mass_only: bool, pilot: bool, control: bool, full: bool, hybrid: bool) bool {
    const count = @as(u8, @intFromBool(mass_only)) + @as(u8, @intFromBool(pilot)) + @as(u8, @intFromBool(control)) + @as(u8, @intFromBool(full));
    return (count != 0) == phase and count <= 1 and (count == 0 or hybrid);
}

test "context production768 authenticated complete-future offline replay diagnostic" {
    const binding_path = platform.env.getenv("ANTFLY_METAL_PRODUCTION768_OFFLINE_BINDING") orelse return error.SkipZigTest;
    const output_path = platform.env.getenv("ANTFLY_METAL_PRODUCTION768_OFFLINE_RESULT") orelse return error.InvalidOfflineReplayBinding;
    if (!std.fs.path.isAbsolute(binding_path) or !std.fs.path.isAbsolute(output_path)) return error.InvalidOfflineReplayBinding;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, binding_path, std.testing.allocator, .limited(256 * 1024));
    defer std.testing.allocator.free(bytes);
    var parsed = try std.json.parseFromSlice(BindingJson, std.testing.allocator, bytes, .{ .ignore_unknown_fields = false });
    defer parsed.deinit();
    const b = parsed.value;
    const v16 = std.mem.eql(u8, b.schema, "antfly.production768-offline-replay-binding.v16");
    const coverage_enabled = b.historical_coverage != null;
    const codec_diagnostic = b.encoding != .f16;
    const half_retention = b.rotated4_risk_retention == .half_v2;
    if (half_retention and b.encoding != .rotated4) return error.InvalidOfflineReplayBinding;
    if (codec_diagnostic and ((b.encoding != .fp8 and b.encoding != .rotated4) or !b.hybrid_coverage_anchor)) return error.InvalidOfflineReplayBinding;
    const planned_exact_enabled = b.train_exact_reservation_policy != .disabled;
    const primary_comparison_enabled = b.train_primary_policy_comparison;
    const primary_cv_enabled = b.train_primary_policy_cross_validation;
    const surrogate_minimax_enabled = b.postfit_policy == .train_global_residual_range_mass_l2_then_surrogate_partition_minimax_exact_rounded_v1;
    const surrogate_minimax_planned_exact = surrogate_minimax_enabled and planned_exact_enabled;
    const train_challenge_enabled = b.train_challenge_historical_selection;
    const hybrid_coverage_anchor = b.hybrid_coverage_anchor;
    const mass_only_diagnostic = b.train_challenge_mass_only;
    const serial_value_diagnostic = b.serial_pilot_fp64_value_backend;
    const full_value_diagnostic = b.serial_full_fp64_value_backend;
    const full_value_control = b.serial_full_fp64_control;
    const accelerated_coverage = b.historical_coverage_probability_backend == .accelerate_fp64_gather_eligible_v1;
    var historical_coverage_probability_backend_policy_sha256: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(replay.historical_coverage_accelerate_policy, &historical_coverage_probability_backend_policy_sha256, .{});
    var train_challenge_policy_sha256: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(train_challenge.policy, &train_challenge_policy_sha256, .{});
    if (primary_cv_enabled and !primary_comparison_enabled) return error.InvalidOfflineReplayBinding;
    if (surrogate_minimax_enabled and (primary_cv_enabled or primary_comparison_enabled)) return error.InvalidOfflineReplayBinding;
    if (accelerated_coverage and !train_challenge_enabled) return error.InvalidOfflineReplayBinding;
    if (hybrid_coverage_anchor and (!train_challenge_enabled or !accelerated_coverage)) return error.InvalidOfflineReplayBinding;
    if (!valueDiagnosticValid(b.phase_profiling, mass_only_diagnostic, serial_value_diagnostic, full_value_control, full_value_diagnostic, hybrid_coverage_anchor)) return error.InvalidOfflineReplayBinding;
    if (train_challenge_enabled and (!coverage_enabled or !planned_exact_enabled or primary_cv_enabled or primary_comparison_enabled or surrogate_minimax_enabled or
        b.critical_core_policy != .train_probability_gte_0_025_no_top1_v1 or
        b.postfit_policy != .train_global_residual_range_mass_l2_exact_f32_cache_v1 or
        b.performance_comparison_policy != .owned_capture_summary_incremental_reduction_critical025_v1 or
        !b.train_probability_census or b.rejected_detail_diagnostic)) return error.InvalidOfflineReplayBinding;
    const expected_schema = if (v16) "antfly.production768-offline-replay-binding.v16" else if (full_value_control or full_value_diagnostic) "antfly.production768-offline-replay-binding.v15" else if (serial_value_diagnostic) "antfly.production768-offline-replay-binding.v14" else if (mass_only_diagnostic) "antfly.production768-offline-replay-binding.v13" else if (half_retention) "antfly.production768-offline-replay-binding.v12" else if (codec_diagnostic) "antfly.production768-offline-replay-binding.v11" else if (hybrid_coverage_anchor) "antfly.production768-offline-replay-binding.v10" else if (accelerated_coverage) "antfly.production768-offline-replay-binding.v9" else if (train_challenge_enabled) "antfly.production768-offline-replay-binding.v8" else if (surrogate_minimax_planned_exact) "antfly.production768-offline-replay-binding.v7" else if (surrogate_minimax_enabled) "antfly.production768-offline-replay-binding.v6" else if (primary_cv_enabled) "antfly.production768-offline-replay-binding.v5" else if (primary_comparison_enabled) "antfly.production768-offline-replay-binding.v4" else if (planned_exact_enabled) "antfly.production768-offline-replay-binding.v3" else if (coverage_enabled) "antfly.production768-offline-replay-binding.v2" else "antfly.production768-offline-replay-binding.v1";
    if (!std.mem.eql(u8, b.schema, expected_schema) or (v16 and !(full_value_control or full_value_diagnostic)) or !rawSelectionTupleValid(b.raw_selection_solver, b.raw_selection_solver_explicit, b.raw_selection_policy_sha256, v16) or b.runtime_chunk_rows != 256 or b.export_metadata_chunk_rows != 1024 or b.max_transient_host_bytes != 3840 * 1024 * 1024 or !b.diagnostic_only or b.qualification_credit or b.production_qualified) return error.InvalidOfflineReplayBinding;
    inline for (.{ b.whole_manifest, b.complete_manifest, b.producer_source_freeze, b.producer_binary, b.producer_profile, b.integrity_closure }) |ref| try authenticate(ref);
    if (b.historical_coverage) |coverage| {
        try authenticate(coverage.dense_outer_manifest);
        try authenticate(coverage.dense_nested_manifest);
        try authenticate(coverage.dense_reference_parent_manifest);
        if (!std.mem.eql(u8, &coverage.dense_reference_parent_manifest.sha256, &coverage.dense_reference_parent_manifest_sha256) or !std.fs.path.isAbsolute(coverage.dense_dir) or coverage.maximum_workspace_bytes != 128 * 1024 * 1024) return error.InvalidOfflineReplayBinding;
    }
    var authority = b.authority;
    // The nested capture manifest's generation_contract_sha256 describes the
    // generation evidence. CompleteAuthority instead binds the TRAIN policy
    // digest used by writeComplete/readComplete; these are intentionally
    // distinct authorities.
    authority.generation_contract_sha256 = future_training.policyDigest();
    const authority_digest = authority.digest();
    if (!std.mem.eql(u8, &authority_digest, &b.complete_authority_sha256)) return error.InvalidOfflineReplayBinding;
    var result = replay.run(std.testing.allocator, std.testing.io, .{
        .raw_selection_solver = b.raw_selection_solver,
        .raw_selection_solver_explicit = b.raw_selection_solver_explicit,
        .raw_selection_policy_sha256 = if (b.raw_selection_solver == .range_shared_fp64_gram_original_kkt_v1) replay.range_shared_policy_sha256 else null,
        .train_challenge_mass_only = b.train_challenge_mass_only,
        .phase_profiling = b.phase_profiling,
        .serial_pilot_fp64_value_backend = b.serial_pilot_fp64_value_backend,
        .serial_full_fp64_value_backend = b.serial_full_fp64_value_backend,
        .serial_full_fp64_control = b.serial_full_fp64_control,
        .train_challenge_historical_selection = b.train_challenge_historical_selection,
        .hybrid_coverage_anchor = b.hybrid_coverage_anchor,
        .encoding = b.encoding,
        .rotated4_risk_retention = b.rotated4_risk_retention,
        .historical_coverage_probability_backend = b.historical_coverage_probability_backend,
        .train_exact_reservation_policy = b.train_exact_reservation_policy,
        .train_primary_policy_comparison = b.train_primary_policy_comparison,
        .train_primary_policy_cross_validation = b.train_primary_policy_cross_validation,
        .critical_core_policy = b.critical_core_policy,
        .postfit_policy = b.postfit_policy,
        .rejected_detail_diagnostic = b.rejected_detail_diagnostic,
        .train_probability_census = b.train_probability_census,
        .performance_comparison_policy = b.performance_comparison_policy,
        .historical_coverage = if (b.historical_coverage) |coverage| .{
            .dense = .{ .output_dir = coverage.dense_dir, .max_bytes = 128 * 1024 * 1024 - 64 * 1024, .parent_manifest_sha256 = b.parent_manifest_sha256, .identity = b.identity, .source_freeze_sha256 = b.source_freeze_sha256, .binary_sha256 = b.binary_sha256, .exporter_sha256 = b.exporter_sha256, .parent_study_tokens_sha256 = coverage.dense_parent_study_tokens_sha256, .study_tokens_sha256 = coverage.dense_study_tokens_sha256, .samples_per_stratum = 512 },
            .dense_outer = .{ .manifest_path = coverage.dense_outer_manifest.path, .manifest_sha256 = coverage.dense_outer_manifest.sha256, .nested_manifest_sha256 = coverage.dense_nested_manifest.sha256, .parent_manifest_sha256 = coverage.dense_reference_parent_manifest_sha256, .identity = b.identity, .source_freeze_sha256 = b.source_freeze_sha256, .binary_sha256 = b.binary_sha256, .exporter_sha256 = b.exporter_sha256 },
            .dense_nested_manifest_sha256 = coverage.dense_nested_manifest.sha256,
            .maximum_workspace_bytes = coverage.maximum_workspace_bytes,
        } else null,
        .whole_fit = .{ .output_dir = b.whole_fit_dir, .identity = b.identity, .config_sha256 = b.config_sha256, .source_freeze_sha256 = b.source_freeze_sha256, .binary_sha256 = b.binary_sha256, .exporter_sha256 = b.exporter_sha256, .limits = checkpoint.serving_limits },
        .complete_future = .{ .output_dir = b.complete_future_dir, .manifest_sha256 = b.complete_manifest_sha256, .primary_parent_manifest_sha256 = b.parent_manifest_sha256, .identity = b.identity, .source_freeze_sha256 = b.source_freeze_sha256, .binary_sha256 = b.binary_sha256, .exporter_sha256 = b.exporter_sha256, .authority = authority, .historical_study_tokens_sha256 = b.historical_study_tokens_sha256 },
        .maximum_workers = if (full_value_control or full_value_diagnostic) 1 else 4,
        .max_workspace_bytes = 1024 * 1024 * 1024,
        .max_transient_host_bytes = b.max_transient_host_bytes,
        .query_reservation_bytes = b.query_reservation_bytes,
    }) catch |err| {
        const failure = try std.json.Stringify.valueAlloc(std.testing.allocator, .{ .schema = if (v16) "antfly.production768-offline-replay-failure.v16" else if (full_value_control or full_value_diagnostic) "antfly.production768-offline-replay-failure.v15" else if (serial_value_diagnostic) "antfly.production768-offline-replay-failure.v14" else if (mass_only_diagnostic) "antfly.production768-offline-replay-failure.v13" else if (half_retention) "antfly.production768-offline-replay-failure.v12" else if (codec_diagnostic) "antfly.production768-offline-replay-failure.v11" else if (hybrid_coverage_anchor) "antfly.production768-offline-replay-failure.v10" else if (accelerated_coverage) "antfly.production768-offline-replay-failure.v8" else if (train_challenge_enabled) "antfly.production768-offline-replay-failure.v7" else if (surrogate_minimax_planned_exact) "antfly.production768-offline-replay-failure.v6" else if (surrogate_minimax_enabled) "antfly.production768-offline-replay-failure.v5" else if (primary_cv_enabled) "antfly.production768-offline-replay-failure.v4" else if (primary_comparison_enabled) "antfly.production768-offline-replay-failure.v3" else if (planned_exact_enabled) "antfly.production768-offline-replay-failure.v2" else "antfly.production768-offline-replay-failure.v1", .diagnostic_only = true, .qualification_credit = false, .production_qualified = false, .encoding = b.encoding, .rotated4_risk_retention = b.rotated4_risk_retention, .policy = replay.policy, .serial_pilot_fp64_value_backend = b.serial_pilot_fp64_value_backend, .serial_full_fp64_value_backend = b.serial_full_fp64_value_backend, .serial_full_fp64_control = b.serial_full_fp64_control, .serial_pilot_fp64_value_stats = null, .historical_coverage_wall_ns = null, .serial_pilot_rewrite_wall_ns = null, .serial_pilot_wall_ns = null, .serial_pilot_input_extract_ns = null, .serial_pilot_query_frontend_ns = null, .serial_pilot_selection_fit_ns = null, .serial_pilot_value_fit_ns = null, .serial_pilot_validation_and_postfit_ns = null, .serial_pilot_unclassified_ns = null, .serial_pilot_observer_timing = null, .serial_pilot_reload_ns = null, .serial_pilot_encode_rewrite_ns = null, .pilot_timing_semantics = "all arrays nanoseconds; wall encloses query_frontend+selection_fit+value_fit+validation_and_postfit; input_extract precedes wall; reload and encode_rewrite follow wall; raw selection encloses active-set scalar solves and is not additive with nested scalar backend clocks; zero means unavailable or unused slot", .final_engine_wall_ns = null, .train_challenge_mass_only = b.train_challenge_mass_only, .phase_profiling = b.phase_profiling, .train_challenge_mass_only_calls = null, .train_challenge_mass_only_gemm_calls = null, .train_challenge_validation_summary_calls = null, .fitter_phase_counters = null, .train_challenge_historical_selection = b.train_challenge_historical_selection, .hybrid_coverage_anchor = b.hybrid_coverage_anchor, .historical_coverage_probability_backend = b.historical_coverage_probability_backend, .historical_coverage_probability_backend_policy_sha256 = if (accelerated_coverage) std.fmt.bytesToHex(historical_coverage_probability_backend_policy_sha256, .lower) else null, .train_challenge_policy = if (train_challenge_enabled) train_challenge.policy else null, .train_challenge_policy_sha256 = if (train_challenge_enabled) std.fmt.bytesToHex(train_challenge_policy_sha256, .lower) else null, .train_primary_policy_comparison = b.train_primary_policy_comparison, .train_primary_policy_cross_validation = b.train_primary_policy_cross_validation, .train_primary_policy_cv_policy = if (primary_cv_enabled) train_cv.policy else null, .train_primary_policy_cv_fit_rows_per_consumer = if (primary_cv_enabled) @as(usize, 672) else null, .train_primary_policy_cv_rows_per_consumer = if (primary_cv_enabled) @as(usize, 96) else null, .train_exact_reservation_policy = b.train_exact_reservation_policy, .performance_comparison_policy = b.performance_comparison_policy, .critical_core_policy = b.critical_core_policy, .postfit_policy = b.postfit_policy, .runtime_chunk_rows = b.runtime_chunk_rows, .export_metadata_chunk_rows = b.export_metadata_chunk_rows, .failure_class = if (err == error.CompactionWorkspaceLimit or err == error.InvalidFitConfig) "admission_or_config" else "numerical_or_runtime", .error_name = @errorName(err), .accepted = false }, .{});
        defer std.testing.allocator.free(failure);
        var failed = try std.Io.Dir.cwd().createFile(std.testing.io, output_path, .{ .exclusive = true });
        defer failed.close(std.testing.io);
        try failed.writeStreamingAll(std.testing.io, failure);
        try failed.writeStreamingAll(std.testing.io, "\n");
        try failed.sync(std.testing.io);
        return err;
    };
    defer result.deinit();
    const body = try std.json.Stringify.valueAlloc(std.testing.allocator, .{ .schema = if (v16) "antfly.production768-offline-replay-result.v16" else if (full_value_control or full_value_diagnostic) "antfly.production768-offline-replay-result.v15" else if (serial_value_diagnostic) "antfly.production768-offline-replay-result.v14" else if (mass_only_diagnostic) "antfly.production768-offline-replay-result.v13" else if (half_retention) "antfly.production768-offline-replay-result.v12" else if (codec_diagnostic) "antfly.production768-offline-replay-result.v11" else if (hybrid_coverage_anchor) "antfly.production768-offline-replay-result.v10" else if (accelerated_coverage) "antfly.production768-offline-replay-result.v9" else if (train_challenge_enabled) "antfly.production768-offline-replay-result.v8" else if (surrogate_minimax_planned_exact) "antfly.production768-offline-replay-result.v7" else if (surrogate_minimax_enabled) "antfly.production768-offline-replay-result.v6" else if (primary_cv_enabled) "antfly.production768-offline-replay-result.v5" else if (primary_comparison_enabled) "antfly.production768-offline-replay-result.v4" else if (planned_exact_enabled) "antfly.production768-offline-replay-result.v3" else if (coverage_enabled) "antfly.production768-offline-replay-result.v2" else "antfly.production768-offline-replay-result.v1", .diagnostic_only = true, .qualification_credit = false, .production_qualified = false, .encoding = b.encoding, .rotated4_risk_retention = b.rotated4_risk_retention, .policy = replay.policy, .serial_pilot_fp64_value_backend = b.serial_pilot_fp64_value_backend, .serial_full_fp64_value_backend = b.serial_full_fp64_value_backend, .serial_full_fp64_control = b.serial_full_fp64_control, .serial_pilot_fp64_value_stats = result.serial_pilot_fp64_value_stats, .historical_coverage_wall_ns = result.historical_coverage_wall_ns, .serial_pilot_rewrite_wall_ns = result.serial_pilot_rewrite_wall_ns, .serial_pilot_wall_ns = result.serial_pilot_wall_ns, .serial_pilot_input_extract_ns = result.serial_pilot_input_extract_ns, .serial_pilot_query_frontend_ns = result.serial_pilot_query_frontend_ns, .serial_pilot_selection_fit_ns = result.serial_pilot_selection_fit_ns, .serial_pilot_value_fit_ns = result.serial_pilot_value_fit_ns, .serial_pilot_validation_and_postfit_ns = result.serial_pilot_validation_and_postfit_ns, .serial_pilot_unclassified_ns = result.serial_pilot_validation_and_postfit_ns, .serial_pilot_observer_timing = result.serial_pilot_observer_timing, .serial_pilot_reload_ns = result.serial_pilot_reload_ns, .serial_pilot_encode_rewrite_ns = result.serial_pilot_encode_rewrite_ns, .pilot_timing_semantics = "all arrays nanoseconds; wall encloses query_frontend+selection_fit+value_fit+validation_and_postfit; input_extract precedes wall; reload and encode_rewrite follow wall; raw selection encloses active-set scalar solves and is not additive with nested scalar backend clocks; zero means unavailable or unused slot", .final_engine_wall_ns = result.final_engine_wall_ns, .train_challenge_mass_only = b.train_challenge_mass_only, .phase_profiling = b.phase_profiling, .train_challenge_mass_only_calls = result.train_challenge_mass_only_calls, .train_challenge_mass_only_gemm_calls = result.train_challenge_mass_only_gemm_calls, .train_challenge_validation_summary_calls = result.train_challenge_validation_summary_calls, .fitter_phase_counters = result.fitter_phase_counters, .train_challenge_historical_selection = b.train_challenge_historical_selection, .hybrid_coverage_anchor = b.hybrid_coverage_anchor, .historical_coverage_probability_backend = b.historical_coverage_probability_backend, .historical_coverage_probability_backend_policy_sha256 = if (accelerated_coverage) std.fmt.bytesToHex(historical_coverage_probability_backend_policy_sha256, .lower) else null, .train_challenge_policy = if (train_challenge_enabled) train_challenge.policy else null, .train_challenge_policy_sha256 = if (train_challenge_enabled) std.fmt.bytesToHex(train_challenge_policy_sha256, .lower) else null, .train_challenge_evidence = result.train_challenge_evidence, .train_primary_policy_comparison = b.train_primary_policy_comparison, .train_primary_policy_cross_validation = b.train_primary_policy_cross_validation, .train_primary_policy_cv_policy = if (primary_cv_enabled) train_cv.policy else null, .train_primary_policy_cv_fit_rows_per_consumer = if (primary_cv_enabled) @as(usize, 672) else null, .train_primary_policy_cv_rows_per_consumer = if (primary_cv_enabled) @as(usize, 96) else null, .train_exact_reservation_policy = b.train_exact_reservation_policy, .historical_coverage_policy_sha256 = if (coverage_enabled) std.fmt.bytesToHex(@import("runtime/kv/attention_match_production768_historical_coverage.zig").policy_digest, .lower) else null, .historical_coverage = result.historical_coverage_evidence, .planned_exact = result.planned_exact, .performance_comparison_policy = b.performance_comparison_policy, .critical_core_policy = b.critical_core_policy, .postfit_policy = b.postfit_policy, .workers = result.workers, .source_host_bytes = result.source_host_bytes, .admitted_fit_host_bytes = result.admitted_fit_host_bytes, .runtime_chunk_rows = b.runtime_chunk_rows, .export_metadata_chunk_rows = b.export_metadata_chunk_rows, .stats = result.stats, .source_physical_bytes = result.source_physical_bytes, .candidate_physical_bytes = result.candidate_physical_bytes, .physical_ratio = result.physical_ratio, .primary_budget_percent = result.primary_budget_percent, .fallback_budget_percent = result.fallback_budget_percent, .maximum_fallbacks = result.maximum_fallbacks, .accepted = result.accepted }, .{});
    defer std.testing.allocator.free(body);
    var file = try std.Io.Dir.cwd().createFile(std.testing.io, output_path, .{ .exclusive = true });
    defer file.close(std.testing.io);
    try file.writeStreamingAll(std.testing.io, body);
    try file.writeStreamingAll(std.testing.io, "\n");
    try file.sync(std.testing.io);
}

test "production768 V15 full-value diagnostic is exclusive and phase bound" {
    try std.testing.expect(valueDiagnosticValid(false, false, false, false, false, false));
    try std.testing.expect(valueDiagnosticValid(true, false, false, true, false, true));
    try std.testing.expect(valueDiagnosticValid(true, false, false, false, true, true));
    try std.testing.expect(!valueDiagnosticValid(false, false, false, false, true, true));
    try std.testing.expect(!valueDiagnosticValid(true, false, false, true, true, true));
    try std.testing.expect(!valueDiagnosticValid(true, false, false, false, true, false));
}
