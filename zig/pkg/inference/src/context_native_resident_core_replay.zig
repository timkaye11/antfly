// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//! REAL, default-off replay of the retained 7936-row native resident core.
//! This is captured-development evidence only; it is not an 8128/live-profile
//! result and grants no formal or production qualification credit.
const std = @import("std");
const platform = @import("antfly_platform");
const checkpoint = @import("runtime/kv/attention_match_checkpoint.zig");
const whole = @import("runtime/kv/attention_match_whole_fit_input.zig");
const augmentation = @import("runtime/kv/attention_match_training_augmentation.zig");
const future_augmentation = @import("runtime/kv/attention_match_future_augmentation.zig");
const future_policy = @import("runtime/kv/attention_match_future_decode_training.zig");
const coverage = @import("runtime/kv/attention_match_production768_historical_coverage.zig");
const bridge = @import("runtime/kv/attention_match_native_resident_hybrid_bridge.zig");
const profile = @import("runtime/kv/attention_match_resident_native_hybrid_profile.zig");
const engine = @import("runtime/kv/attention_match_engine.zig");
const planned_exact = @import("runtime/kv/attention_match_planned_exact_context.zig");
const metal_storage = @import("ops/metal_attention_match_storage.zig");
const validation = @import("ops/metal_attention_match_validation.zig");
const Fitter = @import("ops/metal_attention_match_fit.zig").Fitter;

const PerformanceMode = enum { uncached_cpu_fp64, exact_page_cache_cpu_fp64, exact_reference_summary_cpu_fp64, metal_gpu_f32_train_mass };

const Ref = struct { path: []const u8, sha256: [32]u8 };
const DenseBinding = struct {
    dense_dir: []const u8,
    dense_outer_manifest: Ref,
    dense_nested_manifest: Ref,
    dense_reference_parent_manifest: Ref,
    dense_reference_parent_manifest_sha256: [32]u8,
    dense_parent_study_tokens_sha256: [32]u8,
    dense_study_tokens_sha256: [32]u8,
    maximum_workspace_bytes: usize,
};
const Binding = struct {
    schema: []const u8,
    whole_fit_dir: []const u8,
    complete_future_dir: []const u8,
    whole_manifest: Ref,
    complete_manifest: Ref,
    producer_source_freeze: Ref,
    producer_binary: Ref,
    producer_profile: Ref,
    integrity_closure: Ref,
    historical_coverage: DenseBinding,
    identity: checkpoint.Identity,
    config_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    binary_sha256: [32]u8,
    exporter_sha256: [32]u8,
    complete_manifest_sha256: [32]u8,
    parent_manifest_sha256: [32]u8,
    historical_study_tokens_sha256: [32]u8,
    complete_authority_sha256: [32]u8,
    authority: @FieldType(@import("runtime/kv/attention_match_production768_offline_replay.zig").CompleteAuthority, "authority"),
    expected_shape_sha256: [32]u8,
    config_authority_sha256: [32]u8,
    collector_manifest: Ref,
    collector_required_bytes: usize,
    source_length: usize,
    repeat_length: usize,
    diagnostic_only: bool,
    qualification_credit: bool,
    formal_performance_credit: bool,
    production_qualified: bool,
    diagnostic_query_batch_rows: u16 = 64,
    performance_mode: PerformanceMode = .uncached_cpu_fp64,
};

fn hashFile(path: []const u8) ![32]u8 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, std.testing.allocator, .limited(512 * 1024 * 1024));
    defer std.testing.allocator.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return digest;
}
fn authenticate(ref: Ref) !void {
    if (!std.fs.path.isAbsolute(ref.path) or !std.crypto.timing_safe.eql([32]u8, try hashFile(ref.path), ref.sha256))
        return error.InvalidNativeCoreReplayBinding;
}
fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn denseBackingBytes(dense: augmentation.OwnedTraining) !usize {
    var bytes = try std.math.mul(usize, dense.values.len, @sizeOf(f32));
    bytes = try add(bytes, try std.math.mul(usize, dense.positions.len, @sizeOf(u64)));
    return add(bytes, try std.math.mul(usize, dense.batches.len, @sizeOf(engine.ReferenceBatch)));
}

fn denseOwnedBytes(owner: augmentation.Owned) !usize {
    var result = try add(owner.manifest_bytes.len, try denseBackingBytes(.{ .allocator = owner.allocator, .batches = owner.batches, .values = owner.training, .positions = owner.positions }));
    inline for (.{ owner.consumer_layers, owner.query_heads, owner.counts, owner.dimensions }) |items|
        result = try add(result, try std.math.mul(usize, items.len, @sizeOf(u32)));
    return result;
}

fn exactHeadEqual(source: checkpoint.State, candidate: checkpoint.State, donor_layer: u32, head_index: usize) bool {
    for (source.donors, candidate.donors) |before_donor, after_donor| if (before_donor.layer == donor_layer) {
        if (after_donor.layer != donor_layer or head_index >= before_donor.heads.len or head_index >= after_donor.heads.len) return false;
        const before = before_donor.heads[head_index];
        const after = after_donor.heads[head_index];
        return before.dim == after.dim and before.compacted_until == after.compacted_until and
            std.mem.eql(u64, before.positions, after.positions) and std.mem.eql(u64, before.coverage_ends, after.coverage_ends) and
            std.mem.eql(u8, std.mem.sliceAsBytes(before.keys), std.mem.sliceAsBytes(after.keys)) and
            std.mem.eql(u8, std.mem.sliceAsBytes(before.values), std.mem.sliceAsBytes(after.values)) and
            std.mem.eql(u8, std.mem.sliceAsBytes(before.bias), std.mem.sliceAsBytes(after.bias)) and
            std.meta.eql(before.fp8, after.fp8) and before.compressed.version == after.compressed.version and
            std.mem.eql(u8, std.mem.sliceAsBytes(before.compressed.spans), std.mem.sliceAsBytes(after.compressed.spans)) and
            std.mem.eql(u8, before.compressed.keys, after.compressed.keys) and std.mem.eql(u8, before.compressed.values, after.compressed.values);
    };
    return false;
}

fn candidateRawDigest(state: checkpoint.State) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("antfly.native-resident-complete-state.v1\x00");
    const framed = struct {
        fn bytes(hash: *std.crypto.hash.sha2.Sha256, value: []const u8) void {
            var length: [8]u8 = undefined;
            std.mem.writeInt(u64, &length, value.len, .little);
            hash.update(&length);
            hash.update(value);
        }
    }.bytes;
    h.update(std.mem.asBytes(&state.identity.weights));
    h.update(std.mem.asBytes(&state.identity.tokenizer));
    h.update(std.mem.asBytes(&state.identity.template));
    h.update(std.mem.asBytes(&state.identity.layout));
    h.update(std.mem.asBytes(&state.logical_length));
    framed(&h, std.mem.sliceAsBytes(state.tokens));
    framed(&h, std.mem.sliceAsBytes(state.logits));
    framed(&h, std.mem.sliceAsBytes(state.layer_donors));
    var donor_count: [8]u8 = undefined;
    std.mem.writeInt(u64, &donor_count, state.donors.len, .little);
    h.update(&donor_count);
    for (state.donors) |donor| {
        h.update(std.mem.asBytes(&donor.layer));
        h.update(std.mem.asBytes(&donor.window));
        var head_count: [8]u8 = undefined;
        std.mem.writeInt(u64, &head_count, donor.heads.len, .little);
        h.update(&head_count);
        for (donor.heads) |head| {
            h.update(std.mem.asBytes(&head.dim));
            h.update(std.mem.asBytes(&head.compacted_until));
            framed(&h, std.mem.sliceAsBytes(head.positions));
            framed(&h, std.mem.sliceAsBytes(head.coverage_ends));
            framed(&h, std.mem.sliceAsBytes(head.keys));
            framed(&h, std.mem.sliceAsBytes(head.values));
            framed(&h, std.mem.sliceAsBytes(head.bias));
            h.update(std.mem.asBytes(&head.fp8.first_row));
            h.update(std.mem.asBytes(&head.fp8.rows));
            h.update(std.mem.asBytes(&head.fp8.key_scale));
            h.update(std.mem.asBytes(&head.fp8.value_scale));
            h.update(std.mem.asBytes(&head.compressed.version));
            var span_count: [8]u8 = undefined;
            std.mem.writeInt(u64, &span_count, head.compressed.spans.len, .little);
            h.update(&span_count);
            for (head.compressed.spans) |span| {
                h.update(std.mem.asBytes(&span.first));
                h.update(std.mem.asBytes(&span.rows));
                h.update(std.mem.asBytes(&span.offset));
            }
            framed(&h, head.compressed.keys);
            framed(&h, head.compressed.values);
        }
    }
    return h.finalResult();
}

test "REAL retained native resident core replay is captured-development only" {
    const binding_path = platform.env.getenv("ANTFLY_METAL_NATIVE_CORE_REPLAY_BINDING") orelse return error.SkipZigTest;
    const result_path = platform.env.getenv("ANTFLY_METAL_NATIVE_CORE_REPLAY_RESULT") orelse return error.InvalidNativeCoreReplayBinding;
    if (!std.fs.path.isAbsolute(binding_path) or !std.fs.path.isAbsolute(result_path)) return error.InvalidNativeCoreReplayBinding;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, binding_path, std.testing.allocator, .limited(256 * 1024));
    defer std.testing.allocator.free(bytes);
    var parsed = try std.json.parseFromSlice(Binding, std.testing.allocator, bytes, .{ .ignore_unknown_fields = false });
    defer parsed.deinit();
    const b = parsed.value;
    if ((b.diagnostic_query_batch_rows != 64 and b.diagnostic_query_batch_rows != 256) or
        !std.mem.eql(u8, b.schema, "antfly.native-resident-core-replay-binding.v2") or b.source_length != 7936 or b.repeat_length != 512 or !b.diagnostic_only or b.qualification_credit or b.formal_performance_credit or b.production_qualified or b.historical_coverage.maximum_workspace_bytes != 128 * 1024 * 1024) return error.InvalidNativeCoreReplayBinding;
    inline for (.{ b.whole_manifest, b.complete_manifest, b.producer_source_freeze, b.producer_binary, b.producer_profile, b.integrity_closure, b.collector_manifest, b.historical_coverage.dense_outer_manifest, b.historical_coverage.dense_nested_manifest, b.historical_coverage.dense_reference_parent_manifest }) |ref| try authenticate(ref);
    std.debug.print("native_core_replay phase=bindings_authenticated scope=captured_development source_rows=7936 qualification_credit=false\n", .{});

    var authority = b.authority;
    authority.generation_contract_sha256 = future_policy.policyDigest();
    if (!std.crypto.timing_safe.eql([32]u8, authority.digest(), b.complete_authority_sha256)) return error.InvalidNativeCoreReplayBinding;
    var input = try whole.read(std.testing.allocator, std.testing.io, .{ .output_dir = b.whole_fit_dir, .identity = b.identity, .config_sha256 = b.config_sha256, .source_freeze_sha256 = b.source_freeze_sha256, .binary_sha256 = b.binary_sha256, .exporter_sha256 = b.exporter_sha256, .limits = checkpoint.serving_limits });
    defer input.deinit();
    var complete = try future_augmentation.readComplete(std.testing.allocator, std.testing.io, .{ .output_dir = b.complete_future_dir, .manifest_sha256 = b.complete_manifest_sha256, .primary_parent_manifest_sha256 = b.parent_manifest_sha256, .identity = b.identity, .source_freeze_sha256 = b.source_freeze_sha256, .binary_sha256 = b.binary_sha256, .exporter_sha256 = b.exporter_sha256, .authority = authority, .historical_study_tokens_sha256 = b.historical_study_tokens_sha256, .primary_batches = input.batches });
    defer complete.deinit();
    try coverage.authenticateOuter(std.testing.allocator, std.testing.io, .{ .manifest_path = b.historical_coverage.dense_outer_manifest.path, .manifest_sha256 = b.historical_coverage.dense_outer_manifest.sha256, .nested_manifest_sha256 = b.historical_coverage.dense_nested_manifest.sha256, .parent_manifest_sha256 = b.historical_coverage.dense_reference_parent_manifest_sha256, .identity = b.identity, .source_freeze_sha256 = b.source_freeze_sha256, .binary_sha256 = b.binary_sha256, .exporter_sha256 = b.exporter_sha256 });
    var dense = try augmentation.readReferenceCoverageDense512(std.testing.allocator, std.testing.io, .{ .output_dir = b.historical_coverage.dense_dir, .max_bytes = b.historical_coverage.maximum_workspace_bytes - 64 * 1024, .parent_manifest_sha256 = b.parent_manifest_sha256, .identity = b.identity, .source_freeze_sha256 = b.source_freeze_sha256, .binary_sha256 = b.binary_sha256, .exporter_sha256 = b.exporter_sha256, .parent_study_tokens_sha256 = b.historical_coverage.dense_parent_study_tokens_sha256, .study_tokens_sha256 = b.historical_coverage.dense_study_tokens_sha256, .samples_per_stratum = 512 });
    defer dense.deinit();
    var nested_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(dense.manifest_bytes, &nested_digest, .{});
    if (!std.crypto.timing_safe.eql([32]u8, nested_digest, b.historical_coverage.dense_nested_manifest.sha256) or input.batches.len != 56 or complete.future.batches.len != 56 or dense.batches.len != 56 or dense.source_length != b.source_length or dense.repeat_length != b.repeat_length) return error.InvalidNativeCoreReplayBinding;
    if (!std.crypto.timing_safe.eql([32]u8, b.collector_manifest.sha256, nested_digest)) return error.InvalidNativeCoreReplayBinding;
    var retained_manifest = try std.json.parseFromSlice(struct { required_bytes: usize }, std.testing.allocator, dense.manifest_bytes, .{ .ignore_unknown_fields = true });
    defer retained_manifest.deinit();
    if (b.collector_required_bytes != retained_manifest.value.required_bytes or b.collector_required_bytes < try denseOwnedBytes(dense)) return error.InvalidNativeCoreReplayBinding;
    const dense_view: augmentation.OwnedTraining = .{ .allocator = std.testing.allocator, .batches = dense.batches, .values = dense.training, .positions = dense.positions };
    std.debug.print("native_core_replay phase=captured_inputs_loaded primary=56 dense=56 future=56\n", .{});

    const source_host_bytes = try input.checkpoint_owned.state.residentBytes();
    // Captured checkpoint arrays are host-owned. No native KV device owner is
    // installed by this replay; codec/device allocation numbers below are
    // page-rounded storage-budget evidence only.
    const native_device_bytes: usize = 0;
    var external_live = try input.liveBytes();
    external_live = try add(external_live, try denseOwnedBytes(dense));
    external_live = try add(external_live, try denseOwnedBytes(complete.future));
    external_live = try add(external_live, try denseOwnedBytes(complete.historical));
    external_live = try add(external_live, 4 * 1024 * 1024);
    const admission: profile.Admission = .{ .retained_host_source_bytes = external_live, .retained_native_device_bytes = native_device_bytes, .dense_backing_bytes = 0, .request_owner_bytes = 0, .maximum_head_clone_bytes = 0, .fitter_workspace_bytes = profile.head_workspace_bytes };
    if (try admission.fitWorkers() != 1 or source_host_bytes == 0) return error.InvalidNativeCoreReplayBinding;
    var fitter = try Fitter.initWorker(std.testing.allocator, null, profile.head_workspace_bytes, profile.head_workspace_bytes);
    defer fitter.deinit();
    try fitter.setQueryBatchRows(b.diagnostic_query_batch_rows);
    var cpu_context: validation.Context = .{};
    const bridge_started = std.Io.Timestamp.now(std.testing.io, .awake);
    var native = try bridge.prepare(std.testing.allocator, input.checkpoint_owned.state, input.layout, input.batches, &dense_view, complete.future.batches, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .reject_exact_active_fixed_point = true, .max_relative_mass_error = 0.05, .max_relative_output_rmse = 0.05, .max_sweeps = input.fit.max_sweeps, .nnls_tolerance = input.fit.nnls_tolerance, .min_weight = input.fit.min_weight, .max_workspace_bytes = profile.head_workspace_bytes, .backend = fitter.backend() }, validation.backend(&cpu_context), .{ .source_length = dense.source_length, .repeat_length = dense.repeat_length, .source_freeze_sha256 = b.source_freeze_sha256, .config_authority_sha256 = b.config_authority_sha256, .expected_shape_sha256 = b.expected_shape_sha256, .collector_required_bytes = b.collector_required_bytes, .collector_manifest_sha256 = b.collector_manifest.sha256, .external_live_bytes = external_live, .actual_native_device_bytes = native_device_bytes, .combined_cap = profile.head_workspace_bytes, .diagnostic_query_batch_rows = b.diagnostic_query_batch_rows, .train_mass_precision = if (b.performance_mode == .metal_gpu_f32_train_mass) .metal_gpu_f32_pre_eval_v1 else .cpu_fp64, .reference_summary_cache_policy = if (b.performance_mode == .exact_reference_summary_cpu_fp64) .exact_original_train_hho_v1 else .disabled });
    defer native.deinit();
    const bridge_wall_ns: u64 = @intCast(@max(1, bridge_started.durationTo(std.Io.Timestamp.now(std.testing.io, .awake)).nanoseconds));
    if (native.planned_exact_donor_layer != 5 or native.planned_exact_head_index != 0 or native.rewrite_evidence.pilot_head_count != 7 or native.rewrite_evidence.skipped_exact_head_count != 1 or native.rewrite_evidence.rewritten_batch_count != 52 or native.rewrite_evidence.skipped_exact_batch_count != 4 or native.rewrite_evidence.heldout_value_observations != 0 or cpu_context.successfulCalls() == 0) return error.NativeCoreReplayEvidenceMismatch;
    std.debug.print("native_core_replay phase=bridge_complete exact=D5H0 pilots=7 rewritten=52 exact_batches=4 hho_observations=0\n", .{});

    const final_query_bytes = try native.final_training.liveBytes();
    const non_source_bytes = try add(external_live - source_host_bytes, final_query_bytes);
    const fit_budget = std.math.sub(usize, profile.maximum_transient_host_bytes, non_source_bytes) catch return error.CompactionWorkspaceLimit;
    const worker_plan = try Fitter.parallelPlan(source_host_bytes, 8, profile.head_workspace_bytes, fit_budget, 1);
    if (worker_plan.workers != 1) return error.CompactionWorkspaceLimit;
    var stats: engine.Stats = .{};
    var exact_context: planned_exact.Context = .{ .enabled = true, .donor_layer = native.planned_exact_donor_layer, .head_index = native.planned_exact_head_index };
    const options: engine.Options = .{ .admission = .cumulative_ladder_4_3_2, .cumulative_page_cache_policy = if (b.performance_mode == .exact_page_cache_cpu_fp64) .exact_1024_v1 else .disabled, .fit_strategy = .identity_chunk_coefficients_adaptive_v18, .staged_operator_strategy = true, .adaptive_v18_fn = planned_exact.Context.callback, .adaptive_v18_context = &exact_context, .adaptive_v18_workspace_profile = .resident_native_hybrid_2g_v1, .adaptive_v18_numerical_policy = .accepted_f32_gram_reuse_v1, .adaptive_v18_geometry_route = .flat_gram_only, .adaptive_v18_critical_core_policy = .train_probability_gte_0_025_no_top1_v1, .adaptive_v18_postfit_policy = .train_global_residual_range_mass_l2_exact_f32_cache_v1, .geometry_chooser_policy = .incremental_gram_v1, .independent_validation_policy = .accelerate_fp64_summary_v1, .private_performance_policy = if (b.diagnostic_query_batch_rows == 256) .diagnostic_capture_gram_batch256_v1 else .owned_capture_summary_incremental_reduction_critical025_v1, .stats = &stats, .prefix_rows = 128, .recent_rows = 2048, .chunk_rows = 256, .retention_divisor = 4, .min_retention_divisor = 2, .rejected_head_policy = .preserve_original, .max_queries_per_consumer = 768, .validation = .whole_head, .independent_fp64_publication = true, .storage_precision = .f16, .max_parallel_validations = 1, .max_transient_host_bytes = worker_plan.admitted_bytes, .min_global_savings_ratio = 1.5, .adaptive_v18_primary_budget_percent = 91, .adaptive_v18_fallback_budget_percent = 95, .adaptive_v18_max_budget_fallbacks = profile.maximum_p95_attempts, .fit = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .reject_exact_active_fixed_point = true, .max_relative_mass_error = 0.05, .max_relative_output_rmse = 0.05, .max_sweeps = input.fit.max_sweeps, .nnls_tolerance = input.fit.nnls_tolerance, .min_weight = input.fit.min_weight, .max_workspace_bytes = profile.head_workspace_bytes, .backend = fitter.backend() } };
    var final_options = options;
    final_options.v18_reference_summary_cache_policy = if (b.performance_mode == .exact_reference_summary_cpu_fp64) .exact_original_train_hho_v1 else .disabled;
    const final_fit_started = std.Io.Timestamp.now(std.testing.io, .awake);
    var candidate = try engine.prepare(std.testing.allocator, input.checkpoint_owned.state, native.final_training.batches, input.layout, final_options);
    defer candidate.deinit();
    const final_fit_wall_ns: u64 = @intCast(@max(1, final_fit_started.durationTo(std.Io.Timestamp.now(std.testing.io, .awake)).nanoseconds));
    stats.v18_planned_exact_heads = exact_context.planned_exact_heads.load(.acquire);
    if (stats.v18_planned_exact_heads != 1) return error.NativeCoreReplayEvidenceMismatch;
    const source_physical = try metal_storage.ordinaryGlobalAllocationBytesWithControl(input.checkpoint_owned.state, checkpoint.serving_limits, .{});
    const candidate_physical = try metal_storage.globalAllocationBytesWithControl(candidate.state, checkpoint.serving_limits, .{});
    if (candidate_physical == 0) return error.NativeCoreReplayEvidenceMismatch;
    const ratio = @as(f64, @floatFromInt(source_physical)) / @as(f64, @floatFromInt(candidate_physical));
    if (!exactHeadEqual(input.checkpoint_owned.state, candidate.state, 5, 0) or !std.math.isFinite(stats.max_head_mass_error) or !std.math.isFinite(stats.max_head_output_rmse) or stats.v18_preserved_heads != 0 or stats.v18_eligible_heads != 8 or stats.v18_attempted_heads != 8 or stats.v18_accepted_heads != 7 or exact_context.adaptive_fit_attempts.load(.acquire) != 7 or stats.v18_budget_fallback_heads > 3 or ratio < 1.5 or stats.max_head_mass_error > 0.05 or stats.max_head_output_rmse > 0.05) return error.NativeCoreReplayEvidenceMismatch;
    if (stats.v18_query_batch_rows != b.diagnostic_query_batch_rows or stats.training_reduction_tiles == 0 or stats.training_reduction_backend_calls != stats.training_reduction_tiles * 2 or stats.v18_gram_backend_calls == 0) return error.NativeCoreReplayEvidenceMismatch;
    const body = try std.json.Stringify.valueAlloc(std.testing.allocator, .{ .schema = "antfly.native-resident-core-replay-result.v2", .performance_mode = b.performance_mode, .candidate_raw_sha256 = std.fmt.bytesToHex(candidateRawDigest(candidate.state), .lower), .requested_query_batch_rows = b.diagnostic_query_batch_rows, .actual_query_batch_rows = stats.v18_query_batch_rows, .bridge_wall_ns = bridge_wall_ns, .final_fit_wall_ns = final_fit_wall_ns, .training_reduction_tiles = stats.training_reduction_tiles, .training_reduction_backend_calls = stats.training_reduction_backend_calls, .gram_backend_calls = stats.v18_gram_backend_calls, .cache_prepare_ns = stats.v18_cache_prepare_ns, .selection_freeze_ns = stats.v18_selection_freeze_ns, .joint_value_fit_ns = stats.v18_joint_value_fit_ns, .gram_construction_ns = stats.v18_gram_construction_ns, .chooser_ns = stats.v18_chooser_ns, .final_heldout_ns = stats.v18_final_heldout_ns, .scope = "captured_development_7936", .live_profile_8128_claim = false, .live_device_residency_claim = false, .retained_native_device_bytes = native_device_bytes, .collector_required_bytes_scope = "retained_dense_exporter_required_bytes", .fit_workers = worker_plan.workers, .fit_host_admitted_bytes = worker_plan.admitted_bytes, .exact_head_bytes_equal = true, .diagnostic_only = true, .qualification_credit = false, .formal_performance_credit = false, .production_qualified = false, .bridge_policy = bridge.policy, .profile_policy = profile.policy, .planned_exact_donor_layer = native.planned_exact_donor_layer, .planned_exact_head_index = native.planned_exact_head_index, .pilot_heads = native.rewrite_evidence.pilot_head_count, .rewritten_batches = native.rewrite_evidence.rewritten_batch_count, .exact_batches = native.rewrite_evidence.skipped_exact_batch_count, .heldout_value_observations = native.rewrite_evidence.heldout_value_observations, .adaptive_accepted_heads = stats.v18_accepted_heads, .exact_head_count = 1, .source_physical_bytes = source_physical, .candidate_physical_bytes = candidate_physical, .physical_ratio = ratio, .max_head_mass_error = stats.max_head_mass_error, .max_head_output_rmse = stats.max_head_output_rmse, .cpu_fp64_validation_calls = cpu_context.successfulCalls(), .bridge_ledger = native.ledger, .stats = stats, .accepted = true }, .{});
    defer std.testing.allocator.free(body);
    var output = try std.Io.Dir.cwd().createFile(std.testing.io, result_path, .{ .exclusive = true });
    defer output.close(std.testing.io);
    try output.writeStreamingAll(std.testing.io, body);
    try output.writeStreamingAll(std.testing.io, "\n");
    try output.sync(std.testing.io);
    std.debug.print("native_core_replay phase=complete accepted=7 exact=1 physical_ratio={d} qualification_credit=false\n", .{ratio});
}
