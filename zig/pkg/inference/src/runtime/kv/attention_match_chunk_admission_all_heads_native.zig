// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const platform = @import("antfly_platform");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const whole = @import("attention_match_whole_fit_input.zig");
const augmentation = @import("attention_match_training_augmentation.zig");
const replay = @import("attention_match_chunk_admission_replay.zig");

const workspace_bytes: usize = 1024 * 1024 * 1024;
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
    expected_global_heads: usize,
    result_path: []const u8,
    candidate_checkpoint_path: []const u8,
    max_workspace_bytes: usize,
    retention_ladder: [3]usize,
    mass_gate: f64,
    output_gate: f64,
    global_savings_gate: f64,
    heldout_controls_admission: bool,
    heldout_controls_fit: bool,
    raw_selection_accelerated: bool = false,
    reject_exact_active_fixed_point: bool = false,
};
const HeadResult = struct {
    donor: usize,
    head: usize,
    rows_before: usize,
    rows_after: usize,
    source_head_bytes: usize,
    candidate_head_bytes: usize,
    accepted_chunks: usize,
    preserved_chunks: usize,
    final_mass_error: f64,
    final_output_rmse: f64,
    chunks: []replay.ChunkResult,
};
fn policyShape(plan: Plan) bool {
    return !(!std.mem.eql(u8, plan.schema, "antfly.attention-match-chunk-admission-plan.v13") or !std.fs.path.isAbsolute(plan.input_dir) or !std.fs.path.isAbsolute(plan.augmentation_dir) or !std.fs.path.isAbsolute(plan.result_path) or !std.fs.path.isAbsolute(plan.candidate_checkpoint_path) or std.mem.eql(u8, plan.result_path, plan.candidate_checkpoint_path) or !std.mem.eql(u8, plan.method, "top1") or plan.query_samples != 512 or plan.expected_global_heads != 8 or plan.max_workspace_bytes != workspace_bytes or plan.retention_ladder[0] != 4 or plan.retention_ladder[1] != 3 or plan.retention_ladder[2] != 2 or plan.mass_gate != 0.05 or plan.output_gate != 0.05 or plan.global_savings_gate != 1.5 or !plan.heldout_controls_admission or plan.heldout_controls_fit);
}

fn parseDigest(text: []const u8) ![32]u8 {
    if (text.len != 64) return error.InvalidWholeFitInput;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, text) catch return error.InvalidWholeFitInput;
    return out;
}

test "chunk cumulative admission all eight global heads source72 augmented replay" {
    if (comptime builtin.os.tag != .macos or !build_options.enable_metal) return error.SkipZigTest;
    const plan_path = platform.env.getenv("ANTFLY_METAL_CHUNK_ADMISSION_ALL_HEADS_PLAN") orelse return error.SkipZigTest;
    if (!std.fs.path.isAbsolute(plan_path)) return error.InvalidWholeFitInput;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, plan_path, allocator, .limited(1024 * 1024));
    defer allocator.free(bytes);
    const parsed = try std.json.parseFromSlice(Plan, allocator, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false });
    defer parsed.deinit();
    const plan = parsed.value;
    var plan_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &plan_digest, .{});
    if (!policyShape(plan)) return error.InvalidWholeFitInput;
    const manifest_path = try std.fs.path.join(allocator, &.{ plan.input_dir, "manifest.json" });
    defer allocator.free(manifest_path);
    const manifest = try std.Io.Dir.cwd().readFileAlloc(io, manifest_path, allocator, .limited(whole.manifest_reservation));
    defer allocator.free(manifest);
    var manifest_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(manifest, &manifest_digest, .{});
    if (!std.mem.eql(u8, &manifest_digest, &(try parseDigest(plan.input_manifest_sha256))) or plan.fixed_source65_manifest_sha256.len != 64) return error.InvalidWholeFitInput;
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
    if (input.query_samples != 512 or input.method != .top1 or input.source_has_metadata) return error.InvalidWholeFitInput;
    const Fitter = @import("../../ops/metal_attention_match_fit.zig").Fitter;
    var fitter = try Fitter.init(allocator, null, workspace_bytes);
    var fitter_live = true;
    defer if (fitter_live) fitter.deinit();
    const heads = try allocator.alloc(HeadResult, plan.expected_global_heads);
    var head_count: usize = 0;
    defer {
        for (heads[0..head_count]) |head| allocator.free(head.chunks);
        allocator.free(heads);
    }
    var modeled_source_bytes: usize = 0;
    var modeled_candidate_bytes: usize = 0;
    var candidate = try checkpoint.cloneMetadata(allocator, input.checkpoint_owned.state);
    defer candidate.deinit();
    const started = platform.time.monotonicNs();
    for (input.checkpoint_owned.state.donors, 0..) |donor, donor_index| for (donor.heads, 0..) |source_head, head_index| {
        if (donor.window != 0) {
            @constCast(candidate.state.donors[donor_index].heads)[head_index] = try checkpoint.cloneHead(allocator, source_head);
            continue;
        }
        if (head_count >= heads.len) return error.InvalidWholeFitInput;
        var fit_backend = fitter.backend();
        if (!plan.raw_selection_accelerated) fit_backend.raw_select_fn = null;
        const head_started = platform.time.monotonicNs();
        std.debug.print("chunk_admission_head phase=begin donor={d} head={d} raw_selection_accelerated={any}\n", .{ donor_index, head_index, plan.raw_selection_accelerated });
        var result = try replay.runInputWithPolicy(allocator, &input, augmented.batches, fit_backend, donor_index, head_index, workspace_bytes, plan.reject_exact_active_fixed_point);
        std.debug.print("chunk_admission_head phase=complete donor={d} head={d} rows={d} mass_error={d} output_rmse={d} elapsed_ns={d}\n", .{ donor_index, head_index, result.head.rows, result.final_mass_error, result.final_output_rmse, platform.time.monotonicNs() -| head_started });
        defer result.deinit();
        const rows_before = source_head.positions.len;
        const source_head_bytes = try std.math.mul(usize, try std.math.mul(usize, rows_before, source_head.dim), 4);
        const candidate_head_bytes = try std.math.add(usize, try std.math.mul(usize, try std.math.mul(usize, result.head.rows, source_head.dim), 4), try std.math.mul(usize, result.head.rows, 20));
        modeled_source_bytes = try std.math.add(usize, modeled_source_bytes, source_head_bytes);
        modeled_candidate_bytes = try std.math.add(usize, modeled_candidate_bytes, candidate_head_bytes);
        var accepted_chunks: usize = 0;
        for (result.chunks) |chunk| if (chunk.chosen_divisor != 0) {
            accepted_chunks += 1;
        };
        const keys = try allocator.alloc(f32, result.head.keys.len);
        var keys_owned = true;
        errdefer if (keys_owned) allocator.free(keys);
        const values = try allocator.alloc(f32, result.head.values.len);
        var values_owned = true;
        errdefer if (values_owned) allocator.free(values);
        const bias = try allocator.alloc(f32, result.head.bias.?.len);
        var bias_owned = true;
        errdefer if (bias_owned) allocator.free(bias);
        const positions = try allocator.alloc(u64, result.head.rows);
        var positions_owned = true;
        errdefer if (positions_owned) allocator.free(positions);
        const coverage = try allocator.alloc(u64, result.head.rows);
        var coverage_owned = true;
        errdefer if (coverage_owned) allocator.free(coverage);
        for (keys, result.head.keys) |*destination, value| destination.* = @floatCast(value);
        for (values, result.head.values) |*destination, value| destination.* = @floatCast(value);
        for (bias, result.head.bias.?) |*destination, value| destination.* = @floatCast(value);
        for (positions, result.source_indices, 0..) |*destination, source_index, index| {
            if (source_index >= source_head.positions.len or (index > 0 and source_index <= result.source_indices[index - 1])) return error.InvalidKvCheckpoint;
            destination.* = source_head.positions[source_index];
        }
        for (coverage, result.coverage_source_indices, 0..) |*destination, source_index, index| {
            if (source_index >= source_head.coverage_ends.len or (index > 0 and source_index < result.coverage_source_indices[index - 1])) return error.InvalidKvCheckpoint;
            destination.* = source_head.coverage_ends[source_index];
            if (destination.* <= positions[index] or destination.* > input.checkpoint_owned.state.logical_length) return error.InvalidKvCheckpoint;
        }
        const old_end = source_head.positions.len - input.fit.recent_rows;
        @constCast(candidate.state.donors[donor_index].heads)[head_index] = .{ .dim = source_head.dim, .compacted_until = if (accepted_chunks == 0) source_head.compacted_until else source_head.coverage_ends[old_end - 1], .positions = positions, .coverage_ends = coverage, .keys = keys, .values = values, .bias = bias };
        keys_owned = false;
        values_owned = false;
        bias_owned = false;
        positions_owned = false;
        coverage_owned = false;
        heads[head_count] = .{ .donor = donor_index, .head = head_index, .rows_before = rows_before, .rows_after = result.head.rows, .source_head_bytes = source_head_bytes, .candidate_head_bytes = candidate_head_bytes, .accepted_chunks = accepted_chunks, .preserved_chunks = result.chunks.len - accepted_chunks, .final_mass_error = result.final_mass_error, .final_output_rmse = result.final_output_rmse, .chunks = try allocator.dupe(replay.ChunkResult, result.chunks) };
        head_count += 1;
    };
    if (head_count != heads.len) return error.InvalidWholeFitInput;
    try candidate.state.validateWithControl(checkpoint.serving_limits, .{});
    const source_global_bytes = try engine.globalStorageBytes(input.checkpoint_owned.state, .f16, false);
    const candidate_global_bytes = try engine.globalStorageBytes(candidate.state, .f16, true);
    if (source_global_bytes != input.original_global_device_bytes or candidate_global_bytes == 0 or modeled_source_bytes != source_global_bytes or modeled_candidate_bytes != candidate_global_bytes) return error.InvalidWholeFitInput;
    const savings_ratio = @as(f64, @floatFromInt(source_global_bytes)) / @as(f64, @floatFromInt(candidate_global_bytes));
    const fitting_phase_counters = fitter.phaseCounters();
    std.debug.print("attention_match_fitting_phases raw_selection_calls={d} raw_selection_active_solves={d} scalar_rhs_solves={d} vector_rhs_solves={d} max_scalar_q={d} max_scalar_m={d} max_vector_q={d} max_vector_m={d} max_vector_d={d} diagnostic_only=1\n", .{
        fitting_phase_counters.raw_selection_calls,
        fitting_phase_counters.raw_selection_active_solves,
        fitting_phase_counters.scalar_rhs_solves,
        fitting_phase_counters.vector_rhs_solves,
        fitting_phase_counters.max_scalar_query_rows,
        fitting_phase_counters.max_scalar_selected_rows,
        fitting_phase_counters.max_vector_query_rows,
        fitting_phase_counters.max_vector_selected_rows,
        fitting_phase_counters.max_vector_width,
    });
    const accelerated_value_solves = fitter.deinitAndTakeValueSolves();
    fitter_live = false;
    const elapsed_ns = platform.time.monotonicNs() -| started;
    if (accelerated_value_solves == 0 or elapsed_ns == 0) return error.InvalidWholeFitInput;
    const expected_checkpoint_bytes = try checkpoint.encodedSize(candidate.state, checkpoint.serving_limits, .f16, .{});
    const checkpoint_bytes = try checkpoint.encodeWithEncoding(allocator, candidate.state, checkpoint.serving_limits, .f16, .{});
    defer allocator.free(checkpoint_bytes);
    if (checkpoint_bytes.len != expected_checkpoint_bytes) return error.InvalidKvCheckpoint;
    var checkpoint_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(checkpoint_bytes, &checkpoint_digest, .{});
    var checkpoint_file = try std.Io.Dir.createFileAbsolute(io, plan.candidate_checkpoint_path, .{ .exclusive = true, .truncate = false });
    defer checkpoint_file.close(io);
    try checkpoint_file.writeStreamingAll(io, checkpoint_bytes);
    try checkpoint_file.sync(io);
    const json = try std.json.Stringify.valueAlloc(allocator, .{ .schema = "antfly.attention-match-chunk-admission-result.v13", .authenticated = true, .plan_sha256 = std.fmt.bytesToHex(plan_digest, .lower), .input_manifest_sha256 = std.fmt.bytesToHex(manifest_digest, .lower), .augmentation_manifest_sha256 = std.fmt.bytesToHex(augmentation_digest, .lower), .fixed_source65_manifest_sha256 = plan.fixed_source65_manifest_sha256, .identity = plan.identity, .config_sha256 = plan.config_sha256, .parent_study_tokens_sha256 = plan.parent_study_tokens_sha256, .study_tokens_sha256 = plan.study_tokens_sha256, .source_freeze_sha256 = plan.source_freeze_sha256, .binary_sha256 = plan.binary_sha256, .exporter_sha256 = plan.exporter_sha256, .method = plan.method, .query_samples = plan.query_samples, .expected_global_heads = plan.expected_global_heads, .max_workspace_bytes = plan.max_workspace_bytes, .retention_ladder = plan.retention_ladder, .mass_gate = plan.mass_gate, .output_gate = plan.output_gate, .global_savings_gate = plan.global_savings_gate, .heldout_controls_admission = true, .heldout_controls_fit = false, .raw_selection_accelerated = plan.raw_selection_accelerated, .reject_exact_active_fixed_point = plan.reject_exact_active_fixed_point, .chronological_first_passing = true, .rejected_chunk_policy = "preserve_original", .storage_precision = "f16_keys_values_f32_bias", .final_uncached_crosscheck = true, .source_has_metadata = false, .candidate_has_metadata = true, .source_global_bytes = source_global_bytes, .candidate_global_bytes = candidate_global_bytes, .savings_ratio = savings_ratio, .meets_global_savings_gate = savings_ratio >= plan.global_savings_gate, .candidate_checkpoint = .{ .path = plan.candidate_checkpoint_path, .sha256 = std.fmt.bytesToHex(checkpoint_digest, .lower), .bytes = checkpoint_bytes.len, .encoding = "f16-v3", .exclusive = true, .device_install_measured = false, .owner_use_requires_global_savings_gate = true }, .checkpoint_encoding_memory_scope = "outer_8GiB_process_guard_after_fit_workspace_release", .accelerated_value_solves = accelerated_value_solves, .elapsed_ns = elapsed_ns, .heads = heads, .diagnostic_only = true, .quality_credit = false, .production_qualified = false, .publication_claim = false, .admission_claim = false }, .{});
    defer allocator.free(json);
    var file = try std.Io.Dir.createFileAbsolute(io, plan.result_path, .{ .exclusive = true, .truncate = false });
    defer file.close(io);
    try file.writeStreamingAll(io, json);
    try file.writeStreamingAll(io, "\n");
    try file.sync(io);
}

test "chunk admission v13 all-head policy shape is exact" {
    const digest = "0000000000000000000000000000000000000000000000000000000000000000";
    const good: Plan = .{ .schema = "antfly.attention-match-chunk-admission-plan.v13", .input_dir = "/input", .augmentation_dir = "/augmentation", .identity = .{ .weights = digest, .tokenizer = digest, .template = digest, .layout = digest }, .config_sha256 = digest, .source_freeze_sha256 = digest, .binary_sha256 = digest, .exporter_sha256 = digest, .input_manifest_sha256 = digest, .augmentation_manifest_sha256 = digest, .fixed_source65_manifest_sha256 = digest, .parent_study_tokens_sha256 = digest, .study_tokens_sha256 = digest, .method = "top1", .query_samples = 512, .expected_global_heads = 8, .result_path = "/result", .candidate_checkpoint_path = "/candidate.f16-v3", .max_workspace_bytes = workspace_bytes, .retention_ladder = .{ 4, 3, 2 }, .mass_gate = 0.05, .output_gate = 0.05, .global_savings_gate = 1.5, .heldout_controls_admission = true, .heldout_controls_fit = false, .raw_selection_accelerated = false };
    try std.testing.expect(policyShape(good));
    var bad = good;
    bad.global_savings_gate = 1.49;
    try std.testing.expect(!policyShape(bad));
    bad = good;
    bad.expected_global_heads = 7;
    try std.testing.expect(!policyShape(bad));
    bad = good;
    bad.heldout_controls_fit = true;
    try std.testing.expect(!policyShape(bad));
    bad = good;
    bad.candidate_checkpoint_path = bad.result_path;
    try std.testing.expect(!policyShape(bad));
}
