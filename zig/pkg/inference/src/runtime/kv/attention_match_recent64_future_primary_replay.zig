const std = @import("std");
const builtin = @import("builtin");
const checkpoint = @import("attention_match_checkpoint.zig");
const direct = @import("attention_match_recent64_future_primary_direct.zig");
const future_capture = @import("attention_match_future_train_p64_capture.zig");
const hybrid = @import("attention_match_recent64_future_primary_hybrid.zig");
const recent_fixture = @import("attention_match_recent64_captured_fixture.zig");
const heldout_bundle = @import("attention_match_prepared_heldout_bundle.zig");
const rejected_fixture = @import("attention_match_rejected_fixture.zig");
const model_runtime = @import("../../graph/model_runtime.zig");
const solver = @import("attention_match.zig");
const Fitter = @import("../../ops/metal_attention_match_fit.zig").Fitter;

pub const Binding = struct {
    recent_manifest_path: []const u8,
    recent_manifest_sha256: [32]u8,
    future_report_path: []const u8,
    future_report_sha256: [32]u8,
    expected_binary_sha256: [32]u8,
    expected_source_freeze_sha256: [32]u8,
    producer_binary_sha256: ?[32]u8 = null,
    producer_source_freeze_sha256: ?[32]u8 = null,
    allow_cross_producer_builtin_test: bool = false,
    source_afkv_path: []const u8,
    heldout_path: []const u8,
    heldout_authority_sha256: [32]u8,
    mode: direct.Mode,
    rejected_fixture_output_dir: ?[]const u8 = null,
    rejected_fixture_config_sha256: ?[32]u8 = null,
};

const FutureReport = struct {
    schema: []const u8,
    role: []const u8,
    diagnostic_only: bool,
    qualification_credit: bool,
    serving_activation: bool,
    source_checkpoint_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    binary_sha256: [32]u8,
    continuation_ids_sha256: [32]u8,
    continuation_origin_sha256: [32]u8,
    producer_receipt_sha256: [32]u8,
    training_authority_sha256: [32]u8,
    payload_sha256: [32]u8,
    values_sha256: [32]u8,
    positions_sha256: [32]u8,
    source_authority: model_runtime.ExactPagedKvSourceAuthority,
    branch_authority: model_runtime.ExactPagedKvSourceAuthority,
    batch_count: usize,
    heldout_rows: usize,
    train_rows: usize,
    first_train_position: usize,
    last_train_position: usize,
    native_plan_calls: u64,
    native_commit_calls: u64,
    native_copy_bytes: u64,
    capture_elapsed_ns: u64,
    host_scratch_envelope_bytes: usize,
    backend_scratch_envelope_bytes: usize,
    budget_host_scratch_before: usize,
    budget_backend_scratch_before: usize,
    budget_host_scratch_after_reserve: usize,
    budget_backend_scratch_after_reserve: usize,
    phase_count: ?usize = null,
    phase_counts: ?[]const usize = null,
};

fn digest(bytes: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return out;
}

fn readChecked(allocator: std.mem.Allocator, io: std.Io, path: []const u8, maximum: usize, expected: [32]u8) ![]u8 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(try std.math.add(usize, maximum, 1)));
    errdefer allocator.free(bytes);
    if (bytes.len > maximum or !std.mem.eql(u8, &digest(bytes), &expected)) return error.RecentFuturePrimaryFixtureChanged;
    return bytes;
}

fn validateProducerBinding(binding: Binding, captured_binary: [32]u8, captured_freeze: [32]u8, is_test: bool) !void {
    const producer_binary = binding.producer_binary_sha256 orelse binding.expected_binary_sha256;
    const producer_freeze = binding.producer_source_freeze_sha256 orelse binding.expected_source_freeze_sha256;
    const cross = !std.mem.eql(u8, &producer_binary, &binding.expected_binary_sha256) or !std.mem.eql(u8, &producer_freeze, &binding.expected_source_freeze_sha256);
    if (cross and (!is_test or !binding.allow_cross_producer_builtin_test or binding.producer_binary_sha256 == null or binding.producer_source_freeze_sha256 == null)) return error.RecentFuturePrimaryFixtureChanged;
    if (!std.mem.eql(u8, &captured_binary, &producer_binary) or !std.mem.eql(u8, &captured_freeze, &producer_freeze)) return error.RecentFuturePrimaryFixtureChanged;
}

fn futureProducerReceipt(report: FutureReport, payload_digest: [32]u8, phase_balanced: bool) [32]u8 {
    var receipt = std.crypto.hash.sha2.Sha256.init(.{});
    receipt.update(if (phase_balanced) "antfly.teacher-forced-future-train-phase-p64-producer-receipt.v1\x00" else "antfly.teacher-forced-future-train-p64-producer-receipt.v2\x00");
    receipt.update(&report.training_authority_sha256);
    receipt.update(&payload_digest);
    inline for (.{ report.source_authority.model_weights_sha256, report.source_authority.model_identity_layout_sha256, report.source_authority.context_layout_sha256, report.source_authority.token_stream_sha256, report.branch_authority.model_weights_sha256, report.branch_authority.model_identity_layout_sha256, report.branch_authority.context_layout_sha256, report.branch_authority.token_stream_sha256, report.continuation_ids_sha256, report.continuation_origin_sha256 }) |field| receipt.update(&field);
    inline for (.{ report.source_authority.source_revision, report.branch_authority.source_revision, @as(u64, report.source_authority.source_sequence), @as(u64, report.branch_authority.source_sequence) }) |value| {
        var bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &bytes, value, .little);
        receipt.update(&bytes);
    }
    return receipt.finalResult();
}

fn loadFuture(allocator: std.mem.Allocator, io: std.Io, report_path: []const u8, expected_report: [32]u8, recent: recent_fixture.Manifest, mode: direct.Mode) !future_capture.TrainingOutput {
    const report_bytes = try readChecked(allocator, io, report_path, 256 * 1024, expected_report);
    defer allocator.free(report_bytes);
    var parsed = try std.json.parseFromSlice(FutureReport, allocator, report_bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false });
    defer parsed.deinit();
    const report = parsed.value;
    const phase_balanced = mode == .full128_phase_balanced_g;
    const expected_first: usize = if (phase_balanced) 8129 else 8132;
    const expected_last: usize = if (phase_balanced) 8633 else 8636;
    const expected_calls: u64 = if (phase_balanced) 49 else 7;
    if (!std.mem.eql(u8, report.schema, if (phase_balanced) "antfly.teacher-forced-future-train-phase-p64-diagnostic.v1" else "antfly.teacher-forced-future-train-p64-diagnostic.v3") or !std.mem.eql(u8, report.role, if (phase_balanced) "TRAIN_P64_PHASE_BALANCED_V1" else "TRAIN_P64") or
        !report.diagnostic_only or report.qualification_credit or report.serving_activation or report.batch_count != future_capture.batch_count or
        report.train_rows != 64 or report.heldout_rows != 0 or report.first_train_position != expected_first or report.last_train_position != expected_last or
        report.native_plan_calls != expected_calls or report.native_commit_calls != expected_calls or report.native_copy_bytes != 7_340_032 or
        !std.mem.eql(u8, &report.source_checkpoint_sha256, &recent.source_checkpoint_sha256) or
        !std.mem.eql(u8, &report.source_freeze_sha256, &recent.source_freeze_sha256) or
        !std.mem.eql(u8, &report.binary_sha256, &recent.binary_sha256) or
        !std.mem.eql(u8, &report.source_authority.model_identity_layout_sha256, &recent.identity.layout) or
        !std.mem.eql(u8, &report.source_authority.model_weights_sha256, &recent.identity.weights) or
        !std.mem.eql(u8, &report.branch_authority.model_identity_layout_sha256, &recent.identity.layout) or
        !std.mem.eql(u8, &report.branch_authority.model_weights_sha256, &recent.identity.weights) or
        !std.mem.eql(u8, &report.branch_authority.context_layout_sha256, &report.source_authority.context_layout_sha256) or
        report.source_authority.logical_tokens != 8128 or report.branch_authority.logical_tokens != 8640 or
        report.branch_authority.source_sequence != report.source_authority.source_sequence or report.branch_authority.source_revision == report.source_authority.source_revision)
        return error.InvalidRecentFuturePrimaryFixture;
    if (phase_balanced and (report.phase_count != 7 or report.phase_counts == null or !std.mem.eql(usize, report.phase_counts.?, &.{ 10, 9, 9, 9, 9, 9, 9 }))) return error.InvalidRecentFuturePrimaryFixture;
    if (report.host_scratch_envelope_bytes != 32 * 1024 * 1024 or report.backend_scratch_envelope_bytes != 8 * 1024 * 1024 or
        report.budget_host_scratch_after_reserve != try std.math.add(usize, report.budget_host_scratch_before, report.host_scratch_envelope_bytes) or
        report.budget_backend_scratch_after_reserve != try std.math.add(usize, report.budget_backend_scratch_before, report.backend_scratch_envelope_bytes))
        return error.InvalidRecentFuturePrimaryFixture;
    try report.source_authority.validate();
    try report.branch_authority.validate();
    const root = std.fs.path.dirname(report_path) orelse return error.InvalidRecentFuturePrimaryFixture;
    const values_path = try std.fs.path.join(allocator, &.{ root, if (phase_balanced) "future-train-phase-p64-values.f32" else "future-train-p64-values.f32" });
    defer allocator.free(values_path);
    const positions_path = try std.fs.path.join(allocator, &.{ root, if (phase_balanced) "future-train-phase-p64-positions.u64" else "future-train-p64-positions.u64" });
    defer allocator.free(positions_path);
    const value_count = future_capture.batch_count * future_capture.sample_rows * future_capture.head_dim;
    const position_count = future_capture.batch_count * future_capture.sample_rows;
    const value_bytes = try readChecked(allocator, io, values_path, value_count * @sizeOf(f32), report.values_sha256);
    defer allocator.free(value_bytes);
    const position_bytes = try readChecked(allocator, io, positions_path, position_count * @sizeOf(u64), report.positions_sha256);
    defer allocator.free(position_bytes);
    const values = try allocator.alloc(f32, value_count);
    errdefer allocator.free(values);
    @memcpy(std.mem.sliceAsBytes(values), value_bytes);
    const positions = try allocator.alloc(u64, position_count);
    errdefer allocator.free(positions);
    @memcpy(std.mem.sliceAsBytes(positions), position_bytes);
    var payload = std.crypto.hash.sha2.Sha256.init(.{});
    payload.update(if (phase_balanced) "antfly.attention-match-teacher-forced-future-train-phase-p64.v1.payload\x00" else future_capture.schema ++ ".payload\x00");
    payload.update(value_bytes);
    payload.update(position_bytes);
    const payload_digest = payload.finalResult();
    if (!std.mem.eql(u8, &payload_digest, &report.payload_sha256)) return error.RecentFuturePrimaryFixtureChanged;
    const receipt_digest = futureProducerReceipt(report, payload_digest, phase_balanced);
    if (!std.mem.eql(u8, &receipt_digest, &report.producer_receipt_sha256)) return error.RecentFuturePrimaryFixtureChanged;
    const batches = try allocator.alloc(@import("attention_match_engine.zig").ReferenceBatch, future_capture.batch_count);
    errdefer allocator.free(batches);
    for (batches, 0..) |*batch, index| {
        const consumer_index = index / 8;
        const consumers = recent_fixture.canonical_consumers;
        const train = values[index * 64 * 512 ..][0 .. 64 * 512];
        const train_positions = positions[index * 64 ..][0..64];
        for (train_positions, 0..) |position, row| if (position != if (phase_balanced) 8129 + 8 * row + row % 7 else 8132 + 8 * row) return error.InvalidRecentFuturePrimaryFixture;
        batch.* = .{ .consumer_layer = consumers[consumer_index], .query_head = @intCast(index % 8), .train = train, .train_positions = train_positions, .heldout = &.{}, .heldout_positions = &.{} };
    }
    return .{ .allocator = allocator, .batches = batches, .values = values, .positions = positions, .authority_sha256 = receipt_digest, .payload_sha256 = payload_digest };
}

fn trainingMergeMode(mode: direct.Mode) hybrid.Mode {
    return if (mode == .full128_phase_balanced_g) .full_recent64_future64_phase_balanced else if (mode == .full128_incremental_reduction_e or mode == .full128_budget_91_95_3_f) .full_recent64_future64 else .fixed_recent32_future32;
}

test "mode F retains the exact full128 merge geometry" {
    try std.testing.expectEqual(hybrid.Mode.full_recent64_future64, trainingMergeMode(.full128_incremental_reduction_e));
    try std.testing.expectEqual(hybrid.Mode.full_recent64_future64, trainingMergeMode(.full128_budget_91_95_3_f));
    try std.testing.expectEqual(hybrid.Mode.fixed_recent32_future32, trainingMergeMode(.incremental_reduction_c));
}

pub fn run(allocator: std.mem.Allocator, io: std.Io, binding: Binding, control: solver.Control) !direct.Result {
    const recent_manifest_bytes = try readChecked(allocator, io, binding.recent_manifest_path, 256 * 1024, binding.recent_manifest_sha256);
    defer allocator.free(recent_manifest_bytes);
    var recent_manifest = try std.json.parseFromSlice(recent_fixture.Manifest, allocator, recent_manifest_bytes, .{ .allocate = .alloc_always });
    defer recent_manifest.deinit();
    try recent_fixture.validateManifest(recent_manifest.value);
    try validateProducerBinding(binding, recent_manifest.value.binary_sha256, recent_manifest.value.source_freeze_sha256, builtin.is_test);
    const recent_root = std.fs.path.dirname(binding.recent_manifest_path) orelse return error.InvalidRecentFuturePrimaryFixture;
    const rv_path = try std.fs.path.join(allocator, &.{ recent_root, recent_fixture.values_name });
    defer allocator.free(rv_path);
    const rp_path = try std.fs.path.join(allocator, &.{ recent_root, recent_fixture.positions_name });
    defer allocator.free(rp_path);
    const rv = try readChecked(allocator, io, rv_path, recent_fixture.value_count * @sizeOf(f32), recent_manifest.value.values.sha256);
    defer allocator.free(rv);
    const rp = try readChecked(allocator, io, rp_path, recent_fixture.position_count * @sizeOf(u64), recent_manifest.value.positions.sha256);
    defer allocator.free(rp);
    var recent = try recent_fixture.reconstruct(allocator, recent_manifest.value, rv, rp);
    defer recent.deinit();
    var future = try loadFuture(allocator, io, binding.future_report_path, binding.future_report_sha256, recent_manifest.value, binding.mode);
    defer future.deinit();
    const hybrid_mode = trainingMergeMode(binding.mode);
    var mixed = try hybrid.merge(allocator, recent.batches, recent.authority_sha256, future.batches, future.authority_sha256, hybrid_mode);
    defer mixed.deinit();
    const source_bytes = try readChecked(allocator, io, binding.source_afkv_path, checkpoint.serving_limits.max_file_bytes, recent_manifest.value.source_checkpoint_sha256);
    defer allocator.free(source_bytes);
    var source = try checkpoint.decodeWithControl(allocator, source_bytes, .{ .weights = recent_manifest.value.identity.weights, .tokenizer = recent_manifest.value.identity.tokenizer, .template = recent_manifest.value.identity.template, .layout = recent_manifest.value.identity.layout }, checkpoint.serving_limits, control);
    defer source.deinit();
    const hho_bytes = try std.Io.Dir.cwd().readFileAlloc(io, binding.heldout_path, allocator, .limited(16 * 1024 * 1024 + 1));
    defer allocator.free(hho_bytes);
    var hho = try heldout_bundle.decodeOwned(allocator, hho_bytes, binding.heldout_authority_sha256);
    defer hho.deinit();
    if ((binding.rejected_fixture_output_dir == null) != (binding.rejected_fixture_config_sha256 == null)) return error.InvalidRecentFuturePrimaryFixture;
    var rejected_writer: ?rejected_fixture.Writer = if (binding.rejected_fixture_output_dir) |output_dir| blk: {
        if (!builtin.is_test or !std.fs.path.isAbsolute(output_dir) or binding.mode != .full128_incremental_reduction_e) return error.InvalidRecentFuturePrimaryFixture;
        break :blk try rejected_fixture.Writer.init(allocator, io, .{ .output_dir = output_dir, .identity = source.state.identity, .config_sha256 = binding.rejected_fixture_config_sha256.? });
    } else null;
    defer if (rejected_writer) |*writer| writer.deinit();
    var fitter = try Fitter.initWorker(allocator, null, direct.head_workspace_bytes, direct.head_workspace_bytes);
    defer _ = fitter.deinitAndTakeValueSolves();
    // The solver choice is part of D's explicit numerical policy and must be
    // installed before profiling, backend exposure, or any fit operation.
    if (binding.mode == .optimized_gram_advisory_d) try fitter.setRawSelectionSolver(.fp64_gram_advisory);
    try fitter.setPhaseProfiling(true);
    const capture = if (rejected_writer) |*writer| writer.callback() else null;
    const result = direct.runWithRejectedCapture(allocator, source.state, mixed, hho, fitter.backend(), control, binding.mode, capture) catch |err| {
        std.debug.print("recent-future-primary-fitter-failure-profile-json {f}\n", .{std.json.fmt(.{ .schema = "antfly.recent-future-primary-fitter-failure-profile.v1", .mode = @tagName(binding.mode), .phase_counters = fitter.phaseCounters(), .error_name = @errorName(err), .diagnostic_only = true, .qualification_credit = false }, .{})});
        if (rejected_writer) |*writer| {
            const captured = try writer.finish();
            if (!captured.committed or captured.required_bytes == 0 or captured.required_bytes > rejected_fixture.maximum_bytes) return error.InvalidRejectedHeadFixture;
            std.debug.print("attention_match_v18_rejected_head_fixture committed=1 bytes={d} divisor=2 diagnostic_only=true quality_credit=false\n", .{captured.required_bytes});
        }
        return err;
    };
    const counters = fitter.phaseCounters();
    if (binding.mode == .optimized_gram_advisory_d and
        (counters.raw_selection_solver != .fp64_gram_advisory or counters.raw_selection_calls == 0 or
            counters.gram_advisory.calls == 0 or counters.gram_advisory.candidate_attempts == 0 or counters.gram_advisory.native_original_p_kkt_checks == 0))
        return error.InvalidRecentFuturePrimaryDiagnostic;
    if (binding.mode != .optimized_gram_advisory_d and counters.raw_selection_solver != .householder)
        return error.InvalidRecentFuturePrimaryDiagnostic;
    std.debug.print("recent-future-primary-fitter-success-profile-json {f}\n", .{std.json.fmt(.{ .schema = "antfly.recent-future-primary-fitter-success-profile.v1", .mode = @tagName(binding.mode), .phase_counters = counters, .diagnostic_only = true, .qualification_credit = false }, .{})});
    return result;
}

test "cross producer binding is explicit builtin-test-only and mutation-sensitive" {
    const base: Binding = .{ .recent_manifest_path = "r", .recent_manifest_sha256 = @splat(1), .future_report_path = "p", .future_report_sha256 = @splat(2), .expected_binary_sha256 = @splat(3), .expected_source_freeze_sha256 = @splat(4), .source_afkv_path = "s", .heldout_path = "h", .heldout_authority_sha256 = @splat(5), .mode = .established_a };
    try validateProducerBinding(base, @splat(3), @splat(4), false);
    var cross = base;
    cross.producer_binary_sha256 = @splat(6);
    cross.producer_source_freeze_sha256 = @splat(7);
    cross.allow_cross_producer_builtin_test = true;
    try std.testing.expectError(error.RecentFuturePrimaryFixtureChanged, validateProducerBinding(cross, @splat(6), @splat(7), false));
    try validateProducerBinding(cross, @splat(6), @splat(7), true);
    try std.testing.expectError(error.RecentFuturePrimaryFixtureChanged, validateProducerBinding(cross, @splat(8), @splat(7), true));
    cross.producer_source_freeze_sha256 = null;
    try std.testing.expectError(error.RecentFuturePrimaryFixtureChanged, validateProducerBinding(cross, @splat(6), @splat(7), true));
}

test "future producer receipt binds training authority and source branch revisions" {
    var report: FutureReport = undefined;
    report.training_authority_sha256 = @splat(1);
    report.continuation_ids_sha256 = @splat(2);
    report.continuation_origin_sha256 = @splat(3);
    report.source_authority.model_weights_sha256 = @splat(4);
    report.source_authority.model_identity_layout_sha256 = @splat(5);
    report.source_authority.context_layout_sha256 = @splat(6);
    report.source_authority.token_stream_sha256 = @splat(7);
    report.source_authority.source_revision = 8;
    report.source_authority.source_sequence = 9;
    report.branch_authority.model_weights_sha256 = @splat(10);
    report.branch_authority.model_identity_layout_sha256 = @splat(11);
    report.branch_authority.context_layout_sha256 = @splat(12);
    report.branch_authority.token_stream_sha256 = @splat(13);
    report.branch_authority.source_revision = 14;
    report.branch_authority.source_sequence = 15;
    const expected = futureProducerReceipt(report, @splat(16), false);
    try std.testing.expect(!std.mem.eql(u8, &expected, &futureProducerReceipt(report, @splat(16), true)));
    report.training_authority_sha256[0] ^= 1;
    try std.testing.expect(!std.mem.eql(u8, &expected, &futureProducerReceipt(report, @splat(16), false)));
    report.training_authority_sha256[0] ^= 1;
    report.branch_authority.source_revision += 1;
    try std.testing.expect(!std.mem.eql(u8, &expected, &futureProducerReceipt(report, @splat(16), false)));
}
