const std = @import("std");
const platform = @import("antfly_platform");
const checkpoint = @import("runtime/kv/attention_match_checkpoint.zig");
const future = @import("runtime/kv/attention_match_future_decode_training.zig");
const whole = @import("runtime/kv/attention_match_whole_fit_input.zig");
const augmentation = @import("runtime/kv/attention_match_training_augmentation.zig");
const coverage = @import("runtime/kv/attention_match_production768_historical_coverage.zig");
const benchmark = @import("runtime/kv/attention_match_native_coverage_probability_benchmark.zig");
const admission = @import("runtime/kv/attention_match_cumulative_cache.zig");

const Ref = struct { path: []const u8, sha256: [32]u8 };
const Coverage = struct {
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
    historical_coverage: ?Coverage,
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
    query_reservation_bytes: usize,
    diagnostic_only: bool,
    qualification_credit: bool,
    production_qualified: bool,
};

fn hashFile(path: []const u8) ![32]u8 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, std.testing.allocator, .limited(512 * 1024 * 1024));
    defer std.testing.allocator.free(bytes);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return out;
}
fn authenticate(ref: Ref) !void {
    if (!std.fs.path.isAbsolute(ref.path)) return error.InvalidOfflineReplayBinding;
    const actual = try hashFile(ref.path);
    if (!std.mem.eql(u8, &actual, &ref.sha256)) return error.InvalidOfflineReplayBinding;
}
fn ranges() [23]admission.Range {
    var out: [23]admission.Range = undefined;
    var start: usize = 128;
    for (&out, 0..) |*range, i| {
        range.* = .{ .start = start, .rows = if (i + 1 == out.len) 128 else 256 };
        start += range.rows;
    }
    return out;
}
fn batchBytes(batches: []const @import("runtime/kv/attention_match_engine.zig").ReferenceBatch) !usize {
    var total = try std.math.mul(usize, batches.len, @sizeOf(@TypeOf(batches[0])));
    for (batches) |batch| {
        total = try std.math.add(usize, total, try std.math.mul(usize, batch.train.len, @sizeOf(f32)));
        total = try std.math.add(usize, total, try std.math.mul(usize, batch.train_positions.len + batch.heldout_positions.len, @sizeOf(u64)));
    }
    return total;
}

test "context authenticated captured native coverage scalar accelerate benchmark" {
    const binding_path = platform.env.getenv("ANTFLY_NATIVE_COVERAGE_PROBABILITY_BENCHMARK_BINDING") orelse return error.SkipZigTest;
    const result_path = platform.env.getenv("ANTFLY_NATIVE_COVERAGE_PROBABILITY_BENCHMARK_RESULT") orelse return error.InvalidOfflineReplayBinding;
    const donor = try std.fmt.parseInt(u32, platform.env.getenv("ANTFLY_NATIVE_COVERAGE_PROBABILITY_BENCHMARK_DONOR") orelse return error.InvalidOfflineReplayBinding, 10);
    const head = try std.fmt.parseInt(usize, platform.env.getenv("ANTFLY_NATIVE_COVERAGE_PROBABILITY_BENCHMARK_HEAD") orelse return error.InvalidOfflineReplayBinding, 10);
    if (!std.fs.path.isAbsolute(binding_path) or !std.fs.path.isAbsolute(result_path)) return error.InvalidOfflineReplayBinding;
    const binding_bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, binding_path, std.testing.allocator, .limited(256 * 1024));
    defer std.testing.allocator.free(binding_bytes);
    var parsed = try std.json.parseFromSlice(Binding, std.testing.allocator, binding_bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const b = parsed.value;
    if (!b.diagnostic_only or b.qualification_credit or b.production_qualified) return error.InvalidOfflineReplayBinding;
    const dense_binding = b.historical_coverage orelse return error.InvalidOfflineReplayBinding;
    inline for (.{ b.whole_manifest, b.complete_manifest, b.producer_source_freeze, b.producer_binary, b.producer_profile, b.integrity_closure, dense_binding.dense_outer_manifest, dense_binding.dense_nested_manifest, dense_binding.dense_reference_parent_manifest }) |ref| try authenticate(ref);
    var authority = b.authority;
    authority.generation_contract_sha256 = future.policyDigest();
    const authority_digest = authority.digest();
    if (!std.mem.eql(u8, &authority_digest, &b.complete_authority_sha256)) return error.InvalidOfflineReplayBinding;
    var input = try whole.read(std.testing.allocator, std.testing.io, .{ .output_dir = b.whole_fit_dir, .identity = b.identity, .config_sha256 = b.config_sha256, .source_freeze_sha256 = b.source_freeze_sha256, .binary_sha256 = b.binary_sha256, .exporter_sha256 = b.exporter_sha256, .limits = checkpoint.serving_limits });
    defer input.deinit();
    var complete = try @import("runtime/kv/attention_match_future_augmentation.zig").readComplete(std.testing.allocator, std.testing.io, .{ .output_dir = b.complete_future_dir, .manifest_sha256 = b.complete_manifest_sha256, .primary_parent_manifest_sha256 = b.parent_manifest_sha256, .identity = b.identity, .source_freeze_sha256 = b.source_freeze_sha256, .binary_sha256 = b.binary_sha256, .exporter_sha256 = b.exporter_sha256, .authority = authority, .historical_study_tokens_sha256 = b.historical_study_tokens_sha256, .primary_batches = input.batches });
    defer complete.deinit();
    try coverage.authenticateOuter(std.testing.allocator, std.testing.io, .{ .manifest_path = dense_binding.dense_outer_manifest.path, .manifest_sha256 = dense_binding.dense_outer_manifest.sha256, .nested_manifest_sha256 = dense_binding.dense_nested_manifest.sha256, .parent_manifest_sha256 = dense_binding.dense_reference_parent_manifest_sha256, .identity = b.identity, .source_freeze_sha256 = b.source_freeze_sha256, .binary_sha256 = b.binary_sha256, .exporter_sha256 = b.exporter_sha256 });
    var dense = try augmentation.readReferenceCoverageDense512(std.testing.allocator, std.testing.io, .{ .output_dir = dense_binding.dense_dir, .max_bytes = dense_binding.maximum_workspace_bytes - 64 * 1024, .parent_manifest_sha256 = b.parent_manifest_sha256, .identity = b.identity, .source_freeze_sha256 = b.source_freeze_sha256, .binary_sha256 = b.binary_sha256, .exporter_sha256 = b.exporter_sha256, .parent_study_tokens_sha256 = dense_binding.dense_parent_study_tokens_sha256, .study_tokens_sha256 = dense_binding.dense_study_tokens_sha256, .samples_per_stratum = 512 });
    defer dense.deinit();
    var dense_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(dense.manifest_bytes, &dense_digest, .{});
    if (!std.mem.eql(u8, &dense_digest, &dense_binding.dense_nested_manifest.sha256)) return error.InvalidOfflineReplayBinding;
    var live = try input.checkpoint_owned.state.residentBytes();
    live = try std.math.add(usize, live, try batchBytes(input.batches));
    live = try std.math.add(usize, live, try batchBytes(dense.batches));
    live = try std.math.add(usize, live, try batchBytes(complete.future.batches));
    live = try std.math.add(usize, live, try batchBytes(complete.historical.batches));
    live = try std.math.add(usize, live, b.query_reservation_bytes);
    const before = try hashFile(b.producer_source_freeze.path);
    const range_values = ranges();
    const measured = try benchmark.run(std.testing.allocator, input.checkpoint_owned.state, input.layout, input.batches, &dense, complete.future.batches, .{ .donor_layer = donor, .head_index = head, .ranges = &range_values, .combined_cap = 1024 * 1024 * 1024, .external_live_bytes = live });
    const after = try hashFile(b.producer_source_freeze.path);
    if (!std.mem.eql(u8, &before, &after) or !std.mem.eql(u8, &after, &b.producer_source_freeze.sha256)) return error.InvalidOfflineReplayBinding;
    const body = try std.json.Stringify.valueAlloc(std.testing.allocator, .{ .schema = "antfly.native-coverage-probability-benchmark.v2", .diagnostic_only = true, .qualification_credit = false, .formal_ttft_credit = false, .policy = benchmark.policy, .donor_layer = donor, .head_index = head, .dataset_live_bytes = live, .combined_cap = @as(usize, 1024 * 1024 * 1024), .scalar_elapsed_ns = measured.scalar_elapsed_ns, .accelerated_elapsed_ns = measured.accelerated_elapsed_ns, .max_probability_weight_abs_diff = measured.max_probability_weight_abs_diff, .baseline_covered_equal = measured.baseline_covered_equal, .selected_indices_equal = measured.selected_indices_equal, .scalar_authority_digest = measured.scalar_authority_digest, .accelerated_authority_digest = measured.accelerated_authority_digest, .accelerated_calls = measured.accelerated_calls, .accelerated_workspace_peak_bytes = measured.accelerated_workspace_peak_bytes, .source_stable = true, .hho_query_values_supplied_to_adapters = false }, .{});
    defer std.testing.allocator.free(body);
    var output = try std.Io.Dir.cwd().createFile(std.testing.io, result_path, .{ .exclusive = true });
    defer output.close(std.testing.io);
    try output.writeStreamingAll(std.testing.io, body);
    try output.writeStreamingAll(std.testing.io, "\n");
    try output.sync(std.testing.io);
}
