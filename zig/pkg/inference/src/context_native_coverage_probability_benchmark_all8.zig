const std = @import("std");
const platform = @import("antfly_platform");
const checkpoint = @import("runtime/kv/attention_match_checkpoint.zig");
const future = @import("runtime/kv/attention_match_future_decode_training.zig");
const future_augmentation = @import("runtime/kv/attention_match_future_augmentation.zig");
const whole = @import("runtime/kv/attention_match_whole_fit_input.zig");
const augmentation = @import("runtime/kv/attention_match_training_augmentation.zig");
const coverage = @import("runtime/kv/attention_match_production768_historical_coverage.zig");
const benchmark = @import("runtime/kv/attention_match_native_coverage_probability_benchmark.zig");
const accelerate = @import("runtime/kv/attention_match_accelerate_fp64_probability_scan.zig");
const admission = @import("runtime/kv/attention_match_cumulative_cache.zig");
const native_merge = @import("runtime/kv/attention_match_native_coverage_training_merge.zig");

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
    legacy_scalar_evidence: Ref,
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

const LegacyClosure = struct {
    schema: []const u8,
    pass: bool,
    diagnostic_only: bool,
    quality_credit: bool,
    production_qualified: bool,
    historical_selector: struct {
        batch_count: usize,
        selection_digest: []const u8,
        derived_historical_sha256: []const u8,
    },
};

fn decodeHex32(value: []const u8) ![32]u8 {
    if (value.len != 64) return error.InvalidOfflineReplayBinding;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, value) catch return error.InvalidOfflineReplayBinding;
    return out;
}

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
fn denseOwnerBytes(owned: augmentation.Owned) !usize {
    var bytes = owned.manifest_bytes.len;
    bytes = try std.math.add(usize, bytes, try std.math.mul(usize, owned.batches.len, @sizeOf(@import("runtime/kv/attention_match_engine.zig").ReferenceBatch)));
    bytes = try std.math.add(usize, bytes, try std.math.mul(usize, owned.training.len, @sizeOf(f32)));
    bytes = try std.math.add(usize, bytes, try std.math.mul(usize, owned.positions.len, @sizeOf(u64)));
    inline for (.{ owned.consumer_layers, owned.query_heads, owned.counts, owned.dimensions }) |items|
        bytes = try std.math.add(usize, bytes, try std.math.mul(usize, items.len, @sizeOf(u32)));
    return bytes;
}

fn batchBytes(batches: []const @import("runtime/kv/attention_match_engine.zig").ReferenceBatch) !usize {
    var total = try std.math.mul(usize, batches.len, @sizeOf(@TypeOf(batches[0])));
    for (batches) |batch| {
        total = try std.math.add(usize, total, try std.math.mul(usize, batch.train.len, @sizeOf(f32)));
        total = try std.math.add(usize, total, try std.math.mul(usize, batch.train_positions.len + batch.heldout_positions.len, @sizeOf(u64)));
    }
    return total;
}

fn ownedBytes(value: coverage.Owned) !usize {
    return value.liveBytes();
}

fn batchOutputsEqual(a: coverage.Owned, b: coverage.Owned) bool {
    if (a.batches.len != 56 or b.batches.len != 56 or !std.mem.eql(u8, std.mem.sliceAsBytes(a.values), std.mem.sliceAsBytes(b.values)) or !std.mem.eql(u8, std.mem.sliceAsBytes(a.positions), std.mem.sliceAsBytes(b.positions))) return false;
    for (a.batches, b.batches) |x, y| {
        if (x.consumer_layer != y.consumer_layer or x.query_head != y.query_head or !std.mem.eql(u8, std.mem.sliceAsBytes(x.train), std.mem.sliceAsBytes(y.train)) or !std.mem.eql(u8, std.mem.sliceAsBytes(x.train_positions), std.mem.sliceAsBytes(y.train_positions))) return false;
    }
    return true;
}

fn stage2HeadDigest(state: checkpoint.State, layout: @import("runtime/kv/attention_match_engine.zig").Layout, donor: usize, head_index: usize, batches: []const @import("runtime/kv/attention_match_engine.zig").ReferenceBatch) ![32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.native-coverage-selected-historical-positions.v1\x00");
    hash.update(std.mem.asBytes(&donor));
    hash.update(std.mem.asBytes(&head_index));
    var count: usize = 0;
    for (batches) |batch| {
        const mapped_donor = state.layer_donors[batch.consumer_layer];
        const group = layout.query_heads_by_layer[batch.consumer_layer] / state.donors[mapped_donor].heads.len;
        if (mapped_donor != donor or batch.query_head / group != head_index) continue;
        hash.update(std.mem.asBytes(&batch.consumer_layer));
        hash.update(std.mem.asBytes(&batch.query_head));
        hash.update(std.mem.sliceAsBytes(batch.train_positions));
        count += 1;
    }
    if (count != 4 and count != 16) return error.InvalidOfflineReplayBinding;
    return hash.finalResult();
}

test "context authenticated complete all8 scalar gathered accelerate parity" {
    const binding_path = platform.env.getenv("ANTFLY_NATIVE_COVERAGE_ALL8_PARITY_BINDING") orelse return error.SkipZigTest;
    const result_path = platform.env.getenv("ANTFLY_NATIVE_COVERAGE_ALL8_PARITY_RESULT") orelse return error.InvalidOfflineReplayBinding;
    if (!std.fs.path.isAbsolute(binding_path) or !std.fs.path.isAbsolute(result_path)) return error.InvalidOfflineReplayBinding;
    const binding_bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, binding_path, std.testing.allocator, .limited(256 * 1024));
    defer std.testing.allocator.free(binding_bytes);
    var parsed = try std.json.parseFromSlice(Binding, std.testing.allocator, binding_bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const b = parsed.value;
    if (!std.mem.eql(u8, b.schema, "antfly.native-coverage-all8-parity-binding.v2") or !b.diagnostic_only or b.qualification_credit or b.production_qualified) return error.InvalidOfflineReplayBinding;
    const dense_binding = b.historical_coverage orelse return error.InvalidOfflineReplayBinding;
    inline for (.{ b.whole_manifest, b.complete_manifest, b.producer_source_freeze, b.producer_binary, b.producer_profile, b.integrity_closure, b.legacy_scalar_evidence, dense_binding.dense_outer_manifest, dense_binding.dense_nested_manifest, dense_binding.dense_reference_parent_manifest }) |item| try authenticate(item);
    const legacy_bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, b.legacy_scalar_evidence.path, std.testing.allocator, .limited(256 * 1024));
    defer std.testing.allocator.free(legacy_bytes);
    var legacy = try std.json.parseFromSlice(LegacyClosure, std.testing.allocator, legacy_bytes, .{ .ignore_unknown_fields = true });
    defer legacy.deinit();
    if (!std.mem.eql(u8, legacy.value.schema, "antfly.production768-historical-coverage-dev-comparison-closure.v1") or !legacy.value.pass or !legacy.value.diagnostic_only or legacy.value.quality_credit or legacy.value.production_qualified or legacy.value.historical_selector.batch_count != 56) return error.InvalidOfflineReplayBinding;
    const expected_scalar_selection = try decodeHex32(legacy.value.historical_selector.selection_digest);
    const expected_scalar_historical = try decodeHex32(legacy.value.historical_selector.derived_historical_sha256);
    var authority = b.authority;
    authority.generation_contract_sha256 = future.policyDigest();
    const authority_digest = authority.digest();
    if (!std.mem.eql(u8, &authority_digest, &b.complete_authority_sha256)) return error.InvalidOfflineReplayBinding;
    var input = try whole.read(std.testing.allocator, std.testing.io, .{ .output_dir = b.whole_fit_dir, .identity = b.identity, .config_sha256 = b.config_sha256, .source_freeze_sha256 = b.source_freeze_sha256, .binary_sha256 = b.binary_sha256, .exporter_sha256 = b.exporter_sha256, .limits = checkpoint.serving_limits });
    defer input.deinit();
    var complete = try future_augmentation.readComplete(std.testing.allocator, std.testing.io, .{ .output_dir = b.complete_future_dir, .manifest_sha256 = b.complete_manifest_sha256, .primary_parent_manifest_sha256 = b.parent_manifest_sha256, .identity = b.identity, .source_freeze_sha256 = b.source_freeze_sha256, .binary_sha256 = b.binary_sha256, .exporter_sha256 = b.exporter_sha256, .authority = authority, .historical_study_tokens_sha256 = b.historical_study_tokens_sha256, .primary_batches = input.batches });
    defer complete.deinit();
    try coverage.authenticateOuter(std.testing.allocator, std.testing.io, .{ .manifest_path = dense_binding.dense_outer_manifest.path, .manifest_sha256 = dense_binding.dense_outer_manifest.sha256, .nested_manifest_sha256 = dense_binding.dense_nested_manifest.sha256, .parent_manifest_sha256 = dense_binding.dense_reference_parent_manifest_sha256, .identity = b.identity, .source_freeze_sha256 = b.source_freeze_sha256, .binary_sha256 = b.binary_sha256, .exporter_sha256 = b.exporter_sha256 });
    var dense = try augmentation.readReferenceCoverageDense512(std.testing.allocator, std.testing.io, .{ .output_dir = dense_binding.dense_dir, .max_bytes = dense_binding.maximum_workspace_bytes - 64 * 1024, .parent_manifest_sha256 = b.parent_manifest_sha256, .identity = b.identity, .source_freeze_sha256 = b.source_freeze_sha256, .binary_sha256 = b.binary_sha256, .exporter_sha256 = b.exporter_sha256, .parent_study_tokens_sha256 = dense_binding.dense_parent_study_tokens_sha256, .study_tokens_sha256 = dense_binding.dense_study_tokens_sha256, .samples_per_stratum = 512 });
    defer dense.deinit();
    var dense_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(dense.manifest_bytes, &dense_digest, .{});
    if (!std.mem.eql(u8, &dense_digest, &dense_binding.dense_nested_manifest.sha256) or input.batches.len != 56 or complete.future.batches.len != 56 or dense.batches.len != 56) return error.InvalidOfflineReplayBinding;
    var live = try input.liveBytes();
    live = try std.math.add(usize, live, try denseOwnerBytes(dense));
    live = try std.math.add(usize, live, try denseOwnerBytes(complete.historical));
    live = try std.math.add(usize, live, try denseOwnerBytes(complete.future));
    // Bound binding bytes, parsed metadata and the small controller arrays in
    // this pinned diagnostic. Future fit query reservations are not allocated
    // during probability parity and do not overlap this phase.
    live = try std.math.add(usize, live, 4 * 1024 * 1024);
    const before = try hashFile(b.producer_source_freeze.path);
    const range_values = ranges();
    var identities: [8][2]usize = undefined;
    var identity_count: usize = 0;
    for (input.batches) |batch| {
        const donor = input.checkpoint_owned.state.layer_donors[batch.consumer_layer];
        const head_count = input.checkpoint_owned.state.donors[donor].heads.len;
        const group = input.layout.query_heads_by_layer[batch.consumer_layer] / head_count;
        const identity = [2]usize{ donor, batch.query_head / group };
        var found = false;
        for (identities[0..identity_count]) |known| if (known[0] == identity[0] and known[1] == identity[1]) {
            found = true;
            break;
        };
        if (!found) {
            if (identity_count == identities.len) return error.InvalidOfflineReplayBinding;
            identities[identity_count] = identity;
            identity_count += 1;
        }
    }
    if (identity_count != 8) return error.InvalidOfflineReplayBinding;
    const canonical = [_][2]usize{ .{ 5, 0 }, .{ 5, 1 }, .{ 11, 0 }, .{ 11, 1 }, .{ 17, 0 }, .{ 17, 1 }, .{ 23, 0 }, .{ 23, 1 } };
    for (canonical) |expected| {
        var found = false;
        for (identities) |identity| {
            if (input.checkpoint_owned.state.donors[identity[0]].layer == expected[0] and identity[1] == expected[1]) {
                found = true;
                break;
            }
        }
        if (!found) return error.InvalidOfflineReplayBinding;
    }
    var head_consumer_sum: usize = 0;
    var head_scalar_ns: u64 = 0;
    var head_accelerated_ns: u64 = 0;
    var accelerated_calls: usize = 0;
    var stage1_digests: [8][32]u8 = undefined;
    for (identities, 0..) |identity, cell_at| {
        var consumers: usize = 0;
        for (input.batches) |batch| {
            const donor = input.checkpoint_owned.state.layer_donors[batch.consumer_layer];
            const group = input.layout.query_heads_by_layer[batch.consumer_layer] / input.checkpoint_owned.state.donors[donor].heads.len;
            if (donor == identity[0] and batch.query_head / group == identity[1]) consumers += 1;
        }
        const measured = try benchmark.run(std.testing.allocator, input.checkpoint_owned.state, input.layout, input.batches, &dense, complete.future.batches, .{ .donor_layer = @intCast(identity[0]), .head_index = identity[1], .ranges = &range_values, .combined_cap = 1024 * 1024 * 1024, .external_live_bytes = live, .gather_eligible_queries = true });
        if (measured.max_probability_weight_abs_diff != 0 or !measured.baseline_covered_equal or !measured.selected_indices_equal or measured.accelerated_calls == 0) return error.InvalidOfflineReplayBinding;
        if (!std.mem.eql(u8, &measured.scalar_selected_positions_digest, &measured.accelerated_selected_positions_digest)) return error.InvalidOfflineReplayBinding;
        stage1_digests[cell_at] = measured.scalar_selected_positions_digest;
        head_consumer_sum += consumers;
        head_scalar_ns += measured.scalar_elapsed_ns;
        head_accelerated_ns += measured.accelerated_elapsed_ns;
        accelerated_calls += measured.accelerated_calls;
    }
    if (head_consumer_sum != 56) return error.InvalidOfflineReplayBinding;
    const scalar_start = platform.time.monotonicNs();
    var scalar = try coverage.derive(std.testing.allocator, input.checkpoint_owned.state, input.layout, input.batches, complete.future.batches, &dense, dense_binding.maximum_workspace_bytes, .{});
    defer scalar.deinit();
    const scalar_end = platform.time.monotonicNs();
    const scalar_live = try ownedBytes(scalar);
    if (try std.math.add(usize, try std.math.add(usize, live, scalar_live), dense_binding.maximum_workspace_bytes) > 1024 * 1024 * 1024) return error.CompactionWorkspaceLimit;
    var accel_context: accelerate.Context = .{};
    const accelerated_start = platform.time.monotonicNs();
    var accelerated = try coverage.deriveWithProbabilityBackend(std.testing.allocator, input.checkpoint_owned.state, input.layout, input.batches, complete.future.batches, &dense, dense_binding.maximum_workspace_bytes, .{}, .{ .backend = accelerate.backend(&accel_context), .gather_eligible_historical = true });
    defer accelerated.deinit();
    const accelerated_end = platform.time.monotonicNs();
    if (!batchOutputsEqual(scalar, accelerated) or !std.mem.eql(u8, &scalar.evidence.derived_historical_sha256, &accelerated.evidence.derived_historical_sha256)) return error.InvalidOfflineReplayBinding;
    if (!std.mem.eql(u8, &scalar.evidence.selection_digest, &expected_scalar_selection) or !std.mem.eql(u8, &scalar.evidence.derived_historical_sha256, &expected_scalar_historical)) return error.InvalidOfflineReplayBinding;
    for (identities, stage1_digests) |identity, stage1| {
        const stage2 = try stage2HeadDigest(input.checkpoint_owned.state, input.layout, identity[0], identity[1], scalar.batches);
        if (!std.mem.eql(u8, &stage1, &stage2)) return error.InvalidOfflineReplayBinding;
    }
    if (scalar.evidence.batch_count != accelerated.evidence.batch_count or scalar.evidence.eligible_rows != accelerated.evidence.eligible_rows or scalar.evidence.excluded_rows != accelerated.evidence.excluded_rows or scalar.evidence.baseline_covered_key_count != accelerated.evidence.baseline_covered_key_count or scalar.evidence.newly_covered_key_count != accelerated.evidence.newly_covered_key_count or scalar.evidence.coverage_selected_rows != accelerated.evidence.coverage_selected_rows or scalar.evidence.kcenter_selected_rows != accelerated.evidence.kcenter_selected_rows) return error.InvalidOfflineReplayBinding;
    if (std.mem.eql(u8, &scalar.evidence.selection_digest, &accelerated.evidence.selection_digest) or accel_context.calls == 0) return error.InvalidOfflineReplayBinding;
    const after = try hashFile(b.producer_source_freeze.path);
    if (!std.mem.eql(u8, &before, &after) or !std.mem.eql(u8, &after, &b.producer_source_freeze.sha256)) return error.InvalidOfflineReplayBinding;
    const body = try std.json.Stringify.valueAlloc(std.testing.allocator, .{ .schema = "antfly.native-coverage-all8-parity.v2", .diagnostic_only = true, .qualification_credit = false, .formal_performance_credit = false, .head_count = identity_count, .consumer_count = head_consumer_sum, .head_scalar_elapsed_ns = head_scalar_ns, .head_accelerated_elapsed_ns = head_accelerated_ns, .head_accelerated_calls = accelerated_calls, .complete_scalar_elapsed_ns = scalar_end - scalar_start, .complete_accelerated_elapsed_ns = accelerated_end - accelerated_start, .complete_accelerated_calls = accel_context.calls, .all_head_weights_baseline_selection_equal = true, .all_56_batches_positions_values_equal = true, .derived_historical_sha256_equal = true, .scalar_selection_digest = scalar.evidence.selection_digest, .scalar_derived_historical_sha256 = scalar.evidence.derived_historical_sha256, .scalar_legacy_digest_preserved = true, .stage1_stage2_selected_positions_equal = true, .observed_donor_indices_and_heads = identities, .canonical_donor_layers_and_heads = canonical, .accelerated_policy_digest_distinct = true, .hho_query_values_read = false, .source_stable = true }, .{});
    defer std.testing.allocator.free(body);
    var output = try std.Io.Dir.cwd().createFile(std.testing.io, result_path, .{ .exclusive = true });
    defer output.close(std.testing.io);
    try output.writeStreamingAll(std.testing.io, body);
    try output.writeStreamingAll(std.testing.io, "\n");
    try output.sync(std.testing.io);
}

test "context retained native merge matches Source918 H CSR and exact head" {
    const binding_path = platform.env.getenv("ANTFLY_NATIVE_MERGE_SOURCE918_PARITY_BINDING") orelse return error.SkipZigTest;
    const result_path = platform.env.getenv("ANTFLY_NATIVE_MERGE_SOURCE918_PARITY_RESULT") orelse return error.InvalidOfflineReplayBinding;
    if (!std.fs.path.isAbsolute(binding_path) or !std.fs.path.isAbsolute(result_path)) return error.InvalidOfflineReplayBinding;
    const binding_bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, binding_path, std.testing.allocator, .limited(256 * 1024));
    defer std.testing.allocator.free(binding_bytes);
    var parsed = try std.json.parseFromSlice(Binding, std.testing.allocator, binding_bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const b = parsed.value;
    if (!b.diagnostic_only or b.qualification_credit or b.production_qualified) return error.InvalidOfflineReplayBinding;
    const dense_binding = b.historical_coverage orelse return error.InvalidOfflineReplayBinding;
    const legacy = b.legacy_scalar_evidence;
    inline for (.{ b.whole_manifest, b.complete_manifest, b.producer_source_freeze, b.producer_binary, b.producer_profile, b.integrity_closure, dense_binding.dense_outer_manifest, dense_binding.dense_nested_manifest, dense_binding.dense_reference_parent_manifest, legacy }) |item| try authenticate(item);
    var authority = b.authority;
    authority.generation_contract_sha256 = future.policyDigest();
    const authority_digest = authority.digest();
    if (!std.mem.eql(u8, &authority_digest, &b.complete_authority_sha256)) return error.InvalidOfflineReplayBinding;
    var input = try whole.read(std.testing.allocator, std.testing.io, .{ .output_dir = b.whole_fit_dir, .identity = b.identity, .config_sha256 = b.config_sha256, .source_freeze_sha256 = b.source_freeze_sha256, .binary_sha256 = b.binary_sha256, .exporter_sha256 = b.exporter_sha256, .limits = checkpoint.serving_limits });
    defer input.deinit();
    var complete = try future_augmentation.readComplete(std.testing.allocator, std.testing.io, .{ .output_dir = b.complete_future_dir, .manifest_sha256 = b.complete_manifest_sha256, .primary_parent_manifest_sha256 = b.parent_manifest_sha256, .identity = b.identity, .source_freeze_sha256 = b.source_freeze_sha256, .binary_sha256 = b.binary_sha256, .exporter_sha256 = b.exporter_sha256, .authority = authority, .historical_study_tokens_sha256 = b.historical_study_tokens_sha256, .primary_batches = input.batches });
    defer complete.deinit();
    try coverage.authenticateOuter(std.testing.allocator, std.testing.io, .{ .manifest_path = dense_binding.dense_outer_manifest.path, .manifest_sha256 = dense_binding.dense_outer_manifest.sha256, .nested_manifest_sha256 = dense_binding.dense_nested_manifest.sha256, .parent_manifest_sha256 = dense_binding.dense_reference_parent_manifest_sha256, .identity = b.identity, .source_freeze_sha256 = b.source_freeze_sha256, .binary_sha256 = b.binary_sha256, .exporter_sha256 = b.exporter_sha256 });
    var dense = try augmentation.readReferenceCoverageDense512(std.testing.allocator, std.testing.io, .{ .output_dir = dense_binding.dense_dir, .max_bytes = dense_binding.maximum_workspace_bytes - 64 * 1024, .parent_manifest_sha256 = b.parent_manifest_sha256, .identity = b.identity, .source_freeze_sha256 = b.source_freeze_sha256, .binary_sha256 = b.binary_sha256, .exporter_sha256 = b.exporter_sha256, .parent_study_tokens_sha256 = dense_binding.dense_parent_study_tokens_sha256, .study_tokens_sha256 = dense_binding.dense_study_tokens_sha256, .samples_per_stratum = 512 });
    defer dense.deinit();
    // No fitter is live in this comparison. Charge each actual retained
    // owner rather than the inactive query/fitter reservation from the export.
    var live = try input.liveBytes();
    live = try std.math.add(usize, live, try denseOwnerBytes(dense));
    live = try std.math.add(usize, live, try denseOwnerBytes(complete.future));
    live = try std.math.add(usize, live, try denseOwnerBytes(complete.historical));
    live = try std.math.add(usize, live, 4 * 1024 * 1024);
    var accel_context: accelerate.Context = .{};
    const source918_start = platform.time.monotonicNs();
    var source918 = try coverage.deriveWithProbabilityBackend(std.testing.allocator, input.checkpoint_owned.state, input.layout, input.batches, complete.future.batches, &dense, dense_binding.maximum_workspace_bytes, .{}, .{ .backend = accelerate.backend(&accel_context), .gather_eligible_historical = true });
    defer source918.deinit();
    const source918_end = platform.time.monotonicNs();
    std.debug.print("native_source918_parity phase=source918_complete\n", .{});
    var plans = std.ArrayList(native_merge.Plan).empty;
    defer plans.deinit(std.testing.allocator);
    for (input.checkpoint_owned.state.donors, 0..) |donor, donor_index| if (donor.window == 0) for (donor.heads, 0..) |head, head_index| {
        const plan = (try @import("runtime/kv/attention_match_engine.zig").headPlan(input.checkpoint_owned.state.logical_length, head, .{ .prefix_rows = 128, .recent_rows = 2048, .chunk_rows = 256, .retention_divisor = 2 })) orelse return error.InvalidOfflineReplayBinding;
        try plans.append(std.testing.allocator, .{ .donor = donor_index, .head_index = head_index, .plan = plan });
    };
    if (plans.items.len != 8) return error.InvalidOfflineReplayBinding;
    live = try std.math.add(usize, live, try std.math.mul(usize, plans.capacity, @sizeOf(native_merge.Plan)));
    const source918_live = try source918.liveBytes();
    const native_start = platform.time.monotonicNs();
    var native = try native_merge.merge(std.testing.allocator, input.checkpoint_owned.state, input.layout, input.batches, dense.batches, complete.future.batches, plans.items, 256, 256, try std.math.add(usize, live, source918_live), 1024 * 1024 * 1024, .{});
    defer native.deinit();
    const native_end = platform.time.monotonicNs();
    std.debug.print("native_source918_parity phase=native_merge_complete\n", .{});
    if (native.batches.len != 56 or source918.batches.len != 56 or native.heads.len != 8 or accel_context.calls == 0) return error.InvalidOfflineReplayBinding;
    var native_live = try std.math.mul(usize, native.batches.len, @sizeOf(@TypeOf(native.batches[0])));
    native_live = try std.math.add(usize, native_live, try std.math.mul(usize, native.values.len + native.heldout_values.len, @sizeOf(f32)));
    native_live = try std.math.add(usize, native_live, try std.math.mul(usize, native.positions.len + native.heldout_positions.len, @sizeOf(u64)));
    native_live = try std.math.add(usize, native_live, try std.math.mul(usize, native.coverage_offsets.len + native.coverage_indices.len, @sizeOf(u32)));
    native_live = try std.math.add(usize, native_live, try std.math.mul(usize, native.heads.len, @sizeOf(native_merge.HeadEvidence)));
    for (native.heads) |head| native_live = try std.math.add(usize, native_live, try std.math.mul(usize, head.maximum_probability.len, @sizeOf(f64)));
    const range_values = ranges();
    var direct_maxp_comparisons: usize = 0;
    for (native.heads) |head| {
        const measured = try benchmark.run(std.testing.allocator, input.checkpoint_owned.state, input.layout, input.batches, &dense, complete.future.batches, .{ .donor_layer = head.donor_layer, .head_index = head.head_index, .ranges = &range_values, .combined_cap = 1024 * 1024 * 1024, .external_live_bytes = try std.math.add(usize, try std.math.add(usize, live, source918_live), native_live), .gather_eligible_queries = true });
        var native_maxp_hash = std.crypto.hash.sha2.Sha256.init(.{});
        native_maxp_hash.update("antfly.native-coverage-maximum-probability.v1\x00");
        native_maxp_hash.update(std.mem.sliceAsBytes(head.maximum_probability));
        const native_maxp_digest = native_maxp_hash.finalResult();
        if (!std.mem.eql(u8, &native_maxp_digest, &measured.scalar_max_probability_sha256) or !std.mem.eql(u8, &native_maxp_digest, &measured.accelerated_max_probability_sha256) or measured.max_probability_weight_abs_diff != 0) {
            std.debug.print("native_source918_parity phase=maxp_mismatch donor={d} head={d} native_scalar_equal={} native_accel_equal={} scalar_accel_abs_diff={d}\n", .{ head.donor_layer, head.head_index, std.mem.eql(u8, &native_maxp_digest, &measured.scalar_max_probability_sha256), std.mem.eql(u8, &native_maxp_digest, &measured.accelerated_max_probability_sha256), measured.max_probability_weight_abs_diff });
            return error.NativeMaximumProbabilityMismatch;
        }
        direct_maxp_comparisons += 1;
        std.debug.print("native_source918_parity phase=maxp_equal donor={d} head={d}\n", .{ head.donor_layer, head.head_index });
    }
    if (direct_maxp_comparisons != 8) return error.InvalidOfflineReplayBinding;
    var native_h_digest = std.crypto.hash.sha2.Sha256.init(.{});
    native_h_digest.update("antfly.production768-derived-historical-domain.v1\x00");
    for (native.batches, source918.batches) |actual, expected| {
        if (actual.consumer_layer != expected.consumer_layer or actual.query_head != expected.query_head or actual.independent_train_split != 256 or actual.independent_train_second_split != 512 or actual.train_positions.len != 768 or expected.train_positions.len != 256) {
            std.debug.print("native_source918_parity phase=tuple_mismatch actual_layer={d} expected_layer={d} actual_qhead={d} expected_qhead={d} primary_split={?d} second_split={?d} native_rows={d} expected_hrows={d}\n", .{ actual.consumer_layer, expected.consumer_layer, actual.query_head, expected.query_head, actual.independent_train_split, actual.independent_train_second_split, actual.train_positions.len, expected.train_positions.len });
            return error.NativeTrainingTupleMismatch;
        }
        const dim = actual.train.len / actual.train_positions.len;
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(actual.train[256 * dim .. 512 * dim]), std.mem.sliceAsBytes(expected.train)) or !std.mem.eql(u8, std.mem.sliceAsBytes(actual.train_positions[256..512]), std.mem.sliceAsBytes(expected.train_positions))) {
            std.debug.print("native_source918_parity phase=historical_mismatch layer={d} query_head={d} values_equal={} positions_equal={}\n", .{ actual.consumer_layer, actual.query_head, std.mem.eql(u8, std.mem.sliceAsBytes(actual.train[256 * dim .. 512 * dim]), std.mem.sliceAsBytes(expected.train)), std.mem.eql(u8, std.mem.sliceAsBytes(actual.train_positions[256..512]), std.mem.sliceAsBytes(expected.train_positions)) });
            return error.NativeHistoricalSelectionMismatch;
        }
        native_h_digest.update(std.mem.sliceAsBytes(actual.train[256 * dim .. 512 * dim]));
    }
    for (native.batches) |actual| native_h_digest.update(std.mem.sliceAsBytes(actual.train_positions[256..512]));
    const native_derived_historical_sha256 = native_h_digest.finalResult();
    if (!std.mem.eql(u8, &native_derived_historical_sha256, &source918.evidence.derived_historical_sha256)) return error.NativeHistoricalDigestMismatch;
    if (!std.mem.eql(u32, native.coverage_offsets, source918.coverage_offsets) or !std.mem.eql(u32, native.coverage_indices, source918.coverage_indices) or !std.mem.eql(u8, &native.coverage_authority_sha256, &source918.evidence.coverage_authority_sha256)) return error.NativeCoverageAuthorityMismatch;
    std.debug.print("native_source918_parity phase=historical_and_csr_equal batches=56\n", .{});
    var native_best: f64 = -1;
    var native_exact_layer: u32 = 0;
    var native_exact_head: usize = 0;
    var maxima_hash = std.crypto.hash.sha2.Sha256.init(.{});
    maxima_hash.update("antfly.native-merge-all8-maximum-probability.v1\x00");
    for (native.heads) |head| {
        maxima_hash.update(std.mem.sliceAsBytes(head.maximum_probability));
        for (head.maximum_probability) |value| if (value > native_best) {
            native_best = value;
            native_exact_layer = head.donor_layer;
            native_exact_head = head.head_index;
        };
    }
    std.debug.print("native_source918_parity phase=exact_choice native_raw_layer={d} native_raw_head={d} source918_pressure_layer={d} source918_pressure_head={d}\n", .{ native_exact_layer, native_exact_head, source918.evidence.chosen_exact_donor_layer, source918.evidence.chosen_exact_head_index });
    if (native_exact_layer != source918.evidence.chosen_exact_donor_layer or native_exact_head != source918.evidence.chosen_exact_head_index) return error.NativeExactReservationPolicyMismatch;
    const body = try std.json.Stringify.valueAlloc(std.testing.allocator, .{ .schema = "antfly.retained-native-merge-source918-parity.v1", .diagnostic_only = true, .qualification_credit = false, .formal_performance_credit = false, .batch_count = native.batches.len, .head_count = native.heads.len, .h256_f32_bits_equal = true, .h256_positions_equal = true, .coverage_offsets_equal = true, .coverage_indices_equal = true, .coverage_authority_sha256 = native.coverage_authority_sha256, .source918_derived_historical_sha256 = source918.evidence.derived_historical_sha256, .native_maximum_probability_sha256 = maxima_hash.finalResult(), .root1074_maximum_probability_parity_authenticated = true, .exact_donor_layer = native_exact_layer, .exact_head_index = native_exact_head, .source918_elapsed_ns = source918_end - source918_start, .native_elapsed_ns = native_end - native_start, .timing_diagnostic_only = true, .activation_allowed = true }, .{});
    defer std.testing.allocator.free(body);
    var output = try std.Io.Dir.cwd().createFile(std.testing.io, result_path, .{ .exclusive = true });
    defer output.close(std.testing.io);
    try output.writeStreamingAll(std.testing.io, body);
    try output.writeStreamingAll(std.testing.io, "\n");
    try output.sync(std.testing.io);
}
