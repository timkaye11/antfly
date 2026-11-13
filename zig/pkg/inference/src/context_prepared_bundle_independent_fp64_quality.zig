// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Independent CPU-FP64 validation using only a current prepared-quality
//! source/candidate and its transaction-bound heldout bundle.
const std = @import("std");
const platform = @import("antfly_platform");
const checkpoint = @import("runtime/kv/attention_match_checkpoint.zig");
const bundle_mod = @import("runtime/kv/attention_match_prepared_heldout_bundle.zig");
const solver = @import("runtime/kv/attention_match.zig");
const storage = @import("ops/metal_attention_match_storage.zig");

const Ref = struct { path: []const u8, bytes: usize, sha256: [32]u8 };
const Binding = struct {
    schema: []const u8,
    diagnostic_only: bool,
    quality_credit: bool,
    identity: checkpoint.Identity,
    source: Ref,
    candidate: Ref,
    closure: Ref,
    heldout_bundle: Ref,
    expected_source_global_bytes: usize,
    expected_candidate_global_bytes: usize,
    minimum_physical_ratio: f64,
    maximum_relative_mass_error: f64,
    maximum_relative_output_rmse: f64,
};
const Closure = struct { record: struct { checkpoint_sha256: []const u8, source_checkpoint_sha256: []const u8, encoding: []const u8, logical_tokens: usize, future_training_authority: struct { original_heldout_sha256: []const u8 } } };
const HeadResult = struct { donor_layer: u32, head_index: usize, heldout_queries: usize, source_rows: usize, candidate_rows: usize, mass_error: f64, output_rmse: f64 };
const Result = struct {
    schema: []const u8 = "antfly.prepared-bundle-independent-fp64-quality.result.v1",
    diagnostic_only: bool = true,
    quality_credit: bool = false,
    source_sha256: [32]u8,
    candidate_sha256: [32]u8,
    closure_sha256: [32]u8,
    heldout_bundle_sha256: [32]u8,
    heldout_authority_sha256: [32]u8,
    source_positions_sha256: [32]u8,
    candidate_positions_sha256: [32]u8,
    candidate_codec_geometry_sha256: [32]u8,
    heads: []const HeadResult,
    source_global_bytes: usize,
    candidate_global_bytes: usize,
    physical_ratio: f64,
    f16_rows: usize,
    fp8_rows: usize,
    rotated4_rows: usize,
    f16_heads: usize,
    fp8_heads: usize,
    rotated4_heads: usize,
    maximum_mass_error: f64,
    maximum_output_rmse: f64,
};
fn hash(bytes: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return out;
}
fn positionDigest(state: checkpoint.State) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("antfly.stored-candidate-position-coverage.v1\x00");
    var scalar: [8]u8 = undefined;
    for (state.donors, 0..) |donor, donor_index| for (donor.heads, 0..) |head, head_index| {
        inline for (.{ donor_index, head_index, head.positions.len }) |value| {
            std.mem.writeInt(u64, &scalar, @intCast(value), .little);
            h.update(&scalar);
        }
        h.update(std.mem.sliceAsBytes(head.positions));
        h.update(std.mem.sliceAsBytes(head.coverage_ends));
    };
    return h.finalResult();
}
fn codecGeometryDigest(state: checkpoint.State) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("antfly.stored-candidate-codec-geometry.v1\x00");
    var scalar: [8]u8 = undefined;
    for (state.donors, 0..) |donor, donor_index| for (donor.heads, 0..) |head, head_index| {
        inline for (.{ donor_index, head_index, head.positions.len, head.fp8.rows, head.compressed.rows() }) |value| {
            std.mem.writeInt(u64, &scalar, @intCast(value), .little);
            h.update(&scalar);
        }
    };
    return h.finalResult();
}
fn readRef(a: std.mem.Allocator, io: std.Io, ref: Ref, maximum: usize) ![]u8 {
    if (!std.fs.path.isAbsolute(ref.path) or ref.bytes == 0 or ref.bytes > maximum) return error.InvalidPreparedBundleQualityBinding;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, ref.path, a, .limited(maximum));
    errdefer a.free(bytes);
    if (bytes.len != ref.bytes or !std.mem.eql(u8, &hash(bytes), &ref.sha256)) return error.PreparedBundleQualityInputChanged;
    return bytes;
}
fn converted(a: std.mem.Allocator, values: []const f32) ![]f64 {
    const out = try a.alloc(f64, values.len);
    for (out, values) |*dst, value| dst.* = value;
    return out;
}
fn queryHeadsForLayer(batches: []const @import("runtime/kv/attention_match_engine.zig").ReferenceBatch, layer: u32) !usize {
    var seen: [256]bool = @splat(false);
    var maximum: usize = 0;
    for (batches) |batch| if (batch.consumer_layer == layer) {
        if (batch.query_head >= seen.len) return error.InvalidPreparedHeldoutBundle;
        seen[batch.query_head] = true;
        maximum = @max(maximum, @as(usize, batch.query_head) + 1);
    };
    if (maximum == 0) return error.InvalidPreparedHeldoutBundle;
    for (seen[0..maximum]) |value| if (!value) return error.InvalidPreparedHeldoutBundle;
    return maximum;
}
fn validateConsumerTuples(batches: []const @import("runtime/kv/attention_match_engine.zig").ReferenceBatch) !void {
    if (batches.len != 56) return error.InvalidPreparedHeldoutBundle;
    var layers: [7]u32 = undefined;
    var layer_count: usize = 0;
    for (batches, 0..) |batch, index| {
        if (batch.heldout_positions.len != 64 or batch.heldout.len != 64 * 512) return error.InvalidPreparedHeldoutBundle;
        for (batches[0..index]) |before| if (before.consumer_layer == batch.consumer_layer and before.query_head == batch.query_head) return error.InvalidPreparedHeldoutBundle;
        var found = false;
        for (layers[0..layer_count]) |layer| found = found or layer == batch.consumer_layer;
        if (!found) {
            if (layer_count == layers.len) return error.InvalidPreparedHeldoutBundle;
            layers[layer_count] = batch.consumer_layer;
            layer_count += 1;
        }
    }
    if (layer_count != layers.len) return error.InvalidPreparedHeldoutBundle;
    for (layers) |layer| if (try queryHeadsForLayer(batches, layer) != 8) return error.InvalidPreparedHeldoutBundle;
}
fn belongs(batch: @import("runtime/kv/attention_match_engine.zig").ReferenceBatch, state: checkpoint.State, batches: []const @import("runtime/kv/attention_match_engine.zig").ReferenceBatch, donor: usize, head: usize) !bool {
    if (state.layer_donors[batch.consumer_layer] != donor) return false;
    const query_heads = try queryHeadsForLayer(batches, batch.consumer_layer);
    if (query_heads % state.donors[donor].heads.len != 0) return error.InvalidPreparedHeldoutBundle;
    return batch.query_head / (query_heads / state.donors[donor].heads.len) == head;
}
fn validateHead(a: std.mem.Allocator, heldout: *const bundle_mod.Owned, source_state: checkpoint.State, candidate_state: checkpoint.State, donor: usize, head_index: usize) !HeadResult {
    const source_head = source_state.donors[donor].heads[head_index];
    const candidate_head = candidate_state.donors[donor].heads[head_index];
    var elements: usize = 0;
    var rows: usize = 0;
    for (heldout.batches) |batch| if (try belongs(batch, source_state, heldout.batches, donor, head_index)) {
        elements += batch.heldout.len;
        rows += batch.heldout_positions.len;
    };
    if (rows == 0 or elements != rows * source_head.dim) return error.InvalidPreparedHeldoutBundle;
    const queries = try a.alloc(f64, elements);
    defer a.free(queries);
    var at: usize = 0;
    for (heldout.batches) |batch| if (try belongs(batch, source_state, heldout.batches, donor, head_index)) for (batch.heldout) |value| {
        queries[at] = value;
        at += 1;
    };
    const sk = try converted(a, source_head.keys);
    defer a.free(sk);
    const sv = try converted(a, source_head.values);
    defer a.free(sv);
    const sb = try converted(a, source_head.bias);
    defer a.free(sb);
    const ck = try converted(a, candidate_head.keys);
    defer a.free(ck);
    const cv = try converted(a, candidate_head.values);
    defer a.free(cv);
    const cb = try converted(a, candidate_head.bias);
    defer a.free(cb);
    const original: solver.Head = .{ .keys = sk, .values = sv, .bias = sb, .rows = source_head.positions.len, .dim = source_head.dim, .scale = heldout.scales[donor], .softcap = heldout.softcaps[donor] };
    const compact: solver.Head = .{ .keys = ck, .values = cv, .bias = cb, .rows = candidate_head.positions.len, .dim = candidate_head.dim, .scale = heldout.scales[donor], .softcap = heldout.softcaps[donor] };
    const metrics = try solver.validationMetrics(a, original, compact, queries, .{ .target_rows = compact.rows, .max_relative_mass_error = std.math.floatMax(f64), .max_relative_output_rmse = std.math.floatMax(f64) });
    if (!metrics.quality_evaluated) return error.PreparedBundleIndependentQualityRejected;
    return .{ .donor_layer = source_state.donors[donor].layer, .head_index = head_index, .heldout_queries = rows, .source_rows = original.rows, .candidate_rows = compact.rows, .mass_error = metrics.max_relative_mass_error, .output_rmse = metrics.relative_output_rmse };
}
fn run(a: std.mem.Allocator, io: std.Io, binding: Binding) !Result {
    if (!std.mem.eql(u8, binding.schema, "antfly.prepared-bundle-independent-fp64-quality.binding.v1") or !binding.diagnostic_only or binding.quality_credit or binding.minimum_physical_ratio < 1.5 or binding.maximum_relative_mass_error != 0.05 or binding.maximum_relative_output_rmse != 0.05) return error.InvalidPreparedBundleQualityBinding;
    const closure_bytes = try readRef(a, io, binding.closure, 4 * 1024 * 1024);
    defer a.free(closure_bytes);
    var closure = try std.json.parseFromSlice(Closure, a, closure_bytes, .{ .ignore_unknown_fields = true });
    defer closure.deinit();
    const authority = try @import("context_supplied_checkpoint_quality.zig").parseDigest(closure.value.record.future_training_authority.original_heldout_sha256);
    const source_hex = std.fmt.bytesToHex(binding.source.sha256, .lower);
    const candidate_hex = std.fmt.bytesToHex(binding.candidate.sha256, .lower);
    if (!std.mem.eql(u8, closure.value.record.source_checkpoint_sha256, &source_hex) or !std.mem.eql(u8, closure.value.record.checkpoint_sha256, &candidate_hex)) return error.InvalidPreparedBundleQualityBinding;
    const heldout_bytes = try readRef(a, io, binding.heldout_bundle, bundle_mod.maximum_bytes);
    defer a.free(heldout_bytes);
    var heldout = try bundle_mod.decodeOwned(a, heldout_bytes, authority);
    defer heldout.deinit();
    try validateConsumerTuples(heldout.batches);
    const source_wire = try readRef(a, io, binding.source, checkpoint.serving_limits.max_file_bytes);
    defer a.free(source_wire);
    const candidate_wire = try readRef(a, io, binding.candidate, checkpoint.serving_limits.max_file_bytes);
    defer a.free(candidate_wire);
    if (try checkpoint.encodedWireEncoding(source_wire) != .f16 or !std.mem.eql(u8, closure.value.record.encoding, @tagName(try checkpoint.encodedWireEncoding(candidate_wire)))) return error.InvalidPreparedBundleQualityBinding;
    var source = try checkpoint.decode(a, source_wire, binding.identity, checkpoint.serving_limits);
    defer source.deinit();
    var candidate = try checkpoint.decode(a, candidate_wire, binding.identity, checkpoint.serving_limits);
    defer candidate.deinit();
    if (source.state.logical_length != closure.value.record.logical_tokens or source.state.logical_length != candidate.state.logical_length or source.state.donors.len != candidate.state.donors.len or heldout.scales.len != source.state.donors.len or
        !std.mem.eql(u32, source.state.tokens, candidate.state.tokens) or !std.mem.eql(u32, source.state.layer_donors, candidate.state.layer_donors)) return error.PreparedBundleSourceMismatch;
    for (source.state.donors, candidate.state.donors) |source_donor, candidate_donor| {
        if (source_donor.layer != candidate_donor.layer or source_donor.window != candidate_donor.window or source_donor.heads.len != candidate_donor.heads.len) return error.PreparedBundleSourceMismatch;
        for (source_donor.heads, candidate_donor.heads) |source_head, candidate_head| if (source_head.dim != candidate_head.dim) return error.PreparedBundleSourceMismatch;
    }
    for (heldout.batches) |batch| if (batch.consumer_layer >= source.state.layer_donors.len) return error.PreparedBundleSourceMismatch;
    var count: usize = 0;
    for (source.state.donors) |donor| if (donor.window == 0) {
        count += donor.heads.len;
    };
    if (count != 8) return error.PreparedBundleSourceMismatch;
    const heads = try a.alloc(HeadResult, count);
    errdefer a.free(heads);
    var index: usize = 0;
    var max_mass: f64 = 0;
    var max_output: f64 = 0;
    var f16_rows: usize = 0;
    var fp8_rows: usize = 0;
    var rotated_rows: usize = 0;
    var f16_heads: usize = 0;
    var fp8_heads: usize = 0;
    var rotated_heads: usize = 0;
    for (source.state.donors, 0..) |donor, donor_index| if (donor.window == 0) for (donor.heads, 0..) |_, head_index| {
        heads[index] = try validateHead(a, &heldout, source.state, candidate.state, donor_index, head_index);
        max_mass = @max(max_mass, heads[index].mass_error);
        max_output = @max(max_output, heads[index].output_rmse);
        const h = candidate.state.donors[donor_index].heads[head_index];
        f16_rows += h.positions.len - h.fp8.rows - h.compressed.rows();
        fp8_rows += h.fp8.rows;
        rotated_rows += h.compressed.rows();
        if (h.compressed.rows() != 0) rotated_heads += 1 else if (h.fp8.rows != 0) fp8_heads += 1 else f16_heads += 1;
        index += 1;
    };
    const source_bytes = try storage.ordinaryGlobalAllocationBytesWithControl(source.state, checkpoint.serving_limits, .{});
    const candidate_bytes = try storage.globalAllocationBytesWithControl(candidate.state, checkpoint.serving_limits, .{});
    const ratio = @as(f64, @floatFromInt(source_bytes)) / @as(f64, @floatFromInt(candidate_bytes));
    for (heads) |head| std.debug.print(
        "prepared_bundle_fp64_head donor={d} head={d} heldout={d} source_rows={d} candidate_rows={d} mass_error={d:.17} output_rmse={d:.17}\n",
        .{ head.donor_layer, head.head_index, head.heldout_queries, head.source_rows, head.candidate_rows, head.mass_error, head.output_rmse },
    );
    std.debug.print(
        "prepared_bundle_fp64_gate source_bytes={d} expected_source_bytes={d} candidate_bytes={d} expected_candidate_bytes={d} ratio={d:.17} minimum_ratio={d:.17} max_mass={d:.17} maximum_mass={d:.17} max_output={d:.17} maximum_output={d:.17} f16_rows={d} fp8_rows={d} rotated4_rows={d} f16_heads={d} fp8_heads={d} rotated4_heads={d}\n",
        .{ source_bytes, binding.expected_source_global_bytes, candidate_bytes, binding.expected_candidate_global_bytes, ratio, binding.minimum_physical_ratio, max_mass, binding.maximum_relative_mass_error, max_output, binding.maximum_relative_output_rmse, f16_rows, fp8_rows, rotated_rows, f16_heads, fp8_heads, rotated_heads },
    );
    if (source_bytes != binding.expected_source_global_bytes or candidate_bytes != binding.expected_candidate_global_bytes or ratio < binding.minimum_physical_ratio or max_mass > binding.maximum_relative_mass_error or max_output > binding.maximum_relative_output_rmse) return error.PreparedBundleIndependentQualityRejected;
    return .{ .source_sha256 = hash(source_wire), .candidate_sha256 = hash(candidate_wire), .closure_sha256 = hash(closure_bytes), .heldout_bundle_sha256 = hash(heldout_bytes), .heldout_authority_sha256 = authority, .source_positions_sha256 = positionDigest(source.state), .candidate_positions_sha256 = positionDigest(candidate.state), .candidate_codec_geometry_sha256 = codecGeometryDigest(candidate.state), .heads = heads, .source_global_bytes = source_bytes, .candidate_global_bytes = candidate_bytes, .physical_ratio = ratio, .f16_rows = f16_rows, .fp8_rows = fp8_rows, .rotated4_rows = rotated_rows, .f16_heads = f16_heads, .fp8_heads = fp8_heads, .rotated4_heads = rotated_heads, .maximum_mass_error = max_mass, .maximum_output_rmse = max_output };
}
test "context prepared bundle independently recomputes CPU FP64 heldout quality" {
    const binding_path = platform.env.getenv("ANTFLY_PREPARED_BUNDLE_FP64_QUALITY_BINDING") orelse return error.SkipZigTest;
    const result_path = platform.env.getenv("ANTFLY_PREPARED_BUNDLE_FP64_QUALITY_RESULT") orelse return error.SkipZigTest;
    if (!std.fs.path.isAbsolute(binding_path) or !std.fs.path.isAbsolute(result_path)) return error.InvalidPreparedBundleQualityBinding;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, binding_path, std.testing.allocator, .limited(256 * 1024));
    defer std.testing.allocator.free(bytes);
    var parsed = try std.json.parseFromSlice(Binding, std.testing.allocator, bytes, .{ .ignore_unknown_fields = false });
    defer parsed.deinit();
    const result = try run(std.testing.allocator, std.testing.io, parsed.value);
    defer std.testing.allocator.free(result.heads);
    const encoded = try std.json.Stringify.valueAlloc(std.testing.allocator, result, .{});
    defer std.testing.allocator.free(encoded);
    var file = try std.Io.Dir.cwd().createFile(std.testing.io, result_path, .{ .exclusive = true });
    defer file.close(std.testing.io);
    try file.writeStreamingAll(std.testing.io, encoded);
}
