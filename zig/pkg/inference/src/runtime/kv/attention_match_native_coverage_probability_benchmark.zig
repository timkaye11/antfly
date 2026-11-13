const std = @import("std");
const platform = @import("antfly_platform");
const solver = @import("attention_match.zig");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const admission = @import("attention_match_cumulative_cache.zig");
const augmentation = @import("attention_match_training_augmentation.zig");
const adapter = @import("attention_match_native_coverage_data_adapter.zig");
const accelerate = @import("attention_match_accelerate_fp64_probability_scan.zig");

pub const policy = "native_coverage_probability_benchmark_v1|captured_train_only|scalar_vs_accelerate_fp64|no_hho_query_values|diagnostic_only";

pub const Config = struct {
    donor_layer: u32,
    head_index: usize,
    ranges: []const admission.Range,
    combined_cap: usize,
    external_live_bytes: usize,
    control: solver.Control = .{},
    gather_eligible_queries: bool = false,
};

pub const Result = struct {
    scalar_elapsed_ns: u64,
    accelerated_elapsed_ns: u64,
    max_probability_weight_abs_diff: f64,
    baseline_covered_equal: bool,
    selected_indices_equal: bool,
    scalar_authority_digest: [32]u8,
    accelerated_authority_digest: [32]u8,
    accelerated_calls: usize,
    accelerated_workspace_peak_bytes: usize,
    scalar_selected_positions_digest: [32]u8,
    accelerated_selected_positions_digest: [32]u8,
    scalar_max_probability_sha256: [32]u8,
    accelerated_max_probability_sha256: [32]u8,
};

fn maximumProbabilityDigest(values: []const f64) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.native-coverage-maximum-probability.v1\x00");
    hash.update(std.mem.sliceAsBytes(values));
    return hash.finalResult();
}

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}

fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}

fn ownedLiveBytes(value: adapter.Owned) !usize {
    var bytes = try mul(value.source_key_weights.len, @sizeOf(f64));
    bytes = try add(bytes, try mul(value.baseline_covered_keys.len, @sizeOf(u32)));
    bytes = try add(bytes, try mul(value.selection.consumers.len, @sizeOf(@TypeOf(value.selection.consumers[0]))));
    for (value.selection.consumers) |consumer| bytes = try add(bytes, try mul(consumer.selected_indices.len, @sizeOf(u32)));
    return bytes;
}

fn mappedHead(state: checkpoint.State, layout: engine.Layout, batch: engine.ReferenceBatch) !struct { donor: usize, head: usize } {
    if (batch.consumer_layer >= state.layer_donors.len or batch.consumer_layer >= layout.query_heads_by_layer.len) return error.NativeCoverageShapeUnsupported;
    const donor = state.layer_donors[batch.consumer_layer];
    if (donor >= state.donors.len or donor >= layout.scales_by_donor.len or donor >= layout.softcaps_by_donor.len) return error.NativeCoverageShapeUnsupported;
    const heads = state.donors[donor].heads.len;
    if (heads == 0 or layout.query_heads_by_layer[batch.consumer_layer] % heads != 0) return error.NativeCoverageShapeUnsupported;
    const group = layout.query_heads_by_layer[batch.consumer_layer] / heads;
    if (group == 0) return error.NativeCoverageShapeUnsupported;
    const head = batch.query_head / group;
    if (head >= heads) return error.NativeCoverageShapeUnsupported;
    return .{ .donor = donor, .head = head };
}

fn findBatch(batches: []const engine.ReferenceBatch, layer: u32, qhead: usize) !engine.ReferenceBatch {
    var found: ?engine.ReferenceBatch = null;
    for (batches) |batch| if (batch.consumer_layer == layer and batch.query_head == qhead) {
        if (found != null) return error.NativeCoverageShapeUnsupported;
        found = batch;
    };
    return found orelse error.NativeCoverageShapeUnsupported;
}

fn selectedPositionsDigestMapped(state: checkpoint.State, layout: engine.Layout, donor: usize, head_index: usize, primary: []const engine.ReferenceBatch, historical: []const engine.ReferenceBatch, selected: []const @import("attention_match_native_historical_coverage_bridge.zig").ConsumerEvidence) ![32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.native-coverage-selected-historical-positions.v1\x00");
    hash.update(std.mem.asBytes(&donor));
    hash.update(std.mem.asBytes(&head_index));
    var at: usize = 0;
    for (primary) |p| {
        const mapped = try mappedHead(state, layout, p);
        if (mapped.donor != donor or mapped.head != head_index) continue;
        const h = try findBatch(historical, p.consumer_layer, p.query_head);
        if (at >= selected.len) return error.NativeCoverageShapeUnsupported;
        hash.update(std.mem.asBytes(&p.consumer_layer));
        hash.update(std.mem.asBytes(&p.query_head));
        for (selected[at].selected_indices) |index| {
            if (index >= h.train_positions.len) return error.NativeCoverageShapeUnsupported;
            hash.update(std.mem.asBytes(&h.train_positions[index]));
        }
        at += 1;
    }
    if (at != selected.len) return error.NativeCoverageShapeUnsupported;
    return hash.finalResult();
}

pub fn run(
    allocator: std.mem.Allocator,
    state: checkpoint.State,
    layout: engine.Layout,
    primary: []const engine.ReferenceBatch,
    historical: *const augmentation.Owned,
    future: []const engine.ReferenceBatch,
    cfg: Config,
) !Result {
    if (primary.len == 0 or historical.batches.len != primary.len or future.len != primary.len) return error.NativeCoverageShapeUnsupported;
    var consumers = std.ArrayList(adapter.Consumer).empty;
    defer consumers.deinit(allocator);
    var source: ?checkpoint.Head = null;
    var scale: f64 = 0;
    var softcap: f64 = 0;
    for (primary) |p| {
        const mapped = try mappedHead(state, layout, p);
        if (mapped.donor != cfg.donor_layer or mapped.head != cfg.head_index) continue;
        const h = try findBatch(historical.batches, p.consumer_layer, p.query_head);
        const f = try findBatch(future, p.consumer_layer, p.query_head);
        if (p.train_positions.len == 0 or p.train.len % p.train_positions.len != 0) return error.NativeCoverageShapeUnsupported;
        try consumers.append(allocator, .{
            .primary_queries = p.train,
            .future_queries = f.train,
            .historical_queries = h.train,
            .historical_positions = h.train_positions,
            .heldout_positions = p.heldout_positions,
        });
        source = state.donors[mapped.donor].heads[mapped.head];
        scale = layout.scales_by_donor[mapped.donor];
        softcap = layout.softcaps_by_donor[mapped.donor];
    }
    const head = source orelse return error.NativeCoverageShapeUnsupported;
    const first = consumers.items[0];
    const dim = head.dim;
    const shape = @import("attention_match_native_historical_coverage_bridge.zig").Shape{
        .dim = dim,
        .source_rows = head.positions.len,
        .primary_rows = first.primary_queries.len / dim,
        .historical_rows = first.historical_queries.len / dim,
        .future_rows = first.future_queries.len / dim,
        .heldout_rows = first.heldout_positions.len,
        .selected_historical_rows = 256,
        .consumer_count = consumers.items.len,
        .expected_head_count = state.donors[cfg.donor_layer].heads.len,
        .actual_head_count = state.donors[cfg.donor_layer].heads.len,
        .route = .flat,
    };
    const common: adapter.Config = .{ .shape = shape, .ranges = cfg.ranges, .donor_layer = cfg.donor_layer, .head_index = cfg.head_index, .expected_donor_layer = cfg.donor_layer, .expected_head_index = cfg.head_index, .scale = scale, .softcap = softcap, .external_live_bytes = cfg.external_live_bytes, .combined_cap = cfg.combined_cap, .control = cfg.control };
    const scalar_start = platform.time.monotonicNs();
    var scalar = try adapter.derive(allocator, head, consumers.items, common);
    defer scalar.deinit();
    const scalar_end = platform.time.monotonicNs();
    var context: accelerate.Context = .{};
    var accelerated_cfg = common;
    accelerated_cfg.external_live_bytes = try add(accelerated_cfg.external_live_bytes, try add(try ownedLiveBytes(scalar), try mul(consumers.capacity, @sizeOf(adapter.Consumer))));
    if (accelerated_cfg.external_live_bytes > accelerated_cfg.combined_cap) return error.CompactionWorkspaceLimit;
    accelerated_cfg.probability_backend = accelerate.backend(&context);
    accelerated_cfg.gather_eligible_queries = cfg.gather_eligible_queries;
    const accelerated_start = platform.time.monotonicNs();
    var accelerated = try adapter.derive(allocator, head, consumers.items, accelerated_cfg);
    defer accelerated.deinit();
    const accelerated_end = platform.time.monotonicNs();
    var max_diff: f64 = 0;
    for (scalar.source_key_weights, accelerated.source_key_weights) |a, b| max_diff = @max(max_diff, @abs(a - b));
    var selected_equal = scalar.selection.consumers.len == accelerated.selection.consumers.len;
    if (selected_equal) {
        for (scalar.selection.consumers, accelerated.selection.consumers) |a, b| {
            selected_equal = selected_equal and std.mem.eql(u32, a.selected_indices, b.selected_indices);
        }
    }
    return .{
        .scalar_elapsed_ns = scalar_end - scalar_start,
        .accelerated_elapsed_ns = accelerated_end - accelerated_start,
        .max_probability_weight_abs_diff = max_diff,
        .baseline_covered_equal = std.mem.eql(u32, scalar.baseline_covered_keys, accelerated.baseline_covered_keys),
        .selected_indices_equal = selected_equal,
        .scalar_authority_digest = scalar.authority_digest,
        .accelerated_authority_digest = accelerated.authority_digest,
        .accelerated_calls = context.calls,
        .accelerated_workspace_peak_bytes = context.workspace_peak_bytes,
        .scalar_selected_positions_digest = try selectedPositionsDigestMapped(state, layout, cfg.donor_layer, cfg.head_index, primary, historical.batches, scalar.selection.consumers),
        .accelerated_selected_positions_digest = try selectedPositionsDigestMapped(state, layout, cfg.donor_layer, cfg.head_index, primary, historical.batches, accelerated.selection.consumers),
        .scalar_max_probability_sha256 = maximumProbabilityDigest(scalar.source_key_weights),
        .accelerated_max_probability_sha256 = maximumProbabilityDigest(accelerated.source_key_weights),
    };
}
