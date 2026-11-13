const std = @import("std");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const adapter = @import("attention_match_native_coverage_data_adapter.zig");
const bridge = @import("attention_match_native_historical_coverage_bridge.zig");

pub const policy = "native_coverage_training_merge_v1|roles=P,H512,F,HHOpositions|output=P,Hselected,F+unchangedHHO|head_ranges=current_plan|activation=none";

pub const Plan = struct { donor: usize, head_index: usize, plan: engine.HeadPlan };

pub const HeadEvidence = struct {
    donor_layer: u32,
    head_index: usize,
    maximum_probability: []f64,
    authority_digest: [32]u8,
};

pub const Owned = struct {
    allocator: std.mem.Allocator,
    batches: []engine.ReferenceBatch,
    values: []f32,
    positions: []u64,
    heldout_values: []f32,
    heldout_positions: []u64,
    coverage_offsets: []u32,
    coverage_indices: []u32,
    coverage_authority_sha256: [32]u8,
    heads: []HeadEvidence,
    digest: [32]u8,

    pub fn deinit(self: *@This()) void {
        for (self.heads) |head| self.allocator.free(head.maximum_probability);
        self.allocator.free(self.heads);
        self.allocator.free(self.heldout_positions);
        self.allocator.free(self.heldout_values);
        self.allocator.free(self.coverage_indices);
        self.allocator.free(self.coverage_offsets);
        self.allocator.free(self.positions);
        self.allocator.free(self.values);
        self.allocator.free(self.batches);
        self.* = undefined;
    }
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}

fn mappedHead(state: checkpoint.State, layout: engine.Layout, batch: engine.ReferenceBatch) ![2]usize {
    if (batch.consumer_layer >= state.layer_donors.len or batch.consumer_layer >= layout.query_heads_by_layer.len) return error.NativeCoverageShapeUnsupported;
    const donor = state.layer_donors[batch.consumer_layer];
    if (donor >= state.donors.len or donor >= layout.scales_by_donor.len or donor >= layout.softcaps_by_donor.len or state.donors[donor].heads.len == 0) return error.NativeCoverageShapeUnsupported;
    const query_heads = layout.query_heads_by_layer[batch.consumer_layer];
    if (query_heads % state.donors[donor].heads.len != 0) return error.NativeCoverageShapeUnsupported;
    const group = query_heads / state.donors[donor].heads.len;
    if (group == 0) return error.NativeCoverageShapeUnsupported;
    const head_index = batch.query_head / group;
    if (head_index >= state.donors[donor].heads.len) return error.NativeCoverageShapeUnsupported;
    return .{ donor, head_index };
}

fn planFor(plans: []const Plan, donor: usize, head_index: usize) !engine.HeadPlan {
    var found: ?engine.HeadPlan = null;
    for (plans) |entry| if (entry.donor == donor and entry.head_index == head_index) {
        if (found != null) return error.NativeCoverageShapeUnsupported;
        found = entry.plan;
    };
    return found orelse error.NativeCoverageShapeUnsupported;
}

fn validatePlans(plans: []const Plan) !void {
    if (plans.len == 0) return error.NativeCoverageShapeUnsupported;
    for (plans, 0..) |entry, index| for (plans[index + 1 ..]) |other|
        if (entry.donor == other.donor and entry.head_index == other.head_index) return error.NativeCoverageShapeUnsupported;
}

pub fn merge(
    allocator: std.mem.Allocator,
    state: checkpoint.State,
    layout: engine.Layout,
    primary: []const engine.ReferenceBatch,
    historical_dense: []const engine.ReferenceBatch,
    future: []const engine.ReferenceBatch,
    plans: []const Plan,
    selected_historical_rows: usize,
    chunk_rows: usize,
    caller_external_live_bytes: usize,
    combined_cap: usize,
    control: @import("attention_match.zig").Control,
) !Owned {
    if (primary.len == 0 or primary.len != historical_dense.len or primary.len != future.len or plans.len == 0 or chunk_rows == 0 or caller_external_live_bytes > combined_cap)
        return error.NativeCoverageShapeUnsupported;
    try validatePlans(plans);
    var total_values: usize = 0;
    var total_positions: usize = 0;
    var total_heldout_values: usize = 0;
    var total_heldout_positions: usize = 0;
    var total_probability_rows: usize = 0;
    var maximum_head_scratch: usize = 0;
    for (primary, historical_dense, future) |p, h, f| {
        if (p.consumer_layer != h.consumer_layer or p.consumer_layer != f.consumer_layer or p.query_head != h.query_head or p.query_head != f.query_head or
            p.train_positions.len == 0 or h.train_positions.len == 0 or f.train_positions.len == 0 or p.heldout_positions.len == 0)
            return error.NativeCoverageShapeUnsupported;
        const dim = p.train.len / p.train_positions.len;
        if (dim == 0 or p.train.len != p.train_positions.len * dim or h.train.len != h.train_positions.len * dim or f.train.len != f.train_positions.len * dim or p.heldout.len != p.heldout_positions.len * dim)
            return error.NativeCoverageShapeUnsupported;
        const mapped = try mappedHead(state, layout, p);
        _ = try planFor(plans, mapped[0], mapped[1]);
        total_values = try add(total_values, try mul(try add(try add(p.train_positions.len, selected_historical_rows), f.train_positions.len), dim));
        total_positions = try add(total_positions, try add(try add(p.train_positions.len, selected_historical_rows), f.train_positions.len));
        total_heldout_values = try add(total_heldout_values, p.heldout.len);
        total_heldout_positions = try add(total_heldout_positions, p.heldout_positions.len);
    }
    for (plans) |binding| {
        if (binding.donor >= state.donors.len or binding.head_index >= state.donors[binding.donor].heads.len) return error.NativeCoverageShapeUnsupported;
        const rows = state.donors[binding.donor].heads[binding.head_index].positions.len;
        total_probability_rows = try add(total_probability_rows, rows);
        var consumers_for_head: usize = 0;
        for (primary) |batch| {
            const mapped = try mappedHead(state, layout, batch);
            consumers_for_head += @intFromBool(mapped[0] == binding.donor and mapped[1] == binding.head_index);
        }
        var scratch = try mul(rows, @sizeOf(engine.FitRange) + @sizeOf(@import("attention_match_cumulative_cache.zig").Range));
        scratch = try add(scratch, try mul(consumers_for_head, @sizeOf(adapter.Consumer)));
        maximum_head_scratch = @max(maximum_head_scratch, scratch);
    }
    var output_live = try mul(primary.len, @sizeOf(engine.ReferenceBatch));
    output_live = try add(output_live, try mul(total_values, @sizeOf(f32)));
    output_live = try add(output_live, try mul(total_positions, @sizeOf(u64)));
    output_live = try add(output_live, try mul(total_heldout_values, @sizeOf(f32)));
    output_live = try add(output_live, try mul(total_heldout_positions, @sizeOf(u64)));
    output_live = try add(output_live, try mul(plans.len, @sizeOf(HeadEvidence)));
    output_live = try add(output_live, try mul(total_probability_rows, @sizeOf(f64)));
    const maximum_coverage_indices = try mul(primary.len, selected_historical_rows);
    var coverage_storage_bound = try mul(primary.len, @sizeOf(?[]u32));
    coverage_storage_bound = try add(coverage_storage_bound, try mul(try add(primary.len, 1), @sizeOf(u32)));
    // Per-batch copies, precise flattening capacity, and a possible
    // allocate-copy shrink may overlap before the copies are retired.
    coverage_storage_bound = try add(coverage_storage_bound, try mul(try mul(maximum_coverage_indices, 3), @sizeOf(u32)));
    output_live = try add(output_live, coverage_storage_bound);
    if (output_live > combined_cap - caller_external_live_bytes) return error.CompactionWorkspaceLimit;

    const batches = try allocator.alloc(engine.ReferenceBatch, primary.len);
    errdefer allocator.free(batches);
    const values = try allocator.alloc(f32, total_values);
    errdefer allocator.free(values);
    const positions = try allocator.alloc(u64, total_positions);
    errdefer allocator.free(positions);
    const heldout_values = try allocator.alloc(f32, total_heldout_values);
    errdefer allocator.free(heldout_values);
    const heldout_positions = try allocator.alloc(u64, total_heldout_positions);
    errdefer allocator.free(heldout_positions);
    const coverage_offsets = try allocator.alloc(u32, primary.len + 1);
    errdefer allocator.free(coverage_offsets);
    coverage_offsets[0] = 0;
    var coverage_list = std.ArrayList(u32).empty;
    errdefer coverage_list.deinit(allocator);
    try coverage_list.ensureTotalCapacityPrecise(allocator, maximum_coverage_indices);
    const coverage_by_batch = try allocator.alloc(?[]u32, primary.len);
    defer allocator.free(coverage_by_batch);
    @memset(coverage_by_batch, null);
    errdefer for (coverage_by_batch) |entry| {
        if (entry) |owned| allocator.free(owned);
    };
    const heads = try allocator.alloc(HeadEvidence, plans.len);
    var heads_done: usize = 0;
    errdefer {
        for (heads[0..heads_done]) |head| allocator.free(head.maximum_probability);
        allocator.free(heads);
    }

    var value_at: usize = 0;
    var position_at: usize = 0;
    var heldout_value_at: usize = 0;
    var heldout_position_at: usize = 0;
    var aggregate = std.crypto.hash.sha2.Sha256.init(.{});
    aggregate.update(policy);
    for (plans, 0..) |binding, plan_index| {
        try control.check();
        if (binding.donor >= state.donors.len or binding.head_index >= state.donors[binding.donor].heads.len) return error.NativeCoverageShapeUnsupported;
        const source = state.donors[binding.donor].heads[binding.head_index];
        const fit_ranges = try engine.headFitRanges(allocator, source, binding.plan, chunk_rows);
        defer allocator.free(fit_ranges);
        const ranges = try allocator.alloc(@import("attention_match_cumulative_cache.zig").Range, fit_ranges.len);
        defer allocator.free(ranges);
        for (ranges, fit_ranges) |*out, range| out.* = .{ .start = range.start, .rows = range.rows };
        var consumer_count: usize = 0;
        for (primary) |batch| {
            const mapped = try mappedHead(state, layout, batch);
            consumer_count += @intFromBool(mapped[0] == binding.donor and mapped[1] == binding.head_index);
        }
        if (consumer_count == 0) return error.NativeCoverageShapeUnsupported;
        const adapter_consumers = try allocator.alloc(adapter.Consumer, consumer_count);
        defer allocator.free(adapter_consumers);
        var adapter_at: usize = 0;
        var first_index: ?usize = null;
        for (primary, historical_dense, future, 0..) |p, h, f, batch_index| {
            const mapped = try mappedHead(state, layout, p);
            if (mapped[0] != binding.donor or mapped[1] != binding.head_index) continue;
            if (first_index == null) first_index = batch_index;
            adapter_consumers[adapter_at] = .{ .primary_queries = p.train, .future_queries = f.train, .historical_queries = h.train, .historical_positions = h.train_positions, .heldout_positions = p.heldout_positions };
            adapter_at += 1;
        }
        const first = first_index orelse return error.NativeCoverageShapeUnsupported;
        const dim: usize = source.dim;
        const shape: bridge.Shape = .{ .dim = dim, .source_rows = source.positions.len, .primary_rows = primary[first].train_positions.len, .historical_rows = historical_dense[first].train_positions.len, .future_rows = future[first].train_positions.len, .heldout_rows = primary[first].heldout_positions.len, .selected_historical_rows = selected_historical_rows, .consumer_count = consumer_count, .expected_head_count = plans.len, .actual_head_count = plans.len, .route = .flat };
        const live = try add(try add(caller_external_live_bytes, output_live), maximum_head_scratch);
        var derived = try adapter.derive(allocator, source, adapter_consumers, .{ .shape = shape, .ranges = ranges, .donor_layer = state.donors[binding.donor].layer, .head_index = binding.head_index, .expected_donor_layer = state.donors[binding.donor].layer, .expected_head_index = binding.head_index, .scale = layout.scales_by_donor[binding.donor], .softcap = layout.softcaps_by_donor[binding.donor], .external_live_bytes = live, .combined_cap = combined_cap, .control = control });
        defer derived.deinit();
        const maximum_probability = try allocator.dupe(f64, derived.source_key_weights);
        heads[plan_index] = .{ .donor_layer = state.donors[binding.donor].layer, .head_index = binding.head_index, .maximum_probability = maximum_probability, .authority_digest = derived.authority_digest };
        heads_done += 1;
        aggregate.update(&derived.authority_digest);
        var selected_at: usize = 0;
        for (primary, historical_dense, future, 0..) |p, h, f, batch_index| {
            const mapped = try mappedHead(state, layout, p);
            if (mapped[0] != binding.donor or mapped[1] != binding.head_index) continue;
            const selected = derived.selection.consumers[selected_at].selected_indices;
            const coverage = derived.selection.consumers[selected_at].coverage_indices;
            if (coverage_by_batch[batch_index] != null) return error.NativeCoverageShapeUnsupported;
            selected_at += 1;
            const begin_value = value_at;
            const begin_position = position_at;
            const begin_heldout_value = heldout_value_at;
            const begin_heldout_position = heldout_position_at;
            @memcpy(values[value_at..][0..p.train.len], p.train);
            @memcpy(positions[position_at..][0..p.train_positions.len], p.train_positions);
            value_at += p.train.len;
            position_at += p.train_positions.len;
            for (selected, 0..) |row, out_row| {
                @memcpy(values[value_at + out_row * dim ..][0..dim], h.train[@as(usize, row) * dim ..][0..dim]);
                positions[position_at + out_row] = h.train_positions[row];
            }
            value_at += selected.len * dim;
            position_at += selected.len;
            @memcpy(values[value_at..][0..f.train.len], f.train);
            @memcpy(positions[position_at..][0..f.train_positions.len], f.train_positions);
            value_at += f.train.len;
            position_at += f.train_positions.len;
            @memcpy(heldout_values[heldout_value_at..][0..p.heldout.len], p.heldout);
            @memcpy(heldout_positions[heldout_position_at..][0..p.heldout_positions.len], p.heldout_positions);
            heldout_value_at += p.heldout.len;
            heldout_position_at += p.heldout_positions.len;
            const primary_end = begin_value + p.train.len;
            const historical_end = primary_end + selected.len * dim;
            batches[batch_index] = .{ .consumer_layer = p.consumer_layer, .query_head = p.query_head, .train = values[begin_value..value_at], .heldout = heldout_values[begin_heldout_value..heldout_value_at], .train_positions = positions[begin_position..position_at], .heldout_positions = heldout_positions[begin_heldout_position..heldout_position_at], .independent_train_split = p.train_positions.len, .independent_train_second_split = p.train_positions.len + selected.len };
            for (coverage) |row| if (std.mem.indexOfScalar(u32, selected, row) == null) return error.NativeCoverageShapeUnsupported;
            coverage_by_batch[batch_index] = try allocator.dupe(u32, coverage);
            aggregate.update(std.mem.sliceAsBytes(values[primary_end..historical_end]));
            aggregate.update(std.mem.sliceAsBytes(positions[begin_position + p.train_positions.len ..][0..selected.len]));
        }
    }
    if (value_at != total_values or position_at != total_positions or heldout_value_at != total_heldout_values or heldout_position_at != total_heldout_positions or heads_done != plans.len) return error.NativeCoverageShapeUnsupported;
    for (coverage_by_batch, 0..) |entry, batch_index| {
        // Empty coverage is valid when P/F already cover every critical key.
        const owned = entry orelse return error.NativeCoverageShapeUnsupported;
        try coverage_list.appendSlice(allocator, owned);
        coverage_offsets[batch_index + 1] = @intCast(coverage_list.items.len);
    }
    for (coverage_by_batch) |entry| {
        if (entry) |owned| allocator.free(owned);
    }
    @memset(coverage_by_batch, null);
    for (batches) |batch| {
        aggregate.update(std.mem.asBytes(&batch.consumer_layer));
        aggregate.update(std.mem.asBytes(&batch.query_head));
        aggregate.update(std.mem.sliceAsBytes(batch.train));
        aggregate.update(std.mem.sliceAsBytes(batch.train_positions));
        aggregate.update(std.mem.sliceAsBytes(batch.heldout));
        aggregate.update(std.mem.sliceAsBytes(batch.heldout_positions));
        aggregate.update(std.mem.asBytes(&batch.independent_train_split));
        aggregate.update(std.mem.asBytes(&batch.independent_train_second_split));
    }
    const coverage_indices = try coverage_list.toOwnedSlice(allocator);
    // This is the authority consumed by the TRAIN-challenge rewrite. Bind the
    // CSR to its ordered consumer tuples, not merely to two untyped arrays.
    var coverage_hash = std.crypto.hash.sha2.Sha256.init(.{});
    coverage_hash.update("antfly.production768-historical-coverage-csr-authority.v1\x00");
    var bytes8: [8]u8 = undefined;
    var bytes4: [4]u8 = undefined;
    std.mem.writeInt(u64, &bytes8, @intCast(batches.len), .little);
    coverage_hash.update(&bytes8);
    for (batches, 0..) |batch, batch_index| {
        const begin: usize = coverage_offsets[batch_index];
        const end: usize = coverage_offsets[batch_index + 1];
        var frame: [16]u8 = undefined;
        std.mem.writeInt(u32, frame[0..4], batch.consumer_layer, .little);
        std.mem.writeInt(u32, frame[4..8], batch.query_head, .little);
        std.mem.writeInt(u32, frame[8..12], @intCast(batch_index), .little);
        std.mem.writeInt(u32, frame[12..16], @intCast(end - begin), .little);
        coverage_hash.update(&frame);
        for (coverage_indices[begin..end]) |index| {
            std.mem.writeInt(u32, &bytes4, index, .little);
            coverage_hash.update(&bytes4);
        }
    }
    for (coverage_offsets) |offset| {
        std.mem.writeInt(u32, &bytes4, offset, .little);
        coverage_hash.update(&bytes4);
    }
    return .{ .allocator = allocator, .batches = batches, .values = values, .positions = positions, .heldout_values = heldout_values, .heldout_positions = heldout_positions, .coverage_offsets = coverage_offsets, .coverage_indices = coverage_indices, .coverage_authority_sha256 = coverage_hash.finalResult(), .heads = heads, .digest = aggregate.finalResult() };
}

fn mergeFixture(allocator: std.mem.Allocator, cap: usize, control: @import("attention_match.zig").Control) !void {
    const source_positions = [_]u64{ 0, 1 };
    const source_coverage = [_]u64{ 1, 2 };
    const source_keys = [_]f32{ 1, -1 };
    const source_values = [_]f32{ 3, 4 };
    const source_bias = [_]f32{ 0, 0 };
    const heads = [_]checkpoint.Head{.{ .dim = 1, .positions = &source_positions, .coverage_ends = &source_coverage, .keys = &source_keys, .values = &source_values, .bias = &source_bias }};
    const donors = [_]checkpoint.Donor{.{ .layer = 5, .heads = &heads }};
    const state: checkpoint.State = .{ .identity = undefined, .logical_length = 2, .layer_donors = &.{ 0, 0 }, .donors = &donors };
    const layout: engine.Layout = .{ .query_heads_by_layer = &.{ 1, 1 }, .scales_by_donor = &.{1}, .softcaps_by_donor = &.{0} };
    const primary = [_]engine.ReferenceBatch{
        .{ .consumer_layer = 0, .query_head = 0, .train = &.{1}, .train_positions = &.{100}, .heldout = &.{7}, .heldout_positions = &.{11} },
        .{ .consumer_layer = 1, .query_head = 0, .train = &.{-1}, .train_positions = &.{200}, .heldout = &.{8}, .heldout_positions = &.{21} },
    };
    const historical = [_]engine.ReferenceBatch{
        .{ .consumer_layer = 0, .query_head = 0, .train = &.{ 1, -1, 2 }, .train_positions = &.{ 10, 11, 12 }, .heldout = &.{}, .heldout_positions = &.{} },
        .{ .consumer_layer = 1, .query_head = 0, .train = &.{ -1, -2, 1 }, .train_positions = &.{ 20, 21, 22 }, .heldout = &.{}, .heldout_positions = &.{} },
    };
    const future = [_]engine.ReferenceBatch{
        .{ .consumer_layer = 0, .query_head = 0, .train = &.{0.5}, .train_positions = &.{300}, .heldout = &.{}, .heldout_positions = &.{} },
        .{ .consumer_layer = 1, .query_head = 0, .train = &.{-0.5}, .train_positions = &.{400}, .heldout = &.{}, .heldout_positions = &.{} },
    };
    const plans = [_]Plan{.{ .donor = 0, .head_index = 0, .plan = .{ .old_rows = 2, .prefix_rows = 0, .target_rows = 1, .block_rows = 1 } }};
    var owned = try merge(allocator, state, layout, &primary, &historical, &future, &plans, 2, 2, 0, cap, control);
    defer owned.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 2, 0.5 }, owned.batches[0].train);
    try std.testing.expectEqualSlices(u64, &.{ 100, 10, 12, 300 }, owned.batches[0].train_positions);
    try std.testing.expectEqualSlices(f32, primary[0].heldout, owned.batches[0].heldout);
    try std.testing.expectEqualSlices(u64, primary[0].heldout_positions, owned.batches[0].heldout_positions);
    try std.testing.expectEqual(@as(usize, 2), owned.heads[0].maximum_probability.len);
}

test "native merge body owns roles rejects cap cancels and is allocation atomic" {
    try std.testing.expectError(error.CompactionWorkspaceLimit, mergeFixture(std.testing.failing_allocator, 1, .{}));
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, mergeFixture(std.testing.allocator, 1 << 20, .{ .check_fn = Cancel.check }));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, mergeFixture, .{ @as(usize, 1 << 20), @import("attention_match.zig").Control{} });
}
