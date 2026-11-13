const std = @import("std");
const solver = @import("attention_match.zig");
const checkpoint = @import("attention_match_checkpoint.zig");
const admission = @import("attention_match_cumulative_cache.zig");
const selector = @import("attention_match_train_historical_coverage_selection.zig");
const bridge = @import("attention_match_native_historical_coverage_bridge.zig");
const accelerate_scan = @import("attention_match_accelerate_fp64_probability_scan.zig");

pub const policy = "native_checkpoint_coverage_data_adapter_v1|prob=scalar_fp64_then_f32_widened|baseline=P_plus_F|candidate=H|HHO=positions_only|source=checkpoint_head|export=none";
pub const gather_policy = "native_checkpoint_coverage_accelerate_gather_eligible_v1|eligible=position_exclusion_before_query_read|mapping=original_rows|publication=f32";
pub const threshold: f64 = 0.025;

pub const Consumer = struct {
    primary_queries: []const f32,
    future_queries: []const f32,
    historical_queries: []const f32,
    historical_positions: []const u64,
    heldout_positions: []const u64,
};

pub const Config = struct {
    shape: bridge.Shape,
    ranges: []const admission.Range,
    donor_layer: u32,
    head_index: usize,
    expected_donor_layer: u32,
    expected_head_index: usize,
    scale: f64,
    softcap: f64,
    external_live_bytes: usize,
    combined_cap: usize,
    control: solver.Control = .{},
    probability_backend: ?accelerate_scan.Backend = null,
    gather_eligible_queries: bool = false,
};

pub const Owned = struct {
    allocator: std.mem.Allocator,
    selection: bridge.Owned,
    source_key_weights: []f64,
    baseline_covered_keys: []u32,
    authority_digest: [32]u8,

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.baseline_covered_keys);
        self.allocator.free(self.source_key_weights);
        self.selection.deinit();
        self.* = undefined;
    }
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}

fn validateRanges(ranges: []const admission.Range, source_rows: usize) !void {
    if (ranges.len == 0) return error.NativeCoverageShapeUnsupported;
    var prior_end: usize = 0;
    for (ranges, 0..) |range, index| {
        const end = std.math.add(usize, range.start, range.rows) catch return error.NativeCoverageShapeUnsupported;
        if (range.rows == 0 or end > source_rows or (index != 0 and range.start < prior_end)) return error.NativeCoverageShapeUnsupported;
        prior_end = end;
    }
}

fn excluded(position: u64, heldout: []const u64) bool {
    for (heldout) |value| if (position == value) return true;
    return false;
}

fn scan(
    allocator: std.mem.Allocator,
    head: solver.Head,
    queries: []const f32,
    positions: ?[]const u64,
    heldout_positions: []const u64,
    weights: []f64,
    baseline: ?[]bool,
    offsets: ?[]u32,
    keys: ?*std.ArrayList(u32),
    query: []f64,
    probabilities: []f64,
    control: solver.Control,
    backend: ?accelerate_scan.Backend,
    gather_eligible_queries: bool,
    external_live_bytes: usize,
    combined_cap: usize,
) !void {
    if (queries.len == 0 or queries.len % head.dim != 0 or weights.len != head.rows) return error.NativeCoverageShapeUnsupported;
    const rows = queries.len / head.dim;
    if (positions) |p| if (p.len != rows) return error.NativeCoverageShapeUnsupported;
    if (offsets) |o| if (o.len != rows + 1 or keys == null) return error.NativeCoverageShapeUnsupported;
    if (backend) |accelerated| {
        const Callback = struct {
            allocator: std.mem.Allocator,
            positions: ?[]const u64,
            heldout_positions: []const u64,
            weights: []f64,
            baseline: ?[]bool,
            offsets: ?[]u32,
            keys: ?*std.ArrayList(u32),
            original_row_base: usize = 0,
            original_rows: ?[]const u32 = null,
            next_original_row: usize = 0,

            fn consume(raw: ?*anyopaque, local_row: usize, row_probabilities: []const f32) !void {
                const self: *@This() = @ptrCast(@alignCast(raw.?));
                const row = if (self.original_rows) |mapping| mapping[local_row] else self.original_row_base + local_row;
                if (self.offsets) |o| {
                    while (self.next_original_row < row) : (self.next_original_row += 1) o[self.next_original_row + 1] = @intCast(self.keys.?.items.len);
                }
                var count: usize = 0;
                for (row_probabilities, self.weights, 0..) |probability32, *maximum, source_row| {
                    const probability: f64 = probability32;
                    if (!std.math.isFinite(probability) or probability < 0 or probability > 1) return error.NonFiniteAttention;
                    maximum.* = @max(maximum.*, probability);
                    if (probability >= threshold) {
                        if (self.baseline) |covered| covered[source_row] = true;
                        if (self.keys) |list| {
                            if (count == selector.max_critical_keys_per_query) return error.NativeCoverageShapeUnsupported;
                            try list.append(self.allocator, @intCast(source_row));
                            count += 1;
                        }
                    }
                }
                if (self.offsets) |o| o[row + 1] = @intCast(self.keys.?.items.len);
                self.next_original_row = row + 1;
            }
        };
        var callback: Callback = .{ .allocator = allocator, .positions = positions, .heldout_positions = heldout_positions, .weights = weights, .baseline = baseline, .offsets = offsets, .keys = keys };
        if (gather_eligible_queries and positions != null) {
            var eligible_count: usize = 0;
            for (positions.?) |position| eligible_count += @intFromBool(!excluded(position, heldout_positions));
            if (eligible_count == 0) {
                if (offsets) |o| {
                    for (1..o.len) |index| o[index] = @intCast(keys.?.items.len);
                }
                return;
            }
            const gathered_values = try mul(eligible_count, head.dim);
            const gather_bytes = try add(try mul(gathered_values, @sizeOf(f32)), try mul(eligible_count, @sizeOf(u32)));
            if (external_live_bytes > combined_cap or gather_bytes > combined_cap - external_live_bytes) return error.CompactionWorkspaceLimit;
            const gathered = try allocator.alloc(f32, gathered_values);
            defer allocator.free(gathered);
            const original_rows = try allocator.alloc(u32, eligible_count);
            defer allocator.free(original_rows);
            var eligible_at: usize = 0;
            for (positions.?, 0..) |position, original_row| {
                if (excluded(position, heldout_positions)) {
                    continue;
                }
                original_rows[eligible_at] = @intCast(original_row);
                @memcpy(gathered[eligible_at * head.dim ..][0..head.dim], queries[original_row * head.dim ..][0..head.dim]);
                eligible_at += 1;
            }
            callback.original_rows = original_rows;
            try accelerated.run(allocator, head, gathered, try add(external_live_bytes, gather_bytes), combined_cap, control, .{ .context = &callback, .row_fn = Callback.consume });
            if (offsets) |o| {
                while (callback.next_original_row < rows) : (callback.next_original_row += 1) o[callback.next_original_row + 1] = @intCast(keys.?.items.len);
            }
            return;
        }
        var at: usize = 0;
        while (at < rows) {
            try control.check();
            if (positions) |p| if (excluded(p[at], heldout_positions)) {
                if (offsets) |o| o[at + 1] = @intCast(keys.?.items.len);
                at += 1;
                continue;
            };
            const begin = at;
            while (at < rows) : (at += 1) {
                if (positions) |p| if (excluded(p[at], heldout_positions)) break;
            }
            callback.original_row_base = begin;
            try accelerated.run(allocator, head, queries[begin * head.dim .. at * head.dim], external_live_bytes, combined_cap, control, .{ .context = &callback, .row_fn = Callback.consume });
        }
        return;
    }
    for (0..rows) |row| {
        try control.check();
        if (positions) |p| if (excluded(p[row], heldout_positions)) {
            if (offsets) |o| o[row + 1] = @intCast(keys.?.items.len);
            continue;
        };
        for (query, queries[row * head.dim ..][0..head.dim]) |*out, value| {
            if (!std.math.isFinite(value)) return error.NonFiniteAttention;
            out.* = value;
        }
        _ = try solver.attend(head, query, probabilities, null);
        var count: usize = 0;
        for (probabilities, weights, 0..) |raw, *maximum, source_row| {
            const probability: f64 = @floatCast(@as(f32, @floatCast(raw)));
            if (!std.math.isFinite(probability) or probability < 0 or probability > 1) return error.NonFiniteAttention;
            maximum.* = @max(maximum.*, probability);
            if (probability >= threshold) {
                if (baseline) |covered| covered[source_row] = true;
                if (keys) |list| {
                    if (count == selector.max_critical_keys_per_query) return error.NativeCoverageShapeUnsupported;
                    try list.append(allocator, @intCast(source_row));
                    count += 1;
                }
            }
        }
        if (offsets) |o| o[row + 1] = @intCast(keys.?.items.len);
    }
}

pub fn derive(allocator: std.mem.Allocator, source: checkpoint.Head, consumers: []const Consumer, cfg: Config) !Owned {
    try cfg.shape.validate();
    try cfg.control.check();
    try validateRanges(cfg.ranges, cfg.shape.source_rows);
    if (cfg.donor_layer != cfg.expected_donor_layer or cfg.head_index != cfg.expected_head_index or
        (cfg.gather_eligible_queries and cfg.probability_backend == null) or
        consumers.len != cfg.shape.consumer_count or source.dim != cfg.shape.dim or source.positions.len != cfg.shape.source_rows or
        source.keys.len != try mul(cfg.shape.source_rows, cfg.shape.dim) or source.values.len != source.keys.len or source.bias.len != cfg.shape.source_rows or
        cfg.external_live_bytes > cfg.combined_cap or !std.math.isFinite(cfg.scale) or cfg.scale <= 0 or !std.math.isFinite(cfg.softcap) or cfg.softcap < 0)
        return error.NativeCoverageShapeUnsupported;
    for (consumers) |consumer| {
        if (consumer.primary_queries.len != try mul(cfg.shape.primary_rows, cfg.shape.dim) or
            consumer.future_queries.len != try mul(cfg.shape.future_rows, cfg.shape.dim) or
            consumer.historical_queries.len != try mul(cfg.shape.historical_rows, cfg.shape.dim) or
            consumer.historical_positions.len != cfg.shape.historical_rows or consumer.heldout_positions.len != cfg.shape.heldout_rows)
            return error.NativeCoverageShapeUnsupported;
    }

    const normalized_elements = try add(try mul(cfg.shape.source_rows, try mul(cfg.shape.dim, 2)), cfg.shape.source_rows);
    var retained = try mul(normalized_elements, @sizeOf(f64));
    retained = try add(retained, try mul(cfg.shape.source_rows, @sizeOf(f64) + @sizeOf(bool)));
    retained = try add(retained, try mul(cfg.shape.dim + cfg.shape.source_rows, @sizeOf(f64)));
    retained = try add(retained, try mul(consumers.len, @sizeOf(bridge.ConsumerInput)));
    retained = try add(retained, try mul(consumers.len, @sizeOf([]u32)));
    retained = try add(retained, try mul(consumers.len, @sizeOf(std.ArrayList(u32))));
    retained = try add(retained, try mul(consumers.len, try mul(cfg.shape.historical_rows + 1, @sizeOf(u32))));
    retained = try add(retained, try mul(try mul(consumers.len, cfg.shape.historical_rows), try mul(selector.max_critical_keys_per_query, @sizeOf(u32))));
    retained = try add(retained, try mul(cfg.shape.source_rows, @sizeOf(u32)));
    if (retained > cfg.combined_cap - cfg.external_live_bytes) return error.CompactionWorkspaceLimit;

    const keys = try allocator.alloc(f64, source.keys.len);
    defer allocator.free(keys);
    const values = try allocator.alloc(f64, source.values.len);
    defer allocator.free(values);
    const bias = try allocator.alloc(f64, source.bias.len);
    defer allocator.free(bias);
    for (keys, source.keys) |*out, value| out.* = value;
    for (values, source.values) |*out, value| out.* = value;
    for (bias, source.bias) |*out, value| out.* = value;
    const head: solver.Head = .{ .keys = keys, .values = values, .bias = bias, .rows = cfg.shape.source_rows, .dim = cfg.shape.dim, .scale = cfg.scale, .softcap = cfg.softcap };
    try head.validate();

    const weights = try allocator.alloc(f64, cfg.shape.source_rows);
    errdefer allocator.free(weights);
    @memset(weights, 0);
    const covered = try allocator.alloc(bool, cfg.shape.source_rows);
    defer allocator.free(covered);
    @memset(covered, false);
    const query = try allocator.alloc(f64, cfg.shape.dim);
    defer allocator.free(query);
    const probabilities = try allocator.alloc(f64, cfg.shape.source_rows);
    defer allocator.free(probabilities);

    const bridge_inputs = try allocator.alloc(bridge.ConsumerInput, consumers.len);
    defer allocator.free(bridge_inputs);
    const offsets_owner = try allocator.alloc([]u32, consumers.len);
    for (offsets_owner) |*offsets| offsets.* = &.{};
    defer {
        for (offsets_owner) |offsets| if (offsets.len != 0) allocator.free(offsets);
        allocator.free(offsets_owner);
    }
    var key_lists = try allocator.alloc(std.ArrayList(u32), consumers.len);
    defer {
        for (key_lists) |*list| list.deinit(allocator);
        allocator.free(key_lists);
    }
    for (key_lists) |*list| list.* = .empty;

    for (consumers, 0..) |consumer, index| {
        const scan_external = try add(cfg.external_live_bytes, retained);
        try scan(allocator, head, consumer.primary_queries, null, &.{}, weights, covered, null, null, query, probabilities, cfg.control, cfg.probability_backend, false, scan_external, cfg.combined_cap);
        try scan(allocator, head, consumer.future_queries, null, &.{}, weights, covered, null, null, query, probabilities, cfg.control, cfg.probability_backend, false, scan_external, cfg.combined_cap);
        const offsets = try allocator.alloc(u32, cfg.shape.historical_rows + 1);
        offsets_owner[index] = offsets;
        @memset(offsets, 0);
        try key_lists[index].ensureTotalCapacityPrecise(allocator, try mul(cfg.shape.historical_rows, selector.max_critical_keys_per_query));
        try scan(allocator, head, consumer.historical_queries, consumer.historical_positions, consumer.heldout_positions, weights, null, offsets, &key_lists[index], query, probabilities, cfg.control, cfg.probability_backend, cfg.gather_eligible_queries, scan_external, cfg.combined_cap);
    }
    var covered_count: usize = 0;
    for (covered) |value| covered_count += @intFromBool(value);
    const baseline_keys = try allocator.alloc(u32, covered_count);
    errdefer allocator.free(baseline_keys);
    var covered_at: usize = 0;
    for (covered, 0..) |value, index| if (value) {
        baseline_keys[covered_at] = @intCast(index);
        covered_at += 1;
    };
    for (consumers, bridge_inputs, offsets_owner, key_lists) |consumer, *input, offsets, list| input.* = .{
        .primary_queries = consumer.primary_queries,
        .future_queries = consumer.future_queries,
        .historical_queries = consumer.historical_queries,
        .historical_positions = consumer.historical_positions,
        .heldout_positions = consumer.heldout_positions,
        .source_key_weights = weights,
        .baseline_covered_keys = baseline_keys,
        .historical_candidate_offsets = offsets,
        .historical_candidate_keys = list.items,
    };
    const adapter_live = try add(cfg.external_live_bytes, retained);
    var selection = try bridge.select(allocator, cfg.shape, bridge_inputs, adapter_live, cfg.combined_cap, cfg.control);
    errdefer selection.deinit();
    var digest = std.crypto.hash.sha2.Sha256.init(.{});
    digest.update(policy);
    if (cfg.probability_backend != null) digest.update(accelerate_scan.policy);
    if (cfg.gather_eligible_queries) digest.update(gather_policy);
    digest.update(std.mem.asBytes(&cfg.donor_layer));
    digest.update(std.mem.asBytes(&cfg.head_index));
    digest.update(std.mem.asBytes(&cfg.scale));
    digest.update(std.mem.asBytes(&cfg.softcap));
    inline for (.{ "dim", "source_rows", "primary_rows", "historical_rows", "future_rows", "heldout_rows", "selected_historical_rows", "consumer_count", "expected_head_count", "actual_head_count" }) |field| {
        const value = @field(cfg.shape, field);
        digest.update(std.mem.asBytes(&value));
    }
    digest.update(&.{@backingInt(cfg.shape.route)});
    digest.update(std.mem.sliceAsBytes(source.positions));
    digest.update(std.mem.sliceAsBytes(source.coverage_ends));
    digest.update(std.mem.sliceAsBytes(source.keys));
    digest.update(std.mem.sliceAsBytes(source.values));
    digest.update(std.mem.sliceAsBytes(source.bias));
    for (cfg.ranges) |range| {
        digest.update(std.mem.asBytes(&range.start));
        digest.update(std.mem.asBytes(&range.rows));
    }
    for (consumers) |consumer| {
        digest.update(std.mem.sliceAsBytes(consumer.primary_queries));
        digest.update(std.mem.sliceAsBytes(consumer.future_queries));
        if (cfg.probability_backend == null) {
            // Preserve the established scalar authority byte stream.
            digest.update(std.mem.sliceAsBytes(consumer.historical_queries));
        } else {
            for (consumer.historical_positions, 0..) |position, row| if (!excluded(position, consumer.heldout_positions))
                digest.update(std.mem.sliceAsBytes(consumer.historical_queries[row * cfg.shape.dim ..][0..cfg.shape.dim]));
        }
        digest.update(std.mem.sliceAsBytes(consumer.historical_positions));
        digest.update(std.mem.sliceAsBytes(consumer.heldout_positions));
    }
    digest.update(std.mem.sliceAsBytes(weights));
    digest.update(std.mem.sliceAsBytes(baseline_keys));
    digest.update(&selection.digest);
    return .{ .allocator = allocator, .selection = selection, .source_key_weights = weights, .baseline_covered_keys = baseline_keys, .authority_digest = digest.finalResult() };
}

test "checkpoint adapter computes rounded probability authority and excludes HHO historical row" {
    const positions = [_]u64{ 0, 1 };
    const coverage = [_]u64{ 1, 2 };
    const keys = [_]f32{ 1, -1 };
    const values = [_]f32{ 1, 2 };
    const bias = [_]f32{ 0, 0 };
    const source: checkpoint.Head = .{ .dim = 1, .positions = &positions, .coverage_ends = &coverage, .keys = &keys, .values = &values, .bias = &bias };
    const consumer: Consumer = .{ .primary_queries = &.{1}, .future_queries = &.{1}, .historical_queries = &.{ 1, -1, 2 }, .historical_positions = &.{ 10, 11, 12 }, .heldout_positions = &.{11} };
    const shape: bridge.Shape = .{ .dim = 1, .source_rows = 2, .primary_rows = 1, .historical_rows = 3, .future_rows = 1, .heldout_rows = 1, .selected_historical_rows = 2, .consumer_count = 1, .expected_head_count = 1, .actual_head_count = 1, .route = .flat };
    var owned = try derive(std.testing.allocator, source, &.{consumer}, .{ .shape = shape, .ranges = &.{.{ .start = 0, .rows = 2 }}, .donor_layer = 5, .head_index = 0, .expected_donor_layer = 5, .expected_head_index = 0, .scale = 1, .softcap = 0, .external_live_bytes = 0, .combined_cap = 1 << 20 });
    defer owned.deinit();
    try std.testing.expectEqual(@as(usize, 2), owned.selection.consumers[0].selected_indices.len);
    for (owned.selection.consumers[0].selected_indices) |index| try std.testing.expect(index != 1);
    try std.testing.expect(!std.mem.allEqual(u8, &owned.authority_digest, 0));
    const expected: f64 = @floatCast(@as(f32, @floatCast(1 / (1 + @exp(@as(f64, -4))))));
    try std.testing.expectEqual(expected, owned.source_key_weights[0]);
}

test "checkpoint adapter rejects mismatched topology range and cap before probability work" {
    const empty: checkpoint.Head = .{};
    const shape: bridge.Shape = .{ .dim = 1, .source_rows = 2, .primary_rows = 1, .historical_rows = 3, .future_rows = 1, .heldout_rows = 1, .selected_historical_rows = 2, .consumer_count = 1, .expected_head_count = 1, .actual_head_count = 1, .route = .flat };
    try std.testing.expectError(error.NativeCoverageShapeUnsupported, derive(std.testing.failing_allocator, empty, &.{}, .{ .shape = shape, .ranges = &.{.{ .start = 2, .rows = 1 }}, .donor_layer = 0, .head_index = 0, .expected_donor_layer = 0, .expected_head_index = 0, .scale = 1, .softcap = 0, .external_live_bytes = 0, .combined_cap = 1 }));
}

fn deriveFixture(allocator: std.mem.Allocator, control: solver.Control, cap: usize) !void {
    const positions = [_]u64{ 0, 1 };
    const coverage = [_]u64{ 1, 2 };
    const keys = [_]f32{ 1, -1 };
    const values = [_]f32{ 1, 2 };
    const bias = [_]f32{ 0, 0 };
    const source: checkpoint.Head = .{ .dim = 1, .positions = &positions, .coverage_ends = &coverage, .keys = &keys, .values = &values, .bias = &bias };
    const consumers = [_]Consumer{
        .{ .primary_queries = &.{1}, .future_queries = &.{1}, .historical_queries = &.{ 1, -1, 2 }, .historical_positions = &.{ 10, 11, 12 }, .heldout_positions = &.{11} },
        .{ .primary_queries = &.{-1}, .future_queries = &.{0}, .historical_queries = &.{ -1, -2, 1 }, .historical_positions = &.{ 20, 21, 22 }, .heldout_positions = &.{21} },
    };
    const shape: bridge.Shape = .{ .dim = 1, .source_rows = 2, .primary_rows = 1, .historical_rows = 3, .future_rows = 1, .heldout_rows = 1, .selected_historical_rows = 2, .consumer_count = 2, .expected_head_count = 1, .actual_head_count = 1, .route = .flat };
    var owned = try derive(allocator, source, &consumers, .{ .shape = shape, .ranges = &.{.{ .start = 0, .rows = 2 }}, .donor_layer = 5, .head_index = 0, .expected_donor_layer = 5, .expected_head_index = 0, .scale = 1, .softcap = 0, .external_live_bytes = 0, .combined_cap = cap, .control = control });
    owned.deinit();
}

test "checkpoint adapter body rejects one-byte cap cancels mid scan and is allocation atomic" {
    try std.testing.expectError(error.CompactionWorkspaceLimit, deriveFixture(std.testing.failing_allocator, .{}, 1));
    const Cancel = struct {
        var calls: usize = 0;
        fn check(_: ?*anyopaque) !void {
            calls += 1;
            if (calls == 4) return error.Cancelled;
        }
    };
    Cancel.calls = 0;
    try std.testing.expectError(error.Cancelled, deriveFixture(std.testing.allocator, .{ .check_fn = Cancel.check }, 1 << 20));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, deriveFixture, .{ solver.Control{}, @as(usize, 1 << 20) });
}

test "accelerated adapter never reads excluded historical query values and preserves eligible selection" {
    const Gemm = struct {
        fn run(_: *accelerate_scan.Context, m: c_int, n: c_int, k: c_int, alpha: f64, a: [*]const f64, lda: c_int, b: [*]const f64, ldb: c_int, beta: f64, c: [*]f64, ldc: c_int) void {
            for (0..@intCast(m)) |i| for (0..@intCast(n)) |j| {
                var sum: f64 = 0;
                for (0..@intCast(k)) |x| sum += a[i * @as(usize, @intCast(lda)) + x] * b[j * @as(usize, @intCast(ldb)) + x];
                const out = i * @as(usize, @intCast(ldc)) + j;
                c[out] = alpha * sum + if (beta == 0) 0 else beta * c[out];
            };
        }
    };
    const positions = [_]u64{ 0, 1 };
    const coverage = [_]u64{ 1, 2 };
    const keys = [_]f32{ 1, -1 };
    const values = [_]f32{ 1, 2 };
    const bias = [_]f32{ 0, 0 };
    const source: checkpoint.Head = .{ .dim = 1, .positions = &positions, .coverage_ends = &coverage, .keys = &keys, .values = &values, .bias = &bias };
    const historical = [_]f32{ 1, std.math.nan(f32), 2 };
    const consumer: Consumer = .{ .primary_queries = &.{1}, .future_queries = &.{-1}, .historical_queries = &historical, .historical_positions = &.{ 10, 11, 12 }, .heldout_positions = &.{11} };
    const shape: bridge.Shape = .{ .dim = 1, .source_rows = 2, .primary_rows = 1, .historical_rows = 3, .future_rows = 1, .heldout_rows = 1, .selected_historical_rows = 2, .consumer_count = 1, .expected_head_count = 1, .actual_head_count = 1, .route = .flat };
    const common: Config = .{ .shape = shape, .ranges = &.{.{ .start = 0, .rows = 2 }}, .donor_layer = 5, .head_index = 0, .expected_donor_layer = 5, .expected_head_index = 0, .scale = 1, .softcap = 0, .external_live_bytes = 0, .combined_cap = 1 << 20 };
    var scalar = try derive(std.testing.allocator, source, &.{consumer}, common);
    defer scalar.deinit();
    var context: accelerate_scan.Context = .{ .gemm_fn = Gemm.run };
    var accelerated_cfg = common;
    accelerated_cfg.probability_backend = accelerate_scan.backend(&context);
    var accelerated = try derive(std.testing.allocator, source, &.{consumer}, accelerated_cfg);
    defer accelerated.deinit();
    try std.testing.expectEqualSlices(f64, scalar.source_key_weights, accelerated.source_key_weights);
    try std.testing.expectEqualSlices(u32, scalar.selection.consumers[0].selected_indices, accelerated.selection.consumers[0].selected_indices);
    try std.testing.expect(!std.mem.eql(u8, &scalar.authority_digest, &accelerated.authority_digest));
    const span_calls = context.calls;
    var gather_context: accelerate_scan.Context = .{ .gemm_fn = Gemm.run };
    var gather_cfg = common;
    gather_cfg.probability_backend = accelerate_scan.backend(&gather_context);
    gather_cfg.gather_eligible_queries = true;
    var gathered = try derive(std.testing.allocator, source, &.{consumer}, gather_cfg);
    defer gathered.deinit();
    try std.testing.expectEqualSlices(f64, scalar.source_key_weights, gathered.source_key_weights);
    try std.testing.expectEqualSlices(u32, scalar.baseline_covered_keys, gathered.baseline_covered_keys);
    try std.testing.expectEqualSlices(u32, scalar.selection.consumers[0].selected_indices, gathered.selection.consumers[0].selected_indices);
    try std.testing.expect(gather_context.calls < span_calls);
    try std.testing.expect(!std.mem.eql(u8, &accelerated.authority_digest, &gathered.authority_digest));
    const changed_historical = [_]f32{ 1, std.math.inf(f32), 2 };
    var changed_consumer = consumer;
    changed_consumer.historical_queries = &changed_historical;
    var changed = try derive(std.testing.allocator, source, &.{changed_consumer}, accelerated_cfg);
    defer changed.deinit();
    try std.testing.expectEqualSlices(u8, &accelerated.authority_digest, &changed.authority_digest);
    var gathered_changed = try derive(std.testing.allocator, source, &.{changed_consumer}, gather_cfg);
    defer gathered_changed.deinit();
    try std.testing.expectEqualSlices(u8, &gathered.authority_digest, &gathered_changed.authority_digest);
    const Cancel = struct {
        var calls: usize = 0;
        fn check(_: ?*anyopaque) !void {
            calls += 1;
            if (calls == 7) return error.Cancelled;
        }
    };
    Cancel.calls = 0;
    var cancelled_cfg = accelerated_cfg;
    cancelled_cfg.control = .{ .check_fn = Cancel.check };
    try std.testing.expectError(error.Cancelled, derive(std.testing.allocator, source, &.{consumer}, cancelled_cfg));
    Cancel.calls = 0;
    var cancelled_gather_cfg = gather_cfg;
    cancelled_gather_cfg.control = .{ .check_fn = Cancel.check };
    try std.testing.expectError(error.Cancelled, derive(std.testing.allocator, source, &.{consumer}, cancelled_gather_cfg));
    var cap_cfg = gather_cfg;
    cap_cfg.combined_cap = 1;
    try std.testing.expectError(error.CompactionWorkspaceLimit, derive(std.testing.allocator, source, &.{consumer}, cap_cfg));
}
