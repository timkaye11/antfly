const std = @import("std");
const solver = @import("attention_match.zig");
const selector = @import("attention_match_train_historical_coverage_selection.zig");

pub const policy = "native_inmemory_historical_coverage_bridge_v1|shape=explicit|roles=P,H,F,HHO_positions_only|selection=coverage_then_q_kcenter|bounded=typed_reject|publication=none";

pub const Route = enum { flat, bounded };

/// Every cardinality comes from an authenticated producer binding. This module
/// intentionally has no 7936/23/8/p91 constants and publishes no policy.
pub const Shape = struct {
    dim: usize,
    source_rows: usize,
    primary_rows: usize,
    historical_rows: usize,
    future_rows: usize,
    heldout_rows: usize,
    selected_historical_rows: usize,
    consumer_count: usize,
    expected_head_count: usize,
    actual_head_count: usize,
    route: Route,

    pub fn validate(self: @This()) !void {
        if (self.route != .flat) return error.NativeCoverageBoundedUnsupported;
        if (self.dim == 0 or self.dim > selector.max_dim or self.source_rows == 0 or
            self.primary_rows == 0 or self.future_rows == 0 or self.historical_rows < 2 or
            self.historical_rows > selector.max_rows or self.heldout_rows == 0 or
            self.selected_historical_rows == 0 or self.selected_historical_rows > selector.max_selected or
            self.selected_historical_rows > self.historical_rows or self.consumer_count == 0 or
            self.expected_head_count == 0 or self.actual_head_count != self.expected_head_count)
            return error.NativeCoverageShapeUnsupported;
    }
};

pub const ConsumerInput = struct {
    primary_queries: []const f32,
    future_queries: []const f32,
    historical_queries: []const f32,
    historical_positions: []const u64,
    heldout_positions: []const u64,
    source_key_weights: []const f64,
    baseline_covered_keys: []const u32,
    historical_candidate_offsets: []const u32,
    historical_candidate_keys: []const u32,
};

pub const ConsumerEvidence = struct {
    selected_indices: []u32,
    coverage_indices: []u32,
    selector_evidence: selector.Evidence,
};

pub const Owned = struct {
    allocator: std.mem.Allocator,
    consumers: []ConsumerEvidence,
    digest: [32]u8,

    pub fn deinit(self: *@This()) void {
        for (self.consumers) |entry| {
            self.allocator.free(entry.coverage_indices);
            self.allocator.free(entry.selected_indices);
        }
        self.allocator.free(self.consumers);
        self.* = undefined;
    }
};

fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}

fn validateConsumer(shape: Shape, input: ConsumerInput) !void {
    if (input.primary_queries.len != try mul(shape.primary_rows, shape.dim) or
        input.future_queries.len != try mul(shape.future_rows, shape.dim) or
        input.historical_queries.len != try mul(shape.historical_rows, shape.dim) or
        input.historical_positions.len != shape.historical_rows or
        input.heldout_positions.len != shape.heldout_rows or
        input.source_key_weights.len != shape.source_rows or
        input.historical_candidate_offsets.len != shape.historical_rows + 1)
        return error.NativeCoverageShapeUnsupported;
}

pub fn select(
    allocator: std.mem.Allocator,
    shape: Shape,
    inputs: []const ConsumerInput,
    external_live_bytes: usize,
    combined_cap: usize,
    control: solver.Control,
) !Owned {
    try shape.validate();
    try control.check();
    if (inputs.len != shape.consumer_count or external_live_bytes > combined_cap)
        return error.NativeCoverageShapeUnsupported;
    var selector_peak: usize = 0;
    for (inputs) |input| {
        try validateConsumer(shape, input);
        selector_peak = @max(selector_peak, try selector.workspaceBytes(shape.historical_rows, shape.dim, shape.source_rows, shape.selected_historical_rows));
    }
    const output_bytes = try mul(shape.consumer_count, @sizeOf(ConsumerEvidence) + try mul(2 * shape.selected_historical_rows, @sizeOf(u32)));
    const required = std.math.add(usize, output_bytes, selector_peak) catch return error.CompactionWorkspaceLimit;
    if (required > combined_cap - external_live_bytes) return error.CompactionWorkspaceLimit;

    const consumers = try allocator.alloc(ConsumerEvidence, shape.consumer_count);
    var completed: usize = 0;
    errdefer {
        for (consumers[0..completed]) |entry| {
            allocator.free(entry.coverage_indices);
            allocator.free(entry.selected_indices);
        }
        allocator.free(consumers);
    }
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(policy);
    inline for (.{ "dim", "source_rows", "primary_rows", "historical_rows", "future_rows", "heldout_rows", "selected_historical_rows", "consumer_count", "expected_head_count", "actual_head_count" }) |field|
        hash.update(std.mem.asBytes(&@field(shape, field)));
    hash.update(&.{@backingInt(shape.route)});
    for (inputs, consumers) |input, *out| {
        try control.check();
        var result = try selector.select(allocator, .{
            .queries = input.historical_queries,
            .positions = input.historical_positions,
            .heldout_positions = input.heldout_positions,
            .dim = shape.dim,
            .selected_count = shape.selected_historical_rows,
            .source_key_weights = input.source_key_weights,
            .baseline_covered_keys = input.baseline_covered_keys,
            .candidate_offsets = input.historical_candidate_offsets,
            .candidate_keys = input.historical_candidate_keys,
        }, combined_cap - external_live_bytes - output_bytes, control);
        defer result.deinit();
        const indices = try allocator.dupe(u32, result.indices);
        errdefer allocator.free(indices);
        const coverage_indices = try allocator.dupe(u32, result.coverage_indices);
        out.* = .{ .selected_indices = indices, .coverage_indices = coverage_indices, .selector_evidence = result.evidence };
        completed += 1;
        hash.update(std.mem.sliceAsBytes(indices));
        hash.update(&result.evidence.selection_digest);
    }
    return .{ .allocator = allocator, .consumers = consumers, .digest = hash.finalResult() };
}

test "native bridge uses explicit shape and excludes HHO positions before selection" {
    const shape: Shape = .{ .dim = 1, .source_rows = 2, .primary_rows = 1, .historical_rows = 3, .future_rows = 1, .heldout_rows = 1, .selected_historical_rows = 2, .consumer_count = 1, .expected_head_count = 1, .actual_head_count = 1, .route = .flat };
    const input: ConsumerInput = .{
        .primary_queries = &.{0},
        .future_queries = &.{0},
        .historical_queries = &.{ 1, 2, 3 },
        .historical_positions = &.{ 10, 11, 12 },
        .heldout_positions = &.{11},
        .source_key_weights = &.{ 1, 0.5 },
        .baseline_covered_keys = &.{},
        .historical_candidate_offsets = &.{ 0, 1, 2, 3 },
        .historical_candidate_keys = &.{ 0, 0, 1 },
    };
    var owned = try select(std.testing.allocator, shape, &.{input}, 0, 1 << 20, .{});
    defer owned.deinit();
    try std.testing.expectEqual(@as(usize, 2), owned.consumers[0].selected_indices.len);
    for (owned.consumers[0].selected_indices) |index| try std.testing.expect(index != 1);
}

test "native bridge typed rejects unsupported topology and bounded route" {
    var shape: Shape = .{ .dim = 1, .source_rows = 2, .primary_rows = 1, .historical_rows = 3, .future_rows = 1, .heldout_rows = 1, .selected_historical_rows = 2, .consumer_count = 1, .expected_head_count = 2, .actual_head_count = 1, .route = .flat };
    try std.testing.expectError(error.NativeCoverageShapeUnsupported, shape.validate());
    shape.actual_head_count = 2;
    shape.route = .bounded;
    try std.testing.expectError(error.NativeCoverageBoundedUnsupported, shape.validate());
}

test "native bridge rejects cap before output allocation and unwinds failures" {
    const shape: Shape = .{ .dim = 1, .source_rows = 2, .primary_rows = 1, .historical_rows = 3, .future_rows = 1, .heldout_rows = 1, .selected_historical_rows = 2, .consumer_count = 1, .expected_head_count = 1, .actual_head_count = 1, .route = .flat };
    const input: ConsumerInput = .{ .primary_queries = &.{0}, .future_queries = &.{0}, .historical_queries = &.{ 1, 2, 3 }, .historical_positions = &.{ 10, 11, 12 }, .heldout_positions = &.{11}, .source_key_weights = &.{ 1, 0.5 }, .baseline_covered_keys = &.{}, .historical_candidate_offsets = &.{ 0, 1, 2, 3 }, .historical_candidate_keys = &.{ 0, 0, 1 } };
    try std.testing.expectError(error.CompactionWorkspaceLimit, select(std.testing.failing_allocator, shape, &.{input}, 0, 1, .{}));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn run(a: std.mem.Allocator, s: Shape, i: ConsumerInput) !void {
            var owned = try select(a, s, &.{i}, 0, 1 << 20, .{});
            owned.deinit();
        }
    }.run, .{ shape, input });
}
