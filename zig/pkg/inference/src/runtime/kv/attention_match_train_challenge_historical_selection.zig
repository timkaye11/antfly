const std = @import("std");
const solver = @import("attention_match.zig");
const anchors = @import("attention_match_train_historical_anchor_authority.zig");
const hybrid = @import("attention_match_train_challenge_hybrid_selection.zig");

pub const policy = "train_challenge_historical_selection_v3|role_exclude_hho_positions_before_query_read_and_digest|score=exact_fp64_raw_query_relative_partition_mass_error|diversity=normalized_query_cosine|select=error_weight_times_one_minus_max_cosine|tie=lowest_original_index|serialize=chronological|authority=eligible_queries_full_positions_role_mask";
pub const maximum_rows: usize = 512;
pub const maximum_dim: usize = 512;

pub const MassBackend = struct {
    context: ?*anyopaque,
    workspace_fn: *const fn (?*anyopaque, usize, usize, usize) anyerror!usize,
    masses_fn: *const fn (?*anyopaque, std.mem.Allocator, solver.Head, []const f64, []f64, solver.Control) anyerror!void,
};

pub const Result = struct {
    allocator: std.mem.Allocator,
    indices: []u32,
    eligible_rows: usize,
    excluded_rows: usize,
    maximum_error: f64,
    mean_error: f64,
    selection_digest: [32]u8,
    hybrid_mandatory_count: usize = 0,
    hybrid_coverage_count: usize = 0,
    hybrid_anchor_count: usize = 0,
    hybrid_challenge_count: usize = 0,
    coverage_authority_sha256: [32]u8 = @splat(0),
    hybrid_authority_sha256: [32]u8 = @splat(0),
    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.indices);
        self.* = undefined;
    }
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}

pub fn workspaceBytes(rows: usize, dim: usize, selected_count: usize, source_rows: usize, pilot_rows: usize, backend: solver.Backend) !usize {
    // raw queries, normalized queries, source targets, and pilot targets are
    // simultaneously live while both summaries are evaluated.
    var bytes = try mul(try mul(try mul(rows, dim), 4), @sizeOf(f64));
    bytes = try add(bytes, try mul(rows, 5 * @sizeOf(f64) + @sizeOf(usize) + @sizeOf(bool)));
    bytes = try add(bytes, try mul(selected_count, @sizeOf(u32)));
    const source_backend = try backend.workspace_fn(backend.context, source_rows, 1, rows, dim);
    const pilot_backend = try backend.workspace_fn(backend.context, pilot_rows, 1, rows, dim);
    const backend_bytes = @max(source_backend, pilot_backend);
    return add(bytes, backend_bytes);
}

fn roleExcluded(position: u64, heldout: []const u64) bool {
    return std.mem.indexOfScalar(u64, heldout, position) != null;
}

fn hashU64(hash: *std.crypto.hash.sha2.Sha256, value: u64) void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, value, .little);
    hash.update(&bytes);
}

fn hashHead(hash: *std.crypto.hash.sha2.Sha256, head: solver.Head) void {
    hashU64(hash, @intCast(head.rows));
    hashU64(hash, @intCast(head.dim));
    hashU64(hash, @bitCast(head.scale));
    hashU64(hash, @bitCast(head.softcap));
    hash.update(std.mem.sliceAsBytes(head.keys));
    hash.update(std.mem.sliceAsBytes(head.values));
    if (head.bias) |bias| {
        hash.update(&.{1});
        hash.update(std.mem.sliceAsBytes(bias));
    } else hash.update(&.{0});
}

fn selectInternal(allocator: std.mem.Allocator, source: solver.Head, pilot: solver.Head, queries_f32: []const f32, positions: []const u64, heldout_positions: []const u64, selected_count: usize, original_h256: []const u32, coverage: []const u32, coverage_authority_sha256: [32]u8, hybrid_enabled: bool, backend: solver.Backend, mass_backend: ?MassBackend, external_live_bytes: usize, combined_cap: usize, control: solver.Control) !Result {
    try control.check();
    try source.validate();
    try pilot.validate();
    if (source.dim != pilot.dim or source.dim == 0 or source.dim > maximum_dim or positions.len == 0 or positions.len > maximum_rows or queries_f32.len != try mul(positions.len, source.dim) or selected_count == 0 or selected_count > positions.len or backend.summary_fn == null) return error.InvalidFitConfig;
    if (heldout_positions.len > 1) for (heldout_positions[1..], heldout_positions[0 .. heldout_positions.len - 1]) |next, previous| if (next <= previous) return error.InvalidFitConfig;
    if (positions.len > 1) for (positions[1..], positions[0 .. positions.len - 1]) |next, previous| if (next <= previous) return error.InvalidFitConfig;
    var eligible_count: usize = 0;
    for (positions) |position| eligible_count += @intFromBool(!roleExcluded(position, heldout_positions));
    if (eligible_count < selected_count) return error.InvalidFitConfig;
    const workspace = try workspaceBytes(eligible_count, source.dim, selected_count, source.rows, pilot.rows, backend);
    var required_workspace = workspace;
    if (mass_backend) |mass| required_workspace = @max(required_workspace, @max(try mass.workspace_fn(mass.context, source.rows, eligible_count, source.dim), try mass.workspace_fn(mass.context, pilot.rows, eligible_count, source.dim)));
    if (hybrid_enabled) {
        required_workspace = try add(required_workspace, try anchors.workspaceBytes(original_h256.len, @max(coverage.len, anchors.mandatory_floor)));
        required_workspace = try add(required_workspace, try hybrid.workspaceBytes(eligible_count, selected_count));
    }
    if (external_live_bytes > combined_cap or required_workspace > combined_cap - external_live_bytes) return error.CompactionWorkspaceLimit;

    const raw_queries = try allocator.alloc(f64, try mul(eligible_count, source.dim));
    defer allocator.free(raw_queries);
    const normalized_queries = try allocator.alloc(f64, raw_queries.len);
    defer allocator.free(normalized_queries);
    const originals = try allocator.alloc(usize, eligible_count);
    defer allocator.free(originals);
    var at: usize = 0;
    for (positions, 0..) |position, row| if (!roleExcluded(position, heldout_positions)) {
        originals[at] = row;
        var norm2: f64 = 0;
        for (0..source.dim) |di| {
            const value: f64 = queries_f32[row * source.dim + di];
            raw_queries[at * source.dim + di] = value;
            normalized_queries[at * source.dim + di] = value;
            norm2 += value * value;
        }
        if (!std.math.isFinite(norm2) or norm2 <= 0) return error.NonFiniteAttention;
        const inverse = 1.0 / @sqrt(norm2);
        for (normalized_queries[at * source.dim ..][0..source.dim]) |*value| value.* *= inverse;
        at += 1;
    };

    const source_mass = try allocator.alloc(f64, eligible_count);
    defer allocator.free(source_mass);
    const pilot_mass = try allocator.alloc(f64, eligible_count);
    defer allocator.free(pilot_mass);
    if (mass_backend) |mass| {
        try mass.masses_fn(mass.context, allocator, source, raw_queries, source_mass, control);
        try mass.masses_fn(mass.context, allocator, pilot, raw_queries, pilot_mass, control);
    } else {
        const source_targets = try allocator.alloc(f64, try mul(eligible_count, source.dim));
        defer allocator.free(source_targets);
        const pilot_targets = try allocator.alloc(f64, source_targets.len);
        defer allocator.free(pilot_targets);
        try backend.summary_fn.?(backend.context, allocator, source, raw_queries, source_targets, source_mass, control);
        try backend.summary_fn.?(backend.context, allocator, pilot, raw_queries, pilot_targets, pilot_mass, control);
    }
    const errors = try allocator.alloc(f64, eligible_count);
    defer allocator.free(errors);
    var maximum_error: f64 = 0;
    var sum_error: f64 = 0;
    for (source_mass, pilot_mass, errors) |reference, candidate, *out| {
        out.* = @abs(std.math.exp(candidate - reference) - 1);
        if (!std.math.isFinite(out.*)) return error.NonFiniteAttention;
        maximum_error = @max(maximum_error, out.*);
        sum_error += out.*;
    }
    const denom = @max(maximum_error, std.math.floatEps(f64));
    if (hybrid_enabled) {
        var query_hash = std.crypto.hash.sha2.Sha256.init(.{});
        query_hash.update("antfly.role-eligible-normalized-train-query-authority.v1\x00");
        for (originals) |value| {
            var bytes: [8]u8 = undefined;
            std.mem.writeInt(u64, &bytes, @intCast(value), .little);
            query_hash.update(&bytes);
        }
        for (normalized_queries) |value| {
            var bytes: [8]u8 = undefined;
            std.mem.writeInt(u64, &bytes, @bitCast(value), .little);
            query_hash.update(&bytes);
        }
        var mandatory = try anchors.buildEligible(allocator, originals, normalized_queries, source.dim, original_h256, coverage, query_hash.finalResult(), combined_cap - external_live_bytes, control);
        defer mandatory.deinit();
        var selected = try hybrid.select(allocator, originals, normalized_queries, errors, source.dim, mandatory.indices, selected_count, combined_cap - external_live_bytes, control);
        var authority = std.crypto.hash.sha2.Sha256.init(.{});
        authority.update("antfly.train-challenge-hybrid-result.v1\x00");
        authority.update(&coverage_authority_sha256);
        authority.update(&mandatory.digest);
        authority.update(&selected.digest);
        const digest = authority.finalResult();
        const indices = selected.indices;
        selected = undefined;
        return .{ .allocator = allocator, .indices = indices, .eligible_rows = eligible_count, .excluded_rows = positions.len - eligible_count, .maximum_error = maximum_error, .mean_error = sum_error / @as(f64, @floatFromInt(eligible_count)), .selection_digest = digest, .hybrid_mandatory_count = mandatory.indices.len, .hybrid_coverage_count = coverage.len, .hybrid_anchor_count = mandatory.indices.len - coverage.len, .hybrid_challenge_count = selected_count - mandatory.indices.len, .coverage_authority_sha256 = coverage_authority_sha256, .hybrid_authority_sha256 = digest };
    }
    const nearest = try allocator.alloc(f64, eligible_count);
    defer allocator.free(nearest);
    @memset(nearest, 0);
    const chosen = try allocator.alloc(bool, eligible_count);
    defer allocator.free(chosen);
    @memset(chosen, false);
    const output = try allocator.alloc(u32, selected_count);
    errdefer allocator.free(output);
    for (0..selected_count) |selection_i| {
        try control.check();
        var best: ?usize = null;
        var best_score: f64 = -1;
        for (0..eligible_count) |row| if (!chosen[row]) {
            const score = (errors[row] / denom) * (1 - nearest[row]);
            if (best == null or score > best_score or (score == best_score and originals[row] < originals[best.?])) {
                best = row;
                best_score = score;
            }
        };
        const picked = best orelse return error.InvalidFitConfig;
        chosen[picked] = true;
        output[selection_i] = @intCast(originals[picked]);
        for (0..eligible_count) |row| if (!chosen[row]) {
            var cosine: f64 = 0;
            for (0..source.dim) |di| cosine += normalized_queries[picked * source.dim + di] * normalized_queries[row * source.dim + di];
            nearest[row] = @max(nearest[row], std.math.clamp(cosine, 0, 1));
        };
    }
    std.mem.sort(u32, output, {}, std.sort.asc(u32));
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(policy);
    hashHead(&hash, source);
    hashHead(&hash, pilot);
    // Bind only role-eligible query values. The complete positions and role
    // mask below still bind which rows were excluded without reading HHO data.
    hash.update(std.mem.sliceAsBytes(raw_queries));
    hash.update(std.mem.sliceAsBytes(output));
    hash.update(std.mem.sliceAsBytes(positions));
    hash.update(std.mem.sliceAsBytes(heldout_positions));
    hashU64(&hash, @intCast(selected_count));
    for (positions) |position| hash.update(&.{@intFromBool(roleExcluded(position, heldout_positions))});
    return .{ .allocator = allocator, .indices = output, .eligible_rows = eligible_count, .excluded_rows = positions.len - eligible_count, .maximum_error = maximum_error, .mean_error = sum_error / @as(f64, @floatFromInt(eligible_count)), .selection_digest = hash.finalResult() };
}

pub fn select(allocator: std.mem.Allocator, source: solver.Head, pilot: solver.Head, queries_f32: []const f32, positions: []const u64, heldout_positions: []const u64, selected_count: usize, backend: solver.Backend, external_live_bytes: usize, combined_cap: usize, control: solver.Control) !Result {
    return selectInternal(allocator, source, pilot, queries_f32, positions, heldout_positions, selected_count, &.{}, &.{}, @splat(0), false, backend, null, external_live_bytes, combined_cap, control);
}

pub fn selectHybrid(allocator: std.mem.Allocator, source: solver.Head, pilot: solver.Head, queries_f32: []const f32, positions: []const u64, heldout_positions: []const u64, selected_count: usize, original_h256: []const u32, coverage: []const u32, coverage_authority_sha256: [32]u8, backend: solver.Backend, external_live_bytes: usize, combined_cap: usize, control: solver.Control) !Result {
    return selectInternal(allocator, source, pilot, queries_f32, positions, heldout_positions, selected_count, original_h256, coverage, coverage_authority_sha256, true, backend, null, external_live_bytes, combined_cap, control);
}

pub fn selectHybridMass(allocator: std.mem.Allocator, source: solver.Head, pilot: solver.Head, queries_f32: []const f32, positions: []const u64, heldout_positions: []const u64, selected_count: usize, original_h256: []const u32, coverage: []const u32, coverage_authority_sha256: [32]u8, backend: solver.Backend, mass_backend: MassBackend, external_live_bytes: usize, combined_cap: usize, control: solver.Control) !Result {
    return selectInternal(allocator, source, pilot, queries_f32, positions, heldout_positions, selected_count, original_h256, coverage, coverage_authority_sha256, true, backend, mass_backend, external_live_bytes, combined_cap, control);
}

const TestScalarSummary = struct {
    fn workspace(_: ?*anyopaque, head_rows: usize, _: usize, query_rows: usize, _: usize) !usize {
        return mul(try mul(head_rows, query_rows), @sizeOf(f64));
    }

    fn run(_: ?*anyopaque, _: std.mem.Allocator, head: solver.Head, queries: []const f64, targets: []f64, masses: []f64, control: solver.Control) !void {
        const query_rows = masses.len;
        if (queries.len != try mul(query_rows, head.dim) or targets.len != try mul(query_rows, head.dim)) return error.InvalidFitConfig;
        @memset(targets, 0);
        for (0..query_rows) |qi| {
            try control.check();
            var maximum = -std.math.inf(f64);
            for (0..head.rows) |ki| {
                var logit: f64 = 0;
                for (0..head.dim) |di| logit += queries[qi * head.dim + di] * head.keys[ki * head.dim + di];
                logit *= head.scale;
                if (head.softcap > 0) logit = head.softcap * std.math.tanh(logit / head.softcap);
                if (head.bias) |bias| logit += bias[ki];
                maximum = @max(maximum, logit);
            }
            var total: f64 = 0;
            for (0..head.rows) |ki| {
                var logit: f64 = 0;
                for (0..head.dim) |di| logit += queries[qi * head.dim + di] * head.keys[ki * head.dim + di];
                logit *= head.scale;
                if (head.softcap > 0) logit = head.softcap * std.math.tanh(logit / head.softcap);
                if (head.bias) |bias| logit += bias[ki];
                total += @exp(logit - maximum);
            }
            if (!std.math.isFinite(total) or total <= 0) return error.NonFiniteAttention;
            masses[qi] = maximum + @log(total);
        }
    }
};

fn testScalarBackend() solver.Backend {
    return .{ .workspace_fn = TestScalarSummary.workspace, .summary_fn = TestScalarSummary.run };
}

test "high TRAIN challenge is retained while HHO role is excluded" {
    const source: solver.Head = .{ .keys = &.{ 2, -2 }, .values = &.{ 0, 0 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    const pilot: solver.Head = .{ .keys = &.{2}, .values = &.{0}, .bias = &.{0}, .rows = 1, .dim = 1, .scale = 1 };
    const backend = testScalarBackend();
    var result = try select(std.testing.allocator, source, pilot, &.{ 1, -1, -1, 1 }, &.{ 10, 11, 12, 13 }, &.{11}, 2, backend, 0, 1 << 20, .{});
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 1), result.excluded_rows);
    try std.testing.expect(std.mem.indexOfScalar(u32, result.indices, 1) == null);
    try std.testing.expect(std.mem.indexOfScalar(u32, result.indices, 2) != null);
}

test "excluded HHO query bytes are neither read nor hashed" {
    const source: solver.Head = .{ .keys = &.{ 2, -2 }, .values = &.{ 0, 0 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    const pilot: solver.Head = .{ .keys = &.{2}, .values = &.{0}, .bias = &.{0}, .rows = 1, .dim = 1, .scale = 1 };
    const backend = testScalarBackend();
    var first = try select(std.testing.allocator, source, pilot, &.{ 1, 17, -1 }, &.{ 10, 11, 12 }, &.{11}, 1, backend, 0, 1 << 20, .{});
    defer first.deinit();
    var second = try select(std.testing.allocator, source, pilot, &.{ 1, std.math.nan(f32), -1 }, &.{ 10, 11, 12 }, &.{11}, 1, backend, 0, 1 << 20, .{});
    defer second.deinit();
    try std.testing.expectEqualSlices(u32, first.indices, second.indices);
    try std.testing.expectEqualSlices(u8, &first.selection_digest, &second.selection_digest);
}

test "selector rejects cap before allocation and cancellation unwinds" {
    const Cancel = struct {
        calls: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            if (self.calls == 3) return error.Cancelled;
        }
    };
    const head: solver.Head = .{ .keys = &.{ 1, -1 }, .values = &.{ 0, 0 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    const backend = testScalarBackend();
    try std.testing.expectError(error.CompactionWorkspaceLimit, select(std.testing.failing_allocator, head, head, &.{ 1, -1 }, &.{ 1, 2 }, &.{}, 1, backend, 0, 1, .{}));
    var cancel: Cancel = .{};
    try std.testing.expectError(error.Cancelled, select(std.testing.allocator, head, head, &.{ 1, -1 }, &.{ 1, 2 }, &.{}, 1, backend, 0, 1 << 20, .{ .context = &cancel, .check_fn = Cancel.check }));
}

test "exact mass uses raw queries and allocation failures publish nothing" {
    const Summary = struct {
        calls: usize = 0,
        fn workspace(_: ?*anyopaque, head_rows: usize, _: usize, _: usize, _: usize) !usize {
            return head_rows;
        }
        fn run(raw: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, queries: []const f64, targets: []f64, masses: []f64, _: solver.Control) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            try std.testing.expectEqualSlices(f64, &.{ 3, 4, 6, 8 }, queries);
            @memset(targets, 0);
            @memset(masses, 0);
            self.calls += 1;
        }
        fn allocationRun(allocator: std.mem.Allocator) !void {
            var state: @This() = .{};
            const backend: solver.Backend = .{ .context = &state, .workspace_fn = workspace, .summary_fn = run };
            const head: solver.Head = .{ .keys = &.{ 1, 0, 0, 1 }, .values = &.{ 0, 0, 0, 0 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 2, .scale = 1 };
            var result = try select(allocator, head, head, &.{ 3, 4, 6, 8 }, &.{ 1, 2 }, &.{}, 1, backend, 0, 1 << 20, .{});
            result.deinit();
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, Summary.allocationRun, .{});
}
