const std = @import("std");
const platform = @import("antfly_platform");
const checkpoint = @import("attention_match_checkpoint.zig");
const solver = @import("attention_match.zig");
const fixture_mod = @import("attention_match_rejected_fixture.zig");
const replay = @import("attention_match_joint_chunk_mass_replay.zig");
const admission = @import("attention_match_cumulative_cache.zig");
const bounded = @import("attention_match_joint_bounded_run.zig");
const gram_delta = @import("attention_match_gram_delta_accelerate.zig");

pub const experiment_schema = "antfly.p91-production-geometry-train-snapshot-plan.v1";
pub const policy = "production_incremental_gram_p91_all960_train_snapshot_v1";
pub const policy_sha256 = "cc18c98c95eae82521b10e5525ce52a6715538353724453052f509842a1cae66";
const combined_cap: usize = 1024 * 1024 * 1024;
const source_rows: usize = 7936;
const dim: usize = 512;
const fixed_rows: usize = 128 + 2048;
const middle_rows: usize = 5760;
const expected_middle_target: usize = (middle_rows / 2) * 91 / 100;
const expected_candidate_rows: usize = fixed_rows + expected_middle_target;

const Consumer = struct { layer: u32, query_head: u32 };
const Shard = struct { fixture_dir: []const u8, fixture_manifest_sha256: []const u8, fixture_provenance_sha256: []const u8, consumers: []const Consumer };
const Plan = struct {
    schema: []const u8,
    result_path: []const u8,
    snapshot_output_dir: []const u8,
    fixture_shards: []const Shard,
    identity: struct { weights: []const u8, tokenizer: []const u8, template: []const u8, layout: []const u8 },
    config_sha256: []const u8,
    donor_layer: u32,
    kv_head: usize,
    geometry_budget_percent: u8,
    fallback_geometry_budget_percent: u8,
    prospective_assignment_unresolved: bool,
    original_head_sha256: []const u8,
    source_freeze_sha256: []const u8,
    worker_binary_sha256: []const u8,
    producer_input_manifest_sha256: []const u8,
    complete_future_mix_manifest_sha256: []const u8,
    complete_future_mix_policy_sha256: []const u8,
};
const Provenance = struct {
    schema: []const u8,
    source_freeze_sha256: []const u8,
    binary_sha256: []const u8,
    config_sha256: []const u8,
    all_eligible_policy_sha256: []const u8,
    fixture_manifest_sha256: []const u8,
    identity: struct { weights: []const u8, tokenizer: []const u8, template: []const u8, layout: []const u8 },
    origins: struct { complete: struct { sha256: []const u8 }, parent: struct { sha256: []const u8 } },
};

fn validateChosenMiddleRows(cap: usize, chosen: usize, sum: usize) !void {
    if (cap != expected_middle_target or chosen != sum or sum > cap) return error.InvalidP91SnapshotPlan;
}

fn digest(value: []const u8) ![32]u8 {
    if (value.len != 64) return error.InvalidP91SnapshotPlan;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, value) catch return error.InvalidP91SnapshotPlan;
    return out;
}
fn verifyFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8, expected: []const u8) !void {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(2 * 1024 * 1024));
    defer allocator.free(bytes);
    var actual: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &actual, .{});
    if (!std.mem.eql(u8, &actual, &(try digest(expected)))) return error.InvalidP91SnapshotPlan;
}
fn sameOriginal(a: *const fixture_mod.Fixture, b: *const fixture_mod.Fixture) bool {
    return std.meta.eql(a.geometry, b.geometry) and std.mem.eql(f64, a.original_keys, b.original_keys) and std.mem.eql(f64, a.original_values, b.original_values) and std.mem.eql(f64, a.original_bias, b.original_bias) and std.mem.eql(u64, a.original_positions, b.original_positions) and std.mem.eql(u64, a.original_coverage, b.original_coverage);
}
fn originalDigest(fixture: *const fixture_mod.Fixture) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("antfly.p91-original-head.v1\x00");
    h.update(std.mem.sliceAsBytes(fixture.original_keys));
    h.update(std.mem.sliceAsBytes(fixture.original_values));
    h.update(std.mem.sliceAsBytes(fixture.original_bias));
    h.update(std.mem.sliceAsBytes(fixture.original_positions));
    h.update(std.mem.sliceAsBytes(fixture.original_coverage));
    return h.finalResult();
}
fn fillSelectedSourceKeys(range: admission.Range, level: @import("attention_match_joint_adaptive_geometry.zig").Level, local: ?[]const usize, out: []usize) !void {
    if (level == .identity) {
        if (local != null or out.len != range.rows) return error.InvalidP91SnapshotPlan;
        for (out, 0..) |*value, key| value.* = range.start + key;
        return;
    }
    const selected = local orelse return error.InvalidP91SnapshotPlan;
    if (selected.len != out.len) return error.InvalidP91SnapshotPlan;
    for (selected, out) |key, *value| {
        if (key >= range.rows) return error.InvalidP91SnapshotPlan;
        value.* = range.start + key;
    }
}
const SnapshotContext = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    output_dir: []const u8,
    original_head_sha256: []const u8,
    config_sha256: []const u8,
    model_sha256: []const u8,
    source_freeze_sha256: []const u8,
    worker_binary_sha256: []const u8,
    producer_input_manifest_sha256: []const u8,
    complete_future_mix_manifest_sha256: []const u8,
    complete_future_mix_policy_sha256: []const u8,
    runs: []const replay.TrainSnapshotDomainRun,
    external_fixture_bytes: usize,
    external_train_bytes: usize,
    external_metadata_bytes: usize,
    callback_output_reserve_bytes: usize,
    manifest: ?[32]u8 = null,
    fn write(raw: ?*anyopaque, view: bounded.GeometryFreezeView, control: solver.Control) !void {
        const self: *SnapshotContext = @ptrCast(@alignCast(raw orelse return error.InvalidP91SnapshotPlan));
        try control.check();
        if (view.geometry_budget_percent != 91 or view.train_queries == 0 or view.combined_cap != combined_cap or view.route_external_bytes > combined_cap or view.scoring_workspace_bound_bytes > combined_cap or view.geometry_workspace_bound_bytes > combined_cap or view.ranges.len != 23 or view.adaptive_plan.branches[2].targets.len != 23) return error.InvalidP91SnapshotPlan;
        const branch = view.adaptive_plan.branches[2];
        var target_total: usize = 0;
        for (branch.targets) |target| target_total = try std.math.add(usize, target_total, target);
        try validateChosenMiddleRows(branch.middle_cap_rows, branch.middle_rows, target_total);
        var domain_total: usize = 0;
        for (self.runs) |domain_run| domain_total = try std.math.add(usize, domain_total, domain_run.count);
        if (domain_total != view.train_queries) return error.InvalidP91SnapshotPlan;
        var selected: [23][]usize = undefined;
        var allocated: usize = 0;
        defer for (selected[0..allocated]) |keys| self.allocator.free(keys);
        for (view.ranges, branch.levels, 0..) |range, level, i| {
            const count = branch.targets[i];
            selected[i] = try self.allocator.alloc(usize, count);
            allocated += 1;
            if (level == .identity) {
                try fillSelectedSourceKeys(range, level, null, selected[i]);
            } else {
                const candidate = view.adaptive_plan.scored[i][@backingInt(level)] orelse return error.InvalidP91SnapshotPlan;
                const local = candidate.frozen.chunks[0].selected_local;
                try fillSelectedSourceKeys(range, level, local, selected[i]);
            }
        }
        var positions_hash: [32]u8 = undefined;
        var coverage_hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(view.source_positions), &positions_hash, .{});
        std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(view.source_coverage), &coverage_hash, .{});
        const body = try std.json.Stringify.valueAlloc(self.allocator, .{ .schema = bounded.geometry_snapshot_schema, .training_only = true, .heldout_present = false, .original_head_sha256 = self.original_head_sha256, .config_sha256 = self.config_sha256, .model_sha256 = self.model_sha256, .source_freeze_sha256 = self.source_freeze_sha256, .worker_binary_sha256 = self.worker_binary_sha256, .producer_input_manifest_sha256 = self.producer_input_manifest_sha256, .complete_future_mix_manifest_sha256 = self.complete_future_mix_manifest_sha256, .complete_future_mix_policy_sha256 = self.complete_future_mix_policy_sha256, .geometry_policy = view.adaptive_plan.geometry_policy, .geometry_budget_percent = view.geometry_budget_percent, .fallback_geometry_budget_percent = 0, .train_queries = view.train_queries, .domain_runs = self.runs, .source_positions_sha256 = std.fmt.bytesToHex(positions_hash, .lower), .source_coverage_sha256 = std.fmt.bytesToHex(coverage_hash, .lower), .geometry_digest = std.fmt.bytesToHex(branch.digest, .lower), .geometry_targets = branch.targets, .selected_source_keys = selected, .middle_cap_rows = branch.middle_cap_rows, .candidate_rows = fixed_rows + branch.middle_rows, .memory_ledger = .{ .fixture_bytes = self.external_fixture_bytes, .train_bytes = self.external_train_bytes, .metadata_bytes = self.external_metadata_bytes, .callback_output_reserve_bytes = self.callback_output_reserve_bytes, .route_external_bytes = view.route_external_bytes, .scoring_workspace_bound_bytes = view.scoring_workspace_bound_bytes, .geometry_workspace_bound_bytes = view.geometry_workspace_bound_bytes, .combined_cap = view.combined_cap }, .prospective_assignment_unresolved = true, .diagnostic_only = true, .quality_credit = false, .production_qualified = false }, .{});
        defer self.allocator.free(body);
        var selected_bytes: usize = 0;
        for (selected) |keys| selected_bytes = try std.math.add(usize, selected_bytes, try std.math.mul(usize, keys.len, @sizeOf(usize)));
        if (try std.math.add(usize, body.len + 1, selected_bytes) > self.callback_output_reserve_bytes) return error.CompactionWorkspaceLimit;
        var dir = try std.Io.Dir.openDirAbsolute(self.io, self.output_dir, .{ .follow_symlinks = false });
        defer dir.close(self.io);
        var file = try dir.createFile(self.io, "manifest.json", .{ .exclusive = true, .truncate = false });
        defer file.close(self.io);
        try file.writeStreamingAll(self.io, body);
        try file.writeStreamingAll(self.io, "\n");
        try file.sync(self.io);
        var hash: [32]u8 = undefined;
        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        hasher.update(body);
        hasher.update("\n");
        hash = hasher.finalResult();
        self.manifest = hash;
    }
};

fn run(plan_path: []const u8) !void {
    if (!std.fs.path.isAbsolute(plan_path)) return error.InvalidP91SnapshotPlan;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, plan_path, allocator, .limited(2 * 1024 * 1024));
    defer allocator.free(bytes);
    var plan_hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &plan_hash, .{});
    const parsed = try std.json.parseFromSlice(Plan, allocator, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false });
    defer parsed.deinit();
    const p = parsed.value;
    if (!std.mem.eql(u8, p.schema, experiment_schema) or !std.fs.path.isAbsolute(p.result_path) or !std.fs.path.isAbsolute(p.snapshot_output_dir) or p.geometry_budget_percent != 91 or p.fallback_geometry_budget_percent != 0 or !p.prospective_assignment_unresolved or p.fixture_shards.len == 0 or p.fixture_shards.len > 4 or (p.donor_layer != 5 and p.donor_layer != 11 and p.donor_layer != 17 and p.donor_layer != 23) or p.kv_head > 1) return error.InvalidP91SnapshotPlan;
    const identity: checkpoint.Identity = .{ .weights = try digest(p.identity.weights), .tokenizer = try digest(p.identity.tokenizer), .template = try digest(p.identity.template), .layout = try digest(p.identity.layout) };
    const config = try digest(p.config_sha256);
    inline for (.{ p.original_head_sha256, p.source_freeze_sha256, p.worker_binary_sha256, p.producer_input_manifest_sha256, p.complete_future_mix_manifest_sha256, p.complete_future_mix_policy_sha256 }) |authority| _ = try digest(authority);
    const expected_consumers: usize = if (p.donor_layer == 23) 16 else 4;
    const train = try allocator.alloc(f64, expected_consumers * 960 * dim);
    defer allocator.free(train);
    var train_at: usize = 0;
    var first: ?fixture_mod.Fixture = null;
    defer if (first) |*value| value.deinit();
    var runs = try std.ArrayList(replay.TrainSnapshotDomainRun).initCapacity(allocator, expected_consumers * 3);
    defer runs.deinit(allocator);
    var seen: usize = 0;
    var consumer_keys: [16]u64 = undefined;
    for (p.fixture_shards) |shard| {
        const sidecar = try std.fs.path.join(allocator, &.{ shard.fixture_dir, "provenance.json" });
        defer allocator.free(sidecar);
        try verifyFile(allocator, io, sidecar, shard.fixture_provenance_sha256);
        const provenance_bytes = try std.Io.Dir.cwd().readFileAlloc(io, sidecar, allocator, .limited(2 * 1024 * 1024));
        defer allocator.free(provenance_bytes);
        const provenance_parsed = try std.json.parseFromSlice(Provenance, allocator, provenance_bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = true });
        defer provenance_parsed.deinit();
        const provenance = provenance_parsed.value;
        if (!std.mem.eql(u8, provenance.schema, "antfly.all-head-fixture-shard-provenance.v1") or !std.mem.eql(u8, provenance.source_freeze_sha256, p.source_freeze_sha256) or !std.mem.eql(u8, provenance.binary_sha256, p.worker_binary_sha256) or !std.mem.eql(u8, provenance.config_sha256, p.config_sha256) or !std.mem.eql(u8, provenance.fixture_manifest_sha256, shard.fixture_manifest_sha256) or !std.mem.eql(u8, provenance.all_eligible_policy_sha256, p.complete_future_mix_policy_sha256) or !std.mem.eql(u8, provenance.origins.complete.sha256, p.complete_future_mix_manifest_sha256) or !std.mem.eql(u8, provenance.origins.parent.sha256, p.producer_input_manifest_sha256) or !std.mem.eql(u8, provenance.identity.weights, p.identity.weights) or !std.mem.eql(u8, provenance.identity.tokenizer, p.identity.tokenizer) or !std.mem.eql(u8, provenance.identity.template, p.identity.template) or !std.mem.eql(u8, provenance.identity.layout, p.identity.layout)) return error.InvalidP91SnapshotPlan;
        var fixture = try fixture_mod.readTrainingOnly(allocator, io, shard.fixture_dir, .{ .output_dir = shard.fixture_dir, .identity = identity, .config_sha256 = config });
        var moved = false;
        defer if (!moved) fixture.deinit();
        var manifest_hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(fixture.manifest_bytes, &manifest_hash, .{});
        if (!std.mem.eql(u8, &manifest_hash, &(try digest(shard.fixture_manifest_sha256))) or fixture.geometry.donor_layer != p.donor_layer or fixture.geometry.kv_head != p.kv_head or fixture.geometry.original_rows != source_rows or fixture.geometry.head_dim != dim or fixture.query_consumer_layers.len != shard.consumers.len or fixture.query_train_counts.len != shard.consumers.len) return error.InvalidP91SnapshotPlan;
        if (first) |*base| {
            if (!sameOriginal(base, &fixture)) return error.InvalidP91SnapshotPlan;
        } else {
            first = fixture;
            moved = true;
        }
        for (shard.consumers, 0..) |consumer, i| {
            if (fixture.query_consumer_layers[i] != consumer.layer or fixture.query_heads[i] != consumer.query_head or fixture.query_train_counts[i] != 960 or consumer.query_head / 4 != p.kv_head) return error.InvalidP91SnapshotPlan;
            const key = (@as(u64, consumer.layer) << 32) | consumer.query_head;
            if (seen == consumer_keys.len or (seen != 0 and key <= consumer_keys[seen - 1])) return error.InvalidP91SnapshotPlan;
            consumer_keys[seen] = key;
            const start = seen * 960;
            try runs.append(allocator, .{ .domain = .primary, .start = start, .count = 256 });
            try runs.append(allocator, .{ .domain = .historical, .start = start + 256, .count = 448 });
            try runs.append(allocator, .{ .domain = .future, .start = start + 704, .count = 256 });
            seen += 1;
        }
        @memcpy(train[train_at..][0..fixture.train.len], fixture.train);
        train_at += fixture.train.len;
    }
    if (seen != expected_consumers or train_at != train.len or first == null) return error.InvalidP91SnapshotPlan;
    var expected_index: usize = 0;
    const layers = [_]u32{ p.donor_layer, 29, 35, 41 };
    const layer_count: usize = if (p.donor_layer == 23) 4 else 1;
    for (layers[0..layer_count]) |layer| for (0..4) |offset| {
        const expected_key = (@as(u64, layer) << 32) | @as(u64, @intCast(p.kv_head * 4 + offset));
        if (consumer_keys[expected_index] != expected_key) return error.InvalidP91SnapshotPlan;
        expected_index += 1;
    };
    const actual_original_digest = originalDigest(&first.?);
    const expected_original_digest = try digest(p.original_head_sha256);
    if (!std.mem.eql(u8, &actual_original_digest, &expected_original_digest)) return error.InvalidP91SnapshotPlan;
    var ranges: [23]admission.Range = undefined;
    for (&ranges, 0..) |*range, i| range.* = .{ .start = 128 + i * 256, .rows = @min(@as(usize, 256), 5888 - (128 + i * 256)) };
    const Fitter = @import("../../ops/metal_attention_match_fit.zig").Fitter;
    var fitter = try Fitter.init(allocator, null, combined_cap);
    defer fitter.deinit();
    try fitter.setQueryBatchRows(256);
    const fixture_bytes: usize = 128 * 1024 * 1024;
    const train_bytes = try std.math.mul(usize, train.len, @sizeOf(f64));
    const metadata_bytes: usize = 2 * 1024 * 1024;
    const callback_output_reserve_bytes: usize = 4 * 1024 * 1024;
    var external = try std.math.add(usize, fixture_bytes, train_bytes);
    external = try std.math.add(usize, external, metadata_bytes);
    external = try std.math.add(usize, external, callback_output_reserve_bytes);
    if (external > combined_cap) return error.CompactionWorkspaceLimit;
    var context: SnapshotContext = .{ .allocator = allocator, .io = io, .output_dir = p.snapshot_output_dir, .original_head_sha256 = p.original_head_sha256, .config_sha256 = p.config_sha256, .model_sha256 = p.identity.weights, .source_freeze_sha256 = p.source_freeze_sha256, .worker_binary_sha256 = p.worker_binary_sha256, .producer_input_manifest_sha256 = p.producer_input_manifest_sha256, .complete_future_mix_manifest_sha256 = p.complete_future_mix_manifest_sha256, .complete_future_mix_policy_sha256 = p.complete_future_mix_policy_sha256, .runs = runs.items, .external_fixture_bytes = fixture_bytes, .external_train_bytes = train_bytes, .external_metadata_bytes = metadata_bytes, .callback_output_reserve_bytes = callback_output_reserve_bytes };
    var placeholder: [dim]f64 = @splat(0);
    const original = first.?.originalHead();
    const transient = try bounded.streamedGramBuilderTransientBytes(original, train.len / dim, 256, 64, .{ .target_rows = expected_candidate_rows, .backend = fitter.backend(), .max_workspace_bytes = combined_cap });
    _ = bounded.runPreparedViewBoundedTimedAcceleratedBudgetGeometryHook(allocator, .{ .source = original, .train = train, .heldout = &placeholder, .ranges = &ranges, .cfg = .{ .target_rows = expected_candidate_rows, .backend = fitter.backend(), .max_workspace_bytes = combined_cap }, .external_live_bytes = external, .combined_cap = combined_cap }, &fitter, 64, transient, 91, null, .{ .backend = gram_delta.accelerateBackend() orelse return error.AcceleratedDeltaUnsupported }, .{ .context = &context, .source_positions = first.?.original_positions, .source_coverage = first.?.original_coverage, .invoke_fn = SnapshotContext.write }) catch |err| if (err != error.TrainSnapshotExported) return err;
    const snapshot_hash = context.manifest orelse return error.InvalidP91SnapshotPlan;
    const body = try std.json.Stringify.valueAlloc(allocator, .{ .schema = "antfly.p91-production-geometry-train-snapshot-result.v1", .plan_sha256 = std.fmt.bytesToHex(plan_hash, .lower), .donor_layer = p.donor_layer, .kv_head = p.kv_head, .consumer_count = seen, .train_rows = train.len / dim, .domain_run_count = runs.items.len, .geometry_budget_percent = 91, .fallback_geometry_budget_percent = 0, .middle_rows = middle_rows, .middle_target_cap = expected_middle_target, .candidate_row_cap = expected_candidate_rows, .candidate_physical_bytes_cap = 9_928_704, .snapshot_manifest_sha256 = std.fmt.bytesToHex(snapshot_hash, .lower), .heldout_payload_opened = false, .heldout_evaluated = false, .prospective_assignment_unresolved = true, .diagnostic_only = true, .quality_credit = false, .production_qualified = false }, .{});
    defer allocator.free(body);
    var out = try std.Io.Dir.createFileAbsolute(io, p.result_path, .{ .exclusive = true, .truncate = false });
    defer out.close(io);
    try out.writeStreamingAll(io, body);
    try out.writeStreamingAll(io, "\n");
    try out.sync(io);
}

test "attention match production p91 TRAIN geometry snapshot diagnostic" {
    const path = platform.env.getenv("ANTFLY_METAL_P91_GEOMETRY_SNAPSHOT_PLAN") orelse return error.SkipZigTest;
    try run(path);
}

test "p91 production cap arithmetic is exact" {
    try std.testing.expectEqual(@as(usize, 2620), expected_middle_target);
    try std.testing.expectEqual(@as(usize, 4796), expected_candidate_rows);
}

test "p91 preserved identity range emits every source key without indexing scored level three" {
    var keys: [4]usize = undefined;
    try fillSelectedSourceKeys(.{ .start = 128, .rows = 4 }, .identity, null, &keys);
    try std.testing.expectEqualSlices(usize, &.{ 128, 129, 130, 131 }, &keys);
    try std.testing.expectError(error.InvalidP91SnapshotPlan, fillSelectedSourceKeys(.{ .start = 128, .rows = 4 }, .identity, &.{0}, &keys));
}

test "p91 snapshot digest parser enforces complete valid encoding" {
    const decoded = try digest("0000000000000000000000000000000000000000000000000000000000000000");
    const expected: [32]u8 = @splat(0);
    try std.testing.expectEqualSlices(u8, &expected, &decoded);
    try std.testing.expectError(error.InvalidP91SnapshotPlan, digest("00"));
    try std.testing.expectError(error.InvalidP91SnapshotPlan, digest("zz00000000000000000000000000000000000000000000000000000000000000"));
}

test "p91 chosen geometry may stop below the unchanged cap" {
    try validateChosenMiddleRows(2620, 2620, 2620);
    try validateChosenMiddleRows(2620, 2616, 2616);
    try std.testing.expectError(error.InvalidP91SnapshotPlan, validateChosenMiddleRows(2621, 2616, 2616));
    try std.testing.expectError(error.InvalidP91SnapshotPlan, validateChosenMiddleRows(2620, 2616, 2615));
    try std.testing.expectError(error.InvalidP91SnapshotPlan, validateChosenMiddleRows(2620, 2621, 2621));
}
