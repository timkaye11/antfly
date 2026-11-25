const std = @import("std");
const solver = @import("attention_match.zig");
const engine = @import("attention_match_engine.zig");
const checkpoint = @import("attention_match_checkpoint.zig");
const fixture_mod = @import("attention_match_rejected_fixture.zig");
const builtin = @import("builtin");
const build_options = @import("build_options");
const platform = @import("antfly_platform");

pub const Cell = struct { divisor: usize, accelerated: bool, accepted: bool, attempted_chunks: usize, completed_chunks: usize, validated_heads: usize, preserved_heads: usize, strict_mass_error: f64, strict_output_rmse: f64 };
pub const Report = struct { captured_mass_error: f64, captured_output_rmse: f64, cells: [6]Cell };

fn toF32(allocator: std.mem.Allocator, values: []const f64) ![]f32 {
    const out = try allocator.alloc(f32, values.len);
    errdefer allocator.free(out);
    for (out, values) |*dst, src| {
        if (!std.math.isFinite(src)) return error.InvalidRejectedHeadFixture;
        dst.* = @floatCast(src);
    }
    return out;
}

/// Refit exact captured source through the common engine for divisors 4/3/2.
/// The optional backend changes fitting only; final authorization remains the
/// common backend-null FP64 whole-head validator in every cell.
pub fn run(allocator: std.mem.Allocator, fixture: *const fixture_mod.Fixture, identity: checkpoint.Identity, base: engine.Options, accelerated_backend: solver.Backend) !Report {
    const original = fixture.originalHead();
    const rejected = fixture.candidateHead();
    if (fixture.geometry.retention_divisor != 2 or original.rows != fixture.geometry.original_rows or rejected.rows != fixture.geometry.candidate_rows or original.dim != fixture.geometry.head_dim)
        return error.InvalidRejectedHeadFixture;
    var captured_cfg = base.fit;
    captured_cfg.backend = null;
    captured_cfg.metrics = null;
    captured_cfg.max_relative_mass_error = std.math.floatMax(f64);
    captured_cfg.max_relative_output_rmse = std.math.floatMax(f64);
    const captured = try solver.validationMetrics(allocator, original, rejected, fixture.heldout, captured_cfg);
    if ((captured.max_relative_mass_error <= base.fit.max_relative_mass_error and captured.relative_output_rmse <= base.fit.max_relative_output_rmse) or
        !std.math.isFinite(captured.max_relative_mass_error) or !std.math.isFinite(captured.relative_output_rmse))
        return error.InvalidRejectedHeadReplay;

    const keys = try toF32(allocator, fixture.original_keys);
    defer allocator.free(keys);
    const values = try toF32(allocator, fixture.original_values);
    defer allocator.free(values);
    const bias = try toF32(allocator, fixture.original_bias);
    defer allocator.free(bias);
    const logical = std.math.cast(usize, fixture.geometry.logical_length) orelse return error.InvalidRejectedHeadFixture;
    const tokens = try allocator.alloc(u32, logical);
    defer allocator.free(tokens);
    @memset(tokens, 0);
    const logits = [_]f32{0};
    const heads = [_]checkpoint.Head{.{ .dim = @intCast(fixture.geometry.head_dim), .positions = fixture.original_positions, .coverage_ends = fixture.original_coverage, .keys = keys, .values = values, .bias = bias }};
    const donors = [_]checkpoint.Donor{.{ .layer = 0, .window = 0, .heads = &heads }};
    const layer_donors = [_]u32{0};
    const source: checkpoint.State = .{ .identity = identity, .logical_length = fixture.geometry.logical_length, .tokens = tokens, .logits = &logits, .layer_donors = &layer_donors, .donors = &donors };
    var batches = try fixture.batches(allocator);
    defer batches.deinit();
    for (batches.rows, 0..) |*batch, index| {
        batch.consumer_layer = 0;
        batch.query_head = @intCast(index);
    }
    const query_heads = [_]u32{@intCast(batches.rows.len)};
    const scales = [_]f64{fixture.geometry.scale};
    const softcaps = [_]f64{fixture.geometry.softcap};
    const layout: engine.Layout = .{ .query_heads_by_layer = &query_heads, .scales_by_donor = &scales, .softcaps_by_donor = &softcaps };
    var report: Report = .{ .captured_mass_error = captured.max_relative_mass_error, .captured_output_rmse = captured.relative_output_rmse, .cells = undefined };
    var cell_index: usize = 0;
    for ([_]usize{ 4, 3, 2 }) |divisor| for ([_]?solver.Backend{ null, accelerated_backend }) |backend| {
        var stats: engine.Stats = .{};
        var options = base;
        options.stats = &stats;
        options.rejected_head_capture = null;
        options.retention_divisor = divisor;
        options.min_retention_divisor = divisor;
        options.prefix_rows = fixture.geometry.prefix_rows;
        options.recent_rows = fixture.geometry.recent_rows;
        options.chunk_rows = fixture.geometry.chunk_rows;
        options.storage_precision = .f16;
        options.validation = .whole_head;
        options.rejected_head_policy = .preserve_original;
        options.min_global_savings_ratio = null;
        options.fit.backend = backend;
        options.fit.metrics = null;
        var replayed = try engine.prepare(allocator, source, batches.rows, layout, options);
        const accepted = stats.validated_heads == 1 and stats.preserved_heads == 0;
        const preserved = stats.validated_heads == 0 and stats.preserved_heads == 1;
        const mass_error = if (accepted) stats.max_head_mass_error else stats.max_rejected_mass_error;
        const output_rmse = if (accepted) stats.max_head_output_rmse else stats.max_rejected_output_rmse;
        if ((!accepted and !preserved) or stats.attempted_chunks == 0 or stats.completed_chunks == 0 or stats.completed_chunks > stats.attempted_chunks or
            !std.math.isFinite(mass_error) or !std.math.isFinite(output_rmse) or
            (accepted and (mass_error > options.fit.max_relative_mass_error or output_rmse > options.fit.max_relative_output_rmse)) or
            (preserved and mass_error <= options.fit.max_relative_mass_error and output_rmse <= options.fit.max_relative_output_rmse))
        {
            replayed.deinit();
            return error.InvalidRejectedHeadReplay;
        }
        report.cells[cell_index] = .{ .divisor = divisor, .accelerated = backend != null, .accepted = accepted, .attempted_chunks = stats.attempted_chunks, .completed_chunks = stats.completed_chunks, .validated_heads = stats.validated_heads, .preserved_heads = stats.preserved_heads, .strict_mass_error = mass_error, .strict_output_rmse = output_rmse };
        replayed.deinit();
        cell_index += 1;
    };
    for (0..3) |pair_index| {
        const scalar = report.cells[pair_index * 2];
        const metal = report.cells[pair_index * 2 + 1];
        if (scalar.accelerated or !metal.accelerated or scalar.divisor != metal.divisor)
            return error.InvalidRejectedHeadReplay;
    }
    return report;
}

const ReplayPlan = struct {
    schema: []const u8,
    fixture_dir: []const u8,
    identity: struct { weights: []const u8, tokenizer: []const u8, template: []const u8, layout: []const u8 },
    config_sha256: []const u8,
};

fn parseDigest(text: []const u8) ![32]u8 {
    if (text.len != 64) return error.InvalidRejectedHeadReplayPlan;
    var result: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&result, text) catch return error.InvalidRejectedHeadReplayPlan;
    return result;
}

test "attention match rejected head fixture six cell replay" {
    if (comptime builtin.os.tag != .macos or !build_options.enable_metal) return error.SkipZigTest;
    const plan_path = platform.env.getenv("ANTFLY_METAL_REJECTED_HEAD_REPLAY_PLAN") orelse return error.SkipZigTest;
    if (!std.fs.path.isAbsolute(plan_path)) return error.InvalidRejectedHeadReplayPlan;
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, plan_path, allocator, .limited(1024 * 1024));
    defer allocator.free(bytes);
    const parsed = try std.json.parseFromSlice(ReplayPlan, allocator, bytes, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    const plan = parsed.value;
    if (!std.mem.eql(u8, plan.schema, "antfly.source61-first-rejected-head-replay-plan.v1") or !std.fs.path.isAbsolute(plan.fixture_dir))
        return error.InvalidRejectedHeadReplayPlan;
    const identity: checkpoint.Identity = .{
        .weights = try parseDigest(plan.identity.weights),
        .tokenizer = try parseDigest(plan.identity.tokenizer),
        .template = try parseDigest(plan.identity.template),
        .layout = try parseDigest(plan.identity.layout),
    };
    const binding: fixture_mod.Binding = .{
        .output_dir = plan.fixture_dir,
        .identity = identity,
        .config_sha256 = try parseDigest(plan.config_sha256),
    };
    var fixture = try fixture_mod.read(allocator, io, plan.fixture_dir, binding);
    defer fixture.deinit();
    const Fitter = @import("../../ops/metal_attention_match_fit.zig").Fitter;
    var fitter = try Fitter.init(allocator, null, 1024 * 1024 * 1024);
    defer fitter.deinit();
    var options: engine.Options = .{
        .storage_precision = .f16,
        .prefix_rows = 128,
        .chunk_rows = 1024,
        .min_retention_divisor = 2,
        .validation = .whole_head,
        .independent_fp64_publication = true,
        .max_transient_host_bytes = 3840 * 1024 * 1024,
        .fit = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set },
    };
    options.fit.backend = fitter.backend();
    const report = try run(allocator, &fixture, identity, options, fitter.backend());
    std.debug.print("attention_match_rejected_head_replay captured_mass_error={d} captured_output_rmse={d} cells=6 scalar_fp64_authorization=1 diagnostic_only=1 quality_credit=0\n", .{ report.captured_mass_error, report.captured_output_rmse });
    for (report.cells) |cell| std.debug.print("attention_match_rejected_head_replay divisor={d} accelerated={} accepted={} attempted_chunks={d} completed_chunks={d} validated_heads={d} preserved_heads={d} strict_mass_error={d} strict_output_rmse={d}\n", .{ cell.divisor, cell.accelerated, cell.accepted, cell.attempted_chunks, cell.completed_chunks, cell.validated_heads, cell.preserved_heads, cell.strict_mass_error, cell.strict_output_rmse });
}

const CoverageCell = struct {
    accelerated: bool,
    accepted: bool,
    mandatory_counts: [6]usize,
    mandatory_total: usize,
    mandatory_max: usize,
    attempted_chunks: usize,
    completed_chunks: usize,
    validated_heads: usize,
    preserved_heads: usize,
    strict_mass_error: f64,
    strict_output_rmse: f64,
};

/// A deliberately separate diagnostic from the six-cell baseline. It runs only
/// divisor 2, where the captured mandatory training-argmax union fits the
/// retained-row budget. Acceptance and strict rejection are both observations;
/// the common backend-null whole-head validator remains authoritative.
fn runMandatoryCoverageDivisor2(
    allocator: std.mem.Allocator,
    fixture: *const fixture_mod.Fixture,
    identity: checkpoint.Identity,
    base: engine.Options,
    accelerated_backend: solver.Backend,
) ![2]CoverageCell {
    const original = fixture.originalHead();
    const rejected = fixture.candidateHead();
    if (fixture.geometry.retention_divisor != 2 or original.rows != fixture.geometry.original_rows or
        rejected.rows != fixture.geometry.candidate_rows or original.dim != fixture.geometry.head_dim)
        return error.InvalidRejectedHeadFixture;

    const keys = try toF32(allocator, fixture.original_keys);
    defer allocator.free(keys);
    const values = try toF32(allocator, fixture.original_values);
    defer allocator.free(values);
    const bias = try toF32(allocator, fixture.original_bias);
    defer allocator.free(bias);
    const logical = std.math.cast(usize, fixture.geometry.logical_length) orelse return error.InvalidRejectedHeadFixture;
    const tokens = try allocator.alloc(u32, logical);
    defer allocator.free(tokens);
    @memset(tokens, 0);
    const logits = [_]f32{0};
    const heads = [_]checkpoint.Head{.{ .dim = @intCast(fixture.geometry.head_dim), .positions = fixture.original_positions, .coverage_ends = fixture.original_coverage, .keys = keys, .values = values, .bias = bias }};
    const donors = [_]checkpoint.Donor{.{ .layer = 0, .window = 0, .heads = &heads }};
    const layer_donors = [_]u32{0};
    const source: checkpoint.State = .{ .identity = identity, .logical_length = fixture.geometry.logical_length, .tokens = tokens, .logits = &logits, .layer_donors = &layer_donors, .donors = &donors };
    var batches = try fixture.batches(allocator);
    defer batches.deinit();
    for (batches.rows, 0..) |*batch, index| {
        batch.consumer_layer = 0;
        batch.query_head = @intCast(index);
    }
    const query_heads = [_]u32{@intCast(batches.rows.len)};
    const scales = [_]f64{fixture.geometry.scale};
    const softcaps = [_]f64{fixture.geometry.softcap};
    const layout: engine.Layout = .{ .query_heads_by_layer = &query_heads, .scales_by_donor = &scales, .softcaps_by_donor = &softcaps };

    // Match the engine's per-chunk selection exactly. The captured train
    // queries are already lossless f64; do not route them through OwnedBatches'
    // f32 reconstruction. This remains a CPU reference observation, not a GPU
    // counter. Avoid a full-head probability matrix: one chunk is live at once.
    const train_rows = fixture.train.len / fixture.geometry.head_dim;
    if (train_rows == 0 or train_rows * fixture.geometry.head_dim != fixture.train.len or
        fixture.geometry.prefix_rows != 128 or fixture.geometry.chunk_rows != 1024)
        return error.InvalidRejectedHeadReplay;
    const expected_counts = [_]usize{ 375, 212, 167, 181, 240, 183 };
    var mandatory_counts: [6]usize = undefined;
    var mandatory_total: usize = 0;
    var mandatory_max: usize = 0;
    var chunk_index: usize = 0;
    var chunk_start = fixture.geometry.prefix_rows;
    const old_rows = fixture.geometry.original_rows - fixture.geometry.recent_rows;
    if (old_rows != 5888) return error.InvalidRejectedHeadReplay;
    while (chunk_start < old_rows) : (chunk_index += 1) {
        if (chunk_index >= mandatory_counts.len) return error.InvalidRejectedHeadReplay;
        const rows = @min(fixture.geometry.chunk_rows, old_rows - chunk_start);
        const chunk_head: solver.Head = .{
            .keys = fixture.original_keys[chunk_start * fixture.geometry.head_dim ..][0 .. rows * fixture.geometry.head_dim],
            .values = fixture.original_values[chunk_start * fixture.geometry.head_dim ..][0 .. rows * fixture.geometry.head_dim],
            .bias = fixture.original_bias[chunk_start..][0..rows],
            .rows = rows,
            .dim = fixture.geometry.head_dim,
            .scale = fixture.geometry.scale,
            .softcap = fixture.geometry.softcap,
        };
        const probabilities = try allocator.alloc(f64, try std.math.mul(usize, train_rows, rows));
        defer allocator.free(probabilities);
        try solver.attendBatch(allocator, chunk_head, fixture.train, probabilities, null, null, .{ .target_rows = 1 });
        const budget = std.math.divCeil(usize, rows, 2) catch unreachable;
        const mandatory = try allocator.alloc(usize, budget);
        defer allocator.free(mandatory);
        const count = try solver.mandatoryTrainingArgmaxSeeds(probabilities, train_rows, rows, mandatory);
        std.debug.print("attention_match_mandatory_reference chunk_start={d} rows={d} count={d} expected={d} budget={d}\n", .{ chunk_start, rows, count, expected_counts[chunk_index], budget });
        if (count != expected_counts[chunk_index] or count > budget) return error.InvalidRejectedHeadReplay;
        mandatory_counts[chunk_index] = count;
        mandatory_total = try std.math.add(usize, mandatory_total, count);
        mandatory_max = @max(mandatory_max, count);
        chunk_start += rows;
    }
    if (chunk_index != mandatory_counts.len or mandatory_total != 1358 or mandatory_max != 375)
        return error.InvalidRejectedHeadReplay;

    var result: [2]CoverageCell = undefined;
    for ([_]?solver.Backend{ null, accelerated_backend }, 0..) |backend, index| {
        var stats: engine.Stats = .{};
        var options = base;
        options.stats = &stats;
        options.rejected_head_capture = null;
        options.retention_divisor = 2;
        options.min_retention_divisor = 2;
        options.prefix_rows = fixture.geometry.prefix_rows;
        options.recent_rows = fixture.geometry.recent_rows;
        options.chunk_rows = fixture.geometry.chunk_rows;
        options.storage_precision = .f16;
        options.validation = .whole_head;
        options.rejected_head_policy = .preserve_original;
        options.min_global_savings_ratio = null;
        options.fit.backend = backend;
        options.fit.metrics = null;
        options.fit.mandatory_training_argmax = true;
        var replayed = try engine.prepare(allocator, source, batches.rows, layout, options);
        defer replayed.deinit();
        const accepted = stats.validated_heads == 1 and stats.preserved_heads == 0;
        const preserved = stats.validated_heads == 0 and stats.preserved_heads == 1;
        const mass_error = if (accepted) stats.max_head_mass_error else stats.max_rejected_mass_error;
        const output_rmse = if (accepted) stats.max_head_output_rmse else stats.max_rejected_output_rmse;
        if ((!accepted and !preserved) or stats.attempted_chunks == 0 or stats.completed_chunks == 0 or
            stats.completed_chunks > stats.attempted_chunks or !std.math.isFinite(mass_error) or
            !std.math.isFinite(output_rmse) or
            (accepted and (mass_error > options.fit.max_relative_mass_error or output_rmse > options.fit.max_relative_output_rmse)) or
            (preserved and mass_error <= options.fit.max_relative_mass_error and output_rmse <= options.fit.max_relative_output_rmse))
            return error.InvalidRejectedHeadReplay;
        result[index] = .{
            .accelerated = backend != null,
            .accepted = accepted,
            .mandatory_counts = mandatory_counts,
            .mandatory_total = mandatory_total,
            .mandatory_max = mandatory_max,
            .attempted_chunks = stats.attempted_chunks,
            .completed_chunks = stats.completed_chunks,
            .validated_heads = stats.validated_heads,
            .preserved_heads = stats.preserved_heads,
            .strict_mass_error = mass_error,
            .strict_output_rmse = output_rmse,
        };
    }
    if (result[0].accelerated or !result[1].accelerated or
        !std.mem.eql(usize, &result[0].mandatory_counts, &result[1].mandatory_counts))
        return error.InvalidRejectedHeadReplay;
    return result;
}

test "attention match rejected head fixture divisor2 mandatory training argmax coverage replay" {
    if (comptime builtin.os.tag != .macos or !build_options.enable_metal) return error.SkipZigTest;
    const plan_path = platform.env.getenv("ANTFLY_METAL_REJECTED_HEAD_REPLAY_PLAN") orelse return error.SkipZigTest;
    if (!std.fs.path.isAbsolute(plan_path)) return error.InvalidRejectedHeadReplayPlan;
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, plan_path, allocator, .limited(1024 * 1024));
    defer allocator.free(bytes);
    const parsed = try std.json.parseFromSlice(ReplayPlan, allocator, bytes, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    const plan = parsed.value;
    if (!std.mem.eql(u8, plan.schema, "antfly.source61-first-rejected-head-replay-plan.v1") or !std.fs.path.isAbsolute(plan.fixture_dir))
        return error.InvalidRejectedHeadReplayPlan;
    const identity: checkpoint.Identity = .{ .weights = try parseDigest(plan.identity.weights), .tokenizer = try parseDigest(plan.identity.tokenizer), .template = try parseDigest(plan.identity.template), .layout = try parseDigest(plan.identity.layout) };
    const binding: fixture_mod.Binding = .{ .output_dir = plan.fixture_dir, .identity = identity, .config_sha256 = try parseDigest(plan.config_sha256) };
    var fixture = try fixture_mod.read(allocator, io, plan.fixture_dir, binding);
    defer fixture.deinit();
    const Fitter = @import("../../ops/metal_attention_match_fit.zig").Fitter;
    var fitter = try Fitter.init(allocator, null, 1024 * 1024 * 1024);
    defer fitter.deinit();
    var options: engine.Options = .{ .storage_precision = .f16, .prefix_rows = 128, .chunk_rows = 1024, .min_retention_divisor = 2, .validation = .whole_head, .independent_fp64_publication = true, .max_transient_host_bytes = 3840 * 1024 * 1024, .fit = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .mandatory_training_argmax = true } };
    options.fit.backend = fitter.backend();
    const cells = try runMandatoryCoverageDivisor2(allocator, &fixture, identity, options, fitter.backend());
    for (cells) |cell| std.debug.print("attention_match_rejected_head_mandatory_argmax divisor=2 accelerated={} accepted={} mandatory_counts={any} mandatory_total={d} mandatory_max={d} attempted_chunks={d} completed_chunks={d} validated_heads={d} preserved_heads={d} strict_mass_error={d} strict_output_rmse={d} scalar_fp64_authorization=1 diagnostic_only=1 quality_credit=0\n", .{ cell.accelerated, cell.accepted, cell.mandatory_counts, cell.mandatory_total, cell.mandatory_max, cell.attempted_chunks, cell.completed_chunks, cell.validated_heads, cell.preserved_heads, cell.strict_mass_error, cell.strict_output_rmse });
}

const Top2Cell = struct {
    accelerated: bool,
    accepted: bool,
    elapsed_ns: u64,
    accelerated_value_solves: u64,
    max_nnls_sweeps: usize,
    attempted_chunks: usize,
    completed_chunks: usize,
    validated_heads: usize,
    preserved_heads: usize,
    strict_mass_error: f64,
    strict_output_rmse: f64,
};

const Top2ReplayPlan = struct {
    schema: []const u8,
    fixture_dir: []const u8,
    identity: struct { weights: []const u8, tokenizer: []const u8, template: []const u8, layout: []const u8 },
    config_sha256: []const u8,
    replay_plan: struct { path: []const u8, sha256: []const u8 },
    top2_override: struct {
        schema: []const u8,
        mandatory_training_argmax: bool,
        mandatory_training_top_k: usize,
        retention_divisor: usize,
        original_config_sha256: []const u8,
        parent_fixture_manifest_sha256: []const u8,
        parent_replay_plan_sha256: []const u8,
        top2_solver_patch_sha256: []const u8,
        top2_replay_patch_sha256: []const u8,
        derived_config_binding: []const u8,
        derived_config_sha256: []const u8,
    },
    rejected_metal_capture_dir: ?[]const u8 = null,
};

test "attention match rejected head fixture divisor2 mandatory training top2 coverage replay" {
    if (comptime builtin.os.tag != .macos or !build_options.enable_metal) return error.SkipZigTest;
    const plan_path = platform.env.getenv("ANTFLY_METAL_REJECTED_HEAD_TOP2_REPLAY_PLAN") orelse return error.SkipZigTest;
    if (!std.fs.path.isAbsolute(plan_path)) return error.InvalidRejectedHeadReplayPlan;
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, plan_path, allocator, .limited(1024 * 1024));
    defer allocator.free(bytes);
    const parsed = try std.json.parseFromSlice(Top2ReplayPlan, allocator, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = true });
    defer parsed.deinit();
    const plan = parsed.value;
    if (!std.mem.eql(u8, plan.schema, "antfly.source61-guarded-diagnostic-test-plan.v3") or
        !std.mem.eql(u8, plan.top2_override.schema, "antfly.source63-top2-rejected-head-fit-config.v1") or
        !std.fs.path.isAbsolute(plan.fixture_dir) or !plan.top2_override.mandatory_training_argmax or
        plan.top2_override.mandatory_training_top_k != 2 or plan.top2_override.retention_divisor != 2 or
        !std.mem.eql(u8, plan.config_sha256, plan.top2_override.original_config_sha256))
        return error.InvalidRejectedHeadReplayPlan;
    _ = try parseDigest(plan.top2_override.parent_fixture_manifest_sha256);
    _ = try parseDigest(plan.top2_override.parent_replay_plan_sha256);
    const derived_config = try parseDigest(plan.top2_override.derived_config_sha256);
    const expected_binding = try std.fmt.allocPrint(allocator, "schema=antfly.source63-top2-rejected-head-fit-config.v1\noriginal_config_sha256={s}\nparent_fixture_manifest_sha256={s}\nparent_replay_plan_sha256={s}\nmandatory_training_argmax=true\nmandatory_training_top_k=2\nretention_divisor=2\nstorage_precision=f16\nfit_backends=cpu,metal\nfinal_authorization=backend_null_fp64_whole_head\ntop2_solver_patch_sha256=a264f78a1a52d7c6cd067999fd197c898b93db1b7e7dc0359602fbe2fa5d7d78\ntop2_replay_patch_sha256={s}\n", .{ plan.config_sha256, plan.top2_override.parent_fixture_manifest_sha256, plan.top2_override.parent_replay_plan_sha256, plan.top2_override.top2_replay_patch_sha256 });
    defer allocator.free(expected_binding);
    var binding_hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(expected_binding, &binding_hash, .{});
    if (!std.mem.eql(u8, expected_binding, plan.top2_override.derived_config_binding) or !std.mem.eql(u8, &binding_hash, &derived_config)) return error.InvalidRejectedHeadReplayPlan;
    const identity: checkpoint.Identity = .{ .weights = try parseDigest(plan.identity.weights), .tokenizer = try parseDigest(plan.identity.tokenizer), .template = try parseDigest(plan.identity.template), .layout = try parseDigest(plan.identity.layout) };
    const binding: fixture_mod.Binding = .{ .output_dir = plan.fixture_dir, .identity = identity, .config_sha256 = try parseDigest(plan.config_sha256) };
    var fixture = try fixture_mod.read(allocator, io, plan.fixture_dir, binding);
    defer fixture.deinit();
    var fixture_hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(fixture.manifest_bytes, &fixture_hash, .{});
    const expected_fixture_hash = try parseDigest(plan.top2_override.parent_fixture_manifest_sha256);
    const expected_parent_plan_hash = try parseDigest(plan.top2_override.parent_replay_plan_sha256);
    if (!std.mem.eql(u8, &fixture_hash, &expected_fixture_hash) or
        !std.mem.eql(u8, plan.replay_plan.sha256, plan.top2_override.parent_replay_plan_sha256) or
        !std.fs.path.isAbsolute(plan.replay_plan.path))
        return error.InvalidRejectedHeadReplayPlan;
    const parent_plan_bytes = try std.Io.Dir.cwd().readFileAlloc(io, plan.replay_plan.path, allocator, .limited(1024 * 1024));
    defer allocator.free(parent_plan_bytes);
    var parent_plan_hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(parent_plan_bytes, &parent_plan_hash, .{});
    if (!std.mem.eql(u8, &parent_plan_hash, &expected_parent_plan_hash)) return error.InvalidRejectedHeadReplayPlan;
    var batches = try fixture.batches(allocator);
    defer batches.deinit();
    for (batches.rows, 0..) |*batch, index| {
        batch.consumer_layer = 0;
        batch.query_head = @intCast(index);
    }

    const train_rows = fixture.train.len / fixture.geometry.head_dim;
    const old_rows = fixture.geometry.original_rows - fixture.geometry.recent_rows;
    if (train_rows == 0 or train_rows * fixture.geometry.head_dim != fixture.train.len or
        fixture.geometry.prefix_rows != 128 or fixture.geometry.chunk_rows != 1024 or old_rows != 5888)
        return error.InvalidRejectedHeadReplay;
    var reference_counts: [6]usize = undefined;
    var reference_total: usize = 0;
    var reference_max: usize = 0;
    var chunk_index: usize = 0;
    var chunk_start = fixture.geometry.prefix_rows;
    while (chunk_start < old_rows) : (chunk_index += 1) {
        if (chunk_index >= reference_counts.len) return error.InvalidRejectedHeadReplay;
        const rows = @min(fixture.geometry.chunk_rows, old_rows - chunk_start);
        const chunk_head: solver.Head = .{ .keys = fixture.original_keys[chunk_start * fixture.geometry.head_dim ..][0 .. rows * fixture.geometry.head_dim], .values = fixture.original_values[chunk_start * fixture.geometry.head_dim ..][0 .. rows * fixture.geometry.head_dim], .bias = fixture.original_bias[chunk_start..][0..rows], .rows = rows, .dim = fixture.geometry.head_dim, .scale = fixture.geometry.scale, .softcap = fixture.geometry.softcap };
        const probabilities = try allocator.alloc(f64, try std.math.mul(usize, train_rows, rows));
        defer allocator.free(probabilities);
        try solver.attendBatch(allocator, chunk_head, fixture.train, probabilities, null, null, .{ .target_rows = 1 });
        const selected = try allocator.alloc(usize, rows);
        defer allocator.free(selected);
        const count = try solver.mandatoryTrainingTopKSeeds(probabilities, train_rows, rows, 2, selected);
        const budget = std.math.divCeil(usize, rows, 2) catch unreachable;
        if (count > budget) return error.CompactionFitRejected;
        reference_counts[chunk_index] = count;
        reference_total = try std.math.add(usize, reference_total, count);
        reference_max = @max(reference_max, count);
        std.debug.print("attention_match_top2_reference chunk_start={d} rows={d} count={d} budget={d} stable_ties=lower_index\n", .{ chunk_start, rows, count, budget });
        chunk_start += rows;
    }
    if (chunk_index != reference_counts.len) return error.InvalidRejectedHeadReplay;

    const keys = try toF32(allocator, fixture.original_keys);
    defer allocator.free(keys);
    const values = try toF32(allocator, fixture.original_values);
    defer allocator.free(values);
    const bias = try toF32(allocator, fixture.original_bias);
    defer allocator.free(bias);
    const logical = std.math.cast(usize, fixture.geometry.logical_length) orelse return error.InvalidRejectedHeadFixture;
    const tokens = try allocator.alloc(u32, logical);
    defer allocator.free(tokens);
    @memset(tokens, 0);
    const logits = [_]f32{0};
    const heads = [_]checkpoint.Head{.{ .dim = @intCast(fixture.geometry.head_dim), .positions = fixture.original_positions, .coverage_ends = fixture.original_coverage, .keys = keys, .values = values, .bias = bias }};
    const donors = [_]checkpoint.Donor{.{ .layer = 0, .window = 0, .heads = &heads }};
    const layer_donors = [_]u32{0};
    const source: checkpoint.State = .{ .identity = identity, .logical_length = fixture.geometry.logical_length, .tokens = tokens, .logits = &logits, .layer_donors = &layer_donors, .donors = &donors };
    const query_heads = [_]u32{@intCast(batches.rows.len)};
    const scales = [_]f64{fixture.geometry.scale};
    const softcaps = [_]f64{fixture.geometry.softcap};
    const layout: engine.Layout = .{ .query_heads_by_layer = &query_heads, .scales_by_donor = &scales, .softcaps_by_donor = &softcaps };
    const Fitter = @import("../../ops/metal_attention_match_fit.zig").Fitter;
    var fitter = try Fitter.init(allocator, null, 1024 * 1024 * 1024);
    var fitter_live = true;
    defer if (fitter_live) fitter.deinit();
    var optional_writer: ?fixture_mod.Writer = null;
    if (plan.rejected_metal_capture_dir) |dir| {
        if (!std.fs.path.isAbsolute(dir) or std.mem.eql(u8, dir, plan.fixture_dir)) return error.InvalidRejectedHeadReplayPlan;
        optional_writer = try fixture_mod.Writer.init(allocator, io, .{ .output_dir = dir, .identity = identity, .config_sha256 = derived_config });
    }
    defer if (optional_writer) |*writer| writer.deinit();

    var cells: [2]Top2Cell = undefined;
    for ([_]?solver.Backend{ null, fitter.backend() }, 0..) |backend, index| {
        var stats: engine.Stats = .{};
        var options: engine.Options = .{ .storage_precision = .f16, .prefix_rows = fixture.geometry.prefix_rows, .recent_rows = fixture.geometry.recent_rows, .chunk_rows = fixture.geometry.chunk_rows, .retention_divisor = 2, .min_retention_divisor = 2, .validation = .whole_head, .rejected_head_policy = .preserve_original, .independent_fp64_publication = true, .max_transient_host_bytes = 3840 * 1024 * 1024, .stats = &stats, .fit = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .mandatory_training_argmax = true, .mandatory_training_top_k = 2 } };
        options.min_global_savings_ratio = null;
        options.fit.backend = backend;
        options.fit.metrics = null;
        if (index == 1) if (optional_writer) |*writer| {
            options.rejected_head_capture = writer.callback();
        };
        const started = platform.time.monotonicNs();
        var replayed = try engine.prepare(allocator, source, batches.rows, layout, options);
        defer replayed.deinit();
        const elapsed = platform.time.monotonicNs() -| started;
        const accepted = stats.validated_heads == 1 and stats.preserved_heads == 0;
        const preserved = stats.validated_heads == 0 and stats.preserved_heads == 1;
        const mass_error = if (accepted) stats.max_head_mass_error else stats.max_rejected_mass_error;
        const output_rmse = if (accepted) stats.max_head_output_rmse else stats.max_rejected_output_rmse;
        if ((!accepted and !preserved) or elapsed == 0 or stats.attempted_chunks == 0 or stats.completed_chunks == 0 or stats.completed_chunks > stats.attempted_chunks or
            !std.math.isFinite(mass_error) or !std.math.isFinite(output_rmse) or
            (accepted and (mass_error > options.fit.max_relative_mass_error or output_rmse > options.fit.max_relative_output_rmse)) or
            (preserved and mass_error <= options.fit.max_relative_mass_error and output_rmse <= options.fit.max_relative_output_rmse))
            return error.InvalidRejectedHeadReplay;
        if (index == 1 and preserved) if (optional_writer) |*writer| {
            const captured = try writer.finish();
            if (!captured.committed or captured.required_bytes > fixture_mod.maximum_bytes) return error.InvalidRejectedHeadReplay;
        };
        cells[index] = .{ .accelerated = backend != null, .accepted = accepted, .elapsed_ns = elapsed, .accelerated_value_solves = 0, .max_nnls_sweeps = stats.max_nnls_sweeps, .attempted_chunks = stats.attempted_chunks, .completed_chunks = stats.completed_chunks, .validated_heads = stats.validated_heads, .preserved_heads = stats.preserved_heads, .strict_mass_error = mass_error, .strict_output_rmse = output_rmse };
    }
    const solves = fitter.deinitAndTakeValueSolves();
    fitter_live = false;
    cells[1].accelerated_value_solves = solves;
    if (cells[0].accelerated or !cells[1].accelerated or solves == 0) return error.InvalidRejectedHeadReplay;
    for (cells) |cell| std.debug.print("attention_match_rejected_head_top2 divisor=2 accelerated={} accepted={} elapsed_ns={d} accelerated_value_solves={d} max_nnls_sweeps={d} mandatory_counts={any} mandatory_total={d} mandatory_max={d} attempted_chunks={d} completed_chunks={d} validated_heads={d} preserved_heads={d} strict_mass_error={d} strict_output_rmse={d} scalar_fp64_authorization=1 derived_fixture={} diagnostic_only=1 quality_credit=0\n", .{ cell.accelerated, cell.accepted, cell.elapsed_ns, cell.accelerated_value_solves, cell.max_nnls_sweeps, reference_counts, reference_total, reference_max, cell.attempted_chunks, cell.completed_chunks, cell.validated_heads, cell.preserved_heads, cell.strict_mass_error, cell.strict_output_rmse, plan.rejected_metal_capture_dir != null and !cell.accepted });
}
