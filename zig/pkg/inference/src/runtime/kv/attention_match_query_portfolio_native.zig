// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const platform = @import("antfly_platform");
const checkpoint = @import("attention_match_checkpoint.zig");
const solver = @import("attention_match.zig");
const whole = @import("attention_match_whole_fit_input.zig");
const augmentation = @import("attention_match_training_augmentation.zig");
const replay = @import("attention_match_query_portfolio_replay.zig");

const workspace_bytes: usize = 1024 * 1024 * 1024;
const original_train_rows: usize = 2916;
const portfolio_rows: usize = 512;
const heldout_rows: usize = 412;

const Plan = struct {
    schema: []const u8,
    input_dir: []const u8,
    augmentation_dir: []const u8,
    identity: struct { weights: []const u8, tokenizer: []const u8, template: []const u8, layout: []const u8 },
    config_sha256: []const u8,
    source_freeze_sha256: []const u8,
    binary_sha256: []const u8,
    exporter_sha256: []const u8,
    input_manifest_sha256: []const u8,
    augmentation_manifest_sha256: []const u8,
    fixed_source65_manifest_sha256: []const u8,
    parent_study_tokens_sha256: []const u8,
    study_tokens_sha256: []const u8,
    method: []const u8,
    query_samples: u16,
    diagnostic_donor: usize,
    diagnostic_head: usize,
    portfolio_policy: []const u8,
    original_train_rows: usize,
    portfolio_rows: usize,
    heldout_rows: usize,
    original_train_sha256: []const u8,
    sample_indices_sha256: []const u8,
    result_path: []const u8,
    checkpoint_mapping_authority: []const u8,
    mapped_consumer_layers: []const usize,
    donor_head_count: usize,
    donor_window: usize,
    max_workspace_bytes: usize,
    retention_ladder: [3]usize,
    mass_gate: f64,
    output_gate: f64,
    heldout_controls_admission: bool,
    heldout_controls_fit: bool,
    raw_selection_accelerated: bool,
};

fn digest(text: []const u8) ![32]u8 {
    if (text.len != 64) return error.InvalidWholeFitInput;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, text) catch return error.InvalidWholeFitInput;
    return out;
}

fn portfolioBackend(backend: solver.Backend) solver.Backend {
    var isolated = backend;
    isolated.raw_select_fn = null;
    return isolated;
}

fn exactMappedConsumers(layer_donors: []const u32, donor: usize, expected: []const usize) bool {
    var at: usize = 0;
    for (layer_donors, 0..) |mapped, layer| if (mapped == donor) {
        if (at >= expected.len or expected[at] != layer) return false;
        at += 1;
    };
    return at == expected.len;
}

test "query portfolio chunk admission source72 augmented replay" {
    if (comptime builtin.os.tag != .macos or !build_options.enable_metal) return error.SkipZigTest;
    const plan_path = platform.env.getenv("ANTFLY_METAL_QUERY_PORTFOLIO_PLAN") orelse return error.SkipZigTest;
    if (!std.fs.path.isAbsolute(plan_path)) return error.InvalidWholeFitInput;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, plan_path, allocator, .limited(1024 * 1024));
    defer allocator.free(bytes);
    const parsed = try std.json.parseFromSlice(Plan, allocator, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false });
    defer parsed.deinit();
    const plan = parsed.value;
    var plan_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &plan_sha, .{});
    if (!std.mem.eql(u8, plan.schema, "antfly.attention-match-query-portfolio-plan.v15") or !std.fs.path.isAbsolute(plan.input_dir) or !std.fs.path.isAbsolute(plan.augmentation_dir) or !std.fs.path.isAbsolute(plan.result_path) or !std.mem.eql(u8, plan.method, "top1") or plan.query_samples != 512 or plan.diagnostic_donor != 5 or plan.diagnostic_head != 1 or !std.mem.eql(u8, plan.portfolio_policy, "equal_width_stratum_centers_v1") or plan.original_train_rows != original_train_rows or plan.portfolio_rows != portfolio_rows or plan.heldout_rows != heldout_rows or plan.max_workspace_bytes != workspace_bytes or plan.retention_ladder[0] != 4 or plan.retention_ladder[1] != 3 or plan.retention_ladder[2] != 2 or plan.mass_gate != 0.05 or plan.output_gate != 0.05 or !plan.heldout_controls_admission or plan.heldout_controls_fit or plan.raw_selection_accelerated or !std.mem.eql(u8, plan.checkpoint_mapping_authority, "authenticated_original_checkpoint_f16_v3_layer_donors") or plan.mapped_consumer_layers.len != 1 or plan.mapped_consumer_layers[0] != 5 or plan.donor_head_count != 2 or plan.donor_window != 0) return error.InvalidWholeFitInput;
    const input_manifest_path = try std.fs.path.join(allocator, &.{ plan.input_dir, "manifest.json" });
    defer allocator.free(input_manifest_path);
    const input_manifest = try std.Io.Dir.cwd().readFileAlloc(io, input_manifest_path, allocator, .limited(whole.manifest_reservation));
    defer allocator.free(input_manifest);
    var input_manifest_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(input_manifest, &input_manifest_sha, .{});
    if (!std.mem.eql(u8, &input_manifest_sha, &(try digest(plan.input_manifest_sha256)))) return error.InvalidWholeFitInput;
    const augmentation_manifest_path = try std.fs.path.join(allocator, &.{ plan.augmentation_dir, "manifest.json" });
    defer allocator.free(augmentation_manifest_path);
    const augmentation_manifest = try std.Io.Dir.cwd().readFileAlloc(io, augmentation_manifest_path, allocator, .limited(64 * 1024));
    defer allocator.free(augmentation_manifest);
    var augmentation_manifest_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(augmentation_manifest, &augmentation_manifest_sha, .{});
    if (!std.mem.eql(u8, &augmentation_manifest_sha, &(try digest(plan.augmentation_manifest_sha256)))) return error.InvalidWholeFitInput;
    const identity: checkpoint.Identity = .{ .weights = try digest(plan.identity.weights), .tokenizer = try digest(plan.identity.tokenizer), .template = try digest(plan.identity.template), .layout = try digest(plan.identity.layout) };
    var input = try whole.read(allocator, io, .{ .output_dir = plan.input_dir, .identity = identity, .config_sha256 = try digest(plan.config_sha256), .source_freeze_sha256 = try digest(plan.source_freeze_sha256), .binary_sha256 = try digest(plan.binary_sha256), .exporter_sha256 = try digest(plan.exporter_sha256), .limits = checkpoint.serving_limits });
    defer input.deinit();
    var augmented = try augmentation.read(allocator, io, .{ .output_dir = plan.augmentation_dir, .parent_manifest_sha256 = input_manifest_sha, .identity = identity, .source_freeze_sha256 = try digest(plan.source_freeze_sha256), .binary_sha256 = try digest(plan.binary_sha256), .exporter_sha256 = try digest(plan.exporter_sha256), .parent_study_tokens_sha256 = try digest(plan.parent_study_tokens_sha256), .study_tokens_sha256 = try digest(plan.study_tokens_sha256) });
    defer augmented.deinit();
    if (input.query_samples != 512 or input.method != .top1) return error.InvalidWholeFitInput;
    if (!exactMappedConsumers(input.checkpoint_owned.state.layer_donors, plan.diagnostic_donor, plan.mapped_consumer_layers) or input.checkpoint_owned.state.donors[plan.diagnostic_donor].heads.len != plan.donor_head_count or input.checkpoint_owned.state.donors[plan.diagnostic_donor].window != plan.donor_window) return error.InvalidWholeFitInput;
    const Fitter = @import("../../ops/metal_attention_match_fit.zig").Fitter;
    var fitter = try Fitter.init(allocator, null, workspace_bytes);
    var live = true;
    defer if (live) fitter.deinit();
    const started = platform.time.monotonicNs();
    var result = try replay.runInput(allocator, &input, augmented.batches, portfolioBackend(fitter.backend()), plan.diagnostic_donor, plan.diagnostic_head, portfolio_rows, try digest(plan.original_train_sha256), try digest(plan.sample_indices_sha256), workspace_bytes);
    defer result.deinit();
    const solves = fitter.deinitAndTakeValueSolves();
    live = false;
    const elapsed_ns = platform.time.monotonicNs() -| started;
    if (result.original_train_rows != original_train_rows or result.portfolio_rows != portfolio_rows or result.heldout_rows != heldout_rows or !std.mem.eql(u8, &result.original_train_sha256, &(try digest(plan.original_train_sha256))) or !std.mem.eql(u8, &result.sample_indices_sha256, &(try digest(plan.sample_indices_sha256))) or solves == 0 or elapsed_ns == 0) return error.InvalidWholeFitInput;
    const source_head = input.checkpoint_owned.state.donors[plan.diagnostic_donor].heads[plan.diagnostic_head];
    const source_bytes = try std.math.mul(usize, try std.math.mul(usize, source_head.positions.len, source_head.dim), 4);
    const candidate_bytes = try std.math.add(usize, try std.math.mul(usize, try std.math.mul(usize, result.replayed.head.rows, source_head.dim), 4), try std.math.mul(usize, result.replayed.head.rows, 20));
    var accepted_chunks: usize = 0;
    for (result.replayed.chunks) |chunk| if (chunk.chosen_divisor != 0) {
        accepted_chunks += 1;
    };
    const json = try std.json.Stringify.valueAlloc(allocator, .{ .schema = "antfly.attention-match-query-portfolio-result.v15", .authenticated = true, .plan_sha256 = std.fmt.bytesToHex(plan_sha, .lower), .input_manifest_sha256 = std.fmt.bytesToHex(input_manifest_sha, .lower), .augmentation_manifest_sha256 = std.fmt.bytesToHex(augmentation_manifest_sha, .lower), .fixed_source65_manifest_sha256 = plan.fixed_source65_manifest_sha256, .identity = plan.identity, .config_sha256 = plan.config_sha256, .parent_study_tokens_sha256 = plan.parent_study_tokens_sha256, .study_tokens_sha256 = plan.study_tokens_sha256, .source_freeze_sha256 = plan.source_freeze_sha256, .binary_sha256 = plan.binary_sha256, .exporter_sha256 = plan.exporter_sha256, .method = plan.method, .query_samples = plan.query_samples, .diagnostic_donor = plan.diagnostic_donor, .diagnostic_head = plan.diagnostic_head, .portfolio_policy = plan.portfolio_policy, .original_train_rows = result.original_train_rows, .portfolio_rows = result.portfolio_rows, .heldout_rows = result.heldout_rows, .original_train_sha256 = std.fmt.bytesToHex(result.original_train_sha256, .lower), .sample_indices_sha256 = std.fmt.bytesToHex(result.sample_indices_sha256, .lower), .max_workspace_bytes = plan.max_workspace_bytes, .retention_ladder = plan.retention_ladder, .mass_gate = plan.mass_gate, .output_gate = plan.output_gate, .heldout_controls_admission = true, .heldout_controls_fit = false, .training_portfolio_only = true, .portfolio_selector_has_heldout_input = false, .parameter_sweep = false, .raw_selection_accelerated = false, .checkpoint_mapping_authority = plan.checkpoint_mapping_authority, .mapped_consumer_layers = plan.mapped_consumer_layers, .donor_head_count = plan.donor_head_count, .donor_window = plan.donor_window, .cumulative_heldout_cache_f64 = true, .final_mass_error = result.replayed.final_mass_error, .final_output_rmse = result.replayed.final_output_rmse, .rows_before = source_head.positions.len, .rows_after = result.replayed.head.rows, .accepted_chunks = accepted_chunks, .preserved_chunks = result.replayed.chunks.len - accepted_chunks, .source_head_bytes = source_bytes, .candidate_head_bytes = candidate_bytes, .single_head_savings_ratio = @as(f64, @floatFromInt(source_bytes)) / @as(f64, @floatFromInt(candidate_bytes)), .global_savings_claim = false, .accelerated_value_solves = solves, .elapsed_ns = elapsed_ns, .chunks = result.replayed.chunks, .full_heldout_backend_null_fp64 = true, .diagnostic_only = true, .quality_credit = false, .production_qualified = false, .publication_claim = false, .admission_claim = false }, .{});
    defer allocator.free(json);
    var file = try std.Io.Dir.createFileAbsolute(io, plan.result_path, .{ .exclusive = true, .truncate = false });
    defer file.close(io);
    try file.writeStreamingAll(io, json);
    try file.writeStreamingAll(io, "\n");
    try file.sync(io);
}

test "query portfolio backend disables raw selector only" {
    const Trap = struct {
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn raw(_: ?*anyopaque, _: std.mem.Allocator, _: []const f64, _: []const f64, _: []const f64, _: usize, _: usize, _: []const usize, _: []usize, _: []f64, _: solver.Config) !usize {
            return 0;
        }
    };
    const original: solver.Backend = .{ .workspace_fn = Trap.workspace, .raw_select_fn = Trap.raw };
    const isolated = portfolioBackend(original);
    try std.testing.expect(original.raw_select_fn != null);
    try std.testing.expect(isolated.raw_select_fn == null);
    try std.testing.expectEqual(original.workspace_fn, isolated.workspace_fn);
}

test "query portfolio checkpoint mapping separates donor 5 from shared late donor" {
    const mapping = [_]u32{ 0, 5, 23, 23, 5, 23 };
    try std.testing.expect(exactMappedConsumers(&mapping, 5, &.{ 1, 4 }));
    try std.testing.expect(exactMappedConsumers(&mapping, 23, &.{ 2, 3, 5 }));
    try std.testing.expect(!exactMappedConsumers(&mapping, 5, &.{ 1, 2, 3, 4 }));
}
