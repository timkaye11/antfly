// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//! Allocation-bounded adapter from an authenticated flat TRAIN cache page to
//! the optional FP64 packed numerator-delta kernel. Mass keeps scalar order.
const std = @import("std");
const solver = @import("attention_match.zig");
const training = @import("attention_match_joint_training_source.zig");
const accelerated = @import("attention_match_gram_delta_accelerate.zig");

pub const Evidence = struct {
    tiles: usize = 0,
    backend_calls: usize = 0,
    packed_bytes_peak: usize = 0,
    staged_bytes_peak: usize = 0,
    page_bytes_peak: usize = 0,
    fallbacks: usize = 0,
};

fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}
fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}

pub fn workspaceBytes(query_rows: usize, range_rows: usize, selected_rows: usize, dim: usize) !usize {
    const page = try mul(try mul(query_rows, range_rows), @sizeOf(f64));
    const adapter_staging = try mul(try mul(query_rows, dim), @sizeOf(f64));
    return add(page, try add(adapter_staging, try accelerated.workspaceBytes(query_rows, selected_rows, dim)));
}

/// `ratios` is q-independent and caller-owned. `external_live_bytes` includes
/// ratios, the retained scored plan/cache and every caller allocation. Outputs
/// are unchanged on error because numerator publication is staged and mass is
/// computed only after successful acceleration.
pub fn fill(
    allocator: std.mem.Allocator,
    cached: *const training.Source,
    original: solver.Head,
    train: []const f64,
    cfg: solver.Config,
    range_start: usize,
    range_rows: usize,
    selected_local: []const usize,
    ratios: []const f64,
    fitted_values: []const f64,
    first_query: usize,
    query_rows: usize,
    backend: accelerated.Backend,
    external_live_bytes: usize,
    combined_cap: usize,
    control: solver.Control,
    delta_mass: []f64,
    delta_numerator: []f64,
    evidence: *Evidence,
) !void {
    // The owning replay authenticated the complete Head when it prepared the
    // bound training Source. Repeating Head.validate here would rescan all N*D
    // keys and values for every query tile. Source.validate below binds the
    // exact pointers, shape, scale, softcap, and backend. The packed kernel
    // independently validates every consumed page/range/value payload.
    cached.validate(original, train, cfg) catch |err| {
        std.debug.print("attention_match_delta_adapter_failure stage=cache_validate first_query={d} query_rows={d} range_start={d} range_rows={d} error={s} diagnostic_only=true\n", .{ first_query, query_rows, range_start, range_rows, @errorName(err) });
        return err;
    };
    if (cfg.control.context != control.context or cfg.control.check_fn != control.check_fn) {
        std.debug.print("attention_match_delta_adapter_failure stage=control_identity first_query={d} query_rows={d} range_start={d} range_rows={d} error=InvalidFitConfig diagnostic_only=true\n", .{ first_query, query_rows, range_start, range_rows });
        return error.InvalidFitConfig;
    }
    const query_end = try add(first_query, query_rows);
    const range_end = try add(range_start, range_rows);
    const qd = try mul(query_rows, original.dim);
    if (query_rows == 0 or range_rows == 0 or selected_local.len == 0 or
        query_end > cached.query_count or range_end > original.rows or
        ratios.len != selected_local.len or fitted_values.len != try mul(selected_local.len, original.dim) or
        delta_mass.len != query_rows or delta_numerator.len != qd)
        return error.InvalidAttentionShape;
    for (selected_local, ratios) |local, ratio| if (local >= range_rows or !std.math.isFinite(ratio) or ratio < 0) return error.InvalidAttentionShape;
    const page_len = try mul(query_rows, range_rows);
    const page_bytes = try mul(page_len, @sizeOf(f64));
    const owned = try workspaceBytes(query_rows, range_rows, selected_local.len, original.dim);
    if (external_live_bytes > combined_cap or owned > combined_cap - external_live_bytes) return error.CompactionWorkspaceLimit;
    control.check() catch |err| {
        std.debug.print("attention_match_delta_adapter_failure stage=control_before_page first_query={d} query_rows={d} range_start={d} range_rows={d} error={s} diagnostic_only=true\n", .{ first_query, query_rows, range_start, range_rows, @errorName(err) });
        return err;
    };
    const page = try allocator.alloc(f64, page_len);
    defer allocator.free(page);
    for (0..query_rows) |local_q| {
        control.check() catch |err| {
            std.debug.print("attention_match_delta_adapter_failure stage=control_page_copy first_query={d} query_rows={d} local_query={d} range_start={d} range_rows={d} error={s} diagnostic_only=true\n", .{ first_query, query_rows, local_q, range_start, range_rows, @errorName(err) });
            return err;
        };
        for (0..range_rows) |local| page[local_q * range_rows + local] = try cached.probability(first_query + local_q, range_start + local);
    }
    const original_values = original.values[try mul(range_start, original.dim)..][0..try mul(range_rows, original.dim)];
    var kernel: accelerated.Evidence = .{};
    const numerator_external = try add(external_live_bytes, page_bytes);
    const staged_numerator = try allocator.alloc(f64, qd);
    defer allocator.free(staged_numerator);
    // Charge the adapter-owned staging explicitly; the kernel then charges its
    // own packed+staged allocations against the remaining physical authority.
    const staging_bytes = try mul(qd, @sizeOf(f64));
    const kernel_external = try add(numerator_external, staging_bytes);
    if (kernel_external > combined_cap) return error.CompactionWorkspaceLimit;
    accelerated.compute(allocator, page, original_values, selected_local, ratios, fitted_values, query_rows, range_rows, original.dim, .{
        .backend = backend,
        .require_accelerated = true,
        .external_live_bytes = kernel_external,
        .max_workspace_bytes = combined_cap,
        .control = control,
    }, staged_numerator, &kernel) catch |err| {
        std.debug.print("attention_match_delta_adapter_failure stage=accelerated_compute first_query={d} query_rows={d} range_start={d} range_rows={d} selected={d} error={s} diagnostic_only=true\n", .{ first_query, query_rows, range_start, range_rows, selected_local.len, @errorName(err) });
        return err;
    };
    if (!kernel.accelerated or kernel.backend_calls != 2 or kernel.unsupported_fallbacks != 0 or kernel.backend_error_fallbacks != 0 or kernel.nonfinite_fallbacks != 0)
        return error.AcceleratedDeltaUnsupported;
    const staged_mass = try allocator.alloc(f64, query_rows);
    defer allocator.free(staged_mass);
    for (0..query_rows) |qi| {
        try control.check();
        var mass: f64 = 0;
        for (0..range_rows) |local| mass -= page[qi * range_rows + local];
        for (selected_local, ratios) |local, ratio| mass += page[qi * range_rows + local] * ratio;
        if (!std.math.isFinite(mass)) return error.NonFiniteAttention;
        staged_mass[qi] = mass;
    }
    @memcpy(delta_mass, staged_mass);
    @memcpy(delta_numerator, staged_numerator);
    evidence.tiles = try add(evidence.tiles, 1);
    evidence.backend_calls = try add(evidence.backend_calls, kernel.backend_calls);
    evidence.packed_bytes_peak = @max(evidence.packed_bytes_peak, kernel.packed_bytes);
    evidence.staged_bytes_peak = @max(evidence.staged_bytes_peak, try add(kernel.staged_bytes, staging_bytes));
    evidence.page_bytes_peak = @max(evidence.page_bytes_peak, page_bytes);
}

test "adapter workspace includes page and both staging layers" {
    try std.testing.expectEqual(@as(usize, (2 * 4 + 2 * 2 + 2 * 3 + 2 * 2) * 8), try workspaceBytes(2, 4, 3, 2));
}
