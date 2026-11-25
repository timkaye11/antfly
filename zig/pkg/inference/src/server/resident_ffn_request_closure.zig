const std = @import("std");
const ops = @import("../ops/ops.zig");

pub const Snapshot = struct {
    request: u64,
    owner_id: u64,
    counters: ops.NativeQuantTimingStats,
};

pub const Evidence = struct {
    schema: []const u8 = "antfly.resident-ffn-request-counters.v1",
    request: u64,
    owner_id: u64,
    decode_steps: u64,
    transformer_layers: u64,
    direct_successes: u64,
    direct_fallbacks: u64,
    backend_fallbacks: u64,
    runtime_failures: u64,
    e4_attempts: u64,
    e4_prefill_hits: u64,
    e4_pair_dispatches: u64,
    e4_down_dispatches: u64,
    e4_fallbacks: u64,
    e4_fatal_failures: u64,
};

fn delta(after: u64, before: u64) !u64 {
    if (after < before) return error.InvalidResidentFfnCounters;
    return after - before;
}

pub fn close(before: Snapshot, after: Snapshot, raw_output_tokens: usize, transformer_layers: usize) !Evidence {
    if (before.request == 0 or before.request != after.request or before.owner_id == 0 or
        before.owner_id != after.owner_id or raw_output_tokens < 2 or transformer_layers != 42)
        return error.InvalidResidentFfnCounters;
    const decode_steps: u64 = @intCast(raw_output_tokens - 1);
    const layers: u64 = @intCast(transformer_layers);
    const e4_hits = try delta(after.counters.e4_fused_half_hits, before.counters.e4_fused_half_hits);
    const e4_attempts = try delta(after.counters.e4_fused_half_attempts, before.counters.e4_fused_half_attempts);
    const direct = try delta(after.counters.quantized_gated_ffn_direct_successes, before.counters.quantized_gated_ffn_direct_successes);
    const expected_decode = std.math.mul(u64, decode_steps, layers) catch return error.InvalidResidentFfnCounters;
    const pair = try delta(after.counters.e4_fused_half_pair_dispatches, before.counters.e4_fused_half_pair_dispatches);
    const down = try delta(after.counters.e4_fused_half_down_dispatches, before.counters.e4_fused_half_down_dispatches);
    const direct_fallbacks = try delta(after.counters.quantized_gated_ffn_direct_fallbacks, before.counters.quantized_gated_ffn_direct_fallbacks);
    const backend_fallbacks = try delta(after.counters.quantized_gated_ffn_backend_fallbacks, before.counters.quantized_gated_ffn_backend_fallbacks);
    const runtime_failures = try delta(after.counters.quantized_gated_ffn_runtime_failures, before.counters.quantized_gated_ffn_runtime_failures);
    const e4_fallbacks = try delta(after.counters.e4_fused_half_fallbacks, before.counters.e4_fused_half_fallbacks);
    const e4_fatal = try delta(after.counters.e4_fused_half_fatal_failures, before.counters.e4_fused_half_fatal_failures);
    // An appended-token-free checkpoint resume restores the saved logits. The
    // first output token is sampled from that cache without a model forward;
    // only raw_output_tokens-1 singleton decode forwards execute here. M64
    // source-prefill execution is authenticated by the quality Report, not by
    // inventing a prefill call inside this request-scoped delta.
    if (direct != expected_decode or e4_attempts != expected_decode or e4_hits != 0 or pair != 0 or down != 0 or
        direct_fallbacks != 0 or backend_fallbacks != 0 or runtime_failures != 0 or
        e4_fallbacks != expected_decode or e4_fatal != 0) return error.InvalidResidentFfnCounters;
    return .{
        .request = before.request,
        .owner_id = before.owner_id,
        .decode_steps = decode_steps,
        .transformer_layers = layers,
        .direct_successes = direct,
        .direct_fallbacks = direct_fallbacks,
        .backend_fallbacks = backend_fallbacks,
        .runtime_failures = runtime_failures,
        .e4_attempts = e4_attempts,
        .e4_prefill_hits = e4_hits,
        .e4_pair_dispatches = pair,
        .e4_down_dispatches = down,
        .e4_fallbacks = e4_fallbacks,
        .e4_fatal_failures = e4_fatal,
    };
}

test "resident FFN closure binds cached first token and singleton decode only" {
    var before: Snapshot = .{ .request = 7, .owner_id = 9, .counters = .{} };
    var after = before;
    after.counters.quantized_gated_ffn_direct_successes = 42 * 63;
    after.counters.e4_fused_half_attempts = 42 * 63;
    after.counters.e4_fused_half_fallbacks = 42 * 63;
    const result = try close(before, after, 64, 42);
    try std.testing.expectEqual(@as(u64, 63), result.decode_steps);
    inline for (.{
        "direct",
        "attempts",
        "hits",
        "pair",
        "down",
        "fallbacks",
        "fatal",
    }) |mutation| {
        var bad = after;
        if (std.mem.eql(u8, mutation, "direct")) bad.counters.quantized_gated_ffn_direct_successes += 1;
        if (std.mem.eql(u8, mutation, "attempts")) bad.counters.e4_fused_half_attempts += 1;
        if (std.mem.eql(u8, mutation, "hits")) bad.counters.e4_fused_half_hits = 1;
        if (std.mem.eql(u8, mutation, "pair")) bad.counters.e4_fused_half_pair_dispatches = 1;
        if (std.mem.eql(u8, mutation, "down")) bad.counters.e4_fused_half_down_dispatches = 1;
        if (std.mem.eql(u8, mutation, "fallbacks")) bad.counters.e4_fused_half_fallbacks -= 1;
        if (std.mem.eql(u8, mutation, "fatal")) bad.counters.e4_fused_half_fatal_failures = 1;
        try std.testing.expectError(error.InvalidResidentFfnCounters, close(before, bad, 64, 42));
    }
    before.owner_id += 1;
    try std.testing.expectError(error.InvalidResidentFfnCounters, close(before, after, 64, 42));
}
