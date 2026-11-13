const std = @import("std");
const solver = @import("attention_match.zig");

pub const policy = "train_only_primary_critical_policy_probe_v1|variants=fixed025,calibrated_res16|metric=exact_f32_rounded_train_max_error|tie=fixed025|heldout=forbidden|single_heldout_ladder_after_freeze|candidate_owners=both_admitted";

pub const Variant = enum { fixed_probability_0_025_no_top1, calibrated_floor_0_0025_p91_residual16 };

pub const Candidate = struct {
    variant: Variant,
    variant_policy_sha256: [32]u8,
    source_authority_sha256: [32]u8,
    train_domain_sha256: [32]u8,
    candidate_sha256: [32]u8,
    primary_budget_percent: u8,
    candidate_rows: usize,
    candidate_live_bytes: usize,
    exact_rounded_train_max_error: f64,
    postfit_attempts: usize,
    postfit_completions: usize,
    heldout_observations: usize = 0,
    frozen_before_heldout: bool = true,
};

pub const Config = struct { candidate_live_bytes_cap: usize, control: solver.Control = .{} };

pub const Result = struct {
    chosen_index: usize,
    chosen_variant: Variant,
    chosen_policy_sha256: [32]u8,
    chosen_candidate_sha256: [32]u8,
    exact_rounded_train_max_error: f64,
    simultaneous_candidate_live_bytes: usize,
    evidence_sha256: [32]u8,
    frozen_before_heldout: bool,
};

fn nonzero(value: [32]u8) bool {
    return !std.mem.allEqual(u8, &value, 0);
}

fn validCandidate(candidate: Candidate) bool {
    return nonzero(candidate.variant_policy_sha256) and nonzero(candidate.source_authority_sha256) and
        nonzero(candidate.train_domain_sha256) and nonzero(candidate.candidate_sha256) and
        candidate.primary_budget_percent == 91 and
        candidate.candidate_rows > 0 and candidate.candidate_live_bytes > 0 and
        std.math.isFinite(candidate.exact_rounded_train_max_error) and candidate.exact_rounded_train_max_error >= 0 and
        candidate.postfit_attempts == 1 and candidate.postfit_completions == 1 and
        candidate.heldout_observations == 0 and candidate.frozen_before_heldout;
}

fn preferred(a: Candidate, b: Candidate) bool {
    if (a.exact_rounded_train_max_error < b.exact_rounded_train_max_error) return true;
    if (a.exact_rounded_train_max_error > b.exact_rounded_train_max_error) return false;
    return a.variant == .fixed_probability_0_025_no_top1;
}

pub fn select(candidates: [2]Candidate, cfg: Config) !Result {
    try cfg.control.check();
    if (cfg.candidate_live_bytes_cap == 0 or !validCandidate(candidates[0]) or !validCandidate(candidates[1]) or
        candidates[0].variant != .fixed_probability_0_025_no_top1 or
        candidates[1].variant != .calibrated_floor_0_0025_p91_residual16 or
        !std.mem.eql(u8, &candidates[0].source_authority_sha256, &candidates[1].source_authority_sha256) or
        !std.mem.eql(u8, &candidates[0].train_domain_sha256, &candidates[1].train_domain_sha256) or
        candidates[0].primary_budget_percent != candidates[1].primary_budget_percent) return error.InvalidFitConfig;
    const live = try std.math.add(usize, candidates[0].candidate_live_bytes, candidates[1].candidate_live_bytes);
    if (live > cfg.candidate_live_bytes_cap) return error.CompactionWorkspaceLimit;
    const chosen: usize = if (preferred(candidates[0], candidates[1])) 0 else 1;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(policy);
    hash.update(&candidates[0].source_authority_sha256);
    hash.update(&candidates[0].train_domain_sha256);
    hash.update(&[_]u8{candidates[0].primary_budget_percent});
    for (candidates) |candidate| {
        hash.update(&[_]u8{@backingInt(candidate.variant)});
        hash.update(&candidate.variant_policy_sha256);
        hash.update(&candidate.candidate_sha256);
        hash.update(std.mem.asBytes(&candidate.candidate_rows));
        hash.update(std.mem.asBytes(&candidate.candidate_live_bytes));
        hash.update(std.mem.asBytes(&candidate.exact_rounded_train_max_error));
    }
    hash.update(&[_]u8{@intCast(chosen)});
    hash.update(std.mem.asBytes(&live));
    return .{ .chosen_index = chosen, .chosen_variant = candidates[chosen].variant, .chosen_policy_sha256 = candidates[chosen].variant_policy_sha256, .chosen_candidate_sha256 = candidates[chosen].candidate_sha256, .exact_rounded_train_max_error = candidates[chosen].exact_rounded_train_max_error, .simultaneous_candidate_live_bytes = live, .evidence_sha256 = hash.finalResult(), .frozen_before_heldout = true };
}

fn fixture(variant: Variant, score: f64, byte: u8) Candidate {
    return .{ .variant = variant, .variant_policy_sha256 = @splat(byte), .source_authority_sha256 = @splat(7), .train_domain_sha256 = @splat(8), .candidate_sha256 = @splat(byte + 10), .primary_budget_percent = 91, .candidate_rows = 4796, .candidate_live_bytes = 1024, .exact_rounded_train_max_error = score, .postfit_attempts = 1, .postfit_completions = 1 };
}

test "TRAIN probe chooses smaller exact rounded error and freezes before heldout" {
    const result = try select(.{ fixture(.fixed_probability_0_025_no_top1, 0.048, 1), fixture(.calibrated_floor_0_0025_p91_residual16, 0.054, 2) }, .{ .candidate_live_bytes_cap = 2048 });
    try std.testing.expectEqual(Variant.fixed_probability_0_025_no_top1, result.chosen_variant);
    try std.testing.expect(result.frozen_before_heldout);
    try std.testing.expectEqual(@as(usize, 2048), result.simultaneous_candidate_live_bytes);
}

test "TRAIN probe exact tie prefers canonical fixed 0.025" {
    const result = try select(.{ fixture(.fixed_probability_0_025_no_top1, 0.05, 1), fixture(.calibrated_floor_0_0025_p91_residual16, 0.05, 2) }, .{ .candidate_live_bytes_cap = 2048 });
    try std.testing.expectEqual(@as(usize, 0), result.chosen_index);
    try std.testing.expectError(error.InvalidFitConfig, select(.{ fixture(.calibrated_floor_0_0025_p91_residual16, 0.05, 2), fixture(.fixed_probability_0_025_no_top1, 0.05, 1) }, .{ .candidate_live_bytes_cap = 2048 }));
}

test "TRAIN probe rejects fallback budget candidates" {
    var fallback = fixture(.fixed_probability_0_025_no_top1, 0.04, 1);
    fallback.primary_budget_percent = 95;
    try std.testing.expectError(error.InvalidFitConfig, select(.{ fallback, fixture(.calibrated_floor_0_0025_p91_residual16, 0.05, 2) }, .{ .candidate_live_bytes_cap = 2048 }));
}

test "TRAIN probe rejects heldout observation and mismatched authority" {
    var heldout = fixture(.fixed_probability_0_025_no_top1, 0.04, 1);
    heldout.heldout_observations = 1;
    try std.testing.expectError(error.InvalidFitConfig, select(.{ heldout, fixture(.calibrated_floor_0_0025_p91_residual16, 0.05, 2) }, .{ .candidate_live_bytes_cap = 2048 }));
    var mismatch = fixture(.calibrated_floor_0_0025_p91_residual16, 0.05, 2);
    mismatch.train_domain_sha256 = @splat(9);
    try std.testing.expectError(error.InvalidFitConfig, select(.{ fixture(.fixed_probability_0_025_no_top1, 0.04, 1), mismatch }, .{ .candidate_live_bytes_cap = 2048 }));
}

test "TRAIN probe charges both retained candidate owners" {
    try std.testing.expectError(error.CompactionWorkspaceLimit, select(.{ fixture(.fixed_probability_0_025_no_top1, 0.04, 1), fixture(.calibrated_floor_0_0025_p91_residual16, 0.05, 2) }, .{ .candidate_live_bytes_cap = 2047 }));
}
