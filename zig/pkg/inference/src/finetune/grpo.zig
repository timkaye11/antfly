// Copyright 2026 Antfly, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const std = @import("std");

pub const RewardFn = *const fn (
    ctx: *anyopaque,
    prompt_idx: usize,
    completion_tokens: []const i32,
) anyerror!f32;

pub const Rewarder = struct {
    ctx: *anyopaque,
    call: RewardFn,

    pub fn score(self: Rewarder, prompt_idx: usize, tokens: []const i32) !f32 {
        return self.call(self.ctx, prompt_idx, tokens);
    }
};

pub const RewardScale = enum {
    group,
    batch,
    none,
};

pub const LossType = enum {
    grpo,
    bnpo,
    dr_grpo,
    dapo,
};

pub const GRPOConfig = struct {
    group_size: usize = 8,
    clip_epsilon: f32 = 0.2,
    epsilon_high: ?f32 = null,
    kl_coef: f32 = 0.04,
    advantage_eps: f32 = 1e-4,
    scale_rewards: RewardScale = .group,
    loss_type: LossType = .bnpo,
    /// Required by Dr. GRPO's constant denominator. DAPO is equivalent to
    /// BNPO only for one logical group; callers must enforce that admission.
    max_completion_tokens: usize = 0,
    /// Exclude every token from a completion that exhausted its generation
    /// budget without producing EOS. Rewards still include the completion so
    /// group-relative advantages retain the sampled population semantics.
    mask_truncated_completions: bool = false,
    /// Legacy compatibility switch. False maps to `scale_rewards = .none`.
    normalize_advantage: bool = true,
};

pub const AdaptiveKLConfig = struct {
    target: f32,
    horizon: f32,
    min_coef: f32,
    max_coef: f32,
};

/// Proportional KL controller used by the original RLHF/PPO recipe and TRL.
/// `horizon` is measured in admitted optimizer groups in Antfly. Callers use
/// the coefficient returned by `value` for the current group, then call
/// `update` with that group's unweighted mean K3 divergence to obtain the
/// coefficient for the next group.
pub const AdaptiveKLController = struct {
    value: f32,
    config: AdaptiveKLConfig,

    pub fn init(initial_coef: f32, config: AdaptiveKLConfig) !AdaptiveKLController {
        if (!std.math.isFinite(initial_coef) or initial_coef < 0.0 or
            !std.math.isFinite(config.target) or config.target <= 0.0 or
            !std.math.isFinite(config.horizon) or config.horizon <= 0.0 or
            !std.math.isFinite(config.min_coef) or config.min_coef < 0.0 or
            !std.math.isFinite(config.max_coef) or config.max_coef < config.min_coef or
            initial_coef < config.min_coef or initial_coef > config.max_coef)
        {
            return error.InvalidAdaptiveKlConfig;
        }
        return .{ .value = initial_coef, .config = config };
    }

    pub fn update(self: *AdaptiveKLController, current_mean_kl: f32, admitted_groups: usize) !f32 {
        if (!std.math.isFinite(current_mean_kl) or current_mean_kl < 0.0 or admitted_groups == 0) {
            return error.InvalidAdaptiveKlObservation;
        }
        const ratio = @as(f64, current_mean_kl) / @as(f64, self.config.target);
        const proportional_error = std.math.clamp(ratio - 1.0, -0.2, 0.2);
        const multiplier = 1.0 + proportional_error *
            @as(f64, @floatFromInt(admitted_groups)) / @as(f64, self.config.horizon);
        if (!std.math.isFinite(multiplier) or multiplier <= 0.0) {
            return error.InvalidAdaptiveKlUpdate;
        }
        const updated = @as(f64, self.value) * multiplier;
        if (!std.math.isFinite(updated)) return error.InvalidAdaptiveKlUpdate;
        self.value = @floatCast(std.math.clamp(
            updated,
            @as(f64, self.config.min_coef),
            @as(f64, self.config.max_coef),
        ));
        return self.value;
    }
};

pub const Completion = struct {
    prompt_idx: usize,
    tokens: []const i32,
    old_logps: []const f32,
    ref_logps: []const f32,
    truncated: bool = false,
};

pub const GroupAdvantages = struct {
    allocator: std.mem.Allocator,
    rewards: []f32,
    advantages: []f32,
    num_groups: usize,

    pub fn deinit(self: *GroupAdvantages) void {
        self.allocator.free(self.rewards);
        self.allocator.free(self.advantages);
        self.* = undefined;
    }
};

pub fn scoreGroup(
    allocator: std.mem.Allocator,
    rewarder: Rewarder,
    completions: []const Completion,
) !GroupAdvantages {
    const rewards = try allocator.alloc(f32, completions.len);
    errdefer allocator.free(rewards);
    const advantages = try allocator.alloc(f32, completions.len);
    errdefer allocator.free(advantages);

    var max_prompt: usize = 0;
    var any = false;
    for (completions, 0..) |c, i| {
        rewards[i] = try rewarder.score(c.prompt_idx, c.tokens);
        advantages[i] = 0;
        if (!any or c.prompt_idx > max_prompt) {
            max_prompt = c.prompt_idx;
            any = true;
        }
    }
    const num_groups: usize = if (any) max_prompt + 1 else 0;

    return GroupAdvantages{
        .allocator = allocator,
        .rewards = rewards,
        .advantages = advantages,
        .num_groups = num_groups,
    };
}

pub fn computeAdvantages(
    ga: *GroupAdvantages,
    completions: []const Completion,
    config: GRPOConfig,
) void {
    const n = completions.len;
    if (n == 0) return;

    const reward_scale: RewardScale = if (config.normalize_advantage)
        config.scale_rewards
    else
        .none;
    var batch_std: f64 = 0.0;
    if (reward_scale == .batch and ga.rewards.len > 1) {
        var reward_sum: f64 = 0.0;
        for (ga.rewards) |reward| reward_sum += reward;
        const reward_mean = reward_sum / @as(f64, @floatFromInt(ga.rewards.len));
        var variance_sum: f64 = 0.0;
        for (ga.rewards) |reward| {
            const delta = @as(f64, reward) - reward_mean;
            variance_sum += delta * delta;
        }
        batch_std = @sqrt(
            variance_sum / @as(f64, @floatFromInt(ga.rewards.len - 1)),
        );
    }

    var g: usize = 0;
    while (g < ga.num_groups) : (g += 1) {
        var count: usize = 0;
        var sum: f64 = 0;
        for (completions, 0..) |c, i| {
            if (c.prompt_idx == g) {
                sum += ga.rewards[i];
                count += 1;
            }
        }
        if (count == 0) continue;
        const mean: f64 = sum / @as(f64, @floatFromInt(count));

        var std_val: f64 = 0;
        if (reward_scale == .group) {
            var var_sum: f64 = 0;
            for (completions, 0..) |c, i| {
                if (c.prompt_idx == g) {
                    const d = @as(f64, ga.rewards[i]) - mean;
                    var_sum += d * d;
                }
            }
            // Match the canonical GRPO implementations' unbiased sample
            // standard deviation. A singleton group has no reward variation
            // and therefore receives exactly zero centered advantage below.
            const variance = if (count > 1)
                var_sum / @as(f64, @floatFromInt(count - 1))
            else
                0.0;
            std_val = @sqrt(variance);
        }

        for (completions, 0..) |c, i| {
            if (c.prompt_idx == g) {
                const centered = @as(f64, ga.rewards[i]) - mean;
                if (reward_scale != .none) {
                    if (reward_scale == .batch) std_val = batch_std;
                    const denom = std_val + @as(f64, config.advantage_eps);
                    ga.advantages[i] = @floatCast(centered / denom);
                } else {
                    ga.advantages[i] = @floatCast(centered);
                }
            }
        }
    }
}

/// Exact reward-variation predicate used to skip groups that cannot produce a
/// policy-gradient signal. Reward providers are required to emit finite f32s;
/// equality is therefore stable across checkpoint resume and report replay.
pub fn rewardsHaveVariation(rewards: []const f32) bool {
    if (rewards.len < 2) return false;
    const first = rewards[0];
    for (rewards[1..]) |reward| {
        if (reward != first) return true;
    }
    return false;
}

pub const GRPOLossResult = struct {
    loss: f32,
    pg_loss: f32,
    kl_loss: f32,
    mean_kl: f32,
    clip_fraction: f32,
    grad_new_logps: []f32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *GRPOLossResult) void {
        self.allocator.free(self.grad_new_logps);
        self.* = undefined;
    }
};

pub fn grpoLoss(
    allocator: std.mem.Allocator,
    completions: []const Completion,
    new_logps: []const f32,
    advantages: []const f32,
    config: GRPOConfig,
) !GRPOLossResult {
    if (!std.math.isFinite(config.kl_coef) or config.kl_coef < 0.0) {
        return error.InvalidGrpoKlCoefficient;
    }
    var total_tokens: usize = 0;
    var active_tokens: usize = 0;
    for (completions) |c| {
        if (c.tokens.len == 0) return error.EmptyCompletion;
        if (c.old_logps.len != c.tokens.len or c.ref_logps.len != c.tokens.len) {
            return error.LogpLenMismatch;
        }
        for (c.old_logps) |logp| {
            if (!std.math.isFinite(logp)) return error.NonFiniteGrpoLogprob;
        }
        for (c.ref_logps) |logp| {
            if (!std.math.isFinite(logp)) return error.NonFiniteGrpoLogprob;
        }
        total_tokens = std.math.add(usize, total_tokens, c.tokens.len) catch
            return error.TokenCountOverflow;
        if (!config.mask_truncated_completions or !c.truncated) {
            active_tokens = std.math.add(usize, active_tokens, c.tokens.len) catch
                return error.TokenCountOverflow;
        }
    }

    if (new_logps.len != total_tokens) return error.LogpLenMismatch;
    if (advantages.len != completions.len) return error.AdvLenMismatch;
    for (new_logps) |logp| {
        if (!std.math.isFinite(logp)) return error.NonFiniteGrpoLogprob;
    }
    for (advantages) |advantage| {
        if (!std.math.isFinite(advantage)) return error.NonFiniteGrpoAdvantage;
    }

    const grad = try allocator.alloc(f32, total_tokens);
    errdefer allocator.free(grad);
    @memset(grad, 0);

    if (active_tokens == 0) {
        return GRPOLossResult{
            .loss = 0,
            .pg_loss = 0,
            .kl_loss = 0,
            .mean_kl = 0,
            .clip_fraction = 0,
            .grad_new_logps = grad,
            .allocator = allocator,
        };
    }

    const n_f: f32 = @floatFromInt(active_tokens);
    const eps_low = config.clip_epsilon;
    const eps_high = config.epsilon_high orelse eps_low;
    if (!std.math.isFinite(eps_low) or eps_low <= 0.0 or eps_low > 1.0 or
        !std.math.isFinite(eps_high) or eps_high <= 0.0 or eps_high > 1.0)
    {
        return error.InvalidGrpoClipEpsilon;
    }
    if (config.loss_type == .dr_grpo) {
        if (config.max_completion_tokens == 0) return error.InvalidMaxCompletionTokens;
        for (completions) |completion| {
            if (completion.tokens.len > config.max_completion_tokens) {
                return error.CompletionExceedsConfiguredMaximum;
            }
        }
    }
    const kl = config.kl_coef;

    var pg_sum: f64 = 0;
    var raw_kl_sum: f64 = 0;
    var kl_sum: f64 = 0;
    var clipped_count: usize = 0;

    var off: usize = 0;
    for (completions, 0..) |c, ci| {
        const adv: f32 = advantages[ci];
        const completion_masked = config.mask_truncated_completions and c.truncated;
        const token_weight: f32 = switch (config.loss_type) {
            .grpo => 1.0 / (@as(f32, @floatFromInt(completions.len)) * @as(f32, @floatFromInt(c.tokens.len))),
            .bnpo, .dapo => 1.0 / n_f,
            .dr_grpo => 1.0 / (@as(f32, @floatFromInt(completions.len)) * @as(f32, @floatFromInt(config.max_completion_tokens))),
        };
        var t: usize = 0;
        while (t < c.tokens.len) : (t += 1) {
            if (completion_masked) continue;
            const new_lp = new_logps[off + t];
            const old_lp = c.old_logps[t];
            const ref_lp = c.ref_logps[t];

            const policy_log_ratio = new_lp - old_lp;
            if (!std.math.isFinite(policy_log_ratio)) return error.GrpoPolicyLogRatioOutOfRange;
            const ratio = @exp(policy_log_ratio);
            if (!std.math.isFinite(ratio)) return error.GrpoPolicyLogRatioOutOfRange;
            const pg_1 = ratio * adv;
            const clipped_ratio = std.math.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high);
            const pg_2 = clipped_ratio * adv;
            if (!std.math.isFinite(pg_1) or !std.math.isFinite(pg_2)) {
                return error.NonFiniteGrpoComputation;
            }

            // -min(pg_1, pg_2)
            const chosen = if (pg_1 < pg_2) pg_1 else pg_2;
            const pg_token = -chosen;
            pg_sum += @as(f64, pg_token) * @as(f64, token_weight);

            // KL k3: exp(ref - new) - (ref - new) - 1
            const diff = ref_lp - new_lp;
            if (!std.math.isFinite(diff) or diff > 80.0) return error.GrpoKlLogRatioOutOfRange;
            // expm1 keeps K3 and its gradient stable when the policy remains
            // close to the reference: exp(diff) - diff - 1 is otherwise a
            // cancellation-prone subtraction around zero.
            const expm1_diff = std.math.expm1(diff);
            if (!std.math.isFinite(expm1_diff)) return error.GrpoKlLogRatioOutOfRange;
            const k3 = @max(expm1_diff - diff, 0.0);
            raw_kl_sum += k3;
            kl_sum += @as(f64, kl) * @as(f64, k3) * @as(f64, token_weight);

            // Gradient w.r.t. new_lp.
            //
            // PG branch:
            //   If pg_1 <= pg_2 (unclipped chosen) OR clip is inactive,
            //   d(-pg_1)/d(new_lp) = -ratio * adv.
            //   If pg_2 < pg_1, clip binds and grad is 0.
            var g_pg: f32 = 0;
            const clip_binds = pg_2 < pg_1;
            if (clip_binds) {
                clipped_count += 1;
                g_pg = 0;
            } else {
                g_pg = -ratio * adv;
            }

            // KL branch:
            //   d k3 / d new_lp = d/d new_lp [exp(ref - new) - (ref - new) - 1]
            //                   = -exp(ref - new) + 1
            //                   = 1 - exp(ref - new)
            //   Loss contribution is +kl_coef * k3, so grad is +kl_coef * (1 - exp_diff).
            const g_kl: f32 = -kl * expm1_diff;
            if (!std.math.isFinite(g_pg) or !std.math.isFinite(g_kl)) {
                return error.NonFiniteGrpoComputation;
            }

            const token_gradient = (g_pg + g_kl) * token_weight;
            if (!std.math.isFinite(token_gradient)) return error.NonFiniteGrpoComputation;
            grad[off + t] = token_gradient;
        }
        off += c.tokens.len;
    }

    const f32_max: f64 = std.math.floatMax(f32);
    const loss_sum = pg_sum + kl_sum;
    if (!std.math.isFinite(pg_sum) or !std.math.isFinite(kl_sum) or
        !std.math.isFinite(raw_kl_sum) or @abs(pg_sum) > f32_max or
        @abs(kl_sum) > f32_max or !std.math.isFinite(loss_sum) or
        @abs(loss_sum) > f32_max or
        raw_kl_sum > f32_max * @as(f64, n_f))
    {
        return error.NonFiniteGrpoComputation;
    }
    const pg_loss: f32 = @floatCast(pg_sum);
    const kl_loss: f32 = @floatCast(kl_sum);
    const mean_kl: f32 = @floatCast(raw_kl_sum / @as(f64, n_f));
    const loss: f32 = @floatCast(loss_sum);
    const clip_fraction: f32 = @as(f32, @floatFromInt(clipped_count)) / n_f;

    return GRPOLossResult{
        .loss = loss,
        .pg_loss = pg_loss,
        .kl_loss = kl_loss,
        .mean_kl = mean_kl,
        .clip_fraction = clip_fraction,
        .grad_new_logps = grad,
        .allocator = allocator,
    };
}

// -------------------- tests --------------------

const testing = std.testing;

const ConstRewardCtx = struct { value: f32 };

fn constReward(
    ctx: *anyopaque,
    prompt_idx: usize,
    completion_tokens: []const i32,
) anyerror!f32 {
    _ = prompt_idx;
    _ = completion_tokens;
    const self: *ConstRewardCtx = @ptrCast(@alignCast(ctx));
    return self.value;
}

test "scoreGroup constant reward" {
    const alloc = testing.allocator;
    var ctx = ConstRewardCtx{ .value = 1.25 };
    const rewarder = Rewarder{ .ctx = &ctx, .call = constReward };

    const tokens = [_]i32{ 1, 2, 3 };
    const lp = [_]f32{ -0.1, -0.2, -0.3 };
    const comps = [_]Completion{
        .{ .prompt_idx = 0, .tokens = &tokens, .old_logps = &lp, .ref_logps = &lp },
        .{ .prompt_idx = 0, .tokens = &tokens, .old_logps = &lp, .ref_logps = &lp },
        .{ .prompt_idx = 1, .tokens = &tokens, .old_logps = &lp, .ref_logps = &lp },
    };

    var ga = try scoreGroup(alloc, rewarder, &comps);
    defer ga.deinit();

    try testing.expectEqual(@as(usize, 2), ga.num_groups);
    for (ga.rewards) |r| try testing.expectApproxEqAbs(@as(f32, 1.25), r, 1e-6);
}

test "computeAdvantages equal rewards -> zero" {
    const alloc = testing.allocator;
    var ctx = ConstRewardCtx{ .value = 2.0 };
    const rewarder = Rewarder{ .ctx = &ctx, .call = constReward };

    const tokens = [_]i32{1};
    const lp = [_]f32{-0.5};
    const comps = [_]Completion{
        .{ .prompt_idx = 0, .tokens = &tokens, .old_logps = &lp, .ref_logps = &lp },
        .{ .prompt_idx = 0, .tokens = &tokens, .old_logps = &lp, .ref_logps = &lp },
        .{ .prompt_idx = 0, .tokens = &tokens, .old_logps = &lp, .ref_logps = &lp },
    };

    var ga = try scoreGroup(alloc, rewarder, &comps);
    defer ga.deinit();

    const cfg = GRPOConfig{};
    computeAdvantages(&ga, &comps, cfg);
    for (ga.advantages) |a| try testing.expectApproxEqAbs(@as(f32, 0), a, 1e-6);
}

const ArrayRewardCtx = struct { values: []const f32 };

fn arrayReward(
    ctx: *anyopaque,
    prompt_idx: usize,
    completion_tokens: []const i32,
) anyerror!f32 {
    _ = prompt_idx;
    _ = completion_tokens;
    const self: *ArrayRewardCtx = @ptrCast(@alignCast(ctx));
    // caller uses a counter stored in values... simpler: we'll return values[0] then shift.
    // Instead use index trick below.
    return self.values[0];
}

const IndexedRewardCtx = struct {
    values: []const f32,
    index: usize = 0,
};

fn indexedReward(
    ctx: *anyopaque,
    prompt_idx: usize,
    completion_tokens: []const i32,
) anyerror!f32 {
    _ = prompt_idx;
    _ = completion_tokens;
    const self: *IndexedRewardCtx = @ptrCast(@alignCast(ctx));
    const v = self.values[self.index];
    self.index += 1;
    return v;
}

test "computeAdvantages uses unbiased reward standard deviation" {
    const alloc = testing.allocator;
    const values = [_]f32{ 1.0, 3.0 };
    var ctx = IndexedRewardCtx{ .values = &values };
    const rewarder = Rewarder{ .ctx = &ctx, .call = indexedReward };

    const tokens = [_]i32{1};
    const lp = [_]f32{0.0};
    const comps = [_]Completion{
        .{ .prompt_idx = 0, .tokens = &tokens, .old_logps = &lp, .ref_logps = &lp },
        .{ .prompt_idx = 0, .tokens = &tokens, .old_logps = &lp, .ref_logps = &lp },
    };

    var ga = try scoreGroup(alloc, rewarder, &comps);
    defer ga.deinit();

    const cfg = GRPOConfig{};
    computeAdvantages(&ga, &comps, cfg);
    const expected = @as(f32, @floatCast(1.0 / @sqrt(2.0)));
    try testing.expectApproxEqAbs(-expected, ga.advantages[0], 1e-4);
    try testing.expectApproxEqAbs(expected, ga.advantages[1], 1e-4);
    try testing.expect(rewardsHaveVariation(ga.rewards));
}

test "uniform reward groups are explicitly zero variance without NaN" {
    const rewards = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    try testing.expect(!rewardsHaveVariation(&rewards));
    const tokens = [_]i32{1};
    const logps = [_]f32{-0.5};
    const completions = [_]Completion{
        .{ .prompt_idx = 0, .tokens = &tokens, .old_logps = &logps, .ref_logps = &logps },
        .{ .prompt_idx = 0, .tokens = &tokens, .old_logps = &logps, .ref_logps = &logps },
        .{ .prompt_idx = 0, .tokens = &tokens, .old_logps = &logps, .ref_logps = &logps },
        .{ .prompt_idx = 0, .tokens = &tokens, .old_logps = &logps, .ref_logps = &logps },
    };
    var advantages = GroupAdvantages{
        .allocator = testing.allocator,
        .rewards = try testing.allocator.dupe(f32, &rewards),
        .advantages = try testing.allocator.alloc(f32, rewards.len),
        .num_groups = 1,
    };
    defer advantages.deinit();
    @memset(advantages.advantages, std.math.nan(f32));
    computeAdvantages(&advantages, &completions, .{});
    for (advantages.advantages) |advantage| {
        try testing.expectEqual(@as(f32, 0.0), advantage);
        try testing.expect(std.math.isFinite(advantage));
    }
}

test "computeAdvantages supports batch and none reward scaling" {
    const rewards = [_]f32{ 0.0, 2.0, 10.0, 14.0 };
    const tokens = [_]i32{1};
    const logps = [_]f32{-0.5};
    const completions = [_]Completion{
        .{ .prompt_idx = 0, .tokens = &tokens, .old_logps = &logps, .ref_logps = &logps },
        .{ .prompt_idx = 0, .tokens = &tokens, .old_logps = &logps, .ref_logps = &logps },
        .{ .prompt_idx = 1, .tokens = &tokens, .old_logps = &logps, .ref_logps = &logps },
        .{ .prompt_idx = 1, .tokens = &tokens, .old_logps = &logps, .ref_logps = &logps },
    };
    var advantages = GroupAdvantages{
        .allocator = testing.allocator,
        .rewards = try testing.allocator.dupe(f32, &rewards),
        .advantages = try testing.allocator.alloc(f32, rewards.len),
        .num_groups = 2,
    };
    defer advantages.deinit();

    computeAdvantages(&advantages, &completions, .{ .scale_rewards = .none });
    try testing.expectEqualSlices(f32, &.{ -1.0, 1.0, -2.0, 2.0 }, advantages.advantages);

    computeAdvantages(&advantages, &completions, .{ .scale_rewards = .batch });
    const batch_std = @sqrt(@as(f32, 131.0 / 3.0));
    try testing.expectApproxEqAbs(-1.0 / (batch_std + 1e-4), advantages.advantages[0], 1e-6);
    try testing.expectApproxEqAbs(2.0 / (batch_std + 1e-4), advantages.advantages[3], 1e-6);
}

test "grpoLoss zero when ratio=1, adv=0, ref=new" {
    const alloc = testing.allocator;
    const tokens = [_]i32{ 1, 2 };
    const lp = [_]f32{ -0.3, -0.7 };
    const comps = [_]Completion{
        .{ .prompt_idx = 0, .tokens = &tokens, .old_logps = &lp, .ref_logps = &lp },
        .{ .prompt_idx = 0, .tokens = &tokens, .old_logps = &lp, .ref_logps = &lp },
    };
    const new_lp = [_]f32{ -0.3, -0.7, -0.3, -0.7 };
    const advs = [_]f32{ 0.0, 0.0 };

    var res = try grpoLoss(alloc, &comps, &new_lp, &advs, .{});
    defer res.deinit();

    try testing.expectApproxEqAbs(@as(f32, 0), res.loss, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), res.pg_loss, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), res.kl_loss, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), res.mean_kl, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), res.clip_fraction, 1e-6);
    for (res.grad_new_logps) |g| try testing.expectApproxEqAbs(@as(f32, 0), g, 1e-6);
}

test "grpoLoss rejects completion logprob length mismatches" {
    const tokens = [_]i32{ 1, 2 };
    const short_logps = [_]f32{-0.3};
    const full_logps = [_]f32{ -0.3, -0.7 };
    const new_logps = [_]f32{ -0.3, -0.7 };
    const advantages = [_]f32{0.0};

    const short_old = [_]Completion{.{
        .prompt_idx = 0,
        .tokens = &tokens,
        .old_logps = &short_logps,
        .ref_logps = &full_logps,
    }};
    try testing.expectError(
        error.LogpLenMismatch,
        grpoLoss(testing.allocator, &short_old, &new_logps, &advantages, .{}),
    );

    const short_reference = [_]Completion{.{
        .prompt_idx = 0,
        .tokens = &tokens,
        .old_logps = &full_logps,
        .ref_logps = &short_logps,
    }};
    try testing.expectError(
        error.LogpLenMismatch,
        grpoLoss(testing.allocator, &short_reference, &new_logps, &advantages, .{}),
    );
}

test "grpoLoss rejects invalid coefficients and non-finite inputs" {
    const tokens = [_]i32{1};
    const finite_logps = [_]f32{-0.3};
    const finite_advantages = [_]f32{1.0};
    const completions = [_]Completion{.{
        .prompt_idx = 0,
        .tokens = &tokens,
        .old_logps = &finite_logps,
        .ref_logps = &finite_logps,
    }};

    try testing.expectError(
        error.InvalidGrpoKlCoefficient,
        grpoLoss(testing.allocator, &completions, &finite_logps, &finite_advantages, .{ .kl_coef = -0.1 }),
    );
    try testing.expectError(
        error.InvalidGrpoKlCoefficient,
        grpoLoss(testing.allocator, &completions, &finite_logps, &finite_advantages, .{ .kl_coef = std.math.nan(f32) }),
    );

    const non_finite = [_]f32{std.math.nan(f32)};
    try testing.expectError(
        error.NonFiniteGrpoLogprob,
        grpoLoss(testing.allocator, &completions, &non_finite, &finite_advantages, .{}),
    );
    try testing.expectError(
        error.NonFiniteGrpoAdvantage,
        grpoLoss(testing.allocator, &completions, &finite_logps, &non_finite, .{}),
    );

    const overflowing_new = [_]f32{std.math.floatMax(f32)};
    const overflowing_old = [_]f32{-std.math.floatMax(f32)};
    const overflowing_completions = [_]Completion{.{
        .prompt_idx = 0,
        .tokens = &tokens,
        .old_logps = &overflowing_old,
        .ref_logps = &finite_logps,
    }};
    try testing.expectError(
        error.GrpoPolicyLogRatioOutOfRange,
        grpoLoss(testing.allocator, &overflowing_completions, &overflowing_new, &finite_advantages, .{}),
    );

    // Each component fits in f32 and the opposing gradient terms remain
    // finite, but their positive loss sum does not. Reject before the final
    // f32 conversion rather than returning +inf to the training loop.
    const zero_logps = [_]f32{0.0};
    const positive_reference = [_]f32{@log(@as(f32, 2.0))};
    const combined_overflow_completions = [_]Completion{.{
        .prompt_idx = 0,
        .tokens = &tokens,
        .old_logps = &zero_logps,
        .ref_logps = &positive_reference,
    }};
    const large_negative_advantage = [_]f32{-2.5e38};
    try testing.expectError(
        error.NonFiniteGrpoComputation,
        grpoLoss(
            testing.allocator,
            &combined_overflow_completions,
            &zero_logps,
            &large_negative_advantage,
            .{ .kl_coef = 3.3e38 },
        ),
    );
}

test "grpoLoss exposes unweighted stable mean K3 when beta is zero" {
    const alloc = testing.allocator;
    const tokens = [_]i32{1};
    const old = [_]f32{-1.0};
    const reference = [_]f32{-0.5};
    const completions = [_]Completion{
        .{ .prompt_idx = 0, .tokens = &tokens, .old_logps = &old, .ref_logps = &reference },
    };
    const new_logps = [_]f32{-1.0};
    const advantages = [_]f32{0.0};

    var result = try grpoLoss(
        alloc,
        &completions,
        &new_logps,
        &advantages,
        .{ .kl_coef = 0.0 },
    );
    defer result.deinit();

    const expected = std.math.expm1(@as(f32, 0.5)) - 0.5;
    try testing.expectApproxEqAbs(expected, result.mean_kl, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), result.kl_loss, 1e-6);
}

test "grpoLoss implements documented normalization modes" {
    const alloc = testing.allocator;
    const short_tokens = [_]i32{1};
    const long_tokens = [_]i32{ 2, 3, 4 };
    const short_logps = [_]f32{-1.0};
    const long_logps = [_]f32{ -1.0, -1.0, -1.0 };
    const completions = [_]Completion{
        .{ .prompt_idx = 0, .tokens = &short_tokens, .old_logps = &short_logps, .ref_logps = &short_logps },
        .{ .prompt_idx = 0, .tokens = &long_tokens, .old_logps = &long_logps, .ref_logps = &long_logps },
    };
    const new_logps = [_]f32{ -1.0, -1.0, -1.0, -1.0 };
    const advantages = [_]f32{ 1.0, 1.0 };

    var sequence = try grpoLoss(alloc, &completions, &new_logps, &advantages, .{ .loss_type = .grpo });
    defer sequence.deinit();
    try testing.expectApproxEqAbs(@as(f32, -1.0), sequence.pg_loss, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -0.5), sequence.grad_new_logps[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -1.0 / 6.0), sequence.grad_new_logps[1], 1e-6);

    var batch = try grpoLoss(alloc, &completions, &new_logps, &advantages, .{ .loss_type = .bnpo });
    defer batch.deinit();
    try testing.expectApproxEqAbs(@as(f32, -1.0), batch.pg_loss, 1e-6);
    for (batch.grad_new_logps) |gradient| {
        try testing.expectApproxEqAbs(@as(f32, -0.25), gradient, 1e-6);
    }

    var dapo = try grpoLoss(alloc, &completions, &new_logps, &advantages, .{ .loss_type = .dapo });
    defer dapo.deinit();
    try testing.expectApproxEqAbs(batch.pg_loss, dapo.pg_loss, 1e-6);

    var dimension_reduced = try grpoLoss(alloc, &completions, &new_logps, &advantages, .{
        .loss_type = .dr_grpo,
        .max_completion_tokens = 4,
    });
    defer dimension_reduced.deinit();
    try testing.expectApproxEqAbs(@as(f32, -0.5), dimension_reduced.pg_loss, 1e-6);
    for (dimension_reduced.grad_new_logps) |gradient| {
        try testing.expectApproxEqAbs(@as(f32, -0.125), gradient, 1e-6);
    }
}

test "grpoLoss masks every token from truncated completions" {
    const alloc = testing.allocator;
    const truncated_tokens = [_]i32{ 1, 2 };
    const terminated_tokens = [_]i32{ 3, 4 };
    const logps = [_]f32{ -1.0, -1.0 };
    const completions = [_]Completion{
        .{
            .prompt_idx = 0,
            .tokens = &truncated_tokens,
            .old_logps = &logps,
            .ref_logps = &logps,
            .truncated = true,
        },
        .{
            .prompt_idx = 0,
            .tokens = &terminated_tokens,
            .old_logps = &logps,
            .ref_logps = &logps,
        },
    };
    const new_logps = [_]f32{ -1.0, -1.0, -1.0, -1.0 };
    const advantages = [_]f32{ 1.0, -1.0 };

    var result = try grpoLoss(alloc, &completions, &new_logps, &advantages, .{
        .loss_type = .bnpo,
        .mask_truncated_completions = true,
    });
    defer result.deinit();
    try testing.expectApproxEqAbs(@as(f32, 1.0), result.pg_loss, 1e-6);
    try testing.expectEqual(@as(f32, 0.0), result.grad_new_logps[0]);
    try testing.expectEqual(@as(f32, 0.0), result.grad_new_logps[1]);
    try testing.expectApproxEqAbs(@as(f32, 0.5), result.grad_new_logps[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), result.grad_new_logps[3], 1e-6);

    const all_truncated = [_]Completion{
        .{
            .prompt_idx = 0,
            .tokens = &truncated_tokens,
            .old_logps = &logps,
            .ref_logps = &logps,
            .truncated = true,
        },
    };
    var empty = try grpoLoss(alloc, &all_truncated, logps[0..], &[_]f32{1.0}, .{
        .mask_truncated_completions = true,
    });
    defer empty.deinit();
    try testing.expectEqual(@as(f32, 0.0), empty.loss);
    for (empty.grad_new_logps) |gradient| try testing.expectEqual(@as(f32, 0.0), gradient);
}

test "grpoLoss uses asymmetric high clipping" {
    const alloc = testing.allocator;
    const tokens = [_]i32{1};
    const old_logps = [_]f32{0.0};
    const ref_logps = [_]f32{@log(@as(f32, 1.25))};
    const completions = [_]Completion{
        .{ .prompt_idx = 0, .tokens = &tokens, .old_logps = &old_logps, .ref_logps = &ref_logps },
    };
    const new_logps = [_]f32{@log(@as(f32, 1.25))};
    const advantages = [_]f32{1.0};
    var result = try grpoLoss(alloc, &completions, &new_logps, &advantages, .{
        .clip_epsilon = 0.2,
        .epsilon_high = 0.1,
        .kl_coef = 0.0,
    });
    defer result.deinit();
    try testing.expectApproxEqAbs(@as(f32, -1.1), result.pg_loss, 1e-6);
    try testing.expectEqual(@as(f32, 0.0), result.grad_new_logps[0]);
    try testing.expectEqual(@as(f32, 1.0), result.clip_fraction);
}

test "adaptive KL controller is bounded and updates the next-group coefficient" {
    var controller = try AdaptiveKLController.init(0.04, .{
        .target = 0.01,
        .horizon = 100.0,
        .min_coef = 0.001,
        .max_coef = 0.05,
    });
    const below_target = try controller.update(0.0, 1);
    try testing.expectApproxEqAbs(@as(f32, 0.03992), below_target, 1e-7);
    const above_target = try controller.update(1.0, 1);
    try testing.expect(above_target > below_target);

    var upper = try AdaptiveKLController.init(0.05, .{
        .target = 0.01,
        .horizon = 1.0,
        .min_coef = 0.001,
        .max_coef = 0.05,
    });
    try testing.expectEqual(@as(f32, 0.05), try upper.update(1.0, 1));
}

fn lossOnly(
    alloc: std.mem.Allocator,
    comps: []const Completion,
    new_lp: []const f32,
    advs: []const f32,
    cfg: GRPOConfig,
) !f32 {
    var res = try grpoLoss(alloc, comps, new_lp, advs, cfg);
    defer res.deinit();
    return res.loss;
}

test "grpoLoss finite-difference gradient check" {
    const alloc = testing.allocator;
    const tokens0 = [_]i32{ 5, 6 };
    const tokens1 = [_]i32{ 7, 8, 9 };
    const old0 = [_]f32{ -0.4, -0.9 };
    const ref0 = [_]f32{ -0.5, -1.0 };
    const old1 = [_]f32{ -0.2, -0.8, -1.1 };
    const ref1 = [_]f32{ -0.3, -0.7, -1.2 };

    const comps = [_]Completion{
        .{ .prompt_idx = 0, .tokens = &tokens0, .old_logps = &old0, .ref_logps = &ref0 },
        .{ .prompt_idx = 0, .tokens = &tokens1, .old_logps = &old1, .ref_logps = &ref1 },
    };
    var new_lp = [_]f32{ -0.35, -0.85, -0.25, -0.82, -1.05 };
    const advs = [_]f32{ 0.7, -0.3 };

    const cfg = GRPOConfig{ .clip_epsilon = 0.5, .kl_coef = 0.1 };

    var res = try grpoLoss(alloc, &comps, &new_lp, &advs, cfg);
    defer res.deinit();

    const h: f32 = 1e-3;
    var i: usize = 0;
    while (i < new_lp.len) : (i += 1) {
        const saved = new_lp[i];
        new_lp[i] = saved + h;
        const lp = try lossOnly(alloc, &comps, &new_lp, &advs, cfg);
        new_lp[i] = saved - h;
        const lm = try lossOnly(alloc, &comps, &new_lp, &advs, cfg);
        new_lp[i] = saved;
        const num = (lp - lm) / (2.0 * h);
        try testing.expectApproxEqAbs(num, res.grad_new_logps[i], 5e-3);
    }
}
