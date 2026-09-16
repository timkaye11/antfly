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

//! Pure objective resolution and validation, independent of trainer execution.
const std = @import("std");
const schema = @import("recipe_schema.zig");
const grpo = @import("grpo.zig");
const preference_loss = @import("preference_loss.zig");
const Recipe = schema.Recipe;
const PreferenceConfig = schema.PreferenceConfig;
const GrpoConfig = schema.GrpoConfig;
const GrpoSamplingConfig = schema.GrpoSamplingConfig;
pub const default_grpo_max_completion_tokens: usize = 16;

pub const DpoLossType = enum {
    sigmoid,
    ipo,
    simpo,
};

pub const ResolvedDpoObjectiveConfig = struct {
    loss_type: DpoLossType,
    preference: preference_loss.PreferenceConfig,

    pub fn logprobAggregation(self: ResolvedDpoObjectiveConfig) []const u8 {
        return switch (self.loss_type) {
            .sigmoid => "sum",
            .ipo, .simpo => "completion-token-mean",
        };
    }

    pub fn needsReference(self: ResolvedDpoObjectiveConfig) bool {
        return self.loss_type != .simpo;
    }
};

pub fn resolveDpoObjectiveConfig(config: PreferenceConfig) !ResolvedDpoObjectiveConfig {
    const raw_loss_type = config.loss_type orelse "sigmoid";
    const loss_type: DpoLossType = if (std.ascii.eqlIgnoreCase(raw_loss_type, "sigmoid") or std.ascii.eqlIgnoreCase(raw_loss_type, "dpo"))
        .sigmoid
    else if (std.ascii.eqlIgnoreCase(raw_loss_type, "ipo"))
        .ipo
    else if (std.ascii.eqlIgnoreCase(raw_loss_type, "simpo") or std.ascii.eqlIgnoreCase(raw_loss_type, "sigmoid_norm"))
        .simpo
    else if (std.ascii.eqlIgnoreCase(raw_loss_type, "orpo") or std.ascii.eqlIgnoreCase(raw_loss_type, "cpo") or std.ascii.eqlIgnoreCase(raw_loss_type, "kto"))
        return error.DpoLossTypeNotYetSupported
    else
        return error.InvalidDpoLossType;

    const beta = config.beta orelse 0.1;
    if (!std.math.isFinite(beta) or beta <= 0.0) return error.InvalidDpoBeta;
    const label_smoothing = config.label_smoothing orelse 0.0;
    if (!std.math.isFinite(label_smoothing) or label_smoothing < 0.0 or label_smoothing >= 0.5) {
        return error.InvalidDpoLabelSmoothing;
    }
    if (loss_type != .sigmoid and label_smoothing != 0.0) {
        return error.DpoLabelSmoothingRequiresSigmoid;
    }
    if (config.sft_lambda != null) return error.DpoSftAuxiliaryLossNotYetSupported;

    const simpo_gamma = config.simpo_gamma orelse 0.5;
    if (!std.math.isFinite(simpo_gamma) or simpo_gamma < 0.0) return error.InvalidSimpoGamma;
    const ipo_tau = config.ipo_tau orelse beta;
    if (!std.math.isFinite(ipo_tau) or ipo_tau <= 0.0) return error.InvalidIpoTau;
    switch (loss_type) {
        .sigmoid => if (config.simpo_gamma != null or config.ipo_tau != null) {
            return error.DpoOptionNotUsedByLoss;
        },
        .ipo => if (config.simpo_gamma != null) return error.DpoOptionNotUsedByLoss,
        .simpo => if (config.ipo_tau != null) return error.DpoOptionNotUsedByLoss,
    }

    return .{
        .loss_type = loss_type,
        .preference = .{
            .kind = switch (loss_type) {
                .sigmoid => .dpo,
                .ipo => .ipo,
                .simpo => .simpo,
            },
            .beta = beta,
            .label_smoothing = label_smoothing,
            .simpo_gamma = simpo_gamma,
            .ipo_tau = ipo_tau,
        },
    };
}

pub const ResolvedGrpoSamplingConfig = struct {
    temperature: f32,
    top_p: f32,
    top_k: usize,
};

pub fn resolveGrpoSamplingConfig(config: GrpoConfig) !ResolvedGrpoSamplingConfig {
    const sampling = config.sampling orelse GrpoSamplingConfig{};
    const temperature = sampling.temperature orelse 1.0;
    const top_p = sampling.top_p orelse 1.0;
    const top_k = sampling.top_k orelse 0;
    if (!std.math.isFinite(temperature) or temperature <= 0.0) {
        return error.InvalidGrpoSamplingTemperature;
    }
    if (!std.math.isFinite(top_p) or top_p <= 0.0 or top_p > 1.0) {
        return error.InvalidGrpoSamplingTopP;
    }
    return .{ .temperature = temperature, .top_p = top_p, .top_k = top_k };
}

pub const ResolvedGrpoObjectiveConfig = struct {
    loss_type: grpo.LossType,
    scale_rewards: grpo.RewardScale,
    epsilon_low: f32,
    epsilon_high: f32,
    max_completion_tokens: usize,
    mask_truncated_completions: bool,
};

pub fn parseGrpoLossType(value: []const u8) !grpo.LossType {
    return std.meta.stringToEnum(grpo.LossType, value) orelse
        error.InvalidGrpoLossType;
}

pub fn parseGrpoRewardScale(value: []const u8) !grpo.RewardScale {
    return std.meta.stringToEnum(grpo.RewardScale, value) orelse
        error.InvalidGrpoRewardScale;
}

pub fn resolveGrpoObjectiveConfig(config: GrpoConfig, requested_gradient_accumulation_steps: u32) !ResolvedGrpoObjectiveConfig {
    const epsilon_low = config.clip_epsilon orelse 0.2;
    const epsilon_high = config.epsilon_high orelse epsilon_low;
    if (!std.math.isFinite(epsilon_low) or epsilon_low <= 0.0 or epsilon_low > 1.0 or
        !std.math.isFinite(epsilon_high) or epsilon_high <= 0.0 or epsilon_high > 1.0)
    {
        return error.InvalidGrpoClipEpsilon;
    }
    const loss_type = try parseGrpoLossType(config.loss_type orelse "bnpo");
    if (loss_type == .dapo and requested_gradient_accumulation_steps != 1) {
        // Exact DAPO normalization spans every active token in an accumulation
        // window. The current product loop materializes one group at a time,
        // so only a one-group window has a truthful global denominator.
        return error.GrpoDapoRequiresUnitGradientAccumulation;
    }
    const legacy_normalize = config.normalize_advantage orelse true;
    const scale_rewards = if (config.scale_rewards) |value|
        try parseGrpoRewardScale(value)
    else if (legacy_normalize)
        grpo.RewardScale.group
    else
        grpo.RewardScale.none;
    if (config.scale_rewards != null and !legacy_normalize and scale_rewards != .none) {
        return error.ConflictingGrpoRewardScale;
    }
    const max_completion_tokens = config.max_completion_tokens orelse default_grpo_max_completion_tokens;
    if (max_completion_tokens == 0) return error.InvalidMaxCompletionTokens;
    return .{
        .loss_type = loss_type,
        .scale_rewards = scale_rewards,
        .epsilon_low = epsilon_low,
        .epsilon_high = epsilon_high,
        .max_completion_tokens = max_completion_tokens,
        .mask_truncated_completions = config.mask_truncated_completions orelse false,
    };
}

pub fn resolveGrpoCoreConfig(recipe: Recipe) !grpo.GRPOConfig {
    const objective = try resolveGrpoObjectiveConfig(
        recipe.grpo,
        recipe.optimizer.gradient_accumulation_steps orelse 1,
    );
    return .{
        .group_size = recipe.grpo.group_size orelse 2,
        .clip_epsilon = objective.epsilon_low,
        .epsilon_high = objective.epsilon_high,
        .kl_coef = recipe.grpo.kl_coef orelse 0.04,
        .advantage_eps = recipe.grpo.advantage_eps orelse 1e-4,
        .scale_rewards = objective.scale_rewards,
        .loss_type = objective.loss_type,
        .max_completion_tokens = objective.max_completion_tokens,
        .mask_truncated_completions = objective.mask_truncated_completions,
        .normalize_advantage = true,
    };
}

pub const ResolvedGrpoKlControl = struct {
    pub const BudgetPolicy = enum {
        skip_group,
        abort,
    };

    train_max_kl: f32,
    budget_policy: BudgetPolicy,
    adaptive: bool,
    target_kl: ?f32,
    kl_horizon: ?f32,
    min_kl_coef: ?f32,
    max_kl_coef: ?f32,
};

pub fn resolveGrpoKlControl(config: GrpoConfig) !ResolvedGrpoKlControl {
    const train_max_kl = config.train_max_kl orelse 0.1;
    if (!std.math.isFinite(train_max_kl) or train_max_kl <= 0.0) {
        return error.InvalidGrpoTrainKlBudget;
    }
    const budget_policy = std.meta.stringToEnum(
        ResolvedGrpoKlControl.BudgetPolicy,
        config.train_max_kl_policy orelse "skip_group",
    ) orelse return error.InvalidGrpoTrainKlPolicy;

    const adaptive = config.adaptive_kl orelse false;
    if (!adaptive) {
        if (config.target_kl != null or config.kl_horizon != null or
            config.min_kl_coef != null or config.max_kl_coef != null)
        {
            return error.IncompleteGrpoAdaptiveKlConfig;
        }
        return .{
            .train_max_kl = train_max_kl,
            .budget_policy = budget_policy,
            .adaptive = false,
            .target_kl = null,
            .kl_horizon = null,
            .min_kl_coef = null,
            .max_kl_coef = null,
        };
    }

    const target_kl = config.target_kl orelse return error.IncompleteGrpoAdaptiveKlConfig;
    const kl_horizon = config.kl_horizon orelse return error.IncompleteGrpoAdaptiveKlConfig;
    const min_kl_coef = config.min_kl_coef orelse 0.001;
    const max_kl_coef = config.max_kl_coef orelse 1.0;
    const initial_kl_coef = config.kl_coef orelse 0.04;
    if (!std.math.isFinite(target_kl) or target_kl <= 0.0 or target_kl >= train_max_kl or
        !std.math.isFinite(kl_horizon) or kl_horizon < 1.0 or
        !std.math.isFinite(min_kl_coef) or min_kl_coef < 0.0 or
        !std.math.isFinite(max_kl_coef) or max_kl_coef < min_kl_coef or
        initial_kl_coef <= 0.0 or initial_kl_coef < min_kl_coef or initial_kl_coef > max_kl_coef)
    {
        return error.InvalidGrpoAdaptiveKlConfig;
    }
    return .{
        .train_max_kl = train_max_kl,
        .budget_policy = budget_policy,
        .adaptive = true,
        .target_kl = target_kl,
        .kl_horizon = kl_horizon,
        .min_kl_coef = min_kl_coef,
        .max_kl_coef = max_kl_coef,
    };
}
