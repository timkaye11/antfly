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

// Fused chunker loss configuration, boundary head graph, and boundary metrics.

const std = @import("std");
const ml = @import("ml");
const Graph = ml.graph.Graph;
const NodeId = ml.graph.NodeId;
const Shape = ml.graph.Shape;
const Builder = ml.graph.Builder;

// ----------------------------------------------------------------------------
// FusedLossConfig
// ----------------------------------------------------------------------------

pub const FusedLossConfig = struct {
    lambda_chunk: f32 = 1.0,
    lambda_embed: f32 = 0.3,
    lambda_coherence: f32 = 0.2,
    focal_gamma: f32 = 2.0,
    focal_alpha: f32 = 0.75,
    temperature: f32 = 0.07,
    num_hard_negatives: u32 = 7,
    coherence_margin: f32 = 0.2,
    pos_weight: f32 = 5.0,
    boundary_rank_loss_weight: f32 = 0.0,
    boundary_rank_loss_margin: f32 = 1.0,
    boundary_rank_loss_top_k: u32 = 1,
    boundary_same_token_rank_loss_weight: f32 = 0.0,
    boundary_same_token_rank_loss_top_k: u32 = 4,
    boundary_same_token_negative_weight: f32 = 1.0,
    boundary_same_token_negative_top_k: u32 = 0,
    boundary_candidate_rank_loss_weight: f32 = 0.0,
    boundary_candidate_rank_loss_top_k: u32 = 8,
    boundary_candidate_negative_weight: f32 = 1.0,
    boundary_gold_count_rank_loss_weight: f32 = 0.0,
    boundary_gold_count_rank_loss_margin: f32 = 1.0,
    boundary_gold_count_rank_loss_negative_multiplier: u32 = 1,
    boundary_local_window_loss_weight: f32 = 0.0,
    boundary_local_window_radius: u32 = 12,
    use_focal: bool = false,
    contrastive_focal_gamma: f32 = 0.0,
    contrastive_focal_alpha: f32 = 0.75,

    // MRL (Matryoshka Representation Learning)
    use_mrl: bool = false,
    mrl_dims: []const u32 = &.{ 768, 256, 128 },
    mrl_weights: []const f32 = &.{ 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 },

    // SPLADE sparse embeddings
    enable_splade: bool = false,
    lambda_splade: f32 = 0.15,
    lambda_flops: f32 = 3e-5,
    splade_focus_epoch: u32 = 4, // epoch when SPLADE activates
};

/// Returns a config tuned for boundary-focused training:
/// lambda_embed=0.1, lambda_coherence=0.0 (ignore coherence term).
pub fn boundaryFocusConfig() FusedLossConfig {
    return .{
        .lambda_embed = 0.1,
        .lambda_coherence = 0.0,
    };
}

// ----------------------------------------------------------------------------
// BoundaryHeadGraph
// ----------------------------------------------------------------------------

pub const BoundaryHeadGraph = struct {
    graph: Graph,
    feature_id: NodeId,
    target_id: NodeId,
    mask_id: NodeId,
    feature_dropout_mask_id: NodeId,
    hidden_dropout_mask_id: NodeId,
    w1_id: NodeId,
    b1_id: NodeId,
    w2_id: NodeId,
    b2_id: NodeId,
    logits_id: NodeId,
    loss_id: NodeId,
    total: usize,
    hidden_dim: usize,
    mlp_dim: usize,
    pos_weight: f32,
    use_focal: bool,
    focal_gamma: f32,
    focal_alpha: f32,
    boundary_dropout: f32,

    /// Build the 2-layer MLP boundary head graph.
    ///
    /// Architecture:
    ///   features [total, hidden_dim]
    ///     -> linear(w1[mlp_dim, hidden_dim] + b1[mlp_dim])
    ///     -> gelu
    ///     -> linear(w2[2, mlp_dim] + b2[2])
    ///     -> masked weighted crossEntropyLoss(logits, targets, mask) by default
    ///        or masked Go-compatible focal loss when use_focal is true
    ///        otherwise masked weighted crossEntropyLoss(logits, targets, mask)
    pub fn init(
        allocator: std.mem.Allocator,
        total: usize,
        hidden_dim: usize,
        mlp_dim: usize,
        pos_weight: f32,
        use_focal: bool,
        focal_gamma: f32,
        focal_alpha: f32,
        boundary_dropout: f32,
    ) !BoundaryHeadGraph {
        var graph = Graph.init(allocator);
        errdefer graph.deinit();
        var b = Builder.init(&graph);

        const targets = try b.parameter("targets", Shape.init(.f32, &.{ @as(i64, @intCast(total)), 2 }));
        const mask = try b.parameter("mask", Shape.init(.f32, &.{ @as(i64, @intCast(total)), 1 }));
        const forward = try buildBoundaryHeadForwardNodes(&b, total, hidden_dim, mlp_dim);
        const loss = if (use_focal)
            try buildMaskedBoundaryFocalLoss(allocator, &b, forward.logits_id, targets, mask, total, focal_gamma, focal_alpha)
        else
            try buildMaskedWeightedBoundaryCrossEntropyLoss(allocator, &b, forward.logits_id, targets, mask, total, pos_weight);
        try graph.markOutput(loss);

        return .{
            .graph = graph,
            .feature_id = forward.feature_id,
            .target_id = targets,
            .mask_id = mask,
            .feature_dropout_mask_id = forward.feature_dropout_mask_id,
            .hidden_dropout_mask_id = forward.hidden_dropout_mask_id,
            .w1_id = forward.w1_id,
            .b1_id = forward.b1_id,
            .w2_id = forward.w2_id,
            .b2_id = forward.b2_id,
            .logits_id = forward.logits_id,
            .loss_id = loss,
            .total = total,
            .hidden_dim = hidden_dim,
            .mlp_dim = mlp_dim,
            .pos_weight = pos_weight,
            .use_focal = use_focal,
            .focal_gamma = focal_gamma,
            .focal_alpha = focal_alpha,
            .boundary_dropout = boundary_dropout,
        };
    }

    pub fn deinit(self: *BoundaryHeadGraph) void {
        self.graph.deinit();
        self.* = undefined;
    }
};

/// The boundary-head MLP forward shared by the CE graph, the forward-only
/// graph, and the synthetic-VJP graph:
///   dropped_features = features * feature_dropout_mask
///   dense            = dropped_features @ w1^T + b1
///   hidden           = geluExact(dense)
///   dropped_hidden   = hidden * hidden_dropout_mask
///   logits           = dropped_hidden @ w2^T + b2
const BoundaryHeadForwardNodes = struct {
    feature_id: NodeId,
    feature_dropout_mask_id: NodeId,
    hidden_dropout_mask_id: NodeId,
    w1_id: NodeId,
    b1_id: NodeId,
    w2_id: NodeId,
    b2_id: NodeId,
    logits_id: NodeId,
};

fn buildBoundaryHeadForwardNodes(
    b: *Builder,
    total: usize,
    hidden_dim: usize,
    mlp_dim: usize,
) !BoundaryHeadForwardNodes {
    const features = try b.parameter("features", Shape.init(.f32, &.{ @as(i64, @intCast(total)), @as(i64, @intCast(hidden_dim)) }));
    const feature_dropout_mask = try b.parameter("feature_dropout_mask", Shape.init(.f32, &.{ @as(i64, @intCast(total)), @as(i64, @intCast(hidden_dim)) }));
    const hidden_dropout_mask = try b.parameter("hidden_dropout_mask", Shape.init(.f32, &.{ @as(i64, @intCast(total)), @as(i64, @intCast(mlp_dim)) }));
    const w1 = try b.parameter("w1", Shape.init(.f32, &.{ @as(i64, @intCast(mlp_dim)), @as(i64, @intCast(hidden_dim)) }));
    const b1 = try b.parameter("b1", Shape.init(.f32, &.{@as(i64, @intCast(mlp_dim))}));
    const w2 = try b.parameter("w2", Shape.init(.f32, &.{ 2, @as(i64, @intCast(mlp_dim)) }));
    const b2 = try b.parameter("b2", Shape.init(.f32, &.{2}));

    const dropped_features = try b.mul(features, feature_dropout_mask);
    const dense = try b.linear(dropped_features, w1, b1, @intCast(total), @intCast(hidden_dim), @intCast(mlp_dim));
    const hidden = try b.geluExact(dense);
    const dropped_hidden = try b.mul(hidden, hidden_dropout_mask);
    const logits = try b.linear(dropped_hidden, w2, b2, @intCast(total), @intCast(mlp_dim), 2);

    return .{
        .feature_id = features,
        .feature_dropout_mask_id = feature_dropout_mask,
        .hidden_dropout_mask_id = hidden_dropout_mask,
        .w1_id = w1,
        .b1_id = b1,
        .w2_id = w2,
        .b2_id = b2,
        .logits_id = logits,
    };
}

/// Forward-only boundary head graph with `logits` as the sole output. Used
/// by the compiled boundary-head fast path: logits are computed on the graph
/// runtime while the CE loss (and its logits gradient) stays on the CPU
/// exactly as in the eager ops path.
pub const BoundaryHeadForwardGraph = struct {
    graph: Graph,
    nodes: BoundaryHeadForwardNodes,
    total: usize,
    hidden_dim: usize,
    mlp_dim: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        total: usize,
        hidden_dim: usize,
        mlp_dim: usize,
    ) !BoundaryHeadForwardGraph {
        var graph = Graph.init(allocator);
        errdefer graph.deinit();
        var b = Builder.init(&graph);
        const nodes = try buildBoundaryHeadForwardNodes(&b, total, hidden_dim, mlp_dim);
        try graph.markOutput(nodes.logits_id);
        return .{
            .graph = graph,
            .nodes = nodes,
            .total = total,
            .hidden_dim = hidden_dim,
            .mlp_dim = mlp_dim,
        };
    }

    pub fn deinit(self: *BoundaryHeadForwardGraph) void {
        self.graph.deinit();
        self.* = undefined;
    }
};

/// Boundary head forward plus the synthetic upstream loss
/// `loss = sum(logits * dlogits)`, where `dlogits` is the CPU-computed CE
/// logits gradient fed as an input. Differentiating this scalar wrt the head
/// parameters (and optionally the features) reproduces the eager manual
/// backward chain through the dropout masks, GELU, and both linears.
pub const BoundaryHeadSyntheticVJPGraph = struct {
    graph: Graph,
    nodes: BoundaryHeadForwardNodes,
    dlogits_id: NodeId,
    loss_id: NodeId,
    total: usize,
    hidden_dim: usize,
    mlp_dim: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        total: usize,
        hidden_dim: usize,
        mlp_dim: usize,
    ) !BoundaryHeadSyntheticVJPGraph {
        var graph = Graph.init(allocator);
        errdefer graph.deinit();
        var b = Builder.init(&graph);
        const nodes = try buildBoundaryHeadForwardNodes(&b, total, hidden_dim, mlp_dim);
        const dlogits = try b.parameter("dlogits", Shape.init(.f32, &.{ @as(i64, @intCast(total)), 2 }));
        const weighted = try b.mul(nodes.logits_id, dlogits);
        const loss = try b.reduceSum(weighted, &.{ 0, 1 });
        try graph.markOutput(loss);
        return .{
            .graph = graph,
            .nodes = nodes,
            .dlogits_id = dlogits,
            .loss_id = loss,
            .total = total,
            .hidden_dim = hidden_dim,
            .mlp_dim = mlp_dim,
        };
    }

    pub fn deinit(self: *BoundaryHeadSyntheticVJPGraph) void {
        self.graph.deinit();
        self.* = undefined;
    }
};

fn buildMaskedWeightedBoundaryCrossEntropyLoss(
    allocator: std.mem.Allocator,
    b: *Builder,
    logits: NodeId,
    targets: NodeId,
    mask: NodeId,
    total: usize,
    pos_weight: f32,
) !NodeId {
    const weights = try allocator.alloc(f32, total * 2);
    defer allocator.free(weights);
    for (0..total) |i| {
        weights[i * 2] = 1.0;
        weights[i * 2 + 1] = pos_weight;
    }

    const class_weights = try b.tensorConst(weights, Shape.init(.f32, &.{ @as(i64, @intCast(total)), 2 }));
    const log_probs = try b.logSoftmax(logits);
    const weighted_targets = try b.mul(targets, class_weights);
    const weighted_log_probs = try b.mul(weighted_targets, log_probs);
    const class_sum = try b.reduceSum(weighted_log_probs, &.{1});
    const per_token_loss = try b.neg(class_sum);
    const masked_loss = try b.mul(per_token_loss, mask);
    const numerator = try b.reduceSum(masked_loss, &.{ 0, 1 });
    const denom = try b.reduceSum(mask, &.{ 0, 1 });
    const eps = try b.scalarConst(.f32, 1e-12);
    const denom_safe = try b.add(denom, eps);
    return b.div(numerator, denom_safe);
}

fn buildMaskedBoundaryFocalLoss(
    allocator: std.mem.Allocator,
    b: *Builder,
    logits: NodeId,
    targets: NodeId,
    mask: NodeId,
    total: usize,
    focal_gamma: f32,
    focal_alpha: f32,
) !NodeId {
    if (@abs(focal_gamma - 1.0) > 1e-6 and @abs(focal_gamma - 2.0) > 1e-6) {
        return error.UnsupportedBoundaryFocalGamma;
    }

    const alpha_weights = try allocator.alloc(f32, total * 2);
    defer allocator.free(alpha_weights);
    for (0..total) |i| {
        alpha_weights[i * 2] = 1.0 - focal_alpha;
        alpha_weights[i * 2 + 1] = focal_alpha;
    }

    const class_alpha = try b.tensorConst(alpha_weights, Shape.init(.f32, &.{ @as(i64, @intCast(total)), 2 }));
    const probs = try b.softmax(logits);
    const target_probs = try b.mul(targets, probs);
    const p_t = try b.reduceSum(target_probs, &.{1});
    const weighted_targets = try b.mul(targets, class_alpha);
    const alpha_t = try b.reduceSum(weighted_targets, &.{1});

    const one = try b.scalarConst(.f32, 1.0);
    const one_minus_pt = try b.sub(one, p_t);
    const focal_weight = if (@abs(focal_gamma - 2.0) <= 1e-6)
        try b.mul(one_minus_pt, one_minus_pt)
    else
        one_minus_pt;

    const eps = try b.scalarConst(.f32, 1e-12);
    const p_t_safe = try b.add(p_t, eps);
    const log_p_t = try b.logOp(p_t_safe);
    const ce_loss = try b.neg(log_p_t);
    const focal_scaled = try b.mul(alpha_t, focal_weight);
    const per_token_loss = try b.mul(focal_scaled, ce_loss);
    const masked_loss = try b.mul(per_token_loss, mask);
    const numerator = try b.reduceSum(masked_loss, &.{ 0, 1 });
    const denom = try b.reduceSum(mask, &.{ 0, 1 });
    const denom_safe = try b.add(denom, eps);
    return b.div(numerator, denom_safe);
}

// ----------------------------------------------------------------------------
// BoundaryHeadWeights
// ----------------------------------------------------------------------------

pub const BoundaryHeadWeights = struct {
    allocator: std.mem.Allocator,
    w1: []f32, // [mlp_dim, hidden_dim]
    b1: []f32, // [mlp_dim]
    w2: []f32, // [2, mlp_dim]
    b2: []f32, // [2]
    hidden_dim: usize,
    mlp_dim: usize,

    /// Initialise weights with deterministic GoMLX-compatible dense-layer
    /// defaults: He normal weights and zero biases. Go stores dense weights as
    /// [input, output]; these buffers store [output, input], so samples are
    /// transposed into the local layout.
    pub fn initDeterministic(allocator: std.mem.Allocator, hidden_dim: usize, mlp_dim: usize) !BoundaryHeadWeights {
        const w1 = try allocator.alloc(f32, mlp_dim * hidden_dim);
        errdefer allocator.free(w1);
        const b1 = try allocator.alloc(f32, mlp_dim);
        errdefer allocator.free(b1);
        const w2 = try allocator.alloc(f32, 2 * mlp_dim);
        errdefer allocator.free(w2);
        const b2 = try allocator.alloc(f32, 2);
        errdefer allocator.free(b2);

        var prng = std.Random.DefaultPrng.init(42);
        const rng = prng.random();

        fillDenseHeNormalTransposed(w1, hidden_dim, mlp_dim, rng);
        @memset(b1, 0);

        fillDenseHeNormalTransposed(w2, mlp_dim, 2, rng);
        @memset(b2, 0);

        return .{
            .allocator = allocator,
            .w1 = w1,
            .b1 = b1,
            .w2 = w2,
            .b2 = b2,
            .hidden_dim = hidden_dim,
            .mlp_dim = mlp_dim,
        };
    }

    pub fn deinit(self: *BoundaryHeadWeights) void {
        self.allocator.free(self.w1);
        self.allocator.free(self.b1);
        self.allocator.free(self.w2);
        self.allocator.free(self.b2);
        self.* = undefined;
    }
};

fn fillDenseHeNormalTransposed(values: []f32, fan_in: usize, fan_out: usize, rng: std.Random) void {
    std.debug.assert(values.len == fan_in * fan_out);
    const scale = @max(@as(f32, @floatFromInt(fan_in)), 1.0);
    const stddev: f32 = @sqrt(2.0 / scale);
    for (0..fan_in) |in_i| {
        for (0..fan_out) |out_i| {
            values[out_i * fan_in + in_i] = rng.floatNorm(f32) * stddev;
        }
    }
}

// ----------------------------------------------------------------------------
// BoundaryMetrics
// ----------------------------------------------------------------------------

pub const BoundaryMetrics = struct {
    tp: u64,
    fp: u64,
    fn_: u64,

    pub fn precision(self: BoundaryMetrics) f32 {
        const denom = self.tp + self.fp;
        if (denom == 0) return 0.0;
        return @as(f32, @floatFromInt(self.tp)) / @as(f32, @floatFromInt(denom));
    }

    pub fn recall(self: BoundaryMetrics) f32 {
        const denom = self.tp + self.fn_;
        if (denom == 0) return 0.0;
        return @as(f32, @floatFromInt(self.tp)) / @as(f32, @floatFromInt(denom));
    }

    pub fn f1(self: BoundaryMetrics) f32 {
        const p = self.precision();
        const r = self.recall();
        const denom = p + r;
        if (denom == 0.0) return 0.0;
        return 2.0 * p * r / denom;
    }
};

pub fn positiveBoundaryProbability(logit_not_boundary: f32, logit_boundary: f32) f32 {
    const margin = logit_boundary - logit_not_boundary;
    if (margin >= 0.0) {
        return 1.0 / (1.0 + @exp(-margin));
    }
    const e = @exp(margin);
    return e / (1.0 + e);
}

/// Decision threshold corresponding to an unweighted posterior of 0.5 when a
/// binary boundary head was trained with positive-class weighted CE.
pub fn weightedCePositiveThreshold(pos_weight: f32) f32 {
    if (pos_weight <= 0.0) return 0.5;
    const threshold = pos_weight / (1.0 + pos_weight);
    return @min(@max(threshold, 0.0), 1.0);
}

/// Compute boundary detection metrics from flat logits and labels.
///
/// logits: flat [total * 2] f32, row-major — two class scores per token.
/// labels: flat [total] f32, values 0 or 1.
/// mask:   flat [total] f32 (1=valid, 0=pad); pass null to count all tokens.
pub fn computeBoundaryMetrics(logits: []const f32, labels: []const f32, mask: ?[]const f32) BoundaryMetrics {
    return computeBoundaryMetricsWithThreshold(logits, labels, mask, 0.5);
}

/// Compute boundary detection metrics after applying a probability threshold to
/// the two-logit boundary head. A threshold of 0.5 is equivalent to argmax.
pub fn computeBoundaryMetricsWithThreshold(
    logits: []const f32,
    labels: []const f32,
    mask: ?[]const f32,
    threshold: f32,
) BoundaryMetrics {
    var tp: u64 = 0;
    var fp: u64 = 0;
    var fn_: u64 = 0;

    const total = labels.len;
    for (0..total) |i| {
        if (mask) |m| {
            if (m[i] <= 0.5) continue;
        }
        const prob = positiveBoundaryProbability(logits[i * 2 + 0], logits[i * 2 + 1]);
        const predicted: u1 = if (prob > threshold) 1 else 0;
        const true_label: bool = labels[i] > 0.5;

        if (predicted == 1 and true_label) {
            tp += 1;
        } else if (predicted == 1 and !true_label) {
            fp += 1;
        } else if (predicted == 0 and true_label) {
            fn_ += 1;
        }
    }

    return .{ .tp = tp, .fp = fp, .fn_ = fn_ };
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

test "FusedLossConfig defaults" {
    const cfg = FusedLossConfig{};
    try std.testing.expectApproxEqAbs(cfg.lambda_chunk, 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(cfg.lambda_embed, 0.3, 1e-6);
    try std.testing.expectApproxEqAbs(cfg.lambda_coherence, 0.2, 1e-6);
    try std.testing.expectApproxEqAbs(cfg.focal_gamma, 2.0, 1e-6);
    try std.testing.expectApproxEqAbs(cfg.focal_alpha, 0.75, 1e-6);
    try std.testing.expectApproxEqAbs(cfg.temperature, 0.07, 1e-6);
    try std.testing.expectEqual(cfg.num_hard_negatives, 7);
    try std.testing.expectApproxEqAbs(cfg.coherence_margin, 0.2, 1e-6);
    try std.testing.expectApproxEqAbs(cfg.pos_weight, 5.0, 1e-6);
    try std.testing.expectApproxEqAbs(cfg.boundary_rank_loss_weight, 0.0, 1e-6);
    try std.testing.expectApproxEqAbs(cfg.boundary_rank_loss_margin, 1.0, 1e-6);
    try std.testing.expectEqual(@as(u32, 1), cfg.boundary_rank_loss_top_k);
    try std.testing.expectApproxEqAbs(cfg.boundary_same_token_rank_loss_weight, 0.0, 1e-6);
    try std.testing.expectEqual(@as(u32, 4), cfg.boundary_same_token_rank_loss_top_k);
    try std.testing.expectApproxEqAbs(cfg.boundary_same_token_negative_weight, 1.0, 1e-6);
    try std.testing.expectEqual(@as(u32, 0), cfg.boundary_same_token_negative_top_k);
    try std.testing.expectApproxEqAbs(cfg.boundary_candidate_rank_loss_weight, 0.0, 1e-6);
    try std.testing.expectEqual(@as(u32, 8), cfg.boundary_candidate_rank_loss_top_k);
    try std.testing.expectApproxEqAbs(cfg.boundary_candidate_negative_weight, 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(cfg.boundary_gold_count_rank_loss_weight, 0.0, 1e-6);
    try std.testing.expectApproxEqAbs(cfg.boundary_gold_count_rank_loss_margin, 1.0, 1e-6);
    try std.testing.expectEqual(@as(u32, 1), cfg.boundary_gold_count_rank_loss_negative_multiplier);
    try std.testing.expect(!cfg.use_focal);
}

test "BoundaryHeadGraph uses exact GELU erf form" {
    const allocator = std.testing.allocator;

    var head_graph = try BoundaryHeadGraph.init(
        allocator,
        4,
        8,
        6,
        1.0,
        false,
        2.0,
        0.75,
        0.1,
    );
    defer head_graph.deinit();

    var saw_erf = false;
    var saw_fused_gelu = false;
    for (0..head_graph.graph.nodeCount()) |i| {
        switch (head_graph.graph.node(@intCast(i)).op) {
            .erf => saw_erf = true,
            .fused_gelu => saw_fused_gelu = true,
            else => {},
        }
    }

    try std.testing.expect(saw_erf);
    try std.testing.expect(!saw_fused_gelu);
}

test "BoundaryHeadWeights init uses Go-compatible He normal scale" {
    const allocator = std.testing.allocator;

    var head = try BoundaryHeadWeights.initDeterministic(allocator, 768, 256);
    defer head.deinit();

    const w1_expected_rms = @sqrt(2.0 / @as(f32, @floatFromInt(768)));
    const w2_expected_rms = @sqrt(2.0 / @as(f32, @floatFromInt(256)));
    var w1_sum_sq: f64 = 0;
    var w2_sum_sq: f64 = 0;
    for (head.w1) |value| w1_sum_sq += @as(f64, value) * @as(f64, value);
    for (head.w2) |value| w2_sum_sq += @as(f64, value) * @as(f64, value);

    const w1_rms: f32 = @floatCast(@sqrt(w1_sum_sq / @as(f64, @floatFromInt(head.w1.len))));
    const w2_rms: f32 = @floatCast(@sqrt(w2_sum_sq / @as(f64, @floatFromInt(head.w2.len))));
    try std.testing.expectApproxEqAbs(w1_expected_rms, w1_rms, 0.005);
    try std.testing.expectApproxEqAbs(w2_expected_rms, w2_rms, 0.02);
    for (head.b1) |value| try std.testing.expectEqual(@as(f32, 0.0), value);
    for (head.b2) |value| try std.testing.expectEqual(@as(f32, 0.0), value);
}

test "computeBoundaryMetrics basic" {
    // 4 tokens total; mask marks tokens 0 and 2 as valid, 1 and 3 as padding.
    //
    // Token 0: logits=[0.1, 0.9] → predicted=1, label=1 → TP
    // Token 1: masked out (ignored)
    // Token 2: logits=[0.8, 0.2] → predicted=0, label=1 → FN
    // Token 3: masked out (ignored)
    const logits = [_]f32{
        0.1, 0.9, // token 0
        0.3, 0.7, // token 1 (masked)
        0.8, 0.2, // token 2
        0.4, 0.6, // token 3 (masked)
    };
    const labels = [_]f32{ 1.0, 0.0, 1.0, 0.0 };
    const mask = [_]f32{ 1.0, 0.0, 1.0, 0.0 };

    const m = computeBoundaryMetrics(&logits, &labels, &mask);
    try std.testing.expectEqual(m.tp, 1);
    try std.testing.expectEqual(m.fp, 0);
    try std.testing.expectEqual(m.fn_, 1);

    try std.testing.expectApproxEqAbs(m.precision(), 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(m.recall(), 0.5, 1e-6);
    // f1 = 2*1*0.5/(1+0.5) = 1/1.5 ≈ 0.6667
    try std.testing.expectApproxEqAbs(m.f1(), 2.0 / 3.0, 1e-5);
}

test "weightedCePositiveThreshold maps positive class weight to posterior threshold" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), weightedCePositiveThreshold(1.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0 / 6.0), weightedCePositiveThreshold(5.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), weightedCePositiveThreshold(0.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), weightedCePositiveThreshold(-3.0), 1e-6);
}

test "computeBoundaryMetricsWithThreshold exposes calibrated operating points" {
    const logits = [_]f32{
        0.0, 1.0, // positive probability ~= 0.731
        0.0, 0.2, // positive probability ~= 0.550
        0.0, -1.0, // positive probability ~= 0.269
    };
    const labels = [_]f32{ 1.0, 0.0, 1.0 };

    const default_m = computeBoundaryMetrics(&logits, &labels, null);
    try std.testing.expectEqual(@as(u64, 1), default_m.tp);
    try std.testing.expectEqual(@as(u64, 1), default_m.fp);
    try std.testing.expectEqual(@as(u64, 1), default_m.fn_);

    const strict_m = computeBoundaryMetricsWithThreshold(&logits, &labels, null, 0.7);
    try std.testing.expectEqual(@as(u64, 1), strict_m.tp);
    try std.testing.expectEqual(@as(u64, 0), strict_m.fp);
    try std.testing.expectEqual(@as(u64, 1), strict_m.fn_);

    const loose_m = computeBoundaryMetricsWithThreshold(&logits, &labels, null, 0.2);
    try std.testing.expectEqual(@as(u64, 2), loose_m.tp);
    try std.testing.expectEqual(@as(u64, 1), loose_m.fp);
    try std.testing.expectEqual(@as(u64, 0), loose_m.fn_);
}
