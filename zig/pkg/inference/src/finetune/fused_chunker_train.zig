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

// Full training loop for the fused chunker-embedder model.
//
// Orchestrates:
//   1. Boundary head training (Metal CE fast path, graph fallback/focal path)
//   2. CPU InfoNCE contrastive loss
//   3. AdamW optimizer steps for all trainable parameters
//   4. Evaluation (micro-F1)
//   5. Checkpoint save/load (simple binary format)

const std = @import("std");
const ml = @import("ml");
const Graph = ml.graph.Graph;
const NodeId = ml.graph.NodeId;
const Shape = ml.graph.Shape;
const Builder = ml.graph.Builder;
const optimizers = ml.graph.optimizers;
const ops_mod = @import("../ops/ops.zig");
const ComputeBackend = ops_mod.ComputeBackend;
const CT = ops_mod.CT;

pub const contrastive_gradient_path = "direct_mean_pool_scatter";
const tensor_probe_hash_offset: u64 = 14695981039346656037;
const tensor_probe_hash_prime: u64 = 1099511628211;
const tensor_probe_top_abs_count: usize = 8;
const training = @import("../graph/training.zig");
const compat = @import("../io/compat.zig");
const native_compute = @import("../ops/native_compute.zig");
const fused_chunker_data = @import("fused_chunker_data.zig");
const fused_chunker = @import("fused_chunker.zig");
const fused_chunker_loss = @import("fused_chunker_loss.zig");
const infonce_cpu = @import("infonce_cpu.zig");
const fused_chunker_lora = @import("lora_adapter_set.zig");
const LoRAAdapterSet = fused_chunker_lora.LoRAAdapterSet;

// ----------------------------------------------------------------------------
// FusedTrainingConfig
// ----------------------------------------------------------------------------

pub const FusedTrainingConfig = struct {
    // Model
    max_seq_len: u32 = 384,
    embedding_dim: u32 = 768,
    hidden_size: u32 = 768,
    boundary_mlp_dim: u32 = 256,
    max_chunks: u32 = 32,

    // Training
    batch_size: u32 = 16,
    num_epochs: u32 = 10,
    learning_rate: f32 = 1e-4,
    warmup_steps: u32 = 50,
    total_steps: u32 = 1000,
    weight_decay: f32 = 0.01,
    max_grad_norm: f32 = 0.0,
    seed: u64 = 42,
    step_log_every: u32 = 1,

    // AdamW
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    adam_epsilon: f32 = 1e-8,

    // Loss (delegates to FusedLossConfig)
    lambda_chunk: f32 = 1.0,
    lambda_embed: f32 = 0.3,
    temperature: f32 = 0.07,
    use_boundary_focal: bool = false,
    focal_gamma: f32 = 2.0,
    focal_alpha: f32 = 0.75,
    contrastive_focal_gamma: f32 = 0.0,
    contrastive_focal_alpha: f32 = 0.75,
    pos_weight: f32 = 5.0,
    boundary_dropout: f32 = 0.1,

    // Curriculum
    boundary_focus_epochs: u32 = 3,

    // Checkpointing
    checkpoint_every: u32 = 0,
    checkpoint_every_steps: u32 = 0,

    // Gradient accumulation (Feature 2)
    grad_accum_steps: u32 = 1,

    // Schedule-Free AdamW (Feature 3)
    use_schedule_free: bool = false,

    // Cross-Batch Memory (Feature 1)
    xbm_capacity: usize = 0,

    // NEFTune noise (Feature 5)
    neftune_alpha: f32 = 0.0,

    // Layer-wise LR decay (Feature 6)
    llrd_decay: f32 = 1.0,

    // Length bucketing (Feature 8)
    length_bucketing: bool = false,
    bucket_size: usize = 256,

    // Mixed precision (Feature 9 CLI flag — stored for downstream use)
    mixed_precision: bool = false,

    // SPLADE sparse embedding head
    enable_splade: bool = false,
    lambda_splade: f32 = 0.15,
    lambda_flops: f32 = 3e-5,
    splade_focus_epoch: u32 = 4,

    // Matryoshka Representation Learning
    use_mrl: bool = false,
    mrl_dims: []const u32 = &.{ 768, 256, 128 },
    mrl_weights: []const f32 = &.{ 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 },

    // LoRA+ ratio: multiplier on the LoRA B-matrix learning rate relative to A
    lora_plus_ratio: f32 = 1.0,

    pub fn lrSchedule(self: FusedTrainingConfig) optimizers.LearningRateSchedule {
        return .{ .warmup_cosine = .{
            .initial_lr = self.learning_rate,
            .min_lr = 0.0,
            .warmup_steps = self.warmup_steps,
            .total_steps = self.total_steps,
        } };
    }
};

pub fn selectActiveLoRALayers(
    allocator: std.mem.Allocator,
    num_layers: u32,
    lisa_sample_layers: u32,
    lisa_top_k: u32,
    step: u64,
    seed: u64,
) ![]bool {
    const n: usize = @intCast(num_layers);
    const active = try allocator.alloc(bool, n);
    errdefer allocator.free(active);

    if (lisa_sample_layers == 0) {
        @memset(active, true);
        return active;
    }

    @memset(active, false);
    const top_k: usize = @min(@as(usize, @intCast(lisa_top_k)), n);
    const top_start = n - top_k;
    for (top_start..n) |layer_idx| active[layer_idx] = true;

    const remaining_count = top_start;
    const sample_count: usize = @min(@as(usize, @intCast(lisa_sample_layers)), remaining_count);
    if (sample_count == 0) return active;

    const remaining = try allocator.alloc(u32, remaining_count);
    defer allocator.free(remaining);
    for (remaining, 0..) |*layer_idx, i| layer_idx.* = @intCast(i);

    var prng = std.Random.DefaultPrng.init(seed +% step);
    const rng = prng.random();
    var i = remaining.len;
    while (i > 1) {
        i -= 1;
        const j = rng.uintLessThan(usize, i + 1);
        const tmp = remaining[i];
        remaining[i] = remaining[j];
        remaining[j] = tmp;
    }
    for (remaining[0..sample_count]) |layer_idx| active[@intCast(layer_idx)] = true;
    return active;
}

pub inline fn isLoRALayerActive(active_layers: []const bool, layer_idx: u32) bool {
    const idx: usize = @intCast(layer_idx);
    return idx < active_layers.len and active_layers[idx];
}

// ----------------------------------------------------------------------------
// CrossBatchMemory (Feature 1)
// ----------------------------------------------------------------------------

/// Ring buffer of chunk embeddings from recent batches.
/// Expands the effective negative set for InfoNCE contrastive learning.
pub const CrossBatchMemory = struct {
    allocator: std.mem.Allocator,
    capacity: usize,
    embed_dim: usize,
    embeddings: []f32, // [capacity, embed_dim] circular buffer
    doc_ids: []u32, // [capacity]
    count: usize, // number of valid entries
    head: usize, // next write position

    pub fn init(allocator: std.mem.Allocator, capacity: usize, embed_dim: usize) !CrossBatchMemory {
        const embeddings = try allocator.alloc(f32, capacity * embed_dim);
        errdefer allocator.free(embeddings);
        const doc_ids = try allocator.alloc(u32, capacity);
        errdefer allocator.free(doc_ids);
        @memset(embeddings, 0);
        @memset(doc_ids, 0);
        return .{
            .allocator = allocator,
            .capacity = capacity,
            .embed_dim = embed_dim,
            .embeddings = embeddings,
            .doc_ids = doc_ids,
            .count = 0,
            .head = 0,
        };
    }

    pub fn deinit(self: *CrossBatchMemory) void {
        self.allocator.free(self.embeddings);
        self.allocator.free(self.doc_ids);
        self.* = undefined;
    }

    /// Add a batch of chunk embeddings to the memory.
    /// embeddings: [num_chunks * embed_dim], mask: [num_chunks]
    /// doc_id_offset is added to each doc_id before storing, so that entries from
    /// different batches have globally unique IDs (Fix 1: XBM doc_id collision).
    pub fn add(self: *CrossBatchMemory, embeddings: []const f32, doc_ids: []const u32, chunk_mask: []const f32, num_chunks: usize, doc_id_offset: u64) void {
        for (0..num_chunks) |ci| {
            if (chunk_mask[ci] <= 0.5) continue;
            const src = embeddings[ci * self.embed_dim .. (ci + 1) * self.embed_dim];
            const dst = self.embeddings[self.head * self.embed_dim .. (self.head + 1) * self.embed_dim];
            @memcpy(dst, src);
            self.doc_ids[self.head] = @truncate(doc_ids[ci] + doc_id_offset);
            self.head = (self.head + 1) % self.capacity;
            if (self.count < self.capacity) self.count += 1;
        }
    }

    /// Get all valid stored embeddings and their doc_ids.
    /// Returns slices into internal storage (valid until next add call).
    pub fn getStored(self: *CrossBatchMemory) struct { embeddings: []const f32, doc_ids: []const u32, count: usize } {
        return .{
            .embeddings = self.embeddings[0 .. self.count * self.embed_dim],
            .doc_ids = self.doc_ids[0..self.count],
            .count = self.count,
        };
    }
};

// ----------------------------------------------------------------------------
// ScheduleFreeAdamW (Feature 3)
// ----------------------------------------------------------------------------

pub const ScheduleFreeAdamWState = struct {
    allocator: std.mem.Allocator,
    z: []f32, // base iterate (same shape as parameter)
    v: []f32, // second moment
    step: u64,

    pub fn init(allocator: std.mem.Allocator, initial_weights: []const f32) !ScheduleFreeAdamWState {
        const z = try allocator.dupe(f32, initial_weights);
        errdefer allocator.free(z);
        const v = try allocator.alloc(f32, initial_weights.len);
        errdefer allocator.free(v);
        @memset(v, 0);
        return .{
            .allocator = allocator,
            .z = z,
            .v = v,
            .step = 0,
        };
    }

    pub fn deinit(self: *ScheduleFreeAdamWState) void {
        self.allocator.free(self.z);
        self.allocator.free(self.v);
        self.* = undefined;
    }
};

/// Schedule-Free AdamW step (Defazio et al. 2024).
/// weights (x = Polyak average) is updated in place.
/// lr is used directly — caller's LR schedule already handles warmup (Fix 3).
pub fn scheduleFreeAdamWStep(
    weights: []f32, // x (Polyak average) — updated in place
    grad: []const f32,
    state: *ScheduleFreeAdamWState,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    warmup_steps: u32, // retained for API compatibility; no longer used internally
) void {
    _ = warmup_steps;
    state.step += 1;
    const lr_t = lr;
    // Polyak mixing coefficient: c = min(β₁, 1/t). Decreases as 1/t so the
    // running average converges; β₁ caps it from above on the first few steps.
    const c = @min(beta1, 1.0 / @as(f32, @floatFromInt(state.step)));

    for (0..weights.len) |i| {
        const g = grad[i];
        // Update second moment
        state.v[i] = beta2 * state.v[i] + (1.0 - beta2) * g * g;
        // Update z (base iterate)
        const denom = @sqrt(state.v[i]) + epsilon;
        state.z[i] = state.z[i] - lr_t * g / denom - lr_t * weight_decay * state.z[i];
        // Update x (Polyak average)
        weights[i] = (1.0 - c) * weights[i] + c * state.z[i];
    }
}

// ----------------------------------------------------------------------------
// BoundaryHead
// ----------------------------------------------------------------------------

pub const BoundaryHead = struct {
    allocator: std.mem.Allocator,
    w1: []f32, // [mlp_dim, hidden_dim]
    b1: []f32, // [mlp_dim]
    w2: []f32, // [2, mlp_dim]
    b2: []f32, // [2]
    hidden_dim: usize,
    mlp_dim: usize,

    pub fn init(allocator: std.mem.Allocator, hidden_dim: usize, mlp_dim: usize) !BoundaryHead {
        return initWithSeed(allocator, hidden_dim, mlp_dim, 42);
    }

    /// Initialise weights with GoMLX's default dense-layer semantics: He normal
    /// weights and zero biases. Go stores dense weights as [input, output], while
    /// this trainer stores them as [output, input], so the sampled logical dense
    /// matrix is transposed into Zig's layout.
    pub fn initWithSeed(allocator: std.mem.Allocator, hidden_dim: usize, mlp_dim: usize, seed: u64) !BoundaryHead {
        const w1 = try allocator.alloc(f32, mlp_dim * hidden_dim);
        errdefer allocator.free(w1);
        const b1 = try allocator.alloc(f32, mlp_dim);
        errdefer allocator.free(b1);
        const w2 = try allocator.alloc(f32, 2 * mlp_dim);
        errdefer allocator.free(w2);
        const b2 = try allocator.alloc(f32, 2);
        errdefer allocator.free(b2);

        var prng = std.Random.DefaultPrng.init(seed);
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

    pub fn deinit(self: *BoundaryHead) void {
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

pub const LegacyDenseBoundaryHead = struct {
    allocator: std.mem.Allocator,
    weight: []f32, // [2, hidden_dim]
    bias: []f32, // [2]
    hidden_dim: usize,

    pub fn init(allocator: std.mem.Allocator, hidden_dim: usize) !LegacyDenseBoundaryHead {
        const weight = try allocator.alloc(f32, 2 * hidden_dim);
        errdefer allocator.free(weight);
        const bias = try allocator.alloc(f32, 2);
        errdefer allocator.free(bias);
        @memset(weight, 0);
        @memset(bias, 0);
        return .{
            .allocator = allocator,
            .weight = weight,
            .bias = bias,
            .hidden_dim = hidden_dim,
        };
    }

    pub fn deinit(self: *LegacyDenseBoundaryHead) void {
        self.allocator.free(self.weight);
        self.allocator.free(self.bias);
        self.* = undefined;
    }
};

// ----------------------------------------------------------------------------
// TrainStepSummary
// ----------------------------------------------------------------------------

pub const TrainStepSummary = struct {
    boundary_loss: f32 = 0,
    contrastive_loss: f64 = 0,
    total_loss: f32 = 0,
    boundary_grad_norm: f64 = 0,
    boundary_tp: u64 = 0,
    boundary_fp: u64 = 0,
    boundary_fn: u64 = 0,
    step: u32 = 0,
    learning_rate: f32 = 0,
};

pub const TrainStepWithGradSummary = struct {
    summary: TrainStepSummary,
    /// dL/d(features): [total_tokens * hidden_size] — owned, caller must free.
    /// null if no gradient was available.
    features_grad: ?[]f32,
    boundary_features_grad_stats: BoundaryProbeTensorStats = .{},
    contrastive_features_grad_stats: BoundaryProbeTensorStats = .{},
    combined_features_grad_stats: BoundaryProbeTensorStats = .{},

    pub fn deinit(self: *TrainStepWithGradSummary, allocator: std.mem.Allocator) void {
        if (self.features_grad) |g| allocator.free(g);
        self.* = undefined;
    }
};

pub const BoundaryStepDebugSummary = struct {
    boundary_loss: f32 = 0,
    grad_norm_w1: f32 = 0,
    grad_norm_b1: f32 = 0,
    grad_norm_w2: f32 = 0,
    grad_norm_b2: f32 = 0,
    grad_max_abs_w1: f32 = 0,
    grad_max_abs_b1: f32 = 0,
    grad_max_abs_w2: f32 = 0,
    grad_max_abs_b2: f32 = 0,
    features_grad_norm: f32 = 0,
    features_grad_max_abs: f32 = 0,
    has_features_grad: bool = false,
    eval_predicted_positives: u64 = 0,
    eval_tp: u64 = 0,
    eval_fp: u64 = 0,
    eval_fn: u64 = 0,
    eval_f1: f32 = 0,
    eval_mean_prob_gold_positive: f32 = 0,
    eval_mean_prob_gold_negative: f32 = 0,
    boundary_forward_probe: BoundaryForwardProbe = .{},
    boundary_checkpoint_probe: BoundaryCheckpointProbe = .{},
};

pub const BoundaryProbeTensorStats = struct {
    elems: u64 = 0,
    mean: f32 = 0,
    rms: f32 = 0,
    max_abs: f32 = 0,
    max_abs_index: u64 = 0,
    max_abs_value: f32 = 0,
    hash: u64 = tensor_probe_hash_offset,
    sample_len: u8 = 0,
    sample: [16]f32 = [_]f32{0} ** 16,
    top_abs_len: u8 = 0,
    top_abs_indices: [tensor_probe_top_abs_count]u64 = [_]u64{0} ** tensor_probe_top_abs_count,
    top_abs_values: [tensor_probe_top_abs_count]f32 = [_]f32{0} ** tensor_probe_top_abs_count,
};

pub const BoundaryForwardProbe = struct {
    final_norm_input: BoundaryProbeTensorStats = .{},
    boundary_head_input: BoundaryProbeTensorStats = .{},
    dense1_pre_activation: BoundaryProbeTensorStats = .{},
    dense1_post_activation: BoundaryProbeTensorStats = .{},
    logits: BoundaryProbeTensorStats = .{},
};

pub const BoundaryCheckpointProbe = struct {
    w1: BoundaryProbeTensorStats = .{},
    b1: BoundaryProbeTensorStats = .{},
    w2: BoundaryProbeTensorStats = .{},
    b2: BoundaryProbeTensorStats = .{},
};

// ----------------------------------------------------------------------------
// EvalSummary
// ----------------------------------------------------------------------------

pub const EvalSummary = struct {
    pub const diagnostic_threshold_count = 9;
    pub const histogram_bucket_count = 10;

    pub const ThresholdPoint = struct {
        threshold: f32 = 0,
        f1: f32 = 0,
        precision: f32 = 0,
        recall: f32 = 0,
        tp: u64 = 0,
        fp: u64 = 0,
        fn_: u64 = 0,
        predicted_positive_rate: f32 = 0,
    };

    boundary_f1: f32 = 0,
    boundary_precision: f32 = 0,
    boundary_recall: f32 = 0,
    boundary_tp: u64 = 0,
    boundary_fp: u64 = 0,
    boundary_fn: u64 = 0,
    best_boundary_f1: f32 = 0,
    best_boundary_precision: f32 = 0,
    best_boundary_recall: f32 = 0,
    best_boundary_threshold: f32 = 0.5,
    best_boundary_tp: u64 = 0,
    best_boundary_fp: u64 = 0,
    best_boundary_fn: u64 = 0,
    valid_tokens: u64 = 0,
    gold_positives: u64 = 0,
    gold_positive_rate: f32 = 0,
    predicted_positives: u64 = 0,
    predicted_positive_rate: f32 = 0,
    best_predicted_positives: u64 = 0,
    best_predicted_positive_rate: f32 = 0,
    mean_positive_probability_gold_positive: f32 = 0,
    mean_positive_probability_gold_negative: f32 = 0,
    mean_boundary_margin_gold_positive: f32 = 0,
    mean_boundary_margin_gold_negative: f32 = 0,
    mean_logit0_gold_positive: f32 = 0,
    mean_logit1_gold_positive: f32 = 0,
    mean_logit0_gold_negative: f32 = 0,
    mean_logit1_gold_negative: f32 = 0,
    num_batches: u32 = 0,
    threshold_points: [diagnostic_threshold_count]ThresholdPoint = [_]ThresholdPoint{.{}} ** diagnostic_threshold_count,
    probability_histogram_gold_positive: [histogram_bucket_count]u64 = [_]u64{0} ** histogram_bucket_count,
    probability_histogram_gold_negative: [histogram_bucket_count]u64 = [_]u64{0} ** histogram_bucket_count,
};

pub const BoundaryEvalAccumulator = struct {
    pub const sweep_count = 101;
    pub const diagnostic_thresholds = [_]f32{ 0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50 };

    agg: fused_chunker_loss.BoundaryMetrics = .{ .tp = 0, .fp = 0, .fn_ = 0 },
    sweep_metrics: [sweep_count]fused_chunker_loss.BoundaryMetrics = [_]fused_chunker_loss.BoundaryMetrics{.{ .tp = 0, .fp = 0, .fn_ = 0 }} ** sweep_count,
    valid_tokens: u64 = 0,
    gold_positives: u64 = 0,
    prob_sum_gold_positive: f64 = 0,
    prob_sum_gold_negative: f64 = 0,
    margin_sum_gold_positive: f64 = 0,
    margin_sum_gold_negative: f64 = 0,
    logit0_sum_gold_positive: f64 = 0,
    logit1_sum_gold_positive: f64 = 0,
    logit0_sum_gold_negative: f64 = 0,
    logit1_sum_gold_negative: f64 = 0,
    probability_histogram_gold_positive: [EvalSummary.histogram_bucket_count]u64 = [_]u64{0} ** EvalSummary.histogram_bucket_count,
    probability_histogram_gold_negative: [EvalSummary.histogram_bucket_count]u64 = [_]u64{0} ** EvalSummary.histogram_bucket_count,
    num_batches: u32 = 0,

    fn probabilityBucket(prob: f32) usize {
        if (!std.math.isFinite(prob) or prob <= 0) return 0;
        if (prob >= 1) return EvalSummary.histogram_bucket_count - 1;
        return @min(@as(usize, @intFromFloat(prob * @as(f32, @floatFromInt(EvalSummary.histogram_bucket_count)))), EvalSummary.histogram_bucket_count - 1);
    }

    pub fn addLogits(
        self: *BoundaryEvalAccumulator,
        allocator: std.mem.Allocator,
        logits: []const f32,
        labels: []const f32,
        mask: ?[]const f32,
    ) !void {
        if (labels.len % 2 != 0) return error.InvalidBoundaryLabelShape;
        const total = labels.len / 2;
        const scalar_labels = try allocator.alloc(f32, total);
        defer allocator.free(scalar_labels);
        for (0..total) |i| {
            scalar_labels[i] = if (labels[i * 2 + 1] > 0.5) 1.0 else 0.0;
        }

        for (0..total) |i| {
            if (mask) |m| {
                if (m[i] <= 0.5) continue;
            }
            self.valid_tokens += 1;
            const logit0 = logits[i * 2];
            const logit1 = logits[i * 2 + 1];
            const margin = logit1 - logit0;
            const prob = fused_chunker_loss.positiveBoundaryProbability(logit0, logit1);
            const bucket = probabilityBucket(prob);
            if (scalar_labels[i] > 0.5) {
                self.gold_positives += 1;
                self.prob_sum_gold_positive += prob;
                self.margin_sum_gold_positive += margin;
                self.logit0_sum_gold_positive += logit0;
                self.logit1_sum_gold_positive += logit1;
                self.probability_histogram_gold_positive[bucket] += 1;
            } else {
                self.prob_sum_gold_negative += prob;
                self.margin_sum_gold_negative += margin;
                self.logit0_sum_gold_negative += logit0;
                self.logit1_sum_gold_negative += logit1;
                self.probability_histogram_gold_negative[bucket] += 1;
            }
        }

        const metrics = fused_chunker_loss.computeBoundaryMetrics(logits, scalar_labels, mask);
        self.agg.tp += metrics.tp;
        self.agg.fp += metrics.fp;
        self.agg.fn_ += metrics.fn_;

        for (&self.sweep_metrics, 0..) |*sweep, threshold_idx| {
            const threshold = @as(f32, @floatFromInt(threshold_idx)) / @as(f32, @floatFromInt(sweep_count - 1));
            const threshold_metrics = fused_chunker_loss.computeBoundaryMetricsWithThreshold(logits, scalar_labels, mask, threshold);
            sweep.tp += threshold_metrics.tp;
            sweep.fp += threshold_metrics.fp;
            sweep.fn_ += threshold_metrics.fn_;
        }
        self.num_batches += 1;
    }

    pub fn finish(self: BoundaryEvalAccumulator) EvalSummary {
        var best_idx: usize = 50;
        var best_metrics = self.sweep_metrics[best_idx];
        var best_f1 = best_metrics.f1();
        for (self.sweep_metrics, 0..) |metrics, threshold_idx| {
            const f1 = metrics.f1();
            if (f1 > best_f1) {
                best_idx = threshold_idx;
                best_metrics = metrics;
                best_f1 = f1;
            }
        }
        const best_threshold = @as(f32, @floatFromInt(best_idx)) / @as(f32, @floatFromInt(sweep_count - 1));
        const predicted_positives = self.agg.tp + self.agg.fp;
        const best_predicted_positives = best_metrics.tp + best_metrics.fp;
        const valid_f: f32 = if (self.valid_tokens == 0) 0 else @floatFromInt(self.valid_tokens);
        const gold_positive_rate = if (self.valid_tokens == 0)
            0.0
        else
            @as(f32, @floatFromInt(self.gold_positives)) / valid_f;
        const predicted_positive_rate = if (self.valid_tokens == 0)
            0.0
        else
            @as(f32, @floatFromInt(predicted_positives)) / valid_f;
        const best_predicted_positive_rate = if (self.valid_tokens == 0)
            0.0
        else
            @as(f32, @floatFromInt(best_predicted_positives)) / valid_f;
        const gold_negatives = self.valid_tokens - self.gold_positives;
        const mean_pos_prob_gold_positive = if (self.gold_positives == 0)
            0.0
        else
            @as(f32, @floatCast(self.prob_sum_gold_positive / @as(f64, @floatFromInt(self.gold_positives))));
        const mean_pos_prob_gold_negative = if (gold_negatives == 0)
            0.0
        else
            @as(f32, @floatCast(self.prob_sum_gold_negative / @as(f64, @floatFromInt(gold_negatives))));
        const mean_margin_gold_positive = if (self.gold_positives == 0)
            0.0
        else
            @as(f32, @floatCast(self.margin_sum_gold_positive / @as(f64, @floatFromInt(self.gold_positives))));
        const mean_margin_gold_negative = if (gold_negatives == 0)
            0.0
        else
            @as(f32, @floatCast(self.margin_sum_gold_negative / @as(f64, @floatFromInt(gold_negatives))));
        const mean_logit0_gold_positive = if (self.gold_positives == 0)
            0.0
        else
            @as(f32, @floatCast(self.logit0_sum_gold_positive / @as(f64, @floatFromInt(self.gold_positives))));
        const mean_logit1_gold_positive = if (self.gold_positives == 0)
            0.0
        else
            @as(f32, @floatCast(self.logit1_sum_gold_positive / @as(f64, @floatFromInt(self.gold_positives))));
        const mean_logit0_gold_negative = if (gold_negatives == 0)
            0.0
        else
            @as(f32, @floatCast(self.logit0_sum_gold_negative / @as(f64, @floatFromInt(gold_negatives))));
        const mean_logit1_gold_negative = if (gold_negatives == 0)
            0.0
        else
            @as(f32, @floatCast(self.logit1_sum_gold_negative / @as(f64, @floatFromInt(gold_negatives))));
        var threshold_points: [EvalSummary.diagnostic_threshold_count]EvalSummary.ThresholdPoint = [_]EvalSummary.ThresholdPoint{.{}} ** EvalSummary.diagnostic_threshold_count;
        for (diagnostic_thresholds, 0..) |threshold, i| {
            const sweep_idx = @min(@as(usize, @intFromFloat(@round(threshold * @as(f32, @floatFromInt(sweep_count - 1))))), sweep_count - 1);
            const metrics = self.sweep_metrics[sweep_idx];
            const predicted = metrics.tp + metrics.fp;
            threshold_points[i] = .{
                .threshold = @as(f32, @floatFromInt(sweep_idx)) / @as(f32, @floatFromInt(sweep_count - 1)),
                .f1 = metrics.f1(),
                .precision = metrics.precision(),
                .recall = metrics.recall(),
                .tp = metrics.tp,
                .fp = metrics.fp,
                .fn_ = metrics.fn_,
                .predicted_positive_rate = if (self.valid_tokens == 0) 0.0 else @as(f32, @floatFromInt(predicted)) / valid_f,
            };
        }

        return EvalSummary{
            .boundary_f1 = self.agg.f1(),
            .boundary_precision = self.agg.precision(),
            .boundary_recall = self.agg.recall(),
            .boundary_tp = self.agg.tp,
            .boundary_fp = self.agg.fp,
            .boundary_fn = self.agg.fn_,
            .best_boundary_f1 = best_f1,
            .best_boundary_precision = best_metrics.precision(),
            .best_boundary_recall = best_metrics.recall(),
            .best_boundary_threshold = best_threshold,
            .best_boundary_tp = best_metrics.tp,
            .best_boundary_fp = best_metrics.fp,
            .best_boundary_fn = best_metrics.fn_,
            .valid_tokens = self.valid_tokens,
            .gold_positives = self.gold_positives,
            .gold_positive_rate = gold_positive_rate,
            .predicted_positives = predicted_positives,
            .predicted_positive_rate = predicted_positive_rate,
            .best_predicted_positives = best_predicted_positives,
            .best_predicted_positive_rate = best_predicted_positive_rate,
            .mean_positive_probability_gold_positive = mean_pos_prob_gold_positive,
            .mean_positive_probability_gold_negative = mean_pos_prob_gold_negative,
            .mean_boundary_margin_gold_positive = mean_margin_gold_positive,
            .mean_boundary_margin_gold_negative = mean_margin_gold_negative,
            .mean_logit0_gold_positive = mean_logit0_gold_positive,
            .mean_logit1_gold_positive = mean_logit1_gold_positive,
            .mean_logit0_gold_negative = mean_logit0_gold_negative,
            .mean_logit1_gold_negative = mean_logit1_gold_negative,
            .num_batches = self.num_batches,
            .threshold_points = threshold_points,
            .probability_histogram_gold_positive = self.probability_histogram_gold_positive,
            .probability_histogram_gold_negative = self.probability_histogram_gold_negative,
        };
    }
};

pub fn printBoundaryQualityDiagnostics(label: []const u8, summary: EvalSummary) void {
    std.debug.print("{s} margin_means gold_pos={d:.6} gold_neg={d:.6} logit0_pos={d:.6} logit1_pos={d:.6} logit0_neg={d:.6} logit1_neg={d:.6}\n", .{
        label,
        summary.mean_boundary_margin_gold_positive,
        summary.mean_boundary_margin_gold_negative,
        summary.mean_logit0_gold_positive,
        summary.mean_logit1_gold_positive,
        summary.mean_logit0_gold_negative,
        summary.mean_logit1_gold_negative,
    });

    std.debug.print("{s} threshold_sweep", .{label});
    for (summary.threshold_points) |point| {
        std.debug.print(" t={d:.2}:f1={d:.4},p={d:.4},r={d:.4},pred={d:.6}", .{
            point.threshold,
            point.f1,
            point.precision,
            point.recall,
            point.predicted_positive_rate,
        });
    }
    std.debug.print("\n", .{});

    std.debug.print("{s} prob_hist gold_pos", .{label});
    for (summary.probability_histogram_gold_positive, 0..) |count, i| {
        std.debug.print(" [{d:.1},{d:.1})={d}", .{
            @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(EvalSummary.histogram_bucket_count)),
            @as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(EvalSummary.histogram_bucket_count)),
            count,
        });
    }
    std.debug.print(" | gold_neg", .{});
    for (summary.probability_histogram_gold_negative, 0..) |count, i| {
        std.debug.print(" [{d:.1},{d:.1})={d}", .{
            @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(EvalSummary.histogram_bucket_count)),
            @as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(EvalSummary.histogram_bucket_count)),
            count,
        });
    }
    std.debug.print("\n", .{});
}

// ----------------------------------------------------------------------------
// FusedTrainer
// ----------------------------------------------------------------------------

pub const FusedTrainer = struct {
    allocator: std.mem.Allocator,
    config: FusedTrainingConfig,
    loss_config: fused_chunker_loss.FusedLossConfig,
    cb: *const ComputeBackend,

    // Head weights
    boundary_head: BoundaryHead,
    legacy_dense_boundary_head: ?LegacyDenseBoundaryHead = null,

    // Optimizer
    optimizer: optimizers.Optimizer,
    optimizer_state: optimizers.OptimizerState,
    step_count: u32 = 0,
    lr_schedule: optimizers.LearningRateSchedule,

    // LoRA adapters (optional — set externally before training begins)
    lora_adapters: ?LoRAAdapterSet = null,

    // Cross-Batch Memory (Feature 1)
    xbm: ?CrossBatchMemory = null,
    // Monotonic base added to doc_ids before XBM storage; incremented by a large
    // prime each batch to guarantee globally unique IDs across stored generations
    // (Fix 1: XBM doc_id collision).
    xbm_doc_id_base: u64 = 0,

    // Gradient accumulation (Feature 2)
    accum_count: u32 = 0,
    grad_accum_w1: []f32 = &.{},
    grad_accum_b1: []f32 = &.{},
    grad_accum_w2: []f32 = &.{},
    grad_accum_b2: []f32 = &.{},

    // Schedule-Free AdamW states (Feature 3)
    sf_state_w1: ?ScheduleFreeAdamWState = null,
    sf_state_b1: ?ScheduleFreeAdamWState = null,
    sf_state_w2: ?ScheduleFreeAdamWState = null,
    sf_state_b2: ?ScheduleFreeAdamWState = null,

    // LoRA features_grad accumulator for grad_accum_steps > 1 (Fix 4)
    grad_accum_lora_features_grad_accum: ?[]f32 = null,

    // Shape-keyed cache for the boundary-head training graph. The common
    // training path uses a fixed [batch * max_seq_len] shape, so compiling the
    // autodiff graph once avoids rebuilding/lowering it every step.
    boundary_graph_cache: std.AutoHashMapUnmanaged(usize, BoundaryGraphCacheEntry) = .{},

    const BoundaryGraphCacheEntry = struct {
        graph: fused_chunker_loss.BoundaryHeadGraph,
        head_session: ?training.CompiledTrainSession = null,
        encoder_grad_session: ?training.CompiledTrainSession = null,

        fn deinit(self: *BoundaryGraphCacheEntry) void {
            if (self.head_session) |*session| session.deinit();
            if (self.encoder_grad_session) |*session| session.deinit();
            self.graph.deinit();
            self.* = undefined;
        }
    };

    pub fn init(
        allocator: std.mem.Allocator,
        config: FusedTrainingConfig,
        cb: *const ComputeBackend,
    ) !FusedTrainer {
        const loss_config = fused_chunker_loss.FusedLossConfig{
            .lambda_chunk = config.lambda_chunk,
            .lambda_embed = config.lambda_embed,
            .use_focal = config.use_boundary_focal,
            .focal_gamma = config.focal_gamma,
            .focal_alpha = config.focal_alpha,
            .contrastive_focal_gamma = config.contrastive_focal_gamma,
            .contrastive_focal_alpha = config.contrastive_focal_alpha,
            .temperature = config.temperature,
            .pos_weight = config.pos_weight,
            .enable_splade = config.enable_splade,
            .lambda_splade = config.lambda_splade,
            .lambda_flops = config.lambda_flops,
            .splade_focus_epoch = config.splade_focus_epoch,
            .use_mrl = config.use_mrl,
            .mrl_dims = config.mrl_dims,
            .mrl_weights = config.mrl_weights,
        };

        var head = try BoundaryHead.initWithSeed(
            allocator,
            @intCast(config.hidden_size),
            @intCast(config.boundary_mlp_dim),
            config.seed,
        );
        errdefer head.deinit();

        const optimizer = optimizers.Optimizer{ .adamw = .{
            .beta1 = config.beta1,
            .beta2 = config.beta2,
            .eps = config.adam_epsilon,
            .weight_decay = config.weight_decay,
        } };

        const opt_state = optimizers.OptimizerState.init(allocator);
        const lr_schedule = config.lrSchedule();

        // Gradient accumulation buffers (Feature 2)
        const accum_w1 = try allocator.alloc(f32, head.w1.len);
        errdefer allocator.free(accum_w1);
        @memset(accum_w1, 0);
        const accum_b1 = try allocator.alloc(f32, head.b1.len);
        errdefer allocator.free(accum_b1);
        @memset(accum_b1, 0);
        const accum_w2 = try allocator.alloc(f32, head.w2.len);
        errdefer allocator.free(accum_w2);
        @memset(accum_w2, 0);
        const accum_b2 = try allocator.alloc(f32, head.b2.len);
        errdefer allocator.free(accum_b2);
        @memset(accum_b2, 0);

        // Schedule-Free AdamW states (Feature 3)
        var sf_w1: ?ScheduleFreeAdamWState = null;
        var sf_b1: ?ScheduleFreeAdamWState = null;
        var sf_w2: ?ScheduleFreeAdamWState = null;
        var sf_b2: ?ScheduleFreeAdamWState = null;
        if (config.use_schedule_free) {
            sf_w1 = try ScheduleFreeAdamWState.init(allocator, head.w1);
            errdefer if (sf_w1) |*s| s.deinit();
            sf_b1 = try ScheduleFreeAdamWState.init(allocator, head.b1);
            errdefer if (sf_b1) |*s| s.deinit();
            sf_w2 = try ScheduleFreeAdamWState.init(allocator, head.w2);
            errdefer if (sf_w2) |*s| s.deinit();
            sf_b2 = try ScheduleFreeAdamWState.init(allocator, head.b2);
            errdefer if (sf_b2) |*s| s.deinit();
        }

        // Cross-Batch Memory (Feature 1)
        var xbm: ?CrossBatchMemory = null;
        if (config.xbm_capacity > 0) {
            xbm = try CrossBatchMemory.init(allocator, config.xbm_capacity, config.embedding_dim);
            errdefer if (xbm) |*x| x.deinit();
        }

        return .{
            .allocator = allocator,
            .config = config,
            .loss_config = loss_config,
            .cb = cb,
            .boundary_head = head,
            .optimizer = optimizer,
            .optimizer_state = opt_state,
            .step_count = 0,
            .lr_schedule = lr_schedule,
            .xbm = xbm,
            .accum_count = 0,
            .grad_accum_w1 = accum_w1,
            .grad_accum_b1 = accum_b1,
            .grad_accum_w2 = accum_w2,
            .grad_accum_b2 = accum_b2,
            .sf_state_w1 = sf_w1,
            .sf_state_b1 = sf_b1,
            .sf_state_w2 = sf_w2,
            .sf_state_b2 = sf_b2,
        };
    }

    pub fn deinit(self: *FusedTrainer) void {
        self.boundary_head.deinit();
        if (self.legacy_dense_boundary_head) |*head| head.deinit();
        self.optimizer_state.deinit();
        // Gradient accumulation buffers (Feature 2)
        self.allocator.free(self.grad_accum_w1);
        self.allocator.free(self.grad_accum_b1);
        self.allocator.free(self.grad_accum_w2);
        self.allocator.free(self.grad_accum_b2);
        // Schedule-Free states (Feature 3)
        if (self.sf_state_w1) |*s| s.deinit();
        if (self.sf_state_b1) |*s| s.deinit();
        if (self.sf_state_w2) |*s| s.deinit();
        if (self.sf_state_b2) |*s| s.deinit();
        // Cross-Batch Memory (Feature 1)
        if (self.xbm) |*x| x.deinit();
        // LoRA features_grad accumulator (Fix 4)
        if (self.grad_accum_lora_features_grad_accum) |buf| self.allocator.free(buf);
        var cache_it = self.boundary_graph_cache.iterator();
        while (cache_it.next()) |entry| entry.value_ptr.deinit();
        self.boundary_graph_cache.deinit(self.allocator);
        self.* = undefined;
    }

    fn sanitizeGradientBufferInPlace(values: []f32) void {
        _ = optimizers.sanitizeSlice(values);
    }

    fn sanitizeBoundaryStepGradients(step: *BoundaryTrainStepResult) void {
        sanitizeGradientBufferInPlace(step.w1_grad);
        sanitizeGradientBufferInPlace(step.b1_grad);
        sanitizeGradientBufferInPlace(step.w2_grad);
        sanitizeGradientBufferInPlace(step.b2_grad);
        if (step.features_grad) |grad| sanitizeGradientBufferInPlace(grad);
    }

    fn sanitizeBoundaryHeadParameters(self: *FusedTrainer) void {
        sanitizeGradientBufferInPlace(self.boundary_head.w1);
        sanitizeGradientBufferInPlace(self.boundary_head.b1);
        sanitizeGradientBufferInPlace(self.boundary_head.w2);
        sanitizeGradientBufferInPlace(self.boundary_head.b2);
        if (self.legacy_dense_boundary_head) |*head| {
            sanitizeGradientBufferInPlace(head.weight);
            sanitizeGradientBufferInPlace(head.bias);
        }
    }

    fn addGradientNormSq(total_sq: *f64, values: []const f32) void {
        for (values) |value| {
            const v: f64 = @floatCast(value);
            total_sq.* += v * v;
        }
    }

    fn scaleGradientBuffer(values: []f32, scale: f32) void {
        for (values) |*value| value.* *= scale;
    }

    fn resetBoundaryGradientAccum(self: *FusedTrainer) void {
        @memset(self.grad_accum_w1, 0);
        @memset(self.grad_accum_b1, 0);
        @memset(self.grad_accum_w2, 0);
        @memset(self.grad_accum_b2, 0);
    }

    fn sanitizeAndClipBoundaryGradientAccum(self: *FusedTrainer) void {
        sanitizeGradientBufferInPlace(self.grad_accum_w1);
        sanitizeGradientBufferInPlace(self.grad_accum_b1);
        sanitizeGradientBufferInPlace(self.grad_accum_w2);
        sanitizeGradientBufferInPlace(self.grad_accum_b2);

        var total_sq: f64 = 0;
        addGradientNormSq(&total_sq, self.grad_accum_w1);
        addGradientNormSq(&total_sq, self.grad_accum_b1);
        addGradientNormSq(&total_sq, self.grad_accum_w2);
        addGradientNormSq(&total_sq, self.grad_accum_b2);

        if (!std.math.isFinite(total_sq)) {
            self.resetBoundaryGradientAccum();
            return;
        }

        const total_norm: f32 = @floatCast(@sqrt(total_sq));
        if (!std.math.isFinite(total_norm) or total_norm == 0.0) return;
        if (self.config.max_grad_norm > 0.0 and total_norm > self.config.max_grad_norm) {
            const grad_scale = self.config.max_grad_norm / (total_norm + 1e-6);
            scaleGradientBuffer(self.grad_accum_w1, grad_scale);
            scaleGradientBuffer(self.grad_accum_b1, grad_scale);
            scaleGradientBuffer(self.grad_accum_w2, grad_scale);
            scaleGradientBuffer(self.grad_accum_b2, grad_scale);
        }
    }

    fn sanitizeScheduleFreeState(state: *ScheduleFreeAdamWState) void {
        sanitizeGradientBufferInPlace(state.z);
        sanitizeGradientBufferInPlace(state.v);
    }

    fn sanitizeBoundaryOptimizerState(self: *FusedTrainer) void {
        _ = optimizers.sanitizeState(&self.optimizer_state);
        if (self.sf_state_w1) |*state| sanitizeScheduleFreeState(state);
        if (self.sf_state_b1) |*state| sanitizeScheduleFreeState(state);
        if (self.sf_state_w2) |*state| sanitizeScheduleFreeState(state);
        if (self.sf_state_b2) |*state| sanitizeScheduleFreeState(state);
    }

    fn applyAccumulatedBoundaryHeadStep(self: *FusedTrainer, lr: f32) !void {
        self.sanitizeAndClipBoundaryGradientAccum();
        self.sanitizeBoundaryOptimizerState();
        if (self.config.use_schedule_free) {
            scheduleFreeAdamWStep(self.boundary_head.w1, self.grad_accum_w1, &self.sf_state_w1.?, lr, self.config.beta1, self.config.beta2, self.config.adam_epsilon, self.config.weight_decay, self.config.warmup_steps);
            scheduleFreeAdamWStep(self.boundary_head.b1, self.grad_accum_b1, &self.sf_state_b1.?, lr, self.config.beta1, self.config.beta2, self.config.adam_epsilon, self.config.weight_decay, self.config.warmup_steps);
            scheduleFreeAdamWStep(self.boundary_head.w2, self.grad_accum_w2, &self.sf_state_w2.?, lr, self.config.beta1, self.config.beta2, self.config.adam_epsilon, self.config.weight_decay, self.config.warmup_steps);
            scheduleFreeAdamWStep(self.boundary_head.b2, self.grad_accum_b2, &self.sf_state_b2.?, lr, self.config.beta1, self.config.beta2, self.config.adam_epsilon, self.config.weight_decay, self.config.warmup_steps);
        } else {
            try optimizers.step(self.optimizer, &self.optimizer_state, lr, "w1", self.boundary_head.w1, self.grad_accum_w1);
            try optimizers.step(self.optimizer, &self.optimizer_state, lr, "b1", self.boundary_head.b1, self.grad_accum_b1);
            try optimizers.step(self.optimizer, &self.optimizer_state, lr, "w2", self.boundary_head.w2, self.grad_accum_w2);
            try optimizers.step(self.optimizer, &self.optimizer_state, lr, "b2", self.boundary_head.b2, self.grad_accum_b2);
        }
        self.sanitizeBoundaryOptimizerState();
        self.sanitizeBoundaryHeadParameters();
        self.resetBoundaryGradientAccum();
        self.accum_count = 0;
    }

    fn getBoundaryGraphCacheEntry(self: *FusedTrainer, total_tokens: usize) !*BoundaryGraphCacheEntry {
        if (self.boundary_graph_cache.getPtr(total_tokens)) |entry| return entry;

        var graph = try fused_chunker_loss.BoundaryHeadGraph.init(
            self.allocator,
            total_tokens,
            self.config.hidden_size,
            self.config.boundary_mlp_dim,
            self.loss_config.pos_weight,
            self.loss_config.use_focal,
            self.loss_config.focal_gamma,
            self.loss_config.focal_alpha,
            self.config.boundary_dropout,
        );
        errdefer graph.deinit();

        try self.boundary_graph_cache.put(self.allocator, total_tokens, .{
            .graph = graph,
        });
        return self.boundary_graph_cache.getPtr(total_tokens).?;
    }

    fn getHeadTrainSession(self: *FusedTrainer, entry: *BoundaryGraphCacheEntry) !*training.CompiledTrainSession {
        if (entry.head_session == null) {
            entry.head_session = try training.CompiledTrainSession.init(
                self.allocator,
                &entry.graph.graph,
                entry.graph.loss_id,
                .{ .trainable_params = &.{ "w1", "b1", "w2", "b2" } },
            );
        }
        return &entry.head_session.?;
    }

    fn getEncoderGradTrainSession(self: *FusedTrainer, entry: *BoundaryGraphCacheEntry) !*training.CompiledTrainSession {
        if (entry.encoder_grad_session == null) {
            entry.encoder_grad_session = try training.CompiledTrainSession.init(
                self.allocator,
                &entry.graph.graph,
                entry.graph.loss_id,
                .{ .trainable_params = &.{ "features", "w1", "b1", "w2", "b2" } },
            );
        }
        return &entry.encoder_grad_session.?;
    }

    const BoundaryTrainStepResult = struct {
        boundary_loss: f32,
        w1_grad: []f32,
        b1_grad: []f32,
        w2_grad: []f32,
        b2_grad: []f32,
        features_grad: ?[]f32 = null,

        fn deinit(self: *BoundaryTrainStepResult, allocator: std.mem.Allocator) void {
            allocator.free(self.w1_grad);
            allocator.free(self.b1_grad);
            allocator.free(self.w2_grad);
            allocator.free(self.b2_grad);
            if (self.features_grad) |grad| allocator.free(grad);
            self.* = undefined;
        }

        fn takeFeaturesGrad(self: *BoundaryTrainStepResult) ?[]f32 {
            const grad = self.features_grad;
            self.features_grad = null;
            return grad;
        }

        fn gradNorm(self: *const BoundaryTrainStepResult) f64 {
            var total_sq: f64 = 0;
            addGradientNormSq(&total_sq, self.w1_grad);
            addGradientNormSq(&total_sq, self.b1_grad);
            addGradientNormSq(&total_sq, self.w2_grad);
            addGradientNormSq(&total_sq, self.b2_grad);
            return @sqrt(total_sq);
        }
    };

    fn runBoundaryHeadTraining(
        self: *FusedTrainer,
        allocator: std.mem.Allocator,
        features: []const f32,
        boundary_labels: []const f32,
        attention_mask: []const f32,
        total_tokens: usize,
        want_features_grad: bool,
    ) !BoundaryTrainStepResult {
        if (!self.loss_config.use_focal and self.cb.kind() == .metal) {
            return self.runBoundaryHeadCeOpsStep(
                allocator,
                features,
                boundary_labels,
                attention_mask,
                total_tokens,
                want_features_grad,
            ) catch |err| {
                std.log.debug("fused_chunker boundary CE Metal fast path failed with {s}; falling back to graph", .{@errorName(err)});
                return self.runBoundaryHeadGraphStep(
                    allocator,
                    features,
                    boundary_labels,
                    attention_mask,
                    total_tokens,
                    want_features_grad,
                );
            };
        }

        return self.runBoundaryHeadGraphStep(
            allocator,
            features,
            boundary_labels,
            attention_mask,
            total_tokens,
            want_features_grad,
        );
    }

    fn runBoundaryHeadGraphStep(
        self: *FusedTrainer,
        allocator: std.mem.Allocator,
        features: []const f32,
        boundary_labels: []const f32,
        attention_mask: []const f32,
        total_tokens: usize,
        want_features_grad: bool,
    ) !BoundaryTrainStepResult {
        const graph_entry = try self.getBoundaryGraphCacheEntry(total_tokens);
        const graph = &graph_entry.graph;
        const train_session = if (want_features_grad)
            try self.getEncoderGradTrainSession(graph_entry)
        else
            try self.getHeadTrainSession(graph_entry);

        const feature_dropout_mask = try allocInvertedDropoutMask(
            allocator,
            total_tokens * @as(usize, @intCast(self.config.hidden_size)),
            self.config.boundary_dropout,
            dropoutSeed(self.config.seed, self.step_count, 0xB0A0_DF00_DF00_0001),
        );
        defer allocator.free(feature_dropout_mask);
        const hidden_dropout_mask = try allocInvertedDropoutMask(
            allocator,
            total_tokens * @as(usize, @intCast(self.config.boundary_mlp_dim)),
            self.config.boundary_dropout,
            dropoutSeed(self.config.seed, self.step_count, 0xB0A0_DF00_DF00_0002),
        );
        defer allocator.free(hidden_dropout_mask);

        var rt = std.AutoHashMapUnmanaged(NodeId, CT){};
        defer {
            var it = rt.iterator();
            while (it.next()) |e| self.cb.free(e.value_ptr.*);
            rt.deinit(allocator);
        }

        try putRuntimeInput(allocator, self.cb, &rt, graph.feature_id, features, &.{
            @intCast(total_tokens),
            @intCast(self.config.hidden_size),
        });
        try putRuntimeInput(allocator, self.cb, &rt, graph.target_id, boundary_labels, &.{
            @intCast(total_tokens),
            2,
        });
        try putRuntimeInput(allocator, self.cb, &rt, graph.mask_id, attention_mask, &.{
            @intCast(total_tokens),
            1,
        });
        try putRuntimeInput(allocator, self.cb, &rt, graph.feature_dropout_mask_id, feature_dropout_mask, &.{
            @intCast(total_tokens),
            @intCast(self.config.hidden_size),
        });
        try putRuntimeInput(allocator, self.cb, &rt, graph.hidden_dropout_mask_id, hidden_dropout_mask, &.{
            @intCast(total_tokens),
            @intCast(self.config.boundary_mlp_dim),
        });
        try putRuntimeInput(allocator, self.cb, &rt, graph.w1_id, self.boundary_head.w1, &.{
            @intCast(self.config.boundary_mlp_dim),
            @intCast(self.config.hidden_size),
        });
        try putRuntimeInput(allocator, self.cb, &rt, graph.b1_id, self.boundary_head.b1, &.{
            @intCast(self.config.boundary_mlp_dim),
        });
        try putRuntimeInput(allocator, self.cb, &rt, graph.w2_id, self.boundary_head.w2, &.{
            2,
            @intCast(self.config.boundary_mlp_dim),
        });
        try putRuntimeInput(allocator, self.cb, &rt, graph.b2_id, self.boundary_head.b2, &.{2});

        var step_result = try train_session.execute(self.cb, rt);
        defer step_result.deinit();

        const w1_src = step_result.gradients.get("w1") orelse return error.MissingGradient;
        const b1_src = step_result.gradients.get("b1") orelse return error.MissingGradient;
        const w2_src = step_result.gradients.get("w2") orelse return error.MissingGradient;
        const b2_src = step_result.gradients.get("b2") orelse return error.MissingGradient;

        const w1_grad = try allocator.dupe(f32, w1_src);
        errdefer allocator.free(w1_grad);
        const b1_grad = try allocator.dupe(f32, b1_src);
        errdefer allocator.free(b1_grad);
        const w2_grad = try allocator.dupe(f32, w2_src);
        errdefer allocator.free(w2_grad);
        const b2_grad = try allocator.dupe(f32, b2_src);
        errdefer allocator.free(b2_grad);

        var features_grad: ?[]f32 = null;
        errdefer if (features_grad) |g| allocator.free(g);
        if (want_features_grad) {
            if (step_result.gradients.get("features")) |g| {
                features_grad = try allocator.dupe(f32, g);
            }
        }
        return .{
            .boundary_loss = step_result.loss,
            .w1_grad = w1_grad,
            .b1_grad = b1_grad,
            .w2_grad = w2_grad,
            .b2_grad = b2_grad,
            .features_grad = features_grad,
        };
    }

    fn runBoundaryHeadCeOpsStep(
        self: *FusedTrainer,
        allocator: std.mem.Allocator,
        features: []const f32,
        boundary_labels: []const f32,
        attention_mask: []const f32,
        total_tokens: usize,
        want_features_grad: bool,
    ) !BoundaryTrainStepResult {
        const H: usize = @intCast(self.config.hidden_size);
        const M: usize = @intCast(self.config.boundary_mlp_dim);
        if (features.len != total_tokens * H) return error.UnexpectedOutputShape;
        if (boundary_labels.len != total_tokens * 2) return error.UnexpectedOutputShape;
        if (attention_mask.len != total_tokens) return error.UnexpectedOutputShape;

        const feature_dropout_mask = try allocInvertedDropoutMask(
            allocator,
            total_tokens * H,
            self.config.boundary_dropout,
            dropoutSeed(self.config.seed, self.step_count, 0xB0A0_DF00_DF00_0001),
        );
        defer allocator.free(feature_dropout_mask);
        const hidden_dropout_mask = try allocInvertedDropoutMask(
            allocator,
            total_tokens * M,
            self.config.boundary_dropout,
            dropoutSeed(self.config.seed, self.step_count, 0xB0A0_DF00_DF00_0002),
        );
        defer allocator.free(hidden_dropout_mask);

        const features_ct = try self.cb.fromFloat32Shape(features, &.{ @intCast(total_tokens), @intCast(H) });
        defer self.cb.free(features_ct);
        const feature_dropout_mask_ct = try self.cb.fromFloat32Shape(feature_dropout_mask, &.{ @intCast(total_tokens), @intCast(H) });
        defer self.cb.free(feature_dropout_mask_ct);
        const hidden_dropout_mask_ct = try self.cb.fromFloat32Shape(hidden_dropout_mask, &.{ @intCast(total_tokens), @intCast(M) });
        defer self.cb.free(hidden_dropout_mask_ct);
        const w1_ct = try self.cb.fromFloat32Shape(self.boundary_head.w1, &.{ @intCast(M), @intCast(H) });
        defer self.cb.free(w1_ct);
        const b1_ct = try self.cb.fromFloat32Shape(self.boundary_head.b1, &.{@intCast(M)});
        defer self.cb.free(b1_ct);
        const w2_ct = try self.cb.fromFloat32Shape(self.boundary_head.w2, &.{ 2, @intCast(M) });
        defer self.cb.free(w2_ct);
        const b2_ct = try self.cb.fromFloat32Shape(self.boundary_head.b2, &.{2});
        defer self.cb.free(b2_ct);

        const dropped_features_ct = try self.cb.multiply(features_ct, feature_dropout_mask_ct);
        defer self.cb.free(dropped_features_ct);
        const dense_ct = try self.cb.linear(dropped_features_ct, w1_ct, b1_ct, total_tokens, H, M);
        defer self.cb.free(dense_ct);
        const hidden_ct = try self.cb.gelu(dense_ct);
        defer self.cb.free(hidden_ct);
        const dropped_hidden_ct = try self.cb.multiply(hidden_ct, hidden_dropout_mask_ct);
        defer self.cb.free(dropped_hidden_ct);
        const logits_ct = try self.cb.linear(dropped_hidden_ct, w2_ct, b2_ct, total_tokens, M, 2);
        defer self.cb.free(logits_ct);

        var forward_cts = [_]CT{ logits_ct, dense_ct };
        const forward_host = try self.cb.toFloat32Batch(&forward_cts, allocator);
        defer {
            for (forward_host) |slice| allocator.free(slice);
            allocator.free(forward_host);
        }
        const logits = forward_host[0];
        const dense = forward_host[1];

        const ce = try computeBoundaryWeightedCeAndLogitGrad(
            allocator,
            logits,
            boundary_labels,
            attention_mask,
            total_tokens,
            self.loss_config.pos_weight,
        );
        defer allocator.free(ce.logit_grad);

        const gelu_deriv = try allocGeluExactDerivative(allocator, dense);
        defer allocator.free(gelu_deriv);

        const dlogits_ct = try self.cb.fromFloat32Shape(ce.logit_grad, &.{ @intCast(total_tokens), 2 });
        defer self.cb.free(dlogits_ct);
        const gelu_deriv_ct = try self.cb.fromFloat32Shape(gelu_deriv, &.{ @intCast(total_tokens), @intCast(M) });
        defer self.cb.free(gelu_deriv_ct);

        const total_i64: i64 = @intCast(total_tokens);
        const h_i64: i64 = @intCast(H);
        const m_i64: i64 = @intCast(M);
        const grad_w2_ct = try self.cb.primDotGeneral(
            dlogits_ct,
            dropped_hidden_ct,
            &.{ total_i64, 2 },
            &.{ total_i64, m_i64 },
            &.{0},
            &.{0},
            &.{},
            &.{},
        );
        defer self.cb.free(grad_w2_ct);
        const grad_b2_ct = try self.cb.primReduceSum(dlogits_ct, &.{0}, &.{ total_i64, 2 });
        defer self.cb.free(grad_b2_ct);
        const d_hidden_drop_ct = try self.cb.primDotGeneral(
            dlogits_ct,
            w2_ct,
            &.{ total_i64, 2 },
            &.{ 2, m_i64 },
            &.{1},
            &.{0},
            &.{},
            &.{},
        );
        defer self.cb.free(d_hidden_drop_ct);
        const d_hidden_ct = try self.cb.multiply(d_hidden_drop_ct, hidden_dropout_mask_ct);
        defer self.cb.free(d_hidden_ct);
        const d_dense_ct = try self.cb.multiply(d_hidden_ct, gelu_deriv_ct);
        defer self.cb.free(d_dense_ct);
        const grad_w1_ct = try self.cb.primDotGeneral(
            d_dense_ct,
            dropped_features_ct,
            &.{ total_i64, m_i64 },
            &.{ total_i64, h_i64 },
            &.{0},
            &.{0},
            &.{},
            &.{},
        );
        defer self.cb.free(grad_w1_ct);
        const grad_b1_ct = try self.cb.primReduceSum(d_dense_ct, &.{0}, &.{ total_i64, m_i64 });
        defer self.cb.free(grad_b1_ct);

        var features_grad_ct_opt: ?CT = null;
        defer if (features_grad_ct_opt) |ct| self.cb.free(ct);
        if (want_features_grad) {
            const raw_features_grad_ct = try self.cb.primDotGeneral(
                d_dense_ct,
                w1_ct,
                &.{ total_i64, m_i64 },
                &.{ m_i64, h_i64 },
                &.{1},
                &.{0},
                &.{},
                &.{},
            );
            defer self.cb.free(raw_features_grad_ct);
            features_grad_ct_opt = try self.cb.multiply(raw_features_grad_ct, feature_dropout_mask_ct);
        }

        var grad_cts_buf: [5]CT = undefined;
        grad_cts_buf[0] = grad_w1_ct;
        grad_cts_buf[1] = grad_b1_ct;
        grad_cts_buf[2] = grad_w2_ct;
        grad_cts_buf[3] = grad_b2_ct;
        var grad_ct_count: usize = 4;
        if (features_grad_ct_opt) |ct| {
            grad_cts_buf[grad_ct_count] = ct;
            grad_ct_count += 1;
        }

        const grad_host = try self.cb.toFloat32Batch(grad_cts_buf[0..grad_ct_count], allocator);
        errdefer {
            for (grad_host) |slice| allocator.free(slice);
            allocator.free(grad_host);
        }

        const out = BoundaryTrainStepResult{
            .boundary_loss = ce.loss,
            .w1_grad = grad_host[0],
            .b1_grad = grad_host[1],
            .w2_grad = grad_host[2],
            .b2_grad = grad_host[3],
            .features_grad = if (grad_ct_count == 5) grad_host[4] else null,
        };
        allocator.free(grad_host);
        return out;
    }

    const ContrastiveStepResult = struct {
        result: infonce_cpu.ContrastiveLossResult,
        xbm_expanded_embeddings: ?[]f32 = null,
        xbm_expanded_doc_ids: ?[]u32 = null,

        fn deinit(self: *ContrastiveStepResult, allocator: std.mem.Allocator) void {
            self.result.deinit(allocator);
            if (self.xbm_expanded_embeddings) |e| allocator.free(e);
            if (self.xbm_expanded_doc_ids) |d| allocator.free(d);
            self.* = undefined;
        }
    };

    fn mrlContrastiveResult(
        self: *FusedTrainer,
        allocator: std.mem.Allocator,
        mrl: *infonce_cpu.MatryoshkaResult,
    ) infonce_cpu.ContrastiveLossResult {
        allocator.free(mrl.per_scale_loss);
        mrl.per_scale_loss = &.{};
        for (mrl.grad) |*g| g.* *= self.loss_config.lambda_embed;
        const grad = mrl.grad;
        mrl.grad = &.{};
        return .{
            .contrastive_loss = mrl.total_loss,
            .total_loss = mrl.total_loss * @as(f64, self.loss_config.lambda_embed),
            .grad = grad,
        };
    }

    fn computeContrastiveStep(
        self: *FusedTrainer,
        allocator: std.mem.Allocator,
        chunk_embeddings: []const f32,
        chunk_mask: []const f32,
        doc_ids: []const u32,
        B: usize,
        C: usize,
        E: usize,
    ) !ContrastiveStepResult {
        var out = ContrastiveStepResult{
            .result = .{
                .contrastive_loss = 0,
                .total_loss = 0,
                .grad = &.{},
            },
        };
        var result_ready = false;
        errdefer {
            if (result_ready) out.result.deinit(allocator);
            if (out.xbm_expanded_embeddings) |e| allocator.free(e);
            if (out.xbm_expanded_doc_ids) |d| allocator.free(d);
        }

        if (self.xbm) |*xbm| {
            const stored = xbm.getStored();
            if (stored.count > 0) {
                const n_current = B * C;
                const n_total = n_current + stored.count;
                const expanded = try allocator.alloc(f32, n_total * E);
                out.xbm_expanded_embeddings = expanded;
                @memcpy(expanded[0 .. n_current * E], chunk_embeddings);
                @memcpy(expanded[n_current * E ..], stored.embeddings[0 .. stored.count * E]);

                const exp_ids = try allocator.alloc(u32, n_total);
                out.xbm_expanded_doc_ids = exp_ids;
                @memcpy(exp_ids[0..n_current], doc_ids);
                @memcpy(exp_ids[n_current..], stored.doc_ids[0..stored.count]);

                const exp_mask = try allocator.alloc(f32, n_total);
                defer allocator.free(exp_mask);
                @memcpy(exp_mask[0..n_current], chunk_mask[0..n_current]);
                @memset(exp_mask[n_current..], 1.0);

                out.result = if (self.loss_config.use_mrl) mrl_blk: {
                    const mrl_config = infonce_cpu.MatryoshkaConfig{
                        .dims = self.loss_config.mrl_dims,
                        .weights = self.loss_config.mrl_weights,
                    };
                    var mrl = try infonce_cpu.computeMatryoshkaLossAndGrad(
                        allocator,
                        expanded,
                        exp_mask,
                        exp_ids,
                        n_total,
                        E,
                        mrl_config,
                        self.loss_config.temperature,
                        self.loss_config.contrastive_focal_gamma,
                        self.loss_config.contrastive_focal_alpha,
                    );
                    break :mrl_blk self.mrlContrastiveResult(allocator, &mrl);
                } else try infonce_cpu.computeContrastiveLossOnCPU(
                    allocator,
                    expanded,
                    exp_mask,
                    exp_ids,
                    @as(f64, self.loss_config.temperature),
                    @as(f64, self.loss_config.lambda_embed),
                    1,
                    n_total,
                    E,
                    @as(f64, self.loss_config.contrastive_focal_gamma),
                    @as(f64, self.loss_config.contrastive_focal_alpha),
                );
                result_ready = true;
                return out;
            }
        }

        out.result = if (self.loss_config.use_mrl) mrl_blk: {
            const mrl_config = infonce_cpu.MatryoshkaConfig{
                .dims = self.loss_config.mrl_dims,
                .weights = self.loss_config.mrl_weights,
            };
            var mrl = try infonce_cpu.computeMatryoshkaLossAndGrad(
                allocator,
                chunk_embeddings,
                chunk_mask,
                doc_ids,
                B * C,
                E,
                mrl_config,
                self.loss_config.temperature,
                self.loss_config.contrastive_focal_gamma,
                self.loss_config.contrastive_focal_alpha,
            );
            break :mrl_blk self.mrlContrastiveResult(allocator, &mrl);
        } else try infonce_cpu.computeContrastiveLossOnCPU(
            allocator,
            chunk_embeddings,
            chunk_mask,
            doc_ids,
            @as(f64, self.loss_config.temperature),
            @as(f64, self.loss_config.lambda_embed),
            B,
            C,
            E,
            @as(f64, self.loss_config.contrastive_focal_gamma),
            @as(f64, self.loss_config.contrastive_focal_alpha),
        );
        result_ready = true;
        return out;
    }

    /// Run one training step.
    ///
    /// features:          [total_tokens * hidden_size] encoder hidden states
    /// boundary_labels:   [total_tokens * 2] one-hot
    /// attention_mask:    [total_tokens] 0 or 1
    /// chunk_embeddings:  [B*C*E] late-chunked embeddings
    /// chunk_mask:        [B*C] valid chunk mask
    /// doc_ids:           [B*C] document index per chunk
    pub fn trainStep(
        self: *FusedTrainer,
        allocator: std.mem.Allocator,
        features: []const f32,
        boundary_labels: []const f32,
        attention_mask: []const f32,
        chunk_embeddings: []const f32,
        chunk_mask: []const f32,
        doc_ids: []const u32,
        total_tokens: usize,
        B: usize,
        C: usize,
        E: usize,
    ) !TrainStepSummary {
        self.optimizer_state.step_count = self.step_count + 1;
        self.sanitizeBoundaryHeadParameters();
        var boundary_step = try self.runBoundaryHeadTraining(
            allocator,
            features,
            boundary_labels,
            attention_mask,
            total_tokens,
            false,
        );
        defer boundary_step.deinit(allocator);
        sanitizeBoundaryStepGradients(&boundary_step);

        const boundary_loss = boundary_step.boundary_loss;
        const boundary_grad_norm = boundary_step.gradNorm();

        // 4. Get current learning rate
        const lr = self.lr_schedule.lr(self.step_count);

        // 5. Accumulate gradients — scale by 1/accum_steps before adding
        const w1_grad = boundary_step.w1_grad;
        const b1_grad = boundary_step.b1_grad;
        const w2_grad = boundary_step.w2_grad;
        const b2_grad = boundary_step.b2_grad;

        const accum_steps = self.config.grad_accum_steps;
        const scale: f32 = 1.0 / @as(f32, @floatFromInt(@max(accum_steps, 1)));
        for (w1_grad, self.grad_accum_w1) |g, *a| a.* += g * scale;
        for (b1_grad, self.grad_accum_b1) |g, *a| a.* += g * scale;
        for (w2_grad, self.grad_accum_w2) |g, *a| a.* += g * scale;
        for (b2_grad, self.grad_accum_b2) |g, *a| a.* += g * scale;
        self.accum_count += 1;

        // Apply optimizer step only when accumulation window is full
        var applied_lr: f32 = 0.0;
        if (self.accum_count >= accum_steps) {
            applied_lr = lr;
            try self.applyAccumulatedBoundaryHeadStep(lr);
        }

        // 6. CPU InfoNCE contrastive loss — with optional XBM expansion (Feature 1)
        var contrastive_result: infonce_cpu.ContrastiveLossResult = undefined;
        var xbm_expanded_embeddings: ?[]f32 = null;
        var xbm_expanded_doc_ids: ?[]u32 = null;
        defer if (xbm_expanded_embeddings) |e| allocator.free(e);
        defer if (xbm_expanded_doc_ids) |d| allocator.free(d);

        const eff_embeddings: []const f32 = blk: {
            if (self.xbm) |*xbm| {
                const stored = xbm.getStored();
                if (stored.count > 0) {
                    const n_current = B * C;
                    const n_total = n_current + stored.count;
                    const expanded = try allocator.alloc(f32, n_total * E);
                    xbm_expanded_embeddings = expanded;
                    @memcpy(expanded[0 .. n_current * E], chunk_embeddings);
                    @memcpy(expanded[n_current * E ..], stored.embeddings[0 .. stored.count * E]);
                    // Stored doc_ids already carry globally unique offsets (Fix 1: each
                    // generation was stored with xbm_doc_id_base applied). Copy as-is.
                    const exp_ids = try allocator.alloc(u32, n_total);
                    xbm_expanded_doc_ids = exp_ids;
                    @memcpy(exp_ids[0..n_current], doc_ids);
                    @memcpy(exp_ids[n_current..], stored.doc_ids[0..stored.count]);
                    // Expanded chunk mask: copy current batch mask then fill rest with 1.0
                    const exp_mask = try allocator.alloc(f32, n_total);
                    defer allocator.free(exp_mask);
                    @memcpy(exp_mask[0..n_current], chunk_mask[0..n_current]);
                    @memset(exp_mask[n_current..], 1.0);
                    contrastive_result = if (self.loss_config.use_mrl) mrl_blk: {
                        const mrl_config = infonce_cpu.MatryoshkaConfig{
                            .dims = self.loss_config.mrl_dims,
                            .weights = self.loss_config.mrl_weights,
                        };
                        var mrl = try infonce_cpu.computeMatryoshkaLossAndGrad(
                            allocator,
                            expanded,
                            exp_mask,
                            exp_ids,
                            n_total,
                            E,
                            mrl_config,
                            self.loss_config.temperature,
                            self.loss_config.contrastive_focal_gamma,
                            self.loss_config.contrastive_focal_alpha,
                        );
                        break :mrl_blk self.mrlContrastiveResult(allocator, &mrl);
                    } else try infonce_cpu.computeContrastiveLossOnCPU(
                        allocator,
                        expanded,
                        exp_mask,
                        exp_ids,
                        @as(f64, self.loss_config.temperature),
                        @as(f64, self.loss_config.lambda_embed),
                        1,
                        n_total,
                        E,
                        @as(f64, self.loss_config.contrastive_focal_gamma),
                        @as(f64, self.loss_config.contrastive_focal_alpha),
                    );
                    break :blk expanded;
                }
            }
            contrastive_result = if (self.loss_config.use_mrl) mrl_blk: {
                const mrl_config = infonce_cpu.MatryoshkaConfig{
                    .dims = self.loss_config.mrl_dims,
                    .weights = self.loss_config.mrl_weights,
                };
                var mrl = try infonce_cpu.computeMatryoshkaLossAndGrad(
                    allocator,
                    chunk_embeddings,
                    chunk_mask,
                    doc_ids,
                    B * C,
                    E,
                    mrl_config,
                    self.loss_config.temperature,
                    self.loss_config.contrastive_focal_gamma,
                    self.loss_config.contrastive_focal_alpha,
                );
                break :mrl_blk self.mrlContrastiveResult(allocator, &mrl);
            } else try infonce_cpu.computeContrastiveLossOnCPU(
                allocator,
                chunk_embeddings,
                chunk_mask,
                doc_ids,
                @as(f64, self.loss_config.temperature),
                @as(f64, self.loss_config.lambda_embed),
                B,
                C,
                E,
                @as(f64, self.loss_config.contrastive_focal_gamma),
                @as(f64, self.loss_config.contrastive_focal_alpha),
            );
            break :blk chunk_embeddings;
        };
        _ = eff_embeddings;
        defer contrastive_result.deinit(allocator);

        // Add current batch to XBM after computing loss (Feature 1 / Fix 1).
        // Store doc_ids with the current global base so each generation's IDs are unique.
        if (self.xbm) |*xbm| {
            xbm.add(chunk_embeddings, doc_ids, chunk_mask, B * C, self.xbm_doc_id_base);
            self.xbm_doc_id_base +%= 100003;
        }

        // Note: contrastive gradients w.r.t. encoder weights are not applied here;
        // segmented backprop through the encoder is a future feature.

        // 7. Total loss
        const total_loss = self.loss_config.lambda_chunk * boundary_loss +
            @as(f32, @floatCast(contrastive_result.total_loss));

        // 8. Increment step count
        self.step_count += 1;
        if (self.config.step_log_every > 0 and self.step_count % self.config.step_log_every == 0) {
            std.log.info("fused_chunker step={d} boundary_loss={d:.4} total_loss={d:.4} lr={d}", .{ self.step_count, boundary_loss, total_loss, applied_lr });
        }

        return TrainStepSummary{
            .boundary_loss = boundary_loss,
            .contrastive_loss = contrastive_result.contrastive_loss,
            .total_loss = total_loss,
            .boundary_grad_norm = boundary_grad_norm,
            .boundary_tp = 0,
            .boundary_fp = 0,
            .boundary_fn = 0,
            .step = self.step_count,
            .learning_rate = applied_lr,
        };
    }

    /// Run one training step and also return dL/d(features) for encoder LoRA backprop.
    ///
    /// Identical to trainStep except "features" is added to the trainable_params
    /// list so that autodiff computes the gradient w.r.t. the input features tensor.
    /// The gradient is duped into an owned slice before the training.trainStep result
    /// is released; the caller is responsible for freeing it via
    /// TrainStepWithGradSummary.deinit().
    ///
    /// If training.trainStep does not produce a "features" gradient (e.g. the
    /// parameter is not present in the graph), features_grad is set to null and
    /// no error is returned.
    pub fn trainStepWithEncoderGrad(
        self: *FusedTrainer,
        allocator: std.mem.Allocator,
        features: []const f32,
        boundary_labels: []const f32,
        attention_mask: []const f32,
        chunk_embeddings: []const f32,
        chunk_mask: []const f32,
        doc_ids: []const u32,
        total_tokens: usize,
        B: usize,
        C: usize,
        E: usize,
        pooled_chunk_starts: []const i32,
        pooled_chunk_ends: []const i32,
        pooled_batch_size: usize,
        pooled_max_seq_len: usize,
        pooled_max_chunks: usize,
        want_upstream_grad_probe: bool,
    ) !TrainStepWithGradSummary {
        self.optimizer_state.step_count = self.step_count + 1;
        self.sanitizeBoundaryHeadParameters();
        var boundary_step = try self.runBoundaryHeadTraining(
            allocator,
            features,
            boundary_labels,
            attention_mask,
            total_tokens,
            true,
        );
        defer boundary_step.deinit(allocator);
        sanitizeBoundaryStepGradients(&boundary_step);

        const boundary_loss = boundary_step.boundary_loss;
        const boundary_grad_norm = boundary_step.gradNorm();

        // 4. Get current learning rate
        const lr = self.lr_schedule.lr(self.step_count);

        // 5. Take ownership of the feature gradient so the existing LoRA
        //    scatter/accumulation path can consume it.
        var features_grad_owned: ?[]f32 = boundary_step.takeFeaturesGrad();
        errdefer if (features_grad_owned) |g| allocator.free(g);
        const boundary_features_grad_stats = if (features_grad_owned) |grad|
            computeBoundaryProbeTensorStats(grad)
        else
            BoundaryProbeTensorStats{};
        var contrastive_features_grad_stats = BoundaryProbeTensorStats{};
        var combined_features_grad_stats = BoundaryProbeTensorStats{};

        // 6. Accumulate gradients — scale by 1/accum_steps before adding
        const w1_grad = boundary_step.w1_grad;
        const b1_grad = boundary_step.b1_grad;
        const w2_grad = boundary_step.w2_grad;
        const b2_grad = boundary_step.b2_grad;

        const accum_steps = self.config.grad_accum_steps;
        const scale: f32 = 1.0 / @as(f32, @floatFromInt(@max(accum_steps, 1)));
        for (w1_grad, self.grad_accum_w1) |g, *a| a.* += g * scale;
        for (b1_grad, self.grad_accum_b1) |g, *a| a.* += g * scale;
        for (w2_grad, self.grad_accum_w2) |g, *a| a.* += g * scale;
        for (b2_grad, self.grad_accum_b2) |g, *a| a.* += g * scale;
        self.accum_count += 1;

        var applied_lr: f32 = 0.0;
        if (self.accum_count >= accum_steps) {
            applied_lr = lr;
            try self.applyAccumulatedBoundaryHeadStep(lr);
        }

        // 7. CPU InfoNCE contrastive loss — with optional XBM expansion (Feature 1)
        var contrastive_result: infonce_cpu.ContrastiveLossResult = undefined;
        var xbm_expanded_embeddings: ?[]f32 = null;
        var xbm_expanded_doc_ids: ?[]u32 = null;
        defer if (xbm_expanded_embeddings) |e| allocator.free(e);
        defer if (xbm_expanded_doc_ids) |d| allocator.free(d);

        const eff_embeddings: []const f32 = blk: {
            if (self.xbm) |*xbm| {
                const stored = xbm.getStored();
                if (stored.count > 0) {
                    const n_current = B * C;
                    const n_total = n_current + stored.count;
                    const expanded = try allocator.alloc(f32, n_total * E);
                    xbm_expanded_embeddings = expanded;
                    @memcpy(expanded[0 .. n_current * E], chunk_embeddings);
                    @memcpy(expanded[n_current * E ..], stored.embeddings[0 .. stored.count * E]);
                    // Stored doc_ids already carry globally unique offsets (Fix 1).
                    const exp_ids = try allocator.alloc(u32, n_total);
                    xbm_expanded_doc_ids = exp_ids;
                    @memcpy(exp_ids[0..n_current], doc_ids);
                    @memcpy(exp_ids[n_current..], stored.doc_ids[0..stored.count]);
                    const exp_mask = try allocator.alloc(f32, n_total);
                    defer allocator.free(exp_mask);
                    @memcpy(exp_mask[0..n_current], chunk_mask[0..n_current]);
                    @memset(exp_mask[n_current..], 1.0);
                    contrastive_result = if (self.loss_config.use_mrl) mrl_blk: {
                        const mrl_config = infonce_cpu.MatryoshkaConfig{
                            .dims = self.loss_config.mrl_dims,
                            .weights = self.loss_config.mrl_weights,
                        };
                        var mrl = try infonce_cpu.computeMatryoshkaLossAndGrad(
                            allocator,
                            expanded,
                            exp_mask,
                            exp_ids,
                            n_total,
                            E,
                            mrl_config,
                            self.loss_config.temperature,
                            self.loss_config.contrastive_focal_gamma,
                            self.loss_config.contrastive_focal_alpha,
                        );
                        break :mrl_blk self.mrlContrastiveResult(allocator, &mrl);
                    } else try infonce_cpu.computeContrastiveLossOnCPU(
                        allocator,
                        expanded,
                        exp_mask,
                        exp_ids,
                        @as(f64, self.loss_config.temperature),
                        @as(f64, self.loss_config.lambda_embed),
                        1,
                        n_total,
                        E,
                        @as(f64, self.loss_config.contrastive_focal_gamma),
                        @as(f64, self.loss_config.contrastive_focal_alpha),
                    );
                    break :blk expanded;
                }
            }
            contrastive_result = if (self.loss_config.use_mrl) mrl_blk: {
                const mrl_config = infonce_cpu.MatryoshkaConfig{
                    .dims = self.loss_config.mrl_dims,
                    .weights = self.loss_config.mrl_weights,
                };
                var mrl = try infonce_cpu.computeMatryoshkaLossAndGrad(
                    allocator,
                    chunk_embeddings,
                    chunk_mask,
                    doc_ids,
                    B * C,
                    E,
                    mrl_config,
                    self.loss_config.temperature,
                    self.loss_config.contrastive_focal_gamma,
                    self.loss_config.contrastive_focal_alpha,
                );
                break :mrl_blk self.mrlContrastiveResult(allocator, &mrl);
            } else try infonce_cpu.computeContrastiveLossOnCPU(
                allocator,
                chunk_embeddings,
                chunk_mask,
                doc_ids,
                @as(f64, self.loss_config.temperature),
                @as(f64, self.loss_config.lambda_embed),
                B,
                C,
                E,
                @as(f64, self.loss_config.contrastive_focal_gamma),
                @as(f64, self.loss_config.contrastive_focal_alpha),
            );
            break :blk chunk_embeddings;
        };
        _ = eff_embeddings;
        defer contrastive_result.deinit(allocator);

        // Add current batch to XBM after computing loss (Feature 1 / Fix 1).
        if (self.xbm) |*xbm| {
            xbm.add(chunk_embeddings, doc_ids, chunk_mask, B * C, self.xbm_doc_id_base);
            self.xbm_doc_id_base +%= 100003;
        }

        // Scatter the InfoNCE gradient on pooled current-batch chunks back into
        // token features so LoRA receives the retrieval signal. This is the one
        // remaining non-compiled backward segment in the fused chunker path;
        // Go parity should replace it with embedding-head + late-pooling
        // backward graph lowering once that graph surface is ready.
        const pooled_chunk_count = pooled_batch_size * pooled_max_chunks;
        if (pooled_chunk_count > 0 and E > 0 and contrastive_result.grad.len >= pooled_chunk_count * E) {
            if (want_upstream_grad_probe) {
                var contrastive_features_grad: ?[]f32 = try allocator.alloc(f32, total_tokens * self.config.hidden_size);
                errdefer if (contrastive_features_grad) |grad| allocator.free(grad);
                @memset(contrastive_features_grad.?, 0);
                try fused_chunker_data.addMeanPoolChunkEmbeddingGradToFeatures(
                    allocator,
                    contrastive_features_grad.?,
                    features,
                    contrastive_result.grad[0 .. pooled_chunk_count * E],
                    pooled_chunk_starts,
                    pooled_chunk_ends,
                    chunk_mask[0..pooled_chunk_count],
                    attention_mask,
                    pooled_batch_size,
                    pooled_max_seq_len,
                    pooled_max_chunks,
                    E,
                );
                contrastive_features_grad_stats = computeBoundaryProbeTensorStats(contrastive_features_grad.?);
                if (features_grad_owned) |grad| {
                    for (grad, contrastive_features_grad.?) |*dst, src| dst.* += src;
                    allocator.free(contrastive_features_grad.?);
                    contrastive_features_grad = null;
                } else {
                    features_grad_owned = contrastive_features_grad.?;
                    contrastive_features_grad = null;
                }
            } else {
                if (features_grad_owned == null) {
                    features_grad_owned = try allocator.alloc(f32, total_tokens * self.config.hidden_size);
                    @memset(features_grad_owned.?, 0);
                }
                try fused_chunker_data.addMeanPoolChunkEmbeddingGradToFeatures(
                    allocator,
                    features_grad_owned.?,
                    features,
                    contrastive_result.grad[0 .. pooled_chunk_count * E],
                    pooled_chunk_starts,
                    pooled_chunk_ends,
                    chunk_mask[0..pooled_chunk_count],
                    attention_mask,
                    pooled_batch_size,
                    pooled_max_seq_len,
                    pooled_max_chunks,
                    E,
                );
            }
        }
        if (features_grad_owned) |grad| {
            sanitizeGradientBufferInPlace(grad);
            combined_features_grad_stats = computeBoundaryProbeTensorStats(grad);
        }

        // Accumulate features_grad across microbatches for LoRA (Fix 4).
        // We null features_grad_owned after consuming it so the errdefer above
        // doesn't double-free on any subsequent error.
        if (accum_steps > 1) {
            if (features_grad_owned) |fgo| {
                const lora_scale = 1.0 / @as(f32, @floatFromInt(accum_steps));
                if (self.grad_accum_lora_features_grad_accum == null or
                    self.grad_accum_lora_features_grad_accum.?.len != fgo.len)
                {
                    if (self.grad_accum_lora_features_grad_accum) |old| self.allocator.free(old);
                    self.grad_accum_lora_features_grad_accum = try self.allocator.alloc(f32, fgo.len);
                    @memset(self.grad_accum_lora_features_grad_accum.?, 0);
                }
                for (self.grad_accum_lora_features_grad_accum.?, fgo) |*a, g| {
                    a.* += g * lora_scale;
                }
                // Free the per-step copy and null the variable so errdefer won't fire.
                self.allocator.free(fgo);
                features_grad_owned = null;
            }
        }

        // Resolve the features_grad to return to the caller (Fix 4).
        // When grad_accum_steps > 1, return the accumulated buffer only on the
        // optimizer step boundary; otherwise return nil so LoRA skips this step.
        const final_features_grad: ?[]f32 = if (accum_steps > 1) blk: {
            if (self.accum_count == 0) {
                // We just reset accum_count — this was the optimizer step boundary.
                // Hand ownership of the accumulated buffer to the caller.
                const buf = self.grad_accum_lora_features_grad_accum;
                self.grad_accum_lora_features_grad_accum = null;
                break :blk buf;
            }
            // Not yet at optimizer step — LoRA should skip this microbatch.
            break :blk null;
        } else features_grad_owned; // accum_steps == 1: return as-is

        // 8. Total loss
        const total_loss = self.loss_config.lambda_chunk * boundary_loss +
            @as(f32, @floatCast(contrastive_result.total_loss));

        // 9. Increment step count
        self.step_count += 1;
        if (self.config.step_log_every > 0 and self.step_count % self.config.step_log_every == 0) {
            std.log.info("fused_chunker step={d} boundary_loss={d:.4} total_loss={d:.4} lr={d}", .{ self.step_count, boundary_loss, total_loss, applied_lr });
        }

        const summary = TrainStepSummary{
            .boundary_loss = boundary_loss,
            .contrastive_loss = contrastive_result.contrastive_loss,
            .total_loss = total_loss,
            .boundary_grad_norm = boundary_grad_norm,
            .boundary_tp = 0,
            .boundary_fp = 0,
            .boundary_fn = 0,
            .step = self.step_count,
            .learning_rate = applied_lr,
        };

        return TrainStepWithGradSummary{
            .summary = summary,
            .features_grad = final_features_grad,
            .boundary_features_grad_stats = boundary_features_grad_stats,
            .contrastive_features_grad_stats = contrastive_features_grad_stats,
            .combined_features_grad_stats = combined_features_grad_stats,
        };
    }

    /// Evaluate micro-F1 over a list of feature/label batches.
    ///
    /// features_list:      slice of [total_tokens * hidden_size] batches
    /// labels_list:        slice of [total_tokens * 2] one-hot batches
    /// mask_list:          slice of [total_tokens] attention mask batches
    /// total_tokens_list:  total tokens per batch
    pub fn evaluate(
        self: *FusedTrainer,
        allocator: std.mem.Allocator,
        features_list: []const []const f32,
        labels_list: []const []const f32,
        mask_list: []const []const f32,
        total_tokens_list: []const usize,
    ) !EvalSummary {
        var acc = BoundaryEvalAccumulator{};

        for (features_list, labels_list, mask_list, total_tokens_list) |features, labels, mask, total| {
            try self.evaluateBatchInto(allocator, &acc, features, labels, mask, total);
        }

        return acc.finish();
    }

    pub fn evaluateBatchInto(
        self: *FusedTrainer,
        allocator: std.mem.Allocator,
        acc: *BoundaryEvalAccumulator,
        features: []const f32,
        labels: []const f32,
        mask: []const f32,
        total: usize,
    ) !void {
        const logits = if (self.legacy_dense_boundary_head) |*legacy_head|
            try evaluateLegacyDenseBoundaryLogits(allocator, legacy_head, features, total)
        else
            try evaluateBoundaryLogitsSimple(
                allocator,
                &self.boundary_head,
                features,
                total,
            );
        defer allocator.free(logits);

        try acc.addLogits(allocator, logits, labels, mask);
    }

    /// Run the boundary-head training substep and probability diagnostics
    /// without mutating trainer weights, optimizer state, or step counters.
    pub fn debugBoundaryStep(
        self: *FusedTrainer,
        allocator: std.mem.Allocator,
        features: []const f32,
        boundary_labels: []const f32,
        attention_mask: []const f32,
        total_tokens: usize,
        want_features_grad: bool,
    ) !BoundaryStepDebugSummary {
        self.sanitizeBoundaryHeadParameters();
        var boundary_step = try self.runBoundaryHeadTraining(
            allocator,
            features,
            boundary_labels,
            attention_mask,
            total_tokens,
            want_features_grad,
        );
        defer boundary_step.deinit(allocator);
        sanitizeBoundaryStepGradients(&boundary_step);

        var forward_probe = BoundaryForwardProbe{};
        const logits = if (self.legacy_dense_boundary_head) |*legacy_head| blk: {
            const legacy_logits = try evaluateLegacyDenseBoundaryLogits(allocator, legacy_head, features, total_tokens);
            forward_probe.boundary_head_input = computeBoundaryProbeTensorStats(features);
            forward_probe.logits = computeBoundaryProbeTensorStats(legacy_logits);
            break :blk legacy_logits;
        } else blk: {
            const forward = try evaluateBoundaryForwardProbeSimple(allocator, &self.boundary_head, features, total_tokens);
            forward_probe = forward.probe;
            break :blk forward.logits;
        };
        defer allocator.free(logits);

        var predicted_positives: u64 = 0;
        var tp: u64 = 0;
        var fp: u64 = 0;
        var fn_: u64 = 0;
        var prob_sum_pos: f64 = 0;
        var prob_sum_neg: f64 = 0;
        var count_pos: u64 = 0;
        var count_neg: u64 = 0;
        for (0..total_tokens) |i| {
            if (attention_mask[i] <= 0.5) continue;
            const prob = fused_chunker_loss.positiveBoundaryProbability(logits[i * 2], logits[i * 2 + 1]);
            const predicted = prob > 0.5;
            if (predicted) predicted_positives += 1;
            const is_positive = boundary_labels[i * 2 + 1] > 0.5;
            if (predicted and is_positive) {
                tp += 1;
            } else if (predicted and !is_positive) {
                fp += 1;
            } else if (!predicted and is_positive) {
                fn_ += 1;
            }
            if (is_positive) {
                prob_sum_pos += @floatCast(prob);
                count_pos += 1;
            } else {
                prob_sum_neg += @floatCast(prob);
                count_neg += 1;
            }
        }

        return .{
            .boundary_loss = boundary_step.boundary_loss,
            .grad_norm_w1 = l2NormF32(boundary_step.w1_grad),
            .grad_norm_b1 = l2NormF32(boundary_step.b1_grad),
            .grad_norm_w2 = l2NormF32(boundary_step.w2_grad),
            .grad_norm_b2 = l2NormF32(boundary_step.b2_grad),
            .grad_max_abs_w1 = maxAbsF32(boundary_step.w1_grad),
            .grad_max_abs_b1 = maxAbsF32(boundary_step.b1_grad),
            .grad_max_abs_w2 = maxAbsF32(boundary_step.w2_grad),
            .grad_max_abs_b2 = maxAbsF32(boundary_step.b2_grad),
            .features_grad_norm = if (boundary_step.features_grad) |g| l2NormF32(g) else 0,
            .features_grad_max_abs = if (boundary_step.features_grad) |g| maxAbsF32(g) else 0,
            .has_features_grad = boundary_step.features_grad != null,
            .eval_predicted_positives = predicted_positives,
            .eval_tp = tp,
            .eval_fp = fp,
            .eval_fn = fn_,
            .eval_f1 = (fused_chunker_loss.BoundaryMetrics{ .tp = tp, .fp = fp, .fn_ = fn_ }).f1(),
            .eval_mean_prob_gold_positive = if (count_pos > 0) @as(f32, @floatCast(prob_sum_pos / @as(f64, @floatFromInt(count_pos)))) else 0,
            .eval_mean_prob_gold_negative = if (count_neg > 0) @as(f32, @floatCast(prob_sum_neg / @as(f64, @floatFromInt(count_neg)))) else 0,
            .boundary_forward_probe = forward_probe,
            .boundary_checkpoint_probe = computeBoundaryCheckpointProbe(&self.boundary_head),
        };
    }

    /// Flush any partial gradient accumulation window at epoch end.
    ///
    /// Call this after the last `trainStep`/`trainStepWithEncoderGrad` of each epoch.
    /// If the number of training steps is not divisible by `grad_accum_steps`, the
    /// remaining accumulated gradients would otherwise be silently discarded.
    /// Has no effect when `accum_count == 0` (window is already empty).
    pub fn flushEpochEnd(self: *FusedTrainer, allocator: std.mem.Allocator) !void {
        if (self.accum_count == 0) return;
        const lr = self.config.lrSchedule().lr(self.step_count);
        _ = allocator;
        try self.applyAccumulatedBoundaryHeadStep(lr);
    }

    /// Save boundary head weights to a SafeTensors checkpoint.
    ///
    /// The file is written using the SafeTensors format (see
    /// src/finetune/safetensors_checkpoint.zig) so that it can be read by
    /// standard tooling (Python safetensors library, Go readers, etc.).
    pub fn saveCheckpoint(self: *FusedTrainer, allocator: std.mem.Allocator, path: []const u8) !void {
        const st = @import("safetensors_checkpoint.zig");
        const tensors = [_]st.NamedTensor{
            .{
                .name = "w1",
                .data = self.boundary_head.w1,
                .shape = &.{ self.boundary_head.mlp_dim, self.boundary_head.hidden_dim },
            },
            .{
                .name = "b1",
                .data = self.boundary_head.b1,
                .shape = &.{self.boundary_head.mlp_dim},
            },
            .{
                .name = "w2",
                .data = self.boundary_head.w2,
                .shape = &.{ 2, self.boundary_head.mlp_dim },
            },
            .{
                .name = "b2",
                .data = self.boundary_head.b2,
                .shape = &.{2},
            },
        };
        try st.save(allocator, path, &tensors);
    }

    /// Load boundary head weights from a checkpoint file.
    ///
    /// Tries SafeTensors format first. If the file does not parse as SafeTensors
    /// (e.g. it was created by an older build using the legacy binary format),
    /// falls back to the original binary reader so that existing checkpoints
    /// remain usable.
    pub fn loadCheckpoint(self: *FusedTrainer, allocator: std.mem.Allocator, path: []const u8) !void {
        // Try SafeTensors format.
        if (self.loadCheckpointSafetensors(allocator, path)) |_| {
            return;
        } else |_| {}

        // Fallback: legacy binary format written by the old saveCheckpoint.
        // This keeps existing .bin checkpoints loadable after the format migration.
        return self.loadCheckpointBinary(allocator, path);
    }

    fn loadCheckpointSafetensors(self: *FusedTrainer, allocator: std.mem.Allocator, path: []const u8) !void {
        const safetensors = @import("../models/safetensors.zig");

        // Read the whole file into memory and hand ownership to MMapReader.
        // MMapReader.fromBytes stores the slice and frees it in deinit(), so we
        // must NOT also free it ourselves.
        const file_bytes = try compat.cwd().readFileAlloc(compat.io(), path, allocator, .unlimited);

        // fromBytes takes ownership of file_bytes (freed via deinit).
        var reader = safetensors.MMapReader.fromBytes(allocator, file_bytes) catch |err| {
            allocator.free(file_bytes);
            return err;
        };
        defer reader.deinit();

        if (reader.header.tensors.contains("w1")) {
            if (!try copySafetensorFirst(&reader, &.{"w1"}, self.boundary_head.w1)) return error.TensorNotFound;
            if (!try copySafetensorFirst(&reader, &.{"b1"}, self.boundary_head.b1)) return error.TensorNotFound;
            if (!try copySafetensorFirst(&reader, &.{"w2"}, self.boundary_head.w2)) return error.TensorNotFound;
            if (!try copySafetensorFirst(&reader, &.{"b2"}, self.boundary_head.b2)) return error.TensorNotFound;
            return;
        }

        const go_w1_names = [_][]const u8{
            "boundary_head/mlp_dense1/weight",
            "fused_chunker_embedder/boundary_head/mlp_dense1/weight",
            "var:/fused_chunker_embedder/boundary_head/mlp_dense1/weight",
        };
        const go_b1_names = [_][]const u8{
            "boundary_head/mlp_dense1/bias",
            "fused_chunker_embedder/boundary_head/mlp_dense1/bias",
            "var:/fused_chunker_embedder/boundary_head/mlp_dense1/bias",
        };
        const go_w2_names = [_][]const u8{
            "boundary_head/mlp_dense2/weight",
            "fused_chunker_embedder/boundary_head/mlp_dense2/weight",
            "var:/fused_chunker_embedder/boundary_head/mlp_dense2/weight",
        };
        const go_b2_names = [_][]const u8{
            "boundary_head/mlp_dense2/bias",
            "fused_chunker_embedder/boundary_head/mlp_dense2/bias",
            "var:/fused_chunker_embedder/boundary_head/mlp_dense2/bias",
        };

        if (hasAnySafetensorName(&reader, &go_w1_names)) {
            if (!try copySafetensorFirstTransposed(
                &reader,
                &go_w1_names,
                self.boundary_head.w1,
                self.boundary_head.hidden_dim,
                self.boundary_head.mlp_dim,
            )) return error.TensorNotFound;
            if (!try copySafetensorFirst(&reader, &go_b1_names, self.boundary_head.b1)) return error.TensorNotFound;
            if (!try copySafetensorFirstTransposed(
                &reader,
                &go_w2_names,
                self.boundary_head.w2,
                self.boundary_head.mlp_dim,
                2,
            )) return error.TensorNotFound;
            if (!try copySafetensorFirst(&reader, &go_b2_names, self.boundary_head.b2)) return error.TensorNotFound;
            return;
        }

        const legacy_w_names = [_][]const u8{
            "boundary_head/dense/weight",
            "fused_chunker_embedder/boundary_head/dense/weight",
            "var:/fused_chunker_embedder/boundary_head/dense/weight",
        };
        const legacy_b_names = [_][]const u8{
            "boundary_head/dense/bias",
            "fused_chunker_embedder/boundary_head/dense/bias",
            "var:/fused_chunker_embedder/boundary_head/dense/bias",
        };

        if (hasAnySafetensorName(&reader, &legacy_w_names)) {
            if (self.legacy_dense_boundary_head) |*old| {
                old.deinit();
                self.legacy_dense_boundary_head = null;
            }
            self.legacy_dense_boundary_head = try LegacyDenseBoundaryHead.init(allocator, self.boundary_head.hidden_dim);
            if (self.legacy_dense_boundary_head) |*legacy| {
                if (!try copySafetensorFirstTransposed(
                    &reader,
                    &legacy_w_names,
                    legacy.weight,
                    self.boundary_head.hidden_dim,
                    2,
                )) return error.TensorNotFound;
                if (!try copySafetensorFirst(&reader, &legacy_b_names, legacy.bias)) return error.TensorNotFound;
            }
            return;
        }

        return error.TensorNotFound;
    }

    fn hasAnySafetensorName(reader: anytype, names: []const []const u8) bool {
        for (names) |name| {
            if (reader.header.tensors.contains(name)) return true;
        }
        return false;
    }

    fn copySafetensorFirst(reader: anytype, names: []const []const u8, dest: []f32) !bool {
        for (names) |name| {
            if (!reader.header.tensors.contains(name)) continue;
            var tensor = try reader.readTensor(name);
            defer tensor.deinit();
            if (tensor.dtype != .f32) return error.InvalidCheckpoint;
            const src = tensor.asFloat32();
            if (src.len != dest.len) return error.CheckpointSizeMismatch;
            @memcpy(dest, src);
            return true;
        }
        return false;
    }

    fn copySafetensorFirstTransposed(
        reader: anytype,
        names: []const []const u8,
        dest: []f32,
        rows: usize,
        cols: usize,
    ) !bool {
        for (names) |name| {
            if (!reader.header.tensors.contains(name)) continue;
            var tensor = try reader.readTensor(name);
            defer tensor.deinit();
            if (tensor.dtype != .f32) return error.InvalidCheckpoint;
            if (tensor.shape.len != 2 or tensor.shape[0] != @as(i64, @intCast(rows)) or tensor.shape[1] != @as(i64, @intCast(cols))) {
                return error.CheckpointSizeMismatch;
            }
            const src = tensor.asFloat32();
            if (src.len != rows * cols or dest.len != rows * cols) return error.CheckpointSizeMismatch;
            for (0..rows) |r| {
                for (0..cols) |c| {
                    dest[c * rows + r] = src[r * cols + c];
                }
            }
            return true;
        }
        return false;
    }

    fn loadCheckpointBinary(self: *FusedTrainer, allocator: std.mem.Allocator, path: []const u8) !void {
        const data = try compat.cwd().readFileAlloc(compat.io(), path, allocator, .unlimited);
        defer allocator.free(data);
        var offset: usize = 0;

        const tensor_targets = [_]struct {
            name: []const u8,
            dest: *[]f32,
        }{
            .{ .name = "w1", .dest = &self.boundary_head.w1 },
            .{ .name = "b1", .dest = &self.boundary_head.b1 },
            .{ .name = "w2", .dest = &self.boundary_head.w2 },
            .{ .name = "b2", .dest = &self.boundary_head.b2 },
        };

        for (tensor_targets) |tgt| {
            const name_len = try readU32(data, &offset);
            if (name_len > 256) return error.InvalidCheckpoint;
            if (offset + name_len > data.len) return error.IncompleteRead;
            var name_buf: [256]u8 = undefined;
            @memcpy(name_buf[0..name_len], data[offset .. offset + name_len]);
            offset += name_len;
            const name = name_buf[0..name_len];
            if (!std.mem.eql(u8, name, tgt.name)) return error.CheckpointNameMismatch;

            const size = try readU32(data, &offset);
            if (size != tgt.dest.*.len) return error.CheckpointSizeMismatch;
            for (tgt.dest.*) |*val| {
                val.* = @bitCast(try readU32(data, &offset));
            }
        }
    }

    fn readU32(data: []const u8, offset: *usize) !u32 {
        if (offset.* + 4 > data.len) return error.IncompleteRead;
        const value = std.mem.readInt(u32, data[offset.*..][0..4], .little);
        offset.* += 4;
        return value;
    }

    /// Save Adam optimizer state to a SafeTensors file.
    ///
    /// Saves m/v moment buffers for w1/b1/w2/b2 under names "adam_m_w1",
    /// "adam_v_w1", etc., plus "adam_step" as a 1-element f32 tensor.
    ///
    /// For Schedule-Free AdamW, also saves the z and v buffers as
    /// "sf_z_w1", "sf_v_w1", etc.
    pub fn saveOptimizerState(self: *FusedTrainer, allocator: std.mem.Allocator, path: []const u8) !void {
        const st = @import("safetensors_checkpoint.zig");

        var tensor_list = std.ArrayListUnmanaged(st.NamedTensor).empty;
        defer tensor_list.deinit(allocator);

        // Heap-allocated name strings that must outlive tensor_list.
        // We collect them here and free them all after save().
        var name_storage = std.ArrayListUnmanaged([]u8).empty;
        defer {
            for (name_storage.items) |n| allocator.free(n);
            name_storage.deinit(allocator);
        }

        var shape_storage = std.ArrayListUnmanaged([]usize).empty;
        defer {
            for (shape_storage.items) |shape| allocator.free(shape);
            shape_storage.deinit(allocator);
        }

        // adam_step as a 1-element f32 scalar.
        const step_val = [1]f32{@as(f32, @floatFromInt(self.optimizer_state.step_count))};
        const step_shape = try allocator.dupe(usize, &.{1});
        try shape_storage.append(allocator, step_shape);
        try tensor_list.append(allocator, .{
            .name = "adam_step",
            .data = &step_val,
            .shape = step_shape,
        });

        // AdamW moment buffers for each head parameter.
        const param_names = [_][]const u8{ "w1", "b1", "w2", "b2" };
        for (param_names) |pname| {
            if (self.optimizer_state.param_states.get(pname)) |ps| {
                const m_name = try std.fmt.allocPrint(allocator, "adam_m_{s}", .{pname});
                try name_storage.append(allocator, m_name);
                const m_shape = try allocator.dupe(usize, &.{ps.m.len});
                try shape_storage.append(allocator, m_shape);
                try tensor_list.append(allocator, .{
                    .name = m_name,
                    .data = ps.m,
                    .shape = m_shape,
                });
                if (ps.v.len > 0) {
                    const v_name = try std.fmt.allocPrint(allocator, "adam_v_{s}", .{pname});
                    try name_storage.append(allocator, v_name);
                    const v_shape = try allocator.dupe(usize, &.{ps.v.len});
                    try shape_storage.append(allocator, v_shape);
                    try tensor_list.append(allocator, .{
                        .name = v_name,
                        .data = ps.v,
                        .shape = v_shape,
                    });
                }
            }
        }

        // Schedule-Free states (if active).
        const sf_pairs = [_]struct {
            state: ?ScheduleFreeAdamWState,
            pname: []const u8,
        }{
            .{ .state = self.sf_state_w1, .pname = "w1" },
            .{ .state = self.sf_state_b1, .pname = "b1" },
            .{ .state = self.sf_state_w2, .pname = "w2" },
            .{ .state = self.sf_state_b2, .pname = "b2" },
        };
        for (sf_pairs) |pair| {
            if (pair.state) |sf| {
                const z_name = try std.fmt.allocPrint(allocator, "sf_z_{s}", .{pair.pname});
                try name_storage.append(allocator, z_name);
                const v_name = try std.fmt.allocPrint(allocator, "sf_v_{s}", .{pair.pname});
                try name_storage.append(allocator, v_name);
                const z_shape = try allocator.dupe(usize, &.{sf.z.len});
                try shape_storage.append(allocator, z_shape);
                const v_shape = try allocator.dupe(usize, &.{sf.v.len});
                try shape_storage.append(allocator, v_shape);
                try tensor_list.append(allocator, .{
                    .name = z_name,
                    .data = sf.z,
                    .shape = z_shape,
                });
                try tensor_list.append(allocator, .{
                    .name = v_name,
                    .data = sf.v,
                    .shape = v_shape,
                });
            }
        }

        try st.save(allocator, path, tensor_list.items);
    }

    /// Load Adam optimizer state from a SafeTensors file written by saveOptimizerState.
    pub fn loadOptimizerState(self: *FusedTrainer, allocator: std.mem.Allocator, path: []const u8) !void {
        const safetensors = @import("../models/safetensors.zig");

        const file_bytes = try compat.cwd().readFileAlloc(compat.io(), path, allocator, .unlimited);
        errdefer allocator.free(file_bytes);

        // fromBytes takes ownership of file_bytes (freed via deinit).
        var reader = try safetensors.MMapReader.fromBytes(allocator, file_bytes);
        defer reader.deinit();

        // Restore step count from the scalar tensor.
        if (reader.header.tensors.get("adam_step")) |_| {
            var step_tensor = try reader.readTensor("adam_step");
            defer step_tensor.deinit();
            const step_f32 = step_tensor.asFloat32();
            if (step_f32.len > 0) {
                const restored_step: u32 = @intFromFloat(step_f32[0]);
                self.optimizer_state.step_count = restored_step;
                // Keep the trainer-level step counter in sync.
                self.step_count = restored_step;
            }
        }

        // Restore AdamW moment buffers for each head parameter.
        const param_names = [_][]const u8{ "w1", "b1", "w2", "b2" };
        const param_sizes = [_]usize{
            self.boundary_head.w1.len,
            self.boundary_head.b1.len,
            self.boundary_head.w2.len,
            self.boundary_head.b2.len,
        };
        for (param_names, param_sizes) |pname, psize| {
            var m_name_buf: [32]u8 = undefined;
            var v_name_buf: [32]u8 = undefined;
            const m_name = try std.fmt.bufPrint(&m_name_buf, "adam_m_{s}", .{pname});
            const v_name = try std.fmt.bufPrint(&v_name_buf, "adam_v_{s}", .{pname});

            const has_m = reader.header.tensors.get(m_name) != null;
            const has_v = reader.header.tensors.get(v_name) != null;
            if (!has_m) continue;

            const ps = try self.optimizer_state.getOrCreate(pname, psize, has_v);

            var m_tensor = try reader.readTensor(m_name);
            defer m_tensor.deinit();
            const m_src = m_tensor.asFloat32();
            if (m_src.len == ps.m.len) @memcpy(ps.m, m_src);

            if (has_v) {
                var v_tensor = try reader.readTensor(v_name);
                defer v_tensor.deinit();
                const v_src = v_tensor.asFloat32();
                if (v_src.len == ps.v.len) @memcpy(ps.v, v_src);
            }
        }

        // Restore Schedule-Free states if present.
        const sf_entries = [_]struct {
            state: *?ScheduleFreeAdamWState,
            weights: []f32,
            pname: []const u8,
        }{
            .{ .state = &self.sf_state_w1, .weights = self.boundary_head.w1, .pname = "w1" },
            .{ .state = &self.sf_state_b1, .weights = self.boundary_head.b1, .pname = "b1" },
            .{ .state = &self.sf_state_w2, .weights = self.boundary_head.w2, .pname = "w2" },
            .{ .state = &self.sf_state_b2, .weights = self.boundary_head.b2, .pname = "b2" },
        };
        for (sf_entries) |entry| {
            var z_name_buf: [32]u8 = undefined;
            var v_name_buf: [32]u8 = undefined;
            const z_name = try std.fmt.bufPrint(&z_name_buf, "sf_z_{s}", .{entry.pname});
            const v_name = try std.fmt.bufPrint(&v_name_buf, "sf_v_{s}", .{entry.pname});

            const has_z = reader.header.tensors.get(z_name) != null;
            const has_v = reader.header.tensors.get(v_name) != null;
            if (!has_z or !has_v) continue;

            // Initialise the SF state if it doesn't exist yet.
            if (entry.state.* == null) {
                entry.state.* = try ScheduleFreeAdamWState.init(self.allocator, entry.weights);
            }

            const sf = &entry.state.*.?;
            var z_tensor = try reader.readTensor(z_name);
            defer z_tensor.deinit();
            const z_src = z_tensor.asFloat32();
            if (z_src.len == sf.z.len) @memcpy(sf.z, z_src);

            var v_tensor = try reader.readTensor(v_name);
            defer v_tensor.deinit();
            const v_src = v_tensor.asFloat32();
            if (v_src.len == sf.v.len) @memcpy(sf.v, v_src);
        }
    }
};

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

fn dropoutSeed(base_seed: u64, step_count: u32, salt: u64) u64 {
    return base_seed ^ ((@as(u64, step_count) + 1) *% 0x9E37_79B9_7F4A_7C15) ^ salt;
}

fn allocInvertedDropoutMask(
    allocator: std.mem.Allocator,
    len: usize,
    dropout_prob: f32,
    seed: u64,
) ![]f32 {
    const mask = try allocator.alloc(f32, len);
    if (dropout_prob <= 0.0) {
        @memset(mask, 1.0);
        return mask;
    }
    if (dropout_prob >= 1.0) {
        allocator.free(mask);
        return error.InvalidBoundaryDropout;
    }

    const keep_prob = 1.0 - dropout_prob;
    const scale = 1.0 / keep_prob;
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (mask) |*v| {
        v.* = if (rng.float(f32) < keep_prob) scale else 0.0;
    }
    return mask;
}

fn l2NormF32(values: []const f32) f32 {
    var total_sq: f64 = 0;
    for (values) |value| {
        if (!std.math.isFinite(value)) continue;
        const v: f64 = @floatCast(value);
        total_sq += v * v;
    }
    return @floatCast(@sqrt(total_sq));
}

fn maxAbsF32(values: []const f32) f32 {
    var out: f32 = 0;
    for (values) |value| {
        if (!std.math.isFinite(value)) continue;
        out = @max(out, @abs(value));
    }
    return out;
}

fn hashProbeF32(hash: *u64, value: f32) void {
    var bits: u32 = @bitCast(value);
    for (0..4) |_| {
        hash.* ^= bits & 0xff;
        hash.* *%= tensor_probe_hash_prime;
        bits >>= 8;
    }
}

fn addBoundaryProbeTopAbs(stats: *BoundaryProbeTensorStats, index: usize, value: f32) void {
    const ax = @abs(value);
    if (ax > stats.max_abs) {
        stats.max_abs = ax;
        stats.max_abs_index = @intCast(index);
        stats.max_abs_value = value;
    }

    var insert_at: usize = undefined;
    if (stats.top_abs_len < stats.top_abs_values.len) {
        insert_at = stats.top_abs_len;
        stats.top_abs_len += 1;
    } else if (ax > @abs(stats.top_abs_values[stats.top_abs_values.len - 1])) {
        insert_at = stats.top_abs_values.len - 1;
    } else {
        return;
    }

    while (insert_at > 0 and ax > @abs(stats.top_abs_values[insert_at - 1])) : (insert_at -= 1) {
        stats.top_abs_values[insert_at] = stats.top_abs_values[insert_at - 1];
        stats.top_abs_indices[insert_at] = stats.top_abs_indices[insert_at - 1];
    }
    stats.top_abs_values[insert_at] = value;
    stats.top_abs_indices[insert_at] = @intCast(index);
}

pub fn computeBoundaryProbeTensorStats(values: []const f32) BoundaryProbeTensorStats {
    var out = BoundaryProbeTensorStats{
        .elems = @intCast(values.len),
    };
    var sum: f64 = 0;
    var sum_sq: f64 = 0;
    var finite_count: u64 = 0;
    for (values, 0..) |value, i| {
        if (i < out.sample.len) {
            out.sample[i] = value;
            out.sample_len = @intCast(i + 1);
        }
        hashProbeF32(&out.hash, value);
        if (!std.math.isFinite(value)) continue;
        const v: f64 = @floatCast(value);
        sum += v;
        sum_sq += v * v;
        addBoundaryProbeTopAbs(&out, i, value);
        finite_count += 1;
    }
    if (finite_count > 0) {
        const denom = @as(f64, @floatFromInt(finite_count));
        out.mean = @floatCast(sum / denom);
        out.rms = @floatCast(@sqrt(sum_sq / denom));
    }
    return out;
}

pub fn computeZeroBoundaryProbeTensorStats(elems: usize) BoundaryProbeTensorStats {
    var out = BoundaryProbeTensorStats{
        .elems = @intCast(elems),
        .sample_len = @intCast(@min(elems, 16)),
    };
    for (0..elems) |_| hashProbeF32(&out.hash, 0);
    return out;
}

fn computeTransposedBoundaryProbeTensorStats(
    values: []const f32,
    rows: usize,
    cols: usize,
) BoundaryProbeTensorStats {
    var out = computeBoundaryProbeTensorStats(values);
    const sample_len = @min(values.len, out.sample.len);
    out.sample_len = @intCast(sample_len);
    for (0..sample_len) |i| {
        const src_col = i / rows;
        const src_row = i % rows;
        out.sample[i] = values[src_row * cols + src_col];
    }
    return out;
}

fn computeBoundaryCheckpointProbe(head: *const BoundaryHead) BoundaryCheckpointProbe {
    return .{
        .w1 = computeTransposedBoundaryProbeTensorStats(head.w1, head.mlp_dim, head.hidden_dim),
        .b1 = computeBoundaryProbeTensorStats(head.b1),
        .w2 = computeTransposedBoundaryProbeTensorStats(head.w2, 2, head.mlp_dim),
        .b2 = computeBoundaryProbeTensorStats(head.b2),
    };
}

const BoundaryCeGradResult = struct {
    loss: f32,
    logit_grad: []f32,
};

fn computeBoundaryWeightedCeAndLogitGrad(
    allocator: std.mem.Allocator,
    logits: []const f32,
    targets: []const f32,
    mask: []const f32,
    total: usize,
    pos_weight: f32,
) !BoundaryCeGradResult {
    if (logits.len != total * 2) return error.UnexpectedOutputShape;
    if (targets.len != total * 2) return error.UnexpectedOutputShape;
    if (mask.len != total) return error.UnexpectedOutputShape;

    const grad = try allocator.alloc(f32, total * 2);
    errdefer allocator.free(grad);

    var mask_sum: f64 = 0;
    for (mask) |m| mask_sum += @as(f64, m);
    const denom = @as(f32, @floatCast(mask_sum)) + 1e-12;

    var numerator: f64 = 0;
    for (0..total) |i| {
        const z0 = logits[i * 2];
        const z1 = logits[i * 2 + 1];
        const max_z = @max(z0, z1);
        const e0 = @exp(z0 - max_z);
        const e1 = @exp(z1 - max_z);
        const inv_sum = 1.0 / (e0 + e1);
        const p0 = e0 * inv_sum;
        const p1 = e1 * inv_sum;
        const log_sum = @log(e0 + e1);
        const logp0 = z0 - max_z - log_sum;
        const logp1 = z1 - max_z - log_sum;

        const t0 = targets[i * 2];
        const t1 = targets[i * 2 + 1];
        const wt0 = t0;
        const wt1 = t1 * pos_weight;
        const wt_sum = wt0 + wt1;
        const m = mask[i];

        numerator += @as(f64, m * -(wt0 * logp0 + wt1 * logp1));
        const row_scale = m / denom;
        grad[i * 2] = row_scale * (p0 * wt_sum - wt0);
        grad[i * 2 + 1] = row_scale * (p1 * wt_sum - wt1);
    }

    return .{
        .loss = @floatCast(numerator / @as(f64, denom)),
        .logit_grad = grad,
    };
}

fn allocGeluExactDerivative(
    allocator: std.mem.Allocator,
    dense: []const f32,
) ![]f32 {
    const out = try allocator.alloc(f32, dense.len);
    errdefer allocator.free(out);

    const inv_sqrt2: f32 = 0.7071067811865475;
    const inv_sqrt_2pi: f32 = 0.3989422804014327;
    for (dense, out) |x, *dst| {
        const cdf = 0.5 * (1.0 + erfApproxF32(x * inv_sqrt2));
        const pdf = inv_sqrt_2pi * @exp(-0.5 * x * x);
        dst.* = cdf + x * pdf;
    }
    return out;
}

/// Copy f32 data into a ComputeBackend tensor and insert it into the runtime map.
fn putRuntimeInput(
    allocator: std.mem.Allocator,
    cb: *const ComputeBackend,
    rt: *std.AutoHashMapUnmanaged(NodeId, CT),
    id: NodeId,
    data: []const f32,
    dims: []const i32,
) !void {
    const ct = try cb.fromFloat32Shape(data, dims);
    errdefer cb.free(ct);
    try rt.put(allocator, id, ct);
}

/// Per-tensor gradient L2 norm clipping (in-place).
fn clipGradient(grad: []f32, max_norm: f32) void {
    var norm_sq: f32 = 0;
    for (grad) |g| norm_sq += g * g;
    const norm = @sqrt(norm_sq);
    if (norm > max_norm) {
        const scale = max_norm / norm;
        for (grad) |*g| g.* *= scale;
    }
}

/// Write one tensor to a writer in the checkpoint binary format:
///   [name_len: u32 LE][name bytes][elem_count: u32 LE][f32 data as u32 LE...]
fn writeTensor(w: anytype, name: []const u8, data: []const f32) !void {
    try w.writeInt(u32, @intCast(name.len), .little);
    try w.writeAll(name);
    try w.writeInt(u32, @intCast(data.len), .little);
    for (data) |val| {
        try w.writeInt(u32, @bitCast(val), .little);
    }
}

/// Pure CPU forward pass through the 2-layer MLP boundary head.
///
/// Returns a freshly allocated [total * 2] f32 logit array (caller owns).
///
/// Architecture:
///   dense1 = features @ w1^T + b1    [total, mlp_dim]
///   hidden = gelu(dense1)
///   logits = hidden @ w2^T + b2      [total, 2]
fn evaluateBoundaryLogitsSimple(
    allocator: std.mem.Allocator,
    head: *const BoundaryHead,
    features: []const f32,
    total: usize,
) ![]f32 {
    const hidden_dim = head.hidden_dim;
    const mlp_dim = head.mlp_dim;

    // dense1 = features @ w1^T + b1   [total, mlp_dim]
    const dense1 = try allocator.alloc(f32, total * mlp_dim);
    defer allocator.free(dense1);

    for (0..total) |i| {
        for (0..mlp_dim) |j| {
            var acc: f32 = head.b1[j];
            for (0..hidden_dim) |k| {
                acc += features[i * hidden_dim + k] * head.w1[j * hidden_dim + k];
            }
            dense1[i * mlp_dim + j] = acc;
        }
    }

    // hidden = gelu(dense1)            [total, mlp_dim]
    const hidden = try allocator.alloc(f32, total * mlp_dim);
    defer allocator.free(hidden);

    for (dense1, hidden) |x, *h| {
        h.* = geluF32(x);
    }

    // logits = hidden @ w2^T + b2     [total, 2]
    const logits = try allocator.alloc(f32, total * 2);
    for (0..total) |i| {
        for (0..2) |j| {
            var acc: f32 = head.b2[j];
            for (0..mlp_dim) |k| {
                acc += hidden[i * mlp_dim + k] * head.w2[j * mlp_dim + k];
            }
            logits[i * 2 + j] = acc;
        }
    }

    return logits;
}

const BoundaryForwardProbeResult = struct {
    logits: []f32,
    probe: BoundaryForwardProbe,

    fn deinit(self: *BoundaryForwardProbeResult, allocator: std.mem.Allocator) void {
        allocator.free(self.logits);
        self.* = undefined;
    }
};

fn evaluateBoundaryForwardProbeSimple(
    allocator: std.mem.Allocator,
    head: *const BoundaryHead,
    features: []const f32,
    total: usize,
) !BoundaryForwardProbeResult {
    const hidden_dim = head.hidden_dim;
    const mlp_dim = head.mlp_dim;

    const dense1 = try allocator.alloc(f32, total * mlp_dim);
    defer allocator.free(dense1);

    for (0..total) |i| {
        for (0..mlp_dim) |j| {
            var acc: f32 = head.b1[j];
            for (0..hidden_dim) |k| {
                acc += features[i * hidden_dim + k] * head.w1[j * hidden_dim + k];
            }
            dense1[i * mlp_dim + j] = acc;
        }
    }

    const hidden = try allocator.alloc(f32, total * mlp_dim);
    defer allocator.free(hidden);

    for (dense1, hidden) |x, *h| {
        h.* = geluF32(x);
    }

    const logits = try allocator.alloc(f32, total * 2);
    errdefer allocator.free(logits);
    for (0..total) |i| {
        for (0..2) |j| {
            var acc: f32 = head.b2[j];
            for (0..mlp_dim) |k| {
                acc += hidden[i * mlp_dim + k] * head.w2[j * mlp_dim + k];
            }
            logits[i * 2 + j] = acc;
        }
    }

    return .{
        .logits = logits,
        .probe = .{
            .boundary_head_input = computeBoundaryProbeTensorStats(features),
            .dense1_pre_activation = computeBoundaryProbeTensorStats(dense1),
            .dense1_post_activation = computeBoundaryProbeTensorStats(hidden),
            .logits = computeBoundaryProbeTensorStats(logits),
        },
    };
}

fn evaluateLegacyDenseBoundaryLogits(
    allocator: std.mem.Allocator,
    head: *const LegacyDenseBoundaryHead,
    features: []const f32,
    total: usize,
) ![]f32 {
    const hidden_dim = head.hidden_dim;
    if (features.len < total * hidden_dim) return error.UnexpectedOutputShape;

    const logits = try allocator.alloc(f32, total * 2);
    for (0..total) |i| {
        for (0..2) |j| {
            var acc: f32 = head.bias[j];
            for (0..hidden_dim) |k| {
                acc += features[i * hidden_dim + k] * head.weight[j * hidden_dim + k];
            }
            logits[i * 2 + j] = acc;
        }
    }
    return logits;
}

/// Exact-form GELU activation used by the Go fused boundary head:
/// 0.5 * x * (1 + erf(x / sqrt(2))).
inline fn geluF32(x: f32) f32 {
    return 0.5 * x * (1.0 + erfApproxF32(x * 0.7071067811865475));
}

// Abramowitz & Stegun 7.1.26, matching the native primitive erf op.
inline fn erfApproxF32(x: f32) f32 {
    const a1: f32 = 0.254829592;
    const a2: f32 = -0.284496736;
    const a3: f32 = 1.421413741;
    const a4: f32 = -1.453152027;
    const a5: f32 = 1.061405429;
    const p: f32 = 0.3275911;
    const sign: f32 = if (x < 0) -1.0 else 1.0;
    const ax = @abs(x);
    const t = 1.0 / (1.0 + p * ax);
    const poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    return sign * (1.0 - poly * @exp(-ax * ax));
}

fn expectApproxEqSlices(expected: []const f32, actual: []const f32, tolerance: f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        try std.testing.expectApproxEqAbs(e, a, tolerance);
    }
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

test "BoundaryHead init and deinit" {
    const allocator = std.testing.allocator;

    var head = try BoundaryHead.init(allocator, 16, 8);
    defer head.deinit();

    try std.testing.expectEqual(@as(usize, 16), head.hidden_dim);
    try std.testing.expectEqual(@as(usize, 8), head.mlp_dim);
    try std.testing.expectEqual(@as(usize, 8 * 16), head.w1.len);
    try std.testing.expectEqual(@as(usize, 8), head.b1.len);
    try std.testing.expectEqual(@as(usize, 2 * 8), head.w2.len);
    try std.testing.expectEqual(@as(usize, 2), head.b2.len);

    // Biases should be zero
    for (head.b1) |v| try std.testing.expectEqual(@as(f32, 0.0), v);
    for (head.b2) |v| try std.testing.expectEqual(@as(f32, 0.0), v);

    // Weights should be non-zero (deterministic init)
    var any_nonzero = false;
    for (head.w1) |v| {
        if (v != 0.0) {
            any_nonzero = true;
            break;
        }
    }
    try std.testing.expect(any_nonzero);
}

test "FusedTrainingConfig lrSchedule warmup" {
    const config = FusedTrainingConfig{
        .learning_rate = 1e-4,
        .warmup_steps = 50,
        .total_steps = 1000,
    };

    const schedule = config.lrSchedule();

    const lr0 = schedule.lr(0);
    const lr25 = schedule.lr(25);
    const lr50 = schedule.lr(50);
    const lr1000 = schedule.lr(1000);

    // During warmup: lr increases linearly to learning_rate, using the same
    // first-update nonzero convention as the Go fused trainer.
    try std.testing.expect(lr0 < lr25);
    try std.testing.expect(lr25 < lr50);

    // At step 0: lr should be learning_rate / warmup_steps.
    try std.testing.expectApproxEqAbs(config.learning_rate / @as(f32, @floatFromInt(config.warmup_steps)), lr0, 1e-8);

    // At end of warmup (step 50): lr should equal initial_lr
    try std.testing.expectApproxEqAbs(config.learning_rate, lr50, 1e-6);

    // The fused Go schedule decays to zero by the final configured step.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), lr1000, 1e-8);
}

test "boundary gradient sanitizer drops nonfinite values before clipping" {
    const allocator = std.testing.allocator;

    var weight_store = native_compute.WeightStore{
        .allocator = allocator,
        .resident_weights = .{},
        .lazy_weights = .{},
    };
    defer {
        native_compute.deinitPrefetchQueue(&weight_store);
        weight_store.resident_weights.deinit(allocator);
        weight_store.lazy_weights.deinit(allocator);
    }
    var compute = native_compute.NativeCompute.init(allocator, &weight_store, null);
    var cb = compute.computeBackend();

    const config = FusedTrainingConfig{
        .hidden_size = 2,
        .embedding_dim = 2,
        .boundary_mlp_dim = 1,
        .max_grad_norm = 1.0,
    };
    var trainer = try FusedTrainer.init(allocator, config, &cb);
    defer trainer.deinit();

    @memset(trainer.grad_accum_w1, 0);
    @memset(trainer.grad_accum_b1, 0);
    @memset(trainer.grad_accum_w2, 0);
    @memset(trainer.grad_accum_b2, 0);

    trainer.grad_accum_w1[0] = std.math.nan(f32);
    trainer.grad_accum_w1[1] = 4.0;
    trainer.grad_accum_b1[0] = std.math.inf(f32);
    trainer.grad_accum_w2[0] = 3.0;
    trainer.grad_accum_w2[1] = -std.math.inf(f32);
    trainer.grad_accum_b2[0] = 0.0;
    trainer.grad_accum_b2[1] = 0.0;

    trainer.sanitizeAndClipBoundaryGradientAccum();

    try std.testing.expectEqual(@as(f32, 0.0), trainer.grad_accum_w1[0]);
    try std.testing.expectEqual(@as(f32, 0.0), trainer.grad_accum_b1[0]);
    try std.testing.expectEqual(@as(f32, 0.0), trainer.grad_accum_w2[1]);

    var total_sq: f64 = 0.0;
    FusedTrainer.addGradientNormSq(&total_sq, trainer.grad_accum_w1);
    FusedTrainer.addGradientNormSq(&total_sq, trainer.grad_accum_b1);
    FusedTrainer.addGradientNormSq(&total_sq, trainer.grad_accum_w2);
    FusedTrainer.addGradientNormSq(&total_sq, trainer.grad_accum_b2);
    try std.testing.expect(@sqrt(total_sq) <= 1.000001);
}

test "BoundaryEvalAccumulator reports boundary rate and probability diagnostics" {
    const allocator = std.testing.allocator;

    const logits = [_]f32{
        0.0, 1.0,
        1.0, 0.0,
        0.0, 0.0,
    };
    const labels = [_]f32{
        0.0, 1.0,
        1.0, 0.0,
        0.0, 1.0,
    };
    const mask = [_]f32{ 1.0, 1.0, 1.0 };

    var acc = BoundaryEvalAccumulator{};
    try acc.addLogits(allocator, &logits, &labels, &mask);
    const summary = acc.finish();

    try std.testing.expectEqual(@as(u64, 3), summary.valid_tokens);
    try std.testing.expectEqual(@as(u64, 2), summary.gold_positives);
    try std.testing.expectEqual(@as(u64, 1), summary.predicted_positives);
    try std.testing.expectEqual(@as(u64, 2), summary.best_predicted_positives);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), summary.gold_positive_rate, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), summary.predicted_positive_rate, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), summary.best_predicted_positive_rate, 1e-6);

    const p0 = fused_chunker_loss.positiveBoundaryProbability(0.0, 1.0);
    const p1 = fused_chunker_loss.positiveBoundaryProbability(1.0, 0.0);
    const p2 = fused_chunker_loss.positiveBoundaryProbability(0.0, 0.0);
    try std.testing.expectApproxEqAbs((p0 + p2) * 0.5, summary.mean_positive_probability_gold_positive, 1e-6);
    try std.testing.expectApproxEqAbs(p1, summary.mean_positive_probability_gold_negative, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), summary.mean_boundary_margin_gold_positive, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), summary.mean_boundary_margin_gold_negative, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), summary.mean_logit0_gold_positive, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), summary.mean_logit1_gold_positive, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), summary.mean_logit0_gold_negative, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), summary.mean_logit1_gold_negative, 1e-6);
    try std.testing.expectEqual(@as(f32, 0.01), summary.threshold_points[0].threshold);
    try std.testing.expectEqual(@as(f32, 0.5), summary.threshold_points[summary.threshold_points.len - 1].threshold);
    try std.testing.expectEqual(@as(u64, 1), summary.probability_histogram_gold_positive[5]);
    try std.testing.expectEqual(@as(u64, 1), summary.probability_histogram_gold_positive[7]);
    try std.testing.expectEqual(@as(u64, 1), summary.probability_histogram_gold_negative[2]);
}

test "trainStep treats max_grad_norm zero as clipping disabled" {
    const allocator = std.testing.allocator;

    var weight_store = native_compute.WeightStore{
        .allocator = allocator,
        .resident_weights = .{},
        .lazy_weights = .{},
    };
    defer {
        native_compute.deinitPrefetchQueue(&weight_store);
        weight_store.resident_weights.deinit(allocator);
        weight_store.lazy_weights.deinit(allocator);
    }
    var compute = native_compute.NativeCompute.init(allocator, &weight_store, null);
    var cb = compute.computeBackend();

    const config = FusedTrainingConfig{
        .hidden_size = 2,
        .embedding_dim = 2,
        .boundary_mlp_dim = 2,
        .max_chunks = 1,
        .learning_rate = 0.01,
        .warmup_steps = 1,
        .weight_decay = 0.0,
        .max_grad_norm = 0.0,
        .lambda_embed = 0.0,
        .pos_weight = 5.0,
        .total_steps = 2,
    };
    try std.testing.expectEqual(@as(f32, 0.0), config.max_grad_norm);

    var trainer = try FusedTrainer.init(allocator, config, &cb);
    defer trainer.deinit();

    @memset(trainer.boundary_head.w1, 0);
    @memset(trainer.boundary_head.b1, 0);
    @memset(trainer.boundary_head.w2, 0);
    @memset(trainer.boundary_head.b2, 0);

    const features = [_]f32{
        0.0, 0.0,
        0.0, 0.0,
    };
    const labels = [_]f32{
        0.0, 1.0,
        1.0, 0.0,
    };
    const attention_mask = [_]f32{ 1.0, 1.0 };
    const chunk_embeddings = [_]f32{ 0.0, 0.0 };
    const chunk_mask = [_]f32{0.0};
    const doc_ids = [_]u32{0};

    _ = try trainer.trainStep(
        allocator,
        &features,
        &labels,
        &attention_mask,
        &chunk_embeddings,
        &chunk_mask,
        &doc_ids,
        2,
        1,
        1,
        2,
    );

    try std.testing.expect(@abs(trainer.boundary_head.b2[0]) > 0.0);
    try std.testing.expect(@abs(trainer.boundary_head.b2[1]) > 0.0);
}

test "trainStep applies boundary pos_weight and attention mask" {
    const allocator = std.testing.allocator;

    var weight_store = native_compute.WeightStore{
        .allocator = allocator,
        .resident_weights = .{},
        .lazy_weights = .{},
    };
    defer {
        native_compute.deinitPrefetchQueue(&weight_store);
        weight_store.resident_weights.deinit(allocator);
        weight_store.lazy_weights.deinit(allocator);
    }
    var compute = native_compute.NativeCompute.init(allocator, &weight_store, null);
    var cb = compute.computeBackend();

    const config = FusedTrainingConfig{
        .hidden_size = 2,
        .embedding_dim = 2,
        .boundary_mlp_dim = 2,
        .max_chunks = 1,
        .learning_rate = 0.0,
        .lambda_embed = 0.0,
        .pos_weight = 5.0,
        .total_steps = 1,
    };
    var trainer = try FusedTrainer.init(allocator, config, &cb);
    defer trainer.deinit();

    @memset(trainer.boundary_head.w1, 0);
    @memset(trainer.boundary_head.b1, 0);
    @memset(trainer.boundary_head.w2, 0);
    @memset(trainer.boundary_head.b2, 0);

    const features = [_]f32{
        0.0, 0.0,
        0.0, 0.0,
    };
    const labels = [_]f32{
        0.0, 1.0,
        1.0, 0.0,
    };
    const attention_mask = [_]f32{ 1.0, 0.0 };
    const chunk_embeddings = [_]f32{ 0.0, 0.0 };
    const chunk_mask = [_]f32{0.0};
    const doc_ids = [_]u32{0};

    const summary = try trainer.trainStep(
        allocator,
        &features,
        &labels,
        &attention_mask,
        &chunk_embeddings,
        &chunk_mask,
        &doc_ids,
        2,
        1,
        1,
        2,
    );

    try std.testing.expectApproxEqAbs(@log(@as(f32, 2.0)) * config.pos_weight, summary.boundary_loss, 1e-5);
}

test "trainStep applies Go-compatible boundary focal loss when enabled" {
    const allocator = std.testing.allocator;

    var weight_store = native_compute.WeightStore{
        .allocator = allocator,
        .resident_weights = .{},
        .lazy_weights = .{},
    };
    defer {
        native_compute.deinitPrefetchQueue(&weight_store);
        weight_store.resident_weights.deinit(allocator);
        weight_store.lazy_weights.deinit(allocator);
    }
    var compute = native_compute.NativeCompute.init(allocator, &weight_store, null);
    var cb = compute.computeBackend();

    const config = FusedTrainingConfig{
        .hidden_size = 2,
        .embedding_dim = 2,
        .boundary_mlp_dim = 2,
        .max_chunks = 1,
        .learning_rate = 0.0,
        .lambda_embed = 0.0,
        .use_boundary_focal = true,
        .focal_gamma = 2.0,
        .focal_alpha = 0.75,
        .total_steps = 1,
    };
    var trainer = try FusedTrainer.init(allocator, config, &cb);
    defer trainer.deinit();

    @memset(trainer.boundary_head.w1, 0);
    @memset(trainer.boundary_head.b1, 0);
    @memset(trainer.boundary_head.w2, 0);
    @memset(trainer.boundary_head.b2, 0);

    const features = [_]f32{
        0.0, 0.0,
        0.0, 0.0,
    };
    const labels = [_]f32{
        0.0, 1.0,
        1.0, 0.0,
    };
    const attention_mask = [_]f32{ 1.0, 0.0 };
    const chunk_embeddings = [_]f32{ 0.0, 0.0 };
    const chunk_mask = [_]f32{0.0};
    const doc_ids = [_]u32{0};

    const summary = try trainer.trainStep(
        allocator,
        &features,
        &labels,
        &attention_mask,
        &chunk_embeddings,
        &chunk_mask,
        &doc_ids,
        2,
        1,
        1,
        2,
    );

    const expected = config.focal_alpha * 0.25 * @log(@as(f32, 2.0));
    try std.testing.expectApproxEqAbs(expected, summary.boundary_loss, 1e-5);
}

test "trainStepWithEncoderGrad scatters contrastive gradients into features" {
    const allocator = std.testing.allocator;

    var weight_store = native_compute.WeightStore{
        .allocator = allocator,
        .resident_weights = .{},
        .lazy_weights = .{},
    };
    defer {
        native_compute.deinitPrefetchQueue(&weight_store);
        weight_store.resident_weights.deinit(allocator);
        weight_store.lazy_weights.deinit(allocator);
    }
    var compute = native_compute.NativeCompute.init(allocator, &weight_store, null);
    var cb = compute.computeBackend();

    const config = FusedTrainingConfig{
        .hidden_size = 2,
        .embedding_dim = 2,
        .boundary_mlp_dim = 2,
        .max_chunks = 3,
        .lambda_embed = 1.0,
        .temperature = 0.5,
        .total_steps = 4,
    };
    var trainer = try FusedTrainer.init(allocator, config, &cb);
    defer trainer.deinit();

    @memset(trainer.boundary_head.w1, 0);
    @memset(trainer.boundary_head.b1, 0);
    @memset(trainer.boundary_head.w2, 0);
    @memset(trainer.boundary_head.b2, 0);

    const features = [_]f32{
        1.0,  0.0,
        0.0,  1.0,
        -1.0, 0.0,
    };
    const labels = [_]f32{
        1.0, 0.0,
        1.0, 0.0,
        1.0, 0.0,
    };
    const attention_mask = [_]f32{ 1.0, 1.0, 1.0 };
    const chunk_embeddings = features;
    const chunk_mask = [_]f32{ 1.0, 1.0, 1.0 };
    const doc_ids = [_]u32{ 0, 0, 1 };
    const chunk_starts = [_]i32{ 0, 1, 2 };
    const chunk_ends = [_]i32{ 1, 2, 3 };

    var result = try trainer.trainStepWithEncoderGrad(
        allocator,
        &features,
        &labels,
        &attention_mask,
        &chunk_embeddings,
        &chunk_mask,
        &doc_ids,
        3,
        1,
        3,
        2,
        &chunk_starts,
        &chunk_ends,
        1,
        3,
        3,
        false,
    );
    defer result.deinit(allocator);

    const grad = result.features_grad orelse return error.ExpectedFeatureGradient;
    var nonzero_count: usize = 0;
    for (grad) |g| {
        if (@abs(g) > 1e-6) nonzero_count += 1;
    }
    try std.testing.expect(nonzero_count > 0);
    try std.testing.expectEqual(@as(usize, 1), trainer.boundary_graph_cache.count());
    const cached = trainer.boundary_graph_cache.getPtr(3) orelse return error.MissingBoundaryGraphCacheEntry;
    try std.testing.expect(cached.encoder_grad_session != null);
}

test "MRL contrastive result scales gradients by lambda_embed" {
    const allocator = std.testing.allocator;

    var weight_store = native_compute.WeightStore{
        .allocator = allocator,
        .resident_weights = .{},
        .lazy_weights = .{},
    };
    defer {
        native_compute.deinitPrefetchQueue(&weight_store);
        weight_store.resident_weights.deinit(allocator);
        weight_store.lazy_weights.deinit(allocator);
    }
    var compute = native_compute.NativeCompute.init(allocator, &weight_store, null);
    var cb = compute.computeBackend();

    const config = FusedTrainingConfig{
        .hidden_size = 4,
        .embedding_dim = 4,
        .boundary_mlp_dim = 2,
        .max_chunks = 2,
        .lambda_embed = 0.25,
        .use_mrl = true,
    };
    var trainer = try FusedTrainer.init(allocator, config, &cb);
    defer trainer.deinit();

    var mrl = infonce_cpu.MatryoshkaResult{
        .total_loss = 2.0,
        .per_scale_loss = try allocator.dupe(f64, &[_]f64{ 1.0, 3.0 }),
        .grad = try allocator.dupe(f32, &[_]f32{ 4.0, -8.0, 0.0, 2.0 }),
    };
    var result = trainer.mrlContrastiveResult(allocator, &mrl);
    defer result.deinit(allocator);

    try std.testing.expectApproxEqAbs(@as(f64, 2.0), result.contrastive_loss, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), result.total_loss, 1e-9);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, -2.0, 0.0, 0.5 }, result.grad);
}

test "boundary CE ops step matches graph gradients" {
    const allocator = std.testing.allocator;

    var weight_store = native_compute.WeightStore{
        .allocator = allocator,
        .resident_weights = .{},
        .lazy_weights = .{},
    };
    defer {
        native_compute.deinitPrefetchQueue(&weight_store);
        weight_store.resident_weights.deinit(allocator);
        weight_store.lazy_weights.deinit(allocator);
    }
    var compute = native_compute.NativeCompute.init(allocator, &weight_store, null);
    var cb = compute.computeBackend();

    const config = FusedTrainingConfig{
        .hidden_size = 2,
        .embedding_dim = 2,
        .boundary_mlp_dim = 3,
        .max_chunks = 1,
        .learning_rate = 0.0,
        .lambda_embed = 0.0,
        .pos_weight = 2.5,
        .boundary_dropout = 0.0,
        .total_steps = 1,
    };
    var trainer = try FusedTrainer.init(allocator, config, &cb);
    defer trainer.deinit();

    @memcpy(trainer.boundary_head.w1, &[_]f32{
        0.10,  -0.20,
        0.30,  0.05,
        -0.15, 0.25,
    });
    @memcpy(trainer.boundary_head.b1, &[_]f32{ 0.01, -0.02, 0.03 });
    @memcpy(trainer.boundary_head.w2, &[_]f32{
        0.20,  -0.10, 0.15,
        -0.05, 0.12,  -0.18,
    });
    @memcpy(trainer.boundary_head.b2, &[_]f32{ 0.04, -0.03 });

    const features = [_]f32{
        0.50,  -0.25,
        -0.75, 0.40,
    };
    const labels = [_]f32{
        0.0, 1.0,
        1.0, 0.0,
    };
    const attention_mask = [_]f32{ 1.0, 0.5 };

    var graph_step = try trainer.runBoundaryHeadGraphStep(
        allocator,
        &features,
        &labels,
        &attention_mask,
        2,
        true,
    );
    defer graph_step.deinit(allocator);

    var ce_step = try trainer.runBoundaryHeadCeOpsStep(
        allocator,
        &features,
        &labels,
        &attention_mask,
        2,
        true,
    );
    defer ce_step.deinit(allocator);

    try std.testing.expectApproxEqAbs(graph_step.boundary_loss, ce_step.boundary_loss, 1e-5);
    try expectApproxEqSlices(graph_step.w1_grad, ce_step.w1_grad, 1e-4);
    try expectApproxEqSlices(graph_step.b1_grad, ce_step.b1_grad, 1e-4);
    try expectApproxEqSlices(graph_step.w2_grad, ce_step.w2_grad, 1e-4);
    try expectApproxEqSlices(graph_step.b2_grad, ce_step.b2_grad, 1e-4);
    try expectApproxEqSlices(graph_step.features_grad.?, ce_step.features_grad.?, 1e-4);
}

test "LISA layer selector disables to all active layers" {
    const allocator = std.testing.allocator;
    const active = try selectActiveLoRALayers(allocator, 6, 0, 2, 11, 42);
    defer allocator.free(active);

    try std.testing.expectEqual(@as(usize, 6), active.len);
    for (active) |is_active| try std.testing.expect(is_active);
}

test "LISA layer selector keeps top K and samples remaining deterministically" {
    const allocator = std.testing.allocator;
    const active_a = try selectActiveLoRALayers(allocator, 8, 2, 3, 17, 42);
    defer allocator.free(active_a);
    const active_b = try selectActiveLoRALayers(allocator, 8, 2, 3, 17, 42);
    defer allocator.free(active_b);

    try std.testing.expectEqual(@as(usize, 8), active_a.len);
    try std.testing.expectEqualSlices(bool, active_a, active_b);

    // Highest layer indices are always active.
    try std.testing.expect(active_a[5]);
    try std.testing.expect(active_a[6]);
    try std.testing.expect(active_a[7]);

    var active_count: usize = 0;
    for (active_a) |is_active| {
        if (is_active) active_count += 1;
    }
    try std.testing.expectEqual(@as(usize, 5), active_count);
}

test "LISA layer selector clamps top and sample counts" {
    const allocator = std.testing.allocator;
    const active = try selectActiveLoRALayers(allocator, 4, 99, 99, 1, 2);
    defer allocator.free(active);

    try std.testing.expectEqual(@as(usize, 4), active.len);
    for (active) |is_active| try std.testing.expect(is_active);
}
