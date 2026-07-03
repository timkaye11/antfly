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
const mpsgraph_executor = @import("../graph/mpsgraph_executor.zig");
const platform = @import("antfly_platform");
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

pub const BoundaryFeatureMode = enum {
    token,
    prev_diff,
    prev_current_diff,
    prev_current_diff_concat,
    prev_current_next_diff_concat,
    window_context_diff,
};

pub fn boundaryFeatureModeName(mode: BoundaryFeatureMode) []const u8 {
    return switch (mode) {
        .token => "token",
        .prev_diff => "prev-diff",
        .prev_current_diff => "prev-current-diff",
        .prev_current_diff_concat => "prev-current-diff-concat",
        .prev_current_next_diff_concat => "prev-current-next-diff-concat",
        .window_context_diff => "window-context-diff",
    };
}

pub fn boundaryFeatureMultiplier(mode: BoundaryFeatureMode) usize {
    return switch (mode) {
        .token,
        .prev_diff,
        .prev_current_diff,
        => 1,
        .prev_current_diff_concat => 3,
        .prev_current_next_diff_concat,
        .window_context_diff,
        => 5,
    };
}

pub fn boundaryFeatureDim(mode: BoundaryFeatureMode, hidden_size: usize) usize {
    return hidden_size * boundaryFeatureMultiplier(mode);
}

pub const FusedTrainingConfig = struct {
    // Model
    max_seq_len: u32 = 384,
    embedding_dim: u32 = 768,
    hidden_size: u32 = 768,
    boundary_mlp_dim: u32 = 256,
    boundary_feature_mode: BoundaryFeatureMode = .token,
    max_chunks: u32 = 32,

    // Training
    batch_size: u32 = 16,
    num_epochs: u32 = 10,
    learning_rate: f32 = 1e-4,
    boundary_head_lr_multiplier: f32 = 1.0,
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

    pub fn boundaryHeadLearningRate(self: FusedTrainingConfig, base_lr: f32) f32 {
        return base_lr * self.boundary_head_lr_multiplier;
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

fn hasPreviousValidToken(attention_mask: []const f32, flat_index: usize, max_seq_len: usize) bool {
    if (max_seq_len == 0 or flat_index == 0) return false;
    if (flat_index % max_seq_len == 0) return false;
    return attention_mask[flat_index] > 0.5 and attention_mask[flat_index - 1] > 0.5;
}

fn hasNextValidToken(attention_mask: []const f32, flat_index: usize, total_tokens: usize, max_seq_len: usize) bool {
    if (max_seq_len == 0 or flat_index + 1 >= total_tokens) return false;
    if ((flat_index + 1) % max_seq_len == 0) return false;
    return attention_mask[flat_index] > 0.5 and attention_mask[flat_index + 1] > 0.5;
}

const boundary_window_context_radius: usize = 16;

fn windowContextMean(
    features: []const f32,
    attention_mask: []const f32,
    flat_index: usize,
    hidden_index: usize,
    hidden_size: usize,
    total_tokens: usize,
    max_seq_len: usize,
    comptime direction: enum { left, right },
) struct { mean: f32, count: usize } {
    const seq_start = (flat_index / max_seq_len) * max_seq_len;
    const seq_end = @min(seq_start + max_seq_len, total_tokens);
    var sum: f32 = 0.0;
    var count: usize = 0;
    switch (direction) {
        .left => {
            var j = flat_index;
            while (j > seq_start and count < boundary_window_context_radius) {
                j -= 1;
                if (attention_mask[j] <= 0.5) continue;
                sum += features[j * hidden_size + hidden_index];
                count += 1;
            }
        },
        .right => {
            var j = flat_index + 1;
            while (j < seq_end and count < boundary_window_context_radius) : (j += 1) {
                if (attention_mask[j] <= 0.5) continue;
                sum += features[j * hidden_size + hidden_index];
                count += 1;
            }
        },
    }
    return .{
        .mean = if (count == 0) 0.0 else sum / @as(f32, @floatFromInt(count)),
        .count = count,
    };
}

fn scatterWindowContextMeanGrad(
    out: []f32,
    attention_mask: []const f32,
    flat_index: usize,
    hidden_index: usize,
    hidden_size: usize,
    total_tokens: usize,
    max_seq_len: usize,
    grad: f32,
    comptime direction: enum { left, right },
) void {
    const seq_start = (flat_index / max_seq_len) * max_seq_len;
    const seq_end = @min(seq_start + max_seq_len, total_tokens);
    var count: usize = 0;
    switch (direction) {
        .left => {
            var j = flat_index;
            while (j > seq_start and count < boundary_window_context_radius) {
                j -= 1;
                if (attention_mask[j] <= 0.5) continue;
                count += 1;
            }
            if (count == 0) return;
            const each = grad / @as(f32, @floatFromInt(count));
            j = flat_index;
            var seen: usize = 0;
            while (j > seq_start and seen < boundary_window_context_radius) {
                j -= 1;
                if (attention_mask[j] <= 0.5) continue;
                out[j * hidden_size + hidden_index] += each;
                seen += 1;
            }
        },
        .right => {
            var j = flat_index + 1;
            while (j < seq_end and count < boundary_window_context_radius) : (j += 1) {
                if (attention_mask[j] <= 0.5) continue;
                count += 1;
            }
            if (count == 0) return;
            const each = grad / @as(f32, @floatFromInt(count));
            j = flat_index + 1;
            var seen: usize = 0;
            while (j < seq_end and seen < boundary_window_context_radius) : (j += 1) {
                if (attention_mask[j] <= 0.5) continue;
                out[j * hidden_size + hidden_index] += each;
                seen += 1;
            }
        },
    }
}

pub fn transformBoundaryFeaturesForHead(
    allocator: std.mem.Allocator,
    mode: BoundaryFeatureMode,
    features: []const f32,
    attention_mask: []const f32,
    total_tokens: usize,
    hidden_size: usize,
    max_seq_len: usize,
) !?[]f32 {
    if (mode == .token) return null;
    if (max_seq_len == 0) return error.InvalidBoundaryFeatureMaxSeqLen;
    if (features.len != total_tokens * hidden_size) return error.UnexpectedOutputShape;
    if (attention_mask.len != total_tokens) return error.UnexpectedOutputShape;

    const head_dim = boundaryFeatureDim(mode, hidden_size);
    const out = try allocator.alloc(f32, total_tokens * head_dim);
    errdefer allocator.free(out);
    @memset(out, 0);

    for (0..total_tokens) |i| {
        if (attention_mask[i] <= 0.5) continue;
        const has_prev = hasPreviousValidToken(attention_mask, i, max_seq_len);
        const has_next = hasNextValidToken(attention_mask, i, total_tokens, max_seq_len);
        for (0..hidden_size) |k| {
            const current = features[i * hidden_size + k];
            const prev = if (has_prev) features[(i - 1) * hidden_size + k] else 0.0;
            const next = if (has_next) features[(i + 1) * hidden_size + k] else 0.0;
            switch (mode) {
                .token => out[i * head_dim + k] = current,
                .prev_diff => out[i * head_dim + k] = if (has_prev) current - prev else current,
                .prev_current_diff => out[i * head_dim + k] = if (has_prev) current + (current - prev) else current,
                .prev_current_diff_concat => {
                    const base = i * head_dim;
                    out[base + k] = prev;
                    out[base + hidden_size + k] = current;
                    out[base + 2 * hidden_size + k] = if (has_prev) current - prev else current;
                },
                .prev_current_next_diff_concat => {
                    const base = i * head_dim;
                    out[base + k] = prev;
                    out[base + hidden_size + k] = current;
                    out[base + 2 * hidden_size + k] = next;
                    out[base + 3 * hidden_size + k] = if (has_prev) current - prev else current;
                    out[base + 4 * hidden_size + k] = if (has_next) current - next else current;
                },
                .window_context_diff => {
                    const left = windowContextMean(features, attention_mask, i, k, hidden_size, total_tokens, max_seq_len, .left).mean;
                    const right = windowContextMean(features, attention_mask, i, k, hidden_size, total_tokens, max_seq_len, .right).mean;
                    const base = i * head_dim;
                    out[base + k] = left;
                    out[base + hidden_size + k] = current;
                    out[base + 2 * hidden_size + k] = right;
                    out[base + 3 * hidden_size + k] = current - left;
                    out[base + 4 * hidden_size + k] = current - right;
                },
            }
        }
    }

    return out;
}

pub fn scatterBoundaryFeatureGradToEncoder(
    allocator: std.mem.Allocator,
    mode: BoundaryFeatureMode,
    transformed_grad: []const f32,
    attention_mask: []const f32,
    total_tokens: usize,
    hidden_size: usize,
    max_seq_len: usize,
) !?[]f32 {
    if (mode == .token) return null;
    if (max_seq_len == 0) return error.InvalidBoundaryFeatureMaxSeqLen;
    const head_dim = boundaryFeatureDim(mode, hidden_size);
    if (transformed_grad.len != total_tokens * head_dim) return error.UnexpectedOutputShape;
    if (attention_mask.len != total_tokens) return error.UnexpectedOutputShape;

    const out = try allocator.alloc(f32, total_tokens * hidden_size);
    errdefer allocator.free(out);
    @memset(out, 0);

    for (0..total_tokens) |i| {
        if (attention_mask[i] <= 0.5) continue;
        const has_prev = hasPreviousValidToken(attention_mask, i, max_seq_len);
        const has_next = hasNextValidToken(attention_mask, i, total_tokens, max_seq_len);
        for (0..hidden_size) |k| {
            const grad = transformed_grad[i * head_dim + k];
            switch (mode) {
                .token => out[i * hidden_size + k] += grad,
                .prev_diff => {
                    out[i * hidden_size + k] += grad;
                    if (has_prev) out[(i - 1) * hidden_size + k] -= grad;
                },
                .prev_current_diff => {
                    out[i * hidden_size + k] += if (has_prev) 2.0 * grad else grad;
                    if (has_prev) out[(i - 1) * hidden_size + k] -= grad;
                },
                .prev_current_diff_concat => {
                    const base = i * head_dim;
                    const prev_grad = transformed_grad[base + k];
                    const current_grad = transformed_grad[base + hidden_size + k];
                    const diff_grad = transformed_grad[base + 2 * hidden_size + k];
                    out[i * hidden_size + k] += current_grad + diff_grad;
                    if (has_prev) {
                        out[(i - 1) * hidden_size + k] += prev_grad - diff_grad;
                    }
                },
                .prev_current_next_diff_concat => {
                    const base = i * head_dim;
                    const prev_grad = transformed_grad[base + k];
                    const current_grad = transformed_grad[base + hidden_size + k];
                    const next_grad = transformed_grad[base + 2 * hidden_size + k];
                    const diff_prev_grad = transformed_grad[base + 3 * hidden_size + k];
                    const diff_next_grad = transformed_grad[base + 4 * hidden_size + k];
                    out[i * hidden_size + k] += current_grad + diff_prev_grad + diff_next_grad;
                    if (has_prev) out[(i - 1) * hidden_size + k] += prev_grad - diff_prev_grad;
                    if (has_next) out[(i + 1) * hidden_size + k] += next_grad - diff_next_grad;
                },
                .window_context_diff => {
                    const base = i * head_dim;
                    const left_grad = transformed_grad[base + k];
                    const current_grad = transformed_grad[base + hidden_size + k];
                    const right_grad = transformed_grad[base + 2 * hidden_size + k];
                    const diff_left_grad = transformed_grad[base + 3 * hidden_size + k];
                    const diff_right_grad = transformed_grad[base + 4 * hidden_size + k];
                    out[i * hidden_size + k] += current_grad + diff_left_grad + diff_right_grad;
                    scatterWindowContextMeanGrad(out, attention_mask, i, k, hidden_size, total_tokens, max_seq_len, left_grad - diff_left_grad, .left);
                    scatterWindowContextMeanGrad(out, attention_mask, i, k, hidden_size, total_tokens, max_seq_len, right_grad - diff_right_grad, .right);
                },
            }
        }
    }

    return out;
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
    boundary_ce_loss: f32 = 0,
    boundary_rank_loss: f32 = 0,
    boundary_local_window_loss: f32 = 0,
    contrastive_loss: f64 = 0,
    total_loss: f32 = 0,
    boundary_grad_norm: f64 = 0,
    boundary_tp: u64 = 0,
    boundary_fp: u64 = 0,
    boundary_fn: u64 = 0,
    step: u32 = 0,
    learning_rate: f32 = 0,
    boundary_head_learning_rate: f32 = 0,
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
    pub const diagnostic_threshold_count = 14;
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
    calibrated_threshold_available: bool = false,
    calibrated_boundary_threshold: f32 = 0,
    calibrated_boundary_f1: f32 = 0,
    calibrated_boundary_precision: f32 = 0,
    calibrated_boundary_recall: f32 = 0,
    calibrated_boundary_tp: u64 = 0,
    calibrated_boundary_fp: u64 = 0,
    calibrated_boundary_fn: u64 = 0,
    calibrated_predicted_positive_rate: f32 = 0,
    fitted_threshold_available: bool = false,
    fitted_boundary_threshold: f32 = 0,
    fitted_boundary_f1: f32 = 0,
    fitted_boundary_precision: f32 = 0,
    fitted_boundary_recall: f32 = 0,
    fitted_boundary_tp: u64 = 0,
    fitted_boundary_fp: u64 = 0,
    fitted_boundary_fn: u64 = 0,
    fitted_predicted_positive_rate: f32 = 0,
    average_precision: f32 = 0,
    precision_at_gold_count: f32 = 0,
    recall_at_gold_count: f32 = 0,
    f1_at_gold_count: f32 = 0,
    threshold_at_gold_count: f32 = 0,
    predicted_positive_rate_at_gold_count: f32 = 0,
    gold_count_tp: u64 = 0,
    gold_count_fp: u64 = 0,
    gold_count_fn: u64 = 0,
    max_rank_f1: f32 = 0,
    max_rank_precision: f32 = 0,
    max_rank_recall: f32 = 0,
    max_rank_threshold: f32 = 0,
    max_rank_predicted_positive_rate: f32 = 0,
    max_rank_tp: u64 = 0,
    max_rank_fp: u64 = 0,
    max_rank_fn: u64 = 0,
    sample_oracle_count_samples: u64 = 0,
    sample_oracle_count_topk_f1: f32 = 0,
    sample_oracle_count_topk_precision: f32 = 0,
    sample_oracle_count_topk_recall: f32 = 0,
    sample_oracle_count_topk_tp: u64 = 0,
    sample_oracle_count_topk_fp: u64 = 0,
    sample_oracle_count_topk_fn: u64 = 0,
    sample_oracle_count_nms_f1: f32 = 0,
    sample_oracle_count_nms_precision: f32 = 0,
    sample_oracle_count_nms_recall: f32 = 0,
    sample_oracle_count_nms_tp: u64 = 0,
    sample_oracle_count_nms_fp: u64 = 0,
    sample_oracle_count_nms_fn: u64 = 0,
    sample_oracle_count_nms_radius: u32 = 8,
    sample_oracle_count_length_window_f1: f32 = 0,
    sample_oracle_count_length_window_precision: f32 = 0,
    sample_oracle_count_length_window_recall: f32 = 0,
    sample_oracle_count_length_window_tp: u64 = 0,
    sample_oracle_count_length_window_fp: u64 = 0,
    sample_oracle_count_length_window_fn: u64 = 0,
    sample_oracle_count_length_window_min_radius: u32 = 16,
    sample_oracle_count_length_window_radius_fraction: f32 = 0.35,
    gold_positive_mean_rank: f32 = 0,
    gold_positive_mean_rank_percentile: f32 = 0,
    gold_positive_median_rank: u64 = 0,
    gold_positive_median_rank_percentile: f32 = 0,
    gold_positive_p90_rank: u64 = 0,
    gold_positive_p90_rank_percentile: f32 = 0,
    gold_positive_p99_rank: u64 = 0,
    gold_positive_p99_rank_percentile: f32 = 0,
    gold_positive_worst_rank: u64 = 0,
    gold_positive_top_5x_count: u64 = 0,
    gold_positive_top_5x_recall: f32 = 0,
    gold_positive_top_10x_count: u64 = 0,
    gold_positive_top_10x_recall: f32 = 0,
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
    pub const oracle_count_nms_radius = 8;
    pub const oracle_count_length_window_min_radius = 16;
    pub const oracle_count_length_window_radius_fraction: f32 = 0.35;
    pub const diagnostic_thresholds = [_]f32{ 0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99 };

    const RankItem = struct {
        probability: f32,
        label: bool,
        ordinal: u64,
    };

    const SequenceRankItem = struct {
        probability: f32,
        label: bool,
        position: usize,
        ordinal: u64,
    };

    const RankMetrics = struct {
        average_precision: f32 = 0,
        precision_at_gold_count: f32 = 0,
        recall_at_gold_count: f32 = 0,
        f1_at_gold_count: f32 = 0,
        threshold_at_gold_count: f32 = 0,
        predicted_positive_rate_at_gold_count: f32 = 0,
        gold_count_tp: u64 = 0,
        gold_count_fp: u64 = 0,
        gold_count_fn: u64 = 0,
        max_rank_f1: f32 = 0,
        max_rank_precision: f32 = 0,
        max_rank_recall: f32 = 0,
        max_rank_threshold: f32 = 0,
        max_rank_predicted_positive_rate: f32 = 0,
        max_rank_tp: u64 = 0,
        max_rank_fp: u64 = 0,
        max_rank_fn: u64 = 0,
        gold_positive_mean_rank: f32 = 0,
        gold_positive_mean_rank_percentile: f32 = 0,
        gold_positive_median_rank: u64 = 0,
        gold_positive_median_rank_percentile: f32 = 0,
        gold_positive_p90_rank: u64 = 0,
        gold_positive_p90_rank_percentile: f32 = 0,
        gold_positive_p99_rank: u64 = 0,
        gold_positive_p99_rank_percentile: f32 = 0,
        gold_positive_worst_rank: u64 = 0,
        gold_positive_top_5x_count: u64 = 0,
        gold_positive_top_5x_recall: f32 = 0,
        gold_positive_top_10x_count: u64 = 0,
        gold_positive_top_10x_recall: f32 = 0,
    };

    agg: fused_chunker_loss.BoundaryMetrics = .{ .tp = 0, .fp = 0, .fn_ = 0 },
    sweep_metrics: [sweep_count]fused_chunker_loss.BoundaryMetrics = [_]fused_chunker_loss.BoundaryMetrics{.{ .tp = 0, .fp = 0, .fn_ = 0 }} ** sweep_count,
    rank_items: std.ArrayListUnmanaged(RankItem) = .empty,
    calibrated_threshold: ?f32 = null,
    fitted_threshold: ?f32 = null,
    sample_oracle_count_topk: fused_chunker_loss.BoundaryMetrics = .{ .tp = 0, .fp = 0, .fn_ = 0 },
    sample_oracle_count_nms: fused_chunker_loss.BoundaryMetrics = .{ .tp = 0, .fp = 0, .fn_ = 0 },
    sample_oracle_count_length_window: fused_chunker_loss.BoundaryMetrics = .{ .tp = 0, .fp = 0, .fn_ = 0 },
    sample_oracle_count_samples: u64 = 0,
    sample_oracle_count_nms_radius: u32 = oracle_count_nms_radius,
    sample_oracle_count_length_window_min_radius: u32 = oracle_count_length_window_min_radius,
    sample_oracle_count_length_window_radius_fraction: f32 = oracle_count_length_window_radius_fraction,
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

    pub fn deinit(self: *BoundaryEvalAccumulator, allocator: std.mem.Allocator) void {
        self.rank_items.deinit(allocator);
    }

    fn rankProbability(prob: f32) f32 {
        return if (std.math.isFinite(prob)) prob else -std.math.floatMax(f32);
    }

    fn rankItemGreaterThan(_: void, a: RankItem, b: RankItem) bool {
        const pa = rankProbability(a.probability);
        const pb = rankProbability(b.probability);
        if (pa == pb) return a.ordinal < b.ordinal;
        return pa > pb;
    }

    fn sequenceRankItemGreaterThan(_: void, a: SequenceRankItem, b: SequenceRankItem) bool {
        const pa = rankProbability(a.probability);
        const pb = rankProbability(b.probability);
        if (pa == pb) return a.ordinal < b.ordinal;
        return pa > pb;
    }

    fn f1FromPrecisionRecall(precision: f32, recall: f32) f32 {
        const denom = precision + recall;
        return if (denom == 0) 0 else (2.0 * precision * recall) / denom;
    }

    fn percentileRank(ranks: []const u64, percentile: f32) u64 {
        if (ranks.len == 0) return 0;
        const raw_idx: usize = @intFromFloat(@ceil(percentile * @as(f32, @floatFromInt(ranks.len))));
        const idx = if (raw_idx == 0) 0 else @min(raw_idx - 1, ranks.len - 1);
        return ranks[idx];
    }

    fn computeRankMetrics(self: BoundaryEvalAccumulator, allocator: std.mem.Allocator) !RankMetrics {
        if (self.rank_items.items.len == 0 or self.gold_positives == 0) return .{};

        const items = try allocator.dupe(RankItem, self.rank_items.items);
        defer allocator.free(items);
        std.mem.sort(RankItem, items, {}, rankItemGreaterThan);
        const gold_ranks = try allocator.alloc(u64, @intCast(self.gold_positives));
        defer allocator.free(gold_ranks);

        const valid_f: f32 = @floatFromInt(self.valid_tokens);
        const gold_f: f32 = @floatFromInt(self.gold_positives);
        const gold_cutoff: usize = @min(@as(usize, @intCast(self.gold_positives)), items.len);
        const top_5x_cutoff: usize = @min(items.len, gold_cutoff *| 5);
        const top_10x_cutoff: usize = @min(items.len, gold_cutoff *| 10);

        var metrics = RankMetrics{};
        var tp_seen: u64 = 0;
        var ap_sum: f64 = 0;
        var rank_sum: f64 = 0;

        for (items, 0..) |item, i| {
            const rank = i + 1;
            if (item.label) {
                gold_ranks[@intCast(tp_seen)] = @intCast(rank);
                tp_seen += 1;
                ap_sum += @as(f64, @floatFromInt(tp_seen)) / @as(f64, @floatFromInt(rank));
                rank_sum += @as(f64, @floatFromInt(rank));
                if (rank <= top_5x_cutoff) metrics.gold_positive_top_5x_count += 1;
                if (rank <= top_10x_cutoff) metrics.gold_positive_top_10x_count += 1;
            }

            if (rank == gold_cutoff) {
                metrics.gold_count_tp = tp_seen;
                metrics.gold_count_fp = @as(u64, @intCast(gold_cutoff)) - tp_seen;
                metrics.gold_count_fn = self.gold_positives - tp_seen;
                metrics.precision_at_gold_count = @as(f32, @floatFromInt(tp_seen)) / @as(f32, @floatFromInt(gold_cutoff));
                metrics.recall_at_gold_count = @as(f32, @floatFromInt(tp_seen)) / gold_f;
                metrics.f1_at_gold_count = f1FromPrecisionRecall(metrics.precision_at_gold_count, metrics.recall_at_gold_count);
                metrics.threshold_at_gold_count = item.probability;
                metrics.predicted_positive_rate_at_gold_count = @as(f32, @floatFromInt(gold_cutoff)) / valid_f;
            }

            const rank_f: f32 = @floatFromInt(rank);
            const precision = @as(f32, @floatFromInt(tp_seen)) / rank_f;
            const recall = @as(f32, @floatFromInt(tp_seen)) / gold_f;
            const f1 = f1FromPrecisionRecall(precision, recall);
            if (f1 > metrics.max_rank_f1) {
                metrics.max_rank_f1 = f1;
                metrics.max_rank_precision = precision;
                metrics.max_rank_recall = recall;
                metrics.max_rank_threshold = item.probability;
                metrics.max_rank_predicted_positive_rate = rank_f / valid_f;
                metrics.max_rank_tp = tp_seen;
                metrics.max_rank_fp = @as(u64, @intCast(rank)) - tp_seen;
                metrics.max_rank_fn = self.gold_positives - tp_seen;
            }
        }

        metrics.average_precision = @as(f32, @floatCast(ap_sum / @as(f64, @floatFromInt(self.gold_positives))));
        metrics.gold_positive_mean_rank = @as(f32, @floatCast(rank_sum / @as(f64, @floatFromInt(self.gold_positives))));
        metrics.gold_positive_mean_rank_percentile = metrics.gold_positive_mean_rank / valid_f;
        metrics.gold_positive_median_rank = percentileRank(gold_ranks, 0.50);
        metrics.gold_positive_median_rank_percentile = @as(f32, @floatFromInt(metrics.gold_positive_median_rank)) / valid_f;
        metrics.gold_positive_p90_rank = percentileRank(gold_ranks, 0.90);
        metrics.gold_positive_p90_rank_percentile = @as(f32, @floatFromInt(metrics.gold_positive_p90_rank)) / valid_f;
        metrics.gold_positive_p99_rank = percentileRank(gold_ranks, 0.99);
        metrics.gold_positive_p99_rank_percentile = @as(f32, @floatFromInt(metrics.gold_positive_p99_rank)) / valid_f;
        metrics.gold_positive_worst_rank = gold_ranks[gold_ranks.len - 1];
        metrics.gold_positive_top_5x_recall = @as(f32, @floatFromInt(metrics.gold_positive_top_5x_count)) / gold_f;
        metrics.gold_positive_top_10x_recall = @as(f32, @floatFromInt(metrics.gold_positive_top_10x_count)) / gold_f;
        return metrics;
    }

    fn computeMetricsAtProbabilityThreshold(self: BoundaryEvalAccumulator, threshold: f32) fused_chunker_loss.BoundaryMetrics {
        var metrics = fused_chunker_loss.BoundaryMetrics{ .tp = 0, .fp = 0, .fn_ = 0 };
        for (self.rank_items.items) |item| {
            const predicted = item.probability > threshold;
            if (predicted and item.label) {
                metrics.tp += 1;
            } else if (predicted and !item.label) {
                metrics.fp += 1;
            } else if (!predicted and item.label) {
                metrics.fn_ += 1;
            }
        }
        return metrics;
    }

    fn positionsWithinRadius(a: usize, b: usize, radius: u32) bool {
        const delta = if (a >= b) a - b else b - a;
        return delta <= radius;
    }

    fn positionAlreadySelected(position: usize, selected_positions: []const usize) bool {
        for (selected_positions) |selected_position| {
            if (position == selected_position) return true;
        }
        return false;
    }

    fn addSampleOracleCountMetrics(
        self: *BoundaryEvalAccumulator,
        allocator: std.mem.Allocator,
        logits: []const f32,
        scalar_labels: []const f32,
        mask: ?[]const f32,
        max_seq_len: usize,
    ) !void {
        if (max_seq_len == 0) return error.InvalidBoundaryEvalMaxSeqLen;
        const total = scalar_labels.len;
        if (logits.len < total * 2) return error.InvalidBoundaryLogitShape;

        var base: usize = 0;
        while (base < total) : (base += max_seq_len) {
            const end = @min(base + max_seq_len, total);
            const seq_len = end - base;
            var items = try allocator.alloc(SequenceRankItem, seq_len);
            defer allocator.free(items);

            var active_count: usize = 0;
            var gold_count: u64 = 0;
            for (base..end) |flat_idx| {
                if (mask) |m| {
                    if (m[flat_idx] <= 0.5) continue;
                }
                const label = scalar_labels[flat_idx] > 0.5;
                if (label) gold_count += 1;
                items[active_count] = .{
                    .probability = fused_chunker_loss.positiveBoundaryProbability(logits[flat_idx * 2], logits[flat_idx * 2 + 1]),
                    .label = label,
                    .position = flat_idx - base,
                    .ordinal = @intCast(flat_idx),
                };
                active_count += 1;
            }

            if (active_count == 0 or gold_count == 0) continue;
            self.sample_oracle_count_samples += 1;

            const active_items = items[0..active_count];
            std.mem.sort(SequenceRankItem, active_items, {}, sequenceRankItemGreaterThan);
            const target_count: usize = @min(@as(usize, @intCast(gold_count)), active_count);

            var topk_tp: u64 = 0;
            for (active_items[0..target_count]) |item| {
                if (item.label) topk_tp += 1;
            }
            self.sample_oracle_count_topk.tp += topk_tp;
            self.sample_oracle_count_topk.fp += @as(u64, @intCast(target_count)) - topk_tp;
            self.sample_oracle_count_topk.fn_ += gold_count - topk_tp;

            var selected_positions = try allocator.alloc(usize, target_count);
            defer allocator.free(selected_positions);
            var selected_count: usize = 0;
            var nms_tp: u64 = 0;
            for (active_items) |item| {
                if (selected_count >= target_count) break;
                var suppressed = false;
                for (selected_positions[0..selected_count]) |selected_position| {
                    if (positionsWithinRadius(item.position, selected_position, self.sample_oracle_count_nms_radius)) {
                        suppressed = true;
                        break;
                    }
                }
                if (suppressed) continue;
                selected_positions[selected_count] = item.position;
                selected_count += 1;
                if (item.label) nms_tp += 1;
            }
            self.sample_oracle_count_nms.tp += nms_tp;
            self.sample_oracle_count_nms.fp += @as(u64, @intCast(selected_count)) - nms_tp;
            self.sample_oracle_count_nms.fn_ += gold_count - nms_tp;

            var length_selected_positions = try allocator.alloc(usize, target_count);
            defer allocator.free(length_selected_positions);
            var length_selected_count: usize = 0;
            var length_window_tp: u64 = 0;
            const segment_len = @as(f32, @floatFromInt(active_count)) / @as(f32, @floatFromInt(gold_count + 1));
            const dynamic_radius: u32 = @intFromFloat(@ceil(segment_len * self.sample_oracle_count_length_window_radius_fraction));
            const window_radius = @max(self.sample_oracle_count_length_window_min_radius, dynamic_radius);
            for (0..target_count) |slot_idx| {
                const boundary_ordinal = slot_idx + 1;
                const center_f = segment_len * @as(f32, @floatFromInt(boundary_ordinal));
                const center_position = @min(@as(usize, @intFromFloat(@round(center_f))), seq_len - 1);

                var chosen_index: ?usize = null;
                for (active_items, 0..) |item, item_idx| {
                    if (positionAlreadySelected(item.position, length_selected_positions[0..length_selected_count])) continue;
                    if (!positionsWithinRadius(item.position, center_position, window_radius)) continue;
                    chosen_index = item_idx;
                    break;
                }
                if (chosen_index == null) {
                    for (active_items, 0..) |item, item_idx| {
                        if (positionAlreadySelected(item.position, length_selected_positions[0..length_selected_count])) continue;
                        chosen_index = item_idx;
                        break;
                    }
                }
                if (chosen_index) |item_idx| {
                    const item = active_items[item_idx];
                    length_selected_positions[length_selected_count] = item.position;
                    length_selected_count += 1;
                    if (item.label) length_window_tp += 1;
                }
            }
            self.sample_oracle_count_length_window.tp += length_window_tp;
            self.sample_oracle_count_length_window.fp += @as(u64, @intCast(length_selected_count)) - length_window_tp;
            self.sample_oracle_count_length_window.fn_ += gold_count - length_window_tp;
        }
    }

    pub fn addLogits(
        self: *BoundaryEvalAccumulator,
        allocator: std.mem.Allocator,
        logits: []const f32,
        labels: []const f32,
        mask: ?[]const f32,
    ) !void {
        try self.addLogitsInternal(allocator, logits, labels, mask, null);
    }

    pub fn addLogitsBySample(
        self: *BoundaryEvalAccumulator,
        allocator: std.mem.Allocator,
        logits: []const f32,
        labels: []const f32,
        mask: ?[]const f32,
        max_seq_len: usize,
    ) !void {
        try self.addLogitsInternal(allocator, logits, labels, mask, max_seq_len);
    }

    fn addLogitsInternal(
        self: *BoundaryEvalAccumulator,
        allocator: std.mem.Allocator,
        logits: []const f32,
        labels: []const f32,
        mask: ?[]const f32,
        max_seq_len: ?usize,
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
            const is_positive = scalar_labels[i] > 0.5;
            try self.rank_items.append(allocator, .{
                .probability = prob,
                .label = is_positive,
                .ordinal = self.valid_tokens - 1,
            });
            if (is_positive) {
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

        if (max_seq_len) |seq_len| {
            try self.addSampleOracleCountMetrics(allocator, logits, scalar_labels, mask, seq_len);
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

    pub fn finish(self: BoundaryEvalAccumulator, allocator: std.mem.Allocator) !EvalSummary {
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
        const rank_metrics = try self.computeRankMetrics(allocator);
        const calibrated_threshold = self.calibrated_threshold orelse 0.0;
        const calibrated_metrics = if (self.calibrated_threshold != null)
            self.computeMetricsAtProbabilityThreshold(calibrated_threshold)
        else
            fused_chunker_loss.BoundaryMetrics{ .tp = 0, .fp = 0, .fn_ = 0 };
        const calibrated_predicted = calibrated_metrics.tp + calibrated_metrics.fp;
        const fitted_threshold = self.fitted_threshold orelse 0.0;
        const fitted_metrics = if (self.fitted_threshold != null)
            self.computeMetricsAtProbabilityThreshold(fitted_threshold)
        else
            fused_chunker_loss.BoundaryMetrics{ .tp = 0, .fp = 0, .fn_ = 0 };
        const fitted_predicted = fitted_metrics.tp + fitted_metrics.fp;

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
            .calibrated_threshold_available = self.calibrated_threshold != null,
            .calibrated_boundary_threshold = calibrated_threshold,
            .calibrated_boundary_f1 = calibrated_metrics.f1(),
            .calibrated_boundary_precision = calibrated_metrics.precision(),
            .calibrated_boundary_recall = calibrated_metrics.recall(),
            .calibrated_boundary_tp = calibrated_metrics.tp,
            .calibrated_boundary_fp = calibrated_metrics.fp,
            .calibrated_boundary_fn = calibrated_metrics.fn_,
            .calibrated_predicted_positive_rate = if (self.valid_tokens == 0) 0.0 else @as(f32, @floatFromInt(calibrated_predicted)) / valid_f,
            .fitted_threshold_available = self.fitted_threshold != null,
            .fitted_boundary_threshold = fitted_threshold,
            .fitted_boundary_f1 = fitted_metrics.f1(),
            .fitted_boundary_precision = fitted_metrics.precision(),
            .fitted_boundary_recall = fitted_metrics.recall(),
            .fitted_boundary_tp = fitted_metrics.tp,
            .fitted_boundary_fp = fitted_metrics.fp,
            .fitted_boundary_fn = fitted_metrics.fn_,
            .fitted_predicted_positive_rate = if (self.valid_tokens == 0) 0.0 else @as(f32, @floatFromInt(fitted_predicted)) / valid_f,
            .average_precision = rank_metrics.average_precision,
            .precision_at_gold_count = rank_metrics.precision_at_gold_count,
            .recall_at_gold_count = rank_metrics.recall_at_gold_count,
            .f1_at_gold_count = rank_metrics.f1_at_gold_count,
            .threshold_at_gold_count = rank_metrics.threshold_at_gold_count,
            .predicted_positive_rate_at_gold_count = rank_metrics.predicted_positive_rate_at_gold_count,
            .gold_count_tp = rank_metrics.gold_count_tp,
            .gold_count_fp = rank_metrics.gold_count_fp,
            .gold_count_fn = rank_metrics.gold_count_fn,
            .max_rank_f1 = rank_metrics.max_rank_f1,
            .max_rank_precision = rank_metrics.max_rank_precision,
            .max_rank_recall = rank_metrics.max_rank_recall,
            .max_rank_threshold = rank_metrics.max_rank_threshold,
            .max_rank_predicted_positive_rate = rank_metrics.max_rank_predicted_positive_rate,
            .max_rank_tp = rank_metrics.max_rank_tp,
            .max_rank_fp = rank_metrics.max_rank_fp,
            .max_rank_fn = rank_metrics.max_rank_fn,
            .sample_oracle_count_samples = self.sample_oracle_count_samples,
            .sample_oracle_count_topk_f1 = self.sample_oracle_count_topk.f1(),
            .sample_oracle_count_topk_precision = self.sample_oracle_count_topk.precision(),
            .sample_oracle_count_topk_recall = self.sample_oracle_count_topk.recall(),
            .sample_oracle_count_topk_tp = self.sample_oracle_count_topk.tp,
            .sample_oracle_count_topk_fp = self.sample_oracle_count_topk.fp,
            .sample_oracle_count_topk_fn = self.sample_oracle_count_topk.fn_,
            .sample_oracle_count_nms_f1 = self.sample_oracle_count_nms.f1(),
            .sample_oracle_count_nms_precision = self.sample_oracle_count_nms.precision(),
            .sample_oracle_count_nms_recall = self.sample_oracle_count_nms.recall(),
            .sample_oracle_count_nms_tp = self.sample_oracle_count_nms.tp,
            .sample_oracle_count_nms_fp = self.sample_oracle_count_nms.fp,
            .sample_oracle_count_nms_fn = self.sample_oracle_count_nms.fn_,
            .sample_oracle_count_nms_radius = self.sample_oracle_count_nms_radius,
            .sample_oracle_count_length_window_f1 = self.sample_oracle_count_length_window.f1(),
            .sample_oracle_count_length_window_precision = self.sample_oracle_count_length_window.precision(),
            .sample_oracle_count_length_window_recall = self.sample_oracle_count_length_window.recall(),
            .sample_oracle_count_length_window_tp = self.sample_oracle_count_length_window.tp,
            .sample_oracle_count_length_window_fp = self.sample_oracle_count_length_window.fp,
            .sample_oracle_count_length_window_fn = self.sample_oracle_count_length_window.fn_,
            .sample_oracle_count_length_window_min_radius = self.sample_oracle_count_length_window_min_radius,
            .sample_oracle_count_length_window_radius_fraction = self.sample_oracle_count_length_window_radius_fraction,
            .gold_positive_mean_rank = rank_metrics.gold_positive_mean_rank,
            .gold_positive_mean_rank_percentile = rank_metrics.gold_positive_mean_rank_percentile,
            .gold_positive_median_rank = rank_metrics.gold_positive_median_rank,
            .gold_positive_median_rank_percentile = rank_metrics.gold_positive_median_rank_percentile,
            .gold_positive_p90_rank = rank_metrics.gold_positive_p90_rank,
            .gold_positive_p90_rank_percentile = rank_metrics.gold_positive_p90_rank_percentile,
            .gold_positive_p99_rank = rank_metrics.gold_positive_p99_rank,
            .gold_positive_p99_rank_percentile = rank_metrics.gold_positive_p99_rank_percentile,
            .gold_positive_worst_rank = rank_metrics.gold_positive_worst_rank,
            .gold_positive_top_5x_count = rank_metrics.gold_positive_top_5x_count,
            .gold_positive_top_5x_recall = rank_metrics.gold_positive_top_5x_recall,
            .gold_positive_top_10x_count = rank_metrics.gold_positive_top_10x_count,
            .gold_positive_top_10x_recall = rank_metrics.gold_positive_top_10x_recall,
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
    std.debug.print("{s} rank_metrics ap={d:.6} p_at_gold={d:.4} r_at_gold={d:.4} f1_at_gold={d:.4} threshold_at_gold={d:.6} pred_at_gold={d:.6} counts_at_gold tp={d} fp={d} fn={d} max_rank_f1={d:.4} max_rank_p={d:.4} max_rank_r={d:.4} max_rank_threshold={d:.6} max_rank_pred={d:.6} max_rank_counts tp={d} fp={d} fn={d}\n", .{
        label,
        summary.average_precision,
        summary.precision_at_gold_count,
        summary.recall_at_gold_count,
        summary.f1_at_gold_count,
        summary.threshold_at_gold_count,
        summary.predicted_positive_rate_at_gold_count,
        summary.gold_count_tp,
        summary.gold_count_fp,
        summary.gold_count_fn,
        summary.max_rank_f1,
        summary.max_rank_precision,
        summary.max_rank_recall,
        summary.max_rank_threshold,
        summary.max_rank_predicted_positive_rate,
        summary.max_rank_tp,
        summary.max_rank_fp,
        summary.max_rank_fn,
    });

    if (summary.sample_oracle_count_samples > 0) {
        std.debug.print("{s} sample_oracle_count samples={d} topk_f1={d:.4} topk_p={d:.4} topk_r={d:.4} topk_counts tp={d} fp={d} fn={d} nms_f1={d:.4} nms_p={d:.4} nms_r={d:.4} nms_radius={d} nms_counts tp={d} fp={d} fn={d} length_window_f1={d:.4} length_window_p={d:.4} length_window_r={d:.4} length_window_min_radius={d} length_window_radius_fraction={d:.3} length_window_counts tp={d} fp={d} fn={d}\n", .{
            label,
            summary.sample_oracle_count_samples,
            summary.sample_oracle_count_topk_f1,
            summary.sample_oracle_count_topk_precision,
            summary.sample_oracle_count_topk_recall,
            summary.sample_oracle_count_topk_tp,
            summary.sample_oracle_count_topk_fp,
            summary.sample_oracle_count_topk_fn,
            summary.sample_oracle_count_nms_f1,
            summary.sample_oracle_count_nms_precision,
            summary.sample_oracle_count_nms_recall,
            summary.sample_oracle_count_nms_radius,
            summary.sample_oracle_count_nms_tp,
            summary.sample_oracle_count_nms_fp,
            summary.sample_oracle_count_nms_fn,
            summary.sample_oracle_count_length_window_f1,
            summary.sample_oracle_count_length_window_precision,
            summary.sample_oracle_count_length_window_recall,
            summary.sample_oracle_count_length_window_min_radius,
            summary.sample_oracle_count_length_window_radius_fraction,
            summary.sample_oracle_count_length_window_tp,
            summary.sample_oracle_count_length_window_fp,
            summary.sample_oracle_count_length_window_fn,
        });
    }

    std.debug.print("{s} margin_means gold_pos={d:.6} gold_neg={d:.6} logit0_pos={d:.6} logit1_pos={d:.6} logit0_neg={d:.6} logit1_neg={d:.6}\n", .{
        label,
        summary.mean_boundary_margin_gold_positive,
        summary.mean_boundary_margin_gold_negative,
        summary.mean_logit0_gold_positive,
        summary.mean_logit1_gold_positive,
        summary.mean_logit0_gold_negative,
        summary.mean_logit1_gold_negative,
    });

    if (summary.calibrated_threshold_available) {
        std.debug.print("{s} calibrated_threshold threshold={d:.6} f1={d:.4} precision={d:.4} recall={d:.4} pred={d:.6} counts tp={d} fp={d} fn={d}\n", .{
            label,
            summary.calibrated_boundary_threshold,
            summary.calibrated_boundary_f1,
            summary.calibrated_boundary_precision,
            summary.calibrated_boundary_recall,
            summary.calibrated_predicted_positive_rate,
            summary.calibrated_boundary_tp,
            summary.calibrated_boundary_fp,
            summary.calibrated_boundary_fn,
        });
    }

    if (summary.fitted_threshold_available) {
        std.debug.print("{s} fitted_threshold threshold={d:.6} f1={d:.4} precision={d:.4} recall={d:.4} pred={d:.6} counts tp={d} fp={d} fn={d}\n", .{
            label,
            summary.fitted_boundary_threshold,
            summary.fitted_boundary_f1,
            summary.fitted_boundary_precision,
            summary.fitted_boundary_recall,
            summary.fitted_predicted_positive_rate,
            summary.fitted_boundary_tp,
            summary.fitted_boundary_fp,
            summary.fitted_boundary_fn,
        });
    }

    std.debug.print("{s} gold_rank mean={d:.2}({d:.4}) median={d}({d:.4}) p90={d}({d:.4}) p99={d}({d:.4}) worst={d} top5x={d}/{d:.4} top10x={d}/{d:.4}\n", .{
        label,
        summary.gold_positive_mean_rank,
        summary.gold_positive_mean_rank_percentile,
        summary.gold_positive_median_rank,
        summary.gold_positive_median_rank_percentile,
        summary.gold_positive_p90_rank,
        summary.gold_positive_p90_rank_percentile,
        summary.gold_positive_p99_rank,
        summary.gold_positive_p99_rank_percentile,
        summary.gold_positive_worst_rank,
        summary.gold_positive_top_5x_count,
        summary.gold_positive_top_5x_recall,
        summary.gold_positive_top_10x_count,
        summary.gold_positive_top_10x_recall,
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

    // Shape-keyed cache for the opt-in compiled boundary-head fast path
    // (ANTFLY_FUSED_CHUNKER_COMPILED_BOUNDARY_HEAD=1): one compiled forward
    // executable producing logits plus one compiled synthetic-VJP training
    // session per gradient variant. The CE loss and its logits gradient stay
    // on the CPU exactly like the eager ops path.
    compiled_boundary_head_cache: std.AutoHashMapUnmanaged(usize, CompiledBoundaryHeadCacheEntry) = .{},
    compiled_boundary_head_failed: bool = false,

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

    const CompiledBoundaryHeadCacheEntry = struct {
        forward_graph: fused_chunker_loss.BoundaryHeadForwardGraph,
        forward_compiled: mpsgraph_executor.CompiledGraph,
        vjp_graph: fused_chunker_loss.BoundaryHeadSyntheticVJPGraph,
        head_session: ?training.CompiledTrainSession = null,
        encoder_grad_session: ?training.CompiledTrainSession = null,

        fn deinit(self: *CompiledBoundaryHeadCacheEntry) void {
            if (self.head_session) |*session| session.deinit();
            if (self.encoder_grad_session) |*session| session.deinit();
            self.forward_compiled.deinit();
            self.forward_graph.deinit();
            self.vjp_graph.deinit();
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
            .boundary_rank_loss_weight = config.boundary_rank_loss_weight,
            .boundary_rank_loss_margin = config.boundary_rank_loss_margin,
            .boundary_rank_loss_top_k = config.boundary_rank_loss_top_k,
            .boundary_same_token_rank_loss_weight = config.boundary_same_token_rank_loss_weight,
            .boundary_same_token_rank_loss_top_k = config.boundary_same_token_rank_loss_top_k,
            .boundary_same_token_negative_weight = config.boundary_same_token_negative_weight,
            .boundary_same_token_negative_top_k = config.boundary_same_token_negative_top_k,
            .boundary_candidate_rank_loss_weight = config.boundary_candidate_rank_loss_weight,
            .boundary_candidate_rank_loss_top_k = config.boundary_candidate_rank_loss_top_k,
            .boundary_candidate_negative_weight = config.boundary_candidate_negative_weight,
            .boundary_gold_count_rank_loss_weight = config.boundary_gold_count_rank_loss_weight,
            .boundary_gold_count_rank_loss_margin = config.boundary_gold_count_rank_loss_margin,
            .boundary_gold_count_rank_loss_negative_multiplier = config.boundary_gold_count_rank_loss_negative_multiplier,
            .boundary_local_window_loss_weight = config.boundary_local_window_loss_weight,
            .boundary_local_window_radius = config.boundary_local_window_radius,
            .enable_splade = config.enable_splade,
            .lambda_splade = config.lambda_splade,
            .lambda_flops = config.lambda_flops,
            .splade_focus_epoch = config.splade_focus_epoch,
            .use_mrl = config.use_mrl,
            .mrl_dims = config.mrl_dims,
            .mrl_weights = config.mrl_weights,
        };

        const boundary_input_dim = boundaryFeatureDim(config.boundary_feature_mode, @intCast(config.hidden_size));
        var head = try BoundaryHead.initWithSeed(
            allocator,
            boundary_input_dim,
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
        var compiled_cache_it = self.compiled_boundary_head_cache.iterator();
        while (compiled_cache_it.next()) |entry| entry.value_ptr.deinit();
        self.compiled_boundary_head_cache.deinit(self.allocator);
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
            self.boundary_head.hidden_dim,
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

    fn compiledBoundaryHeadEnabled(self: *FusedTrainer) bool {
        if (self.compiled_boundary_head_failed) return false;
        return platform.env.getenvBoolDefault("ANTFLY_FUSED_CHUNKER_COMPILED_BOUNDARY_HEAD", false);
    }

    fn getCompiledBoundaryHeadEntry(self: *FusedTrainer, total_tokens: usize) !*CompiledBoundaryHeadCacheEntry {
        if (self.compiled_boundary_head_cache.getPtr(total_tokens)) |entry| return entry;

        const H = self.boundary_head.hidden_dim;
        const M: usize = @intCast(self.config.boundary_mlp_dim);
        var forward_graph = try fused_chunker_loss.BoundaryHeadForwardGraph.init(self.allocator, total_tokens, H, M);
        errdefer forward_graph.deinit();
        var forward_compiled = try mpsgraph_executor.CompiledGraph.compile(self.allocator, &forward_graph.graph);
        errdefer forward_compiled.deinit();
        var vjp_graph = try fused_chunker_loss.BoundaryHeadSyntheticVJPGraph.init(self.allocator, total_tokens, H, M);
        errdefer vjp_graph.deinit();

        try self.compiled_boundary_head_cache.put(self.allocator, total_tokens, .{
            .forward_graph = forward_graph,
            .forward_compiled = forward_compiled,
            .vjp_graph = vjp_graph,
        });
        return self.compiled_boundary_head_cache.getPtr(total_tokens).?;
    }

    fn getCompiledBoundaryHeadSession(
        self: *FusedTrainer,
        entry: *CompiledBoundaryHeadCacheEntry,
        want_features_grad: bool,
    ) !*training.CompiledTrainSession {
        if (want_features_grad) {
            if (entry.encoder_grad_session == null) {
                entry.encoder_grad_session = try training.CompiledTrainSession.init(
                    self.allocator,
                    &entry.vjp_graph.graph,
                    entry.vjp_graph.loss_id,
                    .{
                        .trainable_params = &.{ "features", "w1", "b1", "w2", "b2" },
                        .execution_strategy = .mpsgraph_preferred,
                    },
                );
            }
            return &entry.encoder_grad_session.?;
        }
        if (entry.head_session == null) {
            entry.head_session = try training.CompiledTrainSession.init(
                self.allocator,
                &entry.vjp_graph.graph,
                entry.vjp_graph.loss_id,
                .{
                    .trainable_params = &.{ "w1", "b1", "w2", "b2" },
                    .execution_strategy = .mpsgraph_preferred,
                },
            );
        }
        return &entry.head_session.?;
    }

    /// Compiled variant of `runBoundaryHeadCeOpsStep`: the head forward runs
    /// as one cached MPSGraph executable, the CE loss and its logits gradient
    /// are computed on the CPU exactly as in the eager ops path (same host
    /// RNG for the dropout masks), and the parameter gradients come from one
    /// cached synthetic-VJP training session fed with those logits gradients.
    fn runBoundaryHeadCeCompiledStep(
        self: *FusedTrainer,
        allocator: std.mem.Allocator,
        features: []const f32,
        boundary_labels: []const f32,
        attention_mask: []const f32,
        input_ids: ?[]const i32,
        boundary_candidate_mask: ?[]const f32,
        total_tokens: usize,
        want_features_grad: bool,
    ) !BoundaryTrainStepResult {
        const H = self.boundary_head.hidden_dim;
        const M: usize = @intCast(self.config.boundary_mlp_dim);
        if (features.len != total_tokens * H) return error.UnexpectedOutputShape;
        if (boundary_labels.len != total_tokens * 2) return error.UnexpectedOutputShape;
        if (attention_mask.len != total_tokens) return error.UnexpectedOutputShape;

        const entry = try self.getCompiledBoundaryHeadEntry(total_tokens);

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

        const forward_nodes = entry.forward_graph.nodes;
        const forward_feeds = [_]mpsgraph_executor.HostInput{
            .{ .node_id = forward_nodes.feature_id, .data = features },
            .{ .node_id = forward_nodes.feature_dropout_mask_id, .data = feature_dropout_mask },
            .{ .node_id = forward_nodes.hidden_dropout_mask_id, .data = hidden_dropout_mask },
            .{ .node_id = forward_nodes.w1_id, .data = self.boundary_head.w1 },
            .{ .node_id = forward_nodes.b1_id, .data = self.boundary_head.b1 },
            .{ .node_id = forward_nodes.w2_id, .data = self.boundary_head.w2 },
            .{ .node_id = forward_nodes.b2_id, .data = self.boundary_head.b2 },
        };
        var forward_result = try entry.forward_compiled.executeWithHostInputs(
            allocator,
            self.cb,
            null,
            &forward_feeds,
        );
        defer forward_result.deinit();
        if (forward_result.outputs.len != 1 or forward_result.outputs[0].len != total_tokens * 2) {
            return error.UnexpectedOutputShape;
        }
        const logits = forward_result.outputs[0];

        const ce = try computeBoundaryWeightedCeAndLogitGradWithCandidateMask(
            allocator,
            logits,
            boundary_labels,
            attention_mask,
            total_tokens,
            self.loss_config.pos_weight,
            self.loss_config.boundary_rank_loss_weight,
            self.loss_config.boundary_rank_loss_margin,
            self.loss_config.boundary_rank_loss_top_k,
            self.loss_config.boundary_same_token_rank_loss_weight,
            self.loss_config.boundary_same_token_rank_loss_top_k,
            self.loss_config.boundary_same_token_negative_weight,
            self.loss_config.boundary_same_token_negative_top_k,
            self.loss_config.boundary_candidate_rank_loss_weight,
            self.loss_config.boundary_candidate_rank_loss_top_k,
            self.loss_config.boundary_candidate_negative_weight,
            boundary_candidate_mask,
            input_ids,
            @intCast(self.config.max_seq_len),
            self.loss_config.boundary_gold_count_rank_loss_weight,
            self.loss_config.boundary_gold_count_rank_loss_margin,
            self.loss_config.boundary_gold_count_rank_loss_negative_multiplier,
            self.loss_config.boundary_local_window_loss_weight,
            self.loss_config.boundary_local_window_radius,
        );
        defer allocator.free(ce.logit_grad);

        const session = try self.getCompiledBoundaryHeadSession(entry, want_features_grad);
        const vjp_nodes = entry.vjp_graph.nodes;
        const vjp_feeds = [_]training.HostFeed{
            .{ .node_id = vjp_nodes.feature_id, .data = features },
            .{ .node_id = vjp_nodes.feature_dropout_mask_id, .data = feature_dropout_mask },
            .{ .node_id = vjp_nodes.hidden_dropout_mask_id, .data = hidden_dropout_mask },
            .{ .node_id = vjp_nodes.w1_id, .data = self.boundary_head.w1 },
            .{ .node_id = vjp_nodes.b1_id, .data = self.boundary_head.b1 },
            .{ .node_id = vjp_nodes.w2_id, .data = self.boundary_head.w2 },
            .{ .node_id = vjp_nodes.b2_id, .data = self.boundary_head.b2 },
            .{ .node_id = entry.vjp_graph.dlogits_id, .data = ce.logit_grad },
        };
        var step_result = try session.executeWithHostFeeds(self.cb, null, &vjp_feeds);
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
            const features_src = step_result.gradients.get("features") orelse return error.MissingGradient;
            features_grad = try allocator.dupe(f32, features_src);
        }

        return .{
            .boundary_loss = ce.loss,
            .boundary_ce_loss = ce.ce_loss,
            .boundary_rank_loss = ce.rank_loss,
            .boundary_local_window_loss = ce.local_window_loss,
            .w1_grad = w1_grad,
            .b1_grad = b1_grad,
            .w2_grad = w2_grad,
            .b2_grad = b2_grad,
            .features_grad = features_grad,
        };
    }

    const BoundaryTrainStepResult = struct {
        boundary_loss: f32,
        boundary_ce_loss: f32 = 0,
        boundary_rank_loss: f32 = 0,
        boundary_local_window_loss: f32 = 0,
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
        input_ids: ?[]const i32,
        boundary_candidate_mask: ?[]const f32,
        total_tokens: usize,
        want_features_grad: bool,
    ) !BoundaryTrainStepResult {
        const H: usize = @intCast(self.config.hidden_size);
        const head_features_owned = try transformBoundaryFeaturesForHead(
            allocator,
            self.config.boundary_feature_mode,
            features,
            attention_mask,
            total_tokens,
            H,
            @intCast(self.config.max_seq_len),
        );
        defer if (head_features_owned) |owned| allocator.free(owned);
        const head_features = head_features_owned orelse features;

        var result = try self.runBoundaryHeadTrainingRaw(
            allocator,
            head_features,
            boundary_labels,
            attention_mask,
            input_ids,
            boundary_candidate_mask,
            total_tokens,
            want_features_grad,
        );
        errdefer result.deinit(allocator);

        if (head_features_owned != null) {
            if (result.features_grad) |grad| {
                const encoder_grad = try scatterBoundaryFeatureGradToEncoder(
                    allocator,
                    self.config.boundary_feature_mode,
                    grad,
                    attention_mask,
                    total_tokens,
                    H,
                    @intCast(self.config.max_seq_len),
                );
                if (encoder_grad) |mapped| {
                    allocator.free(grad);
                    result.features_grad = mapped;
                }
            }
        }

        return result;
    }

    fn runBoundaryHeadTrainingRaw(
        self: *FusedTrainer,
        allocator: std.mem.Allocator,
        features: []const f32,
        boundary_labels: []const f32,
        attention_mask: []const f32,
        input_ids: ?[]const i32,
        boundary_candidate_mask: ?[]const f32,
        total_tokens: usize,
        want_features_grad: bool,
    ) !BoundaryTrainStepResult {
        if (!self.loss_config.use_focal and self.cb.kind() == .metal) {
            return self.runBoundaryHeadCeOpsStep(
                allocator,
                features,
                boundary_labels,
                attention_mask,
                input_ids,
                boundary_candidate_mask,
                total_tokens,
                want_features_grad,
            ) catch |err| {
                if (self.loss_config.boundary_rank_loss_weight > 0.0 or
                    self.loss_config.boundary_same_token_rank_loss_weight > 0.0 or
                    self.loss_config.boundary_same_token_negative_weight > 1.0 or
                    self.loss_config.boundary_candidate_rank_loss_weight > 0.0 or
                    self.loss_config.boundary_candidate_negative_weight > 1.0 or
                    self.loss_config.boundary_gold_count_rank_loss_weight > 0.0 or
                    self.loss_config.boundary_local_window_loss_weight > 0.0) return err;
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
        if (self.loss_config.boundary_rank_loss_weight > 0.0 or
            self.loss_config.boundary_same_token_rank_loss_weight > 0.0 or
            self.loss_config.boundary_same_token_negative_weight > 1.0 or
            self.loss_config.boundary_candidate_rank_loss_weight > 0.0 or
            self.loss_config.boundary_candidate_negative_weight > 1.0 or
            self.loss_config.boundary_gold_count_rank_loss_weight > 0.0 or
            self.loss_config.boundary_local_window_loss_weight > 0.0) return error.UnsupportedManualBoundaryLossGraphPath;

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

        const H = self.boundary_head.hidden_dim;
        if (features.len != total_tokens * H) return error.UnexpectedOutputShape;
        const feature_dropout_mask = try allocInvertedDropoutMask(
            allocator,
            total_tokens * H,
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
            @intCast(H),
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
            @intCast(H),
        });
        try putRuntimeInput(allocator, self.cb, &rt, graph.hidden_dropout_mask_id, hidden_dropout_mask, &.{
            @intCast(total_tokens),
            @intCast(self.config.boundary_mlp_dim),
        });
        try putRuntimeInput(allocator, self.cb, &rt, graph.w1_id, self.boundary_head.w1, &.{
            @intCast(self.config.boundary_mlp_dim),
            @intCast(H),
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
            .boundary_ce_loss = step_result.loss,
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
        input_ids: ?[]const i32,
        boundary_candidate_mask: ?[]const f32,
        total_tokens: usize,
        want_features_grad: bool,
    ) !BoundaryTrainStepResult {
        const H = self.boundary_head.hidden_dim;
        const M: usize = @intCast(self.config.boundary_mlp_dim);
        if (features.len != total_tokens * H) return error.UnexpectedOutputShape;
        if (boundary_labels.len != total_tokens * 2) return error.UnexpectedOutputShape;
        if (attention_mask.len != total_tokens) return error.UnexpectedOutputShape;
        if (input_ids) |ids| {
            if (ids.len != total_tokens) return error.UnexpectedOutputShape;
        } else if (self.loss_config.boundary_same_token_rank_loss_weight > 0.0 or
            self.loss_config.boundary_same_token_negative_weight > 1.0)
        {
            return error.MissingBoundarySameTokenIds;
        }
        if (boundary_candidate_mask) |candidates| {
            if (candidates.len != total_tokens) return error.UnexpectedOutputShape;
        } else if (self.loss_config.boundary_candidate_rank_loss_weight > 0.0 or
            self.loss_config.boundary_candidate_negative_weight > 1.0)
        {
            return error.MissingBoundaryCandidateMask;
        }

        if (self.compiledBoundaryHeadEnabled()) {
            if (self.runBoundaryHeadCeCompiledStep(
                allocator,
                features,
                boundary_labels,
                attention_mask,
                input_ids,
                boundary_candidate_mask,
                total_tokens,
                want_features_grad,
            )) |result| {
                return result;
            } else |err| {
                self.compiled_boundary_head_failed = true;
                std.log.warn(
                    "fused_chunker compiled boundary head failed with {s}; falling back to eager ops path",
                    .{@errorName(err)},
                );
            }
        }

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

        const ce = try computeBoundaryWeightedCeAndLogitGradWithCandidateMask(
            allocator,
            logits,
            boundary_labels,
            attention_mask,
            total_tokens,
            self.loss_config.pos_weight,
            self.loss_config.boundary_rank_loss_weight,
            self.loss_config.boundary_rank_loss_margin,
            self.loss_config.boundary_rank_loss_top_k,
            self.loss_config.boundary_same_token_rank_loss_weight,
            self.loss_config.boundary_same_token_rank_loss_top_k,
            self.loss_config.boundary_same_token_negative_weight,
            self.loss_config.boundary_same_token_negative_top_k,
            self.loss_config.boundary_candidate_rank_loss_weight,
            self.loss_config.boundary_candidate_rank_loss_top_k,
            self.loss_config.boundary_candidate_negative_weight,
            boundary_candidate_mask,
            input_ids,
            @intCast(self.config.max_seq_len),
            self.loss_config.boundary_gold_count_rank_loss_weight,
            self.loss_config.boundary_gold_count_rank_loss_margin,
            self.loss_config.boundary_gold_count_rank_loss_negative_multiplier,
            self.loss_config.boundary_local_window_loss_weight,
            self.loss_config.boundary_local_window_radius,
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
            .boundary_ce_loss = ce.ce_loss,
            .boundary_rank_loss = ce.rank_loss,
            .boundary_local_window_loss = ce.local_window_loss,
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
    /// input_ids:         optional [total_tokens] token ids for same-token rank loss
    /// candidate_mask:    optional [total_tokens] sentence-like negative mask
    /// chunk_embeddings:  [B*C*E] late-chunked embeddings
    /// chunk_mask:        [B*C] valid chunk mask
    /// doc_ids:           [B*C] document index per chunk
    pub fn trainStep(
        self: *FusedTrainer,
        allocator: std.mem.Allocator,
        features: []const f32,
        boundary_labels: []const f32,
        attention_mask: []const f32,
        input_ids: ?[]const i32,
        boundary_candidate_mask: ?[]const f32,
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
            input_ids,
            boundary_candidate_mask,
            total_tokens,
            false,
        );
        defer boundary_step.deinit(allocator);
        sanitizeBoundaryStepGradients(&boundary_step);

        const boundary_loss = boundary_step.boundary_loss;
        const boundary_ce_loss = boundary_step.boundary_ce_loss;
        const boundary_rank_loss = boundary_step.boundary_rank_loss;
        const boundary_local_window_loss = boundary_step.boundary_local_window_loss;
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
        var applied_boundary_head_lr: f32 = 0.0;
        if (self.accum_count >= accum_steps) {
            applied_lr = lr;
            applied_boundary_head_lr = self.config.boundaryHeadLearningRate(lr);
            try self.applyAccumulatedBoundaryHeadStep(applied_boundary_head_lr);
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
            std.log.info("fused_chunker step={d} boundary_loss={d:.4} total_loss={d:.4} lr={d} boundary_head_lr={d}", .{ self.step_count, boundary_loss, total_loss, applied_lr, applied_boundary_head_lr });
        }

        return TrainStepSummary{
            .boundary_loss = boundary_loss,
            .boundary_ce_loss = boundary_ce_loss,
            .boundary_rank_loss = boundary_rank_loss,
            .boundary_local_window_loss = boundary_local_window_loss,
            .contrastive_loss = contrastive_result.contrastive_loss,
            .total_loss = total_loss,
            .boundary_grad_norm = boundary_grad_norm,
            .boundary_tp = 0,
            .boundary_fp = 0,
            .boundary_fn = 0,
            .step = self.step_count,
            .learning_rate = applied_lr,
            .boundary_head_learning_rate = applied_boundary_head_lr,
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
        input_ids: ?[]const i32,
        boundary_candidate_mask: ?[]const f32,
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
            input_ids,
            boundary_candidate_mask,
            total_tokens,
            true,
        );
        defer boundary_step.deinit(allocator);
        sanitizeBoundaryStepGradients(&boundary_step);

        const boundary_loss = boundary_step.boundary_loss;
        const boundary_ce_loss = boundary_step.boundary_ce_loss;
        const boundary_rank_loss = boundary_step.boundary_rank_loss;
        const boundary_local_window_loss = boundary_step.boundary_local_window_loss;
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
        var applied_boundary_head_lr: f32 = 0.0;
        if (self.accum_count >= accum_steps) {
            applied_lr = lr;
            applied_boundary_head_lr = self.config.boundaryHeadLearningRate(lr);
            try self.applyAccumulatedBoundaryHeadStep(applied_boundary_head_lr);
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
            std.log.info("fused_chunker step={d} boundary_loss={d:.4} total_loss={d:.4} lr={d} boundary_head_lr={d}", .{ self.step_count, boundary_loss, total_loss, applied_lr, applied_boundary_head_lr });
        }

        const summary = TrainStepSummary{
            .boundary_loss = boundary_loss,
            .boundary_ce_loss = boundary_ce_loss,
            .boundary_rank_loss = boundary_rank_loss,
            .boundary_local_window_loss = boundary_local_window_loss,
            .contrastive_loss = contrastive_result.contrastive_loss,
            .total_loss = total_loss,
            .boundary_grad_norm = boundary_grad_norm,
            .boundary_tp = 0,
            .boundary_fp = 0,
            .boundary_fn = 0,
            .step = self.step_count,
            .learning_rate = applied_lr,
            .boundary_head_learning_rate = applied_boundary_head_lr,
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
        var acc = BoundaryEvalAccumulator{
            .calibrated_threshold = fused_chunker_loss.weightedCePositiveThreshold(self.loss_config.pos_weight),
        };
        defer acc.deinit(allocator);

        for (features_list, labels_list, mask_list, total_tokens_list) |features, labels, mask, total| {
            try self.evaluateBatchInto(allocator, &acc, features, labels, mask, total);
        }

        return try acc.finish(allocator);
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
        const logits = try self.evaluateBoundaryLogitsOwnedWithMask(allocator, features, mask, total);
        defer allocator.free(logits);

        try acc.addLogitsBySample(allocator, logits, labels, mask, @intCast(self.config.max_seq_len));
    }

    pub fn evaluateBoundaryLogitsOwned(
        self: *FusedTrainer,
        allocator: std.mem.Allocator,
        features: []const f32,
        total: usize,
    ) ![]f32 {
        const synthetic_mask = try allocator.alloc(f32, total);
        defer allocator.free(synthetic_mask);
        @memset(synthetic_mask, 1.0);
        return try self.evaluateBoundaryLogitsOwnedWithMask(allocator, features, synthetic_mask, total);
    }

    pub fn evaluateBoundaryLogitsOwnedWithMask(
        self: *FusedTrainer,
        allocator: std.mem.Allocator,
        features: []const f32,
        mask: []const f32,
        total: usize,
    ) ![]f32 {
        const H: usize = @intCast(self.config.hidden_size);
        const head_features_owned = try transformBoundaryFeaturesForHead(
            allocator,
            self.config.boundary_feature_mode,
            features,
            mask,
            total,
            H,
            @intCast(self.config.max_seq_len),
        );
        defer if (head_features_owned) |owned| allocator.free(owned);
        const head_features = head_features_owned orelse features;
        return if (self.legacy_dense_boundary_head) |*legacy_head|
            try evaluateLegacyDenseBoundaryLogits(allocator, legacy_head, head_features, total)
        else
            try evaluateBoundaryLogitsSimple(
                allocator,
                &self.boundary_head,
                head_features,
                total,
            );
    }

    /// Run the boundary-head training substep and probability diagnostics
    /// without mutating trainer weights, optimizer state, or step counters.
    pub fn debugBoundaryStep(
        self: *FusedTrainer,
        allocator: std.mem.Allocator,
        features: []const f32,
        boundary_labels: []const f32,
        attention_mask: []const f32,
        input_ids: ?[]const i32,
        boundary_candidate_mask: ?[]const f32,
        total_tokens: usize,
        want_features_grad: bool,
    ) !BoundaryStepDebugSummary {
        self.sanitizeBoundaryHeadParameters();
        var boundary_step = try self.runBoundaryHeadTraining(
            allocator,
            features,
            boundary_labels,
            attention_mask,
            input_ids,
            boundary_candidate_mask,
            total_tokens,
            want_features_grad,
        );
        defer boundary_step.deinit(allocator);
        sanitizeBoundaryStepGradients(&boundary_step);

        const H: usize = @intCast(self.config.hidden_size);
        const head_features_owned = try transformBoundaryFeaturesForHead(
            allocator,
            self.config.boundary_feature_mode,
            features,
            attention_mask,
            total_tokens,
            H,
            @intCast(self.config.max_seq_len),
        );
        defer if (head_features_owned) |owned| allocator.free(owned);
        const head_features = head_features_owned orelse features;

        var forward_probe = BoundaryForwardProbe{};
        const logits = if (self.legacy_dense_boundary_head) |*legacy_head| blk: {
            const legacy_logits = try evaluateLegacyDenseBoundaryLogits(allocator, legacy_head, head_features, total_tokens);
            forward_probe.boundary_head_input = computeBoundaryProbeTensorStats(head_features);
            forward_probe.logits = computeBoundaryProbeTensorStats(legacy_logits);
            break :blk legacy_logits;
        } else blk: {
            const forward = try evaluateBoundaryForwardProbeSimple(allocator, &self.boundary_head, head_features, total_tokens);
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

    /// Advance the global step counter WITHOUT computing gradients or touching
    /// any trainable parameter or optimizer moment. Used by the late SPLADE
    /// head-only stage (Go Phase 32 parity), where the boundary head, dense
    /// path, and encoder LoRA adapters are frozen: because this function is the
    /// only trainer entry point the stage calls, boundary weights are
    /// structurally incapable of changing during head-only epochs.
    ///
    /// The returned summary carries the schedule learning rate for the step so
    /// downstream consumers (SPLADE AdamW, step metrics) see the same LR a
    /// normal step would have applied.
    pub fn frozenStepSummary(self: *FusedTrainer) TrainStepSummary {
        const lr = self.lr_schedule.lr(self.step_count);
        self.step_count += 1;
        self.optimizer_state.step_count = self.step_count;
        return TrainStepSummary{
            .step = self.step_count,
            .learning_rate = lr,
            .boundary_head_learning_rate = 0,
        };
    }

    /// Reset optimizer moments while keeping the current model weights
    /// unchanged. Mirrors Go FusedTrainer.ResetOptimizerState for the late
    /// SPLADE stage transition:
    ///   - AdamW: first/second moments zeroed (state map reinitialized; fresh
    ///     zeroed states are lazily recreated on the next update).
    ///   - Schedule-Free: z is reset to the current parameters and v/step are
    ///     zeroed so the stage restarts from the restored checkpoint.
    ///   - Pending gradient accumulation is discarded.
    /// The global step counter is intentionally preserved (Go keeps
    /// currentStep for the LR schedule).
    pub fn resetOptimizerStateForStage(self: *FusedTrainer) void {
        const saved_step_count = self.optimizer_state.step_count;
        self.optimizer_state.deinit();
        self.optimizer_state = optimizers.OptimizerState.init(self.allocator);
        self.optimizer_state.step_count = saved_step_count;

        if (self.sf_state_w1) |*state| resetScheduleFreeStateToWeights(state, self.boundary_head.w1);
        if (self.sf_state_b1) |*state| resetScheduleFreeStateToWeights(state, self.boundary_head.b1);
        if (self.sf_state_w2) |*state| resetScheduleFreeStateToWeights(state, self.boundary_head.w2);
        if (self.sf_state_b2) |*state| resetScheduleFreeStateToWeights(state, self.boundary_head.b2);

        self.resetBoundaryGradientAccum();
        self.accum_count = 0;
    }

    fn resetScheduleFreeStateToWeights(state: *ScheduleFreeAdamWState, weights: []const f32) void {
        @memcpy(state.z, weights);
        @memset(state.v, 0);
        state.step = 0;
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
    ce_loss: f32 = 0,
    rank_loss: f32 = 0,
    local_window_loss: f32 = 0,
    logit_grad: []f32,

    fn deinit(self: *BoundaryCeGradResult, allocator: std.mem.Allocator) void {
        allocator.free(self.logit_grad);
        self.* = undefined;
    }
};

fn computeBoundaryWeightedCeAndLogitGrad(
    allocator: std.mem.Allocator,
    logits: []const f32,
    targets: []const f32,
    mask: []const f32,
    total: usize,
    pos_weight: f32,
    rank_loss_weight: f32,
    rank_loss_margin: f32,
    rank_loss_top_k: u32,
    same_token_rank_loss_weight: f32,
    same_token_rank_loss_top_k: u32,
    same_token_negative_weight: f32,
    same_token_negative_top_k: u32,
    input_ids: ?[]const i32,
    max_seq_len: usize,
    gold_count_rank_loss_weight: f32,
    gold_count_rank_loss_margin: f32,
    gold_count_rank_loss_negative_multiplier: u32,
    local_window_loss_weight: f32,
    local_window_radius: u32,
) !BoundaryCeGradResult {
    return computeBoundaryWeightedCeAndLogitGradWithCandidateMask(
        allocator,
        logits,
        targets,
        mask,
        total,
        pos_weight,
        rank_loss_weight,
        rank_loss_margin,
        rank_loss_top_k,
        same_token_rank_loss_weight,
        same_token_rank_loss_top_k,
        same_token_negative_weight,
        same_token_negative_top_k,
        0.0,
        8,
        1.0,
        null,
        input_ids,
        max_seq_len,
        gold_count_rank_loss_weight,
        gold_count_rank_loss_margin,
        gold_count_rank_loss_negative_multiplier,
        local_window_loss_weight,
        local_window_radius,
    );
}

fn computeBoundaryWeightedCeAndLogitGradWithCandidateMask(
    allocator: std.mem.Allocator,
    logits: []const f32,
    targets: []const f32,
    mask: []const f32,
    total: usize,
    pos_weight: f32,
    rank_loss_weight: f32,
    rank_loss_margin: f32,
    rank_loss_top_k: u32,
    same_token_rank_loss_weight: f32,
    same_token_rank_loss_top_k: u32,
    same_token_negative_weight: f32,
    same_token_negative_top_k: u32,
    candidate_rank_loss_weight: f32,
    candidate_rank_loss_top_k: u32,
    candidate_negative_weight: f32,
    candidate_mask: ?[]const f32,
    input_ids: ?[]const i32,
    max_seq_len: usize,
    gold_count_rank_loss_weight: f32,
    gold_count_rank_loss_margin: f32,
    gold_count_rank_loss_negative_multiplier: u32,
    local_window_loss_weight: f32,
    local_window_radius: u32,
) !BoundaryCeGradResult {
    if (logits.len != total * 2) return error.UnexpectedOutputShape;
    if (targets.len != total * 2) return error.UnexpectedOutputShape;
    if (mask.len != total) return error.UnexpectedOutputShape;
    if (same_token_negative_weight > 1.0) {
        const ids = input_ids orelse return error.MissingBoundarySameTokenIds;
        if (ids.len != total) return error.UnexpectedOutputShape;
    }
    if (candidate_mask) |candidates| {
        if (candidates.len != total) return error.UnexpectedOutputShape;
    } else if (candidate_rank_loss_weight > 0.0 or candidate_negative_weight > 1.0) {
        return error.MissingBoundaryCandidateMask;
    }

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
        const same_token_negative_scale: f32 = if (same_token_negative_weight > 1.0 and
            t0 > 0.5 and
            isSameTokenBoundaryNegative(logits, targets, mask, input_ids.?, total, i, same_token_negative_top_k))
            same_token_negative_weight
        else
            1.0;
        const candidate_negative_scale: f32 = if (candidate_negative_weight > 1.0 and
            t0 > 0.5 and
            candidate_mask.?[i] > 0.5)
            candidate_negative_weight
        else
            1.0;
        const wt0 = t0 * @max(same_token_negative_scale, candidate_negative_scale);
        const wt1 = t1 * pos_weight;
        const wt_sum = wt0 + wt1;
        const m = mask[i];

        numerator += @as(f64, m * -(wt0 * logp0 + wt1 * logp1));
        const row_scale = m / denom;
        grad[i * 2] = row_scale * (p0 * wt_sum - wt0);
        grad[i * 2 + 1] = row_scale * (p1 * wt_sum - wt1);
    }

    const ce_loss = numerator / @as(f64, denom);
    const rank_loss = try addBoundaryPairwiseRankLossAndGrad(
        allocator,
        logits,
        targets,
        mask,
        null,
        false,
        null,
        false,
        total,
        rank_loss_weight,
        rank_loss_margin,
        rank_loss_top_k,
        grad,
    );
    const same_token_rank_loss = try addBoundaryPairwiseRankLossAndGrad(
        allocator,
        logits,
        targets,
        mask,
        input_ids,
        true,
        null,
        false,
        total,
        same_token_rank_loss_weight,
        rank_loss_margin,
        same_token_rank_loss_top_k,
        grad,
    );
    const candidate_rank_loss = try addBoundaryPairwiseRankLossAndGrad(
        allocator,
        logits,
        targets,
        mask,
        null,
        false,
        candidate_mask,
        true,
        total,
        candidate_rank_loss_weight,
        rank_loss_margin,
        candidate_rank_loss_top_k,
        grad,
    );
    const gold_count_rank_loss = try addBoundaryGoldCountRankLossAndGrad(
        allocator,
        logits,
        targets,
        mask,
        total,
        max_seq_len,
        gold_count_rank_loss_weight,
        gold_count_rank_loss_margin,
        gold_count_rank_loss_negative_multiplier,
        grad,
    );
    const local_window_loss = addBoundaryLocalWindowLossAndGrad(
        logits,
        targets,
        mask,
        total,
        max_seq_len,
        local_window_loss_weight,
        local_window_radius,
        grad,
    );
    const total_rank_loss = rank_loss + same_token_rank_loss + candidate_rank_loss + gold_count_rank_loss;
    const loss = ce_loss + total_rank_loss + local_window_loss;

    return .{
        .loss = @floatCast(loss),
        .ce_loss = @floatCast(ce_loss),
        .rank_loss = @floatCast(total_rank_loss),
        .local_window_loss = @floatCast(local_window_loss),
        .logit_grad = grad,
    };
}

fn isSameTokenBoundaryNegative(
    logits: []const f32,
    targets: []const f32,
    mask: []const f32,
    token_ids: []const i32,
    total: usize,
    neg_i: usize,
    top_k_u32: u32,
) bool {
    if (neg_i >= total) return false;
    if (mask[neg_i] <= 0.5 or targets[neg_i * 2 + 1] > 0.5) return false;
    const token_id = token_ids[neg_i];
    const top_k: usize = @intCast(top_k_u32);
    const neg_margin = logits[neg_i * 2 + 1] - logits[neg_i * 2];
    for (0..total) |pos_i| {
        if (pos_i == neg_i) continue;
        if (mask[pos_i] <= 0.5 or targets[pos_i * 2 + 1] <= 0.5) continue;
        if (token_ids[pos_i] != token_id) continue;
        if (top_k == 0) return true;

        var harder_same_token_negatives: usize = 0;
        for (0..total) |other_i| {
            if (other_i == neg_i) continue;
            if (mask[other_i] <= 0.5 or targets[other_i * 2 + 1] > 0.5) continue;
            if (token_ids[other_i] != token_id) continue;
            const other_margin = logits[other_i * 2 + 1] - logits[other_i * 2];
            if (other_margin > neg_margin) harder_same_token_negatives += 1;
        }
        if (harder_same_token_negatives < top_k) return true;
    }
    return false;
}

fn addBoundaryPairwiseRankLossAndGrad(
    allocator: std.mem.Allocator,
    logits: []const f32,
    targets: []const f32,
    mask: []const f32,
    token_ids: ?[]const i32,
    require_same_token: bool,
    candidate_mask: ?[]const f32,
    require_candidate_negative: bool,
    total: usize,
    rank_loss_weight: f32,
    rank_loss_margin: f32,
    rank_loss_top_k: u32,
    grad: []f32,
) !f64 {
    if (rank_loss_weight <= 0.0) return 0.0;
    if (rank_loss_margin <= 0.0) return 0.0;
    if (require_same_token) {
        const ids = token_ids orelse return error.MissingBoundarySameTokenIds;
        if (ids.len != total) return error.UnexpectedOutputShape;
    }
    if (require_candidate_negative) {
        const candidates = candidate_mask orelse return error.MissingBoundaryCandidateMask;
        if (candidates.len != total) return error.UnexpectedOutputShape;
    }

    var positive_count: usize = 0;
    var negative_count: usize = 0;
    for (0..total) |i| {
        if (mask[i] <= 0.5) continue;
        if (targets[i * 2 + 1] > 0.5) {
            positive_count += 1;
        } else {
            if (require_candidate_negative and candidate_mask.?[i] <= 0.5) continue;
            negative_count += 1;
        }
    }
    if (positive_count == 0) return 0.0;
    if (negative_count == 0) return 0.0;

    const requested_top_k = if (rank_loss_top_k == 0) 1 else @as(usize, @intCast(rank_loss_top_k));
    const effective_top_k = @min(requested_top_k, negative_count);
    const top_neg_indices = try allocator.alloc(usize, effective_top_k);
    defer allocator.free(top_neg_indices);
    const top_violations = try allocator.alloc(f32, effective_top_k);
    defer allocator.free(top_violations);

    const inv_positive_count = 1.0 / @as(f32, @floatFromInt(positive_count));
    var loss: f64 = 0.0;

    for (0..total) |pos_i| {
        if (mask[pos_i] <= 0.5 or targets[pos_i * 2 + 1] <= 0.5) continue;
        const pos_margin = logits[pos_i * 2 + 1] - logits[pos_i * 2];
        var selected_count: usize = 0;

        for (0..total) |neg_i| {
            if (mask[neg_i] <= 0.5 or targets[neg_i * 2 + 1] > 0.5) continue;
            if (require_same_token and token_ids.?[neg_i] != token_ids.?[pos_i]) continue;
            if (require_candidate_negative and candidate_mask.?[neg_i] <= 0.5) continue;
            const neg_margin = logits[neg_i * 2 + 1] - logits[neg_i * 2];
            const violation = rank_loss_margin + neg_margin - pos_margin;
            if (violation <= 0.0) continue;

            if (selected_count < effective_top_k) {
                var insert_at = selected_count;
                selected_count += 1;
                while (insert_at > 0 and violation > top_violations[insert_at - 1]) : (insert_at -= 1) {
                    top_violations[insert_at] = top_violations[insert_at - 1];
                    top_neg_indices[insert_at] = top_neg_indices[insert_at - 1];
                }
                top_violations[insert_at] = violation;
                top_neg_indices[insert_at] = neg_i;
            } else if (violation > top_violations[selected_count - 1]) {
                var insert_at = selected_count - 1;
                while (insert_at > 0 and violation > top_violations[insert_at - 1]) : (insert_at -= 1) {
                    top_violations[insert_at] = top_violations[insert_at - 1];
                    top_neg_indices[insert_at] = top_neg_indices[insert_at - 1];
                }
                top_violations[insert_at] = violation;
                top_neg_indices[insert_at] = neg_i;
            }
        }

        if (selected_count == 0) continue;
        const grad_scale = rank_loss_weight * inv_positive_count / @as(f32, @floatFromInt(selected_count));

        // L = margin + neg_margin - pos_margin, where margin_i = logit1 - logit0.
        // Gradient descent therefore increases positive margins and decreases
        // the selected hard-negative margins.
        for (0..selected_count) |rank_i| {
            const neg_i = top_neg_indices[rank_i];
            loss += @as(f64, grad_scale * top_violations[rank_i]);
            grad[pos_i * 2] += grad_scale;
            grad[pos_i * 2 + 1] -= grad_scale;
            grad[neg_i * 2] -= grad_scale;
            grad[neg_i * 2 + 1] += grad_scale;
        }
    }

    return loss;
}

fn addBoundaryGoldCountRankLossAndGrad(
    allocator: std.mem.Allocator,
    logits: []const f32,
    targets: []const f32,
    mask: []const f32,
    total: usize,
    max_seq_len: usize,
    loss_weight: f32,
    margin: f32,
    negative_multiplier_u32: u32,
    grad: []f32,
) !f64 {
    if (loss_weight <= 0.0) return 0.0;
    if (margin <= 0.0) return 0.0;
    if (max_seq_len == 0) return error.InvalidMaxSeqLen;
    if (logits.len != total * 2 or targets.len != total * 2 or mask.len != total or grad.len != total * 2) return error.UnexpectedOutputShape;

    var loss: f64 = 0.0;
    var seq_start: usize = 0;
    while (seq_start < total) : (seq_start += max_seq_len) {
        const seq_end = @min(seq_start + max_seq_len, total);
        var positive_count: usize = 0;
        var negative_count: usize = 0;
        for (seq_start..seq_end) |i| {
            if (mask[i] <= 0.5) continue;
            if (targets[i * 2 + 1] > 0.5) {
                positive_count += 1;
            } else {
                negative_count += 1;
            }
        }
        if (positive_count == 0 or negative_count == 0) continue;

        const negative_multiplier = @max(@as(usize, @intCast(negative_multiplier_u32)), 1);
        const requested_top_k = if (positive_count > std.math.maxInt(usize) / negative_multiplier)
            negative_count
        else
            positive_count * negative_multiplier;
        const effective_top_k = @min(requested_top_k, negative_count);
        if (effective_top_k == 0) continue;

        const top_neg_indices = try allocator.alloc(usize, effective_top_k);
        defer allocator.free(top_neg_indices);
        const top_neg_margins = try allocator.alloc(f32, effective_top_k);
        defer allocator.free(top_neg_margins);

        var selected_count: usize = 0;
        for (seq_start..seq_end) |neg_i| {
            if (mask[neg_i] <= 0.5 or targets[neg_i * 2 + 1] > 0.5) continue;
            const neg_margin = logits[neg_i * 2 + 1] - logits[neg_i * 2];
            if (selected_count < effective_top_k) {
                var insert_at = selected_count;
                selected_count += 1;
                while (insert_at > 0 and neg_margin > top_neg_margins[insert_at - 1]) : (insert_at -= 1) {
                    top_neg_margins[insert_at] = top_neg_margins[insert_at - 1];
                    top_neg_indices[insert_at] = top_neg_indices[insert_at - 1];
                }
                top_neg_margins[insert_at] = neg_margin;
                top_neg_indices[insert_at] = neg_i;
            } else if (neg_margin > top_neg_margins[selected_count - 1]) {
                var insert_at = selected_count - 1;
                while (insert_at > 0 and neg_margin > top_neg_margins[insert_at - 1]) : (insert_at -= 1) {
                    top_neg_margins[insert_at] = top_neg_margins[insert_at - 1];
                    top_neg_indices[insert_at] = top_neg_indices[insert_at - 1];
                }
                top_neg_margins[insert_at] = neg_margin;
                top_neg_indices[insert_at] = neg_i;
            }
        }
        if (selected_count == 0) continue;

        const pair_count = positive_count * selected_count;
        const grad_scale = loss_weight / @as(f32, @floatFromInt(pair_count));
        for (seq_start..seq_end) |pos_i| {
            if (mask[pos_i] <= 0.5 or targets[pos_i * 2 + 1] <= 0.5) continue;
            const pos_margin = logits[pos_i * 2 + 1] - logits[pos_i * 2];
            for (0..selected_count) |rank_i| {
                const neg_i = top_neg_indices[rank_i];
                const violation = margin + top_neg_margins[rank_i] - pos_margin;
                if (violation <= 0.0) continue;
                loss += @as(f64, grad_scale * violation);
                grad[pos_i * 2] += grad_scale;
                grad[pos_i * 2 + 1] -= grad_scale;
                grad[neg_i * 2] -= grad_scale;
                grad[neg_i * 2 + 1] += grad_scale;
            }
        }
    }

    return loss;
}

fn addBoundaryLocalWindowLossAndGrad(
    logits: []const f32,
    targets: []const f32,
    mask: []const f32,
    total: usize,
    max_seq_len: usize,
    loss_weight: f32,
    radius_u32: u32,
    grad: []f32,
) f64 {
    if (loss_weight <= 0.0) return 0.0;
    if (radius_u32 == 0 or max_seq_len == 0) return 0.0;

    var positive_count: usize = 0;
    for (0..total) |i| {
        if (mask[i] <= 0.5) continue;
        if (targets[i * 2 + 1] > 0.5) positive_count += 1;
    }
    if (positive_count == 0) return 0.0;

    const radius: usize = @intCast(radius_u32);
    const grad_scale = loss_weight / @as(f32, @floatFromInt(positive_count));
    var loss: f64 = 0.0;

    for (0..total) |pos_i| {
        if (mask[pos_i] <= 0.5 or targets[pos_i * 2 + 1] <= 0.5) continue;

        const sample_start = (pos_i / max_seq_len) * max_seq_len;
        const sample_end = @min(sample_start + max_seq_len, total);
        const window_start = if (pos_i > sample_start + radius) pos_i - radius else sample_start;
        const window_end = @min(pos_i + radius + 1, sample_end);

        var max_margin = -std.math.inf(f32);
        var candidate_count: usize = 0;
        for (window_start..window_end) |j| {
            if (mask[j] <= 0.5) continue;
            if (j != pos_i and targets[j * 2 + 1] > 0.5) continue;
            const margin = logits[j * 2 + 1] - logits[j * 2];
            max_margin = @max(max_margin, margin);
            candidate_count += 1;
        }
        if (candidate_count <= 1) continue;

        var denom: f64 = 0;
        var pos_exp: f64 = 0;
        for (window_start..window_end) |j| {
            if (mask[j] <= 0.5) continue;
            if (j != pos_i and targets[j * 2 + 1] > 0.5) continue;
            const margin = logits[j * 2 + 1] - logits[j * 2];
            const e = @exp(@as(f64, margin - max_margin));
            denom += e;
            if (j == pos_i) pos_exp = e;
        }
        if (denom <= 0.0 or pos_exp <= 0.0) continue;

        loss += @as(f64, loss_weight) * -@log(pos_exp / denom) / @as(f64, @floatFromInt(positive_count));
        const inv_denom = 1.0 / denom;
        for (window_start..window_end) |j| {
            if (mask[j] <= 0.5) continue;
            if (j != pos_i and targets[j * 2 + 1] > 0.5) continue;
            const margin = logits[j * 2 + 1] - logits[j * 2];
            const prob: f32 = @floatCast(@exp(@as(f64, margin - max_margin)) * inv_denom);
            const target: f32 = if (j == pos_i) 1.0 else 0.0;
            const d_margin = grad_scale * (prob - target);
            grad[j * 2] -= d_margin;
            grad[j * 2 + 1] += d_margin;
        }
    }

    return loss;
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

fn meanBoundaryProbabilityForLabel(
    logits: []const f32,
    labels: []const f32,
    mask: []const f32,
    want_positive: bool,
) f32 {
    std.debug.assert(labels.len * 2 == logits.len);
    std.debug.assert(mask.len == labels.len);

    var sum: f64 = 0.0;
    var count: usize = 0;
    for (labels, 0..) |label, i| {
        if (mask[i] <= 0.5) continue;
        if ((label > 0.5) != want_positive) continue;
        sum += @floatCast(fused_chunker_loss.positiveBoundaryProbability(logits[i * 2], logits[i * 2 + 1]));
        count += 1;
    }
    if (count == 0) return 0.0;
    return @floatCast(sum / @as(f64, @floatFromInt(count)));
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

test "boundary feature transforms use local previous-token context" {
    const allocator = std.testing.allocator;
    const features = [_]f32{ 1.0, 3.0, 10.0 };
    const mask = [_]f32{ 1.0, 1.0, 1.0 };

    try std.testing.expectEqual(@as(usize, 3), boundaryFeatureMultiplier(.prev_current_diff_concat));
    try std.testing.expectEqual(@as(usize, 3), boundaryFeatureDim(.prev_current_diff_concat, 1));
    try std.testing.expectEqual(@as(usize, 5), boundaryFeatureMultiplier(.prev_current_next_diff_concat));
    try std.testing.expectEqual(@as(usize, 5), boundaryFeatureDim(.prev_current_next_diff_concat, 1));
    try std.testing.expectEqual(@as(usize, 5), boundaryFeatureMultiplier(.window_context_diff));
    try std.testing.expectEqual(@as(usize, 5), boundaryFeatureDim(.window_context_diff, 1));

    const prev_diff = (try transformBoundaryFeaturesForHead(
        allocator,
        .prev_diff,
        &features,
        &mask,
        3,
        1,
        3,
    )).?;
    defer allocator.free(prev_diff);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0, 7.0 }, prev_diff);

    const prev_current_diff = (try transformBoundaryFeaturesForHead(
        allocator,
        .prev_current_diff,
        &features,
        &mask,
        3,
        1,
        3,
    )).?;
    defer allocator.free(prev_current_diff);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 5.0, 17.0 }, prev_current_diff);

    const prev_current_diff_concat = (try transformBoundaryFeaturesForHead(
        allocator,
        .prev_current_diff_concat,
        &features,
        &mask,
        3,
        1,
        3,
    )).?;
    defer allocator.free(prev_current_diff_concat);
    try std.testing.expectEqualSlices(f32, &[_]f32{
        0.0, 1.0,  1.0,
        1.0, 3.0,  2.0,
        3.0, 10.0, 7.0,
    }, prev_current_diff_concat);

    const prev_current_next_diff_concat = (try transformBoundaryFeaturesForHead(
        allocator,
        .prev_current_next_diff_concat,
        &features,
        &mask,
        3,
        1,
        3,
    )).?;
    defer allocator.free(prev_current_next_diff_concat);
    try std.testing.expectEqualSlices(f32, &[_]f32{
        0.0, 1.0,  3.0,  1.0, -2.0,
        1.0, 3.0,  10.0, 2.0, -7.0,
        3.0, 10.0, 0.0,  7.0, 10.0,
    }, prev_current_next_diff_concat);

    const window_context_diff = (try transformBoundaryFeaturesForHead(
        allocator,
        .window_context_diff,
        &features,
        &mask,
        3,
        1,
        3,
    )).?;
    defer allocator.free(window_context_diff);
    try expectApproxEqSlices(&[_]f32{
        0.0, 1.0,  6.5,  1.0, -5.5,
        1.0, 3.0,  10.0, 2.0, -7.0,
        2.0, 10.0, 0.0,  8.0, 10.0,
    }, window_context_diff, 1e-6);
}

test "boundary feature gradient scatter maps local context back to encoder tokens" {
    const allocator = std.testing.allocator;
    const grad = [_]f32{ 1.0, 2.0, 3.0 };
    const mask = [_]f32{ 1.0, 1.0, 1.0 };

    const prev_diff = (try scatterBoundaryFeatureGradToEncoder(
        allocator,
        .prev_diff,
        &grad,
        &mask,
        3,
        1,
        3,
    )).?;
    defer allocator.free(prev_diff);
    try std.testing.expectEqualSlices(f32, &[_]f32{ -1.0, -1.0, 3.0 }, prev_diff);

    const prev_current_diff = (try scatterBoundaryFeatureGradToEncoder(
        allocator,
        .prev_current_diff,
        &grad,
        &mask,
        3,
        1,
        3,
    )).?;
    defer allocator.free(prev_current_diff);
    try std.testing.expectEqualSlices(f32, &[_]f32{ -1.0, 1.0, 6.0 }, prev_current_diff);

    const prev_current_diff_concat = (try scatterBoundaryFeatureGradToEncoder(
        allocator,
        .prev_current_diff_concat,
        &[_]f32{
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        },
        &mask,
        3,
        1,
        3,
    )).?;
    defer allocator.free(prev_current_diff_concat);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 3.0, 9.0, 17.0 }, prev_current_diff_concat);

    const prev_current_next_diff_concat = (try scatterBoundaryFeatureGradToEncoder(
        allocator,
        .prev_current_next_diff_concat,
        &[_]f32{
            1.0,  2.0,  3.0,  4.0,  5.0,
            6.0,  7.0,  8.0,  9.0,  10.0,
            11.0, 12.0, 13.0, 14.0, 15.0,
        },
        &mask,
        3,
        1,
        3,
    )).?;
    defer allocator.free(prev_current_next_diff_concat);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 8.0, 21.0, 39.0 }, prev_current_next_diff_concat);

    const window_context_diff = (try scatterBoundaryFeatureGradToEncoder(
        allocator,
        .window_context_diff,
        &[_]f32{
            1.0,  2.0,  3.0,  4.0,  5.0,
            6.0,  7.0,  8.0,  9.0,  10.0,
            11.0, 12.0, 13.0, 14.0, 15.0,
        },
        &mask,
        3,
        1,
        3,
    )).?;
    defer allocator.free(window_context_diff);
    try expectApproxEqSlices(&[_]f32{ 6.5, 23.5, 38.0 }, window_context_diff, 1e-6);
}

test "boundary feature transforms do not cross max sequence boundaries" {
    const allocator = std.testing.allocator;
    const features = [_]f32{ 1.0, 3.0, 10.0, 15.0 };
    const mask = [_]f32{ 1.0, 1.0, 1.0, 1.0 };

    const prev_diff = (try transformBoundaryFeaturesForHead(
        allocator,
        .prev_diff,
        &features,
        &mask,
        4,
        1,
        2,
    )).?;
    defer allocator.free(prev_diff);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0, 10.0, 5.0 }, prev_diff);

    const scattered = (try scatterBoundaryFeatureGradToEncoder(
        allocator,
        .prev_diff,
        &[_]f32{ 1.0, 2.0, 3.0, 4.0 },
        &mask,
        4,
        1,
        2,
    )).?;
    defer allocator.free(scattered);
    try std.testing.expectEqualSlices(f32, &[_]f32{ -1.0, 2.0, -1.0, 4.0 }, scattered);

    const window_context_diff = (try transformBoundaryFeaturesForHead(
        allocator,
        .window_context_diff,
        &features,
        &mask,
        4,
        1,
        2,
    )).?;
    defer allocator.free(window_context_diff);
    try expectApproxEqSlices(&[_]f32{
        0.0,  1.0,  3.0,  1.0,  -2.0,
        1.0,  3.0,  0.0,  2.0,  3.0,
        0.0,  10.0, 15.0, 10.0, -5.0,
        10.0, 15.0, 0.0,  5.0,  15.0,
    }, window_context_diff, 1e-6);
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

test "FusedTrainingConfig boundary head lr multiplier preserves base schedule" {
    const config = FusedTrainingConfig{
        .learning_rate = 2e-5,
        .boundary_head_lr_multiplier = 25.0,
        .warmup_steps = 20,
        .total_steps = 1000,
    };

    const base_lr = config.lrSchedule().lr(20);
    try std.testing.expectApproxEqAbs(@as(f32, 2e-5), base_lr, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f32, 5e-4), config.boundaryHeadLearningRate(base_lr), 1e-8);
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
    defer acc.deinit(allocator);
    try acc.addLogits(allocator, &logits, &labels, &mask);
    const summary = try acc.finish(allocator);

    try std.testing.expectEqual(@as(u64, 3), summary.valid_tokens);
    try std.testing.expectEqual(@as(u64, 2), summary.gold_positives);
    try std.testing.expectEqual(@as(u64, 1), summary.predicted_positives);
    try std.testing.expectEqual(@as(u64, 2), summary.best_predicted_positives);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), summary.gold_positive_rate, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), summary.predicted_positive_rate, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), summary.best_predicted_positive_rate, 1e-6);
    try std.testing.expect(!summary.calibrated_threshold_available);
    try std.testing.expect(!summary.fitted_threshold_available);

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
    try std.testing.expectEqual(@as(f32, 0.5), summary.threshold_points[8].threshold);
    try std.testing.expectEqual(@as(f32, 0.99), summary.threshold_points[summary.threshold_points.len - 1].threshold);
    try std.testing.expectEqual(@as(u64, 1), summary.probability_histogram_gold_positive[5]);
    try std.testing.expectEqual(@as(u64, 1), summary.probability_histogram_gold_positive[7]);
    try std.testing.expectEqual(@as(u64, 1), summary.probability_histogram_gold_negative[2]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), summary.average_precision, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), summary.precision_at_gold_count, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), summary.recall_at_gold_count, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), summary.f1_at_gold_count, 1e-6);
    try std.testing.expectApproxEqAbs(p2, summary.threshold_at_gold_count, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), summary.predicted_positive_rate_at_gold_count, 1e-6);
    try std.testing.expectEqual(@as(u64, 2), summary.gold_count_tp);
    try std.testing.expectEqual(@as(u64, 0), summary.gold_count_fp);
    try std.testing.expectEqual(@as(u64, 0), summary.gold_count_fn);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), summary.max_rank_f1, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), summary.max_rank_precision, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), summary.max_rank_recall, 1e-6);
    try std.testing.expectApproxEqAbs(p2, summary.max_rank_threshold, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), summary.max_rank_predicted_positive_rate, 1e-6);
    try std.testing.expectEqual(@as(u64, 2), summary.max_rank_tp);
    try std.testing.expectEqual(@as(u64, 0), summary.max_rank_fp);
    try std.testing.expectEqual(@as(u64, 0), summary.max_rank_fn);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), summary.gold_positive_mean_rank, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), summary.gold_positive_mean_rank_percentile, 1e-6);
    try std.testing.expectEqual(@as(u64, 1), summary.gold_positive_median_rank);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), summary.gold_positive_median_rank_percentile, 1e-6);
    try std.testing.expectEqual(@as(u64, 2), summary.gold_positive_p90_rank);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), summary.gold_positive_p90_rank_percentile, 1e-6);
    try std.testing.expectEqual(@as(u64, 2), summary.gold_positive_p99_rank);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), summary.gold_positive_p99_rank_percentile, 1e-6);
    try std.testing.expectEqual(@as(u64, 2), summary.gold_positive_worst_rank);
    try std.testing.expectEqual(@as(u64, 2), summary.gold_positive_top_5x_count);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), summary.gold_positive_top_5x_recall, 1e-6);
    try std.testing.expectEqual(@as(u64, 2), summary.gold_positive_top_10x_count);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), summary.gold_positive_top_10x_recall, 1e-6);
    try std.testing.expectEqual(@as(u64, 0), summary.sample_oracle_count_samples);
}

test "BoundaryEvalAccumulator reports sample-local oracle-count decoder metrics" {
    const allocator = std.testing.allocator;

    const logits = [_]f32{
        0.0, 3.0, // false positive cluster head
        0.0, 2.0, // false positive suppressed by NMS radius 1
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.5, // gold, below the false-positive cluster
        0.0, 0.0,
        0.0, 1.0, // gold, selected by NMS after suppressing token 1
    };
    const labels = [_]f32{
        1.0, 0.0,
        1.0, 0.0,
        1.0, 0.0,
        1.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        0.0, 1.0,
    };
    const mask = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

    var acc = BoundaryEvalAccumulator{
        .sample_oracle_count_nms_radius = 1,
        .sample_oracle_count_length_window_min_radius = 1,
        .sample_oracle_count_length_window_radius_fraction = 0.0,
    };
    defer acc.deinit(allocator);
    try acc.addLogitsBySample(allocator, &logits, &labels, &mask, 8);
    const summary = try acc.finish(allocator);

    try std.testing.expectEqual(@as(u64, 1), summary.sample_oracle_count_samples);
    try std.testing.expectEqual(@as(u64, 0), summary.sample_oracle_count_topk_tp);
    try std.testing.expectEqual(@as(u64, 2), summary.sample_oracle_count_topk_fp);
    try std.testing.expectEqual(@as(u64, 2), summary.sample_oracle_count_topk_fn);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), summary.sample_oracle_count_topk_f1, 1e-6);
    try std.testing.expectEqual(@as(u64, 1), summary.sample_oracle_count_nms_tp);
    try std.testing.expectEqual(@as(u64, 1), summary.sample_oracle_count_nms_fp);
    try std.testing.expectEqual(@as(u64, 1), summary.sample_oracle_count_nms_fn);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), summary.sample_oracle_count_nms_f1, 1e-6);
    try std.testing.expectEqual(@as(u32, 1), summary.sample_oracle_count_nms_radius);
    try std.testing.expectEqual(@as(u64, 1), summary.sample_oracle_count_length_window_tp);
    try std.testing.expectEqual(@as(u64, 1), summary.sample_oracle_count_length_window_fp);
    try std.testing.expectEqual(@as(u64, 1), summary.sample_oracle_count_length_window_fn);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), summary.sample_oracle_count_length_window_f1, 1e-6);
    try std.testing.expectEqual(@as(u32, 1), summary.sample_oracle_count_length_window_min_radius);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), summary.sample_oracle_count_length_window_radius_fraction, 1e-6);
}

test "BoundaryEvalAccumulator reports calibrated threshold metrics" {
    const allocator = std.testing.allocator;

    const logits = [_]f32{
        0.0, 2.0,
        0.0, 1.0,
        0.0, -1.0,
    };
    const labels = [_]f32{
        0.0, 1.0,
        1.0, 0.0,
        0.0, 1.0,
    };
    const mask = [_]f32{ 1.0, 1.0, 1.0 };

    var acc = BoundaryEvalAccumulator{
        .calibrated_threshold = 0.8,
    };
    defer acc.deinit(allocator);
    try acc.addLogits(allocator, &logits, &labels, &mask);
    const summary = try acc.finish(allocator);

    try std.testing.expect(summary.calibrated_threshold_available);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), summary.calibrated_boundary_threshold, 1e-6);
    try std.testing.expectEqual(@as(u64, 1), summary.calibrated_boundary_tp);
    try std.testing.expectEqual(@as(u64, 0), summary.calibrated_boundary_fp);
    try std.testing.expectEqual(@as(u64, 1), summary.calibrated_boundary_fn);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), summary.calibrated_boundary_precision, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), summary.calibrated_boundary_recall, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), summary.calibrated_boundary_f1, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), summary.calibrated_predicted_positive_rate, 1e-6);
}

test "BoundaryEvalAccumulator reports fitted threshold metrics" {
    const allocator = std.testing.allocator;

    const logits = [_]f32{
        0.0, 2.0,
        0.0, 1.0,
        0.0, -1.0,
    };
    const labels = [_]f32{
        0.0, 1.0,
        1.0, 0.0,
        0.0, 1.0,
    };
    const mask = [_]f32{ 1.0, 1.0, 1.0 };

    var acc = BoundaryEvalAccumulator{
        .calibrated_threshold = 0.8,
        .fitted_threshold = 0.3,
    };
    defer acc.deinit(allocator);
    try acc.addLogits(allocator, &logits, &labels, &mask);
    const summary = try acc.finish(allocator);

    try std.testing.expect(summary.fitted_threshold_available);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), summary.fitted_boundary_threshold, 1e-6);
    try std.testing.expectEqual(@as(u64, 2), summary.fitted_boundary_tp);
    try std.testing.expectEqual(@as(u64, 1), summary.fitted_boundary_fp);
    try std.testing.expectEqual(@as(u64, 0), summary.fitted_boundary_fn);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), summary.fitted_boundary_precision, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), summary.fitted_boundary_recall, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), summary.fitted_boundary_f1, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), summary.fitted_predicted_positive_rate, 1e-6);
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
        .boundary_head_lr_multiplier = 3.0,
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

    const summary = try trainer.trainStep(
        allocator,
        &features,
        &labels,
        &attention_mask,
        null,
        null,
        &chunk_embeddings,
        &chunk_mask,
        &doc_ids,
        2,
        1,
        1,
        2,
    );

    try std.testing.expectApproxEqAbs(@as(f32, 0.01), summary.learning_rate, 1e-8);
    try std.testing.expectApproxEqAbs(@as(f32, 0.03), summary.boundary_head_learning_rate, 1e-8);
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
        null,
        null,
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

test "boundary rank loss pushes gold margins above hard negatives" {
    const allocator = std.testing.allocator;
    const logits = [_]f32{
        0.0, 0.0, // gold positive margin 0
        0.0, 2.0, // hard negative margin 2
    };
    const targets = [_]f32{
        0.0, 1.0,
        1.0, 0.0,
    };
    const mask = [_]f32{ 1.0, 1.0 };

    var base = try computeBoundaryWeightedCeAndLogitGrad(
        allocator,
        &logits,
        &targets,
        &mask,
        2,
        1.0,
        0.0,
        1.0,
        1,
        0.0,
        4,
        1.0,
        0,
        null,
        2,
        0.0,
        1.0,
        1,
        0.0,
        1,
    );
    defer base.deinit(allocator);

    var ranked = try computeBoundaryWeightedCeAndLogitGrad(
        allocator,
        &logits,
        &targets,
        &mask,
        2,
        1.0,
        0.5,
        1.0,
        1,
        0.0,
        4,
        1.0,
        0,
        null,
        2,
        0.0,
        1.0,
        1,
        0.0,
        1,
    );
    defer ranked.deinit(allocator);

    try std.testing.expectApproxEqAbs(base.loss, base.ce_loss, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), base.rank_loss, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), base.local_window_loss, 1e-6);
    try std.testing.expectApproxEqAbs(base.ce_loss, ranked.ce_loss, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), ranked.rank_loss, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ranked.local_window_loss, 1e-6);
    try std.testing.expectApproxEqAbs(base.loss + 1.5, ranked.loss, 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[0] + 0.5, ranked.logit_grad[0], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[1] - 0.5, ranked.logit_grad[1], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[2] - 0.5, ranked.logit_grad[2], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[3] + 0.5, ranked.logit_grad[3], 1e-6);
}

test "boundary rank loss can average top-k hard negatives" {
    const allocator = std.testing.allocator;
    const logits = [_]f32{
        0.0, 0.0, // gold positive margin 0
        0.0, 2.0, // hard negative margin 2 -> violation 3
        0.0, 1.0, // hard negative margin 1 -> violation 2
        0.0, -1.0, // easy negative margin -1 -> violation 0
    };
    const targets = [_]f32{
        0.0, 1.0,
        1.0, 0.0,
        1.0, 0.0,
        1.0, 0.0,
    };
    const mask = [_]f32{ 1.0, 1.0, 1.0, 1.0 };

    var base = try computeBoundaryWeightedCeAndLogitGrad(
        allocator,
        &logits,
        &targets,
        &mask,
        4,
        1.0,
        0.0,
        1.0,
        1,
        0.0,
        4,
        1.0,
        0,
        null,
        4,
        0.0,
        1.0,
        1,
        0.0,
        1,
    );
    defer base.deinit(allocator);

    var ranked = try computeBoundaryWeightedCeAndLogitGrad(
        allocator,
        &logits,
        &targets,
        &mask,
        4,
        1.0,
        0.6,
        1.0,
        2,
        0.0,
        4,
        1.0,
        0,
        null,
        4,
        0.0,
        1.0,
        1,
        0.0,
        1,
    );
    defer ranked.deinit(allocator);

    try std.testing.expectApproxEqAbs(base.ce_loss, ranked.ce_loss, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), ranked.rank_loss, 1e-6);
    try std.testing.expectApproxEqAbs(base.loss + 1.5, ranked.loss, 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[0] + 0.6, ranked.logit_grad[0], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[1] - 0.6, ranked.logit_grad[1], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[2] - 0.3, ranked.logit_grad[2], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[3] + 0.3, ranked.logit_grad[3], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[4] - 0.3, ranked.logit_grad[4], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[5] + 0.3, ranked.logit_grad[5], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[6], ranked.logit_grad[6], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[7], ranked.logit_grad[7], 1e-6);
}

test "boundary same-token rank loss targets surface-confusable negatives" {
    const allocator = std.testing.allocator;
    const logits = [_]f32{
        0.0, 0.0, // gold positive token 380 margin 0
        0.0, 2.0, // same-token hard negative margin 2 -> selected
        0.0, 3.0, // different-token harder negative -> ignored by same-token term
    };
    const targets = [_]f32{
        0.0, 1.0,
        1.0, 0.0,
        1.0, 0.0,
    };
    const mask = [_]f32{ 1.0, 1.0, 1.0 };
    const token_ids = [_]i32{ 380, 380, 754 };

    var base = try computeBoundaryWeightedCeAndLogitGrad(
        allocator,
        &logits,
        &targets,
        &mask,
        3,
        1.0,
        0.0,
        1.0,
        1,
        0.0,
        4,
        1.0,
        0,
        null,
        3,
        0.0,
        1.0,
        1,
        0.0,
        1,
    );
    defer base.deinit(allocator);

    var ranked = try computeBoundaryWeightedCeAndLogitGrad(
        allocator,
        &logits,
        &targets,
        &mask,
        3,
        1.0,
        0.0,
        1.0,
        1,
        0.4,
        1,
        1.0,
        0,
        &token_ids,
        3,
        0.0,
        1.0,
        1,
        0.0,
        1,
    );
    defer ranked.deinit(allocator);

    try std.testing.expectApproxEqAbs(base.ce_loss, ranked.ce_loss, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.2), ranked.rank_loss, 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[0] + 0.4, ranked.logit_grad[0], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[1] - 0.4, ranked.logit_grad[1], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[2] - 0.4, ranked.logit_grad[2], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[3] + 0.4, ranked.logit_grad[3], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[4], ranked.logit_grad[4], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[5], ranked.logit_grad[5], 1e-6);
}

test "boundary same-token negative weight hardens CE for surface-confusable negatives" {
    const allocator = std.testing.allocator;
    const logits = [_]f32{
        0.0, 0.0, // gold positive token 380
        0.0, 0.0, // same-token negative should receive extra CE weight
        0.0, 0.0, // different-token negative stays at baseline weight
    };
    const targets = [_]f32{
        0.0, 1.0,
        1.0, 0.0,
        1.0, 0.0,
    };
    const mask = [_]f32{ 1.0, 1.0, 1.0 };
    const token_ids = [_]i32{ 380, 380, 754 };

    var base = try computeBoundaryWeightedCeAndLogitGrad(
        allocator,
        &logits,
        &targets,
        &mask,
        3,
        1.0,
        0.0,
        1.0,
        1,
        0.0,
        4,
        1.0,
        0,
        null,
        3,
        0.0,
        1.0,
        1,
        0.0,
        1,
    );
    defer base.deinit(allocator);

    var weighted = try computeBoundaryWeightedCeAndLogitGrad(
        allocator,
        &logits,
        &targets,
        &mask,
        3,
        1.0,
        0.0,
        1.0,
        1,
        0.0,
        4,
        3.0,
        0,
        &token_ids,
        3,
        0.0,
        1.0,
        1,
        0.0,
        1,
    );
    defer weighted.deinit(allocator);

    try std.testing.expectApproxEqAbs(@log(@as(f32, 2.0)), base.ce_loss, 1e-6);
    try std.testing.expectApproxEqAbs(@log(@as(f32, 2.0)) * @as(f32, 5.0 / 3.0), weighted.ce_loss, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), weighted.rank_loss, 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[0], weighted.logit_grad[0], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[1], weighted.logit_grad[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), weighted.logit_grad[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), weighted.logit_grad[3], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[4], weighted.logit_grad[4], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[5], weighted.logit_grad[5], 1e-6);
}

test "boundary candidate negative weight hardens CE for sentence-like candidates" {
    const allocator = std.testing.allocator;
    const logits = [_]f32{
        0.0, 0.0, // gold positive
        0.0, 0.0, // sentence-like negative should receive extra CE weight
        0.0, 0.0, // ordinary negative stays at baseline weight
    };
    const targets = [_]f32{
        0.0, 1.0,
        1.0, 0.0,
        1.0, 0.0,
    };
    const mask = [_]f32{ 1.0, 1.0, 1.0 };
    const candidate_mask = [_]f32{ 1.0, 1.0, 0.0 };

    var base = try computeBoundaryWeightedCeAndLogitGrad(
        allocator,
        &logits,
        &targets,
        &mask,
        3,
        1.0,
        0.0,
        1.0,
        1,
        0.0,
        4,
        1.0,
        0,
        null,
        3,
        0.0,
        1.0,
        1,
        0.0,
        1,
    );
    defer base.deinit(allocator);

    var weighted = try computeBoundaryWeightedCeAndLogitGradWithCandidateMask(
        allocator,
        &logits,
        &targets,
        &mask,
        3,
        1.0,
        0.0,
        1.0,
        1,
        0.0,
        4,
        1.0,
        0,
        0.0,
        8,
        3.0,
        &candidate_mask,
        null,
        3,
        0.0,
        1.0,
        1,
        0.0,
        1,
    );
    defer weighted.deinit(allocator);

    try std.testing.expectApproxEqAbs(@log(@as(f32, 2.0)), base.ce_loss, 1e-6);
    try std.testing.expectApproxEqAbs(@log(@as(f32, 2.0)) * @as(f32, 5.0 / 3.0), weighted.ce_loss, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), weighted.rank_loss, 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[0], weighted.logit_grad[0], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[1], weighted.logit_grad[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), weighted.logit_grad[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), weighted.logit_grad[3], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[4], weighted.logit_grad[4], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[5], weighted.logit_grad[5], 1e-6);
}

test "boundary candidate rank loss targets sentence-like hard negatives" {
    const allocator = std.testing.allocator;
    const logits = [_]f32{
        0.0, 0.0, // gold positive margin 0
        0.0, 2.0, // sentence-like negative selected by candidate rank loss
        0.0, 3.0, // harder non-candidate negative ignored by this term
    };
    const targets = [_]f32{
        0.0, 1.0,
        1.0, 0.0,
        1.0, 0.0,
    };
    const mask = [_]f32{ 1.0, 1.0, 1.0 };
    const candidate_mask = [_]f32{ 1.0, 1.0, 0.0 };

    var base = try computeBoundaryWeightedCeAndLogitGrad(
        allocator,
        &logits,
        &targets,
        &mask,
        3,
        1.0,
        0.0,
        1.0,
        1,
        0.0,
        4,
        1.0,
        0,
        null,
        3,
        0.0,
        1.0,
        1,
        0.0,
        1,
    );
    defer base.deinit(allocator);

    var ranked = try computeBoundaryWeightedCeAndLogitGradWithCandidateMask(
        allocator,
        &logits,
        &targets,
        &mask,
        3,
        1.0,
        0.0,
        1.0,
        1,
        0.0,
        4,
        1.0,
        0,
        0.4,
        1,
        1.0,
        &candidate_mask,
        null,
        3,
        0.0,
        1.0,
        1,
        0.0,
        1,
    );
    defer ranked.deinit(allocator);

    try std.testing.expectApproxEqAbs(base.ce_loss, ranked.ce_loss, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.2), ranked.rank_loss, 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[0] + 0.4, ranked.logit_grad[0], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[1] - 0.4, ranked.logit_grad[1], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[2] - 0.4, ranked.logit_grad[2], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[3] + 0.4, ranked.logit_grad[3], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[4], ranked.logit_grad[4], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[5], ranked.logit_grad[5], 1e-6);
}

test "boundary same-token negative top-k weights only hardest surface negatives" {
    const allocator = std.testing.allocator;
    const logits = [_]f32{
        0.0, 0.0, // gold positive token 380
        0.0, 2.0, // highest-margin same-token negative should be weighted
        0.0, -1.0, // easier same-token negative should remain baseline
        0.0, 3.0, // different-token negative should remain baseline
    };
    const targets = [_]f32{
        0.0, 1.0,
        1.0, 0.0,
        1.0, 0.0,
        1.0, 0.0,
    };
    const mask = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const token_ids = [_]i32{ 380, 380, 380, 754 };

    var base = try computeBoundaryWeightedCeAndLogitGrad(
        allocator,
        &logits,
        &targets,
        &mask,
        4,
        1.0,
        0.0,
        1.0,
        1,
        0.0,
        4,
        1.0,
        0,
        null,
        4,
        0.0,
        1.0,
        1,
        0.0,
        1,
    );
    defer base.deinit(allocator);

    var weighted = try computeBoundaryWeightedCeAndLogitGrad(
        allocator,
        &logits,
        &targets,
        &mask,
        4,
        1.0,
        0.0,
        1.0,
        1,
        0.0,
        4,
        3.0,
        1,
        &token_ids,
        4,
        0.0,
        1.0,
        1,
        0.0,
        1,
    );
    defer weighted.deinit(allocator);

    try std.testing.expect(weighted.ce_loss > base.ce_loss);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), weighted.rank_loss, 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[0], weighted.logit_grad[0], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[1], weighted.logit_grad[1], 1e-6);
    try std.testing.expect(!std.math.approxEqAbs(f32, base.logit_grad[2], weighted.logit_grad[2], 1e-6));
    try std.testing.expect(!std.math.approxEqAbs(f32, base.logit_grad[3], weighted.logit_grad[3], 1e-6));
    try std.testing.expectApproxEqAbs(base.logit_grad[4], weighted.logit_grad[4], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[5], weighted.logit_grad[5], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[6], weighted.logit_grad[6], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[7], weighted.logit_grad[7], 1e-6);
}

test "boundary gold-count rank loss selects hardest negatives per sample" {
    const allocator = std.testing.allocator;
    const logits = [_]f32{
        0.0, 0.0, // gold positive margin 0
        0.0, 1.0, // gold positive margin 1
        0.0, 2.0, // hardest negative margin 2
        0.0, 0.5, // second-hardest negative margin 0.5
        0.0, -5.0, // easy negative outside gold-count selection
    };
    const targets = [_]f32{
        0.0, 1.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 0.0,
        1.0, 0.0,
    };
    const mask = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0 };

    var base = try computeBoundaryWeightedCeAndLogitGrad(
        allocator,
        &logits,
        &targets,
        &mask,
        5,
        1.0,
        0.0,
        1.0,
        1,
        0.0,
        4,
        1.0,
        0,
        null,
        5,
        0.0,
        1.0,
        1,
        0.0,
        1,
    );
    defer base.deinit(allocator);

    var ranked = try computeBoundaryWeightedCeAndLogitGrad(
        allocator,
        &logits,
        &targets,
        &mask,
        5,
        1.0,
        0.0,
        1.0,
        1,
        0.0,
        4,
        1.0,
        0,
        null,
        5,
        0.4,
        1.0,
        1,
        0.0,
        1,
    );
    defer ranked.deinit(allocator);

    try std.testing.expectApproxEqAbs(base.ce_loss, ranked.ce_loss, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), ranked.rank_loss, 1e-6);
    try std.testing.expectApproxEqAbs(base.loss + 0.7, ranked.loss, 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[0] + 0.2, ranked.logit_grad[0], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[1] - 0.2, ranked.logit_grad[1], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[2] + 0.2, ranked.logit_grad[2], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[3] - 0.2, ranked.logit_grad[3], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[4] - 0.2, ranked.logit_grad[4], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[5] + 0.2, ranked.logit_grad[5], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[6] - 0.2, ranked.logit_grad[6], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[7] + 0.2, ranked.logit_grad[7], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[8], ranked.logit_grad[8], 1e-6);
    try std.testing.expectApproxEqAbs(base.logit_grad[9], ranked.logit_grad[9], 1e-6);
}

test "boundary local window loss sharpens exact gold token" {
    const allocator = std.testing.allocator;
    const logits = [_]f32{
        0.0, 0.0, // far negative outside window
        0.0, 2.0, // near false positive
        0.0, 0.0, // gold boundary
        0.0, 1.0, // near false positive
        0.0, 0.0, // far negative outside window
    };
    const targets = [_]f32{
        1.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 0.0,
    };
    const mask = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0 };

    var result = try computeBoundaryWeightedCeAndLogitGrad(
        allocator,
        &logits,
        &targets,
        &mask,
        5,
        1.0,
        0.0,
        1.0,
        1,
        0.0,
        4,
        1.0,
        0,
        null,
        5,
        0.0,
        1.0,
        1,
        1.0,
        1,
    );
    defer result.deinit(allocator);

    try std.testing.expect(result.ce_loss > 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result.rank_loss, 1e-6);
    try std.testing.expect(result.local_window_loss > 0.0);
    try std.testing.expectApproxEqAbs(result.ce_loss + result.local_window_loss, result.loss, 1e-6);
    try std.testing.expect(result.loss > 0.0);
    try std.testing.expect(result.logit_grad[2 * 2] > 0.0);
    try std.testing.expect(result.logit_grad[2 * 2 + 1] < 0.0);
    try std.testing.expect(result.logit_grad[1 * 2] < 0.0);
    try std.testing.expect(result.logit_grad[1 * 2 + 1] > 0.0);
    try std.testing.expect(result.logit_grad[3 * 2] < 0.0);
    try std.testing.expect(result.logit_grad[3 * 2 + 1] > 0.0);
}

test "trainStep learns positive class polarity on separable boundary features" {
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
        .learning_rate = 0.05,
        .warmup_steps = 1,
        .weight_decay = 0.0,
        .lambda_embed = 0.0,
        .pos_weight = 1.0,
        .total_steps = 32,
        .step_log_every = 0,
    };
    var trainer = try FusedTrainer.init(allocator, config, &cb);
    defer trainer.deinit();

    @memcpy(trainer.boundary_head.w1, &[_]f32{
        1.0, 0.0,
        0.0, 1.0,
    });
    @memset(trainer.boundary_head.b1, 0);
    @memset(trainer.boundary_head.w2, 0);
    @memset(trainer.boundary_head.b2, 0);

    const features = [_]f32{
        2.0, 0.0,
        0.0, 2.0,
        1.5, 0.0,
        0.0, 1.5,
    };
    const labels_one_hot = [_]f32{
        0.0, 1.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
    };
    const scalar_labels = [_]f32{ 1.0, 0.0, 1.0, 0.0 };
    const attention_mask = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const chunk_embeddings = features;
    const chunk_mask = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const doc_ids = [_]u32{ 0, 1, 2, 3 };

    const before_logits = try trainer.evaluateBoundaryLogitsOwned(allocator, &features, 4);
    defer allocator.free(before_logits);
    const before_pos = meanBoundaryProbabilityForLabel(before_logits, &scalar_labels, &attention_mask, true);
    const before_neg = meanBoundaryProbabilityForLabel(before_logits, &scalar_labels, &attention_mask, false);

    var final_loss: f32 = 0.0;
    for (0..32) |_| {
        const summary = try trainer.trainStep(
            allocator,
            &features,
            &labels_one_hot,
            &attention_mask,
            null,
            null,
            &chunk_embeddings,
            &chunk_mask,
            &doc_ids,
            4,
            4,
            1,
            2,
        );
        final_loss = summary.boundary_loss;
    }

    const after_logits = try trainer.evaluateBoundaryLogitsOwned(allocator, &features, 4);
    defer allocator.free(after_logits);
    const after_pos = meanBoundaryProbabilityForLabel(after_logits, &scalar_labels, &attention_mask, true);
    const after_neg = meanBoundaryProbabilityForLabel(after_logits, &scalar_labels, &attention_mask, false);

    try std.testing.expect(final_loss < @log(@as(f32, 2.0)));
    try std.testing.expect(after_pos > before_pos + 0.25);
    try std.testing.expect(after_neg < before_neg - 0.25);
    try std.testing.expect(after_pos > 0.75);
    try std.testing.expect(after_neg < 0.25);
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
        null,
        null,
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
        null,
        null,
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
        null,
        null,
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
