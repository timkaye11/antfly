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

//! Compiled MPSGraph segmented encoder forward for fused-chunker EVAL paths.
//!
//! The trainer's step forward already has a compiled fast path
//! (`runCompiledSegmentedEncoderForward` in train/train_fused_chunker.zig)
//! that mirrors `modern_bert.forwardCapturingActivations`, including the
//! per-layer activation captures the LoRA backward needs. Eval forwards
//! (the trainer's validation loop and the standalone eval-fused-chunker
//! binary) only need the FINAL hidden states, so this module runs the same
//! parity-validated segment graphs capture-free:
//!
//!   embeddings (eager, identical numerics)
//!     -> per-layer compiled MPSGraph segment sessions (LoRA fed from host
//!        adapter buffers, `emit_captures = false`)
//!     -> final LayerNorm (eager `cb.layerNorm`, identical numerics)
//!
//! Everything is opt-in behind ANTFLY_FUSED_CHUNKER_COMPILED_SEGMENT_FORWARD
//! (or the corresponding CLI flags), defaults OFF, and callers fall back to
//! the eager `modern_bert.forward` on any failure.
//!
//! SERVING (src/pipelines/fused_chunking.zig) reuses the same forward over a
//! MERGED export: LoRA is already folded into the base weights, so serving
//! constructs via `initMergedBase` (rank 0 -> plain base-weight segment
//! graphs, no adapter injection) and calls `forward` with an empty
//! `lora_layers` slice. Serving defaults the compiled path ON for the Metal
//! backend behind the ANTFLY_FUSED_CHUNKER_SERVING_COMPILED_FORWARD kill
//! switch (see `servingEnvEnabled`), again with the eager forward as the
//! first-failure fallback.
//!
//! Numerical note: the compiled and eager forwards intentionally run
//! different f32 kernel implementations of the same math; final features
//! differ by cross-kernel noise (measured ~2e-3 max-rel at depth 22).
//! Boundary metrics under the flag must be validated on GPU by running the
//! same eval twice with the flag on/off and diffing the results JSON.

const std = @import("std");
const platform = @import("antfly_platform");
const ops_mod = @import("../ops/ops.zig");
const ComputeBackend = ops_mod.ComputeBackend;
const modern_bert = @import("../architectures/modern_bert.zig");
const modern_bert_graph = @import("../architectures/modern_bert_graph.zig");
const segmented_encoder = @import("../graph/segmented_encoder.zig");
const fused_chunker_lora = @import("lora_adapter_set.zig");
const policy = @import("fused_chunker_compiled_forward_policy.zig");

/// Same env flag the trainer's step forward uses; eval paths honor it too.
pub const env_flag = "ANTFLY_FUSED_CHUNKER_COMPILED_SEGMENT_FORWARD";

pub fn envEnabled() bool {
    return platform.env.getenvBoolDefault(env_flag, false);
}

/// Serving kill switch: the fused_chunking pipeline defaults the compiled
/// segment forward ON when running on the Metal backend; set this to "0"
/// (or "false"/"no"/"off") to force the eager per-window forward.
pub const serving_env_flag = "ANTFLY_FUSED_CHUNKER_SERVING_COMPILED_FORWARD";

pub fn servingEnvEnabled() bool {
    return policy.servingCompiledForwardEnabled(platform.env.getenv(serving_env_flag));
}

/// Layers fused into each compiled serving segment. Default 1 (the shipped,
/// parity-validated per-layer segmentation). Larger values fuse more encoder
/// layers into one MPSGraph execution — fewer GPU round-trips and fewer
/// inter-segment host hidden-state copies per forward — at the cost of a bigger
/// compiled graph and more transient GPU memory per execution. Used by both the
/// fused-chunker boundary forward and the frozen embedder's compiled forward.
pub const serving_layers_per_segment_env = "ANTFLY_FUSED_CHUNKER_SERVING_LAYERS_PER_SEGMENT";

pub fn servingLayersPerSegment() u32 {
    const v = platform.env.getenvUsize(serving_layers_per_segment_env) orelse return 1;
    if (v == 0) return 1;
    return std.math.cast(u32, v) orelse 1;
}

/// ModernBERT-base head count; the fused chunker encoder is not configurable
/// on this axis (matches the eager `modern_bert.Config` default and the
/// trainer's compiled forward config).
pub const eval_num_attention_heads = policy.eval_num_attention_heads;

/// Build the graph-side ModernBERT config used by compiled segment sessions
/// from the eval-facing dimensions, mirroring the eager modern_bert.Config
/// defaults for everything not exposed on the eval CLI. Mirrors the config
/// the trainer constructs for its compiled step forward.
pub fn graphConfigForEval(
    hidden_size: u32,
    num_layers: u32,
    intermediate_size: u32,
) !modern_bert_graph.Config {
    const head_dim = try policy.headDimForHiddenSize(hidden_size);
    const eager_defaults = modern_bert.Config{};
    return .{
        .vocab_size = eager_defaults.vocab_size,
        .hidden_size = hidden_size,
        .num_hidden_layers = num_layers,
        .num_attention_heads = eval_num_attention_heads,
        .head_dim = head_dim,
        .intermediate_size = intermediate_size,
        .max_position_embeddings = eager_defaults.max_position_embeddings,
        .layer_norm_eps = eager_defaults.layer_norm_eps,
        .rope_theta = eager_defaults.global_rope_theta,
        .local_rope_theta = eager_defaults.local_rope_theta,
        .local_attention_window = eager_defaults.local_attention_window,
        .global_attn_every_n_layers = eager_defaults.global_attn_every_n_layers,
        .rope_interleaved = eager_defaults.rope_interleaved,
        .attn_norm0_identity = eager_defaults.attn_norm0_identity,
    };
}

/// Build the graph-side ModernBERT config for SERVING from the eager config
/// the pipeline derived from the export's config.json. Unlike
/// `graphConfigForEval` (which mirrors eval-CLI defaults), every geometry and
/// attention/RoPE hyperparameter comes from the export so the compiled
/// segments match the eager `modern_bert.forwardCT` the pipeline would
/// otherwise run.
pub fn graphConfigFromBertConfig(bert: modern_bert.Config) !modern_bert_graph.Config {
    if (bert.num_attention_heads == 0 or
        bert.hidden_size == 0 or
        bert.hidden_size % bert.num_attention_heads != 0)
    {
        return error.InvalidHeadDim;
    }
    return .{
        .vocab_size = bert.vocab_size,
        .hidden_size = bert.hidden_size,
        .num_hidden_layers = bert.num_hidden_layers,
        .num_attention_heads = bert.num_attention_heads,
        .head_dim = bert.hidden_size / bert.num_attention_heads,
        .intermediate_size = bert.intermediate_size,
        .max_position_embeddings = bert.max_position_embeddings,
        .layer_norm_eps = bert.layer_norm_eps,
        .rope_theta = bert.global_rope_theta,
        .local_rope_theta = bert.local_rope_theta,
        .local_attention_window = bert.local_attention_window,
        .global_attn_every_n_layers = bert.global_attn_every_n_layers,
        // Carry the eager forward's RoPE/layer-0 conventions so the compiled
        // segments match the exact `modern_bert.forward` the pipeline would run.
        // Fused chunker exports keep the defaults (interleaved RoPE, layer-0
        // LayerNorm); the frozen text embedder sets rotate_half + layer-0
        // Identity via session_factory.parseModernBertConfig.
        .rope_interleaved = bert.rope_interleaved,
        .attn_norm0_identity = bert.attn_norm0_identity,
    };
}

/// Whether a given eval batch should take the compiled path.
/// See fused_chunker_compiled_forward_policy.zig for the rules and tests.
pub const shouldUseCompiledForBatch = policy.shouldUseCompiledForBatch;

/// Whether a serving forward window should take the compiled path.
/// See fused_chunker_compiled_forward_policy.zig for the rules and tests.
pub const shouldUseCompiledForServingWindow = policy.shouldUseCompiledForServingWindow;

/// Capture-free compiled segmented eval forward.
///
/// Owns one lazily-built `ModernBertSegmentForwardSession` slot per segment
/// start layer (same caching scheme as the trainer's compiled step forward).
/// Callers should treat any `forward` error as a signal to latch `failed`
/// and fall back to `modern_bert.forward` for the remainder of the run.
pub const CompiledEvalForward = struct {
    allocator: std.mem.Allocator,
    graph_config: modern_bert_graph.Config,
    layers_per_segment: u32,
    lora_rank: u32,
    lora_alpha: f32,
    sessions: []?segmented_encoder.ModernBertSegmentForwardSession,
    /// First-failure latch. Set by callers when `forward` errors so later
    /// batches skip straight to the eager fallback.
    failed: bool = false,

    pub fn init(
        allocator: std.mem.Allocator,
        graph_config: modern_bert_graph.Config,
        layers_per_segment: u32,
        lora_rank: u32,
        lora_alpha: f32,
    ) !CompiledEvalForward {
        if (lora_rank == 0) return error.LoRARankRequired;
        return initInternal(allocator, graph_config, layers_per_segment, lora_rank, lora_alpha);
    }

    /// Construction over a MERGED export (serving): LoRA is already folded
    /// into the base weights, so segment graphs compile WITHOUT adapter
    /// injection and `forward` must be called with an empty `lora_layers`
    /// slice. Do not gate calls through `shouldUse` (that is the eval
    /// adapter-checkpoint policy, which rejects rank 0); serving gates with
    /// `shouldUseCompiledForServingWindow` instead.
    pub fn initMergedBase(
        allocator: std.mem.Allocator,
        graph_config: modern_bert_graph.Config,
        layers_per_segment: u32,
    ) !CompiledEvalForward {
        return initInternal(allocator, graph_config, layers_per_segment, 0, 0.0);
    }

    fn initInternal(
        allocator: std.mem.Allocator,
        graph_config: modern_bert_graph.Config,
        layers_per_segment: u32,
        lora_rank: u32,
        lora_alpha: f32,
    ) !CompiledEvalForward {
        if (graph_config.num_hidden_layers == 0) return error.InvalidLayerRange;
        const sessions = try allocator.alloc(
            ?segmented_encoder.ModernBertSegmentForwardSession,
            graph_config.num_hidden_layers,
        );
        for (sessions) |*slot| slot.* = null;
        return .{
            .allocator = allocator,
            .graph_config = graph_config,
            .layers_per_segment = policy.normalizedLayersPerSegment(layers_per_segment),
            .lora_rank = lora_rank,
            .lora_alpha = lora_alpha,
            .sessions = sessions,
        };
    }

    pub fn deinit(self: *CompiledEvalForward) void {
        for (self.sessions) |*slot| {
            if (slot.*) |*session| session.deinit();
        }
        self.allocator.free(self.sessions);
        self.* = undefined;
    }

    pub fn shouldUse(self: *const CompiledEvalForward, batch_count: usize, full_batch_size: usize) bool {
        return shouldUseCompiledForBatch(self.failed, self.lora_rank, batch_count, full_batch_size);
    }

    /// Build (or rebuild on geometry change) the compiled session for one
    /// segment, returning a stable pointer into the session slot.
    fn ensureSession(
        self: *CompiledEvalForward,
        segment_start: u32,
        segment_end: u32,
        actual_batch: usize,
        max_seq: usize,
    ) !*segmented_encoder.ModernBertSegmentForwardSession {
        const session_slot = &self.sessions[@intCast(segment_start)];
        if (session_slot.*) |*session| {
            if (session.batch != actual_batch or
                session.seq_len != max_seq or
                session.start_layer != segment_start or
                session.end_layer != segment_end or
                session.config.hidden_size != self.graph_config.hidden_size or
                session.config.intermediate_size != self.graph_config.intermediate_size)
            {
                session.deinit();
                session_slot.* = null;
            }
        }
        if (session_slot.* == null) {
            session_slot.* = try segmented_encoder.ModernBertSegmentForwardSession.init(
                self.allocator,
                self.graph_config,
                @intCast(actual_batch),
                @intCast(max_seq),
                segment_start,
                segment_end,
                self.lora_rank,
                self.lora_alpha,
                false, // eval/serving forwards need no activation captures
            );
        }
        if (session_slot.*) |*session| return session;
        return error.MissingCompiledForwardSessions;
    }

    /// Pre-compile every segment session for one [batch, seq] geometry so the
    /// first forward doesn't pay the cold MPSGraph compile. Serving calls
    /// this at pipeline load with its only geometry (batch=1, max_seq_len).
    pub fn warmup(self: *CompiledEvalForward, actual_batch: usize, max_seq: usize) !void {
        const num_layers = self.graph_config.num_hidden_layers;
        var segment_start: u32 = 0;
        while (segment_start < num_layers) {
            const segment_end = @min(segment_start + self.layers_per_segment, num_layers);
            _ = try self.ensureSession(segment_start, segment_end, actual_batch, max_seq);
            segment_start = segment_end;
        }
    }

    /// Run the eval forward through the compiled segment sessions.
    /// Returns the owned final feature buffer `[batch * seq * hidden]`,
    /// matching the shape and semantics of `modern_bert.forward`.
    ///
    /// `bert_config` drives the eager embedding stage and the final
    /// LayerNorm; eval callers must not set NEFTune fields (eval has no
    /// embedding noise). LoRA comes exclusively from `lora_layers` host
    /// buffers, never from weight-store lora_a/lora_b tensors; merged-base
    /// construction (`initMergedBase`) takes an empty `lora_layers` slice.
    ///
    /// The returned buffer is allocated from the CompiledEvalForward's own
    /// allocator (the one passed at init), not a per-call allocator.
    pub fn forward(
        self: *CompiledEvalForward,
        cb: *const ComputeBackend,
        bert_config: modern_bert.Config,
        ids_i64: []const i64,
        mask_i64: []const i64,
        actual_batch: usize,
        max_seq: usize,
        lora_layers: []fused_chunker_lora.LoRALayer,
    ) ![]f32 {
        const allocator = self.allocator;
        const total_tokens = actual_batch * max_seq;
        const H: usize = @intCast(self.graph_config.hidden_size);
        const num_layers = self.graph_config.num_hidden_layers;
        if (self.sessions.len < num_layers) return error.MissingCompiledForwardSessions;
        if (ids_i64.len != total_tokens or mask_i64.len != total_tokens) return error.UnexpectedOutputShape;

        const attn_mask_f32 = try allocator.alloc(f32, total_tokens);
        defer allocator.free(attn_mask_f32);
        for (mask_i64, attn_mask_f32) |m, *out| out.* = @floatFromInt(m);

        var shared_bias = segmented_encoder.SharedSegmentAttnBias.init(
            allocator,
            @intCast(actual_batch),
            @intCast(max_seq),
            self.graph_config.num_attention_heads,
            self.graph_config.local_attention_window,
            attn_mask_f32,
        );
        defer shared_bias.deinit();

        // Embeddings stay on the eager path (token gather + LayerNorm), so
        // their numerics match modern_bert.forward exactly. No captures.
        const emb_ct = try modern_bert.embeddingsForwardCT(
            cb,
            allocator,
            bert_config,
            ids_i64,
            total_tokens,
            max_seq,
            null,
        );
        const embedding_host = blk: {
            defer cb.free(emb_ct);
            break :blk try cb.toFloat32(emb_ct, allocator);
        };
        var hidden_owned: []f32 = embedding_host;
        defer allocator.free(hidden_owned);
        if (hidden_owned.len != total_tokens * H) return error.UnexpectedOutputShape;

        var segment_start: u32 = 0;
        while (segment_start < num_layers) {
            const segment_end = @min(segment_start + self.layers_per_segment, num_layers);

            const session = try self.ensureSession(segment_start, segment_end, actual_batch, max_seq);

            var result = try session.execute(cb, hidden_owned, attn_mask_f32, lora_layers, &shared_bias);
            defer result.deinit();

            const next_hidden = try allocator.dupe(f32, result.hidden_out);
            allocator.free(hidden_owned);
            hidden_owned = next_hidden;
            segment_start = segment_end;
        }

        // Final LayerNorm through the same backend op as the eager forward.
        const hidden_ct = try cb.fromFloat32Shape(hidden_owned, &.{ @intCast(total_tokens), @intCast(H) });
        defer cb.free(hidden_ct);
        const fn_w = try cb.getWeight("model.final_norm.weight");
        defer cb.free(fn_w);
        const fn_b = cb.getWeight("model.final_norm.bias") catch |err| blk: {
            if (err != error.MissingWeight) return err;
            const zeros = try allocator.alloc(f32, H);
            defer allocator.free(zeros);
            @memset(zeros, 0);
            break :blk try cb.fromFloat32Shape(zeros, &.{@as(i32, @intCast(H))});
        };
        defer cb.free(fn_b);
        const normed_ct = try cb.layerNorm(hidden_ct, fn_w, fn_b, H, bert_config.layer_norm_eps);
        defer cb.free(normed_ct);
        return cb.toFloat32(normed_ct, allocator);
    }
};

// ---------------------------------------------------------------------------
// Tests (pure logic; session execution requires a Metal device and is
// validated by the GPU runbook, not unit tests). Import-free decision logic
// and its tests live in fused_chunker_compiled_forward_policy.zig, which has
// a standalone test target (test-fused-chunker-compiled-forward).
// ---------------------------------------------------------------------------

test "graphConfigForEval mirrors eager ModernBERT defaults" {
    const cfg = try graphConfigForEval(768, 22, 1152);
    try std.testing.expectEqual(@as(u32, 768), cfg.hidden_size);
    try std.testing.expectEqual(@as(u32, 22), cfg.num_hidden_layers);
    try std.testing.expectEqual(@as(u32, 1152), cfg.intermediate_size);
    try std.testing.expectEqual(@as(u32, 12), cfg.num_attention_heads);
    try std.testing.expectEqual(@as(u32, 64), cfg.head_dim);

    const eager = modern_bert.Config{};
    try std.testing.expectEqual(eager.vocab_size, cfg.vocab_size);
    try std.testing.expectEqual(eager.local_attention_window, cfg.local_attention_window);
    try std.testing.expectEqual(eager.global_attn_every_n_layers, cfg.global_attn_every_n_layers);
    try std.testing.expectEqual(eager.global_rope_theta, cfg.rope_theta);
    try std.testing.expectEqual(eager.local_rope_theta, cfg.local_rope_theta);
    try std.testing.expectEqual(eager.layer_norm_eps, cfg.layer_norm_eps);
}

test "graphConfigForEval rejects hidden sizes not divisible by head count" {
    try std.testing.expectError(error.InvalidHeadDim, graphConfigForEval(770, 22, 1152));
    try std.testing.expectError(error.InvalidHeadDim, graphConfigForEval(0, 22, 1152));
}

test "CompiledEvalForward init allocates lazy session slots" {
    const allocator = std.testing.allocator;
    const cfg = try graphConfigForEval(768, 22, 1152);
    var cef = try CompiledEvalForward.init(allocator, cfg, 1, 8, 16.0);
    defer cef.deinit();
    try std.testing.expectEqual(@as(usize, 22), cef.sessions.len);
    for (cef.sessions) |slot| try std.testing.expect(slot == null);
    try std.testing.expect(!cef.failed);
    try std.testing.expect(cef.shouldUse(32, 32));
    cef.failed = true;
    try std.testing.expect(!cef.shouldUse(32, 32));
}

test "CompiledEvalForward init requires LoRA rank and normalizes segment size" {
    const allocator = std.testing.allocator;
    const cfg = try graphConfigForEval(768, 4, 1152);
    try std.testing.expectError(error.LoRARankRequired, CompiledEvalForward.init(allocator, cfg, 1, 0, 16.0));
    var cef = try CompiledEvalForward.init(allocator, cfg, 0, 4, 8.0);
    defer cef.deinit();
    try std.testing.expectEqual(@as(u32, 1), cef.layers_per_segment);
}

test "initMergedBase builds a rank-0 (no-LoRA) forward for merged exports" {
    const allocator = std.testing.allocator;
    const cfg = try graphConfigForEval(768, 22, 1152);
    var cef = try CompiledEvalForward.initMergedBase(allocator, cfg, 1);
    defer cef.deinit();
    try std.testing.expectEqual(@as(u32, 0), cef.lora_rank);
    try std.testing.expectEqual(@as(usize, 22), cef.sessions.len);
    for (cef.sessions) |slot| try std.testing.expect(slot == null);
    // The eval adapter-batch policy rejects rank 0 by design; serving gates
    // through shouldUseCompiledForServingWindow instead.
    try std.testing.expect(!cef.shouldUse(32, 32));
    try std.testing.expect(shouldUseCompiledForServingWindow(cef.failed, 384, 384));
    cef.failed = true;
    try std.testing.expect(!shouldUseCompiledForServingWindow(cef.failed, 384, 384));
}

test "graphConfigFromBertConfig maps the export's eager config" {
    const bert = modern_bert.Config{
        .vocab_size = 50368,
        .hidden_size = 768,
        .num_hidden_layers = 22,
        .num_attention_heads = 12,
        .intermediate_size = 1152,
        .max_position_embeddings = 8192,
        .global_rope_theta = 160000.0,
        .local_rope_theta = 10000.0,
        .global_attn_every_n_layers = 3,
        .local_attention_window = 128,
        .layer_norm_eps = 1e-5,
    };
    const cfg = try graphConfigFromBertConfig(bert);
    // Default (fused chunker) conventions carry through.
    try std.testing.expectEqual(true, cfg.rope_interleaved);
    try std.testing.expectEqual(false, cfg.attn_norm0_identity);
    // The frozen text embedder's HF conventions carry through too.
    var embedder = bert;
    embedder.rope_interleaved = false;
    embedder.attn_norm0_identity = true;
    const embedder_cfg = try graphConfigFromBertConfig(embedder);
    try std.testing.expectEqual(false, embedder_cfg.rope_interleaved);
    try std.testing.expectEqual(true, embedder_cfg.attn_norm0_identity);
    try std.testing.expectEqual(bert.vocab_size, cfg.vocab_size);
    try std.testing.expectEqual(bert.hidden_size, cfg.hidden_size);
    try std.testing.expectEqual(bert.num_hidden_layers, cfg.num_hidden_layers);
    try std.testing.expectEqual(bert.num_attention_heads, cfg.num_attention_heads);
    try std.testing.expectEqual(@as(u32, 64), cfg.head_dim);
    try std.testing.expectEqual(bert.intermediate_size, cfg.intermediate_size);
    try std.testing.expectEqual(bert.max_position_embeddings, cfg.max_position_embeddings);
    try std.testing.expectEqual(bert.global_rope_theta, cfg.rope_theta);
    try std.testing.expectEqual(bert.local_rope_theta, cfg.local_rope_theta);
    try std.testing.expectEqual(bert.local_attention_window, cfg.local_attention_window);
    try std.testing.expectEqual(bert.global_attn_every_n_layers, cfg.global_attn_every_n_layers);

    // Non-default head counts flow through instead of the eval-CLI fixed 12.
    var wide = bert;
    wide.num_attention_heads = 16;
    const wide_cfg = try graphConfigFromBertConfig(wide);
    try std.testing.expectEqual(@as(u32, 16), wide_cfg.num_attention_heads);
    try std.testing.expectEqual(@as(u32, 48), wide_cfg.head_dim);

    var bad = bert;
    bad.num_attention_heads = 7;
    try std.testing.expectError(error.InvalidHeadDim, graphConfigFromBertConfig(bad));
    bad.num_attention_heads = 0;
    try std.testing.expectError(error.InvalidHeadDim, graphConfigFromBertConfig(bad));
}
