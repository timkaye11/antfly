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
    };
}

/// Whether a given eval batch should take the compiled path.
/// See fused_chunker_compiled_forward_policy.zig for the rules and tests.
pub const shouldUseCompiledForBatch = policy.shouldUseCompiledForBatch;

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

    /// Run the eval forward through the compiled segment sessions.
    /// Returns the owned final feature buffer `[batch * seq * hidden]`,
    /// matching the shape and semantics of `modern_bert.forward`.
    ///
    /// `bert_config` drives the eager embedding stage and the final
    /// LayerNorm; eval callers must not set NEFTune fields (eval has no
    /// embedding noise). LoRA comes exclusively from `lora_layers` host
    /// buffers, never from weight-store lora_a/lora_b tensors.
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

            const session_slot = &self.sessions[@intCast(segment_start)];
            var rebuild_session = session_slot.* == null;
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
                    rebuild_session = true;
                }
            }
            if (rebuild_session) {
                session_slot.* = try segmented_encoder.ModernBertSegmentForwardSession.init(
                    allocator,
                    self.graph_config,
                    @intCast(actual_batch),
                    @intCast(max_seq),
                    segment_start,
                    segment_end,
                    self.lora_rank,
                    self.lora_alpha,
                    false, // eval needs no activation captures
                );
            }
            const session = &(session_slot.* orelse return error.MissingCompiledForwardSessions);

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
