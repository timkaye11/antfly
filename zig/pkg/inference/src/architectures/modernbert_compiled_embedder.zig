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

//! Parity-gated compiled MPSGraph forward for the frozen text embedder
//! (plain HF/nomic ModernBERT — no LoRA, rank 0).
//!
//! The eager native ModernBERT encoder forward (`modern_bert.forward`) is
//! latency-bound by 22 sequential per-op Metal dispatches, so it dominates
//! split-encoder /chunk dense + SPLADE embedding cost (~865 ms/chunk). This
//! reuses the boundary path's compiled segment executables
//! (`CompiledEvalForward.initMergedBase`, rank 0) to run the whole encoder as a
//! handful of cached MPSGraph executions — the same ~4.4x lever the boundary
//! window already gets.
//!
//! The compiled segment graph and the eager forward run different f32 kernels
//! of the same math, and the compiled graph is now parameterized to honor the
//! embedder's RoPE (`rotate_half`) and layer-0 (`nn.Identity`) conventions via
//! `modern_bert_graph.Config.rope_interleaved` / `.attn_norm0_identity` (the
//! fused chunker keeps the interleaved + layer-0-LayerNorm defaults). Because
//! served dense embeddings must stay cosine >= 0.9995 vs the eager output (to
//! protect retrieval), the compiled path is DISABLED until a startup parity
//! self-check confirms the pooled per-row cosine clears that bar on the first
//! real batch; any later compiled/head failure latches back to eager.
//!
//! Geometry: chunks are run one row at a time (batch=1) padded up to the next
//! rung of a fixed length ladder, so only a bounded set of MPSGraph geometries
//! is ever compiled (each reused across every chunk that rounds to it).

const std = @import("std");
const platform = @import("antfly_platform");
const ops_mod = @import("../ops/ops.zig");
const ComputeBackend = ops_mod.ComputeBackend;
const modern_bert = @import("modern_bert.zig");
const modern_bert_graph = @import("modern_bert_graph.zig");
const fused_chunker_compiled_forward = @import("../finetune/fused_chunker_compiled_forward.zig");
const fused_chunker_lora = @import("../finetune/lora_adapter_set.zig");
const CompiledEvalForward = fused_chunker_compiled_forward.CompiledEvalForward;

/// Kill switch. Default ON: the pipeline attempts the compiled forward, but the
/// startup parity self-check must pass before it actually serves compiled
/// features. Set to "0"/"false"/"no"/"off" to force the eager forward.
pub const env_flag = "ANTFLY_MODERNBERT_COMPILED_FORWARD";

pub fn envEnabled() bool {
    return platform.env.getenvBoolDefault(env_flag, true);
}

/// Fixed geometry ladder (in tokens). A serving `run` call rounds its sequence
/// length up to the smallest rung that fits and runs every row batch=1 at that
/// rung, so at most `bucket_ladder.len` MPSGraph geometries are ever compiled.
/// Split-encoder chunk batches span ~33..512 tokens; sequences longer than the
/// last rung fall back to the eager forward.
const bucket_ladder = [_]usize{ 64, 128, 192, 256, 320, 384, 448, 512 };

/// The hard parity gate: served dense/SPLADE embeddings must stay at least this
/// cosine-similar to the eager output (protects the 0.335 retrieval).
pub const parity_cosine_min: f32 = 0.9995;

/// Activation phase. `unchecked` runs eager and self-checks on the first batch;
/// `active` serves compiled features; `disabled` permanently serves eager.
pub const Phase = enum { unchecked, active, disabled };

pub const ModernBertCompiledForward = struct {
    allocator: std.mem.Allocator,
    bert_config: modern_bert.Config,
    graph_config: modern_bert_graph.Config,
    /// One lazily-built merged-base compiled forward per ladder rung.
    sessions: [bucket_ladder.len]?CompiledEvalForward = [_]?CompiledEvalForward{null} ** bucket_ladder.len,
    phase: Phase = .unchecked,
    /// Serializes lazy compile, GPU submission, and phase transitions.
    lock: std.Thread.Mutex = .{},
    /// Latched parity result for logging/introspection.
    checked_cosine: f32 = 0,

    pub fn create(
        allocator: std.mem.Allocator,
        bert_config: modern_bert.Config,
    ) !*ModernBertCompiledForward {
        const graph_config = try fused_chunker_compiled_forward.graphConfigFromBertConfig(bert_config);
        const self = try allocator.create(ModernBertCompiledForward);
        self.* = .{
            .allocator = allocator,
            .bert_config = bert_config,
            .graph_config = graph_config,
        };
        return self;
    }

    pub fn deinit(self: *ModernBertCompiledForward) void {
        for (&self.sessions) |*slot| {
            if (slot.*) |*cef| cef.deinit();
        }
        const allocator = self.allocator;
        allocator.destroy(self);
    }

    pub fn currentPhase(self: *ModernBertCompiledForward) Phase {
        self.lock.lock();
        defer self.lock.unlock();
        return self.phase;
    }

    fn ladderIndex(seq_len: usize) ?usize {
        for (bucket_ladder, 0..) |rung, i| {
            if (seq_len <= rung) return i;
        }
        return null;
    }

    /// Build (once) and return the merged-base compiled forward for one ladder
    /// rung. Must be called with `self.lock` held.
    fn ensureSession(self: *ModernBertCompiledForward, rung_idx: usize) !*CompiledEvalForward {
        const slot = &self.sessions[rung_idx];
        if (slot.* == null) {
            var cef = try CompiledEvalForward.initMergedBase(self.allocator, self.graph_config, 1);
            errdefer cef.deinit();
            try cef.warmup(1, bucket_ladder[rung_idx]);
            slot.* = cef;
        }
        return &slot.*.?;
    }

    /// Run the encoder forward for `[batch, seq_len]` through the compiled
    /// segments, returning an owned `[batch * seq_len * hidden]` last-hidden
    /// buffer (same shape/semantics as `modern_bert.forward`), or null when the
    /// caller should use the eager forward instead (disabled, sequence longer
    /// than the ladder, or a first compiled/execution failure that latches
    /// `disabled`). Locks internally.
    /// `result_allocator` owns the returned buffer (the caller frees it), and
    /// must be the same allocator the eager fallback path uses so the two are
    /// interchangeable. Internal scratch uses the session allocator.
    pub fn tryForward(
        self: *ModernBertCompiledForward,
        result_allocator: std.mem.Allocator,
        cb: *const ComputeBackend,
        ids_i64: []const i64,
        mask_i64: []const i64,
        batch: usize,
        seq_len: usize,
    ) !?[]f32 {
        self.lock.lock();
        defer self.lock.unlock();
        if (self.phase == .disabled) return null;
        return self.forwardLocked(result_allocator, cb, ids_i64, mask_i64, batch, seq_len);
    }

    /// Core compiled forward, assuming `self.lock` is held. On any compiled or
    /// execution error latches `phase = .disabled` (logged once) and returns
    /// null so the caller falls back to eager. Returns null (without latching)
    /// when the sequence exceeds the ladder — that batch simply runs eager.
    fn forwardLocked(
        self: *ModernBertCompiledForward,
        result_allocator: std.mem.Allocator,
        cb: *const ComputeBackend,
        ids_i64: []const i64,
        mask_i64: []const i64,
        batch: usize,
        seq_len: usize,
    ) !?[]f32 {
        const H: usize = @intCast(self.bert_config.hidden_size);
        if (ids_i64.len != batch * seq_len or mask_i64.len != batch * seq_len) return null;
        const rung_idx = ladderIndex(seq_len) orelse return null;
        const W = bucket_ladder[rung_idx];

        const cef = self.ensureSession(rung_idx) catch |err| {
            self.phase = .disabled;
            std.log.warn(
                "modernbert compiled forward: segment compile failed ({s}); using eager forward",
                .{@errorName(err)},
            );
            return null;
        };

        const out = try result_allocator.alloc(f32, batch * seq_len * H);
        errdefer result_allocator.free(out);
        @memset(out, 0);

        const pad_ids = try self.allocator.alloc(i64, W);
        defer self.allocator.free(pad_ids);
        const pad_mask = try self.allocator.alloc(i64, W);
        defer self.allocator.free(pad_mask);

        var no_lora: [0]fused_chunker_lora.LoRALayer = .{};

        for (0..batch) |b| {
            const row_off = b * seq_len;
            @memcpy(pad_ids[0..seq_len], ids_i64[row_off .. row_off + seq_len]);
            @memcpy(pad_mask[0..seq_len], mask_i64[row_off .. row_off + seq_len]);
            // Positions [seq_len, W) are attention-masked padding: masked out of
            // attention (bias -1e9) and never pooled, so their id is irrelevant.
            @memset(pad_ids[seq_len..W], 0);
            @memset(pad_mask[seq_len..W], 0);

            const hidden = cef.forward(
                cb,
                self.bert_config,
                pad_ids,
                pad_mask,
                1,
                W,
                &no_lora,
            ) catch |err| {
                self.phase = .disabled;
                std.log.warn(
                    "modernbert compiled forward: execution failed ({s}); using eager forward",
                    .{@errorName(err)},
                );
                return null;
            };
            defer self.allocator.free(hidden);
            // W >= seq_len, so the first seq_len rows carry every real token.
            @memcpy(out[row_off * H .. (row_off + seq_len) * H], hidden[0 .. seq_len * H]);
        }

        return out;
    }

    /// First-batch parity self-check. Runs the compiled forward for the same
    /// `[batch, seq_len]` input the caller just ran eagerly, compares the mean-
    /// pooled normalized per-row vectors (the served embedding), and latches
    /// `active` iff the minimum cosine clears `parity_cosine_min`; otherwise
    /// latches `disabled`. A sequence longer than the ladder leaves the phase
    /// `unchecked` so a later in-range batch can self-check. Locks internally.
    pub fn recordSelfCheck(
        self: *ModernBertCompiledForward,
        cb: *const ComputeBackend,
        ids_i64: []const i64,
        mask_i64: []const i64,
        batch: usize,
        seq_len: usize,
        eager_hidden: []const f32,
    ) void {
        self.lock.lock();
        defer self.lock.unlock();
        if (self.phase != .unchecked) return;
        if (batch == 0 or seq_len == 0) return;

        const H: usize = @intCast(self.bert_config.hidden_size);
        if (eager_hidden.len != batch * seq_len * H) return;

        const compiled = self.forwardLocked(self.allocator, cb, ids_i64, mask_i64, batch, seq_len) catch {
            // forwardLocked already latched disabled on a hard failure.
            return;
        } orelse {
            // Sequence out of ladder range or shape mismatch: retry next batch.
            return;
        };
        defer self.allocator.free(compiled);

        var min_cos: f32 = 1.0;
        var sum_cos: f64 = 0;
        var rows: usize = 0;
        var pooled_eager = self.allocator.alloc(f32, H) catch return;
        defer self.allocator.free(pooled_eager);
        var pooled_comp = self.allocator.alloc(f32, H) catch return;
        defer self.allocator.free(pooled_comp);

        for (0..batch) |b| {
            const cos = rowPooledCosine(
                eager_hidden,
                compiled,
                mask_i64,
                b,
                seq_len,
                H,
                pooled_eager,
                pooled_comp,
            ) orelse continue;
            min_cos = @min(min_cos, cos);
            sum_cos += cos;
            rows += 1;
        }

        if (rows == 0) return; // no active rows to judge; retry next batch.

        self.checked_cosine = min_cos;
        const mean_cos = sum_cos / @as(f64, @floatFromInt(rows));
        if (min_cos >= parity_cosine_min) {
            self.phase = .active;
            std.log.info(
                "modernbert compiled forward: parity self-check PASSED (min cosine {d:.6}, mean {d:.6} over {d} rows); serving compiled encoder (set {s}=0 to disable)",
                .{ min_cos, mean_cos, rows, env_flag },
            );
        } else {
            self.phase = .disabled;
            std.log.warn(
                "modernbert compiled forward: parity self-check FAILED (min cosine {d:.6} < {d:.6}); keeping eager forward",
                .{ min_cos, parity_cosine_min },
            );
        }
    }
};

/// Mean-pool one row's active (mask==1) token hidden states, L2-normalize both
/// eager and compiled pooled vectors, and return their cosine. Returns null
/// when the row has no active tokens or a degenerate (zero-norm) vector.
fn rowPooledCosine(
    eager_hidden: []const f32,
    compiled_hidden: []const f32,
    mask_i64: []const i64,
    row: usize,
    seq_len: usize,
    H: usize,
    pooled_eager: []f32,
    pooled_comp: []f32,
) ?f32 {
    @memset(pooled_eager, 0);
    @memset(pooled_comp, 0);
    var n: usize = 0;
    for (0..seq_len) |s| {
        if (mask_i64[row * seq_len + s] == 0) continue;
        const base = (row * seq_len + s) * H;
        for (0..H) |h| {
            pooled_eager[h] += eager_hidden[base + h];
            pooled_comp[h] += compiled_hidden[base + h];
        }
        n += 1;
    }
    if (n == 0) return null;

    var dot: f64 = 0;
    var ne: f64 = 0;
    var nc: f64 = 0;
    for (0..H) |h| {
        const e: f64 = pooled_eager[h];
        const c: f64 = pooled_comp[h];
        dot += e * c;
        ne += e * e;
        nc += c * c;
    }
    if (ne <= 0 or nc <= 0) return null;
    return @floatCast(dot / (@sqrt(ne) * @sqrt(nc)));
}

test "ladderIndex rounds up to the smallest fitting rung" {
    try std.testing.expectEqual(@as(?usize, 0), ModernBertCompiledForward.ladderIndex(1));
    try std.testing.expectEqual(@as(?usize, 0), ModernBertCompiledForward.ladderIndex(64));
    try std.testing.expectEqual(@as(?usize, 1), ModernBertCompiledForward.ladderIndex(65));
    try std.testing.expectEqual(@as(?usize, 5), ModernBertCompiledForward.ladderIndex(384));
    try std.testing.expectEqual(@as(?usize, 7), ModernBertCompiledForward.ladderIndex(512));
    try std.testing.expectEqual(@as(?usize, null), ModernBertCompiledForward.ladderIndex(513));
}

test "rowPooledCosine is 1.0 for identical rows and ignores masked tokens" {
    const H = 2;
    const seq_len = 3;
    // Row 0: two active tokens + one masked; compiled == eager on active tokens,
    // divergent on the masked token (must be ignored).
    const eager = [_]f32{ 1, 0, 0, 1, 9, 9 };
    const compiled = [_]f32{ 1, 0, 0, 1, -5, -5 };
    const mask = [_]i64{ 1, 1, 0 };
    var pe: [H]f32 = undefined;
    var pc: [H]f32 = undefined;
    const cos = rowPooledCosine(&eager, &compiled, &mask, 0, seq_len, H, &pe, &pc) orelse
        return error.NoCosine;
    try std.testing.expect(cos > 0.99999);
}
