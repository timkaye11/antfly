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
//! Geometry: chunks are run through compiled MPSGraph executions keyed by a
//! two-axis ladder — a sequence-length ladder (padding each row up to the next
//! rung) crossed with a batch-size ladder (packing many rows into one padded
//! batched execution). A doc's chunks are grouped by seq rung and run in
//! batched groups (rows beyond the real count are attention-masked padding),
//! replacing one GPU round-trip per chunk with one per group. Geometries build
//! lazily and are bounded by capping the max batch rung, looping the remainder,
//! and refusing new geometries past an RSS ceiling. Batching is gated by its
//! own startup self-check (batched output must match eager within the same
//! cosine bar); on failure the path stays row-wise (batch=1), and
//! ANTFLY_MODERNBERT_COMPILED_BATCHED=0 forces row-wise for A/B.

const std = @import("std");
const builtin = @import("builtin");
const platform = @import("antfly_platform");
const ops_mod = @import("../ops/ops.zig");
const ComputeBackend = ops_mod.ComputeBackend;
const modern_bert = @import("modern_bert.zig");
const modern_bert_graph = @import("modern_bert_graph.zig");
const fused_chunker_compiled_forward = @import("../finetune/fused_chunker_compiled_forward.zig");
const fused_chunker_lora = @import("../finetune/lora_adapter_set.zig");
const CompiledEvalForward = fused_chunker_compiled_forward.CompiledEvalForward;

/// This Zig fork's `std.atomic.Mutex` exposes only `tryLock`/`unlock`; callers
/// spin (matching pipelines/fused_chunking.zig and server/model_manager.zig).
fn spinLock(m: *std.atomic.Mutex) void {
    while (!m.tryLock()) {
        std.atomic.spinLoopHint();
    }
}

/// Kill switch. Default ON: the pipeline attempts the compiled forward, but the
/// startup parity self-check must pass before it actually serves compiled
/// features. Set to "0"/"false"/"no"/"off" to force the eager forward.
pub const env_flag = "ANTFLY_MODERNBERT_COMPILED_FORWARD";

pub fn envEnabled() bool {
    return platform.env.getenvBoolDefault(env_flag, true);
}

/// Fixed geometry ladder (in tokens). A serving `run` call rounds its sequence
/// length up to the smallest rung that fits and runs every row batch=1 at that
/// rung, so at most `bucket_ladder.len` MPSGraph geometries are ever compiled
/// (rungs build lazily, only as chunk lengths demand). Fused-chunker chunk
/// token spans are capped at 512, and the nomic `search_document: ` prefix +
/// [CLS]/[SEP] push the max embedder sequence to ~517, so the 576 rung is the
/// one the largest chunks land on. The 768/1024 rungs give headroom for raw
/// (non-chunk) embedding of longer texts; anything past the last rung falls
/// back to the eager forward.
const bucket_ladder = [_]usize{ 128, 256, 384, 512, 576, 768, 1024 };

/// Fixed BATCH-size ladder. For N sequences of a given seq rung, rows are
/// processed in groups of at most the largest rung (`max_batch_rung`); each
/// group's real-row count is rounded UP to the smallest batch rung that fits
/// and the padding rows are attention-masked (mask==0) so they never affect a
/// real row's pooled output (attention is per-row; RoPE is position-absolute).
/// One compiled MPSGraph `(batch_rung, seq_rung)` geometry runs the whole
/// padded group, replacing `batch_rung` sequential batch=1 executions. Capping
/// at 16 and looping the remainder bounds the number of distinct geometries
/// (RSS) — a 25-chunk doc builds only the (16, rung) geometry plus one small
/// tail rung, not one geometry per chunk count.
const batch_ladder = [_]usize{ 1, 2, 4, 8, 16 };
const max_batch_rung = batch_ladder[batch_ladder.len - 1];

/// Kill switch for the batched compiled forward (default ON). When off (or when
/// the startup batched-vs-rowwise self-check fails), the compiled path runs one
/// row per MPSGraph execution (the prior behavior) — still compiled, still
/// parity-gated vs eager, just not batched. This is the A/B toggle used to
/// compare batched vs batch=1 latency on one binary.
pub const batched_env_flag = "ANTFLY_MODERNBERT_COMPILED_BATCHED";

pub fn batchedEnvEnabled() bool {
    return platform.env.getenvBoolDefault(batched_env_flag, true);
}

/// Peak-RSS ceiling (bytes). Before compiling a NEW (batch, seq) geometry the
/// forward checks peak RSS; at/above this it refuses to cache more geometries
/// and the batch falls back to the eager forward for that call (compiled
/// geometries already built keep serving). Env override in whole GiB via
/// `ANTFLY_MODERNBERT_COMPILED_RSS_CEIL_GIB`. 24 GiB machine → ~18 GiB default.
pub const rss_ceiling_env_flag = "ANTFLY_MODERNBERT_COMPILED_RSS_CEIL_GIB";
const default_rss_ceiling_gib: usize = 18;

fn rssCeilingBytes() usize {
    const gib = platform.env.getenvUsize(rss_ceiling_env_flag) orelse default_rss_ceiling_gib;
    return gib * 1024 * 1024 * 1024;
}

/// Process peak resident set size in bytes (getrusage max; Darwin reports
/// bytes, Linux KiB). Used as a conservative — peak only grows — geometry-cache
/// admission guard.
fn peakRssBytes() usize {
    const usage = std.posix.getrusage(std.posix.rusage.SELF);
    if (usage.maxrss <= 0) return 0;
    const maxrss: usize = @intCast(usage.maxrss);
    return switch (builtin.os.tag) {
        .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => maxrss,
        .linux => std.math.mul(usize, maxrss, 1024) catch std.math.maxInt(usize),
        else => maxrss,
    };
}

/// The hard parity gate: served dense/SPLADE embeddings must stay at least this
/// cosine-similar to the eager output (protects the 0.335 retrieval).
pub const parity_cosine_min: f32 = 0.9995;

/// Activation phase. `unchecked` runs eager and self-checks on the first batch;
/// `active` serves compiled features; `disabled` permanently serves eager.
pub const Phase = enum { unchecked, active, disabled };

/// Batching mode within the compiled path, resolved by the startup self-check.
/// `batched` runs padded (batch_rung, seq_rung) geometries; `rowwise` runs one
/// row per execution (the pre-batching behavior); `unchecked` before the first
/// self-check. When batched output diverges from the row-wise output the
/// self-check latches `rowwise` — "if batching can't hit parity, stay batch=1".
pub const BatchMode = enum { unchecked, batched, rowwise };

pub const ModernBertCompiledForward = struct {
    allocator: std.mem.Allocator,
    bert_config: modern_bert.Config,
    graph_config: modern_bert_graph.Config,
    /// One lazily-built merged-base compiled forward per (batch rung, seq rung)
    /// geometry. Row [0] (batch rung 1) is the row-wise / fallback geometry set;
    /// higher rows are the batched geometries, built only as documents demand.
    sessions: [batch_ladder.len][bucket_ladder.len]?CompiledEvalForward =
        [_][bucket_ladder.len]?CompiledEvalForward{[_]?CompiledEvalForward{null} ** bucket_ladder.len} ** batch_ladder.len,
    phase: Phase = .unchecked,
    /// Whether the compiled path batches; resolved by the self-check.
    batch_mode: BatchMode = .unchecked,
    /// Serializes lazy compile, GPU submission, and phase transitions.
    lock: std.atomic.Mutex = .unlocked,
    /// Latched parity result for logging/introspection.
    checked_cosine: f32 = 0,
    /// Latched batched-vs-rowwise agreement (min pooled cosine) from the
    /// self-check, for logging/introspection.
    checked_batch_cosine: f32 = 0,
    /// Set once if a geometry compile was ever skipped for the RSS ceiling.
    rss_ceiling_hit: bool = false,

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
        for (&self.sessions) |*row| {
            for (row) |*slot| {
                if (slot.*) |*cef| cef.deinit();
            }
        }
        const allocator = self.allocator;
        allocator.destroy(self);
    }

    pub fn currentPhase(self: *ModernBertCompiledForward) Phase {
        spinLock(&self.lock);
        defer self.lock.unlock();
        return self.phase;
    }

    fn ladderIndex(seq_len: usize) ?usize {
        for (bucket_ladder, 0..) |rung, i| {
            if (seq_len <= rung) return i;
        }
        return null;
    }

    /// Smallest batch rung that fits `n` real rows (`n <= max_batch_rung`).
    fn batchLadderIndex(n: usize) usize {
        for (batch_ladder, 0..) |rung, i| {
            if (n <= rung) return i;
        }
        return batch_ladder.len - 1;
    }

    /// Build (once) and return the merged-base compiled forward for one
    /// (batch rung, seq rung) geometry. Must be called with `self.lock` held.
    /// Refuses to compile a NEW geometry once peak RSS reaches the ceiling
    /// (returns error.RssCeiling; the caller degrades that batch to eager).
    fn ensureSession(self: *ModernBertCompiledForward, batch_idx: usize, rung_idx: usize) !*CompiledEvalForward {
        const slot = &self.sessions[batch_idx][rung_idx];
        if (slot.* == null) {
            if (peakRssBytes() >= rssCeilingBytes()) {
                self.rss_ceiling_hit = true;
                return error.RssCeiling;
            }
            var cef = try CompiledEvalForward.initMergedBase(
                self.allocator,
                self.graph_config,
                fused_chunker_compiled_forward.servingLayersPerSegment(),
            );
            errdefer cef.deinit();
            try cef.warmup(batch_ladder[batch_idx], bucket_ladder[rung_idx]);
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
        spinLock(&self.lock);
        defer self.lock.unlock();
        if (self.phase == .disabled) return null;
        return self.forwardLocked(result_allocator, cb, ids_i64, mask_i64, batch, seq_len);
    }

    /// Core compiled forward, assuming `self.lock` is held. Dispatches to the
    /// batched or row-wise implementation per `self.batch_mode` (`.unchecked`
    /// runs row-wise, matching the pre-batching behavior the self-check
    /// compares against). On any compiled or execution error latches
    /// `phase = .disabled` (logged once) and returns null so the caller falls
    /// back to eager. Returns null (without latching) when the sequence exceeds
    /// the ladder — that batch simply runs eager.
    fn forwardLocked(
        self: *ModernBertCompiledForward,
        result_allocator: std.mem.Allocator,
        cb: *const ComputeBackend,
        ids_i64: []const i64,
        mask_i64: []const i64,
        batch: usize,
        seq_len: usize,
    ) !?[]f32 {
        if (ids_i64.len != batch * seq_len or mask_i64.len != batch * seq_len) return null;
        if (self.batch_mode == .batched) {
            return self.forwardBatched(result_allocator, cb, ids_i64, mask_i64, batch, seq_len);
        }
        return self.forwardRowwise(result_allocator, cb, ids_i64, mask_i64, batch, seq_len);
    }

    /// One MPSGraph execution per row, each padded up to seq rung `W` at batch
    /// rung 1. The pre-batching behavior; also the parity/latency baseline the
    /// `ANTFLY_MODERNBERT_COMPILED_BATCHED=0` toggle selects.
    fn forwardRowwise(
        self: *ModernBertCompiledForward,
        result_allocator: std.mem.Allocator,
        cb: *const ComputeBackend,
        ids_i64: []const i64,
        mask_i64: []const i64,
        batch: usize,
        seq_len: usize,
    ) !?[]f32 {
        const H: usize = @intCast(self.bert_config.hidden_size);
        const rung_idx = ladderIndex(seq_len) orelse return null;
        const W = bucket_ladder[rung_idx];

        const cef = self.ensureSession(0, rung_idx) catch |err| {
            if (err == error.RssCeiling) return null;
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
                result_allocator.free(out);
                return null;
            };
            defer self.allocator.free(hidden);
            // W >= seq_len, so the first seq_len rows carry every real token.
            @memcpy(out[row_off * H .. (row_off + seq_len) * H], hidden[0 .. seq_len * H]);
        }

        return out;
    }

    /// Batched compiled forward: rows are processed in groups of at most
    /// `max_batch_rung`; each group's real-row count `g` rounds UP to batch rung
    /// `B`, with rows `[g, B)` attention-masked padding. One `(B, W)` MPSGraph
    /// execution replaces `B` row-wise executions. Padding rows never affect a
    /// real row (attention is per-row, RoPE is position-absolute). On the RSS
    /// ceiling (would compile a new geometry) returns null so the caller runs
    /// this batch eager; on a hard compile/exec error latches `disabled`.
    fn forwardBatched(
        self: *ModernBertCompiledForward,
        result_allocator: std.mem.Allocator,
        cb: *const ComputeBackend,
        ids_i64: []const i64,
        mask_i64: []const i64,
        batch: usize,
        seq_len: usize,
    ) !?[]f32 {
        const H: usize = @intCast(self.bert_config.hidden_size);
        const rung_idx = ladderIndex(seq_len) orelse return null;
        const W = bucket_ladder[rung_idx];

        const out = try result_allocator.alloc(f32, batch * seq_len * H);
        errdefer result_allocator.free(out);
        @memset(out, 0);

        // Reusable padded (batch rung x W) id/mask scratch, sized for the
        // largest group; each group uses only its leading B*W entries.
        const pad_ids = try self.allocator.alloc(i64, max_batch_rung * W);
        defer self.allocator.free(pad_ids);
        const pad_mask = try self.allocator.alloc(i64, max_batch_rung * W);
        defer self.allocator.free(pad_mask);

        var no_lora: [0]fused_chunker_lora.LoRALayer = .{};

        var row: usize = 0;
        while (row < batch) {
            const g = @min(batch - row, max_batch_rung); // real rows this exec
            const batch_idx = batchLadderIndex(g);
            const B = batch_ladder[batch_idx]; // padded batch rung (>= g)

            // Pack g real rows (each padded seq_len -> W); zero the rest.
            for (0..g) |r| {
                const src = (row + r) * seq_len;
                const dst = r * W;
                @memcpy(pad_ids[dst .. dst + seq_len], ids_i64[src .. src + seq_len]);
                @memcpy(pad_mask[dst .. dst + seq_len], mask_i64[src .. src + seq_len]);
                @memset(pad_ids[dst + seq_len .. dst + W], 0);
                @memset(pad_mask[dst + seq_len .. dst + W], 0);
            }
            @memset(pad_ids[g * W .. B * W], 0);
            @memset(pad_mask[g * W .. B * W], 0);

            const cef = self.ensureSession(batch_idx, rung_idx) catch |err| {
                if (err == error.RssCeiling) {
                    // Refuse to grow the geometry cache: run this batch eager.
                    result_allocator.free(out);
                    return null;
                }
                self.phase = .disabled;
                std.log.warn(
                    "modernbert compiled forward: batched segment compile failed ({s}); using eager forward",
                    .{@errorName(err)},
                );
                result_allocator.free(out);
                return null;
            };

            const hidden = cef.forward(
                cb,
                self.bert_config,
                pad_ids[0 .. B * W],
                pad_mask[0 .. B * W],
                B,
                W,
                &no_lora,
            ) catch |err| {
                self.phase = .disabled;
                std.log.warn(
                    "modernbert compiled forward: batched execution failed ({s}); using eager forward",
                    .{@errorName(err)},
                );
                result_allocator.free(out);
                return null;
            };
            defer self.allocator.free(hidden);

            // Scatter the g real rows' first seq_len tokens back into `out`.
            for (0..g) |r| {
                const dst = (row + r) * seq_len * H;
                const srch = r * W * H;
                @memcpy(out[dst .. dst + seq_len * H], hidden[srch .. srch + seq_len * H]);
            }
            row += g;
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
        spinLock(&self.lock);
        defer self.lock.unlock();
        if (self.phase != .unchecked) return;
        if (batch == 0 or seq_len == 0) return;

        const H: usize = @intCast(self.bert_config.hidden_size);
        if (eager_hidden.len != batch * seq_len * H) return;

        const pooled_a = self.allocator.alloc(f32, H) catch return;
        defer self.allocator.free(pooled_a);
        const pooled_b = self.allocator.alloc(f32, H) catch return;
        defer self.allocator.free(pooled_b);

        // Stage 1: row-wise compiled vs eager — the retrieval parity gate.
        const rowwise = self.forwardRowwise(self.allocator, cb, ids_i64, mask_i64, batch, seq_len) catch {
            // forwardRowwise already latched disabled on a hard failure.
            return;
        } orelse {
            // Sequence out of ladder range / RSS ceiling: retry next batch.
            return;
        };
        defer self.allocator.free(rowwise);

        const rowwise_stats = minPooledCosine(eager_hidden, rowwise, mask_i64, batch, seq_len, H, pooled_a, pooled_b) orelse
            return; // no active rows to judge; retry next batch.
        self.checked_cosine = rowwise_stats.min;
        if (rowwise_stats.min < parity_cosine_min) {
            self.phase = .disabled;
            std.log.warn(
                "modernbert compiled forward: parity self-check FAILED (min cosine {d:.6} < {d:.6}); keeping eager forward",
                .{ rowwise_stats.min, parity_cosine_min },
            );
            return;
        }
        self.phase = .active;

        // Stage 2: resolve the batch mode. Batching is served only if the
        // padded batched geometry stays within the same retrieval gate vs
        // eager; otherwise it falls back to the (already-passing) row-wise path.
        if (!batchedEnvEnabled()) {
            self.batch_mode = .rowwise;
            std.log.info(
                "modernbert compiled forward: parity self-check PASSED (min cosine {d:.6}, mean {d:.6} over {d} rows); serving compiled encoder ROW-WISE (batching off via {s}; set {s}=0 to disable compiled)",
                .{ rowwise_stats.min, rowwise_stats.mean, rowwise_stats.rows, batched_env_flag, env_flag },
            );
            return;
        }

        const batched = self.forwardBatched(self.allocator, cb, ids_i64, mask_i64, batch, seq_len) catch null;
        if (batched) |bh| {
            defer self.allocator.free(bh);
            // forwardBatched may have latched disabled on a hard error before
            // returning a (stale) buffer path; it does not, but restore active
            // defensively since row-wise already validated the compiled path.
            self.phase = .active;
            const be = minPooledCosine(eager_hidden, bh, mask_i64, batch, seq_len, H, pooled_a, pooled_b);
            const bvr = minPooledCosine(rowwise, bh, mask_i64, batch, seq_len, H, pooled_a, pooled_b);
            if (bvr) |s| self.checked_batch_cosine = s.min;
            if (be != null and be.?.min >= parity_cosine_min) {
                self.batch_mode = .batched;
                std.log.info(
                    "modernbert compiled forward: parity self-check PASSED (row-wise min cosine {d:.6} over {d} rows); serving compiled encoder BATCHED (batched-vs-eager min {d:.6}, batched-vs-rowwise min {d:.6}; batch ladder up to {d}; set {s}=0 to force row-wise, {s}=0 to disable compiled)",
                    .{ rowwise_stats.min, rowwise_stats.rows, be.?.min, self.checked_batch_cosine, max_batch_rung, batched_env_flag, env_flag },
                );
                return;
            }
            self.batch_mode = .rowwise;
            std.log.warn(
                "modernbert compiled forward: BATCHED self-check below gate (batched-vs-eager min {d:.6} < {d:.6}); serving compiled encoder ROW-WISE",
                .{ if (be) |s| s.min else 0.0, parity_cosine_min },
            );
            return;
        }
        // Batched unavailable (out-of-ladder / RSS ceiling / error): row-wise.
        self.phase = .active;
        self.batch_mode = .rowwise;
        std.log.info(
            "modernbert compiled forward: parity self-check PASSED (min cosine {d:.6}, mean {d:.6} over {d} rows); serving compiled encoder ROW-WISE (batched geometry unavailable)",
            .{ rowwise_stats.min, rowwise_stats.mean, rowwise_stats.rows },
        );
    }
};

const PooledCosineStats = struct { min: f32, mean: f64, rows: usize };

/// Min/mean pooled per-row cosine between two `[batch, seq_len, H]` hidden
/// buffers over active (mask==1) tokens. Returns null when no row has active
/// tokens. `pooled_a`/`pooled_b` are `[H]` scratch (overwritten).
fn minPooledCosine(
    a_hidden: []const f32,
    b_hidden: []const f32,
    mask_i64: []const i64,
    batch: usize,
    seq_len: usize,
    H: usize,
    pooled_a: []f32,
    pooled_b: []f32,
) ?PooledCosineStats {
    var min_cos: f32 = 1.0;
    var sum_cos: f64 = 0;
    var rows: usize = 0;
    for (0..batch) |b| {
        const cos = rowPooledCosine(a_hidden, b_hidden, mask_i64, b, seq_len, H, pooled_a, pooled_b) orelse continue;
        min_cos = @min(min_cos, cos);
        sum_cos += cos;
        rows += 1;
    }
    if (rows == 0) return null;
    return .{ .min = min_cos, .mean = sum_cos / @as(f64, @floatFromInt(rows)), .rows = rows };
}

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
    try std.testing.expectEqual(@as(?usize, 0), ModernBertCompiledForward.ladderIndex(128));
    try std.testing.expectEqual(@as(?usize, 1), ModernBertCompiledForward.ladderIndex(129));
    try std.testing.expectEqual(@as(?usize, 2), ModernBertCompiledForward.ladderIndex(384));
    try std.testing.expectEqual(@as(?usize, 3), ModernBertCompiledForward.ladderIndex(512));
    // 512-token chunks + prefix/specials (~517) round up to the 576 rung.
    try std.testing.expectEqual(@as(?usize, 4), ModernBertCompiledForward.ladderIndex(517));
    try std.testing.expectEqual(@as(?usize, 6), ModernBertCompiledForward.ladderIndex(1024));
    try std.testing.expectEqual(@as(?usize, null), ModernBertCompiledForward.ladderIndex(1025));
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
