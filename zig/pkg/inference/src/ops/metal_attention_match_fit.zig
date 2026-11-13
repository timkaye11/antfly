// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
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

//! Bounded Metal P/Y acceleration for the common checkpoint compaction engine.
//! Metal's FP32 attention frontend feeds shared FP64 OMP/NNLS with bounded
//! Accelerate Householder solves. The native direct KKT and FP64 publication
//! validation remain unchanged.
//! No CPU attention fallback or CUDA context is used by this adapter.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("antfly_platform");
const shared = @import("../backends/metal_generated_kernels.zig");
const solver = @import("../runtime/kv/attention_match.zig");
const training_source = @import("../runtime/kv/attention_match_joint_training_source.zig");
const train_challenge = @import("../runtime/kv/attention_match_train_challenge_historical_selection.zig");
const engine = @import("../runtime/kv/attention_match_engine.zig");
const fp64 = @import("metal_attention_match_lapack.zig");

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}

test "captured page source fails every retained shared allocation and restores tracker" {
    if (!@import("build_options").enable_metal) return error.SkipZigTest;
    if (!@import("antfly_platform").env.getenvBool("ANTFLY_METAL_ATTENTION_MATCH_FLOATPAIR_DIAGNOSTIC")) return error.SkipZigTest;
    const a = std.testing.allocator;
    const head: solver.Head = .{ .keys = &.{ 1, 2 }, .values = &.{ 3, 4 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    const queries = [_]f64{ 0.25, -0.5 };
    var tracker = shared.BufferAllocationTracker.init(null, 0);
    var fitter = try Fitter.init(a, &tracker, 16 * 1024 * 1024);
    defer fitter.deinit();
    const baseline = tracker.snapshot().shared_backing_live_bytes;
    // The fault seam covers the four retained K/bias/Q/pair allocations;
    // captureNormalizer's temporary buffers use the ordinary tracked path.
    for (0..4) |fail_index| {
        fitter.armTestProbabilityPageBufferFailure(fail_index);
        try std.testing.expectError(error.OutOfMemory, fitter.capturePageSource(a, head, &queries, 0, 16 * 1024 * 1024, .{}));
        try std.testing.expectEqual(baseline, tracker.snapshot().shared_backing_live_bytes);
    }
    try std.testing.expectError(error.CompactionWorkspaceLimit, fitter.capturePageSource(a, head, &queries, 16 * 1024 * 1024, 16 * 1024 * 1024, .{}));
    try std.testing.expectEqual(baseline, tracker.snapshot().shared_backing_live_bytes);
    var captured = try fitter.capturePageSource(a, head, &queries, 0, 16 * 1024 * 1024, .{});
    try std.testing.expect(tracker.snapshot().shared_backing_live_bytes > baseline);
    captured.deinit();
    try std.testing.expectEqual(baseline, tracker.snapshot().shared_backing_live_bytes);
}

fn testRawRangeIdentity(rows: usize) solver.Backend.RawRangeIdentity {
    return .{ .range_start = 7, .range_rows = rows, .query_rows = 3, .source_rows = 32, .source_identity = @splat(1), .train_identity = @splat(2), .critical_authority_identity = @splat(3), .policy_identity = @splat(4) };
}

test "range session machine rejects nested begin and closes exactly once" {
    var machine: Fitter.RangeSessionMachine = .{};
    const admitted = try Fitter.rawRangeWorkspace(null, 8);
    const token = try machine.begin(std.testing.allocator, testRawRangeIdentity(8), admitted);
    try std.testing.expectError(error.InvalidFitConfig, machine.begin(std.testing.allocator, testRawRangeIdentity(8), admitted));
    try std.testing.expect(machine.end(token));
    try std.testing.expect(!machine.end(token));
    try std.testing.expectEqual(@as(u64, 1), machine.end_count);
}

test "range session machine stale token fails closed without losing live owner" {
    var machine: Fitter.RangeSessionMachine = .{};
    const token = try machine.begin(std.testing.allocator, testRawRangeIdentity(8), try Fitter.rawRangeWorkspace(null, 8));
    var stale = token;
    stale.generation +%= 1;
    try std.testing.expect(!machine.end(stale));
    try std.testing.expect(machine.session != null);
    try std.testing.expect(machine.end(token));
    try std.testing.expectEqual(@as(u64, 1), machine.end_count);
}

test "range session machine accepts critical residual width and rejects drift" {
    var machine: Fitter.RangeSessionMachine = .{};
    const token = try machine.begin(std.testing.allocator, testRawRangeIdentity(8), try Fitter.rawRangeWorkspace(null, 8));
    defer _ = machine.end(token);
    _ = try machine.consume(token, 3, 5);
    _ = try machine.consume(token, 3, 5);
    try std.testing.expectError(error.RawRangeIdentityMismatch, machine.consume(token, 3, 6));
}

test "range session machine begin allocation failure leaves reusable idle state" {
    var machine: Fitter.RangeSessionMachine = .{};
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    const identity = testRawRangeIdentity(8);
    const admitted = try Fitter.rawRangeWorkspace(null, 8);
    try std.testing.expectError(error.OutOfMemory, machine.begin(failing.allocator(), identity, admitted));
    try std.testing.expect(machine.session == null);
    const token = try machine.begin(std.testing.allocator, identity, admitted);
    try std.testing.expect(machine.end(token));
}

test "default off null planning backend is null safe" {
    const backend = Fitter.backendForContext(null);
    try std.testing.expectEqual(solver.RawRangePolicy.off, backend.raw_range_policy);
}

test "prepared failure falls back to Householder without completed counter skew" {
    if (!fp64.supported) return error.SkipZigTest;
    var cpu: Fitter.CpuRangeTestBackend = .{};
    const backend = cpu.backend();
    const identity = testRawRangeIdentity(3);
    const admitted = try backend.raw_range_workspace_fn.?(backend.context, 3);
    const token = try backend.raw_range_begin_fn.?(backend.context, std.testing.allocator, identity, admitted);
    defer backend.raw_range_end_fn.?(backend.context, token);
    try cpu.armPreparedFailure();
    const p = [_]f64{ 1, 0, 0, 0, 1, 0, 0.25, 0.25, 1 };
    const rhs = [_]f64{ 1, 1, 1 };
    const importance = [_]f64{ 1.0625, 1.0625, 1 };
    var indices: [2]usize = undefined;
    var weights: [2]f64 = undefined;
    const actual_solves = try backend.raw_select_fn.?(backend.context, std.testing.allocator, &p, &importance, &rhs, 3, 3, &.{}, &indices, &weights, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .raw_range_policy = .range_shared_fp64_gram_original_kkt_v1, .raw_range_session = token });
    var expected_indices: [2]usize = undefined;
    var expected_weights: [2]f64 = undefined;
    const expected_solves = try fp64.selectRawTarget(null, std.testing.allocator, &p, &importance, &rhs, 3, 3, &.{}, &expected_indices, &expected_weights, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set });
    try std.testing.expectEqual(expected_solves, actual_solves);
    try std.testing.expectEqualSlices(usize, &expected_indices, &indices);
    try std.testing.expectEqualSlices(f64, &expected_weights, &weights);
    try std.testing.expectEqual(@as(u64, 0), cpu.counters.gram_advisory.calls);
    try std.testing.expectEqual(@as(u64, 1), cpu.counters.gram_advisory.gram_precomputes);
    try std.testing.expectEqual(cpu.counters.gram_advisory.candidate_attempts, cpu.counters.gram_advisory.native_original_p_kkt_checks);
    try std.testing.expectEqual(@as(u64, 0), cpu.counters.gram_advisory.native_original_p_kkt_checks);
    try std.testing.expect(cpu.machine.session.?.poisoned);
}

fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn rounded(bytes: usize) !usize {
    const page = std.heap.pageSize();
    return (try add(bytes, page - 1)) & ~(page - 1);
}
pub const Fitter = struct {
    pub const TrainMassPrecisionStats = struct {
        gpu_f32_calls: usize = 0,
        gpu_f32_query_rows: usize = 0,
        gpu_f32_workspace_peak_bytes: usize = 0,
    };
    const TrainingCapture = struct {
        owner: *Fitter,
        head: solver.Head,
        batch_rows: usize,
        keys: shared.Buffer,
        values: shared.Buffer,
        bias: shared.Buffer,
        query: shared.Buffer,
        result: shared.Buffer,
        fn deinit(self: *@This()) void {
            self.result.deinit();
            self.query.deinit();
            self.bias.deinit();
            self.values.deinit();
            self.keys.deinit();
        }
    };
    pub const Normalizer = struct {
        allocator: std.mem.Allocator,
        owner: *Fitter,
        keys: [*]const f64,
        values: [*]const f64,
        bias: ?[*]const f64,
        queries: [*]const f64,
        rows: usize,
        dim: usize,
        query_count: usize,
        scale: f64,
        softcap: f64,
        layout: KeyLayout,
        batch_rows: usize,
        pairs: []f32,
        outputs: []f32,
        pub fn retainedBytes(self: *const Normalizer) !usize {
            return mul(try add(self.pairs.len, self.outputs.len), @sizeOf(f32));
        }
        pub fn deinit(self: *Normalizer) void {
            self.allocator.free(self.pairs);
            self.allocator.free(self.outputs);
            self.* = undefined;
        }
    };
    /// Immutable captured provider used only by the bounded diagnostic route.
    /// K, bias, all queries and all FloatPairs are uploaded once. The ordinary
    /// Normalizer/replayPageQueries API remains the checked scalar authority.
    pub const CapturedPageSource = struct {
        normalizer: Normalizer,
        owner: *Fitter,
        keys: shared.Buffer,
        bias: shared.Buffer,
        queries: shared.Buffer,
        pairs: shared.Buffer,
        rows: usize,
        dim: usize,
        query_count: usize,
        layout: KeyLayout,
        batch_rows: usize,
        retained_bytes: usize,
        fingerprint: [32]u8,
        generation: u64,
        design_rows: usize = 0,

        pub fn deinit(self: *CapturedPageSource) void {
            if (self.owner.captured_page_generation == self.generation) {
                self.owner.captured_page_generation +|= 1;
                self.owner.captured_page_fingerprint = @splat(0);
            }
            self.pairs.deinit();
            self.queries.deinit();
            self.bias.deinit();
            self.keys.deinit();
            self.normalizer.deinit();
            self.* = undefined;
        }
    };
    pub const KeyLayout = enum { row_major, column_major };
    pub const RawSelectionSolver = enum { householder, fp64_gram_advisory, range_shared_fp64_gram_original_kkt_v1 };
    const RangeGramSession = struct {
        allocator: std.mem.Allocator,
        identity: solver.Backend.RawRangeIdentity,
        admitted_bytes: usize,
        generation: u64,
        design_rows: usize = 0,
        prepared: ?fp64.PreparedRawGram = null,
        poisoned: bool = false,
        test_fail_prepared_once: bool = false,
    };
    const RangeSessionMachine = struct {
        session: ?*RangeGramSession = null,
        generation: u64 = 0,
        end_count: u64 = 0,

        fn begin(self: *@This(), allocator: std.mem.Allocator, identity: solver.Backend.RawRangeIdentity, admitted_bytes: usize) !solver.Backend.RawRangeSession {
            if (self.session != null) return error.InvalidFitConfig;
            const required = try rawRangeWorkspace(null, identity.range_rows);
            if (admitted_bytes < required) return error.CompactionWorkspaceLimit;
            const owned = try allocator.create(RangeGramSession);
            errdefer allocator.destroy(owned);
            self.generation +%= 1;
            owned.* = .{ .allocator = allocator, .identity = identity, .admitted_bytes = admitted_bytes, .generation = self.generation };
            self.session = owned;
            return .{ .context = owned, .generation = owned.generation };
        }

        /// Stale and duplicate tokens never close the live generation.
        fn end(self: *@This(), token: solver.Backend.RawRangeSession) bool {
            const owned = self.session orelse return false;
            if (token.context != @as(*anyopaque, @ptrCast(owned)) or token.generation != owned.generation) return false;
            if (owned.prepared) |*prepared| prepared.deinit();
            const allocator = owned.allocator;
            self.session = null;
            allocator.destroy(owned);
            self.end_count +|= 1;
            return true;
        }

        fn consume(self: *@This(), token: solver.Backend.RawRangeSession, q: usize, n: usize) !*RangeGramSession {
            const owned = self.session orelse return error.RawRangeSessionRequired;
            if (token.context != @as(*anyopaque, @ptrCast(owned)) or token.generation != owned.generation or owned.identity.query_rows != q or n > owned.identity.range_rows) return error.RawRangeIdentityMismatch;
            if (owned.design_rows == 0) owned.design_rows = n else if (owned.design_rows != n) return error.RawRangeIdentityMismatch;
            return owned;
        }
    };
    pub const PhaseCounters = struct {
        raw_selection_solver: RawSelectionSolver = .householder,
        raw_selection_calls: u64 = 0,
        raw_selection_active_solves: u64 = 0,
        raw_range_end_rejections: u64 = 0,
        scalar_rhs_solves: u64 = 0,
        vector_rhs_solves: u64 = 0,
        max_scalar_query_rows: usize = 0,
        max_scalar_selected_rows: usize = 0,
        max_vector_query_rows: usize = 0,
        max_vector_selected_rows: usize = 0,
        max_vector_width: usize = 0,
        gram_advisory: fp64.GramAdvisoryCounters = .{},
        phase_durations: PhaseDurations = .{},
    };

    pub const PhaseDurations = struct {
        raw_selection_ns: u64 = 0,
        solve_values_ns: u64 = 0,
        attend_total_ns: u64 = 0,
        attend_allocate_ns: u64 = 0,
        attend_fill_ns: u64 = 0,
        attend_gpu_commit_wait_ns: u64 = 0,
        attend_gpu_active_ns: u64 = 0,
        attend_gpu_wait_ns: u64 = 0,
        attend_scatter_finite_ns: u64 = 0,
        raw_selection_calls: u64 = 0,
        solve_values_calls: u64 = 0,
        attend_calls: u64 = 0,
        gpu_dispatches: u64 = 0,
        training_capture_tiles: u64 = 0,
        training_capture_gpu_commit_wait_ns: u64 = 0,
        training_capture_gpu_active_ns: u64 = 0,
        training_capture_gpu_wait_ns: u64 = 0,
        clock_reads: u64 = 0,
        clock_overhead_ns: u64 = 0,
    };

    pub const CpuRangeTestBackend = if (builtin.is_test) struct {
        machine: RangeSessionMachine = .{},
        counters: PhaseCounters = .{ .raw_selection_solver = .range_shared_fp64_gram_original_kkt_v1 },
        begins: usize = 0,
        ends: usize = 0,
        consumes: usize = 0,
        q: [3]usize = @splat(0),
        n: [3]usize = @splat(0),
        p_sha256: [3][32]u8 = @splat(@splat(0)),
        rhs_sha256: [3][32]u8 = @splat(@splat(0)),

        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn rangeWorkspace(_: ?*anyopaque, rows: usize) !usize {
            return rawRangeWorkspace(null, rows);
        }
        fn begin(raw: ?*anyopaque, allocator: std.mem.Allocator, identity: solver.Backend.RawRangeIdentity, admitted: usize) !solver.Backend.RawRangeSession {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            const token = try self.machine.begin(allocator, identity, admitted);
            self.begins += 1;
            return token;
        }
        fn end(raw: ?*anyopaque, token: solver.Backend.RawRangeSession) void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            if (self.machine.end(token)) self.ends += 1;
        }
        fn select(raw: ?*anyopaque, allocator: std.mem.Allocator, p: []const f64, importance: []const f64, rhs: []const f64, q: usize, n: usize, mandatory: []const usize, indices: []usize, weights: []f64, cfg: solver.Config) !usize {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            const ordinal = self.consumes;
            if (ordinal >= self.q.len) return error.TestUnexpectedResult;
            self.q[ordinal] = q;
            self.n[ordinal] = n;
            std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(p), &self.p_sha256[ordinal], .{});
            std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(rhs), &self.rhs_sha256[ordinal], .{});
            const result = try selectRangeShared(&self.machine, &self.counters, allocator, p, importance, rhs, q, n, mandatory, indices, weights, cfg);
            self.consumes += 1;
            return result;
        }
        pub fn backend(self: *@This()) solver.Backend {
            return .{ .context = self, .workspace_fn = @This().workspace, .raw_select_fn = @This().select, .raw_range_policy = .range_shared_fp64_gram_original_kkt_v1, .raw_range_workspace_fn = @This().rangeWorkspace, .raw_range_begin_fn = @This().begin, .raw_range_end_fn = @This().end };
        }
        pub fn armPreparedFailure(self: *@This()) !void {
            const session = self.machine.session orelse return error.RawRangeSessionRequired;
            session.test_fail_prepared_once = true;
        }
    } else struct {};

    /// Largest query matrix accepted by one fitting dispatch. The production
    /// E2B global owner has forty physical consumer strata: 256 training rows
    /// per stratum produce 10,240 rows. Keep headroom for supported layouts;
    /// the solver's unchanged byte budget remains the authoritative limit.
    pub const max_query_rows: usize = 16384;
    pub const max_wide_capture_query_rows: usize = 65536;
    pub const default_query_batch_rows: usize = 16;
    pub const maximum_query_batch_rows: usize = 64;
    /// Diagnostic-only upper bound. Production policy remains 16/64 unless a
    /// typed flat-batch policy has completed admission before the first call.
    pub const diagnostic_query_batch_rows: usize = 256;

    kernels: shared.Kernels,
    max_device_bytes: usize,
    query_batch_rows: usize = default_query_batch_rows,
    key_layout: KeyLayout = .row_major,
    raw_selection_solver: RawSelectionSolver = .householder,
    raw_range_machine: RangeSessionMachine = .{},
    derived_readback_bytes: u64 = 0,
    dispatches: u64 = 0,
    attention_calls: u64 = 0,
    train_mass_precision_stats: TrainMassPrecisionStats = .{},
    normalizer_captures: u64 = 0,
    probability_page_calls: u64 = 0,
    probability_page_rows: u64 = 0,
    training_capture_allocations: u64 = 0,
    training_capture_frees: u64 = 0,
    captured_page_generation: u64 = 0,
    captured_page_fingerprint: [32]u8 = @splat(0),
    captured_page_rows: usize = 0,
    captured_page_dim: usize = 0,
    captured_page_queries: usize = 0,
    captured_page_layout: KeyLayout = .row_major,
    captured_page_batch_rows: usize = 0,
    captured_page_retained_bytes: usize = 0,
    captured_page_scale: f64 = 0,
    captured_page_softcap: f64 = 0,
    accelerated_value_solves: u64 = 0,
    phase_counters: PhaseCounters = .{},
    profile_phases: bool = false,
    // Test-only fault seam for the five shared buffers allocated by one
    // probability-page replay. Production never arms this hook.
    page_buffer_fail_index: ?usize = null,
    page_buffer_allocation_index: usize = 0,

    pub fn init(allocator: std.mem.Allocator, tracker: ?*shared.BufferAllocationTracker, max_device_bytes: usize) !Fitter {
        _ = allocator;
        if (max_device_bytes < 6 * std.heap.pageSize()) return error.CompactionWorkspaceLimit;
        return .{
            .kernels = try shared.Kernels.initSourceTracked(@embedFile("metal_attention_match_fit.metal"), "metal_attention_match_fit", tracker),
            .max_device_bytes = max_device_bytes,
        };
    }

    pub fn initWorker(allocator: std.mem.Allocator, tracker: ?*shared.BufferAllocationTracker, max_device_bytes: usize, workspace_limit: usize) !Fitter {
        if (workspace_limit == 0 or workspace_limit > max_device_bytes) return error.CompactionWorkspaceLimit;
        return init(allocator, tracker, workspace_limit);
    }

    pub fn deinit(self: *Fitter) void {
        std.debug.assert(self.raw_range_machine.session == null);
        self.kernels.deinit();
        self.* = undefined;
    }

    /// Save counters before deinit invalidates the fitter. Preparation retires
    /// workers before physical measurements and cannot inspect them afterward.
    pub fn deinitAndTakeValueSolves(self: *Fitter) u64 {
        const count = self.accelerated_value_solves;
        self.deinit();
        return count;
    }

    /// Diagnostic-only phase split. It does not participate in fitting or
    /// authorization and remains valid until `deinitAndTakeValueSolves`.
    pub fn dispatchCount(self: *const Fitter) u64 {
        return self.dispatches;
    }
    pub fn attentionCallCount(self: *const Fitter) u64 {
        return self.attention_calls;
    }

    pub fn trainMassPrecisionStats(self: *const Fitter) TrainMassPrecisionStats {
        return self.train_mass_precision_stats;
    }

    fn trainMassWorkspace(raw: ?*anyopaque, rows: usize, queries: usize, dim: usize) !usize {
        const self: *Fitter = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
        return workspace(self, rows, 1, queries, dim);
    }

    fn trainMassGpuF32(raw: ?*anyopaque, _: std.mem.Allocator, head: solver.Head, queries: []const f64, masses: []f64, control: solver.Control) !void {
        const self: *Fitter = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
        if (queries.len == 0 or queries.len % head.dim != 0 or masses.len != queries.len / head.dim) return error.InvalidAttentionShape;
        const query_rows = queries.len / head.dim;
        const admitted = try trainMassWorkspace(self, head.rows, query_rows, head.dim);
        try attendImpl(self, head, queries, null, null, masses, null, null, control);
        self.train_mass_precision_stats.gpu_f32_calls +|= 1;
        self.train_mass_precision_stats.gpu_f32_query_rows +|= query_rows;
        self.train_mass_precision_stats.gpu_f32_workspace_peak_bytes = @max(self.train_mass_precision_stats.gpu_f32_workspace_peak_bytes, admitted);
    }

    /// Explicit pre-evaluation TRAIN-only mass provider. The returned f64
    /// values are widened Metal f32 results; this is never an FP64 validator.
    pub fn trainMassGpuF32Backend(self: *Fitter) train_challenge.MassBackend {
        return .{ .context = self, .workspace_fn = trainMassWorkspace, .masses_fn = trainMassGpuF32 };
    }

    pub const TrainingCaptureCounters = struct {
        allocations: u64,
        frees: u64,
    };

    pub fn trainingCaptureCounters(self: *const Fitter) TrainingCaptureCounters {
        return .{ .allocations = self.training_capture_allocations, .frees = self.training_capture_frees };
    }

    pub const ProbabilityPageCounters = struct {
        normalizer_captures: u64,
        calls: u64,
        rows: u64,
    };

    pub fn probabilityPageCounters(self: *const Fitter) ProbabilityPageCounters {
        return .{
            .normalizer_captures = self.normalizer_captures,
            .calls = self.probability_page_calls,
            .rows = self.probability_page_rows,
        };
    }

    pub fn armTestProbabilityPageBufferFailure(self: *Fitter, fail_index: usize) void {
        if (comptime !builtin.is_test) @compileError("test-only probability-page fault hook");
        self.page_buffer_fail_index = fail_index;
        self.page_buffer_allocation_index = 0;
    }

    fn probabilityPageBuffer(self: *Fitter, bytes: usize) !shared.Buffer {
        if (comptime builtin.is_test) {
            const allocation_index = self.page_buffer_allocation_index;
            self.page_buffer_allocation_index +|= 1;
            if (self.page_buffer_fail_index) |fail_index| if (allocation_index == fail_index) {
                self.page_buffer_fail_index = null;
                return error.OutOfMemory;
            };
        }
        return self.kernels.buffer(bytes);
    }

    pub fn setQueryBatchRows(self: *Fitter, rows: usize) !void {
        if ((rows != default_query_batch_rows and rows != maximum_query_batch_rows and rows != diagnostic_query_batch_rows) or self.attention_calls != 0 or self.dispatches != 0)
            return error.InvalidFitConfig;
        self.query_batch_rows = rows;
    }

    pub fn queryBatchRows(self: *const Fitter) usize {
        return self.query_batch_rows;
    }

    pub fn setKeyLayout(self: *Fitter, layout: KeyLayout) !void {
        if (self.attention_calls != 0 or self.dispatches != 0) return error.InvalidFitConfig;
        self.key_layout = layout;
    }

    pub fn keyLayout(self: *const Fitter) KeyLayout {
        return self.key_layout;
    }

    pub fn setRawSelectionSolver(self: *Fitter, value: RawSelectionSolver) !void {
        if (self.attention_calls != 0 or self.dispatches != 0 or self.phase_counters.raw_selection_calls != 0)
            return error.InvalidFitConfig;
        self.raw_selection_solver = value;
        self.phase_counters.raw_selection_solver = value;
    }

    pub fn rawSelectionSolver(self: *const Fitter) RawSelectionSolver {
        return self.raw_selection_solver;
    }

    pub fn phaseCounters(self: *const Fitter) PhaseCounters {
        return self.phase_counters;
    }

    pub fn setPhaseProfiling(self: *Fitter, enabled: bool) !void {
        if (self.attention_calls != 0 or self.dispatches != 0 or self.accelerated_value_solves != 0 or self.phase_counters.raw_selection_calls != 0)
            return error.InvalidFitConfig;
        self.profile_phases = enabled;
    }

    pub fn phaseDurations(self: *const Fitter) PhaseDurations {
        return self.phase_counters.phase_durations;
    }

    fn phaseNow(self: *Fitter) u64 {
        const before = platform.time.monotonicNs();
        const after = platform.time.monotonicNs();
        self.phase_counters.phase_durations.clock_reads +|= 2;
        self.phase_counters.phase_durations.clock_overhead_ns +|= after -| before;
        return after;
    }

    pub fn releaseWorkspace(_: *Fitter) void {
        // Every synchronous call frees its admitted buffers before returning.
    }

    pub fn backend(self: *Fitter) solver.Backend {
        return backendForContext(self);
    }

    /// Recover the worker-local fitter only from the exact backend produced by
    /// `backend()`. This prevents the bounded serving route from casting an
    /// arbitrary callback context or accidentally using another worker.
    pub fn fromBackend(value: solver.Backend) !*Fitter {
        const context = value.context orelse return error.InvalidFitConfig;
        const expected = backendForContext(context);
        if (value.raw_range_policy != expected.raw_range_policy or value.workspace_fn != expected.workspace_fn or value.attend_fn != expected.attend_fn or
            value.summary_fn != expected.summary_fn or value.values_fn != expected.values_fn or
            value.select_fn != expected.select_fn or value.raw_select_fn != expected.raw_select_fn or
            value.raw_range_workspace_fn != expected.raw_range_workspace_fn or value.raw_range_begin_fn != expected.raw_range_begin_fn or value.raw_range_end_fn != expected.raw_range_end_fn)
            return error.InvalidFitConfig;
        return @ptrCast(@alignCast(context));
    }

    fn backendForContext(context: ?*anyopaque) solver.Backend {
        return .{
            .context = context,
            .validation_batch = .rows_64,
            .raw_range_policy = if (context) |raw| if ((@as(*Fitter, @ptrCast(@alignCast(raw)))).raw_selection_solver == .range_shared_fp64_gram_original_kkt_v1) .range_shared_fp64_gram_original_kkt_v1 else .off else .off,
            .workspace_fn = workspace,
            .attend_fn = attend,
            .summary_fn = summary,
            .values_fn = if (fp64.supported) solveValuesAccelerated else null,
            .select_fn = if (fp64.supported) fp64.select else null,
            .raw_select_fn = if (fp64.supported) selectRawTargetAccelerated else null,
            .raw_range_workspace_fn = if (fp64.supported) rawRangeWorkspace else null,
            .raw_range_begin_fn = if (fp64.supported) rawRangeBegin else null,
            .raw_range_end_fn = if (fp64.supported) rawRangeEnd else null,
        };
    }

    fn rawRangeWorkspace(_: ?*anyopaque, n: usize) !usize {
        return std.math.add(usize, try fp64.rawGramPayloadBytes(n), @sizeOf(RangeGramSession));
    }

    fn rawRangeBegin(context: ?*anyopaque, allocator: std.mem.Allocator, identity: solver.Backend.RawRangeIdentity, admitted_bytes: usize) !solver.Backend.RawRangeSession {
        const self: *Fitter = @ptrCast(@alignCast(context orelse return error.InvalidCallbackContext));
        if (self.raw_selection_solver != .range_shared_fp64_gram_original_kkt_v1) return error.InvalidFitConfig;
        return self.raw_range_machine.begin(allocator, identity, admitted_bytes);
    }

    fn rawRangeEnd(context: ?*anyopaque, token: solver.Backend.RawRangeSession) void {
        const self: *Fitter = @ptrCast(@alignCast(context orelse return));
        if (!self.raw_range_machine.end(token)) self.phase_counters.raw_range_end_rejections +|= 1;
    }

    /// Shared CPU implementation used by the production Fitter callback and
    /// cross-layer tests. It owns the real prepareRawGram/selectImpl/fallback
    /// sequence; tests do not substitute a mock Gram implementation.
    fn selectRangeShared(machine: *RangeSessionMachine, counters: *PhaseCounters, allocator: std.mem.Allocator, p: []const f64, importance: []const f64, rhs: []const f64, q: usize, n: usize, mandatory: []const usize, indices: []usize, weights: []f64, cfg: solver.Config) !usize {
        const token = cfg.raw_range_session orelse return error.RawRangeSessionRequired;
        const session = try machine.consume(token, q, n);
        if (session.poisoned) return fp64.selectRawTarget(null, allocator, p, importance, rhs, q, n, mandatory, indices, weights, cfg);
        if (session.prepared == null) {
            session.prepared = fp64.prepareRawGram(session.allocator, p, rhs, q, n, cfg.control) catch |err| switch (err) {
                error.Cancelled => return err,
                error.OutOfMemory, error.NonFiniteAttention, error.UnsupportedOperation => {
                    session.poisoned = true;
                    return fp64.selectRawTarget(null, allocator, p, importance, rhs, q, n, mandatory, indices, weights, cfg);
                },
                else => return err,
            };
            counters.gram_advisory.gram_precomputes +|= 1;
        }
        const result: anyerror!usize = if (comptime builtin.is_test) blk: {
            if (session.test_fail_prepared_once) {
                session.test_fail_prepared_once = false;
                break :blk error.TestPreparedProposalFailure;
            }
            break :blk fp64.selectRawTargetPreparedGram(&counters.gram_advisory, &session.prepared.?, allocator, p, importance, rhs, q, n, mandatory, indices, weights, cfg);
        } else fp64.selectRawTargetPreparedGram(&counters.gram_advisory, &session.prepared.?, allocator, p, importance, rhs, q, n, mandatory, indices, weights, cfg);
        return result catch |err| switch (err) {
            error.RawRangeIdentityMismatch => return err,
            error.Cancelled => return err,
            else => {
                session.prepared.?.deinit();
                session.prepared = null;
                session.poisoned = true;
                return fp64.selectRawTarget(null, allocator, p, importance, rhs, q, n, mandatory, indices, weights, cfg);
            },
        };
    }

    fn selectRawTargetAccelerated(context: ?*anyopaque, allocator: std.mem.Allocator, p: []const f64, importance: []const f64, rhs: []const f64, q: usize, n: usize, mandatory: []const usize, indices: []usize, weights: []f64, cfg: solver.Config) !usize {
        const self: *Fitter = @ptrCast(@alignCast(context orelse return error.InvalidCallbackContext));
        const started = if (self.profile_phases) self.phaseNow() else 0;
        defer if (self.profile_phases) {
            self.phase_counters.phase_durations.raw_selection_ns +|= self.phaseNow() -| started;
        };
        if (self.profile_phases) self.phase_counters.phase_durations.raw_selection_calls +|= 1;
        const active_solves = switch (self.raw_selection_solver) {
            .householder => try fp64.selectRawTarget(null, allocator, p, importance, rhs, q, n, mandatory, indices, weights, cfg),
            .fp64_gram_advisory => try fp64.selectRawTargetGramAdvisory(&self.phase_counters.gram_advisory, allocator, p, importance, rhs, q, n, mandatory, indices, weights, cfg),
            .range_shared_fp64_gram_original_kkt_v1 => try selectRangeShared(&self.raw_range_machine, &self.phase_counters, allocator, p, importance, rhs, q, n, mandatory, indices, weights, cfg),
        };
        self.phase_counters.raw_selection_calls +|= 1;
        self.phase_counters.raw_selection_active_solves +|= active_solves;
        return active_solves;
    }

    fn solveValuesAccelerated(context: ?*anyopaque, allocator: std.mem.Allocator, a: []const f64, y: []const f64, q: usize, m: usize, d: usize, ridge: f64, control: solver.Control, out: []f64) !void {
        const self: *Fitter = @ptrCast(@alignCast(context orelse return error.InvalidCallbackContext));
        const started = if (self.profile_phases) self.phaseNow() else 0;
        defer if (self.profile_phases) {
            self.phase_counters.phase_durations.solve_values_ns +|= self.phaseNow() -| started;
        };
        if (self.profile_phases) self.phase_counters.phase_durations.solve_values_calls +|= 1;
        try fp64.values(null, allocator, a, y, q, m, d, ridge, control, out);
        self.accelerated_value_solves +|= 1;
        if (d == 1) {
            self.phase_counters.scalar_rhs_solves +|= 1;
            self.phase_counters.max_scalar_query_rows = @max(self.phase_counters.max_scalar_query_rows, q);
            self.phase_counters.max_scalar_selected_rows = @max(self.phase_counters.max_scalar_selected_rows, m);
        } else {
            self.phase_counters.vector_rhs_solves +|= 1;
            self.phase_counters.max_vector_query_rows = @max(self.phase_counters.max_vector_query_rows, q);
            self.phase_counters.max_vector_selected_rows = @max(self.phase_counters.max_vector_selected_rows, m);
            self.phase_counters.max_vector_width = @max(self.phase_counters.max_vector_width, d);
        }
    }

    pub fn headHostWorkspace(original_rows: usize, target_rows: usize, block_rows: usize, train_elements: usize, heldout_elements: usize, dim: usize, options: engine.Options) !engine.HeadWorkspace {
        var planned = options;
        planned.fit.backend = backendForContext(null);
        return engine.headWorkspaceBytes(original_rows, target_rows, block_rows, train_elements, heldout_elements, dim, planned);
    }

    pub const ParallelPlan = struct {
        workers: usize,
        resident_bytes: usize,
        per_worker_bytes: usize,
        admitted_bytes: usize,
    };

    /// Select independently owned Metal fitters under one aggregate host bound.
    /// The common engine accounts source + worst-case candidate in `resident`;
    /// each admitted worker then receives exactly the largest planned head
    /// workspace, never the whole request reservation.
    pub fn parallelPlan(source_bytes: usize, head_count: usize, head_workspace_bytes: usize, aggregate_limit: usize, maximum_workers: usize) !ParallelPlan {
        if (maximum_workers == 0 or maximum_workers > 4) return error.InvalidFitConfig;
        const resident = try engine.parallelResidentBytes(source_bytes, head_count);
        if (resident > aggregate_limit) return error.CompactionWorkspaceLimit;
        if (head_workspace_bytes == 0) return .{
            .workers = 1,
            .resident_bytes = resident,
            .per_worker_bytes = 0,
            .admitted_bytes = resident,
        };
        if (head_count == 0) return error.InvalidFitConfig;
        const capacity = (aggregate_limit - resident) / head_workspace_bytes;
        if (capacity == 0) return error.CompactionWorkspaceLimit;
        const workers = @min(@min(maximum_workers, head_count), capacity);
        const admitted = try engine.parallelHostBytes(source_bytes, head_count, head_workspace_bytes, workers);
        if (admitted > aggregate_limit) return error.CompactionWorkspaceLimit;
        return .{
            .workers = workers,
            .resident_bytes = resident,
            .per_worker_bytes = head_workspace_bytes,
            .admitted_bytes = admitted,
        };
    }

    fn workspace(raw: ?*anyopaque, n: usize, _: usize, q: usize, d: usize) !usize {
        const configured_rows = if (raw) |context| (@as(*Fitter, @ptrCast(@alignCast(context)))).query_batch_rows else maximum_query_batch_rows;
        return workspaceBytesShape(n, q, d, configured_rows);
    }

    /// Pure shape authority shared by the production callback and planner
    /// regressions; it allocates and dispatches nothing.
    pub fn workspaceBytesShape(n: usize, q: usize, d: usize, configured_rows: usize) !usize {
        return workspaceBytesShapeLimit(n, q, d, configured_rows, max_query_rows);
    }

    fn workspaceBytesShapeLimit(n: usize, q: usize, d: usize, configured_rows: usize, query_limit: usize) !usize {
        if (n == 0 or n > 131072 or q == 0 or q > query_limit or d == 0 or d > 1024) return error.InvalidAttentionShape;
        if (configured_rows != default_query_batch_rows and configured_rows != maximum_query_batch_rows and configured_rows != diagnostic_query_batch_rows) return error.InvalidAttentionShape;
        const kv = try rounded(try mul(try mul(n, d), 4));
        const batch_rows = @min(q, configured_rows);
        const queries = try rounded(try mul(try mul(batch_rows, d), 4));
        const bias = try rounded(try mul(n, 4));
        const result = try rounded(try mul(try mul(batch_rows, try add(try add(n, d), 1)), 4));
        // Shared-buffer backing is physical host memory on Metal. Include it
        // in the engine's transient admission as well as the per-call cap.
        return add(try add(try add(try mul(kv, 2), queries), try add(bias, result)), std.heap.pageSize());
    }

    fn attend(raw: ?*anyopaque, _: std.mem.Allocator, head: solver.Head, queries: []const f64, probabilities: []f64, targets: ?[]f64, masses: ?[]f64, control: solver.Control) !void {
        return attendImpl(raw, head, queries, probabilities, targets, masses, null, null, control);
    }

    pub fn trainingCaptureBackend(self: *Fitter) training_source.CaptureBackend {
        return .{ .context = self, .configure_fn = trainingCaptureConfigure, .workspace_fn = trainingCaptureWorkspace, .begin_fn = trainingCaptureBegin, .tile_fn = trainingCaptureTile, .end_fn = trainingCaptureEnd };
    }
    fn trainingCaptureConfigure(raw: ?*anyopaque, batch_rows: usize) !void {
        const self: *Fitter = @ptrCast(@alignCast(raw orelse return error.InvalidFitConfig));
        // A worker-local fitter serves more than one checkpoint head.  The
        // capture policy configures every head, but an already-active fitter
        // may only retain (never change) its established batch size.
        if (batch_rows != self.query_batch_rows) try self.setQueryBatchRows(batch_rows);
    }
    fn trainingCaptureWorkspace(raw: ?*anyopaque, head: solver.Head, batch_rows: usize) !usize {
        const self: *Fitter = @ptrCast(@alignCast(raw orelse return error.InvalidFitConfig));
        if (batch_rows != self.query_batch_rows) return error.InvalidFitConfig;
        return add(try workspace(self, head.rows, 1, batch_rows, head.dim), @sizeOf(TrainingCapture));
    }
    fn trainingCaptureBegin(raw: ?*anyopaque, allocator: std.mem.Allocator, head: solver.Head, batch_rows: usize, control: solver.Control) !?*anyopaque {
        const self: *Fitter = @ptrCast(@alignCast(raw orelse return error.InvalidFitConfig));
        try head.validate();
        if (batch_rows != self.query_batch_rows) return error.InvalidFitConfig;
        if (try trainingCaptureWorkspace(self, head, batch_rows) > self.max_device_bytes) return error.CompactionWorkspaceLimit;
        try control.check();
        const owned = try allocator.create(TrainingCapture);
        errdefer allocator.destroy(owned);
        var keys = try self.probabilityPageBuffer(try mul(head.keys.len, 4));
        errdefer keys.deinit();
        var values = try self.probabilityPageBuffer(try mul(head.values.len, 4));
        errdefer values.deinit();
        var bias = try self.probabilityPageBuffer(try mul(head.rows, 4));
        errdefer bias.deinit();
        var query = try self.probabilityPageBuffer(try mul(try mul(batch_rows, head.dim), 4));
        errdefer query.deinit();
        var result = try self.probabilityPageBuffer(try mul(try mul(batch_rows, try add(try add(head.rows, head.dim), 1)), 4));
        errdefer result.deinit();
        try fillKeys(keys.floats(), head.keys, head.rows, head.dim, self.key_layout, control);
        try fill(values.floats(), head.values, control);
        if (head.bias) |input| try fill(bias.floats(), input, control) else @memset(bias.floats(), 0);
        try control.check();
        owned.* = .{ .owner = self, .head = head, .batch_rows = batch_rows, .keys = keys, .values = values, .bias = bias, .query = query, .result = result };
        self.attention_calls +|= 1;
        self.training_capture_allocations +|= 5;
        return owned;
    }
    fn trainingCaptureTile(raw: ?*anyopaque, queries: []const f64, probabilities: []f64, outputs: []f64, control: solver.Control) !void {
        const capture: *TrainingCapture = @ptrCast(@alignCast(raw orelse return error.InvalidFitConfig));
        const head = capture.head;
        if (queries.len == 0 or queries.len % head.dim != 0) return error.InvalidAttentionShape;
        const rows = queries.len / head.dim;
        if (rows > capture.batch_rows or probabilities.len != rows * head.rows or outputs.len != queries.len) return error.InvalidAttentionShape;
        try fill(capture.query.floats()[0..queries.len], queries, control);
        const gpu_started = if (capture.owner.profile_phases) capture.owner.phaseNow() else 0;
        const timing = try capture.owner.kernels.runThreadgroupsPipeline(capture.owner.kernels.pipeline, .{ .op = 0, .n = @intCast(head.rows), .dim = @intCast(head.dim), .groups = @intCast(rows), .mode = if (capture.owner.key_layout == .column_major) 1 else 0, .scale = @floatCast(head.scale), .reserved = @floatCast(head.softcap) }, &.{ capture.keys.ref(), capture.values.ref(), capture.query.ref(), capture.bias.ref() }, capture.result.ref(), .{ rows, 1, 1 }, .{ 128, 1, 1 });
        if (capture.owner.profile_phases) {
            capture.owner.phase_counters.phase_durations.training_capture_tiles +|= 1;
            capture.owner.phase_counters.phase_durations.training_capture_gpu_commit_wait_ns +|= capture.owner.phaseNow() -| gpu_started;
            capture.owner.phase_counters.phase_durations.training_capture_gpu_active_ns +|= timing.gpu_nanos;
            capture.owner.phase_counters.phase_durations.training_capture_gpu_wait_ns +|= timing.wait_nanos;
        }
        capture.owner.dispatches +|= 1;
        try control.check();
        const pn = rows * head.rows;
        const raw_out = capture.result.floats()[0 .. pn + queries.len + rows];
        for (probabilities, raw_out[0..pn]) |*dst, v| {
            if (!std.math.isFinite(v)) return error.NonFiniteAttention;
            dst.* = v;
        }
        for (outputs, raw_out[pn..][0..queries.len]) |*dst, v| {
            if (!std.math.isFinite(v)) return error.NonFiniteAttention;
            dst.* = v;
        }
        capture.owner.derived_readback_bytes +|= try mul(raw_out.len, @sizeOf(f32));
    }
    fn trainingCaptureEnd(raw: ?*anyopaque, allocator: std.mem.Allocator) void {
        const capture: *TrainingCapture = @ptrCast(@alignCast(raw orelse return));
        capture.owner.training_capture_frees +|= 5;
        capture.deinit();
        allocator.destroy(capture);
    }

    fn summary(raw: ?*anyopaque, _: std.mem.Allocator, head: solver.Head, queries: []const f64, targets: []f64, masses: []f64, control: solver.Control) !void {
        return attendImpl(raw, head, queries, null, targets, masses, null, null, control);
    }

    pub fn captureWorkspaceBytes(self: *Fitter, head: solver.Head, query_elements: usize) !usize {
        return self.captureWorkspaceBytesLimit(head, query_elements, max_query_rows);
    }

    fn captureWorkspaceBytesWide(self: *Fitter, head: solver.Head, query_elements: usize) !usize {
        return self.captureWorkspaceBytesLimit(head, query_elements, max_wide_capture_query_rows);
    }

    fn captureWorkspaceBytesLimit(self: *Fitter, head: solver.Head, query_elements: usize, query_limit: usize) !usize {
        try head.validate();
        if (query_elements == 0 or query_elements % head.dim != 0) return error.InvalidAttentionShape;
        const q = query_elements / head.dim;
        const retained_bytes = try mul(try add(query_elements, try mul(q, 2)), @sizeOf(f32));
        return add(try workspaceBytesShapeLimit(head.rows, q, head.dim, self.query_batch_rows, query_limit), try add(retained_bytes, try rounded(try mul(@min(q, self.query_batch_rows), 4))));
    }

    pub fn captureNormalizer(self: *Fitter, allocator: std.mem.Allocator, head: solver.Head, queries: []const f64, control: solver.Control) !Normalizer {
        return self.captureNormalizerLimit(allocator, head, queries, max_query_rows, control);
    }

    fn captureNormalizerWide(self: *Fitter, allocator: std.mem.Allocator, head: solver.Head, queries: []const f64, control: solver.Control) !Normalizer {
        return self.captureNormalizerLimit(allocator, head, queries, max_wide_capture_query_rows, control);
    }

    fn captureNormalizerLimit(self: *Fitter, allocator: std.mem.Allocator, head: solver.Head, queries: []const f64, query_limit: usize, control: solver.Control) !Normalizer {
        try head.validate();
        if (queries.len == 0 or queries.len % head.dim != 0) return error.InvalidAttentionShape;
        const q = queries.len / head.dim;
        const capture_bytes = try self.captureWorkspaceBytesLimit(head, queries.len, query_limit);
        if (capture_bytes > self.max_device_bytes) return error.CompactionWorkspaceLimit;
        try control.check();
        const pairs = try allocator.alloc(f32, try mul(q, 2));
        errdefer allocator.free(pairs);
        const outputs = try allocator.alloc(f32, queries.len);
        errdefer allocator.free(outputs);
        try self.attendNormalizerLimit(head, queries, outputs, pairs, query_limit, control);
        self.normalizer_captures +|= 1;
        return .{ .allocator = allocator, .owner = self, .keys = head.keys.ptr, .values = head.values.ptr, .bias = if (head.bias) |b| b.ptr else null, .queries = queries.ptr, .rows = head.rows, .dim = head.dim, .query_count = q, .scale = head.scale, .softcap = head.softcap, .layout = self.key_layout, .batch_rows = self.query_batch_rows, .pairs = pairs, .outputs = outputs };
    }

    pub fn capturedPageRetainedBytes(_: *const Fitter, head: solver.Head, query_elements: usize) !usize {
        try head.validate();
        if (query_elements == 0 or query_elements % head.dim != 0) return error.InvalidAttentionShape;
        const q = query_elements / head.dim;
        const host = try mul(try add(query_elements, try mul(q, 2)), @sizeOf(f32));
        const keys = try rounded(try mul(head.keys.len, @sizeOf(f32)));
        const bias = try rounded(try mul(head.rows, @sizeOf(f32)));
        const all_queries = try rounded(try mul(query_elements, @sizeOf(f32)));
        const all_pairs = try rounded(try mul(try mul(q, 2), @sizeOf(f32)));
        return add(host, try add(try add(keys, bias), try add(all_queries, all_pairs)));
    }

    pub fn capturePageSource(self: *Fitter, allocator: std.mem.Allocator, head: solver.Head, queries: []const f64, external_live_bytes: usize, combined_cap: usize, control: solver.Control) !CapturedPageSource {
        return self.capturePageSourceLimit(allocator, head, queries, external_live_bytes, combined_cap, max_query_rows, control);
    }

    pub fn capturePageSourceWide(self: *Fitter, allocator: std.mem.Allocator, head: solver.Head, queries: []const f64, external_live_bytes: usize, combined_cap: usize, control: solver.Control) !CapturedPageSource {
        return self.capturePageSourceLimit(allocator, head, queries, external_live_bytes, combined_cap, max_wide_capture_query_rows, control);
    }

    fn admitCapturedPage(external_live_bytes: usize, combined_cap: usize, capture_peak: usize, retained: usize) !void {
        if (external_live_bytes > combined_cap or @max(capture_peak, retained) > combined_cap - external_live_bytes)
            return error.CompactionWorkspaceLimit;
    }

    fn capturePageSourceLimit(self: *Fitter, allocator: std.mem.Allocator, head: solver.Head, queries: []const f64, external_live_bytes: usize, combined_cap: usize, query_limit: usize, control: solver.Control) !CapturedPageSource {
        const retained = try self.capturedPageRetainedBytes(head, queries.len);
        const capture_peak = try self.captureWorkspaceBytesLimit(head, queries.len, query_limit);
        try admitCapturedPage(external_live_bytes, combined_cap, capture_peak, retained);
        var normalizer = if (query_limit == max_wide_capture_query_rows)
            try self.captureNormalizerWide(allocator, head, queries, control)
        else
            try self.captureNormalizer(allocator, head, queries, control);
        errdefer normalizer.deinit();
        var keys = try self.probabilityPageBuffer(try mul(head.keys.len, @sizeOf(f32)));
        errdefer keys.deinit();
        var bias = try self.probabilityPageBuffer(try mul(head.rows, @sizeOf(f32)));
        errdefer bias.deinit();
        var captured_queries = try self.probabilityPageBuffer(try mul(queries.len, @sizeOf(f32)));
        errdefer captured_queries.deinit();
        var pairs = try self.probabilityPageBuffer(try mul(normalizer.pairs.len, @sizeOf(f32)));
        errdefer pairs.deinit();
        try fillKeys(keys.floats(), head.keys, head.rows, head.dim, self.key_layout, control);
        if (head.bias) |input| try fill(bias.floats(), input, control) else @memset(bias.floats(), 0);
        try fill(captured_queries.floats(), queries, control);
        @memcpy(pairs.floats(), normalizer.pairs);
        try control.check();
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update("antfly.captured-page-source.v1\x00");
        h.update(std.mem.sliceAsBytes(keys.floats()));
        h.update(std.mem.sliceAsBytes(bias.floats()));
        h.update(std.mem.sliceAsBytes(captured_queries.floats()));
        h.update(std.mem.sliceAsBytes(pairs.floats()));
        const fingerprint = h.finalResult();
        self.captured_page_generation +|= 1;
        if (self.captured_page_generation == 0) return error.InvalidFitConfig;
        self.captured_page_fingerprint = fingerprint;
        self.captured_page_rows = head.rows;
        self.captured_page_dim = head.dim;
        self.captured_page_queries = queries.len / head.dim;
        self.captured_page_layout = self.key_layout;
        self.captured_page_batch_rows = self.query_batch_rows;
        self.captured_page_retained_bytes = retained;
        self.captured_page_scale = head.scale;
        self.captured_page_softcap = head.softcap;
        return .{ .normalizer = normalizer, .owner = self, .keys = keys, .bias = bias, .queries = captured_queries, .pairs = pairs, .rows = head.rows, .dim = head.dim, .query_count = queries.len / head.dim, .layout = self.key_layout, .batch_rows = self.query_batch_rows, .retained_bytes = retained, .fingerprint = fingerprint, .generation = self.captured_page_generation };
    }

    pub fn capturedPageTransientBytes(self: *const Fitter, source: *const CapturedPageSource, page_rows: usize, query_count: usize) !usize {
        if (source.owner != self or source.generation != self.captured_page_generation or !std.mem.eql(u8, &source.fingerprint, &self.captured_page_fingerprint) or
            source.rows != self.captured_page_rows or source.dim != self.captured_page_dim or source.query_count != self.captured_page_queries or
            source.layout != self.captured_page_layout or source.batch_rows != self.captured_page_batch_rows or
            source.retained_bytes != self.captured_page_retained_bytes or
            source.normalizer.owner != self or source.normalizer.rows != source.rows or source.normalizer.dim != source.dim or
            source.normalizer.query_count != source.query_count or source.normalizer.layout != source.layout or source.normalizer.batch_rows != source.batch_rows or
            source.normalizer.scale != self.captured_page_scale or source.normalizer.softcap != self.captured_page_softcap or
            page_rows == 0 or page_rows > source.rows or query_count == 0 or query_count > source.query_count) return error.InvalidFitConfig;
        const batch = @min(query_count, source.batch_rows);
        return add(try rounded(try mul(try mul(batch, page_rows), @sizeOf(f32))), std.heap.pageSize());
    }

    /// Fast bounded provider. It never receives Head or query slices and thus
    /// cannot rescan or recopy them. `out` is discard-on-error, as in the
    /// legacy checked provider.
    pub fn replayCapturedPageQueries(self: *Fitter, source: *const CapturedPageSource, page_start: usize, page_rows: usize, first_query: usize, query_count: usize, out: []f64, control: solver.Control) !void {
        try control.check();
        if (source.owner != self or source.layout != self.key_layout or source.batch_rows != self.query_batch_rows or
            page_rows == 0 or page_start > source.rows or page_rows > source.rows - page_start or
            query_count == 0 or first_query > source.query_count or query_count > source.query_count - first_query or
            out.len != try mul(query_count, page_rows)) return error.InvalidFitConfig;
        const transient = try self.capturedPageTransientBytes(source, page_rows, query_count);
        if (try add(source.retained_bytes, transient) > self.max_device_bytes) return error.CompactionWorkspaceLimit;
        const batch = @min(query_count, source.batch_rows);
        var result = try self.probabilityPageBuffer(try mul(try mul(batch, page_rows), @sizeOf(f32)));
        defer result.deinit();
        var offset: usize = 0;
        while (offset < query_count) {
            try control.check();
            const rows = @min(source.batch_rows, query_count - offset);
            const query_start = first_query + offset;
            _ = try self.kernels.runThreadgroupsPipeline(self.kernels.pipeline, .{ .op = 2, .n = @intCast(source.rows), .dim = @intCast(source.dim), .groups = @intCast(source.query_count), .position = @intCast(page_start), .stride = @intCast(page_rows), .mode = if (source.layout == .column_major) 1 else 0, .index = @intCast(query_start), .scale = @floatCast(source.normalizer.scale), .reserved = @floatCast(source.normalizer.softcap) }, &.{ source.keys.ref(), source.keys.ref(), source.queries.ref(), source.bias.ref(), source.pairs.ref() }, result.ref(), .{ rows, 1, 1 }, .{ 128, 1, 1 });
            self.dispatches +|= 1;
            try control.check();
            for (result.floats()[0 .. rows * page_rows]) |value| if (!std.math.isFinite(value)) return error.NonFiniteAttention;
            for (out[offset * page_rows ..][0 .. rows * page_rows], result.floats()[0 .. rows * page_rows]) |*destination, value| destination.* = value;
            self.probability_page_calls +|= 1;
            self.probability_page_rows +|= page_rows;
            offset += rows;
        }
    }

    fn attendNormalizer(self: *Fitter, head: solver.Head, queries: []const f64, outputs: []f32, pairs: []f32, control: solver.Control) !void {
        return attendImpl(self, head, queries, null, null, null, outputs, pairs, control);
    }

    fn attendNormalizerLimit(self: *Fitter, head: solver.Head, queries: []const f64, outputs: []f32, pairs: []f32, query_limit: usize, control: solver.Control) !void {
        return attendImplLimit(self, head, queries, null, null, null, outputs, pairs, query_limit, control);
    }

    /// On any error the caller must discard `out`; completed earlier query
    /// batches are not rolled back. A fenced batch is validated before publish.
    pub fn replayPage(self: *Fitter, handle: *const Normalizer, head: solver.Head, queries: []const f64, page_start: usize, page_rows: usize, out: []f64, control: solver.Control) !void {
        return self.replayPageQueries(handle, head, queries, page_start, page_rows, 0, handle.query_count, out, control);
    }

    /// Replay a bounded query tile without retaining a full Q-by-page matrix.
    /// The tile still binds to the exact query allocation captured by `handle`.
    pub fn replayPageQueries(self: *Fitter, handle: *const Normalizer, head: solver.Head, queries: []const f64, page_start: usize, page_rows: usize, first_query: usize, query_count: usize, out: []f64, control: solver.Control) !void {
        try control.check();
        try head.validate();
        if (handle.owner != self or page_rows == 0 or page_start > head.rows or page_rows > head.rows - page_start or query_count == 0 or first_query > handle.query_count or query_count > handle.query_count - first_query or out.len != try mul(query_count, page_rows) or
            head.keys.ptr != handle.keys or head.values.ptr != handle.values or (if (head.bias) |b| b.ptr else null) != handle.bias or queries.ptr != handle.queries or
            head.rows != handle.rows or head.dim != handle.dim or queries.len != try mul(handle.query_count, head.dim) or head.scale != handle.scale or head.softcap != handle.softcap or
            self.key_layout != handle.layout or self.query_batch_rows != handle.batch_rows) return error.InvalidFitConfig;
        const bytes = try self.probabilityPageTileWorkspaceBytes(handle, page_rows, query_count);
        if (bytes > self.max_device_bytes) return error.CompactionWorkspaceLimit;
        // The device buffers cover this requested tile, not the complete
        // captured query set. Keep allocation identical to the transient byte
        // authority for nonzero/small query tiles.
        const batch = @min(query_count, self.query_batch_rows);
        for (handle.pairs) |value| if (!std.math.isFinite(value)) return error.NonFiniteAttention;
        for (0..handle.query_count) |query_index| if (!(handle.pairs[query_index * 2 + 1] > 0)) return error.NonFiniteAttention;
        var keys = try self.probabilityPageBuffer(try mul(head.keys.len, 4));
        defer keys.deinit();
        var query = try self.probabilityPageBuffer(try mul(try mul(batch, head.dim), 4));
        defer query.deinit();
        var bias = try self.probabilityPageBuffer(try mul(head.rows, 4));
        defer bias.deinit();
        var pairs = try self.probabilityPageBuffer(try mul(try mul(batch, 2), 4));
        defer pairs.deinit();
        var result = try self.probabilityPageBuffer(try mul(try mul(batch, page_rows), 4));
        defer result.deinit();
        try fillKeys(keys.floats(), head.keys, head.rows, head.dim, self.key_layout, control);
        if (head.bias) |input| try fill(bias.floats(), input, control) else @memset(bias.floats(), 0);
        var offset: usize = 0;
        while (offset < query_count) {
            try control.check();
            const rows = @min(self.query_batch_rows, query_count - offset);
            const source_query = first_query + offset;
            try fill(query.floats()[0 .. rows * head.dim], queries[source_query * head.dim ..][0 .. rows * head.dim], control);
            @memcpy(pairs.floats()[0 .. rows * 2], handle.pairs[source_query * 2 ..][0 .. rows * 2]);
            _ = try self.kernels.runThreadgroupsPipeline(self.kernels.pipeline, .{ .op = 2, .n = @intCast(head.rows), .dim = @intCast(head.dim), .groups = @intCast(rows), .position = @intCast(page_start), .stride = @intCast(page_rows), .mode = if (self.key_layout == .column_major) 1 else 0, .scale = @floatCast(head.scale), .reserved = @floatCast(head.softcap) }, &.{ keys.ref(), keys.ref(), query.ref(), bias.ref(), pairs.ref() }, result.ref(), .{ rows, 1, 1 }, .{ 128, 1, 1 });
            self.dispatches +|= 1;
            try control.check();
            for (result.floats()[0 .. rows * page_rows]) |value| if (!std.math.isFinite(value)) return error.NonFiniteAttention;
            for (out[offset * page_rows ..][0 .. rows * page_rows], result.floats()[0 .. rows * page_rows]) |*destination, value| destination.* = value;
            self.probability_page_calls +|= 1;
            self.probability_page_rows +|= page_rows;
            offset += rows;
        }
    }

    /// Complete provider-side bound while the retained normalizer remains
    /// live. Callers add their own signed-delta vectors and scored packets.
    pub fn probabilityPageWorkspaceBytes(self: *const Fitter, handle: *const Normalizer, page_rows: usize) !usize {
        return self.probabilityPageTileWorkspaceBytes(handle, page_rows, handle.query_count);
    }

    pub fn probabilityPageTileWorkspaceBytes(self: *const Fitter, handle: *const Normalizer, page_rows: usize, query_count: usize) !usize {
        return add(try handle.retainedBytes(), try self.probabilityPageTransientBytes(handle, page_rows, query_count));
    }

    /// Device/shared-buffer transient only. The retained Normalizer is charged
    /// separately by the replay planner's external-live authority.
    pub fn probabilityPageTransientBytes(self: *const Fitter, handle: *const Normalizer, page_rows: usize, query_count: usize) !usize {
        if (handle.owner != self or page_rows == 0 or query_count == 0 or query_count > handle.query_count) return error.InvalidFitConfig;
        const batch = @min(query_count, self.query_batch_rows);
        const key_bytes = try rounded(try mul(try mul(handle.rows, handle.dim), 4));
        const bias_bytes = try rounded(try mul(handle.rows, 4));
        const query_bytes = try rounded(try mul(try mul(batch, handle.dim), 4));
        const pair_bytes = try rounded(try mul(try mul(batch, 2), 4));
        const page_bytes = try rounded(try mul(try mul(batch, page_rows), 4));
        const device_bytes = try add(try add(try add(key_bytes, bias_bytes), try add(query_bytes, pair_bytes)), page_bytes);
        return add(device_bytes, std.heap.pageSize());
    }

    fn fill(out: []f32, input: []const f64, control: solver.Control) !void {
        if (out.len != input.len) return error.InvalidAttentionShape;
        for (input, out, 0..) |value, *destination, i| {
            if (i % 4096 == 0) try control.check();
            if (!std.math.isFinite(value) or @abs(value) > std.math.floatMax(f32)) return error.NonFiniteAttention;
            destination.* = @floatCast(value);
        }
    }

    fn fillKeys(out: []f32, input: []const f64, rows: usize, dim: usize, layout: KeyLayout, control: solver.Control) !void {
        if (out.len != input.len or input.len != try mul(rows, dim)) return error.InvalidAttentionShape;
        if (layout == .row_major) return fill(out, input, control);
        for (0..rows) |row| for (0..dim) |col| {
            const source = row * dim + col;
            if (source % 4096 == 0) try control.check();
            const value = input[source];
            if (!std.math.isFinite(value) or @abs(value) > std.math.floatMax(f32)) return error.NonFiniteAttention;
            out[col * rows + row] = @floatCast(value);
        };
    }

    fn attendImpl(raw: ?*anyopaque, head: solver.Head, queries: []const f64, probabilities: ?[]f64, targets: ?[]f64, masses: ?[]f64, captured_outputs: ?[]f32, captured_pairs: ?[]f32, control: solver.Control) !void {
        return attendImplLimit(raw, head, queries, probabilities, targets, masses, captured_outputs, captured_pairs, max_query_rows, control);
    }

    fn attendImplLimit(raw: ?*anyopaque, head: solver.Head, queries: []const f64, probabilities: ?[]f64, targets: ?[]f64, masses: ?[]f64, captured_outputs: ?[]f32, captured_pairs: ?[]f32, query_limit: usize, control: solver.Control) !void {
        const self: *Fitter = @ptrCast(@alignCast(raw orelse return error.InvalidAttentionShape));
        const total_started = if (self.profile_phases) self.phaseNow() else 0;
        defer if (self.profile_phases) {
            self.phase_counters.phase_durations.attend_total_ns +|= self.phaseNow() -| total_started;
        };
        if (self.profile_phases) self.phase_counters.phase_durations.attend_calls +|= 1;
        try control.check();
        try head.validate();
        if (queries.len == 0 or queries.len % head.dim != 0) return error.InvalidAttentionShape;
        const q = queries.len / head.dim;
        if (probabilities) |out| if (out.len != try mul(q, head.rows)) return error.InvalidAttentionShape;
        if (targets) |out| if (out.len != queries.len) return error.InvalidAttentionShape;
        if (masses) |out| if (out.len != q) return error.InvalidAttentionShape;
        if ((captured_outputs == null) != (captured_pairs == null)) return error.InvalidAttentionShape;
        if (captured_outputs) |out| if (out.len != queries.len or captured_pairs.?.len != try mul(q, 2)) return error.InvalidAttentionShape;
        var bytes = try workspaceBytesShapeLimit(head.rows, q, head.dim, self.query_batch_rows, query_limit);
        if (captured_pairs != null) bytes = try add(bytes, try rounded(try mul(@min(q, self.query_batch_rows), 4)));
        if (bytes > self.max_device_bytes) return error.CompactionWorkspaceLimit;
        if (head.scale > std.math.floatMax(f32) or head.softcap > std.math.floatMax(f32)) return error.NonFiniteAttention;
        const scale: f32 = @floatCast(head.scale);
        const softcap: f32 = @floatCast(head.softcap);
        if (scale <= 0 or (head.softcap > 0 and softcap <= 0)) return error.NonFiniteAttention;
        self.attention_calls +|= 1;
        const allocate_started = if (self.profile_phases) self.phaseNow() else 0;
        var keys = try self.kernels.buffer(try mul(head.keys.len, 4));
        defer keys.deinit();
        var values = try self.kernels.buffer(try mul(head.values.len, 4));
        defer values.deinit();
        const batch_rows = @min(q, self.query_batch_rows);
        var query = try self.kernels.buffer(try mul(try mul(batch_rows, head.dim), 4));
        defer query.deinit();
        var bias = try self.kernels.buffer(try mul(head.rows, 4));
        defer bias.deinit();
        const trailer: usize = if (captured_pairs != null) 2 else 1;
        var result = try self.kernels.buffer(try mul(try mul(batch_rows, try add(try add(head.rows, head.dim), trailer)), 4));
        defer result.deinit();
        if (self.profile_phases) self.phase_counters.phase_durations.attend_allocate_ns +|= self.phaseNow() -| allocate_started;
        const static_fill_started = if (self.profile_phases) self.phaseNow() else 0;
        try fillKeys(keys.floats(), head.keys, head.rows, head.dim, self.key_layout, control);
        try fill(values.floats(), head.values, control);
        for (queries, 0..) |value, i| {
            if (i % 4096 == 0) try control.check();
            if (!std.math.isFinite(value) or @abs(value) > std.math.floatMax(f32)) return error.NonFiniteAttention;
        }
        if (head.bias) |input| try fill(bias.floats(), input, control) else @memset(bias.floats(), 0);
        if (self.profile_phases) self.phase_counters.phase_durations.attend_fill_ns +|= self.phaseNow() -| static_fill_started;
        // Fence each bounded query chunk before polling cancellation. The
        // device buffers cover only one chunk. Scatter each fenced result into
        // the complete caller-owned matrices used by the shared host solver.
        var offset: usize = 0;
        while (offset < q) {
            try control.check();
            const rows = @min(self.query_batch_rows, q - offset);
            const query_elements = try mul(rows, head.dim);
            const query_fill_started = if (self.profile_phases) self.phaseNow() else 0;
            try fill(query.floats()[0..query_elements], queries[offset * head.dim ..][0..query_elements], control);
            if (self.profile_phases) self.phase_counters.phase_durations.attend_fill_ns +|= self.phaseNow() -| query_fill_started;
            const gpu_started = if (self.profile_phases) self.phaseNow() else 0;
            const pair_timing = try self.kernels.runThreadgroupsPipeline(self.kernels.pipeline, .{
                .op = if (captured_pairs != null) 1 else 0,
                .n = @intCast(head.rows),
                .dim = @intCast(head.dim),
                .groups = @intCast(rows),
                .mode = if (self.key_layout == .column_major) 1 else 0,
                .scale = scale,
                .reserved = softcap,
            }, &.{ keys.ref(), values.ref(), query.ref(), bias.ref() }, result.ref(), .{ rows, 1, 1 }, .{ 128, 1, 1 });
            if (self.profile_phases) {
                self.phase_counters.phase_durations.attend_gpu_commit_wait_ns +|= self.phaseNow() -| gpu_started;
                self.phase_counters.phase_durations.attend_gpu_active_ns +|= pair_timing.gpu_nanos;
                self.phase_counters.phase_durations.attend_gpu_wait_ns +|= pair_timing.wait_nanos;
                self.phase_counters.phase_durations.gpu_dispatches +|= 1;
            }
            self.dispatches += 1;
            try control.check();
            const scatter_started = if (self.profile_phases) self.phaseNow() else 0;
            const pn = try mul(rows, head.rows);
            const result_elements = try add(try add(pn, query_elements), try mul(rows, trailer));
            const output = result.floats()[0..result_elements];
            for (output, 0..) |value, i| {
                if (i % 4096 == 0) try control.check();
                if (!std.math.isFinite(value)) return error.NonFiniteAttention;
            }
            if (probabilities) |out| for (out[offset * head.rows ..][0..pn], output[0..pn]) |*destination, value| {
                destination.* = value;
            };
            if (targets) |out| for (out[offset * head.dim ..][0..query_elements], output[pn..][0..query_elements]) |*destination, value| {
                destination.* = value;
            };
            if (masses) |out| for (out[offset..][0..rows], output[pn + query_elements ..][0..rows]) |*destination, value| {
                destination.* = value;
            };
            if (captured_outputs) |out| @memcpy(out[offset * head.dim ..][0..query_elements], output[pn..][0..query_elements]);
            if (captured_pairs) |out| @memcpy(out[offset * 2 ..][0 .. rows * 2], output[pn + query_elements ..][0 .. rows * 2]);
            self.derived_readback_bytes +|= try mul(result_elements, 4);
            if (self.profile_phases) self.phase_counters.phase_durations.attend_scatter_finite_ns +|= self.phaseNow() -| scatter_started;
            offset += rows;
        }
    }
};

test "private FloatPair replay is bitwise identical to full Metal probabilities" {
    if (!@import("build_options").enable_metal) return error.SkipZigTest;
    if (!@import("antfly_platform").env.getenvBool("ANTFLY_METAL_ATTENTION_MATCH_FLOATPAIR_DIAGNOSTIC")) return error.SkipZigTest;
    const a = std.testing.allocator;
    for ([_]Fitter.KeyLayout{ .row_major, .column_major }) |layout| {
        for ([_]usize{ 16, 64 }) |batch_rows| {
            for ([_]usize{ 1, 16, 17, 65 }) |q| {
                const n: usize = 257;
                const d: usize = 2;
                const keys = try a.alloc(f64, n * d);
                defer a.free(keys);
                const values = try a.alloc(f64, n * d);
                defer a.free(values);
                const bias = try a.alloc(f64, n);
                defer a.free(bias);
                const queries = try a.alloc(f64, q * d);
                defer a.free(queries);
                for (keys, 0..) |*v, i| v.* = @as(f64, @floatFromInt(i % 31)) / 31 - 0.5;
                for (values, 0..) |*v, i| v.* = @as(f64, @floatFromInt(i % 17)) / 17;
                for (bias, 0..) |*v, i| v.* = @as(f64, @floatFromInt(i % 7)) / 100;
                for (queries, 0..) |*v, i| v.* = @as(f64, @floatFromInt(i % 13)) / 13 - 0.25;
                const head: solver.Head = .{ .keys = keys, .values = values, .bias = bias, .rows = n, .dim = d, .scale = 0.75, .softcap = 2 };
                var tracker = shared.BufferAllocationTracker.init(null, 0);
                var fitter = try Fitter.init(a, &tracker, 256 * 1024 * 1024);
                defer fitter.deinit();
                try fitter.setQueryBatchRows(batch_rows);
                try fitter.setKeyLayout(layout);
                const cfg: solver.Config = .{ .target_rows = 1, .backend = fitter.backend(), .max_workspace_bytes = 256 * 1024 * 1024 };
                const full = try a.alloc(f64, q * n);
                defer a.free(full);
                const y = try a.alloc(f64, q * d);
                defer a.free(y);
                try solver.attendBatch(a, head, queries, full, y, null, cfg);
                const capture_backend = fitter.trainingCaptureBackend();
                if (layout == .row_major and batch_rows == 16 and q == 1) for (0..5) |fail_index| {
                    fitter.armTestProbabilityPageBufferFailure(fail_index);
                    try std.testing.expectError(error.OutOfMemory, capture_backend.begin_fn(capture_backend.context, a, head, batch_rows, .{}));
                    const recovered = (try capture_backend.begin_fn(capture_backend.context, a, head, batch_rows, .{})).?;
                    capture_backend.end_fn(recovered, a);
                };
                const owner = (try capture_backend.begin_fn(capture_backend.context, a, head, batch_rows, .{})).?;
                defer capture_backend.end_fn(owner, a);
                const lifecycle_p = try a.alloc(f64, q * n);
                defer a.free(lifecycle_p);
                const lifecycle_y = try a.alloc(f64, q * d);
                defer a.free(lifecycle_y);
                var capture_offset: usize = 0;
                while (capture_offset < q) {
                    const capture_rows = @min(batch_rows, q - capture_offset);
                    try capture_backend.tile_fn(owner, queries[capture_offset * d ..][0 .. capture_rows * d], lifecycle_p[capture_offset * n ..][0 .. capture_rows * n], lifecycle_y[capture_offset * d ..][0 .. capture_rows * d], .{});
                    capture_offset += capture_rows;
                }
                try std.testing.expectEqualSlices(f64, full, lifecycle_p);
                try std.testing.expectEqualSlices(f64, y, lifecycle_y);
                var normalizer = try fitter.captureNormalizer(a, head, queries, .{});
                defer normalizer.deinit();
                var captured = try fitter.capturePageSource(a, head, queries, 0, 256 * 1024 * 1024, .{});
                defer captured.deinit();
                try std.testing.expectEqual(try fitter.capturedPageRetainedBytes(head, queries.len), captured.retained_bytes);
                try std.testing.expect(!std.mem.allEqual(u8, &captured.fingerprint, 0));
                const saved_fingerprint = captured.fingerprint;
                captured.fingerprint[0] ^= 1;
                var rejected: [1]f64 = undefined;
                try std.testing.expectError(error.InvalidFitConfig, fitter.replayCapturedPageQueries(&captured, 0, 1, 0, 1, &rejected, .{}));
                captured.fingerprint = saved_fingerprint;
                captured.dim += 1;
                try std.testing.expectError(error.InvalidFitConfig, fitter.replayCapturedPageQueries(&captured, 0, 1, 0, 1, &rejected, .{}));
                captured.dim -= 1;
                captured.retained_bytes -= 1;
                try std.testing.expectError(error.InvalidFitConfig, fitter.replayCapturedPageQueries(&captured, 0, 1, 0, 1, &rejected, .{}));
                captured.retained_bytes += 1;
                captured.normalizer.scale += 0.25;
                try std.testing.expectError(error.InvalidFitConfig, fitter.replayCapturedPageQueries(&captured, 0, 1, 0, 1, &rejected, .{}));
                captured.normalizer.scale -= 0.25;
                captured.normalizer.softcap += 0.5;
                try std.testing.expectError(error.InvalidFitConfig, fitter.replayCapturedPageQueries(&captured, 0, 1, 0, 1, &rejected, .{}));
                captured.normalizer.softcap -= 0.5;
                captured.normalizer.rows += 1;
                try std.testing.expectError(error.InvalidFitConfig, fitter.replayCapturedPageQueries(&captured, 0, 1, 0, 1, &rejected, .{}));
                captured.normalizer.rows -= 1;
                for ([_]struct { start: usize, rows: usize }{ .{ .start = 0, .rows = 1 }, .{ .start = 1, .rows = 256 } }) |page| {
                    const replay = try a.alloc(f64, q * page.rows);
                    defer a.free(replay);
                    try fitter.replayPage(&normalizer, head, queries, page.start, page.rows, replay, .{});
                    const fast = try a.alloc(f64, q * page.rows);
                    defer a.free(fast);
                    try fitter.replayCapturedPageQueries(&captured, page.start, page.rows, 0, q, fast, .{});
                    try std.testing.expectEqualSlices(f64, replay, fast);
                    for (0..q) |qi| try std.testing.expectEqualSlices(f64, full[qi * n + page.start ..][0..page.rows], replay[qi * page.rows ..][0..page.rows]);
                    if (q > 1) {
                        const first = q / 2;
                        const take = q - first;
                        const tiled = try a.alloc(f64, take * page.rows);
                        defer a.free(tiled);
                        try fitter.replayPageQueries(&normalizer, head, queries, page.start, page.rows, first, take, tiled, .{});
                        const fast_tiled = try a.alloc(f64, take * page.rows);
                        defer a.free(fast_tiled);
                        try fitter.replayCapturedPageQueries(&captured, page.start, page.rows, first, take, fast_tiled, .{});
                        try std.testing.expectEqualSlices(f64, tiled, fast_tiled);
                        for (0..take) |local| try std.testing.expectEqualSlices(f64, full[(first + local) * n + page.start ..][0..page.rows], tiled[local * page.rows ..][0..page.rows]);
                    }
                }
                for (normalizer.outputs, y) |got, want| try std.testing.expectEqual(got, @as(f32, @floatCast(want)));
                if (batch_rows == 16 and q == 17) {
                    const CancelSecond = struct {
                        const Context = struct { fitter: *Fitter, cancel_after: u64 };
                        fn check(raw: ?*anyopaque) !void {
                            const context: *Context = @ptrCast(@alignCast(raw.?));
                            if (context.fitter.dispatchCount() >= context.cancel_after) return error.Cancelled;
                        }
                    };
                    const live_before = tracker.snapshot().shared_backing_live_bytes;
                    const before = fitter.dispatchCount();
                    var context: CancelSecond.Context = .{ .fitter = &fitter, .cancel_after = before + 2 };
                    const cancelled = try a.alloc(f64, q);
                    defer a.free(cancelled);
                    @memset(cancelled, -123);
                    try std.testing.expectError(error.Cancelled, fitter.replayPage(&normalizer, head, queries, 0, 1, cancelled, .{ .context = &context, .check_fn = CancelSecond.check }));
                    try std.testing.expectEqual(before + 2, fitter.dispatchCount());
                    for (cancelled[0..16]) |value| try std.testing.expect(value != -123);
                    try std.testing.expectEqual(@as(f64, -123), cancelled[16]);
                    try std.testing.expectEqual(live_before, tracker.snapshot().shared_backing_live_bytes);
                    const fast_before = fitter.dispatchCount();
                    var fast_context: CancelSecond.Context = .{ .fitter = &fitter, .cancel_after = fast_before + 2 };
                    @memset(cancelled, -123);
                    try std.testing.expectError(error.Cancelled, fitter.replayCapturedPageQueries(&captured, 0, 1, 0, q, cancelled, .{ .context = &fast_context, .check_fn = CancelSecond.check }));
                    try std.testing.expectEqual(fast_before + 2, fitter.dispatchCount());
                    for (cancelled[0..16]) |value| try std.testing.expect(value != -123);
                    try std.testing.expectEqual(@as(f64, -123), cancelled[16]);
                    try std.testing.expectEqual(live_before, tracker.snapshot().shared_backing_live_bytes);
                }
            }
        }
    }
}

test "training capture reuses one fitter across heads at the established batch size" {
    if (!@import("build_options").enable_metal) return error.SkipZigTest;
    const a = std.testing.allocator;
    const head: solver.Head = .{ .rows = 2, .dim = 1, .keys = &.{ 1, -1 }, .values = &.{ 2, 3 }, .bias = &.{ 0, 0 }, .scale = 1, .softcap = 0 };
    var tracker = shared.BufferAllocationTracker.init(null, 0);
    var fitter = try Fitter.init(a, &tracker, 16 * 1024 * 1024);
    defer fitter.deinit();
    const capture = fitter.trainingCaptureBackend();
    try capture.configure_fn.?(capture.context, Fitter.default_query_batch_rows);
    const first = (try capture.begin_fn(capture.context, a, head, Fitter.default_query_batch_rows, .{})).?;
    capture.end_fn(first, a);
    try capture.configure_fn.?(capture.context, Fitter.default_query_batch_rows);
    const second = (try capture.begin_fn(capture.context, a, head, Fitter.default_query_batch_rows, .{})).?;
    capture.end_fn(second, a);
    try std.testing.expectError(error.InvalidFitConfig, capture.configure_fn.?(capture.context, Fitter.maximum_query_batch_rows));
    try std.testing.expectEqual(@as(u64, 10), fitter.training_capture_allocations);
    try std.testing.expectEqual(fitter.training_capture_allocations, fitter.training_capture_frees);
}

test "private FloatPair replay covers 32K ragged tail and rejects binding corruption" {
    if (!@import("build_options").enable_metal) return error.SkipZigTest;
    if (!@import("antfly_platform").env.getenvBool("ANTFLY_METAL_ATTENTION_MATCH_FLOATPAIR_DIAGNOSTIC")) return error.SkipZigTest;
    const a = std.testing.allocator;
    const n: usize = 32771;
    const d: usize = 1;
    const keys = try a.alloc(f64, n);
    defer a.free(keys);
    const values = try a.alloc(f64, n);
    defer a.free(values);
    const bias = try a.alloc(f64, n);
    defer a.free(bias);
    for (keys, values, bias, 0..) |*key, *value, *b, i| {
        key.* = @as(f64, @floatFromInt(i % 97)) / 97 - 0.5;
        value.* = @as(f64, @floatFromInt(i % 29)) / 29;
        b.* = @as(f64, @floatFromInt(i % 11)) / 100;
    }
    const queries = [_]f64{0.25};
    const head: solver.Head = .{ .keys = keys, .values = values, .bias = bias, .rows = n, .dim = d, .scale = 0.5, .softcap = 2 };
    var tracker = shared.BufferAllocationTracker.init(null, 0);
    var fitter = try Fitter.init(a, &tracker, 256 * 1024 * 1024);
    defer fitter.deinit();
    const cfg: solver.Config = .{ .target_rows = 1, .backend = fitter.backend(), .max_workspace_bytes = 256 * 1024 * 1024 };
    const full = try a.alloc(f64, n);
    defer a.free(full);
    try solver.attendBatch(a, head, &queries, full, null, null, cfg);
    var fail_pairs = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, fitter.captureNormalizer(fail_pairs.allocator(), head, &queries, .{}));
    var fail_outputs = std.testing.FailingAllocator.init(a, .{ .fail_index = 1 });
    try std.testing.expectError(error.OutOfMemory, fitter.captureNormalizer(fail_outputs.allocator(), head, &queries, .{}));
    const CancelCapture = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    const before_cancel = fitter.dispatchCount();
    try std.testing.expectError(error.Cancelled, fitter.captureNormalizer(a, head, &queries, .{ .check_fn = CancelCapture.check }));
    try std.testing.expectEqual(before_cancel, fitter.dispatchCount());
    var normalizer = try fitter.captureNormalizer(a, head, &queries, .{});
    defer normalizer.deinit();
    var tail: [3]f64 = undefined;
    try fitter.replayPage(&normalizer, head, &queries, 32768, 3, &tail, .{});
    try std.testing.expectEqualSlices(f64, full[32768..32771], &tail);
    var other = try Fitter.init(a, null, 256 * 1024 * 1024);
    defer other.deinit();
    try std.testing.expectError(error.InvalidFitConfig, other.replayPage(&normalizer, head, &queries, 32768, 3, &tail, .{}));
    var wrong_queries = queries;
    try std.testing.expectError(error.InvalidFitConfig, fitter.replayPage(&normalizer, head, &wrong_queries, 32768, 3, &tail, .{}));
    try std.testing.expectError(error.InvalidFitConfig, fitter.replayPage(&normalizer, head, &queries, n, 2, &tail, .{}));
    normalizer.layout = .column_major;
    try std.testing.expectError(error.InvalidFitConfig, fitter.replayPage(&normalizer, head, &queries, 32768, 3, &tail, .{}));
    normalizer.layout = .row_major;
    normalizer.batch_rows = 64;
    try std.testing.expectError(error.InvalidFitConfig, fitter.replayPage(&normalizer, head, &queries, 32768, 3, &tail, .{}));
    normalizer.batch_rows = 16;
    const saved = normalizer.pairs[1];
    normalizer.pairs[1] = 0;
    try std.testing.expectError(error.NonFiniteAttention, fitter.replayPage(&normalizer, head, &queries, 32768, 3, &tail, .{}));
    normalizer.pairs[1] = -1;
    try std.testing.expectError(error.NonFiniteAttention, fitter.replayPage(&normalizer, head, &queries, 32768, 3, &tail, .{}));
    normalizer.pairs[1] = std.math.inf(f32);
    try std.testing.expectError(error.NonFiniteAttention, fitter.replayPage(&normalizer, head, &queries, 32768, 3, &tail, .{}));
    normalizer.pairs[1] = saved;
    const saved_maximum = normalizer.pairs[0];
    normalizer.pairs[0] = std.math.nan(f32);
    try std.testing.expectError(error.NonFiniteAttention, fitter.replayPage(&normalizer, head, &queries, 32768, 3, &tail, .{}));
    normalizer.pairs[0] = saved_maximum;
    const live_before_oom = tracker.snapshot().shared_backing_live_bytes;
    const dispatches_before_oom = fitter.dispatchCount();
    for (0..5) |fail_index| {
        fitter.armTestProbabilityPageBufferFailure(fail_index);
        try std.testing.expectError(error.OutOfMemory, fitter.replayPage(&normalizer, head, &queries, 32768, 3, &tail, .{}));
        try std.testing.expectEqual(dispatches_before_oom, fitter.dispatchCount());
        try std.testing.expectEqual(live_before_oom, tracker.snapshot().shared_backing_live_bytes);
    }
    const cap = fitter.max_device_bytes;
    fitter.max_device_bytes = 1;
    try std.testing.expectError(error.CompactionWorkspaceLimit, fitter.replayPage(&normalizer, head, &queries, 32768, 3, &tail, .{}));
    fitter.max_device_bytes = cap;
    const Cancel = struct {
        const Context = struct { fitter: *Fitter, before: u64 };
        fn check(raw: ?*anyopaque) !void {
            const context: *Context = @ptrCast(@alignCast(raw.?));
            if (context.fitter.dispatchCount() > context.before) return error.Cancelled;
        }
    };
    const live_before_cancel = tracker.snapshot().shared_backing_live_bytes;
    @memset(&tail, -123);
    const dispatches_before_cancel = fitter.dispatchCount();
    var cancel_context: Cancel.Context = .{ .fitter = &fitter, .before = dispatches_before_cancel };
    try std.testing.expectError(error.Cancelled, fitter.replayPage(&normalizer, head, &queries, 32768, 3, &tail, .{ .context = &cancel_context, .check_fn = Cancel.check }));
    try std.testing.expectEqual(dispatches_before_cancel + 1, fitter.dispatchCount());
    for (tail) |value| try std.testing.expectEqual(@as(f64, -123), value);
    try std.testing.expectEqual(live_before_cancel, tracker.snapshot().shared_backing_live_bytes);
}

test "attention_match Metal diagnostic HD512 frontend matches independent FP64 across query batches" {
    if (!@import("build_options").enable_metal) return error.SkipZigTest;
    const a = std.testing.allocator;
    if (!@import("antfly_platform").env.getenvBool("ANTFLY_METAL_ATTENTION_MATCH_HD512_DIAGNOSTIC"))
        return error.SkipZigTest;

    const dim: usize = 512;
    const rows: usize = 1024;
    const Seed = struct {
        state: u64,

        fn uniform(self: *@This()) f64 {
            self.state = self.state *% 6364136223846793005 +% 1442695040888963407;
            const bits = (self.state >> 11) | 1;
            return @as(f64, @floatFromInt(bits)) * 0x1.0p-53;
        }

        fn normal(self: *@This()) f64 {
            const uniform_first = self.uniform();
            const uniform_second = self.uniform();
            return @sqrt(-2 * @log(uniform_first)) * @cos(2 * std.math.pi * uniform_second);
        }
    };
    const query_counts = [_]usize{ 80, 529 };
    const dynamic_scales = [_]f64{ 1, 4 };
    for (dynamic_scales, 0..) |dynamic, dynamic_case| {
        var random: Seed = .{ .state = 0x6a09e667f3bcc909 ^ @as(u64, @intCast(dynamic_case)) };
        const keys = try a.alloc(f64, rows * dim);
        defer a.free(keys);
        const values = try a.alloc(f64, rows * dim);
        defer a.free(values);
        const bias = try a.alloc(f64, rows);
        defer a.free(bias);
        for (keys, values, 0..) |*key, *value, i| {
            const branch = if (dynamic_case == 0 or (i / dim) % 7 != 0) 1.0 else dynamic;
            key.* = random.normal() * branch;
            value.* = random.normal();
        }
        for (bias, 0..) |*value, i| {
            const bias_scale: f64 = if (dynamic_case == 0 or i % 11 != 0) 0.25 else 6.0;
            value.* = random.normal() * bias_scale;
        }
        const head: solver.Head = .{
            .rows = rows,
            .dim = dim,
            .keys = keys,
            .values = values,
            .bias = bias,
            .scale = 1.0 / @sqrt(@as(f64, @floatFromInt(dim))),
            .softcap = if (dynamic_case == 0) 0 else 12,
        };
        for (query_counts) |query_rows| {
            const queries = try a.alloc(f64, query_rows * dim);
            defer a.free(queries);
            for (queries, 0..) |*query, i| {
                const branch = if (dynamic_case == 0 or (i / dim) % 5 != 0) 1.0 else dynamic;
                query.* = random.normal() * branch;
            }
            const expected_p = try a.alloc(f64, query_rows * rows);
            defer a.free(expected_p);
            const actual_p = try a.alloc(f64, query_rows * rows);
            defer a.free(actual_p);
            const expected_y = try a.alloc(f64, queries.len);
            defer a.free(expected_y);
            const actual_y = try a.alloc(f64, queries.len);
            defer a.free(actual_y);
            const expected_mass = try a.alloc(f64, query_rows);
            defer a.free(expected_mass);
            const actual_mass = try a.alloc(f64, query_rows);
            defer a.free(actual_mass);

            var config: solver.Config = .{ .target_rows = rows / 2 };
            try solver.attendBatch(a, head, queries, expected_p, expected_y, expected_mass, config);
            const budget = try Fitter.workspace(null, rows, 1, query_rows, dim);
            try std.testing.expect(budget <= 128 * 1024 * 1024);
            var tracker = shared.BufferAllocationTracker.init(null, 0);
            var fitter = try Fitter.init(a, &tracker, budget);
            defer fitter.deinit();
            const tracker_baseline = tracker.snapshot().shared_backing_live_bytes;
            config.backend = fitter.backend();
            try solver.attendBatch(a, head, queries, actual_p, actual_y, actual_mass, config);
            try std.testing.expectEqual(@as(u64, @intCast(std.math.divCeil(usize, query_rows, Fitter.default_query_batch_rows) catch unreachable)), fitter.dispatches);
            var max_p: f64 = 0;
            var max_y: f64 = 0;
            var max_mass: f64 = 0;
            for (expected_p, actual_p) |want, got| {
                try std.testing.expect(std.math.isFinite(got));
                max_p = @max(max_p, @abs(want - got));
            }
            for (expected_y, actual_y) |want, got| {
                try std.testing.expect(std.math.isFinite(got));
                max_y = @max(max_y, @abs(want - got));
            }
            for (expected_mass, actual_mass) |want, got| {
                try std.testing.expect(std.math.isFinite(got));
                max_mass = @max(max_mass, @abs(want - got));
            }
            try std.testing.expect(max_p <= 2e-5);
            try std.testing.expect(max_y <= 2e-4);
            try std.testing.expect(max_mass <= 2e-4);
            try std.testing.expect(tracker.snapshot().shared_backing_peak_bytes - tracker_baseline <= budget);
            try std.testing.expectEqual(tracker_baseline, tracker.snapshot().shared_backing_live_bytes);
        }
    }
}

test "attention_match Metal checkpoint fitter matches FP64 with prior bias after softcap" {
    if (!@import("build_options").enable_metal) return error.SkipZigTest;
    const a = std.testing.allocator;
    const head: solver.Head = .{
        .rows = 4,
        .dim = 2,
        .keys = &.{ 1, 0, 0, 1, -1, 0, 0, -1 },
        .values = &.{ 1, 2, 3, 4, 5, 6, 7, 8 },
        .bias = &.{ 0.25, -0.5, 1.25, -1.5 },
        .scale = 0.75,
        .softcap = 0.5,
    };
    const queries = [_]f64{ 1, 0.5, -1, 1, 2, -2 };
    var expected_p: [12]f64 = undefined;
    var expected_y: [6]f64 = undefined;
    var expected_z: [3]f64 = undefined;
    const config: solver.Config = .{ .target_rows = 2 };
    try solver.attendBatch(a, head, &queries, &expected_p, &expected_y, &expected_z, config);
    var fitter = try Fitter.init(a, null, 1024 * 1024);
    defer fitter.deinit();
    var p: [12]f64 = undefined;
    var y: [6]f64 = undefined;
    var z: [3]f64 = undefined;
    var accelerated = config;
    accelerated.backend = fitter.backend();
    try solver.attendBatch(a, head, &queries, &p, &y, &z, accelerated);
    for (expected_p, p) |expected, actual| try std.testing.expectApproxEqAbs(expected, actual, 2e-5);
    for (expected_y, y) |expected, actual| try std.testing.expectApproxEqAbs(expected, actual, 2e-5);
    for (expected_z, z) |expected, actual| try std.testing.expectApproxEqAbs(expected, actual, 2e-5);
    try solver.attendSummary(a, head, &queries, &y, &z, accelerated);
    for (expected_y, y) |expected, actual| try std.testing.expectApproxEqAbs(expected, actual, 2e-5);
    for (expected_z, z) |expected, actual| try std.testing.expectApproxEqAbs(expected, actual, 2e-5);
    try std.testing.expectEqual(@as(u64, 2), fitter.dispatches);
    const before = fitter.dispatches;
    fitter.max_device_bytes = 1;
    try std.testing.expectError(error.CompactionWorkspaceLimit, solver.attendBatch(a, head, &queries, &p, &y, &z, accelerated));
    try std.testing.expectEqual(before, fitter.dispatches);
}

test "attention_match Metal checkpoint fitter includes shared backing in pure admission" {
    const bytes = try Fitter.workspace(null, 512, 128, 128, 512);
    try std.testing.expect(bytes > 512 * 512 * 8);
    // Gemma4 E2B's shared global owner contributes 40 consumer/GQA strata.
    // Every fifth study query is held out: 320 rows leave 256 for training.
    const e2b = try Fitter.workspace(null, 1024, 512, 40 * 256, 512);
    try std.testing.expect(e2b > bytes);
    try std.testing.expectEqual(try Fitter.workspace(null, 1024, 512, Fitter.maximum_query_batch_rows, 512), e2b);
    const config: solver.Config = .{ .target_rows = 512, .backend = Fitter.backendForContext(null) };
    const solver_bytes = try solver.fitWorkspaceBytes(1024, 512, 40 * 256, 512, 40 * 64, config);
    try std.testing.expect(solver_bytes <= config.max_workspace_bytes);
    try std.testing.expectError(error.InvalidAttentionShape, Fitter.workspace(null, 1024, 512, Fitter.max_query_rows + 1, 512));
    try std.testing.expectError(error.InvalidAttentionShape, Fitter.workspace(null, 0, 1, 1, 512));
}

test "E2 q30720 wide capture is batch bounded and admission rejects one byte short" {
    const n: usize = 32_512;
    const q: usize = 30_720;
    const d: usize = 512;
    const batch: usize = 64;
    try std.testing.expectError(error.InvalidAttentionShape, Fitter.workspaceBytesShape(n, q, d, batch));
    const tiled = try Fitter.workspaceBytesShapeLimit(n, q, d, batch, Fitter.max_wide_capture_query_rows);
    const query_elements = try std.math.mul(usize, q, d);
    const retained = try std.math.mul(usize, try std.math.add(usize, query_elements, try std.math.mul(usize, q, 2)), @sizeOf(f32));
    const capture_peak = try std.math.add(usize, tiled, try std.math.add(usize, retained, std.heap.pageSize()));
    const required = @max(capture_peak, retained);
    try Fitter.admitCapturedPage(0, required, capture_peak, retained);
    try std.testing.expectError(error.CompactionWorkspaceLimit, Fitter.admitCapturedPage(0, required - 1, capture_peak, retained));
    const forbidden_full_probability_bytes = try std.math.mul(usize, try std.math.mul(usize, q, n), @sizeOf(f32));
    try std.testing.expect(tiled < forbidden_full_probability_bytes);
}

test "E2 q30720 real Metal wide capture batches64 and256 agree under exact cap cancel and recovery" {
    if (comptime builtin.os.tag != .macos or !@import("build_options").enable_metal) return error.SkipZigTest;
    const a = std.testing.allocator;
    const rows: usize = 8;
    // Real q30720/N8/D512 capture: N is deliberately small so this isolates
    // query batching while the preceding pure fixture binds N32512 admission.
    const dim: usize = 512;
    const q: usize = 30_720;
    const keys = try a.alloc(f64, rows * dim);
    defer a.free(keys);
    const values = try a.alloc(f64, rows * dim);
    defer a.free(values);
    const queries = try a.alloc(f64, q * dim);
    defer a.free(queries);
    for (keys, 0..) |*value, i| value.* = (@as(f64, @floatFromInt(i % 29)) - 14) / 64;
    for (values, 0..) |*value, i| value.* = (@as(f64, @floatFromInt(i % 17)) - 8) / 32;
    for (queries, 0..) |*value, i| value.* = (@as(f64, @floatFromInt(i % 37)) - 18) / 128;
    const head: solver.Head = .{ .rows = rows, .dim = dim, .keys = keys, .values = values, .bias = &.{ 0.25, -0.5, 0.75, -1, 0.125, 0.5, -0.25, 0 }, .scale = 0.04419417382415922, .softcap = 1.5 };
    var tracker = shared.BufferAllocationTracker.init(null, 0);
    const baseline = tracker.snapshot().shared_backing_live_bytes;
    var expected: [64 * rows]f64 = undefined;
    inline for (.{ @as(usize, 64), @as(usize, 256) }, 0..) |batch_rows, pass| {
        // This instance computes bounds only and allocates no page buffers.
        // Actual capture/replay fitters below retain their exact caps and
        // cap-minus-one rejection checks.
        var sizing = try Fitter.init(a, &tracker, std.math.maxInt(usize));
        try sizing.setQueryBatchRows(batch_rows);
        const capture_peak = try sizing.captureWorkspaceBytesWide(head, queries.len);
        const retained = try sizing.capturedPageRetainedBytes(head, queries.len);
        // Capture and replay have distinct exact peaks. The retained captured
        // source remains live while replay owns its result page, so a cap that
        // admits capture alone is intentionally insufficient for the complete
        // workflow.
        var probe = try Fitter.init(a, &tracker, std.math.maxInt(usize));
        try probe.setQueryBatchRows(batch_rows);
        var probe_source = try probe.capturePageSourceWide(a, head, queries, 0, std.math.maxInt(usize), .{});
        const replay_transient = try probe.capturedPageTransientBytes(&probe_source, rows, 64);
        probe_source.deinit();
        probe.deinit();
        sizing.deinit();
        try std.testing.expectEqual(baseline, tracker.snapshot().shared_backing_live_bytes);
        const capture_cap = @max(capture_peak, retained);
        const exact_cap = @max(capture_peak, try add(retained, replay_transient));
        try std.testing.expect(exact_cap > capture_cap);
        try std.testing.expectEqual(baseline, tracker.snapshot().shared_backing_live_bytes);

        var fitter = try Fitter.init(a, &tracker, exact_cap);
        errdefer fitter.deinit();
        try fitter.setQueryBatchRows(batch_rows);
        const fitter_baseline = tracker.snapshot().shared_backing_live_bytes;
        try std.testing.expectError(error.CompactionWorkspaceLimit, fitter.capturePageSourceWide(a, head, queries, 0, capture_cap - 1, .{}));
        var replay_short = try Fitter.init(a, &tracker, exact_cap - 1);
        try replay_short.setQueryBatchRows(batch_rows);
        var replay_short_source = try replay_short.capturePageSourceWide(a, head, queries, 0, capture_cap, .{});
        var replay_short_out: [64 * rows]f64 = undefined;
        try std.testing.expectError(error.CompactionWorkspaceLimit, replay_short.replayCapturedPageQueries(&replay_short_source, 0, rows, q - 64, 64, &replay_short_out, .{}));
        replay_short_source.deinit();
        replay_short.deinit();
        try std.testing.expectEqual(fitter_baseline, tracker.snapshot().shared_backing_live_bytes);
        const Cancel = struct {
            checks: usize = 0,
            fn check(raw: ?*anyopaque) !void {
                const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
                self.checks += 1;
                if (self.checks == 4) return error.Cancelled;
            }
        };
        var cancel: Cancel = .{};
        try std.testing.expectError(error.Cancelled, fitter.capturePageSourceWide(a, head, queries, 0, exact_cap, .{ .context = &cancel, .check_fn = Cancel.check }));
        try std.testing.expectEqual(@as(usize, 4), cancel.checks);
        var captured = try fitter.capturePageSourceWide(a, head, queries, 0, exact_cap, .{});
        errdefer captured.deinit();
        var actual: [64 * rows]f64 = undefined;
        try fitter.replayCapturedPageQueries(&captured, 0, rows, q - 64, 64, &actual, .{});
        for (actual) |value| try std.testing.expect(std.math.isFinite(value) and value >= 0);
        var logits: [rows]f64 = undefined;
        var maximum = -std.math.inf(f64);
        const last_query = queries[(q - 1) * dim ..][0..dim];
        for (0..rows) |row| {
            var dot: f64 = 0;
            for (0..dim) |column| dot += last_query[column] * keys[row * dim + column];
            var logit = dot * head.scale;
            logit = head.softcap * std.math.tanh(logit / head.softcap) + head.bias.?[row];
            logits[row] = logit;
            maximum = @max(maximum, logit);
        }
        var denominator: f64 = 0;
        for (&logits) |*logit| {
            logit.* = @exp(logit.* - maximum);
            denominator += logit.*;
        }
        for (logits, actual[63 * rows ..][0..rows]) |numerator, probability|
            try std.testing.expectApproxEqAbs(numerator / denominator, probability, 2e-5);
        if (pass == 0) expected = actual else try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&expected), std.mem.sliceAsBytes(&actual));
        try std.testing.expectEqual(q, captured.query_count);
        try std.testing.expectEqual(batch_rows, captured.batch_rows);
        captured.deinit();
        fitter.deinit();
    }
    try std.testing.expectEqual(baseline, tracker.snapshot().shared_backing_live_bytes);
}

test "attention_match Metal pure workspace admits only paired diagnostic batch 256" {
    const rows: usize = 1024;
    const queries: usize = 768;
    const dim: usize = 512;
    const batch64 = try Fitter.workspaceBytesShape(rows, queries, dim, Fitter.maximum_query_batch_rows);
    const batch256 = try Fitter.workspaceBytesShape(rows, queries, dim, Fitter.diagnostic_query_batch_rows);
    try std.testing.expect(batch256 > batch64);
    const query64 = try rounded(try mul(try mul(Fitter.maximum_query_batch_rows, dim), 4));
    const query256 = try rounded(try mul(try mul(Fitter.diagnostic_query_batch_rows, dim), 4));
    const result_width = try add(try add(rows, dim), 1);
    const result64 = try rounded(try mul(try mul(Fitter.maximum_query_batch_rows, result_width), 4));
    const result256 = try rounded(try mul(try mul(Fitter.diagnostic_query_batch_rows, result_width), 4));
    try std.testing.expectEqual((query256 - query64) + (result256 - result64), batch256 - batch64);
    try std.testing.expectError(error.InvalidAttentionShape, Fitter.workspaceBytesShape(rows, queries, dim, 65));
    try std.testing.expectError(error.InvalidAttentionShape, Fitter.workspaceBytesShape(rows, queries, dim, 257));
}

test "attention_match Metal parallel plan bounds independent workers under two GiB" {
    const aggregate = 2 * 1024 * 1024 * 1024;
    const source = 512 * 1024 * 1024;
    const workspace = 200 * 1024 * 1024;
    const plan = try Fitter.parallelPlan(source, 8, workspace, aggregate, 4);
    try std.testing.expectEqual(@as(usize, 4), plan.workers);
    try std.testing.expectEqual(workspace, plan.per_worker_bytes);
    try std.testing.expectEqual(
        try engine.parallelHostBytes(source, 8, workspace, 4),
        plan.admitted_bytes,
    );
    try std.testing.expect(plan.admitted_bytes <= aggregate);

    const one_worker_limit = try engine.parallelHostBytes(source, 8, workspace, 1);
    const serial = try Fitter.parallelPlan(source, 8, workspace, one_worker_limit, 4);
    try std.testing.expectEqual(@as(usize, 1), serial.workers);
    try std.testing.expectError(
        error.CompactionWorkspaceLimit,
        Fitter.parallelPlan(source, 8, workspace, one_worker_limit - 1, 4),
    );
    try std.testing.expectError(
        error.InvalidFitConfig,
        Fitter.parallelPlan(source, 8, workspace, aggregate, 5),
    );
    try std.testing.expectError(
        error.CompactionWorkspaceLimit,
        Fitter.parallelPlan(std.math.maxInt(usize), 8, workspace, aggregate, 4),
    );
}

test "attention_match Metal fitter bounds resident query chunks and preserves a ragged tail" {
    if (!@import("build_options").enable_metal) return error.SkipZigTest;
    const a = std.testing.allocator;
    const rows = 64;
    const dim = 64;
    const query_rows = 35;
    var keys: [rows * dim]f64 = undefined;
    var values: [rows * dim]f64 = undefined;
    var queries: [query_rows * dim]f64 = undefined;
    var bias: [rows]f64 = undefined;
    for (&keys, &values, 0..) |*k, *v, i| {
        k.* = (@as(f64, @floatFromInt(i % 11)) - 5) / 16;
        v.* = (@as(f64, @floatFromInt(i % 7)) - 3) / 8;
    }
    for (&queries, 0..) |*q, i| q.* = (@as(f64, @floatFromInt(i % 13)) - 6) / 8;
    for (&bias, 0..) |*b, i| b.* = if (i % 2 == 0) 0.25 else -0.5;
    const head: solver.Head = .{ .rows = rows, .dim = dim, .keys = &keys, .values = &values, .bias = &bias, .scale = 0.125, .softcap = 1.5 };
    var expected_p: [query_rows * rows]f64 = undefined;
    var expected_y: [query_rows * dim]f64 = undefined;
    var expected_z: [query_rows]f64 = undefined;
    var p: [query_rows * rows]f64 = undefined;
    var y: [query_rows * dim]f64 = undefined;
    var z: [query_rows]f64 = undefined;
    const budget = try Fitter.workspace(null, rows, 1, Fitter.maximum_query_batch_rows, dim);
    var tracker = shared.BufferAllocationTracker.init(null, 0);
    {
        var fitter = try Fitter.init(a, &tracker, budget);
        defer fitter.deinit();
        const baseline = tracker.snapshot().shared_backing_live_bytes;
        var config: solver.Config = .{ .target_rows = 1 };
        try solver.attendBatch(a, head, &queries, &expected_p, &expected_y, &expected_z, config);
        config.backend = fitter.backend();
        try solver.attendBatch(a, head, &queries, &p, &y, &z, config);
        try std.testing.expectEqual(@as(u64, 3), fitter.dispatches);
        try std.testing.expectEqual(@as(u64, 1), fitter.attentionCallCount());
        for (expected_p, p) |want, got| try std.testing.expectApproxEqAbs(want, got, 2e-5);
        for (expected_y, y) |want, got| try std.testing.expectApproxEqAbs(want, got, 2e-5);
        for (expected_z, z) |want, got| try std.testing.expectApproxEqAbs(want, got, 2e-5);
        try solver.attendSummary(a, head, &queries, &y, &z, config);
        try std.testing.expectEqual(@as(u64, 6), fitter.dispatches);
        try std.testing.expectEqual(@as(u64, 2), fitter.attentionCallCount());
        for (expected_y, y) |want, got| try std.testing.expectApproxEqAbs(want, got, 2e-5);
        for (expected_z, z) |want, got| try std.testing.expectApproxEqAbs(want, got, 2e-5);
        try std.testing.expectEqual(@as(u64, 2 * query_rows * (rows + dim + 1) * 4), fitter.derived_readback_bytes);
        try std.testing.expect(tracker.snapshot().shared_backing_peak_bytes - baseline <= budget);
        try std.testing.expectEqual(baseline, tracker.snapshot().shared_backing_live_bytes);
    }
    try std.testing.expectEqual(@as(u64, 0), tracker.snapshot().shared_backing_live_bytes);
}

test "attention_match Metal fitter chunks queries and drains tracked buffers on cancellation" {
    if (!@import("build_options").enable_metal) return error.SkipZigTest;
    const a = std.testing.allocator;
    var tracker = shared.BufferAllocationTracker.init(null, 0);
    {
        var fitter = try Fitter.init(a, &tracker, 16 * 1024 * 1024);
        defer fitter.deinit();
        const baseline = tracker.snapshot().shared_backing_live_bytes;
        const head: solver.Head = .{
            .rows = 4,
            .dim = 2,
            .keys = &.{ 1, 0, 0, 1, -1, 0, 0, -1 },
            .values = &.{ 1, 2, 3, 4, 5, 6, 7, 8 },
            .scale = 0.75,
        };
        var queries: [34]f64 = undefined;
        for (&queries, 0..) |*value, i| value.* = @as(f64, @floatFromInt(i % 5)) - 2;
        var probabilities: [68]f64 = @splat(-123);
        var targets: [34]f64 = @splat(-123);
        var masses: [17]f64 = @splat(-123);
        const CancelAfterDispatch = struct {
            fn check(raw: ?*anyopaque) !void {
                const current: *Fitter = @ptrCast(@alignCast(raw.?));
                if (current.dispatches > 0) return error.Cancelled;
            }
        };
        var config: solver.Config = .{
            .target_rows = 2,
            .backend = fitter.backend(),
            .control = .{ .context = &fitter, .check_fn = CancelAfterDispatch.check },
        };
        try std.testing.expectError(error.Cancelled, solver.attendBatch(a, head, &queries, &probabilities, &targets, &masses, config));
        try std.testing.expectEqual(@as(u64, 1), fitter.dispatches);
        for (probabilities) |value| try std.testing.expectEqual(@as(f64, -123), value);
        try std.testing.expectEqual(baseline, tracker.snapshot().shared_backing_live_bytes);
        try std.testing.expect(tracker.snapshot().shared_backing_peak_bytes > baseline);
        config.control = .{};
        try solver.attendBatch(a, head, &queries, &probabilities, &targets, &masses, config);
        try std.testing.expectEqual(@as(u64, 3), fitter.dispatches);
        var reference_p: [68]f64 = undefined;
        var reference_y: [34]f64 = undefined;
        var reference_z: [17]f64 = undefined;
        config.backend = null;
        try solver.attendBatch(a, head, &queries, &reference_p, &reference_y, &reference_z, config);
        for (reference_p, probabilities) |expected, actual| try std.testing.expectApproxEqAbs(expected, actual, 2e-5);
        for (reference_y, targets) |expected, actual| try std.testing.expectApproxEqAbs(expected, actual, 2e-5);
        for (reference_z, masses) |expected, actual| try std.testing.expectApproxEqAbs(expected, actual, 2e-5);
        try std.testing.expectEqual(baseline, tracker.snapshot().shared_backing_live_bytes);
    }
    try std.testing.expectEqual(@as(u64, 0), tracker.snapshot().shared_backing_live_bytes);
}

test "attention_match Metal query batch16 and batch64 are bit identical across tails" {
    if (!@import("build_options").enable_metal) return error.SkipZigTest;
    const a = std.testing.allocator;
    const head: solver.Head = .{ .rows = 4, .dim = 2, .keys = &.{ 1, 0, 0, 1, -1, 0, 0, -1 }, .values = &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, .bias = &.{ 0.25, -0.5, 0.75, -1 }, .scale = 0.75, .softcap = 1.5 };
    var queries: [65 * 2]f64 = undefined;
    for (&queries, 0..) |*value, i| value.* = (@as(f64, @floatFromInt(i % 11)) - 5) / 8;
    for ([_]usize{ 1, 16, 17, 65 }) |q| {
        var p16: [65 * 4]f64 = undefined;
        var y16: [65 * 2]f64 = undefined;
        var z16: [65]f64 = undefined;
        var p64: [65 * 4]f64 = undefined;
        var y64: [65 * 2]f64 = undefined;
        var z64: [65]f64 = undefined;
        var pc: [65 * 4]f64 = undefined;
        var yc: [65 * 2]f64 = undefined;
        var zc: [65]f64 = undefined;
        const budget = try Fitter.workspace(null, head.rows, 1, q, head.dim);
        var fitter16 = try Fitter.init(a, null, budget);
        defer fitter16.deinit();
        const c16: solver.Config = .{ .target_rows = 1, .backend = fitter16.backend() };
        try solver.attendBatch(a, head, queries[0 .. q * 2], p16[0 .. q * 4], y16[0 .. q * 2], z16[0..q], c16);
        var fitter64 = try Fitter.init(a, null, budget);
        defer fitter64.deinit();
        try fitter64.setQueryBatchRows(64);
        const c64: solver.Config = .{ .target_rows = 1, .backend = fitter64.backend() };
        try solver.attendBatch(a, head, queries[0 .. q * 2], p64[0 .. q * 4], y64[0 .. q * 2], z64[0..q], c64);
        var column = try Fitter.init(a, null, budget);
        defer column.deinit();
        try column.setKeyLayout(.column_major);
        const cc: solver.Config = .{ .target_rows = 1, .backend = column.backend() };
        try solver.attendBatch(a, head, queries[0 .. q * 2], pc[0 .. q * 4], yc[0 .. q * 2], zc[0..q], cc);
        try std.testing.expectEqualSlices(f64, p16[0 .. q * 4], p64[0 .. q * 4]);
        try std.testing.expectEqualSlices(f64, y16[0 .. q * 2], y64[0 .. q * 2]);
        try std.testing.expectEqualSlices(f64, z16[0..q], z64[0..q]);
        try std.testing.expectEqualSlices(f64, p16[0 .. q * 4], pc[0 .. q * 4]);
        try std.testing.expectEqualSlices(f64, y16[0 .. q * 2], yc[0 .. q * 2]);
        try std.testing.expectEqualSlices(f64, z16[0..q], zc[0..q]);
        try std.testing.expectError(error.InvalidFitConfig, column.setKeyLayout(.row_major));
        try std.testing.expectEqual(@as(u64, 1), fitter16.attentionCallCount());
        try std.testing.expectEqual(@as(u64, 1), fitter64.attentionCallCount());
        try std.testing.expectEqual(@as(u64, @intCast(std.math.divCeil(usize, q, 16) catch unreachable)), fitter16.dispatchCount());
        try std.testing.expectEqual(@as(u64, @intCast(std.math.divCeil(usize, q, 64) catch unreachable)), fitter64.dispatchCount());
        try std.testing.expectError(error.InvalidFitConfig, fitter64.setQueryBatchRows(16));
        if (budget == 6 * std.heap.pageSize()) {
            try std.testing.expectError(error.CompactionWorkspaceLimit, Fitter.init(a, null, budget - 1));
        } else {
            var too_small = try Fitter.init(a, null, budget - 1);
            defer too_small.deinit();
            try too_small.setQueryBatchRows(64);
            const small: solver.Config = .{ .target_rows = 1, .backend = too_small.backend() };
            try std.testing.expectError(error.CompactionWorkspaceLimit, solver.attendBatch(a, head, queries[0 .. q * 2], p64[0 .. q * 4], y64[0 .. q * 2], z64[0..q], small));
            try std.testing.expectEqual(@as(u64, 0), too_small.dispatchCount());
            try std.testing.expectEqual(@as(u64, 0), too_small.attentionCallCount());
        }
    }
    // A production-sized D64/q65 shape is above the constructor floor. Prove
    // cap-1 reaches the callback admission check, dispatches no GPU work, and
    // leaves every transient buffer released.
    const large_rows = 64;
    const large_dim = 64;
    const large_q = 65;
    const large_keys: [large_rows * large_dim]f64 = @splat(0);
    const large_values: [large_rows * large_dim]f64 = @splat(0);
    const large_bias: [large_rows]f64 = @splat(0);
    const large_queries: [large_q * large_dim]f64 = @splat(0);
    var large_p: [large_q * large_rows]f64 = undefined;
    var large_y: [large_q * large_dim]f64 = undefined;
    var large_z: [large_q]f64 = undefined;
    const large_head: solver.Head = .{ .rows = large_rows, .dim = large_dim, .keys = &large_keys, .values = &large_values, .bias = &large_bias, .scale = 0.125 };
    const large_budget = try Fitter.workspace(null, large_rows, 1, large_q, large_dim);
    try std.testing.expect(large_budget > 6 * std.heap.pageSize());
    var tracker = shared.BufferAllocationTracker.init(null, 0);
    var capped = try Fitter.init(a, &tracker, large_budget - 1);
    defer capped.deinit();
    try capped.setQueryBatchRows(64);
    const baseline = tracker.snapshot().shared_backing_live_bytes;
    const capped_config: solver.Config = .{ .target_rows = 1, .backend = capped.backend() };
    try std.testing.expectError(error.CompactionWorkspaceLimit, solver.attendBatch(a, large_head, &large_queries, &large_p, &large_y, &large_z, capped_config));
    try std.testing.expectEqual(@as(u64, 0), capped.dispatchCount());
    try std.testing.expectEqual(@as(u64, 0), capped.attentionCallCount());
    try std.testing.expectEqual(baseline, tracker.snapshot().shared_backing_live_bytes);
    var invalid = try Fitter.init(a, null, 1024 * 1024);
    defer invalid.deinit();
    try std.testing.expectError(error.InvalidFitConfig, invalid.setQueryBatchRows(32));
}

test "attention_match Metal capture configures 256 then 64 before workspace admission" {
    if (!@import("build_options").enable_metal) return error.SkipZigTest;
    const head: solver.Head = .{ .rows = 4, .dim = 2, .keys = &.{ 1, 0, 0, 1, -1, 0, 0, -1 }, .values = &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, .bias = &.{ 0, 0, 0, 0 }, .scale = 0.5 };
    // This fixture tests configuration order; capture has a distinct five-buffer workspace.
    const budget = 1024 * 1024;
    var fitter = try Fitter.init(std.testing.allocator, null, budget);
    defer fitter.deinit();
    const backend = fitter.trainingCaptureBackend();
    const configure = backend.configure_fn orelse return error.InvalidFitConfig;
    try configure(backend.context, 256);
    try std.testing.expectEqual(@as(usize, 256), fitter.queryBatchRows());
    try std.testing.expect((try backend.workspace_fn(backend.context, head, 256)) <= budget);
    try configure(backend.context, 64);
    try std.testing.expectEqual(@as(usize, 64), fitter.queryBatchRows());
    try std.testing.expect((try backend.workspace_fn(backend.context, head, 64)) <= budget);
    try std.testing.expectEqual(@as(u64, 0), fitter.attentionCallCount());
    try std.testing.expectEqual(@as(u64, 0), fitter.dispatchCount());
}

test "attention_match Metal engine fits and roundtrips FP8 through the shared checkpoint contract" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    const checkpoint = @import("../runtime/kv/attention_match_checkpoint.zig");
    const a = std.testing.allocator;
    const state: checkpoint.State = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .logical_length = 6,
        .tokens = &.{ 0, 0, 0, 0, 0, 0 },
        .logits = &.{0},
        .layer_donors = &.{0},
        .donors = &.{.{ .heads = &.{.{
            .dim = 2,
            .positions = &.{ 0, 1, 2, 3, 4, 5 },
            .coverage_ends = &.{ 1, 2, 3, 4, 5, 6 },
            .keys = &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            .values = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 },
            .bias = &.{ 0, 0, 0, 0, 0, 0 },
        }} }},
    };
    const original = try checkpoint.encodeWithEncoding(a, state, .{}, .f16, .{});
    defer a.free(original);
    const batches = [_]engine.ReferenceBatch{.{
        .consumer_layer = 0,
        .query_head = 0,
        .train = &.{ 0.25, 0.5, 1, 0, 0, 1, -0.5, 0.25 },
        .heldout = &.{ -1, 1, 0.5, -0.5 },
        .train_positions = &.{ 6, 7, 8, 9 },
        .heldout_positions = &.{ 10, 11 },
    }};
    const layout: engine.Layout = .{ .query_heads_by_layer = &.{1}, .scales_by_donor = &.{1}, .softcaps_by_donor = &.{0} };
    var fitter = try Fitter.init(a, null, 16 * 1024 * 1024);
    defer fitter.deinit();
    var stats: engine.Stats = .{};
    var result = try engine.prepare(a, state, &batches, layout, .{
        .stats = &stats,
        .retention_divisor = 2,
        .min_retention_divisor = 2,
        .prefix_rows = 1,
        .recent_rows = 1,
        .chunk_rows = 2,
        .storage_precision = .f16,
        .validation = .whole_head,
        .fp8_checkpoint = true,
        .fit = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .backend = fitter.backend() },
    });
    defer result.deinit();
    const head = result.state.donors[0].heads[0];
    try std.testing.expectEqual(@as(usize, 4), head.positions.len);
    try std.testing.expectEqual(@as(usize, 1), stats.validated_heads);
    try std.testing.expectEqual(@as(usize, 1), stats.fp8_heads);
    try std.testing.expect(fitter.dispatches > 0);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2 }, head.values[0..2]);
    try std.testing.expectEqualSlices(f32, &.{ 11, 12 }, head.values[head.values.len - 2 ..]);
    const wire = try checkpoint.encodeWithEncoding(a, result.state, .{}, .fp8, .{});
    defer a.free(wire);
    var restored = try checkpoint.decode(a, wire, state.identity, .{});
    defer restored.deinit();
    try std.testing.expectEqualDeep(head.fp8, restored.state.donors[0].heads[0].fp8);
    try std.testing.expectEqualSlices(f32, head.values, restored.state.donors[0].heads[0].values);
    try std.testing.expectEqualSlices(f32, head.bias, restored.state.donors[0].heads[0].bias);
    const unchanged = try checkpoint.encodeWithEncoding(a, state, .{}, .f16, .{});
    defer a.free(unchanged);
    try std.testing.expectEqualSlices(u8, original, unchanged);
}

test "context private typed capture summary executes before unchanged quality rejection" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    const checkpoint = @import("../runtime/kv/attention_match_checkpoint.zig");
    const adaptive = @import("../runtime/kv/attention_match_joint_adaptive_quality.zig");
    const a = std.testing.allocator;
    const rows: usize = 512;
    const dim: usize = 64;
    const positions = try a.alloc(u64, rows);
    defer a.free(positions);
    const coverage = try a.alloc(u64, rows);
    defer a.free(coverage);
    const keys = try a.alloc(f32, rows * dim);
    defer a.free(keys);
    const values = try a.alloc(f32, rows * dim);
    defer a.free(values);
    const bias = try a.alloc(f32, rows);
    defer a.free(bias);
    for (0..rows) |ri| {
        positions[ri] = @intCast(ri);
        coverage[ri] = @intCast(ri + 1);
        bias[ri] = 0;
        for (0..dim) |di| {
            keys[ri * dim + di] = @floatFromInt((ri + 3 * di) % 17);
            values[ri * dim + di] = @floatFromInt((5 * ri + di) % 19);
        }
    }
    const tokens = try a.alloc(u32, rows);
    defer a.free(tokens);
    @memset(tokens, 0);
    const state: checkpoint.State = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .logical_length = @intCast(rows),
        .tokens = tokens,
        .logits = &.{0},
        .layer_donors = &.{0},
        .donors = &.{.{ .heads = &.{.{ .dim = dim, .positions = positions, .coverage_ends = coverage, .keys = keys, .values = values, .bias = bias }} }},
    };
    const train = try a.alloc(f32, 8 * dim);
    defer a.free(train);
    const heldout = try a.alloc(f32, 2 * dim);
    defer a.free(heldout);
    for (train, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) / 13;
    for (heldout, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) / 11;
    const batch: engine.ReferenceBatch = .{ .consumer_layer = 0, .query_head = 0, .train = train, .heldout = heldout, .train_positions = &.{ 512, 513, 514, 515, 516, 517, 518, 519 }, .heldout_positions = &.{ 520, 521 } };
    try std.testing.expectEqual(@as(usize, 8), batch.train_positions.len);
    try std.testing.expectEqual(@as(usize, 2), batch.heldout_positions.len);
    for (batch.train_positions) |train_position| for (batch.heldout_positions) |heldout_position|
        try std.testing.expect(train_position != heldout_position);
    const layout: engine.Layout = .{ .query_heads_by_layer = &.{1}, .scales_by_donor = &.{1}, .softcaps_by_donor = &.{0} };
    const original = try checkpoint.encodeWithEncoding(a, state, .{}, .f16, .{});
    defer a.free(original);
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    for ([_]checkpoint.Encoding{ .f16, .fp8, .rotated4 }) |encoding| {
        for ([_]engine.Options.PrivatePerformancePolicy{ .owned_capture_summary_v1, .owned_capture_summary_incremental_v1, .owned_capture_summary_incremental_reduction_v1 }) |private_policy| {
            var fitter = try Fitter.init(a, null, 1 << 30);
            defer fitter.deinit();
            try fitter.setQueryBatchRows(64);
            var stats: engine.Stats = .{};
            var options: engine.Options = .{
                .stats = &stats,
                .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
                .staged_operator_strategy = true,
                .adaptive_v18_fn = adaptive.engineCallback,
                .adaptive_v18_numerical_policy = .accepted_f32_gram_reuse_v1,
                .adaptive_v18_geometry_route = .flat_gram_only,
                .private_performance_policy = private_policy,
                .geometry_chooser_policy = switch (private_policy) {
                    .owned_capture_summary_incremental_v1, .owned_capture_summary_incremental_reduction_v1 => .incremental_gram_v1,
                    else => .legacy_full_scan,
                },
                .independent_validation_policy = .accelerate_fp64_summary_v1,
                .admission = .cumulative_ladder_4_3_2,
                .retention_divisor = 4,
                .min_retention_divisor = 2,
                .min_global_savings_ratio = 1.000001,
                .chunk_rows = 256,
                .recent_rows = 128,
                .storage_precision = .f16,
                .validation = .whole_head,
                .independent_fp64_publication = true,
                .max_queries_per_consumer = 16,
                .max_transient_host_bytes = 1 << 30,
                .fit = .{ .target_rows = 1, .backend = fitter.backend(), .max_workspace_bytes = 1 << 30 },
            };
            options.fp8_checkpoint = encoding == .fp8;
            options.rotated4_checkpoint = encoding == .rotated4;
            try std.testing.expectError(error.CompactionSavingsInsufficient, engine.prepare(a, state, &.{batch}, layout, options));
            const capture = fitter.trainingCaptureCounters();
            try std.testing.expectEqual(@as(u64, 5), capture.allocations);
            try std.testing.expectEqual(@as(u64, 5), capture.frees);
            try std.testing.expect(stats.fp64_validation_summary_calls > 0);
            try std.testing.expectEqual(stats.fp64_validation_summary_calls * 2, stats.fp64_validation_dgemm_calls);
            try std.testing.expectEqual(engine.Options.IndependentValidationPolicy.accelerate_fp64_summary_v1, stats.fp64_validation_policy);
            const incremental = private_policy == .owned_capture_summary_incremental_v1 or private_policy == .owned_capture_summary_incremental_reduction_v1;
            try std.testing.expectEqual(@as(usize, if (incremental) 3 else 0), stats.incremental_gram_chooser_calls);
            if (private_policy == .owned_capture_summary_incremental_reduction_v1) {
                try std.testing.expect(stats.training_reduction_local_original_calls > 0);
                try std.testing.expect(stats.training_reduction_rounded_harm_calls > 0);
                try std.testing.expect(stats.training_reduction_tiles > 0);
                try std.testing.expectEqual(stats.training_reduction_tiles * 2, stats.training_reduction_backend_calls);
                try std.testing.expect(stats.training_reduction_workspace_peak_bytes > 0);
            } else {
                try std.testing.expectEqual(@as(usize, 0), stats.training_reduction_local_original_calls);
                try std.testing.expectEqual(@as(usize, 0), stats.training_reduction_rounded_harm_calls);
                try std.testing.expectEqual(@as(usize, 0), stats.training_reduction_tiles);
                try std.testing.expectEqual(@as(usize, 0), stats.training_reduction_backend_calls);
                try std.testing.expectEqual(@as(usize, 0), stats.training_reduction_workspace_peak_bytes);
            }
            const unchanged = try checkpoint.encodeWithEncoding(a, state, .{}, .f16, .{});
            defer a.free(unchanged);
            try std.testing.expectEqualSlices(u8, original, unchanged);
            if (private_policy == .owned_capture_summary_incremental_reduction_v1) {
                var capped = options;
                capped.max_transient_host_bytes = 1;
                try std.testing.expectError(error.CompactionWorkspaceLimit, engine.prepare(a, state, &.{batch}, layout, capped));
                const after_cap = try checkpoint.encodeWithEncoding(a, state, .{}, .f16, .{});
                defer a.free(after_cap);
                try std.testing.expectEqualSlices(u8, original, after_cap);

                var cancelled = options;
                cancelled.fit.control.check_fn = Cancel.check;
                try std.testing.expectError(error.Cancelled, engine.prepare(a, state, &.{batch}, layout, cancelled));
                const after_cancel = try checkpoint.encodeWithEncoding(a, state, .{}, .f16, .{});
                defer a.free(after_cancel);
                try std.testing.expectEqualSlices(u8, original, after_cancel);
            }
        }
    }
}

test "attention_match Metal fitter retirement preserves actual FP64 solver calls" {
    if (comptime @import("builtin").os.tag != .macos or !fp64.supported) return error.SkipZigTest;
    var fitter = try Fitter.init(std.testing.allocator, null, 8 * 1024 * 1024);
    var live = true;
    defer if (live) fitter.deinit();
    const backend = fitter.backend();
    var out: [2]f64 = undefined;
    try backend.values_fn.?(backend.context, std.testing.allocator, &.{ 1, 0, 0, 1, 1, 1 }, &.{ 2, 3, 5 }, 3, 2, 1, 0, .{}, &out);
    var vector_out: [4]f64 = undefined;
    try backend.values_fn.?(backend.context, std.testing.allocator, &.{ 1, 0, 0, 1, 1, 1 }, &.{ 2, 4, 3, 6, 5, 10 }, 3, 2, 2, 1e-8, .{}, &vector_out);
    var selected: [1]usize = undefined;
    var selected_weights: [1]f64 = undefined;
    _ = try backend.raw_select_fn.?(backend.context, std.testing.allocator, &.{ 0.9, 0.1, 0.1, 0.9 }, &.{ 0.82, 0.82 }, &.{ 1, 1 }, 2, 2, &.{}, &selected, &selected_weights, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set });
    const phases = fitter.phaseCounters();
    try std.testing.expectEqual(@as(u64, 1), phases.raw_selection_calls);
    try std.testing.expect(phases.scalar_rhs_solves >= 1);
    try std.testing.expectEqual(@as(u64, 1), phases.vector_rhs_solves);
    try std.testing.expectEqual(phases.scalar_rhs_solves + phases.vector_rhs_solves, fitter.accelerated_value_solves);
    try std.testing.expectEqual(@as(usize, 2), phases.max_vector_width);
    const calls = fitter.deinitAndTakeValueSolves();
    live = false;
    try std.testing.expectEqual(phases.scalar_rhs_solves + phases.vector_rhs_solves, calls);
    try std.testing.expectApproxEqAbs(@as(f64, 2), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), out[1], 1e-12);
}

test "metal fitter Gram advisory is private explicit and immutable after work" {
    var fitter = try Fitter.init(std.testing.allocator, null, 8 * 1024 * 1024);
    defer fitter.deinit();
    try std.testing.expectEqual(Fitter.RawSelectionSolver.householder, fitter.rawSelectionSolver());
    try fitter.setRawSelectionSolver(.fp64_gram_advisory);
    try std.testing.expectEqual(Fitter.RawSelectionSolver.fp64_gram_advisory, fitter.rawSelectionSolver());
    fitter.phase_counters.raw_selection_calls = 1;
    try std.testing.expectError(error.InvalidFitConfig, fitter.setRawSelectionSolver(.householder));
}

test "attention_match real Metal capture batches64 and256 are bit identical at HD512 tail257" {
    if (comptime builtin.os.tag != .macos or !@import("build_options").enable_metal) return error.SkipZigTest;
    const a = std.testing.allocator;
    const dim = 512;
    const rows = 8;
    const q = 257;
    const keys = try a.alloc(f64, rows * dim);
    defer a.free(keys);
    const values = try a.alloc(f64, rows * dim);
    defer a.free(values);
    const queries = try a.alloc(f64, q * dim);
    defer a.free(queries);
    for (keys, 0..) |*value, i| value.* = (@as(f64, @floatFromInt(i % 19)) - 9) / 32;
    for (values, 0..) |*value, i| value.* = (@as(f64, @floatFromInt(i % 23)) - 11) / 16;
    for (queries, 0..) |*value, i| value.* = (@as(f64, @floatFromInt(i % 31)) - 15) / 64;
    const head: solver.Head = .{ .rows = rows, .dim = dim, .keys = keys, .values = values, .bias = &.{ 0.25, -0.5, 0.75, -1, 0.125, 0.5, -0.25, 0 }, .scale = 0.04419417382415922, .softcap = 1.5 };
    var tracker = shared.BufferAllocationTracker.init(null, 0);
    const baseline = tracker.snapshot().shared_backing_live_bytes;
    {
        const budget = 64 * 1024 * 1024;
        var fitter64 = try Fitter.init(a, &tracker, budget);
        defer fitter64.deinit();
        try fitter64.setQueryBatchRows(64);
        var counts64: training_source.Counters = .{};
        const cfg64: solver.Config = .{ .target_rows = 2, .max_workspace_bytes = 128 * 1024 * 1024, .backend = fitter64.backend() };
        var source64 = (try training_source.prepare(a, head, queries, cfg64, .{ .external_live_bytes = 0, .consumer_peak_bytes = 1, .probability_storage = .exact_f32_widened, .query_batch_rows = 64, .capture_backend = fitter64.trainingCaptureBackend() }, &counts64)).ready;
        defer source64.deinit();
        var fitter256 = try Fitter.init(a, &tracker, budget);
        defer fitter256.deinit();
        try fitter256.setQueryBatchRows(256);
        var counts256: training_source.Counters = .{};
        const cfg256: solver.Config = .{ .target_rows = 2, .max_workspace_bytes = 128 * 1024 * 1024, .backend = fitter256.backend() };
        var source256 = (try training_source.prepare(a, head, queries, cfg256, .{ .external_live_bytes = 0, .consumer_peak_bytes = 1, .probability_storage = .exact_f32_widened, .query_batch_rows = 256, .capture_backend = fitter256.trainingCaptureBackend() }, &counts256)).ready;
        defer source256.deinit();
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(source64.probabilities_f32), std.mem.sliceAsBytes(source256.probabilities_f32));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(source64.outputs), std.mem.sliceAsBytes(source256.outputs));
        try std.testing.expectEqual(@as(usize, 5), counts64.capture_tiles);
        try std.testing.expectEqual(@as(usize, 2), counts256.capture_tiles);
        try std.testing.expect(fitter64.dispatchCount() > 0);
        try std.testing.expect(fitter256.dispatchCount() > 0);
        for (source256.outputs) |value| try std.testing.expect(std.math.isFinite(value));
        for (source256.probabilities_f32) |value| try std.testing.expect(std.math.isFinite(value) and value >= 0);
    }
    try std.testing.expectEqual(baseline, tracker.snapshot().shared_backing_live_bytes);
}
