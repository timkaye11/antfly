// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Request-owned qualification hooks for the portable checkpoint engine.
//! Slot is void in release binaries. No HTTP input can install a session.
const std = @import("std");
const builtin = @import("builtin");
const checkpoint = @import("runtime/kv/attention_match_checkpoint.zig");

pub const Kind = enum { allocation, nonfinite, gpu_completion };
pub const Observation = struct {
    completed_gpu_batches: u64 = 0,
    poisoned_input: bool = false,
    allocator_failure_observed: bool = false,
    finite_check_observed: bool = false,
    simulated_completion_error: bool = false,
    injected: bool = false,
    live_allocations: usize = 0,
    live_bytes: usize = 0,
    peak_bytes: usize = 0,
    source_before: [32]u8 = @splat(0),
    source_after: [32]u8 = @splat(0),
    revision_before: u64 = 0,
    revision_after: u64 = 0,
    source_preserved: bool = false,
    engine_error: ?[]const u8 = null,
};

pub const Session = struct {
    kind: Kind,
    mutex: std.atomic.Mutex = .unlocked,
    upstream: std.mem.Allocator = std.heap.page_allocator,
    allocator_bound: bool = false,
    observation: Observation = .{},

    fn lock(self: *Session) void {
        while (!self.mutex.tryLock()) std.atomic.spinLoopHint();
    }

    pub fn snapshot(self: *Session) Observation {
        self.lock();
        defer self.mutex.unlock();
        return self.observation;
    }

    pub fn progress(self: *Session, stage: []const u8) void {
        const json = std.json.Stringify.valueAlloc(std.heap.page_allocator, .{
            .schema = "antfly.native-checkpoint-fault-progress.v1",
            .stage = stage,
            .kind = @tagName(self.kind),
            .allocator_bound = self.allocator_bound,
            .observation = self.snapshot(),
        }, .{}) catch return;
        defer std.heap.page_allocator.free(json);
        std.debug.print("native_checkpoint_fault_progress {s}\n", .{json});
    }

    /// Wrap the actual candidate allocator, including parallel worker owners.
    /// No failure is armed until a production Metal batch has completed.
    pub fn bindAllocator(self: *Session, upstream: std.mem.Allocator) !std.mem.Allocator {
        if (!builtin.is_test or self.allocator_bound) return error.InvalidNativeCheckpointFault;
        self.upstream = upstream;
        self.allocator_bound = true;
        self.progress("candidate_allocator_bound");
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }

    fn alloc(raw: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *Session = @ptrCast(@alignCast(raw));
        self.lock();
        defer self.mutex.unlock();
        if (self.kind == .allocation and self.observation.completed_gpu_batches > 0 and !self.observation.injected) {
            // This is the allocation requested by the real engine, through
            // Zig's failing allocator; it is not a throwaway probe allocation.
            var failing = std.testing.FailingAllocator.init(self.upstream, .{ .fail_index = 0 });
            const denied = failing.allocator().rawAlloc(len, alignment, ra);
            std.debug.assert(denied == null and failing.has_induced_failure);
            self.observation.allocator_failure_observed = true;
            self.observation.injected = true;
            return denied;
        }
        const result = self.upstream.rawAlloc(len, alignment, ra) orelse return null;
        self.observation.live_allocations += 1;
        self.observation.live_bytes += len;
        self.observation.peak_bytes = @max(self.observation.peak_bytes, self.observation.live_bytes);
        return result;
    }

    fn resize(raw: *anyopaque, memory: []u8, alignment: std.mem.Alignment, len: usize, ra: usize) bool {
        const self: *Session = @ptrCast(@alignCast(raw));
        self.lock();
        defer self.mutex.unlock();
        if (!self.upstream.rawResize(memory, alignment, len, ra)) return false;
        self.observation.live_bytes = self.observation.live_bytes - memory.len + len;
        self.observation.peak_bytes = @max(self.observation.peak_bytes, self.observation.live_bytes);
        return true;
    }

    fn remap(raw: *anyopaque, memory: []u8, alignment: std.mem.Alignment, len: usize, ra: usize) ?[*]u8 {
        const self: *Session = @ptrCast(@alignCast(raw));
        self.lock();
        defer self.mutex.unlock();
        const result = self.upstream.rawRemap(memory, alignment, len, ra) orelse return null;
        self.observation.live_bytes = self.observation.live_bytes - memory.len + len;
        self.observation.peak_bytes = @max(self.observation.peak_bytes, self.observation.live_bytes);
        return result;
    }

    fn free(raw: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const self: *Session = @ptrCast(@alignCast(raw));
        self.lock();
        defer self.mutex.unlock();
        self.upstream.rawFree(memory, alignment, ra);
        self.observation.live_bytes -= memory.len;
        self.observation.live_allocations -= 1;
    }

    /// Called only after the real dispatch, finite readback and scatter.
    pub fn completedBatch(self: *Session) !void {
        self.lock();
        defer self.mutex.unlock();
        self.observation.completed_gpu_batches += 1;
        if (self.kind == .gpu_completion and !self.observation.injected) {
            self.observation.simulated_completion_error = true;
            self.observation.injected = true;
            return error.GeneratedMetalDispatchFailed;
        }
    }

    pub fn takePoison(self: *Session) bool {
        self.lock();
        defer self.mutex.unlock();
        if (self.kind != .nonfinite or self.observation.completed_gpu_batches == 0 or self.observation.poisoned_input)
            return false;
        self.observation.poisoned_input = true;
        return true;
    }

    /// Annotate the ordinary fitter finite-input rejection, never synthesize it.
    pub fn observedNonfinite(self: *Session) void {
        self.lock();
        defer self.mutex.unlock();
        if (self.kind == .nonfinite and self.observation.poisoned_input) {
            self.observation.finite_check_observed = true;
            self.observation.injected = true;
        }
    }

    pub fn expects(self: *Session, err: anyerror) bool {
        const observed = self.snapshot();
        if (!observed.injected) return false;
        return switch (self.kind) {
            .allocation => observed.allocator_failure_observed and err == error.OutOfMemory,
            .nonfinite => observed.finite_check_observed and err == error.NonFiniteAttention,
            .gpu_completion => observed.simulated_completion_error and err == error.GeneratedMetalDispatchFailed,
        };
    }

    pub fn beginSource(self: *Session, state: checkpoint.State, revision: u64) void {
        self.observation.source_before = sourceDigest(state);
        self.observation.revision_before = revision;
        self.progress("fit_source_bound");
    }

    /// Engine workers have joined and retired their candidate allocations.
    pub fn finishError(self: *Session, state: checkpoint.State, revision: u64, err: anyerror) !void {
        self.observation.source_after = sourceDigest(state);
        self.observation.revision_after = revision;
        self.observation.source_preserved = revision == self.observation.revision_before and
            std.mem.eql(u8, &self.observation.source_before, &self.observation.source_after);
        self.observation.engine_error = @errorName(err);
        const json = try std.json.Stringify.valueAlloc(std.heap.page_allocator, .{
            .schema = "antfly.native-checkpoint-fault.v1",
            .kind = @tagName(self.kind),
            .observation = self.snapshot(),
            .diagnostic_only = true,
            .quality_credit = false,
            .release_authority = false,
        }, .{});
        defer std.heap.page_allocator.free(json);
        std.debug.print("native_checkpoint_fault {s}\n", .{json});
    }
};

pub const Slot = if (builtin.is_test) ?*Session else void;

fn sourceDigest(state: checkpoint.State) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(&state.identity.weights);
    hash.update(&state.identity.tokenizer);
    hash.update(&state.identity.template);
    hash.update(&state.identity.layout);
    hash.update(std.mem.asBytes(&state.logical_length));
    digestSlice(&hash, state.tokens);
    digestSlice(&hash, state.logits);
    digestSlice(&hash, state.layer_donors);
    const donors: u64 = @intCast(state.donors.len);
    hash.update(std.mem.asBytes(&donors));
    for (state.donors) |donor| {
        hash.update(std.mem.asBytes(&donor.layer));
        hash.update(std.mem.asBytes(&donor.window));
        const heads: u64 = @intCast(donor.heads.len);
        hash.update(std.mem.asBytes(&heads));
        for (donor.heads) |head| {
            hash.update(std.mem.asBytes(&head.dim));
            hash.update(std.mem.asBytes(&head.compacted_until));
            digestSlice(&hash, head.positions);
            digestSlice(&hash, head.coverage_ends);
            digestSlice(&hash, head.keys);
            digestSlice(&hash, head.values);
            digestSlice(&hash, head.bias);
        }
    }
    return hash.finalResult();
}

fn digestSlice(hash: *std.crypto.hash.sha2.Sha256, values: anytype) void {
    const count: u64 = @intCast(values.len);
    hash.update(std.mem.asBytes(&count));
    hash.update(std.mem.sliceAsBytes(values));
}

comptime {
    if (!builtin.is_test and Slot != void) @compileError("release exposes native checkpoint faults");
}

test "context native checkpoint allocation fault waits for work and retires actual allocations" {
    var session: Session = .{ .kind = .allocation };
    const allocator = try session.bindAllocator(std.testing.allocator);
    const before = try allocator.alloc(u8, 16);
    try std.testing.expect(!session.snapshot().injected);
    try session.completedBatch();
    try std.testing.expectError(error.OutOfMemory, allocator.alloc(u8, 32));
    allocator.free(before);
    try std.testing.expectEqual(@as(usize, 0), session.snapshot().live_bytes);
    try std.testing.expectEqual(@as(usize, 0), session.snapshot().live_allocations);
    try std.testing.expect(session.expects(error.OutOfMemory));
    try std.testing.expect(!session.expects(error.NonFiniteAttention));
}

test "context native checkpoint nonfinite fault requires poisoned input and actual detector" {
    var session: Session = .{ .kind = .nonfinite };
    try std.testing.expect(!session.takePoison());
    try session.completedBatch();
    try std.testing.expect(session.takePoison());
    try std.testing.expect(!session.takePoison());
    try std.testing.expect(!session.expects(error.NonFiniteAttention));
    session.observedNonfinite();
    try std.testing.expect(session.expects(error.NonFiniteAttention));
    try std.testing.expect(!session.expects(error.OutOfMemory));
}

test "context native checkpoint simulated completion error is one shot after completed work" {
    var session: Session = .{ .kind = .gpu_completion };
    try std.testing.expect(!session.expects(error.GeneratedMetalDispatchFailed));
    try std.testing.expectError(error.GeneratedMetalDispatchFailed, session.completedBatch());
    try std.testing.expect(session.expects(error.GeneratedMetalDispatchFailed));
    try std.testing.expect(!session.expects(error.OutOfMemory));
    try session.completedBatch();
    try std.testing.expectEqual(@as(u64, 2), session.snapshot().completed_gpu_batches);
}

test "context native checkpoint source digest binds numeric source and logical identity" {
    var values = [_]f32{2};
    const heads = [_]checkpoint.Head{.{ .dim = 1, .positions = &.{0}, .coverage_ends = &.{1}, .keys = &.{1}, .values = &values, .bias = &.{0} }};
    const donors = [_]checkpoint.Donor{.{ .layer = 0, .heads = &heads }};
    var state: checkpoint.State = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .logical_length = 1,
        .tokens = &.{7},
        .logits = &.{1},
        .layer_donors = &.{0},
        .donors = &donors,
    };
    const before = sourceDigest(state);
    values[0] = 3;
    try std.testing.expect(!std.mem.eql(u8, &before, &sourceDigest(state)));
    values[0] = 2;
    state.logical_length = 2;
    try std.testing.expect(!std.mem.eql(u8, &before, &sourceDigest(state)));
    state.logical_length = 1;
    try std.testing.expectEqualSlices(u8, &before, &sourceDigest(state));
}
