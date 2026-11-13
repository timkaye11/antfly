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

const std = @import("std");
const build_options = @import("build_options");
const ml = @import("ml");

const cache_mod = @import("cache.zig");
const compiled_backend = @import("compiled_backend.zig");
const metal_capabilities = @import("metal_capabilities.zig");
const metal_executor = if (build_options.enable_metal) @import("metal_executor.zig") else struct {};
const metal_partition_executor = @import("metal_partition_executor.zig");
const model_runtime = @import("model_runtime.zig");
const multi_executor = @import("multi_executor.zig");
const partition_mod = @import("partition.zig");
const checkpoint = @import("../runtime/kv/attention_match_checkpoint.zig");

const Graph = ml.graph.Graph;
const OpCode = ml.graph.OpCode;

fn supportsForMode(
    _: compiled_backend.AttachContext,
    _: compiled_backend.CompileMode,
) *const fn (op: OpCode) bool {
    return &metal_capabilities.supportsMetalEagerGraph;
}

fn decideForMode(
    _: compiled_backend.AttachContext,
    _: compiled_backend.CompileMode,
) ?*const fn (query: partition_mod.CapabilityQuery) partition_mod.CapabilityDecision {
    return &metal_capabilities.decideMetalEagerGraph;
}

fn shouldAttach(
    context: compiled_backend.AttachContext,
    _: compiled_backend.CompileMode,
) bool {
    if (comptime !build_options.enable_metal) return false;
    if (context.requested_backend != .metal) return false;
    return context.cb.kind() == .metal;
}

fn partitionHasCompute(graph: *const Graph, part: partition_mod.Partition) bool {
    for (part.node_ids) |node_id| {
        const op = graph.node(node_id).op;
        if (op != .parameter and op != .constant) return true;
    }
    return false;
}

fn isMetalPartitionEligible(graph: *const Graph, part: partition_mod.Partition) bool {
    return part.backend == .metal and partitionHasCompute(graph, part);
}

fn hasCompilablePartition(
    graph: *const Graph,
    dpp: *const multi_executor.DevicePartitionPlan,
    context: compiled_backend.AttachContext,
    _: compiled_backend.CompileMode,
) bool {
    if (!compiled_backend.attachmentTargetAllowsPlan(graph, dpp, .metal, context.attachment_target)) return false;
    for (dpp.base.partitions) |part| {
        if (isMetalPartitionEligible(graph, part)) return true;
    }
    return false;
}

fn attachExecutors(
    allocator: std.mem.Allocator,
    entry: *cache_mod.CacheEntry,
    graph: *const Graph,
    dpp: *multi_executor.DevicePartitionPlan,
    context: compiled_backend.AttachContext,
    _: compiled_backend.CompileMode,
) !void {
    if (comptime !build_options.enable_metal) return;

    entry.selectCompiledPartitionsBackend(.metal, context.attachment_target);

    if (entry.compiled_partitions_status == .uninitialized) {
        if (!compiled_backend.attachmentTargetAllowsPlan(graph, dpp, .metal, context.attachment_target)) {
            entry.compiled_partitions = null;
            entry.compiled_partitions_status = .unavailable;
            return;
        }

        var compiled = std.ArrayListUnmanaged(cache_mod.CompiledPartition).empty;
        errdefer {
            for (compiled.items) |*cp| cp.executor.deinitExecutor();
            compiled.deinit(allocator);
        }

        for (dpp.base.partitions, 0..) |part, part_idx| {
            if (!isMetalPartitionEligible(graph, part)) continue;

            const exec = try metal_partition_executor.MetalPartitionExecutor.create(
                allocator,
                graph,
                context.cb,
            );
            var transferred = false;
            errdefer if (!transferred) exec.partitionExecutor().deinitExecutor();

            try compiled.append(allocator, .{
                .partition_idx = @intCast(part_idx),
                .executor = exec.partitionExecutor().*,
            });
            transferred = true;
        }

        if (compiled.items.len > 0) {
            entry.compiled_partitions = try compiled.toOwnedSlice(allocator);
            entry.compiled_partitions_status = .ready;
        } else {
            entry.compiled_partitions = null;
            entry.compiled_partitions_status = .unavailable;
        }
    }

    if (entry.compiled_partitions_status == .ready) {
        if (entry.compiled_partitions) |compiled| {
            for (compiled) |*cp| {
                dpp.base.partitions[cp.partition_idx].executor = &cp.executor;
            }
            dpp.base.owns_executors = false;
        }
    }
}

const CachedModelExecutor = struct {
    allocator: std.mem.Allocator,
    executor: model_runtime.ModelExecutor,
    model_identity: ?checkpoint.Identity,
};

fn sameOptionalModelIdentity(a: ?checkpoint.Identity, b: ?checkpoint.Identity) bool {
    if (a == null or b == null) return a == null and b == null;
    return a.?.eql(b.?);
}

test "compiled Metal executor cache identity rejects none upgrade and mismatch" {
    const first = checkpoint.Identity{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    var changed = first;
    changed.weights[0] ^= 1;
    try std.testing.expect(sameOptionalModelIdentity(null, null));
    try std.testing.expect(!sameOptionalModelIdentity(null, first));
    try std.testing.expect(!sameOptionalModelIdentity(first, null));
    try std.testing.expect(sameOptionalModelIdentity(first, first));
    try std.testing.expect(!sameOptionalModelIdentity(first, changed));
}

test "fresh prefill retires session runtime while continuation retains it" {
    const Probe = struct {
        deinits: *usize,

        fn prefill(_: *anyopaque, _: std.mem.Allocator, _: model_runtime.PrefillRequest) !model_runtime.ModelOutput {
            return error.UnexpectedPrefill;
        }

        fn deinit(ctx: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.deinits.* += 1;
        }

        const vtable = model_runtime.ModelRuntime.VTable{
            .prefill = prefill,
            .deinit = deinit,
        };
    };

    var cache = cache_mod.GraphCache.init(std.testing.allocator);
    defer cache.deinit();
    var deinits: usize = 0;
    var first = Probe{ .deinits = &deinits };
    cache.putSessionCompiledModelRuntime(.metal, .whole_model, .{ .ptr = &first, .vtable = &Probe.vtable });

    const continuation = model_runtime.PrefillRequest{
        .input_ids = &.{3},
        .seq_len = 3,
        .query_seq_len = 1,
        .attention_mode = .paged_prefill,
    };
    try std.testing.expect(!try retireSessionRuntimeBeforeFreshPrefill(&cache, null, continuation));
    try std.testing.expectEqual(@as(usize, 0), deinits);
    try std.testing.expect(cache.getSessionCompiledModelRuntime(.metal, .whole_model) != null);

    const fresh = model_runtime.PrefillRequest{
        .input_ids = &.{ 4, 5 },
        .seq_len = 2,
        .query_seq_len = 2,
        .attention_mode = .paged_prefill,
    };
    try std.testing.expect(try retireSessionRuntimeBeforeFreshPrefill(&cache, null, fresh));
    try std.testing.expectEqual(@as(usize, 1), deinits);
    try std.testing.expect(cache.getSessionCompiledModelRuntime(.metal, .whole_model) == null);

    // Re-creation installs a distinct request runtime in the same session
    // cache; the retained compiled executor is owned by the graph entry and is
    // intentionally outside this retirement operation.
    var second = Probe{ .deinits = &deinits };
    cache.putSessionCompiledModelRuntime(.metal, .whole_model, .{ .ptr = &second, .vtable = &Probe.vtable });
    try std.testing.expect(cache.getSessionCompiledModelRuntime(.metal, .whole_model).?.ptr == @as(*anyopaque, @ptrCast(&second)));
}

test "packed-style early runtime retirement detaches once and accepts a fresh owner" {
    const Probe = struct {
        binding: ?model_runtime.ContextOwnerBinding = null,
        deinits: *usize,

        fn prefill(_: *anyopaque, _: std.mem.Allocator, _: model_runtime.PrefillRequest) !model_runtime.ModelOutput {
            return error.UnexpectedPrefill;
        }
        fn bind(ctx: *anyopaque, binding: model_runtime.ContextOwnerBinding) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            try binding.validate();
            if (self.binding) |installed| {
                if (installed.owner.tracker != binding.owner.tracker) return error.ContextFrameOwnershipMismatch;
                return;
            }
            try binding.owner.bindCompiledStorage(binding.witness, binding.owner.tracker);
            self.binding = binding;
        }
        fn deinit(ctx: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            if (self.binding) |binding| {
                binding.owner.detachCompiledStorage(binding.witness, binding.owner.tracker) catch
                    @panic("test runtime failed to detach owner");
                self.binding = null;
            }
            self.deinits.* += 1;
        }
        const vtable = model_runtime.ModelRuntime.VTable{
            .prefill = prefill,
            .deinit = deinit,
            .bind_context_owner = bind,
        };
    };

    var cache = cache_mod.GraphCache.init(std.testing.allocator);
    defer cache.deinit();
    var deinits: usize = 0;

    var first_owner = try @import("../runtime/kv/host_allocation_tracker.zig").RequestAllocationOwner.init(null);
    const first_witness = try first_owner.beginContextScope(31);
    try first_owner.requireCompiledStorage(first_witness);
    var first = Probe{ .deinits = &deinits };
    cache.putSessionCompiledModelRuntime(.metal, .whole_model, .{ .ptr = &first, .vtable = &Probe.vtable });
    try cache.getSessionCompiledModelRuntime(.metal, .whole_model).?.bindContextOwner(.{ .owner = &first_owner, .witness = first_witness });
    try first_owner.invalidateContextScope(first_witness);

    // The packed bridge retires early; the outer request cleanup repeats the
    // reset. Only the first reset owns teardown/detach.
    cache.resetSessionCompiledModelRuntime();
    cache.resetSessionCompiledModelRuntime();
    try std.testing.expectEqual(@as(usize, 1), deinits);
    try std.testing.expectError(error.ContextFrameOwnershipMismatch, first_owner.detachCompiledStorage(first_witness, first_owner.tracker));
    const first_receipt = first_owner.release();
    try std.testing.expect(!first_receipt.context_allocation_scope_complete);

    var second_owner = try @import("../runtime/kv/host_allocation_tracker.zig").RequestAllocationOwner.init(null);
    const second_witness = try second_owner.beginContextScope(31);
    try second_owner.requireCompiledStorage(second_witness);
    var second = Probe{ .deinits = &deinits };
    cache.putSessionCompiledModelRuntime(.metal, .whole_model, .{ .ptr = &second, .vtable = &Probe.vtable });
    try cache.getSessionCompiledModelRuntime(.metal, .whole_model).?.bindContextOwner(.{ .owner = &second_owner, .witness = second_witness });
    cache.resetSessionCompiledModelRuntime();
    try std.testing.expectEqual(@as(usize, 2), deinits);
    try std.testing.expectError(error.ContextFrameOwnershipMismatch, second_owner.detachCompiledStorage(second_witness, second_owner.tracker));
    _ = second_owner.release();
}

test "cancelled fresh prefill does not retire active session runtime" {
    const Probe = struct {
        deinits: *usize,
        fn prefill(_: *anyopaque, _: std.mem.Allocator, _: model_runtime.PrefillRequest) !model_runtime.ModelOutput {
            return error.UnexpectedPrefill;
        }
        fn deinit(ctx: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.deinits.* += 1;
        }
        const vtable = model_runtime.ModelRuntime.VTable{ .prefill = prefill, .deinit = deinit };
    };
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };

    var cache = cache_mod.GraphCache.init(std.testing.allocator);
    defer cache.deinit();
    var deinits: usize = 0;
    var probe = Probe{ .deinits = &deinits };
    cache.putSessionCompiledModelRuntime(.metal, .whole_model, .{ .ptr = &probe, .vtable = &Probe.vtable });
    const fresh = model_runtime.PrefillRequest{
        .execution_control = .{ .check_fn = Cancel.check },
        .input_ids = &.{1},
        .seq_len = 1,
        .query_seq_len = 1,
        .attention_mode = .paged_prefill,
    };
    try std.testing.expectError(error.Cancelled, retireSessionRuntimeBeforeFreshPrefill(&cache, null, fresh));
    try std.testing.expectEqual(@as(usize, 0), deinits);
    try std.testing.expect(cache.getSessionCompiledModelRuntime(.metal, .whole_model) != null);
}

test "fresh prefill identity mismatch never retires cached runtime" {
    const Probe = struct {
        deinits: *usize,
        fn prefill(_: *anyopaque, _: std.mem.Allocator, _: model_runtime.PrefillRequest) !model_runtime.ModelOutput {
            return error.UnexpectedPrefill;
        }
        fn deinit(ctx: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.deinits.* += 1;
        }
        const vtable = model_runtime.ModelRuntime.VTable{ .prefill = prefill, .deinit = deinit };
    };

    const a = checkpoint.Identity{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    var b = a;
    b.layout[0] ^= 1;
    const fresh = model_runtime.PrefillRequest{
        .input_ids = &.{1},
        .seq_len = 1,
        .query_seq_len = 1,
        .attention_mode = .paged_prefill,
    };

    var deinits: usize = 0;
    {
        var cache = cache_mod.GraphCache.init(std.testing.allocator);
        defer cache.deinit();
        var probe = Probe{ .deinits = &deinits };
        cache.putSessionCompiledModelRuntime(.metal, .whole_model, .{ .ptr = &probe, .vtable = &Probe.vtable });
        try std.testing.expectError(error.CompiledModelIdentityMismatch, retireSessionRuntimeBeforeFreshPrefill(&cache, a, fresh));
        try std.testing.expectEqual(@as(usize, 0), deinits);
        try std.testing.expect(cache.getSessionCompiledModelRuntime(.metal, .whole_model) != null);
    }
    try std.testing.expectEqual(@as(usize, 1), deinits);

    {
        var cache = cache_mod.GraphCache.init(std.testing.allocator);
        defer cache.deinit();
        var probe = Probe{ .deinits = &deinits };
        cache.putSessionCompiledModelRuntime(.metal, .whole_model, .{ .ptr = &probe, .vtable = &Probe.vtable, .model_identity = a });
        try std.testing.expectError(error.CompiledModelIdentityMismatch, retireSessionRuntimeBeforeFreshPrefill(&cache, null, fresh));
        try std.testing.expectEqual(@as(usize, 1), deinits);
        try std.testing.expect(cache.getSessionCompiledModelRuntime(.metal, .whole_model) != null);
    }
    try std.testing.expectEqual(@as(usize, 2), deinits);

    {
        var cache = cache_mod.GraphCache.init(std.testing.allocator);
        defer cache.deinit();
        var probe = Probe{ .deinits = &deinits };
        cache.putSessionCompiledModelRuntime(.metal, .whole_model, .{ .ptr = &probe, .vtable = &Probe.vtable, .model_identity = a });
        try std.testing.expectError(error.CompiledModelIdentityMismatch, retireSessionRuntimeBeforeFreshPrefill(&cache, b, fresh));
        try std.testing.expectEqual(@as(usize, 2), deinits);
        try std.testing.expect(cache.getSessionCompiledModelRuntime(.metal, .whole_model) != null);
    }
    try std.testing.expectEqual(@as(usize, 3), deinits);
}

fn deinitModelExecutorPtr(ctx: *anyopaque) void {
    const cached: *CachedModelExecutor = @ptrCast(@alignCast(ctx));
    cached.executor.deinit();
    cached.allocator.destroy(cached);
}

fn ensureModelExecutor(
    allocator: std.mem.Allocator,
    entry: *cache_mod.CacheEntry,
    context: compiled_backend.AttachContext,
) !?*model_runtime.ModelExecutor {
    if (comptime !build_options.enable_metal) return null;
    if (context.attachment_target != .whole_model) return null;

    entry.selectCompiledModelExecutorBackend(.metal, context.attachment_target);
    if (entry.compiled_model_status == .ready) {
        const cached = entry.compiled_model_executor orelse return null;
        const executor: *CachedModelExecutor = @ptrCast(@alignCast(cached.ptr));
        if (!sameOptionalModelIdentity(executor.model_identity, context.model_identity))
            return error.CompiledModelIdentityMismatch;
        return &executor.executor;
    }
    if (entry.compiled_model_status == .unavailable) return null;

    const session = context.session orelse {
        entry.compiled_model_status = .unavailable;
        return null;
    };
    const gpt_config = context.gpt_config orelse {
        entry.compiled_model_status = .unavailable;
        return null;
    };
    if (!metal_executor.supportsSession(session)) {
        entry.compiled_model_status = .unavailable;
        return null;
    }

    const cached = try allocator.create(CachedModelExecutor);
    errdefer allocator.destroy(cached);
    cached.* = .{
        .allocator = allocator,
        .model_identity = context.model_identity,
        .executor = try metal_executor.createModelExecutorBound(
            allocator,
            session,
            gpt_config,
            context.kv_dtype,
            context.shared_moe_cache,
            context.model_identity,
        ),
    };
    entry.compiled_model_executor = .{
        .ptr = cached,
        .deinit = &deinitModelExecutorPtr,
    };
    entry.compiled_model_status = .ready;
    return &cached.executor;
}

fn resetRuntimeAtPrefillStart(allocator: std.mem.Allocator, runtime: *model_runtime.ModelRuntime, request: model_runtime.PrefillRequest) !void {
    // A fresh prompt starts at position zero. Later chunks must retain the
    // runtime-owned KV written by earlier chunks.
    if (request.seq_len == request.query_seq_len) {
        try request.check();
        try runtime.reset();
        try request.check();
        if (request.prepare_after_fresh_reset) |prepare_request| {
            const prepared = runtime.prepare(allocator, prepare_request) catch |err| {
                runtime.reset() catch {};
                return err;
            };
            if (!prepared) {
                try runtime.reset();
                return error.RequiredCompiledRuntimePrepareDeclined;
            }
            request.check() catch |err| {
                runtime.reset() catch {};
                return err;
            };
        }
    }
}

fn retireSessionRuntimeBeforeFreshPrefill(
    cache: *cache_mod.GraphCache,
    expected_identity: ?checkpoint.Identity,
    request: model_runtime.PrefillRequest,
) !bool {
    // A position-zero prefill starts a new request and must not inherit the
    // prior request's runtime-owned KV slot directory. Keep the compiled model
    // executor (and its weight bindings) cached, but retire the session runtime
    // before any caller obtains a pointer to it. Continuation chunks have
    // seq_len > query_seq_len and deliberately retain the current runtime.
    if (request.seq_len != request.query_seq_len) return false;
    // Validate authority before any destructive lifecycle transition. A fresh
    // request with a None/Some or A/B identity mismatch must not evict the
    // valid runtime installed by the loaded model.
    if (cache.getSessionCompiledModelRuntime(.metal, .whole_model)) |runtime| {
        if (!model_runtime.sameOptionalModelIdentity(runtime.model_identity, expected_identity))
            return error.CompiledModelIdentityMismatch;
    }
    try request.check();
    cache.resetSessionCompiledModelRuntime();
    return true;
}

fn executeModelForward(
    allocator: std.mem.Allocator,
    cache: *cache_mod.GraphCache,
    entry: *cache_mod.CacheEntry,
    graph: *const Graph,
    dpp: *multi_executor.DevicePartitionPlan,
    context: compiled_backend.AttachContext,
    _: compiled_backend.CompileMode,
    request: model_runtime.ForwardRequest,
) !?[]f32 {
    if (comptime !build_options.enable_metal) return null;
    if (context.attachment_target != .whole_model) return null;
    if (!compiled_backend.attachmentTargetAllowsPlan(graph, dpp, .metal, .whole_model)) return null;

    const executor = try ensureModelExecutor(allocator, entry, context) orelse return null;
    switch (request) {
        .prefill => |prefill_request| _ = try retireSessionRuntimeBeforeFreshPrefill(cache, context.model_identity, prefill_request),
        .decode => {},
    }
    const runtime = try compiled_backend.modelRuntimeForExecutor(
        allocator,
        cache,
        entry,
        .metal,
        context.attachment_target,
        executor,
    );
    if (context.context_owner_binding) |binding| try runtime.bindContextOwner(binding);

    var output = switch (request) {
        .prefill => |prefill_request| blk: {
            try resetRuntimeAtPrefillStart(allocator, runtime, prefill_request);
            break :blk runtime.prefill(allocator, prefill_request) catch |err| switch (err) {
                error.UnsupportedShape,
                error.UnsupportedDecode,
                error.MissingPastKeyValue,
                error.MissingValue,
                => return null,
                else => return err,
            };
        },
        .decode => |decode_request| runtime.decode(allocator, decode_request) catch |err| switch (err) {
            error.UnsupportedShape,
            error.UnsupportedDecode,
            error.MissingPastKeyValue,
            error.MissingValue,
            => return null,
            else => return err,
        },
    };
    return try output.takeHostLogits(allocator);
}

fn executeRuntimeForward(
    allocator: std.mem.Allocator,
    runtime: *model_runtime.ModelRuntime,
    request: model_runtime.ForwardRequest,
) !?[]f32 {
    var output = (try executeRuntimeOutput(allocator, runtime, request)) orelse return null;
    return try output.takeHostLogits(allocator);
}

fn executeRuntimeOutput(
    allocator: std.mem.Allocator,
    runtime: *model_runtime.ModelRuntime,
    request: model_runtime.ForwardRequest,
) !?model_runtime.ModelOutput {
    const output = switch (request) {
        .prefill => |prefill_request| blk: {
            try resetRuntimeAtPrefillStart(allocator, runtime, prefill_request);
            break :blk runtime.prefill(allocator, prefill_request) catch |err| switch (err) {
                error.UnsupportedShape,
                error.UnsupportedDecode,
                error.MissingPastKeyValue,
                error.MissingValue,
                => return null,
                else => return err,
            };
        },
        .decode => |decode_request| runtime.decode(allocator, decode_request) catch |err| switch (err) {
            error.UnsupportedShape,
            error.UnsupportedDecode,
            error.MissingPastKeyValue,
            error.MissingValue,
            => return null,
            else => return err,
        },
    };
    return output;
}

fn executeRuntimeGreedy(
    allocator: std.mem.Allocator,
    runtime: *model_runtime.ModelRuntime,
    request: model_runtime.ForwardRequest,
    vocab_size: usize,
) !?i64 {
    return switch (request) {
        .prefill => |prefill_request| blk: {
            var output = (try executeRuntimeOutput(allocator, runtime, request)) orelse return null;
            defer output.deinit(allocator);
            break :blk try output.greedyTokenSuppress(
                allocator,
                vocab_size,
                prefill_request.suppress_token_ids,
            );
        },
        .decode => |decode_request| blk: {
            const greedy = runtime.decodeGreedy(allocator, decode_request) catch |err| switch (err) {
                error.UnsupportedShape,
                error.UnsupportedDecode,
                error.UnsupportedGreedyDecode,
                error.MissingPastKeyValue,
                error.MissingValue,
                => return null,
                else => return err,
            };
            break :blk greedy.token_id;
        },
    };
}

fn executeRuntimeSample(
    allocator: std.mem.Allocator,
    runtime: *model_runtime.ModelRuntime,
    request: model_runtime.SampledDecodeRequest,
) !?i64 {
    // This is the commit boundary. Returning null is allowed only before the
    // runtime submits a decode frame. Once decodeSample starts, every error is
    // propagated and the caller must not replay the token through another
    // forward path.
    if (!runtime.supportsSampleDecodeRequest(request)) return null;
    const sampled = try runtime.decodeSample(allocator, request);
    return sampled.token_id;
}

fn directSessionRuntime(
    allocator: std.mem.Allocator,
    cache: *cache_mod.GraphCache,
    context: compiled_backend.AttachContext,
    _: compiled_backend.CompileMode,
) !?*model_runtime.ModelRuntime {
    if (comptime !build_options.enable_metal) return null;
    if (context.attachment_target != .whole_model) return null;
    if (!shouldAttach(context, .single_device)) return null;

    if (cache.getSessionCompiledModelRuntime(.metal, context.attachment_target)) |runtime| {
        if (!model_runtime.sameOptionalModelIdentity(runtime.model_identity, context.model_identity))
            return error.CompiledModelIdentityMismatch;
        if (context.context_owner_binding) |binding| try runtime.bindContextOwner(binding);
        return runtime;
    }

    const session = context.session orelse return null;
    const gpt_config = context.gpt_config orelse return null;
    if (!metal_executor.supportsSession(session)) return null;

    var executor = try metal_executor.createModelExecutorBound(
        allocator,
        session,
        gpt_config,
        context.kv_dtype,
        context.shared_moe_cache,
        context.model_identity,
    );
    defer executor.deinit();

    const runtime = try compiled_backend.modelRuntimeForSessionExecutor(
        allocator,
        cache,
        .metal,
        context.attachment_target,
        &executor,
    );
    if (context.context_owner_binding) |binding| try runtime.bindContextOwner(binding);
    return runtime;
}

fn executeModelForwardDirect(
    allocator: std.mem.Allocator,
    cache: *cache_mod.GraphCache,
    context: compiled_backend.AttachContext,
    mode: compiled_backend.CompileMode,
    request: model_runtime.ForwardRequest,
) !?[]f32 {
    switch (request) {
        .prefill => |prefill_request| _ = try retireSessionRuntimeBeforeFreshPrefill(cache, context.model_identity, prefill_request),
        .decode => {},
    }
    const runtime = try directSessionRuntime(allocator, cache, context, mode) orelse return null;
    return executeRuntimeForward(allocator, runtime, request);
}

fn executeModelForwardOutputDirect(
    allocator: std.mem.Allocator,
    cache: *cache_mod.GraphCache,
    context: compiled_backend.AttachContext,
    mode: compiled_backend.CompileMode,
    request: model_runtime.ForwardRequest,
) !?model_runtime.ModelOutput {
    switch (request) {
        .prefill => |prefill_request| _ = try retireSessionRuntimeBeforeFreshPrefill(cache, context.model_identity, prefill_request),
        .decode => {},
    }
    const runtime = try directSessionRuntime(allocator, cache, context, mode) orelse return null;
    return executeRuntimeOutput(allocator, runtime, request);
}

fn executeModelGreedyDirect(
    allocator: std.mem.Allocator,
    cache: *cache_mod.GraphCache,
    context: compiled_backend.AttachContext,
    mode: compiled_backend.CompileMode,
    request: model_runtime.ForwardRequest,
    vocab_size: usize,
) !?i64 {
    switch (request) {
        .prefill => |prefill_request| _ = try retireSessionRuntimeBeforeFreshPrefill(cache, context.model_identity, prefill_request),
        .decode => {},
    }
    const runtime = try directSessionRuntime(allocator, cache, context, mode) orelse return null;
    return executeRuntimeGreedy(allocator, runtime, request, vocab_size);
}

fn executeModelSampleDirect(
    allocator: std.mem.Allocator,
    cache: *cache_mod.GraphCache,
    context: compiled_backend.AttachContext,
    mode: compiled_backend.CompileMode,
    request: model_runtime.SampledDecodeRequest,
) !?i64 {
    const runtime = try directSessionRuntime(allocator, cache, context, mode) orelse return null;
    return executeRuntimeSample(allocator, runtime, request);
}

fn prepareModelRuntimeDirect(
    allocator: std.mem.Allocator,
    cache: *cache_mod.GraphCache,
    context: compiled_backend.AttachContext,
    mode: compiled_backend.CompileMode,
    request: model_runtime.PrepareRequest,
) !bool {
    const runtime = try directSessionRuntime(allocator, cache, context, mode) orelse return false;
    return runtime.prepare(allocator, request);
}

pub const backend = compiled_backend.Definition{
    .kind = .metal,
    .priority = 10,
    .model_runtime_strategy = .direct_session,
    .supports_for_mode = &supportsForMode,
    .decide_for_mode = &decideForMode,
    .should_attach = &shouldAttach,
    .has_compilable_partition = &hasCompilablePartition,
    .attach_executors = &attachExecutors,
    .execute_model_forward = &executeModelForward,
    .execute_model_forward_direct = &executeModelForwardDirect,
    .execute_model_forward_output_direct = &executeModelForwardOutputDirect,
    .execute_model_greedy_direct = &executeModelGreedyDirect,
    .execute_model_sample_direct = &executeModelSampleDirect,
    .prepare_model_runtime_direct = &prepareModelRuntimeDirect,
};

test "sampled runtime declines before submission when capability rejects request" {
    const Probe = struct {
        decode_calls: usize = 0,

        fn supports(_: *anyopaque, _: model_runtime.SampledDecodeRequest) bool {
            return false;
        }

        fn decode(ctx: *anyopaque, _: std.mem.Allocator, _: model_runtime.SampledDecodeRequest) !model_runtime.SampledDecodeOutput {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.decode_calls += 1;
            return .{ .token_id = 7 };
        }

        fn prefill(_: *anyopaque, _: std.mem.Allocator, _: model_runtime.PrefillRequest) !model_runtime.ModelOutput {
            return error.UnexpectedPrefill;
        }

        fn deinit(_: *anyopaque) void {}

        const vtable = model_runtime.ModelRuntime.VTable{
            .prefill = prefill,
            .decode_sample = decode,
            .supports_sample_decode_request = supports,
            .deinit = deinit,
        };
    };

    var probe = Probe{};
    var runtime = model_runtime.ModelRuntime{ .ptr = &probe, .vtable = &Probe.vtable };
    const result = try executeRuntimeSample(std.testing.allocator, &runtime, .{
        .decode = .{ .token_id = 1, .position = 1 },
        .sampling = .{ .temperature = 0.8, .top_p = 0.95 },
        .token_history = &.{1},
    });
    try std.testing.expectEqual(@as(?i64, null), result);
    try std.testing.expectEqual(@as(usize, 0), probe.decode_calls);
}

test "sampled runtime propagates post-commit error without replay" {
    const Probe = struct {
        decode_calls: usize = 0,

        fn supports(_: *anyopaque, _: model_runtime.SampledDecodeRequest) bool {
            return true;
        }

        fn decode(ctx: *anyopaque, _: std.mem.Allocator, _: model_runtime.SampledDecodeRequest) !model_runtime.SampledDecodeOutput {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.decode_calls += 1;
            return error.InjectedPostCommitFailure;
        }

        fn prefill(_: *anyopaque, _: std.mem.Allocator, _: model_runtime.PrefillRequest) !model_runtime.ModelOutput {
            return error.UnexpectedPrefill;
        }

        fn deinit(_: *anyopaque) void {}

        const vtable = model_runtime.ModelRuntime.VTable{
            .prefill = prefill,
            .decode_sample = decode,
            .supports_sample_decode_request = supports,
            .deinit = deinit,
        };
    };

    var probe = Probe{};
    var runtime = model_runtime.ModelRuntime{ .ptr = &probe, .vtable = &Probe.vtable };
    try std.testing.expectError(error.InjectedPostCommitFailure, executeRuntimeSample(std.testing.allocator, &runtime, .{
        .decode = .{ .token_id = 1, .position = 1 },
        .sampling = .{ .temperature = 0.8, .top_p = 0.95 },
        .token_history = &.{1},
    }));
    try std.testing.expectEqual(@as(usize, 1), probe.decode_calls);
}

test "fresh prefill reset installs only current prepare and later chunks retain it" {
    const Probe = struct {
        const Event = enum { reset, prepare, prefill };
        events: [8]Event = undefined,
        event_count: usize = 0,
        active_admission: usize = 99,
        prepare_result: enum { success, decline, fail } = .success,

        fn note(self: *@This(), event: Event) void {
            self.events[self.event_count] = event;
            self.event_count += 1;
        }
        fn reset(ctx: *anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.note(.reset);
            self.active_admission = 0;
        }
        fn prepare(ctx: *anyopaque, _: std.mem.Allocator, request: model_runtime.PrepareRequest) !bool {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.note(.prepare);
            switch (self.prepare_result) {
                .success => self.active_admission = request.kv_tokens_hint,
                .decline => return false,
                .fail => return error.InjectedPrepareFailure,
            }
            return true;
        }
        fn prefill(ctx: *anyopaque, allocator: std.mem.Allocator, _: model_runtime.PrefillRequest) !model_runtime.ModelOutput {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.note(.prefill);
            if (self.active_admission != 17) return error.StaleAdmission;
            const logits = try allocator.alloc(f32, 1);
            logits[0] = 1;
            return .{ .logits = logits };
        }
        fn deinit(_: *anyopaque) void {}
        const vtable = model_runtime.ModelRuntime.VTable{
            .prepare = prepare,
            .prefill = prefill,
            .reset = reset,
            .deinit = deinit,
        };
    };

    var probe = Probe{};
    var runtime = model_runtime.ModelRuntime{ .ptr = &probe, .vtable = &Probe.vtable };
    const first = model_runtime.PrefillRequest{
        .input_ids = &.{ 1, 2 },
        .seq_len = 2,
        .query_seq_len = 2,
        .attention_mode = .paged_prefill,
        .prepare_after_fresh_reset = .{ .kv_tokens_hint = 17 },
    };
    try resetRuntimeAtPrefillStart(std.testing.allocator, &runtime, first);
    var first_output = try runtime.prefill(std.testing.allocator, first);
    first_output.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(Probe.Event, &.{ .reset, .prepare, .prefill }, probe.events[0..probe.event_count]);
    try std.testing.expectEqual(@as(usize, 17), probe.active_admission);

    probe.event_count = 0;
    const later = model_runtime.PrefillRequest{
        .input_ids = &.{3},
        .seq_len = 3,
        .query_seq_len = 1,
        .attention_mode = .paged_prefill,
        .prepare_after_fresh_reset = .{ .kv_tokens_hint = 88 },
    };
    try resetRuntimeAtPrefillStart(std.testing.allocator, &runtime, later);
    var later_output = try runtime.prefill(std.testing.allocator, later);
    later_output.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(Probe.Event, &.{.prefill}, probe.events[0..probe.event_count]);
    try std.testing.expectEqual(@as(usize, 17), probe.active_admission);
}

test "required prepare decline and error abort before prefill while ordinary path stays lazy" {
    const Probe = struct {
        resets: usize = 0,
        prepares: usize = 0,
        prefills: usize = 0,
        mode: enum { success, decline, fail } = .success,
        fn reset(ctx: *anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.resets += 1;
        }
        fn prepare(ctx: *anyopaque, _: std.mem.Allocator, _: model_runtime.PrepareRequest) !bool {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.prepares += 1;
            return switch (self.mode) {
                .success => true,
                .decline => false,
                .fail => error.InjectedPrepareFailure,
            };
        }
        fn prefill(ctx: *anyopaque, allocator: std.mem.Allocator, _: model_runtime.PrefillRequest) !model_runtime.ModelOutput {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.prefills += 1;
            const logits = try allocator.alloc(f32, 1);
            return .{ .logits = logits };
        }
        fn deinit(_: *anyopaque) void {}
        const vtable = model_runtime.ModelRuntime.VTable{ .prepare = prepare, .prefill = prefill, .reset = reset, .deinit = deinit };
    };
    const required = model_runtime.PrefillRequest{
        .input_ids = &.{1},
        .seq_len = 1,
        .query_seq_len = 1,
        .attention_mode = .paged_prefill,
        .prepare_after_fresh_reset = .{ .kv_tokens_hint = 17 },
    };
    var probe = Probe{ .mode = .decline };
    var runtime = model_runtime.ModelRuntime{ .ptr = &probe, .vtable = &Probe.vtable };
    try std.testing.expectError(error.RequiredCompiledRuntimePrepareDeclined, resetRuntimeAtPrefillStart(std.testing.allocator, &runtime, required));
    try std.testing.expectEqual(@as(usize, 0), probe.prefills);
    probe.mode = .fail;
    try std.testing.expectError(error.InjectedPrepareFailure, resetRuntimeAtPrefillStart(std.testing.allocator, &runtime, required));
    try std.testing.expectEqual(@as(usize, 0), probe.prefills);

    const CancelAfterPrepare = struct {
        checks: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.checks += 1;
            if (self.checks == 3) return error.Cancelled;
        }
    };
    var cancel = CancelAfterPrepare{};
    var cancelled = required;
    cancelled.execution_control = .{ .ptr = &cancel, .check_fn = CancelAfterPrepare.check };
    probe.mode = .success;
    const resets_before_cancel = probe.resets;
    try std.testing.expectError(error.Cancelled, resetRuntimeAtPrefillStart(std.testing.allocator, &runtime, cancelled));
    try std.testing.expectEqual(resets_before_cancel + 2, probe.resets);
    try std.testing.expectEqual(@as(usize, 0), probe.prefills);

    const ordinary = model_runtime.PrefillRequest{ .input_ids = &.{1}, .seq_len = 1, .query_seq_len = 1, .attention_mode = .paged_prefill };
    try resetRuntimeAtPrefillStart(std.testing.allocator, &runtime, ordinary);
    var output = try runtime.prefill(std.testing.allocator, ordinary);
    output.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 1), probe.prefills);
}
