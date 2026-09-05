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

const std = @import("std");
const platform = @import("antfly_platform");
const backends = @import("../backends/backends.zig");
const cache_mod = @import("cache.zig");
const decode_state_runtime = @import("decode_state_runtime.zig");
const debug_timing = @import("../debug_timing.zig");
const session_factory = @import("../architectures/session_factory.zig");
const generation = @import("../pipelines/generation.zig");
const gpt_arch = @import("../architectures/gpt.zig");
const gpt_mod = @import("../models/gpt.zig");
const ops = @import("../ops/ops.zig");
const runtime = @import("../runtime/root.zig");
const model_runtime = @import("model_runtime.zig");

/// Live whole-model executor for native sessions.
///
/// This is the first step toward treating live sessions and offline artifact
/// packages as the same model-level runtime concept. Unlike the older native
/// generation path, this runtime owns decode-state advancement itself: prefill
/// seeds the cache, decode consumes one token and advances the cache for the
/// next step.
const ExecutorContext = struct {
    allocator: std.mem.Allocator,
    session: backends.Session,
    gpt_config: gpt_mod.Config,
    kv_dtype: ?runtime.kv.pool.KvDType = null,
    shared_moe_cache: ?*runtime.moe.shared.SharedExpertCache,
};

const RuntimeContext = struct {
    allocator: std.mem.Allocator,
    cb: ops.ComputeBackend,
    gpt_config: gpt_mod.Config,
    kv_manager: runtime.kv.manager.KvManager,
    kv_storage: runtime.kv.storage_runtime.KvStorageRuntime,
    pool_id: runtime.kv.block.KvPoolId,
    decode_runtime: decode_state_runtime.DecodeStateRuntime,
    shared_moe_cache: ?*runtime.moe.shared.SharedExpertCache,
    qwen3vl_whole_token_frame_enabled: bool,

    fn init(
        allocator: std.mem.Allocator,
        session: backends.Session,
        gpt_config: gpt_mod.Config,
        kv_dtype_override: ?runtime.kv.pool.KvDType,
        shared_moe_cache: ?*runtime.moe.shared.SharedExpertCache,
    ) !*RuntimeContext {
        const cb = try session_factory.getComputeBackend(session, allocator);
        errdefer {
            var cb_mut = cb;
            cb_mut.deinit();
        }

        var kv_manager = runtime.kv.manager.KvManager.init(allocator);
        errdefer kv_manager.deinit();

        const backend_kind: runtime.kv.pool.BackendKind = switch (session.backend()) {
            .native => .native,
            .metal => .metal,
            .cuda => .cuda,
            .pjrt => return error.UnexpectedPjrtBackend,
            .onnx => return error.UnexpectedOnnxBackend,
            .wasm => return error.UnexpectedWasmBackend,
        };
        const kv_dtype = kv_dtype_override orelse session_factory.recommendedKvDTypeForSession(session, backend_kind);
        const kv_pool_config: runtime.kv.pool.KvPoolConfig = .{
            .backend = backend_kind,
            .dtype = kv_dtype,
            .page_size_tokens = 16,
            .num_layers_packed = @intCast(gpt_config.num_hidden_layers),
            .num_kv_heads = gpt_config.maxKvHeads(),
            .head_dim = gpt_config.maxHeadDim(),
            .sliding_window_size = gpt_config.kvPoolSlidingWindowSize(false),
        };
        const pool_id = try kv_manager.addPool(kv_pool_config);

        var kv_storage = try runtime.kv.storage_runtime.KvStorageRuntime.init(allocator, kv_pool_config);
        errdefer kv_storage.deinit();
        try cb.provisionKvDeviceWriteHook(&kv_storage);

        const ctx = try allocator.create(RuntimeContext);
        ctx.* = .{
            .allocator = allocator,
            .cb = cb,
            .gpt_config = gpt_config,
            .kv_manager = kv_manager,
            .kv_storage = kv_storage,
            .pool_id = pool_id,
            .decode_runtime = undefined,
            .shared_moe_cache = shared_moe_cache,
            .qwen3vl_whole_token_frame_enabled = platform.env.getenvBoolDefault(
                "TERMITE_METAL_ENABLE_QWEN3VL_WHOLE_TOKEN_FRAME",
                false,
            ),
        };
        ctx.decode_runtime = decode_state_runtime.DecodeStateRuntime.initPaged(
            allocator,
            &ctx.kv_manager,
            pool_id,
            shared_moe_cache,
        );
        ctx.decode_runtime.state.kv_storage = &ctx.kv_storage;
        ctx.decode_runtime.configureForGptConfig(gpt_config);
        return ctx;
    }

    fn resetState(self: *RuntimeContext) !void {
        self.decode_runtime.deinit();
        self.decode_runtime = decode_state_runtime.DecodeStateRuntime.initPaged(
            self.allocator,
            &self.kv_manager,
            self.pool_id,
            self.shared_moe_cache,
        );
        self.decode_runtime.state.kv_storage = &self.kv_storage;
        self.decode_runtime.configureForGptConfig(self.gpt_config);
    }

    fn deinit(self: *RuntimeContext) void {
        self.decode_runtime.deinit();
        self.kv_storage.deinit();
        self.kv_manager.deinit();
        self.cb.deinit();
        self.allocator.destroy(self);
    }

    fn currentTokenCount(self: *const RuntimeContext) usize {
        return self.decode_runtime.currentTokenCount();
    }

    fn kvView(self: *const RuntimeContext) ?generation.KvView {
        return self.decode_runtime.kvView();
    }

    fn notePrefill(self: *RuntimeContext, token_count: usize) !void {
        try self.decode_runtime.notePrefill(token_count);
    }

    fn appendPrefillChunk(self: *RuntimeContext, token_count: usize) !void {
        try self.decode_runtime.appendPrefillChunk(token_count);
    }

    fn appendGeneratedToken(self: *RuntimeContext) !usize {
        return self.decode_runtime.appendGeneratedToken();
    }

    fn appendGeneratedTokens(self: *RuntimeContext, count: usize) !usize {
        return self.decode_runtime.appendGeneratedTokens(count);
    }

    fn truncateGeneratedTokens(self: *RuntimeContext, count: usize) !void {
        try self.decode_runtime.truncateGeneratedTokens(count);
    }

    fn compactKvCache(self: *RuntimeContext, config: runtime.kv.compaction.CompactionConfig) !void {
        try self.decode_runtime.compactKvCache(config);
    }

    fn validateDecodePosition(self: *const RuntimeContext, position: usize) !void {
        try self.decode_runtime.validateDecodePosition(position);
    }

    fn makeDecodeContext(
        self: *RuntimeContext,
        seq_len: usize,
        query_seq_len: usize,
        attention_mode: cache_mod.AttentionMode,
    ) gpt_arch.DecodeContext {
        return self.decode_runtime.makeDecodeContext(seq_len, query_seq_len, attention_mode);
    }

    fn preparePrefill(self: *RuntimeContext, seq_len: usize, query_seq_len: usize, attention_mode: cache_mod.AttentionMode) !gpt_arch.DecodeContext {
        return self.decode_runtime.preparePrefill(seq_len, query_seq_len, attention_mode);
    }

    fn beginDecodeStep(self: *RuntimeContext, position: usize, attention_mode: cache_mod.AttentionMode) !struct {
        seq_len: usize,
        decode_context: gpt_arch.DecodeContext,
    } {
        const step = try self.decode_runtime.beginDecodeStep(position, attention_mode);
        return .{
            .seq_len = step.seq_len,
            .decode_context = step.decode_context,
        };
    }
};

const runtime_vtable = model_runtime.ModelRuntime.VTable{
    .capabilities = runtimeCapabilities,
    .prefill = runtimePrefill,
    .decode = runtimeDecode,
    .decode_sample = runtimeDecodeSample,
    .decode_greedy = runtimeDecodeGreedy,
    .deinit = runtimeDeinit,
    .reset = runtimeReset,
    .debug_timing_stats = runtimeDebugTimingStats,
    .reset_debug_timing_stats = runtimeResetDebugTimingStats,
    .print_debug_timing = runtimePrintDebugTiming,
};

const executor_vtable = model_runtime.ModelExecutor.VTable{
    .create_runtime = createRuntime,
    .deinit = executorDeinit,
};

fn supportsBackend(backend: backends.BackendType) bool {
    return switch (backend) {
        .native, .metal => true,
        else => false,
    };
}

pub fn supportsSession(session: backends.Session) bool {
    return supportsBackend(session.backend());
}

pub fn createModelExecutor(
    allocator: std.mem.Allocator,
    session: backends.Session,
    gpt_config: gpt_mod.Config,
    kv_dtype: ?runtime.kv.pool.KvDType,
    shared_moe_cache: ?*runtime.moe.shared.SharedExpertCache,
) !model_runtime.ModelExecutor {
    if (!supportsSession(session)) return error.UnsupportedCompileBackend;
    const ctx = try allocator.create(ExecutorContext);
    ctx.* = .{
        .allocator = allocator,
        .session = session,
        .gpt_config = gpt_config,
        .kv_dtype = kv_dtype,
        .shared_moe_cache = shared_moe_cache,
    };
    return .{ .ptr = ctx, .vtable = &executor_vtable };
}

fn createRuntime(ctx: *anyopaque, allocator: std.mem.Allocator) !model_runtime.ModelRuntime {
    const exec_ctx: *ExecutorContext = @ptrCast(@alignCast(ctx));
    const runtime_ctx = try RuntimeContext.init(
        allocator,
        exec_ctx.session,
        exec_ctx.gpt_config,
        exec_ctx.kv_dtype,
        exec_ctx.shared_moe_cache,
    );
    return .{ .ptr = runtime_ctx, .vtable = &runtime_vtable };
}

fn executorDeinit(ctx: *anyopaque) void {
    const exec_ctx: *ExecutorContext = @ptrCast(@alignCast(ctx));
    exec_ctx.allocator.destroy(exec_ctx);
}

fn runtimeCapabilities(ctx: *anyopaque) model_runtime.RuntimeCapabilities {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    _ = runtime_ctx;
    return .{
        .supports_decode = true,
        .supports_sample_decode = true,
        .supports_greedy_decode = true,
        .state_ownership = .runtime_owned_host_cache,
    };
}

fn runtimeReset(ctx: *anyopaque) !void {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    try runtime_ctx.resetState();
}

fn runtimeDebugTimingStats(ctx: *anyopaque) model_runtime.RuntimeDebugTimingStats {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    return .{
        .backend = runtime_ctx.cb.debugTimingSnapshot(),
        .decoder_runtime_ready = runtime_ctx.cb.decoderRuntimeReady(),
        .decoder_runtime_absolute_embeddings_prepared = runtime_ctx.cb.decoderRuntimeAbsoluteEmbeddingsPrepared(),
    };
}

fn runtimeResetDebugTimingStats(ctx: *anyopaque) void {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    gpt_arch.resetDebugTimingStats();
    runtime_ctx.cb.resetDebugTimingStats();
}

fn runtimePrintDebugTiming(ctx: *anyopaque) void {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    const stats = runtimeDebugTimingStats(ctx);
    debug_timing.printBackendTimingDetails(
        runtime_ctx.cb.kind(),
        stats.backend,
        stats.decoder_runtime_ready,
        stats.decoder_runtime_absolute_embeddings_prepared,
    );
}

fn runtimeDeinit(ctx: *anyopaque) void {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    runtime_ctx.deinit();
}

fn runtimePrefill(
    ctx: *anyopaque,
    allocator: std.mem.Allocator,
    request: model_runtime.PrefillRequest,
) !model_runtime.ModelOutput {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    if (request.input_ids.len == 0 or request.query_seq_len == 0) return error.EmptyPrompt;
    if (request.input_ids.len != request.query_seq_len) return error.UnsupportedShape;
    if (request.query_seq_len > request.seq_len) return error.UnsupportedShape;
    const decode_context = try runtime_ctx.preparePrefill(request.seq_len, request.query_seq_len, request.attention_mode);
    return .{
        .logits = try forwardLastLogits(
            runtime_ctx,
            allocator,
            request.input_ids,
            request.seq_len,
            request.query_seq_len,
            &decode_context,
        ),
    };
}

fn runtimeDecode(
    ctx: *anyopaque,
    allocator: std.mem.Allocator,
    request: model_runtime.DecodeRequest,
) !model_runtime.ModelOutput {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    const step = try runtime_ctx.beginDecodeStep(request.position, request.attention_mode);
    const input_ids = [_]i64{request.token_id};
    return .{ .logits = try forwardLastLogits(runtime_ctx, allocator, input_ids[0..], step.seq_len, 1, &step.decode_context) };
}

fn runtimeDecodeSample(
    ctx: *anyopaque,
    allocator: std.mem.Allocator,
    request: model_runtime.SampledDecodeRequest,
) !model_runtime.SampledDecodeOutput {
    if (request.sampling.isPureGreedy()) {
        const greedy = try runtimeDecodeGreedy(ctx, allocator, request.decode);
        return .{ .token_id = greedy.token_id };
    }

    var output = try runtimeDecode(ctx, allocator, request.decode);
    defer output.deinit(allocator);
    return .{
        .token_id = @intCast(model_runtime.sampleTokenFromLogits(
            allocator,
            try output.hostLogits(allocator),
            request.sampling,
            request.token_history,
        )),
    };
}

fn runtimeDecodeGreedy(
    ctx: *anyopaque,
    allocator: std.mem.Allocator,
    request: model_runtime.DecodeRequest,
) !model_runtime.GreedyDecodeOutput {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    const step = try runtime_ctx.beginDecodeStep(request.position, request.attention_mode);
    const input_ids = [_]i64{request.token_id};
    const qwen3vl_prepared_decode_block = runtime_ctx.cb.kind() == .metal and
        runtime_ctx.gpt_config.family == .qwen3_vl and
        step.decode_context.qwen3vl_text_only and
        request.attention_mode == .paged_decode and
        runtime_ctx.qwen3vl_whole_token_frame_enabled and
        platform.env.getenvBoolDefault("TERMITE_METAL_ENABLE_QWEN3VL_PREPARED_DECODE_BLOCK", false) and
        !platform.env.getenvBoolDefault("TERMITE_METAL_DISABLE_QWEN3VL_PREPARED_DECODE_BLOCK", false);
    if (qwen3vl_prepared_decode_block) {
        _ = try runtime_ctx.cb.decoderRuntimePrepareOrReuseFamily(
            allocator,
            runtime_ctx.gpt_config,
            step.decode_context.kv_sequence_len,
            runtime_ctx.gpt_config.num_hidden_layers,
        );
    }
    var whole_token_frame_active = false;
    if (runtime_ctx.cb.kind() == .metal and
        runtime_ctx.gpt_config.family == .qwen3_vl and
        step.decode_context.qwen3vl_text_only and
        request.attention_mode == .paged_decode and
        runtime_ctx.qwen3vl_whole_token_frame_enabled and
        !runtime_ctx.cb.decoderRuntimeHasActiveFrame())
    {
        whole_token_frame_active = try runtime_ctx.cb.decoderRuntimeBeginFrame();
        if (whole_token_frame_active) try runtime_ctx.cb.decoderRuntimeSetActiveFrameRegime(.decode);
    }
    errdefer if (whole_token_frame_active and runtime_ctx.cb.decoderRuntimeHasActiveFrame())
        runtime_ctx.cb.decoderRuntimeCancelFrame() catch {};

    const token_id = try gpt_arch.forwardGreedyLastToken(
        &runtime_ctx.cb,
        allocator,
        runtime_ctx.gpt_config,
        input_ids[0..],
        1,
        step.seq_len,
        &step.decode_context,
    );
    if (whole_token_frame_active and runtime_ctx.cb.decoderRuntimeHasActiveFrame()) {
        try runtime_ctx.cb.decoderRuntimeSubmitAndWaitFrame();
    }
    whole_token_frame_active = false;
    return .{
        .token_id = @intCast(token_id),
    };
}

fn forwardLastLogits(
    runtime_ctx: *RuntimeContext,
    allocator: std.mem.Allocator,
    input_ids: []const i64,
    seq_len: usize,
    query_seq_len: usize,
    decode_context: *const gpt_arch.DecodeContext,
) ![]f32 {
    const logits = try gpt_arch.forward(
        &runtime_ctx.cb,
        allocator,
        runtime_ctx.gpt_config,
        input_ids,
        1,
        seq_len,
        decode_context,
    );
    defer allocator.free(logits);
    const vocab_size: usize = @intCast(runtime_ctx.gpt_config.vocab_size);
    const last_pos_offset = (query_seq_len - 1) * vocab_size;
    return allocator.dupe(f32, logits[last_pos_offset..][0..vocab_size]);
}

test "live model executor supports native and metal backends" {
    try std.testing.expect(supportsBackend(.native));
    try std.testing.expect(supportsBackend(.metal));
    try std.testing.expect(!supportsBackend(.onnx));
    try std.testing.expect(!supportsBackend(.pjrt));
}
