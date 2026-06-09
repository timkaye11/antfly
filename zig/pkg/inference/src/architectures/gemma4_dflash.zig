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
const build_options = @import("build_options");
const platform = @import("antfly_platform");
const gpt = @import("../models/gpt.zig");
const ops = @import("../ops/ops.zig");
const metal_compute = @import("../ops/metal_compute.zig");

pub const default_block_size: usize = 16;
pub const max_block_size: usize = 64;
pub const default_feature_bank_capacity: usize = 4096;
pub const default_cross_context: usize = 1024;
pub const max_feature_bank_capacity: usize = 4096;
pub const debug_host_fallback_env = "ANTFLY_DFLASH_DEBUG_HOST_FALLBACK";

pub const ResidencyPolicy = enum {
    fail_closed,
    debug_host_fallback,
};

pub const SamplingConfig = struct {
    temperature: f32 = 0,
    top_p: f32 = 0,
    top_k: i32 = 0,
    min_p: f32 = 0,
    repetition_penalty: f32 = 1.0,
    frequency_penalty: f32 = 0,
    presence_penalty: f32 = 0,
};

pub const ResidencyCounters = struct {
    host_fallbacks: usize = 0,
    full_tensor_download_bytes: usize = 0,
    device_feature_captures: usize = 0,
    feature_fusion_nanos: u128 = 0,
    kv_injection_nanos: u128 = 0,
    draft_block_nanos: u128 = 0,
};

pub const DeviceDraftRequest = struct {
    allocator: std.mem.Allocator,
    target_cb: *const ops.ComputeBackend,
    draft_cb: *const ops.ComputeBackend,
    target_config: gpt.Config,
    draft_config: gpt.Config,
    feature_bank: *DFlashFeatureBank,
    token_context: []const i64,
    current_seq_len: usize,
    block_size: usize,
    sampling: SamplingConfig,
    policy: ResidencyPolicy,
    output_tokens: []i64,
};

pub const DeviceDraftResult = struct {
    token_ids: []const i64,
    counters: ResidencyCounters = .{},
};

pub const DFlashInjectedKvContext = struct {
    allocator: std.mem.Allocator,
    cb: *const ops.ComputeBackend,
    layers: []Layer,

    pub const Layer = struct {
        k_ctx: ops.CT,
        v_ctx: ops.CT,
        rows: usize,
    };

    pub fn deinit(self: *DFlashInjectedKvContext) void {
        for (self.layers) |layer| {
            self.cb.free(layer.k_ctx);
            self.cb.free(layer.v_ctx);
        }
        self.allocator.free(self.layers);
        self.* = undefined;
    }
};

pub const DFlashFeatureBank = struct {
    allocator: std.mem.Allocator,
    cb: *const ops.ComputeBackend,
    hidden_size: usize,
    capacity: usize,
    cross_context: usize,
    slots: []Slot,
    capture_count: usize = 0,

    pub const Slot = struct {
        layer_index: usize,
        tensor: ?ops.CT = null,
        start_token: usize = 0,
        rows: usize = 0,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        cb: *const ops.ComputeBackend,
        target_config: gpt.Config,
        draft_config: gpt.Config,
    ) !DFlashFeatureBank {
        try validatePair(target_config, draft_config);
        if (cb.kind() != .metal) return error.DFlashRequiresMetalBackend;
        const layer_count: usize = @intCast(draft_config.dflash_target_feature_layer_count);
        const slots = try allocator.alloc(Slot, layer_count);
        errdefer allocator.free(slots);
        for (slots, 0..) |*slot, idx| {
            slot.* = .{ .layer_index = @intCast(draft_config.dflash_target_feature_layers[idx]) };
        }
        return .{
            .allocator = allocator,
            .cb = cb,
            .hidden_size = @intCast(target_config.hidden_size),
            .capacity = effectiveFeatureBankCapacity(draft_config),
            .cross_context = effectiveCrossContext(draft_config),
            .slots = slots,
        };
    }

    pub fn deinit(self: *DFlashFeatureBank) void {
        for (self.slots) |*slot| {
            if (slot.tensor) |tensor| self.cb.free(tensor);
            slot.* = undefined;
        }
        self.allocator.free(self.slots);
        self.* = undefined;
    }

    pub fn wantsLayer(self: *const DFlashFeatureBank, layer_index: usize) bool {
        return self.slotIndexForLayer(layer_index) != null;
    }

    pub fn capturedLayerCount(self: *const DFlashFeatureBank) usize {
        var count: usize = 0;
        for (self.slots) |slot| {
            if (slot.tensor != null) count += 1;
        }
        return count;
    }

    pub fn totalCaptureCount(self: *const DFlashFeatureBank) usize {
        return self.capture_count;
    }

    pub fn validateReady(self: *const DFlashFeatureBank) !void {
        if (self.slots.len == 0) return error.DFlashTargetFeatureLayersMissing;
        var expected_rows: usize = 0;
        var expected_start: ?usize = null;
        for (self.slots) |slot| {
            const tensor = slot.tensor orelse return error.DFlashTargetFeatureBankIncomplete;
            if (slot.rows == 0) return error.DFlashTargetFeatureBankIncomplete;
            if (expected_rows == 0) {
                expected_rows = slot.rows;
                expected_start = slot.start_token;
            } else if (slot.rows != expected_rows or slot.start_token != expected_start.?) {
                return error.DFlashTargetFeatureBankShapeMismatch;
            }
            if (!hasDeviceTensor(self.cb, tensor)) return error.DFlashFeatureCaptureNotDeviceResident;
        }
    }

    pub fn captureLayer(
        self: *DFlashFeatureBank,
        layer_index: usize,
        hidden: ops.CT,
        row_start_token: usize,
        row_count: usize,
    ) !bool {
        const slot_index = self.slotIndexForLayer(layer_index) orelse return false;
        if (self.cb.kind() != .metal) return error.DFlashRequiresMetalBackend;
        if (row_count == 0) return false;
        if (row_count > self.capacity and self.cross_context == 0) return error.DFlashFeatureCaptureTooLarge;
        if (!hasDeviceTensor(self.cb, hidden)) {
            return error.DFlashFeatureCaptureNotDeviceResident;
        }

        const max_rows = @min(self.capacity, self.cross_context);
        var capture_rows = @min(row_count, max_rows);
        if (capture_rows == 0) return error.DFlashFeatureCaptureTooLarge;
        var capture_start_row = row_count - capture_rows;
        var capture_start_token = row_start_token + capture_start_row;

        const slot = &self.slots[slot_index];
        const previous_rows = slot.rows;
        const previous_end = if (previous_rows == 0) capture_start_token else slot.start_token + previous_rows;
        var append_existing = previous_rows != 0 and capture_start_token <= previous_end;
        if (append_existing and capture_start_token < previous_end) {
            const overlap = previous_end - capture_start_token;
            if (overlap >= capture_rows) return false;
            capture_start_row += overlap;
            capture_start_token += overlap;
            capture_rows -= overlap;
        }
        if (previous_rows != 0 and capture_start_token > previous_end) {
            append_existing = false;
        }

        const view = try self.cb.sliceRows2D(
            self.allocator,
            hidden,
            capture_start_row,
            capture_rows,
            self.hidden_size,
        );
        defer self.cb.free(view);
        if (!hasDeviceTensor(self.cb, view)) {
            return error.DFlashFeatureCaptureNotDeviceResident;
        }

        const prior_tensor: ?ops.CT = if (append_existing) slot.tensor else null;
        const appended = (try metal_compute.MetalCompute.appendTrimRowsDevice(self.cb, prior_tensor, view, self.hidden_size, max_rows)) orelse return error.DFlashFeatureCaptureNotDeviceResident;
        errdefer self.cb.free(appended);
        if (!hasDeviceTensor(self.cb, appended)) return error.DFlashFeatureCaptureNotDeviceResident;

        if (slot.tensor) |old| self.cb.free(old);
        slot.tensor = appended;
        const end_token = capture_start_token + capture_rows;
        slot.rows = if (append_existing) @min(previous_rows + capture_rows, max_rows) else capture_rows;
        slot.start_token = end_token - slot.rows;
        self.capture_count += 1;
        return true;
    }

    pub fn appendRowsFromBank(
        self: *DFlashFeatureBank,
        source: *const DFlashFeatureBank,
        source_row_start: usize,
        token_start: usize,
        row_count: usize,
    ) !usize {
        if (row_count == 0) return 0;
        try source.validateReady();
        var captures: usize = 0;
        for (self.slots) |slot| {
            const source_index = source.slotIndexForLayer(slot.layer_index) orelse return error.DFlashTargetFeatureBankIncomplete;
            const source_slot = source.slots[source_index];
            const source_tensor = source_slot.tensor orelse return error.DFlashTargetFeatureBankIncomplete;
            if (source_row_start + row_count > source_slot.rows) return error.InvalidTensorShape;
            if (!hasDeviceTensor(source.cb, source_tensor)) return error.DFlashFeatureCaptureNotDeviceResident;
            const view = try source.cb.sliceRows2D(
                self.allocator,
                source_tensor,
                source_row_start,
                row_count,
                self.hidden_size,
            );
            defer source.cb.free(view);
            if (!hasDeviceTensor(source.cb, view)) return error.DFlashFeatureCaptureNotDeviceResident;
            if (try self.captureLayer(slot.layer_index, view, token_start, row_count)) captures += 1;
        }
        return captures;
    }

    fn slotIndexForLayer(self: *const DFlashFeatureBank, layer_index: usize) ?usize {
        for (self.slots, 0..) |slot, idx| {
            if (slot.layer_index == layer_index) return idx;
        }
        return null;
    }
};

fn hasDeviceTensor(cb: *const ops.ComputeBackend, tensor: ops.CT) bool {
    if (comptime !build_options.enable_metal) return false;
    return metal_compute.MetalCompute.debugHasDeviceTensor(cb, tensor);
}

pub fn validatePair(target: gpt.Config, draft: gpt.Config) !void {
    if (!draft.dflash_draft) return error.DFlashDraftMetadataMissing;
    if (target.family != .gemma) return error.DFlashRequiresGemma4Target;
    if (!target.isGemma4()) return error.DFlashRequiresGemma4Target;
    if (draft.dflash_target_hidden_size != 0 and draft.dflash_target_hidden_size != target.hidden_size) {
        return error.DFlashTargetHiddenSizeMismatch;
    }
    if (draft.vocab_size != 0 and target.vocab_size != 0 and draft.vocab_size != target.vocab_size) {
        return error.DFlashVocabMismatch;
    }
    if (draft.num_local_experts != 0 or draft.num_experts_per_tok != 0 or draft.num_shared_experts != 0) {
        return error.DFlashMoEDraftUnsupported;
    }
    if (draft.sliding_window > 0) return error.DFlashSlidingWindowDraftUnsupported;
    if (draft.dflash_block_size > max_block_size) return error.DFlashBlockSizeUnsupported;
    if (draft.dflash_target_feature_layer_count == 0) return error.DFlashTargetFeatureLayersMissing;
    if (draft.dflash_feature_bank_capacity > max_feature_bank_capacity) return error.DFlashFeatureBankTooLarge;
    if (draft.dflash_cross_context > effectiveFeatureBankCapacity(draft)) return error.DFlashCrossContextTooLarge;
    if (draft.num_hidden_layers == 0) return error.DFlashDraftMetadataMissing;
    if (draft.num_attention_heads == 0 or draft.effectiveKVHeads() == 0 or draft.headDim() == 0) {
        return error.DFlashDraftMetadataMissing;
    }
}

pub fn validateDeviceRequest(request: DeviceDraftRequest) !void {
    try validatePair(request.target_config, request.draft_config);
    if (request.target_cb.kind() != .metal or request.draft_cb.kind() != .metal) {
        return error.DFlashRequiresMetalBackend;
    }
    if (request.block_size == 0 or request.block_size > max_block_size) {
        return error.DFlashBlockSizeUnsupported;
    }
    if (!isDeterministicSampling(request.sampling)) {
        return error.DFlashRequiresDeterministicSampling;
    }
    if (request.current_seq_len == 0 or request.current_seq_len > request.token_context.len) {
        return error.InvalidTensorShape;
    }
    if (request.output_tokens.len < request.block_size) {
        return error.InvalidTensorShape;
    }
    try request.feature_bank.validateReady();
}

pub fn draftBlockDevice(request: DeviceDraftRequest) !DeviceDraftResult {
    try validateDeviceRequest(request);
    if (request.policy == .debug_host_fallback) return error.DFlashDebugHostFallbackRequested;
    const total_started = monotonicNanoTimestamp();
    var counters = ResidencyCounters{};
    const fusion_started = monotonicNanoTimestamp();
    const fused = try fuseTargetFeaturesDevice(request);
    defer request.draft_cb.free(fused.tensor);
    counters.feature_fusion_nanos = elapsedNanos(fusion_started);

    const kv_started = monotonicNanoTimestamp();
    var injected = try materializeInjectedKvContextDevice(request, fused);
    defer injected.deinit();
    counters.kv_injection_nanos = elapsedNanos(kv_started);

    const noise = try materializeNoiseEmbeddingBlockDevice(request);
    defer request.target_cb.free(noise);

    const draft_hidden = try runMaskedDraftForwardDevice(request, fused, &injected, noise);
    defer request.draft_cb.free(draft_hidden);

    try selectDraftTokenIdsDevice(request, draft_hidden);
    counters.draft_block_nanos = elapsedNanos(total_started);
    return .{
        .token_ids = request.output_tokens[0..request.block_size],
        .counters = counters,
    };
}

fn monotonicNanoTimestamp() u128 {
    if (comptime @import("builtin").os.tag == .freestanding) return 0;
    var ts: std.posix.timespec = undefined;
    return switch (std.posix.errno(std.posix.system.clock_gettime(.MONOTONIC, &ts))) {
        .SUCCESS => @intCast(@as(i128, ts.sec) * std.time.ns_per_s + ts.nsec),
        else => 0,
    };
}

fn elapsedNanos(started: u128) u128 {
    const finished = monotonicNanoTimestamp();
    return if (finished >= started) finished - started else 0;
}

pub fn readScalarTokenIdDevice(cb: *const ops.ComputeBackend, allocator: std.mem.Allocator, token_tensor: ops.CT) !i64 {
    const ids = try cb.toFloat32(token_tensor, allocator);
    defer allocator.free(ids);
    if (ids.len != 1 or !std.math.isFinite(ids[0]) or ids[0] < 0) return error.InvalidTensorShape;
    return @intFromFloat(ids[0]);
}

const FusedTargetFeatures = struct {
    tensor: ops.CT,
    rows: usize,
    nanos: u128 = 0,
};

fn fuseTargetFeaturesDevice(request: DeviceDraftRequest) !FusedTargetFeatures {
    if (try fuseTargetFeaturesWithZLabFcDevice(request)) |fused| return fused;
    return fuseTargetFeaturesWithLegacyPerLayerWeightsDevice(request);
}

fn fuseTargetFeaturesWithZLabFcDevice(request: DeviceDraftRequest) !?FusedTargetFeatures {
    const fc_weight = request.draft_cb.getWeight("fc.weight") catch |err| switch (err) {
        error.MissingWeight => return null,
        else => return err,
    };
    const hidden_norm_weight = request.draft_cb.getWeight("hidden_norm.weight") catch |err| switch (err) {
        error.MissingWeight => return error.DFlashFusionWeightMissing,
        else => return err,
    };
    const rows = try commonFeatureRows(request.feature_bank);
    const target_hidden_size: usize = @intCast(request.target_config.hidden_size);
    const draft_hidden_size: usize = @intCast(request.draft_config.hidden_size);
    const concatenated = try concatenateFeatureBankSlotsLastDim(request, rows, target_hidden_size);
    defer request.draft_cb.free(concatenated);
    const projected = try request.draft_cb.linearNoBias(
        concatenated,
        fc_weight,
        rows,
        target_hidden_size * request.feature_bank.slots.len,
        draft_hidden_size,
    );
    defer request.draft_cb.free(projected);
    const normed = try request.draft_cb.rmsNorm(projected, hidden_norm_weight, draft_hidden_size, request.draft_config.norm_eps);
    errdefer request.draft_cb.free(normed);
    if (!hasDeviceTensor(request.draft_cb, normed)) return error.DFlashFeatureCaptureNotDeviceResident;
    return .{ .tensor = normed, .rows = rows };
}

fn fuseTargetFeaturesWithLegacyPerLayerWeightsDevice(request: DeviceDraftRequest) !FusedTargetFeatures {
    var fused: ?ops.CT = null;
    errdefer if (fused) |tensor| request.draft_cb.free(tensor);
    var fused_rows: usize = 0;
    for (request.feature_bank.slots, 0..) |slot, slot_index| {
        const tensor = slot.tensor orelse return error.DFlashTargetFeatureBankIncomplete;
        if (!hasDeviceTensor(request.feature_bank.cb, tensor)) return error.DFlashFeatureCaptureNotDeviceResident;
        const weight = try loadFeatureFusionWeight(request, slot.layer_index, slot_index);
        defer request.draft_cb.free(weight);
        const projected = try request.draft_cb.linearNoBias(
            tensor,
            weight,
            slot.rows,
            @intCast(request.target_config.hidden_size),
            @intCast(request.draft_config.hidden_size),
        );
        errdefer request.draft_cb.free(projected);
        if (fused) |current| {
            if (slot.rows != fused_rows) return error.DFlashTargetFeatureBankShapeMismatch;
            const combined = try request.draft_cb.add(current, projected);
            request.draft_cb.free(current);
            request.draft_cb.free(projected);
            fused = combined;
        } else {
            fused = projected;
            fused_rows = slot.rows;
        }
    }
    const tensor = fused orelse return error.DFlashTargetFeatureBankIncomplete;
    return .{
        .tensor = tensor,
        .rows = fused_rows,
        .nanos = 0,
    };
}

fn commonFeatureRows(feature_bank: *const DFlashFeatureBank) !usize {
    try feature_bank.validateReady();
    var rows: usize = 0;
    for (feature_bank.slots) |slot| {
        if (rows == 0) rows = slot.rows else if (rows != slot.rows) return error.DFlashTargetFeatureBankShapeMismatch;
    }
    if (rows == 0) return error.DFlashTargetFeatureBankIncomplete;
    return rows;
}

fn concatenateFeatureBankSlotsLastDim(request: DeviceDraftRequest, rows: usize, target_hidden_size: usize) !ops.CT {
    var concatenated: ?ops.CT = null;
    errdefer if (concatenated) |tensor| request.draft_cb.free(tensor);
    var current_dim: usize = 0;
    for (request.feature_bank.slots) |slot| {
        const tensor = slot.tensor orelse return error.DFlashTargetFeatureBankIncomplete;
        if (slot.rows != rows) return error.DFlashTargetFeatureBankShapeMismatch;
        if (!hasDeviceTensor(request.feature_bank.cb, tensor)) return error.DFlashFeatureCaptureNotDeviceResident;
        if (concatenated) |current| {
            const combined = try request.draft_cb.concat(current, tensor, rows, current_dim, target_hidden_size);
            request.draft_cb.free(current);
            concatenated = combined;
            current_dim += target_hidden_size;
        } else {
            const copy = (try request.draft_cb.zeroTensor(rows, target_hidden_size)) orelse return error.DFlashFeatureCaptureNotDeviceResident;
            errdefer request.draft_cb.free(copy);
            if (!(try metal_compute.MetalCompute.copyTensorInto(request.draft_cb, tensor, copy))) {
                return error.DFlashFeatureCaptureNotDeviceResident;
            }
            concatenated = copy;
            current_dim = target_hidden_size;
        }
    }
    return concatenated orelse error.DFlashTargetFeatureBankIncomplete;
}

fn materializeInjectedKvContextDevice(request: DeviceDraftRequest, fused: FusedTargetFeatures) !DFlashInjectedKvContext {
    if (!hasDeviceTensor(request.draft_cb, fused.tensor)) return error.DFlashFeatureCaptureNotDeviceResident;
    const layer_count = effectiveDraftLayerCount(request.draft_config);
    const kv_dim = dflashKvProjectionDim(request.draft_config);
    const head_dim: usize = @intCast(request.draft_config.headDim());
    if (layer_count == 0 or kv_dim == 0 or head_dim == 0) return error.DFlashDraftMetadataMissing;
    var layers = try request.allocator.alloc(DFlashInjectedKvContext.Layer, layer_count);
    var initialized: usize = 0;
    errdefer {
        for (layers[0..initialized]) |layer| {
            request.draft_cb.free(layer.k_ctx);
            request.draft_cb.free(layer.v_ctx);
        }
        request.allocator.free(layers);
    }
    for (0..layer_count) |layer_index| {
        const k_weight = try loadDraftLayerWeight(request, layer_index, "self_attn.k_proj.weight", error.DFlashKvInjectionWeightMissing);
        const v_weight = try loadDraftLayerWeight(request, layer_index, "self_attn.v_proj.weight", error.DFlashKvInjectionWeightMissing);
        const k_norm_weight = try loadDraftLayerWeight(request, layer_index, "self_attn.k_norm.weight", error.DFlashKvInjectionWeightMissing);
        const k_projected = try request.draft_cb.linearNoBias(fused.tensor, k_weight, fused.rows, @intCast(request.draft_config.hidden_size), kv_dim);
        errdefer request.draft_cb.free(k_projected);
        const k_normed = try request.draft_cb.rmsNorm(k_projected, k_norm_weight, head_dim, request.draft_config.norm_eps);
        request.draft_cb.free(k_projected);
        errdefer request.draft_cb.free(k_normed);
        const v_projected = try request.draft_cb.linearNoBias(fused.tensor, v_weight, fused.rows, @intCast(request.draft_config.hidden_size), kv_dim);
        errdefer request.draft_cb.free(v_projected);
        if (!hasDeviceTensor(request.draft_cb, k_normed) or !hasDeviceTensor(request.draft_cb, v_projected)) {
            return error.DFlashFeatureCaptureNotDeviceResident;
        }
        layers[layer_index] = .{ .k_ctx = k_normed, .v_ctx = v_projected, .rows = fused.rows };
        initialized += 1;
    }
    return .{ .allocator = request.allocator, .cb = request.draft_cb, .layers = layers };
}

fn materializeNoiseEmbeddingBlockDevice(request: DeviceDraftRequest) !ops.CT {
    if (request.target_config.hidden_size != request.draft_config.hidden_size) {
        return error.DFlashNoiseEmbeddingProjectionMissing;
    }
    const mask = maskTokenId(request.draft_config) orelse return error.DFlashMaskTokenMissing;
    var ids_buf: [max_block_size]i64 = undefined;
    ids_buf[0] = request.token_context[request.current_seq_len - 1];
    const noise_rows = request.block_size + 1;
    if (noise_rows > ids_buf.len) return error.DFlashBlockSizeUnsupported;
    for (ids_buf[1..noise_rows]) |*id| id.* = mask;
    const embed_w = request.target_cb.getWeight("model.embed_tokens.weight") catch |err| switch (err) {
        error.MissingWeight => request.target_cb.getWeight("embed_tokens.weight") catch |inner| switch (inner) {
            error.MissingWeight => try request.target_cb.getWeight("transformer.wte.weight"),
            else => return inner,
        },
        else => return err,
    };
    const noise = try request.target_cb.embeddingLookup(embed_w, ids_buf[0..noise_rows], noise_rows, @intCast(request.target_config.hidden_size));
    errdefer request.target_cb.free(noise);
    if (!hasDeviceTensor(request.target_cb, noise)) return error.DFlashFeatureCaptureNotDeviceResident;
    return noise;
}

fn runMaskedDraftForwardDevice(
    request: DeviceDraftRequest,
    fused: FusedTargetFeatures,
    injected: *const DFlashInjectedKvContext,
    noise: ops.CT,
) !ops.CT {
    _ = fused;
    const layer_count = effectiveDraftLayerCount(request.draft_config);
    if (injected.layers.len != layer_count) return error.InvalidTensorShape;
    const noise_rows = request.block_size + 1;
    const hidden_size: usize = @intCast(request.draft_config.hidden_size);
    const num_heads: usize = @intCast(request.draft_config.num_attention_heads);
    const num_kv_heads: usize = @intCast(request.draft_config.effectiveKVHeads());
    const head_dim: usize = @intCast(request.draft_config.headDim());
    const q_dim = num_heads * head_dim;
    const kv_dim = num_kv_heads * head_dim;
    if (q_dim == 0 or kv_dim == 0 or hidden_size == 0) return error.DFlashDraftMetadataMissing;

    var hidden = try copyDeviceTensor(request.draft_cb, noise, noise_rows, hidden_size);
    errdefer request.draft_cb.free(hidden);
    for (0..layer_count) |layer_index| {
        const next_hidden = try runDraftLayerDevice(request, hidden, noise_rows, injected.layers[layer_index], layer_index, hidden_size, num_heads, num_kv_heads, head_dim);
        request.draft_cb.free(hidden);
        hidden = next_hidden;
    }
    const norm_w = try loadDraftRootWeight(request, "norm.weight", error.DFlashDraftWeightMissing);
    const final_hidden = try request.draft_cb.rmsNorm(hidden, norm_w, hidden_size, request.draft_config.norm_eps);
    request.draft_cb.free(hidden);
    errdefer request.draft_cb.free(final_hidden);
    if (!hasDeviceTensor(request.draft_cb, final_hidden)) return error.DFlashFeatureCaptureNotDeviceResident;
    return final_hidden;
}

fn runDraftLayerDevice(
    request: DeviceDraftRequest,
    hidden: ops.CT,
    rows: usize,
    injected_layer: DFlashInjectedKvContext.Layer,
    layer_index: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) !ops.CT {
    const q_dim = num_heads * head_dim;
    const kv_dim = num_kv_heads * head_dim;
    const input_norm_w = try loadDraftLayerWeight(request, layer_index, "input_layernorm.weight", error.DFlashDraftWeightMissing);
    const q_w = try loadDraftLayerWeight(request, layer_index, "self_attn.q_proj.weight", error.DFlashDraftWeightMissing);
    const k_w = try loadDraftLayerWeight(request, layer_index, "self_attn.k_proj.weight", error.DFlashDraftWeightMissing);
    const v_w = try loadDraftLayerWeight(request, layer_index, "self_attn.v_proj.weight", error.DFlashDraftWeightMissing);
    const q_norm_w = try loadDraftLayerWeight(request, layer_index, "self_attn.q_norm.weight", error.DFlashDraftWeightMissing);
    const k_norm_w = try loadDraftLayerWeight(request, layer_index, "self_attn.k_norm.weight", error.DFlashDraftWeightMissing);
    const o_w = try loadDraftLayerWeight(request, layer_index, "self_attn.o_proj.weight", error.DFlashDraftWeightMissing);
    const post_norm_w = try loadDraftLayerWeight(request, layer_index, "post_attention_layernorm.weight", error.DFlashDraftWeightMissing);
    const gate_w = try loadDraftLayerWeight(request, layer_index, "mlp.gate_proj.weight", error.DFlashDraftWeightMissing);
    const up_w = try loadDraftLayerWeight(request, layer_index, "mlp.up_proj.weight", error.DFlashDraftWeightMissing);
    const down_w = try loadDraftLayerWeight(request, layer_index, "mlp.down_proj.weight", error.DFlashDraftWeightMissing);
    const intermediate_size = try inferIntermediateSize(request, layer_index);

    const normed = try request.draft_cb.rmsNorm(hidden, input_norm_w, hidden_size, request.draft_config.norm_eps);
    defer request.draft_cb.free(normed);
    const q = try request.draft_cb.linearNoBias(normed, q_w, rows, hidden_size, q_dim);
    defer request.draft_cb.free(q);
    const q_normed = try request.draft_cb.rmsNorm(q, q_norm_w, head_dim, request.draft_config.norm_eps);
    defer request.draft_cb.free(q_normed);
    const k_noise = try request.draft_cb.linearNoBias(normed, k_w, rows, hidden_size, kv_dim);
    defer request.draft_cb.free(k_noise);
    const k_noise_normed = try request.draft_cb.rmsNorm(k_noise, k_norm_w, head_dim, request.draft_config.norm_eps);
    defer request.draft_cb.free(k_noise_normed);
    const v_noise = try request.draft_cb.linearNoBias(normed, v_w, rows, hidden_size, kv_dim);
    defer request.draft_cb.free(v_noise);

    if (injected_layer.rows == 0) return error.DFlashTargetFeatureBankIncomplete;
    const k_all = try request.draft_cb.concatRows2D(request.allocator, injected_layer.k_ctx, k_noise_normed, injected_layer.rows, rows, kv_dim);
    defer request.draft_cb.free(k_all);
    const v_all = try request.draft_cb.concatRows2D(request.allocator, injected_layer.v_ctx, v_noise, injected_layer.rows, rows, kv_dim);
    defer request.draft_cb.free(v_all);
    if (!hasDeviceTensor(request.draft_cb, k_all) or !hasDeviceTensor(request.draft_cb, v_all)) {
        return error.DFlashFeatureCaptureNotDeviceResident;
    }

    const rope_dim = head_dim;
    const q_position_offset = request.current_seq_len - 1;
    const k_position_offset = featureContextStart(request.feature_bank);
    const q_rope = try request.draft_cb.rope(q_normed, rows, head_dim, rope_dim, request.draft_config.rope_theta, request.draft_config.rope_freq_scale, q_position_offset, request.draft_config.rope_layout == .consecutive_pairs);
    defer request.draft_cb.free(q_rope);
    const k_rope = try request.draft_cb.rope(k_all, injected_layer.rows + rows, head_dim, rope_dim, request.draft_config.rope_theta, request.draft_config.rope_freq_scale, k_position_offset, request.draft_config.rope_layout == .consecutive_pairs);
    defer request.draft_cb.free(k_rope);

    const attn = (try request.draft_cb.gqaCrossAttentionFull(q_rope, k_rope, v_all, null, 1, rows, injected_layer.rows + rows, num_heads, num_kv_heads, head_dim)) orelse return error.DFlashCrossAttentionUnavailable;
    defer request.draft_cb.free(attn);
    if (!hasDeviceTensor(request.draft_cb, attn)) return error.DFlashFeatureCaptureNotDeviceResident;
    const attn_proj = try request.draft_cb.linearNoBias(attn, o_w, rows, q_dim, hidden_size);
    defer request.draft_cb.free(attn_proj);
    const attn_res = try request.draft_cb.add(hidden, attn_proj);
    defer request.draft_cb.free(attn_res);

    const ffn_normed = try request.draft_cb.rmsNorm(attn_res, post_norm_w, hidden_size, request.draft_config.norm_eps);
    defer request.draft_cb.free(ffn_normed);
    const gate = try request.draft_cb.linearNoBias(ffn_normed, gate_w, rows, hidden_size, intermediate_size);
    defer request.draft_cb.free(gate);
    const gate_act = try request.draft_cb.silu(gate);
    defer request.draft_cb.free(gate_act);
    const up = try request.draft_cb.linearNoBias(ffn_normed, up_w, rows, hidden_size, intermediate_size);
    defer request.draft_cb.free(up);
    const gated = try request.draft_cb.multiply(gate_act, up);
    defer request.draft_cb.free(gated);
    const down = try request.draft_cb.linearNoBias(gated, down_w, rows, intermediate_size, hidden_size);
    defer request.draft_cb.free(down);
    const out = try request.draft_cb.add(attn_res, down);
    errdefer request.draft_cb.free(out);
    if (!hasDeviceTensor(request.draft_cb, out)) return error.DFlashFeatureCaptureNotDeviceResident;
    return out;
}

fn selectDraftTokenIdsDevice(request: DeviceDraftRequest, hidden: ops.CT) !void {
    const lm_w = try loadTargetLmHeadWeight(request);
    defer request.target_cb.free(lm_w);
    const hidden_size: usize = @intCast(request.draft_config.hidden_size);
    const vocab_size: usize = @intCast(request.target_config.vocab_size);
    var idx: usize = 0;
    while (idx < request.block_size) : (idx += 1) {
        const row = idx + 1;
        const row_hidden = try request.draft_cb.sliceRows2D(request.allocator, hidden, row, 1, hidden_size);
        defer request.draft_cb.free(row_hidden);
        const token_tensor = (try request.draft_cb.linearNoBiasArgmaxLastRowTensor(row_hidden, lm_w, 1, hidden_size, vocab_size)) orelse return error.DFlashDeviceRowArgmaxUnavailable;
        defer request.draft_cb.free(token_tensor);
        request.output_tokens[idx] = try readScalarTokenIdDevice(request.draft_cb, request.allocator, token_tensor);
    }
}

fn effectiveDraftLayerCount(draft: gpt.Config) usize {
    if (draft.dflash_draft_layer_count > 0) return @intCast(draft.dflash_draft_layer_count);
    return @intCast(draft.num_hidden_layers);
}

fn dflashKvProjectionDim(draft: gpt.Config) usize {
    return @as(usize, @intCast(draft.effectiveKVHeads())) * @as(usize, @intCast(draft.headDim()));
}

fn loadDraftLayerWeight(request: DeviceDraftRequest, layer_index: usize, suffix: []const u8, comptime missing_error: anyerror) !ops.CT {
    var name_buf: [192]u8 = undefined;
    if (try maybeGetWeight(request.draft_cb, try std.fmt.bufPrint(&name_buf, "layers.{d}.{s}", .{ layer_index, suffix }))) |weight| return weight;
    if (try maybeGetWeight(request.draft_cb, try std.fmt.bufPrint(&name_buf, "model.layers.{d}.{s}", .{ layer_index, suffix }))) |weight| return weight;
    if (try maybeGetWeight(request.draft_cb, try std.fmt.bufPrint(&name_buf, "dflash.layers.{d}.{s}", .{ layer_index, suffix }))) |weight| return weight;
    return missing_error;
}

fn loadDraftRootWeight(request: DeviceDraftRequest, name: []const u8, comptime missing_error: anyerror) !ops.CT {
    if (try maybeGetWeight(request.draft_cb, name)) |weight| return weight;
    var name_buf: [192]u8 = undefined;
    if (try maybeGetWeight(request.draft_cb, try std.fmt.bufPrint(&name_buf, "model.{s}", .{name}))) |weight| return weight;
    if (try maybeGetWeight(request.draft_cb, try std.fmt.bufPrint(&name_buf, "dflash.{s}", .{name}))) |weight| return weight;
    return missing_error;
}

fn loadTargetLmHeadWeight(request: DeviceDraftRequest) !ops.CT {
    if (!request.target_config.weight_tying) {
        if (try maybeGetWeight(request.target_cb, "lm_head.weight")) |weight| return weight;
    }
    if (try maybeGetWeight(request.target_cb, "model.embed_tokens.weight")) |weight| return weight;
    if (try maybeGetWeight(request.target_cb, "embed_tokens.weight")) |weight| return weight;
    if (try maybeGetWeight(request.target_cb, "transformer.wte.weight")) |weight| return weight;
    return error.DFlashDraftWeightMissing;
}

fn inferIntermediateSize(request: DeviceDraftRequest, layer_index: usize) !usize {
    const configured = request.draft_config.intermediateSize(layer_index);
    if (configured > 0) return @intCast(configured);
    return error.DFlashDraftWeightMissing;
}

fn copyDeviceTensor(cb: *const ops.ComputeBackend, tensor: ops.CT, rows: usize, cols: usize) !ops.CT {
    if (!hasDeviceTensor(cb, tensor)) return error.DFlashFeatureCaptureNotDeviceResident;
    const copy = (try cb.zeroTensor(rows, cols)) orelse return error.DFlashFeatureCaptureNotDeviceResident;
    errdefer cb.free(copy);
    if (!(try metal_compute.MetalCompute.copyTensorInto(cb, tensor, copy))) {
        return error.DFlashFeatureCaptureNotDeviceResident;
    }
    return copy;
}

fn featureContextStart(feature_bank: *const DFlashFeatureBank) usize {
    if (feature_bank.slots.len == 0) return 0;
    var start = feature_bank.slots[0].start_token;
    for (feature_bank.slots[1..]) |slot| start = @min(start, slot.start_token);
    return start;
}

fn maybeGetWeight(cb: *const ops.ComputeBackend, name: []const u8) !?ops.CT {
    return cb.getWeight(name) catch |err| switch (err) {
        error.MissingWeight => null,
        else => return err,
    };
}

fn loadFeatureFusionWeight(request: DeviceDraftRequest, layer_index: usize, slot_index: usize) !ops.CT {
    var name_buf: [160]u8 = undefined;
    if (try maybeGetFeatureFusionWeight(request, try std.fmt.bufPrint(&name_buf, "dflash.feature_fusion.layers.{d}.weight", .{layer_index}))) |weight| return weight;
    if (try maybeGetFeatureFusionWeight(request, try std.fmt.bufPrint(&name_buf, "dflash.feature_projection.layers.{d}.weight", .{layer_index}))) |weight| return weight;
    if (try maybeGetFeatureFusionWeight(request, try std.fmt.bufPrint(&name_buf, "feature_fusion.layers.{d}.weight", .{layer_index}))) |weight| return weight;
    if (try maybeGetFeatureFusionWeight(request, try std.fmt.bufPrint(&name_buf, "target_feature_projection.layers.{d}.weight", .{layer_index}))) |weight| return weight;
    if (try maybeGetFeatureFusionWeight(request, try std.fmt.bufPrint(&name_buf, "dflash.context_proj.layers.{d}.weight", .{layer_index}))) |weight| return weight;
    if (try maybeGetFeatureFusionWeight(request, try std.fmt.bufPrint(&name_buf, "dflash.feature_fusion.{d}.weight", .{slot_index}))) |weight| return weight;
    if (try maybeGetFeatureFusionWeight(request, try std.fmt.bufPrint(&name_buf, "dflash.feature_projection.{d}.weight", .{slot_index}))) |weight| return weight;
    if (try maybeGetFeatureFusionWeight(request, try std.fmt.bufPrint(&name_buf, "feature_fusion.{d}.weight", .{slot_index}))) |weight| return weight;
    const shared_names = [_][]const u8{
        "dflash.feature_fusion.weight",
        "dflash.feature_projection.weight",
        "dflash.target_feature_projection.weight",
        "dflash.context_proj.weight",
        "feature_fusion.weight",
        "feature_projection.weight",
        "target_feature_projection.weight",
    };
    for (shared_names) |name| {
        if (try maybeGetFeatureFusionWeight(request, name)) |weight| return weight;
    }
    return error.DFlashFusionWeightMissing;
}

fn maybeGetFeatureFusionWeight(request: DeviceDraftRequest, name: []const u8) !?ops.CT {
    return request.draft_cb.getWeight(name) catch |err| switch (err) {
        error.MissingWeight => null,
        else => return err,
    };
}

pub fn effectiveBlockSize(draft: gpt.Config, requested: u32) usize {
    if (requested > 0) return @min(@as(usize, @intCast(requested)), max_block_size);
    if (draft.dflash_block_size > 0) return @min(@as(usize, @intCast(draft.dflash_block_size)), max_block_size);
    return default_block_size;
}

pub fn effectiveFeatureBankCapacity(draft: gpt.Config) usize {
    if (draft.dflash_feature_bank_capacity > 0) return @min(@as(usize, @intCast(draft.dflash_feature_bank_capacity)), max_feature_bank_capacity);
    return default_feature_bank_capacity;
}

pub fn effectiveCrossContext(draft: gpt.Config) usize {
    const capacity = effectiveFeatureBankCapacity(draft);
    if (draft.dflash_cross_context > 0) return @min(@as(usize, @intCast(draft.dflash_cross_context)), capacity);
    return @min(default_cross_context, capacity);
}

pub fn maskTokenId(draft: gpt.Config) ?i64 {
    if (draft.dflash_mask_token_id >= 0) return @intCast(draft.dflash_mask_token_id);
    if (draft.pad_token_id >= 0) return @intCast(draft.pad_token_id);
    return null;
}

pub fn requiresFeatureFusion(draft: gpt.Config) bool {
    return draft.dflash_target_feature_layer_count > 0 or draft.dflash_target_hidden_size > 0;
}

pub fn fallbackPolicyFromEnv() ResidencyPolicy {
    return if (platform.env.getenvBool(debug_host_fallback_env)) .debug_host_fallback else .fail_closed;
}

pub fn isDeterministicSampling(config: SamplingConfig) bool {
    return config.temperature == 0 and
        config.top_p == 0 and
        config.top_k == 0 and
        config.min_p == 0 and
        config.repetition_penalty == 1.0 and
        config.frequency_penalty == 0 and
        config.presence_penalty == 0;
}

test "DFlash pair validation requires Gemma4 target and DFlash draft" {
    var target = gpt.Config{
        .family = .gemma,
        .hidden_size = 2048,
        .vocab_size = 256000,
    };
    target.num_kv_shared_layers = 2;
    var draft = gpt.Config{
        .family = .gemma,
        .hidden_size = 1024,
        .vocab_size = 256000,
        .dflash_draft = true,
        .dflash_target_hidden_size = 2048,
        .dflash_block_size = 16,
        .dflash_target_feature_layer_count = 1,
        .dflash_target_feature_layers = blk: {
            var layers = [_]u32{0} ** 16;
            layers[0] = 7;
            break :blk layers;
        },
        .dflash_feature_bank_capacity = 4096,
        .dflash_cross_context = 1024,
    };
    try validatePair(target, draft);
    try std.testing.expectEqual(@as(usize, 4096), effectiveFeatureBankCapacity(draft));
    try std.testing.expectEqual(@as(usize, 1024), effectiveCrossContext(draft));

    draft.dflash_target_hidden_size = 4096;
    try std.testing.expectError(error.DFlashTargetHiddenSizeMismatch, validatePair(target, draft));
}

test "DFlash device request is Metal-only and deterministic" {
    try std.testing.expect(isDeterministicSampling(.{}));
    try std.testing.expect(!isDeterministicSampling(.{ .temperature = 0.7 }));
    try std.testing.expect(!isDeterministicSampling(.{ .top_p = 0.9 }));
    try std.testing.expect(!isDeterministicSampling(.{ .top_k = 1 }));
    try std.testing.expect(!isDeterministicSampling(.{ .min_p = 0.1 }));
    try std.testing.expect(!isDeterministicSampling(.{ .repetition_penalty = 1.1 }));
    try std.testing.expect(!isDeterministicSampling(.{ .frequency_penalty = 0.5 }));
    try std.testing.expect(!isDeterministicSampling(.{ .presence_penalty = 0.5 }));
}

test "DFlash feature bank layer selection is metadata driven" {
    var target = gpt.Config{
        .family = .gemma,
        .hidden_size = 2048,
        .vocab_size = 256000,
    };
    target.num_kv_shared_layers = 2;
    const draft = gpt.Config{
        .family = .gemma,
        .hidden_size = 1024,
        .vocab_size = 256000,
        .dflash_draft = true,
        .dflash_target_hidden_size = 2048,
        .dflash_target_feature_layer_count = 3,
        .dflash_target_feature_layers = blk: {
            var layers = [_]u32{0} ** 16;
            layers[0] = 7;
            layers[1] = 15;
            layers[2] = 23;
            break :blk layers;
        },
    };
    var slots = [_]DFlashFeatureBank.Slot{
        .{ .layer_index = draft.dflash_target_feature_layers[0] },
        .{ .layer_index = draft.dflash_target_feature_layers[1] },
        .{ .layer_index = draft.dflash_target_feature_layers[2] },
    };
    const bank = DFlashFeatureBank{
        .allocator = std.testing.allocator,
        .cb = undefined,
        .hidden_size = target.hidden_size,
        .capacity = effectiveFeatureBankCapacity(draft),
        .cross_context = effectiveCrossContext(draft),
        .slots = slots[0..],
    };
    try std.testing.expect(bank.wantsLayer(7));
    try std.testing.expect(bank.wantsLayer(15));
    try std.testing.expect(!bank.wantsLayer(8));
    try std.testing.expectEqual(@as(usize, 0), bank.capturedLayerCount());
}
