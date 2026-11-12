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

//! Leased research reader (encoder-LoRA candidate, docs/plans/memorilla-copy17-native.md
//! section 5.3): the exported reader weights on the device plus the host F32
//! copy head. Mirrors `memorilla_runtime.zig`; nothing here is promoted to a
//! qualified release and no release pin is relaxed.
const std = @import("std");
const build_options = @import("build_options");
const module = @import("memorilla.zig");
const memorilla_copy = @import("../pipelines/memorilla_copy.zig");
const ops = @import("../ops/ops.zig");
const memory = @import("../runtime/root.zig").tier.memory;
const Control = @import("../execution_control.zig").InferenceExecutionControl;
const cuda = if (build_options.enable_cuda) @import("../ops/cuda/cuda_compute.zig") else struct {};
const A = std.mem.Allocator;
const this_module = @This();

/// Device-resident F32 reader weights (same geometry as the release).
pub const parameter_bytes: usize = 776_495_104;
/// Host-resident F32 copy head: `query.weight [4096,4096]`, `gate.weight
/// [1,4096]`, `gate.bias [1]`.
pub const copy_head_bytes: usize = 67_125_252;
pub const copy_head_width: usize = 4096;
/// Qwen3-8B vocabulary rows whose L2 norms the service caches on the Owner.
pub const table_norms_max_vocab: usize = 151_936;
pub const table_norms_bytes: usize = table_norms_max_vocab * @sizeOf(f32);

pub fn weightPlan() memory.AdmissionAmounts {
    return .{
        .host_weight_bytes = 512 * 1024 * 1024 + parameter_bytes + copy_head_bytes + 16 * 1024 * 1024 + table_norms_bytes,
        .backend_weight_bytes = parameter_bytes,
        .backend_scratch_bytes = 128 * 1024 * 1024,
    };
}

pub fn requireLimits(limits: memory.Limits) !void {
    const plan = weightPlan();
    if (limits.host_limit_bytes == 0 or limits.backend_limit_bytes == 0 or limits.combined_limit_bytes == 0)
        return error.MemorillaFiniteBudgetRequired;
    if (plan.hostTotalBytes() > limits.host_limit_bytes or plan.backendTotalBytes() > limits.backend_limit_bytes or
        (try std.math.add(usize, plan.hostTotalBytes(), plan.backendTotalBytes())) > limits.combined_limit_bytes) return error.ResourceLimitExceeded;
}

pub fn requireAdmission(admission: *const memory.AdmissionLease, limits: memory.Limits) !void {
    try requireLimits(limits);
    const plan = weightPlan();
    if (admission.controller == null or admission.retain_backend_class != .gpu or
        admission.amounts.host_weight_bytes < plan.host_weight_bytes or
        admission.amounts.backend_weight_bytes < plan.backend_weight_bytes or
        admission.amounts.backend_scratch_bytes < plan.backend_scratch_bytes)
        return error.MemorillaReaderAdmissionRequired;
}

/// A research bundle is explicitly non-release with per-token features and
/// the released reader geometry; the pinned research files are verified by
/// `Bundle.open`.
fn requireResearch(bundle: *const module.Bundle) !void {
    if (bundle.release or !bundle.perTokenPooling()) return error.UnsupportedMemorillaProfile;
    if (bundle.candidate_sha256 == null or bundle.protocol == null) return error.UnsupportedMemorillaProfile;
    bundle.config.validate(true) catch return error.UnsupportedMemorillaProfile;
}

pub const Owner = struct {
    pub const runtime = this_module;
    /// Every pinned research file; a drift in any of them retires the reader.
    pub const identity_files = [_][]const u8{ "manifest.json", "config.json", "memory.safetensors", module.research_copy_head_file, module.research_candidate_file, module.research_protocol_file };

    allocator: A,
    bundle: module.Bundle,
    compute: if (build_options.enable_cuda) *cuda.CudaCompute else void,
    /// Host F32 copy head with the protocol temperature applied.
    copy_head: memorilla_copy.CopyHead,
    /// sha256 of the bundle's `candidate.json`; the encoder lock must match.
    candidate_sha256: [64]u8,
    /// Borrowed from `bundle.protocol`; valid for the Owner's lifetime.
    system_prompt: []const u8,
    copy_head_temperature: f32,
    /// Per-row L2 norms of the decoder embedding table, computed once by the
    /// service on first use (snap targets); owned by `allocator`.
    table_norms: ?[]f32 = null,
    table_norms_vocab: usize = 0,
    execution_mutex: std.atomic.Mutex = .unlocked,

    /// Caller owns the GPU admission lease and must release it AFTER deinit.
    pub fn open(a: A, io: std.Io, directory: []const u8, admission: *const memory.AdmissionLease, limits: memory.Limits, control: Control) !Owner {
        try control.check();
        try requireAdmission(admission, limits);
        try @import("memorilla_qualification.zig").requireCudaEnvironment();
        if (comptime !build_options.enable_cuda) return error.CudaNotEnabled;
        const backing = @import("antfly_platform").allocator.reclaimingScratch(a);
        var guard = try control.enterUninterruptible(.uninterruptible);
        defer guard.deinit();
        var bundle = try module.Bundle.open(backing, directory, true);
        errdefer bundle.deinit();
        if (bundle.reader.mmap_region) |*mapping| mapping.discard_on_deinit = false;
        try requireResearch(&bundle);
        try control.check();
        var copy_head = try memorilla_copy.CopyHead.load(backing, directory, copy_head_width);
        errdefer copy_head.deinit();
        const protocol = bundle.protocol.?;
        copy_head.temperature = protocol.copy_head_temperature;
        try control.check();
        var store = try bundle.loadFloat32Store();
        defer store.deinitOwned();
        try control.check();
        const compute = try backing.create(cuda.CudaCompute);
        errdefer backing.destroy(compute);
        compute.* = try cuda.CudaCompute.init(backing);
        errdefer compute.deinit();
        if (compute.ctx.info.compute_major != 12 or compute.ctx.info.compute_minor != 1) return error.UnsupportedMemorillaDevice;
        compute.strict_f32_weights = true;
        compute.capture_gates = .{};
        try compute.resident_weights.ensureUnusedCapacity(backing, store.resident_weights.count());
        var iter = store.resident_weights.iterator();
        while (iter.next()) |entry| {
            try control.check();
            const key = try backing.dupe(u8, entry.key_ptr.*);
            errdefer backing.free(key);
            try compute.insertWeightFromLoaded(key, entry.value_ptr);
        }
        _ = compute.releaseMemorillaRequestResources() catch @panic("Memorilla research reader load cleanup failed; worker restart required");
        if (compute.snapshotStats().resident_weight_bytes != parameter_bytes) return error.UnsupportedMemorillaPrecision;
        _ = io;
        try control.check();
        return .{
            .allocator = backing,
            .bundle = bundle,
            .compute = compute,
            .copy_head = copy_head,
            .candidate_sha256 = bundle.candidate_sha256.?,
            .system_prompt = protocol.system_prompt,
            .copy_head_temperature = protocol.copy_head_temperature,
        };
    }

    /// Takes ownership of `norms` (allocated with `self.allocator`), replacing
    /// any earlier table. Call under `execution_mutex`.
    pub fn setTableNorms(self: *Owner, norms: []f32) void {
        if (self.table_norms) |previous| self.allocator.free(previous);
        self.table_norms = norms;
        self.table_norms_vocab = norms.len;
    }

    /// Borrowed backend. Caller serializes it with execution_mutex, propagates
    /// control, and frees every request CT before releaseRequestResources.
    pub fn computeBackend(self: *Owner) !ops.ComputeBackend {
        if (comptime build_options.enable_cuda) return self.compute.computeBackend();
        return error.CudaNotEnabled;
    }

    pub fn computeBackendWithBudget(self: *Owner, budget: *memory.RunBudget) !ops.ComputeBackend {
        if (comptime build_options.enable_cuda) return self.compute.computeBackendWithStepRunBudget(budget, @intFromPtr(self));
        return error.CudaNotEnabled;
    }

    pub fn releaseRequestResources(self: *Owner) !void {
        if (comptime build_options.enable_cuda) {
            _ = try self.compute.releaseMemorillaRequestResources();
        } else return error.CudaNotEnabled;
    }

    pub fn deinit(self: *Owner) void {
        if (self.table_norms) |norms| self.allocator.free(norms);
        if (comptime build_options.enable_cuda) {
            self.compute.deinit();
            self.allocator.destroy(self.compute);
        }
        self.copy_head.deinit();
        self.bundle.deinit();
        self.* = undefined;
    }
};

test "Memorilla research reader plans device reader plus host copy head and requires admission before IO" {
    const plan = weightPlan();
    try std.testing.expectEqual(parameter_bytes, plan.backend_weight_bytes);
    try std.testing.expect(plan.host_weight_bytes >= parameter_bytes + copy_head_bytes + table_norms_bytes);
    try std.testing.expectEqual(@as(usize, 4096 * 4096 * 4 + 4096 * 4 + 4), copy_head_bytes);
    try std.testing.expectEqual(@as(usize, 843_620_356), parameter_bytes + copy_head_bytes);
    try std.testing.expectEqualStrings("copy-head.safetensors", Owner.identity_files[3]);
    try std.testing.expectEqual(@as(usize, 6), Owner.identity_files.len);
    const released = memory.AdmissionLease{ .controller = null, .amounts = .{}, .amounts_by_backend = @splat(.{}), .retain_backend_class = null, .live_reserved_bytes = 0 };
    try std.testing.expectError(error.MemorillaFiniteBudgetRequired, requireAdmission(&released, .{}));
    const limits = memory.Limits{ .host_limit_bytes = 4 * 1024 * 1024 * 1024, .backend_limit_bytes = 4 * 1024 * 1024 * 1024, .combined_limit_bytes = 6 * 1024 * 1024 * 1024 };
    try std.testing.expectError(error.MemorillaReaderAdmissionRequired, Owner.open(std.testing.allocator, std.testing.io, "/not-opened", &released, limits, .{}));
}
