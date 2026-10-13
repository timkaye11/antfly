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

//! Single derivation of the process envelope, paged-cache budget and context
//! limit for the local Qwen runners, from physical memory and Metal's
//! recommended working set. Explicit requests are validated and reported;
//! they are never clamped to fit.
const std = @import("std");
const options = @import("build_options");
const memory = @import("tier/memory.zig");
const mr = @import("../backends/metal_runtime.zig");

pub const MiB: usize = 1024 * 1024;
/// Longest sequence the attention and state kernels are qualified for. State
/// buffers and token histories are sized from the admitted context, so this
/// is the only host-side bound a longer-context kernel has to raise.
pub const max_context: usize = 4096;
/// Capacity the admission controller's emergency reserve needs inside an
/// envelope.
pub const admission_reserve_bytes: usize = 512 * MiB;
/// Footprint a process already holds when it asks for admission (image,
/// parsed metadata, allocator slack). The controller charges it against the
/// envelope in addition to the plan, so a derived envelope must leave room.
pub const process_baseline_bytes: usize = 256 * MiB;
/// Everything a derived envelope holds beyond the plan itself.
pub const derived_headroom_bytes: usize = admission_reserve_bytes + process_baseline_bytes;

pub const Source = enum { derived, explicit };
pub const Envelope = struct { cap_bytes: usize, source: Source };
pub const Error = error{
    HostMemoryUnavailable,
    ProcessCapExceedsPhysicalMemory,
    MemoryBudgetExceeded,
    MetalWorkingSetExceeded,
    MetalBufferTooLarge,
};

pub const Host = struct {
    physical_bytes: usize,
    /// Zero when no unified-memory Metal device was queried (CPU-only builds
    /// and dry runs). Execution still requires one.
    metal_working_set_bytes: usize = 0,
    metal_max_buffer_bytes: usize = 0,

    pub fn detect() Error!Host {
        const info = memory.currentSystemMemoryInfo() orelse return error.HostMemoryUnavailable;
        var host: Host = .{ .physical_bytes = info.total_bytes };
        if (comptime options.enable_metal) {
            var device: mr.MetalDeviceInfo = .{};
            if (mr.termite_metal_device_info_get(&device) == 0 and device.has_unified_memory != 0) {
                host.metal_working_set_bytes = std.math.cast(usize, device.recommended_max_working_set_size) orelse std.math.maxInt(usize);
                host.metal_max_buffer_bytes = std.math.cast(usize, device.max_buffer_length) orelse std.math.maxInt(usize);
            }
        }
        return host;
    }

    /// Physical memory less the admission controller's stable node headroom.
    /// No derived envelope exceeds this; an explicit one may, up to
    /// physical memory.
    pub fn machineCapBytes(self: Host) usize {
        return self.physical_bytes -| memory.liveHostMemoryHeadroom(self.physical_bytes);
    }
};

/// Resolve the process envelope for a finished plan. `required_bytes` is the
/// plan's host plus backend admission. A derived envelope is the plan plus
/// `derived_headroom_bytes`: a tight bound keeps the controller on its
/// explicit-envelope policy instead of reserving the node headroom twice.
pub fn envelope(host: Host, requested_cap_bytes: usize, required_bytes: usize, explicit_reserve_bytes: usize) Error!Envelope {
    if (requested_cap_bytes != 0) {
        if (requested_cap_bytes > host.physical_bytes) return error.ProcessCapExceedsPhysicalMemory;
        const needed = std.math.add(usize, required_bytes, explicit_reserve_bytes) catch return error.MemoryBudgetExceeded;
        if (needed > requested_cap_bytes) return error.MemoryBudgetExceeded;
        return .{ .cap_bytes = requested_cap_bytes, .source = .explicit };
    }
    const cap = std.math.add(usize, required_bytes, derived_headroom_bytes) catch return error.MemoryBudgetExceeded;
    if (cap > host.machineCapBytes()) return error.MemoryBudgetExceeded;
    return .{ .cap_bytes = cap, .source = .derived };
}

/// Metal's recommended working set bounds GPU-visible bytes only, not the
/// process envelope.
pub fn checkBackend(host: Host, backend_bytes: usize, largest_buffer_bytes: usize) Error!void {
    if (host.metal_working_set_bytes != 0 and backend_bytes > host.metal_working_set_bytes) return error.MetalWorkingSetExceeded;
    if (host.metal_max_buffer_bytes != 0 and largest_buffer_bytes > host.metal_max_buffer_bytes) return error.MetalBufferTooLarge;
}

/// Largest paged-cache budget a plan may add to its fixed bytes: the envelope
/// limit less the derived headroom, also bounded by Metal's working set.
pub fn cacheBudget(host: Host, requested_cap_bytes: usize, fixed_total_bytes: usize, fixed_backend_bytes: usize) Error!usize {
    if (requested_cap_bytes > host.physical_bytes) return error.ProcessCapExceedsPhysicalMemory;
    const limit = if (requested_cap_bytes != 0) requested_cap_bytes else host.machineCapBytes();
    var budget = limit -| fixed_total_bytes -| derived_headroom_bytes;
    if (host.metal_working_set_bytes != 0) budget = @min(budget, host.metal_working_set_bytes -| fixed_backend_bytes);
    return budget;
}

/// Host and envelope facts recorded in every receipt.
pub const Report = struct {
    host_physical_bytes: usize,
    machine_cap_bytes: usize,
    metal_recommended_working_set_bytes: usize,
    metal_max_buffer_bytes: usize,
    process_cap_bytes: usize,
    process_cap_source: []const u8,
    max_context: usize = max_context,
};
pub fn report(host: Host, resolved: Envelope) Report {
    return .{
        .host_physical_bytes = host.physical_bytes,
        .machine_cap_bytes = host.machineCapBytes(),
        .metal_recommended_working_set_bytes = host.metal_working_set_bytes,
        .metal_max_buffer_bytes = host.metal_max_buffer_bytes,
        .process_cap_bytes = resolved.cap_bytes,
        .process_cap_source = @tagName(resolved.source),
    };
}

const GiB = 1024 * MiB;
test "Qwen35 Qwen4Exp machine profile derives envelopes from physical memory" {
    for ([_][2]usize{ .{ 16, 10 }, .{ 24, 18 }, .{ 48, 36 }, .{ 64, 48 }, .{ 128, 104 } }) |pair| {
        const host: Host = .{ .physical_bytes = pair[0] * GiB };
        try std.testing.expectEqual(pair[1] * GiB, host.machineCapBytes());
    }
    const air: Host = .{ .physical_bytes = 16 * GiB, .metal_working_set_bytes = 11453251584 };
    // The streamed 27B plan keeps its historical sub-4 GiB envelope.
    const streamed: usize = 3478392832;
    const derived = try envelope(air, 0, streamed, admission_reserve_bytes);
    try std.testing.expectEqual(Source.derived, derived.source);
    try std.testing.expectEqual(streamed + derived_headroom_bytes, derived.cap_bytes);
    try std.testing.expect(derived.cap_bytes < 4 * GiB);
    const explicit = try envelope(air, 4 * GiB, streamed, admission_reserve_bytes);
    try std.testing.expectEqual(Envelope{ .cap_bytes = 4 * GiB, .source = .explicit }, explicit);
    // An operator may exceed the derived machine cap, never physical memory.
    try std.testing.expectEqual(@as(usize, 12 * GiB), (try envelope(air, 12 * GiB, 11 * GiB, 0)).cap_bytes);
    try std.testing.expectError(error.ProcessCapExceedsPhysicalMemory, envelope(air, 17 * GiB, streamed, 0));
    try std.testing.expectError(error.MemoryBudgetExceeded, envelope(air, 2 * GiB, streamed, admission_reserve_bytes));
    try std.testing.expectError(error.MemoryBudgetExceeded, envelope(air, 0, 10 * GiB - derived_headroom_bytes + 1, 0));
    try std.testing.expectEqual(@as(usize, 10 * GiB), (try envelope(air, 0, 10 * GiB - derived_headroom_bytes, 0)).cap_bytes);
}

test "Qwen35 Qwen4Exp machine profile bounds paged caches by envelope and Metal working set" {
    const studio: Host = .{ .physical_bytes = 48 * GiB, .metal_working_set_bytes = 36 * GiB, .metal_max_buffer_bytes = 24 * GiB };
    const fixed_total = 6 * GiB;
    const fixed_backend = 5 * GiB;
    // 36 GiB machine cap: the envelope binds before the working set.
    try std.testing.expectEqual(36 * GiB - fixed_total - derived_headroom_bytes, try cacheBudget(studio, 0, fixed_total, fixed_backend));
    var tight = studio;
    tight.metal_working_set_bytes = 32 * GiB;
    try std.testing.expectEqual(32 * GiB - fixed_backend, try cacheBudget(tight, 0, fixed_total, fixed_backend));
    try std.testing.expectEqual(40 * GiB - fixed_total - derived_headroom_bytes, try cacheBudget(.{ .physical_bytes = 48 * GiB }, 40 * GiB, fixed_total, fixed_backend));
    try std.testing.expectEqual(@as(usize, 0), try cacheBudget(studio, 4 * GiB, fixed_total, fixed_backend));
    try std.testing.expectError(error.ProcessCapExceedsPhysicalMemory, cacheBudget(studio, 49 * GiB, fixed_total, fixed_backend));
    try checkBackend(studio, 36 * GiB, 24 * GiB);
    try std.testing.expectError(error.MetalWorkingSetExceeded, checkBackend(studio, 36 * GiB + 1, 1));
    try std.testing.expectError(error.MetalBufferTooLarge, checkBackend(studio, GiB, 24 * GiB + 1));
    // Without a queried device only the envelope applies.
    try checkBackend(.{ .physical_bytes = 48 * GiB }, 64 * GiB, 64 * GiB);
}
