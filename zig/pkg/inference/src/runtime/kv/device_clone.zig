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

//! Admission contract for an independently owned exact device prefix. The first
//! adapter supports identity page tables only; arbitrary mappings must never be
//! mistaken for contiguous physical rows.
const std = @import("std");

pub const Donor = struct {
    layer: u32,
    heads: u32,
    dim: u32,
    /// Physical rows required by this donor. Zero means request capacity.
    /// Split-SWA callers bind the exact ring capacity here.
    capacity_rows: u32 = 0,
    host_staged_ring: bool = false,
};
pub const LogicalBlockPolicy = enum {
    /// Generic/CUDA contract: logical page N must name physical block N.
    identity_only,
    /// Metal sequence-slot clone: block IDs prove manager ownership only;
    /// backend SlotBinding/SlotInfo metadata selects physical source buffers.
    metal_sequence_slot_owned,
};
pub const Request = struct {
    source_sequence: u32,
    destination_sequence: u32,
    tokens: usize,
    capacity: usize,
    page_size: u16,
    source_blocks: []const u32,
    destination_blocks: []const u32,
    logical_block_policy: LogicalBlockPolicy = .identity_only,
    donors: []const Donor,
    max_bytes: usize,
    /// Maximum temporary host K+V staging for bounded ring donors. Zero
    /// permits only device-to-device full-history donors.
    max_host_staging_bytes: usize = 0,
    check_ctx: ?*anyopaque = null,
    check_fn: ?*const fn (?*anyopaque) anyerror!void = null,

    pub fn check(self: Request) !void {
        if (self.check_fn) |check_fn| try check_fn(self.check_ctx);
    }

    /// Conservative allocation bound, including a private full-capacity page
    /// table for each donor even when the backend uses contiguous f32 rows.
    pub fn allocationBytes(self: Request, element_bytes: usize) !usize {
        if (self.tokens == 0 or self.capacity < self.tokens or self.page_size == 0 or
            self.capacity % self.page_size != 0 or self.donors.len == 0 or self.donors.len > 256 or
            (element_bytes != 2 and element_bytes != 4)) return error.InvalidDeviceKvClone;
        const pages = std.math.divCeil(usize, self.tokens, self.page_size) catch return error.InvalidDeviceKvClone;
        if (self.source_blocks.len < pages or self.destination_blocks.len < pages) return error.InvalidDeviceKvClone;
        switch (self.logical_block_policy) {
            .identity_only => for (self.source_blocks[0..pages], self.destination_blocks[0..pages], 0..) |src, dst, i| {
                if (src != i or dst != i) return error.UnsupportedDeviceKvCloneLayout;
            },
            .metal_sequence_slot_owned => {
                // Ownership comes from the two authenticated manager views.
                // Require a one-to-one logical-page map without allocating or
                // pretending that physical IDs are identity ordered.
                for (self.source_blocks[0..pages], 0..) |src, i| {
                    for (self.source_blocks[0..i]) |prior| if (prior == src)
                        return error.UnsupportedDeviceKvCloneLayout;
                }
                for (self.destination_blocks[0..pages], 0..) |dst, i| {
                    for (self.destination_blocks[0..i]) |prior| if (prior == dst)
                        return error.UnsupportedDeviceKvCloneLayout;
                }
            },
        }
        var bytes: usize = 0;
        for (self.donors, 0..) |donor, i| {
            if (donor.heads == 0 or donor.heads > 256 or donor.dim == 0 or donor.dim > 4096 or
                (i > 0 and donor.layer <= self.donors[i - 1].layer)) return error.InvalidDeviceKvClone;
            const donor_capacity: usize = if (donor.capacity_rows == 0) self.capacity else donor.capacity_rows;
            if (donor_capacity == 0 or donor_capacity > self.capacity or donor_capacity % self.page_size != 0)
                return error.InvalidDeviceKvClone;
            const row = try std.math.mul(usize, try std.math.mul(usize, donor.heads, donor.dim), element_bytes * 2);
            bytes = try std.math.add(usize, bytes, try std.math.mul(usize, donor_capacity, row));
            bytes = try std.math.add(usize, bytes, try std.math.mul(usize, donor_capacity / self.page_size, @sizeOf(u32)));
        }
        if (bytes > self.max_bytes) return error.DeviceKvCloneBudgetExceeded;
        return bytes;
    }

    /// Computes and enforces the peak per-donor ring staging requirement.
    /// This is called before any destination allocation or backend submission.
    pub fn hostStagingBytes(self: Request, element_bytes: usize) !usize {
        if (element_bytes != 2 and element_bytes != 4) return error.InvalidDeviceKvClone;
        var peak: usize = 0;
        for (self.donors) |donor| {
            if (!donor.host_staged_ring) continue;
            if (donor.capacity_rows == 0) return error.InvalidDeviceKvClone;
            const row = try std.math.mul(usize, try std.math.mul(usize, donor.heads, donor.dim), element_bytes * 2);
            peak = @max(peak, try std.math.mul(usize, donor.capacity_rows, row));
        }
        if (peak > self.max_host_staging_bytes) return error.DeviceKvCloneHostStagingBudgetExceeded;
        return peak;
    }
};

pub const Result = struct { allocation_bound_bytes: usize, copied_bytes: usize, peak_host_staging_bytes: usize = 0, donors: usize };

test "device clone validates mappings donors and admission before allocation" {
    const blocks = [_]u32{ 0, 1 };
    var request = Request{ .source_sequence = 7, .destination_sequence = 0, .tokens = 17, .capacity = 48, .page_size = 16, .source_blocks = &blocks, .destination_blocks = &blocks, .donors = &.{.{ .layer = 5, .heads = 2, .dim = 512 }}, .max_bytes = 200000 };
    try std.testing.expectEqual(@as(usize, 196620), try request.allocationBytes(2));
    try std.testing.expectError(error.DeviceKvCloneBudgetExceeded, request.allocationBytes(4));
    request.source_blocks = &.{ 1, 0 };
    try std.testing.expectError(error.UnsupportedDeviceKvCloneLayout, request.allocationBytes(2));
    request.source_blocks = &blocks;
    request.destination_blocks = &.{0};
    try std.testing.expectError(error.InvalidDeviceKvClone, request.allocationBytes(2));
    request.destination_blocks = &blocks;
    request.donors = &.{ .{ .layer = 5, .heads = 2, .dim = 512 }, .{ .layer = 5, .heads = 2, .dim = 512 } };
    try std.testing.expectError(error.InvalidDeviceKvClone, request.allocationBytes(2));
    request.donors = &.{.{ .layer = 5, .heads = 2, .dim = 512 }};
    request.capacity = std.math.maxInt(usize) - 15;
    try std.testing.expectError(error.Overflow, request.allocationBytes(2));
}

test "device clone rejects ring host staging before backend allocation" {
    const blocks = [_]u32{0};
    var request = Request{ .source_sequence = 7, .destination_sequence = 1, .tokens = 16, .capacity = 16, .page_size = 16, .source_blocks = &blocks, .destination_blocks = &blocks, .donors = &.{.{ .layer = 5, .heads = 2, .dim = 8, .capacity_rows = 16, .host_staged_ring = true }}, .max_bytes = 4096, .max_host_staging_bytes = 1023 };
    try std.testing.expectError(error.DeviceKvCloneHostStagingBudgetExceeded, request.hostStagingBytes(2));
    request.max_host_staging_bytes = 1024;
    try std.testing.expectEqual(@as(usize, 1024), try request.hostStagingBytes(2));
}

test "Metal sequence-slot policy admits owned nonidentity maps only" {
    const donors = [_]Donor{.{ .layer = 5, .heads = 2, .dim = 8 }};
    var request: Request = .{
        .source_sequence = 7,
        .destination_sequence = 9,
        .tokens = 32,
        .capacity = 32,
        .page_size = 16,
        .source_blocks = &.{ 2, 1 },
        .destination_blocks = &.{ 8, 7 },
        .donors = &donors,
        .max_bytes = 4096,
    };
    try std.testing.expectError(error.UnsupportedDeviceKvCloneLayout, request.allocationBytes(2));
    request.logical_block_policy = .metal_sequence_slot_owned;
    try std.testing.expectEqual(@as(usize, 2056), try request.allocationBytes(2));
    request.source_blocks = &.{ 2, 2 };
    try std.testing.expectError(error.UnsupportedDeviceKvCloneLayout, request.allocationBytes(2));
    request.source_blocks = &.{ 2, 1 };
    request.destination_blocks = &.{7};
    try std.testing.expectError(error.InvalidDeviceKvClone, request.allocationBytes(2));
    request.destination_blocks = &.{ 8, 7 };
    request.max_bytes = 2055;
    try std.testing.expectError(error.DeviceKvCloneBudgetExceeded, request.allocationBytes(2));
    request.capacity = std.math.maxInt(usize) - 15;
    try std.testing.expectError(error.Overflow, request.allocationBytes(2));
}

test "device clone admits 32K E4B f16 capacity and rejects f32 under the same bound" {
    var blocks: [2048]u32 = undefined;
    for (&blocks, 0..) |*block, i| block.* = @intCast(i);
    var donors: [24]Donor = undefined;
    for (&donors, 0..) |*donor, i| donor.* = .{
        .layer = @intCast(i),
        .heads = 2,
        .dim = if (i % 6 == 5) 512 else 256,
    };
    const request = Request{
        .source_sequence = 0,
        .destination_sequence = 0,
        .tokens = 32768,
        .capacity = 65536,
        .page_size = 16,
        .source_blocks = &blocks,
        .destination_blocks = &blocks,
        .donors = &donors,
        .max_bytes = 4 * 1024 * 1024 * 1024,
    };
    try std.testing.expectEqual(@as(usize, 3758489600), try request.allocationBytes(2));
    try std.testing.expectError(error.DeviceKvCloneBudgetExceeded, request.allocationBytes(4));
}
