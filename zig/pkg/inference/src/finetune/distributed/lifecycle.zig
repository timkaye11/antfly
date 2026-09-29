// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
const std = @import("std");

pub const Boundary = struct {
    microbatch: u64,
    optimizer: u64,
    complete: bool,
    pause: bool,
    checkpoint: bool,
};

/// Every rank enters this once at the same safe training boundary. A pause or
/// checkpoint requested on either host applies to both, including after resume.
pub fn agreeBoundary(collective: anytype, local: Boundary) !Boundary {
    var header: [19]u8 = undefined;
    std.mem.writeInt(u64, header[0..8], local.microbatch, .little);
    std.mem.writeInt(u64, header[8..16], local.optimizer, .little);
    header[16] = @intFromBool(local.complete);
    header[17] = @intFromBool(local.pause);
    header[18] = @intFromBool(local.checkpoint);
    var gathered: [38]u8 = undefined;
    try collective.allGatherBytes(&header, &gathered);
    if (!std.mem.eql(u8, gathered[0..17], gathered[19..36])) return error.DistributedBoundaryMismatch;
    for ([_]usize{ 16, 17, 18, 35, 36, 37 }) |i| if (gathered[i] > 1) return error.InvalidDistributedBoundary;
    var result = local;
    result.pause = !local.complete and (gathered[17] != 0 or gathered[36] != 0);
    result.checkpoint = result.pause or gathered[18] != 0 or gathered[37] != 0;
    return result;
}

fn acknowledge(collective: anytype, phase: u8, local_error: ?anyerror) !void {
    const header = [_]u8{ phase, @intFromBool(local_error == null) };
    var gathered: [4]u8 = undefined;
    try collective.allGatherBytes(&header, &gathered);
    if (gathered[0] != phase or gathered[2] != phase) return error.DistributedCheckpointPhaseMismatch;
    if (local_error) |err| return err;
    if (gathered[1] != 1 or gathered[3] != 1) return error.DistributedPeerCheckpointFailed;
}

/// The writer must use a fresh generation filename and retain older snapshots
/// and receipts. Receipt publication only starts after BOTH snapshots are
/// durable. A partial receipt publication never destroys the last common pair.
pub fn publishCheckpoint(collective: anytype, writer: anytype) !void {
    var failure: ?anyerror = null;
    writer.writeSnapshot() catch |err| {
        failure = err;
    };
    try acknowledge(collective, 1, failure);
    writer.writeReceipt() catch |err| {
        failure = err;
    };
    try acknowledge(collective, 2, failure);
}
