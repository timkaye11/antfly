// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//! One-head adapter from the authenticated fitting source to AFATT002 export.
//! It never owns or materializes a complete checkpoint.
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const solver = @import("attention_match.zig");

pub const Context = struct {
    source: *engine.StreamingHeadSource,
    active: ?engine.StreamingHeadLease = null,

    pub fn provider(self: *@This()) checkpoint.StreamingHeadProvider {
        return .{ .context = self, .open_fn = open, .verify_empty_fn = verifyEmpty };
    }

    pub fn deinit(self: *@This()) void {
        if (self.active != null) @panic("source checkpoint adapter destroyed with live head");
        self.* = undefined;
    }

    fn open(raw: ?*anyopaque, donor: usize, head_index: usize, control: solver.Control) !checkpoint.StreamingHeadLease {
        const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidKvCheckpoint));
        if (self.active != null) return error.InvalidKvCheckpoint;
        self.active = try self.source.open(donor, head_index, control);
        return .{ .context = self, .close_fn = close, .donor = donor, .head_index = head_index, .head = self.active.?.head };
    }

    fn close(raw: ?*anyopaque, _: *checkpoint.StreamingHeadLease) void {
        const self: *@This() = @ptrCast(@alignCast(raw orelse @panic("missing source checkpoint adapter")));
        if (self.active) |*lease| lease.deinit() else @panic("source checkpoint adapter close without open");
        self.active = null;
    }

    fn verifyEmpty(raw: ?*anyopaque) !void {
        const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidKvCheckpoint));
        if (self.active != null) return error.IncompleteContextAllocationScope;
        try self.source.verify_no_live_fn(self.source.context);
    }
};

test "checkpoint adapter closes source lease on success cancellation and open failure" {
    const Fixture = struct {
        head: checkpoint.Head,
        live: bool = false,
        opens: usize = 0,
        closes: usize = 0,
        fail: bool = false,
        fn open(raw: ?*anyopaque, donor: usize, head_index: usize, control: solver.Control) !engine.StreamingHeadLease {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            try control.check();
            if (self.fail) return error.InjectedFailure;
            if (self.live or donor != 0 or head_index != 0) return error.InvalidKvCheckpoint;
            self.live = true;
            self.opens += 1;
            return .{ .context = self, .close_fn = close, .donor_index = donor, .head_index = head_index, .head = self.head };
        }
        fn close(raw: ?*anyopaque, _: *engine.StreamingHeadLease) void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.live = false;
            self.closes += 1;
        }
        fn verify(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            if (self.live) return error.IncompleteContextAllocationScope;
        }
    };
    var fixture: Fixture = .{ .head = .{ .dim = 1, .positions = &.{0}, .coverage_ends = &.{1}, .keys = &.{1}, .values = &.{2}, .bias = &.{0} } };
    var source: engine.StreamingHeadSource = .{ .metadata = undefined, .expected_structure_sha256 = @splat(1), .expected_state_sha256 = @splat(2), .context = &fixture, .open_fn = Fixture.open, .verify_no_live_fn = Fixture.verify, .external_live_bytes = 0 };
    var adapter: Context = .{ .source = &source };
    defer adapter.deinit();
    var lease = try adapter.provider().open(0, 0, .{});
    lease.deinit();
    try adapter.provider().verify_empty_fn(adapter.provider().context);
    try @import("std").testing.expectEqual(fixture.opens, fixture.closes);
    fixture.fail = true;
    try @import("std").testing.expectError(error.InjectedFailure, adapter.provider().open(0, 0, .{}));
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    try @import("std").testing.expectError(error.Cancelled, adapter.provider().open(0, 0, .{ .check_fn = Cancel.check }));
    try @import("std").testing.expect(!fixture.live);
}
