const std = @import("std");

pub const Phase = enum { request_dispatch, source_prefill, compiled_handoff, query_collection, fit_validation, export_prepare, @"export", upload, install, suffix_prefill, first_token };
pub const Edge = enum { begin, end };
pub const Event = struct { phase: Phase, edge: Edge, monotonic_ns: u64 };

/// Qualification-only process-local observer. The default runtime path passes
/// null. The caller owns storage and serialization; emission allocates nothing.
pub const Observer = struct {
    context: *anyopaque,
    emitFn: *const fn (*anyopaque, Event) anyerror!void,
    pub fn emit(self: Observer, phase: Phase, edge: Edge, now_ns: u64) !void {
        if (now_ns == 0) return error.InvalidPhaseClock;
        try self.emitFn(self.context, .{ .phase = phase, .edge = edge, .monotonic_ns = now_ns });
    }
};

pub const maximum_events: usize = 2 * @typeInfo(Phase).@"enum".field_names.len;
pub const Ledger = struct {
    events: [maximum_events]Event = undefined,
    count: usize = 0,
    open: ?Phase = null,
    terminal: bool = false,
    pub fn observer(self: *Ledger) Observer {
        return .{ .context = self, .emitFn = accept };
    }
    fn accept(raw: *anyopaque, event: Event) !void {
        const self: *Ledger = @ptrCast(@alignCast(raw));
        if (self.terminal or self.count == maximum_events) return error.PhaseLedgerClosed;
        if (self.count != 0) {
            const previous = self.events[self.count - 1];
            if (event.monotonic_ns < previous.monotonic_ns or
                (event.monotonic_ns == previous.monotonic_ns and !(previous.edge == .end and event.edge == .begin)))
                return error.InvalidPhaseClock;
        }
        switch (event.edge) {
            .begin => {
                if (self.open != null) return error.PhaseAlreadyOpen;
                self.open = event.phase;
            },
            .end => {
                if (self.open != event.phase) return error.PhaseClosureMismatch;
                self.open = null;
            },
        }
        self.events[self.count] = event;
        self.count += 1;
        if (event.phase == .first_token and event.edge == .end) self.terminal = true;
    }
    pub fn slice(self: *const Ledger) []const Event {
        return self.events[0..self.count];
    }
    pub fn validateClosed(self: *const Ledger) !void {
        if (!self.terminal or self.open != null or self.count == 0) return error.IncompletePhaseLedger;
    }
};

/// Stable-address request owner. NativeCheckpointPrepare owns this value and
/// destroys it from its existing deinit path after response export or error.
pub const Owned = struct {
    allocator: std.mem.Allocator,
    ledger: *Ledger,

    pub fn init(allocator: std.mem.Allocator) !Owned {
        const ledger = try allocator.create(Ledger);
        ledger.* = .{};
        return .{ .allocator = allocator, .ledger = ledger };
    }
    pub fn deinit(self: *Owned) void {
        self.allocator.destroy(self.ledger);
        self.* = undefined;
    }
    pub fn observer(self: *Owned) Observer {
        return self.ledger.observer();
    }
    pub fn events(self: *const Owned) []const Event {
        return self.ledger.slice();
    }
};

test "bounded observer preserves monotonic begin end closure" {
    var ledger = Ledger{};
    const o = ledger.observer();
    try o.emit(.source_prefill, .begin, 1);
    try o.emit(.source_prefill, .end, 2);
    try o.emit(.first_token, .begin, 3);
    try o.emit(.first_token, .end, 4);
    try ledger.validateClosed();
    try std.testing.expectEqual(@as(usize, 4), ledger.slice().len);
}
test "observer rejects overlap cancellation and post-terminal writes" {
    var ledger = Ledger{};
    const o = ledger.observer();
    try o.emit(.query_collection, .begin, 1);
    try std.testing.expectError(error.PhaseAlreadyOpen, o.emit(.fit_validation, .begin, 2));
    try std.testing.expectError(error.IncompletePhaseLedger, ledger.validateClosed());
    try o.emit(.query_collection, .end, 3);
    try o.emit(.first_token, .begin, 4);
    try o.emit(.first_token, .end, 5);
    try std.testing.expectError(error.PhaseLedgerClosed, o.emit(.@"export", .begin, 6));
}

test "owned ledger keeps stable observer context and has one destruction owner" {
    var owned = try Owned.init(std.testing.allocator);
    defer owned.deinit();
    const observer = owned.observer();
    try observer.emit(.source_prefill, .begin, 1);
    // Simulate cancellation: an incomplete ledger is evidence failure, while
    // request teardown still owns and frees the allocation.
    try std.testing.expectError(error.IncompletePhaseLedger, owned.ledger.validateClosed());
    try observer.emit(.source_prefill, .end, 2);
    try observer.emit(.first_token, .begin, 3);
    try observer.emit(.first_token, .end, 4);
    try owned.ledger.validateClosed();
}
