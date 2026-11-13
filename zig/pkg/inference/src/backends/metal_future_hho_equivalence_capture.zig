const std = @import("std");
const MetalTensor = @import("metal_tensor.zig").MetalTensor;
const metal_runtime = @import("metal_runtime.zig");
const host_mod = @import("../runtime/kv/attention_match_future_hho_equivalence_capture.zig");
const capture_contract = @import("../graph/recent_prefill_query_capture.zig");

const Pending = struct { encoded: bool = false, donor: usize = 0, consumer: usize = 0 };

pub fn buildNativePlan(
    request: capture_contract.NativePlanRequest,
    capture_domain: *anyopaque,
    destination_buffer: *anyopaque,
    destination_base_offset: usize,
    consumer_index: usize,
) !capture_contract.NativePlan {
    if (request.first_absolute_position != host_mod.first_heldout_position or
        request.row_count != host_mod.sample_rows or
        request.num_heads != host_mod.query_heads or
        request.head_dim != host_mod.head_dim or
        request.source_rows != host_mod.continuation_rows or
        request.sample_first_row != host_mod.sample_first_row or
        request.sample_stride != host_mod.sample_stride)
        return error.InvalidFutureHhoEquivalenceCapture;
    const row_elements = try std.math.mul(usize, @as(usize, request.num_heads), @as(usize, request.head_dim));
    const row_bytes = try std.math.mul(usize, row_elements, @sizeOf(f32));
    const byte_count = try std.math.mul(usize, @as(usize, request.row_count), row_bytes);
    const source_byte_offset = try std.math.mul(usize, @as(usize, request.sample_first_row), row_bytes);
    const slot_offset = try std.math.mul(usize, consumer_index, byte_count);
    const destination_byte_offset = try std.math.add(usize, destination_base_offset, slot_offset);
    return .{
        .capture_domain = capture_domain,
        .destination_buffer = destination_buffer,
        .destination_byte_offset = destination_byte_offset,
        .byte_count = byte_count,
        .source_byte_offset = source_byte_offset,
        .sample_rows = request.row_count,
        .sample_stride = request.sample_stride,
        .source_row_bytes = row_bytes,
    };
}

pub const Owner = struct {
    allocator: std.mem.Allocator,
    runtime: *anyopaque,
    host: host_mod.Owner,
    shared: MetalTensor,
    pending: [host_mod.consumer_count]Pending = @splat(.{}),
    epoch: ?u64 = null,
    completed: bool = false,
    poisoned: bool = false,
    native_plan_calls: usize = 0,
    native_commit_calls: usize = 0,
    native_copy_bytes: usize = 0,

    pub fn sharedRequiredBytes() !usize {
        return std.math.mul(usize, host_mod.consumer_count * host_mod.sample_rows * host_mod.query_heads * host_mod.head_dim, @sizeOf(f32));
    }
    pub fn init(allocator: std.mem.Allocator, runtime: *anyopaque, authority: host_mod.Authority, topology: [host_mod.consumer_count]host_mod.Consumer, control: host_mod.Control) !Owner {
        var host = try host_mod.Owner.init(allocator, authority, topology, try host_mod.Owner.requiredBytes(), control);
        errdefer host.deinit();
        const elements = host_mod.consumer_count * host_mod.sample_rows * host_mod.query_heads * host_mod.head_dim;
        var shared = try MetalTensor.deviceAllocateFreshWithAllocator(allocator, runtime, elements * @sizeOf(f32), .shared, &.{@intCast(elements)});
        errdefer shared.deinit();
        return .{ .allocator = allocator, .runtime = runtime, .host = host, .shared = shared };
    }
    pub fn deinit(self: *Owner) void {
        self.shared.deinit();
        self.host.deinit();
        self.* = undefined;
    }
    fn raw(self: *const Owner) *metal_runtime.RawMetalDecodeRuntime {
        return @ptrCast(@alignCast(self.runtime));
    }
    fn index(self: *const Owner, donor: usize, consumer: usize) !usize {
        for (self.host.entries, 0..) |entry, i| if (entry.tuple.donor_layer == donor and entry.tuple.consumer_layer == consumer) return i;
        return error.UnexpectedFutureHhoEquivalenceConsumer;
    }
    fn accepts(ctx: *anyopaque, donor: u32, consumer: u32) bool {
        const self: *Owner = @ptrCast(@alignCast(ctx));
        _ = self.index(donor, consumer) catch return false;
        return true;
    }
    fn lastFrame(ctx: *anyopaque) u64 {
        const self: *Owner = @ptrCast(@alignCast(ctx));
        return self.epoch orelse 0;
    }
    fn unusedEncode(_: *anyopaque, _: capture_contract.EncodeRequest) anyerror!void {
        return error.FutureHhoEquivalenceRequiresNativeGather;
    }
    fn plan(ctx: *anyopaque, request: capture_contract.NativePlanRequest) anyerror!?capture_contract.NativePlan {
        const self: *Owner = @ptrCast(@alignCast(ctx));
        if (self.poisoned or self.completed) return error.FutureHhoEquivalenceCaptureSealed;
        const i = try self.index(request.donor_layer, request.consumer_layer);
        if (self.pending[i].encoded) return error.DuplicateFutureHhoEquivalenceConsumer;
        const native_plan = try buildNativePlan(
            request,
            self,
            self.shared.deviceHandle() orelse return error.MetalBufferAllocFailed,
            self.shared.deviceByteOffset(),
            i,
        );
        self.native_plan_calls += 1;
        return native_plan;
    }
    fn commit(ctx: *anyopaque, request: capture_contract.NativePlanRequest, epoch: u64) anyerror!void {
        const self: *Owner = @ptrCast(@alignCast(ctx));
        if (epoch == 0) return error.InvalidRecentQueryFrame;
        if (self.epoch) |prior| if (prior != epoch and metal_runtime.termite_metal_decode_runtime_last_completed_recent_query_capture_frame_nonce(self.raw(), self) != prior) return error.InvalidRecentQueryFrame;
        self.epoch = epoch;
        const i = try self.index(request.donor_layer, request.consumer_layer);
        if (self.pending[i].encoded) return error.DuplicateFutureHhoEquivalenceConsumer;
        self.pending[i] = .{ .encoded = true, .donor = request.donor_layer, .consumer = request.consumer_layer };
        self.native_commit_calls += 1;
        self.native_copy_bytes += host_mod.sample_rows * host_mod.query_heads * host_mod.head_dim * @sizeOf(f32);
    }
    fn completeCallback(ctx: *anyopaque, c: capture_contract.Completion) anyerror!void {
        const self: *Owner = @ptrCast(@alignCast(ctx));
        try self.complete(c);
    }
    pub fn complete(self: *Owner, c: capture_contract.Completion) !void {
        errdefer self.poisoned = true;
        if (c.authority_epoch != self.host.authority.capture_epoch or !c.submitted or !c.completed or c.cancelled or c.frame_submission_epoch != (self.epoch orelse return error.IncompleteFutureHhoEquivalenceCapture) or c.completed_frame_watermark != c.frame_submission_epoch) return error.IncompleteFutureHhoEquivalenceCapture;
        var downloaded = try self.allocator.alloc(f32, self.shared.elemCount());
        defer self.allocator.free(downloaded);
        try self.shared.downloadF32Into(downloaded);
        const width = host_mod.sample_rows * host_mod.query_heads * host_mod.head_dim;
        for (self.pending, 0..) |pending, i| {
            if (!pending.encoded) return error.IncompleteFutureHhoEquivalenceCapture;
            try self.host.capture(pending.donor, pending.consumer, host_mod.first_heldout_position, host_mod.query_heads, host_mod.head_dim, downloaded[i * width ..][0..width]);
        }
        self.completed = true;
    }
    pub fn descriptor(self: *Owner) capture_contract.Descriptor {
        return .{ .context = self, .authority_epoch = self.host.authority.capture_epoch, .logical_tokens = host_mod.source_tokens, .capture_sequence_end = host_mod.source_tokens + host_mod.continuation_rows, .first_absolute_position = host_mod.first_heldout_position, .fitted_middle_end_exclusive = host_mod.source_tokens, .horizon = host_mod.sample_rows, .expected_consumers = host_mod.consumer_count, .sample_stride = host_mod.sample_stride, .accepts_fn = accepts, .last_encoded_frame_fn = lastFrame, .encode_fn = unusedEncode, .complete_fn = completeCallback, .native_plan_fn = plan, .native_commit_fn = commit };
    }
    pub fn heldout(self: *Owner, allocator: std.mem.Allocator) !host_mod.HeldoutOutput {
        if (!self.completed or self.poisoned) return error.IncompleteFutureHhoEquivalenceCapture;
        return self.host.seal(allocator);
    }
};

test "future HHO equivalence native plan gathers physical row zero with stride eight" {
    var domain: u8 = 0;
    var destination: u8 = 0;
    const request: capture_contract.NativePlanRequest = .{
        .consumer_layer = 5,
        .donor_layer = 5,
        .first_absolute_position = 8128,
        .row_count = 64,
        .num_heads = 8,
        .head_dim = 512,
        .source_rows = 512,
        .sample_first_row = 0,
        .sample_stride = 8,
    };
    const plan = try buildNativePlan(request, &domain, &destination, 4096, 2);
    const row_bytes: usize = 8 * 512 * @sizeOf(f32);
    const capture_bytes: usize = 64 * row_bytes;
    try std.testing.expectEqual(@as(usize, 0), plan.source_byte_offset);
    try std.testing.expectEqual(capture_bytes, plan.byte_count);
    try std.testing.expectEqual(@as(u16, 8), plan.sample_stride);
    try std.testing.expectEqual(row_bytes, plan.source_row_bytes);
    try std.testing.expectEqual(@as(usize, 4096 + 2 * capture_bytes), plan.destination_byte_offset);

    var mislabeled_row_four = request;
    mislabeled_row_four.sample_first_row = 4;
    try std.testing.expectError(
        error.InvalidFutureHhoEquivalenceCapture,
        buildNativePlan(mislabeled_row_four, &domain, &destination, 4096, 2),
    );
}
