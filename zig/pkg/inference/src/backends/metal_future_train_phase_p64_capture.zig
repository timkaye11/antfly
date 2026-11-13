const std = @import("std");
const MetalTensor = @import("metal_tensor.zig").MetalTensor;
const metal_runtime = @import("metal_runtime.zig");
const host_mod = @import("../runtime/kv/attention_match_future_train_phase_p64_capture.zig");
const capture_contract = @import("../graph/recent_prefill_query_capture.zig");

const Pending = struct {
    planned: bool = false,
    committed_phases: u8 = 0,
    donor: usize = 0,
    consumer: usize = 0,
};

fn phaseRows(phase_index: usize) usize {
    return if (phase_index == 0) 10 else 9;
}

fn phaseFirstSourceRow(phase_index: usize) usize {
    return 1 + 9 * phase_index;
}

fn phasePackedStart(phase_index: usize) usize {
    return if (phase_index == 0) 0 else 10 + 9 * (phase_index - 1);
}

fn scatterChronological(grouped_values: []const f32, chronological: []f32, row_width: usize) !void {
    if (row_width == 0 or grouped_values.len != host_mod.sample_rows * row_width or chronological.len != grouped_values.len)
        return error.InvalidFutureTrainPhaseP64Capture;
    for (0..host_mod.sample_rows) |chronological_row| {
        const grouped_row = host_mod.groupedRow(chronological_row);
        const source = grouped_values[grouped_row * row_width ..][0..row_width];
        for (source) |value| if (!std.math.isFinite(value)) return error.NonFiniteFutureTrainPhaseP64Capture;
        @memcpy(chronological[chronological_row * row_width ..][0..row_width], source);
    }
}

fn requestRowBytes(request: capture_contract.NativePlanRequest) !usize {
    return std.math.mul(usize, try std.math.mul(usize, request.num_heads, request.head_dim), @sizeOf(f32));
}

fn validateRequest(request: capture_contract.NativePlanRequest) !void {
    if (request.first_absolute_position != host_mod.first_training_position or
        request.row_count != host_mod.sample_rows or
        request.num_heads != host_mod.query_heads or
        request.head_dim != host_mod.head_dim or
        request.source_rows != host_mod.continuation_rows or
        request.sample_first_row != 1 or
        request.sample_stride != 8)
        return error.InvalidFutureTrainPhaseP64Capture;
}

pub fn buildNativePlanBatch(
    request: capture_contract.NativePlanRequest,
    capture_domain: *anyopaque,
    destination_buffer: *anyopaque,
    destination_base_offset: usize,
    destination_byte_capacity: usize,
    consumer_index: usize,
) !capture_contract.NativePlanBatch {
    try validateRequest(request);
    const row_bytes = try requestRowBytes(request);
    const consumer_bytes = try std.math.mul(usize, host_mod.sample_rows, row_bytes);
    const consumer_offset = try std.math.add(usize, destination_base_offset, try std.math.mul(usize, consumer_index, consumer_bytes));
    var batch: capture_contract.NativePlanBatch = undefined;
    batch.count = host_mod.phase_count;
    for (0..host_mod.phase_count) |phase_index| {
        const rows = phaseRows(phase_index);
        const first_source_row = phaseFirstSourceRow(phase_index);
        const last_source_row = first_source_row + (rows - 1) * 56;
        if (last_source_row >= request.source_rows) return error.InvalidFutureTrainPhaseP64Capture;
        batch.plans[phase_index] = .{
            .capture_domain = capture_domain,
            .destination_buffer = destination_buffer,
            .destination_byte_offset = try std.math.add(usize, consumer_offset, try std.math.mul(usize, phasePackedStart(phase_index), row_bytes)),
            .destination_byte_capacity = destination_byte_capacity,
            .byte_count = try std.math.mul(usize, rows, row_bytes),
            .source_byte_offset = try std.math.mul(usize, first_source_row, row_bytes),
            .sample_rows = @intCast(rows),
            .sample_stride = 56,
            .source_row_bytes = row_bytes,
        };
    }
    return batch;
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
        return error.UnexpectedFutureTrainPhaseP64Consumer;
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
        return error.FutureTrainPhaseP64RequiresNativeGather;
    }

    fn planBatch(ctx: *anyopaque, request: capture_contract.NativePlanRequest) anyerror!?capture_contract.NativePlanBatch {
        const self: *Owner = @ptrCast(@alignCast(ctx));
        errdefer self.poisoned = true;
        if (self.poisoned or self.completed) return error.FutureTrainPhaseP64CaptureSealed;
        const i = try self.index(request.donor_layer, request.consumer_layer);
        if (self.pending[i].planned or self.pending[i].committed_phases != 0) return error.DuplicateFutureTrainPhaseP64Consumer;
        const destination_capacity = try std.math.add(usize, self.shared.deviceByteOffset(), self.shared.deviceByteLen());
        const batch = try buildNativePlanBatch(request, self, self.shared.deviceHandle() orelse return error.MetalBufferAllocFailed, self.shared.deviceByteOffset(), destination_capacity, i);
        self.pending[i] = .{ .planned = true, .donor = request.donor_layer, .consumer = request.consumer_layer };
        self.native_plan_calls += host_mod.phase_count;
        return batch;
    }

    fn commitPhase(ctx: *anyopaque, request: capture_contract.NativePlanRequest, phase: u8, epoch: u64) anyerror!void {
        const self: *Owner = @ptrCast(@alignCast(ctx));
        errdefer self.poisoned = true;
        if (self.poisoned or self.completed) return error.FutureTrainPhaseP64CaptureSealed;
        try validateRequest(request);
        if (phase == 0 or phase > host_mod.phase_count or epoch == 0) return error.InvalidRecentQueryFrame;
        if (self.epoch) |prior| {
            if (prior != epoch) return error.InvalidRecentQueryFrame;
        } else self.epoch = epoch;
        const i = try self.index(request.donor_layer, request.consumer_layer);
        if (!self.pending[i].planned or self.pending[i].donor != request.donor_layer or self.pending[i].consumer != request.consumer_layer) return error.IncompleteFutureTrainPhaseP64Capture;
        const mask = @as(u8, 1) << @intCast(phase - 1);
        if (self.pending[i].committed_phases & mask != 0) return error.DuplicateFutureTrainPhaseP64Consumer;
        self.pending[i].committed_phases |= mask;
        self.native_commit_calls += 1;
        self.native_copy_bytes += phaseRows(phase - 1) * host_mod.query_heads * host_mod.head_dim * @sizeOf(f32);
    }

    fn completeCallback(ctx: *anyopaque, completion: capture_contract.Completion) anyerror!void {
        const self: *Owner = @ptrCast(@alignCast(ctx));
        try self.complete(completion);
    }

    pub fn complete(self: *Owner, completion: capture_contract.Completion) !void {
        errdefer self.poisoned = true;
        const epoch = self.epoch orelse return error.IncompleteFutureTrainPhaseP64Capture;
        if (self.poisoned or self.completed or
            completion.authority_epoch != self.host.authority.capture_epoch or
            !completion.submitted or !completion.completed or completion.cancelled or
            completion.frame_submission_epoch != epoch or
            completion.completed_frame_watermark != epoch or
            metal_runtime.termite_metal_decode_runtime_last_completed_recent_query_capture_frame_nonce(self.raw(), self) != epoch or
            self.native_plan_calls != host_mod.consumer_count * host_mod.phase_count or
            self.native_commit_calls != host_mod.consumer_count * host_mod.phase_count or
            self.native_copy_bytes != try sharedRequiredBytes())
            return error.IncompleteFutureTrainPhaseP64Capture;

        const downloaded = try self.allocator.alloc(f32, self.shared.elemCount());
        defer self.allocator.free(downloaded);
        try self.shared.downloadF32Into(downloaded);
        const row_width = host_mod.query_heads * host_mod.head_dim;
        const consumer_width = host_mod.sample_rows * row_width;
        const chronological = try self.allocator.alloc(f32, consumer_width);
        defer self.allocator.free(chronological);
        const complete_mask: u8 = (@as(u8, 1) << host_mod.phase_count) - 1;
        for (self.pending, 0..) |pending, consumer_index| {
            if (!pending.planned or pending.committed_phases != complete_mask) return error.IncompleteFutureTrainPhaseP64Capture;
            const grouped_values = downloaded[consumer_index * consumer_width ..][0..consumer_width];
            try scatterChronological(grouped_values, chronological, row_width);
            try self.host.capture(pending.donor, pending.consumer, host_mod.first_training_position, host_mod.query_heads, host_mod.head_dim, chronological);
        }
        self.completed = true;
    }

    pub fn descriptor(self: *Owner) capture_contract.Descriptor {
        return .{
            .context = self,
            .authority_epoch = self.host.authority.capture_epoch,
            .logical_tokens = host_mod.source_tokens,
            .capture_sequence_end = host_mod.source_tokens + host_mod.continuation_rows,
            .first_absolute_position = host_mod.first_training_position,
            .fitted_middle_end_exclusive = host_mod.source_tokens,
            .horizon = host_mod.sample_rows,
            .expected_consumers = host_mod.consumer_count,
            .sample_stride = 8,
            .accepts_fn = accepts,
            .last_encoded_frame_fn = lastFrame,
            .encode_fn = unusedEncode,
            .complete_fn = completeCallback,
            .native_batch_plan_fn = planBatch,
            .native_batch_commit_fn = commitPhase,
        };
    }

    pub fn training(self: *Owner, allocator: std.mem.Allocator) !host_mod.TrainingOutput {
        errdefer self.poisoned = true;
        if (!self.completed or self.poisoned) return error.IncompleteFutureTrainPhaseP64Capture;
        return self.host.seal(allocator);
    }
};

test "phase-balanced TRAIN P64 batch is seven disjoint stride-56 segments" {
    var domain: u8 = 0;
    var destination: u8 = 0;
    const request: capture_contract.NativePlanRequest = .{ .consumer_layer = 5, .donor_layer = 5, .first_absolute_position = 8129, .row_count = 64, .num_heads = 8, .head_dim = 512, .source_rows = 512, .sample_first_row = 1, .sample_stride = 8 };
    const batch = try buildNativePlanBatch(request, &domain, &destination, 4096, 4096 + 7 * 64 * 8 * 512 * @sizeOf(f32), 2);
    const plans = try batch.slice();
    try std.testing.expectEqual(@as(usize, 7), plans.len);
    const row_bytes: usize = 8 * 512 * @sizeOf(f32);
    var rows: usize = 0;
    var bytes: usize = 0;
    for (plans, 0..) |plan, phase_index| {
        try std.testing.expectEqual(@as(u16, 56), plan.sample_stride);
        try std.testing.expectEqual(phaseRows(phase_index), plan.sample_rows);
        try std.testing.expectEqual(phaseFirstSourceRow(phase_index) * row_bytes, plan.source_byte_offset);
        rows += plan.sample_rows;
        bytes += plan.byte_count;
    }
    try std.testing.expectEqual(@as(usize, 64), rows);
    try std.testing.expectEqual(@as(usize, 64 * row_bytes), bytes);
    try std.testing.expectEqual(@as(usize, 4096 + 2 * 64 * row_bytes), plans[0].destination_byte_offset);
    for (plans) |plan| try std.testing.expect(plan.destination_byte_offset + plan.byte_count <= plan.destination_byte_capacity);
}

test "phase-balanced TRAIN P64 rejects the legacy physical-row-four request" {
    var domain: u8 = 0;
    var destination: u8 = 0;
    const request: capture_contract.NativePlanRequest = .{ .consumer_layer = 5, .donor_layer = 5, .first_absolute_position = 8132, .row_count = 64, .num_heads = 8, .head_dim = 512, .source_rows = 512, .sample_first_row = 4, .sample_stride = 8 };
    try std.testing.expectError(error.InvalidFutureTrainPhaseP64Capture, buildNativePlanBatch(request, &domain, &destination, 0, 7 * 64 * 8 * 512 * @sizeOf(f32), 0));
}

test "phase-balanced TRAIN P64 scatter restores chronology and rejects nonfinite input" {
    var grouped: [host_mod.sample_rows]f32 = undefined;
    for (0..host_mod.sample_rows) |row| grouped[host_mod.groupedRow(row)] = @floatFromInt(row);
    var chronological: [host_mod.sample_rows]f32 = undefined;
    try scatterChronological(&grouped, &chronological, 1);
    for (chronological, 0..) |value, row| try std.testing.expectEqual(@as(f32, @floatFromInt(row)), value);
    grouped[host_mod.groupedRow(31)] = std.math.nan(f32);
    try std.testing.expectError(error.NonFiniteFutureTrainPhaseP64Capture, scatterChronological(&grouped, &chronological, 1));
}
