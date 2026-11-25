// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

const std = @import("std");

/// Borrowed, request-scoped descriptor for the default-off compiled-prefill
/// recent-query development collector. The callbacks must not retain the
/// descriptor itself. `encode_fn` may only encode a device-to-shared-buffer
/// copy into the supplied active frame; it must not wait or read host memory.
pub const Descriptor = struct {
    context: *anyopaque,
    authority_epoch: u64,
    logical_tokens: u32 = 8128,
    capture_sequence_end: u32 = 8128,
    first_absolute_position: u32 = 8064,
    fitted_middle_end_exclusive: u32 = 6080,
    horizon: u16 = 64,
    expected_consumers: u8 = 7,
    sample_stride: u16 = 1,
    accepts_fn: *const fn (context: *anyopaque, donor_layer: u32, consumer_layer: u32) bool,
    last_encoded_frame_fn: *const fn (context: *anyopaque) u64,
    encode_fn: *const fn (context: *anyopaque, request: EncodeRequest) anyerror!void,
    complete_fn: *const fn (context: *anyopaque, completion: Completion) anyerror!void,
    native_plan_fn: ?*const fn (context: *anyopaque, request: NativePlanRequest) anyerror!?NativePlan = null,
    native_commit_fn: ?*const fn (context: *anyopaque, request: NativePlanRequest, frame_submission_epoch: u64) anyerror!void = null,
    native_batch_plan_fn: ?*const fn (context: *anyopaque, request: NativePlanRequest) anyerror!?NativePlanBatch = null,
    native_batch_commit_fn: ?*const fn (context: *anyopaque, request: NativePlanRequest, phase: u8, frame_submission_epoch: u64) anyerror!void = null,

    pub fn validate(self: Descriptor) !void {
        if (self.authority_epoch == 0 or self.logical_tokens == 0 or self.capture_sequence_end < self.logical_tokens or self.horizon == 0 or self.expected_consumers == 0 or self.sample_stride == 0)
            return error.InvalidRecentPrefillQueryCaptureDescriptor;
        if (self.fitted_middle_end_exclusive > self.logical_tokens)
            return error.InvalidRecentPrefillQueryCaptureDescriptor;
        if (@as(u32, self.horizon) > self.logical_tokens)
            return error.InvalidRecentPrefillQueryCaptureDescriptor;
        const span = std.math.mul(u32, @as(u32, self.horizon) - 1, self.sample_stride) catch return error.InvalidRecentPrefillQueryCaptureDescriptor;
        if (self.first_absolute_position < self.fitted_middle_end_exclusive or self.first_absolute_position >= self.capture_sequence_end or span >= self.capture_sequence_end - self.first_absolute_position)
            return error.InvalidRecentPrefillQueryCaptureDescriptor;
        const single = self.native_plan_fn != null or self.native_commit_fn != null;
        const batch = self.native_batch_plan_fn != null or self.native_batch_commit_fn != null;
        if ((single and (self.native_plan_fn == null or self.native_commit_fn == null)) or
            (batch and (self.native_batch_plan_fn == null or self.native_batch_commit_fn == null)) or
            (single and batch)) return error.InvalidRecentPrefillQueryCaptureDescriptor;
    }

    /// The original source length and the final capture frame are distinct
    /// for future TRAIN probes. All samples must fit in the terminal chunk.
    pub fn validateChunk(self: Descriptor, seq_len: usize, query_rows: usize) !bool {
        try self.validate();
        if (seq_len > self.capture_sequence_end) return error.InvalidRecentPrefillQueryPosition;
        if (query_rows == 0 or query_rows > seq_len) return error.UnsupportedRecentPrefillQueryChunking;
        if (seq_len != self.capture_sequence_end) return false;
        const chunk_first = seq_len - query_rows;
        const sample_last = @as(usize, self.first_absolute_position) + (@as(usize, self.horizon) - 1) * self.sample_stride;
        if (chunk_first > self.first_absolute_position or sample_last >= seq_len)
            return error.UnsupportedRecentPrefillQueryChunking;
        return true;
    }

    pub fn accepts(self: Descriptor, donor_layer: u32, consumer_layer: u32) bool {
        return self.accepts_fn(self.context, donor_layer, consumer_layer);
    }

    pub fn lastEncodedFrame(self: Descriptor) u64 {
        return self.last_encoded_frame_fn(self.context);
    }

    pub fn supportsNativeWholeGraph(self: Descriptor) bool {
        const single = self.native_plan_fn != null and self.native_commit_fn != null;
        const batch = self.native_batch_plan_fn != null and self.native_batch_commit_fn != null;
        return single != batch;
    }
};

pub const NativePlanRequest = struct {
    consumer_layer: u32,
    donor_layer: u32,
    first_absolute_position: u32,
    row_count: u16,
    num_heads: u16,
    head_dim: u16,
    source_rows: u16,
    sample_first_row: u16,
    sample_stride: u16,
};

pub const NativePlan = struct {
    capture_domain: *anyopaque,
    destination_buffer: *anyopaque,
    destination_byte_offset: usize,
    /// Full destination allocation extent. Batch receivers require this to
    /// preflight every range before the first command is encoded. Zero keeps
    /// the legacy singular-plan ABI source-compatible.
    destination_byte_capacity: usize = 0,
    byte_count: usize,
    source_byte_offset: usize,
    sample_rows: u16,
    sample_stride: u16,
    source_row_bytes: usize,
};

pub const maximum_native_batch_plans: usize = 7;
pub const NativePlanBatch = struct {
    count: u8,
    plans: [maximum_native_batch_plans]NativePlan,

    pub fn slice(self: *const @This()) ![]const NativePlan {
        if (self.count == 0 or self.count > maximum_native_batch_plans) return error.InvalidRecentPrefillQueryCaptureDescriptor;
        return self.plans[0..self.count];
    }
};

pub const NativeBatchTransport = struct {
    request: NativePlanRequest,
    batch: NativePlanBatch,
};

test "native batch transport bounds one through seven plans" {
    var batch: NativePlanBatch = undefined;
    batch.count = 0;
    try std.testing.expectError(error.InvalidRecentPrefillQueryCaptureDescriptor, batch.slice());
    batch.count = 7;
    try std.testing.expectEqual(@as(usize, 7), (try batch.slice()).len);
    batch.count = 8;
    try std.testing.expectError(error.InvalidRecentPrefillQueryCaptureDescriptor, batch.slice());
}

pub const EncodeRequest = struct {
    authority_epoch: u64,
    consumer_layer: u32,
    donor_layer: u32,
    first_absolute_position: u32,
    row_count: u16,
    num_heads: u16,
    head_dim: u16,
    source_buffer: *anyopaque,
    source_byte_offset: usize,
    source_row_stride_bytes: usize,
    active_command_encoder: *anyopaque,
    frame_submission_epoch: u64,
};

pub const Completion = struct {
    authority_epoch: u64,
    submitted: bool,
    completed: bool,
    cancelled: bool,
    frame_submission_epoch: u64,
    completed_frame_watermark: u64 = 0,
};

test "recent query descriptor validates future stride bounds without weakening recent defaults" {
    const Callbacks = struct {
        fn accepts(_: *anyopaque, _: u32, _: u32) bool {
            return true;
        }
        fn last(_: *anyopaque) u64 {
            return 0;
        }
        fn encode(_: *anyopaque, _: EncodeRequest) anyerror!void {}
        fn complete(_: *anyopaque, _: Completion) anyerror!void {}
    };
    var marker: u8 = 0;
    const base: Descriptor = .{
        .context = &marker,
        .authority_epoch = 1,
        .accepts_fn = Callbacks.accepts,
        .last_encoded_frame_fn = Callbacks.last,
        .encode_fn = Callbacks.encode,
        .complete_fn = Callbacks.complete,
    };
    try base.validate();
    var future = base;
    future.capture_sequence_end = 8640;
    future.first_absolute_position = 8132;
    future.fitted_middle_end_exclusive = 8128;
    future.sample_stride = 8;
    try future.validate();
    try std.testing.expect(!try base.validateChunk(7680, 512));
    try std.testing.expect(try base.validateChunk(8128, 448));
    try std.testing.expect(!try future.validateChunk(8128, 512));
    try std.testing.expect(try future.validateChunk(8640, 512));
    try std.testing.expectError(error.UnsupportedRecentPrefillQueryChunking, future.validateChunk(8640, 64));
    try std.testing.expectError(error.InvalidRecentPrefillQueryPosition, future.validateChunk(8641, 512));
    future.first_absolute_position = 8136;
    try std.testing.expectError(error.InvalidRecentPrefillQueryCaptureDescriptor, future.validate());
    future.first_absolute_position = 8137;
    try std.testing.expectError(error.InvalidRecentPrefillQueryCaptureDescriptor, future.validate());
    future.first_absolute_position = 8132;
    future.sample_stride = 0;
    try std.testing.expectError(error.InvalidRecentPrefillQueryCaptureDescriptor, future.validate());
}
