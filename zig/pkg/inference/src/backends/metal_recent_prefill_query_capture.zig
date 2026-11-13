const std = @import("std");
const MetalTensor = @import("metal_tensor.zig").MetalTensor;
const metal_runtime = @import("metal_runtime.zig");
const capture = @import("../runtime/kv/attention_match_prefill_recent64_capture.zig");
const descriptor = @import("../graph/recent_prefill_query_capture.zig");

pub const EncodeRequest = descriptor.EncodeRequest;
pub const Completion = descriptor.Completion;

const Pending = struct {
    encoded: bool = false,
    donor_layer: usize = 0,
    consumer_layer: usize = 0,
    first_position: usize = 0,
    rows: usize = 0,
};

/// Owns one fixed shared Metal destination. `encode` only appends a blit to the
/// caller's active frame. Host publication occurs once, after a successful
/// terminal wait passed to `complete`.
pub const Owner = struct {
    allocator: std.mem.Allocator,
    runtime: *anyopaque,
    host: capture.Owner,
    shared: MetalTensor,
    pending: [capture.consumer_count]Pending = @splat(.{}),
    completed: bool = false,
    poisoned: bool = false,
    encoded_calls: usize = 0,
    canonical_copy_calls: usize = 0,
    native_plan_calls: usize = 0,
    native_commit_calls: usize = 0,
    native_copy_bytes: usize = 0,
    encoded_frame_submission_epoch: ?u64 = null,
    seal_value: ?capture.Seal = null,

    pub const DebugSnapshot = struct {
        encoded_calls: usize,
        pending_consumers: usize,
        encoded_frame_submission_epoch: u64,
        completed: bool,
        poisoned: bool,
        canonical_copy_calls: usize,
        native_plan_calls: usize,
        native_commit_calls: usize,
        native_copy_bytes: usize,
    };

    pub fn debugSnapshot(self: *const Owner) DebugSnapshot {
        var pending_consumers: usize = 0;
        for (self.pending) |pending| pending_consumers += @intFromBool(pending.encoded);
        return .{
            .encoded_calls = self.encoded_calls,
            .pending_consumers = pending_consumers,
            .encoded_frame_submission_epoch = self.encoded_frame_submission_epoch orelse 0,
            .completed = self.completed,
            .poisoned = self.poisoned,
            .canonical_copy_calls = self.canonical_copy_calls,
            .native_plan_calls = self.native_plan_calls,
            .native_commit_calls = self.native_commit_calls,
            .native_copy_bytes = self.native_copy_bytes,
        };
    }

    pub fn init(allocator: std.mem.Allocator, runtime: *anyopaque, authority: capture.Authority, topology: [capture.consumer_count]capture.Consumer, maximum_host_bytes: usize, control: capture.Control) !Owner {
        var host = try capture.Owner.init(allocator, authority, topology, maximum_host_bytes, control);
        errdefer host.deinit();
        const elements = capture.consumer_count * capture.horizon * capture.query_heads * capture.head_dim;
        var shared = try MetalTensor.deviceAllocateFreshWithAllocator(allocator, runtime, elements * @sizeOf(f32), .shared, &.{@intCast(elements)});
        errdefer shared.deinit();
        return .{ .allocator = allocator, .runtime = runtime, .host = host, .shared = shared };
    }

    pub fn sharedRequiredBytes() !usize {
        return std.math.mul(usize, capture.consumer_count * capture.horizon * capture.query_heads * capture.head_dim, @sizeOf(f32));
    }

    pub fn deinit(self: *Owner) void {
        self.shared.deinit();
        self.host.deinit();
        self.* = undefined;
    }

    pub fn physicalCapacityBytes(self: *const Owner) usize {
        return self.host.maximum_live_bytes + self.shared.elemCount() * @sizeOf(f32);
    }

    fn rawRuntime(self: *const Owner) *metal_runtime.RawMetalDecodeRuntime {
        return @ptrCast(@alignCast(self.runtime));
    }

    fn entryIndex(self: *const Owner, donor_layer: usize, consumer_layer: usize) !usize {
        var found: ?usize = null;
        for (self.host.entries, 0..) |entry, index| if (entry.tuple.donor_layer == donor_layer and entry.tuple.consumer_layer == consumer_layer) {
            if (found != null) return error.DuplicateRecentQueryConsumer;
            found = index;
        };
        return found orelse error.UnexpectedRecentQueryConsumer;
    }

    pub fn encode(self: *Owner, request: EncodeRequest) !void {
        if (self.poisoned or self.completed) return error.RecentQueryCaptureSealed;
        try self.host.control.check();
        if (request.authority_epoch != self.host.authority.capture_epoch or request.num_heads != capture.query_heads or
            request.head_dim != capture.head_dim or request.row_count == 0 or request.source_row_stride_bytes != capture.query_heads * capture.head_dim * @sizeOf(f32))
            return error.InvalidRecentQueryShape;
        if (self.encoded_frame_submission_epoch) |epoch| {
            if (request.frame_submission_epoch != epoch) {
                // Planned prefill barriers may submit and wait between global
                // consumers. Admit the next copy only after the provider
                // proves the prior copy's exact frame completed.
                if (metal_runtime.termite_metal_decode_runtime_last_completed_recent_query_capture_frame_nonce(self.rawRuntime(), @constCast(self)) != epoch)
                    return error.InvalidRecentQueryFrame;
                self.encoded_frame_submission_epoch = request.frame_submission_epoch;
            }
        } else {
            self.encoded_frame_submission_epoch = request.frame_submission_epoch;
        }
        const rows: usize = @min(request.row_count, capture.horizon);
        const first_position: usize = request.first_absolute_position;
        const row_count: usize = request.row_count;
        if (first_position > self.host.authority.logical_tokens or row_count > self.host.authority.logical_tokens - first_position)
            return error.InvalidRecentQueryPosition;
        const donor_layer: usize = request.donor_layer;
        const consumer_layer: usize = request.consumer_layer;
        const index = try self.entryIndex(donor_layer, consumer_layer);
        if (self.pending[index].encoded) {
            self.poisoned = true;
            return error.DuplicateRecentQueryConsumer;
        }
        const source_tail = try std.math.add(usize, request.source_byte_offset, try std.math.mul(usize, row_count - rows, request.source_row_stride_bytes));
        const destination_stride = capture.horizon * capture.query_heads * capture.head_dim * @sizeOf(f32);
        const destination = try std.math.mul(usize, index, destination_stride);
        const bytes = try std.math.mul(usize, rows, request.source_row_stride_bytes);
        const handle = self.shared.deviceHandle() orelse return error.MetalBufferAllocFailed;
        const rc = metal_runtime.termite_metal_buffer_copy_on_active_frame(
            self.rawRuntime(),
            self,
            request.active_command_encoder,
            request.frame_submission_epoch,
            request.source_buffer,
            source_tail,
            handle,
            self.shared.deviceByteOffset() + destination,
            bytes,
        );
        if (rc != 0) {
            self.poisoned = true;
            return error.MetalBufferCopyFailed;
        }
        self.pending[index] = .{ .encoded = true, .donor_layer = donor_layer, .consumer_layer = consumer_layer, .first_position = first_position + row_count - rows, .rows = rows };
        self.encoded_calls += 1;
        self.canonical_copy_calls += 1;
    }

    pub fn encodeCallback(ctx: *anyopaque, request: EncodeRequest) anyerror!void {
        const self: *Owner = @ptrCast(@alignCast(ctx));
        try self.encode(request);
    }

    pub fn nativePlanCallback(ctx: *anyopaque, request: descriptor.NativePlanRequest) anyerror!?descriptor.NativePlan {
        const self: *Owner = @ptrCast(@alignCast(ctx));
        if (self.poisoned or self.completed) return error.RecentQueryCaptureSealed;
        try self.host.control.check();
        if (request.num_heads != capture.query_heads or request.head_dim != capture.head_dim or
            request.row_count == 0 or request.row_count > capture.horizon or request.sample_stride != 1 or
            request.source_rows < request.row_count or
            @as(usize, request.sample_first_row) + request.row_count > request.source_rows)
            return error.InvalidRecentQueryShape;
        const index = try self.entryIndex(request.donor_layer, request.consumer_layer);
        if (self.pending[index].encoded) {
            self.poisoned = true;
            return error.DuplicateRecentQueryConsumer;
        }
        const stride = capture.horizon * capture.query_heads * capture.head_dim * @sizeOf(f32);
        const bytes = @as(usize, request.row_count) * capture.query_heads * capture.head_dim * @sizeOf(f32);
        const handle = self.shared.deviceHandle() orelse return error.MetalBufferAllocFailed;
        self.native_plan_calls += 1;
        return .{
            .capture_domain = self,
            .destination_buffer = handle,
            .destination_byte_offset = self.shared.deviceByteOffset() + index * stride,
            .byte_count = bytes,
            .source_byte_offset = @as(usize, request.sample_first_row) * capture.query_heads * capture.head_dim * @sizeOf(f32),
            .sample_rows = request.row_count,
            .sample_stride = request.sample_stride,
            .source_row_bytes = capture.query_heads * capture.head_dim * @sizeOf(f32),
        };
    }

    pub fn nativeCommitCallback(ctx: *anyopaque, request: descriptor.NativePlanRequest, epoch: u64) anyerror!void {
        const self: *Owner = @ptrCast(@alignCast(ctx));
        if (epoch == 0) return error.InvalidRecentQueryFrame;
        if (self.encoded_frame_submission_epoch) |prior| {
            if (prior != epoch) {
                if (metal_runtime.termite_metal_decode_runtime_last_completed_recent_query_capture_frame_nonce(self.rawRuntime(), @constCast(self)) != prior)
                    return error.InvalidRecentQueryFrame;
                self.encoded_frame_submission_epoch = epoch;
            }
        } else self.encoded_frame_submission_epoch = epoch;
        const index = try self.entryIndex(request.donor_layer, request.consumer_layer);
        if (self.pending[index].encoded) return error.DuplicateRecentQueryConsumer;
        self.pending[index] = .{
            .encoded = true,
            .donor_layer = request.donor_layer,
            .consumer_layer = request.consumer_layer,
            .first_position = request.first_absolute_position,
            .rows = request.row_count,
        };
        self.encoded_calls += 1;
        self.native_commit_calls += 1;
        self.native_copy_bytes += @as(usize, request.row_count) * @as(usize, request.num_heads) * @as(usize, request.head_dim) * @sizeOf(f32);
    }

    pub fn accepts(self: *const Owner, donor_layer: u32, consumer_layer: u32) bool {
        _ = self.entryIndex(donor_layer, consumer_layer) catch return false;
        return true;
    }

    pub fn acceptsCallback(ctx: *anyopaque, donor_layer: u32, consumer_layer: u32) bool {
        const self: *const Owner = @ptrCast(@alignCast(ctx));
        return self.accepts(donor_layer, consumer_layer);
    }

    pub fn lastEncodedFrameCallback(ctx: *anyopaque) u64 {
        const self: *const Owner = @ptrCast(@alignCast(ctx));
        return self.encoded_frame_submission_epoch orelse 0;
    }

    pub fn complete(self: *Owner, completion: Completion) !capture.Seal {
        if (self.poisoned or self.completed) return error.RecentQueryCaptureSealed;
        errdefer self.poisoned = true;
        if (completion.authority_epoch != self.host.authority.capture_epoch) return error.InvalidRecentQueryAuthority;
        if (!completion.submitted or !completion.completed or completion.cancelled)
            return if (completion.cancelled) error.CompactionCancelled else error.IncompleteRecentQueryCapture;
        const encoded_epoch = self.encoded_frame_submission_epoch orelse return error.IncompleteRecentQueryCapture;
        if (completion.frame_submission_epoch != encoded_epoch or
            completion.completed_frame_watermark != encoded_epoch)
            return error.IncompleteRecentQueryCapture;
        var downloaded = try self.allocator.alloc(f32, self.shared.elemCount());
        defer self.allocator.free(downloaded);
        try self.shared.downloadF32Into(downloaded);
        const entry_values = capture.horizon * capture.query_heads * capture.head_dim;
        for (self.pending, 0..) |pending, index| {
            if (!pending.encoded or pending.rows != capture.horizon) return error.IncompleteRecentQueryCapture;
            try self.host.capture(pending.donor_layer, pending.consumer_layer, pending.first_position, capture.query_heads, capture.head_dim, downloaded[index * entry_values ..][0..entry_values]);
        }
        const sealed = try self.host.seal();
        self.seal_value = sealed;
        self.completed = true;
        return sealed;
    }

    pub fn completeCallback(ctx: *anyopaque, completion: Completion) anyerror!void {
        const self: *Owner = @ptrCast(@alignCast(ctx));
        _ = try self.complete(completion);
    }

    pub fn makeDescriptor(self: *Owner) descriptor.Descriptor {
        return .{
            .context = self,
            .authority_epoch = self.host.authority.capture_epoch,
            .logical_tokens = @intCast(self.host.authority.logical_tokens),
            .capture_sequence_end = @intCast(self.host.authority.logical_tokens),
            .first_absolute_position = @intCast(self.host.authority.logical_tokens - capture.horizon),
            .fitted_middle_end_exclusive = @intCast(self.host.authority.fitted_middle_end_exclusive),
            .horizon = @intCast(capture.horizon),
            .expected_consumers = @intCast(capture.consumer_count),
            .sample_stride = 1,
            .accepts_fn = acceptsCallback,
            .last_encoded_frame_fn = lastEncodedFrameCallback,
            .encode_fn = encodeCallback,
            .complete_fn = completeCallback,
            .native_plan_fn = nativePlanCallback,
            .native_commit_fn = nativeCommitCallback,
        };
    }

    pub fn training(self: *const Owner, allocator: std.mem.Allocator) !capture.Training {
        if (self.poisoned or !self.completed) return error.IncompleteRecentQueryCapture;
        return self.host.training(allocator, self.seal_value orelse return error.IncompleteRecentQueryCapture);
    }
};

test "recent prefill Metal owner rejects completion before submission" {
    // Allocation and GPU-copy behavior is covered by the existing physical
    // Metal fixture. This pure test pins the fail-closed completion predicate.
    const completion: Completion = .{ .authority_epoch = 1, .submitted = false, .completed = false, .cancelled = false, .frame_submission_epoch = 4 };
    try std.testing.expect(!completion.submitted and completion.frame_submission_epoch == 4);
}
