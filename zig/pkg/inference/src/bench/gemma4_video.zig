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

//! Synchronized NVDEC + Gemma4 patch preparation benchmark; optional real projector.
const std = @import("std");
const internal = @import("inference_internal");
const video = internal.native_compute.cuda_video;
const projector = internal.architectures.gemma4_projector;
fn rejectReadback(_: *anyopaque, _: internal.ops.CT, _: std.mem.Allocator) anyerror![]f32 {
    std.debug.dumpCurrentStackTrace(.{ .first_address = @returnAddress() });
    return error.UnexpectedVideoReadback;
}
fn now() !u64 {
    var ts: std.posix.timespec = undefined;
    if (std.posix.errno(std.posix.system.clock_gettime(.MONOTONIC, &ts)) != .SUCCESS) return error.ClockFailed;
    return @intCast(@as(i128, ts.sec) * std.time.ns_per_s + ts.nsec);
}
pub fn main(init: std.process.Init) !void {
    if (comptime !@import("build_options").enable_cuda) return error.CudaRequired;
    internal.execution_control.allowUninterruptibleInProcess();
    const a = std.heap.c_allocator;
    const args = try init.minimal.args.toSlice(a);
    defer a.free(args);
    if (args.len < 3 or args.len > 5) return error.ExpectedVideoAndReportOptionalIterationsAndProjector;
    const iterations = if (args.len >= 4) try std.fmt.parseInt(usize, args[3], 10) else 10;
    if (iterations == 0 or iterations > 1000) return error.InvalidIterations;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], a, .limited(256 * 1024 * 1024));
    defer a.free(bytes);
    var compute = try internal.native_compute.cuda.CudaCompute.init(a);
    defer compute.deinit();
    const plain = compute.computeBackend();
    var resident_vtable = plain.vtable.*;
    resident_vtable.toFloat32 = rejectReadback;
    const cb = internal.ops.ComputeBackend{ .ptr = plain.ptr, .vtable = &resident_vtable };
    var prep = try video.Preprocessor.init(a, &compute.ctx);
    defer prep.deinit();
    const store = if (args.len == 5) try projector.ProjectorStore.open(a, args[4]) else null;
    defer if (store) |p| p.close();
    const Consumer = struct {
        cb: *const internal.ops.ComputeBackend,
        prep: *video.Preprocessor,
        fn consume(user_data: *anyopaque, frame: video.Frame) !void {
            const self: *@This() = @ptrCast(@alignCast(user_data));
            // 70 soft tokens = 630 patches at patch16 / spatial merge3.
            const output = try self.prep.patches(self.cb, frame, 480, 336, 16, .{ 0, 0, 0 }, .{ 1, 1, 1 });
            defer self.cb.free(output);
        }
    };
    var consumer = Consumer{ .cb = &cb, .prep = &prep };
    const Sample = struct { seconds: f64, decoded_frames: usize, selected_frames: usize, d2h_bytes: u64, h2d_bytes: u64 };
    const samples = try a.alloc(Sample, iterations);
    defer a.free(samples);
    const times = try a.alloc(f64, iterations);
    defer a.free(times);
    for (0..iterations + 1) |i| {
        const before = cb.trainingRuntimeStats();
        const start = try now();
        const stats = if (store) |p| blk: {
            var projected = try projector.encodeProjectedVideoResident(&cb, a, p, bytes);
            defer projected.deinit();
            break :blk projected.stats;
        } else try video.decodeMp4(a, &compute.ctx, bytes, .{ .context = &consumer, .consume = Consumer.consume });
        try compute.ctx.synchronize();
        const seconds = @as(f64, @floatFromInt(try now() - start)) / std.time.ns_per_s;
        const after = cb.trainingRuntimeStats();
        if (after.d2h_bytes != before.d2h_bytes) return error.UnexpectedVideoReadback;
        std.debug.print("video_bench: iteration={d} seconds={d:.3} selected={d} h2d={d} d2h={d} cublaslt={any}\n", .{ i, seconds, stats.selected_frames, after.h2d_bytes - before.h2d_bytes, after.d2h_bytes - before.d2h_bytes, compute.cublaslt != null });
        if (i == 0) continue;
        samples[i - 1] = .{ .seconds = seconds, .decoded_frames = stats.decoded_frames, .selected_frames = stats.selected_frames, .d2h_bytes = after.d2h_bytes - before.d2h_bytes, .h2d_bytes = after.h2d_bytes - before.h2d_bytes };
        times[i - 1] = seconds;
    }
    std.mem.sort(f64, times, {}, std.sort.asc(f64));
    const json = try std.json.Stringify.valueAlloc(a, .{ .video = args[1], .gpu = compute.ctx.info.nameSlice(), .scope = if (store != null) "decode-through-projector" else "decode-through-patches", .warmups = 1, .p50_seconds = times[(iterations - 1) / 2], .p95_seconds = times[(iterations * 95 + 99) / 100 - 1], .samples = samples }, .{ .whitespace = .indent_2 });
    defer a.free(json);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[2], .data = json });
}
