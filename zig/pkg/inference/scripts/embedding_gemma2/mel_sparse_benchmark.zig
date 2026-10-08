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

const std = @import("std");
const sparse = @import("gemma4_mel_sparse");

const Timespec = extern struct { tv_sec: isize, tv_nsec: isize };
extern fn clock_gettime(c_int, *Timespec) c_int;

fn now() !u64 {
    var time: Timespec = undefined;
    if (clock_gettime(1, &time) != 0) return error.ClockFailed;
    return @as(u64, @intCast(time.tv_sec)) * 1_000_000_000 + @as(u64, @intCast(time.tv_nsec));
}

fn hzToMel(hz: f32) f32 {
    return 2595.0 * std.math.log10(1.0 + hz / 700.0);
}

fn melToHz(mel: f32) f32 {
    return 700.0 * (std.math.pow(f32, 10.0, mel / 2595.0) - 1.0);
}

/// Exact production HTK filterbank formula for 128 bins, FFT512, 16 kHz.
fn buildFilterbank(filters: []f32) void {
    const mel_bins = 128;
    const frequencies = 257;
    const sample_rate: f32 = 16_000;
    const fft_size: f32 = 512;
    var points: [mel_bins + 2]f32 = undefined;
    const low = hzToMel(0.0);
    const high = hzToMel(sample_rate / 2.0);
    for (&points, 0..) |*point, index| {
        const t = @as(f32, @floatFromInt(index)) / @as(f32, @floatFromInt(points.len - 1));
        point.* = melToHz(low + t * (high - low));
    }
    @memset(filters, 0.0);
    for (0..mel_bins) |mel| {
        const left = points[mel] * fft_size / sample_rate;
        const center = points[mel + 1] * fft_size / sample_rate;
        const right = points[mel + 2] * fft_size / sample_rate;
        for (0..frequencies) |frequency| {
            const bin: f32 = @floatFromInt(frequency);
            filters[mel * frequencies + frequency] = if (bin >= left and bin < center and center > left)
                (bin - left) / (center - left)
            else if (bin >= center and bin < right and right > center)
                (right - bin) / (right - center)
            else
                0.0;
        }
    }
}

fn denseSums(magnitudes: []const f32, filters: []const f32, out: []f32) void {
    for (0..99) |frame| {
        for (0..128) |mel| {
            var sum: f32 = 0.0;
            for (0..257) |frequency|
                sum += magnitudes[frame * 257 + frequency] * filters[mel * 257 + frequency];
            out[frame * 128 + mel] = sum;
        }
    }
}

fn sparseSums(magnitudes: []const f32, filters: []const f32, plan: sparse.Plan, out: []f32) !void {
    for (0..99) |frame|
        try sparse.melSums(filters, magnitudes[frame * 257 ..][0..257], plan, out[frame * 128 ..][0..128]);
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const filters = try allocator.alloc(f32, 128 * 257);
    buildFilterbank(filters);
    const magnitudes = try allocator.alloc(f32, 99 * 257);
    var state: u32 = 1;
    for (magnitudes) |*value| {
        state = state *% 1_664_525 +% 1_013_904_223;
        value.* = @as(f32, @floatFromInt(state & 0xffff)) / 4096.0;
    }
    var plan = try sparse.Plan.init(allocator, filters, 128, 257);
    defer plan.deinit();
    var span_terms: usize = 0;
    for (plan.spans) |span| span_terms += span.end - span.first;

    const dense = try allocator.alloc(f32, 99 * 128);
    const selected = try allocator.alloc(f32, 99 * 128);
    denseSums(magnitudes, filters, dense);
    try sparseSums(magnitudes, filters, plan, selected);
    for (dense, selected, 0..) |expected, actual, index| {
        if (@as(u32, @bitCast(expected)) != @as(u32, @bitCast(actual))) {
            std.debug.print("bitwise mismatch at {d}: expected={d} actual={d}\n", .{ index, expected, actual });
            return error.BitwiseMismatch;
        }
    }

    const iterations = 1000;
    const start = try now();
    for (0..iterations) |_| denseSums(magnitudes, filters, dense);
    const middle = try now();
    for (0..iterations) |_| try sparseSums(magnitudes, filters, plan, selected);
    const finish = try now();
    const dense_ns = middle - start;
    const sparse_ns = finish - middle;
    std.debug.print(
        "raw mel sums: span_terms={d}/{d} ({d:.2}%) outputs={d} bitwise=true dense_ns={d} sparse_ns={d} speedup={d:.2} checksum={d}\n",
        .{ span_terms, 128 * 257, 100.0 * @as(f64, @floatFromInt(span_terms)) / @as(f64, @floatFromInt(128 * 257)), dense.len, dense_ns, sparse_ns, @as(f64, @floatFromInt(dense_ns)) / @as(f64, @floatFromInt(sparse_ns)), selected[123] },
    );
}
