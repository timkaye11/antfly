// Copyright 2026 Antfly, Inc.
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

const backends = @import("backends/backends.zig");
const ops = @import("ops/ops.zig");
const session_factory = @import("architectures/session_factory.zig");

pub const Stats = struct {
    q2_k: u64 = 0,
    q2_k_bias: u64 = 0,
    q2_k_bias_gelu: u64 = 0,
    q3_k: u64 = 0,
    q3_k_bias: u64 = 0,
    q3_k_bias_gelu: u64 = 0,
    q4_0: u64 = 0,
    q4_1: u64 = 0,
    q5_0: u64 = 0,
    q5_1: u64 = 0,
    q4_k: u64 = 0,
    q4_k_bias: u64 = 0,
    q4_k_bias_gelu: u64 = 0,
    q5_k: u64 = 0,
    q5_k_bias: u64 = 0,
    q5_k_bias_gelu: u64 = 0,
    q6_k: u64 = 0,
    q6_k_bias: u64 = 0,
    q6_k_bias_gelu: u64 = 0,
    q8_0: u64 = 0,
    q8_0_bias: u64 = 0,
    q8_0_bias_gelu: u64 = 0,
    q8_0_relu: u64 = 0,
    q8_1: u64 = 0,
    q8_k: u64 = 0,
    q4_k_rows_1: u64 = 0,
    q4_k_rows_2_8: u64 = 0,
    q4_k_rows_9_64: u64 = 0,
    q4_k_rows_65_plus: u64 = 0,
    q6_k_rows_1: u64 = 0,
    q6_k_rows_2_8: u64 = 0,
    q6_k_rows_9_64: u64 = 0,
    q6_k_rows_65_plus: u64 = 0,
    q8_0_rows_1: u64 = 0,
    q8_0_rows_2_8: u64 = 0,
    q8_0_rows_9_64: u64 = 0,
    q8_0_rows_65_plus: u64 = 0,

    pub fn generatedTotal(self: Stats) u64 {
        return self.q2_k +
            self.q2_k_bias +
            self.q2_k_bias_gelu +
            self.q3_k +
            self.q3_k_bias +
            self.q3_k_bias_gelu +
            self.q4_0 +
            self.q4_1 +
            self.q5_0 +
            self.q5_1 +
            self.q4_k +
            self.q4_k_bias +
            self.q4_k_bias_gelu +
            self.q5_k +
            self.q5_k_bias +
            self.q5_k_bias_gelu +
            self.q6_k +
            self.q6_k_bias +
            self.q6_k_bias_gelu +
            self.q8_0 +
            self.q8_0_bias +
            self.q8_0_bias_gelu +
            self.q8_0_relu +
            self.q8_1 +
            self.q8_k;
    }

    pub fn add(lhs: Stats, rhs: Stats) Stats {
        return .{
            .q2_k = lhs.q2_k + rhs.q2_k,
            .q2_k_bias = lhs.q2_k_bias + rhs.q2_k_bias,
            .q2_k_bias_gelu = lhs.q2_k_bias_gelu + rhs.q2_k_bias_gelu,
            .q3_k = lhs.q3_k + rhs.q3_k,
            .q3_k_bias = lhs.q3_k_bias + rhs.q3_k_bias,
            .q3_k_bias_gelu = lhs.q3_k_bias_gelu + rhs.q3_k_bias_gelu,
            .q4_0 = lhs.q4_0 + rhs.q4_0,
            .q4_1 = lhs.q4_1 + rhs.q4_1,
            .q5_0 = lhs.q5_0 + rhs.q5_0,
            .q5_1 = lhs.q5_1 + rhs.q5_1,
            .q4_k = lhs.q4_k + rhs.q4_k,
            .q4_k_bias = lhs.q4_k_bias + rhs.q4_k_bias,
            .q4_k_bias_gelu = lhs.q4_k_bias_gelu + rhs.q4_k_bias_gelu,
            .q5_k = lhs.q5_k + rhs.q5_k,
            .q5_k_bias = lhs.q5_k_bias + rhs.q5_k_bias,
            .q5_k_bias_gelu = lhs.q5_k_bias_gelu + rhs.q5_k_bias_gelu,
            .q6_k = lhs.q6_k + rhs.q6_k,
            .q6_k_bias = lhs.q6_k_bias + rhs.q6_k_bias,
            .q6_k_bias_gelu = lhs.q6_k_bias_gelu + rhs.q6_k_bias_gelu,
            .q8_0 = lhs.q8_0 + rhs.q8_0,
            .q8_0_bias = lhs.q8_0_bias + rhs.q8_0_bias,
            .q8_0_bias_gelu = lhs.q8_0_bias_gelu + rhs.q8_0_bias_gelu,
            .q8_0_relu = lhs.q8_0_relu + rhs.q8_0_relu,
            .q8_1 = lhs.q8_1 + rhs.q8_1,
            .q8_k = lhs.q8_k + rhs.q8_k,
            .q4_k_rows_1 = lhs.q4_k_rows_1 + rhs.q4_k_rows_1,
            .q4_k_rows_2_8 = lhs.q4_k_rows_2_8 + rhs.q4_k_rows_2_8,
            .q4_k_rows_9_64 = lhs.q4_k_rows_9_64 + rhs.q4_k_rows_9_64,
            .q4_k_rows_65_plus = lhs.q4_k_rows_65_plus + rhs.q4_k_rows_65_plus,
            .q6_k_rows_1 = lhs.q6_k_rows_1 + rhs.q6_k_rows_1,
            .q6_k_rows_2_8 = lhs.q6_k_rows_2_8 + rhs.q6_k_rows_2_8,
            .q6_k_rows_9_64 = lhs.q6_k_rows_9_64 + rhs.q6_k_rows_9_64,
            .q6_k_rows_65_plus = lhs.q6_k_rows_65_plus + rhs.q6_k_rows_65_plus,
            .q8_0_rows_1 = lhs.q8_0_rows_1 + rhs.q8_0_rows_1,
            .q8_0_rows_2_8 = lhs.q8_0_rows_2_8 + rhs.q8_0_rows_2_8,
            .q8_0_rows_9_64 = lhs.q8_0_rows_9_64 + rhs.q8_0_rows_9_64,
            .q8_0_rows_65_plus = lhs.q8_0_rows_65_plus + rhs.q8_0_rows_65_plus,
        };
    }

    pub fn diff(before: Stats, after: Stats) Stats {
        return .{
            .q2_k = counterDelta(before.q2_k, after.q2_k),
            .q2_k_bias = counterDelta(before.q2_k_bias, after.q2_k_bias),
            .q2_k_bias_gelu = counterDelta(before.q2_k_bias_gelu, after.q2_k_bias_gelu),
            .q3_k = counterDelta(before.q3_k, after.q3_k),
            .q3_k_bias = counterDelta(before.q3_k_bias, after.q3_k_bias),
            .q3_k_bias_gelu = counterDelta(before.q3_k_bias_gelu, after.q3_k_bias_gelu),
            .q4_0 = counterDelta(before.q4_0, after.q4_0),
            .q4_1 = counterDelta(before.q4_1, after.q4_1),
            .q5_0 = counterDelta(before.q5_0, after.q5_0),
            .q5_1 = counterDelta(before.q5_1, after.q5_1),
            .q4_k = counterDelta(before.q4_k, after.q4_k),
            .q4_k_bias = counterDelta(before.q4_k_bias, after.q4_k_bias),
            .q4_k_bias_gelu = counterDelta(before.q4_k_bias_gelu, after.q4_k_bias_gelu),
            .q5_k = counterDelta(before.q5_k, after.q5_k),
            .q5_k_bias = counterDelta(before.q5_k_bias, after.q5_k_bias),
            .q5_k_bias_gelu = counterDelta(before.q5_k_bias_gelu, after.q5_k_bias_gelu),
            .q6_k = counterDelta(before.q6_k, after.q6_k),
            .q6_k_bias = counterDelta(before.q6_k_bias, after.q6_k_bias),
            .q6_k_bias_gelu = counterDelta(before.q6_k_bias_gelu, after.q6_k_bias_gelu),
            .q8_0 = counterDelta(before.q8_0, after.q8_0),
            .q8_0_bias = counterDelta(before.q8_0_bias, after.q8_0_bias),
            .q8_0_bias_gelu = counterDelta(before.q8_0_bias_gelu, after.q8_0_bias_gelu),
            .q8_0_relu = counterDelta(before.q8_0_relu, after.q8_0_relu),
            .q8_1 = counterDelta(before.q8_1, after.q8_1),
            .q8_k = counterDelta(before.q8_k, after.q8_k),
            .q4_k_rows_1 = counterDelta(before.q4_k_rows_1, after.q4_k_rows_1),
            .q4_k_rows_2_8 = counterDelta(before.q4_k_rows_2_8, after.q4_k_rows_2_8),
            .q4_k_rows_9_64 = counterDelta(before.q4_k_rows_9_64, after.q4_k_rows_9_64),
            .q4_k_rows_65_plus = counterDelta(before.q4_k_rows_65_plus, after.q4_k_rows_65_plus),
            .q6_k_rows_1 = counterDelta(before.q6_k_rows_1, after.q6_k_rows_1),
            .q6_k_rows_2_8 = counterDelta(before.q6_k_rows_2_8, after.q6_k_rows_2_8),
            .q6_k_rows_9_64 = counterDelta(before.q6_k_rows_9_64, after.q6_k_rows_9_64),
            .q6_k_rows_65_plus = counterDelta(before.q6_k_rows_65_plus, after.q6_k_rows_65_plus),
            .q8_0_rows_1 = counterDelta(before.q8_0_rows_1, after.q8_0_rows_1),
            .q8_0_rows_2_8 = counterDelta(before.q8_0_rows_2_8, after.q8_0_rows_2_8),
            .q8_0_rows_9_64 = counterDelta(before.q8_0_rows_9_64, after.q8_0_rows_9_64),
            .q8_0_rows_65_plus = counterDelta(before.q8_0_rows_65_plus, after.q8_0_rows_65_plus),
        };
    }

    pub fn fromProviderStats(stats: ops.NativeQuantTimingStats) Stats {
        return .{
            .q2_k = stats.metal_runtime_antfly_q2_k_small_batch_dispatches,
            .q2_k_bias = stats.metal_runtime_antfly_q2_k_small_batch_bias_dispatches,
            .q2_k_bias_gelu = stats.metal_runtime_antfly_q2_k_small_batch_bias_gelu_dispatches,
            .q3_k = stats.metal_runtime_antfly_q3_k_small_batch_dispatches,
            .q3_k_bias = stats.metal_runtime_antfly_q3_k_small_batch_bias_dispatches,
            .q3_k_bias_gelu = stats.metal_runtime_antfly_q3_k_small_batch_bias_gelu_dispatches,
            .q4_0 = stats.metal_runtime_antfly_q4_0_small_batch_dispatches,
            .q4_1 = stats.metal_runtime_antfly_q4_1_small_batch_dispatches,
            .q5_0 = stats.metal_runtime_antfly_q5_0_small_batch_dispatches,
            .q5_1 = stats.metal_runtime_antfly_q5_1_small_batch_dispatches,
            .q4_k = stats.metal_runtime_antfly_q4_k_small_batch_dispatches,
            .q4_k_bias = stats.metal_runtime_antfly_q4_k_small_batch_bias_dispatches,
            .q4_k_bias_gelu = stats.metal_runtime_antfly_q4_k_small_batch_bias_gelu_dispatches,
            .q5_k = stats.metal_runtime_antfly_q5_k_small_batch_dispatches,
            .q5_k_bias = stats.metal_runtime_antfly_q5_k_small_batch_bias_dispatches,
            .q5_k_bias_gelu = stats.metal_runtime_antfly_q5_k_small_batch_bias_gelu_dispatches,
            .q6_k = stats.metal_runtime_antfly_q6_k_small_batch_dispatches,
            .q6_k_bias = stats.metal_runtime_antfly_q6_k_small_batch_bias_dispatches,
            .q6_k_bias_gelu = stats.metal_runtime_antfly_q6_k_small_batch_bias_gelu_dispatches,
            .q8_0 = stats.metal_runtime_antfly_q8_0_small_batch_dispatches,
            .q8_0_bias = stats.metal_runtime_antfly_q8_0_small_batch_bias_dispatches,
            .q8_0_bias_gelu = stats.metal_runtime_antfly_q8_0_small_batch_bias_gelu_dispatches,
            .q8_0_relu = stats.metal_runtime_antfly_q8_0_small_batch_relu_dispatches,
            .q8_1 = stats.metal_runtime_antfly_q8_1_small_batch_dispatches,
            .q8_k = stats.metal_runtime_antfly_q8_k_small_batch_dispatches,
            .q4_k_rows_1 = stats.metal_runtime_q4_k_linear_reduce_rows_1,
            .q4_k_rows_2_8 = stats.metal_runtime_q4_k_linear_reduce_rows_2_8,
            .q4_k_rows_9_64 = stats.metal_runtime_q4_k_linear_reduce_rows_9_64,
            .q4_k_rows_65_plus = stats.metal_runtime_q4_k_linear_reduce_rows_65_plus,
            .q6_k_rows_1 = stats.metal_runtime_q6_k_linear_reduce_rows_1,
            .q6_k_rows_2_8 = stats.metal_runtime_q6_k_linear_reduce_rows_2_8,
            .q6_k_rows_9_64 = stats.metal_runtime_q6_k_linear_reduce_rows_9_64,
            .q6_k_rows_65_plus = stats.metal_runtime_q6_k_linear_reduce_rows_65_plus,
            .q8_0_rows_1 = stats.metal_runtime_q8_0_linear_rows_1,
            .q8_0_rows_2_8 = stats.metal_runtime_q8_0_linear_rows_2_8,
            .q8_0_rows_9_64 = stats.metal_runtime_q8_0_linear_rows_9_64,
            .q8_0_rows_65_plus = stats.metal_runtime_q8_0_linear_rows_65_plus,
        };
    }
};

pub fn snapshotForSession(
    allocator: std.mem.Allocator,
    session: backends.Session,
) Stats {
    var cb = session_factory.getComputeBackend(session, allocator) catch return .{};
    defer cb.deinit();
    const snapshot = cb.debugTimingSnapshot();
    return Stats.fromProviderStats(snapshot.provider);
}

fn counterDelta(before: u64, after: u64) u64 {
    return if (after >= before) after - before else 0;
}

test "quant kernel compiler metal generated quant stats total covers every generated provider counter" {
    const provider = ops.NativeQuantTimingStats{
        .metal_runtime_antfly_q2_k_small_batch_dispatches = 1,
        .metal_runtime_antfly_q2_k_small_batch_bias_dispatches = 1,
        .metal_runtime_antfly_q2_k_small_batch_bias_gelu_dispatches = 1,
        .metal_runtime_antfly_q3_k_small_batch_dispatches = 1,
        .metal_runtime_antfly_q3_k_small_batch_bias_dispatches = 1,
        .metal_runtime_antfly_q3_k_small_batch_bias_gelu_dispatches = 1,
        .metal_runtime_antfly_q4_0_small_batch_dispatches = 1,
        .metal_runtime_antfly_q4_1_small_batch_dispatches = 1,
        .metal_runtime_antfly_q5_0_small_batch_dispatches = 1,
        .metal_runtime_antfly_q5_1_small_batch_dispatches = 1,
        .metal_runtime_antfly_q4_k_small_batch_dispatches = 1,
        .metal_runtime_antfly_q4_k_small_batch_bias_dispatches = 1,
        .metal_runtime_antfly_q4_k_small_batch_bias_gelu_dispatches = 1,
        .metal_runtime_antfly_q5_k_small_batch_dispatches = 1,
        .metal_runtime_antfly_q5_k_small_batch_bias_dispatches = 1,
        .metal_runtime_antfly_q5_k_small_batch_bias_gelu_dispatches = 1,
        .metal_runtime_antfly_q6_k_small_batch_dispatches = 1,
        .metal_runtime_antfly_q6_k_small_batch_bias_dispatches = 1,
        .metal_runtime_antfly_q6_k_small_batch_bias_gelu_dispatches = 1,
        .metal_runtime_antfly_q8_0_small_batch_dispatches = 1,
        .metal_runtime_antfly_q8_0_small_batch_bias_dispatches = 1,
        .metal_runtime_antfly_q8_0_small_batch_bias_gelu_dispatches = 1,
        .metal_runtime_antfly_q8_0_small_batch_relu_dispatches = 1,
        .metal_runtime_antfly_q8_1_small_batch_dispatches = 1,
        .metal_runtime_antfly_q8_k_small_batch_dispatches = 1,
    };

    const stats = Stats.fromProviderStats(provider);
    try std.testing.expectEqual(@as(u64, 25), stats.generatedTotal());
    try std.testing.expectEqual(@as(u64, 25), Stats.diff(.{}, stats).generatedTotal());
    try std.testing.expectEqual(@as(u64, 50), stats.add(stats).generatedTotal());
}
