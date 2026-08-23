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
const platform = @import("antfly_platform");

// Product entrypoints use a scoped reference instead of mutating process
// environment. The legacy environment switch remains available for internal
// experiments and existing standalone training tools.
var product_enable_refs: std.atomic.Value(u32) = .init(0);

pub const ProductEnableScope = struct {
    active: bool = true,

    pub fn acquire() ProductEnableScope {
        const previous = product_enable_refs.fetchAdd(1, .acq_rel);
        std.debug.assert(previous != std.math.maxInt(u32));
        return .{};
    }

    pub fn deinit(self: *ProductEnableScope) void {
        if (!self.active) return;
        const previous = product_enable_refs.fetchSub(1, .acq_rel);
        std.debug.assert(previous > 0);
        self.active = false;
    }
};

pub fn productEnabled() bool {
    return product_enable_refs.load(.acquire) != 0;
}

pub fn enabled() bool {
    if (platform.env.getenvBoolDefault("TERMITE_DISABLE_TRAINING_GRAPH_EXECUTOR", false)) return false;
    return productEnabled() or
        platform.env.getenvBoolDefault("TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR", false);
}

test "product training executor enablement is scoped and nestable" {
    try std.testing.expect(!productEnabled());
    var outer = ProductEnableScope.acquire();
    defer outer.deinit();
    try std.testing.expect(productEnabled());

    var inner = ProductEnableScope.acquire();
    try std.testing.expect(productEnabled());
    inner.deinit();
    try std.testing.expect(productEnabled());
}
