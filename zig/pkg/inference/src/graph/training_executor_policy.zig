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
// Scopes belong to the synchronous training caller. Other request threads
// must not inherit its execution policy. Capture environment admission once
// at the outermost scope, rather than reading it for every graph operation.
threadlocal var product_enable_refs: u32 = 0;
threadlocal var scoped_enabled: bool = false;

pub const ProductEnableScope = struct {
    active: bool = true,

    pub fn acquire() ProductEnableScope {
        std.debug.assert(product_enable_refs != std.math.maxInt(u32));
        if (product_enable_refs == 0)
            scoped_enabled = !platform.env.getenvBoolDefault("TERMITE_DISABLE_TRAINING_GRAPH_EXECUTOR", false);
        product_enable_refs += 1;
        return .{};
    }

    pub fn deinit(self: *ProductEnableScope) void {
        if (!self.active) return;
        std.debug.assert(product_enable_refs > 0);
        product_enable_refs -= 1;
        self.active = false;
    }
};

pub fn productEnabled() bool {
    return product_enable_refs != 0;
}

pub fn enabled() bool {
    if (productEnabled()) return scoped_enabled;
    if (platform.env.getenvBoolDefault("TERMITE_DISABLE_TRAINING_GRAPH_EXECUTOR", false)) return false;
    return platform.env.getenvBoolDefault("TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR", false);
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

test "gemma4 training executor scopes do not cross request threads" {
    var outer = ProductEnableScope.acquire();
    defer outer.deinit();
    const Worker = struct {
        fn run() void {
            std.debug.assert(!productEnabled());
            var local = ProductEnableScope.acquire();
            std.debug.assert(productEnabled());
            local.deinit();
            std.debug.assert(!productEnabled());
        }
    };
    const worker = try std.Thread.spawn(.{}, Worker.run, .{});
    worker.join();
    try std.testing.expect(productEnabled());
}
