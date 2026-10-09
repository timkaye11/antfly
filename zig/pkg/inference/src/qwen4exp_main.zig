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

//! Focused CLI for the experimental SSD decoder, sharing generate's parser.
const std = @import("std");
const platform = @import("antfly_platform");

pub fn main(init: std.process.Init) !void {
    const allocator = platform.allocator.processAllocator(std.heap.smp_allocator);
    var iterator = std.process.Args.Iterator.init(init.minimal.args);
    _ = iterator.next();
    var arguments: [64][]const u8 = undefined;
    var count: usize = 0;
    while (iterator.next()) |argument| {
        if (count == arguments.len) return error.TooManyArguments;
        arguments[count] = argument;
        count += 1;
    }
    try @import("inference").native_generate.qwen4expMain(allocator, init.io, arguments[0..count]);
}
