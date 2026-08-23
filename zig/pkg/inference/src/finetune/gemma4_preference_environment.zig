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

//! Typed admission for environment controls reachable from Gemma4 preference
//! training. The source policy is shared with the Python qualification tools.

const std = @import("std");

pub const source = @embedFile("gemma4_preference_environment.policy");

const ParsedLine = struct {
    directive: []const u8,
    name: []const u8,
    first_argument: ?[]const u8,
    second_argument: ?[]const u8,
};

fn expectedArgumentCount(directive: []const u8) usize {
    if (std.mem.eql(u8, directive, "scope-prefix") or
        std.mem.eql(u8, directive, "sanitize-prefix") or
        std.mem.eql(u8, directive, "sanitize-exact") or
        std.mem.eql(u8, directive, "deny") or
        std.mem.eql(u8, directive, "allow-bool") or
        std.mem.eql(u8, directive, "allow-presence") or
        std.mem.eql(u8, directive, "allow-uint")) return 0;
    if (std.mem.eql(u8, directive, "strict") or
        std.mem.eql(u8, directive, "allow-fixed")) return 1;
    if (std.mem.eql(u8, directive, "allow-uint-range")) return 2;
    unreachable;
}

fn parseLine(raw_line: []const u8) ?ParsedLine {
    const line = std.mem.trim(u8, raw_line, " \t\r");
    if (line.len == 0 or line[0] == '#') return null;

    var tokens = std.mem.tokenizeAny(u8, line, " \t\r");
    const directive = tokens.next() orelse unreachable;
    const name = tokens.next() orelse unreachable;
    const first_argument = tokens.next();
    const second_argument = tokens.next();
    if (tokens.next() != null) unreachable;
    const actual_argument_count = @as(usize, @intFromBool(first_argument != null)) +
        @as(usize, @intFromBool(second_argument != null));
    if (actual_argument_count != expectedArgumentCount(directive)) unreachable;
    return .{
        .directive = directive,
        .name = name,
        .first_argument = first_argument,
        .second_argument = second_argument,
    };
}

fn isAllowDirective(directive: []const u8) bool {
    return std.mem.eql(u8, directive, "allow-bool") or
        std.mem.eql(u8, directive, "allow-presence") or
        std.mem.eql(u8, directive, "allow-uint") or
        std.mem.eql(u8, directive, "allow-uint-range") or
        std.mem.eql(u8, directive, "allow-fixed");
}

fn isAdmissionDirective(directive: []const u8) bool {
    return isAllowDirective(directive) or std.mem.eql(u8, directive, "deny");
}

pub fn nameInScope(name: []const u8) bool {
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |raw_line| {
        const line = parseLine(raw_line) orelse continue;
        if (std.mem.eql(u8, line.directive, "scope-prefix")) {
            if (std.mem.startsWith(u8, name, line.name)) return true;
            continue;
        }
        if (isAdmissionDirective(line.directive) and std.mem.eql(u8, name, line.name)) return true;
    }
    return false;
}

pub fn nameAllowed(name: []const u8) bool {
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |raw_line| {
        const line = parseLine(raw_line) orelse continue;
        if (isAllowDirective(line.directive) and std.mem.eql(u8, name, line.name)) return true;
    }
    return false;
}

pub fn valueIsCanonical(name: []const u8, value: []const u8) bool {
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |raw_line| {
        const line = parseLine(raw_line) orelse continue;
        if (!std.mem.eql(u8, name, line.name) or !isAllowDirective(line.directive)) continue;

        if (std.mem.eql(u8, line.directive, "allow-bool")) {
            return std.mem.eql(u8, value, "0") or std.mem.eql(u8, value, "1");
        }
        if (std.mem.eql(u8, line.directive, "allow-presence")) {
            return std.mem.eql(u8, value, "1");
        }
        if (std.mem.eql(u8, line.directive, "allow-fixed")) {
            return std.mem.eql(u8, value, line.first_argument orelse unreachable);
        }
        const parsed = std.fmt.parseUnsigned(usize, value, 10) catch return false;
        if (std.mem.eql(u8, line.directive, "allow-uint")) return true;
        const minimum = std.fmt.parseUnsigned(usize, line.first_argument orelse unreachable, 10) catch unreachable;
        const maximum = std.fmt.parseUnsigned(usize, line.second_argument orelse unreachable, 10) catch unreachable;
        return parsed >= minimum and parsed <= maximum;
    }
    return false;
}

test "preference environment policy denies correctness kill switches" {
    for ([_][]const u8{
        "TERMITE_DISABLE_GRAPH_OUTPUT_OWNED_COPY",
        "TERMITE_DISABLE_GRAPH_OUTPUT_ELISION_OVERRIDE",
        "TERMITE_DISABLE_OUTPUT_HOST_MIRROR_RESYNC",
        "TERMITE_DISABLE_PAGED_KV",
    }) |name| {
        try std.testing.expect(nameInScope(name));
        try std.testing.expect(!nameAllowed(name));
        try std.testing.expect(!valueIsCanonical(name, "0"));
        try std.testing.expect(!valueIsCanonical(name, "1"));
    }
}

test "preference environment policy preserves typed canonical values" {
    try std.testing.expect(nameInScope("TERMITE_METAL_DISABLE_FUTURE_TRAINING_FUSION"));
    try std.testing.expect(!nameAllowed("TERMITE_METAL_DISABLE_FUTURE_TRAINING_FUSION"));
    try std.testing.expect(nameInScope("ANTFLY_GEMMA4_GRPO_FUTURE_BATCH_ROUTE"));
    try std.testing.expect(!nameAllowed("ANTFLY_GEMMA4_GRPO_FUTURE_BATCH_ROUTE"));
    try std.testing.expect(nameInScope("ANTFLY_GEMMA4_PREFERENCE_TRACE"));
    try std.testing.expect(!nameAllowed("ANTFLY_GEMMA4_PREFERENCE_TRACE"));
    try std.testing.expect(!nameInScope("HF_HOME"));

    try std.testing.expect(valueIsCanonical("TERMITE_METAL_DISABLE_BF16_SIMDGROUP_M64", "0"));
    try std.testing.expect(!valueIsCanonical("TERMITE_METAL_DISABLE_BF16_SIMDGROUP_M64", "false"));
    try std.testing.expect(valueIsCanonical("TERMITE_METAL_DISABLE_GEMMA_GQA_ATTENTION_FUSION", "1"));
    try std.testing.expect(!valueIsCanonical("TERMITE_METAL_DISABLE_GEMMA_GQA_ATTENTION_FUSION", "0"));
    try std.testing.expect(valueIsCanonical("TERMITE_GEMMA4_SPARSE_LOSS_CHUNK_ROWS", "512"));
    try std.testing.expect(!valueIsCanonical("TERMITE_GEMMA4_SPARSE_LOSS_CHUNK_ROWS", "513"));
    try std.testing.expect(valueIsCanonical("TERMITE_DEBUG_DEVICE_GRAD_NORM", "0"));
    try std.testing.expect(!valueIsCanonical("TERMITE_DEBUG_DEVICE_GRAD_NORM", "1"));
}

test "every allowed or strict policy entry is scoped and canonical" {
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |raw_line| {
        const line = parseLine(raw_line) orelse continue;
        if (isAllowDirective(line.directive)) {
            try std.testing.expect(nameInScope(line.name));
        } else if (std.mem.eql(u8, line.directive, "strict")) {
            try std.testing.expect(nameAllowed(line.name));
            try std.testing.expect(valueIsCanonical(line.name, line.first_argument orelse unreachable));
        }
    }
}
