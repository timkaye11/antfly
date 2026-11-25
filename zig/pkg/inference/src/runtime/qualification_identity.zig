// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Process-wide identities for qualification-only context and capture
//! authorities. Every producer shares this counter so independently extracted
//! runtime modules cannot issue colliding identities in one process.
const std = @import("std");
const builtin = @import("builtin");

var next_identity: if (builtin.is_test) std.atomic.Value(u64) else void = if (builtin.is_test) std.atomic.Value(u64).init(1) else {};

pub fn allocate() !u64 {
    if (comptime !builtin.is_test) return error.QualificationOnly;
    var current = next_identity.load(.monotonic);
    while (true) {
        if (current == 0 or current == std.math.maxInt(u64))
            return error.QualificationFingerprintOverflow;
        if (next_identity.cmpxchgWeak(current, current + 1, .monotonic, .monotonic)) |observed| {
            current = observed;
            continue;
        }
        return current;
    }
}
