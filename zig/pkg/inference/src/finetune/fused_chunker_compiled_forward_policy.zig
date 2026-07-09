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

//! Pure decision logic for the compiled segmented eval forward
//! (fused_chunker_compiled_forward.zig). Kept import-free so it can run
//! under a standalone unit-test target
//! (`zig build test-fused-chunker-compiled-forward`) without pulling the
//! full backend/graph transitive test closure into the test binary.

const std = @import("std");

/// ModernBERT-base head count; the fused chunker encoder is not configurable
/// on this axis (matches the eager `modern_bert.Config` default and the
/// trainer's compiled step-forward config).
pub const eval_num_attention_heads: u32 = 12;

/// Per-head dimension for a given hidden size, or error when the hidden size
/// is incompatible with the fixed ModernBERT head count.
pub fn headDimForHiddenSize(hidden_size: u32) !u32 {
    if (hidden_size == 0 or hidden_size % eval_num_attention_heads != 0) return error.InvalidHeadDim;
    return hidden_size / eval_num_attention_heads;
}

/// Whether a given eval batch should take the compiled path.
///
/// Segment sessions are compiled for a fixed [batch, seq] shape. A partial
/// trailing batch would force a recompile of every segment session and a
/// second recompile on the next full batch, so partial batches always run
/// the eager forward instead.
pub fn shouldUseCompiledForBatch(
    failed: bool,
    lora_rank: u32,
    batch_count: usize,
    full_batch_size: usize,
) bool {
    if (failed) return false;
    // The segment graph builder injects LoRA unconditionally; without
    // adapters the eager forward is the canonical path.
    if (lora_rank == 0) return false;
    if (batch_count == 0) return false;
    return batch_count == full_batch_size;
}

/// Segment size normalization: 0 means "one layer per segment".
pub fn normalizedLayersPerSegment(layers_per_segment: u32) u32 {
    return if (layers_per_segment == 0) 1 else layers_per_segment;
}

// ---------------------------------------------------------------------------
// Serving (fused_chunking pipeline) policy
// ---------------------------------------------------------------------------

/// Serving kill-switch parse for ANTFLY_FUSED_CHUNKER_SERVING_COMPILED_FORWARD.
///
/// Serving defaults the compiled segment forward ON when the pipeline runs on
/// the Metal backend, so the env var is a kill switch: unset means enabled,
/// and "0" / "false" / "no" / "off" / empty force the eager forward. Mirrors
/// platform.env.getenvBoolDefault(_, true) semantics without the import so it
/// stays testable in this standalone target.
pub fn servingCompiledForwardEnabled(env_value: ?[]const u8) bool {
    const value = env_value orelse return true;
    if (value.len == 0) return false;
    if (std.mem.eql(u8, value, "0")) return false;
    if (std.ascii.eqlIgnoreCase(value, "false")) return false;
    if (std.ascii.eqlIgnoreCase(value, "no")) return false;
    if (std.ascii.eqlIgnoreCase(value, "off")) return false;
    return true;
}

/// Whether one serving forward window should take the compiled path.
///
/// Serving runs batch=1 windows of exactly max_seq_len tokens, plus one
/// ragged final window. Segment sessions are compiled for a fixed
/// [1, max_seq_len] geometry, so the ragged window runs the eager forward
/// instead (same session-recompile-thrash rule as eval's partial trailing
/// batches). The first-failure latch always wins.
pub fn shouldUseCompiledForServingWindow(
    failed: bool,
    window_tokens: usize,
    max_seq_len: usize,
) bool {
    if (failed) return false;
    if (window_tokens == 0) return false;
    return window_tokens == max_seq_len;
}

test "headDimForHiddenSize matches ModernBERT-base and rejects bad sizes" {
    try std.testing.expectEqual(@as(u32, 64), try headDimForHiddenSize(768));
    try std.testing.expectEqual(@as(u32, 32), try headDimForHiddenSize(384));
    try std.testing.expectError(error.InvalidHeadDim, headDimForHiddenSize(770));
    try std.testing.expectError(error.InvalidHeadDim, headDimForHiddenSize(0));
}

test "shouldUseCompiledForBatch gates on failure, rank, and full batches" {
    // Happy path: full batch, LoRA present, no prior failure.
    try std.testing.expect(shouldUseCompiledForBatch(false, 8, 32, 32));
    // First-failure latch wins.
    try std.testing.expect(!shouldUseCompiledForBatch(true, 8, 32, 32));
    // No LoRA adapters -> eager path.
    try std.testing.expect(!shouldUseCompiledForBatch(false, 0, 32, 32));
    // Partial trailing batch -> eager path (avoids session recompile thrash).
    try std.testing.expect(!shouldUseCompiledForBatch(false, 8, 7, 32));
    try std.testing.expect(!shouldUseCompiledForBatch(false, 8, 0, 32));
}

test "normalizedLayersPerSegment maps zero to one" {
    try std.testing.expectEqual(@as(u32, 1), normalizedLayersPerSegment(0));
    try std.testing.expectEqual(@as(u32, 1), normalizedLayersPerSegment(1));
    try std.testing.expectEqual(@as(u32, 4), normalizedLayersPerSegment(4));
}

test "servingCompiledForwardEnabled defaults on and honors the kill switch" {
    // Unset -> default ON.
    try std.testing.expect(servingCompiledForwardEnabled(null));
    // Explicit truthy values keep it on.
    try std.testing.expect(servingCompiledForwardEnabled("1"));
    try std.testing.expect(servingCompiledForwardEnabled("true"));
    try std.testing.expect(servingCompiledForwardEnabled("on"));
    // Kill-switch spellings force the eager forward.
    try std.testing.expect(!servingCompiledForwardEnabled("0"));
    try std.testing.expect(!servingCompiledForwardEnabled(""));
    try std.testing.expect(!servingCompiledForwardEnabled("false"));
    try std.testing.expect(!servingCompiledForwardEnabled("FALSE"));
    try std.testing.expect(!servingCompiledForwardEnabled("no"));
    try std.testing.expect(!servingCompiledForwardEnabled("off"));
    try std.testing.expect(!servingCompiledForwardEnabled("Off"));
}

test "shouldUseCompiledForServingWindow gates on latch and full windows" {
    // Full window, no prior failure: compiled.
    try std.testing.expect(shouldUseCompiledForServingWindow(false, 384, 384));
    // First-failure latch wins.
    try std.testing.expect(!shouldUseCompiledForServingWindow(true, 384, 384));
    // Ragged final window -> eager (fixed-geometry sessions).
    try std.testing.expect(!shouldUseCompiledForServingWindow(false, 383, 384));
    try std.testing.expect(!shouldUseCompiledForServingWindow(false, 1, 384));
    try std.testing.expect(!shouldUseCompiledForServingWindow(false, 0, 384));
    // Oversized windows never happen, but must not take the compiled path.
    try std.testing.expect(!shouldUseCompiledForServingWindow(false, 385, 384));
}
