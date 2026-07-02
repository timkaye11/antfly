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
