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

pub const attention_matching = @import("attention_matching.zig");
pub const attention_match = @import("attention_match.zig");
pub const attention_match_study = @import("attention_match_study.zig");
pub const context_session = @import("context_session.zig");
pub const context_view = @import("context_view.zig");
pub const host_allocation_tracker = @import("host_allocation_tracker.zig");
pub const block = @import("block.zig");
pub const pool = @import("pool.zig");
pub const storage = @import("storage.zig");
pub const block_table = @import("block_table.zig");
pub const manager = @import("manager.zig");
pub const storage_runtime = @import("storage_runtime.zig");
pub const linalg = @import("linalg.zig");
pub const compaction = @import("compaction.zig");
pub const attention_match_checkpoint = @import("attention_match_checkpoint.zig");
pub const attention_match_resume = @import("attention_match_resume.zig");
pub const attention_match_engine = @import("attention_match_engine.zig");
pub const attention_match_device_checkpoint_provider = @import("attention_match_device_checkpoint_provider.zig");
pub const attention_match_e4_formal_authority = @import("attention_match_e4_formal_authority.zig");
pub const attention_match_e4_32512_shape_authority = @import("attention_match_e4_32512_shape_authority.zig");
pub const attention_match_e4_32512_pressure_profile = @import("attention_match_e4_32512_pressure_profile.zig");
pub const attention_match_e4_32512_packed_head_source = @import("attention_match_e4_32512_packed_head_source.zig");
pub const attention_match_e4_32512_metal_head_provider = @import("attention_match_e4_32512_metal_head_provider.zig");
pub const attention_match_e4_32512_source_metadata = @import("attention_match_e4_32512_source_metadata.zig");
pub const attention_match_e4_32512_streaming_source = @import("attention_match_e4_32512_streaming_source.zig");
pub const attention_match_streaming_source_checkpoint_provider = @import("attention_match_streaming_source_checkpoint_provider.zig");
pub const attention_match_e4_32512_resident_authority = @import("attention_match_e4_32512_resident_authority.zig");
pub const attention_match_e4_32512_device_constructor = @import("attention_match_e4_32512_device_constructor.zig");
pub const attention_match_resident_shape_authority = @import("attention_match_resident_shape_authority.zig");
pub const attention_match_whole_fit_input = @import("attention_match_whole_fit_input.zig");
pub const attention_match_training_augmentation = @import("attention_match_training_augmentation.zig");
pub const attention_match_future_augmentation = @import("attention_match_future_augmentation.zig");
pub const attention_match_self_study_training = @import("attention_match_self_study_training.zig");
pub const attention_match_future_decode_training = @import("attention_match_future_decode_training.zig");
pub const attention_match_reference_coverage = @import("attention_match_reference_coverage.zig");
pub const attention_match_reference_coverage_training = @import("attention_match_reference_coverage_training.zig");
pub const attention_match_joint_tiled_geometry = @import("attention_match_joint_tiled_geometry.zig");
pub const attention_match_joint_bounded_tiled_replay = @import("attention_match_joint_bounded_tiled_replay.zig");
pub const attention_match_joint_bounded_page_scoring = @import("attention_match_joint_bounded_page_scoring.zig");
pub const attention_match_joint_bounded_page_replay = @import("attention_match_joint_bounded_page_replay.zig");
pub const attention_match_joint_prepared_input = @import("attention_match_joint_prepared_input.zig");
pub const attention_match_joint_bounded_run = @import("attention_match_joint_bounded_run.zig");
pub const attention_match_joint_train_snapshot = @import("attention_match_joint_train_snapshot.zig");
pub const attention_match_quantized = @import("attention_match_quantized.zig");
pub const attention_match_queries = @import("attention_match_queries.zig");
pub const attention_match_transaction = @import("attention_match_transaction.zig");
pub const attention_match_snapshot = @import("attention_match_snapshot.zig");
pub const device_clone = @import("device_clone.zig");
pub const selection = @import("selection.zig");
pub const turboquant = @import("turboquant.zig");
pub const prompt_cache = @import("prompt_cache.zig");

test {
    _ = attention_matching;
    _ = attention_match;
    _ = attention_match_study;
    _ = context_session;
    _ = context_view;
    _ = host_allocation_tracker;
    _ = block;
    _ = pool;
    _ = storage;
    _ = block_table;
    _ = manager;
    _ = storage_runtime;
    _ = linalg;
    _ = compaction;
    _ = attention_match_checkpoint;
    _ = attention_match_resume;
    _ = attention_match_engine;
    _ = attention_match_device_checkpoint_provider;
    _ = attention_match_e4_formal_authority;
    _ = attention_match_e4_32512_shape_authority;
    _ = attention_match_e4_32512_pressure_profile;
    _ = attention_match_e4_32512_packed_head_source;
    _ = attention_match_e4_32512_metal_head_provider;
    _ = attention_match_e4_32512_source_metadata;
    _ = attention_match_e4_32512_streaming_source;
    _ = attention_match_e4_32512_resident_authority;
    _ = attention_match_e4_32512_device_constructor;
    _ = attention_match_resident_shape_authority;
    _ = attention_match_quantized;
    _ = attention_match_self_study_training;
    _ = attention_match_future_decode_training;
    _ = attention_match_reference_coverage;
    _ = attention_match_reference_coverage_training;
    _ = attention_match_joint_tiled_geometry;
    _ = attention_match_joint_bounded_tiled_replay;
    _ = attention_match_joint_bounded_page_scoring;
    _ = attention_match_joint_bounded_page_replay;
    _ = attention_match_joint_prepared_input;
    _ = attention_match_joint_bounded_run;
    _ = attention_match_joint_train_snapshot;
    _ = attention_match_queries;
    _ = attention_match_transaction;
    _ = attention_match_snapshot;
    _ = device_clone;
    _ = selection;
    _ = turboquant;
    _ = prompt_cache;
}
