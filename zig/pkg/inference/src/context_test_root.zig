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

test {
    _ = @import("runtime/kv/attention_match_train_critical_policy_probe.zig");
    _ = @import("runtime/kv/attention_match_primary_train_candidate.zig");
    _ = @import("runtime/kv/attention_match_fp64_reference_summary_cache.zig");
    _ = @import("runtime/kv/attention_match_train_domain_cv_split.zig");
    _ = @import("runtime/kv/attention_match_train_l2_surrogate_minimax_postfit.zig");
    _ = @import("runtime/kv/attention_match_native_historical_coverage_bridge.zig");
    _ = @import("runtime/kv/attention_match_native_coverage_data_adapter.zig");
    _ = @import("runtime/kv/attention_match_train_challenge_historical_selection.zig");
    _ = @import("runtime/kv/attention_match_train_challenge_hybrid_selection.zig");
    _ = @import("runtime/kv/attention_match_train_historical_anchor_authority.zig");
    _ = @import("runtime/kv/attention_match_flat_shape_authority.zig");
    _ = @import("runtime/kv/attention_match_planned_exact_context.zig");
    _ = @import("runtime/kv/attention_match_topology_physical_proposal.zig");
    _ = @import("runtime/kv/attention_match_resident_native_hybrid_profile.zig");
    _ = @import("runtime/kv/attention_match_native_resident_hybrid_bridge.zig");
    _ = @import("context_resident_hybrid_staged_configuration.zig");
    _ = @import("context_resident_hybrid_compatibility_profile.zig");
    _ = @import("context_native_resident_core_replay.zig");
    _ = @import("runtime/kv/attention_match_e4_flat8128_pressure_profile.zig");
    _ = @import("runtime/kv/attention_match_train_challenge_pilot_adapter.zig");
    _ = @import("runtime/kv/attention_match_checkpoint_phase_loader.zig");
    _ = @import("runtime/kv/attention_match_production768_train_challenge_rewrite.zig");
    _ = @import("context_native_coverage_probability_benchmark_v2.zig");
    _ = @import("context_native_coverage_probability_benchmark_all8.zig");
    _ = @import("runtime/kv/attention_match_accelerate_fp64_probability_scan.zig");
    _ = @import("runtime/kv/attention_match_native_coverage_training_merge.zig");
    _ = @import("runtime/kv/attention_match_train_surrogate_partition_minimax.zig");
    _ = @import("runtime/kv/attention_match_train32_augmentation_reader.zig");
    _ = @import("runtime/kv/attention_match_case_replay_native.zig");
    _ = @import("runtime/kv/attention_match_joint_adaptive_quality.zig");
    _ = @import("runtime/kv/attention_match_physical_budget.zig");
    _ = @import("runtime/kv/attention_match_bounded_page_delta_accelerate.zig");
    _ = @import("runtime/context_policy.zig");
    _ = @import("runtime/kv/payload.zig");
    _ = @import("runtime/kv/compaction_profile.zig");
    _ = @import("backends/compaction_coordinator.zig");
    _ = @import("backends/context_stored_all36_driver.zig");
    _ = @import("backends/context_stored_all36_coordinator_fixture.zig");
    _ = @import("backends/context_stored_installed_invocation.zig");
    _ = @import("context_stored_installed_caller.zig");
    _ = @import("context_model_test.zig");
    _ = @import("context_metal_checkpoint_qualify.zig");
    _ = @import("context_quality_worker_staged_candidate.zig");
    _ = @import("context_production768_offline_replay.zig");
    _ = @import("runtime/kv/attention_match_production768_offline_replay.zig");
    _ = @import("runtime/kv/metal_checkpoint_cycles.zig");
    _ = @import("runtime/kv/attention_match_fit_policy.zig");
    _ = @import("server/native_kv_wire.zig");
    _ = @import("server/resident_ffn_request_closure.zig");
    _ = @import("server/native_kv_v2_metadata.zig");
    _ = @import("server/native_kv_producer_authority.zig");
    _ = @import("server/native_kv_verified_sidecar.zig");
    _ = @import("context_native_kv_operator_candidate.zig");
    _ = @import("context_qualified_private_startup.zig");
    _ = @import("context_private_dense_enrollment.zig");
    _ = @import("private_native_checkpoint_policy.zig");
    _ = @import("cli/native_kv_verify_v2.zig");
    _ = @import("runtime/kv/metal_checkpoint_soak.zig");
    _ = @import("cli/native_kv_upload.zig");
    _ = @import("runtime/kv/metal_checkpoint_quality.zig");
    _ = @import("context_native_checkpoint_quality.zig");
    _ = @import("context_supplied_checkpoint_quality.zig");
    _ = @import("context_prepared_serving_quality.zig");
    _ = @import("context_prepared_bundle_independent_fp64_quality.zig");
    _ = @import("server/native_kv_prepared_quality_closure.zig");
    _ = @import("context_query_campaign_capture.zig");
    _ = @import("context_fault_injection.zig");
    _ = @import("backends/context_teacher_forced.zig");
    _ = @import("context_numerical_capture_inputs.zig");
    _ = @import("context_native_head_qualify.zig");
    _ = @import("context_release_build_binding.zig");
    _ = @import("context_private_dense_profile.zig");
    _ = @import("context_exact_test.zig");
    _ = @import("context_capture_coverage.zig");
    _ = @import("context_serving_test.zig");
    // Discover driver/message tests even when compile-time filters omit the
    // model-serving tests that otherwise instantiate these imports lazily.
    _ = @import("context_serving_driver.zig");
    _ = @import("context_serving_messages.zig");
    _ = @import("context_serving_protocol.zig");
    _ = @import("context_quality_materializer.zig");
    _ = @import("context_native_quality_materializer.zig");
    _ = @import("backends/context_kernels.zig");
    _ = @import("backends/context_flash.zig");
    _ = @import("backends/context_q1_split_flash_synthetic_bench.zig");
    _ = @import("backends/context_execution_pool.zig");
    _ = @import("backends/context_query_protocol.zig");
    _ = @import("backends/context_query_campaign_pool.zig");
    _ = @import("backends/context_study_runner.zig");
    _ = @import("backends/context_study_templates.zig");
    _ = @import("backends/context_compaction.zig");
    _ = @import("backends/context_attention_match.zig");
    _ = @import("backends/context_native_packed_parity_test.zig");
    _ = @import("backends/packed_context_storage.zig");
    _ = @import("backends/replay_context.zig");
    _ = @import("backends/model_replay_context.zig");
    _ = @import("backends/packed_context_session.zig");
    _ = @import("backends/context_fit_capture_test.zig");
    _ = @import("backends/metal_context_session.zig");
    _ = @import("backends/metal_kv_storage.zig");
    // These regressions must remain discoverable when focused filters omit
    // the higher-level tests that otherwise instantiate their imports.
    _ = @import("architectures/gemma4_runtime.zig");
    _ = @import("backends/metal_runtime.zig");
    _ = @import("backends/metal_generated_kernels.zig");
    _ = @import("backends/decoder_gated_runtime.zig");
    _ = @import("ops/metal_compute.zig");
    _ = @import("ops/metal_attention_match_fit.zig");
    _ = @import("ops/metal_attention_match_lapack_test.zig");
    _ = @import("ops/metal_attention_match_storage.zig");
    _ = @import("runtime/kv/context_view.zig");
    _ = @import("runtime/kv/attention_matching.zig");
    _ = @import("runtime/kv/attention_match.zig");
    _ = @import("runtime/kv/attention_match_reference_omp.zig");
    _ = @import("runtime/kv/attention_match_engine.zig");
    _ = @import("runtime/kv/attention_match_cumulative_cache.zig");
    _ = @import("runtime/kv/attention_match_cumulative_admission.zig");
    _ = @import("runtime/kv/attention_match_adaptive_chunk_budget.zig");
    _ = @import("runtime/kv/attention_match_global_residual_diagnostic.zig");
    _ = @import("runtime/kv/attention_match_global_training_cache.zig");
    _ = @import("runtime/kv/attention_match_rejected_fixture.zig");
    _ = @import("runtime/kv/attention_match_v18_rejected_domain_replay.zig");
    _ = @import("runtime/kv/attention_match_rejected_replay.zig");
    _ = @import("runtime/kv/attention_match_exact_peak_residual_replay.zig");
    _ = @import("runtime/kv/attention_match_p91_geometry_snapshot_replay.zig");
    _ = @import("runtime/kv/attention_match_exact_core_mass_calibration.zig");
    _ = @import("runtime/kv/attention_match_exact_peak_residual.zig");
    _ = @import("runtime/kv/attention_match_fixed_support_global_refit.zig");
    _ = @import("runtime/kv/attention_match_joint_mass_bias_refit.zig");
    _ = @import("runtime/kv/attention_match_joint_bounded_run.zig");
    _ = @import("runtime/kv/attention_match_e2_historical_pressure_derivation.zig");
    _ = @import("runtime/kv/attention_match_streamed_gram_selection.zig");
    _ = @import("runtime/kv/attention_match_train_critical_source_keys.zig");
    _ = @import("runtime/kv/attention_match_train_critical_source_keys_streaming.zig");
    _ = @import("runtime/kv/attention_match_train_budget_calibrated_critical_threshold.zig");
    _ = @import("runtime/kv/attention_match_train_budget_calibrated_critical_threshold_reserve16.zig");
    _ = @import("runtime/kv/attention_match_train_historical_coverage_selection.zig");
    _ = @import("runtime/kv/attention_match_production768_historical_coverage.zig");
    _ = @import("runtime/kv/attention_match_train_pressure_exact_head_reservation.zig");
    _ = @import("runtime/kv/attention_match_critical_core_adapter.zig");
    _ = @import("pipelines/native_checkpoint_prepare_phase_observer.zig");
    _ = @import("runtime/kv/attention_match_joint_multiplicity_prior_cv.zig");
    _ = @import("runtime/kv/attention_match_accelerate_fp64_value_backend.zig");
    _ = @import("runtime/kv/attention_match_whole_fit_input.zig");
    _ = @import("runtime/kv/attention_match_training_augmentation.zig");
    _ = @import("runtime/kv/attention_match_future_augmentation.zig");
    _ = @import("runtime/kv/attention_match_exact_peak_whole_replay.zig");
    _ = @import("runtime/kv/attention_match_chunk_admission_diagnostic.zig");
    _ = @import("runtime/kv/attention_match_chunk_admission_replay.zig");
    _ = @import("runtime/kv/attention_match_chunk_admission_native.zig");
    _ = @import("runtime/kv/attention_match_joint_chunk_mass_diagnostic.zig");
    _ = @import("runtime/kv/attention_match_joint_chunk_mass_replay.zig");
    _ = @import("runtime/kv/attention_match_train_group_mass_lawson.zig");
    _ = @import("runtime/kv/attention_match_joint_chunk_mass_native.zig");
    _ = @import("runtime/kv/attention_match_joint_training_source.zig");
    _ = @import("runtime/kv/attention_match_joint_chunk_mass_anchored.zig");
    _ = @import("runtime/kv/attention_match_joint_replay_planner.zig");
    _ = @import("runtime/kv/attention_match_joint_adaptive_geometry.zig");
    _ = @import("runtime/kv/attention_match_chunk_admission_all_heads_native.zig");
    _ = @import("runtime/kv/attention_match_query_portfolio.zig");
    _ = @import("runtime/kv/attention_match_query_portfolio_replay.zig");
    _ = @import("runtime/kv/attention_match_query_portfolio_native.zig");
    _ = @import("runtime/kv/attention_match_checkpoint.zig");
    _ = @import("runtime/kv/attention_match_resume.zig");
    _ = @import("runtime/kv/attention_match_resident_future_training_authority_test.zig");
    _ = @import("runtime/kv/attention_match_prepared_heldout_bundle.zig");
    _ = @import("runtime/kv/attention_match_study.zig");
    _ = @import("runtime/kv/reference_replay.zig");
    _ = @import("native_generate.zig");
    _ = @import("pipelines/generation.zig");
    _ = @import("server/server.zig");
    _ = @import("server/native_kv_batch.zig");
    _ = @import("models/gpt.zig");
    _ = @import("runtime/kv/compaction.zig");
    _ = @import("runtime/kv/storage_runtime.zig");
}

test "context generation entrypoints compile" {
    // Keep admission tests from passing while executable-only wiring remains
    // uninstantiated by Zig's lazy analysis. No model is loaded by this test.
    const std = @import("std");
    std.mem.doNotOptimizeAway(&@import("native_generate.zig").main);
    std.mem.doNotOptimizeAway(&@import("server/server.zig").Node.generateContent);
    std.mem.doNotOptimizeAway(&@import("server/server.zig").Node.generateBatchContent);
    std.mem.doNotOptimizeAway(&@import("server/server.zig").Node.resumeNativeKvCheckpointBatch);
    std.mem.doNotOptimizeAway(&@import("pipelines/generation.zig").NativeGenerationPipeline.runCompactionStudy);
}

comptime {
    _ = @import("context_e2_8128_compiled_authority_observation.zig");
    _ = @import("runtime/kv/attention_match_recent64_captured_fixture.zig");
    _ = @import("runtime/kv/attention_match_recent64_captured_replay.zig");
    _ = @import("runtime/kv/attention_match_future_train_p64_capture.zig");
    _ = @import("runtime/kv/attention_match_recent64_future_primary_hybrid.zig");
    _ = @import("runtime/kv/attention_match_recent64_future_primary_direct.zig");
    _ = @import("runtime/kv/attention_match_recent64_future_primary_replay.zig");
}

comptime {
    _ = @import("runtime/kv/attention_match_future_train_p64_capture.zig");
    _ = @import("backends/metal_future_train_p64_capture.zig");
}
