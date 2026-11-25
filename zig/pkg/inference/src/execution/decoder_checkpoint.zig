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

//! Physical checkpoint preparation, fit/query qualification, snapshot and
//! restore. Context borrows the real model/backend owner and execution lane;
//! it has no tokenizer, task configuration or generation-pipeline dependency.
const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const platform = @import("antfly_platform");
const ops = @import("../ops/ops.zig");
const gpt_mod = @import("../models/gpt.zig");
const gpt_arch = @import("../architectures/gpt.zig");
const runtime = @import("../runtime/root.zig");
const activations = @import("../backends/activations.zig");
const execution_control = @import("../execution_control.zig");
const NativeDecodeState = @import("../graph/decode_state_runtime.zig").NativeDecodeState;
const native_kv_checkpoint = runtime.kv.attention_match_checkpoint;
const cuda_compute = if (build_options.enable_cuda) @import("../ops/cuda/cuda_compute.zig") else struct {};
const decoder_storage = @import("decoder_storage.zig");
const decoder_forward = @import("decoder_forward.zig");
const mtpProfileTimestamp = decoder_forward.profileTimestamp;
const mtpProfileElapsedNs = decoder_forward.profileElapsedNs;
const cudaReplayLastLogitsEnabled = decoder_forward.cudaReplayLastLogitsEnabled;
const writeF32LeFile = @import("../util/c_file.zig").writeF32LeFile;
const quality_mod = @import("../runtime/kv/metal_checkpoint_quality.zig");

pub const NativeCheckpointTrainingPolicy = enum { primary_only, private_dense512_reference_v1, private_future_decode_mix_v1, private_train_challenge_hybrid_resident_v1 };
pub const ResidentHybridCohort = quality_mod.Request.ResidentHybridAuthority.Cohort;
pub const ResidentHybridAuthority = quality_mod.Request.ResidentHybridAuthority;

pub const NativeCheckpointOperatorGrant = struct {
    pub const PrepareHostOwnerProfile = enum { none, bounded_page_e4_resident_1152m_v1, bounded_page_e4_formal_1152m_v1 };
};

fn augmentationSamplesAllowed(samples: u16, dense_reference: bool) bool {
    return runtime.kv.attention_match_training_augmentation.allowedSamples(samples) or
        (dense_reference and samples == runtime.kv.attention_match_training_augmentation.dense_reference_samples_per_stratum);
}

fn nativeMetalBatch64Validation(config: gpt_mod.Config) bool {
    if (!platform.env.getenvBool("ANTFLY_INFERENCE_METAL_GEMMA4_COMPACTION_FP64_BATCH64")) return false;
    const gemma4 = @import("../architectures/gemma4_runtime.zig");
    return gemma4.isExactE2bDenseRuntimeConfig(config) or gemma4.isExactE4bDenseRuntimeConfig(config);
}

pub const Context = struct {
    allocator: std.mem.Allocator,
    io: ?std.Io = null,
    backend: *const ops.ComputeBackend,
    model: *const gpt_mod.Config,
    kv_dtype: ?runtime.kv.pool.KvDType,
    identity: ?native_kv_checkpoint.Identity,
    routes: struct { graph_runtime: bool = false, compiled_partition: bool = false, draft: bool = false, prompt_cache: bool = false } = .{},
    execution: execution_control.ExecutionAccess = .{},
    work_control: runtime.kv.attention_match.Control = .{},
};

/// The native checkpoint request owns its output until the transport finishes.
/// A savings rejection returns an exact checkpoint with an explicit outcome.
pub const PrepareRequest = struct {
    pub const CompiledPrefillExactKvHandoff = struct {
        pub const LayoutPolicy = enum {
            split_swa_permitted,
            full_history_exact,
            /// Uniform full-history through 8K; above 8K, local donors use
            /// their exact page-aligned SWA span while global donors retain
            /// complete logical history. Candidate v8 authenticates this
            /// policy independently of all older full-history grants.
            full_history_exact_8k_split_swa_32k,
        };
        pub const CodecScope = enum {
            f16_only,
            f16_fp8_rotated4,
        };

        /// Hard device allocation ceiling admitted by the process-local
        /// qualification owner. This is not populated from HTTP or CLI input.
        max_copy_bytes: usize,
        max_host_staging_bytes: usize,
        codec_scope: CodecScope = .f16_only,
        max_logical_tokens: usize = std.math.maxInt(usize),
        layout_policy: LayoutPolicy = .split_swa_permitted,

        pub fn requiresUniformFullHistory(self: @This(), logical_tokens: usize) bool {
            return switch (self.layout_policy) {
                .split_swa_permitted => false,
                .full_history_exact => true,
                .full_history_exact_8k_split_swa_32k => logical_tokens <= 8192,
            };
        }

        pub fn authorizesBounded32kTsqr(self: @This()) bool {
            if (self.max_logical_tokens != 32768) return false;
            return self.layout_policy == .full_history_exact or
                self.layout_policy == .full_history_exact_8k_split_swa_32k;
        }

        pub fn allows(self: @This(), encoding: runtime.kv.attention_match_checkpoint.Encoding, logical_tokens: usize) bool {
            if (logical_tokens == 0 or logical_tokens > self.max_logical_tokens) return false;
            return switch (self.codec_scope) {
                .f16_only => encoding == .f16,
                .f16_fp8_rotated4 => encoding == .f16 or encoding == .fp8 or encoding == .rotated4,
            };
        }
    };
    pub const CompiledPrefillExactKvReceipt = struct {
        destination_storage_address: usize,
        device_allocation_bound_bytes: usize,
        copied_bytes: usize,
        peak_host_staging_bytes: usize,
        source_slots_released: usize,
        source_live_bytes_before: u64,
        source_live_bytes_after: u64,
        source_physically_released_bytes: u64,
    };
    pub const QualityFitStrategy = @import("../runtime/kv/metal_checkpoint_quality.zig").FitStrategy;
    pub const FitPolicy = @import("../runtime/kv/attention_match_fit_policy.zig").Policy;
    pub const Admission = runtime.kv.attention_match_engine.Options.Admission;
    pub const WholeFitInputExport = struct {
        output_dir: []const u8,
        augmentation_output_dir: ?[]const u8 = null,
        augmentation_samples: u16 = runtime.kv.attention_match_training_augmentation.legacy_samples_per_stratum,
        samples: u16 = 320,
        method: runtime.kv.attention_match_whole_fit_input.Method = .top1,
        source_freeze_sha256: [32]u8,
        binary_sha256: [32]u8,
        exporter_sha256: [32]u8,
        self_study: ?*SelfStudyTrainingCapture = null,
        reference_coverage: ?*ReferenceCoverageTrainingCapture = null,

        pub fn validateAugmentation(self: @This()) !void {
            if (self.self_study != null and self.reference_coverage != null) return error.InvalidTrainingAugmentationBinding;
            if (self.reference_coverage) |capture| {
                const expected: u16 = switch (capture.policy) {
                    .coverage320 => runtime.kv.attention_match_reference_coverage.sample_count,
                    .dense512 => runtime.kv.attention_match_reference_coverage.dense512_sample_count,
                };
                if (self.augmentation_samples != expected) return error.InvalidTrainingAugmentationBinding;
            }
            if ((self.self_study != null or self.reference_coverage != null) and self.augmentation_output_dir == null) return error.InvalidTrainingAugmentationBinding;
            if (self.augmentation_output_dir) |path| {
                const dense_reference = if (self.reference_coverage) |capture| capture.policy == .dense512 and self.augmentation_samples == runtime.kv.attention_match_training_augmentation.dense_reference_samples_per_stratum else false;
                if (path.len == 0 or !std.fs.path.isAbsolute(path) or !augmentationSamplesAllowed(self.augmentation_samples, dense_reference)) return error.InvalidTrainingAugmentationBinding;
            } else if (self.augmentation_samples != runtime.kv.attention_match_training_augmentation.legacy_samples_per_stratum) return error.InvalidTrainingAugmentationBinding;
        }
    };
    pub const SelfStudyTrainingCapture = struct {
        allocator: std.mem.Allocator,
        queries: runtime.kv.attention_match_queries.Collector,
        generated_ids: []i64,
        prompt_tokens: []const i64,
        prompt_tokens_sha256: [32]u8,
        rendered_suffix_sha256: [32]u8,
        generation_contract_sha256: [32]u8,
        generated_ids_sha256: [32]u8,
        complete_repeat_sha256: [32]u8,
        source_before: runtime.kv.attention_match_self_study_training.SourceBoundary,
        source_after: runtime.kv.attention_match_self_study_training.SourceBoundary,
        primary_parent_manifest_sha256: [32]u8,
        pub fn deinit(self: *@This()) void {
            self.queries.deinit();
            self.allocator.free(self.generated_ids);
            self.* = undefined;
        }
    };
    /// Unqualified staged capture from the unmodified exact source. Unlike the
    /// template study this starts at the source's own next-token logits.
    pub const FutureDecodeTrainingCapture = struct {
        allocator: std.mem.Allocator,
        queries: runtime.kv.attention_match_queries.Collector,
        generated_ids: []i64,
        generated_ids_sha256: [32]u8,
        source_before: runtime.kv.attention_match_self_study_training.SourceBoundary,
        source_after: runtime.kv.attention_match_self_study_training.SourceBoundary,
        early_eos_observations: u16,
        backend_owned_full_logits_calls: u16 = 0,
        backend_owned_full_logits_ns: u64 = 0,
        backend_owned_frame_successes: u16 = 0,
        generic_last_logits_calls: u16 = 0,
        backend_owned_full_logits_selected: bool = false,
        gpu_greedy_calls: u16 = 0,
        gpu_greedy_ns: u64 = 0,
        gpu_greedy_frame_successes: u16 = 0,
        gpu_greedy_selected: bool = false,
        pub fn deinit(self: *@This()) void {
            self.queries.deinit();
            self.allocator.free(self.generated_ids);
            self.* = undefined;
        }
    };
    pub const ReferenceCoverageTrainingCapture = struct {
        pub const Policy = enum { coverage320, dense512 };
        allocator: std.mem.Allocator,
        queries: runtime.kv.attention_match_queries.Collector,
        selection: runtime.kv.attention_match_reference_coverage.Selection,
        policy: Policy,
        source_before: runtime.kv.attention_match_self_study_training.SourceBoundary,
        source_after: runtime.kv.attention_match_self_study_training.SourceBoundary,
        primary_parent_manifest_sha256: [32]u8,
        pub fn deinit(self: *@This()) void {
            self.queries.deinit();
            self.selection.deinit();
            self.* = undefined;
        }
    };
    // Keep HTTP admission and the engine's enforced fitting bound identical,
    // with this ceiling above the request-specific geometry plan. Query capture
    // and transfer buffers remain separately reserved. Process RSS is also gated.
    pub const max_transient_host_bytes: usize = 3840 * 1024 * 1024;
    /// Prefer the tighter aggregate for ordinary 8K Metal fits. Larger source
    /// snapshots (notably 32K) still use the unchanged 3,840 MiB hard ceiling;
    /// source+candidate alone can exceed two GiB there.
    pub const metal_parallel_target_host_bytes: usize = 2 * 1024 * 1024 * 1024;
    /// Large fits use the established hard host ceiling with at most two
    /// workers. Process RSS remains independently guarded; source local rings
    /// avoid retaining redundant full-history device storage during fitting.
    pub const metal_large_source_target_host_bytes: usize = max_transient_host_bytes;
    // Large source/candidate snapshots coexist with model weights and native
    // device KV. Limit overlapping head work at that boundary; the unchanged
    // host reservation alone does not bound the model process's total RSS.
    pub const metal_large_source_fit_workers: usize = 2;
    pub const auxiliary_host_bytes: usize = 128 * 1024 * 1024;
    /// Peak local-ring K+V staging, within the existing auxiliary reservation
    /// alongside the 64 MiB query collector and snapshot transfer buffers.
    pub const query_clone_host_staging_bytes: usize = 4 * 1024 * 1024;
    pub const fit_workers: usize = 4;
    pub const collection_options: runtime.kv.attention_match_queries.Options = .{ .samples_per_consumer = 320 };
    pub const max_repeat_tokens: usize = 512;

    pub const HostPlan = struct { fit_bytes: usize, reservation_bytes: usize, source_bytes: usize, head_workspace_bytes: usize, workers: usize, query_reserve_bytes: usize = auxiliary_host_bytes, admission: Admission = .fixed, chunk_rows: u16 = 1024, reject_exact_active_fixed_point: bool = false, fit_strategy: QualityFitStrategy = .legacy };

    pub fn privateDenseQueryMaximumBytes() !usize {
        // Reserve complete collector caps, normalized Dense rows, and one
        // merged TRAIN owner. The future path clears the old merge before
        // capturing its continuation or allocating the replacement.
        const primary = collection_options.max_bytes;
        const dense: usize = 80 * 1024 * 1024;
        const future = runtime.kv.attention_match_self_study_training.maximum_bytes;
        const metadata: usize = 4 * 1024 * 1024;
        const authenticated_batch: usize = 2 * 1024 * 1024;
        const selector = runtime.kv.attention_match_future_decode_training.historical_selection.maximum_workspace_bytes;
        const generated_ids = runtime.kv.attention_match_future_decode_training.generated_tokens * @sizeOf(i64);
        var total: usize = 0;
        for ([_]usize{ primary, dense, dense, primary, dense, future, metadata, authenticated_batch, selector, generated_ids }) |bytes|
            total = try std.math.add(usize, total, bytes);
        return total;
    }

    pub fn fitPolicy(self: *const PrepareRequest) FitPolicy {
        return .{
            .admission = self.admission,
            .chunk_rows = self.chunk_rows,
            .reject_exact_active_fixed_point = self.reject_exact_active_fixed_point,
        };
    }

    pub fn validatePlanAdmission(plan: HostPlan, requested: Admission, chunk_rows: u16, reject_exact_active_fixed_point: bool) !void {
        if (plan.admission != requested or plan.chunk_rows != chunk_rows or plan.reject_exact_active_fixed_point != reject_exact_active_fixed_point)
            return error.InvalidNativeCheckpointRequest;
    }

    pub fn v18ParallelPlan(source_bytes: usize, eligible_heads: usize) !HostPlan {
        return v18ParallelPlanWithWorkspace(source_bytes, eligible_heads, 512 * 1024 * 1024);
    }

    pub const V18WorkspaceProfile = enum { standard, resident_native_hybrid_2g_v1 };

    pub fn v18ParallelPlanWithWorkspace(source_bytes: usize, eligible_heads: usize, per_head: usize) !HostPlan {
        return v18ParallelPlanWithWorkspaceProfile(source_bytes, eligible_heads, per_head, .standard);
    }

    pub fn v18ParallelPlanWithWorkspaceProfile(source_bytes: usize, eligible_heads: usize, per_head: usize, profile: V18WorkspaceProfile) !HostPlan {
        if (eligible_heads == 0) return error.InvalidNativeCheckpointRequest;
        const resident_hybrid = profile == .resident_native_hybrid_2g_v1;
        if (resident_hybrid) {
            if (per_head != 2 * 1024 * 1024 * 1024) return error.InvalidNativeCheckpointRequest;
        } else if (per_head != 512 * 1024 * 1024 and per_head != 1024 * 1024 * 1024) {
            return error.InvalidNativeCheckpointRequest;
        }
        // Primary collector (64 MiB), Dense512 collector (80 MiB), normalized
        // Dense TRAIN, and the merged TRAIN descriptor/value copy are live
        // together until the synchronous fit returns. Reserve their complete
        // configured maxima conservatively rather than relying on allocator
        // reuse or average payload sizes.
        const query_reserve: usize = if (per_head == 1024 * 1024 * 1024 or resident_hybrid) try privateDenseQueryMaximumBytes() else auxiliary_host_bytes;
        const resident = try runtime.kv.attention_match_engine.parallelResidentBytes(source_bytes, eligible_heads);
        // The private Dense512 route already reserves a full 1 GiB for every
        // worker. Use the existing 3,840 MiB hard ceiling for that explicit
        // profile so independent heads can run concurrently; this raises no
        // limit. The ordinary 512 MiB route retains the historical 2 GiB
        // target whenever its resident source/candidate pair fits there.
        const gross = if (per_head == 1024 * 1024 * 1024 or resident_hybrid)
            max_transient_host_bytes
        else if (resident <= metal_parallel_target_host_bytes)
            metal_parallel_target_host_bytes
        else
            max_transient_host_bytes;
        if (query_reserve >= gross) return error.CompactionWorkspaceLimit;
        const aggregate = gross - query_reserve;
        const planned = try @import("../ops/metal_attention_match_fit.zig").Fitter.parallelPlan(source_bytes, eligible_heads, per_head, aggregate, fit_workers);
        return .{ .fit_bytes = planned.admitted_bytes, .reservation_bytes = try std.math.add(usize, planned.admitted_bytes, query_reserve), .source_bytes = source_bytes, .head_workspace_bytes = per_head, .workers = planned.workers, .query_reserve_bytes = query_reserve, .admission = .cumulative_ladder_4_3_2, .chunk_rows = 256, .reject_exact_active_fixed_point = true, .fit_strategy = .identity_chunk_coefficients_adaptive_v18 };
    }

    pub fn v18EligibleHeadCount(config: gpt_mod.Config) !usize {
        var count: usize = 0;
        for (0..config.num_hidden_layers) |layer| if (config.isKvDonorLayer(layer) and !config.layerUsesSlidingAttention(layer)) {
            count = try std.math.add(usize, count, (try config.kvDonorGeometry(layer)).num_kv_heads);
        };
        return count;
    }

    pub fn fittingOptions() runtime.kv.attention_match_engine.Options {
        return fittingOptionsForAdmission(.fixed);
    }

    pub fn fittingOptionsForAdmission(admission: Admission) runtime.kv.attention_match_engine.Options {
        return fittingOptionsForPolicy(admission, 1024);
    }

    pub fn fittingOptionsForPolicy(admission: Admission, chunk_rows: u16) runtime.kv.attention_match_engine.Options {
        return fittingOptionsForQualityPolicy(admission, chunk_rows, false);
    }

    pub fn fittingOptionsForQualityPolicy(admission: Admission, chunk_rows: u16, reject_exact_active_fixed_point: bool) runtime.kv.attention_match_engine.Options {
        var options: runtime.kv.attention_match_engine.Options = .{
            .admission = admission,
            .storage_precision = .f16,
            .checkpoint_limits = native_kv_checkpoint.serving_limits,
            .prefix_rows = 128,
            .chunk_rows = chunk_rows,
            .min_retention_divisor = 2,
            .validation = .whole_head,
            .fit = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set },
        };
        if (admission == .cumulative_ladder_4_3_2) {
            options.retention_divisor = 4;
            options.min_retention_divisor = 2;
            options.independent_fp64_publication = true;
            options.min_global_savings_ratio = 1.5;
        }
        options.fit.reject_exact_active_fixed_point = reject_exact_active_fixed_point;
        return options;
    }

    /// Exact-prefix preparation only. Admission uses the same per-head sizing
    /// as fitting, including the largest worker and a full-size fallback cache.
    /// Existing compacted states must use their own coverage-aware planner.
    pub fn hostPlan(config: gpt_mod.Config, tokens: usize, encoding: runtime.kv.attention_match_checkpoint.Encoding, metal: bool) !HostPlan {
        return hostPlanWithAdmission(config, tokens, encoding, metal, .fixed);
    }

    pub fn hostPlanWithAdmission(config: gpt_mod.Config, tokens: usize, encoding: runtime.kv.attention_match_checkpoint.Encoding, metal: bool, admission: Admission) !HostPlan {
        return hostPlanWithPolicy(config, tokens, encoding, metal, admission, 1024);
    }

    pub fn hostPlanWithPolicy(config: gpt_mod.Config, tokens: usize, encoding: runtime.kv.attention_match_checkpoint.Encoding, metal: bool, admission: Admission, chunk_rows: u16) !HostPlan {
        return hostPlanWithQualityPolicy(config, tokens, encoding, metal, admission, chunk_rows, false);
    }

    pub fn hostPlanWithQualityPolicy(config: gpt_mod.Config, tokens: usize, encoding: runtime.kv.attention_match_checkpoint.Encoding, metal: bool, admission: Admission, chunk_rows: u16, reject_exact_active_fixed_point: bool) !HostPlan {
        try (FitPolicy{ .admission = admission, .chunk_rows = chunk_rows, .reject_exact_active_fixed_point = reject_exact_active_fixed_point }).validate(encoding, metal);
        if (metal) {
            if (comptime !build_options.enable_metal) return error.UnsupportedCompactionQueryRoute;
            if (encoding != .f16 and encoding != .fp8 and encoding != .rotated4)
                return error.UnsupportedMetalKvCheckpointEncoding;
        } else if (comptime !build_options.enable_cuda) return error.UnsupportedCompactionQueryRoute;
        if ((admission == .fixed and (chunk_rows != 1024 or reject_exact_active_fixed_point)) or
            (admission == .cumulative_ladder_4_3_2 and (!metal or
                (encoding != .f16 and encoding != .fp8 and encoding != .rotated4) or
                (chunk_rows != 128 and chunk_rows != 256 and chunk_rows != 1024)))) return error.InvalidNativeCheckpointRequest;
        if (config.family != .gemma or !config.usesGemma4Channels() or
            config.num_hidden_layers == 0 or config.num_hidden_layers > 256)
            return error.UnsupportedCompactionQueryModel;
        if (tokens < 2 or tokens > native_kv_checkpoint.max_context_tokens) return error.CompactionQueryLimit;
        const snapshot = runtime.kv.attention_match_snapshot;
        var options = fittingOptionsForQualityPolicy(admission, chunk_rows, reject_exact_active_fixed_point);
        options.independent_fp64_publication = metal;
        if (metal and nativeMetalBatch64Validation(config)) options.independent_validation_policy = .native_accelerate_fp64_summary64_v1;
        options.fp8_checkpoint = encoding == .fp8;
        options.rotated4_checkpoint = encoding == .rotated4;
        // Reserve for the largest bounded retry, even when 4:1 succeeds.
        if (admission == .fixed) options.retention_divisor = options.min_retention_divisor.?;
        var donors: [256]snapshot.Donor = undefined;
        var mapping: [256]u32 = undefined;
        var count: usize = 0;
        var heads: usize = 0;
        for (0..config.num_hidden_layers) |layer| {
            const geometry = try config.kvDonorGeometry(layer);
            if (config.isKvDonorLayer(layer)) {
                donors[count] = .{ .layer = @intCast(layer), .window = if (config.layerUsesSlidingAttention(layer)) config.sliding_window else 0, .heads = geometry.num_kv_heads, .dim = geometry.head_dim };
                count += 1;
            }
            var index: ?u32 = null;
            for (donors[0..count], 0..) |donor, i| {
                if (donor.layer == geometry.donor_layer) {
                    index = @intCast(i);
                    break;
                }
            }
            mapping[layer] = index orelse return error.UnsupportedKvSnapshotLayout;
        }
        const source_bytes = try snapshot.geometryBytes(tokens, config.vocab_size, mapping[0..config.num_hidden_layers], donors[0..count], options.checkpoint_limits);
        const samples = @min(@min(tokens, max_repeat_tokens), collection_options.samples_per_consumer);
        const heldout = (samples - 1) / collection_options.heldout_stride + 1;
        var maximum: usize = 0;
        for (donors[0..count], 0..) |donor, di| {
            if (donor.window != 0) continue;
            heads = try std.math.add(usize, heads, donor.heads);
            const old_rows = tokens - @min(tokens, options.recent_rows);
            const prefix = @min(tokens, options.prefix_rows);
            if (old_rows <= prefix or old_rows - prefix < 2) continue;
            var target = tokens - old_rows + prefix;
            var offset = prefix;
            while (offset < old_rows) {
                const rows = @min(options.chunk_rows, old_rows - offset);
                target = try std.math.add(usize, target, try std.math.divCeil(usize, rows, options.retention_divisor));
                offset += rows;
            }
            if (target == tokens) continue;
            var consumers: usize = 0;
            for (mapping[0..config.num_hidden_layers]) |mapped| if (mapped == di) {
                consumers += 1;
            };
            if (config.num_attention_heads == 0 or config.num_attention_heads % donor.heads != 0) return error.InvalidCompactionQueries;
            const width = try std.math.mul(usize, try std.math.mul(usize, consumers, config.num_attention_heads / donor.heads), donor.dim);
            const workspace = if (metal) workspace: {
                if (comptime build_options.enable_metal) {
                    const value = try @import("../ops/metal_attention_match_fit.zig").Fitter.headHostWorkspace(tokens, target, @min(old_rows - prefix, options.chunk_rows), try std.math.mul(usize, width, samples - heldout), try std.math.mul(usize, width, heldout), donor.dim, options);
                    break :workspace .{ .bytes = value.bytes, .solver_bytes = value.solver_bytes };
                }
                unreachable;
            } else workspace: {
                if (comptime build_options.enable_cuda) {
                    const value = try @import("../ops/cuda/attention_match_fit.zig").Fitter.headHostWorkspace(tokens, target, @min(old_rows - prefix, options.chunk_rows), try std.math.mul(usize, width, samples - heldout), try std.math.mul(usize, width, heldout), donor.dim, options);
                    break :workspace .{ .bytes = value.bytes, .solver_bytes = value.solver_bytes };
                }
                unreachable;
            };
            if (workspace.solver_bytes > options.fit.max_workspace_bytes) return error.CompactionWorkspaceLimit;
            maximum = @max(maximum, workspace.bytes);
        }
        return finishHostPlan(source_bytes, heads, maximum, metal, admission, chunk_rows, reject_exact_active_fixed_point);
    }

    /// Admission for an installed checkpoint uses its retained rows and
    /// immutable fitted coverage, including each head's largest retry.
    pub fn hostPlanForState(config: gpt_mod.Config, state: runtime.kv.attention_match_checkpoint.State, encoding: runtime.kv.attention_match_checkpoint.Encoding, control: runtime.kv.attention_match.Control) !HostPlan {
        return hostPlanForStateWithAdmission(config, state, encoding, .fixed, control);
    }

    pub fn hostPlanForStateWithAdmission(config: gpt_mod.Config, state: runtime.kv.attention_match_checkpoint.State, encoding: runtime.kv.attention_match_checkpoint.Encoding, admission: Admission, control: runtime.kv.attention_match.Control) !HostPlan {
        return hostPlanForStateWithPolicy(config, state, encoding, admission, 1024, control);
    }

    pub fn hostPlanForStateWithPolicy(config: gpt_mod.Config, state: runtime.kv.attention_match_checkpoint.State, encoding: runtime.kv.attention_match_checkpoint.Encoding, admission: Admission, chunk_rows: u16, control: runtime.kv.attention_match.Control) !HostPlan {
        return hostPlanForStateWithQualityPolicy(config, state, encoding, admission, chunk_rows, false, control);
    }

    pub fn hostPlanForStateWithQualityPolicy(config: gpt_mod.Config, state: runtime.kv.attention_match_checkpoint.State, encoding: runtime.kv.attention_match_checkpoint.Encoding, admission: Admission, chunk_rows: u16, reject_exact_active_fixed_point: bool, control: runtime.kv.attention_match.Control) !HostPlan {
        try (FitPolicy{ .admission = admission, .chunk_rows = chunk_rows, .reject_exact_active_fixed_point = reject_exact_active_fixed_point }).validate(encoding, true);
        if (comptime !build_options.enable_metal) return error.UnsupportedCompactionQueryRoute;
        if (encoding != .f16 and encoding != .fp8 and encoding != .rotated4)
            return error.UnsupportedMetalKvCheckpointEncoding;
        if ((admission == .fixed and (chunk_rows != 1024 or reject_exact_active_fixed_point)) or
            (admission == .cumulative_ladder_4_3_2 and
                ((encoding != .f16 and encoding != .fp8 and encoding != .rotated4) or
                    (chunk_rows != 128 and chunk_rows != 256 and chunk_rows != 1024)))) return error.InvalidNativeCheckpointRequest;
        if (config.family != .gemma or !config.usesGemma4Channels() or config.num_hidden_layers == 0 or
            config.num_hidden_layers > 256 or config.num_hidden_layers != state.layer_donors.len or
            config.vocab_size != state.logits.len or state.logical_length < 2)
            return error.UnsupportedCompactionQueryModel;
        try state.validateWithControl(native_kv_checkpoint.serving_limits, control);
        var options = fittingOptionsForQualityPolicy(admission, chunk_rows, reject_exact_active_fixed_point);
        options.independent_fp64_publication = true;
        if (nativeMetalBatch64Validation(config)) options.independent_validation_policy = .native_accelerate_fp64_summary64_v1;
        options.fp8_checkpoint = encoding == .fp8;
        options.rotated4_checkpoint = encoding == .rotated4;
        if (admission == .fixed) options.retention_divisor = options.min_retention_divisor.?;
        const samples = @min(@min(state.tokens.len, max_repeat_tokens), collection_options.samples_per_consumer);
        const heldout = (samples - 1) / collection_options.heldout_stride + 1;
        var maximum: usize = 0;
        var heads: usize = 0;
        var expected_donors: usize = 0;
        for (0..config.num_hidden_layers) |layer| {
            const geometry = try config.kvDonorGeometry(layer);
            const donor = state.donors[state.layer_donors[layer]];
            if (donor.layer != geometry.donor_layer or donor.heads.len != geometry.num_kv_heads or
                donor.heads[0].dim != geometry.head_dim or
                donor.window != (if (config.layerUsesSlidingAttention(layer)) config.sliding_window else @as(u32, 0)))
                return error.UnsupportedKvSnapshotLayout;
            expected_donors += @intFromBool(config.isKvDonorLayer(layer));
        }
        if (state.donors.len != expected_donors) return error.UnsupportedKvSnapshotLayout;
        for (state.donors, 0..) |donor, di| {
            try control.check();
            if (donor.window != 0) continue;
            heads = try std.math.add(usize, heads, donor.heads.len);
            var consumers: usize = 0;
            for (state.layer_donors) |mapped| {
                if (mapped == di) consumers += 1;
            }
            if (consumers == 0 or config.num_attention_heads == 0 or config.num_attention_heads % donor.heads.len != 0)
                return error.InvalidCompactionQueries;
            for (donor.heads) |head| {
                try control.check();
                const plan = (try runtime.kv.attention_match_engine.headPlan(state.logical_length, head, options)) orelse continue;
                const width = try std.math.mul(usize, try std.math.mul(usize, consumers, config.num_attention_heads / donor.heads.len), head.dim);
                const workspace = try @import("../ops/metal_attention_match_fit.zig").Fitter.headHostWorkspace(head.positions.len, plan.target_rows, plan.block_rows, try std.math.mul(usize, width, samples - heldout), try std.math.mul(usize, width, heldout), head.dim, options);
                if (workspace.solver_bytes > options.fit.max_workspace_bytes) return error.CompactionWorkspaceLimit;
                maximum = @max(maximum, workspace.bytes);
            }
        }
        return finishHostPlan(try state.residentBytes(), heads, maximum, true, admission, chunk_rows, reject_exact_active_fixed_point);
    }

    pub fn finishHostPlan(source_bytes: usize, heads: usize, maximum: usize, metal: bool, admission: Admission, chunk_rows: u16, reject_exact_active_fixed_point: bool) !HostPlan {
        const engine = runtime.kv.attention_match_engine;
        const fit_bytes, const workers = if (metal) plan: {
            if (comptime build_options.enable_metal) {
                const resident = try engine.parallelResidentBytes(source_bytes, heads);
                const aggregate_limit = if (resident <= metal_parallel_target_host_bytes)
                    metal_parallel_target_host_bytes
                else
                    metal_large_source_target_host_bytes;
                const maximum_workers = if (resident <= metal_parallel_target_host_bytes)
                    fit_workers
                else
                    metal_large_source_fit_workers;
                const worker_plan = try @import("../ops/metal_attention_match_fit.zig").Fitter.parallelPlan(
                    source_bytes,
                    heads,
                    maximum,
                    aggregate_limit,
                    maximum_workers,
                );
                break :plan .{ worker_plan.admitted_bytes, worker_plan.workers };
            }
            unreachable;
        } else plan: {
            const resident = try engine.parallelResidentBytes(source_bytes, heads);
            if (resident > max_transient_host_bytes) return error.CompactionWorkspaceLimit;
            const selected = if (maximum == 0) fit_workers else @min(fit_workers, (max_transient_host_bytes - resident) / maximum);
            if (selected == 0) return error.CompactionWorkspaceLimit;
            const bytes = try engine.parallelHostBytes(source_bytes, heads, maximum, selected);
            if (bytes > max_transient_host_bytes) return error.CompactionWorkspaceLimit;
            break :plan .{ bytes, selected };
        };
        return .{ .fit_bytes = fit_bytes, .reservation_bytes = try std.math.add(usize, fit_bytes, auxiliary_host_bytes), .source_bytes = source_bytes, .head_workspace_bytes = maximum, .workers = workers, .admission = admission, .chunk_rows = chunk_rows, .reject_exact_active_fixed_point = reject_exact_active_fixed_point };
    }

    pub const PreparedQualitySourceSink = struct {
        context: *anyopaque,
        write_all_fn: *const fn (*anyopaque, []const u8) anyerror!void,
        pub fn writeAll(self: *@This(), bytes: []const u8) !void {
            return self.write_all_fn(self.context, bytes);
        }
    };
    pub const PreparedQualityHeldoutSink = struct {
        context: *anyopaque,
        write_fn: *const fn (*anyopaque, []const runtime.kv.attention_match_engine.ReferenceBatch, runtime.kv.attention_match_future_decode_training.Authority, runtime.kv.attention_match_engine.Layout) anyerror!void,
        pub fn write(self: *@This(), batches: []const runtime.kv.attention_match_engine.ReferenceBatch, authority: runtime.kv.attention_match_future_decode_training.Authority, layout: runtime.kv.attention_match_engine.Layout) !void {
            return self.write_fn(self.context, batches, authority, layout);
        }
    };

    test_native_fault: @import("../context_native_checkpoint_fault.zig").Slot = if (builtin.is_test) null else {},
    fp64_accelerated_value_solves: u64 = 0,
    prepare_host_owner_profile: NativeCheckpointOperatorGrant.PrepareHostOwnerProfile = .none,
    prepare_host_owner_identity: u64 = 0,

    planned_fit_host_bytes: ?usize = null,
    planned_fit_workers: ?usize = null,
    planned_fit_head_workspace_bytes: ?usize = null,
    planned_private_query_bytes: ?usize = null,
    admission: Admission = .fixed,
    chunk_rows: u16 = 1024,
    reject_exact_active_fixed_point: bool = false,
    quality_fit_strategy: QualityFitStrategy = .legacy,
    /// Set only from a validated process-local operator grant. It does not
    /// enroll v18 in the registry or change the legacy default.
    staged_operator_strategy: bool = false,
    /// Explicit process-local opt-in for the private >96-candidate bounded
    /// geometry numerator backend. Never enabled by legacy or registry state.
    bounded_delta_accelerated: bool = false,
    /// Candidate-v4-only authorization for the bounded 32K TSQR value phase.
    /// This does not alter the CPU FP64 heldout/publication path.
    private_bounded_tsqr_value_backend: bool = false,
    /// Process-local private flat-route performance tuple; never HTTP-derived.
    private_capture_summary: bool = false,
    /// Process-local diagnostic selection of the incremental Gram chooser for
    /// the private capture/summary tuple. Never populated from HTTP input.
    private_incremental_budget: bool = false,
    /// Sealed private flat-route TRAIN reduction. This is process-local
    /// operator authority, never HTTP-derived, and requires the complete
    /// capture/summary/incremental tuple.
    private_training_reduction: bool = false,
    /// Test/diagnostic-only paired flat capture+Gram batch experiment. Only
    /// the focused test worker's explicit environment gate sets this field;
    /// server/grant paths and ordinary requests remain 64.
    private_capture_gram_batch256_diagnostic: bool = false,
    /// Request-private independent TRAIN domain. This is set only from the
    /// process-local operator grant; ordinary requests and enrolled profiles
    /// retain primary-only fitting.
    training_policy: NativeCheckpointTrainingPolicy = .primary_only,
    future_decode_training_policy_sha256: [32]u8 = @splat(0),
    future_training_authority: ?runtime.kv.attention_match_future_decode_training.Authority = null,
    /// Test-only, diagnostic complete future-mix export. The writer runs after
    /// the whole-fit parent commits and before numerical fitting starts.
    future_mix_output_dir: ?[]const u8 = null,
    future_training_capture: ?runtime.kv.attention_match_future_augmentation.FutureTrainingCaptureEvidence = null,
    future_historical_study_tokens_sha256: [32]u8 = @splat(0),
    /// Private staged route only. Null retains the eager checkpoint producer.
    compiled_prefill_exact_kv_handoff: ?CompiledPrefillExactKvHandoff = null,
    compiled_prefill_exact_kv_receipt: ?CompiledPrefillExactKvReceipt = null,
    compiled_prefill_preparation_wall_started_at: ?std.Io.Timestamp = null,
    compiled_prefill_preparation_wall_ns: u64 = 0,
    /// Ephemeral synchronous handoff owned by prepareNativeCheckpoint. Never
    /// serialized or retained after prepareMetalAttentionMatchFit returns.
    private_training_batches: ?[]const runtime.kv.attention_match_engine.ReferenceBatch = null,
    /// Borrowed only for the synchronous resident-hybrid rewrite. Both owners
    /// remain in prepareNativeCheckpoint until probeAttentionMatchFit returns.
    private_dense_training: ?*const runtime.kv.attention_match_training_augmentation.OwnedTraining = null,
    private_future_training_batches: ?[]const runtime.kv.attention_match_engine.ReferenceBatch = null,
    /// Actual collector authority borrowed only by the synchronous resident
    /// bridge. These values come from the live collector, never staged config.
    private_collector_source_length: usize = 0,
    private_collector_repeat_length: usize = 0,
    private_collector_required_bytes: usize = 0,
    private_collector_manifest_sha256: [32]u8 = @splat(0),
    /// Frozen pre-evaluation authorities supplied by the authenticated staged
    /// configuration. They are not interchangeable with training digests.
    resident_hybrid_authority: ?ResidentHybridAuthority = null,
    private_dense512_query_rows: usize = 0,
    private_query_live_bytes: usize = 0,
    /// Process-local qualification telemetry. Ordinary requests leave this null.
    qualification_phase_telemetry: ?@import("../pipelines/native_checkpoint_prepare_phase_observer.zig").Owned = null,
    primary_query_collection: quality_mod.QueryCollectionDiagnostic = .{},
    dense512_query_collection: quality_mod.QueryCollectionDiagnostic = .{},
    query_collection_combined_before_fit_ns: u64 = 0,
    /// Only the independent quality collector requests these extra seals.
    capture_quality_evidence: bool = false,
    quality_source_sha256: [32]u8 = @splat(0),
    quality_calibration: ?runtime.kv.attention_match_queries.Seal = null,
    whole_fit_input: ?WholeFitInputExport = null,
    producer_profile_sha256: [32]u8 = @splat(0),
    producer_profile_semantic_sha256: [32]u8 = @splat(0),
    /// Set from the validated process-local operator grant before resident
    /// preparation. Never populated from public checkpoint request input.
    producer_binary_sha256: [32]u8 = @splat(0),
    /// Process-local request from the public producer to retain a strict
    /// prepared-serving quality closure after AFATT002 export.
    prepared_quality_closure_requested: bool = false,
    prepared_quality_source_sink: ?PreparedQualitySourceSink = null,
    prepared_quality_heldout_sink: ?PreparedQualityHeldoutSink = null,
    prepared_quality_prompt_sha256: [32]u8 = @splat(0),
    prepared_quality_prompt_tokens: usize = 0,

    encoding: runtime.kv.attention_match_checkpoint.Encoding = .f16,
    minimum_savings_ratio: f64 = 1.5,
    validation_workers: usize = 1,
    result: ?Result = null,
    pub const DevicePayload = struct {
        allocator: std.mem.Allocator,
        device: @import("../ops/metal_attention_match_storage.zig").DeviceState,
        metadata: runtime.kv.attention_match_checkpoint.StreamingMetadata,
        maximum_head_download_bytes: usize,

        fn deinit(self: *@This()) void {
            self.device.deinit();
            self.allocator.free(self.metadata.donors);
            self.allocator.free(self.metadata.layer_donors);
            self.allocator.free(self.metadata.logits);
            self.allocator.free(self.metadata.tokens);
            self.* = undefined;
        }
    };
    pub const Payload = union(enum) {
        host: runtime.kv.attention_match_checkpoint.Owned,
        device: DevicePayload,

        pub fn deinit(self: *@This()) void {
            switch (self.*) {
                .host => |*owned| owned.deinit(),
                .device => |*owned| owned.deinit(),
            }
            self.* = undefined;
        }
        pub fn identity(self: *const @This()) runtime.kv.attention_match_checkpoint.Identity {
            return switch (self.*) {
                .host => |owned| owned.state.identity,
                .device => |owned| owned.metadata.identity,
            };
        }
        pub fn logicalLength(self: *const @This()) usize {
            return switch (self.*) {
                .host => |owned| owned.state.logical_length,
                .device => |owned| owned.metadata.logical_length,
            };
        }
        pub fn encodedSize(self: *@This(), encoding: runtime.kv.attention_match_checkpoint.Encoding, control: runtime.kv.attention_match.Control) !usize {
            return switch (self.*) {
                .host => |owned| runtime.kv.attention_match_checkpoint.encodedSize(owned.state, runtime.kv.attention_match_checkpoint.serving_limits, encoding, control),
                .device => |*owned| blk: {
                    var provider: runtime.kv.attention_match_device_checkpoint_provider.Context = .{ .allocator = owned.allocator, .state = &owned.device, .donors = owned.metadata.donors, .maximum_head_download_bytes = owned.maximum_head_download_bytes };
                    defer provider.deinit();
                    break :blk runtime.kv.attention_match_checkpoint.encodedSizeFromProvider(owned.metadata, provider.provider(), runtime.kv.attention_match_checkpoint.serving_limits, encoding, control);
                },
            };
        }
        pub fn exportEncoded(self: *@This(), encoding: runtime.kv.attention_match_checkpoint.Encoding, sink: anytype, control: runtime.kv.attention_match.Control) !runtime.kv.attention_match_checkpoint.ExportDigest {
            return switch (self.*) {
                .host => |owned| runtime.kv.attention_match_checkpoint.exportEncoded(owned.state, runtime.kv.attention_match_checkpoint.serving_limits, encoding, sink, control),
                .device => |*owned| blk: {
                    var provider: runtime.kv.attention_match_device_checkpoint_provider.Context = .{ .allocator = owned.allocator, .state = &owned.device, .donors = owned.metadata.donors, .maximum_head_download_bytes = owned.maximum_head_download_bytes };
                    defer provider.deinit();
                    break :blk runtime.kv.attention_match_checkpoint.exportEncodedFromProvider(owned.metadata, provider.provider(), runtime.kv.attention_match_checkpoint.serving_limits, encoding, sink, control);
                },
            };
        }
        pub fn artifactDigest(self: *@This(), encoding: runtime.kv.attention_match_checkpoint.Encoding, control: runtime.kv.attention_match.Control) ![32]u8 {
            const Sink = struct {
                pub fn writeAll(_: *@This(), _: []const u8) !void {}
            };
            var sink: Sink = .{};
            return (try self.exportEncoded(encoding, &sink, control)).file_sha256;
        }
    };
    pub const Result = struct {
        payload: Payload,
        /// Source owner revision observed before snapshot/export. This is
        /// producer provenance only; a receiver creates its own revision.
        source_revision: u64,
        outcome: enum { compacted, unchanged_insufficient_savings },
        stats: runtime.kv.attention_match_engine.Stats,
        source_host_bytes: usize,
        query_ns: u64,
        snapshot_ns: u64,
        fit_ns: u64,
        quality_fit_strategy: QualityFitStrategy = .legacy,
        v18_policy_sha256: [32]u8 = @splat(0),
        v18_eligible_heads: usize = 0,
        v18_attempted_heads: usize = 0,
        v18_accepted_heads: usize = 0,
        v18_preserved_heads: usize = 0,
        v18_fallback_heads: usize = 0,
        v18_cache_ready_heads: usize = 0,
        v18_planned_exact_heads: usize = 0,
        incremental_gram_chooser_calls: usize = 0,
        v18_query_batch_rows: usize = 0,
        v18_planned_head_workspace_bytes: usize = 0,
        v18_cache_phase_peak_bound_bytes: usize = 0,
        v18_evidence_profile: quality_mod.V18EvidenceProfile = .none,
        v18_evidence_profile_sha256: [32]u8 = @splat(0),
        v18_codec_physical_profile: quality_mod.V18EvidenceProfile = .none,
        v18_codec_physical_profile_sha256: [32]u8 = @splat(0),
        v18_bounded_tsqr_policy_sha256: [32]u8 = @splat(0),
        v18_bounded_tsqr_heads: usize = 0,
        v18_bounded_tsqr_calls: usize = 0,
        v18_bounded_tsqr_workspace_peak_bytes: usize = 0,
        v18_bounded_tsqr_cancellation_checks: usize = 0,
        v18_exact_f32_probability_heads: usize = 0,
        v18_gram_heads: usize = 0,
        v18_scored_reuse_heads: usize = 0,
        v18_gram_backend_calls: usize = 0,
        v18_bounded_heads: usize = 0,
        v18_primary_budget_percent: u8 = 0,
        v18_budget_fallback_heads: usize = 0,
        v18_bounded_provider_calls: u64 = 0,
        v18_bounded_provider_rows: u64 = 0,
        v18_bounded_workspace_bound_bytes: usize = 0,
        v18_bounded_scoring_cache_prepares: usize = 0,
        v18_bounded_scoring_selection_hits: usize = 0,
        v18_bounded_scoring_fit_hits: usize = 0,
        v18_bounded_range_cache_prepares: usize = 0,
        v18_bounded_geometry_cache_requests: usize = 0,
        v18_bounded_geometry_cache_hits: usize = 0,
        v18_bounded_geometry_cache_loads: usize = 0,
        v18_bounded_delta_heads: usize = 0,
        v18_bounded_delta_tiles: usize = 0,
        v18_bounded_delta_backend_calls: usize = 0,
        v18_bounded_delta_packed_bytes_peak: usize = 0,
        v18_bounded_delta_staged_bytes_peak: usize = 0,
        v18_bounded_delta_fallbacks: usize = 0,
        v18_numerical_policy: runtime.kv.attention_match_engine.Options.AdaptiveV18NumericalPolicy = .legacy_dense_f64,
        v18_cache_prepare_ns: u64 = 0,
        v18_selection_freeze_ns: u64 = 0,
        v18_joint_value_fit_ns: u64 = 0,
        v18_gram_construction_ns: u64 = 0,
        v18_chooser_ns: u64 = 0,
        v18_final_heldout_ns: u64 = 0,
        training_reduction_local_original_calls: usize = 0,
        training_reduction_rounded_harm_calls: usize = 0,
        training_reduction_tiles: usize = 0,
        training_reduction_backend_calls: usize = 0,
        training_reduction_workspace_peak_bytes: usize = 0,
        training_policy: NativeCheckpointTrainingPolicy = .primary_only,
        future_training_authority: ?runtime.kv.attention_match_future_decode_training.Authority = null,
        private_dense512_query_rows: usize = 0,
        private_query_live_bytes: usize = 0,
        planned_private_query_bytes: usize = 0,
        primary_query_collection: quality_mod.QueryCollectionDiagnostic = .{},
        dense512_query_collection: quality_mod.QueryCollectionDiagnostic = .{},
        query_collection_combined_before_fit_ns: u64 = 0,
        /// Set only after full candidate validation and request-private
        /// publication have completed. This is not a consumer installation.
        validation_complete: bool = false,
        request_private_publish_complete: bool = false,
        prompt_tokens_sha256: [32]u8 = @splat(0),
        prompt_tokens: usize = 0,
        source_checkpoint_sha256: [32]u8 = @splat(0),
    };
    pub fn deinit(self: *@This()) void {
        if (self.result) |*result| result.payload.deinit();
        self.result = null;
        if (self.qualification_phase_telemetry) |*owned| owned.deinit();
        self.qualification_phase_telemetry = null;
    }
};

pub const QueryOptions = struct {
    prefix_mode: enum { clone, replay } = .clone,
    repeat_mode: enum { reference, exact, exact_split } = .reference,
    stats: ?*QueryStats = null,
    collection: runtime.kv.attention_match_queries.Options = .{},
    chunk_size: usize = 256,
    /// Experimental bounded repeat: append only this many suffix tokens to
    /// the complete cloned/replayed prefix. Queries still come from a real
    /// forward at future positions. Null preserves full-repeat semantics.
    max_repeat_tokens: ?usize = null,
    /// Select prefix-wide study passages within the same repeat budget.
    representative_repeat: bool = false,
    study_pass: u8 = 0,
    explicit_repeat_tokens: ?[]const i64 = null,
    explicit_repeat_sha256: ?[32]u8 = null,
    explicit_sample_offsets: ?[]const u32 = null,
    max_source_tokens: usize = native_kv_checkpoint.max_context_tokens,
    /// Additional shadow KV capacity, including growth copies. Activations
    /// still use the backend's observed request allocation budget.
    max_shadow_kv_bytes: usize = 4 * 1024 * 1024 * 1024,
    owner_component_diagnostic_request: u64 = 0,
};

pub const QueryStats = struct {
    ragged_repeat_calls: usize = 0,
    prefix_ns: u64 = 0,
    repeat_ns: u64 = 0,
    forward_calls: usize = 0,
    repeat_kernel_launches: usize = 0,
    split_kernel_launches: usize = 0,
    scratch_bytes: usize = 0,
    setup_ns: u64 = 0,
    cleanup_ns: u64 = 0,
    cloned_bytes: usize = 0,
    shadow_allocation_bound_bytes: usize = 0,
    shadow_owner_id: u64 = 0,
    shadow_scope_generation: u64 = 0,
    shadow_owner_peak_bytes: u64 = 0,
    shadow_scope_released: bool = false,
    shadow_scope_delta_bytes: ?u64 = null,
    shadow_uses_source_owner: bool = false,
    shadow_scope_before_bytes: u64 = 0,
    shadow_scope_after_bytes: u64 = 0,
    shadow_scope_before_logical_bytes: u64 = 0,
    shadow_scope_after_logical_bytes: u64 = 0,
    shadow_scope_before_physical_bytes: u64 = 0,
    shadow_scope_after_physical_bytes: u64 = 0,
    shadow_scope_before_shared_bytes: u64 = 0,
    shadow_scope_after_shared_bytes: u64 = 0,
    shadow_scope_before_private_bytes: u64 = 0,
    shadow_scope_after_private_bytes: u64 = 0,
    parent_retained_before_bytes: u64 = 0,
    parent_retained_after_bytes: u64 = 0,
    parent_retained_delta_bytes: ?i64 = null,
    parent_retained_before_logical_bytes: u64 = 0,
    parent_retained_after_logical_bytes: u64 = 0,
    parent_retained_before_physical_bytes: u64 = 0,
    parent_retained_after_physical_bytes: u64 = 0,
    parent_retained_before_shared_bytes: u64 = 0,
    parent_retained_after_shared_bytes: u64 = 0,
    parent_retained_before_private_bytes: u64 = 0,
    parent_retained_after_private_bytes: u64 = 0,
};

/// Diagnostic sink deliberately cannot publish a checkpoint. The probe has
/// no loader-authenticated artifact identity and must never produce an
/// importable attachment. It exercises the bounded portable encoder only.
const ProbeSink = struct {
    bytes: usize = 0,
    pub fn writeAll(self: *@This(), data: []const u8) !void {
        self.bytes = try std.math.add(usize, self.bytes, data.len);
    }
};

pub const FitOptions = struct {
    cycle: ?usize = null,
    minimum_savings: ?f64 = null,
    checkpoint_roundtrip: bool = false,
    prepare_request: ?*PrepareRequest = null,
};

/// Address-stable private cache owner. Initialize only at its final address:
/// state, storage and device hooks borrow pointers into this object.
pub const Shadow = struct {
    manager: ?runtime.kv.manager.KvManager = null,
    storage: ?runtime.kv.storage_runtime.KvStorageRuntime = null,
    state: ?NativeDecodeState = null,

    pub fn deinit(self: *@This()) void {
        if (self.state) |*state| state.deinit();
        if (self.storage) |*storage| storage.deinit();
        if (self.manager) |*manager| manager.deinit();
        self.* = .{};
    }

    pub fn initEmpty(self: *@This(), runner: *Runtime) !void {
        if (self.manager != null or self.storage != null or self.state != null) return error.InvalidCompactionProbe;
        const dtype = runner.context.kv_dtype orelse return error.UnsupportedCompactionQueryDType;
        if (runner.context.backend.kind() != .cuda or dtype != .f16) return error.UnsupportedCompactionQueryRoute;
        var execution_scope = try runner.context.execution.acquireScope();
        defer if (execution_scope) |*scope| scope.deinit();
        self.manager = runtime.kv.manager.KvManager.init(runner.context.allocator);
        errdefer self.deinit();
        const config = decoder_storage.checkpointPool(dtype, runner.context.model.*);
        const pool = try self.manager.?.addPool(config);
        self.storage = try runtime.kv.storage_runtime.KvStorageRuntime.init(runner.context.allocator, config);
        try runner.context.backend.provisionKvDeviceWriteHook(&self.storage.?);
        self.state = NativeDecodeState.initPaged(runner.context.allocator, &self.manager.?, pool, null);
        self.state.?.kv_storage = &self.storage.?;
        self.state.?.configureForGptConfig(runner.context.model.*);
        self.state.?.allow_swa_ring = false;
    }

    pub fn init(self: *@This(), runner: *Runtime, source: *const NativeDecodeState, capacity: usize) !void {
        if (self.manager != null or self.storage != null or self.state != null) return error.InvalidCompactionProbe;
        const dtype = runner.context.kv_dtype orelse return error.UnsupportedCompactionQueryDType;
        if (runner.context.backend.kind() != .cuda or dtype != .f16 or !source.isPaged() or source.isKvCompacted()) return error.UnsupportedCompactionQueryRoute;
        const source_storage = source.kv_storage orelse return error.UnsupportedCompactionQueryRoute;
        const view = source.kv_view orelse return error.UnsupportedCompactionQueryRoute;
        if (view.position_offset != 0 or view.token_count != source.total_tokens or source.sequence_id != view.sequence_id or capacity < source.total_tokens or capacity > native_kv_checkpoint.max_context_tokens) return error.InvalidCompactionProbe;
        var donors: [256]runtime.kv.device_clone.Donor = undefined;
        var donor_count: usize = 0;
        if (runner.context.model.num_hidden_layers > donors.len) return error.UnsupportedCompactionQueryRoute;
        for (0..runner.context.model.num_hidden_layers) |layer| {
            if (!runner.context.model.isKvDonorLayer(layer)) continue;
            const geometry = try runner.context.model.kvDonorGeometry(layer);
            donors[donor_count] = .{ .layer = @intCast(layer), .heads = geometry.num_kv_heads, .dim = geometry.head_dim };
            donor_count += 1;
        }
        var execution_scope = try runner.context.execution.acquireScope();
        defer if (execution_scope) |*scope| scope.deinit();
        self.manager = runtime.kv.manager.KvManager.init(runner.context.allocator);
        errdefer self.deinit();
        const config = decoder_storage.checkpointPool(dtype, runner.context.model.*);
        const pool = try self.manager.?.addPool(config);
        self.storage = try runtime.kv.storage_runtime.KvStorageRuntime.init(runner.context.allocator, config);
        try runner.context.backend.provisionKvDeviceWriteHook(&self.storage.?);
        const hook = self.storage.?.device_write_hook orelse return error.UnsupportedCompactionQueryRoute;
        self.state = NativeDecodeState.initPaged(runner.context.allocator, &self.manager.?, pool, null);
        self.state.?.kv_storage = &self.storage.?;
        self.state.?.configureForGptConfig(runner.context.model.*);
        self.state.?.allow_swa_ring = false;
        try self.state.?.notePrefill(source.total_tokens);
        const clone_request: runtime.kv.device_clone.Request = .{
            .source_sequence = view.sequence_id,
            .destination_sequence = self.state.?.sequence_id.?,
            .tokens = source.total_tokens,
            .capacity = std.mem.alignForward(usize, capacity, 16),
            .page_size = config.page_size_tokens,
            .source_blocks = view.logical_blocks orelse return error.UnsupportedDeviceKvCloneLayout,
            .destination_blocks = self.state.?.kv_block_ids.items,
            .donors = donors[0..donor_count],
            .max_bytes = 4 * 1024 * 1024 * 1024,
            .check_ctx = runner,
            .check_fn = checkAttentionMatchQueryWork,
        };
        const compute: *cuda_compute.CudaCompute = @ptrCast(@alignCast(runner.context.backend.ptr));
        if ((try cuda_compute.installedAttentionMatchActivity(compute, source_storage, view.sequence_id)) != null) {
            _ = (try cuda_compute.cloneInstalledAttentionMatch(compute, &self.storage.?, source_storage, clone_request)) orelse return error.UnsupportedAttentionMatchRoute;
        } else {
            _ = try hook.cloneExactPrefix(source_storage.device_write_hook orelse return error.UnsupportedCompactionQueryRoute, clone_request);
        }
    }
};

pub const Runtime = struct {
    context: Context,
    /// Bounded numerical reproducer, not a resumable KV checkpoint. Layout is
    /// LE f32: [version, rows, dim, train_rows, heldout_rows, scale], K,V,Qtrain,Qheldout.
    fn dumpAttentionMatchFitFixture(self: *Runtime, state: runtime.kv.attention_match_checkpoint.State, queries: *runtime.kv.attention_match_queries.Collector) !void {
        const path = platform.env.getenv("ANTFLY_INFERENCE_ATTENTION_MATCH_FIT_FIXTURE") orelse return;
        if (path.len == 0 or state.logical_length <= 2049) return;
        var dumped: usize = 0;
        for (state.donors, 0..) |donor, donor_index| {
            if (donor.window != 0) continue;
            for (donor.heads, 0..) |head, head_index| {
                const rows: usize = if (platform.env.getenvBool("ANTFLY_INFERENCE_ATTENTION_MATCH_FIT_FIXTURE_FULL_HEAD")) head.positions.len else @intCast(@min(512, state.logical_length - 2048));
                var train: usize = 0;
                var heldout: usize = 0;
                const group = self.context.model.num_attention_heads / donor.heads.len;
                for (queries.batches) |batch| {
                    if (state.layer_donors[batch.consumer_layer] != donor_index or batch.query_head / group != head_index) continue;
                    train += batch.train.len;
                    heldout += batch.heldout.len;
                }
                const count = 6 + rows * head.dim * 2 + train + heldout;
                if (count > 40 * 1024 * 1024) return error.CompactionWorkspaceLimit;
                const values = try self.context.allocator.alloc(f32, count);
                defer self.context.allocator.free(values);
                values[0..6].* = .{ 1, @floatFromInt(rows), @floatFromInt(head.dim), @floatFromInt(train / head.dim), @floatFromInt(heldout / head.dim), 1.0 / @sqrt(@as(f32, @floatFromInt(head.dim))) };
                @memcpy(values[6..][0 .. rows * head.dim], head.keys[0 .. rows * head.dim]);
                @memcpy(values[6 + rows * head.dim ..][0 .. rows * head.dim], head.values[0 .. rows * head.dim]);
                var offset = 6 + rows * head.dim * 2;
                for (queries.batches) |batch| {
                    if (state.layer_donors[batch.consumer_layer] != donor_index or batch.query_head / group != head_index) continue;
                    @memcpy(values[offset..][0..batch.train.len], batch.train);
                    offset += batch.train.len;
                }
                for (queries.batches) |batch| {
                    if (state.layer_donors[batch.consumer_layer] != donor_index or batch.query_head / group != head_index) continue;
                    @memcpy(values[offset..][0..batch.heldout.len], batch.heldout);
                    offset += batch.heldout.len;
                }
                const destination = if (dumped == 0) try self.context.allocator.dupe(u8, path) else try std.fmt.allocPrint(self.context.allocator, "{s}.layer{d}.head{d}.f32", .{ path, donor.layer, head_index });
                defer self.context.allocator.free(destination);
                try writeF32LeFile(self.context.allocator, self.context.io orelse return error.MissingIo, destination, values);
                std.log.info("attention_match_fit_fixture donor_layer={d} head={d} rows={d} train_rows={d} heldout_rows={d}", .{ donor.layer, head_index, rows, train / head.dim, heldout / head.dim });
                dumped += 1;
                if (!platform.env.getenvBool("ANTFLY_INFERENCE_ATTENTION_MATCH_FIT_FIXTURE_ALL_HEADS")) return;
            }
        }
        std.log.info("attention_match_fit_fixture_complete heads={d}", .{dumped});
    }

    /// Restore a bounded, decoded checkpoint into a new request owner. The
    /// loaded model supplies identity and geometry; the uploaded header never
    /// grants compatibility. The caller supplies the complete rendered prompt.
    /// Returned logits belong to the caller and represent the saved boundary.
    pub fn validateAttentionMatchResume(self: *Runtime, saved: runtime.kv.attention_match_checkpoint.State, full_prompt: []const u32, max_new_tokens: usize) !void {
        const checkpoint = runtime.kv.attention_match_checkpoint;
        const resume_contract = runtime.kv.attention_match_resume;
        if (self.context.model.num_hidden_layers == 0 or self.context.model.num_hidden_layers > 256) return error.UnsupportedAttentionMatchRoute;
        if (full_prompt.len > native_kv_checkpoint.max_context_tokens or max_new_tokens == 0 or max_new_tokens > native_kv_checkpoint.max_context_tokens - full_prompt.len) return error.KvCheckpointAppendLimit;
        const identity = self.context.identity orelse return error.UnsupportedKvIdentity;
        const limits: checkpoint.Limits = native_kv_checkpoint.serving_limits;
        const control = self.context.work_control;
        var donors: [256]resume_contract.Geometry = undefined;
        var mapping: [256]u32 = undefined;
        var count: usize = 0;
        for (0..self.context.model.num_hidden_layers) |layer| {
            const geometry = try self.context.model.kvDonorGeometry(layer);
            if (self.context.model.isKvDonorLayer(layer)) {
                donors[count] = .{ .layer = @intCast(layer), .window = if (self.context.model.layerUsesSlidingAttention(layer)) self.context.model.sliding_window else 0, .heads = geometry.num_kv_heads, .dim = geometry.head_dim };
                count += 1;
            }
            var index: ?u32 = null;
            for (donors[0..count], 0..) |donor, i| if (donor.layer == geometry.donor_layer) {
                index = @intCast(i);
                break;
            };
            mapping[layer] = index orelse return error.UnsupportedKvSnapshotLayout;
        }
        _ = try resume_contract.validateWithControl(saved, .{ .identity = identity, .vocab_size = self.context.model.vocab_size, .max_logical_tokens = native_kv_checkpoint.max_context_tokens, .max_appended_tokens = 8192, .layer_donors = mapping[0..self.context.model.num_hidden_layers], .donors = donors[0..count] }, full_prompt, limits, control);
        _ = try @import("../ops/cuda/attention_match_storage.zig").allocationBytesWithControl(saved, .f16, limits, control);
    }

    pub fn restoreAttentionMatchState(self: *Runtime, saved: runtime.kv.attention_match_checkpoint.State, full_prompt: []const u32, max_new_tokens: usize, diagnostic_only: bool, destination: *NativeDecodeState) ![]f32 {
        if (comptime !build_options.enable_cuda) return error.UnsupportedAttentionMatchRoute;
        const checkpoint = runtime.kv.attention_match_checkpoint;
        const device_storage = @import("../ops/cuda/attention_match_storage.zig");
        if (self.context.backend.kind() != .cuda or self.context.kv_dtype != .f16 or self.context.model.family != .gemma or
            !self.context.model.usesGemma4Channels() or self.context.model.gemma4_mtp_assistant or
            self.context.routes.graph_runtime or self.context.routes.compiled_partition or self.context.routes.draft or self.context.routes.prompt_cache or
            !destination.isPaged() or destination.isKvCompacted() or destination.total_tokens != 0 or destination.sequence_id != null or
            self.context.model.num_hidden_layers == 0 or self.context.model.num_hidden_layers > 256)
            return error.UnsupportedAttentionMatchRoute;
        try self.validateAttentionMatchResume(saved, full_prompt, max_new_tokens);
        const limits: checkpoint.Limits = native_kv_checkpoint.serving_limits;
        const control = self.context.work_control;
        const logits = try self.context.allocator.dupe(f32, saved.logits);
        errdefer self.context.allocator.free(logits);
        var execution_scope = try self.context.execution.acquireScope();
        defer if (execution_scope) |*scope| scope.deinit();
        const compute: *cuda_compute.CudaCompute = @ptrCast(@alignCast(self.context.backend.ptr));
        const storage = destination.kv_storage orelse return error.UnsupportedAttentionMatchRoute;
        if (storage.sequences.items.len != 0 or storage.storage.config.store_cpu_bytes) return error.UnsupportedAttentionMatchRoute;
        const revision = try cuda_compute.attentionMatchRevision(compute, storage);
        try control.check();
        // This request owner is disposable on any failure. Metadata establishes
        // the original logical positions without forwarding or replaying tokens.
        destination.allow_swa_ring = false;
        try destination.notePrefill(saved.tokens.len);
        const backend: device_storage.Backend = .{ .allocator = self.context.allocator, .context = &compute.ctx, .precision = .f16, .limits = limits };
        const bound = @min(compute.ctx.device_allocations.snapshot().limit, 18 * 1024 * 1024 * 1024);
        var uploaded: ?device_storage.DeviceState = try backend.upload(saved, bound, control);
        defer if (uploaded) |*device| device.deinit();
        try cuda_compute.restoreAttentionMatch(compute, storage, destination.sequence_id orelse return error.InvalidCompactionProbe, &uploaded, saved.tokens.len, revision, bound, control, .{ .diagnostic_only = diagnostic_only });
        return logits;
    }

    /// Private end-to-end portable restore check. Compare a fresh imported owner
    /// against an independent clone of the fitted cache, without replaying its
    /// prefix or mutating the live request. This checks restore parity, not the
    /// lossy fit's quality against original full KV (qualified separately).
    pub fn verifyRestoredAttentionMatch(self: *Runtime, tokens: []const i64, logits: []const f32, source: *const NativeDecodeState) !void {
        if (comptime !build_options.enable_cuda) return error.UnsupportedAttentionMatchRoute;
        if (tokens.len >= native_kv_checkpoint.max_context_tokens or tokens.len != source.total_tokens) return error.InvalidCompactionProbe;
        const answer_limit = @min(@as(usize, 16), native_kv_checkpoint.max_context_tokens - tokens.len);
        const checkpoint = runtime.kv.attention_match_checkpoint;
        const compute: *cuda_compute.CudaCompute = @ptrCast(@alignCast(self.context.backend.ptr));
        const source_storage = source.kv_storage orelse return error.UnsupportedAttentionMatchRoute;
        const source_revision = try cuda_compute.attentionMatchRevision(compute, source_storage);
        const source_sequence = source.sequence_id orelse return error.UnsupportedAttentionMatchRoute;
        const identity = self.context.identity orelse return error.UnsupportedKvIdentity;
        const limits: checkpoint.Limits = native_kv_checkpoint.serving_limits;
        const control = self.context.work_control;
        const ids = try self.context.allocator.alloc(u32, tokens.len);
        defer self.context.allocator.free(ids);
        for (ids, tokens) |*id, token| id.* = std.math.cast(u32, token) orelse return error.InvalidCompactionProbe;
        var saved = blk: {
            var execution_scope = try self.context.execution.acquireScope();
            defer if (execution_scope) |*scope| scope.deinit();
            if ((try cuda_compute.installedAttentionMatchActivity(compute, source_storage, source_sequence)) == null) return error.UnsupportedAttentionMatchRoute;
            break :blk try self.snapshotAttentionMatchState(ids, logits, source, identity, limits, 2 * 1024 * 1024 * 1024);
        };
        defer saved.deinit();
        // Rotated rows retain an immutable codec representation. The parity
        // probe must exercise its real wire format rather than discard it.
        const encoding: checkpoint.Encoding = encoding: {
            for (saved.state.donors) |donor| for (donor.heads) |head| {
                if (head.compressed.spans.len != 0) break :encoding .rotated4;
            };
            break :encoding .f32;
        };
        const encoded = try checkpoint.encodeWithEncoding(self.context.allocator, saved.state, limits, encoding, control);
        defer self.context.allocator.free(encoded);
        var restored = try checkpoint.decodeWithControl(self.context.allocator, encoded, identity, limits, control);
        defer restored.deinit();
        var clone: Shadow = .{};
        // init must use cloneInstalledAttentionMatch when its source is fitted.
        try clone.init(self, source, tokens.len + answer_limit);
        defer clone.deinit();
        var fresh: Shadow = .{};
        try fresh.initEmpty(self);
        defer fresh.deinit();
        var clone_logits = try self.context.allocator.dupe(f32, logits);
        defer self.context.allocator.free(clone_logits);
        var fresh_logits = try self.restoreAttentionMatchState(restored.state, ids, answer_limit, true, &fresh.state.?);
        defer self.context.allocator.free(fresh_logits);
        var comparisons: usize = 0;
        var appended: usize = 0;
        for (0..answer_limit) |_| {
            try checkAttentionMatchQueryWork(self);
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(clone_logits), std.mem.sliceAsBytes(fresh_logits))) return error.KvCheckpointResumeMismatch;
            const chosen = activations.argmax(clone_logits);
            comparisons += 1;
            if (self.context.model.isEosToken(chosen)) break;
            const next: [1]i64 = .{@intCast(chosen)};
            const next_clone = try self.appendAttentionMatchProbeTokens(&clone.state.?, &next);
            self.context.allocator.free(clone_logits);
            clone_logits = next_clone;
            const next_fresh = try self.appendAttentionMatchProbeTokens(&fresh.state.?, &next);
            self.context.allocator.free(fresh_logits);
            fresh_logits = next_fresh;
            appended += 1;
        }
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(clone_logits), std.mem.sliceAsBytes(fresh_logits))) return error.KvCheckpointResumeMismatch;
        if (source.total_tokens != tokens.len or source.sequence_id != source_sequence or
            try cuda_compute.attentionMatchRevision(compute, source_storage) != source_revision) return error.CompactionSourceChanged;
        const activity = (try cuda_compute.installedAttentionMatchActivity(compute, &fresh.storage.?, fresh.state.?.sequence_id.?)) orelse return error.InvalidCompactionProbe;
        if (activity.invalid or fresh.state.?.total_tokens != tokens.len + appended or clone.state.?.total_tokens != fresh.state.?.total_tokens or
            (appended > 0 and (activity.attention_calls == 0 or activity.donor_appends == 0))) return error.KvCheckpointResumeMismatch;
        std.log.info("attention_match_restore wire_version={d} source_tokens={d} logical_tokens={d} greedy_comparisons={d} appended_tokens={d} attention_calls={d} donor_appends={d} checkpoint_roundtrip=1 fresh_owner=1 replayed_tokens=0 bitwise_logits=1 source_unchanged=1 invalid=0", .{ encoding.wireVersion(), tokens.len, fresh.state.?.total_tokens, comparisons, appended, activity.attention_calls, activity.donor_appends });
    }

    pub fn prepareNativeCheckpoint(self: *Runtime, request: *PrepareRequest, tokens: []const i64, logits: []const f32, source: *const NativeDecodeState) !void {
        if (request.result != null or !std.math.isFinite(request.minimum_savings_ratio) or request.minimum_savings_ratio <= 1 or request.minimum_savings_ratio > 4 or request.validation_workers != 1)
            return error.InvalidNativeCheckpointRequest;
        const plan = try PrepareRequest.hostPlan(self.context.model.*, tokens.len, request.encoding, false);
        if (request.planned_fit_host_bytes) |admitted| {
            if (admitted != plan.fit_bytes) return error.InvalidNativeCheckpointRequest;
        }
        request.planned_fit_host_bytes = plan.fit_bytes;
        if (request.planned_fit_workers) |admitted| {
            if (admitted != plan.workers) return error.InvalidNativeCheckpointRequest;
        }
        request.planned_fit_workers = plan.workers;
        const started = mtpProfileTimestamp(true, self.context.io);
        var queries = try self.collectAttentionMatchQueries(tokens, source, .{
            .prefix_mode = .clone,
            .repeat_mode = .exact_split,
            .collection = PrepareRequest.collection_options,
            .chunk_size = 256,
            .max_repeat_tokens = PrepareRequest.max_repeat_tokens,
            .representative_repeat = true,
        });
        defer queries.deinit();
        const query_ns = mtpProfileElapsedNs(true, self.context.io, started);
        _ = try self.probeAttentionMatchFit(tokens, logits, source, &queries, query_ns, .{ .prepare_request = request });
        if (request.result == null) return error.InvalidNativeCheckpointResult;
    }

    pub fn probeAttentionMatchFit(
        self: *Runtime,
        tokens: []const i64,
        logits: []const f32,
        source: *const NativeDecodeState,
        queries: *runtime.kv.attention_match_queries.Collector,
        query_ns: u64,
        probe_options: FitOptions,
    ) !runtime.kv.attention_match_engine.Stats {
        if (comptime !build_options.enable_cuda) return error.UnsupportedCompactionQueryRoute;
        if (self.context.backend.kind() != .cuda or self.context.kv_dtype != .f16) return error.UnsupportedCompactionQueryDType;
        var execution_scope = try self.context.execution.acquireScope();
        defer if (execution_scope) |*scope| scope.deinit();
        const checkpoint = runtime.kv.attention_match_checkpoint;
        const engine = runtime.kv.attention_match_engine;
        const storage = @import("../ops/cuda/attention_match_storage.zig");
        const Fitter = @import("../ops/cuda/attention_match_fit.zig").Fitter;
        const compute: *cuda_compute.CudaCompute = @ptrCast(@alignCast(self.context.backend.ptr));
        const control = self.context.work_control;
        const source_storage = source.kv_storage orelse return error.UnsupportedAttentionMatchRoute;
        const source_revision = try cuda_compute.attentionMatchRevision(compute, source_storage);
        const limits: checkpoint.Limits = native_kv_checkpoint.serving_limits;
        var phase: []const u8 = "snapshot";
        var stats: engine.Stats = .{};
        const started = mtpProfileTimestamp(true, self.context.io);
        errdefer std.log.err("attention_match_fit_rejected phase={s} elapsed_ns={d} donor_layer={d} head={d} chunk_start={d} attempted_chunks={d} completed_chunks={d} mass_error={d} output_rmse={d} nnls_sweeps={d} validating_storage={d} rounded_mass_error={d} rounded_output_rmse={d} validating_head={d} validated_heads={d}", .{
            phase, mtpProfileElapsedNs(true, self.context.io, started), stats.donor_layer, stats.head, stats.chunk_start, stats.attempted_chunks, stats.completed_chunks, stats.last_fit.max_relative_mass_error, stats.last_fit.relative_output_rmse, stats.last_fit.nnls_sweeps, @intFromBool(stats.validating_storage), stats.rounded_fit.max_relative_mass_error, stats.rounded_fit.relative_output_rmse, @intFromBool(stats.validating_head), stats.validated_heads,
        });
        const ids = try self.context.allocator.alloc(u32, tokens.len);
        defer self.context.allocator.free(ids);
        for (tokens, ids) |token, *id| id.* = std.math.cast(u32, token) orelse return error.InvalidCompactionQueries;
        const identity = self.context.identity orelse return error.UnsupportedKvIdentity;
        var original = try self.snapshotAttentionMatchState(ids, logits, source, identity, limits, 2 * 1024 * 1024 * 1024);
        var original_live = true;
        defer if (original_live) {
            original.deinit();
        };
        const source_host_bytes = try original.state.residentBytes();
        if (probe_options.prepare_request == null) try self.dumpAttentionMatchFitFixture(original.state, queries);
        const snapshot_ns = mtpProfileElapsedNs(true, self.context.io, started);
        var query_heads: [256]u32 = undefined;
        @memset(query_heads[0..original.state.layer_donors.len], self.context.model.num_attention_heads);
        var scales: [256]f64 = undefined;
        var caps: [256]f64 = undefined;
        for (original.state.donors, 0..) |donor, i| {
            // Captured queries already include Gemma's sqrt(head_dim)
            // correction, exactly as consumed by CUDA attention.
            scales[i] = 1.0 / @sqrt(@as(f64, @floatFromInt(donor.heads[0].dim)));
            caps[i] = 0;
        }
        phase = "fit";
        const fit_started = mtpProfileTimestamp(true, self.context.io);
        const residual_fit = probe_options.prepare_request != null or platform.env.getenvBool("ANTFLY_INFERENCE_ATTENTION_MATCH_FIT_RESIDUAL");
        const cuda_fit = probe_options.prepare_request != null or platform.env.getenvBool("ANTFLY_INFERENCE_ATTENTION_MATCH_FIT_CUDA");
        const cooperative_fit = probe_options.prepare_request != null or platform.env.getenvBool("ANTFLY_INFERENCE_ATTENTION_MATCH_FIT_COOPERATIVE");
        if (cooperative_fit and !cuda_fit) return error.UnsupportedCompactionQueryRoute;
        const worker_text = platform.env.getenv("ANTFLY_INFERENCE_ATTENTION_MATCH_FIT_WORKERS") orelse "1";
        const fit_workers = if (probe_options.prepare_request) |request| request.planned_fit_workers orelse return error.InvalidNativeCheckpointRequest else try std.fmt.parseInt(usize, worker_text, 10);
        if (fit_workers == 0 or fit_workers > 4 or (fit_workers > 1 and !cooperative_fit)) return error.UnsupportedCompactionQueryRoute;
        const old_device_limit = compute.ctx.device_allocations.snapshot().limit;
        try compute.ctx.device_allocations.setLimit(@min(old_device_limit, 18 * 1024 * 1024 * 1024));
        defer compute.ctx.device_allocations.setLimit(old_device_limit) catch {};
        var cuda_fitters: [4]Fitter = undefined;
        var cuda_fitter_count: usize = 0;
        defer for (cuda_fitters[0..cuda_fitter_count]) |*fitter| fitter.deinit();
        var fit_backends: [4]runtime.kv.attention_match.Backend = undefined;
        if (cuda_fit) {
            if (!residual_fit or @import("../ops/cuda/libraries.zig").Policy.current() == .off) return error.UnsupportedCompactionQueryRoute;
            for (cuda_fitters[0..fit_workers], fit_backends[0..fit_workers]) |*fitter, *fit_backend| {
                fitter.* = if (fit_workers > 1)
                    try Fitter.initWorker(self.context.allocator, &compute.ctx, 18 * 1024 * 1024 * 1024, 512 * 1024 * 1024)
                else
                    try Fitter.init(self.context.allocator, &compute.ctx, 18 * 1024 * 1024 * 1024);
                cuda_fitter_count += 1;
                fitter.cooperative_enabled = cooperative_fit;
                if (cooperative_fit and fitter.cooperative_blocks == 0) return error.UnsupportedCooperativeFit;
                fit_backend.* = fitter.backend();
            }
        }
        const diagnostic_encoding: checkpoint.Encoding = if (probe_options.prepare_request != null) .f32 else std.meta.stringToEnum(checkpoint.Encoding, platform.env.getenv("ANTFLY_INFERENCE_ATTENTION_MATCH_FIT_ENCODING") orelse "f32") orelse return error.InvalidCompactionProbe;
        const native_policy = probe_options.prepare_request != null or platform.env.getenvBool("ANTFLY_INFERENCE_ATTENTION_MATCH_FIT_NATIVE_POLICY");
        var fit_options: engine.Options = if (native_policy) PrepareRequest.fittingOptions() else .{};
        fit_options.fp8_checkpoint = if (probe_options.prepare_request) |request| request.encoding == .fp8 else diagnostic_encoding == .fp8;
        fit_options.rotated4_checkpoint = if (probe_options.prepare_request) |request| request.encoding == .rotated4 else diagnostic_encoding == .rotated4;
        fit_options.stats = &stats;
        fit_options.storage_precision = .f16;
        fit_options.checkpoint_limits = limits;
        fit_options.fit.control = control;
        fit_options.max_parallel_validations = if (probe_options.prepare_request) |request| request.validation_workers else platform.env.getenvUsize("ANTFLY_INFERENCE_ATTENTION_MATCH_VALIDATION_WORKERS") orelse 2;
        fit_options.source_has_metadata = (try cuda_compute.installedAttentionMatchActivity(compute, source_storage, source.sequence_id orelse return error.UnsupportedAttentionMatchRoute)) != null;
        const fit_host_bytes = if (probe_options.prepare_request) |request| request.planned_fit_host_bytes orelse return error.InvalidNativeCheckpointRequest else try std.fmt.parseInt(usize, platform.env.getenv("ANTFLY_INFERENCE_ATTENTION_MATCH_FIT_HOST_BYTES") orelse "2147483648", 10);
        if (fit_host_bytes == 0 or fit_host_bytes > 4 * 1024 * 1024 * 1024) return error.CompactionWorkspaceLimit;
        fit_options.max_transient_host_bytes = fit_host_bytes;
        if (cuda_fit) fit_options.fit.backend = fit_backends[0];
        if (residual_fit and !native_policy) {
            fit_options.prefix_rows = 128;
            fit_options.chunk_rows = 1024;
            fit_options.validation = .whole_head;
            fit_options.fit.selection = .residual_omp;
            fit_options.fit.mass_solver = .active_set;
        }
        if (probe_options.prepare_request != null or platform.env.getenvBool("ANTFLY_INFERENCE_ATTENTION_MATCH_FIT_PRESERVE_REJECTED"))
            fit_options.rejected_head_policy = .preserve_original;
        const minimum_saving = platform.env.getenv("ANTFLY_INFERENCE_ATTENTION_MATCH_FIT_MIN_SAVINGS") orelse "";
        if (probe_options.prepare_request) |request|
            fit_options.min_global_savings_ratio = request.minimum_savings_ratio
        else if (minimum_saving.len != 0)
            fit_options.min_global_savings_ratio = try std.fmt.parseFloat(f64, minimum_saving);
        if (probe_options.minimum_savings) |minimum| {
            if (probe_options.cycle == null or !std.math.isFinite(minimum) or minimum <= 1) return error.InvalidCompactionProbe;
            fit_options.min_global_savings_ratio = minimum;
        }
        const fit_layout: engine.Layout = .{
            .query_heads_by_layer = query_heads[0..original.state.layer_donors.len],
            .scales_by_donor = scales[0..original.state.donors.len],
            .softcaps_by_donor = caps[0..original.state.donors.len],
        };
        const fit_queries = try queries.finish();
        var candidate = (if (cuda_fitter_count > 1)
            engine.prepareParallel(original.state, fit_queries, fit_layout, fit_options, fit_backends[0..cuda_fitter_count])
        else
            engine.prepare(self.context.allocator, original.state, fit_queries, fit_layout, fit_options)) catch |err| {
            if (err == error.CompactionSavingsInsufficient) {
                if (probe_options.prepare_request) |request| {
                    try control.check();
                    request.result = .{
                        .checkpoint = original,
                        .outcome = .unchanged_insufficient_savings,
                        .stats = stats,
                        .source_host_bytes = source_host_bytes,
                        .query_ns = query_ns,
                        .snapshot_ns = snapshot_ns,
                        .fit_ns = mtpProfileElapsedNs(true, self.context.io, fit_started),
                    };
                    original_live = false;
                    return stats;
                }
            }
            std.log.err("attention_match_fit_error error={s}", .{@errorName(err)});
            return err;
        };
        var candidate_live = true;
        defer if (candidate_live) {
            candidate.deinit();
        };
        var cooperative_launches: usize = 0;
        var cooperative_restarts: usize = 0;
        var workspace_reuses: usize = 0;
        const executed_workers = cuda_fitter_count;
        // Include worker retirement in the measured fit phase, and release its
        // retained device capacity before checkpoint upload and installation.
        for (cuda_fitters[0..cuda_fitter_count]) |*fitter| {
            cooperative_launches += fitter.cooperative_launches;
            cooperative_restarts += fitter.cooperative_restarts;
            workspace_reuses += fitter.workspace_reuses;
            fitter.deinit();
        }
        cuda_fitter_count = 0;
        // Compare the immutable fitted candidate with the authoritative host
        // snapshot before retiring it. Live publication still checks the CUDA
        // owner revision after all uploads and fences.
        const guard: checkpoint.Store = .{ .current = original, .generation = source_revision };
        try guard.validateCompactionWithControl(candidate.state, source_revision, limits, control);
        original.deinit();
        original_live = false;
        const fit_ns = mtpProfileElapsedNs(true, self.context.io, fit_started);
        if (probe_options.prepare_request) |request| {
            try control.check();
            request.result = .{
                .checkpoint = candidate,
                .outcome = .compacted,
                .stats = stats,
                .source_host_bytes = source_host_bytes,
                .query_ns = query_ns,
                .snapshot_ns = snapshot_ns,
                .fit_ns = fit_ns,
            };
            candidate_live = false;
            return stats;
        }
        phase = "export";
        const export_started = mtpProfileTimestamp(true, self.context.io);
        var sink: ProbeSink = .{};
        const exported = try checkpoint.exportEncoded(candidate.state, limits, diagnostic_encoding, &sink, control);
        if (exported.bytes != sink.bytes) return error.InvalidKvCheckpoint;
        if (probe_options.checkpoint_roundtrip) {
            // Exercise actual bytes, decoder and independently owned imported
            // arrays before replacing the previous cycle's live binding.
            const encoded = try checkpoint.encodeWithEncoding(self.context.allocator, candidate.state, limits, diagnostic_encoding, control);
            defer self.context.allocator.free(encoded);
            var file_digest: [32]u8 = undefined;
            var file_hash = std.crypto.hash.sha2.Sha256.init(.{});
            var file_offset: usize = 0;
            while (file_offset < encoded.len) {
                try control.check();
                const count = @min(@as(usize, 65536), encoded.len - file_offset);
                file_hash.update(encoded[file_offset..][0..count]);
                file_offset += count;
            }
            file_hash.final(&file_digest);
            if (!std.mem.eql(u8, &file_digest, &exported.file_sha256)) return error.KvCheckpointChecksum;
            const imported = try checkpoint.decodeWithControl(self.context.allocator, encoded, identity, limits, control);
            candidate.deinit();
            candidate = imported;
        }
        const export_ns = mtpProfileElapsedNs(true, self.context.io, export_started);
        phase = "upload";
        const upload_started = mtpProfileTimestamp(true, self.context.io);
        const backend: storage.Backend = .{ .allocator = self.context.allocator, .context = &compute.ctx, .precision = .f16, .limits = limits };
        var uploaded: ?storage.DeviceState = try backend.upload(candidate.state, 18 * 1024 * 1024 * 1024, control);
        defer if (uploaded) |*device| device.deinit();
        const candidate_device_bytes = uploaded.?.allocated_bytes;
        const upload_ns = mtpProfileElapsedNs(true, self.context.io, upload_started);
        const install = platform.env.getenvBool("ANTFLY_INFERENCE_ATTENTION_MATCH_FIT_INSTALL");
        var install_ns: u64 = 0;
        if (install) {
            if (!cuda_fit or !residual_fit) return error.UnsupportedAttentionMatchRoute;
            phase = "install";
            const install_started = mtpProfileTimestamp(true, self.context.io);
            try cuda_compute.installAttentionMatchWithOptions(compute, source_storage, source.sequence_id orelse return error.UnsupportedAttentionMatchRoute, &uploaded, source.total_tokens, source_revision, 18 * 1024 * 1024 * 1024, control, .{ .diagnostic_only = probe_options.cycle != null });
            install_ns = mtpProfileElapsedNs(true, self.context.io, install_started);
        }
        const digest = exported.file_sha256;
        // Keep the complete diagnostic on one line without exceeding the
        // standard formatter's 32-argument limit.
        var codec_log_buffer: [512]u8 = undefined;
        const codec_log = try std.fmt.bufPrint(&codec_log_buffer, "export_version={d} fp8_heads={d} fp8_fallback_heads={d} min_retention_divisor={d} retention_retries={d} rotated4_heads={d} rotated4_fallback_heads={d} rotated4_eligible_rows={d} rotated4_compressed_rows={d} rotated4_exact_rows={d}", .{
            diagnostic_encoding.wireVersion(),
            stats.fp8_heads,
            stats.fp8_fallback_heads,
            fit_options.min_retention_divisor orelse fit_options.retention_divisor,
            stats.retention_retries,
            stats.rotated4_heads,
            stats.rotated4_fallback_heads,
            stats.rotated4_eligible_rows,
            stats.rotated4_compressed_rows,
            stats.rotated4_exact_rows,
        });
        std.log.info("{s} source_tokens={d} snapshot_ns={d} fit_ns={d} export_ns={d} upload_ns={d} total_ns={d} source_host_bytes={d} candidate_host_bytes={d} candidate_device_bytes={d} export_bytes={d} completed_chunks={d} max_nnls_sweeps={d} export_sha256={s} validated_heads={d} max_head_mass_error={d} max_head_output_rmse={d} residual_fit={d} preserved_heads={d} source_global_bytes={d} candidate_global_bytes={d} actual_global_savings_ratio={d} cuda_fit={d} diagnostic_only=1 installed={d} install_ns={d} cooperative_fit={d} cooperative_launches={d} cooperative_restarts={d} fit_workers={d} workspace_reuses={d} host_workspace_limit_bytes={d} {s}", .{
            if (probe_options.cycle != null) @as([]const u8, "attention_match_cycle_fit") else "attention_match_fit",
            tokens.len,
            snapshot_ns,
            fit_ns,
            export_ns,
            upload_ns,
            query_ns + mtpProfileElapsedNs(true, self.context.io, started),
            source_host_bytes,
            try candidate.state.residentBytes(),
            candidate_device_bytes,
            sink.bytes,
            stats.completed_chunks,
            stats.max_nnls_sweeps,
            std.fmt.bytesToHex(digest, .lower),
            stats.validated_heads,
            stats.max_head_mass_error,
            stats.max_head_output_rmse,
            @intFromBool(residual_fit),
            stats.preserved_heads,
            stats.source_global_bytes,
            stats.candidate_global_bytes,
            stats.actual_global_savings_ratio,
            @intFromBool(cuda_fit),
            @intFromBool(install),
            install_ns,
            @intFromBool(cooperative_fit),
            cooperative_launches,
            cooperative_restarts,
            executed_workers,
            workspace_reuses,
            fit_host_bytes,
            codec_log,
        });
        return stats;
    }

    /// Eager append with a single last-token logit projection. Keeping the last
    /// token separate avoids materializing [chunk, vocabulary] host logits and
    /// avoids graph/replay state shared with the original request.
    fn appendAttentionMatchProbeTokens(self: *Runtime, state: *NativeDecodeState, tokens: []const i64) ![]f32 {
        if (tokens.len == 0) return error.InvalidCompactionProbe;
        var execution_scope = try self.context.execution.acquireScope();
        defer if (execution_scope) |*scope| scope.deinit();
        var offset: usize = 0;
        while (offset + 1 < tokens.len) {
            try checkAttentionMatchQueryWork(self);
            const end = offset + @min(256, tokens.len - 1 - offset);
            try state.appendPrefillChunk(end - offset);
            var context = state.gptDecodeContext(state.total_tokens, end - offset);
            try gpt_arch.forwardHiddenOnly(self.context.backend, self.context.allocator, self.context.model.*, tokens[offset..end], 1, state.total_tokens, &context);
            offset = end;
        }
        try checkAttentionMatchQueryWork(self);
        try state.appendPrefillChunk(1);
        var context = state.gptDecodeContext(state.total_tokens, 1);
        return gpt_arch.forward(self.context.backend, self.context.allocator, self.context.model.*, tokens[tokens.len - 1 ..], 1, state.total_tokens, &context);
    }

    /// Diagnostic only: both branches start from the same original exact prefix.
    /// Each candidate is exported/imported and installed into the same private
    /// owner as the preceding cycle. The independent reference never compacts.
    /// Identical appended and teacher-forced answer tokens keep comparisons at
    /// the same logical positions, even when a candidate's greedy choice differs.
    pub fn probeAttentionMatchCycles(self: *Runtime, tokens: []const i64, logits: []const f32, source: *const NativeDecodeState, cycles: usize, append_tokens: usize, answer_tokens: usize, minimum_savings: f64) !void {
        if (comptime !build_options.enable_cuda) return error.UnsupportedCompactionQueryRoute;
        if (!platform.env.getenvBool("ANTFLY_INFERENCE_ATTENTION_MATCH_FIT_INSTALL") or
            !platform.env.getenvBool("ANTFLY_INFERENCE_ATTENTION_MATCH_FIT_CUDA") or
            !platform.env.getenvBool("ANTFLY_INFERENCE_ATTENTION_MATCH_FIT_RESIDUAL") or
            !platform.env.getenvBool("ANTFLY_INFERENCE_ATTENTION_MATCH_FIT_COOPERATIVE") or
            cycles == 0 or cycles > 20 or append_tokens < 512 or append_tokens > 2048 or append_tokens > tokens.len or
            answer_tokens == 0 or answer_tokens > 32 or !std.math.isFinite(minimum_savings) or minimum_savings <= 1 or
            logits.len != self.context.model.vocab_size) return error.InvalidCompactionProbe;
        const capacity = tokens.len + cycles * (append_tokens + answer_tokens);
        if (capacity > native_kv_checkpoint.max_context_tokens) return error.CompactionQueryLimit;
        const compute: *cuda_compute.CudaCompute = @ptrCast(@alignCast(self.context.backend.ptr));
        const source_storage = source.kv_storage orelse return error.UnsupportedCompactionQueryRoute;
        const source_revision = try cuda_compute.attentionMatchRevision(compute, source_storage);
        const source_length = source.total_tokens;
        const source_sequence = source.sequence_id;
        var exact: Shadow = .{};
        try exact.init(self, source, capacity);
        defer exact.deinit();
        var compacted: Shadow = .{};
        try compacted.init(self, source, capacity);
        defer compacted.deinit();
        const history = try self.context.allocator.alloc(i64, capacity);
        defer self.context.allocator.free(history);
        @memcpy(history[0..tokens.len], tokens);
        var length = tokens.len;
        var exact_logits = try self.context.allocator.dupe(f32, logits);
        defer self.context.allocator.free(exact_logits);
        var compacted_logits = try self.context.allocator.dupe(f32, logits);
        defer self.context.allocator.free(compacted_logits);
        var all_matches: usize = 0;
        var all_comparisons: usize = 0;
        for (0..cycles) |cycle| {
            try checkAttentionMatchQueryWork(self);
            if (cycle > 0) {
                const suffix = tokens[tokens.len - append_tokens ..];
                const next_exact = try self.appendAttentionMatchProbeTokens(&exact.state.?, suffix);
                self.context.allocator.free(exact_logits);
                exact_logits = next_exact;
                const next_compacted = try self.appendAttentionMatchProbeTokens(&compacted.state.?, suffix);
                self.context.allocator.free(compacted_logits);
                compacted_logits = next_compacted;
                @memcpy(history[length..][0..suffix.len], suffix);
                length += suffix.len;
            }
            const before_revision = try cuda_compute.attentionMatchRevision(compute, &compacted.storage.?);
            var query_stats: QueryStats = .{};
            const started = mtpProfileTimestamp(true, self.context.io);
            var queries = try self.collectAttentionMatchQueries(history[0..length], &compacted.state.?, .{
                .prefix_mode = .clone,
                .repeat_mode = if (cycle == 0) .exact_split else .reference,
                .stats = &query_stats,
                .collection = .{ .samples_per_consumer = 320 },
                .max_repeat_tokens = 512,
                .representative_repeat = platform.env.getenvBool("ANTFLY_INFERENCE_ATTENTION_MATCH_FIT_NATIVE_POLICY"),
            });
            defer queries.deinit();
            if (cycle > 0 and query_stats.ragged_repeat_calls == 0) return error.UnsupportedCompactionQueryRoute;
            const stats = try self.probeAttentionMatchFit(history[0..length], compacted_logits, &compacted.state.?, &queries, mtpProfileElapsedNs(true, self.context.io, started), .{
                .cycle = cycle + 1,
                .minimum_savings = if (cycle == 0) null else minimum_savings,
                .checkpoint_roundtrip = true,
            });
            const after_revision = try cuda_compute.attentionMatchRevision(compute, &compacted.storage.?);
            if (after_revision != before_revision + 1 or stats.completed_chunks == 0 or stats.candidate_global_bytes >= stats.source_global_bytes) return error.CompactionCycleDidNotCompact;
            const answer_start = length;
            var matches: usize = 0;
            var comparisons: usize = 0;
            var max_logit_relative_l2: f64 = 0;
            for (0..answer_tokens) |_| {
                if (exact_logits.len != compacted_logits.len) return error.InvalidCompactionProbe;
                var error_squared: f64 = 0;
                var reference_squared: f64 = 0;
                for (exact_logits, compacted_logits) |reference, candidate| {
                    if (!std.math.isFinite(reference) or !std.math.isFinite(candidate)) return error.InvalidCompactionProbe;
                    const difference = @as(f64, candidate) - reference;
                    error_squared += difference * difference;
                    reference_squared += @as(f64, reference) * reference;
                }
                max_logit_relative_l2 = @max(max_logit_relative_l2, @sqrt(error_squared / @max(reference_squared, 1e-30)));
                const expected = activations.argmax(exact_logits);
                const actual = activations.argmax(compacted_logits);
                matches += @intFromBool(actual == expected);
                comparisons += 1;
                if (self.context.model.isEosToken(expected)) break;
                const next_token: [1]i64 = .{@intCast(expected)};
                const next_exact = try self.appendAttentionMatchProbeTokens(&exact.state.?, &next_token);
                self.context.allocator.free(exact_logits);
                exact_logits = next_exact;
                const next_compacted = try self.appendAttentionMatchProbeTokens(&compacted.state.?, &next_token);
                self.context.allocator.free(compacted_logits);
                compacted_logits = next_compacted;
                history[length] = @intCast(expected);
                length += 1;
            }
            const activity = (try cuda_compute.installedAttentionMatchActivity(compute, &compacted.storage.?, compacted.state.?.sequence_id.?)) orelse return error.InvalidCompactionProbe;
            if (activity.invalid or exact.state.?.total_tokens != length or compacted.state.?.total_tokens != length or
                source.total_tokens != source_length or source.sequence_id != source_sequence or
                try cuda_compute.attentionMatchRevision(compute, source_storage) != source_revision) return error.CompactionSourceChanged;
            all_matches += matches;
            all_comparisons += comparisons;
            std.log.info("attention_match_cycle cycle={d} logical_tokens={d} completed_chunks={d} greedy_matches={d} greedy_comparisons={d} answer_appends={d} max_logit_relative_l2={d} ragged_repeat_calls={d} installed_revision={d} checkpoint_roundtrip=1 source_unchanged=1 invalid=0 incremental_savings_ratio={d}", .{
                cycle + 1, length, stats.completed_chunks, matches, comparisons, length - answer_start, max_logit_relative_l2, query_stats.ragged_repeat_calls, after_revision, stats.actual_global_savings_ratio,
            });
        }
        std.log.info("attention_match_cycles_complete cycles={d} initial_tokens={d} final_tokens={d} greedy_matches={d} greedy_comparisons={d} source_unchanged=1", .{ cycles, tokens.len, length, all_matches, all_comparisons });
        // Token parity is a strict diagnostic gate, not a replacement for the
        // task-level quality suite. Record all cycles before reporting drift.
        if (all_matches != all_comparisons) return error.CompactionCycleGreedyMismatch;
    }

    /// Export an exact, locked ingestion boundary from authoritative device
    /// storage. Identity must come from loaded model artifacts, never an upload.
    /// Logical positions, donor sharing and next-token logits are preserved.
    /// This does not authorize compacted-cache serving or bypass its route gates.
    pub fn snapshotAttentionMatchState(
        self: *Runtime,
        tokens: []const u32,
        logits: []const f32,
        source: *const NativeDecodeState,
        identity: runtime.kv.attention_match_checkpoint.Identity,
        limits: runtime.kv.attention_match_checkpoint.Limits,
        max_host_bytes: usize,
    ) !runtime.kv.attention_match_checkpoint.Owned {
        if (self.context.backend.kind() != .cuda or self.context.model.family != .gemma or !self.context.model.usesGemma4Channels() or
            self.context.routes.graph_runtime or self.context.routes.compiled_partition or self.context.routes.draft or self.context.routes.prompt_cache or
            !source.isPaged() or source.isKvCompacted() or source.total_tokens != tokens.len or
            logits.len != self.context.model.vocab_size or self.context.model.num_hidden_layers > 256)
            return error.UnsupportedKvSnapshotLayout;
        const storage = source.kv_storage orelse return error.DeviceReadUnsupported;
        const hook = storage.device_write_hook orelse return error.DeviceReadUnsupported;
        const view = source.kv_view orelse return error.UnsupportedKvSnapshotLayout;
        if (view.position_offset != 0 or view.token_count != tokens.len or source.sequence_id != view.sequence_id)
            return error.UnsupportedKvSnapshotLayout;
        var donors: [256]runtime.kv.attention_match_snapshot.Donor = undefined;
        var layer_donors: [256]u32 = undefined;
        var count: usize = 0;
        for (0..self.context.model.num_hidden_layers) |layer| {
            const geometry = try self.context.model.kvDonorGeometry(layer);
            if (self.context.model.isKvDonorLayer(layer)) {
                donors[count] = .{ .layer = @intCast(layer), .window = if (self.context.model.layerUsesSlidingAttention(layer)) self.context.model.sliding_window else 0, .heads = geometry.num_kv_heads, .dim = geometry.head_dim };
                count += 1;
            }
            var index: ?u32 = null;
            for (donors[0..count], 0..) |donor, i| {
                if (donor.layer == geometry.donor_layer) {
                    index = @intCast(i);
                    break;
                }
            }
            layer_donors[layer] = index orelse return error.UnsupportedKvSnapshotLayout;
        }
        const request: runtime.kv.attention_match_snapshot.Request = .{
            .identity = identity,
            .sequence = source.sequence_id orelse return error.UnsupportedKvSnapshotLayout,
            .tokens = tokens,
            .logits = logits,
            .layer_donors = layer_donors[0..self.context.model.num_hidden_layers],
            .donors = donors[0..count],
            .limits = limits,
            .max_host_bytes = max_host_bytes,
            .control = .{ .context = self, .check_fn = checkAttentionMatchQueryWork },
        };
        if (comptime build_options.enable_cuda) {
            const compute: *cuda_compute.CudaCompute = @ptrCast(@alignCast(self.context.backend.ptr));
            if (try cuda_compute.snapshotInstalledAttentionMatch(compute, storage, self.context.allocator, request)) |snapshot| return snapshot;
        }
        return runtime.kv.attention_match_snapshot.capture(self.context.allocator, request, try hook.attentionMatchReader());
    }

    /// Qualification adapter: clone an exact prefix into owned device buffers,
    /// then repeat its tokens at future logical positions. Replay remains an
    /// explicit reference mode. Caller owns the returned queries. This does not
    /// commit or serve a compacted cache; source must be locked and immutable.
    pub fn collectAttentionMatchQueries(
        self: *Runtime,
        source_tokens: []const i64,
        source: *const NativeDecodeState,
        options: QueryOptions,
    ) !runtime.kv.attention_match_queries.Collector {
        const queries = runtime.kv.attention_match_queries;
        const collection_started = mtpProfileTimestamp(true, self.context.io);
        var cleanup_started: ?std.Io.Timestamp = null;
        defer if (options.stats) |stats| {
            if (cleanup_started) |started| stats.cleanup_ns = mtpProfileElapsedNs(true, self.context.io, started);
        };
        if (self.context.backend.kind() != .cuda or self.context.model.family != .gemma or
            !self.context.model.usesGemma4Channels()) return error.UnsupportedCompactionQueryModel;
        if (self.context.routes.graph_runtime or
            self.context.routes.compiled_partition or self.context.routes.draft or self.context.routes.prompt_cache)
            return error.UnsupportedCompactionQueryRoute;
        // Graph and pinned-temporary state belongs to the live request. Never
        // reset it or accidentally capture pointers to short-lived shadow KV.
        const replay = platform.env.getenv("ANTFLY_INFERENCE_CUDA_DECODE_GRAPH_REPLAY") orelse "off";
        if (self.context.backend.debugCudaGraphReplayRequested() or !std.mem.eql(u8, replay, "off") or cudaReplayLastLogitsEnabled() or
            (platform.env.getenvUsize("ANTFLY_INFERENCE_CUDA_CAPTURE_FORCE_KV_CAPACITY") orelse 0) != 0 or
            (platform.env.getenvUsize("ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD") orelse 0) != 0 or
            platform.env.getenvBool("ANTFLY_INFERENCE_CUDA_TEMP_ARENA_AUTOPLAN"))
            return error.UnsupportedCompactionQueryReplay;
        const dtype = self.context.kv_dtype orelse return error.UnsupportedCompactionQueryDType;
        if (dtype != .f16 and dtype != .f32) return error.UnsupportedCompactionQueryDType;
        if (options.repeat_mode != .reference and dtype != .f16) return error.UnsupportedCompactionQueryDType;
        if (source_tokens.len < 2 or source_tokens.len > options.max_source_tokens or
            source_tokens.len > self.context.model.max_position_embeddings / 2 or
            options.chunk_size == 0 or options.chunk_size > 512 or
            options.chunk_size > options.collection.max_capture_rows or self.context.model.num_hidden_layers > 256)
            return error.CompactionQueryLimit;
        const study = if (options.representative_repeat)
            try queries.studyTokens(self.context.allocator, source_tokens, options.max_repeat_tokens orelse source_tokens.len)
        else
            null;
        defer if (study) |tokens| self.context.allocator.free(tokens);
        const repeat_tokens = study orelse source_tokens[source_tokens.len - @min(source_tokens.len, options.max_repeat_tokens orelse source_tokens.len) ..];
        if (repeat_tokens.len < 2) return error.CompactionQueryLimit;
        for (source_tokens) |token| if (token < 0 or token >= self.context.model.vocab_size) return error.InvalidCompactionQueries;

        try checkAttentionMatchQueryWork(self);
        if (options.stats) |stats| stats.* = .{};
        var consumers: [256]queries.Consumer = undefined;
        var donors: [256]runtime.kv.device_clone.Donor = undefined;
        var donor_count: usize = 0;
        var count: usize = 0;
        var kv_row_bytes: usize = 0;
        for (0..self.context.model.num_hidden_layers) |layer| {
            const geometry = try self.context.model.kvDonorGeometry(layer);
            if (self.context.model.isKvDonorLayer(layer)) {
                kv_row_bytes = try std.math.add(usize, kv_row_bytes, try dtype.bytesForTokenPairChecked(geometry.num_kv_heads, geometry.head_dim));
                donors[donor_count] = .{ .layer = @intCast(layer), .heads = geometry.num_kv_heads, .dim = geometry.head_dim };
                donor_count += 1;
            }
            if (self.context.model.layerUsesSlidingAttention(layer)) continue;
            consumers[count] = .{
                .layer = @intCast(layer),
                .donor = @intCast(geometry.donor_layer),
                .query_heads = self.context.model.num_attention_heads,
                .kv_heads = geometry.num_kv_heads,
                .dim = geometry.head_dim,
            };
            count += 1;
        }
        const shadow_rows = std.mem.alignForward(usize, source_tokens.len + repeat_tokens.len, 16);
        // Doubling capacities plus all prior growth copies are below 4x final
        // required capacity. Page tables/activations are charged by the backend.
        const shadow_kv_bytes = try std.math.mul(usize, try std.math.mul(usize, shadow_rows, 4), kv_row_bytes);
        if (options.prefix_mode == .replay and shadow_kv_bytes > options.max_shadow_kv_bytes) return error.CompactionQueryLimit;
        var collector = try queries.Collector.init(self.context.allocator, consumers[0..count], source_tokens.len, repeat_tokens.len, options.collection);
        errdefer collector.deinit();
        var execution_scope = try self.context.execution.acquireScope();
        defer if (execution_scope) |*scope| scope.deinit();
        var manager = runtime.kv.manager.KvManager.init(self.context.allocator);
        defer manager.deinit();
        const pool_config = decoder_storage.checkpointPool(dtype, self.context.model.*);
        const pool_id = try manager.addPool(pool_config);
        var storage = try runtime.kv.storage_runtime.KvStorageRuntime.init(self.context.allocator, pool_config);
        defer storage.deinit();
        // Provisioning a separate hook does not begin/reset the model request.
        try self.context.backend.provisionKvDeviceWriteHook(&storage);
        if (storage.device_write_hook == null) return error.UnsupportedCompactionQueryRoute;
        var shadow = NativeDecodeState.initPaged(self.context.allocator, &manager, pool_id, null);
        shadow.kv_storage = &storage;
        shadow.configureForGptConfig(self.context.model.*);
        shadow.allow_swa_ring = false;
        defer shadow.deinit();
        if (options.stats) |stats| stats.setup_ns = mtpProfileElapsedNs(true, self.context.io, collection_started);
        const prefix_started = mtpProfileTimestamp(true, self.context.io);
        var ragged_shadow = false;
        if (options.prefix_mode == .clone) {
            if (!source.isPaged() or source.isKvCompacted() or source.total_tokens != source_tokens.len)
                return error.UnsupportedDeviceKvCloneLayout;
            const source_storage = source.kv_storage orelse return error.DeviceKvCloneUnsupported;
            if (source_storage.storage.config.dtype != dtype) return error.InvalidDeviceKvClone;
            const source_view = source.kv_view orelse return error.InvalidDeviceKvClone;
            if (source_view.position_offset != 0 or source_view.token_count != source_tokens.len or
                source.sequence_id != source_view.sequence_id) return error.UnsupportedDeviceKvCloneLayout;
            const source_blocks = source_view.logical_blocks orelse return error.UnsupportedDeviceKvCloneLayout;
            try shadow.notePrefill(source_tokens.len);
            const clone_request: runtime.kv.device_clone.Request = .{ .source_sequence = source_view.sequence_id, .destination_sequence = shadow.sequence_id.?, .tokens = source_tokens.len, .capacity = shadow_rows, .page_size = pool_config.page_size_tokens, .source_blocks = source_blocks, .destination_blocks = shadow.kv_block_ids.items, .donors = donors[0..donor_count], .max_bytes = options.max_shadow_kv_bytes, .check_ctx = self, .check_fn = checkAttentionMatchQueryWork };
            const fitted_clone = if (comptime build_options.enable_cuda) blk: {
                const compute: *cuda_compute.CudaCompute = @ptrCast(@alignCast(self.context.backend.ptr));
                break :blk try cuda_compute.cloneInstalledAttentionMatch(compute, &storage, source_storage, clone_request);
            } else null;
            const clone = if (fitted_clone) |value| blk: {
                // The reference mode follows the installed bias/coverage-aware
                // backend. Dense exact-repeat kernel options cannot describe it.
                if (options.repeat_mode != .reference) return error.UnsupportedCompactionQueryRoute;
                ragged_shadow = true;
                break :blk value;
            } else try storage.device_write_hook.?.cloneExactPrefix(source_storage.device_write_hook orelse return error.DeviceKvCloneUnsupported, clone_request);
            if (options.stats) |stats| {
                stats.prefix_ns = mtpProfileElapsedNs(true, self.context.io, prefix_started);
                stats.cloned_bytes = clone.copied_bytes;
                stats.shadow_allocation_bound_bytes = clone.allocation_bound_bytes;
            }
        } else if (options.stats) |stats| stats.shadow_allocation_bound_bytes = shadow_kv_bytes;
        const first_pass: usize = if (options.prefix_mode == .clone) 1 else 0;
        const profile_repeat = platform.env.getenvBool("ANTFLY_INFERENCE_CUDA_PROFILE_PREFILL_OPS");
        for (first_pass..2) |pass| {
            var capture_metrics = gpt_arch.DecodeContext.SelectionQueryCapture.Metrics{};
            const profile_before = if (comptime build_options.enable_cuda) blk: {
                const compute: *cuda_compute.CudaCompute = @ptrCast(@alignCast(self.context.backend.ptr));
                break :blk if (profile_repeat) compute.attentionMatchProfile() else cuda_compute.CudaCompute.AttentionMatchProfile{};
            } else {};
            var repeat_scope: if (build_options.enable_cuda) ?cuda_compute.CudaCompute.AttentionMatchRepeatScope else ?void = null;
            if (comptime build_options.enable_cuda) {
                if (pass == 1 and options.repeat_mode != .reference and !ragged_shadow) {
                    const compute: *cuda_compute.CudaCompute = @ptrCast(@alignCast(self.context.backend.ptr));
                    repeat_scope = try compute.beginAttentionMatchRepeat(options.repeat_mode == .exact_split, options.chunk_size);
                    compute.kernels.attention_match_repeat_check_ctx = self;
                    compute.kernels.attention_match_repeat_check_fn = checkAttentionMatchQueryWork;
                }
            }
            defer if (comptime build_options.enable_cuda) {
                if (repeat_scope) |*scope| scope.deinit();
            };
            const pass_started = mtpProfileTimestamp(true, self.context.io);
            const pass_tokens = if (pass == 0) source_tokens else repeat_tokens;
            var offset: usize = 0;
            while (offset < pass_tokens.len) {
                try checkAttentionMatchQueryWork(self);
                const end = offset + @min(options.chunk_size, pass_tokens.len - offset);
                try shadow.appendPrefillChunk(end - offset);
                var context = shadow.gptDecodeContext(shadow.total_tokens, end - offset);
                if (pass == 1) context.selection_query_capture = .{
                    .ctx = &collector,
                    .allocator = self.context.allocator,
                    .horizon = options.chunk_size,
                    .captureFn = queries.Collector.capture,
                    .metrics = if (profile_repeat) &capture_metrics else null,
                };
                try gpt_arch.forwardHiddenOnly(self.context.backend, self.context.allocator, self.context.model.*, pass_tokens[offset..end], 1, shadow.total_tokens, &context);
                offset = end;
            }
            if (comptime build_options.enable_cuda) {
                if (ragged_shadow) {
                    const compute: *cuda_compute.CudaCompute = @ptrCast(@alignCast(self.context.backend.ptr));
                    const activity = (try cuda_compute.installedAttentionMatchActivity(compute, &storage, shadow.sequence_id.?)) orelse return error.UnsupportedCompactionQueryRoute;
                    if (activity.invalid or activity.attention_calls == 0 or activity.donor_appends == 0) return error.UnsupportedCompactionQueryRoute;
                    if (options.stats) |stats| stats.ragged_repeat_calls = activity.attention_calls;
                }
                if (repeat_scope) |scope| {
                    const launches = scope.compute.kernels.attention_match_repeat_launches;
                    if (launches == 0) return error.UnsupportedCompactionQueryRoute;
                    if (options.repeat_mode == .exact_split and scope.compute.kernels.attention_match_repeat_split_launches == 0)
                        return error.UnsupportedCompactionQueryRoute;
                    if (options.stats) |stats| {
                        stats.repeat_kernel_launches = launches;
                        stats.split_kernel_launches = scope.compute.kernels.attention_match_repeat_split_launches;
                        stats.scratch_bytes = scope.compute.kernels.attention_match_repeat_scratch.len;
                    }
                    try scope.compute.ctx.synchronize();
                }
            }
            if (comptime build_options.enable_cuda) {
                if (profile_repeat) {
                    const compute: *cuda_compute.CudaCompute = @ptrCast(@alignCast(self.context.backend.ptr));
                    const after = compute.attentionMatchProfile();
                    std.log.info("attention_match_phase phase={s} repeat_mode={s} attention_us={d} other_gpu_us={d} downloads={d} download_bytes={d} allocations={d} events={d} capture_calls={d} capture_bytes={d} capture_host_wait_ns={d} capture_callback_ns={d}", .{
                        if (pass == 0) @as([]const u8, "prefix_replay") else "repeat", @tagName(options.repeat_mode),
                        after.attention_us - profile_before.attention_us,              after.other_gpu_us - profile_before.other_gpu_us,
                        after.downloads - profile_before.downloads,                    after.download_bytes - profile_before.download_bytes,
                        after.allocations - profile_before.allocations,                after.events - profile_before.events,
                        capture_metrics.calls,                                         capture_metrics.bytes,
                        capture_metrics.host_wait_ns,                                  capture_metrics.callback_ns,
                    });
                }
            }
            if (options.stats) |stats| {
                const elapsed = mtpProfileElapsedNs(true, self.context.io, pass_started);
                if (pass == 0) stats.prefix_ns = elapsed else stats.repeat_ns = elapsed;
            }
        }
        _ = try collector.finish();
        if (self.context.execution.control) |control| try control.check();
        cleanup_started = mtpProfileTimestamp(true, self.context.io);
        return collector;
    }
};

fn checkAttentionMatchQueryWork(raw: ?*anyopaque) !void {
    const owner: *Runtime = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
    try owner.context.work_control.check();
}

test "checkpoint shadow provisioning failure retires metadata and execution ownership at every allocation failure" {
    if (comptime !build_options.enable_native) return error.SkipZigTest;
    const native = @import("../ops/native_compute.zig");
    const Case = struct {
        fn run(allocator: std.mem.Allocator) !void {
            var lane = execution_control.ExecutionLane{};
            var mutex = std.atomic.Mutex.unlocked;
            defer std.testing.expect(!lane.owned) catch @panic("checkpoint execution lane leaked");
            const Probe = struct {
                lane: *execution_control.ExecutionLane,
                mutex: *std.atomic.Mutex,
                entered: bool = false,
                fn kind(_: *anyopaque) ops.BackendKind {
                    return .cuda;
                }
                fn provision(raw: *anyopaque, _: *runtime.kv.storage_runtime.KvStorageRuntime) !void {
                    const self: *@This() = @ptrCast(@alignCast(raw));
                    self.entered = true;
                    try std.testing.expect(self.lane.owned);
                    try std.testing.expect(!self.mutex.tryLock());
                    return error.InjectedCheckpointProvisionFailure;
                }
            };
            var probe = Probe{ .lane = &lane, .mutex = &mutex };
            var vtable = native.vtable_impl;
            vtable.backendKind = Probe.kind;
            vtable.provisionKvDeviceWriteHook = Probe.provision;
            const backend = ops.ComputeBackend{ .ptr = &probe, .vtable = &vtable };
            const config = gpt_mod.Config{ .family = .gemma, .num_hidden_layers = 2, .num_attention_heads = 2, .num_key_value_heads = 1, .attention_head_dim = 8 };
            var executor = Runtime{ .context = .{ .allocator = allocator, .io = std.testing.io, .backend = &backend, .model = &config, .kv_dtype = .f16, .identity = null, .execution = .{ .mutex = &mutex, .gate = &lane, .io = std.testing.io } } };
            var shadow = Shadow{};
            defer shadow.deinit();
            shadow.initEmpty(&executor) catch |err| switch (err) {
                error.InjectedCheckpointProvisionFailure => {},
                else => return err,
            };
            try std.testing.expect(probe.entered);
            try std.testing.expect(!lane.owned);
            try std.testing.expect(mutex.tryLock());
            mutex.unlock();
            try std.testing.expect(shadow.manager == null and shadow.storage == null and shadow.state == null);
        }
    };
    try @import("antfly_platform").allocator.checkAllAllocationFailures(std.testing.allocator, Case.run, .{});
}

test "checkpoint shadow CUDA owners retain independent positions without retaining the execution lane" {
    if (comptime !build_options.enable_cuda) return error.SkipZigTest;
    const a = std.testing.allocator;
    var backend_owner = @import("../backends/compute_context.zig").BackendContext.init(a, .cuda, null) catch |err| switch (err) {
        error.CudaUnavailable, error.NoCudaDevices => return error.SkipZigTest,
        else => return err,
    };
    defer backend_owner.deinit(a);
    const backend = backend_owner.computeBackend();
    const config = gpt_mod.Config{ .family = .gemma, .hidden_size = 16, .num_hidden_layers = 2, .num_attention_heads = 2, .num_key_value_heads = 1, .attention_head_dim = 8, .intermediate_size = 32, .vocab_size = 8, .max_position_embeddings = 64 };
    var lane = execution_control.ExecutionLane{};
    var mutex = std.atomic.Mutex.unlocked;
    var executor = Runtime{ .context = .{ .allocator = a, .io = std.testing.io, .backend = &backend, .model = &config, .kv_dtype = .f16, .identity = null, .execution = .{ .mutex = &mutex, .gate = &lane, .io = std.testing.io } } };
    var first = Shadow{};
    defer first.deinit();
    try first.initEmpty(&executor);
    try std.testing.expect(!lane.owned);
    var second = Shadow{};
    defer second.deinit();
    try second.initEmpty(&executor);
    try std.testing.expect(!lane.owned);
    try std.testing.expect(&first.storage.? != &second.storage.?);
    try first.state.?.notePrefill(1);
    try second.state.?.notePrefill(3);
    try std.testing.expectEqual(@as(usize, 1), first.state.?.total_tokens);
    try std.testing.expectEqual(@as(usize, 3), second.state.?.total_tokens);
    try std.testing.expect(!first.storage.?.storage.config.store_cpu_bytes);
    try std.testing.expect(first.storage.?.storage.config.sliding_window_size == null);
}

test "checkpoint shadow queries reject typed automatic replay before reading source KV" {
    if (comptime !build_options.enable_cuda) return error.SkipZigTest;
    const Probe = struct {
        fn kind(_: *anyopaque) ops.BackendKind {
            return .cuda;
        }
        fn requested(_: *anyopaque) bool {
            return true;
        }
    };
    var marker: u8 = 0;
    var table = @import("../ops/native_compute.zig").vtable_impl;
    table.backendKind = Probe.kind;
    table.regions.replay.requested = Probe.requested;
    const backend = ops.ComputeBackend{ .ptr = &marker, .vtable = &table };
    const config = gpt_mod.Config{ .family = .gemma, .ple_hidden_size = 256 };
    var executor = Runtime{ .context = .{ .allocator = std.testing.allocator, .io = std.testing.io, .backend = &backend, .model = &config, .kv_dtype = .f16, .identity = null } };
    // No KV/allocator callback is permitted before the replay exclusion.
    var source: NativeDecodeState = undefined;
    try std.testing.expectError(error.UnsupportedCompactionQueryReplay, executor.collectAttentionMatchQueries(&.{ 1, 2 }, &source, .{}));
}
