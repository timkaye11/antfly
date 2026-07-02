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

// Training binary for the fused chunker-embedder.
//
// Usage:
//   train-fused-chunker --data <path> --output <dir> [options]
//
// Options:
//   --data <path>           JSONL data path (file or directory)
//   --output <dir>          Output directory for checkpoints
//   --model-dir <dir>       Model directory (for future encoder loading, currently unused)
//   --epochs <n>            Number of epochs (default: 10)
//   --batch-size <n>        Batch size (default: 16)
//   --lr <f>                Learning rate (default: 1e-4)
//   --warmup-steps <n>      Linear warmup steps (default: 200)
//   --lr-total-steps <n>    Override cosine LR schedule total steps (default: epochs * steps/epoch)
//   --hidden-size <n>       Encoder hidden size (default: 768)
//   --max-seq-len <n>       Max token sequence length (default: 384)
//   --max-chunks <n>        Max chunks per sample (default: 32)
//   --checkpoint-every <n>  Save checkpoint every N epochs (default: 0=disabled)
//   --log-every <n>         Print every N steps (default: 1, 0=epoch summaries only)
//   --split <name>          Dataset split name filter (default: "train")
//   --seed <n>              Random seed (default: 42)
//   --lora-rank <n>         LoRA rank (default: 0 = disabled)
//   --intermediate-size <n> ModernBERT intermediate_size (default: 1152)
//   --backend native|metal|auto Select compute backend (default: auto)

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const platform = @import("antfly_platform");
const native_compute = @import("../../ops/native_compute.zig");
const metal_compute = if (build_options.enable_metal) @import("../../ops/metal_compute.zig") else struct {};
const gpu_hosted_store = @import("../../ops/gpu_hosted_store.zig");
const metal_runtime = if (build_options.enable_metal) @import("../../backends/metal_runtime.zig") else struct {
    pub fn metalDeviceAvailable() bool {
        return false;
    }
    pub fn mpsGraphAvailable() bool {
        return false;
    }
    pub fn mpsGraphSmokeAdd(_: f32, _: f32) ?f32 {
        return null;
    }
};
const ops_mod = @import("../../ops/ops.zig");
const ComputeBackend = ops_mod.ComputeBackend;
const fused_chunker_train = @import("../fused_chunker_train.zig");
const fused_chunker_data = @import("../fused_chunker_data.zig");
const fused_chunker_mod = @import("../fused_chunker.zig");
const fused_chunker_splade = @import("../fused_chunker_splade.zig");
const fused_chunker_loss = @import("../fused_chunker_loss.zig");
const infonce_cpu = @import("../infonce_cpu.zig");
const safetensors_checkpoint = @import("../safetensors_checkpoint.zig");
const compat = @import("../../io/compat.zig");
const modern_bert = @import("../../architectures/modern_bert.zig");
const modern_bert_graph = @import("../../architectures/modern_bert_graph.zig");
const weight_source_mod = @import("../../models/weight_source.zig");
const SafetensorsSource = weight_source_mod.SafetensorsSource;
const LoadedWeight = weight_source_mod.LoadedWeight;
const tokenizer_batch_mod = @import("../tokenizer_batch.zig");
const TokenizerBatch = tokenizer_batch_mod.TokenizerBatch;
const TokenFnCtx = tokenizer_batch_mod.TokenFnCtx;
const fused_chunker_lora = @import("../lora_adapter_set.zig");
const tensor_mod = @import("../../backends/tensor.zig");
const safetensors = @import("../../models/safetensors.zig");
const lora = @import("../lora.zig");
const graph_input_binder = @import("../graph_input_binder.zig");
const segmented_encoder = @import("../../graph/segmented_encoder.zig");
const graph_training = @import("../../graph/training.zig");
const debug_timing = @import("../../debug_timing.zig");
const ml = @import("ml");

const fused_chunker_lora_target_modules = [_][]const u8{
    "query_proj",
    "value_proj",
    "key_proj",
    "out_proj",
    "wo",
};
const optimizers = ml.graph.optimizers;

const FusedTrainer = fused_chunker_train.FusedTrainer;
const FusedTrainingConfig = fused_chunker_train.FusedTrainingConfig;
const TrainStepSummary = fused_chunker_train.TrainStepSummary;
const MetalWeightStore = if (build_options.enable_metal) gpu_hosted_store.WeightStore else void;

const print = std.debug.print;

const fused_manifest_file_name = "fused_training_manifest.json";
const fused_metrics_file_name = "fused_training_metrics.jsonl";
const fused_manifest_schema_version = "fused_chunker_training/v1";
const fused_artifact_family_version = "fused_chunker_phase20/v1";
const fnv_offset: u64 = 14695981039346656037;
const fnv_prime: u64 = 1099511628211;

// ---------------------------------------------------------------------------
// CLI options
// ---------------------------------------------------------------------------

const FusedBackend = enum {
    auto,
    metal,
    native,
};

const BoundaryLossType = enum {
    ce,
    focal,
};

fn parseBoundaryFeatureMode(raw: []const u8) !fused_chunker_train.BoundaryFeatureMode {
    if (std.mem.eql(u8, raw, "token")) return .token;
    if (std.mem.eql(u8, raw, "prev-diff")) return .prev_diff;
    if (std.mem.eql(u8, raw, "prev-current-diff")) return .prev_current_diff;
    if (std.mem.eql(u8, raw, "prev-current-diff-concat")) return .prev_current_diff_concat;
    if (std.mem.eql(u8, raw, "prev-current-next-diff-concat")) return .prev_current_next_diff_concat;
    if (std.mem.eql(u8, raw, "window-context-diff")) return .window_context_diff;
    return error.InvalidBoundaryFeatureMode;
}

const EncoderVJPMode = enum {
    direct,
    last_layer,
    full,
    full_hidden_direct,
};

fn encoderVJPModeName(mode: EncoderVJPMode) []const u8 {
    return switch (mode) {
        .direct => "direct",
        .last_layer => "last-layer",
        .full => "full",
        .full_hidden_direct => "full-hidden-direct",
    };
}

fn usesFullEncoderVJP(mode: EncoderVJPMode) bool {
    return mode == .full or mode == .full_hidden_direct;
}

fn usesExactFullAdapterGrads(mode: EncoderVJPMode) bool {
    return mode == .full;
}

const Options = struct {
    data_path: []const u8,
    output_dir: []const u8,
    val_data_path: ?[]const u8 = null,
    model_dir: ?[]const u8 = null,
    epochs: u32 = 10,
    batch_size: u32 = 16,
    learning_rate: f32 = 1e-4,
    boundary_head_lr_multiplier: f32 = 1.0,
    warmup_steps: u32 = 200,
    lr_total_steps: u32 = 0,
    weight_decay: f32 = 0.01,
    max_grad_norm: f32 = 0.0,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    adam_epsilon: f32 = 1e-8,
    lambda_chunk: f32 = 1.0,
    lambda_embed: f32 = 0.3,
    boundary_focus_epochs: u32 = 3,
    boundary_focus_lambda_embed: f32 = 0.1,
    boundary_focal_gamma: f32 = 2.0,
    boundary_focal_alpha: f32 = 0.75,
    contrastive_focal_gamma: f32 = 0.0,
    contrastive_focal_alpha: f32 = 0.75,
    boundary_pos_weight: f32 = 5.0,
    boundary_pos_weight_auto: bool = false,
    boundary_pos_weight_auto_max_examples: usize = 2048,
    boundary_pos_weight_auto_observed_examples: usize = 0,
    boundary_pos_weight_auto_valid_tokens: u64 = 0,
    boundary_pos_weight_auto_gold_tokens: u64 = 0,
    boundary_pos_weight_auto_gold_rate: f32 = 0.0,
    boundary_rank_loss_weight: f32 = 0.0,
    boundary_rank_loss_margin: f32 = 1.0,
    boundary_rank_loss_top_k: u32 = 1,
    boundary_same_token_rank_loss_weight: f32 = 0.0,
    boundary_same_token_rank_loss_top_k: u32 = 4,
    boundary_same_token_negative_weight: f32 = 1.0,
    boundary_same_token_negative_top_k: u32 = 0,
    boundary_candidate_rank_loss_weight: f32 = 0.0,
    boundary_candidate_rank_loss_top_k: u32 = 8,
    boundary_candidate_negative_weight: f32 = 1.0,
    boundary_gold_count_rank_loss_weight: f32 = 0.0,
    boundary_gold_count_rank_loss_margin: f32 = 1.0,
    boundary_gold_count_rank_loss_negative_multiplier: u32 = 1,
    boundary_local_window_loss_weight: f32 = 0.0,
    boundary_local_window_radius: u32 = 12,
    boundary_feature_mode: fused_chunker_train.BoundaryFeatureMode = .token,
    boundary_dropout: f32 = 0.1,
    boundary_loss_type: BoundaryLossType = .ce,
    hidden_size: u32 = 768,
    num_layers: u32 = 22,
    max_seq_len: u32 = 384,
    max_chunks: u32 = 32,
    checkpoint_every: u32 = 0,
    checkpoint_every_steps: u32 = 0,
    log_every: u32 = 1,
    eval_every: u32 = 1,
    eval_every_steps: u32 = 0,
    step_eval_max_examples: usize = 0,
    step_train_eval_max_examples: usize = 0,
    checkpoint_roundtrip_eval: bool = false,
    checkpoint_roundtrip_max_examples: usize = 0,
    max_steps: u64 = 0,
    split: []const u8 = "train",
    val_split: []const u8 = "val",
    seed: u64 = 42,
    lora_rank: u32 = 0,
    lora_alpha: f32 = 32.0,
    lora_start_epoch: u32 = 0,
    intermediate_size: u32 = 1152,
    backend: FusedBackend = .auto,
    // Feature 2: gradient accumulation
    grad_accum: u32 = 1,
    // Feature 3: schedule-free AdamW
    schedule_free: bool = false,
    // Feature 5: NEFTune
    neftune_alpha: f32 = 0.0,
    encoder_neftune: bool = true,
    // Feature 1: XBM
    xbm_capacity: usize = 0,
    // Feature 6: LLRD
    llrd_decay: f32 = 1.0,
    // Go parity: LISA layer selection. 0 disables it and trains all layers.
    lisa_sample_layers: u32 = 0,
    lisa_top_k: u32 = 5,
    lora_train_top_k: u32 = 0,
    encoder_vjp: EncoderVJPMode = .direct,
    layers_per_segment: u32 = 1,
    // Opt-in compiled segmented encoder forward (P1 fast path). Also settable
    // via ANTFLY_FUSED_CHUNKER_COMPILED_SEGMENT_FORWARD=1.
    compiled_segment_forward: bool = false,
    // Feature 4: LoRA+
    lora_plus_ratio: f32 = 1.0,
    // Feature 8: length bucketing
    length_bucketing: bool = false,
    bucket_size: usize = 256,
    mixed_precision: bool = false,
    // SPLADE sparse embedding head
    splade: bool = false,
    lambda_splade: f32 = 0.15,
    lambda_flops: f32 = 3e-5,
    splade_focus_epoch: u32 = 4,
    // Matryoshka Representation Learning
    mrl: bool = false,
    mrl_dims_str: []const u8 = "768,256,128",
    // Checkpoint resumption
    resume_from: []const u8 = "",
    save_optimizer_state: bool = false,
    deterministic: bool = false,
    go_epoch_shuffle: bool = false,
    report_to: ?[]const u8 = null,
    manifest_path: ?[]const u8 = null,
    boundary_alignment_dump_dir: ?[]const u8 = null,
    boundary_alignment_dump_top_k: usize = 32,
    memory_sample_every: u32 = 1,
    memory_warn_rss_bytes: u64 = 0,
    memory_abort_rss_bytes: u64 = 0,
    debug_first_boundary_step: bool = false,
    debug_first_boundary_step_exit: bool = false,
    debug_boundary_step: u32 = 0,
    debug_boundary_step_exit: bool = false,
    debug_update_step: u32 = 0,
    debug_update_step_exit: bool = false,
    debug_boundary_head_overfit_steps: u32 = 0,
    debug_boundary_head_overfit_lr: f32 = 1e-3,
    debug_frozen_feature_probe: bool = false,
    debug_frozen_feature_train_examples: usize = 64,
    debug_frozen_feature_val_examples: usize = 64,
    debug_frozen_feature_train_offset: usize = 0,
    debug_frozen_feature_val_offset: usize = 0,
    debug_frozen_feature_epochs: u32 = 3,
    debug_frozen_feature_lr: f32 = 5e-3,
    debug_step_json_path: ?[]const u8 = null,
    debug_batch_offset: ?usize = null,
    debug_encoder_probe_layer: u32 = 0,
    debug_encoder_layer_inputs_only: bool = false,
    debug_encoder_replay_input_path: ?[]const u8 = null,
    debug_encoder_replay_upstream_path: ?[]const u8 = null,
    debug_layer_backward_decomp: bool = false,
    debug_qkv_split_vjp: bool = false,
};

const StepTiming = struct {
    total_ns: u64 = 0,
    batch_ns: u64 = 0,
    lora_refresh_ns: u64 = 0,
    encoder_ns: u64 = 0,
    hard_neg_ns: u64 = 0,
    train_ns: u64 = 0,
    lora_update_ns: u64 = 0,
    splade_ns: u64 = 0,
};

const TimingTotals = struct {
    steps: u64 = 0,
    examples: u64 = 0,
    total_ns: u64 = 0,
    batch_ns: u64 = 0,
    lora_refresh_ns: u64 = 0,
    encoder_ns: u64 = 0,
    hard_neg_ns: u64 = 0,
    train_ns: u64 = 0,
    lora_update_ns: u64 = 0,
    splade_ns: u64 = 0,

    fn add(self: *TimingTotals, timing: StepTiming, examples: usize) void {
        self.steps += 1;
        self.examples += @intCast(examples);
        self.total_ns += timing.total_ns;
        self.batch_ns += timing.batch_ns;
        self.lora_refresh_ns += timing.lora_refresh_ns;
        self.encoder_ns += timing.encoder_ns;
        self.hard_neg_ns += timing.hard_neg_ns;
        self.train_ns += timing.train_ns;
        self.lora_update_ns += timing.lora_update_ns;
        self.splade_ns += timing.splade_ns;
    }
};

const ProcessMemorySnapshot = struct {
    available: bool = false,
    resident_bytes: u64 = 0,
    footprint_bytes: u64 = 0,
};

const MemoryTracker = struct {
    peak_resident_bytes: u64 = 0,
    first_resident_bytes: u64 = 0,
    last_resident_bytes: u64 = 0,
    warning_emitted: bool = false,

    fn observe(self: *MemoryTracker) ProcessMemorySnapshot {
        const snapshot = processMemorySnapshot();
        if (!snapshot.available) return snapshot;
        if (self.first_resident_bytes == 0) self.first_resident_bytes = snapshot.resident_bytes;
        self.last_resident_bytes = snapshot.resident_bytes;
        self.peak_resident_bytes = @max(self.peak_resident_bytes, snapshot.resident_bytes);
        return snapshot;
    }
};

const SupervisionCounts = struct {
    valid_tokens: u64 = 0,
    boundary_positive_tokens: u64 = 0,

    fn ignoredTokens(self: SupervisionCounts, total_tokens: usize) u64 {
        const total: u64 = @intCast(total_tokens);
        return if (total > self.valid_tokens) total - self.valid_tokens else 0;
    }
};

fn nowNs() u64 {
    return platform.time.monotonicNs();
}

fn elapsedNs(start_ns: u64) u64 {
    const end_ns = nowNs();
    return if (end_ns > start_ns) end_ns - start_ns else 0;
}

fn selectActiveLoRALayersForStep(
    allocator: std.mem.Allocator,
    opts: *const Options,
    num_layers: u32,
    step: u64,
) ![]bool {
    if (opts.lora_train_top_k > 0) {
        const n: usize = @intCast(num_layers);
        const active = try allocator.alloc(bool, n);
        errdefer allocator.free(active);
        @memset(active, false);
        const top_k: usize = @min(@as(usize, @intCast(opts.lora_train_top_k)), n);
        const start = n - top_k;
        for (start..n) |layer_idx| active[layer_idx] = true;
        return active;
    }
    return fused_chunker_train.selectActiveLoRALayers(
        allocator,
        num_layers,
        opts.lisa_sample_layers,
        opts.lisa_top_k,
        step,
        opts.seed,
    );
}

inline fn isFiniteF32(x: f32) bool {
    return !std.math.isNan(x) and x != std.math.inf(f32) and x != -std.math.inf(f32);
}

inline fn isFiniteF64(x: f64) bool {
    return !std.math.isNan(x) and x != std.math.inf(f64) and x != -std.math.inf(f64);
}

fn getenvU32OrNull(name: [*:0]const u8) ?u32 {
    const raw = platform.env.getenv(name) orelse return null;
    return std.fmt.parseUnsigned(u32, raw, 10) catch null;
}

fn shouldCaptureCompiledForwardLayer(active_layers: ?[]const bool, layer_idx: usize) bool {
    const layers = active_layers orelse return true;
    return layer_idx < layers.len and layers[layer_idx];
}

/// P1: run the ModernBERT encoder forward through compiled MPSGraph segment
/// sessions, populating `act_buf` with the same capture set the eager
/// `modern_bert.forwardCapturingActivations` produces (layer inputs, the
/// Q/K/V / out_proj / mlp "wo" projection inputs for active LoRA layers, and
/// the final-norm captures). Embeddings and the final LayerNorm run through
/// the eager backend ops so their numerics (incl. NEFTune RNG) are unchanged.
/// Returns the owned final feature buffer `[total * hidden]`.
fn runCompiledSegmentedEncoderForward(
    allocator: std.mem.Allocator,
    cb: *const ops_mod.ComputeBackend,
    bert_config: modern_bert.Config,
    graph_config: modern_bert_graph.Config,
    sessions: []?segmented_encoder.ModernBertSegmentForwardSession,
    layers_per_segment: u32,
    lora_rank: u32,
    lora_alpha: f32,
    ids_i64: []const i64,
    mask_i64: []const i64,
    actual_batch: usize,
    max_seq: usize,
    active_lora_layers: ?[]const bool,
    capture_layer_inputs: bool,
    capture_all_layer_inputs: bool,
    lora_layers: []fused_chunker_lora.LoRALayer,
    act_buf: *modern_bert.ActivationBuffer,
) ![]f32 {
    const total_tokens = actual_batch * max_seq;
    const H: usize = @intCast(graph_config.hidden_size);
    const num_layers = graph_config.num_hidden_layers;
    if (sessions.len < num_layers) return error.MissingCompiledForwardSessions;

    const attn_mask_f32 = try allocator.alloc(f32, total_tokens);
    defer allocator.free(attn_mask_f32);
    for (mask_i64[0..total_tokens], attn_mask_f32) |m, *out| out.* = @floatFromInt(m);

    var shared_bias = segmented_encoder.SharedSegmentAttnBias.init(
        allocator,
        @intCast(actual_batch),
        @intCast(max_seq),
        graph_config.num_attention_heads,
        graph_config.local_attention_window,
        attn_mask_f32,
    );
    defer shared_bias.deinit();

    // Embeddings stay on the eager path (token gather + NEFTune + LayerNorm).
    const emb_ct = try modern_bert.embeddingsForwardCT(
        cb,
        allocator,
        bert_config,
        ids_i64,
        total_tokens,
        max_seq,
        act_buf,
    );
    const embedding_host = blk: {
        defer cb.free(emb_ct);
        break :blk try cb.toFloat32(emb_ct, allocator);
    };
    var hidden_owned: []f32 = embedding_host;
    defer allocator.free(hidden_owned);
    if (hidden_owned.len != total_tokens * H) return error.UnexpectedOutputShape;

    var segment_start: u32 = 0;
    while (segment_start < num_layers) {
        const segment_end = @min(segment_start + layers_per_segment, num_layers);

        const session_slot = &sessions[@intCast(segment_start)];
        var rebuild_session = session_slot.* == null;
        if (session_slot.*) |*session| {
            if (session.batch != actual_batch or
                session.seq_len != max_seq or
                session.start_layer != segment_start or
                session.end_layer != segment_end or
                session.config.hidden_size != graph_config.hidden_size or
                session.config.intermediate_size != graph_config.intermediate_size)
            {
                session.deinit();
                session_slot.* = null;
                rebuild_session = true;
            }
        }
        if (rebuild_session) {
            session_slot.* = try segmented_encoder.ModernBertSegmentForwardSession.init(
                allocator,
                graph_config,
                @intCast(actual_batch),
                @intCast(max_seq),
                segment_start,
                segment_end,
                lora_rank,
                lora_alpha,
            );
        }
        const session = &(session_slot.* orelse return error.MissingCompiledForwardSessions);

        var result = try session.execute(cb, hidden_owned, attn_mask_f32, lora_layers, &shared_bias);
        defer result.deinit();

        for (result.layers, 0..) |capture, idx| {
            const layer_idx: usize = @intCast(capture.layer_idx);
            const capture_layer = shouldCaptureCompiledForwardLayer(active_lora_layers, layer_idx);
            const capture_layer_input = capture_layer_inputs and
                (capture_all_layer_inputs or capture_layer);
            if (capture_layer_input) {
                const layer_input: []const f32 = if (idx == 0)
                    hidden_owned
                else
                    result.layers[idx - 1].hidden_out;
                try act_buf.addLayerInput(capture.layer_idx, layer_input, total_tokens, H);
            }
            if (!capture_layer) continue;
            // Same capture set as the eager path: normed_attn feeds Q/K/V,
            // attn_merged feeds attn.Wo ("out_proj"), mlp_activated feeds
            // mlp.Wo ("wo").
            try act_buf.add(capture.layer_idx, "wo", capture.mlp_activated, @intCast(graph_config.intermediate_size), H, total_tokens);
            try act_buf.add(capture.layer_idx, "query_proj", capture.attn_normed, H, H, total_tokens);
            const shared_normed = act_buf.items.items[act_buf.items.items.len - 1].input;
            try act_buf.addAlias(capture.layer_idx, "key_proj", shared_normed, H, H, total_tokens);
            try act_buf.addAlias(capture.layer_idx, "value_proj", shared_normed, H, H, total_tokens);
            try act_buf.add(capture.layer_idx, "out_proj", capture.attn_merged, H, H, total_tokens);
        }

        const next_hidden = try allocator.dupe(f32, result.layers[result.layers.len - 1].hidden_out);
        allocator.free(hidden_owned);
        hidden_owned = next_hidden;
        segment_start = segment_end;
    }

    // Final LayerNorm through the same backend op as the eager forward.
    const hidden_ct = try cb.fromFloat32Shape(hidden_owned, &.{ @intCast(total_tokens), @intCast(H) });
    defer cb.free(hidden_ct);
    const fn_w = try cb.getWeight("model.final_norm.weight");
    defer cb.free(fn_w);
    const fn_b = cb.getWeight("model.final_norm.bias") catch |err| blk: {
        if (err != error.MissingWeight) return err;
        const zeros = try allocator.alloc(f32, H);
        defer allocator.free(zeros);
        @memset(zeros, 0);
        break :blk try cb.fromFloat32Shape(zeros, &.{@as(i32, @intCast(H))});
    };
    defer cb.free(fn_b);
    const normed_ct = try cb.layerNorm(hidden_ct, fn_w, fn_b, H, bert_config.layer_norm_eps);
    defer cb.free(normed_ct);
    const features = try cb.toFloat32(normed_ct, allocator);
    errdefer allocator.free(features);

    const final_norm_input = try allocator.dupe(f32, hidden_owned);
    var owns_final_norm_input = true;
    errdefer if (owns_final_norm_input) allocator.free(final_norm_input);
    const final_norm_weight = try cb.toFloat32(fn_w, allocator);
    var owns_final_norm_weight = true;
    errdefer if (owns_final_norm_weight) allocator.free(final_norm_weight);
    try act_buf.setFinalNormOwned(final_norm_input, final_norm_weight, total_tokens, H);
    owns_final_norm_input = false;
    owns_final_norm_weight = false;

    return features;
}

/// Default parity tolerance for the compiled-vs-eager forward check.
///
/// The two paths intentionally run DIFFERENT f32 implementations of the same
/// math (eager: host sgemm + fused Metal SDPA/LayerNorm/RoPE kernels;
/// compiled: MPSGraph kernels), so they cannot agree to f32 ulp precision.
/// Measured noise floor with kernel-exact RoPE tables (deterministic probe,
/// batch 8 x seq 384): ~1e-5 max-rel at layer 0 growing to ~1.4e-2 at layer
/// 21 through 22 layers of residual-stream amplification. The default is set
/// a small margin above that floor; genuine implementation bugs (wrong rope
/// convention, mask, weight wiring, ...) produce O(0.1..1) divergence and
/// still fail loudly. Override with
/// ANTFLY_FUSED_CHUNKER_COMPILED_FORWARD_CHECK_TOL for tighter probes on
/// shallow segments.
const compiled_forward_check_default_tol: f32 = 5e-2;

fn compiledForwardCheckTolerance() f32 {
    const raw = platform.env.getenv("ANTFLY_FUSED_CHUNKER_COMPILED_FORWARD_CHECK_TOL") orelse
        return compiled_forward_check_default_tol;
    return std.fmt.parseFloat(f32, raw) catch compiled_forward_check_default_tol;
}

/// ANTFLY_FUSED_CHUNKER_COMPILED_FORWARD_CHECK_DEBUG=1: print every tensor's
/// max_rel and keep going past mismatches so a single run localizes where the
/// compiled forward first diverges from the eager forward.
fn compiledForwardCheckDebug() bool {
    return platform.env.getenvBoolDefault("ANTFLY_FUSED_CHUNKER_COMPILED_FORWARD_CHECK_DEBUG", false);
}

/// Elementwise comparison of a compiled-vs-eager tensor pair, restricted to
/// rows whose token is unmasked (padded rows may legitimately diverge because
/// the graph path uses a -1e9 additive bias where the eager path uses -inf).
fn verifyCompiledForwardTensor(
    name: []const u8,
    layer_idx: u32,
    compiled: []const f32,
    eager: []const f32,
    cols: usize,
    mask_i64: []const i64,
    tol: f32,
) !void {
    if (compiled.len != eager.len) {
        print(
            "compiled_forward_check mismatch tensor={s} layer={d} reason=length compiled={d} eager={d}\n",
            .{ name, layer_idx, compiled.len, eager.len },
        );
        return error.CompiledForwardParityMismatch;
    }
    if (cols == 0 or compiled.len % cols != 0) return error.CompiledForwardParityMismatch;
    const rows = compiled.len / cols;
    var max_rel: f64 = 0.0;
    var max_index: usize = 0;
    for (0..rows) |row| {
        if (row < mask_i64.len and mask_i64[row] == 0) continue;
        for (0..cols) |col| {
            const i = row * cols + col;
            const a = compiled[i];
            const b = eager[i];
            const denom = @max(1.0, @max(@abs(a), @abs(b)));
            const rel = @abs(a - b) / denom;
            if (rel > max_rel) {
                max_rel = rel;
                max_index = i;
            }
        }
    }
    if (compiledForwardCheckDebug()) {
        print(
            "compiled_forward_check detail tensor={s} layer={d} max_rel={e:.6}\n",
            .{ name, layer_idx, max_rel },
        );
    }
    if (max_rel > tol) {
        print(
            "compiled_forward_check mismatch tensor={s} layer={d} max_rel={e:.6} tol={e:.6} index={d} compiled={e:.9} eager={e:.9}\n",
            .{ name, layer_idx, max_rel, tol, max_index, compiled[max_index], eager[max_index] },
        );
        if (compiledForwardCheckDebug()) return;
        return error.CompiledForwardParityMismatch;
    }
}

/// First-step validation gate (ANTFLY_FUSED_CHUNKER_COMPILED_FORWARD_CHECK=1):
/// compare the compiled forward's features and captured activations against
/// the eager capture forward and error loudly on any divergence above the
/// tolerance (see `compiled_forward_check_default_tol`; override with
/// ANTFLY_FUSED_CHUNKER_COMPILED_FORWARD_CHECK_TOL).
fn verifyCompiledForwardParity(
    compiled_features: []const f32,
    eager_features: []const f32,
    compiled_buf: *const modern_bert.ActivationBuffer,
    eager_buf: *const modern_bert.ActivationBuffer,
    mask_i64: []const i64,
    hidden: usize,
) !void {
    const tol = compiledForwardCheckTolerance();

    if (compiled_buf.items.items.len != eager_buf.items.items.len) {
        print(
            "compiled_forward_check mismatch tensor=items reason=count compiled={d} eager={d}\n",
            .{ compiled_buf.items.items.len, eager_buf.items.items.len },
        );
        return error.CompiledForwardParityMismatch;
    }
    for (compiled_buf.items.items) |*capture| {
        const eager_capture = blk: {
            for (eager_buf.items.items) |*candidate| {
                if (candidate.layer_idx == capture.layer_idx and
                    std.mem.eql(u8, candidate.module_name, capture.module_name))
                {
                    break :blk candidate;
                }
            }
            print(
                "compiled_forward_check mismatch tensor={s} layer={d} reason=missing_eager_capture\n",
                .{ capture.module_name, capture.layer_idx },
            );
            return error.CompiledForwardParityMismatch;
        };
        try verifyCompiledForwardTensor(
            capture.module_name,
            capture.layer_idx,
            capture.input,
            eager_capture.input,
            capture.in_features,
            mask_i64,
            tol,
        );
    }

    if (compiled_buf.layer_inputs.items.len != eager_buf.layer_inputs.items.len) {
        print(
            "compiled_forward_check mismatch tensor=layer_inputs reason=count compiled={d} eager={d}\n",
            .{ compiled_buf.layer_inputs.items.len, eager_buf.layer_inputs.items.len },
        );
        return error.CompiledForwardParityMismatch;
    }
    for (compiled_buf.layer_inputs.items) |*capture| {
        const eager_capture = eager_buf.findLayerInput(capture.layer_idx) orelse {
            print(
                "compiled_forward_check mismatch tensor=layer_input layer={d} reason=missing_eager_capture\n",
                .{capture.layer_idx},
            );
            return error.CompiledForwardParityMismatch;
        };
        try verifyCompiledForwardTensor(
            "layer_input",
            capture.layer_idx,
            capture.input,
            eager_capture.input,
            capture.hidden,
            mask_i64,
            tol,
        );
    }

    const compiled_final = compiled_buf.final_norm_input orelse return error.CompiledForwardParityMismatch;
    const eager_final = eager_buf.final_norm_input orelse return error.CompiledForwardParityMismatch;
    try verifyCompiledForwardTensor("final_norm_input", 0, compiled_final, eager_final, hidden, mask_i64, tol);

    // Checked last (it is the deepest tensor): a mismatch in any earlier
    // capture localizes the divergence to a specific layer/module instead
    // of only reporting the accumulated error at the network output.
    try verifyCompiledForwardTensor("features", 0, compiled_features, eager_features, hidden, mask_i64, tol);
}

fn segmentVJPParityReferenceStrategy() graph_training.CompiledExecutionStrategy {
    const raw = platform.env.getenv("TERMITE_SEGMENT_VJP_PARITY_REFERENCE") orelse return .partitioned_required;
    if (std.mem.eql(u8, raw, "interpreter")) return .interpreter;
    if (std.mem.eql(u8, raw, "partitioned") or std.mem.eql(u8, raw, "partitioned_required") or
        std.mem.eql(u8, raw, "metal") or std.mem.eql(u8, raw, "metal_required"))
    {
        return .partitioned_required;
    }
    if (std.mem.eql(u8, raw, "mpsgraph") or std.mem.eql(u8, raw, "mpsgraph_required")) return .mpsgraph_required;
    return .partitioned_required;
}

fn segmentVJPStrategyName(strategy: graph_training.CompiledExecutionStrategy) []const u8 {
    return switch (strategy) {
        .interpreter => "interpreter",
        .partitioned_preferred => "partitioned_preferred",
        .partitioned_required => "partitioned_required",
        .mpsgraph_preferred => "mpsgraph_preferred",
        .mpsgraph_required => "mpsgraph_required",
    };
}

fn segmentVJPProfileRuntimeName(profile: segmented_encoder.SegmentVJPProfile) []const u8 {
    if (profile.mpsgraph_runtime) return "mpsgraph";
    if (profile.partitioned_runtime) return "partitioned";
    return "interpreter";
}

fn isFiniteTrainStepSummary(summary: TrainStepSummary) bool {
    return isFiniteF32(summary.boundary_loss) and
        isFiniteF32(summary.boundary_ce_loss) and
        isFiniteF32(summary.boundary_rank_loss) and
        isFiniteF32(summary.boundary_local_window_loss) and
        isFiniteF64(summary.contrastive_loss) and
        isFiniteF32(summary.total_loss) and
        isFiniteF32(summary.learning_rate) and
        isFiniteF32(summary.boundary_head_learning_rate);
}

fn sanitizeLoRAAdapterSet(adapters: *fused_chunker_lora.LoRAAdapterSet) usize {
    var repaired: usize = 0;
    for (adapters.layers) |*ll| {
        repaired += optimizers.sanitizeSlice(ll.A);
        repaired += optimizers.sanitizeSlice(ll.B);
        repaired += optimizers.sanitizeSlice(ll.grad_A);
        repaired += optimizers.sanitizeSlice(ll.grad_B);
        if (ll.magnitude) |magnitude| repaired += optimizers.sanitizeSlice(magnitude);
        if (ll.grad_magnitude) |grad_magnitude| repaired += optimizers.sanitizeSlice(grad_magnitude);
    }
    return repaired;
}

const FloatSliceStats = struct {
    elems: usize = 0,
    nonzero: usize = 0,
    sum_abs: f64 = 0,
    sum_sq: f64 = 0,
    max_abs: f32 = 0,

    fn addValue(self: *FloatSliceStats, value: f32) void {
        const abs_value = absF32(value);
        self.elems += 1;
        if (value != 0.0) self.nonzero += 1;
        self.sum_abs += @as(f64, @floatCast(abs_value));
        self.sum_sq += @as(f64, value) * @as(f64, value);
        self.max_abs = @max(self.max_abs, abs_value);
    }

    fn addSlice(self: *FloatSliceStats, values: []const f32) void {
        for (values) |value| self.addValue(value);
    }

    fn addDelta(self: *FloatSliceStats, before: []const f32, after: []const f32) !void {
        if (before.len != after.len) return error.UpdateDiagnosticShapeMismatch;
        for (before, after) |old, new| self.addValue(new - old);
    }

    fn l2(self: FloatSliceStats) f64 {
        return @sqrt(self.sum_sq);
    }

    fn meanAbs(self: FloatSliceStats) f64 {
        if (self.elems == 0) return 0.0;
        return self.sum_abs / @as(f64, @floatFromInt(self.elems));
    }
};

const StepParityTensorSliceStats = struct {
    stats: FloatSliceStats = .{},
    sample: [16]f32 = [_]f32{0} ** 16,
    sample_len: usize = 0,

    fn addValue(self: *StepParityTensorSliceStats, value: f32) void {
        if (self.sample_len < self.sample.len) {
            self.sample[self.sample_len] = value;
            self.sample_len += 1;
        }
        self.stats.addValue(value);
    }

    fn addSlice(self: *StepParityTensorSliceStats, values: []const f32) void {
        for (values) |value| self.addValue(value);
    }

    fn addStats(self: *StepParityTensorSliceStats, other: StepParityTensorSliceStats) void {
        self.stats.elems += other.stats.elems;
        self.stats.nonzero += other.stats.nonzero;
        self.stats.sum_abs += other.stats.sum_abs;
        self.stats.sum_sq += other.stats.sum_sq;
        self.stats.max_abs = @max(self.stats.max_abs, other.stats.max_abs);
        const remaining = self.sample.len - self.sample_len;
        const copy_len = @min(remaining, other.sample_len);
        if (copy_len > 0) {
            @memcpy(self.sample[self.sample_len .. self.sample_len + copy_len], other.sample[0..copy_len]);
            self.sample_len += copy_len;
        }
    }
};

const StepParitySoftmaxVJPCase = struct {
    name: []const u8,
    status: []const u8 = "captured",
    reason: []const u8 = "",
    outer: usize = 0,
    queries: usize = 0,
    keys: usize = 0,
    mask_bias: f32 = 0.0,
    has_mask: bool = false,
    scores_masked: StepParityTensorSliceStats = .{},
    probs: StepParityTensorSliceStats = .{},
    upstream_probs_grad: StepParityTensorSliceStats = .{},
    scores_masked_grad: StepParityTensorSliceStats = .{},
    cpu_scores_masked_grad: StepParityTensorSliceStats = .{},
    cpu_abs_error: StepParityTensorSliceStats = .{},
    valid_scores_masked_grad: StepParityTensorSliceStats = .{},
    masked_scores_masked_grad: StepParityTensorSliceStats = .{},
};

const StepParitySoftmaxVJPProbe = struct {
    status: []const u8 = "captured",
    version: u32 = 1,
    runtime: []const u8 = "mpsgraph",
    cases: []StepParitySoftmaxVJPCase = &.{},

    fn deinit(self: *StepParitySoftmaxVJPProbe, allocator: std.mem.Allocator) void {
        if (self.cases.len > 0) allocator.free(self.cases);
        self.* = undefined;
    }
};

const StepParityNamedTensorStats = struct {
    name: []u8,
    stats: StepParityTensorSliceStats,

    fn deinit(self: *StepParityNamedTensorStats, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        self.* = undefined;
    }
};

const StepParityQKVSplitVJPCase = struct {
    name: []const u8,
    status: []const u8 = "captured",
    reason: []const u8 = "",
    batch: usize = 0,
    seq_len: usize = 0,
    num_heads: usize = 0,
    head_dim: usize = 0,
    hidden_size: usize = 0,
    outer: usize = 0,
    components: []StepParityNamedTensorStats = &.{},

    fn deinit(self: *StepParityQKVSplitVJPCase, allocator: std.mem.Allocator) void {
        for (self.components) |*entry| entry.deinit(allocator);
        if (self.components.len > 0) allocator.free(self.components);
        self.* = undefined;
    }
};

const StepParityQKVSplitVJPProbe = struct {
    status: []const u8 = "captured",
    version: u32 = 1,
    runtime: []const u8 = "mpsgraph",
    cases: []StepParityQKVSplitVJPCase = &.{},

    fn deinit(self: *StepParityQKVSplitVJPProbe, allocator: std.mem.Allocator) void {
        for (self.cases) |*case| case.deinit(allocator);
        if (self.cases.len > 0) allocator.free(self.cases);
        self.* = undefined;
    }
};

const StepParitySegmentVJPProbe = struct {
    target_layer: u32 = 0,
    segment_start: u32 = 0,
    segment_end: u32 = 0,
    include_hidden_grad: bool = false,
    include_adapter_grads: bool = false,
    runtime: []const u8 = "unknown",
    profile: segmented_encoder.SegmentVJPProfile = .{},
    upstream: StepParityTensorSliceStats = .{},
    hidden_grad: StepParityTensorSliceStats = .{},
    adapter_a: StepParityTensorSliceStats = .{},
    adapter_b: StepParityTensorSliceStats = .{},
    adapter_a_by_name: []StepParityNamedTensorStats = &.{},
    adapter_b_by_name: []StepParityNamedTensorStats = &.{},

    fn deinit(self: *StepParitySegmentVJPProbe, allocator: std.mem.Allocator) void {
        for (self.adapter_a_by_name) |*entry| entry.deinit(allocator);
        allocator.free(self.adapter_a_by_name);
        for (self.adapter_b_by_name) |*entry| entry.deinit(allocator);
        allocator.free(self.adapter_b_by_name);
        self.* = undefined;
    }
};

const StepParityBackwardDecompComponent = struct {
    name: []u8,
    stats: StepParityTensorSliceStats,

    fn deinit(self: *StepParityBackwardDecompComponent, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        self.* = undefined;
    }
};

const StepParityBackwardDecompStage = struct {
    status: []const u8 = "missing",
    reason: []const u8 = "",
    stats: StepParityTensorSliceStats = .{},
    components: []StepParityBackwardDecompComponent = &.{},
    adapter_a: StepParityTensorSliceStats = .{},
    adapter_b: StepParityTensorSliceStats = .{},
    adapter_a_by_name: []StepParityNamedTensorStats = &.{},
    adapter_b_by_name: []StepParityNamedTensorStats = &.{},

    fn deinit(self: *StepParityBackwardDecompStage, allocator: std.mem.Allocator) void {
        for (self.components) |*entry| entry.deinit(allocator);
        if (self.components.len > 0) allocator.free(self.components);
        for (self.adapter_a_by_name) |*entry| entry.deinit(allocator);
        if (self.adapter_a_by_name.len > 0) allocator.free(self.adapter_a_by_name);
        for (self.adapter_b_by_name) |*entry| entry.deinit(allocator);
        if (self.adapter_b_by_name.len > 0) allocator.free(self.adapter_b_by_name);
        self.* = undefined;
    }
};

const StepParityLayerBackwardDecompProbe = struct {
    status: []const u8 = "captured",
    version: u32 = 4,
    target_layer: u32 = 0,
    segment_start: u32 = 0,
    segment_end: u32 = 0,
    runtime: []const u8 = "unknown",
    incoming_upstream: StepParityBackwardDecompStage = .{},
    full_layer_hidden_grad: StepParityBackwardDecompStage = .{},
    mlp_wo: StepParityBackwardDecompStage = .{},
    mlp_gelu_input: StepParityBackwardDecompStage = .{},
    mlp_gate_value: StepParityBackwardDecompStage = .{},
    mlp_gate_input: StepParityBackwardDecompStage = .{},
    mlp_wi_output: StepParityBackwardDecompStage = .{},
    mlp_norm_output: StepParityBackwardDecompStage = .{},
    mlp_hidden_after_attn: StepParityBackwardDecompStage = .{},
    attn_out_proj: StepParityBackwardDecompStage = .{},
    attention_core: StepParityBackwardDecompStage = .{},
    attention_core_post_rope: StepParityBackwardDecompStage = .{},
    attention_scores_raw: StepParityBackwardDecompStage = .{},
    attention_scores_masked: StepParityBackwardDecompStage = .{},
    attention_probs: StepParityBackwardDecompStage = .{},
    qkv_proj: StepParityBackwardDecompStage = .{},
    qkv_proj_split: StepParityBackwardDecompStage = .{},
    attn_norm_hidden_in: StepParityBackwardDecompStage = .{},

    fn deinit(self: *StepParityLayerBackwardDecompProbe, allocator: std.mem.Allocator) void {
        self.incoming_upstream.deinit(allocator);
        self.full_layer_hidden_grad.deinit(allocator);
        self.mlp_wo.deinit(allocator);
        self.mlp_gelu_input.deinit(allocator);
        self.mlp_gate_value.deinit(allocator);
        self.mlp_gate_input.deinit(allocator);
        self.mlp_wi_output.deinit(allocator);
        self.mlp_norm_output.deinit(allocator);
        self.mlp_hidden_after_attn.deinit(allocator);
        self.attn_out_proj.deinit(allocator);
        self.attention_core.deinit(allocator);
        self.attention_core_post_rope.deinit(allocator);
        self.attention_scores_raw.deinit(allocator);
        self.attention_scores_masked.deinit(allocator);
        self.attention_probs.deinit(allocator);
        self.qkv_proj.deinit(allocator);
        self.qkv_proj_split.deinit(allocator);
        self.attn_norm_hidden_in.deinit(allocator);
        self.* = undefined;
    }
};

const LoRAGradClipSummary = struct {
    elems: usize = 0,
    nonfinite_repaired: usize = 0,
    lora_norm_before_clip: f64 = 0,
    lora_norm_after_clip: f64 = 0,
    extra_norm_before_clip: f64 = 0,
    extra_norm_after_clip: f64 = 0,
    norm_before_clip: f64 = 0,
    norm_after_clip: f64 = 0,
    max_abs_before_clip: f32 = 0,
    max_abs_after_clip: f32 = 0,
    clip_scale: f32 = 1.0,
};

fn sanitizeAndClipLoRAGrads(layers: []fused_chunker_lora.LoRALayer, max_norm: f32, extra_norm: f64) LoRAGradClipSummary {
    var total_sq: f64 = 0;
    var out = LoRAGradClipSummary{};
    for (layers) |*ll| {
        for (ll.grad_A) |*g| {
            if (!isFiniteF32(g.*)) {
                g.* = 0;
                out.nonfinite_repaired += 1;
                continue;
            }
            out.elems += 1;
            out.max_abs_before_clip = @max(out.max_abs_before_clip, absF32(g.*));
            total_sq += @as(f64, g.*) * @as(f64, g.*);
        }
        for (ll.grad_B) |*g| {
            if (!isFiniteF32(g.*)) {
                g.* = 0;
                out.nonfinite_repaired += 1;
                continue;
            }
            out.elems += 1;
            out.max_abs_before_clip = @max(out.max_abs_before_clip, absF32(g.*));
            total_sq += @as(f64, g.*) * @as(f64, g.*);
        }
        if (ll.grad_magnitude) |gm| {
            for (gm) |*g| {
                if (!isFiniteF32(g.*)) {
                    g.* = 0;
                    out.nonfinite_repaired += 1;
                    continue;
                }
                out.elems += 1;
                out.max_abs_before_clip = @max(out.max_abs_before_clip, absF32(g.*));
                total_sq += @as(f64, g.*) * @as(f64, g.*);
            }
        }
    }

    const extra_sq = if (std.math.isFinite(extra_norm) and extra_norm > 0) extra_norm * extra_norm else 0;
    const combined_sq = total_sq + extra_sq;
    out.lora_norm_before_clip = @sqrt(total_sq);
    out.lora_norm_after_clip = out.lora_norm_before_clip;
    out.extra_norm_before_clip = @sqrt(extra_sq);
    out.extra_norm_after_clip = out.extra_norm_before_clip;
    out.norm_before_clip = @sqrt(combined_sq);
    out.norm_after_clip = out.norm_before_clip;
    out.max_abs_after_clip = out.max_abs_before_clip;
    if (max_norm <= 0) return out;

    const total_norm: f32 = @floatCast(@sqrt(combined_sq));
    if (!isFiniteF32(total_norm) or total_norm <= max_norm) return out;

    const scale = max_norm / (total_norm + 1e-6);
    out.clip_scale = scale;
    out.lora_norm_after_clip = out.lora_norm_before_clip * @as(f64, scale);
    out.extra_norm_after_clip = out.extra_norm_before_clip * @as(f64, scale);
    out.norm_after_clip = @as(f64, total_norm) * @as(f64, scale);
    out.max_abs_after_clip = out.max_abs_before_clip * scale;
    for (layers) |*ll| {
        for (ll.grad_A) |*g| g.* *= scale;
        for (ll.grad_B) |*g| g.* *= scale;
        if (ll.grad_magnitude) |gm| {
            for (gm) |*g| g.* *= scale;
        }
    }
    return out;
}

const LoRAUpdateDebugAggregate = struct {
    active_layers: usize = 0,
    active_matrices: usize = 0,
    detail_printed: usize = 0,
    param_stats: FloatSliceStats = .{},
    update_stats: FloatSliceStats = .{},
    adam_m_stats: FloatSliceStats = .{},
    adam_v_stats: FloatSliceStats = .{},
    matrix_stats: std.ArrayListUnmanaged(LoRAUpdateMatrixDebug) = .empty,

    fn deinit(self: *LoRAUpdateDebugAggregate, allocator: std.mem.Allocator) void {
        for (self.matrix_stats.items) |*entry| entry.deinit(allocator);
        self.matrix_stats.deinit(allocator);
        self.* = undefined;
    }
};

const LoRAUpdateMatrixDebug = struct {
    name: []u8,
    update_stats: FloatSliceStats = .{},
    adam_m_stats: FloatSliceStats = .{},
    adam_v_stats: FloatSliceStats = .{},

    fn deinit(self: *LoRAUpdateMatrixDebug, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        self.* = undefined;
    }
};

fn appendLoRAUpdateMatrixDebug(
    allocator: std.mem.Allocator,
    update_debug: *LoRAUpdateDebugAggregate,
    layer_idx: u32,
    module_name: []const u8,
    matrix_name: []const u8,
    before: []const f32,
    after: []const f32,
    maybe_state: ?optimizers.ParamState,
) !void {
    var entry = LoRAUpdateMatrixDebug{
        .name = try std.fmt.allocPrint(
            allocator,
            "layer_{d:0>2}_{s}_{s}",
            .{ layer_idx, module_name, matrix_name },
        ),
    };
    errdefer entry.deinit(allocator);
    try entry.update_stats.addDelta(before, after);
    if (maybe_state) |state| {
        entry.adam_m_stats.addSlice(state.m);
        entry.adam_v_stats.addSlice(state.v);
    }
    try update_debug.matrix_stats.append(allocator, entry);
}

const StepParityBatchHashes = struct {
    sample_indices: u64 = fnv_offset,
    input_ids: u64 = fnv_offset,
    attention_mask: u64 = fnv_offset,
    labels: u64 = fnv_offset,
    chunks: u64 = fnv_offset,
};

const StepParityBoundaryContext = struct {
    batch_stats: BoundaryBatchDebugStats,
    debug: fused_chunker_train.BoundaryStepDebugSummary,
    embedding_probe: EmbeddingProbe = .{},
    embedding_table_rows: []const EmbeddingRowProbe = &.{},
    embedding_lookup_rows: []const EmbeddingRowProbe = &.{},
    final_norm_weight: fused_chunker_train.BoundaryProbeTensorStats = .{},
    final_norm_bias: fused_chunker_train.BoundaryProbeTensorStats = .{},
    encoder_activation_inputs: []const EncoderActivationInputProbe = &.{},
    encoder_projection_decompositions: []const EncoderProjectionDecompositionProbe = &.{},
    encoder_attention_internals: []const EncoderAttentionInternalProbe = &.{},
    encoder_attention_rows: []const EncoderAttentionRowProbe = &.{},
    encoder_layer_inputs: []const EncoderLayerInputProbe = &.{},
    encoder_layer_states: []const EncoderLayerStateProbe = &.{},
    encoder_replay_input: ?EncoderReplayInputProbe = null,
};

const StepParityContrastiveContext = struct {
    active_chunks: usize = 0,
    first_active_index: ?usize = null,
    first_active_doc_id: ?u32 = null,
    embedding_stats: FloatSliceStats = .{},
    grad_stats: FloatSliceStats = .{},
    contrastive_loss: f64 = 0,
    total_loss: f64 = 0,
    first_active_embedding_sample: [16]f32 = [_]f32{0} ** 16,
    first_active_embedding_sample_len: usize = 0,
    first_active_grad_sample: [16]f32 = [_]f32{0} ** 16,
    first_active_grad_sample_len: usize = 0,
    active_doc_id_sample: [8]u32 = [_]u32{0} ** 8,
    active_doc_id_sample_len: usize = 0,
    active_embedding_norm_sample: [8]f32 = [_]f32{0} ** 8,
    active_embedding_norm_sample_len: usize = 0,
    active_grad_norm_sample: [8]f32 = [_]f32{0} ** 8,
    active_grad_norm_sample_len: usize = 0,
};

const EmbeddingProbe = struct {
    word_embedding_weight: fused_chunker_train.BoundaryProbeTensorStats = .{},
    token_lookup: fused_chunker_train.BoundaryProbeTensorStats = .{},
    layer_norm_output: fused_chunker_train.BoundaryProbeTensorStats = .{},
    layer_norm_weight: fused_chunker_train.BoundaryProbeTensorStats = .{},
    layer_norm_bias: fused_chunker_train.BoundaryProbeTensorStats = .{},
};

const EmbeddingRowProbe = struct {
    position: usize,
    token_id: i32,
    stats: fused_chunker_train.BoundaryProbeTensorStats = .{},
};

const EncoderLayerInputProbe = struct {
    layer_idx: u32,
    stats: fused_chunker_train.BoundaryProbeTensorStats,
};

const EncoderLayerStateProbe = struct {
    layer_idx: u32,
    name: []const u8,
    stats: fused_chunker_train.BoundaryProbeTensorStats,
};

const EncoderReplayInputProbe = struct {
    path: []const u8,
    layer_idx: u32,
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
    elems: usize,
    stats: fused_chunker_train.BoundaryProbeTensorStats,
};

const EncoderReplayUpstreamProbe = struct {
    path: []const u8,
    target_layer: u32,
    segment_start: u32,
    segment_end: u32,
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
    elems: usize,
    stats: fused_chunker_train.BoundaryProbeTensorStats,
};

const StepParityNamedProbeTensorStats = struct {
    name: []u8,
    stats: fused_chunker_train.BoundaryProbeTensorStats = .{},

    fn deinit(self: *StepParityNamedProbeTensorStats, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        self.* = undefined;
    }
};

const StepParityUpstreamGradProbe = struct {
    status: []const u8 = "captured",
    target_layer: u32 = 0,
    boundary_features_grad: fused_chunker_train.BoundaryProbeTensorStats = .{},
    contrastive_features_grad: fused_chunker_train.BoundaryProbeTensorStats = .{},
    combined_features_grad: fused_chunker_train.BoundaryProbeTensorStats = .{},
    final_norm_input: fused_chunker_train.BoundaryProbeTensorStats = .{},
    final_norm_weight: fused_chunker_train.BoundaryProbeTensorStats = .{},
    lora_output_grad: fused_chunker_train.BoundaryProbeTensorStats = .{},
    target_segment_upstream: fused_chunker_train.BoundaryProbeTensorStats = .{},
    upper_encoder_ladder: []StepParityNamedProbeTensorStats = &.{},

    fn deinit(self: *StepParityUpstreamGradProbe, allocator: std.mem.Allocator) void {
        for (self.upper_encoder_ladder) |*entry| entry.deinit(allocator);
        if (self.upper_encoder_ladder.len > 0) allocator.free(self.upper_encoder_ladder);
        self.* = undefined;
    }
};

const EncoderActivationInputProbe = struct {
    layer_idx: u32,
    module_name: []const u8,
    stats: fused_chunker_train.BoundaryProbeTensorStats,
};

const EncoderProjectionDecompositionProbe = struct {
    layer_idx: u32,
    name: []const u8,
    input: fused_chunker_train.BoundaryProbeTensorStats = .{},
    base: fused_chunker_train.BoundaryProbeTensorStats = .{},
    lora_a: fused_chunker_train.BoundaryProbeTensorStats = .{},
    lora_b: fused_chunker_train.BoundaryProbeTensorStats = .{},
    delta: fused_chunker_train.BoundaryProbeTensorStats = .{},
    output: fused_chunker_train.BoundaryProbeTensorStats = .{},
    weight: fused_chunker_train.BoundaryProbeTensorStats = .{},
    bias: fused_chunker_train.BoundaryProbeTensorStats = .{},
    lora_a_weight: fused_chunker_train.BoundaryProbeTensorStats = .{},
    lora_b_weight: fused_chunker_train.BoundaryProbeTensorStats = .{},
    base_reference_error: fused_chunker_train.BoundaryProbeTensorStats = .{},
    lora_a_reference_error: fused_chunker_train.BoundaryProbeTensorStats = .{},
    lora_b_reference_error: fused_chunker_train.BoundaryProbeTensorStats = .{},
    delta_reference_error: fused_chunker_train.BoundaryProbeTensorStats = .{},
    output_reference_error: fused_chunker_train.BoundaryProbeTensorStats = .{},
    scale: f32 = 0,
    rank: usize = 0,
    rows: usize = 0,
    in_dim: usize = 0,
    out_dim: usize = 0,
    has_bias: bool = false,
};

const EncoderAttentionInternalProbe = struct {
    layer_idx: u32,
    name: []const u8,
    stats: fused_chunker_train.BoundaryProbeTensorStats,
};

const EncoderAttentionRowProbe = struct {
    layer_idx: u32,
    name: []const u8,
    batch_idx: usize,
    head_idx: usize,
    query_idx: usize,
    valid_keys: usize,
    score_mean: f64,
    score_rms: f64,
    score_min: f32,
    score_max: f32,
    score_argmax: usize,
    prob_entropy: f64,
    prob_max: f32,
    prob_argmax: usize,
    prob_top2_gap: f32,
    query_rms: f64,
    query_max_abs: f32,
    key_query_rms: f64,
    key_query_max_abs: f32,
    value_query_rms: f64,
    value_query_max_abs: f32,
    output_mean: f64,
    output_rms: f64,
    output_max_abs: f32,
    query_sample_len: usize,
    query_sample: [attention_row_sample_max]f32,
    key_query_sample_len: usize,
    key_query_sample: [attention_row_sample_max]f32,
    value_query_sample_len: usize,
    value_query_sample: [attention_row_sample_max]f32,
    score_sample_len: usize,
    score_sample: [attention_row_sample_max]f32,
    prob_sample_len: usize,
    prob_sample: [attention_row_sample_max]f32,
    output_sample_len: usize,
    output_sample: [attention_row_sample_max]f32,
};

const Layer0SdpaReferenceProbeStats = struct {
    token_ref: fused_chunker_train.BoundaryProbeTensorStats,
    token_delta: fused_chunker_train.BoundaryProbeTensorStats,
    kernel_ref: fused_chunker_train.BoundaryProbeTensorStats,
    kernel_delta: fused_chunker_train.BoundaryProbeTensorStats,
};

const embedding_row_probe_max = 8;
const attention_row_sample_max = 16;

const StepParityUpdateSummary = struct {
    grad_clip: LoRAGradClipSummary,
    active_adapters: usize,
    active_matrices: usize,
    base_lr: f32,
    lora_plus_ratio: f32,
    optimizer_step_count: u64,
    param_elems: usize,
    param_norm: f64,
    update_elems: usize,
    update_norm: f64,
    update_max_abs: f32,
    update_mean_abs: f64,
    update_to_param: f64,
    adam_m_norm: f64,
    adam_v_norm: f64,
    repaired_after_update: usize,
    matrix_stats: []const LoRAUpdateMatrixDebug = &.{},
};

fn hashU64(hash: *u64, value: u64) void {
    var v = value;
    for (0..8) |_| {
        hash.* ^= v & 0xff;
        hash.* *%= fnv_prime;
        v >>= 8;
    }
}

fn hashI32(hash: *u64, value: i32) void {
    hashU64(hash, @as(u32, @bitCast(value)));
}

fn computeStepParityBatchHashes(
    batch: *const fused_chunker_data.FusedBatch,
    total_tokens: usize,
) StepParityBatchHashes {
    var out = StepParityBatchHashes{};
    for (batch.sample_indices) |idx| hashU64(&out.sample_indices, idx);
    for (0..total_tokens) |i| {
        hashI32(&out.input_ids, batch.input_ids[i]);
        hashI32(&out.attention_mask, batch.attention_mask[i]);
        hashU64(&out.labels, if (batch.boundary_labels[i] > 0.5) 1 else 0);
    }
    for (0..batch.batch_size * batch.max_chunks) |i| {
        hashI32(&out.chunks, batch.chunk_starts[i]);
        hashI32(&out.chunks, batch.chunk_ends[i]);
        hashU64(&out.chunks, if (batch.chunk_mask[i] > 0.5) 1 else 0);
    }
    return out;
}

fn writeUsizeJsonArray(writer: anytype, values: []const usize) !void {
    try writer.interface.writeByte('[');
    for (values, 0..) |value, i| {
        if (i > 0) try writer.interface.writeByte(',');
        try writer.interface.print("{d}", .{value});
    }
    try writer.interface.writeByte(']');
}

fn writeProbeF32SliceJson(writer: anytype, values: []const f32) !void {
    try writer.interface.writeByte('[');
    for (values, 0..) |value, i| {
        if (i > 0) try writer.interface.writeByte(',');
        try writer.interface.print("{d:.9}", .{value});
    }
    try writer.interface.writeByte(']');
}

fn writeProbeFloatSampleJson(writer: anytype, stats: fused_chunker_train.BoundaryProbeTensorStats) !void {
    try writeProbeF32SliceJson(writer, stats.sample[0..stats.sample_len]);
}

fn writeProbeU64SampleJson(writer: anytype, values: []const u64) !void {
    try writer.interface.writeByte('[');
    for (values, 0..) |value, i| {
        if (i > 0) try writer.interface.writeByte(',');
        try writer.interface.print("{d}", .{value});
    }
    try writer.interface.writeByte(']');
}

fn writeProbeTensorStatsJson(writer: anytype, stats: fused_chunker_train.BoundaryProbeTensorStats) !void {
    try writer.interface.print(
        "{{\"elems\":{d},\"mean\":{d:.9},\"rms\":{d:.9},\"max_abs\":{d:.9},\"max_abs_index\":{d},\"max_abs_value\":{d:.9},\"hash\":\"{x}\",\"sample\":",
        .{ stats.elems, stats.mean, stats.rms, stats.max_abs, stats.max_abs_index, stats.max_abs_value, stats.hash },
    );
    try writeProbeFloatSampleJson(writer, stats);
    try writer.interface.writeAll(",\"top_abs_indices\":");
    try writeProbeU64SampleJson(writer, stats.top_abs_indices[0..stats.top_abs_len]);
    try writer.interface.writeAll(",\"top_abs_values\":");
    try writeProbeF32SliceJson(writer, stats.top_abs_values[0..stats.top_abs_len]);
    try writer.interface.writeByte('}');
}

fn writeBoundaryForwardProbeJson(writer: anytype, probe: fused_chunker_train.BoundaryForwardProbe) !void {
    try writer.interface.writeAll("{\"final_norm_input\":");
    try writeProbeTensorStatsJson(writer, probe.final_norm_input);
    try writer.interface.writeAll(",\"boundary_head_input\":");
    try writeProbeTensorStatsJson(writer, probe.boundary_head_input);
    try writer.interface.writeAll(",\"dense1_pre_activation\":");
    try writeProbeTensorStatsJson(writer, probe.dense1_pre_activation);
    try writer.interface.writeAll(",\"dense1_post_activation\":");
    try writeProbeTensorStatsJson(writer, probe.dense1_post_activation);
    try writer.interface.writeAll(",\"logits\":");
    try writeProbeTensorStatsJson(writer, probe.logits);
    try writer.interface.writeByte('}');
}

fn writeBoundaryCheckpointProbeJson(
    writer: anytype,
    probe: fused_chunker_train.BoundaryCheckpointProbe,
    final_norm_weight: fused_chunker_train.BoundaryProbeTensorStats,
    final_norm_bias: fused_chunker_train.BoundaryProbeTensorStats,
) !void {
    try writer.interface.writeAll("{\"final_norm_weight\":");
    try writeProbeTensorStatsJson(writer, final_norm_weight);
    try writer.interface.writeAll(",\"final_norm_bias\":");
    try writeProbeTensorStatsJson(writer, final_norm_bias);
    try writer.interface.writeAll(",\"w1\":");
    try writeProbeTensorStatsJson(writer, probe.w1);
    try writer.interface.writeAll(",\"b1\":");
    try writeProbeTensorStatsJson(writer, probe.b1);
    try writer.interface.writeAll(",\"w2\":");
    try writeProbeTensorStatsJson(writer, probe.w2);
    try writer.interface.writeAll(",\"b2\":");
    try writeProbeTensorStatsJson(writer, probe.b2);
    try writer.interface.writeByte('}');
}

fn writeEmbeddingProbeJson(writer: anytype, probe: EmbeddingProbe) !void {
    try writer.interface.writeAll("{\"word_embedding_weight\":");
    try writeProbeTensorStatsJson(writer, probe.word_embedding_weight);
    try writer.interface.writeAll(",\"token_lookup\":");
    try writeProbeTensorStatsJson(writer, probe.token_lookup);
    try writer.interface.writeAll(",\"layer_norm_output\":");
    try writeProbeTensorStatsJson(writer, probe.layer_norm_output);
    try writer.interface.writeAll(",\"layer_norm_weight\":");
    try writeProbeTensorStatsJson(writer, probe.layer_norm_weight);
    try writer.interface.writeAll(",\"layer_norm_bias\":");
    try writeProbeTensorStatsJson(writer, probe.layer_norm_bias);
    try writer.interface.writeByte('}');
}

fn writeEmbeddingRowProbeJson(writer: anytype, probes: []const EmbeddingRowProbe) !void {
    try writer.interface.writeByte('{');
    for (probes, 0..) |probe, i| {
        if (i > 0) try writer.interface.writeByte(',');
        try writer.interface.print("\"pos_{d:0>3}_id_{d}\":", .{ probe.position, probe.token_id });
        try writeProbeTensorStatsJson(writer, probe.stats);
    }
    try writer.interface.writeByte('}');
}

fn writeEncoderLayerInputProbeJson(writer: anytype, probes: []const EncoderLayerInputProbe) !void {
    try writer.interface.writeByte('{');
    for (probes, 0..) |probe, i| {
        if (i > 0) try writer.interface.writeByte(',');
        try writer.interface.print("\"layer_{d:0>2}\":", .{probe.layer_idx});
        try writeProbeTensorStatsJson(writer, probe.stats);
    }
    try writer.interface.writeByte('}');
}

fn writeEncoderLayerStateProbeJson(writer: anytype, probes: []const EncoderLayerStateProbe) !void {
    try writer.interface.writeByte('{');
    for (probes, 0..) |probe, i| {
        if (i > 0) try writer.interface.writeByte(',');
        try writer.interface.print("\"layer_{d:0>2}_{s}\":", .{ probe.layer_idx, probe.name });
        try writeProbeTensorStatsJson(writer, probe.stats);
    }
    try writer.interface.writeByte('}');
}

fn writeEncoderReplayInputProbeJson(writer: anytype, maybe_probe: ?EncoderReplayInputProbe) !void {
    const probe = maybe_probe orelse {
        try writer.interface.writeAll("null");
        return;
    };
    try writer.interface.print(
        "{{\"status\":\"captured\",\"path\":\"{s}\",\"layer_idx\":{d},\"batch_size\":{d},\"seq_len\":{d},\"hidden_size\":{d},\"elems\":{d},\"dtype\":\"f32\",\"endianness\":\"little\",\"stats\":",
        .{
            probe.path,
            probe.layer_idx,
            probe.batch_size,
            probe.seq_len,
            probe.hidden_size,
            probe.elems,
        },
    );
    try writeProbeTensorStatsJson(writer, probe.stats);
    try writer.interface.writeByte('}');
}

fn writeEncoderReplayUpstreamProbeJson(writer: anytype, maybe_probe: ?EncoderReplayUpstreamProbe) !void {
    const probe = maybe_probe orelse {
        try writer.interface.writeAll("null");
        return;
    };
    try writer.interface.print(
        "{{\"status\":\"captured\",\"path\":\"{s}\",\"target_layer\":{d},\"segment_start\":{d},\"segment_end\":{d},\"batch_size\":{d},\"seq_len\":{d},\"hidden_size\":{d},\"elems\":{d},\"dtype\":\"f32\",\"endianness\":\"little\",\"stats\":",
        .{
            probe.path,
            probe.target_layer,
            probe.segment_start,
            probe.segment_end,
            probe.batch_size,
            probe.seq_len,
            probe.hidden_size,
            probe.elems,
        },
    );
    try writeProbeTensorStatsJson(writer, probe.stats);
    try writer.interface.writeByte('}');
}

fn writeStepParityNamedProbeTensorStatsMapJson(writer: anytype, entries: []const StepParityNamedProbeTensorStats) !void {
    try writer.interface.writeByte('{');
    for (entries, 0..) |entry, i| {
        if (i > 0) try writer.interface.writeByte(',');
        try writer.interface.print("\"{s}\":", .{entry.name});
        try writeProbeTensorStatsJson(writer, entry.stats);
    }
    try writer.interface.writeByte('}');
}

fn writeStepParityUpstreamGradProbeJson(writer: anytype, maybe_probe: ?StepParityUpstreamGradProbe) !void {
    const probe = maybe_probe orelse {
        try writer.interface.writeAll("null");
        return;
    };
    try writer.interface.print(
        "{{\"status\":\"{s}\",\"target_layer\":{d},\"stages\":{{\"boundary_features_grad\":",
        .{ probe.status, probe.target_layer },
    );
    try writeProbeTensorStatsJson(writer, probe.boundary_features_grad);
    try writer.interface.writeAll(",\"contrastive_features_grad\":");
    try writeProbeTensorStatsJson(writer, probe.contrastive_features_grad);
    try writer.interface.writeAll(",\"combined_features_grad\":");
    try writeProbeTensorStatsJson(writer, probe.combined_features_grad);
    try writer.interface.writeAll(",\"final_norm_input\":");
    try writeProbeTensorStatsJson(writer, probe.final_norm_input);
    try writer.interface.writeAll(",\"final_norm_weight\":");
    try writeProbeTensorStatsJson(writer, probe.final_norm_weight);
    try writer.interface.writeAll(",\"lora_output_grad\":");
    try writeProbeTensorStatsJson(writer, probe.lora_output_grad);
    try writer.interface.writeAll(",\"target_segment_upstream\":");
    try writeProbeTensorStatsJson(writer, probe.target_segment_upstream);
    try writer.interface.writeAll("},\"upper_encoder_ladder\":");
    try writeStepParityNamedProbeTensorStatsMapJson(writer, probe.upper_encoder_ladder);
    try writer.interface.writeByte('}');
}

fn writeEncoderActivationInputProbeJson(writer: anytype, probes: []const EncoderActivationInputProbe) !void {
    try writer.interface.writeByte('{');
    for (probes, 0..) |probe, i| {
        if (i > 0) try writer.interface.writeByte(',');
        try writer.interface.print("\"layer_{d:0>2}_{s}\":", .{ probe.layer_idx, probe.module_name });
        try writeProbeTensorStatsJson(writer, probe.stats);
    }
    try writer.interface.writeByte('}');
}

fn fusedProbeStatsFromModernBert(stats: modern_bert.TensorProbeStats) fused_chunker_train.BoundaryProbeTensorStats {
    return .{
        .elems = stats.elems,
        .mean = stats.mean,
        .rms = stats.rms,
        .max_abs = stats.max_abs,
        .max_abs_index = stats.max_abs_index,
        .max_abs_value = stats.max_abs_value,
        .hash = stats.hash,
        .sample_len = stats.sample_len,
        .sample = stats.sample,
        .top_abs_len = stats.top_abs_len,
        .top_abs_indices = stats.top_abs_indices,
        .top_abs_values = stats.top_abs_values,
    };
}

fn writeEncoderProjectionDecompositionProbeJson(writer: anytype, probes: []const EncoderProjectionDecompositionProbe) !void {
    try writer.interface.writeByte('{');
    for (probes, 0..) |probe, i| {
        if (i > 0) try writer.interface.writeByte(',');
        try writer.interface.print(
            "\"layer_{d:0>2}_{s}\":{{\"scale\":{d:.9},\"rank\":{d},\"rows\":{d},\"in_dim\":{d},\"out_dim\":{d},\"has_bias\":{},\"input\":",
            .{
                probe.layer_idx,
                probe.name,
                probe.scale,
                probe.rank,
                probe.rows,
                probe.in_dim,
                probe.out_dim,
                probe.has_bias,
            },
        );
        try writeProbeTensorStatsJson(writer, probe.input);
        try writer.interface.writeAll(",\"base\":");
        try writeProbeTensorStatsJson(writer, probe.base);
        try writer.interface.writeAll(",\"lora_a\":");
        try writeProbeTensorStatsJson(writer, probe.lora_a);
        try writer.interface.writeAll(",\"lora_b\":");
        try writeProbeTensorStatsJson(writer, probe.lora_b);
        try writer.interface.writeAll(",\"delta\":");
        try writeProbeTensorStatsJson(writer, probe.delta);
        try writer.interface.writeAll(",\"output\":");
        try writeProbeTensorStatsJson(writer, probe.output);
        try writer.interface.writeAll(",\"weight\":");
        try writeProbeTensorStatsJson(writer, probe.weight);
        try writer.interface.writeAll(",\"bias\":");
        try writeProbeTensorStatsJson(writer, probe.bias);
        try writer.interface.writeAll(",\"lora_a_weight\":");
        try writeProbeTensorStatsJson(writer, probe.lora_a_weight);
        try writer.interface.writeAll(",\"lora_b_weight\":");
        try writeProbeTensorStatsJson(writer, probe.lora_b_weight);
        try writer.interface.writeAll(",\"base_reference_error\":");
        try writeProbeTensorStatsJson(writer, probe.base_reference_error);
        try writer.interface.writeAll(",\"lora_a_reference_error\":");
        try writeProbeTensorStatsJson(writer, probe.lora_a_reference_error);
        try writer.interface.writeAll(",\"lora_b_reference_error\":");
        try writeProbeTensorStatsJson(writer, probe.lora_b_reference_error);
        try writer.interface.writeAll(",\"delta_reference_error\":");
        try writeProbeTensorStatsJson(writer, probe.delta_reference_error);
        try writer.interface.writeAll(",\"output_reference_error\":");
        try writeProbeTensorStatsJson(writer, probe.output_reference_error);
        try writer.interface.writeByte('}');
    }
    try writer.interface.writeByte('}');
}

fn writeEncoderAttentionInternalProbeJson(writer: anytype, probes: []const EncoderAttentionInternalProbe) !void {
    try writer.interface.writeByte('{');
    for (probes, 0..) |probe, i| {
        if (i > 0) try writer.interface.writeByte(',');
        try writer.interface.print("\"layer_{d:0>2}_{s}\":", .{ probe.layer_idx, probe.name });
        try writeProbeTensorStatsJson(writer, probe.stats);
    }
    try writer.interface.writeByte('}');
}

fn writeF32LittleEndianFile(allocator: std.mem.Allocator, path: []const u8, values: []const f32) !void {
    var file = try compat.cwd().createFile(compat.io(), path, .{ .truncate = true });
    defer file.close(compat.io());
    const bytes = try allocator.alloc(u8, values.len * @sizeOf(f32));
    defer allocator.free(bytes);
    for (values, 0..) |value, i| {
        std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(value), .little);
    }
    try file.writeStreamingAll(compat.io(), bytes);
}

fn writeFixedF32SampleJson(writer: anytype, sample: []const f32) !void {
    try writer.interface.writeByte('[');
    for (sample, 0..) |value, i| {
        if (i > 0) try writer.interface.writeByte(',');
        try writer.interface.print("{d:.9}", .{value});
    }
    try writer.interface.writeByte(']');
}

fn writeFixedU32SampleJson(writer: anytype, sample: []const u32) !void {
    try writer.interface.writeByte('[');
    for (sample, 0..) |value, i| {
        if (i > 0) try writer.interface.writeByte(',');
        try writer.interface.print("{d}", .{value});
    }
    try writer.interface.writeByte(']');
}

fn writeEncoderAttentionRowProbeJson(writer: anytype, probes: []const EncoderAttentionRowProbe) !void {
    try writer.interface.writeByte('{');
    for (probes, 0..) |probe, i| {
        if (i > 0) try writer.interface.writeByte(',');
        try writer.interface.print(
            "\"layer_{d:0>2}_{s}\":{{\"batch\":{d},\"head\":{d},\"query\":{d},\"valid_keys\":{d},\"score_mean\":{d:.9},\"score_rms\":{d:.9},\"score_min\":{d:.9},\"score_max\":{d:.9},\"score_argmax\":{d},\"prob_entropy\":{d:.9},\"prob_max\":{d:.9},\"prob_argmax\":{d},\"prob_top2_gap\":{d:.9},\"query_rms\":{d:.9},\"query_max_abs\":{d:.9},\"key_query_rms\":{d:.9},\"key_query_max_abs\":{d:.9},\"value_query_rms\":{d:.9},\"value_query_max_abs\":{d:.9},\"output_mean\":{d:.9},\"output_rms\":{d:.9},\"output_max_abs\":{d:.9},\"query_sample\":",
            .{
                probe.layer_idx,
                probe.name,
                probe.batch_idx,
                probe.head_idx,
                probe.query_idx,
                probe.valid_keys,
                probe.score_mean,
                probe.score_rms,
                probe.score_min,
                probe.score_max,
                probe.score_argmax,
                probe.prob_entropy,
                probe.prob_max,
                probe.prob_argmax,
                probe.prob_top2_gap,
                probe.query_rms,
                probe.query_max_abs,
                probe.key_query_rms,
                probe.key_query_max_abs,
                probe.value_query_rms,
                probe.value_query_max_abs,
                probe.output_mean,
                probe.output_rms,
                probe.output_max_abs,
            },
        );
        try writeFixedF32SampleJson(writer, probe.query_sample[0..probe.query_sample_len]);
        try writer.interface.writeAll(",\"key_query_sample\":");
        try writeFixedF32SampleJson(writer, probe.key_query_sample[0..probe.key_query_sample_len]);
        try writer.interface.writeAll(",\"value_query_sample\":");
        try writeFixedF32SampleJson(writer, probe.value_query_sample[0..probe.value_query_sample_len]);
        try writer.interface.writeAll(",\"score_sample\":");
        try writeFixedF32SampleJson(writer, probe.score_sample[0..probe.score_sample_len]);
        try writer.interface.writeAll(",\"prob_sample\":");
        try writeFixedF32SampleJson(writer, probe.prob_sample[0..probe.prob_sample_len]);
        try writer.interface.writeAll(",\"output_sample\":");
        try writeFixedF32SampleJson(writer, probe.output_sample[0..probe.output_sample_len]);
        try writer.interface.writeByte('}');
    }
    try writer.interface.writeByte('}');
}

fn writeFloatSliceStatsJson(writer: anytype, stats: FloatSliceStats) !void {
    try writer.interface.print(
        "{{\"elems\":{d},\"nonzero\":{d},\"l2\":{d:.9},\"max_abs\":{d:.9},\"mean_abs\":{d:.9}}}",
        .{
            stats.elems,
            stats.nonzero,
            stats.l2(),
            stats.max_abs,
            stats.meanAbs(),
        },
    );
}

fn writeLoRAUpdateMatrixStatsJson(writer: anytype, entries: []const LoRAUpdateMatrixDebug) !void {
    try writer.interface.writeByte('{');
    for (entries, 0..) |entry, i| {
        if (i > 0) try writer.interface.writeByte(',');
        try writer.interface.print("\"{s}\":{{\"update\":", .{entry.name});
        try writeFloatSliceStatsJson(writer, entry.update_stats);
        try writer.interface.writeAll(",\"adam_m\":");
        try writeFloatSliceStatsJson(writer, entry.adam_m_stats);
        try writer.interface.writeAll(",\"adam_v\":");
        try writeFloatSliceStatsJson(writer, entry.adam_v_stats);
        try writer.interface.writeByte('}');
    }
    try writer.interface.writeByte('}');
}

fn writeFloatSliceStatsFromSliceJson(writer: anytype, values: []const f32, scale: f32) !void {
    var stats = FloatSliceStats{};
    const sample_len = @min(values.len, 16);
    for (values) |value| stats.addValue(value * scale);
    try writer.interface.print(
        "{{\"elems\":{d},\"nonzero\":{d},\"l2\":{d:.9},\"max_abs\":{d:.9},\"mean_abs\":{d:.9},\"sample\":",
        .{
            stats.elems,
            stats.nonzero,
            stats.l2(),
            stats.max_abs,
            stats.meanAbs(),
        },
    );
    try writer.interface.writeByte('[');
    for (values[0..sample_len], 0..) |value, i| {
        if (i > 0) try writer.interface.writeByte(',');
        try writer.interface.print("{d:.9}", .{value * scale});
    }
    try writer.interface.writeAll("]}");
}

fn writeStepParityTensorSliceStatsJson(writer: anytype, value: StepParityTensorSliceStats) !void {
    try writer.interface.print(
        "{{\"elems\":{d},\"nonzero\":{d},\"l2\":{d:.9},\"max_abs\":{d:.9},\"mean_abs\":{d:.9},\"sample\":",
        .{
            value.stats.elems,
            value.stats.nonzero,
            value.stats.l2(),
            value.stats.max_abs,
            value.stats.meanAbs(),
        },
    );
    try writeFixedF32SampleJson(writer, value.sample[0..value.sample_len]);
    try writer.interface.writeByte('}');
}

fn writeStepParityNamedTensorStatsMapJson(writer: anytype, entries: []const StepParityNamedTensorStats) !void {
    try writer.interface.writeByte('{');
    for (entries, 0..) |entry, i| {
        if (i > 0) try writer.interface.writeByte(',');
        try writer.interface.print("\"{s}\":", .{entry.name});
        try writeStepParityTensorSliceStatsJson(writer, entry.stats);
    }
    try writer.interface.writeByte('}');
}

fn writeStepParitySegmentVJPProbeJson(writer: anytype, maybe_probe: ?StepParitySegmentVJPProbe) !void {
    const probe = maybe_probe orelse {
        try writer.interface.writeAll("null");
        return;
    };
    try writer.interface.print(
        "{{\"target_layer\":{d},\"segment_start\":{d},\"segment_end\":{d},\"include_hidden_grad\":{},\"include_adapter_grads\":{},\"runtime\":\"{s}\",\"profile\":{{\"runtime_input_ms\":{d:.9},\"compiled_execute_ms\":{d:.9},\"compiled_extract_ms\":{d:.9},\"compiled_total_ms\":{d:.9},\"accumulate_ms\":{d:.9},\"total_ms\":{d:.9},\"mpsgraph_runtime\":{},\"partitioned_runtime\":{},\"partitions_executed\":{d},\"backend_command_dispatches\":{d},\"planned_operator_dispatches\":{d},\"graph_region_dispatches\":{d},\"interpreter_fallbacks\":{d}}},\"upstream\":",
        .{
            probe.target_layer,
            probe.segment_start,
            probe.segment_end,
            probe.include_hidden_grad,
            probe.include_adapter_grads,
            probe.runtime,
            nsToMs(probe.profile.runtime_input_ns),
            nsToMs(probe.profile.compiled_execute_ns),
            nsToMs(probe.profile.compiled_extract_ns),
            nsToMs(probe.profile.compiled_total_ns),
            nsToMs(probe.profile.accumulate_ns),
            nsToMs(probe.profile.total_ns),
            probe.profile.mpsgraph_runtime,
            probe.profile.partitioned_runtime,
            probe.profile.partitions_executed,
            probe.profile.backend_command_dispatches,
            probe.profile.planned_operator_dispatches,
            probe.profile.graph_region_dispatches,
            probe.profile.interpreter_fallbacks,
        },
    );
    try writeStepParityTensorSliceStatsJson(writer, probe.upstream);
    try writer.interface.writeAll(",\"hidden_grad\":");
    try writeStepParityTensorSliceStatsJson(writer, probe.hidden_grad);
    try writer.interface.writeAll(",\"adapter_a\":");
    try writeStepParityTensorSliceStatsJson(writer, probe.adapter_a);
    try writer.interface.writeAll(",\"adapter_b\":");
    try writeStepParityTensorSliceStatsJson(writer, probe.adapter_b);
    try writer.interface.writeAll(",\"adapter_a_by_name\":");
    try writeStepParityNamedTensorStatsMapJson(writer, probe.adapter_a_by_name);
    try writer.interface.writeAll(",\"adapter_b_by_name\":");
    try writeStepParityNamedTensorStatsMapJson(writer, probe.adapter_b_by_name);
    try writer.interface.writeByte('}');
}

fn writeStepParityBackwardDecompComponentsJson(writer: anytype, entries: []const StepParityBackwardDecompComponent) !void {
    try writer.interface.writeByte('{');
    for (entries, 0..) |entry, i| {
        if (i > 0) try writer.interface.writeByte(',');
        try writer.interface.print("\"{s}\":", .{entry.name});
        try writeStepParityTensorSliceStatsJson(writer, entry.stats);
    }
    try writer.interface.writeByte('}');
}

fn writeStepParityBackwardDecompStageJson(writer: anytype, stage: StepParityBackwardDecompStage) !void {
    try writer.interface.print("{{\"status\":\"{s}\",\"reason\":\"{s}\"", .{ stage.status, stage.reason });
    if (std.mem.eql(u8, stage.status, "captured")) {
        try writer.interface.writeAll(",\"stats\":");
        try writeStepParityTensorSliceStatsJson(writer, stage.stats);
        try writer.interface.writeAll(",\"components\":");
        try writeStepParityBackwardDecompComponentsJson(writer, stage.components);
        try writer.interface.writeAll(",\"adapter_a\":");
        try writeStepParityTensorSliceStatsJson(writer, stage.adapter_a);
        try writer.interface.writeAll(",\"adapter_b\":");
        try writeStepParityTensorSliceStatsJson(writer, stage.adapter_b);
        try writer.interface.writeAll(",\"adapter_a_by_name\":");
        try writeStepParityNamedTensorStatsMapJson(writer, stage.adapter_a_by_name);
        try writer.interface.writeAll(",\"adapter_b_by_name\":");
        try writeStepParityNamedTensorStatsMapJson(writer, stage.adapter_b_by_name);
    }
    try writer.interface.writeByte('}');
}

fn writeStepParityLayerBackwardDecompProbeJson(writer: anytype, maybe_probe: ?StepParityLayerBackwardDecompProbe) !void {
    const probe = maybe_probe orelse {
        try writer.interface.writeAll("null");
        return;
    };
    try writer.interface.print(
        "{{\"status\":\"{s}\",\"version\":{d},\"target_layer\":{d},\"segment_start\":{d},\"segment_end\":{d},\"runtime\":\"{s}\",\"stages\":{{\"incoming_upstream\":",
        .{
            probe.status,
            probe.version,
            probe.target_layer,
            probe.segment_start,
            probe.segment_end,
            probe.runtime,
        },
    );
    try writeStepParityBackwardDecompStageJson(writer, probe.incoming_upstream);
    try writer.interface.writeAll(",\"full_layer_hidden_grad\":");
    try writeStepParityBackwardDecompStageJson(writer, probe.full_layer_hidden_grad);
    try writer.interface.writeAll(",\"mlp_wo\":");
    try writeStepParityBackwardDecompStageJson(writer, probe.mlp_wo);
    try writer.interface.writeAll(",\"mlp_gelu_input\":");
    try writeStepParityBackwardDecompStageJson(writer, probe.mlp_gelu_input);
    try writer.interface.writeAll(",\"mlp_gate_value\":");
    try writeStepParityBackwardDecompStageJson(writer, probe.mlp_gate_value);
    try writer.interface.writeAll(",\"mlp_gate_input\":");
    try writeStepParityBackwardDecompStageJson(writer, probe.mlp_gate_input);
    try writer.interface.writeAll(",\"mlp_wi_output\":");
    try writeStepParityBackwardDecompStageJson(writer, probe.mlp_wi_output);
    try writer.interface.writeAll(",\"mlp_norm_output\":");
    try writeStepParityBackwardDecompStageJson(writer, probe.mlp_norm_output);
    try writer.interface.writeAll(",\"mlp_hidden_after_attn\":");
    try writeStepParityBackwardDecompStageJson(writer, probe.mlp_hidden_after_attn);
    try writer.interface.writeAll(",\"attn_out_proj\":");
    try writeStepParityBackwardDecompStageJson(writer, probe.attn_out_proj);
    try writer.interface.writeAll(",\"attention_core\":");
    try writeStepParityBackwardDecompStageJson(writer, probe.attention_core);
    try writer.interface.writeAll(",\"attention_core_post_rope\":");
    try writeStepParityBackwardDecompStageJson(writer, probe.attention_core_post_rope);
    try writer.interface.writeAll(",\"attention_scores_raw\":");
    try writeStepParityBackwardDecompStageJson(writer, probe.attention_scores_raw);
    try writer.interface.writeAll(",\"attention_scores_masked\":");
    try writeStepParityBackwardDecompStageJson(writer, probe.attention_scores_masked);
    try writer.interface.writeAll(",\"attention_probs\":");
    try writeStepParityBackwardDecompStageJson(writer, probe.attention_probs);
    try writer.interface.writeAll(",\"qkv_proj\":");
    try writeStepParityBackwardDecompStageJson(writer, probe.qkv_proj);
    try writer.interface.writeAll(",\"qkv_proj_split\":");
    try writeStepParityBackwardDecompStageJson(writer, probe.qkv_proj_split);
    try writer.interface.writeAll(",\"attn_norm_hidden_in\":");
    try writeStepParityBackwardDecompStageJson(writer, probe.attn_norm_hidden_in);
    try writer.interface.writeAll("}}");
}

fn writeStepParitySoftmaxVJPCaseJson(writer: anytype, case: StepParitySoftmaxVJPCase) !void {
    try writer.interface.print(
        "{{\"status\":\"{s}\",\"reason\":\"{s}\",\"outer\":{d},\"queries\":{d},\"keys\":{d},\"has_mask\":{},\"mask_bias\":{d}",
        .{
            case.status,
            case.reason,
            case.outer,
            case.queries,
            case.keys,
            case.has_mask,
            case.mask_bias,
        },
    );
    try writer.interface.writeAll(",\"scores_masked\":");
    try writeStepParityTensorSliceStatsJson(writer, case.scores_masked);
    try writer.interface.writeAll(",\"probs\":");
    try writeStepParityTensorSliceStatsJson(writer, case.probs);
    try writer.interface.writeAll(",\"upstream_probs_grad\":");
    try writeStepParityTensorSliceStatsJson(writer, case.upstream_probs_grad);
    try writer.interface.writeAll(",\"scores_masked_grad\":");
    try writeStepParityTensorSliceStatsJson(writer, case.scores_masked_grad);
    try writer.interface.writeAll(",\"cpu_scores_masked_grad\":");
    try writeStepParityTensorSliceStatsJson(writer, case.cpu_scores_masked_grad);
    try writer.interface.writeAll(",\"cpu_abs_error\":");
    try writeStepParityTensorSliceStatsJson(writer, case.cpu_abs_error);
    try writer.interface.writeAll(",\"valid_scores_masked_grad\":");
    try writeStepParityTensorSliceStatsJson(writer, case.valid_scores_masked_grad);
    try writer.interface.writeAll(",\"masked_scores_masked_grad\":");
    try writeStepParityTensorSliceStatsJson(writer, case.masked_scores_masked_grad);
    try writer.interface.writeByte('}');
}

fn writeStepParitySoftmaxVJPProbeJson(writer: anytype, maybe_probe: ?StepParitySoftmaxVJPProbe) !void {
    const probe = maybe_probe orelse {
        try writer.interface.writeAll("null");
        return;
    };
    try writer.interface.print(
        "{{\"status\":\"{s}\",\"version\":{d},\"runtime\":\"{s}\",\"cases\":{{",
        .{ probe.status, probe.version, probe.runtime },
    );
    for (probe.cases, 0..) |case, i| {
        if (i > 0) try writer.interface.writeByte(',');
        try writer.interface.print("\"{s}\":", .{case.name});
        try writeStepParitySoftmaxVJPCaseJson(writer, case);
    }
    try writer.interface.writeAll("}}");
}

fn writeStepParityQKVSplitVJPCaseJson(writer: anytype, case: StepParityQKVSplitVJPCase) !void {
    try writer.interface.print(
        "{{\"status\":\"{s}\",\"reason\":\"{s}\",\"batch\":{d},\"seq_len\":{d},\"num_heads\":{d},\"head_dim\":{d},\"hidden_size\":{d},\"outer\":{d},\"components\":",
        .{
            case.status,
            case.reason,
            case.batch,
            case.seq_len,
            case.num_heads,
            case.head_dim,
            case.hidden_size,
            case.outer,
        },
    );
    try writeStepParityNamedTensorStatsMapJson(writer, case.components);
    try writer.interface.writeByte('}');
}

fn writeStepParityQKVSplitVJPProbeJson(writer: anytype, maybe_probe: ?StepParityQKVSplitVJPProbe) !void {
    const probe = maybe_probe orelse {
        try writer.interface.writeAll("null");
        return;
    };
    try writer.interface.print(
        "{{\"status\":\"{s}\",\"version\":{d},\"runtime\":\"{s}\",\"cases\":{{",
        .{ probe.status, probe.version, probe.runtime },
    );
    for (probe.cases, 0..) |case, i| {
        if (i > 0) try writer.interface.writeByte(',');
        try writer.interface.print("\"{s}\":", .{case.name});
        try writeStepParityQKVSplitVJPCaseJson(writer, case);
    }
    try writer.interface.writeAll("}}");
}

fn writeStepParityLoRAGradMatrixStatsJson(
    writer: anytype,
    adapters: ?*const fused_chunker_lora.LoRAAdapterSet,
    scale: f32,
) !void {
    const la = adapters orelse {
        try writer.interface.writeAll("null");
        return;
    };
    try writer.interface.writeByte('{');
    var first = true;
    for (la.layers) |*ll| {
        if (!first) try writer.interface.writeByte(',');
        first = false;
        try writer.interface.print("\"layer_{d:0>2}_{s}_A\":", .{ ll.layer_idx, ll.module_name });
        try writeFloatSliceStatsFromSliceJson(writer, ll.grad_A, scale);

        try writer.interface.print(",\"layer_{d:0>2}_{s}_B\":", .{ ll.layer_idx, ll.module_name });
        try writeFloatSliceStatsFromSliceJson(writer, ll.grad_B, scale);
    }
    try writer.interface.writeByte('}');
}

fn writeStepParityContrastiveJson(writer: anytype, maybe_ctx: ?StepParityContrastiveContext) !void {
    if (maybe_ctx) |ctx| {
        try writer.interface.print(
            "{{\"active_chunks\":{d},\"contrastive_loss\":{d:.9},\"total_loss\":{d:.9},\"embeddings\":",
            .{
                ctx.active_chunks,
                ctx.contrastive_loss,
                ctx.total_loss,
            },
        );
        try writeFloatSliceStatsJson(writer, ctx.embedding_stats);
        try writer.interface.writeAll(",\"grad\":");
        try writeFloatSliceStatsJson(writer, ctx.grad_stats);
        try writer.interface.writeAll(",\"first_active_index\":");
        if (ctx.first_active_index) |index| {
            try writer.interface.print("{d}", .{index});
        } else {
            try writer.interface.writeAll("null");
        }
        try writer.interface.writeAll(",\"first_active_doc_id\":");
        if (ctx.first_active_doc_id) |doc_id| {
            try writer.interface.print("{d}", .{doc_id});
        } else {
            try writer.interface.writeAll("null");
        }
        try writer.interface.writeAll(",\"first_active_embedding_sample\":");
        try writeFixedF32SampleJson(writer, ctx.first_active_embedding_sample[0..ctx.first_active_embedding_sample_len]);
        try writer.interface.writeAll(",\"first_active_grad_sample\":");
        try writeFixedF32SampleJson(writer, ctx.first_active_grad_sample[0..ctx.first_active_grad_sample_len]);
        try writer.interface.writeAll(",\"active_doc_id_sample\":");
        try writeFixedU32SampleJson(writer, ctx.active_doc_id_sample[0..ctx.active_doc_id_sample_len]);
        try writer.interface.writeAll(",\"active_embedding_norm_sample\":");
        try writeFixedF32SampleJson(writer, ctx.active_embedding_norm_sample[0..ctx.active_embedding_norm_sample_len]);
        try writer.interface.writeAll(",\"active_grad_norm_sample\":");
        try writeFixedF32SampleJson(writer, ctx.active_grad_norm_sample[0..ctx.active_grad_norm_sample_len]);
        try writer.interface.writeByte('}');
    } else {
        try writer.interface.writeAll("null");
    }
}

fn writeStepParityJson(
    path: []const u8,
    phase: []const u8,
    opts: *const Options,
    effective_lambda_embed: f32,
    epoch: usize,
    local_step: u32,
    global_step: u32,
    trainer_step_count: u32,
    batch_indices: []const usize,
    batch: *const fused_chunker_data.FusedBatch,
    total_tokens: usize,
    boundary_ctx: StepParityBoundaryContext,
    contrastive_ctx: ?StepParityContrastiveContext,
    upstream_grad_probe: ?StepParityUpstreamGradProbe,
    encoder_replay_upstream_probe: ?EncoderReplayUpstreamProbe,
    segment_vjp_probe: ?StepParitySegmentVJPProbe,
    layer_backward_decomp_probe: ?StepParityLayerBackwardDecompProbe,
    softmax_vjp_probe: ?StepParitySoftmaxVJPProbe,
    qkv_split_vjp_probe: ?StepParityQKVSplitVJPProbe,
    update_summary: ?StepParityUpdateSummary,
    lora_grad_adapters: ?*const fused_chunker_lora.LoRAAdapterSet,
) !void {
    const effective_boundary_dropout: f32 = if (opts.deterministic) 0.0 else opts.boundary_dropout;
    const hashes = computeStepParityBatchHashes(batch, total_tokens);
    const gold_rate = if (boundary_ctx.batch_stats.valid_tokens > 0)
        @as(f64, @floatFromInt(boundary_ctx.batch_stats.gold_positives)) / @as(f64, @floatFromInt(boundary_ctx.batch_stats.valid_tokens))
    else
        0.0;
    const pre_clip_lora_grad_scale: f32 = if (update_summary) |u|
        if (u.grad_clip.clip_scale > 0.0) 1.0 / u.grad_clip.clip_scale else 1.0
    else
        1.0;

    var file = try compat.cwd().createFile(compat.io(), path, .{ .truncate = true });
    defer file.close(compat.io());
    var buf: [8192]u8 = undefined;
    var writer = file.writerStreaming(compat.io(), &buf);
    try writer.interface.print(
        "{{\"tool\":\"zig_fused_step_parity\",\"schema_version\":1,\"phase\":\"{s}\",\"epoch\":{d},\"local_step\":{d},\"global_step\":{d},\"trainer_step_count\":{d},\"batch_size\":{d},\"max_seq_len\":{d},\"max_chunks\":{d},\"deterministic\":{},\"mixed_precision\":{},\"target_probe_layer\":{d},\"boundary_loss_type\":\"{s}\",\"pos_weight\":{d},\"boundary_dropout\":{d},\"boundary_focal_gamma\":{d},\"boundary_focal_alpha\":{d},\"contrastive_focal_gamma\":{d},\"contrastive_focal_alpha\":{d},\"lambda_chunk\":{d},\"lambda_embed\":{d},\"configured_lambda_embed\":{d},\"hashes\":{{\"sample_indices\":\"{x}\",\"input_ids\":\"{x}\",\"attention_mask\":\"{x}\",\"labels\":\"{x}\",\"chunks\":\"{x}\"}},\"batch_indices\":",
        .{
            phase,
            epoch + 1,
            local_step + 1,
            global_step,
            trainer_step_count,
            batch.batch_size,
            batch.max_seq_len,
            batch.max_chunks,
            opts.deterministic,
            opts.mixed_precision,
            opts.debug_encoder_probe_layer,
            @tagName(opts.boundary_loss_type),
            opts.boundary_pos_weight,
            effective_boundary_dropout,
            opts.boundary_focal_gamma,
            opts.boundary_focal_alpha,
            opts.contrastive_focal_gamma,
            opts.contrastive_focal_alpha,
            opts.lambda_chunk,
            effective_lambda_embed,
            opts.lambda_embed,
            hashes.sample_indices,
            hashes.input_ids,
            hashes.attention_mask,
            hashes.labels,
            hashes.chunks,
        },
    );
    try writeUsizeJsonArray(&writer, batch_indices);
    try writer.interface.print(
        ",\"supervision\":{{\"valid_tokens\":{d},\"gold_positives\":{d},\"gold_rate\":{d:.9}}},\"features\":{{\"mean\":{d:.9},\"rms\":{d:.9},\"max_abs\":{d:.9}}},\"boundary\":{{\"loss\":{d:.9},\"eval_f1\":{d:.9},\"tp\":{d},\"fp\":{d},\"fn\":{d},\"predicted_positives\":{d},\"prob_gold_pos\":{d:.9},\"prob_gold_neg\":{d:.9},\"grad_norm_w1\":{d:.9},\"grad_norm_b1\":{d:.9},\"grad_norm_w2\":{d:.9},\"grad_norm_b2\":{d:.9},\"grad_max_abs_w1\":{d:.9},\"grad_max_abs_b1\":{d:.9},\"grad_max_abs_w2\":{d:.9},\"grad_max_abs_b2\":{d:.9},\"features_grad_norm\":{d:.9},\"features_grad_max_abs\":{d:.9},\"has_features_grad\":{}}}",
        .{
            boundary_ctx.batch_stats.valid_tokens,
            boundary_ctx.batch_stats.gold_positives,
            gold_rate,
            boundary_ctx.batch_stats.feature_mean,
            boundary_ctx.batch_stats.feature_rms,
            boundary_ctx.batch_stats.feature_max_abs,
            boundary_ctx.debug.boundary_loss,
            boundary_ctx.debug.eval_f1,
            boundary_ctx.debug.eval_tp,
            boundary_ctx.debug.eval_fp,
            boundary_ctx.debug.eval_fn,
            boundary_ctx.debug.eval_predicted_positives,
            boundary_ctx.debug.eval_mean_prob_gold_positive,
            boundary_ctx.debug.eval_mean_prob_gold_negative,
            boundary_ctx.debug.grad_norm_w1,
            boundary_ctx.debug.grad_norm_b1,
            boundary_ctx.debug.grad_norm_w2,
            boundary_ctx.debug.grad_norm_b2,
            boundary_ctx.debug.grad_max_abs_w1,
            boundary_ctx.debug.grad_max_abs_b1,
            boundary_ctx.debug.grad_max_abs_w2,
            boundary_ctx.debug.grad_max_abs_b2,
            boundary_ctx.debug.features_grad_norm,
            boundary_ctx.debug.features_grad_max_abs,
            boundary_ctx.debug.has_features_grad,
        },
    );
    try writer.interface.writeAll(",\"contrastive\":");
    try writeStepParityContrastiveJson(&writer, contrastive_ctx);
    try writer.interface.writeAll(",\"boundary_forward_probe\":");
    try writeBoundaryForwardProbeJson(&writer, boundary_ctx.debug.boundary_forward_probe);
    try writer.interface.writeAll(",\"embedding_probe\":");
    try writeEmbeddingProbeJson(&writer, boundary_ctx.embedding_probe);
    try writer.interface.writeAll(",\"embedding_table_row_probe\":");
    try writeEmbeddingRowProbeJson(&writer, boundary_ctx.embedding_table_rows);
    try writer.interface.writeAll(",\"embedding_lookup_row_probe\":");
    try writeEmbeddingRowProbeJson(&writer, boundary_ctx.embedding_lookup_rows);
    try writer.interface.writeAll(",\"boundary_checkpoint_probe\":");
    try writeBoundaryCheckpointProbeJson(
        &writer,
        boundary_ctx.debug.boundary_checkpoint_probe,
        boundary_ctx.final_norm_weight,
        boundary_ctx.final_norm_bias,
    );
    try writer.interface.writeAll(",\"encoder_activation_input_probe\":");
    try writeEncoderActivationInputProbeJson(&writer, boundary_ctx.encoder_activation_inputs);
    try writer.interface.writeAll(",\"encoder_projection_decomposition_probe\":");
    try writeEncoderProjectionDecompositionProbeJson(&writer, boundary_ctx.encoder_projection_decompositions);
    try writer.interface.writeAll(",\"encoder_attention_internal_probe\":");
    try writeEncoderAttentionInternalProbeJson(&writer, boundary_ctx.encoder_attention_internals);
    try writer.interface.writeAll(",\"encoder_attention_row_probe\":");
    try writeEncoderAttentionRowProbeJson(&writer, boundary_ctx.encoder_attention_rows);
    try writer.interface.writeAll(",\"encoder_layer_input_probe\":");
    try writeEncoderLayerInputProbeJson(&writer, boundary_ctx.encoder_layer_inputs);
    try writer.interface.writeAll(",\"encoder_layer_state_probe\":");
    try writeEncoderLayerStateProbeJson(&writer, boundary_ctx.encoder_layer_states);
    try writer.interface.writeAll(",\"encoder_replay_input\":");
    try writeEncoderReplayInputProbeJson(&writer, boundary_ctx.encoder_replay_input);
    try writer.interface.writeAll(",\"encoder_replay_upstream\":");
    try writeEncoderReplayUpstreamProbeJson(&writer, encoder_replay_upstream_probe);
    try writer.interface.writeAll(",\"upstream_grad_probe\":");
    try writeStepParityUpstreamGradProbeJson(&writer, upstream_grad_probe);
    try writer.interface.writeAll(",\"segment_vjp_probe\":");
    try writeStepParitySegmentVJPProbeJson(&writer, segment_vjp_probe);
    try writer.interface.writeAll(",\"layer_backward_decomp_probe\":");
    try writeStepParityLayerBackwardDecompProbeJson(&writer, layer_backward_decomp_probe);
    try writer.interface.writeAll(",\"softmax_vjp_probe\":");
    try writeStepParitySoftmaxVJPProbeJson(&writer, softmax_vjp_probe);
    try writer.interface.writeAll(",\"qkv_split_vjp_probe\":");
    try writeStepParityQKVSplitVJPProbeJson(&writer, qkv_split_vjp_probe);
    try writer.interface.writeAll(",\"update\":");
    if (update_summary) |u| {
        const boundary_grad_scale: f64 = @floatCast(u.grad_clip.clip_scale);
        try writer.interface.print(
            "{{\"grad_elems\":{d},\"lora_grad_norm_pre_clip\":{d:.9},\"lora_grad_norm_post_clip\":{d:.9},\"extra_grad_norm_pre_clip\":{d:.9},\"extra_grad_norm_post_clip\":{d:.9},\"boundary_dense1_weight_grad_norm_pre_clip\":{d:.9},\"boundary_dense1_weight_grad_norm_post_clip\":{d:.9},\"boundary_dense1_bias_grad_norm_pre_clip\":{d:.9},\"boundary_dense1_bias_grad_norm_post_clip\":{d:.9},\"boundary_dense2_weight_grad_norm_pre_clip\":{d:.9},\"boundary_dense2_weight_grad_norm_post_clip\":{d:.9},\"boundary_dense2_bias_grad_norm_pre_clip\":{d:.9},\"boundary_dense2_bias_grad_norm_post_clip\":{d:.9}",
            .{
                u.grad_clip.elems,
                u.grad_clip.lora_norm_before_clip,
                u.grad_clip.lora_norm_after_clip,
                u.grad_clip.extra_norm_before_clip,
                u.grad_clip.extra_norm_after_clip,
                boundary_ctx.debug.grad_norm_w1,
                @as(f64, boundary_ctx.debug.grad_norm_w1) * boundary_grad_scale,
                boundary_ctx.debug.grad_norm_b1,
                @as(f64, boundary_ctx.debug.grad_norm_b1) * boundary_grad_scale,
                boundary_ctx.debug.grad_norm_w2,
                @as(f64, boundary_ctx.debug.grad_norm_w2) * boundary_grad_scale,
                boundary_ctx.debug.grad_norm_b2,
                @as(f64, boundary_ctx.debug.grad_norm_b2) * boundary_grad_scale,
            },
        );
        try writer.interface.print(
            ",\"grad_norm_pre_clip\":{d:.9},\"grad_norm_post_clip\":{d:.9},\"grad_max_abs_pre_clip\":{d:.9},\"grad_max_abs_post_clip\":{d:.9},\"grad_clip_scale\":{d:.9},\"invalid_grad_repaired\":{d},\"max_grad_norm\":{d},\"active_adapters\":{d},\"active_matrices\":{d},\"base_lr\":{d},\"lora_plus_ratio\":{d},\"optimizer_step_count\":{d},\"param_elems\":{d},\"param_norm\":{d:.9},\"update_elems\":{d},\"update_norm\":{d:.9},\"update_max_abs\":{d:.9},\"update_mean_abs\":{d:.9},\"update_to_param\":{d:.9},\"adam_m_norm\":{d:.9},\"adam_v_norm\":{d:.9},\"repaired_after_update\":{d},\"lora_update_matrix_stats\":",
            .{
                u.grad_clip.norm_before_clip,
                u.grad_clip.norm_after_clip,
                u.grad_clip.max_abs_before_clip,
                u.grad_clip.max_abs_after_clip,
                u.grad_clip.clip_scale,
                u.grad_clip.nonfinite_repaired,
                opts.max_grad_norm,
                u.active_adapters,
                u.active_matrices,
                u.base_lr,
                u.lora_plus_ratio,
                u.optimizer_step_count,
                u.param_elems,
                u.param_norm,
                u.update_elems,
                u.update_norm,
                u.update_max_abs,
                u.update_mean_abs,
                u.update_to_param,
                u.adam_m_norm,
                u.adam_v_norm,
                u.repaired_after_update,
            },
        );
        try writeLoRAUpdateMatrixStatsJson(&writer, u.matrix_stats);
        try writer.interface.writeByte('}');
    } else {
        try writer.interface.writeAll("null");
    }
    try writer.interface.writeAll(",\"lora_grad_matrix_stats\":");
    try writeStepParityLoRAGradMatrixStatsJson(&writer, lora_grad_adapters, 1.0);
    try writer.interface.writeAll(",\"lora_grad_matrix_stats_post_clip\":");
    try writeStepParityLoRAGradMatrixStatsJson(&writer, lora_grad_adapters, 1.0);
    try writer.interface.writeAll(",\"lora_grad_matrix_stats_pre_clip\":");
    try writeStepParityLoRAGradMatrixStatsJson(&writer, lora_grad_adapters, pre_clip_lora_grad_scale);
    try writer.interface.writeAll("}\n");
    try writer.interface.flush();
}

fn computeStepParityContrastiveContext(
    allocator: std.mem.Allocator,
    loss_config: fused_chunker_loss.FusedLossConfig,
    chunk_embeddings: []const f32,
    chunk_mask: []const f32,
    doc_ids: []const u32,
    B: usize,
    C: usize,
    E: usize,
) !StepParityContrastiveContext {
    const n = B * C;
    var ctx = StepParityContrastiveContext{};
    ctx.embedding_stats.addSlice(chunk_embeddings);
    const active_limit = @min(chunk_mask.len, n);
    for (chunk_mask[0..active_limit], 0..) |m, chunk_idx| {
        if (m <= 0.5) continue;
        ctx.active_chunks += 1;
        if (ctx.active_doc_id_sample_len < ctx.active_doc_id_sample.len and chunk_idx < doc_ids.len) {
            ctx.active_doc_id_sample[ctx.active_doc_id_sample_len] = doc_ids[chunk_idx];
            ctx.active_doc_id_sample_len += 1;
        }
        const base = chunk_idx * E;
        if (base + E > chunk_embeddings.len) continue;
        var norm_sq: f64 = 0;
        for (chunk_embeddings[base .. base + E]) |value| {
            norm_sq += @as(f64, value) * @as(f64, value);
        }
        if (ctx.active_embedding_norm_sample_len < ctx.active_embedding_norm_sample.len) {
            ctx.active_embedding_norm_sample[ctx.active_embedding_norm_sample_len] = @as(f32, @floatCast(@sqrt(norm_sq)));
            ctx.active_embedding_norm_sample_len += 1;
        }
        if (ctx.first_active_index == null) {
            ctx.first_active_index = chunk_idx;
            if (chunk_idx < doc_ids.len) ctx.first_active_doc_id = doc_ids[chunk_idx];
            const sample_len = @min(E, ctx.first_active_embedding_sample.len);
            @memcpy(ctx.first_active_embedding_sample[0..sample_len], chunk_embeddings[base .. base + sample_len]);
            ctx.first_active_embedding_sample_len = sample_len;
        }
    }

    var result = if (loss_config.use_mrl) mrl_blk: {
        const mrl_config = infonce_cpu.MatryoshkaConfig{
            .dims = loss_config.mrl_dims,
            .weights = loss_config.mrl_weights,
        };
        var mrl = try infonce_cpu.computeMatryoshkaLossAndGrad(
            allocator,
            chunk_embeddings,
            chunk_mask,
            doc_ids,
            n,
            E,
            mrl_config,
            loss_config.temperature,
            loss_config.contrastive_focal_gamma,
            loss_config.contrastive_focal_alpha,
        );
        defer mrl.deinit(allocator);
        const grad = try allocator.dupe(f32, mrl.grad);
        errdefer allocator.free(grad);
        for (grad) |*g| g.* *= loss_config.lambda_embed;
        break :mrl_blk infonce_cpu.ContrastiveLossResult{
            .contrastive_loss = mrl.total_loss,
            .total_loss = mrl.total_loss * @as(f64, loss_config.lambda_embed),
            .grad = grad,
        };
    } else try infonce_cpu.computeContrastiveLossOnCPU(
        allocator,
        chunk_embeddings,
        chunk_mask,
        doc_ids,
        @as(f64, loss_config.temperature),
        @as(f64, loss_config.lambda_embed),
        B,
        C,
        E,
        @as(f64, loss_config.contrastive_focal_gamma),
        @as(f64, loss_config.contrastive_focal_alpha),
    );
    defer result.deinit(allocator);

    ctx.contrastive_loss = result.contrastive_loss;
    ctx.total_loss = result.total_loss;
    ctx.grad_stats.addSlice(result.grad);
    if (ctx.first_active_index) |chunk_idx| {
        const base = chunk_idx * E;
        if (base + E <= result.grad.len) {
            const sample_len = @min(E, ctx.first_active_grad_sample.len);
            @memcpy(ctx.first_active_grad_sample[0..sample_len], result.grad[base .. base + sample_len]);
            ctx.first_active_grad_sample_len = sample_len;
        }
    }
    for (chunk_mask[0..active_limit], 0..) |m, chunk_idx| {
        if (m <= 0.5) continue;
        const base = chunk_idx * E;
        if (base + E > result.grad.len) continue;
        var norm_sq: f64 = 0;
        for (result.grad[base .. base + E]) |value| {
            norm_sq += @as(f64, value) * @as(f64, value);
        }
        if (ctx.active_grad_norm_sample_len < ctx.active_grad_norm_sample.len) {
            ctx.active_grad_norm_sample[ctx.active_grad_norm_sample_len] = @as(f32, @floatCast(@sqrt(norm_sq)));
            ctx.active_grad_norm_sample_len += 1;
        }
    }
    return ctx;
}

fn printLoRAUpdateDebugDetail(
    step: u32,
    layer_idx: u32,
    module_name: []const u8,
    matrix_name: []const u8,
    lr: f32,
    grad: []const f32,
    before: []const f32,
    after: []const f32,
    maybe_state: ?optimizers.ParamState,
) !void {
    var grad_stats = FloatSliceStats{};
    grad_stats.addSlice(grad);
    var param_stats = FloatSliceStats{};
    param_stats.addSlice(after);
    var update_stats = FloatSliceStats{};
    try update_stats.addDelta(before, after);
    var m_stats = FloatSliceStats{};
    var v_stats = FloatSliceStats{};
    if (maybe_state) |state| {
        m_stats.addSlice(state.m);
        v_stats.addSlice(state.v);
    }
    print(
        "lora_update_detail step={d} layer={d} module={s} matrix={s} lr={d} grad_norm={d:.9} grad_max_abs={d:.9} param_norm={d:.9} update_norm={d:.9} update_max_abs={d:.9} update_mean_abs={d:.9} adam_m_norm={d:.9} adam_v_norm={d:.9}\n",
        .{
            step,
            layer_idx,
            module_name,
            matrix_name,
            lr,
            grad_stats.l2(),
            grad_stats.max_abs,
            param_stats.l2(),
            update_stats.l2(),
            update_stats.max_abs,
            update_stats.meanAbs(),
            m_stats.l2(),
            v_stats.l2(),
        },
    );
}

fn captureStepParitySegmentVJPProbe(
    allocator: std.mem.Allocator,
    target_layer: u32,
    segment_start: u32,
    segment_end: u32,
    include_hidden_grad: bool,
    include_adapter_grads: bool,
    upstream_grad: []const f32,
    vjp_result: *const segmented_encoder.SegmentVJPResult,
    lora_layers: []const fused_chunker_lora.LoRALayer,
) !StepParitySegmentVJPProbe {
    var probe = StepParitySegmentVJPProbe{
        .target_layer = target_layer,
        .segment_start = segment_start,
        .segment_end = segment_end,
        .include_hidden_grad = include_hidden_grad,
        .include_adapter_grads = include_adapter_grads,
        .runtime = segmentVJPProfileRuntimeName(vjp_result.profile),
        .profile = vjp_result.profile,
    };
    errdefer probe.deinit(allocator);

    probe.upstream.addSlice(upstream_grad);
    if (vjp_result.hidden_grad) |hidden_grad| {
        probe.hidden_grad.addSlice(hidden_grad);
    }

    var adapter_a_by_name: std.ArrayListUnmanaged(StepParityNamedTensorStats) = .empty;
    defer adapter_a_by_name.deinit(allocator);
    errdefer for (adapter_a_by_name.items) |*entry| entry.deinit(allocator);
    var adapter_b_by_name: std.ArrayListUnmanaged(StepParityNamedTensorStats) = .empty;
    defer adapter_b_by_name.deinit(allocator);
    errdefer for (adapter_b_by_name.items) |*entry| entry.deinit(allocator);

    for (lora_layers) |*layer| {
        if (layer.layer_idx < segment_start or layer.layer_idx >= segment_end) continue;

        var a_stats = StepParityTensorSliceStats{};
        a_stats.addSlice(layer.grad_A);
        probe.adapter_a.addStats(a_stats);
        const a_name = try std.fmt.allocPrint(
            allocator,
            "layer_{d:0>2}_{s}_A",
            .{ layer.layer_idx, layer.module_name },
        );
        adapter_a_by_name.append(allocator, .{ .name = a_name, .stats = a_stats }) catch |err| {
            allocator.free(a_name);
            return err;
        };

        var b_stats = StepParityTensorSliceStats{};
        b_stats.addSlice(layer.grad_B);
        probe.adapter_b.addStats(b_stats);
        const b_name = try std.fmt.allocPrint(
            allocator,
            "layer_{d:0>2}_{s}_B",
            .{ layer.layer_idx, layer.module_name },
        );
        adapter_b_by_name.append(allocator, .{ .name = b_name, .stats = b_stats }) catch |err| {
            allocator.free(b_name);
            return err;
        };
    }

    probe.adapter_a_by_name = try adapter_a_by_name.toOwnedSlice(allocator);
    adapter_a_by_name = .empty;
    probe.adapter_b_by_name = try adapter_b_by_name.toOwnedSlice(allocator);
    adapter_b_by_name = .empty;
    return probe;
}

fn backwardDecompComponentName(name: []const u8) []const u8 {
    if (std.mem.eql(u8, name, "__substage_mlp_wo_input")) return "mlp_wo_input_grad";
    if (std.mem.eql(u8, name, "__substage_gelu_input")) return "gelu_input_grad";
    if (std.mem.eql(u8, name, "__substage_gate_value")) return "gate_value_grad";
    if (std.mem.eql(u8, name, "__substage_gate_input")) return "gate_input_grad";
    if (std.mem.eql(u8, name, "__substage_wi_output")) return "wi_output_grad";
    if (std.mem.eql(u8, name, "__substage_mlp_norm_output")) return "mlp_norm_output_grad";
    if (std.mem.eql(u8, name, "__substage_hidden_after_attn")) return "hidden_after_attn_grad";
    if (std.mem.eql(u8, name, "__substage_attn_merged")) return "attn_merged_grad";
    if (std.mem.eql(u8, name, "__substage_q_raw")) return "q_raw_grad";
    if (std.mem.eql(u8, name, "__substage_k_raw")) return "k_raw_grad";
    if (std.mem.eql(u8, name, "__substage_v_raw")) return "v_raw_grad";
    if (std.mem.eql(u8, name, "__substage_q_rope")) return "q_rope_grad";
    if (std.mem.eql(u8, name, "__substage_k_rope")) return "k_rope_grad";
    if (std.mem.eql(u8, name, "__substage_v_attention_input")) return "v_attention_input_grad";
    if (std.mem.eql(u8, name, "__substage_scores_raw")) return "scores_raw_grad";
    if (std.mem.eql(u8, name, "__substage_scores_masked")) return "scores_masked_grad";
    if (std.mem.eql(u8, name, "__substage_attention_probs")) return "probs_grad";
    if (std.mem.eql(u8, name, "__substage_attn_normed")) return "attn_normed_grad";
    if (std.mem.eql(u8, name, "__substage_q_attn_normed")) return "q_attn_normed_grad";
    if (std.mem.eql(u8, name, "__substage_k_attn_normed")) return "k_attn_normed_grad";
    if (std.mem.eql(u8, name, "__substage_v_attn_normed")) return "v_attn_normed_grad";
    if (std.mem.eql(u8, name, "__substage_hidden_in")) return "hidden_in_grad";
    return name;
}

fn missingBackwardDecompStage(reason: []const u8) StepParityBackwardDecompStage {
    return .{
        .status = "missing",
        .reason = reason,
    };
}

fn capturedBackwardDecompStatsStage(values: []const f32) StepParityBackwardDecompStage {
    var stage = StepParityBackwardDecompStage{
        .status = "captured",
        .reason = "",
    };
    stage.stats.addSlice(values);
    return stage;
}

fn cloneStepParityNamedStatsSlice(
    allocator: std.mem.Allocator,
    entries: []const StepParityNamedTensorStats,
) ![]StepParityNamedTensorStats {
    var out: std.ArrayListUnmanaged(StepParityNamedTensorStats) = .empty;
    defer out.deinit(allocator);
    errdefer for (out.items) |*entry| entry.deinit(allocator);
    for (entries) |entry| {
        const name = try allocator.dupe(u8, entry.name);
        errdefer allocator.free(name);
        try out.append(allocator, .{
            .name = name,
            .stats = entry.stats,
        });
    }
    const owned = try out.toOwnedSlice(allocator);
    out = .empty;
    return owned;
}

fn capturedBackwardDecompFullLayerStage(
    allocator: std.mem.Allocator,
    segment_probe: StepParitySegmentVJPProbe,
) !StepParityBackwardDecompStage {
    var stage = StepParityBackwardDecompStage{
        .status = "captured",
        .reason = "",
        .stats = segment_probe.hidden_grad,
        .adapter_a = segment_probe.adapter_a,
        .adapter_b = segment_probe.adapter_b,
    };
    errdefer stage.deinit(allocator);
    stage.adapter_a_by_name = try cloneStepParityNamedStatsSlice(allocator, segment_probe.adapter_a_by_name);
    stage.adapter_b_by_name = try cloneStepParityNamedStatsSlice(allocator, segment_probe.adapter_b_by_name);
    return stage;
}

fn addBackwardDecompAdapterGradientStats(
    allocator: std.mem.Allocator,
    stage: *StepParityBackwardDecompStage,
    gradients: []const segmented_encoder.NamedGradient,
) !void {
    var adapter_a_by_name: std.ArrayListUnmanaged(StepParityNamedTensorStats) = .empty;
    defer adapter_a_by_name.deinit(allocator);
    errdefer for (adapter_a_by_name.items) |*entry| entry.deinit(allocator);
    var adapter_b_by_name: std.ArrayListUnmanaged(StepParityNamedTensorStats) = .empty;
    defer adapter_b_by_name.deinit(allocator);
    errdefer for (adapter_b_by_name.items) |*entry| entry.deinit(allocator);

    for (gradients) |grad| {
        var stats = StepParityTensorSliceStats{};
        stats.addSlice(grad.values);
        const name = try allocator.dupe(u8, grad.name);
        errdefer allocator.free(name);
        if (std.mem.endsWith(u8, grad.name, "_A")) {
            stage.adapter_a.addStats(stats);
            try adapter_a_by_name.append(allocator, .{ .name = name, .stats = stats });
        } else if (std.mem.endsWith(u8, grad.name, "_B")) {
            stage.adapter_b.addStats(stats);
            try adapter_b_by_name.append(allocator, .{ .name = name, .stats = stats });
        } else {
            allocator.free(name);
        }
    }

    stage.adapter_a_by_name = try adapter_a_by_name.toOwnedSlice(allocator);
    adapter_a_by_name = .empty;
    stage.adapter_b_by_name = try adapter_b_by_name.toOwnedSlice(allocator);
    adapter_b_by_name = .empty;
}

fn capturedBackwardDecompStageFromResult(
    allocator: std.mem.Allocator,
    result: *const segmented_encoder.LayerBackwardSubstageResult,
) !StepParityBackwardDecompStage {
    var stage = StepParityBackwardDecompStage{
        .status = "captured",
        .reason = "",
    };
    errdefer stage.deinit(allocator);

    var components: std.ArrayListUnmanaged(StepParityBackwardDecompComponent) = .empty;
    defer components.deinit(allocator);
    errdefer for (components.items) |*entry| entry.deinit(allocator);

    for (result.stage_grads) |grad| {
        var stats = StepParityTensorSliceStats{};
        stats.addSlice(grad.values);
        stage.stats.addStats(stats);
        const name = try allocator.dupe(u8, backwardDecompComponentName(grad.name));
        errdefer allocator.free(name);
        try components.append(allocator, .{
            .name = name,
            .stats = stats,
        });
    }

    stage.components = try components.toOwnedSlice(allocator);
    components = .empty;
    try addBackwardDecompAdapterGradientStats(allocator, &stage, result.adapter_grads);
    return stage;
}

fn captureLayerBackwardSubstage(
    allocator: std.mem.Allocator,
    cb: *const ComputeBackend,
    graph_config: modern_bert_graph.Config,
    actual_batch: usize,
    max_seq: usize,
    target_layer: u32,
    stage: modern_bert_graph.LayerBackwardSubstage,
    lora_rank: u32,
    lora_alpha: f32,
    upstream_grad: []const f32,
    inputs: segmented_encoder.LayerBackwardSubstageInputs,
    lora_layers: []fused_chunker_lora.LoRALayer,
) !StepParityBackwardDecompStage {
    var session = try segmented_encoder.ModernBertLayerBackwardSubstageSession.init(
        allocator,
        graph_config,
        @intCast(actual_batch),
        @intCast(max_seq),
        target_layer,
        stage,
        lora_rank,
        lora_alpha,
        .mpsgraph_required,
    );
    defer session.deinit();

    var result = try session.execute(cb, .{
        .upstream_grad = upstream_grad,
        .primary = inputs.primary,
        .q_raw = inputs.q_raw,
        .k_raw = inputs.k_raw,
        .v_raw = inputs.v_raw,
        .hidden_in_aux = inputs.hidden_in_aux,
        .hidden_after_attn_aux = inputs.hidden_after_attn_aux,
        .gate_input_aux = inputs.gate_input_aux,
        .gate_value_aux = inputs.gate_value_aux,
        .attention_mask = inputs.attention_mask,
    }, lora_layers);
    defer result.deinit();
    return capturedBackwardDecompStageFromResult(allocator, &result);
}

fn decompStageCaptured(stage: StepParityBackwardDecompStage) bool {
    return std.mem.eql(u8, stage.status, "captured");
}

fn captureStepParityLayerBackwardDecompProbe(
    allocator: std.mem.Allocator,
    cb: *const ComputeBackend,
    graph_config: modern_bert_graph.Config,
    actual_batch: usize,
    max_seq: usize,
    target_layer: u32,
    segment_start: u32,
    segment_end: u32,
    lora_rank: u32,
    lora_alpha: f32,
    upstream_grad: []const f32,
    attention_mask: []const f32,
    act_buf: *const modern_bert.ActivationBuffer,
    segment_probe: StepParitySegmentVJPProbe,
    lora_layers: []fused_chunker_lora.LoRALayer,
) !StepParityLayerBackwardDecompProbe {
    var probe = StepParityLayerBackwardDecompProbe{
        .target_layer = target_layer,
        .segment_start = segment_start,
        .segment_end = segment_end,
        .runtime = segment_probe.runtime,
        .incoming_upstream = capturedBackwardDecompStatsStage(upstream_grad),
        .full_layer_hidden_grad = try capturedBackwardDecompFullLayerStage(allocator, segment_probe),
    };
    errdefer probe.deinit(allocator);

    if (segment_start != target_layer or segment_end != target_layer + 1) {
        const reason = "requires_single_layer_vjp_segment";
        probe.status = "partial";
        probe.mlp_wo = missingBackwardDecompStage(reason);
        probe.mlp_gelu_input = missingBackwardDecompStage(reason);
        probe.mlp_gate_value = missingBackwardDecompStage(reason);
        probe.mlp_gate_input = missingBackwardDecompStage(reason);
        probe.mlp_wi_output = missingBackwardDecompStage(reason);
        probe.mlp_norm_output = missingBackwardDecompStage(reason);
        probe.mlp_hidden_after_attn = missingBackwardDecompStage(reason);
        probe.attn_out_proj = missingBackwardDecompStage(reason);
        probe.attention_core = missingBackwardDecompStage(reason);
        probe.attention_core_post_rope = missingBackwardDecompStage(reason);
        probe.attention_scores_raw = missingBackwardDecompStage(reason);
        probe.attention_scores_masked = missingBackwardDecompStage(reason);
        probe.attention_probs = missingBackwardDecompStage(reason);
        probe.qkv_proj = missingBackwardDecompStage(reason);
        probe.qkv_proj_split = missingBackwardDecompStage(reason);
        probe.attn_norm_hidden_in = missingBackwardDecompStage(reason);
        return probe;
    }

    const hidden_in = if (act_buf.findLayerInput(target_layer)) |layer_input| layer_input.input else null;
    const hidden_after_attn = act_buf.findLayerState(target_layer, "hidden_after_attn");
    const mlp_norm_output = act_buf.findLayerState(target_layer, "mlp_norm_output");
    const wi_output = act_buf.findLayerState(target_layer, "wi_output");
    const gate_input = act_buf.findLayerState(target_layer, "gate_input");
    const gate_value = act_buf.findLayerState(target_layer, "gate_value");
    const gelu_input = act_buf.findLayerState(target_layer, "gelu_input");
    const mlp_wo_input = findLayerActivationInput(act_buf, target_layer, "wo");
    const attn_merged = findLayerActivationInput(act_buf, target_layer, "out_proj");
    const attn_normed = findLayerActivationInput(act_buf, target_layer, "query_proj");
    const q_raw = findLayerAttentionInternal(act_buf, target_layer, "q_raw");
    const k_raw = findLayerAttentionInternal(act_buf, target_layer, "k_raw");
    const v_raw = findLayerAttentionInternal(act_buf, target_layer, "v_raw");
    const q_rope = findLayerAttentionInternal(act_buf, target_layer, "q_rope");
    const k_rope = findLayerAttentionInternal(act_buf, target_layer, "k_rope");
    var attention_core_tensors: ?StepParityAttentionCoreTensors = null;
    defer if (attention_core_tensors) |*tensors| tensors.deinit(allocator);
    if (q_rope != null and k_rope != null) {
        attention_core_tensors = try computeStepParityAttentionCoreTensors(
            allocator,
            q_rope.?,
            k_rope.?,
            attention_mask,
            actual_batch,
            max_seq,
            @intCast(graph_config.num_attention_heads),
            @intCast(graph_config.head_dim),
            !modern_bert_graph.isGlobalAttentionLayer(graph_config, target_layer),
            graph_config.local_attention_window,
        );
    }

    probe.mlp_wo = if (mlp_wo_input != null and hidden_after_attn != null)
        try captureLayerBackwardSubstage(
            allocator,
            cb,
            graph_config,
            actual_batch,
            max_seq,
            target_layer,
            .mlp_wo,
            lora_rank,
            lora_alpha,
            upstream_grad,
            .{
                .upstream_grad = upstream_grad,
                .primary = mlp_wo_input.?,
                .hidden_after_attn_aux = hidden_after_attn.?,
                .attention_mask = attention_mask,
            },
            lora_layers,
        )
    else
        missingBackwardDecompStage("missing_mlp_wo_or_hidden_after_attn_capture");

    probe.mlp_gelu_input = if (gelu_input != null and gate_value != null and hidden_after_attn != null)
        try captureLayerBackwardSubstage(
            allocator,
            cb,
            graph_config,
            actual_batch,
            max_seq,
            target_layer,
            .mlp_gelu_input,
            lora_rank,
            lora_alpha,
            upstream_grad,
            .{
                .upstream_grad = upstream_grad,
                .primary = gelu_input.?,
                .gate_value_aux = gate_value.?,
                .hidden_after_attn_aux = hidden_after_attn.?,
                .attention_mask = attention_mask,
            },
            lora_layers,
        )
    else
        missingBackwardDecompStage("missing_gelu_input_gate_value_or_hidden_after_attn_capture");

    probe.mlp_gate_value = if (gate_value != null and gate_input != null and hidden_after_attn != null)
        try captureLayerBackwardSubstage(
            allocator,
            cb,
            graph_config,
            actual_batch,
            max_seq,
            target_layer,
            .mlp_gate_value,
            lora_rank,
            lora_alpha,
            upstream_grad,
            .{
                .upstream_grad = upstream_grad,
                .primary = gate_value.?,
                .gate_input_aux = gate_input.?,
                .hidden_after_attn_aux = hidden_after_attn.?,
                .attention_mask = attention_mask,
            },
            lora_layers,
        )
    else
        missingBackwardDecompStage("missing_gate_value_gate_input_or_hidden_after_attn_capture");

    probe.mlp_gate_input = if (gate_input != null and gate_value != null and hidden_after_attn != null)
        try captureLayerBackwardSubstage(
            allocator,
            cb,
            graph_config,
            actual_batch,
            max_seq,
            target_layer,
            .mlp_gate_input,
            lora_rank,
            lora_alpha,
            upstream_grad,
            .{
                .upstream_grad = upstream_grad,
                .primary = gate_input.?,
                .gate_value_aux = gate_value.?,
                .hidden_after_attn_aux = hidden_after_attn.?,
                .attention_mask = attention_mask,
            },
            lora_layers,
        )
    else
        missingBackwardDecompStage("missing_gate_input_gate_value_or_hidden_after_attn_capture");

    probe.mlp_wi_output = if (wi_output != null and hidden_after_attn != null)
        try captureLayerBackwardSubstage(
            allocator,
            cb,
            graph_config,
            actual_batch,
            max_seq,
            target_layer,
            .mlp_wi_output,
            lora_rank,
            lora_alpha,
            upstream_grad,
            .{
                .upstream_grad = upstream_grad,
                .primary = wi_output.?,
                .hidden_after_attn_aux = hidden_after_attn.?,
                .attention_mask = attention_mask,
            },
            lora_layers,
        )
    else
        missingBackwardDecompStage("missing_wi_output_or_hidden_after_attn_capture");

    probe.mlp_norm_output = if (mlp_norm_output != null and hidden_after_attn != null)
        try captureLayerBackwardSubstage(
            allocator,
            cb,
            graph_config,
            actual_batch,
            max_seq,
            target_layer,
            .mlp_norm_output,
            lora_rank,
            lora_alpha,
            upstream_grad,
            .{
                .upstream_grad = upstream_grad,
                .primary = mlp_norm_output.?,
                .hidden_after_attn_aux = hidden_after_attn.?,
                .attention_mask = attention_mask,
            },
            lora_layers,
        )
    else
        missingBackwardDecompStage("missing_mlp_norm_output_or_hidden_after_attn_capture");

    probe.mlp_hidden_after_attn = if (hidden_after_attn != null)
        try captureLayerBackwardSubstage(
            allocator,
            cb,
            graph_config,
            actual_batch,
            max_seq,
            target_layer,
            .mlp_hidden_after_attn,
            lora_rank,
            lora_alpha,
            upstream_grad,
            .{
                .upstream_grad = upstream_grad,
                .primary = hidden_after_attn.?,
                .attention_mask = attention_mask,
            },
            lora_layers,
        )
    else
        missingBackwardDecompStage("missing_hidden_after_attn_capture");

    probe.attn_out_proj = if (attn_merged != null and hidden_in != null)
        try captureLayerBackwardSubstage(
            allocator,
            cb,
            graph_config,
            actual_batch,
            max_seq,
            target_layer,
            .attn_out_proj,
            lora_rank,
            lora_alpha,
            upstream_grad,
            .{
                .upstream_grad = upstream_grad,
                .primary = attn_merged.?,
                .hidden_in_aux = hidden_in.?,
                .attention_mask = attention_mask,
            },
            lora_layers,
        )
    else
        missingBackwardDecompStage("missing_attn_merged_or_hidden_in_capture");

    probe.attention_core = if (q_raw != null and k_raw != null and v_raw != null and hidden_in != null)
        try captureLayerBackwardSubstage(
            allocator,
            cb,
            graph_config,
            actual_batch,
            max_seq,
            target_layer,
            .attention_core,
            lora_rank,
            lora_alpha,
            upstream_grad,
            .{
                .upstream_grad = upstream_grad,
                .q_raw = q_raw.?,
                .k_raw = k_raw.?,
                .v_raw = v_raw.?,
                .hidden_in_aux = hidden_in.?,
                .attention_mask = attention_mask,
            },
            lora_layers,
        )
    else
        missingBackwardDecompStage("missing_attention_core_capture");

    probe.attention_core_post_rope = if (q_rope != null and k_rope != null and v_raw != null and hidden_in != null)
        try captureLayerBackwardSubstage(
            allocator,
            cb,
            graph_config,
            actual_batch,
            max_seq,
            target_layer,
            .attention_core_post_rope,
            lora_rank,
            lora_alpha,
            upstream_grad,
            .{
                .upstream_grad = upstream_grad,
                .q_raw = q_rope.?,
                .k_raw = k_rope.?,
                .v_raw = v_raw.?,
                .hidden_in_aux = hidden_in.?,
                .attention_mask = attention_mask,
            },
            lora_layers,
        )
    else
        missingBackwardDecompStage("missing_attention_core_post_rope_capture");

    probe.attention_scores_raw = if (attention_core_tensors != null and v_raw != null and hidden_in != null)
        try captureLayerBackwardSubstage(
            allocator,
            cb,
            graph_config,
            actual_batch,
            max_seq,
            target_layer,
            .attention_scores_raw,
            lora_rank,
            lora_alpha,
            upstream_grad,
            .{
                .upstream_grad = upstream_grad,
                .primary = attention_core_tensors.?.scores_raw,
                .v_raw = v_raw.?,
                .hidden_in_aux = hidden_in.?,
                .attention_mask = attention_mask,
            },
            lora_layers,
        )
    else
        missingBackwardDecompStage("missing_attention_scores_raw_capture");

    probe.attention_scores_masked = if (attention_core_tensors != null and v_raw != null and hidden_in != null)
        try captureLayerBackwardSubstage(
            allocator,
            cb,
            graph_config,
            actual_batch,
            max_seq,
            target_layer,
            .attention_scores_masked,
            lora_rank,
            lora_alpha,
            upstream_grad,
            .{
                .upstream_grad = upstream_grad,
                .primary = attention_core_tensors.?.scores_masked,
                .v_raw = v_raw.?,
                .hidden_in_aux = hidden_in.?,
                .attention_mask = attention_mask,
            },
            lora_layers,
        )
    else
        missingBackwardDecompStage("missing_attention_scores_masked_capture");

    probe.attention_probs = if (attention_core_tensors != null and v_raw != null and hidden_in != null)
        try captureLayerBackwardSubstage(
            allocator,
            cb,
            graph_config,
            actual_batch,
            max_seq,
            target_layer,
            .attention_probs,
            lora_rank,
            lora_alpha,
            upstream_grad,
            .{
                .upstream_grad = upstream_grad,
                .primary = attention_core_tensors.?.probs,
                .v_raw = v_raw.?,
                .hidden_in_aux = hidden_in.?,
                .attention_mask = attention_mask,
            },
            lora_layers,
        )
    else
        missingBackwardDecompStage("missing_attention_probs_capture");

    probe.qkv_proj = if (attn_normed != null and hidden_in != null)
        try captureLayerBackwardSubstage(
            allocator,
            cb,
            graph_config,
            actual_batch,
            max_seq,
            target_layer,
            .qkv_proj,
            lora_rank,
            lora_alpha,
            upstream_grad,
            .{
                .upstream_grad = upstream_grad,
                .primary = attn_normed.?,
                .hidden_in_aux = hidden_in.?,
                .attention_mask = attention_mask,
            },
            lora_layers,
        )
    else
        missingBackwardDecompStage("missing_attn_normed_or_hidden_in_capture");

    probe.qkv_proj_split = if (attn_normed != null and hidden_in != null)
        try captureLayerBackwardSubstage(
            allocator,
            cb,
            graph_config,
            actual_batch,
            max_seq,
            target_layer,
            .qkv_proj_split,
            lora_rank,
            lora_alpha,
            upstream_grad,
            .{
                .upstream_grad = upstream_grad,
                .primary = attn_normed.?,
                .hidden_in_aux = hidden_in.?,
                .attention_mask = attention_mask,
            },
            lora_layers,
        )
    else
        missingBackwardDecompStage("missing_attn_normed_or_hidden_in_capture");

    probe.attn_norm_hidden_in = if (hidden_in != null)
        try captureLayerBackwardSubstage(
            allocator,
            cb,
            graph_config,
            actual_batch,
            max_seq,
            target_layer,
            .attn_norm_hidden_in,
            lora_rank,
            lora_alpha,
            upstream_grad,
            .{
                .upstream_grad = upstream_grad,
                .primary = hidden_in.?,
                .attention_mask = attention_mask,
            },
            lora_layers,
        )
    else
        missingBackwardDecompStage("missing_hidden_in_capture");

    if (!decompStageCaptured(probe.mlp_wo) or
        !decompStageCaptured(probe.mlp_gelu_input) or
        !decompStageCaptured(probe.mlp_gate_value) or
        !decompStageCaptured(probe.mlp_gate_input) or
        !decompStageCaptured(probe.mlp_wi_output) or
        !decompStageCaptured(probe.mlp_norm_output) or
        !decompStageCaptured(probe.mlp_hidden_after_attn) or
        !decompStageCaptured(probe.attn_out_proj) or
        !decompStageCaptured(probe.attention_core) or
        !decompStageCaptured(probe.attention_core_post_rope) or
        !decompStageCaptured(probe.attention_scores_raw) or
        !decompStageCaptured(probe.attention_scores_masked) or
        !decompStageCaptured(probe.attention_probs) or
        !decompStageCaptured(probe.qkv_proj) or
        !decompStageCaptured(probe.qkv_proj_split) or
        !decompStageCaptured(probe.attn_norm_hidden_in))
    {
        probe.status = "partial";
    }

    return probe;
}

const SegmentVJPDiffStats = struct {
    tensors: usize = 0,
    elems: usize = 0,
    nonzero_diffs: usize = 0,
    max_abs: f32 = 0.0,
    max_rel: f32 = 0.0,
    sum_abs: f64 = 0.0,
    sum_diff_sq: f64 = 0.0,
    sum_lhs_sq: f64 = 0.0,
    sum_rhs_sq: f64 = 0.0,
    sum_dot: f64 = 0.0,
    sign_mismatches: usize = 0,
    max_name: []const u8 = "",

    fn add(self: *SegmentVJPDiffStats, name: []const u8, lhs: []const f32, rhs: []const f32) !void {
        if (lhs.len != rhs.len) return error.SegmentVJPParityShapeMismatch;
        self.tensors += 1;
        self.elems += lhs.len;
        for (lhs, rhs) |a, b| {
            const diff = absF32(a - b);
            if (diff != 0.0) self.nonzero_diffs += 1;
            self.sum_abs += @as(f64, diff);
            self.sum_diff_sq += @as(f64, diff) * @as(f64, diff);
            self.sum_lhs_sq += @as(f64, a) * @as(f64, a);
            self.sum_rhs_sq += @as(f64, b) * @as(f64, b);
            self.sum_dot += @as(f64, a) * @as(f64, b);
            if ((a < 0.0 and b > 0.0) or (a > 0.0 and b < 0.0)) self.sign_mismatches += 1;
            if (diff > self.max_abs) {
                self.max_abs = diff;
                self.max_name = name;
            }
            const denom = @max(@max(absF32(a), absF32(b)), 1.0e-12);
            self.max_rel = @max(self.max_rel, diff / denom);
        }
    }

    fn meanAbs(self: SegmentVJPDiffStats) f64 {
        if (self.elems == 0) return 0.0;
        return self.sum_abs / @as(f64, @floatFromInt(self.elems));
    }

    fn relL2(self: SegmentVJPDiffStats) f64 {
        if (self.sum_rhs_sq <= 0.0) return if (self.sum_diff_sq == 0.0) 0.0 else std.math.inf(f64);
        return @sqrt(self.sum_diff_sq / self.sum_rhs_sq);
    }

    fn cosine(self: SegmentVJPDiffStats) f64 {
        const denom = @sqrt(self.sum_lhs_sq * self.sum_rhs_sq);
        if (denom <= 0.0) return 0.0;
        return self.sum_dot / denom;
    }
};

fn absF32(value: f32) f32 {
    return if (value < 0.0) -value else value;
}

fn tokenMajorAttentionIndex(
    b: usize,
    token: usize,
    head: usize,
    dim: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) usize {
    return ((b * seq_len + token) * num_heads + head) * head_dim + dim;
}

fn kernelMajorAttentionIndex(
    b: usize,
    token: usize,
    head: usize,
    dim: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) usize {
    return (((b * num_heads + head) * seq_len + token) * head_dim) + dim;
}

const AttentionIndexMode = enum {
    token_major,
    kernel_major,
};

fn attentionIndex(
    mode: AttentionIndexMode,
    b: usize,
    token: usize,
    head: usize,
    dim: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) usize {
    return switch (mode) {
        .token_major => tokenMajorAttentionIndex(b, token, head, dim, seq_len, num_heads, head_dim),
        .kernel_major => kernelMajorAttentionIndex(b, token, head, dim, seq_len, num_heads, head_dim),
    };
}

fn computeSdpaReference(
    allocator: std.mem.Allocator,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    attention_mask: []const i32,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    mode: AttentionIndexMode,
    is_local_attention: bool,
    local_attention_window: u32,
) ![]f32 {
    const total = batch * seq_len * num_heads * head_dim;
    if (q.len != total or k.len != total or v.len != total) return error.StepParityAttentionReferenceShapeMismatch;
    if (attention_mask.len < batch * seq_len) return error.StepParityAttentionReferenceShapeMismatch;

    const out = try allocator.alloc(f32, total);
    errdefer allocator.free(out);
    @memset(out, 0.0);
    const scores = try allocator.alloc(f32, seq_len);
    defer allocator.free(scores);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const window_half: usize = @intCast(local_attention_window / 2);

    for (0..batch) |b| {
        for (0..num_heads) |head| {
            for (0..seq_len) |qi| {
                var best = -std.math.inf(f32);
                for (0..seq_len) |ki| {
                    const diff = if (qi >= ki) qi - ki else ki - qi;
                    const outside_window = is_local_attention and diff > window_half;
                    if (attention_mask[b * seq_len + ki] == 0 or outside_window) {
                        scores[ki] = 0.0;
                        continue;
                    }
                    var score: f32 = 0.0;
                    for (0..head_dim) |dim| {
                        const q_idx = attentionIndex(mode, b, qi, head, dim, seq_len, num_heads, head_dim);
                        const k_idx = attentionIndex(mode, b, ki, head, dim, seq_len, num_heads, head_dim);
                        score += q[q_idx] * k[k_idx];
                    }
                    score *= scale;
                    scores[ki] = score;
                    best = @max(best, score);
                }

                var sum: f32 = 0.0;
                for (0..seq_len) |ki| {
                    const diff = if (qi >= ki) qi - ki else ki - qi;
                    const outside_window = is_local_attention and diff > window_half;
                    if (attention_mask[b * seq_len + ki] == 0 or outside_window) {
                        scores[ki] = 0.0;
                        continue;
                    }
                    const weight = @exp(scores[ki] - best);
                    scores[ki] = weight;
                    sum += weight;
                }
                if (sum <= 0.0) continue;

                for (0..head_dim) |dim| {
                    var accum: f32 = 0.0;
                    for (0..seq_len) |ki| {
                        const diff = if (qi >= ki) qi - ki else ki - qi;
                        const outside_window = is_local_attention and diff > window_half;
                        if (outside_window) continue;
                        if (scores[ki] == 0.0) continue;
                        const v_idx = attentionIndex(mode, b, ki, head, dim, seq_len, num_heads, head_dim);
                        accum += (scores[ki] / sum) * v[v_idx];
                    }
                    const out_idx = attentionIndex(mode, b, qi, head, dim, seq_len, num_heads, head_dim);
                    out[out_idx] = accum;
                }
            }
        }
    }
    return out;
}

const StepParityAttentionCoreTensors = struct {
    scores_raw: []f32,
    scores_masked: []f32,
    probs: []f32,

    fn deinit(self: *StepParityAttentionCoreTensors, allocator: std.mem.Allocator) void {
        allocator.free(self.scores_raw);
        allocator.free(self.scores_masked);
        allocator.free(self.probs);
        self.* = undefined;
    }
};

fn attentionScoreIndex(
    b: usize,
    head: usize,
    query: usize,
    key: usize,
    seq_len: usize,
    num_heads: usize,
) usize {
    return (((b * num_heads + head) * seq_len + query) * seq_len) + key;
}

fn computeStepParityAttentionCoreTensors(
    allocator: std.mem.Allocator,
    q: []const f32,
    k: []const f32,
    attention_mask: []const f32,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    is_local_attention: bool,
    local_attention_window: u32,
) !StepParityAttentionCoreTensors {
    const hidden_total = batch * seq_len * num_heads * head_dim;
    if (q.len != hidden_total or k.len != hidden_total) return error.StepParityAttentionReferenceShapeMismatch;
    if (attention_mask.len != batch * seq_len) return error.StepParityAttentionReferenceShapeMismatch;

    const scores_total = batch * num_heads * seq_len * seq_len;
    const scores_raw = try allocator.alloc(f32, scores_total);
    errdefer allocator.free(scores_raw);
    const scores_masked = try allocator.alloc(f32, scores_total);
    errdefer allocator.free(scores_masked);
    const probs = try allocator.alloc(f32, scores_total);
    errdefer allocator.free(probs);
    @memset(probs, 0.0);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const window_half: usize = @intCast(local_attention_window / 2);

    for (0..batch) |b| {
        for (0..num_heads) |head| {
            for (0..seq_len) |query| {
                var best = -std.math.inf(f32);
                for (0..seq_len) |key| {
                    var raw: f32 = 0.0;
                    for (0..head_dim) |dim| {
                        const q_idx = attentionIndex(.token_major, b, query, head, dim, seq_len, num_heads, head_dim);
                        const k_idx = attentionIndex(.token_major, b, key, head, dim, seq_len, num_heads, head_dim);
                        raw += q[q_idx] * k[k_idx];
                    }

                    const diff = if (query >= key) query - key else key - query;
                    const outside_window = is_local_attention and diff > window_half;
                    var masked_bias: f32 = 0.0;
                    if (attention_mask[b * seq_len + key] < 0.5) masked_bias -= 10000.0;
                    if (outside_window) masked_bias -= 10000.0;
                    const score_idx = attentionScoreIndex(b, head, query, key, seq_len, num_heads);
                    scores_raw[score_idx] = raw;
                    scores_masked[score_idx] = raw * scale + masked_bias;
                    best = @max(best, scores_masked[score_idx]);
                }

                var sum: f32 = 0.0;
                for (0..seq_len) |key| {
                    const score_idx = attentionScoreIndex(b, head, query, key, seq_len, num_heads);
                    const weight = @exp(scores_masked[score_idx] - best);
                    probs[score_idx] = weight;
                    sum += weight;
                }
                if (sum <= 0.0) {
                    for (0..seq_len) |key| {
                        probs[attentionScoreIndex(b, head, query, key, seq_len, num_heads)] = 0.0;
                    }
                    continue;
                }
                const inv_sum = 1.0 / sum;
                for (0..seq_len) |key| {
                    const score_idx = attentionScoreIndex(b, head, query, key, seq_len, num_heads);
                    probs[score_idx] *= inv_sum;
                }
            }
        }
    }

    return .{
        .scores_raw = scores_raw,
        .scores_masked = scores_masked,
        .probs = probs,
    };
}

fn deterministicSoftmaxScore(outer: usize, query: usize, key: usize) f32 {
    const mixed = (outer * 37 + query * 17 + key * 29 + 11) % 127;
    const centered: i32 = @as(i32, @intCast(mixed)) - 63;
    return @as(f32, @floatFromInt(centered)) * 0.03125 + @as(f32, @floatFromInt((query + key) % 7)) * 0.002;
}

fn deterministicSoftmaxUpstream(outer: usize, query: usize, key: usize) f32 {
    const mixed = (outer * 19 + query * 23 + key * 13 + 5) % 89;
    const centered: i32 = @as(i32, @intCast(mixed)) - 44;
    return @as(f32, @floatFromInt(centered)) * 0.00025;
}

fn deterministicSoftmaxMaskedKey(
    query: usize,
    key: usize,
    queries: usize,
    keys: usize,
    has_mask: bool,
    local_attention_window: u32,
) bool {
    if (!has_mask) return false;
    const tail_mask_start = keys - @max(@as(usize, 1), keys / 16);
    if (key >= tail_mask_start) return true;
    if (queries == keys and local_attention_window > 0) {
        const window_half: usize = @intCast(local_attention_window / 2);
        const diff = if (query >= key) query - key else key - query;
        if (diff > window_half) return true;
    }
    return false;
}

fn fillDeterministicSoftmaxVJPTensors(
    scores: []f32,
    upstream: []f32,
    outer: usize,
    queries: usize,
    keys: usize,
    has_mask: bool,
    mask_bias: f32,
    local_attention_window: u32,
) void {
    var idx: usize = 0;
    for (0..outer) |outer_idx| {
        for (0..queries) |query| {
            for (0..keys) |key| {
                scores[idx] = deterministicSoftmaxScore(outer_idx, query, key);
                if (deterministicSoftmaxMaskedKey(query, key, queries, keys, has_mask, local_attention_window)) {
                    scores[idx] += mask_bias;
                }
                upstream[idx] = deterministicSoftmaxUpstream(outer_idx, query, key);
                idx += 1;
            }
        }
    }
}

fn buildSoftmaxVJPCPUReferenceStats(
    allocator: std.mem.Allocator,
    case: *StepParitySoftmaxVJPCase,
    scores: []const f32,
    upstream: []const f32,
    mps_grad: []const f32,
    local_attention_window: u32,
) !void {
    const total = case.outer * case.queries * case.keys;
    if (scores.len != total or upstream.len != total or mps_grad.len != total) return error.InvalidSegmentTensorShape;
    const probs = try allocator.alloc(f32, case.keys);
    defer allocator.free(probs);

    var row_start: usize = 0;
    for (0..case.outer) |outer_idx| {
        _ = outer_idx;
        for (0..case.queries) |query| {
            var best = -std.math.inf(f32);
            for (0..case.keys) |key| {
                best = @max(best, scores[row_start + key]);
            }

            var sum: f32 = 0.0;
            for (0..case.keys) |key| {
                const prob = @exp(scores[row_start + key] - best);
                probs[key] = prob;
                sum += prob;
            }
            if (sum <= 0.0) {
                @memset(probs, 0.0);
            } else {
                const inv_sum = 1.0 / sum;
                for (probs) |*prob| prob.* *= inv_sum;
            }

            var dot: f32 = 0.0;
            for (0..case.keys) |key| {
                const prob = probs[key];
                case.probs.addValue(prob);
                dot += upstream[row_start + key] * prob;
            }

            for (0..case.keys) |key| {
                const idx = row_start + key;
                const cpu_grad = probs[key] * (upstream[idx] - dot);
                const err = absF32(mps_grad[idx] - cpu_grad);
                case.cpu_scores_masked_grad.addValue(cpu_grad);
                case.cpu_abs_error.addValue(err);
                if (deterministicSoftmaxMaskedKey(query, key, case.queries, case.keys, case.has_mask, local_attention_window)) {
                    case.masked_scores_masked_grad.addValue(mps_grad[idx]);
                } else {
                    case.valid_scores_masked_grad.addValue(mps_grad[idx]);
                }
            }
            row_start += case.keys;
        }
    }
}

fn captureStepParitySoftmaxVJPCase(
    allocator: std.mem.Allocator,
    cb: *const ComputeBackend,
    name: []const u8,
    outer: usize,
    queries: usize,
    keys: usize,
    has_mask: bool,
    mask_bias: f32,
    local_attention_window: u32,
) !StepParitySoftmaxVJPCase {
    const total = outer * queries * keys;
    const scores = try allocator.alloc(f32, total);
    defer allocator.free(scores);
    const upstream = try allocator.alloc(f32, total);
    defer allocator.free(upstream);
    fillDeterministicSoftmaxVJPTensors(scores, upstream, outer, queries, keys, has_mask, mask_bias, local_attention_window);

    var graph = ml.graph.Graph.init(allocator);
    defer graph.deinit();
    var bld = ml.graph.Builder.init(&graph);
    const scores_node = try bld.parameter("scores_masked", ml.graph.Shape.init(.f32, &.{
        @as(i64, @intCast(outer)),
        @as(i64, @intCast(queries)),
        @as(i64, @intCast(keys)),
    }));
    const upstream_node = try bld.parameter("upstream_probs_grad", ml.graph.Shape.init(.f32, &.{
        @as(i64, @intCast(outer)),
        @as(i64, @intCast(queries)),
        @as(i64, @intCast(keys)),
    }));
    const probs_node = try bld.softmax(scores_node);
    const weighted = try bld.mul(probs_node, upstream_node);
    const loss = try bld.reduceSum(weighted, &.{ 0, 1, 2 });
    try graph.markOutput(loss);

    const trainable = [_][]const u8{"scores_masked"};
    var session = try graph_training.CompiledTrainSession.init(
        allocator,
        &graph,
        loss,
        .{
            .trainable_params = trainable[0..],
            .execution_strategy = .mpsgraph_required,
        },
    );
    defer session.deinit();

    const dims = [_]i32{
        @intCast(outer),
        @intCast(queries),
        @intCast(keys),
    };
    const scores_ct = try cb.fromFloat32Shape(scores, &dims);
    defer cb.free(scores_ct);
    const upstream_ct = try cb.fromFloat32Shape(upstream, &dims);
    defer cb.free(upstream_ct);

    var runtime_inputs = std.AutoHashMapUnmanaged(ml.graph.NodeId, ops_mod.CT){};
    defer runtime_inputs.deinit(allocator);
    try runtime_inputs.put(allocator, scores_node, scores_ct);
    try runtime_inputs.put(allocator, upstream_node, upstream_ct);

    var result = try session.execute(cb, runtime_inputs);
    defer result.deinit();
    const grad = result.gradients.get("scores_masked") orelse return error.MissingSubstageGradient;
    if (grad.len != total) return error.InvalidSegmentTensorShape;

    var case = StepParitySoftmaxVJPCase{
        .name = name,
        .outer = outer,
        .queries = queries,
        .keys = keys,
        .mask_bias = if (has_mask) mask_bias else 0.0,
        .has_mask = has_mask,
    };
    case.scores_masked.addSlice(scores);
    case.upstream_probs_grad.addSlice(upstream);
    case.scores_masked_grad.addSlice(grad);
    try buildSoftmaxVJPCPUReferenceStats(
        allocator,
        &case,
        scores,
        upstream,
        grad,
        local_attention_window,
    );
    return case;
}

fn captureStepParitySoftmaxVJPProbe(
    allocator: std.mem.Allocator,
    cb: *const ComputeBackend,
    actual_batch: usize,
    seq_len: usize,
    num_heads: usize,
    local_attention_window: u32,
) !StepParitySoftmaxVJPProbe {
    const CaseConfig = struct {
        name: []const u8,
        outer: usize,
        queries: usize,
        keys: usize,
        has_mask: bool,
        mask_bias: f32,
    };
    const layer_outer = actual_batch * num_heads;
    const configs = [_]CaseConfig{
        .{
            .name = "synthetic_small_nomask",
            .outer = 1,
            .queries = 8,
            .keys = 16,
            .has_mask = false,
            .mask_bias = 0.0,
        },
        .{
            .name = "layer14_shape_mask_neg1e9",
            .outer = layer_outer,
            .queries = seq_len,
            .keys = seq_len,
            .has_mask = true,
            .mask_bias = -1.0e9,
        },
        .{
            .name = "layer14_shape_mask_neg10000",
            .outer = layer_outer,
            .queries = seq_len,
            .keys = seq_len,
            .has_mask = true,
            .mask_bias = -10000.0,
        },
    };

    var cases = try allocator.alloc(StepParitySoftmaxVJPCase, configs.len);
    errdefer allocator.free(cases);
    for (configs, 0..) |cfg, i| {
        cases[i] = try captureStepParitySoftmaxVJPCase(
            allocator,
            cb,
            cfg.name,
            cfg.outer,
            cfg.queries,
            cfg.keys,
            cfg.has_mask,
            cfg.mask_bias,
            local_attention_window,
        );
    }

    return .{
        .status = "captured",
        .version = 1,
        .runtime = "mpsgraph",
        .cases = cases,
    };
}

fn deterministicQKVSplitValue(tag: usize, a: usize, b: usize, c: usize) f32 {
    const mixed = (tag * 1009 + a * 131 + b * 37 + c * 17) % 2003;
    return (@as(f32, @floatFromInt(mixed)) - 1001.0) * 0.00025;
}

fn fillDeterministicFlatQKVTensor(values: []f32, rows: usize, cols: usize, tag: usize) void {
    var idx: usize = 0;
    for (0..rows) |row| {
        for (0..cols) |col| {
            values[idx] = deterministicQKVSplitValue(tag, row, col, 0);
            idx += 1;
        }
    }
}

fn fillDeterministicBhsdTensor(values: []f32, outer: usize, seq_len: usize, head_dim: usize, tag: usize) void {
    var idx: usize = 0;
    for (0..outer) |outer_idx| {
        for (0..seq_len) |pos| {
            for (0..head_dim) |dim| {
                values[idx] = deterministicQKVSplitValue(tag, outer_idx, pos, dim);
                idx += 1;
            }
        }
    }
}

fn fillDeterministicScoreTensor(values: []f32, outer: usize, seq_len: usize, tag: usize) void {
    var idx: usize = 0;
    for (0..outer) |outer_idx| {
        for (0..seq_len) |query| {
            for (0..seq_len) |key| {
                values[idx] = deterministicQKVSplitValue(tag, outer_idx, query, key);
                idx += 1;
            }
        }
    }
}

fn fillDeterministicAttentionProbTensor(values: []f32, outer: usize, seq_len: usize, tag: usize) void {
    var idx: usize = 0;
    for (0..outer) |outer_idx| {
        for (0..seq_len) |query| {
            var row_sum: f32 = 0.0;
            for (0..seq_len) |key| {
                const raw = 0.1 + @abs(deterministicQKVSplitValue(tag, outer_idx, query, key));
                values[idx + key] = raw;
                row_sum += raw;
            }
            const inv_sum = if (row_sum > 0.0) 1.0 / row_sum else 0.0;
            for (0..seq_len) |key| values[idx + key] *= inv_sum;
            idx += seq_len;
        }
    }
}

fn stepParitySplitHeads(
    bld: *ml.graph.Builder,
    flat: ml.graph.NodeId,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) !ml.graph.NodeId {
    const bsnh = try bld.reshape(
        flat,
        ml.graph.Shape.init(.f32, &.{
            @as(i64, @intCast(batch)),
            @as(i64, @intCast(seq_len)),
            @as(i64, @intCast(num_heads)),
            @as(i64, @intCast(head_dim)),
        }),
    );
    const bnsh = try bld.transpose(bsnh, &.{ 0, 2, 1, 3 });
    return bld.reshape(
        bnsh,
        ml.graph.Shape.init(.f32, &.{
            @as(i64, @intCast(batch * num_heads)),
            @as(i64, @intCast(seq_len)),
            @as(i64, @intCast(head_dim)),
        }),
    );
}

fn stepParityMergeHeads(
    bld: *ml.graph.Builder,
    bhsd: ml.graph.NodeId,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) !ml.graph.NodeId {
    const bnsh = try bld.reshape(
        bhsd,
        ml.graph.Shape.init(.f32, &.{
            @as(i64, @intCast(batch)),
            @as(i64, @intCast(num_heads)),
            @as(i64, @intCast(seq_len)),
            @as(i64, @intCast(head_dim)),
        }),
    );
    const bsnh = try bld.transpose(bnsh, &.{ 0, 2, 1, 3 });
    return bld.reshape(
        bsnh,
        ml.graph.Shape.init(.f32, &.{
            @as(i64, @intCast(batch * seq_len)),
            @as(i64, @intCast(num_heads * head_dim)),
        }),
    );
}

fn stepParityNamedStats(
    allocator: std.mem.Allocator,
    name: []const u8,
    values: []const f32,
) !StepParityNamedTensorStats {
    var stats = StepParityTensorSliceStats{};
    stats.addSlice(values);
    return .{
        .name = try allocator.dupe(u8, name),
        .stats = stats,
    };
}

fn stepParityNamedStatsFromStats(
    allocator: std.mem.Allocator,
    name: []const u8,
    stats: StepParityTensorSliceStats,
) !StepParityNamedTensorStats {
    return .{
        .name = try allocator.dupe(u8, name),
        .stats = stats,
    };
}

fn stepParityAbsErrorStats(lhs: []const f32, rhs: []const f32) !StepParityTensorSliceStats {
    if (lhs.len != rhs.len) return error.StepParityAttentionReferenceShapeMismatch;
    var stats = StepParityTensorSliceStats{};
    for (lhs, rhs) |a, b| stats.addValue(absF32(a - b));
    return stats;
}

fn captureQKVSplitHeadsVJPCase(
    allocator: std.mem.Allocator,
    cb: *const ComputeBackend,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) !StepParityQKVSplitVJPCase {
    const hidden_size = num_heads * head_dim;
    const total = batch * seq_len;
    const flat_elems = total * hidden_size;
    const bhsd_elems = batch * num_heads * seq_len * head_dim;
    const flat = try allocator.alloc(f32, flat_elems);
    defer allocator.free(flat);
    const upstream = try allocator.alloc(f32, bhsd_elems);
    defer allocator.free(upstream);
    const cpu_grad = try allocator.alloc(f32, flat_elems);
    defer allocator.free(cpu_grad);
    fillDeterministicFlatQKVTensor(flat, total, hidden_size, 1);
    fillDeterministicBhsdTensor(upstream, batch * num_heads, seq_len, head_dim, 2);

    @memset(cpu_grad, 0.0);
    for (0..batch) |b| {
        for (0..num_heads) |h| {
            for (0..seq_len) |s| {
                for (0..head_dim) |d| {
                    const out_idx = (((b * num_heads + h) * seq_len + s) * head_dim) + d;
                    const flat_idx = ((b * seq_len + s) * hidden_size) + h * head_dim + d;
                    cpu_grad[flat_idx] += upstream[out_idx];
                }
            }
        }
    }

    var graph = ml.graph.Graph.init(allocator);
    defer graph.deinit();
    var bld = ml.graph.Builder.init(&graph);
    const flat_node = try bld.parameter("flat", ml.graph.Shape.init(.f32, &.{
        @as(i64, @intCast(total)),
        @as(i64, @intCast(hidden_size)),
    }));
    const upstream_node = try bld.parameter("upstream", ml.graph.Shape.init(.f32, &.{
        @as(i64, @intCast(batch * num_heads)),
        @as(i64, @intCast(seq_len)),
        @as(i64, @intCast(head_dim)),
    }));
    const split = try stepParitySplitHeads(&bld, flat_node, batch, seq_len, num_heads, head_dim);
    const weighted = try bld.mul(split, upstream_node);
    const loss = try bld.reduceSum(weighted, &.{ 0, 1, 2 });
    try graph.markOutput(loss);

    const trainable = [_][]const u8{"flat"};
    var session = try graph_training.CompiledTrainSession.init(
        allocator,
        &graph,
        loss,
        .{
            .trainable_params = trainable[0..],
            .execution_strategy = .mpsgraph_required,
        },
    );
    defer session.deinit();

    const flat_dims = [_]i32{ @intCast(total), @intCast(hidden_size) };
    const upstream_dims = [_]i32{ @intCast(batch * num_heads), @intCast(seq_len), @intCast(head_dim) };
    const flat_ct = try cb.fromFloat32Shape(flat, &flat_dims);
    defer cb.free(flat_ct);
    const upstream_ct = try cb.fromFloat32Shape(upstream, &upstream_dims);
    defer cb.free(upstream_ct);

    var runtime_inputs = std.AutoHashMapUnmanaged(ml.graph.NodeId, ops_mod.CT){};
    defer runtime_inputs.deinit(allocator);
    try runtime_inputs.put(allocator, flat_node, flat_ct);
    try runtime_inputs.put(allocator, upstream_node, upstream_ct);

    var result = try session.execute(cb, runtime_inputs);
    defer result.deinit();
    const grad = result.gradients.get("flat") orelse return error.MissingSubstageGradient;
    if (grad.len != flat_elems) return error.InvalidSegmentTensorShape;
    const err_stats = try stepParityAbsErrorStats(grad, cpu_grad);
    var components = try allocator.alloc(StepParityNamedTensorStats, 4);
    errdefer allocator.free(components);
    components[0] = try stepParityNamedStats(allocator, "upstream", upstream);
    components[1] = try stepParityNamedStats(allocator, "flat_grad", grad);
    components[2] = try stepParityNamedStats(allocator, "cpu_flat_grad", cpu_grad);
    components[3] = try stepParityNamedStatsFromStats(allocator, "flat_grad_cpu_abs_error", err_stats);
    return .{
        .name = "split_heads_vjp",
        .batch = batch,
        .seq_len = seq_len,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .hidden_size = hidden_size,
        .outer = batch * num_heads,
        .components = components,
    };
}

fn captureQKVScoreMatmulVJPCase(
    allocator: std.mem.Allocator,
    cb: *const ComputeBackend,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) !StepParityQKVSplitVJPCase {
    const outer = batch * num_heads;
    const bhsd_elems = outer * seq_len * head_dim;
    const score_elems = outer * seq_len * seq_len;
    const q = try allocator.alloc(f32, bhsd_elems);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, bhsd_elems);
    defer allocator.free(k);
    const upstream = try allocator.alloc(f32, score_elems);
    defer allocator.free(upstream);
    const cpu_q_grad = try allocator.alloc(f32, bhsd_elems);
    defer allocator.free(cpu_q_grad);
    const cpu_k_grad = try allocator.alloc(f32, bhsd_elems);
    defer allocator.free(cpu_k_grad);
    fillDeterministicBhsdTensor(q, outer, seq_len, head_dim, 3);
    fillDeterministicBhsdTensor(k, outer, seq_len, head_dim, 4);
    fillDeterministicScoreTensor(upstream, outer, seq_len, 5);
    @memset(cpu_q_grad, 0.0);
    @memset(cpu_k_grad, 0.0);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    for (0..outer) |o| {
        for (0..seq_len) |query| {
            for (0..seq_len) |key| {
                const upstream_idx = (o * seq_len + query) * seq_len + key;
                const up = upstream[upstream_idx] * scale;
                for (0..head_dim) |dim| {
                    const q_idx = (o * seq_len + query) * head_dim + dim;
                    const k_idx = (o * seq_len + key) * head_dim + dim;
                    cpu_q_grad[q_idx] += up * k[k_idx];
                    cpu_k_grad[k_idx] += up * q[q_idx];
                }
            }
        }
    }

    var graph = ml.graph.Graph.init(allocator);
    defer graph.deinit();
    var bld = ml.graph.Builder.init(&graph);
    const q_node = try bld.parameter("q", ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(outer)), @as(i64, @intCast(seq_len)), @as(i64, @intCast(head_dim)) }));
    const k_node = try bld.parameter("k", ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(outer)), @as(i64, @intCast(seq_len)), @as(i64, @intCast(head_dim)) }));
    const upstream_node = try bld.parameter("upstream", ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(outer)), @as(i64, @intCast(seq_len)), @as(i64, @intCast(seq_len)) }));
    const k_t = try bld.transpose(k_node, &.{ 0, 2, 1 });
    const scores_raw = try bld.matmul3D(q_node, k_t);
    const scale_node = try bld.scalarConst(.f32, scale);
    const scores = try bld.mul(scores_raw, scale_node);
    const weighted = try bld.mul(scores, upstream_node);
    const loss = try bld.reduceSum(weighted, &.{ 0, 1, 2 });
    try graph.markOutput(loss);

    const trainable = [_][]const u8{ "q", "k" };
    var session = try graph_training.CompiledTrainSession.init(allocator, &graph, loss, .{ .trainable_params = trainable[0..], .execution_strategy = .mpsgraph_required });
    defer session.deinit();

    const bhsd_dims = [_]i32{ @intCast(outer), @intCast(seq_len), @intCast(head_dim) };
    const score_dims = [_]i32{ @intCast(outer), @intCast(seq_len), @intCast(seq_len) };
    const q_ct = try cb.fromFloat32Shape(q, &bhsd_dims);
    defer cb.free(q_ct);
    const k_ct = try cb.fromFloat32Shape(k, &bhsd_dims);
    defer cb.free(k_ct);
    const upstream_ct = try cb.fromFloat32Shape(upstream, &score_dims);
    defer cb.free(upstream_ct);

    var runtime_inputs = std.AutoHashMapUnmanaged(ml.graph.NodeId, ops_mod.CT){};
    defer runtime_inputs.deinit(allocator);
    try runtime_inputs.put(allocator, q_node, q_ct);
    try runtime_inputs.put(allocator, k_node, k_ct);
    try runtime_inputs.put(allocator, upstream_node, upstream_ct);

    var result = try session.execute(cb, runtime_inputs);
    defer result.deinit();
    const q_grad = result.gradients.get("q") orelse return error.MissingSubstageGradient;
    const k_grad = result.gradients.get("k") orelse return error.MissingSubstageGradient;
    const q_err = try stepParityAbsErrorStats(q_grad, cpu_q_grad);
    const k_err = try stepParityAbsErrorStats(k_grad, cpu_k_grad);
    var components = try allocator.alloc(StepParityNamedTensorStats, 7);
    errdefer allocator.free(components);
    components[0] = try stepParityNamedStats(allocator, "upstream_scores_grad", upstream);
    components[1] = try stepParityNamedStats(allocator, "q_grad", q_grad);
    components[2] = try stepParityNamedStats(allocator, "k_grad", k_grad);
    components[3] = try stepParityNamedStats(allocator, "cpu_q_grad", cpu_q_grad);
    components[4] = try stepParityNamedStats(allocator, "cpu_k_grad", cpu_k_grad);
    components[5] = try stepParityNamedStatsFromStats(allocator, "q_grad_cpu_abs_error", q_err);
    components[6] = try stepParityNamedStatsFromStats(allocator, "k_grad_cpu_abs_error", k_err);
    return .{
        .name = "score_matmul_vjp",
        .batch = batch,
        .seq_len = seq_len,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .hidden_size = num_heads * head_dim,
        .outer = outer,
        .components = components,
    };
}

fn captureQKVValueContextVJPCase(
    allocator: std.mem.Allocator,
    cb: *const ComputeBackend,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) !StepParityQKVSplitVJPCase {
    const outer = batch * num_heads;
    const score_elems = outer * seq_len * seq_len;
    const bhsd_elems = outer * seq_len * head_dim;
    const probs = try allocator.alloc(f32, score_elems);
    defer allocator.free(probs);
    const v = try allocator.alloc(f32, bhsd_elems);
    defer allocator.free(v);
    const upstream = try allocator.alloc(f32, bhsd_elems);
    defer allocator.free(upstream);
    const cpu_probs_grad = try allocator.alloc(f32, score_elems);
    defer allocator.free(cpu_probs_grad);
    const cpu_v_grad = try allocator.alloc(f32, bhsd_elems);
    defer allocator.free(cpu_v_grad);
    fillDeterministicAttentionProbTensor(probs, outer, seq_len, 6);
    fillDeterministicBhsdTensor(v, outer, seq_len, head_dim, 7);
    fillDeterministicBhsdTensor(upstream, outer, seq_len, head_dim, 8);
    @memset(cpu_probs_grad, 0.0);
    @memset(cpu_v_grad, 0.0);
    for (0..outer) |o| {
        for (0..seq_len) |query| {
            for (0..seq_len) |key| {
                var sum: f32 = 0.0;
                for (0..head_dim) |dim| {
                    const upstream_idx = (o * seq_len + query) * head_dim + dim;
                    const v_idx = (o * seq_len + key) * head_dim + dim;
                    sum += upstream[upstream_idx] * v[v_idx];
                    cpu_v_grad[v_idx] += probs[(o * seq_len + query) * seq_len + key] * upstream[upstream_idx];
                }
                cpu_probs_grad[(o * seq_len + query) * seq_len + key] = sum;
            }
        }
    }

    var graph = ml.graph.Graph.init(allocator);
    defer graph.deinit();
    var bld = ml.graph.Builder.init(&graph);
    const probs_node = try bld.parameter("probs", ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(outer)), @as(i64, @intCast(seq_len)), @as(i64, @intCast(seq_len)) }));
    const v_node = try bld.parameter("v", ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(outer)), @as(i64, @intCast(seq_len)), @as(i64, @intCast(head_dim)) }));
    const upstream_node = try bld.parameter("upstream", ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(outer)), @as(i64, @intCast(seq_len)), @as(i64, @intCast(head_dim)) }));
    const context = try bld.matmul3D(probs_node, v_node);
    const weighted = try bld.mul(context, upstream_node);
    const loss = try bld.reduceSum(weighted, &.{ 0, 1, 2 });
    try graph.markOutput(loss);

    const trainable = [_][]const u8{ "probs", "v" };
    var session = try graph_training.CompiledTrainSession.init(allocator, &graph, loss, .{ .trainable_params = trainable[0..], .execution_strategy = .mpsgraph_required });
    defer session.deinit();
    const score_dims = [_]i32{ @intCast(outer), @intCast(seq_len), @intCast(seq_len) };
    const bhsd_dims = [_]i32{ @intCast(outer), @intCast(seq_len), @intCast(head_dim) };
    const probs_ct = try cb.fromFloat32Shape(probs, &score_dims);
    defer cb.free(probs_ct);
    const v_ct = try cb.fromFloat32Shape(v, &bhsd_dims);
    defer cb.free(v_ct);
    const upstream_ct = try cb.fromFloat32Shape(upstream, &bhsd_dims);
    defer cb.free(upstream_ct);

    var runtime_inputs = std.AutoHashMapUnmanaged(ml.graph.NodeId, ops_mod.CT){};
    defer runtime_inputs.deinit(allocator);
    try runtime_inputs.put(allocator, probs_node, probs_ct);
    try runtime_inputs.put(allocator, v_node, v_ct);
    try runtime_inputs.put(allocator, upstream_node, upstream_ct);

    var result = try session.execute(cb, runtime_inputs);
    defer result.deinit();
    const probs_grad = result.gradients.get("probs") orelse return error.MissingSubstageGradient;
    const v_grad = result.gradients.get("v") orelse return error.MissingSubstageGradient;
    const probs_err = try stepParityAbsErrorStats(probs_grad, cpu_probs_grad);
    const v_err = try stepParityAbsErrorStats(v_grad, cpu_v_grad);
    var components = try allocator.alloc(StepParityNamedTensorStats, 7);
    errdefer allocator.free(components);
    components[0] = try stepParityNamedStats(allocator, "upstream_context_grad", upstream);
    components[1] = try stepParityNamedStats(allocator, "probs_grad", probs_grad);
    components[2] = try stepParityNamedStats(allocator, "v_grad", v_grad);
    components[3] = try stepParityNamedStats(allocator, "cpu_probs_grad", cpu_probs_grad);
    components[4] = try stepParityNamedStats(allocator, "cpu_v_grad", cpu_v_grad);
    components[5] = try stepParityNamedStatsFromStats(allocator, "probs_grad_cpu_abs_error", probs_err);
    components[6] = try stepParityNamedStatsFromStats(allocator, "v_grad_cpu_abs_error", v_err);
    return .{
        .name = "value_context_vjp",
        .batch = batch,
        .seq_len = seq_len,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .hidden_size = num_heads * head_dim,
        .outer = outer,
        .components = components,
    };
}

fn captureQKVRopeVJPCase(
    allocator: std.mem.Allocator,
    cb: *const ComputeBackend,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rope_theta: f32,
) !StepParityQKVSplitVJPCase {
    const outer = batch * num_heads;
    const bhsd_elems = outer * seq_len * head_dim;
    const q = try allocator.alloc(f32, bhsd_elems);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, bhsd_elems);
    defer allocator.free(k);
    const upstream_q = try allocator.alloc(f32, bhsd_elems);
    defer allocator.free(upstream_q);
    const upstream_k = try allocator.alloc(f32, bhsd_elems);
    defer allocator.free(upstream_k);
    fillDeterministicBhsdTensor(q, outer, seq_len, head_dim, 11);
    fillDeterministicBhsdTensor(k, outer, seq_len, head_dim, 12);
    fillDeterministicBhsdTensor(upstream_q, outer, seq_len, head_dim, 13);
    fillDeterministicBhsdTensor(upstream_k, outer, seq_len, head_dim, 14);

    const rope = try graph_input_binder.QwenPlaceholderPrep.buildRopeCosSin(
        allocator,
        @intCast(seq_len),
        @intCast(head_dim),
        rope_theta,
    );
    defer {
        allocator.free(rope.cos);
        allocator.free(rope.sin);
    }

    var graph = ml.graph.Graph.init(allocator);
    defer graph.deinit();
    var bld = ml.graph.Builder.init(&graph);
    const bhsd_shape = ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(outer)), @as(i64, @intCast(seq_len)), @as(i64, @intCast(head_dim)) });
    const rope_shape = ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(seq_len)), @as(i64, @intCast(head_dim)) });
    const q_node = try bld.parameter("q", bhsd_shape);
    const k_node = try bld.parameter("k", bhsd_shape);
    const upstream_q_node = try bld.parameter("upstream_q", bhsd_shape);
    const upstream_k_node = try bld.parameter("upstream_k", bhsd_shape);
    const rope_cos_node = try bld.parameter("rope_cos", rope_shape);
    const rope_sin_node = try bld.parameter("rope_sin", rope_shape);
    const q_roped = try bld.ropeWithOptions(
        q_node,
        rope_cos_node,
        rope_sin_node,
        @intCast(seq_len),
        @intCast(head_dim),
        @intCast(head_dim),
        rope_theta,
        true,
    );
    const k_roped = try bld.ropeWithOptions(
        k_node,
        rope_cos_node,
        rope_sin_node,
        @intCast(seq_len),
        @intCast(head_dim),
        @intCast(head_dim),
        rope_theta,
        true,
    );
    const q_weighted = try bld.mul(q_roped, upstream_q_node);
    const k_weighted = try bld.mul(k_roped, upstream_k_node);
    const weighted = try bld.add(q_weighted, k_weighted);
    const loss = try bld.reduceSum(weighted, &.{ 0, 1, 2 });
    try graph.markOutput(loss);

    const trainable = [_][]const u8{ "q", "k" };
    var session = try graph_training.CompiledTrainSession.init(allocator, &graph, loss, .{ .trainable_params = trainable[0..], .execution_strategy = .mpsgraph_required });
    defer session.deinit();

    const bhsd_dims = [_]i32{ @intCast(outer), @intCast(seq_len), @intCast(head_dim) };
    const rope_dims = [_]i32{ @intCast(seq_len), @intCast(head_dim) };
    const q_ct = try cb.fromFloat32Shape(q, &bhsd_dims);
    defer cb.free(q_ct);
    const k_ct = try cb.fromFloat32Shape(k, &bhsd_dims);
    defer cb.free(k_ct);
    const upstream_q_ct = try cb.fromFloat32Shape(upstream_q, &bhsd_dims);
    defer cb.free(upstream_q_ct);
    const upstream_k_ct = try cb.fromFloat32Shape(upstream_k, &bhsd_dims);
    defer cb.free(upstream_k_ct);
    const rope_cos_ct = try cb.fromFloat32Shape(rope.cos, &rope_dims);
    defer cb.free(rope_cos_ct);
    const rope_sin_ct = try cb.fromFloat32Shape(rope.sin, &rope_dims);
    defer cb.free(rope_sin_ct);

    var runtime_inputs = std.AutoHashMapUnmanaged(ml.graph.NodeId, ops_mod.CT){};
    defer runtime_inputs.deinit(allocator);
    try runtime_inputs.put(allocator, q_node, q_ct);
    try runtime_inputs.put(allocator, k_node, k_ct);
    try runtime_inputs.put(allocator, upstream_q_node, upstream_q_ct);
    try runtime_inputs.put(allocator, upstream_k_node, upstream_k_ct);
    try runtime_inputs.put(allocator, rope_cos_node, rope_cos_ct);
    try runtime_inputs.put(allocator, rope_sin_node, rope_sin_ct);

    var result = try session.execute(cb, runtime_inputs);
    defer result.deinit();
    const q_grad = result.gradients.get("q") orelse return error.MissingSubstageGradient;
    const k_grad = result.gradients.get("k") orelse return error.MissingSubstageGradient;
    var components = try allocator.alloc(StepParityNamedTensorStats, 4);
    errdefer allocator.free(components);
    components[0] = try stepParityNamedStats(allocator, "upstream_q_grad", upstream_q);
    components[1] = try stepParityNamedStats(allocator, "upstream_k_grad", upstream_k);
    components[2] = try stepParityNamedStats(allocator, "q_grad", q_grad);
    components[3] = try stepParityNamedStats(allocator, "k_grad", k_grad);
    return .{
        .name = "rope_qk_vjp",
        .batch = batch,
        .seq_len = seq_len,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .hidden_size = num_heads * head_dim,
        .outer = outer,
        .components = components,
    };
}

fn captureQKVSumConsistencyCase(
    allocator: std.mem.Allocator,
    cb: *const ComputeBackend,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) !StepParityQKVSplitVJPCase {
    const hidden_size = num_heads * head_dim;
    const total = batch * seq_len;
    const flat_elems = total * hidden_size;
    const x = try allocator.alloc(f32, flat_elems);
    defer allocator.free(x);
    const upstream = try allocator.alloc(f32, flat_elems);
    defer allocator.free(upstream);
    fillDeterministicFlatQKVTensor(x, total, hidden_size, 9);
    fillDeterministicFlatQKVTensor(upstream, total, hidden_size, 10);

    const shared_grad = blk: {
        var graph = ml.graph.Graph.init(allocator);
        defer graph.deinit();
        var bld = ml.graph.Builder.init(&graph);
        const x_node = try bld.parameter("x", ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(total)), @as(i64, @intCast(hidden_size)) }));
        const upstream_node = try bld.parameter("upstream", ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(total)), @as(i64, @intCast(hidden_size)) }));
        const q = try stepParitySplitHeads(&bld, x_node, batch, seq_len, num_heads, head_dim);
        const k = try stepParitySplitHeads(&bld, x_node, batch, seq_len, num_heads, head_dim);
        const v = try stepParitySplitHeads(&bld, x_node, batch, seq_len, num_heads, head_dim);
        const k_t = try bld.transpose(k, &.{ 0, 2, 1 });
        const scores = try bld.matmul3D(q, k_t);
        const scale = try bld.scalarConst(.f32, 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))));
        const scaled = try bld.mul(scores, scale);
        const probs = try bld.softmax(scaled);
        const context = try bld.matmul3D(probs, v);
        const merged = try stepParityMergeHeads(&bld, context, batch, seq_len, num_heads, head_dim);
        const weighted = try bld.mul(merged, upstream_node);
        const loss = try bld.reduceSum(weighted, &.{ 0, 1 });
        try graph.markOutput(loss);
        const trainable = [_][]const u8{"x"};
        var session = try graph_training.CompiledTrainSession.init(allocator, &graph, loss, .{ .trainable_params = trainable[0..], .execution_strategy = .mpsgraph_required });
        defer session.deinit();
        const flat_dims = [_]i32{ @intCast(total), @intCast(hidden_size) };
        const x_ct = try cb.fromFloat32Shape(x, &flat_dims);
        defer cb.free(x_ct);
        const upstream_ct = try cb.fromFloat32Shape(upstream, &flat_dims);
        defer cb.free(upstream_ct);
        var runtime_inputs = std.AutoHashMapUnmanaged(ml.graph.NodeId, ops_mod.CT){};
        defer runtime_inputs.deinit(allocator);
        try runtime_inputs.put(allocator, x_node, x_ct);
        try runtime_inputs.put(allocator, upstream_node, upstream_ct);
        var result = try session.execute(cb, runtime_inputs);
        defer result.deinit();
        const grad = result.gradients.get("x") orelse return error.MissingSubstageGradient;
        break :blk try allocator.dupe(f32, grad);
    };
    defer allocator.free(shared_grad);

    const separate = blk: {
        var graph = ml.graph.Graph.init(allocator);
        defer graph.deinit();
        var bld = ml.graph.Builder.init(&graph);
        const q_node = try bld.parameter("q", ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(total)), @as(i64, @intCast(hidden_size)) }));
        const k_node = try bld.parameter("k", ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(total)), @as(i64, @intCast(hidden_size)) }));
        const v_node = try bld.parameter("v", ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(total)), @as(i64, @intCast(hidden_size)) }));
        const upstream_node = try bld.parameter("upstream", ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(total)), @as(i64, @intCast(hidden_size)) }));
        const q = try stepParitySplitHeads(&bld, q_node, batch, seq_len, num_heads, head_dim);
        const k = try stepParitySplitHeads(&bld, k_node, batch, seq_len, num_heads, head_dim);
        const v = try stepParitySplitHeads(&bld, v_node, batch, seq_len, num_heads, head_dim);
        const k_t = try bld.transpose(k, &.{ 0, 2, 1 });
        const scores = try bld.matmul3D(q, k_t);
        const scale = try bld.scalarConst(.f32, 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))));
        const scaled = try bld.mul(scores, scale);
        const probs = try bld.softmax(scaled);
        const context = try bld.matmul3D(probs, v);
        const merged = try stepParityMergeHeads(&bld, context, batch, seq_len, num_heads, head_dim);
        const weighted = try bld.mul(merged, upstream_node);
        const loss = try bld.reduceSum(weighted, &.{ 0, 1 });
        try graph.markOutput(loss);
        const trainable = [_][]const u8{ "q", "k", "v" };
        var session = try graph_training.CompiledTrainSession.init(allocator, &graph, loss, .{ .trainable_params = trainable[0..], .execution_strategy = .mpsgraph_required });
        defer session.deinit();
        const flat_dims = [_]i32{ @intCast(total), @intCast(hidden_size) };
        const x_ct = try cb.fromFloat32Shape(x, &flat_dims);
        defer cb.free(x_ct);
        const upstream_ct = try cb.fromFloat32Shape(upstream, &flat_dims);
        defer cb.free(upstream_ct);
        var runtime_inputs = std.AutoHashMapUnmanaged(ml.graph.NodeId, ops_mod.CT){};
        defer runtime_inputs.deinit(allocator);
        try runtime_inputs.put(allocator, q_node, x_ct);
        try runtime_inputs.put(allocator, k_node, x_ct);
        try runtime_inputs.put(allocator, v_node, x_ct);
        try runtime_inputs.put(allocator, upstream_node, upstream_ct);
        var result = try session.execute(cb, runtime_inputs);
        defer result.deinit();
        const q_grad = result.gradients.get("q") orelse return error.MissingSubstageGradient;
        const k_grad = result.gradients.get("k") orelse return error.MissingSubstageGradient;
        const v_grad = result.gradients.get("v") orelse return error.MissingSubstageGradient;
        const q_copy = try allocator.dupe(f32, q_grad);
        errdefer allocator.free(q_copy);
        const k_copy = try allocator.dupe(f32, k_grad);
        errdefer allocator.free(k_copy);
        const v_copy = try allocator.dupe(f32, v_grad);
        errdefer allocator.free(v_copy);
        break :blk .{ .q = q_copy, .k = k_copy, .v = v_copy };
    };
    defer allocator.free(separate.q);
    defer allocator.free(separate.k);
    defer allocator.free(separate.v);

    const sum_grad = try allocator.alloc(f32, flat_elems);
    defer allocator.free(sum_grad);
    for (sum_grad, separate.q, separate.k, separate.v) |*dst, qv, kv, vv| dst.* = qv + kv + vv;
    const delta = try computeSliceDelta(allocator, shared_grad, sum_grad);
    defer allocator.free(delta);

    var components = try allocator.alloc(StepParityNamedTensorStats, 6);
    errdefer allocator.free(components);
    components[0] = try stepParityNamedStats(allocator, "shared_grad", shared_grad);
    components[1] = try stepParityNamedStats(allocator, "separate_q_grad", separate.q);
    components[2] = try stepParityNamedStats(allocator, "separate_k_grad", separate.k);
    components[3] = try stepParityNamedStats(allocator, "separate_v_grad", separate.v);
    components[4] = try stepParityNamedStats(allocator, "sum_separate_grad", sum_grad);
    components[5] = try stepParityNamedStats(allocator, "shared_minus_sum", delta);
    return .{
        .name = "qkv_sum_consistency",
        .batch = batch,
        .seq_len = seq_len,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .hidden_size = hidden_size,
        .outer = batch * num_heads,
        .components = components,
    };
}

fn captureStepParityQKVSplitVJPProbe(
    allocator: std.mem.Allocator,
    cb: *const ComputeBackend,
    actual_batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rope_theta: f32,
) !StepParityQKVSplitVJPProbe {
    var cases = try allocator.alloc(StepParityQKVSplitVJPCase, 5);
    errdefer allocator.free(cases);
    cases[0] = try captureQKVSplitHeadsVJPCase(allocator, cb, actual_batch, seq_len, num_heads, head_dim);
    cases[1] = try captureQKVScoreMatmulVJPCase(allocator, cb, actual_batch, seq_len, num_heads, head_dim);
    cases[2] = try captureQKVValueContextVJPCase(allocator, cb, actual_batch, seq_len, num_heads, head_dim);
    cases[3] = try captureQKVRopeVJPCase(allocator, cb, actual_batch, seq_len, num_heads, head_dim, rope_theta);
    cases[4] = try captureQKVSumConsistencyCase(allocator, cb, actual_batch, seq_len, num_heads, head_dim);
    return .{
        .status = "captured",
        .version = 1,
        .runtime = "mpsgraph",
        .cases = cases,
    };
}

fn computeSliceDelta(
    allocator: std.mem.Allocator,
    lhs: []const f32,
    rhs: []const f32,
) ![]f32 {
    if (lhs.len != rhs.len) return error.StepParityAttentionReferenceShapeMismatch;
    const out = try allocator.alloc(f32, lhs.len);
    errdefer allocator.free(out);
    for (lhs, rhs, out) |a, b, *dst| dst.* = a - b;
    return out;
}

fn computeAttentionRowProbe(
    allocator: std.mem.Allocator,
    layer_idx: u32,
    name: []const u8,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    attention_mask: []const i32,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    mode: AttentionIndexMode,
    is_local_attention: bool,
    local_attention_window: u32,
    batch_idx: usize,
    head_idx: usize,
    query_idx: usize,
) !EncoderAttentionRowProbe {
    const total = batch * seq_len * num_heads * head_dim;
    if (q.len != total or k.len != total or v.len != total) return error.StepParityAttentionReferenceShapeMismatch;
    if (attention_mask.len < batch * seq_len) return error.StepParityAttentionReferenceShapeMismatch;
    if (batch_idx >= batch or head_idx >= num_heads or query_idx >= seq_len) return error.StepParityAttentionReferenceShapeMismatch;

    const scores = try allocator.alloc(f32, seq_len);
    defer allocator.free(scores);
    const probs = try allocator.alloc(f32, seq_len);
    defer allocator.free(probs);
    @memset(scores, 0.0);
    @memset(probs, 0.0);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const window_half: usize = @intCast(local_attention_window / 2);
    var out = EncoderAttentionRowProbe{
        .layer_idx = layer_idx,
        .name = name,
        .batch_idx = batch_idx,
        .head_idx = head_idx,
        .query_idx = query_idx,
        .valid_keys = 0,
        .score_mean = 0.0,
        .score_rms = 0.0,
        .score_min = std.math.inf(f32),
        .score_max = -std.math.inf(f32),
        .score_argmax = 0,
        .prob_entropy = 0.0,
        .prob_max = 0.0,
        .prob_argmax = 0,
        .prob_top2_gap = 0.0,
        .query_rms = 0.0,
        .query_max_abs = 0.0,
        .key_query_rms = 0.0,
        .key_query_max_abs = 0.0,
        .value_query_rms = 0.0,
        .value_query_max_abs = 0.0,
        .output_mean = 0.0,
        .output_rms = 0.0,
        .output_max_abs = 0.0,
        .query_sample_len = 0,
        .query_sample = [_]f32{0.0} ** attention_row_sample_max,
        .key_query_sample_len = 0,
        .key_query_sample = [_]f32{0.0} ** attention_row_sample_max,
        .value_query_sample_len = 0,
        .value_query_sample = [_]f32{0.0} ** attention_row_sample_max,
        .score_sample_len = 0,
        .score_sample = [_]f32{0.0} ** attention_row_sample_max,
        .prob_sample_len = 0,
        .prob_sample = [_]f32{0.0} ** attention_row_sample_max,
        .output_sample_len = 0,
        .output_sample = [_]f32{0.0} ** attention_row_sample_max,
    };

    var query_sum_sq: f64 = 0.0;
    var key_query_sum_sq: f64 = 0.0;
    var value_query_sum_sq: f64 = 0.0;
    for (0..head_dim) |dim| {
        const q_value = q[attentionIndex(mode, batch_idx, query_idx, head_idx, dim, seq_len, num_heads, head_dim)];
        const k_value = k[attentionIndex(mode, batch_idx, query_idx, head_idx, dim, seq_len, num_heads, head_dim)];
        const v_value = v[attentionIndex(mode, batch_idx, query_idx, head_idx, dim, seq_len, num_heads, head_dim)];
        if (out.query_sample_len < attention_row_sample_max) {
            out.query_sample[out.query_sample_len] = q_value;
            out.query_sample_len += 1;
        }
        if (out.key_query_sample_len < attention_row_sample_max) {
            out.key_query_sample[out.key_query_sample_len] = k_value;
            out.key_query_sample_len += 1;
        }
        if (out.value_query_sample_len < attention_row_sample_max) {
            out.value_query_sample[out.value_query_sample_len] = v_value;
            out.value_query_sample_len += 1;
        }
        query_sum_sq += @as(f64, q_value) * @as(f64, q_value);
        key_query_sum_sq += @as(f64, k_value) * @as(f64, k_value);
        value_query_sum_sq += @as(f64, v_value) * @as(f64, v_value);
        out.query_max_abs = @max(out.query_max_abs, absF32(q_value));
        out.key_query_max_abs = @max(out.key_query_max_abs, absF32(k_value));
        out.value_query_max_abs = @max(out.value_query_max_abs, absF32(v_value));
    }
    if (head_dim > 0) {
        const head_denom = @as(f64, @floatFromInt(head_dim));
        out.query_rms = @sqrt(query_sum_sq / head_denom);
        out.key_query_rms = @sqrt(key_query_sum_sq / head_denom);
        out.value_query_rms = @sqrt(value_query_sum_sq / head_denom);
    }

    var score_sum: f64 = 0.0;
    var score_sum_sq: f64 = 0.0;
    for (0..seq_len) |key_idx| {
        const diff = if (query_idx >= key_idx) query_idx - key_idx else key_idx - query_idx;
        const outside_window = is_local_attention and diff > window_half;
        if (attention_mask[batch_idx * seq_len + key_idx] == 0 or outside_window) continue;
        var score: f32 = 0.0;
        for (0..head_dim) |dim| {
            const q_idx = attentionIndex(mode, batch_idx, query_idx, head_idx, dim, seq_len, num_heads, head_dim);
            const k_idx = attentionIndex(mode, batch_idx, key_idx, head_idx, dim, seq_len, num_heads, head_dim);
            score += q[q_idx] * k[k_idx];
        }
        score *= scale;
        scores[key_idx] = score;
        if (out.score_sample_len < attention_row_sample_max) {
            out.score_sample[out.score_sample_len] = score;
            out.score_sample_len += 1;
        }
        out.valid_keys += 1;
        score_sum += score;
        score_sum_sq += @as(f64, score) * @as(f64, score);
        if (score < out.score_min) out.score_min = score;
        if (score > out.score_max) {
            out.score_max = score;
            out.score_argmax = key_idx;
        }
    }

    if (out.valid_keys == 0) {
        out.score_min = 0.0;
        out.score_max = 0.0;
        return out;
    }

    const valid_denom = @as(f64, @floatFromInt(out.valid_keys));
    out.score_mean = score_sum / valid_denom;
    out.score_rms = @sqrt(score_sum_sq / valid_denom);

    var exp_sum: f32 = 0.0;
    for (0..seq_len) |key_idx| {
        const diff = if (query_idx >= key_idx) query_idx - key_idx else key_idx - query_idx;
        const outside_window = is_local_attention and diff > window_half;
        if (attention_mask[batch_idx * seq_len + key_idx] == 0 or outside_window) continue;
        const weight = @exp(scores[key_idx] - out.score_max);
        probs[key_idx] = weight;
        exp_sum += weight;
    }
    if (exp_sum <= 0.0) return out;

    var top1: f32 = -std.math.inf(f32);
    var top2: f32 = -std.math.inf(f32);
    for (0..seq_len) |key_idx| {
        const diff = if (query_idx >= key_idx) query_idx - key_idx else key_idx - query_idx;
        const outside_window = is_local_attention and diff > window_half;
        if (attention_mask[batch_idx * seq_len + key_idx] == 0 or outside_window) continue;
        const prob = probs[key_idx] / exp_sum;
        probs[key_idx] = prob;
        if (out.prob_sample_len < attention_row_sample_max) {
            out.prob_sample[out.prob_sample_len] = prob;
            out.prob_sample_len += 1;
        }
        if (prob > 0.0) out.prob_entropy -= @as(f64, prob) * @log(@as(f64, prob));
        if (prob > out.prob_max) {
            out.prob_max = prob;
            out.prob_argmax = key_idx;
        }
        if (prob > top1) {
            top2 = top1;
            top1 = prob;
        } else if (prob > top2) {
            top2 = prob;
        }
    }
    out.prob_top2_gap = if (top2 == -std.math.inf(f32)) top1 else top1 - top2;

    var output_sum: f64 = 0.0;
    var output_sum_sq: f64 = 0.0;
    for (0..head_dim) |dim| {
        var accum: f32 = 0.0;
        for (0..seq_len) |key_idx| {
            const prob = probs[key_idx];
            if (prob == 0.0) continue;
            const v_idx = attentionIndex(mode, batch_idx, key_idx, head_idx, dim, seq_len, num_heads, head_dim);
            accum += prob * v[v_idx];
        }
        if (out.output_sample_len < attention_row_sample_max) {
            out.output_sample[out.output_sample_len] = accum;
            out.output_sample_len += 1;
        }
        output_sum += accum;
        output_sum_sq += @as(f64, accum) * @as(f64, accum);
        out.output_max_abs = @max(out.output_max_abs, absF32(accum));
    }
    if (head_dim > 0) {
        const head_denom = @as(f64, @floatFromInt(head_dim));
        out.output_mean = output_sum / head_denom;
        out.output_rms = @sqrt(output_sum_sq / head_denom);
    }
    return out;
}

fn findLayerAttentionInternal(
    act_buf: *const modern_bert.ActivationBuffer,
    layer_idx: u32,
    name: []const u8,
) ?[]const f32 {
    for (act_buf.attention_internals.items) |cap| {
        if (cap.layer_idx == layer_idx and std.mem.eql(u8, cap.name, name)) return cap.values;
    }
    return null;
}

fn findLayerActivationInput(
    act_buf: *const modern_bert.ActivationBuffer,
    layer_idx: u32,
    module_name: []const u8,
) ?[]const f32 {
    for (act_buf.items.items) |cap| {
        if (cap.layer_idx == layer_idx and std.mem.eql(u8, cap.module_name, module_name)) return cap.input;
    }
    return null;
}

fn computeLayerSdpaReferenceProbeStats(
    allocator: std.mem.Allocator,
    act_buf: *const modern_bert.ActivationBuffer,
    layer_idx: u32,
    attention_mask: []const i32,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    is_local_attention: bool,
    local_attention_window: u32,
) !?Layer0SdpaReferenceProbeStats {
    const q = findLayerAttentionInternal(act_buf, layer_idx, "q_rope") orelse return null;
    const k = findLayerAttentionInternal(act_buf, layer_idx, "k_rope") orelse return null;
    const v = findLayerAttentionInternal(act_buf, layer_idx, "v_raw") orelse return null;
    const attn_out = findLayerActivationInput(act_buf, layer_idx, "out_proj") orelse return null;
    const total = batch * seq_len * num_heads * head_dim;
    if (q.len != total or k.len != total or v.len != total or attn_out.len != total) return null;

    const token_ref = try computeSdpaReference(allocator, q, k, v, attention_mask, batch, seq_len, num_heads, head_dim, .token_major, is_local_attention, local_attention_window);
    defer allocator.free(token_ref);
    const token_delta = try computeSliceDelta(allocator, attn_out, token_ref);
    defer allocator.free(token_delta);
    const kernel_ref = try computeSdpaReference(allocator, q, k, v, attention_mask, batch, seq_len, num_heads, head_dim, .kernel_major, is_local_attention, local_attention_window);
    defer allocator.free(kernel_ref);
    const kernel_delta = try computeSliceDelta(allocator, attn_out, kernel_ref);
    defer allocator.free(kernel_delta);

    return .{
        .token_ref = fused_chunker_train.computeBoundaryProbeTensorStats(token_ref),
        .token_delta = fused_chunker_train.computeBoundaryProbeTensorStats(token_delta),
        .kernel_ref = fused_chunker_train.computeBoundaryProbeTensorStats(kernel_ref),
        .kernel_delta = fused_chunker_train.computeBoundaryProbeTensorStats(kernel_delta),
    };
}

fn computeLayerAttentionRowProbes(
    allocator: std.mem.Allocator,
    act_buf: *const modern_bert.ActivationBuffer,
    layer_idx: u32,
    attention_mask: []const i32,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    is_local_attention: bool,
    local_attention_window: u32,
) ![]EncoderAttentionRowProbe {
    const q = findLayerAttentionInternal(act_buf, layer_idx, "q_rope") orelse return try allocator.alloc(EncoderAttentionRowProbe, 0);
    const k = findLayerAttentionInternal(act_buf, layer_idx, "k_rope") orelse return try allocator.alloc(EncoderAttentionRowProbe, 0);
    const v = findLayerAttentionInternal(act_buf, layer_idx, "v_raw") orelse return try allocator.alloc(EncoderAttentionRowProbe, 0);
    const total = batch * seq_len * num_heads * head_dim;
    if (q.len != total or k.len != total or v.len != total or attention_mask.len < batch * seq_len or batch == 0) return try allocator.alloc(EncoderAttentionRowProbe, 0);

    var valid_keys: usize = 0;
    for (0..seq_len) |token_idx| {
        if (attention_mask[token_idx] != 0) valid_keys += 1;
    }
    if (valid_keys == 0) return try allocator.alloc(EncoderAttentionRowProbe, 0);

    var query_positions = [_]usize{ 0, 0, 0 };
    const target_ranks = [_]usize{ 0, valid_keys / 2, valid_keys - 1 };
    var valid_rank: usize = 0;
    for (0..seq_len) |token_idx| {
        if (attention_mask[token_idx] == 0) continue;
        for (target_ranks, 0..) |target, slot| {
            if (valid_rank == target) query_positions[slot] = token_idx;
        }
        valid_rank += 1;
    }

    const row_names = [2][3][]const u8{
        .{ "attn_token_row_h00_first", "attn_token_row_h00_mid", "attn_token_row_h00_last" },
        .{ "attn_token_row_h01_first", "attn_token_row_h01_mid", "attn_token_row_h01_last" },
    };
    const heads_to_probe = @min(num_heads, row_names.len);
    const rows = try allocator.alloc(EncoderAttentionRowProbe, heads_to_probe * query_positions.len);
    errdefer allocator.free(rows);

    var row_idx: usize = 0;
    for (0..heads_to_probe) |head_idx| {
        for (query_positions, 0..) |query_idx, query_slot| {
            rows[row_idx] = try computeAttentionRowProbe(
                allocator,
                layer_idx,
                row_names[head_idx][query_slot],
                q,
                k,
                v,
                attention_mask,
                batch,
                seq_len,
                num_heads,
                head_dim,
                .token_major,
                is_local_attention,
                local_attention_window,
                0,
                head_idx,
                query_idx,
            );
            row_idx += 1;
        }
    }
    return rows;
}

fn cloneLoRALayersForSegmentVJPParity(
    allocator: std.mem.Allocator,
    source: []const fused_chunker_lora.LoRALayer,
) ![]fused_chunker_lora.LoRALayer {
    const layers = try allocator.alloc(fused_chunker_lora.LoRALayer, source.len);
    errdefer allocator.free(layers);
    var initialized: usize = 0;
    errdefer {
        for (layers[0..initialized]) |*layer| layer.deinit();
    }

    for (source, 0..) |*src, i| {
        const rank: u32 = @intCast(if (src.in_features > 0) src.A.len / src.in_features else 0);
        layers[i] = try fused_chunker_lora.LoRALayer.initWithDoRA(
            allocator,
            src.layer_idx,
            src.module_name,
            src.in_features,
            src.out_features,
            rank,
            src.magnitude != null,
        );
        initialized += 1;
        @memcpy(layers[i].A, src.A);
        @memcpy(layers[i].B, src.B);
        layers[i].zeroGrads();
        if (src.magnitude) |src_magnitude| {
            if (layers[i].magnitude) |dst_magnitude| @memcpy(dst_magnitude, src_magnitude);
        }
    }

    return layers;
}

fn deinitClonedLoRALayers(layers: []fused_chunker_lora.LoRALayer, allocator: std.mem.Allocator) void {
    for (layers) |*layer| layer.deinit();
    allocator.free(layers);
}

fn findLoRALayerForSegmentVJPParity(
    layers: []fused_chunker_lora.LoRALayer,
    layer_idx: u32,
    module_name: []const u8,
) ?*fused_chunker_lora.LoRALayer {
    for (layers) |*layer| {
        if (layer.layer_idx == layer_idx and std.mem.eql(u8, layer.module_name, module_name)) return layer;
    }
    return null;
}

fn runSegmentVJPParityDiagnostic(
    allocator: std.mem.Allocator,
    cb: *const ComputeBackend,
    graph_config: modern_bert_graph.Config,
    actual_batch: usize,
    max_seq: usize,
    segment_start: u32,
    segment_end: u32,
    lora_rank: u32,
    lora_alpha: f32,
    include_hidden_grad: bool,
    include_adapter_grads: bool,
    segment_input: []const f32,
    upstream_grad: []const f32,
    attention_mask: []const f32,
    lora_layers: []fused_chunker_lora.LoRALayer,
) !void {
    var mps_session = try segmented_encoder.ModernBertSegmentVJPSession.initWithGradientOptionsAndStrategy(
        allocator,
        graph_config,
        @intCast(actual_batch),
        @intCast(max_seq),
        segment_start,
        segment_end,
        lora_rank,
        lora_alpha,
        include_hidden_grad,
        include_adapter_grads,
        .mpsgraph_required,
    );
    defer mps_session.deinit();

    const reference_strategy = segmentVJPParityReferenceStrategy();
    var reference_session = try segmented_encoder.ModernBertSegmentVJPSession.initWithGradientOptionsAndStrategy(
        allocator,
        graph_config,
        @intCast(actual_batch),
        @intCast(max_seq),
        segment_start,
        segment_end,
        lora_rank,
        lora_alpha,
        include_hidden_grad,
        include_adapter_grads,
        reference_strategy,
    );
    defer reference_session.deinit();

    const mps_layers = try cloneLoRALayersForSegmentVJPParity(allocator, lora_layers);
    defer deinitClonedLoRALayers(mps_layers, allocator);
    const reference_layers = try cloneLoRALayersForSegmentVJPParity(allocator, lora_layers);
    defer deinitClonedLoRALayers(reference_layers, allocator);

    var mps_result = try mps_session.executeWithOptions(
        cb,
        segment_input,
        upstream_grad,
        attention_mask,
        mps_layers,
        include_adapter_grads,
    );
    defer mps_result.deinit();

    var reference_result = try reference_session.executeWithOptions(
        cb,
        segment_input,
        upstream_grad,
        attention_mask,
        reference_layers,
        include_adapter_grads,
    );
    defer reference_result.deinit();

    var hidden_stats = SegmentVJPDiffStats{};
    if (include_hidden_grad) {
        const mps_hidden = mps_result.hidden_grad orelse return error.SegmentVJPParityMissingMpsHiddenGrad;
        const reference_hidden = reference_result.hidden_grad orelse return error.SegmentVJPParityMissingPartitionedHiddenGrad;
        try hidden_stats.add("hidden_grad", mps_hidden, reference_hidden);
    }

    var adapter_a_stats = SegmentVJPDiffStats{};
    var adapter_b_stats = SegmentVJPDiffStats{};
    if (include_adapter_grads) {
        for (mps_layers) |*mps_layer| {
            if (mps_layer.layer_idx < segment_start or mps_layer.layer_idx >= segment_end) continue;
            const reference_layer = findLoRALayerForSegmentVJPParity(
                reference_layers,
                mps_layer.layer_idx,
                mps_layer.module_name,
            ) orelse return error.SegmentVJPParityMissingAdapterLayer;
            try adapter_a_stats.add(mps_layer.module_name, mps_layer.grad_A, reference_layer.grad_A);
            try adapter_b_stats.add(mps_layer.module_name, mps_layer.grad_B, reference_layer.grad_B);
        }
    }

    print(
        "segment_vjp_parity_profile segment={d}-{d} reference={s} mps_runtime={s} reference_runtime={s} mps_exec_ms={d:.2} reference_exec_ms={d:.2} reference_interpreter_fallbacks={d} reference_graph_regions={d}\n",
        .{
            segment_start,
            segment_end,
            segmentVJPStrategyName(reference_strategy),
            segmentVJPProfileRuntimeName(mps_result.profile),
            segmentVJPProfileRuntimeName(reference_result.profile),
            @as(f64, @floatFromInt(mps_result.profile.compiled_execute_ns)) / 1_000_000.0,
            @as(f64, @floatFromInt(reference_result.profile.compiled_execute_ns)) / 1_000_000.0,
            reference_result.profile.interpreter_fallbacks,
            reference_result.profile.graph_region_dispatches,
        },
    );
    print(
        "segment_vjp_parity segment={d}-{d} hidden_tensors={d} hidden_elems={d} hidden_max_abs={d:.9} hidden_mean_abs={d:.9} hidden_rel_l2={d:.9} hidden_cos={d:.9} hidden_sign_mismatch={d} hidden_max={s} adapter_a_tensors={d} adapter_a_elems={d} adapter_a_max_abs={d:.9} adapter_a_mean_abs={d:.9} adapter_a_rel_l2={d:.9} adapter_a_cos={d:.9} adapter_a_sign_mismatch={d} adapter_a_max={s} adapter_b_tensors={d} adapter_b_elems={d} adapter_b_max_abs={d:.9} adapter_b_mean_abs={d:.9} adapter_b_rel_l2={d:.9} adapter_b_cos={d:.9} adapter_b_sign_mismatch={d} adapter_b_max={s}\n",
        .{
            segment_start,
            segment_end,
            hidden_stats.tensors,
            hidden_stats.elems,
            hidden_stats.max_abs,
            hidden_stats.meanAbs(),
            hidden_stats.relL2(),
            hidden_stats.cosine(),
            hidden_stats.sign_mismatches,
            hidden_stats.max_name,
            adapter_a_stats.tensors,
            adapter_a_stats.elems,
            adapter_a_stats.max_abs,
            adapter_a_stats.meanAbs(),
            adapter_a_stats.relL2(),
            adapter_a_stats.cosine(),
            adapter_a_stats.sign_mismatches,
            adapter_a_stats.max_name,
            adapter_b_stats.tensors,
            adapter_b_stats.elems,
            adapter_b_stats.max_abs,
            adapter_b_stats.meanAbs(),
            adapter_b_stats.relL2(),
            adapter_b_stats.cosine(),
            adapter_b_stats.sign_mismatches,
            adapter_b_stats.max_name,
        },
    );

    if (platform.env.getenvBoolDefault("TERMITE_SEGMENT_VJP_PARITY_FAIL", false)) {
        const tol: f32 = 1.0e-3;
        if (hidden_stats.max_abs > tol or adapter_a_stats.max_abs > tol or adapter_b_stats.max_abs > tol) {
            return error.SegmentVJPParityMismatch;
        }
    }
}

fn runMpsGraphLinearVJPParityDiagnostic(
    allocator: std.mem.Allocator,
    cb: *const ComputeBackend,
) !void {
    const rows = 5;
    const in_dim = 3;
    const out_dim = 4;

    var graph = ml.graph.Graph.init(allocator);
    defer graph.deinit();
    var bld = ml.graph.Builder.init(&graph);

    const x = try bld.parameter("x", ml.graph.Shape.init(.f32, &.{ rows, in_dim }));
    const w = try bld.parameter("w", ml.graph.Shape.init(.f32, &.{ out_dim, in_dim }));
    const upstream = try bld.parameter("upstream", ml.graph.Shape.init(.f32, &.{ rows, out_dim }));
    const y = try bld.linearNoBias(x, w, rows, in_dim, out_dim);
    const weighted = try bld.mul(y, upstream);
    const loss = try bld.reduceSum(weighted, &.{ 0, 1 });
    try graph.markOutput(loss);

    const trainable = [_][]const u8{ "x", "w" };
    var mps_session = try graph_training.CompiledTrainSession.init(
        allocator,
        &graph,
        loss,
        .{
            .trainable_params = trainable[0..],
            .execution_strategy = .mpsgraph_required,
        },
    );
    defer mps_session.deinit();

    var partitioned_session = try graph_training.CompiledTrainSession.init(
        allocator,
        &graph,
        loss,
        .{
            .trainable_params = trainable[0..],
            .execution_strategy = .partitioned_required,
        },
    );
    defer partitioned_session.deinit();

    var x_data: [rows * in_dim]f32 = undefined;
    for (&x_data, 0..) |*value, i| {
        const centered: i32 = @as(i32, @intCast((i * 17 + 3) % 11)) - 5;
        value.* = @as(f32, @floatFromInt(centered)) * 0.125;
    }
    var w_data: [out_dim * in_dim]f32 = undefined;
    for (&w_data, 0..) |*value, i| {
        const centered: i32 = @as(i32, @intCast((i * 13 + 5) % 17)) - 8;
        value.* = @as(f32, @floatFromInt(centered)) * 0.0625;
    }
    var upstream_data: [rows * out_dim]f32 = undefined;
    for (&upstream_data, 0..) |*value, i| {
        const centered: i32 = @as(i32, @intCast((i * 19 + 7) % 13)) - 6;
        value.* = @as(f32, @floatFromInt(centered)) * 0.2;
    }

    const x_shape = [_]i32{ rows, in_dim };
    const w_shape = [_]i32{ out_dim, in_dim };
    const upstream_shape = [_]i32{ rows, out_dim };
    const x_ct = try cb.fromFloat32Shape(x_data[0..], &x_shape);
    defer cb.free(x_ct);
    const w_ct = try cb.fromFloat32Shape(w_data[0..], &w_shape);
    defer cb.free(w_ct);
    const upstream_ct = try cb.fromFloat32Shape(upstream_data[0..], &upstream_shape);
    defer cb.free(upstream_ct);

    var runtime_inputs = std.AutoHashMapUnmanaged(ml.graph.NodeId, ops_mod.CT){};
    defer runtime_inputs.deinit(allocator);
    try runtime_inputs.put(allocator, x, x_ct);
    try runtime_inputs.put(allocator, w, w_ct);
    try runtime_inputs.put(allocator, upstream, upstream_ct);

    var mps_result = try mps_session.execute(cb, runtime_inputs);
    defer mps_result.deinit();
    var partitioned_result = try partitioned_session.execute(cb, runtime_inputs);
    defer partitioned_result.deinit();

    const mps_w = mps_result.gradients.get("w") orelse return error.MpsGraphLinearParityMissingWGradient;
    const partitioned_w = partitioned_result.gradients.get("w") orelse return error.MpsGraphLinearParityMissingPartitionedWGradient;

    var x_stats = SegmentVJPDiffStats{};
    const x_present = blk: {
        const mps_x = mps_result.gradients.get("x") orelse break :blk false;
        const partitioned_x = partitioned_result.gradients.get("x") orelse return error.MpsGraphLinearParityMissingPartitionedXGradient;
        try x_stats.add("x", mps_x, partitioned_x);
        break :blk true;
    };
    var w_stats = SegmentVJPDiffStats{};
    try w_stats.add("w", mps_w, partitioned_w);
    const loss_abs = absF32(mps_result.loss - partitioned_result.loss);

    print(
        "mpsgraph_linear_vjp_parity loss_mps={d:.9} loss_partitioned={d:.9} loss_abs={d:.9} mps_gradients={d} partitioned_gradients={d} x_present={} x_elems={d} x_max_abs={d:.9} x_mean_abs={d:.9} x_rel_l2={d:.9} x_cos={d:.9} x_sign_mismatch={d} w_elems={d} w_max_abs={d:.9} w_mean_abs={d:.9} w_rel_l2={d:.9} w_cos={d:.9} w_sign_mismatch={d} mps_exec_ms={d:.3} partitioned_exec_ms={d:.3}\n",
        .{
            mps_result.loss,
            partitioned_result.loss,
            loss_abs,
            mps_result.gradients.count(),
            partitioned_result.gradients.count(),
            x_present,
            x_stats.elems,
            x_stats.max_abs,
            x_stats.meanAbs(),
            x_stats.relL2(),
            x_stats.cosine(),
            x_stats.sign_mismatches,
            w_stats.elems,
            w_stats.max_abs,
            w_stats.meanAbs(),
            w_stats.relL2(),
            w_stats.cosine(),
            w_stats.sign_mismatches,
            @as(f64, @floatFromInt(mps_result.profile.execute_ns)) / 1_000_000.0,
            @as(f64, @floatFromInt(partitioned_result.profile.execute_ns)) / 1_000_000.0,
        },
    );

    if (platform.env.getenvBoolDefault("TERMITE_MPSGRAPH_LINEAR_VJP_PARITY_FAIL", false)) {
        const tol: f32 = 1.0e-4;
        if (loss_abs > tol or (x_present and x_stats.max_abs > tol) or w_stats.max_abs > tol) {
            return error.MpsGraphLinearVJPParityMismatch;
        }
    }
}

fn accumulateDirectLoRAGradsForLayerRange(
    cb: ComputeBackend,
    allocator: std.mem.Allocator,
    act_buf: *const modern_bert.ActivationBuffer,
    active_layers: []const bool,
    start_layer: u32,
    end_layer: u32,
    d_output: []const f32,
    adapter_set: *fused_chunker_lora.LoRAAdapterSet,
) !void {
    var n_caps: usize = 0;
    for (act_buf.items.items) |cap| {
        if (cap.layer_idx < start_layer or cap.layer_idx >= end_layer) continue;
        if (!fused_chunker_train.isLoRALayerActive(active_layers, cap.layer_idx)) continue;
        n_caps += 1;
    }
    if (n_caps == 0) return;

    const cap_layers = try allocator.alloc(u32, n_caps);
    defer allocator.free(cap_layers);
    const cap_modules = try allocator.alloc([]const u8, n_caps);
    defer allocator.free(cap_modules);
    const cap_inputs = try allocator.alloc([]const f32, n_caps);
    defer allocator.free(cap_inputs);
    const cap_in_feat = try allocator.alloc(usize, n_caps);
    defer allocator.free(cap_in_feat);
    const cap_out_feat = try allocator.alloc(usize, n_caps);
    defer allocator.free(cap_out_feat);

    var ci: usize = 0;
    for (act_buf.items.items) |cap| {
        if (cap.layer_idx < start_layer or cap.layer_idx >= end_layer) continue;
        if (!fused_chunker_train.isLoRALayerActive(active_layers, cap.layer_idx)) continue;
        cap_layers[ci] = cap.layer_idx;
        cap_modules[ci] = cap.module_name;
        cap_inputs[ci] = cap.input;
        cap_in_feat[ci] = cap.in_features;
        cap_out_feat[ci] = cap.out_features;
        ci += 1;
    }

    try segmented_encoder.backwardLoRADirect(
        cb,
        allocator,
        cap_layers,
        cap_modules,
        cap_inputs,
        cap_in_feat,
        cap_out_feat,
        d_output,
        adapter_set.layers,
        adapter_set.config.alpha,
    );
}

fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

fn enabledName(enabled: bool) []const u8 {
    return if (enabled) "enabled" else "disabled";
}

fn printPhase20ParityReport(opts: Options) void {
    print("phase20_parity_contract source=\"gopeft Phase 20 best boundary run\" f1=0.786 scope=\"boundary+dense\" splade={s}\n", .{
        enabledName(opts.splade),
    });
    print("phase20_parity_hparams epochs={d} batch_size={d} max_seq_len={d} max_chunks={d} lr={d} boundary_head_lr_multiplier={d} warmup_steps={d} lr_total_steps={d} weight_decay={d} max_grad_norm={d}\n", .{
        opts.epochs,
        opts.batch_size,
        opts.max_seq_len,
        opts.max_chunks,
        opts.learning_rate,
        opts.boundary_head_lr_multiplier,
        opts.warmup_steps,
        opts.lr_total_steps,
        opts.weight_decay,
        opts.max_grad_norm,
    });
    print("phase20_parity_lora rank={d} alpha={d} go_targets=query_proj,value_proj,key_proj,Wo zig_targets=query_proj,key_proj,value_proj,out_proj,wo\n", .{
        opts.lora_rank,
        opts.lora_alpha,
    });
    const effective_boundary_dropout: f32 = if (opts.deterministic) 0.0 else opts.boundary_dropout;
    const effective_neftune_alpha: f32 = if (opts.deterministic) 0.0 else opts.neftune_alpha;
    const effective_encoder_neftune_alpha: f32 = if (opts.encoder_neftune) effective_neftune_alpha else 0.0;
    print("phase20_parity_order deterministic={} go_epoch_shuffle={} shuffle_seed_rule=\"seed+epoch+1 when enabled\"\n", .{
        opts.deterministic,
        opts.go_epoch_shuffle,
    });
    print("phase20_parity_loss lambda_chunk={d} lambda_embed={d} boundary_focus_epochs={d} boundary_focus_lambda_embed={d} boundary_dropout={d} neftune_alpha={d} encoder_neftune_alpha={d} encoder_neftune={s} mrl={s} mrl_dims={s} loss_type={s} pos_weight={d} pos_weight_auto={} boundary_rank_loss_weight={d} boundary_rank_loss_margin={d} boundary_rank_loss_top_k={d} boundary_same_token_rank_loss_weight={d} boundary_same_token_rank_loss_top_k={d} boundary_same_token_negative_weight={d} boundary_same_token_negative_top_k={d} boundary_candidate_rank_loss_weight={d} boundary_candidate_rank_loss_top_k={d} boundary_candidate_negative_weight={d} boundary_gold_count_rank_loss_weight={d} boundary_gold_count_rank_loss_margin={d} boundary_gold_count_rank_loss_negative_multiplier={d} boundary_local_window_loss_weight={d} boundary_local_window_radius={d} contrastive_focal_gamma={d} contrastive_focal_alpha={d}\n", .{
        opts.lambda_chunk,
        opts.lambda_embed,
        opts.boundary_focus_epochs,
        opts.boundary_focus_lambda_embed,
        effective_boundary_dropout,
        effective_neftune_alpha,
        effective_encoder_neftune_alpha,
        enabledName(effective_encoder_neftune_alpha > 0.0),
        enabledName(opts.mrl),
        opts.mrl_dims_str,
        @tagName(opts.boundary_loss_type),
        opts.boundary_pos_weight,
        opts.boundary_pos_weight_auto,
        opts.boundary_rank_loss_weight,
        opts.boundary_rank_loss_margin,
        opts.boundary_rank_loss_top_k,
        opts.boundary_same_token_rank_loss_weight,
        opts.boundary_same_token_rank_loss_top_k,
        opts.boundary_same_token_negative_weight,
        opts.boundary_same_token_negative_top_k,
        opts.boundary_candidate_rank_loss_weight,
        opts.boundary_candidate_rank_loss_top_k,
        opts.boundary_candidate_negative_weight,
        opts.boundary_gold_count_rank_loss_weight,
        opts.boundary_gold_count_rank_loss_margin,
        opts.boundary_gold_count_rank_loss_negative_multiplier,
        opts.boundary_local_window_loss_weight,
        opts.boundary_local_window_radius,
        opts.contrastive_focal_gamma,
        opts.contrastive_focal_alpha,
    });
    print("phase20_parity_note token_level_ce_requires_pos_weight_matching_observed_boundary_rate; use --pos-weight auto for quality probes\n", .{});
    if (opts.model_dir) |mdir| {
        print("phase20_parity_artifacts model_dir={s} tokenizer_path={s}/tokenizer.json sha256=see_phase20_runner_if_available\n", .{ mdir, mdir });
    } else {
        print("phase20_parity_artifacts model_dir=none tokenizer_path=none sha256=unavailable\n", .{});
    }
}

fn avgMs(ns: u64, steps: u64) f64 {
    if (steps == 0) return 0.0;
    return nsToMs(ns) / @as(f64, @floatFromInt(steps));
}

fn examplesPerSecond(examples: u64, ns: u64) f64 {
    if (ns == 0) return 0.0;
    return (@as(f64, @floatFromInt(examples)) * 1_000_000_000.0) / @as(f64, @floatFromInt(ns));
}

fn processMemorySnapshot() ProcessMemorySnapshot {
    if (builtin.os.tag != .macos) return .{};
    var info: darwin.rusage_info_current = std.mem.zeroes(darwin.rusage_info_current);
    if (darwin.proc_pid_rusage(darwin.getpid(), darwin.RUSAGE_INFO_CURRENT, &info) != 0) return .{};
    return .{
        .available = true,
        .resident_bytes = info.ri_resident_size,
        .footprint_bytes = info.ri_phys_footprint,
    };
}

const darwin = if (builtin.os.tag == .macos) struct {
    pub const RUSAGE_INFO_CURRENT: i32 = 6;

    pub const rusage_info_current = extern struct {
        ri_uuid: [16]u8,
        ri_user_time: u64,
        ri_system_time: u64,
        ri_pkg_idle_wkups: u64,
        ri_interrupt_wkups: u64,
        ri_pageins: u64,
        ri_wired_size: u64,
        ri_resident_size: u64,
        ri_phys_footprint: u64,
        ri_proc_start_abstime: u64,
        ri_proc_exit_abstime: u64,
        ri_child_user_time: u64,
        ri_child_system_time: u64,
        ri_child_pkg_idle_wkups: u64,
        ri_child_interrupt_wkups: u64,
        ri_child_pageins: u64,
        ri_child_elapsed_abstime: u64,
        ri_diskio_bytesread: u64,
        ri_diskio_byteswritten: u64,
        ri_cpu_time_qos_default: u64,
        ri_cpu_time_qos_maintenance: u64,
        ri_cpu_time_qos_background: u64,
        ri_cpu_time_qos_utility: u64,
        ri_cpu_time_qos_legacy: u64,
        ri_cpu_time_qos_user_initiated: u64,
        ri_cpu_time_qos_user_interactive: u64,
        ri_billed_system_time: u64,
        ri_serviced_system_time: u64,
        ri_logical_writes: u64,
        ri_lifetime_max_phys_footprint: u64,
        ri_instructions: u64,
        ri_cycles: u64,
        ri_billed_energy: u64,
        ri_serviced_energy: u64,
        ri_interval_max_phys_footprint: u64,
        ri_runnable_time: u64,
        ri_flags: u64,
        ri_user_ptime: u64,
        ri_system_ptime: u64,
        ri_pinstructions: u64,
        ri_pcycles: u64,
        ri_energy_nj: u64,
        ri_penergy_nj: u64,
        ri_secure_time_in_system: u64,
        ri_secure_ptime_in_system: u64,
        ri_reserved: [12]u64,
    };

    extern "c" fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *rusage_info_current) i32;
    extern "c" fn getpid() i32;
} else struct {};

fn parseMemoryGigabytesToBytes(raw: []const u8) !u64 {
    const value = try std.fmt.parseFloat(f64, raw);
    if (!std.math.isFinite(value) or value < 0) return error.InvalidMemoryThreshold;
    return @intFromFloat(value * 1024.0 * 1024.0 * 1024.0);
}

fn computeSupervisionCounts(boundary_labels: []const f32, attention_mask: []const f32, total_tokens: usize) SupervisionCounts {
    var counts = SupervisionCounts{};
    const has_two_class_one_hot_labels = boundary_labels.len / 2 >= total_tokens;
    for (0..total_tokens) |idx| {
        if (idx >= attention_mask.len or attention_mask[idx] <= 0.5) continue;
        counts.valid_tokens += 1;
        const label_idx = if (has_two_class_one_hot_labels) idx * 2 + 1 else idx;
        if (label_idx < boundary_labels.len and boundary_labels[label_idx] > 0.5) counts.boundary_positive_tokens += 1;
    }
    return counts;
}

test "computeSupervisionCounts handles scalar and one-hot boundary labels" {
    const attention_mask = [_]f32{ 1, 1, 0, 1, 1 };
    const scalar_labels = [_]f32{ 0, 1, 1, 0, 1 };
    const one_hot_labels = [_]f32{
        1, 0,
        0, 1,
        0, 1,
        1, 0,
        0, 1,
    };

    const scalar_counts = computeSupervisionCounts(scalar_labels[0..], attention_mask[0..], 5);
    try std.testing.expectEqual(@as(u64, 4), scalar_counts.valid_tokens);
    try std.testing.expectEqual(@as(u64, 2), scalar_counts.boundary_positive_tokens);
    try std.testing.expectEqual(@as(u64, 1), scalar_counts.ignoredTokens(5));

    const one_hot_counts = computeSupervisionCounts(one_hot_labels[0..], attention_mask[0..], 5);
    try std.testing.expectEqual(@as(u64, 4), one_hot_counts.valid_tokens);
    try std.testing.expectEqual(@as(u64, 2), one_hot_counts.boundary_positive_tokens);
    try std.testing.expectEqual(@as(u64, 1), one_hot_counts.ignoredTokens(5));
}

fn writeStepMetric(
    metrics_writer: anytype,
    epoch: usize,
    local_step: u32,
    actual_batch: usize,
    total_tokens: usize,
    summary: TrainStepSummary,
    timing: StepTiming,
    supervision: SupervisionCounts,
    use_metal: bool,
    vjp_profile: ?segmented_encoder.SegmentVJPProfile,
    memory: ProcessMemorySnapshot,
    peak_resident_bytes: u64,
) !void {
    const runtime_name = if (vjp_profile) |profile|
        if (profile.mpsgraph_runtime) "mpsgraph" else if (profile.partitioned_runtime) "partitioned" else "interpreter"
    else
        "none";
    const profile = vjp_profile orelse segmented_encoder.SegmentVJPProfile{};
    try std.json.Stringify.value(.{
        .event = "step",
        .schema_version = 1,
        .epoch = epoch + 1,
        .local_step = local_step,
        .step = summary.step,
        .examples = actual_batch,
        .total_tokens = total_tokens,
        .valid_tokens = supervision.valid_tokens,
        .supervised_token_count = supervision.valid_tokens,
        .entity_token_count = supervision.boundary_positive_tokens,
        .ignored_token_count = supervision.ignoredTokens(total_tokens),
        .loss = summary.total_loss,
        .boundary_loss = summary.boundary_loss,
        .boundary_ce_loss = summary.boundary_ce_loss,
        .boundary_rank_loss = summary.boundary_rank_loss,
        .boundary_local_window_loss = summary.boundary_local_window_loss,
        .contrastive_loss = summary.contrastive_loss,
        .learning_rate = summary.learning_rate,
        .boundary_head_learning_rate = summary.boundary_head_learning_rate,
        .backend = if (use_metal) "metal" else "native",
        .step_wall_ms = nsToMs(timing.total_ns),
        .batch_ms = nsToMs(timing.batch_ns),
        .lora_refresh_ms = nsToMs(timing.lora_refresh_ns),
        .encoder_ms = nsToMs(timing.encoder_ns),
        .hard_negative_ms = nsToMs(timing.hard_neg_ns),
        .train_step_ms = nsToMs(timing.train_ns),
        .lora_update_ms = nsToMs(timing.lora_update_ns),
        .splade_ms = nsToMs(timing.splade_ns),
        .examples_per_second = examplesPerSecond(@intCast(actual_batch), timing.total_ns),
        .vjp_runtime = runtime_name,
        .vjp_runtime_input_ms = nsToMs(profile.runtime_input_ns),
        .vjp_execute_ms = nsToMs(profile.compiled_execute_ns),
        .vjp_extract_ms = nsToMs(profile.compiled_extract_ns),
        .vjp_accumulate_ms = nsToMs(profile.accumulate_ns),
        .vjp_total_ms = nsToMs(profile.total_ns),
        .vjp_partitions_executed = profile.partitions_executed,
        .vjp_backend_command_dispatches = profile.backend_command_dispatches,
        .vjp_planned_operator_dispatches = profile.planned_operator_dispatches,
        .vjp_graph_region_dispatches = profile.graph_region_dispatches,
        .vjp_interpreter_fallbacks = profile.interpreter_fallbacks,
        .memory_available = memory.available,
        .resident_bytes = memory.resident_bytes,
        .footprint_bytes = memory.footprint_bytes,
        .peak_resident_bytes = peak_resident_bytes,
    }, .{}, &metrics_writer.interface);
    try metrics_writer.interface.writeByte('\n');
    try metrics_writer.interface.flush();
}

fn writeValidationMetric(
    metrics_writer: anytype,
    event: []const u8,
    epoch: usize,
    step: u64,
    samples: usize,
    total_samples: usize,
    eval_ms: f64,
    summary: fused_chunker_train.EvalSummary,
) !void {
    try std.json.Stringify.value(.{
        .event = event,
        .schema_version = 1,
        .epoch = epoch + 1,
        .step = step,
        .samples = samples,
        .total_samples = total_samples,
        .eval_ms = eval_ms,
        .f1 = summary.boundary_f1,
        .precision = summary.boundary_precision,
        .recall = summary.boundary_recall,
        .tp = summary.boundary_tp,
        .fp = summary.boundary_fp,
        .@"fn" = summary.boundary_fn,
        .best_f1 = summary.best_boundary_f1,
        .best_threshold = summary.best_boundary_threshold,
        .best_tp = summary.best_boundary_tp,
        .best_fp = summary.best_boundary_fp,
        .best_fn = summary.best_boundary_fn,
        .calibrated_threshold_available = summary.calibrated_threshold_available,
        .calibrated_boundary_threshold = summary.calibrated_boundary_threshold,
        .calibrated_boundary_f1 = summary.calibrated_boundary_f1,
        .calibrated_boundary_precision = summary.calibrated_boundary_precision,
        .calibrated_boundary_recall = summary.calibrated_boundary_recall,
        .calibrated_boundary_tp = summary.calibrated_boundary_tp,
        .calibrated_boundary_fp = summary.calibrated_boundary_fp,
        .calibrated_boundary_fn = summary.calibrated_boundary_fn,
        .calibrated_predicted_positive_rate = summary.calibrated_predicted_positive_rate,
        .fitted_threshold_available = summary.fitted_threshold_available,
        .fitted_boundary_threshold = summary.fitted_boundary_threshold,
        .fitted_boundary_f1 = summary.fitted_boundary_f1,
        .fitted_boundary_precision = summary.fitted_boundary_precision,
        .fitted_boundary_recall = summary.fitted_boundary_recall,
        .fitted_boundary_tp = summary.fitted_boundary_tp,
        .fitted_boundary_fp = summary.fitted_boundary_fp,
        .fitted_boundary_fn = summary.fitted_boundary_fn,
        .fitted_predicted_positive_rate = summary.fitted_predicted_positive_rate,
        .average_precision = summary.average_precision,
        .precision_at_gold_count = summary.precision_at_gold_count,
        .recall_at_gold_count = summary.recall_at_gold_count,
        .f1_at_gold_count = summary.f1_at_gold_count,
        .threshold_at_gold_count = summary.threshold_at_gold_count,
        .predicted_positive_rate_at_gold_count = summary.predicted_positive_rate_at_gold_count,
        .gold_count_tp = summary.gold_count_tp,
        .gold_count_fp = summary.gold_count_fp,
        .gold_count_fn = summary.gold_count_fn,
        .max_rank_f1 = summary.max_rank_f1,
        .max_rank_precision = summary.max_rank_precision,
        .max_rank_recall = summary.max_rank_recall,
        .max_rank_threshold = summary.max_rank_threshold,
        .max_rank_predicted_positive_rate = summary.max_rank_predicted_positive_rate,
        .max_rank_tp = summary.max_rank_tp,
        .max_rank_fp = summary.max_rank_fp,
        .max_rank_fn = summary.max_rank_fn,
        .sample_oracle_count_samples = summary.sample_oracle_count_samples,
        .sample_oracle_count_topk_f1 = summary.sample_oracle_count_topk_f1,
        .sample_oracle_count_topk_precision = summary.sample_oracle_count_topk_precision,
        .sample_oracle_count_topk_recall = summary.sample_oracle_count_topk_recall,
        .sample_oracle_count_topk_tp = summary.sample_oracle_count_topk_tp,
        .sample_oracle_count_topk_fp = summary.sample_oracle_count_topk_fp,
        .sample_oracle_count_topk_fn = summary.sample_oracle_count_topk_fn,
        .sample_oracle_count_nms_f1 = summary.sample_oracle_count_nms_f1,
        .sample_oracle_count_nms_precision = summary.sample_oracle_count_nms_precision,
        .sample_oracle_count_nms_recall = summary.sample_oracle_count_nms_recall,
        .sample_oracle_count_nms_tp = summary.sample_oracle_count_nms_tp,
        .sample_oracle_count_nms_fp = summary.sample_oracle_count_nms_fp,
        .sample_oracle_count_nms_fn = summary.sample_oracle_count_nms_fn,
        .sample_oracle_count_nms_radius = summary.sample_oracle_count_nms_radius,
        .sample_oracle_count_length_window_f1 = summary.sample_oracle_count_length_window_f1,
        .sample_oracle_count_length_window_precision = summary.sample_oracle_count_length_window_precision,
        .sample_oracle_count_length_window_recall = summary.sample_oracle_count_length_window_recall,
        .sample_oracle_count_length_window_tp = summary.sample_oracle_count_length_window_tp,
        .sample_oracle_count_length_window_fp = summary.sample_oracle_count_length_window_fp,
        .sample_oracle_count_length_window_fn = summary.sample_oracle_count_length_window_fn,
        .sample_oracle_count_length_window_min_radius = summary.sample_oracle_count_length_window_min_radius,
        .sample_oracle_count_length_window_radius_fraction = summary.sample_oracle_count_length_window_radius_fraction,
        .gold_positive_mean_rank = summary.gold_positive_mean_rank,
        .gold_positive_mean_rank_percentile = summary.gold_positive_mean_rank_percentile,
        .gold_positive_median_rank = summary.gold_positive_median_rank,
        .gold_positive_median_rank_percentile = summary.gold_positive_median_rank_percentile,
        .gold_positive_p90_rank = summary.gold_positive_p90_rank,
        .gold_positive_p90_rank_percentile = summary.gold_positive_p90_rank_percentile,
        .gold_positive_p99_rank = summary.gold_positive_p99_rank,
        .gold_positive_p99_rank_percentile = summary.gold_positive_p99_rank_percentile,
        .gold_positive_worst_rank = summary.gold_positive_worst_rank,
        .gold_positive_top_5x_count = summary.gold_positive_top_5x_count,
        .gold_positive_top_5x_recall = summary.gold_positive_top_5x_recall,
        .gold_positive_top_10x_count = summary.gold_positive_top_10x_count,
        .gold_positive_top_10x_recall = summary.gold_positive_top_10x_recall,
        .valid_tokens = summary.valid_tokens,
        .gold_positives = summary.gold_positives,
        .gold_positive_rate = summary.gold_positive_rate,
        .predicted_positive_rate = summary.predicted_positive_rate,
        .best_predicted_positive_rate = summary.best_predicted_positive_rate,
        .mean_positive_probability_gold_positive = summary.mean_positive_probability_gold_positive,
        .mean_positive_probability_gold_negative = summary.mean_positive_probability_gold_negative,
        .mean_boundary_margin_gold_positive = summary.mean_boundary_margin_gold_positive,
        .mean_boundary_margin_gold_negative = summary.mean_boundary_margin_gold_negative,
        .mean_logit0_gold_positive = summary.mean_logit0_gold_positive,
        .mean_logit1_gold_positive = summary.mean_logit1_gold_positive,
        .mean_logit0_gold_negative = summary.mean_logit0_gold_negative,
        .mean_logit1_gold_negative = summary.mean_logit1_gold_negative,
    }, .{}, &metrics_writer.interface);
    try metrics_writer.interface.writeByte('\n');
    try metrics_writer.interface.flush();
}

fn writeFrozenFeatureProbeCompleteMetric(
    metrics_writer: anytype,
    train_examples: usize,
    val_examples: usize,
    train_offset: usize,
    val_offset: usize,
    epochs: u32,
    lr: f32,
    final_step: u32,
    final_boundary_loss: f32,
    final_total_loss: f32,
    train_after: fused_chunker_train.EvalSummary,
    val_after: ?fused_chunker_train.EvalSummary,
) !void {
    try std.json.Stringify.value(.{
        .event = "frozen_feature_probe",
        .schema_version = 1,
        .status = "complete",
        .train_examples = train_examples,
        .val_examples = val_examples,
        .train_offset = train_offset,
        .val_offset = val_offset,
        .epochs = epochs,
        .lr = lr,
        .final_step = final_step,
        .final_boundary_loss = final_boundary_loss,
        .final_total_loss = final_total_loss,
        .train_best_f1 = train_after.best_boundary_f1,
        .train_fixed_f1 = train_after.boundary_f1,
        .train_max_rank_f1 = train_after.max_rank_f1,
        .train_average_precision = train_after.average_precision,
        .train_probability_gap = train_after.mean_positive_probability_gold_positive - train_after.mean_positive_probability_gold_negative,
        .val_best_f1 = if (val_after) |summary| summary.best_boundary_f1 else null,
        .val_fixed_f1 = if (val_after) |summary| summary.boundary_f1 else null,
        .val_max_rank_f1 = if (val_after) |summary| summary.max_rank_f1 else null,
        .val_average_precision = if (val_after) |summary| summary.average_precision else null,
        .val_probability_gap = if (val_after) |summary| summary.mean_positive_probability_gold_positive - summary.mean_positive_probability_gold_negative else null,
    }, .{}, &metrics_writer.interface);
    try metrics_writer.interface.writeByte('\n');
    try metrics_writer.interface.flush();
}

const BoundaryAlignmentCandidate = struct {
    probability: f32,
    flat_index: usize,
};

const BoundaryAlignmentFeatureStats = struct {
    mean: f32,
    rms: f32,
    l2: f32,
    max_abs: f32,
    first0: f32,
    first1: f32,
    first2: f32,
    first3: f32,
};

const BoundaryAlignmentChunkMarkers = struct {
    chunk_start_index: ?usize = null,
    chunk_end_exclusive_index: ?usize = null,
    previous_chunk_end_index: ?usize = null,
};

fn boundaryAlignmentCandidateGreaterThan(_: void, a: BoundaryAlignmentCandidate, b: BoundaryAlignmentCandidate) bool {
    if (a.probability == b.probability) return a.flat_index < b.flat_index;
    return a.probability > b.probability;
}

fn boundaryAlignmentDumpPath(
    allocator: std.mem.Allocator,
    dump_dir: []const u8,
    eval_label: []const u8,
    step: u64,
) ![]u8 {
    return try std.fmt.allocPrint(allocator, "{s}/boundary_alignment_{s}_step_{d}.jsonl", .{ dump_dir, eval_label, step });
}

fn boundaryAlignmentChunkMarkers(
    batch: *const fused_chunker_data.FusedBatch,
    local_sample_idx: usize,
    token_idx: usize,
) BoundaryAlignmentChunkMarkers {
    const base = local_sample_idx * batch.max_chunks;
    var markers = BoundaryAlignmentChunkMarkers{};
    for (0..batch.max_chunks) |chunk_idx| {
        if (batch.chunk_mask[base + chunk_idx] <= 0.5) continue;
        const start = batch.chunk_starts[base + chunk_idx];
        const end = batch.chunk_ends[base + chunk_idx];
        if (start >= 0 and @as(usize, @intCast(start)) == token_idx) {
            markers.chunk_start_index = chunk_idx;
        }
        if (end >= 0 and @as(usize, @intCast(end)) == token_idx) {
            markers.chunk_end_exclusive_index = chunk_idx;
        }
        if (end > 0 and @as(usize, @intCast(end - 1)) == token_idx) {
            markers.previous_chunk_end_index = chunk_idx;
        }
    }
    return markers;
}

fn nearestBoundaryGoldDelta(
    batch: *const fused_chunker_data.FusedBatch,
    local_sample_idx: usize,
    token_idx: usize,
) ?i32 {
    const base = local_sample_idx * batch.max_seq_len;
    var best_delta: ?i32 = null;
    var best_abs: u32 = std.math.maxInt(u32);
    for (0..batch.max_seq_len) |candidate_idx| {
        if (batch.boundary_labels[base + candidate_idx] <= 0.5) continue;
        const delta = @as(i32, @intCast(candidate_idx)) - @as(i32, @intCast(token_idx));
        const abs_delta: u32 = @intCast(if (delta < 0) -delta else delta);
        if (abs_delta < best_abs) {
            best_abs = abs_delta;
            best_delta = delta;
        }
    }
    return best_delta;
}

fn boundaryAlignmentFeatureStats(
    features: []const f32,
    flat_index: usize,
    hidden_size: usize,
) ?BoundaryAlignmentFeatureStats {
    if (hidden_size == 0) return null;
    const start = flat_index * hidden_size;
    const end = start + hidden_size;
    if (end > features.len) return null;

    var sum: f64 = 0;
    var sum_sq: f64 = 0;
    var max_abs: f32 = 0;
    for (features[start..end]) |value| {
        sum += value;
        sum_sq += @as(f64, value) * @as(f64, value);
        const abs_value = @abs(value);
        if (abs_value > max_abs) max_abs = abs_value;
    }
    const hidden_f64: f64 = @floatFromInt(hidden_size);
    const row = features[start..end];
    return .{
        .mean = @floatCast(sum / hidden_f64),
        .rms = @floatCast(@sqrt(sum_sq / hidden_f64)),
        .l2 = @floatCast(@sqrt(sum_sq)),
        .max_abs = max_abs,
        .first0 = row[0],
        .first1 = if (hidden_size > 1) row[1] else 0,
        .first2 = if (hidden_size > 2) row[2] else 0,
        .first3 = if (hidden_size > 3) row[3] else 0,
    };
}

fn writeBoundaryAlignmentRecord(
    writer: *std.Io.Writer,
    eval_label: []const u8,
    step: u64,
    batch_idx: usize,
    batch: *const fused_chunker_data.FusedBatch,
    logits: []const f32,
    features: []const f32,
    hidden_size: usize,
    flat_index: usize,
    record_kind: []const u8,
    top_probability_rank: ?usize,
    probability_rank: ?usize,
    valid_token_count: usize,
) !void {
    const local_sample_idx = flat_index / batch.max_seq_len;
    const token_idx = flat_index % batch.max_seq_len;
    const logit0 = logits[flat_index * 2 + 0];
    const logit1 = logits[flat_index * 2 + 1];
    const probability = fused_chunker_loss.positiveBoundaryProbability(logit0, logit1);
    const markers = boundaryAlignmentChunkMarkers(batch, local_sample_idx, token_idx);
    const probability_rank_percentile: ?f64 = if (probability_rank) |rank|
        @as(f64, @floatFromInt(rank)) / @as(f64, @floatFromInt(valid_token_count))
    else
        null;
    try std.json.Stringify.value(.{
        .event = "boundary_alignment_token",
        .schema_version = 2,
        .eval = eval_label,
        .step = step,
        .batch = batch_idx,
        .record_kind = record_kind,
        .top_probability_rank = top_probability_rank,
        .probability_rank = probability_rank,
        .probability_rank_percentile = probability_rank_percentile,
        .valid_token_count = valid_token_count,
        .eval_sample_index = batch.sample_indices[local_sample_idx],
        .local_sample_index = local_sample_idx,
        .sample_token_index = token_idx,
        .token_id = batch.input_ids[flat_index],
        .attention = batch.attention_mask[flat_index],
        .gold_boundary = batch.boundary_labels[flat_index] > 0.5,
        .probability = probability,
        .margin = logit1 - logit0,
        .logit0 = logit0,
        .logit1 = logit1,
        .nearest_gold_delta = nearestBoundaryGoldDelta(batch, local_sample_idx, token_idx),
        .chunk_start_index = markers.chunk_start_index,
        .chunk_end_exclusive_index = markers.chunk_end_exclusive_index,
        .previous_chunk_end_index = markers.previous_chunk_end_index,
        .feature_stats = boundaryAlignmentFeatureStats(features, flat_index, hidden_size),
    }, .{}, writer);
    try writer.writeByte('\n');
}

fn writeBoundaryAlignmentDumpBatch(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    eval_label: []const u8,
    step: u64,
    batch_idx: usize,
    batch: *const fused_chunker_data.FusedBatch,
    logits: []const f32,
    features: []const f32,
    hidden_size: usize,
    top_k: usize,
) !void {
    const total_tokens = batch.batch_size * batch.max_seq_len;

    var candidates = std.ArrayListUnmanaged(BoundaryAlignmentCandidate).empty;
    defer candidates.deinit(allocator);
    try candidates.ensureTotalCapacity(allocator, total_tokens);

    for (0..total_tokens) |flat_index| {
        if (batch.attention_mask[flat_index] == 0) continue;
        const logit0 = logits[flat_index * 2 + 0];
        const logit1 = logits[flat_index * 2 + 1];
        try candidates.append(allocator, .{
            .probability = fused_chunker_loss.positiveBoundaryProbability(logit0, logit1),
            .flat_index = flat_index,
        });
    }

    std.mem.sort(BoundaryAlignmentCandidate, candidates.items, {}, boundaryAlignmentCandidateGreaterThan);

    const rank_by_flat_index = try allocator.alloc(usize, total_tokens);
    defer allocator.free(rank_by_flat_index);
    @memset(rank_by_flat_index, 0);
    for (candidates.items, 0..) |candidate, rank| {
        rank_by_flat_index[candidate.flat_index] = rank + 1;
    }

    for (0..total_tokens) |flat_index| {
        if (batch.boundary_labels[flat_index] > 0.5) {
            try writeBoundaryAlignmentRecord(
                writer,
                eval_label,
                step,
                batch_idx,
                batch,
                logits,
                features,
                hidden_size,
                flat_index,
                "gold_boundary",
                null,
                if (rank_by_flat_index[flat_index] == 0) null else rank_by_flat_index[flat_index],
                candidates.items.len,
            );
        }
    }

    const limit = @min(top_k, candidates.items.len);
    for (candidates.items[0..limit], 0..) |candidate, rank| {
        try writeBoundaryAlignmentRecord(
            writer,
            eval_label,
            step,
            batch_idx,
            batch,
            logits,
            features,
            hidden_size,
            candidate.flat_index,
            "top_probability",
            rank + 1,
            rank + 1,
            candidates.items.len,
        );
    }
}

fn writeEpochMetric(
    metrics_writer: anytype,
    epoch: usize,
    total_epochs: u32,
    steps: u64,
    timing: TimingTotals,
) !void {
    try std.json.Stringify.value(.{
        .event = "epoch",
        .schema_version = 1,
        .epoch = epoch + 1,
        .epochs = total_epochs,
        .steps = steps,
        .avg_step_ms = avgMs(timing.total_ns, timing.steps),
        .batch_ms = avgMs(timing.batch_ns, timing.steps),
        .lora_refresh_ms = avgMs(timing.lora_refresh_ns, timing.steps),
        .encoder_ms = avgMs(timing.encoder_ns, timing.steps),
        .hard_negative_ms = avgMs(timing.hard_neg_ns, timing.steps),
        .train_ms = avgMs(timing.train_ns, timing.steps),
        .lora_update_ms = avgMs(timing.lora_update_ns, timing.steps),
        .splade_ms = avgMs(timing.splade_ns, timing.steps),
        .throughput_examples_per_second = examplesPerSecond(timing.examples, timing.total_ns),
    }, .{}, &metrics_writer.interface);
    try metrics_writer.interface.writeByte('\n');
    try metrics_writer.interface.flush();
}

const BoundaryThresholdRecommendation = struct {
    available: bool = false,
    threshold: f32 = 0.5,
    validation_f1: f32 = 0,
    fixed_f1: f32 = 0,
    average_precision: f32 = 0,
    max_rank_f1: f32 = 0,
    source_event: []const u8 = "default",
    step: u64 = 0,
    epoch: u32 = 0,
    samples: usize = 0,
    total_samples: usize = 0,
    full_validation: bool = false,
};

fn maybeUpdateBoundaryThresholdRecommendation(
    recommendation: *BoundaryThresholdRecommendation,
    summary: fused_chunker_train.EvalSummary,
    source_event: []const u8,
    step: u64,
    epoch: u32,
    samples: usize,
    total_samples: usize,
) void {
    if (summary.valid_tokens == 0) return;
    if (!std.math.isFinite(summary.best_boundary_threshold) or
        summary.best_boundary_threshold < 0 or
        summary.best_boundary_threshold > 1 or
        !std.math.isFinite(summary.best_boundary_f1))
    {
        return;
    }

    const full_validation = total_samples > 0 and samples == total_samples;
    const should_update = !recommendation.available or
        (full_validation and !recommendation.full_validation) or
        (full_validation == recommendation.full_validation and
            summary.best_boundary_f1 > recommendation.validation_f1);
    if (!should_update) return;

    recommendation.* = .{
        .available = true,
        .threshold = summary.best_boundary_threshold,
        .validation_f1 = summary.best_boundary_f1,
        .fixed_f1 = summary.boundary_f1,
        .average_precision = summary.average_precision,
        .max_rank_f1 = summary.max_rank_f1,
        .source_event = source_event,
        .step = step,
        .epoch = epoch,
        .samples = samples,
        .total_samples = total_samples,
        .full_validation = full_validation,
    };
}

fn writeFusedTrainingManifest(
    allocator: std.mem.Allocator,
    path: []const u8,
    opts: *const Options,
    backend_name: []const u8,
    metrics_path: []const u8,
    train_stats: fused_chunker_data.FusedDatasetStats,
    val_stats: ?fused_chunker_data.FusedDatasetStats,
    status: []const u8,
    total_steps: u64,
    final_loss: ?f32,
    final_checkpoint_path: ?[]const u8,
    best_val_step: u64,
    best_val_epoch: u32,
    best_val_f1: f32,
    boundary_threshold_recommendation: BoundaryThresholdRecommendation,
    peak_resident_bytes: u64,
) !void {
    _ = allocator;
    var file = try compat.cwd().createFile(compat.io(), path, .{ .truncate = true });
    defer file.close(compat.io());
    var buf: [8192]u8 = undefined;
    var writer = file.writerStreaming(compat.io(), &buf);
    try std.json.Stringify.value(.{
        .schema_version = fused_manifest_schema_version,
        .artifact_family_version = fused_artifact_family_version,
        .status = status,
        .metrics_file = fused_metrics_file_name,
        .metrics_path = metrics_path,
        .data_path = opts.data_path,
        .val_data_path = opts.val_data_path,
        .output_dir = opts.output_dir,
        .model_dir = opts.model_dir,
        .backend = backend_name,
        .epochs = opts.epochs,
        .batch_size = opts.batch_size,
        .learning_rate = opts.learning_rate,
        .boundary_head_lr_multiplier = opts.boundary_head_lr_multiplier,
        .warmup_steps = opts.warmup_steps,
        .lr_total_steps = opts.lr_total_steps,
        .weight_decay = opts.weight_decay,
        .max_grad_norm = opts.max_grad_norm,
        .hidden_size = opts.hidden_size,
        .num_layers = opts.num_layers,
        .max_seq_len = opts.max_seq_len,
        .max_chunks = opts.max_chunks,
        .lora_rank = opts.lora_rank,
        .lora_alpha = opts.lora_alpha,
        .encoder_vjp = encoderVJPModeName(opts.encoder_vjp),
        .layers_per_segment = opts.layers_per_segment,
        .deterministic = opts.deterministic,
        .go_epoch_shuffle = opts.go_epoch_shuffle,
        .splade = opts.splade,
        .mrl = opts.mrl,
        .mrl_dims = opts.mrl_dims_str,
        .boundary_loss_type = @tagName(opts.boundary_loss_type),
        .boundary_pos_weight = opts.boundary_pos_weight,
        .boundary_pos_weight_auto = opts.boundary_pos_weight_auto,
        .boundary_pos_weight_auto_max_examples = opts.boundary_pos_weight_auto_max_examples,
        .boundary_pos_weight_auto_observed_examples = opts.boundary_pos_weight_auto_observed_examples,
        .boundary_pos_weight_auto_valid_tokens = opts.boundary_pos_weight_auto_valid_tokens,
        .boundary_pos_weight_auto_gold_tokens = opts.boundary_pos_weight_auto_gold_tokens,
        .boundary_pos_weight_auto_gold_rate = opts.boundary_pos_weight_auto_gold_rate,
        .boundary_rank_loss_weight = opts.boundary_rank_loss_weight,
        .boundary_rank_loss_margin = opts.boundary_rank_loss_margin,
        .boundary_rank_loss_top_k = opts.boundary_rank_loss_top_k,
        .boundary_same_token_rank_loss_weight = opts.boundary_same_token_rank_loss_weight,
        .boundary_same_token_rank_loss_top_k = opts.boundary_same_token_rank_loss_top_k,
        .boundary_same_token_negative_weight = opts.boundary_same_token_negative_weight,
        .boundary_same_token_negative_top_k = opts.boundary_same_token_negative_top_k,
        .boundary_candidate_rank_loss_weight = opts.boundary_candidate_rank_loss_weight,
        .boundary_candidate_rank_loss_top_k = opts.boundary_candidate_rank_loss_top_k,
        .boundary_candidate_negative_weight = opts.boundary_candidate_negative_weight,
        .boundary_gold_count_rank_loss_weight = opts.boundary_gold_count_rank_loss_weight,
        .boundary_gold_count_rank_loss_margin = opts.boundary_gold_count_rank_loss_margin,
        .boundary_gold_count_rank_loss_negative_multiplier = opts.boundary_gold_count_rank_loss_negative_multiplier,
        .boundary_local_window_loss_weight = opts.boundary_local_window_loss_weight,
        .boundary_local_window_radius = opts.boundary_local_window_radius,
        .boundary_feature_mode = fused_chunker_train.boundaryFeatureModeName(opts.boundary_feature_mode),
        .boundary_head_input_dim = fused_chunker_train.boundaryFeatureDim(
            opts.boundary_feature_mode,
            @intCast(opts.hidden_size),
        ),
        .boundary_dropout = if (opts.deterministic) 0.0 else opts.boundary_dropout,
        .configured_neftune_alpha = if (opts.deterministic) 0.0 else opts.neftune_alpha,
        .encoder_neftune = opts.encoder_neftune and !opts.deterministic,
        .encoder_neftune_alpha = if (opts.encoder_neftune and !opts.deterministic) opts.neftune_alpha else 0.0,
        .boundary_focal_gamma = opts.boundary_focal_gamma,
        .boundary_focal_alpha = opts.boundary_focal_alpha,
        .contrastive_focal_gamma = opts.contrastive_focal_gamma,
        .contrastive_focal_alpha = opts.contrastive_focal_alpha,
        .lambda_chunk = opts.lambda_chunk,
        .lambda_embed = opts.lambda_embed,
        .boundary_focus_epochs = opts.boundary_focus_epochs,
        .boundary_focus_lambda_embed = opts.boundary_focus_lambda_embed,
        .debug_frozen_feature_probe = opts.debug_frozen_feature_probe,
        .debug_frozen_feature_train_examples = opts.debug_frozen_feature_train_examples,
        .debug_frozen_feature_val_examples = opts.debug_frozen_feature_val_examples,
        .debug_frozen_feature_train_offset = opts.debug_frozen_feature_train_offset,
        .debug_frozen_feature_val_offset = opts.debug_frozen_feature_val_offset,
        .debug_frozen_feature_epochs = opts.debug_frozen_feature_epochs,
        .debug_frozen_feature_lr = opts.debug_frozen_feature_lr,
        .train_dataset = train_stats,
        .validation_dataset = val_stats,
        .total_steps = total_steps,
        .final_loss = final_loss,
        .final_checkpoint_path = final_checkpoint_path,
        .best_val_step = best_val_step,
        .best_val_epoch = best_val_epoch,
        .best_val_f1 = best_val_f1,
        .recommended_boundary_threshold_available = boundary_threshold_recommendation.available,
        .recommended_boundary_threshold = boundary_threshold_recommendation.threshold,
        .recommended_boundary_threshold_f1 = boundary_threshold_recommendation.validation_f1,
        .recommended_boundary_threshold_fixed_f1 = boundary_threshold_recommendation.fixed_f1,
        .recommended_boundary_threshold_average_precision = boundary_threshold_recommendation.average_precision,
        .recommended_boundary_threshold_max_rank_f1 = boundary_threshold_recommendation.max_rank_f1,
        .recommended_boundary_threshold_source = boundary_threshold_recommendation.source_event,
        .recommended_boundary_threshold_step = boundary_threshold_recommendation.step,
        .recommended_boundary_threshold_epoch = boundary_threshold_recommendation.epoch,
        .recommended_boundary_threshold_samples = boundary_threshold_recommendation.samples,
        .recommended_boundary_threshold_total_samples = boundary_threshold_recommendation.total_samples,
        .recommended_boundary_threshold_full_validation = boundary_threshold_recommendation.full_validation,
        .peak_resident_bytes = peak_resident_bytes,
        .go_phase20_reference_f1 = 0.786,
        .go_phase20_parity_floor_f1 = 0.766,
    }, .{ .whitespace = .indent_2 }, &writer.interface);
    try writer.interface.writeByte('\n');
    try writer.interface.flush();
}

const BoundaryBatchDebugStats = struct {
    valid_tokens: u64 = 0,
    gold_positives: u64 = 0,
    feature_mean: f32 = 0,
    feature_rms: f32 = 0,
    feature_max_abs: f32 = 0,
};

const FinalNormCheckpointStats = struct {
    weight: fused_chunker_train.BoundaryProbeTensorStats = .{},
    bias: fused_chunker_train.BoundaryProbeTensorStats = .{},
};

const EmbeddingCheckpointStats = struct {
    word_embedding_weight: fused_chunker_train.BoundaryProbeTensorStats = .{},
    layer_norm_weight: fused_chunker_train.BoundaryProbeTensorStats = .{},
    layer_norm_bias: fused_chunker_train.BoundaryProbeTensorStats = .{},
};

fn loadedWeightFloatSlice(loaded: *const LoadedWeight) ?[]const f32 {
    if (loaded.tensor.dtype != .f32) return null;
    const aligned: []align(@alignOf(f32)) const u8 = @alignCast(loaded.tensor.data);
    return std.mem.bytesAsSlice(f32, aligned);
}

fn loadedWeightProbeStats(loaded: *const LoadedWeight) fused_chunker_train.BoundaryProbeTensorStats {
    const values = loadedWeightFloatSlice(loaded) orelse return .{};
    return fused_chunker_train.computeBoundaryProbeTensorStats(values);
}

fn nativeWeightProbeStats(
    weight_store: *native_compute.WeightStore,
    key: []const u8,
) fused_chunker_train.BoundaryProbeTensorStats {
    const loaded = weight_store.resident_weights.getPtr(key) orelse return .{};
    return loadedWeightProbeStats(loaded);
}

fn metalWeightProbeStats(
    weight_store: *MetalWeightStore,
    key: []const u8,
) fused_chunker_train.BoundaryProbeTensorStats {
    if (comptime !build_options.enable_metal) return .{};
    const entry = weight_store.lazy_weights.getPtr(key) orelse return .{};
    const loaded = entry.host_loaded orelse return .{};
    return loadedWeightProbeStats(&loaded);
}

fn nativeWeightFloatSlice(
    weight_store: *native_compute.WeightStore,
    key: []const u8,
) ?[]const f32 {
    const loaded = weight_store.resident_weights.getPtr(key) orelse return null;
    return loadedWeightFloatSlice(loaded);
}

fn metalWeightFloatSlice(
    weight_store: *MetalWeightStore,
    key: []const u8,
) ?[]const f32 {
    if (comptime !build_options.enable_metal) return null;
    const entry = weight_store.lazy_weights.getPtr(key) orelse return null;
    const loaded = entry.host_loaded orelse return null;
    return loadedWeightFloatSlice(&loaded);
}

fn allocEmbeddingTableRowProbes(
    allocator: std.mem.Allocator,
    input_ids: []const i32,
    total_tokens: usize,
    hidden_size: usize,
    values: ?[]const f32,
) ![]EmbeddingRowProbe {
    const count = @min(@min(input_ids.len, total_tokens), embedding_row_probe_max);
    const probes = try allocator.alloc(EmbeddingRowProbe, count);
    for (probes, 0..) |*probe, i| {
        const token_id = input_ids[i];
        probe.* = .{ .position = i, .token_id = token_id };
        if (values) |weight_values| {
            if (token_id >= 0) {
                const row_idx: usize = @intCast(token_id);
                const start = row_idx * hidden_size;
                const end = start + hidden_size;
                if (end <= weight_values.len) {
                    probe.stats = fused_chunker_train.computeBoundaryProbeTensorStats(weight_values[start..end]);
                }
            }
        }
    }
    return probes;
}

fn allocEmbeddingLookupRowProbes(
    allocator: std.mem.Allocator,
    input_ids: []const i32,
    total_tokens: usize,
    hidden_size: usize,
    lookup_values: ?[]const f32,
) ![]EmbeddingRowProbe {
    const count = @min(@min(input_ids.len, total_tokens), embedding_row_probe_max);
    const probes = try allocator.alloc(EmbeddingRowProbe, count);
    for (probes, 0..) |*probe, i| {
        const token_id = input_ids[i];
        probe.* = .{ .position = i, .token_id = token_id };
        if (lookup_values) |values| {
            const start = i * hidden_size;
            const end = start + hidden_size;
            if (end <= values.len) {
                probe.stats = fused_chunker_train.computeBoundaryProbeTensorStats(values[start..end]);
            }
        }
    }
    return probes;
}

fn activeEmbeddingWeightValues(
    use_metal: bool,
    weight_store: *native_compute.WeightStore,
    metal_weight_store: *MetalWeightStore,
) ?[]const f32 {
    if (use_metal) {
        return metalWeightFloatSlice(metal_weight_store, "model.embeddings.tok_embeddings.weight");
    }
    return nativeWeightFloatSlice(weight_store, "model.embeddings.tok_embeddings.weight");
}

fn computeFinalNormCheckpointStats(
    use_metal: bool,
    weight_store: *native_compute.WeightStore,
    metal_weight_store: *MetalWeightStore,
) FinalNormCheckpointStats {
    var out: FinalNormCheckpointStats = if (use_metal)
        .{
            .weight = metalWeightProbeStats(metal_weight_store, "model.final_norm.weight"),
            .bias = metalWeightProbeStats(metal_weight_store, "model.final_norm.bias"),
        }
    else
        .{
            .weight = nativeWeightProbeStats(weight_store, "model.final_norm.weight"),
            .bias = nativeWeightProbeStats(weight_store, "model.final_norm.bias"),
        };
    if (out.bias.elems == 0 and out.weight.elems > 0) {
        out.bias = fused_chunker_train.computeZeroBoundaryProbeTensorStats(@intCast(out.weight.elems));
    }
    return out;
}

fn computeEmbeddingCheckpointStats(
    use_metal: bool,
    weight_store: *native_compute.WeightStore,
    metal_weight_store: *MetalWeightStore,
) EmbeddingCheckpointStats {
    var out: EmbeddingCheckpointStats = if (use_metal)
        .{
            .word_embedding_weight = metalWeightProbeStats(metal_weight_store, "model.embeddings.tok_embeddings.weight"),
            .layer_norm_weight = metalWeightProbeStats(metal_weight_store, "model.embeddings.norm.weight"),
            .layer_norm_bias = metalWeightProbeStats(metal_weight_store, "model.embeddings.norm.bias"),
        }
    else
        .{
            .word_embedding_weight = nativeWeightProbeStats(weight_store, "model.embeddings.tok_embeddings.weight"),
            .layer_norm_weight = nativeWeightProbeStats(weight_store, "model.embeddings.norm.weight"),
            .layer_norm_bias = nativeWeightProbeStats(weight_store, "model.embeddings.norm.bias"),
        };
    if (out.layer_norm_bias.elems == 0 and out.layer_norm_weight.elems > 0) {
        out.layer_norm_bias = fused_chunker_train.computeZeroBoundaryProbeTensorStats(@intCast(out.layer_norm_weight.elems));
    }
    return out;
}

fn computeBoundaryBatchDebugStats(
    features: []const f32,
    boundary_labels: []const f32,
    attention_mask: []const f32,
    total_tokens: usize,
    hidden_size: usize,
) BoundaryBatchDebugStats {
    var out = BoundaryBatchDebugStats{};
    var feature_sum: f64 = 0;
    var feature_sq_sum: f64 = 0;
    var feature_count: u64 = 0;

    for (0..total_tokens) |t| {
        if (attention_mask[t] <= 0.5) continue;
        out.valid_tokens += 1;
        if (boundary_labels[t * 2 + 1] > 0.5) out.gold_positives += 1;
        const row = features[t * hidden_size .. (t + 1) * hidden_size];
        for (row) |value| {
            if (!std.math.isFinite(value)) continue;
            const v: f64 = @floatCast(value);
            feature_sum += v;
            feature_sq_sum += v * v;
            out.feature_max_abs = @max(out.feature_max_abs, @abs(value));
            feature_count += 1;
        }
    }

    if (feature_count > 0) {
        const denom = @as(f64, @floatFromInt(feature_count));
        out.feature_mean = @floatCast(feature_sum / denom);
        out.feature_rms = @floatCast(@sqrt(feature_sq_sum / denom));
    }
    return out;
}

fn runBoundaryHeadFrozenOverfitProbe(
    allocator: std.mem.Allocator,
    trainer: *fused_chunker_train.FusedTrainer,
    features: []const f32,
    boundary_labels: []const f32,
    attention_mask: []const f32,
    chunk_embeddings: []const f32,
    chunk_mask: []const f32,
    doc_ids: []const u32,
    total_tokens: usize,
    batch_size: usize,
    max_chunks: usize,
    hidden_size: usize,
    steps: u32,
    lr: f32,
) !void {
    const stats = computeBoundaryBatchDebugStats(features, boundary_labels, attention_mask, total_tokens, hidden_size);
    const gold_rate = if (stats.valid_tokens > 0)
        @as(f32, @floatFromInt(stats.gold_positives)) / @as(f32, @floatFromInt(stats.valid_tokens))
    else
        0.0;

    const saved_loss_config = trainer.loss_config;
    const saved_lr_schedule = trainer.lr_schedule;
    const saved_config = trainer.config;
    trainer.loss_config.lambda_embed = 0.0;
    trainer.lr_schedule = .{ .constant = lr };
    trainer.config.grad_accum_steps = 1;
    defer {
        trainer.loss_config = saved_loss_config;
        trainer.lr_schedule = saved_lr_schedule;
        trainer.config = saved_config;
    }

    const before = try trainer.debugBoundaryStep(
        allocator,
        features,
        boundary_labels,
        attention_mask,
        null,
        null,
        total_tokens,
        false,
    );
    const before_gap = before.eval_mean_prob_gold_positive - before.eval_mean_prob_gold_negative;

    var final_boundary_loss: f32 = before.boundary_loss;
    var final_total_loss: f32 = before.boundary_loss;
    var final_step: u32 = trainer.step_count;
    for (0..steps) |_| {
        const summary = try trainer.trainStep(
            allocator,
            features,
            boundary_labels,
            attention_mask,
            null,
            null,
            chunk_embeddings,
            chunk_mask,
            doc_ids,
            total_tokens,
            batch_size,
            max_chunks,
            hidden_size,
        );
        final_boundary_loss = summary.boundary_loss;
        final_total_loss = summary.total_loss;
        final_step = summary.step;
    }

    const after = try trainer.debugBoundaryStep(
        allocator,
        features,
        boundary_labels,
        attention_mask,
        null,
        null,
        total_tokens,
        false,
    );
    const after_gap = after.eval_mean_prob_gold_positive - after.eval_mean_prob_gold_negative;

    print(
        "boundary_head_frozen_overfit status=complete steps={d} lr={d:.8} final_step={d} total_tokens={d} valid={d} gold={d} gold_rate={d:.6} feature_mean={d:.6} feature_rms={d:.6} feature_max_abs={d:.6} before_loss={d:.6} before_f1={d:.6} before_tp={d} before_fp={d} before_fn={d} before_predicted_positives={d} before_prob_gold_pos={d:.6} before_prob_gold_neg={d:.6} before_prob_gap={d:.6} final_train_boundary_loss={d:.6} final_train_total_loss={d:.6} after_loss={d:.6} after_f1={d:.6} after_tp={d} after_fp={d} after_fn={d} after_predicted_positives={d} after_prob_gold_pos={d:.6} after_prob_gold_neg={d:.6} after_prob_gap={d:.6}\n",
        .{
            steps,
            lr,
            final_step,
            total_tokens,
            stats.valid_tokens,
            stats.gold_positives,
            gold_rate,
            stats.feature_mean,
            stats.feature_rms,
            stats.feature_max_abs,
            before.boundary_loss,
            before.eval_f1,
            before.eval_tp,
            before.eval_fp,
            before.eval_fn,
            before.eval_predicted_positives,
            before.eval_mean_prob_gold_positive,
            before.eval_mean_prob_gold_negative,
            before_gap,
            final_boundary_loss,
            final_total_loss,
            after.boundary_loss,
            after.eval_f1,
            after.eval_tp,
            after.eval_fp,
            after.eval_fn,
            after.eval_predicted_positives,
            after.eval_mean_prob_gold_positive,
            after.eval_mean_prob_gold_negative,
            after_gap,
        },
    );
}

const FrozenFeatureProbeBatch = struct {
    features: []f32,
    boundary_labels: []f32,
    attention_mask: []f32,
    chunk_embeddings: []f32,
    chunk_mask: []f32,
    doc_ids: []u32,
    total_tokens: usize,
    batch_size: usize,
    max_chunks: usize,
    hidden_size: usize,
    valid_tokens: u64,
    gold_positives: u64,

    fn deinit(self: *FrozenFeatureProbeBatch, allocator: std.mem.Allocator) void {
        allocator.free(self.features);
        allocator.free(self.boundary_labels);
        allocator.free(self.attention_mask);
        allocator.free(self.chunk_embeddings);
        allocator.free(self.chunk_mask);
        allocator.free(self.doc_ids);
        self.* = undefined;
    }
};

const FrozenFeatureProbeCache = struct {
    batches: std.ArrayListUnmanaged(FrozenFeatureProbeBatch) = .empty,
    examples: usize = 0,
    start_offset: usize = 0,
    valid_tokens: u64 = 0,
    gold_positives: u64 = 0,

    fn deinit(self: *FrozenFeatureProbeCache, allocator: std.mem.Allocator) void {
        for (self.batches.items) |*batch| batch.deinit(allocator);
        self.batches.deinit(allocator);
        self.* = undefined;
    }
};

fn frozenProbeGoldRate(valid_tokens: u64, gold_positives: u64) f32 {
    return if (valid_tokens == 0)
        0.0
    else
        @as(f32, @floatFromInt(gold_positives)) / @as(f32, @floatFromInt(valid_tokens));
}

fn materializeFrozenFeatureProbeCache(
    allocator: std.mem.Allocator,
    opts: *const Options,
    cb: *const ComputeBackend,
    tokenizer: *TokenizerBatch,
    samples: []const fused_chunker_data.FusedSample,
    max_examples: usize,
    start_offset: usize,
    label: []const u8,
) !FrozenFeatureProbeCache {
    if (start_offset > samples.len) return error.DebugFrozenFeatureOffsetOutOfRange;

    var cache = FrozenFeatureProbeCache{ .start_offset = start_offset };
    errdefer cache.deinit(allocator);

    const available_examples = samples.len - start_offset;
    const capped_examples = @min(max_examples, available_examples);
    if (capped_examples == 0) return cache;

    const start_ns = nowNs();
    const bs: usize = @max(@as(usize, 1), @as(usize, @intCast(opts.batch_size)));
    const max_seq: usize = @intCast(opts.max_seq_len);
    const max_chunks: usize = @intCast(opts.max_chunks);
    const hidden_size: usize = @intCast(opts.hidden_size);
    const total_batches = (capped_examples + bs - 1) / bs;

    var sample_idx: usize = 0;
    var batch_idx: usize = 0;
    while (sample_idx < capped_examples) {
        const end = @min(sample_idx + bs, capped_examples);
        const count = end - sample_idx;

        const indices = try allocator.alloc(usize, count);
        defer allocator.free(indices);
        for (0..count) |k| indices[k] = start_offset + sample_idx + k;

        var tok_ctx = tokenizer.makeTokenFnCtx();
        var batch = try fused_chunker_data.assembleTokenBatch(
            allocator,
            samples,
            indices,
            max_seq,
            max_chunks,
            &tok_ctx,
            TokenFnCtx.call,
        );
        defer batch.deinit(allocator);

        const actual_batch = batch.batch_size;
        const total_tokens = actual_batch * max_seq;

        const ids_i64 = try allocator.alloc(i64, total_tokens);
        defer allocator.free(ids_i64);
        for (batch.input_ids[0..total_tokens], ids_i64) |id32, *id64| id64.* = @intCast(id32);

        const mask_i64 = try allocator.alloc(i64, total_tokens);
        defer allocator.free(mask_i64);
        for (batch.attention_mask[0..total_tokens], mask_i64) |m32, *m64| m64.* = @intCast(m32);

        const bert_config = modern_bert.Config{
            .hidden_size = opts.hidden_size,
            .num_hidden_layers = opts.num_layers,
            .intermediate_size = opts.intermediate_size,
            .lora_rank = 0,
            .lora_alpha = 0.0,
            .neftune_alpha = 0.0,
            .neftune_seed = 0,
        };

        var features_owned: ?[]f32 = try modern_bert.forward(
            cb,
            allocator,
            bert_config,
            ids_i64,
            mask_i64,
            actual_batch,
            max_seq,
        );
        errdefer if (features_owned) |features| allocator.free(features);

        var boundary_labels_owned: ?[]f32 = try allocator.alloc(f32, total_tokens * 2);
        errdefer if (boundary_labels_owned) |labels| allocator.free(labels);
        for (0..total_tokens) |t| {
            const is_boundary = batch.boundary_labels[t] > 0.5;
            boundary_labels_owned.?[t * 2 + 0] = if (is_boundary) 0.0 else 1.0;
            boundary_labels_owned.?[t * 2 + 1] = if (is_boundary) 1.0 else 0.0;
        }

        var attention_mask_owned: ?[]f32 = try allocator.alloc(f32, total_tokens);
        errdefer if (attention_mask_owned) |mask| allocator.free(mask);
        for (batch.attention_mask[0..total_tokens], attention_mask_owned.?) |m, *out| {
            out.* = @floatFromInt(m);
        }

        var chunk_embeddings_owned: ?[]f32 = try fused_chunker_data.meanPoolChunkEmbeddings(
            allocator,
            features_owned.?,
            &batch,
            hidden_size,
        );
        errdefer if (chunk_embeddings_owned) |embeddings| allocator.free(embeddings);

        const chunk_count = actual_batch * max_chunks;
        var chunk_mask_owned: ?[]f32 = try allocator.dupe(f32, batch.chunk_mask[0..chunk_count]);
        errdefer if (chunk_mask_owned) |mask| allocator.free(mask);

        var doc_ids_owned: ?[]u32 = try allocator.alloc(u32, chunk_count);
        errdefer if (doc_ids_owned) |doc_ids| allocator.free(doc_ids);
        for (0..actual_batch) |b_idx| {
            for (0..max_chunks) |c_idx| {
                doc_ids_owned.?[b_idx * max_chunks + c_idx] = @intCast(start_offset + sample_idx + b_idx);
            }
        }

        const stats = computeBoundaryBatchDebugStats(
            features_owned.?,
            boundary_labels_owned.?,
            attention_mask_owned.?,
            total_tokens,
            hidden_size,
        );

        try cache.batches.append(allocator, .{
            .features = features_owned.?,
            .boundary_labels = boundary_labels_owned.?,
            .attention_mask = attention_mask_owned.?,
            .chunk_embeddings = chunk_embeddings_owned.?,
            .chunk_mask = chunk_mask_owned.?,
            .doc_ids = doc_ids_owned.?,
            .total_tokens = total_tokens,
            .batch_size = actual_batch,
            .max_chunks = max_chunks,
            .hidden_size = hidden_size,
            .valid_tokens = stats.valid_tokens,
            .gold_positives = stats.gold_positives,
        });
        features_owned = null;
        boundary_labels_owned = null;
        attention_mask_owned = null;
        chunk_embeddings_owned = null;
        chunk_mask_owned = null;
        doc_ids_owned = null;

        cache.examples += count;
        cache.valid_tokens += stats.valid_tokens;
        cache.gold_positives += stats.gold_positives;

        batch_idx += 1;
        sample_idx = end;
        print(
            "frozen_feature_probe_cache split={s} offset={d} batch={d}/{d} examples={d}/{d} valid={d} gold={d} elapsed_ms={d:.2}\n",
            .{
                label,
                start_offset,
                batch_idx,
                total_batches,
                cache.examples,
                capped_examples,
                cache.valid_tokens,
                cache.gold_positives,
                nsToMs(elapsedNs(start_ns)),
            },
        );
    }

    print(
        "frozen_feature_probe_cache split={s} status=complete offset={d} examples={d} batches={d} valid={d} gold={d} gold_rate={d:.8} elapsed_ms={d:.2}\n",
        .{
            label,
            start_offset,
            cache.examples,
            cache.batches.items.len,
            cache.valid_tokens,
            cache.gold_positives,
            frozenProbeGoldRate(cache.valid_tokens, cache.gold_positives),
            nsToMs(elapsedNs(start_ns)),
        },
    );
    return cache;
}

fn evaluateFrozenFeatureProbeCache(
    allocator: std.mem.Allocator,
    trainer: *fused_chunker_train.FusedTrainer,
    cache: *const FrozenFeatureProbeCache,
) !fused_chunker_train.EvalSummary {
    var acc = fused_chunker_train.BoundaryEvalAccumulator{
        .calibrated_threshold = fused_chunker_loss.weightedCePositiveThreshold(trainer.loss_config.pos_weight),
    };
    defer acc.deinit(allocator);

    for (cache.batches.items) |*batch| {
        try trainer.evaluateBatchInto(
            allocator,
            &acc,
            batch.features,
            batch.boundary_labels,
            batch.attention_mask,
            batch.total_tokens,
        );
    }

    return try acc.finish(allocator);
}

fn printFrozenFeatureProbeEval(label: []const u8, summary: fused_chunker_train.EvalSummary) void {
    const prob_gap = summary.mean_positive_probability_gold_positive - summary.mean_positive_probability_gold_negative;
    print(
        "{s} boundary_f1={d:.6} best_f1={d:.6} best_threshold={d:.6} max_rank_f1={d:.6} ap={d:.6} valid={d} gold={d} gold_rate={d:.8} pred_rate={d:.8} prob_pos={d:.6} prob_neg={d:.6} prob_gap={d:.6}\n",
        .{
            label,
            summary.boundary_f1,
            summary.best_boundary_f1,
            summary.best_boundary_threshold,
            summary.max_rank_f1,
            summary.average_precision,
            summary.valid_tokens,
            summary.gold_positives,
            summary.gold_positive_rate,
            summary.predicted_positive_rate,
            summary.mean_positive_probability_gold_positive,
            summary.mean_positive_probability_gold_negative,
            prob_gap,
        },
    );
    fused_chunker_train.printBoundaryQualityDiagnostics(label, summary);
}

fn runFrozenFeatureProbe(
    allocator: std.mem.Allocator,
    opts: *const Options,
    metrics_writer: anytype,
    trainer: *fused_chunker_train.FusedTrainer,
    cb: *const ComputeBackend,
    tokenizer_opt: ?*TokenizerBatch,
    samples: []const fused_chunker_data.FusedSample,
    val_samples: []const fused_chunker_data.FusedSample,
    encoder_loaded: bool,
) !void {
    if (!encoder_loaded) return error.DebugFrozenFeatureProbeRequiresEncoder;
    const tokenizer = tokenizer_opt orelse return error.DebugFrozenFeatureProbeRequiresTokenizer;
    if (opts.debug_frozen_feature_epochs == 0) return error.InvalidDebugFrozenFeatureEpochs;
    if (opts.debug_frozen_feature_train_examples == 0) return error.InvalidDebugFrozenFeatureTrainExamples;
    if (opts.lora_rank != 0) return error.DebugFrozenFeatureProbeRequiresNoLoRA;
    if (opts.encoder_neftune and opts.neftune_alpha != 0.0 and !opts.deterministic) return error.DebugFrozenFeatureProbeRequiresNoEncoderNeftune;

    const saved_loss_config = trainer.loss_config;
    const saved_lr_schedule = trainer.lr_schedule;
    const saved_config = trainer.config;
    const saved_xbm = trainer.xbm;
    trainer.loss_config.lambda_embed = 0.0;
    trainer.loss_config.use_mrl = false;
    trainer.loss_config.enable_splade = false;
    trainer.lr_schedule = .{ .constant = opts.debug_frozen_feature_lr };
    trainer.config.grad_accum_steps = 1;
    trainer.config.xbm_capacity = 0;
    trainer.xbm = null;
    defer {
        trainer.loss_config = saved_loss_config;
        trainer.lr_schedule = saved_lr_schedule;
        trainer.config = saved_config;
        trainer.xbm = saved_xbm;
    }

    print(
        "frozen_feature_probe status=starting train_examples={d} val_examples={d} train_offset={d} val_offset={d} epochs={d} lr={d:.8} batch_size={d} max_seq_len={d} max_chunks={d} hidden={d}\n",
        .{
            opts.debug_frozen_feature_train_examples,
            opts.debug_frozen_feature_val_examples,
            opts.debug_frozen_feature_train_offset,
            opts.debug_frozen_feature_val_offset,
            opts.debug_frozen_feature_epochs,
            opts.debug_frozen_feature_lr,
            opts.batch_size,
            opts.max_seq_len,
            opts.max_chunks,
            opts.hidden_size,
        },
    );

    var train_cache = try materializeFrozenFeatureProbeCache(
        allocator,
        opts,
        cb,
        tokenizer,
        samples,
        opts.debug_frozen_feature_train_examples,
        opts.debug_frozen_feature_train_offset,
        "train",
    );
    defer train_cache.deinit(allocator);
    if (train_cache.examples == 0) return error.DebugFrozenFeatureProbeEmptyTrainCache;

    var val_cache: ?FrozenFeatureProbeCache = null;
    defer if (val_cache) |*cache| cache.deinit(allocator);
    if (val_samples.len > 0 and opts.debug_frozen_feature_val_examples > 0) {
        val_cache = try materializeFrozenFeatureProbeCache(
            allocator,
            opts,
            cb,
            tokenizer,
            val_samples,
            opts.debug_frozen_feature_val_examples,
            opts.debug_frozen_feature_val_offset,
            "val",
        );
    } else {
        print("frozen_feature_probe_cache split=val status=skipped reason=no_validation_examples\n", .{});
    }

    const train_before = try evaluateFrozenFeatureProbeCache(allocator, trainer, &train_cache);
    printFrozenFeatureProbeEval("frozen_feature_probe_train_before", train_before);
    try writeValidationMetric(
        metrics_writer,
        "frozen_feature_probe_train_before",
        0,
        trainer.step_count,
        train_cache.examples,
        train_cache.examples,
        0,
        train_before,
    );
    if (val_cache) |*cache| {
        const val_before = try evaluateFrozenFeatureProbeCache(allocator, trainer, cache);
        printFrozenFeatureProbeEval("frozen_feature_probe_val_before", val_before);
        try writeValidationMetric(
            metrics_writer,
            "frozen_feature_probe_val_before",
            0,
            trainer.step_count,
            cache.examples,
            cache.examples,
            0,
            val_before,
        );
    }

    var final_boundary_loss: f32 = 0.0;
    var final_total_loss: f32 = 0.0;
    const train_start_ns = nowNs();
    for (0..opts.debug_frozen_feature_epochs) |epoch_idx| {
        var epoch_boundary_loss_sum: f64 = 0;
        var epoch_total_loss_sum: f64 = 0;
        var epoch_batches: u64 = 0;
        for (train_cache.batches.items) |*batch| {
            const summary = try trainer.trainStep(
                allocator,
                batch.features,
                batch.boundary_labels,
                batch.attention_mask,
                null,
                null,
                batch.chunk_embeddings,
                batch.chunk_mask,
                batch.doc_ids,
                batch.total_tokens,
                batch.batch_size,
                batch.max_chunks,
                batch.hidden_size,
            );
            final_boundary_loss = summary.boundary_loss;
            final_total_loss = summary.total_loss;
            epoch_boundary_loss_sum += summary.boundary_loss;
            epoch_total_loss_sum += summary.total_loss;
            epoch_batches += 1;
        }
        const denom = @max(epoch_batches, 1);
        print(
            "frozen_feature_probe_epoch epoch={d}/{d} batches={d} avg_boundary_loss={d:.6} avg_total_loss={d:.6} final_step={d} elapsed_ms={d:.2}\n",
            .{
                epoch_idx + 1,
                opts.debug_frozen_feature_epochs,
                epoch_batches,
                @as(f32, @floatCast(epoch_boundary_loss_sum / @as(f64, @floatFromInt(denom)))),
                @as(f32, @floatCast(epoch_total_loss_sum / @as(f64, @floatFromInt(denom)))),
                trainer.step_count,
                nsToMs(elapsedNs(train_start_ns)),
            },
        );
    }

    const train_after = try evaluateFrozenFeatureProbeCache(allocator, trainer, &train_cache);
    printFrozenFeatureProbeEval("frozen_feature_probe_train_after", train_after);
    var val_after_opt: ?fused_chunker_train.EvalSummary = null;
    try writeValidationMetric(
        metrics_writer,
        "frozen_feature_probe_train_after",
        opts.debug_frozen_feature_epochs - 1,
        trainer.step_count,
        train_cache.examples,
        train_cache.examples,
        0,
        train_after,
    );
    if (val_cache) |*cache| {
        const val_after = try evaluateFrozenFeatureProbeCache(allocator, trainer, cache);
        val_after_opt = val_after;
        printFrozenFeatureProbeEval("frozen_feature_probe_val_after", val_after);
        try writeValidationMetric(
            metrics_writer,
            "frozen_feature_probe_val_after",
            opts.debug_frozen_feature_epochs - 1,
            trainer.step_count,
            cache.examples,
            cache.examples,
            0,
            val_after,
        );
    }

    const val_after_f1 = if (val_after_opt) |summary| summary.best_boundary_f1 else 0.0;
    const val_after_rank_f1 = if (val_after_opt) |summary| summary.max_rank_f1 else 0.0;
    const val_after_ap = if (val_after_opt) |summary| summary.average_precision else 0.0;
    print(
        "frozen_feature_probe status=complete train_offset={d} val_offset={d} epochs={d} lr={d:.8} final_step={d} final_boundary_loss={d:.6} final_total_loss={d:.6} train_best_f1={d:.6} train_max_rank_f1={d:.6} train_ap={d:.6} val_best_f1={d:.6} val_max_rank_f1={d:.6} val_ap={d:.6}\n",
        .{
            train_cache.start_offset,
            if (val_cache) |cache| cache.start_offset else opts.debug_frozen_feature_val_offset,
            opts.debug_frozen_feature_epochs,
            opts.debug_frozen_feature_lr,
            trainer.step_count,
            final_boundary_loss,
            final_total_loss,
            train_after.best_boundary_f1,
            train_after.max_rank_f1,
            train_after.average_precision,
            val_after_f1,
            val_after_rank_f1,
            val_after_ap,
        },
    );
    try writeFrozenFeatureProbeCompleteMetric(
        metrics_writer,
        train_cache.examples,
        if (val_cache) |cache| cache.examples else 0,
        train_cache.start_offset,
        if (val_cache) |cache| cache.start_offset else opts.debug_frozen_feature_val_offset,
        opts.debug_frozen_feature_epochs,
        opts.debug_frozen_feature_lr,
        trainer.step_count,
        final_boundary_loss,
        final_total_loss,
        train_after,
        val_after_opt,
    );
}

fn printIndexSlice(label: []const u8, values: []const usize) void {
    print("{s}=[", .{label});
    for (values, 0..) |value, i| {
        if (i > 0) print(",", .{});
        print("{d}", .{value});
    }
    print("]\n", .{});
}

fn resetIdentityIndices(indices: []usize) void {
    for (indices, 0..) |*idx, i| idx.* = i;
}

fn shuffleIndicesFisherYates(indices: []usize, rng: std.Random) void {
    var i: usize = indices.len;
    while (i > 1) {
        i -= 1;
        const j = rng.uintLessThan(usize, i + 1);
        const tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

fn printPerSampleBoundaryDebug(
    prefix: []const u8,
    batch: *const fused_chunker_data.FusedBatch,
    max_seq: usize,
) void {
    print("{s} per_sample=", .{prefix});
    for (0..batch.batch_size) |b_idx| {
        var valid: u64 = 0;
        var gold: u64 = 0;
        const base = b_idx * max_seq;
        for (0..max_seq) |t| {
            if (batch.attention_mask[base + t] != 0) valid += 1;
            if (batch.boundary_labels[base + t] > 0.5) gold += 1;
        }
        if (b_idx > 0) print(",", .{});
        print("{d}:valid={d}:gold={d}", .{ b_idx, valid, gold });
    }
    print("\n", .{});
}

const BoundaryPosWeightEstimate = struct {
    examples: usize = 0,
    batches: usize = 0,
    valid_tokens: u64 = 0,
    gold_tokens: u64 = 0,
    gold_rate: f32 = 0.0,
    balanced_pos_weight: f32 = 1.0,
};

fn estimateBoundaryPosWeight(
    allocator: std.mem.Allocator,
    samples: []const fused_chunker_data.FusedSample,
    tokenizer: *TokenizerBatch,
    max_examples: usize,
    batch_size: usize,
    max_seq: usize,
    max_chunks: usize,
) !BoundaryPosWeightEstimate {
    const capped_examples = if (max_examples == 0) samples.len else @min(samples.len, max_examples);
    if (capped_examples == 0) return error.EmptyBoundaryPosWeightEstimate;

    const bs = @max(@as(usize, 1), batch_size);
    var indices = try allocator.alloc(usize, bs);
    defer allocator.free(indices);

    var out = BoundaryPosWeightEstimate{ .examples = capped_examples };
    var sample_idx: usize = 0;
    while (sample_idx < capped_examples) {
        const count = @min(bs, capped_examples - sample_idx);
        for (0..count) |i| indices[i] = sample_idx + i;

        var tok_ctx = tokenizer.makeTokenFnCtx();
        var batch = try fused_chunker_data.assembleTokenBatch(
            allocator,
            samples,
            indices[0..count],
            max_seq,
            max_chunks,
            &tok_ctx,
            TokenFnCtx.call,
        );
        defer batch.deinit(allocator);

        const total_tokens = count * max_seq;
        for (0..total_tokens) |t| {
            if (batch.attention_mask[t] == 0) continue;
            out.valid_tokens += 1;
            if (batch.boundary_labels[t] > 0.5) out.gold_tokens += 1;
        }

        out.batches += 1;
        sample_idx += count;
    }

    if (out.valid_tokens == 0) return error.EmptyBoundaryPosWeightEstimate;
    if (out.gold_tokens == 0) return error.NoBoundaryTokensForPosWeightEstimate;

    const gold_f: f32 = @floatFromInt(out.gold_tokens);
    const valid_f: f32 = @floatFromInt(out.valid_tokens);
    out.gold_rate = gold_f / valid_f;
    out.balanced_pos_weight = @max(1.0, (valid_f - gold_f) / gold_f);
    return out;
}

// ---------------------------------------------------------------------------
// Dummy token function (placeholder until tokenizer is wired in)
// ---------------------------------------------------------------------------

/// Fills out_ids and out_mask with zeros and returns 0 tokens produced.
/// This is a placeholder used when no tokenizer is loaded.
fn dummyTokenFn(_: void, text: []const u8, out_ids: []i32, out_mask: []i32, out_offsets: ?[][2]u32) usize {
    _ = text;
    _ = out_offsets;
    @memset(out_ids, 0);
    @memset(out_mask, 0);
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next(); // skip binary name

    // Parse CLI
    var data_path: ?[]const u8 = null;
    var output_dir: ?[]const u8 = null;
    var val_data_path: ?[]const u8 = null;
    var model_dir: ?[]const u8 = null;
    var epochs: u32 = 10;
    var batch_size: u32 = 16;
    var learning_rate: f32 = 1e-4;
    var boundary_head_lr_multiplier: f32 = 1.0;
    var warmup_steps: u32 = 200;
    var lr_total_steps: u32 = 0;
    var weight_decay: f32 = 0.01;
    var max_grad_norm: f32 = 0.0;
    var beta1: f32 = 0.9;
    var beta2: f32 = 0.999;
    var adam_epsilon: f32 = 1e-8;
    var lambda_chunk: f32 = 1.0;
    var lambda_embed: f32 = 0.3;
    var boundary_focus_epochs: u32 = 3;
    var boundary_focus_lambda_embed: f32 = 0.1;
    var boundary_focal_gamma: f32 = 2.0;
    var boundary_focal_alpha: f32 = 0.75;
    var contrastive_focal_gamma: f32 = 0.0;
    var contrastive_focal_alpha: f32 = 0.75;
    var boundary_pos_weight: f32 = 5.0;
    var boundary_pos_weight_auto: bool = false;
    var boundary_pos_weight_auto_max_examples: usize = 2048;
    var boundary_rank_loss_weight: f32 = 0.0;
    var boundary_rank_loss_margin: f32 = 1.0;
    var boundary_rank_loss_top_k: u32 = 1;
    var boundary_same_token_rank_loss_weight: f32 = 0.0;
    var boundary_same_token_rank_loss_top_k: u32 = 4;
    var boundary_same_token_negative_weight: f32 = 1.0;
    var boundary_same_token_negative_top_k: u32 = 0;
    var boundary_candidate_rank_loss_weight: f32 = 0.0;
    var boundary_candidate_rank_loss_top_k: u32 = 8;
    var boundary_candidate_negative_weight: f32 = 1.0;
    var boundary_gold_count_rank_loss_weight: f32 = 0.0;
    var boundary_gold_count_rank_loss_margin: f32 = 1.0;
    var boundary_gold_count_rank_loss_negative_multiplier: u32 = 1;
    var boundary_local_window_loss_weight: f32 = 0.0;
    var boundary_local_window_radius: u32 = 12;
    var boundary_feature_mode: fused_chunker_train.BoundaryFeatureMode = .token;
    var boundary_dropout: f32 = 0.1;
    var boundary_loss_type: BoundaryLossType = .ce;
    var hidden_size: u32 = 768;
    var num_layers: u32 = 22;
    var max_seq_len: u32 = 384;
    var max_chunks: u32 = 32;
    var checkpoint_every: u32 = 0;
    var checkpoint_every_steps: u32 = 0;
    var log_every: u32 = 1;
    var eval_every: u32 = 1;
    var eval_every_steps: u32 = 0;
    var step_eval_max_examples: usize = 0;
    var step_train_eval_max_examples: usize = 0;
    var checkpoint_roundtrip_eval: bool = false;
    var checkpoint_roundtrip_max_examples: usize = 0;
    var max_steps: u64 = 0;
    var split: []const u8 = "train";
    var val_split: []const u8 = "val";
    var seed: u64 = 42;
    var lora_rank: u32 = 0;
    var lora_alpha: f32 = 32.0;
    var lora_start_epoch: u32 = 0;
    var intermediate_size: u32 = 1152;
    var backend: @TypeOf((Options{
        .data_path = "",
        .output_dir = "",
    }).backend) = .auto;
    var grad_accum: u32 = 1;
    var schedule_free: bool = false;
    var neftune_alpha: f32 = 0.0;
    var encoder_neftune: bool = true;
    var xbm_capacity: usize = 0;
    var llrd_decay: f32 = 1.0;
    var lisa_sample_layers: u32 = 0;
    var lisa_top_k: u32 = 5;
    var lora_train_top_k: u32 = 0;
    var encoder_vjp: EncoderVJPMode = .direct;
    var layers_per_segment: u32 = 1;
    var compiled_segment_forward: bool =
        platform.env.getenvBoolDefault("ANTFLY_FUSED_CHUNKER_COMPILED_SEGMENT_FORWARD", false);
    var lora_plus_ratio: f32 = 1.0;
    var length_bucketing: bool = false;
    var bucket_size: usize = 256;
    var mixed_precision: bool = false;
    var splade: bool = false;
    var lambda_splade: f32 = 0.15;
    var lambda_flops: f32 = 3e-5;
    var splade_focus_epoch: u32 = 4;
    var mrl: bool = false;
    var mrl_dims_str: []const u8 = "768,256,128";
    var resume_from: []const u8 = "";
    var save_optimizer_state: bool = false;
    var deterministic: bool = false;
    var go_epoch_shuffle: bool = false;
    var report_to: ?[]const u8 = null;
    var manifest_path: ?[]const u8 = null;
    var boundary_alignment_dump_dir: ?[]const u8 = null;
    var boundary_alignment_dump_top_k: usize = 32;
    var memory_sample_every: u32 = 1;
    var memory_warn_rss_bytes: u64 = 0;
    var memory_abort_rss_bytes: u64 = 0;
    var debug_first_boundary_step: bool = false;
    var debug_first_boundary_step_exit: bool = false;
    var debug_boundary_step: u32 = 0;
    var debug_boundary_step_exit: bool = false;
    var debug_update_step: u32 = 0;
    var debug_update_step_exit: bool = false;
    var debug_boundary_head_overfit_steps: u32 = 0;
    var debug_boundary_head_overfit_lr: f32 = 1e-3;
    var debug_frozen_feature_probe: bool = false;
    var debug_frozen_feature_train_examples: usize = 64;
    var debug_frozen_feature_val_examples: usize = 64;
    var debug_frozen_feature_train_offset: usize = 0;
    var debug_frozen_feature_val_offset: usize = 0;
    var debug_frozen_feature_epochs: u32 = 3;
    var debug_frozen_feature_lr: f32 = 5e-3;
    var debug_step_json_path: ?[]const u8 = null;
    var debug_batch_offset: ?usize = null;
    var debug_encoder_probe_layer: u32 = 0;
    var debug_encoder_layer_inputs_only: bool = false;
    var debug_encoder_replay_input_path: ?[]const u8 = null;
    var debug_encoder_replay_upstream_path: ?[]const u8 = null;
    var debug_layer_backward_decomp: bool = false;
    var debug_qkv_split_vjp: bool = false;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--data")) {
            data_path = args.next() orelse return error.MissingDataPath;
        } else if (std.mem.eql(u8, arg, "--output")) {
            output_dir = args.next() orelse return error.MissingOutputDir;
        } else if (std.mem.eql(u8, arg, "--val-data")) {
            val_data_path = args.next() orelse return error.MissingValDataPath;
        } else if (std.mem.eql(u8, arg, "--model-dir")) {
            model_dir = args.next() orelse return error.MissingModelDir;
        } else if (std.mem.eql(u8, arg, "--epochs")) {
            const val = args.next() orelse return error.MissingEpochs;
            epochs = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--batch-size")) {
            const val = args.next() orelse return error.MissingBatchSize;
            batch_size = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--lr") or std.mem.eql(u8, arg, "--learning-rate")) {
            const val = args.next() orelse return error.MissingLr;
            learning_rate = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--boundary-head-lr-multiplier")) {
            const val = args.next() orelse return error.MissingBoundaryHeadLrMultiplier;
            boundary_head_lr_multiplier = try std.fmt.parseFloat(f32, val);
            if (!std.math.isFinite(boundary_head_lr_multiplier) or boundary_head_lr_multiplier <= 0.0) return error.InvalidBoundaryHeadLrMultiplier;
        } else if (std.mem.eql(u8, arg, "--warmup-steps")) {
            const val = args.next() orelse return error.MissingWarmupSteps;
            warmup_steps = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--lr-total-steps")) {
            const val = args.next() orelse return error.MissingLrTotalSteps;
            lr_total_steps = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--weight-decay")) {
            const val = args.next() orelse return error.MissingWeightDecay;
            weight_decay = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--max-grad-norm")) {
            const val = args.next() orelse return error.MissingMaxGradNorm;
            max_grad_norm = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--beta1")) {
            const val = args.next() orelse return error.MissingBeta1;
            beta1 = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--beta2")) {
            const val = args.next() orelse return error.MissingBeta2;
            beta2 = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--adam-epsilon")) {
            const val = args.next() orelse return error.MissingAdamEpsilon;
            adam_epsilon = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--lambda-chunk")) {
            const val = args.next() orelse return error.MissingLambdaChunk;
            lambda_chunk = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--lambda-embed")) {
            const val = args.next() orelse return error.MissingLambdaEmbed;
            lambda_embed = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--boundary-focus-epochs")) {
            const val = args.next() orelse return error.MissingBoundaryFocusEpochs;
            boundary_focus_epochs = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--boundary-focus-lambda-embed")) {
            const val = args.next() orelse return error.MissingBoundaryFocusLambdaEmbed;
            boundary_focus_lambda_embed = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--boundary-focal-gamma") or std.mem.eql(u8, arg, "--focal-gamma")) {
            const val = args.next() orelse return error.MissingBoundaryFocalGamma;
            boundary_focal_gamma = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--boundary-focal-alpha") or std.mem.eql(u8, arg, "--focal-alpha")) {
            const val = args.next() orelse return error.MissingBoundaryFocalAlpha;
            boundary_focal_alpha = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--contrastive-focal-gamma")) {
            const val = args.next() orelse return error.MissingContrastiveFocalGamma;
            contrastive_focal_gamma = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--contrastive-focal-alpha")) {
            const val = args.next() orelse return error.MissingContrastiveFocalAlpha;
            contrastive_focal_alpha = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--boundary-pos-weight") or std.mem.eql(u8, arg, "--pos-weight")) {
            const val = args.next() orelse return error.MissingBoundaryPosWeight;
            if (std.mem.eql(u8, val, "auto")) {
                boundary_pos_weight_auto = true;
            } else {
                boundary_pos_weight = try std.fmt.parseFloat(f32, val);
                boundary_pos_weight_auto = false;
            }
        } else if (std.mem.eql(u8, arg, "--boundary-pos-weight-auto")) {
            boundary_pos_weight_auto = true;
        } else if (std.mem.eql(u8, arg, "--boundary-pos-weight-auto-max-examples")) {
            const val = args.next() orelse return error.MissingBoundaryPosWeightAutoMaxExamples;
            boundary_pos_weight_auto_max_examples = try std.fmt.parseUnsigned(usize, val, 10);
        } else if (std.mem.eql(u8, arg, "--boundary-rank-loss-weight")) {
            const val = args.next() orelse return error.MissingBoundaryRankLossWeight;
            boundary_rank_loss_weight = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--boundary-rank-loss-margin")) {
            const val = args.next() orelse return error.MissingBoundaryRankLossMargin;
            boundary_rank_loss_margin = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--boundary-rank-loss-top-k")) {
            const val = args.next() orelse return error.MissingBoundaryRankLossTopK;
            boundary_rank_loss_top_k = try std.fmt.parseUnsigned(u32, val, 10);
            if (boundary_rank_loss_top_k == 0) return error.InvalidBoundaryRankLossTopK;
        } else if (std.mem.eql(u8, arg, "--boundary-same-token-rank-loss-weight")) {
            const val = args.next() orelse return error.MissingBoundarySameTokenRankLossWeight;
            boundary_same_token_rank_loss_weight = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--boundary-same-token-rank-loss-top-k")) {
            const val = args.next() orelse return error.MissingBoundarySameTokenRankLossTopK;
            boundary_same_token_rank_loss_top_k = try std.fmt.parseUnsigned(u32, val, 10);
            if (boundary_same_token_rank_loss_top_k == 0) return error.InvalidBoundarySameTokenRankLossTopK;
        } else if (std.mem.eql(u8, arg, "--boundary-same-token-negative-weight")) {
            const val = args.next() orelse return error.MissingBoundarySameTokenNegativeWeight;
            boundary_same_token_negative_weight = try std.fmt.parseFloat(f32, val);
            if (!std.math.isFinite(boundary_same_token_negative_weight) or boundary_same_token_negative_weight < 1.0) return error.InvalidBoundarySameTokenNegativeWeight;
        } else if (std.mem.eql(u8, arg, "--boundary-same-token-negative-top-k")) {
            const val = args.next() orelse return error.MissingBoundarySameTokenNegativeTopK;
            boundary_same_token_negative_top_k = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--boundary-candidate-rank-loss-weight")) {
            const val = args.next() orelse return error.MissingBoundaryCandidateRankLossWeight;
            boundary_candidate_rank_loss_weight = try std.fmt.parseFloat(f32, val);
            if (!std.math.isFinite(boundary_candidate_rank_loss_weight) or boundary_candidate_rank_loss_weight < 0.0) return error.InvalidBoundaryCandidateRankLossWeight;
        } else if (std.mem.eql(u8, arg, "--boundary-candidate-rank-loss-top-k")) {
            const val = args.next() orelse return error.MissingBoundaryCandidateRankLossTopK;
            boundary_candidate_rank_loss_top_k = try std.fmt.parseUnsigned(u32, val, 10);
            if (boundary_candidate_rank_loss_top_k == 0) return error.InvalidBoundaryCandidateRankLossTopK;
        } else if (std.mem.eql(u8, arg, "--boundary-candidate-negative-weight")) {
            const val = args.next() orelse return error.MissingBoundaryCandidateNegativeWeight;
            boundary_candidate_negative_weight = try std.fmt.parseFloat(f32, val);
            if (!std.math.isFinite(boundary_candidate_negative_weight) or boundary_candidate_negative_weight < 1.0) return error.InvalidBoundaryCandidateNegativeWeight;
        } else if (std.mem.eql(u8, arg, "--boundary-gold-count-rank-loss-weight")) {
            const val = args.next() orelse return error.MissingBoundaryGoldCountRankLossWeight;
            boundary_gold_count_rank_loss_weight = try std.fmt.parseFloat(f32, val);
            if (!std.math.isFinite(boundary_gold_count_rank_loss_weight) or boundary_gold_count_rank_loss_weight < 0.0) return error.InvalidBoundaryGoldCountRankLossWeight;
        } else if (std.mem.eql(u8, arg, "--boundary-gold-count-rank-loss-margin")) {
            const val = args.next() orelse return error.MissingBoundaryGoldCountRankLossMargin;
            boundary_gold_count_rank_loss_margin = try std.fmt.parseFloat(f32, val);
            if (!std.math.isFinite(boundary_gold_count_rank_loss_margin) or boundary_gold_count_rank_loss_margin <= 0.0) return error.InvalidBoundaryGoldCountRankLossMargin;
        } else if (std.mem.eql(u8, arg, "--boundary-gold-count-rank-loss-negative-multiplier")) {
            const val = args.next() orelse return error.MissingBoundaryGoldCountRankLossNegativeMultiplier;
            boundary_gold_count_rank_loss_negative_multiplier = try std.fmt.parseUnsigned(u32, val, 10);
            if (boundary_gold_count_rank_loss_negative_multiplier == 0) return error.InvalidBoundaryGoldCountRankLossNegativeMultiplier;
        } else if (std.mem.eql(u8, arg, "--boundary-local-window-loss-weight")) {
            const val = args.next() orelse return error.MissingBoundaryLocalWindowLossWeight;
            boundary_local_window_loss_weight = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--boundary-local-window-radius")) {
            const val = args.next() orelse return error.MissingBoundaryLocalWindowRadius;
            boundary_local_window_radius = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--boundary-feature-mode")) {
            boundary_feature_mode = try parseBoundaryFeatureMode(args.next() orelse return error.MissingBoundaryFeatureMode);
        } else if (std.mem.eql(u8, arg, "--boundary-dropout")) {
            const val = args.next() orelse return error.MissingBoundaryDropout;
            boundary_dropout = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--boundary-loss-type") or std.mem.eql(u8, arg, "--loss-type")) {
            const val = args.next() orelse return error.MissingBoundaryLossType;
            if (std.mem.eql(u8, val, "ce")) {
                boundary_loss_type = .ce;
            } else if (std.mem.eql(u8, val, "focal")) {
                boundary_loss_type = .focal;
            } else {
                return error.InvalidBoundaryLossType;
            }
        } else if (std.mem.eql(u8, arg, "--hidden-size")) {
            const val = args.next() orelse return error.MissingHiddenSize;
            hidden_size = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--num-layers") or std.mem.eql(u8, arg, "--max-layers")) {
            const val = args.next() orelse return error.MissingNumLayers;
            num_layers = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--max-seq-len")) {
            const val = args.next() orelse return error.MissingMaxSeqLen;
            max_seq_len = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--max-chunks")) {
            const val = args.next() orelse return error.MissingMaxChunks;
            max_chunks = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--checkpoint-every")) {
            const val = args.next() orelse return error.MissingCheckpointEvery;
            checkpoint_every = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--checkpoint-every-steps")) {
            const val = args.next() orelse return error.MissingCheckpointEvery;
            checkpoint_every_steps = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--log-every")) {
            const val = args.next() orelse return error.MissingLogEvery;
            log_every = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--eval-every")) {
            const val = args.next() orelse return error.MissingEvalEvery;
            eval_every = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--eval-every-steps")) {
            const val = args.next() orelse return error.MissingEvalEvery;
            eval_every_steps = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--step-eval-max-examples")) {
            const val = args.next() orelse return error.MissingStepEvalMaxExamples;
            step_eval_max_examples = try std.fmt.parseUnsigned(usize, val, 10);
        } else if (std.mem.eql(u8, arg, "--step-train-eval-max-examples")) {
            const val = args.next() orelse return error.MissingStepTrainEvalMaxExamples;
            step_train_eval_max_examples = try std.fmt.parseUnsigned(usize, val, 10);
        } else if (std.mem.eql(u8, arg, "--checkpoint-roundtrip-eval")) {
            checkpoint_roundtrip_eval = true;
        } else if (std.mem.eql(u8, arg, "--checkpoint-roundtrip-max-examples")) {
            const val = args.next() orelse return error.MissingCheckpointRoundtripMaxExamples;
            checkpoint_roundtrip_max_examples = try std.fmt.parseUnsigned(usize, val, 10);
        } else if (std.mem.eql(u8, arg, "--max-steps")) {
            const val = args.next() orelse return error.MissingMaxSteps;
            max_steps = try std.fmt.parseUnsigned(u64, val, 10);
        } else if (std.mem.eql(u8, arg, "--split")) {
            split = args.next() orelse return error.MissingSplit;
        } else if (std.mem.eql(u8, arg, "--val-split")) {
            val_split = args.next() orelse return error.MissingValSplit;
        } else if (std.mem.eql(u8, arg, "--seed")) {
            const val = args.next() orelse return error.MissingSeed;
            seed = try std.fmt.parseUnsigned(u64, val, 10);
        } else if (std.mem.eql(u8, arg, "--lora-rank")) {
            const val = args.next() orelse return error.MissingLoraRank;
            lora_rank = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--lora-alpha")) {
            const val = args.next() orelse return error.MissingLoraAlpha;
            lora_alpha = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--lora-start-epoch")) {
            const val = args.next() orelse return error.MissingLoRAStartEpoch;
            lora_start_epoch = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--intermediate-size")) {
            const val = args.next() orelse return error.MissingIntermediateSize;
            intermediate_size = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--backend")) {
            const val = args.next() orelse return error.MissingBackend;
            if (std.mem.eql(u8, val, "native")) {
                backend = .native;
            } else if (std.mem.eql(u8, val, "metal")) {
                backend = .metal;
            } else if (std.mem.eql(u8, val, "auto")) {
                backend = .auto;
            } else {
                print("error: unknown backend '{s}': expected native, metal, or auto\n", .{val});
                std.process.exit(1);
            }
        } else if (std.mem.eql(u8, arg, "--grad-accum")) {
            const val = args.next() orelse return error.MissingGradAccum;
            grad_accum = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--schedule-free")) {
            schedule_free = true;
        } else if (std.mem.eql(u8, arg, "--optimizer")) {
            const val = args.next() orelse return error.MissingOptimizer;
            if (std.mem.eql(u8, val, "adamw")) {
                schedule_free = false;
            } else if (std.mem.eql(u8, val, "schedule-free")) {
                schedule_free = true;
            } else {
                return error.InvalidOptimizer;
            }
        } else if (std.mem.eql(u8, arg, "--neftune-alpha")) {
            const val = args.next() orelse return error.MissingNeftuneAlpha;
            neftune_alpha = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--encoder-neftune")) {
            encoder_neftune = true;
        } else if (std.mem.eql(u8, arg, "--disable-encoder-neftune")) {
            encoder_neftune = false;
        } else if (std.mem.eql(u8, arg, "--xbm-capacity")) {
            const val = args.next() orelse return error.MissingXbmCapacity;
            xbm_capacity = try std.fmt.parseUnsigned(usize, val, 10);
        } else if (std.mem.eql(u8, arg, "--llrd-decay")) {
            const val = args.next() orelse return error.MissingLlrdDecay;
            llrd_decay = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--lisa-sample")) {
            const val = args.next() orelse return error.MissingLisaSample;
            lisa_sample_layers = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--lisa-top-k")) {
            const val = args.next() orelse return error.MissingLisaTopK;
            lisa_top_k = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--lora-train-top-k")) {
            const val = args.next() orelse return error.MissingLoRATrainTopK;
            lora_train_top_k = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--encoder-vjp")) {
            const val = args.next() orelse return error.MissingEncoderVJP;
            if (std.mem.eql(u8, val, "direct")) {
                encoder_vjp = .direct;
            } else if (std.mem.eql(u8, val, "last-layer")) {
                encoder_vjp = .last_layer;
            } else if (std.mem.eql(u8, val, "full")) {
                encoder_vjp = .full;
            } else if (std.mem.eql(u8, val, "full-hidden-direct")) {
                encoder_vjp = .full_hidden_direct;
            } else {
                return error.InvalidEncoderVJP;
            }
        } else if (std.mem.eql(u8, arg, "--layers-per-segment")) {
            const val = args.next() orelse return error.MissingLayersPerSegment;
            layers_per_segment = try std.fmt.parseUnsigned(u32, val, 10);
            if (layers_per_segment == 0) return error.InvalidLayersPerSegment;
        } else if (std.mem.eql(u8, arg, "--compiled-segment-forward")) {
            compiled_segment_forward = true;
        } else if (std.mem.eql(u8, arg, "--lora-plus-ratio")) {
            const val = args.next() orelse return error.MissingLoraPlusRatio;
            lora_plus_ratio = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--length-bucketing")) {
            length_bucketing = true;
        } else if (std.mem.eql(u8, arg, "--bucket-size")) {
            const val = args.next() orelse return error.MissingBucketSize;
            bucket_size = try std.fmt.parseUnsigned(usize, val, 10);
        } else if (std.mem.eql(u8, arg, "--mixed-precision")) {
            mixed_precision = true;
        } else if (std.mem.eql(u8, arg, "--splade")) {
            splade = true;
        } else if (std.mem.eql(u8, arg, "--lambda-splade")) {
            const val = args.next() orelse return error.MissingLambdaSplade;
            lambda_splade = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--lambda-flops")) {
            const val = args.next() orelse return error.MissingLambdaFlops;
            lambda_flops = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--splade-focus-epoch")) {
            const val = args.next() orelse return error.MissingSPLADEFocusEpoch;
            splade_focus_epoch = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--mrl")) {
            mrl = true;
        } else if (std.mem.eql(u8, arg, "--mrl-dims")) {
            mrl_dims_str = args.next() orelse return error.MissingMrlDims;
        } else if (std.mem.eql(u8, arg, "--resume-from")) {
            resume_from = args.next() orelse return error.MissingResumeFrom;
        } else if (std.mem.eql(u8, arg, "--save-optimizer-state")) {
            save_optimizer_state = true;
        } else if (std.mem.eql(u8, arg, "--deterministic")) {
            deterministic = true;
        } else if (std.mem.eql(u8, arg, "--go-epoch-shuffle")) {
            go_epoch_shuffle = true;
        } else if (std.mem.eql(u8, arg, "--report-to")) {
            report_to = args.next() orelse return error.MissingReportTo;
        } else if (std.mem.eql(u8, arg, "--manifest")) {
            manifest_path = args.next() orelse return error.MissingManifestPath;
        } else if (std.mem.eql(u8, arg, "--boundary-alignment-dump-dir")) {
            boundary_alignment_dump_dir = args.next() orelse return error.MissingBoundaryAlignmentDumpDir;
        } else if (std.mem.eql(u8, arg, "--boundary-alignment-dump-top-k")) {
            const val = args.next() orelse return error.MissingBoundaryAlignmentDumpTopK;
            boundary_alignment_dump_top_k = try std.fmt.parseUnsigned(usize, val, 10);
        } else if (std.mem.eql(u8, arg, "--memory-sample-every")) {
            const val = args.next() orelse return error.MissingMemorySampleEvery;
            memory_sample_every = try std.fmt.parseUnsigned(u32, val, 10);
            if (memory_sample_every == 0) return error.InvalidMemorySampleEvery;
        } else if (std.mem.eql(u8, arg, "--memory-warn-rss-gb")) {
            memory_warn_rss_bytes = try parseMemoryGigabytesToBytes(args.next() orelse return error.MissingMemoryWarnRssGb);
        } else if (std.mem.eql(u8, arg, "--memory-abort-rss-gb")) {
            memory_abort_rss_bytes = try parseMemoryGigabytesToBytes(args.next() orelse return error.MissingMemoryAbortRssGb);
        } else if (std.mem.eql(u8, arg, "--debug-first-boundary-step")) {
            debug_first_boundary_step = true;
        } else if (std.mem.eql(u8, arg, "--debug-first-boundary-step-exit")) {
            debug_first_boundary_step = true;
            debug_first_boundary_step_exit = true;
        } else if (std.mem.eql(u8, arg, "--debug-boundary-step")) {
            const val = args.next() orelse return error.MissingDebugBoundaryStep;
            debug_boundary_step = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--debug-boundary-step-exit")) {
            debug_boundary_step_exit = true;
        } else if (std.mem.eql(u8, arg, "--debug-update-step")) {
            const val = args.next() orelse return error.MissingDebugUpdateStep;
            debug_update_step = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--debug-update-step-exit")) {
            debug_update_step_exit = true;
        } else if (std.mem.eql(u8, arg, "--debug-boundary-head-overfit-steps")) {
            const val = args.next() orelse return error.MissingDebugBoundaryHeadOverfitSteps;
            debug_boundary_head_overfit_steps = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--debug-boundary-head-overfit-lr")) {
            const val = args.next() orelse return error.MissingDebugBoundaryHeadOverfitLr;
            debug_boundary_head_overfit_lr = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--debug-frozen-feature-probe")) {
            debug_frozen_feature_probe = true;
        } else if (std.mem.eql(u8, arg, "--debug-frozen-feature-train-examples")) {
            const val = args.next() orelse return error.MissingDebugFrozenFeatureTrainExamples;
            debug_frozen_feature_train_examples = try std.fmt.parseUnsigned(usize, val, 10);
        } else if (std.mem.eql(u8, arg, "--debug-frozen-feature-val-examples")) {
            const val = args.next() orelse return error.MissingDebugFrozenFeatureValExamples;
            debug_frozen_feature_val_examples = try std.fmt.parseUnsigned(usize, val, 10);
        } else if (std.mem.eql(u8, arg, "--debug-frozen-feature-train-offset")) {
            const val = args.next() orelse return error.MissingDebugFrozenFeatureTrainOffset;
            debug_frozen_feature_train_offset = try std.fmt.parseUnsigned(usize, val, 10);
        } else if (std.mem.eql(u8, arg, "--debug-frozen-feature-val-offset")) {
            const val = args.next() orelse return error.MissingDebugFrozenFeatureValOffset;
            debug_frozen_feature_val_offset = try std.fmt.parseUnsigned(usize, val, 10);
        } else if (std.mem.eql(u8, arg, "--debug-frozen-feature-epochs")) {
            const val = args.next() orelse return error.MissingDebugFrozenFeatureEpochs;
            debug_frozen_feature_epochs = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--debug-frozen-feature-lr")) {
            const val = args.next() orelse return error.MissingDebugFrozenFeatureLr;
            debug_frozen_feature_lr = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--debug-step-json")) {
            debug_step_json_path = args.next() orelse return error.MissingDebugStepJson;
        } else if (std.mem.eql(u8, arg, "--debug-batch-offset")) {
            const val = args.next() orelse return error.MissingDebugBatchOffset;
            debug_batch_offset = try std.fmt.parseUnsigned(usize, val, 10);
        } else if (std.mem.eql(u8, arg, "--debug-encoder-probe-layer")) {
            const val = args.next() orelse return error.MissingDebugEncoderProbeLayer;
            debug_encoder_probe_layer = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--debug-encoder-layer-inputs-only")) {
            debug_encoder_layer_inputs_only = true;
        } else if (std.mem.eql(u8, arg, "--debug-encoder-replay-input")) {
            debug_encoder_replay_input_path = args.next() orelse return error.MissingDebugEncoderReplayInput;
        } else if (std.mem.eql(u8, arg, "--debug-encoder-replay-upstream")) {
            debug_encoder_replay_upstream_path = args.next() orelse return error.MissingDebugEncoderReplayUpstream;
        } else if (std.mem.eql(u8, arg, "--debug-layer-backward-decomp")) {
            debug_layer_backward_decomp = true;
        } else if (std.mem.eql(u8, arg, "--debug-qkv-split-vjp")) {
            debug_qkv_split_vjp = true;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            return;
        } else {
            print("unknown argument: {s}\n", .{arg});
            printUsage();
            std.process.exit(1);
        }
    }
    if (debug_encoder_probe_layer >= num_layers) return error.InvalidDebugEncoderProbeLayer;

    const opts = Options{
        .data_path = data_path orelse {
            print("error: --data is required\n", .{});
            printUsage();
            std.process.exit(1);
        },
        .output_dir = output_dir orelse {
            print("error: --output is required\n", .{});
            printUsage();
            std.process.exit(1);
        },
        .val_data_path = val_data_path,
        .model_dir = model_dir,
        .epochs = epochs,
        .batch_size = batch_size,
        .learning_rate = learning_rate,
        .boundary_head_lr_multiplier = boundary_head_lr_multiplier,
        .warmup_steps = warmup_steps,
        .lr_total_steps = lr_total_steps,
        .weight_decay = weight_decay,
        .max_grad_norm = max_grad_norm,
        .beta1 = beta1,
        .beta2 = beta2,
        .adam_epsilon = adam_epsilon,
        .lambda_chunk = lambda_chunk,
        .lambda_embed = lambda_embed,
        .boundary_focus_epochs = boundary_focus_epochs,
        .boundary_focus_lambda_embed = boundary_focus_lambda_embed,
        .boundary_focal_gamma = boundary_focal_gamma,
        .boundary_focal_alpha = boundary_focal_alpha,
        .contrastive_focal_gamma = contrastive_focal_gamma,
        .contrastive_focal_alpha = contrastive_focal_alpha,
        .boundary_pos_weight = boundary_pos_weight,
        .boundary_pos_weight_auto = boundary_pos_weight_auto,
        .boundary_pos_weight_auto_max_examples = boundary_pos_weight_auto_max_examples,
        .boundary_rank_loss_weight = boundary_rank_loss_weight,
        .boundary_rank_loss_margin = boundary_rank_loss_margin,
        .boundary_rank_loss_top_k = boundary_rank_loss_top_k,
        .boundary_same_token_rank_loss_weight = boundary_same_token_rank_loss_weight,
        .boundary_same_token_rank_loss_top_k = boundary_same_token_rank_loss_top_k,
        .boundary_same_token_negative_weight = boundary_same_token_negative_weight,
        .boundary_same_token_negative_top_k = boundary_same_token_negative_top_k,
        .boundary_candidate_rank_loss_weight = boundary_candidate_rank_loss_weight,
        .boundary_candidate_rank_loss_top_k = boundary_candidate_rank_loss_top_k,
        .boundary_candidate_negative_weight = boundary_candidate_negative_weight,
        .boundary_gold_count_rank_loss_weight = boundary_gold_count_rank_loss_weight,
        .boundary_gold_count_rank_loss_margin = boundary_gold_count_rank_loss_margin,
        .boundary_gold_count_rank_loss_negative_multiplier = boundary_gold_count_rank_loss_negative_multiplier,
        .boundary_local_window_loss_weight = boundary_local_window_loss_weight,
        .boundary_local_window_radius = boundary_local_window_radius,
        .boundary_feature_mode = boundary_feature_mode,
        .boundary_dropout = boundary_dropout,
        .boundary_loss_type = boundary_loss_type,
        .hidden_size = hidden_size,
        .num_layers = num_layers,
        .max_seq_len = max_seq_len,
        .max_chunks = max_chunks,
        .checkpoint_every = checkpoint_every,
        .checkpoint_every_steps = checkpoint_every_steps,
        .log_every = log_every,
        .eval_every = eval_every,
        .eval_every_steps = eval_every_steps,
        .step_eval_max_examples = step_eval_max_examples,
        .step_train_eval_max_examples = step_train_eval_max_examples,
        .checkpoint_roundtrip_eval = checkpoint_roundtrip_eval,
        .checkpoint_roundtrip_max_examples = checkpoint_roundtrip_max_examples,
        .max_steps = max_steps,
        .split = split,
        .val_split = val_split,
        .seed = seed,
        .lora_rank = lora_rank,
        .lora_alpha = lora_alpha,
        .lora_start_epoch = lora_start_epoch,
        .intermediate_size = intermediate_size,
        .backend = backend,
        .grad_accum = grad_accum,
        .schedule_free = schedule_free,
        .neftune_alpha = neftune_alpha,
        .encoder_neftune = encoder_neftune,
        .xbm_capacity = xbm_capacity,
        .llrd_decay = llrd_decay,
        .lisa_sample_layers = lisa_sample_layers,
        .lisa_top_k = lisa_top_k,
        .lora_train_top_k = lora_train_top_k,
        .encoder_vjp = encoder_vjp,
        .layers_per_segment = layers_per_segment,
        .compiled_segment_forward = compiled_segment_forward,
        .lora_plus_ratio = lora_plus_ratio,
        .length_bucketing = length_bucketing,
        .bucket_size = bucket_size,
        .mixed_precision = mixed_precision,
        .splade = splade,
        .lambda_splade = lambda_splade,
        .lambda_flops = lambda_flops,
        .splade_focus_epoch = splade_focus_epoch,
        .mrl = mrl,
        .mrl_dims_str = mrl_dims_str,
        .resume_from = resume_from,
        .save_optimizer_state = save_optimizer_state,
        .deterministic = deterministic,
        .go_epoch_shuffle = go_epoch_shuffle,
        .report_to = report_to,
        .manifest_path = manifest_path,
        .boundary_alignment_dump_dir = boundary_alignment_dump_dir,
        .boundary_alignment_dump_top_k = boundary_alignment_dump_top_k,
        .memory_sample_every = memory_sample_every,
        .memory_warn_rss_bytes = memory_warn_rss_bytes,
        .memory_abort_rss_bytes = memory_abort_rss_bytes,
        .debug_first_boundary_step = debug_first_boundary_step,
        .debug_first_boundary_step_exit = debug_first_boundary_step_exit,
        .debug_boundary_step = debug_boundary_step,
        .debug_boundary_step_exit = debug_boundary_step_exit,
        .debug_update_step = debug_update_step,
        .debug_update_step_exit = debug_update_step_exit,
        .debug_boundary_head_overfit_steps = debug_boundary_head_overfit_steps,
        .debug_boundary_head_overfit_lr = debug_boundary_head_overfit_lr,
        .debug_frozen_feature_probe = debug_frozen_feature_probe,
        .debug_frozen_feature_train_examples = debug_frozen_feature_train_examples,
        .debug_frozen_feature_val_examples = debug_frozen_feature_val_examples,
        .debug_frozen_feature_train_offset = debug_frozen_feature_train_offset,
        .debug_frozen_feature_val_offset = debug_frozen_feature_val_offset,
        .debug_frozen_feature_epochs = debug_frozen_feature_epochs,
        .debug_frozen_feature_lr = debug_frozen_feature_lr,
        .debug_step_json_path = debug_step_json_path,
        .debug_batch_offset = debug_batch_offset,
        .debug_encoder_probe_layer = debug_encoder_probe_layer,
        .debug_encoder_layer_inputs_only = debug_encoder_layer_inputs_only,
        .debug_encoder_replay_input_path = debug_encoder_replay_input_path,
        .debug_encoder_replay_upstream_path = debug_encoder_replay_upstream_path,
        .debug_layer_backward_decomp = debug_layer_backward_decomp,
        .debug_qkv_split_vjp = debug_qkv_split_vjp,
    };

    try run(allocator, opts);
}

// ---------------------------------------------------------------------------
// Phase 3A: LoRA pre-merge helpers
// ---------------------------------------------------------------------------

/// Merge LoRA delta into WeightStore base weights before encoder forward.
/// Returns a map of original weight byte slices that must be passed to restoreLoRAWeights.
fn mergeLoRAIntoWeights(
    allocator: std.mem.Allocator,
    weight_store: *native_compute.WeightStore,
    la: *const fused_chunker_lora.LoRAAdapterSet,
) !std.StringHashMapUnmanaged([]u8) {
    var originals = std.StringHashMapUnmanaged([]u8).empty;
    errdefer {
        var it = originals.iterator();
        while (it.next()) |e| allocator.free(e.value_ptr.*);
        originals.deinit(allocator);
    }

    for (la.layers) |*ll| {
        // Weight key follows modern_bert.zig's getLayerWeight convention:
        // "model.layers.N.attn.query_proj.weight" etc.
        const suffix: []const u8 = if (std.mem.eql(u8, ll.module_name, "query_proj"))
            "attn.query_proj.weight"
        else if (std.mem.eql(u8, ll.module_name, "value_proj"))
            "attn.value_proj.weight"
        else if (std.mem.eql(u8, ll.module_name, "key_proj"))
            "attn.key_proj.weight"
        else if (std.mem.eql(u8, ll.module_name, "out_proj"))
            "attn.Wo.weight"
        else if (std.mem.eql(u8, ll.module_name, "wo"))
            "mlp.Wo.weight"
        else
            continue;

        var key_buf: [128]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "model.layers.{d}.{s}", .{ ll.layer_idx, suffix });

        const lw = weight_store.resident_weights.getPtr(key) orelse continue;
        if (lw.tensor.dtype != .f32) continue;

        const base_aligned: []align(@alignOf(f32)) const u8 = @alignCast(lw.tensor.data);
        const base_f32 = std.mem.bytesAsSlice(f32, base_aligned);

        // Allocate merged buffer
        const merged = try allocator.alloc(f32, base_f32.len);
        errdefer allocator.free(merged);

        const base_mat = lora.Matrix{
            .rows = ll.out_features,
            .cols = ll.in_features,
            .data = base_f32,
        };
        lora.mergeInto(base_mat, ll.asMatrixA(), ll.asMatrixB(), la.config.alpha, merged);

        // Save original data bytes before replacing
        const orig_bytes = try allocator.dupe(u8, lw.tensor.data);
        try originals.put(allocator, try allocator.dupe(u8, key), orig_bytes);

        // Replace tensor data with merged (as bytes)
        lw.tensor.data = std.mem.sliceAsBytes(merged);
    }
    return originals;
}

/// Restore original weight data after encoder forward.
fn restoreLoRAWeights(
    allocator: std.mem.Allocator,
    weight_store: *native_compute.WeightStore,
    originals: *std.StringHashMapUnmanaged([]u8),
) void {
    var it = originals.iterator();
    while (it.next()) |e| {
        if (weight_store.resident_weights.getPtr(e.key_ptr.*)) |lw| {
            // Free the merged buffer (the current tensor.data)
            const merged_aligned: []align(@alignOf(f32)) const u8 = @alignCast(lw.tensor.data);
            const merged_f32 = std.mem.bytesAsSlice(f32, merged_aligned);
            allocator.free(merged_f32);
            // Restore original
            lw.tensor.data = e.value_ptr.*;
        } else {
            allocator.free(e.value_ptr.*);
        }
        allocator.free(e.key_ptr.*);
    }
    originals.deinit(allocator);
}

fn mergeLoRAIntoMetalWeights(
    allocator: std.mem.Allocator,
    weight_store: *MetalWeightStore,
    la: *const fused_chunker_lora.LoRAAdapterSet,
) !std.StringHashMapUnmanaged([]u8) {
    if (comptime !build_options.enable_metal) return error.MetalBackendUnavailable;
    var originals = std.StringHashMapUnmanaged([]u8).empty;
    errdefer {
        var it = originals.iterator();
        while (it.next()) |e| {
            allocator.free(e.key_ptr.*);
            allocator.free(e.value_ptr.*);
        }
        originals.deinit(allocator);
    }

    for (la.layers) |*ll| {
        const suffix: []const u8 = if (std.mem.eql(u8, ll.module_name, "query_proj"))
            "attn.query_proj.weight"
        else if (std.mem.eql(u8, ll.module_name, "value_proj"))
            "attn.value_proj.weight"
        else if (std.mem.eql(u8, ll.module_name, "key_proj"))
            "attn.key_proj.weight"
        else if (std.mem.eql(u8, ll.module_name, "out_proj"))
            "attn.Wo.weight"
        else if (std.mem.eql(u8, ll.module_name, "wo"))
            "mlp.Wo.weight"
        else
            continue;

        var key_buf: [128]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "model.layers.{d}.{s}", .{ ll.layer_idx, suffix });
        const entry = weight_store.lazy_weights.getPtr(key) orelse continue;
        const lw = if (entry.host_loaded) |*loaded| loaded else continue;
        if (lw.tensor.dtype != .f32) continue;

        const base_aligned: []align(@alignOf(f32)) const u8 = @alignCast(lw.tensor.data);
        const base_f32 = std.mem.bytesAsSlice(f32, base_aligned);

        const merged = try allocator.alloc(f32, base_f32.len);
        errdefer allocator.free(merged);

        const base_mat = lora.Matrix{
            .rows = ll.out_features,
            .cols = ll.in_features,
            .data = base_f32,
        };
        lora.mergeInto(base_mat, ll.asMatrixA(), ll.asMatrixB(), la.config.alpha, merged);

        const orig_bytes = try allocator.dupe(u8, lw.tensor.data);
        const owned_key = try allocator.dupe(u8, key);
        errdefer allocator.free(owned_key);
        errdefer allocator.free(orig_bytes);
        try originals.put(allocator, owned_key, orig_bytes);

        lw.tensor.data = std.mem.sliceAsBytes(merged);
        lw.tensor.owns_data = true;
        entry.loaded_bytes = lw.tensor.data.len;
    }
    return originals;
}

fn restoreMetalLoRAWeights(
    allocator: std.mem.Allocator,
    weight_store: *MetalWeightStore,
    originals: *std.StringHashMapUnmanaged([]u8),
) void {
    if (comptime !build_options.enable_metal) return;
    var it = originals.iterator();
    while (it.next()) |e| {
        if (weight_store.lazy_weights.getPtr(e.key_ptr.*)) |entry| {
            if (entry.host_loaded) |*lw| {
                const merged_aligned: []align(@alignOf(f32)) const u8 = @alignCast(lw.tensor.data);
                const merged_f32 = std.mem.bytesAsSlice(f32, merged_aligned);
                allocator.free(merged_f32);
                lw.tensor.data = e.value_ptr.*;
                lw.tensor.owns_data = true;
                entry.loaded_bytes = lw.tensor.data.len;
            } else {
                allocator.free(e.value_ptr.*);
            }
        } else {
            allocator.free(e.value_ptr.*);
        }
        allocator.free(e.key_ptr.*);
    }
    originals.deinit(allocator);
}

/// Insert or update a LoRA matrix in the native WeightStore under `key`.
/// The tensor is a 2-D f32 matrix of shape [rows, cols].
/// If a weight already exists under `key` its data is replaced with a fresh
/// copy of `data` so that optimizer updates are visible each step.
fn insertLoRAIntoNativeStore(
    allocator: std.mem.Allocator,
    weight_store: *native_compute.WeightStore,
    key: []const u8,
    data: []const f32,
    rows: usize,
    cols: usize,
) !void {
    const shape = [2]i64{ @intCast(rows), @intCast(cols) };
    if (weight_store.resident_weights.getPtr(key)) |existing| {
        // Update the data in place: free old bytes, copy fresh data.
        const new_bytes = try existing.tensor.allocator.dupe(u8, std.mem.sliceAsBytes(data));
        if (existing.tensor.owns_data) {
            existing.tensor.allocator.free(existing.tensor.data);
        }
        existing.tensor.data = new_bytes;
        existing.tensor.owns_data = true;
        return;
    }
    var tensor = try tensor_mod.Tensor.initFloat32(allocator, key, &shape, data);
    errdefer tensor.deinit();
    const owned_key = try allocator.dupe(u8, key);
    errdefer allocator.free(owned_key);
    try weight_store.resident_weights.put(allocator, owned_key, weight_source_mod.LoadedWeight{ .tensor = tensor });
}

fn appendCheckpointTensor(
    allocator: std.mem.Allocator,
    tensor_list: *std.ArrayListUnmanaged(safetensors_checkpoint.NamedTensor),
    shape_storage: *std.ArrayListUnmanaged([]usize),
    name: []const u8,
    data: []const f32,
    shape: []const usize,
) !void {
    const owned_shape = try allocator.dupe(usize, shape);
    try shape_storage.append(allocator, owned_shape);
    try tensor_list.append(allocator, .{
        .name = name,
        .data = data,
        .shape = owned_shape,
    });
}

const LoRARuntimeTarget = struct {
    scope: []const u8,
    projection: []const u8,
};

fn loraRuntimeTarget(module_name: []const u8) ?LoRARuntimeTarget {
    if (std.mem.eql(u8, module_name, "query_proj") or
        std.mem.eql(u8, module_name, "key_proj") or
        std.mem.eql(u8, module_name, "value_proj"))
    {
        return .{ .scope = "attn", .projection = module_name };
    }
    if (std.mem.eql(u8, module_name, "out_proj")) return .{ .scope = "attn", .projection = "Wo" };
    if (std.mem.eql(u8, module_name, "wo")) return .{ .scope = "mlp", .projection = "Wo" };
    return null;
}

fn saveFusedCheckpoint(
    allocator: std.mem.Allocator,
    path: []const u8,
    trainer: *FusedTrainer,
    lora_adapters: ?*const fused_chunker_lora.LoRAAdapterSet,
) !void {
    var tensor_list = std.ArrayListUnmanaged(safetensors_checkpoint.NamedTensor).empty;
    defer tensor_list.deinit(allocator);

    var name_storage = std.ArrayListUnmanaged([]u8).empty;
    defer {
        for (name_storage.items) |name| allocator.free(name);
        name_storage.deinit(allocator);
    }

    var shape_storage = std.ArrayListUnmanaged([]usize).empty;
    defer {
        for (shape_storage.items) |shape| allocator.free(shape);
        shape_storage.deinit(allocator);
    }

    const w1_shape = [_]usize{ trainer.boundary_head.mlp_dim, trainer.boundary_head.hidden_dim };
    const b1_shape = [_]usize{trainer.boundary_head.mlp_dim};
    const w2_shape = [_]usize{ 2, trainer.boundary_head.mlp_dim };
    const b2_shape = [_]usize{2};
    try appendCheckpointTensor(allocator, &tensor_list, &shape_storage, "w1", trainer.boundary_head.w1, &w1_shape);
    try appendCheckpointTensor(allocator, &tensor_list, &shape_storage, "b1", trainer.boundary_head.b1, &b1_shape);
    try appendCheckpointTensor(allocator, &tensor_list, &shape_storage, "w2", trainer.boundary_head.w2, &w2_shape);
    try appendCheckpointTensor(allocator, &tensor_list, &shape_storage, "b2", trainer.boundary_head.b2, &b2_shape);

    if (lora_adapters) |la| {
        const rank_scalar = [_]f32{@floatFromInt(la.config.rank)};
        const alpha_scalar = [_]f32{la.config.alpha};
        const scalar_shape = [_]usize{1};
        try appendCheckpointTensor(allocator, &tensor_list, &shape_storage, "lora_rank", &rank_scalar, &scalar_shape);
        try appendCheckpointTensor(allocator, &tensor_list, &shape_storage, "lora_alpha", &alpha_scalar, &scalar_shape);

        for (la.layers) |*ll| {
            const target = loraRuntimeTarget(ll.module_name) orelse continue;
            const rank = ll.A.len / ll.in_features;
            if (rank == 0) continue;

            const name_a = try std.fmt.allocPrint(
                allocator,
                "model.layers.{d}.{s}.{s}.lora_a",
                .{ ll.layer_idx, target.scope, target.projection },
            );
            try name_storage.append(allocator, name_a);
            const shape_a = [_]usize{ rank, ll.in_features };
            try appendCheckpointTensor(allocator, &tensor_list, &shape_storage, name_a, ll.A, &shape_a);

            const name_b = try std.fmt.allocPrint(
                allocator,
                "model.layers.{d}.{s}.{s}.lora_b",
                .{ ll.layer_idx, target.scope, target.projection },
            );
            try name_storage.append(allocator, name_b);
            const shape_b = [_]usize{ ll.out_features, rank };
            try appendCheckpointTensor(allocator, &tensor_list, &shape_storage, name_b, ll.B, &shape_b);
        }
    }

    try safetensors_checkpoint.save(allocator, path, tensor_list.items);
}

fn appendOptimizerStateTensor(
    allocator: std.mem.Allocator,
    tensor_list: *std.ArrayListUnmanaged(safetensors_checkpoint.NamedTensor),
    name_storage: *std.ArrayListUnmanaged([]u8),
    shape_storage: *std.ArrayListUnmanaged([]usize),
    name: []const u8,
    data: []const f32,
    shape: []const usize,
) !void {
    const owned_name = try allocator.dupe(u8, name);
    errdefer allocator.free(owned_name);
    try name_storage.append(allocator, owned_name);
    try appendCheckpointTensor(allocator, tensor_list, shape_storage, owned_name, data, shape);
}

fn appendLoRAOptimizerStateTensor(
    allocator: std.mem.Allocator,
    tensor_list: *std.ArrayListUnmanaged(safetensors_checkpoint.NamedTensor),
    name_storage: *std.ArrayListUnmanaged([]u8),
    shape_storage: *std.ArrayListUnmanaged([]usize),
    prefix: []const u8,
    layer_idx: u32,
    module_name: []const u8,
    matrix_name: []const u8,
    data: []const f32,
    shape: []const usize,
) !void {
    const name = try std.fmt.allocPrint(
        allocator,
        "{s}_lora.{d}.{s}.{s}",
        .{ prefix, layer_idx, module_name, matrix_name },
    );
    errdefer allocator.free(name);
    try name_storage.append(allocator, name);
    try appendCheckpointTensor(allocator, tensor_list, shape_storage, name, data, shape);
}

fn saveFusedOptimizerState(
    allocator: std.mem.Allocator,
    path: []const u8,
    trainer: *const FusedTrainer,
    lora_adapters: ?*const fused_chunker_lora.LoRAAdapterSet,
    lora_optimizer_state: ?*const optimizers.OptimizerState,
) !void {
    var tensor_list = std.ArrayListUnmanaged(safetensors_checkpoint.NamedTensor).empty;
    defer tensor_list.deinit(allocator);

    var name_storage = std.ArrayListUnmanaged([]u8).empty;
    defer {
        for (name_storage.items) |name| allocator.free(name);
        name_storage.deinit(allocator);
    }

    var shape_storage = std.ArrayListUnmanaged([]usize).empty;
    defer {
        for (shape_storage.items) |shape| allocator.free(shape);
        shape_storage.deinit(allocator);
    }

    const step_val = [1]f32{@as(f32, @floatFromInt(trainer.optimizer_state.step_count))};
    try appendCheckpointTensor(allocator, &tensor_list, &shape_storage, "adam_step", &step_val, &.{1});

    const head_params = [_]struct {
        name: []const u8,
        values: []const f32,
    }{
        .{ .name = "w1", .values = trainer.boundary_head.w1 },
        .{ .name = "b1", .values = trainer.boundary_head.b1 },
        .{ .name = "w2", .values = trainer.boundary_head.w2 },
        .{ .name = "b2", .values = trainer.boundary_head.b2 },
    };
    for (head_params) |param| {
        if (trainer.optimizer_state.param_states.get(param.name)) |state| {
            const m_name = try std.fmt.allocPrint(allocator, "adam_m_{s}", .{param.name});
            defer allocator.free(m_name);
            try appendOptimizerStateTensor(allocator, &tensor_list, &name_storage, &shape_storage, m_name, state.m, &.{state.m.len});
            if (state.v.len > 0) {
                const v_name = try std.fmt.allocPrint(allocator, "adam_v_{s}", .{param.name});
                defer allocator.free(v_name);
                try appendOptimizerStateTensor(allocator, &tensor_list, &name_storage, &shape_storage, v_name, state.v, &.{state.v.len});
            }
        }
    }

    if (lora_adapters) |la| {
        if (lora_optimizer_state) |state| {
            const lora_step_val = [1]f32{@as(f32, @floatFromInt(state.step_count))};
            try appendCheckpointTensor(allocator, &tensor_list, &shape_storage, "lora_adam_step", &lora_step_val, &.{1});
            for (la.layers) |*ll| {
                const rank = ll.A.len / ll.in_features;
                if (rank == 0) continue;

                const shape_a = [_]usize{ rank, ll.in_features };
                const shape_b = [_]usize{ ll.out_features, rank };

                const a_key = try std.fmt.allocPrint(allocator, "lora.{d}.{s}.A", .{ ll.layer_idx, ll.module_name });
                defer allocator.free(a_key);
                if (state.param_states.get(a_key)) |param_state| {
                    try appendLoRAOptimizerStateTensor(allocator, &tensor_list, &name_storage, &shape_storage, "adam_m", ll.layer_idx, ll.module_name, "A", param_state.m, &shape_a);
                    if (param_state.v.len > 0) {
                        try appendLoRAOptimizerStateTensor(allocator, &tensor_list, &name_storage, &shape_storage, "adam_v", ll.layer_idx, ll.module_name, "A", param_state.v, &shape_a);
                    }
                }

                const b_key = try std.fmt.allocPrint(allocator, "lora.{d}.{s}.B", .{ ll.layer_idx, ll.module_name });
                defer allocator.free(b_key);
                if (state.param_states.get(b_key)) |param_state| {
                    try appendLoRAOptimizerStateTensor(allocator, &tensor_list, &name_storage, &shape_storage, "adam_m", ll.layer_idx, ll.module_name, "B", param_state.m, &shape_b);
                    if (param_state.v.len > 0) {
                        try appendLoRAOptimizerStateTensor(allocator, &tensor_list, &name_storage, &shape_storage, "adam_v", ll.layer_idx, ll.module_name, "B", param_state.v, &shape_b);
                    }
                }
            }
        }
    }

    try safetensors_checkpoint.save(allocator, path, tensor_list.items);
}

fn copyOptimizerTensorIfPresent(
    reader: *const safetensors.MMapReader,
    name: []const u8,
    dest: []f32,
) !bool {
    if (reader.header.tensors.get(name) == null) return false;
    var tensor = try reader.readTensor(name);
    defer tensor.deinit();
    if (tensor.dtype != .f32) return error.InvalidTensorDType;
    const src = tensor.asFloat32();
    if (src.len != dest.len) return error.OptimizerStateShapeMismatch;
    @memcpy(dest, src);
    return true;
}

fn loadFusedLoRAOptimizerState(
    allocator: std.mem.Allocator,
    reader: *const safetensors.MMapReader,
    lora_adapters: *const fused_chunker_lora.LoRAAdapterSet,
    lora_optimizer_state: *optimizers.OptimizerState,
) !usize {
    var restored: usize = 0;
    if (reader.header.tensors.get("lora_adam_step")) |_| {
        var step_tensor = try reader.readTensor("lora_adam_step");
        defer step_tensor.deinit();
        const values = step_tensor.asFloat32();
        if (values.len > 0) lora_optimizer_state.step_count = @intFromFloat(values[0]);
    } else if (reader.header.tensors.get("adam_step")) |_| {
        var step_tensor = try reader.readTensor("adam_step");
        defer step_tensor.deinit();
        const values = step_tensor.asFloat32();
        if (values.len > 0) lora_optimizer_state.step_count = @intFromFloat(values[0]);
    }

    for (lora_adapters.layers) |*ll| {
        const rank = ll.A.len / ll.in_features;
        if (rank == 0) continue;

        const a_key = try std.fmt.allocPrint(allocator, "lora.{d}.{s}.A", .{ ll.layer_idx, ll.module_name });
        defer allocator.free(a_key);
        const a_m_name = try std.fmt.allocPrint(allocator, "adam_m_{s}", .{a_key});
        defer allocator.free(a_m_name);
        const a_v_name = try std.fmt.allocPrint(allocator, "adam_v_{s}", .{a_key});
        defer allocator.free(a_v_name);
        const a_has_m = reader.header.tensors.get(a_m_name) != null;
        const a_has_v = reader.header.tensors.get(a_v_name) != null;
        if (a_has_m) {
            const state = try lora_optimizer_state.getOrCreate(a_key, ll.A.len, a_has_v);
            _ = try copyOptimizerTensorIfPresent(reader, a_m_name, state.m);
            restored += 1;
            if (a_has_v) {
                _ = try copyOptimizerTensorIfPresent(reader, a_v_name, state.v);
                restored += 1;
            }
        }

        const b_key = try std.fmt.allocPrint(allocator, "lora.{d}.{s}.B", .{ ll.layer_idx, ll.module_name });
        defer allocator.free(b_key);
        const b_m_name = try std.fmt.allocPrint(allocator, "adam_m_{s}", .{b_key});
        defer allocator.free(b_m_name);
        const b_v_name = try std.fmt.allocPrint(allocator, "adam_v_{s}", .{b_key});
        defer allocator.free(b_v_name);
        const b_has_m = reader.header.tensors.get(b_m_name) != null;
        const b_has_v = reader.header.tensors.get(b_v_name) != null;
        if (b_has_m) {
            const state = try lora_optimizer_state.getOrCreate(b_key, ll.B.len, b_has_v);
            _ = try copyOptimizerTensorIfPresent(reader, b_m_name, state.m);
            restored += 1;
            if (b_has_v) {
                _ = try copyOptimizerTensorIfPresent(reader, b_v_name, state.v);
                restored += 1;
            }
        }
    }
    return restored;
}

fn loadFusedOptimizerState(
    allocator: std.mem.Allocator,
    path: []const u8,
    trainer: *FusedTrainer,
    lora_adapters: ?*const fused_chunker_lora.LoRAAdapterSet,
    lora_optimizer_state: *optimizers.OptimizerState,
) !void {
    try trainer.loadOptimizerState(allocator, path);
    const la = lora_adapters orelse return;

    const file_bytes = try compat.cwd().readFileAlloc(compat.io(), path, allocator, .unlimited);
    var reader = try safetensors.MMapReader.fromBytes(allocator, file_bytes);
    defer reader.deinit();
    const restored = try loadFusedLoRAOptimizerState(allocator, &reader, la, lora_optimizer_state);
    print("restored {d} LoRA optimizer tensors from {s}\n", .{ restored, path });
}

fn savePeriodicTrainingCheckpoint(
    allocator: std.mem.Allocator,
    opts: *const Options,
    trainer: *FusedTrainer,
    lora_adapters: ?*const fused_chunker_lora.LoRAAdapterSet,
    lora_optimizer_state: ?*const optimizers.OptimizerState,
    splade_w: ?[]const f32,
    splade_vocab_size: usize,
    label: []const u8,
    value: u32,
) !void {
    var path_buf: [512]u8 = undefined;
    const ckpt_path = try std.fmt.bufPrint(&path_buf, "{s}/checkpoint_{s}_{d}.safetensors", .{
        opts.output_dir,
        label,
        value,
    });
    try saveFusedCheckpoint(allocator, ckpt_path, trainer, lora_adapters);
    print("checkpoint saved to {s}\n", .{ckpt_path});

    if (opts.save_optimizer_state) {
        var opt_path_buf: [512]u8 = undefined;
        const opt_path = try std.fmt.bufPrint(&opt_path_buf, "{s}/checkpoint_{s}_{d}_optimizer.safetensors", .{
            opts.output_dir,
            label,
            value,
        });
        try saveFusedOptimizerState(allocator, opt_path, trainer, lora_adapters, lora_optimizer_state);
        print("optimizer state saved to {s}\n", .{opt_path});
    }

    if (splade_w) |w| {
        const splade_ckpt_path = try std.fmt.allocPrint(allocator, "{s}/splade_w_{s}_{d}.safetensors", .{
            opts.output_dir,
            label,
            value,
        });
        defer allocator.free(splade_ckpt_path);
        const splade_tensors = [_]safetensors_checkpoint.NamedTensor{
            .{ .name = "splade_proj_weight", .data = w, .shape = &.{ splade_vocab_size, @as(usize, opts.hidden_size) } },
        };
        try safetensors_checkpoint.save(allocator, splade_ckpt_path, &splade_tensors);
        print("SPLADE weight saved to {s}\n", .{splade_ckpt_path});
    }
}

fn stripEncoderPrefix(name: []const u8) []const u8 {
    const prefix = "encoder.";
    if (std.mem.startsWith(u8, name, prefix)) return name[prefix.len..];
    return name;
}

fn normalizeModernBertWeightName(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    const stripped = stripEncoderPrefix(name);
    if (std.mem.startsWith(u8, stripped, "layers.") or
        std.mem.startsWith(u8, stripped, "embeddings.") or
        std.mem.startsWith(u8, stripped, "final_norm."))
    {
        return std.fmt.allocPrint(allocator, "model.{s}", .{stripped});
    }
    return allocator.dupe(u8, stripped);
}

fn isGoFusedEmbeddingWeightName(name: []const u8) bool {
    return std.mem.eql(u8, name, "model.embeddings.tok_embeddings.weight");
}

fn transposeLoadedEmbeddingWeightLikeGoFused(allocator: std.mem.Allocator, loaded: *LoadedWeight) !void {
    if (loaded.tensor.dtype != .f32) return error.UnsupportedGoFusedEmbeddingTensor;
    if (loaded.tensor.shape.len != 2) return error.InvalidGoFusedEmbeddingTensor;
    if (loaded.tensor.shape[0] < 0 or loaded.tensor.shape[1] < 0) return error.InvalidGoFusedEmbeddingTensor;

    const rows: usize = @intCast(loaded.tensor.shape[0]);
    const cols: usize = @intCast(loaded.tensor.shape[1]);
    const values = loaded.tensor.asFloat32();
    if (values.len != rows * cols) return error.InvalidGoFusedEmbeddingTensor;

    const transposed = try allocator.alloc(f32, values.len);
    defer allocator.free(transposed);
    for (0..cols) |r| {
        for (0..rows) |c| {
            transposed[c * cols + r] = values[r * rows + c];
        }
    }

    var tensor = try tensor_mod.Tensor.initFloat32(allocator, loaded.tensor.name, loaded.tensor.shape, transposed);
    errdefer tensor.deinit();
    loaded.tensor.deinit();
    loaded.tensor = tensor;
}

fn cloneLoadedWeight(allocator: std.mem.Allocator, loaded: LoadedWeight) !LoadedWeight {
    if (loaded.quantized or loaded.quantized_storage != null) return error.UnsupportedQuantizedTrainingWeight;
    const owned_data = try allocator.dupe(u8, loaded.tensor.data);
    errdefer allocator.free(owned_data);
    const owned_shape = try allocator.dupe(i64, loaded.tensor.shape);
    errdefer allocator.free(owned_shape);
    return .{
        .tensor = .{
            .data = owned_data,
            .dtype = loaded.tensor.dtype,
            .shape = owned_shape,
            .name = "",
            .allocator = allocator,
            .owns_data = true,
            .owns_shape = true,
        },
        .quantized = false,
    };
}

fn modernBertQkvLayer(name: []const u8) ?[]const u8 {
    const prefix = "model.layers.";
    const suffix = ".attn.Wqkv.weight";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    if (!std.mem.endsWith(u8, name, suffix)) return null;
    return name[prefix.len .. name.len - suffix.len];
}

fn copyModernBertQkvPart(
    allocator: std.mem.Allocator,
    loaded: LoadedWeight,
    part: usize,
) !struct { data: []f32, hidden: usize } {
    if (loaded.tensor.dtype != .f32) return error.UnsupportedModernBertQkvTensor;
    if (loaded.tensor.shape.len != 2) return error.InvalidModernBertQkvTensor;
    if (loaded.tensor.shape[0] < 0 or loaded.tensor.shape[1] < 0) return error.InvalidModernBertQkvTensor;
    const rows: usize = @intCast(loaded.tensor.shape[0]);
    const hidden: usize = @intCast(loaded.tensor.shape[1]);
    if (rows != hidden * 3) return error.InvalidModernBertQkvTensor;

    const elems = hidden * hidden;
    const elem_start = part * elems;
    const byte_start = elem_start * @sizeOf(f32);
    const byte_end = byte_start + elems * @sizeOf(f32);
    if (byte_end > loaded.tensor.data.len) return error.InvalidModernBertQkvTensor;

    const data = try allocator.alloc(f32, elems);
    errdefer allocator.free(data);
    @memcpy(std.mem.sliceAsBytes(data), loaded.tensor.data[byte_start..byte_end]);
    return .{ .data = data, .hidden = hidden };
}

fn insertModernBertQkvSplitsIntoNativeStore(
    allocator: std.mem.Allocator,
    weight_store: *native_compute.WeightStore,
    normalized_name: []const u8,
    loaded: LoadedWeight,
) !usize {
    const layer = modernBertQkvLayer(normalized_name) orelse return 0;
    const projections = [_][]const u8{ "query_proj", "key_proj", "value_proj" };
    var inserted: usize = 0;
    for (projections, 0..) |projection, part| {
        const split = try copyModernBertQkvPart(allocator, loaded, part);
        defer allocator.free(split.data);
        const split_name = try std.fmt.allocPrint(allocator, "model.layers.{s}.attn.{s}.weight", .{ layer, projection });
        errdefer allocator.free(split_name);
        const shape = [_]i64{ @intCast(split.hidden), @intCast(split.hidden) };
        var tensor = try tensor_mod.Tensor.initFloat32(allocator, split_name, &shape, split.data);
        errdefer tensor.deinit();
        try weight_store.resident_weights.put(allocator, split_name, weight_source_mod.LoadedWeight{ .tensor = tensor });
        inserted += 1;
    }
    return inserted;
}

fn insertModernBertQkvSplitsIntoMetalStore(
    allocator: std.mem.Allocator,
    weight_store: *MetalWeightStore,
    normalized_name: []const u8,
    loaded: LoadedWeight,
) !usize {
    if (comptime !build_options.enable_metal) return error.MetalBackendUnavailable;
    const layer = modernBertQkvLayer(normalized_name) orelse return 0;
    const projections = [_][]const u8{ "query_proj", "key_proj", "value_proj" };
    var inserted: usize = 0;
    for (projections, 0..) |projection, part| {
        const split = try copyModernBertQkvPart(allocator, loaded, part);
        defer allocator.free(split.data);
        const split_name = try std.fmt.allocPrint(allocator, "model.layers.{s}.attn.{s}.weight", .{ layer, projection });
        errdefer allocator.free(split_name);
        const shape = [_]i64{ @intCast(split.hidden), @intCast(split.hidden) };
        var tensor = try tensor_mod.Tensor.initFloat32(allocator, split_name, &shape, split.data);
        errdefer tensor.deinit();
        try weight_store.lazy_weights.put(allocator, split_name, .{
            .tensor_ref = undefined,
            .host_loaded = .{ .tensor = tensor },
            .active_tier = .host,
            .loaded_bytes = tensor.data.len,
        });
        inserted += 1;
    }
    return inserted;
}

fn loadSafetensorsIntoMetalStore(
    allocator: std.mem.Allocator,
    weight_store: *MetalWeightStore,
    st_path: []const u8,
) !usize {
    if (comptime !build_options.enable_metal) return error.MetalBackendUnavailable;
    var source = try SafetensorsSource.initAbsolute(allocator, st_path);
    errdefer source.weightSource().deinit();
    const ws = source.weightSource();
    const names = try ws.listNames(allocator);
    defer allocator.free(names);

    var loaded_count: usize = 0;
    for (names) |name| {
        var loaded = ws.getTensor(name) catch continue;
        defer loaded.deinit();
        var owned_loaded = try cloneLoadedWeight(allocator, loaded);
        errdefer owned_loaded.deinit();
        const owned_name = try normalizeModernBertWeightName(allocator, name);
        errdefer allocator.free(owned_name);
        if (isGoFusedEmbeddingWeightName(owned_name)) {
            try transposeLoadedEmbeddingWeightLikeGoFused(allocator, &owned_loaded);
        }
        try weight_store.lazy_weights.put(allocator, owned_name, .{
            .tensor_ref = undefined,
            .host_loaded = owned_loaded,
            .active_tier = .host,
            .loaded_bytes = owned_loaded.tensor.data.len,
        });
        loaded_count += 1;
        loaded_count += try insertModernBertQkvSplitsIntoMetalStore(allocator, weight_store, owned_name, loaded);
    }
    source.weightSource().deinit();
    return loaded_count;
}

fn insertLoRAIntoMetalStore(
    allocator: std.mem.Allocator,
    weight_store: *MetalWeightStore,
    key: []const u8,
    data: []const f32,
    rows: usize,
    cols: usize,
) !void {
    if (comptime !build_options.enable_metal) return error.MetalBackendUnavailable;
    const shape = [2]i64{ @intCast(rows), @intCast(cols) };
    if (weight_store.lazy_weights.getPtr(key)) |entry| {
        const src_bytes = std.mem.sliceAsBytes(data);
        if (entry.host_loaded) |*loaded| {
            if (loaded.tensor.dtype == .f32 and
                loaded.tensor.shape.len == shape.len and
                loaded.tensor.shape[0] == shape[0] and
                loaded.tensor.shape[1] == shape[1] and
                loaded.tensor.data.len == src_bytes.len)
            {
                @memcpy(loaded.tensor.data, src_bytes);
                entry.active_tier = .host;
                entry.loaded_bytes = loaded.tensor.data.len;
                return;
            }
        }

        var tensor = try tensor_mod.Tensor.initFloat32(allocator, key, &shape, data);
        errdefer tensor.deinit();
        if (entry.host_loaded) |*loaded| loaded.deinit();
        if (entry.quantized_storage) |*storage| storage.deinit();
        entry.* = .{
            .tensor_ref = undefined,
            .host_loaded = .{ .tensor = tensor },
            .active_tier = .host,
            .loaded_bytes = tensor.data.len,
        };
        return;
    }
    var tensor = try tensor_mod.Tensor.initFloat32(allocator, key, &shape, data);
    errdefer tensor.deinit();
    const owned_key = try allocator.dupe(u8, key);
    errdefer allocator.free(owned_key);
    try weight_store.lazy_weights.put(allocator, owned_key, .{
        .tensor_ref = undefined,
        .host_loaded = .{ .tensor = tensor },
        .active_tier = .host,
        .loaded_bytes = tensor.data.len,
    });
}

fn refreshLoRAWeightsForForward(
    allocator: std.mem.Allocator,
    use_metal: bool,
    native_weight_store: *native_compute.WeightStore,
    metal_weight_store: *MetalWeightStore,
    la: *const fused_chunker_lora.LoRAAdapterSet,
) !void {
    const rank: usize = @intCast(la.config.rank);
    for (la.layers) |*ll| {
        const target = loraRuntimeTarget(ll.module_name) orelse continue;

        var key_buf_a: [128]u8 = undefined;
        var key_buf_b: [128]u8 = undefined;
        const key_a = try std.fmt.bufPrint(
            &key_buf_a,
            "model.layers.{d}.{s}.{s}.lora_a",
            .{ ll.layer_idx, target.scope, target.projection },
        );
        const key_b = try std.fmt.bufPrint(
            &key_buf_b,
            "model.layers.{d}.{s}.{s}.lora_b",
            .{ ll.layer_idx, target.scope, target.projection },
        );

        if (use_metal) {
            if (comptime build_options.enable_metal) {
                try insertLoRAIntoMetalStore(allocator, metal_weight_store, key_a, ll.A, rank, ll.in_features);
                try insertLoRAIntoMetalStore(allocator, metal_weight_store, key_b, ll.B, ll.out_features, rank);
            }
        } else {
            try insertLoRAIntoNativeStore(allocator, native_weight_store, key_a, ll.A, rank, ll.in_features);
            try insertLoRAIntoNativeStore(allocator, native_weight_store, key_b, ll.B, ll.out_features, rank);
        }
    }
}

fn loadLoRAAdaptersFromCheckpoint(
    allocator: std.mem.Allocator,
    checkpoint_path: []const u8,
    la: *fused_chunker_lora.LoRAAdapterSet,
) !usize {
    const file_bytes = try compat.cwd().readFileAlloc(compat.io(), checkpoint_path, allocator, .unlimited);
    var reader = safetensors.MMapReader.fromBytes(allocator, file_bytes) catch |err| {
        allocator.free(file_bytes);
        return err;
    };
    defer reader.deinit();

    var loaded: usize = 0;
    for (la.layers) |*ll| {
        const target = loraRuntimeTarget(ll.module_name) orelse continue;
        const rank = ll.A.len / ll.in_features;
        if (rank == 0) continue;

        var zig_a_buf: [128]u8 = undefined;
        var zig_b_buf: [128]u8 = undefined;
        var go_a_buf: [160]u8 = undefined;
        var go_b_buf: [160]u8 = undefined;
        const zig_a = try std.fmt.bufPrint(
            &zig_a_buf,
            "model.layers.{d}.{s}.{s}.lora_a",
            .{ ll.layer_idx, target.scope, target.projection },
        );
        const zig_b = try std.fmt.bufPrint(
            &zig_b_buf,
            "model.layers.{d}.{s}.{s}.lora_b",
            .{ ll.layer_idx, target.scope, target.projection },
        );
        const go_a = try std.fmt.bufPrint(
            &go_a_buf,
            "var:/fused_chunker_embedder/encoder/layer/{d}/{s}/{s}/lora_A",
            .{ ll.layer_idx, target.scope, target.projection },
        );
        const go_b = try std.fmt.bufPrint(
            &go_b_buf,
            "var:/fused_chunker_embedder/encoder/layer/{d}/{s}/{s}/lora_B",
            .{ ll.layer_idx, target.scope, target.projection },
        );

        if (try loadLoRATensorFromCheckpoint(&reader, zig_a, go_a, ll.A, rank, ll.in_features)) loaded += 1;
        if (try loadLoRATensorFromCheckpoint(&reader, zig_b, go_b, ll.B, ll.out_features, rank)) loaded += 1;
    }
    return loaded;
}

fn loadLoRATensorFromCheckpoint(
    reader: anytype,
    zig_name: []const u8,
    go_name: []const u8,
    dest: []f32,
    rows: usize,
    cols: usize,
) !bool {
    if (reader.header.tensors.contains(zig_name)) {
        var tensor = try reader.readTensor(zig_name);
        defer tensor.deinit();
        if (tensor.dtype != .f32) return error.InvalidCheckpoint;
        if (tensor.shape.len != 2 or
            tensor.shape[0] != @as(i64, @intCast(rows)) or
            tensor.shape[1] != @as(i64, @intCast(cols)))
        {
            return error.CheckpointSizeMismatch;
        }
        const src = tensor.asFloat32();
        if (src.len != dest.len) return error.CheckpointSizeMismatch;
        @memcpy(dest, src);
        return true;
    }

    if (reader.header.tensors.contains(go_name)) {
        var tensor = try reader.readTensor(go_name);
        defer tensor.deinit();
        if (tensor.dtype != .f32) return error.InvalidCheckpoint;
        if (tensor.shape.len != 2 or
            tensor.shape[0] != @as(i64, @intCast(cols)) or
            tensor.shape[1] != @as(i64, @intCast(rows)))
        {
            return error.CheckpointSizeMismatch;
        }
        const src = tensor.asFloat32();
        if (src.len != dest.len) return error.CheckpointSizeMismatch;
        for (0..cols) |c| {
            for (0..rows) |r| {
                dest[r * cols + c] = src[c * rows + r];
            }
        }
        return true;
    }

    return false;
}

fn deinitMetalWeightStore(allocator: std.mem.Allocator, weight_store: *MetalWeightStore) void {
    if (comptime !build_options.enable_metal) return;
    metal_compute.deinitPrefetchQueue(weight_store);
    var it = weight_store.lazy_weights.iterator();
    while (it.next()) |entry| {
        allocator.free(entry.key_ptr.*);
        if (entry.value_ptr.host_loaded) |*loaded| loaded.deinit();
        if (entry.value_ptr.quantized_storage) |*storage| storage.deinit();
    }
    weight_store.lazy_weights.deinit(allocator);
}

fn evaluateBoundarySamplesStreaming(
    allocator: std.mem.Allocator,
    opts: *const Options,
    trainer: *FusedTrainer,
    cb: *const ComputeBackend,
    native_weight_store: *native_compute.WeightStore,
    metal_weight_store: *MetalWeightStore,
    tokenizer_opt: ?*TokenizerBatch,
    samples: []const fused_chunker_data.FusedSample,
    encoder_loaded: bool,
    use_metal: bool,
    lora_adapters: ?*const fused_chunker_lora.LoRAAdapterSet,
    eval_label: []const u8,
    eval_step: u64,
    fitted_threshold: ?f32,
) !fused_chunker_train.EvalSummary {
    if (encoder_loaded) {
        if (lora_adapters) |la| {
            try refreshLoRAWeightsForForward(allocator, use_metal, native_weight_store, metal_weight_store, la);
        }
    }

    var acc = fused_chunker_train.BoundaryEvalAccumulator{
        .calibrated_threshold = fused_chunker_loss.weightedCePositiveThreshold(trainer.loss_config.pos_weight),
        .fitted_threshold = fitted_threshold,
    };
    defer acc.deinit(allocator);
    const eval_start_ns = nowNs();
    const bs: usize = @intCast(opts.batch_size);
    const max_seq: usize = @intCast(opts.max_seq_len);
    const max_chunks: usize = @intCast(opts.max_chunks);
    const hidden_size: usize = @intCast(opts.hidden_size);
    const total_batches = (samples.len + bs - 1) / bs;

    var alignment_path: ?[]u8 = null;
    defer if (alignment_path) |path| allocator.free(path);
    var alignment_file: ?std.Io.File = null;
    if (opts.boundary_alignment_dump_dir) |dump_dir| {
        try compat.cwd().createDirPath(compat.io(), dump_dir);
        const path = try boundaryAlignmentDumpPath(allocator, dump_dir, eval_label, eval_step);
        alignment_path = path;
        alignment_file = try compat.cwd().createFile(compat.io(), path, .{ .truncate = true });
    }
    defer if (alignment_file) |*file| file.close(compat.io());
    var alignment_buf: [8192]u8 = undefined;
    var alignment_writer = if (alignment_file) |*file| file.writerStreaming(compat.io(), &alignment_buf) else null;
    defer if (alignment_writer) |*writer| writer.interface.flush() catch {};

    var sample_idx: usize = 0;
    var batch_idx: usize = 0;
    while (sample_idx < samples.len) {
        const end = @min(sample_idx + bs, samples.len);
        const count = end - sample_idx;

        const indices = try allocator.alloc(usize, count);
        defer allocator.free(indices);
        for (0..count) |k| indices[k] = sample_idx + k;

        var batch: fused_chunker_data.FusedBatch = undefined;
        if (tokenizer_opt) |tb| {
            var tok_ctx = tb.makeTokenFnCtx();
            batch = try fused_chunker_data.assembleTokenBatch(
                allocator,
                samples,
                indices,
                max_seq,
                max_chunks,
                &tok_ctx,
                TokenFnCtx.call,
            );
        } else {
            batch = try fused_chunker_data.assembleTokenBatch(
                allocator,
                samples,
                indices,
                max_seq,
                max_chunks,
                {},
                dummyTokenFn,
            );
        }
        defer batch.deinit(allocator);

        const total_tokens = count * max_seq;
        const features = if (encoder_loaded and tokenizer_opt != null) blk: {
            const ids_i64 = try allocator.alloc(i64, total_tokens);
            defer allocator.free(ids_i64);
            for (batch.input_ids[0..total_tokens], ids_i64) |id32, *id64| id64.* = @intCast(id32);

            const mask_i64 = try allocator.alloc(i64, total_tokens);
            defer allocator.free(mask_i64);
            for (batch.attention_mask[0..total_tokens], mask_i64) |m32, *m64| m64.* = @intCast(m32);

            const lora_rank = if (lora_adapters) |la| la.config.rank else 0;
            const lora_alpha = if (lora_adapters) |la| la.config.alpha else 0.0;
            const bert_config = modern_bert.Config{
                .hidden_size = opts.hidden_size,
                .num_hidden_layers = opts.num_layers,
                .intermediate_size = opts.intermediate_size,
                .lora_rank = lora_rank,
                .lora_alpha = lora_alpha,
            };
            break :blk try modern_bert.forward(
                cb,
                allocator,
                bert_config,
                ids_i64,
                mask_i64,
                count,
                max_seq,
            );
        } else zblk: {
            const zeros = try allocator.alloc(f32, total_tokens * hidden_size);
            @memset(zeros, 0);
            break :zblk zeros;
        };
        defer allocator.free(features);

        const labels = try allocator.alloc(f32, total_tokens * 2);
        defer allocator.free(labels);
        for (0..total_tokens) |t| {
            const is_boundary = batch.boundary_labels[t] > 0.5;
            labels[t * 2 + 0] = if (is_boundary) 0.0 else 1.0;
            labels[t * 2 + 1] = if (is_boundary) 1.0 else 0.0;
        }

        const mask = try allocator.alloc(f32, total_tokens);
        defer allocator.free(mask);
        for (0..total_tokens) |t| {
            mask[t] = if (batch.attention_mask[t] != 0) 1.0 else 0.0;
        }

        const logits = try trainer.evaluateBoundaryLogitsOwned(allocator, features, total_tokens);
        defer allocator.free(logits);

        try acc.addLogitsBySample(allocator, logits, labels, mask, max_seq);
        if (alignment_writer) |*writer| {
            try writeBoundaryAlignmentDumpBatch(
                allocator,
                &writer.interface,
                eval_label,
                eval_step,
                batch_idx + 1,
                &batch,
                logits,
                features,
                hidden_size,
                opts.boundary_alignment_dump_top_k,
            );
        }

        batch_idx += 1;
        sample_idx = end;
        if (batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == total_batches) {
            print("validation batch {d}/{d} | elapsed_ms {d:.2}\n", .{
                batch_idx,
                total_batches,
                nsToMs(elapsedNs(eval_start_ns)),
            });
        }
    }

    return try acc.finish(allocator);
}

// ---------------------------------------------------------------------------
// Core training routine
// ---------------------------------------------------------------------------

fn run(allocator: std.mem.Allocator, input_opts: Options) !void {
    var opts = input_opts;

    // ------------------------------------------------------------------
    // 1. Create output directory
    // ------------------------------------------------------------------
    try compat.cwd().createDirPath(compat.io(), opts.output_dir);

    const default_metrics_path = try std.fs.path.join(allocator, &.{ opts.output_dir, fused_metrics_file_name });
    defer allocator.free(default_metrics_path);
    const default_manifest_path = try std.fs.path.join(allocator, &.{ opts.output_dir, fused_manifest_file_name });
    defer allocator.free(default_manifest_path);
    const metrics_path = opts.report_to orelse default_metrics_path;
    const manifest_path = opts.manifest_path orelse default_manifest_path;

    var metrics_file = try compat.cwd().createFile(compat.io(), metrics_path, .{ .truncate = true });
    defer metrics_file.close(compat.io());
    var metrics_buf: [8192]u8 = undefined;
    var metrics_writer = metrics_file.writerStreaming(compat.io(), &metrics_buf);

    const effective_boundary_dropout: f32 = if (opts.deterministic) 0.0 else opts.boundary_dropout;
    const effective_neftune_alpha: f32 = if (opts.deterministic) 0.0 else opts.neftune_alpha;
    const effective_encoder_neftune_alpha: f32 = if (opts.encoder_neftune) effective_neftune_alpha else 0.0;

    print("train-fused-chunker data={s} output={s} epochs={d} batch_size={d} lr={d} boundary_head_lr_multiplier={d} warmup_steps={d} lr_total_steps={d} weight_decay={d} beta1={d} beta2={d} adam_epsilon={d} hidden={d} layers={d} max_seq_len={d} max_chunks={d}\n", .{
        opts.data_path,
        opts.output_dir,
        opts.epochs,
        opts.batch_size,
        opts.learning_rate,
        opts.boundary_head_lr_multiplier,
        opts.warmup_steps,
        opts.lr_total_steps,
        opts.weight_decay,
        opts.beta1,
        opts.beta2,
        opts.adam_epsilon,
        opts.hidden_size,
        opts.num_layers,
        opts.max_seq_len,
        opts.max_chunks,
    });
    print("train-fused-chunker loss_knobs loss_type={s} pos_weight={d} pos_weight_auto={} boundary_rank_loss_weight={d} boundary_rank_loss_margin={d} boundary_rank_loss_top_k={d} boundary_same_token_rank_loss_weight={d} boundary_same_token_rank_loss_top_k={d} boundary_same_token_negative_weight={d} boundary_same_token_negative_top_k={d} boundary_candidate_rank_loss_weight={d} boundary_candidate_rank_loss_top_k={d} boundary_candidate_negative_weight={d} boundary_gold_count_rank_loss_weight={d} boundary_gold_count_rank_loss_margin={d} boundary_gold_count_rank_loss_negative_multiplier={d} boundary_local_window_loss_weight={d} boundary_local_window_radius={d}\n", .{
        @tagName(opts.boundary_loss_type),
        opts.boundary_pos_weight,
        opts.boundary_pos_weight_auto,
        opts.boundary_rank_loss_weight,
        opts.boundary_rank_loss_margin,
        opts.boundary_rank_loss_top_k,
        opts.boundary_same_token_rank_loss_weight,
        opts.boundary_same_token_rank_loss_top_k,
        opts.boundary_same_token_negative_weight,
        opts.boundary_same_token_negative_top_k,
        opts.boundary_candidate_rank_loss_weight,
        opts.boundary_candidate_rank_loss_top_k,
        opts.boundary_candidate_negative_weight,
        opts.boundary_gold_count_rank_loss_weight,
        opts.boundary_gold_count_rank_loss_margin,
        opts.boundary_gold_count_rank_loss_negative_multiplier,
        opts.boundary_local_window_loss_weight,
        opts.boundary_local_window_radius,
    });
    print("train-fused-chunker runtime_knobs boundary_dropout={d} lambda_embed={d} boundary_focus_epochs={d} boundary_focus_lambda_embed={d} optimizer={s} log_every={d} checkpoint_every_steps={d} eval_every_steps={d} seed={d} lisa_sample={d} lisa_top_k={d} lora_train_top_k={d} lora_start_epoch={d} encoder_vjp={s} layers_per_segment={d}\n", .{
        effective_boundary_dropout,
        opts.lambda_embed,
        opts.boundary_focus_epochs,
        opts.boundary_focus_lambda_embed,
        if (opts.schedule_free) "schedule-free" else "adamw",
        opts.log_every,
        opts.checkpoint_every_steps,
        opts.eval_every_steps,
        opts.seed,
        opts.lisa_sample_layers,
        opts.lisa_top_k,
        opts.lora_train_top_k,
        opts.lora_start_epoch,
        encoderVJPModeName(opts.encoder_vjp),
        opts.layers_per_segment,
    });
    print("encoder_neftune={s} configured_neftune_alpha={d} applied_encoder_neftune_alpha={d}\n", .{
        enabledName(effective_encoder_neftune_alpha > 0.0),
        effective_neftune_alpha,
        effective_encoder_neftune_alpha,
    });
    print("train-fused-chunker dense_contrastive_focal gamma={d} alpha={d}\n", .{
        opts.contrastive_focal_gamma,
        opts.contrastive_focal_alpha,
    });
    print("contrastive_grad_path={s}\n", .{fused_chunker_train.contrastive_gradient_path});
    print("step_eval_max_examples={d}\n", .{opts.step_eval_max_examples});
    print("step_train_eval_max_examples={d}\n", .{opts.step_train_eval_max_examples});
    print("max_steps={d}\n", .{opts.max_steps});
    print("telemetry manifest={s} metrics={s} deterministic={} memory_sample_every={d} memory_warn_rss_bytes={d} memory_abort_rss_bytes={d}\n", .{
        manifest_path,
        metrics_path,
        opts.deterministic,
        opts.memory_sample_every,
        opts.memory_warn_rss_bytes,
        opts.memory_abort_rss_bytes,
    });
    printPhase20ParityReport(opts);

    if (opts.model_dir) |mdir| {
        print("model_dir={s}\n", .{mdir});
    } else {
        print("model_dir=none (encoder features will be zero-filled)\n", .{});
    }

    // ------------------------------------------------------------------
    // 2. Set up a minimal ComputeBackend for graph-based boundary head ops
    //    (also used for the encoder forward pass when weights are loaded)
    // ------------------------------------------------------------------
    var weight_store = native_compute.WeightStore{
        .allocator = allocator,
        .resident_weights = .{},
        .lazy_weights = .{},
    };
    defer {
        native_compute.deinitPrefetchQueue(&weight_store);
        var it = weight_store.resident_weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit();
        }
        weight_store.resident_weights.deinit(allocator);
    }

    const metal_available = if (comptime build_options.enable_metal) metal_runtime.metalDeviceAvailable() else false;
    if (platform.env.getenvBoolDefault("TERMITE_MPSGRAPH_SMOKE", false)) {
        const mpsgraph_available = if (comptime build_options.enable_metal) metal_runtime.mpsGraphAvailable() else false;
        if (mpsgraph_available) {
            const smoke = metal_runtime.mpsGraphSmokeAdd(2.0, 3.0);
            if (smoke) |value| {
                print("mpsgraph_smoke available=true add_2_3={d:.6}\n", .{value});
            } else {
                print("mpsgraph_smoke available=true add_2_3=failed\n", .{});
            }
        } else {
            print("mpsgraph_smoke available=false\n", .{});
        }
    }
    const selected_backend: FusedBackend = switch (opts.backend) {
        .auto => if (metal_available) .metal else .native,
        .metal => blk: {
            if (!metal_available) {
                print("error: --backend metal requested but Metal is not compiled in or no Metal device is available\n", .{});
                std.process.exit(1);
            }
            break :blk .metal;
        },
        .native => .native,
    };
    const use_metal = selected_backend == .metal;

    if (opts.mixed_precision) {
        std.debug.print("Warning: --mixed-precision is not supported by the direct Metal fused-chunker trainer yet, ignoring.\n", .{});
    }

    // Declare both backends at outer scope so their addresses are stable for
    // the ComputeBackend vtable pointer that FusedTrainer holds.
    var native_backend = native_compute.NativeCompute.init(allocator, &weight_store, null);

    var metal_weight_store: MetalWeightStore = undefined;
    var metal_backend: if (build_options.enable_metal) metal_compute.MetalCompute else void = undefined;

    const cb: ComputeBackend = if (use_metal) blk: {
        if (comptime build_options.enable_metal) {
            metal_weight_store = MetalWeightStore{
                .allocator = allocator,
                .prefix = "",
                .lazy_weights = .{},
            };
            metal_compute.initPrefetchQueue(&metal_weight_store, allocator);
            metal_backend = try metal_compute.MetalCompute.init(allocator, &metal_weight_store, null);
            break :blk metal_backend.computeBackend();
        } else unreachable;
    } else native_backend.computeBackend();
    defer if (use_metal) {
        if (comptime build_options.enable_metal) deinitMetalWeightStore(allocator, &metal_weight_store);
    };
    defer if (use_metal) {
        if (comptime build_options.enable_metal) metal_backend.deinit();
    };

    print("backend: {s}\n", .{if (use_metal) "metal" else "native"});
    if (platform.env.getenvBoolDefault("TERMITE_MPSGRAPH_LINEAR_VJP_PARITY", false)) {
        try runMpsGraphLinearVJPParityDiagnostic(allocator, &cb);
        if (platform.env.getenvBoolDefault("TERMITE_MPSGRAPH_LINEAR_VJP_PARITY_EXIT", false)) return;
    }

    // ------------------------------------------------------------------
    // 2a. Tokenizer loading
    // ------------------------------------------------------------------
    var tokenizer_opt: ?TokenizerBatch = null;
    defer if (tokenizer_opt) |*tb| tb.deinit();

    if (opts.model_dir) |mdir| {
        tokenizer_opt = TokenizerBatch.loadFromDir(allocator, mdir, opts.max_seq_len) catch |err| blk: {
            print("warning: could not load tokenizer from {s}: {}\n", .{ mdir, err });
            break :blk null;
        };
        if (tokenizer_opt != null) print("tokenizer loaded from {s}\n", .{mdir});
    }

    // ------------------------------------------------------------------
    // 2b. Weight loading into WeightStore
    // ------------------------------------------------------------------
    var encoder_loaded = false;
    if (opts.model_dir) |mdir| {
        var path_buf: [512]u8 = undefined;
        const st_path = std.fmt.bufPrint(&path_buf, "{s}/model.safetensors", .{mdir}) catch null;
        if (st_path) |p| {
            const exists = compat.cwd().statFile(compat.io(), p, .{}) catch null;
            if (exists != null) {
                if (use_metal) {
                    if (loadSafetensorsIntoMetalStore(allocator, &metal_weight_store, p)) |loaded_count| {
                        encoder_loaded = true;
                        print("loaded {d} weights into Metal store from {s}\n", .{ loaded_count, p });
                    } else |err| {
                        print("warning: could not load weights into Metal store from {s}: {}\n", .{ p, err });
                    }
                } else if (SafetensorsSource.initAbsolute(allocator, p)) |src| {
                    var source_ptr = src;
                    defer source_ptr.weightSource().deinit();
                    const ws = source_ptr.weightSource();
                    if (ws.listNames(allocator)) |names| {
                        defer allocator.free(names);
                        var load_ok = true;
                        var loaded_count: usize = 0;
                        for (names) |name| {
                            if (ws.getTensor(name)) |lw| {
                                const owned_name = normalizeModernBertWeightName(allocator, name) catch {
                                    load_ok = false;
                                    break;
                                };
                                var store_loaded = lw;
                                var owns_store_loaded = false;
                                if (isGoFusedEmbeddingWeightName(owned_name)) {
                                    store_loaded = cloneLoadedWeight(allocator, lw) catch {
                                        allocator.free(owned_name);
                                        load_ok = false;
                                        break;
                                    };
                                    owns_store_loaded = true;
                                    transposeLoadedEmbeddingWeightLikeGoFused(allocator, &store_loaded) catch {
                                        store_loaded.deinit();
                                        allocator.free(owned_name);
                                        load_ok = false;
                                        break;
                                    };
                                }
                                weight_store.resident_weights.put(allocator, owned_name, store_loaded) catch {
                                    if (owns_store_loaded) store_loaded.deinit();
                                    allocator.free(owned_name);
                                    load_ok = false;
                                    break;
                                };
                                loaded_count += 1;
                                loaded_count += insertModernBertQkvSplitsIntoNativeStore(allocator, &weight_store, owned_name, lw) catch {
                                    load_ok = false;
                                    break;
                                };
                            } else |_| {}
                        }
                        if (load_ok) {
                            encoder_loaded = true;
                            print("loaded {d} weights from {s}\n", .{ loaded_count, p });
                        }
                    } else |err| {
                        print("warning: could not list weights from {s}: {}\n", .{ p, err });
                    }
                } else |err| {
                    print("warning: could not open {s}: {}\n", .{ p, err });
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 2c. LoRA adapter init
    // ------------------------------------------------------------------
    var lora_adapters_opt: ?fused_chunker_lora.LoRAAdapterSet = null;
    defer if (lora_adapters_opt) |*la| la.deinit();

    if (opts.lora_rank > 0) {
        lora_adapters_opt = try fused_chunker_lora.LoRAAdapterSet.init(
            allocator,
            fused_chunker_lora.LoRAConfig{
                .rank = opts.lora_rank,
                .alpha = opts.lora_alpha,
                .target_modules = fused_chunker_lora_target_modules[0..],
                .num_layers = opts.num_layers,
                .lora_plus_ratio = opts.lora_plus_ratio,
            },
            @intCast(opts.hidden_size),
            @intCast(opts.intermediate_size),
        );
        print("LoRA adapters: rank={d} alpha={d} target_modules=query_proj,value_proj,key_proj,out_proj,wo lora_plus_ratio={d:.1}", .{ opts.lora_rank, opts.lora_alpha, opts.lora_plus_ratio });
        if (opts.lisa_sample_layers > 0) {
            print(" lisa_sample={d} lisa_top_k={d}", .{ opts.lisa_sample_layers, opts.lisa_top_k });
        }
        if (opts.lora_train_top_k > 0) {
            print(" lora_train_top_k={d}", .{opts.lora_train_top_k});
        }
        if (opts.encoder_vjp != .direct) {
            print(" encoder_vjp={s}", .{encoderVJPModeName(opts.encoder_vjp)});
            if (usesFullEncoderVJP(opts.encoder_vjp)) {
                print(" layers_per_segment={d}", .{opts.layers_per_segment});
            }
        }
        print("\n", .{});
    }

    // ------------------------------------------------------------------
    // 2d. LoRA optimizer state
    // ------------------------------------------------------------------
    var lora_opt_state = optimizers.OptimizerState.init(allocator);
    defer lora_opt_state.deinit();

    var last_layer_vjp_session: ?segmented_encoder.ModernBertSegmentVJPSession = null;
    defer if (last_layer_vjp_session) |*session| session.deinit();
    var full_vjp_sessions: ?[]?segmented_encoder.ModernBertSegmentVJPSession = null;
    defer if (full_vjp_sessions) |sessions| {
        for (sessions) |*slot| {
            if (slot.*) |*session| session.deinit();
        }
        allocator.free(sessions);
    };
    var segment_vjp_parity_pending = platform.env.getenvBoolDefault("TERMITE_SEGMENT_VJP_PARITY", false);
    const segment_vjp_parity_layer = getenvU32OrNull("TERMITE_SEGMENT_VJP_PARITY_LAYER");
    if (usesFullEncoderVJP(opts.encoder_vjp) and opts.lora_rank > 0) {
        const sessions = try allocator.alloc(?segmented_encoder.ModernBertSegmentVJPSession, opts.num_layers);
        errdefer allocator.free(sessions);
        for (sessions) |*slot| slot.* = null;
        full_vjp_sessions = sessions;
    }

    // Opt-in compiled segmented encoder forward (P1). Sessions are cached per
    // segment start layer, like the full-VJP sessions above. A first-failure
    // latch falls the step back to the eager capture forward.
    var compiled_forward_sessions: ?[]?segmented_encoder.ModernBertSegmentForwardSession = null;
    defer if (compiled_forward_sessions) |sessions| {
        for (sessions) |*slot| {
            if (slot.*) |*session| session.deinit();
        }
        allocator.free(sessions);
    };
    var compiled_forward_failed = false;
    const compiled_forward_check = platform.env.getenvBoolDefault(
        "ANTFLY_FUSED_CHUNKER_COMPILED_FORWARD_CHECK",
        false,
    );
    if (opts.compiled_segment_forward and opts.lora_rank > 0) {
        const sessions = try allocator.alloc(?segmented_encoder.ModernBertSegmentForwardSession, opts.num_layers);
        errdefer allocator.free(sessions);
        for (sessions) |*slot| slot.* = null;
        compiled_forward_sessions = sessions;
    }

    // ------------------------------------------------------------------
    // 2e. SPLADE projection weight W and Adam state
    // ------------------------------------------------------------------
    const splade_vocab_size: usize = 50368;

    var splade_w: ?[]f32 = null;
    var splade_adam_m: ?[]f32 = null;
    var splade_adam_v: ?[]f32 = null;
    var splade_adam_step: u64 = 0;

    if (opts.splade and encoder_loaded) {
        const w_size = splade_vocab_size * @as(usize, opts.hidden_size);
        const w_alloc = try allocator.alloc(f32, w_size);
        errdefer allocator.free(w_alloc);
        const m_alloc = try allocator.alloc(f32, w_size);
        errdefer allocator.free(m_alloc);
        const v_alloc = try allocator.alloc(f32, w_size);
        // No errdefer needed here: if this fails, the outer defers never see
        // non-null values, so there's no double-free risk; m_alloc and w_alloc
        // are freed by their own errdefers above.

        @memset(m_alloc, 0);
        @memset(v_alloc, 0);

        // Kaiming uniform init: scale = sqrt(2 / hidden_size)
        const init_std: f32 = @sqrt(2.0 / @as(f32, @floatFromInt(opts.hidden_size)));
        var splade_prng = std.Random.DefaultPrng.init(opts.seed ^ 0x5F1ADE00);
        const splade_rng = splade_prng.random();
        for (w_alloc) |*w_val| {
            w_val.* = (splade_rng.float(f32) * 2.0 - 1.0) * init_std;
        }

        // Commit all three allocations (only after all three succeed).
        splade_w = w_alloc;
        splade_adam_m = m_alloc;
        splade_adam_v = v_alloc;
        print("SPLADE weight W initialized: vocab={d} hidden={d} init_std={d}\n", .{ splade_vocab_size, opts.hidden_size, init_std });
    }

    defer if (splade_w) |w| allocator.free(w);
    defer if (splade_adam_m) |m| allocator.free(m);
    defer if (splade_adam_v) |v| allocator.free(v);

    // ------------------------------------------------------------------
    // 3. Load training samples
    // ------------------------------------------------------------------
    print("loading samples from {s} (split={s})...\n", .{ opts.data_path, opts.split });
    var loaded = try fused_chunker_data.loadSamples(allocator, opts.data_path, opts.split);
    defer loaded.deinit();

    const samples = loaded.samples;
    if (samples.len == 0) {
        print("error: no samples found\n", .{});
        std.process.exit(1);
    }

    const train_stats = fused_chunker_data.computeStats(samples);
    print("loaded {d} samples  avg_chars={d:.0}  avg_chunks={d:.1}  min_chunks={d}  max_chunks={d}  contrastive_pos_samples={d}  boundary_target_samples={d}  boundary_targets={d}\n", .{
        train_stats.num_samples,
        train_stats.avg_text_chars,
        train_stats.avg_chunks_per_sample,
        train_stats.min_chunks,
        train_stats.max_chunks,
        train_stats.samples_with_contrastive_positives,
        train_stats.samples_with_boundary_targets,
        train_stats.total_boundary_targets,
    });

    if (opts.boundary_pos_weight_auto) {
        const tokenizer = if (tokenizer_opt) |*tb| tb else return error.AutoBoundaryPosWeightRequiresTokenizer;
        const estimate = try estimateBoundaryPosWeight(
            allocator,
            samples,
            tokenizer,
            opts.boundary_pos_weight_auto_max_examples,
            @intCast(opts.batch_size),
            @intCast(opts.max_seq_len),
            @intCast(opts.max_chunks),
        );
        opts.boundary_pos_weight = estimate.balanced_pos_weight;
        opts.boundary_pos_weight_auto_observed_examples = estimate.examples;
        opts.boundary_pos_weight_auto_valid_tokens = estimate.valid_tokens;
        opts.boundary_pos_weight_auto_gold_tokens = estimate.gold_tokens;
        opts.boundary_pos_weight_auto_gold_rate = estimate.gold_rate;
        print("auto_boundary_pos_weight examples={d} batches={d} valid_tokens={d} gold_tokens={d} gold_rate={d:.8} pos_weight={d:.6}\n", .{
            estimate.examples,
            estimate.batches,
            estimate.valid_tokens,
            estimate.gold_tokens,
            estimate.gold_rate,
            estimate.balanced_pos_weight,
        });
    }

    var val_loaded_opt: ?fused_chunker_data.LoadedSamples = null;
    defer if (val_loaded_opt) |*val_loaded| val_loaded.deinit();
    var val_samples: []const fused_chunker_data.FusedSample = &.{};
    var validation_stats: ?fused_chunker_data.FusedDatasetStats = null;
    if (opts.val_data_path) |val_path| {
        print("loading validation samples from {s} (split={s})...\n", .{ val_path, opts.val_split });
        val_loaded_opt = try fused_chunker_data.loadSamples(allocator, val_path, opts.val_split);
        val_samples = val_loaded_opt.?.samples;
        if (val_samples.len == 0) {
            print("warning: no validation samples found in {s} for split {s}; validation disabled\n", .{ val_path, opts.val_split });
            if (val_loaded_opt) |*val_loaded| val_loaded.deinit();
            val_loaded_opt = null;
            val_samples = &.{};
        } else {
            const val_stats = fused_chunker_data.computeStats(val_samples);
            validation_stats = val_stats;
            print("loaded {d} validation samples  avg_chars={d:.0}  avg_chunks={d:.1}  min_chunks={d}  max_chunks={d}  contrastive_pos_samples={d}  boundary_target_samples={d}  boundary_targets={d}\n", .{
                val_stats.num_samples,
                val_stats.avg_text_chars,
                val_stats.avg_chunks_per_sample,
                val_stats.min_chunks,
                val_stats.max_chunks,
                val_stats.samples_with_contrastive_positives,
                val_stats.samples_with_boundary_targets,
                val_stats.total_boundary_targets,
            });
        }
    }

    // ------------------------------------------------------------------
    // 4. Build FusedTrainingConfig and FusedTrainer
    // ------------------------------------------------------------------

    // Match the Go fused trainer: the final partial batch counts as a step.
    const batch_size_for_steps: usize = @intCast(@max(1, opts.batch_size));
    const steps_per_epoch = @max(@as(u32, 1), @as(u32, @intCast((samples.len + batch_size_for_steps - 1) / batch_size_for_steps)));
    const derived_total_steps = steps_per_epoch * opts.epochs;
    const total_steps = if (opts.lr_total_steps > 0) opts.lr_total_steps else derived_total_steps;
    const warmup_steps = opts.warmup_steps;
    if (warmup_steps > total_steps) {
        print("warning: warmup_steps={d} exceeds total_steps={d}; learning rate will remain in warmup for the full run\n", .{ warmup_steps, total_steps });
    }
    if (opts.lr_total_steps > 0 and opts.lr_total_steps != derived_total_steps) {
        print("lr schedule total_steps override: derived={d} using={d}\n", .{ derived_total_steps, opts.lr_total_steps });
    }

    // Parse --mrl-dims comma-separated string (e.g. "768,256,128") into a fixed-size array.
    var mrl_dims_buf: [8]u32 = undefined;
    var mrl_weights_buf: [8]f32 = undefined;
    var mrl_dims_count: usize = 0;
    if (opts.mrl) {
        var it = std.mem.splitScalar(u8, opts.mrl_dims_str, ',');
        while (it.next()) |token| {
            if (mrl_dims_count >= mrl_dims_buf.len) break;
            const trimmed = std.mem.trim(u8, token, " \t");
            if (trimmed.len == 0) continue;
            mrl_dims_buf[mrl_dims_count] = try std.fmt.parseUnsigned(u32, trimmed, 10);
            mrl_dims_count += 1;
        }
        if (mrl_dims_count > 0) {
            const weight = 1.0 / @as(f32, @floatFromInt(mrl_dims_count));
            for (mrl_weights_buf[0..mrl_dims_count]) |*w| w.* = weight;
        }
    }

    var loss_config = fused_chunker_loss.FusedLossConfig{};
    loss_config.enable_splade = opts.splade;
    loss_config.lambda_splade = opts.lambda_splade;
    loss_config.lambda_flops = opts.lambda_flops;
    loss_config.splade_focus_epoch = opts.splade_focus_epoch;
    loss_config.contrastive_focal_gamma = opts.contrastive_focal_gamma;
    loss_config.contrastive_focal_alpha = opts.contrastive_focal_alpha;
    loss_config.use_mrl = opts.mrl;
    if (opts.mrl and mrl_dims_count > 0) {
        loss_config.mrl_dims = mrl_dims_buf[0..mrl_dims_count];
        loss_config.mrl_weights = mrl_weights_buf[0..mrl_dims_count];
    }

    const config = FusedTrainingConfig{
        .max_seq_len = opts.max_seq_len,
        .max_chunks = opts.max_chunks,
        .hidden_size = opts.hidden_size,
        .embedding_dim = opts.hidden_size,
        .batch_size = opts.batch_size,
        .num_epochs = opts.epochs,
        .learning_rate = opts.learning_rate,
        .boundary_head_lr_multiplier = opts.boundary_head_lr_multiplier,
        .max_grad_norm = opts.max_grad_norm,
        .weight_decay = opts.weight_decay,
        .beta1 = opts.beta1,
        .beta2 = opts.beta2,
        .adam_epsilon = opts.adam_epsilon,
        .lambda_chunk = opts.lambda_chunk,
        .lambda_embed = opts.lambda_embed,
        .boundary_focus_epochs = opts.boundary_focus_epochs,
        .use_boundary_focal = opts.boundary_loss_type == .focal,
        .focal_gamma = opts.boundary_focal_gamma,
        .focal_alpha = opts.boundary_focal_alpha,
        .contrastive_focal_gamma = opts.contrastive_focal_gamma,
        .contrastive_focal_alpha = opts.contrastive_focal_alpha,
        .pos_weight = opts.boundary_pos_weight,
        .boundary_rank_loss_weight = opts.boundary_rank_loss_weight,
        .boundary_rank_loss_margin = opts.boundary_rank_loss_margin,
        .boundary_rank_loss_top_k = opts.boundary_rank_loss_top_k,
        .boundary_same_token_rank_loss_weight = opts.boundary_same_token_rank_loss_weight,
        .boundary_same_token_rank_loss_top_k = opts.boundary_same_token_rank_loss_top_k,
        .boundary_same_token_negative_weight = opts.boundary_same_token_negative_weight,
        .boundary_same_token_negative_top_k = opts.boundary_same_token_negative_top_k,
        .boundary_candidate_rank_loss_weight = opts.boundary_candidate_rank_loss_weight,
        .boundary_candidate_rank_loss_top_k = opts.boundary_candidate_rank_loss_top_k,
        .boundary_candidate_negative_weight = opts.boundary_candidate_negative_weight,
        .boundary_gold_count_rank_loss_weight = opts.boundary_gold_count_rank_loss_weight,
        .boundary_gold_count_rank_loss_margin = opts.boundary_gold_count_rank_loss_margin,
        .boundary_gold_count_rank_loss_negative_multiplier = opts.boundary_gold_count_rank_loss_negative_multiplier,
        .boundary_local_window_loss_weight = opts.boundary_local_window_loss_weight,
        .boundary_local_window_radius = opts.boundary_local_window_radius,
        .boundary_feature_mode = opts.boundary_feature_mode,
        .boundary_dropout = effective_boundary_dropout,
        .warmup_steps = warmup_steps,
        .total_steps = @max(1, total_steps),
        .seed = opts.seed,
        .step_log_every = opts.log_every,
        .checkpoint_every = opts.checkpoint_every,
        .checkpoint_every_steps = opts.checkpoint_every_steps,
        // Feature 2: gradient accumulation
        .grad_accum_steps = @max(1, opts.grad_accum),
        // Feature 3: schedule-free AdamW
        .use_schedule_free = opts.schedule_free,
        // Feature 1: XBM
        .xbm_capacity = opts.xbm_capacity,
        // Feature 5: NEFTune
        .neftune_alpha = effective_encoder_neftune_alpha,
        // Feature 6: LLRD
        .llrd_decay = opts.llrd_decay,
        // Feature 8: length bucketing
        .length_bucketing = opts.length_bucketing,
        .bucket_size = opts.bucket_size,
        .mixed_precision = opts.mixed_precision,
        // SPLADE
        .enable_splade = loss_config.enable_splade,
        .lambda_splade = loss_config.lambda_splade,
        .lambda_flops = loss_config.lambda_flops,
        .splade_focus_epoch = loss_config.splade_focus_epoch,
        // MRL
        .use_mrl = loss_config.use_mrl,
        .mrl_dims = loss_config.mrl_dims,
        .mrl_weights = loss_config.mrl_weights,
    };

    var trainer = try FusedTrainer.init(allocator, config, &cb);
    defer trainer.deinit();

    print("trainer ready  encoder_hidden={d} boundary_head_input={d} boundary_feature_mode={s} mlp_dim={d}\n", .{
        config.hidden_size,
        trainer.boundary_head.hidden_dim,
        fused_chunker_train.boundaryFeatureModeName(config.boundary_feature_mode),
        config.boundary_mlp_dim,
    });

    // ------------------------------------------------------------------
    // 4a. Resume from checkpoint (if --resume-from was supplied)
    // ------------------------------------------------------------------
    if (opts.resume_from.len > 0) {
        print("resuming weights from {s}\n", .{opts.resume_from});
        try trainer.loadCheckpoint(allocator, opts.resume_from);
        if (lora_adapters_opt) |*la| {
            const loaded_lora_tensors = try loadLoRAAdaptersFromCheckpoint(allocator, opts.resume_from, la);
            if (loaded_lora_tensors > 0) {
                print("restored {d} LoRA tensors from {s}\n", .{ loaded_lora_tensors, opts.resume_from });
            }
        }

        // Look for a companion optimizer-state file next to the checkpoint.
        // Convention: replace the extension of the checkpoint path with
        // "_optimizer.safetensors", e.g.
        //   checkpoint_final.safetensors -> checkpoint_final_optimizer.safetensors
        var opt_state_path_buf: [512]u8 = undefined;
        const opt_state_path = blk: {
            // Strip a trailing .safetensors or .bin extension if present, then
            // append the optimizer-state suffix.
            const base = if (std.mem.endsWith(u8, opts.resume_from, ".safetensors"))
                opts.resume_from[0 .. opts.resume_from.len - ".safetensors".len]
            else if (std.mem.endsWith(u8, opts.resume_from, ".bin"))
                opts.resume_from[0 .. opts.resume_from.len - ".bin".len]
            else
                opts.resume_from;
            break :blk std.fmt.bufPrint(&opt_state_path_buf, "{s}_optimizer.safetensors", .{base}) catch null;
        };
        if (opt_state_path) |p| {
            const exists = compat.cwd().statFile(compat.io(), p, .{}) catch null;
            if (exists != null) {
                print("restoring optimizer state from {s}\n", .{p});
                loadFusedOptimizerState(
                    allocator,
                    p,
                    &trainer,
                    if (lora_adapters_opt) |*la| la else null,
                    &lora_opt_state,
                ) catch |err| {
                    print("warning: could not load optimizer state from {s}: {}\n", .{ p, err });
                };
            }
        }
    }

    if (opts.debug_frozen_feature_probe) {
        var frozen_probe_memory_tracker = MemoryTracker{};
        _ = frozen_probe_memory_tracker.observe();
        try writeFusedTrainingManifest(
            allocator,
            manifest_path,
            &opts,
            if (use_metal) "metal" else "native",
            metrics_path,
            train_stats,
            validation_stats,
            "frozen_feature_probe_running",
            trainer.step_count,
            null,
            null,
            0,
            0,
            0.0,
            BoundaryThresholdRecommendation{},
            frozen_probe_memory_tracker.peak_resident_bytes,
        );

        const tokenizer_ptr: ?*TokenizerBatch = if (tokenizer_opt) |*tb| tb else null;
        try runFrozenFeatureProbe(
            allocator,
            &opts,
            &metrics_writer,
            &trainer,
            &cb,
            tokenizer_ptr,
            samples,
            val_samples,
            encoder_loaded,
        );

        _ = frozen_probe_memory_tracker.observe();
        try writeFusedTrainingManifest(
            allocator,
            manifest_path,
            &opts,
            if (use_metal) "metal" else "native",
            metrics_path,
            train_stats,
            validation_stats,
            "frozen_feature_probe_complete",
            trainer.step_count,
            null,
            null,
            0,
            0,
            0.0,
            BoundaryThresholdRecommendation{},
            frozen_probe_memory_tracker.peak_resident_bytes,
        );
        return;
    }

    // ------------------------------------------------------------------
    // 5. Training loop
    // ------------------------------------------------------------------

    // Build a mutable index array for shuffling
    var indices = try allocator.alloc(usize, samples.len);
    defer allocator.free(indices);
    resetIdentityIndices(indices);

    var prng = std.Random.DefaultPrng.init(opts.seed);
    const rng = prng.random();

    // Global step counter for NEFTune PRNG seeding - never resets across epochs
    // or checkpoint resumes.
    var global_neft_step: u64 = trainer.step_count;

    if (tokenizer_opt == null) {
        print("warning: tokenizer not loaded — using dummy zero-fill token_fn; boundary labels will be inactive until encoder is wired\n", .{});
    }

    const max_chunks: usize = @intCast(opts.max_chunks);
    const batch_sz: usize = @intCast(opts.batch_size);
    const max_seq: usize = @intCast(opts.max_seq_len);
    var best_val_f1: f32 = 0.0;
    var best_val_epoch: u32 = 0;
    var best_val_step: u32 = 0;
    var boundary_threshold_recommendation = BoundaryThresholdRecommendation{};
    var last_validation_summary: ?fused_chunker_train.EvalSummary = null;
    var last_validation_event: []const u8 = "";
    var last_validation_epoch_index: usize = 0;
    var last_validation_step: u64 = 0;
    var last_validation_samples: usize = 0;
    var last_step_loss: ?f32 = null;
    var memory_tracker = MemoryTracker{};
    _ = memory_tracker.observe();

    try writeFusedTrainingManifest(
        allocator,
        manifest_path,
        &opts,
        if (use_metal) "metal" else "native",
        metrics_path,
        train_stats,
        validation_stats,
        "running",
        trainer.step_count,
        null,
        null,
        @intCast(best_val_step),
        best_val_epoch,
        best_val_f1,
        boundary_threshold_recommendation,
        memory_tracker.peak_resident_bytes,
    );

    const resume_step_count: u32 = trainer.step_count;
    const resume_epoch: usize = if (steps_per_epoch == 0)
        0
    else
        @min(@as(usize, @intCast(resume_step_count / steps_per_epoch)), @as(usize, @intCast(opts.epochs)));
    const resume_step_in_epoch: u32 = if (steps_per_epoch == 0)
        0
    else
        resume_step_count % steps_per_epoch;
    const resume_batch_offset: usize = @min(@as(usize, @intCast(resume_step_in_epoch)) * batch_sz, samples.len);
    if (opts.debug_batch_offset) |requested_offset| {
        if (requested_offset > samples.len) return error.InvalidDebugBatchOffset;
    }
    const effective_resume_batch_offset: usize = opts.debug_batch_offset orelse resume_batch_offset;
    if (resume_step_count > 0) {
        print(
            "resume_position global_step={d} steps_per_epoch={d} resume_epoch={d} resume_step_in_epoch={d} resume_batch_offset={d}\n",
            .{
                resume_step_count,
                steps_per_epoch,
                resume_epoch + 1,
                resume_step_in_epoch,
                resume_batch_offset,
            },
        );
    }
    if (opts.debug_batch_offset) |requested_offset| {
        print(
            "debug_batch_offset_override requested={d} effective_batch_offset={d} resume_batch_offset={d}\n",
            .{ requested_offset, effective_resume_batch_offset, resume_batch_offset },
        );
    }

    var stop_training = false;
    for (0..opts.epochs) |epoch| {
        const epoch_lambda_embed: f32 = if (epoch < opts.boundary_focus_epochs)
            @min(opts.lambda_embed, opts.boundary_focus_lambda_embed)
        else
            opts.lambda_embed;
        trainer.loss_config.lambda_embed = epoch_lambda_embed;
        print("epoch {d}/{d} curriculum lambda_chunk={d} lambda_embed={d}\n", .{
            epoch + 1,
            opts.epochs,
            trainer.loss_config.lambda_chunk,
            trainer.loss_config.lambda_embed,
        });

        const order_preview_len = @min(indices.len, 16);
        if (opts.deterministic) {
            resetIdentityIndices(indices);
            print("epoch_order epoch={d}/{d} shuffle_mode=deterministic shuffle_seed=none reset_indices=true preview_len={d} ", .{
                epoch + 1,
                opts.epochs,
                order_preview_len,
            });
            printIndexSlice("first_indices", indices[0..order_preview_len]);
        } else if (opts.go_epoch_shuffle) {
            resetIdentityIndices(indices);
            const epoch_shuffle_seed = opts.seed + @as(u64, @intCast(epoch)) + 1;
            var epoch_prng = std.Random.DefaultPrng.init(epoch_shuffle_seed);
            shuffleIndicesFisherYates(indices, epoch_prng.random());
            print("epoch_order epoch={d}/{d} shuffle_mode=go_epoch_reset shuffle_seed={d} reset_indices=true preview_len={d} ", .{
                epoch + 1,
                opts.epochs,
                epoch_shuffle_seed,
                order_preview_len,
            });
            printIndexSlice("first_indices", indices[0..order_preview_len]);
        } else {
            shuffleIndicesFisherYates(indices, rng);
            print("epoch_order epoch={d}/{d} shuffle_mode=legacy_stream shuffle_seed=stream reset_indices=false preview_len={d} ", .{
                epoch + 1,
                opts.epochs,
                order_preview_len,
            });
            printIndexSlice("first_indices", indices[0..order_preview_len]);
        }

        // Feature 8: Length bucketing — sort within windows after shuffle
        var bucketed_indices: ?[]usize = null;
        defer if (bucketed_indices) |b| allocator.free(b);
        const active_indices: []const usize = if (opts.length_bucketing) blk: {
            bucketed_indices = try fused_chunker_data.sortByLength(
                allocator,
                samples,
                indices,
                opts.bucket_size,
            );
            break :blk bucketed_indices.?;
        } else indices;

        var epoch_timing = TimingTotals{};
        if (epoch < resume_epoch) {
            if (resume_step_count > 0) {
                print("resume_skip_epoch epoch={d}/{d} already_covered_by_global_step={d}\n", .{ epoch + 1, opts.epochs, resume_step_count });
            }
            continue;
        }

        var step: u32 = if (epoch == resume_epoch) resume_step_in_epoch else 0;
        var batch_start: usize = if (epoch == resume_epoch) effective_resume_batch_offset else 0;
        if (resume_step_count > 0 and epoch == resume_epoch and batch_start > 0) {
            print(
                "resume_epoch_start epoch={d}/{d} local_step={d} batch_start={d}/{d}\n",
                .{ epoch + 1, opts.epochs, step, batch_start, active_indices.len },
            );
        }

        while (batch_start < active_indices.len) {
            const step_start_ns = nowNs();
            var step_timing = StepTiming{};
            const lora_training_active = lora_adapters_opt != null and epoch >= @as(usize, opts.lora_start_epoch);

            const batch_end = @min(batch_start + batch_sz, active_indices.len);
            const batch_indices = active_indices[batch_start..batch_end];
            batch_start = batch_end;

            // Assemble token batch — use real tokenizer when available, dummy otherwise.
            const batch_assemble_start_ns = nowNs();
            var batch: fused_chunker_data.FusedBatch = undefined;
            if (tokenizer_opt) |*tb| {
                var tok_ctx = tb.makeTokenFnCtx();
                batch = try fused_chunker_data.assembleTokenBatch(
                    allocator,
                    samples,
                    batch_indices,
                    max_seq,
                    max_chunks,
                    &tok_ctx,
                    TokenFnCtx.call,
                );
            } else {
                batch = try fused_chunker_data.assembleTokenBatch(
                    allocator,
                    samples,
                    batch_indices,
                    max_seq,
                    max_chunks,
                    {},
                    dummyTokenFn,
                );
            }
            step_timing.batch_ns = elapsedNs(batch_assemble_start_ns);
            defer batch.deinit(allocator);

            const actual_batch = batch.batch_size;
            // total_tokens: use allocated shape [actual_batch * max_seq] so tensors
            // stay valid regardless of how many real tokens the tokenizer produced.
            const total_tokens: usize = actual_batch * max_seq;

            var active_lora_layers: ?[]bool = null;
            defer if (active_lora_layers) |layers| allocator.free(layers);
            if (lora_adapters_opt) |*la| {
                active_lora_layers = try selectActiveLoRALayersForStep(
                    allocator,
                    &opts,
                    la.config.num_layers,
                    @as(u64, trainer.step_count) + 1,
                );
                const repaired = sanitizeLoRAAdapterSet(la) + optimizers.sanitizeState(&lora_opt_state);
                if (repaired > 0) {
                    print("warning: sanitized {d} nonfinite LoRA adapter/optimizer values before forward at global_step={d}\n", .{
                        repaired,
                        trainer.step_count + 1,
                    });
                }
            }

            // ------------------------------------------------------------------
            // Phase 3A: Pre-merge LoRA weights into WeightStore before encoder
            // forward so that the compute backend sees the merged weights.
            // Skip when linearLoRA is available: it applies the LoRA delta inline
            // without needing the base weights mutated (Fix 3).
            // ------------------------------------------------------------------
            const lora_refresh_start_ns = nowNs();
            var lora_originals = std.StringHashMapUnmanaged([]u8).empty;
            var metal_lora_originals = std.StringHashMapUnmanaged([]u8).empty;
            var lora_merged = false;
            var metal_lora_merged = false;
            if (cb.vtable.linearLoRA == null) {
                if (lora_adapters_opt) |*la| {
                    if (encoder_loaded) {
                        if (use_metal) {
                            if (comptime build_options.enable_metal) {
                                metal_lora_originals = try mergeLoRAIntoMetalWeights(allocator, &metal_weight_store, la);
                                metal_lora_merged = true;
                            }
                        } else {
                            lora_originals = try mergeLoRAIntoWeights(allocator, &weight_store, la);
                            lora_merged = true;
                        }
                    }
                }
            }
            defer if (lora_merged) restoreLoRAWeights(allocator, &weight_store, &lora_originals);
            defer if (metal_lora_merged) {
                if (comptime build_options.enable_metal) restoreMetalLoRAWeights(allocator, &metal_weight_store, &metal_lora_originals);
            };

            // ------------------------------------------------------------------
            // Fix 2: Register LoRA A/B tensors in the active WeightStore so that
            // modern_bert.zig can retrieve them via cb.getWeight for linearLoRA.
            // Key names: "model.layers.{N}.{scope}.{projection}.lora_{a,b}".
            // The adapter modules map to ModernBERT scopes via loraRuntimeTarget.
            // ------------------------------------------------------------------
            if (lora_adapters_opt) |*la| {
                if (encoder_loaded) {
                    try refreshLoRAWeightsForForward(allocator, use_metal, &weight_store, &metal_weight_store, la);
                }
            }
            step_timing.lora_refresh_ns = elapsedNs(lora_refresh_start_ns);

            // ------------------------------------------------------------------
            // Encoder forward pass (or zero-fill fallback)
            // ------------------------------------------------------------------
            const debug_encoder_timing_step = getenvU32OrNull("ANTFLY_FUSED_CHUNKER_DEBUG_ENCODER_TIMING_STEP");
            const debug_encoder_timing = debug_encoder_timing_step != null and
                trainer.step_count + 1 == debug_encoder_timing_step.?;
            if (debug_encoder_timing) cb.resetDebugTimingStats();
            const encoder_start_ns = nowNs();
            var features_owned: ?[]f32 = null;
            defer if (features_owned) |f| allocator.free(f);

            var activations_opt: ?modern_bert.ActivationBuffer = null;
            defer if (activations_opt) |*ab| ab.deinit();

            if (encoder_loaded) {
                // Convert i32 token IDs → i64 for modern_bert.forward
                const ids_i64 = try allocator.alloc(i64, total_tokens);
                defer allocator.free(ids_i64);
                for (batch.input_ids[0..total_tokens], ids_i64) |id32, *id64| id64.* = @intCast(id32);

                const mask_i64 = try allocator.alloc(i64, total_tokens);
                defer allocator.free(mask_i64);
                for (batch.attention_mask[0..total_tokens], mask_i64) |m32, *m64| m64.* = @intCast(m32);

                const lora_alpha_for_config: f32 = if (lora_adapters_opt) |la| la.config.alpha else 0.0;
                const neftune_seed = if (effective_encoder_neftune_alpha > 0.0) blk: {
                    const seed = opts.seed ^ global_neft_step;
                    global_neft_step += 1;
                    break :blk seed;
                } else @as(u64, 0);
                const bert_config = modern_bert.Config{
                    .hidden_size = opts.hidden_size,
                    .num_hidden_layers = opts.num_layers,
                    .intermediate_size = opts.intermediate_size,
                    .lora_rank = if (lora_adapters_opt != null) opts.lora_rank else 0,
                    .lora_alpha = lora_alpha_for_config,
                    .neftune_alpha = effective_encoder_neftune_alpha,
                    .neftune_seed = neftune_seed,
                };

                if (lora_training_active) {
                    var act_buf = modern_bert.ActivationBuffer.init(allocator);
                    const debug_captures_requested = !opts.debug_encoder_layer_inputs_only and
                        ((opts.debug_first_boundary_step and epoch == 0 and step == 0) or
                            (opts.debug_boundary_step > 0 and trainer.step_count + 1 == opts.debug_boundary_step));
                    if (debug_captures_requested) {
                        act_buf.capture_attention_internal_layer = opts.debug_encoder_probe_layer;
                        act_buf.capture_projection_decomposition_layer = opts.debug_encoder_probe_layer;
                        act_buf.capture_layer_state_layer = opts.debug_encoder_probe_layer;
                    }

                    // P1 opt-in fast path: run the encoder layers through
                    // compiled MPSGraph segment sessions. Debug probe steps
                    // (which need extra captures) always use the eager path.
                    var compiled_features: ?[]f32 = null;
                    if (compiled_forward_sessions != null and !compiled_forward_failed and
                        lora_adapters_opt != null and !debug_captures_requested)
                    {
                        const la_forward = &lora_adapters_opt.?;
                        const forward_layers_per_segment = if (opts.lisa_sample_layers == 0)
                            opts.layers_per_segment
                        else
                            1;
                        const eager_defaults = modern_bert.Config{};
                        const forward_num_heads: u32 = 12;
                        if (opts.hidden_size % forward_num_heads != 0) return error.InvalidHeadDim;
                        const forward_graph_config = modern_bert_graph.Config{
                            .vocab_size = 50368,
                            .hidden_size = opts.hidden_size,
                            .num_hidden_layers = opts.num_layers,
                            .num_attention_heads = forward_num_heads,
                            .head_dim = opts.hidden_size / forward_num_heads,
                            .intermediate_size = opts.intermediate_size,
                            .max_position_embeddings = eager_defaults.max_position_embeddings,
                            .layer_norm_eps = eager_defaults.layer_norm_eps,
                            .rope_theta = eager_defaults.global_rope_theta,
                            .local_rope_theta = eager_defaults.local_rope_theta,
                            .local_attention_window = eager_defaults.local_attention_window,
                            .global_attn_every_n_layers = eager_defaults.global_attn_every_n_layers,
                        };
                        compiled_features = runCompiledSegmentedEncoderForward(
                            allocator,
                            &cb,
                            bert_config,
                            forward_graph_config,
                            compiled_forward_sessions.?,
                            forward_layers_per_segment,
                            opts.lora_rank,
                            la_forward.config.alpha,
                            ids_i64,
                            mask_i64,
                            actual_batch,
                            max_seq,
                            if (active_lora_layers) |layers| layers else null,
                            opts.encoder_vjp != .direct,
                            usesFullEncoderVJP(opts.encoder_vjp),
                            la_forward.layers,
                            &act_buf,
                        ) catch |err| blk: {
                            compiled_forward_failed = true;
                            print(
                                "warning: compiled segment forward failed with {s}; falling back to eager capture forward\n",
                                .{@errorName(err)},
                            );
                            break :blk null;
                        };
                        if (compiled_features == null) {
                            // A partial compiled attempt may have populated
                            // captures; restart with a fresh buffer. Debug
                            // capture requests never take the compiled path,
                            // so no capture settings need restoring.
                            act_buf.deinit();
                            act_buf = modern_bert.ActivationBuffer.init(allocator);
                        }
                    }

                    if (compiled_features != null and compiled_forward_check and trainer.step_count == 0) {
                        var check_owned_features = compiled_features;
                        errdefer if (check_owned_features) |f| allocator.free(f);
                        errdefer act_buf.deinit();
                        var eager_buf = modern_bert.ActivationBuffer.init(allocator);
                        defer eager_buf.deinit();
                        const eager_fwd = try modern_bert.forwardCapturingActivations(
                            &cb,
                            allocator,
                            bert_config,
                            ids_i64,
                            mask_i64,
                            actual_batch,
                            max_seq,
                            if (active_lora_layers) |layers| layers else null,
                            opts.encoder_vjp != .direct,
                            usesFullEncoderVJP(opts.encoder_vjp),
                            &eager_buf,
                        );
                        defer allocator.free(eager_fwd);
                        try verifyCompiledForwardParity(
                            compiled_features.?,
                            eager_fwd,
                            &act_buf,
                            &eager_buf,
                            mask_i64,
                            @intCast(opts.hidden_size),
                        );
                        check_owned_features = null;
                        print("compiled_forward_check step=1 status=ok\n", .{});
                    }

                    if (compiled_features) |fwd| {
                        features_owned = fwd;
                        activations_opt = act_buf;
                    } else {
                        const fwd = try modern_bert.forwardCapturingActivations(
                            &cb,
                            allocator,
                            bert_config,
                            ids_i64,
                            mask_i64,
                            actual_batch,
                            max_seq,
                            if (active_lora_layers) |layers| layers else null,
                            opts.encoder_vjp != .direct,
                            usesFullEncoderVJP(opts.encoder_vjp),
                            &act_buf,
                        );
                        features_owned = fwd;
                        activations_opt = act_buf;
                    }
                } else {
                    features_owned = try modern_bert.forward(
                        &cb,
                        allocator,
                        bert_config,
                        ids_i64,
                        mask_i64,
                        actual_batch,
                        max_seq,
                    );
                }
            }

            // Fallback: zero-fill when encoder is not loaded.
            var features_fallback: ?[]f32 = null;
            defer if (features_fallback) |f| allocator.free(f);

            const features: []const f32 = if (features_owned) |f| f else blk: {
                const zeros = try allocator.alloc(f32, total_tokens * @as(usize, opts.hidden_size));
                @memset(zeros, 0);
                features_fallback = zeros;
                break :blk zeros;
            };
            step_timing.encoder_ns = elapsedNs(encoder_start_ns);
            if (debug_encoder_timing) {
                print(
                    "encoder_timing_probe step={d} encoder_ms={d:.3} backend={s} decoder_runtime_ready={}\n",
                    .{
                        trainer.step_count + 1,
                        nsToMs(step_timing.encoder_ns),
                        if (use_metal) "metal" else "native",
                        cb.decoderRuntimeReady(),
                    },
                );
                debug_timing.printBackendTimingDetails(
                    cb.kind(),
                    cb.debugTimingSnapshot(),
                    cb.decoderRuntimeReady(),
                    false,
                );
            }

            // Attention mask: convert i32 -> f32
            const attn_mask_f32 = try allocator.alloc(f32, total_tokens);
            defer allocator.free(attn_mask_f32);
            for (batch.attention_mask[0..total_tokens], attn_mask_f32) |m, *out| {
                out.* = @floatFromInt(m);
            }

            // Boundary labels one-hot [total_tokens * 2]: build from batch.boundary_labels
            // batch.boundary_labels is [batch_size * max_seq_len] with 1.0 at boundary positions
            const boundary_labels_2 = try allocator.alloc(f32, total_tokens * 2);
            defer allocator.free(boundary_labels_2);
            for (0..total_tokens) |t| {
                const is_boundary = batch.boundary_labels[t] > 0.5;
                boundary_labels_2[t * 2 + 0] = if (is_boundary) 0.0 else 1.0; // class 0: non-boundary
                boundary_labels_2[t * 2 + 1] = if (is_boundary) 1.0 else 0.0; // class 1: boundary
            }

            const E: usize = @intCast(opts.hidden_size);

            const next_train_step = trainer.step_count + 1;
            const debug_first_boundary_step_now = opts.debug_first_boundary_step and epoch == 0 and step == 0;
            const debug_selected_boundary_step_now = opts.debug_boundary_step > 0 and next_train_step == opts.debug_boundary_step;
            var embedding_probe = EmbeddingProbe{};
            var embedding_lookup_values: ?[]const f32 = null;
            var embedding_table_row_probes: []EmbeddingRowProbe = &.{};
            defer if (embedding_table_row_probes.len > 0) allocator.free(embedding_table_row_probes);
            var embedding_lookup_row_probes: []EmbeddingRowProbe = &.{};
            defer if (embedding_lookup_row_probes.len > 0) allocator.free(embedding_lookup_row_probes);
            var encoder_activation_input_probes: []EncoderActivationInputProbe = &.{};
            defer if (encoder_activation_input_probes.len > 0) allocator.free(encoder_activation_input_probes);
            var encoder_projection_decomposition_probes: []EncoderProjectionDecompositionProbe = &.{};
            defer if (encoder_projection_decomposition_probes.len > 0) allocator.free(encoder_projection_decomposition_probes);
            var encoder_attention_internal_probes: []EncoderAttentionInternalProbe = &.{};
            defer if (encoder_attention_internal_probes.len > 0) allocator.free(encoder_attention_internal_probes);
            var encoder_attention_row_probes: []EncoderAttentionRowProbe = &.{};
            defer if (encoder_attention_row_probes.len > 0) allocator.free(encoder_attention_row_probes);
            var encoder_layer_input_probes: []EncoderLayerInputProbe = &.{};
            defer if (encoder_layer_input_probes.len > 0) allocator.free(encoder_layer_input_probes);
            var encoder_layer_state_probes: []EncoderLayerStateProbe = &.{};
            defer if (encoder_layer_state_probes.len > 0) allocator.free(encoder_layer_state_probes);
            var encoder_replay_input_probe: ?EncoderReplayInputProbe = null;
            var encoder_replay_upstream_probe: ?EncoderReplayUpstreamProbe = null;
            var step_parity_upstream_grad_probe: ?StepParityUpstreamGradProbe = null;
            defer if (step_parity_upstream_grad_probe) |*probe| probe.deinit(allocator);
            var step_parity_boundary: ?StepParityBoundaryContext = null;
            var step_parity_contrastive: ?StepParityContrastiveContext = null;
            var step_parity_segment_vjp: ?StepParitySegmentVJPProbe = null;
            defer if (step_parity_segment_vjp) |*probe| probe.deinit(allocator);
            var step_parity_layer_backward_decomp: ?StepParityLayerBackwardDecompProbe = null;
            defer if (step_parity_layer_backward_decomp) |*probe| probe.deinit(allocator);
            var step_parity_softmax_vjp: ?StepParitySoftmaxVJPProbe = null;
            defer if (step_parity_softmax_vjp) |*probe| probe.deinit(allocator);
            var step_parity_qkv_split_vjp: ?StepParityQKVSplitVJPProbe = null;
            defer if (step_parity_qkv_split_vjp) |*probe| probe.deinit(allocator);
            var exit_after_step_parity_contrastive = false;
            if (epoch == 0 and step == 0 and !debug_first_boundary_step_now) {
                const first_batch_stats = computeBoundaryBatchDebugStats(
                    features,
                    boundary_labels_2,
                    attn_mask_f32,
                    total_tokens,
                    E,
                );
                const gold_rate = if (first_batch_stats.valid_tokens > 0)
                    @as(f32, @floatFromInt(first_batch_stats.gold_positives)) / @as(f32, @floatFromInt(first_batch_stats.valid_tokens))
                else
                    0.0;
                print(
                    "first_batch_supervision actual_batch={d} total_tokens={d} valid={d} gold={d} gold_rate={d:.6} pos_weight={d} boundary_dropout={d} encoder_loaded={} use_metal={}\n",
                    .{
                        actual_batch,
                        total_tokens,
                        first_batch_stats.valid_tokens,
                        first_batch_stats.gold_positives,
                        gold_rate,
                        opts.boundary_pos_weight,
                        config.boundary_dropout,
                        encoder_loaded,
                        use_metal,
                    },
                );
            }
            if (debug_first_boundary_step_now or debug_selected_boundary_step_now) {
                const debug_prefix = if (debug_first_boundary_step_now) "debug_first_boundary_step" else "debug_boundary_step";
                const debug_batch_stats = computeBoundaryBatchDebugStats(
                    features,
                    boundary_labels_2,
                    attn_mask_f32,
                    total_tokens,
                    E,
                );
                var debug = try trainer.debugBoundaryStep(
                    allocator,
                    features,
                    boundary_labels_2,
                    attn_mask_f32,
                    batch.input_ids[0..total_tokens],
                    batch.boundary_candidate_mask[0..total_tokens],
                    total_tokens,
                    true,
                );
                if (activations_opt) |*act_buf| {
                    if (act_buf.embedding_lookup) |embedding_lookup| {
                        if (embedding_lookup.len == total_tokens * E) {
                            embedding_lookup_values = embedding_lookup;
                            embedding_probe.token_lookup =
                                fused_chunker_train.computeBoundaryProbeTensorStats(embedding_lookup);
                        }
                    }
                    if (act_buf.embedding_output) |embedding_output| {
                        if (embedding_output.len == total_tokens * E) {
                            embedding_probe.layer_norm_output =
                                fused_chunker_train.computeBoundaryProbeTensorStats(embedding_output);
                        }
                    }
                    if (act_buf.final_norm_input) |final_norm_input| {
                        if (final_norm_input.len == total_tokens * E) {
                            debug.boundary_forward_probe.final_norm_input =
                                fused_chunker_train.computeBoundaryProbeTensorStats(final_norm_input);
                        }
                    }
                    const has_final_norm_input_probe = if (act_buf.final_norm_input) |final_norm_input|
                        final_norm_input.len == total_tokens * E
                    else
                        false;
                    const final_probe_count: usize = if (has_final_norm_input_probe) 1 else 0;
                    encoder_layer_input_probes = try allocator.alloc(EncoderLayerInputProbe, act_buf.layer_inputs.items.len + final_probe_count);
                    var probe_idx: usize = 0;
                    for (act_buf.layer_inputs.items) |layer_input| {
                        encoder_layer_input_probes[probe_idx] = .{
                            .layer_idx = layer_input.layer_idx,
                            .stats = fused_chunker_train.computeBoundaryProbeTensorStats(layer_input.input),
                        };
                        probe_idx += 1;
                    }
                    if (has_final_norm_input_probe) {
                        const final_norm_input = act_buf.final_norm_input.?;
                        encoder_layer_input_probes[probe_idx] = .{
                            .layer_idx = opts.num_layers,
                            .stats = fused_chunker_train.computeBoundaryProbeTensorStats(final_norm_input),
                        };
                    }
                    if (opts.debug_encoder_replay_input_path) |replay_path| {
                        const layer_input = act_buf.findLayerInput(opts.debug_encoder_probe_layer) orelse return error.MissingEncoderReplayInputCapture;
                        if (layer_input.input.len != total_tokens * E) return error.InvalidEncoderReplayInputShape;
                        try writeF32LittleEndianFile(allocator, replay_path, layer_input.input);
                        encoder_replay_input_probe = .{
                            .path = replay_path,
                            .layer_idx = layer_input.layer_idx,
                            .batch_size = actual_batch,
                            .seq_len = max_seq,
                            .hidden_size = E,
                            .elems = layer_input.input.len,
                            .stats = fused_chunker_train.computeBoundaryProbeTensorStats(layer_input.input),
                        };
                    }
                    var layer_state_probe_count: usize = 0;
                    for (act_buf.layer_states.items) |cap| {
                        if (cap.layer_idx == opts.debug_encoder_probe_layer) layer_state_probe_count += 1;
                    }
                    if (layer_state_probe_count > 0) {
                        encoder_layer_state_probes = try allocator.alloc(EncoderLayerStateProbe, layer_state_probe_count);
                        var layer_state_probe_idx: usize = 0;
                        for (act_buf.layer_states.items) |cap| {
                            if (cap.layer_idx != opts.debug_encoder_probe_layer) continue;
                            encoder_layer_state_probes[layer_state_probe_idx] = .{
                                .layer_idx = cap.layer_idx,
                                .name = cap.name,
                                .stats = fused_chunker_train.computeBoundaryProbeTensorStats(cap.values),
                            };
                            layer_state_probe_idx += 1;
                        }
                    }
                    var activation_probe_count: usize = 0;
                    for (act_buf.items.items) |cap| {
                        if (cap.layer_idx == opts.debug_encoder_probe_layer) activation_probe_count += 1;
                    }
                    if (activation_probe_count > 0) {
                        encoder_activation_input_probes = try allocator.alloc(EncoderActivationInputProbe, activation_probe_count);
                        var activation_probe_idx: usize = 0;
                        for (act_buf.items.items) |cap| {
                            if (cap.layer_idx != opts.debug_encoder_probe_layer) continue;
                            encoder_activation_input_probes[activation_probe_idx] = .{
                                .layer_idx = cap.layer_idx,
                                .module_name = cap.module_name,
                                .stats = fused_chunker_train.computeBoundaryProbeTensorStats(cap.input),
                            };
                            activation_probe_idx += 1;
                        }
                    }
                    var projection_probe_count: usize = 0;
                    for (act_buf.projection_decompositions.items) |cap| {
                        if (cap.layer_idx == opts.debug_encoder_probe_layer) projection_probe_count += 1;
                    }
                    if (projection_probe_count > 0) {
                        encoder_projection_decomposition_probes = try allocator.alloc(EncoderProjectionDecompositionProbe, projection_probe_count);
                        var projection_probe_idx: usize = 0;
                        for (act_buf.projection_decompositions.items) |cap| {
                            if (cap.layer_idx != opts.debug_encoder_probe_layer) continue;
                            encoder_projection_decomposition_probes[projection_probe_idx] = .{
                                .layer_idx = cap.layer_idx,
                                .name = cap.name,
                                .input = fusedProbeStatsFromModernBert(cap.input),
                                .base = fusedProbeStatsFromModernBert(cap.base),
                                .lora_a = fusedProbeStatsFromModernBert(cap.lora_a),
                                .lora_b = fusedProbeStatsFromModernBert(cap.lora_b),
                                .delta = fusedProbeStatsFromModernBert(cap.delta),
                                .output = fusedProbeStatsFromModernBert(cap.output),
                                .weight = fusedProbeStatsFromModernBert(cap.weight),
                                .bias = fusedProbeStatsFromModernBert(cap.bias),
                                .lora_a_weight = fusedProbeStatsFromModernBert(cap.lora_a_weight),
                                .lora_b_weight = fusedProbeStatsFromModernBert(cap.lora_b_weight),
                                .base_reference_error = fusedProbeStatsFromModernBert(cap.base_reference_error),
                                .lora_a_reference_error = fusedProbeStatsFromModernBert(cap.lora_a_reference_error),
                                .lora_b_reference_error = fusedProbeStatsFromModernBert(cap.lora_b_reference_error),
                                .delta_reference_error = fusedProbeStatsFromModernBert(cap.delta_reference_error),
                                .output_reference_error = fusedProbeStatsFromModernBert(cap.output_reference_error),
                                .scale = cap.scale,
                                .rank = cap.rank,
                                .rows = cap.rows,
                                .in_dim = cap.in_dim,
                                .out_dim = cap.out_dim,
                                .has_bias = cap.has_bias,
                            };
                            projection_probe_idx += 1;
                        }
                    }
                    var attention_internal_probe_count: usize = 0;
                    for (act_buf.attention_internals.items) |cap| {
                        if (cap.layer_idx == opts.debug_encoder_probe_layer) attention_internal_probe_count += 1;
                    }
                    var sdpa_reference_stats: ?Layer0SdpaReferenceProbeStats = null;
                    var attention_core_tensors: ?StepParityAttentionCoreTensors = null;
                    defer if (attention_core_tensors) |*tensors| tensors.deinit(allocator);
                    const attention_defaults = modern_bert.Config{};
                    const is_local_attention_probe = attention_defaults.global_attn_every_n_layers > 0 and
                        opts.debug_encoder_probe_layer % attention_defaults.global_attn_every_n_layers != 0;
                    const attention_output_probe = findLayerActivationInput(act_buf, opts.debug_encoder_probe_layer, "out_proj");
                    const sdpa_num_heads: usize = 12;
                    if (E % sdpa_num_heads == 0) {
                        if (findLayerAttentionInternal(act_buf, opts.debug_encoder_probe_layer, "q_rope")) |q_rope| {
                            if (findLayerAttentionInternal(act_buf, opts.debug_encoder_probe_layer, "k_rope")) |k_rope| {
                                attention_core_tensors = try computeStepParityAttentionCoreTensors(
                                    allocator,
                                    q_rope,
                                    k_rope,
                                    attn_mask_f32,
                                    actual_batch,
                                    max_seq,
                                    sdpa_num_heads,
                                    E / sdpa_num_heads,
                                    is_local_attention_probe,
                                    attention_defaults.local_attention_window,
                                );
                            }
                        }
                        sdpa_reference_stats = try computeLayerSdpaReferenceProbeStats(
                            allocator,
                            act_buf,
                            opts.debug_encoder_probe_layer,
                            batch.attention_mask[0..total_tokens],
                            actual_batch,
                            max_seq,
                            sdpa_num_heads,
                            E / sdpa_num_heads,
                            is_local_attention_probe,
                            attention_defaults.local_attention_window,
                        );
                        encoder_attention_row_probes = try computeLayerAttentionRowProbes(
                            allocator,
                            act_buf,
                            opts.debug_encoder_probe_layer,
                            batch.attention_mask[0..total_tokens],
                            actual_batch,
                            max_seq,
                            sdpa_num_heads,
                            E / sdpa_num_heads,
                            is_local_attention_probe,
                            attention_defaults.local_attention_window,
                        );
                    }
                    const sdpa_reference_probe_count: usize = if (sdpa_reference_stats != null) 6 else 0;
                    const attention_core_probe_count: usize = if (attention_core_tensors != null) 3 else 0;
                    const attention_output_probe_count: usize = if (attention_output_probe != null) 1 else 0;
                    if (attention_internal_probe_count + attention_core_probe_count + attention_output_probe_count + sdpa_reference_probe_count > 0) {
                        encoder_attention_internal_probes = try allocator.alloc(EncoderAttentionInternalProbe, attention_internal_probe_count + attention_core_probe_count + attention_output_probe_count + sdpa_reference_probe_count);
                        var attention_internal_probe_idx: usize = 0;
                        for (act_buf.attention_internals.items) |cap| {
                            if (cap.layer_idx != opts.debug_encoder_probe_layer) continue;
                            encoder_attention_internal_probes[attention_internal_probe_idx] = .{
                                .layer_idx = cap.layer_idx,
                                .name = cap.name,
                                .stats = fused_chunker_train.computeBoundaryProbeTensorStats(cap.values),
                            };
                            attention_internal_probe_idx += 1;
                        }
                        if (attention_core_tensors) |core| {
                            encoder_attention_internal_probes[attention_internal_probe_idx] = .{
                                .layer_idx = opts.debug_encoder_probe_layer,
                                .name = "attn_scores_raw",
                                .stats = fused_chunker_train.computeBoundaryProbeTensorStats(core.scores_raw),
                            };
                            attention_internal_probe_idx += 1;
                            encoder_attention_internal_probes[attention_internal_probe_idx] = .{
                                .layer_idx = opts.debug_encoder_probe_layer,
                                .name = "attn_scores_masked",
                                .stats = fused_chunker_train.computeBoundaryProbeTensorStats(core.scores_masked),
                            };
                            attention_internal_probe_idx += 1;
                            encoder_attention_internal_probes[attention_internal_probe_idx] = .{
                                .layer_idx = opts.debug_encoder_probe_layer,
                                .name = "attn_probs",
                                .stats = fused_chunker_train.computeBoundaryProbeTensorStats(core.probs),
                            };
                            attention_internal_probe_idx += 1;
                        }
                        if (attention_output_probe) |attn_output| {
                            encoder_attention_internal_probes[attention_internal_probe_idx] = .{
                                .layer_idx = opts.debug_encoder_probe_layer,
                                .name = "attn_output",
                                .stats = fused_chunker_train.computeBoundaryProbeTensorStats(attn_output),
                            };
                            attention_internal_probe_idx += 1;
                        }
                        if (sdpa_reference_stats) |stats| {
                            encoder_attention_internal_probes[attention_internal_probe_idx] = .{
                                .layer_idx = opts.debug_encoder_probe_layer,
                                .name = "attn_context_ref",
                                .stats = stats.token_ref,
                            };
                            attention_internal_probe_idx += 1;
                            encoder_attention_internal_probes[attention_internal_probe_idx] = .{
                                .layer_idx = opts.debug_encoder_probe_layer,
                                .name = "attn_context_delta",
                                .stats = stats.token_delta,
                            };
                            attention_internal_probe_idx += 1;
                            encoder_attention_internal_probes[attention_internal_probe_idx] = .{
                                .layer_idx = opts.debug_encoder_probe_layer,
                                .name = "attn_token_ref",
                                .stats = stats.token_ref,
                            };
                            attention_internal_probe_idx += 1;
                            encoder_attention_internal_probes[attention_internal_probe_idx] = .{
                                .layer_idx = opts.debug_encoder_probe_layer,
                                .name = "attn_token_delta",
                                .stats = stats.token_delta,
                            };
                            attention_internal_probe_idx += 1;
                            encoder_attention_internal_probes[attention_internal_probe_idx] = .{
                                .layer_idx = opts.debug_encoder_probe_layer,
                                .name = "attn_kernel_ref",
                                .stats = stats.kernel_ref,
                            };
                            attention_internal_probe_idx += 1;
                            encoder_attention_internal_probes[attention_internal_probe_idx] = .{
                                .layer_idx = opts.debug_encoder_probe_layer,
                                .name = "attn_kernel_delta",
                                .stats = stats.kernel_delta,
                            };
                        }
                    }
                }
                const gold_rate = if (debug_batch_stats.valid_tokens > 0)
                    @as(f32, @floatFromInt(debug_batch_stats.gold_positives)) / @as(f32, @floatFromInt(debug_batch_stats.valid_tokens))
                else
                    0.0;
                print(
                    "{s} epoch={d} local_step={d} global_step={d} trainer_step_count={d} actual_batch={d} total_tokens={d} valid={d} gold={d} gold_rate={d:.6} pos_weight={d} boundary_dropout={d} encoder_loaded={} use_metal={}\n",
                    .{
                        debug_prefix,
                        epoch + 1,
                        step + 1,
                        next_train_step,
                        trainer.step_count,
                        actual_batch,
                        total_tokens,
                        debug_batch_stats.valid_tokens,
                        debug_batch_stats.gold_positives,
                        gold_rate,
                        opts.boundary_pos_weight,
                        config.boundary_dropout,
                        encoder_loaded,
                        use_metal,
                    },
                );
                const indices_label = if (debug_first_boundary_step_now) "debug_first_boundary_step batch_indices" else "debug_boundary_step batch_indices";
                printIndexSlice(indices_label, batch_indices);
                printPerSampleBoundaryDebug(debug_prefix, &batch, max_seq);
                print(
                    "{s} features mean={d:.6} rms={d:.6} max_abs={d:.6}\n",
                    .{ debug_prefix, debug_batch_stats.feature_mean, debug_batch_stats.feature_rms, debug_batch_stats.feature_max_abs },
                );
                print(
                    "{s} loss={d:.6} eval_f1={d:.4} eval_counts tp={d} fp={d} fn={d} eval_predicted_positives={d} eval_prob_gold_pos={d:.6} eval_prob_gold_neg={d:.6}\n",
                    .{
                        debug_prefix,
                        debug.boundary_loss,
                        debug.eval_f1,
                        debug.eval_tp,
                        debug.eval_fp,
                        debug.eval_fn,
                        debug.eval_predicted_positives,
                        debug.eval_mean_prob_gold_positive,
                        debug.eval_mean_prob_gold_negative,
                    },
                );
                print(
                    "{s} grad_norms w1={d:.6} b1={d:.6} w2={d:.6} b2={d:.6} features={d:.6} has_features_grad={}\n",
                    .{
                        debug_prefix,
                        debug.grad_norm_w1,
                        debug.grad_norm_b1,
                        debug.grad_norm_w2,
                        debug.grad_norm_b2,
                        debug.features_grad_norm,
                        debug.has_features_grad,
                    },
                );
                print(
                    "{s} grad_max_abs w1={d:.6} b1={d:.6} w2={d:.6} b2={d:.6} features={d:.6}\n",
                    .{
                        debug_prefix,
                        debug.grad_max_abs_w1,
                        debug.grad_max_abs_b1,
                        debug.grad_max_abs_w2,
                        debug.grad_max_abs_b2,
                        debug.features_grad_max_abs,
                    },
                );
                const final_norm_checkpoint = computeFinalNormCheckpointStats(use_metal, &weight_store, &metal_weight_store);
                const embedding_checkpoint = computeEmbeddingCheckpointStats(use_metal, &weight_store, &metal_weight_store);
                embedding_probe.word_embedding_weight = embedding_checkpoint.word_embedding_weight;
                embedding_probe.layer_norm_weight = embedding_checkpoint.layer_norm_weight;
                embedding_probe.layer_norm_bias = embedding_checkpoint.layer_norm_bias;
                embedding_table_row_probes = try allocEmbeddingTableRowProbes(
                    allocator,
                    batch.input_ids,
                    total_tokens,
                    E,
                    activeEmbeddingWeightValues(use_metal, &weight_store, &metal_weight_store),
                );
                embedding_lookup_row_probes = try allocEmbeddingLookupRowProbes(
                    allocator,
                    batch.input_ids,
                    total_tokens,
                    E,
                    embedding_lookup_values,
                );
                step_parity_boundary = .{
                    .batch_stats = debug_batch_stats,
                    .debug = debug,
                    .embedding_probe = embedding_probe,
                    .embedding_table_rows = embedding_table_row_probes,
                    .embedding_lookup_rows = embedding_lookup_row_probes,
                    .final_norm_weight = final_norm_checkpoint.weight,
                    .final_norm_bias = final_norm_checkpoint.bias,
                    .encoder_activation_inputs = encoder_activation_input_probes,
                    .encoder_projection_decompositions = encoder_projection_decomposition_probes,
                    .encoder_attention_internals = encoder_attention_internal_probes,
                    .encoder_attention_rows = encoder_attention_row_probes,
                    .encoder_layer_inputs = encoder_layer_input_probes,
                    .encoder_layer_states = encoder_layer_state_probes,
                    .encoder_replay_input = encoder_replay_input_probe,
                };
                if ((debug_first_boundary_step_now and opts.debug_first_boundary_step_exit) or
                    (debug_selected_boundary_step_now and opts.debug_boundary_step_exit))
                {
                    if (opts.debug_step_json_path != null) {
                        exit_after_step_parity_contrastive = true;
                    } else {
                        print("{s} exiting before optimizer update\n", .{debug_prefix});
                        return;
                    }
                }
            }

            // Late chunk embeddings: mean-pool token features over chunk spans
            // and L2-normalize them to match the Go late-chunking head.
            const C: usize = max_chunks;
            const chunk_embeddings = try fused_chunker_data.meanPoolChunkEmbeddings(
                allocator,
                features,
                &batch,
                E,
            );
            defer allocator.free(chunk_embeddings);

            // Chunk mask: [B * max_chunks] — use batch.chunk_mask directly
            // (already sized batch_size * max_chunks)
            const chunk_mask = batch.chunk_mask;

            // Doc IDs: [B * max_chunks] — assign each sample its own doc id
            const doc_ids = try allocator.alloc(u32, actual_batch * C);
            defer allocator.free(doc_ids);
            for (0..actual_batch) |b_idx| {
                for (0..C) |c_idx| {
                    doc_ids[b_idx * C + c_idx] = @intCast(b_idx);
                }
            }

            if (opts.debug_boundary_head_overfit_steps > 0) {
                try runBoundaryHeadFrozenOverfitProbe(
                    allocator,
                    &trainer,
                    features,
                    boundary_labels_2,
                    attn_mask_f32,
                    chunk_embeddings,
                    chunk_mask,
                    doc_ids,
                    total_tokens,
                    actual_batch,
                    C,
                    E,
                    opts.debug_boundary_head_overfit_steps,
                    opts.debug_boundary_head_overfit_lr,
                );
                return;
            }

            var contrastive_embeddings: []const f32 = chunk_embeddings;
            var contrastive_mask: []const f32 = chunk_mask;
            var contrastive_doc_ids: []const u32 = doc_ids;
            var contrastive_B: usize = actual_batch;
            var contrastive_C: usize = C;
            var hard_neg_ids_i64: ?[]i64 = null;
            var hard_neg_mask_i64: ?[]i64 = null;
            var hard_neg_mask_i32: ?[]i32 = null;
            var hard_neg_features: ?[]f32 = null;
            var hard_neg_embeddings: ?[]f32 = null;
            var contrastive_embeddings_owned: ?[]f32 = null;
            var contrastive_mask_owned: ?[]f32 = null;
            var contrastive_doc_ids_owned: ?[]u32 = null;
            defer if (hard_neg_ids_i64) |buf| allocator.free(buf);
            defer if (hard_neg_mask_i64) |buf| allocator.free(buf);
            defer if (hard_neg_mask_i32) |buf| allocator.free(buf);
            defer if (hard_neg_features) |buf| allocator.free(buf);
            defer if (hard_neg_embeddings) |buf| allocator.free(buf);
            defer if (contrastive_embeddings_owned) |buf| allocator.free(buf);
            defer if (contrastive_mask_owned) |buf| allocator.free(buf);
            defer if (contrastive_doc_ids_owned) |buf| allocator.free(buf);

            const hard_neg_start_ns = nowNs();
            if (encoder_loaded and batch.hard_neg_ids != null and batch.hard_neg_mask != null) {
                const hn_ids = batch.hard_neg_ids.?;
                const hn_mask = batch.hard_neg_mask.?;
                const denom = actual_batch * max_seq;
                const neg_stride = if (denom > 0) hn_ids.len / denom else 0;
                var valid_neg_count: usize = 0;
                if (neg_stride > 0 and hn_mask.len >= hn_ids.len) {
                    for (0..actual_batch) |b_idx| {
                        for (0..neg_stride) |neg_idx| {
                            const src_base = (b_idx * neg_stride + neg_idx) * max_seq;
                            var any_token = false;
                            for (hn_mask[src_base .. src_base + max_seq]) |m| {
                                if (m != 0) {
                                    any_token = true;
                                    break;
                                }
                            }
                            if (any_token) valid_neg_count += 1;
                        }
                    }
                }

                if (valid_neg_count > 0) {
                    const neg_ids_i64 = try allocator.alloc(i64, valid_neg_count * max_seq);
                    const neg_mask_i64 = try allocator.alloc(i64, valid_neg_count * max_seq);
                    const neg_mask_i32 = try allocator.alloc(i32, valid_neg_count * max_seq);
                    hard_neg_ids_i64 = neg_ids_i64;
                    hard_neg_mask_i64 = neg_mask_i64;
                    hard_neg_mask_i32 = neg_mask_i32;

                    var out_neg: usize = 0;
                    for (0..actual_batch) |b_idx| {
                        for (0..neg_stride) |neg_idx| {
                            const src_base = (b_idx * neg_stride + neg_idx) * max_seq;
                            var any_token = false;
                            for (hn_mask[src_base .. src_base + max_seq]) |m| {
                                if (m != 0) {
                                    any_token = true;
                                    break;
                                }
                            }
                            if (!any_token) continue;

                            const dst_base = out_neg * max_seq;
                            for (0..max_seq) |t| {
                                neg_ids_i64[dst_base + t] = @intCast(hn_ids[src_base + t]);
                                neg_mask_i64[dst_base + t] = @intCast(hn_mask[src_base + t]);
                                neg_mask_i32[dst_base + t] = hn_mask[src_base + t];
                            }
                            out_neg += 1;
                        }
                    }

                    const lora_alpha_for_config: f32 = if (lora_adapters_opt) |la| la.config.alpha else 0.0;
                    const hard_neg_neftune_seed = if (effective_encoder_neftune_alpha > 0.0) blk: {
                        const seed = opts.seed ^ global_neft_step;
                        global_neft_step += 1;
                        break :blk seed;
                    } else @as(u64, 0);
                    const hard_neg_config = modern_bert.Config{
                        .hidden_size = opts.hidden_size,
                        .num_hidden_layers = opts.num_layers,
                        .intermediate_size = opts.intermediate_size,
                        .lora_rank = if (lora_adapters_opt != null) opts.lora_rank else 0,
                        .lora_alpha = lora_alpha_for_config,
                        .neftune_alpha = effective_encoder_neftune_alpha,
                        .neftune_seed = hard_neg_neftune_seed,
                    };
                    hard_neg_features = modern_bert.forward(
                        &cb,
                        allocator,
                        hard_neg_config,
                        neg_ids_i64,
                        neg_mask_i64,
                        valid_neg_count,
                        max_seq,
                    ) catch |err| hblk: {
                        print("warning: hard-negative encoder forward failed at step {d}: {}\n", .{ step, err });
                        break :hblk null;
                    };

                    if (hard_neg_features) |neg_features| {
                        hard_neg_embeddings = try fused_chunker_data.meanPoolSequenceEmbeddings(
                            allocator,
                            neg_features,
                            neg_mask_i32,
                            valid_neg_count,
                            max_seq,
                            E,
                        );

                        const current_vectors = actual_batch * C;
                        const total_vectors = current_vectors + valid_neg_count;
                        const combined_embeddings = try allocator.alloc(f32, total_vectors * E);
                        const combined_mask = try allocator.alloc(f32, total_vectors);
                        const combined_doc_ids = try allocator.alloc(u32, total_vectors);
                        contrastive_embeddings_owned = combined_embeddings;
                        contrastive_mask_owned = combined_mask;
                        contrastive_doc_ids_owned = combined_doc_ids;

                        @memcpy(combined_embeddings[0 .. current_vectors * E], chunk_embeddings);
                        @memcpy(combined_embeddings[current_vectors * E ..], hard_neg_embeddings.?);
                        @memcpy(combined_mask[0..current_vectors], chunk_mask[0..current_vectors]);
                        @memset(combined_mask[current_vectors..], 1.0);
                        @memcpy(combined_doc_ids[0..current_vectors], doc_ids[0..current_vectors]);
                        for (0..valid_neg_count) |neg_i| {
                            combined_doc_ids[current_vectors + neg_i] = @intCast(actual_batch + neg_i);
                        }

                        contrastive_embeddings = combined_embeddings;
                        contrastive_mask = combined_mask;
                        contrastive_doc_ids = combined_doc_ids;
                        contrastive_B = 1;
                        contrastive_C = total_vectors;
                    }
                }
            }
            step_timing.hard_neg_ns = elapsedNs(hard_neg_start_ns);

            if (opts.debug_step_json_path != null and step_parity_boundary != null) {
                step_parity_contrastive = try computeStepParityContrastiveContext(
                    allocator,
                    trainer.loss_config,
                    contrastive_embeddings,
                    contrastive_mask,
                    contrastive_doc_ids,
                    contrastive_B,
                    contrastive_C,
                    E,
                );
            }

            if (exit_after_step_parity_contrastive) {
                const json_path = opts.debug_step_json_path orelse return error.MissingDebugStepJson;
                const boundary_ctx = step_parity_boundary orelse return error.MissingDebugBoundaryStepForParityJson;
                try writeStepParityJson(
                    json_path,
                    "no_update",
                    &opts,
                    trainer.loss_config.lambda_embed,
                    epoch,
                    step,
                    next_train_step,
                    trainer.step_count,
                    batch_indices,
                    &batch,
                    total_tokens,
                    boundary_ctx,
                    step_parity_contrastive,
                    step_parity_upstream_grad_probe,
                    encoder_replay_upstream_probe,
                    step_parity_segment_vjp,
                    step_parity_layer_backward_decomp,
                    step_parity_softmax_vjp,
                    step_parity_qkv_split_vjp,
                    null,
                    null,
                );
                print("debug_boundary_step exiting before optimizer update after contrastive diagnostics\n", .{});
                return;
            }

            // ------------------------------------------------------------------
            // Training step + optional LoRA backprop
            // ------------------------------------------------------------------
            var summary: TrainStepSummary = undefined;
            var last_vjp_profile: ?segmented_encoder.SegmentVJPProfile = null;

            if (lora_adapters_opt) |*la| {
                if (activations_opt) |*act_buf| {
                    // Use the gradient-returning variant so we can backprop into LoRA.
                    const train_step_start_ns = nowNs();
                    var result_with_grad = try trainer.trainStepWithEncoderGrad(
                        allocator,
                        features,
                        boundary_labels_2,
                        attn_mask_f32,
                        batch.input_ids[0..total_tokens],
                        batch.boundary_candidate_mask[0..total_tokens],
                        contrastive_embeddings,
                        contrastive_mask,
                        contrastive_doc_ids,
                        total_tokens,
                        contrastive_B,
                        contrastive_C,
                        E,
                        batch.chunk_starts,
                        batch.chunk_ends,
                        actual_batch,
                        max_seq,
                        C,
                        opts.debug_step_json_path != null,
                    );
                    step_timing.train_ns = elapsedNs(train_step_start_ns);
                    defer result_with_grad.deinit(allocator);
                    summary = result_with_grad.summary;

                    if (!isFiniteTrainStepSummary(summary)) {
                        const repaired = sanitizeLoRAAdapterSet(la) + optimizers.sanitizeState(&lora_opt_state);
                        print(
                            "warning: nonfinite train summary at global_step={d}; skipping encoder/LoRA backward boundary={d} contrastive={d} total={d} repaired_lora_state={d}\n",
                            .{
                                summary.step,
                                summary.boundary_loss,
                                summary.contrastive_loss,
                                summary.total_loss,
                                repaired,
                            },
                        );
                    } else if (result_with_grad.features_grad) |d_features| {
                        const lora_update_start_ns = nowNs();
                        var final_norm_grad: ?[]f32 = null;
                        defer if (final_norm_grad) |buf| allocator.free(buf);
                        const lora_output_grad: []const f32 = if (act_buf.final_norm_input) |final_norm_input| blk_grad: {
                            const final_norm_weight = act_buf.final_norm_weight orelse break :blk_grad d_features;
                            if (act_buf.final_norm_total != total_tokens or
                                act_buf.final_norm_hidden != E or
                                final_norm_input.len != d_features.len or
                                final_norm_weight.len != E)
                            {
                                break :blk_grad d_features;
                            }
                            final_norm_grad = try segmented_encoder.backpropLayerNormInput(
                                allocator,
                                d_features,
                                final_norm_input,
                                final_norm_weight,
                                total_tokens,
                                E,
                                (modern_bert.Config{}).layer_norm_eps,
                            );
                            break :blk_grad final_norm_grad.?;
                        } else d_features;

                        if (opts.debug_step_json_path != null) {
                            var upstream_probe = StepParityUpstreamGradProbe{
                                .target_layer = opts.debug_encoder_probe_layer,
                                .boundary_features_grad = result_with_grad.boundary_features_grad_stats,
                                .contrastive_features_grad = result_with_grad.contrastive_features_grad_stats,
                                .combined_features_grad = result_with_grad.combined_features_grad_stats,
                                .lora_output_grad = fused_chunker_train.computeBoundaryProbeTensorStats(lora_output_grad),
                            };
                            if (act_buf.final_norm_input) |final_norm_input| {
                                upstream_probe.final_norm_input = fused_chunker_train.computeBoundaryProbeTensorStats(final_norm_input);
                            }
                            if (act_buf.final_norm_weight) |final_norm_weight| {
                                upstream_probe.final_norm_weight = fused_chunker_train.computeBoundaryProbeTensorStats(final_norm_weight);
                            }
                            step_parity_upstream_grad_probe = upstream_probe;
                        }

                        const active_layers = active_lora_layers orelse return error.MissingActiveLoRALayers;

                        var direct_skip_layer: ?u32 = null;
                        var direct_vjp_complete = false;
                        if (usesFullEncoderVJP(opts.encoder_vjp) and la.config.num_layers > 0) {
                            const sessions = full_vjp_sessions orelse return error.MissingFullVJPSessions;
                            if (sessions.len < @as(usize, @intCast(la.config.num_layers))) return error.MissingFullVJPSessions;
                            const total_hidden = total_tokens * E;
                            if (lora_output_grad.len != total_hidden) return error.InvalidSegmentVJPShape;
                            const num_heads: u32 = 12;
                            if (opts.hidden_size % num_heads != 0) return error.InvalidHeadDim;
                            const eager_defaults = modern_bert.Config{};
                            var upstream_grad: []const f32 = lora_output_grad;
                            var upstream_owned: ?[]f32 = null;
                            defer if (upstream_owned) |buf| allocator.free(buf);
                            // P2 opt-in: share the two attention-bias variants
                            // across all segment VJP executions in this step.
                            var vjp_shared_bias: ?segmented_encoder.SharedSegmentAttnBias = null;
                            defer if (vjp_shared_bias) |*bias| bias.deinit();
                            if (segmented_encoder.vjpFeedCacheEnabled()) {
                                vjp_shared_bias = segmented_encoder.SharedSegmentAttnBias.init(
                                    allocator,
                                    @intCast(actual_batch),
                                    @intCast(max_seq),
                                    num_heads,
                                    eager_defaults.local_attention_window,
                                    attn_mask_f32,
                                );
                            }
                            var aggregate_profile = segmented_encoder.SegmentVJPProfile{};
                            const exact_adapter_grads = usesExactFullAdapterGrads(opts.encoder_vjp);
                            const layers_per_segment = if (opts.lisa_sample_layers == 0)
                                opts.layers_per_segment
                            else
                                1;
                            const top_k_boundary: ?u32 = if (opts.lora_train_top_k > 0) blk: {
                                const top_k = @min(opts.lora_train_top_k, la.config.num_layers);
                                break :blk la.config.num_layers - top_k;
                            } else null;
                            const vjp_stop_layer = top_k_boundary orelse 0;

                            if (opts.debug_step_json_path != null and opts.debug_encoder_probe_layer < la.config.num_layers) {
                                if (step_parity_upstream_grad_probe) |*probe| {
                                    const ladder_count: usize = @intCast(la.config.num_layers - opts.debug_encoder_probe_layer);
                                    if (probe.upper_encoder_ladder.len == 0 and ladder_count > 0) {
                                        var ladder = try allocator.alloc(StepParityNamedProbeTensorStats, ladder_count);
                                        var filled: usize = 0;
                                        errdefer {
                                            for (ladder[0..filled]) |*entry| entry.deinit(allocator);
                                            allocator.free(ladder);
                                        }
                                        for (0..ladder_count) |i| {
                                            const layer_idx = la.config.num_layers - 1 - @as(u32, @intCast(i));
                                            ladder[i] = .{
                                                .name = try std.fmt.allocPrint(allocator, "after_layer_{d}", .{layer_idx}),
                                            };
                                            filled += 1;
                                        }
                                        probe.upper_encoder_ladder = ladder;
                                    }
                                }
                            }

                            var layer_cursor: u32 = la.config.num_layers;
                            while (layer_cursor > vjp_stop_layer) {
                                const segment_end = layer_cursor;
                                var segment_start = if (segment_end > layers_per_segment)
                                    segment_end - layers_per_segment
                                else
                                    0;
                                if (top_k_boundary) |boundary| {
                                    if (segment_start < boundary and segment_end > boundary) {
                                        segment_start = boundary;
                                    }
                                }
                                const segment_input = act_buf.findLayerInput(segment_start) orelse return error.MissingSegmentLayerInputCapture;
                                if (segment_input.input.len != total_hidden or upstream_grad.len != total_hidden) {
                                    return error.InvalidSegmentVJPShape;
                                }
                                if (step_parity_upstream_grad_probe) |*probe| {
                                    if (probe.upper_encoder_ladder.len > 0 and
                                        segment_end > opts.debug_encoder_probe_layer and
                                        segment_end <= la.config.num_layers)
                                    {
                                        const ladder_index: usize = @intCast(la.config.num_layers - segment_end);
                                        if (ladder_index < probe.upper_encoder_ladder.len) {
                                            probe.upper_encoder_ladder[ladder_index].stats =
                                                fused_chunker_train.computeBoundaryProbeTensorStats(upstream_grad);
                                        }
                                    }
                                }
                                var segment_has_active_lora = false;
                                var active_layer_idx = segment_start;
                                while (active_layer_idx < segment_end) : (active_layer_idx += 1) {
                                    if (fused_chunker_train.isLoRALayerActive(active_layers, active_layer_idx)) {
                                        segment_has_active_lora = true;
                                        break;
                                    }
                                }
                                const include_hidden_grad = segment_start > vjp_stop_layer;
                                const include_adapter_grads = exact_adapter_grads and segment_has_active_lora;
                                const needs_segment_vjp = include_hidden_grad or include_adapter_grads;

                                if (!exact_adapter_grads and segment_has_active_lora) {
                                    try accumulateDirectLoRAGradsForLayerRange(
                                        cb,
                                        allocator,
                                        act_buf,
                                        active_layers,
                                        segment_start,
                                        segment_end,
                                        upstream_grad,
                                        la,
                                    );
                                }

                                if (needs_segment_vjp) {
                                    const graph_config = modern_bert_graph.Config{
                                        .vocab_size = 50368,
                                        .hidden_size = opts.hidden_size,
                                        .num_hidden_layers = la.config.num_layers,
                                        .num_attention_heads = num_heads,
                                        .head_dim = opts.hidden_size / num_heads,
                                        .intermediate_size = opts.intermediate_size,
                                        .max_position_embeddings = eager_defaults.max_position_embeddings,
                                        .layer_norm_eps = eager_defaults.layer_norm_eps,
                                        .rope_theta = eager_defaults.global_rope_theta,
                                        .local_rope_theta = eager_defaults.local_rope_theta,
                                        .local_attention_window = eager_defaults.local_attention_window,
                                        .global_attn_every_n_layers = eager_defaults.global_attn_every_n_layers,
                                    };
                                    const session_slot = &sessions[@intCast(segment_start)];
                                    var rebuild_vjp_session = session_slot.* == null;
                                    if (session_slot.*) |*session| {
                                        if (session.batch != actual_batch or
                                            session.seq_len != max_seq or
                                            session.start_layer != segment_start or
                                            session.end_layer != segment_end or
                                            session.include_hidden_grad != include_hidden_grad or
                                            session.include_adapter_grads != include_adapter_grads or
                                            session.config.hidden_size != graph_config.hidden_size or
                                            session.config.intermediate_size != graph_config.intermediate_size)
                                        {
                                            session.deinit();
                                            session_slot.* = null;
                                            rebuild_vjp_session = true;
                                        }
                                    }
                                    if (rebuild_vjp_session) {
                                        session_slot.* = try segmented_encoder.ModernBertSegmentVJPSession.initWithGradientOptions(
                                            allocator,
                                            graph_config,
                                            @intCast(actual_batch),
                                            @intCast(max_seq),
                                            segment_start,
                                            segment_end,
                                            opts.lora_rank,
                                            la.config.alpha,
                                            include_hidden_grad,
                                            include_adapter_grads,
                                        );
                                    }
                                    const session = &(session_slot.* orelse return error.MissingFullVJPSessions);
                                    const run_segment_vjp_parity = segment_vjp_parity_pending and
                                        (segment_vjp_parity_layer == null or
                                            (segment_vjp_parity_layer.? >= segment_start and segment_vjp_parity_layer.? < segment_end));
                                    if (run_segment_vjp_parity) {
                                        segment_vjp_parity_pending = false;
                                        try runSegmentVJPParityDiagnostic(
                                            allocator,
                                            &cb,
                                            graph_config,
                                            actual_batch,
                                            max_seq,
                                            segment_start,
                                            segment_end,
                                            opts.lora_rank,
                                            la.config.alpha,
                                            include_hidden_grad,
                                            include_adapter_grads,
                                            segment_input.input,
                                            upstream_grad,
                                            attn_mask_f32,
                                            la.layers,
                                        );
                                    }
                                    var vjp_result = try session.executeWithSharedBias(
                                        &cb,
                                        segment_input.input,
                                        upstream_grad,
                                        attn_mask_f32,
                                        la.layers,
                                        include_adapter_grads,
                                        if (vjp_shared_bias) |*bias| bias else null,
                                    );
                                    aggregate_profile.add(vjp_result.profile);
                                    if (opts.debug_step_json_path != null and
                                        step_parity_segment_vjp == null and
                                        opts.debug_encoder_probe_layer >= segment_start and
                                        opts.debug_encoder_probe_layer < segment_end)
                                    {
                                        if (step_parity_upstream_grad_probe) |*probe| {
                                            probe.target_segment_upstream = fused_chunker_train.computeBoundaryProbeTensorStats(upstream_grad);
                                        }
                                        var captured_segment_probe = try captureStepParitySegmentVJPProbe(
                                            allocator,
                                            opts.debug_encoder_probe_layer,
                                            segment_start,
                                            segment_end,
                                            include_hidden_grad,
                                            include_adapter_grads,
                                            upstream_grad,
                                            &vjp_result,
                                            la.layers,
                                        );
                                        var captured_segment_probe_owned = true;
                                        errdefer if (captured_segment_probe_owned) captured_segment_probe.deinit(allocator);

                                        if (opts.debug_encoder_replay_upstream_path) |replay_path| {
                                            if (upstream_grad.len != total_hidden) return error.InvalidEncoderReplayUpstreamShape;
                                            try writeF32LittleEndianFile(allocator, replay_path, upstream_grad);
                                            encoder_replay_upstream_probe = .{
                                                .path = replay_path,
                                                .target_layer = opts.debug_encoder_probe_layer,
                                                .segment_start = segment_start,
                                                .segment_end = segment_end,
                                                .batch_size = actual_batch,
                                                .seq_len = max_seq,
                                                .hidden_size = E,
                                                .elems = upstream_grad.len,
                                                .stats = fused_chunker_train.computeBoundaryProbeTensorStats(upstream_grad),
                                            };
                                        }

                                        if (opts.debug_layer_backward_decomp and step_parity_layer_backward_decomp == null) {
                                            step_parity_layer_backward_decomp = try captureStepParityLayerBackwardDecompProbe(
                                                allocator,
                                                &cb,
                                                graph_config,
                                                actual_batch,
                                                max_seq,
                                                opts.debug_encoder_probe_layer,
                                                segment_start,
                                                segment_end,
                                                opts.lora_rank,
                                                la.config.alpha,
                                                upstream_grad,
                                                attn_mask_f32,
                                                act_buf,
                                                captured_segment_probe,
                                                la.layers,
                                            );
                                        }
                                        if (opts.debug_layer_backward_decomp and step_parity_softmax_vjp == null) {
                                            step_parity_softmax_vjp = try captureStepParitySoftmaxVJPProbe(
                                                allocator,
                                                &cb,
                                                actual_batch,
                                                max_seq,
                                                @intCast(graph_config.num_attention_heads),
                                                graph_config.local_attention_window,
                                            );
                                        }
                                        if (opts.debug_qkv_split_vjp and step_parity_qkv_split_vjp == null) {
                                            step_parity_qkv_split_vjp = try captureStepParityQKVSplitVJPProbe(
                                                allocator,
                                                &cb,
                                                actual_batch,
                                                max_seq,
                                                @intCast(graph_config.num_attention_heads),
                                                @intCast(graph_config.hidden_size / graph_config.num_attention_heads),
                                                modern_bert_graph.ropeThetaForLayer(
                                                    graph_config,
                                                    opts.debug_encoder_probe_layer,
                                                ),
                                            );
                                        }

                                        step_parity_segment_vjp = captured_segment_probe;
                                        captured_segment_probe_owned = false;
                                    }
                                    const next_upstream = if (include_hidden_grad) blk: {
                                        const hidden_grad = vjp_result.hidden_grad orelse return error.MissingHiddenGradient;
                                        vjp_result.hidden_grad = null;
                                        break :blk hidden_grad;
                                    } else null;
                                    vjp_result.deinit();
                                    if (next_upstream) |buf| {
                                        if (upstream_owned) |old| allocator.free(old);
                                        upstream_owned = buf;
                                        upstream_grad = buf;
                                    }
                                }
                                layer_cursor = segment_start;
                            }
                            last_vjp_profile = aggregate_profile;
                            direct_vjp_complete = true;
                        } else if (opts.encoder_vjp == .last_layer and la.config.num_layers > 0) {
                            const last_layer_idx = la.config.num_layers - 1;
                            if (fused_chunker_train.isLoRALayerActive(active_layers, last_layer_idx)) {
                                const layer_input = act_buf.findLayerInput(last_layer_idx) orelse return error.MissingSegmentLayerInputCapture;
                                const total_hidden = total_tokens * E;
                                if (layer_input.input.len != total_hidden or lora_output_grad.len != total_hidden) {
                                    return error.InvalidSegmentVJPShape;
                                }
                                const num_heads: u32 = 12;
                                if (opts.hidden_size % num_heads != 0) return error.InvalidHeadDim;
                                const eager_defaults = modern_bert.Config{};
                                const graph_config = modern_bert_graph.Config{
                                    .vocab_size = 50368,
                                    .hidden_size = opts.hidden_size,
                                    .num_hidden_layers = la.config.num_layers,
                                    .num_attention_heads = num_heads,
                                    .head_dim = opts.hidden_size / num_heads,
                                    .intermediate_size = opts.intermediate_size,
                                    .max_position_embeddings = eager_defaults.max_position_embeddings,
                                    .layer_norm_eps = eager_defaults.layer_norm_eps,
                                    .rope_theta = eager_defaults.global_rope_theta,
                                    .local_rope_theta = eager_defaults.local_rope_theta,
                                    .local_attention_window = eager_defaults.local_attention_window,
                                    .global_attn_every_n_layers = eager_defaults.global_attn_every_n_layers,
                                };
                                var rebuild_vjp_session = last_layer_vjp_session == null;
                                if (last_layer_vjp_session) |*session| {
                                    if (session.batch != actual_batch or
                                        session.seq_len != max_seq or
                                        session.start_layer != last_layer_idx or
                                        session.end_layer != last_layer_idx + 1 or
                                        session.config.hidden_size != graph_config.hidden_size or
                                        session.config.intermediate_size != graph_config.intermediate_size)
                                    {
                                        session.deinit();
                                        last_layer_vjp_session = null;
                                        rebuild_vjp_session = true;
                                    }
                                }
                                if (rebuild_vjp_session) {
                                    last_layer_vjp_session = try segmented_encoder.ModernBertSegmentVJPSession.initWithOptions(
                                        allocator,
                                        graph_config,
                                        @intCast(actual_batch),
                                        @intCast(max_seq),
                                        last_layer_idx,
                                        last_layer_idx + 1,
                                        opts.lora_rank,
                                        la.config.alpha,
                                        false,
                                    );
                                }
                                if (last_layer_vjp_session) |*session| {
                                    var vjp_result = try session.execute(
                                        &cb,
                                        layer_input.input,
                                        lora_output_grad,
                                        attn_mask_f32,
                                        la.layers,
                                    );
                                    defer vjp_result.deinit();
                                    last_vjp_profile = vjp_result.profile;
                                    direct_skip_layer = last_layer_idx;
                                }
                            }
                        }

                        // Build parallel slices from ActivationBuffer.
                        var n_caps: usize = 0;
                        if (!direct_vjp_complete) {
                            for (act_buf.items.items) |cap| {
                                if (direct_skip_layer) |skip| {
                                    if (cap.layer_idx == skip) continue;
                                }
                                if (fused_chunker_train.isLoRALayerActive(active_layers, cap.layer_idx)) n_caps += 1;
                            }
                        }
                        const cap_layers = try allocator.alloc(u32, n_caps);
                        defer allocator.free(cap_layers);
                        const cap_modules = try allocator.alloc([]const u8, n_caps);
                        defer allocator.free(cap_modules);
                        const cap_inputs = try allocator.alloc([]const f32, n_caps);
                        defer allocator.free(cap_inputs);
                        const cap_in_feat = try allocator.alloc(usize, n_caps);
                        defer allocator.free(cap_in_feat);
                        const cap_out_feat = try allocator.alloc(usize, n_caps);
                        defer allocator.free(cap_out_feat);
                        var ci: usize = 0;
                        if (!direct_vjp_complete) {
                            for (act_buf.items.items) |cap| {
                                if (direct_skip_layer) |skip| {
                                    if (cap.layer_idx == skip) continue;
                                }
                                if (!fused_chunker_train.isLoRALayerActive(active_layers, cap.layer_idx)) continue;
                                cap_layers[ci] = cap.layer_idx;
                                cap_modules[ci] = cap.module_name;
                                cap_inputs[ci] = cap.input;
                                cap_in_feat[ci] = cap.in_features;
                                cap_out_feat[ci] = cap.out_features;
                                ci += 1;
                            }
                        }

                        if (!direct_vjp_complete) {
                            try segmented_encoder.backwardLoRADirect(
                                cb,
                                allocator,
                                cap_layers,
                                cap_modules,
                                cap_inputs,
                                cap_in_feat,
                                cap_out_feat,
                                lora_output_grad,
                                la.layers,
                                la.config.alpha,
                            );
                        }
                        const debug_update_step_now = opts.debug_update_step > 0 and summary.step == opts.debug_update_step;
                        const lora_grad_clip = sanitizeAndClipLoRAGrads(la.layers, opts.max_grad_norm, summary.boundary_grad_norm);
                        var update_debug = LoRAUpdateDebugAggregate{};
                        defer update_debug.deinit(allocator);
                        if (debug_update_step_now) {
                            print(
                                "lora_update_diagnostic_pre step={d} grad_elems={d} lora_grad_norm_pre_clip={d:.9} lora_grad_norm_post_clip={d:.9} extra_grad_norm_pre_clip={d:.9} extra_grad_norm_post_clip={d:.9} grad_norm_pre_clip={d:.9} grad_norm_post_clip={d:.9} grad_max_abs_pre_clip={d:.9} grad_max_abs_post_clip={d:.9} grad_clip_scale={d:.9} invalid_grad_repaired={d} max_grad_norm={d}\n",
                                .{
                                    summary.step,
                                    lora_grad_clip.elems,
                                    lora_grad_clip.lora_norm_before_clip,
                                    lora_grad_clip.lora_norm_after_clip,
                                    lora_grad_clip.extra_norm_before_clip,
                                    lora_grad_clip.extra_norm_after_clip,
                                    lora_grad_clip.norm_before_clip,
                                    lora_grad_clip.norm_after_clip,
                                    lora_grad_clip.max_abs_before_clip,
                                    lora_grad_clip.max_abs_after_clip,
                                    lora_grad_clip.clip_scale,
                                    lora_grad_clip.nonfinite_repaired,
                                    opts.max_grad_norm,
                                },
                            );
                        }

                        // Apply optimizer steps for all LoRA parameters.
                        // Feature 4 (LoRA+): use lr * lora_plus_ratio for lora_B.
                        // Feature 6 (LLRD): use per-layer decayed learning rate.
                        const base_lr = summary.learning_rate;
                        lora_opt_state.step_count = @intCast(summary.step);
                        const num_layers_f: f32 = @floatFromInt(la.config.num_layers);
                        for (la.layers) |*ll| {
                            if (!fused_chunker_train.isLoRALayerActive(active_layers, ll.layer_idx)) continue;
                            update_debug.active_layers += 1;
                            // LLRD: layer 0 = first encoder layer (shallowest, closest to embeddings) → lowest LR
                            // layer N-1 = last encoder layer (deepest, closest to task head) → highest LR (= base_lr)
                            // Formula: lr[i] = base_lr * decay^(num_layers - 1 - i)
                            const layer_lr = if (opts.llrd_decay != 1.0) blk_lr: {
                                const exp_f: f32 = num_layers_f - 1.0 - @as(f32, @floatFromInt(ll.layer_idx));
                                break :blk_lr base_lr * std.math.pow(f32, opts.llrd_decay, exp_f);
                            } else base_lr;
                            const b_lr = layer_lr * la.config.lora_plus_ratio;
                            const a_key = try std.fmt.allocPrint(
                                allocator,
                                "lora.{d}.{s}.A",
                                .{ ll.layer_idx, ll.module_name },
                            );
                            defer allocator.free(a_key);
                            const b_key = try std.fmt.allocPrint(
                                allocator,
                                "lora.{d}.{s}.B",
                                .{ ll.layer_idx, ll.module_name },
                            );
                            defer allocator.free(b_key);
                            var before_a: ?[]f32 = null;
                            var before_b: ?[]f32 = null;
                            defer if (before_a) |buf| allocator.free(buf);
                            defer if (before_b) |buf| allocator.free(buf);
                            if (debug_update_step_now) {
                                before_a = try allocator.dupe(f32, ll.A);
                                before_b = try allocator.dupe(f32, ll.B);
                            }
                            try optimizers.step(trainer.optimizer, &lora_opt_state, layer_lr, a_key, ll.A, ll.grad_A);
                            try optimizers.step(trainer.optimizer, &lora_opt_state, b_lr, b_key, ll.B, ll.grad_B);
                            if (debug_update_step_now) {
                                update_debug.active_matrices += 2;
                                update_debug.param_stats.addSlice(ll.A);
                                update_debug.param_stats.addSlice(ll.B);
                                try update_debug.update_stats.addDelta(before_a.?, ll.A);
                                try update_debug.update_stats.addDelta(before_b.?, ll.B);
                                const a_state = lora_opt_state.param_states.get(a_key);
                                const b_state = lora_opt_state.param_states.get(b_key);
                                if (a_state) |state| {
                                    update_debug.adam_m_stats.addSlice(state.m);
                                    update_debug.adam_v_stats.addSlice(state.v);
                                }
                                if (b_state) |state| {
                                    update_debug.adam_m_stats.addSlice(state.m);
                                    update_debug.adam_v_stats.addSlice(state.v);
                                }
                                try appendLoRAUpdateMatrixDebug(
                                    allocator,
                                    &update_debug,
                                    ll.layer_idx,
                                    ll.module_name,
                                    "A",
                                    before_a.?,
                                    ll.A,
                                    a_state,
                                );
                                try appendLoRAUpdateMatrixDebug(
                                    allocator,
                                    &update_debug,
                                    ll.layer_idx,
                                    ll.module_name,
                                    "B",
                                    before_b.?,
                                    ll.B,
                                    b_state,
                                );
                                if (update_debug.detail_printed < 12) {
                                    try printLoRAUpdateDebugDetail(summary.step, ll.layer_idx, ll.module_name, "A", layer_lr, ll.grad_A, before_a.?, ll.A, a_state);
                                    update_debug.detail_printed += 1;
                                }
                                if (update_debug.detail_printed < 12) {
                                    try printLoRAUpdateDebugDetail(summary.step, ll.layer_idx, ll.module_name, "B", b_lr, ll.grad_B, before_b.?, ll.B, b_state);
                                    update_debug.detail_printed += 1;
                                }
                            }
                        }
                        const repaired = sanitizeLoRAAdapterSet(la) + optimizers.sanitizeState(&lora_opt_state);
                        if (repaired > 0) {
                            print("warning: sanitized {d} nonfinite LoRA adapter/optimizer values after update at global_step={d}\n", .{
                                repaired,
                                summary.step,
                            });
                        }
                        if (debug_update_step_now) {
                            const update_to_param = if (update_debug.param_stats.l2() > 0.0)
                                update_debug.update_stats.l2() / update_debug.param_stats.l2()
                            else
                                0.0;
                            print(
                                "lora_update_diagnostic step={d} active_adapters={d} active_matrices={d} base_lr={d} lora_plus_ratio={d} optimizer_step_count={d} param_elems={d} param_norm={d:.9} update_elems={d} update_norm={d:.9} update_max_abs={d:.9} update_mean_abs={d:.9} update_to_param={d:.9} adam_m_norm={d:.9} adam_v_norm={d:.9} repaired_after_update={d}\n",
                                .{
                                    summary.step,
                                    update_debug.active_layers,
                                    update_debug.active_matrices,
                                    base_lr,
                                    la.config.lora_plus_ratio,
                                    lora_opt_state.step_count,
                                    update_debug.param_stats.elems,
                                    update_debug.param_stats.l2(),
                                    update_debug.update_stats.elems,
                                    update_debug.update_stats.l2(),
                                    update_debug.update_stats.max_abs,
                                    update_debug.update_stats.meanAbs(),
                                    update_to_param,
                                    update_debug.adam_m_stats.l2(),
                                    update_debug.adam_v_stats.l2(),
                                    repaired,
                                },
                            );
                            if (opts.debug_step_json_path) |json_path| {
                                const boundary_ctx = step_parity_boundary orelse return error.MissingDebugBoundaryStepForParityJson;
                                try writeStepParityJson(
                                    json_path,
                                    "apply_update",
                                    &opts,
                                    trainer.loss_config.lambda_embed,
                                    epoch,
                                    step,
                                    summary.step,
                                    trainer.step_count,
                                    batch_indices,
                                    &batch,
                                    total_tokens,
                                    boundary_ctx,
                                    step_parity_contrastive,
                                    step_parity_upstream_grad_probe,
                                    encoder_replay_upstream_probe,
                                    step_parity_segment_vjp,
                                    step_parity_layer_backward_decomp,
                                    step_parity_softmax_vjp,
                                    step_parity_qkv_split_vjp,
                                    .{
                                        .grad_clip = lora_grad_clip,
                                        .active_adapters = update_debug.active_layers,
                                        .active_matrices = update_debug.active_matrices,
                                        .base_lr = base_lr,
                                        .lora_plus_ratio = la.config.lora_plus_ratio,
                                        .optimizer_step_count = lora_opt_state.step_count,
                                        .param_elems = update_debug.param_stats.elems,
                                        .param_norm = update_debug.param_stats.l2(),
                                        .update_elems = update_debug.update_stats.elems,
                                        .update_norm = update_debug.update_stats.l2(),
                                        .update_max_abs = update_debug.update_stats.max_abs,
                                        .update_mean_abs = update_debug.update_stats.meanAbs(),
                                        .update_to_param = update_to_param,
                                        .adam_m_norm = update_debug.adam_m_stats.l2(),
                                        .adam_v_norm = update_debug.adam_v_stats.l2(),
                                        .repaired_after_update = repaired,
                                        .matrix_stats = update_debug.matrix_stats.items,
                                    },
                                    la,
                                );
                            }
                            if (opts.debug_update_step_exit) {
                                print("debug_update_step exiting after optimizer update\n", .{});
                                return;
                            }
                        }
                        la.zeroGrads();
                        step_timing.lora_update_ns = elapsedNs(lora_update_start_ns);
                    }
                } else {
                    // LoRA is disabled for this epoch or no activations were captured
                    // (for example, encoder weights are not loaded). Train heads only.
                    const train_step_start_ns = nowNs();
                    summary = try trainer.trainStep(
                        allocator,
                        features,
                        boundary_labels_2,
                        attn_mask_f32,
                        batch.input_ids[0..total_tokens],
                        batch.boundary_candidate_mask[0..total_tokens],
                        contrastive_embeddings,
                        contrastive_mask,
                        contrastive_doc_ids,
                        total_tokens,
                        contrastive_B,
                        contrastive_C,
                        E,
                    );
                    step_timing.train_ns = elapsedNs(train_step_start_ns);
                }
            } else {
                const train_step_start_ns = nowNs();
                summary = try trainer.trainStep(
                    allocator,
                    features,
                    boundary_labels_2,
                    attn_mask_f32,
                    batch.input_ids[0..total_tokens],
                    batch.boundary_candidate_mask[0..total_tokens],
                    contrastive_embeddings,
                    contrastive_mask,
                    contrastive_doc_ids,
                    total_tokens,
                    contrastive_B,
                    contrastive_C,
                    E,
                );
                step_timing.train_ns = elapsedNs(train_step_start_ns);
            }

            // ------------------------------------------------------------------
            // SPLADE training: forward + backward + AdamW update for W.
            // Activates only after splade_focus_epoch so boundary training
            // stabilises first.
            // ------------------------------------------------------------------
            const splade_active = opts.splade and encoder_loaded and epoch >= @as(usize, opts.splade_focus_epoch);
            const splade_start_ns = nowNs();
            if (splade_active) {
                if (splade_w) |w| {
                    // Build a fused_chunker Config with just the fields we need.
                    const splade_fused_config = fused_chunker_mod.Config{
                        .hidden_size = opts.hidden_size,
                        .splade_config = .{
                            .vocab_size = @intCast(splade_vocab_size),
                            .pooling = .max,
                        },
                    };

                    // Count valid chunks first so we can allocate compact arrays.
                    var num_valid_chunks: usize = 0;
                    for (0..actual_batch * C) |ci| {
                        if (chunk_mask[ci] > 0.5) num_valid_chunks += 1;
                    }

                    if (num_valid_chunks >= 2) {
                        // Compute per-chunk SPLADE vectors (compact: num_valid_chunks entries).
                        // features: [actual_batch * max_seq * hidden_size]
                        const splade_vecs = try fused_chunker_mod.computeChunkSpladeVectors(
                            allocator,
                            splade_fused_config,
                            features,
                            w,
                            batch.chunk_starts,
                            batch.chunk_ends,
                            chunk_mask,
                            actual_batch,
                            max_seq,
                            C,
                        );
                        defer allocator.free(splade_vecs);

                        // Build compact all-ones mask and compact doc_ids for the
                        // contrastive loss (splade_vecs is already compact).
                        const compact_mask = try allocator.alloc(f32, num_valid_chunks);
                        defer allocator.free(compact_mask);
                        @memset(compact_mask, 1.0);

                        const compact_doc_ids = try allocator.alloc(u32, num_valid_chunks);
                        defer allocator.free(compact_doc_ids);
                        var vi_fill: usize = 0;
                        for (0..actual_batch) |b_idx| {
                            for (0..C) |c_idx| {
                                if (chunk_mask[b_idx * C + c_idx] > 0.5) {
                                    compact_doc_ids[vi_fill] = @intCast(b_idx);
                                    vi_fill += 1;
                                }
                            }
                        }

                        // Contrastive loss + gradient w.r.t. splade_vecs.
                        var splade_contrastive = try fused_chunker_splade.computeSpladeContrastiveLoss(
                            allocator,
                            splade_vecs,
                            compact_mask,
                            compact_doc_ids,
                            num_valid_chunks,
                            @intCast(splade_vocab_size),
                            (fused_chunker_loss.FusedLossConfig{}).temperature,
                        );
                        defer splade_contrastive.deinit(allocator);

                        // FLOPS regularization loss + gradient.
                        const flops_grad = try allocator.alloc(f32, num_valid_chunks * splade_vocab_size);
                        defer allocator.free(flops_grad);
                        const flops_loss = fused_chunker_splade.computeSpladeFlopsLoss(
                            splade_vecs,
                            flops_grad,
                            num_valid_chunks,
                            @intCast(splade_vocab_size),
                            opts.lambda_flops,
                        );
                        _ = flops_loss;

                        // Combined gradient: lambda_splade * contrastive_grad + flops_grad.
                        const combined_grad = try allocator.alloc(f32, num_valid_chunks * splade_vocab_size);
                        defer allocator.free(combined_grad);
                        for (combined_grad, splade_contrastive.grad, flops_grad) |*cg, sg, fg| {
                            cg.* = opts.lambda_splade * sg + fg;
                        }

                        // Backprop through SPLADE to get dL/dW.
                        // Re-run forward with info per chunk to get argmax tokens.
                        const dW = try allocator.alloc(f32, splade_vocab_size * @as(usize, opts.hidden_size));
                        defer allocator.free(dW);
                        @memset(dW, 0);

                        var valid_chunk_idx: usize = 0;
                        for (0..actual_batch) |b_idx| {
                            for (0..C) |c_idx| {
                                const mask_val = chunk_mask[b_idx * C + c_idx];
                                if (mask_val < 0.5) continue;

                                const tok_start: usize = @intCast(@max(0, batch.chunk_starts[b_idx * C + c_idx]));
                                const tok_end: usize = @min(
                                    @as(usize, @intCast(@max(0, batch.chunk_ends[b_idx * C + c_idx]))),
                                    max_seq,
                                );
                                if (tok_start >= tok_end) {
                                    valid_chunk_idx += 1;
                                    continue;
                                }

                                const chunk_tokens = tok_end - tok_start;
                                const H: usize = opts.hidden_size;
                                const hidden_offset = b_idx * max_seq * H + tok_start * H;
                                const chunk_hidden = features[hidden_offset .. hidden_offset + chunk_tokens * H];

                                var info = try fused_chunker_splade.computeSpladeActivationWithInfo(
                                    allocator,
                                    chunk_hidden,
                                    w,
                                    chunk_tokens,
                                    H,
                                    @intCast(splade_vocab_size),
                                );
                                defer info.deinit();

                                const chunk_grad = combined_grad[valid_chunk_idx * splade_vocab_size .. (valid_chunk_idx + 1) * splade_vocab_size];
                                fused_chunker_splade.backwardSpladeWeight(chunk_grad, &info, chunk_hidden, H, dW);

                                valid_chunk_idx += 1;
                            }
                        }

                        // AdamW update for W.
                        splade_adam_step += 1;
                        const splade_lr = summary.learning_rate;
                        if (splade_lr > 0) {
                            const beta1: f32 = 0.9;
                            const beta2: f32 = 0.999;
                            const eps: f32 = 1e-8;
                            const t_f: f32 = @floatFromInt(splade_adam_step);
                            const bc1: f32 = 1.0 - std.math.pow(f32, beta1, t_f);
                            const bc2: f32 = 1.0 - std.math.pow(f32, beta2, t_f);
                            for (w, splade_adam_m.?, splade_adam_v.?, dW) |*wi, *mi, *vi, gi| {
                                mi.* = beta1 * mi.* + (1.0 - beta1) * gi;
                                vi.* = beta2 * vi.* + (1.0 - beta2) * gi * gi;
                                const m_hat = mi.* / bc1;
                                const v_hat = vi.* / bc2;
                                wi.* -= splade_lr * m_hat / (@sqrt(v_hat) + eps);
                            }
                        }

                        print("  splade_loss: {d:.4}\n", .{splade_contrastive.loss});
                    }
                }
            }
            step_timing.splade_ns = if (splade_active) elapsedNs(splade_start_ns) else 0;

            step_timing.total_ns = elapsedNs(step_start_ns);
            epoch_timing.add(step_timing, actual_batch);
            step += 1;
            last_step_loss = summary.total_loss;

            const supervision_counts = computeSupervisionCounts(
                boundary_labels_2,
                attn_mask_f32,
                total_tokens,
            );
            const sample_memory_now = opts.memory_sample_every <= 1 or
                summary.step == 1 or
                summary.step % opts.memory_sample_every == 0;
            const memory_snapshot = if (sample_memory_now)
                memory_tracker.observe()
            else
                ProcessMemorySnapshot{
                    .available = memory_tracker.last_resident_bytes > 0,
                    .resident_bytes = memory_tracker.last_resident_bytes,
                    .footprint_bytes = 0,
                };
            try writeStepMetric(
                &metrics_writer,
                epoch,
                step,
                actual_batch,
                total_tokens,
                summary,
                step_timing,
                supervision_counts,
                use_metal,
                last_vjp_profile,
                memory_snapshot,
                memory_tracker.peak_resident_bytes,
            );
            if (memory_snapshot.available and opts.memory_warn_rss_bytes > 0 and
                !memory_tracker.warning_emitted and
                memory_snapshot.resident_bytes >= opts.memory_warn_rss_bytes)
            {
                memory_tracker.warning_emitted = true;
                print("memory_watchdog warning resident_bytes={d} warn_threshold={d} peak_resident_bytes={d} global_step={d}\n", .{
                    memory_snapshot.resident_bytes,
                    opts.memory_warn_rss_bytes,
                    memory_tracker.peak_resident_bytes,
                    summary.step,
                });
            }
            if (memory_snapshot.available and opts.memory_abort_rss_bytes > 0 and
                memory_snapshot.resident_bytes >= opts.memory_abort_rss_bytes)
            {
                const abort_ckpt_path = try std.fmt.allocPrint(
                    allocator,
                    "{s}/checkpoint_memory_abort_{d}.safetensors",
                    .{ opts.output_dir, summary.step },
                );
                defer allocator.free(abort_ckpt_path);
                try savePeriodicTrainingCheckpoint(
                    allocator,
                    &opts,
                    &trainer,
                    if (lora_adapters_opt) |*la| la else null,
                    &lora_opt_state,
                    splade_w,
                    splade_vocab_size,
                    "memory_abort",
                    summary.step,
                );
                try writeFusedTrainingManifest(
                    allocator,
                    manifest_path,
                    &opts,
                    if (use_metal) "metal" else "native",
                    metrics_path,
                    train_stats,
                    validation_stats,
                    "memory_abort",
                    trainer.step_count,
                    summary.total_loss,
                    abort_ckpt_path,
                    @intCast(best_val_step),
                    best_val_epoch,
                    best_val_f1,
                    boundary_threshold_recommendation,
                    memory_tracker.peak_resident_bytes,
                );
                print("memory_watchdog abort resident_bytes={d} abort_threshold={d} peak_resident_bytes={d} checkpoint={s}\n", .{
                    memory_snapshot.resident_bytes,
                    opts.memory_abort_rss_bytes,
                    memory_tracker.peak_resident_bytes,
                    abort_ckpt_path,
                });
                return error.MemoryAbortThresholdExceeded;
            }

            if (opts.log_every > 0 and (step == 1 or step % opts.log_every == 0)) {
                print(
                    "epoch {d}/{d} step {d} | loss {d:.4} | boundary {d:.4} | contrastive {d:.4} | lr {d} | boundary_head_lr {d} | step_ms {d:.2} | batch {d:.2} | refresh {d:.2} | enc {d:.2} | hn {d:.2} | train {d:.2} | lora {d:.2} | splade {d:.2} | ex/s {d:.1}\n",
                    .{
                        epoch + 1,
                        opts.epochs,
                        summary.step,
                        summary.total_loss,
                        summary.boundary_loss,
                        summary.contrastive_loss,
                        summary.learning_rate,
                        summary.boundary_head_learning_rate,
                        nsToMs(step_timing.total_ns),
                        nsToMs(step_timing.batch_ns),
                        nsToMs(step_timing.lora_refresh_ns),
                        nsToMs(step_timing.encoder_ns),
                        nsToMs(step_timing.hard_neg_ns),
                        nsToMs(step_timing.train_ns),
                        nsToMs(step_timing.lora_update_ns),
                        nsToMs(step_timing.splade_ns),
                        examplesPerSecond(@intCast(actual_batch), step_timing.total_ns),
                    },
                );
                if (last_vjp_profile) |profile| {
                    print(
                        "  vjp_profile | input {d:.2} | exec {d:.2} | extract {d:.2} | accum {d:.2} | compiled_total {d:.2} | total {d:.2} | runtime {s} | parts {d} | cmds {d} | planned {d} | graph_regions {d} | fallbacks {d}\n",
                        .{
                            nsToMs(profile.runtime_input_ns),
                            nsToMs(profile.compiled_execute_ns),
                            nsToMs(profile.compiled_extract_ns),
                            nsToMs(profile.accumulate_ns),
                            nsToMs(profile.compiled_total_ns),
                            nsToMs(profile.total_ns),
                            if (profile.mpsgraph_runtime) "mpsgraph" else if (profile.partitioned_runtime) "partitioned" else "interpreter",
                            profile.partitions_executed,
                            profile.backend_command_dispatches,
                            profile.planned_operator_dispatches,
                            profile.graph_region_dispatches,
                            profile.interpreter_fallbacks,
                        },
                    );
                }
            }

            if (opts.checkpoint_every_steps > 0 and summary.step > 0 and summary.step % opts.checkpoint_every_steps == 0) {
                try savePeriodicTrainingCheckpoint(
                    allocator,
                    &opts,
                    &trainer,
                    if (lora_adapters_opt) |*la| la else null,
                    &lora_opt_state,
                    splade_w,
                    splade_vocab_size,
                    "step",
                    summary.step,
                );
            }

            if (val_samples.len > 0 and opts.eval_every_steps > 0 and summary.step > 0 and summary.step % opts.eval_every_steps == 0) {
                var step_fitted_threshold: ?f32 = null;
                if (opts.step_train_eval_max_examples > 0 and samples.len > 0) {
                    const train_eval_start_ns = nowNs();
                    const train_eval_count = @min(opts.step_train_eval_max_examples, samples.len);
                    const train_eval_samples = samples[0..train_eval_count];
                    const train_eval_summary = try evaluateBoundarySamplesStreaming(
                        allocator,
                        &opts,
                        &trainer,
                        &cb,
                        &weight_store,
                        &metal_weight_store,
                        if (tokenizer_opt) |*tb| tb else null,
                        train_eval_samples,
                        encoder_loaded,
                        use_metal,
                        if (lora_adapters_opt) |*la| la else null,
                        "train_validation",
                        summary.step,
                        null,
                    );
                    step_fitted_threshold = train_eval_summary.best_boundary_threshold;
                    const train_eval_ms = nsToMs(elapsedNs(train_eval_start_ns));
                    print(
                        "train validation step {d} epoch {d}/{d} samples={d}/{d} | f1 {d:.4} precision {d:.4} recall {d:.4} counts tp={d} fp={d} fn={d} | best_f1 {d:.4} threshold {d:.2} best_counts tp={d} fp={d} fn={d} | rank ap={d:.6} p_at_gold={d:.4} rank_f1={d:.4} rank_threshold={d:.6} | valid={d} gold_pos={d} gold_rate={d:.6} pred_rate={d:.6} best_pred_rate={d:.6} prob_pos={d:.4} prob_neg={d:.4} margin_pos={d:.4} margin_neg={d:.4} | eval_ms {d:.2}\n",
                        .{
                            summary.step,
                            epoch + 1,
                            opts.epochs,
                            train_eval_samples.len,
                            samples.len,
                            train_eval_summary.boundary_f1,
                            train_eval_summary.boundary_precision,
                            train_eval_summary.boundary_recall,
                            train_eval_summary.boundary_tp,
                            train_eval_summary.boundary_fp,
                            train_eval_summary.boundary_fn,
                            train_eval_summary.best_boundary_f1,
                            train_eval_summary.best_boundary_threshold,
                            train_eval_summary.best_boundary_tp,
                            train_eval_summary.best_boundary_fp,
                            train_eval_summary.best_boundary_fn,
                            train_eval_summary.average_precision,
                            train_eval_summary.precision_at_gold_count,
                            train_eval_summary.max_rank_f1,
                            train_eval_summary.max_rank_threshold,
                            train_eval_summary.valid_tokens,
                            train_eval_summary.gold_positives,
                            train_eval_summary.gold_positive_rate,
                            train_eval_summary.predicted_positive_rate,
                            train_eval_summary.best_predicted_positive_rate,
                            train_eval_summary.mean_positive_probability_gold_positive,
                            train_eval_summary.mean_positive_probability_gold_negative,
                            train_eval_summary.mean_boundary_margin_gold_positive,
                            train_eval_summary.mean_boundary_margin_gold_negative,
                            train_eval_ms,
                        },
                    );
                    try writeValidationMetric(
                        &metrics_writer,
                        "train_validation_step",
                        epoch,
                        summary.step,
                        train_eval_samples.len,
                        samples.len,
                        train_eval_ms,
                        train_eval_summary,
                    );
                    fused_chunker_train.printBoundaryQualityDiagnostics("train_validation_step_quality", train_eval_summary);
                }

                const val_eval_start_ns = nowNs();
                const step_eval_samples = if (opts.step_eval_max_examples > 0 and opts.step_eval_max_examples < val_samples.len)
                    val_samples[0..opts.step_eval_max_examples]
                else
                    val_samples;
                const step_eval_is_full = step_eval_samples.len == val_samples.len;
                const val_summary = try evaluateBoundarySamplesStreaming(
                    allocator,
                    &opts,
                    &trainer,
                    &cb,
                    &weight_store,
                    &metal_weight_store,
                    if (tokenizer_opt) |*tb| tb else null,
                    step_eval_samples,
                    encoder_loaded,
                    use_metal,
                    if (lora_adapters_opt) |*la| la else null,
                    "validation",
                    summary.step,
                    step_fitted_threshold,
                );
                const eval_ms = nsToMs(elapsedNs(val_eval_start_ns));
                print(
                    "validation step {d} epoch {d}/{d} samples={d}/{d} | f1 {d:.4} precision {d:.4} recall {d:.4} counts tp={d} fp={d} fn={d} | best_f1 {d:.4} threshold {d:.2} best_counts tp={d} fp={d} fn={d} | rank ap={d:.6} p_at_gold={d:.4} rank_f1={d:.4} rank_threshold={d:.6} | valid={d} gold_pos={d} gold_rate={d:.6} pred_rate={d:.6} best_pred_rate={d:.6} prob_pos={d:.4} prob_neg={d:.4} margin_pos={d:.4} margin_neg={d:.4} | eval_ms {d:.2}\n",
                    .{
                        summary.step,
                        epoch + 1,
                        opts.epochs,
                        step_eval_samples.len,
                        val_samples.len,
                        val_summary.boundary_f1,
                        val_summary.boundary_precision,
                        val_summary.boundary_recall,
                        val_summary.boundary_tp,
                        val_summary.boundary_fp,
                        val_summary.boundary_fn,
                        val_summary.best_boundary_f1,
                        val_summary.best_boundary_threshold,
                        val_summary.best_boundary_tp,
                        val_summary.best_boundary_fp,
                        val_summary.best_boundary_fn,
                        val_summary.average_precision,
                        val_summary.precision_at_gold_count,
                        val_summary.max_rank_f1,
                        val_summary.max_rank_threshold,
                        val_summary.valid_tokens,
                        val_summary.gold_positives,
                        val_summary.gold_positive_rate,
                        val_summary.predicted_positive_rate,
                        val_summary.best_predicted_positive_rate,
                        val_summary.mean_positive_probability_gold_positive,
                        val_summary.mean_positive_probability_gold_negative,
                        val_summary.mean_boundary_margin_gold_positive,
                        val_summary.mean_boundary_margin_gold_negative,
                        eval_ms,
                    },
                );
                try writeValidationMetric(
                    &metrics_writer,
                    "validation_step",
                    epoch,
                    summary.step,
                    step_eval_samples.len,
                    val_samples.len,
                    eval_ms,
                    val_summary,
                );
                last_validation_summary = val_summary;
                last_validation_event = "validation_step";
                last_validation_epoch_index = epoch;
                last_validation_step = summary.step;
                last_validation_samples = step_eval_samples.len;
                maybeUpdateBoundaryThresholdRecommendation(
                    &boundary_threshold_recommendation,
                    val_summary,
                    "validation_step",
                    summary.step,
                    @intCast(epoch + 1),
                    step_eval_samples.len,
                    val_samples.len,
                );
                fused_chunker_train.printBoundaryQualityDiagnostics("validation_step_quality", val_summary);

                if (step_eval_is_full and val_summary.boundary_f1 > best_val_f1) {
                    best_val_f1 = val_summary.boundary_f1;
                    best_val_epoch = @intCast(epoch + 1);
                    best_val_step = summary.step;
                    var best_path_buf: [512]u8 = undefined;
                    const best_path = try std.fmt.bufPrint(&best_path_buf, "{s}/best_model.safetensors", .{opts.output_dir});
                    try saveFusedCheckpoint(allocator, best_path, &trainer, if (lora_adapters_opt) |*la| la else null);
                    print("best checkpoint saved to {s} (step={d} epoch={d} f1={d:.4})\n", .{
                        best_path,
                        best_val_step,
                        best_val_epoch,
                        best_val_f1,
                    });
                }
            }

            if (opts.max_steps > 0 and summary.step >= opts.max_steps) {
                print("max steps reached at global_step={d}; stopping training loop\n", .{summary.step});
                stop_training = true;
                break;
            }
        }

        print(
            "epoch {d}/{d} done  steps={d} | avg_step_ms {d:.2} | batch {d:.2} | refresh {d:.2} | enc {d:.2} | hn {d:.2} | train {d:.2} | lora {d:.2} | splade {d:.2} | throughput_ex_s {d:.1}\n",
            .{
                epoch + 1,
                opts.epochs,
                step,
                avgMs(epoch_timing.total_ns, epoch_timing.steps),
                avgMs(epoch_timing.batch_ns, epoch_timing.steps),
                avgMs(epoch_timing.lora_refresh_ns, epoch_timing.steps),
                avgMs(epoch_timing.encoder_ns, epoch_timing.steps),
                avgMs(epoch_timing.hard_neg_ns, epoch_timing.steps),
                avgMs(epoch_timing.train_ns, epoch_timing.steps),
                avgMs(epoch_timing.lora_update_ns, epoch_timing.steps),
                avgMs(epoch_timing.splade_ns, epoch_timing.steps),
                examplesPerSecond(epoch_timing.examples, epoch_timing.total_ns),
            },
        );
        try writeEpochMetric(&metrics_writer, epoch, opts.epochs, step, epoch_timing);

        // Flush any partial gradient accumulation window left at epoch end.
        try trainer.flushEpochEnd(allocator);

        if (stop_training) {
            break;
        }

        // Optional checkpoint save
        if (opts.checkpoint_every > 0 and (epoch + 1) % opts.checkpoint_every == 0) {
            try savePeriodicTrainingCheckpoint(
                allocator,
                &opts,
                &trainer,
                if (lora_adapters_opt) |*la| la else null,
                &lora_opt_state,
                splade_w,
                splade_vocab_size,
                "epoch",
                @intCast(epoch + 1),
            );
        }

        if (val_samples.len > 0 and opts.eval_every > 0 and (epoch + 1) % opts.eval_every == 0) {
            const val_eval_start_ns = nowNs();
            const val_summary = try evaluateBoundarySamplesStreaming(
                allocator,
                &opts,
                &trainer,
                &cb,
                &weight_store,
                &metal_weight_store,
                if (tokenizer_opt) |*tb| tb else null,
                val_samples,
                encoder_loaded,
                use_metal,
                if (lora_adapters_opt) |*la| la else null,
                "epoch_validation",
                trainer.step_count,
                null,
            );
            const eval_ms = nsToMs(elapsedNs(val_eval_start_ns));
            print(
                "validation epoch {d}/{d} | f1 {d:.4} precision {d:.4} recall {d:.4} counts tp={d} fp={d} fn={d} | best_f1 {d:.4} threshold {d:.2} best_counts tp={d} fp={d} fn={d} | rank ap={d:.6} p_at_gold={d:.4} rank_f1={d:.4} rank_threshold={d:.6} | valid={d} gold_pos={d} gold_rate={d:.6} pred_rate={d:.6} best_pred_rate={d:.6} prob_pos={d:.4} prob_neg={d:.4} margin_pos={d:.4} margin_neg={d:.4} | eval_ms {d:.2}\n",
                .{
                    epoch + 1,
                    opts.epochs,
                    val_summary.boundary_f1,
                    val_summary.boundary_precision,
                    val_summary.boundary_recall,
                    val_summary.boundary_tp,
                    val_summary.boundary_fp,
                    val_summary.boundary_fn,
                    val_summary.best_boundary_f1,
                    val_summary.best_boundary_threshold,
                    val_summary.best_boundary_tp,
                    val_summary.best_boundary_fp,
                    val_summary.best_boundary_fn,
                    val_summary.average_precision,
                    val_summary.precision_at_gold_count,
                    val_summary.max_rank_f1,
                    val_summary.max_rank_threshold,
                    val_summary.valid_tokens,
                    val_summary.gold_positives,
                    val_summary.gold_positive_rate,
                    val_summary.predicted_positive_rate,
                    val_summary.best_predicted_positive_rate,
                    val_summary.mean_positive_probability_gold_positive,
                    val_summary.mean_positive_probability_gold_negative,
                    val_summary.mean_boundary_margin_gold_positive,
                    val_summary.mean_boundary_margin_gold_negative,
                    eval_ms,
                },
            );
            try writeValidationMetric(
                &metrics_writer,
                "validation_epoch",
                epoch,
                trainer.step_count,
                val_samples.len,
                val_samples.len,
                eval_ms,
                val_summary,
            );
            last_validation_summary = val_summary;
            last_validation_event = "validation_epoch";
            last_validation_epoch_index = epoch;
            last_validation_step = trainer.step_count;
            last_validation_samples = val_samples.len;
            maybeUpdateBoundaryThresholdRecommendation(
                &boundary_threshold_recommendation,
                val_summary,
                "validation_epoch",
                trainer.step_count,
                @intCast(epoch + 1),
                val_samples.len,
                val_samples.len,
            );
            fused_chunker_train.printBoundaryQualityDiagnostics("validation_epoch_quality", val_summary);

            if (val_summary.boundary_f1 > best_val_f1) {
                best_val_f1 = val_summary.boundary_f1;
                best_val_epoch = @intCast(epoch + 1);
                best_val_step = trainer.step_count;
                var best_path_buf: [512]u8 = undefined;
                const best_path = try std.fmt.bufPrint(&best_path_buf, "{s}/best_model.safetensors", .{opts.output_dir});
                try saveFusedCheckpoint(allocator, best_path, &trainer, if (lora_adapters_opt) |*la| la else null);
                print("best checkpoint saved to {s} (step={d} epoch={d} f1={d:.4})\n", .{
                    best_path,
                    best_val_step,
                    best_val_epoch,
                    best_val_f1,
                });
            }
        }
    }

    // ------------------------------------------------------------------
    // 6. Save final checkpoint
    // ------------------------------------------------------------------
    var final_buf: [512]u8 = undefined;
    const final_path = try std.fmt.bufPrint(&final_buf, "{s}/checkpoint_final.safetensors", .{opts.output_dir});
    try saveFusedCheckpoint(allocator, final_path, &trainer, if (lora_adapters_opt) |*la| la else null);
    print("final checkpoint saved to {s}\n", .{final_path});

    if (opts.save_optimizer_state) {
        var opt_final_buf: [512]u8 = undefined;
        const opt_final_path = try std.fmt.bufPrint(&opt_final_buf, "{s}/checkpoint_final_optimizer.safetensors", .{opts.output_dir});
        try saveFusedOptimizerState(
            allocator,
            opt_final_path,
            &trainer,
            if (lora_adapters_opt) |*la| la else null,
            &lora_opt_state,
        );
        print("optimizer state saved to {s}\n", .{opt_final_path});
    }

    if (opts.checkpoint_roundtrip_eval and val_samples.len > 0) {
        const roundtrip_samples = if (opts.checkpoint_roundtrip_max_examples > 0 and opts.checkpoint_roundtrip_max_examples < val_samples.len)
            val_samples[0..opts.checkpoint_roundtrip_max_examples]
        else
            val_samples;

        var roundtrip_trainer = try FusedTrainer.init(allocator, config, &cb);
        defer roundtrip_trainer.deinit();
        try roundtrip_trainer.loadCheckpoint(allocator, final_path);

        var roundtrip_lora_adapters_opt: ?fused_chunker_lora.LoRAAdapterSet = null;
        defer if (roundtrip_lora_adapters_opt) |*la| la.deinit();
        if (lora_adapters_opt) |*la| {
            roundtrip_lora_adapters_opt = try fused_chunker_lora.LoRAAdapterSet.init(
                allocator,
                la.config,
                @intCast(opts.hidden_size),
                @intCast(opts.intermediate_size),
            );
            const loaded_lora = try loadLoRAAdaptersFromCheckpoint(allocator, final_path, &roundtrip_lora_adapters_opt.?);
            print("checkpoint_same_process_roundtrip loaded_lora_tensors={d} checkpoint={s}\n", .{ loaded_lora, final_path });
        }

        const roundtrip_eval_start_ns = nowNs();
        const roundtrip_summary = try evaluateBoundarySamplesStreaming(
            allocator,
            &opts,
            &roundtrip_trainer,
            &cb,
            &weight_store,
            &metal_weight_store,
            if (tokenizer_opt) |*tb| tb else null,
            roundtrip_samples,
            encoder_loaded,
            use_metal,
            if (roundtrip_lora_adapters_opt) |*la| la else null,
            "checkpoint_roundtrip",
            trainer.step_count,
            null,
        );
        const roundtrip_eval_ms = nsToMs(elapsedNs(roundtrip_eval_start_ns));
        print(
            "checkpoint_same_process_roundtrip_validation samples={d}/{d} | f1 {d:.4} best_f1 {d:.4} ap {d:.6} rank_f1 {d:.4} prob_pos={d:.4} prob_neg={d:.4} | eval_ms {d:.2}\n",
            .{
                roundtrip_samples.len,
                val_samples.len,
                roundtrip_summary.boundary_f1,
                roundtrip_summary.best_boundary_f1,
                roundtrip_summary.average_precision,
                roundtrip_summary.max_rank_f1,
                roundtrip_summary.mean_positive_probability_gold_positive,
                roundtrip_summary.mean_positive_probability_gold_negative,
                roundtrip_eval_ms,
            },
        );
        if (last_validation_summary) |live| {
            print(
                "checkpoint_same_process_roundtrip_delta live_event={s} live_step={d} live_samples={d} delta_f1={d:.6} delta_best_f1={d:.6} delta_ap={d:.6} delta_prob_gap={d:.6}\n",
                .{
                    last_validation_event,
                    last_validation_step,
                    last_validation_samples,
                    roundtrip_summary.boundary_f1 - live.boundary_f1,
                    roundtrip_summary.best_boundary_f1 - live.best_boundary_f1,
                    roundtrip_summary.average_precision - live.average_precision,
                    (roundtrip_summary.mean_positive_probability_gold_positive - roundtrip_summary.mean_positive_probability_gold_negative) -
                        (live.mean_positive_probability_gold_positive - live.mean_positive_probability_gold_negative),
                },
            );
        }
        try writeValidationMetric(
            &metrics_writer,
            "checkpoint_same_process_roundtrip_validation",
            last_validation_epoch_index,
            trainer.step_count,
            roundtrip_samples.len,
            val_samples.len,
            roundtrip_eval_ms,
            roundtrip_summary,
        );
        fused_chunker_train.printBoundaryQualityDiagnostics("checkpoint_same_process_roundtrip_quality", roundtrip_summary);
    }

    // Save final SPLADE projection weight W.
    if (splade_w) |w| {
        const splade_final_path = try std.fmt.allocPrint(allocator, "{s}/splade_w_final.safetensors", .{opts.output_dir});
        defer allocator.free(splade_final_path);
        const splade_tensors = [_]safetensors_checkpoint.NamedTensor{
            .{ .name = "splade_proj_weight", .data = w, .shape = &.{ splade_vocab_size, @as(usize, opts.hidden_size) } },
        };
        try safetensors_checkpoint.save(allocator, splade_final_path, &splade_tensors);
        print("SPLADE weight saved to {s}\n", .{splade_final_path});
    }

    if (best_val_epoch > 0) {
        print("best validation checkpoint: step={d} epoch={d} f1={d:.4} path={s}/best_model.safetensors\n", .{
            best_val_step,
            best_val_epoch,
            best_val_f1,
            opts.output_dir,
        });
    }

    try writeFusedTrainingManifest(
        allocator,
        manifest_path,
        &opts,
        if (use_metal) "metal" else "native",
        metrics_path,
        train_stats,
        validation_stats,
        "complete",
        trainer.step_count,
        last_step_loss,
        final_path,
        @intCast(best_val_step),
        best_val_epoch,
        best_val_f1,
        boundary_threshold_recommendation,
        memory_tracker.peak_resident_bytes,
    );

    print("training complete\n", .{});
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

fn printUsage() void {
    print(
        \\usage: train-fused-chunker --data <path> --output <dir> [options]
        \\
        \\  --data <path>             JSONL data path (file or directory)
        \\  --output <dir>            Output directory for checkpoints
        \\  --val-data <path>         Optional JSONL validation data path
        \\  --model-dir <dir>         Model directory (tokenizer + encoder weights)
        \\  --epochs <n>              Number of epochs (default: 10)
        \\  --batch-size <n>          Batch size (default: 16)
        \\  --lr <f>                  Learning rate (default: 1e-4)
        \\  --learning-rate <f>       Alias for --lr
        \\  --boundary-head-lr-multiplier <f> Multiplier on base LR for boundary-head weights only (default: 1.0)
        \\  --warmup-steps <n>        Linear warmup steps (default: 200)
        \\  --lr-total-steps <n>      Override cosine LR schedule total steps (default: epochs * steps/epoch)
        \\  --weight-decay <f>        AdamW weight decay (default: 0.01)
        \\  --max-grad-norm <f>       Global gradient clip norm (default: 0, disabled)
        \\  --beta1 <f>               Adam beta1 (default: 0.9)
        \\  --beta2 <f>               Adam beta2 (default: 0.999)
        \\  --adam-epsilon <f>        Adam epsilon (default: 1e-8)
        \\  --lambda-chunk <f>        Boundary loss weight (default: 1.0)
        \\  --lambda-embed <f>        Dense contrastive loss weight (default: 0.3)
        \\  --boundary-focus-epochs <n> Epochs using reduced contrastive weight (default: 3)
        \\  --boundary-focus-lambda-embed <f> Dense contrastive weight during boundary focus (default: 0.1)
        \\  --boundary-loss-type ce|focal Boundary loss type (default: ce)
        \\  --loss-type ce|focal      Alias for --boundary-loss-type
        \\  --boundary-focal-gamma <f> Boundary focal gamma when loss type is focal (default: 2.0)
        \\  --focal-gamma <f>         Alias for --boundary-focal-gamma
        \\  --boundary-focal-alpha <f> Boundary focal positive alpha when loss type is focal (default: 0.75)
        \\  --focal-alpha <f>         Alias for --boundary-focal-alpha
        \\  --contrastive-focal-gamma <f> Dense contrastive focal gamma (default: 0.0, disabled)
        \\  --contrastive-focal-alpha <f> Dense contrastive focal alpha (default: 0.75)
        \\  --boundary-pos-weight <f|auto> Positive class weight for CE loss (default: 5.0)
        \\  --pos-weight <f|auto>     Alias for --boundary-pos-weight
        \\  --boundary-pos-weight-auto Estimate CE positive weight from tokenized train labels
        \\  --boundary-pos-weight-auto-max-examples <n> Max train examples for auto estimate (default: 2048)
        \\  --boundary-rank-loss-weight <f> Pairwise gold-vs-hard-negative boundary rank loss weight (default: 0.0, disabled)
        \\  --boundary-rank-loss-margin <f> Boundary rank loss hinge margin (default: 1.0)
        \\  --boundary-rank-loss-top-k <n> Average top-k violating negatives per gold boundary (default: 1)
        \\  --boundary-same-token-rank-loss-weight <f> Extra rank loss against negatives with same token id as gold boundaries (default: 0.0, disabled)
        \\  --boundary-same-token-rank-loss-top-k <n> Same-token hard negatives per gold boundary (default: 4)
        \\  --boundary-same-token-negative-weight <f> CE negative-class multiplier for non-boundaries sharing a token id with a gold boundary in the batch (default: 1.0, disabled)
        \\  --boundary-same-token-negative-top-k <n> Limit same-token CE multiplier to top-k highest-margin same-token negatives (default: 0, all)
        \\  --boundary-candidate-rank-loss-weight <f> Extra rank loss against sentence-like non-boundary candidates from token offsets (default: 0.0, disabled)
        \\  --boundary-candidate-rank-loss-top-k <n> Sentence-like hard negatives per gold boundary (default: 8)
        \\  --boundary-candidate-negative-weight <f> CE negative-class multiplier for sentence-like non-boundary candidates from token offsets (default: 1.0, disabled)
        \\  --boundary-gold-count-rank-loss-weight <f> Sample-local rank loss against gold-count-scaled hard-negative endpoints (default: 0.0, disabled)
        \\  --boundary-gold-count-rank-loss-margin <f> Gold-count rank loss hinge margin (default: 1.0)
        \\  --boundary-gold-count-rank-loss-negative-multiplier <n> Hard negatives per sequence = gold_count * n (default: 1)
        \\  --boundary-local-window-loss-weight <f> Local softmax exact-boundary loss weight (default: 0.0, disabled)
        \\  --boundary-local-window-radius <n> Tokens on each side for local boundary loss (default: 12)
        \\  --boundary-feature-mode token|prev-diff|prev-current-diff|prev-current-diff-concat|prev-current-next-diff-concat|window-context-diff Boundary head input feature view (default: token)
        \\  --boundary-dropout <f>    Boundary head dropout rate (default: 0.1)
        \\  --hidden-size <n>         Encoder hidden size (default: 768)
        \\  --num-layers <n>          ModernBERT layer count (default: 22)
        \\  --max-layers <n>          Alias for --num-layers
        \\  --max-seq-len <n>         Max token sequence length (default: 384)
        \\  --max-chunks <n>          Max chunks per sample (default: 32)
        \\  --checkpoint-every <n>    Save checkpoint every N epochs (0=disabled)
        \\  --checkpoint-every-steps <n> Save checkpoint every N global steps (0=disabled)
        \\  --log-every <n>           Print every N steps (default: 1, 0=epoch summaries only)
        \\  --eval-every <n>          Validate every N epochs when --val-data is set (default: 1, 0=disabled)
        \\  --eval-every-steps <n>    Validate every N global steps when --val-data is set (0=disabled)
        \\  --step-eval-max-examples <n> Cap step-triggered validation examples only (0=full validation)
        \\  --step-train-eval-max-examples <n> Also evaluate first N train examples at each step validation gate (0=disabled)
        \\  --checkpoint-roundtrip-eval Reload final checkpoint in-process and re-evaluate validation slice before exit
        \\  --checkpoint-roundtrip-max-examples <n> Cap same-process roundtrip validation examples (0=full validation)
        \\  --max-steps <n>           Stop after global training step N (0=disabled)
        \\  --split <name>            Dataset split name filter (default: "train")
        \\  --val-split <name>        Validation split name filter (default: "val")
        \\  --seed <n>                Random seed (default: 42)
        \\  --lora-rank <n>           LoRA rank (default: 0 = disabled)
        \\  --lora-alpha <f>          LoRA alpha scaling (default: 32.0)
        \\  --lora-start-epoch <n>    Freeze LoRA updates until epoch N, 0-indexed (default: 0)
        \\  --intermediate-size <n>   ModernBERT intermediate_size (default: 1152)
        \\  --backend native|metal|auto Compute backend (default: auto; auto prefers Metal)
        \\  --grad-accum <n>          Gradient accumulation steps (default: 1)
        \\  --optimizer adamw|schedule-free Optimizer selection (default: adamw)
        \\  --schedule-free           Use Schedule-Free AdamW
        \\  --neftune-alpha <f>       NEFTune noise magnitude (default: 0.0=disabled)
        \\  --encoder-neftune         Apply NEFTune to encoder forwards when alpha > 0 (default)
        \\  --disable-encoder-neftune Keep configured NEFTune alpha but disable encoder noise
        \\  --xbm-capacity <n>        Cross-Batch Memory capacity (default: 0=disabled)
        \\  --llrd-decay <f>          Layer-wise LR decay (default: 1.0=disabled)
        \\  --lisa-sample <n>         LISA random layers per step (default: 0=disabled)
        \\  --lisa-top-k <n>          LISA top layers always active (default: 5)
        \\  --lora-train-top-k <n>    Train only the top N LoRA layers (default: 0=all/LISA)
        \\  --encoder-vjp direct|last-layer|full|full-hidden-direct Encoder LoRA backward mode (default: direct)
        \\  --layers-per-segment <n>  Full-VJP encoder layers per reverse segment (default: 1)
        \\  --compiled-segment-forward Run the LoRA encoder forward through compiled MPSGraph segment sessions (default: off; env ANTFLY_FUSED_CHUNKER_COMPILED_SEGMENT_FORWARD=1)
        \\  --lora-plus-ratio <f>     LoRA+ B/A LR ratio (default: 1.0=disabled)
        \\  --length-bucketing        Enable length bucketing
        \\  --bucket-size <n>         Bucket window size (default: 256)
        \\  --mixed-precision         Accepted for compatibility; ignored by direct Metal trainer
        \\  --splade                  Enable SPLADE sparse embedding head
        \\  --lambda-splade <f>       SPLADE contrastive loss weight (default: 0.15)
        \\  --lambda-flops <f>        SPLADE FLOPS regularization weight (default: 3e-5)
        \\  --splade-focus-epoch <n>  Epoch when SPLADE activates (default: 4)
        \\  --mrl                     Enable Matryoshka Representation Learning
        \\  --mrl-dims <s>            Comma-separated MRL dims (default: "768,256,128")
        \\  --resume-from <path>      Resume training from a checkpoint file
        \\  --save-optimizer-state    Save Adam optimizer state alongside each checkpoint
        \\  --deterministic           Disable shuffle, boundary dropout, and NEFTune for parity probes
        \\  --go-epoch-shuffle        Reset to identity and reseed shuffle as seed+epoch+1 to match Go training epoch order
        \\  --report-to <path>        Write fused training metrics JSONL (default: <output>/fused_training_metrics.jsonl)
        \\  --manifest <path>         Write fused training manifest JSON (default: <output>/fused_training_manifest.json)
        \\  --boundary-alignment-dump-dir <dir> Write per-eval JSONL token alignment dumps for gold and high-probability boundary tokens
        \\  --boundary-alignment-dump-top-k <n>  High-probability tokens to dump per eval batch (default: 32)
        \\  --memory-sample-every <n> Sample process RSS every N steps (default: 1)
        \\  --memory-warn-rss-gb <f>  Print a memory watchdog warning above this RSS in GiB
        \\  --memory-abort-rss-gb <f> Save checkpoint and abort above this RSS in GiB
        \\  --debug-first-boundary-step Print first-batch boundary diagnostics before the first update
        \\  --debug-first-boundary-step-exit Print first-batch diagnostics and exit before the first update
        \\  --debug-boundary-step <n> Print boundary diagnostics before global training step N
        \\  --debug-boundary-step-exit Exit after --debug-boundary-step diagnostics
        \\  --debug-update-step <n>   Print LoRA gradient clip, AdamW moment, and update-delta diagnostics after global training step N
        \\  --debug-update-step-exit  Exit after --debug-update-step diagnostics
        \\  --debug-boundary-head-overfit-steps <n> Train only the boundary head on the first frozen encoded batch for N steps, then exit
        \\  --debug-boundary-head-overfit-lr <f> Constant LR for --debug-boundary-head-overfit-steps (default: 1e-3)
        \\  --debug-frozen-feature-probe Encode bounded train/val slices once, train only the boundary head on cached features, then exit
        \\  --debug-frozen-feature-train-examples <n> Train examples to cache for --debug-frozen-feature-probe (default: 64)
        \\  --debug-frozen-feature-val-examples <n> Validation examples to cache for --debug-frozen-feature-probe (default: 64)
        \\  --debug-frozen-feature-train-offset <n> First train example index for cached feature probe (default: 0)
        \\  --debug-frozen-feature-val-offset <n> First validation example index for cached feature probe (default: 0)
        \\  --debug-frozen-feature-epochs <n> Boundary-head passes over cached train features (default: 3)
        \\  --debug-frozen-feature-lr <f> Constant LR for cached boundary-head probe (default: 5e-3)
        \\  --debug-step-json <path>  Write machine-readable frozen-step parity diagnostics for the selected debug step
        \\  --debug-batch-offset <n>  Override resumed batch offset for frozen-step parity diagnostics
        \\  --debug-encoder-probe-layer <n> Capture target encoder layer internals in frozen-step parity JSON (default: 0)
        \\  --debug-encoder-layer-inputs-only Capture the encoder layer-input ladder without target-layer internals
        \\  --debug-encoder-replay-input <path> Write target encoder layer input as little-endian f32 for same-input replay
        \\  --debug-encoder-replay-upstream <path> Write target encoder upstream gradient as little-endian f32 for same-upstream replay
        \\  --debug-layer-backward-decomp Emit opt-in layer backward decomposition probe in frozen-step parity JSON
        \\  --debug-qkv-split-vjp     Emit opt-in QKV split VJP primitive probe in frozen-step parity JSON
        \\
    , .{});
}
