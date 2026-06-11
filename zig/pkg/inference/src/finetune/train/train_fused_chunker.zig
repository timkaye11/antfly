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
const mlx_mod = if (build_options.enable_mlx) @import("../../backends/mlx.zig") else struct {
    pub const c = struct {
        pub fn mlx_map_string_to_array_new() void {}
        pub fn mlx_map_string_to_array_free(_: void) void {}
        pub const mlx_array = void;
        pub const mlx_map_string_to_array = void;
    };
    pub fn openDefaultStream() struct { stream: void } {
        return .{ .stream = {} };
    }
};
const fused_chunker_train = @import("../fused_chunker_train.zig");
const fused_chunker_data = @import("../fused_chunker_data.zig");
const fused_chunker_mod = @import("../fused_chunker.zig");
const fused_chunker_splade = @import("../fused_chunker_splade.zig");
const fused_chunker_loss = @import("../fused_chunker_loss.zig");
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
const lora = @import("../lora.zig");
const segmented_encoder = @import("../../graph/segmented_encoder.zig");
const graph_training = @import("../../graph/training.zig");
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
    boundary_pos_weight: f32 = 5.0,
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
    // Feature 4: LoRA+
    lora_plus_ratio: f32 = 1.0,
    // Feature 8: length bucketing
    length_bucketing: bool = false,
    bucket_size: usize = 256,
    // mixed precision flag (stored for downstream use)
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
    debug_first_boundary_step: bool = false,
    debug_first_boundary_step_exit: bool = false,
    debug_boundary_step: u32 = 0,
    debug_boundary_step_exit: bool = false,
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

fn isFiniteTrainStepSummary(summary: TrainStepSummary) bool {
    return isFiniteF32(summary.boundary_loss) and
        isFiniteF64(summary.contrastive_loss) and
        isFiniteF32(summary.total_loss) and
        isFiniteF32(summary.learning_rate);
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

fn sanitizeAndClipLoRAGrads(layers: []fused_chunker_lora.LoRALayer, max_norm: f32) void {
    var total_sq: f64 = 0;
    for (layers) |*ll| {
        for (ll.grad_A) |*g| {
            if (!isFiniteF32(g.*)) {
                g.* = 0;
                continue;
            }
            total_sq += @as(f64, g.*) * @as(f64, g.*);
        }
        for (ll.grad_B) |*g| {
            if (!isFiniteF32(g.*)) {
                g.* = 0;
                continue;
            }
            total_sq += @as(f64, g.*) * @as(f64, g.*);
        }
        if (ll.grad_magnitude) |gm| {
            for (gm) |*g| {
                if (!isFiniteF32(g.*)) {
                    g.* = 0;
                    continue;
                }
                total_sq += @as(f64, g.*) * @as(f64, g.*);
            }
        }
    }

    if (max_norm <= 0) return;

    const total_norm: f32 = @floatCast(@sqrt(total_sq));
    if (!isFiniteF32(total_norm) or total_norm <= max_norm) return;

    const scale = max_norm / (total_norm + 1e-6);
    for (layers) |*ll| {
        for (ll.grad_A) |*g| g.* *= scale;
        for (ll.grad_B) |*g| g.* *= scale;
        if (ll.grad_magnitude) |gm| {
            for (gm) |*g| g.* *= scale;
        }
    }
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

    var partitioned_session = try segmented_encoder.ModernBertSegmentVJPSession.initWithGradientOptionsAndStrategy(
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
        .partitioned_required,
    );
    defer partitioned_session.deinit();

    const mps_layers = try cloneLoRALayersForSegmentVJPParity(allocator, lora_layers);
    defer deinitClonedLoRALayers(mps_layers, allocator);
    const partitioned_layers = try cloneLoRALayersForSegmentVJPParity(allocator, lora_layers);
    defer deinitClonedLoRALayers(partitioned_layers, allocator);

    var mps_result = try mps_session.executeWithOptions(
        cb,
        segment_input,
        upstream_grad,
        attention_mask,
        mps_layers,
        include_adapter_grads,
    );
    defer mps_result.deinit();

    var partitioned_result = try partitioned_session.executeWithOptions(
        cb,
        segment_input,
        upstream_grad,
        attention_mask,
        partitioned_layers,
        include_adapter_grads,
    );
    defer partitioned_result.deinit();

    var hidden_stats = SegmentVJPDiffStats{};
    if (include_hidden_grad) {
        const mps_hidden = mps_result.hidden_grad orelse return error.SegmentVJPParityMissingMpsHiddenGrad;
        const partitioned_hidden = partitioned_result.hidden_grad orelse return error.SegmentVJPParityMissingPartitionedHiddenGrad;
        try hidden_stats.add("hidden_grad", mps_hidden, partitioned_hidden);
    }

    var adapter_a_stats = SegmentVJPDiffStats{};
    var adapter_b_stats = SegmentVJPDiffStats{};
    if (include_adapter_grads) {
        for (mps_layers) |*mps_layer| {
            if (mps_layer.layer_idx < segment_start or mps_layer.layer_idx >= segment_end) continue;
            const partitioned_layer = findLoRALayerForSegmentVJPParity(
                partitioned_layers,
                mps_layer.layer_idx,
                mps_layer.module_name,
            ) orelse return error.SegmentVJPParityMissingAdapterLayer;
            try adapter_a_stats.add(mps_layer.module_name, mps_layer.grad_A, partitioned_layer.grad_A);
            try adapter_b_stats.add(mps_layer.module_name, mps_layer.grad_B, partitioned_layer.grad_B);
        }
    }

    print(
        "segment_vjp_parity segment={d}-{d} hidden_tensors={d} hidden_elems={d} hidden_max_abs={d:.9} hidden_mean_abs={d:.9} hidden_rel_l2={d:.9} hidden_cos={d:.9} hidden_sign_mismatch={d} hidden_max={s} adapter_a_tensors={d} adapter_a_elems={d} adapter_a_max_abs={d:.9} adapter_a_mean_abs={d:.9} adapter_a_rel_l2={d:.9} adapter_a_cos={d:.9} adapter_a_sign_mismatch={d} adapter_a_max={s} adapter_b_tensors={d} adapter_b_elems={d} adapter_b_max_abs={d:.9} adapter_b_mean_abs={d:.9} adapter_b_rel_l2={d:.9} adapter_b_cos={d:.9} adapter_b_sign_mismatch={d} adapter_b_max={s} mps_exec_ms={d:.2} partitioned_exec_ms={d:.2}\n",
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
            @as(f64, @floatFromInt(mps_result.profile.compiled_execute_ns)) / 1_000_000.0,
            @as(f64, @floatFromInt(partitioned_result.profile.compiled_execute_ns)) / 1_000_000.0,
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
    print("phase20_parity_hparams epochs={d} batch_size={d} max_seq_len={d} max_chunks={d} lr={d} warmup_steps={d} lr_total_steps={d} weight_decay={d} max_grad_norm={d}\n", .{
        opts.epochs,
        opts.batch_size,
        opts.max_seq_len,
        opts.max_chunks,
        opts.learning_rate,
        opts.warmup_steps,
        opts.lr_total_steps,
        opts.weight_decay,
        opts.max_grad_norm,
    });
    print("phase20_parity_lora rank={d} alpha={d} go_targets=query_proj,value_proj,key_proj,Wo zig_targets=query_proj,key_proj,value_proj,out_proj,wo\n", .{
        opts.lora_rank,
        opts.lora_alpha,
    });
    print("phase20_parity_loss lambda_chunk={d} lambda_embed={d} boundary_focus_epochs={d} boundary_focus_lambda_embed={d} boundary_dropout={d} neftune_alpha={d} mrl={s} mrl_dims={s} loss_type={s} pos_weight={d}\n", .{
        opts.lambda_chunk,
        opts.lambda_embed,
        opts.boundary_focus_epochs,
        opts.boundary_focus_lambda_embed,
        opts.boundary_dropout,
        opts.neftune_alpha,
        enabledName(opts.mrl),
        opts.mrl_dims_str,
        @tagName(opts.boundary_loss_type),
        opts.boundary_pos_weight,
    });
    print("phase20_parity_note go_cli_default_pos_weight=5.0 phase20_best_pos_weight=1.0\n", .{});
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

const BoundaryBatchDebugStats = struct {
    valid_tokens: u64 = 0,
    gold_positives: u64 = 0,
    feature_mean: f32 = 0,
    feature_rms: f32 = 0,
    feature_max_abs: f32 = 0,
};

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

fn printIndexSlice(label: []const u8, values: []const usize) void {
    print("{s}=[", .{label});
    for (values, 0..) |value, i| {
        if (i > 0) print(",", .{});
        print("{d}", .{value});
    }
    print("]\n", .{});
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
    var boundary_pos_weight: f32 = 5.0;
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
    var xbm_capacity: usize = 0;
    var llrd_decay: f32 = 1.0;
    var lisa_sample_layers: u32 = 0;
    var lisa_top_k: u32 = 5;
    var lora_train_top_k: u32 = 0;
    var encoder_vjp: EncoderVJPMode = .direct;
    var layers_per_segment: u32 = 1;
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
    var debug_first_boundary_step: bool = false;
    var debug_first_boundary_step_exit: bool = false;
    var debug_boundary_step: u32 = 0;
    var debug_boundary_step_exit: bool = false;

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
        } else if (std.mem.eql(u8, arg, "--boundary-pos-weight") or std.mem.eql(u8, arg, "--pos-weight")) {
            const val = args.next() orelse return error.MissingBoundaryPosWeight;
            boundary_pos_weight = try std.fmt.parseFloat(f32, val);
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
            if (std.mem.eql(u8, val, "native") or std.mem.eql(u8, val, "blas")) {
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
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            return;
        } else {
            print("unknown argument: {s}\n", .{arg});
            printUsage();
            std.process.exit(1);
        }
    }

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
        .boundary_pos_weight = boundary_pos_weight,
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
        .xbm_capacity = xbm_capacity,
        .llrd_decay = llrd_decay,
        .lisa_sample_layers = lisa_sample_layers,
        .lisa_top_k = lisa_top_k,
        .lora_train_top_k = lora_train_top_k,
        .encoder_vjp = encoder_vjp,
        .layers_per_segment = layers_per_segment,
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
        .debug_first_boundary_step = debug_first_boundary_step,
        .debug_first_boundary_step_exit = debug_first_boundary_step_exit,
        .debug_boundary_step = debug_boundary_step,
        .debug_boundary_step_exit = debug_boundary_step_exit,
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

/// Insert (or update) a LoRA matrix in the BLAS WeightStore under `key`.
/// The tensor is a 2-D f32 matrix of shape [rows, cols].
/// If a weight already exists under `key` its data is replaced with a fresh
/// copy of `data` so that optimizer updates are visible each step.
fn insertLoRAIntoBlasStore(
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

fn savePeriodicTrainingCheckpoint(
    allocator: std.mem.Allocator,
    opts: *const Options,
    trainer: *FusedTrainer,
    lora_adapters: ?*const fused_chunker_lora.LoRAAdapterSet,
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
        try trainer.saveOptimizerState(allocator, opt_path);
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
            try insertLoRAIntoBlasStore(allocator, native_weight_store, key_a, ll.A, rank, ll.in_features);
            try insertLoRAIntoBlasStore(allocator, native_weight_store, key_b, ll.B, ll.out_features, rank);
        }
    }
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
    if (comptime build_options.enable_mlx) {
        _ = mlx_mod.c.mlx_map_string_to_array_free(weight_store.resident_weights);
    }
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
) !fused_chunker_train.EvalSummary {
    if (encoder_loaded) {
        if (lora_adapters) |la| {
            try refreshLoRAWeightsForForward(allocator, use_metal, native_weight_store, metal_weight_store, la);
        }
    }

    var acc = fused_chunker_train.BoundaryEvalAccumulator{};
    const eval_start_ns = nowNs();
    const bs: usize = @intCast(opts.batch_size);
    const max_seq: usize = @intCast(opts.max_seq_len);
    const max_chunks: usize = @intCast(opts.max_chunks);
    const hidden_size: usize = @intCast(opts.hidden_size);
    const total_batches = (samples.len + bs - 1) / bs;

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

        try trainer.evaluateBatchInto(allocator, &acc, features, labels, mask, total_tokens);

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

    return acc.finish();
}

// ---------------------------------------------------------------------------
// Core training routine
// ---------------------------------------------------------------------------

fn run(allocator: std.mem.Allocator, opts: Options) !void {
    // ------------------------------------------------------------------
    // 1. Create output directory
    // ------------------------------------------------------------------
    try compat.cwd().createDirPath(compat.io(), opts.output_dir);

    print("train-fused-chunker data={s} output={s} epochs={d} batch_size={d} lr={d} warmup_steps={d} lr_total_steps={d} weight_decay={d} beta1={d} beta2={d} adam_epsilon={d} hidden={d} layers={d} max_seq_len={d} max_chunks={d} loss_type={s} pos_weight={d} boundary_dropout={d} lambda_embed={d} boundary_focus_epochs={d} boundary_focus_lambda_embed={d} optimizer={s} log_every={d} checkpoint_every_steps={d} eval_every_steps={d} seed={d} lisa_sample={d} lisa_top_k={d} lora_train_top_k={d} lora_start_epoch={d} encoder_vjp={s} layers_per_segment={d}\n", .{
        opts.data_path,
        opts.output_dir,
        opts.epochs,
        opts.batch_size,
        opts.learning_rate,
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
        @tagName(opts.boundary_loss_type),
        opts.boundary_pos_weight,
        opts.boundary_dropout,
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
    print("contrastive_grad_path={s}\n", .{fused_chunker_train.contrastive_gradient_path});
    print("step_eval_max_examples={d}\n", .{opts.step_eval_max_examples});
    print("max_steps={d}\n", .{opts.max_steps});
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
    var blas_backend = native_compute.NativeCompute.init(allocator, &weight_store, null);

    var metal_weight_store: MetalWeightStore = undefined;
    var metal_backend: if (build_options.enable_metal) metal_compute.MetalCompute else void = undefined;

    const cb: ComputeBackend = if (use_metal) blk: {
        if (comptime build_options.enable_metal) {
            metal_weight_store = MetalWeightStore{
                .allocator = allocator,
                .resident_weights = if (comptime build_options.enable_mlx) mlx_mod.c.mlx_map_string_to_array_new() else {},
                .stream = if (comptime build_options.enable_mlx) mlx_mod.openDefaultStream().stream else {},
                .prefix = "",
                .lazy_weights = .{},
            };
            metal_compute.initPrefetchQueue(&metal_weight_store, allocator);
            metal_backend = try metal_compute.MetalCompute.init(allocator, &metal_weight_store, null);
            break :blk metal_backend.computeBackend();
        } else unreachable;
    } else blas_backend.computeBackend();
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
                                weight_store.resident_weights.put(allocator, owned_name, lw) catch {
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
    if (usesFullEncoderVJP(opts.encoder_vjp) and opts.lora_rank > 0) {
        const sessions = try allocator.alloc(?segmented_encoder.ModernBertSegmentVJPSession, opts.num_layers);
        errdefer allocator.free(sessions);
        for (sessions) |*slot| slot.* = null;
        full_vjp_sessions = sessions;
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

    const stats = fused_chunker_data.computeStats(samples);
    print("loaded {d} samples  avg_chars={d:.0}  avg_chunks={d:.1}  min_chunks={d}  max_chunks={d}  contrastive_pos_samples={d}  boundary_target_samples={d}  boundary_targets={d}\n", .{
        stats.num_samples,
        stats.avg_text_chars,
        stats.avg_chunks_per_sample,
        stats.min_chunks,
        stats.max_chunks,
        stats.samples_with_contrastive_positives,
        stats.samples_with_boundary_targets,
        stats.total_boundary_targets,
    });

    var val_loaded_opt: ?fused_chunker_data.LoadedSamples = null;
    defer if (val_loaded_opt) |*val_loaded| val_loaded.deinit();
    var val_samples: []const fused_chunker_data.FusedSample = &.{};
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
        .pos_weight = opts.boundary_pos_weight,
        .boundary_dropout = opts.boundary_dropout,
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
        .neftune_alpha = opts.neftune_alpha,
        // Feature 6: LLRD
        .llrd_decay = opts.llrd_decay,
        // Feature 8: length bucketing
        .length_bucketing = opts.length_bucketing,
        .bucket_size = opts.bucket_size,
        // mixed precision
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

    print("trainer ready  boundary_head hidden={d} mlp_dim={d}\n", .{
        config.hidden_size,
        config.boundary_mlp_dim,
    });

    // ------------------------------------------------------------------
    // 4a. Resume from checkpoint (if --resume-from was supplied)
    // ------------------------------------------------------------------
    if (opts.resume_from.len > 0) {
        print("resuming weights from {s}\n", .{opts.resume_from});
        try trainer.loadCheckpoint(allocator, opts.resume_from);

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
                trainer.loadOptimizerState(allocator, p) catch |err| {
                    print("warning: could not load optimizer state from {s}: {}\n", .{ p, err });
                };
            }
        }
    }

    // ------------------------------------------------------------------
    // 5. Training loop
    // ------------------------------------------------------------------

    // Build a mutable index array for shuffling
    var indices = try allocator.alloc(usize, samples.len);
    defer allocator.free(indices);
    for (indices, 0..) |*idx, i| idx.* = i;

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
    var best_val_f1: f32 = -std.math.inf(f32);
    var best_val_epoch: u32 = 0;
    var best_val_step: u32 = 0;

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

    var stop_training = false;
    for (0..opts.epochs) |epoch| {
        const epoch_lambda_embed: f32 = if (epoch < opts.boundary_focus_epochs)
            opts.boundary_focus_lambda_embed
        else
            opts.lambda_embed;
        trainer.loss_config.lambda_embed = epoch_lambda_embed;
        print("epoch {d}/{d} curriculum lambda_chunk={d} lambda_embed={d}\n", .{
            epoch + 1,
            opts.epochs,
            trainer.loss_config.lambda_chunk,
            trainer.loss_config.lambda_embed,
        });

        // Shuffle indices using Fisher-Yates
        var i: usize = indices.len;
        while (i > 1) {
            i -= 1;
            const j = rng.uintLessThan(usize, i + 1);
            const tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
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
        var batch_start: usize = if (epoch == resume_epoch) resume_batch_offset else 0;
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
                const neftune_seed = if (opts.neftune_alpha > 0.0) blk: {
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
                    .neftune_alpha = opts.neftune_alpha,
                    .neftune_seed = neftune_seed,
                };

                if (lora_training_active) {
                    var act_buf = modern_bert.ActivationBuffer.init(allocator);
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
                        opts.boundary_dropout,
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
                const debug = try trainer.debugBoundaryStep(
                    allocator,
                    features,
                    boundary_labels_2,
                    attn_mask_f32,
                    total_tokens,
                    true,
                );
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
                        opts.boundary_dropout,
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
                if ((debug_first_boundary_step_now and opts.debug_first_boundary_step_exit) or
                    (debug_selected_boundary_step_now and opts.debug_boundary_step_exit))
                {
                    print("{s} exiting before optimizer update\n", .{debug_prefix});
                    return;
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
                    const hard_neg_neftune_seed = if (opts.neftune_alpha > 0.0) blk: {
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
                        .neftune_alpha = opts.neftune_alpha,
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

                            var layer_cursor: u32 = la.config.num_layers;
                            while (layer_cursor > 0) {
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
                                var segment_has_active_lora = false;
                                var active_layer_idx = segment_start;
                                while (active_layer_idx < segment_end) : (active_layer_idx += 1) {
                                    if (fused_chunker_train.isLoRALayerActive(active_layers, active_layer_idx)) {
                                        segment_has_active_lora = true;
                                        break;
                                    }
                                }
                                const include_hidden_grad = segment_start > 0;
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
                                    if (segment_vjp_parity_pending) {
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
                                    var vjp_result = try session.executeWithOptions(
                                        &cb,
                                        segment_input.input,
                                        upstream_grad,
                                        attn_mask_f32,
                                        la.layers,
                                        include_adapter_grads,
                                    );
                                    aggregate_profile.add(vjp_result.profile);
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
                        sanitizeAndClipLoRAGrads(la.layers, opts.max_grad_norm);

                        // Apply optimizer steps for all LoRA parameters.
                        // Feature 4 (LoRA+): use lr * lora_plus_ratio for lora_B.
                        // Feature 6 (LLRD): use per-layer decayed learning rate.
                        const base_lr = summary.learning_rate;
                        lora_opt_state.step_count = @intCast(summary.step);
                        const num_layers_f: f32 = @floatFromInt(la.config.num_layers);
                        for (la.layers) |*ll| {
                            if (!fused_chunker_train.isLoRALayerActive(active_layers, ll.layer_idx)) continue;
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
                            try optimizers.step(trainer.optimizer, &lora_opt_state, layer_lr, a_key, ll.A, ll.grad_A);
                            try optimizers.step(trainer.optimizer, &lora_opt_state, b_lr, b_key, ll.B, ll.grad_B);
                        }
                        const repaired = sanitizeLoRAAdapterSet(la) + optimizers.sanitizeState(&lora_opt_state);
                        if (repaired > 0) {
                            print("warning: sanitized {d} nonfinite LoRA adapter/optimizer values after update at global_step={d}\n", .{
                                repaired,
                                summary.step,
                            });
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

            if (opts.log_every > 0 and (step == 1 or step % opts.log_every == 0)) {
                print(
                    "epoch {d}/{d} step {d} | loss {d:.4} | boundary {d:.4} | contrastive {d:.4} | lr {d} | step_ms {d:.2} | batch {d:.2} | refresh {d:.2} | enc {d:.2} | hn {d:.2} | train {d:.2} | lora {d:.2} | splade {d:.2} | ex/s {d:.1}\n",
                    .{
                        epoch + 1,
                        opts.epochs,
                        summary.step,
                        summary.total_loss,
                        summary.boundary_loss,
                        summary.contrastive_loss,
                        summary.learning_rate,
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
                    splade_w,
                    splade_vocab_size,
                    "step",
                    summary.step,
                );
            }

            if (val_samples.len > 0 and opts.eval_every_steps > 0 and summary.step > 0 and summary.step % opts.eval_every_steps == 0) {
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
                );
                print(
                    "validation step {d} epoch {d}/{d} samples={d}/{d} | f1 {d:.4} precision {d:.4} recall {d:.4} counts tp={d} fp={d} fn={d} | best_f1 {d:.4} threshold {d:.2} best_counts tp={d} fp={d} fn={d} | valid={d} gold_pos={d} gold_rate={d:.6} pred_rate={d:.6} best_pred_rate={d:.6} prob_pos={d:.4} prob_neg={d:.4} | eval_ms {d:.2}\n",
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
                        val_summary.valid_tokens,
                        val_summary.gold_positives,
                        val_summary.gold_positive_rate,
                        val_summary.predicted_positive_rate,
                        val_summary.best_predicted_positive_rate,
                        val_summary.mean_positive_probability_gold_positive,
                        val_summary.mean_positive_probability_gold_negative,
                        nsToMs(elapsedNs(val_eval_start_ns)),
                    },
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
            );
            print(
                "validation epoch {d}/{d} | f1 {d:.4} precision {d:.4} recall {d:.4} counts tp={d} fp={d} fn={d} | best_f1 {d:.4} threshold {d:.2} best_counts tp={d} fp={d} fn={d} | valid={d} gold_pos={d} gold_rate={d:.6} pred_rate={d:.6} best_pred_rate={d:.6} prob_pos={d:.4} prob_neg={d:.4} | eval_ms {d:.2}\n",
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
                    val_summary.valid_tokens,
                    val_summary.gold_positives,
                    val_summary.gold_positive_rate,
                    val_summary.predicted_positive_rate,
                    val_summary.best_predicted_positive_rate,
                    val_summary.mean_positive_probability_gold_positive,
                    val_summary.mean_positive_probability_gold_negative,
                    nsToMs(elapsedNs(val_eval_start_ns)),
                },
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
        try trainer.saveOptimizerState(allocator, opt_final_path);
        print("optimizer state saved to {s}\n", .{opt_final_path});
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
        \\  --boundary-pos-weight <f> Positive class weight for CE loss (default: 5.0)
        \\  --pos-weight <f>          Alias for --boundary-pos-weight
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
        \\  --xbm-capacity <n>        Cross-Batch Memory capacity (default: 0=disabled)
        \\  --llrd-decay <f>          Layer-wise LR decay (default: 1.0=disabled)
        \\  --lisa-sample <n>         LISA random layers per step (default: 0=disabled)
        \\  --lisa-top-k <n>          LISA top layers always active (default: 5)
        \\  --lora-train-top-k <n>    Train only the top N LoRA layers (default: 0=all/LISA)
        \\  --encoder-vjp direct|last-layer|full|full-hidden-direct Encoder LoRA backward mode (default: direct)
        \\  --layers-per-segment <n>  Full-VJP encoder layers per reverse segment (default: 1)
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
        \\  --debug-first-boundary-step Print first-batch boundary diagnostics before the first update
        \\  --debug-first-boundary-step-exit Print first-batch diagnostics and exit before the first update
        \\  --debug-boundary-step <n> Print boundary diagnostics before global training step N
        \\  --debug-boundary-step-exit Exit after --debug-boundary-step diagnostics
        \\
    , .{});
}
