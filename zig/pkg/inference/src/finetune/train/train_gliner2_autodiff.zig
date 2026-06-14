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

// CLI training driver for GLiNER2 (DeBERTa-based NER) via the level-3
// real-autodiff path.
//
// Wires: CLI arg parsing → DeBERTa config loading → safetensors weight
// loading → JSONL data ingestion → GlinerAutodiffCtx + RealAutodiffTrainer
// step loop → LoRA adapter saving.
//
// Usage:
//   train-gliner2-autodiff --model-dir <path> --train-data <path> --out-dir <path> [options]
//
// Options:
//   --model-dir <path>           Directory with DeBERTa model (config.json + model.safetensors + tokenizer.json)
//   --train-data <path>          JSONL training data (file or directory)
//   --out-dir <path>             Output directory for saved LoRA adapters
//   --epochs <n>                 Number of training epochs (default: 10)
//   --batch-size <n>             Examples per step (default: 16)
//   --seq-len <n>                Max sequence length (default: 256)
//   --learning-rate <f>          Learning rate (default: 5e-4)
//   --weight-decay <f>           AdamW weight decay (default: 0)
//   --lora-rank <n>              LoRA rank (default: 16)
//   --lora-alpha <f>             LoRA alpha scaling (default: 32)
//   --lora-dropout <f>           LoRA dropout probability (default: 0.1)
//   --lora-targets <csv>         Target module groups (default: upstream GLiNER2 LoRA groups)
//   --num-classes <n>            Entity classes including O (default: 5)
//   --objective <name>           token or span-start (default: token)
//   --max-span-width <n>         Max span width for span-start objective (default: 4)
//   --max-examples <n>           Cap on training examples (0 = all, default: 0)
//   --max-grad-norm <f>          Gradient clipping norm (default: 1.0)
//   --grad-accum <n>             Gradient accumulation steps (default: 1)
//   --seed <n>                   RNG seed (default: 42)
//   --initial-adapter-checkpoint <path>
//                                  Optional PEFT safetensors checkpoint used to seed LoRA weights
//   --lora-only-trainables       Freeze regular task-head params; train LoRA only

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const inference = @import("inference_internal");
const ml = @import("ml");
const native_compute = inference.native_compute.native;
const metal_compute = if (build_options.enable_metal) inference.native_compute.metal else struct {};
const gpu_hosted_store = inference.native_compute.gpu_hosted_store;
const metal_runtime = inference.metal_runtime;
const compat = inference.io.compat;
const weight_source_mod = inference.models.weight_source;
const safetensors = inference.models.safetensors;
const SafetensorsSource = weight_source_mod.SafetensorsSource;
const LoadedWeight = weight_source_mod.LoadedWeight;
const Tensor = inference.backends.Tensor;
const MetalWeightStore = if (build_options.enable_metal) gpu_hosted_store.WeightStore else void;

// MLX backend (Apple Silicon GPU acceleration).
const mlx_mod = inference.backends.mlx;
const mlx = if (build_options.enable_mlx) mlx_mod else struct {};
const mlx_compute = if (build_options.enable_mlx) inference.native_compute.mlx else struct {};
const mlx_c = if (build_options.enable_mlx) mlx_mod.c else struct {};

// Finetune module imports — accessed via the termite internal module tree.
const gliner2_data = inference.finetune.gliner2_data;
const gliner2_bundle = inference.finetune.gliner2;
const gliner2_autodiff = inference.finetune.gliner2_real_autodiff;
const real_autodiff = inference.finetune.real_autodiff_trainer;
const optimizers = ml.graph.optimizers;
const run_validation = inference.finetune.gliner2_run_validation;
const deberta_arch = inference.architectures.deberta;
const deberta_graph = inference.architectures.deberta_graph;

const print = std.debug.print;

// ---------------------------------------------------------------------------
// CLI options
// ---------------------------------------------------------------------------

const Options = struct {
    model_dir: []const u8,
    train_data: []const u8,
    out_dir: []const u8,
    epochs: u32 = 10,
    batch_size: u32 = 16,
    seq_len: u32 = 256,
    learning_rate: f32 = 5e-4,
    weight_decay: f32 = 0.0,
    lora_rank: u32 = 16,
    lora_alpha: f32 = 32.0,
    lora_dropout: f32 = gliner2_bundle.default_lora_dropout,
    lora_targets: []const u8 = "encoder,span_rep,classifier,count_embed,count_pred",
    num_classes: u32 = 5,
    entity_types_csv: ?[]const u8 = null,
    objective: gliner2_autodiff.GlinerObjective = .token,
    max_span_width: u32 = 4,
    span_loss: gliner2_autodiff.SpanStartLossKind = .bce,
    span_loss_reduction: gliner2_autodiff.SpanStartLossReduction = .mean,
    span_positive_weight: f32 = 32.0,
    span_label_positive_weights: ?[]const u8 = null,
    span_negative_weight: f32 = 1.0,
    span_hard_negative_weight: f32 = 1.0,
    span_negative_mask_rate: f32 = 0.0,
    max_examples: usize = 0,
    max_grad_norm: f32 = 1.0,
    grad_accum: u32 = 1,
    seed: u64 = 42,
    initial_adapter_checkpoint: ?[]const u8 = null,
    backend: Gliner2TrainBackend = .auto,
    compiled_required: bool = false,
    dump_span_parity: bool = false,
    dump_optimizer_parity: bool = false,
    lora_only_trainables: bool = false,
    deterministic: bool = false,
    eval_strategy: EvalStrategy = .epoch,
    eval_steps: u32 = 0,
    save_best: bool = false,
    report_to: ReportTo = .stdout,
    allow_large_memory: bool = false,
};

const Gliner2TrainBackend = enum {
    auto,
    metal,
    mlx,
    native,
};

const EvalStrategy = enum {
    epoch,
    steps,
    none,
};

const ReportTo = enum {
    stdout,
    jsonl,
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next(); // skip binary name

    // -- Parse CLI args ----------------------------------------------------
    var model_dir: ?[]const u8 = null;
    var train_data: ?[]const u8 = null;
    var out_dir: ?[]const u8 = null;
    var epochs: u32 = 10;
    var batch_size: u32 = 16;
    var seq_len: u32 = 256;
    var learning_rate: f32 = 5e-4;
    var weight_decay: f32 = 0.0;
    var lora_rank: u32 = 16;
    var lora_alpha: f32 = 32.0;
    var lora_dropout: f32 = gliner2_bundle.default_lora_dropout;
    var lora_targets: []const u8 = "encoder,span_rep,classifier,count_embed,count_pred";
    var num_classes: u32 = 5;
    var entity_types_csv: ?[]const u8 = null;
    var objective: gliner2_autodiff.GlinerObjective = .token;
    var max_span_width: u32 = 4;
    var span_loss: gliner2_autodiff.SpanStartLossKind = .bce;
    var span_loss_reduction: gliner2_autodiff.SpanStartLossReduction = .mean;
    var span_positive_weight: f32 = 32.0;
    var span_label_positive_weights: ?[]const u8 = null;
    var span_negative_weight: f32 = 1.0;
    var span_hard_negative_weight: f32 = 1.0;
    var span_negative_mask_rate: f32 = 0.0;
    var max_examples: usize = 0;
    var max_grad_norm: f32 = 1.0;
    var grad_accum: u32 = 1;
    var seed: u64 = 42;
    var initial_adapter_checkpoint: ?[]const u8 = null;
    var backend: Gliner2TrainBackend = .auto;
    var compiled_required: bool = false;
    var dump_span_parity: bool = false;
    var dump_optimizer_parity: bool = false;
    var lora_only_trainables: bool = false;
    var deterministic: bool = false;
    var eval_strategy: EvalStrategy = .epoch;
    var eval_steps: u32 = 0;
    var save_best: bool = false;
    var report_to: ReportTo = .stdout;
    var allow_large_memory: bool = false;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            return;
        } else if (std.mem.eql(u8, arg, "--model-dir")) {
            model_dir = args.next() orelse return error.MissingModelDir;
        } else if (std.mem.eql(u8, arg, "--train-data")) {
            train_data = args.next() orelse return error.MissingTrainData;
        } else if (std.mem.eql(u8, arg, "--out-dir")) {
            out_dir = args.next() orelse return error.MissingOutDir;
        } else if (std.mem.eql(u8, arg, "--epochs")) {
            const val = args.next() orelse return error.MissingEpochs;
            epochs = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--batch-size")) {
            const val = args.next() orelse return error.MissingBatchSize;
            batch_size = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--seq-len")) {
            const val = args.next() orelse return error.MissingSeqLen;
            seq_len = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--learning-rate") or std.mem.eql(u8, arg, "--lr")) {
            const val = args.next() orelse return error.MissingLearningRate;
            learning_rate = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--weight-decay")) {
            const val = args.next() orelse return error.MissingWeightDecay;
            weight_decay = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--lora-rank")) {
            const val = args.next() orelse return error.MissingLoraRank;
            lora_rank = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--lora-alpha")) {
            const val = args.next() orelse return error.MissingLoraAlpha;
            lora_alpha = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--lora-dropout")) {
            const val = args.next() orelse return error.MissingLoraDropout;
            lora_dropout = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--lora-targets")) {
            lora_targets = args.next() orelse return error.MissingLoraTargets;
        } else if (std.mem.eql(u8, arg, "--num-classes")) {
            const val = args.next() orelse return error.MissingNumClasses;
            num_classes = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--entity-types")) {
            entity_types_csv = args.next() orelse return error.MissingEntityTypes;
        } else if (std.mem.eql(u8, arg, "--objective")) {
            const val = args.next() orelse return error.MissingObjective;
            objective = try parseObjective(val);
        } else if (std.mem.eql(u8, arg, "--max-span-width")) {
            const val = args.next() orelse return error.MissingMaxSpanWidth;
            max_span_width = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--span-loss")) {
            const val = args.next() orelse return error.MissingSpanLoss;
            span_loss = try parseSpanLoss(val);
        } else if (std.mem.eql(u8, arg, "--span-loss-reduction")) {
            const val = args.next() orelse return error.MissingSpanLossReduction;
            span_loss_reduction = try parseSpanLossReduction(val);
        } else if (std.mem.eql(u8, arg, "--span-positive-weight")) {
            const val = args.next() orelse return error.MissingSpanPositiveWeight;
            span_positive_weight = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--span-label-positive-weights")) {
            span_label_positive_weights = args.next() orelse return error.MissingSpanLabelPositiveWeights;
        } else if (std.mem.eql(u8, arg, "--span-negative-weight")) {
            const val = args.next() orelse return error.MissingSpanNegativeWeight;
            span_negative_weight = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--span-hard-negative-weight")) {
            const val = args.next() orelse return error.MissingSpanHardNegativeWeight;
            span_hard_negative_weight = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--span-negative-mask-rate")) {
            const val = args.next() orelse return error.MissingSpanNegativeMaskRate;
            span_negative_mask_rate = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--max-examples")) {
            const val = args.next() orelse return error.MissingMaxExamples;
            max_examples = try std.fmt.parseUnsigned(usize, val, 10);
        } else if (std.mem.eql(u8, arg, "--max-grad-norm")) {
            const val = args.next() orelse return error.MissingMaxGradNorm;
            max_grad_norm = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--grad-accum")) {
            const val = args.next() orelse return error.MissingGradAccum;
            grad_accum = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            const val = args.next() orelse return error.MissingSeed;
            seed = try std.fmt.parseUnsigned(u64, val, 10);
        } else if (std.mem.eql(u8, arg, "--initial-adapter-checkpoint")) {
            initial_adapter_checkpoint = args.next() orelse return error.MissingInitialAdapterCheckpoint;
        } else if (std.mem.eql(u8, arg, "--backend")) {
            const val = args.next() orelse return error.MissingBackend;
            backend = parseBackend(val) orelse return error.InvalidBackend;
        } else if (std.mem.eql(u8, arg, "--compiled-required")) {
            compiled_required = true;
        } else if (std.mem.eql(u8, arg, "--dump-span-parity")) {
            dump_span_parity = true;
        } else if (std.mem.eql(u8, arg, "--dump-optimizer-parity")) {
            dump_optimizer_parity = true;
        } else if (std.mem.eql(u8, arg, "--lora-only-trainables")) {
            lora_only_trainables = true;
        } else if (std.mem.eql(u8, arg, "--deterministic")) {
            deterministic = true;
        } else if (std.mem.eql(u8, arg, "--eval-strategy")) {
            const val = args.next() orelse return error.MissingEvalStrategy;
            eval_strategy = parseEvalStrategy(val) orelse {
                print("error: unsupported --eval-strategy '{s}' (expected epoch, steps, or none)\n", .{val});
                return error.InvalidEvalStrategy;
            };
        } else if (std.mem.eql(u8, arg, "--eval-steps")) {
            const val = args.next() orelse return error.MissingEvalSteps;
            eval_steps = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--save-best")) {
            save_best = true;
        } else if (std.mem.eql(u8, arg, "--allow-large-memory")) {
            allow_large_memory = true;
        } else if (std.mem.eql(u8, arg, "--report-to")) {
            const val = args.next() orelse return error.MissingReportTo;
            report_to = parseReportTo(val) orelse {
                print("error: unsupported --report-to '{s}' (expected stdout or jsonl)\n", .{val});
                return error.InvalidReportTo;
            };
        } else {
            print("error: unknown argument: {s}\n", .{arg});
            printUsage();
            return error.InvalidArguments;
        }
    }

    if (deterministic) {
        // Force off every per-step stochastic regularizer on this training
        // path so repeated runs (and Python/Zig parity comparisons) are
        // bitwise-stable given the same seed and adapter init.
        //
        // NEFTune-style embedding noise is not wired into the GLiNER2
        // real-autodiff path, so LoRA dropout masks and span negative
        // masking are the only per-step stochastic inputs to disable here.
        if (lora_dropout != 0.0) {
            print("warning: --deterministic overrides --lora-dropout {d:.3} -> 0.0\n", .{lora_dropout});
            lora_dropout = 0.0;
        }
        if (span_negative_mask_rate != 0.0) {
            print("warning: --deterministic overrides --span-negative-mask-rate {d:.3} -> 0.0\n", .{span_negative_mask_rate});
            span_negative_mask_rate = 0.0;
        }
    }
    if (eval_strategy == .steps and eval_steps == 0) {
        print("error: --eval-strategy steps requires --eval-steps > 0\n", .{});
        return error.InvalidEvalSteps;
    }

    const opts = Options{
        .model_dir = model_dir orelse {
            print("error: --model-dir is required\n", .{});
            printUsage();
            return error.InvalidArguments;
        },
        .train_data = train_data orelse {
            print("error: --train-data is required\n", .{});
            printUsage();
            return error.InvalidArguments;
        },
        .out_dir = out_dir orelse {
            print("error: --out-dir is required\n", .{});
            printUsage();
            return error.InvalidArguments;
        },
        .epochs = epochs,
        .batch_size = batch_size,
        .seq_len = seq_len,
        .learning_rate = learning_rate,
        .weight_decay = weight_decay,
        .lora_rank = lora_rank,
        .lora_alpha = lora_alpha,
        .lora_dropout = lora_dropout,
        .lora_targets = lora_targets,
        .num_classes = num_classes,
        .entity_types_csv = entity_types_csv,
        .objective = objective,
        .max_span_width = max_span_width,
        .span_loss = span_loss,
        .span_loss_reduction = span_loss_reduction,
        .span_positive_weight = span_positive_weight,
        .span_label_positive_weights = span_label_positive_weights,
        .span_negative_weight = span_negative_weight,
        .span_hard_negative_weight = span_hard_negative_weight,
        .span_negative_mask_rate = span_negative_mask_rate,
        .max_examples = max_examples,
        .max_grad_norm = max_grad_norm,
        .grad_accum = grad_accum,
        .seed = seed,
        .initial_adapter_checkpoint = initial_adapter_checkpoint,
        .backend = backend,
        .compiled_required = compiled_required,
        .dump_span_parity = dump_span_parity,
        .dump_optimizer_parity = dump_optimizer_parity,
        .lora_only_trainables = lora_only_trainables,
        .deterministic = deterministic,
        .eval_strategy = eval_strategy,
        .eval_steps = eval_steps,
        .save_best = save_best,
        .report_to = report_to,
        .allow_large_memory = allow_large_memory,
    };

    try runTraining(allocator, opts);
}

// ---------------------------------------------------------------------------
// Core training routine
// ---------------------------------------------------------------------------

fn runTraining(allocator: std.mem.Allocator, opts: Options) !void {
    // ------------------------------------------------------------------
    // 1. Create output directory
    // ------------------------------------------------------------------
    try compat.cwd().createDirPath(compat.io(), opts.out_dir);

    print("train-gliner2-autodiff\n  model_dir={s}\n  train_data={s}\n  out_dir={s}\n", .{
        opts.model_dir,
        opts.train_data,
        opts.out_dir,
    });
    print("  epochs={d} batch_size={d} seq_len={d} lr={e:.6} weight_decay={e:.6}\n", .{
        opts.epochs,
        opts.batch_size,
        opts.seq_len,
        opts.learning_rate,
        opts.weight_decay,
    });
    print("  lora_rank={d} lora_alpha={d:.1} lora_dropout={d:.3} lora_targets={s}\n", .{
        opts.lora_rank,
        opts.lora_alpha,
        opts.lora_dropout,
        opts.lora_targets,
    });
    try gliner2_bundle.validateLoRADropout(opts.lora_dropout);
    print("  num_classes={d} seed={d} max_grad_norm={d:.2} grad_accum={d}\n", .{
        opts.num_classes,
        opts.seed,
        opts.max_grad_norm,
        opts.grad_accum,
    });
    print("  objective={s} max_span_width={d} span_loss={s} span_loss_reduction={s} span_pos_weight={d:.3} span_neg_weight={d:.3} span_hard_neg_weight={d:.3}\n", .{
        objectiveName(opts.objective),
        opts.max_span_width,
        spanLossName(opts.span_loss),
        spanLossReductionName(opts.span_loss_reduction),
        opts.span_positive_weight,
        opts.span_negative_weight,
        opts.span_hard_negative_weight,
    });
    if (opts.span_label_positive_weights) |weights| {
        print("  span_label_positive_weights={s}\n", .{weights});
    }
    if (!std.math.isFinite(opts.span_positive_weight) or opts.span_positive_weight <= 0.0) return error.InvalidSpanPositiveWeight;
    if (!std.math.isFinite(opts.span_negative_weight) or opts.span_negative_weight <= 0.0) return error.InvalidSpanNegativeWeight;
    if (!std.math.isFinite(opts.span_hard_negative_weight) or opts.span_hard_negative_weight <= 0.0) return error.InvalidSpanHardNegativeWeight;
    if (!std.math.isFinite(opts.span_negative_mask_rate) or opts.span_negative_mask_rate < 0.0 or opts.span_negative_mask_rate > 1.0) return error.InvalidSpanNegativeMaskRate;

    // Bounds-check integer CLI args before they feed size/shape math downstream.
    // A 0 here causes div-by-zero, cast/underflow, or a silent no-op; num_classes
    // must be >= 2 because the entity-type count is num_classes - 1.
    if (opts.batch_size < 1) {
        print("error: --batch-size must be >= 1 (got {d})\n", .{opts.batch_size});
        return error.InvalidBatchSize;
    }
    if (opts.seq_len < 1) {
        print("error: --seq-len must be >= 1 (got {d})\n", .{opts.seq_len});
        return error.InvalidSeqLen;
    }
    if (opts.lora_rank < 1) {
        print("error: --lora-rank must be >= 1 (got {d})\n", .{opts.lora_rank});
        return error.InvalidLoRARank;
    }
    if (opts.num_classes < 2) {
        print("error: --num-classes must be >= 2 (entity types = num_classes - 1; got {d})\n", .{opts.num_classes});
        return error.InvalidNumClasses;
    }
    if (opts.grad_accum < 1) {
        print("error: --grad-accum must be >= 1 (got {d})\n", .{opts.grad_accum});
        return error.InvalidGradAccum;
    }
    if (opts.epochs < 1) {
        print("error: --epochs must be >= 1 (got {d})\n", .{opts.epochs});
        return error.InvalidEpochs;
    }

    // ------------------------------------------------------------------
    // 2. Load DeBERTa config — GLiNER2 stores the encoder config under
    //    encoder_config/config.json, falling back to config.json.
    // ------------------------------------------------------------------
    var config_path_buf: [512]u8 = undefined;
    const encoder_config_path = try std.fmt.bufPrint(&config_path_buf, "{s}/encoder_config/config.json", .{opts.model_dir});
    const config_bytes = compat.cwd().readFileAlloc(compat.io(), encoder_config_path, allocator, .limited(8 * 1024 * 1024)) catch blk: {
        var fallback_buf: [512]u8 = undefined;
        const fallback_path = try std.fmt.bufPrint(&fallback_buf, "{s}/config.json", .{opts.model_dir});
        break :blk try compat.cwd().readFileAlloc(compat.io(), fallback_path, allocator, .limited(8 * 1024 * 1024));
    };
    defer allocator.free(config_bytes);

    const deberta_config = try parseDebertaConfig(allocator, config_bytes);
    print("  deberta: hidden={d} layers={d} heads={d} vocab={d}\n", .{
        deberta_config.hidden_size,
        deberta_config.num_hidden_layers,
        deberta_config.num_attention_heads,
        deberta_config.vocab_size,
    });

    // ------------------------------------------------------------------
    // 3. Set up compute backend + load weights
    //
    // Use MLX (Apple Silicon GPU) when available, falling back to
    // native CPU BLAS.
    // ------------------------------------------------------------------
    var st_path_buf: [512]u8 = undefined;
    const st_path = try std.fmt.bufPrint(&st_path_buf, "{s}/model.safetensors", .{opts.model_dir});

    // We need these variables to live for the whole function regardless
    // of which backend branch we take.
    var mlx_ws: if (build_options.enable_mlx) mlx_compute.WeightStore else void = undefined;
    var mlx_backend: if (build_options.enable_mlx) *mlx_compute.MlxCompute else void = undefined;
    var metal_ws: MetalWeightStore = undefined;
    var metal_backend: if (build_options.enable_metal) metal_compute.MetalCompute else void = undefined;
    var native_ws: native_compute.WeightStore = undefined;
    var native_backend: native_compute.NativeCompute = undefined;

    // SafetensorsSource kept alive for native path (mmap'd data).
    var safetensors_source: ?*SafetensorsSource = null;
    defer if (safetensors_source) |s| s.weightSource().deinit();

    const force_native = envFlag("TERMITE_GLINER2_FORCE_NATIVE");
    const metal_runtime_available = if (comptime build_options.enable_metal)
        (!force_native and metal_runtime.metalDeviceAvailable())
    else
        false;
    const mlx_runtime_available = if (comptime build_options.enable_mlx)
        (!force_native and (mlx.metalDeviceAvailable() or mlx.allowCpuStreamWithoutMetal()))
    else
        false;
    if (comptime build_options.enable_mlx) {
        if (force_native) {
            print("info: TERMITE_GLINER2_FORCE_NATIVE is set; using native CPU/BLAS\n", .{});
        } else if (!mlx_runtime_available) {
            print("warning: MLX build enabled but no Metal device is available; falling back to native CPU/BLAS\n", .{});
        }
    }

    const selected_backend = selectBackend(opts.backend, force_native, metal_runtime_available, mlx_runtime_available) catch |err| {
        switch (err) {
            error.MetalBackendUnavailable => print("error: --backend metal requested but Metal is not built or no Metal device is available\n", .{}),
            error.MlxBackendUnavailable => print("error: --backend mlx requested but MLX is not built or unavailable\n", .{}),
        }
        return err;
    };

    const cb = if (selected_backend == .metal) blk: {
        if (comptime build_options.enable_metal) {
            metal_ws = .{
                .allocator = allocator,
                .resident_weights = if (comptime build_options.enable_mlx) mlx_c.mlx_map_string_to_array_new() else {},
                .stream = if (comptime build_options.enable_mlx) mlx.openDefaultStream().stream else {},
                .prefix = "",
                .lazy_weights = .{},
            };
            try loadSafetensorsIntoGpuHostedStore(allocator, &metal_ws, st_path);
            try initClassifierHeadInGpuHostedStore(allocator, &metal_ws, opts.seed, deberta_config.hidden_size, opts.num_classes);
            metal_compute.initPrefetchQueue(&metal_ws, allocator);
            metal_backend = try metal_compute.MetalCompute.init(allocator, &metal_ws, null);
            break :blk metal_backend.computeBackend();
        } else unreachable;
    } else if (selected_backend == .mlx) blk: {
        if (comptime !build_options.enable_mlx) unreachable;
        // ── MLX path: load weights directly into MLX arrays ──────────
        const raw_weights = try mlx.loadSafetensors(st_path, allocator, mlx.openDefaultStream().stream);
        // Build a new map with "encoder." prefix stripped.
        const stripped_weights = mlx_c.mlx_map_string_to_array_new();
        const it = mlx_c.mlx_map_string_to_array_iterator_new(raw_weights);
        defer _ = mlx_c.mlx_map_string_to_array_iterator_free(it);
        var loaded_count: usize = 0;
        while (true) {
            var key: [*c]const u8 = null;
            var val = mlx_c.mlx_array_new();
            if (mlx_c.mlx_map_string_to_array_iterator_next(&key, &val, it) != 0) {
                _ = mlx_c.mlx_array_free(val);
                break;
            }
            if (key == null) {
                _ = mlx_c.mlx_array_free(val);
                break;
            }
            const name = std.mem.span(key);
            const stripped = stripEncoderPrefix(name);
            const stripped_z = try allocator.dupeZ(u8, stripped);
            defer allocator.free(stripped_z);
            _ = mlx_c.mlx_map_string_to_array_insert(stripped_weights, stripped_z.ptr, val);
            _ = mlx_c.mlx_array_free(val);
            loaded_count += 1;
        }
        _ = mlx_c.mlx_map_string_to_array_free(raw_weights);
        print("  loaded {d} weights via MLX from {s}\n", .{ loaded_count, st_path });

        // Initialize classifier head as MLX arrays.
        {
            var rng_init = std.Random.DefaultPrng.init(opts.seed);
            var prng_init = rng_init.random();
            const H = deberta_config.hidden_size;
            const C = opts.num_classes;

            const w_data = try allocator.alloc(f32, C * H);
            defer allocator.free(w_data);
            const sd: f32 = 0.02;
            for (w_data) |*v| v.* = prng_init.floatNorm(f32) * sd;
            const w_shape = [_]i32{ @intCast(C), @intCast(H) };
            const w_arr = mlx.arrayFromFloat32(w_data, &w_shape);
            try mlx.insertWeight(stripped_weights, allocator, "task_classifier.weight", w_arr);

            const b_data = try allocator.alloc(f32, C);
            defer allocator.free(b_data);
            @memset(b_data, 0.0);
            const b_shape = [_]i32{@intCast(C)};
            const b_arr = mlx.arrayFromFloat32(b_data, &b_shape);
            try mlx.insertWeight(stripped_weights, allocator, "task_classifier.bias", b_arr);
            print("  initialized classifier head (MLX): [{d}, {d}] + [{d}]\n", .{ C, H, C });
        }
        try initParityTopLevelWeightsMlx(allocator, stripped_weights, deberta_config.hidden_size);

        mlx_ws = .{
            .allocator = allocator,
            .resident_weights = stripped_weights,
            .stream = mlx.openDefaultStream().stream,
            .prefix = "",
            .lazy_weights = .{},
        };
        mlx_backend = try allocator.create(mlx_compute.MlxCompute);
        mlx_backend.* = try mlx_compute.MlxCompute.init(allocator, &mlx_ws, null);
        break :blk mlx_backend.computeBackend();
    } else blk: {
        // ── Native CPU/BLAS fallback ─────────────────────────────────
        native_ws = .{
            .allocator = allocator,
            .resident_weights = .{},
            .lazy_weights = .{},
        };

        if (SafetensorsSource.initAbsolute(allocator, st_path)) |src| {
            safetensors_source = src;
            const ws = src.weightSource();
            if (ws.listNames(allocator)) |names| {
                defer allocator.free(names);
                var loaded_count: usize = 0;
                for (names) |name| {
                    if (ws.getTensor(name)) |lw| {
                        const stripped = stripEncoderPrefix(name);
                        const owned_name = try allocator.dupe(u8, stripped);
                        errdefer allocator.free(owned_name);
                        try native_ws.resident_weights.put(allocator, owned_name, lw);
                        loaded_count += 1;
                    } else |_| {}
                }
                print("  loaded {d} weights (native) from {s}\n", .{ loaded_count, st_path });
            } else |err| {
                print("warning: could not list weights: {}\n", .{err});
            }
        } else |err| {
            return err;
        }

        // Initialize classifier head.
        {
            var rng_init = std.Random.DefaultPrng.init(opts.seed);
            var prng_init = rng_init.random();
            const H = deberta_config.hidden_size;
            const C = opts.num_classes;

            const w_data = try allocator.alloc(f32, C * H);
            const sd: f32 = 0.02;
            for (w_data) |*v| v.* = prng_init.floatNorm(f32) * sd;
            const w_tensor = try Tensor.initFloat32(allocator, "task_classifier.weight", &.{ C, H }, w_data);
            allocator.free(w_data);
            try native_ws.resident_weights.put(allocator, try allocator.dupe(u8, "task_classifier.weight"), .{ .tensor = w_tensor });

            const b_data = try allocator.alloc(f32, C);
            @memset(b_data, 0.0);
            const b_tensor = try Tensor.initFloat32(allocator, "task_classifier.bias", &.{C}, b_data);
            allocator.free(b_data);
            try native_ws.resident_weights.put(allocator, try allocator.dupe(u8, "task_classifier.bias"), .{ .tensor = b_tensor });
            print("  initialized classifier head (native): [{d}, {d}] + [{d}]\n", .{ C, H, C });
        }
        try initParityTopLevelWeightsNative(allocator, &native_ws, deberta_config.hidden_size);

        native_backend = native_compute.NativeCompute.init(allocator, &native_ws, null);
        break :blk native_backend.computeBackend();
    };
    defer switch (selected_backend) {
        .native => deinitNativeWeightStore(allocator, &native_ws),
        .metal => if (comptime build_options.enable_metal) deinitGpuHostedWeightStore(allocator, &metal_ws),
        else => {},
    };
    defer switch (selected_backend) {
        .metal => if (comptime build_options.enable_metal) metal_backend.deinit(),
        .mlx => if (comptime build_options.enable_mlx) mlx_backend.deinit(),
        else => {},
    };

    print("  backend: {s}\n", .{backendLabel(selected_backend)});

    // ------------------------------------------------------------------
    // 5. Load training data (JSONL with text + entities)
    // ------------------------------------------------------------------
    var train_loaded = try gliner2_data.loadExamples(allocator, opts.train_data, null);
    defer train_loaded.deinit();
    var train_records_loaded = if (opts.objective == .gliner2_total_loss)
        try gliner2_data.loadTrainingRecords(allocator, opts.train_data, null)
    else
        null;
    defer if (train_records_loaded) |*loaded| loaded.deinit();

    var examples = train_loaded.examples;
    var training_records: []gliner2_data.UpstreamRecord = if (train_records_loaded) |*loaded| loaded.records else &.{};
    if (opts.max_examples > 0 and examples.len > opts.max_examples) {
        examples = examples[0..opts.max_examples];
    }
    if (opts.max_examples > 0 and training_records.len > opts.max_examples) {
        training_records = training_records[0..opts.max_examples];
    }

    const stats = try gliner2_data.computeStats(allocator, examples);
    print("  training examples: {d} (avg_chars={d:.1}, avg_entities={d:.2}, unique_labels={d})\n", .{
        stats.num_examples,
        stats.avg_text_chars,
        stats.avg_entities,
        stats.unique_labels,
    });

    if (examples.len == 0) {
        print("error: no training examples loaded\n", .{});
        return error.NoTrainingData;
    }
    if (opts.objective == .gliner2_total_loss and training_records.len != examples.len) {
        print("error: gliner2-total-loss requires aligned upstream records ({d}) and flattened examples ({d})\n", .{ training_records.len, examples.len });
        return error.InvalidGliner2Example;
    }

    // ------------------------------------------------------------------
    // 6. Build a label-to-class-index mapping from the training data
    // ------------------------------------------------------------------
    // Class 0 is always the "O" (no entity) class. Prefer an explicit caller
    // entity order so training, manifest export, and evaluation agree on
    // class IDs; legacy direct invocations fall back to sorted dataset labels.
    var label_map = std.StringHashMapUnmanaged(u32){};
    defer label_map.deinit(allocator);
    const extra_entity_types = if (opts.entity_types_csv) |csv| try parseEntityTypesCsvOwned(allocator, csv) else null;
    defer if (extra_entity_types) |items| {
        for (items) |label| allocator.free(label);
        allocator.free(items);
    };
    const entity_types = if (opts.objective == .gliner2_total_loss)
        try gliner2_data.buildUpstreamTaskLabelVocab(allocator, training_records, extra_entity_types)
    else if (extra_entity_types) |items|
        try dupeStringSlice(allocator, items)
    else
        try gliner2_data.buildLabelVocab(allocator, examples, null);
    defer {
        for (entity_types) |label| allocator.free(label);
        allocator.free(entity_types);
    }
    const effective_num_classes: u32 = if (opts.objective == .gliner2_total_loss and opts.entity_types_csv == null)
        @intCast(entity_types.len + 1)
    else
        opts.num_classes;
    if (entity_types.len + 1 > effective_num_classes) {
        print("error: dataset has {d} entity labels but num_classes={d} only has {d} entity slots\n", .{
            entity_types.len,
            effective_num_classes,
            if (effective_num_classes > 0) effective_num_classes - 1 else 0,
        });
        return error.TooManyEntityTypes;
    }
    if ((opts.objective == .span_start or opts.objective == .gliner2_total_loss) and entity_types.len + 1 != @as(usize, @intCast(effective_num_classes))) {
        print("error: span-start objective currently requires num_classes == entity_label_count + 1 ({d}); got {d}\n", .{
            entity_types.len + 1,
            effective_num_classes,
        });
        return error.SpanObjectiveRequiresExactClassCount;
    }
    for (entity_types, 0..) |label, idx| {
        try label_map.put(allocator, label, @intCast(idx + 1));
    }
    print("  entity labels mapped: {d} (num_classes={d})\n", .{ label_map.count(), effective_num_classes });
    const resolved_span_label_positive_weights = try resolveSpanLabelPositiveWeights(
        allocator,
        opts.span_label_positive_weights,
        entity_types,
        opts.span_positive_weight,
    );
    defer allocator.free(resolved_span_label_positive_weights);

    // ------------------------------------------------------------------
    // 6b. Initialize the HF tokenizer for proper DeBERTa-v3 encoding
    // ------------------------------------------------------------------
    var tokenizer = try gliner2_data.Tokenizer.initGLiNER2HF(allocator, opts.model_dir);
    defer tokenizer.deinit(allocator);
    print("  tokenizer: vocab_size={d} cls={d} sep={d} ent={d} sep_text={d}\n", .{
        tokenizer.vocab_size,
        tokenizer.cls_id,
        tokenizer.sep_id,
        tokenizer.ent_id,
        tokenizer.sep_text_token_id,
    });

    // ------------------------------------------------------------------
    // 6c. Fit-to-data sequence length. The DeBERTa graph bakes in seq_len
    // (relative-position tables) and is built+cached once, so padding every
    // batch to the fixed --seq-len wastes S^2-proportional attention compute
    // (and can OOM) on padding. Scan the ACTUAL encoded token length of the
    // data (via the same batch builders the training loop uses) and size the
    // single graph to that, capped by --seq-len (upper bound) and the
    // relative-position table limit (512). Matches upstream GLiNER2Trainer's
    // dynamic padding. The .token objective keeps the fixed length.
    // ------------------------------------------------------------------
    const effective_seq_len: usize = blk: {
        if (opts.objective == .token) break :blk opts.seq_len;
        // Escape hatch: force the fixed --seq-len (e.g. to reproduce a
        // fixed-padding baseline or match another runner's seq length).
        if (std.c.getenv("TERMITE_GLINER2_DISABLE_FIT_SEQ_LEN") != null) break :blk opts.seq_len;
        var scanned_max: usize = 0;
        var i: usize = 0;
        while (i < examples.len) : (i += 1) {
            var enc = switch (opts.objective) {
                .gliner2_total_loss => try gliner2_data.buildUpstreamTaskBatch(allocator, &tokenizer, training_records[i .. i + 1], entity_types, opts.seq_len, opts.max_span_width, 1),
                else => try gliner2_data.buildSimpleBatch(allocator, &tokenizer, examples[i .. i + 1], entity_types, opts.seq_len, opts.max_span_width, 1),
            };
            defer enc.deinit();
            var used: usize = 0;
            for (enc.attention_mask) |m| {
                if (m != 0) used += 1;
            }
            if (used > scanned_max) scanned_max = used;
        }
        const floor_len: usize = @as(usize, @intCast(opts.max_span_width)) + 8;
        const rounded = (@max(scanned_max, floor_len) + 7) / 8 * 8;
        break :blk @min(@as(usize, 512), @min(opts.seq_len, rounded));
    };
    print("  effective seq_len: {d} (cap --seq-len={d})\n", .{ effective_seq_len, opts.seq_len });

    // ------------------------------------------------------------------
    // 6d. Memory pre-flight. The disentangled-attention intermediates scale
    // batch*S^2 and have previously hard-OOMed the whole machine (not just
    // this process) on 16GB hosts. Legacy frame-retained execution must budget
    // every layer's transient attention workspace. Metal's default in-frame
    // private-buffer reuse budgets the live workspace instead, matching the
    // allocator behavior this gate is meant to prove.
    // ------------------------------------------------------------------
    {
        const reuse_preflight = selected_backend == .metal and metalBufferReuseEnabledForPreflight();
        const est = estimateTrainingPeakBytes(
            @intCast(deberta_config.vocab_size),
            @intCast(deberta_config.hidden_size),
            @intCast(deberta_config.intermediate_size),
            @intCast(deberta_config.num_hidden_layers),
            @intCast(deberta_config.num_attention_heads),
            opts.batch_size,
            effective_seq_len,
            reuse_preflight,
        );
        if (physicalMemoryBytes()) |total| {
            const budget = total * 6 / 10;
            print("  estimated peak memory ({s}): {d:.2} GiB (budget {d:.2} GiB of {d:.2} GiB physical)\n", .{
                if (reuse_preflight) "metal-reuse-live-set" else "frame-retained",
                @as(f64, @floatFromInt(est)) / (1024.0 * 1024.0 * 1024.0),
                @as(f64, @floatFromInt(budget)) / (1024.0 * 1024.0 * 1024.0),
                @as(f64, @floatFromInt(total)) / (1024.0 * 1024.0 * 1024.0),
            });
            if (est > budget and !opts.allow_large_memory) {
                print("error: estimated peak memory exceeds the safe budget; lower --batch-size or --seq-len (attention intermediates scale batch*seq^2), enable Metal buffer reuse, or pass --allow-large-memory to proceed anyway\n", .{});
                return error.EstimatedMemoryExceedsBudget;
            }
            if (est > budget) {
                print("warning: --allow-large-memory set; proceeding past the safe memory budget\n", .{});
            }
        }
    }

    // ------------------------------------------------------------------
    // 7. Parse LoRA target patterns
    // ------------------------------------------------------------------
    var target_patterns = std.ArrayListUnmanaged([]const u8).empty;
    defer target_patterns.deinit(allocator);
    {
        var iter = std.mem.tokenizeScalar(u8, opts.lora_targets, ',');
        while (iter.next()) |tok| {
            try target_patterns.append(allocator, std.mem.trim(u8, tok, " "));
        }
    }
    const resolved_target_patterns = try gliner2_bundle.expandLoRATargetModules(allocator, target_patterns.items);
    defer {
        for (resolved_target_patterns) |item| allocator.free(item);
        allocator.free(resolved_target_patterns);
    }

    // ------------------------------------------------------------------
    // 8. Build the DeBERTa graph config + GlinerAutodiffCtx
    // ------------------------------------------------------------------
    const graph_config = deberta_graph.Config{
        .vocab_size = deberta_config.vocab_size,
        .hidden_size = deberta_config.hidden_size,
        .num_hidden_layers = deberta_config.num_hidden_layers,
        .num_attention_heads = deberta_config.num_attention_heads,
        .intermediate_size = deberta_config.intermediate_size,
        .max_position_embeddings = deberta_config.max_position_embeddings,
        .position_buckets = deberta_config.position_buckets,
        .layer_norm_eps = deberta_config.layer_norm_eps,
    };

    if (selected_backend == .metal) {
        const preplan_config = debertaArchConfigFromJson(deberta_config);
        const preplanned = deberta_arch.preplanMetalDebertaEncoderFrame(
            &cb,
            allocator,
            preplan_config,
            opts.batch_size,
            effective_seq_len,
        ) catch |err| blk: {
            print("warning: Metal DeBERTa encoder frame preplan failed: {s}; continuing with graph runtime fallback\n", .{@errorName(err)});
            break :blk false;
        };
        print("  metal deberta encoder frame preplan: {s}\n", .{if (preplanned) "ready" else "not-ready"});
    }

    const structure_max_instances: u32 = blk: {
        var n: u32 = if (opts.objective == .gliner2_total_loss)
            computeMaxStructureInstances(training_records)
        else
            1;
        // Perf-A/B override: force the structure-loss path width regardless of
        // data (e.g. =1 forces the legacy single-instance path on multi-instance
        // data for a same-fixture baseline). Loss is not meaningful when forced
        // below the data's real max; intended only for timing comparisons.
        if (std.c.getenv("TERMITE_GLINER2_STRUCT_MAX_INSTANCES")) |cstr| {
            const val = std.mem.span(cstr);
            if (std.fmt.parseInt(u32, std.mem.trim(u8, val, " \t\r\n"), 10)) |forced| {
                if (forced >= 1) {
                    print("  [override] structure_max_instances forced to {d} (was {d})\n", .{ forced, n });
                    n = forced;
                }
            } else |_| {}
        }
        break :blk n;
    };
    if (structure_max_instances > 1) {
        print("  structure max instances (per-instance struct loss): {d}\n", .{structure_max_instances});
    }

    var gliner_ctx = gliner2_autodiff.GlinerAutodiffCtx.init(.{
        .graph_config = graph_config,
        .num_classes = effective_num_classes,
        .objective = opts.objective,
        .span_start_loss = opts.span_loss,
        .span_start_loss_reduction = opts.span_loss_reduction,
        .span_start_positive_weight = opts.span_positive_weight,
        .span_start_negative_weight = opts.span_negative_weight,
        .structure_max_instances = structure_max_instances,
    });

    // ------------------------------------------------------------------
    // 9. Initialize the RealAutodiffTrainer
    // ------------------------------------------------------------------
    const lora_config = ml.graph.lora.LoRAConfig{
        .rank = opts.lora_rank,
        .alpha = opts.lora_alpha,
        .dropout = opts.lora_dropout,
        .target_patterns = resolved_target_patterns,
        .strict_target_patterns = true,
    };
    const regular_trainable_params_with_classifier = [_][]const u8{ "task_classifier.weight", "task_classifier.bias" };
    const regular_trainable_params_bias_only = [_][]const u8{"task_classifier.bias"};
    const no_regular_trainable_params = [_][]const u8{};
    const regular_trainable_params = if (opts.lora_only_trainables)
        no_regular_trainable_params[0..]
    else if (stringSliceContains(resolved_target_patterns, "classifier"))
        regular_trainable_params_bias_only[0..]
    else
        regular_trainable_params_with_classifier[0..];
    print("  trainable mode: LoRA{s} regular_trainable_params={d}\n", .{
        if (opts.lora_only_trainables) " only" else " + regular head",
        regular_trainable_params.len,
    });

    var trainer = try real_autodiff.RealAutodiffTrainer.init(
        allocator,
        &cb,
        .{
            .lora = lora_config,
            .optimizer = .{ .weight_decay = opts.weight_decay },
            .lr_schedule = .{ .constant = opts.learning_rate },
            .max_grad_norm = opts.max_grad_norm,
            .grad_accum_steps = opts.grad_accum,
            .hidden_size_hint = deberta_config.hidden_size,
            .num_layers_hint = deberta_config.num_hidden_layers,
            .seed = opts.seed,
            .regular_trainable_params = regular_trainable_params,
            .execution_engine = switch (selected_backend) {
                .metal => .compiled_metal,
                .mlx => .compiled_mlx,
                else => .interpreter,
            },
            .compiled_required = opts.compiled_required,
            .checkpoint_config = activationCheckpointConfig(),
        },
    );
    defer trainer.deinit();

    // ------------------------------------------------------------------
    // 10. Training loop
    // ------------------------------------------------------------------
    const total_examples = examples.len;
    const steps_per_epoch = (total_examples + opts.batch_size - 1) / opts.batch_size;

    print("\nStarting training: {d} epochs x {d} steps/epoch ({d} examples)\n", .{
        opts.epochs,
        steps_per_epoch,
        total_examples,
    });

    // Pre-allocate batch buffers (sized to the fit-to-data effective length).
    const sl: usize = effective_seq_len;
    const bs: usize = opts.batch_size;
    const nc: usize = effective_num_classes;
    const batch_tokens = bs * sl;
    const use_label_positive_weights = opts.span_label_positive_weights != null;
    const span_entity_types: usize = if (effective_num_classes > 1) @as(usize, @intCast(effective_num_classes)) - 1 else 0;
    const span_target_width: usize = if (opts.objective == .gliner2_total_loss)
        gliner2_autodiff.gliner2TotalLossTargetWidthEx(span_entity_types, structure_max_instances)
    else if (use_label_positive_weights)
        gliner2_autodiff.weightedSpanStartTargetWidth(span_entity_types)
    else
        gliner2_autodiff.spanStartTargetWidth(span_entity_types);
    const max_span_target_values = bs * sl * @as(usize, @intCast(opts.max_span_width)) * span_target_width;
    const target_buf_values = @max(batch_tokens * nc, max_span_target_values);

    var input_ids = try allocator.alloc(i64, batch_tokens);
    defer allocator.free(input_ids);
    var attention_mask = try allocator.alloc(f32, batch_tokens);
    defer allocator.free(attention_mask);
    // Token mode: [batch * seq_len, num_classes].
    // Span mode: [batch * max_spans, 2 * entity_types + 2], or
    // [batch * max_spans, 3 * entity_types + 2] when per-label positive
    // weights are packed into the target tensor.
    var targets_buf = try allocator.alloc(f32, target_buf_values);
    defer allocator.free(targets_buf);
    // Staging windows for one batch. The autodiff graph is compiled once for
    // a fixed (batch, seq_len), so a final partial batch is padded back up to
    // `bs` by repeating the last real example; the padded rows then have all
    // supervision labels/masks zeroed (see
    // gliner2_autodiff.zeroPaddedSpanTargetRows) so they are exact no-ops in
    // every loss component while keeping the packed gather indices in-bounds.
    var batch_examples = try allocator.alloc(gliner2_data.Example, bs);
    defer allocator.free(batch_examples);
    var batch_records = try allocator.alloc(gliner2_data.UpstreamRecord, bs);
    defer allocator.free(batch_records);

    if (opts.initial_adapter_checkpoint) |checkpoint_path| {
        try ensureTrainerGraphBuiltFromFirstBatch(
            allocator,
            opts,
            effective_seq_len,
            &tokenizer,
            entity_types,
            examples,
            training_records,
            &label_map,
            effective_num_classes,
            use_label_positive_weights,
            resolved_span_label_positive_weights,
            input_ids,
            attention_mask,
            targets_buf,
            &trainer,
            &gliner_ctx,
        );
        try loadPeftAdaptersIntoTrainer(allocator, checkpoint_path, &trainer);
        print("  loaded initial LoRA adapter checkpoint: {s}\n", .{checkpoint_path});
    }

    var rng = std.Random.DefaultPrng.init(opts.seed);
    var prng = rng.random();

    var cumulative_loss: f64 = 0.0;
    var total_steps: u64 = 0;
    var run_target_stats = BatchTargetStats{};
    var metrics_jsonl: std.Io.Writer.Allocating = .init(allocator);
    defer metrics_jsonl.deinit();

    // Eval/save-best bookkeeping. With no held-out eval set wired into this
    // CLI yet, the eval metric is the average training loss over the eval
    // window (per epoch, or per --eval-steps optimizer steps).
    var best_eval_loss: f64 = std.math.inf(f64);
    var eval_window_loss: f64 = 0.0;
    var eval_window_steps: u64 = 0;

    for (0..opts.epochs) |epoch| {
        // Shuffle examples at the start of each epoch.
        if (opts.objective == .gliner2_total_loss) {
            if (!opts.dump_span_parity) shuffleExamplesAndRecords(&prng, examples, training_records);
        } else {
            prng.shuffle(gliner2_data.Example, examples);
        }

        const epoch_started_ns = monotonicNowNs();
        var epoch_loss: f64 = 0.0;
        var epoch_steps: u64 = 0;
        var epoch_target_stats = BatchTargetStats{};

        var batch_start: usize = 0;
        while (batch_start < total_examples) {
            const batch_end = @min(batch_start + bs, total_examples);
            // Pad a partial final batch up to the fixed graph batch size by
            // repeating the last real example; the padded targets are zero-
            // masked below so they contribute nothing to any loss component.
            const real_batch: usize = batch_end - batch_start;
            for (0..bs) |slot| {
                const src = batch_start + @min(slot, real_batch - 1);
                batch_examples[slot] = examples[src];
                if (opts.objective == .gliner2_total_loss) batch_records[slot] = training_records[src];
            }
            const actual_batch: u32 = @intCast(bs);
            const ab: usize = bs;

            // Tokenize batch + build entity/span targets.
            const step_started_ns = monotonicNowNs();
            var target_stats = BatchTargetStats{};
            var targets_shape: ml.graph.Shape = undefined;
            var target_slice: []const f32 = undefined;
            switch (opts.objective) {
                .token => {
                    target_stats = fillBatchBuffers(
                        allocator,
                        &tokenizer,
                        entity_types,
                        batch_examples,
                        @intCast(sl),
                        opts.num_classes,
                        &label_map,
                        input_ids,
                        attention_mask,
                        targets_buf[0 .. ab * sl * nc],
                    );
                    // Padded duplicate examples become all-zero target rows,
                    // which the masked token loss ignores entirely.
                    if (real_batch < ab) @memset(targets_buf[real_batch * sl * nc .. ab * sl * nc], 0.0);
                    targets_shape = gliner2_autodiff.tokenTargetsShape(
                        actual_batch,
                        @intCast(sl),
                        opts.num_classes,
                    );
                    target_slice = targets_buf[0 .. ab * sl * nc];
                },
                .span_start, .gliner2_total_loss => {
                    var encoded = if (opts.objective == .gliner2_total_loss)
                        try gliner2_data.buildUpstreamTaskBatch(
                            allocator,
                            &tokenizer,
                            batch_records,
                            entity_types,
                            sl,
                            opts.max_span_width,
                            ab,
                        )
                    else
                        try gliner2_data.buildSimpleBatch(
                            allocator,
                            &tokenizer,
                            batch_examples,
                            entity_types,
                            sl,
                            opts.max_span_width,
                            ab,
                        );
                    defer encoded.deinit();

                    if (encoded.input_ids.len != ab * sl or encoded.attention_mask.len != ab * sl) return error.InvalidGlinerBatchShape;
                    for (0..ab * sl) |i| {
                        input_ids[i] = encoded.input_ids[i];
                        attention_mask[i] = @floatFromInt(encoded.attention_mask[i]);
                    }

                    const width = if (opts.objective == .gliner2_total_loss)
                        gliner2_autodiff.gliner2TotalLossTargetWidthEx(encoded.num_entity_types, structure_max_instances)
                    else if (use_label_positive_weights)
                        gliner2_autodiff.weightedSpanStartTargetWidth(encoded.num_entity_types)
                    else
                        gliner2_autodiff.spanStartTargetWidth(encoded.num_entity_types);
                    const target_len = encoded.batch_size * encoded.max_spans * width;
                    const span_stats = if (opts.objective == .gliner2_total_loss)
                        try fillGliner2TotalLossTargetsFromRecords(
                            allocator,
                            &encoded,
                            batch_records,
                            entity_types,
                            structure_max_instances,
                            targets_buf[0..target_len],
                        )
                    else
                        try gliner2_autodiff.fillSpanStartTargetsFromEncodedBatchWithOptions(
                            &encoded,
                            .{
                                .positive_weights_by_entity_type = if (use_label_positive_weights) resolved_span_label_positive_weights else null,
                                .hard_negative_weight = opts.span_hard_negative_weight,
                            },
                            targets_buf[0..target_len],
                        );
                    if (opts.span_negative_mask_rate > 0.0 and opts.objective == .span_start) {
                        applySpanNegativeMask(
                            targets_buf[0..target_len],
                            encoded.max_spans,
                            encoded.num_entity_types,
                            use_label_positive_weights,
                            opts.span_negative_mask_rate,
                            opts.seed ^ total_steps,
                        );
                    }
                    if (real_batch < ab) try gliner2_autodiff.zeroPaddedSpanTargetRows(
                        targets_buf[0..target_len],
                        opts.objective,
                        encoded.num_entity_types,
                        encoded.max_spans,
                        real_batch,
                        ab,
                        use_label_positive_weights,
                        structure_max_instances,
                    );
                    target_stats = BatchTargetStats.fromSpanStart(span_stats, encoded.num_entity_types);
                    targets_shape = if (opts.objective == .gliner2_total_loss)
                        gliner2_autodiff.gliner2TotalLossTargetsShapeEx(
                            actual_batch,
                            @intCast(encoded.max_spans),
                            @intCast(encoded.num_entity_types),
                            structure_max_instances,
                        )
                    else if (use_label_positive_weights)
                        gliner2_autodiff.weightedSpanStartTargetsShape(
                            actual_batch,
                            @intCast(encoded.max_spans),
                            @intCast(encoded.num_entity_types),
                        )
                    else
                        gliner2_autodiff.spanStartTargetsShape(
                            actual_batch,
                            @intCast(encoded.max_spans),
                            @intCast(encoded.num_entity_types),
                        );
                    target_slice = targets_buf[0..target_len];
                    if (opts.dump_span_parity and total_steps == 0) {
                        printSpanPreprocessDebug(&encoded);
                    }
                },
            }
            const target_built_ns = monotonicNowNs();

            // Build TrainerInput via the GLiNER2 convenience builder.
            const trainer_input = gliner2_autodiff.makeTrainerInput(
                &gliner_ctx,
                input_ids[0 .. ab * sl],
                attention_mask[0 .. ab * sl],
                target_slice,
                targets_shape,
                actual_batch,
                @intCast(sl),
            );

            if (opts.dump_span_parity and opts.objective == .span_start and total_steps == 0) {
                const logits = try gliner2_autodiff.spanStartLogitsForBatch(
                    allocator,
                    &trainer,
                    &gliner_ctx,
                    input_ids[0 .. ab * sl],
                    attention_mask[0 .. ab * sl],
                    target_slice,
                    targets_shape,
                    actual_batch,
                    @intCast(sl),
                );
                defer allocator.free(logits);
                try printSpanParityDebug(logits, target_slice, targets_shape, entity_types.len, use_label_positive_weights, opts);
                const components = try gliner2_autodiff.spanStartComponentDebugForBatch(
                    allocator,
                    &trainer,
                    &gliner_ctx,
                    input_ids[0 .. ab * sl],
                    attention_mask[0 .. ab * sl],
                    target_slice,
                    targets_shape,
                    actual_batch,
                    @intCast(sl),
                    entity_types.len,
                );
                printSpanComponentDebug(components);
            }
            if (opts.dump_span_parity and opts.objective == .gliner2_total_loss and total_steps == 0) {
                // The legacy span-parity dumps assume the single-instance
                // structure logits shape `[span_rows, E]`. The per-instance
                // structure loss emits `[span_rows, max_gold * E]`, so skip
                // those dumps there and still emit the component-loss debug
                // (which the parity harness reads).
                if (gliner_ctx.config.structure_max_instances == 1) {
                    const logits = try gliner2_autodiff.spanStartLogitsForBatch(
                        allocator,
                        &trainer,
                        &gliner_ctx,
                        input_ids[0 .. ab * sl],
                        attention_mask[0 .. ab * sl],
                        target_slice,
                        targets_shape,
                        actual_batch,
                        @intCast(sl),
                    );
                    defer allocator.free(logits);
                    try printSpanParityDebug(logits, target_slice, targets_shape, entity_types.len, false, opts);
                    if (gliner2_autodiff.spanStartComponentDebugForBatch(
                        allocator,
                        &trainer,
                        &gliner_ctx,
                        input_ids[0 .. ab * sl],
                        attention_mask[0 .. ab * sl],
                        target_slice,
                        targets_shape,
                        actual_batch,
                        @intCast(sl),
                        entity_types.len,
                    )) |span_components| {
                        printSpanComponentDebug(span_components);
                    } else |err| switch (err) {
                        error.MissingPositiveSpanLabel => {},
                        else => return err,
                    }
                }
                const components = try gliner2_autodiff.gliner2TotalLossComponentDebugForBatch(
                    allocator,
                    &trainer,
                    &gliner_ctx,
                    input_ids[0 .. ab * sl],
                    attention_mask[0 .. ab * sl],
                    target_slice,
                    targets_shape,
                    actual_batch,
                    @intCast(sl),
                );
                printGliner2TotalLossComponentDebug(components);
                const classification_logits = try gliner2_autodiff.gliner2ClassificationLogitsForBatch(
                    allocator,
                    &trainer,
                    &gliner_ctx,
                    input_ids[0 .. ab * sl],
                    attention_mask[0 .. ab * sl],
                    target_slice,
                    targets_shape,
                    actual_batch,
                    @intCast(sl),
                );
                defer allocator.free(classification_logits);
                try printGliner2ClassificationDebug(classification_logits, target_slice, targets_shape, entity_types.len);
            }

            const result = try trainer.step(trainer_input);
            const step_finished_ns = monotonicNowNs();
            const timing = StepTiming{
                .target_build_ns = elapsedNs(step_started_ns, target_built_ns),
                .train_step_ns = elapsedNs(target_built_ns, step_finished_ns),
                .step_wall_ns = elapsedNs(step_started_ns, step_finished_ns),
                .profile = result.profile,
            };
            epoch_loss += result.loss;
            epoch_steps += 1;
            epoch_target_stats.add(target_stats);
            run_target_stats.add(target_stats);
            total_steps += 1;
            if (result.optimizer_stepped) {
                try syncZeroGradLoraOptimizerSteps(allocator, &trainer, opts.learning_rate);
                if (opts.dump_optimizer_parity) {
                    try printOptimizerParityDump(allocator, &trainer, total_steps);
                }
            }
            eval_window_loss += result.loss;
            eval_window_steps += 1;
            try writeStepMetric(&metrics_jsonl.writer, epoch + 1, total_steps, epoch_steps, result.loss, result.grad_norm, result.optimizer_stepped, target_stats, timing);

            if (opts.report_to == .jsonl) {
                print(
                    "{{\"event\":\"step\",\"step\":{d},\"epoch\":{d},\"epoch_step\":{d},\"loss\":{d:.9},\"grad_norm\":{d:.9},\"lr\":{e:.6}}}\n",
                    .{ total_steps, epoch + 1, epoch_steps, result.loss, result.grad_norm, opts.learning_rate },
                );
            }
            if (total_steps % 10 == 0 or batch_end >= total_examples) {
                print("  [epoch {d}/{d}] step {d}/{d}  loss={d:.6}  grad_norm={d:.4}  supervised_tok/s={d:.2}{s}\n", .{
                    epoch + 1,
                    opts.epochs,
                    epoch_steps,
                    steps_per_epoch,
                    result.loss,
                    result.grad_norm,
                    timing.supervisedTokensPerSecond(target_stats),
                    if (result.optimizer_stepped) "" else " (accum)",
                });
            }

            if (opts.eval_strategy == .steps and opts.eval_steps > 0 and total_steps % opts.eval_steps == 0) {
                const eval_loss = eval_window_loss / @as(f64, @floatFromInt(eval_window_steps));
                eval_window_loss = 0.0;
                eval_window_steps = 0;
                try emitEvalEvent(allocator, opts, &metrics_jsonl.writer, epoch + 1, total_steps, eval_loss, &best_eval_loss, &trainer);
            }

            batch_start = batch_end;
        }

        const avg_epoch_loss = if (epoch_steps > 0) epoch_loss / @as(f64, @floatFromInt(epoch_steps)) else 0.0;
        cumulative_loss += avg_epoch_loss;

        // -- End-of-epoch evaluation summary --------------------------------
        // Cross-entropy loss is the primary eval metric for token
        // classification. Loss ≈ -log(p_correct), so:
        //   loss=0.5 → ~61% accuracy, loss=0.1 → ~90%, loss=0.01 → ~99%
        const approx_acc: f64 = @exp(-avg_epoch_loss) * 100.0;
        var gold_ent_count: u64 = 0;
        for (examples) |ex| gold_ent_count += ex.entities.len;

        print("  epoch {d}/{d} complete -- avg_loss={d:.6}  ~acc={d:.1}%  ({d} gold entities)\n", .{
            epoch + 1,
            opts.epochs,
            avg_epoch_loss,
            approx_acc,
            gold_ent_count,
        });
        const epoch_timing = EpochTiming{
            .epoch_wall_ns = elapsedNs(epoch_started_ns, monotonicNowNs()),
        };
        try writeEpochMetric(&metrics_jsonl.writer, epoch + 1, avg_epoch_loss, approx_acc, gold_ent_count, epoch_steps, epoch_target_stats, epoch_timing);

        if (opts.eval_strategy == .epoch) {
            eval_window_loss = 0.0;
            eval_window_steps = 0;
            try emitEvalEvent(allocator, opts, &metrics_jsonl.writer, epoch + 1, total_steps, avg_epoch_loss, &best_eval_loss, &trainer);
        }
    }

    // ------------------------------------------------------------------
    // 11. Save adapters
    // ------------------------------------------------------------------
    try trainer.syncDeviceTrainablesToHost();
    try trainer.saveAdapters(opts.out_dir);
    const autodiff_params = try collectAutodiffAdapterParams(allocator, &trainer);
    defer allocator.free(autodiff_params);
    var peft_export = try gliner2_bundle.exportAutodiffAdaptersAsPeftBundle(
        allocator,
        opts.out_dir,
        opts.model_dir,
        opts.lora_rank,
        opts.lora_alpha,
        opts.lora_dropout,
        target_patterns.items,
        autodiff_params,
    );
    defer gliner2_bundle.freeAutodiffAdapterExportSummary(allocator, &peft_export);
    var regular_params = try collectTaskHeadExportParams(allocator, &trainer, opts.num_classes, deberta_config.hidden_size);
    defer regular_params.deinit();
    var regular_export = try gliner2_bundle.exportAutodiffRegularParamsAsSafetensors(
        allocator,
        opts.out_dir,
        regular_params.params,
    );
    defer gliner2_bundle.freeAutodiffRegularParamExportSummary(allocator, &regular_export);

    const metrics_path = try std.fs.path.join(allocator, &.{ opts.out_dir, run_validation.metrics_file_name });
    defer allocator.free(metrics_path);
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = metrics_path, .data = metrics_jsonl.written() });

    const final_avg = if (opts.epochs > 0) cumulative_loss / @as(f64, @floatFromInt(opts.epochs)) else 0.0;
    try writeTrainingManifest(allocator, opts, backendLabel(selected_backend), deberta_config.hidden_size, effective_num_classes, entity_types, resolved_span_label_positive_weights, examples.len, total_steps, final_avg, trainer.lora_params.items.len, peft_export.exported_tensor_count, regular_export.exported_tensor_count, regular_trainable_params, resolved_target_patterns, run_target_stats);
    print("\nLoRA adapters saved to {s}\n", .{opts.out_dir});

    print("training complete -- {d} total steps, final avg loss={d:.6}\n", .{ total_steps, final_avg });
}

fn writeStepMetric(
    writer: *std.Io.Writer,
    epoch: usize,
    global_step: u64,
    epoch_step: u64,
    loss: f32,
    grad_norm: f32,
    optimizer_stepped: bool,
    target_stats: BatchTargetStats,
    timing: StepTiming,
) !void {
    try std.json.Stringify.value(.{
        .event = "step",
        .epoch = epoch,
        .step = global_step,
        .epoch_step = epoch_step,
        .loss = loss,
        .grad_norm = grad_norm,
        .optimizer_stepped = optimizer_stepped,
        .supervised_token_count = target_stats.supervised_token_count,
        .entity_token_count = target_stats.entity_token_count,
        .ignored_token_count = target_stats.ignored_token_count,
        .entity_token_rate = target_stats.entityTokenRate(),
        .entity_label_positive_counts = target_stats.positiveCounts(),
        .target_build_ms = nsToMillis(timing.target_build_ns),
        .train_step_ms = nsToMillis(timing.train_step_ns),
        .step_wall_ms = nsToMillis(timing.step_wall_ns),
        .graph_build_ms = nsToMillis(timing.profile.graph_build_ns),
        .runtime_input_ms = nsToMillis(timing.profile.runtime_input_ns),
        .compile_ms = nsToMillis(timing.profile.compile_ns),
        .autodiff_ms = nsToMillis(timing.profile.autodiff_ns),
        .execute_ms = nsToMillis(timing.profile.execute_ns),
        .extract_ms = nsToMillis(timing.profile.extract_ns),
        .optimizer_update_ms = nsToMillis(timing.profile.optimizer_update_ns),
        .device_optimizer_ms = nsToMillis(timing.profile.device_optimizer_ns),
        .optimizer_backend = @tagName(timing.profile.optimizer_backend),
        .device_trainable_transfer_count = timing.profile.device_resident_transfer_count,
        .device_resident_transfer_count = timing.profile.device_resident_transfer_count,
        .device_trainable_bytes = timing.profile.device_trainable_bytes,
        .graph_executor_fallback_reason = timing.profile.graph_executor_fallback_reason,
        .graph_executor_partitions = timing.profile.graph_executor_partitions,
        .graph_executor_command_dispatches = timing.profile.graph_executor_command_dispatches,
        .graph_executor_planned_dispatches = timing.profile.graph_executor_planned_dispatches,
        .graph_executor_interpreter_fallbacks = timing.profile.graph_executor_interpreter_fallbacks,
        .graph_executor_host_outputs = timing.profile.graph_executor_host_outputs,
        .graph_executor_device_outputs = timing.profile.graph_executor_device_outputs,
        .graph_executor_regions = timing.profile.graph_executor_regions,
        .graph_executor_region_ops = timing.profile.graph_executor_region_ops,
        .graph_executor_runtime_region_dispatches = timing.profile.graph_executor_runtime_region_dispatches,
        .graph_executor_runtime_region_fallbacks = timing.profile.graph_executor_runtime_region_fallbacks,
        .graph_executor_runtime_region_active_regions = timing.profile.graph_executor_runtime_region_active_regions,
        .graph_executor_runtime_region_covered_nodes = timing.profile.graph_executor_runtime_region_covered_nodes,
        .graph_executor_runtime_region_elided_nodes = timing.profile.graph_executor_runtime_region_elided_nodes,
        .graph_executor_runtime_region_plan_compiles = timing.profile.graph_executor_runtime_region_plan_compiles,
        .graph_executor_runtime_region_plan_reuses = timing.profile.graph_executor_runtime_region_plan_reuses,
        .graph_executor_runtime_frame_candidates = timing.profile.graph_executor_runtime_frame_candidates,
        .graph_executor_runtime_frame_eligible = timing.profile.graph_executor_runtime_frame_eligible,
        .graph_executor_plan_build_ms = nsToMillis(timing.profile.graph_executor_plan_build_ns),
        .graph_executor_buffer_plan_build_ms = nsToMillis(timing.profile.graph_executor_buffer_plan_build_ns),
        .graph_executor_plan_cache_hits = timing.profile.graph_executor_plan_cache_hits,
        .graph_executor_plan_cache_misses = timing.profile.graph_executor_plan_cache_misses,
        .metal_frame_wait_ms = nsToMillis(timing.profile.metal_frame_wait_ns),
        .metal_frame_gpu_ms = nsToMillis(timing.profile.metal_frame_gpu_ns),
        .metal_tensor_device_owned_live_bytes = timing.profile.metal_tensor_device_owned_live_bytes,
        .metal_tensor_device_owned_peak_live_bytes = timing.profile.metal_tensor_device_owned_peak_live_bytes,
        .metal_runtime_total_bytes = timing.profile.metal_runtime_total_bytes,
        .metal_runtime_frame_retained_bytes = timing.profile.metal_runtime_frame_retained_bytes,
        .metal_runtime_reuse_pool_bytes = timing.profile.metal_runtime_reuse_pool_bytes,
        .metal_runtime_reuse_pool_slots = timing.profile.metal_runtime_reuse_pool_slots,
        .metal_runtime_reuse_pool_peak_slots = timing.profile.metal_runtime_reuse_pool_peak_slots,
        .metal_runtime_reuse_alloc_delta = timing.profile.metal_runtime_reuse_alloc_delta,
        .metal_runtime_reuse_hit_delta = timing.profile.metal_runtime_reuse_hit_delta,
        .metal_runtime_reuse_hit_rate = ratio(timing.profile.metal_runtime_reuse_hit_delta, timing.profile.metal_runtime_reuse_alloc_delta),
        .metal_last_frame_compute_encoders = timing.profile.metal_last_frame_compute_encoders,
        .metal_last_frame_blit_encoders = timing.profile.metal_last_frame_blit_encoders,
        .metal_last_frame_planned_scopes = timing.profile.metal_last_frame_planned_scopes,
        .metal_last_frame_planned_barriers = timing.profile.metal_last_frame_planned_barriers,
        .metal_last_frame_planned_command_ops = timing.profile.metal_last_frame_planned_command_ops,
        .metal_deberta_encoder_plan_attempts = timing.profile.metal_deberta_encoder_plan_attempts,
        .metal_deberta_encoder_plan_successes = timing.profile.metal_deberta_encoder_plan_successes,
        .metal_deberta_encoder_plan_reuses = timing.profile.metal_deberta_encoder_plan_reuses,
        .metal_deberta_encoder_plan_failures = timing.profile.metal_deberta_encoder_plan_failures,
        .metal_deberta_encoder_layer_attempts = timing.profile.metal_deberta_encoder_layer_attempts,
        .metal_deberta_encoder_layer_successes = timing.profile.metal_deberta_encoder_layer_successes,
        .metal_deberta_encoder_layer_fallbacks = timing.profile.metal_deberta_encoder_layer_fallbacks,
        .metal_deberta_relative_qk_pair_calls = timing.profile.metal_deberta_relative_qk_pair_calls,
        .metal_deberta_relative_qk_pair_fallbacks = timing.profile.metal_deberta_relative_qk_pair_fallbacks,
        .metal_deberta_ffn_fused_calls = timing.profile.metal_deberta_ffn_fused_calls,
        .metal_deberta_ffn_fused_mps_matmuls = timing.profile.metal_deberta_ffn_fused_mps_matmuls,
        .metal_deberta_ffn_fused_fallbacks = timing.profile.metal_deberta_ffn_fused_fallbacks,
        .metal_deberta_attention_flash_calls = timing.profile.metal_deberta_attention_flash_calls,
        .metal_deberta_attention_gemm_calls = timing.profile.metal_deberta_attention_gemm_calls,
        .metal_deberta_attention_gemm_fallbacks = timing.profile.metal_deberta_attention_gemm_fallbacks,
        .metal_deberta_attention_legacy_calls = timing.profile.metal_deberta_attention_legacy_calls,
        .trainer_total_ms = nsToMillis(timing.profile.total_ns),
        .peak_resident_bytes = timing.profile.peak_resident_bytes,
        .supervised_tokens_per_second = timing.supervisedTokensPerSecond(target_stats),
    }, .{}, writer);
    try writer.writeByte('\n');
}

fn writeEpochMetric(
    writer: *std.Io.Writer,
    epoch: usize,
    avg_loss: f64,
    approx_accuracy_percent: f64,
    gold_entities: u64,
    steps: u64,
    target_stats: BatchTargetStats,
    timing: EpochTiming,
) !void {
    try std.json.Stringify.value(.{
        .event = "epoch",
        .epoch = epoch,
        .avg_loss = avg_loss,
        .approx_accuracy_percent = approx_accuracy_percent,
        .gold_entities = gold_entities,
        .steps = steps,
        .supervised_token_count = target_stats.supervised_token_count,
        .entity_token_count = target_stats.entity_token_count,
        .ignored_token_count = target_stats.ignored_token_count,
        .entity_token_rate = target_stats.entityTokenRate(),
        .entity_label_positive_counts = target_stats.positiveCounts(),
        .epoch_wall_ms = nsToMillis(timing.epoch_wall_ns),
        .supervised_tokens_per_second = tokensPerSecond(target_stats.supervised_token_count, timing.epoch_wall_ns),
    }, .{}, writer);
    try writer.writeByte('\n');
}

/// Emit one eval event (stdout text or JSONL, plus the metrics JSONL file)
/// and, when --save-best is set and the eval metric improved, snapshot the
/// current adapters into `<out_dir>/best`.
///
/// LIMITATION: this CLI has no held-out eval dataset input yet, so the eval
/// metric is the average training loss over the eval window.
fn emitEvalEvent(
    allocator: std.mem.Allocator,
    opts: Options,
    metrics_writer: *std.Io.Writer,
    epoch: usize,
    global_step: u64,
    eval_loss: f64,
    best_eval_loss: *f64,
    trainer: *real_autodiff.RealAutodiffTrainer,
) !void {
    const improved = eval_loss < best_eval_loss.*;
    if (improved) best_eval_loss.* = eval_loss;

    switch (opts.report_to) {
        .jsonl => print(
            "{{\"event\":\"eval\",\"step\":{d},\"epoch\":{d},\"loss\":{d:.9},\"best_loss\":{d:.9},\"improved\":{},\"lr\":{e:.6}}}\n",
            .{ global_step, epoch, eval_loss, best_eval_loss.*, improved, opts.learning_rate },
        ),
        .stdout => print(
            "  eval [epoch {d}] step {d}  avg_train_loss={d:.6}  best={d:.6}{s}\n",
            .{ epoch, global_step, eval_loss, best_eval_loss.*, if (improved) " (improved)" else "" },
        ),
    }
    try std.json.Stringify.value(.{
        .event = "eval",
        .epoch = epoch,
        .step = global_step,
        .loss = eval_loss,
        .best_loss = best_eval_loss.*,
        .improved = improved,
        .eval_metric = "avg_train_loss",
    }, .{}, metrics_writer);
    try metrics_writer.writeByte('\n');

    if (opts.save_best and improved) {
        const best_dir = try std.fs.path.join(allocator, &.{ opts.out_dir, "best" });
        defer allocator.free(best_dir);
        try compat.cwd().createDirPath(compat.io(), best_dir);
        try trainer.syncDeviceTrainablesToHost();
        try trainer.saveAdapters(best_dir);
        print("  saved best adapters (loss={d:.6}) to {s}\n", .{ eval_loss, best_dir });
    }
}

fn writeTrainingManifest(
    allocator: std.mem.Allocator,
    opts: Options,
    backend_label: []const u8,
    hidden_size: u32,
    num_classes: u32,
    entity_labels: []const []const u8,
    span_label_positive_weights: []const f32,
    example_count: usize,
    total_steps: u64,
    final_avg_loss: f64,
    adapter_parameter_file_count: usize,
    peft_adapter_tensor_count: usize,
    regular_trainable_tensor_count: usize,
    regular_trainable_params: []const []const u8,
    resolved_lora_targets: []const []const u8,
    target_stats: BatchTargetStats,
) !void {
    const manifest_path = try std.fs.path.join(allocator, &.{ opts.out_dir, run_validation.manifest_file_name });
    defer allocator.free(manifest_path);

    var buffer: std.Io.Writer.Allocating = .init(allocator);
    defer buffer.deinit();
    try std.json.Stringify.value(.{
        .schema_version = "gliner2_autodiff_training/v1",
        .artifact_family_version = "gliner2_autodiff_adapter/v1",
        .model_dir = opts.model_dir,
        .backend = backend_label,
        .compiled_required = opts.compiled_required,
        .train_data = opts.train_data,
        .out_dir = opts.out_dir,
        .metrics_file = run_validation.metrics_file_name,
        .adapter_parameter_format = "real_autodiff_bin/v1",
        .adapter_parameter_file_count = adapter_parameter_file_count,
        .peft_adapter_checkpoint = gliner2_bundle.adapter_checkpoint_file_name,
        .peft_adapter_config = gliner2_bundle.adapter_config_file_name,
        .peft_adapter_tensor_count = peft_adapter_tensor_count,
        .regular_trainable_checkpoint = gliner2_bundle.task_head_checkpoint_file_name,
        .regular_trainable_tensor_count = regular_trainable_tensor_count,
        .regular_trainable_params = regular_trainable_params,
        .lora_only_trainables = opts.lora_only_trainables,
        .deterministic = opts.deterministic,
        .eval_strategy = @tagName(opts.eval_strategy),
        .eval_steps = opts.eval_steps,
        .save_best = opts.save_best,
        .epochs = opts.epochs,
        .batch_size = opts.batch_size,
        .seq_len = opts.seq_len,
        .learning_rate = opts.learning_rate,
        .lora_rank = opts.lora_rank,
        .lora_alpha = opts.lora_alpha,
        .lora_dropout = opts.lora_dropout,
        .lora_targets = opts.lora_targets,
        .resolved_lora_targets = resolved_lora_targets,
        .num_classes = num_classes,
        .objective = objectiveName(opts.objective),
        .max_span_width = opts.max_span_width,
        .span_loss = spanLossName(opts.span_loss),
        .span_loss_reduction = spanLossReductionName(opts.span_loss_reduction),
        .span_positive_weight = opts.span_positive_weight,
        .span_label_positive_weights = span_label_positive_weights,
        .span_negative_weight = opts.span_negative_weight,
        .span_hard_negative_weight = opts.span_hard_negative_weight,
        .hidden_size = hidden_size,
        .entity_labels = entity_labels,
        .entity_label_count = entity_labels.len,
        .entity_label_positive_counts = target_stats.positiveCounts(),
        .supervised_token_count = target_stats.supervised_token_count,
        .entity_token_count = target_stats.entity_token_count,
        .ignored_token_count = target_stats.ignored_token_count,
        .entity_token_rate = target_stats.entityTokenRate(),
        .max_examples = opts.max_examples,
        .max_grad_norm = opts.max_grad_norm,
        .grad_accum = opts.grad_accum,
        .seed = opts.seed,
        .example_count = example_count,
        .total_steps = total_steps,
        .final_avg_loss = final_avg_loss,
    }, .{ .whitespace = .indent_2 }, &buffer.writer);
    try buffer.writer.writeByte('\n');
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = manifest_path, .data = buffer.written() });
}

/// Mirror torch AdamW per-parameter step semantics for LoRA pairs whose
/// gradient is exactly zero while their module still participates in the
/// loss.
///
/// With the standard LoRA init (lora_B = 0) the lora_A gradient is exactly
/// zero on the first optimizer step (grad_A = B^T · δ · x^T with B = 0), but
/// torch's autograd still materializes that zero gradient, so torch AdamW
/// creates optimizer state for lora_A and advances its per-parameter `step`
/// counter on step 1. The Zig trainer skips all-zero gradients entirely
/// (matching torch's behavior for grad=None params whose module never ran),
/// which leaves lora_A's step counter one behind its sibling lora_B. When the
/// first nonzero lora_A gradient arrives on step 2, the two sides then apply
/// different Adam bias corrections (t=1 vs t=2), producing ~0.25·lr per-element
/// weight divergence that compounds every following step.
///
/// This sync replays the missed zero-gradient AdamW steps for any lora_A
/// whose sibling lora_B has stepped (nonzero grad ⇒ the module participated
/// in this batch's loss, which is exactly torch's grad-is-not-None criterion).
/// A zero-gradient AdamW step leaves m/v at zero and only applies decoupled
/// weight decay + the step-count increment — identical to what torch does.
fn syncZeroGradLoraOptimizerSteps(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    lr: f32,
) !void {
    // Diagnostic escape hatch: lets the parity harness reproduce the
    // pre-fix divergence without a rebuild.
    if (std.c.getenv("TERMITE_GLINER2_DISABLE_ZERO_GRAD_STEP_SYNC")) |raw| {
        const value = std.mem.span(raw);
        if (value.len > 0 and !std.mem.eql(u8, value, "0")) return;
    }
    const suffix_b = ".lora_B";
    const suffix_a = ".lora_A";
    for (trainer.lora_params.items) |*slot_b| {
        if (!std.mem.endsWith(u8, slot_b.name, suffix_b)) continue;
        // Device-resident optimizer paths (Metal/MLX) keep m/v on device and
        // never consult the host OptimizerState; skip them here.
        if (slot_b.device != null) continue;
        const target_steps = blk: {
            const state_b = trainer.optimizer_state.param_states.getPtr(slot_b.name) orelse continue;
            break :blk state_b.step_count;
        };
        if (target_steps == 0) continue;
        const base = slot_b.name[0 .. slot_b.name.len - suffix_b.len];
        var sibling: ?*real_autodiff.RealAutodiffTrainer.ParamSlot = null;
        for (trainer.lora_params.items) |*candidate| {
            if (candidate.name.len != base.len + suffix_a.len) continue;
            if (!std.mem.startsWith(u8, candidate.name, base)) continue;
            if (!std.mem.endsWith(u8, candidate.name, suffix_a)) continue;
            sibling = candidate;
            break;
        }
        const slot_a = sibling orelse continue;
        if (slot_a.device != null) continue;
        const current_steps = if (trainer.optimizer_state.param_states.getPtr(slot_a.name)) |state_a|
            state_a.step_count
        else
            0;
        if (current_steps >= target_steps) continue;
        const zero_grad = try allocator.alloc(f32, slot_a.weights.len);
        defer allocator.free(zero_grad);
        @memset(zero_grad, 0);
        var remaining = target_steps - current_steps;
        while (remaining > 0) : (remaining -= 1) {
            try optimizers.step(
                .{ .adamw = trainer.config.optimizer },
                &trainer.optimizer_state,
                lr,
                slot_a.name,
                slot_a.weights,
                zero_grad,
            );
        }
    }
}

/// Emit one `GLINER2_OPT_PARITY` JSON line per optimizer step with, for every
/// host-resident LoRA parameter: the post-update weights, Adam m/v state and
/// per-parameter step counter (first 8 elements + f64 abs-sum per tensor).
/// Consumed by scripts/compare_gliner2_lora_python_zig.py --dump-optimizer-parity.
fn printOptimizerParityDump(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    step: u64,
) !void {
    var buffer: std.Io.Writer.Allocating = .init(allocator);
    defer buffer.deinit();
    const w = &buffer.writer;
    try w.print("GLINER2_OPT_PARITY {{\"step\":{d},\"tensors\":[", .{step});
    var first = true;
    for (trainer.lora_params.items) |slot| {
        if (slot.device != null) continue;
        const peft_name = gliner2_bundle.autodiffParamNameToPeftName(allocator, slot.name) catch continue;
        defer allocator.free(peft_name);
        if (!first) try w.writeAll(",");
        first = false;
        const state = trainer.optimizer_state.param_states.getPtr(slot.name);
        const step_count: u32 = if (state) |s| s.step_count else 0;
        try w.print("{{\"name\":\"{s}\",\"step_count\":{d}", .{ peft_name, step_count });
        try writeOptParityTensor(w, "weight", slot.weights);
        try writeOptParityTensor(w, "m", if (state) |s| s.m else &.{});
        try writeOptParityTensor(w, "v", if (state) |s| s.v else &.{});
        try w.writeAll("}");
    }
    try w.writeAll("]}\n");
    print("{s}", .{buffer.written()});
}

fn writeOptParityTensor(w: *std.Io.Writer, name: []const u8, values: []const f32) !void {
    try w.print(",\"{s}\":[", .{name});
    const head_len = @min(values.len, 8);
    for (values[0..head_len], 0..) |v, idx| {
        if (idx > 0) try w.writeAll(",");
        try w.print("{e}", .{v});
    }
    var abs_sum: f64 = 0.0;
    for (values) |v| abs_sum += @abs(@as(f64, v));
    try w.print("],\"{s}_abs_sum\":{e}", .{ name, abs_sum });
}

fn collectAutodiffAdapterParams(
    allocator: std.mem.Allocator,
    trainer: *const real_autodiff.RealAutodiffTrainer,
) ![]gliner2_bundle.AutodiffAdapterParam {
    const params = try allocator.alloc(gliner2_bundle.AutodiffAdapterParam, trainer.lora_params.items.len);
    for (trainer.lora_params.items, 0..) |slot, idx| {
        params[idx] = .{
            .name = slot.name,
            .dims = slot.dims,
            .weights = slot.weights,
        };
    }
    return params;
}

fn collectRegularTrainableParams(
    allocator: std.mem.Allocator,
    trainer: *const real_autodiff.RealAutodiffTrainer,
) ![]gliner2_bundle.AutodiffAdapterParam {
    const params = try allocator.alloc(gliner2_bundle.AutodiffAdapterParam, trainer.regular_params.items.len);
    for (trainer.regular_params.items, 0..) |slot, idx| {
        params[idx] = .{
            .name = slot.name,
            .dims = slot.dims,
            .weights = slot.weights,
        };
    }
    return params;
}

fn ensureTrainerGraphBuiltFromFirstBatch(
    allocator: std.mem.Allocator,
    opts: Options,
    effective_seq_len: usize,
    tokenizer: *const gliner2_data.Tokenizer,
    entity_types: []const []const u8,
    examples: []const gliner2_data.Example,
    training_records: []const gliner2_data.UpstreamRecord,
    label_map: *const std.StringHashMapUnmanaged(u32),
    effective_num_classes: u32,
    use_label_positive_weights: bool,
    resolved_span_label_positive_weights: []const f32,
    input_ids: []i64,
    attention_mask: []f32,
    targets_buf: []f32,
    trainer: *real_autodiff.RealAutodiffTrainer,
    gliner_ctx: *gliner2_autodiff.GlinerAutodiffCtx,
) !void {
    if (examples.len == 0) return error.NoTrainingExamples;
    // The graph is compiled for the configured batch size, so pad a dataset
    // smaller than one batch by repeating the last example — identical to
    // the padding in the main training loop.
    const ab: usize = opts.batch_size;
    const real_batch = @min(ab, examples.len);
    const actual_batch: u32 = @intCast(ab);
    const sl: usize = effective_seq_len;
    const nc: usize = effective_num_classes;

    var batch_examples = try allocator.alloc(gliner2_data.Example, ab);
    defer allocator.free(batch_examples);
    var batch_records = try allocator.alloc(gliner2_data.UpstreamRecord, ab);
    defer allocator.free(batch_records);
    for (0..ab) |slot| {
        const src = @min(slot, real_batch - 1);
        batch_examples[slot] = examples[src];
        if (opts.objective == .gliner2_total_loss) batch_records[slot] = training_records[src];
    }

    var targets_shape: ml.graph.Shape = undefined;
    var target_slice: []const f32 = undefined;
    switch (opts.objective) {
        .token => {
            _ = fillBatchBuffers(
                allocator,
                tokenizer,
                entity_types,
                batch_examples,
                @intCast(sl),
                effective_num_classes,
                label_map,
                input_ids,
                attention_mask,
                targets_buf[0 .. ab * sl * nc],
            );
            if (real_batch < ab) @memset(targets_buf[real_batch * sl * nc .. ab * sl * nc], 0.0);
            targets_shape = gliner2_autodiff.tokenTargetsShape(actual_batch, @intCast(sl), effective_num_classes);
            target_slice = targets_buf[0 .. ab * sl * nc];
        },
        .span_start, .gliner2_total_loss => {
            var encoded = if (opts.objective == .gliner2_total_loss)
                try gliner2_data.buildUpstreamTaskBatch(
                    allocator,
                    tokenizer,
                    batch_records,
                    entity_types,
                    sl,
                    opts.max_span_width,
                    ab,
                )
            else
                try gliner2_data.buildSimpleBatch(
                    allocator,
                    tokenizer,
                    batch_examples,
                    entity_types,
                    sl,
                    opts.max_span_width,
                    ab,
                );
            defer encoded.deinit();

            if (encoded.input_ids.len != ab * sl or encoded.attention_mask.len != ab * sl) return error.InvalidGlinerBatchShape;
            for (0..ab * sl) |i| {
                input_ids[i] = encoded.input_ids[i];
                attention_mask[i] = @floatFromInt(encoded.attention_mask[i]);
            }

            const width = if (opts.objective == .gliner2_total_loss)
                gliner2_autodiff.gliner2TotalLossTargetWidthEx(encoded.num_entity_types, gliner_ctx.config.structure_max_instances)
            else if (use_label_positive_weights)
                gliner2_autodiff.weightedSpanStartTargetWidth(encoded.num_entity_types)
            else
                gliner2_autodiff.spanStartTargetWidth(encoded.num_entity_types);
            const target_len = encoded.batch_size * encoded.max_spans * width;
            _ = if (opts.objective == .gliner2_total_loss)
                try fillGliner2TotalLossTargetsFromRecords(
                    allocator,
                    &encoded,
                    batch_records,
                    entity_types,
                    gliner_ctx.config.structure_max_instances,
                    targets_buf[0..target_len],
                )
            else
                try gliner2_autodiff.fillSpanStartTargetsFromEncodedBatchWithOptions(
                    &encoded,
                    .{
                        .positive_weights_by_entity_type = if (use_label_positive_weights) resolved_span_label_positive_weights else null,
                        .hard_negative_weight = opts.span_hard_negative_weight,
                    },
                    targets_buf[0..target_len],
                );
            if (real_batch < ab) try gliner2_autodiff.zeroPaddedSpanTargetRows(
                targets_buf[0..target_len],
                opts.objective,
                encoded.num_entity_types,
                encoded.max_spans,
                real_batch,
                ab,
                use_label_positive_weights,
                gliner_ctx.config.structure_max_instances,
            );
            targets_shape = if (opts.objective == .gliner2_total_loss)
                gliner2_autodiff.gliner2TotalLossTargetsShapeEx(
                    actual_batch,
                    @intCast(encoded.max_spans),
                    @intCast(encoded.num_entity_types),
                    gliner_ctx.config.structure_max_instances,
                )
            else if (use_label_positive_weights)
                gliner2_autodiff.weightedSpanStartTargetsShape(
                    actual_batch,
                    @intCast(encoded.max_spans),
                    @intCast(encoded.num_entity_types),
                )
            else
                gliner2_autodiff.spanStartTargetsShape(
                    actual_batch,
                    @intCast(encoded.max_spans),
                    @intCast(encoded.num_entity_types),
                );
            target_slice = targets_buf[0..target_len];
        },
    }

    const trainer_input = gliner2_autodiff.makeTrainerInput(
        gliner_ctx,
        input_ids[0 .. ab * sl],
        attention_mask[0 .. ab * sl],
        target_slice,
        targets_shape,
        actual_batch,
        @intCast(sl),
    );
    try trainer.ensureGraphBuilt(trainer_input);
}

fn loadPeftAdaptersIntoTrainer(
    allocator: std.mem.Allocator,
    adapter_checkpoint_path: []const u8,
    trainer: *real_autodiff.RealAutodiffTrainer,
) !void {
    var reader = try safetensors.MMapReader.openFileAbsolute(allocator, adapter_checkpoint_path);
    defer reader.deinit();
    for (trainer.lora_params.items) |*slot| {
        const peft_name = try autodiffSlotNameToOfficialPeftName(allocator, slot.name);
        defer allocator.free(peft_name);
        var tensor = reader.readTensor(peft_name) catch blk: {
            const zig_peft_name = try autodiffSlotNameToZigPeftName(allocator, slot.name);
            defer allocator.free(zig_peft_name);
            break :blk try reader.readTensor(zig_peft_name);
        };
        defer tensor.deinit();
        if (tensor.elementCount() != slot.weights.len) return error.AdapterTensorShapeMismatch;
        try copyTensorF32Into(slot.weights, &tensor);
        @memset(slot.grad_accum, 0.0);
    }
}

fn copyTensorF32Into(dst: []f32, tensor: *const Tensor) !void {
    if (tensor.dtype != .f32) return error.AdapterTensorDTypeMismatch;
    if (tensor.data.len != dst.len * @sizeOf(f32)) return error.AdapterTensorShapeMismatch;
    for (dst, 0..) |*value, idx| {
        const raw = tensor.data[idx * @sizeOf(f32) ..][0..@sizeOf(f32)];
        value.* = @bitCast(std.mem.readInt(u32, raw, .little));
    }
}

fn autodiffSlotNameToOfficialPeftName(allocator: std.mem.Allocator, name: []const u8) ![]const u8 {
    if (std.mem.endsWith(u8, name, ".lora_A")) {
        const base = name[0 .. name.len - ".lora_A".len];
        return autodiffBaseToPeftName(allocator, tensorBaseName(base), "lora_A", false);
    }
    if (std.mem.endsWith(u8, name, ".lora_B")) {
        const base = name[0 .. name.len - ".lora_B".len];
        return autodiffBaseToPeftName(allocator, tensorBaseName(base), "lora_B", false);
    }
    return error.InvalidAutodiffAdapterName;
}

fn autodiffSlotNameToZigPeftName(allocator: std.mem.Allocator, name: []const u8) ![]const u8 {
    if (std.mem.endsWith(u8, name, ".lora_A")) {
        const base = name[0 .. name.len - ".lora_A".len];
        return autodiffBaseToPeftName(allocator, tensorBaseName(base), "lora_A", true);
    }
    if (std.mem.endsWith(u8, name, ".lora_B")) {
        const base = name[0 .. name.len - ".lora_B".len];
        return autodiffBaseToPeftName(allocator, tensorBaseName(base), "lora_B", true);
    }
    return error.InvalidAutodiffAdapterName;
}

fn autodiffBaseToPeftName(allocator: std.mem.Allocator, base_no_weight: []const u8, adapter_name: []const u8, zig_weight_suffix: bool) ![]const u8 {
    const suffix = if (zig_weight_suffix) ".weight" else "";
    if (std.mem.startsWith(u8, base_no_weight, "encoder.layer.")) {
        return std.fmt.allocPrint(allocator, "encoder.{s}.{s}{s}", .{ base_no_weight, adapter_name, suffix });
    }
    return std.fmt.allocPrint(allocator, "{s}.{s}{s}", .{ base_no_weight, adapter_name, suffix });
}

fn tensorBaseName(tensor_name: []const u8) []const u8 {
    if (std.mem.endsWith(u8, tensor_name, ".weight")) return tensor_name[0 .. tensor_name.len - ".weight".len];
    return tensor_name;
}

const TaskHeadExportParams = struct {
    allocator: std.mem.Allocator,
    params: []gliner2_bundle.AutodiffAdapterParam,
    owned_dims: [][]i32,
    owned_weights: [][]f32,

    fn deinit(self: *TaskHeadExportParams) void {
        for (self.owned_dims) |dims| self.allocator.free(dims);
        for (self.owned_weights) |weights| self.allocator.free(weights);
        self.allocator.free(self.owned_dims);
        self.allocator.free(self.owned_weights);
        self.allocator.free(self.params);
        self.* = undefined;
    }
};

fn collectTaskHeadExportParams(
    allocator: std.mem.Allocator,
    trainer: *const real_autodiff.RealAutodiffTrainer,
    num_classes: u32,
    hidden_size: u32,
) !TaskHeadExportParams {
    const source_names = [_][]const u8{ "task_classifier.weight", "task_classifier.bias" };
    const export_names = [_][]const u8{ "classifier.weight", "classifier.bias" };
    var params = try allocator.alloc(gliner2_bundle.AutodiffAdapterParam, source_names.len);
    errdefer allocator.free(params);
    var owned_dims = try allocator.alloc([]i32, source_names.len);
    errdefer allocator.free(owned_dims);
    var owned_weights = try allocator.alloc([]f32, source_names.len);
    errdefer allocator.free(owned_weights);
    for (owned_dims) |*dims| dims.* = &.{};
    for (owned_weights) |*weights| weights.* = &.{};
    errdefer {
        for (owned_dims) |dims| allocator.free(dims);
        for (owned_weights) |weights| allocator.free(weights);
    }

    for (source_names, 0..) |name, idx| {
        if (findRegularParamSlot(trainer, name)) |slot| {
            owned_dims[idx] = try allocator.dupe(i32, slot.dims);
            owned_weights[idx] = try allocator.dupe(f32, slot.weights);
        } else {
            const ct = try trainer.compute_backend.getWeight(name);
            defer trainer.compute_backend.free(ct);
            owned_weights[idx] = try trainer.compute_backend.toFloat32(ct, allocator);
            owned_dims[idx] = try inferTaskHeadDims(allocator, name, owned_weights[idx].len, num_classes, hidden_size);
        }
        params[idx] = .{
            .name = export_names[idx],
            .dims = owned_dims[idx],
            .weights = owned_weights[idx],
        };
    }

    return .{
        .allocator = allocator,
        .params = params,
        .owned_dims = owned_dims,
        .owned_weights = owned_weights,
    };
}

fn findRegularParamSlot(
    trainer: *const real_autodiff.RealAutodiffTrainer,
    name: []const u8,
) ?*const real_autodiff.RealAutodiffTrainer.ParamSlot {
    for (trainer.regular_params.items) |*slot| {
        if (std.mem.eql(u8, slot.name, name)) return slot;
    }
    return null;
}

fn inferTaskHeadDims(
    allocator: std.mem.Allocator,
    name: []const u8,
    element_count: usize,
    num_classes: u32,
    hidden_size: u32,
) ![]i32 {
    if (std.mem.eql(u8, name, "task_classifier.bias")) {
        if (element_count != num_classes) return error.TaskHeadParameterShapeMismatch;
        const dims = try allocator.alloc(i32, 1);
        dims[0] = @intCast(num_classes);
        return dims;
    }
    if (std.mem.eql(u8, name, "task_classifier.weight")) {
        const expected = @as(usize, @intCast(num_classes)) * @as(usize, @intCast(hidden_size));
        if (element_count != expected) return error.TaskHeadParameterShapeMismatch;
        const dims = try allocator.alloc(i32, 2);
        dims[0] = @intCast(num_classes);
        dims[1] = @intCast(hidden_size);
        return dims;
    }
    return error.InvalidTaskHeadParameter;
}

fn deinitNativeWeightStore(allocator: std.mem.Allocator, weight_store: *native_compute.WeightStore) void {
    var it = weight_store.resident_weights.iterator();
    while (it.next()) |entry| {
        allocator.free(entry.key_ptr.*);
        entry.value_ptr.deinit();
    }
    weight_store.resident_weights.deinit(allocator);
    weight_store.lazy_weights.deinit(allocator);
}

// ---------------------------------------------------------------------------
// DeBERTa config parsing
// ---------------------------------------------------------------------------

/// Subset of DeBERTa config.json fields needed for the graph builder.
const DebertaJsonConfig = struct {
    vocab_size: u32 = 128100,
    hidden_size: u32 = 768,
    num_hidden_layers: u32 = 12,
    num_attention_heads: u32 = 12,
    intermediate_size: u32 = 3072,
    max_position_embeddings: u32 = 512,
    position_buckets: u32 = 256,
    layer_norm_eps: f32 = 1e-7,
};

fn debertaArchConfigFromJson(config: DebertaJsonConfig) deberta_arch.Config {
    return .{
        .vocab_size = config.vocab_size,
        .hidden_size = config.hidden_size,
        .num_hidden_layers = config.num_hidden_layers,
        .num_attention_heads = config.num_attention_heads,
        .intermediate_size = config.intermediate_size,
        .max_position_embeddings = config.max_position_embeddings,
        .position_buckets = config.position_buckets,
        .layer_norm_eps = config.layer_norm_eps,
    };
}

fn parseDebertaConfig(allocator: std.mem.Allocator, json_bytes: []const u8) !DebertaJsonConfig {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{});
    defer parsed.deinit();

    const obj = parsed.value.object;
    var config = DebertaJsonConfig{};

    if (obj.get("vocab_size")) |v| config.vocab_size = jsonU32(v) orelse config.vocab_size;
    if (obj.get("hidden_size")) |v| config.hidden_size = jsonU32(v) orelse config.hidden_size;
    if (obj.get("num_hidden_layers")) |v| config.num_hidden_layers = jsonU32(v) orelse config.num_hidden_layers;
    if (obj.get("num_attention_heads")) |v| config.num_attention_heads = jsonU32(v) orelse config.num_attention_heads;
    if (obj.get("intermediate_size")) |v| config.intermediate_size = jsonU32(v) orelse config.intermediate_size;
    if (obj.get("max_position_embeddings")) |v| config.max_position_embeddings = jsonU32(v) orelse config.max_position_embeddings;
    if (obj.get("position_buckets")) |v| config.position_buckets = jsonU32(v) orelse config.position_buckets;
    if (obj.get("layer_norm_eps")) |v| config.layer_norm_eps = jsonF32(v) orelse config.layer_norm_eps;

    return config;
}

fn jsonU32(val: std.json.Value) ?u32 {
    return switch (val) {
        .integer => |i| @intCast(i),
        else => null,
    };
}

fn jsonF32(val: std.json.Value) ?f32 {
    return switch (val) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

// ---------------------------------------------------------------------------
// Batch buffer construction (real HF tokenizer + entity targets)
// ---------------------------------------------------------------------------

const BatchTargetStats = struct {
    supervised_token_count: u64 = 0,
    entity_token_count: u64 = 0,
    ignored_token_count: u64 = 0,
    entity_type_count: usize = 0,
    positive_counts_by_entity_type: [gliner2_autodiff.max_span_start_entity_types]u64 = [_]u64{0} ** gliner2_autodiff.max_span_start_entity_types,

    fn entityTokenRate(self: BatchTargetStats) f64 {
        if (self.supervised_token_count == 0) return 0.0;
        return @as(f64, @floatFromInt(self.entity_token_count)) /
            @as(f64, @floatFromInt(self.supervised_token_count));
    }

    fn positiveCounts(self: *const BatchTargetStats) []const u64 {
        return self.positive_counts_by_entity_type[0..self.entity_type_count];
    }

    fn add(self: *BatchTargetStats, other: BatchTargetStats) void {
        self.supervised_token_count += other.supervised_token_count;
        self.entity_token_count += other.entity_token_count;
        self.ignored_token_count += other.ignored_token_count;
        if (other.entity_type_count > self.entity_type_count) self.entity_type_count = other.entity_type_count;
        for (0..other.entity_type_count) |idx| {
            self.positive_counts_by_entity_type[idx] += other.positive_counts_by_entity_type[idx];
        }
    }

    fn fromSpanStart(stats: gliner2_autodiff.SpanStartTargetStats, num_entity_types: usize) BatchTargetStats {
        var out = BatchTargetStats{
            .supervised_token_count = stats.valid_span_count * @as(u64, @intCast(num_entity_types)),
            .entity_token_count = stats.positive_span_label_count,
            .ignored_token_count = stats.ignored_span_count * @as(u64, @intCast(num_entity_types)),
            .entity_type_count = stats.entity_type_count,
        };
        for (0..stats.entity_type_count) |idx| {
            out.positive_counts_by_entity_type[idx] = stats.positive_counts_by_entity_type[idx];
        }
        return out;
    }
};

const StepTiming = struct {
    target_build_ns: u64,
    train_step_ns: u64,
    step_wall_ns: u64,
    profile: real_autodiff.StepProfile = .{},

    fn supervisedTokensPerSecond(self: StepTiming, stats: BatchTargetStats) f64 {
        return tokensPerSecond(stats.supervised_token_count, self.step_wall_ns);
    }
};

const EpochTiming = struct {
    epoch_wall_ns: u64,
};

fn monotonicNowNs() u64 {
    var ts: std.posix.timespec = undefined;
    switch (std.posix.errno(std.posix.system.clock_gettime(.MONOTONIC, &ts))) {
        .SUCCESS => return @intCast(@as(i128, ts.sec) * std.time.ns_per_s + ts.nsec),
        else => return 0,
    }
}

fn elapsedNs(start_ns: u64, end_ns: u64) u64 {
    if (end_ns <= start_ns) return 0;
    return end_ns - start_ns;
}

fn nsToMillis(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(std.time.ns_per_ms));
}

fn tokensPerSecond(tokens: u64, ns: u64) f64 {
    if (tokens == 0 or ns == 0) return 0.0;
    const seconds = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(std.time.ns_per_s));
    return @as(f64, @floatFromInt(tokens)) / seconds;
}

fn ratio(numerator: u64, denominator: u64) f64 {
    if (denominator == 0) return 0.0;
    return @as(f64, @floatFromInt(numerator)) / @as(f64, @floatFromInt(denominator));
}

/// Fills input_ids, attention_mask, and one-hot targets for one batch
/// using the real DeBERTa-v3 Unigram tokenizer with the GLiNER2 HF
/// prompt format: [P] entity_types... [E] [SEP_TEXT] text_tokens...
///
/// Entity annotations map to word-level positions via the tokenizer's
/// `words_mask` / `first_token_positions` outputs, then to one-hot
/// targets per text-token position. Prompt, entity-label, separator, and
/// padding tokens are represented by all-zero rows so they do not contribute
/// to the token-classifier fallback loss.
fn fillBatchBuffers(
    allocator: std.mem.Allocator,
    tokenizer: *const gliner2_data.Tokenizer,
    entity_types: []const []const u8,
    batch_examples: []const gliner2_data.Example,
    seq_len: u32,
    num_classes: u32,
    label_map: *const std.StringHashMapUnmanaged(u32),
    input_ids: []i64,
    attention_mask: []f32,
    targets: []f32,
) BatchTargetStats {
    const sl: usize = seq_len;
    const nc: usize = num_classes;
    var stats = BatchTargetStats{};

    // Scratch buffers for the tokenizer (i32 outputs).
    var tok_ids_buf: [4096]i32 = undefined;
    var tok_mask_buf: [4096]i32 = undefined;
    var words_mask_buf: [4096]i32 = undefined;
    var first_pos_buf: [4096]i32 = undefined;
    var e_tok_pos_buf: [128]i32 = undefined;
    var e_tok_end_buf: [128]i32 = undefined;

    for (batch_examples, 0..) |example, b| {
        const tok_offset = b * sl;
        const tgt_offset = b * sl * nc;

        // -- Tokenize with real HF tokenizer --------------------------------
        const tok_ids = tok_ids_buf[0..sl];
        const tok_mask = tok_mask_buf[0..sl];
        const words_mask = words_mask_buf[0..sl];
        const first_pos = first_pos_buf[0..sl];
        const e_pos = e_tok_pos_buf[0..@min(entity_types.len, e_tok_pos_buf.len)];
        const e_end = e_tok_end_buf[0..@min(entity_types.len, e_tok_end_buf.len)];

        const result = tokenizer.encodeInto(
            allocator,
            example.text,
            entity_types,
            tok_ids,
            tok_mask,
            words_mask,
            first_pos,
            e_pos,
            e_end,
        );
        const num_words = result.num_words;

        // Convert i32 token IDs → i64 and i32 mask → f32.
        for (0..sl) |p| {
            input_ids[tok_offset + p] = @as(i64, tok_ids[p]);
            attention_mask[tok_offset + p] = @as(f32, @floatFromInt(tok_mask[p]));
        }

        // -- Build word-level entity class map --------------------------------
        // Map entity byte spans to word indices, then to class IDs.
        // word_class[w] = class index for word w, or 0 (O) if no entity.
        var word_class_buf: [4096]u32 = undefined;
        const max_words = @min(num_words, word_class_buf.len);
        @memset(word_class_buf[0..max_words], 0);

        // Build a mapping from byte offset → word index by splitting
        // the text the same way the tokenizer does (whitespace split).
        var word_starts: [4096]usize = undefined;
        var word_ends: [4096]usize = undefined;
        var n_words: usize = 0;
        {
            var iter = std.mem.tokenizeAny(u8, example.text, " \t\r\n");
            while (iter.next()) |word| {
                if (n_words >= max_words) break;
                // Compute byte offset: iter.index points past the delimiter.
                const word_start = @intFromPtr(word.ptr) - @intFromPtr(example.text.ptr);
                word_starts[n_words] = word_start;
                word_ends[n_words] = word_start + word.len;
                n_words += 1;
            }
        }

        // For each entity, find overlapping words.
        for (example.entities) |ent| {
            const cls = label_map.get(ent.label) orelse 0;
            if (cls == 0) continue;
            for (0..n_words) |w| {
                // Word overlaps entity if [word_start, word_end) ∩ [ent.start, ent.end) ≠ ∅
                if (word_starts[w] < ent.end and word_ends[w] > ent.start) {
                    word_class_buf[w] = cls;
                }
            }
        }

        // -- Build one-hot targets from word classes --------------------------
        @memset(targets[tgt_offset .. tgt_offset + sl * nc], 0.0);

        for (0..sl) |p| {
            const row = tgt_offset + p * nc;
            const wm = words_mask[p];
            if (wm > 0 and @as(usize, @intCast(wm - 1)) < max_words) {
                // This token belongs to word (wm-1). Use its class.
                const cls = word_class_buf[@intCast(wm - 1)];
                targets[row + cls] = 1.0;
                stats.supervised_token_count += 1;
                if (cls > 0) stats.entity_token_count += 1;
            } else if (tok_mask[p] != 0) {
                // Non-padding, non-word token (prompt/special tokens): ignore.
                stats.ignored_token_count += 1;
            } else {
                stats.ignored_token_count += 1;
            }
            // Padding (tok_mask==0): all-zero → contributes 0 to loss.
        }
    }

    return stats;
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

fn stripEncoderPrefix(name: []const u8) []const u8 {
    const prefix = "encoder.";
    if (std.mem.startsWith(u8, name, prefix)) {
        return name[prefix.len..];
    }
    return name;
}

fn parseBackend(value: []const u8) ?Gliner2TrainBackend {
    if (std.ascii.eqlIgnoreCase(value, "auto")) return .auto;
    if (std.ascii.eqlIgnoreCase(value, "metal")) return .metal;
    if (std.ascii.eqlIgnoreCase(value, "mlx")) return .mlx;
    if (std.ascii.eqlIgnoreCase(value, "native")) return .native;
    return null;
}

fn parseEvalStrategy(value: []const u8) ?EvalStrategy {
    if (std.ascii.eqlIgnoreCase(value, "epoch")) return .epoch;
    if (std.ascii.eqlIgnoreCase(value, "steps")) return .steps;
    if (std.ascii.eqlIgnoreCase(value, "none") or std.ascii.eqlIgnoreCase(value, "no")) return .none;
    return null;
}

fn parseReportTo(value: []const u8) ?ReportTo {
    if (std.ascii.eqlIgnoreCase(value, "stdout")) return .stdout;
    if (std.ascii.eqlIgnoreCase(value, "jsonl")) return .jsonl;
    return null;
}

fn selectBackend(
    requested: Gliner2TrainBackend,
    force_native: bool,
    metal_available: bool,
    mlx_available: bool,
) !Gliner2TrainBackend {
    if (force_native) return .native;
    return switch (requested) {
        .auto => if (metal_available) .metal else if (mlx_available) .mlx else .native,
        .metal => if (metal_available) .metal else error.MetalBackendUnavailable,
        .mlx => if (mlx_available) .mlx else error.MlxBackendUnavailable,
        .native => .native,
    };
}

fn backendLabel(backend: Gliner2TrainBackend) []const u8 {
    return switch (backend) {
        .auto => "auto",
        .metal => "Metal",
        .mlx => "MLX (Apple Silicon)",
        .native => "native CPU/BLAS",
    };
}

fn loadSafetensorsIntoGpuHostedStore(
    allocator: std.mem.Allocator,
    weight_store: *MetalWeightStore,
    st_path: []const u8,
) !void {
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
        var owned_loaded = try cloneLoadedWeight(allocator, loaded, stripEncoderPrefix(name));
        errdefer owned_loaded.deinit();
        const stripped = stripEncoderPrefix(name);
        const owned_name = try allocator.dupe(u8, stripped);
        errdefer allocator.free(owned_name);
        try weight_store.lazy_weights.put(allocator, owned_name, .{
            .tensor_ref = undefined,
            .host_loaded = owned_loaded,
            .active_tier = .host,
            .loaded_bytes = owned_loaded.tensor.data.len,
        });
        loaded_count += 1;
    }
    print("  loaded {d} weights via Metal from {s}\n", .{ loaded_count, st_path });
    source.weightSource().deinit();
}

fn cloneLoadedWeight(allocator: std.mem.Allocator, loaded: LoadedWeight, name: []const u8) !LoadedWeight {
    if (loaded.quantized or loaded.quantized_storage != null) return error.UnsupportedQuantizedTrainingWeight;
    const owned_data = try allocator.dupe(u8, loaded.tensor.data);
    errdefer allocator.free(owned_data);
    const owned_shape = try allocator.dupe(i64, loaded.tensor.shape);
    errdefer allocator.free(owned_shape);
    _ = name;
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

fn initClassifierHeadInGpuHostedStore(
    allocator: std.mem.Allocator,
    weight_store: *MetalWeightStore,
    seed: u64,
    hidden_size: u32,
    num_classes: u32,
) !void {
    if (comptime !build_options.enable_metal) return error.MetalBackendUnavailable;
    var rng_init = std.Random.DefaultPrng.init(seed);
    var prng_init = rng_init.random();
    const H = hidden_size;
    const C = num_classes;

    const w_data = try allocator.alloc(f32, @as(usize, @intCast(C)) * @as(usize, @intCast(H)));
    defer allocator.free(w_data);
    const sd: f32 = 0.02;
    for (w_data) |*v| v.* = prng_init.floatNorm(f32) * sd;
    const w_tensor = try Tensor.initFloat32(allocator, "task_classifier.weight", &.{ C, H }, w_data);
    try weight_store.lazy_weights.put(allocator, try allocator.dupe(u8, "task_classifier.weight"), .{
        .tensor_ref = undefined,
        .host_loaded = .{ .tensor = w_tensor },
        .active_tier = .host,
        .loaded_bytes = w_tensor.data.len,
    });

    const b_data = try allocator.alloc(f32, C);
    defer allocator.free(b_data);
    @memset(b_data, 0.0);
    const b_tensor = try Tensor.initFloat32(allocator, "task_classifier.bias", &.{C}, b_data);
    try weight_store.lazy_weights.put(allocator, try allocator.dupe(u8, "task_classifier.bias"), .{
        .tensor_ref = undefined,
        .host_loaded = .{ .tensor = b_tensor },
        .active_tier = .host,
        .loaded_bytes = b_tensor.data.len,
    });
    print("  initialized classifier head (Metal): [{d}, {d}] + [{d}]\n", .{ C, H, C });
    try initParityTopLevelWeightsMetal(allocator, weight_store, H);
}

fn fillRectIdentity(data: []f32, out_dim: usize, in_dim: usize) void {
    @memset(data, 0.0);
    for (0..@min(out_dim, in_dim)) |i| data[i * in_dim + i] = 1.0;
}

fn initParityTopLevelWeightsNative(
    allocator: std.mem.Allocator,
    weight_store: *native_compute.WeightStore,
    hidden_size: u32,
) !void {
    const specs = parityWeightSpecs(hidden_size);
    for (specs) |spec| {
        const out_dim: usize = @intCast(spec.out_dim);
        const in_dim: usize = @intCast(spec.in_dim);
        const data = try allocator.alloc(f32, out_dim * in_dim);
        defer allocator.free(data);
        fillRectIdentity(data, out_dim, in_dim);
        const tensor = try Tensor.initFloat32(allocator, spec.name, &.{ @as(i64, @intCast(out_dim)), @as(i64, @intCast(in_dim)) }, data);
        try native_wsPutMissingOwned(allocator, weight_store, spec.name, tensor);
    }
    const vector_specs = parityVectorSpecs(hidden_size);
    for (vector_specs) |spec| {
        const dim: usize = @intCast(spec.dim);
        const data = try allocator.alloc(f32, dim);
        defer allocator.free(data);
        fillVector(data, spec.fill);
        const tensor = try Tensor.initFloat32(allocator, spec.name, &.{@as(i64, @intCast(dim))}, data);
        try native_wsPutMissingOwned(allocator, weight_store, spec.name, tensor);
    }
}

fn native_wsPutMissingOwned(
    allocator: std.mem.Allocator,
    weight_store: *native_compute.WeightStore,
    name: []const u8,
    tensor: Tensor,
) !void {
    if (weight_store.resident_weights.contains(name)) {
        var discard = tensor;
        discard.deinit();
        return;
    }
    const owned_name = try allocator.dupe(u8, name);
    try weight_store.resident_weights.put(allocator, owned_name, .{ .tensor = tensor });
}

fn initParityTopLevelWeightsMlx(
    allocator: std.mem.Allocator,
    weights: if (build_options.enable_mlx) mlx_c.mlx_map_string_to_array_t else void,
    hidden_size: u32,
) !void {
    if (comptime !build_options.enable_mlx) return;
    const specs = parityWeightSpecs(hidden_size);
    for (specs) |spec| {
        if (mlxMapContains(weights, spec.name)) continue;
        const out_dim: usize = @intCast(spec.out_dim);
        const in_dim: usize = @intCast(spec.in_dim);
        const shape = [_]i32{ @intCast(out_dim), @intCast(in_dim) };
        const data = try allocator.alloc(f32, out_dim * in_dim);
        defer allocator.free(data);
        fillRectIdentity(data, out_dim, in_dim);
        const arr = mlx.arrayFromFloat32(data, &shape);
        try mlx.insertWeight(weights, allocator, spec.name, arr);
    }
    const vector_specs = parityVectorSpecs(hidden_size);
    for (vector_specs) |spec| {
        if (mlxMapContains(weights, spec.name)) continue;
        const dim: usize = @intCast(spec.dim);
        const shape = [_]i32{@intCast(dim)};
        const data = try allocator.alloc(f32, dim);
        defer allocator.free(data);
        fillVector(data, spec.fill);
        const arr = mlx.arrayFromFloat32(data, &shape);
        try mlx.insertWeight(weights, allocator, spec.name, arr);
    }
}

fn initParityTopLevelWeightsMetal(
    allocator: std.mem.Allocator,
    weight_store: *MetalWeightStore,
    hidden_size: u32,
) !void {
    if (comptime !build_options.enable_metal) return;
    const specs = parityWeightSpecs(hidden_size);
    for (specs) |spec| {
        if (weight_store.lazy_weights.contains(spec.name)) continue;
        const out_dim: usize = @intCast(spec.out_dim);
        const in_dim: usize = @intCast(spec.in_dim);
        const data = try allocator.alloc(f32, out_dim * in_dim);
        defer allocator.free(data);
        fillRectIdentity(data, out_dim, in_dim);
        const tensor = try Tensor.initFloat32(allocator, spec.name, &.{ @as(i64, @intCast(out_dim)), @as(i64, @intCast(in_dim)) }, data);
        try weight_store.lazy_weights.put(allocator, try allocator.dupe(u8, spec.name), .{
            .tensor_ref = undefined,
            .host_loaded = .{ .tensor = tensor },
            .active_tier = .host,
            .loaded_bytes = tensor.data.len,
        });
    }
    const vector_specs = parityVectorSpecs(hidden_size);
    for (vector_specs) |spec| {
        if (weight_store.lazy_weights.contains(spec.name)) continue;
        const dim: usize = @intCast(spec.dim);
        const data = try allocator.alloc(f32, dim);
        defer allocator.free(data);
        fillVector(data, spec.fill);
        const tensor = try Tensor.initFloat32(allocator, spec.name, &.{@as(i64, @intCast(dim))}, data);
        try weight_store.lazy_weights.put(allocator, try allocator.dupe(u8, spec.name), .{
            .tensor_ref = undefined,
            .host_loaded = .{ .tensor = tensor },
            .active_tier = .host,
            .loaded_bytes = tensor.data.len,
        });
    }
}

const ParityWeightSpec = struct {
    name: []const u8,
    out_dim: u32,
    in_dim: u32,
};

fn parityWeightSpecs(hidden_size: u32) [25]ParityWeightSpec {
    const H = hidden_size;
    const FF = H * 4;
    const MID = H * 2;
    const COUNT: u32 = 128;
    const COUNT_FF: u32 = 256;
    return .{
        .{ .name = "span_rep.span_rep_layer.project_start.0.weight", .out_dim = FF, .in_dim = H },
        .{ .name = "span_rep.span_rep_layer.project_start.3.weight", .out_dim = H, .in_dim = FF },
        .{ .name = "span_rep.span_rep_layer.project_end.0.weight", .out_dim = FF, .in_dim = H },
        .{ .name = "span_rep.span_rep_layer.project_end.3.weight", .out_dim = H, .in_dim = FF },
        .{ .name = "span_rep.span_rep_layer.out_project.0.weight", .out_dim = FF, .in_dim = H * 2 },
        .{ .name = "span_rep.span_rep_layer.out_project.3.weight", .out_dim = H, .in_dim = FF },
        .{ .name = "classifier.0.weight", .out_dim = MID, .in_dim = H },
        .{ .name = "classifier.2.weight", .out_dim = 1, .in_dim = MID },
        .{ .name = "count_pred.0.weight", .out_dim = MID, .in_dim = H },
        .{ .name = "count_pred.2.weight", .out_dim = 20, .in_dim = MID },
        .{ .name = "count_embed.pos_embedding.weight", .out_dim = 20, .in_dim = H },
        .{ .name = "count_embed.gru.weight_ih_l0", .out_dim = H * 3, .in_dim = H },
        .{ .name = "count_embed.gru.weight_hh_l0", .out_dim = H * 3, .in_dim = H },
        .{ .name = "count_embed.transformer.in_projector.weight", .out_dim = COUNT, .in_dim = H },
        .{ .name = "count_embed.transformer.transformer.layers.0.self_attn.in_proj_weight", .out_dim = COUNT * 3, .in_dim = COUNT },
        .{ .name = "count_embed.transformer.transformer.layers.0.self_attn.out_proj.weight", .out_dim = COUNT, .in_dim = COUNT },
        .{ .name = "count_embed.transformer.transformer.layers.0.linear1.weight", .out_dim = COUNT_FF, .in_dim = COUNT },
        .{ .name = "count_embed.transformer.transformer.layers.0.linear2.weight", .out_dim = COUNT, .in_dim = COUNT_FF },
        .{ .name = "count_embed.transformer.transformer.layers.1.self_attn.in_proj_weight", .out_dim = COUNT * 3, .in_dim = COUNT },
        .{ .name = "count_embed.transformer.transformer.layers.1.self_attn.out_proj.weight", .out_dim = COUNT, .in_dim = COUNT },
        .{ .name = "count_embed.transformer.transformer.layers.1.linear1.weight", .out_dim = COUNT_FF, .in_dim = COUNT },
        .{ .name = "count_embed.transformer.transformer.layers.1.linear2.weight", .out_dim = COUNT, .in_dim = COUNT_FF },
        .{ .name = "count_embed.transformer.out_projector.0.weight", .out_dim = H, .in_dim = H + COUNT },
        .{ .name = "count_embed.transformer.out_projector.2.weight", .out_dim = H, .in_dim = H },
        .{ .name = "count_embed.transformer.out_projector.4.weight", .out_dim = H, .in_dim = H },
    };
}

const ParityVectorFill = enum { zero, one };

const ParityVectorSpec = struct {
    name: []const u8,
    dim: u32,
    fill: ParityVectorFill = .zero,
};

fn fillVector(data: []f32, fill: ParityVectorFill) void {
    @memset(data, switch (fill) {
        .zero => 0.0,
        .one => 1.0,
    });
}

fn parityVectorSpecs(hidden_size: u32) [26]ParityVectorSpec {
    const H = hidden_size;
    const MID = H * 2;
    const COUNT: u32 = 128;
    const COUNT_FF: u32 = 256;
    return .{
        .{ .name = "classifier.0.bias", .dim = MID },
        .{ .name = "classifier.2.bias", .dim = 1 },
        .{ .name = "count_pred.0.bias", .dim = MID },
        .{ .name = "count_pred.2.bias", .dim = 20 },
        .{ .name = "count_embed.gru.bias_ih_l0", .dim = H * 3 },
        .{ .name = "count_embed.gru.bias_hh_l0", .dim = H * 3 },
        .{ .name = "count_embed.transformer.in_projector.bias", .dim = COUNT },
        .{ .name = "count_embed.transformer.transformer.layers.0.self_attn.in_proj_bias", .dim = COUNT * 3 },
        .{ .name = "count_embed.transformer.transformer.layers.0.self_attn.out_proj.bias", .dim = COUNT },
        .{ .name = "count_embed.transformer.transformer.layers.0.norm1.weight", .dim = COUNT, .fill = .one },
        .{ .name = "count_embed.transformer.transformer.layers.0.norm1.bias", .dim = COUNT },
        .{ .name = "count_embed.transformer.transformer.layers.0.linear1.bias", .dim = COUNT_FF },
        .{ .name = "count_embed.transformer.transformer.layers.0.linear2.bias", .dim = COUNT },
        .{ .name = "count_embed.transformer.transformer.layers.0.norm2.weight", .dim = COUNT, .fill = .one },
        .{ .name = "count_embed.transformer.transformer.layers.0.norm2.bias", .dim = COUNT },
        .{ .name = "count_embed.transformer.transformer.layers.1.self_attn.in_proj_bias", .dim = COUNT * 3 },
        .{ .name = "count_embed.transformer.transformer.layers.1.self_attn.out_proj.bias", .dim = COUNT },
        .{ .name = "count_embed.transformer.transformer.layers.1.norm1.weight", .dim = COUNT, .fill = .one },
        .{ .name = "count_embed.transformer.transformer.layers.1.norm1.bias", .dim = COUNT },
        .{ .name = "count_embed.transformer.transformer.layers.1.linear1.bias", .dim = COUNT_FF },
        .{ .name = "count_embed.transformer.transformer.layers.1.linear2.bias", .dim = COUNT },
        .{ .name = "count_embed.transformer.transformer.layers.1.norm2.weight", .dim = COUNT, .fill = .one },
        .{ .name = "count_embed.transformer.transformer.layers.1.norm2.bias", .dim = COUNT },
        .{ .name = "count_embed.transformer.out_projector.0.bias", .dim = H },
        .{ .name = "count_embed.transformer.out_projector.2.bias", .dim = H },
        .{ .name = "count_embed.transformer.out_projector.4.bias", .dim = H },
    };
}

fn mlxMapContains(weights: if (build_options.enable_mlx) mlx_c.mlx_map_string_to_array_t else void, name: []const u8) bool {
    if (comptime !build_options.enable_mlx) return false;
    const it = mlx_c.mlx_map_string_to_array_iterator_new(weights);
    defer _ = mlx_c.mlx_map_string_to_array_iterator_free(it);
    while (true) {
        var key: [*c]const u8 = null;
        var val = mlx_c.mlx_array_new();
        if (mlx_c.mlx_map_string_to_array_iterator_next(&key, &val, it) != 0) {
            _ = mlx_c.mlx_array_free(val);
            return false;
        }
        const found = key != null and std.mem.eql(u8, std.mem.span(key), name);
        _ = mlx_c.mlx_array_free(val);
        if (found) return true;
    }
}

fn deinitGpuHostedWeightStore(allocator: std.mem.Allocator, weight_store: *MetalWeightStore) void {
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
        _ = mlx_c.mlx_map_string_to_array_free(weight_store.resident_weights);
    }
}

const SpanParityDebugStats = struct {
    rows: usize = 0,
    entity_types: usize = 0,
    logits_count: usize = 0,
    valid_weight_count: usize = 0,
    positive_count: usize = 0,
    mask_weight_sum: f64 = 0.0,
    weighted_mask_sum: f64 = 0.0,
    logits_min: f64 = 0.0,
    logits_max: f64 = 0.0,
    logits_mean: f64 = 0.0,
    positive_logits_mean: f64 = 0.0,
    negative_logits_mean: f64 = 0.0,
    bce_unweighted_sum: f64 = 0.0,
    bce_masked_sum: f64 = 0.0,
    bce_masked_positive_sum: f64 = 0.0,
    bce_masked_negative_sum: f64 = 0.0,
    bce_weighted_sum: f64 = 0.0,
    bce_weighted_positive_sum: f64 = 0.0,
    bce_weighted_negative_sum: f64 = 0.0,
    bce_weighted_mean: f64 = 0.0,
};

fn printSpanParityDebug(
    logits: []const f32,
    targets: []const f32,
    targets_shape: ml.graph.Shape,
    entity_types: usize,
    has_label_positive_weights: bool,
    opts: Options,
) !void {
    if (entity_types == 0) return error.InvalidEntityTypes;
    const rows: usize = @intCast(targets_shape.dims[0]);
    const width: usize = @intCast(targets_shape.dims[1]);
    const expected_min_width = if (has_label_positive_weights)
        gliner2_autodiff.weightedSpanStartTargetWidth(entity_types)
    else
        gliner2_autodiff.spanStartTargetWidth(entity_types);
    if (width < expected_min_width) return error.InvalidGlinerSpanTargetShape;
    if (targets.len != rows * width) return error.InvalidGlinerSpanTargetShape;
    if (logits.len != rows * entity_types) return error.InvalidGlinerSpanLogitsShape;

    var stats = SpanParityDebugStats{
        .rows = rows,
        .entity_types = entity_types,
        .logits_count = logits.len,
        .logits_min = 1.0e300,
        .logits_max = -1.0e300,
    };
    var logits_sum: f64 = 0.0;
    var pos_logits_sum: f64 = 0.0;
    var neg_logits_sum: f64 = 0.0;
    var neg_count: usize = 0;
    const top_k = 5;
    var top_bce = [_]f64{-1.0} ** top_k;
    var top_logits = [_]f64{0.0} ** top_k;
    var top_rows = [_]usize{0} ** top_k;
    var top_entities = [_]usize{0} ** top_k;

    for (0..rows) |row_idx| {
        const target_row = row_idx * width;
        const logit_row = row_idx * entity_types;
        for (0..entity_types) |entity_idx| {
            const label = targets[target_row + entity_idx];
            const mask_weight = targets[target_row + entity_types + entity_idx];
            const logit = @as(f64, @floatCast(logits[logit_row + entity_idx]));
            stats.logits_min = @min(stats.logits_min, logit);
            stats.logits_max = @max(stats.logits_max, logit);
            logits_sum += logit;

            const is_positive = label > 0.0;
            if (is_positive) {
                stats.positive_count += 1;
                pos_logits_sum += logit;
            } else {
                neg_count += 1;
                neg_logits_sum += logit;
            }

            const bce = stableBceWithLogits(logit, if (is_positive) 1.0 else 0.0);
            stats.bce_unweighted_sum += bce;
            if (mask_weight > 0.0) {
                stats.valid_weight_count += 1;
                const mask_weight64 = @as(f64, @floatCast(mask_weight));
                const label_weight: f64 = if (is_positive) blk: {
                    if (has_label_positive_weights) {
                        const positive_weights_offset = 2 * entity_types;
                        break :blk @as(f64, @floatCast(targets[target_row + positive_weights_offset + entity_idx]));
                    }
                    break :blk @as(f64, @floatCast(opts.span_positive_weight));
                } else @as(f64, @floatCast(opts.span_negative_weight));
                stats.mask_weight_sum += mask_weight64;
                stats.weighted_mask_sum += mask_weight64 * label_weight;
                stats.bce_masked_sum += bce * mask_weight64;
                stats.bce_weighted_sum += bce * mask_weight64 * label_weight;
                if (is_positive) {
                    stats.bce_masked_positive_sum += bce * mask_weight64;
                    stats.bce_weighted_positive_sum += bce * mask_weight64 * label_weight;
                } else {
                    stats.bce_masked_negative_sum += bce * mask_weight64;
                    stats.bce_weighted_negative_sum += bce * mask_weight64 * label_weight;
                    if (bce > top_bce[top_k - 1]) {
                        var insert_idx: usize = top_k - 1;
                        while (insert_idx > 0 and bce > top_bce[insert_idx - 1]) : (insert_idx -= 1) {
                            top_bce[insert_idx] = top_bce[insert_idx - 1];
                            top_logits[insert_idx] = top_logits[insert_idx - 1];
                            top_rows[insert_idx] = top_rows[insert_idx - 1];
                            top_entities[insert_idx] = top_entities[insert_idx - 1];
                        }
                        top_bce[insert_idx] = bce;
                        top_logits[insert_idx] = logit;
                        top_rows[insert_idx] = row_idx;
                        top_entities[insert_idx] = entity_idx;
                    }
                }
            }
        }
    }

    if (logits.len > 0) stats.logits_mean = logits_sum / @as(f64, @floatFromInt(logits.len));
    if (stats.positive_count > 0) stats.positive_logits_mean = pos_logits_sum / @as(f64, @floatFromInt(stats.positive_count));
    if (neg_count > 0) stats.negative_logits_mean = neg_logits_sum / @as(f64, @floatFromInt(neg_count));
    if (stats.weighted_mask_sum > 0.0) stats.bce_weighted_mean = stats.bce_weighted_sum / stats.weighted_mask_sum;

    print(
        "SPAN_PARITY_DEBUG {{\"rows\":{d},\"entity_types\":{d},\"logits_count\":{d},\"valid_weight_count\":{d},\"positive_count\":{d},\"mask_weight_sum\":{d:.9},\"weighted_mask_sum\":{d:.9},\"logits_min\":{d:.9},\"logits_max\":{d:.9},\"logits_mean\":{d:.9},\"positive_logits_mean\":{d:.9},\"negative_logits_mean\":{d:.9},\"bce_unweighted_sum\":{d:.9},\"bce_masked_sum\":{d:.9},\"bce_masked_positive_sum\":{d:.9},\"bce_masked_negative_sum\":{d:.9},\"bce_weighted_sum\":{d:.9},\"bce_weighted_positive_sum\":{d:.9},\"bce_weighted_negative_sum\":{d:.9},\"bce_weighted_mean\":{d:.9},\"top_valid_negative_logits\":[",
        .{
            stats.rows,
            stats.entity_types,
            stats.logits_count,
            stats.valid_weight_count,
            stats.positive_count,
            stats.mask_weight_sum,
            stats.weighted_mask_sum,
            stats.logits_min,
            stats.logits_max,
            stats.logits_mean,
            stats.positive_logits_mean,
            stats.negative_logits_mean,
            stats.bce_unweighted_sum,
            stats.bce_masked_sum,
            stats.bce_masked_positive_sum,
            stats.bce_masked_negative_sum,
            stats.bce_weighted_sum,
            stats.bce_weighted_positive_sum,
            stats.bce_weighted_negative_sum,
            stats.bce_weighted_mean,
        },
    );
    var printed: usize = 0;
    for (0..top_k) |idx| {
        if (top_bce[idx] < 0.0) continue;
        if (printed > 0) print(",", .{});
        const row = top_rows[idx];
        print(
            "{{\"row\":{d},\"entity\":{d},\"start\":{d},\"width\":{d},\"logit\":{d:.9},\"bce\":{d:.9}}}",
            .{
                row,
                top_entities[idx],
                row / opts.max_span_width,
                row % opts.max_span_width,
                top_logits[idx],
                top_bce[idx],
            },
        );
        printed += 1;
    }
    print("],\"reduction\":\"{s}\"}}\n", .{spanLossReductionName(opts.span_loss_reduction)});
}

fn printSpanComponentDebug(components: gliner2_autodiff.SpanStartComponentDebug) void {
    print(
        "SPAN_COMPONENT_DEBUG {{\"positive_row\":{d},\"positive_entity\":{d},\"schema_row\":{d},\"start_hidden_norm\":{d:.9},\"end_hidden_norm\":{d:.9},\"projected_span_norm\":{d:.9},\"schema_hidden_norm\":{d:.9},\"count_gru_state_norm\":{d:.9},\"count_state_norm\":{d:.9},\"count_in_project_norm\":{d:.9},\"count_layer0_norm\":{d:.9},\"count_layer1_norm\":{d:.9},\"count_out0_norm\":{d:.9},\"count_out2_norm\":{d:.9},\"schema_projection_norm\":{d:.9},\"projected_schema_dot\":{d:.9},\"projected_span_mean\":{d:.9},\"count_gru_state_mean\":{d:.9},\"count_state_mean\":{d:.9},\"count_in_project_mean\":{d:.9},\"count_layer0_mean\":{d:.9},\"count_layer1_mean\":{d:.9},\"count_out0_mean\":{d:.9},\"count_out2_mean\":{d:.9},\"schema_projection_mean\":{d:.9}",
        .{
            components.positive_row,
            components.positive_entity,
            components.schema_row,
            components.start_hidden_norm,
            components.end_hidden_norm,
            components.projected_span_norm,
            components.schema_hidden_norm,
            components.count_gru_state_norm,
            components.count_state_norm,
            components.count_in_project_norm,
            components.count_layer0_norm,
            components.count_layer1_norm,
            components.count_out0_norm,
            components.count_out2_norm,
            components.schema_projection_norm,
            components.projected_schema_dot,
            components.projected_span_mean,
            components.count_gru_state_mean,
            components.count_state_mean,
            components.count_in_project_mean,
            components.count_layer0_mean,
            components.count_layer1_mean,
            components.count_out0_mean,
            components.count_out2_mean,
            components.schema_projection_mean,
        },
    );
    print(
        ",\"negative_row\":{d},\"negative_entity\":{d},\"negative_logit\":{d:.9},\"negative_start_hidden_norm\":{d:.9},\"negative_end_hidden_norm\":{d:.9},\"negative_schema_hidden_norm\":{d:.9},\"negative_projected_span_norm\":{d:.9},\"negative_schema_projection_norm\":{d:.9},\"negative_projected_schema_dot\":{d:.9},\"negative_start_hidden_mean\":{d:.9},\"negative_end_hidden_mean\":{d:.9},\"negative_schema_hidden_mean\":{d:.9},\"negative_projected_span_mean\":{d:.9},\"negative_schema_projection_mean\":{d:.9}}}\n",
        .{
            components.negative_row,
            components.negative_entity,
            components.negative_logit,
            components.negative_start_hidden_norm,
            components.negative_end_hidden_norm,
            components.negative_schema_hidden_norm,
            components.negative_projected_span_norm,
            components.negative_schema_projection_norm,
            components.negative_projected_schema_dot,
            components.negative_start_hidden_mean,
            components.negative_end_hidden_mean,
            components.negative_schema_hidden_mean,
            components.negative_projected_span_mean,
            components.negative_schema_projection_mean,
        },
    );
}

fn printGliner2TotalLossComponentDebug(components: gliner2_autodiff.Gliner2TotalLossComponentDebug) void {
    print(
        "GLINER2_TOTAL_LOSS_COMPONENT_DEBUG {{\"classification_loss\":{d:.9},\"structure_loss\":{d:.9},\"count_loss\":{d:.9},\"total_loss\":{d:.9}}}\n",
        .{
            components.classification_loss,
            components.structure_loss,
            components.count_loss,
            components.total_loss,
        },
    );
}

fn printGliner2ClassificationDebug(
    logits: []const f32,
    targets: []const f32,
    targets_shape: ml.graph.Shape,
    entity_types: usize,
) !void {
    if (entity_types == 0) return error.InvalidEntityTypes;
    if (targets_shape.rank() != 2) return error.InvalidGlinerSpanTargetShape;
    const rows: usize = @intCast(targets_shape.dims[0]);
    const target_width: usize = @intCast(targets_shape.dims[1]);
    if (targets.len != rows * target_width) return error.InvalidGlinerSpanTargetShape;
    if (logits.len != rows * entity_types) return error.InvalidGlinerSpanLogitsShape;
    const label_offset = gliner2_autodiff.gliner2TotalLossClassificationLabelsOffset(entity_types);
    const mask_offset = gliner2_autodiff.gliner2TotalLossClassificationMaskOffset(entity_types);
    if (target_width < mask_offset + entity_types) return error.InvalidGlinerSpanTargetShape;

    var valid_count: usize = 0;
    var positive_count: usize = 0;
    var logits_min: f64 = std.math.inf(f64);
    var logits_max: f64 = -std.math.inf(f64);
    var logits_sum: f64 = 0.0;
    var label_sum: f64 = 0.0;
    var mask_sum: f64 = 0.0;
    var bce_sum: f64 = 0.0;

    for (0..rows) |row_idx| {
        const target_row = row_idx * target_width;
        const logit_row = row_idx * entity_types;
        for (0..entity_types) |entity_idx| {
            const label = @as(f64, @floatCast(targets[target_row + label_offset + entity_idx]));
            const mask = @as(f64, @floatCast(targets[target_row + mask_offset + entity_idx]));
            const logit = @as(f64, @floatCast(logits[logit_row + entity_idx]));
            label_sum += label;
            mask_sum += mask;
            if (label > 0.0) positive_count += 1;
            if (mask > 0.0) {
                valid_count += 1;
                logits_min = @min(logits_min, logit);
                logits_max = @max(logits_max, logit);
                logits_sum += logit;
                bce_sum += stableBceWithLogits(logit, label) * mask;
            }
        }
    }

    if (valid_count == 0) {
        logits_min = 0.0;
        logits_max = 0.0;
    }
    const logits_mean = if (valid_count > 0) logits_sum / @as(f64, @floatFromInt(valid_count)) else 0.0;
    print(
        "GLINER2_CLASSIFICATION_DEBUG {{\"rows\":{d},\"entity_types\":{d},\"logits_count\":{d},\"valid_count\":{d},\"positive_count\":{d},\"label_sum\":{d:.9},\"mask_sum\":{d:.9},\"logits_min\":{d:.9},\"logits_max\":{d:.9},\"logits_mean\":{d:.9},\"bce_sum\":{d:.9},\"logits_head\":[",
        .{
            rows,
            entity_types,
            logits.len,
            valid_count,
            positive_count,
            label_sum,
            mask_sum,
            logits_min,
            logits_max,
            logits_mean,
            bce_sum,
        },
    );
    const head_count = @min(logits.len, 16);
    for (0..head_count) |idx| {
        if (idx != 0) print(",", .{});
        print("{d:.9}", .{logits[idx]});
    }
    print("],\"labels_head\":[", .{});
    const label_head_count = @min(rows * entity_types, 16);
    for (0..label_head_count) |idx| {
        if (idx != 0) print(",", .{});
        const row_idx = idx / entity_types;
        const entity_idx = idx % entity_types;
        print("{d:.1}", .{targets[row_idx * target_width + label_offset + entity_idx]});
    }
    print("],\"mask_head\":[", .{});
    for (0..label_head_count) |idx| {
        if (idx != 0) print(",", .{});
        const row_idx = idx / entity_types;
        const entity_idx = idx % entity_types;
        print("{d:.1}", .{targets[row_idx * target_width + mask_offset + entity_idx]});
    }
    print("],\"valid_logits_head\":[", .{});
    var valid_printed: usize = 0;
    outer: for (0..rows) |row_idx| {
        const target_row = row_idx * target_width;
        const logit_row = row_idx * entity_types;
        for (0..entity_types) |entity_idx| {
            if (targets[target_row + mask_offset + entity_idx] <= 0.0) continue;
            if (valid_printed != 0) print(",", .{});
            print("{d:.9}", .{logits[logit_row + entity_idx]});
            valid_printed += 1;
            if (valid_printed >= 16) break :outer;
        }
    }
    print("],\"valid_labels_head\":[", .{});
    valid_printed = 0;
    outer_labels: for (0..rows) |row_idx| {
        const target_row = row_idx * target_width;
        for (0..entity_types) |entity_idx| {
            if (targets[target_row + mask_offset + entity_idx] <= 0.0) continue;
            if (valid_printed != 0) print(",", .{});
            print("{d:.1}", .{targets[target_row + label_offset + entity_idx]});
            valid_printed += 1;
            if (valid_printed >= 16) break :outer_labels;
        }
    }
    print("]}}\n", .{});
}

fn printSpanPreprocessDebug(batch: *const gliner2_data.EncodedBatch) void {
    const input_ids = batch.input_ids[0..batch.max_length];
    const attention_mask = batch.attention_mask[0..batch.max_length];
    const first_positions = batch.first_token_positions[0..batch.max_words_per_sample];
    const span_mask = batch.span_mask[0..batch.max_spans];
    const span_indices = batch.span_indices[0 .. batch.max_spans * 2];
    const span_labels = batch.span_labels[0 .. batch.max_spans * batch.num_entity_types];
    print("SPAN_PREPROCESS_DEBUG {{\"batch_size\":{d},\"max_length\":{d},\"max_words_per_sample\":{d},\"max_spans\":{d},\"num_entity_types\":{d},\"max_schemas\":{d},\"max_schema_specials\":{d},\"input_ids\":", .{
        batch.batch_size,
        batch.max_length,
        batch.max_words_per_sample,
        batch.max_spans,
        batch.num_entity_types,
        batch.max_schemas,
        batch.max_schema_specials,
    });
    printI32JsonArray(input_ids);
    print(",\"input_ids_all\":", .{});
    printI32JsonArray(batch.input_ids[0 .. batch.batch_size * batch.max_length]);
    print(",\"attention_mask\":", .{});
    printI32JsonArray(attention_mask);
    print(",\"attention_mask_all\":", .{});
    printI32JsonArray(batch.attention_mask[0 .. batch.batch_size * batch.max_length]);
    print(",\"first_token_positions\":", .{});
    printI32JsonArray(first_positions);
    print(",\"first_token_positions_all\":", .{});
    printI32JsonArray(batch.first_token_positions[0 .. batch.batch_size * batch.max_words_per_sample]);
    print(",\"e_token_positions\":", .{});
    printI32JsonArray(batch.e_token_positions[0..batch.num_entity_types]);
    print(",\"e_token_end_positions\":", .{});
    printI32JsonArray(batch.e_token_end_positions[0..batch.num_entity_types]);
    print(",\"e_token_positions_all\":", .{});
    printI32JsonArray(batch.e_token_positions[0 .. batch.batch_size * batch.num_entity_types]);
    print(",\"e_token_end_positions_all\":", .{});
    printI32JsonArray(batch.e_token_end_positions[0 .. batch.batch_size * batch.num_entity_types]);
    if (batch.text_word_counts.len > 0) {
        print(",\"text_word_counts\":", .{});
        printI32JsonArray(batch.text_word_counts[0..batch.batch_size]);
    }
    if (batch.schema_counts.len > 0) {
        print(",\"schema_counts\":", .{});
        printI32JsonArray(batch.schema_counts[0..batch.batch_size]);
    }
    if (batch.task_type_ids.len > 0 and batch.max_schemas > 0) {
        print(",\"task_type_ids\":", .{});
        printI32JsonArray(batch.task_type_ids[0..batch.max_schemas]);
        print(",\"task_type_ids_all\":", .{});
        printI32JsonArray(batch.task_type_ids[0 .. batch.batch_size * batch.max_schemas]);
    }
    if (batch.schema_special_counts.len > 0 and batch.max_schemas > 0) {
        print(",\"schema_special_counts\":", .{});
        printI32JsonArray(batch.schema_special_counts[0..batch.max_schemas]);
        print(",\"schema_special_counts_all\":", .{});
        printI32JsonArray(batch.schema_special_counts[0 .. batch.batch_size * batch.max_schemas]);
    }
    if (batch.schema_special_positions.len > 0 and batch.max_schemas > 0 and batch.max_schema_specials > 0) {
        print(",\"schema_special_positions\":", .{});
        printI32JsonArray(batch.schema_special_positions[0 .. batch.max_schemas * batch.max_schema_specials]);
        print(",\"schema_special_positions_all\":", .{});
        printI32JsonArray(batch.schema_special_positions[0 .. batch.batch_size * batch.max_schemas * batch.max_schema_specials]);
    }
    if (batch.entity_type_kind.len > 0) {
        print(",\"entity_type_kind\":", .{});
        printI32JsonArray(batch.entity_type_kind[0..batch.num_entity_types]);
        print(",\"entity_type_kind_all\":", .{});
        printI32JsonArray(batch.entity_type_kind[0 .. batch.batch_size * batch.num_entity_types]);
    }
    print(",\"span_indices\":", .{});
    printI32JsonArray(span_indices);
    print(",\"span_mask\":", .{});
    printF32JsonArray(span_mask);
    print(",\"span_labels\":", .{});
    printF32JsonArray(span_labels);
    print(",\"span_labels_all\":", .{});
    printF32JsonArray(batch.span_labels[0 .. batch.batch_size * batch.max_spans * batch.num_entity_types]);
    print("}}\n", .{});
}

fn applySpanNegativeMask(
    targets: []f32,
    max_spans: usize,
    entity_types: usize,
    has_label_positive_weights: bool,
    mask_rate: f32,
    seed: u64,
) void {
    if (entity_types == 0) return;
    const width = if (has_label_positive_weights)
        gliner2_autodiff.weightedSpanStartTargetWidth(entity_types)
    else
        gliner2_autodiff.spanStartTargetWidth(entity_types);
    if (width == 0 or targets.len % width != 0) return;
    const rows = targets.len / width;
    _ = max_spans;
    var prng = std.Random.DefaultPrng.init(seed);
    var rng = prng.random();
    for (0..rows) |row_idx| {
        const row = row_idx * width;
        for (0..entity_types) |entity_idx| {
            const label = targets[row + entity_idx];
            const mask_idx = row + entity_types + entity_idx;
            if (label <= 0.0 and targets[mask_idx] > 0.0 and rng.float(f32) < mask_rate) {
                targets[mask_idx] = 0.0;
            }
        }
    }
}

fn printI32JsonArray(values: []const i32) void {
    print("[", .{});
    for (values, 0..) |value, idx| {
        if (idx != 0) print(",", .{});
        print("{d}", .{value});
    }
    print("]", .{});
}

fn printF32JsonArray(values: []const f32) void {
    print("[", .{});
    for (values, 0..) |value, idx| {
        if (idx != 0) print(",", .{});
        print("{d:.6}", .{value});
    }
    print("]", .{});
}

fn stableBceWithLogits(logit: f64, label: f64) f64 {
    return @max(logit, 0.0) - logit * label + @log(1.0 + @exp(-@abs(logit)));
}

// ---------------------------------------------------------------------------
// Memory pre-flight
// ---------------------------------------------------------------------------

const macos_sysctl = if (builtin.os.tag == .macos) struct {
    pub extern fn sysctlbyname(
        name: [*:0]const u8,
        oldp: ?*anyopaque,
        oldlenp: ?*usize,
        newp: ?*anyopaque,
        newlen: usize,
    ) c_int;
} else struct {};

fn physicalMemoryBytes() ?u64 {
    if (builtin.os.tag != .macos) return null;
    var total: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (macos_sysctl.sysctlbyname("hw.memsize", @ptrCast(&total), &len, null, 0) != 0 or total == 0) return null;
    return total;
}

/// Conservative fp32 peak-footprint estimate for a GLiNER2 LoRA training run.
/// Dominant terms: frozen weights, and the disentangled-attention
/// intermediates — the [batch*heads, S, S, head_dim] C2P/P2C products and the
/// [S^2, H] Toeplitz-gathered relative embeddings — held live across forward
/// and backward for every layer. The x2 live-copy factor is calibrated
/// against measured peaks (~1.4 GiB RSS at batch 2 / seq 64 / DeBERTa-base,
/// where this formula yields ~1.6 GiB) and reproduces the observed 16 GiB
/// OOM boundary at batch 1 / seq 256.
fn estimateTrainingPeakBytes(
    vocab_size: u64,
    hidden_size: u64,
    intermediate_size: u64,
    num_layers: u64,
    num_heads: u64,
    batch_size: u64,
    seq_len: u64,
    metal_reuse_live_set: bool,
) u64 {
    const f32_size: u64 = 4;
    const head_dim = hidden_size / @max(num_heads, 1);
    const bh = batch_size * num_heads;
    const ss = seq_len * seq_len;
    const weights = (vocab_size * hidden_size +
        num_layers * (4 * hidden_size * hidden_size + 2 * hidden_size * intermediate_size)) * f32_size;
    const attn_per_layer = (bh * ss * head_dim + ss * hidden_size) * f32_size;
    if (metal_reuse_live_set) {
        const hidden_workspace = batch_size * seq_len * (hidden_size + intermediate_size) * f32_size;
        return weights + 3 * attn_per_layer + 4 * hidden_workspace;
    }
    return weights + 2 * num_layers * attn_per_layer;
}

fn metalBufferReuseEnabledForPreflight() bool {
    const raw = std.c.getenv("TERMITE_METAL_BUFFER_REUSE") orelse return true;
    const value = std.mem.span(raw);
    if (value.len == 0) return true;
    return !(value[0] == '0' or value[0] == 'f' or value[0] == 'F');
}

fn printUsage() void {
    // Zig 0.16: std.debug.print requires a format tuple. For plain string
    // output, use a no-arg format with the text inlined.
    std.debug.print("{s}", .{
        \\usage: train-gliner2-autodiff --model-dir <path> --train-data <path> --out-dir <path> [options]
        \\
        \\required:
        \\  --model-dir <path>        Directory with DeBERTa model (config.json + model.safetensors)
        \\  --train-data <path>       JSONL training data with {text, entities} per line
        \\  --out-dir <path>          Output directory for LoRA adapter weights
        \\
        \\options:
        \\  --epochs <n>              Number of training epochs (default: 10)
        \\  --batch-size <n>          Examples per step (default: 16)
        \\  --seq-len <n>             Max sequence length (default: 256)
        \\  --learning-rate, --lr <f> Learning rate (default: 5e-4)
        \\  --weight-decay <f>        AdamW weight decay (default: 0)
        \\  --lora-rank <n>           LoRA rank (default: 16)
        \\  --lora-alpha <f>          LoRA alpha scaling (default: 32)
        \\  --lora-dropout <f>        LoRA dropout probability (default: 0.1)
        \\  --lora-targets <csv>      Target module groups (default: encoder,span_rep,classifier,count_embed,count_pred)
        \\  --num-classes <n>         Entity classes incl. O tag (default: 5)
        \\  --entity-types <csv>      Entity label order for classes 1..N
        \\  --objective <name>        token, span-start, or gliner2-total-loss (default: token)
        \\  --max-span-width <n>      Max span width for span-start objective (default: 4)
        \\  --span-loss <name>        bce or mse for span-start labels (default: bce)
        \\  --span-loss-reduction <r> mean or sum (default: mean; sum matches upstream GLiNER2)
        \\  --span-positive-weight <f> Positive span-label loss weight (default: 32)
        \\  --span-label-positive-weights <csv> Per-label positive weights, e.g. person=32,organization=96
        \\  --span-negative-weight <f> Negative span-label loss weight (default: 1)
        \\  --span-hard-negative-weight <f> Extra negative weight for spans overlapping gold entities (default: 1)
        \\  --span-negative-mask-rate <f> Randomly mask this fraction of negative span labels (default: 0)
        \\  --max-examples <n>        Cap on training examples (default: 0 = all)
        \\  --max-grad-norm <f>       Gradient clipping norm (default: 1.0)
        \\  --grad-accum <n>          Gradient accumulation steps (default: 1)
        \\  --seed <n>                RNG seed (default: 42)
        \\  --initial-adapter-checkpoint <path> Seed LoRA weights from a PEFT safetensors checkpoint
        \\  --backend <name>          auto, metal, mlx, or native (default: auto)
        \\  --compiled-required       Fail if the requested compiled backend cannot run
        \\  --lora-only-trainables    Freeze regular task-head params; train LoRA params only
        \\  --deterministic           Disable per-step stochastic regularization (forces lora-dropout=0
        \\                            and span-negative-mask-rate=0; prints a warning when overriding)
        \\  --eval-strategy <name>    epoch, steps, or none (default: epoch; eval metric is avg train loss)
        \\  --eval-steps <n>          Eval every N steps when --eval-strategy steps
        \\  --save-best               Snapshot adapters to <out-dir>/best when the eval metric improves
        \\  --allow-large-memory      Proceed even when the estimated peak memory exceeds the safe
        \\                            budget (~60% of physical RAM); risks a system-wide OOM
        \\  --report-to <name>        stdout or jsonl; jsonl emits one JSON object per step/eval to stdout
        \\  --dump-span-parity        Print first span-start batch logits/label/mask BCE stats as JSON
        \\  --dump-optimizer-parity   Print per-step LoRA weight/Adam m/v state heads as JSON
        \\                            (GLINER2_OPT_PARITY lines, host optimizer paths only)
        \\
        \\notes:
        \\  Tokenization uses gliner2_data.Tokenizer.initGLiNER2HF and the
        \\  GLiNER2 prompt format backed by the model tokenizer files.
        \\
        \\example:
        \\  train-gliner2-autodiff --model-dir /models/deberta-v3-base \
        \\    --train-data /data/ner/train.jsonl --out-dir /output/lora \
        \\    --epochs 5 --batch-size 16 --num-classes 7 --learning-rate 2e-5
        \\  train-gliner2-autodiff --model-dir /models/gliner2 \
        \\    --train-data /data/ner/train.jsonl --out-dir /output/span-lora \
        \\    --objective span-start --max-span-width 4
        \\
    });
}

fn parseObjective(value: []const u8) !gliner2_autodiff.GlinerObjective {
    if (std.mem.eql(u8, value, "token")) return .token;
    if (std.mem.eql(u8, value, "span-start") or std.mem.eql(u8, value, "span_start")) return .span_start;
    if (std.mem.eql(u8, value, "gliner2-total-loss") or std.mem.eql(u8, value, "gliner2_total_loss")) return .gliner2_total_loss;
    print("error: unsupported --objective '{s}' (expected token, span-start, or gliner2-total-loss)\n", .{value});
    return error.InvalidObjective;
}

fn parseSpanLoss(value: []const u8) !gliner2_autodiff.SpanStartLossKind {
    if (std.mem.eql(u8, value, "bce") or std.mem.eql(u8, value, "binary-cross-entropy")) return .bce;
    if (std.mem.eql(u8, value, "mse")) return .mse;
    print("error: unsupported --span-loss '{s}' (expected bce or mse)\n", .{value});
    return error.InvalidSpanLoss;
}

fn parseSpanLossReduction(value: []const u8) !gliner2_autodiff.SpanStartLossReduction {
    if (std.mem.eql(u8, value, "mean")) return .mean;
    if (std.mem.eql(u8, value, "sum")) return .sum;
    print("error: unsupported --span-loss-reduction '{s}' (expected mean or sum)\n", .{value});
    return error.InvalidSpanLossReduction;
}

fn resolveSpanLabelPositiveWeights(
    allocator: std.mem.Allocator,
    csv: ?[]const u8,
    entity_labels: []const []const u8,
    default_weight: f32,
) ![]f32 {
    const weights = try allocator.alloc(f32, entity_labels.len);
    errdefer allocator.free(weights);
    @memset(weights, default_weight);
    if (csv == null) return weights;

    var seen = try allocator.alloc(bool, entity_labels.len);
    defer allocator.free(seen);
    @memset(seen, false);

    var iter = std.mem.splitScalar(u8, csv.?, ',');
    while (iter.next()) |raw| {
        const item = std.mem.trim(u8, raw, " \t\r\n");
        if (item.len == 0) continue;
        const eq_idx = std.mem.indexOfScalar(u8, item, '=') orelse return error.InvalidSpanLabelPositiveWeights;
        const label = std.mem.trim(u8, item[0..eq_idx], " \t\r\n");
        const value_text = std.mem.trim(u8, item[eq_idx + 1 ..], " \t\r\n");
        if (label.len == 0 or value_text.len == 0) return error.InvalidSpanLabelPositiveWeights;
        const label_idx = indexOfEntityLabel(entity_labels, label) orelse {
            print("error: unknown label in --span-label-positive-weights: {s}\n", .{label});
            return error.UnknownSpanLabelPositiveWeight;
        };
        if (seen[label_idx]) return error.DuplicateSpanLabelPositiveWeight;
        const weight = try std.fmt.parseFloat(f32, value_text);
        if (!std.math.isFinite(weight) or weight <= 0.0) return error.InvalidSpanPositiveWeight;
        weights[label_idx] = weight;
        seen[label_idx] = true;
    }
    return weights;
}

fn parseEntityTypesCsvOwned(allocator: std.mem.Allocator, csv: []const u8) ![][]const u8 {
    var out = std.ArrayListUnmanaged([]const u8).empty;
    errdefer {
        for (out.items) |item| allocator.free(item);
        out.deinit(allocator);
    }

    var iter = std.mem.splitScalar(u8, csv, ',');
    while (iter.next()) |raw| {
        const item = std.mem.trim(u8, raw, " \t\r\n");
        if (item.len == 0) continue;
        for (out.items) |existing| {
            if (std.mem.eql(u8, existing, item)) return error.DuplicateEntityType;
        }
        try out.append(allocator, try allocator.dupe(u8, item));
    }
    if (out.items.len == 0) return error.NoEntityTypesProvided;
    return out.toOwnedSlice(allocator);
}

fn dupeStringSlice(allocator: std.mem.Allocator, items: []const []const u8) ![][]const u8 {
    const out = try allocator.alloc([]const u8, items.len);
    errdefer allocator.free(out);
    for (items, 0..) |item, idx| {
        out[idx] = try allocator.dupe(u8, item);
    }
    return out;
}

fn shuffleExamplesAndRecords(prng: *std.Random, examples: []gliner2_data.Example, records: []gliner2_data.UpstreamRecord) void {
    if (examples.len != records.len) return;
    var i = examples.len;
    while (i > 1) {
        i -= 1;
        const j = prng.intRangeLessThan(usize, 0, i + 1);
        std.mem.swap(gliner2_data.Example, &examples[i], &examples[j]);
        std.mem.swap(gliner2_data.UpstreamRecord, &records[i], &records[j]);
    }
}

/// Activation (gradient) checkpointing config, env-gated. Enable with
/// `TERMITE_GLINER2_ACTIVATION_CHECKPOINTING=1` to recompute non-checkpoint
/// forward activations during backward instead of keeping them live — bounds
/// peak activation memory (trades ~1.3-2x forward compute for memory), needed
/// for large batch/seq where the full activation tape OOMs the GPU. Optional
/// `TERMITE_GLINER2_CHECKPOINT_INTERVAL=N` (default 1) saves every N layers.
fn activationCheckpointConfig() ?ml.graph.checkpoint.CheckpointConfig {
    const cstr = std.c.getenv("TERMITE_GLINER2_ACTIVATION_CHECKPOINTING") orelse return null;
    const val = std.mem.span(cstr);
    if (val.len == 0 or val[0] == '0') return null;
    var interval: u32 = 1;
    if (std.c.getenv("TERMITE_GLINER2_CHECKPOINT_INTERVAL")) |ic| {
        if (std.fmt.parseInt(u32, std.mem.trim(u8, std.mem.span(ic), " \t\r\n"), 10)) |n| {
            if (n >= 1) interval = n;
        } else |_| {}
    }
    print("  activation checkpointing: ON (every {d} layer(s))\n", .{interval});
    return .{ .strategy = .every_n_layers, .layer_interval = interval };
}

/// Max structure `gold_count` (capped at upstream's 19) across all
/// json_structures / relations tasks in the dataset. Drives
/// `structure_max_instances`: `1` keeps the legacy single-instance structure
/// loss; `>1` activates the per-instance count-conditioned einsum.
fn computeMaxStructureInstances(records: []const gliner2_data.UpstreamRecord) u32 {
    var max_instances: u32 = 1;
    for (records) |record| {
        for (record.tasks) |task| {
            switch (task.kind) {
                .json_structures, .relations => {
                    const c: u32 = @intCast(@min(task.count, @as(usize, 19)));
                    if (c > max_instances) max_instances = c;
                },
                else => {},
            }
        }
    }
    return max_instances;
}

fn fillGliner2TotalLossTargetsFromRecords(
    allocator: std.mem.Allocator,
    encoded: *const gliner2_data.EncodedBatch,
    records: []const gliner2_data.UpstreamRecord,
    entity_labels: []const []const u8,
    max_instances: u32,
    out: []f32,
) !gliner2_autodiff.SpanStartTargetStats {
    if (records.len != encoded.batch_size) return error.InvalidGlinerBatchShape;
    if (entity_labels.len != encoded.num_entity_types) return error.EntityTypeCountMismatch;
    const E = encoded.num_entity_types;
    const total_width = gliner2_autodiff.gliner2TotalLossTargetWidthEx(E, max_instances);
    const rows = encoded.batch_size * encoded.max_spans;
    if (out.len != rows * total_width) return error.InvalidGlinerSpanTargetShape;
    @memset(out, 0.0);

    var stats = gliner2_autodiff.SpanStartTargetStats{ .entity_type_count = E };

    const cls_labels_offset = gliner2_autodiff.gliner2TotalLossClassificationLabelsOffset(E);
    const cls_mask_offset = gliner2_autodiff.gliner2TotalLossClassificationMaskOffset(E);
    const count_labels_offset = gliner2_autodiff.gliner2TotalLossCountLabelsOffset(E);
    const count_mask_offset = gliner2_autodiff.gliner2TotalLossCountMaskOffset(E);
    const parent_idx_offset = gliner2_autodiff.gliner2TotalLossParentIndexOffset(E);
    const schema_idx_offset = 2 * E;
    const row_idx_offset = schema_idx_offset + E;
    const count_idx_offset = row_idx_offset + E;
    const start_idx_offset = count_idx_offset + E;

    for (records, 0..) |record, sample_idx| {
        const word_pos_offset = sample_idx * encoded.max_words_per_sample;
        const schema_count = if (encoded.schema_counts.len > sample_idx)
            @as(usize, @intCast(@max(encoded.schema_counts[sample_idx], 0)))
        else
            record.tasks.len;
        for (0..encoded.max_spans) |span_idx| {
            const flat_span_idx = sample_idx * encoded.max_spans + span_idx;
            const row = out[flat_span_idx * total_width ..][0..total_width];
            for (0..E) |entity_type_idx| {
                const label_token_pos_raw = encoded.e_token_positions[sample_idx * E + entity_type_idx];
                const label_token_pos: usize = if (label_token_pos_raw >= 0) @intCast(label_token_pos_raw) else 0;
                row[schema_idx_offset + entity_type_idx] = @floatFromInt(sample_idx * encoded.max_length + label_token_pos);
                row[row_idx_offset + entity_type_idx] = @floatFromInt(flat_span_idx);
                const count_state = if (encoded.entity_type_kind.len > sample_idx * E + entity_type_idx and encoded.entity_type_kind[sample_idx * E + entity_type_idx] > 0)
                    @as(usize, @intCast(@max(encoded.entity_type_kind[sample_idx * E + entity_type_idx] - 2, 0)))
                else
                    0;
                row[count_idx_offset + entity_type_idx] = @floatFromInt(count_state * encoded.batch_size * E + sample_idx * E + entity_type_idx);
            }

            if (encoded.span_mask[flat_span_idx] <= 0.0) {
                stats.ignored_span_count += 1;
                continue;
            }
            const start_word_raw = encoded.span_indices[flat_span_idx * 2];
            const end_word_raw = encoded.span_indices[flat_span_idx * 2 + 1];
            if (start_word_raw < 0 or end_word_raw < start_word_raw) {
                stats.ignored_span_count += 1;
                continue;
            }
            const start_word: usize = @intCast(start_word_raw);
            const end_word: usize = @intCast(end_word_raw);
            if (start_word >= encoded.max_words_per_sample or end_word >= encoded.max_words_per_sample) return error.InvalidSpanWordIndex;
            const start_token_raw = encoded.first_token_positions[word_pos_offset + start_word];
            const end_token_raw = encoded.first_token_positions[word_pos_offset + end_word];
            if (start_token_raw < 0 or end_token_raw < 0) {
                stats.ignored_span_count += 1;
                continue;
            }
            row[start_idx_offset] = @floatFromInt(sample_idx * encoded.max_length + @as(usize, @intCast(start_token_raw)));
            row[start_idx_offset + 1] = @floatFromInt(sample_idx * encoded.max_length + @as(usize, @intCast(end_token_raw)));
            stats.valid_span_count += 1;
            for (0..E) |entity_type_idx| {
                const active = encoded.entity_type_kind.len > sample_idx * E + entity_type_idx and encoded.entity_type_kind[sample_idx * E + entity_type_idx] > 0;
                if (!active) continue;
                const label = encoded.span_labels[flat_span_idx * E + entity_type_idx];
                row[entity_type_idx] = label;
                row[E + entity_type_idx] = 1.0;
                if (label > 0.0) {
                    stats.positive_span_label_count += 1;
                    stats.positive_counts_by_entity_type[entity_type_idx] += 1;
                }
            }
        }

        const task_limit = @min(record.tasks.len, encoded.max_spans);
        for (record.tasks[0..task_limit], 0..) |task, task_idx| {
            if (task_idx >= schema_count) break;
            const row_idx = sample_idx * encoded.max_spans + task_idx;
            const row = out[row_idx * total_width ..][0..total_width];
            row[parent_idx_offset] = @floatFromInt(sample_idx * encoded.max_length + upstreamSchemaParentTokenPosition(encoded, sample_idx, task_idx));
            switch (task.kind) {
                .classifications => {
                    for (task.labels) |label| {
                        const label_idx = indexOfEntityLabel(entity_labels, label) orelse continue;
                        row[cls_mask_offset + label_idx] = 1.0;
                    }
                    for (task.true_labels) |label| {
                        const label_idx = indexOfEntityLabel(entity_labels, label) orelse continue;
                        row[cls_labels_offset + label_idx] = 1.0;
                        row[cls_mask_offset + label_idx] = 1.0;
                    }
                },
                .entities => {},
                .json_structures, .relations => {
                    // Upstream `_compute_sample_loss` skips schemas whose
                    // structure count is zero entirely (no structure loss and
                    // no count loss), so only emit a count target for
                    // schemas with at least one gold instance.
                    if (task.count > 0) {
                        const count = @min(task.count, @as(usize, 19));
                        row[count_labels_offset + count] = 1.0;
                        row[count_mask_offset] = 1.0;
                    }
                },
            }
        }
    }

    // Per-instance structure-loss trailing columns (only when activated):
    // gold_count[E] (the schema's instance count for active fields, broadcast
    // across the sample's spans) and instance_idx[E] (the document-order
    // instance each located gold (span, field) belongs to).
    if (max_instances > 1) {
        const inst_offset = gliner2_autodiff.gliner2TotalLossInstanceIdxOffset(E);
        const gc_offset = gliner2_autodiff.gliner2TotalLossGoldCountOffset(E);
        const max_span_width = if (encoded.max_words_per_sample > 0)
            encoded.max_spans / encoded.max_words_per_sample
        else
            0;
        for (records, 0..) |record, sample_idx| {
            for (0..encoded.max_spans) |span_idx| {
                const flat = sample_idx * encoded.max_spans + span_idx;
                const row = out[flat * total_width ..][0..total_width];
                for (0..E) |label| {
                    const kind = encoded.entity_type_kind[sample_idx * E + label];
                    if (kind > 1) row[gc_offset + label] = @floatFromInt(kind - 1);
                }
            }

            const char_to_word = try gliner2_data.buildCharToWordMap(allocator, record.text);
            defer allocator.free(char_to_word);
            const prefix_word_count = record.prefix_tokens.len;
            for (record.tasks) |task| {
                switch (task.kind) {
                    .json_structures, .relations => {},
                    else => continue,
                }
                for (task.fields) |field| {
                    const span_idx = gliner2_data.locateUpstreamFieldSpanIdx(
                        field,
                        char_to_word,
                        prefix_word_count,
                        encoded.max_words_per_sample,
                        max_span_width,
                    ) orelse continue;
                    const label = try std.fmt.allocPrint(allocator, "{s}.{s}", .{ task.name, field.name });
                    defer allocator.free(label);
                    const label_idx = indexOfEntityLabel(entity_labels, label) orelse continue;
                    const flat = sample_idx * encoded.max_spans + span_idx;
                    out[flat * total_width + inst_offset + label_idx] = @floatFromInt(field.instance);
                }
            }
        }
    }

    return stats;
}

fn indexOfEntityLabel(entity_labels: []const []const u8, label: []const u8) ?usize {
    for (entity_labels, 0..) |candidate, idx| {
        if (std.mem.eql(u8, candidate, label)) return idx;
    }
    return null;
}

fn upstreamSchemaParentTokenPosition(encoded: *const gliner2_data.EncodedBatch, sample_idx: usize, task_idx: usize) usize {
    if (encoded.max_schemas == 0 or encoded.max_schema_specials == 0) return 1;
    if (task_idx >= encoded.max_schemas) return 1;
    const count_idx = sample_idx * encoded.max_schemas + task_idx;
    if (count_idx >= encoded.schema_special_counts.len or encoded.schema_special_counts[count_idx] <= 0) return 1;
    const pos_idx = (sample_idx * encoded.max_schemas + task_idx) * encoded.max_schema_specials;
    if (pos_idx >= encoded.schema_special_positions.len) return 1;
    const raw = encoded.schema_special_positions[pos_idx];
    if (raw < 0) return 1;
    return @intCast(raw);
}

fn stringSliceContains(items: []const []const u8, needle: []const u8) bool {
    for (items) |item| {
        if (std.mem.eql(u8, item, needle)) return true;
    }
    return false;
}

fn spanLossName(loss: gliner2_autodiff.SpanStartLossKind) []const u8 {
    return switch (loss) {
        .bce => "bce",
        .mse => "mse",
    };
}

fn spanLossReductionName(reduction: gliner2_autodiff.SpanStartLossReduction) []const u8 {
    return switch (reduction) {
        .mean => "mean",
        .sum => "sum",
    };
}

fn objectiveName(objective: gliner2_autodiff.GlinerObjective) []const u8 {
    return switch (objective) {
        .token => "token",
        .span_start => "span-start",
        .gliner2_total_loss => "gliner2-total-loss",
    };
}

fn envFlag(name: [:0]const u8) bool {
    const value = std.c.getenv(name) orelse return false;
    const slice = std.mem.span(value);
    return std.mem.eql(u8, slice, "1") or
        std.ascii.eqlIgnoreCase(slice, "true") or
        std.ascii.eqlIgnoreCase(slice, "yes") or
        std.ascii.eqlIgnoreCase(slice, "on");
}

test "resolveSpanLabelPositiveWeights applies defaults and overrides" {
    const allocator = std.testing.allocator;
    const labels = [_][]const u8{ "location", "organization", "person" };
    const weights = try resolveSpanLabelPositiveWeights(allocator, "organization=96,person=48", labels[0..], 32.0);
    defer allocator.free(weights);

    try std.testing.expectEqual(@as(usize, 3), weights.len);
    try std.testing.expectEqual(@as(f32, 32.0), weights[0]);
    try std.testing.expectEqual(@as(f32, 96.0), weights[1]);
    try std.testing.expectEqual(@as(f32, 48.0), weights[2]);
}

test "resolveSpanLabelPositiveWeights rejects unknown labels" {
    const allocator = std.testing.allocator;
    const labels = [_][]const u8{ "location", "organization", "person" };
    try std.testing.expectError(
        error.UnknownSpanLabelPositiveWeight,
        resolveSpanLabelPositiveWeights(allocator, "product=96", labels[0..], 32.0),
    );
}

test "parseEntityTypesCsvOwned preserves caller order" {
    const allocator = std.testing.allocator;
    const labels = try parseEntityTypesCsvOwned(allocator, "person, organization,location");
    defer {
        for (labels) |label| allocator.free(label);
        allocator.free(labels);
    }

    try std.testing.expectEqual(@as(usize, 3), labels.len);
    try std.testing.expectEqualStrings("person", labels[0]);
    try std.testing.expectEqualStrings("organization", labels[1]);
    try std.testing.expectEqualStrings("location", labels[2]);
}
