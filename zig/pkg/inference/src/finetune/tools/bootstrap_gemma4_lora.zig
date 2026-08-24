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

const std = @import("std");
const inference = @import("inference_internal");
const finetune = inference.finetune.gemma4;
const peft = inference.finetune.peft;

pub const Options = struct {
    model_dir: []const u8,
    out_dir: []const u8,
    bootstrap: finetune.BootstrapOptions,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    var argv: std.ArrayListUnmanaged([]const u8) = .empty;
    defer argv.deinit(allocator);
    while (args.next()) |arg| try argv.append(allocator, arg);
    try runFromArgs(allocator, init.io, argv.items);
}

pub fn runFromArgs(allocator: std.mem.Allocator, io: std.Io, argv: []const []const u8) !void {
    if (argv.len == 1 and isHelpArg(argv[0])) {
        printUsage();
        return;
    }
    var model_dir: ?[]const u8 = null;
    var out_dir: ?[]const u8 = null;
    var named_interface = false;
    var rank: usize = 16;
    var alpha: f32 = 32.0;
    var rank_set = false;
    var alpha_set = false;
    var rank_alpha_flag_seen = false;
    var base_model_name_or_path: ?[]const u8 = null;
    var layer_name: ?[]const u8 = null;
    var target_preset: ?peft.TargetPreset = null;
    var gemma4_target_preset: ?finetune.Gemma4LoRATargetPreset = null;
    var target_modules: ?[]const []const u8 = null;
    defer if (target_modules) |modules| allocator.free(modules);
    var use_dora = false;
    var init_lora_weights: ?[]const u8 = null;
    var initialization_seed: u64 = 0;
    var eva_stats_path: ?[]const u8 = null;
    var lora_ga_stats_path: ?[]const u8 = null;
    var recursive_shared_block_size: ?usize = null;
    var recursive_init_strategy: []const u8 = "average_residual_svd";

    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "--model")) {
            named_interface = true;
            i += 1;
            if (i >= argv.len or model_dir != null) return usageError();
            model_dir = argv[i];
        } else if (std.mem.eql(u8, arg, "--out")) {
            named_interface = true;
            i += 1;
            if (i >= argv.len or out_dir != null) return usageError();
            out_dir = argv[i];
        } else if (std.mem.eql(u8, arg, "--rank")) {
            if (rank_set) return usageError();
            i += 1;
            if (i >= argv.len) return usageError();
            rank = try std.fmt.parseUnsigned(usize, argv[i], 10);
            rank_set = true;
            rank_alpha_flag_seen = true;
        } else if (std.mem.eql(u8, arg, "--alpha")) {
            if (alpha_set) return usageError();
            i += 1;
            if (i >= argv.len) return usageError();
            alpha = try std.fmt.parseFloat(f32, argv[i]);
            alpha_set = true;
            rank_alpha_flag_seen = true;
        } else if (std.mem.eql(u8, arg, "--layer-name") or std.mem.eql(u8, arg, "--layer")) {
            i += 1;
            if (i >= argv.len) return usageError();
            layer_name = argv[i];
        } else if (std.mem.eql(u8, arg, "--target-preset")) {
            i += 1;
            if (i >= argv.len) return usageError();
            const preset_name = argv[i];
            if (finetune.parseGemma4LoRATargetPreset(preset_name)) |preset| {
                gemma4_target_preset = preset;
            } else {
                target_preset = peft.parseTargetPreset(preset_name) orelse return usageError();
            }
        } else if (std.mem.eql(u8, arg, "--target-modules")) {
            if (target_modules != null) return usageError();
            i += 1;
            if (i >= argv.len) return usageError();
            target_modules = try parseTargetModules(allocator, argv[i]);
        } else if (std.mem.eql(u8, arg, "--use-dora")) {
            use_dora = true;
        } else if (std.mem.eql(u8, arg, "--init-lora-weights")) {
            i += 1;
            if (i >= argv.len) return usageError();
            init_lora_weights = argv[i];
        } else if (std.mem.eql(u8, arg, "--initialization-seed")) {
            i += 1;
            if (i >= argv.len) return usageError();
            initialization_seed = try std.fmt.parseUnsigned(u64, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--eva-stats")) {
            i += 1;
            if (i >= argv.len) return usageError();
            eva_stats_path = argv[i];
        } else if (std.mem.eql(u8, arg, "--lora-ga-stats")) {
            i += 1;
            if (i >= argv.len) return usageError();
            lora_ga_stats_path = argv[i];
        } else if (std.mem.eql(u8, arg, "--recursive-shared-block-size")) {
            i += 1;
            if (i >= argv.len) return usageError();
            recursive_shared_block_size = try std.fmt.parseUnsigned(usize, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--recursive-init")) {
            i += 1;
            if (i >= argv.len) return usageError();
            recursive_init_strategy = argv[i];
        } else if (std.mem.startsWith(u8, arg, "--")) {
            return usageError();
        } else if (model_dir == null) {
            model_dir = arg;
        } else if (out_dir == null) {
            out_dir = arg;
        } else if (!rank_alpha_flag_seen and !rank_set) {
            rank = try std.fmt.parseUnsigned(usize, arg, 10);
            rank_set = true;
        } else if (!rank_alpha_flag_seen and !alpha_set) {
            alpha = try std.fmt.parseFloat(f32, arg);
            alpha_set = true;
        } else if (base_model_name_or_path == null) {
            base_model_name_or_path = arg;
        } else {
            return usageError();
        }
    }

    const selection_count = @intFromBool(target_modules != null) +
        @intFromBool(target_preset != null) +
        @intFromBool(gemma4_target_preset != null);
    if (selection_count > 1) return usageError();
    if (named_interface and selection_count == 0) return error.MissingGemma4TargetSelection;
    if (named_interface and target_preset != null) return error.UnsupportedLoRATargetPreset;

    try run(allocator, io, .{
        .model_dir = model_dir orelse return usageError(),
        .out_dir = out_dir orelse return usageError(),
        .bootstrap = .{
            .rank = rank,
            .alpha = alpha,
            .base_model_name_or_path = base_model_name_or_path,
            .layer_name = layer_name,
            .target_modules = target_modules,
            .target_preset = target_preset,
            .gemma4_target_preset = gemma4_target_preset,
            .use_dora = use_dora,
            .init_lora_weights = init_lora_weights,
            .initialization_seed = initialization_seed,
            .eva_stats_path = eva_stats_path,
            .lora_ga_stats_path = lora_ga_stats_path,
            .recursive_shared_block_size = recursive_shared_block_size,
            .recursive_init_strategy = recursive_init_strategy,
        },
    });
}

pub fn run(allocator: std.mem.Allocator, io: std.Io, options: Options) !void {
    if (options.bootstrap.use_dora) return error.DoRAAutodiffNotYetSupported;
    if (options.bootstrap.init_lora_weights) |initializer| {
        if (!std.ascii.eqlIgnoreCase(initializer, "default")) return error.Gemma4InitializerNotSupported;
    }
    var summary = try finetune.bootstrapLoRABundle(allocator, options.model_dir, options.out_dir, options.bootstrap);
    defer finetune.freeBootstrapSummary(allocator, &summary);

    const stdout = std.Io.File.stdout();
    var buf: [4096]u8 = undefined;
    var writer = stdout.writer(io, &buf);
    try std.json.Stringify.value(summary, .{ .whitespace = .indent_2 }, &writer.interface);
    try writer.interface.writeByte('\n');
    try writer.interface.flush();
}

fn usageError() error{InvalidArguments} {
    printUsage();
    return error.InvalidArguments;
}

pub fn printUsage() void {
    std.debug.print(
        \\usage: antfly inference finetune adapter bootstrap gemma4 --model <dir> --out <dir> \\
        \\       --target-preset text-all-linear|peft-qv
        \\       [--rank <n>] [--alpha <float>]
        \\       [--initialization-seed <u64>]
        \\       [--target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]
        \\
        \\Legacy positional form (deprecated for one release):
        \\  antfly inference finetune adapter bootstrap gemma4 <model_dir> <out_dir> [rank] [alpha] [base_model_name_or_path]
        \\
        \\Production Gemma4 bootstrap accepts standard LoRA initialization only.
        \\
    , .{});
}

fn isHelpArg(arg: []const u8) bool {
    return std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "help");
}

fn parseTargetModules(allocator: std.mem.Allocator, csv: []const u8) ![]const []const u8 {
    var modules = std.ArrayList([]const u8).empty;
    errdefer modules.deinit(allocator);
    var it = std.mem.splitScalar(u8, csv, ',');
    while (it.next()) |raw| {
        const item = std.mem.trim(u8, raw, " \t\r\n");
        if (item.len == 0) continue;
        try modules.append(allocator, item);
    }
    if (modules.items.len == 0) return error.InvalidArguments;
    return modules.toOwnedSlice(allocator);
}
