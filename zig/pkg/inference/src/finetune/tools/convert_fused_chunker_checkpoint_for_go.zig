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
const compat = inference.io.compat;
const safetensors = inference.models.safetensors;
const st = inference.finetune.safetensors_checkpoint;

const Options = struct {
    checkpoint: []const u8,
    out: []const u8,
    summary: ?[]const u8 = null,
    optimizer: ?[]const u8 = null,
    optimizer_out: ?[]const u8 = null,
};

const TensorSet = struct {
    allocator: std.mem.Allocator,
    tensors: std.ArrayListUnmanaged(st.NamedTensor) = .empty,
    names: std.ArrayListUnmanaged([]u8) = .empty,
    shapes: std.ArrayListUnmanaged([]usize) = .empty,
    data: std.ArrayListUnmanaged([]f32) = .empty,

    fn init(allocator: std.mem.Allocator) TensorSet {
        return .{ .allocator = allocator };
    }

    fn deinit(self: *TensorSet) void {
        for (self.names.items) |name| self.allocator.free(name);
        self.names.deinit(self.allocator);
        for (self.shapes.items) |shape| self.allocator.free(shape);
        self.shapes.deinit(self.allocator);
        for (self.data.items) |data| self.allocator.free(data);
        self.data.deinit(self.allocator);
        self.tensors.deinit(self.allocator);
    }

    fn appendOwned(self: *TensorSet, name: []const u8, values: []f32, shape: []const usize) !void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_shape = try self.allocator.dupe(usize, shape);
        errdefer self.allocator.free(owned_shape);
        try self.names.append(self.allocator, owned_name);
        try self.shapes.append(self.allocator, owned_shape);
        try self.data.append(self.allocator, values);
        try self.tensors.append(self.allocator, .{
            .name = owned_name,
            .data = values,
            .shape = owned_shape,
        });
    }

    fn appendCopy(self: *TensorSet, name: []const u8, values: []const f32, shape: []const usize) !void {
        const owned_values = try self.allocator.dupe(f32, values);
        errdefer self.allocator.free(owned_values);
        try self.appendOwned(name, owned_values, shape);
    }

    fn appendTranspose2D(
        self: *TensorSet,
        name: []const u8,
        values: []const f32,
        rows: usize,
        cols: usize,
    ) !void {
        if (values.len != rows * cols) return error.InvalidTensorShape;
        const transposed = try self.allocator.alloc(f32, values.len);
        errdefer self.allocator.free(transposed);
        for (0..rows) |r| {
            for (0..cols) |c| {
                transposed[c * rows + r] = values[r * cols + c];
            }
        }
        const shape = [_]usize{ cols, rows };
        try self.appendOwned(name, transposed, &shape);
    }
};

const ConvertStats = struct {
    mapped: usize = 0,
    skipped_metadata: usize = 0,
    skipped_unmapped: usize = 0,
    missing_required: usize = 0,
    optimizer_mapped: usize = 0,
    optimizer_lora_mapped: usize = 0,
    optimizer_missing_lora_state: bool = false,
};

const OptimizerConvertStats = struct {
    mapped: usize = 0,
    lora_mapped: usize = 0,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    const opts = try parseOptions(&args) orelse return;
    const stats = try convert(allocator, opts);
    try writeSummary(allocator, opts, stats);
}

fn parseOptions(args: *std.process.Args.Iterator) !?Options {
    var opts: Options = .{
        .checkpoint = "",
        .out = "",
    };
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--checkpoint")) {
            opts.checkpoint = args.next() orelse return error.MissingCheckpoint;
        } else if (std.mem.eql(u8, arg, "--out")) {
            opts.out = args.next() orelse return error.MissingOut;
        } else if (std.mem.eql(u8, arg, "--summary")) {
            opts.summary = args.next() orelse return error.MissingSummary;
        } else if (std.mem.eql(u8, arg, "--optimizer")) {
            opts.optimizer = args.next() orelse return error.MissingOptimizer;
        } else if (std.mem.eql(u8, arg, "--optimizer-out")) {
            opts.optimizer_out = args.next() orelse return error.MissingOptimizerOut;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            usage();
            return null;
        } else {
            std.debug.print("unknown argument: {s}\n", .{arg});
            usage();
            return error.InvalidArgument;
        }
    }

    if (opts.checkpoint.len == 0 or opts.out.len == 0) {
        usage();
        return error.InvalidArgument;
    }
    if (opts.optimizer != null and opts.optimizer_out == null) return error.MissingOptimizerOut;
    if (opts.optimizer == null and opts.optimizer_out != null) return error.MissingOptimizer;
    return opts;
}

fn usage() void {
    std.debug.print(
        \\usage: convert-fused-chunker-checkpoint-for-go --checkpoint <zig.safetensors> --out <go.safetensors> [--summary <json>] [--optimizer <zig_opt.safetensors> --optimizer-out <go_opt.safetensors>]
        \\
    , .{});
}

fn convert(allocator: std.mem.Allocator, opts: Options) !ConvertStats {
    var stats = ConvertStats{};
    var output = TensorSet.init(allocator);
    defer output.deinit();

    const file_bytes = try compat.cwd().readFileAlloc(compat.io(), opts.checkpoint, allocator, .unlimited);
    var reader = try safetensors.MMapReader.fromBytes(allocator, file_bytes);
    defer reader.deinit();

    const names = try reader.header.tensorNames(allocator);
    defer allocator.free(names);

    for (names) |name| {
        if (std.mem.eql(u8, name, "lora_rank") or std.mem.eql(u8, name, "lora_alpha")) {
            stats.skipped_metadata += 1;
            continue;
        }

        var tensor = try reader.readTensor(name);
        defer tensor.deinit();
        if (tensor.dtype != .f32) return error.InvalidTensorDType;
        const values = tensor.asFloat32();

        if (try appendMappedWeight(allocator, &output, name, values, tensor.shape)) {
            stats.mapped += 1;
        } else {
            stats.skipped_unmapped += 1;
        }
    }

    if (stats.skipped_unmapped != 0 or stats.mapped == 0) {
        stats.missing_required = stats.skipped_unmapped;
        return error.IncompleteCheckpointMapping;
    }
    try st.save(allocator, opts.out, output.tensors.items);

    if (opts.optimizer) |optimizer_path| {
        const out_path = opts.optimizer_out orelse return error.MissingOptimizerOut;
        const opt_stats = try convertOptimizer(allocator, optimizer_path, out_path);
        stats.optimizer_mapped = opt_stats.mapped;
        stats.optimizer_lora_mapped = opt_stats.lora_mapped;
        stats.optimizer_missing_lora_state = opt_stats.lora_mapped == 0;
    }

    return stats;
}

fn appendMappedWeight(
    allocator: std.mem.Allocator,
    output: *TensorSet,
    name: []const u8,
    values: []const f32,
    shape: []const i64,
) !bool {
    _ = allocator;
    if (std.mem.eql(u8, name, "w1")) {
        if (shape.len != 2) return error.InvalidTensorShape;
        try output.appendTranspose2D(
            "var:/fused_chunker_embedder/boundary_head/mlp_dense1/weight",
            values,
            @intCast(shape[0]),
            @intCast(shape[1]),
        );
        return true;
    }
    if (std.mem.eql(u8, name, "b1")) {
        const out_shape = [_]usize{@intCast(shape[0])};
        try output.appendCopy("var:/fused_chunker_embedder/boundary_head/mlp_dense1/bias", values, &out_shape);
        return true;
    }
    if (std.mem.eql(u8, name, "w2")) {
        if (shape.len != 2) return error.InvalidTensorShape;
        try output.appendTranspose2D(
            "var:/fused_chunker_embedder/boundary_head/mlp_dense2/weight",
            values,
            @intCast(shape[0]),
            @intCast(shape[1]),
        );
        return true;
    }
    if (std.mem.eql(u8, name, "b2")) {
        const out_shape = [_]usize{@intCast(shape[0])};
        try output.appendCopy("var:/fused_chunker_embedder/boundary_head/mlp_dense2/bias", values, &out_shape);
        return true;
    }

    if (try mapLoRAName(output.allocator, name)) |go_name| {
        defer output.allocator.free(go_name);
        if (shape.len != 2) return error.InvalidTensorShape;
        try output.appendTranspose2D(go_name, values, @intCast(shape[0]), @intCast(shape[1]));
        return true;
    }

    return false;
}

fn mapLoRAName(allocator: std.mem.Allocator, name: []const u8) !?[]u8 {
    const prefix = "model.layers.";
    if (!std.mem.startsWith(u8, name, prefix)) return null;

    var it = std.mem.splitScalar(u8, name[prefix.len..], '.');
    const layer = it.next() orelse return null;
    const scope = it.next() orelse return null;
    const projection = it.next() orelse return null;
    const leaf = it.next() orelse return null;
    if (it.next() != null) return null;

    const go_leaf = if (std.mem.eql(u8, leaf, "lora_a"))
        "lora_A"
    else if (std.mem.eql(u8, leaf, "lora_b"))
        "lora_B"
    else
        return null;

    if (!std.mem.eql(u8, scope, "attn") and !std.mem.eql(u8, scope, "mlp")) return null;
    return try std.fmt.allocPrint(
        allocator,
        "var:/fused_chunker_embedder/encoder/layer/{s}/{s}/{s}/{s}",
        .{ layer, scope, projection, go_leaf },
    );
}

fn convertOptimizer(allocator: std.mem.Allocator, optimizer_path: []const u8, out_path: []const u8) !OptimizerConvertStats {
    var output = TensorSet.init(allocator);
    defer output.deinit();

    const file_bytes = try compat.cwd().readFileAlloc(compat.io(), optimizer_path, allocator, .unlimited);
    var reader = try safetensors.MMapReader.fromBytes(allocator, file_bytes);
    defer reader.deinit();

    var stats = OptimizerConvertStats{};

    try appendOptimizerTensor(&reader, &output, "adam_step", "optimizer/step", .copy_1d, null, null);
    try appendOptimizerTensor(&reader, &output, "adam_m_w1", "adam_m/var:/fused_chunker_embedder/boundary_head/mlp_dense1/weight", .transpose_2d, 256, 768);
    try appendOptimizerTensor(&reader, &output, "adam_v_w1", "adam_v/var:/fused_chunker_embedder/boundary_head/mlp_dense1/weight", .transpose_2d, 256, 768);
    try appendOptimizerTensor(&reader, &output, "adam_m_b1", "adam_m/var:/fused_chunker_embedder/boundary_head/mlp_dense1/bias", .copy_1d, null, null);
    try appendOptimizerTensor(&reader, &output, "adam_v_b1", "adam_v/var:/fused_chunker_embedder/boundary_head/mlp_dense1/bias", .copy_1d, null, null);
    try appendOptimizerTensor(&reader, &output, "adam_m_w2", "adam_m/var:/fused_chunker_embedder/boundary_head/mlp_dense2/weight", .transpose_2d, 2, 256);
    try appendOptimizerTensor(&reader, &output, "adam_v_w2", "adam_v/var:/fused_chunker_embedder/boundary_head/mlp_dense2/weight", .transpose_2d, 2, 256);
    try appendOptimizerTensor(&reader, &output, "adam_m_b2", "adam_m/var:/fused_chunker_embedder/boundary_head/mlp_dense2/bias", .copy_1d, null, null);
    try appendOptimizerTensor(&reader, &output, "adam_v_b2", "adam_v/var:/fused_chunker_embedder/boundary_head/mlp_dense2/bias", .copy_1d, null, null);

    const after_head = output.tensors.items.len;
    const names = try reader.header.tensorNames(allocator);
    defer allocator.free(names);
    for (names) |name| {
        if (try appendMappedLoRAOptimizerTensor(allocator, &reader, &output, name)) {
            stats.lora_mapped += 1;
        }
    }

    try st.save(allocator, out_path, output.tensors.items);
    stats.mapped = output.tensors.items.len;
    if (after_head > stats.mapped) return error.InvalidOptimizerMapping;
    return stats;
}

const OptimizerTransform = enum { copy_1d, transpose_2d };

fn appendOptimizerTensor(
    reader: *const safetensors.MMapReader,
    output: *TensorSet,
    source_name: []const u8,
    go_name: []const u8,
    transform: OptimizerTransform,
    rows: ?usize,
    cols: ?usize,
) !void {
    if (reader.header.tensors.get(source_name) == null) return;
    var tensor = try reader.readTensor(source_name);
    defer tensor.deinit();
    if (tensor.dtype != .f32) return error.InvalidTensorDType;
    const values = tensor.asFloat32();
    switch (transform) {
        .copy_1d => {
            const shape = [_]usize{values.len};
            try output.appendCopy(go_name, values, &shape);
        },
        .transpose_2d => {
            try output.appendTranspose2D(go_name, values, rows.?, cols.?);
        },
    }
}

fn appendMappedLoRAOptimizerTensor(
    allocator: std.mem.Allocator,
    reader: *const safetensors.MMapReader,
    output: *TensorSet,
    source_name: []const u8,
) !bool {
    const go_name = try mapLoRAOptimizerName(allocator, source_name) orelse return false;
    defer allocator.free(go_name);

    var tensor = try reader.readTensor(source_name);
    defer tensor.deinit();
    if (tensor.dtype != .f32) return error.InvalidTensorDType;
    const values = tensor.asFloat32();
    if (tensor.shape.len == 2) {
        try output.appendTranspose2D(
            go_name,
            values,
            @intCast(tensor.shape[0]),
            @intCast(tensor.shape[1]),
        );
    } else if (tensor.shape.len == 1) {
        const dims = try legacyLoRAOptimizerShape(source_name, values.len);
        try output.appendTranspose2D(go_name, values, dims.rows, dims.cols);
    } else {
        return error.InvalidTensorShape;
    }
    return true;
}

const LoRAOptimizerShape = struct {
    rows: usize,
    cols: usize,
};

fn legacyLoRAOptimizerShape(source_name: []const u8, len: usize) !LoRAOptimizerShape {
    const parsed = parseLoRAOptimizerName(source_name) orelse return error.InvalidTensorShape;
    const rank: usize = 16;
    if (std.mem.eql(u8, parsed.matrix, "A")) {
        const in_features: usize = if (std.mem.eql(u8, parsed.module, "wo")) 1152 else 768;
        if (len != rank * in_features) return error.InvalidTensorShape;
        return .{ .rows = rank, .cols = in_features };
    }
    if (std.mem.eql(u8, parsed.matrix, "B")) {
        if (len != 768 * rank) return error.InvalidTensorShape;
        return .{ .rows = 768, .cols = rank };
    }
    return error.InvalidTensorShape;
}

const ParsedLoRAOptimizerName = struct {
    moment_prefix: []const u8,
    layer: []const u8,
    module: []const u8,
    matrix: []const u8,
};

fn parseLoRAOptimizerName(name: []const u8) ?ParsedLoRAOptimizerName {
    const moment_prefix = if (std.mem.startsWith(u8, name, "adam_m_lora."))
        "adam_m"
    else if (std.mem.startsWith(u8, name, "adam_v_lora."))
        "adam_v"
    else
        return null;
    const rest = name[moment_prefix.len + "_lora.".len ..];
    var it = std.mem.splitScalar(u8, rest, '.');
    const layer = it.next() orelse return null;
    const module = it.next() orelse return null;
    const matrix = it.next() orelse return null;
    if (it.next() != null) return null;
    if (!std.mem.eql(u8, matrix, "A") and !std.mem.eql(u8, matrix, "B")) return null;
    return .{
        .moment_prefix = moment_prefix,
        .layer = layer,
        .module = module,
        .matrix = matrix,
    };
}

fn mapLoRAOptimizerName(allocator: std.mem.Allocator, name: []const u8) !?[]u8 {
    const parsed = parseLoRAOptimizerName(name) orelse return null;
    const go_leaf = if (std.mem.eql(u8, parsed.matrix, "A")) "lora_A" else "lora_B";
    const target = loraOptimizerRuntimeTarget(parsed.module) orelse return null;
    return try std.fmt.allocPrint(
        allocator,
        "{s}/var:/fused_chunker_embedder/encoder/layer/{s}/{s}/{s}/{s}",
        .{ parsed.moment_prefix, parsed.layer, target.scope, target.projection, go_leaf },
    );
}

fn loraOptimizerRuntimeTarget(module_name: []const u8) ?struct { scope: []const u8, projection: []const u8 } {
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

fn writeSummary(allocator: std.mem.Allocator, opts: Options, stats: ConvertStats) !void {
    var buf: std.Io.Writer.Allocating = .init(allocator);
    defer buf.deinit();
    const w = &buf.writer;

    try w.print(
        "{{\"tool\":\"convert_fused_chunker_checkpoint_for_go\",\"schema_version\":1,\"status\":\"passed\",\"checkpoint\":\"{s}\",\"out\":\"{s}\",\"mapped_tensors\":{d},\"skipped_metadata\":{d},\"skipped_unmapped\":{d},\"missing_required\":{d},\"optimizer_mapped_tensors\":{d},\"optimizer_lora_mapped_tensors\":{d},\"optimizer_missing_lora_state\":{}}}\n",
        .{
            opts.checkpoint,
            opts.out,
            stats.mapped,
            stats.skipped_metadata,
            stats.skipped_unmapped,
            stats.missing_required,
            stats.optimizer_mapped,
            stats.optimizer_lora_mapped,
            stats.optimizer_missing_lora_state,
        },
    );

    if (opts.summary) |summary_path| {
        try compat.cwd().writeFile(compat.io(), .{ .sub_path = summary_path, .data = buf.written() });
    }
    var stdout_buf: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(compat.io(), &stdout_buf);
    try stdout.interface.writeAll(buf.written());
    try stdout.interface.flush();
}
