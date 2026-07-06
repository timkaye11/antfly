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
const quant_kernel_compiler = @import("graph/quant_kernel_compiler.zig");

const print = std.debug.print;

const Mode = enum {
    check,
    write,
    check_metal,
};

const GeneratedFile = struct {
    path: []const u8,
    data: []const u8,
    owned: bool = false,
};

const CompiledSourceData = struct {
    data: []const u8,
    owned: bool = false,
};

pub fn main(allocator: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
    const mode = try parseMode(args);
    const files = try generatedFiles(allocator);
    defer freeGeneratedFiles(allocator, files);
    for (files) |source| {
        switch (mode) {
            .check, .check_metal => try checkSource(allocator, io, source),
            .write => try writeSource(allocator, io, source),
        }
    }
    if (mode == .check_metal) try checkMetalArtifacts(allocator, io);
    print("quant kernel generated sources {s}: {d} files\n", .{ @tagName(mode), files.len });
}

fn parseMode(args: []const []const u8) !Mode {
    if (args.len == 0) return .check;
    if (args.len == 1 and std.mem.eql(u8, args[0], "--check")) return .check;
    if (args.len == 1 and std.mem.eql(u8, args[0], "--write")) return .write;
    if (args.len == 1 and std.mem.eql(u8, args[0], "--check-metal")) return .check_metal;
    printUsage();
    return error.InvalidArguments;
}

fn printUsage() void {
    print(
        \\usage: antfly inference quant-kernel-codegen [--check|--write|--check-metal]
        \\  Rewrites or verifies dev-generated quant kernel sources, manifests, and Metal artifacts.
        \\
    , .{});
}

fn generatedFiles(allocator: std.mem.Allocator) ![]GeneratedFile {
    const source_count = generatedSourceFileCount();
    const files = try allocator.alloc(GeneratedFile, source_count + 4);
    @memset(files, .{ .path = "", .data = "" });
    errdefer freeGeneratedFiles(allocator, files);
    var index: usize = 0;
    for (quant_kernel_compiler.first_generated_artifacts) |artifact| {
        const source = try compiledSourceForArtifact(allocator, artifact);
        files[index] = .{
            .path = artifact.source_path,
            .data = source.data,
            .owned = source.owned,
        };
        index += 1;
    }
    for (quant_kernel_compiler.first_generated_artifacts) |artifact| {
        if (artifact.generated_source_path.len != 0 and !std.mem.eql(u8, artifact.generated_source_path, artifact.source_path)) {
            const source = try compiledSourceForArtifact(allocator, artifact);
            files[index] = .{
                .path = artifact.generated_source_path,
                .data = source.data,
                .owned = source.owned,
            };
            index += 1;
        }
        if (artifact.backend == .metal) {
            if (quant_kernel_compiler.metalArtifactSourcePathForKernel(artifact.kernel_id)) |artifact_source_path| {
                if (!std.mem.eql(u8, artifact_source_path, artifact.source_path) and !std.mem.eql(u8, artifact_source_path, artifact.generated_source_path)) {
                    const source = try compiledSourceForArtifact(allocator, artifact);
                    files[index] = .{
                        .path = artifact_source_path,
                        .data = source.data,
                        .owned = source.owned,
                    };
                    index += 1;
                }
            }
        }
    }
    if (index != source_count) return error.GeneratedQuantKernelSourceCountMismatch;
    var manifest_index = index;
    files[manifest_index] = .{
        .path = quant_kernel_compiler.first_spec_manifest_path,
        .data = try quant_kernel_compiler.specManifestJson(allocator),
        .owned = true,
    };
    manifest_index += 1;
    files[manifest_index] = .{
        .path = quant_kernel_compiler.first_artifact_manifest_path,
        .data = try quant_kernel_compiler.artifactManifestJson(allocator),
        .owned = true,
    };
    manifest_index += 1;
    files[manifest_index] = .{
        .path = quant_kernel_compiler.first_benchmark_manifest_path,
        .data = try quant_kernel_compiler.benchmarkManifestJson(allocator),
        .owned = true,
    };
    manifest_index += 1;
    files[manifest_index] = .{
        .path = quant_kernel_compiler.first_conformance_manifest_path,
        .data = try quant_kernel_compiler.conformanceManifestJson(allocator),
        .owned = true,
    };
    return files;
}

fn compiledSourceForArtifact(
    allocator: std.mem.Allocator,
    artifact: quant_kernel_compiler.GeneratedArtifact,
) !CompiledSourceData {
    const compiled = quant_kernel_compiler.compileQuantKernelSource(.{
        .backend = artifact.backend,
        .format = artifact.format,
        .row_bucket = artifact.row_bucket,
        .epilogue = artifact.epilogue,
    }) orelse return error.MissingGeneratedQuantKernelSource;
    if (!std.mem.eql(u8, compiled.artifact.kernel_id, artifact.kernel_id)) {
        return error.MissingGeneratedQuantKernelSource;
    }
    const emitted = try quant_kernel_compiler.emitCompiledSource(allocator, compiled);
    errdefer emitted.deinit(allocator);
    if (!try quant_kernel_compiler.compiledSourceHeaderMatchesSource(allocator, compiled, emitted.data)) {
        return error.GeneratedQuantKernelSourceHeaderMismatch;
    }
    return .{ .data = emitted.data, .owned = emitted.owned };
}

fn generatedSourceFileCount() usize {
    var count = quant_kernel_compiler.first_generated_artifacts.len;
    for (quant_kernel_compiler.first_generated_artifacts) |artifact| {
        if (artifact.generated_source_path.len != 0 and !std.mem.eql(u8, artifact.generated_source_path, artifact.source_path)) count += 1;
        if (artifact.backend == .metal) {
            if (quant_kernel_compiler.metalArtifactSourcePathForKernel(artifact.kernel_id)) |artifact_source_path| {
                if (!std.mem.eql(u8, artifact_source_path, artifact.source_path) and !std.mem.eql(u8, artifact_source_path, artifact.generated_source_path)) count += 1;
            }
        }
    }
    return count;
}

fn freeGeneratedFiles(allocator: std.mem.Allocator, files: []GeneratedFile) void {
    for (files) |file| {
        if (file.owned) allocator.free(file.data);
    }
    allocator.free(files);
}

fn checkSource(allocator: std.mem.Allocator, io: std.Io, source: GeneratedFile) !void {
    const path = existingSourcePath(allocator, io, source.path) catch |err| {
        if (err == error.GeneratedQuantKernelSourceMissing) {
            print("quant kernel generated source missing: path={s}; run `zig build quant-kernel-codegen -- --write`\n", .{source.path});
        }
        return err;
    };
    defer if (path.owned) allocator.free(path.value);
    const max_bytes = @max(source.data.len + 1024 * 1024, 4 * 1024 * 1024);
    const actual = try std.Io.Dir.cwd().readFileAlloc(io, path.value, allocator, .limited(max_bytes));
    defer allocator.free(actual);
    if (!std.mem.eql(u8, source.data, actual)) {
        print(
            "quant kernel generated source stale: path={s} expected_bytes={d} actual_bytes={d}; run `zig build quant-kernel-codegen -- --write`\n",
            .{ path.value, source.data.len, actual.len },
        );
        return error.GeneratedQuantKernelSourceStale;
    }
}

fn writeSource(allocator: std.mem.Allocator, io: std.Io, source: GeneratedFile) !void {
    const path = try writableSourcePath(allocator, io, source.path);
    defer if (path.owned) allocator.free(path.value);
    try ensureParentDir(io, path.value);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path.value, .data = source.data });
}

const ResolvedPath = struct {
    value: []const u8,
    owned: bool = false,
};

fn existingSourcePath(allocator: std.mem.Allocator, io: std.Io, package_path: []const u8) !ResolvedPath {
    if (pathExists(io, package_path)) return .{ .value = package_path };
    const repo_path = try std.fmt.allocPrint(allocator, "zig/pkg/inference/{s}", .{package_path});
    if (pathExists(io, repo_path)) return .{ .value = repo_path, .owned = true };
    allocator.free(repo_path);
    return error.GeneratedQuantKernelSourceMissing;
}

fn writableSourcePath(allocator: std.mem.Allocator, io: std.Io, package_path: []const u8) !ResolvedPath {
    if (existingSourcePath(allocator, io, package_path)) |path| {
        return path;
    } else |err| switch (err) {
        error.GeneratedQuantKernelSourceMissing => {},
        else => return err,
    }
    if (pathExists(io, "zig/pkg/inference/build.zig")) {
        return .{
            .value = try std.fmt.allocPrint(allocator, "zig/pkg/inference/{s}", .{package_path}),
            .owned = true,
        };
    }
    return .{ .value = package_path };
}

fn pathExists(io: std.Io, path: []const u8) bool {
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

fn ensureParentDir(io: std.Io, path: []const u8) !void {
    const parent = std.fs.path.dirname(path) orelse return;
    if (parent.len == 0) return;
    try std.Io.Dir.cwd().createDirPath(io, parent);
}

fn checkMetalArtifacts(allocator: std.mem.Allocator, io: std.Io) !void {
    var count: usize = 0;
    for (quant_kernel_compiler.first_generated_artifacts) |artifact| {
        if (artifact.backend != .metal) continue;
        try checkMetalArtifact(allocator, io, artifact);
        count += 1;
    }
    print("quant kernel generated Metal check: {d} artifacts\n", .{count});
}

fn checkMetalArtifact(allocator: std.mem.Allocator, io: std.Io, artifact: quant_kernel_compiler.GeneratedArtifact) !void {
    const source_path = try existingSourcePath(allocator, io, artifact.source_path);
    defer if (source_path.owned) allocator.free(source_path.value);
    const output_path = metalCheckOutputPath(artifact.check_command) orelse return error.InvalidMetalCheckCommand;
    var child = try std.process.spawn(io, .{
        .argv = &.{ "xcrun", "--toolchain", "Metal", "metal", "-c", source_path.value, "-o", output_path },
        .stdin = .ignore,
        .stdout = .inherit,
        .stderr = .inherit,
    });
    const term = try child.wait(io);
    switch (term) {
        .exited => |code| if (code != 0) return error.MetalCheckFailed,
        else => return error.MetalCheckFailed,
    }
}

fn metalCheckOutputPath(command: []const u8) ?[]const u8 {
    var tokens = std.mem.tokenizeScalar(u8, command, ' ');
    while (tokens.next()) |token| {
        if (!std.mem.eql(u8, token, "-o")) continue;
        return tokens.next();
    }
    return null;
}

fn generatedFilePathExists(sources: []const GeneratedFile, path: []const u8) bool {
    for (sources) |source| {
        if (std.mem.eql(u8, source.path, path)) return true;
    }
    return false;
}

test "quant kernel codegen defaults to check mode" {
    try std.testing.expectEqual(Mode.check, try parseMode(&.{}));
    try std.testing.expectEqual(Mode.check, try parseMode(&.{"--check"}));
    try std.testing.expectEqual(Mode.write, try parseMode(&.{"--write"}));
    try std.testing.expectEqual(Mode.check_metal, try parseMode(&.{"--check-metal"}));
}

test "quant kernel codegen sources map to compiler artifacts" {
    const sources = try generatedFiles(std.testing.allocator);
    defer freeGeneratedFiles(std.testing.allocator, sources);

    try std.testing.expectEqual(generatedSourceFileCount() + 4, sources.len);
    for (quant_kernel_compiler.first_generated_artifacts, sources[0..quant_kernel_compiler.first_generated_artifacts.len]) |artifact, source| {
        try std.testing.expectEqualStrings(artifact.source_path, source.path);
        try std.testing.expect(source.data.len != 0);
    }

    const spec_manifest = sources[sources.len - 4];
    try std.testing.expectEqualStrings(quant_kernel_compiler.first_spec_manifest_path, spec_manifest.path);
    try std.testing.expect(std.mem.containsAtLeast(u8, spec_manifest.data, 1, "antfly.quant_kernel_specs.v1"));
    try std.testing.expect(std.mem.containsAtLeast(u8, spec_manifest.data, 1, "\"format_count\": 28"));
    try std.testing.expect(std.mem.containsAtLeast(u8, spec_manifest.data, 1, "\"format\": \"q4_k\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, spec_manifest.data, 1, "\"block_bytes\": 144"));
    try std.testing.expect(std.mem.containsAtLeast(u8, spec_manifest.data, 1, "\"name\": \"qs\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, spec_manifest.data, 1, "load block_q4_K"));

    const artifact_manifest = sources[sources.len - 3];
    try std.testing.expectEqualStrings(quant_kernel_compiler.first_artifact_manifest_path, artifact_manifest.path);
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, 1, quant_kernel_compiler.first_artifact_manifest_schema));
    const artifact_count = try std.fmt.allocPrint(std.testing.allocator, "\"artifact_count\": {d}", .{quant_kernel_compiler.first_generated_artifacts.len});
    defer std.testing.allocator.free(artifact_count);
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, 1, artifact_count));
    var runtime_evidence_count: usize = 0;
    var runtime_route_evidence_count: usize = 0;
    var promotion_evidence_count: usize = 0;
    var promotion_check_count: usize = 0;
    var metal_evidence_count: usize = 0;
    for (quant_kernel_compiler.first_generated_artifacts) |artifact| {
        try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, 1, artifact.kernel_id));
        try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, 1, artifact.source_path));
        try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, 1, artifact.check_command));
        if (artifact.runtime_evidence_command.len != 0) runtime_evidence_count += 1;
        if (quant_kernel_compiler.artifactNeedsRuntimeRouteEvidence(artifact)) runtime_route_evidence_count += 1;
        if (artifact.promotion_evidence_command.len != 0) promotion_evidence_count += 1;
        if (artifact.promotion_check_command.len != 0) promotion_check_count += 1;
        if (artifact.backend == .metal and artifact.runtime_evidence_command.len != 0) metal_evidence_count += 1;
    }
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, runtime_evidence_count, "\"runtime_evidence_command\":"));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, runtime_route_evidence_count, "\"runtime_route_evidence_command\":"));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, 1, "--runtime-route-kernel"));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, promotion_evidence_count, "\"promotion_evidence_command\":"));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, promotion_check_count, "\"promotion_check_command\":"));
    const checked_in_metal_evidence_count = try std.fmt.allocPrint(std.testing.allocator, "\"checked_in_metal_evidence_count\": {d}", .{quant_kernel_compiler.first_metal_runtime_evidence_count});
    defer std.testing.allocator.free(checked_in_metal_evidence_count);
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, 1, checked_in_metal_evidence_count));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, 1, quant_kernel_compiler.first_general_metal_q4_artifact_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, 1, quant_kernel_compiler.first_general_metal_q8_artifact_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, 1, quant_kernel_compiler.first_general_metal_q8_bias_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, 1, quant_kernel_compiler.first_general_metal_q5_artifact_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, 1, quant_kernel_compiler.first_general_metal_q6_artifact_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, 1, quant_kernel_compiler.first_lazy_benchmark.benchmark_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, 1, quant_kernel_compiler.first_lazy_benchmark_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, metal_evidence_count, quant_kernel_compiler.first_metal_runtime_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, 1, quant_kernel_compiler.first_metal_local_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, 1, quant_kernel_compiler.first_metal_runtime_route_all_build_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, 1, quant_kernel_compiler.first_metal_runtime_route_all_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, 1, quant_kernel_compiler.first_metal_runtime_route_all_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, metal_evidence_count, quant_kernel_compiler.first_metal_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, metal_evidence_count, quant_kernel_compiler.first_metal_promotion_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, metal_evidence_count, "--require-kernel"));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, metal_evidence_count, "\"metal_promotion_min_speedup\": 1.1"));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, metal_evidence_count, "\"metal_promotion_repeat_runs\": 5"));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, metal_evidence_count, "\"metal_promotion_warmup_repeat_runs\": 2"));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, quant_kernel_compiler.first_metal_runtime_evidence_count, "\"candidate_status\": \"promoted\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, 1, "\"candidate_status\": \"dev_only_candidate\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, artifact_manifest.data, 1, "\"candidate_status\": \"blocked_by_evidence\""));

    const benchmark_manifest = sources[sources.len - 2];
    try std.testing.expectEqualStrings(quant_kernel_compiler.first_benchmark_manifest_path, benchmark_manifest.path);
    try std.testing.expect(std.mem.containsAtLeast(u8, benchmark_manifest.data, 1, quant_kernel_compiler.first_benchmark_manifest_schema));
    try std.testing.expect(std.mem.containsAtLeast(u8, benchmark_manifest.data, 1, "\"benchmark_count\": 1"));
    try std.testing.expect(std.mem.containsAtLeast(u8, benchmark_manifest.data, 1, "\"metal_promotion_warmup_repeat_runs\": 2"));
    const metal_production_regression_case_count = try std.fmt.allocPrint(std.testing.allocator, "\"metal_production_regression_expected_case_count\": {d}", .{quant_kernel_compiler.first_metal_production_regression_expected_case_count});
    defer std.testing.allocator.free(metal_production_regression_case_count);
    try std.testing.expect(std.mem.containsAtLeast(u8, benchmark_manifest.data, 1, metal_production_regression_case_count));
    const metal_production_regression_case_fingerprint = try std.fmt.allocPrint(std.testing.allocator, "\"metal_production_regression_case_fingerprint\": {d}", .{quant_kernel_compiler.metalProductionBenchmarkCaseManifestFingerprint()});
    defer std.testing.allocator.free(metal_production_regression_case_fingerprint);
    try std.testing.expect(std.mem.containsAtLeast(u8, benchmark_manifest.data, 1, metal_production_regression_case_fingerprint));
    try std.testing.expect(std.mem.containsAtLeast(u8, benchmark_manifest.data, 1, "\"metal_production_regression_cases\": ["));
    try std.testing.expect(std.mem.containsAtLeast(u8, benchmark_manifest.data, 1, "\"name\": \"q6_k_rows_8_cols_7_bias_gelu\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, benchmark_manifest.data, 1, "\"production_kernel_id\": \"antfly_q6_k_small_batch_bias_gelu_msl_v1\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, benchmark_manifest.data, 1, quant_kernel_compiler.first_metal_production_regression_build_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, benchmark_manifest.data, 1, quant_kernel_compiler.first_metal_production_regression_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, benchmark_manifest.data, 1, quant_kernel_compiler.first_lazy_benchmark.generated_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, benchmark_manifest.data, 1, quant_kernel_compiler.first_lazy_benchmark.benchmark_command));

    const conformance_manifest = sources[sources.len - 1];
    try std.testing.expectEqualStrings(quant_kernel_compiler.first_conformance_manifest_path, conformance_manifest.path);
    try std.testing.expect(std.mem.containsAtLeast(u8, conformance_manifest.data, 1, "antfly.quant_kernel_conformance.v1"));
    try std.testing.expect(std.mem.containsAtLeast(u8, conformance_manifest.data, 1, "\"format\": \"q4_k\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, conformance_manifest.data, 1, "\"cuda_fallback_reason\": \"generated_artifact_missing\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, conformance_manifest.data, 1, "\"metal_fallback_reason\": \"generated_artifact_missing\""));
}

test "quant kernel codegen includes generated and artifact sidecar paths" {
    const sources = try generatedFiles(std.testing.allocator);
    defer freeGeneratedFiles(std.testing.allocator, sources);

    var generated_sidecar_count: usize = 0;
    var metal_artifact_path_count: usize = 0;
    for (quant_kernel_compiler.first_generated_artifacts) |artifact| {
        try std.testing.expect(generatedFilePathExists(sources, artifact.source_path));
        const primary_path = try existingSourcePath(std.testing.allocator, std.testing.io, artifact.source_path);
        defer if (primary_path.owned) std.testing.allocator.free(primary_path.value);

        if (artifact.generated_source_path.len != 0) {
            generated_sidecar_count += 1;
            try std.testing.expect(generatedFilePathExists(sources, artifact.generated_source_path));
            const generated_path = try existingSourcePath(std.testing.allocator, std.testing.io, artifact.generated_source_path);
            defer if (generated_path.owned) std.testing.allocator.free(generated_path.value);
        }

        if (artifact.backend == .metal) {
            if (quant_kernel_compiler.metalArtifactSourcePathForKernel(artifact.kernel_id)) |artifact_source_path| {
                metal_artifact_path_count += 1;
                try std.testing.expect(generatedFilePathExists(sources, artifact_source_path));
                const artifact_path = try existingSourcePath(std.testing.allocator, std.testing.io, artifact_source_path);
                defer if (artifact_path.owned) std.testing.allocator.free(artifact_path.value);
            }
        }
    }

    try std.testing.expect(generated_sidecar_count > 0);
    try std.testing.expect(metal_artifact_path_count > 0);
}

test "quant kernel codegen resolves package-root source paths" {
    const path = try existingSourcePath(
        std.testing.allocator,
        std.testing.io,
        quant_kernel_compiler.first_lazy_benchmark.generated_source_path,
    );
    defer if (path.owned) std.testing.allocator.free(path.value);
    try std.testing.expect(!path.owned);
    try std.testing.expectEqualStrings(quant_kernel_compiler.first_lazy_benchmark.generated_source_path, path.value);
}

test "quant kernel codegen can choose a writable path for missing package-root files" {
    const path = try writableSourcePath(std.testing.allocator, std.testing.io, "src/ops/cuda/generated/does_not_exist.cu");
    defer if (path.owned) std.testing.allocator.free(path.value);
    try std.testing.expect(!path.owned);
    try std.testing.expectEqualStrings("src/ops/cuda/generated/does_not_exist.cu", path.value);
}

test "quant kernel codegen reports missing check targets" {
    try std.testing.expectError(
        error.GeneratedQuantKernelSourceMissing,
        existingSourcePath(std.testing.allocator, std.testing.io, "src/ops/cuda/generated/does_not_exist.cu"),
    );
}

test "quant kernel codegen parses Metal check output path" {
    try std.testing.expectEqualStrings(
        "/tmp/antfly_q4_k_small_batch_bias_gelu_msl_v1.air",
        metalCheckOutputPath(quant_kernel_compiler.first_lazy_metal_check_command) orelse return error.MissingMetalCheckOutputPath,
    );
    try std.testing.expect(metalCheckOutputPath("xcrun --toolchain Metal metal -c source.metal") == null);
}

test "quant kernel codegen write creates parent directories" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const path = try std.fs.path.join(std.testing.allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "nested", "generated.cu" });
    defer std.testing.allocator.free(path);

    try writeSource(std.testing.allocator, std.testing.io, .{ .path = path, .data = "generated", .owned = false });
    const actual = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, std.testing.allocator, .limited(64));
    defer std.testing.allocator.free(actual);
    try std.testing.expectEqualStrings("generated", actual);
}
