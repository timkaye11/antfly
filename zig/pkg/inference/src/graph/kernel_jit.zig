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

//! Shared runtime-kernel JIT contracts.
//!
//! Backend compilers own source compilation and executable handles. This
//! module owns only the stable configuration, cache identity, qualification,
//! and persistence rules used by both Metal and CUDA.

const std = @import("std");
const builtin = @import("builtin");

pub const schema = "antfly.kernel_jit.v1";
pub const codegen_abi_version: u32 = 2;
pub const qualification_policy_version: u32 = 1;
/// Bump the policy version and update this identity whenever correctness
/// tolerances, fixtures, warmup/measurement procedure, or eligibility gates
/// change. It is part of every artifact key, so stale approvals cannot cross
/// a qualification-policy change.
pub const qualification_policy_identity =
    "correctness-first/v1;min-speedup=1.10;warmups=2;paired-repeats=5;worst-repeat-gate=true";
pub const minimum_speedup: f64 = 1.10;
pub const maximum_candidates: usize = 8;
pub const warmup_repeats: usize = 2;
pub const measurement_repeats: usize = 5;
pub const maximum_measurement_repeats: usize = 31;
pub const maximum_source_bytes: usize = 16 * 1024 * 1024;
pub const maximum_artifact_bytes: usize = 64 * 1024 * 1024;
pub const maximum_log_bytes: usize = 64 * 1024;

pub const Mode = enum {
    off,
    shadow,
    on,
    required,

    pub fn compiles(self: Mode) bool {
        return self != .off;
    }

    pub fn activates(self: Mode) bool {
        return self == .on or self == .required;
    }

    pub fn failClosed(self: Mode) bool {
        return self == .required;
    }
};

/// Live compiler and benchmark work is permitted only while the caller owns
/// an exclusive pre-serving startup phase. Dynamic loads may overlap requests
/// for other models and therefore must remain on bundled kernels.
pub const LoadContext = enum {
    dynamic,
    startup_preload,

    pub fn allowsQualification(self: LoadContext) bool {
        return self == .startup_preload;
    }
};

/// Required mode must never silently fall through to another backend after a
/// JIT contract failure. Other backend/model-loading failures retain the
/// normal fallback policy.
pub fn isRequiredFailure(mode: Mode, err: anyerror) bool {
    if (!mode.failClosed()) return false;
    return switch (err) {
        error.CudaJitRequiredRouteFailed,
        error.CudaJitRequiredRouteUnqualified,
        error.MetalJitRequiredRouteFailed,
        error.MetalKernelJitConfigConflict,
        error.KernelJitRequiredFailure,
        error.KernelJitRequiredBackendUnavailable,
        error.KernelJitRequiredDynamicLoad,
        error.KernelJitRequiredPreloadMissing,
        error.KernelJitRequiredPreloadUnmaterialized,
        error.KernelJitRequiredOptionalSessionUnmaterialized,
        error.KernelJitUnsupportedPlatform,
        error.KernelJitPersistentCacheUnsupported,
        error.InvalidKernelJitCacheDir,
        error.InvalidKernelJitPreloadBudget,
        error.InvalidKernelJitCacheBudget,
        => true,
        else => false,
    };
}

pub const Config = struct {
    mode: Mode = .off,
    cache_dir: ?[]const u8 = null,
    max_cache_bytes_mb: usize = 1024,
    /// Start budget for best-effort pre-publication work. One operation may
    /// overrun once started; required mode intentionally completes or fails.
    preload_budget_ms: u64 = 300_000,

    pub fn validate(self: Config) !void {
        if (self.mode.compiles() and !runtimeJitSupported()) {
            return error.KernelJitUnsupportedPlatform;
        }
        if (self.cache_dir) |path| {
            if (path.len == 0 or std.mem.indexOfScalar(u8, path, 0) != null) {
                return error.InvalidKernelJitCacheDir;
            }
            if (self.mode.compiles() and !persistentCacheSupported() and self.max_cache_bytes_mb != 0) {
                return error.KernelJitPersistentCacheUnsupported;
            }
        }
        if (self.preload_budget_ms < 1_000 or self.preload_budget_ms > 3_600_000) {
            return error.InvalidKernelJitPreloadBudget;
        }
        const max_cache_mb: usize = 1024 * 1024;
        if (self.max_cache_bytes_mb > max_cache_mb) return error.InvalidKernelJitCacheBudget;
    }

    pub fn maxCacheBytes(self: Config) !usize {
        try self.validate();
        return std.math.mul(usize, self.max_cache_bytes_mb, 1024 * 1024) catch
            error.InvalidKernelJitCacheBudget;
    }

    /// Returns an owned cache path, or null when persistence is disabled or
    /// no explicit path/HOME is available. Callers may still JIT in memory.
    pub fn resolveCacheDir(
        self: Config,
        allocator: std.mem.Allocator,
        home: ?[]const u8,
    ) !?[]u8 {
        if (try self.maxCacheBytes() == 0) return null;
        if (!persistentCacheSupported()) return null;
        if (self.cache_dir) |path| return try allocator.dupe(u8, path);
        const home_dir = home orelse return null;
        if (home_dir.len == 0 or std.mem.indexOfScalar(u8, home_dir, 0) != null) return null;
        return try std.fs.path.join(allocator, &.{ home_dir, ".antfly", "inference", "jit" });
    }
};

pub fn persistentCacheSupported() bool {
    return builtin.os.tag == .linux or builtin.os.tag == .macos;
}

pub fn runtimeJitSupported() bool {
    return builtin.os.tag == .linux or builtin.os.tag == .macos;
}

pub const Backend = enum(u8) {
    metal,
    cuda,
};

pub const RouteState = enum(u8) {
    missing,
    queued,
    compiling,
    validating,
    tuning,
    eligible,
    active,
    rejected,
    quarantined,
};

pub const CandidateEvidence = struct {
    candidate_index: usize,
    correctness_passed: bool,
    measured_speedup: f64,
    minimum_repeat_speedup: f64,
};

pub fn qualifies(evidence: CandidateEvidence) bool {
    return evidence.correctness_passed and
        std.math.isFinite(evidence.measured_speedup) and
        std.math.isFinite(evidence.minimum_repeat_speedup) and
        evidence.measured_speedup >= minimum_speedup and
        evidence.minimum_repeat_speedup >= minimum_speedup;
}

pub fn selectWinner(evidence: []const CandidateEvidence) ?CandidateEvidence {
    var winner: ?CandidateEvidence = null;
    for (evidence[0..@min(evidence.len, maximum_candidates)]) |candidate| {
        if (!qualifies(candidate)) continue;
        if (winner == null or candidate.measured_speedup > winner.?.measured_speedup) {
            winner = candidate;
        }
    }
    return winner;
}

pub fn evidenceFromPairedNanos(
    candidate_index: usize,
    correctness_passed: bool,
    baseline_nanos: []const u64,
    candidate_nanos: []const u64,
) !CandidateEvidence {
    if (baseline_nanos.len != candidate_nanos.len or
        baseline_nanos.len < measurement_repeats or
        baseline_nanos.len > maximum_measurement_repeats)
    {
        return error.InvalidKernelJitMeasurements;
    }
    var ratios: [maximum_measurement_repeats]f64 = undefined;
    var minimum = std.math.inf(f64);
    for (baseline_nanos, candidate_nanos, 0..) |baseline, candidate, index| {
        if (baseline == 0 or candidate == 0) return error.InvalidKernelJitMeasurements;
        const ratio = @as(f64, @floatFromInt(baseline)) / @as(f64, @floatFromInt(candidate));
        if (!std.math.isFinite(ratio) or ratio <= 0) return error.InvalidKernelJitMeasurements;
        ratios[index] = ratio;
        minimum = @min(minimum, ratio);
    }
    const measured = ratios[0..baseline_nanos.len];
    std.sort.heap(f64, measured, {}, std.sort.asc(f64));
    const middle = measured.len / 2;
    const median = if (measured.len % 2 == 1)
        measured[middle]
    else
        (measured[middle - 1] + measured[middle]) / 2.0;
    return .{
        .candidate_index = candidate_index,
        .correctness_passed = correctness_passed,
        .measured_speedup = median,
        .minimum_repeat_speedup = minimum,
    };
}

pub const QualificationRecord = struct {
    candidate_index: u32,
    repeat_count: u32,
    correctness_passed: bool,
    measured_speedup: f64,
    minimum_repeat_speedup: f64,
    max_absolute_error: f64,
    max_relative_error: f64,

    pub fn validate(self: QualificationRecord) !void {
        if (self.candidate_index >= maximum_candidates or
            self.repeat_count < measurement_repeats or
            self.repeat_count > maximum_measurement_repeats)
        {
            return error.InvalidKernelJitQualification;
        }
        if (!std.math.isFinite(self.measured_speedup) or self.measured_speedup <= 0 or
            !std.math.isFinite(self.minimum_repeat_speedup) or self.minimum_repeat_speedup <= 0 or
            !std.math.isFinite(self.max_absolute_error) or self.max_absolute_error < 0 or
            !std.math.isFinite(self.max_relative_error) or self.max_relative_error < 0)
        {
            return error.InvalidKernelJitQualification;
        }
    }

    pub fn evidence(self: QualificationRecord) !CandidateEvidence {
        try self.validate();
        return .{
            .candidate_index = self.candidate_index,
            .correctness_passed = self.correctness_passed,
            .measured_speedup = self.measured_speedup,
            .minimum_repeat_speedup = self.minimum_repeat_speedup,
        };
    }

    pub fn eligible(self: QualificationRecord) bool {
        return qualifies(self.evidence() catch return false);
    }

    pub fn encode(self: QualificationRecord) ![qualification_record_bytes]u8 {
        try self.validate();
        var out: [qualification_record_bytes]u8 = @splat(0);
        @memcpy(out[0..8], qualification_magic);
        std.mem.writeInt(u32, out[8..12], qualification_version, .little);
        std.mem.writeInt(u32, out[12..16], self.candidate_index, .little);
        std.mem.writeInt(u32, out[16..20], self.repeat_count, .little);
        out[20] = @intFromBool(self.correctness_passed);
        std.mem.writeInt(u64, out[24..32], @bitCast(self.measured_speedup), .little);
        std.mem.writeInt(u64, out[32..40], @bitCast(self.minimum_repeat_speedup), .little);
        std.mem.writeInt(u64, out[40..48], @bitCast(self.max_absolute_error), .little);
        std.mem.writeInt(u64, out[48..56], @bitCast(self.max_relative_error), .little);
        return out;
    }

    pub fn decode(bytes: []const u8) !QualificationRecord {
        if (bytes.len != qualification_record_bytes or
            !std.mem.eql(u8, bytes[0..8], qualification_magic) or
            std.mem.readInt(u32, bytes[8..12], .little) != qualification_version or
            bytes[20] > 1)
        {
            return error.InvalidKernelJitQualification;
        }
        const record = QualificationRecord{
            .candidate_index = std.mem.readInt(u32, bytes[12..16], .little),
            .repeat_count = std.mem.readInt(u32, bytes[16..20], .little),
            .correctness_passed = bytes[20] == 1,
            .measured_speedup = @bitCast(std.mem.readInt(u64, bytes[24..32], .little)),
            .minimum_repeat_speedup = @bitCast(std.mem.readInt(u64, bytes[32..40], .little)),
            .max_absolute_error = @bitCast(std.mem.readInt(u64, bytes[40..48], .little)),
            .max_relative_error = @bitCast(std.mem.readInt(u64, bytes[48..56], .little)),
        };
        try record.validate();
        return record;
    }
};

const qualification_magic = "AFJITQ01";
const qualification_version: u32 = 1;
const qualification_record_bytes: usize = 56;

pub const ArtifactKeyInput = struct {
    backend: Backend,
    semantic_identity: []const u8,
    runtime_identity: []const u8,
    target_identity: []const u8,
    device_identity: []const u8,
    schedule_identity: []const u8,
    baseline_identity: []const u8,
    compiler_identity: []const u8,
    compiler_options: []const u8,
    source: []const u8,
};

pub const ArtifactKey = [std.crypto.hash.sha2.Sha256.digest_length]u8;

fn hashBytes(hasher: *std.crypto.hash.sha2.Sha256, bytes: []const u8) void {
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, bytes.len, .little);
    hasher.update(&len);
    hasher.update(bytes);
}

pub fn artifactKey(input: ArtifactKeyInput) ArtifactKey {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hashBytes(&hasher, schema);
    var abi: [4]u8 = undefined;
    std.mem.writeInt(u32, &abi, codegen_abi_version, .little);
    hasher.update(&abi);
    hashBytes(&hasher, "qualification");
    var policy: [4]u8 = undefined;
    std.mem.writeInt(u32, &policy, qualification_policy_version, .little);
    hasher.update(&policy);
    hashBytes(&hasher, qualification_policy_identity);
    hasher.update(&.{@intFromEnum(input.backend)});
    hashBytes(&hasher, input.semantic_identity);
    hashBytes(&hasher, input.runtime_identity);
    hashBytes(&hasher, input.target_identity);
    hashBytes(&hasher, input.device_identity);
    hashBytes(&hasher, input.schedule_identity);
    hashBytes(&hasher, input.baseline_identity);
    hashBytes(&hasher, input.compiler_identity);
    hashBytes(&hasher, input.compiler_options);
    hashBytes(&hasher, input.source);
    var digest: ArtifactKey = undefined;
    hasher.final(&digest);
    return digest;
}

/// Cache identity for compiler output only. Qualification policy, fixtures,
/// device-specific evidence, and baseline implementation deliberately do not
/// participate so reusable CUDA PTX is not recompiled for every model shape
/// or qualification-policy edit.
pub fn compileArtifactKey(input: ArtifactKeyInput) ArtifactKey {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hashBytes(&hasher, schema);
    var abi: [4]u8 = undefined;
    std.mem.writeInt(u32, &abi, codegen_abi_version, .little);
    hasher.update(&abi);
    hashBytes(&hasher, "compile");
    hasher.update(&.{@intFromEnum(input.backend)});
    hashBytes(&hasher, input.semantic_identity);
    hashBytes(&hasher, input.runtime_identity);
    hashBytes(&hasher, input.target_identity);
    hashBytes(&hasher, input.schedule_identity);
    hashBytes(&hasher, input.compiler_identity);
    hashBytes(&hasher, input.compiler_options);
    hashBytes(&hasher, input.source);
    var digest: ArtifactKey = undefined;
    hasher.final(&digest);
    return digest;
}

pub fn artifactKeyHex(key: ArtifactKey) [std.crypto.hash.sha2.Sha256.digest_length * 2]u8 {
    return std.fmt.bytesToHex(key, .lower);
}

pub const CacheArtifactKind = enum(u8) {
    cuda_ptx,
    qualification,

    fn suffix(self: CacheArtifactKind) []const u8 {
        return switch (self) {
            .cuda_ptx => "ptx",
            .qualification => "qualification",
        };
    }
};

pub const CacheStats = struct {
    hits: usize = 0,
    misses: usize = 0,
    corruptions: usize = 0,
    stores: usize = 0,
    evictions: usize = 0,
    stale_temps_removed: usize = 0,
    temporary_bytes: u64 = 0,
};

const cache_magic = "AFJIT001";
const cache_header_bytes = 88;
const stale_temp_age_ns: i96 = 24 * std.time.ns_per_hour;
const cache_key_hex_bytes = std.crypto.hash.sha2.Sha256.digest_length * 2;

fn isCacheKeyHex(bytes: []const u8) bool {
    if (bytes.len != cache_key_hex_bytes) return false;
    for (bytes) |byte| switch (byte) {
        '0'...'9', 'a'...'f' => {},
        else => return false,
    };
    return true;
}

fn isCacheArtifactName(name: []const u8) bool {
    if (name.len <= cache_key_hex_bytes or !isCacheKeyHex(name[0..cache_key_hex_bytes])) return false;
    const suffix = name[cache_key_hex_bytes..];
    return std.mem.eql(u8, suffix, ".ptx.jit") or
        std.mem.eql(u8, suffix, ".qualification.jit");
}

fn isCacheTemporaryName(name: []const u8) bool {
    if (name.len < 2 or name[0] != '.') return false;
    const marker = ".tmp-";
    const marker_index = std.mem.lastIndexOf(u8, name, marker) orelse return false;
    return marker_index + marker.len < name.len and
        isCacheArtifactName(name[1..marker_index]);
}

fn privateCachePermissions(comptime directory: bool) std.Io.File.Permissions {
    if (std.Io.File.Permissions.has_executable_bit) {
        return .fromMode(if (directory) 0o700 else 0o600);
    }
    return if (directory) .default_dir else .default_file;
}

fn permissionsRejectForeignWriters(permissions: std.Io.File.Permissions) bool {
    if (!std.Io.File.Permissions.has_executable_bit) return false;
    return permissions.toMode() & 0o022 == 0;
}

/// Executable cache entries are a trust boundary, not ordinary disposable
/// data. Verify ownership through the already-open handle so a path swap
/// cannot race this check. Unsupported platforms fail closed and simply run
/// without a persistent cache.
fn handleOwnedByEffectiveUser(handle: std.Io.File.Handle) !bool {
    return switch (builtin.os.tag) {
        .linux => blk: {
            var stat: std.os.linux.Statx = std.mem.zeroes(std.os.linux.Statx);
            const result = std.os.linux.statx(
                handle,
                "",
                std.os.linux.AT.EMPTY_PATH | std.os.linux.AT.SYMLINK_NOFOLLOW,
                std.os.linux.STATX.BASIC_STATS,
                &stat,
            );
            if (std.os.linux.errno(result) != .SUCCESS) return error.KernelJitCacheStatFailed;
            break :blk stat.uid == std.os.linux.geteuid();
        },
        .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => blk: {
            var stat: std.c.Stat = undefined;
            if (std.c.fstat(handle, &stat) != 0) return error.KernelJitCacheStatFailed;
            break :blk stat.uid == std.c.geteuid();
        },
        else => error.KernelJitCacheOwnershipUnsupported,
    };
}

fn validateTrustedDirectory(dir: std.Io.Dir, io: std.Io) !void {
    const stat = try dir.stat(io);
    if (stat.kind != .directory or
        !permissionsRejectForeignWriters(stat.permissions) or
        !try handleOwnedByEffectiveUser(dir.handle))
    {
        return error.UntrustedKernelJitCacheDirectory;
    }
}

fn validateTrustedFile(file: std.Io.File, io: std.Io) !void {
    _ = try trustedFileStat(file, io);
}

fn trustedFileStat(file: std.Io.File, io: std.Io) !std.Io.File.Stat {
    const stat = try file.stat(io);
    if (stat.kind != .file or
        !permissionsRejectForeignWriters(stat.permissions) or
        !try handleOwnedByEffectiveUser(file.handle))
    {
        return error.UntrustedKernelJitCacheEntry;
    }
    return stat;
}

fn syncCacheDirectory(dir: std.Io.Dir, io: std.Io) !void {
    switch (builtin.os.tag) {
        .linux, .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos, .freebsd, .openbsd, .netbsd, .dragonfly => {
            const file: std.Io.File = .{
                .handle = dir.handle,
                .flags = .{ .nonblocking = false },
            };
            try file.sync(io);
        },
        else => {},
    }
}

/// Small content-addressed disk cache shared by backend JIT owners. Entries
/// are checksummed, written through a same-directory temporary file, fsynced,
/// and atomically renamed. A corrupt entry is deleted and treated as a miss;
/// generated code is never loaded from unchecked bytes.
pub const ArtifactCache = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    owns_dir: bool,
    max_bytes: usize,
    mutex: std.Io.Mutex = .init,
    stats: CacheStats = .{},
    temp_sequence: u64 = 0,

    pub fn initPath(
        allocator: std.mem.Allocator,
        io: std.Io,
        path: []const u8,
        max_bytes: usize,
    ) !ArtifactCache {
        if (path.len == 0 or std.mem.indexOfScalar(u8, path, 0) != null) {
            return error.InvalidKernelJitCacheDir;
        }
        const dir = try std.Io.Dir.cwd().createDirPathOpen(io, path, .{
            .permissions = privateCachePermissions(true),
            .open_options = .{
                .iterate = true,
                .follow_symlinks = false,
            },
        });
        return initOpenedDir(allocator, io, dir, true, max_bytes);
    }

    /// Test and embedding hook for a caller-owned directory handle.
    pub fn initDir(
        allocator: std.mem.Allocator,
        io: std.Io,
        dir: std.Io.Dir,
        max_bytes: usize,
    ) !ArtifactCache {
        return initOpenedDir(allocator, io, dir, false, max_bytes);
    }

    fn initOpenedDir(
        allocator: std.mem.Allocator,
        io: std.Io,
        dir: std.Io.Dir,
        owns_dir: bool,
        max_bytes: usize,
    ) !ArtifactCache {
        var cache = ArtifactCache{
            .allocator = allocator,
            .io = io,
            .dir = dir,
            .owns_dir = owns_dir,
            .max_bytes = max_bytes,
        };
        errdefer if (owns_dir) dir.close(io);
        try validateTrustedDirectory(dir, io);
        if (max_bytes != 0) try cache.pruneUnlocked();
        return cache;
    }

    pub fn deinit(self: *ArtifactCache) void {
        if (self.owns_dir) self.dir.close(self.io);
        self.* = undefined;
    }

    pub fn snapshotStats(self: *ArtifactCache) CacheStats {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.stats;
    }

    pub fn load(
        self: *ArtifactCache,
        kind: CacheArtifactKind,
        key: ArtifactKey,
    ) !?[]u8 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (self.max_bytes == 0) {
            self.stats.misses += 1;
            return null;
        }

        try validateTrustedDirectory(self.dir, self.io);

        const name = try cacheEntryName(self.allocator, kind, key);
        defer self.allocator.free(name);
        var file = self.dir.openFile(self.io, name, .{
            .allow_directory = false,
            .follow_symlinks = false,
            .resolve_beneath = true,
        }) catch |err| switch (err) {
            error.FileNotFound => {
                self.stats.misses += 1;
                return null;
            },
            error.SymLinkLoop => {
                self.stats.corruptions += 1;
                self.stats.misses += 1;
                return null;
            },
            else => return err,
        };
        defer file.close(self.io);
        try validateTrustedFile(file, self.io);
        var reader = file.reader(self.io, &.{});
        const encoded = reader.interface.allocRemaining(
            self.allocator,
            .limited(cache_header_bytes + maximum_artifact_bytes),
        ) catch |err| switch (err) {
            error.StreamTooLong => {
                self.dir.deleteFile(self.io, name) catch {};
                self.stats.corruptions += 1;
                self.stats.misses += 1;
                return null;
            },
            error.ReadFailed => return reader.err.?,
            error.OutOfMemory => return error.OutOfMemory,
        };
        defer self.allocator.free(encoded);

        const payload = validateCacheEntry(encoded, kind, key) catch {
            self.dir.deleteFile(self.io, name) catch {};
            self.stats.corruptions += 1;
            self.stats.misses += 1;
            return null;
        };
        const owned = try self.allocator.dupe(u8, payload);
        file.setTimestampsNow(self.io) catch {};
        self.stats.hits += 1;
        return owned;
    }

    pub fn store(
        self: *ArtifactCache,
        kind: CacheArtifactKind,
        key: ArtifactKey,
        payload: []const u8,
    ) !void {
        if (payload.len == 0 or payload.len > maximum_artifact_bytes) {
            return error.InvalidKernelJitArtifactSize;
        }
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (self.max_bytes == 0) return;
        try validateTrustedDirectory(self.dir, self.io);

        const name = try cacheEntryName(self.allocator, kind, key);
        defer self.allocator.free(name);
        const encoded = try encodeCacheEntry(self.allocator, kind, key, payload);
        defer self.allocator.free(encoded);

        var tmp_name: ?[]u8 = null;
        defer if (tmp_name) |value| self.allocator.free(value);
        var tmp_file: ?std.Io.File = null;
        var attempts: usize = 0;
        while (attempts < 16 and tmp_file == null) : (attempts += 1) {
            self.temp_sequence +%= 1;
            const candidate = try std.fmt.allocPrint(
                self.allocator,
                ".{s}.tmp-{x}-{x}-{x}",
                .{ name, @intFromPtr(self), std.Thread.getCurrentId(), self.temp_sequence },
            );
            const opened = self.dir.createFile(self.io, candidate, .{
                .truncate = false,
                .exclusive = true,
                .permissions = privateCachePermissions(false),
                .resolve_beneath = true,
            }) catch |err| switch (err) {
                error.PathAlreadyExists => {
                    self.allocator.free(candidate);
                    continue;
                },
                else => {
                    self.allocator.free(candidate);
                    return err;
                },
            };
            tmp_name = candidate;
            tmp_file = opened;
        }
        const owned_tmp_name = tmp_name orelse return error.KernelJitCacheTempCollision;
        var file = tmp_file.?;
        var file_open = true;
        defer if (file_open) file.close(self.io);
        errdefer self.dir.deleteFile(self.io, owned_tmp_name) catch {};
        try file.writeStreamingAll(self.io, encoded);
        try file.sync(self.io);
        file.close(self.io);
        file_open = false;
        try std.Io.Dir.rename(self.dir, owned_tmp_name, self.dir, name, self.io);
        try syncCacheDirectory(self.dir, self.io);
        self.stats.stores += 1;
        try self.pruneUnlocked();
    }

    fn pruneUnlocked(self: *ArtifactCache) !void {
        const Entry = struct {
            name: []u8,
            size: u64,
            mtime_ns: i128,

            fn lessThan(_: void, lhs: @This(), rhs: @This()) bool {
                if (lhs.mtime_ns != rhs.mtime_ns) return lhs.mtime_ns < rhs.mtime_ns;
                return std.mem.order(u8, lhs.name, rhs.name) == .lt;
            }
        };

        var entries: std.ArrayListUnmanaged(Entry) = .empty;
        defer {
            for (entries.items) |entry| self.allocator.free(entry.name);
            entries.deinit(self.allocator);
        }
        var total: u64 = 0;
        var temporary_bytes: u64 = 0;
        var directory_changed = false;
        const now = std.Io.Timestamp.now(self.io, .real);
        var iterator = self.dir.iterate();
        while (try iterator.next(self.io)) |entry| {
            const is_temporary = isCacheTemporaryName(entry.name);
            const is_artifact = isCacheArtifactName(entry.name);
            if (!is_temporary and !is_artifact) continue;
            const stat = try self.trustedRegularFileStat(entry.name) orelse continue;
            if (is_temporary) {
                // A recent temp file may belong to another cache instance or
                // process. Reclaim only our exact pattern after a full day.
                const age_ns = stat.mtime.durationTo(now).nanoseconds;
                if (age_ns >= stale_temp_age_ns) {
                    if (try self.deleteFileIfPresent(entry.name)) {
                        self.stats.stale_temps_removed += 1;
                        directory_changed = true;
                    }
                    continue;
                }
                temporary_bytes = std.math.add(u64, temporary_bytes, stat.size) catch std.math.maxInt(u64);
                total = std.math.add(u64, total, stat.size) catch std.math.maxInt(u64);
                continue;
            }
            try entries.append(self.allocator, .{
                .name = try self.allocator.dupe(u8, entry.name),
                .size = stat.size,
                .mtime_ns = stat.mtime.toNanoseconds(),
            });
            total = std.math.add(u64, total, stat.size) catch std.math.maxInt(u64);
        }
        self.stats.temporary_bytes = temporary_bytes;
        if (total <= self.max_bytes) {
            if (directory_changed) try syncCacheDirectory(self.dir, self.io);
            return;
        }
        std.sort.heap(Entry, entries.items, {}, Entry.lessThan);
        for (entries.items) |entry| {
            if (total <= self.max_bytes) break;
            const deleted = try self.deleteFileIfPresent(entry.name);
            total -|= entry.size;
            if (deleted) {
                self.stats.evictions += 1;
                directory_changed = true;
            }
        }
        if (directory_changed) try syncCacheDirectory(self.dir, self.io);
    }

    fn trustedRegularFileStat(self: *ArtifactCache, name: []const u8) !?std.Io.File.Stat {
        var file = self.dir.openFile(self.io, name, .{
            .allow_directory = false,
            .path_only = true,
            .follow_symlinks = false,
            .resolve_beneath = true,
        }) catch |err| switch (err) {
            error.FileNotFound, error.SymLinkLoop, error.IsDir => return null,
            else => return err,
        };
        defer file.close(self.io);
        return try trustedFileStat(file, self.io);
    }

    fn deleteFileIfPresent(self: *ArtifactCache, name: []const u8) !bool {
        self.dir.deleteFile(self.io, name) catch |err| switch (err) {
            error.FileNotFound => return false,
            else => return err,
        };
        return true;
    }
};

fn cacheEntryName(
    allocator: std.mem.Allocator,
    kind: CacheArtifactKind,
    key: ArtifactKey,
) ![]u8 {
    const hex = artifactKeyHex(key);
    return std.fmt.allocPrint(allocator, "{s}.{s}.jit", .{ &hex, kind.suffix() });
}

fn encodeCacheEntry(
    allocator: std.mem.Allocator,
    kind: CacheArtifactKind,
    key: ArtifactKey,
    payload: []const u8,
) ![]u8 {
    const total = std.math.add(usize, cache_header_bytes, payload.len) catch
        return error.InvalidKernelJitArtifactSize;
    const encoded = try allocator.alloc(u8, total);
    @memset(encoded[0..cache_header_bytes], 0);
    @memcpy(encoded[0..cache_magic.len], cache_magic);
    std.mem.writeInt(u32, encoded[8..12], codegen_abi_version, .little);
    encoded[12] = @intFromEnum(kind);
    std.mem.writeInt(u64, encoded[16..24], payload.len, .little);
    @memcpy(encoded[24..56], &key);
    var digest: ArtifactKey = undefined;
    std.crypto.hash.sha2.Sha256.hash(payload, &digest, .{});
    @memcpy(encoded[56..88], &digest);
    @memcpy(encoded[cache_header_bytes..], payload);
    return encoded;
}

fn validateCacheEntry(
    encoded: []const u8,
    expected_kind: CacheArtifactKind,
    expected_key: ArtifactKey,
) ![]const u8 {
    if (encoded.len < cache_header_bytes) return error.InvalidKernelJitCacheEntry;
    if (!std.mem.eql(u8, encoded[0..cache_magic.len], cache_magic)) return error.InvalidKernelJitCacheEntry;
    if (std.mem.readInt(u32, encoded[8..12], .little) != codegen_abi_version) return error.InvalidKernelJitCacheEntry;
    if (encoded[12] != @intFromEnum(expected_kind)) return error.InvalidKernelJitCacheEntry;
    const payload_len = std.mem.readInt(u64, encoded[16..24], .little);
    if (payload_len == 0 or payload_len > maximum_artifact_bytes) return error.InvalidKernelJitCacheEntry;
    if (payload_len != encoded.len - cache_header_bytes) return error.InvalidKernelJitCacheEntry;
    if (!std.mem.eql(u8, encoded[24..56], &expected_key)) return error.InvalidKernelJitCacheEntry;
    const payload = encoded[cache_header_bytes..];
    var digest: ArtifactKey = undefined;
    std.crypto.hash.sha2.Sha256.hash(payload, &digest, .{});
    if (!std.mem.eql(u8, encoded[56..88], &digest)) return error.InvalidKernelJitCacheEntry;
    return payload;
}

test "kernel JIT defaults are dormant" {
    const config = Config{};
    try config.validate();
    try std.testing.expect(!config.mode.compiles());
    try std.testing.expect(!config.mode.activates());
    try std.testing.expectEqual(@as(usize, 1024 * 1024 * 1024), try config.maxCacheBytes());
    try std.testing.expect(!LoadContext.dynamic.allowsQualification());
    try std.testing.expect(LoadContext.startup_preload.allowsQualification());
}

test "kernel JIT required failures cannot be hidden by backend fallback" {
    try std.testing.expect(!isRequiredFailure(.on, error.CudaJitRequiredRouteFailed));
    try std.testing.expect(isRequiredFailure(.required, error.CudaJitRequiredRouteFailed));
    try std.testing.expect(isRequiredFailure(.required, error.MetalJitRequiredRouteFailed));
    try std.testing.expect(isRequiredFailure(.required, error.KernelJitRequiredBackendUnavailable));
    try std.testing.expect(isRequiredFailure(.required, error.KernelJitRequiredDynamicLoad));
    try std.testing.expect(isRequiredFailure(.required, error.KernelJitRequiredPreloadMissing));
    try std.testing.expect(isRequiredFailure(.required, error.KernelJitRequiredPreloadUnmaterialized));
    try std.testing.expect(isRequiredFailure(.required, error.KernelJitRequiredOptionalSessionUnmaterialized));
    try std.testing.expect(isRequiredFailure(.required, error.KernelJitUnsupportedPlatform));
    try std.testing.expect(isRequiredFailure(.required, error.InvalidKernelJitPreloadBudget));
    try std.testing.expect(!isRequiredFailure(.required, error.InvalidModel));
}

test "kernel JIT config rejects unsafe values" {
    try std.testing.expectError(error.InvalidKernelJitCacheDir, (Config{ .cache_dir = "" }).validate());
    try std.testing.expectError(error.InvalidKernelJitPreloadBudget, (Config{ .preload_budget_ms = 999 }).validate());
    try std.testing.expectError(error.InvalidKernelJitPreloadBudget, (Config{ .preload_budget_ms = 3_600_001 }).validate());
    try std.testing.expectError(error.InvalidKernelJitCacheBudget, (Config{ .max_cache_bytes_mb = 1024 * 1024 + 1 }).validate());
}

test "kernel JIT cache directory resolution is owned explicit and default-off safe" {
    const explicit = (try (Config{ .cache_dir = "/var/cache/antfly-jit" }).resolveCacheDir(std.testing.allocator, "/ignored")).?;
    defer std.testing.allocator.free(explicit);
    try std.testing.expectEqualStrings("/var/cache/antfly-jit", explicit);

    const implicit = (try (Config{}).resolveCacheDir(std.testing.allocator, "/home/antfly")).?;
    defer std.testing.allocator.free(implicit);
    try std.testing.expectEqualStrings("/home/antfly/.antfly/inference/jit", implicit);
    try std.testing.expectEqual(@as(?[]u8, null), try (Config{}).resolveCacheDir(std.testing.allocator, null));
    try std.testing.expectEqual(@as(?[]u8, null), try (Config{ .max_cache_bytes_mb = 0 }).resolveCacheDir(std.testing.allocator, "/home/antfly"));
}

test "kernel JIT winner requires correctness and stable ten percent speedup" {
    const candidates = [_]CandidateEvidence{
        .{ .candidate_index = 0, .correctness_passed = true, .measured_speedup = 1.40, .minimum_repeat_speedup = 1.09 },
        .{ .candidate_index = 1, .correctness_passed = false, .measured_speedup = 2.0, .minimum_repeat_speedup = 2.0 },
        .{ .candidate_index = 2, .correctness_passed = true, .measured_speedup = 1.20, .minimum_repeat_speedup = 1.11 },
        .{ .candidate_index = 3, .correctness_passed = true, .measured_speedup = 1.30, .minimum_repeat_speedup = 1.10 },
    };
    const winner = selectWinner(&candidates) orelse return error.MissingKernelJitWinner;
    try std.testing.expectEqual(@as(usize, 3), winner.candidate_index);
}

test "kernel JIT evidence uses paired median and worst repeat" {
    const evidence = try evidenceFromPairedNanos(
        2,
        true,
        &.{ 120, 110, 140, 130, 150 },
        &.{ 100, 100, 100, 100, 100 },
    );
    try std.testing.expectApproxEqAbs(@as(f64, 1.3), evidence.measured_speedup, 0.000001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.1), evidence.minimum_repeat_speedup, 0.000001);
    try std.testing.expect(qualifies(evidence));
    try std.testing.expectError(
        error.InvalidKernelJitMeasurements,
        evidenceFromPairedNanos(0, true, &.{ 1, 2 }, &.{ 1, 2 }),
    );
    try std.testing.expectError(
        error.InvalidKernelJitMeasurements,
        evidenceFromPairedNanos(0, true, &.{ 1, 2, 3, 4, 5 }, &.{ 1, 2, 0, 4, 5 }),
    );
}

test "kernel JIT qualification records are stable validated cache payloads" {
    const record = QualificationRecord{
        .candidate_index = 3,
        .repeat_count = measurement_repeats,
        .correctness_passed = true,
        .measured_speedup = 1.24,
        .minimum_repeat_speedup = 1.12,
        .max_absolute_error = 0.0001,
        .max_relative_error = 0.001,
    };
    const encoded = try record.encode();
    const decoded = try QualificationRecord.decode(&encoded);
    try std.testing.expect(decoded.eligible());
    try std.testing.expectEqual(record.candidate_index, decoded.candidate_index);
    try std.testing.expectEqual(record.repeat_count, decoded.repeat_count);
    try std.testing.expectEqual(record.measured_speedup, decoded.measured_speedup);

    var corrupted = encoded;
    corrupted[8] +%= 1;
    try std.testing.expectError(error.InvalidKernelJitQualification, QualificationRecord.decode(&corrupted));
    var too_short = record;
    too_short.repeat_count = measurement_repeats - 1;
    try std.testing.expectError(error.InvalidKernelJitQualification, too_short.encode());
    var too_long = record;
    too_long.repeat_count = maximum_measurement_repeats + 1;
    try std.testing.expectError(error.InvalidKernelJitQualification, too_long.encode());
    var invalid_candidate = record;
    invalid_candidate.candidate_index = maximum_candidates;
    try std.testing.expectError(error.InvalidKernelJitQualification, invalid_candidate.encode());
}

test "kernel JIT artifact identity includes source target compiler and baseline" {
    const base = ArtifactKeyInput{
        .backend = .cuda,
        .semantic_identity = "q4_0/mmv/none",
        .runtime_identity = "rows=1,in=1536,out=6144",
        .target_identity = "sm_89",
        .device_identity = "gpu-0",
        .schedule_identity = "threads=256,cols=4",
        .baseline_identity = "termite_q4_0_tile4/v1",
        .compiler_identity = "nvrtc/13.2",
        .compiler_options = "--gpu-architecture=compute_89",
        .source = "extern \"C\" __global__ void k() {}",
    };
    const first = artifactKey(base);
    const repeated = artifactKey(base);
    try std.testing.expectEqualSlices(u8, &first, &repeated);

    var changed = base;
    changed.source = "extern \"C\" __global__ void k2() {}";
    const source_changed = artifactKey(changed);
    try std.testing.expect(!std.mem.eql(u8, &first, &source_changed));
    changed = base;
    changed.baseline_identity = "termite_q4_0_tile4/v2";
    const baseline_changed = artifactKey(changed);
    try std.testing.expect(!std.mem.eql(u8, &first, &baseline_changed));

    const compile_key = compileArtifactKey(base);
    changed = base;
    changed.baseline_identity = "termite_q4_0_tile4/v2";
    changed.device_identity = "gpu-1";
    const compile_evidence_changed = compileArtifactKey(changed);
    try std.testing.expectEqualSlices(u8, &compile_key, &compile_evidence_changed);
    changed.source = "extern \"C\" __global__ void k2() {}";
    const compile_source_changed = compileArtifactKey(changed);
    try std.testing.expect(!std.mem.eql(u8, &compile_key, &compile_source_changed));
    try std.testing.expect(!std.mem.eql(u8, &first, &compile_key));

    const hex = artifactKeyHex(first);
    try std.testing.expectEqual(@as(usize, 64), hex.len);
}

fn testArtifactKey(source: []const u8) ArtifactKey {
    return artifactKey(.{
        .backend = .cuda,
        .semantic_identity = "test",
        .runtime_identity = "rows=1",
        .target_identity = "sm_test",
        .device_identity = "device",
        .schedule_identity = "canonical",
        .baseline_identity = "baseline",
        .compiler_identity = "compiler",
        .compiler_options = "options",
        .source = source,
    });
}

test "kernel JIT cache round trips only checksummed content-addressed entries" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var cache = try ArtifactCache.initDir(std.testing.allocator, std.testing.io, tmp.dir, 1024 * 1024);
    defer cache.deinit();

    const key = testArtifactKey("kernel-a");
    try std.testing.expectEqual(@as(?[]u8, null), try cache.load(.cuda_ptx, key));
    try cache.store(.cuda_ptx, key, "ptx-data\x00");
    const loaded = (try cache.load(.cuda_ptx, key)) orelse return error.MissingKernelJitCacheEntry;
    defer std.testing.allocator.free(loaded);
    try std.testing.expectEqualStrings("ptx-data\x00", loaded);

    const stats = cache.snapshotStats();
    try std.testing.expectEqual(@as(usize, 1), stats.misses);
    try std.testing.expectEqual(@as(usize, 1), stats.stores);
    try std.testing.expectEqual(@as(usize, 1), stats.hits);
    try std.testing.expectEqual(@as(usize, 0), stats.corruptions);
}

test "kernel JIT cache rejects directories writable by another principal" {
    if (comptime !std.Io.File.Permissions.has_executable_bit or
        (builtin.os.tag != .linux and builtin.os.tag != .macos)) return;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.setPermissions(std.testing.io, .fromMode(0o777));
    defer tmp.dir.setPermissions(std.testing.io, .fromMode(0o700)) catch {};

    try std.testing.expectError(
        error.UntrustedKernelJitCacheDirectory,
        ArtifactCache.initDir(std.testing.allocator, std.testing.io, tmp.dir, 1024 * 1024),
    );
}

test "kernel JIT cache rejects and removes corrupted bytes" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var cache = try ArtifactCache.initDir(std.testing.allocator, std.testing.io, tmp.dir, 1024 * 1024);
    defer cache.deinit();

    const key = testArtifactKey("kernel-corrupt");
    try cache.store(.cuda_ptx, key, "valid");
    const name = try cacheEntryName(std.testing.allocator, .cuda_ptx, key);
    defer std.testing.allocator.free(name);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = name, .data = "not-a-cache-entry" });
    try std.testing.expectEqual(@as(?[]u8, null), try cache.load(.cuda_ptx, key));
    try std.testing.expectError(error.FileNotFound, tmp.dir.statFile(std.testing.io, name, .{}));
    const stats = cache.snapshotStats();
    try std.testing.expectEqual(@as(usize, 1), stats.corruptions);
}

test "kernel JIT cache rejects and removes oversized entries" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var cache = try ArtifactCache.initDir(std.testing.allocator, std.testing.io, tmp.dir, 1024 * 1024);
    defer cache.deinit();

    const key = testArtifactKey("kernel-oversized");
    const name = try cacheEntryName(std.testing.allocator, .cuda_ptx, key);
    defer std.testing.allocator.free(name);
    var file = try tmp.dir.createFile(std.testing.io, name, .{});
    {
        defer file.close(std.testing.io);
        var buffer: [1]u8 = undefined;
        var writer = file.writer(std.testing.io, &buffer);
        try writer.seekTo(cache_header_bytes + maximum_artifact_bytes);
        try writer.interface.writeAll(&.{0});
        try writer.end();
    }

    try std.testing.expectEqual(@as(?[]u8, null), try cache.load(.cuda_ptx, key));
    try std.testing.expectError(error.FileNotFound, tmp.dir.statFile(std.testing.io, name, .{}));
    try std.testing.expectEqual(@as(usize, 1), cache.snapshotStats().corruptions);
}

test "kernel JIT cache enforces its byte budget and zero disables persistence" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const one_entry_budget = cache_header_bytes + 8;
    var cache = try ArtifactCache.initDir(std.testing.allocator, std.testing.io, tmp.dir, one_entry_budget);
    defer cache.deinit();
    try cache.store(.cuda_ptx, testArtifactKey("one"), "12345678");
    try cache.store(.cuda_ptx, testArtifactKey("two"), "abcdefgh");
    try std.testing.expect(cache.snapshotStats().evictions >= 1);

    var disabled_tmp = std.testing.tmpDir(.{ .iterate = true });
    defer disabled_tmp.cleanup();
    var disabled = try ArtifactCache.initDir(std.testing.allocator, std.testing.io, disabled_tmp.dir, 0);
    defer disabled.deinit();
    const key = testArtifactKey("disabled");
    try disabled.store(.cuda_ptx, key, "ignored");
    try std.testing.expectEqual(@as(?[]u8, null), try disabled.load(.cuda_ptx, key));
    try std.testing.expectEqual(@as(usize, 0), disabled.snapshotStats().stores);
}

test "kernel JIT cache enforces a reduced byte budget when opened" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const first_key = testArtifactKey("preexisting-one");
    const second_key = testArtifactKey("preexisting-two");
    var writer = try ArtifactCache.initDir(std.testing.allocator, std.testing.io, tmp.dir, 1024 * 1024);
    try writer.store(.cuda_ptx, first_key, "12345678");
    try writer.store(.cuda_ptx, second_key, "abcdefgh");
    writer.deinit();

    var reopened = try ArtifactCache.initDir(
        std.testing.allocator,
        std.testing.io,
        tmp.dir,
        cache_header_bytes + 8,
    );
    defer reopened.deinit();
    try std.testing.expectEqual(@as(usize, 1), reopened.snapshotStats().evictions);
    const first = try reopened.load(.cuda_ptx, first_key);
    defer if (first) |value| std.testing.allocator.free(value);
    const second = try reopened.load(.cuda_ptx, second_key);
    defer if (second) |value| std.testing.allocator.free(value);
    try std.testing.expect((first == null) != (second == null));
}

test "kernel JIT cache cleanup ignores unrelated jit files" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "unrelated.jit",
        .data = "not an Antfly cache artifact",
    });
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = ".unrelated.jit.tmp-abandoned",
        .data = "not an Antfly cache temporary",
    });

    var cache = try ArtifactCache.initDir(std.testing.allocator, std.testing.io, tmp.dir, 1);
    defer cache.deinit();
    _ = try tmp.dir.statFile(std.testing.io, "unrelated.jit", .{});
    _ = try tmp.dir.statFile(std.testing.io, ".unrelated.jit.tmp-abandoned", .{});
    try std.testing.expectEqual(@as(usize, 0), cache.snapshotStats().evictions);
    try std.testing.expectEqual(@as(usize, 0), cache.snapshotStats().stale_temps_removed);
}

test "kernel JIT cache never removes another writer's temporary file" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const key = testArtifactKey("shared-key");
    const entry_name = try cacheEntryName(std.testing.allocator, .cuda_ptx, key);
    defer std.testing.allocator.free(entry_name);
    const foreign_temp = try std.fmt.allocPrint(std.testing.allocator, ".{s}.tmp-live-writer", .{entry_name});
    defer std.testing.allocator.free(foreign_temp);
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = foreign_temp,
        .data = "in-progress",
    });
    var first = try ArtifactCache.initDir(std.testing.allocator, std.testing.io, tmp.dir, 1024 * 1024);
    defer first.deinit();
    var second = try ArtifactCache.initDir(std.testing.allocator, std.testing.io, tmp.dir, 1024 * 1024);
    defer second.deinit();
    try first.store(.cuda_ptx, key, "first\x00");
    try second.store(.cuda_ptx, key, "second\x00");
    const loaded = (try first.load(.cuda_ptx, key)) orelse return error.MissingKernelJitCacheEntry;
    defer std.testing.allocator.free(loaded);
    try std.testing.expectEqualStrings("second\x00", loaded);
    _ = try tmp.dir.statFile(std.testing.io, foreign_temp, .{});
}

test "kernel JIT cache reclaims stale temporary files when opened" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const entry_name = try cacheEntryName(std.testing.allocator, .cuda_ptx, testArtifactKey("stale"));
    defer std.testing.allocator.free(entry_name);
    const stale_name = try std.fmt.allocPrint(std.testing.allocator, ".{s}.tmp-abandoned", .{entry_name});
    defer std.testing.allocator.free(stale_name);
    var stale = try tmp.dir.createFile(std.testing.io, stale_name, .{});
    try stale.writeStreamingAll(std.testing.io, "stale");
    const old = std.Io.Timestamp.now(std.testing.io, .real).subDuration(.{
        .nanoseconds = stale_temp_age_ns + std.time.ns_per_hour,
    });
    try stale.setTimestamps(std.testing.io, .{
        .access_timestamp = .{ .new = old },
        .modify_timestamp = .{ .new = old },
    });
    stale.close(std.testing.io);

    var cache = try ArtifactCache.initDir(std.testing.allocator, std.testing.io, tmp.dir, 1024 * 1024);
    defer cache.deinit();

    try std.testing.expectError(error.FileNotFound, tmp.dir.statFile(std.testing.io, stale_name, .{}));
    try std.testing.expectEqual(@as(usize, 1), cache.snapshotStats().stale_temps_removed);
}
