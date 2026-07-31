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
const builtin = @import("builtin");
const build_options = @import("build_options");
const platform = @import("antfly_platform");
const runtime = @import("../runtime/root.zig");
const backend_contracts = @import("../graph/backend_contracts.zig");
const c_file = @import("../util/c_file.zig");
const tensor_store_mod = @import("../models/tensor_store.zig");
const weight_source_mod = @import("../models/weight_source.zig");
const safetensors_mod = @import("../models/safetensors.zig");
const native_linalg = @import("../backends/native.zig");
const tier_planner = runtime.tier.planner;
const tier_cache_mod = runtime.tier.cache;
const prefetch_mod = runtime.tier.prefetch;
const tier_shared_mod = runtime.tier.shared;
const moe_residency = runtime.moe.residency;
const supports_native_metal_provider = build_options.enable_metal;
const metal_native_provider_mod = if (supports_native_metal_provider) @import("../backends/metal_native_provider.zig") else struct {
    pub const MetalNativeProvider = void;
};

const QuantizedStorage = weight_source_mod.QuantizedStorage;
const LoadedWeight = weight_source_mod.LoadedWeight;
const Tensor = @import("../backends/tensor.zig").Tensor;
const ExpertCoord = moe_residency.ExpertCoord;
const ResidencyTier = tier_planner.ResidencyTier;
const PlacementPlan = tier_planner.PlacementPlan;

pub const QuantExecutionMode = enum {
    prefer_backend_dense,
    wrapper_direct_quant,
    device_native,
};

pub const PackedExpertViewEntry = struct {
    bytes: []const u8,
    owned_copy: ?[]const u8 = null,
    last_access_epoch: u64 = 0,
    pin_count: usize = 0,

    pub fn deinit(self: *PackedExpertViewEntry) void {
        if (self.owned_copy) |bytes| std.heap.c_allocator.free(@constCast(bytes));
    }
};

pub const LazyWeightEntry = struct {
    tensor_ref: tensor_store_mod.LazyTensorRef,
    quantized_storage: ?QuantizedStorage = null,
    host_loaded: ?@import("../models/weight_source.zig").LoadedWeight = null,
    expert_coord: ?ExpertCoord = null,
    projection_mask: u8 = 0,
    loaded_bytes: usize = 0,
    backend_loaded_bytes: usize = 0,
    pin_count: usize = 0,
    pending_prefetch: bool = false,
    prefetch_score: u64 = 0,
    guard: ?*std.atomic.Mutex = null,
    placement: PlacementPlan = .{
        .class = .other,
        .preferred_tier = .host,
        .spill_tier = .disk,
    },
    prefer_dense: bool = false,
    active_tier: ResidencyTier = .disk,
    last_access_epoch: u64 = 0,
};

pub const PrefetchQueue = prefetch_mod.Queue(*LazyWeightEntry);

/// The source a streaming expert is read from. Today this is always the
/// GGUF-backed TensorStore, whose `describeQuantizedTensorRange` /
/// `loadQuantizedExpertInto` vtable entries perform exactly one fused
/// gate/up pread plus one down pread per expert. A verified contiguous
/// sidecar would slot in behind the same interface if attribution ever
/// proves the two-read layout blocks the performance gate.
pub const ExpertSource = tensor_store_mod.TensorStore;

pub const compact_runtime_layer_capacity = 30;
pub const compact_runtime_slot_capacity = 16;
pub const CompactSlotRef = runtime.moe.streaming_cache.SlotRef;
pub const CompactExpertCache = runtime.moe.streaming_cache.StreamingExpertCache(
    compact_runtime_layer_capacity,
    compact_runtime_slot_capacity,
);

/// Backing memory for one resident-expert slot: a page-aligned anonymous
/// arena published to the device as a no-copy buffer, plus the quantized
/// layout views into it. The slot's lifecycle state lives in the
/// backend-neutral `CompactExpertCache`, never here.
pub const CompactExpertArena = struct {
    arena: ?c_file.MmapRegion = null,
    layout: ?tensor_store_mod.QuantizedExpertLayout = null,
    /// Physical pages returned to the OS while the mapping (and its device
    /// buffer) stays valid; must be recommitted before the next fill.
    decommitted: bool = false,
};

/// Session-scoped compact streaming state shared by every compute wrapper:
/// slot bookkeeping, the residency ledger, and the arenas that back device
/// publication. Living on the WeightStore keeps cache contents warm across
/// wrapper recreation and, later, across server requests.
pub const CompactRuntimeState = struct {
    cache: CompactExpertCache = .{},
    ledger: runtime.moe.budget_ledger.ResidentBudgetLedger,
    arenas: [compact_runtime_layer_capacity * compact_runtime_slot_capacity]CompactExpertArena =
        [_]CompactExpertArena{.{}} ** (compact_runtime_layer_capacity * compact_runtime_slot_capacity),
    /// True under a compact memory profile: budget pressure rejects work.
    /// Opportunistic sessions keep the same machinery in report-only mode.
    enforcing: bool = false,
    evictions: u64 = 0,
    decommitted_bytes: u64 = 0,

    pub fn arenaAt(self: *CompactRuntimeState, layer: usize, slot: usize) *CompactExpertArena {
        return &self.arenas[layer * compact_runtime_slot_capacity + slot];
    }

    pub fn deinitArenas(self: *CompactRuntimeState) void {
        for (&self.arenas) |*slot| {
            if (slot.layout) |*layout| layout.deinit();
            slot.layout = null;
            if (slot.arena) |*arena| arena.deinit();
            slot.arena = null;
        }
    }
};

pub const JinaLoraAdapter = struct {
    allocator: std.mem.Allocator,
    reader: safetensors_mod.MMapReader,
    scale: f32,

    pub fn create(allocator: std.mem.Allocator, adapter_weights_path: []const u8, scale: f32) !*JinaLoraAdapter {
        const self = try allocator.create(JinaLoraAdapter);
        errdefer allocator.destroy(self);
        self.* = .{
            .allocator = allocator,
            .reader = try safetensors_mod.MMapReader.openFileAbsolute(allocator, adapter_weights_path),
            .scale = scale,
        };
        return self;
    }

    pub fn destroy(self: *JinaLoraAdapter) void {
        const allocator = self.allocator;
        self.reader.deinit();
        allocator.destroy(self);
    }

    pub fn mergeIntoLoadedWeight(self: *const JinaLoraAdapter, base_name: []const u8, loaded: *LoadedWeight) !void {
        if (!std.mem.endsWith(u8, base_name, ".weight")) return;

        const adapter_a_name = try jinaAdapterANameForBaseWeight(self.allocator, base_name);
        defer self.allocator.free(adapter_a_name);
        if (!self.reader.header.tensors.contains(adapter_a_name)) return;

        const adapter_b_name = try jinaAdapterBNameForAdapterA(self.allocator, adapter_a_name);
        defer self.allocator.free(adapter_b_name);
        if (!self.reader.header.tensors.contains(adapter_b_name)) return error.IncompleteJinaV5Adapter;

        try ensureLoadedWeightIsOwnedF32(self.allocator, loaded);

        var adapter_a = try readSafetensorAsF32(self.allocator, &self.reader, adapter_a_name);
        defer adapter_a.deinit();
        var adapter_b = try readSafetensorAsF32(self.allocator, &self.reader, adapter_b_name);
        defer adapter_b.deinit();

        try mergeLoraPairIntoWeight(loaded, adapter_a.tensor, adapter_b.tensor, self.scale);
    }
};

fn jinaAdapterANameForBaseWeight(allocator: std.mem.Allocator, base_name: []const u8) ![]const u8 {
    const suffix = ".weight";
    if (!std.mem.endsWith(u8, base_name, suffix)) return error.InvalidAdapterTensorName;
    return try std.fmt.allocPrint(allocator, "base_model.model.{s}.lora_A.weight", .{base_name[0 .. base_name.len - suffix.len]});
}

fn jinaAdapterBNameForAdapterA(allocator: std.mem.Allocator, adapter_a_name: []const u8) ![]const u8 {
    const suffix = ".lora_A.weight";
    if (!std.mem.endsWith(u8, adapter_a_name, suffix)) return error.InvalidAdapterTensorName;
    return try std.fmt.allocPrint(allocator, "{s}.lora_B.weight", .{adapter_a_name[0 .. adapter_a_name.len - suffix.len]});
}

fn readSafetensorAsF32(
    allocator: std.mem.Allocator,
    reader: *const safetensors_mod.MMapReader,
    name: []const u8,
) !LoadedWeight {
    var tensor = try reader.readTensor(name);
    if (tensor.dtype == .f16 or tensor.dtype == .bf16) {
        const converted = try weight_source_mod.convertToF32(allocator, &tensor);
        tensor.deinit();
        tensor = converted;
    } else if (tensor.dtype == .f32 and !tensor.owns_data) {
        const converted = try Tensor.initFloat32(allocator, tensor.name, tensor.shape, tensor.asFloat32());
        tensor.deinit();
        tensor = converted;
    }
    if (tensor.dtype != .f32) {
        tensor.deinit();
        return error.UnsupportedAdapterTensorType;
    }
    return .{ .tensor = tensor, .quantized = false };
}

fn ensureLoadedWeightIsOwnedF32(allocator: std.mem.Allocator, loaded: *LoadedWeight) !void {
    if (loaded.quantized_storage) |*storage| {
        storage.deinit();
        loaded.quantized_storage = null;
        loaded.quantized = false;
    }

    if (loaded.tensor.dtype == .f32 and loaded.tensor.owns_data) return;

    const converted = switch (loaded.tensor.dtype) {
        .f16, .bf16 => try weight_source_mod.convertToF32(allocator, &loaded.tensor),
        .f32 => try Tensor.initFloat32(allocator, loaded.tensor.name, loaded.tensor.shape, loaded.tensor.asFloat32()),
        else => return error.UnsupportedJinaV5AdapterBaseWeight,
    };
    loaded.tensor.deinit();
    loaded.tensor = converted;
}

fn mergeLoraPairIntoWeight(base_weight: *LoadedWeight, adapter_a: Tensor, adapter_b: Tensor, scale: f32) !void {
    if (base_weight.quantized or base_weight.tensor.dtype != .f32) return error.UnsupportedJinaV5AdapterBaseWeight;
    if (base_weight.tensor.shape.len != 2 or adapter_a.shape.len != 2 or adapter_b.shape.len != 2) return error.InvalidAdapterTensorShape;

    const out_dim: usize = @intCast(base_weight.tensor.shape[0]);
    const in_dim: usize = @intCast(base_weight.tensor.shape[1]);
    const rank: usize = @intCast(adapter_a.shape[0]);
    if (rank == 0) return error.InvalidAdapterTensorShape;
    if (@as(usize, @intCast(adapter_a.shape[1])) != in_dim) return error.AdapterInputDimMismatch;
    if (@as(usize, @intCast(adapter_b.shape[0])) != out_dim) return error.AdapterOutputDimMismatch;
    if (@as(usize, @intCast(adapter_b.shape[1])) != rank) return error.AdapterRankMismatch;

    native_linalg.sgemmSync(
        out_dim,
        in_dim,
        rank,
        scale,
        adapter_b.asFloat32(),
        adapter_a.asFloat32(),
        1.0,
        base_weight.tensor.asFloat32Mut(),
    );
}

pub const WeightStore = struct {
    allocator: std.mem.Allocator,
    resident_weight_estimate_bytes: usize = 0,
    prefix: []const u8,
    lazy_weights: std.StringHashMapUnmanaged(LazyWeightEntry),
    prefetch: PrefetchQueue = undefined,
    prefetch_initialized: bool = false,
    tensor_store: ?tensor_store_mod.TensorStore = null,
    moe_num_experts: usize = 0,
    residency: ?moe_residency.SharedResidency = null,
    tier_cache: ?tier_cache_mod.SharedCache = null,
    shared_prefetch: ?*tier_shared_mod.SharedPrefetchState = null,
    allow_direct_quant: bool = true,
    quant_execution_mode: QuantExecutionMode = .prefer_backend_dense,
    prefer_f32_dense_tensors: bool = false,
    mirror_kv_to_manager: bool = true,
    access_epoch: u64 = 1,
    packed_expert_views: std.StringHashMapUnmanaged(PackedExpertViewEntry) = .empty,
    packed_expert_view_bytes: usize = 0,
    parallel_quant_loader: ParallelQuantLoader = .{},
    parallel_expert_loader: ParallelExpertLoader = .{},
    shared_metal_native_provider: if (supports_native_metal_provider) ?*metal_native_provider_mod.MetalNativeProvider else void =
        if (supports_native_metal_provider) null else {},
    shared_metal_native_provider_lock: if (supports_native_metal_provider) std.Io.Mutex else void =
        if (supports_native_metal_provider) .init else {},
    jina_lora_adapter: ?*JinaLoraAdapter = null,
    /// Immutable compact-profile contract for this model instance. Set once
    /// during session creation, before any compute wrapper exists. Presence
    /// makes the compact streaming route mandatory for routed experts.
    compact: ?backend_contracts.CompactInferenceConfig = null,
    /// Shared compact streaming residency (slot cache, ledger, arenas).
    /// Owned by the session; created when the model matches the qualified
    /// streaming geometry, enforcing only under a compact contract.
    compact_runtime: ?*CompactRuntimeState = null,
};

pub fn touchLazyWeight(data: *WeightStore, entry: *LazyWeightEntry) void {
    entry.last_access_epoch = data.access_epoch;
    data.access_epoch +|= 1;
}

/// Simple prefetch callback suitable for backends that don't do their own
/// Simple prefetch callback for Metal-hosted weights. Runs the synchronous host
/// load and resets the pending flag.
pub fn simplePrefetchProcess(ctx: *anyopaque, entry: *LazyWeightEntry) void {
    const data: *WeightStore = @ptrCast(@alignCast(ctx));
    entry.pending_prefetch = false;
    ensureHostLazyWeightLoadedSimple(data, entry) catch {};
}

pub fn simplePrefetchPriority(entry: *LazyWeightEntry) u64 {
    return entry.prefetch_score;
}

/// Install a prefetch queue on the store using caller-supplied callbacks.
pub fn installPrefetchQueue(
    data: *WeightStore,
    allocator: std.mem.Allocator,
    process_fn: *const fn (ctx: *anyopaque, entry: *LazyWeightEntry) void,
    priority_fn: *const fn (entry: *LazyWeightEntry) u64,
) void {
    if (data.prefetch_initialized) return;
    data.prefetch = PrefetchQueue.initWithPriority(allocator, data, process_fn, priority_fn);
    data.prefetch_initialized = true;
    var it = data.lazy_weights.iterator();
    while (it.next()) |entry| {
        entry.value_ptr.guard = data.prefetch.mutexPtr();
    }
}

pub fn startPrefetchWorker(data: *WeightStore) !void {
    try data.prefetch.start();
}

pub fn stopPrefetchWorker(data: *WeightStore) void {
    data.prefetch.stop();
}

pub fn deinitPrefetchQueue(data: *WeightStore) void {
    data.parallel_quant_loader.deinit();
    data.parallel_expert_loader.deinit();
    if (!data.prefetch_initialized) return;
    data.prefetch.deinit();
    data.prefetch_initialized = false;
}

pub fn deinitPackedExpertViews(data: *WeightStore, allocator: std.mem.Allocator) void {
    var it = data.packed_expert_views.iterator();
    while (it.next()) |entry| {
        var view = entry.value_ptr.*;
        view.deinit();
        allocator.free(entry.key_ptr.*);
    }
    data.packed_expert_views.deinit(allocator);
    data.packed_expert_view_bytes = 0;
}

pub fn ensureHostLazyWeightLoadedSimple(data: *WeightStore, entry: *LazyWeightEntry) !void {
    if (entry.expert_coord) |coord| {
        if (data.residency) |*residency| {
            try residency.noteTouch(coord, data.moe_num_experts);
        }
    }
    if (entry.host_loaded != null) return;
    if (entry.quantized_storage != null and !entry.prefer_dense) return;

    const tensor_store = data.tensor_store orelse return error.MissingWeight;
    if (data.allow_direct_quant and !entry.prefer_dense) {
        if (try tensor_store.loadQuantizedStorageRef(&entry.tensor_ref)) |loaded_storage| {
            entry.loaded_bytes = loaded_storage.raw_bytes.len + loaded_storage.prepared.ownedBytes();
            entry.quantized_storage = loaded_storage;
            entry.active_tier = .host;
            if (entry.loaded_bytes != 0) {
                if (data.tier_cache) |*tier_cache| tier_cache.noteResident(.host, entry.loaded_bytes);
            }
            try noteExpertLoadedAndEvict(data, entry);
            return;
        }
    }
    if (data.allow_direct_quant and entry.prefer_dense and entry.quantized_storage == null) {
        if (try tensor_store.loadQuantizedStorageRef(&entry.tensor_ref)) |loaded_storage| {
            entry.quantized_storage = loaded_storage;
        }
    }

    entry.host_loaded = try tensor_store.loadTensorRef(&entry.tensor_ref);
    if (entry.host_loaded.?.quantized_storage) |*storage| {
        storage.deinit();
        entry.host_loaded.?.quantized_storage = null;
        entry.host_loaded.?.quantized = false;
    }
    if (data.jina_lora_adapter) |adapter| {
        try adapter.mergeIntoLoadedWeight(entry.tensor_ref.name, &entry.host_loaded.?);
    }
    entry.loaded_bytes = entry.host_loaded.?.tensor.data.len;
    entry.active_tier = if (entry.loaded_bytes == 0) .disk else .host;
    if (entry.loaded_bytes != 0) {
        if (data.tier_cache) |*tier_cache| tier_cache.noteResident(.host, entry.loaded_bytes);
    }
    try noteExpertLoadedAndEvict(data, entry);
}

const ParallelQuantLoadTask = struct {
    tensor_store: tensor_store_mod.TensorStore,
    entry: *LazyWeightEntry,
    storage: ?QuantizedStorage = null,
    load_error: ?anyerror = null,
};

fn parallelQuantLoadMain(task: *ParallelQuantLoadTask) void {
    task.storage = task.tensor_store.loadQuantizedStorageRef(&task.entry.tensor_ref) catch |err| {
        task.load_error = err;
        return;
    };
}

/// Bounded condition-variable worker pool for background weight reads.
///
/// One caller-owned batch at a time: `begin` publishes the task slice and
/// wakes the workers; a second `begin` before the batch drains is an error,
/// which bounds queued work to exactly one route's misses. Workers park on
/// the work condition instead of spinning, and the batch waiter parks on a
/// completion condition. `cancel` declines every not-yet-started task
/// (surfaced as error.ExpertLoadCanceled); tasks already inside a pread
/// finish normally. `deinit` is shutdown: stop, broadcast, join. Task
/// errors — including short reads surfaced as error.IncompleteRead — stay
/// on the task until the caller harvests results.
fn LoadWorkerPool(comptime Task: type, comptime runTask: fn (*Task) void) type {
    return struct {
        const Self = @This();
        const max_worker_count = 8;

        mutex: std.Io.Mutex = .init,
        work_available: std.Io.Condition = .init,
        batch_complete: std.Io.Condition = .init,
        workers: [max_worker_count]?std.Thread = [_]?std.Thread{null} ** max_worker_count,
        worker_count: usize = 0,
        tasks: ?[]Task = null,
        next_task: usize = 0,
        completed_tasks: usize = 0,
        canceled: bool = false,
        stop_workers: bool = false,

        /// Futex-capable Io for parking plain worker threads; mirrors the
        /// metalComputeLockIo convention used elsewhere in this backend.
        fn syncIo() std.Io {
            return if (builtin.is_test) std.testing.io else std.Options.debug_io;
        }

        fn ensureStarted(self: *Self, requested_workers: usize) !void {
            const io = syncIo();
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            if (self.worker_count != 0) return;
            const count = @min(@as(usize, max_worker_count), @max(@as(usize, 1), requested_workers));
            self.stop_workers = false;
            for (0..count) |index| {
                self.workers[index] = try std.Thread.spawn(.{}, workerMain, .{self});
                self.worker_count += 1;
            }
        }

        fn takeTaskLocked(self: *Self) ?*Task {
            const tasks = self.tasks orelse return null;
            if (self.next_task >= tasks.len) return null;
            const task = &tasks[self.next_task];
            self.next_task += 1;
            return task;
        }

        fn workerMain(self: *Self) void {
            const io = syncIo();
            self.mutex.lockUncancelable(io);
            while (true) {
                if (self.stop_workers) break;
                if (self.takeTaskLocked()) |task| {
                    const declined = self.canceled;
                    self.mutex.unlock(io);
                    if (declined) task.load_error = error.ExpertLoadCanceled else runTask(task);
                    self.mutex.lockUncancelable(io);
                    self.completed_tasks += 1;
                    if (self.tasks) |tasks| {
                        if (self.completed_tasks == tasks.len) self.batch_complete.broadcast(io);
                    }
                } else {
                    self.work_available.waitUncancelable(io, &self.mutex);
                }
            }
            self.mutex.unlock(io);
        }

        fn begin(self: *Self, tasks: []Task, requested_workers: usize) !void {
            if (tasks.len == 0) return;
            try self.ensureStarted(requested_workers);
            const io = syncIo();
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            if (self.tasks != null) return error.ExpertLoaderBusy;
            self.tasks = tasks;
            self.next_task = 0;
            self.completed_tasks = 0;
            self.canceled = false;
            self.work_available.broadcast(io);
        }

        fn wait(self: *Self, task_count: usize) void {
            if (task_count == 0) return;
            const io = syncIo();
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            if (self.tasks == null) return;
            while (self.completed_tasks != task_count) self.batch_complete.waitUncancelable(io, &self.mutex);
            self.tasks = null;
        }

        /// Decline every not-yet-started task in the active batch. The
        /// caller still drains through `wait` (or a batch deinit) so worker
        /// threads never outlive the task slice.
        fn cancel(self: *Self) void {
            const io = syncIo();
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            self.canceled = true;
        }

        fn run(self: *Self, tasks: []Task, requested_workers: usize) !void {
            if (tasks.len == 0) return;
            if (requested_workers <= 1) {
                for (tasks) |*task| runTask(task);
                return;
            }
            try self.begin(tasks, requested_workers);
            self.wait(tasks.len);
        }

        fn deinit(self: *Self) void {
            const io = syncIo();
            self.mutex.lockUncancelable(io);
            if (self.worker_count == 0) {
                self.mutex.unlock(io);
                self.* = .{};
                return;
            }
            self.stop_workers = true;
            self.canceled = true;
            self.work_available.broadcast(io);
            self.mutex.unlock(io);
            for (self.workers[0..self.worker_count]) |worker| if (worker) |handle| handle.join();
            self.* = .{};
        }
    };
}

const ParallelQuantLoader = LoadWorkerPool(ParallelQuantLoadTask, parallelQuantLoadMain);

pub const ResidentExpertLoadRequest = struct {
    entries: [3]*LazyWeightEntry,
    destination: []u8,
};

const ParallelExpertLoadTask = struct {
    tensor_store: tensor_store_mod.TensorStore,
    refs: [3]tensor_store_mod.LazyTensorRef,
    destination: []u8,
    layout: ?tensor_store_mod.QuantizedExpertLayout = null,
    load_error: ?anyerror = null,
};

fn parallelExpertLoadMain(task: *ParallelExpertLoadTask) void {
    task.layout = task.tensor_store.loadQuantizedExpertInto(
        std.heap.c_allocator,
        &task.refs,
        task.destination,
    ) catch |err| {
        task.load_error = err;
        return;
    };
}

const ParallelExpertLoader = LoadWorkerPool(ParallelExpertLoadTask, parallelExpertLoadMain);

pub const ResidentExpertLoadBatch = struct {
    data: *WeightStore,
    tasks: []ParallelExpertLoadTask,
    active: bool = true,

    pub fn finish(
        self: *ResidentExpertLoadBatch,
        outputs: []?tensor_store_mod.QuantizedExpertLayout,
    ) !void {
        if (!self.active or outputs.len != self.tasks.len) return error.InvalidTensorShape;
        self.data.parallel_expert_loader.wait(self.tasks.len);
        self.active = false;
        defer {
            self.data.allocator.free(self.tasks);
            self.tasks = &.{};
        }
        errdefer for (self.tasks) |*task| if (task.layout) |*layout| layout.deinit();

        for (self.tasks, 0..) |*task, index| {
            if (task.load_error) |err| return err;
            outputs[index] = task.layout orelse return error.UnsupportedTensorType;
            task.layout = null;
        }
    }

    pub fn deinit(self: *ResidentExpertLoadBatch) void {
        if (self.tasks.len == 0) return;
        if (self.active) self.data.parallel_expert_loader.wait(self.tasks.len);
        for (self.tasks) |*task| if (task.layout) |*layout| layout.deinit();
        self.data.allocator.free(self.tasks);
        self.tasks = &.{};
        self.active = false;
    }

    /// Cancel not-yet-started reads and drain the batch. Preads already in
    /// flight complete first; every produced layout is dropped. Used by
    /// request cancellation so no loader thread ever outlives the arenas it
    /// writes into.
    pub fn cancel(self: *ResidentExpertLoadBatch) void {
        if (!self.active) return;
        self.data.parallel_expert_loader.cancel();
        self.deinit();
    }
};

/// Start whole-expert preads and return immediately. Only the caller-owned
/// destination arenas and immutable tensor-store references are touched on
/// loader threads; publishing layouts and preparing Metal slots stay on the
/// inference thread in `ResidentExpertLoadBatch.finish`.
pub fn beginResidentExpertLayoutsParallel(
    data: *WeightStore,
    requests: []const ResidentExpertLoadRequest,
    max_workers: usize,
) !ResidentExpertLoadBatch {
    const tensor_store = data.tensor_store orelse return error.MissingWeight;
    const tasks = try data.allocator.alloc(ParallelExpertLoadTask, requests.len);
    errdefer data.allocator.free(tasks);

    for (requests, 0..) |request, index| {
        for (request.entries) |entry| touchLazyWeight(data, entry);
        tasks[index] = .{
            .tensor_store = tensor_store,
            .refs = .{
                request.entries[0].tensor_ref,
                request.entries[1].tensor_ref,
                request.entries[2].tensor_ref,
            },
            .destination = request.destination,
        };
    }
    try data.parallel_expert_loader.begin(tasks, @max(@as(usize, 1), max_workers));
    return .{ .data = data, .tasks = tasks };
}

/// Load whole experts concurrently. Each task performs one fused gate/up
/// pread and one down pread into a caller-owned, page-aligned Metal-visible
/// arena. No projection bytes are retained in the generic lazy cache.
pub fn loadResidentExpertLayoutsParallel(
    data: *WeightStore,
    requests: []const ResidentExpertLoadRequest,
    outputs: []?tensor_store_mod.QuantizedExpertLayout,
    max_workers: usize,
) !void {
    if (requests.len != outputs.len) return error.InvalidTensorShape;
    @memset(outputs, null);
    var batch = try beginResidentExpertLayoutsParallel(data, requests, max_workers);
    defer batch.deinit();
    try batch.finish(outputs);
}

/// Load independent quantized tensor ranges concurrently and publish them as
/// one protected route batch.
pub fn ensureHostLazyWeightsLoadedParallel(
    data: *WeightStore,
    entries: []const *LazyWeightEntry,
    max_workers: usize,
) !void {
    const tensor_store = data.tensor_store orelse return error.MissingWeight;
    const tasks = try data.allocator.alloc(ParallelQuantLoadTask, entries.len);
    defer data.allocator.free(tasks);

    var pending_count: usize = 0;
    data.prefetch.lock();
    for (entries) |entry| {
        touchLazyWeight(data, entry);
        if (entry.host_loaded != null or (entry.quantized_storage != null and !entry.prefer_dense)) continue;
        entry.pending_prefetch = true;
        tasks[pending_count] = .{ .tensor_store = tensor_store, .entry = entry };
        pending_count += 1;
    }
    data.prefetch.unlock();
    const active_tasks = tasks[0..pending_count];
    errdefer {
        data.prefetch.lock();
        defer data.prefetch.unlock();
        for (active_tasks) |task| task.entry.pending_prefetch = false;
    }

    try data.parallel_quant_loader.run(active_tasks, @max(@as(usize, 1), max_workers));
    defer for (active_tasks) |*task| {
        if (task.storage) |*storage| storage.deinit();
    };

    data.prefetch.lock();
    defer data.prefetch.unlock();
    for (active_tasks) |*task| {
        if (task.load_error) |err| {
            for (active_tasks) |cleanup| cleanup.entry.pending_prefetch = false;
            return err;
        }
        const storage = task.storage orelse {
            for (active_tasks) |cleanup| cleanup.entry.pending_prefetch = false;
            return error.UnsupportedTensorType;
        };
        if (task.entry.host_loaded != null or (task.entry.quantized_storage != null and !task.entry.prefer_dense)) {
            var redundant = storage;
            redundant.deinit();
            task.storage = null;
            task.entry.pending_prefetch = false;
            continue;
        }
        task.storage = null;
        task.entry.quantized_storage = storage;
        task.entry.loaded_bytes = storage.raw_bytes.len + storage.prepared.ownedBytes();
        task.entry.active_tier = .host;
        task.entry.pending_prefetch = false;
        if (task.entry.loaded_bytes != 0) {
            if (data.tier_cache) |*tier_cache| tier_cache.noteResident(.host, task.entry.loaded_bytes);
        }
        if (task.entry.expert_coord) |coord| {
            if (data.residency) |*residency| {
                try residency.noteLoad(coord, data.moe_num_experts, task.entry.projection_mask, task.entry.loaded_bytes);
            }
        }
    }

    if (data.residency) |*residency| {
        for (entries) |entry| {
            const coord = entry.expert_coord orelse continue;
            while (residency.isOverCapacity(coord.layer_index)) {
                const victim = findSimpleExpertVictimAvoiding(data, coord.layer_index, entries) orelse break;
                unloadSimpleExpert(data, victim);
            }
        }
    }
}

fn noteExpertLoadedAndEvict(data: *WeightStore, entry: *LazyWeightEntry) !void {
    const coord = entry.expert_coord orelse return;
    if (data.residency) |*residency| {
        try residency.noteLoad(coord, data.moe_num_experts, entry.projection_mask, entry.loaded_bytes);
        while (residency.isOverCapacity(coord.layer_index)) {
            const protected = [_]*LazyWeightEntry{entry};
            const victim = findSimpleExpertVictimAvoiding(data, coord.layer_index, &protected) orelse break;
            unloadSimpleExpert(data, victim);
        }
    }
}

fn findSimpleExpertVictimAvoiding(
    data: *WeightStore,
    layer_index: usize,
    protected_entries: []const *LazyWeightEntry,
) ?ExpertCoord {
    const residency = if (data.residency) |*value| value else return null;
    var victim: ?ExpertCoord = null;
    var it = data.lazy_weights.iterator();
    while (it.next()) |map_entry| {
        const entry = map_entry.value_ptr;
        const coord = entry.expert_coord orelse continue;
        if (coord.layer_index != layer_index or expertIsProtected(coord, protected_entries)) continue;
        if (entry.quantized_storage == null and entry.host_loaded == null) continue;
        if (!simpleExpertCanEvict(data, coord)) continue;
        if (victim == null or residency.isMoreEvictable(coord, victim.?)) victim = coord;
    }
    return victim;
}

fn expertIsProtected(coord: ExpertCoord, protected_entries: []const *LazyWeightEntry) bool {
    for (protected_entries) |entry| {
        const protected = entry.expert_coord orelse continue;
        if (protected.layer_index == coord.layer_index and protected.expert_index == coord.expert_index) return true;
    }
    return false;
}

fn simpleExpertCanEvict(data: *WeightStore, coord: ExpertCoord) bool {
    var found = false;
    var it = data.lazy_weights.iterator();
    while (it.next()) |map_entry| {
        const entry = map_entry.value_ptr;
        const entry_coord = entry.expert_coord orelse continue;
        if (entry_coord.layer_index != coord.layer_index or entry_coord.expert_index != coord.expert_index) continue;
        if (entry.quantized_storage == null and entry.host_loaded == null) continue;
        found = true;
        if (entry.pin_count != 0) return false;
    }
    return found;
}

fn unloadSimpleExpert(data: *WeightStore, coord: ExpertCoord) void {
    var it = data.lazy_weights.iterator();
    while (it.next()) |map_entry| {
        const entry = map_entry.value_ptr;
        const entry_coord = entry.expert_coord orelse continue;
        if (entry_coord.layer_index != coord.layer_index or entry_coord.expert_index != coord.expert_index) continue;
        if (entry.pin_count != 0) continue;

        const released_bytes = entry.loaded_bytes;
        if (entry.quantized_storage) |*storage| storage.deinit();
        entry.quantized_storage = null;
        if (entry.host_loaded) |*loaded| loaded.deinit();
        entry.host_loaded = null;
        entry.loaded_bytes = 0;
        entry.active_tier = .disk;
        if (released_bytes != 0) {
            if (data.tier_cache) |*tier_cache| tier_cache.noteRelease(.host, released_bytes);
            if (data.residency) |*residency| residency.noteUnload(coord, entry.projection_mask, released_bytes);
        }
    }
}

const PoolTestTask = struct {
    completed: *std.atomic.Value(usize),
    gate: ?*std.atomic.Value(bool) = null,
    started: ?*std.atomic.Value(bool) = null,
    fail: bool = false,
    load_error: ?anyerror = null,
};

fn poolTestTaskMain(task: *PoolTestTask) void {
    if (task.started) |started| started.store(true, .release);
    if (task.gate) |gate| {
        while (!gate.load(.acquire)) platform.time.yieldBriefly();
    }
    if (task.fail) {
        task.load_error = error.IncompleteRead;
        return;
    }
    _ = task.completed.fetchAdd(1, .monotonic);
}

const PoolUnderTest = LoadWorkerPool(PoolTestTask, poolTestTaskMain);

test "load worker pool runs a batch and propagates task errors" {
    var pool = PoolUnderTest{};
    defer pool.deinit();
    var completed = std.atomic.Value(usize).init(0);
    var tasks = [_]PoolTestTask{
        .{ .completed = &completed },
        .{ .completed = &completed, .fail = true },
        .{ .completed = &completed },
        .{ .completed = &completed },
    };
    try pool.begin(tasks[0..], 3);
    pool.wait(tasks.len);
    try std.testing.expectEqual(@as(usize, 3), completed.load(.monotonic));
    try std.testing.expectEqual(@as(?anyerror, error.IncompleteRead), tasks[1].load_error);
    try std.testing.expectEqual(@as(?anyerror, null), tasks[0].load_error);
}

test "load worker pool rejects a second batch while one is active" {
    var pool = PoolUnderTest{};
    defer pool.deinit();
    var completed = std.atomic.Value(usize).init(0);
    var gate = std.atomic.Value(bool).init(false);
    var tasks = [_]PoolTestTask{.{ .completed = &completed, .gate = &gate }};
    var second = [_]PoolTestTask{.{ .completed = &completed }};
    try pool.begin(tasks[0..], 1);
    try std.testing.expectError(error.ExpertLoaderBusy, pool.begin(second[0..], 1));
    gate.store(true, .release);
    pool.wait(tasks.len);
    try std.testing.expectEqual(@as(usize, 1), completed.load(.monotonic));
}

test "load worker pool cancellation declines queued tasks and drains" {
    var pool = PoolUnderTest{};
    defer pool.deinit();
    var completed = std.atomic.Value(usize).init(0);
    var gate = std.atomic.Value(bool).init(false);
    var started = std.atomic.Value(bool).init(false);
    // One worker: the gated task holds the only thread, so the trailing
    // tasks are provably unstarted when cancel lands.
    var tasks = [_]PoolTestTask{
        .{ .completed = &completed, .gate = &gate, .started = &started },
        .{ .completed = &completed },
        .{ .completed = &completed },
    };
    try pool.begin(tasks[0..], 1);
    while (!started.load(.acquire)) platform.time.yieldBriefly();
    pool.cancel();
    gate.store(true, .release);
    pool.wait(tasks.len);
    try std.testing.expectEqual(@as(usize, 1), completed.load(.monotonic));
    try std.testing.expectEqual(@as(?anyerror, error.ExpertLoadCanceled), tasks[1].load_error);
    try std.testing.expectEqual(@as(?anyerror, error.ExpertLoadCanceled), tasks[2].load_error);
}

test "load worker pool shutdown joins parked workers and allows restart" {
    var pool = PoolUnderTest{};
    var completed = std.atomic.Value(usize).init(0);
    var tasks = [_]PoolTestTask{
        .{ .completed = &completed },
        .{ .completed = &completed },
    };
    try pool.begin(tasks[0..], 2);
    pool.wait(tasks.len);
    pool.deinit();
    try std.testing.expectEqual(@as(usize, 0), pool.worker_count);
    // A drained pool restarts cleanly after shutdown.
    var more = [_]PoolTestTask{.{ .completed = &completed }};
    try pool.begin(more[0..], 1);
    pool.wait(more.len);
    pool.deinit();
    try std.testing.expectEqual(@as(usize, 3), completed.load(.monotonic));
}

test "jina adapter names map base weight to PEFT LoRA tensors" {
    const allocator = std.testing.allocator;
    const adapter_a = try jinaAdapterANameForBaseWeight(allocator, "layers.0.self_attn.q_proj.weight");
    defer allocator.free(adapter_a);
    try std.testing.expectEqualStrings("base_model.model.layers.0.self_attn.q_proj.lora_A.weight", adapter_a);

    const adapter_b = try jinaAdapterBNameForAdapterA(allocator, adapter_a);
    defer allocator.free(adapter_b);
    try std.testing.expectEqualStrings("base_model.model.layers.0.self_attn.q_proj.lora_B.weight", adapter_b);
}

test "jina adapter merge ignores non-weight tensor names" {
    const allocator = std.testing.allocator;
    const adapter_path = try std.fmt.allocPrint(allocator, "/tmp/termite_jina_lora_empty_{d}.safetensors", .{std.posix.system.getpid()});
    defer allocator.free(adapter_path);
    defer std.Io.Dir.deleteFileAbsolute(std.testing.io, adapter_path) catch {};

    const raw_header = "{}";
    const header_len = std.mem.alignForward(usize, raw_header.len, 8);
    const file_bytes = try allocator.alloc(u8, 8 + header_len);
    defer allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header_len, .little);
    @memcpy(file_bytes[8 .. 8 + raw_header.len], raw_header);
    @memset(file_bytes[8 + raw_header.len ..], ' ');
    {
        var file = try std.Io.Dir.createFileAbsolute(std.testing.io, adapter_path, .{ .truncate = true });
        defer file.close(std.testing.io);
        try file.writeStreamingAll(std.testing.io, file_bytes);
    }

    var adapter = try JinaLoraAdapter.create(allocator, adapter_path, 1.0);
    defer adapter.destroy();

    const shape = [_]i64{1};
    const values = [_]f32{42.0};
    var base_weight = LoadedWeight{
        .tensor = try Tensor.initFloat32(allocator, "rotary_emb.inv_freq", &shape, &values),
    };
    defer base_weight.deinit();

    try adapter.mergeIntoLoadedWeight("rotary_emb.inv_freq", &base_weight);
    try std.testing.expectEqual(@as(f32, 42.0), base_weight.tensor.asFloat32()[0]);
}

test "gpu hosted jina lora merge applies scaled adapter update" {
    const allocator = std.testing.allocator;
    const base_shape = [_]i64{ 2, 3 };
    const a_shape = [_]i64{ 2, 3 };
    const b_shape = [_]i64{ 2, 2 };
    const base_values = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const a_values = [_]f32{
        1.0, 0.0, 2.0,
        0.0, 3.0, 1.0,
    };
    const b_values = [_]f32{
        2.0, 1.0,
        0.0, 4.0,
    };

    var base_weight = LoadedWeight{
        .tensor = try Tensor.initFloat32(allocator, "base", &base_shape, &base_values),
    };
    defer base_weight.deinit();
    var adapter_a = try Tensor.initFloat32(allocator, "a", &a_shape, &a_values);
    defer adapter_a.deinit();
    var adapter_b = try Tensor.initFloat32(allocator, "b", &b_shape, &b_values);
    defer adapter_b.deinit();

    try mergeLoraPairIntoWeight(&base_weight, adapter_a, adapter_b, 0.5);

    try std.testing.expectEqualSlices(f32, &.{
        2.0, 3.5,  5.5,
        4.0, 11.0, 8.0,
    }, base_weight.tensor.asFloat32());
}

test "jina lora adapter merges matching safetensors sidecar into loaded base weight" {
    const allocator = std.testing.allocator;
    const adapter_path = try std.fmt.allocPrint(allocator, "/tmp/termite_jina_lora_adapter_{d}.safetensors", .{std.posix.system.getpid()});
    defer allocator.free(adapter_path);
    defer std.Io.Dir.deleteFileAbsolute(std.testing.io, adapter_path) catch {};

    const a_values = [_]f32{
        1.0, 0.0, 2.0,
        0.0, 3.0, 1.0,
    };
    const b_values = [_]f32{
        2.0, 1.0,
        0.0, 4.0,
    };
    const a_bytes = std.mem.sliceAsBytes(&a_values);
    const b_bytes = std.mem.sliceAsBytes(&b_values);
    const raw_header = try std.fmt.allocPrint(
        allocator,
        "{{\"base_model.model.layers.0.self_attn.q_proj.lora_A.weight\":{{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,{d}]}},\"base_model.model.layers.0.self_attn.q_proj.lora_B.weight\":{{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[{d},{d}]}}}}",
        .{ a_bytes.len, a_bytes.len, a_bytes.len + b_bytes.len },
    );
    defer allocator.free(raw_header);
    const header_len = std.mem.alignForward(usize, raw_header.len, 8);
    const file_bytes = try allocator.alloc(u8, 8 + header_len + a_bytes.len + b_bytes.len);
    defer allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header_len, .little);
    @memcpy(file_bytes[8 .. 8 + raw_header.len], raw_header);
    @memset(file_bytes[8 + raw_header.len .. 8 + header_len], ' ');
    @memcpy(file_bytes[8 + header_len .. 8 + header_len + a_bytes.len], a_bytes);
    @memcpy(file_bytes[8 + header_len + a_bytes.len ..], b_bytes);

    {
        var file = try std.Io.Dir.createFileAbsolute(std.testing.io, adapter_path, .{ .truncate = true });
        defer file.close(std.testing.io);
        try file.writeStreamingAll(std.testing.io, file_bytes);
    }

    var adapter = try JinaLoraAdapter.create(allocator, adapter_path, 0.5);
    defer adapter.destroy();

    const base_shape = [_]i64{ 2, 3 };
    const base_values = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    var base_weight = LoadedWeight{
        .tensor = try Tensor.initFloat32(allocator, "layers.0.self_attn.q_proj.weight", &base_shape, &base_values),
    };
    defer base_weight.deinit();

    try adapter.mergeIntoLoadedWeight("layers.0.self_attn.q_proj.weight", &base_weight);

    try std.testing.expectEqualSlices(f32, &.{
        2.0, 3.5,  5.5,
        4.0, 11.0, 8.0,
    }, base_weight.tensor.asFloat32());
}
