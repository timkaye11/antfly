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

//! Backend-neutral bookkeeping for the compact streaming routed-expert
//! cache. This module owns slot *state* only: which expert occupies which
//! per-layer slot, its lifecycle (empty -> reserved -> loading -> resident
//! -> in_flight), LFU/recency accounting, generation counters, and pin
//! counts. Arena memory, file I/O, and device publication stay in the
//! backend that instantiates the cache.
//!
//! Miss planning is atomic: `planRoute` reserves every victim slot under one
//! lock, so duplicate expert reads and conflicting slot assignments cannot
//! occur even if a second planner raced. Slots referenced by an outstanding
//! plan or by in-flight device commands are pinned and can never be chosen
//! as victims; every mutation checks the slot generation so a stale
//! reference surfaces as an error instead of silent reuse.

const std = @import("std");
const platform = @import("antfly_platform");

pub const invalid_expert: u16 = std.math.maxInt(u16);

pub const SlotState = enum(u8) {
    empty,
    /// Assigned to an expert by an atomic plan; I/O not started yet.
    reserved,
    /// Expert bytes are being read into the slot's arena.
    loading,
    /// Bytes resident and published; safe to reference from new commands.
    resident,
    /// Resident and referenced by at least one in-flight device command.
    in_flight,
};

pub const SlotRef = struct {
    layer: u16,
    slot: u16,
    generation: u64,
};

pub const RoutePlanEntry = struct {
    ref: SlotRef,
    expert_index: u16,
    is_miss: bool,
};

pub const CacheError = error{
    RouteTooWide,
    DuplicateRouteExpert,
    LayerOutOfRange,
    SlotOutOfRange,
    ExpertPendingElsewhere,
    NoEvictableSlot,
    StaleSlotGeneration,
    InvalidSlotState,
};

pub const max_route_width = 8;

pub fn StreamingExpertCache(comptime layer_count: usize, comptime slot_capacity: usize) type {
    return struct {
        const Self = @This();

        pub const Slot = struct {
            state: SlotState = .empty,
            expert_index: u16 = invalid_expert,
            generation: u64 = 0,
            use_count: u64 = 0,
            last_access_epoch: u64 = 0,
            pin_count: u32 = 0,
        };

        pub const RoutePlan = struct {
            entries: [max_route_width]RoutePlanEntry = undefined,
            count: usize = 0,
            miss_count: usize = 0,

            pub fn slice(self: *const RoutePlan) []const RoutePlanEntry {
                return self.entries[0..self.count];
            }
        };

        mutex: std.atomic.Mutex = .unlocked,
        slots: [layer_count * slot_capacity]Slot = [_]Slot{.{}} ** (layer_count * slot_capacity),
        active_slots: usize = slot_capacity,
        epoch: u64 = 0,

        fn lock(self: *Self) void {
            platform.sync.lockYielding(&self.mutex);
        }

        fn unlock(self: *Self) void {
            self.mutex.unlock();
        }

        fn flatIndex(layer: usize, slot: usize) usize {
            return layer * slot_capacity + slot;
        }

        fn slotPtrLocked(self: *Self, ref: SlotRef) CacheError!*Slot {
            if (ref.layer >= layer_count) return error.LayerOutOfRange;
            if (ref.slot >= slot_capacity) return error.SlotOutOfRange;
            const slot = &self.slots[flatIndex(ref.layer, ref.slot)];
            if (slot.generation != ref.generation) return error.StaleSlotGeneration;
            return slot;
        }

        /// Set the active slot count (budget-derived for the compact
        /// profile, clamped to this cache's capacity). Shrinking does not
        /// touch occupied trailing slots by itself; the caller drains them
        /// through `collectEvictions` so it can decommit arenas.
        pub fn setActiveSlots(self: *Self, count: usize) void {
            self.lock();
            defer self.unlock();
            self.active_slots = @min(@max(count, 1), slot_capacity);
        }

        pub fn activeSlots(self: *Self) usize {
            self.lock();
            defer self.unlock();
            return self.active_slots;
        }

        /// Atomically plan one routed-expert set for a layer. Hits are pinned
        /// where they sit; each miss reserves a victim slot with a fresh
        /// generation. Every slot named by the returned plan carries one pin
        /// that `releasePlan` (or failure paths) must drop.
        pub fn planRoute(self: *Self, layer: usize, experts: []const u16) CacheError!RoutePlan {
            if (experts.len > max_route_width) return error.RouteTooWide;
            if (layer >= layer_count) return error.LayerOutOfRange;
            for (experts, 0..) |expert, index| {
                for (experts[0..index]) |previous| {
                    if (previous == expert) return error.DuplicateRouteExpert;
                }
            }

            self.lock();
            defer self.unlock();
            self.epoch += 1;
            var plan = RoutePlan{};
            errdefer self.rollbackPlanLocked(&plan);

            for (experts) |expert| {
                const entry = try self.planExpertLocked(layer, expert);
                plan.entries[plan.count] = entry;
                plan.count += 1;
                if (entry.is_miss) plan.miss_count += 1;
            }
            return plan;
        }

        fn planExpertLocked(self: *Self, layer: usize, expert: u16) CacheError!RoutePlanEntry {
            // Hit scan over the active tier.
            for (0..self.active_slots) |local| {
                const slot = &self.slots[flatIndex(layer, local)];
                if (slot.expert_index != expert) continue;
                switch (slot.state) {
                    .resident, .in_flight => {
                        slot.use_count +|= 1;
                        slot.last_access_epoch = self.epoch;
                        slot.pin_count += 1;
                        return .{
                            .ref = .{ .layer = @intCast(layer), .slot = @intCast(local), .generation = slot.generation },
                            .expert_index = expert,
                            .is_miss = false,
                        };
                    },
                    // A reserved/loading slot already owns this expert on
                    // behalf of another outstanding plan. Attaching would
                    // alias one read into two plans; refuse instead.
                    .reserved, .loading => return error.ExpertPendingElsewhere,
                    .empty => {},
                }
            }

            const victim = self.chooseVictimLocked(layer) orelse return error.NoEvictableSlot;
            const slot = &self.slots[flatIndex(layer, victim)];
            slot.state = .reserved;
            slot.expert_index = expert;
            slot.generation += 1;
            slot.use_count = 1;
            slot.last_access_epoch = self.epoch;
            slot.pin_count = 1;
            return .{
                .ref = .{ .layer = @intCast(layer), .slot = @intCast(victim), .generation = slot.generation },
                .expert_index = expert,
                .is_miss = true,
            };
        }

        /// LFU with recency tie-break over unpinned empty/resident slots in
        /// the active tier. Empty slots win outright so the cache fills
        /// before it evicts.
        fn chooseVictimLocked(self: *Self, layer: usize) ?usize {
            var best: ?usize = null;
            for (0..self.active_slots) |local| {
                const slot = &self.slots[flatIndex(layer, local)];
                if (slot.pin_count != 0) continue;
                switch (slot.state) {
                    .empty => return local,
                    .resident => {},
                    .reserved, .loading, .in_flight => continue,
                }
                if (best) |current| {
                    const current_slot = &self.slots[flatIndex(layer, current)];
                    if (slot.use_count < current_slot.use_count or
                        (slot.use_count == current_slot.use_count and
                            slot.last_access_epoch < current_slot.last_access_epoch))
                    {
                        best = local;
                    }
                } else {
                    best = local;
                }
            }
            return best;
        }

        fn rollbackPlanLocked(self: *Self, plan: *const RoutePlan) void {
            for (plan.slice()) |entry| {
                const slot = &self.slots[flatIndex(entry.ref.layer, entry.ref.slot)];
                if (slot.generation != entry.ref.generation) continue;
                if (slot.pin_count > 0) slot.pin_count -= 1;
                if (entry.is_miss and slot.state == .reserved) {
                    slot.state = .empty;
                    slot.expert_index = invalid_expert;
                }
            }
        }

        /// reserved -> loading once the expert's reads have been issued.
        pub fn noteLoading(self: *Self, ref: SlotRef) CacheError!void {
            self.lock();
            defer self.unlock();
            const slot = try self.slotPtrLocked(ref);
            if (slot.state != .reserved) return error.InvalidSlotState;
            slot.state = .loading;
        }

        /// reserved/loading -> resident once bytes are in place and the
        /// backend finished publication.
        pub fn commitResident(self: *Self, ref: SlotRef) CacheError!void {
            self.lock();
            defer self.unlock();
            const slot = try self.slotPtrLocked(ref);
            switch (slot.state) {
                .reserved, .loading => slot.state = .resident,
                else => return error.InvalidSlotState,
            }
        }

        /// Abandon a planned or loading slot after an I/O failure or
        /// cancellation. Drops the plan pin and frees the slot.
        pub fn releaseFailed(self: *Self, ref: SlotRef) void {
            self.lock();
            defer self.unlock();
            const slot = self.slotPtrLocked(ref) catch return;
            if (slot.pin_count > 0) slot.pin_count -= 1;
            if (slot.pin_count != 0) return;
            slot.state = .empty;
            slot.expert_index = invalid_expert;
        }

        /// resident -> in_flight while device commands reference the slot.
        pub fn beginUse(self: *Self, ref: SlotRef) CacheError!void {
            self.lock();
            defer self.unlock();
            const slot = try self.slotPtrLocked(ref);
            switch (slot.state) {
                .resident, .in_flight => {
                    slot.state = .in_flight;
                    slot.pin_count += 1;
                },
                else => return error.InvalidSlotState,
            }
        }

        /// Drop one in-flight reference after the device fence confirms the
        /// commands completed. The last reference returns the slot to
        /// resident.
        pub fn endUse(self: *Self, ref: SlotRef) CacheError!void {
            self.lock();
            defer self.unlock();
            const slot = try self.slotPtrLocked(ref);
            if (slot.state != .in_flight or slot.pin_count == 0) return error.InvalidSlotState;
            slot.pin_count -= 1;
            if (slot.pin_count == 0) slot.state = .resident;
        }

        /// Drop the plan-time pins taken by `planRoute` once the route has
        /// executed (or was abandoned before any I/O started).
        pub fn releasePlan(self: *Self, plan: *const RoutePlan) void {
            self.lock();
            defer self.unlock();
            for (plan.slice()) |entry| {
                self.releaseEntryLocked(entry.ref);
            }
        }

        /// Drop a single plan pin without freeing the slot. Used when a plan
        /// unwinds mixed hit/miss state and only some entries failed.
        pub fn releaseEntry(self: *Self, ref: SlotRef) void {
            self.lock();
            defer self.unlock();
            self.releaseEntryLocked(ref);
        }

        fn releaseEntryLocked(self: *Self, ref: SlotRef) void {
            const slot = self.slotPtrLocked(ref) catch return;
            if (slot.pin_count > 0) slot.pin_count -= 1;
            if (slot.pin_count == 0 and slot.state == .in_flight) slot.state = .resident;
        }

        /// Evict up to `out.len` unpinned resident slots, coldest first
        /// (global LFU with recency tie-break), including slots stranded
        /// above a lowered active tier. Returns refs the caller must
        /// decommit and unpublish; the bookkeeping is already cleared.
        pub fn collectEvictions(self: *Self, out: []SlotRef) usize {
            self.lock();
            defer self.unlock();
            var produced: usize = 0;
            while (produced < out.len) {
                var best: ?usize = null;
                for (self.slots, 0..) |slot, index| {
                    if (slot.state != .resident or slot.pin_count != 0) continue;
                    if (best) |current| {
                        const current_slot = &self.slots[current];
                        const stranded = (index % slot_capacity) >= self.active_slots;
                        const current_stranded = (current % slot_capacity) >= self.active_slots;
                        if (stranded != current_stranded) {
                            if (stranded) best = index;
                            continue;
                        }
                        if (slot.use_count < current_slot.use_count or
                            (slot.use_count == current_slot.use_count and
                                slot.last_access_epoch < current_slot.last_access_epoch))
                        {
                            best = index;
                        }
                    } else {
                        best = index;
                    }
                }
                const victim = best orelse break;
                const slot = &self.slots[victim];
                out[produced] = .{
                    .layer = @intCast(victim / slot_capacity),
                    .slot = @intCast(victim % slot_capacity),
                    .generation = slot.generation,
                };
                slot.state = .empty;
                slot.expert_index = invalid_expert;
                slot.use_count = 0;
                slot.last_access_epoch = 0;
                produced += 1;
            }
            return produced;
        }

        /// Evict only unpinned resident slots stranded above the active
        /// tier. Used after a tier downshift so in-tier contents survive.
        pub fn collectStranded(self: *Self, out: []SlotRef) usize {
            self.lock();
            defer self.unlock();
            var produced: usize = 0;
            for (&self.slots, 0..) |*slot, index| {
                if (produced >= out.len) break;
                if (slot.state != .resident or slot.pin_count != 0) continue;
                if ((index % slot_capacity) < self.active_slots) continue;
                out[produced] = .{
                    .layer = @intCast(index / slot_capacity),
                    .slot = @intCast(index % slot_capacity),
                    .generation = slot.generation,
                };
                slot.state = .empty;
                slot.expert_index = invalid_expert;
                slot.use_count = 0;
                slot.last_access_epoch = 0;
                produced += 1;
            }
            return produced;
        }

        pub const Counts = struct {
            resident: usize = 0,
            in_flight: usize = 0,
            reserved_or_loading: usize = 0,
        };

        pub fn counts(self: *Self) Counts {
            self.lock();
            defer self.unlock();
            var result = Counts{};
            for (self.slots) |slot| {
                switch (slot.state) {
                    .resident => result.resident += 1,
                    .in_flight => result.in_flight += 1,
                    .reserved, .loading => result.reserved_or_loading += 1,
                    .empty => {},
                }
            }
            return result;
        }
    };
}

const TestCache = StreamingExpertCache(2, 4);

fn commitPlan(cache: *TestCache, plan: *const TestCache.RoutePlan) !void {
    for (plan.slice()) |entry| {
        if (!entry.is_miss) continue;
        try cache.noteLoading(entry.ref);
        try cache.commitResident(entry.ref);
    }
}

test "streaming cache plans misses, hits, and LFU victims with recency tie-break" {
    var cache = TestCache{};
    var plan = try cache.planRoute(0, &.{ 10, 11, 12, 13 });
    try std.testing.expectEqual(@as(usize, 4), plan.miss_count);
    try commitPlan(&cache, &plan);
    cache.releasePlan(&plan);

    // Reuse expert 10 twice so it becomes clearly hot.
    for (0..2) |_| {
        var hot = try cache.planRoute(0, &.{10});
        try std.testing.expectEqual(@as(usize, 0), hot.miss_count);
        try commitPlan(&cache, &hot);
        cache.releasePlan(&hot);
    }

    // A new expert must evict the least-frequently-used slot; 11/12/13 tie
    // on use_count so the oldest epoch (expert 11) loses.
    var evicting = try cache.planRoute(0, &.{14});
    try std.testing.expectEqual(@as(usize, 1), evicting.miss_count);
    try std.testing.expectEqual(@as(u16, 1), evicting.entries[0].ref.slot);
    try commitPlan(&cache, &evicting);
    cache.releasePlan(&evicting);

    // Expert 11 was evicted, so it must now miss; expert 10 still hits.
    var mixed = try cache.planRoute(0, &.{ 10, 11 });
    try std.testing.expectEqual(@as(usize, 1), mixed.miss_count);
    try std.testing.expect(!mixed.entries[0].is_miss);
    try std.testing.expect(mixed.entries[1].is_miss);
    try commitPlan(&cache, &mixed);
    cache.releasePlan(&mixed);
}

test "streaming cache refuses duplicate experts and protects planned slots" {
    var cache = TestCache{};
    try std.testing.expectError(error.DuplicateRouteExpert, cache.planRoute(0, &.{ 5, 5 }));

    // Fill the layer and keep the plan pinned: a follow-up plan that needs a
    // victim must fail rather than steal a pinned slot.
    var pinned = try cache.planRoute(0, &.{ 1, 2, 3, 4 });
    try std.testing.expectError(error.NoEvictableSlot, cache.planRoute(0, &.{9}));
    // The same expert as an outstanding miss cannot be aliased into a second
    // plan while its read is pending.
    try std.testing.expectError(error.ExpertPendingElsewhere, cache.planRoute(0, &.{1}));
    try commitPlan(&cache, &pinned);
    cache.releasePlan(&pinned);

    // Unpinned now: planning expert 9 evicts and succeeds.
    const after = try cache.planRoute(0, &.{9});
    try std.testing.expectEqual(@as(usize, 1), after.miss_count);
    cache.releaseFailed(after.entries[0].ref);
}

test "streaming cache in-flight slots are never victims and generations stay honest" {
    var cache = TestCache{};
    var plan = try cache.planRoute(1, &.{ 21, 22, 23, 24 });
    try commitPlan(&cache, &plan);
    cache.releasePlan(&plan);

    var route = try cache.planRoute(1, &.{21});
    const ref = route.entries[0].ref;
    try cache.beginUse(ref);
    // While in flight (and with the other three unpinned), an eviction sweep
    // must skip the in-flight slot.
    var evicted: [4]SlotRef = undefined;
    const count = cache.collectEvictions(evicted[0..]);
    try std.testing.expectEqual(@as(usize, 3), count);
    for (evicted[0..count]) |victim| {
        try std.testing.expect(victim.slot != ref.slot or victim.layer != ref.layer);
    }
    try cache.endUse(ref);
    cache.releasePlan(&route);

    // An evicted ref can never re-enter use: the slot is either empty under
    // the same generation (InvalidSlotState) or reassigned under a newer one
    // (StaleSlotGeneration).
    const reused = try cache.planRoute(1, &.{40});
    try std.testing.expectEqual(@as(usize, 1), reused.miss_count);
    try std.testing.expect(if (cache.beginUse(evicted[0])) |_| false else |err| switch (err) {
        error.InvalidSlotState, error.StaleSlotGeneration => true,
        else => false,
    });
    for (reused.slice()) |entry| {
        if (entry.is_miss) cache.releaseFailed(entry.ref);
    }
}

test "streaming cache tier shrink strands slots for eviction and keeps planning bounded" {
    var cache = TestCache{};
    var plan = try cache.planRoute(0, &.{ 1, 2, 3, 4 });
    try commitPlan(&cache, &plan);
    cache.releasePlan(&plan);

    cache.setActiveSlots(2);
    // Experts living in slots >= 2 are outside the tier: they miss now, and
    // the planner may only use the first two slots.
    var narrow = try cache.planRoute(0, &.{5});
    try std.testing.expectEqual(@as(usize, 1), narrow.miss_count);
    try std.testing.expect(narrow.entries[0].ref.slot < 2);
    try commitPlan(&cache, &narrow);
    cache.releasePlan(&narrow);

    // Stranded above-tier slots are preferred eviction victims.
    var evicted: [4]SlotRef = undefined;
    const count = cache.collectEvictions(evicted[0..2]);
    try std.testing.expectEqual(@as(usize, 2), count);
    for (evicted[0..count]) |victim| {
        try std.testing.expect(victim.slot >= 2);
    }
    // Expert 5 plus the surviving in-tier expert remain resident.
    try std.testing.expectEqual(@as(usize, 2), cache.counts().resident);
}
