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

//! Process-wide residency ledger for the compact memory profile.
//!
//! Every byte class that contributes to the model instance's footprint is
//! charged here: fixed model buffers, committed expert pages, KV, scratch,
//! retained prompt-cache KV, loader-queue metadata, and Metal buffer
//! objects. Virtual slot capacity (reserved address space) is reported
//! separately from committed resident bytes, because page-aligned arenas can
//! be decommitted while keeping their mapping.
//!
//! Enforcement is two-tiered: crossing the soft limit (ceiling minus the
//! safety reserve) asks the caller to evict and decommit cold expert pages;
//! crossing the hard ceiling rejects the projected allocation or aborts the
//! generation. A Darwin phys_footprint sampler provides the independent
//! process-level guard — it observes KV, scratch, and allocator overhead
//! even before those call sites are individually instrumented.

const std = @import("std");
const builtin = @import("builtin");
const platform = @import("antfly_platform");

pub const BudgetCategory = enum(u8) {
    fixed_model,
    expert_pages,
    kv_cache,
    scratch,
    prompt_cache_kv,
    queue_metadata,
    metal_buffers,
};

pub const category_count = std.enums.values(BudgetCategory).len;

pub const BudgetPressure = enum(u8) {
    ok,
    /// Above ceiling - safety reserve: evict and decommit cold expert pages.
    soft,
    /// At or above the ceiling: reject projected allocations or abort.
    hard,
};

pub const LedgerSnapshot = struct {
    ceiling_bytes: u64,
    safety_reserve_bytes: u64,
    committed_total_bytes: u64,
    committed_by_category: [category_count]u64,
    virtual_slot_capacity_bytes: u64,
    peak_committed_bytes: u64,
    observed_footprint_bytes: u64,
    observed_footprint_peak_bytes: u64,
    soft_limit_trips: u64,
    hard_limit_rejections: u64,
    prompt_cache_overcommit_bytes: u64,
};

pub const ResidentBudgetLedger = struct {
    mutex: std.atomic.Mutex = .unlocked,
    ceiling_bytes: u64,
    safety_reserve_bytes: u64,
    committed: [category_count]u64 = @splat(0),
    virtual_slot_capacity_bytes: u64 = 0,
    peak_committed_bytes: u64 = 0,
    observed_footprint_bytes: u64 = 0,
    observed_footprint_peak_bytes: u64 = 0,
    soft_limit_trips: u64 = 0,
    hard_limit_rejections: u64 = 0,
    /// Retained prompt-cache KV bytes the ledger refused to commit because
    /// they would cross the hard ceiling. Absolute gauge, not a counter: the
    /// prompt-cache observer rewrites it on every accounting update. Nonzero
    /// means the cache's own max_bytes budget admits more retained KV than
    /// this instance's residency contract does.
    prompt_cache_overcommit_bytes: u64 = 0,
    /// Test seam: when set, the sampler reads this instead of the OS.
    footprint_override: ?u64 = null,

    pub fn init(ceiling_bytes: u64, safety_reserve_bytes: u64) ResidentBudgetLedger {
        return .{
            .ceiling_bytes = ceiling_bytes,
            .safety_reserve_bytes = safety_reserve_bytes,
        };
    }

    fn lock(self: *ResidentBudgetLedger) void {
        platform.sync.lockYielding(&self.mutex);
    }

    fn unlock(self: *ResidentBudgetLedger) void {
        self.mutex.unlock();
    }

    pub fn softLimitBytes(self: *const ResidentBudgetLedger) u64 {
        return self.ceiling_bytes -| self.safety_reserve_bytes;
    }

    fn committedTotalLocked(self: *const ResidentBudgetLedger) u64 {
        var total: u64 = 0;
        for (self.committed) |bytes| total +|= bytes;
        return total;
    }

    pub fn committedTotal(self: *ResidentBudgetLedger) u64 {
        self.lock();
        defer self.unlock();
        return self.committedTotalLocked();
    }

    pub fn setVirtualSlotCapacity(self: *ResidentBudgetLedger, bytes: u64) void {
        self.lock();
        defer self.unlock();
        self.virtual_slot_capacity_bytes = bytes;
    }

    /// Charge committed bytes against the hard ceiling. Fail-closed: a
    /// projected total at or above the ceiling is rejected, not clamped.
    pub fn reserve(self: *ResidentBudgetLedger, category: BudgetCategory, bytes: u64) error{CompactBudgetExceeded}!void {
        self.lock();
        defer self.unlock();
        const projected = self.committedTotalLocked() +| bytes;
        if (projected > self.ceiling_bytes) {
            self.hard_limit_rejections +|= 1;
            return error.CompactBudgetExceeded;
        }
        self.committed[@intFromEnum(category)] +|= bytes;
        self.peak_committed_bytes = @max(self.peak_committed_bytes, projected);
    }

    /// Observability-only charge for byte classes whose device allocation
    /// already happened elsewhere (e.g. Metal KV slot buffers): the bytes are
    /// recorded so category accounting, snapshots, and pressure logs stay
    /// honest, but the charge is never rejected — enforcement for these
    /// classes remains with the independent phys-footprint sampler, which
    /// observes them regardless of ledger accounting.
    pub fn noteCommittedObserved(self: *ResidentBudgetLedger, category: BudgetCategory, bytes: u64) void {
        self.lock();
        defer self.unlock();
        self.committed[@intFromEnum(category)] +|= bytes;
        self.peak_committed_bytes = @max(self.peak_committed_bytes, self.committedTotalLocked());
    }

    /// Return committed bytes (page decommit, buffer release, eviction).
    pub fn release(self: *ResidentBudgetLedger, category: BudgetCategory, bytes: u64) void {
        self.lock();
        defer self.unlock();
        const slot = &self.committed[@intFromEnum(category)];
        slot.* -|= bytes;
    }

    /// Replace the prompt-cache overcommit gauge, returning the previous
    /// value so the caller can log only on the idle -> overcommitted edge.
    fn swapPromptCacheOvercommit(self: *ResidentBudgetLedger, bytes: u64) u64 {
        self.lock();
        defer self.unlock();
        const previous = self.prompt_cache_overcommit_bytes;
        self.prompt_cache_overcommit_bytes = bytes;
        return previous;
    }

    /// Sample the Darwin phys_footprint (or the override in tests) and fold
    /// it into the pressure decision alongside the ledger's own accounting.
    pub fn samplePressure(self: *ResidentBudgetLedger) BudgetPressure {
        const sampled: u64 = if (self.footprint_override) |value|
            value
        else blk: {
            const stats = platform.process_memory.pressureSnapshot();
            break :blk if (stats.available) stats.footprint_bytes else 0;
        };
        self.lock();
        defer self.unlock();
        if (sampled != 0) {
            self.observed_footprint_bytes = sampled;
            self.observed_footprint_peak_bytes = @max(self.observed_footprint_peak_bytes, sampled);
        }
        const committed_total = self.committedTotalLocked();
        const pressure_bytes = @max(committed_total, sampled);
        if (pressure_bytes >= self.ceiling_bytes) {
            self.hard_limit_rejections +|= 1;
            return .hard;
        }
        if (pressure_bytes > self.softLimitBytes()) {
            self.soft_limit_trips +|= 1;
            return .soft;
        }
        return .ok;
    }

    pub fn snapshot(self: *ResidentBudgetLedger) LedgerSnapshot {
        self.lock();
        defer self.unlock();
        return .{
            .ceiling_bytes = self.ceiling_bytes,
            .safety_reserve_bytes = self.safety_reserve_bytes,
            .committed_total_bytes = self.committedTotalLocked(),
            .committed_by_category = self.committed,
            .virtual_slot_capacity_bytes = self.virtual_slot_capacity_bytes,
            .peak_committed_bytes = self.peak_committed_bytes,
            .observed_footprint_bytes = self.observed_footprint_bytes,
            .observed_footprint_peak_bytes = self.observed_footprint_peak_bytes,
            .soft_limit_trips = self.soft_limit_trips,
            .hard_limit_rejections = self.hard_limit_rejections,
            .prompt_cache_overcommit_bytes = self.prompt_cache_overcommit_bytes,
        };
    }
};

/// Prompt-prefix-cache resource observer endpoint. `context` is the owning
/// model instance's ledger; the cache reports absolute retained bytes and
/// this converts them to release-then-reserve against `.prompt_cache_kv`.
///
/// The observer contract (runtime/kv/prompt_cache.zig ResourceUsageObserver)
/// is void-return advisory accounting: the cache has already retained or
/// released the KV blocks by the time it reports, and its own `max_bytes`
/// config drives eviction. On reserve failure this therefore stays
/// fail-closed for the LEDGER — committed bytes never cross the ceiling —
/// while the unaccounted retained bytes are surfaced through the
/// `prompt_cache_overcommit_bytes` gauge (and a one-shot warning per
/// overcommit episode) instead of being silently clamped or partially
/// reserved. `current` is set to what was actually committed, so the next
/// update releases exactly that amount. A nonzero gauge is the signal to
/// lower the node prompt-cache target or raise the compact ceiling; the
/// eviction decision itself stays with the cache.
///
/// Exactly one cache feeds a given ledger's `.prompt_cache_kv` category (the
/// per-model PromptPrefixCache), so the absolute-value rewrite is race-free.
pub fn promptCacheObserverUpdate(context: *anyopaque, current: *u64, next: u64) void {
    const ledger: *ResidentBudgetLedger = @ptrCast(@alignCast(context));
    ledger.release(.prompt_cache_kv, current.*);
    if (ledger.reserve(.prompt_cache_kv, next)) {
        _ = ledger.swapPromptCacheOvercommit(0);
        current.* = next;
    } else |_| {
        const previous = ledger.swapPromptCacheOvercommit(next);
        current.* = 0;
        if (previous == 0 and !builtin.is_test) {
            std.log.warn(
                "prompt cache retained KV exceeds the compact residency budget: {d} bytes uncommitted (ceiling {d}); lower the prompt cache target or raise the profile ceiling",
                .{ next, ledger.ceiling_bytes },
            );
        }
    }
}

test "budget ledger accounts categories and rejects at the hard ceiling" {
    var ledger = ResidentBudgetLedger.init(1000, 100);
    try ledger.reserve(.fixed_model, 500);
    try ledger.reserve(.expert_pages, 300);
    try std.testing.expectEqual(@as(u64, 800), ledger.committedTotal());
    try std.testing.expectError(error.CompactBudgetExceeded, ledger.reserve(.kv_cache, 300));
    ledger.release(.expert_pages, 200);
    try ledger.reserve(.kv_cache, 300);
    try std.testing.expectEqual(@as(u64, 900), ledger.committedTotal());

    const snap = ledger.snapshot();
    try std.testing.expectEqual(@as(u64, 900), snap.committed_total_bytes);
    try std.testing.expectEqual(@as(u64, 500), snap.committed_by_category[@intFromEnum(BudgetCategory.fixed_model)]);
    try std.testing.expectEqual(@as(u64, 100), snap.committed_by_category[@intFromEnum(BudgetCategory.expert_pages)]);
    try std.testing.expectEqual(@as(u64, 900), snap.peak_committed_bytes);
    try std.testing.expectEqual(@as(u64, 1), snap.hard_limit_rejections);
}

test "budget ledger observed charges never reject and stay releasable" {
    var ledger = ResidentBudgetLedger.init(1000, 100);
    try ledger.reserve(.fixed_model, 900);
    // A fail-closed reserve would reject this; the observed charge records
    // it so the KV category and pressure logs stay honest.
    ledger.noteCommittedObserved(.kv_cache, 300);
    var snap = ledger.snapshot();
    try std.testing.expectEqual(@as(u64, 300), snap.committed_by_category[@intFromEnum(BudgetCategory.kv_cache)]);
    try std.testing.expectEqual(@as(u64, 1200), snap.committed_total_bytes);
    try std.testing.expectEqual(@as(u64, 1200), snap.peak_committed_bytes);
    try std.testing.expectEqual(@as(u64, 0), snap.hard_limit_rejections);
    // Observed bytes participate in pressure like every other category.
    ledger.footprint_override = 0;
    try std.testing.expectEqual(BudgetPressure.hard, ledger.samplePressure());
    ledger.release(.kv_cache, 300);
    snap = ledger.snapshot();
    try std.testing.expectEqual(@as(u64, 0), snap.committed_by_category[@intFromEnum(BudgetCategory.kv_cache)]);
    try std.testing.expectEqual(@as(u64, 900), snap.committed_total_bytes);
}

test "budget ledger decommit accounting keeps virtual capacity separate" {
    var ledger = ResidentBudgetLedger.init(2_120_000_000, 128 * 1024 * 1024);
    ledger.setVirtualSlotCapacity(30 * 16 * 3_358_720);
    try ledger.reserve(.expert_pages, 12 * 3_358_720);
    ledger.release(.expert_pages, 4 * 3_358_720);
    const snap = ledger.snapshot();
    try std.testing.expectEqual(@as(u64, 8 * 3_358_720), snap.committed_total_bytes);
    try std.testing.expectEqual(@as(u64, 30 * 16 * 3_358_720), snap.virtual_slot_capacity_bytes);
    try std.testing.expect(snap.virtual_slot_capacity_bytes > snap.committed_total_bytes);
}

test "budget ledger pressure tiers use the larger of ledger and sampler" {
    var ledger = ResidentBudgetLedger.init(1000, 100);
    ledger.footprint_override = 0;
    try ledger.reserve(.fixed_model, 800);
    try std.testing.expectEqual(BudgetPressure.ok, ledger.samplePressure());
    // Sampler observes pressure the ledger categories don't yet attribute.
    ledger.footprint_override = 950;
    try std.testing.expectEqual(BudgetPressure.soft, ledger.samplePressure());
    ledger.footprint_override = 1000;
    try std.testing.expectEqual(BudgetPressure.hard, ledger.samplePressure());
    // Ledger accounting alone can also trip the soft limit.
    ledger.footprint_override = 0;
    try ledger.reserve(.kv_cache, 150);
    try std.testing.expectEqual(BudgetPressure.soft, ledger.samplePressure());
    const snap = ledger.snapshot();
    try std.testing.expectEqual(@as(u64, 1000), snap.observed_footprint_peak_bytes);
    try std.testing.expect(snap.soft_limit_trips >= 2);
    try std.testing.expect(snap.hard_limit_rejections >= 1);
}

test "prompt cache observer converts absolute usage to ledger transitions" {
    var ledger = ResidentBudgetLedger.init(1000, 0);
    try ledger.reserve(.fixed_model, 400);
    var accounted: u64 = 0;

    // Grow: 0 -> 300.
    promptCacheObserverUpdate(&ledger, &accounted, 300);
    try std.testing.expectEqual(@as(u64, 300), accounted);
    var snap = ledger.snapshot();
    try std.testing.expectEqual(@as(u64, 300), snap.committed_by_category[@intFromEnum(BudgetCategory.prompt_cache_kv)]);
    try std.testing.expectEqual(@as(u64, 700), snap.committed_total_bytes);
    try std.testing.expectEqual(@as(u64, 0), snap.prompt_cache_overcommit_bytes);

    // Shrink: 300 -> 100 releases the difference.
    promptCacheObserverUpdate(&ledger, &accounted, 100);
    try std.testing.expectEqual(@as(u64, 100), accounted);
    snap = ledger.snapshot();
    try std.testing.expectEqual(@as(u64, 100), snap.committed_by_category[@intFromEnum(BudgetCategory.prompt_cache_kv)]);

    // Release-on-zero: the category empties without touching other classes.
    promptCacheObserverUpdate(&ledger, &accounted, 0);
    try std.testing.expectEqual(@as(u64, 0), accounted);
    snap = ledger.snapshot();
    try std.testing.expectEqual(@as(u64, 0), snap.committed_by_category[@intFromEnum(BudgetCategory.prompt_cache_kv)]);
    try std.testing.expectEqual(@as(u64, 400), snap.committed_total_bytes);
}

test "prompt cache observer stays fail-closed and reports overcommit" {
    var ledger = ResidentBudgetLedger.init(1000, 0);
    try ledger.reserve(.fixed_model, 900);
    var accounted: u64 = 0;

    // Fits: 0 -> 80.
    promptCacheObserverUpdate(&ledger, &accounted, 80);
    try std.testing.expectEqual(@as(u64, 80), accounted);

    // Would cross the ceiling: nothing stays committed, the retained bytes
    // surface as the overcommit gauge, and the rejection is counted.
    promptCacheObserverUpdate(&ledger, &accounted, 200);
    try std.testing.expectEqual(@as(u64, 0), accounted);
    var snap = ledger.snapshot();
    try std.testing.expectEqual(@as(u64, 0), snap.committed_by_category[@intFromEnum(BudgetCategory.prompt_cache_kv)]);
    try std.testing.expectEqual(@as(u64, 900), snap.committed_total_bytes);
    try std.testing.expectEqual(@as(u64, 200), snap.prompt_cache_overcommit_bytes);
    try std.testing.expect(snap.hard_limit_rejections >= 1);

    // The cache evicting back under budget clears the gauge and recommits.
    promptCacheObserverUpdate(&ledger, &accounted, 50);
    try std.testing.expectEqual(@as(u64, 50), accounted);
    snap = ledger.snapshot();
    try std.testing.expectEqual(@as(u64, 50), snap.committed_by_category[@intFromEnum(BudgetCategory.prompt_cache_kv)]);
    try std.testing.expectEqual(@as(u64, 0), snap.prompt_cache_overcommit_bytes);
}

test "budget ledger live darwin sampler reports this process when available" {
    var ledger = ResidentBudgetLedger.init(std.math.maxInt(u64), 0);
    _ = ledger.samplePressure();
    const stats = platform.process_memory.pressureSnapshot();
    if (stats.available and stats.footprint_bytes != 0) {
        try std.testing.expect(ledger.snapshot().observed_footprint_bytes != 0);
    }
}
