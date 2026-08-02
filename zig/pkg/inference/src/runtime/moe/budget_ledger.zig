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
//! Explicitly owned allocations are attributed here when their lifetime is
//! known: committed expert pages, KV, scratch, and retained prompt-cache KV.
//! Categories for fixed model buffers, loader metadata, and Metal buffers are
//! available to owners that can make the same lifetime guarantee, but clean
//! file mappings and backend-internal allocations are not guessed from
//! artifact or virtual sizes. The low-overhead process working-set sampler is
//! therefore the authoritative total-footprint signal; enforcement uses the
//! larger of sampled process pressure and attributed reservations.
//!
//! Virtual slot capacity (reserved address space) is reported separately from
//! committed resident bytes, because page-aligned arenas can be decommitted
//! while keeping their mapping.
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

const FootprintReading = struct {
    available: bool,
    bytes: u64,
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
    growth_limit_rejections: u64,
    sampler_required: bool,
    sampler_available: bool,
    sampler_unavailable_rejections: u64,
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
    growth_limit_rejections: u64 = 0,
    sampler_required: bool = false,
    sampler_available: bool = false,
    sampler_unavailable_rejections: u64 = 0,
    /// Test seam: when set, the sampler reads this instead of the OS.
    footprint_override: ?u64 = null,
    /// Test seam for a failed OS sample. Production code never sets it.
    footprint_unavailable_override: bool = false,

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

    /// Enforcing compact sessions require an authoritative process sample.
    /// This is set once during session construction, before the ledger is
    /// shared with allocation owners.
    pub fn setProcessSamplerRequired(self: *ResidentBudgetLedger, required: bool) void {
        self.lock();
        defer self.unlock();
        self.sampler_required = required;
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

    /// Atomically replace an absolute reservation for one category. This is
    /// used by retained caches so admission happens before ownership changes;
    /// a rejected growth leaves both the old reservation and cache untouched.
    pub fn replaceReservation(
        self: *ResidentBudgetLedger,
        category: BudgetCategory,
        current: u64,
        next: u64,
    ) error{CompactBudgetExceeded}!void {
        self.lock();
        defer self.unlock();
        const category_index = @intFromEnum(category);
        const recorded = self.committed[category_index];
        const removable = @min(current, recorded);
        const base = self.committedTotalLocked() -| removable;
        const projected = base +| next;
        if (projected > self.ceiling_bytes) {
            self.hard_limit_rejections +|= 1;
            return error.CompactBudgetExceeded;
        }
        self.committed[category_index] = recorded -| removable +| next;
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

    /// Sample the process pressure working set (Darwin phys_footprint, Linux
    /// private/RSS fallback, or the override in tests) and fold it into the
    /// pressure decision alongside the ledger's own accounting.
    pub fn samplePressure(self: *ResidentBudgetLedger) BudgetPressure {
        const reading: FootprintReading = if (self.footprint_unavailable_override)
            .{ .available = false, .bytes = @as(u64, 0) }
        else if (self.footprint_override) |value|
            .{ .available = true, .bytes = value }
        else blk: {
            const stats = platform.process_memory.pressureSnapshot();
            break :blk .{
                .available = stats.available,
                .bytes = if (stats.available)
                    platform.process_memory.pressureWorkingSetBytes(stats)
                else
                    0,
            };
        };
        self.lock();
        defer self.unlock();
        self.sampler_available = reading.available;
        if (reading.available) {
            self.observed_footprint_bytes = reading.bytes;
            self.observed_footprint_peak_bytes = @max(self.observed_footprint_peak_bytes, reading.bytes);
        } else if (self.sampler_required) {
            self.sampler_unavailable_rejections +|= 1;
            self.hard_limit_rejections +|= 1;
            return .hard;
        }
        const committed_total = self.committedTotalLocked();
        // A transient unavailable sample must not erase the last valid process
        // observation. Non-enforcing ledgers can continue from that retained
        // high-water point; enforcing ledgers have already failed closed.
        const pressure_bytes = @max(committed_total, self.observed_footprint_bytes);
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

    /// Admission check for a bounded allocation batch. Unlike `reserve`, this
    /// includes the sampled process working set and preserves the configured
    /// safety reserve. The caller still performs the individual category
    /// reservations as ownership is acquired.
    pub fn admitGrowthWithinReserve(
        self: *ResidentBudgetLedger,
        bytes: u64,
    ) error{CompactBudgetExceeded}!void {
        if (bytes == 0) return;
        if (self.samplePressure() == .hard) return error.CompactBudgetExceeded;
        self.lock();
        defer self.unlock();
        const pressure_bytes = @max(self.committedTotalLocked(), self.observed_footprint_bytes);
        const projected = pressure_bytes +| bytes;
        if (projected > self.softLimitBytes()) {
            self.growth_limit_rejections +|= 1;
            return error.CompactBudgetExceeded;
        }
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
            .growth_limit_rejections = self.growth_limit_rejections,
            .sampler_required = self.sampler_required,
            .sampler_available = self.sampler_available,
            .sampler_unavailable_rejections = self.sampler_unavailable_rejections,
        };
    }
};

/// Prompt-prefix-cache resource observer endpoint. `context` is the owning
/// model instance's ledger; the cache reports absolute retained bytes and
/// this converts them to release-then-reserve against `.prompt_cache_kv`.
///
/// The observer is transactional: the prompt cache asks before retaining new
/// blocks. A rejected growth leaves `current` and the ledger unchanged, so
/// the cache can evict or skip insertion without ever holding unaccounted KV.
///
/// Exactly one cache feeds a given ledger's `.prompt_cache_kv` category (the
/// per-model PromptPrefixCache), so the absolute-value rewrite is race-free.
pub fn promptCacheObserverUpdate(context: *anyopaque, current: *u64, next: u64) bool {
    const ledger: *ResidentBudgetLedger = @ptrCast(@alignCast(context));
    if (ledger.replaceReservation(.prompt_cache_kv, current.*, next)) {
        current.* = next;
        return true;
    } else |_| {
        if (!builtin.is_test) {
            std.log.warn(
                "prompt cache growth rejected by compact residency budget: requested={d} ceiling={d}",
                .{ next, ledger.ceiling_bytes },
            );
        }
        return false;
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
    try std.testing.expect(promptCacheObserverUpdate(&ledger, &accounted, 300));
    try std.testing.expectEqual(@as(u64, 300), accounted);
    var snap = ledger.snapshot();
    try std.testing.expectEqual(@as(u64, 300), snap.committed_by_category[@intFromEnum(BudgetCategory.prompt_cache_kv)]);
    try std.testing.expectEqual(@as(u64, 700), snap.committed_total_bytes);

    // Shrink: 300 -> 100 releases the difference.
    try std.testing.expect(promptCacheObserverUpdate(&ledger, &accounted, 100));
    try std.testing.expectEqual(@as(u64, 100), accounted);
    snap = ledger.snapshot();
    try std.testing.expectEqual(@as(u64, 100), snap.committed_by_category[@intFromEnum(BudgetCategory.prompt_cache_kv)]);

    // Release-on-zero: the category empties without touching other classes.
    try std.testing.expect(promptCacheObserverUpdate(&ledger, &accounted, 0));
    try std.testing.expectEqual(@as(u64, 0), accounted);
    snap = ledger.snapshot();
    try std.testing.expectEqual(@as(u64, 0), snap.committed_by_category[@intFromEnum(BudgetCategory.prompt_cache_kv)]);
    try std.testing.expectEqual(@as(u64, 400), snap.committed_total_bytes);
}

test "prompt cache observer rejects growth without changing the live reservation" {
    var ledger = ResidentBudgetLedger.init(1000, 0);
    try ledger.reserve(.fixed_model, 900);
    var accounted: u64 = 0;

    // Fits: 0 -> 80.
    try std.testing.expect(promptCacheObserverUpdate(&ledger, &accounted, 80));
    try std.testing.expectEqual(@as(u64, 80), accounted);

    // Would cross the ceiling: the old reservation stays intact and the
    // caller receives a rejection before retaining additional KV.
    try std.testing.expect(!promptCacheObserverUpdate(&ledger, &accounted, 200));
    try std.testing.expectEqual(@as(u64, 80), accounted);
    var snap = ledger.snapshot();
    try std.testing.expectEqual(@as(u64, 80), snap.committed_by_category[@intFromEnum(BudgetCategory.prompt_cache_kv)]);
    try std.testing.expectEqual(@as(u64, 980), snap.committed_total_bytes);
    try std.testing.expect(snap.hard_limit_rejections >= 1);

    // The cache evicting back under budget recommits the smaller reservation.
    try std.testing.expect(promptCacheObserverUpdate(&ledger, &accounted, 50));
    try std.testing.expectEqual(@as(u64, 50), accounted);
    snap = ledger.snapshot();
    try std.testing.expectEqual(@as(u64, 50), snap.committed_by_category[@intFromEnum(BudgetCategory.prompt_cache_kv)]);
}

test "budget ledger bounded growth preserves the process safety reserve" {
    var ledger = ResidentBudgetLedger.init(1000, 100);
    ledger.footprint_override = 800;
    try ledger.admitGrowthWithinReserve(100);
    try std.testing.expectError(
        error.CompactBudgetExceeded,
        ledger.admitGrowthWithinReserve(101),
    );
    try std.testing.expectEqual(@as(u64, 1), ledger.snapshot().growth_limit_rejections);
}

test "budget ledger retains the last sample and required sampling fails closed" {
    var ledger = ResidentBudgetLedger.init(1000, 100);
    ledger.footprint_override = 950;
    try std.testing.expectEqual(BudgetPressure.soft, ledger.samplePressure());

    ledger.footprint_override = null;
    ledger.footprint_unavailable_override = true;
    // A non-enforcing ledger uses the last valid observation rather than
    // silently treating a failed sample as zero.
    try std.testing.expectEqual(BudgetPressure.soft, ledger.samplePressure());
    try std.testing.expectEqual(@as(u64, 950), ledger.snapshot().observed_footprint_bytes);

    ledger.setProcessSamplerRequired(true);
    try std.testing.expectEqual(BudgetPressure.hard, ledger.samplePressure());
    const snap = ledger.snapshot();
    try std.testing.expect(snap.sampler_required);
    try std.testing.expect(!snap.sampler_available);
    try std.testing.expectEqual(@as(u64, 1), snap.sampler_unavailable_rejections);
}

test "budget ledger live darwin sampler reports this process when available" {
    var ledger = ResidentBudgetLedger.init(std.math.maxInt(u64), 0);
    _ = ledger.samplePressure();
    const stats = platform.process_memory.pressureSnapshot();
    if (stats.available and stats.footprint_bytes != 0) {
        try std.testing.expect(ledger.snapshot().observed_footprint_bytes != 0);
    }
}
