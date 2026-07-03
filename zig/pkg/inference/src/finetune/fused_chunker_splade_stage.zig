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

//! Late SPLADE training-stage transition logic for the fused chunker trainer.
//!
//! Port of gopeft Phase 32 "staged SPLADE remediation" semantics
//! (e2e/finetune/fused_training.go + fused_training_stage.go):
//!
//!   - Joint SPLADE training degraded boundary F1 (0.779 -> 0.703) because the
//!     shared encoder was pulled between boundary and SPLADE objectives.
//!   - Fix: a late head-only SPLADE stage. At `splade_focus_epoch` the trainer
//!     (a) optionally restores the best boundary checkpoint, (b) optionally
//!     resets optimizer moments, and (c) for the remaining epochs updates ONLY
//!     the SPLADE head; the boundary head, dense/contrastive path, and encoder
//!     LoRA adapters are frozen (no gradients are even computed for them).
//!
//! This module is dependency-free (std only) so its tests run in the isolated
//! `test-fused-chunker-splade-stage` build target. The trainer CLI
//! (train/train_fused_chunker.zig) routes every update decision through the
//! functions below, which is what makes boundary F1 structurally incapable of
//! changing during the head-only stage.
//!
//! Go-parity notes:
//!   - `isHeadOnlyPhase`     mirrors FusedTrainingConfig.IsSpladeHeadOnlyPhase.
//!   - `isStageEntryEpoch`   mirrors the maybeEnterSpladeStage gate
//!                           (head-only phase AND epoch == SpladeFocusEpochs).
//!   - `updateAllowed`       mirrors AllowTrainableUpdateForEpoch. The Zig
//!                           SPLADE head is a single projection W whose
//!                           gradient never touches the encoder, so the Go
//!                           "separate SPLADE adapter" isolation holds by
//!                           construction; the flag is still plumbed for CLI
//!                           and manifest parity (see StageConfig).
//!   - `lambdasForEpoch`     mirrors GetSpladeLambdas (half λ_splade ramp-in on
//!                           the first active epoch, quadratic FLOPS ramp).

const std = @import("std");

/// SPLADE stage mode, mirroring the Go --splade-training-mode flag.
///   joint:     SPLADE trains alongside boundary/dense from splade_focus_epoch
///              (pre-Phase-32 behavior).
///   head_only: epochs >= splade_focus_epoch train ONLY the SPLADE head; the
///              boundary/dense paths are frozen (Phase 32 production recipe).
pub const SpladeTrainingMode = enum {
    joint,
    head_only,

    /// Parse a CLI value. Accepts the Go spellings "joint" and "head-only"
    /// (plus "head_only" for convenience).
    pub fn parse(value: []const u8) ?SpladeTrainingMode {
        if (std.mem.eql(u8, value, "joint")) return .joint;
        if (std.mem.eql(u8, value, "head-only")) return .head_only;
        if (std.mem.eql(u8, value, "head_only")) return .head_only;
        return null;
    }

    /// Canonical (Go flag) spelling.
    pub fn name(self: SpladeTrainingMode) []const u8 {
        return switch (self) {
            .joint => "joint",
            .head_only => "head-only",
        };
    }
};

/// Parameter groups the trainer can update. Mirrors the name-based
/// classification in Go's AllowTrainableUpdateForEpoch.
pub const ParamGroup = enum {
    /// Encoder LoRA adapters on the primary chunking/dense path.
    encoder_lora,
    /// Boundary detection head (w1/b1/w2/b2).
    boundary_head,
    /// Dense embedding / contrastive path (late-chunking pool feeds encoder
    /// LoRA gradients; grouped separately for clarity).
    dense_embed,
    /// SPLADE projection weight W.
    splade_head,
};

/// Static stage configuration derived from CLI flags.
pub const StageConfig = struct {
    /// --splade
    enable_splade: bool = false,
    /// --splade-training-mode (Go default: head-only)
    mode: SpladeTrainingMode = .head_only,
    /// --splade-focus-epoch: epoch when SPLADE activates. With head-only mode
    /// this is also the first epoch of the late SPLADE-only stage.
    splade_focus_epoch: u32 = 4,
    /// --restore-best-for-splade-stage (Go default: true)
    restore_best_for_splade_stage: bool = true,
    /// --reset-optimizer-on-splade-stage (Go default: true)
    reset_optimizer_on_splade_stage: bool = true,
    /// --separate-splade-adapter (Go default: true). In Go this creates a
    /// second LoRA branch for the SPLADE encoder path. The Zig SPLADE head
    /// only ever receives gradients for its projection W (never the encoder),
    /// so both settings are gradient-isolated here; the flag is recorded for
    /// config/manifest parity and future adapter-branch work.
    separate_splade_adapter: bool = true,
};

/// Whether `epoch` (0-indexed) is inside the late SPLADE head-only stage.
/// Mirrors Go IsSpladeHeadOnlyPhase.
pub fn isHeadOnlyPhase(cfg: StageConfig, epoch: usize) bool {
    return cfg.enable_splade and cfg.mode == .head_only and
        epoch >= @as(usize, cfg.splade_focus_epoch);
}

/// Whether the stage transition (restore-best / optimizer reset) fires at
/// `epoch`. Mirrors Go maybeEnterSpladeStage's gate: the transition happens
/// exactly once, at the first head-only epoch. When training resumes into a
/// later epoch the transition is intentionally skipped (it already ran in the
/// original process), matching Go.
pub fn isStageEntryEpoch(cfg: StageConfig, epoch: usize) bool {
    return isHeadOnlyPhase(cfg, epoch) and epoch == @as(usize, cfg.splade_focus_epoch);
}

/// Whether the best boundary checkpoint should be restored at stage entry.
/// `best_epoch` is 1-indexed (0 = no best checkpoint recorded yet), matching
/// both the Go trainer's bestEpoch field and the Zig best_val_epoch variable.
pub fn shouldRestoreBest(cfg: StageConfig, epoch: usize, best_epoch: u32) bool {
    return isStageEntryEpoch(cfg, epoch) and
        cfg.restore_best_for_splade_stage and best_epoch > 0;
}

/// Whether optimizer moments should be reset at stage entry.
pub fn shouldResetOptimizer(cfg: StageConfig, epoch: usize) bool {
    return isStageEntryEpoch(cfg, epoch) and cfg.reset_optimizer_on_splade_stage;
}

/// Which parameter groups may receive optimizer updates in `epoch`.
/// Mirrors Go AllowTrainableUpdateForEpoch: during the head-only phase only
/// the SPLADE head updates; otherwise everything trains.
pub fn updateAllowed(cfg: StageConfig, group: ParamGroup, epoch: usize) bool {
    if (isHeadOnlyPhase(cfg, epoch)) {
        return group == .splade_head;
    }
    return true;
}

/// Whether SPLADE loss/updates are active at all in `epoch` (either mode).
pub fn spladeActive(cfg: StageConfig, epoch: usize) bool {
    return cfg.enable_splade and epoch >= @as(usize, cfg.splade_focus_epoch);
}

pub const SpladeLambdas = struct {
    splade: f32 = 0.0,
    flops: f32 = 0.0,
};

/// Per-epoch SPLADE loss weights. Mirrors Go GetSpladeLambdas:
///   - zero before splade_focus_epoch,
///   - half λ_splade on the first active epoch (gentle ramp-in),
///   - λ_flops ramps quadratically over the remaining active epochs
///     (so the first active epoch has zero FLOPS pressure).
pub fn lambdasForEpoch(
    cfg: StageConfig,
    epoch: usize,
    num_epochs: usize,
    lambda_splade: f32,
    lambda_flops: f32,
) SpladeLambdas {
    if (!cfg.enable_splade) return .{};
    const focus: usize = @intCast(cfg.splade_focus_epoch);
    if (epoch < focus) return .{};

    const splade_epochs = epoch - focus;
    var ls = lambda_splade;
    if (splade_epochs == 0) ls = lambda_splade * 0.5;

    var total_active_epochs: usize = 1;
    if (num_epochs > focus) total_active_epochs = num_epochs - focus;

    var progress = @as(f32, @floatFromInt(splade_epochs)) /
        @as(f32, @floatFromInt(total_active_epochs));
    if (progress > 1.0) progress = 1.0;
    const flops_ramp = progress * progress;
    return .{ .splade = ls, .flops = lambda_flops * flops_ramp };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "mode parsing accepts Go spellings and rejects unknown values" {
    try testing.expectEqual(SpladeTrainingMode.joint, SpladeTrainingMode.parse("joint").?);
    try testing.expectEqual(SpladeTrainingMode.head_only, SpladeTrainingMode.parse("head-only").?);
    try testing.expectEqual(SpladeTrainingMode.head_only, SpladeTrainingMode.parse("head_only").?);
    try testing.expect(SpladeTrainingMode.parse("headonly") == null);
    try testing.expect(SpladeTrainingMode.parse("") == null);
    try testing.expectEqualStrings("head-only", SpladeTrainingMode.head_only.name());
    try testing.expectEqualStrings("joint", SpladeTrainingMode.joint.name());
}

test "head-only phase requires splade enabled, head-only mode, and epoch >= focus" {
    const cfg = StageConfig{ .enable_splade = true, .mode = .head_only, .splade_focus_epoch = 4 };
    try testing.expect(!isHeadOnlyPhase(cfg, 0));
    try testing.expect(!isHeadOnlyPhase(cfg, 3));
    try testing.expect(isHeadOnlyPhase(cfg, 4));
    try testing.expect(isHeadOnlyPhase(cfg, 7));

    var joint = cfg;
    joint.mode = .joint;
    try testing.expect(!isHeadOnlyPhase(joint, 4));
    try testing.expect(!isHeadOnlyPhase(joint, 7));

    var disabled = cfg;
    disabled.enable_splade = false;
    try testing.expect(!isHeadOnlyPhase(disabled, 4));
}

test "stage entry fires exactly at the focus epoch" {
    const cfg = StageConfig{ .enable_splade = true, .mode = .head_only, .splade_focus_epoch = 4 };
    try testing.expect(!isStageEntryEpoch(cfg, 3));
    try testing.expect(isStageEntryEpoch(cfg, 4));
    try testing.expect(!isStageEntryEpoch(cfg, 5)); // resume past focus: no re-entry
    var joint = cfg;
    joint.mode = .joint;
    try testing.expect(!isStageEntryEpoch(joint, 4));
}

test "stage entry at focus epoch zero (head-only from the start)" {
    const cfg = StageConfig{ .enable_splade = true, .mode = .head_only, .splade_focus_epoch = 0 };
    try testing.expect(isStageEntryEpoch(cfg, 0));
    try testing.expect(isHeadOnlyPhase(cfg, 0));
    try testing.expect(!isStageEntryEpoch(cfg, 1));
}

test "restore-best requires flag, stage entry, and a recorded best checkpoint" {
    const cfg = StageConfig{ .enable_splade = true, .mode = .head_only, .splade_focus_epoch = 4 };
    try testing.expect(shouldRestoreBest(cfg, 4, 3)); // best from epoch 3
    try testing.expect(!shouldRestoreBest(cfg, 4, 0)); // no best checkpoint yet
    try testing.expect(!shouldRestoreBest(cfg, 5, 3)); // not the entry epoch
    var no_restore = cfg;
    no_restore.restore_best_for_splade_stage = false;
    try testing.expect(!shouldRestoreBest(no_restore, 4, 3));
}

test "optimizer reset requires flag and stage entry" {
    const cfg = StageConfig{ .enable_splade = true, .mode = .head_only, .splade_focus_epoch = 4 };
    try testing.expect(shouldResetOptimizer(cfg, 4));
    try testing.expect(!shouldResetOptimizer(cfg, 3));
    try testing.expect(!shouldResetOptimizer(cfg, 5));
    var no_reset = cfg;
    no_reset.reset_optimizer_on_splade_stage = false;
    try testing.expect(!shouldResetOptimizer(no_reset, 4));
}

test "head-only phase allows updates only for the SPLADE head (Go AllowTrainableUpdateForEpoch parity)" {
    const cfg = StageConfig{ .enable_splade = true, .mode = .head_only, .splade_focus_epoch = 4 };

    // Before the stage: everything trains.
    inline for (.{ .encoder_lora, .boundary_head, .dense_embed, .splade_head }) |group| {
        try testing.expect(updateAllowed(cfg, group, 3));
    }

    // In the stage: only the SPLADE head.
    try testing.expect(!updateAllowed(cfg, .encoder_lora, 4));
    try testing.expect(!updateAllowed(cfg, .boundary_head, 4));
    try testing.expect(!updateAllowed(cfg, .dense_embed, 4));
    try testing.expect(updateAllowed(cfg, .splade_head, 4));
    try testing.expect(!updateAllowed(cfg, .boundary_head, 9));

    // Joint mode never freezes anything.
    var joint = cfg;
    joint.mode = .joint;
    inline for (.{ .encoder_lora, .boundary_head, .dense_embed, .splade_head }) |group| {
        try testing.expect(updateAllowed(joint, group, 6));
    }
}

test "splade activation epoch gating applies to both modes" {
    var cfg = StageConfig{ .enable_splade = true, .mode = .head_only, .splade_focus_epoch = 4 };
    try testing.expect(!spladeActive(cfg, 3));
    try testing.expect(spladeActive(cfg, 4));
    cfg.mode = .joint;
    try testing.expect(!spladeActive(cfg, 3));
    try testing.expect(spladeActive(cfg, 4));
    cfg.enable_splade = false;
    try testing.expect(!spladeActive(cfg, 4));
}

test "lambda schedule mirrors Go GetSpladeLambdas" {
    const cfg = StageConfig{ .enable_splade = true, .mode = .head_only, .splade_focus_epoch = 4 };
    const num_epochs: usize = 8; // 4 active SPLADE epochs
    const ls: f32 = 0.15;
    const lf: f32 = 3e-5;

    // Inactive before the focus epoch.
    var l = lambdasForEpoch(cfg, 3, num_epochs, ls, lf);
    try testing.expectEqual(@as(f32, 0.0), l.splade);
    try testing.expectEqual(@as(f32, 0.0), l.flops);

    // First active epoch: half lambda_splade, zero FLOPS (quadratic ramp at 0).
    l = lambdasForEpoch(cfg, 4, num_epochs, ls, lf);
    try testing.expectApproxEqAbs(@as(f32, 0.075), l.splade, 1e-7);
    try testing.expectEqual(@as(f32, 0.0), l.flops);

    // Second active epoch: full lambda_splade, FLOPS at (1/4)^2 of target.
    l = lambdasForEpoch(cfg, 5, num_epochs, ls, lf);
    try testing.expectApproxEqAbs(@as(f32, 0.15), l.splade, 1e-7);
    try testing.expectApproxEqAbs(lf * 0.0625, l.flops, 1e-12);

    // Last epoch: FLOPS at (3/4)^2 of target.
    l = lambdasForEpoch(cfg, 7, num_epochs, ls, lf);
    try testing.expectApproxEqAbs(lf * 0.5625, l.flops, 1e-12);

    // Degenerate schedule (focus >= num_epochs): Go clamps totalActive to 1.
    const late = StageConfig{ .enable_splade = true, .mode = .head_only, .splade_focus_epoch = 6 };
    l = lambdasForEpoch(late, 7, 6, ls, lf);
    try testing.expectApproxEqAbs(@as(f32, 0.15), l.splade, 1e-7);
    try testing.expectApproxEqAbs(lf, l.flops, 1e-12); // progress clamped to 1

    // Disabled: always zero.
    const off = StageConfig{ .enable_splade = false };
    l = lambdasForEpoch(off, 5, num_epochs, ls, lf);
    try testing.expectEqual(@as(f32, 0.0), l.splade);
}

test "head-only step leaves frozen parameter groups bit-identical (SPLADE-head-gradient-only invariant)" {
    // Simulate one optimizer step of the head-only stage on a tiny synthetic
    // model, applying updates only where updateAllowed permits — exactly how
    // the trainer gates its three update sites (boundary head trainStep,
    // encoder LoRA update, SPLADE AdamW update).
    const cfg = StageConfig{ .enable_splade = true, .mode = .head_only, .splade_focus_epoch = 1 };
    const epoch: usize = 1;

    var boundary_w = [_]f32{ 0.1, -0.2, 0.3, 0.4 };
    var lora_a = [_]f32{ 0.01, 0.02, -0.03, 0.05 };
    var splade_w = [_]f32{ 0.5, -0.5, 0.25, -0.25 };
    const boundary_before = boundary_w;
    const lora_before = lora_a;
    const splade_before = splade_w;

    const grad = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const lr: f32 = 0.1;

    const groups = [_]struct { group: ParamGroup, params: []f32 }{
        .{ .group = .boundary_head, .params = boundary_w[0..] },
        .{ .group = .encoder_lora, .params = lora_a[0..] },
        .{ .group = .splade_head, .params = splade_w[0..] },
    };
    for (groups) |entry| {
        if (!updateAllowed(cfg, entry.group, epoch)) continue;
        for (entry.params, grad) |*w, g| w.* -= lr * g;
    }

    // Frozen groups: bit-identical.
    try testing.expectEqualSlices(f32, boundary_before[0..], boundary_w[0..]);
    try testing.expectEqualSlices(f32, lora_before[0..], lora_a[0..]);
    // SPLADE head: updated.
    for (splade_before, splade_w) |before, after| {
        try testing.expectApproxEqAbs(before - lr, after, 1e-7);
    }
}
