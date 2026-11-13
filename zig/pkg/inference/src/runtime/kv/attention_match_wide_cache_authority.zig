const std = @import("std");
const solver = @import("attention_match.zig");
const training_source = @import("attention_match_joint_training_source.zig");

pub const policy_sha256: [32]u8 = blk: {
    @setEvalBranchQuota(10_000);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("production768_train_challenge_wide_cache_bounded_page_v2|exact_f32_pages|fp64_y|bounded_fp64_tsqr_values|critical_fixed_term_charged|accelerated_fp64_validation|primary_probe_exact", &out, .{});
    break :blk out;
};

pub const FlatAdmission = struct {
    owner_identity: u64,
    probability_logical_bytes: usize,
    output_logical_bytes: usize,
    available_physical_bytes: usize,
    required_physical_bytes: usize,
    fits: bool,
};

pub const Authority = struct {
    context: ?*anyopaque,
    owner_identity: u64,
    policy_sha256: [32]u8,
    flat_admission_fn: *const fn (?*anyopaque, usize, usize) anyerror!FlatAdmission,
    allocation_refusal: ?FlatAdmission = null,

    pub const Retry = struct { refused: FlatAdmission, refreshed: FlatAdmission };

    pub fn flatAdmission(self: @This(), probability_bytes: usize, output_bytes: usize, control: solver.Control) !FlatAdmission {
        if (self.owner_identity == 0 or !std.mem.eql(u8, &self.policy_sha256, &policy_sha256) or probability_bytes == 0 or output_bytes == 0)
            return error.InvalidFitConfig;
        try control.check();
        const value = try self.flat_admission_fn(self.context, probability_bytes, output_bytes);
        try control.check();
        if (value.owner_identity != self.owner_identity or value.probability_logical_bytes != probability_bytes or
            value.output_logical_bytes != output_bytes or value.required_physical_bytes == 0)
            return error.InvalidFitConfig;
        return value;
    }

    fn admitsAtAllocation(raw: ?*anyopaque, probability_bytes: usize, output_bytes: usize, control: solver.Control) !bool {
        const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidFitConfig));
        self.allocation_refusal = null;
        const decision = try self.flatAdmission(probability_bytes, output_bytes, control);
        if (!decision.fits) self.allocation_refusal = decision;
        return decision.fits;
    }

    pub fn allocationAuthority(self: *@This()) training_source.AllocationAuthority {
        return .{ .context = self, .admits_fn = admitsAtAllocation };
    }

    pub fn consumeBoundedRetry(self: *@This(), failure: anyerror, probability_bytes: usize, output_bytes: usize, control: solver.Control) !?Retry {
        const refusal = self.allocation_refusal;
        self.allocation_refusal = null;
        if (failure != error.V18TrainingCacheAuthorityChanged) return null;
        const refused = refusal orelse return null;
        if (refused.fits or refused.owner_identity != self.owner_identity or refused.probability_logical_bytes != probability_bytes or
            refused.output_logical_bytes != output_bytes or refused.required_physical_bytes == 0)
            return error.InvalidFitConfig;
        return .{ .refused = refused, .refreshed = try self.flatAdmission(probability_bytes, output_bytes, control) };
    }
};
