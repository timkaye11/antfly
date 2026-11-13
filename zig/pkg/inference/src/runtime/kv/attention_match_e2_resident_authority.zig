//! Default-off E2/8128 resident admission authority. No live route consumes it.
const std = @import("std");
const shape = @import("attention_match_e2_flat_shape_authority.zig");
const pressure = @import("attention_match_e2_flat8128_pressure_profile.zig");

pub const schema = "antfly.context.resident-native-hybrid-e2-8128.authority.v1";
pub const policy = "e2_resident_authority_v2|shape=e2flatv4|pressure=e2pressurev3|consumer_batches56|per_owner8,8,40|workspace2GiB|host3840MiB|process8GiB|workers1|activation=disabled|qualification=none";
pub const guarded_trial_policy = "e2_resident_authority_v3|shape=e2flatv4|pressure=e2pressurev3|consumer_batches56|per_owner8,8,40|workspace2GiB|host3840MiB|process8GiB|workers1|activation=guarded_development_trial|serving=false|release=false|registry=false|qualification=none";
pub const formal_fit_policy = "e2_resident_authority_v4|shape=e2flatv5|source_rows=7936,32512|pressure=separate_formal_owner_inventory|required|consumer_batches56|per_owner8,8,40|workspace2GiB|host3840MiB|process8GiB|workers1|activation=guarded_development_trial|serving=false|release=false|registry=false|qualification=none";
pub const head_workspace_bytes: usize = 2 * 1024 * 1024 * 1024;
pub const maximum_transient_host_bytes: usize = 3840 * 1024 * 1024;
pub const process_guard_bytes: usize = 8 * 1024 * 1024 * 1024;
/// Source1387 carries the typed E2 authority and complete capture/oracle
/// plumbing, but the production 2 GiB phased peak has not yet been proven at
/// the owner-14 boundary. Keep the live caller fail-closed until that exact
/// admission proof is part of this authority's digest.
pub const activation_enabled = false;

pub fn guardedTrialDigest() [32]u8 {
    @setEvalBranchQuota(100_000);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(guarded_trial_policy, &out, .{});
    return out;
}

pub fn formalFitDigest() [32]u8 {
    @setEvalBranchQuota(100_000);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(formal_fit_policy, &out, .{});
    return out;
}

pub fn digest() [32]u8 {
    @setEvalBranchQuota(100_000);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(policy, &out, .{});
    return out;
}

pub const Admission = struct {
    shape_binding: pressure.Binding,
    retained_host_bytes: usize,
    retained_device_bytes: usize,
    request_owner_bytes: usize,
    maximum_head_clone_bytes: usize,
    fitter_workspace_bytes: usize = head_workspace_bytes,

    fn add(a: usize, b: usize) !usize {
        return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
    }
    pub fn validate(self: @This()) !void {
        _ = try shape.validate(self.shape_binding.shape);
        if (!std.crypto.timing_safe.eql([32]u8, (try shape.validate(self.shape_binding.shape)).digest, self.shape_binding.expected_shape_sha256) or
            self.fitter_workspace_bytes != head_workspace_bytes) return error.InvalidFitConfig;
        var host = try add(self.retained_host_bytes, self.request_owner_bytes);
        host = try add(host, self.maximum_head_clone_bytes);
        host = try add(host, self.fitter_workspace_bytes);
        if (host > maximum_transient_host_bytes or try add(host, self.retained_device_bytes) > process_guard_bytes)
            return error.CompactionWorkspaceLimit;
    }
};

fn binding(byte: u8) pressure.Binding {
    const s: shape.Shape = .{ .authority_sha256 = @splat(byte) };
    return .{ .shape = s, .expected_shape_sha256 = (shape.validate(s) catch unreachable).digest };
}

test "E2 resident authority binds distinct shape census and conservative caps" {
    const admitted: Admission = .{ .shape_binding = binding(1), .retained_host_bytes = 256 * 1024 * 1024, .retained_device_bytes = 128 * 1024 * 1024, .request_owner_bytes = 128 * 1024 * 1024, .maximum_head_clone_bytes = 128 * 1024 * 1024 };
    try admitted.validate();
    try std.testing.expectEqual(@as(usize, 56), admitted.shape_binding.shape.batch_count);
    try std.testing.expect(!std.mem.eql(u8, schema, "antfly.context.resident-native-hybrid.compatibility-profile.v1"));
}

test "E2 resident authority rejects shape identity cap and overflow mutations" {
    var admitted: Admission = .{ .shape_binding = binding(1), .retained_host_bytes = 1, .retained_device_bytes = 1, .request_owner_bytes = 1, .maximum_head_clone_bytes = 1 };
    admitted.shape_binding.expected_shape_sha256 = @splat(9);
    try std.testing.expectError(error.InvalidFitConfig, admitted.validate());
    admitted.shape_binding = binding(1);
    admitted.request_owner_bytes = maximum_transient_host_bytes;
    try std.testing.expectError(error.CompactionWorkspaceLimit, admitted.validate());
    admitted.request_owner_bytes = std.math.maxInt(usize);
    try std.testing.expectError(error.CompactionWorkspaceLimit, admitted.validate());
}
