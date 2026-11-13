const std = @import("std");

pub const policy = "private_train_challenge_hybrid_resident_native_v1|host_head_workspace=2GiB|transient_host=3840MiB|process_guard=8GiB|fit_workers=1|max_p95_attempts=3|default=off|qualification=none";

pub const head_workspace_bytes: usize = 2 * 1024 * 1024 * 1024;
pub const maximum_transient_host_bytes: usize = 3840 * 1024 * 1024;
pub const process_guard_bytes: usize = 8 * 1024 * 1024 * 1024;
pub const maximum_p95_attempts: usize = 3;

pub fn digest() [32]u8 {
    var result: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(policy, &result, .{});
    return result;
}

pub const Admission = struct {
    retained_host_source_bytes: usize,
    retained_native_device_bytes: usize,
    dense_backing_bytes: usize,
    request_owner_bytes: usize,
    maximum_head_clone_bytes: usize,
    fitter_workspace_bytes: usize,

    fn add(a: usize, b: usize) !usize {
        return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
    }

    pub fn hostPeak(self: @This()) !usize {
        var value = try add(self.retained_host_source_bytes, self.dense_backing_bytes);
        value = try add(value, self.request_owner_bytes);
        value = try add(value, self.maximum_head_clone_bytes);
        return add(value, self.fitter_workspace_bytes);
    }

    pub fn processPeak(self: @This()) !usize {
        return add(try self.hostPeak(), self.retained_native_device_bytes);
    }

    pub fn validate(self: @This()) !void {
        if (self.fitter_workspace_bytes != head_workspace_bytes) return error.InvalidFitConfig;
        if (try self.hostPeak() > maximum_transient_host_bytes) return error.CompactionWorkspaceLimit;
        if (try self.processPeak() > process_guard_bytes) return error.CompactionWorkspaceLimit;
    }

    /// The resident profile never admits a second 2 GiB fitter. A caller may
    /// use this pure result before constructing any backend or worker owner.
    pub fn fitWorkers(self: @This()) !usize {
        try self.validate();
        return 1;
    }
};

test "resident native hybrid profile is explicit default-off identity" {
    try std.testing.expect(!std.mem.allEqual(u8, &digest(), 0));
    try std.testing.expectEqual(@as(usize, 2 * 1024 * 1024 * 1024), head_workspace_bytes);
    try std.testing.expectEqual(@as(usize, 3), maximum_p95_attempts);
}

test "resident native hybrid admission separates host and device residency" {
    const admitted: Admission = .{
        .retained_host_source_bytes = 305 * 1024 * 1024,
        .retained_native_device_bytes = 2 * 1024 * 1024 * 1024,
        .dense_backing_bytes = 128 * 1024 * 1024,
        .request_owner_bytes = 64 * 1024 * 1024,
        .maximum_head_clone_bytes = 64 * 1024 * 1024,
        .fitter_workspace_bytes = head_workspace_bytes,
    };
    try admitted.validate();
    try std.testing.expectEqual(@as(usize, 1), try admitted.fitWorkers());
    var host_over = admitted;
    host_over.request_owner_bytes = maximum_transient_host_bytes;
    try std.testing.expectError(error.CompactionWorkspaceLimit, host_over.validate());
    var process_over = admitted;
    process_over.retained_native_device_bytes = process_guard_bytes;
    try std.testing.expectError(error.CompactionWorkspaceLimit, process_over.validate());
}
