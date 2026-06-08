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
const driver_mod = @import("driver.zig");

const CudaDriver = driver_mod.CudaDriver;
const CUcontext = driver_mod.CUcontext;
const CUdevice = driver_mod.CUdevice;
const CUstream = driver_mod.CUstream;

pub const DeviceInfo = struct {
    driver_version: i32 = 0,
    device_count: i32 = 0,
    selected_device: i32 = 0,
    name: [256]u8 = .{0} ** 256,
    name_len: usize = 0,
    compute_major: i32 = 0,
    compute_minor: i32 = 0,

    pub fn nameSlice(self: *const DeviceInfo) []const u8 {
        return self.name[0..self.name_len];
    }
};

pub const RuntimeStats = struct {
    kernel_launches: usize = 0,
    stream_syncs: usize = 0,
};

pub const CudaContext = struct {
    driver: CudaDriver,
    device: CUdevice,
    ctx: CUcontext,
    stream: CUstream,
    info: DeviceInfo,
    stats: RuntimeStats = .{},

    pub fn initDefault() driver_mod.Error!CudaContext {
        var driver = try CudaDriver.open();
        errdefer driver.deinit();

        checkStep(&driver, "cuInit", driver.fns.cuInit(0)) catch |err| return err;

        var info = DeviceInfo{};
        checkStep(&driver, "cuDriverGetVersion", driver.fns.cuDriverGetVersion(&info.driver_version)) catch |err| return err;
        checkStep(&driver, "cuDeviceGetCount", driver.fns.cuDeviceGetCount(&info.device_count)) catch |err| return err;
        if (info.device_count <= 0) return error.NoCudaDevices;

        var device: CUdevice = 0;
        checkStep(&driver, "cuDeviceGet", driver.fns.cuDeviceGet(&device, 0)) catch |err| return err;
        info.selected_device = device;

        var name_buf: [256]u8 = .{0} ** 256;
        checkStep(&driver, "cuDeviceGetName", driver.fns.cuDeviceGetName(&name_buf, name_buf.len, device)) catch |err| return err;
        info.name = name_buf;
        info.name_len = std.mem.indexOfScalar(u8, &info.name, 0) orelse info.name.len;

        checkStep(&driver, "cuDeviceComputeCapability", driver.fns.cuDeviceComputeCapability(&info.compute_major, &info.compute_minor, device)) catch |err| return err;

        var ctx: CUcontext = null;
        checkStep(&driver, "cuDevicePrimaryCtxRetain", driver.fns.cuDevicePrimaryCtxRetain(&ctx, device)) catch |err| return err;
        errdefer _ = driver.fns.cuDevicePrimaryCtxRelease(device);
        checkStep(&driver, "cuCtxSetCurrent", driver.fns.cuCtxSetCurrent(ctx)) catch |err| return err;

        var stream: CUstream = null;
        checkStep(&driver, "cuStreamCreate", driver.fns.cuStreamCreate(&stream, 0)) catch |err| return err;
        errdefer _ = driver.fns.cuStreamDestroy(stream);

        return .{
            .driver = driver,
            .device = device,
            .ctx = ctx,
            .stream = stream,
            .info = info,
        };
    }

    pub fn deinit(self: *CudaContext) void {
        if (self.ctx != null) {
            _ = self.driver.fns.cuCtxSetCurrent(self.ctx);
        }
        if (self.stream != null) {
            _ = self.driver.fns.cuStreamDestroy(self.stream);
            self.stream = null;
        }
        if (self.ctx != null) {
            _ = self.driver.fns.cuCtxSetCurrent(null);
            _ = self.driver.fns.cuDevicePrimaryCtxRelease(self.device);
            self.ctx = null;
        }
        self.driver.deinit();
    }

    pub fn makeCurrent(self: *CudaContext) driver_mod.Error!void {
        if (self.ctx == null) return error.InvalidCudaState;
        try self.driver.check(self.driver.fns.cuCtxSetCurrent(self.ctx));
    }

    pub fn synchronize(self: *CudaContext) driver_mod.Error!void {
        try self.makeCurrent();
        try self.driver.check(self.driver.fns.cuStreamSynchronize(self.stream));
        self.stats.stream_syncs += 1;
    }

    pub fn noteKernelLaunch(self: *CudaContext) void {
        self.stats.kernel_launches += 1;
    }
};

fn checkStep(driver: *const CudaDriver, step: []const u8, result: driver_mod.CUresult) driver_mod.Error!void {
    if (result != driver_mod.CUDA_SUCCESS) {
        std.log.err("CUDA context init failed at {s}: {s} ({s})", .{ step, driver.errorName(result), driver.errorString(result) });
        return error.CudaDriverError;
    }
}

pub fn probeDefault() driver_mod.Error!DeviceInfo {
    var ctx = try CudaContext.initDefault();
    defer ctx.deinit();
    return ctx.info;
}
