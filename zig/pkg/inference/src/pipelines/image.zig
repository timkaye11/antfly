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

// Image preprocessing for vision models (CLIP, Florence2).
//
// Decodes JPEG/PNG/GIF through the shared antfly image layer, then routes
// decoded pixels through the shared resize/normalize/CHW preprocessing path.

const std = @import("std");
const antfly_image = @import("antfly_image");
const shared = antfly_image.processing;

/// Standard ImageNet normalization (used by CLIP, Florence2, most vision models).
pub const IMAGENET_MEAN = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
pub const IMAGENET_STD = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };

/// SigLIP / SigLIP2 normalization: mean/std 0.5 per channel (maps [0,1] -> [-1,1]).
/// See `google/siglip2-*` `preprocessor_config.json` (`image_mean`/`image_std`).
pub const SIGLIP_MEAN = [3]f32{ 0.5, 0.5, 0.5 };
pub const SIGLIP_STD = [3]f32{ 0.5, 0.5, 0.5 };

pub const Resample = shared.Resample;
pub const PixelFormat = shared.PixelFormat;
pub const ImageU8 = shared.ImageU8;

/// Decoded image in HWC u8 format.
pub const Image = struct {
    data: []u8,
    width: u32,
    height: u32,
    channels: u32,

    pub fn deinit(self: Image, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }
};

pub const Pix2StructPatches = struct {
    flattened_patches: []f32,
    attention_mask: []i64,
    rows: usize,
    cols: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Pix2StructPatches) void {
        self.allocator.free(self.flattened_patches);
        self.allocator.free(self.attention_mask);
    }
};

/// Decode a JPEG/PNG/GIF from raw bytes. Always returns 3-channel RGB.
pub fn decode(allocator: std.mem.Allocator, image_bytes: []const u8) !Image {
    if (isPng(image_bytes)) {
        const decoded = antfly_image.png.decodeRgba(allocator, image_bytes) catch |err| switch (err) {
            error.PngDecodeFailed, error.UnsupportedPngFormat => return error.ImageDecodeFailed,
            else => return err,
        };
        errdefer allocator.free(decoded.rgba);
        const rgb = try rgbaToRgbAlloc(allocator, decoded.rgba);
        allocator.free(decoded.rgba);
        return .{
            .data = rgb,
            .width = decoded.width,
            .height = decoded.height,
            .channels = 3,
        };
    }

    if (isJpeg(image_bytes)) {
        const decoded = antfly_image.jpeg.decodeRgba(allocator, image_bytes) catch |err| switch (err) {
            error.JpegDecodeFailed => return error.ImageDecodeFailed,
            else => return err,
        };
        errdefer allocator.free(decoded.rgba);
        const rgb = try rgbaToRgbAlloc(allocator, decoded.rgba);
        allocator.free(decoded.rgba);
        return .{
            .data = rgb,
            .width = decoded.width,
            .height = decoded.height,
            .channels = 3,
        };
    }

    if (isGif(image_bytes)) {
        const frames = antfly_image.gif.decodeFramesAlloc(allocator, image_bytes) catch |err| switch (err) {
            error.GifDecodeFailed, error.UnsupportedGifFormat => return error.ImageDecodeFailed,
            else => return err,
        };
        defer {
            for (frames) |frame| allocator.free(frame.rgba);
            allocator.free(frames);
        }
        if (frames.len == 0) return error.ImageDecodeFailed;
        const rgb = try rgbaToRgbAlloc(allocator, frames[0].rgba);
        return .{
            .data = rgb,
            .width = frames[0].width,
            .height = frames[0].height,
            .channels = 3,
        };
    }

    return error.ImageDecodeFailed;
}

fn rgbaToRgbAlloc(allocator: std.mem.Allocator, rgba: []const u8) ![]u8 {
    if (rgba.len % 4 != 0) return error.ImageDecodeFailed;
    const pixel_count = rgba.len / 4;
    const rgb = try allocator.alloc(u8, pixel_count * 3);
    errdefer allocator.free(rgb);
    for (0..pixel_count) |i| {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
    return rgb;
}

fn isPng(bytes: []const u8) bool {
    return bytes.len >= 8 and std.mem.eql(u8, bytes[0..8], &.{ 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' });
}

fn isJpeg(bytes: []const u8) bool {
    return bytes.len >= 3 and bytes[0] == 0xff and bytes[1] == 0xd8 and bytes[2] == 0xff;
}

fn isGif(bytes: []const u8) bool {
    return bytes.len >= 6 and (std.mem.eql(u8, bytes[0..6], "GIF87a") or std.mem.eql(u8, bytes[0..6], "GIF89a"));
}

const red_png_2x2 = [_]u8{
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
    0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02,
    0x08, 0x02, 0x00, 0x00, 0x00, 0xfd, 0xd4, 0x9a, 0x73, 0x00, 0x00, 0x00,
    0x09, 0x70, 0x48, 0x59, 0x73, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x4f, 0x25, 0xc4, 0xd6, 0x00, 0x00, 0x00, 0x10, 0x49, 0x44,
    0x41, 0x54, 0x78, 0x9c, 0x63, 0xfc, 0xc3, 0x00, 0x02, 0x2c, 0x60, 0x92,
    0x01, 0x00, 0x0d, 0x04, 0x01, 0x02, 0xbf, 0x50, 0x15, 0xb3, 0x00, 0x00,
    0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
};

const animated_gif_1x1 = [_]u8{
    0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x21, 0xff, 0x0b, 0x4e, 0x45, 0x54, 0x53, 0x43, 0x41, 0x50, 0x45,
    0x32, 0x2e, 0x30, 0x03, 0x01, 0x00, 0x00, 0x00, 0x21, 0xf9, 0x04, 0x01,
    0x05, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01,
    0x00, 0x81, 0x00, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00, 0xff, 0x00, 0x00,
    0x00, 0x00, 0x02, 0x02, 0x4c, 0x01, 0x00, 0x21, 0xf9, 0x04, 0x01, 0x07,
    0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
    0x81, 0x00, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00,
    0x00, 0x02, 0x02, 0x54, 0x01, 0x00, 0x3b,
};

test "decode png fixture returns rgb image" {
    const alloc = std.testing.allocator;
    const img = try decode(alloc, &red_png_2x2);
    defer img.deinit(alloc);

    try std.testing.expectEqual(@as(u32, 2), img.width);
    try std.testing.expectEqual(@as(u32, 2), img.height);
    try std.testing.expectEqual(@as(u32, 3), img.channels);
    try std.testing.expectEqualSlices(u8, &.{ 0xfc, 0x00, 0x00 }, img.data[0..3]);
}

test "decode animated gif returns first frame rgb image" {
    const alloc = std.testing.allocator;
    const img = try decode(alloc, &animated_gif_1x1);
    defer img.deinit(alloc);

    try std.testing.expectEqual(@as(u32, 1), img.width);
    try std.testing.expectEqual(@as(u32, 1), img.height);
    try std.testing.expectEqual(@as(u32, 3), img.channels);
    try std.testing.expectEqualSlices(u8, &.{ 0xff, 0x00, 0x00 }, img.data[0..3]);
}

test "decode rejects unsupported image format" {
    const alloc = std.testing.allocator;
    try std.testing.expectError(error.ImageDecodeFailed, decode(alloc, "not-an-image"));
}

test "clipResizeDims resizes shortest edge to target with aspect preserved" {
    // torchvision Resize(int): shortest edge -> target, long edge = trunc(target*long/short).
    const a = clipResizeDims(767, 462, 224); // landscape
    try std.testing.expectEqual(@as(usize, 371), a.width);
    try std.testing.expectEqual(@as(usize, 224), a.height);
    const b = clipResizeDims(598, 728, 224); // portrait
    try std.testing.expectEqual(@as(usize, 224), b.width);
    try std.testing.expectEqual(@as(usize, 272), b.height);
    const c = clipResizeDims(500, 500, 224); // square
    try std.testing.expectEqual(@as(usize, 224), c.width);
    try std.testing.expectEqual(@as(usize, 224), c.height);
}

test "centerCropOffset uses round-half-to-even like torchvision CenterCrop" {
    try std.testing.expectEqual(@as(usize, 74), centerCropOffset(371, 224)); // round(73.5) -> 74 (even)
    try std.testing.expectEqual(@as(usize, 2), centerCropOffset(229, 224)); //  round(2.5)  -> 2  (even)
    try std.testing.expectEqual(@as(usize, 24), centerCropOffset(272, 224)); // 48/2 = 24 (exact)
    try std.testing.expectEqual(@as(usize, 0), centerCropOffset(224, 224));
}

test "clip preprocessing normalizes a target-sized image without resampling" {
    const alloc = std.testing.allocator;
    // 2x2 image with target_size=2 => resized dims == source dims => no resize,
    // zero crop offset, so the output is a pure rescale+normalize (CHW layout).
    var rgb = [_]u8{
        0,   0,   0,
        255, 255, 255,
        0,   0,   0,
        255, 255, 255,
    };
    const img = Image{ .data = rgb[0..], .width = 2, .height = 2, .channels = 3 };
    var out: [12]f32 = undefined;
    try preprocessDecodedClip(alloc, img, &out, 2, .{ 0.0, 0.0, 0.0 }, .{ 1.0, 1.0, 1.0 });
    // channel-0 plane row-major: (0,0)=0, (1,0)=255, (0,1)=0, (1,1)=255.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[3], 1e-6);
}

test "decode png fixture dimensions are stable" {
    const alloc = std.testing.allocator;
    const img = try decode(alloc, &red_png_2x2);
    defer img.deinit(alloc);

    try std.testing.expectEqual(@as(u32, 2), img.width);
    try std.testing.expectEqual(@as(u32, 2), img.height);
    try std.testing.expectEqual(@as(u32, 3), img.channels);
    try std.testing.expectEqualSlices(u8, &.{ 0xfc, 0x00, 0x00 }, img.data[0..3]);
}

/// Preprocess image for a vision model: decode → resize → normalize → CHW f32.
/// Returns [3, target_size, target_size] as f32 in channel-first layout.
pub fn preprocess(
    allocator: std.mem.Allocator,
    image_bytes: []const u8,
    target_size: u32,
    mean: [3]f32,
    std_dev: [3]f32,
) ![]f32 {
    return preprocessWithResample(allocator, image_bytes, target_size, mean, std_dev, .bilinear);
}

/// Preprocess image with explicit resample mode.
pub fn preprocessWithResample(
    allocator: std.mem.Allocator,
    image_bytes: []const u8,
    target_size: u32,
    mean: [3]f32,
    std_dev: [3]f32,
    resample: Resample,
) ![]f32 {
    const img = try decode(allocator, image_bytes);
    defer img.deinit(allocator);
    return preprocessDecodedWithResample(allocator, img, target_size, mean, std_dev, resample);
}

/// Preprocess an already-decoded image.
pub fn preprocessToSize(
    allocator: std.mem.Allocator,
    image_bytes: []const u8,
    target_width: u32,
    target_height: u32,
    mean: [3]f32,
    std_dev: [3]f32,
) ![]f32 {
    const img = try decode(image_bytes);
    defer img.deinit();
    return preprocessDecodedToSize(allocator, img, target_width, target_height, mean, std_dev);
}

pub fn preprocessDecodedToSize(
    allocator: std.mem.Allocator,
    img: Image,
    target_width: u32,
    target_height: u32,
    mean: [3]f32,
    std_dev: [3]f32,
) ![]f32 {
    const tw: usize = target_width;
    const th: usize = target_height;
    const result = try allocator.alloc(f32, 3 * th * tw);
    errdefer allocator.free(result);

    const src_w: f32 = @floatFromInt(img.width);
    const src_h: f32 = @floatFromInt(img.height);
    const scale_x = src_w / @as(f32, @floatFromInt(tw));
    const scale_y = src_h / @as(f32, @floatFromInt(th));

    for (0..th) |y| {
        const src_y = @as(f32, @floatFromInt(y)) * scale_y;
        const y0: u32 = @intFromFloat(@floor(src_y));
        const y1: u32 = @min(y0 + 1, img.height - 1);
        const fy = src_y - @as(f32, @floatFromInt(y0));

        for (0..tw) |x| {
            const src_x = @as(f32, @floatFromInt(x)) * scale_x;
            const x0: u32 = @intFromFloat(@floor(src_x));
            const x1: u32 = @min(x0 + 1, img.width - 1);
            const fx = src_x - @as(f32, @floatFromInt(x0));

            for (0..3) |ch| {
                const p00 = pixelAt(img, x0, y0, ch);
                const p10 = pixelAt(img, x1, y0, ch);
                const p01 = pixelAt(img, x0, y1, ch);
                const p11 = pixelAt(img, x1, y1, ch);

                const top = p00 * (1.0 - fx) + p10 * fx;
                const bot = p01 * (1.0 - fx) + p11 * fx;
                const val = top * (1.0 - fy) + bot * fy;

                result[ch * th * tw + y * tw + x] = (val / 255.0 - mean[ch]) / std_dev[ch];
            }
        }
    }

    return result;
}

fn pixelAt(img: Image, x: u32, y: u32, ch: usize) f32 {
    const channels: usize = @intCast(img.channels);
    const width: usize = @intCast(img.width);
    const xi: usize = @intCast(@min(x, img.width - 1));
    const yi: usize = @intCast(@min(y, img.height - 1));
    const ci = @min(ch, channels - 1);
    const idx = (yi * width + xi) * channels + ci;
    return @floatFromInt(img.data[idx]);
}

pub fn preprocessDecoded(
    allocator: std.mem.Allocator,
    img: Image,
    target_size: u32,
    mean: [3]f32,
    std_dev: [3]f32,
) ![]f32 {
    return preprocessDecodedWithResample(allocator, img, target_size, mean, std_dev, .bilinear);
}

pub fn preprocessDecodedWithResample(
    allocator: std.mem.Allocator,
    img: Image,
    target_size: u32,
    mean: [3]f32,
    std_dev: [3]f32,
    resample: Resample,
) ![]f32 {
    return shared.preprocessDecodedWithResample(allocator, toSharedImage(img), target_size, mean, std_dev, resample);
}

/// Preprocess an already-decoded image to an explicit width/height target.
pub fn preprocessDecodedRect(
    allocator: std.mem.Allocator,
    img: Image,
    target_width: u32,
    target_height: u32,
    mean: [3]f32,
    std_dev: [3]f32,
) ![]f32 {
    return preprocessDecodedRectWithResample(allocator, img, target_width, target_height, mean, std_dev, .bilinear);
}

pub fn preprocessDecodedRectWithResample(
    allocator: std.mem.Allocator,
    img: Image,
    target_width: u32,
    target_height: u32,
    mean: [3]f32,
    std_dev: [3]f32,
    resample: Resample,
) ![]f32 {
    return shared.preprocessDecodedRectWithResample(allocator, toSharedImage(img), target_width, target_height, mean, std_dev, resample);
}

pub fn preprocessDecodedRectScaledWithResample(
    allocator: std.mem.Allocator,
    img: Image,
    target_width: u32,
    target_height: u32,
    mean: [3]f32,
    std_dev: [3]f32,
    rescale_factor: f32,
    resample: Resample,
) ![]f32 {
    var values = try shared.preprocessDecodedRectWithResample(
        allocator,
        toSharedImage(img),
        target_width,
        target_height,
        mean,
        std_dev,
        resample,
    );
    errdefer allocator.free(values);
    const scale = rescale_factor * 255.0;
    if (!std.math.approxEqAbs(f32, scale, 1.0, 1e-6)) {
        const plane_stride = @as(usize, @intCast(target_width)) * @as(usize, @intCast(target_height));
        for (0..3) |ch| {
            const adjust = (mean[ch] * (scale - 1.0)) / std_dev[ch];
            const plane = values[ch * plane_stride ..][0..plane_stride];
            for (plane) |*value| value.* = value.* * scale + adjust;
        }
    }
    return values;
}

pub fn preprocessDecodedPix2Struct(
    allocator: std.mem.Allocator,
    img: Image,
    patch_height: usize,
    patch_width: usize,
    max_patches: usize,
    do_normalize: bool,
    resample: Resample,
) !Pix2StructPatches {
    _ = resample;

    if (img.channels < 3 or patch_height == 0 or patch_width == 0 or max_patches == 0) {
        return error.InvalidImageBuffer;
    }

    const feature_depth = 2 + patch_height * patch_width * 3;
    const flattened_patches = try allocator.alloc(f32, max_patches * feature_depth);
    errdefer allocator.free(flattened_patches);
    @memset(flattened_patches, 0);

    const attention_mask = try allocator.alloc(i64, max_patches);
    errdefer allocator.free(attention_mask);
    @memset(attention_mask, 0);

    const image_height = @as(f64, @floatFromInt(img.height));
    const image_width = @as(f64, @floatFromInt(img.width));
    const patch_h_f = @as(f64, @floatFromInt(patch_height));
    const patch_w_f = @as(f64, @floatFromInt(patch_width));
    const max_patches_f = @as(f64, @floatFromInt(max_patches));

    const scale = @sqrt(max_patches_f * (patch_h_f / image_height) * (patch_w_f / image_width));
    var rows = @max(@as(usize, @intFromFloat(@floor(scale * image_height / patch_h_f))), 1);
    var cols = @max(@as(usize, @intFromFloat(@floor(scale * image_width / patch_w_f))), 1);
    rows = @min(rows, max_patches);
    cols = @min(cols, max_patches);
    while (rows * cols > max_patches) {
        if (cols >= rows and cols > 1) {
            cols -= 1;
        } else if (rows > 1) {
            rows -= 1;
        } else {
            break;
        }
    }

    const resized_height = rows * patch_height;
    const resized_width = cols * patch_width;
    const scale_x = @as(f32, @floatCast(image_width / @as(f64, @floatFromInt(resized_width))));
    const scale_y = @as(f32, @floatCast(image_height / @as(f64, @floatFromInt(resized_height))));

    var mean: f32 = 0;
    var adjusted_stddev: f32 = 1;
    if (do_normalize) {
        var sum: f64 = 0;
        for (img.data) |value| sum += @as(f64, @floatFromInt(value));
        const count = @as(f64, @floatFromInt(img.data.len));
        mean = @floatCast(sum / count);

        var variance_sum: f64 = 0;
        for (img.data) |value| {
            const centered = @as(f64, @floatFromInt(value)) - mean;
            variance_sum += centered * centered;
        }
        const stddev = @sqrt(variance_sum / count);
        const min_stddev = 1.0 / @sqrt(count);
        adjusted_stddev = @floatCast(@max(stddev, min_stddev));
    }

    for (0..rows) |row| {
        for (0..cols) |col| {
            const patch_idx = row * cols + col;
            attention_mask[patch_idx] = 1;

            const base = patch_idx * feature_depth;
            flattened_patches[base] = @floatFromInt(row + 1);
            flattened_patches[base + 1] = @floatFromInt(col + 1);

            var out_idx = base + 2;
            for (0..patch_height) |patch_y| {
                const dst_y = row * patch_height + patch_y;
                for (0..patch_width) |patch_x| {
                    const dst_x = col * patch_width + patch_x;
                    for (0..3) |ch| {
                        const raw = sampleImageResized(img, dst_x, dst_y, ch, scale_x, scale_y);
                        flattened_patches[out_idx] = if (do_normalize)
                            (raw - mean) / adjusted_stddev
                        else
                            raw;
                        out_idx += 1;
                    }
                }
            }
        }
    }

    return .{
        .flattened_patches = flattened_patches,
        .attention_mask = attention_mask,
        .rows = rows,
        .cols = cols,
        .allocator = allocator,
    };
}

fn sampleImageResized(
    img: Image,
    dst_x: usize,
    dst_y: usize,
    ch: usize,
    scale_x: f32,
    scale_y: f32,
) f32 {
    const src_x = (@as(f32, @floatFromInt(dst_x)) + 0.5) * scale_x - 0.5;
    const src_y = (@as(f32, @floatFromInt(dst_y)) + 0.5) * scale_y - 0.5;
    const x0 = clampImageIndex(@intFromFloat(@floor(src_x)), img.width);
    const y0 = clampImageIndex(@intFromFloat(@floor(src_y)), img.height);
    const x1 = clampImageIndex(@as(i32, @intCast(x0)) + 1, img.width);
    const y1 = clampImageIndex(@as(i32, @intCast(y0)) + 1, img.height);
    const fx = src_x - @as(f32, @floatFromInt(x0));
    const fy = src_y - @as(f32, @floatFromInt(y0));

    const p00 = imagePixelAt(img, x0, y0, ch);
    const p10 = imagePixelAt(img, x1, y0, ch);
    const p01 = imagePixelAt(img, x0, y1, ch);
    const p11 = imagePixelAt(img, x1, y1, ch);

    const top = p00 * (1.0 - fx) + p10 * fx;
    const bottom = p01 * (1.0 - fx) + p11 * fx;
    return top * (1.0 - fy) + bottom * fy;
}

fn imagePixelAt(img: Image, x: u32, y: u32, ch: usize) f32 {
    const idx = (@as(usize, y) * @as(usize, img.width) + @as(usize, x)) * @as(usize, img.channels) + ch;
    return @floatFromInt(img.data[idx]);
}

fn clampImageIndex(idx: i32, dim: u32) u32 {
    if (idx <= 0) return 0;
    const max_idx: i32 = @intCast(dim - 1);
    if (idx >= max_idx) return @intCast(max_idx);
    return @intCast(idx);
}

pub fn computeAspectFitWidth(src_width: u32, src_height: u32, target_height: u32, max_width: u32) u32 {
    if (src_width == 0 or src_height == 0 or target_height == 0 or max_width == 0) return 0;
    const scaled = (@as(u64, src_width) * @as(u64, target_height) + @as(u64, src_height / 2)) / @as(u64, src_height);
    const clamped = @min(@as(u64, max_width), @max(@as(u64, 1), scaled));
    return @intCast(clamped);
}

pub fn preprocessDecodedRectKeepAspectPadRightWithResample(
    allocator: std.mem.Allocator,
    img: Image,
    max_width: u32,
    target_height: u32,
    mean: [3]f32,
    std_dev: [3]f32,
    resample: Resample,
    pad_rgb: [3]u8,
) ![]f32 {
    return preprocessDecodedRectKeepAspectPadRightScaledWithResample(
        allocator,
        img,
        max_width,
        target_height,
        mean,
        std_dev,
        1.0 / 255.0,
        resample,
        pad_rgb,
    );
}

pub fn preprocessDecodedRectKeepAspectPadRightScaledWithResample(
    allocator: std.mem.Allocator,
    img: Image,
    max_width: u32,
    target_height: u32,
    mean: [3]f32,
    std_dev: [3]f32,
    rescale_factor: f32,
    resample: Resample,
    pad_rgb: [3]u8,
) ![]f32 {
    const resized_width = computeAspectFitWidth(img.width, img.height, target_height, max_width);
    if (resized_width == 0) return error.InvalidImageBuffer;

    var resized = try preprocessDecodedRectScaledWithResample(
        allocator,
        img,
        resized_width,
        target_height,
        mean,
        std_dev,
        rescale_factor,
        resample,
    );
    defer allocator.free(resized);

    const output_plane_stride = @as(usize, max_width) * @as(usize, target_height);
    const resized_plane_stride = @as(usize, resized_width) * @as(usize, target_height);
    const output = try allocator.alloc(f32, 3 * output_plane_stride);
    errdefer allocator.free(output);

    for (0..3) |ch| {
        const pad_value = ((@as(f32, @floatFromInt(pad_rgb[ch])) * rescale_factor) - mean[ch]) / std_dev[ch];
        @memset(output[ch * output_plane_stride ..][0..output_plane_stride], pad_value);

        const src_plane = resized[ch * resized_plane_stride ..][0..resized_plane_stride];
        const dst_plane = output[ch * output_plane_stride ..][0..output_plane_stride];
        for (0..target_height) |row| {
            const dst_row = row * @as(usize, max_width);
            const src_row = row * @as(usize, resized_width);
            @memcpy(dst_plane[dst_row .. dst_row + @as(usize, resized_width)], src_plane[src_row .. src_row + @as(usize, resized_width)]);
        }
    }

    return output;
}

/// Preprocess a batch of images. Returns [batch, 3, target_size, target_size] as f32.
pub fn preprocessBatch(
    allocator: std.mem.Allocator,
    image_list: []const []const u8,
    target_size: u32,
    mean: [3]f32,
    std_dev: [3]f32,
) ![]f32 {
    const ts: usize = target_size;
    const per_image = 3 * ts * ts;
    const result = try allocator.alloc(f32, image_list.len * per_image);
    errdefer allocator.free(result);

    for (image_list, 0..) |image_bytes, i| {
        const single = try preprocess(allocator, image_bytes, target_size, mean, std_dev);
        defer allocator.free(single);
        @memcpy(result[i * per_image ..][0..per_image], single);
    }

    return result;
}

/// Preprocess CLIP embedding images to match the CLIP canonical contract used
/// by HuggingFace `CLIPImageProcessor` and torchvision
/// `Resize(size, BICUBIC) + CenterCrop(size)` — the exact preprocessing that
/// produced the cached clipclap ViT features the CLIP→text bridge was trained
/// and evaluated against (clipclap `processor_config.json`: `size.shortest_edge
/// = 224`, `resample = 3` (BICUBIC), `do_center_crop`, CLIP `image_mean`/`std`,
/// `rescale_factor = 1/255`, RGB).
///
/// Each image is resized so its SHORTEST edge equals `target_size` (aspect ratio
/// preserved) with an antialiased separable bicubic resampler (PIL semantics),
/// center-cropped to `target_size` × `target_size`, rescaled by 1/255, and
/// normalized with `mean`/`std` into CHW f32.
///
/// This replaces a previous path that center-cropped a `min(dim, target)` window
/// at NATIVE resolution and then near-identity-resized it — which never
/// downsampled large images (charts/figures), so the served image tensor barely
/// resembled the torchvision-bicubic training features (served-vs-offline image
/// cosine as low as ~0.15 on charts). Matching PIL's antialiased bicubic recovers
/// the pixel tensor to cosine ~1.0 vs torchvision. Use only for CLIP/ClipClap
/// image embeddings.
pub fn preprocessClipBatch(
    allocator: std.mem.Allocator,
    image_list: []const []const u8,
    target_size: u32,
    mean: [3]f32,
    std_dev: [3]f32,
) ![]f32 {
    const ts: usize = target_size;
    const per_image = 3 * ts * ts;
    const result = try allocator.alloc(f32, image_list.len * per_image);
    errdefer allocator.free(result);

    for (image_list, 0..) |image_bytes, i| {
        const img = try decode(allocator, image_bytes);
        defer img.deinit(allocator);

        try preprocessDecodedClip(
            allocator,
            img,
            result[i * per_image ..][0..per_image],
            target_size,
            mean,
            std_dev,
        );
    }

    return result;
}

const ClipResizeDims = struct { width: usize, height: usize };

/// torchvision `Resize(size=int)` output size: match the SHORTEST edge to
/// `target` (aspect preserved), long edge = `int(target * long / short)` (Python
/// truncation of the f64 quotient — reproduced here in f64 to be bit-identical).
fn clipResizeDims(width: u32, height: u32, target: u32) ClipResizeDims {
    const t: f64 = @floatFromInt(target);
    const w: f64 = @floatFromInt(width);
    const h: f64 = @floatFromInt(height);
    if (width <= height) {
        const nh: usize = @max(@as(usize, @intFromFloat(t * h / w)), 1);
        return .{ .width = target, .height = nh };
    } else {
        const nw: usize = @max(@as(usize, @intFromFloat(t * w / h)), 1);
        return .{ .width = nw, .height = target };
    }
}

/// torchvision `CenterCrop` top/left offset: `int(round((dim - target) / 2))`
/// using Python's round-half-to-even. Returns 0 when the axis is not larger
/// than the crop.
fn centerCropOffset(dim: usize, target: usize) usize {
    if (dim <= target) return 0;
    const d = dim - target; // >= 1
    const half = d / 2;
    if (d % 2 == 0) return half; // exact integer, no rounding needed
    // d odd => (d/2) ends in .5 => round to the even neighbor (half or half+1).
    return if (half % 2 == 0) half else half + 1;
}

/// CLIP canonical preprocessing for a single decoded image. See
/// `preprocessClipBatch` for the contract. `result` must be `3*target*target`.
fn preprocessDecodedClip(
    allocator: std.mem.Allocator,
    img: Image,
    result: []f32,
    target_size: u32,
    mean: [3]f32,
    std_dev: [3]f32,
) !void {
    std.debug.assert(target_size > 0);
    std.debug.assert(img.width > 0 and img.height > 0);
    const ts: usize = target_size;
    std.debug.assert(result.len == 3 * ts * ts);

    const dims = clipResizeDims(img.width, img.height, target_size);

    // Resize (antialiased separable bicubic) to dims, unless already exact.
    var resized_owned: ?[]u8 = null;
    defer if (resized_owned) |buf| allocator.free(buf);
    var resized_data: []const u8 = undefined;
    var resized_w: usize = undefined;
    var resized_h: usize = undefined;
    var resized_ch: usize = undefined;

    if (dims.width == img.width and dims.height == img.height) {
        resized_data = img.data;
        resized_w = img.width;
        resized_h = img.height;
        resized_ch = img.channels;
    } else {
        const buf = try resizeAntialiasSeparable(allocator, img, dims.width, dims.height, .bicubic);
        resized_owned = buf;
        resized_data = buf;
        resized_w = dims.width;
        resized_h = dims.height;
        resized_ch = 3;
    }

    // Center-crop target×target using torchvision's round-half-to-even offsets.
    const crop_left = centerCropOffset(resized_w, ts);
    const crop_top = centerCropOffset(resized_h, ts);

    for (0..ts) |y| {
        const src_row = (crop_top + y) * resized_w;
        for (0..ts) |x| {
            const base = (src_row + crop_left + x) * resized_ch;
            inline for (0..3) |ch| {
                const v: f32 = @floatFromInt(resized_data[base + ch]);
                result[ch * ts * ts + y * ts + x] = (v / 255.0 - mean[ch]) / std_dev[ch];
            }
        }
    }
}

/// SigLIP canonical preprocessing for a single decoded image: resize the whole
/// image to `target`×`target` SQUARE (aspect ratio is NOT preserved — SigLIP
/// squashes to square: no shortest-edge resize and no center crop) with an
/// antialiased separable **bilinear** resampler (PIL `resample=2` semantics),
/// rescale by 1/255, and normalize with `mean`/`std` into CHW f32. Matches
/// HuggingFace `SiglipImageProcessor` (`google/siglip2-base-patch16-512`
/// `preprocessor_config.json`: `size={512,512}`, `resample=2` (BILINEAR),
/// `rescale_factor=1/255`, `image_mean`/`image_std`=0.5, RGB).
fn preprocessDecodedSiglip(
    allocator: std.mem.Allocator,
    img: Image,
    result: []f32,
    target_size: u32,
    mean: [3]f32,
    std_dev: [3]f32,
) !void {
    std.debug.assert(target_size > 0);
    std.debug.assert(img.width > 0 and img.height > 0);
    const ts: usize = target_size;
    std.debug.assert(result.len == 3 * ts * ts);

    // Resize the whole image to ts×ts (antialiased separable bilinear), unless
    // it is already exactly square at the target size.
    var resized_owned: ?[]u8 = null;
    defer if (resized_owned) |buf| allocator.free(buf);
    var resized_data: []const u8 = undefined;
    var resized_ch: usize = undefined;

    if (img.width == ts and img.height == ts) {
        resized_data = img.data;
        resized_ch = img.channels;
    } else {
        const buf = try resizeAntialiasSeparable(allocator, img, ts, ts, .bilinear);
        resized_owned = buf;
        resized_data = buf;
        resized_ch = 3;
    }

    for (0..ts) |y| {
        const src_row = y * ts;
        for (0..ts) |x| {
            const base = (src_row + x) * resized_ch;
            inline for (0..3) |ch| {
                const v: f32 = @floatFromInt(resized_data[base + ch]);
                result[ch * ts * ts + y * ts + x] = (v / 255.0 - mean[ch]) / std_dev[ch];
            }
        }
    }
}

/// Preprocess SigLIP embedding images to match `SiglipImageProcessor`: square
/// bilinear resize (no crop) + rescale 1/255 + 0.5/0.5 normalize into CHW f32.
/// Returns [batch, 3, target_size, target_size]. Use for SigLIP/SigLIP2 image
/// embeddings (the SigLIP→text bridge is trained on these pixel tensors).
pub fn preprocessSiglipBatch(
    allocator: std.mem.Allocator,
    image_list: []const []const u8,
    target_size: u32,
    mean: [3]f32,
    std_dev: [3]f32,
) ![]f32 {
    const ts: usize = target_size;
    const per_image = 3 * ts * ts;
    const result = try allocator.alloc(f32, image_list.len * per_image);
    errdefer allocator.free(result);

    for (image_list, 0..) |image_bytes, i| {
        const img = try decode(allocator, image_bytes);
        defer img.deinit(allocator);

        try preprocessDecodedSiglip(
            allocator,
            img,
            result[i * per_image ..][0..per_image],
            target_size,
            mean,
            std_dev,
        );
    }

    return result;
}

/// Antialiased separable resample filter selector.
const ResampleFilter = enum { bilinear, bicubic };

/// Antialiased separable resample of a decoded image to `ow`×`oh` u8, matching
/// PIL's `Image.resize(..., filter)` (the resampler torchvision uses for PIL
/// images): separate horizontal then vertical passes over precomputed,
/// support-scaled, normalized coefficients with a u8 rounding step between
/// passes. `filter` selects the kernel (bilinear support 1.0 / bicubic support
/// 2.0). Caller owns the returned RGB (3-channel) buffer.
fn resizeAntialiasSeparable(allocator: std.mem.Allocator, img: Image, ow: usize, oh: usize, filter: ResampleFilter) ![]u8 {
    const sw: usize = img.width;
    const sh: usize = img.height;
    const sc: usize = img.channels;

    // Horizontal pass: sw -> ow, preserving sh rows, emitting 3-channel u8.
    var hc = try computeResampleCoeffs(allocator, sw, ow, filter);
    defer hc.deinit(allocator);
    const hbuf = try allocator.alloc(u8, sh * ow * 3);
    defer allocator.free(hbuf);
    for (0..sh) |y| {
        for (0..ow) |x| {
            const xmin: usize = @intCast(hc.bounds_min[x]);
            const n: usize = @intCast(hc.bounds_size[x]);
            const wbase = x * hc.ksize;
            inline for (0..3) |ch| {
                var acc: f64 = 0;
                for (0..n) |k| {
                    const px: f64 = @floatFromInt(img.data[(y * sw + (xmin + k)) * sc + ch]);
                    acc += px * hc.weights[wbase + k];
                }
                hbuf[(y * ow + x) * 3 + ch] = clip8(acc);
            }
        }
    }

    // Vertical pass: sh -> oh over the horizontally-resized 3-channel buffer.
    var vc = try computeResampleCoeffs(allocator, sh, oh, filter);
    defer vc.deinit(allocator);
    const vbuf = try allocator.alloc(u8, oh * ow * 3);
    errdefer allocator.free(vbuf);
    for (0..oh) |y| {
        const ymin: usize = @intCast(vc.bounds_min[y]);
        const n: usize = @intCast(vc.bounds_size[y]);
        const wbase = y * vc.ksize;
        for (0..ow) |x| {
            inline for (0..3) |ch| {
                var acc: f64 = 0;
                for (0..n) |k| {
                    const px: f64 = @floatFromInt(hbuf[((ymin + k) * ow + x) * 3 + ch]);
                    acc += px * vc.weights[wbase + k];
                }
                vbuf[(y * ow + x) * 3 + ch] = clip8(acc);
            }
        }
    }
    return vbuf;
}

const BicubicCoeffs = struct {
    bounds_min: []i32, // [out] first source index contributing to each output pixel
    bounds_size: []i32, // [out] number of source pixels contributing
    weights: []f64, // [out * ksize] normalized filter weights
    ksize: usize,

    fn deinit(self: *BicubicCoeffs, allocator: std.mem.Allocator) void {
        allocator.free(self.bounds_min);
        allocator.free(self.bounds_size);
        allocator.free(self.weights);
        self.* = undefined;
    }
};

/// Precompute PIL-style resample coefficients for a 1-D axis (`in_size` ->
/// `out_size`) for the given `filter`. Mirrors PIL's `precompute_coeffs`: the
/// filter support scales with the downsampling factor (antialiasing), the
/// sampling window is clamped to the source, and weights are normalized to sum
/// to 1. Base support is 1.0 for bilinear (triangle) and 2.0 for bicubic (Keys).
fn computeResampleCoeffs(allocator: std.mem.Allocator, in_size: usize, out_size: usize, filter: ResampleFilter) !BicubicCoeffs {
    const in_f: f64 = @floatFromInt(in_size);
    const out_f: f64 = @floatFromInt(out_size);
    const scale = in_f / out_f;
    const filterscale = @max(scale, 1.0);
    const base_support: f64 = switch (filter) {
        .bilinear => 1.0,
        .bicubic => 2.0,
    };
    const support = base_support * filterscale;
    const ksize: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;
    const ss = 1.0 / filterscale;
    const in_i: i64 = @intCast(in_size);

    const bounds_min = try allocator.alloc(i32, out_size);
    errdefer allocator.free(bounds_min);
    const bounds_size = try allocator.alloc(i32, out_size);
    errdefer allocator.free(bounds_size);
    const weights = try allocator.alloc(f64, out_size * ksize);
    errdefer allocator.free(weights);
    @memset(weights, 0);

    for (0..out_size) |xx| {
        const center = (@as(f64, @floatFromInt(xx)) + 0.5) * scale;
        // PIL: (int)(center - support + 0.5) / (int)(center + support + 0.5).
        var xmin_i: i64 = @intFromFloat(center - support + 0.5);
        if (xmin_i < 0) xmin_i = 0;
        var xmax_i: i64 = @intFromFloat(center + support + 0.5);
        if (xmax_i > in_i) xmax_i = in_i;
        const xmin: usize = @intCast(xmin_i);
        var n: usize = @intCast(xmax_i - xmin_i);
        if (n > ksize) n = ksize;

        var ww: f64 = 0;
        for (0..n) |k| {
            const arg = (@as(f64, @floatFromInt(k + xmin)) - center + 0.5) * ss;
            const w = switch (filter) {
                .bilinear => bilinearKernel(arg),
                .bicubic => bicubicKernel(arg),
            };
            weights[xx * ksize + k] = w;
            ww += w;
        }
        if (ww != 0.0) {
            for (0..n) |k| weights[xx * ksize + k] /= ww;
        }
        bounds_min[xx] = @intCast(xmin);
        bounds_size[xx] = @intCast(n);
    }

    return .{ .bounds_min = bounds_min, .bounds_size = bounds_size, .weights = weights, .ksize = ksize };
}

/// Triangle (linear) kernel, support 1.0 (PIL/torchvision BILINEAR).
fn bilinearKernel(x: f64) f64 {
    const t = @abs(x);
    if (t < 1.0) return 1.0 - t;
    return 0.0;
}

/// Keys cubic convolution kernel with a = -0.5 (PIL/torchvision BICUBIC).
fn bicubicKernel(x: f64) f64 {
    const a: f64 = -0.5;
    const t = @abs(x);
    if (t < 1.0) return ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0;
    if (t < 2.0) return (((t - 5.0) * t + 8.0) * t - 4.0) * a;
    return 0.0;
}

/// Round a resampled value to u8 with saturation (PIL's clip8: `(int)(v + 0.5)`
/// clamped to [0, 255]).
fn clip8(v: f64) u8 {
    if (v <= 0.0) return 0;
    if (v >= 255.0) return 255;
    return @intFromFloat(v + 0.5);
}

fn toSharedImage(img: Image) ImageU8 {
    return .{
        .data = img.data[0 .. @as(usize, img.width) * @as(usize, img.height) * @as(usize, img.channels)],
        .width = img.width,
        .height = img.height,
        .format = switch (img.channels) {
            3 => .rgb8,
            4 => .rgba8,
            else => .rgb8,
        },
    };
}

// Permanent parity guard for the CLIP/clipclap served image path. The server's
// image preprocessing had never been parity-checked against the CLIP-canonical
// torchvision pipeline the clipclap ViT features + CLIP→text bridge were trained
// on, which let a resize/crop skew ship (served-vs-offline image cosine ~0.15 on
// charts). This asserts the server-preprocessed pixel tensor matches a committed
// torchvision reference (Resize(224, BICUBIC) + CenterCrop(224) + rescale +
// CLIP-normalize) at cosine >= 0.99. Regenerate the fixtures with
// `testdata/clip_preprocess/gen_reference.py`.
test "clip preprocessing matches torchvision CLIP canonical" {
    const alloc = std.testing.allocator;
    const png = @embedFile("testdata/clip_preprocess/input.png");
    const ref_bytes = @embedFile("testdata/clip_preprocess/reference_chw_f16.bin");

    const n = 3 * 224 * 224;
    try std.testing.expectEqual(@as(usize, n * @sizeOf(f16)), ref_bytes.len);

    const img = try decode(alloc, png);
    defer img.deinit(alloc);

    const out = try alloc.alloc(f32, n);
    defer alloc.free(out);
    try preprocessDecodedClip(alloc, img, out, 224, IMAGENET_MEAN, IMAGENET_STD);

    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..n) |i| {
        const bits = std.mem.readInt(u16, ref_bytes[i * 2 ..][0..2], .little);
        const ref: f64 = @floatCast(@as(f16, @bitCast(bits)));
        const a: f64 = out[i];
        dot += a * ref;
        na += a * a;
        nb += ref * ref;
    }
    const cosine = dot / (@sqrt(na) * @sqrt(nb) + 1e-12);
    std.debug.print("[clip-preprocess-parity] cosine(server, torchvision) = {d:.6}\n", .{cosine});
    try std.testing.expect(cosine >= 0.99);
}

// Permanent parity guard for the SigLIP/SigLIP2 served image path. Asserts the
// server-preprocessed pixel tensor (square antialiased-bilinear resize to 512,
// no crop, rescale 1/255, 0.5/0.5 normalize) matches a committed HuggingFace
// `SiglipImageProcessor` reference (google/siglip2-base-patch16-512) at cosine
// >= 0.99. Regenerate the fixtures with
// `testdata/siglip_preprocess/gen_reference.py`.
test "siglip preprocessing matches SiglipImageProcessor canonical" {
    const alloc = std.testing.allocator;
    const png = @embedFile("testdata/siglip_preprocess/input.png");
    const ref_bytes = @embedFile("testdata/siglip_preprocess/reference_chw_f16.bin");

    const ts: u32 = 512;
    const n = 3 * @as(usize, ts) * @as(usize, ts);
    try std.testing.expectEqual(@as(usize, n * @sizeOf(f16)), ref_bytes.len);

    const img = try decode(alloc, png);
    defer img.deinit(alloc);

    const out = try alloc.alloc(f32, n);
    defer alloc.free(out);
    try preprocessDecodedSiglip(alloc, img, out, ts, SIGLIP_MEAN, SIGLIP_STD);

    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..n) |i| {
        const bits = std.mem.readInt(u16, ref_bytes[i * 2 ..][0..2], .little);
        const ref: f64 = @floatCast(@as(f16, @bitCast(bits)));
        const a: f64 = out[i];
        dot += a * ref;
        na += a * a;
        nb += ref * ref;
    }
    const cosine = dot / (@sqrt(na) * @sqrt(nb) + 1e-12);
    std.debug.print("[siglip-preprocess-parity] cosine(server, SiglipImageProcessor) = {d:.6}\n", .{cosine});
    try std.testing.expect(cosine >= 0.99);
}
