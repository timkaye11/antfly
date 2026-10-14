// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
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

//! Pinned tokenizer, chat-template and generation sidecars for the two local
//! Qwen runners. Both artifacts ship byte-identical files, so one set of pins
//! keeps the runners from accepting different sidecars.
const std = @import("std");
const c_file = @import("util/c_file.zig");
const verified_sidecar = @import("util/verified_sidecar.zig");
const generation = @import("pipelines/generation.zig");
const hf = @import("inference_hf_tokenizer");
const tokenizer_mod = @import("inference_tokenizer");
const BoundedAllocator = @import("runtime/bounded_allocator.zig").BoundedAllocator;

pub const Pin = struct { name: []const u8, bytes: usize, sha256: []const u8 };
pub const tokenizer_pin: Pin = .{ .name = "tokenizer.json", .bytes = 12809320, .sha256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3" };
pub const tokenizer_config_pin: Pin = .{ .name = "tokenizer_config.json", .bytes = 17928, .sha256 = "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27" };
pub const generation_config_pin: Pin = .{ .name = "generation_config.json", .bytes = 202, .sha256 = "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e" };
/// Tokenizer parsing, vocabulary and caches are bounded independently of the
/// caller's allocator (debug allocation metadata can otherwise dwarf the
/// vocabulary). Both memory plans include this ceiling in their host allowance.
pub const tokenizer_limit_bytes: usize = 512 * 1024 * 1024;

const TemplateConfig = struct { chat_template: []const u8, bos_token: ?[]const u8 = null, eos_token: ?[]const u8 = null, unk_token: ?[]const u8 = null, pad_token: ?[]const u8 = null };
const GenerationConfig = struct { eos_token_id: []const u32 };

pub const Sidecars = struct {
    tokenizer_allocator: BoundedAllocator = .{ .backing = std.heap.c_allocator, .limit = tokenizer_limit_bytes },
    hf_tokenizer: ?*hf.HfTokenizer = null,
    template_config: ?std.json.Parsed(TemplateConfig) = null,
    generation_config: ?std.json.Parsed(GenerationConfig) = null,
    template: ?generation.ChatTemplate = null,

    /// Sidecars sit beside the GGUF or one directory above it. The value must
    /// not move once loaded: the tokenizer keeps a pointer to its allocator.
    /// `raw` skips the chat template.
    pub fn load(self: *Sidecars, a: std.mem.Allocator, gguf_path: []const u8, raw: bool) !void {
        errdefer self.deinit();
        var dir = std.fs.path.dirname(gguf_path) orelse ".";
        if (!c_file.fileExistsInDir(a, dir, tokenizer_pin.name)) dir = std.fs.path.dirname(dir) orelse return error.MissingQwenTokenizer;
        {
            const bytes = try read(a, dir, tokenizer_pin);
            defer a.free(bytes);
            self.hf_tokenizer = try hf.HfTokenizer.loadFromBytes(self.tokenizer_allocator.allocator(), bytes);
        }
        {
            const bytes = try read(a, dir, tokenizer_config_pin);
            defer a.free(bytes);
            self.template_config = try std.json.parseFromSlice(TemplateConfig, a, bytes, .{ .ignore_unknown_fields = true });
        }
        {
            // EOS is a generation-sidecar contract, separate from any token
            // the model uses internally to reset state.
            const bytes = try read(a, dir, generation_config_pin);
            defer a.free(bytes);
            self.generation_config = try std.json.parseFromSlice(GenerationConfig, a, bytes, .{ .ignore_unknown_fields = true });
        }
        if (!raw) {
            const tc = self.template_config.?.value;
            self.template = try generation.ChatTemplate.init(a, tc.chat_template, tc.bos_token orelse "", tc.eos_token orelse "", tc.unk_token orelse "", tc.pad_token orelse "");
        }
    }

    pub fn deinit(self: *Sidecars) void {
        if (self.template) |*t| t.deinit();
        if (self.generation_config) |config| config.deinit();
        if (self.template_config) |config| config.deinit();
        if (self.hf_tokenizer) |t| t.deinitSelf();
        std.debug.assert(self.tokenizer_allocator.live == 0);
        self.template = null;
        self.generation_config = null;
        self.template_config = null;
        self.hf_tokenizer = null;
    }

    pub fn tokenizer(self: *const Sidecars) tokenizer_mod.Tokenizer {
        return self.hf_tokenizer.?.tokenizer();
    }

    pub fn chatTemplate(self: *Sidecars) ?*generation.ChatTemplate {
        return if (self.template) |*t| t else null;
    }

    pub fn eos(self: *const Sidecars) []const u32 {
        return self.generation_config.?.value.eos_token_id;
    }
};

fn read(a: std.mem.Allocator, dir: []const u8, pin: Pin) ![]u8 {
    return verified_sidecar.read(a, dir, pin.name, pin.bytes, pin.sha256);
}
