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
const gemma_graph = @import("architectures/gemma_graph.zig");
const gemma4_train_command = @import("finetune/gemma4_train_command.zig");
const gemma4_bf16_cli = @import("finetune/test/test_gemma4_bf16_cli.zig");
const metal_compute = @import("ops/metal_compute.zig");
const recipe = @import("finetune/recipe.zig");

test "gemma4 finetune embedded regressions are linked into the focused gate" {
    std.testing.refAllDecls(gemma_graph);
    std.testing.refAllDecls(gemma4_train_command);
    std.testing.refAllDecls(gemma4_bf16_cli);
    std.testing.refAllDecls(metal_compute);
    std.testing.refAllDecls(recipe);
}
