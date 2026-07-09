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

//! Backend-independent operation identities shared by the kernel compiler and
//! backend renderers. ABI and schedule details belong to tagged render plans.

pub const OpKind = enum(u8) {
    small_batch_matmul,
    attention,
    microkernel,
};

pub const MicrokernelKind = enum {
    rms_norm,
};

pub const AttentionKind = enum {
    decode_1x,
    prefill_flash,
};

pub const Epilogue = enum(u8) {
    none,
    bias,
    bias_gelu,
    pair,
    triple,
    relu,
    gelu,
    add,
    argmax,
    /// Fused FFN gate+up: two projections sharing one q8_1-quantized input,
    /// activation multiply, q8_1-requantized output.
    pair_activation,
    /// FFN down projection consuming the q8_1-quantized activated vector
    /// produced by `pair_activation`, with f32 output.
    gated_down,
};
