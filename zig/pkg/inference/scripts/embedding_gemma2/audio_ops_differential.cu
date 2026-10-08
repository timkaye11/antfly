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

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "../../src/ops/cuda/kernels/embedding_gemma2.cuh"

static void cuda_check(cudaError_t status, const char* file, int line) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "%s:%d: %s\n", file, line, cudaGetErrorString(status));
    std::exit(2);
}
#define CUDA(call) cuda_check((call), __FILE__, __LINE__)

static bool compare_value(float actual, float expected, double* max_error) {
    if (std::isnan(expected)) return std::isnan(actual);
    if (std::isinf(expected)) return std::isinf(actual) && std::signbit(actual) == std::signbit(expected);
    if (!std::isfinite(actual)) return false;
    *max_error = std::fmax(*max_error, std::fabs(double(actual) - expected));
    return true;
}

struct Params {
    float* out;
    float* input;
    float* weight;
    unsigned rows;
    unsigned dim;
    unsigned kernel;
    float lower;
    float upper;
    unsigned has_lower;
    unsigned has_upper;
};

static void launch_clamp(Params* p) {
    const unsigned count = p->rows * p->dim;
    termite_gemma4_audio_clamp_f32<<<(count + 255) / 256, 256>>>(
        p->out, p->input, count, p->lower, p->upper, p->has_lower, p->has_upper);
}
static void launch_glu(Params* p) {
    const unsigned count = p->rows * p->dim;
    termite_gemma4_audio_glu_rows_f32<<<(count + 255) / 256, 256>>>(
        p->out, p->input, p->rows, p->dim);
}
static void launch_conv(Params* p) {
    const unsigned count = p->rows * p->dim;
    termite_gemma4_audio_depthwise_causal_conv1d_f32<<<(count + 255) / 256, 256>>>(
        p->out, p->input, p->weight, p->rows, p->dim, p->kernel);
}

template <typename Launch>
static float time_ms(Launch launch, Params* params) {
    cudaEvent_t begin = nullptr, end = nullptr;
    CUDA(cudaEventCreate(&begin)); CUDA(cudaEventCreate(&end));
    for (int i = 0; i < 10; ++i) launch(params);
    CUDA(cudaGetLastError()); CUDA(cudaEventRecord(begin));
    for (int i = 0; i < 1000; ++i) launch(params);
    CUDA(cudaGetLastError()); CUDA(cudaEventRecord(end)); CUDA(cudaEventSynchronize(end));
    float elapsed = 0.0f; CUDA(cudaEventElapsedTime(&elapsed, begin, end));
    CUDA(cudaEventDestroy(begin)); CUDA(cudaEventDestroy(end));
    return elapsed / 1000.0f;
}

int main(int argc, char** argv) {
    const unsigned rows = 25, dim = 1024;
    const unsigned kernel = argc > 1 ? unsigned(std::atoi(argv[1])) : 15;
    if (kernel != 5 && kernel != 15) return 3;
    const size_t count = size_t(rows) * dim;
    float* input = static_cast<float*>(std::malloc(2 * count * sizeof(float)));
    float* weight = static_cast<float*>(std::malloc(size_t(kernel) * dim * sizeof(float)));
    float* output = static_cast<float*>(std::malloc(count * sizeof(float)));
    if (!input || !weight || !output) return 4;
    for (size_t i = 0; i < 2 * count; ++i) input[i] = std::sin(float(i) * .013f) * 2.0f;
    for (size_t i = 0; i < size_t(kernel) * dim; ++i) weight[i] = std::cos(float(i) * .017f) * .1f;
    input[0] = NAN; input[1] = INFINITY; input[2] = -INFINITY;

    float *device_input = nullptr, *device_weight = nullptr, *device_output = nullptr;
    CUDA(cudaMalloc(&device_input, 2 * count * sizeof(float)));
    CUDA(cudaMalloc(&device_weight, size_t(kernel) * dim * sizeof(float)));
    CUDA(cudaMalloc(&device_output, count * sizeof(float)));
    CUDA(cudaMemcpy(device_input, input, 2 * count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(device_weight, weight, size_t(kernel) * dim * sizeof(float), cudaMemcpyHostToDevice));
    Params params{device_output, device_input, device_weight, rows, dim, kernel, -.75f, .625f, 1, 1};

    double max_error = 0.0;
    launch_clamp(&params); CUDA(cudaMemcpy(output, device_output, count * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; ++i) {
        float expected = input[i];
        if (expected < params.lower) expected = params.lower;
        if (expected > params.upper) expected = params.upper;
        if (!compare_value(output[i], expected, &max_error)) return 5;
    }
    if (!std::isnan(output[0]) || output[1] != params.upper || output[2] != params.lower || max_error != 0.0) return 5;
    params.has_upper = 0; launch_clamp(&params); CUDA(cudaMemcpy(output, device_output, count * sizeof(float), cudaMemcpyDeviceToHost));
    double optional_error = 0.0;
    for (size_t i = 0; i < count; ++i) {
        float expected = input[i];
        if (expected < params.lower) expected = params.lower;
        if (!compare_value(output[i], expected, &optional_error)) return 6;
    }
    if (!std::isnan(output[0]) || !std::isinf(output[1]) || std::signbit(output[1]) || output[2] != params.lower) return 6;
    params.has_lower = 0; params.has_upper = 1; launch_clamp(&params); CUDA(cudaMemcpy(output, device_output, count * sizeof(float), cudaMemcpyDeviceToHost));
    optional_error = 0.0;
    for (size_t i = 0; i < count; ++i) {
        float expected = input[i];
        if (expected > params.upper) expected = params.upper;
        if (!compare_value(output[i], expected, &optional_error)) return 7;
    }
    if (!std::isnan(output[0]) || output[1] != params.upper || !std::isinf(output[2]) || !std::signbit(output[2])) return 7;
    params.has_lower = 1;

    launch_glu(&params); CUDA(cudaMemcpy(output, device_output, count * sizeof(float), cudaMemcpyDeviceToHost));
    max_error = 0.0;
    for (unsigned row = 0; row < rows; ++row) for (unsigned column = 0; column < dim; ++column) {
        const size_t i = size_t(row) * dim + column;
        const float expected = input[size_t(row) * 2 * dim + column] /
            (1.0f + std::exp(-input[size_t(row) * 2 * dim + dim + column]));
        if (!compare_value(output[i], expected, &max_error)) return 8;
    }
    if (max_error > 3e-7) return 8;

    input[0] = 0.0f; input[1] = .1f; input[2] = -.1f;
    CUDA(cudaMemcpy(device_input, input, count * sizeof(float), cudaMemcpyHostToDevice));
    launch_conv(&params); CUDA(cudaMemcpy(output, device_output, count * sizeof(float), cudaMemcpyDeviceToHost));
    max_error = 0.0;
    for (unsigned time = 0; time < rows; ++time) for (unsigned channel = 0; channel < dim; ++channel) {
        float expected = 0.0f;
        for (unsigned tap = 0; tap < kernel; ++tap) {
            if (time + tap < kernel - 1) continue;
            const unsigned source_time = time + tap - (kernel - 1);
            expected += input[size_t(source_time) * dim + channel] * weight[size_t(tap) * dim + channel];
        }
        if (!compare_value(output[size_t(time) * dim + channel], expected, &max_error)) return 9;
    }
    if (max_error > 2e-7) return 9;
    CUDA(cudaGetLastError());
    std::printf("kernel=%u clamp_ms=%.8g glu_ms=%.8g conv_ms=%.8g max_conv_error=%.9g\n",
        kernel, time_ms(launch_clamp, &params), time_ms(launch_glu, &params),
        time_ms(launch_conv, &params), max_error);
    cudaFree(device_input); cudaFree(device_weight); cudaFree(device_output);
    std::free(input); std::free(weight); std::free(output);
    return 0;
}
