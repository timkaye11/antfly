// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "../../src/ops/cuda/kernels/embedding_gemma2.cuh"

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        std::exit(2);
    }
}
static std::vector<unsigned char> read(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
    const auto bytes = file.tellg();
    if (bytes <= 0) { std::fprintf(stderr, "empty input %s\n", path.c_str()); std::exit(2); }
    std::vector<unsigned char> result(static_cast<size_t>(bytes));
    file.seekg(0); file.read(reinterpret_cast<char*>(result.data()), bytes);
    if (!file) std::exit(2);
    return result;
}
template<class T> static T* upload(const std::vector<unsigned char>& bytes) {
    T* pointer = nullptr;
    check(cudaMalloc(&pointer, bytes.size()));
    check(cudaMemcpy(pointer, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
    return pointer;
}
int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: image-resize-differential <generated-case-directory>\n"); return 2; }
    bool passed = true;
    for (unsigned index = 0; index < 7; ++index) {
        const auto path = std::string(argv[1]) + "/case-" + std::to_string(index) + "/";
        const auto shape_bytes = read(path + "shape.u64");
        unsigned long long shape[8];
        if (shape_bytes.size() != sizeof(shape)) return 2;
        std::memcpy(shape, shape_bytes.data(), sizeof(shape));
        const auto expected = read(path + "expected.f32");
        const unsigned sw = shape[0], sh = shape[1], tw = shape[3], th = shape[4], patch = shape[5];
        if (!sw || !sh || !tw || !th || patch != 16 || tw % patch || th % patch || shape[6] < 1 || shape[6] > 31 || shape[7] < 1 || shape[7] > 31) return 2;
        const size_t count = static_cast<size_t>(tw) * th * 3;
        if (expected.size() != count * sizeof(float)) return 2;
        auto* input = upload<unsigned char>(read(path + "source.u8"));
        auto* xs = upload<unsigned long long>(read(path + "x_starts.u64"));
        auto* xo = upload<unsigned long long>(read(path + "x_offsets.u64"));
        auto* xw = upload<int>(read(path + "x_weights.i32"));
        auto* ys = upload<unsigned long long>(read(path + "y_starts.u64"));
        auto* yo = upload<unsigned long long>(read(path + "y_offsets.u64"));
        auto* yw = upload<int>(read(path + "y_weights.i32"));
        unsigned char* horizontal = nullptr;
        float* output = nullptr;
        const size_t horizontal_count = static_cast<size_t>(sh) * tw * 3;
        check(cudaMalloc(&horizontal, horizontal_count));
        check(cudaMalloc(&output, expected.size()));
        auto launch = [&]() {
            termite_embedding_gemma2_pillow_horizontal_rgb8<<<(horizontal_count + 255) / 256, 256>>>(horizontal, input, xs, xo, xw, sh, tw, shape[2], shape[6]);
            termite_embedding_gemma2_pillow_vertical_patches_f32<<<(count + 255) / 256, 256>>>(output, horizontal, ys, yo, yw, tw, th, patch, shape[7]);
        };
        launch(); check(cudaGetLastError());
        std::vector<unsigned char> actual(expected.size());
        check(cudaMemcpy(actual.data(), output, actual.size(), cudaMemcpyDeviceToHost));
        size_t differences = 0;
        for (size_t i = 0; i < count; ++i) {
            differences += std::memcmp(actual.data() + i * 4, expected.data() + i * 4, 4) != 0;
        }
        cudaEvent_t begin = nullptr, end = nullptr;
        check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
        for (int iteration = 0; iteration < 5; ++iteration) launch();
        check(cudaEventRecord(begin));
        for (int iteration = 0; iteration < 100; ++iteration) launch();
        check(cudaEventRecord(end)); check(cudaEventSynchronize(end)); check(cudaGetLastError());
        float milliseconds = 0;
        check(cudaEventElapsedTime(&milliseconds, begin, end));
        std::printf("{\"case\":%u,\"source\":[%u,%u],\"target\":[%u,%u],\"elements\":%zu,\"different_float_bits\":%zu,\"kernel_ms\":%.9g,\"pass\":%s}\n", index, sw, sh, tw, th, count, differences, milliseconds / 100, differences == 0 ? "true" : "false");
        passed = passed && differences == 0;
        check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
        for (auto* pointer : {static_cast<void*>(input), static_cast<void*>(xs), static_cast<void*>(xo), static_cast<void*>(xw), static_cast<void*>(ys), static_cast<void*>(yo), static_cast<void*>(yw), static_cast<void*>(horizontal), static_cast<void*>(output)}) check(cudaFree(pointer));
    }
    return passed ? 0 : 1;
}
