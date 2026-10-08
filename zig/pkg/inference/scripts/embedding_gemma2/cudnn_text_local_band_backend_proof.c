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

#include <cudnn.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDNN_CHECK(call) do { \
    cudnnStatus_t status_ = (call); \
    if (status_ != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudnnGetErrorString(status_)); \
        exit(2); \
    } \
} while (0)
#define CUDA_CHECK(call) do { \
    cudaError_t status_ = (call); \
    if (status_ != cudaSuccess) { \
        fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(status_)); \
        exit(2); \
    } \
} while (0)

static void set_attr(cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t name,
                     cudnnBackendAttributeType_t type, int64_t count, void const *value) {
    CUDNN_CHECK(cudnnBackendSetAttribute(descriptor, name, type, count, value));
}

static cudnnBackendDescriptor_t make_tensor(int64_t uid, int64_t batch, int64_t heads,
                                             int64_t sequence, int64_t width, int by_value) {
    cudnnBackendDescriptor_t tensor;
    CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &tensor));
    int64_t dimensions[4] = {batch, heads, sequence, width};
    int64_t strides[4] = {sequence * heads * width, width, heads * width, 1};
    int64_t alignment = 16;
    cudnnDataType_t data_type = by_value ? CUDNN_DATA_FLOAT : CUDNN_DATA_BFLOAT16;
    set_attr(tensor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
    set_attr(tensor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &data_type);
    set_attr(tensor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment);
    if (by_value) {
        int64_t one = 1;
        int is_by_value = 1;
        int64_t scalar_dims[4] = {1, 1, 1, 1};
        int64_t scalar_strides[4] = {1, 1, 1, 1};
        set_attr(tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, scalar_dims);
        set_attr(tensor, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, scalar_strides);
        set_attr(tensor, CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN, 1, &is_by_value);
    } else {
        set_attr(tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dimensions);
        set_attr(tensor, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, strides);
    }
    CUDNN_CHECK(cudnnBackendFinalize(tensor));
    return tensor;
}

static cudnnBackendDescriptor_t make_score_tensor(int64_t uid, int64_t batch, int64_t heads,
                                                   int64_t sequence) {
    cudnnBackendDescriptor_t tensor;
    CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &tensor));
    int64_t dimensions[4] = {batch, heads, sequence, sequence};
    int64_t strides[4] = {heads * sequence * sequence, sequence * sequence, sequence, 1};
    int64_t alignment = 16;
    int is_virtual = 1;
    cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
    set_attr(tensor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
    set_attr(tensor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &data_type);
    set_attr(tensor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment);
    set_attr(tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dimensions);
    set_attr(tensor, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, strides);
    set_attr(tensor, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &is_virtual);
    CUDNN_CHECK(cudnnBackendFinalize(tensor));
    return tensor;
}

static cudnnBackendDescriptor_t make_scalar(int64_t uid, cudnnDataType_t type) {
    cudnnBackendDescriptor_t tensor;
    CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &tensor));
    int64_t one = 1, alignment = 4;
    int is_by_value = 1;
    set_attr(tensor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
    set_attr(tensor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &type);
    set_attr(tensor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment);
    int64_t scalar_dims[4] = {1, 1, 1, 1};
    int64_t scalar_strides[4] = {1, 1, 1, 1};
    set_attr(tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, scalar_dims);
    set_attr(tensor, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, scalar_strides);
    set_attr(tensor, CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN, 1, &is_by_value);
    CUDNN_CHECK(cudnnBackendFinalize(tensor));
    return tensor;
}

static uint16_t to_bfloat16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static void write_values(char const *path, uint16_t const *values, size_t count) {
    FILE *file = fopen(path, "wb");
    if (file == NULL || fwrite(values, sizeof(*values), count, file) != count || fclose(file) != 0) {
        fprintf(stderr, "failed to write %s\n", path);
        exit(2);
    }
}

int main(int argc, char **argv) {
    int64_t batch = argc > 2 ? strtoll(argv[2], NULL, 10) : 8;
    int64_t query_heads = 4;
    int64_t kv_heads = 2;
    int64_t sequence = argc > 1 ? strtoll(argv[1], NULL, 10) : 512;
    int64_t width = 256;
    float amplitude = argc > 3 ? strtof(argv[3], NULL) : 0.1f;
    int boundary_fixture = argc > 4 && strcmp(argv[4], "boundary") == 0;
    int64_t uids[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    cudnnHandle_t handle;
    cudaStream_t stream;
    CUDNN_CHECK(cudnnCreate(&handle));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDNN_CHECK(cudnnSetStream(handle, stream));

    cudnnBackendDescriptor_t tensors[5];
    tensors[0] = make_tensor(1, batch, query_heads, sequence, width, 0);
    tensors[1] = make_tensor(2, batch, kv_heads, sequence, width, 0);
    tensors[2] = make_tensor(3, batch, kv_heads, sequence, width, 0);
    tensors[3] = make_tensor(4, batch, query_heads, sequence, width, 0);
    tensors[4] = make_tensor(5, 1, 1, 1, 1, 1);
    cudnnBackendDescriptor_t left_bound = make_scalar(6, CUDNN_DATA_INT32);
    cudnnBackendDescriptor_t right_bound = make_scalar(7, CUDNN_DATA_INT32);
    cudnnBackendDescriptor_t fill = make_scalar(8, CUDNN_DATA_FLOAT);
    cudnnBackendDescriptor_t scores[3] = {
        make_score_tensor(100, batch, query_heads, sequence),
        make_score_tensor(101, batch, query_heads, sequence),
        make_score_tensor(102, batch, query_heads, sequence),
    };
    cudnnBackendDescriptor_t masks[2];
    for (int i = 0; i < 2; ++i) {
        CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_DIAGONAL_BAND_MASK_DESCRIPTOR, &masks[i]));
        set_attr(masks[i], CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &scores[i]);
        cudnnBackendDescriptor_t bound = i == 0 ? right_bound : left_bound;
        cudnnBackendAttributeName_t attr = i == 0
            ? CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SHIFT_RIGHT_BOUND_DESC
            : CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_LEFT_BOUND_DESC;
        set_attr(masks[i], attr, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &bound);
        set_attr(masks[i], CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_BDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &fill);
        set_attr(masks[i], CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &scores[i + 1]);
        cudnnPointwiseMode_t cmp = i == 0 ? CUDNN_POINTWISE_CMP_GE : CUDNN_POINTWISE_CMP_LT;
        set_attr(masks[i], CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_COMPARISON_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &cmp);
        CUDNN_CHECK(cudnnBackendFinalize(masks[i]));
    }
    cudnnBackendDescriptor_t subgraph;
    CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &subgraph));
    set_attr(subgraph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
    set_attr(subgraph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 2, masks);
    CUDNN_CHECK(cudnnBackendFinalize(subgraph));

    cudnnBackendDescriptor_t operation;
    CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR, &operation));
    set_attr(operation, CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tensors[0]);
    set_attr(operation, CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tensors[1]);
    set_attr(operation, CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tensors[2]);
    set_attr(operation, CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tensors[3]);
    set_attr(operation, CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tensors[4]);
    set_attr(operation, CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &subgraph);
    int64_t subgraph_input_uid = 100, subgraph_output_uid = 102;
    set_attr(operation, CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_INPUT_UID, CUDNN_TYPE_INT64, 1, &subgraph_input_uid);
    set_attr(operation, CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_OUTPUT_UID, CUDNN_TYPE_INT64, 1, &subgraph_output_uid);
    CUDNN_CHECK(cudnnBackendFinalize(operation));

    cudnnBackendDescriptor_t graph;
    CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &graph));
    set_attr(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
    set_attr(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &operation);
    CUDNN_CHECK(cudnnBackendFinalize(graph));

    cudnnBackendDescriptor_t heuristic;
    CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &heuristic));
    cudnnBackendHeurMode_t mode = CUDNN_HEUR_MODE_A;
    set_attr(heuristic, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph);
    set_attr(heuristic, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &mode);
    CUDNN_CHECK(cudnnBackendFinalize(heuristic));

    int64_t config_count = 0;
    CUDNN_CHECK(cudnnBackendGetAttribute(heuristic, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &config_count, NULL));
    if (config_count == 0) return 3;
    cudnnBackendDescriptor_t *configs = calloc((size_t)config_count, sizeof(*configs));
    for (int64_t i = 0; i < config_count; ++i)
        CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &configs[i]));
    int64_t returned = 0;
    CUDNN_CHECK(cudnnBackendGetAttribute(heuristic, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR, config_count, &returned, configs));

    cudnnBackendDescriptor_t plan = NULL;
    int64_t selected = -1;
    for (int64_t i = 0; i < returned; ++i) {
        CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan));
        set_attr(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
        set_attr(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &configs[i]);
        if (cudnnBackendFinalize(plan) == CUDNN_STATUS_SUCCESS) { selected = i; break; }
        CUDNN_CHECK(cudnnBackendDestroyDescriptor(plan));
        plan = NULL;
    }
    if (selected < 0) return 4;

    int64_t workspace_bytes = 0, actual = 0;
    CUDNN_CHECK(cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                         CUDNN_TYPE_INT64, 1, &actual, &workspace_bytes));
    size_t query_count = (size_t)(batch * sequence * query_heads * width);
    size_t kv_count = (size_t)(batch * sequence * kv_heads * width);
    size_t query_bytes = query_count * sizeof(uint16_t);
    size_t kv_bytes = kv_count * sizeof(uint16_t);
    uint16_t *host[4];
    void *pointers[8];
    host[0] = malloc(query_bytes);
    host[1] = malloc(kv_bytes);
    host[2] = malloc(kv_bytes);
    host[3] = malloc(query_bytes);
    for (size_t i = 0; i < query_count; ++i) {
        host[0][i] = to_bfloat16(amplitude * sinf((float)i * 0.001f));
    }
    for (size_t i = 0; i < kv_count; ++i) {
        host[1][i] = to_bfloat16(amplitude * cosf((float)i * 0.0013f));
        host[2][i] = to_bfloat16(amplitude * sinf((float)i * 0.0007f));
    }
    if (boundary_fixture) {
        if (batch != 1 || sequence != 515) {
            fprintf(stderr, "boundary fixture requires batch=1 sequence=515\n");
            return 5;
        }
        memset(host[0], 0, query_bytes);
        memset(host[1], 0, kv_bytes);
        memset(host[2], 0, kv_bytes);
        int64_t sentinel_keys[5] = {0, 1, 512, 513, 514};
        for (int64_t head = 0; head < kv_heads; ++head) {
            for (int64_t channel = 0; channel < 5; ++channel) {
                size_t index = ((size_t)sentinel_keys[channel] * (size_t)kv_heads + (size_t)head) *
                               (size_t)width + (size_t)channel;
                host[2][index] = to_bfloat16(1.0f);
            }
        }
    }
    CUDA_CHECK(cudaMalloc(&pointers[0], query_bytes));
    CUDA_CHECK(cudaMalloc(&pointers[1], kv_bytes));
    CUDA_CHECK(cudaMalloc(&pointers[2], kv_bytes));
    CUDA_CHECK(cudaMalloc(&pointers[3], query_bytes));
    CUDA_CHECK(cudaMemcpy(pointers[0], host[0], query_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pointers[1], host[1], kv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pointers[2], host[2], kv_bytes, cudaMemcpyHostToDevice));
    float attention_scale = 1.0f;
    pointers[4] = &attention_scale;
    int32_t left = 513, right = 512;
    float negative_inf = -INFINITY;
    pointers[5] = &left;
    pointers[6] = &right;
    pointers[7] = &negative_inf;
    void *workspace = NULL;
    if (workspace_bytes != 0) CUDA_CHECK(cudaMalloc(&workspace, (size_t)workspace_bytes));

    cudnnBackendDescriptor_t variant_pack;
    CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &variant_pack));
    set_attr(variant_pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 8, uids);
    set_attr(variant_pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 8, pointers);
    set_attr(variant_pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &workspace);
    CUDNN_CHECK(cudnnBackendFinalize(variant_pack));

    for (int i = 0; i < 10; ++i) CUDNN_CHECK(cudnnBackendExecute(handle, plan, variant_pack));
    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(begin, stream));
    for (int i = 0; i < 100; ++i) CUDNN_CHECK(cudnnBackendExecute(handle, plan, variant_pack));
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, begin, end));
    CUDA_CHECK(cudaMemcpy(host[3], pointers[3], query_bytes, cudaMemcpyDeviceToHost));
    write_values("/tmp/cudnn_text_gqa_q.bf16", host[0], query_count);
    write_values("/tmp/cudnn_text_gqa_k.bf16", host[1], kv_count);
    write_values("/tmp/cudnn_text_gqa_v.bf16", host[2], kv_count);
    write_values("/tmp/cudnn_text_gqa_out.bf16", host[3], query_count);
    printf("version=%zu batch=%ld query_heads=%ld kv_heads=%ld sequence=%ld width=%ld "
           "amplitude=%g configs=%ld selected=%ld workspace=%ld avg_ms=%g\n",
           cudnnGetVersion(), batch, query_heads, kv_heads, sequence, width, amplitude,
           returned, selected, workspace_bytes, elapsed_ms / 100.0f);

    CUDNN_CHECK(cudnnBackendDestroyDescriptor(variant_pack));
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    if (workspace != NULL) CUDA_CHECK(cudaFree(workspace));
    for (int i = 0; i < 4; ++i) { CUDA_CHECK(cudaFree(pointers[i])); free(host[i]); }
    CUDNN_CHECK(cudnnBackendDestroyDescriptor(plan));
    for (int64_t i = 0; i < config_count; ++i) CUDNN_CHECK(cudnnBackendDestroyDescriptor(configs[i]));
    free(configs);
    CUDNN_CHECK(cudnnBackendDestroyDescriptor(heuristic));
    CUDNN_CHECK(cudnnBackendDestroyDescriptor(graph));
    CUDNN_CHECK(cudnnBackendDestroyDescriptor(operation));
    CUDNN_CHECK(cudnnBackendDestroyDescriptor(subgraph));
    for (int i = 0; i < 2; ++i) CUDNN_CHECK(cudnnBackendDestroyDescriptor(masks[i]));
    for (int i = 0; i < 3; ++i) CUDNN_CHECK(cudnnBackendDestroyDescriptor(scores[i]));
    CUDNN_CHECK(cudnnBackendDestroyDescriptor(left_bound));
    CUDNN_CHECK(cudnnBackendDestroyDescriptor(right_bound));
    CUDNN_CHECK(cudnnBackendDestroyDescriptor(fill));
    for (int i = 0; i < 5; ++i) CUDNN_CHECK(cudnnBackendDestroyDescriptor(tensors[i]));
    CUDNN_CHECK(cudnnDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
