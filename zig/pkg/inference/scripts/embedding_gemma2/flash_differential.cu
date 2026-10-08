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
#include <cublasLt.h>
#include <mma.h>

#include "../../src/ops/cuda/kernels/embedding_gemma2.cuh"

__global__ void antfly_eg2_lt_masked_softmax(
    __nv_bfloat16* probs, const float* scores, const long long* mask,
    unsigned sequence, unsigned query_base, unsigned rows,
    unsigned batch_index, unsigned local_radius) {
    const unsigned row = blockIdx.x % rows;
    const unsigned head = blockIdx.x / rows;
    if (row >= rows) return;
    const unsigned lane = threadIdx.x;
    const unsigned query = query_base + row;
    const size_t matrix_offset = size_t(head) * 256u * sequence;
    const bool query_valid = mask[(size_t)batch_index * sequence + query] != 0;
    extern __shared__ float reduce[];
    float local_max = -INFINITY;
    for (unsigned key = lane; key < sequence; key += blockDim.x) {
        const bool visible = query_valid && mask[(size_t)batch_index * sequence + key] &&
            (!local_radius || (key <= query + local_radius && query <= key + local_radius));
        if (visible) local_max = fmaxf(local_max, scores[matrix_offset + (size_t)row * sequence + key]);
    }
    reduce[lane] = local_max;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
        if (lane < stride) reduce[lane] = fmaxf(reduce[lane], reduce[lane + stride]);
        __syncthreads();
    }
    const float row_max = reduce[0];
    float local_sum = 0.0f;
    for (unsigned key = lane; key < sequence; key += blockDim.x) {
        const bool visible = query_valid && mask[(size_t)batch_index * sequence + key] &&
            (!local_radius || (key <= query + local_radius && query <= key + local_radius));
        const float p = visible ? expf(scores[matrix_offset + (size_t)row * sequence + key] - row_max) : 0.0f;
        probs[matrix_offset + (size_t)row * sequence + key] = __float2bfloat16_rn(p);
        local_sum += p;
    }
    reduce[lane] = local_sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
        if (lane < stride) reduce[lane] += reduce[lane + stride];
        __syncthreads();
    }
    const float inv_sum = reduce[0] > 0.0f ? 1.0f / reduce[0] : 0.0f;
    for (unsigned key = lane; key < sequence; key += blockDim.x)
        probs[matrix_offset + (size_t)row * sequence + key] = __float2bfloat16_rn(
            __bfloat162float(probs[matrix_offset + (size_t)row * sequence + key]) * inv_sum);
}

static int antfly_eg2_lt_matmul(
    cublasLtHandle_t handle, cudaStream_t stream,
    const void* a, const void* b, void* d,
    unsigned m, unsigned n, unsigned k,
    unsigned lda, unsigned ldb, unsigned ldd,
    cudaDataType_t a_type, cudaDataType_t b_type, cudaDataType_t d_type,
    cublasOperation_t trans_b, unsigned batch_count,
    int64_t stride_a, int64_t stride_b, int64_t stride_d,
    void* workspace, size_t workspace_bytes) {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr, b_layout = nullptr, d_layout = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasStatus_t status = cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) return 1000 + int(status);
    status = cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB,
        &trans_b, sizeof(trans_b));
    const cublasOperation_t trans_a = CUBLAS_OP_N;
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutCreate(&a_layout, a_type, m, k, lda);
    const unsigned b_rows = trans_b == CUBLAS_OP_T ? n : k;
    const unsigned b_cols = trans_b == CUBLAS_OP_T ? k : n;
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutCreate(&b_layout, b_type, b_rows, b_cols, ldb);
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutCreate(&d_layout, d_type, m, n, ldd);
    const cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutSetAttribute(a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutSetAttribute(b_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutSetAttribute(d_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order));
    const int32_t batches = int32_t(batch_count);
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutSetAttribute(a_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batches, sizeof(batches));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutSetAttribute(b_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batches, sizeof(batches));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutSetAttribute(d_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batches, sizeof(batches));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutSetAttribute(a_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutSetAttribute(b_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutSetAttribute(d_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_d, sizeof(stride_d));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulPreferenceCreate(&preference);
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes));
    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulAlgoGetHeuristic(
        handle, operation, a_layout, b_layout, d_layout, d_layout,
        preference, 1, &heuristic, &returned);
    const float alpha = 1.0f, beta = 0.0f;
    if (status == CUBLAS_STATUS_SUCCESS && returned == 1) status = cublasLtMatmul(
        handle, operation, &alpha, a, a_layout, b, b_layout, &beta,
        d, d_layout, d, d_layout, &heuristic.algo, workspace, workspace_bytes, stream);
    else if (status == CUBLAS_STATUS_SUCCESS) status = CUBLAS_STATUS_NOT_SUPPORTED;
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (d_layout) cublasLtMatrixLayoutDestroy(d_layout);
    if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
    cublasLtMatmulDescDestroy(operation);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : 1000 + int(status);
}

static int antfly_eg2_lt_attention(
    __nv_bfloat16* output, const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, const long long* mask, unsigned batch,
    unsigned sequence, unsigned query_heads, unsigned kv_heads,
    unsigned head_dim, unsigned local_radius, cudaStream_t stream) {
    constexpr unsigned QT = 256;
    constexpr size_t LT_WORKSPACE = 32u * 1024u * 1024u;
    float* scores = nullptr;
    __nv_bfloat16* probs = nullptr;
    void* workspace = nullptr;
    cudaError_t error = cudaMallocAsync(&scores, size_t(query_heads) * QT * sequence * sizeof(float), stream);
    if (error == cudaSuccess) error = cudaMallocAsync(&probs, size_t(query_heads) * QT * sequence * sizeof(__nv_bfloat16), stream);
    if (error == cudaSuccess) error = cudaMallocAsync(&workspace, LT_WORKSPACE, stream);
    if (error != cudaSuccess) return int(error);
    cublasLtHandle_t handle = nullptr;
    cublasStatus_t create_status = cublasLtCreate(&handle);
    if (create_status != CUBLAS_STATUS_SUCCESS) return 1000 + int(create_status);
    int result = 0;
    for (unsigned bi = 0; bi < batch && result == 0; ++bi) {
        const unsigned heads_per_kv = query_heads / kv_heads;
        for (unsigned kvh = 0; kvh < kv_heads && result == 0; ++kvh) {
            const unsigned qh = kvh * heads_per_kv;
            for (unsigned qt = 0; qt < sequence && result == 0; qt += QT) {
                const unsigned rows = min(QT, sequence - qt);
                const __nv_bfloat16* q_tile = q + (((size_t)bi * sequence + qt) * query_heads + qh) * head_dim;
                const __nv_bfloat16* k_head = k + (((size_t)bi * sequence) * kv_heads + kvh) * head_dim;
                const __nv_bfloat16* v_head = v + (((size_t)bi * sequence) * kv_heads + kvh) * head_dim;
                __nv_bfloat16* out_tile = output + (((size_t)bi * sequence + qt) * query_heads + qh) * head_dim;
                result = antfly_eg2_lt_matmul(handle, stream, q_tile, k_head, scores,
                    rows, sequence, head_dim, query_heads * head_dim, kv_heads * head_dim,
                    sequence, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, CUBLAS_OP_T,
                    heads_per_kv, head_dim, 0, size_t(QT) * sequence,
                    workspace, LT_WORKSPACE);
                if (result) break;
                if (local_radius == 0) {
                    // Exercise the exact production kernel and its explicit
                    // fixed-head stride contract for global text attention.
                    termite_embedding_gemma2_attention_softmax_bf16<<<
                        rows * heads_per_kv, 256, 256 * sizeof(float), stream>>>(
                            probs, scores, mask, sequence, qt, rows, bi, QT);
                } else {
                    antfly_eg2_lt_masked_softmax<<<rows * heads_per_kv, 256, 256 * sizeof(float), stream>>>(
                        probs, scores, mask, sequence, qt, rows, bi, local_radius);
                }
                error = cudaGetLastError();
                if (error != cudaSuccess) { result = int(error); break; }
                result = antfly_eg2_lt_matmul(handle, stream, probs, v_head, out_tile,
                    rows, head_dim, sequence, sequence, kv_heads * head_dim,
                    query_heads * head_dim, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                    CUBLAS_OP_N, heads_per_kv, size_t(QT) * sequence, 0, head_dim,
                    workspace, LT_WORKSPACE);
            }
        }
    }
    cublasLtDestroy(handle);
    cudaFreeAsync(workspace, stream);
    cudaFreeAsync(probs, stream);
    cudaFreeAsync(scores, stream);
    return result;
}

extern "C" int antfly_eg2_attention_differential_launch(
    void* output,
    const void* q,
    const void* k,
    const void* v,
    const long long* mask,
    unsigned batch,
    unsigned sequence,
    unsigned query_heads,
    unsigned kv_heads,
    unsigned head_dim,
    unsigned local_radius,
    unsigned use_flash,
    void* stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    if (output == nullptr || q == nullptr || k == nullptr || v == nullptr ||
        mask == nullptr || batch == 0 || sequence == 0 || query_heads != 4 ||
        !((kv_heads == 2 && head_dim == 256) ||
          (kv_heads == 1 && head_dim == 512))) return int(cudaErrorInvalidValue);
    if (use_flash == 2u) {
        if (local_radius == 0) return antfly_eg2_lt_attention(
            static_cast<__nv_bfloat16*>(output), static_cast<const __nv_bfloat16*>(q),
            static_cast<const __nv_bfloat16*>(k), static_cast<const __nv_bfloat16*>(v),
            mask, batch, sequence, query_heads, kv_heads, head_dim, local_radius, stream);
        // Production routes local layers through Q64 full tiles and the exact
        // warp kernel only for the final partial tile.
        use_flash = 1u;
    }
    if (use_flash) {
        if (local_radius == 512u && kv_heads == 2u && head_dim == 256u) {
            const unsigned shared_bytes =
                16u * head_dim * sizeof(__nv_bfloat16) +
                4u * 64u * 16u * sizeof(float) +
                64u * 16u * sizeof(__nv_bfloat16) + 3u * 64u * sizeof(float);
            cudaError_t attribute_status = cudaFuncSetAttribute(
                termite_embedding_gemma2_attention_local_flash_bf16,
                cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared_bytes));
            if (attribute_status != cudaSuccess) return int(attribute_status);
            const unsigned full_tiles = sequence / 64u;
            if (full_tiles) {
                termite_embedding_gemma2_attention_local_flash_bf16<<<
                    dim3(query_heads, full_tiles, batch), 512u,
                    shared_bytes, stream>>>(
                        static_cast<__nv_bfloat16*>(output),
                        static_cast<const __nv_bfloat16*>(q),
                        static_cast<const __nv_bfloat16*>(k),
                        static_cast<const __nv_bfloat16*>(v), mask, batch, sequence,
                        query_heads, kv_heads, head_dim, local_radius);
                cudaError_t status = cudaGetLastError();
                if (status != cudaSuccess) return int(status);
            }
            const unsigned tail = sequence % 64u;
            if (tail) {
                const unsigned query_offset = sequence - tail;
                termite_embedding_gemma2_attention_bf16<<<
                    dim3(tail, query_heads, batch), 32u, 0u, stream>>>(
                        static_cast<__nv_bfloat16*>(output),
                        static_cast<const __nv_bfloat16*>(q),
                        static_cast<const __nv_bfloat16*>(k),
                        static_cast<const __nv_bfloat16*>(v), mask, batch, sequence,
                        query_heads, kv_heads, head_dim, local_radius, query_offset);
            }
            return int(cudaGetLastError());
        }
    }
    {
        unsigned query_offset = 0;
        termite_embedding_gemma2_attention_bf16<<<
            dim3(sequence, query_heads, batch), 32u, 0u, stream>>>(
                static_cast<__nv_bfloat16*>(output),
                static_cast<const __nv_bfloat16*>(q),
                static_cast<const __nv_bfloat16*>(k),
                static_cast<const __nv_bfloat16*>(v), mask, batch, sequence,
                query_heads, kv_heads, head_dim, local_radius, query_offset);
    }
    return int(cudaGetLastError());
}
