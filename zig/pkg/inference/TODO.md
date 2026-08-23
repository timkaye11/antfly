# antfly-inference-zig TODO

## Features (missing from Go inference)

- [x] **Grammar + speculative decode parity**: native generation now applies grammar masking and grammar advancement during target-side speculative verification, so constrained decoding and draft-model acceleration can coexist on the native backend.
- [x] **Constrained decoding backend policy**: constrained decoding is explicitly native-backend only for now. ONNX `/api/generate` requests remain unconstrained-only unless we add token-level masking support there.
- [ ] **Result caching**: Go antfly inference has per-endpoint TTL caches with singleflight dedup for: embedding, sparse embedding, chunking, reranking, NER, reading, transcription. ResultCache and singleflight primitives exist in `src/cache/` but aren't wired into any handler.
- [ ] **Dense binary serialization**: Go antfly inference defaults to `application/octet-stream` for `/api/embed` (LE float32 arrays). JSON is opt-in via Accept header. We always return JSON.
- [ ] **Sparse binary serialization**: Go antfly inference supports binary `SparseVectorsContentType` (`application/x-sparse-vectors`, Accept header-based). Lower priority — JSON format works.
- [x] **Tool-calling parity**: `/api/generate` now executes FunctionGemma-style tool use end to end, including prompt formatting, non-streaming parsing, and streamed `tool_calls` argument deltas.
- [x] **Native multimodal generation**: native Gemma 3, Gemma 4, and Qwen-VL generation accepts image-bearing prompts; Gemma 4 external GGUF projectors run their vision/audio projection and decoder compute through the selected native backend, including CUDA.
- [ ] **Broader multimodal serving parity and performance**: expand request-level success, streaming, multi-image/audio, cancellation, and warm-server coverage across all supported multimodal families. Gemma 4 CUDA image projection now keeps clipping, axial 2D RoPE, long-sequence attention, pooling, and projection device-resident; the remaining Gemma 4 performance work is audio-transform residency and wider server/concurrency qualification.
- [ ] **Native GLiNER parity validation**: GLiNER now has a native DeBERTa + span-head path and prefers native weights when available. The remaining work is proving parity with real GLiNER models across MLX/BLAS, adding backend-specific tests, and tightening any performance gaps in the native head.
- [ ] **GLiNER2 Metal follow-ups**: Metal now beats native at batch 8-16, clearly beats Fastino on long documents, and ties Fastino on short documents (see [GLINER2_METAL.md](./GLINER2_METAL.md)). Remaining: batch-1 long-text attention (flash4 PV parallelization / query_block variant), an evidence-backed `auto` backend routing decision, and optional per-seq-len rel Q_r/K_r caching.
- [ ] **Metal runtime follow-ups from the gliner review**: provider-persistent dynamic linear-slot dedup so sequential requests reuse prepared mirrors instead of re-uploading; export the Metal provider counters (incl. the new host-fallback warns) through server metrics; decide the seq 513-1024 DeBERTa attention envelope (currently GEMM-or-naive).
- [ ] **Gemma 4 QAT Metal baseline performance**: execute the evidence-gated no-MTP roadmap in [GEMMA4_METAL_PERFORMANCE.md](./GEMMA4_METAL_PERFORMANCE.md), beginning with a current-binary pinned llama.cpp re-anchor and the decode GQA split schedule portfolio.

## MLX Gemma Follow-Up

- [ ] **Budget coordination plan**: see [BUDGETS.md](./BUDGETS.md) for the canonical plan covering `LoadBudget`, `RunBudget`, global coordination, and the Hypura-like tiered runtime direction.
- [ ] **Clean up MLX Gemma debug scaffolding**: remove the temporary scheduler/KV/paged-attention debug env toggles and noisy MLX tied-logits logging added during the Gemma 3 decode investigation, while keeping the actual correctness fixes.
- [x] **Validate MLX Gemma before commit**:
  1. short multimodal MLX CLI prompt
  2. short server `/api/generate` multimodal check
  3. one longer text-only MLX sample before committing
