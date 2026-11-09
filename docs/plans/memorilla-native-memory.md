# Native Memorilla memory serving

Status: the supplied-feature Qwen3 implementation is complete and disabled by
default. Exact latent-only operational QA, database collection integration and
Gemma training remain unqualified or future work.

The [serving guide](../design/inference/memorilla-serving.md) defines the API,
configuration, resource limits and supported execution profile. The
[PR integration record](../design/inference/history/memorilla/2026-11-09-pr-integration.md)
collects the validation results and qualification limits. The
[reference-tool guide](../../zig/pkg/inference/scripts/memorilla/README.md)
documents checkpoint conversion, numerical verification and repeatable QA.

## Implemented scope

- Pinned Memorilla source `d8f18891a50bf33c18f04fe5167c22f51e58c9cd`,
  released memory modules, Qwen3-Embedding-4B and Qwen3-8B.
- Explicit trained F32 CUDA profile and F32 KV on GB10/SM121. Mixed precision,
  F16 KV and other devices require their own numerical qualification.
- Query-dependent readout into sixteen decoder input embeddings. Base weights
  and the decoder vocabulary are unchanged; memory tokens are not text or KV.
- Opt-in `POST /ai/v1/memorilla/generate` for a question and one through 64
  supplied document embeddings, with bounded greedy generation.
- Optional source-verified answers using supplied document text, a checked
  fallback or abstention. Caller-supplied alignment is explicit.
- Managed identities, finite admission, physical accounting, cancellation,
  artifact drift rejection, isolated request state and decoder-owned reader
  residency. The database gateway retains inference authorization.
- Independent component/composed numerical oracles, managed lifecycle and HTTP
  contract checks, frozen QA fixtures, and Transformers/vLLM comparisons.

## Qualification limits

The released reader misses all 22 exact novel operational-fact cases in the
native and Python diagnostics, while handling both missing-fact cases. This
blocks exact latent-only factual QA rollout. Numerical compatibility and a
matching wrong answer do not establish production answer quality.

The source-verified path passes the bounded development and held-out checks
recorded in the serving guide. These results do not establish accuracy on an
operator's corpus, sustained throughput or general device support. Evaluate
representative questions and capacity before enabling a deployment.

Warm short-answer latency on GB10 is near parity with the Python controls.
The small measured median differences do not establish a general speedup;
vLLM also uses a different precision profile. Document ingestion and database
retrieval are outside the measured supplied-feature generation path.

Experimental reader training and one-off diagnostic probes are outside this
serving PR. Their failed adaptation results remain historical evidence; no
candidate reader or native BF16 profile is promoted.

## Remaining integration work

1. Bind collection reads to authenticated source owners, current document
   revisions, retention/deletion state and accepted embedding publications.
   Supplied vectors alone do not prove source authority or complete coverage.
2. Start with exact scans of explicitly bounded collections. Label incomplete
   coverage and make approximate selection a separate mode. Recheck authority
   and freshness before model work and response publication.
3. Measure ingestion, selection/read, question encoding, memory readout, prefill
   and decode separately and end to end. Compare against native retrieval and
   full context using the same models, GPU and workload.
4. Establish a useful latent-only QA target before training new readers. Freeze
   independent development/final evaluation before observing outputs. Preserve
   released assets and qualify each adapted reader as a separate artifact.
5. Treat EmbeddingGemma/Gemma readers as separate model-pair and prompt
   contracts, with independent training, numerical and answer-quality gates.
