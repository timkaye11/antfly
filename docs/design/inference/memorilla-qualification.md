# Memorilla qualification

The opt-in [supplied-feature API](memorilla-serving.md) is disabled by default.
Qualification covers pinned Qwen3-Embedding-4B, released Memorilla readers and
Qwen3-8B with F32 arithmetic/F32 KV on NVIDIA GB10 (SM121). It does not qualify
exact latent-only operational QA, other devices, native BF16, database collection
retrieval or a production corpus.

## Numerical and serving checks

- All five released readers pass the explicit `trained-f32-v1` component gates
  on CPU and CUDA. Exact encoder/decoder component comparisons also pass. The
  unchanged default strict gate exposed trained-weight reduction differences;
  the independent F64 analysis and both profiles are documented in
  [PRECISION.md](../../../zig/pkg/inference/scripts/memorilla/PRECISION.md).
- The managed Stage3 encoder/reader/decoder passes the conditional oracle over
  all 151,936 prefill logits and three cached steps. Logit limits remain
  `atol=rtol=1e-3`; final memory/input embeddings use `atol=5e-5`, `rtol=1e-5`.
- Cancellation/recovery, exact A/B/A isolation, the 64-document/128-token resource
  case, physical accounting, reader pinning and eviction/refund checks pass.
- Live HTTP checks pass profile/concurrency isolation, cold limit rejection
  before GPU loading, static/in-flight artifact drift rejection and deadline
  recovery. All 48 repeated native answer sequences match their baseline;
  the source-verified HTTP contract passes 30 requests.
- The serving qualification ran 114 focused CUDA tests with no skips and 127
  gateway tests with one model-dependent ASR skip, with no failures or leaks.
  After removing experimental development tools, the retained Python suite
  passes 408 tests with no skips, including compiled CPU verifier integrations.
  Generated-source, license, formatting and generated error-identity checks pass.
- The source-reliability changes pass 64 focused native tests with CUDA available
  and no skips, plus the generated Go, Python and TypeScript SDK checks.

Mixed encoder arithmetic and F16 KV exceeded the frozen F32 numerical gates.
They remain unqualified; a different reference precision does not erase those
failures. The conditional composed oracle does not independently qualify the
encoder or authenticate supplied document features.

## Answer quality

The frozen Stage3 latent-only diagnostic scores **0/22 exact novel facts** and
**2/2 missing-fact answers** in native and Python. The matched F32 run produces
identical output sequences on all 24 cases. This blocks exact latent-only
operational QA rollout even though numerical compatibility passes.

The revised source-verified assessment passes 40 frozen development cases,
48 held-out factual cases and 24 repeated sanity cases. An independent audit
confirms exact, case-sensitive source values or correct abstentions across all
112 frozen-run answers, with zero wrong published answers.
These checks support the exercised path;
they do not establish production corpus accuracy, source truth or database
authorization. Caller-supplied alignment remains explicit.

The source reliability checks additionally require an exact, bounded occurrence
of the answer in source text before semantic verification. A completed rejection
permits one source-only repair and one fresh check. Invalid or truncated verdicts,
missing literal evidence and unsuccessful repairs abstain. This is conservative
extractive QA: paraphrases and derived values without an exact source occurrence
are outside its supported answer boundary. The semantic check uses the same
decoder and remains fallible; literal evidence is necessary, not proof of truth.

The authored mixed-instruction development diagnostic is runnable through the
existing verified evaluator. Its gold labels stay outside inference requests,
exact names/codes are scored without case folding, and blanket abstention fails
answerable cases. Its diagnostic receipt cannot authorize held-out evaluation.
The revised native build passes all seven controls. The frozen development run
also passes **40/40** with exact source values; all 40 answer texts match the
Python F32 source-control run after trimming outer whitespace.

A separate 16-case authored development challenge exposed two wrong published
answers in the previous build: an injected name and an incomplete shared surname.
The revised native path passes **16/16**, with **11/11 exact factual answers**,
**5/5 correct abstentions**, and **zero wrong published answers**. This establishes
the exercised regression boundary, not general production accuracy.

Official vLLM 0.9.1 and its matched BF16 Transformers control both score 20/32
on the bounded TriviaQA sample. All 56 prompt sequences and 52/56 output
sequences match across the combined operational/TriviaQA diagnostic. This is
not a full released-benchmark reproduction or native BF16 qualification.
Experimental reader adaptation failed its development target and is excluded
from this serving PR; no candidate reader is promoted.

## Performance

Warm short-answer comparisons on GB10 report medians of **0.819–0.821 s native
F32**, **0.826 s Transformers F32**, and **0.823 s official vLLM BF16**.
The integrated native build measures 0.826 s over 48 warm requests, with other
build checks running on the host; that run is an integration sanity measurement,
not a fresh paired Python comparison.

Timing includes question encoding, memory readout, decoder prefill and greedy
generation using supplied document vectors. Document embedding production,
database retrieval, cold model loading and diagnostic tensor capture are
excluded. The small median differences and intervals crossing zero support
near parity, not a general speedup claim. Native/vLLM precision differs and
40/48 output sequences match, so bitwise generation parity is not established.

The revised source-verified HTTP path is a different workload. On the 16-case
development challenge, the 15 warm requests have a **24.266 s median**; the two
successfully repaired answers have a **38.173 s median**. These timings include
serving integrity checks, independent source answering and model verification;
document embedding production and database retrieval are excluded. The sample
is small and has no matched Python HTTP comparison. The earlier ~0.82 s
latent-only pipeline timing does not describe this reliability path.

## Reproduction and retained test data

The [tool guide](../../../zig/pkg/inference/scripts/memorilla/README.md) documents
checkpoint conversion, independent references, native numerical verifiers,
HTTP/QA checks and the Transformers/vLLM comparison runners.

- `fixtures/encoder-tokenizer.json` is a 3.3 KB golden for four encoder inputs.
  It checks exact IDs, trailing EOS, model revision and tokenizer sidecars.
- `fixtures/factual_v2.json` contains 96 authored source-verified QA cases:
  32 scoring/adapter-smoke cases, 16 development cases and 48 held-out cases.
  The scorer and evaluator use its facts, counterfactual groups, contradictions
  and injection controls. Its 5,311 lines are test inputs and labels, not run logs.
  Removing it would remove the default QA corpus and its regression coverage.

Both fixtures and the frozen scorers are unchanged. The five existing mixed-
instruction development controls are preserved, with two observed regressions
added for injected answers and incomplete names. Model weights, environments and
generated benchmark receipts are excluded from Git. Detailed dated records remain
in Git at `7b0d1495d3`; the PR retains this consolidated summary instead of the
session history folder.

Source-reliability receipts are under
`.benchmark-results/memorilla/source-reliability-20261109/`. Their
`validation-receipts.json` SHA256 is
`386198a18c35d16700ba1e658b83ff37ca16cc4c27f7f312f61c89112118ec4d`.
The tested CLI SHA256 is
`0c5f8873e3635c675cdc58bb9f496a3a6a642c83142f1bb1c27189e92a03de43`.
Numerical and paired-engine receipts remain under
`.benchmark-results/memorilla/pr-preparation-20261109/`; their validation
manifest SHA256 is
`27d1309c739a9b110810526ed7392c824df2812d22491243bf1b10e8837d7e83`.
The unchanged composed-verifier SHA256 is
`d58933c535fe3d4bf2432854f2875659c8316346b4f476218170012df829e559`.
Workload quality and capacity still require deployment evaluation.
