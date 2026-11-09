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
  passes 404 tests with no skips, including compiled CPU verifier integrations.
  Generated-source, license, formatting and generated error-identity checks pass.

Mixed encoder arithmetic and F16 KV exceeded the frozen F32 numerical gates.
They remain unqualified; a different reference precision does not erase those
failures. The conditional composed oracle does not independently qualify the
encoder or authenticate supplied document features.

## Answer quality

The frozen Stage3 latent-only diagnostic scores **0/22 exact novel facts** and
**2/2 missing-fact answers** in native and Python. The matched F32 run produces
identical output sequences on all 24 cases. This blocks exact latent-only
operational QA rollout even though numerical compatibility passes.

The bounded source-verified assessment passes 45 development cases, 48 held-out
cases and 24 repeated sanity cases. These checks support the exercised path;
they do not establish production corpus accuracy, source truth or database
authorization. Caller-supplied alignment remains explicit.

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

Both fixtures, the frozen scorer and all other retained QA/template fixtures
are unchanged. Model weights, environments and generated benchmark receipts
are excluded from Git. Detailed dated records remain in Git at `7b0d1495d3`;
the PR retains this consolidated summary instead of the session history folder.

Local integration receipts are under
`.benchmark-results/memorilla/pr-preparation-20261109/`. Their
`validation-receipts.json` SHA256 is
`27d1309c739a9b110810526ed7392c824df2812d22491243bf1b10e8837d7e83`.
The tested CLI SHA256 is
`9ba8cccc0ac78e4a677934a0dfa64d9774716402aceaa6957fe9a9c141b3effd`;
the composed verifier SHA256 is
`d58933c535fe3d4bf2432854f2875659c8316346b4f476218170012df829e559`.
These receipts predate the scope cleanup; serving/CUDA execution sources were
unchanged. Workload quality and capacity still require deployment evaluation.
