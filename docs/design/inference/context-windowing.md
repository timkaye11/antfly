# Context windowing and Metal compaction

Context policy separates the model's positional limit from retained KV storage.
Compaction reduces retained storage; it does not extend the model's position
range, remove prompt tokens from usage, or authorize an approximate serving
route. Release registries and source-bound campaign receipts determine which
artifact, device, encoding, and request range may use compaction.

## Request contract

```json
{
  "model": "google/gemma-4-E2B-it-qat-q4_0-gguf",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 256,
  "context_policy": {
    "mode": "exact",
    "max_context_tokens": 8192,
    "max_kv_tokens": 8192
  }
}
```

The CLI equivalents are `--context-mode exact`, `--context-window 8192`, and
`--max-kv-tokens 8192`. Admission reserves continuation, speculative positions,
and media expansion within the artifact's limit. Exact mode must fit the whole
request and never truncates global attention to meet a smaller KV budget.

`rolling` and `attention_matching` require an explicit KV budget. Policy defaults
protect 256 prefix tokens and 2048 recent tokens; these counts do not establish
message or system-prompt boundaries. Explicit `keep_prefix_tokens: 0` remains
distinct from an omitted value. Approximate routes also require their execution
and release gates to pass; recognizing a policy in the schema is insufficient.

Attention-matching payload selection uses zero-based
`context_policy.compaction_targets` entries containing `message_index` and
`part_index`, or repeatable CLI `--compact-part MESSAGE:PART`. String content is
part zero. Selected user/tool text is eligible; system/developer/assistant
content, special tokens, tool-call metadata, delimiters, boundary-crossing
tokens, and protected intervals remain exact. Rendering and tokenization retain
byte/token provenance without substring matching or inserted markers.

A request cannot combine `context_policy` with `cache_compaction_ratio`. Legacy
ratio `1` is exact identity; approximate legacy ratios are rejected before the
host prototype. The shared validation implementation is
`zig/pkg/inference/src/runtime/context_policy.zig`.

## Compaction transaction

The native checkpoint path ingests the original prompt exactly, prepares a
candidate in an isolated execution scope, validates it, and publishes all owners
atomically. It must preserve donor/shared-KV relationships and exact local
attention rings, including their physical capacities and continuation budgets.

The implementation has these invariants:

- Training and held-out queries remain disjoint and cover the relevant donor
  consumers and GQA heads. Validation cannot select or tune the candidate.
- Fitting and publication charge source KV, unpublished destination storage,
  query capture, workspace, and continuation reservation. Free slots within an
  unchanged arena do not demonstrate physical memory savings.
- Stored encoding is part of validation and release identity. F16, FP8, and
  rotated4 require their own evidence; a passing encoding does not qualify
  another encoding or backend.
- Publication waits for readers and device completion. Cancellation, allocation
  failure, nonfinite values, rejected quality, and GPU errors preserve request
  isolation and release unpublished resources.
- An admitted exact-remainder fallback is a successful continuation, not a
  successful compaction. It may use only the existing physical reservation.
- Native, CLI, HTTP, streaming, export/import, and appended continuation must
  retain the same request ownership and cleanup rules. Scheduler batching,
  speculation, media, and repeated compaction require separate coverage.

The main owners are `backends/packed_context_storage.zig`,
`backends/packed_context_session.zig`, `backends/metal_context_session.zig`,
`runtime/kv/context_session.zig`, and the native checkpoint paths in
`pipelines/generation.zig`. `runtime/kv/attention_match_release.zig` binds release
admission to the qualified identity and request range. An environment profile,
offline fit, or passing smoke test does not create a release certificate.

## Calibration and corpus integrity

Compaction collects real model queries within the admitted request-owned
execution scope. The native fit and independent stored-format validation remain
part of the transaction; training and held-out queries stay disjoint.

The current corpus tools validate sealed source families, rendered bytes,
token IDs, byte/token offsets, payload protection, and loaded tokenizer/template
identity before model execution. `quality_corpus_prepare_v2.py` owns the coverage
receipt checks, while `quality_ordered_messages.py` owns structured-request
identity. These are input-integrity checks and grant no release authority.

Historical offline Python fitters, repeat-fitting collectors, stored-trace
replays, and their schema-specific release collectors are retired. Their local
archives are retained with the cleanup manifest for reproducing earlier work.

## Validation and artifact placement

The reusable qualification tools live in `zig/pkg/inference/scripts/gemma4`:

- `record_context_release_build.py` records complete Git-visible Zig inputs,
  binary identity, and build provenance.
- `qualify_native_checkpoint_http.py` checks compaction, transport, fresh-process
  import, resumed generation, and appended continuation through HTTP.
- `qualify_metal_native_checkpoint.py` launches serial source-bound campaign
  phases. Profiles select runtime controls without granting release authority.
- `qualify_native_checkpoint.py` provides `freeze`, `smoke`, `quality`,
  `performance`, `cycles`, `fault`, and `soak` phases under the declared campaign
  contracts. For example, run `python3 zig/pkg/inference/scripts/gemma4/qualify_native_checkpoint.py quality --help`.

Use each tool's `--help` for its required sealed inputs. See
[the quality corpus contract](gemma4-context-quality-corpus.md) for corpus
construction and evaluation provenance. Keep matched exact baselines and bind
results to model/tokenizer/template, source, binary, runtime configuration,
device/OS, encoding, context length, and route. Unit tests, model-path smokes,
performance comparisons, protected quality evaluation, and serving soaks are
separate evidence. None substitutes for the others.

Commit small, portable regression fixtures and the scripts that consume them.
Store generated prompts, tokens, tensor dumps, checkpoints, binaries, temporary
source copies, local receipts, and historical experiment notes beneath ignored
`.qualification/`. Keep result/resource summaries and identities needed to audit
a retained run. Draft implementations and run-by-run handoff records belong in
local research artifacts rather than the production source tree or design guide.
