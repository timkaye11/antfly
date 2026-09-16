# Gemma 4 historical evidence index

This file is a compact index of historical Gemma 4 experiments. Historical
results do not qualify the current source, and old measurements must not be
reused after a renderer, recipe, kernel, model, or evidence-contract change.

The current implementation contract is in [GEMMA4.md](GEMMA4.md), the external
oracle contract is in [GEMMA4_ORACLE.md](GEMMA4_ORACLE.md), and review fixes and
validation status are in [GEMMA4_REVIEW_REMEDIATION.md](GEMMA4_REVIEW_REMEDIATION.md).

## Evidence policy

- Treat a result as admissible only when its report binds the source revision,
  model/tokenizer/data identities, prepared-input digest, environment policy,
  binary identity, and output artifact hashes.
- Treat `/private/tmp` and other temporary roots as diagnostic evidence unless
  an immutable publication manifest explicitly promotes the artifacts.
- A successful build or finite loss is an implementation check. Production
  qualification additionally requires the declared quality, parity, resume,
  memory, and holdout gates.
- Old experiments are retained for context, regression investigation, and
  performance archaeology. They are not release evidence for a new source.

## Milestones

The branch history contains the complete dated notebook that originally lived
in this file. The principal milestones are:

1. Gemma 4 BF16 Metal SFT foundation and strict device execution.
2. Packed Q4/Q6 frozen-linear backward coverage and adapter publication.
3. Native/Metal DPO and GRPO recipe, checkpoint, and resume contracts.
4. MLX-first numerical and performance harnesses with locked model/source
   provenance.
5. Gemma 4 renderer, paged-KV, fused-attention, and optimizer parity work.
6. Review remediation: explicit training dispatch, v6 preference fingerprints,
   fail-closed GRPO/CCE paths, CI test-selection auditing, and serving parity.

## Current evidence boundary

The retained September 15 Metal-versus-MLX E2B/E4B capture is diagnostic
evidence for its recorded source revision. E4B passed its configured single-seed
quality gate; E2B failed the reward-improvement gate in both backends. GRPO
therefore remains experimental pending a fresh multi-seed, longer-horizon run
and sealed holdout evaluation. The capture is not a production qualification
for later source changes.

The current local review evidence is retained under
`.benchmark-assets/gemma4-review-20260916/`. It covers implementation and
contract tests, not a replacement full-model qualification campaign.

## Recovering the full notebook

The removed detailed notebook remains available in repository history. To
inspect it without restoring it into the working tree:

```bash
git show 334fed88151e4d8eb3b08ff641943e9bef8b7746^:zig/pkg/inference/docs/finetuning/GEMMA4.md
```

Use the immutable reports and manifests referenced by those historical notes
for exact numbers. Do not copy old temporary paths or hashes into a new release
claim.
