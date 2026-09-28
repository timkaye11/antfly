# Harden Gemma4 E2B LoRA SFT/DPO and GRPO validation

Fix retained Metal resources and partial Q/K/V execution that caused memory
growth during Gemma4 LoRA evaluation. Harden checkpoint publication and recovery
verification, recognize all model-configured GRPO stop tokens, and add reproducible
SFT preparation, generation, and quality scoring. Adjust the macOS CLI compiler
reservation to cover the observed build peak.

Validation on E2B:

- Three-seed SFT and DPO development campaigns each complete 1,960 updates per
  seed. Multi-token SFT mean exact match improves from 54.30% to 69.01%, and
  mean token F1 from 78.18% to 83.78%; all three reloads are exact.
- DPO mean evaluation loss improves from 0.69315 to 0.53643, with 75.78%
  preference-margin accuracy for each seed.
- 833 Python contracts and 14 CI-selection checks pass. Debug, ReleaseSafe,
  and ReleaseFast each pass 466 of 469 selected Metal tests, with three optional
  fixture skips. The public CLI build and compiler-reservation test pass.
- E2B merged export, CLI, and HTTP serving comparisons pass their recorded
  workloads. Complete SFT memory traces remain bounded across three roughly
  88-minute runs.

This PR is for review, not production promotion. GRPO remains experimental:
compiled/eager generation agrees on the diagnostic fixture, but the broader
compiled training probe exceeds the unchanged swap-growth guard before
optimizer integration can be verified. Full-scope recovery, GRPO quality,
HF-CUDA/native/Metal oracle and PEFT numerical checks, clean-host performance,
sealed acceptance, and hosted CI on the submitted revision remain open.
E4B qualification is deferred.

Detailed scope and retained evidence are documented in
`zig/pkg/inference/finetuning/GEMMA4_REVIEW_REMEDIATION.md`.
