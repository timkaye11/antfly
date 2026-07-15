#!/usr/bin/env python3
"""Validate exact tokens and paired throughput for a CUDA kernel candidate."""

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import math
import os
import pathlib
import re
import statistics
import subprocess


TOKEN_IDS_RE = re.compile(r"^token_ids:(?P<ids>(?:\s+-?\d+)*)\s*$", re.MULTILINE)
ENVIRONMENT_VARIABLE_RE = re.compile(r"[A-Z_][A-Z0-9_]*")
DEFAULT_PROMPTS = (
    "Write one sentence about ants.",
    "Explain why the sky is blue in two sentences.",
    "List three prime numbers and nothing else.",
)


class CandidateKind(enum.Enum):
    GENERATED_ATTENTION = "generated-attention"
    Q4_0_Q8_1_LM_HEAD_ARGMAX = "q4-0-q8-1-lm-head-argmax"
    Q6_K_Q8_1_LM_HEAD_ARGMAX = "q6-k-q8-1-lm-head-argmax"
    Q4_0_Q8_1_E2B_FFN = "q4-0-q8-1-e2b-ffn"
    Q4_0_E2B_FFN_EXACT = "q4-0-e2b-ffn-exact"

    def __str__(self) -> str:
        return self.value


@dataclasses.dataclass(frozen=True)
class RouteCounter:
    name: str
    label: str


@dataclasses.dataclass(frozen=True)
class CandidateSpec:
    kernel_id: str
    environment_variable: str
    required_route_counters: tuple[RouteCounter, ...]
    forbidden_route_counters: tuple[RouteCounter, ...] = ()
    legacy_kind: CandidateKind | None = None
    requires_explicit_model: bool = False
    fixed_comparison_environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.kernel_id or not re.fullmatch(r"[a-z0-9][a-z0-9._/-]*", self.kernel_id):
            raise ValueError("candidate kernel ID must be a non-empty lowercase catalog ID")
        if not self.environment_variable or not ENVIRONMENT_VARIABLE_RE.fullmatch(self.environment_variable):
            raise ValueError("candidate environment variable must be a valid environment name")
        if not self.required_route_counters:
            raise ValueError("candidate must require at least one route counter")
        names = tuple(counter.name for counter in self.required_route_counters)
        if len(set(names)) != len(names):
            raise ValueError("candidate route counters must be unique")
        forbidden_names = tuple(counter.name for counter in self.forbidden_route_counters)
        if len(set(forbidden_names)) != len(forbidden_names):
            raise ValueError("candidate forbidden counters must be unique")
        if set(names) & set(forbidden_names):
            raise ValueError("required route counters and forbidden counters must be disjoint")
        for counter in self.required_route_counters + self.forbidden_route_counters:
            if not counter.name or not counter.label:
                raise ValueError("route counter names and labels must be non-empty")
        fixed_names = tuple(name for name, _ in self.fixed_comparison_environment)
        if len(set(fixed_names)) != len(fixed_names):
            raise ValueError("fixed comparison environment variables must be unique")
        if self.environment_variable in fixed_names:
            raise ValueError("fixed comparison environment must not override the selected candidate gate")
        for name, value in self.fixed_comparison_environment:
            if not ENVIRONMENT_VARIABLE_RE.fullmatch(name):
                raise ValueError("fixed comparison environment variable name is invalid")
            if any(ord(character) < 32 or ord(character) == 127 for character in value):
                raise ValueError("fixed comparison environment value contains an unsupported control character")

    @property
    def route_counter(self) -> str:
        """Retain the primary-counter interface used by existing JSON consumers."""
        return self.required_route_counters[0].name

    @property
    def catalog_id(self) -> str:
        return self.kernel_id

    @property
    def route_counters(self) -> tuple[RouteCounter, ...]:
        """Compatibility alias for pre-catalog callers."""
        return self.required_route_counters

    @property
    def candidate_forbidden_counters(self) -> tuple[RouteCounter, ...]:
        """Compatibility alias for pre-catalog callers."""
        return self.forbidden_route_counters

    @property
    def kind(self) -> CandidateKind | None:
        """Compatibility alias for legacy enum-backed candidates."""
        return self.legacy_kind


@dataclasses.dataclass(frozen=True)
class TimingMetadata:
    counter_group: str = "cuda"
    throughput_field: str = "decode_tok_per_s"
    throughput_unit: str = "tokens_per_second"
    require_persistent_replay: bool = True
    persistent_replay_counter: str = "graph_capture_persistent_replays"
    graph_discard_counter: str = "graph_capture_discards"
    graph_capacity_skip_counter: str = "graph_capture_capacity_skips"

    def __post_init__(self) -> None:
        for field_name in (
            self.counter_group,
            self.throughput_field,
            self.throughput_unit,
            self.persistent_replay_counter,
            self.graph_discard_counter,
            self.graph_capacity_skip_counter,
        ):
            if not field_name:
                raise ValueError("timing metadata fields must be non-empty")


DEFAULT_TIMING_METADATA = TimingMetadata()


GENERATED_ATTENTION_KERNEL_ID = "cuda.attention.gqa.decode.generated"
SCORE_PREWORK_ATTENTION_KERNEL_ID = "cuda.attention.gqa.decode.score_prework"
Q4_0_Q8_1_LM_HEAD_ARGMAX_KERNEL_ID = "cuda.quant.q4_0-q8_1.lm_head.argmax"
Q6_K_Q8_1_LM_HEAD_ARGMAX_KERNEL_ID = "cuda.quant.q6_k-q8_1.lm_head.argmax"
Q4_0_Q8_1_FFN_KERNEL_ID = "cuda.quant.q4_0-q8_1.ffn.generated"
Q4_0_E2B_FFN_EXACT_KERNEL_ID = "cuda.quant.q4_0.ffn.e2b.f32_exact"
GENERATED_ATTENTION_SPLIT_KV_MIN_TOKENS_ENV = "ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_SPLIT_KV_MIN_TOKENS"
GENERATED_ATTENTION_SPLIT_KV_SPLITS_ENV = "ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_SPLIT_KV_SPLITS"
DEFAULT_CAPTURE_KV_PROMPT_HEADROOM = 1024
SUPPORTED_CACHE_DTYPES = ("f16", "f32", "int8", "fp8", "int4", "polar4", "turbo3")
CAPTURE_KV_CAPACITY_ENV = "ANTFLY_CAPTURE_FORCE_KV_CAPACITY"
RUNTIME_CAPTURE_KV_CAPACITY_ENV = "ANTFLY_INFERENCE_CUDA_CAPTURE_FORCE_KV_CAPACITY"
CAPTURE_KV_CAPACITY_ENVIRONMENT_VARIABLES = frozenset(
    (CAPTURE_KV_CAPACITY_ENV, RUNTIME_CAPTURE_KV_CAPACITY_ENV)
)

# The normal Gemma QAT tuning profile enables Q8_1 pair/down paths. This
# candidate instead mirrors the raw-Q4_0 F32 pair-activation and tile4 down
# kernels, so both validator arms must use that same F32 route.
EXACT_E2B_FFN_F32_COMPARISON_ENVIRONMENT = (
    ("ANTFLY_INFERENCE_CUDA_Q4_0_GATE_UP_ACTIVATION_Q8_1_PRECOMPUTE", "0"),
    ("ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN", "0"),
    ("ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_CATALOG_FFN_CANDIDATES", "0"),
    ("ANTFLY_INFERENCE_CUDA_Q4_0_PAIR_ACTIVATION_Q8_1_DP4A", "0"),
    ("ANTFLY_INFERENCE_CUDA_Q4_0_LINEAR_Q8_1_DP4A", "0"),
    ("ANTFLY_INFERENCE_CUDA_Q4_0_GATED_DOWN_Q8_1_DP4A", "0"),
    ("ANTFLY_INFERENCE_CUDA_Q4_0_LINEAR_TILE8", "0"),
    ("ANTFLY_INFERENCE_CUDA_Q4_0_LINEAR_TILE4_W4", "0"),
    ("ANTFLY_INFERENCE_CUDA_DISABLE_GENERATED_Q4_0_MMV", "1"),
    ("ANTFLY_INFERENCE_CUDA_DEQUANTIZE_Q4_0_MATRIX_WEIGHTS_BF16", "0"),
    ("TERMITE_CUDA_DEQUANTIZE_QUANT_WEIGHTS", "0"),
)

SCORE_PREWORK_ATTENTION_COMPARISON_ENVIRONMENT = (
    ("ANTFLY_CUDA_DISABLE_TURBOQUANT_COMPRESSED_V", "1"),
    ("ANTFLY_INFERENCE_CUDA_TURBOQUANT_MIN_TOKENS", "0"),
    ("ANTFLY_INFERENCE_CUDA_TURBOQUANT_SPLIT_ATTENTION", "0"),
    ("ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_DECODE", "0"),
)


CANDIDATE_CATALOG = {
    GENERATED_ATTENTION_KERNEL_ID: CandidateSpec(
        kernel_id=GENERATED_ATTENTION_KERNEL_ID,
        environment_variable="ANTFLY_GENERATED_ATTENTION_DECODE",
        required_route_counters=(RouteCounter("launch_attention_gqa_decode_generated", "generated attention"),),
        legacy_kind=CandidateKind.GENERATED_ATTENTION,
    ),
    SCORE_PREWORK_ATTENTION_KERNEL_ID: CandidateSpec(
        kernel_id=SCORE_PREWORK_ATTENTION_KERNEL_ID,
        environment_variable="ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_SCORE_PREWORK",
        required_route_counters=(
            RouteCounter("launch_attention_gqa_decode_score_prework", "generated score-prework attention"),
        ),
        forbidden_route_counters=(
            RouteCounter("launch_attention_gqa_decode_fast_fallbacks", "score-prework attention fallbacks"),
        ),
        fixed_comparison_environment=SCORE_PREWORK_ATTENTION_COMPARISON_ENVIRONMENT,
    ),
    Q4_0_Q8_1_LM_HEAD_ARGMAX_KERNEL_ID: CandidateSpec(
        kernel_id=Q4_0_Q8_1_LM_HEAD_ARGMAX_KERNEL_ID,
        environment_variable="ANTFLY_INFERENCE_CUDA_Q4_0_LM_HEAD_Q8_1_ARGMAX",
        required_route_counters=(RouteCounter("lm_head_argmax_fused_q4_0_q8_1", "Q4_0 x Q8_1 LM-head argmax"),),
        forbidden_route_counters=(
            RouteCounter("lm_head_argmax_q4_0_q8_1_fallbacks", "Q4_0 x Q8_1 LM-head argmax fallbacks"),
        ),
        legacy_kind=CandidateKind.Q4_0_Q8_1_LM_HEAD_ARGMAX,
    ),
    Q6_K_Q8_1_LM_HEAD_ARGMAX_KERNEL_ID: CandidateSpec(
        kernel_id=Q6_K_Q8_1_LM_HEAD_ARGMAX_KERNEL_ID,
        environment_variable="ANTFLY_INFERENCE_CUDA_GENERATED_Q6_K_Q8_1_LM_HEAD_ARGMAX",
        required_route_counters=(
            RouteCounter("lm_head_argmax_generated_q6_k_q8_1_hits", "generated Q6_K x Q8_1 LM-head argmax"),
        ),
        forbidden_route_counters=(
            RouteCounter("lm_head_argmax_generated_q6_k_q8_1_fallbacks", "generated Q6_K x Q8_1 LM-head argmax fallbacks"),
        ),
        legacy_kind=CandidateKind.Q6_K_Q8_1_LM_HEAD_ARGMAX,
        requires_explicit_model=True,
    ),
    Q4_0_Q8_1_FFN_KERNEL_ID: CandidateSpec(
        kernel_id=Q4_0_Q8_1_FFN_KERNEL_ID,
        environment_variable="ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN",
        required_route_counters=(
            RouteCounter("q4_0_generated_e2b_pair_q8_hits", "Q4_0 x Q8_1 generated E2B FFN pair route"),
            RouteCounter("q4_0_generated_e2b_down_q8_hits", "Q4_0 x Q8_1 generated E2B FFN down route"),
        ),
        forbidden_route_counters=(
            RouteCounter("q4_0_generated_e2b_pair_q8_fallbacks", "Q4_0 x Q8_1 generated E2B FFN pair fallbacks"),
            RouteCounter("q4_0_generated_e2b_down_q8_fallbacks", "Q4_0 x Q8_1 generated E2B FFN down fallbacks"),
        ),
        legacy_kind=CandidateKind.Q4_0_Q8_1_E2B_FFN,
    ),
    Q4_0_E2B_FFN_EXACT_KERNEL_ID: CandidateSpec(
        kernel_id=Q4_0_E2B_FFN_EXACT_KERNEL_ID,
        environment_variable="ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN_EXACT",
        required_route_counters=(
            RouteCounter("q4_0_generated_e2b_exact_pair_f32_hits", "exact F32 generated E2B FFN pair route"),
            RouteCounter("q4_0_generated_e2b_exact_down_f32_hits", "exact F32 generated E2B FFN down route"),
        ),
        forbidden_route_counters=(
            RouteCounter(
                "q4_0_generated_e2b_exact_pair_f32_fallbacks",
                "exact F32 generated E2B FFN pair fallbacks",
            ),
            RouteCounter(
                "q4_0_generated_e2b_exact_down_f32_fallbacks",
                "exact F32 generated E2B FFN down fallbacks",
            ),
        ),
        legacy_kind=CandidateKind.Q4_0_E2B_FFN_EXACT,
        fixed_comparison_environment=EXACT_E2B_FFN_F32_COMPARISON_ENVIRONMENT,
    ),
}

CANDIDATE_SPECS = {
    spec.legacy_kind: spec
    for spec in CANDIDATE_CATALOG.values()
    if spec.legacy_kind is not None
}

DEFAULT_CANDIDATE = CANDIDATE_SPECS[CandidateKind.GENERATED_ATTENTION]


def parse_args() -> argparse.Namespace:
    repo = pathlib.Path(__file__).resolve().parents[4]
    inference = repo / "zig/pkg/inference"
    parser = argparse.ArgumentParser(description=__doc__)
    candidate_group = parser.add_mutually_exclusive_group()
    candidate_group.add_argument(
        "--candidate",
        type=CandidateKind,
        choices=tuple(CandidateKind),
        default=None,
        help=(
            "legacy candidate alias; generated-attention remains the default, "
            "q4-0-q8-1-lm-head-argmax tests the generated E2B LM-head route, and "
            "q6-k-q8-1-lm-head-argmax tests generated Q6_K LM-head stage 1, and "
            "q4-0-q8-1-e2b-ffn tests the generated Q8-intermediate E2B FFN pair and down routes, and "
            "q4-0-e2b-ffn-exact tests the exact F32 E2B FFN pair and down routes"
        ),
    )
    candidate_group.add_argument(
        "--kernel-id",
        "--catalog-id",
        "--catalog-kernel-id",
        dest="kernel_id",
        default=None,
        help="model-neutral catalog kernel ID; uncatalogued IDs require an environment variable and route counters",
    )
    parser.add_argument(
        "--candidate-environment-variable",
        "--environment-variable",
        dest="candidate_environment_variable",
        default=None,
        help="candidate gate environment variable (required for an uncatalogued kernel ID)",
    )
    parser.add_argument(
        "--candidate-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "candidate-only runtime environment override; repeatable. The selected candidate gate "
            "cannot be overridden. Use this to pin experimental schedules such as "
            f"{GENERATED_ATTENTION_SPLIT_KV_MIN_TOKENS_ENV}=512 or "
            f"{GENERATED_ATTENTION_SPLIT_KV_SPLITS_ENV}=4"
        ),
    )
    parser.add_argument(
        "--common-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "runtime environment override applied identically to baseline and candidate; repeatable. "
            "The selected candidate gate and capture KV capacity cannot be overridden."
        ),
    )
    parser.add_argument(
        "--required-route-counter",
        "--required-route-counters",
        action="append",
        default=None,
        help="required counter as NAME or NAME=LABEL; repeat the option or use comma-separated values",
    )
    parser.add_argument(
        "--forbidden-route-counter",
        "--forbidden-route-counters",
        action="append",
        default=None,
        help="forbidden counter as NAME or NAME=LABEL; repeat the option or use comma-separated values",
    )
    parser.add_argument("--binary", type=pathlib.Path, default=inference / "zig-out/bin/antfly-inference")
    parser.add_argument("--wrapper", type=pathlib.Path, default=inference / "scripts/with_gemma4_qat_cuda_tuning.sh")
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        default=None,
        help=(
            "GGUF model under test; required for candidates whose quantized route is model-specific. "
            "Other candidates default to the local E2B QAT fixture."
        ),
    )
    parser.add_argument(
        "--model-label",
        default=None,
        help="stable model label recorded in results (defaults to the model filename without its suffix)",
    )
    parser.add_argument(
        "--config-label",
        default="default",
        help="stable benchmark configuration label, for example e4b-12b-sm89-f32",
    )
    parser.add_argument(
        "--cache-dtype",
        choices=SUPPORTED_CACHE_DTYPES,
        default="f32",
        help="KV cache dtype applied identically to baseline and candidate",
    )
    parser.add_argument(
        "--timing-counter-group",
        "--counter-group",
        dest="timing_counter_group",
        default=DEFAULT_TIMING_METADATA.counter_group,
        help="dot-separated object path containing route and graph counters",
    )
    parser.add_argument(
        "--timing-throughput-field",
        "--throughput-field",
        dest="timing_throughput_field",
        default=DEFAULT_TIMING_METADATA.throughput_field,
        help="dot-separated timing field containing candidate throughput",
    )
    parser.add_argument(
        "--timing-throughput-unit",
        "--throughput-unit",
        dest="timing_throughput_unit",
        default=DEFAULT_TIMING_METADATA.throughput_unit,
        help="throughput unit label recorded in result metadata",
    )
    parser.add_argument(
        "--no-require-persistent-replay",
        action="store_true",
        help="skip CUDA graph replay checks for candidates whose benchmark route is not graph replayable",
    )
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="paired repetitions per prompt and length; execution order alternates each repetition",
    )
    parser.add_argument(
        "--min-candidate-ratio",
        type=float,
        default=0.0,
        help="require every paired candidate/baseline throughput ratio to meet this value",
    )
    parser.add_argument(
        "--max-cv",
        type=float,
        default=1.0,
        help="maximum baseline and candidate throughput CV within each prompt/length case",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("/tmp/antfly-gemma4-cuda-candidate"))
    parser.add_argument(
        "--capture-kv-capacity",
        type=int,
        default=None,
        help=(
            "forced CUDA graph KV capacity for both paired runs. Defaults to the largest requested "
            f"decode length plus {DEFAULT_CAPTURE_KV_PROMPT_HEADROOM} tokens of prompt headroom"
        ),
    )
    parser.add_argument("--timeout-sec", type=int, default=360)
    args = parser.parse_args()
    if args.candidate is None and args.kernel_id is None:
        args.candidate = CandidateKind.GENERATED_ATTENTION
    q6_selected = (
        args.candidate == CandidateKind.Q6_K_Q8_1_LM_HEAD_ARGMAX
        or args.kernel_id == Q6_K_Q8_1_LM_HEAD_ARGMAX_KERNEL_ID
    )
    if args.model is None and not q6_selected:
        args.model = repo / ".models/unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"
    # run_case executes from the repository root so that the tuning wrapper has
    # a stable working directory. Resolve caller-provided launch operands here
    # while the caller's working directory is still in effect.
    args.binary = args.binary.expanduser().resolve()
    args.wrapper = args.wrapper.expanduser().resolve()
    if args.model is not None:
        args.model = args.model.expanduser().resolve()
    if args.model_label is None and args.model is not None:
        args.model_label = args.model.stem
    return args


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned[:48] or "prompt"


def nested_value(value: object, path: str) -> object | None:
    current = value
    for component in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(component)
    return current


def timing_counter(timing: dict, key: str, metadata: TimingMetadata = DEFAULT_TIMING_METADATA) -> int:
    group = nested_value(timing, metadata.counter_group)
    if not isinstance(group, dict):
        return 0
    try:
        return int(group.get(key) or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def cuda_counter(timing: dict, key: str) -> int:
    """Compatibility wrapper for callers using the original CUDA timing schema."""
    return timing_counter(timing, key)


def candidate_spec(candidate: CandidateKind | str) -> CandidateSpec:
    if isinstance(candidate, CandidateKind):
        return CANDIDATE_SPECS[candidate]
    return CANDIDATE_CATALOG[candidate]


def parse_route_counter(value: str) -> RouteCounter:
    raw = value.strip()
    if not raw:
        raise ValueError("route counter must not be empty")
    name, separator, label = raw.partition("=")
    name = name.strip()
    label = label.strip() if separator else name.replace("_", " ")
    if not name or not label:
        raise ValueError(f"invalid route counter descriptor: {value!r}")
    return RouteCounter(name, label)


def parse_environment_override(value: str, option: str) -> tuple[str, str]:
    """Parse one NAME=VALUE override without shell interpolation."""
    name, separator, configured_value = value.partition("=")
    if not separator:
        raise ValueError(f"{option} must use NAME=VALUE: {value!r}")
    if not ENVIRONMENT_VARIABLE_RE.fullmatch(name):
        raise ValueError(f"{option} variable name is invalid: {name!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in configured_value):
        raise ValueError(f"{option} value contains an unsupported control character: {name!r}")
    return name, configured_value


def parse_candidate_environment(value: str) -> tuple[str, str]:
    """Parse one candidate-only NAME=VALUE override without shell interpolation."""
    return parse_environment_override(value, "candidate environment override")


def parse_common_environment(value: str) -> tuple[str, str]:
    """Parse one baseline-and-candidate NAME=VALUE override without shell interpolation."""
    return parse_environment_override(value, "common environment override")


def parse_candidate_environment_list(values: list[str]) -> tuple[tuple[str, str], ...]:
    overrides = tuple(parse_candidate_environment(value) for value in values)
    names = tuple(name for name, _ in overrides)
    if len(set(names)) != len(names):
        raise ValueError("candidate environment variable overrides must be unique")
    return overrides


def parse_common_environment_list(values: list[str]) -> tuple[tuple[str, str], ...]:
    overrides = tuple(parse_common_environment(value) for value in values)
    names = tuple(name for name, _ in overrides)
    if len(set(names)) != len(names):
        raise ValueError("common environment variable overrides must be unique")
    return overrides


def parse_route_counter_list(values: list[str] | None) -> tuple[RouteCounter, ...] | None:
    if values is None:
        return None
    counters = []
    for value in values:
        counters.extend(parse_route_counter(part) for part in value.split(","))
    return tuple(counters)


def resolve_candidate_spec(args: argparse.Namespace) -> CandidateSpec:
    base = CANDIDATE_CATALOG.get(args.kernel_id) if args.kernel_id is not None else candidate_spec(args.candidate)
    required = parse_route_counter_list(args.required_route_counter)
    forbidden = parse_route_counter_list(args.forbidden_route_counter)
    if base is None:
        if args.candidate_environment_variable is None:
            raise ValueError("uncatalogued kernel ID requires --candidate-environment-variable")
        if not required:
            raise ValueError("uncatalogued kernel ID requires at least one --required-route-counter")
        return CandidateSpec(
            kernel_id=args.kernel_id,
            environment_variable=args.candidate_environment_variable,
            required_route_counters=required,
            forbidden_route_counters=forbidden or (),
        )
    return CandidateSpec(
        kernel_id=base.kernel_id,
        environment_variable=args.candidate_environment_variable or base.environment_variable,
        required_route_counters=required if required is not None else base.required_route_counters,
        forbidden_route_counters=forbidden if forbidden is not None else base.forbidden_route_counters,
        legacy_kind=base.legacy_kind,
        requires_explicit_model=base.requires_explicit_model,
        fixed_comparison_environment=base.fixed_comparison_environment,
    )


def resolve_candidate_environment(
    args: argparse.Namespace,
    spec: CandidateSpec,
) -> tuple[tuple[str, str], ...]:
    overrides = parse_candidate_environment_list(args.candidate_env)
    if any(name == spec.environment_variable for name, _ in overrides):
        raise ValueError(
            f"--candidate-env must not override selected candidate gate {spec.environment_variable}; "
            "use --candidate-environment-variable to select a different gate"
        )
    fixed_names = {name for name, _ in spec.fixed_comparison_environment}
    conflicting = sorted(name for name, _ in overrides if name in fixed_names)
    if conflicting:
        raise ValueError(
            "--candidate-env must not override the fixed comparison environment "
            f"({', '.join(conflicting)})"
        )
    return overrides


def resolve_common_environment(
    args: argparse.Namespace,
    spec: CandidateSpec,
    candidate_environment: tuple[tuple[str, str], ...] = (),
) -> tuple[tuple[str, str], ...]:
    overrides = parse_common_environment_list(args.common_env)
    names = {name for name, _ in overrides}
    if spec.environment_variable in names:
        raise ValueError(
            f"--common-env must not override selected candidate gate {spec.environment_variable}; "
            "use --candidate-environment-variable to select a different gate"
        )
    capture_overrides = sorted(names & CAPTURE_KV_CAPACITY_ENVIRONMENT_VARIABLES)
    if capture_overrides:
        raise ValueError(
            "--common-env must not override capture KV capacity "
            f"({', '.join(capture_overrides)}); use --capture-kv-capacity"
        )
    candidate_names = {name for name, _ in candidate_environment}
    overlapping = sorted(names & candidate_names)
    if overlapping:
        raise ValueError(
            "--common-env and --candidate-env must not override the same variable "
            f"({', '.join(overlapping)})"
        )
    fixed = dict(spec.fixed_comparison_environment)
    conflicts = sorted(
        name
        for name, value in overrides
        if name in fixed and fixed[name] != value
    )
    if conflicts:
        raise ValueError(
            "--common-env must not override the fixed comparison environment "
            f"({', '.join(conflicts)})"
        )
    return spec.fixed_comparison_environment + tuple(
        (name, value) for name, value in overrides if name not in fixed
    )


def timing_metadata_from_args(args: argparse.Namespace) -> TimingMetadata:
    return TimingMetadata(
        counter_group=args.timing_counter_group,
        throughput_field=args.timing_throughput_field,
        throughput_unit=args.timing_throughput_unit,
        require_persistent_replay=not args.no_require_persistent_replay,
    )


def timing_metadata(metadata: TimingMetadata) -> dict:
    return {
        "counter_group": metadata.counter_group,
        "throughput_field": metadata.throughput_field,
        "throughput_unit": metadata.throughput_unit,
        "require_persistent_replay": metadata.require_persistent_replay,
        "persistent_replay_counter": metadata.persistent_replay_counter,
        "graph_discard_counter": metadata.graph_discard_counter,
        "graph_capacity_skip_counter": metadata.graph_capacity_skip_counter,
    }


def candidate_environment_metadata(
    spec: CandidateSpec,
    candidate_overrides: tuple[tuple[str, str], ...],
    common_overrides: tuple[tuple[str, str], ...] = (),
) -> dict:
    configured_candidate = dict(candidate_overrides)
    configured_common = dict(common_overrides)

    def schedule_override_metadata(environment_variable: str) -> dict:
        if environment_variable in configured_candidate:
            return {
                "environment_variable": environment_variable,
                "value": configured_candidate[environment_variable],
                "source": "candidate-env",
            }
        if environment_variable in configured_common:
            return {
                "environment_variable": environment_variable,
                "value": configured_common[environment_variable],
                "source": "common-env",
            }
        inherited = os.environ.get(environment_variable)
        if inherited is not None:
            return {
                "environment_variable": environment_variable,
                "value": inherited,
                "source": "inherited-environment",
            }
        return {
            "environment_variable": environment_variable,
            "value": None,
            "source": "binary-default",
        }

    return {
        "baseline_gate": {spec.environment_variable: "0"},
        "candidate_gate": {spec.environment_variable: "1"},
        "fixed_comparison_overrides": dict(spec.fixed_comparison_environment),
        "common_overrides": configured_common,
        "candidate_overrides": configured_candidate,
        "generated_attention_split_kv_min_tokens": schedule_override_metadata(
            GENERATED_ATTENTION_SPLIT_KV_MIN_TOKENS_ENV
        ),
        "generated_attention_split_kv_splits": schedule_override_metadata(
            GENERATED_ATTENTION_SPLIT_KV_SPLITS_ENV
        ),
    }


def resolve_capture_kv_capacity(args: argparse.Namespace) -> tuple[int, str]:
    if args.capture_kv_capacity is not None:
        if args.capture_kv_capacity < 1:
            raise ValueError("capture-kv-capacity must be positive")
        return args.capture_kv_capacity, "explicit"
    if not args.lengths or any(length < 1 for length in args.lengths):
        raise ValueError("output lengths must be positive before resolving capture KV capacity")
    return max(args.lengths) + DEFAULT_CAPTURE_KV_PROMPT_HEADROOM, "max-output-plus-prompt-headroom"


def result_config_metadata(
    args: argparse.Namespace,
    metadata: TimingMetadata,
    spec: CandidateSpec | None = None,
    candidate_environment: tuple[tuple[str, str], ...] = (),
    capture_kv_capacity: tuple[int, str] | None = None,
    common_environment: tuple[tuple[str, str], ...] = (),
) -> dict:
    environment = (
        candidate_environment_metadata(spec, candidate_environment, common_environment)
        if spec is not None
        else None
    )
    return {
        "model": {
            "path": str(args.model.resolve()),
            "label": args.model_label,
        },
        # Flat aliases preserve compatibility with simple JSON/CSV consumers.
        "model_path": str(args.model.resolve()),
        "model_label": args.model_label,
        "config_label": args.config_label,
        "cache_dtype": args.cache_dtype,
        "repeats": args.repeats,
        "min_candidate_ratio": args.min_candidate_ratio,
        "max_cv": args.max_cv,
        "timing": timing_metadata(metadata),
        "candidate_environment": environment,
        "capture_kv_capacity": (
            {
                "tokens": capture_kv_capacity[0],
                "source": capture_kv_capacity[1],
            }
            if capture_kv_capacity is not None
            else None
        ),
    }


def candidate_metadata(spec: CandidateSpec) -> dict:
    legacy_kind = spec.legacy_kind.value if spec.legacy_kind is not None else None
    return {
        "kind": legacy_kind or spec.kernel_id,
        "legacy_kind": legacy_kind,
        "kernel_id": spec.kernel_id,
        "catalog_id": spec.kernel_id,
        "environment_variable": spec.environment_variable,
        "route_counter": spec.route_counter,
        "route_counters": [route.name for route in spec.required_route_counters],
        "required_route_counters": [route.name for route in spec.required_route_counters],
        "candidate_forbidden_counters": [counter.name for counter in spec.forbidden_route_counters],
        "forbidden_route_counters": [counter.name for counter in spec.forbidden_route_counters],
        "requires_explicit_model": spec.requires_explicit_model,
    }


def configure_candidate_environment(
    env: dict[str, str],
    spec: CandidateSpec,
    enabled: bool,
    overrides: tuple[tuple[str, str], ...] = (),
    common_overrides: tuple[tuple[str, str], ...] = (),
) -> None:
    env.update(common_overrides)
    env[spec.environment_variable] = "1" if enabled else "0"
    if enabled:
        env.update(overrides)


def execution_order(repetition: int) -> tuple[bool, bool]:
    """Return False for baseline and True for candidate."""
    return (False, True) if repetition % 2 == 0 else (True, False)


def repetition_stem(stem: str, repetition: int, repeats: int) -> str:
    return stem if repeats == 1 else f"{stem}-r{repetition + 1:02d}"


def validate_pair(
    baseline: dict,
    candidate: dict,
    requested_tokens: int,
    spec: CandidateSpec = DEFAULT_CANDIDATE,
    timing_info: TimingMetadata = DEFAULT_TIMING_METADATA,
) -> list[str]:
    errors = []
    baseline_ids = baseline.get("token_ids") or []
    candidate_ids = candidate.get("token_ids") or []
    if not baseline_ids or not candidate_ids:
        errors.append("missing generated token IDs")
    elif baseline_ids != candidate_ids:
        mismatch = next(
            (index for index, pair in enumerate(zip(baseline_ids, candidate_ids)) if pair[0] != pair[1]),
            min(len(baseline_ids), len(candidate_ids)),
        )
        errors.append(f"token IDs differ at index {mismatch}")
    if len(baseline_ids) != requested_tokens or len(candidate_ids) != requested_tokens:
        errors.append(
            f"expected {requested_tokens} tokens, got baseline={len(baseline_ids)} candidate={len(candidate_ids)}"
        )

    if timing_info.require_persistent_replay:
        for label, timing in (("baseline", baseline), ("candidate", candidate)):
            minimum_replays = max(1, requested_tokens - 8)
            replays = timing_counter(timing, timing_info.persistent_replay_counter, timing_info)
            if replays < minimum_replays:
                errors.append(f"{label} persistent replays {replays} below {minimum_replays}")
            if timing_counter(timing, timing_info.graph_discard_counter, timing_info) != 0:
                errors.append(f"{label} reported graph capture discards")
            if timing_counter(timing, timing_info.graph_capacity_skip_counter, timing_info) != 0:
                errors.append(f"{label} reported graph capacity skips")

    for route in spec.required_route_counters:
        if timing_counter(baseline, route.name, timing_info) != 0:
            errors.append(f"baseline unexpectedly used {route.label}")
        if timing_counter(candidate, route.name, timing_info) == 0:
            errors.append(f"candidate did not use {route.label}")
    for counter in spec.forbidden_route_counters:
        count = timing_counter(candidate, counter.name, timing_info)
        if count != 0:
            errors.append(f"candidate reported {counter.label}: {count}")
    for label, timing in (("baseline", baseline), ("candidate", candidate)):
        throughput = timing_throughput(timing, timing_info)
        if throughput <= 0.0:
            errors.append(f"{label} reported non-positive decode throughput")
    return errors


def run_case(
    args: argparse.Namespace,
    prompt: str,
    tokens: int,
    enabled: bool,
    stem: str,
    spec: CandidateSpec,
    candidate_environment: tuple[tuple[str, str], ...],
    capture_kv_capacity: int,
    common_environment: tuple[tuple[str, str], ...] = (),
) -> dict:
    mode = "candidate" if enabled else "baseline"
    timing_path = args.output_dir / f"{stem}-{mode}.json"
    log_path = args.output_dir / f"{stem}-{mode}.log"
    command = [
        str(args.wrapper),
        str(args.binary),
        "generate",
        str(args.model),
        prompt,
        "--backend", "cuda",
        "--combined-budget-mb", "22000",
        "--backend-budget-mb", "19000",
        "--kv-budget-mb", "1024",
        "--scratch-budget-mb", "2048",
        "--prefill-chunk-size", "32",
        "--max-tokens", str(tokens),
        "--temperature", "0",
        "--raw-prompt",
        "--no-chat-template",
        "--ignore-eos",
        "--cache-dtype", getattr(args, "cache_dtype", "f32"),
        "--print-token-count",
        "--print-token-ids",
        "--print-timing",
        "--json-timing", str(timing_path),
    ]
    env = os.environ.copy()
    configure_candidate_environment(env, spec, enabled, candidate_environment, common_environment)
    env[CAPTURE_KV_CAPACITY_ENV] = str(capture_kv_capacity)
    completed = subprocess.run(
        command,
        cwd=pathlib.Path(__file__).resolve().parents[4],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout_sec,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"{mode} command failed with exit {completed.returncode}; see {log_path}")
    if not timing_path.is_file():
        raise RuntimeError(f"{mode} command did not write {timing_path}")
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    token_match = TOKEN_IDS_RE.search(completed.stdout)
    timing["token_ids"] = [int(value) for value in token_match.group("ids").split()] if token_match else []
    return timing


def timing_throughput(
    timing: dict,
    metadata: TimingMetadata = DEFAULT_TIMING_METADATA,
) -> float:
    try:
        value = float(nested_value(timing, metadata.throughput_field) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0.0 else 0.0


def paired_throughput(
    baseline: dict,
    candidate: dict,
    metadata: TimingMetadata = DEFAULT_TIMING_METADATA,
) -> dict:
    baseline_tps = timing_throughput(baseline, metadata)
    candidate_tps = timing_throughput(candidate, metadata)
    ratio = candidate_tps / baseline_tps if baseline_tps > 0 else 0.0
    return {
        "baseline_tok_s": baseline_tps,
        "candidate_tok_s": candidate_tps,
        "candidate_ratio": ratio,
        "delta_tok_s": candidate_tps - baseline_tps,
        "delta_percent": (ratio - 1.0) * 100.0 if baseline_tps > 0 else 0.0,
    }


def coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean > 0.0 else 1.0


def flatten_pair_throughputs(items: list[dict]) -> list[dict]:
    flattened = []
    for item in items:
        if item.get("pairs") is not None:
            flattened.extend(pair.get("paired_throughput", pair) for pair in item["pairs"])
        else:
            flattened.append(item.get("paired_throughput", item))
    return flattened


def summarize_throughput(items: list[dict]) -> dict:
    pairs = flatten_pair_throughputs(items)
    if not pairs:
        return {
            "baseline_median_tok_s": 0.0,
            "candidate_median_tok_s": 0.0,
            "median_candidate_ratio": 0.0,
            "min_candidate_ratio": 0.0,
            "baseline_tok_s_cv": 0.0,
            "candidate_tok_s_cv": 0.0,
            "pair_count": 0,
        }
    baseline_values = [float(pair["baseline_tok_s"]) for pair in pairs]
    candidate_values = [float(pair["candidate_tok_s"]) for pair in pairs]
    ratios = [float(pair["candidate_ratio"]) for pair in pairs]
    return {
        "baseline_median_tok_s": statistics.median(baseline_values),
        "candidate_median_tok_s": statistics.median(candidate_values),
        "median_candidate_ratio": statistics.median(ratios),
        "min_candidate_ratio": min(ratios),
        "baseline_tok_s_cv": coefficient_of_variation(baseline_values),
        "candidate_tok_s_cv": coefficient_of_variation(candidate_values),
        "pair_count": len(pairs),
    }


def summarize_case(prompt: str, tokens: int, pairs: list[dict]) -> dict:
    throughput = summarize_throughput(pairs)
    pair_errors = []
    for pair in pairs:
        prefix = f"repeat {pair['repetition']}: " if len(pairs) > 1 else ""
        pair_errors.extend(prefix + error for error in pair["errors"])
    legacy_throughput = {
        "baseline_tok_s": throughput["baseline_median_tok_s"],
        "candidate_tok_s": throughput["candidate_median_tok_s"],
        "candidate_ratio": throughput["median_candidate_ratio"],
        "delta_tok_s": statistics.median(pair["delta_tok_s"] for pair in pairs),
        "delta_percent": (throughput["median_candidate_ratio"] - 1.0) * 100.0,
    }
    return {
        "prompt": prompt,
        "tokens": tokens,
        **legacy_throughput,
        "paired_throughput": {**legacy_throughput, **throughput},
        "baseline_median_tok_s": throughput["baseline_median_tok_s"],
        "candidate_median_tok_s": throughput["candidate_median_tok_s"],
        "median_candidate_ratio": throughput["median_candidate_ratio"],
        "min_candidate_ratio": throughput["min_candidate_ratio"],
        "baseline_tok_s_cv": throughput["baseline_tok_s_cv"],
        "candidate_tok_s_cv": throughput["candidate_tok_s_cv"],
        "repeats": len(pairs),
        "pairs": pairs,
        "token_ids_equal": all(pair["token_ids_equal"] for pair in pairs),
        "errors": pair_errors,
    }


def case_promotion_errors(case: dict, min_candidate_ratio: float, max_cv: float) -> list[str]:
    errors = []
    if case["min_candidate_ratio"] < min_candidate_ratio:
        errors.append(
            f"minimum candidate ratio {case['min_candidate_ratio']:.6f} below {min_candidate_ratio:.6f}"
        )
    if case["baseline_tok_s_cv"] > max_cv:
        errors.append(f"baseline throughput CV {case['baseline_tok_s_cv']:.6f} above {max_cv:.6f}")
    if case["candidate_tok_s_cv"] > max_cv:
        errors.append(f"candidate throughput CV {case['candidate_tok_s_cv']:.6f} above {max_cv:.6f}")
    return errors


def main() -> None:
    args = parse_args()
    try:
        spec = resolve_candidate_spec(args)
        if args.model is None:
            if spec.requires_explicit_model:
                raise ValueError(
                    f"{spec.kernel_id} requires --model for a GGUF with a compatible Q6_K LM head"
                )
            raise ValueError("--model is required")
        if args.model_label is None:
            args.model_label = args.model.stem
        timing_info = timing_metadata_from_args(args)
        candidate_environment = resolve_candidate_environment(args, spec)
        common_environment = resolve_common_environment(args, spec, candidate_environment)
        capture_kv_capacity = resolve_capture_kv_capacity(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not args.binary.is_file() or not args.wrapper.is_file() or not args.model.exists():
        raise SystemExit("binary, tuning wrapper, and model must exist")
    if args.timeout_sec <= 0 or args.repeats <= 0 or any(length < 1 for length in args.lengths):
        raise SystemExit("timeout, repeats, and output lengths must be positive")
    if not math.isfinite(args.min_candidate_ratio) or args.min_candidate_ratio < 0.0:
        raise SystemExit("min-candidate-ratio must be a finite non-negative number")
    if not math.isfinite(args.max_cv) or args.max_cv < 0.0:
        raise SystemExit("max-cv must be a finite non-negative number")
    if not args.model_label.strip() or not args.config_label.strip():
        raise SystemExit("model-label and config-label must be non-empty")
    prompts = args.prompt or list(DEFAULT_PROMPTS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    validation_failures = []
    promotion_failures = []
    for prompt_index, prompt in enumerate(prompts):
        for tokens in args.lengths:
            stem = f"{prompt_index:02d}-{slug(prompt)}-{tokens}"
            pairs = []
            for repetition in range(args.repeats):
                pair_stem = repetition_stem(stem, repetition, args.repeats)
                runs = {}
                order = execution_order(repetition)
                for enabled in order:
                    runs[enabled] = run_case(
                        args,
                        prompt,
                        tokens,
                        enabled,
                        pair_stem,
                        spec,
                        candidate_environment,
                        capture_kv_capacity[0],
                        common_environment,
                    )
                baseline = runs[False]
                candidate = runs[True]
                errors = validate_pair(baseline, candidate, tokens, spec, timing_info)
                throughput = paired_throughput(baseline, candidate, timing_info)
                pair = {
                    "repetition": repetition + 1,
                    "execution_order": ["candidate" if enabled else "baseline" for enabled in order],
                    **throughput,
                    "paired_throughput": throughput,
                    "token_ids_equal": (baseline.get("token_ids") or []) == (candidate.get("token_ids") or []),
                    "errors": errors,
                }
                pairs.append(pair)
                validation_failures.extend(f"{pair_stem}: {error}" for error in errors)
                candidate_name = spec.legacy_kind.value if spec.legacy_kind is not None else spec.kernel_id
                print(
                    f"candidate_parity kind={candidate_name} kernel_id={spec.kernel_id} tokens={tokens} "
                    f"repeat={repetition + 1}/{args.repeats} order={','.join(pair['execution_order'])} "
                    f"baseline={throughput['baseline_tok_s']:.3f} "
                    f"candidate={throughput['candidate_tok_s']:.3f} "
                    f"ratio={throughput['candidate_ratio']:.4f} passed={str(not errors).lower()}"
                )

            case = summarize_case(prompt, tokens, pairs)
            case["promotion_errors"] = case_promotion_errors(case, args.min_candidate_ratio, args.max_cv)
            case["checks"] = {
                "validation": not case["errors"],
                "candidate_ratio": case["min_candidate_ratio"] >= args.min_candidate_ratio,
                "stability": case["baseline_tok_s_cv"] <= args.max_cv and case["candidate_tok_s_cv"] <= args.max_cv,
            }
            case["passed"] = all(case["checks"].values())
            cases.append(case)
            promotion_failures.extend(f"{stem}: {error}" for error in case["promotion_errors"])

    throughput_summary = summarize_throughput(cases)
    throughput_summary["case_count"] = len(cases)
    throughput_summary["max_case_baseline_tok_s_cv"] = max(
        (case["baseline_tok_s_cv"] for case in cases), default=0.0
    )
    throughput_summary["max_case_candidate_tok_s_cv"] = max(
        (case["candidate_tok_s_cv"] for case in cases), default=0.0
    )
    failures = validation_failures + promotion_failures
    summary = {
        "candidate": candidate_metadata(spec),
        "config": result_config_metadata(
            args,
            timing_info,
            spec,
            candidate_environment,
            capture_kv_capacity,
            common_environment,
        ),
        "checks": {
            "validation": not validation_failures,
            "candidate_ratio": all(case["checks"]["candidate_ratio"] for case in cases),
            "stability": all(case["checks"]["stability"] for case in cases),
        },
        "passed": not failures,
        "throughput": throughput_summary,
        "cases": cases,
        "validation_failures": validation_failures,
        "promotion_failures": promotion_failures,
        "failures": failures,
    }
    summary_path = args.output_dir / "candidate_summary.json"
    summary_path.write_text(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    candidate_name = spec.legacy_kind.value if spec.legacy_kind is not None else spec.kernel_id
    print(
        f"candidate_parity kind={candidate_name} kernel_id={spec.kernel_id} "
        f"model={args.model_label} config={args.config_label} passed={str(not failures).lower()} "
        f"median_ratio={summary['throughput']['median_candidate_ratio']:.4f} "
        f"min_ratio={summary['throughput']['min_candidate_ratio']:.4f} "
        f"max_baseline_cv={summary['throughput']['max_case_baseline_tok_s_cv']:.4f} "
        f"max_candidate_cv={summary['throughput']['max_case_candidate_tok_s_cv']:.4f} "
        f"output={summary_path}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
