#!/usr/bin/env python3
"""Run the pinned MLX-LM side of the Gemma4 GRPO parity benchmark.

The runner consumes a closed one-token ranked-sampling contract, never
downloads or tokenizes, and starts from the same Antfly PEFT adapter as Zig.
MLX imports stay lazy so contract tests require only the standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DEFAULT_CASE_PATH = (
    SCRIPT_DIR.parent.parent
    / "testdata"
    / "gemma4_grpo_e2b_seq128_benchmark.json"
)
CASE_SCHEMA_VERSION = "antfly_gemma4_grpo_benchmark_case/v1"
RESULT_SCHEMA_VERSION = "antfly_gemma4_grpo_mlx_benchmark/v1"
FIXED_PROTOCOL = {"cold": 1, "first": 1, "warmup": 3, "measured": 20}
RUNTIME_PACKAGE_NAMES = ("mlx", "mlx-lm")

sys.path.insert(0, str(SCRIPT_DIR))
import run_gemma4_lora_mlx_benchmark as locked  # noqa: E402


class GrpoBenchmarkContractError(RuntimeError):
    """The benchmark input, environment, or result violated the locked contract."""


@dataclass(frozen=True)
class RewardToken:
    token_id: int
    decoded_text: str
    reward: float


@dataclass(frozen=True)
class BenchmarkCase:
    source_path: Path
    name: str
    model_key: str
    target_preset: str
    sequence_length: int
    rank: int
    alpha: float
    learning_rate: float
    optimizer: dict[str, float]
    protocol: dict[str, int]
    sampling_mode: str
    group_size: int
    max_completion_tokens: int
    expected_initial_token_ids: tuple[int, ...]
    clip_epsilon: float
    kl_coef: float
    advantage_epsilon: float
    normalize_advantage: bool
    reward_mode: str
    reward_target: str
    reward_tokens: tuple[RewardToken, ...]
    rendered_prompt: str
    prompt_token_ids: tuple[int, ...]
    semantic_sha256: str

    def reward_for_token(self, token_id: int) -> float:
        for item in self.reward_tokens:
            if item.token_id == token_id:
                return item.reward
        raise GrpoBenchmarkContractError(
            f"ranked token {token_id} is outside the closed reward contract"
        )


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], where: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise GrpoBenchmarkContractError(
            f"{where} fields drifted "
            f"(missing={sorted(expected - actual)}, unknown={sorted(actual - expected)})"
        )


def _finite_float(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GrpoBenchmarkContractError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise GrpoBenchmarkContractError(f"{where} is outside the admitted range")
    return result


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GrpoBenchmarkContractError(f"{where} must be a positive integer")
    return value


def _token_ids(value: Any, where: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise GrpoBenchmarkContractError(f"{where} must be a non-empty token list")
    result: list[int] = []
    for idx, token_id in enumerate(value):
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise GrpoBenchmarkContractError(f"{where}[{idx}] is not a token id")
        result.append(token_id)
    return tuple(result)


def _canonical_case_payload(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def load_case(path: Path) -> BenchmarkCase:
    source_path = path.expanduser().resolve()
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GrpoBenchmarkContractError(f"could not load GRPO case: {exc}") from exc
    if not isinstance(payload, dict):
        raise GrpoBenchmarkContractError("GRPO case root must be an object")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "name",
            "model_key",
            "target_preset",
            "sequence_length",
            "rank",
            "alpha",
            "learning_rate",
            "optimizer",
            "protocol",
            "sampling",
            "grpo",
            "reward",
            "rendered_prompt",
            "prompt_token_ids",
        },
        "GRPO case",
    )
    if payload["schema_version"] != CASE_SCHEMA_VERSION:
        raise GrpoBenchmarkContractError("unsupported GRPO case schema")

    for key in ("name", "model_key", "target_preset", "rendered_prompt"):
        if not isinstance(payload[key], str) or not payload[key]:
            raise GrpoBenchmarkContractError(f"{key} must be a non-empty string")

    optimizer = payload["optimizer"]
    if not isinstance(optimizer, dict):
        raise GrpoBenchmarkContractError("optimizer must be an object")
    _require_exact_keys(
        optimizer,
        {"beta1", "beta2", "epsilon", "weight_decay", "max_grad_norm"},
        "optimizer",
    )
    normalized_optimizer = {
        "beta1": _finite_float(optimizer["beta1"], "optimizer.beta1", positive=True),
        "beta2": _finite_float(optimizer["beta2"], "optimizer.beta2", positive=True),
        "epsilon": _finite_float(
            optimizer["epsilon"], "optimizer.epsilon", positive=True
        ),
        "weight_decay": _finite_float(
            optimizer["weight_decay"], "optimizer.weight_decay"
        ),
        "max_grad_norm": _finite_float(
            optimizer["max_grad_norm"], "optimizer.max_grad_norm", positive=True
        ),
    }
    if not 0.0 < normalized_optimizer["beta1"] < 1.0:
        raise GrpoBenchmarkContractError("optimizer.beta1 must be between zero and one")
    if not 0.0 < normalized_optimizer["beta2"] < 1.0:
        raise GrpoBenchmarkContractError("optimizer.beta2 must be between zero and one")
    if normalized_optimizer["weight_decay"] < 0.0:
        raise GrpoBenchmarkContractError("optimizer.weight_decay cannot be negative")

    protocol = payload["protocol"]
    if not isinstance(protocol, dict):
        raise GrpoBenchmarkContractError("protocol must be an object")
    _require_exact_keys(protocol, set(FIXED_PROTOCOL), "protocol")
    normalized_protocol = {
        key: _positive_int(protocol[key], f"protocol.{key}") for key in FIXED_PROTOCOL
    }
    if normalized_protocol != FIXED_PROTOCOL:
        raise GrpoBenchmarkContractError(
            f"protocol must equal the fixed {FIXED_PROTOCOL} contract"
        )

    sampling = payload["sampling"]
    if not isinstance(sampling, dict):
        raise GrpoBenchmarkContractError("sampling must be an object")
    _require_exact_keys(
        sampling,
        {
            "mode",
            "group_size",
            "max_completion_tokens",
            "expected_initial_token_ids",
        },
        "sampling",
    )
    group_size = _positive_int(sampling["group_size"], "sampling.group_size")
    completion_tokens = _positive_int(
        sampling["max_completion_tokens"], "sampling.max_completion_tokens"
    )
    expected_ids = _token_ids(
        sampling["expected_initial_token_ids"],
        "sampling.expected_initial_token_ids",
    )
    if sampling["mode"] != "deterministic-ranked-top-k":
        raise GrpoBenchmarkContractError("unsupported sampling mode")
    if group_size != 2 or completion_tokens != 1 or len(expected_ids) != group_size:
        raise GrpoBenchmarkContractError(
            "the fixed GRPO benchmark requires two ranked one-token completions"
        )
    if len(set(expected_ids)) != len(expected_ids):
        raise GrpoBenchmarkContractError("expected ranked tokens must be distinct")

    grpo = payload["grpo"]
    if not isinstance(grpo, dict):
        raise GrpoBenchmarkContractError("grpo must be an object")
    _require_exact_keys(
        grpo,
        {
            "clip_epsilon",
            "kl_coef",
            "advantage_epsilon",
            "normalize_advantage",
        },
        "grpo",
    )
    if grpo["normalize_advantage"] is not True:
        raise GrpoBenchmarkContractError(
            "the fixed GRPO benchmark requires normalized advantages"
        )

    reward = payload["reward"]
    if not isinstance(reward, dict):
        raise GrpoBenchmarkContractError("reward must be an object")
    _require_exact_keys(reward, {"mode", "target", "token_contract"}, "reward")
    if reward["mode"] != "prefix-match":
        raise GrpoBenchmarkContractError("the fixed reward mode is prefix-match")
    if not isinstance(reward["target"], str) or not reward["target"]:
        raise GrpoBenchmarkContractError("reward.target must be a non-empty string")
    token_contract = reward["token_contract"]
    if not isinstance(token_contract, list) or len(token_contract) != group_size:
        raise GrpoBenchmarkContractError(
            "reward.token_contract must cover exactly the ranked completion group"
        )
    reward_tokens: list[RewardToken] = []
    for idx, row in enumerate(token_contract):
        if not isinstance(row, dict):
            raise GrpoBenchmarkContractError(f"reward.token_contract[{idx}] is not an object")
        _require_exact_keys(row, {"token_id", "decoded_text", "reward"}, f"reward.token_contract[{idx}]")
        token_id = _token_ids([row["token_id"]], f"reward.token_contract[{idx}].token_id")[0]
        if not isinstance(row["decoded_text"], str) or not row["decoded_text"]:
            raise GrpoBenchmarkContractError(
                f"reward.token_contract[{idx}].decoded_text must be non-empty"
            )
        reward_value = _finite_float(row["reward"], f"reward.token_contract[{idx}].reward")
        if reward_value < 0.0:
            raise GrpoBenchmarkContractError("rewards cannot be negative")
        reward_tokens.append(
            RewardToken(token_id, row["decoded_text"], reward_value)
        )
    if tuple(item.token_id for item in reward_tokens) != expected_ids:
        raise GrpoBenchmarkContractError(
            "reward token order must equal the expected ranked token order"
        )
    if len({item.reward for item in reward_tokens}) < 2:
        raise GrpoBenchmarkContractError("the reward contract must produce an advantage")

    prompt_ids = _token_ids(payload["prompt_token_ids"], "prompt_token_ids")
    sequence_length = _positive_int(payload["sequence_length"], "sequence_length")
    if len(prompt_ids) + completion_tokens > sequence_length:
        raise GrpoBenchmarkContractError("prompt plus completion exceeds sequence_length")

    digest = "sha256:" + hashlib.sha256(_canonical_case_payload(payload)).hexdigest()
    return BenchmarkCase(
        source_path=source_path,
        name=payload["name"],
        model_key=payload["model_key"],
        target_preset=payload["target_preset"],
        sequence_length=sequence_length,
        rank=_positive_int(payload["rank"], "rank"),
        alpha=_finite_float(payload["alpha"], "alpha", positive=True),
        learning_rate=_finite_float(
            payload["learning_rate"], "learning_rate", positive=True
        ),
        optimizer=normalized_optimizer,
        protocol=normalized_protocol,
        sampling_mode=sampling["mode"],
        group_size=group_size,
        max_completion_tokens=completion_tokens,
        expected_initial_token_ids=expected_ids,
        clip_epsilon=_finite_float(
            grpo["clip_epsilon"], "grpo.clip_epsilon", positive=True
        ),
        kl_coef=_finite_float(grpo["kl_coef"], "grpo.kl_coef", positive=True),
        advantage_epsilon=_finite_float(
            grpo["advantage_epsilon"], "grpo.advantage_epsilon", positive=True
        ),
        normalize_advantage=True,
        reward_mode=reward["mode"],
        reward_target=reward["target"],
        reward_tokens=tuple(reward_tokens),
        rendered_prompt=payload["rendered_prompt"],
        prompt_token_ids=prompt_ids,
        semantic_sha256=digest,
    )


def normalized_advantages(rewards: Sequence[float], epsilon: float) -> list[float]:
    if not rewards:
        raise GrpoBenchmarkContractError("cannot normalize an empty reward group")
    mean = statistics.mean(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    denominator = math.sqrt(variance) + epsilon
    return [(reward - mean) / denominator for reward in rewards]


def padded_group(
    prompt: Sequence[int], completion_tokens: Sequence[int], sequence_length: int
) -> list[list[int]]:
    if not prompt or not completion_tokens or len(prompt) + 1 > sequence_length:
        raise GrpoBenchmarkContractError("invalid ranked completion group")
    rows: list[list[int]] = []
    for token_id in completion_tokens:
        row = [*prompt, token_id]
        rows.append(row + [0] * (sequence_length - len(row)))
    return rows


def require_exact_package_versions(
    actual: Mapping[str, str], expected: Mapping[str, str]
) -> dict[str, str]:
    if dict(actual) != dict(expected):
        raise GrpoBenchmarkContractError(
            f"MLX package versions drifted (expected={dict(expected)}, actual={dict(actual)})"
        )
    return dict(actual)


def require_source_revision(root: Path, expected: str, label: str) -> str:
    try:
        checkout = locked.verify_source_checkout(root, expected, source_name=label)
    except locked.ContractError as exc:
        raise GrpoBenchmarkContractError(
            f"could not attest clean {label} source revision: {exc}"
        ) from exc
    return checkout["revision"]


def install_mlx_lm_source_namespace(source_root: Path) -> Path:
    checkout = source_root.expanduser().resolve()
    package_root = checkout / "mlx_lm"
    if not package_root.is_dir():
        raise GrpoBenchmarkContractError(
            f"MLX-LM source checkout has no mlx_lm package: {checkout}"
        )
    mlx_lm_package = types.ModuleType("mlx_lm")
    mlx_lm_package.__path__ = [str(package_root)]
    sys.modules["mlx_lm"] = mlx_lm_package
    models_package = types.ModuleType("mlx_lm.models")
    models_package.__path__ = [str(package_root / "models")]
    sys.modules["mlx_lm.models"] = models_package
    tuner_package = types.ModuleType("mlx_lm.tuner")
    tuner_package.__path__ = [str(package_root / "tuner")]
    sys.modules["mlx_lm.tuner"] = tuner_package
    return package_root


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temp_path = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp_path, destination)
    except FileExistsError as exc:
        raise GrpoBenchmarkContractError(
            f"benchmark output already exists: {destination}"
        ) from exc
    finally:
        temp_path.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    case = load_case(args.case)
    lock = locked.load_lock(args.lock)
    mlx_contract = lock["mlx_reference"]
    locked.force_offline_environment()
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != mlx_contract["python"]:
        raise GrpoBenchmarkContractError(
            f"MLX benchmark requires Python {mlx_contract['python']}, found {actual_python}"
        )
    revisions = mlx_contract["source_revisions"]
    mlx_revision = require_source_revision(
        args.mlx_source_root, revisions["mlx"], "MLX"
    )
    mlx_lm_revision = require_source_revision(
        args.mlx_lm_source_root, revisions["mlx-lm"], "MLX-LM"
    )
    if platform.system() != mlx_contract["required_platform"]:
        raise GrpoBenchmarkContractError("MLX benchmark must run on the locked platform")
    if platform.machine() != mlx_contract["required_machine"]:
        raise GrpoBenchmarkContractError("MLX benchmark must run on the locked machine")
    if case.protocol["warmup"] != mlx_contract["warmup_steps"]:
        raise GrpoBenchmarkContractError("case warmup count differs from the MLX lock")
    if case.protocol["measured"] != mlx_contract["measured_steps"]:
        raise GrpoBenchmarkContractError("case measured count differs from the MLX lock")

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten

    mlx_root = args.mlx_source_root.expanduser().resolve()
    core_path = Path(mx.__file__ or "").resolve()
    if not _path_is_within(core_path, mlx_root):
        raise GrpoBenchmarkContractError(
            f"imported MLX is outside the attested checkout: {core_path}"
        )
    install_mlx_lm_source_namespace(args.mlx_lm_source_root)
    from mlx_lm._version import __version__ as mlx_lm_version
    from mlx_lm.models import gemma4 as mlx_gemma4
    from mlx_lm.tuner.lora import LoRALinear

    actual_package_versions = require_exact_package_versions(
        {
            "mlx": str(mx.__version__),
            "mlx-lm": str(mlx_lm_version),
        },
        {
            name: mlx_contract["packages"][name]
            for name in RUNTIME_PACKAGE_NAMES
        },
    )

    mlx_lm_root = args.mlx_lm_source_root.expanduser().resolve()
    gemma4_source_path = Path(mlx_gemma4.__file__ or "").resolve()
    lora_source_module = sys.modules[LoRALinear.__module__]
    lora_source_path = Path(lora_source_module.__file__ or "").resolve()
    for label, source_path in (
        ("MLX-LM Gemma4", gemma4_source_path),
        ("MLX-LM LoRA", lora_source_path),
    ):
        if not _path_is_within(source_path, mlx_lm_root):
            raise GrpoBenchmarkContractError(
                f"imported {label} is outside the attested checkout: {source_path}"
            )

    model_dir = args.model_dir.expanduser().resolve()
    adapter_dir = args.adapter_dir.expanduser().resolve()
    try:
        manifest = json.loads(
            (adapter_dir / "antfly_finetune_manifest.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise GrpoBenchmarkContractError(f"could not load adapter manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise GrpoBenchmarkContractError("adapter manifest must be an object")
    binding_fields = (
        "base_model_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
    )
    try:
        prepared_summary = {key: manifest[key] for key in binding_fields}
    except KeyError as exc:
        raise GrpoBenchmarkContractError(
            f"adapter manifest is missing model binding: {exc.args[0]}"
        ) from exc
    if any(
        not isinstance(value, str) or len(value) != 64
        for value in prepared_summary.values()
    ):
        raise GrpoBenchmarkContractError("adapter manifest model bindings are malformed")
    base_model_provenance = locked.zig_model_provenance(model_dir)
    for field in binding_fields:
        if prepared_summary[field] != base_model_provenance[field]:
            raise GrpoBenchmarkContractError(
                f"adapter {field} does not match the benchmark model"
            )
    adapter = locked.inspect_initial_adapter(
        adapter_dir,
        lock,
        case.model_key,
        case.target_preset,
        prepared_summary,
    )
    if adapter.semantics["r"] != case.rank or float(
        adapter.semantics["lora_alpha"]
    ) != case.alpha:
        raise GrpoBenchmarkContractError("case rank/alpha differ from the adapter")

    mx.set_default_device(mx.gpu)
    mx.random.seed(42)
    sampler = locked.DarwinProcessMemorySampler()
    sampler.start()
    sampler_active = True
    try:
        load_started = time.perf_counter()
        model, _config = locked.load_locked_mlx_gemma4(
            model_dir,
            mx,
            load_config_fn=lambda path: json.loads(
                (path / "config.json").read_text(encoding="utf-8")
            ),
            get_model_classes_fn=lambda **_kwargs: (
                mlx_gemma4.Model,
                mlx_gemma4.ModelArgs,
            ),
        )
        mx.eval(model.parameters())
        mx.synchronize()
        model.freeze()
        base_inventory = locked.require_bf16_base_model(model, mx)

        prompt_row = list(case.prompt_token_ids) + [0] * (
            case.sequence_length - len(case.prompt_token_ids)
        )
        prompt_input = mx.array([prompt_row], dtype=mx.int32)
        prediction_row = len(case.prompt_token_ids) - 1

        reference_started = time.perf_counter()
        reference_logits = model(prompt_input)[:, prediction_row, :].astype(
            mx.float32
        )[0]
        reference_logprobs = reference_logits - mx.logsumexp(reference_logits)
        mx.eval(reference_logprobs)
        mx.synchronize()
        reference_seconds = time.perf_counter() - reference_started

        targets = locked.target_module_names(
            model, lock, case.model_key, case.target_preset
        )
        target_set = set(targets)
        updates = []
        for name, module in model.named_modules():
            if name not in target_set:
                continue
            if not isinstance(module, nn.Linear):
                raise GrpoBenchmarkContractError(f"non-linear LoRA target: {name}")
            updates.append(
                (
                    name,
                    LoRALinear.from_base(
                        module,
                        r=case.rank,
                        scale=case.alpha / case.rank,
                        dropout=0.0,
                    ),
                )
            )
        if {name for name, _module in updates} != target_set:
            raise GrpoBenchmarkContractError("incomplete LoRA target conversion")
        model.update_modules(tree_unflatten(updates))
        trainable_inventory = locked.require_exact_trainables(model, targets, mx)
        locked.load_exact_initial_adapter(model, targets, adapter, mx)
        model.train()
        mx.eval(model.state)
        mx.synchronize()
        initial_trainables = {
            name: mx.array(value)
            for name, value in tree_flatten(model.trainable_parameters())
        }
        mx.eval(*initial_trainables.values())
        load_seconds = time.perf_counter() - load_started

        optimizer = optim.AdamW(
            learning_rate=case.learning_rate,
            betas=(case.optimizer["beta1"], case.optimizer["beta2"]),
            eps=case.optimizer["epsilon"],
            weight_decay=case.optimizer["weight_decay"],
            bias_correction=True,
        )

        def selected_logps(current_model: Any, tokens: Any, selected: Any) -> Any:
            logits = current_model(tokens)[:, prediction_row, :].astype(mx.float32)
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            return mx.take_along_axis(logprobs, selected[:, None], axis=-1)[:, 0]

        def group_selected_logps(
            current_model: Any, tokens: Any, selected: Any
        ) -> Any:
            # Keep each completion at batch=1, matching Antfly's two physical
            # micro-batches. The pinned MLX Gemma4 E2B graph is not numerically
            # batch-shape invariant enough for the strict sampling/rescore gate.
            return mx.concatenate(
                [
                    selected_logps(
                        current_model,
                        tokens[index : index + 1],
                        selected[index : index + 1],
                    )
                    for index in range(case.group_size)
                ],
                axis=0,
            )

        def grpo_loss(
            current_model: Any,
            tokens: Any,
            selected: Any,
            old_logps: Any,
            reference: Any,
            advantages: Any,
        ) -> tuple[Any, Any, Any, Any, Any]:
            new_logps = group_selected_logps(current_model, tokens, selected)
            ratio = mx.exp(new_logps - old_logps)
            pg_unclipped = ratio * advantages
            pg_clipped = mx.clip(
                ratio,
                1.0 - case.clip_epsilon,
                1.0 + case.clip_epsilon,
            ) * advantages
            pg_tokens = -mx.minimum(pg_unclipped, pg_clipped)
            diff = reference - new_logps
            kl_tokens = case.kl_coef * (mx.exp(diff) - diff - 1.0)
            pg_loss = pg_tokens.mean()
            kl_loss = kl_tokens.mean()
            loss = pg_loss + kl_loss
            clip_fraction = (pg_clipped < pg_unclipped).astype(mx.float32).mean()
            return loss, pg_loss, kl_loss, clip_fraction, new_logps

        loss_and_grad = nn.value_and_grad(model, grpo_loss)
        state = [model.state, optimizer.state, mx.random.state]

        def step(
            tokens: Any,
            selected: Any,
            old_logps: Any,
            reference: Any,
            advantages: Any,
        ) -> tuple[Any, Any, Any, Any, Any]:
            metrics, gradients = loss_and_grad(
                model,
                tokens,
                selected,
                old_logps,
                reference,
                advantages,
            )
            gradients, _norm = optim.clip_grad_norm(
                gradients, case.optimizer["max_grad_norm"]
            )
            optimizer.update(model, gradients)
            return metrics

        compiled = mx.compile(step, inputs=state, outputs=state)

        def execute(update_index: int) -> dict[str, Any]:
            started = time.perf_counter()
            policy_logits = model(prompt_input)[:, prediction_row, :].astype(
                mx.float32
            )[0]
            candidate_ids = mx.argpartition(
                -policy_logits, kth=case.group_size - 1
            )[: case.group_size]
            candidate_logits = policy_logits[candidate_ids]
            order = mx.argsort(-candidate_logits)
            selected = candidate_ids[order]
            policy_logprobs = policy_logits - mx.logsumexp(policy_logits)
            old_logps = policy_logprobs[selected]
            reference = reference_logprobs[selected]
            mx.eval(selected, old_logps, reference)
            mx.synchronize()

            token_ids = [int(value) for value in selected.tolist()]
            if update_index == 0 and tuple(token_ids) != case.expected_initial_token_ids:
                raise GrpoBenchmarkContractError(
                    "MLX initial ranked completion tokens differ from the locked case: "
                    f"expected={case.expected_initial_token_ids}, actual={tuple(token_ids)}"
                )
            rewards = [case.reward_for_token(token_id) for token_id in token_ids]
            advantages = normalized_advantages(rewards, case.advantage_epsilon)
            batch = mx.array(
                padded_group(
                    case.prompt_token_ids,
                    token_ids,
                    case.sequence_length,
                ),
                dtype=mx.int32,
            )
            advantage_array = mx.array(advantages, dtype=mx.float32)
            metrics = compiled(
                batch,
                selected,
                old_logps,
                reference,
                advantage_array,
            )
            loss, pg_loss, kl_loss, clip_fraction, new_logps = metrics
            mx.eval(*metrics, model.state, optimizer.state)
            mx.synchronize()
            elapsed = time.perf_counter() - started

            values = [
                float(loss.item()),
                float(pg_loss.item()),
                float(kl_loss.item()),
                float(clip_fraction.item()),
            ]
            if not all(math.isfinite(value) for value in values):
                raise GrpoBenchmarkContractError("non-finite MLX GRPO metric")
            sampling_rescore_error = float(mx.max(mx.abs(new_logps - old_logps)).item())
            policy_reference_error = float(mx.max(mx.abs(new_logps - reference)).item())
            if update_index == 0 and sampling_rescore_error > 1e-4:
                raise GrpoBenchmarkContractError(
                    "zero-update sampling/rescore logprob parity failed: "
                    f"max_abs_error={sampling_rescore_error}, tokens={token_ids}, "
                    f"sample_logps={[float(value) for value in old_logps.tolist()]}, "
                    f"rescore_logps={[float(value) for value in new_logps.tolist()]}"
                )
            if update_index == 0 and policy_reference_error > 1e-4:
                raise GrpoBenchmarkContractError(
                    "zero-adapter policy/reference logprob parity failed: "
                    f"max_abs_error={policy_reference_error}, tokens={token_ids}, "
                    f"policy_logps={[float(value) for value in new_logps.tolist()]}, "
                    f"reference_logps={[float(value) for value in reference.tolist()]}"
                )
            reward_mean = statistics.mean(rewards)
            reward_variance = sum(
                (reward - reward_mean) ** 2 for reward in rewards
            ) / len(rewards)
            return {
                "seconds": elapsed,
                "loss": values[0],
                "pg_loss": values[1],
                "kl_loss": values[2],
                "clip_fraction": values[3],
                "mean_reward": reward_mean,
                "reward_stddev": math.sqrt(reward_variance),
                "completion_tokens": len(token_ids),
                "completion_token_ids": token_ids,
                "sampling_rescore_max_abs_error": sampling_rescore_error,
                "policy_reference_max_abs_error": policy_reference_error,
                "old_logps": [float(value) for value in old_logps.tolist()],
                "reference_logps": [float(value) for value in reference.tolist()],
            }

        update_index = 0
        cold = execute(update_index)
        update_index += 1
        first = execute(update_index)
        update_index += 1
        warmup = []
        for _ in range(case.protocol["warmup"]):
            warmup.append(execute(update_index))
            update_index += 1
        measured = []
        for _ in range(case.protocol["measured"]):
            measured.append(execute(update_index))
            update_index += 1
        if not any(
            update["policy_reference_max_abs_error"] > 1e-5
            for update in [first, *warmup, *measured]
        ):
            raise GrpoBenchmarkContractError("MLX policy did not move after optimization")

        final_trainables = dict(tree_flatten(model.trainable_parameters()))
        delta_squares = [
            ((final_trainables[name] - initial).astype(mx.float32) ** 2).sum()
            for name, initial in initial_trainables.items()
        ]
        delta_maxima = [
            mx.abs(final_trainables[name] - initial).max()
            for name, initial in initial_trainables.items()
        ]
        mx.eval(*delta_squares, *delta_maxima)
        adapter_delta_l2 = math.sqrt(
            sum(float(value.item()) for value in delta_squares)
        )
        adapter_delta_max_abs = max(float(value.item()) for value in delta_maxima)
        memory = sampler.stop()
        sampler_active = False
    finally:
        if sampler_active:
            sampler.stop()

    measured_seconds = [entry["seconds"] for entry in measured]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "framework": "mlx-lm",
        "algorithm": "grpo",
        "model_key": case.model_key,
        "sequence_length": case.sequence_length,
        "case": {
            "name": case.name,
            "path": str(case.source_path),
            "semantic_sha256": case.semantic_sha256,
            "prompt_tokens": len(case.prompt_token_ids),
            "group_size": case.group_size,
            "max_completion_tokens": case.max_completion_tokens,
        },
        "protocol": case.protocol,
        "sampling_mode": case.sampling_mode,
        "policy_rescore_mode": "independent-batch1-completions",
        "reference_mode": "cached-base-prompt-logprob-distribution",
        "reward": {
            "mode": case.reward_mode,
            "target": case.reward_target,
        },
        "grpo": {
            "clip_epsilon": case.clip_epsilon,
            "kl_coef": case.kl_coef,
            "advantage_epsilon": case.advantage_epsilon,
            "normalize_advantage": case.normalize_advantage,
        },
        "optimizer": {
            "learning_rate": case.learning_rate,
            **case.optimizer,
        },
        "load_seconds": load_seconds,
        "reference_precompute_seconds": reference_seconds,
        "cold": cold,
        "first": first,
        "warmup": warmup,
        "measured": measured,
        "median_seconds": statistics.median(measured_seconds),
        "mean_seconds": statistics.mean(measured_seconds),
        "adapter_delta_l2": adapter_delta_l2,
        "adapter_delta_max_abs": adapter_delta_max_abs,
        "adapter_semantic_sha256": adapter.semantic_sha256,
        "base_inventory_sha256": base_inventory["inventory_sha256"],
        "trainable_inventory_sha256": trainable_inventory["inventory_sha256"],
        "peak_phys_footprint_bytes": memory.peak_phys_footprint_bytes,
        "mlx_allocator_peak_bytes": int(mx.get_peak_memory()),
        "mlx_revision": mlx_revision,
        "mlx_lm_revision": mlx_lm_revision,
        "locked_package_versions": mlx_contract["packages"],
        "actual_runtime_package_versions": actual_package_versions,
        "unused_locked_packages": sorted(
            set(mlx_contract["packages"]) - set(RUNTIME_PACKAGE_NAMES)
        ),
        "python_version": actual_python,
        "base_model_provenance": base_model_provenance,
        "mlx_core_path": str(core_path),
        "mlx_lm_gemma4_path": str(gemma4_source_path),
        "mlx_lm_lora_path": str(lora_source_path),
        "runner_sha256": "sha256:"
        + hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--adapter-dir", type=Path, required=True)
    result.add_argument("--mlx-source-root", type=Path, required=True)
    result.add_argument("--mlx-lm-source-root", type=Path, required=True)
    result.add_argument("--case", type=Path, default=DEFAULT_CASE_PATH)
    result.add_argument("--lock", type=Path, default=locked.LOCK_PATH)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        payload = run(args)
        write_json_exclusive(args.output, payload)
    except (GrpoBenchmarkContractError, locked.ContractError) as exc:
        print(f"Gemma4 GRPO MLX benchmark contract error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
