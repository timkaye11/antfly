#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import torch
from torch import nn

from encoder_lora import (
    LoRALinear,
    adapter_state,
    assert_frozen,
    data_rows,
    frozen_versions,
    inject_lora,
    load_split,
    make_data,
    order_index,
    restore_adapters,
    scores,
    teacher_inputs,
    atomic_checkpoint,
    continuation_contract,
)


def test_lora_is_initially_noop_and_backpropagates_through_frozen_decoder():
    torch.manual_seed(23)
    base = nn.Linear(8, 8).requires_grad_(False)
    original = base.weight.detach().clone()
    adapter = LoRALinear(base, rank=2, alpha=4)
    reader = nn.Linear(8, 8)
    decoder = nn.Linear(8, 13).requires_grad_(False)
    frozen = frozen_versions(adapter, decoder)
    x = torch.randn(4, 8)
    assert torch.equal(base(x), adapter(x))
    logits = decoder(reader(adapter(x)))
    nn.functional.cross_entropy(logits, torch.tensor([1, 2, 3, 4])).backward()
    assert adapter.B.grad.abs().sum() > 0
    assert reader.weight.grad.abs().sum() > 0
    assert adapter.A.grad.abs().sum() == 0  # B starts at zero.
    assert_frozen(frozen)
    optimizer = torch.optim.AdamW([adapter.A, adapter.B, *reader.parameters()], lr=0.01)
    optimizer.step()
    assert torch.equal(base.weight, original)
    assert not torch.equal(base(x), adapter(x))
    assert_frozen(frozen)


def test_control_adapters_stay_frozen_and_checkpoint_restore_matches():
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.ModuleDict(
                {
                    "q_proj": nn.Linear(8, 8),
                    "v_proj": nn.Linear(8, 8),
                    "k_proj": nn.Linear(8, 8),
                }
            )

    model = Encoder().requires_grad_(False)
    adapters = inject_lora(model, 2, 4, False)
    assert set(adapters) == {"layer.q_proj", "layer.v_proj"}
    assert not any(p.requires_grad for p in model.parameters())
    state = adapter_state(adapters)
    with torch.no_grad():
        adapters["layer.q_proj"].B.fill_(0.5)
    restore_adapters(adapters, state)
    assert not adapters["layer.q_proj"].B.any()
    try:
        restore_adapters(adapters, {})
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete adapter state accepted")


def test_dataset_is_deterministic_disjoint_and_has_counterfactuals(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    first = make_data(a, train_groups=8, eval_groups=4)
    second = make_data(b, train_groups=8, eval_groups=4)
    assert first == second
    all_rows = {s: load_split(a, s) for s in ("train", "dev", "test", "ood")}
    for i, split in enumerate(all_rows):
        rows = all_rows[split]
        assert {r["category"] for r in rows} == {"code", "name", "missing"}
        for other in list(all_rows)[i + 1 :]:
            assert not (
                {e for r in rows for e in r["entities"]}
                & {e for r in all_rows[other] for e in r["entities"]}
            )
        for group in {r["group"] for r in rows}:
            members = [r for r in rows if r["group"] == group]
            code = [r for r in members if r["category"] == "code"]
            missing = next(r for r in members if r["category"] == "missing")
            assert code[0]["question"] == code[1]["question"]
            assert code[0]["answer"] != code[1]["answer"]
            assert missing["question"] in {
                r["question"] for r in members if r["category"] != "missing"
            }
            assert missing["answer"] == "UNKNOWN"
        if split == "ood":
            assert {r["template"] for r in rows} == {3}


def test_manifest_rejects_modified_split(tmp_path):
    make_data(tmp_path / "data", train_groups=2, eval_groups=2)
    path = tmp_path / "data" / "dev.json"
    data = json.loads(path.read_text())
    data["cases"][0]["answer"] = "fake"
    path.write_text(json.dumps(data))
    try:
        load_split(tmp_path / "data", "dev")
    except ValueError:
        pass
    else:
        raise AssertionError("modified labels accepted")


def test_order_is_matched_by_absolute_microstep_and_covers_each_epoch():
    first = [order_index(23, i, 32) for i in range(64)]
    assert first == [order_index(23, i, 32) for i in range(64)]
    assert sorted(first[:32]) == sorted(first[32:]) == list(range(32))
    assert first[:32] != first[32:]


def test_teacher_forcing_alignment_and_frozen_parameter_mutation_detection():
    embedding = nn.Embedding(16, 8).requires_grad_(False)
    prefix = torch.randn(7, 8, requires_grad=True)
    targets = torch.tensor([3, 5, 2])
    inputs = teacher_inputs(prefix, embedding, targets)
    assert inputs.shape == (1, 9, 8)
    assert torch.equal(inputs[0, 7:], embedding(targets[:-1]))
    decoder = nn.Linear(8, 16).requires_grad_(False)
    frozen = frozen_versions(embedding, decoder)
    # Cumulative context stands in for attention: answer loss reaches prefix.
    logits = decoder(inputs.cumsum(dim=1))[:, -len(targets) :]
    nn.functional.cross_entropy(logits[0], targets).backward()
    assert prefix.grad.abs().sum() > 0
    assert_frozen(frozen)
    with torch.no_grad():
        decoder.weight.add_(1)
    try:
        assert_frozen(frozen)
    except RuntimeError:
        pass
    else:
        raise AssertionError("frozen mutation accepted")


def test_blanket_unknown_fails_factual_and_counterfactual_scores():
    rows = data_rows(99, "dev", 2)
    results = [
        dict(r, actual="UNKNOWN", correct=r["answer"] == "UNKNOWN") for r in rows
    ]
    score = scores(results)
    assert score["correct"] == 2
    assert (
        score["code"]["correct"]
        == score["name"]["correct"]
        == score["counterfactual_pairs"]["correct"]
        == 0
    )


def test_checkpoint_is_weights_only_readable_and_failed_write_preserves_previous(
    tmp_path,
):
    path = tmp_path / "resume.pt"
    state = {
        "step": 3,
        "weights": torch.ones(2),
        "optimizer": {"lr": 0.01},
        "rng": torch.get_rng_state(),
    }
    atomic_checkpoint(path, state)
    assert torch.equal(torch.load(path, weights_only=True)["weights"], torch.ones(2))

    class Unserializable:
        def __reduce__(self):
            raise RuntimeError("forced save error")

    try:
        atomic_checkpoint(path, {"broken": Unserializable()})
    except RuntimeError:
        pass
    else:
        raise AssertionError("save did not fail")
    assert torch.load(path, weights_only=True)["step"] == 3


def test_continuation_preserves_optimizer_data_arm_and_absolute_step_contract():
    fields = [
        "schema",
        "arm",
        "memory_slots",
        "rank",
        "alpha",
        "target_modules",
        "seed",
        "gradient_accumulation",
        "reader_lr",
        "encoder_lr",
        "dropout",
        "precision",
        "dataset_sha256",
        "system_prompt",
        "decoder_frozen",
        "source_text_in_decoder",
        "heldout_opened",
        "rollout_qualified",
    ]
    parent = {key: key for key in fields}
    parent.update(steps=128, versions={"torch": "pinned"}, script_sha256="old")
    new = dict(parent, steps=512, script_sha256="new", eval_every=128)
    continuation_contract(parent, new, 128)
    for field in fields:
        altered = dict(new, **{field: "different"})
        try:
            continuation_contract(parent, altered, 128)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted changed " + field)
    for step in [64, 512]:
        try:
            continuation_contract(parent, new, step)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted inconsistent completed step")
