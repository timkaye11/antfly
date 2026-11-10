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


def test_reconstruction_targets_are_scheduled_without_changing_encoder_question():
    from encoder_lora import reconstruction_task, SYSTEM, RECONSTRUCTION_SYSTEM
    from encoder_lora_corpus import broaden

    rows = broaden(data_rows(1047, "train", 4))
    for index, case in enumerate(rows):
        original = case["question"]
        q, answer, system, aux = reconstruction_task(case, index, 2)
        assert aux == bool(index % 2)
        assert case["question"] == original
        if aux:
            assert system == RECONSTRUCTION_SYSTEM
            assert q == case["reconstruction"]["question"]
            name_line, code_line = answer.splitlines()
            for field in (
                name_line.removeprefix("Name: "),
                code_line.removeprefix("Code: "),
            ):
                assert field == "UNKNOWN" or any(field in d for d in case["documents"])
                if field != "UNKNOWN":
                    assert field not in q
        else:
            assert (q, answer, system) == (original, case["answer"], SYSTEM)
    assert reconstruction_task(rows[0], 1, 0)[-1] is False


def test_expanded_corpus_preserves_original_dev_and_never_reads_old_test(tmp_path):
    from encoder_lora_corpus import expanded_data, disjoint

    old = tmp_path / "old"
    make_data(old, train_groups=2, eval_groups=2)
    expected = load_split(old, "dev")
    (old / "test.json").unlink()
    (old / "ood.json").unlink()
    out = tmp_path / "new"
    manifest = expanded_data(out, old, train_groups=32, fresh_dev_groups=4)
    train, dev = load_split(out, "train"), load_split(out, "dev")
    assert len(train) == 128 and len(dev) == 24
    assert disjoint(train, dev)
    assert [
        {k: v for k, v in r.items() if k != "cohort"} for r in dev[: len(expected)]
    ] == expected
    assert all("reconstruction" in r for r in train)
    assert all("reconstruction" not in r for r in dev)
    assert manifest["old_qualification_splits_read"] is False
    assert not (out / "test.json").exists()


def test_cohort_scores_do_not_hide_original_development_failures():
    rows = [
        dict(r, actual="UNKNOWN", correct=False, cohort="old" if i < 4 else "fresh")
        for i, r in enumerate(data_rows(71, "dev", 2))
    ]
    value = scores(rows)
    assert value["total"] == 8
    assert value["cohorts"]["old"]["total"] == value["cohorts"]["fresh"]["total"] == 4
    assert value["cohorts"]["old"]["correct"] == 0


def test_postfreeze_challenge_is_disjoint_and_contains_one_character_pairs():
    from encoder_lora_eval import fresh_challenge
    from encoder_lora_corpus import broaden, disjoint

    train = broaden(data_rows(571, "train", 16))
    dev = data_rows(571, "dev", 4)
    seed, rows = fresh_challenge(8549, 8, train, dev)
    assert (seed, rows) == fresh_challenge(8549, 8, train, dev)
    assert len(rows) == 64
    assert disjoint(rows, train) and disjoint(rows, dev)
    assert {r["cohort"] for r in rows} == {"postfreeze", "postfreeze-phrasing"}
    for i in range(0, len(rows), 4):
        a, b = rows[i : i + 2]
        assert a["question"] == b["question"]
        assert sum(x != y for x, y in zip(a["answer"], b["answer"])) == 1
        assert any(b["answer"] in d for d in b["documents"])


def test_postfreeze_filters_colliding_groups_without_changing_seed():
    from encoder_lora_eval import fresh_challenge
    from encoder_lora_corpus import disjoint

    seed = 90817
    forbidden = data_rows(seed, "postfreeze", 16)[:4]
    actual_seed, rows = fresh_challenge(seed, 8, forbidden, [])
    assert actual_seed == seed
    assert len(rows) == 64 and disjoint(rows, forbidden)
    assert not any(r["id"].startswith("postfreeze:postfreeze-0000-") for r in rows)


def test_continuation_rejects_reconstruction_or_initialization_changes():
    from encoder_lora import continuation_contract

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
    parent = {k: k for k in fields}
    parent.update(
        steps=512,
        versions={"torch": "pinned"},
        reconstruct_every=2,
        reconstruction_system="reconstruct",
        initialization={"candidate": "fixed", "optimizer_reset": True},
    )
    child = dict(parent, steps=1024)
    continuation_contract(parent, child, 512)
    for key, value in [
        ("reconstruct_every", 0),
        ("reconstruction_system", "changed"),
        ("initialization", None),
    ]:
        try:
            continuation_contract(parent, dict(child, **{key: value}), 512)
        except ValueError:
            pass
        else:
            raise AssertionError("changed continuation " + key)


def test_training_probe_records_actual_seen_qa_and_reconstruction_rows():
    from encoder_lora_eval import training_exposures, select_training_probe

    protocol = {
        "seed": 23,
        "steps": 9,
        "gradient_accumulation": 4,
        "reconstruct_every": 2,
    }
    visits = training_exposures(protocol, 32)
    expected = [{"qa": 0, "reconstruction": 0} for _ in range(32)]
    for absolute in range(36):
        expected[order_index(23, absolute, 32)][
            "reconstruction" if absolute % 2 else "qa"
        ] += 1
    assert visits == expected
    train = data_rows(55, "train", 16)
    a = select_training_probe(train, protocol)
    b = select_training_probe(train, dict(protocol, reconstruct_every=0))
    assert a == b and len(a) == 32


def test_spelling_reconstruction_preserves_qa_and_exact_characters(tmp_path):
    from encoder_lora_corpus import expanded_data

    old = tmp_path / "old"
    make_data(old, train_groups=2, eval_groups=2)
    fields, spelled = tmp_path / "fields", tmp_path / "spelled"
    expanded_data(fields, old, train_groups=16, fresh_dev_groups=4)
    expanded_data(spelled, old, train_groups=16, fresh_dev_groups=4, spell_values=True)
    assert load_split(fields, "dev") == load_split(spelled, "dev")
    for a, b in zip(load_split(fields, "train"), load_split(spelled, "train")):
        assert {k: v for k, v in a.items() if k != "reconstruction"} == {
            k: v for k, v in b.items() if k != "reconstruction"
        }
        for plain, coded in zip(
            a["reconstruction"]["answer"].splitlines(),
            b["reconstruction"]["answer"].splitlines(),
        ):
            prefix, value = plain.split(": ", 1)
            encoded = coded.split(": ", 1)[1]
            decoded = (
                encoded
                if value == "UNKNOWN"
                else encoded.replace(" / ", "|").replace(" ", "").replace("|", " ")
            )
            assert decoded == value


def test_paired_corpus_comparison_requires_identical_qa_inputs_and_labels():
    import copy
    from encoder_lora_corpus import broaden, spell_targets
    from encoder_lora_eval import qa_corpus

    fields = broaden(data_rows(83, "train", 2))
    spelled = spell_targets(copy.deepcopy(fields))
    assert qa_corpus(fields) == qa_corpus(spelled)
    for key in ["answer", "question"]:
        changed = copy.deepcopy(spelled)
        changed[0][key] += "changed"
        assert qa_corpus(fields) != qa_corpus(changed)
    changed = copy.deepcopy(spelled)
    changed[0]["documents"][0] += "different fact"
    assert qa_corpus(fields) != qa_corpus(changed)


def test_fresh_challenge_tracks_and_authenticates_prior_training_lineage(tmp_path):
    from encoder_lora_eval import prior_dataset_hashes
    from encoder_lora import digest, save_json

    parent = tmp_path / "parent"
    parent.mkdir()
    save_json(parent / "protocol.json", {"dataset_sha256": "old-corpus"})
    save_json(
        parent / "candidate.json", {"protocol_sha256": digest(parent / "protocol.json")}
    )
    current = {
        "initialization": {
            "directory": str(parent),
            "candidate_sha256": digest(parent / "candidate.json"),
        }
    }
    assert prior_dataset_hashes(current) == {"old-corpus"}
    assert prior_dataset_hashes({}) == set()
    save_json(parent / "protocol.json", {"dataset_sha256": "substitution"})
    try:
        prior_dataset_hashes(current)
    except ValueError:
        pass
    else:
        raise AssertionError("changed lineage accepted")
