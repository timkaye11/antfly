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
import math
import re
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
    assert len(rows) == 80
    assert disjoint(rows, train) and disjoint(rows, dev)
    assert {r["cohort"] for r in rows} == {"postfreeze", "postfreeze-phrasing"}
    for group in {r["group"] for r in rows}:
        a, b = [r for r in rows if r["group"] == group and r["category"] == "code"]
        assert a["question"] == b["question"]
        assert sum(x != y for x, y in zip(a["answer"], b["answer"])) == 1
        assert any(b["answer"] in d for d in b["documents"])
        a, b = [r for r in rows if r["group"] == group and r["category"] == "name"]
        assert a["question"] == b["question"] and a["answer"] != b["answer"]
        assert [d.replace(a["answer"], b["answer"]) for d in a["documents"]] == b[
            "documents"
        ]


def test_postfreeze_filters_colliding_groups_without_changing_seed():
    from encoder_lora_eval import fresh_challenge
    from encoder_lora_corpus import disjoint

    seed = 90817
    forbidden = data_rows(seed, "postfreeze", 16)[:4]
    actual_seed, rows = fresh_challenge(seed, 8, forbidden, [])
    assert actual_seed == seed
    assert len(rows) == 80 and disjoint(rows, forbidden)
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


def test_fresh_value_exclusion_includes_codes_seen_only_in_distractors():
    import re
    from encoder_lora_corpus import disjoint

    train, fresh = data_rows(731, "train", 2), data_rows(884, "postfreeze", 2)
    assert disjoint(train, fresh)
    code = train[0]["answer"]
    old = re.search(r"[A-Z][a-z][2-9]-00[0-9]{4}", fresh[0]["documents"][1])[0]
    fresh[0]["documents"][1] = fresh[0]["documents"][1].replace(old, code)
    assert all(r["answer"] != code for r in fresh)
    assert not disjoint(train, fresh)


def test_structured_readout_routes_only_from_public_question_and_preserves_bytes():
    from encoder_lora_eval import readout_prompt, parse_readout

    field, q = readout_prompt(
        "What is the exact access code for Example Depot?", "spelled"
    )
    assert field == "Code" and "Example Depot" in q
    raw = "Name: K o m o d e / T u l o c o\nCode: K g 9 - 0 0 9 3 6 9"
    assert parse_readout(raw, field, "spelled") == "Kg9-009369"
    field, q = readout_prompt("Who holds the brass key at Example Depot?", "spelled")
    assert field == "Name" and parse_readout(raw, field, "spelled") == "Komode Tuloco"
    assert parse_readout("Name: UNKNOWN\nCode: UNKNOWN", field, "spelled") == "UNKNOWN"
    assert parse_readout(raw + "\nextra explanation", field, "spelled") is None
    assert parse_readout(raw.replace("K o m o d e", "Komode"), field, "spelled") is None
    assert (
        parse_readout("Name: Komode Tuloco\nCode: Kg9-009369", "Code", "fields")
        == "Kg9-009369"
    )
    try:
        readout_prompt("What is the weather?", "spelled")
    except ValueError:
        pass
    else:
        raise AssertionError("unrecognized question was routed using hidden labels")


def test_oracle_memory_is_explicitly_bounded_and_has_no_silent_truncation():
    from encoder_lora_eval import literal_memory_ids

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    for kind in ["characters", "bpe"]:
        ids = literal_memory_ids(Tokenizer(), "Kg9-009369", kind)
        assert len(ids) == 16
        assert "".join(chr(i) for i in ids).strip() == "Kg9-009369"
        try:
            literal_memory_ids(Tokenizer(), "x" * 16, kind)
        except ValueError:
            pass
        else:
            raise AssertionError("oracle silently truncated target")


def test_name_worlds_preserve_original_cases_and_group_batches_require_context(
    tmp_path,
):
    from encoder_lora_corpus import expanded_data, name_counterfactual_data, disjoint
    from encoder_lora import CounterfactualGroupSampler
    from encoder_lora_eval import training_exposures

    old, base, paired = tmp_path / "old", tmp_path / "base", tmp_path / "paired"
    make_data(old, train_groups=2, eval_groups=2)
    expanded_data(base, old, train_groups=16, fresh_dev_groups=4)
    name_counterfactual_data(paired, base)
    original, rows = load_split(base, "train"), load_split(paired, "train")
    assert len(rows) == 80 and disjoint(rows, load_split(paired, "dev"))
    assert (base / "dev.json").read_bytes() == (paired / "dev.json").read_bytes()
    by_id = {r["id"]: r for r in rows}
    assert all(by_id[r["id"]] == r for r in original)
    for group in {r["group"] for r in rows}:
        a, b = [r for r in rows if r["group"] == group and r["category"] == "name"]
        assert a["question"] == b["question"] and a["answer"] != b["answer"]
        assert [d.replace(a["answer"], b["answer"]) for d in a["documents"]] == b[
            "documents"
        ]
    sampler = CounterfactualGroupSampler(rows, 23)
    for step in range(18):
        batch = [rows[sampler.index(step * 8 + i)] for i in range(8)]
        assert len({r["group"] for r in batch}) == 1
        assert sum(r["category"] == "missing" for r in batch) == 2
        for a, b in [(batch[0], batch[1]), (batch[2], batch[3])]:
            assert a["question"] == b["question"] and a["answer"] != b["answer"]
        assert all(
            sampler.index(step * 8 + i)
            == CounterfactualGroupSampler(rows, 23).index(step * 8 + i)
            for i in range(8)
        )
    protocol = {
        "steps": 3,
        "gradient_accumulation": 8,
        "seed": 23,
        "reconstruct_every": 0,
        "sampling": "counterfactual-group",
    }
    visits = training_exposures(protocol, len(rows), rows)
    assert sum(v["qa"] for v in visits) == 24 and not any(
        v["reconstruction"] for v in visits
    )
    for index, v in enumerate(visits):
        assert v["qa"] == sum(sampler.index(i) == index for i in range(24))
    try:
        CounterfactualGroupSampler(original, 23)
    except ValueError:
        pass
    else:
        raise AssertionError("unpaired name dataset accepted")


def test_derived_reference_creation_cannot_replace_a_trained_models_reference():
    from encoder_lora_eval import validate_reference_mode

    pending = {"candidate_kind": "parameter_average", "reference_pending": True}
    protocol = {"postprocessing": {"method": "parameter_average"}}
    state = {"completed": False, "evaluations": []}
    validate_reference_mode(pending, protocol, state, True)
    for candidate, result, create in [
        (pending, state, False),
        ({"candidate_kind": "selected_checkpoint"}, state, True),
        (pending, {"completed": True, "evaluations": [1]}, True),
    ]:
        try:
            validate_reference_mode(candidate, protocol, result, create)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid reference replacement accepted")


def test_checkpoint_average_preserves_provenance_and_requires_first_measurement(
    tmp_path,
):
    from encoder_lora_select import derive_candidate
    from encoder_lora import SCHEMA, save_json, digest, candidate_proof
    from safetensors.torch import load_file

    run = tmp_path / "run"
    run.mkdir()
    (run / "harness_source.py").write_text("synthetic unit-only training source")
    protocol = {
        "schema": SCHEMA,
        "memory_slots": 16,
        "steps": 2,
        "script_sha256": digest(run / "harness_source.py"),
    }
    save_json(run / "protocol.json", protocol)
    save_json(run / "base-proofs.json", {"synthetic": True})
    save_json(
        run / "result.json", {"completed": True, "frozen_base_versions_unchanged": True}
    )
    paths = []
    for step, value in [(1, 1.0), (2, 3.0)]:
        path = run / f"{step}.pt"
        torch.save(
            {
                "protocol": protocol,
                "step": step,
                "reader": {"x": torch.tensor([value])},
                "adapters": {"y": torch.tensor([value + 1])},
                "result": {
                    "completed": False,
                    "history": [{"step": i + 1} for i in range(step)],
                    "evaluations": [{"step": step, "rows": []}],
                },
            },
            path,
        )
        paths.append(path)
    out = tmp_path / "average"
    manifest = derive_candidate(run, paths, out, [0.25, 0.75])
    assert load_file(str(out / "reader.safetensors"))["x"].item() == 2.5
    assert (
        manifest["reference_pending"]
        and not json.loads((out / "result.json").read_text())["completed"]
    )
    assert json.loads((out / "protocol.json").read_text())["postprocessing"][
        "checkpoint_steps"
    ] == [1, 2]
    try:
        candidate_proof(out)
    except ValueError:
        pass
    else:
        raise AssertionError("unmeasured average initialized a model")
    selected = tmp_path / "selected"
    manifest = derive_candidate(run, [paths[0]], selected)
    assert (
        not manifest["reference_pending"]
        and json.loads((selected / "protocol.json").read_text())["steps"] == 1
    )
    assert candidate_proof(selected)["steps"] == 1
    # An interrupted parent yields a single selected checkpoint only when the
    # caller says so, and the candidate records that its parent never finished.
    result_path = run / "result.json"
    finished = json.loads(result_path.read_text())
    save_json(result_path, dict(finished, completed=False))
    try:
        derive_candidate(run, [paths[0]], tmp_path / "refused")
    except ValueError:
        pass
    else:
        raise AssertionError("interrupted parent exported without consent")
    try:
        derive_candidate(
            run, paths, tmp_path / "refused-average", allow_incomplete_parent=True
        )
    except ValueError:
        pass
    else:
        raise AssertionError("interrupted parent averaged")
    manifest = derive_candidate(
        run, [paths[0]], tmp_path / "interrupted", allow_incomplete_parent=True
    )
    assert manifest["parent_completed"] is False
    assert candidate_proof(tmp_path / "interrupted")["steps"] == 1
    save_json(result_path, finished)


def test_document_features_pooled_matches_released_path_and_modes_are_unit_norm():
    from encoder_lora import document_features, feature_channel
    from encoder_reference import normalize_features

    torch.manual_seed(5)
    states = torch.randn(3, 12, 2560)
    lengths = [12, 7, 5]
    mask = torch.zeros(3, 12, dtype=torch.long)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
    expected = normalize_features(states, mask).half().bfloat16().float()
    docs, query = document_features(states, mask, "pooled")
    assert torch.equal(docs, expected[:-1].unsqueeze(0))
    assert torch.equal(query, expected[-1:])
    for mode, count in [("tokens", 19), ("chunks4", 8)]:
        docs, same_query = document_features(states, mask, mode)
        assert docs.shape == (1, count, 2560) and torch.equal(same_query, query)
        assert torch.allclose(docs.norm(dim=-1), torch.ones(1, count), atol=2e-2)
    with_question, same_query = document_features(states, mask, "tokens-question")
    assert with_question.shape == (1, 19 + int(mask[-1].sum()), 2560)
    assert torch.equal(same_query, query)
    assert torch.equal(
        with_question[0, -1],
        feature_channel(
            nn.functional.normalize(states[-1, int(mask[-1].sum()) - 1], dim=-1)
        ),
    )
    tokens, _ = document_features(states, mask, "tokens")
    assert torch.equal(
        tokens[0, 12], feature_channel(nn.functional.normalize(states[1, 0], dim=-1))
    )
    chunks, _ = document_features(states, mask, "chunks4")
    first = feature_channel(nn.functional.normalize(states[0, :3].mean(0), dim=-1))
    assert torch.equal(chunks[0, 0], first)
    assert not torch.equal(chunks[0, 0], tokens[0, 0])
    for mode, bad_mask in [("chunks8", mask), ("pooled", mask[:, :11]), ("bad", mask)]:
        try:
            document_features(states, bad_mask, mode)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid document feature request " + mode)


def test_continuation_rejects_document_feature_or_lexical_objective_changes():
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
    parent.update(steps=512, versions={"torch": "pinned"})
    # Protocols written before these fields existed continue with the defaults.
    child = dict(
        parent,
        steps=1024,
        document_features="pooled",
        lexical_loss_weight=0,
        lexical_record="none",
        lexical_recipe=None,
    )
    continuation_contract(parent, child, 512)
    for key, value in [
        ("document_features", "tokens"),
        ("lexical_loss_weight", 0.5),
        ("lexical_record", "owner-code"),
        ("lexical_recipe", "other"),
        ("memory_scale_init", "lexical-target"),
        ("lexical_warmup_steps", 64),
        ("lexical_objective", "token-ce"),
        ("lexical_warmup_batch", 16),
        ("joint_reader_lr", 1e-4),
        ("lexical_warmup_cache", True),
        ("lexical_warmup_schedule", "cosine"),
        ("lexical_name_weight", 3.0),
        ("lexical_warmup_train_encoder", True),
        ("encoder_layer", 20),
        ("dual_layer", 24),
        ("copy_head", True),
        ("copy_head_temperature", 0.05),
        ("copy_pointer_weight", 0.5),
        ("copy_head_lr", 1e-4),
        ("entity_tag_lr", 5e-3),
        ("entity_tag_max_norm", 0.5),
        ("reader_loops", 2),
        ("slot_refiner_lr", 1e-4),
        ("document_space_prefix", True),
        ("document_self_context", True),
        ("document_pooled_context", True),
        ("entity_tags", "oracle"),
        ("copy_restrict_by_type", True),
    ]:
        try:
            continuation_contract(parent, dict(child, **{key: value}), 512)
        except ValueError:
            pass
        else:
            raise AssertionError("changed continuation " + key)


def test_lexical_record_reads_holder_and_code_from_source_only(tmp_path):
    from encoder_lora import asked_field, lexical_record_text, lexical_settings
    from encoder_lora_corpus import broaden, expanded_data, name_counterfactual_data

    def check(rows):
        for case in rows:
            text = lexical_record_text(case)
            holder, code = text.removeprefix(" ").split(", ")
            assert text.startswith(" ") and "UNKNOWN" not in case["question"]
            if case["category"] == "code":
                assert (holder, code) == (case["entities"][2], case["answer"])
            elif case["category"] == "name":
                assert holder == case["answer"] == case["entities"][2]
                assert code != "UNKNOWN" and any(code in d for d in case["documents"])
            else:
                assert case["answer"] == "UNKNOWN" and "UNKNOWN" in (holder, code)
                if "code" in case["question"]:
                    assert (holder, code) == (case["entities"][2], "UNKNOWN")
                else:
                    assert holder == "UNKNOWN" and code != "UNKNOWN"
        return rows

    original = check(data_rows(17, "dev", 4))
    assert {r["category"] for r in original} == {"code", "name", "missing"}
    check(broaden(data_rows(17, "train", 12)))
    old, base, paired = tmp_path / "old", tmp_path / "base", tmp_path / "paired"
    make_data(old, train_groups=2, eval_groups=2)
    expanded_data(base, old, train_groups=8, fresh_dev_groups=2)
    name_counterfactual_data(paired, base)
    rows = check(load_split(paired, "train"))
    a, b = [
        r for r in rows if r["group"] == rows[0]["group"] and r["category"] == "name"
    ]
    assert lexical_record_text(a) != lexical_record_text(b)
    swapped = dict(original[0], documents=list(reversed(original[0]["documents"])))
    assert lexical_record_text(swapped) == lexical_record_text(original[0])
    for case in original:
        holder, code = lexical_record_text(case).removeprefix(" ").split(", ")
        asked = lexical_record_text(case, "asked-first")
        code_question = "access code" in case["question"]
        assert asked_field(case["question"]) == ("code" if code_question else "name")
        assert asked == (
            f" {code}, {holder}" if code_question else f" {holder}, {code}"
        )
        site = case["entities"][0]
        assert lexical_record_text(case, "site-asked") == (
            f" {site} code {code}" if code_question else f" {site} key holder {holder}"
        )
        assert lexical_record_text(case, "asked-site") == (
            f" code {code} at {site}"
            if code_question
            else f" key holder {holder} at {site}"
        )
    try:
        lexical_record_text(original[0], "none")
    except ValueError:
        pass
    else:
        raise AssertionError("record form none accepted")
    try:
        asked_field("Where is the depot?")
    except ValueError:
        pass
    else:
        raise AssertionError("field-free question accepted")
    for broken in [
        dict(original[0], entities=original[0]["entities"][:3]),
        dict(original[0], documents=original[0]["documents"][1:]),
        dict(original[0], documents=[original[0]["documents"][0] + " Zz9-001234."]),
    ]:
        try:
            lexical_record_text(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous source accepted as lexical record")
    assert lexical_settings(0.0, "none")["lexical_recipe"] is None
    assert lexical_settings(0.5, "owner-code")["lexical_loss_weight"] == 0.5
    for weight, record in [(0.5, "none"), (0.0, "owner-code"), (11, "owner-code")]:
        try:
            lexical_settings(weight, record)
        except ValueError:
            pass
        else:
            raise AssertionError("inconsistent lexical settings accepted")


def test_lexical_targets_fit_sixteen_frozen_slots_and_loss_is_zero_at_target():
    from encoder_lora import lexical_alignment_loss, lexical_targets

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    torch.manual_seed(9)
    embedding = nn.Embedding(128, 8).requires_grad_(False)
    target = lexical_targets(Tokenizer(), embedding, " Ab, Cd9-001234")
    assert target.shape == (16, 8) and not target.requires_grad
    assert torch.equal(target[-1], embedding.weight[ord(" ")])
    assert torch.equal(target[0], embedding.weight[ord(" ")])
    assert torch.equal(target[1], embedding.weight[ord("A")])
    for text, module in [
        (" " + "x" * 16, embedding),
        ("", embedding),
        (" Ab", nn.Embedding(128, 8)),
    ]:
        try:
            lexical_targets(Tokenizer(), module, text)
        except ValueError:
            pass
        else:
            raise AssertionError("lexical target silently truncated or unfrozen")
    truncated = lexical_targets(Tokenizer(), embedding, " " + "y" * 20, truncate=True)
    assert truncated.shape == (16, 8)
    assert torch.equal(truncated[-1], embedding.weight[ord("y")])
    assert float(lexical_alignment_loss(target.clone(), target)) < 1e-6
    predicted = (target + torch.randn_like(target)).requires_grad_(True)
    loss = lexical_alignment_loss(predicted, target)
    assert float(loss) > 0
    loss.backward()
    assert predicted.grad.abs().sum() > 0
    # Scale-aware: shrinking a direction-correct memory is still penalized.
    half = float(lexical_alignment_loss(0.5 * target, target))
    assert abs(half - 0.5 * math.log(2) ** 2) < 1e-5
    # A five-fold scale mismatch stays a bounded, O(1) penalty.
    assert float(lexical_alignment_loss(5 * target, target)) < 1.5
    assert float(lexical_alignment_loss(torch.zeros_like(target), target)) > 1

    for bad in [target[:8], target[:, :4], torch.zeros(16, 8)]:
        try:
            if bad.shape == target.shape:
                lexical_alignment_loss(target, bad)
            else:
                lexical_alignment_loss(bad, target)
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched lexical geometry accepted")


def test_token_cross_entropy_prefers_the_target_row_and_penalizes_collapse():
    from encoder_lora import (
        LexicalVocabularyTeacher,
        lexical_settings,
        lexical_target_ids,
        lexical_token_ce_loss,
    )

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    torch.manual_seed(5)
    embedding = nn.Embedding(128, 8).requires_grad_(False)
    teacher = LexicalVocabularyTeacher(embedding)
    ids = torch.tensor(lexical_target_ids(Tokenizer(), " Ab, Cd9-001234"))
    assert ids.shape == (16,) and int(ids[-1]) == ord(" ")
    exact = lexical_token_ce_loss(embedding.weight[ids].clone(), ids, teacher)
    collapsed = lexical_token_ce_loss(
        embedding.weight[ids].mean(dim=0, keepdim=True).expand(16, 8).clone(),
        ids,
        teacher,
    )
    assert float(exact) < float(collapsed)
    predicted = torch.randn(16, 8, requires_grad=True)
    loss = lexical_token_ce_loss(predicted, ids, teacher)
    loss.backward()
    assert predicted.grad.abs().sum() > 0
    stacked = torch.stack([predicted.detach(), embedding.weight[ids].clone()])
    batched = lexical_token_ce_loss(stacked, torch.stack([ids, ids]), teacher)
    assert abs(float(batched) - 0.5 * (float(loss) + float(exact))) < 1e-4
    for bad in [torch.randn(8, 8), torch.randn(16, 4), torch.randn(2, 8, 8)]:
        try:
            lexical_token_ce_loss(bad, ids, teacher)
        except ValueError:
            pass
        else:
            raise AssertionError("geometry mismatch accepted")
    try:
        LexicalVocabularyTeacher(nn.Embedding(128, 8))
    except ValueError:
        pass
    else:
        raise AssertionError("unfrozen teacher accepted")
    assert lexical_settings(1.0, "site-asked", "token-ce")["lexical_recipe"].startswith(
        "frozen-decoder-vocabulary-cross-entropy"
    )
    try:
        lexical_settings(1.0, "site-asked", "other")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown objective accepted")


def test_memory_snapping_returns_exact_nearest_rows_and_requires_frozen_table():
    from encoder_lora import MemoryTokenSnapper, snap_memory_to_tokens

    torch.manual_seed(7)
    embedding = nn.Embedding(64, 8).requires_grad_(False)
    embedding.weight.data = embedding.weight.data.to(torch.bfloat16)
    snapper = MemoryTokenSnapper(embedding)
    assert snapper.table.dtype == torch.bfloat16
    assert snapper.table_normalized.dtype == torch.float32
    ids = torch.tensor([3, 17, 42, 0])
    exact = embedding.weight[ids].clone()
    snapped, found, top1 = snapper(exact * 5.0)  # scale must not matter
    assert torch.equal(found, ids)
    assert torch.equal(snapped, exact) and snapped.dtype == torch.bfloat16
    assert tuple(top1.shape) == (4,) and bool((top1 > 0.999).all())
    perturbed = exact.float() + 0.05 * torch.randn(4, 8)
    snapped, found, top1 = snap_memory_to_tokens(
        perturbed, snapper.table_normalized, snapper.table
    )
    assert torch.equal(found, ids) and torch.equal(snapped, exact)
    assert bool((top1 < 1.0).all()) and bool((top1 > 0.9).all())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        again, _, _ = snapper(perturbed)
    assert torch.equal(again, exact)
    for bad in [torch.randn(4, 4), torch.randn(8), torch.randn(0, 8)]:
        try:
            snapper(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("memory geometry mismatch accepted")
    try:
        snap_memory_to_tokens(perturbed, snapper.table_normalized[:32], snapper.table)
    except ValueError:
        pass
    else:
        raise AssertionError("table geometry mismatch accepted")
    try:
        MemoryTokenSnapper(nn.Embedding(64, 8))
    except ValueError:
        pass
    else:
        raise AssertionError("unfrozen embedding accepted")


def test_cached_states_pad_right_and_adapter_digest_tracks_weights():
    from encoder_lora import adapter_digest, cached_case_states, inject_lora

    cache = {"a": torch.ones(3, 4), "b": 2 * torch.ones(5, 4)}
    states, mask = cached_case_states(cache, ["a", "b", "a"])
    assert states.shape == (3, 5, 4) and mask.tolist()[0] == [1, 1, 1, 0, 0]
    assert torch.equal(states[1], cache["b"]) and states[0, 3:].abs().sum() == 0
    for bad in [
        {},
        {"x": torch.ones(0, 4)},
        {"a": torch.ones(3, 4), "c": torch.ones(2, 5)},
    ]:
        try:
            cached_case_states(bad, list(bad) or ["missing"])
        except (ValueError, KeyError):
            pass
        else:
            raise AssertionError("invalid cache accepted")

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.v_proj = nn.Linear(8, 8), nn.Linear(8, 8)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = Block()

    torch.manual_seed(1)
    block = Encoder()
    adapters = inject_lora(block, 2, 4, True)
    before = adapter_digest(adapters)
    assert before == adapter_digest(adapters)
    with torch.no_grad():
        next(iter(adapters.values())).B.add_(1)
    assert adapter_digest(adapters) != before


def test_warmup_schedule_ramps_then_decays_to_floor():
    from encoder_lora import warmup_lr_scale

    assert warmup_lr_scale(5, 1000, "constant") == 1.0
    values = [warmup_lr_scale(step, 1000, "cosine") for step in range(1000)]
    assert values[0] < values[50] < values[99] <= 1.0
    assert abs(values[99] - 1.0) < 1e-9 and values[500] < values[100]
    assert abs(values[-1] - 0.01) < 1e-3
    for step, total, schedule in [(1000, 1000, "cosine"), (0, 10, "linear")]:
        try:
            warmup_lr_scale(step, total, schedule)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid schedule arguments accepted")


def test_document_batch_padding_masks_only_padded_rows():
    from encoder_lora import pad_document_batch

    docs = [torch.ones(1, 3, 4), 2 * torch.ones(1, 5, 4)]
    batch, mask = pad_document_batch(docs)
    assert batch.shape == (2, 5, 4) and mask.shape == (2, 5)
    assert mask.tolist() == [[False] * 3 + [True] * 2, [False] * 5]
    assert torch.equal(batch[0, :3], docs[0][0]) and batch[0, 3:].abs().sum() == 0
    assert torch.equal(batch[1], docs[1][0])
    for bad in [[], [torch.ones(2, 3, 4)], [torch.ones(1, 3, 4), torch.ones(1, 2, 5)]]:
        try:
            pad_document_batch(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid document batch accepted")


def test_name_slot_weights_emphasize_the_asked_name_and_keep_mean_one():
    from encoder_lora import (
        LexicalVocabularyTeacher,
        lexical_slot_weights,
        lexical_target_ids,
        lexical_token_ce_loss,
    )

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    rows = data_rows(17, "dev", 2)
    name_case = next(r for r in rows if r["category"] == "name")
    code_case = next(r for r in rows if r["category"] == "code")
    tok = Tokenizer()
    plain = lexical_slot_weights(tok, name_case, "asked-site", 64, 1.0, True)
    assert torch.equal(plain, torch.ones(64))
    assert torch.equal(
        lexical_slot_weights(tok, code_case, "asked-site", 64, 3.0, True),
        torch.ones(64),
    )
    weighted = lexical_slot_weights(tok, name_case, "asked-site", 64, 3.0, True)
    label = len(" key holder")
    count = len(" " + name_case["answer"])
    assert abs(float(weighted.mean()) - 1.0) < 1e-6
    assert bool((weighted[label : label + count] > weighted[0]).all())
    assert bool((weighted[label + count :] == weighted[0]).all())
    torch.manual_seed(2)
    embedding = nn.Embedding(128, 8).requires_grad_(False)
    teacher = LexicalVocabularyTeacher(embedding)
    text = " key holder Ab Cd at Ef Depot"
    ids = torch.tensor(lexical_target_ids(tok, text, True, 32))
    predicted = torch.randn(32, 8)
    weights = torch.ones(32)
    weights[2:7] = 3.0
    weights = weights * (32 / weights.sum())
    unweighted = lexical_token_ce_loss(predicted, ids, teacher, 32)
    reweighted = lexical_token_ce_loss(predicted, ids, teacher, 32, weights)
    assert abs(float(unweighted) - float(reweighted)) > 1e-6
    try:
        lexical_token_ce_loss(predicted, ids, teacher, 32, torch.ones(16))
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched slot weights accepted")


def test_memory_rescale_matches_teacher_norm_without_changing_direction():
    from encoder_lora import rescale_memory_projection

    class Reader(nn.Module):
        def __init__(self):
            super().__init__()
            self.memory_projection = nn.Linear(4, 8)

    torch.manual_seed(3)
    reader = Reader()
    inputs = torch.randn(5, 16, 4)
    before = [reader.memory_projection(x) for x in inputs]
    factor = rescale_memory_projection(reader, before, 1.5)
    after = [reader.memory_projection(x) for x in inputs]
    assert abs(torch.stack([a.norm(dim=-1).mean() for a in after]).mean() - 1.5) < 1e-4
    for a, b in zip(after, before):
        assert torch.allclose(a, b * factor, atol=1e-5)
    try:
        rescale_memory_projection(reader, [torch.zeros(16, 8)], 1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("zero memories accepted for rescale")


def test_expanded_memory_queries_keep_released_rows_and_update_config():
    import dataclasses

    from encoder_lora import expand_memory_queries, memory_slot_count

    @dataclasses.dataclass(frozen=True)
    class Config:
        num_memories: int

    class Reader(nn.Module):
        def __init__(self):
            super().__init__()
            self.memory_queries = nn.Parameter(torch.randn(1, 16, 8))
            self.config = Config(num_memories=16)

    torch.manual_seed(11)
    reader = Reader()
    released = reader.memory_queries.detach().clone()
    before = torch.get_rng_state()
    assert expand_memory_queries(reader, 16, 23) is reader
    assert torch.equal(reader.memory_queries, released)
    expand_memory_queries(reader, 24, 23)
    assert torch.equal(before, torch.get_rng_state())  # global stream untouched
    queries = reader.memory_queries
    assert isinstance(queries, nn.Parameter) and queries.requires_grad
    assert queries.shape == (1, 24, 8) and reader.config.num_memories == 24
    assert dict(reader.named_parameters())["memory_queries"] is queries
    assert torch.equal(queries[0, :16], released[0])
    for row in range(16, 24):
        delta = queries[0, row] - released[0, row % 16]
        assert 0 < float(delta.abs().max()) < 0.1 * float(released[0, row % 16].std())
    again = Reader()
    with torch.no_grad():
        again.memory_queries.copy_(released)
    expand_memory_queries(again, 24, 23)
    assert torch.equal(again.memory_queries, queries)  # seeded noise
    with torch.no_grad():
        again.memory_queries.copy_(queries)
    expand_memory_queries(again, 24, 7)
    assert torch.equal(again.memory_queries, queries)  # same count is a no-op
    for bad in (8, 23):
        try:
            expand_memory_queries(again, bad, 23)
        except ValueError:
            pass
        else:
            raise AssertionError("memory queries shrank")
    for bad in (15, 65, 24.0, True):
        try:
            memory_slot_count(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("memory slot count outside 16..64 accepted")
    assert memory_slot_count(64) == 64


def test_lexical_helpers_pad_and_check_the_configured_memory_slots():
    from encoder_lora import (
        LexicalVocabularyTeacher,
        lexical_alignment_loss,
        lexical_target_ids,
        lexical_targets,
        lexical_token_ce_loss,
    )

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    torch.manual_seed(3)
    embedding = nn.Embedding(128, 8).requires_grad_(False)
    teacher = LexicalVocabularyTeacher(embedding)
    ids = lexical_target_ids(Tokenizer(), " Ab, Cd9-001234", slots=24)
    assert len(ids) == 24 and ids[:16] == lexical_target_ids(
        Tokenizer(), " Ab, Cd9-001234"
    )
    assert ids[16:] == [ord(" ")] * 8
    long = " " + "z" * 20
    padded = lexical_target_ids(Tokenizer(), long, slots=24)
    assert len(padded) == 24 and padded[20] == ord("z") and padded[21] == ord(" ")
    assert (
        lexical_target_ids(Tokenizer(), " " + "z" * 30, truncate=True, slots=24)
        == [ord(" ")] + [ord("z")] * 23
    )
    try:
        lexical_target_ids(Tokenizer(), long)
    except ValueError:
        pass
    else:
        raise AssertionError("twenty-one tokens fit sixteen slots")
    target = lexical_targets(Tokenizer(), embedding, long, slots=24)
    assert target.shape == (24, 8)
    assert float(lexical_alignment_loss(target.clone(), target, slots=24)) < 1e-6
    ids = torch.tensor(ids)
    loss = lexical_token_ce_loss(torch.randn(24, 8), ids, teacher, slots=24)
    batched = lexical_token_ce_loss(
        torch.randn(2, 24, 8), torch.stack([ids, ids]), teacher, 24
    )
    assert torch.isfinite(loss) and torch.isfinite(batched)
    for predicted, target_ids, slots in [
        (torch.randn(16, 8), ids[:16], 24),
        (torch.randn(24, 8), ids, 16),
        (torch.randn(24, 8), ids[:16], 24),
    ]:
        try:
            lexical_token_ce_loss(predicted, target_ids, teacher, slots)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("slot count mismatch accepted")
    try:
        lexical_alignment_loss(target[:16], target[:16])
        lexical_alignment_loss(target[:16], target[:16], slots=24)
    except ValueError:
        pass
    else:
        raise AssertionError("sixteen rows accepted as twenty-four slots")


def test_memory_slot_lineage_is_a_contract_and_initialization_accepts_released_parent(
    tmp_path,
):
    from encoder_lora_select import derive_candidate
    from encoder_lora import SCHEMA, save_json, digest, candidate_proof
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
    parent.update(memory_slots=24, steps=8, versions={"torch": "pinned"})
    continuation_contract(parent, dict(parent, steps=16), 8)
    try:
        continuation_contract(parent, dict(parent, steps=16, memory_slots=32), 8)
    except ValueError:
        pass
    else:
        raise AssertionError("continuation changed the memory slot count")

    def run_with(slots, name):
        run = tmp_path / name
        run.mkdir()
        (run / "harness_source.py").write_text("synthetic unit-only training source")
        protocol = {
            "schema": SCHEMA,
            "memory_slots": slots,
            "steps": 1,
            "script_sha256": digest(run / "harness_source.py"),
        }
        save_json(run / "protocol.json", protocol)
        save_json(run / "base-proofs.json", {"synthetic": True})
        save_json(
            run / "result.json",
            {"completed": True, "frozen_base_versions_unchanged": True},
        )
        snapshot = run / "1.pt"
        torch.save(
            {
                "protocol": protocol,
                "step": 1,
                "reader": {"memory_queries": torch.zeros(1, slots, 4)},
                "adapters": {"y": torch.ones(1)},
                "result": {
                    "completed": False,
                    "history": [{"step": 1}],
                    "evaluations": [{"step": 1, "rows": []}],
                },
            },
            snapshot,
        )
        return run, snapshot

    run, snapshot = run_with(24, "wide")
    candidate = derive_candidate(run, [snapshot], tmp_path / "wide-candidate")
    assert candidate["memory_slots"] == 24
    assert candidate_proof(tmp_path / "wide-candidate", 24)["memory_slots"] == 24
    for slots in (16, 32):
        try:
            candidate_proof(tmp_path / "wide-candidate", slots)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "a 24-slot candidate initialized a %d-slot run" % slots
            )
    run, snapshot = run_with(16, "released")
    derive_candidate(run, [snapshot], tmp_path / "released-candidate")
    for slots in (16, 24, 64):
        assert (
            candidate_proof(tmp_path / "released-candidate", slots)["memory_slots"]
            == 16
        )
    mismatched, snapshot = run_with(32, "mismatched")
    save_json(
        mismatched / "protocol.json",
        dict(json.loads((mismatched / "protocol.json").read_text()), memory_slots=16),
    )
    state = torch.load(snapshot, weights_only=True)
    state["protocol"]["memory_slots"] = 16
    torch.save(state, snapshot)
    try:
        derive_candidate(mismatched, [snapshot], tmp_path / "mismatched-candidate")
    except ValueError:
        pass
    else:
        raise AssertionError("reader slots differing from the protocol were exported")


def test_encoder_states_select_the_final_or_an_intermediate_layer():
    from types import SimpleNamespace
    from encoder_lora import check_encoder_layer, encoder_states

    hidden = tuple(torch.full((2, 3, 4), float(i)) for i in range(5))  # 4 layers

    class FakeEncoder:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                last_hidden_state=torch.full((2, 3, 4), -1.0),
                hidden_states=hidden if kwargs.get("output_hidden_states") else None,
            )

    encoder = FakeEncoder()
    tokens = {"input_ids": torch.ones(2, 3, dtype=torch.long)}
    final = encoder_states(encoder, tokens, None)
    assert torch.equal(final, torch.full((2, 3, 4), -1.0))
    assert "output_hidden_states" not in encoder.calls[-1]
    assert encoder.calls[-1]["use_cache"] is False
    for layer in (1, 2, 4):
        states = encoder_states(encoder, tokens, layer)
        assert torch.equal(states, hidden[layer])
        assert encoder.calls[-1]["output_hidden_states"] is True
    for layer in (0, 5, -1):
        try:
            encoder_states(encoder, tokens, layer)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-range encoder layer %r accepted" % layer)
    assert check_encoder_layer(None, 36) is None
    assert check_encoder_layer(36, 36) == 36
    for layer in (0, 37, True, 2.0):
        try:
            check_encoder_layer(layer, 36)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid encoder layer %r accepted" % layer)


def test_dual_layer_features_expose_final_and_intermediate_token_keys():
    from types import SimpleNamespace
    from encoder_lora import (
        cached_case_states,
        cached_case_states_dual,
        check_dual_layer,
        document_features,
        document_features_dual,
        encoder_states_pair,
    )

    torch.manual_seed(7)
    final, layer = torch.randn(3, 12, 2560), torch.randn(3, 12, 2560)
    mask = torch.zeros(3, 12, dtype=torch.long)
    for row, length in enumerate([12, 7, 5]):
        mask[row, :length] = 1
    docs, query = document_features_dual(final, layer, mask)
    final_docs, final_query = document_features(final, mask, "tokens")
    layer_docs, _ = document_features(layer, mask, "tokens")
    assert docs.shape == (1, 38, 2560) and final_docs.shape == (1, 19, 2560)
    assert torch.equal(docs[:, :19], final_docs)
    assert torch.equal(docs[:, 19:], layer_docs)
    assert torch.equal(query, final_query)
    assert not torch.equal(docs[:, :19], docs[:, 19:])
    for bad in [(final, layer[:, :11], mask), (final[:1], layer[:1], mask[:1])]:
        try:
            document_features_dual(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid dual feature request accepted")
    # The combined mode is never a single-tensor feature mode.
    try:
        document_features(final, mask, "tokens-dual")
    except ValueError:
        pass
    else:
        raise AssertionError("tokens-dual accepted without the second layer")

    hidden = tuple(torch.full((2, 3, 4), float(i)) for i in range(5))  # 4 layers
    calls = []

    def fake_encoder(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            last_hidden_state=torch.full((2, 3, 4), -1.0),
            hidden_states=hidden if kwargs.get("output_hidden_states") else None,
        )

    tokens = {"input_ids": torch.ones(2, 3, dtype=torch.long)}
    final_states, layer_states = encoder_states_pair(fake_encoder, tokens, 2)
    assert torch.equal(final_states, torch.full((2, 3, 4), -1.0))
    assert torch.equal(layer_states, hidden[2])
    assert len(calls) == 1 and calls[0]["output_hidden_states"] is True
    assert calls[0]["use_cache"] is False
    for bad_layer in (0, 4, 5):
        try:
            encoder_states_pair(fake_encoder, tokens, bad_layer)
        except ValueError:
            pass
        else:
            raise AssertionError("dual layer %r accepted" % bad_layer)
    assert check_dual_layer(None, 36) is None
    assert check_dual_layer(35, 36) == 35
    for bad_layer in (0, 36, True, 2.0):
        try:
            check_dual_layer(bad_layer, 36)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid dual layer %r accepted" % bad_layer)

    cache = {
        "a": {"final": torch.ones(3, 4), "layer": 3 * torch.ones(3, 4)},
        "b": {"final": 2 * torch.ones(5, 4), "layer": 4 * torch.ones(5, 4)},
    }
    final_rows, layer_rows, mask = cached_case_states_dual(cache, ["a", "b", "a"])
    expected, expected_mask = cached_case_states(
        {k: v["final"] for k, v in cache.items()}, ["a", "b", "a"]
    )
    assert torch.equal(final_rows, expected) and torch.equal(mask, expected_mask)
    assert final_rows.shape == (3, 5, 4) and mask.tolist()[0] == [1, 1, 1, 0, 0]
    assert torch.equal(layer_rows[1], cache["b"]["layer"])
    assert layer_rows[0, 3:].abs().sum() == 0 and layer_rows[0, :3].sum() == 36
    cache["b"]["layer"] = torch.ones(4, 4)
    try:
        cached_case_states_dual(cache, ["a", "b"])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched dual cache rows accepted")


def test_dual_layer_flags_are_validated_before_the_model_loads():
    from encoder_lora import check_dual_layer_flags

    check_dual_layer_flags("tokens-dual", None, 24)
    check_dual_layer_flags("tokens", 24, None)
    check_dual_layer_flags("pooled", None, None)
    for mode, encoder_layer, dual_layer in [
        ("tokens-dual", None, None),
        ("tokens", None, 24),
        ("tokens-dual", None, 0),
        ("tokens-dual", 36, 24),
    ]:
        try:
            check_dual_layer_flags(mode, encoder_layer, dual_layer)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid dual-layer flags accepted: %r" % (mode,))


def test_lora_targets_select_modules_and_added_targets_initialize_at_zero_effect():
    from encoder_lora import (
        DEFAULT_LORA_TARGETS,
        LORA_TARGET_MODULES,
        initialization_adds_targets,
        lora_targets,
    )

    assert lora_targets("q_proj,v_proj") == DEFAULT_LORA_TARGETS
    assert lora_targets("down_proj, o_proj,q_proj") == ("q_proj", "o_proj", "down_proj")
    assert lora_targets(",".join(reversed(LORA_TARGET_MODULES))) == LORA_TARGET_MODULES
    for spec in ("", "q_proj,q_proj", "q_proj,lm_head", "norm"):
        try:
            lora_targets(spec)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid target list %r accepted" % spec)

    def block():
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.ModuleDict(
                    {
                        k: nn.Linear(8, 8)
                        for k in ("q_proj", "k_proj", "v_proj", "o_proj")
                    }
                )
                self.mlp = nn.ModuleDict(
                    {
                        "gate_proj": nn.Linear(8, 16),
                        "up_proj": nn.Linear(8, 16),
                        "down_proj": nn.Linear(16, 8),
                    }
                )

        torch.manual_seed(7)
        return Encoder().requires_grad_(False)

    model = block()
    adapters = inject_lora(model, 2, 4, True, ("q_proj", "o_proj"))
    assert set(adapters) == {"attn.q_proj", "attn.o_proj"}
    assert isinstance(model.attn.q_proj, LoRALinear)
    assert isinstance(model.attn.k_proj, nn.Linear)
    assert inject_lora(block(), 2, 4, False).keys() == {"attn.q_proj", "attn.v_proj"}
    try:
        inject_lora(block(), 2, 4, False, ("q_proj", "lm_head"))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown adapter target accepted")

    # Parent trained Q/V; the child adds O and down_proj.
    parent = inject_lora(block(), 2, 4, True)
    with torch.no_grad():
        for module in parent.values():
            module.A.fill_(0.25)
            module.B.fill_(0.5)
    parent_state = adapter_state(parent)
    child = inject_lora(
        block(), 2, 4, True, ("q_proj", "v_proj", "o_proj", "down_proj")
    )
    with torch.no_grad():
        child["attn.o_proj"].B.fill_(0.75)
    try:
        restore_adapters(child, parent_state)
    except ValueError:
        pass
    else:
        raise AssertionError("strict restore accepted a geometry mismatch")
    restore_adapters(child, parent_state, allow_missing=True)
    for name in ("attn.q_proj", "attn.v_proj"):
        assert torch.equal(child[name].A, parent_state[name + ".A"])
        assert torch.equal(child[name].B, parent_state[name + ".B"])
    for name in ("attn.o_proj", "mlp.down_proj"):
        assert not child[name].B.any()
        x = torch.randn(3, child[name].base.in_features)
        assert torch.equal(child[name](x), child[name].base(x))
    # Partial entries, extra entries and shape changes are never accepted.
    partial = dict(parent_state)
    del partial["attn.q_proj.B"]
    for state in (
        partial,
        dict(parent_state, **{"attn.k_proj.A": torch.zeros(2, 8)}),
        dict(parent_state, **{"attn.q_proj.B": torch.zeros(8, 3)}),
    ):
        try:
            restore_adapters(child, state, allow_missing=True)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed adapter state accepted")
    # Fewer-target children cannot inherit a wider parent.
    assert (
        initialization_adds_targets(("q_proj", "v_proj"), ("q_proj", "v_proj")) is False
    )
    assert initialization_adds_targets(
        ("q_proj", "v_proj"), ("q_proj", "v_proj", "o_proj")
    )
    for parent_targets, child_targets in [
        (("q_proj", "v_proj", "o_proj"), ("q_proj", "v_proj")),
        (("q_proj", "v_proj"), ("q_proj", "o_proj")),
        ((), ("q_proj",)),
    ]:
        try:
            initialization_adds_targets(parent_targets, child_targets)
        except ValueError:
            pass
        else:
            raise AssertionError("child dropped parent adapter targets")


def test_copy_head_gates_between_generated_memories_and_exact_token_copies():
    from encoder_lora import COPY_HEAD_TEMPERATURE, CopyHead

    torch.manual_seed(5)
    width, slots, tokens = 8, 4, 6
    head = CopyHead(width)
    assert head.temperature == COPY_HEAD_TEMPERATURE
    assert all(p.dtype == torch.float32 for p in head.parameters())
    assert torch.equal(head.query.weight, torch.eye(width))
    memories = torch.randn(slots, width)
    # Orthogonal token rows of distinct norms: cosine attention at tau 0.05 is
    # then one-hot to within exp(-20) when a query lies on a row.
    table = torch.linalg.qr(torch.randn(width, width))[0][:tokens] * torch.arange(
        1, tokens + 1
    ).unsqueeze(1)
    output, probs, gate = head(memories, table)
    assert output.shape == (slots, width) and probs.shape == (slots, tokens)
    assert gate.shape == (slots,)
    # A fresh head mostly generates: sigmoid(-2) on every slot.
    assert torch.allclose(gate, torch.full((slots,), 1 / (1 + math.exp(2.0))))
    assert torch.allclose(probs.sum(dim=-1), torch.ones(slots), atol=1e-6)
    expected = gate[:, None] * (probs @ table) + (1 - gate[:, None]) * memories
    assert torch.allclose(output, expected, atol=1e-6)
    # Gate closed -> the memories pass through unchanged.
    with torch.no_grad():
        head.gate.bias.fill_(-1e4)
    closed, _, gate = head(memories, table)
    assert torch.allclose(closed, memories) and float(gate.max()) < 1e-6
    # Gate open with a memory lying on a token direction -> that exact row.
    with torch.no_grad():
        head.gate.bias.fill_(1e4)
    aligned = memories.clone()
    aligned[0] = 3 * table[2]
    opened, probs, gate = head(aligned, table)
    assert float(gate.min()) > 1 - 1e-6
    assert probs[0].argmax() == 2 and float(probs[0, 2]) > 1 - 1e-6
    assert torch.allclose(opened[0], table[2], atol=1e-5)
    # Masked tokens receive zero probability and never enter the copy.
    mask = torch.tensor([True, True, False, True, False, True])
    _, masked_probs, _ = head(aligned, table, mask)
    assert float(masked_probs[:, ~mask].abs().max()) == 0
    assert torch.allclose(masked_probs.sum(dim=-1), torch.ones(slots), atol=1e-6)
    # Batched and padded forms agree with the per-case form.
    with torch.no_grad():
        head.gate.bias.fill_(-2.0)
        head.gate.weight.normal_()
        head.query.weight.add_(0.1 * torch.randn(width, width))
    other = torch.randn(slots, width)
    short = torch.randn(3, width)
    padded = torch.zeros(2, tokens, width)
    padded[0], padded[1, :3] = table, short
    batch_mask = torch.tensor([[True] * tokens, [True] * 3 + [False] * 3])
    out_b, probs_b, gate_b = head(torch.stack([memories, other]), padded, batch_mask)
    out_0, probs_0, gate_0 = head(memories, table)
    out_1, probs_1, gate_1 = head(other, short)
    assert torch.allclose(out_b[0], out_0, atol=1e-6)
    assert torch.allclose(out_b[1], out_1, atol=1e-6)
    assert torch.allclose(probs_b[0], probs_0, atol=1e-6)
    assert torch.allclose(probs_b[1, :, :3], probs_1, atol=1e-6)
    assert float(probs_b[1, :, 3:].abs().max()) == 0
    assert torch.allclose(gate_b, torch.stack([gate_0, gate_1]), atol=1e-6)
    # The gradient reaches both the head and the generating reader output.
    leaf = memories.clone().requires_grad_(True)
    trained, _, _ = head(leaf, table)
    trained.square().sum().backward()
    assert leaf.grad.abs().sum() > 0
    assert (
        head.gate.weight.grad.abs().sum() > 0 and head.query.weight.grad.abs().sum() > 0
    )
    # State round trip reproduces the outputs exactly.
    clone = CopyHead(width)
    clone.load_state_dict({k: v.clone() for k, v in head.state_dict().items()})
    again, _, _ = clone(memories, table)
    assert torch.equal(again, head(memories, table)[0])
    for bad in [
        lambda: head(memories, torch.randn(tokens, width + 1)),
        lambda: head(memories, table, torch.zeros(tokens, dtype=torch.bool)),
        lambda: head(memories, table, torch.ones(tokens)),
        lambda: head(torch.randn(2, slots, width), table.unsqueeze(0)),
        lambda: CopyHead(width, 0.0),
    ]:
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid copy head geometry accepted")


def test_pointer_supervision_marks_copyable_slots_and_scores_copy_logits():
    from encoder_lora import (
        CopyHead,
        copy_excluded_ids,
        copyable_mask,
        pointer_losses,
    )

    class Tokenizer:
        vocabulary = {
            " ": 0,
            " code": 7,
            " key": 8,
            " holder": 9,
            " at": 1,
            " UNKNOWN": 6,
        }

        def encode(self, text, add_special_tokens):
            assert not add_special_tokens
            if text == " key holder":
                return [8, 9]
            return [self.vocabulary[text]]

    excluded = copy_excluded_ids(Tokenizer())
    assert excluded == {0, 7, 8, 9, 1, 6}
    # Slot targets: row 0 holds a label (7) that the document also contains, a
    # value (3) at two positions, the pad (0) and a value (5) at one position;
    # row 1 holds a value only at a padded position (4), a repeated value (2),
    # an absent value (11) and the `` at`` label (1) present in the document.
    targets = torch.tensor([[7, 3, 0, 5], [4, 2, 11, 1]])
    docs = torch.tensor([[3, 7, 5, 3, 0], [2, 1, 2, 4, 0]])
    mask = torch.tensor([[True] * 5, [True, True, True, False, False]])
    copyable, match = copyable_mask(targets, docs, mask, excluded)
    assert copyable.tolist() == [
        [False, True, False, True],
        [False, True, False, False],
    ]
    assert match.dtype == torch.bool and match.shape == (2, 4, 5)
    assert match[0, 1].tolist() == [True, False, False, True, False]
    assert match[0, 3].tolist() == [False, False, True, False, False]
    assert match[1, 1].tolist() == [True, False, True, False, False]
    # Non-copyable slots carry no match, padding never matches, and without
    # exclusions the label and pad slots become copyable.
    assert not match[~copyable].any()
    assert not match[:, :, 3:][1].any()
    plain, _ = copyable_mask(targets, docs, mask, set())
    assert plain.tolist() == [[True, True, True, True], [False, True, False, True]]
    # All copy mass on matching positions -> zero pointer loss; the gate logits
    # agreeing with copyability -> zero gate loss.
    logits = torch.full((2, 4, 5), -40.0)
    logits[match] = 40.0
    gates = torch.where(copyable, 40.0, -40.0)
    pointer, gate = pointer_losses(logits, gates, match, copyable, mask)
    assert float(pointer) < 1e-6 and float(gate) < 1e-6
    # Mass on a non-matching position is penalized; mass split across the two
    # matching positions of a slot is not (log-sum-exp over the matches).
    wrong = logits.clone()
    wrong[0, 1] = torch.tensor([-40.0, 40.0, -40.0, -40.0, -40.0])
    pointer_wrong, _ = pointer_losses(wrong, gates, match, copyable, mask)
    assert float(pointer_wrong) > 10
    split = logits.clone()
    split[0, 1] = torch.tensor([40.0, -40.0, -40.0, 40.0, -40.0])
    pointer_split, _ = pointer_losses(split, gates, match, copyable, mask)
    assert float(pointer_split) < 1e-6
    # The gate loss pushes copyable slots toward 1 and the others toward 0.
    zero_gates = torch.zeros(2, 4, requires_grad=True)
    _, gate_zero = pointer_losses(logits, zero_gates, match, copyable, mask)
    assert abs(float(gate_zero) - math.log(2.0)) < 1e-6
    gate_zero.backward()
    assert (zero_gates.grad[copyable] < 0).all() and (
        zero_gates.grad[~copyable] > 0
    ).all()
    assert float(pointer_losses(logits, -gates, match, copyable, mask)[1]) > 30
    # Masked positions never carry mass even with large logits there.
    leaked = logits.clone()
    leaked[1, 1, 3:] = 80.0
    pointer_leaked, _ = pointer_losses(leaked, gates, match, copyable, mask)
    assert float(pointer_leaked) < 1e-6
    # No copyable slot -> zero pointer loss, gate loss still defined.
    none_copyable = torch.zeros(2, 4, dtype=torch.bool)
    pointer_none, gate_none = pointer_losses(
        logits, gates, torch.zeros_like(match), none_copyable, mask
    )
    assert float(pointer_none) == 0 and abs(float(gate_none) - 15.0) < 1e-4
    # The head exposes the logits the losses consume, consistent with forward.
    torch.manual_seed(3)
    head, width = CopyHead(8), 8
    memories, tokens = torch.randn(2, 4, width), torch.randn(2, 5, width)
    out, probs, gate_prob = head(memories, tokens, mask)
    out_d, copy_logits, gate_logits = head.forward_detailed(memories, tokens, mask)
    assert torch.equal(out, out_d)
    assert torch.allclose(copy_logits.softmax(dim=-1), probs)
    assert torch.allclose(torch.sigmoid(gate_logits), gate_prob)
    assert bool(torch.isinf(copy_logits[1, :, 3:]).all())
    pointer_head, gate_head = pointer_losses(
        copy_logits, gate_logits, match, copyable, mask
    )
    assert float(pointer_head) > 0 and float(gate_head) > 0
    (pointer_head + gate_head).backward()
    assert (
        head.query.weight.grad.abs().sum() > 0 and head.gate.weight.grad.abs().sum() > 0
    )
    for bad in [
        lambda: copyable_mask(targets[0], docs, mask, excluded),
        lambda: copyable_mask(targets, docs, mask[:, :4], excluded),
        lambda: copyable_mask(targets, docs, mask.float(), excluded),
        lambda: copyable_mask(targets, docs[:1], mask[:1], excluded),
        lambda: pointer_losses(logits[:, :, :4], gates, match, copyable, mask),
        lambda: pointer_losses(logits, gates[:, :3], match, copyable, mask),
        lambda: pointer_losses(logits, gates, match.float(), copyable, mask),
        lambda: pointer_losses(logits, gates, match, ~copyable, mask),
        lambda: pointer_losses(logits, gates, match, copyable, torch.zeros_like(mask)),
    ]:
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pointer supervision geometry accepted")


def test_document_token_embeddings_concatenate_documents_and_pad_batches():
    from encoder_lora import (
        assert_shared_token_ids,
        document_token_embeddings,
        document_token_ids,
        pad_token_embeddings,
    )

    class Tokenizer:
        def __init__(self, shift=0):
            self.shift = shift

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [ord(c) + self.shift for c in text]

    torch.manual_seed(3)
    embedding = nn.Embedding(128, 8).requires_grad_(False)
    documents = ["ab", "cde"]
    ids = document_token_ids(Tokenizer(), documents)
    assert ids.tolist() == [ord(c) for c in "abcde"] and ids.dtype == torch.long
    rows, same_ids = document_token_embeddings(Tokenizer(), embedding, documents)
    assert torch.equal(same_ids, ids)
    assert rows.shape == (5, 8) and rows.dtype == torch.float32
    assert not rows.requires_grad
    assert torch.equal(rows, embedding.weight[ids])
    assert torch.equal(
        assert_shared_token_ids(Tokenizer(), Tokenizer(), documents), ids
    )
    try:
        assert_shared_token_ids(Tokenizer(), Tokenizer(1), documents)
    except ValueError:
        pass
    else:
        raise AssertionError("differing tokenizers passed as shared")
    other, other_ids = document_token_embeddings(Tokenizer(), embedding, ["xyz"])
    batch, batch_ids, mask = pad_token_embeddings([(rows, ids), (other, other_ids)])
    assert batch.shape == (2, 5, 8) and batch_ids.shape == (2, 5)
    assert mask.tolist() == [[True] * 5, [True] * 3 + [False] * 2]
    assert torch.equal(batch[0], rows) and torch.equal(batch[1, :3], other)
    assert float(batch[1, 3:].abs().sum()) == 0 and batch_ids[1, 3:].tolist() == [0, 0]
    for bad in [
        lambda: document_token_ids(Tokenizer(), []),
        lambda: document_token_ids(Tokenizer(), [""]),
        lambda: document_token_embeddings(Tokenizer(), nn.Embedding(128, 8), documents),
        lambda: pad_token_embeddings([]),
        lambda: pad_token_embeddings([(rows, ids), (other[:, :4], other_ids)]),
        lambda: pad_token_embeddings([(rows, ids[:2])]),
    ]:
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid copy source accepted")


def test_copy_head_candidates_export_verify_and_restore_their_module(tmp_path):
    from encoder_lora_select import derive_candidate
    from encoder_lora import (
        SCHEMA,
        CopyHead,
        candidate_proof,
        copy_head_state,
        digest,
        save_json,
    )
    from safetensors.torch import load_file

    torch.manual_seed(11)
    head = CopyHead(8)
    with torch.no_grad():
        head.gate.weight.normal_()

    def run_with(name, copy_head, snapshot_state):
        run = tmp_path / name
        run.mkdir()
        (run / "harness_source.py").write_text("synthetic unit-only training source")
        protocol = {
            "schema": SCHEMA,
            "memory_slots": 16,
            "steps": 1,
            "copy_head": copy_head,
            "copy_head_temperature": 0.05 if copy_head else None,
            "script_sha256": digest(run / "harness_source.py"),
        }
        save_json(run / "protocol.json", protocol)
        save_json(run / "base-proofs.json", {"synthetic": True})
        save_json(
            run / "result.json",
            {"completed": True, "frozen_base_versions_unchanged": True},
        )
        snapshot = run / "1.pt"
        torch.save(
            {
                "protocol": protocol,
                "step": 1,
                "reader": {"x": torch.ones(1)},
                "adapters": {"y": torch.ones(1)},
                **snapshot_state,
                "result": {
                    "completed": False,
                    "history": [{"step": 1}],
                    "evaluations": [{"step": 1, "rows": []}],
                },
            },
            snapshot,
        )
        return run, snapshot

    run, snapshot = run_with("copy", True, {"copy_head": copy_head_state(head)})
    out = tmp_path / "copy-candidate"
    candidate = derive_candidate(run, [snapshot], out)
    assert candidate["copy_head_sha256"] == digest(out / "copy-head.safetensors")
    assert candidate_proof(out)["copy_head"] is True
    restored = CopyHead(8)
    restored.load_state_dict(load_file(str(out / "copy-head.safetensors")))
    memories, table = torch.randn(4, 8), torch.randn(5, 8)
    assert torch.equal(restored(memories, table)[0], head(memories, table)[0])
    # A changed module is caught by the proof like any other artifact.
    (out / "copy-head.safetensors").write_bytes(b"tampered")
    try:
        candidate_proof(out)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered copy head passed the candidate proof")
    # A copy-head protocol needs the module in every snapshot, and a plain
    # protocol refuses a snapshot carrying one.
    run, snapshot = run_with("missing", True, {})
    try:
        derive_candidate(run, [snapshot], tmp_path / "missing-candidate")
    except ValueError:
        pass
    else:
        raise AssertionError("copy-head candidate exported without its module")
    run, snapshot = run_with("undeclared", False, {"copy_head": copy_head_state(head)})
    try:
        derive_candidate(run, [snapshot], tmp_path / "undeclared-candidate")
    except ValueError:
        pass
    else:
        raise AssertionError("undeclared copy head exported")
    # A plain candidate exports no copy head and the proof rejects a stray hash.
    run, snapshot = run_with("plain", False, {})
    plain = tmp_path / "plain-candidate"
    candidate = derive_candidate(run, [snapshot], plain)
    assert "copy_head_sha256" not in candidate
    assert not (plain / "copy-head.safetensors").exists()
    assert candidate_proof(plain)["copy_head"] is False
    save_json(plain / "candidate.json", dict(candidate, copy_head_sha256="0" * 64))
    try:
        candidate_proof(plain)
    except ValueError:
        pass
    else:
        raise AssertionError("stray copy head hash passed the candidate proof")


def test_phrasing_set_v2_varies_field_order_and_stays_parseable(tmp_path):
    from encoder_lora import CODE_PATTERN, lexical_record_text
    from encoder_lora_corpus import (
        HELD_OUT_PHRASES,
        broaden,
        expanded_data,
        name_counterfactual_data,
    )

    old = tmp_path / "old"
    make_data(old, train_groups=2, eval_groups=2)
    v1, again = tmp_path / "v1", tmp_path / "v1-again"
    expanded_data(v1, old, train_groups=20, fresh_dev_groups=2)
    expanded_data(again, old, train_groups=20, fresh_dev_groups=2, phrasing_set="v1")
    for name in ("train.json", "dev.json"):
        assert (v1 / name).read_bytes() == (again / name).read_bytes()
    assert json.loads((v1 / "manifest.json").read_text())["phrasing_set"] == "v1"

    base = broaden(data_rows(17, "train", 20))
    v2 = broaden(data_rows(17, "train", 20), "v2")
    assert base == broaden(data_rows(17, "train", 20), "v1")
    assert {r["template"] for r in base} == set(range(6))
    assert {r["template"] for r in v2} == set(range(10))
    assert [r["template"] for r in v2[:24]] == [r["template"] for r in base[:24]]
    for a, b in zip(base, v2):
        assert {k: v for k, v in a.items() if k not in ("documents", "template")} == {
            k: v for k, v in b.items() if k not in ("documents", "template")
        }
        site, other, owner, _ = b["entities"]
        assert [d for d in a["documents"] if other in d] == [
            d for d in b["documents"] if other in d
        ]
        assert [other in d for d in a["documents"]] == [
            other in d for d in b["documents"]
        ]
        (source,) = [d for d in b["documents"] if site in d]
        assert source.count(site) == 1
        assert not any(phrase in source for phrase in HELD_OUT_PHRASES)
        text = lexical_record_text(b, "asked-site")
        if b["category"] != "missing":
            assert "not recorded" not in source
            assert source.count(owner) == 1 and len(CODE_PATTERN.findall(source)) == 1
        elif "access code" in b["question"]:
            assert "not recorded" in source and not CODE_PATTERN.findall(source)
            assert text == f" code UNKNOWN at {site}"
        else:
            assert "not recorded" in source and owner not in source
            assert text == f" key holder UNKNOWN at {site}"

    paired = tmp_path / "paired-v2"
    manifest = name_counterfactual_data(paired, v1, phrasing_set="v2")
    assert manifest["phrasing_set"] == "v2"
    assert (paired / "dev.json").read_bytes() == (v1 / "dev.json").read_bytes()
    rows, original = load_split(paired, "train"), load_split(v1, "train")
    by_id = {r["id"]: r for r in rows}
    assert len(rows) == 100 and {r["template"] for r in rows} == set(range(10))
    for r in original:
        rephrased = by_id[r["id"]]
        assert {
            k: v for k, v in rephrased.items() if k not in ("documents", "template")
        } == {k: v for k, v in r.items() if k not in ("documents", "template")}
        assert rephrased["documents"] != r["documents"] or rephrased["template"] < 6
    for case in rows:
        lexical_record_text(case, "asked-site")
    for bad in (lambda: broaden(data_rows(1, "train", 4), "v4"),):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("unknown phrasing set accepted")


def test_phrasing_set_v3_grammar_varies_sentence_form(tmp_path):
    from encoder_lora import CODE_PATTERN, lexical_record_text
    from encoder_lora_corpus import (
        HELD_OUT_PHRASES,
        V3_ORDERS,
        broaden,
        expanded_data,
        name_counterfactual_data,
        site_document,
    )

    site, owner, value = "Bakolu Depot", "Mira Tola", "Xb3-001234"
    held_out_shape = re.compile(r"^Mira Tola [a-z' -]+ Bakolu Depot, (whose|which)")
    variants = ((False, False), (True, False), (False, True))
    skeletons, orders = set(), set()
    for index in range(2000):
        for missing_name, missing_code in variants:
            document, order = site_document(
                index, "v3", site, owner, value, missing_name, missing_code
            )
            assert (
                document
                == site_document(
                    index, "v3", site, owner, value, missing_name, missing_code
                )[0]
            )
            assert 0 <= order < len(V3_ORDERS)
            orders.add(order)
            assert document.count(site) == 1 and document.endswith(".")
            assert document.count(owner) == (0 if missing_name else 1)
            assert len(CODE_PATTERN.findall(document)) == (0 if missing_code else 1)
            assert ("not recorded" in document) == (missing_name or missing_code)
            assert not any(phrase in document for phrase in HELD_OUT_PHRASES)
            assert not held_out_shape.match(document)
            if index < 400 and not (missing_name or missing_code):
                skeletons.add(
                    document.replace(site, "{site}")
                    .replace(owner, "{owner}")
                    .replace(value, "{value}")
                )
    assert orders == set(range(6)) and len(skeletons) >= 50
    assert site_document(7, "v3", site, owner, value, False, False) != site_document(
        8, "v3", site, owner, value, False, False
    )

    base = broaden(data_rows(17, "train", 40))
    v3 = broaden(data_rows(17, "train", 40), "v3")
    assert {r["template"] for r in v3} == set(range(6))
    for a, b in zip(base, v3):
        assert {k: v for k, v in a.items() if k not in ("documents", "template")} == {
            k: v for k, v in b.items() if k not in ("documents", "template")
        }
        site, other, owner, _ = b["entities"]
        assert [d for d in a["documents"] if other in d] == [
            d for d in b["documents"] if other in d
        ]
        (source,) = [d for d in b["documents"] if site in d]
        assert other not in source and source not in a["documents"]
        text = lexical_record_text(b, "asked-site")
        if b["category"] != "missing":
            assert text in (
                f" code {b['answer']} at {site}",
                f" key holder {owner} at {site}",
            )
        elif "access code" in b["question"]:
            assert text == f" code UNKNOWN at {site}"
        else:
            assert text == f" key holder UNKNOWN at {site}"

    old = tmp_path / "old"
    make_data(old, train_groups=2, eval_groups=2)
    v1, v2, again = tmp_path / "v1", tmp_path / "v2", tmp_path / "v2-again"
    expanded_data(v1, old, train_groups=20, fresh_dev_groups=2)
    expanded_data(v2, old, train_groups=20, fresh_dev_groups=2, phrasing_set="v2")
    expanded_data(again, old, train_groups=20, fresh_dev_groups=2, phrasing_set="v2")
    for name in ("train.json", "dev.json"):
        assert (v2 / name).read_bytes() == (again / name).read_bytes()
    assert (v1 / "dev.json").read_bytes() == (v2 / "dev.json").read_bytes()
    paired = tmp_path / "paired-v3"
    manifest = name_counterfactual_data(paired, v1, phrasing_set="v3")
    assert manifest["phrasing_set"] == "v3"
    assert (paired / "dev.json").read_bytes() == (v1 / "dev.json").read_bytes()
    rows, original = load_split(paired, "train"), load_split(v1, "train")
    assert [r["id"] for r in rows[::5]] == [r["id"] for r in original[::4]]
    by_id = {r["id"]: r for r in rows}
    for r in original:
        rephrased = by_id[r["id"]]
        assert {
            k: v for k, v in rephrased.items() if k not in ("documents", "template")
        } == {k: v for k, v in r.items() if k not in ("documents", "template")}
        assert rephrased["documents"] != r["documents"]
    for case in rows:
        lexical_record_text(case, "asked-site")
    expanded = tmp_path / "expanded-v3"
    manifest = expanded_data(
        expanded, old, train_groups=12, fresh_dev_groups=2, phrasing_set="v3"
    )
    assert manifest["phrasing_set"] == "v3"
    for case in load_split(expanded, "train"):
        lexical_record_text(case, "asked-site")


def test_document_space_prefix_restores_the_record_segmentation_of_initial_names():
    from pathlib import Path

    from encoder_lora import document_token_ids, prefixed_documents

    documents = ["Mujapo Manabo is entrusted with code 7Q.", " Site B keeps code 7Q."]
    assert prefixed_documents(documents, False) == documents
    assert prefixed_documents(documents, False) is not documents
    prefixed = prefixed_documents(documents, True)
    assert prefixed == [" " + documents[0], documents[1]]
    # Idempotent: an already prefixed text keeps its single space.
    assert prefixed_documents(prefixed, True) == prefixed
    assert prefixed_documents([], True) == [] and prefixed_documents([], False) == []

    class Tokenizer:
        """Byte-pair stand-in: a leading space joins the first word of a text
        and a sentence-initial word is split into shorter pieces."""

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            ids = []
            for index, word in enumerate(text.split(" ")):
                if not word:
                    continue
                if index == 0:
                    ids.extend(hash("piece:" + piece) for piece in (word[:1], word[1:]))
                else:
                    ids.append(hash("word: " + word))
            return ids

    tokenizers = [Tokenizer()]
    model = Path(
        "/home/tim/antfly/.worktrees/plan-memorilla-native-memory/"
        ".benchmark-assets/memorilla/models/qwen3-8b"
    )
    if (model / "tokenizer.json").exists():
        from transformers import AutoTokenizer

        tokenizers.append(
            AutoTokenizer.from_pretrained(str(model), local_files_only=True)
        )
    for tokenizer in tokenizers:
        label = len(tokenizer.encode(" key holder", add_special_tokens=False))
        site = len(tokenizer.encode(" at B", add_special_tokens=False))
        record = document_token_ids(tokenizer, [" key holder Mujapo Manabo at B"])
        name = record.tolist()[label:-site]
        assert 2 <= len(name) <= 4

        def contains(ids):
            ids = ids.tolist()
            return any(ids[i : i + len(name)] == name for i in range(len(ids)))

        # Mid-sentence the record's name ids are present; sentence-initial they
        # are not, and the single-space prefix restores them.
        assert contains(document_token_ids(tokenizer, ["by " + documents[0]]))
        assert not contains(document_token_ids(tokenizer, documents))
        assert contains(
            document_token_ids(tokenizer, prefixed_documents(documents, True))
        )


def test_self_context_keeps_the_documents_own_ids_as_its_last_rows():
    from pathlib import Path

    from encoder_lora import (
        document_token_ids,
        prefixed_documents,
        select_last_rows,
        self_context_documents,
        self_context_states,
        self_context_tokens,
    )

    class Tokenizer:
        """Byte-pair stand-in: a leading space joins the first word of a text
        and a sentence-initial word is split into shorter pieces."""

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            ids = []
            for index, word in enumerate(text.split(" ")):
                if not word:
                    continue
                if index == 0:
                    ids.extend(hash("piece:" + piece) for piece in (word[:1], word[1:]))
                else:
                    ids.append(hash("word: " + word))
            return ids

    documents = ["Mujapo Manabo is entrusted with code 7Q.", " Site B keeps code 7Q."]
    tokenizers = [Tokenizer()]
    model = Path(
        "/home/tim/antfly/.worktrees/plan-memorilla-native-memory/"
        ".benchmark-assets/memorilla/models/qwen3-8b"
    )
    if (model / "tokenizer.json").exists():
        from transformers import AutoTokenizer

        tokenizers.append(
            AutoTokenizer.from_pretrained(str(model), local_files_only=True)
        )
    for tokenizer in tokenizers:
        prefixed = prefixed_documents(documents, True)
        for text in prefixed:
            # The prefixed form repeats as-is and its second copy tokenizes to
            # the document's own ids, so the kept states are one per id.
            doubled, count, single, fallback = self_context_tokens(tokenizer, text)
            assert doubled == text + text and not fallback
            assert single == tokenizer.encode(text, add_special_tokens=False)
            doubled_ids = tokenizer.encode(doubled, add_special_tokens=False)
            assert count == len(single) and len(doubled_ids) == 2 * count
            assert doubled_ids[-count:] == single
        # A sentence-initial word re-segments once it follows the first copy,
        # so the single ids fall back to the space-prefixed spelling (which
        # is what the kept states stand for) and the fallback is reported.
        doubled, count, single, fallback = self_context_tokens(tokenizer, documents[0])
        assert doubled == documents[0] + " " + documents[0] and fallback
        assert single == tokenizer.encode(" " + documents[0], add_special_tokens=False)
        assert tokenizer.encode(doubled, add_special_tokens=False)[-count:] == single
        assert not self_context_tokens(tokenizer, documents[1])[3]
        texts, counts, ids, fallbacks = self_context_documents(tokenizer, prefixed)
        assert texts == [t + t for t in prefixed] and fallbacks == 0
        assert counts == [int(i.numel()) for i in ids]
        assert torch.equal(torch.cat(ids), document_token_ids(tokenizer, prefixed))
        assert self_context_documents(tokenizer, documents)[3] == 1
    for bad in ["", "   ", 7]:
        try:
            self_context_tokens(Tokenizer(), bad)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a blank self-context document")
    try:
        self_context_documents(Tokenizer(), [])
    except ValueError:
        pass
    else:
        raise AssertionError("accepted an empty document list")

    # Right-padded rows keep their last ``count`` active states, re-padded.
    torch.manual_seed(11)
    states = torch.randn(3, 10, 6)
    lengths = [10, 8, 5]
    mask = torch.zeros(3, 10, dtype=torch.long)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
    selected, kept = select_last_rows(states, mask, [4, 3, 5])
    assert selected.shape == (3, 5, 6)
    assert torch.equal(selected[0, :4], states[0, 6:10])
    assert torch.equal(selected[0, 4], torch.zeros(6))
    assert torch.equal(selected[1, :3], states[1, 5:8])
    assert torch.equal(selected[2], states[2, :5])
    assert kept.tolist() == [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    leaf = states.clone().requires_grad_(True)
    select_last_rows(leaf, mask, [4, 3, 5])[0].sum().backward()
    assert leaf.grad[0, 6:10].eq(1).all() and leaf.grad[0, :6].eq(0).all()
    for counts, bad_mask in [
        ([0, 3, 5], mask),
        ([11, 3, 5], mask),
        ([4, 3], mask),
        ([4, True, 5], mask),
        ([4, 3, 5], mask.flip(1)),
    ]:
        try:
            select_last_rows(states, bad_mask, counts)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted an invalid last-row selection")
    # The question row (last) always stays whole; a layer pair is kept alike.
    single_states, single_mask = self_context_states(states, mask, [4, 3])
    (final, layer), pair_mask = self_context_states((states, 2 * states), mask, [4, 3])
    assert torch.equal(single_states, final) and torch.equal(single_mask, pair_mask)
    assert torch.equal(final[2], states[2, :5]) and torch.equal(layer, 2 * final)
    assert pair_mask[2].tolist() == [1, 1, 1, 1, 1]
    try:
        self_context_states(states, mask, [4])
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a missing kept count")


def test_pooled_context_mixes_each_documents_pooled_vector_into_its_rows():
    from encoder_lora import (
        add_pooled_context,
        document_features,
        document_features_dual,
        feature_channel,
    )
    from encoder_reference import normalize_features

    torch.manual_seed(7)
    rows = torch.randn(5, 2560) * 3
    pooled = nn.functional.normalize(torch.randn(2560), dim=-1)
    mixed = add_pooled_context(rows, pooled)
    assert mixed.shape == rows.shape
    assert torch.allclose(mixed.norm(dim=-1), torch.ones(5), atol=1e-5)
    unit = nn.functional.normalize(rows, dim=-1)
    assert torch.allclose(mixed, nn.functional.normalize(unit + pooled, dim=-1))
    # The result depends on the pooled vector and leans toward it.
    assert not torch.allclose(mixed, add_pooled_context(rows, -pooled))
    assert (mixed @ pooled > unit @ pooled).all()
    for bad_rows, bad_pooled in [
        (rows[0], pooled),
        (rows, pooled[:-1]),
        (rows[:0], pooled),
        (-pooled.unsqueeze(0), pooled),
    ]:
        try:
            add_pooled_context(bad_rows, bad_pooled)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid pooled context inputs")

    states = torch.randn(3, 12, 2560)
    lengths = [12, 7, 5]
    mask = torch.zeros(3, 12, dtype=torch.long)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
    plain, query = document_features(states, mask, "tokens")
    assert torch.equal(plain, document_features(states, mask, "tokens", False)[0])
    docs, same_query = document_features(states, mask, "tokens", pooled_context=True)
    assert docs.shape == plain.shape and torch.equal(same_query, query)
    assert torch.allclose(docs.norm(dim=-1), torch.ones(1, 19), atol=2e-2)
    assert not torch.allclose(docs, plain, atol=1e-2)
    # Each document's rows get that document's own pooled vector, through the
    # same channel as the released pooled path; the question is untouched.
    pooled_docs = feature_channel(normalize_features(states, mask))
    for begin, index, length in [(0, 0, 12), (12, 1, 7)]:
        expected = feature_channel(
            add_pooled_context(states[index, :length], pooled_docs[index])
        )
        assert torch.allclose(docs[0, begin : begin + length], expected, atol=2e-3)
    chunks, _ = document_features(states, mask, "chunks4", pooled_context=True)
    assert chunks.shape == (1, 8, 2560)
    assert torch.allclose(chunks.norm(dim=-1), torch.ones(1, 8), atol=2e-2)
    try:
        document_features(states, mask, "pooled", pooled_context=True)
    except ValueError:
        pass
    else:
        raise AssertionError("pooled features accepted pooled context")
    layer_states = 0.5 * states + 1
    dual, dual_query = document_features_dual(
        states, layer_states, mask, pooled_context=True
    )
    final_docs, _ = document_features(states, mask, "tokens", pooled_context=True)
    layer_docs, _ = document_features(layer_states, mask, "tokens", pooled_context=True)
    assert torch.equal(dual, torch.cat([final_docs, layer_docs], dim=1))
    assert torch.equal(dual_query, query)
    assert not torch.equal(dual, document_features_dual(states, layer_states, mask)[0])


class OffsetTokenizer:
    """Word-level stand-in for a fast tokenizer: a piece is a word with its
    leading space (or a lone space), reported with character offsets."""

    is_fast = True

    def __init__(self):
        self.vocabulary = {}

    def pieces(self, text):
        return [(m.start(), m.end(), m.group()) for m in re.finditer(r" ?\S+| ", text)]

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [
            self.vocabulary.setdefault(piece, len(self.vocabulary) + 1)
            for _, _, piece in self.pieces(text)
        ]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert add_special_tokens is False and return_offsets_mapping
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(start, end) for start, end, _ in self.pieces(text)],
        }


def test_oracle_entity_tags_type_sites_people_codes_and_the_missing_marker():
    from pathlib import Path

    from encoder_lora import (
        ENTITY_TYPE_CODE,
        ENTITY_TYPE_MISSING,
        ENTITY_TYPE_NONE,
        ENTITY_TYPE_PERSON,
        ENTITY_TYPE_SITE,
        document_entity_tags,
        oracle_entity_tags,
        pad_entity_tags,
        prefixed_documents,
    )

    case = {
        "question": "Who holds the brass key at Kalobe Depot?",
        "entities": ["Kalobe Depot", "Tiraso Depot", "Mujapo Manabo", "Bolema Nuvake"],
    }
    first = "Mujapo Manabo is entrusted with the brass key for Kalobe Depot, whose access code is recorded as Qx7-001234."
    second = "At Kalobe Depot, the brass key holder is Mujapo Manabo. Its access code is not recorded."
    distractor = "The brass key at Tiraso Depot belongs to Bolema Nuvake. The code at Tiraso Depot is Rv3-004321."
    tokenizers = [OffsetTokenizer()]
    model = Path(
        "/home/tim/antfly/.worktrees/plan-memorilla-native-memory/"
        ".benchmark-assets/memorilla/models/qwen3-8b"
    )
    if (model / "tokenizer.json").exists():
        from transformers import AutoTokenizer

        tokenizers.append(
            AutoTokenizer.from_pretrained(str(model), local_files_only=True)
        )
    for tokenizer in tokenizers:
        (prefixed,) = prefixed_documents([first], True)
        tags = oracle_entity_tags(tokenizer, prefixed, case)
        ids = tokenizer.encode(prefixed, add_special_tokens=False)
        assert tags.dtype == torch.long and tags.shape == (len(ids),)
        # The sentence-initial name's subwords lead the text and are all typed
        # person; site and code spans cover exactly their mid-sentence tokens.
        name = len(tokenizer.encode(" Mujapo Manabo", add_special_tokens=False))
        assert tags[:name].tolist() == [ENTITY_TYPE_PERSON] * name
        assert int((tags == ENTITY_TYPE_PERSON).sum()) == name
        site = len(tokenizer.encode(" Kalobe Depot", add_special_tokens=False))
        code = len(tokenizer.encode(" Qx7-001234", add_special_tokens=False))
        assert int((tags == ENTITY_TYPE_SITE).sum()) == site
        assert int((tags == ENTITY_TYPE_CODE).sum()) == code
        assert int((tags == ENTITY_TYPE_MISSING).sum()) == 0
        for type_id in (ENTITY_TYPE_SITE, ENTITY_TYPE_CODE):
            positions = (tags == type_id).nonzero().flatten().tolist()
            assert positions == list(range(positions[0], positions[-1] + 1))
        assert int((tags == ENTITY_TYPE_NONE).sum()) == len(ids) - name - site - code
        # The literal "not recorded" marker is its own type; a mid-sentence
        # name and the site are typed as before and nothing else is.
        tags = oracle_entity_tags(tokenizer, second, case)
        missing = len(tokenizer.encode(" not recorded", add_special_tokens=False))
        assert int((tags == ENTITY_TYPE_MISSING).sum()) == missing
        assert int((tags == ENTITY_TYPE_PERSON).sum()) == name
        assert int((tags == ENTITY_TYPE_SITE).sum()) == site
        assert int((tags == ENTITY_TYPE_CODE).sum()) == 0
        # The other site and the distractor are typed too (two site mentions).
        tags = oracle_entity_tags(tokenizer, distractor, case)
        other = len(tokenizer.encode(" Tiraso Depot", add_special_tokens=False))
        assert int((tags == ENTITY_TYPE_SITE).sum()) == 2 * other
        assert int((tags == ENTITY_TYPE_PERSON).sum()) == len(
            tokenizer.encode(" Bolema Nuvake", add_special_tokens=False)
        )
        assert int((tags == ENTITY_TYPE_CODE).sum()) == len(
            tokenizer.encode(" Rv3-004321", add_special_tokens=False)
        )
        # Under self-context the tags cover the single ids the kept states
        # stand for: the prefixed text repeats, an unprefixed initial name
        # falls back to the prefixed ids.
        assert torch.equal(
            document_entity_tags(tokenizer, prefixed, case, self_context=True),
            oracle_entity_tags(tokenizer, prefixed, case),
        )
        fallback = document_entity_tags(tokenizer, first, case, self_context=True)
        assert fallback.shape[0] == len(
            tokenizer.encode(" " + first, add_special_tokens=False)
        ) or fallback.shape[0] == len(tokenizer.encode(first, add_special_tokens=False))
        assert fallback[0] == ENTITY_TYPE_PERSON
    padded = pad_entity_tags([torch.tensor([1, 2, 3]), torch.tensor([4])])
    assert padded.tolist() == [[1, 2, 3], [4, 0, 0]] and padded.dtype == torch.long

    class SlowTokenizer(OffsetTokenizer):
        is_fast = False

    for bad in [
        lambda: oracle_entity_tags(OffsetTokenizer(), first, {"entities": ["a"]}),
        lambda: oracle_entity_tags(SlowTokenizer(), first, case),
        lambda: oracle_entity_tags(OffsetTokenizer(), "", case),
        lambda: pad_entity_tags([torch.tensor([1.0])]),
        lambda: pad_entity_tags([]),
    ]:
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid entity tagging accepted")


def test_entity_tag_embedding_is_a_zero_init_noop_and_types_only_tagged_rows():
    from encoder_lora import (
        ENTITY_TYPES,
        EntityTagEmbedding,
        document_features,
        document_features_dual,
    )

    torch.manual_seed(17)
    # The feature path checks the encoder's full width.
    width, docs, longest = 2560, 2, 5
    states = torch.randn(docs + 1, longest, width)
    layer = torch.randn(docs + 1, longest, width)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 0, 0, 0]])
    tags = torch.tensor([0, 2, 2, 1, 0, 3, 3], dtype=torch.long)
    module = EntityTagEmbedding(width)
    assert module.table.weight.shape == (len(ENTITY_TYPES), width)
    assert module.table.weight.dtype == torch.float32
    assert float(module.table.weight.abs().max()) == 0
    for mode, pooled in [
        ("tokens", False),
        ("tokens", True),
        ("tokens-question", False),
    ]:
        plain, query = document_features(states, mask, mode, pooled_context=pooled)
        tagged, tagged_query = document_features(
            states, mask, mode, pooled_context=pooled, tag_embedding=module, tags=tags
        )
        # Zero table: bit-identical features and query.
        assert torch.equal(plain, tagged) and torch.equal(query, tagged_query)
    plain_dual, _ = document_features_dual(states, layer, mask)
    tagged_dual, _ = document_features_dual(
        states, layer, mask, tag_embedding=module, tags=tags
    )
    assert torch.equal(plain_dual, tagged_dual)
    rows = torch.randn(7, width)
    assert torch.equal(module.offsets(rows, tags), torch.zeros(7, width))
    assert torch.allclose(
        module(rows, tags), torch.nn.functional.normalize(rows, dim=-1)
    )
    # Nonzero type vectors move exactly the typed rows, keep unit norm, and
    # leave untyped document rows and the question's own rows untouched.
    with torch.no_grad():
        module.table.weight[1:].normal_()
    plain, _ = document_features(states, mask, "tokens-question")
    tagged, _ = document_features(
        states, mask, "tokens-question", tag_embedding=module, tags=tags
    )
    typed = torch.cat([tags != 0, torch.zeros(2, dtype=torch.bool)])
    assert tagged.shape == plain.shape == (1, 9, width)
    assert torch.equal(plain[0, ~typed], tagged[0, ~typed])
    assert bool((plain[0, typed] != tagged[0, typed]).any(dim=-1).all())
    assert torch.allclose(tagged[0].norm(dim=-1), torch.ones(9), atol=2e-3)
    # Dual features type both layers' rows alike; the offsets scale with the
    # row norm so unit rows receive exactly the table vector.
    tagged_dual, _ = document_features_dual(
        states, layer, mask, tag_embedding=module, tags=tags
    )
    assert bool((tagged_dual[0, :7][tags != 0] != plain_dual[0, :7][tags != 0]).any())
    assert bool((tagged_dual[0, 7:][tags != 0] != plain_dual[0, 7:][tags != 0]).any())
    unit = torch.nn.functional.normalize(rows, dim=-1)
    assert torch.allclose(module.offsets(unit, tags), module.table.weight[tags])
    assert torch.allclose(
        module.offsets(3 * unit, tags), 3 * module.table.weight[tags], atol=1e-6
    )
    # The table receives gradient through the feature path.
    out, _ = document_features(states, mask, "tokens", tag_embedding=module, tags=tags)
    out.square().sum().backward()
    assert module.table.weight.grad is not None
    assert float(module.table.weight.grad[2].abs().sum()) > 0
    assert float(module.table.weight.grad[4].abs().sum()) == 0
    for bad in [
        lambda: document_features(states, mask, "tokens", tags=tags),
        lambda: document_features(states, mask, "tokens", tag_embedding=module),
        lambda: document_features(
            states, mask, "chunks4", tag_embedding=module, tags=tags
        ),
        lambda: document_features(
            states, mask, "tokens", tag_embedding=module, tags=tags[:-1]
        ),
        lambda: document_features(
            states, mask, "tokens", tag_embedding=module, tags=torch.full((7,), 5)
        ),
        lambda: module.offsets(rows, tags.float()),
        lambda: EntityTagEmbedding(width, 1),
    ]:
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid entity tag feature geometry accepted")


def test_copy_type_mask_restricts_only_the_asked_values_slots_by_entity_type():
    from encoder_lora import (
        ENTITY_TYPE_CODE,
        ENTITY_TYPE_MISSING,
        ENTITY_TYPE_PERSON,
        copy_type_mask,
        lexical_record_text,
        lexical_target_ids,
        lexical_value_slots,
        oracle_entity_tags,
        pad_entity_tags,
    )

    tokenizer = OffsetTokenizer()
    rows = {r["id"].split("-")[-1]: r for r in data_rows(3, "dev", 1)}
    name_case, code_case, missing_case = rows["owner"], rows["a"], rows["missing"]
    assert missing_case["answer"] == "UNKNOWN"
    # asked-site: `` key holder {owner} at {site}``: the two owner words sit in
    # slots 2 and 3; `` code {code} at {site}``: the code in slot 1.
    name_slots = lexical_value_slots(tokenizer, name_case, "asked-site")
    assert name_slots.tolist() == [False, False, True, True] + [False] * 12
    code_slots = lexical_value_slots(tokenizer, code_case, "asked-site")
    assert code_slots.tolist() == [False, True] + [False] * 14
    # Other records place the asked value elsewhere; UNKNOWN counts as the value.
    assert lexical_value_slots(tokenizer, name_case, "owner-code").tolist()[:3] == [
        True,
        True,
        False,
    ]
    assert lexical_value_slots(tokenizer, code_case, "asked-first").tolist()[:2] == [
        True,
        False,
    ]
    text = lexical_record_text(missing_case, "asked-site")
    assert "UNKNOWN" in text
    unknown = lexical_value_slots(tokenizer, missing_case, "asked-site")
    assert unknown.tolist() == [False, True] + [False] * 14
    assert len(lexical_target_ids(tokenizer, text)) == 16
    # Document tags over both documents; the mask restricts the value slots to
    # the asked field's type and keeps every token on the other slots.
    for case, slots, allowed in [
        (name_case, name_slots, (ENTITY_TYPE_PERSON, ENTITY_TYPE_MISSING)),
        (code_case, code_slots, (ENTITY_TYPE_CODE, ENTITY_TYPE_MISSING)),
        (missing_case, unknown, (ENTITY_TYPE_CODE, ENTITY_TYPE_MISSING)),
    ]:
        tags = torch.cat(
            [oracle_entity_tags(tokenizer, d, case) for d in case["documents"]]
        )
        field = "name" if case is name_case else "code"
        doc_mask = torch.ones(1, tags.shape[0], dtype=torch.bool)
        mask = copy_type_mask(tags[None], doc_mask, slots[None], [field])
        assert mask.shape == (1, 16, tags.shape[0]) and mask.dtype == torch.bool
        typed = torch.isin(tags, torch.tensor(allowed))
        assert bool(typed.any()) and not bool(typed.all())
        for slot in range(16):
            expected = typed if slots[slot] else torch.ones_like(typed)
            assert torch.equal(mask[0, slot], expected)
    missing_tags = torch.cat(
        [
            oracle_entity_tags(tokenizer, d, missing_case)
            for d in missing_case["documents"]
        ]
    )
    assert int((missing_tags == ENTITY_TYPE_MISSING).sum()) == 2
    # Batched: padding never copies; a value slot without a token of its type
    # is refused, as is an unknown field.
    padded = pad_entity_tags([tags, tags[:5]])
    doc_mask = torch.zeros(2, padded.shape[1], dtype=torch.bool)
    doc_mask[0], doc_mask[1, :5] = True, True
    batched = copy_type_mask(
        padded,
        doc_mask,
        torch.stack([code_slots, torch.zeros(16, dtype=torch.bool)]),
        ["code", "name"],
    )
    assert torch.equal(batched[1], doc_mask[1][None].expand(16, -1))
    untyped = ~torch.isin(
        padded[0], torch.tensor([ENTITY_TYPE_CODE, ENTITY_TYPE_MISSING])
    )
    assert not bool(batched[0, 1, untyped].any())
    assert bool(batched[0, 1, ~untyped].all())
    for bad in [
        lambda: copy_type_mask(
            torch.zeros_like(tags)[None],
            torch.ones(1, tags.shape[0], dtype=torch.bool),
            code_slots[None],
            ["code"],
        ),
        lambda: copy_type_mask(
            tags[None],
            torch.ones(1, tags.shape[0], dtype=torch.bool),
            code_slots[None],
            ["site"],
        ),
        lambda: copy_type_mask(
            tags[None], torch.ones(1, tags.shape[0]), code_slots[None], ["code"]
        ),
        lambda: lexical_value_slots(
            tokenizer,
            {
                "question": "Who holds the brass key?",
                "documents": [],
                "entities": ["a"],
            },
            "asked-site",
        ),
    ]:
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid copy type mask accepted")


def test_copy_head_slot_mask_zeroes_probability_on_tokens_outside_the_slots_type():
    from encoder_lora import CopyHead

    torch.manual_seed(23)
    width, slots, tokens = 8, 4, 6
    head = CopyHead(width)
    with torch.no_grad():
        head.gate.bias.fill_(1e4)
    memories = torch.randn(slots, width)
    table = torch.linalg.qr(torch.randn(width, width))[0][:tokens]
    slot_mask = torch.ones(slots, tokens, dtype=torch.bool)
    slot_mask[1] = torch.tensor([False, False, True, True, False, False])
    slot_mask[3] = torch.tensor([True, False, False, False, False, False])
    plain, plain_probs, _ = head(memories, table)
    output, probs, gate = head(memories, table, None, slot_mask)
    assert float(gate.min()) > 1 - 1e-6
    # Unrestricted slots are untouched; restricted ones renormalize over their
    # own tokens and copy only from them.
    assert torch.equal(output[0], plain[0]) and torch.equal(output[2], plain[2])
    assert float(probs[1, ~slot_mask[1]].abs().max()) == 0
    assert float(probs[1, slot_mask[1]].sum()) > 1 - 1e-6
    assert torch.allclose(probs[3], torch.eye(tokens)[0], atol=1e-6)
    assert torch.allclose(output[3], table[0], atol=1e-5)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(slots), atol=1e-6)
    # Batched form composes with the token mask and agrees with the single form.
    batch_mask = torch.tensor([[True] * tokens, [True] * 4 + [False] * 2])
    stacked = torch.stack([slot_mask, torch.ones(slots, tokens, dtype=torch.bool)])
    out_b, probs_b, _ = head(
        memories.expand(2, -1, -1), table.expand(2, -1, -1), batch_mask, stacked
    )
    assert torch.allclose(out_b[0], output, atol=1e-6)
    assert float(probs_b[1, :, 4:].abs().max()) == 0
    _, logits, _ = head.forward_detailed(
        memories[None], table[None], None, slot_mask[None]
    )
    assert bool(torch.isinf(logits[0, 1, ~slot_mask[1]]).all())
    empty = slot_mask.clone()
    empty[2] = False
    outside = slot_mask.clone()
    outside[2] = torch.tensor([False, False, False, False, True, True])
    for bad in [
        lambda: head(memories, table, None, empty),
        lambda: head(memories, table, batch_mask[1], outside),
        lambda: head(memories, table, None, slot_mask.float()),
        lambda: head(memories, table, None, slot_mask[:, :-1]),
    ]:
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid copy slot mask accepted")


def test_entity_tag_candidates_export_verify_and_restore_their_table(tmp_path):
    from encoder_lora_select import derive_candidate
    from encoder_lora import (
        SCHEMA,
        EntityTagEmbedding,
        candidate_artifacts,
        candidate_proof,
        digest,
        entity_tag_state,
        save_json,
    )
    from safetensors.torch import load_file

    torch.manual_seed(29)
    table = EntityTagEmbedding(8)
    with torch.no_grad():
        table.table.weight.normal_()
    assert ("entity_tags_sha256", "entity-tags.safetensors") in candidate_artifacts(
        {"entity_tags": "oracle"}
    )
    assert all(field != "entity_tags_sha256" for field, _ in candidate_artifacts({}))

    def run_with(name, entity_tags, snapshot_state):
        run = tmp_path / name
        run.mkdir()
        (run / "harness_source.py").write_text("synthetic unit-only training source")
        protocol = {
            "schema": SCHEMA,
            "memory_slots": 16,
            "steps": 1,
            "entity_tags": entity_tags,
            "script_sha256": digest(run / "harness_source.py"),
        }
        save_json(run / "protocol.json", protocol)
        save_json(run / "base-proofs.json", {"synthetic": True})
        save_json(
            run / "result.json",
            {"completed": True, "frozen_base_versions_unchanged": True},
        )
        snapshot = run / "1.pt"
        torch.save(
            {
                "protocol": protocol,
                "step": 1,
                "reader": {"x": torch.ones(1)},
                "adapters": {"y": torch.ones(1)},
                **snapshot_state,
                "result": {
                    "completed": False,
                    "history": [{"step": 1}],
                    "evaluations": [{"step": 1, "rows": []}],
                },
            },
            snapshot,
        )
        return run, snapshot

    run, snapshot = run_with("tags", "oracle", {"entity_tags": entity_tag_state(table)})
    out = tmp_path / "tags-candidate"
    candidate = derive_candidate(run, [snapshot], out)
    assert candidate["entity_tags_sha256"] == digest(out / "entity-tags.safetensors")
    assert candidate_proof(out)["entity_tags"] == "oracle"
    restored = EntityTagEmbedding(8)
    restored.load_state_dict(load_file(str(out / "entity-tags.safetensors")))
    assert torch.equal(restored.table.weight, table.table.weight)
    (out / "entity-tags.safetensors").write_bytes(b"tampered")
    try:
        candidate_proof(out)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered tag table passed the candidate proof")
    run, snapshot = run_with("missing", "oracle", {})
    try:
        derive_candidate(run, [snapshot], tmp_path / "missing-candidate")
    except ValueError:
        pass
    else:
        raise AssertionError("entity-tag candidate exported without its table")
    run, snapshot = run_with(
        "undeclared", "none", {"entity_tags": entity_tag_state(table)}
    )
    try:
        derive_candidate(run, [snapshot], tmp_path / "undeclared-candidate")
    except ValueError:
        pass
    else:
        raise AssertionError("undeclared tag table exported")
    run, snapshot = run_with("plain", "none", {})
    plain = tmp_path / "plain-candidate"
    candidate = derive_candidate(run, [snapshot], plain)
    assert "entity_tags_sha256" not in candidate
    assert not (plain / "entity-tags.safetensors").exists()
    assert candidate_proof(plain)["entity_tags"] == "none"
    save_json(plain / "candidate.json", dict(candidate, entity_tags_sha256="0" * 64))
    try:
        candidate_proof(plain)
    except ValueError:
        pass
    else:
        raise AssertionError("stray tag table hash passed the candidate proof")
    run, snapshot = run_with("unknown", "learned", {})
    try:
        derive_candidate(run, [snapshot], tmp_path / "unknown-candidate")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown entity tag mode exported")


def test_encoder_special_rows_are_untyped_and_kept_with_the_self_context_copy():
    from encoder_lora import (
        encoder_special_suffix,
        feature_row_tags,
        self_context_states,
    )

    class Ids:
        def __init__(self, ids):
            self.input_ids = ids

    class Tokenizer:
        """Embedding-style tokenizer: ``__call__`` appends ``suffix`` special
        ids after a text's own ids, ``encode`` without specials does not."""

        def __init__(self, suffix=1, prefix=0):
            self.suffix, self.prefix = suffix, prefix

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [hash("w:" + w) for w in text.split()]

        def __call__(self, text):
            ids = self.encode(text, add_special_tokens=False)
            return Ids([-1] * self.prefix + ids + [-2] * self.suffix)

    assert encoder_special_suffix(Tokenizer(1)) == 1
    assert encoder_special_suffix(Tokenizer(0)) == 0
    try:
        encoder_special_suffix(Tokenizer(1, prefix=1))
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a tokenizer that prepends special tokens")

    tags = torch.tensor([0, 2, 2, 0, 1], dtype=torch.long)
    assert feature_row_tags(tags, 0) is tags
    assert feature_row_tags(tags, 1).tolist() == [0, 2, 2, 0, 1, 0]
    for bad in [(tags.float(), 1), (tags, -1), (tags, True), (tags[None], 1)]:
        try:
            feature_row_tags(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid feature row tags")

    # Two documents of 3 and 2 ids doubled (plus one appended special row
    # each) and a 4-row question: the keep is the whole second copy and the
    # special row, not the count-only window that dropped the copy's first id.
    torch.manual_seed(5)
    states = torch.randn(3, 8, 4)
    mask = torch.zeros(3, 8, dtype=torch.long)
    for row, length in enumerate([7, 5, 4]):
        mask[row, :length] = 1
    kept, kept_mask = self_context_states(states, mask, [3, 2], 1)
    assert kept.shape == (3, 4, 4)
    assert torch.equal(kept[0], states[0, 3:7])
    assert torch.equal(kept[1, :3], states[1, 2:5])
    assert torch.equal(kept[2], states[2, :4])
    assert kept_mask.tolist() == [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1]]
    old, _ = self_context_states(states, mask, [3, 2])
    assert torch.equal(old[0, :3], states[0, 4:7])
    try:
        self_context_states(states, mask, [3, 2], -1)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a negative special-row suffix")


def test_entity_tag_norm_cap_scales_only_rows_above_the_cap():
    from encoder_lora import EntityTagEmbedding, cap_entity_tag_rows

    table = EntityTagEmbedding(width=8, types=5)
    with torch.no_grad():
        table.table.weight.copy_(
            torch.stack([torch.full((8,), float(v)) for v in (0.0, 0.1, 0.5, 1.0, 2.0)])
        )
    before = table.table.weight.clone()
    cap_entity_tag_rows(table, 0.5)
    norms = table.table.weight.norm(dim=-1)
    assert torch.equal(table.table.weight[:2], before[:2])
    assert torch.allclose(norms[2:], torch.full((3,), 0.5), atol=1e-6)
    # Direction is preserved for the scaled rows.
    for row in range(2, 5):
        assert torch.allclose(
            torch.nn.functional.normalize(table.table.weight[row], dim=0),
            torch.nn.functional.normalize(before[row], dim=0),
        )
    for bad in (0, -1.0, "x"):
        try:
            cap_entity_tag_rows(table, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted an invalid norm cap")


def _pinned_reader(embedding_dim=16, output_dim=24, slots=4, heads=2):
    """A small randomly initialized upstream MemoryModule, or None when the
    pinned source asset is absent."""
    import importlib.util
    import sys
    from pathlib import Path
    from types import ModuleType

    source = Path(
        "/home/tim/antfly/.worktrees/plan-memorilla-native-memory/"
        ".benchmark-assets/memorilla/upstream/memorilla/memory.py"
    )
    if not source.exists():
        return None
    utility = ModuleType("memorilla.utils")
    for name in ("CHECKPOINT_FILE", "CONFIG_FILE", "checkpoint_file", "load_memory_state"):
        setattr(utility, name, None)
    previous = sys.modules.get("memorilla.utils")
    sys.modules["memorilla.utils"] = utility
    try:
        spec = importlib.util.spec_from_file_location("_test_pinned_memory", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("memorilla.utils", None)
        else:
            sys.modules["memorilla.utils"] = previous
    torch.manual_seed(31)
    return module.MemoryModule(
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        num_memories=slots,
        num_heads=heads,
        dropout=0.0,
    ).eval()


def test_reader_pass_and_looped_read_reproduce_the_plain_reader_at_zero_feedback():
    from encoder_lora import SlotRefiner, looped_read, reader_pass

    reader = _pinned_reader()
    if reader is None:
        return
    torch.manual_seed(32)
    docs = torch.randn(3, 7, 16)
    mask = torch.zeros(3, 7, dtype=torch.bool)
    mask[1, 5:] = True
    question = torch.randn(3, 16)
    expected = reader(docs, mask, question)
    memories, final = reader_pass(
        reader, docs, mask, question, reader.memory_queries.expand(3, -1, -1)
    )
    assert torch.equal(memories, expected) and final.shape == (3, 4, 16)
    assert torch.equal(looped_read(reader, None, docs, mask, question, 1), expected)
    refiner = SlotRefiner(16, 2)
    assert torch.equal(looped_read(reader, refiner, docs, mask, question, 3), expected)
    with torch.no_grad():
        refiner.proj.weight.normal_()
    looped = looped_read(reader, refiner, docs, mask, question, 2)
    assert looped.shape == expected.shape and not torch.equal(looped, expected)
    looped.sum().backward()
    assert refiner.proj.weight.grad is not None and refiner.attn.in_proj_weight.grad is not None
    assert reader.memory_queries.grad is not None
    # The causal mask keeps slot i independent of slots after it.
    with torch.no_grad():
        refiner.proj.weight.normal_()
    base = torch.randn(1, 4, 16)
    altered = base.clone()
    # A non-constant change, since the refiner's LayerNorm ignores a shift.
    altered[0, 3] += torch.randn(16) * 5.0
    assert torch.allclose(refiner(base)[0, :3], refiner(altered)[0, :3], atol=1e-5)
    assert not torch.allclose(refiner(base)[0, 3], refiner(altered)[0, 3], atol=1e-3)
    for bad in [
        lambda: looped_read(reader, None, docs, mask, question, 2),
        lambda: looped_read(reader, refiner, docs, mask, question, 1),
        lambda: looped_read(reader, refiner, docs, mask, question, 5),
        lambda: reader_pass(reader, docs, mask, question, torch.randn(3, 5, 16)),
        lambda: refiner(torch.randn(3, 4, 8)),
        lambda: SlotRefiner(15, 2),
    ]:
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid looped-read input")


def test_slot_refiner_candidates_export_verify_and_restore(tmp_path):
    from encoder_lora_select import derive_candidate
    from encoder_lora import (
        SCHEMA,
        SlotRefiner,
        candidate_artifacts,
        candidate_proof,
        digest,
        reader_loops_enabled,
        save_json,
        slot_refiner_state,
    )
    from safetensors.torch import load_file

    torch.manual_seed(33)
    refiner = SlotRefiner(8, 2)
    with torch.no_grad():
        refiner.proj.weight.normal_()
    assert reader_loops_enabled({"reader_loops": 2}) and not reader_loops_enabled({})
    for bad in ({"reader_loops": 0}, {"reader_loops": 5}, {"reader_loops": True}):
        try:
            reader_loops_enabled(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid reader loops")
    assert ("slot_refiner_sha256", "slot-refiner.safetensors") in candidate_artifacts(
        {"reader_loops": 2}
    )
    assert all(field != "slot_refiner_sha256" for field, _ in candidate_artifacts({}))

    def run_with(name, loops, snapshot_state):
        run = tmp_path / name
        run.mkdir()
        (run / "harness_source.py").write_text("synthetic unit-only training source")
        protocol = {
            "schema": SCHEMA,
            "memory_slots": 16,
            "steps": 1,
            "reader_loops": loops,
            "script_sha256": digest(run / "harness_source.py"),
        }
        save_json(run / "protocol.json", protocol)
        save_json(run / "base-proofs.json", {"synthetic": True})
        save_json(
            run / "result.json",
            {"completed": True, "frozen_base_versions_unchanged": True},
        )
        snapshot = run / "1.pt"
        torch.save(
            {
                "protocol": protocol,
                "step": 1,
                "reader": {"x": torch.ones(1)},
                "adapters": {"y": torch.ones(1)},
                **snapshot_state,
                "result": {
                    "completed": False,
                    "history": [{"step": 1}],
                    "evaluations": [{"step": 1, "rows": []}],
                },
            },
            snapshot,
        )
        return run, snapshot

    run, snapshot = run_with("looped", 2, {"slot_refiner": slot_refiner_state(refiner)})
    out = tmp_path / "looped-candidate"
    candidate = derive_candidate(run, [snapshot], out)
    assert candidate["slot_refiner_sha256"] == digest(out / "slot-refiner.safetensors")
    assert candidate_proof(out)["reader_loops"] == 2
    restored = SlotRefiner(8, 2)
    restored.load_state_dict(load_file(str(out / "slot-refiner.safetensors")))
    assert torch.equal(restored.proj.weight, refiner.proj.weight)
    (out / "slot-refiner.safetensors").write_bytes(b"tampered")
    try:
        candidate_proof(out)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered refiner passed the candidate proof")
    run, snapshot = run_with("missing", 2, {})
    try:
        derive_candidate(run, [snapshot], tmp_path / "missing-candidate")
    except ValueError:
        pass
    else:
        raise AssertionError("looped candidate exported without its refiner")
    run, snapshot = run_with("undeclared", 1, {"slot_refiner": slot_refiner_state(refiner)})
    try:
        derive_candidate(run, [snapshot], tmp_path / "undeclared-candidate")
    except ValueError:
        pass
    else:
        raise AssertionError("undeclared refiner exported")
    run, snapshot = run_with("plain", 1, {})
    plain = tmp_path / "plain-candidate"
    candidate = derive_candidate(run, [snapshot], plain)
    assert "slot_refiner_sha256" not in candidate
    assert not (plain / "slot-refiner.safetensors").exists()
    save_json(plain / "candidate.json", dict(candidate, slot_refiner_sha256="0" * 64))
    try:
        candidate_proof(plain)
    except ValueError:
        pass
    else:
        raise AssertionError("undeclared refiner hash passed the candidate proof")


def test_unique_copy_positions_repoints_only_the_slots_that_collide():
    from encoder_lora import CopyHead, copy_with_unique_positions

    torch.manual_seed(3)
    width, positions = 16, 5
    tokens = torch.nn.functional.normalize(torch.randn(positions, width), dim=-1)
    head = CopyHead(width, 0.05)
    with torch.no_grad():
        head.gate.bias.fill_(6.0)  # every slot copies (gate ~ 0.998)
    # Slots 0 and 1 both point at position 2 (slot 0 more sharply), slots 2
    # and 3 at their own positions 4 and 1.
    memories = torch.stack(
        [tokens[2] * 1.0, tokens[2] * 0.6 + tokens[0] * 0.4, tokens[4], tokens[1]]
    )
    out, gate, reassigned = copy_with_unique_positions(head, memories, tokens)
    assert out.shape == memories.shape and gate.shape == (4,)
    assert reassigned == 1
    plain, probs, _ = head(memories, tokens)
    # Untouched slots keep the plain soft copy exactly.
    assert torch.equal(out[0], plain[0]) and torch.equal(out[2], plain[2])
    assert torch.equal(out[3], plain[3])
    # The colliding slot moved to its best unclaimed position (0), hard copy.
    assert torch.allclose(
        out[1], gate[1] * tokens[0] + (1 - gate[1]) * memories[1], atol=1e-6
    )
    assert int(probs[1].argmax()) == 2
    # Slots below the gate threshold never claim positions.
    out2, _, reassigned2 = copy_with_unique_positions(
        head, memories, tokens, min_gate=1.0
    )
    assert reassigned2 == 0 and torch.equal(out2, plain)
    for bad in [(memories[None], tokens), (memories, tokens[None])]:
        try:
            copy_with_unique_positions(head, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a batched input")


def test_contiguous_runs_read_one_source_span_per_copying_run():
    from encoder_lora import CopyHead, copy_with_contiguous_runs

    torch.manual_seed(4)
    width, positions = 16, 8
    tokens = torch.nn.functional.normalize(torch.randn(positions, width), dim=-1)
    head = CopyHead(width, 0.05)
    with torch.no_grad():
        head.gate.bias.fill_(6.0)
    # Run A = slots 0..2 should read positions 2,3,4 but slot 1 points at the
    # token of position 6 (a swapped digit); slot 3 generates (gate low via a
    # memory the gate cannot see is irrelevant here, so force it by threshold
    # order instead: use a second run B = slots 4..5 reading positions 0,1).
    memories = torch.stack(
        [tokens[2], tokens[6], tokens[4], torch.zeros(width), tokens[0], tokens[1]]
    )
    with torch.no_grad():
        # Make slot 3 a non-copying slot: its own gate logit is the bias plus
        # gate.weight @ memory; a zero memory keeps the bias, so lower it by
        # giving the gate a weight that reacts to a marker dimension instead.
        head.gate.weight.zero_()
    gate_override = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    # Emulate the gate by thresholding on min_gate with a patched forward.
    original = head.forward_detailed

    def patched(memories_, tokens_, token_mask=None, slot_mask=None):
        out, logits, gl = original(memories_, tokens_, token_mask, slot_mask)
        gl = torch.where(gate_override[None] > 0, torch.full_like(gl, 6.0), torch.full_like(gl, -6.0))
        return out, logits, gl

    head.forward_detailed = patched
    out, gate, moved = copy_with_contiguous_runs(head, memories, tokens)
    assert out.shape == memories.shape and moved == 1
    g = gate[1]
    assert torch.allclose(out[1], g * tokens[3] + (1 - g) * memories[1], atol=1e-6)
    # Other run slots hard-copy their own (already diagonal) positions.
    assert torch.allclose(out[0], gate[0] * tokens[2] + (1 - gate[0]) * memories[0], atol=1e-6)
    assert torch.allclose(out[4], gate[4] * tokens[0] + (1 - gate[4]) * memories[4], atol=1e-6)
    # The generating slot keeps the plain soft output.
    plain, _, _ = original(memories[None], tokens[None])
    assert torch.equal(out[3], plain[0][3])
    for bad in [(memories[None], tokens), (memories, tokens[None])]:
        try:
            copy_with_contiguous_runs(head, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a batched input")


def test_extended_runs_absorb_a_low_gate_slot_pointing_at_the_next_token():
    from encoder_lora import CopyHead, copy_with_contiguous_runs

    torch.manual_seed(6)
    width, positions = 16, 8
    tokens = torch.nn.functional.normalize(torch.randn(positions, width), dim=-1)
    head = CopyHead(width, 0.05)
    # Run = slots 1..2 reading positions 3,4; slot 3 points at position 5 (the
    # value's last subword) but its gate is low; slot 0 points at position 2
    # with a low gate (the first subword eaten by the label slot); slot 4
    # points elsewhere with a low gate and must stay untouched.
    memories = torch.stack([tokens[2], tokens[3], tokens[4], tokens[5], tokens[0]])
    gates = torch.tensor([0.1, 0.9, 0.9, 0.2, 0.1])
    original = head.forward_detailed

    def patched(m, t, token_mask=None, slot_mask=None):
        out, logits, gl = original(m, t, token_mask, slot_mask)
        return out, logits, torch.logit(gates)[None]

    head.forward_detailed = patched
    plain, _, moved0 = copy_with_contiguous_runs(head, memories, tokens)
    assert moved0 == 0
    assert torch.allclose(plain[3], original(memories[None], tokens[None])[0][0][3])
    out, gate, moved = copy_with_contiguous_runs(head, memories, tokens, extend=True)
    assert moved == 0  # every slot already pointed at its diagonal position
    # Absorbed boundary slots are forced to copy their tokens exactly.
    assert torch.allclose(out[0], tokens[2], atol=1e-6)
    assert torch.allclose(out[3], tokens[5], atol=1e-6)
    # Slot 4 (not contiguous) keeps its plain soft output.
    assert torch.equal(out[4], plain[4])
    assert torch.allclose(out[1], gates[1] * tokens[3] + (1 - gates[1]) * memories[1], atol=1e-6)
