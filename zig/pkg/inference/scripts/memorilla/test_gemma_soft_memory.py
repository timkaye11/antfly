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
from pathlib import Path

import gemma_soft_memory
import pytest
import torch
from encoder_lora import (
    CopyHead,
    MemoryTokenSnapper,
    atomic_checkpoint,
    feature_channel,
    pointer_losses,
)
from gemma_memory import pooled_query
from gemma_soft_memory import (
    CHUNK_TOKENS,
    HELPFUL_PROMPT,
    IGNORE_INDEX,
    LEGACY_OPTIONS,
    PLACEHOLDER_ID,
    REPLAY_CAP,
    STAGE2_CONFIGS,
    STREAM_TYPE_FILE,
    UPSTREAM_METRIC_SOURCES,
    EvalMetrics,
    FeatureCache,
    ParquetSplit,
    SoftCaseEncoder,
    StreamTypeEmbedding,
    answer_cross_entropy,
    answer_span,
    answer_variants,
    build_decoder_example,
    build_parser,
    build_protocol,
    cap_chunks,
    chunk_cosine,
    chunk_documents,
    collate_decoder,
    copy_targets,
    digest,
    documents_digest,
    eval_split_name,
    exact_match,
    fit_documents,
    group_shuffled_stream,
    import_upstream_metrics,
    inject_memory,
    keep_top,
    load_trained,
    lr_scale,
    memory_input_summary,
    memory_messages,
    memory_rows_infer,
    memory_rows_train,
    mixture_order,
    noise_rows,
    option_protocol,
    options_active,
    pack_sentences,
    parse_datasets,
    pooled_rows,
    question_type,
    rank_order,
    recipe_parts,
    render_prompt,
    resolve_options,
    resume_protocol,
    rouge_fallback,
    safe_name,
    schedule_steps,
    select_documents,
    set_reader_dropout,
    slot_layout,
    split_complete,
    split_files,
    split_sentences,
    subsample,
    summarize_by_type,
    token_f1,
    whitespace_token_f1,
    write_candidate,
)
from torch import nn

REPO = Path(__file__).resolve().parents[5]
STAGE1_PROTOCOL = REPO / ".benchmark-results/memorilla/gemma-soft/stage1/protocol.json"

TURN_OPEN, TURN_CLOSE, NEWLINE = 105, 106, 107
EOS_IDS = (1, TURN_CLOSE)


class FakeTokenizer:
    """Word pieces (`` ?word`` / punctuation) with Gemma-like specials: BOS 2,
    ``<|turn>`` 105, ``<turn|>`` 106, newline 107, the image soft token
    258880 and a chat template with a system turn."""

    bos_token_id, eos_token_id, pad_token_id = 2, 1, 0

    def __init__(self):
        self.pieces = {
            "<pad>": 0,
            "<eos>": 1,
            "<bos>": 2,
            " ": 3,
            "<|turn>": TURN_OPEN,
            "<turn|>": TURN_CLOSE,
            "\n": NEWLINE,
            "<|image|>": PLACEHOLDER_ID,
        }
        self.next_id = 1000

    def _split(self, text):
        return re.findall(
            r"<\|image\|>|<\|turn>|<turn\|>|<bos>| ?[A-Za-z0-9]+|\S|\s", text
        )

    def encode(self, text, add_special_tokens=True):
        ids = []
        for piece in self._split(text):
            if piece not in self.pieces:
                self.pieces[piece] = self.next_id
                self.next_id += 1
            ids.append(self.pieces[piece])
        return [2, *ids, 1] if add_special_tokens else ids

    def decode(self, ids, skip_special_tokens=False):
        inverse = {v: k for k, v in self.pieces.items()}
        return "".join(
            inverse[int(i)]
            for i in ids
            if not (skip_special_tokens and int(i) in (0, 1, 2))
        )

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        assert not tokenize and messages[0]["role"] == "system"
        text = f"<bos><|turn>system\n{messages[0]['content'].strip()}<turn|>\n"
        for message in messages[1:]:
            role = "model" if message["role"] == "assistant" else message["role"]
            text += f"<|turn>{role}\n{message['content'].strip()}<turn|>\n"
        return text + ("<|turn>model\n" if add_generation_prompt else "")


# ----------------------------------------------------------------------------
# data


def write_split(root, config, split, rows_per_group, shards=1):
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = root / "data" / config
    directory.mkdir(parents=True, exist_ok=True)
    paths, start = [], 0
    for shard in range(shards):
        tables = []
        for size in rows_per_group:
            rows = range(start, start + size)
            tables.append(
                pa.table(
                    {
                        "collection_id": pa.array([i for i in rows], pa.int64()),
                        "question": [f"q{i}" for i in rows],
                        "answer": [f"a{i}" for i in rows],
                        "documents": [[f"d{i}", f"e{i}"] for i in rows],
                    }
                )
            )
            start += size
        path = directory / f"{split}-{shard:05d}-of-{shards:05d}.parquet"
        writer = pq.ParquetWriter(path, tables[0].schema)
        for table in tables:
            writer.write_table(table)
        writer.close()
        paths.append(path)
    return paths


def test_parquet_split_reads_rows_lazily_across_row_groups_and_shards(tmp_path):
    write_split(tmp_path, "squad_v2", "validation", [3, 2], shards=2)
    files = split_files(tmp_path, "squad_v2", "validation")
    assert len(files) == 2 and split_complete(files)
    assert not split_complete(files[:1])
    split = ParquetSplit(tmp_path, "squad_v2", "validation")
    assert len(split) == 10 and split.group_sizes == [3, 2, 3, 2]
    row = split.row(4)
    assert (row["question"], row["answer"], row["documents"]) == (
        "q4",
        "a4",
        ["d4", "e4"],
    )
    assert (row["config"], row["split"], row["row"]) == ("squad_v2", "validation", 4)
    assert row["collection_id"] == 4  # optional column, read when present
    assert len(split.cache) == 1  # one row group decoded at a time
    assert split.row(9)["answer"] == "a9" and len(split.cache) == 1
    with pytest.raises(IndexError):
        split.row(10)
    with pytest.raises(FileNotFoundError):
        ParquetSplit(tmp_path, "squad_v2", "train")


def test_group_shuffled_stream_walks_one_row_group_at_a_time_and_caps():
    generator = torch.Generator().manual_seed(1)
    stream = group_shuffled_stream([3, 4, 2], None, generator).tolist()
    assert sorted(stream) == list(range(9))
    groups = [0 if i < 3 else 1 if i < 7 else 2 for i in stream]
    # Each group's rows are consecutive in the stream.
    assert [g for i, g in enumerate(groups) if i == 0 or groups[i - 1] != g] == sorted(
        set(groups), key=groups.index
    )
    capped = group_shuffled_stream([3, 4, 2], 5, torch.Generator().manual_seed(1))
    assert capped.tolist() == stream[:5]


def test_mixture_order_is_deterministic_covers_every_example_and_interleaves():
    parts = [([3, 4], None), ([5], 4)]
    labels, rows = mixture_order(parts, 7)
    again = mixture_order(parts, 7)
    assert torch.equal(labels, again[0]) and torch.equal(rows, again[1])
    assert labels.numel() == 7 + 4
    assert sorted(rows[labels == 0].tolist()) == list(range(7))
    assert len(set(rows[labels == 1].tolist())) == 4
    assert not torch.equal(labels, mixture_order(parts, 8)[0])
    with pytest.raises(ValueError):
        mixture_order([([0], None)], 1)


def test_parse_datasets_reads_caps():
    assert parse_datasets("enwiki, squad_v2:2000") == [
        ("enwiki", None),
        ("squad_v2", 2000),
    ]
    with pytest.raises(ValueError):
        parse_datasets("enwiki,enwiki")
    with pytest.raises(ValueError):
        parse_datasets("drop:0")


def test_subsample_is_sorted_seeded_and_bounded():
    chosen = subsample(100, 10, 3)
    assert chosen == sorted(chosen) and len(set(chosen)) == 10
    assert subsample(100, 10, 3) == chosen and subsample(5, 10, 3) == list(range(5))


# ----------------------------------------------------------------------------
# documents


def test_fit_documents_caps_two_layer_rows_and_truncates_the_last_kept():
    docs = [list(range(10)), list(range(20)), list(range(30))]
    # Budget 2 * (tokens + EOS) <= max_rows: 2*(11 + 21) = 64 fits, the third
    # document gets 2 remaining tokens (2*(32 + 3) = 70).
    kept = fit_documents(docs, 70)
    assert [len(d) for d in kept] == [10, 20, 2]
    assert fit_documents(docs, 4096) == docs
    assert [len(d) for d in fit_documents(docs, 10)] == [4]
    with pytest.raises(ValueError):
        fit_documents(docs, 2)
    with pytest.raises(ValueError):
        fit_documents([[]], 100)


def test_select_documents_keeps_top_cosine_in_original_order():
    question = torch.tensor([1.0, 0.0])
    vectors = torch.tensor(
        [[0.0, 1.0], [1.0, 0.1], [0.5, 0.5], [1.0, 0.0], [-1.0, 0.0]]
    )
    assert select_documents(vectors, question, 2) == [1, 3]
    assert select_documents(vectors, question, 3) == [1, 2, 3]
    assert select_documents(vectors, question, 9) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        select_documents(vectors, torch.zeros(3), 2)


# ----------------------------------------------------------------------------
# prompt, labels, copy targets


def test_prompt_puts_memory_placeholders_in_the_system_turn():
    tokenizer = FakeTokenizer()
    placeholder = tokenizer.decode([PLACEHOLDER_ID])
    messages = memory_messages(" Who wrote it? ", 3, placeholder, "Twain ")
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert (
        messages[0]["content"]
        == HELPFUL_PROMPT + "\nMemory" + placeholder * 3 + "Memory"
    )
    assert (
        messages[1]["content"] == "Who wrote it?" and messages[2]["content"] == "Twain"
    )
    prompt = render_prompt(tokenizer, "Who wrote it?", 3, placeholder)
    assert prompt.endswith("<|turn>user\nWho wrote it?<turn|>\n<|turn>model\n")
    assert render_prompt(tokenizer, "Who wrote it?", 3, placeholder, "Twain").endswith(
        "<|turn>model\nTwain<turn|>\n"
    )
    with pytest.raises(ValueError):
        memory_messages("q", 0, placeholder)


def test_decoder_example_labels_answer_and_closing_turn_only():
    tokenizer = FakeTokenizer()
    example = build_decoder_example(
        tokenizer, "Who wrote it?", "Mark Twain", 4, PLACEHOLDER_ID, EOS_IDS, 256, 1024
    )
    ids, labels, positions = (
        example["input_ids"],
        example["labels"],
        example["memory_positions"],
    )
    assert len(positions) == 4 and all(ids[p] == PLACEHOLDER_ID for p in positions)
    assert positions == list(range(positions[0], positions[0] + 4))
    prompt_len = labels.index(next(l for l in labels if l != IGNORE_INDEX))
    assert all(l == IGNORE_INDEX for l in labels[:prompt_len])
    assert labels[prompt_len:] == ids[prompt_len:]
    assert (
        tokenizer.decode(ids[prompt_len:]) == "Mark Twain<turn|>"
    )  # trailing newline dropped
    assert ids[-1] == TURN_CLOSE and labels[-1] == TURN_CLOSE
    # Cutting the answer region drops the closing turn.
    cut = build_decoder_example(
        tokenizer, "Who wrote it?", "Mark Twain", 4, PLACEHOLDER_ID, EOS_IDS, 1, 1024
    )
    assert tokenizer.decode(cut["input_ids"][prompt_len:]) == "Mark"
    assert (
        build_decoder_example(
            tokenizer, "Who wrote it?", "x", 4, PLACEHOLDER_ID, EOS_IDS, 256, 5
        )
        is None
    )
    with pytest.raises(ValueError):
        build_decoder_example(
            tokenizer, "Who <|image|>?", "x", 4, PLACEHOLDER_ID, EOS_IDS, 256, 1024
        )


def test_collate_right_pads_and_marks_memory_positions():
    examples = [
        {
            "input_ids": [5, PLACEHOLDER_ID, PLACEHOLDER_ID, 7, 8],
            "labels": [-100, -100, -100, 7, 8],
            "memory_positions": [1, 2],
        },
        {
            "input_ids": [PLACEHOLDER_ID, PLACEHOLDER_ID, 9],
            "labels": [-100, -100, 9],
            "memory_positions": [0, 1],
        },
    ]
    ids, mask, labels, memory = collate_decoder(examples, 0)
    assert ids.shape == (2, 5) and ids[1].tolist() == [
        PLACEHOLDER_ID,
        PLACEHOLDER_ID,
        9,
        0,
        0,
    ]
    assert mask.tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
    assert labels[1].tolist() == [-100, -100, 9, -100, -100]
    assert memory.tolist() == [
        [False, True, True, False, False],
        [True, True, False, False, False],
    ]
    with pytest.raises(ValueError):
        collate_decoder([], 0)


def test_slot_layout_soft_then_copy_slots():
    layout = slot_layout(24, 8)
    assert layout["soft"] == list(range(16)) and layout["copy"] == list(range(16, 24))
    assert slot_layout(16, 0)["copy"] == []
    with pytest.raises(ValueError):
        slot_layout(24, 24)
    with pytest.raises(ValueError):
        slot_layout(20, 4)


def test_copy_targets_left_align_the_verbatim_answer_span():
    tokenizer = FakeTokenizer()
    docs = [
        tokenizer.encode(
            " The Normans gave their name to Normandy.", add_special_tokens=False
        ),
        tokenizer.encode(" Mark Twain wrote it in 1884.", add_special_tokens=False),
    ]
    variants = answer_variants(tokenizer, " Mark Twain ")
    assert variants[0] == tokenizer.encode(" Mark Twain", add_special_tokens=False)
    assert variants[1] == tokenizer.encode("Mark Twain", add_special_tokens=False)
    span = answer_span(docs, variants, 8)
    assert span == (1, 0, 2)
    # Sentence-initial answers match through the bare variant.
    assert answer_span(docs, answer_variants(tokenizer, "The Normans"), 8) == (0, 0, 2)
    assert answer_span(docs, answer_variants(tokenizer, "Shakespeare"), 8) is None
    assert (
        answer_span(docs, answer_variants(tokenizer, "Mark Twain wrote it in 1884"), 4)
        is None
    )
    assert answer_variants(tokenizer, "  ") == []
    total = sum(len(d) for d in docs) + 3  # padded sources
    match, copyable = copy_targets([docs, docs], [span, None], 8, total)
    assert match.shape == (2, 8, total) and copyable.shape == (2, 8)
    assert copyable[0].tolist() == [True, True] + [False] * 6
    assert match[0, 0].nonzero().flatten().tolist() == [len(docs[0])]
    assert match[0, 1].nonzero().flatten().tolist() == [len(docs[0]) + 1]
    assert not copyable[1].any() and not match[1].any()
    # The targets are valid pointer-loss inputs.
    logits = torch.zeros(2, 8, total)
    gate = torch.zeros(2, 8)
    doc_mask = torch.ones(2, total, dtype=torch.bool)
    pointer, gate_loss = pointer_losses(logits, gate, match, copyable, doc_mask)
    assert math.isclose(float(pointer), math.log(total), rel_tol=1e-5)
    assert math.isclose(float(gate_loss), math.log(2), rel_tol=1e-5)
    with pytest.raises(ValueError):
        copy_targets([docs], [(1, 0, 9)], 8, total)


# ----------------------------------------------------------------------------
# decoder injection and loss


class FakeText(nn.Module):
    """Scaled embedding plus an id-dependent per-layer identity table."""

    def __init__(self, vocab=300000, width=8, layers=3, per_layer=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, width)
        self.ple = nn.Embedding(vocab, layers * per_layer)
        self.layers, self.per_layer = layers, per_layer
        with torch.no_grad():
            self.embed_tokens.weight.copy_(torch.randn(vocab, width) * 10)
            self.ple.weight.copy_(torch.randn(vocab, layers * per_layer))

    def get_per_layer_inputs(self, input_ids):
        return self.ple(input_ids).reshape(
            *input_ids.shape, self.layers, self.per_layer
        )


def test_inject_memory_scatters_rows_in_order_and_gives_placeholders_the_pad_identity():
    torch.manual_seed(0)
    text = FakeText().requires_grad_(False)
    ids = torch.tensor(
        [[5, PLACEHOLDER_ID, PLACEHOLDER_ID, 7], [PLACEHOLDER_ID, PLACEHOLDER_ID, 9, 0]]
    )
    memory_mask = ids == PLACEHOLDER_ID
    memories = (
        torch.arange(2 * 2 * 8, dtype=torch.float32)
        .reshape(2, 2, 8)
        .requires_grad_(True)
    )
    embeds, per_layer = inject_memory(text, ids, memory_mask, memories, 0)
    assert torch.equal(embeds[0, 1], memories[0, 0]) and torch.equal(
        embeds[0, 2], memories[0, 1]
    )
    assert torch.equal(embeds[1, 0], memories[1, 0]) and torch.equal(
        embeds[1, 1], memories[1, 1]
    )
    assert torch.equal(embeds[0, 0], text.embed_tokens.weight[5])
    assert torch.equal(embeds[1, 2], text.embed_tokens.weight[9])
    pad_identity = text.get_per_layer_inputs(torch.tensor([0]))[0]
    assert per_layer.shape == (2, 4, 3, 2)
    assert torch.equal(per_layer[0, 1], pad_identity) and torch.equal(
        per_layer[1, 0], pad_identity
    )
    assert torch.equal(per_layer[0, 3], text.get_per_layer_inputs(torch.tensor([7]))[0])
    embeds.sum().backward()
    assert torch.equal(memories.grad, torch.ones_like(memories))
    with pytest.raises(ValueError):
        inject_memory(text, ids, memory_mask, memories[:, :1], 0)


def test_answer_cross_entropy_scores_only_labeled_next_tokens():
    torch.manual_seed(0)
    head = nn.Linear(4, 10, bias=False)
    hidden = torch.randn(2, 5, 4)
    labels = torch.tensor([[-100, -100, 3, 4, -100], [-100, 6, -100, -100, -100]])
    loss, count = answer_cross_entropy(hidden, labels, head, None)
    assert count == 3
    logits = head(hidden)
    expected = torch.stack(
        [
            nn.functional.cross_entropy(logits[0, 1:2], torch.tensor([3])),
            nn.functional.cross_entropy(logits[0, 2:3], torch.tensor([4])),
            nn.functional.cross_entropy(logits[1, 0:1], torch.tensor([6])),
        ]
    ).mean()
    assert torch.allclose(loss, expected)
    capped, _ = answer_cross_entropy(hidden, labels, head, 30.0)
    assert torch.isfinite(capped) and not torch.allclose(capped, loss)
    with pytest.raises(ValueError):
        answer_cross_entropy(hidden, torch.full((2, 5), -100), head, None)


def test_memory_rows_train_and_infer_split_soft_and_copy_slots():
    torch.manual_seed(1)
    width, slots, copy_slots, tokens = 16, 6, 2, 5
    table = nn.Embedding(40, width).requires_grad_(False)
    with torch.no_grad():
        table.weight.copy_(nn.functional.normalize(torch.randn(40, width), dim=-1) * 3)
    head = CopyHead(width=width)
    memories = torch.randn(2, slots, width)
    sources = table.weight[torch.arange(tokens)].unsqueeze(0).expand(2, -1, -1).clone()
    mask = torch.ones(2, tokens, dtype=torch.bool)
    rows, copy_logits, gate_logits = memory_rows_train(
        memories, head, sources, mask, copy_slots
    )
    assert rows.shape == (2, slots, width) and copy_logits.shape == (
        2,
        copy_slots,
        tokens,
    )
    assert torch.equal(rows[:, : slots - copy_slots], memories[:, : slots - copy_slots])
    assert gate_logits.shape == (2, copy_slots)
    soft_only, none_logits, _ = memory_rows_train(memories, head, sources, mask, 0)
    assert torch.equal(soft_only, memories) and none_logits is None
    # Inference: with the gate closed nothing snaps; with the copy slots' memories
    # equal to source rows and the gate open, the run snaps to those tokens.
    snapper = MemoryTokenSnapper(table)
    source_ids = torch.arange(tokens)
    single = memories[0].clone()
    rows, record = memory_rows_infer(
        single, head, sources[0], source_ids, snapper, table, copy_slots
    )
    assert (
        rows.shape == (slots, width)
        and record["spans"] == []
        and record["snapped_ids"] == []
    )
    assert torch.equal(rows[: slots - copy_slots], single[: slots - copy_slots])
    single[slots - copy_slots :] = table.weight[[2, 3]]
    with torch.no_grad():
        head.gate.bias.fill_(9.0)
    rows, record = memory_rows_infer(
        single, head, sources[0], source_ids, snapper, table, copy_slots
    )
    assert record["spans"] == [[0, 2, 2]] and record["snapped_ids"] == [2, 3]
    assert torch.equal(rows[slots - copy_slots :], table.weight[[2, 3]])


# ----------------------------------------------------------------------------
# schedule, metrics, resume


def test_lr_scale_warms_up_then_decays_to_ten_percent():
    assert lr_scale(0, 10000, 500) == pytest.approx(1 / 500)
    assert lr_scale(499, 10000, 500) == pytest.approx(1.0)
    assert lr_scale(9999, 10000, 500) == pytest.approx(0.1, abs=1e-6)
    assert lr_scale(0, 20, 500) == pytest.approx(0.5)  # ramp clamped to total // 10
    assert lr_scale(1, 20, 500) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        lr_scale(20, 20, 500)


def test_squad_metrics_normalize_articles_punctuation_and_case():
    assert exact_match("The Normans", "normans.") == 1.0
    assert token_f1(
        "Chaz Schilens", "Chaz Schilens, JaMarcus Russell"
    ) == pytest.approx(2 * 1 * 0.5 / 1.5)
    assert token_f1("", "") == 1.0 and token_f1("x", "") == 0.0
    assert token_f1("nothing", "France") == 0.0


def test_resume_state_round_trips_weights_optimizer_and_cursor(tmp_path):
    torch.manual_seed(0)
    reader, head = nn.Linear(4, 4), CopyHead(width=4)
    optimizer = torch.optim.AdamW(
        list(reader.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=0.01
    )
    loss = reader(torch.randn(2, 4)).sum() + head.gate(torch.randn(2, 4)).sum()
    loss.backward()
    optimizer.step()
    parts = [([3, 4], None), ([5], 4)]
    labels, rows = mixture_order(parts, 23)
    state = {
        "step": 3,
        "reader": {k: v.detach().cpu() for k, v in reader.state_dict().items()},
        "copy_head": {
            k: v.detach().cpu().clone() for k, v in head.state_dict().items()
        },
        "adapters": None,
        "optimizer": optimizer.state_dict(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": [],
        "result": {"cursor": 7, "history": [{"step": 3}], "memory_scale_factor": 0.5},
    }
    atomic_checkpoint(tmp_path / "resume.pt", state)
    loaded = torch.load(tmp_path / "resume.pt", map_location="cpu", weights_only=True)
    fresh_reader, fresh_head = nn.Linear(4, 4), CopyHead(width=4)
    fresh_reader.load_state_dict(loaded["reader"])
    fresh_head.load_state_dict(loaded["copy_head"])
    fresh_optimizer = torch.optim.AdamW(
        list(fresh_reader.parameters()) + list(fresh_head.parameters()), lr=1e-3
    )
    fresh_optimizer.load_state_dict(loaded["optimizer"])
    assert torch.equal(fresh_reader.weight, reader.weight) and torch.equal(
        fresh_head.gate.bias, head.gate.bias
    )
    assert fresh_optimizer.state_dict()["param_groups"][0]["weight_decay"] == 0.01
    assert loaded["step"] == 3 and loaded["result"]["cursor"] == 7
    # The data cursor resumes on the identical epoch order.
    again_labels, again_rows = mixture_order(parts, 23)
    cursor = loaded["result"]["cursor"]
    assert torch.equal(labels[cursor:], again_labels[cursor:]) and torch.equal(
        rows[cursor:], again_rows[cursor:]
    )


# ----------------------------------------------------------------------------
# Memorilla paper options: chunking, dual streams, recipes, eval metrics


def test_split_sentences_concatenates_back_and_keeps_the_space_prefix():
    text = ' He said "go." Then left! Why? Title line\n\nNext para here.  '
    pieces = split_sentences(text)
    assert "".join(pieces) == text
    assert pieces == [
        ' He said "go."',
        " Then left!",
        " Why?",
        " Title line",
        "\n\nNext para here.  ",
    ]
    assert all(p[0].isspace() for p in pieces)
    assert split_sentences(" no terminal punctuation") == [" no terminal punctuation"]
    assert split_sentences("") == []


def test_pack_sentences_is_greedy_splits_long_sentences_and_keeps_order():
    sentences = [[1] * 3, [2] * 4, [3] * 2, [4] * 11, [5] * 1, [6] * 5, [], [7] * 2]
    chunks = pack_sentences(sentences, 6)
    # 3 fit, +4 would exceed 6 -> close; 4+2 = 6 fits; the 11-long sentence
    # closes nothing new (current full) and is cut 6 + 5 with the 5-id rest
    # open: +1 fits (6), then 5 opens a new chunk, +2 would exceed -> close.
    assert chunks == [
        [1] * 3,
        [2] * 4 + [3] * 2,
        [4] * 6,
        [4] * 5 + [5],
        [6] * 5,
        [7] * 2,
    ]
    assert [t for c in chunks for t in c] == [t for s in sentences for t in s]
    assert all(1 <= len(c) <= 6 for c in chunks)
    assert pack_sentences([[9] * 13], 6) == [[9] * 6, [9] * 6, [9]]
    with pytest.raises(ValueError):
        pack_sentences([[1]], 0)


def test_chunk_documents_packs_per_document_within_128_tokens_in_order():
    tokenizer = FakeTokenizer()

    def encode_many(texts):
        return [tokenizer.encode(t, add_special_tokens=False) for t in texts]

    sentence = " " + " ".join(f"s{i}" for i in range(50)) + "."
    long_sentence = " " + " ".join(f"x{i}" for i in range(300))
    documents = [sentence * 3, long_sentence, " short one."]
    chunks = chunk_documents(documents, encode_many)
    lengths = [len(c) for c in chunks]
    # 51 tokens per sentence: two pack (102), the third opens a chunk; the
    # 300-token sentence is cut 128 + 128 + 44; documents never share a chunk.
    assert lengths == [102, 51, 128, 128, 44, 3]
    assert all(n <= CHUNK_TOKENS for n in lengths)
    flat = [t for c in chunks for t in c]
    assert flat == [t for d in encode_many(documents) for t in d]


def test_cap_chunks_and_rank_order_keep_the_best_in_original_order():
    scores = [0.1, 0.9, 0.5, 0.9, -0.2, 0.3]
    assert rank_order(scores) == [1, 3, 2, 5, 0, 4]  # ties by index
    assert cap_chunks(scores, 3) == ([1, 2, 3], 3)
    assert cap_chunks(scores, 9) == (list(range(6)), 0)
    assert cap_chunks(scores, None) == (list(range(6)), 0)  # eval: no cap
    candidates = [(s, i, f"payload{i}") for i, s in enumerate(scores)]
    top = keep_top(candidates[::-1], 4)  # arrival order does not matter
    assert [i for _, i, _ in top] == rank_order(scores)[:4]
    with pytest.raises(ValueError):
        cap_chunks(scores, 0)


def test_pooled_rows_match_pooled_query_per_chunk_and_keep_chunk_order():
    torch.manual_seed(2)
    final = torch.randn(3, 6, 8)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]],
        dtype=torch.long,
    )
    rows = pooled_rows(final, mask)
    assert rows.shape == (3, 8)
    for i in range(3):
        expected = pooled_query(final[i : i + 1], mask[i : i + 1])[0]
        assert torch.allclose(rows[i], expected, atol=1e-6)
    assert torch.allclose(rows.norm(dim=-1), torch.ones(3), atol=1e-2)
    query = torch.randn(8)
    shared = chunk_cosine(rows, query)
    paired = chunk_cosine(rows, query.expand(3, -1))
    assert torch.allclose(shared, paired, atol=1e-6)
    with pytest.raises(ValueError):
        pooled_rows(final, torch.zeros(3, 6, dtype=torch.long))


class FakeTaps:
    """Encoder taps of a token table: final = table[ids], tap = 2 * final + 1."""

    def __init__(self, table):
        self.table, self.calls = table, []

    def run(self, tokens):
        self.calls.append(tuple(tokens["input_ids"].shape))
        final = self.table[tokens["input_ids"]]
        return final, final * 2 + 1

    def close(self):
        pass


def fake_encoder_models(width=12):
    torch.manual_seed(4)
    return {
        "tokenizer_e": FakeTokenizer(),
        "taps": FakeTaps(torch.randn(3000, width)),
        "embedding": nn.Embedding(3000, 6).requires_grad_(False),
        "text_model": nn.Linear(1, 1),
        "specials": (1, 1),
    }


def expected_pooled(models, ids):
    states = models["taps"].table[torch.tensor([[2, *ids, 1]])]
    return pooled_query(states, torch.ones(1, len(ids) + 2, dtype=torch.long))[0]


def test_dual_stream_encoding_pools_every_chunk_and_reads_tokens_of_the_top_chunks():
    models = fake_encoder_models()
    encoder = SoftCaseEncoder(
        models,
        256,
        16,
        4096,
        chunk=2,
        chunking="sentence128",
        reader_streams="dual",
        max_chunks=7,
        token_chunks=3,
        count_document_tokens=True,
    )
    words = [" ".join(f"d{d}w{i}" for i in range(4 + d)) for d in range(6)]
    documents = [*[w + "." for w in words], " ".join(f"long{i}" for i in range(300))]
    rows = [
        {"question": "Where is d3w1 kept?", "documents": documents},
        {"question": "Who?", "documents": ["Mark Twain wrote it.", "Other text."]},
    ]
    encoded = encoder.encode(rows, False)
    tokenizer = models["tokenizer_e"]
    prefixed = [" " + d for d in documents]
    chunks = chunk_documents(
        prefixed, lambda ts: [tokenizer.encode(t, add_special_tokens=False) for t in ts]
    )
    assert [len(c) for c in chunks] == [5, 6, 7, 8, 9, 10, 128, 128, 44]
    question = tokenizer.encode("Where is d3w1 kept?", add_special_tokens=False)
    query = expected_pooled(models, question)
    vectors = torch.stack([expected_pooled(models, c) for c in chunks])
    scores = chunk_cosine(vectors, query).tolist()
    kept, dropped = cap_chunks(scores, 7)
    top = rank_order(scores)[:3]
    stats = encoded["stats"][0]
    assert stats["chunks"] == 9 and stats["pooled_rows"] == 7
    assert stats["chunks_dropped"] == dropped == 2 and encoder.chunks_dropped == 2
    assert stats["token_chunks"] == 3
    token_rows = 2 * sum(len(chunks[i]) + 1 for i in top)
    assert stats["token_rows"] == token_rows
    assert stats["document_tokens"] == sum(len(c) for c in chunks)
    # Keys: pooled rows of every kept chunk (original order), then the final
    # rows and the tap rows of the top chunks in rank order.
    docs, types = encoded["docs"][0], encoded["types"][0]
    assert torch.allclose(docs[:7], vectors[kept], atol=1e-5)
    assert types[:7].tolist() == [0] * 7 and types[7 : 7 + token_rows].tolist() == (
        [1] * token_rows
    )
    first = chunks[top[0]]
    states = models["taps"].table[torch.tensor([*first, 1])]
    expected_rows = feature_channel(nn.functional.normalize(states, dim=-1))
    assert torch.allclose(docs[7 : 7 + len(first) + 1], expected_rows, atol=1e-6)
    half = token_rows // 2
    tap = feature_channel(nn.functional.normalize(states * 2 + 1, dim=-1))
    assert torch.allclose(docs[7 + half : 7 + half + len(first) + 1], tap, atol=1e-6)
    assert not encoded["mask"][0, : 7 + token_rows].any()
    # Copy sources and targets cover exactly the selected chunks, in rank order.
    selected = [chunks[i] for i in top]
    assert encoded["doc_ids"][0] == selected
    flat = [t for c in selected for t in c]
    assert encoded["source_ids"][0, : len(flat)].tolist() == flat
    assert int(encoded["source_mask"][0].sum()) == len(flat)
    second = chunks[top[1]]
    word = tokenizer.decode([second[2]])
    span = answer_span(encoded["doc_ids"][0], answer_variants(tokenizer, word), 8)
    assert span == (1, 2, 1)
    match, copyable = copy_targets(
        encoded["doc_ids"], [span, None], 8, encoded["source_ids"].shape[1]
    )
    position = len(selected[0]) + 2
    assert match[0, 0].nonzero().flatten().tolist() == [position]
    assert copyable[0].tolist() == [True] + [False] * 7 and not copyable[1].any()
    assert int(encoded["source_ids"][0, position]) == second[2]
    unselected = chunks[rank_order(scores)[-1]]
    missing = tokenizer.decode([unselected[0]])
    assert answer_span(selected, answer_variants(tokenizer, missing), 8) is None
    # The second case shares the batch: its own pooled rows and padding.
    other = encoded["stats"][1]
    assert other["chunks"] == other["pooled_rows"] == 2 and other["token_chunks"] == 2
    used = other["pooled_rows"] + other["token_rows"]
    assert not encoded["mask"][1, :used].any() and encoded["mask"][1, used:].all()
    assert encoded["queries"].shape == (2, 12)
    # One question batch plus ceil(11 chunks / 2) chunk batches.
    assert len(models["taps"].calls) == 1 + 6
    with pytest.raises(ValueError):
        encoder.encode(rows, True)


def test_sentence_chunking_replaces_truncation_on_the_token_stream():
    models = fake_encoder_models()
    encoder = SoftCaseEncoder(models, 256, 16, 4096, chunking="sentence128")
    text = " ".join(f"t{i}" for i in range(300))
    encoded = encoder.encode([{"question": "q?", "documents": [text]}], False)
    assert [len(d) for d in encoded["doc_ids"][0]] == [128, 128, 44]
    assert encoded["types"] is None and encoded["stats"][0]["chunks"] == 3
    legacy = SoftCaseEncoder(models, 256, 16, 4096)
    truncated = legacy.encode([{"question": "q?", "documents": [text]}], False)
    assert [len(d) for d in truncated["doc_ids"][0]] == [256]


def test_stream_type_starts_at_zero_and_init_from_stage1_tolerates_its_absence(
    tmp_path,
):
    stream = StreamTypeEmbedding(width=4)
    assert stream.weight.dtype == torch.float32 and not stream.weight.any()
    types = torch.tensor([[0, 0, 1, 1]])
    assert torch.equal(stream(types), torch.zeros(1, 4, 4))
    with torch.no_grad():
        stream.weight.copy_(torch.tensor([[1.0] * 4, [2.0] * 4]))
    assert stream(types)[0, :, 0].tolist() == [1.0, 1.0, 2.0, 2.0]
    # An interrupted Stage 1 run (resume.pt without stream_type).
    torch.manual_seed(0)
    reader, head = nn.Linear(4, 4), CopyHead(width=4)
    stage1 = tmp_path / "stage1"
    stage1.mkdir()
    atomic_checkpoint(
        stage1 / "resume.pt",
        {
            "step": 1000,
            "reader": reader.state_dict(),
            "copy_head": head.state_dict(),
            "adapters": None,
        },
    )
    fresh_reader, fresh_head = nn.Linear(4, 4), CopyHead(width=4)
    source, new = load_trained(stage1, fresh_reader, fresh_head, {}, stream)
    assert source == "resume.pt@1000" and new == ["stream_type.weight"]
    assert torch.equal(fresh_reader.weight, reader.weight) and not stream.weight.any()
    assert load_trained(stage1, nn.Linear(4, 4), CopyHead(width=4), {}) == (
        "resume.pt@1000",
        [],
    )
    # A dual candidate round-trips its stream-type rows.
    with torch.no_grad():
        stream.weight.fill_(0.25)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "protocol.json").write_text(
        json.dumps({"schema": gemma_soft_memory.SCHEMA})
    )
    written = write_candidate(candidate, reader, {}, head, 24, 8, False, stream)
    assert written["reader_streams"] == "dual"
    assert written["stream_type_sha256"] == digest(candidate / STREAM_TYPE_FILE)
    loaded = StreamTypeEmbedding(width=4)
    source, new = load_trained(
        candidate, nn.Linear(4, 4), CopyHead(width=4), {}, loaded
    )
    assert (
        source == "candidate"
        and new == []
        and torch.equal(loaded.weight, stream.weight)
    )
    with pytest.raises(ValueError):
        load_trained(candidate, nn.Linear(4, 4), CopyHead(width=4), {})


def test_reader_dropout_sets_every_dropout_and_acts_in_train_mode_only():
    torch.manual_seed(0)
    module = nn.ModuleDict(
        {
            "attention": nn.MultiheadAttention(8, 2, dropout=0.0, batch_first=True),
            "ffn": nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.0)),
        }
    )
    set_reader_dropout(module, 0.1)
    assert module["attention"].dropout == 0.1 and module["ffn"][1].p == 0.1
    x = torch.ones(4, 8)
    module.eval()
    assert torch.equal(module["ffn"](x), module["ffn"](x))
    module.train()
    assert not torch.equal(module["ffn"](x), module["ffn"](x))


def stage1_args(*extra):
    output = str(STAGE1_PROTOCOL.parent)
    argv = ["train", "--recipe", "stage1", "--batch", "4", "--accum", "32"]
    return resolve_options(
        build_parser().parse_args([*argv, "--output", output, *extra])
    )


def protocol_from(args, stored, rows):
    parts = recipe_parts(args)
    total = sum(r if c is None else min(c, r) for (_, c), r in zip(parts, rows))
    env = {
        "encoder": stored["encoder"],
        "decoder": stored["decoder"],
        "layout": slot_layout(args.memory_slots, args.copy_slots),
        "targets": ("q_proj", "v_proj"),
        "placeholder": "<|image|>",
        "pad_id": 0,
        "softcap": 30.0,
        "eos_ids": (1, 50, 106),
        "script_sha256": digest(Path(gemma_soft_memory.__file__)),
        "versions": stored["versions"],
    }
    steps = schedule_steps(total, args.batch * args.accum, args.max_steps)
    return build_protocol(args, parts, rows, total, steps, env)


def test_stage1_protocol_is_unchanged_without_the_new_flags():
    if not STAGE1_PROTOCOL.exists():
        pytest.skip("the Stage 1 run's protocol.json is not on this machine")
    raw = STAGE1_PROTOCOL.read_text()
    stored = json.loads(raw)
    args = stage1_args()
    assert {k: getattr(args, k) for k in LEGACY_OPTIONS} == LEGACY_OPTIONS
    assert not options_active(args)
    protocol = protocol_from(args, stored, [d["rows"] for d in stored["datasets"]])
    assert list(protocol) == list(stored)
    differing = [k for k in stored if protocol[k] != stored[k]]
    assert differing in ([], ["script_sha256"])
    kept, change = resume_protocol(stored, protocol)
    assert json.dumps(kept, indent=2, allow_nan=False) + "\n" == raw
    assert change == (protocol["script_sha256"] if differing else None)
    # Any new option is recorded and makes a resume of the Stage 1 run fail.
    dropout = protocol_from(stage1_args("--reader-dropout", "0.1"), stored, [1979809])
    assert (
        dropout["reader_config"]["dropout"] == 0.1 and dropout["reader_dropout"] == 0.1
    )
    with pytest.raises(ValueError):
        resume_protocol(stored, dropout)


def test_resume_protocol_keeps_the_starting_script_and_init_entries():
    previous = {"a": 1, "script_sha256": "old", "init_source": "resume.pt@1000"}
    protocol = {"a": 1, "script_sha256": "new"}
    kept, change = resume_protocol(previous, protocol)
    assert kept == previous and list(kept) == list(previous) and change == "new"
    assert resume_protocol(previous, {"a": 1, "script_sha256": "old"})[1] is None
    with pytest.raises(ValueError):
        resume_protocol(previous, {"a": 2, "script_sha256": "old"})


def test_stage2_replay_recipe_adds_enwiki_replay_and_implies_the_paper_options():
    argv = ["train", "--recipe", "stage2-replay", "--init", "s1", "--output", "o"]
    args = resolve_options(build_parser().parse_args(argv))
    assert (args.chunking, args.reader_streams, args.reader_dropout) == (
        "sentence128",
        "dual",
        0.1,
    )
    assert (args.max_chunks, args.token_chunks, args.encoder_chunk_batch) == (
        512,
        8,
        64,
    )
    parts = recipe_parts(args)
    assert parts == [(n, None) for n in STAGE2_CONFIGS] + [("enwiki", REPLAY_CAP)]
    rows = [100] * 10 + [1979809]
    protocol = option_protocol(args, parts, rows)
    assert protocol["reader_streams"] == "dual" and protocol["slot_routing"] == "union"
    assert protocol["chunk_tokens"] == 128 and protocol["token_chunks"] == 8
    assert protocol["stream_type"]["shape"] == [2, 512]
    mixture = protocol["mixture"]
    assert mixture[-1] == {
        "config": "enwiki",
        "cap": REPLAY_CAP,
        "examples": REPLAY_CAP,
        "share": round(REPLAY_CAP / (1000 + REPLAY_CAP), 5),
    }
    assert sum(m["examples"] for m in mixture) == 1000 + REPLAY_CAP
    # Explicit flags override the preset; the plain stage2 recipe is untouched.
    override = resolve_options(
        build_parser().parse_args(
            [*argv, "--reader-streams", "tokens", "--reader-dropout", "0"]
        )
    )
    assert (override.chunking, override.reader_streams, override.reader_dropout) == (
        "sentence128",
        "tokens",
        0.0,
    )
    assert options_active(override)
    plain = resolve_options(
        build_parser().parse_args(["train", "--recipe", "stage2", "--output", "o"])
    )
    assert recipe_parts(plain) == [(n, None) for n in STAGE2_CONFIGS]
    assert (
        not options_active(plain)
        and option_protocol(plain, recipe_parts(plain), [1] * 10) == {}
    )
    with pytest.raises(ValueError):
        resolve_options(build_parser().parse_args([*argv, "--token-chunks", "600"]))


def test_question_type_buckets_by_first_word():
    cases = {
        "Who is Mark Hunter?": "who",
        "WHICH ROOM DID SHE ENTER?": "which",
        "what's the name of the ship": "what",
        "Why does he leave?": "why",
        "How does it end?": "how",
        "Where is the farm?": "where",
        "When did they marry?": "when",
        '"Who" said it?': "who",
        "In what year?": "other",
        "Whom did he meet?": "other",
        "": "other",
    }
    assert {q: question_type(q) for q in cases} == cases
    rows = [
        {"question_type": "why", "rougeL": 0.5, "f1": 0.25},
        {"question_type": "who", "rougeL": 1.0, "f1": 1.0},
        {"question_type": "why", "rougeL": 0.0, "f1": 0.75},
        {"question_type": "other", "rougeL": 0.2, "f1": 0.0},
    ]
    assert summarize_by_type(rows) == {
        "who": {"examples": 1, "rougeL": 1.0, "f1": 1.0},
        "why": {"examples": 2, "rougeL": 0.25, "f1": 0.5},
        "other": {"examples": 1, "rougeL": 0.2, "f1": 0.0},
    }


def test_noise_rows_match_each_row_norm_and_keep_padding():
    torch.manual_seed(5)
    rows = torch.randn(2, 4, 16) * torch.tensor([1.0, 3.0, 0.5, 2.0])[None, :, None]
    valid = torch.tensor([[True, True, True, False], [True, False, False, False]])
    noisy = noise_rows(rows, valid, torch.Generator().manual_seed(1))
    assert torch.allclose(
        noisy.norm(dim=-1)[valid], rows.norm(dim=-1)[valid], atol=1e-5
    )
    assert torch.equal(noisy[~valid], rows[~valid])
    cosine = nn.functional.cosine_similarity(noisy[valid], rows[valid], dim=-1)
    assert float(cosine.abs().max()) < 0.9
    again = noise_rows(rows, valid, torch.Generator().manual_seed(1))
    assert torch.equal(noisy, again)
    with pytest.raises(ValueError):
        noise_rows(rows, valid.long(), torch.Generator())


def test_rouge_fallback_unigram_and_lcs_f_measures():
    scores = rouge_fallback("The cat sat on the mat", "the cat lay on a mat")
    # Unigrams: the, cat, on, mat = 4 of 6 / 6; LCS: the cat on mat = 4.
    assert scores["rouge1"] == pytest.approx(4 / 6)
    assert scores["rougeL"] == pytest.approx(4 / 6)
    order = rouge_fallback("mat the on cat", "the cat on mat")
    assert order["rouge1"] == 1.0 and order["rougeL"] == pytest.approx(0.5)
    assert rouge_fallback("", "x") == {"rouge1": 0.0, "rougeL": 0.0}
    assert whitespace_token_f1("The cat sat.", "the cat") == pytest.approx(0.8)
    assert whitespace_token_f1("", "x") == 0.0


def test_eval_split_prefers_narrativeqa_test_and_memory_summary():
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        write_split(root, "narrativeqa", "validation", [2])
        assert eval_split_name(root, "narrativeqa", "auto") == "validation"
        write_split(root, "narrativeqa", "test", [2])
        assert eval_split_name(root, "narrativeqa", "auto") == "test"
        assert eval_split_name(root, "squad_v2", "auto") == "validation"
        assert eval_split_name(root, "narrativeqa", "validation") == "validation"
    stats = [
        {
            "chunks": 4,
            "pooled_rows": 4,
            "token_chunks": 2,
            "token_rows": 10,
            "read_tokens": 40,
            "document_tokens": 40,
            "chunks_dropped": 0,
        },
        {
            "chunks": 9,
            "pooled_rows": 7,
            "token_chunks": 3,
            "token_rows": 20,
            "read_tokens": 90,
            "document_tokens": 100,
            "chunks_dropped": 2,
        },
    ]
    summary = memory_input_summary(stats)
    assert summary["chunks_mean"] == 6.5 and summary["chunks_max"] == 9
    assert summary["chunks_dropped"] == 2 and summary["document_tokens_mean"] == 70
    assert "document_tokens_mean" not in memory_input_summary(stats, compact=True)


# ----------------------------------------------------------------------------
# evaluation: per-collection feature cache, whole-collection pooled stream,
# upstream metric parity


def test_feature_cache_round_trips_memory_and_disk_and_guards_documents(tmp_path):
    torch.manual_seed(6)
    pooled = feature_channel(nn.functional.normalize(torch.randn(3, 8), dim=-1))
    # Pooled rows are BF16 values after the feature channel: lossless storage.
    assert torch.equal(pooled.bfloat16().float(), pooled)
    chunks = [[5, 6, 7], [8], [9, 10]]
    row = {"config": "narrativeqa", "split": "test", "collection_id": 12}
    docs = [" one.", " two."]
    key = {"chunk_rule": "x", "encoder_weights_sha256": {"model.safetensors": "abc"}}
    cache = FeatureCache(key, tmp_path)
    digest_ = documents_digest(docs)
    assert cache.get(row, digest_) is None and cache.stats["misses"] == 1
    cache.put(row, digest_, chunks, pooled, 40)
    hit = cache.get(row, digest_)
    assert hit["chunks"] == chunks and torch.equal(hit["pooled"].float(), pooled)
    assert cache.stats["memory_hits"] == 1 and cache.stats["writes"] == 1
    assert (tmp_path / cache.key / "key.json").exists()
    # A new run (empty memory) reads the file; other documents are stale.
    again = FeatureCache(key, tmp_path)
    disk = again.get(row, digest_)
    assert disk["chunks"] == chunks and disk["document_tokens"] == 40
    assert torch.equal(disk["pooled"].float(), pooled)
    assert again.stats["disk_hits"] == 1
    assert FeatureCache(key, tmp_path).get(row, documents_digest([" other."])) is None
    other_key = FeatureCache({**key, "chunk_rule": "y"}, tmp_path)
    assert other_key.key != cache.key and other_key.get(row, digest_) is None
    # In-memory LRU bound.
    small = FeatureCache(max_entries=1)
    small.put(row, digest_, chunks, pooled, 40)
    small.put({**row, "collection_id": 13}, digest_, chunks, pooled, 40)
    assert small.get(row, digest_) is None and len(small.memory) == 1
    assert documents_digest(["ab", "c"]) != documents_digest(["a", "bc"])
    assert safe_name(12) == "12" and len(safe_name("a/b")) == 64
    with pytest.raises(ValueError):
        FeatureCache(None, tmp_path)


def test_cached_collections_give_the_same_dual_features_with_fewer_encoder_rows():
    models = fake_encoder_models()
    words = [" ".join(f"d{d}w{i}" for i in range(4 + d)) for d in range(6)]
    documents = [*[w + "." for w in words], " ".join(f"long{i}" for i in range(300))]
    rows = [
        {
            "question": q,
            "documents": documents,
            "config": "narrativeqa",
            "split": "test",
            "collection_id": 7,
        }
        for q in ("Where is d3w1 kept?", "Who wrote d5w2?")
    ]

    def encoder(cache):
        e = SoftCaseEncoder(
            models,
            256,
            16,
            4096,
            chunk=4,
            chunking="sentence128",
            reader_streams="dual",
            max_chunks=None,
            token_chunks=3,
            count_document_tokens=True,
        )
        e.cache = cache
        return e

    plain = encoder(None).encode(rows, False)
    cache = FeatureCache()
    cached_encoder = encoder(cache)
    first = cached_encoder.encode(rows[:1], False)
    calls = len(models["taps"].calls)
    second = cached_encoder.encode(rows[1:], False)
    # Second question on the same book: question batch + 3 top chunks only.
    assert [shape[0] for shape in models["taps"].calls[calls:]] == [1, 3]
    assert cache.stats["misses"] == 1 and cache.stats["memory_hits"] == 1
    # A question gets identical features whether its collection was cached.
    hit = cached_encoder.encode(rows[:1], False)
    for key in ("docs", "mask", "queries", "sources", "source_ids", "types"):
        assert torch.equal(hit[key], first[key]), key
    for case, encoded in ((0, first), (1, second)):
        assert encoded["doc_ids"][0] == plain["doc_ids"][case]
        stats = encoded["stats"][0]
        assert stats == plain["stats"][case]
        assert stats["pooled_rows"] == stats["chunks"] == 9  # whole collection
        used = stats["pooled_rows"] + stats["token_rows"]
        assert torch.allclose(
            encoded["docs"][0, :used], plain["docs"][case, :used], atol=1e-5
        )
        assert torch.equal(encoded["types"][0, :used], plain["types"][case, :used])
        assert torch.equal(
            encoded["source_ids"][0, : int(encoded["source_mask"][0].sum())],
            plain["source_ids"][case, : int(plain["source_mask"][case].sum())],
        )


def test_eval_defaults_read_the_whole_collection_and_paper_token_limits():
    args = build_parser().parse_args(["eval", "--candidate", "c", "--output", "o"])
    assert args.max_chunks is None and args.feature_cache is None
    assert (args.max_new_tokens, args.long_form_new_tokens) == (64, 512)
    probe = build_parser().parse_args(["probe", "--feature-cache", "d"])
    assert probe.max_chunks is None and probe.feature_cache == Path("d")


UPSTREAM = gemma_soft_memory.DEFAULT_ASSETS["upstream_source"]


def test_upstream_metrics_are_pinned_and_match_rouge_score_with_stemming(tmp_path):
    try:
        module = import_upstream_metrics(UPSTREAM)
    except ImportError as exc:
        pytest.skip(f"upstream metrics unavailable: {exc}")
    from rouge_score import rouge_scorer

    assert not any(
        k == "memorilla" or k.startswith("memorilla.")
        for k in __import__("sys").modules
    )
    metrics = EvalMetrics(UPSTREAM)
    assert metrics.module is not None and "Porter" in metrics.name
    prediction = metrics.clean(" The cats are running<|im_end|> ")
    assert prediction == "The cats are running"
    scored = metrics.row(prediction, " a cat runs ")
    direct = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True).score(
        "a cat runs", prediction
    )
    assert scored["rougeL"] == round(direct["rougeL"].fmeasure, 4) > 0.5
    pairs = [("The cat sat.", "the cat"), ("Mark Twain", "mark twain"), ("x", "y z")]
    for p, r in pairs:
        assert metrics.row(p, r)["f1"] == round(module.token_f1([p], [r]), 4)
        assert metrics.row(p, r)["f1"] == round(whitespace_token_f1(p, r), 4)
    assert set(metrics.corpus(["a b"], ["a b c"])) == {"bleu1", "bleu4"}
    # A changed source is refused; a missing tree falls back.
    copy = tmp_path / "memorilla"
    copy.mkdir()
    for name in UPSTREAM_METRIC_SOURCES:
        (copy / name).write_bytes((Path(UPSTREAM) / "memorilla" / name).read_bytes())
    with (copy / "metrics.py").open("a") as stream:
        stream.write("\n# edited\n")
    with pytest.raises(ValueError):
        import_upstream_metrics(tmp_path)
    fallback = EvalMetrics(tmp_path / "missing")
    assert fallback.module is None and fallback.name.startswith("local fallback")
    assert fallback.corpus(["a"], ["a"]) == {}
