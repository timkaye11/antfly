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

import re

import pytest
import torch
from encoder_lora import SYSTEM, CopyHead, MemoryTokenSnapper
from gemma_memory import (
    DEFAULT_MEMORY_SLOTS,
    MEMORY_SLOT_CHOICES,
    PROBE_RECORDS,
    EncoderTaps,
    assemble_features,
    decode_record,
    decoder_eos_ids,
    encoder_special_counts,
    memory_prompt_messages,
    pooled_query,
    reader_config,
    record_pad_id,
    record_target_ids,
    render_prompt,
    render_record,
    select_document_rows,
)
from torch import nn
from torch.nn import functional as F


class FakeTokenizer:
    """Word-piece tokenizer over `` ?word`` / punctuation pieces with Gemma-like
    specials: BOS 2 and EOS 1 around a text, pad 0, a single-token space."""

    bos_token_id, eos_token_id, pad_token_id = 2, 1, 0

    def __init__(self, space_pieces=1):
        self.pieces = {"<pad>": 0, "<eos>": 1, "<bos>": 2, " ": 3}
        self.space_pieces = space_pieces

    def _split(self, text):
        if text == " ":
            return [" "] * self.space_pieces
        return re.findall(r" ?[A-Za-z0-9]{1,3}|-00|\S|\s", text)

    def encode(self, text, add_special_tokens=True):
        ids = [self.pieces.setdefault(p, len(self.pieces)) for p in self._split(text)]
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
        assert not tokenize and len(messages) == 1 and messages[0]["role"] == "user"
        text = f"<bos><|turn>user\n{messages[0]['content']}<turn|>\n"
        return text + ("<|turn>model\n" if add_generation_prompt else "")


def case():
    return {
        "id": "dev-0000-code-a",
        "question": "What is the exact access code for Pipofo Depot?",
        "documents": [
            "The access register at Pipofo Depot assigns its brass key to Komode Tuloco. Its access code is Kg9-009369.",
            "The brass key at Sopunu Depot belongs to Tirati Govepi. The code at Sopunu Depot is Jz6-001863.",
        ],
        "answer": "Kg9-009369",
        "category": "code",
        "entities": ["Pipofo Depot", "Sopunu Depot", "Komode Tuloco", "Tirati Govepi"],
    }


def test_reader_config_uses_encoder_and_decoder_widths_and_bounded_slots():
    config = reader_config(DEFAULT_MEMORY_SLOTS)
    assert (config["embedding_dim"], config["output_dim"]) == (512, 2560)
    assert config["num_memories"] == 24 and config["num_cross_attn_layers"] == 2
    assert config["dropout"] == 0 and not config["retrieval_init"]
    assert set(MEMORY_SLOT_CHOICES) == {16, 24, 32}
    with pytest.raises(ValueError):
        reader_config(20)


def test_special_counts_and_document_rows_keep_tokens_plus_eos():
    tokenizer = FakeTokenizer()
    assert encoder_special_counts(tokenizer) == (1, 1)
    # Two documents of 3 and 5 tokens, wrapped <bos> ids <eos>, right padded.
    states = torch.arange(2 * 8 * 4, dtype=torch.float32).reshape(2, 8, 4)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0]])
    rows = select_document_rows(states, mask, [3, 5])
    assert rows.shape == (3 + 1 + 5 + 1, 4)
    assert torch.equal(rows[:4], states[0, 1:5])  # tokens + EOS, BOS dropped
    assert torch.equal(rows[4:], states[1, 1:7])
    with pytest.raises(ValueError):
        select_document_rows(states, mask, [3, 4])


def test_assemble_features_orders_final_then_tap_rows_and_pools_the_question():
    torch.manual_seed(1)
    final, tap = torch.randn(3, 7, 6), torch.randn(3, 7, 6)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0, 0]]
    )
    docs, query = assemble_features(final, tap, mask, [2, 4])
    assert docs.shape == (1, 2 * (3 + 5), 6) and query.shape == (1, 6)
    assert torch.allclose(docs[0].norm(dim=-1), torch.ones(16), atol=1e-2)
    assert torch.allclose(query.norm(dim=-1), torch.ones(1), atol=1e-2)
    expected_first = F.normalize(final[0, 1:4], dim=-1).half().bfloat16().float()
    assert torch.allclose(docs[0, :3], expected_first)
    expected_tap = F.normalize(tap[0, 1:4], dim=-1).half().bfloat16().float()
    assert torch.allclose(docs[0, 8:11], expected_tap)
    mean = final[2, :3].mean(dim=0, keepdim=True)
    assert torch.allclose(query, F.normalize(mean, dim=-1).half().bfloat16().float())
    assert torch.allclose(pooled_query(final[2:], mask[2:]), query)
    with pytest.raises(ValueError):
        assemble_features(final, tap[:2], mask, [2, 4])


def test_encoder_taps_capture_final_norm_and_tap_layer_outputs():
    class Layer(nn.Module):
        def forward(self, hidden, **_):
            return (hidden + 1,)

    class Text(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"num_hidden_layers": 3})()
            self.layers = nn.ModuleList([Layer() for _ in range(3)])
            self.norm = nn.Identity()

        def forward(self, input_ids, attention_mask):
            hidden = torch.zeros(*input_ids.shape, 512)
            for layer in self.layers:
                hidden = layer(hidden)[0]
            return self.norm(hidden * 2)

    model = Text()
    taps = EncoderTaps(model, 2)
    tokens = {
        "input_ids": torch.ones(2, 5, dtype=torch.long),
        "attention_mask": torch.ones(2, 5),
    }
    final, tap = taps.run(tokens)
    assert torch.equal(final, torch.full((2, 5, 512), 6.0))
    assert torch.equal(tap, torch.full((2, 5, 512), 2.0))
    taps.close()
    with pytest.raises(ValueError):
        EncoderTaps(model, 4)


def test_record_targets_are_asked_site_records_padded_with_the_space_token():
    tokenizer = FakeTokenizer()
    assert record_pad_id(tokenizer) == 3
    ids = record_target_ids(tokenizer, case(), 16)
    text = " code Kg9-009369 at Pipofo Depot"
    body = len(tokenizer.encode(text, add_special_tokens=False))
    assert len(ids) == 16 and 1 < body < 16
    assert tokenizer.decode(ids) == text + " " * (16 - body)
    name = dict(case(), question="Who holds the brass key at Pipofo Depot?")
    assert tokenizer.decode(record_target_ids(tokenizer, name, 16)).startswith(
        " key holder Komode Tuloco at Pipofo Depot"
    )
    with pytest.raises(ValueError):
        record_pad_id(FakeTokenizer(space_pieces=2))


def gate_direction(memories, open_slots):
    """A gate weight that opens exactly ``open_slots``: the open rows summed,
    with every component along the closed rows projected out."""
    closed = [i for i in range(memories.shape[0]) if i not in open_slots]
    basis = []
    for index in closed:
        vector = memories[index].clone()
        for unit in basis:
            vector = vector - (vector @ unit) * unit
        if vector.norm() > 1e-6:
            basis.append(vector / vector.norm())
    direction = memories[list(open_slots)].sum(dim=0)
    for unit in basis:
        direction = direction - (direction @ unit) * unit
    return (direction * 40).unsqueeze(0)


def test_decode_chain_copies_snaps_and_completes_words():
    torch.manual_seed(3)
    tokenizer = FakeTokenizer()
    # Source pieces: " Kit" "ima" " at" " Moz" "eba" " Dep" "ot" "."
    source_ids = torch.tensor(
        tokenizer.encode(" Kitima at Mozeba Depot.", add_special_tokens=False)
    )
    kit, ima, at, moz, eba = source_ids[:5].tolist()
    label = tokenizer.encode(" code", add_special_tokens=False)  # " cod" "e"
    vocab = len(tokenizer.pieces) + 4
    table = nn.Embedding(vocab, 64).requires_grad_(False)
    with torch.no_grad():
        table.weight.copy_(F.normalize(torch.randn(vocab, 64), dim=-1))
    snapper = MemoryTokenSnapper(table)
    sources = table.weight[source_ids]
    head = CopyHead(width=64)
    pad = record_pad_id(tokenizer)
    # Eight slots: label, value run " Kit" "ima", " at", site run " Moz" "eba", pad.
    layout = [*label, kit, ima, at, moz, eba, pad]
    memories = table.weight[layout].clone()
    with torch.no_grad():
        head.gate.bias.fill_(-9.0)  # nothing copies: pure snapping
    record = decode_record(head, memories, sources, source_ids, snapper, tokenizer, pad)
    assert record["spans"] == [] and record["inserted"] == 0
    assert record["record_ids"] == layout
    assert record["record_text"] == " code Kitima at Mozeba "
    # Gate the value and site slots open while label, " at" and pad generate.
    with torch.no_grad():
        head.gate.weight.copy_(gate_direction(memories, (2, 3, 5, 6)))
    record = decode_record(head, memories, sources, source_ids, snapper, tokenizer, pad)
    assert [s[:2] for s in record["spans"]] == [[2, 4], [5, 7]]
    assert record["record_ids"] == layout
    assert record["snap_cosine_min"] > 0.99 and record["moved"] == 0
    # A site run that stops one subword short is completed from the source.
    short = table.weight[[*label, kit, ima, at, moz, pad, pad]].clone()
    with torch.no_grad():
        head.gate.weight.copy_(gate_direction(short, (2, 3, 5)))
    record = decode_record(head, short, sources, source_ids, snapper, tokenizer, pad)
    assert record["inserted"] == 1 and record["spans"] == [[2, 4, 0], [5, 6, 3]]
    assert record["record_ids"] == layout
    assert render_record(tokenizer, record["record_ids"]) == record["record_text"]


def test_prompt_is_one_user_turn_with_plain_text_memory_framing():
    tokenizer = FakeTokenizer()
    record = " code Kg9-009369 at Mozeba Depot   "
    question = "What is the exact access code for Mozeba Depot? "
    messages = memory_prompt_messages(record, question)
    assert len(messages) == 1 and messages[0]["role"] == "user"
    assert messages[0]["content"] == (
        SYSTEM + "\nMemory code Kg9-009369 at Mozeba Depot   Memory\n\n"
        "What is the exact access code for Mozeba Depot?"
    )
    text = render_prompt(tokenizer, record, question)
    assert text.startswith("<bos><|turn>user\n" + SYSTEM + "\nMemory")
    assert text.endswith(
        "Memory\n\nWhat is the exact access code for Mozeba Depot?<turn|>\n<|turn>model\n"
    )
    with pytest.raises(TypeError):
        memory_prompt_messages(None, question)


def test_eos_ids_merge_generation_config_and_tokenizer(tmp_path):
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": [1, 106]}')
    assert decoder_eos_ids(tmp_path, FakeTokenizer()) == (1, 106)
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": 106}')
    assert decoder_eos_ids(tmp_path, FakeTokenizer()) == (1, 106)


def test_probe_records_cover_the_plan_table():
    assert len(PROBE_RECORDS) == 6
    assert sum(expected == "UNKNOWN" for _, _, expected in PROBE_RECORDS) == 2
    assert all(record.startswith(" ") for record, _, _ in PROBE_RECORDS)
