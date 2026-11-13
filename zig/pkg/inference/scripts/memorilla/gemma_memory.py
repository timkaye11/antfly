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

"""Research-only latent memory for the EmbeddingGemma 2 / Gemma 4 pair.

The copy17 recipe (``encoder_lora.py``: per-token encoder features, a fresh
``MemoryModule`` reader, a gated copy head trained with the lexical token
cross-entropy plus pointer and gate losses, then contiguous-run copying,
nearest-token snapping and word completion at inference) moved to a
bidirectional 512-wide encoder and a frozen Gemma 4 decoder that reads the
snapped record as plain text inside its chat template. Nothing here is a
serving artifact; the Gemma decoder is never trained.

Design decisions, all recorded in ``protocol.json``:

- Encoder features: every document token row of the EmbeddingGemma 2 text
  model's final RMS-normalized state (width 512, before the 768 projection)
  plus its EOS row, and the same rows of ``hidden_states[tap]`` (default tap
  16 of 24), each L2-normalized and passed through the FP16->BF16 channel;
  final rows first, then tap rows. The tokenizer wraps a text as ``<bos> ids
  <eos>``: the BOS row is dropped, the EOS row kept, as the Qwen path kept its
  EOS state. Documents get one leading space (copy17); no self-context, the
  encoder is bidirectional.
- Question query: mean of the question's final-norm rows over its attention
  mask (BOS and EOS included), the model's own pooling at width 512.
- Record: ``asked-site`` ("`` code {code} at {site}``" / "`` key holder {name}
  at {site}``") tokenized by the Gemma 4 tokenizer, padded with the single
  space token (id 236743 in both vocabularies) to ``--memory-slots`` (24).
- Decoder prompt: one user turn of the Gemma 4 chat template holding
  ``SYSTEM + "\\nMemory" + record_text + "Memory\\n\\n" + question``, greedy,
  24 new tokens, stop on the decoder's EOS ids.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import torch
from encoder_lora import (
    COPY_HEAD_TEMPERATURE,
    LEXICAL_RECIPES,
    SYSTEM,
    CopyHead,
    CounterfactualGroupSampler,
    LexicalVocabularyTeacher,
    MemoryTokenSnapper,
    adapter_state,
    assert_frozen,
    atomic_checkpoint,
    complete_copied_words,
    copy_excluded_ids,
    copy_head_state,
    copy_with_contiguous_runs,
    copyable_mask,
    digest,
    feature_channel,
    frozen_versions,
    inject_lora,
    lexical_record_text,
    lexical_slot_weights,
    lexical_target_ids,
    lexical_token_ce_loss,
    load_split,
    lora_targets,
    pad_document_batch,
    pad_token_embeddings,
    pointer_losses,
    prefixed_documents,
    rescale_memory_projection,
    restore_adapters,
    save_json,
    scores,
    token_ids_embeddings,
    warmup_lr_scale,
)
from reference import UPSTREAM_MEMORY_SHA256
from torch import nn
from torch.nn import functional as F

SCHEMA = "antfly.memorilla.gemma-pair-research.v1"
ENCODER_WIDTH = 512
DEFAULT_TAP_LAYER = 16
MEMORY_SLOT_CHOICES = (16, 24, 32)
DEFAULT_MEMORY_SLOTS = 24
LEXICAL_RECORD = "asked-site"
QUERY_POOLING = "mean-final-norm-512-over-attention-mask-v1"
FEATURE_RECIPE = (
    "final-norm-512-rows-plus-tap-rows-eos-kept-bos-dropped-l2-fp16-bf16-v1"
)
PROMPT_RECIPE = "gemma4-chat-template-single-user-turn-plain-text-memory-v1"
DECODE_RECIPE = "contiguous-runs-snap-all-complete-words-v1"
MAX_ENCODER_TOKENS = 256
MAX_NEW_TOKENS = 24
WORKTREES = Path("/home/tim/antfly/.worktrees")
DEFAULT_ASSETS = {
    "encoder": WORKTREES / "fix-cuda-audio/.models/embeddinggemma2",
    "decoder": WORKTREES
    / "plan-memorilla-native-memory/.benchmark-assets/memorilla/models/gemma-4-e4b-it",
    "upstream_source": WORKTREES
    / "plan-memorilla-native-memory/.benchmark-assets/memorilla/upstream",
}
# The six readability records of docs/plans/memorilla-gemma-pair.md, section 2.
PROBE_RECORDS = (
    (
        " code Kg9-009369 at Mozeba Depot",
        "What is the exact access code for Mozeba Depot?",
        "Kg9-009369",
    ),
    (
        " key holder Komode Tuloco at Kopene Depot",
        "Who holds the brass key at Kopene Depot?",
        "Komode Tuloco",
    ),
    (
        " code Kg9-009369 at Mozeba Depot",
        "Who holds the brass key at Mozeba Depot?",
        "UNKNOWN",
    ),
    (
        " code Ly5-009275 at Mozeba",
        "What is the exact access code for Mozeba Depot?",
        "Ly5-009275",
    ),
    (
        " code Zx5-005023 at Nerute Depot",
        "What is the exact access code for Kopene Depot?",
        "UNKNOWN",
    ),
    (
        " key holder Gemebo Bunoka is Z Zehisi Depot",
        "Who holds the brass key at Zehisi Depot?",
        "Gemebo Bunoka",
    ),
)


# ----------------------------------------------------------------------------
# reader


def reader_config(slots, embedding_dim=ENCODER_WIDTH, output_dim=2560):
    """Upstream ``MemoryModule`` constructor arguments for the Gemma pair: the
    release geometry (8 heads, two cross-attention blocks, learned queries)
    at the encoder's 512 width and the decoder's hidden width."""
    if slots not in MEMORY_SLOT_CHOICES:
        raise ValueError(f"memory slots must be one of {MEMORY_SLOT_CHOICES}")
    return {
        "embedding_dim": embedding_dim,
        "output_dim": output_dim,
        "num_memories": slots,
        "num_heads": 8,
        "num_self_attn_layers": 0,
        "num_cross_attn_layers": 2,
        "dropout": 0.0,
        "retrieval_init": False,
    }


def import_upstream_memory(source_root):
    """The pinned upstream ``MemoryModule`` class (same digest check and
    import shim as ``reference.load_upstream``) for a fresh initialization."""
    source = Path(source_root) / "memorilla" / "memory.py"
    source_bytes = source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != UPSTREAM_MEMORY_SHA256:
        raise ValueError("upstream memory.py differs from the pinned source")
    name = "_antfly_memorilla_gemma_pair_reader"
    utility_name = "memorilla.utils"
    utility = ModuleType(utility_name)
    utility.CHECKPOINT_FILE = "memory.pt"
    utility.CONFIG_FILE = "config.json"

    def not_used(*_args, **_kwargs):
        raise RuntimeError("checkpoint I/O is disabled in the pinned reader import")

    utility.checkpoint_file = not_used
    utility.load_memory_state = not_used
    previous = {k: sys.modules.get(k) for k in (utility_name, name)}
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import pinned source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[utility_name] = utility
    sys.modules[name] = module
    try:
        exec(compile(source_bytes, str(source), "exec"), module.__dict__)  # noqa: S102
        return module.MemoryModule
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def build_reader(source_root, slots, seed):
    """A freshly initialized FP32 reader (dropout disabled) for ``slots``."""
    memory_module = import_upstream_memory(source_root)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reader = memory_module(**reader_config(slots))
    reader = reader.float()
    for module in reader.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0
        if isinstance(module, nn.MultiheadAttention):
            module.dropout = 0
    return reader


# ----------------------------------------------------------------------------
# encoder features


def encoder_special_counts(tokenizer, sample=" sample text"):
    """(prefix, suffix) special-token counts the encoder tokenizer adds around a
    text; EmbeddingGemma 2 wraps ``<bos> ids <eos>`` -> (1, 1)."""
    with_specials = list(tokenizer.encode(sample))
    without = list(tokenizer.encode(sample, add_special_tokens=False))
    for prefix in range(len(with_specials) - len(without) + 1):
        if with_specials[prefix : prefix + len(without)] == without:
            return prefix, len(with_specials) - prefix - len(without)
    raise ValueError("encoder special tokens do not wrap the plain token ids")


class EncoderTaps:
    """Forward hooks on the EmbeddingGemma 2 text model capturing the final
    RMS-normalized state (512, before the embedding projection) and the output
    of layer ``tap`` (HF ``hidden_states[tap]``: ``layers[tap - 1]``).

    Hooks are used because the model's recorded ``hidden_states`` replace the
    last entry with the projected 768-wide output, so the pre-projection final
    state is not otherwise reachable.
    """

    def __init__(self, text_model, tap):
        layers = text_model.config.num_hidden_layers
        if isinstance(tap, bool) or not isinstance(tap, int):
            raise TypeError("tap layer must be an integer")
        if not 1 <= tap <= layers:
            raise ValueError(f"tap layer must be within 1..{layers}")
        self.tap, self.captured = tap, {}
        self.handles = [
            text_model.norm.register_forward_hook(self._capture("final")),
            text_model.layers[tap - 1].register_forward_hook(self._capture("tap")),
        ]
        self.text_model = text_model

    def _capture(self, key):
        def hook(_module, _inputs, output):
            self.captured[key] = output[0] if isinstance(output, tuple) else output

        return hook

    def run(self, tokens):
        """-> ``(final [B, L, 512], tap [B, L, 512])`` for a padded batch."""
        self.captured.clear()
        self.text_model(
            input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
        )
        final, tap = self.captured.get("final"), self.captured.get("tap")
        shape = (*tokens["input_ids"].shape, ENCODER_WIDTH)
        if final is None or tap is None or final.shape != shape or tap.shape != shape:
            raise RuntimeError("encoder taps did not capture [B, L, 512] states")
        return final, tap

    def close(self):
        for handle in self.handles:
            handle.remove()


def select_document_rows(states, attention_mask, counts, prefix=1, suffix=1):
    """Per document ``i`` (right-padded ``states`` [B, L, W]) keep its ``counts[i]``
    token rows followed by its ``suffix`` trailing special rows (the EOS
    state), dropping the ``prefix`` leading special rows (BOS) -> [M, W]."""
    if (
        states.ndim != 3
        or tuple(attention_mask.shape) != tuple(states.shape[:2])
        or len(counts) != states.shape[0]
        or any(int(n) < 1 for n in counts)
    ):
        raise ValueError(
            "document rows need [B, L, W] states, a [B, L] mask and B counts"
        )
    pieces = []
    for index, count in enumerate(counts):
        active = int(attention_mask[index].sum())
        if active != prefix + int(count) + suffix:
            raise ValueError("document row count disagrees with its token count")
        pieces.append(states[index, prefix:active])
    return torch.cat(pieces, dim=0)


def pooled_query(final, attention_mask):
    """Mean of the question's final-norm rows over its attention mask ->
    unit [1, W] through the feature channel (the model's own pooling at 512)."""
    if (
        final.ndim != 3
        or final.shape[0] != 1
        or attention_mask.shape != final.shape[:2]
    ):
        raise ValueError("query pooling needs one [1, L, W] question")
    mask = attention_mask[0].to(final.device, torch.float32)
    if not mask.sum() > 0:
        raise ValueError("question has no active tokens")
    pooled = (final[0].float() * mask[:, None]).sum(dim=0, keepdim=True) / mask.sum()
    if not torch.isfinite(pooled).all() or not (pooled.norm() > 0):
        raise ValueError("encoder produced a nonfinite or zero question vector")
    return feature_channel(F.normalize(pooled, p=2, dim=-1))


def assemble_features(final, tap, attention_mask, counts, prefix=1, suffix=1):
    """States for ``[documents..., question]`` -> ``(docs [1, 2M, W], query
    [1, W])``: all final-norm document rows, then the same rows of the tap
    layer, each L2-normalized in FP32 and passed through the feature channel."""
    if final.shape != tap.shape or final.ndim != 3 or final.shape[0] != len(counts) + 1:
        raise ValueError(
            "features need matching final/tap states for documents + question"
        )
    pieces = []
    for states in (final, tap):
        rows = select_document_rows(
            states[:-1], attention_mask[:-1], counts, prefix, suffix
        ).float()
        if not torch.isfinite(rows).all() or bool((rows.norm(dim=-1) == 0).any()):
            raise ValueError("encoder produced a nonfinite or zero document row")
        pieces.append(feature_channel(F.normalize(rows, p=2, dim=-1)))
    docs = torch.cat(pieces, dim=0).unsqueeze(0)
    return docs, pooled_query(final[-1:], attention_mask[-1:])


def case_documents(case):
    return prefixed_documents(case["documents"], True)


# ----------------------------------------------------------------------------
# records, decoding, prompt


def record_pad_id(tokenizer):
    """The record pad: the decoder's single token for one space."""
    pad = tokenizer.encode(" ", add_special_tokens=False)
    if len(pad) != 1:
        raise ValueError("record padding needs a single-token space")
    return int(pad[0])


def record_target_ids(tokenizer, case, slots):
    """Decoder ids of the case's ``asked-site`` record padded to ``slots``
    (truncated when longer; the asked field leads so it survives)."""
    text = lexical_record_text(case, LEXICAL_RECORD)
    return lexical_target_ids(tokenizer, text, truncate=True, slots=slots)


def render_record(tokenizer, ids):
    """Snapped record ids -> the text the decoder reads (pads become spaces)."""
    return tokenizer.decode([int(i) for i in ids], skip_special_tokens=False)


def memory_prompt_messages(record_text, question, system=SYSTEM):
    """One user turn: system text, the record framed by ``Memory``, the question."""
    if not isinstance(record_text, str) or not isinstance(question, str):
        raise TypeError("prompt needs record and question texts")
    return [
        {
            "role": "user",
            "content": f"{system}\nMemory{record_text}Memory\n\n{question.strip()}",
        }
    ]


def render_prompt(tokenizer, record_text, question, system=SYSTEM):
    """The Gemma 4 chat-template text with the generation prompt appended."""
    return tokenizer.apply_chat_template(
        memory_prompt_messages(record_text, question, system),
        tokenize=False,
        add_generation_prompt=True,
    )


def decoder_eos_ids(decoder_dir, tokenizer):
    """EOS ids the generation stops on: the model's generation config list
    (``<eos>``, ``<turn|>``, ...) plus the tokenizer's EOS."""
    ids = set()
    config = Path(decoder_dir) / "generation_config.json"
    if config.exists():
        value = json.loads(config.read_text()).get("eos_token_id")
        ids.update([value] if isinstance(value, int) else value or [])
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    if not ids:
        raise ValueError("decoder has no EOS ids")
    return tuple(sorted(int(i) for i in ids))


def decode_record(copy_head, memories, sources, source_ids, snapper, tokenizer, pad_id):
    """The copy17 inference chain on one case: contiguous-run copying, snapping
    every slot to its nearest decoder embedding row, then word completion of
    the copied spans -> dict with ids, text, spans, gates and diagnostics."""
    with torch.autocast(memories.device.type, enabled=False):
        output, gate, moved, spans = copy_with_contiguous_runs(
            copy_head, memories.float(), sources.float(), return_spans=True
        )
        _, ids, top1 = snapper(output)
    ids = [int(i) for i in ids.tolist()]
    source_ids = [int(i) for i in source_ids.tolist()]
    inserted = 0
    if spans:
        ids, inserted = complete_copied_words(ids, spans, source_ids, tokenizer, pad_id)
    return {
        "record_ids": ids,
        "record_text": render_record(tokenizer, ids),
        "spans": [list(map(int, s)) for s in spans],
        "gate": [round(float(g), 3) for g in gate.tolist()],
        "moved": int(moved),
        "inserted": int(inserted),
        "snap_cosine_min": round(float(top1.min()), 4),
    }


def generate_answer(
    decoder, tokenizer, prompt_text, eos_ids, max_new_tokens=MAX_NEW_TOKENS
):
    """Greedy continuation of the rendered prompt (which carries its own BOS)."""
    encoded = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").to(
        decoder.device
    )
    if tokenizer.bos_token_id is not None and int(encoded["input_ids"][0, 0]) != int(
        tokenizer.bos_token_id
    ):
        raise ValueError("chat template did not start the prompt with BOS")
    started = time.perf_counter()
    with torch.inference_mode():
        output = decoder.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
            eos_token_id=list(eos_ids),
            pad_token_id=int(tokenizer.pad_token_id),
        )
    new_ids = [int(i) for i in output[0, encoded["input_ids"].shape[1] :].tolist()]
    if not new_ids:
        raise RuntimeError("decoder generated nothing")
    return {
        "answer": tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
        "token_ids": new_ids,
        "finish_reason": "eos" if new_ids[-1] in eos_ids else "length",
        "decoder_seconds": time.perf_counter() - started,
        "prompt_tokens": int(encoded["input_ids"].shape[1]),
    }


# ----------------------------------------------------------------------------
# models


def gpu_compute_processes():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        text=True,
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def model_identity(directory):
    """Cheap identity of a local HF model directory: config digest and the
    sizes of its weight files (the 16 GB decoder is not hashed)."""
    directory = Path(directory)
    weights = sorted(directory.glob("*.safetensors"))
    if not weights:
        raise ValueError(f"no safetensors weights in {directory}")
    return {
        "path": str(directory),
        "config_sha256": digest(directory / "config.json"),
        "tokenizer_sha256": digest(directory / "tokenizer.json"),
        "weights": {w.name: w.stat().st_size for w in weights},
    }


def load_models(encoder_dir, decoder_dir, tap, rank, alpha, targets, train_encoder):
    """Encoder text model with adapters (BF16, frozen base), decoder (BF16,
    frozen), tokenizers, taps and the frozen embedding tables."""
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    tokenizer_e = AutoTokenizer.from_pretrained(encoder_dir, local_files_only=True)
    tokenizer_e.padding_side = "right"
    tokenizer_d = AutoTokenizer.from_pretrained(decoder_dir, local_files_only=True)
    encoder = AutoModel.from_pretrained(
        encoder_dir, local_files_only=True, dtype=torch.bfloat16
    )
    text_model = encoder.language_model.requires_grad_(False).cuda()
    del encoder
    if text_model.config.hidden_size != ENCODER_WIDTH:
        raise ValueError("encoder text width is not 512")
    adapters = inject_lora(text_model, rank, alpha, train_encoder, targets)
    expected = text_model.config.num_hidden_layers * len(targets)
    if len(adapters) != expected or any(
        not name.startswith("layers.") or ".self_attn." not in name for name in adapters
    ):
        raise ValueError("adapters did not land on every text attention projection")
    decoder = (
        AutoModelForCausalLM.from_pretrained(
            decoder_dir, local_files_only=True, dtype=torch.bfloat16
        )
        .requires_grad_(False)
        .cuda()
        .eval()
    )
    embedding = decoder.get_input_embeddings()
    if embedding.weight.shape[1] != reader_config(DEFAULT_MEMORY_SLOTS)["output_dim"]:
        raise ValueError("decoder hidden width is not the reader output width")
    return {
        "tokenizer_e": tokenizer_e,
        "tokenizer_d": tokenizer_d,
        "text_model": text_model,
        "adapters": adapters,
        "taps": EncoderTaps(text_model, tap),
        "decoder": decoder,
        "embedding": embedding,
        "eos_ids": decoder_eos_ids(decoder_dir, tokenizer_d),
        "pad_id": record_pad_id(tokenizer_d),
        "specials": encoder_special_counts(tokenizer_e),
        "attention": {
            "encoder": getattr(text_model.config, "_attn_implementation", None),
            "decoder": getattr(
                decoder.config.get_text_config(), "_attn_implementation", None
            ),
        },
    }


def check_shared_document_ids(tokenizer_e, tokenizer_d, documents):
    """Every document (space-prefixed) must tokenize to the same ids under both
    tokenizers; the copy head reads decoder rows at the encoder's positions."""
    checked = 0
    for text in documents:
        if tokenizer_e.encode(text, add_special_tokens=False) != tokenizer_d.encode(
            text, add_special_tokens=False
        ):
            raise ValueError(f"encoder and decoder tokenizers disagree on {text!r}")
        checked += 1
    return checked


class CaseEncoder:
    """Live encoding of cases: tokenizes ``[documents..., question]`` per case,
    runs the taps once over the batch and assembles reader inputs and copy
    sources."""

    def __init__(self, models):
        self.models = models
        self.ids = {}

    def document_ids(self, text):
        if text not in self.ids:
            self.ids[text] = torch.tensor(
                self.models["tokenizer_e"].encode(text, add_special_tokens=False),
                dtype=torch.long,
            )
        return self.ids[text]

    def encode(self, cases, grad):
        """-> ``(docs [B, M, W], mask [B, M], queries [B, W], sources [B, T, W],
        source_ids [B, T], source_mask [B, T])`` on the GPU."""
        tokenizer, taps = self.models["tokenizer_e"], self.models["taps"]
        prefix, suffix = self.models["specials"]
        per_case = [case_documents(case) for case in cases]
        texts = [
            t for docs, case in zip(per_case, cases) for t in docs + [case["question"]]
        ]
        tokens = tokenizer(texts, padding=True, return_tensors="pt").to("cuda")
        if tokens["input_ids"].shape[1] > MAX_ENCODER_TOKENS:
            raise ValueError("encoder batch exceeds the token budget")
        context = contextlib.nullcontext() if grad else torch.no_grad()
        with context, torch.autocast("cuda", dtype=torch.bfloat16):
            final, tap = taps.run(tokens)
        docs, queries, rows, offset = [], [], [], 0
        embedding = self.models["embedding"]
        for case, documents in zip(cases, per_case):
            ids = [self.document_ids(text) for text in documents]
            for index, doc_ids in enumerate(ids):
                row = tokens["input_ids"][
                    offset + index, prefix : prefix + len(doc_ids)
                ]
                if not torch.equal(row.cpu(), doc_ids):
                    raise ValueError(
                        "encoder batch ids differ from the copy source ids"
                    )
            count = len(documents) + 1
            d, q = assemble_features(
                final[offset : offset + count],
                tap[offset : offset + count],
                tokens["attention_mask"][offset : offset + count],
                [len(i) for i in ids],
                prefix,
                suffix,
            )
            docs.append(d)
            queries.append(q)
            copy_ids = torch.cat(ids).to("cuda")
            rows.append((token_ids_embeddings(embedding, copy_ids), copy_ids))
            offset += count
        batch, mask = pad_document_batch(docs)
        sources, source_ids, source_mask = pad_token_embeddings(rows)
        return batch, mask, torch.cat(queries, dim=0), sources, source_ids, source_mask


def read_memories(reader, docs, mask, queries, slots):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        memories = reader(docs, mask, queries)
    if memories.shape[1:] != (slots, reader.config.output_dim):
        raise ValueError("reader memory geometry changed")
    return memories


# ----------------------------------------------------------------------------
# evaluation


def evaluate_cases(models, reader, copy_head, snapper, cases, slots, progress=None):
    """Answer every case through the full chain -> result rows."""
    encoder, rows = CaseEncoder(models), []
    tokenizer_d, decoder = models["tokenizer_d"], models["decoder"]
    for index, case in enumerate(cases):
        started = time.perf_counter()
        with torch.inference_mode():
            docs, mask, query, sources, source_ids, _ = encoder.encode(
                [case], grad=False
            )
            memories = read_memories(reader, docs, mask, query, slots)[0]
            record = decode_record(
                copy_head,
                memories,
                sources[0],
                source_ids[0],
                snapper,
                tokenizer_d,
                models["pad_id"],
            )
        prompt = render_prompt(tokenizer_d, record["record_text"], case["question"])
        answer = generate_answer(decoder, tokenizer_d, prompt, models["eos_ids"])
        row = {
            "id": case["id"],
            "group": case.get("group"),
            "category": case.get("category"),
            "cohort": case.get("cohort", "development"),
            "expected": case["answer"],
            "actual": answer["answer"],
            "correct": answer["answer"].strip() == case["answer"],
            "finish_reason": answer["finish_reason"],
            "seconds": round(time.perf_counter() - started, 3),
            **record,
        }
        rows.append(row)
        if progress is not None:
            progress(index, row)
    return rows


def probe(args):
    """Run the plan's six readability records through the frozen decoder with
    the plain-text prompt framing (no encoder, no reader)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.decoder, local_files_only=True)
    pad_id, eos_ids = record_pad_id(tokenizer), decoder_eos_ids(args.decoder, tokenizer)
    decoder = (
        AutoModelForCausalLM.from_pretrained(
            args.decoder, local_files_only=True, dtype=torch.bfloat16
        )
        .requires_grad_(False)
        .cuda()
        .eval()
    )
    rows = []
    for record, question, expected in PROBE_RECORDS:
        ids = tokenizer.encode(record, add_special_tokens=False)
        if len(ids) > args.memory_slots:
            raise ValueError("probe record exceeds the memory slots")
        ids = ids + [pad_id] * (args.memory_slots - len(ids))
        text = render_record(tokenizer, ids)
        answer = generate_answer(
            decoder, tokenizer, render_prompt(tokenizer, text, question), eos_ids
        )
        rows.append(
            {
                "record": record,
                "question": question,
                "expected": expected,
                "actual": answer["answer"],
                "correct": answer["answer"] == expected,
                "finish_reason": answer["finish_reason"],
                "decoder_seconds": round(answer["decoder_seconds"], 3),
            }
        )
        print(json.dumps({"event": "probe", **rows[-1]}), flush=True)
    summary = {
        "schema": SCHEMA,
        "decoder": model_identity(args.decoder),
        "memory_slots": args.memory_slots,
        "pad_id": pad_id,
        "eos_ids": list(eos_ids),
        "prompt_recipe": PROMPT_RECIPE,
        "system_prompt": SYSTEM,
        "correct": sum(r["correct"] for r in rows),
        "total": len(rows),
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.output, summary)
    print(
        json.dumps(
            {
                "event": "probe-summary",
                "correct": summary["correct"],
                "total": len(rows),
            }
        )
    )
    return summary


# ----------------------------------------------------------------------------
# training


def candidate_files():
    return (
        ("reader_sha256", "reader.safetensors"),
        ("encoder_lora_sha256", "encoder-lora.safetensors"),
        ("copy_head_sha256", "copy-head.safetensors"),
        ("protocol_sha256", "protocol.json"),
    )


def write_candidate(output, reader, adapters, copy_head, slots):
    from safetensors.torch import save_file

    save_file(
        {k: v.detach().cpu().contiguous() for k, v in reader.state_dict().items()},
        str(output / "reader.safetensors"),
    )
    save_file(adapter_state(adapters), str(output / "encoder-lora.safetensors"))
    save_file(
        {k: v.contiguous() for k, v in copy_head_state(copy_head).items()},
        str(output / "copy-head.safetensors"),
    )
    candidate = {
        "schema": SCHEMA,
        "rollout_qualified": False,
        "experimental_only": True,
        "memory_slots": slots,
        **{field: digest(output / file) for field, file in candidate_files()},
    }
    save_json(output / "candidate.json", candidate)
    return candidate


def load_candidate(directory):
    """(protocol, candidate) of a trained directory, digests verified."""
    directory = Path(directory)
    candidate = json.loads((directory / "candidate.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    if candidate["schema"] != SCHEMA or protocol["schema"] != SCHEMA:
        raise ValueError("candidate schema mismatch")
    for field, file in candidate_files():
        if candidate[field] != digest(directory / file):
            raise ValueError(f"{file} differs from the candidate digest")
    return protocol, candidate


def train(args):
    if not 1 <= args.steps <= 8192 or not 1 <= args.eval_every <= args.steps:
        raise ValueError("invalid bounded training budget")
    if not 1 <= args.batch <= 64 or not 1 <= args.accum <= 16:
        raise ValueError("batch must be 1..64 and accumulation 1..16")
    for rate in (args.reader_lr, args.encoder_lr, args.copy_head_lr):
        if not 0 < rate <= 1e-2:
            raise ValueError("learning rates must be within (0, 1e-2]")
    if not 1.0 <= args.lexical_name_weight <= 8.0:
        raise ValueError("name weight must be within 1..8")
    if not 0 <= args.copy_pointer_weight <= 10 or not 0 < args.copy_gate_weight <= 10:
        raise ValueError("pointer weight 0..10 and gate weight (0, 10] required")
    if args.sampling == "counterfactual-group" and args.accum % 8:
        raise ValueError(
            "counterfactual-group sampling needs accumulation in multiples of 8"
        )
    slots = args.memory_slots
    targets = lora_targets(args.lora_targets)
    train_encoder = not args.freeze_encoder
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if gpu_compute_processes() and not args.allow_shared_gpu:
        raise RuntimeError("GPU already occupied; refusing a competing training run")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    import transformers

    protocol = {
        "schema": SCHEMA,
        "arm": "gemma-pair",
        "encoder": model_identity(args.encoder),
        "decoder": model_identity(args.decoder),
        "upstream_memory_sha256": UPSTREAM_MEMORY_SHA256,
        "reader_config": reader_config(slots),
        "reader_init": "fresh",
        "memory_slots": slots,
        "tap_layer": args.tap_layer,
        "feature_recipe": FEATURE_RECIPE,
        "query_pooling": QUERY_POOLING,
        "document_space_prefix": True,
        "document_self_context": False,
        "rank": args.rank,
        "alpha": args.alpha,
        "target_modules": list(targets),
        "train_encoder": train_encoder,
        "seed": args.seed,
        "steps": args.steps,
        "batch": args.batch,
        "gradient_accumulation": args.accum,
        "sampling": args.sampling,
        "reader_lr": args.reader_lr,
        "encoder_lr": args.encoder_lr,
        "copy_head_lr": args.copy_head_lr,
        "schedule": args.schedule,
        "eval_every": args.eval_every,
        "dev_limit": args.dev_limit,
        "precision": "BF16 encoder/decoder autocast; FP32 reader, copy head and adapters; FP16->BF16 feature channel",
        "lexical_record": LEXICAL_RECORD,
        "lexical_objective": "token-ce",
        "lexical_recipe": LEXICAL_RECIPES["token-ce"],
        "lexical_loss_weight": 1.0,
        "lexical_name_weight": args.lexical_name_weight,
        "memory_scale_init": args.memory_scale_init,
        "copy_head": True,
        "copy_head_temperature": COPY_HEAD_TEMPERATURE,
        "copy_pointer_weight": args.copy_pointer_weight,
        "copy_gate_weight": args.copy_gate_weight,
        "decode_recipe": DECODE_RECIPE,
        "prompt_recipe": PROMPT_RECIPE,
        "system_prompt": SYSTEM,
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoder_frozen": True,
        "decoder_reads": "snapped record text",
        "dataset": str(args.data),
        "dataset_sha256": digest(args.data / "manifest.json"),
        "script_sha256": digest(Path(__file__)),
        "development_only": True,
        "rollout_qualified": False,
        "versions": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
        },
    }
    if args.resume:
        previous = json.loads((args.output / "protocol.json").read_text())
        if previous != protocol:
            raise ValueError("resume requires an identical protocol")
    save_json(args.output / "protocol.json", protocol)

    train_rows, dev_rows = load_split(args.data, "train"), load_split(args.data, "dev")
    if args.dev_limit is not None:
        dev_rows = dev_rows[: args.dev_limit]
    if {r["group"] for r in train_rows} & {r["group"] for r in dev_rows}:
        raise ValueError("train/dev group overlap")
    sampler = (
        CounterfactualGroupSampler(train_rows, args.seed)
        if args.sampling == "counterfactual-group"
        else None
    )
    models = load_models(
        args.encoder,
        args.decoder,
        args.tap_layer,
        args.rank,
        args.alpha,
        targets,
        train_encoder,
    )
    tokenizer_e, tokenizer_d = models["tokenizer_e"], models["tokenizer_d"]
    text_model, adapters, embedding = (
        models["text_model"],
        models["adapters"],
        models["embedding"],
    )
    documents = sorted(
        {d for rows in (train_rows, dev_rows) for r in rows for d in case_documents(r)}
    )
    protocol["shared_document_ids_checked"] = check_shared_document_ids(
        tokenizer_e, tokenizer_d, documents
    )
    protocol["encoder_special_rows"] = {
        "prefix_dropped": models["specials"][0],
        "suffix_kept": models["specials"][1],
    }
    protocol["pad_id"] = models["pad_id"]
    protocol["eos_ids"] = list(models["eos_ids"])
    protocol["attention_implementation"] = models["attention"]
    for case in train_rows + dev_rows:
        record_target_ids(tokenizer_d, case, slots)
    save_json(args.output / "protocol.json", protocol)

    reader = (
        build_reader(args.upstream_source, slots, args.seed).cuda().requires_grad_(True)
    )
    copy_head = CopyHead(width=reader.config.output_dim).cuda()
    teacher = LexicalVocabularyTeacher(embedding)
    snapper = MemoryTokenSnapper(embedding)
    excluded = copy_excluded_ids(tokenizer_d)
    encoder = CaseEncoder(models)
    projection = list(reader.memory_projection.parameters())
    projection_ids = {id(p) for p in projection}
    groups = [
        {
            "params": [p for p in reader.parameters() if id(p) not in projection_ids],
            "lr": args.reader_lr,
            "name": "reader_body",
        },
        {"params": projection, "lr": args.reader_lr, "name": "projection"},
        {
            "params": list(copy_head.parameters()),
            "lr": args.copy_head_lr,
            "name": "copy_head",
        },
    ]
    if train_encoder:
        groups.append(
            {
                "params": [p for p in text_model.parameters() if p.requires_grad],
                "lr": args.encoder_lr,
                "name": "encoder_lora",
            }
        )
    base_lr = {g["name"]: g["lr"] for g in groups}
    optimizer = torch.optim.AdamW(groups, weight_decay=0)
    trainable = [p for g in optimizer.param_groups for p in g["params"]]
    frozen = frozen_versions(text_model, models["decoder"])
    result_path = args.output / "result.json"
    result = {
        "schema": SCHEMA,
        "completed": False,
        "rollout_qualified": False,
        "trainable_parameters": {
            g["name"]: sum(p.numel() for p in g["params"]) for g in groups
        },
        "history": [],
        "evaluations": [],
    }
    start_step = 0
    if args.resume:
        state = torch.load(
            args.output / "resume.pt", map_location="cpu", weights_only=True
        )
        reader.load_state_dict(state["reader"])
        restore_adapters(adapters, state["adapters"])
        copy_head.load_state_dict(state["copy_head"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        result, start_step = state["result"], int(state["step"])
        result["completed"] = False
        base_lr["projection"] = args.reader_lr * result.get("memory_scale_factor", 1.0)
    save_json(result_path, result)

    def modes(training):
        text_model.train(training)
        reader.train(training)
        copy_head.train(training)

    def batch_cases(step, micro):
        absolute = step * args.accum + micro
        return [
            train_rows[sampler.index(absolute * args.batch + j)]
            if sampler is not None
            else train_rows[(absolute * args.batch + j) % len(train_rows)]
            for j in range(args.batch)
        ]

    def targets_for(cases):
        ids = torch.tensor(
            [record_target_ids(tokenizer_d, c, slots) for c in cases], device="cuda"
        )
        weights = torch.stack(
            [
                lexical_slot_weights(
                    tokenizer_d,
                    c,
                    LEXICAL_RECORD,
                    slots,
                    args.lexical_name_weight,
                    True,
                )
                for c in cases
            ]
        ).to("cuda")
        return ids, weights

    def evaluate(step):
        modes(False)
        started = time.monotonic()
        rows = evaluate_cases(models, reader, copy_head, snapper, dev_rows, slots)
        evaluation = {
            "step": step,
            "split": "dev",
            "scores": scores(rows),
            "seconds": round(time.monotonic() - started, 1),
            "rows": rows,
        }
        result["evaluations"].append(evaluation)
        save_json(result_path, result)
        print(
            json.dumps(
                {"event": "evaluation", "step": step, "scores": evaluation["scores"]}
            ),
            flush=True,
        )
        assert_frozen(frozen)
        modes(True)

    def checkpoint(step):
        assert_frozen(frozen)
        atomic_checkpoint(
            args.output / "resume.pt",
            {
                "step": step,
                "reader": {k: v.detach().cpu() for k, v in reader.state_dict().items()},
                "adapters": adapter_state(adapters),
                "copy_head": copy_head_state(copy_head),
                "optimizer": optimizer.state_dict(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "result": result,
            },
        )

    try:
        if not args.resume and args.memory_scale_init == "lexical-target":
            # Fold the teacher's row norm into the fresh projection once so the
            # log-norm term starts near zero (the Qwen trainer's init).
            modes(False)
            sample = batch_cases(0, 0)
            with torch.no_grad():
                docs, mask, query, *_ = encoder.encode(sample, grad=False)
                memories = read_memories(reader, docs, mask, query, slots)
                ids, _ = targets_for(sample)
                target_norm = float(
                    teacher.norms.index_select(0, ids.reshape(-1)).mean()
                )
            result["memory_scale_factor"] = rescale_memory_projection(
                reader, list(memories), target_norm
            )
            result["memory_scale_target_norm"] = target_norm
            base_lr["projection"] = args.reader_lr * result["memory_scale_factor"]
            print(
                json.dumps(
                    {
                        "event": "memory-scale-init",
                        "factor": result["memory_scale_factor"],
                        "target_norm": target_norm,
                    }
                ),
                flush=True,
            )
            del memories
        if not args.resume:
            evaluate(0)
        for step in range(start_step, args.steps):
            modes(True)
            optimizer.zero_grad(set_to_none=True)
            scale = warmup_lr_scale(step, args.steps, args.schedule)
            for group in optimizer.param_groups:
                group["lr"] = base_lr[group["name"]] * scale
            started = time.monotonic()
            totals = {
                "loss": 0.0,
                "lexical": 0.0,
                "pointer": 0.0,
                "gate": 0.0,
                "gate_mean": 0.0,
                "copyable": 0.0,
            }
            for micro in range(args.accum):
                cases = batch_cases(step, micro)
                docs, mask, query, sources, source_ids, source_mask = encoder.encode(
                    cases, grad=train_encoder
                )
                memories = read_memories(reader, docs, mask, query, slots)
                output, copy_logits, gate_logits = copy_head.forward_detailed(
                    memories, sources, source_mask
                )
                ids, weights = targets_for(cases)
                lexical = lexical_token_ce_loss(output, ids, teacher, slots, weights)
                copyable, match = copyable_mask(ids, source_ids, source_mask, excluded)
                pointer, gate = pointer_losses(
                    copy_logits, gate_logits, match, copyable, source_mask
                )
                loss = lexical + args.copy_pointer_weight * (
                    pointer + args.copy_gate_weight * gate
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                (loss / args.accum).backward()
                for key, value in (
                    ("loss", loss),
                    ("lexical", lexical),
                    ("pointer", pointer),
                    ("gate", gate),
                    ("gate_mean", torch.sigmoid(gate_logits.detach().float()).mean()),
                    ("copyable", copyable.float().mean()),
                ):
                    totals[key] += float(value.detach()) / args.accum
                del (
                    docs,
                    mask,
                    query,
                    sources,
                    memories,
                    output,
                    copy_logits,
                    gate_logits,
                    loss,
                )
            reader_grad = (
                sum(
                    float(p.grad.float().square().sum())
                    for p in reader.parameters()
                    if p.grad is not None
                )
                ** 0.5
            )
            lora_grad = (
                sum(
                    float(p.grad.float().square().sum())
                    for p in text_model.parameters()
                    if p.requires_grad and p.grad is not None
                )
                ** 0.5
            )
            if (
                not math.isfinite(reader_grad)
                or reader_grad == 0
                or (train_encoder and not lora_grad > 0)
            ):
                raise RuntimeError("missing or nonfinite reader/encoder gradient")
            assert_frozen(frozen)
            norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            )
            optimizer.step()
            torch.cuda.synchronize()
            row = {
                "step": step + 1,
                **{k: round(v, 5) for k, v in totals.items()},
                "grad_norm": round(norm, 4),
                "reader_grad": round(reader_grad, 4),
                "lora_grad": round(lora_grad, 4),
                "lr_scale": round(scale, 5),
                "seconds": round(time.monotonic() - started, 3),
                "max_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }
            result["history"].append(row)
            save_json(result_path, result)
            print(json.dumps({"event": "step", **row}), flush=True)
            if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                evaluate(step + 1)
                checkpoint(step + 1)
        result["completed"] = True
        result["frozen_base_versions_unchanged"] = True
        save_json(result_path, result)
        result["candidate"] = write_candidate(
            args.output, reader, adapters, copy_head, slots
        )
        save_json(result_path, result)
    except BaseException as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save_json(result_path, result)
        raise
    finally:
        models["taps"].close()


# ----------------------------------------------------------------------------
# evaluation command


def evaluate_candidate(args):
    from safetensors.torch import load_file

    protocol, candidate = load_candidate(args.candidate)
    if (args.fresh_groups is None) != (args.fresh_seed is None):
        raise ValueError("fresh challenge needs both --fresh-groups and --fresh-seed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if gpu_compute_processes() and not args.allow_shared_gpu:
        raise RuntimeError("GPU already occupied; refusing a competing evaluation")
    torch.backends.cuda.matmul.allow_tf32 = False
    data = Path(args.data) if args.data is not None else Path(protocol["dataset"])
    if digest(data / "manifest.json") != protocol["dataset_sha256"]:
        raise ValueError("evaluation corpus differs from the training corpus")
    slots = int(protocol["memory_slots"])
    encoder_dir = args.encoder or protocol["encoder"]["path"]
    decoder_dir = args.decoder or protocol["decoder"]["path"]
    if (
        model_identity(encoder_dir) != protocol["encoder"]
        or model_identity(decoder_dir) != protocol["decoder"]
    ):
        raise ValueError("evaluation base models differ from the training protocol")
    if args.fresh_groups is not None:
        from encoder_lora_export import fresh_cases

        effective, cases, prior_hashes = fresh_cases(
            data, args.prior_training_data, args.fresh_groups, args.fresh_seed
        )
        split = {
            "split": "fresh",
            "fresh_groups": args.fresh_groups,
            "fresh_seed": args.fresh_seed,
            "effective_seed": effective,
            "prior_training_manifest_hashes": prior_hashes,
        }
    else:
        cases, split = load_split(data, "dev"), {"split": "dev"}
    if args.limit is not None:
        cases = cases[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    models = load_models(
        encoder_dir,
        decoder_dir,
        int(protocol["tap_layer"]),
        int(protocol["rank"]),
        int(protocol["alpha"]),
        tuple(protocol["target_modules"]),
        False,
    )
    try:
        if (
            models["pad_id"] != protocol["pad_id"]
            or list(models["eos_ids"]) != protocol["eos_ids"]
        ):
            raise ValueError("decoder pad or EOS ids differ from the protocol")
        documents = sorted({d for c in cases for d in case_documents(c)})
        checked = check_shared_document_ids(
            models["tokenizer_e"], models["tokenizer_d"], documents
        )
        reader = build_reader(args.upstream_source, slots, int(protocol["seed"])).cuda()
        reader.load_state_dict(
            load_file(str(args.candidate / "reader.safetensors")), strict=True
        )
        restore_adapters(
            models["adapters"],
            load_file(str(args.candidate / "encoder-lora.safetensors")),
        )
        copy_head = CopyHead(width=reader.config.output_dim).cuda()
        copy_head.load_state_dict(
            load_file(str(args.candidate / "copy-head.safetensors")), strict=True
        )
        reader.eval()
        copy_head.eval()
        models["text_model"].eval()
        snapper = MemoryTokenSnapper(models["embedding"])
        started = time.monotonic()
        rows = evaluate_cases(
            models,
            reader,
            copy_head,
            snapper,
            cases,
            slots,
            progress=lambda i, r: print(
                json.dumps(
                    {
                        "event": "case",
                        "index": i,
                        "id": r["id"],
                        "actual": r["actual"],
                        "correct": r["correct"],
                    }
                ),
                flush=True,
            ),
        )
        result = {
            "schema": SCHEMA,
            "candidate": str(args.candidate),
            "candidate_sha256": digest(args.candidate / "candidate.json"),
            "protocol_sha256": candidate["protocol_sha256"],
            "decode_recipe": DECODE_RECIPE,
            "prompt_recipe": PROMPT_RECIPE,
            "shared_document_ids_checked": checked,
            **split,
            "cases": len(rows),
            "scores": scores(rows),
            "seconds": round(time.monotonic() - started, 1),
            "rows": rows,
        }
        save_json(args.output / "result.json", result)
        print(
            json.dumps({"event": "result", **split, "scores": result["scores"]}),
            flush=True,
        )
        return result
    finally:
        models["taps"].close()


# ----------------------------------------------------------------------------
# CLI


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def assets(p, upstream=True):
        p.add_argument("--encoder", type=Path, default=DEFAULT_ASSETS["encoder"])
        p.add_argument("--decoder", type=Path, default=DEFAULT_ASSETS["decoder"])
        if upstream:
            p.add_argument(
                "--upstream-source",
                type=Path,
                default=DEFAULT_ASSETS["upstream_source"],
            )
        p.add_argument("--allow-shared-gpu", action="store_true")

    t = sub.add_parser("train", help="train reader, copy head and encoder adapters")
    assets(t)
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    t.add_argument(
        "--memory-slots",
        type=int,
        choices=MEMORY_SLOT_CHOICES,
        default=DEFAULT_MEMORY_SLOTS,
    )
    t.add_argument("--tap-layer", type=int, default=DEFAULT_TAP_LAYER)
    t.add_argument("--rank", type=int, default=8)
    t.add_argument("--alpha", type=int, default=16)
    t.add_argument("--lora-targets", default="q_proj,v_proj")
    t.add_argument("--freeze-encoder", action="store_true")
    t.add_argument("--steps", type=int, default=2048)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--accum", type=int, default=8)
    t.add_argument("--eval-every", type=int, default=256)
    t.add_argument(
        "--sampling",
        choices=("counterfactual-group", "sequential"),
        default="counterfactual-group",
    )
    t.add_argument("--reader-lr", type=float, default=2e-4)
    t.add_argument("--encoder-lr", type=float, default=1e-4)
    t.add_argument("--copy-head-lr", type=float, default=2e-4)
    t.add_argument("--schedule", choices=("cosine", "constant"), default="cosine")
    t.add_argument("--lexical-name-weight", type=float, default=4.0)
    t.add_argument("--copy-pointer-weight", type=float, default=1.0)
    t.add_argument("--copy-gate-weight", type=float, default=3.0)
    t.add_argument(
        "--memory-scale-init",
        choices=("none", "lexical-target"),
        default="lexical-target",
    )
    t.add_argument("--seed", type=int, default=23)
    t.add_argument("--dev-limit", type=int)
    t.add_argument("--resume", action="store_true")

    e = sub.add_parser("eval", help="evaluate a candidate on dev or a fresh challenge")
    assets(e)
    e.add_argument("--candidate", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--data", type=Path)
    e.add_argument("--fresh-groups", type=int)
    e.add_argument("--fresh-seed", type=int)
    e.add_argument("--prior-training-data", type=Path, action="append", default=[])
    e.add_argument("--limit", type=int)

    p = sub.add_parser(
        "probe", help="run the plan's six readability records through the decoder"
    )
    assets(p, upstream=False)
    p.add_argument(
        "--memory-slots",
        type=int,
        choices=MEMORY_SLOT_CHOICES,
        default=DEFAULT_MEMORY_SLOTS,
    )
    p.add_argument("--output", type=Path)

    args = parser.parse_args(argv)
    if args.command == "train":
        if not args.encoder.is_dir() or not args.decoder.is_dir():
            raise SystemExit("encoder and decoder directories are required")
        train(args)
    elif args.command == "eval":
        if args.encoder is not None and not Path(args.encoder).is_dir():
            raise SystemExit("encoder directory missing")
        evaluate_candidate(args)
    else:
        probe(args)


if __name__ == "__main__":
    main()
