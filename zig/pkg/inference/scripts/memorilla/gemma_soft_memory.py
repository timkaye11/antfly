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

"""Research-only "summarizing reader" for the EmbeddingGemma 2 / Gemma 4 pair.

Memorilla's own recipe on our encoder/decoder pair: soft memory slots trained
THROUGH the frozen Gemma 4 decoder on Memorilla-Data (Stage 1 ``enwiki``
reconstruction, Stage 2 the ten-config QA/summarization mixture), plus the
copy17 copy head on a tail of COPY slots so exact strings survive. Nothing
here is a serving artifact; the decoder is never trained.

Design decisions, all recorded in ``protocol.json``:

- Encoder features exactly as ``gemma_memory.py``: every document token row of
  the EmbeddingGemma 2 final RMS-normalized state (512) plus the rows of
  ``hidden_states[tap]`` (default 16), L2-normalized through the FP16->BF16
  channel, final rows first; the question query is the mean of its final rows.
  Features are computed live (per-token rows of two million passages do not
  fit on disk). The encoder is frozen by default; ``--encoder-lora`` trains
  rank-8 q/v adapters.
- Documents: one leading space, each truncated to ``--doc-tokens`` encoder
  tokens, at most ``--max-docs`` per example chosen by cosine of the pooled
  (mean final-norm) document vector to the pooled question when there are
  more (original order kept), total feature rows capped at ``--max-rows``
  (both layers counted) by dropping trailing documents and truncating the
  last kept one.
- Reader: the pinned upstream ``MemoryModule`` (512 -> 2560, 8 heads, two
  cross-attention blocks, ``--memory-slots`` learned queries). The first
  ``K - H`` slots are SOFT (trained only through the decoder loss); the last
  ``H = --copy-slots`` are COPY slots carrying the copy17 ``CopyHead``
  (pointer + gate), supervised when the answer appears verbatim as a token
  span of a document: gate target 1 on the span's slots (left-aligned) and 0
  elsewhere, pointer targets the span positions; otherwise gate targets 0 and
  no pointer loss. In training the gate-mixed soft output of the copy slots
  feeds the decoder; at inference copy slots inside a contiguous run (gate
  >= 0.5) snap to their nearest decoder token embedding and the rest pass as
  soft rows.
- Decoder prompt: the Gemma 4 chat template supports a ``system`` role
  (``<|turn>system\\n...<turn|>``), so the upstream layout is kept: system =
  ``HELPFUL_PROMPT + "\\nMemory" + placeholder * K + "Memory"``, user = the
  question, assistant = the answer. The placeholder is the image soft token
  ``<|image|>`` (id 258880): it is a single token, Gemma 4 was trained with
  soft rows at exactly these positions, and the HF multimodal path gives
  those positions the pad token's per-layer (PLE) identity row, which this
  trainer mirrors.
- Decoder inputs: ``Gemma4TextModel.forward`` refuses ``input_ids`` together
  with ``inputs_embeds`` (transformers 5.22), so the trainer builds
  ``inputs_embeds = embed_tokens(input_ids)`` (``Gemma4TextScaledWordEmbedding``:
  rows already scaled by sqrt(2560); the reader output lives in that scaled
  space, and the fresh projection is rescaled once to the mean scaled token
  norm), scatters the memory rows into the placeholder positions and passes
  ``per_layer_inputs = get_per_layer_inputs(input_ids with placeholders ->
  pad)`` explicitly, as the multimodal forward does for image soft tokens.
  Labels cover the answer tokens and the closing ``<turn|>`` (IGNORE_INDEX
  elsewhere); the loss is the mean cross-entropy over labeled tokens with the
  final logit softcap applied, computed only at labeled positions.
- Precision: BF16 autocast for the encoder, reader and decoder; FP32 reader,
  copy head and adapter parameters; gradient checkpointing on the frozen
  decoder (gradients flow to ``inputs_embeds`` only).

Memorilla paper options (arXiv 2610.21375v3; all OFF by default, so a run
started without them keeps a byte-identical ``protocol.json`` and resumes;
``--recipe stage2-replay`` turns the first three on):

- ``--chunking sentence128``: each space-prefixed document is split into
  sentence pieces (terminal punctuation plus closing quotes before
  whitespace, or the last character before a line break; the pieces
  concatenate back to the text) and packed sentence-greedily into chunks of
  at most 128 encoder tokens (the paper's TriviaQA rule); a sentence longer
  than 128 tokens is cut at 128 and its remainder stays open for the next
  sentences. Chunk order is kept and chunks never span documents. Replaces
  the ``--doc-tokens`` truncation of documents.
- ``--reader-streams dual``: the reader's keys are ``[pooled rows; token
  rows]``. Pooled rows: one per chunk for EVERY chunk (in training at most
  ``--max-chunks``, dropping the lowest pooled cosine to the question;
  evaluation reads the whole collection unless capped explicitly), the
  masked mean of its final-norm states over the attention mask (the
  question's own pooling), L2 + feature channel. Token rows: the final-norm
  and tap rows of the top ``--token-chunks`` chunks by that cosine, in rank
  order; copy sources and copy targets cover exactly those chunks' tokens.
  A trainable ``stream_type`` table [2, 512] (zero init, FP32) is added to
  pooled (row 0) and token rows (row 1) before the reader; every slot reads
  the union of keys (``slot_routing: union``: the serving reader takes a
  per-key mask only). One frozen encoder pass over all chunks (batches of
  ``--encoder-chunk-batch``) yields both streams.
- ``--reader-dropout``: dropout of the reader's attention and feed-forward
  blocks in train mode (the paper's 0.1; legacy 0).

Evaluation scores with the hash-pinned upstream ``memorilla.metrics``
(``clean_generation``, exact match, whitespace token F1, ROUGE-1/L with
Porter stemming, corpus BLEU-1/4) when its source tree and ``rouge_score``
/ ``nltk`` are available, and caches each collection's chunk ids and pooled
rows (``FeatureCache``: in memory, optionally on disk with
``--feature-cache``).
"""

from __future__ import annotations

import argparse
import bisect
import collections
import contextlib
import hashlib
import importlib.metadata
import itertools
import json
import math
import re
import string
import sys
import time
from pathlib import Path

import torch
from encoder_lora import (
    COPY_HEAD_TEMPERATURE,
    CopyHead,
    MemoryTokenSnapper,
    adapter_state,
    assert_frozen,
    atomic_checkpoint,
    copy_head_state,
    copy_with_contiguous_runs,
    digest,
    feature_channel,
    frozen_versions,
    lora_targets,
    pad_document_batch,
    pad_token_embeddings,
    pointer_losses,
    prefixed_documents,
    rescale_memory_projection,
    restore_adapters,
    save_json,
    token_ids_embeddings,
)
from gemma_memory import (
    DEFAULT_ASSETS,
    DEFAULT_MEMORY_SLOTS,
    DEFAULT_TAP_LAYER,
    ENCODER_WIDTH,
    FEATURE_RECIPE,
    MEMORY_SLOT_CHOICES,
    QUERY_POOLING,
    assemble_features,
    build_reader,
    gpu_compute_processes,
    load_models,
    model_identity,
    pooled_query,
    read_memories,
    reader_config,
)
from reference import UPSTREAM_MEMORY_SHA256, UPSTREAM_REVISION
from torch import nn
from torch.nn import functional as F

SCHEMA = "antfly.memorilla.gemma-soft-memory-research.v1"
HELPFUL_PROMPT = "You are a helpful assistant."
IGNORE_INDEX = -100
PLACEHOLDER_ID = 258880  # Gemma 4 ``<|image|>`` soft token (config.image_token_id)
PLACEHOLDER_RECIPE = "image-soft-token-258880-ple-identity-pad-v1"
PROMPT_RECIPE = "gemma4-chat-template-system-memory-user-question-v1"
INJECTION_RECIPE = "scaled-embed-tokens-scatter-explicit-per-layer-inputs-v1"
LABEL_RECIPE = "answer-tokens-plus-closing-turn-v1"
SELECTION_RULE = "pooled-final-norm-cosine-to-pooled-question-top-k-original-order-v1"
STAGE1_CONFIGS = ("enwiki",)
STAGE2_CONFIGS = (
    "squad_v2",
    "drop",
    "coqa",
    "pubmedqa",
    "quail",
    "pwc",
    "cnn_dailymail",
    "samsum",
    "dialogsum",
    "msmarco",
)
EVAL_CONFIGS = ("squad_v2", "drop", "triviaqa")
REPLAY_CAP = 200_000  # docs/plans/memorilla-gemma-pair.md section 5.1
RECIPES = {
    "stage1": STAGE1_CONFIGS,
    "stage2": STAGE2_CONFIGS,
    "stage2-replay": (*STAGE2_CONFIGS, "enwiki"),
    "depot": None,
}
RECIPE_CAPS = {"stage2-replay": {"enwiki": REPLAY_CAP}}
DEFAULT_DATA_ROOT = (
    Path("/home/tim/antfly/.worktrees/plan-memorilla-native-memory")
    / ".benchmark-assets/memorilla/data/memorilla-data"
)
SHARD_NAME = re.compile(r"-(\d+)-of-(\d+)\.parquet$")
COLUMNS = ("question", "answer", "documents")
OPTIONAL_COLUMNS = ("collection_id",)  # read when the split has it
MAX_NEW_TOKENS = 24
VALIDATION_SLICE = 256

# Memorilla paper options. ``LEGACY_OPTIONS`` are the values every run before
# them used; a protocol records the options only when one differs (or the
# recipe is a preset), so legacy protocols stay byte-identical.
LEGACY_OPTIONS = {"chunking": "none", "reader_streams": "tokens", "reader_dropout": 0.0}
RECIPE_PRESETS = {
    "stage2-replay": {
        "chunking": "sentence128",
        "reader_streams": "dual",
        "reader_dropout": 0.1,
    }
}
OPTION_DEFAULTS = {"max_chunks": 512, "token_chunks": 8, "encoder_chunk_batch": 64}
CHUNKING_CHOICES = ("none", "sentence128")
CHUNK_TOKENS = 128
CHUNK_RULES = {
    "none": "per-document-truncation-to-doc-tokens-v1",
    "sentence128": "sentence-greedy-128-encoder-tokens-long-sentence-cut-remainder-open-v1",
}
SENTENCE_RULE = (
    "terminal-punctuation-closers-before-space-or-last-char-before-newline-v1"
)
SENTENCE_END = re.compile(r"[.!?]+[\"'\u201d\u2019)\]]*(?=\s)|\S(?=[ \t\r\f\v]*\n)")
STREAM_CHOICES = ("tokens", "dual")
STREAM_TYPE_FILE = "stream-type.safetensors"
FEATURE_CACHE_SCHEMA = "antfly.memorilla.gemma-soft-pooled-chunks.v1"
POOLED_ROW_RECIPE = "mean-final-norm-512-over-attention-mask-l2-fp16-bf16-per-chunk-v1"
CHUNK_CAP_RULE = "drop-lowest-pooled-cosine-to-pooled-question-original-order-v1"
TOKEN_CHUNK_RULE = "top-pooled-cosine-to-pooled-question-rank-order-stable-v1"
SLOT_ROUTING = "union"
QA_NEW_TOKENS = 64  # paper short-answer limit
LONG_FORM_NEW_TOKENS = 512  # paper EnWiki reconstruction limit
LONG_FORM_CONFIGS = ("enwiki",)
QUESTION_TYPES = ("who", "which", "what", "why", "how", "where", "when")
WORDS = re.compile(r"[a-z0-9]+")
# Pinned upstream sources for metric parity (revision ``UPSTREAM_REVISION``):
# metrics.py is executed with benchmarks.py for its constants; data.py and
# paths.py only supply literal constants (read with ``ast``).
UPSTREAM_METRIC_SOURCES = {
    "metrics.py": "075d4942e65bf12484ec3c8380be26438903b1494bd02696b0cf167897e29caf",
    "benchmarks.py": "4302da50594cabdcedf1c9f79e1f1ddae712c55c78a41035f06a963a0ee658ed",
    "data.py": "51ecd40acd88a318b16108fd33aeddcdff520438bb7c988d2663f16068dba596",
    "paths.py": "4cd51d7cc2e026fdf600b82a31f47e63178842b089fd40275664c005b92af17a",
}


# ----------------------------------------------------------------------------
# data


def split_files(root, config, split):
    """Sorted parquet shards of ``data/<config>/<split>-*.parquet``."""
    return sorted((Path(root) / "data" / config).glob(f"{split}-*.parquet"))


def split_complete(files):
    """Whether ``<split>-XXXXX-of-YYYYY.parquet`` shards cover every index
    (upstream ``data.split_complete``); unnamed files count as complete."""
    shards = [m for f in files if (m := SHARD_NAME.search(f.name))]
    if not shards:
        return bool(files)
    totals = {int(m.group(2)) for m in shards}
    return len(totals) == 1 and {int(m.group(1)) for m in shards} == set(
        range(totals.pop())
    )


class ParquetSplit:
    """Lazy row access to one ``config/split``: shards -> row groups, one row
    group decoded at a time (``question``, ``answer``, ``documents``)."""

    def __init__(self, root, config, split, cache_groups=1):
        import pyarrow.parquet as pq

        self.config, self.split = config, split
        self.files = split_files(root, config, split)
        if not self.files or not split_complete(self.files):
            raise FileNotFoundError(f"{config}/{split} is missing or incomplete")
        self.groups, self.offsets = [], [0]
        for path in self.files:
            metadata = pq.ParquetFile(path).metadata
            for index in range(metadata.num_row_groups):
                rows = metadata.row_group(index).num_rows
                if rows:
                    self.groups.append((path, index))
                    self.offsets.append(self.offsets[-1] + rows)
        self.cache = collections.OrderedDict()
        self.cache_groups = max(1, int(cache_groups))
        names = pq.ParquetFile(self.files[0]).schema_arrow.names
        self.columns = [*COLUMNS, *(c for c in OPTIONAL_COLUMNS if c in names)]

    def __len__(self):
        return self.offsets[-1]

    @property
    def group_sizes(self):
        return [b - a for a, b in zip(self.offsets, self.offsets[1:])]

    def _group(self, group):
        if group not in self.cache:
            import pyarrow.parquet as pq

            path, index = self.groups[group]
            table = pq.ParquetFile(path).read_row_group(index, columns=self.columns)
            self.cache[group] = table.to_pydict()
            while len(self.cache) > self.cache_groups:
                self.cache.popitem(last=False)
        return self.cache[group]

    def row(self, index):
        if not 0 <= index < len(self):
            raise IndexError("row outside the split")
        group = bisect.bisect_right(self.offsets, index) - 1
        columns = self._group(group)
        local = index - self.offsets[group]
        row = {name: columns[name][local] for name in self.columns}
        row["config"], row["split"], row["row"] = self.config, self.split, index
        return row


def group_shuffled_stream(group_sizes, cap, generator):
    """Row indices of one split in a row-group-local shuffle: the row groups
    in random order, rows shuffled inside each group, cut to ``cap`` rows.
    Sequential consumption then touches one row group at a time."""
    pieces, offset = [], 0
    starts = []
    for size in group_sizes:
        starts.append(offset)
        offset += size
    order = torch.randperm(len(group_sizes), generator=generator).tolist()
    for group in order:
        size = group_sizes[group]
        pieces.append(starts[group] + torch.randperm(size, generator=generator))
    stream = torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.long)
    if cap is not None:
        stream = stream[: int(cap)]
    return stream


def mixture_order(parts, seed):
    """Deterministic one-epoch order over ``parts`` = ``[(group_sizes, cap),
    ...]`` -> ``(part_index [N], row_index [N])``: every part streams in its
    own row-group-local shuffle while the parts interleave uniformly at
    random, so ``N`` examples ago is always the same example for one seed."""
    generator = torch.Generator().manual_seed(int(seed))
    streams = [group_shuffled_stream(sizes, cap, generator) for sizes, cap in parts]
    labels = torch.cat(
        [torch.full((len(s),), i, dtype=torch.long) for i, s in enumerate(streams)]
    )
    if labels.numel() == 0:
        raise ValueError("the mixture has no examples")
    labels = labels[torch.randperm(labels.numel(), generator=generator)]
    rows = torch.zeros_like(labels)
    for index, stream in enumerate(streams):
        rows[labels == index] = stream
    return labels, rows


def parse_datasets(spec):
    """``"enwiki,squad_v2:20000"`` -> ``[("enwiki", None), ("squad_v2", 20000)]``."""
    parts = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        name, _, cap = item.partition(":")
        cap = int(cap) if cap else None
        if cap is not None and cap < 1:
            raise ValueError("dataset caps must be positive")
        parts.append((name, cap))
    if not parts or len({n for n, _ in parts}) != len(parts):
        raise ValueError("datasets must be distinct config names")
    return parts


# ----------------------------------------------------------------------------
# documents


def fit_documents(doc_ids, max_rows):
    """Keep the leading documents whose feature rows (two layers: ``2 *
    (tokens + EOS)`` each) fit ``max_rows``, truncating the last kept one;
    at least one document (possibly truncated) survives."""
    if not doc_ids or any(len(d) < 1 for d in doc_ids):
        raise ValueError("documents need at least one token each")
    budget_tokens = max_rows // 2
    kept, used = [], 0
    for ids in doc_ids:
        need = len(ids) + 1
        if used + need <= budget_tokens:
            kept.append(list(ids))
            used += need
            continue
        room = budget_tokens - used - 1
        if room >= 1:
            kept.append(list(ids[:room]))
        break
    if not kept:
        raise ValueError("row budget cannot hold one document token")
    return kept


def select_documents(doc_vectors, question_vector, max_docs):
    """Indices (original order) of the ``max_docs`` documents whose pooled
    vectors have the highest cosine with the pooled question."""
    if doc_vectors.ndim != 2 or question_vector.shape != (doc_vectors.shape[1],):
        raise ValueError("document selection needs [D, W] vectors and a [W] query")
    if doc_vectors.shape[0] <= max_docs:
        return list(range(doc_vectors.shape[0]))
    cosine = F.normalize(doc_vectors.float(), dim=-1) @ F.normalize(
        question_vector.float(), dim=-1
    )
    chosen = torch.topk(cosine, max_docs).indices.tolist()
    return sorted(chosen)


def split_sentences(text):
    """Sentence pieces of ``text`` that concatenate back to it: a piece ends
    after terminal punctuation (plus closing quotes or brackets) followed by
    whitespace, or at the last visible character before a line break; the
    next piece starts with that whitespace, so a space-prefixed document gives
    space-prefixed pieces. Trailing whitespace joins the last piece."""
    text = str(text)
    pieces, start = [], 0
    for end in sorted({m.end() for m in SENTENCE_END.finditer(text)}):
        if end > start:
            pieces.append(text[start:end])
            start = end
    rest = text[start:]
    if rest.strip() or not pieces:
        pieces.append(rest)
    else:
        pieces[-1] += rest
    return [piece for piece in pieces if piece]


def pack_sentences(sentence_ids, limit=CHUNK_TOKENS):
    """Sentence-greedy chunks of at most ``limit`` ids: whole sentences are
    packed in order while the next one still fits; a sentence longer than
    ``limit`` closes the open chunk and is cut into ``limit``-id pieces whose
    remainder stays open for the following sentences. Concatenating the
    chunks gives the concatenated input ids."""
    if limit < 1:
        raise ValueError("chunks need a positive token limit")
    chunks, current = [], []
    for ids in sentence_ids:
        ids = list(ids)
        if not ids:
            continue
        if len(current) + len(ids) <= limit:
            current.extend(ids)
            continue
        if current:
            chunks.append(current)
        while len(ids) > limit:
            chunks.append(ids[:limit])
            ids = ids[limit:]
        current = ids
    if current:
        chunks.append(current)
    return chunks


def encode_texts(tokenizer, texts):
    """Plain token ids (no specials) of every text; one batched call for a
    fast HF tokenizer, else ``encode`` per text."""
    texts = list(texts)
    if not texts:
        return []
    if callable(tokenizer):
        batch = tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [list(ids) for ids in batch]
    return [list(tokenizer.encode(t, add_special_tokens=False)) for t in texts]


def chunk_documents(documents, encode_many, limit=CHUNK_TOKENS):
    """Per-document sentence-greedy chunks (``pack_sentences``), documents in
    order, never spanning two documents -> list of id lists. ``encode_many``
    maps a list of texts to their id lists."""
    pieces = [split_sentences(text) for text in documents]
    ids = encode_many([p for doc in pieces for p in doc])
    chunks, offset = [], 0
    for doc in pieces:
        chunks.extend(pack_sentences(ids[offset : offset + len(doc)], limit))
        offset += len(doc)
    return chunks


def chunk_cosine(vectors, queries):
    """Row-wise cosine of ``vectors`` [N, W] with ``queries`` [N, W] or a
    shared [W] query -> [N] FP32."""
    vectors = F.normalize(vectors.float(), dim=-1)
    queries = F.normalize(queries.float(), dim=-1)
    if queries.ndim == 1:
        return vectors @ queries
    return (vectors * queries).sum(dim=-1)


def rank_order(scores):
    """Indices by descending score, ties by ascending index."""
    scores = torch.as_tensor(scores, dtype=torch.float32)
    return torch.sort(scores, descending=True, stable=True).indices.tolist()


def cap_chunks(scores, max_chunks):
    """-> ``(kept indices in original order, dropped count)`` keeping the
    ``max_chunks`` highest scores (pooled cosine to the question); ``None``
    keeps every chunk (evaluation default: the whole collection)."""
    if max_chunks is not None and max_chunks < 1:
        raise ValueError("max_chunks must be positive")
    count = len(scores)
    if max_chunks is None or count <= max_chunks:
        return list(range(count)), 0
    return sorted(rank_order(scores)[:max_chunks]), count - max_chunks


def keep_top(candidates, count):
    """``(score, index, payload)`` candidates -> the ``count`` best by
    descending score, ties by ascending index (``rank_order``'s order)."""
    return sorted(candidates, key=lambda c: (-c[0], c[1]))[:count]


def pooled_rows(final, attention_mask):
    """Per-chunk final-norm states [N, L, W] -> pooled rows [N, W]: the mean
    over each chunk's attention mask (``pooled_query``'s rule per row), L2 in
    FP32, then the feature channel."""
    if final.ndim != 3 or attention_mask.shape != final.shape[:2]:
        raise ValueError("pooled rows need [N, L, W] states and an [N, L] mask")
    mask = attention_mask.to(final.device, torch.float32)
    counts = mask.sum(dim=1, keepdim=True)
    if not bool((counts > 0).all()):
        raise ValueError("a chunk has no active tokens")
    pooled = (final.float() * mask[..., None]).sum(dim=1) / counts
    if not torch.isfinite(pooled).all() or bool((pooled.norm(dim=-1) == 0).any()):
        raise ValueError("encoder produced a nonfinite or zero chunk vector")
    return feature_channel(F.normalize(pooled, p=2, dim=-1))


def unit_feature_rows(rows):
    """Token rows [M, W] -> L2-normalized FP32 rows through the feature
    channel (``assemble_features``' rule)."""
    rows = rows.float()
    if not torch.isfinite(rows).all() or bool((rows.norm(dim=-1) == 0).any()):
        raise ValueError("encoder produced a nonfinite or zero document row")
    return feature_channel(F.normalize(rows, p=2, dim=-1))


def noise_rows(rows, valid, generator):
    """Every ``valid`` row of ``rows`` [B, M, W] replaced by Gaussian noise
    with the same L2 norm (the paper's noise-document sanity test); invalid
    (padding) rows are kept. ``generator`` is a CPU ``torch.Generator``."""
    if rows.ndim != 3 or valid.shape != rows.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("noise rows need [B, M, W] rows and a [B, M] bool mask")
    noise = torch.randn(rows.shape, generator=generator, dtype=torch.float32)
    noise = F.normalize(noise.to(rows.device), dim=-1)
    noise = noise * rows.float().norm(dim=-1, keepdim=True)
    return torch.where(valid[..., None], noise.to(rows.dtype), rows)


# ----------------------------------------------------------------------------
# prompt, labels, copy targets


def memory_messages(question, slots, placeholder, answer=None, system=HELPFUL_PROMPT):
    """Upstream chat layout: system prompt + ``\\nMemory`` + placeholders +
    ``Memory``; user = question; optional assistant answer."""
    if not isinstance(question, str) or slots < 1:
        raise ValueError("prompt needs a question text and positive slot count")
    messages = [
        {"role": "system", "content": f"{system}\nMemory{placeholder * slots}Memory"},
        {"role": "user", "content": question.strip()},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": str(answer).strip()})
    return messages


def render_prompt(tokenizer, question, slots, placeholder, answer=None):
    """Chat-template text; with ``answer`` the full training text, without it
    the generation prompt (assistant header appended)."""
    return tokenizer.apply_chat_template(
        memory_messages(question, slots, placeholder, answer),
        tokenize=False,
        add_generation_prompt=answer is None,
    )


def build_decoder_example(
    tokenizer,
    question,
    answer,
    slots,
    placeholder_id,
    eos_ids,
    max_answer_tokens,
    max_tokens,
):
    """-> ``{"input_ids", "labels", "memory_positions"}`` (Python lists) or
    None when the prompt alone exceeds ``max_tokens``.

    ``labels`` is IGNORE_INDEX on the prompt and ``input_ids`` on the answer
    tokens up to and including the first EOS id (``<turn|>``); the template's
    trailing newline after it is dropped. The answer region is cut to
    ``max_answer_tokens`` (no EOS survives a cut)."""
    placeholder = tokenizer.decode([placeholder_id])
    prompt = tokenizer.encode(
        render_prompt(tokenizer, question, slots, placeholder), add_special_tokens=False
    )
    full = tokenizer.encode(
        render_prompt(tokenizer, question, slots, placeholder, answer),
        add_special_tokens=False,
    )
    if full[: len(prompt)] != prompt or len(full) <= len(prompt):
        raise ValueError("training text does not extend the generation prompt")
    positions = [i for i, t in enumerate(prompt) if t == placeholder_id]
    if len(positions) != slots or any(t == placeholder_id for t in full[len(prompt) :]):
        raise ValueError("prompt does not hold exactly the memory placeholders")
    if len(prompt) >= max_tokens:
        return None
    region = full[len(prompt) :]
    end = next((i + 1 for i, t in enumerate(region) if t in eos_ids), len(region))
    region = region[: min(end, max_answer_tokens, max_tokens - len(prompt))]
    ids = prompt + region
    labels = [IGNORE_INDEX] * len(prompt) + region
    return {"input_ids": ids, "labels": labels, "memory_positions": positions}


def collate_decoder(examples, pad_id):
    """Right-pad decoder examples -> ``(input_ids, attention_mask, labels,
    memory_mask)`` tensors [B, L]."""
    if not examples:
        raise ValueError("empty decoder batch")
    longest = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), longest), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(examples), longest), dtype=torch.long)
    labels = torch.full((len(examples), longest), IGNORE_INDEX, dtype=torch.long)
    memory = torch.zeros((len(examples), longest), dtype=torch.bool)
    for index, example in enumerate(examples):
        n = len(example["input_ids"])
        ids[index, :n] = torch.tensor(example["input_ids"])
        mask[index, :n] = 1
        labels[index, :n] = torch.tensor(example["labels"])
        memory[index, example["memory_positions"]] = True
    return ids, mask, labels, memory


def find_span(haystack, needle):
    """First start index of ``needle`` inside ``haystack`` (lists), or None."""
    n = len(needle)
    if n == 0 or n > len(haystack):
        return None
    first = needle[0]
    for start in range(len(haystack) - n + 1):
        if haystack[start] == first and haystack[start : start + n] == needle:
            return start
    return None


def answer_span(doc_ids, answer_variants, copy_slots):
    """``(document index, start, length)`` of the first verbatim occurrence of
    one of ``answer_variants`` (token id lists, tried in order) inside the
    per-document id lists, when it fits ``copy_slots``; else None."""
    for needle in answer_variants:
        if not 1 <= len(needle) <= copy_slots:
            continue
        for index, ids in enumerate(doc_ids):
            start = find_span(ids, needle)
            if start is not None:
                return index, start, len(needle)
    return None


def copy_targets(doc_ids, spans, copy_slots, total_tokens):
    """Per case ``spans`` (from ``answer_span``) over the concatenated document
    ids -> ``(match [B, H, T] bool, copyable [B, H] bool)``: slot ``k`` of a
    copyable case points at the span's ``k``-th position; cases without a
    span have gate targets 0 everywhere."""
    batch = len(doc_ids)
    match = torch.zeros((batch, copy_slots, total_tokens), dtype=torch.bool)
    copyable = torch.zeros((batch, copy_slots), dtype=torch.bool)
    for case, (ids, span) in enumerate(zip(doc_ids, spans)):
        if span is None:
            continue
        index, start, length = span
        offset = sum(len(d) for d in ids[:index]) + start
        if length > copy_slots or offset + length > total_tokens:
            raise ValueError("answer span outside the copy slots or sources")
        for k in range(length):
            match[case, k, offset + k] = True
            copyable[case, k] = True
    return match, copyable


def answer_variants(tokenizer, answer):
    """Token id lists to look for: the space-prefixed answer (mid-sentence
    segmentation) then the bare one (sentence-initial)."""
    text = str(answer).strip()
    if not text:
        return []
    spaced = tokenizer.encode(" " + text, add_special_tokens=False)
    bare = tokenizer.encode(text, add_special_tokens=False)
    return [spaced] + ([bare] if bare != spaced else [])


# ----------------------------------------------------------------------------
# decoder injection and loss


def inject_memory(text_model, input_ids, memory_mask, memories, pad_id):
    """-> ``(inputs_embeds [B, L, W], per_layer_inputs [B, L, layers, P])``.

    Token rows come from the scaled ``embed_tokens``; the ``memory_mask``
    positions (exactly ``K`` per row, in order) receive ``memories`` [B, K, W]
    cast to the embedding dtype; the PLE identity rows are looked up with the
    placeholders replaced by ``pad_id`` (the multimodal forward's rule)."""
    if memory_mask.shape != input_ids.shape or memories.ndim != 3:
        raise ValueError("memory injection needs [B, L] ids/mask and [B, K, W] rows")
    counts = memory_mask.sum(dim=1)
    if not bool((counts == memories.shape[1]).all()):
        raise ValueError("every row needs exactly K placeholder positions")
    embeds = text_model.embed_tokens(input_ids)
    if memories.shape[2] != embeds.shape[2]:
        raise ValueError("memory width differs from the decoder embedding width")
    embeds = embeds.masked_scatter(
        memory_mask.unsqueeze(-1).expand_as(embeds), memories.to(embeds.dtype)
    )
    identity_ids = input_ids.masked_fill(memory_mask, int(pad_id))
    per_layer = text_model.get_per_layer_inputs(identity_ids)
    return embeds, per_layer


def answer_cross_entropy(hidden, labels, lm_head, softcap):
    """Mean token cross-entropy at the labeled positions only -> ``(loss,
    count)``. ``hidden[t]`` predicts ``labels[t + 1]``; the logits are
    computed for those rows alone (FP32 after the final softcap)."""
    if hidden.ndim != 3 or labels.shape != hidden.shape[:2]:
        raise ValueError("loss needs hidden [B, L, W] and labels [B, L]")
    targets = labels[:, 1:]
    keep = targets != IGNORE_INDEX
    count = int(keep.sum())
    if count == 0:
        raise ValueError("batch has no answer tokens")
    rows = hidden[:, :-1][keep]
    with torch.autocast(hidden.device.type, enabled=False):
        logits = lm_head(rows.to(lm_head.weight.dtype)).float()
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        loss = F.cross_entropy(logits, targets[keep])
    return loss, count


def lr_scale(step, total, warmup, floor=0.1):
    """Linear warmup over ``warmup`` steps (clamped to a tenth of short runs)
    then cosine decay to ``floor`` of the peak at the last step."""
    if not 0 <= step < total:
        raise ValueError("step outside the schedule")
    ramp = min(int(warmup), max(total // 10, 1))
    if step < ramp:
        return (step + 1) / ramp
    progress = (step - ramp) / max(total - ramp, 1)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


# ----------------------------------------------------------------------------
# metrics (SQuAD style)


def normalize_answer(text):
    text = str(text).lower()
    text = "".join(c for c in text if c not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, truth):
    return float(normalize_answer(prediction) == normalize_answer(truth))


def token_f1(prediction, truth):
    p, t = normalize_answer(prediction).split(), normalize_answer(truth).split()
    if not p or not t:
        return float(p == t)
    common = collections.Counter(p) & collections.Counter(t)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(t)
    return 2 * precision * recall / (precision + recall)


def _f_measure(hits, predicted, reference):
    if not hits:
        return 0.0
    precision, recall = hits / predicted, hits / reference
    return 2 * precision * recall / (precision + recall)


def lcs_length(a, b):
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            current[j] = (
                previous[j - 1] + 1 if x == y else max(previous[j], current[j - 1])
            )
        previous = current
    return previous[-1]


def rouge_fallback(prediction, reference):
    """ROUGE-1 / ROUGE-L F-measures over lower-cased ``[a-z0-9]+`` tokens
    without stemming (``qwen_hybrid_eval.rouge_l``'s LCS tokenization, which
    is ``rouge_score``'s tokenizer minus the Porter stemmer)."""
    hyp = WORDS.findall(str(prediction).lower())
    ref = WORDS.findall(str(reference).lower())
    if not hyp or not ref:
        return {"rouge1": 0.0, "rougeL": 0.0}
    unigrams = sum((collections.Counter(hyp) & collections.Counter(ref)).values())
    return {
        "rouge1": _f_measure(unigrams, len(hyp), len(ref)),
        "rougeL": _f_measure(lcs_length(hyp, ref), len(hyp), len(ref)),
    }


def whitespace_token_f1(prediction, reference):
    """Word-overlap F1 on lower-cased whitespace tokens (upstream
    ``memorilla.metrics.token_f1`` for one pair)."""
    predicted, gold = str(prediction).lower().split(), str(reference).lower().split()
    if not predicted or not gold:
        return 0.0
    overlap = sum((collections.Counter(predicted) & collections.Counter(gold)).values())
    return _f_measure(overlap, len(predicted), len(gold))


def _literal_constants(source, names):
    import ast

    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                found[target.id] = ast.literal_eval(node.value)
    missing = set(names) - set(found)
    if missing:
        raise ValueError(f"upstream constants missing: {sorted(missing)}")
    return found


def import_upstream_metrics(source_root):
    """The hash-pinned upstream ``memorilla.metrics`` module, executed under a
    private name. Its imports of ``memorilla.benchmarks`` come from the pinned
    benchmarks.py; ``memorilla.data``/``paths`` are replaced by their literal
    constants and ``memorilla.embeddings``/``utils`` by stubs that refuse to
    run (choice scoring and file writing are not used here). Raises
    ``ImportError`` when the source tree or a dependency (rouge_score, nltk,
    numpy, datasets) is missing and ``ValueError`` when a source differs from
    its pin."""
    from types import ModuleType

    root = Path(source_root) / "memorilla"
    sources = {}
    for name, pinned in UPSTREAM_METRIC_SOURCES.items():
        path = root / name
        if not path.exists():
            raise ImportError(f"upstream {path} is missing")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != pinned:
            raise ValueError(f"upstream {name} differs from the pinned source")
        sources[name] = data

    def not_used(*_args, **_kwargs):
        raise RuntimeError("disabled in the pinned metrics import")

    stubs = {"memorilla": ModuleType("memorilla")}
    stubs["memorilla"].__path__ = []
    benchmarks = ModuleType("memorilla.benchmarks")
    data = ModuleType("memorilla.data")
    data.__dict__.update(
        _literal_constants(
            sources["data.py"].decode(),
            ("QUESTION_COLUMN", "ANSWER_COLUMN", "CHOICES_COLUMN", "HARD_COLUMN"),
        )
    )
    paths = ModuleType("memorilla.paths")
    paths.__dict__.update(
        _literal_constants(sources["paths.py"].decode(), ("DEFAULT_ENCODER",))
    )
    embeddings = ModuleType("memorilla.embeddings")
    embeddings.embed_texts = embeddings.load_encoder = not_used
    utils = ModuleType("memorilla.utils")
    utils.write_json = utils.write_jsonl = not_used
    for module in (benchmarks, data, paths, embeddings, utils):
        stubs[module.__name__] = module
    name = "_antfly_memorilla_pinned_metrics"
    metrics = ModuleType(name)
    metrics.__file__ = str(root / "metrics.py")
    previous = {k: sys.modules.get(k) for k in [*stubs, name]}
    sys.modules.update(stubs)
    sys.modules[name] = metrics
    try:
        for module, file in ((benchmarks, "benchmarks.py"), (metrics, "metrics.py")):
            code = compile(sources[file], str(root / file), "exec")
            exec(code, module.__dict__)  # noqa: S102
        return metrics
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


class EvalMetrics:
    """Per-row and corpus metrics. With the pinned upstream
    ``memorilla.metrics`` (paper parity): ``clean_generation`` on the
    prediction, ``exact_match``, ``token_f1`` (lower-cased whitespace tokens),
    ``rouge`` (ROUGE-1/L F-measure, Porter stemming) per row and corpus
    ``bleu`` (BLEU-1/4); otherwise the local fallback (same EM and token F1,
    ROUGE without stemming, no BLEU). SQuAD-normalized F1 (``squad_f1``) is
    reported in both cases. ``name`` records the implementation."""

    def __init__(self, upstream_source):
        self.module, self.reason = None, None
        try:
            self.module = import_upstream_metrics(upstream_source)
        except ImportError as exc:
            self.reason = f"{type(exc).__name__}: {exc}"
        if self.module is not None:
            versions = {
                p: importlib.metadata.version(p) for p in ("rouge-score", "nltk")
            }
            self.name = (
                f"upstream memorilla.metrics @ {UPSTREAM_REVISION[:12]}"
                f" (metrics.py sha256 {UPSTREAM_METRIC_SOURCES['metrics.py'][:12]};"
                f" rouge-score {versions['rouge-score']}, nltk {versions['nltk']};"
                " ROUGE with Porter stemming)"
            )
        else:
            self.name = (
                "local fallback: SQuAD EM, whitespace token F1,"
                " lcs-lowercase-alnum-no-stemmer-v1 ROUGE, no BLEU"
                f" (upstream metrics unavailable: {self.reason})"
            )

    def clean(self, text):
        if self.module is not None:
            return self.module.clean_generation(str(text))
        return str(text).strip()

    def row(self, prediction, reference):
        """One cleaned prediction against one stripped reference."""
        reference = str(reference).strip()
        if self.module is not None:
            m = self.module
            rouge = m.rouge([prediction], [reference])
            em = m.exact_match([prediction], [reference])
            f1 = m.token_f1([prediction], [reference])
        else:
            rouge = rouge_fallback(prediction, reference)
            em = exact_match(prediction, reference)
            f1 = whitespace_token_f1(prediction, reference)
        return {
            "em": float(em),
            "f1": round(float(f1), 4),
            "squad_f1": round(token_f1(prediction, reference), 4),
            "rouge1": round(float(rouge["rouge1"]), 4),
            "rougeL": round(float(rouge["rougeL"]), 4),
        }

    def corpus(self, predictions, references):
        if self.module is None or not predictions:
            return {}
        scores = self.module.bleu(
            list(predictions), [str(r).strip() for r in references]
        )
        return {k: round(float(v), 4) for k, v in scores.items()}


def question_type(question):
    """The question's first word when it is one of ``QUESTION_TYPES`` (the
    paper's NarrativeQA breakdown), else ``other``."""
    match = re.match(r"[^A-Za-z]*([A-Za-z]+)", str(question))
    word = match.group(1).lower() if match else ""
    return word if word in QUESTION_TYPES else "other"


def summarize_by_type(rows, metrics=("rougeL", "f1")):
    """Per question type (fixed order, ``other`` last) -> count and means."""
    report = {}
    for kind in (*QUESTION_TYPES, "other"):
        chosen = [r for r in rows if r["question_type"] == kind]
        if chosen:
            report[kind] = {
                "examples": len(chosen),
                **{
                    m: round(sum(r[m] for r in chosen) / len(chosen), 4)
                    for m in metrics
                },
            }
    return report


def eval_split_name(root, config, requested):
    """``auto``: NarrativeQA's official test split when complete (the paper's
    protocol), else and for every other config the validation split."""
    if requested != "auto":
        return requested
    if config == "narrativeqa":
        files = split_files(root, config, "test")
        if files and split_complete(files):
            return "test"
    return "validation"


# ----------------------------------------------------------------------------
# models and encoding


def decoder_parts(decoder):
    """``(text_model, lm_head, softcap, pad_id)`` of a loaded
    ``Gemma4ForConditionalGeneration``, checking the scaled embedding."""
    text = decoder.model.language_model
    embedding = decoder.get_input_embeddings()
    if embedding is not text.embed_tokens or not hasattr(embedding, "embed_scale"):
        raise ValueError("decoder input embeddings are not the scaled text table")
    if not getattr(text, "hidden_size_per_layer_input", 0):
        raise ValueError("decoder has no per-layer inputs; injection rule untested")
    config = decoder.config.get_text_config()
    pad_id = config.pad_token_id
    if pad_id is None:
        raise ValueError("decoder text config has no pad token id")
    return text, decoder.lm_head, config.final_logit_softcapping, int(pad_id)


def check_placeholder(tokenizer, decoder, placeholder_id):
    """The placeholder must be the configured image soft token, a single
    token under the tokenizer and never produced by the chat framing."""
    if int(decoder.config.image_token_id) != placeholder_id:
        raise ValueError("placeholder id is not the decoder's image soft token")
    text = tokenizer.decode([placeholder_id])
    if tokenizer.encode(text * 3, add_special_tokens=False) != [placeholder_id] * 3:
        raise ValueError("placeholder text does not round-trip to single tokens")
    return text


class StreamTypeEmbedding(nn.Module):
    """Learned key-type rows added before the reader: row 0 to pooled chunk
    rows, row 1 to token rows. Zero init (a Stage 1 reader starts unchanged),
    FP32."""

    def __init__(self, width=ENCODER_WIDTH):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, width, dtype=torch.float32))

    def forward(self, types):
        return F.embedding(types, self.weight)


def set_reader_dropout(reader, p):
    """Set every dropout of the upstream reader (feed-forward ``Dropout``,
    the residual ``attn_dropout`` and the attention weights) to ``p``; it
    acts in train mode only."""
    for module in reader.modules():
        if isinstance(module, nn.Dropout):
            module.p = float(p)
        if isinstance(module, nn.MultiheadAttention):
            module.dropout = float(p)
    return reader


def documents_digest(documents):
    """sha256 over length-prefixed UTF-8 documents (guards a cached entry)."""
    hasher = hashlib.sha256()
    for text in documents:
        data = str(text).encode("utf-8")
        hasher.update(len(data).to_bytes(8, "little"))
        hasher.update(data)
    return hasher.hexdigest()


def safe_name(value):
    """A file name for a collection id: itself when plain, else its sha256."""
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", text) and text not in (".", ".."):
        return text
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FeatureCache:
    """Per-collection pooled chunk rows and chunk ids for evaluation (dual
    streams). Many questions share one collection (a NarrativeQA book), so
    the pooled pass over its chunks runs once: an in-memory LRU within a run
    and, with ``directory``, one safetensors file per collection across runs
    under ``directory/<key>/<config>/<split>/<collection_id>.safetensors``.
    ``key`` digests ``key_fields`` (chunking rule, pooled-row recipe, encoder
    weights and tokenizer digests); each entry also carries the documents
    digest and is ignored (``stale``) when the row's documents differ. Pooled
    rows are stored as BF16, which is lossless after the feature channel.
    Token rows of the top chunks are not cached (recomputed per question)."""

    def __init__(self, key_fields=None, directory=None, max_entries=512):
        self.key_fields = dict(key_fields or {})
        self.key = hashlib.sha256(
            json.dumps(self.key_fields, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        self.directory = None
        if directory is not None:
            if not key_fields:
                raise ValueError("a disk feature cache needs its key fields")
            self.directory = Path(directory) / self.key
            self.directory.mkdir(parents=True, exist_ok=True)
            manifest = self.directory / "key.json"
            if not manifest.exists():
                save_json(manifest, {"schema": FEATURE_CACHE_SCHEMA, **self.key_fields})
        self.memory = collections.OrderedDict()
        self.max_entries = max(1, int(max_entries))
        self.stats = collections.Counter(
            memory_hits=0, disk_hits=0, misses=0, writes=0, stale=0
        )

    def path(self, row):
        return (
            self.directory
            / safe_name(row["config"])
            / safe_name(row["split"])
            / f"{safe_name(row['collection_id'])}.safetensors"
        )

    @staticmethod
    def entry_key(row):
        return (row["config"], row["split"], str(row["collection_id"]))

    def remember(self, key, entry):
        self.memory[key] = entry
        self.memory.move_to_end(key)
        while len(self.memory) > self.max_entries:
            self.memory.popitem(last=False)

    def get(self, row, digest):
        """-> ``{"chunks", "pooled" [N, W] BF16 CPU, "document_tokens"}`` or
        None (a miss)."""
        key = self.entry_key(row)
        entry = self.memory.get(key)
        if entry is not None and entry["digest"] == digest:
            self.memory.move_to_end(key)
            self.stats["memory_hits"] += 1
            return entry
        if self.directory is not None and self.path(row).exists():
            from safetensors import safe_open

            with safe_open(str(self.path(row)), "pt") as handle:
                metadata = handle.metadata() or {}
                if (
                    metadata.get("documents_sha256") == digest
                    and metadata.get("key") == self.key
                ):
                    ids = handle.get_tensor("ids").tolist()
                    offsets = handle.get_tensor("offsets").tolist()
                    entry = {
                        "digest": digest,
                        "chunks": [ids[a:b] for a, b in itertools.pairwise(offsets)],
                        "pooled": handle.get_tensor("pooled"),
                        "document_tokens": int(metadata["document_tokens"]),
                    }
                    self.remember(key, entry)
                    self.stats["disk_hits"] += 1
                    return entry
            self.stats["stale"] += 1
        self.stats["misses"] += 1
        return None

    def put(self, row, digest, chunks, pooled, document_tokens):
        """Store one collection's chunk ids and pooled rows [N, W]."""
        if pooled.shape[0] != len(chunks):
            raise ValueError("one pooled row per chunk")
        entry = {
            "digest": digest,
            "chunks": [list(c) for c in chunks],
            "pooled": pooled.detach().to("cpu", torch.bfloat16).contiguous(),
            "document_tokens": int(document_tokens),
        }
        self.remember(self.entry_key(row), entry)
        if self.directory is None:
            return entry
        from safetensors.torch import save_file

        offsets = [0]
        for c in chunks:
            offsets.append(offsets[-1] + len(c))
        path = self.path(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        save_file(
            {
                "pooled": entry["pooled"],
                "ids": torch.tensor([t for c in chunks for t in c], dtype=torch.int64),
                "offsets": torch.tensor(offsets, dtype=torch.int64),
            },
            str(partial),
            metadata={
                "schema": FEATURE_CACHE_SCHEMA,
                "key": self.key,
                "config": str(row["config"]),
                "split": str(row["split"]),
                "collection_id": str(row["collection_id"]),
                "documents_sha256": digest,
                "document_tokens": str(int(document_tokens)),
            },
        )
        partial.replace(path)
        self.stats["writes"] += 1
        return entry


def weights_digest(directory):
    """sha256 of every ``*.safetensors`` file of a model directory."""
    files = sorted(Path(directory).glob("*.safetensors"))
    if not files:
        raise ValueError(f"no safetensors weights in {directory}")
    result = {}
    for path in files:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 24), b""):
                hasher.update(block)
        result[path.name] = hasher.hexdigest()
    return result


def feature_cache_key(args, encoder_dir):
    """Fields that decide a cached collection's chunk ids and pooled rows."""
    sentences = args.chunking == "sentence128"
    return {
        "chunking": args.chunking,
        "chunk_rule": CHUNK_RULES[args.chunking],
        "chunk_tokens": CHUNK_TOKENS if sentences else int(args.doc_tokens),
        "sentence_rule": SENTENCE_RULE if sentences else None,
        "document_space_prefix": True,
        "pooled_row_recipe": POOLED_ROW_RECIPE,
        "encoder_weights_sha256": weights_digest(encoder_dir),
        "encoder_tokenizer_sha256": digest(Path(encoder_dir) / "tokenizer.json"),
    }


class SoftCaseEncoder:
    """Live encoding of Memorilla rows: documents (space-prefixed, truncated
    or sentence-chunked, selected, row-capped) and the question through the
    encoder taps, reader features via ``assemble_features`` and copy sources
    from the scaled decoder embedding rows; with ``reader_streams="dual"``
    pooled rows for every chunk plus token rows of the top chunks."""

    def __init__(
        self,
        models,
        doc_tokens,
        max_docs,
        max_rows,
        chunk=64,
        chunking="none",
        reader_streams="tokens",
        max_chunks=512,
        token_chunks=8,
        count_document_tokens=False,
    ):
        self.models = models
        self.doc_tokens, self.max_docs, self.max_rows = doc_tokens, max_docs, max_rows
        self.chunk = chunk
        if chunking not in CHUNKING_CHOICES or reader_streams not in STREAM_CHOICES:
            raise ValueError("unknown chunking or reader stream mode")
        self.chunking, self.reader_streams = chunking, reader_streams
        self.max_chunks, self.token_chunks = max_chunks, token_chunks
        self.count_document_tokens = count_document_tokens
        self.chunks_dropped, self.chunk_calls = 0, 0
        self.cache = None  # FeatureCache (evaluation, dual streams only)
        tokenizer = models["tokenizer_e"]
        self.bos, self.eos = int(tokenizer.bos_token_id), int(tokenizer.eos_token_id)
        self.pad = int(tokenizer.pad_token_id)
        probe = tokenizer.encode(" probe text")
        if (
            probe[0] != self.bos
            or probe[-1] != self.eos
            or models["specials"] != (1, 1)
        ):
            raise ValueError("encoder tokenizer does not wrap <bos> ids <eos>")
        self.selected, self.selection_calls = 0, 0

    def tokenize(self, text, limit):
        ids = self.models["tokenizer_e"].encode(text, add_special_tokens=False)
        return ids[: int(limit)] if ids else [self.pad]

    def encoder_batch(self, sequences):
        longest = max(len(s) for s in sequences) + 2
        ids = torch.full((len(sequences), longest), self.pad, dtype=torch.long)
        mask = torch.zeros((len(sequences), longest), dtype=torch.long)
        for index, seq in enumerate(sequences):
            row = [self.bos, *seq, self.eos]
            ids[index, : len(row)] = torch.tensor(row)
            mask[index, : len(row)] = 1
        device = next(self.models["text_model"].parameters()).device
        return {"input_ids": ids.to(device), "attention_mask": mask.to(device)}

    def run_taps(self, sequences, grad):
        context = contextlib.nullcontext() if grad else torch.no_grad()
        tokens = self.encoder_batch(sequences)
        with context, torch.autocast("cuda", dtype=torch.bfloat16):
            final, tap = self.models["taps"].run(tokens)
        return final, tap, tokens["attention_mask"]

    def choose(self, doc_ids, question_ids):
        """Top ``max_docs`` documents by pooled cosine to the question."""
        self.selection_calls += 1
        vectors = []
        with torch.no_grad():
            final, _, mask = self.run_taps([question_ids], False)
            question = pooled_query(final, mask)[0]
            for start in range(0, len(doc_ids), self.chunk):
                piece = doc_ids[start : start + self.chunk]
                final, _, mask = self.run_taps(piece, False)
                vectors.extend(
                    pooled_query(final[i : i + 1], mask[i : i + 1])[0]
                    for i in range(len(piece))
                )
        chosen = select_documents(torch.stack(vectors), question, self.max_docs)
        self.selected += len(doc_ids) - len(chosen)
        return [doc_ids[i] for i in chosen]

    def encode_many(self, texts):
        return encode_texts(self.models["tokenizer_e"], texts)

    def document_chunks(self, documents):
        """Id lists of the (space-prefixed) documents: one truncated list per
        document, or the sentence-greedy chunks with ``sentence128``."""
        if self.chunking == "sentence128":
            return chunk_documents(documents, self.encode_many, CHUNK_TOKENS)
        return [self.tokenize(d, self.doc_tokens) for d in documents]

    def document_tokens(self, documents, force=False):
        """Untruncated encoder tokens of the whole collection (reported only)."""
        if not (self.count_document_tokens or force):
            return None
        return sum(len(ids) for ids in self.encode_many(documents))

    def prepare(self, row):
        """Document id lists (selected, capped), question ids and statistics
        of one row."""
        documents = prefixed_documents([str(d) for d in row["documents"]], True)
        doc_ids = self.document_chunks(documents)
        question_ids = self.tokenize(str(row["question"]).strip(), self.doc_tokens)
        chunks = len(doc_ids)
        if len(doc_ids) > self.max_docs:
            doc_ids = self.choose(doc_ids, question_ids)
        kept = fit_documents(doc_ids, self.max_rows)
        stats = {
            "chunks": chunks,
            "pooled_rows": 0,
            "token_chunks": len(kept),
            "token_rows": 2 * sum(len(ids) + 1 for ids in kept),
            "read_tokens": sum(len(ids) for ids in kept),
            "document_tokens": self.document_tokens(documents),
        }
        return kept, question_ids, stats

    def encode(self, rows, grad):
        """-> dict with ``docs [B, M, W]``, ``mask [B, M]``, ``queries [B, W]``,
        ``sources [B, T, W]``, ``source_ids [B, T]``, ``source_mask [B, T]``,
        ``doc_ids`` (per case list of id lists, the copy sources' order),
        ``types`` ([B, M] stream index, dual streams only) and ``stats``."""
        if self.reader_streams == "dual":
            return self.encode_dual(rows, grad)
        prepared = [self.prepare(row) for row in rows]
        sequences = [s for docs, q, _ in prepared for s in [*docs, q]]
        final, tap, mask = self.run_taps(sequences, grad)
        docs, queries, sources, offset = [], [], [], 0
        embedding = self.models["embedding"]
        for doc_ids, _, _ in prepared:
            count = len(doc_ids) + 1
            d, q = assemble_features(
                final[offset : offset + count],
                tap[offset : offset + count],
                mask[offset : offset + count],
                [len(i) for i in doc_ids],
                1,
                1,
            )
            docs.append(d)
            queries.append(q)
            ids = torch.tensor([t for doc in doc_ids for t in doc], dtype=torch.long)
            ids = ids.to(embedding.weight.device)
            sources.append((token_ids_embeddings(embedding, ids), ids))
            offset += count
        batch, pad_mask = pad_document_batch(docs)
        source_rows, source_ids, source_mask = pad_token_embeddings(sources)
        return {
            "docs": batch,
            "mask": pad_mask,
            "queries": torch.cat(queries, dim=0),
            "sources": source_rows,
            "source_ids": source_ids,
            "source_mask": source_mask,
            "doc_ids": [docs for docs, _, _ in prepared],
            "types": None,
            "stats": [stats for _, _, stats in prepared],
        }

    def encode_dual(self, rows, grad):
        """Dual streams: pooled rows of every chunk plus the token rows of the
        top ``token_chunks`` chunks per case by pooled cosine to the question.

        Without a cache (training) one frozen encoder pass over all chunks
        (``chunk``-sized batches, longest first) yields each pooled row and
        keeps the running top chunks' token states (BOS dropped, tokens +
        EOS). With ``self.cache`` (evaluation) a cached collection supplies its
        chunk ids and pooled rows, an uncached one gets the pooled pass and is
        cached, and for every case the scores come from its pooled rows and
        the top chunks are re-encoded for token rows, so a question gets the
        same features whether its collection was cached or not."""
        if grad:
            raise ValueError("dual streams read the frozen encoder (no encoder LoRA)")
        recompute = self.cache is not None
        prepared = []
        for row in rows:
            documents = prefixed_documents([str(d) for d in row["documents"]], True)
            question_ids = self.tokenize(str(row["question"]).strip(), self.doc_tokens)
            digest_, entry = None, None
            if self.cache is not None and row.get("collection_id") is not None:
                digest_ = documents_digest(documents)
                entry = self.cache.get(row, digest_)
            if entry is not None:
                document_tokens = (
                    entry["document_tokens"] if self.count_document_tokens else None
                )
                prepared.append(
                    (
                        entry["chunks"],
                        question_ids,
                        document_tokens,
                        entry["pooled"],
                        digest_,
                    )
                )
                continue
            chunks = [c for c in self.document_chunks(documents) if c] or [[self.pad]]
            document_tokens = self.document_tokens(documents, force=digest_ is not None)
            prepared.append((chunks, question_ids, document_tokens, None, digest_))
        items = sorted(
            (
                (case, index)
                for case, (chunks, _, _, cached, _) in enumerate(prepared)
                if cached is None
                for index in range(len(chunks))
            ),
            key=lambda item: -len(prepared[item[0]][0][item[1]]),
        )
        with torch.no_grad():
            final, _, mask = self.run_taps([p[1] for p in prepared], False)
            queries = torch.cat(
                [
                    pooled_query(final[i : i + 1], mask[i : i + 1])
                    for i in range(len(rows))
                ]
            )
            del final, mask
            pooled, scores, keepers = [], [], [[] for _ in prepared]
            for case, (chunks, _, _, cached, _) in enumerate(prepared):
                if cached is None:
                    pooled.append([None] * len(chunks))
                    scores.append([0.0] * len(chunks))
                    continue
                vectors = cached.to(queries.device, torch.float32)
                pooled.append(vectors)
                scores.append(chunk_cosine(vectors, queries[case]).tolist())
            for start in range(0, len(items), self.chunk):
                piece = items[start : start + self.chunk]
                final, tap, mask = self.run_taps(
                    [prepared[case][0][index] for case, index in piece], False
                )
                vectors = pooled_rows(final, mask)
                cosine = chunk_cosine(vectors, queries[[case for case, _ in piece]])
                actives = mask.sum(dim=1).tolist()
                for k, ((case, index), score) in enumerate(zip(piece, cosine.tolist())):
                    pooled[case][index], scores[case][index] = vectors[k], score
                    if recompute:
                        continue  # token rows: re-encoded after the ranking
                    kept = keepers[case]
                    if len(kept) >= self.token_chunks and (-score, index) >= (
                        -kept[-1][0],
                        kept[-1][1],
                    ):
                        continue  # cannot enter the top token chunks
                    active = int(actives[k])
                    rows_k = (final[k, 1:active].clone(), tap[k, 1:active].clone())
                    keepers[case] = keep_top(
                        [*kept, (score, index, rows_k)], self.token_chunks
                    )
                del final, tap, mask, vectors
            for case, (chunks, _, document_tokens, cached, digest_) in enumerate(
                prepared
            ):
                if cached is None:
                    pooled[case] = torch.stack(pooled[case])
                    if digest_ is not None:  # document_tokens forced on misses
                        self.cache.put(
                            rows[case], digest_, chunks, pooled[case], document_tokens
                        )
                if recompute:
                    scores[case] = chunk_cosine(pooled[case], queries[case]).tolist()
            # With a cache: re-encode only each case's top chunks.
            again = [
                (case, index, scores[case][index])
                for case in range(len(prepared) if recompute else 0)
                for index in rank_order(scores[case])[: self.token_chunks]
            ]
            for start in range(0, len(again), self.chunk):
                piece = again[start : start + self.chunk]
                final, tap, mask = self.run_taps(
                    [prepared[case][0][index] for case, index, _ in piece], False
                )
                actives = mask.sum(dim=1).tolist()
                for k, (case, index, score) in enumerate(piece):
                    active = int(actives[k])
                    keepers[case].append(
                        (
                            score,
                            index,
                            (final[k, 1:active].clone(), tap[k, 1:active].clone()),
                        )
                    )
                del final, tap, mask
        docs, types, sources, doc_ids, stats = [], [], [], [], []
        embedding = self.models["embedding"]
        for case, (chunks, _, document_tokens, _, _) in enumerate(prepared):
            kept, dropped = cap_chunks(scores[case], self.max_chunks)
            self.chunks_dropped += dropped
            self.chunk_calls += 1
            ranked = keep_top(keepers[case], self.token_chunks)
            fitted = fit_documents(
                [chunks[index] for _, index, _ in ranked], self.max_rows
            )
            final_rows, tap_rows = [], []
            for ids, (_, _, (f, t)) in zip(fitted, ranked):
                final_rows.append(torch.cat([f[: len(ids)], f[-1:]]))
                tap_rows.append(torch.cat([t[: len(ids)], t[-1:]]))
            token = torch.cat(
                [
                    unit_feature_rows(torch.cat(final_rows)),
                    unit_feature_rows(torch.cat(tap_rows)),
                ]
            )
            vectors = pooled[case][kept]
            docs.append(torch.cat([vectors, token]).unsqueeze(0))
            types.append([0] * len(kept) + [1] * token.shape[0])
            ids = torch.tensor([t for doc in fitted for t in doc], dtype=torch.long)
            ids = ids.to(embedding.weight.device)
            sources.append((token_ids_embeddings(embedding, ids), ids))
            doc_ids.append(fitted)
            stats.append(
                {
                    "chunks": len(chunks),
                    "pooled_rows": len(kept),
                    "chunks_dropped": dropped,
                    "token_chunks": len(fitted),
                    "token_rows": int(token.shape[0]),
                    "read_tokens": sum(len(c) for c in chunks),
                    "document_tokens": document_tokens,
                }
            )
        batch, pad_mask = pad_document_batch(docs)
        type_ids = torch.zeros(pad_mask.shape, dtype=torch.long)
        for case, row_types in enumerate(types):
            type_ids[case, : len(row_types)] = torch.tensor(row_types)
        source_rows, source_ids, source_mask = pad_token_embeddings(sources)
        return {
            "docs": batch,
            "mask": pad_mask,
            "queries": queries,
            "sources": source_rows,
            "source_ids": source_ids,
            "source_mask": source_mask,
            "doc_ids": doc_ids,
            "types": type_ids.to(batch.device),
            "stats": stats,
        }


def slot_layout(slots, copy_slots):
    if slots not in MEMORY_SLOT_CHOICES:
        raise ValueError("unsupported memory slot count")
    if not 0 <= copy_slots < slots:
        raise ValueError("copy slots must leave at least one soft slot")
    return {
        "soft": list(range(slots - copy_slots)),
        "copy": list(range(slots - copy_slots, slots)),
    }


def memory_rows_train(memories, copy_head, sources, source_mask, copy_slots):
    """Training rows: soft slots as read, copy slots gate-mixed through the
    head -> ``(rows [B, K, W] FP32, copy_logits [B, H, T], gate_logits [B, H])``."""
    soft = memories[:, : memories.shape[1] - copy_slots].float()
    if copy_slots == 0:
        return soft, None, None
    output, copy_logits, gate_logits = copy_head.forward_detailed(
        memories[:, memories.shape[1] - copy_slots :].float(), sources, source_mask
    )
    return torch.cat([soft, output], dim=1), copy_logits, gate_logits


def memory_rows_infer(
    memories, copy_head, sources, source_ids, snapper, embedding, copy_slots
):
    """Inference rows for one case: soft slots as read; copy slots inside a
    contiguous run (gate >= 0.5) snap to their nearest decoder token (scaled
    embedding row), the others pass as the head's soft output."""
    soft = memories[: memories.shape[0] - copy_slots].float()
    if copy_slots == 0:
        return soft, {"spans": [], "gate": [], "snapped_ids": [], "snapped_text": ""}
    with torch.autocast(memories.device.type, enabled=False):
        output, gate, moved, spans = copy_with_contiguous_runs(
            copy_head,
            memories[memories.shape[0] - copy_slots :].float(),
            sources.float(),
            return_spans=True,
        )
        _, ids, top1 = snapper(output)
    hard = output.clone()
    snapped = []
    for begin, end, start in spans:
        for slot in range(begin, end):
            hard[slot] = embedding(ids[slot : slot + 1])[0].float()
            snapped.append(
                (slot, int(ids[slot]), int(source_ids[start + slot - begin]))
            )
    return torch.cat([soft, hard], dim=0), {
        "spans": [list(map(int, s)) for s in spans],
        "gate": [round(float(g), 3) for g in gate.tolist()],
        "moved": int(moved),
        "snapped_ids": [i for _, i, _ in snapped],
        "snap_cosine_min": round(float(top1.min()), 4) if spans else None,
    }


def decoder_hidden(
    decoder, text_model, input_ids, attention_mask, memory_mask, rows, pad_id
):
    embeds, per_layer = inject_memory(text_model, input_ids, memory_mask, rows, pad_id)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = decoder.model(
            inputs_embeds=embeds,
            per_layer_inputs=per_layer,
            attention_mask=attention_mask,
            use_cache=False,
        )
    return out.last_hidden_state


@torch.no_grad()
def generate_with_memory(
    decoder, text_model, input_ids, memory_mask, rows, pad_id, eos_ids, max_new_tokens
):
    """Greedy continuation of one injected prompt [1, L] with a KV cache."""
    embeds, per_layer = inject_memory(text_model, input_ids, memory_mask, rows, pad_id)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = decoder(inputs_embeds=embeds, per_layer_inputs=per_layer, use_cache=True)
    cache, new_ids = out.past_key_values, []
    next_id = int(out.logits[0, -1].float().argmax())
    for _ in range(max_new_tokens):
        new_ids.append(next_id)
        if next_id in eos_ids:
            break
        step = torch.tensor([[next_id]], device=input_ids.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = decoder(input_ids=step, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_id = int(out.logits[0, -1].float().argmax())
    return new_ids


# ----------------------------------------------------------------------------
# training


def recipe_parts(args):
    if args.recipe == "depot":
        raise SystemExit(
            "recipe 'depot' (Stage-3 analog on the depot corpus via gemma_memory's"
            " loader) is not implemented yet; use stage1, stage2 or --datasets"
        )
    if args.datasets:
        return parse_datasets(args.datasets)
    if args.recipe is None:
        raise SystemExit(
            "choose --recipe stage1|stage2|stage2-replay or pass --datasets"
        )
    caps = RECIPE_CAPS.get(args.recipe, {})
    return [(name, caps.get(name)) for name in RECIPES[args.recipe]]


def resolve_options(args):
    """Fill the paper options left unset (None) on the command line: the
    recipe preset (``stage2-replay``) else the legacy values."""
    preset = RECIPE_PRESETS.get(args.recipe, {})
    for key, legacy in LEGACY_OPTIONS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, preset.get(key, legacy))
    for key, default in OPTION_DEFAULTS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, default)
    if args.chunking not in CHUNKING_CHOICES or args.reader_streams not in (
        STREAM_CHOICES
    ):
        raise ValueError("unknown chunking or reader stream mode")
    if not 0 <= float(args.reader_dropout) < 1:
        raise ValueError("reader dropout must be within [0, 1)")
    if not 1 <= args.token_chunks <= args.max_chunks or args.encoder_chunk_batch < 1:
        raise ValueError(
            "need 1 <= --token-chunks <= --max-chunks and a positive chunk batch"
        )
    return args


def options_active(args):
    """Whether any paper option differs from the legacy run (or the recipe is
    a preset); only then does the protocol record them."""
    return args.recipe in RECIPE_PRESETS or any(
        getattr(args, key) != legacy for key, legacy in LEGACY_OPTIONS.items()
    )


def option_protocol(args, parts, split_rows):
    """Protocol entries of the paper options ({} for a legacy run) including
    the training mixture (examples per config and share)."""
    if not options_active(args):
        return {}
    examples = [
        rows if cap is None else min(cap, rows)
        for (_, cap), rows in zip(parts, split_rows)
    ]
    total = max(sum(examples), 1)
    dual = args.reader_streams == "dual"
    sentences = args.chunking == "sentence128"
    return {
        "chunking": args.chunking,
        "chunk_rule": CHUNK_RULES[args.chunking],
        "chunk_tokens": CHUNK_TOKENS if sentences else args.doc_tokens,
        "sentence_rule": SENTENCE_RULE if sentences else None,
        "reader_streams": args.reader_streams,
        "max_chunks": args.max_chunks if dual else None,
        "chunk_cap_rule": CHUNK_CAP_RULE if dual else None,
        "pooled_row_recipe": POOLED_ROW_RECIPE if dual else None,
        "token_chunks": args.token_chunks if dual else None,
        "token_chunk_rule": TOKEN_CHUNK_RULE if dual else None,
        "stream_type": {
            "shape": [2, ENCODER_WIDTH],
            "rows": ["pooled", "token"],
            "init": "zeros",
            "dtype": "float32",
            "file": STREAM_TYPE_FILE,
        }
        if dual
        else None,
        "slot_routing": SLOT_ROUTING,
        "encoder_chunk_batch": args.encoder_chunk_batch,
        "encoder_frozen_for_pooled_rows": dual,
        "reader_dropout": float(args.reader_dropout),
        "mixture": [
            {
                "config": name,
                "cap": cap,
                "examples": count,
                "share": round(count / total, 5),
            }
            for (name, cap), count in zip(parts, examples)
        ],
    }


def candidate_files(train_encoder, dual=False):
    files = [
        ("reader_sha256", "reader.safetensors"),
        ("copy_head_sha256", "copy-head.safetensors"),
        ("protocol_sha256", "protocol.json"),
    ]
    if train_encoder:
        files.insert(1, ("encoder_lora_sha256", "encoder-lora.safetensors"))
    if dual:
        files.insert(-1, ("stream_type_sha256", STREAM_TYPE_FILE))
    return tuple(files)


def write_candidate(
    output,
    reader,
    adapters,
    copy_head,
    slots,
    copy_slots,
    train_encoder,
    stream_type=None,
):
    from safetensors.torch import save_file

    save_file(
        {k: v.detach().cpu().contiguous() for k, v in reader.state_dict().items()},
        str(output / "reader.safetensors"),
    )
    save_file(
        {k: v.contiguous() for k, v in copy_head_state(copy_head).items()},
        str(output / "copy-head.safetensors"),
    )
    if train_encoder:
        save_file(adapter_state(adapters), str(output / "encoder-lora.safetensors"))
    dual = stream_type is not None
    if dual:
        save_file(
            {
                k: v.detach().cpu().contiguous()
                for k, v in stream_type.state_dict().items()
            },
            str(output / STREAM_TYPE_FILE),
        )
    candidate = {
        "schema": SCHEMA,
        "rollout_qualified": False,
        "experimental_only": True,
        "memory_slots": slots,
        "copy_slots": copy_slots,
        "train_encoder": train_encoder,
        **({"reader_streams": "dual"} if dual else {}),
        **{
            field: digest(output / file)
            for field, file in candidate_files(train_encoder, dual)
        },
    }
    save_json(output / "candidate.json", candidate)
    return candidate


def load_candidate(directory):
    directory = Path(directory)
    candidate = json.loads((directory / "candidate.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    if candidate["schema"] != SCHEMA or protocol["schema"] != SCHEMA:
        raise ValueError("candidate schema mismatch")
    for field, file in candidate_files(
        bool(candidate.get("train_encoder")), candidate.get("reader_streams") == "dual"
    ):
        if candidate[field] != digest(directory / file):
            raise ValueError(f"{file} differs from the candidate digest")
    return protocol, candidate


def restore_stream_type(stream_type, state, new_tensors):
    """Load ``state`` into ``stream_type`` or, when the source has none (a
    Stage 1 / token-stream reader), zero it and record it as new."""
    if stream_type is None:
        if state is not None:
            raise ValueError(
                "source carries a stream_type embedding; use --reader-streams dual"
            )
        return
    if state is None:
        with torch.no_grad():
            stream_type.weight.zero_()
        new_tensors.append("stream_type.weight")
        return
    stream_type.load_state_dict(state, strict=True)


def load_trained(directory, reader, copy_head, adapters, stream_type=None):
    """Load reader, copy head and (when present) adapters and stream-type rows
    from a candidate or an interrupted run's ``resume.pt`` -> ``(source,
    new_tensors)``; ``new_tensors`` names the trainables the source did not
    carry (initialized fresh: the zero stream-type rows)."""
    from safetensors.torch import load_file

    directory, new_tensors = Path(directory), []
    if (directory / "candidate.json").exists():
        load_candidate(directory)
        reader.load_state_dict(
            load_file(str(directory / "reader.safetensors")), strict=True
        )
        copy_head.load_state_dict(
            load_file(str(directory / "copy-head.safetensors")), strict=True
        )
        lora = directory / "encoder-lora.safetensors"
        if lora.exists():
            restore_adapters(adapters, load_file(str(lora)))
        types = directory / STREAM_TYPE_FILE
        restore_stream_type(
            stream_type, load_file(str(types)) if types.exists() else None, new_tensors
        )
        return "candidate", new_tensors
    state = torch.load(directory / "resume.pt", map_location="cpu", weights_only=True)
    reader.load_state_dict(state["reader"], strict=True)
    copy_head.load_state_dict(state["copy_head"], strict=True)
    if state.get("adapters"):
        restore_adapters(adapters, state["adapters"])
    restore_stream_type(stream_type, state.get("stream_type"), new_tensors)
    return f"resume.pt@{int(state['step'])}", new_tensors


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, allow_nan=False), flush=True)


class Trainer:
    """Shared state of ``train``, ``eval`` and ``probe``: models, reader, copy
    head, encoder, decoder parts and the per-batch loss."""

    def __init__(
        self,
        args,
        protocol_slots=None,
        protocol_copy=None,
        tap=None,
        train_encoder=False,
    ):
        slots = protocol_slots if protocol_slots is not None else args.memory_slots
        copy_slots = protocol_copy if protocol_copy is not None else args.copy_slots
        self.layout = slot_layout(slots, copy_slots)
        self.slots, self.copy_slots = slots, copy_slots
        self.train_encoder = train_encoder
        targets = lora_targets(getattr(args, "lora_targets", "q_proj,v_proj"))
        tap = tap or getattr(args, "tap_layer", DEFAULT_TAP_LAYER)
        self.models = load_models(
            args.encoder,
            args.decoder,
            tap,
            args.rank,
            args.alpha,
            targets,
            train_encoder,
        )
        self.targets = targets
        decoder = self.models["decoder"]
        self.text, self.lm_head, self.softcap, self.pad_id = decoder_parts(decoder)
        self.placeholder = check_placeholder(
            self.models["tokenizer_d"], decoder, PLACEHOLDER_ID
        )
        self.eos_ids = tuple(self.models["eos_ids"])
        self.reader = (
            build_reader(args.upstream_source, slots, args.seed)
            .cuda()
            .requires_grad_(True)
        )
        self.reader_dropout = float(getattr(args, "reader_dropout", 0.0) or 0.0)
        if self.reader_dropout:
            set_reader_dropout(self.reader, self.reader_dropout)
        self.copy_head = CopyHead(width=self.reader.config.output_dim).cuda()
        self.snapper = MemoryTokenSnapper(self.models["embedding"])
        self.reader_streams = getattr(args, "reader_streams", None) or "tokens"
        self.stream_type = (
            StreamTypeEmbedding().cuda() if self.reader_streams == "dual" else None
        )
        self.encoder = SoftCaseEncoder(
            self.models,
            args.doc_tokens,
            args.max_docs,
            args.max_rows,
            chunk=getattr(args, "encoder_chunk_batch", None)
            or OPTION_DEFAULTS["encoder_chunk_batch"],
            chunking=getattr(args, "chunking", None) or "none",
            reader_streams=self.reader_streams,
            max_chunks=getattr(args, "max_chunks", OPTION_DEFAULTS["max_chunks"]),
            token_chunks=getattr(args, "token_chunks", None)
            or OPTION_DEFAULTS["token_chunks"],
            count_document_tokens=bool(getattr(args, "report_doc_tokens", False)),
        )
        self.noise = None  # CPU generator of the noise-document test (eval)
        self.max_answer_tokens, self.max_tokens = (
            args.max_answer_tokens,
            args.max_tokens,
        )
        decoder.config.use_cache = False

    def decoder_batch(self, rows):
        examples = []
        for row in rows:
            example = build_decoder_example(
                self.models["tokenizer_d"],
                str(row["question"]),
                str(row["answer"]),
                self.slots,
                PLACEHOLDER_ID,
                self.eos_ids,
                self.max_answer_tokens,
                self.max_tokens,
            )
            examples.append(example)
        return examples

    def reader_memories(self, encoded):
        """Reader output for encoded cases: the key rows (noise-replaced in
        the noise test) plus the stream-type rows in dual mode."""
        docs = encoded["docs"]
        if self.noise is not None:
            docs = noise_rows(docs, ~encoded["mask"], self.noise)
        if self.stream_type is not None:
            docs = docs + self.stream_type(encoded["types"])
        return read_memories(
            self.reader, docs, encoded["mask"], encoded["queries"], self.slots
        )

    def copy_sources(self, encoded):
        """Copy-head source rows (noise-replaced in the noise test: they carry
        the documents' tokens too)."""
        if self.noise is None:
            return encoded["sources"]
        return noise_rows(encoded["sources"], encoded["source_mask"], self.noise)

    def losses(self, rows, examples, grad):
        """One micro-batch -> dict of FP32 loss tensors and statistics."""
        encoded = self.encoder.encode(rows, grad and self.train_encoder)
        memories = self.reader_memories(encoded)
        rows_out, copy_logits, gate_logits = memory_rows_train(
            memories,
            self.copy_head,
            self.copy_sources(encoded),
            encoded["source_mask"],
            self.copy_slots,
        )
        input_ids, attention_mask, labels, memory_mask = (
            t.cuda() for t in collate_decoder(examples, self.pad_id)
        )
        hidden = decoder_hidden(
            self.models["decoder"],
            self.text,
            input_ids,
            attention_mask,
            memory_mask,
            rows_out,
            self.pad_id,
        )
        ce, count = answer_cross_entropy(hidden, labels, self.lm_head, self.softcap)
        result = {
            "ce": ce,
            "answer_tokens": count,
            "decoder_tokens": int(attention_mask.sum()),
            "encoder_tokens": int(encoded["source_mask"].sum()),
            "stats": encoded["stats"],
        }
        if self.copy_slots:
            tokenizer = self.models["tokenizer_d"]
            spans = [
                answer_span(
                    ids, answer_variants(tokenizer, row["answer"]), self.copy_slots
                )
                for ids, row in zip(encoded["doc_ids"], rows)
            ]
            match, copyable = copy_targets(
                encoded["doc_ids"],
                spans,
                self.copy_slots,
                encoded["source_ids"].shape[1],
            )
            match, copyable = match.cuda(), copyable.cuda()
            pointer, gate = pointer_losses(
                copy_logits, gate_logits, match, copyable, encoded["source_mask"]
            )
            result.update(
                pointer=pointer,
                gate=gate,
                copyable_cases=sum(s is not None for s in spans),
                gate_mean=float(torch.sigmoid(gate_logits.detach().float()).mean()),
            )
        return result

    def modes(self, training):
        self.models["text_model"].train(training and self.train_encoder)
        self.reader.train(training)  # reader dropout acts in train mode only
        self.copy_head.train(training)
        self.models["decoder"].train(
            training
        )  # gradient checkpointing is active in train mode only; dropout is 0

    @torch.no_grad()
    def infer_rows(self, row):
        encoded = self.encoder.encode([row], False)
        memories = self.reader_memories(encoded)[0]
        rows_out, record = memory_rows_infer(
            memories,
            self.copy_head,
            self.copy_sources(encoded)[0],
            encoded["source_ids"][0],
            self.snapper,
            self.models["embedding"],
            self.copy_slots,
        )
        return rows_out.unsqueeze(0), record, encoded

    def answer(self, row, max_new_tokens=MAX_NEW_TOKENS):
        tokenizer = self.models["tokenizer_d"]
        rows_out, record, encoded = self.infer_rows(row)
        prompt = tokenizer.encode(
            render_prompt(
                tokenizer, str(row["question"]), self.slots, self.placeholder
            ),
            add_special_tokens=False,
        )
        input_ids = torch.tensor([prompt], device="cuda")
        memory_mask = input_ids == PLACEHOLDER_ID
        new_ids = generate_with_memory(
            self.models["decoder"],
            self.text,
            input_ids,
            memory_mask,
            rows_out,
            self.pad_id,
            self.eos_ids,
            max_new_tokens,
        )
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return {
            "answer": text,
            "token_ids": new_ids,
            "finish_reason": "eos" if new_ids[-1] in self.eos_ids else "length",
            "memory_inputs": encoded["stats"][0],
            **record,
        }


def memory_input_summary(stats, compact=False):
    """Means (and the chunk maximum) of per-case encoder statistics
    (``SoftCaseEncoder`` ``stats``); ``compact`` drops document tokens."""
    if not stats:
        return {}
    keys = ["chunks", "pooled_rows", "token_chunks", "token_rows", "read_tokens"]
    summary = {
        f"{key}_mean": round(sum(s[key] for s in stats) / len(stats), 2) for key in keys
    }
    summary["chunks_max"] = max(s["chunks"] for s in stats)
    summary["chunks_dropped"] = sum(s.get("chunks_dropped", 0) for s in stats)
    tokens = [s["document_tokens"] for s in stats if s.get("document_tokens")]
    if tokens and not compact:
        summary["document_tokens_mean"] = round(sum(tokens) / len(tokens), 1)
    return summary


def validation_loss(trainer, splits, batch, limit=VALIDATION_SLICE):
    """Mean answer-token CE over the first ``limit`` rows of each split."""
    trainer.modes(False)
    report = {}
    with torch.no_grad():
        for split in splits:
            total, tokens, used = 0.0, 0, 0
            indices = list(range(min(limit, len(split))))
            for start in range(0, len(indices), batch):
                rows = [split.row(i) for i in indices[start : start + batch]]
                examples = trainer.decoder_batch(rows)
                pairs = [(r, e) for r, e in zip(rows, examples) if e is not None]
                if not pairs:
                    continue
                losses = trainer.losses(
                    [r for r, _ in pairs], [e for _, e in pairs], False
                )
                total += float(losses["ce"]) * losses["answer_tokens"]
                tokens += losses["answer_tokens"]
                used += len(pairs)
            report[split.config] = {
                "loss": round(total / max(tokens, 1), 5),
                "answer_tokens": tokens,
                "examples": used,
            }
    trainer.modes(True)
    return report


def schedule_steps(total_examples, effective, max_steps=None):
    steps = math.ceil(total_examples / effective)
    return steps if max_steps is None else min(steps, max_steps)


def build_protocol(args, parts, split_rows, total_examples, steps, env):
    """The run protocol. ``env`` carries what needs the loaded models or the
    disk (``encoder``/``decoder`` identities, the trainer's ``layout``,
    ``targets``, ``placeholder``, ``pad_id``, ``softcap``, ``eos_ids``, the
    ``script_sha256`` and the library ``versions``). The paper options are
    appended only when active (``option_protocol``), so a legacy run's
    protocol is unchanged byte for byte."""
    effective = args.batch * args.accum
    config = reader_config(args.memory_slots)
    if float(getattr(args, "reader_dropout", 0.0) or 0.0):
        config["dropout"] = float(args.reader_dropout)
    protocol = {
        "schema": SCHEMA,
        "arm": "gemma-soft-memory",
        "recipe": args.recipe,
        "datasets": [
            {"config": n, "cap": c, "rows": rows}
            for (n, c), rows in zip(parts, split_rows)
        ],
        "data_root": str(args.data_root),
        "total_examples": total_examples,
        "steps": steps,
        "epochs": 1,
        "encoder": env["encoder"],
        "decoder": env["decoder"],
        "upstream_memory_sha256": UPSTREAM_MEMORY_SHA256,
        "reader_config": config,
        "reader_init": "fresh" if args.init is None else str(args.init),
        "memory_slots": args.memory_slots,
        "copy_slots": args.copy_slots,
        "slot_layout": env["layout"],
        "tap_layer": args.tap_layer,
        "feature_recipe": FEATURE_RECIPE,
        "query_pooling": QUERY_POOLING,
        "document_space_prefix": True,
        "doc_tokens": args.doc_tokens,
        "max_docs": args.max_docs,
        "max_rows": args.max_rows,
        "document_selection": SELECTION_RULE,
        "max_answer_tokens": args.max_answer_tokens,
        "max_tokens": args.max_tokens,
        "train_encoder": args.encoder_lora,
        "rank": args.rank,
        "alpha": args.alpha,
        "target_modules": list(env["targets"]),
        "seed": args.seed,
        "batch": args.batch,
        "gradient_accumulation": args.accum,
        "effective_batch": effective,
        "lr": args.lr,
        "encoder_lr": args.encoder_lr,
        "weight_decay": 0.01,
        "warmup_steps": args.warmup_steps,
        "schedule": "linear-warmup-cosine-to-10pct",
        "clip_grad_norm": 1.0,
        "save_every": args.save_every,
        "eval_every": args.eval_every,
        "validation_slice": args.validation_slice,
        "copy_pointer_weight": args.copy_pointer_weight,
        "copy_gate_weight": args.copy_gate_weight,
        "copy_head_temperature": COPY_HEAD_TEMPERATURE,
        "memory_scale_init": args.memory_scale_init,
        "placeholder_id": PLACEHOLDER_ID,
        "placeholder_text": env["placeholder"],
        "placeholder_recipe": PLACEHOLDER_RECIPE,
        "ple_identity_id": env["pad_id"],
        "prompt_recipe": PROMPT_RECIPE,
        "system_prompt": HELPFUL_PROMPT,
        "chat_template_system_role": True,
        "injection_recipe": INJECTION_RECIPE,
        "embedding_space": "embed_tokens output (rows scaled by sqrt(hidden))",
        "label_recipe": LABEL_RECIPE,
        "final_logit_softcapping": env["softcap"],
        "eos_ids": list(env["eos_ids"]),
        "decoder_frozen": True,
        "decoder_gradient_checkpointing": True,
        "precision": "BF16 autocast encoder/reader/decoder; FP32 reader, copy head, adapters; FP16->BF16 feature channel",
        "script_sha256": env["script_sha256"],
        "development_only": True,
        "rollout_qualified": False,
        "versions": env["versions"],
    }
    protocol.update(option_protocol(args, parts, split_rows))
    return protocol


INIT_KEYS = ("init_source", "init_new_tensors")


def resume_protocol(previous, protocol):
    """Check a resumed run's protocol -> ``(protocol to keep, new script
    digest or None)``. Everything must match except ``script_sha256`` (the
    script that started the run stays recorded; a different resuming script
    is returned for ``result.json``) and the ``--init`` entries written after
    the first save (kept from ``previous``: resume.pt supersedes ``--init``)."""
    ignored = (*INIT_KEYS, "script_sha256")

    def core(p):
        return {k: v for k, v in p.items() if k not in ignored}

    if "script_sha256" not in previous or core(previous) != core(protocol):
        raise ValueError("resume requires an identical protocol")
    kept = dict(protocol)
    kept["script_sha256"] = previous["script_sha256"]
    kept.update({k: previous[k] for k in INIT_KEYS if k in previous})
    changed = protocol["script_sha256"] != previous["script_sha256"]
    return kept, protocol["script_sha256"] if changed else None


def train(args):
    resolve_options(args)
    if not 1 <= args.batch <= 64 or not 1 <= args.accum <= 256:
        raise ValueError("batch must be 1..64 and accumulation 1..256")
    if not 0 < args.lr <= 1e-2 or not 0 < args.encoder_lr <= 1e-2:
        raise ValueError("learning rates must be within (0, 1e-2]")
    if args.validation_slice < 1:
        raise ValueError("validation slice must be positive")
    if args.reader_streams == "dual" and args.encoder_lora:
        raise SystemExit(
            "--reader-streams dual keeps the pooled store on the frozen encoder;"
            " --encoder-lora is not supported with it"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if gpu_compute_processes() and not args.allow_shared_gpu:
        raise RuntimeError("GPU already occupied; refusing a competing training run")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parts = recipe_parts(args)
    if args.recipe in ("stage2", "stage2-replay") and args.init is None:
        raise SystemExit(
            f"--recipe {args.recipe} needs --init DIR (the Stage 1 candidate)"
        )
    args.output.mkdir(parents=True, exist_ok=args.resume)
    import transformers

    splits = [ParquetSplit(args.data_root, name, "train") for name, _ in parts]
    labels, rows_order = mixture_order(
        [(s.group_sizes, cap) for s, (_, cap) in zip(splits, parts)], args.seed
    )
    total_examples = int(labels.numel())
    effective = args.batch * args.accum
    steps = schedule_steps(total_examples, effective, args.max_steps)
    validation = []
    for name, _ in parts:
        try:
            validation.append(ParquetSplit(args.data_root, name, "validation"))
        except FileNotFoundError:
            emit(
                "warning",
                message=f"{name} has no validation split; skipped in evaluations",
            )
    trainer = Trainer(args, train_encoder=args.encoder_lora)
    protocol = build_protocol(
        args,
        parts,
        [len(s) for s in splits],
        total_examples,
        steps,
        {
            "encoder": model_identity(args.encoder),
            "decoder": model_identity(args.decoder),
            "layout": trainer.layout,
            "targets": trainer.targets,
            "placeholder": trainer.placeholder,
            "pad_id": trainer.pad_id,
            "softcap": trainer.softcap,
            "eos_ids": trainer.eos_ids,
            "script_sha256": digest(Path(__file__)),
            "versions": {
                "torch": str(torch.__version__),
                "transformers": str(transformers.__version__),
            },
        },
    )
    script_change = None
    if args.resume:
        previous = json.loads((args.output / "protocol.json").read_text())
        protocol, script_change = resume_protocol(previous, protocol)
    save_json(args.output / "protocol.json", protocol)
    decoder = trainer.models["decoder"]
    decoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if args.init is not None and not args.resume:
        source, new_tensors = load_trained(
            args.init,
            trainer.reader,
            trainer.copy_head,
            trainer.models["adapters"],
            trainer.stream_type,
        )
        protocol["init_source"] = source
        if trainer.stream_type is not None:
            protocol["init_new_tensors"] = new_tensors
        save_json(args.output / "protocol.json", protocol)
        emit("init", source=source, path=str(args.init), new_tensors=new_tensors)
    reader, copy_head = trainer.reader, trainer.copy_head
    projection = list(reader.memory_projection.parameters())
    projection_ids = {id(p) for p in projection}
    groups = [
        {
            "params": [p for p in reader.parameters() if id(p) not in projection_ids],
            "lr": args.lr,
            "name": "reader_body",
        },
        {"params": projection, "lr": args.lr, "name": "projection"},
        {"params": list(copy_head.parameters()), "lr": args.lr, "name": "copy_head"},
    ]
    stream_type = trainer.stream_type
    if stream_type is not None:
        groups.append(
            {
                "params": list(stream_type.parameters()),
                "lr": args.lr,
                "name": "stream_type",
            }
        )
    if args.encoder_lora:
        groups.append(
            {
                "params": [
                    p
                    for p in trainer.models["text_model"].parameters()
                    if p.requires_grad
                ],
                "lr": args.encoder_lr,
                "name": "encoder_lora",
            }
        )
    base_lr = {g["name"]: g["lr"] for g in groups}
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    trainable = [p for g in optimizer.param_groups for p in g["params"]]
    frozen = frozen_versions(trainer.models["text_model"], decoder)
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
        "skipped_examples": 0,
        "cursor": 0,
    }
    start_step = 0
    if args.resume:
        state = torch.load(
            args.output / "resume.pt", map_location="cpu", weights_only=True
        )
        reader.load_state_dict(state["reader"])
        copy_head.load_state_dict(state["copy_head"])
        if state.get("adapters"):
            restore_adapters(trainer.models["adapters"], state["adapters"])
        if stream_type is not None:
            stream_type.load_state_dict(state["stream_type"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        result, start_step = state["result"], int(state["step"])
        result["completed"] = False
        base_lr["projection"] = args.lr * result.get("memory_scale_factor", 1.0)
        if script_change is not None:
            # protocol.json keeps the digest of the script that started the run.
            result.setdefault("script_resumes", []).append(
                {"step": start_step, "script_sha256": script_change}
            )
    save_json(result_path, result)
    cursor = {"value": int(result["cursor"])}

    def next_rows(count):
        """``count`` rows from the epoch order at the cursor, skipping rows
        whose prompt does not fit; -> ``(rows, examples)``."""
        rows, examples = [], []
        while len(rows) < count and cursor["value"] < total_examples:
            index = cursor["value"]
            cursor["value"] += 1
            row = splits[int(labels[index])].row(int(rows_order[index]))
            example = trainer.decoder_batch([row])[0]
            if example is None:
                result["skipped_examples"] += 1
                continue
            rows.append(row)
            examples.append(example)
        if not rows:
            raise RuntimeError("the epoch order is exhausted")
        return rows, examples

    def evaluate(step):
        started = time.monotonic()
        report = validation_loss(trainer, validation, args.batch, args.validation_slice)
        evaluation = {
            "step": step,
            "cursor": cursor["value"],
            "validation": report,
            "seconds": round(time.monotonic() - started, 1),
        }
        result["evaluations"].append(evaluation)
        save_json(result_path, result)
        emit("evaluation", **evaluation)
        assert_frozen(frozen)

    def checkpoint(step):
        assert_frozen(frozen)
        result["cursor"] = cursor["value"]
        atomic_checkpoint(
            args.output / "resume.pt",
            {
                "step": step,
                "reader": {k: v.detach().cpu() for k, v in reader.state_dict().items()},
                "copy_head": copy_head_state(copy_head),
                "adapters": adapter_state(trainer.models["adapters"])
                if args.encoder_lora
                else None,
                **(
                    {
                        "stream_type": {
                            k: v.detach().cpu()
                            for k, v in stream_type.state_dict().items()
                        }
                    }
                    if stream_type is not None
                    else {}
                ),
                "optimizer": optimizer.state_dict(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "result": result,
            },
        )
        save_json(result_path, result)
        emit(
            "checkpoint",
            step=step,
            cursor=cursor["value"],
            path=str(args.output / "resume.pt"),
        )

    emit(
        "start",
        steps=steps,
        total_examples=total_examples,
        effective_batch=effective,
        start_step=start_step,
        cursor=cursor["value"],
        recipe=args.recipe,
        datasets=protocol["datasets"],
    )
    try:
        if (
            not args.resume
            and args.init is None
            and args.memory_scale_init == "lexical-target"
        ):
            # Fold the scaled token-embedding norm into the fresh projection so
            # the soft rows start at the decoder's own input scale.
            trainer.modes(False)
            peek = [
                splits[int(labels[i])].row(int(rows_order[i]))
                for i in range(min(args.batch, total_examples))
            ]
            with torch.no_grad():
                encoded = trainer.encoder.encode(peek, False)
                memories = trainer.reader_memories(encoded)
                examples = [e for e in trainer.decoder_batch(peek) if e is not None]
                ids, mask, *_ = collate_decoder(examples, trainer.pad_id)
                rows_e = trainer.text.embed_tokens(ids.cuda())[
                    mask.cuda().bool()
                ].float()
                target_norm = float(rows_e.norm(dim=-1).mean())
            result["memory_scale_factor"] = rescale_memory_projection(
                reader, list(memories), target_norm
            )
            result["memory_scale_target_norm"] = target_norm
            base_lr["projection"] = args.lr * result["memory_scale_factor"]
            emit(
                "memory-scale-init",
                factor=result["memory_scale_factor"],
                target_norm=target_norm,
            )
            del memories, encoded
        if not args.resume and not args.skip_initial_eval:
            evaluate(0)
        for step in range(start_step, steps):
            trainer.modes(True)
            optimizer.zero_grad(set_to_none=True)
            scale = lr_scale(step, steps, args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = base_lr[group["name"]] * scale
            started = time.monotonic()
            totals = collections.defaultdict(float)
            counts = collections.defaultdict(int)
            inputs = []
            for _ in range(args.accum):
                rows, examples = next_rows(args.batch)
                losses = trainer.losses(rows, examples, True)
                loss = losses["ce"]
                if args.copy_slots:
                    loss = (
                        loss
                        + args.copy_pointer_weight * losses["pointer"]
                        + args.copy_gate_weight * losses["gate"]
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                (loss / args.accum).backward()
                totals["loss"] += float(loss.detach()) / args.accum
                totals["ce"] += float(losses["ce"].detach()) / args.accum
                if args.copy_slots:
                    totals["pointer"] += float(losses["pointer"].detach()) / args.accum
                    totals["gate"] += float(losses["gate"].detach()) / args.accum
                    totals["gate_mean"] += losses["gate_mean"] / args.accum
                    counts["copyable_cases"] += losses["copyable_cases"]
                counts["examples"] += len(rows)
                counts["answer_tokens"] += losses["answer_tokens"]
                counts["decoder_tokens"] += losses["decoder_tokens"]
                counts["encoder_tokens"] += losses["encoder_tokens"]
                inputs.extend(losses["stats"])
                del losses, loss
            reader_grad = (
                sum(
                    float(p.grad.float().square().sum())
                    for p in reader.parameters()
                    if p.grad is not None
                )
                ** 0.5
            )
            if not math.isfinite(reader_grad) or reader_grad == 0:
                raise RuntimeError("missing or nonfinite reader gradient")
            assert_frozen(frozen)
            norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            )
            optimizer.step()
            torch.cuda.synchronize()
            row = {
                "step": step + 1,
                **{k: round(v, 5) for k, v in totals.items()},
                **dict(counts),
                "grad_norm": round(norm, 4),
                "reader_grad": round(reader_grad, 4),
                "lr_scale": round(scale, 5),
                "cursor": cursor["value"],
                "skipped": result["skipped_examples"],
                "seconds": round(time.monotonic() - started, 3),
                "max_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }
            if options_active(args):
                row.update(memory_input_summary(inputs, compact=True))
            result["history"].append(row)
            emit("step", **row)
            done = step + 1 == steps
            if (step + 1) % args.eval_every == 0 or done:
                evaluate(step + 1)
            if (step + 1) % args.save_every == 0 or done:
                checkpoint(step + 1)
        result["completed"] = True
        result["frozen_base_versions_unchanged"] = True
        result["selection"] = {
            "calls": trainer.encoder.selection_calls,
            "documents_dropped": trainer.encoder.selected,
        }
        if options_active(args):
            result["selection"].update(
                chunk_calls=trainer.encoder.chunk_calls,
                chunks_dropped=trainer.encoder.chunks_dropped,
            )
        result["candidate"] = write_candidate(
            args.output,
            reader,
            trainer.models["adapters"],
            copy_head,
            args.memory_slots,
            args.copy_slots,
            args.encoder_lora,
            stream_type,
        )
        save_json(result_path, result)
        emit("completed", steps=steps, cursor=cursor["value"], output=str(args.output))
    except BaseException as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save_json(result_path, result)
        raise
    finally:
        trainer.models["taps"].close()


# ----------------------------------------------------------------------------
# eval and probe commands


def load_for_inference(args):
    """Models plus a trained reader/copy head from ``--candidate`` (a finished
    candidate directory or an interrupted run with ``resume.pt``); without a
    candidate a fresh reader at the CLI geometry (plumbing probes only)."""
    if args.candidate is not None:
        directory = Path(args.candidate)
        if (directory / "candidate.json").exists():
            protocol, _ = load_candidate(directory)
        else:
            protocol = json.loads((directory / "protocol.json").read_text())
    else:
        protocol = None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if gpu_compute_processes() and not args.allow_shared_gpu:
        raise RuntimeError("GPU already occupied; refusing a competing run")
    torch.backends.cuda.matmul.allow_tf32 = False
    defaults = {
        "doc_tokens": 256,
        "max_docs": 16,
        "max_rows": 4096,
        "max_answer_tokens": 256,
        "max_tokens": 1024,
        "rank": 8,
        "alpha": 16,
    }
    for key, default in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, protocol[key] if protocol is not None else default)
    # Paper options: the run's own (absent in legacy protocols: the legacy
    # values) unless overridden on the command line; the stream layout is
    # part of the trained reader and always comes from the protocol.
    # The pooled stream reads the whole collection unless --max-chunks caps it
    # (the training cap is not inherited).
    stored = protocol or {}
    for key, default in {**LEGACY_OPTIONS, **OPTION_DEFAULTS}.items():
        if key == "max_chunks":
            continue
        if key == "reader_streams" or getattr(args, key, None) is None:
            value = stored.get(key)
            setattr(args, key, default if value is None else value)
    args.max_chunks = getattr(args, "max_chunks", None)
    args.reader_dropout = 0.0  # inference: dropout is inactive in eval mode
    if protocol is not None:
        args.lora_targets = ",".join(protocol["target_modules"])
        args.seed = int(protocol["seed"])
        trainer = Trainer(
            args,
            int(protocol["memory_slots"]),
            int(protocol["copy_slots"]),
            int(protocol["tap_layer"]),
            False,
        )
        source, _ = load_trained(
            args.candidate,
            trainer.reader,
            trainer.copy_head,
            trainer.models["adapters"],
            trainer.stream_type,
        )
    else:
        args.seed = 23
        trainer = Trainer(args, DEFAULT_MEMORY_SLOTS, 8, DEFAULT_TAP_LAYER, False)
        source = "fresh-untrained-reader"
    trainer.modes(False)
    return trainer, protocol, source


def subsample(length, limit, seed):
    if limit is None or limit >= length:
        return list(range(length))
    generator = torch.Generator().manual_seed(int(seed))
    return sorted(torch.randperm(length, generator=generator)[:limit].tolist())


def document_token_report(stats, slots):
    """Document tokens per query (the paper's compression view): the decoder
    reads ``slots`` memory rows as its document context; the collection is
    the untruncated encoder tokenization of every document."""
    collection = [s["document_tokens"] for s in stats if s.get("document_tokens")]
    read = [s["read_tokens"] for s in stats]
    if not collection:
        return None
    return {
        "decoder_document_tokens_per_query": slots,
        "collection_tokens_mean": round(sum(collection) / len(collection), 1),
        "collection_tokens_max": max(collection),
        "encoder_read_tokens_mean": round(sum(read) / max(len(read), 1), 1),
        "compression_mean": round(
            sum(c / slots for c in collection) / len(collection), 1
        ),
    }


def attach_feature_cache(trainer, args):
    """Evaluation with dual streams: cache each collection's chunk ids and
    pooled rows (in memory; also on disk with ``--feature-cache DIR``)."""
    if trainer.reader_streams != "dual":
        return None
    directory = getattr(args, "feature_cache", None)
    key = feature_cache_key(args, args.encoder) if directory is not None else None
    trainer.encoder.cache = FeatureCache(key, directory)
    return trainer.encoder.cache


def evaluate_candidate(args):
    trainer, _protocol, source = load_for_inference(args)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = EvalMetrics(args.upstream_source)
    cache = attach_feature_cache(trainer, args)
    if args.noise_documents:
        trainer.noise = torch.Generator().manual_seed(int(args.eval_seed))
    report = {
        "schema": SCHEMA,
        "candidate": str(args.candidate),
        "source": source,
        "protocol_sha256": digest(Path(args.candidate) / "protocol.json")
        if args.candidate
        else None,
        "limit": args.limit,
        "eval_seed": args.eval_seed,
        "max_new_tokens": args.max_new_tokens,
        "long_form_new_tokens": args.long_form_new_tokens,
        "long_form_configs": list(LONG_FORM_CONFIGS),
        "metrics": metrics.name,
        "noise_documents": bool(args.noise_documents),
        "noise_scope": "reader key rows and copy-source rows, Gaussian, per-row L2 norm matched"
        if args.noise_documents
        else None,
        "memory_options": {
            key: getattr(args, key)
            for key in ("chunking", "reader_streams", "max_chunks", "token_chunks")
        },
        "pooled_stream": "whole collection"
        if args.max_chunks is None
        else f"top {args.max_chunks} chunks by pooled cosine",
        "feature_cache": None
        if cache is None
        else {
            "directory": str(cache.directory) if cache.directory else None,
            "key": cache.key,
        },
        "datasets": {},
    }
    try:
        for config in args.configs.split(","):
            split_name = eval_split_name(args.data_root, config, args.split)
            split = ParquetSplit(args.data_root, config, split_name)
            max_new = (
                args.long_form_new_tokens
                if config in LONG_FORM_CONFIGS
                else args.max_new_tokens
            )
            indices = subsample(len(split), args.limit, args.eval_seed)
            rows, started = [], time.monotonic()
            for index in indices:
                row = split.row(index)
                answer = trainer.answer(row, max_new)
                actual = metrics.clean(answer["answer"])
                scored = {
                    "row": index,
                    "collection_id": row.get("collection_id"),
                    "question": row["question"],
                    "question_type": question_type(row["question"]),
                    "expected": row["answer"],
                    "actual": actual,
                    **metrics.row(actual, row["answer"]),
                    "finish_reason": answer["finish_reason"],
                    "spans": answer["spans"],
                    "gate": answer["gate"],
                    "snapped_ids": answer["snapped_ids"],
                    "memory_inputs": answer["memory_inputs"],
                }
                rows.append(scored)
                emit(
                    "case",
                    config=config,
                    row=index,
                    actual=actual[:300],
                    expected=str(row["answer"])[:300],
                    em=scored["em"],
                    f1=scored["f1"],
                    rougeL=scored["rougeL"],
                )
            answered = time.monotonic()
            loss = validation_loss(
                trainer, [_Slice(split, indices)], args.batch, len(indices)
            )[config]
            count = max(len(rows), 1)
            summary = {
                "split": split_name,
                "max_new_tokens": max_new,
                "examples": len(rows),
                **{
                    key: round(sum(r[key] for r in rows) / count, 4)
                    for key in ("em", "f1", "squad_f1", "rouge1", "rougeL")
                },
                **metrics.corpus(
                    [r["actual"] for r in rows], [r["expected"] for r in rows]
                ),
                "validation_loss": loss["loss"],
                "seconds": round(time.monotonic() - started, 1),
                "answer_seconds": round(answered - started, 1),
                "validation_seconds": round(time.monotonic() - answered, 1),
                "memory_inputs": memory_input_summary(
                    [r["memory_inputs"] for r in rows], compact=True
                ),
            }
            if config == "narrativeqa":
                summary["by_question_type"] = summarize_by_type(rows)
            if args.report_doc_tokens:
                summary["document_tokens"] = document_token_report(
                    [r["memory_inputs"] for r in rows], trainer.slots
                )
            if cache is not None:
                summary["feature_cache"] = dict(cache.stats)
            summary["rows"] = rows
            report["datasets"][config] = summary
            emit(
                "result",
                config=config,
                **{k: v for k, v in summary.items() if k != "rows"},
            )
            save_json(args.output / "result.json", report)
    finally:
        trainer.models["taps"].close()
    return report


class _Slice:
    """A split restricted to chosen row indices (for validation loss)."""

    def __init__(self, split, indices):
        self.split, self.indices, self.config = split, list(indices), split.config

    def __len__(self):
        return len(self.indices)

    def row(self, index):
        return self.split.row(self.indices[index])


def probe(args):
    trainer, _protocol, source = load_for_inference(args)
    attach_feature_cache(trainer, args)
    try:
        split = ParquetSplit(args.data_root, args.config, "validation")
        row = split.row(args.index)
        answer = trainer.answer(row, args.max_new_tokens)
        example = trainer.decoder_batch([row])[0]
        loss = None
        if example is not None:
            with torch.no_grad():
                loss = round(float(trainer.losses([row], [example], False)["ce"]), 4)
        report = {
            "schema": SCHEMA,
            "candidate": str(args.candidate),
            "source": source,
            "config": args.config,
            "row": args.index,
            "question": row["question"],
            "documents": row["documents"],
            "passage": row["answer"],
            "generated": answer["answer"],
            "finish_reason": answer["finish_reason"],
            "spans": answer["spans"],
            "gate": answer["gate"],
            "snapped_ids": answer["snapped_ids"],
            "teacher_forced_ce": loss,
            "f1": round(token_f1(answer["answer"], row["answer"]), 4),
        }
        emit("probe", **report)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            save_json(args.output, report)
        return report
    finally:
        trainer.models["taps"].close()


# ----------------------------------------------------------------------------
# CLI


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def assets(p):
        p.add_argument("--encoder", type=Path, default=DEFAULT_ASSETS["encoder"])
        p.add_argument("--decoder", type=Path, default=DEFAULT_ASSETS["decoder"])
        p.add_argument(
            "--upstream-source", type=Path, default=DEFAULT_ASSETS["upstream_source"]
        )
        p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
        p.add_argument("--allow-shared-gpu", action="store_true")
        p.add_argument("--batch", type=int, default=4)

    def geometry(p, required_defaults):
        d = (lambda v: v) if required_defaults else (lambda v: None)
        p.add_argument("--doc-tokens", type=int, default=d(256))
        p.add_argument("--max-docs", type=int, default=d(16))
        p.add_argument("--max-rows", type=int, default=d(4096))
        p.add_argument("--max-answer-tokens", type=int, default=d(256))
        p.add_argument("--max-tokens", type=int, default=d(1024))
        p.add_argument("--rank", type=int, default=d(8))
        p.add_argument("--alpha", type=int, default=d(16))

    def options(p, training):
        """Paper options; unset (None) values resolve to the recipe preset or
        legacy value (train) or to the candidate's protocol (eval, probe)."""
        p.add_argument(
            "--chunking",
            choices=CHUNKING_CHOICES,
            help="sentence128: sentence-greedy chunks of <= 128 encoder tokens"
            " instead of per-document truncation (train default: none, or the"
            " recipe preset)",
        )
        p.add_argument(
            "--max-chunks",
            type=int,
            help="dual streams: pooled rows per example, lowest pooled cosine"
            f" dropped (train default {OPTION_DEFAULTS['max_chunks']}; eval and"
            " probe default: no cap, the whole collection)",
        )
        p.add_argument(
            "--token-chunks",
            type=int,
            help="dual streams: chunks whose token rows the reader reads and the"
            f" copy head copies from (default {OPTION_DEFAULTS['token_chunks']})",
        )
        p.add_argument(
            "--encoder-chunk-batch",
            type=int,
            help="chunks per encoder forward"
            f" (default {OPTION_DEFAULTS['encoder_chunk_batch']})",
        )
        if not training:
            p.add_argument(
                "--feature-cache",
                type=Path,
                help="dual streams: directory caching each collection's chunk ids"
                " and pooled rows across runs (always cached in memory within a"
                " run)",
            )
        if training:
            p.add_argument(
                "--reader-streams",
                choices=STREAM_CHOICES,
                help="dual: [pooled rows of every chunk; token rows of the top"
                " chunks] with a stream-type embedding (default tokens, or the"
                " recipe preset)",
            )
            p.add_argument(
                "--reader-dropout",
                type=float,
                help="reader dropout in train mode (default 0, or the recipe"
                " preset; the paper uses 0.1)",
            )

    t = sub.add_parser(
        "train", help="train soft slots through the frozen decoder plus the copy head"
    )
    assets(t)
    geometry(t, True)
    options(t, True)
    t.add_argument(
        "--recipe",
        choices=tuple(RECIPES),
        help="stage2-replay = the ten Stage 2 configs + enwiki:200000, implying"
        " --chunking sentence128 --reader-streams dual --reader-dropout 0.1",
    )
    t.add_argument(
        "--datasets", help="config[:cap],... overriding the recipe's configs"
    )
    t.add_argument("--init", type=Path, help="candidate or run directory to start from")
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--memory-slots", type=int, default=DEFAULT_MEMORY_SLOTS)
    t.add_argument("--copy-slots", type=int, default=8)
    t.add_argument(
        "--cross-layers",
        type=int,
        default=2,
        choices=(2,),
        help="reader cross-attention blocks (the pinned reader_config fixes 2)",
    )
    t.add_argument("--tap-layer", type=int, default=DEFAULT_TAP_LAYER)
    t.add_argument("--encoder-lora", action="store_true")
    t.add_argument("--lora-targets", default="q_proj,v_proj")
    t.add_argument("--accum", type=int, default=32)
    t.add_argument("--max-steps", type=int)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--encoder-lr", type=float, default=1e-4)
    t.add_argument("--warmup-steps", type=int, default=500)
    t.add_argument("--save-every", type=int, default=1000)
    t.add_argument("--eval-every", type=int, default=2000)
    t.add_argument(
        "--validation-slice",
        type=int,
        default=VALIDATION_SLICE,
        help="leading validation rows per config at every evaluation",
    )
    t.add_argument("--copy-pointer-weight", type=float, default=1.0)
    t.add_argument("--copy-gate-weight", type=float, default=3.0)
    t.add_argument(
        "--memory-scale-init",
        choices=("none", "lexical-target"),
        default="lexical-target",
    )
    t.add_argument("--skip-initial-eval", action="store_true")
    t.add_argument("--seed", type=int, default=23)
    t.add_argument("--resume", action="store_true")

    e = sub.add_parser(
        "eval", help="exact match / F1 and validation loss on QA validation splits"
    )
    assets(e)
    geometry(e, False)
    options(e, False)
    e.add_argument("--candidate", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    e.add_argument(
        "--configs",
        default=",".join(EVAL_CONFIGS),
        help="Memorilla-Data configs, e.g. squad_v2,drop,triviaqa,narrativeqa,enwiki",
    )
    e.add_argument(
        "--split",
        choices=("auto", "validation", "test"),
        default="auto",
        help="auto: narrativeqa's test split when present, else validation",
    )
    e.add_argument("--limit", type=int, default=200)
    e.add_argument("--eval-seed", type=int, default=20261113)
    e.add_argument(
        "--max-new-tokens",
        type=int,
        default=QA_NEW_TOKENS,
        help="greedy limit for QA configs (the paper's 64)",
    )
    e.add_argument(
        "--long-form-new-tokens",
        type=int,
        default=LONG_FORM_NEW_TOKENS,
        help="greedy limit for enwiki reconstruction (the paper's 512)",
    )
    e.add_argument(
        "--noise-documents",
        action="store_true",
        help="sanity test: every reader key row and copy-source row replaced by"
        " Gaussian noise of the same norm",
    )
    e.add_argument(
        "--report-doc-tokens",
        action="store_true",
        help="report document tokens per query (decoder side: the memory slots)",
    )

    p = sub.add_parser(
        "probe", help="round-trip one validation example through the injected memory"
    )
    assets(p)
    geometry(p, False)
    options(p, False)
    p.add_argument(
        "--candidate", type=Path, help="omit for a fresh-reader plumbing probe"
    )
    p.add_argument("--config", default="enwiki")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--output", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "train":
        if not args.encoder.is_dir() or not args.decoder.is_dir():
            raise SystemExit("encoder and decoder directories are required")
        train(args)
    elif args.command == "eval":
        evaluate_candidate(args)
    else:
        probe(args)


if __name__ == "__main__":
    sys.exit(main())
