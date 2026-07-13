#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
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
"""Validate the multi-domain fused-chunker training mix against RawRecord.

Mirrors the Zig parser contract (src/finetune/fused_chunker_data.zig
RawRecord/RawChunk): each JSONL line must be an object with a non-empty
"text" string and a "chunks" (or legacy "chunk_boundaries") array of
{"start_char": int, "end_char": int} spans. Offsets are UTF-8 BYTE
offsets into text.

Strictness is per record provenance:
  - new-domain records (they carry a "domain" field) must be strictly
    contiguous, non-overlapping, cover [0, byte_len) exactly, land on
    UTF-8 codepoint starts, and have >= 2 chunks;
  - base pass-through records (no "domain" field; `chunk_boundaries`
    variant) keep their historical convention: monotonic spans, 0/1-byte
    separator gaps, final end_char within a small slack of byte_len.

Also reports per-domain stats (docs, mean doc/chunk bytes, boundary rate
per whitespace token, degenerate docs) and warns when a domain's
boundary rate leaves the sane 0.2-2% band or its mean chunk size leaves
the 1200-2200 target band (new domains only).

stdlib-only. Exit code 1 on any structural violation.

Usage:
  python3 validate_fused_chunker_multidomain_mix.py \
      --train /private/tmp/multidomain_mix/train_mix.jsonl \
      --val /private/tmp/multidomain_mix/val_mix.jsonl \
      [--stats-out /private/tmp/multidomain_mix/validation_stats.json]
"""

import argparse
import json
import sys

BASE_GAP_ALLOWED = (0, 1)   # historical 1-byte separator gaps
BASE_END_SLACK = 8          # historical final end_char overshoot
BAND = (1200, 2200)
BOUNDARY_RATE_BAND = (0.2, 2.0)


class Agg:
    def __init__(self):
        self.docs = 0
        self.doc_bytes = 0
        self.chunks = 0
        self.chunk_bytes = 0
        self.ws_tokens = 0
        self.one_chunk_docs = 0

    def add(self, n_bytes, chunk_sizes, ws_tokens):
        self.docs += 1
        self.doc_bytes += n_bytes
        self.chunks += len(chunk_sizes)
        self.chunk_bytes += sum(chunk_sizes)
        self.ws_tokens += ws_tokens
        if len(chunk_sizes) <= 1:
            self.one_chunk_docs += 1

    def summary(self):
        boundaries = self.chunks - self.docs
        return {
            "docs": self.docs,
            "mean_doc_chars": round(self.doc_bytes / self.docs) if self.docs else 0,
            "mean_chunks_per_doc": round(self.chunks / self.docs, 2) if self.docs else 0,
            "mean_chunk_chars": round(self.chunk_bytes / self.chunks) if self.chunks else 0,
            "boundary_rate_ws_tokens_pct": (
                round(100.0 * boundaries / self.ws_tokens, 3) if self.ws_tokens else 0.0),
            "one_chunk_docs": self.one_chunk_docs,
        }


def is_codepoint_start(raw, off):
    return off >= len(raw) or (raw[off] & 0xC0) != 0x80


def check_record(rec, lineno, path, errors, allow_singletons=False):
    """Returns (domain, n_bytes, chunk_sizes, ws_tokens) or None."""
    def err(msg):
        errors.append(f"{path}:{lineno}: {msg}")

    if not isinstance(rec, dict):
        err("record is not an object")
        return None
    text = rec.get("text")
    if not isinstance(text, str) or not text.strip():
        err("missing/empty text")
        return None
    chunks = rec.get("chunks")
    legacy = False
    if chunks is None:
        chunks = rec.get("chunk_boundaries")
        legacy = True
    if not isinstance(chunks, list) or not chunks:
        err("missing/empty chunks")
        return None

    domain = rec.get("domain")
    strict = domain is not None
    raw = text.encode("utf-8")
    n_bytes = len(raw)
    spans = []
    for i, c in enumerate(chunks):
        if not isinstance(c, dict):
            err(f"chunk[{i}] not an object")
            return None
        a, b = c.get("start_char"), c.get("end_char")
        if not isinstance(a, int) or not isinstance(b, int) or isinstance(a, bool) or isinstance(b, bool):
            err(f"chunk[{i}] start_char/end_char not ints")
            return None
        if a < 0 or b <= a:
            err(f"chunk[{i}] bad span [{a},{b})")
            return None
        spans.append((a, b))

    for i in range(1, len(spans)):
        gap = spans[i][0] - spans[i - 1][1]
        if strict and gap != 0:
            err(f"chunk[{i}] not contiguous (gap {gap})")
        if not strict and gap not in BASE_GAP_ALLOWED:
            err(f"chunk[{i}] base gap {gap} outside {BASE_GAP_ALLOWED}")
    if strict:
        if spans[0][0] != 0:
            err(f"first chunk starts at {spans[0][0]}, not 0")
        if spans[-1][1] != n_bytes:
            err(f"last chunk ends at {spans[-1][1]}, text is {n_bytes} bytes")
        if len(spans) < 2 and not allow_singletons:
            err("degenerate 1-chunk doc in new domain")
        for a, b in spans:
            if not is_codepoint_start(raw, a) or not is_codepoint_start(raw, min(b, n_bytes)):
                err(f"span [{a},{b}) not on UTF-8 codepoint boundary")
                break
        if legacy:
            err("new-domain record uses legacy chunk_boundaries field")
    else:
        if spans[0][0] != 0:
            err(f"base first chunk starts at {spans[0][0]}")
        if not (n_bytes - BASE_END_SLACK <= spans[-1][1] <= n_bytes + BASE_END_SLACK):
            err(f"base last end {spans[-1][1]} vs byte len {n_bytes} (slack {BASE_END_SLACK})")

    sizes = [b - a for a, b in spans]
    return (domain or "base", n_bytes, sizes, len(text.split()))


def validate_file(path, aggs, errors, max_report=50, allow_singletons=False):
    n = 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"{path}:{lineno}: invalid JSON: {e}")
                continue
            before = len(errors)
            out = check_record(rec, lineno, path, errors, allow_singletons)
            if out is None or len(errors) > before:
                if len(errors) >= max_report:
                    errors.append(f"{path}: too many errors, stopping report")
                    return n
                continue
            domain, n_bytes, sizes, ws = out
            aggs.setdefault(domain, Agg()).add(n_bytes, sizes, ws)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="/private/tmp/multidomain_mix/train_mix.jsonl")
    ap.add_argument("--val", default="/private/tmp/multidomain_mix/val_mix.jsonl")
    ap.add_argument("--stats-out", default="")
    ap.add_argument("--windowed", action="store_true",
                    help="validating a pre-windowed (~1024-token) mix: 1-chunk "
                         "records are intentional packing/oversize singletons and "
                         "records are window-sized, so skip the chunk-band warning")
    args = ap.parse_args()

    all_stats = {}
    errors = []
    for split, path in (("train", args.train), ("val", args.val)):
        aggs = {}
        n = validate_file(path, aggs, errors, allow_singletons=args.windowed)
        stats = {d: a.summary() for d, a in sorted(aggs.items())}
        all_stats[split] = {"file": path, "valid_docs": n, "domains": stats}
        print(f"== {split}: {path} ({n} valid docs)")
        for d, s in stats.items():
            flags = []
            if d != "base" and not args.windowed:
                if not (BAND[0] <= s["mean_chunk_chars"] <= BAND[1]):
                    flags.append("CHUNK-BAND")
                if s["one_chunk_docs"]:
                    flags.append("1-CHUNK-DOCS")
            rate = s["boundary_rate_ws_tokens_pct"]
            if not (BOUNDARY_RATE_BAND[0] <= rate <= BOUNDARY_RATE_BAND[1]):
                flags.append("BOUNDARY-RATE")
            print(f"  {d:12s} docs={s['docs']:6d} mean_doc={s['mean_doc_chars']:6d} "
                  f"chunks/doc={s['mean_chunks_per_doc']:5.2f} mean_chunk={s['mean_chunk_chars']:5d} "
                  f"boundary_rate={rate:6.3f}% one_chunk={s['one_chunk_docs']:5d} "
                  f"{'WARN:' + ','.join(flags) if flags else 'ok'}")

    if args.stats_out:
        with open(args.stats_out, "w", encoding="utf-8") as f:
            json.dump({"stats": all_stats, "errors": errors}, f, indent=2)
            f.write("\n")

    if errors:
        print(f"\n{len(errors)} structural errors:", file=sys.stderr)
        for e in errors[:50]:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    print("\nvalidation passed")


if __name__ == "__main__":
    main()
