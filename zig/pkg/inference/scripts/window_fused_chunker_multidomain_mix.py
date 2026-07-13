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
"""Pre-window the multi-domain fused-chunker mix into ~1024-token records.

The Zig loader (src/finetune/fused_chunker_data.zig charToTokenBoundary /
assembleTokenBatch) masks out any chunk that does not fit ENTIRELY inside
the tokenized window, and a boundary label is emitted only at the first
token of each valid chunk after the first valid chunk. Training documents
longer than --seq-len tokens therefore contribute few or no gold
boundaries. This transform makes the mix trainable at --seq-len by
greedily packing consecutive WHOLE chunks into records whose byte length
approximates the token window, cutting only at existing chunk boundaries:

  - per-domain byte budget = (seq_len - 4) * bytes_per_token * safety,
    with bytes/token measured against the actual ModernBERT tokenizer via
    count-fused-tokenization on stratified 120-doc samples;
  - records that already fit pass through unchanged (all base semantic
    docs ~2.5KB = ~570 tokens pass through);
  - a single chunk longer than the budget becomes its own 1-chunk record
    (truncation-tolerant text; counted as singleton_oversize);
  - when two adjacent chunks cannot share the budget the head chunk closes
    as a 1-chunk record (counted as singleton_packing; dense domains -
    code ~2.4 B/tok, legal bills ~2.9 B/tok - hit this when the 1200-2200
    char chunk band exceeds ~half the token window);
  - chunk spans are re-based to record-local UTF-8 byte offsets; base
    records keep their legacy chunk_boundaries field shape (1-byte
    separator gaps preserved, final end clamped to the window).

Outputs <out-dir>/train_mix_w1024.jsonl, val_mix_w1024.jsonl and
manifest_mix_w1024.json (per-domain record counts, mean bytes and
estimated tokens per record, boundary-per-record stats, singleton counts,
and the predicted fraction of records with zero in-window boundaries).

stdlib-only. Validate the outputs with
validate_fused_chunker_multidomain_mix.py, then confirm in-window gold
boundaries land with:
  zig build count-fused-tokenization -- --data <sample> \
      --model-dir ~/.cache/modernbert-base --max-seq-len 1024 --json

Usage:
  python3 window_fused_chunker_multidomain_mix.py \
      --out-dir /private/tmp/multidomain_mix [--seq-len 1024]
"""

import argparse
import json
import os
import sys
import time

# bytes per ModernBERT token, measured with count-fused-tokenization
# (--max-seq-len 6144) on stratified 120-doc samples per domain.
DEFAULT_RATIOS = {
    "base": 4.398,
    "code": 2.386,
    "legal": 2.883,
    "financial": 5.080,
    "technical": 3.734,
    "transcripts": 4.078,
}
DEFAULT_SAFETY = 0.97
SPECIAL_TOKEN_SLACK = 4  # CLS/SEP + rounding


class Agg:
    def __init__(self):
        self.records = 0
        self.passthrough = 0
        self.bytes = 0
        self.est_tokens = 0.0
        self.chunks = 0
        self.singleton_oversize = 0
        self.singleton_packing = 0
        self.multi_chunk = 0

    def add(self, n_bytes, n_chunks, ratio, oversize):
        self.records += 1
        self.bytes += n_bytes
        self.est_tokens += n_bytes / ratio
        self.chunks += n_chunks
        if n_chunks >= 2:
            self.multi_chunk += 1
        elif oversize:
            self.singleton_oversize += 1
        else:
            self.singleton_packing += 1

    def summary(self):
        boundaries = self.chunks - self.records
        return {
            "records": self.records,
            "passthrough_records": self.passthrough,
            "mean_bytes_per_record": round(self.bytes / self.records) if self.records else 0,
            "mean_est_tokens_per_record": round(self.est_tokens / self.records) if self.records else 0,
            "mean_chunks_per_record": round(self.chunks / self.records, 2) if self.records else 0,
            "boundaries_per_record": round(boundaries / self.records, 2) if self.records else 0,
            "records_with_boundary": self.multi_chunk,
            "singleton_oversize": self.singleton_oversize,
            "singleton_packing": self.singleton_packing,
            "zero_boundary_record_fraction": (
                round((self.singleton_oversize + self.singleton_packing) / self.records, 4)
                if self.records else 0.0),
        }


def get_spans(rec):
    """(spans, field_name): byte spans from either RawRecord chunk variant."""
    chunks = rec.get("chunks")
    field = "chunks"
    if chunks is None:
        chunks = rec.get("chunk_boundaries") or []
        field = "chunk_boundaries"
    spans = [(c["start_char"], c["end_char"]) for c in chunks]
    return spans, field


def emit(out, rec, field, raw, group, suffix):
    """Write one windowed record; returns its byte length."""
    g_start = group[0][0]
    g_end = min(group[-1][1], len(raw))  # base final end may overshoot by a few bytes
    w_raw = raw[g_start:g_end]
    w_len = len(w_raw)
    new_chunks = []
    for a, b in group:
        new_chunks.append({"start_char": a - g_start,
                           "end_char": min(b - g_start, w_len)})
    new_rec = dict(rec)
    new_rec["text"] = w_raw.decode("utf-8")
    new_rec.pop("chunks", None)
    new_rec.pop("chunk_boundaries", None)
    new_rec[field] = new_chunks
    if "source" in new_rec:
        new_rec["source"] = f"{new_rec['source']}|{suffix}"
    out.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
    return w_len


def window_file(in_path, out_path, ratios, seq_len, safety, aggs):
    budgets = {d: int((seq_len - SPECIAL_TOKEN_SLACK) * r * safety) for d, r in ratios.items()}
    with open(in_path, encoding="utf-8") as f, \
         open(out_path + ".part", "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            domain = rec.get("domain", "base")
            ratio = ratios.get(domain, ratios["base"])
            budget = budgets.get(domain, budgets["base"])
            agg = aggs.setdefault(domain, Agg())
            raw = rec["text"].encode("utf-8")
            spans, field = get_spans(rec)
            if not spans:
                continue
            if len(raw) <= budget:
                # Already fits the token window: pass through unchanged.
                out.write(line + "\n")
                agg.add(len(raw), len(spans), ratio, oversize=False)
                agg.passthrough += 1
                continue
            # Greedy pack consecutive whole chunks under the byte budget.
            # Extent is measured from the group's first byte so base's
            # 1-byte separator gaps stay inside the window.
            group = []
            w_i = 0
            for span in spans:
                if group and min(span[1], len(raw)) - group[0][0] > budget:
                    n = emit(out, rec, field, raw, group, f"w{seq_len}:{w_i}")
                    agg.add(n, len(group), ratio,
                            oversize=len(group) == 1 and (group[0][1] - group[0][0]) > budget)
                    group = []
                    w_i += 1
                group.append(span)
            if group:
                n = emit(out, rec, field, raw, group, f"w{seq_len}:{w_i}")
                agg.add(n, len(group), ratio,
                        oversize=len(group) == 1 and (group[0][1] - group[0][0]) > budget)
    os.replace(out_path + ".part", out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="/private/tmp/multidomain_mix")
    ap.add_argument("--train-in", default="")
    ap.add_argument("--val-in", default="")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--safety", type=float, default=DEFAULT_SAFETY,
                    help="byte budget = (seq_len-4) * bytes_per_token * safety")
    ap.add_argument("--ratios", default="",
                    help='JSON object of bytes-per-token per domain (default: measured)')
    args = ap.parse_args()

    train_in = args.train_in or os.path.join(args.out_dir, "train_mix.jsonl")
    val_in = args.val_in or os.path.join(args.out_dir, "val_mix.jsonl")
    ratios = dict(DEFAULT_RATIOS)
    if args.ratios:
        ratios.update(json.loads(args.ratios))

    results = {}
    for split, in_path in (("train", train_in), ("val", val_in)):
        out_path = os.path.join(args.out_dir, f"{split}_mix_w{args.seq_len}.jsonl")
        aggs = {}
        window_file(in_path, out_path, ratios, args.seq_len, args.safety, aggs)
        results[split] = {
            "input": in_path,
            "output": out_path,
            "domains": {d: a.summary() for d, a in sorted(aggs.items())},
            "total_records": sum(a.records for a in aggs.values()),
        }
        print(f"== {split}: {out_path} ({results[split]['total_records']} records)")
        for d, s in results[split]["domains"].items():
            print(f"  {d:12s} records={s['records']:6d} est_tokens={s['mean_est_tokens_per_record']:5d} "
                  f"chunks/rec={s['mean_chunks_per_record']:5.2f} boundaries/rec={s['boundaries_per_record']:5.2f} "
                  f"zero_boundary_frac={s['zero_boundary_record_fraction']:6.4f} "
                  f"(oversize={s['singleton_oversize']}, packing={s['singleton_packing']})")

    manifest = {
        "generator": "window_fused_chunker_multidomain_mix.py",
        "created_unix": int(time.time()),
        "seq_len": args.seq_len,
        "safety": args.safety,
        "bytes_per_token": ratios,
        "byte_budgets": {d: int((args.seq_len - SPECIAL_TOKEN_SLACK) * r * args.safety)
                         for d, r in ratios.items()},
        "semantics": "cut only at existing chunk boundaries; whole chunks per record; "
                     "records sized so all chunks fit the token window (the Zig loader "
                     "masks chunks that end past max_seq_len and labels boundaries only "
                     "on fully-in-window chunks after the first)",
        "source_manifest": os.path.join(args.out_dir, "manifest_mix.json"),
        "splits": results,
    }
    manifest_path = os.path.join(args.out_dir, f"manifest_mix_w{args.seq_len}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"manifest: {manifest_path}")

    dense = [d for d in ratios
             if results["train"]["domains"].get(d, {}).get("zero_boundary_record_fraction", 0) > 0.5]
    if dense:
        print(f"\nNOTE: high zero-boundary fraction in {dense}: their 1200-2200 char "
              f"chunk band exceeds half the {args.seq_len}-token window at their token "
              f"density; train those at --max-seq-len 2048 or regenerate with a "
              f"token-aware chunk band.", file=sys.stderr)


if __name__ == "__main__":
    main()
