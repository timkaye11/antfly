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
"""Materialize open retrieval datasets for the fused chunker retrieval lane.

Produces, per manifest domain (evals/chunker/manifest.json retrieval_ndcg),
the corpus + queries-with-qrels JSONL that the retrieval benchmark scripts and
zig tools consume (see scripts/run_fused_chunker_retrieval_benchmark.sh):

  <out-dir>/<domain>/corpus.jsonl
      {"document_id": str, "title": str, "text": str}
  <out-dir>/<domain>/queries.jsonl
      {"query_id": str, "text": str, "relevant_ids": [document_id, ...]}
  <out-dir>/<domain>/meta.json
      provenance + record counts

Document and query ids are sanitized to never contain '#' or ':' because
materialize-fused-chunker-retrieval-embeddings mints chunk ids of the form
"<document_id>#chunk:<n>" and the doc-level run collapse splits on '#chunk:'.

Domain -> open dataset mapping (only open/reproducible sources, per the
manifest local_claim_policy). The manifest domains cannot all be matched
exactly by open BEIR-style corpora; closest available substitutes:

  technical_docs -> BEIR scidocs      scientific paper titles+abstracts
                                      (caveat: abstracts, not full manuals)
  web            -> BEIR quora        web community questions
                                      (caveat: short texts, weak chunking signal)
  code           -> BEIR cqadupstack  'programmers' StackExchange subforum
                                      (caveat: programming Q&A prose, not source code)
  medical       -> BEIR nfcorpus      NutritionFacts medical corpus
  conversation  -> LongEmbed qmsum    multi-speaker meeting transcripts
  law           -> mteb/legalbench_corporate_lobbying   US bill texts
                                      (caveat: small; <500 queries total)
  finance       -> BEIR fiqa          financial opinion Q&A
  long_context  -> LongEmbed narrativeqa  book/script-length documents
                                      (caveat: docs truncated at --max-doc-chars)

Laptop-friendly caps (this is a laptop, not a cluster): each domain is
subsampled deterministically (--seed) to at most --max-docs documents
(all judged-relevant docs for the selected queries + random distractors)
and --max-queries queries; documents are truncated at --max-doc-chars.

stdlib-only; no pip installs. Sources are fetched with urllib:
  - BEIR zips from public.ukp.informatik.tu-darmstadt.de (as released by BEIR)
  - LongEmbed (dwzhu/LongEmbed) files via huggingface.co resolve URLs
  - MTEB law rows via the HuggingFace datasets-server rows API
    (same approach as gopeft scripts/generate_chonky_eval_datasets.py)

Usage:
  python3 generate_fused_chunker_retrieval_datasets.py \
      --out-dir /private/tmp/fused_retrieval_datasets \
      --domains medical,finance
"""

import argparse
import gzip
import io
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile

BEIR_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
HF_BASE = "https://huggingface.co"
HF_ROWS_BASE = "https://datasets-server.huggingface.co"

DEFAULT_MAX_DOCS = 5000
DEFAULT_MAX_QUERIES = 500
DEFAULT_MAX_DOC_CHARS = 60000

DOMAINS = {
    "technical_docs": {
        "loader": "beir_zip",
        "dataset": "scidocs",
        "notes": "scientific paper titles+abstracts; not full technical manuals",
    },
    "web": {
        "loader": "beir_zip",
        "dataset": "quora",
        "notes": "short web community questions; weak chunking signal",
    },
    "code": {
        "loader": "beir_zip",
        "dataset": "cqadupstack",
        "inner": "programmers",
        "notes": "programming Q&A prose, not source code",
    },
    "medical": {
        "loader": "beir_zip",
        "dataset": "nfcorpus",
        "notes": "",
    },
    "conversation": {
        "loader": "longembed",
        "dataset": "qmsum",
        "max_doc_chars": 200000,
        "notes": "meeting transcripts; long multi-speaker documents",
    },
    "law": {
        "loader": "hf_rows",
        "dataset": "mteb/legalbench_corporate_lobbying",
        "notes": "US bill texts; small dataset (<500 queries)",
    },
    "finance": {
        "loader": "beir_zip",
        "dataset": "fiqa",
        "notes": "",
    },
    "long_context": {
        "loader": "longembed",
        "dataset": "narrativeqa",
        "max_docs": 300,
        "max_queries": 300,
        "max_doc_chars": 200000,
        "notes": "book/script-length documents, truncated at max-doc-chars",
    },
}


def sanitize_id(raw):
    """Chunk ids are '<doc>#chunk:<n>'; source ids must not collide."""
    return re.sub(r"[#:]", "_", str(raw))


def http_get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "antfly-fused-chunker-eval/1.0"})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 30 * (attempt + 1)
                print(f"  429 rate-limited on {url}, sleeping {wait}s", file=sys.stderr)
                time.sleep(wait)
            elif e.code >= 500 and attempt < 7:
                time.sleep(5 * (attempt + 1))
            else:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 7:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"kept failing to fetch {url}")


def download_file(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  cached {dest}", file=sys.stderr)
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  downloading {url}", file=sys.stderr)
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "antfly-fused-chunker-eval/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as f:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            f.write(block)
    os.replace(tmp, dest)
    print(f"  saved {dest} ({os.path.getsize(dest)} bytes)", file=sys.stderr)
    return dest


def pick(record, *keys):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def compose_text(title, body):
    title = (title or "").strip()
    body = (body or "").strip()
    if title and body:
        return title + "\n\n" + body
    return title or body


def iter_jsonl(lines):
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        line = line.strip()
        if line:
            yield json.loads(line)


def parse_qrels_tsv(lines):
    """query-id \\t corpus-id \\t score (BEIR); header tolerated."""
    out = []
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        qid, did, score = parts[0], parts[1], parts[2]
        try:
            score_val = float(score)
        except ValueError:
            continue  # header row
        if score_val > 0:
            out.append((sanitize_id(qid), sanitize_id(did)))
    return out


class DomainBuilder:
    """Select queries + relevant docs + distractors under deterministic caps."""

    def __init__(self, domain, source, seed, max_docs, max_queries, max_doc_chars):
        self.domain = domain
        self.source = source
        self.rng = random.Random(seed)
        self.max_docs = max_docs
        self.max_queries = max_queries
        self.max_doc_chars = max_doc_chars
        self.qrels = {}  # qid -> set(doc ids), positives only
        self.query_text = {}  # qid -> text
        self.truncated_docs = 0

    def add_qrel(self, qid, did):
        self.qrels.setdefault(qid, set()).add(did)

    def add_query(self, qid, text):
        if text and text.strip():
            self.query_text[qid] = text.strip()

    def select_queries(self):
        eligible = sorted(q for q in self.qrels if q in self.query_text)
        if not eligible:
            raise RuntimeError(f"{self.domain}: no queries with both text and positive qrels")
        self.rng.shuffle(eligible)
        selected = eligible[: self.max_queries]
        # Ensure the union of positives fits within the doc cap.
        while selected:
            relevant = set()
            for qid in selected:
                relevant.update(self.qrels[qid])
            if len(relevant) <= self.max_docs:
                break
            selected = selected[: max(1, len(selected) - max(1, len(selected) // 10))]
        return selected, relevant

    def build(self, corpus_iter, out_dir):
        """corpus_iter yields (doc_id, title, text). Streams once."""
        selected, relevant = self.select_queries()
        kept = {}  # doc_id -> (title, text)
        distractors = []  # reservoir of (doc_id, title, text)
        seen_distractors = 0
        corpus_total = 0
        for did, title, text in corpus_iter:
            corpus_total += 1
            text = compose_text(title, text)
            if not text:
                continue
            if len(text) > self.max_doc_chars:
                text = text[: self.max_doc_chars]
                self.truncated_docs += 1
            if did in relevant:
                kept[did] = (title or "", text)
            elif did not in kept:
                seen_distractors += 1
                budget = self.max_docs  # reservoir over non-relevant docs
                if len(distractors) < budget:
                    distractors.append((did, title or "", text))
                else:
                    slot = self.rng.randrange(seen_distractors)
                    if slot < budget:
                        distractors[slot] = (did, title or "", text)
        relevant_missing = len(relevant) - len(kept)
        relevant_kept = len(kept)
        fill = self.max_docs - len(kept)
        for did, title, text in distractors[: max(0, fill)]:
            if did not in kept:
                kept[did] = (title, text)

        queries = []
        for qid in selected:
            ids = sorted(d for d in self.qrels[qid] if d in kept)
            if ids:
                queries.append({
                    "query_id": qid,
                    "text": self.query_text[qid],
                    "relevant_ids": ids,
                })
        if not queries:
            raise RuntimeError(f"{self.domain}: no queries survived corpus subsampling")

        os.makedirs(out_dir, exist_ok=True)
        corpus_path = os.path.join(out_dir, "corpus.jsonl")
        queries_path = os.path.join(out_dir, "queries.jsonl")
        with open(corpus_path, "w", encoding="utf-8") as f:
            for did in sorted(kept):
                title, text = kept[did]
                f.write(json.dumps(
                    {"document_id": did, "title": title, "text": text},
                    ensure_ascii=False) + "\n")
        with open(queries_path, "w", encoding="utf-8") as f:
            for query in queries:
                f.write(json.dumps(query, ensure_ascii=False) + "\n")
        meta = {
            "domain": self.domain,
            "source": self.source,
            "documents": len(kept),
            "queries": len(queries),
            "relevant_docs_kept": relevant_kept,
            "relevant_docs_missing_from_corpus": relevant_missing,
            "corpus_total_scanned": corpus_total,
            "truncated_docs": self.truncated_docs,
            "caps": {
                "max_docs": self.max_docs,
                "max_queries": self.max_queries,
                "max_doc_chars": self.max_doc_chars,
            },
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")
        print(f"  {self.domain}: docs={meta['documents']} queries={meta['queries']} "
              f"truncated={self.truncated_docs} missing_relevant={relevant_missing}")
        return meta


# --- BEIR zip loader ---------------------------------------------------------

def load_beir_zip(builder, raw_dir, dataset, inner=None):
    zip_path = download_file(f"{BEIR_BASE}/{dataset}.zip",
                             os.path.join(raw_dir, f"{dataset}.zip"))
    zf = zipfile.ZipFile(zip_path)
    prefix = f"{dataset}/{inner}/" if inner else f"{dataset}/"

    def member(name):
        path = prefix + name
        if path in zf.namelist():
            return path
        matches = [m for m in zf.namelist() if m.endswith("/" + name) and (inner or "") in m]
        if not matches:
            raise RuntimeError(f"{dataset}: no member matching {name} (inner={inner})")
        return sorted(matches)[0]

    qrels_name = None
    for split in ("qrels/test.tsv", "qrels/dev.tsv", "qrels/train.tsv"):
        try:
            qrels_name = member(split)
            break
        except RuntimeError:
            continue
    if qrels_name is None:
        raise RuntimeError(f"{dataset}: no qrels tsv found")

    with zf.open(qrels_name) as f:
        for qid, did in parse_qrels_tsv(io.TextIOWrapper(f, encoding="utf-8")):
            builder.add_qrel(qid, did)
    with zf.open(member("queries.jsonl")) as f:
        for record in iter_jsonl(io.TextIOWrapper(f, encoding="utf-8")):
            builder.add_query(sanitize_id(pick(record, "_id", "id", "query_id")),
                              pick(record, "text", "query"))

    def corpus_iter():
        with zf.open(member("corpus.jsonl")) as f:
            for record in iter_jsonl(io.TextIOWrapper(f, encoding="utf-8")):
                yield (sanitize_id(pick(record, "_id", "id", "document_id")),
                       pick(record, "title"),
                       pick(record, "text", "body"))

    return corpus_iter


# --- LongEmbed (dwzhu/LongEmbed) loader --------------------------------------

def hf_tree(repo, path=""):
    url = f"{HF_BASE}/api/datasets/{repo}/tree/main"
    if path:
        url += f"/{path}"
    return json.loads(http_get(url))


def load_longembed(builder, raw_dir, task):
    repo = "dwzhu/LongEmbed"
    entries = hf_tree(repo, task)
    files = {os.path.basename(e["path"]): e["path"] for e in entries if e.get("type") == "file"}

    def fetch(kind):
        candidates = [n for n in files if kind in n.lower()]
        if not candidates:
            raise RuntimeError(f"LongEmbed/{task}: no '{kind}' file among {sorted(files)}")
        name = sorted(candidates)[0]
        gz = name.endswith(".gz")
        dest = os.path.join(raw_dir, f"longembed_{task}_{name}")
        download_file(f"{HF_BASE}/datasets/{repo}/resolve/main/{files[name]}", dest)
        if name.endswith(".jsonl") or name.endswith(".jsonl.gz") or name.endswith(".json"):
            opener = gzip.open if gz else open
            with opener(dest, "rt", encoding="utf-8") as f:
                return list(iter_jsonl(f))
        if name.endswith(".tsv") or name.endswith(".tsv.gz"):
            opener = gzip.open if gz else open
            with opener(dest, "rt", encoding="utf-8") as f:
                return [("tsv", line) for line in f]
        raise RuntimeError(f"LongEmbed/{task}: unsupported file format {name}")

    qrels_rows = fetch("qrels")
    for row in qrels_rows:
        if isinstance(row, tuple):  # tsv
            for qid, did in parse_qrels_tsv([row[1]]):
                builder.add_qrel(qid, did)
        else:
            qid = pick(row, "qid", "query-id", "query_id", "q_id")
            did = pick(row, "doc_id", "corpus-id", "corpus_id", "pid", "did", "docid")
            score = pick(row, "score", "relevance")
            if qid is None or did is None:
                raise RuntimeError(f"LongEmbed/{task}: unrecognized qrels keys {sorted(row)}")
            if score is None or float(score) > 0:
                builder.add_qrel(sanitize_id(qid), sanitize_id(did))
    for row in fetch("queries"):
        builder.add_query(sanitize_id(pick(row, "_id", "id", "qid", "query_id")),
                          pick(row, "text", "query"))
    corpus_rows = fetch("corpus")

    def corpus_iter():
        for row in corpus_rows:
            yield (sanitize_id(pick(row, "_id", "id", "pid", "doc_id", "document_id")),
                   pick(row, "title"),
                   pick(row, "text", "passage", "body"))

    return corpus_iter


# --- HuggingFace datasets-server rows loader (MTEB-style repos) ---------------

def hf_rows(dataset, config, split):
    rows = []
    offset = 0
    while True:
        url = (f"{HF_ROWS_BASE}/rows?dataset={urllib.request.quote(dataset, safe='')}"
               f"&config={urllib.request.quote(config, safe='')}"
               f"&split={urllib.request.quote(split, safe='')}"
               f"&offset={offset}&length=100")
        data = json.loads(http_get(url))
        page = data.get("rows", [])
        if not page:
            break
        for r in page:
            rows.append(r["row"])
        offset += len(page)
        if offset >= data.get("num_rows_total", 0):
            break
    return rows


def hf_splits(dataset):
    url = f"{HF_ROWS_BASE}/splits?dataset={urllib.request.quote(dataset, safe='')}"
    return json.loads(http_get(url)).get("splits", [])


def load_hf_rows(builder, raw_dir, dataset):
    del raw_dir
    splits = hf_splits(dataset)
    by_config = {}
    for s in splits:
        by_config.setdefault(s["config"], []).append(s["split"])

    def find_config(*names):
        for name in names:
            for config in by_config:
                if config.lower() == name:
                    return config
        return None

    corpus_config = find_config("corpus")
    queries_config = find_config("queries")
    qrels_config = find_config("qrels", "default")
    if not (corpus_config and queries_config and qrels_config):
        raise RuntimeError(f"{dataset}: expected corpus/queries/qrels configs, got {sorted(by_config)}")

    def first_split(config):
        preferred = [s for s in by_config[config] if s in ("test", "train", "corpus", "queries", "default")]
        return (preferred or by_config[config])[0]

    for row in hf_rows(dataset, qrels_config, first_split(qrels_config)):
        qid = pick(row, "query-id", "query_id", "qid")
        did = pick(row, "corpus-id", "corpus_id", "doc_id", "did")
        score = pick(row, "score", "relevance")
        if qid is None or did is None:
            raise RuntimeError(f"{dataset}: unrecognized qrels keys {sorted(row)}")
        if score is None or float(score) > 0:
            builder.add_qrel(sanitize_id(qid), sanitize_id(did))
    for row in hf_rows(dataset, queries_config, first_split(queries_config)):
        builder.add_query(sanitize_id(pick(row, "_id", "id", "query_id", "qid")),
                          pick(row, "text", "query"))
    corpus_rows = hf_rows(dataset, corpus_config, first_split(corpus_config))

    def corpus_iter():
        for row in corpus_rows:
            yield (sanitize_id(pick(row, "_id", "id", "doc_id", "document_id")),
                   pick(row, "title"),
                   pick(row, "text", "body"))

    return corpus_iter


LOADERS = {
    "beir_zip": lambda builder, raw_dir, spec: load_beir_zip(
        builder, raw_dir, spec["dataset"], spec.get("inner")),
    "longembed": lambda builder, raw_dir, spec: load_longembed(
        builder, raw_dir, spec["dataset"]),
    "hf_rows": lambda builder, raw_dir, spec: load_hf_rows(
        builder, raw_dir, spec["dataset"]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="/private/tmp/fused_retrieval_datasets")
    ap.add_argument("--domains", default=",".join(DOMAINS),
                    help="comma-separated manifest domains (default: all)")
    ap.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS)
    ap.add_argument("--max-queries", type=int, default=DEFAULT_MAX_QUERIES)
    ap.add_argument("--max-doc-chars", type=int, default=DEFAULT_MAX_DOC_CHARS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="rebuild domains even if corpus.jsonl already exists")
    args = ap.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    unknown = [d for d in domains if d not in DOMAINS]
    if unknown:
        ap.error(f"unknown domains {unknown}; known: {sorted(DOMAINS)}")

    raw_dir = os.path.join(args.out_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)
    failures = {}
    for domain in domains:
        spec = DOMAINS[domain]
        out_dir = os.path.join(args.out_dir, domain)
        if not args.force and os.path.exists(os.path.join(out_dir, "corpus.jsonl")):
            print(f"== {domain}: already materialized at {out_dir} (use --force to rebuild)")
            continue
        print(f"== {domain} <- {spec['loader']}:{spec['dataset']}"
              + (f"/{spec['inner']}" if spec.get("inner") else ""))
        builder = DomainBuilder(
            domain=domain,
            source=f"{spec['loader']}:{spec['dataset']}" + (f"/{spec['inner']}" if spec.get("inner") else ""),
            seed=args.seed,
            max_docs=spec.get("max_docs", args.max_docs),
            max_queries=spec.get("max_queries", args.max_queries),
            max_doc_chars=spec.get("max_doc_chars", args.max_doc_chars),
        )
        try:
            corpus_iter = LOADERS[spec["loader"]](builder, raw_dir, spec)
            builder.build(corpus_iter(), out_dir)
        except Exception as e:  # noqa: BLE001 - report per-domain, fail at end
            failures[domain] = f"{type(e).__name__}: {e}"
            print(f"  FAILED {domain}: {failures[domain]}", file=sys.stderr)

    if failures:
        print("\ndomain failures:", file=sys.stderr)
        for domain, err in failures.items():
            print(f"  {domain}: {err}", file=sys.stderr)
        sys.exit(1)
    print("\nall requested domains materialized under " + args.out_dir)


if __name__ == "__main__":
    main()
