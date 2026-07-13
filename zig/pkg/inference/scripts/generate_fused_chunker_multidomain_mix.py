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
"""Materialize the multi-domain fused-chunker boundary training mix.

Closes the MTCB gap (learned chunker R@10 0.736 size-matched vs chonkie
Recursive-2048 0.808) diagnosed as training-data diversity: the boundary
model was trained on wiki-style semantic labels only, while MTCB spans
code/filings/tables/lectures with structure-aligned relevance.

Produces, in the exact RawRecord schema the Zig trainer consumes
(src/finetune/fused_chunker_data.zig):

  <out-dir>/train_mix.jsonl, <out-dir>/val_mix.jsonl
      {"text": str, "chunks": [{"start_char": int, "end_char": int}, ...],
       "domain": str, "source": str}
      start_char/end_char are UTF-8 BYTE offsets (matching the existing
      fused_train.jsonl convention where the final end_char equals the
      UTF-8 byte length of text). New-domain chunks are contiguous,
      non-overlapping, and cover [0, byte_len) exactly. Base records are
      passed through unchanged (they use the `chunk_boundaries` field
      variant with 1-byte separator gaps; the parser accepts both).
  <out-dir>/manifest_mix.json
      per-domain doc counts, mean doc/chunk sizes, boundary rates,
      source provenance, and contamination exclusions.

Domains (boundary labels are deterministic/structural or pre-existing
gold only - no LLM labeling):

  base        gopeft fused_train.jsonl semantic docs, passed through
              unchanged; val docs sampled from fused_val.jsonl.
  code        GitHub repo tarballs pinned at release tags (codeload):
              markdown headers for .md docs/READMEs; top-level
              def/class (py), func/type (go), function/class/export
              (js/ts) declarations for source files.
  legal       FiscalNote/billsum US bill texts ("SEC. n." headers) +
              rcds/MultiLegalSBD en_judgements (BVA decisions, gold
              paragraph structure from blank-line layout).
  financial   eloukas/edgar-corpus SEC 10-K filings with pre-extracted
              "Item N" sections (section starts are the boundaries).
  technical   neuralwork/arxiver scientific papers in markdown
              (# section headers).
  transcripts MediaSum (nbroad/mediasum) NPR/CNN interview transcripts;
              boundaries at speaker-turn starts. (MIT OCW is inside
              MTCB - deliberately NOT used.)

Structural boundaries are merged to the 1200-2200 char band (max ~4000
split) because MTCB rewards Recursive-2048-scale chunks, then long
sources are windowed into 2..6-chunk training docs.

CONTAMINATION: MTCB's own corpora (chonkie-ai/{gacha,ficha,macha,cocha,
tacha,sencha,hojicha,ryokucha,genmaicha}, served under their canonical
feyninc/* names by the HF datasets-server) are downloaded and any
candidate doc sharing >= `--contamination-hits` normalized 64-char
shingles with any benchmark doc is dropped. Gutenberg literature, CUAD,
TAT-QA, QASPER, NICE/CDC/WHO guidelines, and MIT OCW are additionally
excluded by construction (never sourced).

stdlib-only; no pip installs. Sources are fetched with urllib:
  - HF file downloads via huggingface.co resolve URLs
  - HF datasets-server rows API (same approach as
    generate_fused_chunker_retrieval_datasets.py)
  - GitHub tag tarballs via codeload.github.com

Usage:
  python3 generate_fused_chunker_multidomain_mix.py \
      --out-dir /private/tmp/multidomain_mix \
      --base /Users/tim/Documents/af/gopeft/data/fused_train.jsonl \
      --base-val /Users/tim/Documents/af/gopeft/data/fused_val.jsonl
  # then validate:
  python3 validate_fused_chunker_multidomain_mix.py \
      --train /private/tmp/multidomain_mix/train_mix.jsonl \
      --val /private/tmp/multidomain_mix/val_mix.jsonl
"""

import argparse
import hashlib
import json
import lzma
import os
import random
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request

HF_BASE = "https://huggingface.co"
HF_ROWS_BASE = "https://datasets-server.huggingface.co"
USER_AGENT = "antfly-fused-chunker-data-mix/1.0"

# Size band: MTCB rewards Recursive-2048-scale chunks. The char band is
# derived per domain from a TOKEN band so that two whole chunks fit a
# 1024-token training window (the Zig loader only labels boundaries on
# chunks that fit entirely inside max_seq_len): chars = tokens x
# bytes-per-ModernBERT-token measured with count-fused-tokenization.
# For ~4.1-5.1 B/tok prose this reproduces the 1200-2200 char band; dense
# domains (code ~2.4, legal bills ~2.9, LaTeX-markdown ~3.7) get
# proportionally smaller char bands at the same token scale.
# Band chosen so TWO whole chunks always share a 1024-token window with
# margin (band_merge fills toward band_max, so chunks land ~340-400
# tokens and pairs ~700-800 << ~990 usable):
TOKEN_BAND_MIN = 260
TOKEN_BAND_MAX = 420
TOKEN_HARD_MAX = 800
BYTES_PER_TOKEN = {
    "base": 4.398,
    "code": 2.386,
    "legal": 2.883,
    "financial": 5.080,
    "technical": 3.734,
    "transcripts": 4.078,
}

def domain_band(domain):
    r = BYTES_PER_TOKEN[domain]
    return (int(TOKEN_BAND_MIN * r), int(TOKEN_BAND_MAX * r), int(TOKEN_HARD_MAX * r))

# Legacy prose-scale defaults (band_merge callers without a domain).
BAND_MIN = 1200
BAND_MAX = 2200
HARD_MAX = 4000
# Training docs: 2..6 chunks, roughly base-file scale and beyond.
DOC_MIN_CHARS = 2500
DOC_MAX_CHARS = 9500
DOC_MAX_CHUNKS = 6

# MTCB benchmark corpora to exclude (canonical chonkie-ai names; the hub
# /parquet API resolves them to the mirror the datasets-server indexes).
MTCB_CORPORA = [
    "gacha", "ficha", "macha", "cocha", "tacha",
    "sencha", "hojicha", "ryokucha", "genmaicha",
]

# (repo, tag, langs) - pinned release tags for determinism. Small repos
# first so every repo contributes before the domain doc target fills.
CODE_REPOS = [
    ("pallets/flask", "3.0.3", ("py", "md")),
    ("psf/requests", "v2.32.3", ("py", "md")),
    ("gin-gonic/gin", "v1.10.0", ("go", "md")),
    ("expressjs/express", "4.19.2", ("js", "md")),
    ("mkdocs/mkdocs", "1.6.0", ("py", "md")),
    ("vuejs/core", "v3.4.31", ("js", "md")),
    ("facebook/react", "v18.3.1", ("js", "md")),
    ("fastapi/fastapi", "0.111.0", ("py", "md")),
    ("prometheus/prometheus", "v2.53.0", ("go", "md")),
    ("gohugoio/hugo", "v0.128.0", ("go", "md")),
    ("django/django", "5.0.6", ("py", "md")),
    ("kubernetes/kubernetes", "v1.30.2", ("go", "md")),
    ("golang/go", "go1.22.4", ("go", "md")),
]

EXT_LANG = {
    ".py": "py", ".go": "go",
    ".js": "js", ".jsx": "js", ".ts": "js", ".tsx": "js",
    ".md": "md", ".markdown": "md",
}

SKIP_PATH_PARTS = (
    "vendor/", "third_party/", "node_modules/", "testdata/", "test/fixtures/",
    "locale/", "translations/", ".github/", "changelogs/",
)

GENERATED_MARKERS = ("Code generated by", "DO NOT EDIT", "@generated", "auto-generated")


# --- HTTP helpers (style of generate_fused_chunker_retrieval_datasets.py) ----

def hf_token():
    """Optional HF token (env or standard cache path); raises rows-API limits."""
    tok = os.environ.get("HF_TOKEN", "")
    if not tok:
        path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(path):
            with open(path) as f:
                tok = f.read().strip()
    return tok


_HF_TOKEN = None

def _headers(url):
    global _HF_TOKEN
    headers = {"User-Agent": USER_AGENT}
    if "huggingface.co" in url:
        if _HF_TOKEN is None:
            _HF_TOKEN = hf_token()
        if _HF_TOKEN:
            headers["Authorization"] = f"Bearer {_HF_TOKEN}"
    return headers


def http_get(url, timeout=180):
    req = urllib.request.Request(url, headers=_headers(url))
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = e.headers.get("Retry-After") if e.headers else None
                wait = int(retry_after) if retry_after and retry_after.isdigit() \
                    else 20 * (attempt + 1)
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
    req = urllib.request.Request(url, headers=_headers(url))
    with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as f:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            f.write(block)
    os.replace(tmp, dest)
    print(f"  saved {dest} ({os.path.getsize(dest)} bytes)", file=sys.stderr)
    return dest


def try_pyarrow():
    """Optional parquet fast-path (e.g. venv under /private/tmp with pyarrow).

    The datasets-server rows API rate-limits bulk crawls hard (429s even
    authenticated); the auto-converted parquet shards on the hub CDN do not.
    Everything still works stdlib-only via the rows API - just slower.
    """
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
        return pq
    except ImportError:
        return None


def hf_rows_from_parquet(pq, raw_dir, dataset, config, split, max_rows, cache):
    urls = json.loads(http_get(f"{HF_BASE}/api/datasets/{dataset}/parquet"))[config][split]
    rows, exhausted = [], True
    tmp = cache + ".rebuild"
    with open(tmp, "w", encoding="utf-8") as out:
        for i, url in enumerate(urls):
            if len(rows) >= max_rows:
                exhausted = False
                break
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{dataset}__{config}__{split}__{i}")
            shard = download_file(url, os.path.join(raw_dir, "parquet", safe + ".parquet"))
            pf = pq.ParquetFile(shard)
            for batch in pf.iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    rows.append(row)
                if len(rows) >= max_rows:
                    exhausted = False
                    break
    os.replace(tmp, cache)
    if exhausted:
        with open(cache + ".done", "w") as f:
            f.write("exhausted\n")
    return rows[:max_rows]


def hf_rows_cached(raw_dir, dataset, config, split, max_rows, page=100):
    """Fetch up to max_rows rows, cached as JSONL.

    Prefers parquet shards from the hub CDN when pyarrow is importable;
    falls back to the datasets-server rows API (stdlib-only) with adaptive
    page size: on truncated cells the page is refetched smaller so long
    documents (SEC filings, papers, transcripts) come back whole.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{dataset}__{config}__{split}")
    cache = os.path.join(raw_dir, "hf_rows", safe + ".jsonl")
    done_marker = cache + ".done"  # split exhausted; never re-probe the server
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    rows = []
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    if len(rows) >= max_rows or os.path.exists(done_marker):
        return rows[:max_rows]
    pq = try_pyarrow()
    if pq is not None:
        return hf_rows_from_parquet(pq, raw_dir, dataset, config, split, max_rows, cache)
    exhausted = False
    with open(cache, "a", encoding="utf-8") as out:
        offset = len(rows)
        while len(rows) < max_rows:
            length = min(page, max_rows - len(rows))
            got = None
            while True:
                url = (f"{HF_ROWS_BASE}/rows?dataset={urllib.request.quote(dataset, safe='')}"
                       f"&config={urllib.request.quote(config, safe='')}"
                       f"&split={urllib.request.quote(split, safe='')}"
                       f"&offset={offset}&length={length}")
                data = json.loads(http_get(url))
                page_rows = data.get("rows", [])
                if not page_rows:
                    got = []
                    exhausted = True
                    break
                if any(r.get("truncated_cells") for r in page_rows) and length > 1:
                    # Shrink for the rest of this source, not just this page.
                    length = max(1, length // 4)
                    page = length
                    continue
                got = [r["row"] for r in page_rows]
                total = data.get("num_rows_total", 0)
                break
            if not got:
                break
            for row in got:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.extend(got)
            offset += len(got)
            time.sleep(0.7)  # pace the datasets-server; it 429s on bursts
            if total and offset >= total:
                exhausted = True
                break
    if exhausted:
        with open(done_marker, "w") as f:
            f.write("exhausted\n")
    return rows[:max_rows]


# --- size-band merge + windowing ---------------------------------------------

def best_break(text, lo, hi, want):
    """Best split position in [lo, hi) near `want`: para > line > sentence > space."""
    window = text[max(lo, want - 600):min(hi, want + 600)]
    base = max(lo, want - 600)
    for pattern in (r"\n\s*\n", r"\n", r"(?<=[.!?])\s", r"\s"):
        candidates = [base + m.end() for m in re.finditer(pattern, window)]
        candidates = [c for c in candidates if lo < c < hi]
        if candidates:
            return min(candidates, key=lambda c: abs(c - want))
    return want if lo < want < hi else (lo + hi) // 2


def band_merge(text, cuts, band_min=BAND_MIN, band_max=BAND_MAX, hard_max=HARD_MAX):
    """Merge structural cut points into contiguous chunk spans in the size band.

    cuts: sorted char offsets of structural boundaries (0 < cut < len(text)).
    Returns [(start, end), ...] contiguous, covering [0, len(text)).
    """
    n = len(text)
    if n == 0:
        return []
    edges = [0] + [c for c in sorted(set(cuts)) if 0 < c < n] + [n]
    # Split oversize sections at the best internal break.
    sections = []
    stack = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)][::-1]
    while stack:
        a, b = stack.pop()
        if b - a > hard_max:
            mid = best_break(text, a + band_min // 2, b - band_min // 2, a + (band_min + band_max) // 2)
            if a < mid < b:
                stack.append((mid, b))
                stack.append((a, mid))
                continue
        sections.append((a, b))
    sections.sort()
    # Greedy merge into the band.
    spans = []
    cur_start, cur_end = sections[0]
    for a, b in sections[1:]:
        cur_len = cur_end - cur_start
        if cur_len >= band_min and (cur_len + (b - a) > band_max or cur_len >= band_max):
            spans.append((cur_start, cur_end))
            cur_start, cur_end = a, b
        else:
            cur_end = b
    spans.append((cur_start, cur_end))
    # Fold a tiny trailing chunk into its predecessor.
    if len(spans) >= 2 and spans[-1][1] - spans[-1][0] < band_min // 2:
        spans[-2] = (spans[-2][0], spans[-1][1])
        spans.pop()
    # Enforce a hard chunk cap (merging small sections can overshoot
    # band_max by up to one section): re-split oversize chunks at the best
    # internal break so that any two adjacent chunks fit a ~1024-token
    # window together (see window_fused_chunker_multidomain_mix.py).
    cap = int(band_max * 1.15)
    capped = []
    stack = spans[::-1]
    while stack:
        a, b = stack.pop()
        if b - a > cap:
            mid = best_break(text, a + band_min // 2, b - band_min // 2, (a + b) // 2)
            if a < mid < b:
                stack.append((mid, b))
                stack.append((a, mid))
                continue
        capped.append((a, b))
    capped.sort()
    return capped


def window_spans(spans, max_doc_chars=DOC_MAX_CHARS, max_chunks=DOC_MAX_CHUNKS):
    """Pack chunk spans into windows of 2..max_chunks chunks, <= max_doc_chars."""
    windows = []
    cur = []
    for span in spans:
        if cur and (len(cur) >= max_chunks or span[1] - cur[0][0] > max_doc_chars):
            windows.append(cur)
            cur = []
        cur.append(span)
    if cur:
        windows.append(cur)
    # A trailing 1-chunk window folds into the previous window.
    if len(windows) >= 2 and len(windows[-1]) == 1:
        windows[-2].extend(windows.pop())
    return [w for w in windows if len(w) >= 2]


def make_records(doc_id, text, cuts, domain, source, max_windows):
    """band-merge + window one source document into training records."""
    if "\x00" in text:
        return []  # binary junk; offsets would desync
    spans = band_merge(text, cuts, *domain_band(domain))
    windows = window_spans(spans)
    if not windows:
        return []
    if len(windows) > max_windows:
        # Deterministic, evenly spaced selection across the document.
        idx = sorted({round(i * (len(windows) - 1) / (max_windows - 1)) for i in range(max_windows)})
        windows = [windows[i] for i in idx]
    records = []
    for w_i, window in enumerate(windows):
        w_start, w_end = window[0][0], window[-1][1]
        w_text = text[w_start:w_end]
        chunks = []
        byte_off = 0
        for a, b in window:
            blen = len(text[a:b].encode("utf-8"))
            chunks.append({"start_char": byte_off, "end_char": byte_off + blen})
            byte_off += blen
        records.append({
            "text": w_text,
            "chunks": chunks,
            "domain": domain,
            "source": f"{source}#{doc_id}#w{w_i}",
        })
    return records


# --- contamination (MTCB corpora shingle index) --------------------------------

_WS_RE = re.compile(r"\s+")

def normalize_for_shingles(text):
    return _WS_RE.sub(" ", text.lower()).strip()


def shingle_hashes(text, size=64, stride=32):
    norm = normalize_for_shingles(text)
    out = set()
    for i in range(0, max(1, len(norm) - size + 1), stride):
        out.add(hashlib.blake2b(norm[i:i + size].encode("utf-8"), digest_size=8).digest())
    return out


def resolve_mtcb_dataset(name):
    """chonkie-ai/<name> -> dataset id the datasets-server indexes."""
    data = json.loads(http_get(f"{HF_BASE}/api/datasets/chonkie-ai/{name}/parquet"))
    first = data["corpus"]["train"][0]
    m = re.search(r"/api/datasets/([^/]+/[^/]+)/parquet", first)
    if not m:
        raise RuntimeError(f"cannot resolve parquet mirror for chonkie-ai/{name}: {first}")
    return m.group(1)


def build_exclusion_index(raw_dir):
    """Shingle index over every MTCB benchmark corpus document."""
    index = set()
    provenance = {}
    for name in MTCB_CORPORA:
        mirror = resolve_mtcb_dataset(name)
        rows = hf_rows_cached(raw_dir, mirror, "corpus", "train", max_rows=5000, page=5)
        n_sh = 0
        for row in rows:
            text = row.get("text") or ""
            sh = shingle_hashes(text)
            n_sh += len(sh)
            index |= sh
        provenance[f"chonkie-ai/{name}"] = {"mirror": mirror, "docs": len(rows), "shingles": n_sh}
        print(f"  exclusion: chonkie-ai/{name} ({mirror}) docs={len(rows)} shingles={n_sh}",
              file=sys.stderr)
    return index, provenance


def contaminated(text, index, min_hits):
    hits = 0
    for h in shingle_hashes(text):
        if h in index:
            hits += 1
            if hits >= min_hits:
                return True
    return False


# --- domain builders -----------------------------------------------------------
# Each yields (doc_id, text, cuts, source_tag, max_windows).

MD_HEADER_RE = re.compile(r"^#{1,6}\s+\S")
FENCE_RE = re.compile(r"^(```|~~~)")

def markdown_cuts(text):
    cuts, in_fence, pos = [], False, 0
    for line in text.splitlines(keepends=True):
        if FENCE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence and MD_HEADER_RE.match(line):
            cuts.append(pos)
        pos += len(line)
    return [c for c in cuts if c > 0]


SOURCE_CUT_RES = {
    "py": re.compile(r"^(?:def |class |@\w|async def )"),
    "go": re.compile(r"^(?:func |type \w+ (?:struct|interface)\b)"),
    "js": re.compile(
        r"^(?:export\s+)?(?:default\s+)?(?:abstract\s+)?"
        r"(?:async\s+)?(?:function\b|class\b|interface\b|"
        r"(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?(?:\(|function\b))"),
}

def source_cuts(text, lang):
    pat = SOURCE_CUT_RES[lang]
    cuts, pos = [], 0
    for line in text.splitlines(keepends=True):
        if pat.match(line):
            cuts.append(pos)
        pos += len(line)
    return [c for c in cuts if c > 0]


def usable_text(data):
    if b"\x00" in data:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    head = text[:400]
    if any(m in head for m in GENERATED_MARKERS):
        return None
    lines = text.splitlines()
    if not lines:
        return None
    if max(len(l) for l in lines) > 800:  # minified/embedded blobs
        return None
    return text


def build_code(raw_dir, caps):
    # Generous caps: the MTCB shingle filter drops a sizable share of code
    # docs afterwards (shared license boilerplate matches macha/cocha docs).
    md_cap_per_repo = caps.get("code_md_per_repo", 320)
    src_cap_per_repo = caps.get("code_src_per_repo", 700)
    for repo, tag, langs in CODE_REPOS:
        safe = repo.replace("/", "__") + "__" + tag
        dest = os.path.join(raw_dir, "repos", f"{safe}.tar.gz")
        try:
            download_file(f"https://codeload.github.com/{repo}/tar.gz/refs/tags/{tag}", dest)
        except Exception as e:  # noqa: BLE001 - skip missing tags, report at end
            print(f"  code: SKIP {repo}@{tag}: {e}", file=sys.stderr)
            continue
        md_seen = src_seen = 0
        with tarfile.open(dest, "r:gz") as tf:
            members = sorted((m for m in tf.getmembers() if m.isfile()),
                             key=lambda m: m.name)
            for m in members:
                lower = m.name.lower()
                if any(part in lower for part in SKIP_PATH_PARTS):
                    continue
                ext = os.path.splitext(lower)[1]
                lang = EXT_LANG.get(ext)
                if lang is None or lang not in langs:
                    continue
                if lang == "md" and md_seen >= md_cap_per_repo:
                    continue
                if lang != "md" and src_seen >= src_cap_per_repo:
                    continue
                if not (2500 <= m.size <= 120_000):
                    continue
                text = usable_text(tf.extractfile(m).read())
                if text is None:
                    continue
                cuts = markdown_cuts(text) if lang == "md" else source_cuts(text, lang)
                if len(cuts) < 2:
                    continue
                if lang == "md":
                    md_seen += 1
                else:
                    src_seen += 1
                rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
                yield (f"{repo}@{tag}:{rel}", text, cuts,
                       f"github:{repo}@{tag}:{lang}", 3)
        print(f"  code: {repo}@{tag} md={md_seen} src={src_seen}", file=sys.stderr)


BILL_SEC_RE = re.compile(r"^\s{0,4}(?:SECTION|SEC\.|TITLE)\s+\d+", re.MULTILINE)
PARA_RE = re.compile(r"(?:\r?\n)\s*(?:\r?\n)+")

def paragraph_cuts(text):
    cuts = []
    for m in PARA_RE.finditer(text):
        rest = text[m.end():]
        lead = len(rest) - len(rest.lstrip())
        cut = m.end() + lead
        if 0 < cut < len(text):
            cuts.append(cut)
    return cuts


def build_legal(raw_dir, caps):
    # BVA judgements: pre-existing gold layout (MultiLegalSBD source text);
    # paragraph boundaries from the original blank-line structure.
    for i in range(4):
        dest = os.path.join(raw_dir, "multilegal", f"en_judgements_{i}.jsonl.xz")
        try:
            download_file(f"{HF_BASE}/datasets/rcds/MultiLegalSBD/resolve/main/data/en_judgements_{i}.jsonl.xz", dest)
        except Exception as e:  # noqa: BLE001
            print(f"  legal: SKIP en_judgements_{i}: {e}", file=sys.stderr)
            continue
        with lzma.open(dest, "rt", encoding="utf-8") as f:
            for j, line in enumerate(f):
                rec = json.loads(line)
                text = rec["text"]
                if len(text) < DOC_MIN_CHARS:
                    continue
                yield (f"en_judgements_{i}:{j}", text, paragraph_cuts(text),
                       "hf:rcds/MultiLegalSBD/en_judgements", 6)
    # US bills: "SEC. n." headers are the structural boundaries.
    n_bills = 0
    rows = hf_rows_cached(raw_dir, "FiscalNote/billsum", "default", "train",
                          max_rows=caps.get("billsum_rows", 11000))
    for j, row in enumerate(rows):
        text = row.get("text") or ""
        if len(text) < DOC_MIN_CHARS:
            continue
        cuts = [m.start() for m in BILL_SEC_RE.finditer(text) if m.start() > 0]
        cuts = sorted(set(cuts) | set(paragraph_cuts(text))) if len(cuts) < 2 else cuts
        if len(cuts) < 1:
            continue
        n_bills += 1
        yield (f"billsum:{j}", text, cuts, "hf:FiscalNote/billsum", 2)
    print(f"  legal: billsum docs considered={n_bills}", file=sys.stderr)


EDGAR_SECTION_RE = re.compile(r"^section_(\d+[A-Za-z]?)$")

def edgar_section_order(key):
    m = EDGAR_SECTION_RE.match(key)
    num = int(re.match(r"\d+", m.group(1)).group())
    suffix = m.group(1)[len(str(num)):]
    return (num, suffix)


def build_financial(raw_dir, caps):
    # Two test-split years (~180MB each); the MTCB shingle filter drops a
    # sizable share of filing windows afterwards (10-K boilerplate matches
    # ficha docs), so source generously.
    years = caps.get("edgar_years", ("2018", "2017"))
    max_filings = caps.get("edgar_filings", 2200)
    n = 0
    for year in years:
        dest = os.path.join(raw_dir, "edgar", f"{year}_test.jsonl")
        download_file(f"{HF_BASE}/datasets/eloukas/edgar-corpus/resolve/main/{year}/test.jsonl", dest)
        with open(dest, encoding="utf-8") as f:
            for line in f:
                if n >= max_filings:
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                keys = sorted((k for k, v in rec.items()
                               if EDGAR_SECTION_RE.match(k) and isinstance(v, str) and len(v.strip()) > 200),
                              key=edgar_section_order)
                if len(keys) < 3:
                    continue
                parts, cuts, pos = [], [], 0
                for k in keys:
                    body = rec[k].strip()
                    item = k[len("section_"):]
                    if not body.lower().startswith("item"):
                        body = f"Item {item}. {body}"
                    if parts:
                        cuts.append(pos)
                    parts.append(body)
                    pos += len(body) + 2
                text = "\n\n".join(parts)
                if len(text) < DOC_MIN_CHARS:
                    continue
                n += 1
                doc_id = rec.get("filename") or rec.get("cik") or str(n)
                yield (f"edgar{year}:{doc_id}", text, cuts,
                       f"hf:eloukas/edgar-corpus/{year}", 7)
    print(f"  financial: filings used={n}", file=sys.stderr)


def build_technical(raw_dir, caps):
    rows = hf_rows_cached(raw_dir, "neuralwork/arxiver", "default", "train",
                          max_rows=caps.get("arxiver_rows", 1600), page=20)
    n = 0
    for j, row in enumerate(rows):
        text = row.get("markdown") or ""
        if len(text) < DOC_MIN_CHARS:
            continue
        cuts = markdown_cuts(text)
        if len(cuts) < 3:
            continue
        n += 1
        yield (f"arxiver:{row.get('id', j)}", text, cuts, "hf:neuralwork/arxiver", 4)
    print(f"  technical: papers used={n}", file=sys.stderr)


def build_transcripts(raw_dir, caps):
    rows = hf_rows_cached(raw_dir, "nbroad/mediasum", "mediasum", "train",
                          max_rows=caps.get("mediasum_rows", 9000), page=50)
    n = 0
    for j, row in enumerate(rows):
        utts = row.get("utt") or []
        speakers = row.get("speaker") or []
        if len(utts) < 6 or len(utts) != len(speakers):
            continue
        parts, cuts, pos = [], [], 0
        for spk, utt in zip(speakers, utts):
            turn = f"{spk}: {utt}"
            if parts:
                cuts.append(pos)
            parts.append(turn)
            pos += len(turn) + 1
        text = "\n".join(parts)
        if len(text) < DOC_MIN_CHARS:
            continue
        n += 1
        yield (f"mediasum:{row.get('id', j)}", text, cuts, "hf:nbroad/mediasum", 4)
    print(f"  transcripts: interviews used={n}", file=sys.stderr)


DOMAIN_BUILDERS = {
    "code": (build_code, 6000),
    "legal": (build_legal, 5000),
    "financial": (build_financial, 5000),
    "technical": (build_technical, 4500),
    "transcripts": (build_transcripts, 3000),
}


# --- assembly ------------------------------------------------------------------

def val_pick(doc_key, val_frac):
    h = int.from_bytes(hashlib.blake2b(doc_key.encode("utf-8"), digest_size=4).digest(), "big")
    return (h % 10_000) < int(val_frac * 10_000)


def record_stats(rec):
    text = rec["text"]
    chunks = rec.get("chunks") or rec.get("chunk_boundaries") or []
    n_bytes = len(text.encode("utf-8"))
    n_ws_tokens = len(text.split())
    chunk_bytes = [c["end_char"] - c["start_char"] for c in chunks]
    return n_bytes, n_ws_tokens, chunk_bytes


class DomainAgg:
    def __init__(self):
        self.docs = 0
        self.train_docs = 0
        self.val_docs = 0
        self.doc_bytes = 0
        self.ws_tokens = 0
        self.chunks = 0
        self.chunk_bytes = 0
        self.one_chunk_docs = 0
        self.contaminated = 0
        self.sources = {}

    def add(self, rec, is_val):
        n_bytes, n_ws, chunk_bytes = record_stats(rec)
        self.docs += 1
        self.val_docs += 1 if is_val else 0
        self.train_docs += 0 if is_val else 1
        self.doc_bytes += n_bytes
        self.ws_tokens += n_ws
        self.chunks += len(chunk_bytes)
        self.chunk_bytes += sum(chunk_bytes)
        if len(chunk_bytes) <= 1:
            self.one_chunk_docs += 1
        src = (rec.get("source") or "base").split("#")[0]
        self.sources[src] = self.sources.get(src, 0) + 1

    def summary(self):
        boundaries = self.chunks - self.docs  # first chunk of each doc is not a boundary
        return {
            "docs": self.docs,
            "train_docs": self.train_docs,
            "val_docs": self.val_docs,
            "mean_doc_chars": round(self.doc_bytes / self.docs) if self.docs else 0,
            "mean_chunks_per_doc": round(self.chunks / self.docs, 2) if self.docs else 0,
            "mean_chunk_chars": round(self.chunk_bytes / self.chunks) if self.chunks else 0,
            "boundary_rate_ws_tokens_pct": (
                round(100.0 * boundaries / self.ws_tokens, 3) if self.ws_tokens else 0),
            "one_chunk_docs": self.one_chunk_docs,
            "contaminated_dropped": self.contaminated,
            "sources": self.sources,
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="/private/tmp/multidomain_mix")
    ap.add_argument("--base", default="/Users/tim/Documents/af/gopeft/data/fused_train.jsonl")
    ap.add_argument("--base-val", default="/Users/tim/Documents/af/gopeft/data/fused_val.jsonl")
    ap.add_argument("--base-val-docs", type=int, default=1000,
                    help="base val docs sampled from --base-val (base train passes through whole)")
    ap.add_argument("--domains", default=",".join(DOMAIN_BUILDERS),
                    help="comma-separated new domains (default: all)")
    ap.add_argument("--val-frac", type=float, default=0.03,
                    help="held-out fraction per new domain (2-5%%)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--contamination-hits", type=int, default=3,
                    help="drop doc when >= this many 64-char shingles match an MTCB doc")
    ap.add_argument("--no-contamination-check", action="store_true",
                    help="skip the MTCB shingle filter (testing only)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild per-domain jsonl even if cached under <out-dir>/domains/")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw_dir = os.path.join(args.out_dir, "raw")
    domains_dir = os.path.join(args.out_dir, "domains")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(domains_dir, exist_ok=True)

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    unknown = [d for d in domains if d not in DOMAIN_BUILDERS]
    if unknown:
        ap.error(f"unknown domains {unknown}; known: {sorted(DOMAIN_BUILDERS)}")

    exclusion_provenance = {}
    exclusion_index = set()
    if not args.no_contamination_check:
        print("== building MTCB exclusion index")
        exclusion_index, exclusion_provenance = build_exclusion_index(raw_dir)
        print(f"  exclusion index: {len(exclusion_index)} shingles")

    aggs = {}
    failures = {}

    # --- new domains -> domains/<d>.jsonl (cached) ---
    for domain in domains:
        out_path = os.path.join(domains_dir, f"{domain}.jsonl")
        if os.path.exists(out_path) and not args.force:
            print(f"== {domain}: cached at {out_path}")
            continue
        builder, target_docs = DOMAIN_BUILDERS[domain]
        print(f"== {domain}: building (target ~{target_docs} docs)")
        agg_drop = 0
        n_written = 0
        seen_hashes = set()
        try:
            with open(out_path + ".part", "w", encoding="utf-8") as out:
                for doc_id, text, cuts, source, max_windows in builder(raw_dir, {}):
                    if n_written >= target_docs:
                        break
                    for rec in make_records(doc_id, text, cuts, domain, source, max_windows):
                        if n_written >= target_docs:
                            break
                        digest = hashlib.blake2b(
                            normalize_for_shingles(rec["text"]).encode("utf-8"),
                            digest_size=8).digest()
                        if digest in seen_hashes:
                            continue
                        seen_hashes.add(digest)
                        if exclusion_index and contaminated(
                                rec["text"], exclusion_index, args.contamination_hits):
                            agg_drop += 1
                            continue
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_written += 1
            os.replace(out_path + ".part", out_path)
            print(f"  {domain}: wrote {n_written} docs (contaminated dropped: {agg_drop})")
            with open(os.path.join(domains_dir, f"{domain}.drop.json"), "w") as f:
                json.dump({"contaminated_dropped": agg_drop}, f)
        except Exception as e:  # noqa: BLE001 - report per-domain, fail at end
            failures[domain] = f"{type(e).__name__}: {e}"
            print(f"  FAILED {domain}: {failures[domain]}", file=sys.stderr)

    # --- assemble train_mix / val_mix ---
    print("== assembling mix")
    train_path = os.path.join(args.out_dir, "train_mix.jsonl")
    val_path = os.path.join(args.out_dir, "val_mix.jsonl")
    aggs["base"] = DomainAgg()
    with open(train_path + ".part", "w", encoding="utf-8") as train_out, \
         open(val_path + ".part", "w", encoding="utf-8") as val_out:
        # Base semantic docs: passed through unchanged.
        with open(args.base, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                train_out.write(line + "\n")
                aggs["base"].add(json.loads(line), is_val=False)
        # Base val: sampled (deterministically) from the existing held-out file.
        base_val_rng = random.Random(args.seed + 1)
        with open(args.base_val, encoding="utf-8") as f:
            base_val_lines = [l.strip() for l in f if l.strip()]
        base_val_rng.shuffle(base_val_lines)
        for line in base_val_lines[: args.base_val_docs]:
            val_out.write(line + "\n")
            aggs["base"].add(json.loads(line), is_val=True)
        # New domains, per-domain held-out split.
        for domain in domains:
            if domain in failures:
                continue
            path = os.path.join(domains_dir, f"{domain}.jsonl")
            if not os.path.exists(path):
                continue
            aggs[domain] = DomainAgg()
            drop_path = os.path.join(domains_dir, f"{domain}.drop.json")
            if os.path.exists(drop_path):
                with open(drop_path) as f:
                    aggs[domain].contaminated = json.load(f).get("contaminated_dropped", 0)
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    # Hold out at source-document level so windows of one
                    # document never straddle the train/val split.
                    is_val = val_pick(rec["source"].rsplit("#", 1)[0], args.val_frac)
                    (val_out if is_val else train_out).write(line + "\n")
                    aggs[domain].add(rec, is_val)
    os.replace(train_path + ".part", train_path)
    os.replace(val_path + ".part", val_path)

    manifest = {
        "generator": "generate_fused_chunker_multidomain_mix.py",
        "created_unix": int(time.time()),
        "seed": args.seed,
        "schema": {
            "record": '{"text": str, "chunks": [{"start_char": int, "end_char": int}]}',
            "offsets": "UTF-8 byte offsets; new-domain chunks contiguous covering [0, byte_len); "
                       "base records pass through unchanged (chunk_boundaries variant, 1-byte gaps)",
            "extra_fields": "domain/source are provenance only; the Zig RawRecord parser "
                            "ignores unknown fields",
        },
        "size_band": {"token_band": [TOKEN_BAND_MIN, TOKEN_BAND_MAX, TOKEN_HARD_MAX],
                      "bytes_per_token": BYTES_PER_TOKEN,
                      "char_bands": {d: domain_band(d) for d in BYTES_PER_TOKEN},
                      "doc_max_chars": DOC_MAX_CHARS, "doc_max_chunks": DOC_MAX_CHUNKS},
        "labels": "deterministic structural boundaries (headers/declarations/Item sections/"
                  "SEC. headers/speaker turns/paragraph layout); no LLM labeling",
        "contamination": {
            "checked": not args.no_contamination_check,
            "method": f"normalized 64-char shingles (stride 32, blake2b-64); doc dropped on "
                      f">= {args.contamination_hits} shingle hits against any MTCB corpus doc",
            "excluded_mtcb_corpora": exclusion_provenance,
            "excluded_by_construction": [
                "Project Gutenberg literature (gacha)",
                "CUAD legal contracts (hojicha)",
                "TAT-QA financial tables (tacha)",
                "QASPER scientific papers (sencha)",
                "NICE/CDC/WHO medical guidelines (ryokucha)",
                "MIT OCW lecture transcripts (genmaicha) - transcripts sourced from "
                "MediaSum interviews instead",
                "GitHub README corpus (macha) and multilingual code corpus (cocha) - "
                "only shingle-screened pinned-tag repo tarballs used",
            ],
        },
        "domains": {d: a.summary() for d, a in aggs.items()},
        "totals": {
            "docs": sum(a.docs for a in aggs.values()),
            "train_docs": sum(a.train_docs for a in aggs.values()),
            "val_docs": sum(a.val_docs for a in aggs.values()),
        },
        "failures": failures,
    }
    manifest_path = os.path.join(args.out_dir, "manifest_mix.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"\ntrain: {train_path}\nval:   {val_path}\nmanifest: {manifest_path}")
    for d, a in aggs.items():
        s = a.summary()
        print(f"  {d:12s} docs={s['docs']:6d} mean_doc={s['mean_doc_chars']:6d} "
              f"mean_chunk={s['mean_chunk_chars']:5d} boundary_rate={s['boundary_rate_ws_tokens_pct']}% "
              f"dropped={s['contaminated_dropped']}")
    if failures:
        print("\ndomain failures:", file=sys.stderr)
        for domain, err in failures.items():
            print(f"  {domain}: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
