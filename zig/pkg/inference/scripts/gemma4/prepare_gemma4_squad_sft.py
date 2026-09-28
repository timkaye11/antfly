#!/usr/bin/env python3
"""Prepare deterministic, article-disjoint multi-token SQuAD SFT development data.

Only the official training JSON is read. The official development JSON remains
reserved for later acceptance. Native dataset preparation must subsequently
verify exact token lengths and zero truncation; the tokenizer check here is a
conservative admission filter, not a substitute for that check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable


DOMAIN = "gemma4-squad-sft/v1"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def order_key(value: str) -> str:
    return sha(f"{DOMAIN}:{value}".encode())


def select(
    dataset: dict,
    token_count: Callable[[str], int],
    *,
    train_count: int = 1960,
    eval_count: int = 256,
    max_sequence: int = 512,
) -> dict:
    if (
        type(train_count) is not int
        or type(eval_count) is not int
        or min(train_count, eval_count) < 1
        or max_sequence < 128
    ):
        raise ValueError("positive split counts and sequence capacity >=128 required")
    if dataset.get("version") != "1.1" or not isinstance(dataset.get("data"), list):
        raise ValueError("official SQuAD v1.1 training data required")
    articles = dataset["data"]
    titles = [a["title"] for a in articles]
    if (
        len(titles) < 10
        or any(not isinstance(t, str) or not t.strip() for t in titles)
        or len(set(titles)) != len(titles)
    ):
        raise ValueError("at least ten uniquely named articles required")
    ordered_titles = sorted(titles, key=order_key)
    eval_titles = set(ordered_titles[: max(1, len(titles) // 5)])
    candidates = {"train": {}, "eval": {}}
    seen_ids = set()
    all_contexts = {"train": set(), "eval": set()}
    for article in articles:
        title = article["title"]
        split = "eval" if title in eval_titles else "train"
        rows = []
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            if not isinstance(context, str) or not context.strip():
                raise ValueError("empty or invalid context")
            context_hash = sha(context.encode())
            all_contexts[split].add(context_hash)
            for qa in paragraph["qas"]:
                qid = qa["id"]
                if not isinstance(qid, str) or not qid or qid in seen_ids:
                    raise ValueError("missing or duplicate question ID")
                seen_ids.add(qid)
                question = qa["question"]
                if (
                    not isinstance(question, str)
                    or not question.strip()
                    or not qa["answers"]
                ):
                    raise ValueError("question and answers are required")
                references = []
                for answer in qa["answers"]:
                    text, start = answer["text"], answer["answer_start"]
                    if (
                        not isinstance(text, str)
                        or not text.strip()
                        or type(start) is not int
                        or start < 0
                        or context[start : start + len(text)] != text
                    ):
                        raise ValueError("answer is not its declared source span")
                    references.append(text)
                references = sorted(set(references))
                eligible = [a for a in references if 2 <= token_count(a) <= 64]
                if not eligible:
                    continue
                answer = eligible[0]
                prompt = (
                    "Read the passage and answer the question using only the shortest exact "
                    "span from the passage. Do not explain your answer.\n\nPassage: "
                    + context
                    + "\n\nQuestion: "
                    + question
                    + "\nAnswer:"
                )
                if token_count(prompt) + token_count(answer) + 64 > max_sequence:
                    continue
                rows.append(
                    {
                        "id": "squad-" + qid,
                        "article": title,
                        "context_sha256": context_hash,
                        "references": references,
                        "answer_tokens": token_count(answer),
                        "record": {
                            "schema": "gemma_chat/v1",
                            "id": "squad-" + qid,
                            "messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": answer},
                            ],
                        },
                    }
                )
        # One question per passage, selected by identity rather than input order.
        unique_contexts = {}
        for row in sorted(rows, key=lambda row: order_key(row["id"])):
            unique_contexts.setdefault(row["context_sha256"], row)
        candidates[split][title] = list(unique_contexts.values())
    if all_contexts["train"] & all_contexts["eval"]:
        raise ValueError("duplicate passage crosses the article partition")
    result = {}
    for split, count in (("train", train_count), ("eval", eval_count)):
        selected = []
        names = sorted(candidates[split], key=order_key)
        # Round-robin articles avoids letting a few articles dominate the sample.
        for offset in range(
            max((len(v) for v in candidates[split].values()), default=0)
        ):
            for name in names:
                rows = candidates[split][name]
                if offset < len(rows):
                    selected.append(rows[offset])
                    if len(selected) == count:
                        break
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError(
                f"insufficient eligible {split} examples: {len(selected)} < {count}"
            )
        result[split] = selected
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from tokenizers import Tokenizer

    source = args.train_json.read_bytes()
    tokenizer_payload = args.tokenizer_json.read_bytes()
    tokenizer = Tokenizer.from_str(tokenizer_payload.decode())
    selected = select(
        json.loads(source),
        lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}
    for split, rows in selected.items():
        payload = b"".join(
            (json.dumps(row["record"], ensure_ascii=False) + "\n").encode()
            for row in rows
        )
        path = args.output / f"{split}.jsonl"
        with path.open("xb") as stream:
            stream.write(payload)
        references = [{k: v for k, v in row.items() if k != "record"} for row in rows]
        reference_path = args.output / f"{split}-references.json"
        reference_path.write_text(
            json.dumps(references, indent=2, ensure_ascii=False) + "\n"
        )
        files[split] = {
            "dataset_sha256": sha(payload),
            "references_sha256": sha(reference_path.read_bytes()),
            "examples": len(rows),
            "articles": len({row["article"] for row in rows}),
            "min_answer_tokens": min(row["answer_tokens"] for row in rows),
            "max_answer_tokens": max(row["answer_tokens"] for row in rows),
        }
    manifest = {
        "schema": DOMAIN,
        "source_sha256": sha(source),
        "tokenizer_sha256": sha(tokenizer_payload),
        "script_sha256": sha(Path(__file__).read_bytes()),
        "splits": files,
        "article_overlap": 0,
        "passage_overlap": 0,
        "reserved_dev_labels_read": False,
        "native_preparation_required": True,
        "production_qualified": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
