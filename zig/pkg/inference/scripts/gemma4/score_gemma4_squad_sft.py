#!/usr/bin/env python3
"""Score paired multi-token QA outputs, treating articles as independent units.

Answer normalization and token F1 follow the official SQuAD scorer. Generation
must finish with Gemma's end-of-turn token: a truncated answer gets no primary
credit even when its visible prefix matches a reference. This scores supplied
predictions; it does not attest the model or prove native generation parity.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import string


def normalize(text: str) -> str:
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_scores(prediction: str, references: list[str]) -> tuple[float, float]:
    if (
        not isinstance(prediction, str)
        or not references
        or any(not isinstance(x, str) for x in references)
    ):
        raise ValueError("prediction and reference strings required")
    # The official scorer drops normalized-empty alternatives unless every
    # reference is empty (the explicit no-answer case).
    references = [r for r in references if normalize(r)] or [""]
    predicted = normalize(prediction).split()
    exact, f1 = 0.0, 0.0
    for reference in references:
        gold = normalize(reference).split()
        exact = max(exact, float(gold == predicted))
        common = sum((Counter(gold) & Counter(predicted)).values())
        score = (
            2.0 * common / (len(gold) + len(predicted))
            if gold and predicted
            else float(gold == predicted)
        )
        f1 = max(f1, score)
    return exact, f1


def sign_test(deltas: list[float]) -> dict:
    if not deltas or any(not math.isfinite(x) for x in deltas):
        raise ValueError("finite nonempty independent deltas required")
    # The frozen protocol classifies floating arithmetic noise as ties.
    rounded = [round(x, 12) for x in deltas]
    wins = sum(x > 0 for x in rounded)
    losses = sum(x < 0 for x in rounded)
    n = wins + losses
    p = sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n if n else 1.0
    return {
        "groups": len(deltas),
        "wins": wins,
        "losses": losses,
        "ties": len(deltas) - n,
        "one_sided_exact_p_value": p,
        "passed": wins > losses and p <= 0.05,
    }


def checked_predictions(rows: list[dict], identities: set[str]) -> dict[str, dict]:
    if not isinstance(rows, list):
        raise ValueError("prediction rows must be a list")
    result = {}
    for row in rows:
        qid, tokens = row.get("id"), row.get("token_ids")
        if not isinstance(qid, str) or qid not in identities or qid in result:
            raise ValueError("missing, unknown or duplicate prediction ID")
        if (
            not isinstance(row.get("text"), str)
            or type(row.get("terminated")) is not bool
            or not isinstance(tokens, list)
            or not tokens
            or len(tokens) > 65
            or any(type(x) is not int or x < 0 for x in tokens)
        ):
            raise ValueError("invalid prediction text, termination or token budget")
        terminated = tokens[-1] == 106
        if (
            106 in tokens[:-1]
            or row["terminated"] != terminated
            or (not terminated and len(tokens) != 65)
        ):
            raise ValueError("prediction termination does not match its tokens")
        result[qid] = row
    if result.keys() != identities:
        raise ValueError("prediction coverage differs from references")
    return result


def score_pairs(
    references: list[dict], baseline: list[dict], trained: list[dict]
) -> dict:
    refs = {}
    for row in references:
        qid = row.get("id")
        if (
            not isinstance(qid, str)
            or not qid
            or qid in refs
            or not isinstance(row.get("article"), str)
            or not row["article"]
            or not isinstance(row.get("references"), list)
            or not row["references"]
            or any(not isinstance(x, str) or not x.strip() for x in row["references"])
        ):
            raise ValueError(
                "unique IDs, articles and nonempty reference answers required"
            )
        refs[qid] = row
    if not refs:
        raise ValueError("reference set is empty")
    predictions = {
        "baseline": checked_predictions(baseline, set(refs)),
        "trained": checked_predictions(trained, set(refs)),
    }
    evaluations = {}
    article_scores = {}
    for label, predicted in predictions.items():
        rows = []
        by_article = {}
        for qid in sorted(refs):
            reference, output = refs[qid], predicted[qid]
            em, f1 = answer_scores(output["text"], reference["references"])
            completed = output["terminated"]
            row = {
                "id": qid,
                "article": reference["article"],
                "raw_exact_match": em,
                "raw_token_f1": f1,
                "exact_match": em if completed else 0.0,
                "token_f1": f1 if completed else 0.0,
                "completed": completed,
            }
            rows.append(row)
            by_article.setdefault(reference["article"], []).append(row["token_f1"])
        evaluations[label] = {
            "exact_match": math.fsum(r["exact_match"] for r in rows) / len(rows),
            "token_f1": math.fsum(r["token_f1"] for r in rows) / len(rows),
            "completed_fraction": sum(r["completed"] for r in rows) / len(rows),
            "rows": rows,
        }
        article_scores[label] = {
            a: math.fsum(v) / len(v) for a, v in by_article.items()
        }
    article_deltas = {
        a: article_scores["trained"][a] - before
        for a, before in article_scores["baseline"].items()
    }
    paired = sign_test(list(article_deltas.values()))
    before, after = evaluations["baseline"], evaluations["trained"]
    checks = {
        "f1_improved": after["token_f1"] > before["token_f1"],
        "exact_match_nonregression": after["exact_match"] >= before["exact_match"],
        "completion_passed": after["completed_fraction"] >= 0.99,
        "article_significance_passed": paired["passed"],
    }
    return {
        "passed": all(checks.values()),
        "production_qualified": False,
        "examples": len(refs),
        "articles": len(article_deltas),
        "checks": checks,
        "paired_article_test": paired,
        "article_f1_deltas": article_deltas,
        "evaluations": evaluations,
    }


def score_seeds(references: list[dict], predictions: dict[int, dict]) -> dict:
    """Recompute individual scores and average seed deltas within each article."""
    if set(predictions) != {17, 42, 991}:
        raise ValueError("exactly the frozen seeds 17, 42 and 991 are required")
    results = {
        seed: score_pairs(references, value["baseline"], value["trained"])
        for seed, value in predictions.items()
    }
    articles = set(results[42]["article_f1_deltas"])
    if any(set(result["article_f1_deltas"]) != articles for result in results.values()):
        raise ValueError("seed article inventories differ")
    averaged = {
        article: math.fsum(
            result["article_f1_deltas"][article] for result in results.values()
        )
        / 3
        for article in sorted(articles)
    }
    aggregate = sign_test(list(averaged.values()))
    return {
        "passed": all(result["passed"] for result in results.values())
        and aggregate["passed"],
        "production_qualified": False,
        "per_seed": results,
        "article_f1_deltas_averaged_across_seeds": averaged,
        "aggregate_article_test": aggregate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    refs, predictions = args.references.read_bytes(), args.predictions.read_bytes()
    payload = json.loads(predictions)
    result = score_pairs(json.loads(refs), payload["baseline"], payload["trained"])
    result.update(
        schema="antfly.gemma4.squad-sft-score/v1",
        input_sha256={
            str(args.references): hashlib.sha256(refs).hexdigest(),
            str(args.predictions): hashlib.sha256(predictions).hexdigest(),
            str(Path(__file__)): hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "passed",
                    "examples",
                    "articles",
                    "checks",
                    "paired_article_test",
                )
            }
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
