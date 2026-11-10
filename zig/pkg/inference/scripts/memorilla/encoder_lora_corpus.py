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

"""Generate a larger research corpus without opening existing qualification splits."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from encoder_lora import SCHEMA, data_rows, digest, load_split, save_json


def inventories(rows):
    return (
        {e for r in rows for e in r["entities"]},
        {r["answer"] for r in rows if r["category"] != "missing"}
        | {
            code
            for r in rows
            for doc in r["documents"]
            for code in re.findall(r"\b[A-Z][a-z][2-9]-00[0-9]{4}\b", doc)
        },
        {d for r in rows for d in r["documents"]},
    )


def disjoint(a, b):
    return all(not (x & y) for x, y in zip(inventories(a), inventories(b)))


def broaden(rows):
    for offset in range(0, len(rows), 4):
        group = rows[offset : offset + 4]
        site, other, owner, distractor = group[0]["entities"]
        first, second = group[0]["answer"], group[1]["answer"]
        index = offset // 4
        code_question = [
            f"What is the exact access code for {site}?",
            f"Give the access code assigned to {site}.",
            f"Which access code belongs to {site}?",
            f"Read the access code recorded for {site}.",
        ][index % 4]
        name_question = [
            f"Who holds the brass key at {site}?",
            f"Give the exact name of the brass-key holder for {site}.",
            f"Who is recorded as the brass key owner at {site}?",
            f"Name the person entrusted with the brass key at {site}.",
        ][index % 4]
        for row in group:
            missing_code = row["category"] == "missing" and index % 2 == 0
            missing_name = row["category"] == "missing" and index % 2 == 1
            value = second if row["id"].endswith("code-b") else first
            actual_code = "UNKNOWN" if missing_code else value
            actual_name = "UNKNOWN" if missing_name else owner
            code_field = "not recorded" if missing_code else value
            name_field = "not recorded" if missing_name else owner
            templates = [
                f"The access register at {site} assigns its brass key to {name_field}. Its access code is {code_field}.",
                f"At {site}, the brass-key holder is {name_field}. The code for this depot is {code_field}.",
                f"Depot: {site}. Brass key owner: {name_field}. Access code: {code_field}.",
                f"Access record for {site}: the code is {code_field}; the brass key is assigned to {name_field}.",
                f"The named brass-key holder for {site} is {name_field}, and the associated access code is {code_field}.",
                f"Register entry — {site}. Code on file: {code_field}. Person holding the brass key: {name_field}.",
            ]
            row["documents"][0] = templates[index % len(templates)]
            row["template"] = index % len(templates)
            if index % 2:
                row["documents"].reverse()
            row["question"] = (
                code_question
                if row["category"] == "code" or missing_code
                else name_question
            )
            row["reconstruction"] = {
                "question": f"Reconstruct the brass-key record for {site}. Return exactly two lines: Name: followed by the holder's name; Code: followed by the access code. Use UNKNOWN for an absent field.",
                "answer": f"Name: {actual_name}\nCode: {actual_code}",
            }
    return rows


def spell_targets(rows):
    for row in rows:
        task = row["reconstruction"]
        lines = task["answer"].splitlines()
        if (
            len(lines) != 2
            or not lines[0].startswith("Name: ")
            or not lines[1].startswith("Code: ")
        ):
            raise ValueError("invalid field reconstruction target")
        name, code = lines[0][6:], lines[1][6:]

        def spell(value, name_field=False):
            if value == "UNKNOWN":
                return value
            return (
                " / ".join(" ".join(word) for word in value.split(" "))
                if name_field
                else " ".join(value)
            )

        task["answer"] = "Name: " + spell(name, True) + "\nCode: " + spell(code)
        task["question"] += (
            " For each present field, spell its value with one space between characters. Separate words in a name with / . Keep UNKNOWN unspelled."
        )
    return rows


def expanded_data(
    output,
    development_from,
    seed=20261110,
    train_groups=4096,
    fresh_dev_groups=8,
    spell_values=False,
):
    if not 1 <= train_groups <= 50000 or not 1 <= fresh_dev_groups <= 1000:
        raise ValueError("bounded positive corpus sizes required")
    original = load_split(development_from, "dev")
    for row in original:
        row["cohort"] = "original-development"
    for attempt in range(128):
        effective = seed + attempt * 65537
        train = broaden(data_rows(effective, "train", train_groups))
        if spell_values:
            spell_targets(train)
        fresh = data_rows(effective, "newdev", fresh_dev_groups)
        if (
            disjoint(train, original)
            and disjoint(train, fresh)
            and disjoint(fresh, original)
        ):
            break
    else:
        raise ValueError("could not construct disjoint training/development splits")
    for row in fresh:
        row["cohort"] = "fresh-development"
    dev = original + fresh
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": SCHEMA,
        "recipe": "expanded-field-spelling-v1"
        if spell_values
        else "expanded-field-reconstruction-v1",
        "seed": effective,
        "requested_seed": seed,
        "development_only": True,
        "rollout_qualified": False,
        "old_qualification_splits_read": False,
        "source_sha256": digest(Path(__file__)),
        "original_development_sha256": digest(development_from / "dev.json"),
        "split_policy": "Train/dev entities, complete non-UNKNOWN values and documents disjoint; old qualification sets not read. New qualification facts must be generated after checkpoint freeze.",
        "files": {},
    }
    for split, rows in [("train", train), ("dev", dev)]:
        path = output / (split + ".json")
        save_json(path, {"schema": SCHEMA, "split": split, "cases": rows})
        manifest["files"][split] = {
            "sha256": digest(path),
            "cases": len(rows),
            "groups": len({r["group"] for r in rows}),
        }
    save_json(output / "manifest.json", manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--development-from", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261110)
    p.add_argument("--train-groups", type=int, default=4096)
    p.add_argument("--fresh-dev-groups", type=int, default=8)
    p.add_argument("--spell-values", action="store_true")
    args = p.parse_args()
    if not 1 <= args.train_groups <= 50000 or not 1 <= args.fresh_dev_groups <= 1000:
        p.error("bounded positive corpus sizes required")
    print(
        expanded_data(
            args.output,
            args.development_from,
            args.seed,
            args.train_groups,
            args.fresh_dev_groups,
            args.spell_values,
        )
    )


if __name__ == "__main__":
    main()
