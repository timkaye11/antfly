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
import copy
import random
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


PHRASING_SETS = ("v1", "v2", "v3")
# Held-out evaluation wording (``data_rows`` template 3, the OOD split). No
# training phrasing may reuse either of its distinctive clauses.
HELD_OUT_PHRASES = ("is entrusted with the brass key for", "access code is recorded as")

# ``v3`` grammar: field orders, site-free holder verb phrases with the
# preposition that binds them to the site, code clauses (present, missing),
# code-first clauses that name the site, site-subject code predicates, code
# nouns and clause connectors. Every missing variant says "not recorded".
V3_ORDERS = (
    ("name", "site", "code"),
    ("site", "name", "code"),
    ("site", "code", "name"),
    ("code", "site", "name"),
    ("name", "code", "site"),
    ("code", "name", "site"),
)
V3_HOLDER_VERBS = (
    ("holds the brass key", "for"),
    ("carries the brass key", "at"),
    ("is listed as brass-key holder", "for"),
    ("keeps the depot's brass key", "at"),
    ("has custody of the brass key", "of"),
    ("is the brass key custodian", "for"),
    ("looks after the brass key", "at"),
    ("is registered as the key holder", "of"),
)
V3_MISSING_HOLDERS = (
    "the brass-key holder",
    "who holds the brass key",
    "the brass key custodian",
    "the registered key holder",
    "the person carrying the brass key",
)
V3_CODE_PHRASES = (
    ("the access code is {value}", "the access code is not recorded"),
    ("access code: {value}", "access code: not recorded"),
    (
        "the depot opens with code {value}",
        "the code the depot opens with is not recorded",
    ),
    ("its door code reads {value}", "its door code is not recorded"),
    (
        "the register shows the code {value}",
        "the register shows the code is not recorded",
    ),
    ("code on file: {value}", "code on file: not recorded"),
    ("the recorded entry code is {value}", "the entry code is not recorded"),
    ("unlock code {value}", "unlock code not recorded"),
)
V3_CODE_SITE_CLAUSES = (
    ("Code {value} opens {site}", "The access code is not recorded for {site}"),
    (
        "Access code {value} is on file for {site}",
        "No access code is on file for {site}; it is not recorded",
    ),
    (
        "The register shows code {value} for {site}",
        "The register shows the code for {site} is not recorded",
    ),
    (
        "Unlock code {value} belongs to {site}",
        "The unlock code for {site} is not recorded",
    ),
    (
        "Door code {value} is assigned to {site}",
        "The door code assigned to {site} is not recorded",
    ),
    (
        "The recorded entry code for {site} is {value}",
        "The entry code for {site} is not recorded",
    ),
)
V3_SITE_CODE_PREDICATES = (
    ("opens with code {value}", "opens with a code that is not recorded"),
    ("has access code {value}", "has an access code that is not recorded"),
    ("is unlocked with code {value}", "is unlocked with a code that is not recorded"),
    ("lists code {value} on file", "has no code on file (not recorded)"),
    ("carries door code {value}", "carries a door code that is not recorded"),
)
V3_CODE_NOUNS = ("access code", "door code", "entry code", "unlock code")
V3_CONNECTORS = (". ", "; ", ", and ", " — ")


def v3_site_document(index, site, owner, value, missing_name, missing_code):
    """Compose one site document from the ``v3`` grammar, seeded by ``index``.

    All choices are drawn before rendering, so the rows of one group (present,
    missing-name and missing-code variants) share a skeleton. The grammar never
    attaches a relative clause after ``{owner} <verb> {site}``, which is the
    held-out sentence's shape, and never uses either held-out clause.
    """
    rng = random.Random(f"phrasing-v3-{index}")
    order = rng.randrange(len(V3_ORDERS))
    verb, prep = rng.choice(V3_HOLDER_VERBS)
    missing_holder = rng.choice(V3_MISSING_HOLDERS)
    code_present, code_missing = rng.choice(V3_CODE_PHRASES)
    code_site_present, code_site_missing = rng.choice(V3_CODE_SITE_CLAUSES)
    predicate_present, predicate_missing = rng.choice(V3_SITE_CODE_PREDICATES)
    noun = rng.choice(V3_CODE_NOUNS)
    connector = rng.choice(V3_CONNECTORS)
    layout = rng.randrange(4)
    there = rng.choice((" there", ""))

    def cap(text):
        return text[0].upper() + text[1:]

    def join(first, second):
        return first + connector + (cap(second) if connector == ". " else second)

    # Site-free holder clause, holder clause bound to the site, code clause,
    # code-first clause naming the site, and site-subject code predicate.
    holder = f"{missing_holder} is not recorded" if missing_name else f"{owner} {verb}"
    holder_site = (
        f"{missing_holder} {prep} {site} is not recorded"
        if missing_name
        else f"{owner} {verb} {prep} {site}"
    )
    code = code_missing if missing_code else code_present.format(value=value)
    code_site = (code_site_missing if missing_code else code_site_present).format(
        value=value, site=site
    )
    predicate = (predicate_missing if missing_code else predicate_present).format(
        value=value
    )
    if order == 0:  # name, site, code
        document = [
            join(cap(holder_site), code),
            join("Note that " + holder_site, code),
            f"{cap(holder_site)}. For that depot, {code}",
            join("According to the register, " + holder_site, code),
        ][layout]
    elif order == 1:  # site, name, code
        document = [
            join(f"At {site}, {holder}", code),
            f"{site}, where {holder}, {predicate}",
            join(f"{site}: {holder}", code),
            join(f"Regarding {site}, {holder}", code),
        ][layout]
    elif order == 2:  # site, code, name
        document = [
            join(f"At {site}, {code}", holder + there),
            join(f"{site} {predicate}", holder + there),
            join(f"{site}: {code}", holder),
            f"{site}, which {predicate}, is where {holder}",
        ][layout]
    elif order == 3:  # code, site, name
        document = [
            join(code_site, holder + there),
            f"{code_site}, where {holder}",
            f"{code_site}. At that depot, {holder}",
            join(code_site, holder + there),
        ][layout]
    elif order == 4:  # name, code, site
        if missing_name or layout == 2:
            document = f"{join(cap(holder), code)}. The depot is {site}"
        elif missing_code:
            document = [
                f"{owner}, with the {noun} not recorded, {verb} {prep} {site}",
                f"{owner} — {code} — {verb} {prep} {site}",
                None,
                f"{owner} ({noun} not recorded) {verb} {prep} {site}",
            ][layout]
        else:
            document = [
                f"{owner}, with {noun} {value} on file, {verb} {prep} {site}",
                f"{owner} — {code} — {verb} {prep} {site}",
                None,
                f"{owner} ({noun} {value}) {verb} {prep} {site}",
            ][layout]
    else:  # code, name, site
        if missing_code:
            opener = [
                None,
                f"With the {noun} not recorded, {holder_site}",
                f"No {noun} is on file (not recorded), and {holder_site}",
                None,
            ][layout]
        else:
            opener = [
                None,
                f"Under {noun} {value}, {holder_site}",
                f"{cap(noun)} {value} is on file, and {holder_site}",
                None,
            ][layout]
        document = [
            join(cap(code), holder_site),
            opener,
            opener,
            f"{cap(code)}. The key record: {holder_site}",
        ][layout]
    return document + ".", order


def site_document(index, phrasing_set, site, owner, value, missing_name, missing_code):
    """Source sentence for one group's template; "not recorded" marks a gap.

    Every variant names the site once, the holder once unless the name is
    missing and the code once unless the code is missing, so
    ``lexical_record_text`` reads one holder/code record from it.
    """
    if phrasing_set not in PHRASING_SETS:
        raise ValueError("unknown phrasing set")
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
    if phrasing_set == "v2":
        # Varied field order and cues: holder first, code first, terse and
        # list-style records. A missing field gets explicit wording per
        # template instead of substituting "not recorded" for the value.
        if missing_name:
            templates += [
                f"The brass-key holder for {site} is not recorded; the depot's access code is {value}.",
                f"Code {value} opens {site}. Who carries the brass key there is not recorded.",
                f"Brass key, {site} — holder not recorded. Access code: {value}.",
                f"For {site} the register lists access code {value}; the brass-key holder is not recorded.",
            ]
        elif missing_code:
            templates += [
                f"{owner} keeps the brass key for {site}; the depot's access code is not recorded.",
                f"The access code that opens {site} is not recorded. The brass key there is carried by {owner}.",
                f"{owner} — brass key, {site}. Access code: not recorded.",
                f"For {site} the register lists brass-key holder {owner}; the access code is not recorded.",
            ]
        else:
            templates += [
                f"{owner} keeps the brass key for {site}; the depot's access code is {value}.",
                f"Code {value} opens {site}. The brass key there is carried by {owner}.",
                f"{owner} — brass key, {site}. Access code: {value}.",
                f"For {site} the register lists access code {value} and brass-key holder {owner}.",
            ]
    if phrasing_set == "v3":
        # Grammar-composed documents; ``template`` records the field order.
        return v3_site_document(index, site, owner, value, missing_name, missing_code)
    return templates[index % len(templates)], index % len(templates)


def broaden(rows, phrasing_set="v1"):
    """Rewrite each four-row group's site document and questions.

    The site document is located by the public entity list, so a corpus that was
    already broadened can be re-phrased with another phrasing set.
    """
    for offset in range(0, len(rows), 4):
        group = rows[offset : offset + 4]
        site, other, owner, distractor = group[0]["entities"]
        if len(group) != 4 or len({r["group"] for r in group}) != 1:
            raise ValueError("broaden requires complete four-row groups")
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
            distractors = [d for d in row["documents"] if other in d and site not in d]
            if len(row["documents"]) != 2 or len(distractors) != 1:
                raise ValueError(
                    "broaden requires one site and one distractor document"
                )
            document, row["template"] = site_document(
                index, phrasing_set, site, owner, value, missing_name, missing_code
            )
            row["documents"] = [document, distractors[0]]
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
    phrasing_set="v1",
):
    if not 1 <= train_groups <= 50000 or not 1 <= fresh_dev_groups <= 1000:
        raise ValueError("bounded positive corpus sizes required")
    if phrasing_set not in PHRASING_SETS:
        raise ValueError("unknown phrasing set")
    original = load_split(development_from, "dev")
    for row in original:
        row["cohort"] = "original-development"
    for attempt in range(128):
        effective = seed + attempt * 65537
        train = broaden(data_rows(effective, "train", train_groups), phrasing_set)
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
        "phrasing_set": phrasing_set,
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


def name_counterfactual_data(output, base_corpus, seed=20261110, phrasing_set="v1"):
    if phrasing_set not in PHRASING_SETS:
        raise ValueError("unknown phrasing set")
    train, dev = load_split(base_corpus, "train"), load_split(base_corpus, "dev")
    if phrasing_set != "v1":
        # Re-phrase the preserved base facts; entities, values, questions,
        # row order and development bytes stay the same.
        train = broaden(train, phrasing_set)
    groups = {}
    for row in train:
        groups.setdefault(row["group"], []).append(row)
    if any(len(g) != 4 for g in groups.values()):
        raise ValueError("base corpus must have four original rows per group")
    reserved = set().union(*inventories(train + dev)[:2])
    alternatives = iter(
        data_rows(seed, "train-name-counterfactual", max(len(groups) * 2, 16))[2::4]
    )
    expanded = []
    for group in groups.values():
        owners = [r for r in group if r["category"] == "name"]
        if len(owners) != 1:
            raise ValueError("base group must have one owner question")
        original = owners[0]
        while True:
            name = next(alternatives)["answer"]
            if name not in reserved:
                reserved.add(name)
                break
        other = copy.deepcopy(original)
        other["id"] += "-b"
        other["answer"] = name
        other["entities"][2] = name
        other["documents"] = [
            d.replace(original["answer"], name) for d in other["documents"]
        ]
        if "reconstruction" in other:
            if original["answer"] not in other["reconstruction"]["answer"]:
                raise ValueError("use the unspelled field corpus as the base")
            other["reconstruction"]["answer"] = other["reconstruction"][
                "answer"
            ].replace(original["answer"], name)
        expanded.extend(group + [other])
    if not disjoint(expanded, dev):
        raise ValueError("expanded name worlds collide with development")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": SCHEMA,
        "recipe": "paired-name-code-facts-v1",
        "seed": seed,
        "phrasing_set": phrasing_set,
        "development_only": True,
        "rollout_qualified": False,
        "old_qualification_splits_read": False,
        "source_manifest_sha256": digest(base_corpus / "manifest.json"),
        "source_sha256": digest(Path(__file__)),
        "source_rows_preserved": len(train),
        "name_worlds_added": len(groups),
        "files": {},
    }
    for split, rows in [("train", expanded), ("dev", dev)]:
        path = output / (split + ".json")
        save_json(path, {"schema": SCHEMA, "split": split, "cases": rows})
        manifest["files"][split] = {
            "sha256": digest(path),
            "cases": len(rows),
            "groups": len({r["group"] for r in rows}),
        }
    if (output / "dev.json").read_bytes() != (base_corpus / "dev.json").read_bytes():
        raise ValueError("development bytes changed")
    save_json(output / "manifest.json", manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--development-from", type=Path)
    source.add_argument("--base-corpus", type=Path)
    p.add_argument("--seed", type=int, default=20261110)
    p.add_argument("--train-groups", type=int, default=4096)
    p.add_argument("--fresh-dev-groups", type=int, default=8)
    p.add_argument("--spell-values", action="store_true")
    p.add_argument("--phrasing-set", choices=PHRASING_SETS, default="v1")
    args = p.parse_args()
    if args.base_corpus is not None:
        if args.spell_values:
            p.error("base-corpus mode expects unspelled field targets")
        print(
            name_counterfactual_data(
                args.output, args.base_corpus, args.seed, args.phrasing_set
            )
        )
        return
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
            args.phrasing_set,
        )
    )


if __name__ == "__main__":
    main()
