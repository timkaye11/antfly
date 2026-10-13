#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Exclude complete official held-out queries/code from qualification training.

MS MARCO training documents may belong to its shared retrieval corpus; its
held-out queries remain excluded. CodeSearchNet test functions, including
their use as negatives, must remain excluded as well as their docstrings.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from evaluate_finetune_full_corpus import jsonl
from finetune_contract import (
    canonical_hash, file_sha256, load_json, normalized_content_hash,
    validate_dataset_manifest, validate_official_source_manifest,
)
from qualify_finetune_quality import validate_reviewed_inventory


def exclude(records, queries, code_documents):
    retained, removed = [], []
    for row in records:
        reasons = []
        if row['split'] == 'train':
            if normalized_content_hash(row['query']) in queries:
                reasons.append('official_query_overlap')
            if any(normalized_content_hash(doc['text']) in code_documents
                   for doc in [row['positive'], *row['negatives']]):
                reasons.append('official_code_document_overlap')
        if reasons:
            removed.append({'query_id': row['query_id'], 'reasons': reasons})
        else:
            retained.append(row)
    if not any(row['split'] == 'train' for row in retained):
        raise ValueError('official evaluation exclusions leave no training records')
    return retained, removed


def update_manifest(original, records):
    manifest = copy.deepcopy(original)
    train = [row for row in records if row['split'] == 'train']
    for split, name in (('train', 'train'), ('validation', 'validation'), ('test', 'heldout')):
        manifest['splits'][name + '_content_sha256'] = sorted({
            normalized_content_hash(text) for row in records if row['split'] == split
            for text in (row['query'], row['positive']['text'])})
    corpus = sorted({doc['id'] for row in train for doc in [row['positive'], *row['negatives']]})
    manifest['splits']['train_corpus_ids'] = corpus
    manifest['splits']['train_positive_pairs'] = [list(pair) for pair in sorted({
        (row['query_id'], row['positive']['id']) for row in train})]
    manifest['miner']['corpus_ids'] = corpus
    manifest['miner']['negative_pairs'] = [list(pair) for pair in sorted({
        (row['query_id'], doc['id']) for row in train for doc in row['negatives']})]
    validate_dataset_manifest(manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'manifest', 'official-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    source_manifest = args.official_dir / 'source_manifest.json'
    inventory = validate_official_source_manifest(load_json(source_manifest))
    validate_reviewed_inventory(inventory)
    queries, code, verified = set(), set(), []
    for (dataset, language, _), official in inventory.items():
        directory = args.official_dir / (f'{dataset}-{language}' if language else dataset)
        for name in ('queries', 'corpus') if dataset == 'codesearchnet' else ('queries',):
            path = directory / f'{name}.jsonl'
            digest = file_sha256(path)
            if digest != official[name + '_sha256']:
                raise ValueError('evaluation exclusion inputs differ from the reviewed full corpus')
            values = queries if name == 'queries' else code
            for row in jsonl(path):
                values.add(normalized_content_hash(row['text']))
            verified.append({'path': str(path), 'sha256': digest})
    original = load_json(args.manifest)
    validate_dataset_manifest(original)
    records, removed = exclude(list(jsonl(args.input)), queries, code)
    manifest = update_manifest(original, records)
    manifest['full_evaluation_exclusion'] = {
        'policy': 'all-official-queries-and-test-code-functions-v1',
        'input_sha256': file_sha256(args.input), 'input_manifest_sha256': file_sha256(args.manifest),
        'official_manifest_sha256': file_sha256(source_manifest), 'verified_inputs': verified,
        'query_hashes_sha256': canonical_hash(sorted(queries)),
        'code_document_hashes_sha256': canonical_hash(sorted(code)),
        'unique_queries': len(queries), 'unique_code_documents': len(code), 'removed': removed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'retrieval.jsonl').open('w') as output:
        for row in records:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    (args.output_dir / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'records': len(records), 'train': sum(row['split'] == 'train' for row in records),
                      'removed_train_records': len(removed), 'output': str(args.output_dir)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
