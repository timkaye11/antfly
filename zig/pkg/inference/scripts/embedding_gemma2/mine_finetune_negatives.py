#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mine code negatives with an authenticated frozen native CUDA base model.

Only training positives enter the candidate pool. Code query prompts, the
128/512 token caps, source language, repository exclusion and known positives
match the finetuning contract. Preserve source-provided retrieval negatives.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from evaluate_finetune_full_corpus import HttpEmbedder, TextInputLimiter, document_input, embedding_identity, jsonl
from exclude_official_evaluation import update_manifest
from finetune_contract import BASE_WEIGHT_SHA256, file_sha256, load_json, normalized_content_hash, validate_dataset_manifest


def cohort(record):
    if record['task'] != 'code_retrieval':
        return record['task']
    parts = record['query_id'].split(':')
    language = parts[1] if len(parts) > 2 and parts[0] == 'codesearchnet' else 'unspecified'
    return 'code_retrieval:' + language


class NoAdmissibleNegative(ValueError):
    pass


def validate_margin(value):
    if value is not None and (isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError('Relative positive-score margin must be finite and between zero and one')


def choose_negative(record, candidates, scores, known_positive_ids=(), *, positive_score=None, relative_margin=None):
    validate_margin(relative_margin)
    if relative_margin is not None and (positive_score is None or not math.isfinite(positive_score)):
        raise ValueError('Guided negative selection requires a finite positive cosine')
    if len(candidates) != len(scores) or not candidates:
        raise ValueError('Candidate scores must cover the training pool')
    if not all(math.isfinite(score) for score in scores):
        raise ValueError('Candidate scores must be finite')
    positive_ids = {record['positive']['id'], *record['known_positive_ids'], *known_positive_ids}
    positive_hash = normalized_content_hash(record['positive']['text'])
    ranked = sorted(zip(candidates, scores), key=lambda item: (-item[1], item[0]['positive']['id']))
    for candidate, score in ranked:
        document = candidate['positive']
        if (candidate['split'] != 'train' or cohort(candidate) != cohort(record)
                or candidate['group_id'] == record['group_id'] or document['id'] in positive_ids
                or normalized_content_hash(document['text']) == positive_hash):
            continue
        if relative_margin is not None and score > positive_score - abs(positive_score) * relative_margin:
            continue
        return copy.deepcopy(document), score
    raise NoAdmissibleNegative('No admissible training negative remains')


def checked_vectors(values, count):
    import numpy as np
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (count, 768) or not np.isfinite(result).all():
        raise ValueError('Frozen teacher must return finite full 768-vectors')
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError('Frozen teacher returned a zero or non-finite vector norm')
    return result / norms


def mine(records, embed, *, relative_margin=None, exclusions=None):
    """Return a fresh dataset and auditable selections without touching splits."""
    validate_margin(relative_margin)
    if exclusions is not None and relative_margin is None:
        raise ValueError('Dropping unmineable anchors requires explicit guided selection')
    if exclusions is not None and (not isinstance(exclusions, list) or exclusions):
        raise ValueError('Exclusion receipt must start as an empty list')
    result = copy.deepcopy(records)
    train = [row for row in result if row['split'] == 'train']
    if len({row['query_id'] for row in train}) != len(train):
        raise ValueError('Training query IDs must be unique')
    known = {}
    for row in train:
        known.setdefault(normalized_content_hash(row['query']), set()).update(
            [row['positive']['id'], *row['known_positive_ids']])
    selections = []
    groups = sorted({cohort(row) for row in train if row['task'] == 'code_retrieval'})
    for group in groups:
        members = [row for row in train if cohort(row) == group]
        # Several docstrings can point at one function; encode each function once.
        by_id = {}
        for row in members:
            identifier = row['positive']['id']
            if identifier in by_id and by_id[identifier]['positive'] != row['positive']:
                raise ValueError('Document ID binds different code bodies')
            by_id.setdefault(identifier, row)
        candidates = [by_id[key] for key in sorted(by_id)]
        positive_indexes = {row['positive']['id']: index for index, row in enumerate(candidates)}
        docs = checked_vectors(embed([document_input(row['positive']) for row in candidates], 'RETRIEVAL_DOCUMENT'), len(candidates))
        for offset in range(0, len(members), 128):
            chunk = members[offset:offset + 128]
            queries = checked_vectors(embed([row['query'] for row in chunk], 'CODE_RETRIEVAL'), len(chunk))
            scores = queries @ docs.T
            for row, values in zip(chunk, scores):
                positive_score = float(values[positive_indexes[row['positive']['id']]])
                try:
                    negative, score = choose_negative(row, candidates, values.tolist(),
                        known[normalized_content_hash(row['query'])], positive_score=positive_score,
                        relative_margin=relative_margin)
                except NoAdmissibleNegative:
                    if exclusions is None:
                        raise
                    exclusions.append({'query_id': row['query_id'], 'positive_cosine': positive_score,
                        'reason': 'no-admissible-guided-negative',
                        'maximum_negative_cosine': positive_score - abs(positive_score)*relative_margin})
                    continue
                row['negatives'] = [negative]
                selections.append({'query_id': row['query_id'], 'negative_id': negative['id'],
                                   'cosine': score, 'positive_cosine': positive_score, 'cohort': group})
        print(json.dumps({'cohort': group, 'queries': len(members), 'candidate_documents': len(candidates)}), flush=True)
    excluded = {row['query_id'] for row in exclusions or []}
    return [row for row in result if row['split'] != 'train' or row['query_id'] not in excluded], selections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'manifest', 'model-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--model', default='embeddinggemma-2')
    parser.add_argument('--relative-margin', type=float, help='Require negative <= positive - abs(positive)*MARGIN')
    parser.add_argument('--drop-unmineable', action='store_true', help='Record and exclude training anchors without a guided negative')
    args = parser.parse_args()
    validate_margin(args.relative_margin)
    if args.drop_unmineable and args.relative_margin is None:
        parser.error('--drop-unmineable requires --relative-margin')
    if args.output_dir.exists():
        raise FileExistsError('Mining output directory must be new')
    if file_sha256(args.model_dir / 'model.safetensors') != BASE_WEIGHT_SHA256:
        raise ValueError('Hard-negative mining requires the pinned frozen base')
    source = load_json(args.manifest)
    validate_dataset_manifest(source)
    from transformers import AutoTokenizer
    from finetune_reference import TOKENIZER_SHA256
    if file_sha256(args.model_dir / 'tokenizer.json') != TOKENIZER_SHA256:
        raise ValueError('Mining tokenizer differs from the training contract')
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    identity = embedding_identity(args.model_dir, weights_digest=BASE_WEIGHT_SHA256)
    limiter = TextInputLimiter(tokenizer)
    embed = HttpEmbedder(args.url, args.model, 600, limiter=limiter, expected_identity=identity)
    exclusions = [] if args.drop_unmineable else None
    mined, selections = mine(list(jsonl(args.input)), embed, relative_margin=args.relative_margin,
                             exclusions=exclusions)
    if not selections or embed.provenance != {'cuda'}:
        raise ValueError('Mining did not authenticate a native CUDA teacher')
    # Change policy before rebuilding inventories: the old policy may be random.
    source['miner']['policy'] = 'frozen-base-cosine-train-only-v1'
    source['miner']['frozen_base_model'] = 'google/embeddinggemma-2#' + identity
    manifest = update_manifest(source, mined)
    manifest['miner'].update({'teacher_identity': identity, 'base_weight_sha256': BASE_WEIGHT_SHA256,
        'query_task': 'CODE_RETRIEVAL', 'document_task': 'RETRIEVAL_DOCUMENT',
        'query_max_tokens': 128, 'document_max_tokens': 512,
        'pool_policy': 'same-language-training-positives;exclude-repository-and-known-positives',
        'retrieval_negative_policy': 'preserve-source-provided', 'collector_sha256': file_sha256(Path(__file__)),
        'relative_positive_score_margin': args.relative_margin,
        'margin_policy': 'negative<=positive-abs(positive)*margin' if args.relative_margin is not None else 'disabled',
        'unmineable_anchor_policy': 'record-and-drop-training-anchors' if args.drop_unmineable else 'fail',
        'excluded_training_anchors': exclusions or [],
        'input_sha256': file_sha256(args.input), 'input_manifest_sha256': file_sha256(args.manifest),
        'selections': selections})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'retrieval.jsonl').open('x') as stream:
        for row in mined:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    (args.output_dir / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'mined_code_queries': len(selections), 'excluded_training_anchors': len(exclusions or []),
                      'output': str(args.output_dir)}))


if __name__ == '__main__':
    main()
