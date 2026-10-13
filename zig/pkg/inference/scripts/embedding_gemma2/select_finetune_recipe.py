#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reject weak development candidates before expensive held-out qualification.

Every language, metric and MRL dimension must clear its own confidence bound.
Development eligibility never authorizes production promotion.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from finetune_contract import file_sha256

TARGET_ROLES = ('codesearchnet-python', 'codesearchnet-go')
RETENTION_ROLES = ('nfcorpus', 'scifact')
DIMENSIONS = (128, 256, 512, 768)
METRICS = ('mrr_at_10', 'ndcg_at_10', 'recall_at_100')


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def summarize_validation(report, candidate='candidate'):
    """Require all 48 distinct metric bounds; a macro gain cannot hide a loss."""
    if (report.get('scope') != 'independent-recipe-selection-only; ineligible-for-release-promotion'
            or report.get('production_qualified') is not False):
        raise ValueError('Only independent development evidence is eligible for recipe selection')
    if candidate == 'base' or candidate not in report.get('identities', {}):
        raise ValueError('An identified finetuned candidate is required')
    expected = {(role, dim, metric) for role in (*TARGET_ROLES, *RETENTION_ROLES)
                for dim in DIMENSIONS for metric in METRICS}
    seen, failures, target_scores = set(), [], []
    for row in report['rows']:
        if row.get('candidate') != candidate:
            continue
        dim = row.get('dimension')
        key = (row.get('role'), dim, row.get('metric'))
        if type(dim) is not int or key not in expected or key in seen:
            raise ValueError('Unexpected or duplicate development metric cell')
        seen.add(key)
        bounds, mean = row.get('delta_ci95'), row.get('mean_delta')
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or not all(finite_number(value) for value in (*bounds, mean))
                or not -1 <= bounds[0] <= bounds[1] <= 1 or not -1 <= mean <= 1):
            raise ValueError('Development deltas and confidence bounds must be finite and ordered')
        target = key[0] in TARGET_ROLES
        threshold = 0.0 if target else -0.01
        if bounds[0] < threshold:
            failures.append(dict(row, threshold=threshold))
        if target and key[2] == 'ndcg_at_10':
            target_scores.append(mean)
    if seen != expected:
        raise ValueError('All four development roles, four dimensions and three metrics are required')
    return {'eligible': not failures, 'production_qualified': False,
            'checked_metric_bounds': len(seen), 'failures': failures,
            'target_macro_ndcg_delta': statistics.fmean(target_scores),
            'target_min_delta': 0.0, 'retention_min_delta': -0.01}


def verify_full_validation(report, manifest_path):
    """Bind the report to complete eligible development queries and corpus bytes."""
    manifest = json.loads(manifest_path.read_text())
    if report.get('producer', {}).get('validation_manifest_sha256') != file_sha256(manifest_path):
        raise ValueError('Development report is not bound to the supplied manifest')
    if manifest.get('query_policy') != 'all-eligible-after-training-test-exclusions':
        raise ValueError('Sampled development queries cannot select the final recipe')
    if manifest.get('corpus_policy') != 'all-source-documents':
        raise ValueError('Development must retain the complete source corpora')
    seen = set()
    for role in manifest['roles']:
        name = role['name']
        if name not in (*TARGET_ROLES, *RETENTION_ROLES) or name in seen:
            raise ValueError('Unexpected or duplicate development role')
        seen.add(name)
        count = role['query_count']
        if (type(count) is not int or count <= 0 or count != role['eligible_queries']
                or type(role['corpus_count']) is not int or role['corpus_count'] <= 0):
            raise ValueError('All eligible development queries and documents must be covered')
        directory = manifest_path.parent / name
        for artifact in ('corpus', 'queries', 'qrels'):
            if file_sha256(directory / (artifact + '.jsonl')) != role['files'][artifact]['sha256']:
                raise ValueError('Development input bytes changed')
        ids = report['roles'][name]['query_ids']
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError('Development report query coverage is incomplete')
        with (directory / 'queries.jsonl').open() as stream:
            expected_ids = [json.loads(line)['id'] for line in stream]
        if ids != expected_ids:
            raise ValueError('Development report query IDs differ from the supplied inventory')
    if seen != set((*TARGET_ROLES, *RETENTION_ROLES)):
        raise ValueError('All four independent development roles are required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--validation-manifest', type=Path, required=True)
    parser.add_argument('--candidate', default='candidate')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    verify_full_validation(data, args.validation_manifest)
    summary = summarize_validation(data, args.candidate)
    summary.update(input_sha256=file_sha256(args.input),
                   validation_manifest_sha256=file_sha256(args.validation_manifest))
    args.report.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'eligible': summary['eligible'], 'failed_bounds': len(summary['failures']),
                      'production_qualified': False}))
    return 0 if summary['eligible'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
