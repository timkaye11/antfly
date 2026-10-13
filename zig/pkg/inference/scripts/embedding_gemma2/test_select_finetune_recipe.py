#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0

import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path

from finetune_contract import file_sha256
from select_finetune_recipe import (
    DIMENSIONS, METRICS, RETENTION_ROLES, TARGET_ROLES,
    summarize_validation, verify_full_validation,
)


def report():
    return {'scope': 'independent-recipe-selection-only; ineligible-for-release-promotion',
            'production_qualified': False, 'identities': {'base': 'base', 'candidate': 'candidate'},
            'rows': [{'candidate': 'candidate', 'role': role, 'dimension': dim, 'metric': metric,
                      'mean_delta': .005, 'delta_ci95': [0., .01]}
                     for role, dim, metric in itertools.product(
                         (*TARGET_ROLES, *RETENTION_ROLES), DIMENSIONS, METRICS)]}


class RecipeSelectionTests(unittest.TestCase):
    def test_every_bound_passes_without_claiming_production(self):
        value = summarize_validation(report())
        self.assertTrue(value['eligible'])
        self.assertFalse(value['production_qualified'])
        self.assertEqual(value['checked_metric_bounds'], 48)

    def test_positive_macro_gain_cannot_mask_one_language_dimension_or_metric(self):
        for role, dim, metric in itertools.product(TARGET_ROLES, DIMENSIONS, METRICS):
            value = report()
            row = next(r for r in value['rows'] if (r['role'], r['dimension'], r['metric']) == (role, dim, metric))
            row['delta_ci95'][0] = -1e-8
            with self.subTest(role=role, dim=dim, metric=metric):
                summary = summarize_validation(value)
                self.assertGreater(summary['target_macro_ndcg_delta'], 0.)
                self.assertFalse(summary['eligible'])
                self.assertEqual(len(summary['failures']), 1)

    def test_retention_threshold_is_unchanged_and_inclusive(self):
        value = report()
        row = next(r for r in value['rows'] if r['role'] in RETENTION_ROLES)
        row['delta_ci95'][0] = -.01
        self.assertTrue(summarize_validation(value)['eligible'])
        row['delta_ci95'][0] = -.010000001
        self.assertFalse(summarize_validation(value)['eligible'])

    def test_missing_duplicate_and_unknown_cells_are_rejected(self):
        for mutate in (lambda r: r['rows'].pop(), lambda r: r['rows'].append(copy.deepcopy(r['rows'][0])),
                       lambda r: r['rows'][0].update(dimension=129),
                       lambda r: r['rows'][0].update(role='msmarco')):
            value = report(); mutate(value)
            with self.assertRaises(ValueError): summarize_validation(value)

    def test_invalid_numbers_and_reversed_bounds_are_rejected(self):
        for patch in ({'mean_delta': float('nan')}, {'mean_delta': True},
                      {'delta_ci95': [0., float('inf')]}, {'delta_ci95': [.1, 0.]},
                      {'delta_ci95': [-2., 0.]}, {'dimension': True}):
            value = report(); value['rows'][0].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError): summarize_validation(value)

    def test_release_evidence_and_base_are_ineligible_for_recipe_selection(self):
        for patch in ({'scope': 'official-full-corpus'}, {'production_qualified': True}):
            value = report(); value.update(patch)
            with self.assertRaises(ValueError): summarize_validation(value)
        with self.assertRaises(ValueError): summarize_validation(report(), 'base')

    def test_manifest_binding_full_coverage_and_query_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {'query_policy': 'all-eligible-after-training-test-exclusions',
                        'corpus_policy': 'all-source-documents', 'roles': []}
            value = report(); value['roles'] = {}
            for name in (*TARGET_ROLES, *RETENTION_ROLES):
                directory = root / name; directory.mkdir()
                for artifact in ('corpus', 'queries', 'qrels'):
                    (directory / (artifact + '.jsonl')).write_text('{"id":"q"}\n')
                manifest['roles'].append({'name': name, 'query_count': 1, 'eligible_queries': 1,
                    'corpus_count': 1, 'files': {name: {'sha256': file_sha256(directory / (name + '.jsonl'))}
                                               for name in ('corpus', 'queries', 'qrels')}})
                value['roles'][name] = {'query_ids': ['q']}
            path = root / 'manifest.json'; path.write_text(json.dumps(manifest))
            value['producer'] = {'validation_manifest_sha256': file_sha256(path)}
            verify_full_validation(value, path)
            value['roles'][TARGET_ROLES[0]]['query_ids'] = ['wrong']
            with self.assertRaisesRegex(ValueError, 'query IDs'): verify_full_validation(value, path)
            value['roles'][TARGET_ROLES[0]]['query_ids'] = ['q']
            manifest['query_policy'] = 'sampled'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'bound'): verify_full_validation(value, path)
            value['producer']['validation_manifest_sha256'] = file_sha256(path)
            with self.assertRaisesRegex(ValueError, 'Sampled'): verify_full_validation(value, path)


if __name__ == '__main__':
    unittest.main()
