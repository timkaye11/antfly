#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0

import unittest

from exclude_official_evaluation import exclude, update_manifest
from finetune_contract import normalized_content_hash
from test_finetune_contract import manifest


def record(query, body, *, split='train', negative='negative'):
    return {'query': query, 'query_id': query, 'split': split,
            'positive': {'id': body, 'text': body}, 'negatives': [{'id': negative, 'text': negative}]}


class ExclusionTests(unittest.TestCase):
    def test_full_queries_positive_code_and_negative_code_are_excluded(self):
        records = [record('safe', 'shared MSMARCO document'), record('heldout query', 'body'),
                   record('positive code', 'test function'), record('negative code', 'body', negative='test function'),
                   record('heldout query', 'test function', split='test')]
        kept, removed = exclude(records, {normalized_content_hash('HELDOUT QUERY')},
                                {normalized_content_hash('test function')})
        self.assertEqual(kept, [records[0], records[-1]])
        self.assertEqual(len(removed), 3)
        self.assertEqual(removed[2]['reasons'], ['official_code_document_overlap'])
        with self.assertRaisesRegex(ValueError, 'no training records'):
            exclude([records[1]], {normalized_content_hash('heldout query')}, set())

    def test_source_manifest_rebuilds_training_and_mining_inventories(self):
        original = manifest()
        records = [record('q1', 'd1', negative='d2'), record('q1', 'd1', negative='d2'),
                   record('qv', 'dv', split='validation'), record('qt', 'dt', split='test')]
        rebuilt = update_manifest(original, records)
        self.assertEqual(rebuilt['splits']['train_positive_pairs'], [['q1', 'd1']])
        self.assertEqual(rebuilt['miner']['negative_pairs'], [['q1', 'd2']])
        self.assertEqual(rebuilt['miner']['corpus_ids'], ['d1', 'd2'])
        self.assertEqual(rebuilt['splits']['heldout_content_sha256'], sorted(map(normalized_content_hash, ['qt', 'dt'])))
        self.assertEqual(original['splits']['train_content_sha256'], ['a' * 64])


if __name__ == '__main__':
    unittest.main()
