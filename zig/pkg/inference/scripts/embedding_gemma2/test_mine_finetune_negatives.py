#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0

import copy
import sys
import types
import unittest
from unittest.mock import patch

from mine_finetune_negatives import NoAdmissibleNegative, checked_vectors, choose_negative, cohort, mine
from finetune_contract import normalized_content_hash
from prepare_training_dataset import add_frozen_base_negatives, decontaminate, expand_positive_aliases


def record(name, group, text, language='python', split='train'):
    return {'query_id': f'codesearchnet:{language}:{name}', 'query': 'query-' + name,
            'task': 'code_retrieval', 'group_id': group, 'split': split,
            'positive': {'id': name, 'text': text}, 'known_positive_ids': [name],
            'negatives': []}


class MiningTests(unittest.TestCase):
    def test_duplicate_query_positive_content_aliases_cannot_be_mined_as_negatives(self):
        a, b, c, d = [record(name, 'repo-'+name, body) for name, body in
                      [('a', 'anchor code'), ('b', 'alternate positive code'),
                       ('c', ' ALTERNATE\npositive CODE '), ('d', 'true negative code')]]
        a['query'] = b['query'] = 'shared query'
        retained, _ = decontaminate([a, b, c, d])
        self.assertEqual([row['positive']['id'] for row in retained], ['a', 'c', 'd'])
        self.assertEqual(retained[0]['known_positive_ids'], ['a', 'b', 'c'])
        negative, _ = choose_negative(retained[0], retained, [.95, .8, .7],
                                      positive_score=.95, relative_margin=.05)
        self.assertEqual(negative['id'], 'd')

    def test_streamed_discarded_positive_ids_resolve_to_retained_content_aliases(self):
        a, c = record('a', 'repo-a', 'anchor code'), record('c', 'repo-c', 'alternate positive code')
        a['known_positive_ids'].append('discarded-b')
        bindings = {('train', 'discarded-b'): normalized_content_hash(c['positive']['text'])}
        before = copy.deepcopy(bindings)
        expand_positive_aliases([a, c], positive_hashes=bindings)
        self.assertEqual(a['known_positive_ids'], ['a', 'c', 'discarded-b'])
        self.assertEqual(bindings, before)

    def test_positive_content_aliases_do_not_cross_splits_and_reject_id_collisions(self):
        train = record('train', 'repo-train', 'same code')
        validation = record('validation', 'repo-val', 'same code', split='validation')
        expand_positive_aliases([train, validation])
        self.assertEqual(train['known_positive_ids'], ['train'])
        collision = record('train', 'repo-other', 'different code')
        with self.assertRaisesRegex(ValueError, 'different content'):
            expand_positive_aliases([train, collision])

    def test_teacher_margin_rejects_negatives_too_close_to_positive(self):
        query = record('q', 'repo-q', 'positive code')
        pool = [record('false', 'repo-f', 'similar code'), record('hard', 'repo-h', 'different code')]
        self.assertEqual(choose_negative(query, pool, [.96, .85])[0]['id'], 'false')
        doc, _ = choose_negative(query, pool, [.96, .85], positive_score=.9, relative_margin=.05)
        self.assertEqual(doc['id'], 'hard')
        self.assertEqual(choose_negative(query, pool, [.95, .85], positive_score=1., relative_margin=.05)[0]['id'], 'false')
        self.assertEqual(choose_negative(query, pool, [-.31, -.35], positive_score=-.3, relative_margin=.1)[0]['id'], 'hard')

    def test_invalid_or_missing_margin_scores_fail_before_mining(self):
        query = record('q', 'repo-q', 'positive code')
        pool = [record('other', 'repo-other', 'different code')]
        for margin, positive in ((-.1, 1.), (1.1, 1.), (float('nan'), 1.),
                                 (.05, None), (.05, float('inf'))):
            with self.subTest(margin=margin, positive=positive), self.assertRaises(ValueError):
                choose_negative(query, pool, [.9], positive_score=positive, relative_margin=margin)
        with self.assertRaises(NoAdmissibleNegative):
            choose_negative(query, pool, [.96], positive_score=1., relative_margin=.05)

    def test_guided_anchor_exclusion_is_explicit_and_preserves_nontraining_rows(self):
        a, b, c = [record(name, 'repo-'+name, name+' code') for name in ('a', 'b', 'c')]
        validation = record('v', 'repo-v', 'validation code', split='validation')
        source = [a, b, c, validation]
        before = copy.deepcopy(source)
        def embed(texts, task):
            coordinates = [(1., 0.), (0., 1.), (-1., 0.)] if task == 'RETRIEVAL_DOCUMENT' else [(-1., 0.), (0., 1.), (-1., 0.)]
            return [[*xy, *([0.] * 766)] for xy in coordinates]
        with self.assertRaises(NoAdmissibleNegative):
            mine(source, embed, relative_margin=.05)
        dropped = []
        output, selections = mine(source, embed, relative_margin=.05, exclusions=dropped)
        self.assertEqual([r['query_id'] for r in output], [b['query_id'], c['query_id'], validation['query_id']])
        self.assertEqual(dropped[0]['query_id'], a['query_id'])
        self.assertEqual(output[-1], validation)
        self.assertEqual(source, before)
        self.assertTrue(all(r['cosine'] <= r['positive_cosine'] - abs(r['positive_cosine'])*.05 for r in selections))
        with self.assertRaises(ValueError):
            mine(source, embed, exclusions=[])
        with self.assertRaises(ValueError):
            mine(source, embed, relative_margin=.05, exclusions=[{'query_id': 'stale'}])

    def test_teacher_vectors_require_full_width_finite_nonzero_norms(self):
        import numpy as np
        self.assertEqual(checked_vectors([[1.] + [0.] * 767], 1).shape, (1, 768))
        for values in ([[]], [[0.] * 768], [[float('nan')] * 768], [[1e308] * 768]):
            with self.subTest(values=values[:1]), np.errstate(over='ignore'), self.assertRaises(ValueError):
                checked_vectors(values, 1)

    def test_best_score_excludes_repository_positives_duplicates_and_other_splits(self):
        query = record('q', 'repo', 'source code')
        pool = [record('same-repo', 'repo', 'other'), record('known', 'other', 'known code'),
                record('duplicate', 'other', ' Source\nCODE '), record('go', 'other', 'go code', 'go'),
                record('validation', 'other', 'val code', split='validation'), record('hard', 'other', 'hard code')]
        document, score = choose_negative(query, pool, [1., .99, .98, .97, .96, .8], ['known'])
        self.assertEqual(document['id'], 'hard')
        self.assertEqual(score, .8)
        document['text'] = 'mutated'
        self.assertEqual(pool[-1]['positive']['text'], 'hard code')

    def test_ties_use_document_ids_and_invalid_scores_fail(self):
        query = record('q', 'repo', 'query code')
        pool = [record('z', 'other', 'z code'), record('a', 'another', 'a code')]
        self.assertEqual(choose_negative(query, pool, [.8, .8])[0]['id'], 'a')
        for scores in ([], [.8], [float('nan'), .8]):
            with self.assertRaises(ValueError):choose_negative(query, pool, scores)
        with self.assertRaisesRegex(ValueError, 'No admissible'):
            choose_negative(query, [query], [1.])

    def test_mining_uses_code_queries_and_document_prompts_without_changing_other_splits(self):
        a, b = record('a', 'repo-a', 'a code'), record('b', 'repo-b', 'b code')
        validation = record('v', 'repo-v', 'validation code', split='validation')
        retrieval = record('r', 'repo-r', 'retrieval text')
        retrieval['task'] = 'retrieval'
        retrieval['negatives'] = [{'id': 'upstream', 'text': 'upstream hard negative'}]
        values = [a, b, validation, retrieval]
        before = copy.deepcopy(values)
        calls = []
        def embed(texts, task):
            calls.append((texts, task))
            return [[1.] + [0.] * 767 for _ in texts]
        mined, selections = mine(values, embed)
        self.assertEqual([task for _, task in calls], ['RETRIEVAL_DOCUMENT', 'CODE_RETRIEVAL'])
        self.assertEqual(mined[0]['negatives'][0]['id'], 'b')
        self.assertEqual(mined[1]['negatives'][0]['id'], 'a')
        self.assertEqual(mined[2:], before[2:])
        self.assertEqual(values, before)
        self.assertEqual(len(selections), 2)
        self.assertNotEqual(cohort(a), cohort(record('g', 'repo-g', 'go code', 'go')))

    def test_all_positives_for_a_shared_query_are_excluded(self):
        a, b, c = (record(name, 'repo-' + name, name + ' code') for name in ('a', 'b', 'c'))
        a['query'] = b['query'] = 'shared query'
        def embed(texts, task):return [[1.] + [0.] * 767 for _ in texts]
        mined, _ = mine([a, b, c], embed)
        self.assertEqual(mined[0]['negatives'][0]['id'], 'c')
        self.assertEqual(mined[1]['negatives'][0]['id'], 'c')

    def test_sentence_transformer_mining_renders_code_document_and_title_prompts(self):
        import numpy as np
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):return list(text)
            def __call__(self, text, **kwargs):return {'offset_mapping': [(i, i+1) for i in range(len(text))]}
        calls = []
        class Teacher:
            tokenizer = Tokenizer()
            def __init__(self, *args, **kwargs):pass
            def encode(self, texts, **kwargs):
                self_test.assertEqual(kwargs['prompt'], '')
                calls.append(texts)
                return np.asarray([[1., 0.] for _ in texts])
        self_test = self
        a, b = record('a', 'repo-a', 'a code'), record('b', 'repo-b', 'b code')
        b['positive']['title'] = 'Example'
        module = types.SimpleNamespace(SentenceTransformer=Teacher)
        with patch.dict(sys.modules, {'sentence_transformers': module}):
            add_frozen_base_negatives([a, b], '/unused', 2)
        self.assertEqual(calls[0], ['task: code retrieval | query: query-a',
                                   'task: code retrieval | query: query-b'])
        self.assertEqual(calls[1], ['title: none | text: a code', 'title: Example | text: b code'])
        self.assertEqual(a['negatives'][0]['id'], 'b')
        self.assertEqual(b['negatives'][0]['id'], 'a')


if __name__ == '__main__':unittest.main()
