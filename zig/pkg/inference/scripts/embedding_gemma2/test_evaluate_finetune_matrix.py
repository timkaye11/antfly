#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evaluate_finetune_full_corpus as original
import evaluate_finetune_matrix as subject
from test_evaluate_finetune_full_corpus import FakeEmbedder, full


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


class MatrixTests(unittest.TestCase):
    def test_http_requires_the_exact_authenticated_model_identity(self):
        expected = 'a' * 64
        embed = original.HttpEmbedder('http://unused', 'model', 1, expected_identity=expected)
        for actual in (None, 'b' * 64, expected):
            def endpoint(request, **kwargs):
                self.assertEqual(json.loads(request.data)['model_identity'], expected)
                return io.BytesIO(json.dumps({'backend': 'cuda', 'model_identity': actual,
                                             'data': [{'index': 0, 'embedding': full(1)}]}).encode())
            with patch.object(original.urllib.request, 'urlopen', endpoint):
                if actual == expected:
                    self.assertEqual(embed(['body'], 'RETRIEVAL_DOCUMENT'), [full(1)])
                else:
                    with self.assertRaisesRegex(ValueError, 'model identity differs'):
                        embed(['body'], 'RETRIEVAL_DOCUMENT')

    def test_lossless_keys_match_string_order_including_leading_zeroes(self):
        codec = subject.IntegerDocIds()
        ids = ['MED-2', 'MED-100', 'MED-10', 'MED-1', 'MED-0001', 'MED-0', 'MED-999999999999999']
        keys = [codec.encode(value) for value in ids]
        self.assertEqual([codec.decode(key) for key in sorted(keys)], sorted(ids))
        with self.assertRaisesRegex(ValueError, 'inconsistent prefixes'):
            codec.encode('OTHER-1')
        for value in ('MED-', 'MED-x', 'MED-1000000000000000'):
            with self.assertRaises(ValueError):
                codec.encode(value)

    def test_http_batches_preserve_titles_without_a_second_prompt(self):
        requests = []

        def endpoint(request, **kwargs):
            body = json.loads(request.data)
            requests.append(body)
            return io.BytesIO(json.dumps({'backend': 'cuda', 'data': [
                {'index': i, 'embedding': full(1)} for i in range(len(body['input']))
            ]}).encode())

        embed = original.HttpEmbedder('http://unused', 'model', 1, batch_size=2)
        rows = [original.document_input({'title': 'T', 'text': 'body'}), 'plain', 'last']
        with patch.object(original.urllib.request, 'urlopen', endpoint):
            self.assertEqual(len(embed(rows, 'RETRIEVAL_DOCUMENT')), 3)
        self.assertEqual([len(row['input']) for row in requests], [2, 1])
        self.assertEqual(requests[0]['input'], [
            {'title': 'T', 'content': [{'type': 'text', 'text': 'body'}]},
            {'content': [{'type': 'text', 'text': 'plain'}]},
        ])

    def test_text_caps_include_prompt_bos_eos_and_bound_long_titles(self):
        limiter = original.TextInputLimiter(CharacterTokenizer(), query_cap=40, document_cap=40)
        for task, value in (
            ('CODE_RETRIEVAL', 'x' * 500),
            ('RETRIEVAL_DOCUMENT', {'title': 'title' * 100, 'content': [{'type': 'text', 'text': 'body' * 100}]}),
            ('RETRIEVAL_QUERY', '你好' * 100),
        ):
            result = limiter(value, task)
            if isinstance(result, dict):
                rendered = f"title: {result['title']} | text: " + result['content'][0]['text']
            else:
                rendered = original.PROMPTS[task] + result
            self.assertLessEqual(limiter.length(rendered), 40)
        self.assertEqual(limiter.truncated, 3)
        self.assertEqual(limiter('ok', 'RETRIEVAL_QUERY'), 'ok')
        self.assertEqual(limiter.truncated, 3)

    def test_tensor_top_k_preserves_exact_and_near_tie_order_across_chunks(self):
        import torch
        codec = subject.IntegerDocIds()
        top = subject.TensorTopK(1, 'cpu', k=3)
        top.add(torch.tensor([[1.0, 1.0]], dtype=torch.float64),
                torch.tensor([codec.encode('d2'), codec.encode('d10')]))
        top.add(torch.tensor([[1.0, 1.0 + 1e-12]], dtype=torch.float64),
                torch.tensor([codec.encode('d1'), codec.encode('d3')]))
        self.assertEqual(top.rows(codec), [['d3', 'd1', 'd10']])
        with self.assertRaisesRegex(ValueError, 'finite float64'):
            top.add(torch.tensor([[float('nan')]], dtype=torch.float64), torch.tensor([codec.encode('d4')]))

    def test_single_corpus_pass_matches_independent_mrl_rankings_and_code_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = {'corpus': [{'id': f'd{i}', 'text': f'doc{i}', 'title': 'T' if i == 0 else ''} for i in range(5)],
                    'queries': [{'id': f'q{i}', 'text': f'query{i}'} for i in range(3)],
                    'qrels': [{'query_id': f'q{i}', 'corpus_id': f'd{i}', 'score': 2} for i in range(3)]}
            for name, values in rows.items():
                (root / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in values))
            vectors = {f'query{i}': full(int(i == 0), int(i == 1), int(i == 2)) for i in range(3)}
            for i in range(5):
                vectors['title: T | text: doc0' if i == 0 else f'doc{i}'] = full(int(i == 0), int(i == 1), int(i == 2), int(i > 2))
            embedders = {key: FakeEmbedder(vectors) for key in ('base', '17', '29', '43')}
            paths = {name: root / f'{name}.jsonl' for name in rows}
            ids, values, count = subject.evaluate_matrix(**paths, embedders=embedders,
                query_task='CODE_RETRIEVAL', query_chunk_size=2, corpus_chunk_size=2, device='cpu')
            self.assertEqual(count, 5)
            self.assertEqual(ids, ['q0', 'q1', 'q2'])
            for embed in embedders.values():
                self.assertEqual(sum(task == 'RETRIEVAL_DOCUMENT' for _, task in embed.calls), 3)
                self.assertEqual([task for _, task in embed.calls[:2]], ['CODE_RETRIEVAL'] * 2)
            for dimension in subject.MRL_DIMENSIONS:
                _, expected, _, _ = original.evaluate(**paths,
                    base_embed=FakeEmbedder(vectors), candidate_embed=FakeEmbedder(vectors),
                    dimension=dimension, query_task='CODE_RETRIEVAL', query_chunk_size=2, corpus_chunk_size=2)
                for key in embedders:
                    self.assertEqual(values[key, dimension], expected)
            rows['qrels'][0]['corpus_id'] = 'd999'
            paths['qrels'].write_text(''.join(json.dumps(row) + '\n' for row in rows['qrels']))
            with self.assertRaisesRegex(ValueError, 'absent from the full corpus'):
                subject.evaluate_matrix(**paths, embedders=embedders, device='cpu')


if __name__ == '__main__':
    unittest.main()
