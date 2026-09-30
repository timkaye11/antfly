#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc. SPDX-License-Identifier: Apache-2.0
import base64
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from training_datasets import DatasetError, MAX_FILE, normalize, source_rows, convert, repository, sha256, tokenizer_sha256, validate_spec
from training_service import Manager
from test_training_service import Fixture, wait_for, SCRIPTS


def example(identity='one'):
    return {'version': 1, 'id': identity, 'text': 'Zoë met Ada', 'schema': {'entities': ['person']},
            'entities': [{'id': 'person-1', 'type': 'person', 'span': {'start': 0, 'end': 4}}]}


def spec(**updates):
    return {'request_id': 'dataset-request-01', 'name': 'Example data', 'family': 'gliner25',
            'source': 'upload', 'format': 'gliner25', 'filename': 'sample.jsonl', 'size_bytes': 1,
            'max_rows': 1000, **updates}


class ConversionTests(unittest.TestCase):
    def test_bio_uses_declared_label_universe_and_utf8_offsets(self):
        options = spec(format='gliner_bio', label_names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG'])
        result = normalize({'tokens': ['Zoë', 'Kaye', 'met', 'Acme'], 'ner_tags': [1, 2, 0, 3]}, options, 0)
        self.assertEqual(result['text'], 'Zoë Kaye met Acme')
        self.assertEqual(result['schema']['entities'], ['PER', 'ORG'])
        self.assertEqual(result['entities'][0]['span'], {'start': 0, 'end': 9, 'unit': 'utf8_bytes'})
        self.assertEqual(result['entities'][1]['span']['start'], 14)
        with self.assertRaises(DatasetError):
            normalize({'tokens': ['Ada'], 'ner_tags': [99]}, options, 0)

    def test_hf_classlabel_metadata_supplies_bio_ids_without_row_inference(self):
        features = [{'name': 'ner_tags', 'type': {'_type': 'List', 'feature': {'_type': 'ClassLabel', 'names': ['O', 'B-PER', 'B-ORG']}}}]
        row = normalize({'tokens': ['Ada'], 'ner_tags': [1]}, spec(format='gliner_bio'), 0, features)
        self.assertEqual(row['schema']['entities'], ['PER', 'ORG'])
        with self.assertRaises(DatasetError):
            normalize({'tokens': ['Ada'], 'ner_tags': ['B-PER']}, spec(format='gliner_bio'), 0)

    def test_gemma_mapping_rejects_missing_responses_and_media(self):
        options = spec(family='gemma4', format='gemma_instruction', columns={'prompt': 'question', 'response': 'answer', 'input': 'context'})
        self.assertEqual(normalize({'question': 'Why?', 'answer': 'Because.', 'context': 'Facts'}, options, 0),
                         {'prompt': 'Why?\n\nFacts', 'response': 'Because.'})
        with self.assertRaises(DatasetError):
            normalize({'question': 'Why?', 'answer': ''}, options, 0)
        with self.assertRaises(DatasetError):
            normalize({'messages': [{'role': 'user', 'content': [{'type': 'image'}]}]}, spec(format='gemma_chat'), 0)

    def test_hf_download_is_paginated_and_rejects_truncated_or_reordered_rows(self):
        options = spec(source='huggingface', hf_dataset='org/data', hf_config='default', hf_split='train', hf_offset=20, max_rows=102)
        def page(endpoint, parameters):
            self.assertEqual(endpoint, 'rows')
            return {'rows': [{'row_idx': index, 'row': {'text': str(index)}, 'truncated_cells': []}
                             for index in range(parameters['offset'], parameters['offset'] + parameters['length'])]}
        with patch('training_datasets.hf_get', side_effect=page) as query:
            self.assertEqual(len(list(source_rows(options, None))), 102)
            self.assertEqual([call.args[1]['length'] for call in query.call_args_list], [100, 2])
        for entry in ({'row_idx': 20, 'row': {}, 'truncated_cells': ['text']}, {'row_idx': 21, 'row': {}, 'truncated_cells': []}):
            with patch('training_datasets.hf_get', return_value={'rows': [entry]}), self.assertRaises(DatasetError):
                list(source_rows(options, None))

    def test_csv_is_converted_with_explicit_columns_and_bounded_row_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.csv'
            source.write_text('question,answer\n"Why, exactly?",Because\nHello,World\nIgnored,Row\n')
            options = spec(family='gemma4', format='gemma_instruction', filename='input.csv', model_dir='/tmp/model',
                           max_rows=2, columns={'prompt': 'question', 'response': 'answer'})
            result = convert({'spec': options, 'source': str(source), 'output': str(root / 'dataset.jsonl')})
            self.assertEqual(result['row_count'], 2)
            rows = [json.loads(row) for row in (root / 'dataset.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['prompt'], 'Why, exactly?')
            self.assertEqual(result['sha256'], sha256(root / 'dataset.jsonl'))

    def test_limits_urls_and_unexpected_fields_fail_closed(self):
        for value in ('https://evil.example/data', '../data', 'org/data?url=evil', 'org/a/b'):
            with self.assertRaises(DatasetError):
                repository(value)
        for options in (spec(size_bytes=MAX_FILE + 1), spec(max_rows=10001), spec(max_rows=True), spec(command='curl'), spec(filename='data.zip'),
                        spec(family='gemma4', format='gemma_instruction', model_dir='/tmp/model', max_rows=10000, max_seq_len=8192)):
            with self.assertRaises(DatasetError):
                validate_spec(options)


class StoreTests(Fixture):
    def upload(self, data, **updates):
        _, record = self.manager.request('POST', 'datasets', spec(size_bytes=len(data), **updates))
        _, record = self.manager.request('PUT', f'datasets/{record["id"]}/chunks', {'offset': 0, 'data': base64.b64encode(data).decode()})
        return record

    def test_chunk_retries_resume_after_restart_and_reject_different_bytes(self):
        data = b'abcdefghij'
        _, record = self.manager.request('POST', 'datasets', spec(size_bytes=len(data)))
        endpoint = f'datasets/{record["id"]}/chunks'
        chunk = {'offset': 0, 'data': base64.b64encode(data[:5]).decode()}
        self.manager.request('PUT', endpoint, chunk)
        self.manager.request('PUT', endpoint, chunk)
        with self.assertRaises(DatasetError):
            self.manager.request('PUT', endpoint, {'offset': 0, 'data': base64.b64encode(b'wrong').decode()})
        with self.assertRaises(DatasetError):
            self.manager.request('PUT', endpoint, {'offset': 6, 'data': base64.b64encode(b'gap').decode()})
        with self.assertRaises(DatasetError):
            self.manager.request('POST', f'datasets/{record["id"]}/prepare', {})
        self.manager.close()
        self.manager = Manager(self.config)
        _, resumed = self.manager.request('GET', f'datasets/{record["id"]}', {})
        self.assertEqual(resumed['uploaded_bytes'], 5)
        self.manager.request('PUT', endpoint, {'offset': 5, 'data': base64.b64encode(data[5:]).decode()})
        self.assertEqual((self.manager.datasets.root / record['id'] / 'source').read_bytes(), data)

    def test_import_idempotency_and_family_selection(self):
        options = spec()
        _, first = self.manager.request('POST', 'datasets', options)
        _, second = self.manager.request('POST', 'datasets', options)
        self.assertEqual(first['id'], second['id'])
        with self.assertRaises(DatasetError):
            self.manager.request('POST', 'datasets', {**options, 'format': 'gliner_bio'})
        with self.assertRaises(DatasetError):
            self.manager.datasets.ready(first['id'], 'gliner25')
        self.manager.datasets.records[first['id']].update(status='ready', path='/tmp/data')
        with self.assertRaises(DatasetError):
            self.manager.datasets.ready(first['id'], 'gemma4')
        self.manager.jobs['job'] = {'spec': {'dataset_id': first['id']}}
        with self.assertRaisesRegex(DatasetError, 'references'):
            self.manager.request('DELETE', f'datasets/{first["id"]}', {})

    def test_preparation_restart_does_not_claim_ready_and_concurrency_is_bounded(self):
        record = self.upload(b'{}\n')
        stored = self.manager.datasets.records[record['id']]
        stored['status'] = 'preparing'
        self.manager.datasets.save(stored)
        self.manager.close()
        self.manager = Manager(self.config)
        self.assertEqual(self.manager.datasets.records[record['id']]['status'], 'failed')
        self.manager.workers['other'] = object()
        try:
            with self.assertRaisesRegex(DatasetError, 'active'):
                self.manager.request('POST', f'datasets/{record["id"]}/prepare', {})
        finally:
            self.manager.workers.clear()

    def test_peer_staging_verifies_bytes_and_never_publishes_partial_files(self):
        data = b'{"row":"Zo\xc3\xab"}\n'
        identity = 'a' * 32
        import hashlib
        task = {'action': 'stage_dataset', 'config': self.config, 'dataset_id': identity,
                'filename': 'dataset.jsonl', 'size_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        command = [sys.executable, str(SCRIPTS / 'training_peer.py')]
        target = self.root / 'runs/datasets' / identity / 'dataset.jsonl'
        result = subprocess.run([*command, json.dumps(task)], input=data[:-1], capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(target.exists())
        self.assertFalse(list(target.parent.glob('*.upload')))
        result = subprocess.run([*command, json.dumps(task)], input=data, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(target.read_bytes(), data)
        changed = {**task, 'sha256': 'b' * 64}
        result = subprocess.run([*command, json.dumps(changed)], input=data, capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(target.read_bytes(), data)

    def test_manager_streams_prepared_bytes_through_peer_command(self):
        data = b'{"text":"prepared example"}\n' * 5000
        identity = 'c' * 32
        directory = self.manager.datasets.root / identity
        directory.mkdir()
        source = directory / 'dataset.jsonl'
        source.write_bytes(data)
        record = {'id': identity, 'path': str(source), 'sha256': sha256(source)}
        remote_root = self.root / 'remote-runs'
        remote_root.mkdir()
        def relay(destination, arguments):
            self.assertEqual(destination, 'fixture')
            task = json.loads(arguments[-1])
            task['config']['output_root'] = str(remote_root)
            return [sys.executable, *arguments[1:-1], json.dumps(task)]
        with patch('launch_jaccl_finetune.remote_command', side_effect=relay):
            self.manager.datasets.stage(record, {'ssh_destination': 'fixture'}, threading.Event())
        self.assertEqual((remote_root / 'datasets' / identity / 'dataset.jsonl').read_bytes(), data)
        source.write_bytes(b'changed')
        with self.assertRaisesRegex(DatasetError, 'bytes changed'):
            self.manager.datasets.stage(record, {'ssh_destination': 'fixture'}, threading.Event())

    def test_local_jobs_reject_changed_prepared_training_and_evaluation_data(self):
        record = self.upload(b'{}\n')
        path = self.manager.datasets.root / record['id'] / 'dataset.jsonl'
        path.write_bytes(b'original\n')
        self.manager.datasets.records[record['id']].update(status='ready', path=str(path), sha256=sha256(path))
        spec = self.gliner_spec()
        del spec['peer_id'], spec['coordinator']
        spec['execution_mode'] = 'local'
        path.write_bytes(b'modified\n')
        for field in ('dataset_id', 'calibration_dataset_id', 'test_dataset_id'):
            for preflight in (True, False):
                with self.subTest(field=field, preflight=preflight), \
                        patch.object(self.manager, 'run_peer', return_value={}), \
                        patch.object(self.manager, 'managed_run') as launch:
                    job = self.manager.start({**spec, field: record['id'], 'request_id': f'changed-{field}-{preflight}'}, preflight)
                    wait_for(lambda: not self.manager.workers)
                    failed = self.manager.jobs[job['id']]
                    self.assertEqual(failed['status'], 'failed')
                    self.assertIn('bytes changed', failed['error'])
                    launch.assert_not_called()

    @unittest.skipUnless(os.environ.get('ANTFLY_TRAINING_TEST_BINARY'), 'requires built native CLI')
    def test_real_native_gliner_import_selects_snapshot_and_rejects_invalid_offsets(self):
        (self.root / 'tools/bin').mkdir()
        (self.root / 'tools/bin/antfly-inference').symlink_to(os.environ['ANTFLY_TRAINING_TEST_BINARY'])
        data = ('\n'.join(json.dumps(example(identity)) for identity in ('one', 'two')) + '\n').encode()
        record = self.upload(data)
        self.manager.request('POST', f'datasets/{record["id"]}/prepare', {})
        wait_for(lambda: not self.manager.workers, 20)
        ready = self.manager.datasets.ready(record['id'], 'gliner25')
        self.assertEqual(ready['row_count'], 2)
        job = self.gliner_spec()
        before = Path(job['gliner25_config']).read_bytes()
        chosen = self.manager.validate_spec({**job, 'dataset_id': record['id'], 'calibration_dataset_id': None})
        self.assertEqual(self.manager.gliner_job(chosen)['train_file'], ready['path'])
        self.assertIsNone(self.manager.gliner_job(chosen)['calibration_file'])
        self.assertEqual(Path(job['gliner25_config']).read_bytes(), before)
        bad = example('bad')
        bad['entities'][0]['span']['end'] = 3  # Interior of the UTF-8 ë sequence.
        invalid = self.upload((json.dumps(bad) + '\n').encode(), request_id='invalid-dataset-02')
        self.manager.request('POST', f'datasets/{invalid["id"]}/prepare', {})
        wait_for(lambda: not self.manager.workers, 20)
        self.assertEqual(self.manager.datasets.records[invalid['id']]['status'], 'failed')
        self.assertIn('InvalidUtf8Boundary', self.manager.datasets.records[invalid['id']]['error'])

    @unittest.skipUnless(os.environ.get('ANTFLY_TRAINING_TEST_BINARY'), 'requires built native CLI')
    def test_real_native_gemma_preparation_binds_tokenizer_and_masks_prompt(self):
        (self.root / 'tools/bin').mkdir()
        (self.root / 'tools/bin/antfly-inference').symlink_to(os.environ['ANTFLY_TRAINING_TEST_BINARY'])
        model = self.root / 'models'
        tokenizer = {'version': '1.0', 'added_tokens': [], 'pre_tokenizer': {'type': 'Whitespace'},
                     'model': {'type': 'WordPiece', 'unk_token': '[UNK]', 'continuing_subword_prefix': '##',
                               'max_input_chars_per_word': 100, 'vocab': {'[UNK]': 0, '<bos>': 1, '<eos>': 2,
                               '<start_of_turn>': 3, '<end_of_turn>': 4, 'user': 5, 'model': 6, 'Hello': 7, 'World': 8}}}
        (model / 'tokenizer.json').write_text(json.dumps(tokenizer))
        (model / 'config.json').write_text('{"model_type":"gemma4","vocab_size":9}')
        (model / 'tokenizer_config.json').write_text('{"tokenizer_class":"GemmaTokenizerFast","bos_token":"<bos>","eos_token":"<eos>"}')
        data = b'{"instruction":"Hello","output":"World"}\n' * 3
        record = self.upload(data, family='gemma4', format='gemma_instruction', model_dir=str(model), max_seq_len=64)
        self.manager.request('POST', f'datasets/{record["id"]}/prepare', {})
        wait_for(lambda: not self.manager.workers, 20)
        ready = self.manager.datasets.ready(record['id'], 'gemma4')
        summary = json.loads(Path(ready['path']).read_text())['summary']
        self.assertEqual(len(summary['examples']), 3)
        for row in summary['examples']:
            self.assertGreater(row['num_supervised_tokens'], 0)
            self.assertIn(-100, row['labels'])
        self.assertEqual(ready['tokenizer_sha256'], tokenizer_sha256(model))
        (model / 'tokenizer_config.json').write_text('{"tokenizer_class":"Changed"}')
        self.assertNotEqual(ready['tokenizer_sha256'], tokenizer_sha256(model))


if __name__ == '__main__':
    unittest.main()
