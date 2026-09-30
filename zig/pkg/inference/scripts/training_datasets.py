#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc. SPDX-License-Identifier: Apache-2.0
"""Bounded dataset import and preparation for the local training coordinator."""
import base64
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import threading
import time
from urllib.parse import urlencode
from urllib.request import build_opener, HTTPRedirectHandler, Request
from urllib.error import HTTPError
import uuid

MAX_FILE = 64 * 1024 * 1024
MAX_ROW = 1024 * 1024
CHUNK_SIZE = 32768
MAX_ROWS = 10000
FORMATS = {'gliner25': {'gliner25', 'gliner_bio'},
           'gemma4': {'gemma_chat', 'gemma_instruction', 'gemma_completion'}}
ACTIVE = {'importing', 'preparing'}


class DatasetError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def number(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise DatasetError(f'{label} must be an integer between {low} and {high}')
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_sha256(model):
    model = Path(model)
    if not any((model / name).is_file() for name in ('tokenizer.json', 'tokenizer.model', 'vocab.json', 'vocab.txt')):
        raise DatasetError('dataset preparation requires tokenizer files beside the Gemma model')
    digest = hashlib.sha256()
    for name in ('tokenizer.json', 'tokenizer.model', 'tokenizer_config.json', 'special_tokens_map.json',
                 'added_tokens.json', 'vocab.json', 'vocab.txt', 'merges.txt', 'config.json', 'chat_template.jinja'):
        path = model / name
        digest.update(name.encode() + b'\0')
        digest.update(sha256(path).encode() if path.is_file() else b'absent')
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as output:
        json.dump(value, output, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def repository(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]{0,95}(?:/[A-Za-z0-9_-][A-Za-z0-9_.-]{0,95})?', value) or '..' in value:
        raise DatasetError('use a Hugging Face dataset ID such as organization/dataset')
    return value


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise DatasetError('the dataset service returned an unexpected redirect')


def hf_get(endpoint, parameters):
    # Never fetch URLs embedded in rows, media columns, or dataset scripts.
    url = 'https://datasets-server.huggingface.co/' + endpoint + '?' + urlencode(parameters)
    try:
        with build_opener(NoRedirect).open(Request(url, headers={'User-Agent': 'Antfly-training/1'}), timeout=10) as response:
            data = response.read(8 * 1024 * 1024 + 1)
    except HTTPError as error:
        raise DatasetError(f'Hugging Face returned HTTP {error.code}. Use a public dataset with Dataset Viewer support.') from error
    if len(data) > 8 * 1024 * 1024:
        raise DatasetError('Hugging Face response exceeds 8 MiB; choose a smaller row range')
    return json.loads(data)


def preview_row(row):
    # Explicitly truncated display only; preparation always uses complete rows.
    return json.dumps(row, ensure_ascii=False).encode('utf-8')[:2048].decode('utf-8', errors='ignore')


def hf_preview(body):
    dataset = repository(body.get('dataset'))
    if not body.get('config') or not body.get('split'):
        payload = hf_get('splits', {'dataset': dataset})
        splits = payload.get('splits', [])
        if not splits:
            raise DatasetError('No supported subsets/splits were found for this public dataset')
        return {'splits': [{'config': row['config'], 'split': row['split']} for row in splits[:200]], 'features': [], 'rows': []}
    for key in ('config', 'split'):
        if not isinstance(body[key], str) or not 0 < len(body[key]) <= 256:
            raise DatasetError('invalid Hugging Face ' + key)
    payload = hf_get('rows', {'dataset': dataset, 'config': body['config'], 'split': body['split'], 'offset': 0, 'length': 3})
    return {'splits': [], 'features': payload.get('features', [])[:128],
            'rows': [preview_row(row['row']) for row in payload.get('rows', [])],
            'total_rows': payload.get('num_rows_total')}


def validate_spec(spec):
    allowed = {'request_id', 'name', 'family', 'source', 'format', 'filename', 'size_bytes',
               'hf_dataset', 'hf_config', 'hf_split', 'hf_offset', 'columns', 'label_names',
               'max_rows', 'max_seq_len', 'model_dir'}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise DatasetError('unknown dataset configuration fields')
    if len(json.dumps(spec, allow_nan=False).encode()) > 32768:
        raise DatasetError('dataset configuration exceeds 32 KiB')
    if not re.fullmatch(r'[a-zA-Z0-9_-]{8,80}', str(spec.get('request_id', ''))):
        raise DatasetError('a stable dataset request_id is required')
    if not isinstance(spec.get('name'), str) or not 1 <= len(spec['name']) <= 128:
        raise DatasetError('dataset name must contain 1-128 characters')
    if spec.get('format') not in FORMATS.get(spec.get('family'), set()):
        raise DatasetError('dataset format does not match the model family')
    number(spec.get('max_rows', 1000), 1, MAX_ROWS, 'max_rows')
    if spec.get('source') == 'upload':
        name = spec.get('filename', '')
        if not isinstance(name, str) or len(name) > 255 or Path(name).suffix.lower() not in ('.csv', '.jsonl', '.ndjson'):
            raise DatasetError('upload a UTF-8 CSV or JSONL file')
        number(spec.get('size_bytes'), 1, MAX_FILE, 'size_bytes')
    elif spec.get('source') == 'huggingface':
        repository(spec.get('hf_dataset'))
        for key in ('hf_config', 'hf_split'):
            if not isinstance(spec.get(key), str) or not 0 < len(spec[key]) <= 256:
                raise DatasetError(key + ' is required')
        number(spec.get('hf_offset', 0), 0, 1000000000, 'hf_offset')
    else:
        raise DatasetError('dataset source must be upload or huggingface')
    columns = spec.get('columns', {})
    if not isinstance(columns, dict) or set(columns) - {'messages', 'prompt', 'input', 'response', 'text', 'tokens', 'tags'}:
        raise DatasetError('unsupported column mapping')
    if any(not isinstance(value, str) or not 1 <= len(value) <= 128 for value in columns.values()):
        raise DatasetError('column names must contain 1-128 characters')
    labels = spec.get('label_names', [])
    if not isinstance(labels, list) or len(labels) > 1024 or any(not isinstance(label, str) or not 1 <= len(label) <= 128 for label in labels):
        raise DatasetError('invalid BIO label names')
    if spec['family'] == 'gemma4':
        number(spec.get('max_seq_len', 512), 8, 8192, 'max_seq_len')
        if spec.get('max_rows', 1000) * spec.get('max_seq_len', 512) > 2 * 1024**2:
            raise DatasetError('Gemma preparation is limited to 2,097,152 tokens (maximum rows × sequence length); reduce either setting')
        if not isinstance(spec.get('model_dir'), str) or not Path(spec['model_dir']).is_absolute():
            raise DatasetError('Gemma preparation requires an absolute model directory')
    return copy.deepcopy(spec)


def cell(row, key, structured=False):
    value = row.get(key)
    if structured and isinstance(value, str):
        value = json.loads(value)
    return value


def normalize(row, spec, index, features=()):
    if not isinstance(row, dict):
        raise DatasetError(f'row {index + 1} must be an object')
    kind, columns = spec['format'], spec.get('columns', {})
    if kind == 'gliner25':
        if row.get('version') != 1 or not isinstance(row.get('schema'), dict):
            raise DatasetError('Native GLiNER rows require version 1, id, text, schema, and annotations')
        return row
    if kind == 'gliner_bio':
        tokens = cell(row, columns.get('tokens', 'tokens'), True)
        tags_column = columns.get('tags', 'ner_tags')
        tags = cell(row, tags_column, True)
        labels = spec.get('label_names', [])
        if not labels:
            for feature in features:
                if feature.get('name') == tags_column:
                    value = feature.get('type', {})
                    while isinstance(value, dict) and 'feature' in value:
                        value = value['feature']
                    labels = value.get('names', []) if isinstance(value, dict) else []
        if not labels or any(label != 'O' and not re.fullmatch(r'[BI]-.+', label) for label in labels):
            raise DatasetError('BIO conversion requires an ordered label list containing O and B-/I- labels')
        if not isinstance(tokens, list) or not isinstance(tags, list) or not tokens or len(tokens) != len(tags):
            raise DatasetError('tokens and BIO tags must be nonempty arrays of equal length')
        if any(not isinstance(token, str) or not token or any(c.isspace() for c in token) for token in tokens):
            raise DatasetError('BIO tokens must be nonempty strings without whitespace')
        text = ' '.join(tokens)
        types = list(dict.fromkeys(label[2:] for label in labels if label != 'O'))
        entities, current, cursor = [], None, 0
        for token, tag in zip(tokens, tags):
            if type(tag) is int:
                if not 0 <= tag < len(labels):
                    raise DatasetError('BIO tag index is outside the ordered label list')
                tag = labels[tag]
            if tag not in labels:
                raise DatasetError('BIO tag is not in the declared label list')
            end = cursor + len(token.encode('utf-8'))
            if tag == 'O':
                current = None
            elif tag.startswith('B-') or current is None or current['type'] != tag[2:]:
                current = {'id': f'e{len(entities)}', 'type': tag[2:], 'span': {'start': cursor, 'end': end, 'unit': 'utf8_bytes'}}
                entities.append(current)
            else:
                current['span']['end'] = end
            cursor = end + 1
        return {'version': 1, 'id': f'row-{index}', 'text': text, 'schema': {'entities': types}, 'entities': entities}
    if kind == 'gemma_chat':
        messages = cell(row, columns.get('messages', 'messages'), True)
        if not isinstance(messages, list) or not messages:
            raise DatasetError('chat messages must be a nonempty array')
        for message in messages:
            if not isinstance(message, dict) or message.get('role') not in ('system', 'user', 'assistant') or not isinstance(message.get('content'), str):
                raise DatasetError('text chat imports require system/user/assistant messages with string content')
        if not any(message['role'] == 'assistant' and message['content'].strip() for message in messages):
            raise DatasetError('each chat row needs a nonempty assistant response')
        return {'schema': 'gemma_chat/v1', 'id': f'row-{index}', 'messages': messages}
    if kind == 'gemma_completion':
        text = cell(row, columns.get('text', 'text'))
        if not isinstance(text, str) or not text.strip():
            raise DatasetError('completion text must be a nonempty string')
        return {'text': text}
    prompt = cell(row, columns.get('prompt', 'instruction'))
    response = cell(row, columns.get('response', 'output'))
    extra = cell(row, columns['input']) if columns.get('input') else None
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(response, str) or not response.strip():
        raise DatasetError('instruction imports require nonempty prompt and response strings')
    if extra is not None and not isinstance(extra, str):
        raise DatasetError('the optional input column must contain text')
    return {'prompt': prompt + ('\n\n' + extra if extra else ''), 'response': response}


def source_rows(spec, source_path):
    maximum = spec.get('max_rows', 1000)
    if spec['source'] == 'huggingface':
        offset = spec.get('hf_offset', 0)
        while maximum:
            length = min(100, maximum)
            payload = hf_get('rows', {'dataset': spec['hf_dataset'], 'config': spec['hf_config'],
                                     'split': spec['hf_split'], 'offset': offset, 'length': length})
            rows = payload.get('rows', [])
            if len(rows) > length:
                raise DatasetError('Hugging Face returned too many rows')
            for entry in rows:
                if entry.get('truncated_cells'):
                    raise DatasetError('Hugging Face returned truncated cells; use a complete file upload')
                if entry.get('row_idx') != offset:
                    raise DatasetError('Hugging Face returned a noncontiguous row range')
                yield entry['row'], payload.get('features', [])
                offset += 1
                maximum -= 1
            if len(rows) < length:
                break
        return
    csv.field_size_limit(MAX_ROW)
    with Path(source_path).open(encoding='utf-8-sig', newline='') as source:
        if Path(spec['filename']).suffix.lower() == '.csv':
            rows = csv.DictReader(source)
            if not rows.fieldnames or len(set(rows.fieldnames)) != len(rows.fieldnames):
                raise DatasetError('CSV needs unique column headers')
            for index, row in enumerate(rows):
                if index >= maximum:
                    break
                if None in row or any(value is None for value in row.values()):
                    raise DatasetError('CSV row does not match its column headers')
                yield row, []
        else:
            count = 0
            while count < maximum:
                line = source.readline(MAX_ROW + 1)
                if not line:
                    break
                if len(line.encode('utf-8')) > MAX_ROW:
                    raise DatasetError('dataset row exceeds 1 MiB')
                if line.strip():
                    yield json.loads(line), []
                    count += 1


def convert(task):
    spec = validate_spec(task['spec'])
    count, size, samples = 0, 0, []
    target = Path(task['output'])
    with target.open('wb') as output:
        for row, features in source_rows(spec, task['source']):
            try:
                normalized = normalize(row, spec, count, features)
            except (ValueError, TypeError, KeyError) as error:
                raise DatasetError(f'row {count + 1}: {error}') from error
            data = (json.dumps(normalized, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
            if len(data) > MAX_ROW or size + len(data) > MAX_FILE:
                raise DatasetError('prepared dataset exceeds the 1 MiB row or 64 MiB file limit')
            output.write(data)
            if len(samples) < 3:
                samples.append(preview_row(normalized))
            count += 1
            size += len(data)
        output.flush()
        os.fsync(output.fileno())
    if not count:
        raise DatasetError('the selected row range contains no examples')
    return {'row_count': count, 'size_bytes': size, 'sha256': sha256(target), 'preview': samples}


class DatasetStore:
    def __init__(self, owner):
        self.owner = owner
        self.root = Path(owner.config['output_root']) / 'datasets'
        self.root.mkdir(mode=0o700, exist_ok=True)
        self.metadata = owner.state / 'datasets'
        self.metadata.mkdir(mode=0o700, exist_ok=True)
        self.records = {}
        for path in self.metadata.glob('*.json'):
            if path.stat().st_size > 65536:
                raise DatasetError('dataset record exceeds metadata limit')
            record = json.loads(path.read_text())
            if not re.fullmatch(r'[0-9a-f]{32}', record['id']) or record['id'] != path.stem:
                raise DatasetError('invalid dataset record identity')
            if record['status'] in ACTIVE:
                record.update(status='failed', error='Preparation was interrupted by a coordinator restart; retry preparation.')
                self.save(record)
            self.records[record['id']] = record

    def save(self, record):
        record['updated_at'] = time.time()
        atomic_json(self.metadata / (record['id'] + '.json'), record)

    def public(self, record):
        result = copy.deepcopy(record)
        result.pop('fingerprint', None)
        if result['spec']['source'] == 'upload':
            result['uploaded_bytes'] = (self.root / record['id'] / 'source').stat().st_size
        return result

    def create(self, body):
        spec = validate_spec(body)
        if spec['family'] == 'gemma4':
            from training_peer import contained
            contained(spec['model_dir'], self.owner.config['input_roots'])
        fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        with self.owner.lock:
            for record in self.records.values():
                if record['spec']['request_id'] == spec['request_id']:
                    if record['fingerprint'] != fingerprint:
                        raise DatasetError('dataset request_id was already used for another configuration', 409)
                    return self.public(record)
            if len(self.records) >= 50:
                raise DatasetError('remove an unused dataset before importing more (limit 50)', 409)
            if shutil.disk_usage(self.root).free < MAX_FILE * 4 + 256 * 1024**2:
                raise DatasetError('insufficient disk headroom to import a dataset')
            identity = uuid.uuid4().hex
            folder = self.root / identity
            folder.mkdir(mode=0o700)
            (folder / 'source').touch(mode=0o600)
            record = {'id': identity, 'name': spec['name'], 'family': spec['family'], 'spec': spec,
                      'fingerprint': fingerprint, 'status': 'uploading' if spec['source'] == 'upload' else 'pending',
                      'created_at': time.time()}
            try:
                self.save(record)
            except BaseException:
                shutil.rmtree(folder)
                raise
            self.records[identity] = record
            return self.public(record)

    def chunk(self, record, body):
        with self.owner.lock:
            if record['status'] not in ('uploading', 'cancelled') or record['spec']['source'] != 'upload':
                raise DatasetError('dataset is not accepting upload chunks', 409)
            offset = number(body.get('offset'), 0, MAX_FILE, 'offset')
            encoded = body.get('data', '')
            if not isinstance(encoded, str) or len(encoded) > 4 * ((CHUNK_SIZE + 2) // 3):
                raise DatasetError('upload chunk exceeds 32 KiB')
            try:
                data = base64.b64decode(encoded, validate=True)
            except ValueError as error:
                raise DatasetError('invalid upload chunk encoding') from error
            if not 0 < len(data) <= CHUNK_SIZE or offset + len(data) > record['spec']['size_bytes']:
                raise DatasetError('upload chunk exceeds the declared file size')
            path = self.root / record['id'] / 'source'
            size = path.stat().st_size
            if offset < size:
                with path.open('rb') as source:
                    source.seek(offset)
                    if source.read(len(data)) != data:
                        raise DatasetError('retried upload chunk differs from the stored bytes', 409)
            elif offset == size:
                with path.open('ab') as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
            else:
                raise DatasetError('upload offset does not match the stored bytes', 409)
            if record['status'] != 'uploading':
                record['status'] = 'uploading'
                self.save(record)
            return self.public(record)

    def prepare(self, record):
        owner = self.owner
        with owner.lock:
            if record['status'] in ACTIVE or record['status'] == 'ready':
                return self.public(record)
            if owner.workers or owner.closing.is_set():
                raise DatasetError('another training or dataset operation is active', 409)
            if record['spec']['source'] == 'upload' and (self.root / record['id'] / 'source').stat().st_size != record['spec']['size_bytes']:
                raise DatasetError('finish uploading the complete file before preparation', 409)
            cancel = threading.Event()
            record.update(status='importing', error=None)
            self.save(record)
            def work():
                owner.operation.deadline = time.monotonic() + 900
                try:
                    directory = self.root / record['id']
                    spec = record['spec']
                    path = directory / 'dataset.jsonl'
                    task = {'spec': spec, 'source': str(directory / 'source'), 'output': str(path)}
                    summary = json.loads(owner.capture([sys.executable, str(Path(__file__).resolve()), 'convert', json.dumps(task)], cancel, 900))
                    with owner.lock:
                        record.update(summary, status='preparing')
                        self.save(record)
                    binary = str(Path(owner.config['toolchain_dir']) / 'bin/antfly-inference')
                    if record['family'] == 'gliner25':
                        owner.capture([binary, 'finetune', 'dataset', 'inspect', 'gliner25', str(path)], cancel, 300)
                    else:
                        from training_peer import check_input
                        model = check_input(spec['model_dir'], owner.config['input_roots'])
                        tokenizer_digest = tokenizer_sha256(model)
                        path = directory / 'prepared.json'
                        owner.capture([binary, 'finetune', 'dataset', 'prepare', 'gemma4-lora', str(model),
                                       str(directory / 'dataset.jsonl'), '-', str(path), '--max-examples', str(summary['row_count']),
                                       '--max-seq-len', str(spec.get('max_seq_len', 512)), '--quiet'], cancel, 600)
                        if path.stat().st_size > MAX_FILE:
                            raise DatasetError('tokenized dataset exceeds 64 MiB; import fewer rows or use shorter sequences')
                        if tokenizer_sha256(model) != tokenizer_digest:
                            raise DatasetError('the tokenizer changed during preparation; retry with a stable model')
                        owner.capture([binary, 'finetune', 'train', 'run', 'gemma4-lora', str(model), str(model), str(path),
                                       str(directory / 'unused-output'), '--trainer', 'autodiff', '--max-examples', '0', '--validate-local-only'], cancel, 60)
                        record['tokenizer_sha256'] = tokenizer_digest
                    if cancel.is_set() or owner.closing.is_set():
                        raise DatasetError('dataset preparation cancelled')
                    with owner.lock:
                        record.update(status='ready', path=str(path), size_bytes=path.stat().st_size, sha256=sha256(path))
                except Exception as error:
                    with owner.lock:
                        record.update(status='cancelled' if cancel.is_set() or owner.closing.is_set() else 'failed', error=str(error))
                finally:
                    with owner.lock:
                        try:
                            self.save(record)
                        finally:
                            owner.workers.pop(record['id'], None)
                            owner.controls.pop(record['id'], None)
            worker = threading.Thread(target=work, daemon=True)
            owner.workers[record['id']] = worker
            owner.controls[record['id']] = {'cancel': cancel}
            worker.start()
            return self.public(record)

    def ready(self, identity, family):
        with self.owner.lock:
            record = self.records.get(identity)
            if not record or record['status'] != 'ready' or record['family'] != family:
                raise DatasetError('select a prepared dataset for the chosen model family', 409)
            return copy.deepcopy(record)

    def verify(self, record):
        from training_peer import contained
        path = contained(record['path'], [str(self.root)])
        if path.stat().st_size > MAX_FILE or sha256(path) != record['sha256']:
            raise DatasetError('prepared dataset bytes changed; import it again')
        return path

    def stage(self, record, peer, cancel):
        from launch_jaccl_finetune import remote_command
        path = self.verify(record)
        task = {'action': 'stage_dataset', 'config': self.owner.config, 'dataset_id': record['id'],
                'filename': path.name, 'size_bytes': path.stat().st_size, 'sha256': record['sha256']}
        command = remote_command(peer['ssh_destination'], ['python3', '-c',
            (Path(__file__).parent / 'training_peer.py').read_text(), json.dumps(task)])
        with path.open('rb') as source:
            result = json.loads(self.owner.capture(command, cancel, 600, stdin=source))
        if result.get('sha256') != record['sha256']:
            raise DatasetError('remote dataset hash did not match')

    def request(self, method, parts, body):
        with self.owner.lock:
            if len(parts) == 1:
                if method == 'GET':
                    return 200, {'datasets': [self.public(row) for row in sorted(self.records.values(), key=lambda row: row['created_at'], reverse=True)]}
                if method == 'POST':
                    return 201, self.create(body)
            record = self.records.get(parts[1]) if len(parts) > 1 else None
            if record is None:
                raise DatasetError('dataset not found', 404)
            if method == 'GET' and len(parts) == 2:
                return 200, self.public(record)
            if method == 'PUT' and parts[2:] == ['chunks']:
                return 200, self.chunk(record, body)
            if method == 'POST' and parts[2:] == ['prepare']:
                return 202, self.prepare(record)
            if method == 'POST' and parts[2:] == ['cancel']:
                control = self.owner.controls.get(record['id'])
                if control:
                    control['cancel'].set()
                else:
                    if record['status'] == 'ready':
                        raise DatasetError('dataset preparation has already completed', 409)
                    record['status'] = 'cancelled'
                    self.save(record)
                return 200, self.public(record)
            if method == 'DELETE' and len(parts) == 2:
                if record['id'] in self.owner.workers:
                    raise DatasetError('cancel preparation before removing the dataset', 409)
                for job in self.owner.jobs.values():
                    if any(job['spec'].get(key) == record['id'] for key in ('dataset_id', 'calibration_dataset_id', 'test_dataset_id')):
                        raise DatasetError('a retained training job references this dataset', 409)
                shutil.rmtree(self.root / record['id'])
                (self.metadata / (record['id'] + '.json')).unlink()
                del self.records[record['id']]
                return 200, {'removed': True}
        raise DatasetError('dataset operation not found', 404)


if __name__ == '__main__':
    signal.alarm(900)
    try:
        action, task = sys.argv[1], json.loads(sys.argv[2])
        result = hf_preview(task) if action == 'huggingface' else convert(task) if action == 'convert' else None
        if result is None:
            raise DatasetError('unsupported dataset helper operation')
        print(json.dumps(result, allow_nan=False))
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
