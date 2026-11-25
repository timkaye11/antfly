#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Source-bound checkpoint qualification phases; no implicit release authority."""
import argparse
import copy
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import context_release_build_adapter as builds
import native_whole_head_campaign as contract
import native_whole_head_campaign as evidence
from native_whole_head_quality import EXECUTION_FIELDS, configuration_contract
from native_repeat_campaign_resources import run_guarded_combined
import native_repeat_campaign_resources as resources
import qualify_metal_native_checkpoint as models
from qualify_context_management import git_provenance
from record_context_release_build import source_snapshot
from metal_checkpoint_qualification_ledger import validate_report
from qualify_native_checkpoint_http import (
    REPO, canonical_choices, generate, normalize_request_body, prepare_headers,
    requests, resume, server, sha256_file, validate_metal_memory,
)
from qualify_native_cancellation import observe_guard


FILTER = 'context Metal native checkpoint paired quality qualification'

FLAGS = {field: 'ANTFLY_METAL_ATTENTION_MATCH_' + field.upper() for field in EXECUTION_FIELDS}

FLAGS['resident_gated_ffn_fusion'] = 'ANTFLY_METAL_ATTENTION_MATCH_GATED_FFN_FUSION'

SCRIPTS = Path(__file__).resolve().parent

ROUTE_FLAGS = {'ANTFLY_METAL_QUALITY_ORDINARY_PREPARED_GREEDY': '1', 'ANTFLY_METAL_QUALITY_SERVING_PREFILL': '0'}

BUCKETS = frozenset(('document_qa', 'exact_retrieval', 'code', 'structured_tool', 'multiturn'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for raw in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(raw)
    return h.hexdigest()

def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

def reference(root, path):
    path = Path(path)
    contract.require(path.is_relative_to(root) and path.is_file() and not path.is_symlink(), 'unsafe output reference')
    return {'path': str(path.relative_to(root)), 'sha256': sha(path)}

def validate_freeze(repo, root, freeze_ref):
    """Authenticate the complete runtime and frozen corpus before any gold read or model load."""
    frozen = contract.json_ref(root, freeze_ref, 'native campaign freeze')
    contract.require(type(frozen) is dict and frozen.get('schema') == 'antfly.native-whole-head.campaign-freeze.v2' and
                     frozen.get('release_authority') is False and frozen.get('runtime_frozen') is True,
                     'unsupported or unfrozen whole-head campaign')
    encodings = contract.qualification_encodings(frozen)
    build_path = contract.regular(root, frozen['build']['path'], 'native build')
    contract.read_ref(root, frozen['build'], 'native build')
    build = builds.validate(build_path)
    current_head, closure = source_snapshot(repo)
    contract.require(current_head == build['head'] and closure == build['source'] and
                     git_provenance(repo) == frozen['source'], 'runtime source changed after campaign freeze')
    binary = {'path': str(build['binary_path']), 'sha256': build['binary_sha256']}
    contract.require(frozen.get('binary') == binary, 'frozen runtime executable changed')
    configuration = contract.json_ref(root, frozen['configuration'], 'native configuration probe')
    contract.require(configuration.get('passed') is True and configuration.get('source') == frozen['source'] and
                     configuration.get('binary') == binary, 'configuration probe does not attest the frozen executable')
    configuration_contract(configuration['configuration'], configuration['configuration'])
    implementation = configuration['configuration'].get('implementation_sha256')
    contract.require(type(implementation) is list and len(implementation) == 32 and
                     all(type(value) is int and 0 <= value <= 255 for value in implementation) and any(implementation) and
                     frozen.get('implementation_sha256') == bytes(implementation).hex(),
                     'configuration probe omits the compiled implementation identity')
    expected_flags = {FLAGS[key]: '1' if configuration['configuration']['execution'][key] else '0' for key in configuration['configuration']['execution']}
    contract.require(frozen.get('runtime_environment') == expected_flags and
                     configuration.get('runtime_environment') == expected_flags, 'frozen execution flags changed')
    audit = contract.json_ref(root, frozen['configuration_audit'], 'configuration environment audit')
    contract.require(audit.get('passed') is True and audit.get('source') == frozen['source'] and
                     audit.get('probe_observations_sha256') == frozen['configuration']['sha256'] and
                     audit.get('backend_environment_sha256') == bytes(configuration['configuration']['observed']['backend_environment_sha256']).hex(),
                     'configuration environment was not independently audited')
    index = contract.json_ref(root, frozen['input_index'], 'frozen quality index')
    contract.require(index.get('schema') == 'antfly.native-quality-input-preparation.v1' and
                     index.get('preparation_only') is True and index.get('model_quality_outputs_observed') is False and
                     index.get('corpus_fingerprint_sha256') == contract.digest(frozen.get('corpus_fingerprint_sha256'), 'corpus'),
                     'quality inputs differ from the frozen fresh preparation')
    contract.require(contract.qualification_encodings(index) == encodings,
                     'quality input qualification encodings differ from frozen scope')
    # Do not decode reference text or scoring answers here. The worker reveals
    # each reference only after its two greedy branches; audits run afterwards.
    contract.read_ref(root, index['scoring_cases'], 'frozen scoring metadata', 64 * 1024 * 1024)
    preparation = contract.json_ref(root, index['preparation_audit'], 'preparation audit')
    preparation_seal = contract.json_ref(root, index['preparation_source_seal'], 'preparation source seal')
    contract.require(preparation.get('schema') == 'antfly.native-quality-preparation.audit.v2' and
                     preparation.get('passed') is True and preparation.get('prepared_cases') == 2000 and
                     preparation.get('model_quality_outputs_observed') is False and
                     preparation.get('model_quality_qualified') is False and preparation.get('release_eligible') is False and
                     preparation.get('protected_prefix_rows') == 128 and preparation.get('protected_recent_rows') == 2048 and
                     preparation.get('corpus_fingerprint_sha256') == frozen['corpus_fingerprint_sha256'],
                     'invalid or exposed preparation audit')
    contract.require(preparation_seal.get('schema') == 'antfly.native-quality-preparation.source-seal.v1' and
                     preparation_seal.get('preparation_only') is True and preparation_seal.get('release_eligible') is False and
                     preparation_seal.get('source') == preparation.get('source') and
                     preparation_seal.get('producer_binary') == preparation.get('binary'),
                     'preparation source and audit do not agree')
    archive = Path(preparation_seal['archive']['path'])
    contract.require(archive.is_absolute() and archive.is_relative_to(root), 'preparation archive escaped campaign')
    archive = contract.regular(root, str(archive.relative_to(root)), 'preparation archive')
    contract.require(archive.stat().st_size == preparation_seal['archive']['bytes'] and
                     sha(archive) == contract.digest(preparation_seal['archive']['sha256'], 'preparation archive'),
                     'preparation archive changed')
    preparation_cells = {(model, context, bucket) for model in contract.MODELS
                         for context in (8192, 32768) for bucket in BUCKETS}
    audited_cells = preparation.get('cells')
    contract.require(type(audited_cells) is list and len(audited_cells) == len(preparation_cells) and
                     {(row['model'], row['context_tokens'], row['bucket']) for row in audited_cells} == preparation_cells and
                     all(type(row['cases']) is int and row['cases'] == 100 for row in audited_cells),
                     'incomplete audited preparation cohort')
    expected_cells = {(model, encoding, context, bucket) for model in contract.MODELS
                      for encoding in encodings for context in (8192, 32768) for bucket in BUCKETS}
    batches = index['batches']
    contract.require(type(batches) is list and len(batches) == len(expected_cells), 'incomplete frozen native cohort')
    cells, identities = set(), {}
    for batch in batches:
        cell = (batch['model'], batch['encoding'], batch['context_tokens'], batch['bucket'])
        contract.require(cell in expected_cells and cell not in cells and type(batch['cases']) is int and batch['cases'] == 100,
                         'duplicate, missing or undersized frozen native cell')
        cells.add(cell)
        manifest = contract.json_ref(root, batch['input'], 'frozen native manifest')
        model = batch['model']
        contract.require(manifest.get('schema') == 'antfly.native-checkpoint-quality.input.v1' and
                         manifest.get('model') == contract.MODELS[model][0] and len(manifest['cases']) == 100,
                         'manifest differs from frozen native cell')
        identity = (manifest['main_gguf_sha256'], manifest['native_identity'])
        model_spec = frozen['models'][model]
        contract.require(type(manifest['native_identity']) is dict and
                         set(manifest['native_identity']) == {'weights', 'tokenizer', 'template', 'layout'} and
                         all(type(value) is list and len(value) == 32 and
                             all(type(byte) is int and 0 <= byte <= 255 for byte in value)
                             for value in manifest['native_identity'].values()) and
                         {key: bytes(value).hex() for key, value in manifest['native_identity'].items()} ==
                         model_spec['loaded_identity'] and
                         manifest['main_gguf_sha256'] == model_spec['artifact_sha256'],
                         'frozen model identity differs from native cohort')
        contract.require(model not in identities or identities[model] == identity, 'model identity varies between native cells')
        identities[model] = identity
        contract.require(len({case['id'] for case in manifest['cases']}) == 100 and
                         all(case['bucket'] == batch['bucket'] and case['prompt_count'] + 256 == batch['context_tokens'] and
                             case['request'] == {'encoding': batch['encoding'], 'output_tokens': 256, 'minimum_savings_ratio': 1.5}
                             for case in manifest['cases']), 'changed native cases or reservation')
    contract.require(cells == expected_cells, 'frozen cohort coverage changed')
    performance_cells = {(model, encoding, context) for model in contract.MODELS
                         for encoding in encodings for context in (8192, 32768)}
    observed_cells = set()
    for entry in frozen['performance']:
        cell = (entry['model'], entry['encoding'], entry['context_tokens'])
        contract.require(cell in performance_cells and cell not in observed_cells, 'repeated performance prerequisite')
        observed_cells.add(cell)
        report = contract.json_ref(root, entry['observations'], 'public performance', 256 * 1024 * 1024)
        resource = contract.json_ref(root, entry['resource'], 'public performance combined guard', 256 * 1024 * 1024)
        contract.combined_resource_contract(resource)
        worker_log = contract.read_ref(root, entry['worker_log'], 'public performance worker log', 256 * 1024 * 1024)
        job = contract.json_ref(root, entry['job'], 'public performance producer job')
        contract.require(job.get('source') == frozen['source'] and job.get('binary') == binary and
                         job.get('runtime_environment') == expected_flags, 'performance job configuration differs')
        for ref in job['producers']:
            contract.read_ref(root, ref, 'public performance producer')
        expected = {key: entry[key] for key in ('model', 'encoding', 'context_tokens', 'request')}
        expected.update(source=frozen['source'], binary=binary, loaded_identity=frozen['models'][entry['model']]['loaded_identity'])
        result = contract.performance_contract(report, expected, worker_log)
        contract.require(result['performance_gate_passed'], 'public performance prerequisite failed')
    contract.require(observed_cells == performance_cells, 'public performance prerequisite matrix is incomplete')
    return frozen, build, index, configuration['configuration']

def model_artifact(root, frozen, batch, manifest):
    spec = frozen['models'][batch['model']]
    directory = Path(spec['directory']).resolve(strict=True)
    registry_root = directory.parent.parent
    models.checked_model_registry(registry_root, directory, contract.MODELS[batch['model']][0])
    download = directory / '.antfly-download-complete.json'
    contract.require(sha(download) == spec['manifest_sha256'], 'model download manifest changed')
    contract.require(spec['artifact_sha256'] == manifest['main_gguf_sha256'], 'main model artifact differs from native inputs')
    models.checked_model_artifact(directory, spec['artifact'], spec['artifact_sha256'])
    return directory, registry_root

def worker(args):
    root, output = args.campaign_root.resolve(strict=True), args.output.resolve(strict=True)
    spec = contract.json_ref(output, {'path': 'spec.json', 'sha256': args.worker_spec_sha256}, 'quality worker spec')
    contract.read_ref(root, spec['manifest'], 'quality worker manifest')
    contract.require(git_provenance(args.root) == spec['source'] and sha(spec['binary']['path']) == spec['binary']['sha256'],
                     'quality worker runtime changed')
    flags = dict(spec['runtime_environment'], **ROUTE_FLAGS)
    flags.update(ANTFLY_METAL_NATIVE_QUALITY_INPUT=str(contract.regular(root, spec['manifest']['path'], 'native batch')),
                 ANTFLY_METAL_NATIVE_QUALITY_INPUT_ROOT=str(root), ANTFLY_METAL_NATIVE_QUALITY_OUTPUT_ROOT=str(output / 'outputs'),
                 ANTFLY_METAL_CHECKPOINT_MODELS_ROOT=spec['models_root'], ANTFLY_METAL_CHECKPOINT_MODEL=spec['request_model'])
    env = os.environ.copy()
    for key in ('ANTFLY_CONTEXT_PROFILE', 'TERMITE_METAL_TRACE_ENCODER_TIMING', 'TERMITE_METAL_TRACE_QUANT_BLOCK',
                'TERMITE_METAL_TRACE_DISPATCH_PROFILE', 'ANTFLY_METAL_QUALITY_CONFIGURATION_PROBE'):
        env.pop(key, None)
    env.update(flags)
    write_new(output / 'worker-environment.json', flags)
    models.headroom(output, 'native-whole-head-quality')
    with (output / 'quality.stdout').open('xb') as stdout, (output / 'quality.stderr').open('xb') as stderr:
        process = subprocess.run(spec['command'], env=env, stdout=stdout, stderr=stderr, timeout=spec['timeout_seconds'] - 30)
    return process.returncode

def collect(args):
    root, repo = args.campaign_root.resolve(strict=True), args.root.resolve(strict=True)
    freeze_ref = {'path': str(args.freeze), 'sha256': args.freeze_sha256}
    frozen, build, index, configuration = validate_freeze(repo, root, freeze_ref)
    contract.require(type(args.batch) is int and 0 <= args.batch < len(index['batches']), 'invalid batch index')
    batch = index['batches'][args.batch]
    manifest = contract.json_ref(root, batch['input'], 'native input')
    directory, registry_root = model_artifact(root, frozen, batch, manifest)
    output = args.output.absolute().resolve(strict=False)
    contract.require(output.is_relative_to(root) and not output.exists() and not output.is_symlink(), 'unsafe or existing quality output')
    # The advisory lock survives no worker process. A stale file after a crash
    # cannot trap a resumed campaign or permit overlapping model loads.
    with (root / 'native-whole-head-active.lock').open('a+') as active:
        fcntl.flock(active.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        (output / 'outputs').mkdir()
        timeout = 900 * batch['cases'] + 120
        command = [str(build['binary_path']), '--test-filter', FILTER]
        spec = {'schema': 'antfly.native-whole-head.quality-job.v2', 'source': frozen['source'],
                'binary': frozen['binary'], 'freeze': freeze_ref, 'manifest': batch['input'], 'batch': batch,
                'configuration_sha256': frozen['configuration']['sha256'], 'runtime_environment': frozen['runtime_environment'],
                'request_model': contract.MODELS[batch['model']][0], 'models_root': str(registry_root),
                'model_directory': str(directory), 'command': command, 'timeout_seconds': timeout,
                'producer_sha256': sha(__file__), 'release_authority': False, 'production_ready': False}
        write_new(output / 'spec.json', spec)
        spec_sha256 = sha(output / 'spec.json')
        # This immutable claim prevents a changed runtime from silently starting
        # a new campaign on a corpus whose model outputs have been exposed.
        claim = root / ('native-whole-head-exposure-' + frozen['corpus_fingerprint_sha256'] + '.json')
        expected_claim = {'schema': 'antfly.native-whole-head.exposure-claim.v2', 'freeze': freeze_ref,
                          'source': frozen['source'], 'binary': frozen['binary']}
        if claim.exists():
            contract.require(json.loads(claim.read_bytes()) == expected_claim, 'fresh corpus was exposed under a different runtime freeze')
        else:
            write_new(claim, expected_claim)
        child = [sys.executable, str(Path(__file__).resolve()), 'quality',
                 '--root', str(args.root), '--campaign-root', str(args.campaign_root),
                 '--freeze', str(args.freeze), '--freeze-sha256', args.freeze_sha256,
                 '--batch', str(args.batch), '--output', str(args.output),
                 '--worker', '--worker-spec-sha256', spec_sha256]
        guard = run_guarded_combined(child, output / 'resource.json', timeout=timeout, max_rss_mib=8192,
                                     min_free_percent=15, max_swap_growth_mib=0, min_disk_free_mib=4096)
        result = {'schema': 'antfly.native-whole-head.quality-batch-seal.v2', 'collection_complete': False,
                  'source': frozen['source'], 'binary': frozen['binary'], 'batch': batch,
                  'spec': reference(output, output / 'spec.json'), 'resource': reference(output, output / 'resource.json'),
                  'passed': False, 'full_serving_owner_qualified': False, 'production_ready': False, 'release_authority': False}
        try:
            contract.combined_resource_contract(guard)
            contract.require(sha(output / 'spec.json') == spec_sha256, 'quality worker spec changed')
            contract.require(git_provenance(repo) == frozen['source'] and sha(build['binary_path']) == build['binary_sha256'] and
                             sha(__file__) == spec['producer_sha256'], 'quality runtime or producer changed')
            model_artifact(root, frozen, batch, manifest)
            outputs = [reference(output / 'outputs', output / 'outputs' / f'case-{i:04}.json') for i in range(batch['cases'])]
            sampling = contract.json_ref(root, index['scoring_cases'], 'scoring metadata', 64 * 1024 * 1024)
            implementation = bytes(configuration['implementation_sha256']).hex()
            audit = contract.audit_batch(root, batch['input'], output / 'outputs', outputs, batch, configuration,
                                         implementation, sampling)
            write_new(output / 'audit.json', audit)
            result.update(passed=True, collection_complete=True, audit=reference(output, output / 'audit.json'), outputs=outputs)
        except Exception as error:
            result['error'] = f'{type(error).__name__}: {error}'
        finally:
            result['evidence'] = [reference(output, path) for path in sorted(output.rglob('*')) if path.is_file()]
            write_new(output / 'seal.json', result)
    return 0 if result['passed'] else 1

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--campaign-root', type=Path, required=True)
    p.add_argument('--freeze', type=Path, required=True, help='Immutable campaign-relative freeze path')
    p.add_argument('--freeze-sha256', required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--worker-spec-sha256', help=argparse.SUPPRESS)
    return p

def quality_main(argv=None):
    args = parser().parse_args(argv)
    return worker(args) if args.worker else collect(args)

SOAK_FILTER = 'context Metal checkpoint persistent installed owner soak qualification'

PREFIXES = {8192: 8128, 32768: 32512}

def checkpoint_ref(root, value):
    contract.require(type(value) is dict and set(value) == {'path', 'sha256'}, 'invalid soak checkpoint reference')
    path = contract.regular(root, value['path'], 'soak checkpoint')
    contract.require(0 < path.stat().st_size <= 2 * 1024**3 and
                     sha(path) == contract.digest(value['sha256'], 'soak checkpoint'),
                     'soak checkpoint changed')
    return path

def soak_worker(args):
    root, output = args.campaign_root.resolve(strict=True), args.output.resolve(strict=True)
    spec = contract.json_ref(output, {'path': 'spec.json', 'sha256': args.worker_spec_sha256}, 'soak worker spec')
    contract.require(git_provenance(args.root) == spec['source'] and
                     sha(spec['binary']['path']) == spec['binary']['sha256'], 'soak worker runtime changed')
    contract.read_ref(output, spec['input'], 'soak worker input')
    for ref in spec['checkpoints']:
        checkpoint_ref(root, ref)
    env = os.environ.copy()
    for key in ('ANTFLY_CONTEXT_PROFILE', 'TERMITE_METAL_TRACE_ENCODER_TIMING', 'TERMITE_METAL_TRACE_QUANT_BLOCK',
                'TERMITE_METAL_TRACE_Q80_BLOCK', 'TERMITE_METAL_TRACE_DISPATCH_PROFILE', 'TERMITE_METAL_TRACE_FRAME',
                'ANTFLY_METAL_QUALITY_ORDINARY_PREPARED_GREEDY', 'ANTFLY_METAL_QUALITY_SERVING_PREFILL',
                'ANTFLY_METAL_QUALITY_CONFIGURATION_PROBE'):
        env.pop(key, None)
    flags = dict(spec['runtime_environment'],
                 ANTFLY_METAL_CHECKPOINT_SOAK_INPUT=str(output / 'input.json'),
                 ANTFLY_METAL_CHECKPOINT_SOAK_OUTPUT=str(output / 'report.json'),
                 ANTFLY_METAL_CHECKPOINT_SOAK_JOURNAL=str(output / 'journal.jsonl'),
                 ANTFLY_METAL_CHECKPOINT_MODELS_ROOT=spec['models_root'],
                 ANTFLY_METAL_CHECKPOINT_MODEL=spec['request_model'])
    env.update(flags)
    write_new(output / 'worker-environment.json', flags)
    models.headroom(output, 'native-persistent-owner-soak')
    with (output / 'soak.stdout').open('xb') as stdout, (output / 'soak.stderr').open('xb') as stderr:
        process = subprocess.run(spec['command'], env=env, stdout=stdout, stderr=stderr,
                                 timeout=spec['timeout_seconds'] - 30)
    return process.returncode

def soak_collect(args):
    root, repo = args.campaign_root.resolve(strict=True), args.root.resolve(strict=True)
    freeze_ref = {'path': str(args.freeze), 'sha256': args.freeze_sha256}
    frozen, build, index, configuration = validate_freeze(repo, root, freeze_ref)
    contract.require(args.model in contract.MODELS and args.encoding in contract.qualification_encodings(frozen),
                     'soak cell outside frozen qualification scope')
    contract.require(type(args.duration_seconds) is int and 0 < args.duration_seconds <= evidence.FORMAL_SECONDS and
                     (args.diagnostic or args.duration_seconds == evidence.FORMAL_SECONDS),
                     'short soak requires explicit diagnostic scope')
    batch = next(row for row in index['batches'] if row['model'] == args.model and row['encoding'] == args.encoding)
    manifest = contract.json_ref(root, batch['input'], 'soak model identity')
    directory, models_root = model_artifact(root, frozen, batch, manifest)
    selected = [next(row for row in frozen['performance'] if
                     (row['model'], row['encoding'], row['context_tokens']) == (args.model, args.encoding, context))
                for context in (8192, 32768)]
    refs = [row['checkpoint'] for row in selected]
    paths = [checkpoint_ref(root, ref) for ref in refs]
    # Bind each checkpoint to the actual fresh preparation receipt, rather than
    # accepting a filename adjacent to otherwise qualified performance evidence.
    for row, ref in zip(selected, refs):
        report = contract.json_ref(root, row['observations'], 'soak performance preparation', 256 * 1024**2)
        preparations = [request['result'] for request in report['worker']['requests'] if request['kind'] == 'prepare']
        contract.require(len(preparations) == 2 and preparations[1]['file_sha256'] == ref['sha256'],
                         'soak checkpoint differs from the fresh public preparation')
    output = args.output.absolute().resolve(strict=False)
    contract.require(output.is_relative_to(root) and not output.exists() and not output.is_symlink(),
                     'unsafe or existing soak output')
    with (root / 'native-whole-head-active.lock').open('a+') as active:
        fcntl.flock(active.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        timeout = int(args.duration_seconds / .9) + 900
        value = {'schema': 'antfly.native-checkpoint-soak.input.v2', 'model': contract.MODELS[args.model][0],
                 'main_gguf_sha256': manifest['main_gguf_sha256'], 'native_identity': manifest['native_identity'],
                 'checkpoints': [{'path': str(path), 'sha256': ref['sha256']} for path, ref in zip(paths, refs)],
                 'request': {'encoding': args.encoding, 'duration_seconds': args.duration_seconds}}
        write_new(output / 'input.json', value)
        spec = {'schema': 'antfly.native-checkpoint-soak.job.v2', 'source': frozen['source'], 'binary': frozen['binary'],
                'freeze': freeze_ref, 'input': reference(output, output / 'input.json'),
                'checkpoints': refs, 'model': args.model, 'encoding': args.encoding,
                'duration_seconds': args.duration_seconds, 'diagnostic_only': args.diagnostic,
                'runtime_environment': frozen['runtime_environment'], 'request_model': contract.MODELS[args.model][0],
                'models_root': str(models_root), 'model_directory': str(directory),
                'command': [str(build['binary_path']), '--test-filter', SOAK_FILTER], 'timeout_seconds': timeout,
                'producer_sha256': sha(__file__), 'release_authority': False}
        write_new(output / 'spec.json', spec)
        spec_sha = sha(output / 'spec.json')
        child = [sys.executable, str(Path(__file__).resolve()), 'soak',
                 '--root', str(args.root), '--campaign-root', str(args.campaign_root),
                 '--freeze', str(args.freeze), '--freeze-sha256', args.freeze_sha256,
                 '--model', args.model, '--encoding', args.encoding,
                 '--output', str(args.output), '--duration-seconds', str(args.duration_seconds),
                 '--worker', '--worker-spec-sha256', spec_sha]
        if args.diagnostic:
            child.append('--diagnostic')
        guard = run_guarded_combined(child, output / 'resource.json', timeout=timeout, max_rss_mib=8192,
                                             min_free_percent=15, max_swap_growth_mib=0, min_disk_free_mib=4096)
        result = {'schema': 'antfly.native-checkpoint-soak.seal.v2', 'collection_complete': False, 'passed': False,
                  'source': frozen['source'], 'binary': frozen['binary'], 'spec': reference(output, output / 'spec.json'),
                  'resource': reference(output, output / 'resource.json'),
                  'production_ready': False, 'release_authority': False}
        try:
            contract.combined_resource_contract(guard)
            contract.require(git_provenance(repo) == frozen['source'] and
                             sha(build['binary_path']) == build['binary_sha256'] and
                             sha(output / 'spec.json') == spec_sha and sha(__file__) == spec['producer_sha256'],
                             'soak runtime, spec or producer changed')
            model_artifact(root, frozen, batch, manifest)
            for ref in refs:
                checkpoint_ref(root, ref)
            report_ref = reference(output, output / 'report.json')
            journal_ref = reference(output, output / 'journal.jsonl')
            report = contract.json_ref(output, report_ref, 'native soak report')
            journal = contract.read_ref(output, journal_ref, 'native soak journal', 256 * 1024**2)
            expected = {'model': args.model, 'encoding': args.encoding, 'model_directory': str(directory),
                        'input_sha256': spec['input']['sha256'], 'artifact_sha256': manifest['main_gguf_sha256'],
                        'implementation_sha256': frozen['implementation_sha256'], 'native_identity': manifest['native_identity'],
                        'configuration': configuration, 'duration_seconds': args.duration_seconds,
                        'prefix_tokens': [PREFIXES[context] for context in (8192, 32768)],
                        'checkpoint_sha256': [ref['sha256'] for ref in refs]}
            audit = evidence.audit(report, journal, guard, expected, formal=not args.diagnostic)
            write_new(output / 'audit.json', audit)
            result.update(passed=True, collection_complete=True, report=report_ref, journal=journal_ref,
                          audit=reference(output, output / 'audit.json'))
        except Exception as error:
            result['error'] = f'{type(error).__name__}: {error}'
        finally:
            result['evidence'] = [reference(output, path) for path in sorted(output.rglob('*')) if path.is_file()]
            write_new(output / 'seal.json', result)
    return 0 if result['passed'] else 1

def soak_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--model', choices=sorted(contract.MODELS), required=True)
    parser.add_argument('--encoding', choices=sorted(contract.ENCODINGS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration-seconds', type=int, default=evidence.FORMAL_SECONDS)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--worker-spec-sha256', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    return soak_worker(args) if args.worker else soak_collect(args)

ERRORS = {"allocation": "OutOfMemory", "nonfinite": "NonFiniteAttention",
          "gpu_completion": "GeneratedMetalDispatchFailed"}

def validate_fault(journal, kind):
    if (journal.get("schema") != "antfly.native-checkpoint-fault.v1"
            or journal.get("kind") != kind or journal.get("diagnostic_only") is not True
            or journal.get("quality_credit") is not False
            or journal.get("release_authority") is not False):
        raise AssertionError("invalid portable native fault journal")
    row = journal["observation"]
    if (row.get("injected") is not True or type(row.get("completed_gpu_batches")) is not int
            or row["completed_gpu_batches"] <= 0 or row.get("engine_error") != ERRORS[kind]
            or row.get("live_allocations") != 0 or row.get("live_bytes") != 0
            or row.get("source_preserved") is not True
            or row.get("source_before") != row.get("source_after")
            or row.get("revision_before") != row.get("revision_after")):
        raise AssertionError("fault lacks completed work, source preservation or candidate retirement")
    if any(type(row.get(key)) is not int or row[key] < 0 for key in (
            "live_allocations", "live_bytes", "revision_before", "revision_after")):
        raise AssertionError("invalid fault counters")
    digest = row.get("source_before")
    # Zig renders a byte array as a JSON string when it is valid UTF-8, and
    # as an integer array otherwise. Both encode the same exact source digest.
    if isinstance(digest, str):
        digest = list(digest.encode("utf-8"))
    if (not isinstance(digest, list) or len(digest) != 32 or not any(digest)
            or any(type(value) is not int or not 0 <= value <= 255 for value in digest)):
        raise AssertionError("invalid source digest")
    required = {"allocation": "allocator_failure_observed", "nonfinite": "finite_check_observed",
                "gpu_completion": "simulated_completion_error"}[kind]
    if row.get(required) is not True or (kind == "nonfinite" and row.get("poisoned_input") is not True):
        raise AssertionError("fault lacks actual triggering operation")
    return row

def read_failure_evidence(path, offset, kind):
    data = path.read_bytes()[offset:]
    if len(data) > 4 * 1024 * 1024:
        raise AssertionError("excessive fault diagnostics")
    faults, phases = [], []
    for line in data.decode().splitlines():
        if line.startswith("native_checkpoint_fault "):
            faults.append(json.loads(line.partition(" ")[2]))
        elif line.startswith("metal_checkpoint_qualification "):
            phases.append(json.loads(line.partition(" ")[2]))
    if len(faults) != 1:
        raise AssertionError("missing or duplicate native fault journal")
    validate_fault(faults[0], kind)
    starts = [row for row in phases if row.get("phase") == "prefill_started"
              and row.get("route") == "checkpoint_prepare"]
    if len(starts) != 1:
        raise AssertionError("missing fresh portable prepare owner")
    owner = starts[0]
    identity = tuple(owner[key] for key in ("request", "route", "owner_id"))
    owned = [row for row in phases
             if tuple(row.get(key) for key in ("request", "route", "owner_id")) == identity]
    released = [row for row in owned if row.get("phase") == "final_release"]
    if (len(released) != 1 or released[0].get("released") is not True
            or released[0].get("post_release_live_bytes") != 0
            or any(row.get("phase") == "generation_committed" for row in owned)):
        raise AssertionError("failed prepare published a candidate or retained its owner")
    return dict(journal=faults[0], prepare_owner=owner, release=released[0])

def fault_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--backend", choices=["metal"], default="metal")
    parser.add_argument("--server-kind", choices=["metal-test"], default="metal-test")
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--encoding", choices=["f16", "fp8"], required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fault", choices=ERRORS, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.fixture = "fact"
    args.qualification_minimum_savings = 1.5
    args.inference_admission_capacity = 48
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence = dict(passed=False, production_qualified=False, release_authority=False,
                    backend="metal", server_kind="metal-test", fault=args.fault,
                    encoding=args.encoding, source=git_provenance(REPO),
                    binary_sha256=sha256_file(args.server), runner_sha256=sha256_file(Path(__file__)),
                    reference_sha256=sha256_file(args.reference),
                    checkpoint_sha256=sha256_file(args.checkpoint), public_synthetic_fixture=True)
    previous = os.environ.get("ANTFLY_METAL_CHECKPOINT_NATIVE_FAULT")
    os.environ["ANTFLY_METAL_CHECKPOINT_NATIVE_FAULT"] = args.fault
    try:
        reference = json.loads(args.reference.read_bytes())
        if not reference.get("passed"):
            raise AssertionError("reference did not pass")
        prepared = next(row["result"] for row in reference["exporter"]["requests"]
                        if row["kind"] == "prepare")
        if (prepared["file_sha256"] != evidence["checkpoint_sha256"]
                or prepared["metadata"]["encoding"] != args.encoding):
            raise AssertionError("recovery checkpoint mismatch")
        original = copy.deepcopy(next(iter(requests(args.model, "f16", [742], False, "fact"))))
        body = copy.deepcopy(original)
        body.pop("context_management")
        body.pop("cache_dtype")
        body = normalize_request_body(args, body)
        with server(args, "worker", evidence) as worker:
            baseline = generate(args, original, worker)
            saved = resume(args, body, worker, prepared, "before_fault")
            if canonical_choices(baseline["choices"]) != canonical_choices(saved["choices"]):
                raise AssertionError("synthetic recovery fixture differs before fault")
            log = Path(worker["log_path"])
            offset = log.stat().st_size
            connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=args.timeout)
            try:
                connection.request("POST", "/ai/v1/generate/context/checkpoint",
                                   json.dumps(body).encode(), prepare_headers(args))
                response = connection.getresponse()
                raw = response.read(65536)
                evidence["failure_response_observed"] = dict(status=response.status, body_prefix_hex=raw[:64].hex(), bytes_read=len(raw))
                if len(raw) == 65536 or not 400 <= response.status <= 599:
                    raise AssertionError("fault did not fail bounded checkpoint publication")
                payload = json.loads(raw)
                if not payload.get("error"):
                    raise AssertionError("fault returned no structured request error")
                evidence["failure_response"] = dict(status=response.status, body=payload)
            finally:
                connection.close()
            observe_guard(args, evidence, 400, 5)
            evidence["fault_evidence"] = read_failure_evidence(log, offset, args.fault)
            recovered = resume(args, body, worker, prepared, "after_fault")
            ordinary = generate(args, original, worker)
            if (canonical_choices(recovered["choices"]) != canonical_choices(saved["choices"])
                    or recovered["usage"] != saved["usage"]
                    or canonical_choices(ordinary["choices"]) != canonical_choices(baseline["choices"])
                    or ordinary["usage"] != baseline["usage"]):
                raise AssertionError("same-worker checkpoint or ordinary recovery changed")
        validate_metal_memory(worker["memory"])
        if (not worker["graceful_stop"] or worker["returncode"] != 0
                or worker["remaining_group_members"] or worker["cleanup_errors"]):
            raise AssertionError("fault worker did not drain cleanly")
        if git_provenance(REPO) != evidence["source"]:
            raise AssertionError("source changed")
        for path, key in [(args.server, "binary_sha256"), (args.reference, "reference_sha256"),
                          (args.checkpoint, "checkpoint_sha256"), (Path(__file__), "runner_sha256")]:
            if sha256_file(path) != evidence[key]:
                raise AssertionError("fault producer or recovery artifact changed")
        evidence["passed"] = True
    except Exception as error:
        evidence["error"] = f"{type(error).__name__}: {error}"
    finally:
        if previous is None:
            os.environ.pop("ANTFLY_METAL_CHECKPOINT_NATIVE_FAULT", None)
        else:
            os.environ["ANTFLY_METAL_CHECKPOINT_NATIVE_FAULT"] = previous
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(dict(passed=evidence["passed"], error=evidence.get("error"))), flush=True)
    return 0 if evidence["passed"] else 1

REPO = Path(__file__).resolve().parents[5]

CYCLES_FILTER = "context Metal checkpoint successive installed owner model qualification"

def validate_source_build(repo, build):
    head, source = source_snapshot(repo)
    models.require(head == build["head"] and source == build["source"],
                     "cycle executable does not cover current source")
    return git_provenance(repo)

def validate_input(value):
    models.require(type(value) is dict and value.get("schema") ==
                     "antfly.metal-checkpoint-cycles.input.v1" and
                     value.get("model") in {
                         "google/gemma-4-E2B-it-qat-q4_0-gguf",
                         "google/gemma-4-E4B-it-qat-q4_0-gguf"}, "invalid cycle input")
    request = value.get("request")
    models.require(type(request) is dict and request.get("encoding") in
                     {"f16", "fp8", "rotated4"}, "invalid cycle encoding")
    bounds = {"cycles": (1, 20), "append_tokens": (1, 2048), "answer_tokens": (1, 32)}
    for key, (minimum, maximum) in bounds.items():
        models.require(type(request.get(key)) is int and minimum <= request[key] <= maximum,
                         "invalid cycle " + key)
    initial = value.get("expected_prompt_tokens")
    models.require(type(initial) is int and initial >= 2 and
                     initial + request["cycles"] *
                     (request["append_tokens"] + request["answer_tokens"]) <= 32768 - 512,
                     "cycle logical capacity exceeds query-tail reserve")
    models.require(type(value.get("messages")) is list and value["messages"],
                     "cycle messages are missing")

def select_artifact(artifacts, path):
    models.require(isinstance(path, str) and path.endswith(".gguf"),
                     "cycle model artifact must be an explicit GGUF path")
    selected = [row for row in artifacts if row["path"] == path]
    models.require(len(selected) == 1, "main GGUF artifact must match exactly one manifest entry")
    return selected[0]

def cycles_collect(args):
    repo = args.root.resolve(strict=True)
    build = builds.validate(args.build_receipt)
    source = validate_source_build(repo, build)
    input_path = args.input.resolve(strict=True)
    data = json.loads(input_path.read_bytes())
    validate_input(data)
    model_dir = args.model_directory.resolve(strict=True)
    models_root = model_dir.parent.parent
    models.checked_model_registry(models_root, model_dir, data["model"])
    identity_path = args.identity_receipt.resolve(strict=True)
    metadata = json.loads(identity_path.read_bytes())
    refs = {}
    for size in ("8k", "32k"):
        path = identity_path.parent / metadata["requests"]["request_" + size]["file"]
        refs[size] = {"path": str(path.resolve(strict=True)), "sha256": models.sha(path)}
    identity = models.identity_receipt_contract(identity_path, data["model"], refs)
    manifest = model_dir / ".antfly-download-complete.json"
    artifacts = json.loads(manifest.read_bytes())["artifacts"]
    artifact = select_artifact(artifacts, args.artifact)
    models.checked_model_artifact(model_dir, artifact["path"], artifact["sha256"])
    output = args.output.absolute()
    models.require(not output.is_symlink(), "symlinked cycle output")
    output.mkdir(parents=True, exist_ok=False)
    limits = {"timeout_seconds": 7200, "max_combined_rss_mib": 8192,
              "min_free_percent": 15, "max_swap_growth_mib": 0,
              "min_disk_free_mib": 4096, "load_preflight_free_percent": 76}
    spec = {
        "schema": "antfly.metal-native-cycles.spec.v1", "diagnostic_only": True,
        "production_ready": False, "source": source,
        "build_receipt": {"path": str(build["receipt_path"]), "sha256": build["receipt_sha256"]},
        "source_archive_sha256": build["source_archive_sha256"],
        "binary": {"path": str(build["binary_path"]), "sha256": build["binary_sha256"]},
        "input": {"path": str(input_path), "sha256": models.sha(input_path)},
        "runner_sha256": models.sha(__file__),
        "model_directory": str(model_dir), "artifact": artifact,
        "model_manifest_sha256": models.sha(manifest),
        "identity_receipt": {"path": str(identity_path), "sha256": models.sha(identity_path)},
        "performance_requests": refs, "loaded_identity": identity,
        "request": data["request"], "limits": limits,
    }
    models.write_new(output / "spec.json", spec)
    models.headroom(output, "native-cycles")
    saved = os.environ.copy()
    os.environ.clear()
    os.environ.update({key: value for key, value in saved.items()
                       if not key.startswith(models.RUNTIME_PREFIXES)})
    os.environ.update({
        "ANTFLY_METAL_CHECKPOINT_MODELS_ROOT": str(models_root),
        "ANTFLY_METAL_CHECKPOINT_MODEL": data["model"],
        "ANTFLY_METAL_CHECKPOINT_CYCLE_INPUT": str(input_path),
        "ANTFLY_METAL_CHECKPOINT_CYCLE_OUTPUT": str(output / "native-report.json"),
    })
    try:
        guard = resources.run_guarded_combined(
            [str(build["binary_path"]), "--test-filter", CYCLES_FILTER], output / "resource.json",
            timeout=limits["timeout_seconds"], max_rss_mib=limits["max_combined_rss_mib"],
            min_free_percent=limits["min_free_percent"], max_swap_growth_mib=0,
            min_disk_free_mib=limits["min_disk_free_mib"])
    finally:
        os.environ.clear()
        os.environ.update(saved)
    models.require(guard.get("pass") is True and guard.get("returncode") == 0 and
                     guard.get("swapout_growth_mib") == 0 and
                     guard.get("remaining_child_group_live_processes") == 0,
                     "native cycle resource/execution failure")
    models.require(validate_source_build(repo, build) == source,
                     "source changed during native cycles")
    for ref in (spec["input"], spec["binary"], spec["build_receipt"],
                spec["identity_receipt"], *spec["performance_requests"].values()):
        models.checked_ref(ref, "cycle bound input")
    models.require(models.sha(manifest) == spec["model_manifest_sha256"] and
                     models.sha(__file__) == spec["runner_sha256"], "cycle producer inputs changed")
    models.checked_model_artifact(model_dir, artifact["path"], artifact["sha256"])
    value = json.loads((output / "native-report.json").read_bytes())
    summary = validate_report(value, data, input_sha256=spec["input"]["sha256"],
                              model_directory=model_dir, loaded_identity=identity)
    report = {
        "schema": "antfly.metal-native-cycles.seal.v1", "passed": True,
        "diagnostic_only": True, "production_ready": False,
        "twenty_cycle_gate_completed": summary["completed_cycles"] == 20,
        "summary": summary,
        "spec_sha256": models.sha(output / "spec.json"),
        "resource_sha256": models.sha(output / "resource.json"),
        "native_report_sha256": models.sha(output / "native-report.json"),
        "stdout_sha256": models.sha(output / "resource.stdout"),
        "stderr_sha256": models.sha(output / "resource.stderr"),
    }
    models.write_new(output / "seal.json", report)
    print(json.dumps(report, indent=2))

def cycles_parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=REPO)
    result.add_argument("--build-receipt", type=Path, required=True)
    result.add_argument("--input", type=Path, required=True)
    result.add_argument("--model-directory", type=Path, required=True)
    result.add_argument("--artifact", required=True)
    result.add_argument("--identity-receipt", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result

def freeze(repo, root, specification_ref, output):
    root, repo = Path(root).resolve(strict=True), Path(repo).resolve(strict=True)
    specification = contract.json_ref(root, specification_ref, 'campaign preparation specification')
    fields = {'schema', 'build', 'configuration', 'configuration_audit',
              'input_index', 'models', 'performance'}
    contract.require(specification.get('schema') == 'antfly.native-whole-head.freeze-input.v2' and
                     set(specification) in (fields, fields | {'qualification_encodings'}),
                     'invalid freeze preparation')
    encodings = contract.qualification_encodings(specification)
    output = Path(output).absolute().resolve(strict=False)
    contract.require(output.is_relative_to(root) and not output.exists() and not output.is_symlink(),
                     'unsafe or existing campaign freeze')
    build_path = contract.regular(root, specification['build']['path'], 'full build')
    contract.read_ref(root, specification['build'], 'full build')
    build = builds.validate(build_path)
    configuration = contract.json_ref(root, specification['configuration'], 'configuration probe')
    identity = configuration['configuration']
    implementation = identity['implementation_sha256']
    contract.require(type(implementation) is list and len(implementation) == 32 and
                     all(type(byte) is int and 0 <= byte <= 255 for byte in implementation) and any(implementation),
                     'missing compiled implementation identity')
    index = contract.json_ref(root, specification['input_index'], 'quality preparation index')
    corpus = contract.digest(index.get('corpus_fingerprint_sha256'), 'corpus')
    implementation_hex = bytes(implementation).hex()
    frozen = {'schema': 'antfly.native-whole-head.campaign-freeze.v2', 'runtime_frozen': True,
              'release_authority': False, 'source': git_provenance(repo),
              'binary': {'path': str(build['binary_path']), 'sha256': build['binary_sha256']},
              'implementation_sha256': implementation_hex, 'corpus_fingerprint_sha256': corpus,
              'runtime_environment': {FLAGS[field]: '1' if identity['execution'][field] else '0'
                                      for field in identity['execution']},
              'statistical_seed': hashlib.sha256(('antfly.native-whole-head.statistics.v2\n' + corpus +
                                                   '\n' + implementation_hex + '\n').encode()).hexdigest(),
              'preparation_specification': specification_ref,
              **{key: specification[key] for key in ('build', 'configuration', 'configuration_audit',
                                                      'input_index', 'models', 'performance')}}
    if 'qualification_encodings' in specification:
        frozen['qualification_encodings'] = list(encodings)
        frozen['statistical_seed'] = hashlib.sha256(
            (frozen['statistical_seed'] + '\nqualification_encodings=' + ','.join(encodings) + '\n').encode()).hexdigest()
    with (root / 'native-whole-head-active.lock').open('a+') as active:
        fcntl.flock(active.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.parent.mkdir(parents=True, exist_ok=True)
        pending = output.with_name(output.name + '.pending')
        write_new(pending, frozen)
        pending_ref = reference(root, pending)
        # This performs full source/build/configuration/corpus/performance
        # reconciliation without decoding scoring answers or loading a model.
        validate_freeze(repo, root, pending_ref)
        for model in contract.MODELS:
            batch = next(batch for batch in index['batches'] if batch['model'] == model)
            manifest = contract.json_ref(root, batch['input'], 'native model binding')
            model_artifact(root, frozen, batch, manifest)
        # link() publishes the exact validated bytes exclusively, including if
        # another actor created the destination during validation.
        os.link(pending, output)
        pending.unlink()
        return reference(root, output)

def freeze_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--specification', type=Path, required=True)
    parser.add_argument('--specification-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = freeze(args.root, args.campaign_root,
                    {'path': str(args.specification), 'sha256': args.specification_sha256}, args.output)
    print(result['path'] + ' ' + result['sha256'])

def cycles_main(argv=None):
    return cycles_collect(cycles_parser().parse_args(argv))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'smoke', 'quality', 'performance', 'cycles', 'fault', 'soak'))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    selected = parser.parse_args(argv)
    local = {'freeze': freeze_main, 'quality': quality_main, 'cycles': cycles_main,
             'fault': fault_main, 'soak': soak_main}
    if selected.phase in local:
        return local[selected.phase](selected.arguments)
    if selected.phase == 'smoke':
        import qualify_native_checkpoint_http as phase
    else:
        import qualify_metal_native_checkpoint_performance as phase
    saved = sys.argv
    try:
        sys.argv = [saved[0] + ' ' + selected.phase, *selected.arguments]
        return phase.main()
    finally:
        sys.argv = saved


if __name__ == '__main__':
    raise SystemExit(main())
