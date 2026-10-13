#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compare the deployed CUDA training primitives with CPU PyTorch autograd.

Load the actual SM89 artifact rather than compiling alternate test kernels.
This qualifies FP32 loss/adjoint primitives; full encoder qualification uses
the separate BF16 optimizer/resume gate. No timing claims are collected here.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import json
from pathlib import Path

from finetune_contract import file_sha256


class Driver:
    def __init__(self, artifact):
        self.api = ct.CDLL('libcuda.so.1')
        self.allocations = []
        self.functions = {}
        signatures = {
            'cuInit': [ct.c_uint], 'cuDeviceGet': [ct.POINTER(ct.c_int), ct.c_int],
            'cuDriverGetVersion': [ct.POINTER(ct.c_int)],
            'cuDeviceComputeCapability': [ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.c_int],
            'cuCtxCreate_v2': [ct.POINTER(ct.c_void_p), ct.c_uint, ct.c_int],
            'cuCtxDestroy_v2': [ct.c_void_p], 'cuCtxSynchronize': [],
            'cuModuleLoad': [ct.POINTER(ct.c_void_p), ct.c_char_p],
            'cuModuleUnload': [ct.c_void_p],
            'cuModuleGetFunction': [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p],
            'cuMemAlloc_v2': [ct.POINTER(ct.c_uint64), ct.c_size_t],
            'cuMemFree_v2': [ct.c_uint64],
            'cuMemcpyHtoD_v2': [ct.c_uint64, ct.c_void_p, ct.c_size_t],
            'cuMemcpyDtoH_v2': [ct.c_void_p, ct.c_uint64, ct.c_size_t],
            'cuLaunchKernel': [ct.c_void_p, *([ct.c_uint] * 7), ct.c_void_p,
                               ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_void_p)],
        }
        for name, types in signatures.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = types, ct.c_int
        self.check('cuInit', 0)
        device, major, minor, version = ct.c_int(), ct.c_int(), ct.c_int(), ct.c_int()
        self.check('cuDeviceGet', ct.byref(device), 0)
        self.check('cuDeviceComputeCapability', ct.byref(major), ct.byref(minor), device)
        self.check('cuDriverGetVersion', ct.byref(version))
        if (major.value, minor.value) != (8, 9):
            raise ValueError('This qualification artifact requires SM89')
        self.hardware = {'compute_capability': [major.value, minor.value], 'driver_version': version.value}
        self.context, self.module = ct.c_void_p(), ct.c_void_p()
        self.check('cuCtxCreate_v2', ct.byref(self.context), 0, device)
        try:
            self.check('cuModuleLoad', ct.byref(self.module), str(artifact.resolve()).encode())
        except Exception:
            self.check('cuCtxDestroy_v2', self.context)
            raise

    def check(self, name, *args):
        status = getattr(self.api, name)(*args)
        if status != 0:
            raise RuntimeError(f'{name} failed with CUDA status {status}')

    def upload(self, values):
        import numpy as np
        array = np.ascontiguousarray(values)
        pointer = ct.c_uint64()
        self.check('cuMemAlloc_v2', ct.byref(pointer), array.nbytes)
        self.allocations.append(pointer)
        self.check('cuMemcpyHtoD_v2', pointer, ct.c_void_p(array.ctypes.data), array.nbytes)
        return pointer

    def download(self, pointer, shape):
        import numpy as np
        values = np.empty(shape, dtype=np.float32)
        self.check('cuMemcpyDtoH_v2', ct.c_void_p(values.ctypes.data), pointer, values.nbytes)
        return values

    def launch(self, name, args, grid, threads):
        if name not in self.functions:
            function = ct.c_void_p()
            self.check('cuModuleGetFunction', ct.byref(function), self.module, name.encode())
            self.functions[name] = function
        parameters = (ct.c_void_p * len(args))(*(ct.cast(ct.byref(value), ct.c_void_p) for value in args))
        self.check('cuLaunchKernel', self.functions[name], *grid, threads, 1, 1, 0, None, parameters, None)
        self.check('cuCtxSynchronize')

    def reset(self):
        self.check('cuCtxSynchronize')
        for pointer in reversed(self.allocations):
            self.check('cuMemFree_v2', pointer)
        self.allocations.clear()

    def close(self):
        self.reset()
        self.check('cuModuleUnload', self.module)
        self.check('cuCtxDestroy_v2', self.context)


def compare(actual, expected):
    import numpy as np
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Numeric differential requires matching finite arrays')
    error = np.abs(actual - expected)
    limit = 1e-5 + 1e-5 * np.abs(expected)
    return {'pass': bool((error <= limit).all()), 'elements': actual.size,
            'maximum_absolute_error': float(error.max()), 'maximum_scaled_error': float((error/limit).max())}


def ce_matrix(driver, rows, generator):
    import numpy as np
    import torch
    for queries, candidates in ((1, 1), (3, 5), (7, 17), (16, 32), (16, 96),
                                (32, 128), (32, 192), (64, 320)):
        for loss_scale in (1., .125):
            for target_only in (False, True):
                logits = torch.randn((queries, candidates), generator=generator) * .1
                targets = torch.arange(queries) % candidates
                valid = torch.rand((queries, candidates), generator=generator) > .2
                if target_only:
                    valid.zero_()
                valid[torch.arange(queries), targets] = True
                reference = logits.clone().requires_grad_()
                losses = torch.nn.functional.cross_entropy(
                    (reference*20).masked_fill(~valid, torch.finfo(torch.float32).min), targets, reduction='none') * .25
                (losses.mean()*loss_scale).backward()
                coefficients = driver.upload(np.zeros(logits.shape, dtype=np.float32))
                device_losses = driver.upload(np.zeros(queries, dtype=np.float32))
                args = [coefficients, device_losses, driver.upload(logits.numpy()),
                        driver.upload(valid.numpy().astype(np.uint8)), driver.upload(targets.numpy().astype(np.uint32)),
                        ct.c_uint(queries), ct.c_uint(candidates), ct.c_float(20), ct.c_float(.25), ct.c_float(loss_scale)]
                driver.launch('termite_embedding_gemma2_training_contrastive_ce_f32', args, (queries, 1, 1), 32)
                for field, actual, expected in (
                    ('loss', driver.download(device_losses, (queries,)), losses.detach().numpy()),
                    ('gradient', driver.download(coefficients, logits.shape), reference.grad.numpy())):
                    rows.append({'primitive': 'cross_entropy', 'queries': queries, 'candidates': candidates,
                        'loss_scale': loss_scale, 'target_only': target_only, 'field': field, **compare(actual, expected)})
                driver.reset()
    # Invalid targets must fail closed, not silently produce a usable loss.
    for kind in ('out_of_range', 'masked_target'):
        queries, candidates = 3, 5
        targets = np.array([0, 1, 2], dtype=np.uint32)
        valid = np.ones((queries, candidates), dtype=np.uint8)
        if kind == 'out_of_range':
            targets[1] = candidates
        else:
            valid[1, 1] = 0
        coeff = driver.upload(np.ones((queries, candidates), dtype=np.float32))
        losses = driver.upload(np.zeros(queries, dtype=np.float32))
        driver.launch('termite_embedding_gemma2_training_contrastive_ce_f32',
            [coeff, losses, driver.upload(np.zeros((queries, candidates), dtype=np.float32)),
             driver.upload(valid), driver.upload(targets), ct.c_uint(queries), ct.c_uint(candidates),
             ct.c_float(20), ct.c_float(.25), ct.c_float(1)], (queries, 1, 1), 32)
        actual_loss, actual_grad = driver.download(losses, (queries,)), driver.download(coeff, (queries, candidates))
        rows.append({'primitive': 'invalid_cross_entropy_target', 'case': kind,
                     'pass': bool(np.isposinf(actual_loss[1]) and (actual_grad[1] == 0).all())})
        driver.reset()


def mrl_matrix(driver, rows, generator):
    import numpy as np
    import torch
    for count in (1, 3, 16, 64, 384):
        for width in (128, 256, 512, 768):
            values = torch.randn((count, 768), generator=generator)
            gradient = torch.randn((count, width), generator=generator)
            reference = values.clone().requires_grad_()
            normalized = torch.nn.functional.normalize(reference[:, :width], dim=1, eps=1e-12)
            (normalized*gradient).sum().backward()
            inputs = driver.upload(values.numpy())
            output = driver.upload(np.zeros((count, width), dtype=np.float32))
            adjoint = driver.upload(np.zeros((count, 768), dtype=np.float32))
            driver.launch('termite_embedding_gemma2_mrl_normalize_f32',
                [output, inputs, ct.c_uint(count), ct.c_uint(width)], (count, 1, 1), 128)
            driver.launch('termite_embedding_gemma2_mrl_normalize_vjp_f32',
                [adjoint, driver.upload(gradient.numpy()), inputs, ct.c_uint(count), ct.c_uint(width), ct.c_float(1)],
                (count, 1, 1), 128)
            for field, actual, expected in (
                ('forward', driver.download(output, (count, width)), normalized.detach().numpy()),
                ('gradient', driver.download(adjoint, (count, 768)), reference.grad.numpy())):
                rows.append({'primitive': 'mrl_normalization', 'rows': count, 'width': width, 'field': field,
                             **compare(actual, expected)})
            driver.reset()


def softmax_matrix(driver, rows, generator):
    import numpy as np
    import torch
    for sequence in (17, 20, 22, 32, 64, 128, 512):
        for radius in (0, 3):
            for masking in ('all_valid', 'partial', 'all_masked'):
                shape = (2, 2, sequence, sequence)
                scores = torch.randn(shape, generator=generator)
                gradient = torch.randn(shape, generator=generator)
                mask = torch.ones((2, sequence), dtype=torch.int64)
                if masking == 'partial':
                    mask[0, sequence//2:] = 0
                elif masking == 'all_masked':
                    mask.zero_()
                allowed = (mask[:, None, :, None] != 0) & (mask[:, None, None, :] != 0)
                if radius:
                    positions = torch.arange(sequence)
                    allowed = allowed & ((positions[:, None]-positions[None, :]).abs() <= radius)
                allowed = allowed.expand(shape)
                reference = scores.clone().requires_grad_()
                probabilities = torch.softmax(reference.masked_fill(~allowed, torch.finfo(torch.float32).min), dim=-1)
                probabilities = torch.where(allowed.any(dim=-1, keepdim=True), probabilities, 0.)
                (probabilities*gradient).sum().backward()
                output = driver.upload(np.zeros(shape, dtype=np.float32))
                adjoint = driver.upload(np.zeros(shape, dtype=np.float32))
                driver.launch('termite_embedding_gemma2_training_softmax_f32',
                    [output, driver.upload(scores.numpy()), driver.upload(mask.numpy()), ct.c_uint(2),
                     ct.c_uint(sequence), ct.c_uint(2), ct.c_uint(radius)], (sequence, 2, 2), 32)
                driver.launch('termite_embedding_gemma2_training_softmax_vjp_f32',
                    [adjoint, driver.upload(gradient.numpy()), output, ct.c_uint(2), ct.c_uint(sequence), ct.c_uint(2)],
                    (sequence, 2, 2), 32)
                for field, actual, expected in (
                    ('forward', driver.download(output, shape), probabilities.detach().numpy()),
                    ('gradient', driver.download(adjoint, shape), reference.grad.numpy())):
                    rows.append({'primitive': 'attention_softmax', 'sequence': sequence, 'local_radius': radius,
                        'masking': masking, 'field': field, **compare(actual, expected)})
                driver.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, default=Path(__file__).resolve().parents[2]/
                        'src/ops/cuda/artifacts/inference_cuda_kernels_sm89.cubin')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    artifact_sha = file_sha256(args.artifact)
    rows, driver = [], Driver(args.artifact)
    try:
        generator = torch.Generator().manual_seed(20261013)
        ce_matrix(driver, rows, generator)
        mrl_matrix(driver, rows, generator)
        softmax_matrix(driver, rows, generator)
    finally:
        driver.close()
    if file_sha256(args.artifact) != artifact_sha:
        raise ValueError('CUDA artifact changed during qualification')
    report = {'schema': 'antfly.embedding_gemma2.cuda_training_numeric_differential.v1',
              'scope': 'current-artifact FP32 training loss/adjoint primitives; full encoder uses BF16 gate',
              'production_qualified': False, 'pass': all(row['pass'] for row in rows), 'rows': rows,
              'artifact': {'path': str(args.artifact.resolve()), 'sha256': artifact_sha},
              'collector_sha256': file_sha256(Path(__file__)), 'hardware': driver.hardware,
              'reference': {'torch': torch.__version__, 'device': 'cpu', 'precision': 'fp32', 'seed': 20261013},
              'tolerances': {'atol': 1e-5, 'rtol': 1e-5}}
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({'pass': report['pass'], 'checks': len(rows), 'failed_checks': sum(not r['pass'] for r in rows)}))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
