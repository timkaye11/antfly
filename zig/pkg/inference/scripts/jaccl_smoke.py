#!/usr/bin/env python3
"""Two-rank JACCL RDMA smoke test for the Antfly C bridge.

Run through launch_jaccl_finetune.py so both ranks receive the same topology,
coordinator, and library settings. This loads no model or training data.
"""

import ctypes
import json
import os
import sys


def main():
    rank = int(os.environ['ANTFLY_JACCL_RANK'])
    if rank not in (0, 1):
        raise ValueError('the smoke test requires ranks 0 and 1')
    bridge = ctypes.CDLL(os.environ['ANTFLY_JACCL_LIBRARY'])
    bridge.antfly_jaccl_open.argtypes = [ctypes.c_int, ctypes.c_char_p,
                                        ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    bridge.antfly_jaccl_open.restype = ctypes.c_int
    bridge.antfly_jaccl_rank.argtypes = [ctypes.c_void_p]
    bridge.antfly_jaccl_rank.restype = ctypes.c_int
    bridge.antfly_jaccl_size.argtypes = [ctypes.c_void_p]
    bridge.antfly_jaccl_size.restype = ctypes.c_int
    bridge.antfly_jaccl_all_sum_f32.argtypes = [ctypes.c_void_p,
                                                ctypes.POINTER(ctypes.c_float),
                                                ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
    bridge.antfly_jaccl_all_sum_f32.restype = ctypes.c_int
    bridge.antfly_jaccl_all_gather.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                               ctypes.c_void_p, ctypes.c_size_t]
    bridge.antfly_jaccl_all_gather.restype = ctypes.c_int
    bridge.antfly_jaccl_barrier.argtypes = [ctypes.c_void_p]
    bridge.antfly_jaccl_barrier.restype = ctypes.c_int
    bridge.antfly_jaccl_close.argtypes = [ctypes.c_void_p]
    bridge.antfly_jaccl_last_error.restype = ctypes.c_char_p

    def check(code, action):
        if code != 0:
            detail = bridge.antfly_jaccl_last_error().decode(errors='replace')
            raise RuntimeError(f'{action}: {detail}')

    handle = ctypes.c_void_p()
    check(bridge.antfly_jaccl_open(
        rank,
        os.environ['ANTFLY_JACCL_COORDINATOR'].encode(),
        os.environ['ANTFLY_JACCL_DEVICES_FILE'].encode(),
        ctypes.byref(handle)), 'open')
    try:
        if bridge.antfly_jaccl_rank(handle) != rank or bridge.antfly_jaccl_size(handle) != 2:
            raise RuntimeError('JACCL group rank or size mismatch')
        inputs = (ctypes.c_float * 3)(rank + 1, 2 * (rank + 1), 0.25)
        outputs = (ctypes.c_float * 3)()
        check(bridge.antfly_jaccl_all_sum_f32(handle, inputs, outputs, 3), 'all_sum')
        actual = list(outputs)
        if actual != [3.0, 6.0, 0.5]:
            raise RuntimeError(f'all_sum result mismatch: {actual}')
        for count in (1024, 1024 * 1024):
            large_input = (ctypes.c_float * count)(*([rank + 1] * count))
            large_output = (ctypes.c_float * count)()
            check(bridge.antfly_jaccl_all_sum_f32(handle, large_input, large_output, count),
                  f'all_sum {count} elements')
            for index in (0, count // 2, count - 1):
                if large_output[index] != 3.0:
                    raise RuntimeError(f'all_sum {count} element {index}: {large_output[index]}')
            if sum(large_output) != 3 * count:
                raise RuntimeError(f'all_sum {count} checksum mismatch')
        token = (ctypes.c_uint8 * 1)(rank)
        gathered = (ctypes.c_uint8 * 2)()
        check(bridge.antfly_jaccl_all_gather(handle, token, gathered, 1), 'all_gather')
        if list(gathered) != [0, 1]:
            raise RuntimeError(f'all_gather rank order mismatch: {list(gathered)}')
        check(bridge.antfly_jaccl_barrier(handle), 'barrier')
        print(json.dumps({'event': 'jaccl_smoke', 'rank': rank, 'size': 2,
                          'all_sum': actual, 'all_sum_bucket_elements': 1024 * 1024,
                          'all_gather': list(gathered), 'status': 'pass'}),
              flush=True)
    finally:
        bridge.antfly_jaccl_close(handle)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'jaccl_smoke: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
