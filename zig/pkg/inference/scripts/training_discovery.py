#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc. SPDX-License-Identifier: Apache-2.0
"""Native macOS Bonjour discovery; advertisements carry no model inventory."""
import ctypes as C
from pathlib import Path
import select
import socket
import sys
import threading
import time
import uuid

SERVICE = b'_antfly._tcp'
MAX_PEERS = 128


def parse_txt(raw):
    fields = {}
    while raw:
        size, raw = raw[0], raw[1:]
        if size > len(raw):
            return {}
        part, raw = raw[:size], raw[size:]
        key, _, value = part.decode(errors='replace').partition('=')
        fields[key] = value
    return fields


class Discovery:
    def __init__(self, state_dir, port=22):
        self.peers = {}
        self.error = None
        self.closed = threading.Event()
        self.lock = threading.Lock()
        self.refs = []
        self.callbacks = []
        self.port = port
        directory = Path(state_dir)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        identity = directory / 'discovery-id'
        try:
            with identity.open('x') as target:
                target.write(str(uuid.uuid4()))
        except FileExistsError:
            pass
        self.node_id = identity.read_text().strip()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.lock:
            # One host can be visible on Ethernet, Wi-Fi and Thunderbolt.
            return list({peer['id']: dict(peer) for peer in self.peers.values()}.values())

    def close(self):
        self.closed.set()
        self.thread.join(timeout=3)

    def run(self):
        if sys.platform != 'darwin':
            self.error = 'Bonjour discovery is available on macOS; configure an SSH peer manually.'
            return
        try:
            lib = C.CDLL('/usr/lib/libSystem.B.dylib')
            lib.DNSServiceRefSockFD.argtypes = [C.c_void_p]
            lib.DNSServiceProcessResult.argtypes = [C.c_void_p]
            lib.DNSServiceRefDeallocate.argtypes = [C.c_void_p]
            lib.DNSServiceRefDeallocate.restype = None
            callback_type = C.CFUNCTYPE(None, C.c_void_p, C.c_uint32, C.c_uint32, C.c_int32,
                                       C.c_char_p, C.c_char_p, C.c_char_p, C.c_void_p)
            resolve_type = C.CFUNCTYPE(None, C.c_void_p, C.c_uint32, C.c_uint32, C.c_int32,
                                      C.c_char_p, C.c_char_p, C.c_uint16, C.c_uint16, C.c_void_p, C.c_void_p)
            register_type = C.CFUNCTYPE(None, C.c_void_p, C.c_uint32, C.c_int32,
                                       C.c_char_p, C.c_char_p, C.c_char_p, C.c_void_p)
            lib.DNSServiceRegister.argtypes = [C.POINTER(C.c_void_p), C.c_uint32, C.c_uint32,
                C.c_char_p, C.c_char_p, C.c_char_p, C.c_char_p, C.c_uint16, C.c_uint16,
                C.c_void_p, register_type, C.c_void_p]
            lib.DNSServiceBrowse.argtypes = [C.POINTER(C.c_void_p), C.c_uint32, C.c_uint32,
                C.c_char_p, C.c_char_p, callback_type, C.c_void_p]
            lib.DNSServiceResolve.argtypes = [C.POINTER(C.c_void_p), C.c_uint32, C.c_uint32,
                C.c_char_p, C.c_char_p, C.c_char_p, resolve_type, C.c_void_p]
            pending = {}
            retired = []

            def resolved(ref, flags, interface, error, fullname, host, port, length, raw, context):
                if error:
                    return
                fields = parse_txt(C.string_at(raw, length))
                node_id = fields.get('id')
                if fields.get('v') != '1' or fields.get('control') != 'ssh' or not node_id or len(node_id) > 64 or node_id == self.node_id:
                    return
                key = next((key for key, value in pending.items() if value.value == ref), None)
                if key is None:
                    return
                with self.lock:
                    self.peers[key] = {'id': node_id, 'name': key[1].decode(errors='replace'),
                        'hostname': host.decode().rstrip('.'), 'ssh_port': socket.ntohs(port),
                        'status': 'discovered', 'seen_at': time.time()}

            resolved_callback = resolve_type(resolved)
            self.callbacks.append(resolved_callback)

            def browsed(ref, flags, interface, error, name, kind, domain, context):
                if error:
                    self.error = 'Bonjour browse failed: ' + str(error)
                    return
                key = (interface, name, domain)
                if not flags & 2:
                    with self.lock:
                        self.peers.pop(key, None)
                    resolution = pending.pop(key, None)
                    if resolution:
                        retired.append(resolution)
                    return
                if key in pending or len(pending) >= MAX_PEERS:
                    return
                resolution = C.c_void_p()
                status = lib.DNSServiceResolve(C.byref(resolution), 0, interface, name, kind, domain,
                                               resolved_callback, None)
                if status == 0:
                    pending[key] = resolution
                    self.refs.append(resolution)

            callback = callback_type(browsed)
            self.callbacks.append(callback)
            def registered(ref, flags, error, name, kind, domain, context):
                if error:
                    self.error = 'Bonjour registration failed: ' + str(error)
            registration_callback = register_type(registered)
            self.callbacks.append(registration_callback)
            advert = C.c_void_p()
            values = [b'v=1', b'control=ssh', ('id=' + self.node_id).encode()]
            txt = b''.join(bytes([len(value)]) + value for value in values)
            status = lib.DNSServiceRegister(C.byref(advert), 0, 0, socket.gethostname().encode(), SERVICE,
                                           None, None, socket.htons(self.port), len(txt), txt, registration_callback, None)
            if status:
                raise RuntimeError('Bonjour registration failed: ' + str(status))
            self.refs.append(advert)
            browse = C.c_void_p()
            status = lib.DNSServiceBrowse(C.byref(browse), 0, 0, SERVICE, None, callback, None)
            if status:
                raise RuntimeError('Bonjour browse failed: ' + str(status))
            self.refs.append(browse)
            while not self.closed.is_set():
                refs = list(self.refs)
                fds = {lib.DNSServiceRefSockFD(ref): ref for ref in refs}
                ready, _, _ = select.select(list(fds), [], [], 0.5)
                for fd in ready:
                    if lib.DNSServiceProcessResult(fds[fd]):
                        raise RuntimeError('Bonjour connection closed')
                for ref in retired:
                    self.refs.remove(ref)
                    lib.DNSServiceRefDeallocate(ref)
                retired.clear()
        except Exception as error:
            self.error = str(error)
        finally:
            if 'lib' in locals():
                for ref in self.refs:
                    lib.DNSServiceRefDeallocate(ref)


if __name__ == '__main__':
    discovery = Discovery(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 22)
    try:
        while sys.stdin.buffer.read(1):
            pass
    finally:
        discovery.close()
