// Copyright 2026 Antfly, Inc. SPDX-License-Identifier: Apache-2.0
import { createWasmAbi } from './wasm-abi.js';

let cpuModule;
async function configurationModule() {
  if (!cpuModule) {
    cpuModule = (async () => {
      const response = await fetch(new URL('../antfly-extraction-cpu.wasm', import.meta.url));
      if (!response.ok) throw new Error(`Configuration runtime unavailable (${response.status})`);
      return WebAssembly.compile(await response.arrayBuffer());
    })().catch(error => { cpuModule = undefined; throw error; });
  }
  return cpuModule;
}

// An isolated instance runs the loader's configuration parser without weights,
// a tokenizer, GPU admission, or any changes to the active inference session.
export async function configurationReason(files, config, encoderConfig, precision = 'fp32') {
  const wasm = (await WebAssembly.instantiate(await configurationModule(), { env: {} })).exports;
  if (wasm.extraction_abi_version?.() !== 2) throw new Error('Extraction WASM ABI mismatch; rebuild browser assets');
  const abi = createWasmAbi(wasm);
  const metadata = { config: JSON.stringify(config), encoder_config: JSON.stringify(encoderConfig), tokenizer_config: '{}', precision };
  if (['embedding_gemma2', 'embedding_gemma2_text'].includes(config.model_type)) {
    const { readJson } = await import('./extraction-bundle.js');
    metadata.processor_config = JSON.stringify(await readJson(files, 'processor_config.json'));
  }
  const bytes = new TextEncoder().encode(JSON.stringify(metadata));
  const ptr = abi.alloc(bytes.length);
  if (!ptr) throw new Error('WASM memory budget exceeded');
  try {
    abi.copyBytesIn(ptr, bytes);
    if (wasm.extraction_create(ptr, abi.size(bytes.length))) return undefined;
    return new TextDecoder().decode(abi.view(Uint8Array, wasm.extraction_error_ptr(), Number(wasm.extraction_error_len())));
  } finally {
    wasm.extraction_unload();
    abi.free(ptr, bytes.length);
  }
}
