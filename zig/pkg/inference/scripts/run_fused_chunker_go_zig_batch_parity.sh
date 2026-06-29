#!/usr/bin/env bash
# Copyright 2026 Antfly, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pkg_root="$(cd -- "$script_dir/.." && pwd)"
repo_root="$(cd -- "$pkg_root/../../.." && pwd)"

zig_bin="${ANTFLY_ZIG_BIN:-}"
if [[ -z "$zig_bin" ]]; then
  if command -v zig >/dev/null 2>&1; then
    zig_bin="$(command -v zig)"
  elif [[ -x "$HOME/.local/share/zigup/0.16.0-dev.3144+ac6fb0b59/files/zig" ]]; then
    zig_bin="$HOME/.local/share/zigup/0.16.0-dev.3144+ac6fb0b59/files/zig"
  else
    echo "missing Zig compiler; set ANTFLY_ZIG_BIN=/path/to/zig" >&2
    exit 1
  fi
fi

gopeft_dir="${ANTFLY_GOPEFT_DIR:-/Users/tim/Documents/af/gopeft}"
data_path="${ANTFLY_FUSED_CHUNKER_PARITY_DATA:-/Users/tim/Documents/af/gopeft/data/fused_train.jsonl}"
model_dir="${ANTFLY_FUSED_CHUNKER_MODEL_DIR:-$HOME/.cache/modernbert-embed-base}"
split="${ANTFLY_FUSED_CHUNKER_PARITY_SPLIT:-train}"
max_seq_len="${ANTFLY_FUSED_CHUNKER_MAX_SEQ_LEN:-384}"
max_chunks="${ANTFLY_FUSED_CHUNKER_MAX_CHUNKS:-32}"
batch_size="${ANTFLY_FUSED_CHUNKER_BATCH_SIZE:-8}"
offset="${ANTFLY_FUSED_CHUNKER_PARITY_OFFSET:-0}"
limit="${ANTFLY_FUSED_CHUNKER_PARITY_LIMIT:-8}"
out_dir="${ANTFLY_FUSED_CHUNKER_PARITY_OUT:-/tmp/fused_chunker_go_zig_parity}"

mkdir -p "$out_dir"
zig_json="$out_dir/zig_batch.json"
zig_stdout="$out_dir/zig_stdout.log"
zig_stderr="$out_dir/zig_stderr.log"
go_json="$out_dir/go_batch.json"
go_helper="$out_dir/go_fused_batch_parity.go"
compare_py="$out_dir/compare_fused_batch_parity.py"

if [[ ! -d "$gopeft_dir" ]]; then
  echo "missing gopeft checkout at $gopeft_dir" >&2
  exit 1
fi
if [[ ! -f "$data_path" ]]; then
  echo "missing fused data at $data_path" >&2
  exit 1
fi
if [[ ! -f "$model_dir/tokenizer.json" ]]; then
  echo "missing tokenizer at $model_dir/tokenizer.json" >&2
  exit 1
fi

cat > "$go_helper" <<'GOEOF'
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"gopeft/e2e/finetune"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

const fnvOffset uint64 = 14695981039346656037
const fnvPrime uint64 = 1099511628211

type Hashes struct {
	SampleIndices string `json:"sample_indices"`
	InputIDs      string `json:"input_ids"`
	AttentionMask string `json:"attention_mask"`
	Offsets       string `json:"offsets"`
	Labels        string `json:"labels"`
	Chunks        string `json:"chunks"`
}

type ChunkSpan struct {
	Idx   int `json:"idx"`
	Start int `json:"start"`
	End   int `json:"end"`
	Mask  int `json:"mask"`
}

type SourceChunkBoundary struct {
	Idx        int `json:"idx"`
	StartChar  int `json:"start_char"`
	EndChar    int `json:"end_char"`
	StartToken int `json:"start_token"`
	EndToken   int `json:"end_token"`
}

type SourceTokenMapping struct {
	Idx           int `json:"idx"`
	StartChar     int `json:"start_char"`
	EndChar       int `json:"end_char"`
	RawStart      int `json:"raw_start"`
	RawEnd        int `json:"raw_end"`
	ResolvedStart int `json:"resolved_start"`
	ResolvedEnd   int `json:"resolved_end"`
	Valid         int `json:"valid"`
	PrevOff       [2]int `json:"prev_off"`
	EndOff        [2]int `json:"end_off"`
}

type DebugSourceTokenMappings struct {
	SampleIndex int                  `json:"sample_index"`
	Mappings    []SourceTokenMapping `json:"mappings"`
}

type BatchSample struct {
	SampleIndex           int                   `json:"sample_index"`
	ValidTokens           int                   `json:"valid_tokens"`
	InputIDs              []int32               `json:"input_ids"`
	AttentionMask         []int32               `json:"attention_mask"`
	BoundaryPositions     []int                 `json:"boundary_positions"`
	SourceChunkBoundaries []SourceChunkBoundary `json:"source_chunk_boundaries"`
	ChunkSpans            []ChunkSpan           `json:"chunk_spans"`
}

type FirstBatch struct {
	SampleIndices []int         `json:"sample_indices"`
	Samples       []BatchSample `json:"samples"`
}

type Output struct {
	Tool                            string      `json:"tool"`
	SchemaVersion                   int         `json:"schema_version"`
	Offset                          int         `json:"offset"`
	Samples                         int         `json:"samples"`
	Batches                         int         `json:"batches"`
	MaxSeqLen                       int         `json:"max_seq_len"`
	MaxChunks                       int         `json:"max_chunks"`
	ValidTokens                     uint64      `json:"valid_tokens"`
	BoundaryGoldTokens              uint64      `json:"boundary_gold_tokens"`
	GoldRate                        float64     `json:"gold_rate"`
	SamplesWithActiveBoundaryLabels uint64      `json:"samples_with_active_boundary_labels"`
	ContrastivePosSamples           int         `json:"contrastive_pos_samples"`
	BoundaryTargetSamples           int         `json:"boundary_target_samples"`
	BoundaryTargets                 int         `json:"boundary_targets"`
	Hashes                          Hashes      `json:"hashes"`
	FirstBatch                      *FirstBatch `json:"first_batch,omitempty"`
	DebugSourceTokenMappings        []DebugSourceTokenMappings `json:"debug_source_token_mappings,omitempty"`
}

func hashU64(h *uint64, v uint64) {
	for i := 0; i < 8; i++ {
		*h ^= v & 0xff
		*h *= fnvPrime
		v >>= 8
	}
}

func hashI32(h *uint64, v int32) {
	hashU64(h, uint64(uint32(v)))
}

func tensorI32(t *tensors.Tensor) []int32 {
	var out []int32
	t.ConstFlatData(func(flat any) {
		src := flat.([]int32)
		out = append([]int32(nil), src...)
	})
	return out
}

func tensorF32(t *tensors.Tensor) []float32 {
	var out []float32
	t.ConstFlatData(func(flat any) {
		src := flat.([]float32)
		out = append([]float32(nil), src...)
	})
	return out
}

func validCount(mask []int32) int {
	n := 0
	for _, v := range mask {
		if v == 0 {
			break
		}
		n++
	}
	return n
}

func boundaryPositions(labels []float32, mask []int32) []int {
	var out []int
	for i, label := range labels {
		if i >= len(mask) || mask[i] == 0 {
			break
		}
		if label > 0.5 {
			out = append(out, i)
		}
	}
	if out == nil {
		return []int{}
	}
	return out
}

func sourceTokenMappings(sample finetune.FusedSample, offsets [][2]int) []SourceTokenMapping {
	out := make([]SourceTokenMapping, 0, len(sample.ChunkBoundaries))
	for ci, cb := range sample.ChunkBoundaries {
		startToken := -1
		endToken := -1

		for t, offset := range offsets {
			if offset[0] == 0 && offset[1] == 0 && t > 0 {
				continue
			}
			if startToken == -1 && offset[0] <= cb.StartChar && cb.StartChar < offset[1] {
				startToken = t
			}
			if offset[0] < cb.EndChar && cb.EndChar <= offset[1] {
				endToken = t + 1
			}
			if offset[0] >= cb.EndChar {
				break
			}
		}

		resolvedStart := startToken
		resolvedEnd := endToken
		if cb.StartToken > 0 {
			resolvedStart = cb.StartToken
		}
		if cb.EndToken > 0 {
			resolvedEnd = cb.EndToken
		}
		valid := 0
		if resolvedStart >= 0 && resolvedEnd > resolvedStart {
			valid = 1
		}
		prevOff := [2]int{0, 0}
		if endToken > 0 && endToken-1 < len(offsets) {
			prevOff = offsets[endToken-1]
		}
		endOff := [2]int{0, 0}
		if endToken >= 0 && endToken < len(offsets) {
			endOff = offsets[endToken]
		}
		out = append(out, SourceTokenMapping{
			Idx:           ci,
			StartChar:     cb.StartChar,
			EndChar:       cb.EndChar,
			RawStart:      startToken,
			RawEnd:        endToken,
			ResolvedStart: resolvedStart,
			ResolvedEnd:   resolvedEnd,
			Valid:         valid,
			PrevOff:       prevOff,
			EndOff:        endOff,
		})
	}
	return out
}

func main() {
	dataPath := flag.String("data", "", "fused JSONL data")
	modelDir := flag.String("model-dir", "", "model dir containing tokenizer.json")
	split := flag.String("split", "train", "split")
	maxSeqLen := flag.Int("max-seq-len", 384, "max sequence length")
	maxChunks := flag.Int("max-chunks", 32, "max chunks")
	batchSize := flag.Int("batch-size", 8, "batch size")
	offset := flag.Int("offset", 0, "sample offset")
	limit := flag.Int("limit", 8, "sample limit")
	dumpBatch := flag.Bool("dump-batch", false, "dump first batch arrays")
	flag.Parse()

	if *dataPath == "" || *modelDir == "" {
		fmt.Fprintln(os.Stderr, "--data and --model-dir are required")
		os.Exit(2)
	}

	samples, err := finetune.LoadFusedSamples(*dataPath, *split)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load samples: %v\n", err)
		os.Exit(1)
	}
	if *offset < 0 || *offset > len(samples) {
		fmt.Fprintf(os.Stderr, "invalid offset %d for %d samples\n", *offset, len(samples))
		os.Exit(1)
	}
	end := len(samples)
	if *limit > 0 && *offset+*limit < end {
		end = *offset + *limit
	}
	sourceOffset := *offset
	samples = samples[*offset:end]

	tokenizer, err := finetune.NewFusedHFTokenizer(filepath.Join(*modelDir, "tokenizer.json"), 50368)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load tokenizer: %v\n", err)
		os.Exit(1)
	}

	contrastivePosSamples := 0
	boundaryTargetSamples := 0
	boundaryTargets := 0
	for _, sample := range samples {
		if len(sample.PositivePairs) > 0 {
			contrastivePosSamples++
		}
		if len(sample.ChunkBoundaries) > 1 {
			boundaryTargetSamples++
			boundaryTargets += len(sample.ChunkBoundaries) - 1
		}
	}

	ds := finetune.NewFusedDataset(samples, tokenizer, *maxSeqLen, *maxChunks, *batchSize, false, 42)

	var valid, gold, batches, activeBoundarySamples uint64
	idsHash, maskHash, offsetsHash := fnvOffset, fnvOffset, fnvOffset
	labelsHash, chunksHash, sampleIndicesHash := fnvOffset, fnvOffset, fnvOffset

	var firstBatch *FirstBatch
	var debugSourceTokenMappings []DebugSourceTokenMappings
	for ds.HasNext() {
		batch := ds.NextBatch()
		if batch == nil {
			break
		}
		batches++
		inputIDs := tensorI32(batch.InputIDs)
		attentionMask := tensorI32(batch.AttentionMask)
		boundaryLabels := tensorF32(batch.BoundaryLabels)
		chunkStarts := tensorI32(batch.ChunkStarts)
		chunkEnds := tensorI32(batch.ChunkEnds)
		chunkMask := tensorF32(batch.ChunkMask)

		for _, idx := range batch.SampleIndices {
			hashU64(&sampleIndicesHash, uint64(sourceOffset+idx))
		}
		batchGoldBySample := make([]uint64, len(batch.SampleIndices))
		total := len(batch.SampleIndices) * *maxSeqLen
		for i := 0; i < total; i++ {
			hashI32(&idsHash, inputIDs[i])
			hashI32(&maskHash, attentionMask[i])
			if boundaryLabels[i] > 0.5 {
				hashU64(&labelsHash, 1)
			} else {
				hashU64(&labelsHash, 0)
			}
			if attentionMask[i] != 0 {
				valid++
				if boundaryLabels[i] > 0.5 {
					gold++
					batchGoldBySample[i / *maxSeqLen]++
				}
			}
		}
		for _, sampleGold := range batchGoldBySample {
			if sampleGold > 0 {
				activeBoundarySamples++
			}
		}
		for i := 0; i < len(batch.SampleIndices)**maxChunks; i++ {
			hashI32(&chunksHash, chunkStarts[i])
			hashI32(&chunksHash, chunkEnds[i])
			if chunkMask[i] > 0.5 {
				hashU64(&chunksHash, 1)
			} else {
				hashU64(&chunksHash, 0)
			}
		}

		if *dumpBatch && firstBatch == nil {
			absoluteSampleIndices := make([]int, len(batch.SampleIndices))
			for i, idx := range batch.SampleIndices {
				absoluteSampleIndices[i] = sourceOffset + idx
			}
			fb := &FirstBatch{
				SampleIndices: absoluteSampleIndices,
				Samples:       make([]BatchSample, 0, len(batch.SampleIndices)),
			}
			debugSourceTokenMappings = make([]DebugSourceTokenMappings, 0, len(batch.SampleIndices))
			for bi, sampleIndex := range batch.SampleIndices {
				seqStart := bi * *maxSeqLen
				seqEnd := seqStart + *maxSeqLen
				chunkStart := bi * *maxChunks
				chunkEnd := chunkStart + *maxChunks
				spans := make([]ChunkSpan, 0, *maxChunks)
				for ci := chunkStart; ci < chunkEnd; ci++ {
					m := 0
					if chunkMask[ci] > 0.5 {
						m = 1
					}
					spans = append(spans, ChunkSpan{
						Idx:   ci - chunkStart,
						Start: int(chunkStarts[ci]),
						End:   int(chunkEnds[ci]),
						Mask:  m,
					})
				}
				sourceBoundaries := make([]SourceChunkBoundary, 0, len(samples[sampleIndex].ChunkBoundaries))
				for ci, chunk := range samples[sampleIndex].ChunkBoundaries {
					sourceBoundaries = append(sourceBoundaries, SourceChunkBoundary{
						Idx:        ci,
						StartChar:  chunk.StartChar,
						EndChar:    chunk.EndChar,
						StartToken: chunk.StartToken,
						EndToken:   chunk.EndToken,
					})
				}
				fb.Samples = append(fb.Samples, BatchSample{
					SampleIndex:           sourceOffset + sampleIndex,
					ValidTokens:           validCount(attentionMask[seqStart:seqEnd]),
					InputIDs:              append([]int32(nil), inputIDs[seqStart:seqEnd]...),
					AttentionMask:         append([]int32(nil), attentionMask[seqStart:seqEnd]...),
					BoundaryPositions:     boundaryPositions(boundaryLabels[seqStart:seqEnd], attentionMask[seqStart:seqEnd]),
					SourceChunkBoundaries: sourceBoundaries,
					ChunkSpans:            spans,
				})
				_, _, offsets := tokenizer.Encode(samples[sampleIndex].Text, *maxSeqLen)
				debugSourceTokenMappings = append(debugSourceTokenMappings, DebugSourceTokenMappings{
					SampleIndex: sourceOffset + sampleIndex,
					Mappings:    sourceTokenMappings(samples[sampleIndex], offsets),
				})
			}
			firstBatch = fb
		}
	}

	for _, sample := range samples {
		_, _, offsets := tokenizer.Encode(sample.Text, *maxSeqLen)
		for _, off := range offsets {
			hashU64(&offsetsHash, uint64(off[0]))
			hashU64(&offsetsHash, uint64(off[1]))
		}
	}

	goldRate := 0.0
	if valid > 0 {
		goldRate = float64(gold) / float64(valid)
	}
	out := Output{
		Tool:                            "go_fused_tokenization_parity",
		SchemaVersion:                   2,
		Offset:                          sourceOffset,
		Samples:                         len(samples),
		Batches:                         int(batches),
		MaxSeqLen:                       *maxSeqLen,
		MaxChunks:                       *maxChunks,
		ValidTokens:                     valid,
		BoundaryGoldTokens:              gold,
		GoldRate:                        goldRate,
		SamplesWithActiveBoundaryLabels: activeBoundarySamples,
		ContrastivePosSamples:           contrastivePosSamples,
		BoundaryTargetSamples:           boundaryTargetSamples,
		BoundaryTargets:                 boundaryTargets,
		Hashes: Hashes{
			SampleIndices: fmt.Sprintf("%x", sampleIndicesHash),
			InputIDs:      fmt.Sprintf("%x", idsHash),
			AttentionMask: fmt.Sprintf("%x", maskHash),
			Offsets:       fmt.Sprintf("%x", offsetsHash),
			Labels:        fmt.Sprintf("%x", labelsHash),
			Chunks:        fmt.Sprintf("%x", chunksHash),
		},
		FirstBatch:               firstBatch,
		DebugSourceTokenMappings: debugSourceTokenMappings,
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(out); err != nil {
		fmt.Fprintf(os.Stderr, "encode json: %v\n", err)
		os.Exit(1)
	}
}
GOEOF

cat > "$compare_py" <<'PYEOF'
#!/usr/bin/env python3
import json
import sys

zig_path, go_path = sys.argv[1], sys.argv[2]
with open(zig_path, "r", encoding="utf-8") as f:
    zig = json.load(f)
with open(go_path, "r", encoding="utf-8") as f:
    go = json.load(f)

fields = [
    "schema_version",
    "offset",
    "samples",
    "batches",
    "max_seq_len",
    "max_chunks",
    "valid_tokens",
    "boundary_gold_tokens",
    "samples_with_active_boundary_labels",
    "contrastive_pos_samples",
    "boundary_target_samples",
    "boundary_targets",
    "first_batch",
]

diffs = []
for field in fields:
    if zig.get(field) != go.get(field):
        diffs.append(field)

hash_diffs = []
for key in ["sample_indices", "input_ids", "attention_mask", "labels", "chunks"]:
    if zig.get("hashes", {}).get(key) != go.get("hashes", {}).get(key):
        hash_diffs.append(key)
if hash_diffs:
    diffs.append("hashes")

if diffs:
    print("fused batch parity mismatch", file=sys.stderr)
    print(f"zig_json={zig_path}", file=sys.stderr)
    print(f"go_json={go_path}", file=sys.stderr)
    for field in diffs[:12]:
        print(f"field={field}", file=sys.stderr)
        print(f"  zig={zig.get(field)!r}", file=sys.stderr)
        print(f"  go ={go.get(field)!r}", file=sys.stderr)
    if hash_diffs:
        print(f"hash_diffs={hash_diffs}", file=sys.stderr)
    sys.exit(1)

print("fused batch parity ok")
print(f"samples={zig['samples']} batches={zig['batches']} valid={zig['valid_tokens']} gold={zig['boundary_gold_tokens']} hashes={zig['hashes']}")
if zig.get("hashes", {}).get("offsets") != go.get("hashes", {}).get("offsets"):
    print(f"offset_hash_note zig={zig['hashes']['offsets']} go={go['hashes']['offsets']}")
PYEOF
chmod +x "$compare_py"

echo "running Zig fused batch parity dump"
(
  cd "$pkg_root"
  "$zig_bin" build -Doptimize=ReleaseFast -Dmetal=true count-fused-tokenization -- \
    --data "$data_path" \
    --model-dir "$model_dir" \
    --split "$split" \
    --max-seq-len "$max_seq_len" \
    --max-chunks "$max_chunks" \
    --batch-size "$batch_size" \
    --offset "$offset" \
    --limit "$limit" \
    --json \
    --dump-first \
    --dump-batch
) > "$zig_stdout" 2>"$zig_stderr"
if [[ -s "$zig_stdout" ]]; then
  cp "$zig_stdout" "$zig_json"
else
  cp "$zig_stderr" "$zig_json"
fi

echo "running Go fused batch parity dump"
(
  cd "$gopeft_dir"
  go run "$go_helper" \
    --data "$data_path" \
    --model-dir "$model_dir" \
    --split "$split" \
    --max-seq-len "$max_seq_len" \
    --max-chunks "$max_chunks" \
    --batch-size "$batch_size" \
    --offset "$offset" \
    --limit "$limit" \
    --dump-batch
) > "$go_json" 2>"$out_dir/go_stderr.log"

python3 "$compare_py" "$zig_json" "$go_json"
echo "zig_json=$zig_json"
echo "go_json=$go_json"
