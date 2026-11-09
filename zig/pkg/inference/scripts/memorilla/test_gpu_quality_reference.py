# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0
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

import json

import pytest
import torch

from gpu_quality_reference import released_cases, score_case, validate_cache


def test_reference_cache_rejects_precision_or_device_substitution():
    cache = ((torch.zeros(1), torch.zeros(1)),)
    validate_cache(cache, torch.float32, "cpu")
    with pytest.raises(ValueError, match="precision/device"):
        validate_cache(cache, torch.bfloat16, "cpu")
    with pytest.raises(ValueError, match="precision/device"):
        validate_cache(cache, torch.float32)
    with pytest.raises(ValueError, match="precision/device"):
        validate_cache((), torch.float32, "cpu")


def test_released_benchmark_suffix_does_not_change_encoder_query(tmp_path):
    source = tmp_path / "upstream" / "memorilla"
    source.mkdir(parents=True)
    (source / "benchmarks.py").write_text(
        'TRIVIA_PROMPT = "Trivia system"\nTRIVIA_SUFFIX = "\\nYour answer:"\n'
    )
    sample = tmp_path / "sample.json"
    sample.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "index": 7,
                        "question": "Who owns it?  ",
                        "documents": ["Mira owns it."],
                        "answer": "Mira",
                    }
                ]
            }
        )
    )
    _, rows, system = released_cases(sample, "triviaqa", source.parent)
    assert system == "Trivia system"
    assert rows[0]["question"] == "Who owns it?"
    assert rows[0]["decoder_question"] == "Who owns it?\nYour answer:"
    value = json.loads(sample.read_text())
    value["rows"][0]["documents"] *= 65
    sample.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="document bound"):
        released_cases(sample, "triviaqa", source.parent)


def test_benchmark_exactness_does_not_reward_answer_mentions():
    case = {"benchmark_answer": "the Pacific Ocean"}
    assert score_case("Pacific Ocean.", case)["correct"]
    assert not score_case("Not the Pacific Ocean", case)["correct"]
    assert not score_case("Pacific Ocean or Atlantic Ocean", case)["correct"]
