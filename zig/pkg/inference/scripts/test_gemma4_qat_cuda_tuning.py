#!/usr/bin/env python3

import os
import pathlib
import subprocess
import unittest


TUNING_SCRIPT = pathlib.Path(__file__).resolve().with_name("gemma4_qat_cuda_tuning.sh")
WRAPPER_SCRIPT = pathlib.Path(__file__).resolve().with_name("with_gemma4_qat_cuda_tuning.sh")


def configured_environment(**overrides: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(overrides)
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; gemma4_qat_cuda_tuning_env 777; printf "%s\\n" "${GEMMA4_QAT_CUDA_ENV[@]}"',
            "bash",
            str(TUNING_SCRIPT),
        ],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )
    return dict(line.split("=", 1) for line in completed.stdout.splitlines())


def wrapped_environment(**overrides: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(overrides)
    return subprocess.run(
        [str(WRAPPER_SCRIPT), "/usr/bin/env"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


class Gemma4QatCudaTuningTest(unittest.TestCase):
    def test_generated_attention_defaults_disabled_and_honors_explicit_opt_in(self):
        environment = configured_environment()
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_DECODE"])
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_SCORE_PREWORK"])
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_TURBOQUANT_SPLIT_ATTENTION"])

        for name in (
            "ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_DECODE",
            "ANTFLY_GENERATED_ATTENTION_DECODE",
            "antfly_generated_attention_decode",
        ):
            with self.subTest(name=name):
                environment = configured_environment(**{name: "1"})
                self.assertEqual("1", environment["ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_DECODE"])

        for name in (
            "ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_SCORE_PREWORK",
            "ANTFLY_GENERATED_ATTENTION_SCORE_PREWORK",
            "antfly_generated_attention_score_prework",
        ):
            with self.subTest(name=name):
                environment = configured_environment(**{name: "1"})
                self.assertEqual("1", environment["ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_SCORE_PREWORK"])

        for name in (
            "ANTFLY_INFERENCE_CUDA_TURBOQUANT_SPLIT_ATTENTION",
            "ANTFLY_TURBOQUANT_SPLIT_ATTENTION",
            "antfly_turboquant_split_attention",
        ):
            with self.subTest(name=name):
                environment = configured_environment(**{name: "1"})
                self.assertEqual("1", environment["ANTFLY_INFERENCE_CUDA_TURBOQUANT_SPLIT_ATTENTION"])

    def test_readback_tuning_defaults_to_enabled(self):
        environment = configured_environment()
        self.assertEqual("1", environment["ANTFLY_INFERENCE_CUDA_ASYNC_I32_DOWNLOAD_STAGING"])
        self.assertEqual("1", environment["ANTFLY_INFERENCE_CUDA_GREEDY_PENDING_TOKEN_READBACK"])
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_GENERATED_Q6_K_Q8_1_LM_HEAD_ARGMAX"])
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN_EXACT"])
        self.assertEqual("853", environment["ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD"])
        self.assertEqual("2500", environment["ANTFLY_INFERENCE_CUDA_TEMP_SLOT_SKIP"])
        self.assertEqual("1", environment["ANTFLY_INFERENCE_CUDA_SERVER_REQUEST_GRAPH_RESET"])

    def test_graph_off_removes_graph_only_temp_and_request_state(self):
        environment = configured_environment(ANTFLY_DECODE_GRAPH_REPLAY="off")
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD"])
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_TEMP_SLOT_SKIP"])
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_SERVER_REQUEST_GRAPH_RESET"])
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_CAPTURE_PERSISTENT_REPLAY"])

    def test_batching_wrapper_requires_graph_replay_off(self):
        rejected = wrapped_environment(
            ANTFLY_SERVER_DISABLE_CONTINUOUS_BATCHING="0",
            ANTFLY_SERVER_DECODE_GRAPH_REPLAY="required",
        )
        self.assertEqual(2, rejected.returncode)
        self.assertIn("requires ANTFLY_SERVER_DECODE_GRAPH_REPLAY=off", rejected.stderr)

        accepted = wrapped_environment(
            ANTFLY_SERVER_DISABLE_CONTINUOUS_BATCHING="0",
            ANTFLY_SERVER_DECODE_GRAPH_REPLAY="off",
        )
        self.assertEqual(0, accepted.returncode, accepted.stderr)
        environment = dict(line.split("=", 1) for line in accepted.stdout.splitlines() if "=" in line)
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_SERVER_REQUEST_GRAPH_RESET"])
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD"])
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_TEMP_SLOT_SKIP"])

    def test_temp_slot_schedule_honors_ambient_overrides(self):
        for name in (
            "ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD",
            "ANTFLY_CUDA_TEMP_SLOT_PERIOD",
            "antfly_cuda_temp_slot_period",
        ):
            with self.subTest(name=name):
                environment = configured_environment(**{name: "997"})
                self.assertEqual("997", environment["ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD"])

        for name in (
            "ANTFLY_INFERENCE_CUDA_TEMP_SLOT_SKIP",
            "ANTFLY_CUDA_TEMP_SLOT_SKIP",
            "antfly_cuda_temp_slot_skip",
        ):
            with self.subTest(name=name):
                environment = configured_environment(**{name: "2112"})
                self.assertEqual("2112", environment["ANTFLY_INFERENCE_CUDA_TEMP_SLOT_SKIP"])

    def test_readback_tuning_honors_ambient_overrides(self):
        environment = configured_environment(
            ANTFLY_INFERENCE_CUDA_ASYNC_I32_DOWNLOAD_STAGING="0",
            ANTFLY_INFERENCE_CUDA_GREEDY_PENDING_TOKEN_READBACK="0",
        )
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_ASYNC_I32_DOWNLOAD_STAGING"])
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_GREEDY_PENDING_TOKEN_READBACK"])

    def test_generated_q6_lm_head_gate_honors_ambient_override(self):
        environment = configured_environment(
            ANTFLY_INFERENCE_CUDA_GENERATED_Q6_K_Q8_1_LM_HEAD_ARGMAX="1",
        )
        self.assertEqual("1", environment["ANTFLY_INFERENCE_CUDA_GENERATED_Q6_K_Q8_1_LM_HEAD_ARGMAX"])

    def test_exact_e2b_ffn_gate_honors_ambient_override(self):
        environment = configured_environment(
            ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN_EXACT="1",
        )
        self.assertEqual("1", environment["ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN_EXACT"])

    def test_f32_ffn_comparison_q8_routes_honor_explicit_overrides(self):
        environment = configured_environment(
            ANTFLY_INFERENCE_CUDA_Q4_0_GATE_UP_ACTIVATION_Q8_1_PRECOMPUTE="0",
            ANTFLY_INFERENCE_CUDA_Q4_0_LINEAR_Q8_1_DP4A="0",
            ANTFLY_INFERENCE_CUDA_Q4_0_PAIR_Q8_1_DP4A="0",
            ANTFLY_INFERENCE_CUDA_Q4_0_PAIR_ACTIVATION_Q8_1_DP4A="0",
            ANTFLY_INFERENCE_CUDA_Q4_0_GATED_DOWN_Q8_1_DP4A="0",
            ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN="0",
            ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN_EXACT="1",
        )
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_Q4_0_GATE_UP_ACTIVATION_Q8_1_PRECOMPUTE"])
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_Q4_0_LINEAR_Q8_1_DP4A"])
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_Q4_0_PAIR_Q8_1_DP4A"])
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_Q4_0_PAIR_ACTIVATION_Q8_1_DP4A"])
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_Q4_0_GATED_DOWN_Q8_1_DP4A"])
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN"])
        self.assertEqual("1", environment["ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN_EXACT"])


if __name__ == "__main__":
    unittest.main()
