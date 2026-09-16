"""Regression checks for the PEFT smoke's fail-closed comparison and CLI boundary."""

import copy
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from gemma4_oracle_contract import ContractError
from verify_gemma4_peft_export_roundtrip import require_same_adapter, run_antfly


class RoundtripContractTests(unittest.TestCase):
    def test_tensor_and_semantics_tampering_is_rejected(self):
        source = {
            "inventory": ["layer.lora_A"],
            "semantics": {"r": 2, "lora_alpha": 4},
            "tensors": {
                "layer.lora_A": {"shape": [1, 2], "dtype": "F32", "values": [0.0, 1.0]}
            },
        }
        require_same_adapter(source, copy.deepcopy(source))
        for owner, key, value in [
            ("root", "inventory", []),
            ("semantics", "r", 3),
            ("semantics", "lora_alpha", 8),
            ("tensor", "shape", [2, 1]),
            ("tensor", "dtype", "F16"),
            ("tensor", "values", [0.0, 2.0]),
        ]:
            changed = copy.deepcopy(source)
            target = (
                changed
                if owner == "root"
                else changed["semantics"]
                if owner == "semantics"
                else changed["tensors"]["layer.lora_A"]
            )
            target[key] = value
            with self.subTest(key=key), self.assertRaises(ContractError):
                require_same_adapter(source, changed)
        changed = copy.deepcopy(source)
        changed["tensors"].clear()
        with self.assertRaises(ContractError):
            require_same_adapter(source, changed)

    @patch("verify_gemma4_peft_export_roundtrip.subprocess.run")
    def test_cli_exit_json_shape_and_offline_environment(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, '{"ok":true}', "")
        self.assertEqual(
            run_antfly(Path("/fixture/antfly"), ["inference"]), {"ok": True}
        )
        self.assertEqual(run.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "1")
        for code, output in [(1, "{}"), (0, "[]"), (0, "not json")]:
            run.return_value = subprocess.CompletedProcess(
                [], code, output, "diagnostic"
            )
            with (
                self.subTest(code=code, output=output),
                self.assertRaises(ContractError),
            ):
                run_antfly(Path("/fixture/antfly"), ["inference"])


if __name__ == "__main__":
    unittest.main()
