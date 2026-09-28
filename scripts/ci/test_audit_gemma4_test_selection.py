"""The CI audit must detect silently unselected and duplicate-short-name tests."""

import unittest
from audit_gemma4_test_selection import audit


class SelectionTests(unittest.TestCase):
    def test_missing_module_is_not_hidden_by_same_short_name(self):
        diff = (
            '+++ b/zig/pkg/inference/src/finetune/chat_template.zig\n+test "empty" {\n'
        )
        self.assertEqual(len(audit(diff, "1/1 other.test.empty...OK")["missing"]), 1)
        self.assertEqual(
            audit(diff, "1/1 finetune.chat_template.test.empty...OK")["missing"], []
        )

    def test_skips_are_visible_and_anonymous_roots_are_not_named_tests(self):
        diff = '+++ b/zig/pkg/inference/src/ops/native_compute.zig\n+test {\n+test "fixture" {\n'
        report = audit(diff, "1/1 ops.native_compute.test.fixture...SKIP")
        self.assertEqual(report["added_named_tests"], 1)
        self.assertEqual(report["missing"], [])
        self.assertEqual(
            report["optional_skipped"], ["ops.native_compute.test.fixture"]
        )


if __name__ == "__main__":
    unittest.main()
