"""Cross-language RNG/CDF fixtures and independent sampling edge cases."""
import json
import hashlib
import math
from pathlib import Path
import struct
import unittest
from unittest.mock import patch

from gemma4_grpo_sampling import (
    CompletionSampler, EVAL_DOMAIN, MASK64, SamplingPolicy, TRAIN_DOMAIN,
    ZigRandom, completion_seed, group_seed, ranked_tokens, categorical_rollout_group,
)


class SamplingTests(unittest.TestCase):
    def test_rollout_preserves_fixture_streams_across_tokens_and_prompt_order(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/grpo_sampling_zig_0_16.json').read_text())
        # Compare full rollout wiring against executable Zig outputs, not a
        # second call to the Python sampler. EOS is deliberately unreachable.
        keys = {(r[0], r[2], r[3]) for r in fixture['rows'] if r[1] == TRAIN_DOMAIN}
        for run, epoch, prompt in sorted(keys, reverse=True):
            rows = [r for r in fixture['rows'] if r[:4] == [run, TRAIN_DOMAIN, epoch, prompt]]
            completions = sorted({r[4] for r in rows})
            steps = max(r[7] for r in rows) + 1
            for policy_index, options in enumerate(fixture['policies']):
                calls = []
                logits = fixture['logits'] + [-10000.0]
                def predict(prefix):
                    calls.append(list(prefix))
                    return logits
                sequences, _ = categorical_rollout_group(
                    predict, [7], run_seed=run, epoch=epoch, prompt_index=prompt,
                    evaluation=False, policy=SamplingPolicy(**options),
                    group_size=max(completions) + 1, max_completion_tokens=steps,
                    eos_token_id=len(logits) - 1,
                )
                expected = [[r[9 + policy_index] for r in rows if r[4] == i] for i in completions]
                self.assertEqual(expected, [sequences[i] for i in completions])
                self.assertEqual(1 + (max(completions) + 1) * (steps - 1), len(calls))

    def test_rollout_eos_is_included_and_siblings_keep_their_stream(self):
        def predict(prefix):
            return [0., 0., 0., 0.]
        kwargs = dict(run_seed=42, epoch=0, prompt_index=7, evaluation=False,
                      policy=SamplingPolicy(), group_size=4, max_completion_tokens=4)
        full, _ = categorical_rollout_group(predict, [8], eos_token_id=3, **kwargs)
        alternate, _ = categorical_rollout_group(predict, [8], eos_token_id=2, **kwargs)
        for a, b in zip(full, alternate):
            self.assertEqual(a[:min(len(a), len(b))], b[:min(len(a), len(b))])
        self.assertTrue(any(len(values) < 4 for values in full + alternate))

    def test_rollout_scores_unmodified_policy_and_evaluation_zero_is_greedy(self):
        logits = [0., 2., -4.]
        sequences, logps = categorical_rollout_group(
            lambda prefix: logits, [8], run_seed=17, epoch=0, prompt_index=9,
            evaluation=True, policy=SamplingPolicy(temperature=3., top_k=1),
            group_size=4, max_completion_tokens=3, eos_token_id=2,
        )
        expected = 2. - math.log(sum(math.exp(v) for v in logits))
        self.assertEqual([[1, 1, 1]] * 4, sequences)
        for values in logps:
            for value in values:
                self.assertAlmostEqual(expected, value, places=7)
                self.assertLess(value, 0.)  # Filtering to one token must not score it as log(1).

    def test_rollout_rejects_invalid_model_output_and_empty_prompt(self):
        kwargs = dict(run_seed=0, epoch=0, prompt_index=0, evaluation=False,
                      policy=SamplingPolicy(), group_size=2, max_completion_tokens=2, eos_token_id=1)
        for output, prompt in (([0., float('nan')], [0]), ([0.], [0]), ([0., 1.], [])):
            with self.assertRaises(ValueError):
                categorical_rollout_group(lambda prefix: output, prompt, **kwargs)

    def test_fixture_matches_production_source(self):
        from generate_gemma4_grpo_sampling_fixture import extract_sampling_source
        fixture = json.loads((Path(__file__).parent / 'fixtures/grpo_sampling_zig_0_16.json').read_text())
        self.assertEqual(hashlib.sha256(extract_sampling_source().encode()).hexdigest(), fixture['extracted_source_sha256'], 'Production sampler changed; regenerate the executable Zig fixture')

    def test_executable_zig_fixture(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/grpo_sampling_zig_0_16.json').read_text())
        streams = {}
        policies = [SamplingPolicy(**options) for options in fixture['policies']]
        for row in fixture['rows']:
            run, domain, epoch, prompt, completion, group, seed, step, bits, *tokens = row
            self.assertEqual(group_seed(run, domain, epoch, prompt), group)
            self.assertEqual(completion_seed(group, completion), seed)
            key = (run, domain, epoch, prompt, completion)
            if step == 0:
                streams[key] = ZigRandom(seed)
            draw = streams[key].uniform()
            self.assertEqual(struct.unpack('<Q', struct.pack('<d', draw))[0], bits)
            self.assertEqual([p.select(fixture['logits'], draw) for p in policies], tokens)

    def test_upstream_xoshiro_sequence(self):
        rng = ZigRandom(0)
        self.assertEqual([rng.next_u64() for _ in range(6)], [
            0x53175D61490B23DF, 0x61DA6F3DC380D507, 0x5C0FDF91EC9A7BFC,
            0x02EEBF8C3BBE5E1A, 0x7ECA04EBAF4A5EEA, 0x0543C37757F08D9A,
        ])

    def test_rare_exponent_draws(self):
        rng = ZigRandom(0)
        with patch.object(rng, 'next_u64', side_effect=[3, 0, 1 << 63]) as draw:
            value = rng.uniform()
            expected = struct.unpack('<d', struct.pack('<Q', ((1022 - 76) << 52) | 3))[0]
            self.assertEqual(value, expected)
            self.assertEqual(draw.call_count, 3)
        with patch.object(rng, 'next_u64', side_effect=[0] * 17):
            self.assertEqual(rng.uniform(), 0.0)

    def test_uniform_cdf_preserves_vocabulary_order_and_strict_boundary(self):
        policy = SamplingPolicy()
        self.assertEqual([policy.select([0, 0, 0, 0], x) for x in [0, .25, .5, .75, math.nextafter(1, 0)]], [0, 1, 2, 3, 3])
        # Equal probabilities are still sampled with replacement.
        self.assertEqual([policy.select([0, 0], 0) for _ in range(3)], [0, 0, 0])

    def test_nucleus_retains_crossing_token_and_top_k_precedes_top_p(self):
        self.assertEqual(SamplingPolicy(top_p=.5).select([0, 0, 0], .99), 1)
        self.assertEqual(SamplingPolicy(top_k=2, top_p=.5).select([0, 0, 0], .99), 0)
        self.assertEqual(SamplingPolicy(top_k=100).select([0, 0], .99), 1)

    def test_ties_use_lowest_id_after_f32_rounding(self):
        self.assertEqual(ranked_tokens([1, 1 + 1e-9, 1], 3), [0, 1, 2])
        self.assertEqual(SamplingPolicy(top_k=1).select([2, 2, 1], .99), 0)

    def test_extreme_finite_logits(self):
        for draw in (0, .5, math.nextafter(1, 0)):
            self.assertEqual(SamplingPolicy().select([-3e38, 3e38], draw), 1)

    def test_invalid_inputs_fail_closed(self):
        for logits in ([], [math.nan], [math.inf], [1e100]):
            with self.assertRaises(ValueError):
                SamplingPolicy(top_k=1).select(logits, 0)
        for draw in (-.1, 1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                SamplingPolicy(top_k=1).select([0], draw)
        for options in ({'temperature': 0}, {'temperature': 1e-100}, {'temperature': math.inf}, {'top_p': 0}, {'top_p': 1.1}, {'top_k': -1}, {'top_k': True}):
            with self.assertRaises(ValueError):
                SamplingPolicy(**options)
        for seed in (-1, MASK64 + 1, True, .5):
            with self.assertRaises(ValueError):
                ZigRandom(seed)

    def test_streams_use_original_prompt_identity(self):
        def sample(prompt):
            sampler = CompletionSampler(run_seed=42, epoch=7, prompt_index=prompt, completion_index=3, evaluation=False, policy=SamplingPolicy())
            return [sampler.select([0] * 100) for _ in range(16)]
        ordered = {p: sample(p) for p in [0, 1, 2]}
        shuffled = {p: sample(p) for p in [2, 0, 1]}
        self.assertEqual(ordered, shuffled)
        self.assertNotEqual(ordered[0], ordered[1])
        self.assertNotEqual(group_seed(42, TRAIN_DOMAIN, 0, 0), group_seed(42, EVAL_DOMAIN, 0, 0))

    def test_evaluation_first_completion_is_always_greedy(self):
        sampler = CompletionSampler(run_seed=42, epoch=0, prompt_index=0, completion_index=0, evaluation=True, policy=SamplingPolicy())
        with patch.object(sampler.random, 'uniform', side_effect=AssertionError('greedy must not draw')):
            self.assertEqual(sampler.select([0, 2, 2]), 1)
            self.assertEqual(sampler.select([3, 0, 0]), 0)

    def test_training_completion_zero_is_stochastic(self):
        sampler = CompletionSampler(run_seed=42, epoch=0, prompt_index=0, completion_index=0, evaluation=False, policy=SamplingPolicy())
        with patch.object(sampler.random, 'uniform', return_value=0):
            self.assertEqual(sampler.select([0, 1]), 0)

    def test_eos_in_one_completion_cannot_perturb_siblings(self):
        def sampler(index):
            return CompletionSampler(run_seed=42, epoch=0, prompt_index=17, completion_index=index, evaluation=False, policy=SamplingPolicy())
        first, sibling, isolated = sampler(0), sampler(1), sampler(1)
        first.select([0, 0])  # Stops at EOS, leaving sibling's stream untouched.
        self.assertEqual([sibling.select([0] * 32) for _ in range(16)], [isolated.select([0] * 32) for _ in range(16)])


if __name__ == '__main__':
    unittest.main()
