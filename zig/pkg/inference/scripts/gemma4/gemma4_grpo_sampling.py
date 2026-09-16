"""Host-side Gemma4 rollout sampling matching the pinned Zig 0.16 contract.

Policy scores use temperature-scaled logits. Top-k/top-p filter rollouts only;
the training objective retains the full vocabulary normalizer. This primitive does not qualify a GRPO campaign.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import heapq
import struct
from typing import Callable, Sequence

MASK64 = (1 << 64) - 1
TRAIN_DOMAIN = 0x4752504F54524149
EVAL_DOMAIN = 0x4752504F4556414C


def _u64(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MASK64
    ):
        raise ValueError("seed and logical indices must be unsigned 64-bit integers")
    return value


def _f32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def mix_seed(value: int) -> int:
    value = (_u64(value) + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def group_seed(run_seed: int, domain: int, epoch: int, prompt_index: int) -> int:
    """prompt_index is the original prepared dataset position, before shuffling."""
    seed = mix_seed(_u64(run_seed) ^ _u64(domain))
    seed = mix_seed(seed ^ _u64(epoch))
    return mix_seed(seed ^ _u64(prompt_index))


def completion_seed(seed: int, completion_index: int) -> int:
    return mix_seed(_u64(seed) ^ _u64(completion_index))


class ZigRandom:
    """Xoshiro256++ with Zig 0.16 Random.float(f64), including rare extra draws."""

    def __init__(self, seed: int):
        state = _u64(seed)
        self.state: list[int] = []
        for _ in range(4):
            self.state.append(mix_seed(state))
            state = (state + 0x9E3779B97F4A7C15) & MASK64

    @staticmethod
    def _rotl(value: int, count: int) -> int:
        return ((value << count) | (value >> (64 - count))) & MASK64

    def next_u64(self) -> int:
        s = self.state
        result = (self._rotl((s[0] + s[3]) & MASK64, 23) + s[0]) & MASK64
        temporary = (s[1] << 17) & MASK64
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= temporary
        s[3] = self._rotl(s[3], 45)
        return result

    def uniform(self) -> float:
        word = self.next_u64()
        leading_zeroes = 64 - word.bit_length()
        if leading_zeroes >= 12:
            leading_zeroes = 12
            while True:
                extra = 64 - self.next_u64().bit_length()
                leading_zeroes += extra
                if extra != 64:
                    break
                if leading_zeroes >= 1022:
                    leading_zeroes = 1022
                    break
        bits = ((1022 - leading_zeroes) << 52) | (word & ((1 << 52) - 1))
        return struct.unpack("<d", struct.pack("<Q", bits))[0]


def ranked_tokens(logits: Sequence[float], count: int) -> list[int]:
    """Stable descending F32 scores, with the lowest token ID winning ties."""
    values = _logits(logits)
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 1 <= count <= len(values)
    ):
        raise ValueError("invalid ranked token count")
    return heapq.nsmallest(
        count, range(len(values)), key=lambda token: (-values[token], token)
    )


def _logits(logits: Sequence[float]) -> list[float]:
    values = [_f32(float(value)) for value in logits]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("sampling requires nonempty finite F32 logits")
    return values


@dataclass(frozen=True)
class SamplingPolicy:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    def __post_init__(self) -> None:
        temperature, top_p = _f32(self.temperature), _f32(self.top_p)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("invalid sampling temperature")
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("invalid sampling top-p")
        _u64(self.top_k)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "top_p", top_p)

    def select(self, logits: Sequence[float], draw: float) -> int:
        values = _logits(logits)
        if not math.isfinite(draw) or not 0 <= draw < 1:
            raise ValueError("sampling draw must be finite in [0, 1)")
        if self.top_k == 1:
            return max(range(len(values)), key=values.__getitem__)
        if self.top_k == 0 and self.top_p == 1:
            candidates = list(range(len(values)))
        else:
            count = min(self.top_k, len(values)) if self.top_k else len(values)
            candidates = heapq.nsmallest(
                count, range(len(values)), key=lambda token: (-values[token], token)
            )
        maximum = max(values[token] for token in candidates)
        weights = [
            math.exp(_f32(values[token] - maximum) / self.temperature)
            for token in candidates
        ]
        # Explicit left-to-right sums: Python 3.12 sum(float) compensates, whereas
        # Zig's categorical CDF uses ordinary F64 accumulation.
        total = 0.0
        for weight in weights:
            total += weight
        if self.top_p < 1:
            cumulative = 0.0
            for index, weight in enumerate(weights):
                cumulative += weight
                if cumulative >= self.top_p * total:
                    candidates, weights = candidates[: index + 1], weights[: index + 1]
                    break
            total = 0.0
            for weight in weights:
                total += weight
        threshold = draw * total
        cumulative = 0.0
        for token, weight in zip(candidates, weights):
            cumulative += weight
            if cumulative > threshold:
                return token
        return candidates[-1]


class CompletionSampler:
    """Independent stream per completion; early EOS cannot perturb siblings."""

    def __init__(
        self,
        *,
        run_seed: int,
        epoch: int,
        prompt_index: int,
        completion_index: int,
        evaluation: bool,
        policy: SamplingPolicy,
    ):
        seed = group_seed(
            run_seed, EVAL_DOMAIN if evaluation else TRAIN_DOMAIN, epoch, prompt_index
        )
        self.random = ZigRandom(completion_seed(seed, completion_index))
        self.greedy = evaluation and completion_index == 0
        self.policy = policy

    def select(self, logits: Sequence[float]) -> int:
        if self.greedy:
            return ranked_tokens(logits, 1)[0]
        return self.policy.select(logits, self.random.uniform())


def categorical_rollout_group(
    predict: Callable[[Sequence[int]], Sequence[float]],
    prompt: Sequence[int],
    *,
    run_seed: int,
    epoch: int,
    prompt_index: int,
    evaluation: bool,
    policy: SamplingPolicy,
    group_size: int,
    max_completion_tokens: int,
    eos_token_id: int,
) -> tuple[list[list[int]], list[list[float]]]:
    """Shared prompt, independent streams, and unmodified policy log-probabilities.

    ``predict`` returns the last-position logits for exactly the supplied prefix.
    The caller owns padding and model execution. EOS is included in the result;
    a completion ending at the budget is retained for the caller's mask policy.
    """
    if not prompt or any(
        isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in prompt
    ):
        raise ValueError("rollout prompt must contain nonnegative integer token IDs")
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size < 2
        or isinstance(max_completion_tokens, bool)
        or not isinstance(max_completion_tokens, int)
        or max_completion_tokens < 1
        or isinstance(eos_token_id, bool)
        or not isinstance(eos_token_id, int)
        or eos_token_id < 0
    ):
        raise ValueError("invalid rollout shape or EOS token")
    samplers = [
        CompletionSampler(
            run_seed=run_seed,
            epoch=epoch,
            prompt_index=prompt_index,
            completion_index=i,
            evaluation=evaluation,
            policy=policy,
        )
        for i in range(group_size)
    ]
    sequences: list[list[int]] = [[] for _ in samplers]
    logps: list[list[float]] = [[] for _ in samplers]
    first = _logits(predict(list(prompt)))
    if eos_token_id >= len(first):
        raise ValueError("EOS token is outside the model vocabulary")
    for step in range(max_completion_tokens):
        for i, sampler in enumerate(samplers):
            if sequences[i] and sequences[i][-1] == eos_token_id:
                continue
            logits = (
                first if step == 0 else _logits(predict(list(prompt) + sequences[i]))
            )
            if len(logits) != len(first):
                raise ValueError("rollout vocabulary changed between tokens")
            token = sampler.select(logits)
            # Match Zig's F32 temperature division and F64 normalization.
            # Filtering (including the greedy evaluation anchor) does not
            # renormalize the full-support training policy.
            logits = [_f32(value / _f32(policy.temperature)) for value in logits]
            maximum = max(logits)
            total = 0.0
            for value in logits:
                total += math.exp(_f32(value - maximum))
            logp = _f32(float(logits[token]) - (maximum + math.log(total)))
            sequences[i].append(token)
            logps[i].append(logp)
        if all(values[-1] == eos_token_id for values in sequences):
            break
    return sequences, logps
