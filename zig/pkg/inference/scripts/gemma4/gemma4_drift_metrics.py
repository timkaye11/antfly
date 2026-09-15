"""Diagnostic vector distances; no release tolerances or parity claims."""

from __future__ import annotations

import math
from typing import Any


def summarize_squares(
    reference_squared: float,
    candidate_squared: float,
    dot: float,
    difference_squared: float,
    max_abs: float,
) -> dict[str, float | None]:
    values = (reference_squared, candidate_squared, dot, difference_squared, max_abs)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite vector metric")
    if min(reference_squared, candidate_squared, difference_squared, max_abs) < 0:
        raise ValueError("negative norm or distance")
    reference_norm = math.sqrt(reference_squared)
    candidate_norm = math.sqrt(candidate_squared)
    difference_norm = math.sqrt(difference_squared)
    denominator = reference_norm * candidate_norm
    return {
        "reference_l2": reference_norm,
        "candidate_l2": candidate_norm,
        "difference_l2": difference_norm,
        "relative_l2_error": difference_norm / reference_norm if reference_norm else (0.0 if not candidate_norm else None),
        "relative_norm_difference": abs(candidate_norm - reference_norm) / reference_norm if reference_norm else (0.0 if not candidate_norm else None),
        "cosine": max(-1.0, min(1.0, dot / denominator)) if denominator else (1.0 if not reference_norm and not candidate_norm else None),
        "max_abs": max_abs,
    }


def compare_vectors(reference: Any, candidate: Any) -> dict[str, float | None]:
    import numpy as np

    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("vector shapes must match and be nonempty")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("non-finite vector")
    delta = right - left
    return summarize_squares(
        float(np.sum(left * left)), float(np.sum(right * right)),
        float(np.sum(left * right)), float(np.sum(delta * delta)),
        float(np.max(np.abs(delta))),
    )
