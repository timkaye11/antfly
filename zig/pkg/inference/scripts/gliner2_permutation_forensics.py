#!/usr/bin/env python3
"""Phase 0.3 permutation forensics (Metal_Gliner_Next_steps.md §0.3).

Given raw f32 dumps of the node-1405 multiply operand produced by
TERMITE_GRAPH_NODE_DUMP on the native backend (ground truth) and on the
Metal backend (old/new stride semantics), recover the element permutation
truth -> wrong over the candidate axis factorizations. The permutation's
structure (an axis transpose, a stride mismatch, a block swap) IS the
diagnosis.

Usage:
  python3 scripts/gliner2_permutation_forensics.py \
      --truth /tmp/forensics/native_a.bin \
      --wrong /tmp/forensics/metal_new_a.bin \
      [--also /tmp/forensics/metal_old_a.bin]

Dumps are raw little-endian float32, 6,291,456 elements (24*64*64*64).
"""

import argparse
import itertools
import sys

import numpy as np

NUMEL = 24 * 64 * 64 * 64  # 6,291,456

FACTORIZATIONS = [
    ("bh,qi,ki,d", (24, 64, 64, 64)),
    ("b,h,qi,ki,d", (2, 12, 64, 64, 64)),
    ("bhqi,ki,d", (1536, 64, 64)),
]


def load(path: str) -> np.ndarray:
    arr = np.fromfile(path, dtype="<f4")
    if arr.size != NUMEL:
        sys.exit(f"{path}: expected {NUMEL} floats, got {arr.size}")
    return arr


def ordsum(arr: np.ndarray) -> float:
    # Matches Zig orderChecksum: sum(v[i] * ((i % 97) + 1)).
    weights = (np.arange(arr.size, dtype=np.float64) % 97) + 1
    return float(np.dot(arr.astype(np.float64), weights))


def value_match_report(truth: np.ndarray, wrong: np.ndarray) -> bool:
    """Verify the two buffers are value-preserving permutations of each other."""
    ts, ws = np.sort(truth), np.sort(wrong)
    if np.array_equal(ts, ws):
        print("value multiset: IDENTICAL (pure permutation)")
        return True
    diff = np.abs(ts - ws)
    print(
        f"value multiset: DIFFERS (max sorted diff {diff.max():.6g}, "
        f"count>1e-6 {(diff > 1e-6).sum()}) — NOT a pure permutation"
    )
    return False


def try_axis_transposes(truth: np.ndarray, wrong: np.ndarray) -> bool:
    """Test whether wrong == truth reshaped to shape, axes permuted."""
    found = False
    for name, shape in FACTORIZATIONS:
        t = truth.reshape(shape)
        ndim = len(shape)
        for perm in itertools.permutations(range(ndim)):
            if perm == tuple(range(ndim)):
                continue
            if sorted(t.transpose(perm).shape) != sorted(shape):
                continue
            cand = t.transpose(perm).reshape(-1)
            if cand.size != wrong.size:
                continue
            if np.array_equal(cand, wrong):
                print(f"MATCH: wrong = truth.reshape({name}={shape}).transpose{perm}.flatten()")
                found = True
    if not found:
        print("no single axis-transpose over the candidate factorizations reproduces `wrong`")
    return found


def exact_permutation(truth: np.ndarray, wrong: np.ndarray) -> np.ndarray | None:
    """Recover index permutation p with wrong[i] == truth[p[i]].

    Values are near-unique; duplicates are assigned in order (stable), which
    keeps block structure visible.
    """
    order_t = np.argsort(truth, kind="stable")
    order_w = np.argsort(wrong, kind="stable")
    if not np.array_equal(truth[order_t], wrong[order_w]):
        return None
    p = np.empty(NUMEL, dtype=np.int64)
    p[order_w] = order_t
    return p


def describe_permutation(p: np.ndarray) -> None:
    moved = p != np.arange(NUMEL)
    n_moved = int(moved.sum())
    print(f"permutation: {n_moved}/{NUMEL} elements moved ({100.0 * n_moved / NUMEL:.2f}%)")
    if n_moved == 0:
        return
    delta = p - np.arange(NUMEL)
    vals, counts = np.unique(delta[moved], return_counts=True)
    top = np.argsort(counts)[::-1][:12]
    print("top displacement deltas (delta: count):")
    for i in top:
        print(f"  {vals[i]:+12d}: {counts[i]}")
    for name, shape in FACTORIZATIONS:
        src = np.array(np.unravel_index(p, shape)).T  # for each dest flat i: source coords
        dst = np.array(np.unravel_index(np.arange(NUMEL), shape)).T
        same = (src == dst).all(axis=0)
        labels = name.split(",")
        print(f"factorization {name}: axes preserved per-element: ", end="")
        print(", ".join(f"{labels[ax]}={'ALL' if same[ax] else f'{(src[:, ax] == dst[:, ax]).mean():.4f}'}" for ax in range(len(shape))))
        # Detect pure coordinate permutation: for moved elements, is the source
        # coordinate tuple a fixed permutation of the dest coordinates?
        m = moved
        for perm in itertools.permutations(range(len(shape))):
            if perm == tuple(range(len(shape))):
                continue
            if all(shape[perm[ax]] == shape[ax] for ax in range(len(shape))) and np.array_equal(
                src[m], dst[m][:, list(perm)]
            ):
                print(f"  -> EXACT coordinate permutation under {name}: source = dest axes {perm}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, help="native-backend dump (ground truth)")
    ap.add_argument("--wrong", required=True, help="metal dump to diagnose")
    ap.add_argument("--also", help="optional third dump (e.g. metal-old) for checksum table")
    args = ap.parse_args()

    truth = load(args.truth)
    wrong = load(args.wrong)

    print(f"truth: abs={np.abs(truth).sum():.4f} ordsum={ordsum(truth):.4f}")
    print(f"wrong: abs={np.abs(wrong).sum():.4f} ordsum={ordsum(wrong):.4f}")
    if args.also:
        also = load(args.also)
        print(f"also:  abs={np.abs(also).sum():.4f} ordsum={ordsum(also):.4f}")

    if np.array_equal(truth, wrong):
        print("buffers are IDENTICAL — no permutation; the defect is elsewhere")
        return

    pure = value_match_report(truth, wrong)
    try_axis_transposes(truth, wrong)
    if pure:
        p = exact_permutation(truth, wrong)
        if p is None:
            print("could not recover exact permutation (unexpected for a pure permutation)")
        else:
            describe_permutation(p)


if __name__ == "__main__":
    main()
