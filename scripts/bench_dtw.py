"""Time full vs banded DTW on a synthetic one-hour video.

    python scripts/bench_dtw.py                      # 500 keyframes x 1,000 segments
    python scripts/bench_dtw.py --keyframes 2000 --segments 4000

DTW fills an n x m table, so cost grows with the product of the two lengths. The band
only fills cells whose segment lies within `band` seconds of the keyframe, which turns
it into O(n * k). The script also checks that the banded path costs the same as the
full one, i.e. that the band didn't cut off the optimal alignment.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from vidsense.align import align


def synthetic(n_keyframes: int, n_segments: int, duration: float, seed: int = 0):
    rng = np.random.default_rng(seed)

    def spans(n: int, gap: float) -> np.ndarray:
        edges = np.concatenate([[0.0], np.sort(rng.uniform(0, duration, n - 1)), [duration]])
        return np.stack([edges[:-1], np.maximum(edges[:-1], edges[1:] - gap)], axis=1)

    similarity = rng.normal(0.22, 0.03, size=(n_keyframes, n_segments))  # CLIP-like cosine noise
    return spans(n_keyframes, 0.0), spans(n_segments, 0.2), similarity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keyframes", type=int, default=500)
    parser.add_argument("--segments", type=int, default=1000)
    parser.add_argument("--duration", type=float, default=3600.0, help="video length in seconds")
    args = parser.parse_args()

    keyframes, segments, similarity = synthetic(args.keyframes, args.segments, args.duration)
    print(f"{args.keyframes} keyframes x {args.segments} segments over {args.duration / 60:.0f} min\n")
    print(f"{'band':>8} {'cells':>12} {'seconds':>9}  same cost as full")
    full_cost = None
    for band in (0, 120, 60, 30):
        started = time.perf_counter()
        result = align(keyframes, segments, similarity, semantic_weight=0.3, band_seconds=band)
        elapsed = time.perf_counter() - started
        if band == 0:
            full_cost = result.total_cost
        label = "full" if band == 0 else f"{band:.0f} s"
        same = "-" if band == 0 else ("yes" if abs(result.total_cost - full_cost) < 1e-6 else f"no ({result.total_cost:.3f} vs {full_cost:.3f})")
        print(f"{label:>8} {result.cells:>12,} {elapsed:>9.3f}  {same}")


if __name__ == "__main__":
    main()
