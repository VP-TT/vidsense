"""Keyframe selection: temporal K-means over CLIP frame embeddings.

Each sampled frame becomes the feature vector

    [ CLIP embedding (512-d, unit length) , time_weight * t / T ]

where T = duration / K is the expected length of one cluster. Plain K-means on the
embeddings would merge a scene with its later reappearance (the speaker returns after
a slide, say) and pick one frame for both. DTW needs keyframes in time order, so the
time term keeps clusters contiguous: a scene that comes back gets its own keyframe.

The keyframe of each cluster is its medoid (the real frame nearest the centroid).
Finally, neighbouring keyframes that are near-duplicates are merged, which is what
keeps a long static shot from producing a row of identical thumbnails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MAX_CLUSTERS = 300  # keeps K-means fast on long videos


@dataclass
class KeyframeChoice:
    frame_index: int  # index into the sampled frames
    span_start: float  # time range of the frames this keyframe stands for
    span_end: float
    frame_count: int


def choose_k(duration: float, n_frames: int, seconds_per_keyframe: float) -> int:
    return max(1, min(n_frames, MAX_CLUSTERS, math.ceil(duration / max(seconds_per_keyframe, 1e-6))))


def select_keyframes(
    times: np.ndarray,
    embeddings: np.ndarray,
    *,
    frame_interval: float,
    seconds_per_keyframe: float = 5.0,
    time_weight: float = 0.3,
    dedup_threshold: float = 0.95,
    seed: int = 0,
) -> list[KeyframeChoice]:
    """Pick keyframes from sampled frames (times ascending, embeddings L2-normalised)."""
    n = len(times)
    if n == 0:
        return []
    times = np.asarray(times, dtype=np.float64)
    duration = float(times[-1] - times[0]) + frame_interval
    k = choose_k(duration, n, seconds_per_keyframe)

    if k == n:
        labels = np.arange(n)
        centers = None
        features = embeddings
    else:
        from sklearn.cluster import KMeans

        cluster_len = duration / k
        time_feature = time_weight * (times - times[0]) / cluster_len
        features = np.hstack([embeddings, time_feature[:, None]])
        km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(features)
        labels, centers = km.labels_, km.cluster_centers_

    choices = []
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c)
        if centers is None:
            medoid = int(members[0])
        else:
            medoid = int(members[np.argmin(np.linalg.norm(features[members] - centers[c], axis=1))])
        choices.append(
            KeyframeChoice(
                frame_index=medoid,
                span_start=float(times[members].min()),
                span_end=float(times[members].max() + frame_interval),
                frame_count=len(members),
            )
        )
    choices.sort(key=lambda choice: choice.frame_index)
    return merge_duplicates(choices, embeddings, dedup_threshold)


def merge_duplicates(choices: list[KeyframeChoice], embeddings: np.ndarray, threshold: float) -> list[KeyframeChoice]:
    """Merge consecutive keyframes whose CLIP cosine similarity is at least `threshold`.

    The merged keyframe keeps the frame that stood for more of the video.
    """
    if not choices:
        return []
    merged = [choices[0]]
    for choice in choices[1:]:
        last = merged[-1]
        similarity = float(embeddings[choice.frame_index] @ embeddings[last.frame_index])
        if similarity >= threshold:
            keep = last if last.frame_count >= choice.frame_count else choice
            merged[-1] = KeyframeChoice(
                frame_index=keep.frame_index,
                span_start=min(last.span_start, choice.span_start),
                span_end=max(last.span_end, choice.span_end),
                frame_count=last.frame_count + choice.frame_count,
            )
        else:
            merged.append(choice)
    return merged


def coverage(embeddings: np.ndarray, keyframe_indices: list[int], threshold: float = 0.9) -> dict[str, float]:
    """How much of the sampled footage the keyframes represent.

    For every sampled frame we take its best cosine similarity to any keyframe.
    `coverage` is the share of frames at or above `threshold`; `compression` is
    keyframes / sampled frames. Together they say "N keyframes stand in for X% of
    the video", which is a measurable version of "information retention".
    """
    if len(embeddings) == 0 or not keyframe_indices:
        return {"coverage": 0.0, "mean_best_similarity": 0.0, "threshold": threshold, "compression": 0.0}
    best = (embeddings @ embeddings[keyframe_indices].T).max(axis=1)
    return {
        "coverage": round(float((best >= threshold).mean()), 4),
        "mean_best_similarity": round(float(best.mean()), 4),
        "threshold": threshold,
        "compression": round(len(keyframe_indices) / len(embeddings), 4),
    }
