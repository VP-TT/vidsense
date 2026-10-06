"""Align keyframes with transcript segments using banded dynamic time warping (DTW).

Both sequences are in time order: n keyframes, each standing for a span of video, and
m Whisper segments. DTW finds the monotonic matching with the lowest total cost:
every keyframe is matched to at least one segment, every segment to at least one
keyframe, and the matching never goes back in time. That guarantee is the reason to
use DTW rather than bucketing by timestamp: no sentence is left without a picture, no
keyframe without text, and the path can stretch or squeeze where the narration runs
ahead of or behind what is on screen.

The cost of matching keyframe i with segment j is

    cost(i, j) = (1 - w) * gap(i, j) / time_scale  +  w * (1 - semantic(i, j))

where gap is the distance in seconds from the middle of the segment to the keyframe's
span (0 if the midpoint falls inside it). Using the midpoint means a sentence that
straddles a cut goes to the scene it mostly belongs to. semantic compares the CLIP
similarity of the segment text with this keyframe against its best-matching keyframe:
a shortfall of SIMILARITY_SCALE or more counts as a full mismatch. A fixed scale (rather
than rescaling each sentence to [0, 1]) keeps CLIP noise on generic sentences such as
"around the world." from moving them. With w = 0 the alignment uses
timestamps only; with w > 0 a sentence can shift to the neighbouring keyframe it
describes better, by up to about time_scale * w / (1 - w) seconds.

The band (Sakoe-Chiba style, but measured in seconds because neither sequence is
uniformly sampled) skips every pair more than `band_seconds` apart. That turns the
O(n * m) table into O(n * k), where k is the number of segments inside the window.
If the band is too narrow to connect start and end, we fall back to full DTW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .schemas import Chunk, Keyframe, Segment

# CLIP ViT-B/32 image-text cosines differ by roughly 0.1 between a matching and an
# unrelated caption; differences of a few hundredths are mostly noise.
SIMILARITY_SCALE = 0.1


@dataclass
class Alignment:
    path: list[tuple[int, int]]  # (keyframe index, segment index), from (0, 0) to (n-1, m-1)
    path_costs: list[float]  # cost of each matched pair on the path
    total_cost: float
    cells: int  # table cells evaluated; n * m for full DTW
    banded: bool  # False if the band could not connect the ends and full DTW ran instead


def cost_matrix(
    keyframe_spans: np.ndarray,
    segment_spans: np.ndarray,
    similarity: np.ndarray | None = None,
    *,
    semantic_weight: float = 0.0,
    time_scale: float = 10.0,
) -> np.ndarray:
    """(n, m) matching costs. Spans are (start, end) rows in seconds."""
    k_start, k_end = keyframe_spans[:, :1], keyframe_spans[:, 1:]
    middle = segment_spans.mean(axis=1)[None, :]
    gap = np.maximum(0.0, np.maximum(k_start - middle, middle - k_end))
    cost = gap / time_scale
    if similarity is not None and semantic_weight > 0:
        shortfall = similarity.max(axis=0, keepdims=True) - similarity
        mismatch = np.clip(shortfall / SIMILARITY_SCALE, 0.0, 1.0)
        cost = (1 - semantic_weight) * cost + semantic_weight * mismatch
    return cost


def band_rows(keyframe_spans: np.ndarray, segment_spans: np.ndarray, band_seconds: float) -> list[tuple[int, int]]:
    """For each keyframe, the contiguous range of segments whose midpoint is within `band_seconds` of its span."""
    middles = np.maximum.accumulate(segment_spans.mean(axis=1))  # monotone, so searchsorted is valid
    rows = []
    for k_start, k_end in keyframe_spans:
        lo = int(np.searchsorted(middles, k_start - band_seconds, side="left"))
        hi = int(np.searchsorted(middles, k_end + band_seconds, side="right")) - 1
        rows.append((lo, hi))
    return rows


def dtw(cost: np.ndarray, rows: list[tuple[int, int]] | None = None) -> tuple[list[tuple[int, int]], float, int] | None:
    """Classic DTW with steps (1,0), (0,1), (1,1). Returns (path, total cost, cells) or None.

    `rows` limits row i to columns rows[i][0]..rows[i][1]; None means the full table.
    Plain Python lists are used in the inner loop because they are several times
    faster than indexing numpy scalars one at a time.
    """
    n, m = cost.shape
    inf = math.inf
    table = [[inf] * (m + 1) for _ in range(n + 1)]
    table[0][0] = 0.0
    costs = cost.tolist()
    cells = 0
    for i in range(1, n + 1):
        lo, hi = rows[i - 1] if rows is not None else (0, m - 1)
        prev, cur, row_cost = table[i - 1], table[i], costs[i - 1]
        for j in range(max(lo, 0) + 1, min(hi, m - 1) + 2):
            best = min(prev[j - 1], prev[j], cur[j - 1])
            if best < inf:
                cur[j] = row_cost[j - 1] + best
            cells += 1
    if not math.isfinite(table[n][m]):
        return None

    path, i, j = [], n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        # Prefer the diagonal on ties: it advances both sequences together.
        _, i, j = min((table[i - 1][j - 1], i - 1, j - 1), (table[i - 1][j], i - 1, j), (table[i][j - 1], i, j - 1))
    path.reverse()
    return path, table[n][m], cells


def align(
    keyframe_spans: np.ndarray,
    segment_spans: np.ndarray,
    similarity: np.ndarray | None = None,
    *,
    semantic_weight: float = 0.3,
    time_scale: float = 10.0,
    band_seconds: float = 60.0,
) -> Alignment:
    n, m = len(keyframe_spans), len(segment_spans)
    if n == 0 or m == 0:
        return Alignment(path=[], path_costs=[], total_cost=0.0, cells=0, banded=False)
    keyframe_spans = np.asarray(keyframe_spans, dtype=float).reshape(n, 2)
    segment_spans = np.asarray(segment_spans, dtype=float).reshape(m, 2)
    cost = cost_matrix(keyframe_spans, segment_spans, similarity, semantic_weight=semantic_weight, time_scale=time_scale)

    result = dtw(cost, band_rows(keyframe_spans, segment_spans, band_seconds)) if band_seconds > 0 else None
    banded = result is not None
    path, total, cells = result if banded else dtw(cost)
    return Alignment(path, [round(float(cost[i, j]), 4) for i, j in path], float(total), cells, banded)


# ---------------------------------------------------------------- chunking


# A keyframe belongs to a chunk if it was on screen for at least this long during it.
MIN_OVERLAP_SECONDS = 2.0
MIN_OVERLAP_SHARE = 0.25  # ... or this share of the chunk, whichever is smaller

_SENTENCE_END = (".", "?", "!", "…", "。", "？", "！")


def _chars(segments: list[Segment]) -> int:
    return sum(len(s.text) + 1 for s in segments)


def _ends_sentence(text: str) -> bool:
    return text.rstrip().rstrip("\"'”’)]").endswith(_SENTENCE_END)


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _gap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, max(a_start, b_start) - min(a_end, b_end))


def _merge_tags(keyframes: list[Keyframe], limit: int = 5) -> list[str]:
    best: dict[str, float] = {}
    for keyframe in keyframes:
        for label, score in keyframe.tags:
            best[label] = max(score, best.get(label, 0.0))
    return [label for label, _ in sorted(best.items(), key=lambda item: -item[1])[:limit]]


def _group_segments(
    segments: list[Segment],
    primary: dict[int, int],
    *,
    max_chars: int,
    max_seconds: float,
    min_seconds: float,
    silence_gap: float,
) -> list[list[Segment]]:
    groups: list[list[Segment]] = []
    current: list[Segment] = []
    scene = None  # keyframe the current chunk started in
    for j, seg in enumerate(segments):
        if current:
            too_big = _chars(current) + len(seg.text) > max_chars or seg.end - current[0].start > max_seconds
            pause = seg.start - current[-1].end > silence_gap
            # Whisper segments often end mid-sentence, so a scene change waits for a sentence end.
            scene_cut = (
                primary.get(j) is not None
                and scene is not None
                and primary[j] != scene
                and current[-1].end - current[0].start >= min_seconds
                and _ends_sentence(current[-1].text)
            )
            if too_big or pause or scene_cut:
                groups.append(current)
                current = []
        if not current:
            scene = primary.get(j)
        current.append(seg)
    if current:
        groups.append(current)
    return groups


def _merge_small(groups: list[list[Segment]], min_chars: int, silence_gap: float) -> list[list[Segment]]:
    """Fold chunks with almost no text ("Okay.", "Right.") into a neighbour."""
    merged: list[list[Segment]] = []
    for group in groups:
        if merged and _chars(group) < min_chars and group[0].start - merged[-1][-1].end <= silence_gap:
            merged[-1] = merged[-1] + group
        else:
            merged.append(group)
    # A tiny first chunk has no previous neighbour, so it joins the next one.
    if len(merged) > 1 and _chars(merged[0]) < min_chars and merged[1][0].start - merged[0][-1].end <= silence_gap:
        merged[1] = merged[0] + merged[1]
        merged.pop(0)
    return merged


def build_chunks(
    alignment: Alignment,
    keyframes: list[Keyframe],
    segments: list[Segment],
    *,
    video_id: str,
    max_chars: int = 900,
    max_seconds: float = 45.0,
    min_chars: int = 60,
    min_seconds: float = 5.0,
    silence_gap: float = 8.0,
) -> list[Chunk]:
    """Turn the alignment into non-overlapping chunks.

    Walk the segments in order and close the current chunk when
      * adding the next segment would pass max_chars or max_seconds, or
      * there is a pause longer than silence_gap, or
      * the scene changed (the next segment's best-matching keyframe on the DTW path
        differs from the one the chunk started in), the chunk already lasts
        min_seconds, and the last segment ends a sentence.
    Each chunk keeps the keyframes DTW matched to its segments that were on screen
    for a meaningful part of it. Keyframes left without a chunk, such as scenes during
    a long silence, become visual-only chunks so they stay searchable through their tags.
    """
    matched: dict[int, list[int]] = {}  # segment index -> keyframe indices on the path
    primary: dict[int, int] = {}  # segment index -> its lowest-cost keyframe
    best_cost: dict[int, float] = {}
    for (i, j), cost in zip(alignment.path, alignment.path_costs):
        matched.setdefault(j, []).append(i)
        if cost < best_cost.get(j, math.inf):
            best_cost[j], primary[j] = cost, i

    groups = _group_segments(
        segments, primary, max_chars=max_chars, max_seconds=max_seconds, min_seconds=min_seconds, silence_gap=silence_gap
    )
    groups = _merge_small(groups, min_chars, silence_gap)

    position = {seg.id: j for j, seg in enumerate(segments)}
    chunks: list[Chunk] = []
    used: set[int] = set()
    for group in groups:
        start, end = group[0].start, group[-1].end
        candidates = sorted({i for seg in group for i in matched.get(position[seg.id], [])})
        needed = min(MIN_OVERLAP_SECONDS, MIN_OVERLAP_SHARE * (end - start))
        overlap = {i: _overlap(keyframes[i].span_start, keyframes[i].span_end, start, end) for i in candidates}
        near = [i for i in candidates if overlap[i] >= needed]
        if not near and candidates:  # always keep the closest picture
            near = [min(candidates, key=lambda i: _gap(keyframes[i].span_start, keyframes[i].span_end, start, end))]
        used.update(near)
        chunks.append(
            Chunk(
                id="",
                index=0,
                start=start,
                end=end,
                transcript=" ".join(seg.text for seg in group),
                tags=_merge_tags([keyframes[i] for i in near]),
                segment_ids=[seg.id for seg in group],
                keyframe_ids=[keyframes[i].id for i in near],
            )
        )

    for i, keyframe in enumerate(keyframes):
        if i not in used and keyframe.tags:
            chunks.append(
                Chunk(
                    id="",
                    index=0,
                    start=keyframe.span_start,
                    end=keyframe.span_end,
                    transcript="",
                    tags=_merge_tags([keyframe]),
                    keyframe_ids=[keyframe.id],
                )
            )

    chunks.sort(key=lambda chunk: (chunk.start, chunk.end))
    for index, chunk in enumerate(chunks):
        chunk.index = index
        chunk.id = f"{video_id}-{index:04d}"
    return chunks
