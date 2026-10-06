import itertools
import math

import numpy as np
import pytest

from vidsense.align import Alignment, align, band_rows, build_chunks, cost_matrix, dtw
from vidsense.schemas import Keyframe, Segment


def _brute_force(cost):
    """Minimum DTW cost by enumerating every monotonic path (tiny matrices only)."""
    n, m = cost.shape
    best = math.inf

    def walk(i, j, total):
        nonlocal best
        total += cost[i, j]
        if (i, j) == (n - 1, m - 1):
            best = min(best, total)
            return
        for di, dj in ((1, 0), (0, 1), (1, 1)):
            if i + di < n and j + dj < m:
                walk(i + di, j + dj, total)

    walk(0, 0, 0.0)
    return best


@pytest.mark.parametrize("seed", range(8))
def test_dtw_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    cost = rng.random((rng.integers(1, 5), rng.integers(1, 6)))
    path, total, _ = dtw(cost)
    assert total == pytest.approx(_brute_force(cost))
    assert sum(cost[i, j] for i, j in path) == pytest.approx(total)


def _check_path(path, n, m):
    assert path[0] == (0, 0) and path[-1] == (n - 1, m - 1)
    for (i1, j1), (i2, j2) in itertools.pairwise(path):
        assert (i2 - i1, j2 - j1) in {(1, 0), (0, 1), (1, 1)}
    assert {i for i, _ in path} == set(range(n))  # every keyframe matched
    assert {j for _, j in path} == set(range(m))  # every segment matched


def test_time_only_alignment_follows_timestamps():
    keyframes = np.array([[0, 10], [10, 20], [20, 30]], dtype=float)
    segments = np.array([[1, 4], [5, 9], [11, 18], [21, 25], [26, 29]], dtype=float)
    result = align(keyframes, segments, semantic_weight=0.0, band_seconds=60)
    _check_path(result.path, 3, 5)
    assert result.path == [(0, 0), (0, 1), (1, 2), (2, 3), (2, 4)]
    assert result.total_cost == 0 and result.banded


def test_semantic_term_moves_a_boundary_sentence():
    # Segment 1 (8-11s) straddles the cut at 10s; CLIP says it describes keyframe 1.
    keyframes = np.array([[0, 10], [10, 20]], dtype=float)
    segments = np.array([[0, 7], [8, 11], [12, 19]], dtype=float)
    similarity = np.array([[0.30, 0.10, 0.10], [0.10, 0.30, 0.30]])
    assert align(keyframes, segments, semantic_weight=0.0).path[1] == (0, 1)
    assert align(keyframes, segments, similarity, semantic_weight=0.5).path[1] == (1, 1)


def test_band_limits_work_and_falls_back_when_disconnected():
    keyframes = np.array([[i * 10, i * 10 + 10] for i in range(30)], dtype=float)
    segments = np.array([[j * 3, j * 3 + 2.5] for j in range(100)], dtype=float)
    banded = align(keyframes, segments, band_seconds=15)
    full = align(keyframes, segments, band_seconds=0)
    assert banded.banded and not full.banded
    assert banded.cells < full.cells == 30 * 100
    assert banded.total_cost == pytest.approx(full.total_cost)

    # A 200 s silence between keyframe 0 and the only segment cannot be bridged by a 30 s band.
    far = align(np.array([[0, 5], [300, 305]], dtype=float), np.array([[301, 303]], dtype=float), band_seconds=30)
    assert not far.banded
    _check_path(far.path, 2, 1)


def test_band_rows_are_windows():
    rows = band_rows(np.array([[50, 60]], dtype=float), np.array([[j * 10, j * 10 + 5] for j in range(12)], dtype=float), 20)
    assert rows == [(3, 7)]  # segment midpoints between 30s and 80s


def test_cost_matrix_uses_segment_midpoints():
    spans = np.array([[10, 20]], dtype=float)
    cost = cost_matrix(spans, np.array([[15, 25], [30, 40], [8, 11]], dtype=float), time_scale=10)
    assert cost.tolist() == [[0.0, 1.5, 0.05]]  # 8-11s has its middle 0.5s before the span


def test_sentence_on_a_cut_goes_to_the_scene_holding_its_middle():
    keyframes = np.array([[0, 10], [10, 20]], dtype=float)
    segments = np.array([[2, 9], [9, 15], [15, 19]], dtype=float)  # middle one is mostly after the cut
    assert align(keyframes, segments, semantic_weight=0.0).path == [(0, 0), (1, 1), (1, 2)]


def test_empty_inputs():
    assert align(np.zeros((0, 2)), np.zeros((3, 2))).path == []
    assert align(np.zeros((2, 2)), np.zeros((0, 2))).path == []


# ---------------------------------------------------------------- chunking


def _keyframe(i, start, end, tags=()):
    return Keyframe(id=i, time=start, span_start=start, span_end=end, frame_count=1, image="", tags=list(tags))


def _segments(spans_and_text):
    return [Segment(id=j, start=s, end=e, text=t) for j, (s, e, t) in enumerate(spans_and_text)]


def _chunks(keyframes, segments, **kwargs):
    alignment = align(
        np.array([[k.span_start, k.span_end] for k in keyframes], dtype=float).reshape(-1, 2),
        np.array([[s.start, s.end] for s in segments], dtype=float).reshape(-1, 2),
        semantic_weight=0.0,
    )
    return build_chunks(alignment, keyframes, segments, video_id="vid", **kwargs)


def test_chunks_follow_scene_changes_and_partition_segments():
    keyframes = [_keyframe(0, 0, 10, [("a forest", 0.9)]), _keyframe(1, 10, 20, [("a city street", 0.8)])]
    segments = _segments([(0, 4, "Trees grow tall here."), (4, 9, "The forest is very quiet."), (10, 15, "Now the city."), (15, 19, "Traffic is slow.")])
    chunks = _chunks(keyframes, segments, min_chars=10)
    assert [c.segment_ids for c in chunks] == [[0, 1], [2, 3]]
    assert [c.keyframe_ids for c in chunks] == [[0], [1]]
    assert chunks[0].tags == ["a forest"] and chunks[1].tags == ["a city street"]
    assert [c.id for c in chunks] == ["vid-0000", "vid-0001"]
    assert chunks[0].embedding_text() == "Trees grow tall here. The forest is very quiet.\nOn screen: a forest"


def test_scene_change_mid_sentence_waits_for_the_sentence_end():
    keyframes = [_keyframe(0, 0, 8, [("a title card", 0.3)]), _keyframe(1, 8, 17, [("a sailboat", 0.7)])]
    segments = _segments([(0.4, 5.7, "In about a minute we will visit five very different places"), (5.7, 8.9, "around the world."), (8.9, 14, "Our first stop is the ocean."), (14, 16, "A boat drifts by.")])
    alignment = align(
        np.array([[0, 8], [8, 17]], dtype=float),
        np.array([[s.start, s.end] for s in segments], dtype=float),
        np.array([[0.20, 0.20, 0.10, 0.10], [0.10, 0.25, 0.30, 0.30]]),  # CLIP prefers the ocean for "around the world."
        semantic_weight=0.5,
    )
    assert alignment.path[1] == (1, 1)  # DTW moved the sentence tail to the ocean keyframe...
    chunks = build_chunks(alignment, keyframes, segments, video_id="v", min_chars=10, min_seconds=5)
    assert [c.segment_ids for c in chunks] == [[0, 1], [2, 3]]  # ...but the chunk keeps the whole sentence
    assert chunks[0].tags == ["a title card"]  # the ocean was on screen for <2s of the first chunk


def test_size_limit_splits_long_scenes():
    keyframes = [_keyframe(0, 0, 300)]
    segments = _segments([(i * 10, i * 10 + 9, f"Sentence number {i} about one long scene.") for i in range(30)])
    chunks = _chunks(keyframes, segments, max_chars=200, max_seconds=45, min_chars=10)
    assert len(chunks) > 1
    assert all(len(c.transcript) <= 200 and c.end - c.start <= 45 for c in chunks)
    assert sorted(j for c in chunks for j in c.segment_ids) == list(range(30))
    assert all(c.keyframe_ids == [0] for c in chunks)


def test_tiny_chunks_merge_into_neighbours():
    keyframes = [_keyframe(0, 0, 9), _keyframe(1, 9, 20)]
    segments = _segments([(0, 8.5, "This is the first full sentence of the video."), (9.5, 10, "Okay."), (10.5, 19, "And this is another full sentence.")])
    chunks = _chunks(keyframes, segments, min_chars=20, min_seconds=5)
    assert all(len(c.transcript) >= 20 for c in chunks)
    assert sorted(j for c in chunks for j in c.segment_ids) == [0, 1, 2]


def test_silent_scene_becomes_visual_only_chunk():
    keyframes = [_keyframe(0, 0, 10), _keyframe(1, 30, 60, [("fireworks", 0.7)]), _keyframe(2, 80, 90)]
    segments = _segments([(1, 9, "Welcome to the show, everyone."), (81, 89, "That was the fireworks finale.")])
    chunks = _chunks(keyframes, segments, min_chars=5)
    visual = [c for c in chunks if not c.transcript]
    assert len(visual) == 1 and visual[0].tags == ["fireworks"] and (visual[0].start, visual[0].end) == (30, 60)
    assert [c.start for c in chunks] == sorted(c.start for c in chunks)


def test_no_speech_or_no_video():
    keyframes = [_keyframe(0, 0, 10, [("a dog", 0.9)]), _keyframe(1, 10, 20)]
    only_visual = build_chunks(Alignment([], [], 0.0, 0, False), keyframes, [], video_id="v")
    assert [c.tags for c in only_visual] == [["a dog"]]  # untagged keyframes have nothing to index

    segments = _segments([(0, 5, "Just audio here, no pictures at all.")])
    only_audio = build_chunks(Alignment([], [], 0.0, 0, False), [], segments, video_id="v")
    assert len(only_audio) == 1 and only_audio[0].keyframe_ids == []
