import numpy as np
import pytest

from vidsense.keyframes import KeyframeChoice, choose_k, coverage, merge_duplicates, select_keyframes


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _scenes(layout, seconds_per_scene=20, dim=64, noise=0.02, seed=0):
    """Frames at 1 fps; `layout` names the scene shown in each block, e.g. "ABAC"."""
    rng = np.random.default_rng(seed)
    bases = {name: _unit(rng.normal(size=dim)) for name in set(layout)}
    embeddings = np.vstack(
        [_unit(bases[name] + noise * rng.normal(size=(seconds_per_scene, dim))) for name in layout]
    )
    times = np.arange(len(embeddings), dtype=float)
    return times, embeddings.astype(np.float32)


def test_one_keyframe_per_scene_and_recurring_scene_kept_separate():
    times, embeddings = _scenes("ABAC")
    keyframes = select_keyframes(times, embeddings, frame_interval=1.0, seconds_per_keyframe=5.0)
    kf_times = [times[k.frame_index] for k in keyframes]

    assert len(keyframes) == 4
    for block, t in enumerate(kf_times):
        assert block * 20 <= t < (block + 1) * 20  # one keyframe inside each block, in order
    # Scene A comes back at 40-59s; time-aware clustering gives it its own keyframe.
    assert float(embeddings[keyframes[0].frame_index] @ embeddings[keyframes[2].frame_index]) > 0.95


def test_static_video_collapses_to_one_keyframe():
    times, embeddings = _scenes("A", seconds_per_scene=120)
    keyframes = select_keyframes(times, embeddings, frame_interval=1.0, seconds_per_keyframe=5.0)
    assert len(keyframes) == 1
    assert keyframes[0].span_start == 0 and keyframes[0].span_end == 120
    assert keyframes[0].frame_count == 120


def test_spans_cover_every_sampled_frame():
    times, embeddings = _scenes("ABCDE", seconds_per_scene=12)
    keyframes = select_keyframes(times, embeddings, frame_interval=1.0, seconds_per_keyframe=5.0)
    assert sum(k.frame_count for k in keyframes) == len(times)
    assert [k.frame_index for k in keyframes] == sorted(k.frame_index for k in keyframes)


def test_edge_cases():
    assert select_keyframes(np.array([]), np.zeros((0, 8)), frame_interval=1.0) == []
    one = select_keyframes(np.array([0.0]), _unit(np.ones((1, 8))), frame_interval=1.0)
    assert len(one) == 1 and one[0].frame_index == 0


@pytest.mark.parametrize(
    ("duration", "n_frames", "spk", "expected"),
    [(60, 60, 5, 12), (60, 5, 5, 5), (2, 2, 5, 1), (36000, 1800, 5, 300)],
)
def test_choose_k(duration, n_frames, spk, expected):
    assert choose_k(duration, n_frames, spk) == expected


def test_merge_keeps_the_more_representative_frame():
    embeddings = _unit(np.array([[1.0, 0.0], [1.0, 0.01], [0.0, 1.0]]))
    choices = [KeyframeChoice(0, 0, 5, 2), KeyframeChoice(1, 5, 20, 9), KeyframeChoice(2, 20, 25, 3)]
    merged = merge_duplicates(choices, embeddings, threshold=0.95)
    assert [(c.frame_index, c.span_start, c.span_end, c.frame_count) for c in merged] == [(1, 0, 20, 11), (2, 20, 25, 3)]


def test_coverage():
    times, embeddings = _scenes("AB")
    assert coverage(embeddings, [0, 20])["coverage"] == 1.0
    partial = coverage(embeddings, [0])
    assert partial["coverage"] == pytest.approx(0.5)
    assert partial["compression"] == pytest.approx(1 / 40)
