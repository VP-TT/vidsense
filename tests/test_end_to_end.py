"""End-to-end: generate a short narrated video, process it with real models, search it.

Slow (downloads Whisper tiny, CLIP and MiniLM on first run), so it only runs with
`pytest -m slow`. Needs text-to-speech: `say` on macOS or espeak-ng on Linux.
"""

import shutil

import pytest

from vidsense.config import AnswerConfig, ProcessingConfig, Settings
from vidsense.demo import QUESTIONS, SCENES, make_demo
from vidsense.evaluate import evaluate, load_jsonl
from vidsense.pipeline import process_video
from vidsense.retrieval import search
from vidsense.store import VideoLibrary

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    if not (shutil.which("say") or shutil.which("espeak-ng") or shutil.which("espeak")):
        pytest.skip("no text-to-speech available")
    root = tmp_path_factory.mktemp("e2e")
    scenes = [s for s in SCENES if s.key in ("ocean", "forest", "desert")]
    video, qa = make_demo(root / "demo", scenes=scenes, questions=[q for q in QUESTIONS if q[2] in ("ocean", "forest", "desert")])
    settings = Settings(data_dir=root / "data", processing=ProcessingConfig(whisper_model="tiny"), answer=AnswerConfig(provider="none"))
    return settings, process_video(video, settings), qa


def test_pipeline_builds_an_aligned_index(processed):
    settings, record, _ = processed
    assert record.status == "ready" and record.language == "en"
    assert record.stats["keyframes"] >= 3 and record.stats["chunks"] >= 3
    library = VideoLibrary(settings)
    assert all(c.keyframe_ids for c in library.chunks(record.video_id) if c.transcript)
    assert library.path(record.video_id, "transcript.vtt").read_text().startswith("WEBVTT")


def test_search_finds_the_right_scene(processed):
    settings, record, _ = processed
    best = search(settings, record.video_id, "how old are the trees", k=1)[0].chunk
    assert "forest" in best.transcript.lower()
    best = search(settings, record.video_id, "sand dunes", k=1)[0].chunk
    assert "desert" in best.transcript.lower() or any("desert" in t for t in best.tags)


def test_reprocessing_the_same_file_is_instant(processed):
    settings, record, _ = processed
    again = process_video(record.source_path, settings)
    assert again.video_id == record.video_id and again.created_at == record.created_at


def test_evaluation_on_the_generated_questions(processed):
    settings, _, qa = processed
    summary, rows = evaluate(settings, load_jsonl(qa), k=3)
    assert summary["retrieval"]["hit_at_1"] >= 0.75
    assert summary["latency"]["retrieval_ms"]["p50"] < 1000
