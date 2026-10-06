import numpy as np
import pytest

from vidsense.config import Settings
from vidsense.schemas import Chunk, VideoRecord
from vidsense.store import ChunkIndex, VideoLibrary


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path)


def _record(video_id, title, created_at, status="ready"):
    return VideoRecord(video_id=video_id, title=title, source_path="/x.mp4", duration=10.0, created_at=created_at, status=status)


def test_library_save_list_and_resolve(settings):
    library = VideoLibrary(settings)
    library.save(_record("aaaa1111", "First", "2026-01-01T00:00:00+00:00"))
    library.save(_record("aaaa2222", "Second", "2026-01-02T00:00:00+00:00"))
    library.save(_record("bbbb3333", "Broken", "2026-01-03T00:00:00+00:00", status="error"))

    assert [r.video_id for r in library.list()] == ["aaaa2222", "aaaa1111"]  # ready only, newest first
    assert len(library.list(status=None)) == 3
    assert library.resolve("latest").video_id == "aaaa2222"
    assert library.resolve("bbbb").title == "Broken"
    assert library.resolve("First").video_id == "aaaa1111"
    with pytest.raises(LookupError, match="matches 2"):
        library.resolve("aaaa")
    with pytest.raises(LookupError, match="no video"):
        library.resolve("zzzz")


def test_chunk_index_roundtrip(settings):
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(4, 384)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    chunks = [Chunk(id=f"vid-{i:04d}", index=i, start=i * 10.0, end=i * 10.0 + 9, transcript=f"chunk {i}") for i in range(4)]

    index = ChunkIndex(settings)
    index.replace("vid", chunks, vectors)
    assert index.count("vid") == 4
    results = index.query("vid", vectors[2], k=2)
    assert results[0][0] == "vid-0002" and results[0][1] == pytest.approx(1.0, abs=1e-4)
    assert len(index.query("vid", vectors[0], k=50)) == 4  # k larger than the collection

    index.replace("vid", chunks[:1], vectors[:1])  # re-indexing replaces, never appends
    assert index.count("vid") == 1
    index.delete("vid")
    index.delete("vid")  # deleting twice is fine
    assert index.count("vid") == 0
