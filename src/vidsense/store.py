"""Storage: one folder per video for readable artifacts, one ChromaDB collection per video.

    data/
      videos/<video_id>/
        manifest.json        VideoRecord: title, duration, settings used, stats
        source.mp4           the playable file (uploads; CLI runs can reference the original)
        transcript.json      Whisper segments          transcript.vtt   captions for the player
        keyframes.json       keyframes + CLIP tags      keyframes/        thumbnails
        chunks.json          what gets embedded        alignment.json    the DTW path
      chroma/                ChromaDB, collection "vidsense_<video_id>"

A collection per video works as a namespace: deleting or re-indexing one video never
touches another, and a query can only ever return chunks from the video being asked
about. With several users, the same idea extends to one namespace per user.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings
from .schemas import Chunk, Keyframe, Segment, VideoRecord


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)  # readers never see a half-written file


@lru_cache(maxsize=64)
def _read_json_cached(path: str, mtime_ns: int) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_json(path: Path) -> Any:
    return _read_json_cached(str(path), path.stat().st_mtime_ns)


class VideoLibrary:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.videos_dir

    def dir(self, video_id: str) -> Path:
        return self.root / video_id

    def path(self, video_id: str, name: str) -> Path:
        return self.dir(video_id) / name

    # ---------------------------------------------------------------- manifests

    def get(self, video_id: str) -> VideoRecord | None:
        manifest = self.path(video_id, "manifest.json")
        if not manifest.is_file():
            return None
        return VideoRecord.from_dict(_read_json(manifest))

    def save(self, record: VideoRecord) -> None:
        _write_json_atomic(self.path(record.video_id, "manifest.json"), record.to_dict())

    def list(self, status: str | None = "ready") -> list[VideoRecord]:
        if not self.root.is_dir():
            return []
        records = [r for d in self.root.iterdir() if d.is_dir() and (r := self.get(d.name))]
        if status:
            records = [r for r in records if r.status == status]
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    def resolve(self, ref: str) -> VideoRecord:
        """Find a video by id, unique id prefix, exact title, or 'latest'."""
        records = self.list(status=None)
        if ref == "latest":
            ready = [r for r in records if r.status == "ready"]
            if ready:
                return ready[0]
            raise LookupError("no processed videos yet")
        matches = [r for r in records if r.video_id.startswith(ref)] or [r for r in records if r.title == ref]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise LookupError(f"no video matches {ref!r}; run `vidsense list`")
        raise LookupError(f"{ref!r} matches {len(matches)} videos; use more of the id")

    def delete(self, video_id: str) -> None:
        ChunkIndex(self.settings).delete(video_id)
        shutil.rmtree(self.dir(video_id), ignore_errors=True)

    # ---------------------------------------------------------------- artifacts

    def read_json(self, video_id: str, name: str, default: Any = None) -> Any:
        path = self.path(video_id, name)
        return _read_json(path) if path.is_file() else default

    def write_json(self, video_id: str, name: str, data: Any) -> None:
        if isinstance(data, list):
            data = [asdict(item) if is_dataclass(item) else item for item in data]
        _write_json_atomic(self.path(video_id, name), data)

    def segments(self, video_id: str) -> list[Segment]:
        data = _read_json(self.path(video_id, "transcript.json"))
        return [Segment.from_dict(s) for s in data["segments"]]

    def keyframes(self, video_id: str) -> list[Keyframe]:
        return [Keyframe.from_dict(k) for k in _read_json(self.path(video_id, "keyframes.json"))]

    def chunks(self, video_id: str) -> list[Chunk]:
        return [Chunk.from_dict(c) for c in _read_json(self.path(video_id, "chunks.json"))]

    def alignment(self, video_id: str) -> dict[str, Any]:
        return _read_json(self.path(video_id, "alignment.json"))


@lru_cache(maxsize=4)
def _client(path: str):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    return chromadb.PersistentClient(path=path, settings=ChromaSettings(anonymized_telemetry=False))


class ChunkIndex:
    """Chunk embeddings in ChromaDB (cosine distance, HNSW index)."""

    def __init__(self, settings: Settings):
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = _client(str(settings.chroma_dir))

    @staticmethod
    def collection_name(video_id: str) -> str:
        return f"vidsense_{video_id}"

    def replace(self, video_id: str, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self.delete(video_id)
        collection = self.client.create_collection(
            self.collection_name(video_id),
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"video_id": video_id},
            embedding_function=None,  # we always pass our own MiniLM vectors
        )
        if chunks:
            collection.add(
                ids=[c.id for c in chunks],
                embeddings=embeddings,
                documents=[c.embedding_text() for c in chunks],
                metadatas=[{"start": c.start, "end": c.end, "index": c.index} for c in chunks],
            )

    def query(self, video_id: str, embedding: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Top-k (chunk id, cosine similarity) pairs, best first."""
        collection = self.client.get_collection(self.collection_name(video_id), embedding_function=None)
        result = collection.query(query_embeddings=[embedding], n_results=max(1, k), include=["distances"])
        return [(cid, 1.0 - float(distance)) for cid, distance in zip(result["ids"][0], result["distances"][0])]

    def count(self, video_id: str) -> int:
        try:
            return self.client.get_collection(self.collection_name(video_id), embedding_function=None).count()
        except Exception:
            return 0

    def delete(self, video_id: str) -> None:
        from chromadb.errors import NotFoundError

        try:
            self.client.delete_collection(self.collection_name(video_id))
        except NotFoundError:
            pass
