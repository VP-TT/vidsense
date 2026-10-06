"""Semantic search over one video's chunks."""

from __future__ import annotations

from .config import Settings
from .embed import embed_texts
from .schemas import Hit
from .store import ChunkIndex, VideoLibrary


def search(settings: Settings, video_id: str, query: str, k: int = 5) -> list[Hit]:
    """Embed the query with the same MiniLM model used at index time and return the top-k chunks."""
    library = VideoLibrary(settings)
    record = library.get(video_id)
    if record is None or record.status != "ready":
        raise LookupError(f"video {video_id} is not processed")
    model = record.processing.get("embed_model", settings.processing.embed_model)
    query_vector = embed_texts([query], model)[0]
    chunks = {chunk.id: chunk for chunk in library.chunks(video_id)}
    results = ChunkIndex(settings).query(video_id, query_vector, k)
    return [
        Hit(chunk=chunks[chunk_id], score=round(score, 4), rank=rank)
        for rank, (chunk_id, score) in enumerate(results, start=1)
        if chunk_id in chunks
    ]
