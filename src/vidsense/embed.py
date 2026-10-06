"""Text embeddings with a Sentence Transformer (all-MiniLM-L6-v2, 384 dimensions).

This is the only vector space VidSense stores. Chunks and questions are both embedded
here, so retrieval compares two vectors from the same model; the 512-d CLIP space is
never searched.
"""

from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

from .hub import load_local_first, quiet_transformers

_load_lock = threading.Lock()


@lru_cache(maxsize=2)
def _load(model_name: str):
    import torch
    from sentence_transformers import SentenceTransformer

    # MiniLM is small enough that the CPU is fast, and keeping it off the Apple GPU
    # means a search from the UI never competes with CLIP running in a processing job.
    quiet_transformers()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_local_first(SentenceTransformer, model_name, device=device)


def load_embedder(model_name: str):
    with _load_lock:
        return _load(model_name)


def embed_texts(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    """L2-normalised (n, 384) float32 embeddings, so cosine similarity is a dot product."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = load_embedder(model_name)
    vectors = model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    return np.asarray(vectors, dtype=np.float32)
