"""Runtime settings.

Every field below has a default and can be overridden with an environment variable
(or a ``.env`` file in the working directory). Processing fields map to
``VIDSENSE_<FIELD>`` and answer fields to ``VIDSENSE_LLM_<FIELD>``, for example
``VIDSENSE_WHISPER_MODEL=small`` or ``VIDSENSE_LLM_PROVIDER=openai``. The CLI flags
and the Streamlit sidebar override them per run with ``dataclasses.replace``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

PROVIDERS = ("ollama", "openai", "none")
DEFAULT_MODELS = {"ollama": "deepseek-r1:8b", "openai": "gpt-4o-mini", "none": ""}


@dataclass(frozen=True)
class ProcessingConfig:
    """Settings that change what gets indexed. They are saved with every video."""

    # Audio: Whisper via faster-whisper (CTranslate2).
    whisper_model: str = "large-v3-turbo"
    language: str | None = None  # None means auto-detect
    translate: bool = False  # Whisper's "translate" task: any language -> English text
    beam_size: int = 5
    vad_filter: bool = True  # skip silence; cuts hallucinated text in music and pauses

    # Vision: CLIP frame embeddings, then temporal K-means for keyframes.
    frame_fps: float = 1.0  # frames sampled per second of video
    max_frames: int = 1800  # long videos are sampled more sparsely to stay under this
    clip_model: str = "openai/clip-vit-base-patch32"
    seconds_per_keyframe: float = 5.0  # K = duration / this; over-cluster, then merge near-duplicates
    time_weight: float = 0.3  # how strongly K-means keeps clusters contiguous in time
    dedup_threshold: float = 0.95  # merge neighbouring keyframes above this cosine similarity
    visual_tags: bool = True  # zero-shot CLIP labels for each keyframe
    tag_top_k: int = 3
    tag_min_prob: float = 0.10
    tag_vocab: str | None = None  # path to a custom label list, one label per line

    # Alignment: banded DTW between keyframes and transcript segments.
    dtw_semantic_weight: float = 0.3  # 0 = timestamps only; >0 mixes in CLIP image-text similarity
    dtw_time_scale: float = 10.0  # seconds of time gap that cost as much as a full semantic mismatch
    dtw_band_seconds: float = 60.0  # Sakoe-Chiba style window; pairs further apart are never matched

    # Chunking and text embeddings.
    max_chunk_chars: int = 900
    max_chunk_seconds: float = 45.0
    min_chunk_chars: int = 60  # smaller chunks are folded into a neighbour
    min_chunk_seconds: float = 5.0  # a scene change only splits a chunk at least this long
    silence_gap: float = 8.0  # a pause this long always starts a new chunk
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnswerConfig:
    """Settings for the answer step. They don't affect the index."""

    provider: str = "ollama"  # "ollama", "openai" or "none" (retrieval only)
    model: str = ""  # empty means the provider default in DEFAULT_MODELS
    temperature: float = 0.3
    top_k: int = 5
    use_history: bool = True  # rewrite follow-up questions into standalone ones
    ollama_url: str = "http://localhost:11434"
    num_ctx: int = 8192  # Ollama context window; its own default is too small for summaries

    @property
    def resolved_model(self) -> str:
        return self.model or DEFAULT_MODELS.get(self.provider, "")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"model": self.resolved_model}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    answer: AnswerConfig = field(default_factory=AnswerConfig)

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"


def _coerce(raw: str, default: Any) -> Any:
    raw = raw.strip()
    if isinstance(default, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if default is None and raw.lower() in {"", "none", "auto"}:
        return None
    return raw


def _from_env(cls: type, prefix: str) -> Any:
    values = {}
    for f in fields(cls):
        raw = os.environ.get(f"VIDSENSE_{prefix}{f.name.upper()}")
        if raw is not None and raw.strip() != "":
            values[f.name] = _coerce(raw, f.default)
    return cls(**values)


def load_settings(env_file: str | Path = ".env") -> Settings:
    """Build settings from defaults, a .env file (if present) and the environment."""
    if Path(env_file).is_file():
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)

    answer = _from_env(AnswerConfig, "LLM_")
    if answer.provider not in PROVIDERS:
        raise ValueError(f"VIDSENSE_LLM_PROVIDER must be one of {PROVIDERS}, got {answer.provider!r}")

    data_dir = Path(os.environ.get("VIDSENSE_DATA_DIR", "data")).expanduser().resolve()
    return Settings(data_dir=data_dir, processing=_from_env(ProcessingConfig, ""), answer=answer)
