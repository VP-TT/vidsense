"""Plain records shared by the pipeline, the vector store and the UI.

Everything here round-trips through JSON, so a processed video is just a folder of
readable files next to its Chroma collection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Segment:
    """One Whisper transcript segment, usually a sentence or a clause."""

    id: int
    start: float
    end: float
    text: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Segment:
        return cls(id=int(d["id"]), start=float(d["start"]), end=float(d["end"]), text=str(d["text"]))


@dataclass
class Keyframe:
    """A representative frame picked by temporal K-means over CLIP embeddings."""

    id: int
    time: float  # timestamp of the chosen frame, in seconds
    span_start: float  # time range of the sampled frames this keyframe stands for
    span_end: float
    frame_count: int  # number of sampled frames it stands for
    image: str  # thumbnail path, relative to the video's folder
    tags: list[tuple[str, float]] = field(default_factory=list)  # zero-shot CLIP labels, best first

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Keyframe:
        return cls(
            id=int(d["id"]),
            time=float(d["time"]),
            span_start=float(d["span_start"]),
            span_end=float(d["span_end"]),
            frame_count=int(d["frame_count"]),
            image=str(d["image"]),
            tags=[(str(label), float(score)) for label, score in d.get("tags", [])],
        )


@dataclass
class Chunk:
    """The unit that gets embedded and retrieved.

    A chunk is a run of consecutive transcript segments plus the keyframes that DTW
    aligned to them. Chunks never overlap, so every segment lives in exactly one.
    """

    id: str
    index: int
    start: float
    end: float
    transcript: str
    tags: list[str] = field(default_factory=list)
    segment_ids: list[int] = field(default_factory=list)
    keyframe_ids: list[int] = field(default_factory=list)

    def embedding_text(self) -> str:
        """Text that goes into the 384-d MiniLM space: what was said plus what was seen."""
        parts = [self.transcript] if self.transcript else []
        if self.tags:
            parts.append("On screen: " + "; ".join(self.tags))
        return "\n".join(parts)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(
            id=str(d["id"]),
            index=int(d["index"]),
            start=float(d["start"]),
            end=float(d["end"]),
            transcript=str(d["transcript"]),
            tags=list(d.get("tags", [])),
            segment_ids=[int(i) for i in d.get("segment_ids", [])],
            keyframe_ids=[int(i) for i in d.get("keyframe_ids", [])],
        )


@dataclass
class VideoRecord:
    """The manifest of a processed video (``manifest.json``)."""

    video_id: str
    title: str
    source_path: str  # playable file, absolute path
    duration: float
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_audio: bool = True
    has_video: bool = True
    language: str | None = None
    created_at: str = ""
    status: str = "processing"  # "processing", "ready" or "error"
    error: str | None = None
    processing: dict[str, Any] = field(default_factory=dict)  # ProcessingConfig used
    stats: dict[str, Any] = field(default_factory=dict)  # counts, timings, keyframe coverage

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VideoRecord:
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Hit:
    """One retrieval result."""

    chunk: Chunk
    score: float  # cosine similarity between query and chunk in the MiniLM space
    rank: int
