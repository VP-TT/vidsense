"""The processing pipeline: a video file in, a searchable index out.

    decode ─┬─ audio ── Whisper ───────────────────────────────┐
            └─ frames ─ CLIP ─ temporal K-means ─ CLIP tags ───┴─ DTW ─ chunks ─ MiniLM ─ ChromaDB

The audio and vision branches don't need each other until alignment, so they run in
two threads: Whisper on the CPU (CTranslate2) and CLIP on the GPU when there is one.
"""

from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

from .align import align, build_chunks
from .config import ProcessingConfig, Settings
from .embed import embed_texts
from .keyframes import KeyframeChoice, coverage, select_keyframes
from .media import MediaInfo, Progress, file_digest, jpeg_bytes, probe, sample_frames, sampling_interval
from .schemas import Keyframe, VideoRecord
from .store import ChunkIndex, VideoLibrary
from .timeutil import to_vtt
from .transcribe import Transcript, transcribe

ProgressFn = Callable[[float, str, str], None]  # (overall fraction 0-1, stage, message)
Placement = Literal["reference", "copy", "move"]

# Rough share of processing time per stage, used to report one overall fraction.
STAGE_WEIGHTS = {"prepare": 0.03, "audio": 0.55, "vision": 0.25, "align": 0.05, "index": 0.12}


class _Tracker:
    """Combines progress from both branches into one number. Thread-safe."""

    def __init__(self, callback: ProgressFn | None):
        self.callback = callback
        self.done = dict.fromkeys(STAGE_WEIGHTS, 0.0)
        self.lock = threading.Lock()

    def update(self, stage: str, fraction: float, message: str = "") -> None:
        with self.lock:
            self.done[stage] = max(self.done[stage], min(1.0, fraction))
            if self.callback:
                self.callback(sum(STAGE_WEIGHTS[s] * f for s, f in self.done.items()), stage, message)

    def stage(self, name: str) -> Progress:
        return lambda fraction, message="": self.update(name, fraction, message)


@dataclass
class _VisionResult:
    times: np.ndarray
    embeddings: np.ndarray  # (n_frames, 512) CLIP, L2-normalised
    thumbnails: list[bytes]  # JPEG per sampled frame; only keyframes get written to disk
    choices: list[KeyframeChoice]
    tags: list[list[tuple[str, float]]]
    interval: float
    device: str = ""


def _run_audio(source: Path, cfg: ProcessingConfig, info: MediaInfo, progress: Progress) -> Transcript:
    if not info.has_audio:
        progress(1.0, "no audio stream")
        return Transcript([], None, 0.0, 0.0)
    progress(0.0, f"loading Whisper {cfg.whisper_model}")
    transcript = transcribe(source, cfg, progress)
    progress(1.0, f"{len(transcript.segments)} transcript segments")
    return transcript


def _run_vision(source: Path, cfg: ProcessingConfig, info: MediaInfo, progress: Progress) -> _VisionResult:
    empty = _VisionResult(np.zeros(0), np.zeros((0, 512), dtype=np.float32), [], [], [], 0.0)
    if not info.has_video:
        progress(1.0, "no video stream")
        return empty
    from .vision import load_clip, load_tagger

    progress(0.0, "loading CLIP")
    clip = load_clip(cfg.clip_model)
    interval = sampling_interval(info.duration, cfg.frame_fps, cfg.max_frames)
    times, thumbnails, batches, batch = [], [], [], []
    for frame in sample_frames(source, interval, duration=info.duration, progress=lambda f, m: progress(0.85 * f, m)):
        times.append(frame.time)
        thumbnails.append(jpeg_bytes(frame.image))
        batch.append(frame.image)
        if len(batch) == 64:  # embed as we decode, so only 64 full frames are in memory at once
            batches.append(clip.encode_images(batch))
            batch = []
    if batch:
        batches.append(clip.encode_images(batch))
    if not times:
        progress(1.0, "no decodable frames")
        return empty

    embeddings = np.concatenate(batches)
    frame_times = np.array(times)
    progress(0.9, f"clustering {len(times)} frames")
    choices = select_keyframes(
        frame_times,
        embeddings,
        frame_interval=interval,
        seconds_per_keyframe=cfg.seconds_per_keyframe,
        time_weight=cfg.time_weight,
        dedup_threshold=cfg.dedup_threshold,
    )
    tags: list[list[tuple[str, float]]] = [[] for _ in choices]
    if cfg.visual_tags and choices:
        tagger = load_tagger(cfg.clip_model, cfg.tag_vocab)
        tags = tagger.tag(embeddings[[c.frame_index for c in choices]], cfg.tag_top_k, cfg.tag_min_prob)
    progress(1.0, f"{len(choices)} keyframes from {len(times)} frames")
    return _VisionResult(frame_times, embeddings, thumbnails, choices, tags, interval, clip.device)


def process_video(
    source: str | Path,
    settings: Settings,
    *,
    title: str | None = None,
    processing: ProcessingConfig | None = None,
    placement: Placement = "reference",
    force: bool = False,
    progress: ProgressFn | None = None,
) -> VideoRecord:
    """Process a video and return its manifest. Re-processing the same file is a no-op unless force=True.

    placement: "reference" keeps the video where it is, "copy" copies it into the
    data folder, "move" moves it there (used for uploads).
    """
    cfg = processing or settings.processing
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    library = VideoLibrary(settings)
    tracker = _Tracker(progress)
    started = time.perf_counter()

    tracker.update("prepare", 0.1, "hashing file")
    video_id = file_digest(source)
    existing = library.get(video_id)
    if existing and existing.status == "ready" and not force and Path(existing.source_path).is_file():
        if placement == "move" and source != Path(existing.source_path):
            source.unlink(missing_ok=True)  # a duplicate upload
        for stage in STAGE_WEIGHTS:
            tracker.update(stage, 1.0, "already processed")
        return existing

    info = probe(source)
    if not (info.has_audio or info.has_video):
        raise ValueError(f"{source.name} has no audio or video stream")

    folder = library.dir(video_id)
    shutil.rmtree(folder / "keyframes", ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    title = title or source.stem
    if placement in ("copy", "move"):
        destination = folder / f"source{source.suffix.lower()}"
        if source != destination:
            (shutil.copy2 if placement == "copy" else shutil.move)(source, destination)
        source = destination

    record = VideoRecord(
        video_id=video_id,
        title=title,
        source_path=str(source),
        duration=round(info.duration, 3),
        width=info.width,
        height=info.height,
        fps=round(info.fps, 3),
        has_audio=info.has_audio,
        has_video=info.has_video,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        status="processing",
        processing=cfg.to_dict(),
    )
    library.save(record)
    tracker.update("prepare", 1.0, "starting audio and vision")
    try:
        _build_index(record, source, info, cfg, settings, library, tracker)
    except BaseException as exc:
        record.status, record.error = "error", f"{type(exc).__name__}: {exc}"
        library.save(record)
        raise
    record.stats["timings"]["total"] = round(time.perf_counter() - started, 2)
    record.status, record.error = "ready", None
    library.save(record)
    return record


def _build_index(
    record: VideoRecord,
    source: Path,
    info: MediaInfo,
    cfg: ProcessingConfig,
    settings: Settings,
    library: VideoLibrary,
    tracker: _Tracker,
) -> None:
    timings: dict[str, float] = {}

    def timed(name, fn, *args):
        start = time.perf_counter()
        try:
            return fn(*args)
        finally:
            timings[name] = round(time.perf_counter() - start, 2)

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="vidsense") as pool:
        audio_job = pool.submit(timed, "transcribe", _run_audio, source, cfg, info, tracker.stage("audio"))
        vision_job = pool.submit(timed, "vision", _run_vision, source, cfg, info, tracker.stage("vision"))
        transcript, vision = audio_job.result(), vision_job.result()

    segments = transcript.segments
    last_frame = float(vision.times[-1]) + vision.interval if len(vision.times) else 0.0
    duration = max(info.duration, transcript.duration, last_frame)

    folder = library.dir(record.video_id)
    (folder / "keyframes").mkdir(exist_ok=True)
    keyframes = []
    for i, (choice, tags) in enumerate(zip(vision.choices, vision.tags)):
        image = f"keyframes/kf_{i:04d}.jpg"
        (folder / image).write_bytes(vision.thumbnails[choice.frame_index])
        keyframes.append(
            Keyframe(
                id=i,
                time=round(float(vision.times[choice.frame_index]), 2),
                span_start=round(choice.span_start, 2),
                span_end=round(min(choice.span_end, duration), 2),
                frame_count=choice.frame_count,
                image=image,
                tags=tags,
            )
        )

    start = time.perf_counter()
    tracker.update("align", 0.1, "aligning keyframes with the transcript (DTW)")
    similarity = None
    if cfg.dtw_semantic_weight > 0 and keyframes and segments:
        from .vision import load_clip

        text_embeddings = load_clip(cfg.clip_model).encode_texts([s.text for s in segments])
        similarity = vision.embeddings[[c.frame_index for c in vision.choices]] @ text_embeddings.T
    alignment = align(
        np.array([[k.span_start, k.span_end] for k in keyframes], dtype=float).reshape(-1, 2),
        np.array([[s.start, s.end] for s in segments], dtype=float).reshape(-1, 2),
        similarity,
        semantic_weight=cfg.dtw_semantic_weight,
        time_scale=cfg.dtw_time_scale,
        band_seconds=cfg.dtw_band_seconds,
    )
    chunks = build_chunks(
        alignment,
        keyframes,
        segments,
        video_id=record.video_id,
        max_chars=cfg.max_chunk_chars,
        max_seconds=cfg.max_chunk_seconds,
        min_chars=cfg.min_chunk_chars,
        min_seconds=cfg.min_chunk_seconds,
        silence_gap=cfg.silence_gap,
    )
    timings["align"] = round(time.perf_counter() - start, 3)
    if not chunks:
        raise ValueError("Nothing to index: no speech was detected and no keyframe got a confident visual tag.")
    tracker.update("align", 1.0, f"{len(chunks)} chunks")

    start = time.perf_counter()
    tracker.update("index", 0.1, f"embedding {len(chunks)} chunks")
    vectors = embed_texts([c.embedding_text() for c in chunks], cfg.embed_model)
    ChunkIndex(settings).replace(record.video_id, chunks, vectors)
    timings["index"] = round(time.perf_counter() - start, 2)

    library.write_json(
        record.video_id,
        "transcript.json",
        {
            "language": transcript.language,
            "language_probability": round(transcript.language_probability, 3),
            "segments": [asdict(s) for s in segments],
        },
    )
    library.path(record.video_id, "transcript.vtt").write_text(to_vtt(segments), encoding="utf-8")
    library.write_json(record.video_id, "keyframes.json", keyframes)
    library.write_json(record.video_id, "chunks.json", chunks)
    library.write_json(
        record.video_id,
        "alignment.json",
        {
            "path": alignment.path,
            "path_costs": alignment.path_costs,
            "total_cost": round(alignment.total_cost, 4),
            "cells": alignment.cells,
            "full_cells": len(keyframes) * len(segments),
            "banded": alignment.banded,
        },
    )
    tracker.update("index", 1.0, "done")

    record.duration = round(duration, 3)
    record.language = transcript.language
    key_indices = [c.frame_index for c in vision.choices]
    record.stats = {
        "segments": len(segments),
        "words": sum(len(s.text.split()) for s in segments),
        "sampled_frames": int(len(vision.times)),
        "frame_interval": round(vision.interval, 3),
        "keyframes": len(keyframes),
        "keyframe_coverage": coverage(vision.embeddings, key_indices) if key_indices else {},
        "chunks": len(chunks),
        "visual_only_chunks": sum(1 for c in chunks if not c.transcript),
        "dtw": {"cells": alignment.cells, "full_cells": len(keyframes) * len(segments), "banded": alignment.banded},
        "language_probability": round(transcript.language_probability, 3),
        "devices": {"whisper": _whisper_device(), "clip": vision.device or "-"},
        "timings": timings,
    }


def _whisper_device() -> str:
    import ctranslate2

    return "cuda (float16)" if ctranslate2.get_cuda_device_count() > 0 else "cpu (int8)"
