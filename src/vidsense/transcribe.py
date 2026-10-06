"""Speech to text with Whisper, using the faster-whisper (CTranslate2) implementation.

faster-whisper runs the same Whisper weights roughly 4x faster than the reference
implementation and returns segment timestamps, which is all VidSense needs: the
segments become the text side of the DTW alignment.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import ProcessingConfig
from .hub import load_local_first
from .media import Progress, load_audio
from .schemas import Segment
from .timeutil import format_ts

log = logging.getLogger(__name__)
_load_lock = threading.Lock()
SAMPLE_RATE = 16000  # Whisper's input rate


@dataclass
class Transcript:
    segments: list[Segment]
    language: str | None
    language_probability: float
    duration: float  # audio duration in seconds, as Whisper measured it


@lru_cache(maxsize=2)
def _load(model_name: str):
    import ctranslate2
    from faster_whisper import WhisperModel

    if ctranslate2.get_cuda_device_count() > 0:
        return load_local_first(WhisperModel, model_name, device="cuda", compute_type="float16")
    # CTranslate2 has no Apple GPU backend, so on a Mac this runs on the CPU with int8 weights.
    threads = max(4, (os.cpu_count() or 8) // 2 + 1)
    return load_local_first(WhisperModel, model_name, device="cpu", compute_type="int8", cpu_threads=threads)


def load_whisper(model_name: str):
    with _load_lock:
        return _load(model_name)


def transcribe(path: str | Path, cfg: ProcessingConfig, progress: Progress | None = None) -> Transcript:
    model = load_whisper(cfg.whisper_model)
    task = "translate" if cfg.translate else "transcribe"
    if cfg.translate and "turbo" in cfg.whisper_model:
        log.warning("%s was not trained for translation; use large-v3 or medium for --translate", cfg.whisper_model)

    audio = load_audio(path, SAMPLE_RATE)
    if audio.size == 0:
        return Transcript([], None, 0.0, 0.0)
    segments_iter, info = model.transcribe(
        audio,
        language=cfg.language,
        task=task,
        beam_size=cfg.beam_size,
        vad_filter=cfg.vad_filter,
    )
    segments: list[Segment] = []
    total = audio.size / SAMPLE_RATE
    for seg in segments_iter:  # a generator: decoding happens while we iterate
        text = seg.text.strip()
        if text:
            segments.append(Segment(id=len(segments), start=round(seg.start, 2), end=round(seg.end, 2), text=text))
        if progress and total:
            progress(min(seg.end / total, 1.0), f"transcribed {format_ts(seg.end)} of {format_ts(total)}")
    return Transcript(segments, info.language, float(info.language_probability or 0.0), float(total))
