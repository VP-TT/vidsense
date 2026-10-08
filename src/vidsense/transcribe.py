"""Speech to text with Whisper, using the faster-whisper (CTranslate2) implementation.

faster-whisper runs the same Whisper weights roughly 4x faster than the reference
implementation. Whisper's own segments are decoding windows rather than sentences
(large-v3-turbo happily ends one at "...five very different places around"), so we ask
for word timestamps and re-cut the transcript into sentences. Sentences are the text
side of the DTW alignment, and chunk boundaries can only fall between them.
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
_SENTENCE_END = (".", "?", "!", "…", "。", "？", "！")
PAUSE_SECONDS = 1.5  # a silence this long ends a sentence even without punctuation
MAX_SENTENCE_SECONDS = 20.0  # unpunctuated speech (lyrics, auto-captions style) is cut here


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
        word_timestamps=True,
    )
    words, windows = [], []
    total = audio.size / SAMPLE_RATE
    for seg in segments_iter:  # a generator: decoding happens while we iterate
        words.extend(seg.words or [])
        if seg.text.strip():
            windows.append(Segment(id=len(windows), start=round(seg.start, 2), end=round(seg.end, 2), text=seg.text.strip()))
        if progress and total:
            progress(min(seg.end / total, 1.0), f"transcribed {format_ts(seg.end)} of {format_ts(total)}")
    sentences = split_sentences(words) if words else windows
    return Transcript(sentences, info.language, float(info.language_probability or 0.0), float(total))


def split_sentences(words) -> list[Segment]:
    """Group timestamped words (objects with .start, .end, .word) into sentences."""
    sentences: list[Segment] = []
    current: list = []

    def close() -> None:
        text = "".join(w.word for w in current).strip()
        if text:
            sentences.append(Segment(id=len(sentences), start=round(current[0].start, 2), end=round(current[-1].end, 2), text=text))
        current.clear()

    for word in words:
        if current and (word.start - current[-1].end > PAUSE_SECONDS or word.end - current[0].start > MAX_SENTENCE_SECONDS):
            close()
        current.append(word)
        if word.word.strip().rstrip("\"'”’)]").endswith(_SENTENCE_END):
            close()
    if current:
        close()
    return sentences
