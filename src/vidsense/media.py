"""Video and audio decoding with PyAV, which bundles FFmpeg (no system ffmpeg needed)."""

from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from PIL import Image

Progress = Callable[[float, str], None]

VIDEO_EXTENSIONS = ("mp4", "mov", "m4v", "webm", "mkv", "avi")
AUDIO_EXTENSIONS = ("mp3", "m4a", "wav", "flac", "ogg")


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    has_video: bool
    has_audio: bool
    width: int = 0
    height: int = 0
    fps: float = 0.0


@dataclass
class SampledFrame:
    time: float
    image: Image.Image  # RGB, upright, short side about 256 px


def file_digest(path: str | Path, length: int = 16) -> str:
    """Content hash used as the video id, so re-uploading the same file reuses its index."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:length]


def probe(path: str | Path) -> MediaInfo:
    with av.open(str(path)) as container:
        video = container.streams.video[0] if container.streams.video else None
        audio = container.streams.audio[0] if container.streams.audio else None
        duration = container.duration / av.time_base if container.duration else 0.0
        if not duration:
            for stream in (video, audio):
                if stream is not None and stream.duration and stream.time_base:
                    duration = float(stream.duration * stream.time_base)
                    break
        width = height = 0
        fps = 0.0
        if video is not None:
            width, height = video.codec_context.width, video.codec_context.height
            fps = float(video.average_rate) if video.average_rate else 0.0
        return MediaInfo(
            duration=float(duration),
            has_video=video is not None,
            has_audio=audio is not None,
            width=width,
            height=height,
            fps=fps,
        )


def sampling_interval(duration: float, fps: float, max_frames: int) -> float:
    """Seconds between sampled frames: 1/fps, stretched so long videos stay under max_frames."""
    interval = 1.0 / fps
    if duration > 0 and duration / interval > max_frames:
        interval = duration / max_frames
    return interval


def _to_image(frame: av.VideoFrame, short_side: int) -> Image.Image:
    scale = min(1.0, short_side / max(1, min(frame.width, frame.height)))
    width = max(2, int(frame.width * scale) // 2 * 2)
    height = max(2, int(frame.height * scale) // 2 * 2)
    image = frame.to_image(width=width, height=height)
    rotation = frame.rotation  # counterclockwise degrees from the display matrix (phone videos)
    if rotation % 360:
        image = image.rotate(rotation, expand=True)
    return image


def sample_frames(
    path: str | Path,
    interval: float,
    short_side: int = 256,
    duration: float = 0.0,
    progress: Progress | None = None,
) -> Iterator[SampledFrame]:
    """Yield one frame per `interval` seconds, decoding sequentially.

    Sequential decoding with frame threading is faster and more robust than seeking
    for typical sampling rates. Only the sampled frames get converted to RGB.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        offset = container.start_time / av.time_base if container.start_time else 0.0
        next_time = 0.0
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            t = frame.time - offset
            if t + 1e-3 < next_time:
                continue
            yield SampledFrame(time=max(0.0, t), image=_to_image(frame, short_side))
            next_time = (math.floor(t / interval + 1e-6) + 1) * interval
            if progress and duration:
                progress(min(t / duration, 1.0), f"frames sampled up to {t:.0f}s")


def load_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Decode the first audio stream to mono float32 at `sample_rate` (Whisper expects 16 kHz).

    We decode here instead of letting faster-whisper do it: its decoder passes an
    argument to av.open() that newer PyAV releases removed.
    """
    resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
    pieces = []
    with av.open(str(path)) as container:
        frames = container.decode(container.streams.audio[0])
        while True:
            try:
                frame = next(frames)
            except StopIteration:
                break
            except av.error.InvalidDataError:  # skip a corrupt packet rather than fail the video
                continue
            frame.pts = None  # let the resampler ignore broken timestamps
            pieces += [out.to_ndarray().reshape(-1) for out in resampler.resample(frame)]
    pieces += [out.to_ndarray().reshape(-1) for out in resampler.resample(None)]
    return np.concatenate(pieces).astype(np.float32) if pieces else np.zeros(0, dtype=np.float32)


def jpeg_bytes(image: Image.Image, quality: int = 85) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()
