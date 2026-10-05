"""Timestamp formatting and parsing, plus WebVTT export for the video player."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .schemas import Segment

_TS_RE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)\s*$")


def format_ts(seconds: float) -> str:
    """135.7 -> '02:15'; 3725 -> '1:02:05'. Rounds down, like a video player."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def format_range(start: float, end: float) -> str:
    return f"{format_ts(start)}–{format_ts(end)}"


def parse_ts(text: str) -> float:
    """Parse 'h:mm:ss', 'mm:ss' or plain seconds into seconds."""
    text = text.strip()
    match = _TS_RE.match(text)
    if match:
        hours, minutes, secs = match.groups()
        return int(hours or 0) * 3600 + int(minutes) * 60 + float(secs)
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"not a timestamp: {text!r}") from None


def _vtt_ts(seconds: float) -> str:
    millis = max(0, round(seconds * 1000))
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def to_vtt(segments: Iterable[Segment]) -> str:
    """WebVTT subtitles, so the Streamlit player can show the transcript as captions."""
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines += [f"{_vtt_ts(seg.start)} --> {_vtt_ts(seg.end)}", seg.text.strip(), ""]
    return "\n".join(lines)
