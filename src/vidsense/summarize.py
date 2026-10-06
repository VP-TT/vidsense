"""Whole-video summaries with chapters, via LangChain.

Retrieval picks a few chunks, which is right for questions but wrong for a summary:
a summary has to read everything. Short videos go to the model in one prompt. Long
ones use map-reduce: each part of the transcript becomes timestamped notes (map), then
the notes become one summary (reduce). Results are cached per model in summary.json.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from langchain_core.prompts import ChatPromptTemplate

from .config import Settings
from .llm import make_chat_model, message_text, split_reasoning
from .schemas import Chunk
from .store import VideoLibrary
from .timeutil import format_ts, parse_ts

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Summarize this video from its {source}.

Use exactly this format:
OVERVIEW: <two to four sentences>
CHAPTERS:
[mm:ss] <short title> - <one sentence>
[mm:ss] <short title> - <one sentence>

Give 3 to 8 chapters in time order. Use only timestamps that appear below.

Video: {title} (length {duration})

{text}""",
        )
    ]
)

NOTES_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Here is part of a video transcript with timestamps. Write 3 to 8 bullet notes on its \
main points, in order. Start each bullet with the [mm:ss] timestamp where the point begins. \
Reply with the bullets only.

{text}""",
        )
    ]
)

_CHAPTER = re.compile(r"^\[?((?:\d{1,2}:)?\d{1,2}:\d{2})\]?\s*[-–—:]?\s*(.+)$")


@dataclass
class Chapter:
    start: float
    title: str
    description: str = ""


@dataclass
class Summary:
    overview: str
    chapters: list[Chapter] = field(default_factory=list)
    model: str = ""
    created_at: str = ""
    raw: str = ""


def transcript_lines(chunks: list[Chunk]) -> list[str]:
    lines = []
    for chunk in chunks:
        line = f"[{format_ts(chunk.start)}] {chunk.transcript or '(no speech)'}"
        if chunk.tags:
            line += f" (on screen: {', '.join(chunk.tags[:3])})"
        lines.append(line)
    return lines


def parse_summary(text: str, duration: float | None = None) -> tuple[str, list[Chapter]]:
    """Read "OVERVIEW: ... CHAPTERS: [mm:ss] title - sentence" output into (overview, chapters).

    Models drift from the format (bold markers, bullets, chapters run together on one
    line, "CHAP 00:08:" prefixes), so chapters are split at bracketed timestamps and
    chapter markers first, and any line that starts with a timestamp is a chapter.
    """
    text = text.replace("**", "")
    # "CHAP 00:08: ..." or "Chapter 2: ..." start a new line; a chapter number is dropped,
    # but digits followed by ":" and more digits are the timestamp and stay.
    text = re.sub(r"\bCHAP(?:TER)?\b\.?\s*(?:\d+(?=[:.]\s*\D)[:.])?", "\n", text, flags=re.I)
    text = re.sub(r"\s*(\[(?:\d{1,2}:)?\d{1,2}:\d{2}\])", r"\n\1", text)  # one bracketed timestamp per line
    overview, chapters = [], []
    for line in text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line or re.fullmatch(r"CHAPTERS?\s*:?", line, flags=re.I):
            continue
        if line.upper().startswith("OVERVIEW"):
            line = line.split(":", 1)[1].strip() if ":" in line else ""
        match = _CHAPTER.match(line)
        if match:
            start = parse_ts(match.group(1))
            if duration is not None and start > duration + 1:
                continue
            title, *rest = re.split(r"\s+[-–—]\s+|:\s+", match.group(2), maxsplit=1)
            chapters.append(Chapter(start=start, title=title.strip(" .*"), description=rest[0].strip() if rest else ""))
        elif not chapters and line:
            overview.append(line)
    return " ".join(overview).strip(), sorted(chapters, key=lambda c: c.start)


def _pack(lines: list[str], budget: int) -> list[str]:
    """Join lines into blocks of at most `budget` characters (a long line gets its own block)."""
    blocks, current = [], ""
    for line in lines:
        if current and len(current) + len(line) + 1 > budget:
            blocks.append(current)
            current = ""
        current = f"{current}\n{line}" if current else line
    if current:
        blocks.append(current)
    return blocks


def _char_budget(settings: Settings) -> int:
    # About 4 characters per token; leave room for the instructions and the model's reply
    # (reasoning models need a lot of room to think).
    if settings.answer.provider == "ollama":
        return int(settings.answer.num_ctx * 4 * 0.5)
    return 60_000


def summarize(
    settings: Settings,
    video_id: str,
    *,
    force: bool = False,
    progress: Callable[[float, str], None] | None = None,
) -> Summary:
    library = VideoLibrary(settings)
    record = library.get(video_id)
    if record is None or record.status != "ready":
        raise LookupError(f"video {video_id} is not processed")
    model_name = settings.answer.resolved_model
    cache = library.read_json(video_id, "summary.json", default={})
    if not force and model_name in cache:
        data = cache[model_name]
        return Summary(**{**data, "chapters": [Chapter(**c) for c in data["chapters"]]})

    llm = make_chat_model(settings.answer)
    if llm is None:
        raise ValueError("Summaries need an LLM; set VIDSENSE_LLM_PROVIDER to ollama or openai.")

    def run(prompt: ChatPromptTemplate, **inputs) -> str:
        _, answer = split_reasoning(message_text((prompt | llm).invoke(inputs).content))
        return answer

    budget = _char_budget(settings)
    text, source = "\n".join(transcript_lines(library.chunks(video_id))), "timestamped transcript"
    rounds = 0
    while len(text) > budget and rounds < 3:  # map: notes per block, until everything fits in one prompt
        blocks = _pack(text.splitlines(), budget)
        notes = []
        for i, block in enumerate(blocks):
            if progress:
                progress(i / (len(blocks) + 1), f"reading part {i + 1} of {len(blocks)}")
            notes.append(run(NOTES_PROMPT, text=block))
        text, source, rounds = "\n".join(notes), "timestamped notes", rounds + 1

    if progress:
        progress(0.9, "writing the summary")
    raw = run(SUMMARY_PROMPT, source=source, title=record.title, duration=format_ts(record.duration), text=text[:budget])
    overview, chapters = parse_summary(raw, record.duration)
    summary = Summary(
        overview=overview or raw.strip(),
        chapters=chapters,
        model=model_name,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        raw=raw,
    )
    library.write_json(video_id, "summary.json", {**cache, model_name: asdict(summary)})
    return summary
