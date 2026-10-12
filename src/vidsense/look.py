"""Looking at frames: a vision-language model inspects the retrieved keyframes for a question.

The index knows what was said plus coarse CLIP labels ("cars on a road"), and the answer
model reads only text, so a question about something shown but never said ("what colour
are the cars?") can't be answered from the excerpts alone. Here a small vision-language
model looks at the keyframes of the best-matching moments, with the question in mind,
and its observations go into the answer prompt next to the transcript.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from PIL import Image

from .llm import message_text, split_reasoning
from .media import grab_frame
from .schemas import Keyframe, VideoRecord
from .store import VideoLibrary
from .timeutil import format_ts

log = logging.getLogger(__name__)
NOTHING = "nothing relevant"

LOOK_PROMPT = """This is a frame from a video, at {time}. Someone asked about the video: "{question}"

Describe what you can see in this frame that helps answer the question: colours, objects, \
people, actions and any visible text. Be specific and brief, at most three sentences. \
If nothing in the frame is relevant to the question, reply only: {nothing}."""


def frames_to_inspect(docs: list[Document], keyframes: dict[int, Keyframe], limit: int) -> list[tuple[Document, Keyframe]]:
    """The first keyframe of each retrieved chunk, best-matching chunks first."""
    picks = []
    for doc in docs:
        ids = [i for i in doc.metadata.get("keyframe_ids", []) if i in keyframes]
        if ids:
            picks.append((doc, keyframes[ids[0]]))
        if len(picks) >= limit:
            break
    return picks


def frame_image(record: VideoRecord, library: VideoLibrary, keyframe: Keyframe) -> Image.Image:
    """A sharp copy of the keyframe from the video; the stored thumbnail if the video is gone."""
    source = Path(record.source_path)
    if source.is_file() and record.has_video:
        try:
            return grab_frame(source, keyframe.time)
        except Exception as exc:  # unusual codecs or a truncated file: the thumbnail still works
            log.warning("could not decode the frame at %.1fs: %s", keyframe.time, exc)
    return Image.open(library.path(record.video_id, keyframe.image)).convert("RGB")


def describe(vision_model, image: Image.Image, question: str, seconds: float) -> str:
    """Ask the vision model about one frame. Returns "" when it sees nothing relevant."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    data = base64.b64encode(buffer.getvalue()).decode()
    prompt = LOOK_PROMPT.format(time=format_ts(seconds), question=question, nothing=NOTHING)
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}},
        ]
    )
    _, text = split_reasoning(message_text(vision_model.invoke([message]).content))
    text = text.strip()
    return "" if text.lower().strip(" .").startswith(NOTHING) else text
