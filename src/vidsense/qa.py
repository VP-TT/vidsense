"""Question answering over one video with LangChain.

    question ─▶ (follow-up? rewrite it with the chat history) ─▶ VideoRetriever ─▶ look at the top keyframes
                                                                                              │ (vision model)
                     answer with [mm:ss] citations  ◀── chat model ◀── prompt ◀──────────────┘

LangChain is the orchestration layer here: the retriever is a LangChain retriever, the
prompts are ChatPromptTemplates, and `prompt | model` is an LCEL chain, so swapping
DeepSeek-R1 on Ollama for an OpenAI model is a settings change.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .config import Settings
from .embed import load_embedder
from .llm import (
    ReasoningSplitter,
    check_vision,
    extract_citations,
    make_chat_model,
    make_vision_model,
    message_text,
    split_reasoning,
)
from .look import describe, frame_image, frames_to_inspect
from .retrieval import search
from .schemas import Chunk
from .store import VideoLibrary
from .timeutil import format_range, format_ts

SYSTEM_PROMPT = """You are VidSense. You answer questions about one video using excerpts from it.
Each excerpt shows its time range, what was said (a Whisper transcript), what was on screen \
(automatic CLIP labels, which can be wrong) and sometimes what a vision model saw when it \
looked at that moment's frame for this question. Trust the vision model's observations for \
visual details such as colours, objects and on-screen text.

Rules:
- Answer only from the excerpts. If they don't contain the answer, say you couldn't find it in the video.
- If the question is ambiguous (for example, which scene "the third one" means), pick the most likely \
reading from the excerpts, say which one you picked, and answer it. Don't deliberate at length.
- After each claim, cite the moment as [mm:ss] (or [h:mm:ss] for long videos), using the excerpt's start time.
- Keep it short: one to four sentences, or a short list when the question asks for steps.
- Talk about "the video", never about "excerpts"."""

HUMAN_PROMPT = """Video: {title} (length {duration})

Outline of the whole video, one line per moment (use it for questions about order or \
position, such as "the third scene" or "what happens after the forest"):
{outline}

Excerpts:
{context}

Question: {question}"""

CONDENSE_PROMPT = """Rewrite the follow-up question so it makes sense without the conversation. \
Keep every name and detail it refers to. Reply with the rewritten question only.

Conversation:
{history}

Follow-up question: {question}"""

log = logging.getLogger(__name__)

QA_TEMPLATE = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
CONDENSE_TEMPLATE = ChatPromptTemplate.from_messages([("human", CONDENSE_PROMPT)])


class VideoRetriever(BaseRetriever):
    """LangChain retriever over one video's chunks in ChromaDB."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    settings: Settings
    video_id: str
    k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return [
            Document(
                page_content=hit.chunk.embedding_text(),
                metadata={
                    "chunk_id": hit.chunk.id,
                    "start": hit.chunk.start,
                    "end": hit.chunk.end,
                    "transcript": hit.chunk.transcript,
                    "tags": hit.chunk.tags,
                    "keyframe_ids": hit.chunk.keyframe_ids,
                    "score": hit.score,
                    "rank": hit.rank,
                },
            )
            for hit in search(self.settings, self.video_id, query, self.k)
        ]


def format_context(docs: Sequence[Document]) -> str:
    blocks = []
    for number, doc in enumerate(docs, start=1):
        meta = doc.metadata
        said = meta["transcript"] or "(no speech)"
        lines = [f"[{number}] {format_range(meta['start'], meta['end'])}", f'Said: "{said}"']
        if meta["tags"]:
            lines.append("On screen: " + "; ".join(meta["tags"]))
        if meta.get("seen"):
            lines.append(f"Looked at the frame ({format_ts(meta['seen_at'])}): {meta['seen']}")
        blocks.append("\n".join(lines))
    # Chronological order reads more naturally to the model than relevance order.
    return "\n\n".join(b for _, b in sorted(zip((d.metadata["start"] for d in docs), blocks)))


def video_outline(chunks: Sequence, max_lines: int = 40) -> str:
    """One short line per chunk (time, opening words, top label), at most max_lines lines.

    The excerpts are only the best matches, so on their own they can't tell the model
    which scene is "the third one". Long videos are thinned to every n-th chunk.
    """
    step = max(1, -(-len(chunks) // max_lines))
    lines = []
    for chunk in list(chunks)[::step]:
        words = chunk.transcript.split()
        said = " ".join(words[:12]) + (" ..." if len(words) > 12 else "") if words else "(no speech)"
        label = f" (on screen: {chunk.tags[0]})" if chunk.tags else ""
        lines.append(f"[{format_ts(chunk.start)}] {said}{label}")
    return "\n".join(lines)


def format_history(history: Sequence[dict], turns: int = 3) -> str:
    recent = list(history)[-2 * turns :]
    return "\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in recent)


@dataclass
class Answer:
    question: str
    search_query: str  # the question after follow-up rewriting
    text: str
    reasoning: str = ""
    sources: list[Document] = field(default_factory=list)
    citations: list[float] = field(default_factory=list)  # cited timestamps in seconds
    observations: list[tuple[float, str]] = field(default_factory=list)  # (keyframe time, what the vision model saw)
    model: str = ""
    retrieval_ms: float = 0.0
    total_ms: float = 0.0


def _retrieval_only_answer(docs: Sequence[Document], intro: str = "No LLM is available, so here are the moments that best match your question:") -> str:
    if not docs:
        return "I couldn't find anything related in this video."
    lines = [intro]
    for doc in docs:
        snippet = doc.metadata["transcript"] or "On screen: " + "; ".join(doc.metadata["tags"])
        lines.append(f"- [{format_ts(doc.metadata['start'])}] {snippet[:200]}")
        if doc.metadata.get("seen"):
            lines.append(f"  Seen at [{format_ts(doc.metadata['seen_at'])}]: {doc.metadata['seen']}")
    return "\n".join(lines)


class VideoQA:
    def __init__(self, settings: Settings, video_id: str, llm=None, vision="auto"):
        """`vision` is a chat model that reads images, None to never look at frames, or "auto"
        to use the configured vision model when it is available."""
        self.settings = settings
        self.library = VideoLibrary(settings)
        self.record = self.library.get(video_id)
        if self.record is None or self.record.status != "ready":
            raise LookupError(f"video {video_id} is not processed")
        self.retriever = VideoRetriever(settings=settings, video_id=video_id, k=settings.answer.top_k)
        # Load the embedding model now, so the first question's retrieval time is the search itself.
        load_embedder(self.record.processing.get("embed_model", settings.processing.embed_model))
        self.llm = llm if llm is not None else make_chat_model(settings.answer)
        self.model_name = settings.answer.resolved_model if self.llm is not None else "retrieval only"
        if vision == "auto":
            vision = make_vision_model(settings.answer) if self.llm is not None and check_vision(settings.answer)[0] else None
        self.vision = vision if self.record.has_video else None

    def standalone_question(self, question: str, history: Sequence[dict]) -> str:
        """Rewrite a follow-up ("what happens after that?") into a question retrieval can use."""
        if not history or self.llm is None or not self.settings.answer.use_history:
            return question
        response = (CONDENSE_TEMPLATE | self.llm).invoke({"history": format_history(history), "question": question})
        _, rewritten = split_reasoning(message_text(response.content))
        rewritten = rewritten.strip().strip('"')
        return rewritten if 0 < len(rewritten) <= 400 else question

    def stream(self, question: str, history: Sequence[dict] = ()) -> Iterator[tuple[str, object]]:
        """Yield ("sources", docs); ("looking", n) and ("seen", (time, text)) when the vision model
    inspects keyframes; ("reasoning" | "answer", text) pieces; then ("done", Answer)."""
        started = time.perf_counter()
        query = self.standalone_question(question, history)
        retrieval_start = time.perf_counter()
        docs = self.retriever.invoke(query)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        yield "sources", docs

        observations: list[tuple[float, str]] = []
        if self.vision is not None and docs:
            keyframes = {k.id: k for k in self.library.keyframes(self.record.video_id)}
            picks = frames_to_inspect(docs, keyframes, self.settings.answer.vision_frames)
            if picks:
                yield "looking", len(picks)
            for doc, keyframe in picks:
                try:
                    seen = describe(self.vision, frame_image(self.record, self.library, keyframe), query, keyframe.time)
                except Exception as exc:  # a missing or failing vision model shouldn't cost the answer
                    log.warning("the vision model couldn't look at %.1fs: %s", keyframe.time, exc)
                    continue
                if seen:
                    doc.metadata["seen"], doc.metadata["seen_at"] = seen, keyframe.time
                    observations.append((keyframe.time, seen))
                    yield "seen", (keyframe.time, seen)

        reasoning, answer = [], []
        if self.llm is None:
            answer.append(_retrieval_only_answer(docs))
            yield "answer", answer[0]
        else:
            inputs = {
                "title": self.record.title,
                "duration": format_ts(self.record.duration),
                "outline": video_outline(
                    [Chunk.from_dict(c) for c in self.library.read_json(self.record.video_id, "chunks.json", default=[])]
                )
                or "(not available)",
                "context": format_context(docs) if docs else "(nothing relevant was found)",
                "question": query,
            }
            splitter = ReasoningSplitter()
            for chunk in (QA_TEMPLATE | self.llm).stream(inputs):
                pieces = []
                if chunk.additional_kwargs.get("reasoning_content"):  # providers that return it separately
                    pieces.append(("reasoning", chunk.additional_kwargs["reasoning_content"]))
                pieces += splitter.feed(message_text(chunk.content))
                for kind, text in pieces:
                    (reasoning if kind == "reasoning" else answer).append(text)
                    yield kind, text
            for kind, text in splitter.flush():
                (reasoning if kind == "reasoning" else answer).append(text)
                yield kind, text

        text = "".join(answer).strip()
        # Catch the case where only </think> appeared, so the stream couldn't tell thinking from answer.
        late_reasoning, text = split_reasoning(text)
        if not text and self.llm is not None:  # e.g. a reasoning model used its whole token budget thinking
            text = _retrieval_only_answer(docs, "The model ran out of room before answering, so here is what was found:")
        yield "done", Answer(
            question=question,
            search_query=query,
            text=text,
            reasoning="\n\n".join(p for p in ("".join(reasoning).strip(), late_reasoning) if p),
            sources=list(docs),
            citations=extract_citations(text, self.record.duration, [(d.metadata["start"], d.metadata["end"]) for d in docs]),
            observations=observations,
            model=self.model_name,
            retrieval_ms=round(retrieval_ms, 1),
            total_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def ask(self, question: str, history: Sequence[dict] = ()) -> Answer:
        for kind, payload in self.stream(question, history):
            if kind == "done":
                return payload
        raise RuntimeError("answer stream ended early")
