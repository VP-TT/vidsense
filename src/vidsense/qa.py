"""Question answering over one video with LangChain.

    question ─▶ (follow-up? rewrite it with the chat history) ─▶ VideoRetriever ─▶ prompt ─▶ chat model
                                                                                              │
                     answer with [mm:ss] citations  ◀── strip <think> reasoning ◀─────────────┘

LangChain is the orchestration layer here: the retriever is a LangChain retriever, the
prompts are ChatPromptTemplates, and `prompt | model` is an LCEL chain, so swapping
DeepSeek-R1 on Ollama for an OpenAI model is a settings change.
"""

from __future__ import annotations

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
from .llm import ReasoningSplitter, extract_citations, make_chat_model, message_text, split_reasoning
from .retrieval import search
from .store import VideoLibrary
from .timeutil import format_range, format_ts

SYSTEM_PROMPT = """You are VidSense. You answer questions about one video using excerpts from it.
Each excerpt shows its time range, what was said (a Whisper transcript) and what was on screen \
(automatic CLIP labels, which can be wrong).

Rules:
- Answer only from the excerpts. If they don't contain the answer, say you couldn't find it in the video.
- After each claim, cite the moment as [mm:ss] (or [h:mm:ss] for long videos), using the excerpt's start time.
- Keep it short: one to four sentences, or a short list when the question asks for steps.
- Talk about "the video", never about "excerpts"."""

HUMAN_PROMPT = """Video: {title} (length {duration})

Excerpts:
{context}

Question: {question}"""

CONDENSE_PROMPT = """Rewrite the follow-up question so it makes sense without the conversation. \
Keep every name and detail it refers to. Reply with the rewritten question only.

Conversation:
{history}

Follow-up question: {question}"""

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
        blocks.append("\n".join(lines))
    # Chronological order reads more naturally to the model than relevance order.
    return "\n\n".join(b for _, b in sorted(zip((d.metadata["start"] for d in docs), blocks)))


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
    model: str = ""
    retrieval_ms: float = 0.0
    total_ms: float = 0.0


def _retrieval_only_answer(docs: Sequence[Document]) -> str:
    if not docs:
        return "I couldn't find anything related in this video."
    lines = ["No LLM is available, so here are the moments that best match your question:"]
    for doc in docs:
        snippet = doc.metadata["transcript"] or "On screen: " + "; ".join(doc.metadata["tags"])
        lines.append(f"- [{format_ts(doc.metadata['start'])}] {snippet[:200]}")
    return "\n".join(lines)


class VideoQA:
    def __init__(self, settings: Settings, video_id: str, llm=None):
        self.settings = settings
        self.record = VideoLibrary(settings).get(video_id)
        if self.record is None or self.record.status != "ready":
            raise LookupError(f"video {video_id} is not processed")
        self.retriever = VideoRetriever(settings=settings, video_id=video_id, k=settings.answer.top_k)
        # Load the embedding model now, so the first question's retrieval time is the search itself.
        load_embedder(self.record.processing.get("embed_model", settings.processing.embed_model))
        self.llm = llm if llm is not None else make_chat_model(settings.answer)
        self.model_name = settings.answer.resolved_model if self.llm is not None else "retrieval only"

    def standalone_question(self, question: str, history: Sequence[dict]) -> str:
        """Rewrite a follow-up ("what happens after that?") into a question retrieval can use."""
        if not history or self.llm is None or not self.settings.answer.use_history:
            return question
        response = (CONDENSE_TEMPLATE | self.llm).invoke({"history": format_history(history), "question": question})
        _, rewritten = split_reasoning(message_text(response.content))
        rewritten = rewritten.strip().strip('"')
        return rewritten if 0 < len(rewritten) <= 400 else question

    def stream(self, question: str, history: Sequence[dict] = ()) -> Iterator[tuple[str, object]]:
        """Yield ("sources", docs), then ("reasoning" | "answer", text) pieces, then ("done", Answer)."""
        started = time.perf_counter()
        query = self.standalone_question(question, history)
        retrieval_start = time.perf_counter()
        docs = self.retriever.invoke(query)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        yield "sources", docs

        reasoning, answer = [], []
        if self.llm is None:
            answer.append(_retrieval_only_answer(docs))
            yield "answer", answer[0]
        else:
            inputs = {
                "title": self.record.title,
                "duration": format_ts(self.record.duration),
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
        if not text and self.llm is not None:
            text = "The model returned no answer. Try again, or switch to a non-reasoning model."
        yield "done", Answer(
            question=question,
            search_query=query,
            text=text,
            reasoning="\n\n".join(p for p in ("".join(reasoning).strip(), late_reasoning) if p),
            sources=list(docs),
            citations=extract_citations(text, self.record.duration, [(d.metadata["start"], d.metadata["end"]) for d in docs]),
            model=self.model_name,
            retrieval_ms=round(retrieval_ms, 1),
            total_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def ask(self, question: str, history: Sequence[dict] = ()) -> Answer:
        for kind, payload in self.stream(question, history):
            if kind == "done":
                return payload
        raise RuntimeError("answer stream ended early")
