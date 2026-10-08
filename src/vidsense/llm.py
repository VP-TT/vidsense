"""Chat models for the answer step (through LangChain), plus helpers for reasoning models.

Providers:
  ollama  local models; the default is DeepSeek-R1 8B
  openai  OpenAI, or any OpenAI-compatible server via OPENAI_BASE_URL
          (DeepSeek's API, LM Studio, vLLM, ...)
  none    retrieval only: VidSense still finds the moments, it just doesn't write prose

DeepSeek-R1 and similar models think out loud before answering. Ollama returns that
reasoning in a separate field when asked; other servers inline it as <think>...</think>.
Either way we keep it (the UI shows it folded away) but never mix it into the answer
or into the citations.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from functools import lru_cache

from .config import AnswerConfig
from .timeutil import parse_ts

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


@lru_cache(maxsize=16)
def ollama_can_think(base_url: str, model: str, timeout: float = 3.0) -> bool:
    """Whether Ollama reports the "thinking" capability (DeepSeek-R1, Qwen3, ...).

    For those models we ask Ollama to return the reasoning separately; sending that flag
    to a model without the capability is an error, so other models get the default.
    """
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/show",
        data=json.dumps({"model": model}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return "thinking" in json.load(response).get("capabilities", [])
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _accepts_temperature(model: str) -> bool:
    # OpenAI's reasoning models reject a temperature other than the default.
    return not model.startswith(("o1", "o3", "o4", "gpt-5"))


def make_chat_model(cfg: AnswerConfig):
    """A LangChain chat model for the configured provider, or None for retrieval-only mode."""
    model = cfg.resolved_model
    if cfg.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=cfg.ollama_url,
            temperature=cfg.temperature,
            num_ctx=cfg.num_ctx,  # explicit window, so long prompts aren't silently truncated
            num_predict=cfg.num_predict,
            keep_alive="30m",
            reasoning=True if ollama_can_think(cfg.ollama_url, model) else None,
        )
    if cfg.provider == "openai":
        from langchain_openai import ChatOpenAI

        extra = {"temperature": cfg.temperature} if _accepts_temperature(model) else {}
        return ChatOpenAI(model=model, **extra)
    return None


def check_llm(cfg: AnswerConfig, timeout: float = 2.0) -> tuple[bool, str]:
    """(ready, message) without generating anything. The message says how to fix problems."""
    model = cfg.resolved_model
    if cfg.provider == "none":
        return False, "Retrieval only: no LLM is configured."
    if cfg.provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            return False, "Set OPENAI_API_KEY in your shell or in a .env file to use OpenAI."
        return True, f"OpenAI · {model}"
    try:
        with urllib.request.urlopen(f"{cfg.ollama_url.rstrip('/')}/api/tags", timeout=timeout) as response:
            names = {m["name"] for m in json.load(response).get("models", [])}
    except (urllib.error.URLError, OSError, ValueError):
        return False, f"Ollama isn't running at {cfg.ollama_url}. Install it from ollama.com and start it (`ollama serve`)."
    if model not in names and f"{model}:latest" not in names:
        return False, f"The model {model} isn't downloaded yet. Run `ollama pull {model}`."
    return True, f"Ollama · {model}"


def message_text(content) -> str:
    """Text of a LangChain message content, which is a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def split_reasoning(text: str) -> tuple[str, str]:
    """Split a complete response into (reasoning, answer)."""
    reasoning = re.findall(r"<think>(.*?)</think>", text, flags=re.S)
    answer = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if THINK_OPEN in answer:  # generation stopped while still thinking
        answer, _, tail = answer.partition(THINK_OPEN)
        reasoning.append(tail)
    if THINK_CLOSE in answer:  # some templates open the think block in the prompt, so only </think> appears
        head, _, answer = answer.rpartition(THINK_CLOSE)
        reasoning.insert(0, head)
    return "\n\n".join(r.strip() for r in reasoning if r.strip()), answer.strip()


def _partial_suffix(text: str, tag: str) -> int:
    """Length of the longest end of `text` that could be the start of `tag`."""
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


class ReasoningSplitter:
    """Separates <think>...</think> from streamed text, even when a tag is split across chunks."""

    def __init__(self) -> None:
        self.buffer = ""
        self.thinking = False
        self.answer_started = False

    def _emit(self, kind: str, text: str, out: list[tuple[str, str]]) -> None:
        if kind == "answer" and not self.answer_started:
            text = text.lstrip()  # R1 puts blank lines after </think>
            self.answer_started = bool(text)
        if text:
            out.append((kind, text))

    def feed(self, text: str) -> list[tuple[str, str]]:
        """Return [(kind, text)] pieces ready to show, kind being "reasoning" or "answer"."""
        self.buffer += text
        out: list[tuple[str, str]] = []
        while True:
            tag = THINK_CLOSE if self.thinking else THINK_OPEN
            kind = "reasoning" if self.thinking else "answer"
            index = self.buffer.find(tag)
            if index >= 0:
                self._emit(kind, self.buffer[:index], out)
                self.buffer = self.buffer[index + len(tag) :]
                self.thinking = not self.thinking
                continue
            keep = _partial_suffix(self.buffer, tag)
            self._emit(kind, self.buffer[: len(self.buffer) - keep], out)
            self.buffer = self.buffer[len(self.buffer) - keep :]
            return out

    def flush(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        self._emit("reasoning" if self.thinking else "answer", self.buffer, out)
        self.buffer = ""
        return out


_BRACKETED = re.compile(r"[\[(]([^\[\]()]{3,80})[\])]")
_TIMESTAMP = re.compile(r"\b(?:\d{1,2}:)?\d{1,2}:\d{2}\b")


def extract_citations(
    text: str, duration: float | None = None, source_ranges: list[tuple[float, float]] | None = None
) -> list[float]:
    """Cited timestamps in order of appearance, in seconds.

    Counts [mm:ss], [h:mm:ss], (mm:ss), [mm:ss-mm:ss] (its start) and [mm:ss, mm:ss].
    Small models often drop the brackets ("from 00:40 to 00:46"), so a bare timestamp
    also counts when it falls inside one of the retrieved excerpts. Timestamps past the
    end of the video are dropped: an invented moment shouldn't get a jump button.
    """
    found: list[tuple[int, float]] = []  # (position in text, seconds)
    for group in _BRACKETED.finditer(text):
        offset = group.start(1)
        for part in re.finditer(r"[^,;]+", group.group(1)):
            match = _TIMESTAMP.search(part.group())
            if match:
                found.append((offset + part.start() + match.start(), parse_ts(match.group())))
    if source_ranges:
        for match in _TIMESTAMP.finditer(text):
            seconds = parse_ts(match.group())
            if any(start - 1 <= seconds <= end + 1 for start, end in source_ranges):
                found.append((match.start(), seconds))
    citations: list[float] = []
    for _, seconds in sorted(found):
        if (duration is None or seconds <= duration + 1) and seconds not in citations:
            citations.append(seconds)
    return citations
