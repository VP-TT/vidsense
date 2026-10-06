import pytest

from vidsense.config import AnswerConfig
from vidsense.llm import ReasoningSplitter, check_llm, extract_citations, split_reasoning
from vidsense.summarize import parse_summary


@pytest.mark.parametrize(
    ("text", "reasoning", "answer"),
    [
        ("<think>Let me look.</think>\n\nThe trees are 200 years old [00:15].", "Let me look.", "The trees are 200 years old [00:15]."),
        ("Plain answer.", "", "Plain answer."),
        ("<think>still thinking when it stopped", "still thinking when it stopped", ""),
        ("opened in the prompt</think>Answer [01:02].", "opened in the prompt", "Answer [01:02]."),
        ("<think>a</think>first <think>b</think>second", "a\n\nb", "first second"),
    ],
)
def test_split_reasoning(text, reasoning, answer):
    assert split_reasoning(text) == (reasoning, answer)


def _stream(text, size):
    splitter = ReasoningSplitter()
    pieces = []
    for i in range(0, len(text), size):
        pieces += splitter.feed(text[i : i + size])
    pieces += splitter.flush()
    reasoning = "".join(t for k, t in pieces if k == "reasoning")
    answer = "".join(t for k, t in pieces if k == "answer")
    return reasoning, answer


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 64])
def test_streaming_splitter_handles_tags_cut_across_chunks(size):
    text = "<think>Step one. Step two.</think>\n\nThe answer is 50 degrees [00:32]."
    reasoning, answer = _stream(text, size)
    assert reasoning == "Step one. Step two."
    assert answer == "The answer is 50 degrees [00:32]."


def test_streaming_splitter_passes_plain_text_through():
    assert _stream("No reasoning here, just <b>html</b>.", 4) == ("", "No reasoning here, just <b>html</b>.")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("It starts at [02:15] and again at [1:02:05].", [135, 3725]),
        ("See (00:42) for the stars.", [42]),
        ("Range [00:15–00:25] counts once.", [15]),
        ("Both [00:08, 00:32] are relevant.", [8, 32]),
        ("Repeated [00:08] and [00:08].", [8]),
        ("A ratio of 3:1 or a time 10:30 outside brackets is ignored.", []),
        ("Beyond the end [59:00] is dropped.", []),
    ],
)
def test_extract_citations(text, expected):
    assert extract_citations(text, duration=3800 if "1:02:05" in text else 100) == expected


def test_bare_timestamps_count_only_inside_retrieved_excerpts():
    text = "The stars appear from 00:40 to 00:46, after the 10:30 train and before [00:05]."
    assert extract_citations(text, duration=100, source_ranges=[(40.0, 46.0)]) == [40, 46, 5]
    assert extract_citations(text, duration=100) == [5]


def test_check_llm_messages(monkeypatch):
    assert check_llm(AnswerConfig(provider="none"))[0] is False
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ready, message = check_llm(AnswerConfig(provider="openai"))
    assert not ready and "OPENAI_API_KEY" in message
    ready, message = check_llm(AnswerConfig(provider="ollama", ollama_url="http://127.0.0.1:9"), timeout=0.5)
    assert not ready and "isn't running" in message


def test_parse_summary_formats():
    raw = """**OVERVIEW:** A short tour of five places.
It ends at night.

CHAPTERS:
- **[00:00] Welcome** - The host introduces the tour.
[00:15] Pine forest: Trees over 200 years old.
• 00:40 Night sky – Stars over the mountains
[09:99] broken line
[59:00] Too late - beyond the video
"""
    overview, chapters = parse_summary(raw, duration=60)
    assert overview == "A short tour of five places. It ends at night."
    assert [(c.start, c.title, c.description) for c in chapters] == [
        (0, "Welcome", "The host introduces the tour."),
        (15, "Pine forest", "Trees over 200 years old."),
        (40, "Night sky", "Stars over the mountains"),
    ]


def test_parse_summary_inline_chapters_from_a_small_model():
    raw = (
        "The video introduces VidSense, a tour of five places. CHAP 00:08: Open Ocean - A sailboat drifts. "
        "CHAP 00:15: Forest - Trees over 200 years old. [00:40] Night Sky - Stars above the mountains. "
        "Chapter 4: [00:46] The end"
    )
    overview, chapters = parse_summary(raw, duration=60)
    assert overview == "The video introduces VidSense, a tour of five places."
    assert [(c.start, c.title, c.description) for c in chapters] == [
        (8, "Open Ocean", "A sailboat drifts."),
        (15, "Forest", "Trees over 200 years old."),
        (40, "Night Sky", "Stars above the mountains."),
        (46, "The end", ""),
    ]
