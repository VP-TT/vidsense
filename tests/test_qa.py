from dataclasses import replace

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel, GenericFakeChatModel, ParrotFakeChatModel
from langchain_core.messages import AIMessage

from vidsense import qa
from vidsense.config import AnswerConfig, Settings
from vidsense.schemas import Chunk, Hit, VideoRecord
from vidsense.store import VideoLibrary

CHUNKS = [
    Chunk(id="v-0000", index=0, start=15.0, end=25.0, transcript="Some of these trees are more than 200 years old.", tags=["a forest with trees"]),
    Chunk(id="v-0001", index=1, start=40.0, end=46.0, transcript="Thousands of stars appear above the mountains.", tags=["mountains"]),
]


@pytest.fixture
def queries(monkeypatch):
    """Records every query that reaches retrieval; search returns the two chunks, stars first."""
    seen = []

    def fake_search(_settings, _video_id, query, k=5):
        seen.append(query)
        return [Hit(chunk=c, score=0.9 - 0.1 * i, rank=i + 1) for i, c in enumerate(reversed(CHUNKS[:k]))]

    monkeypatch.setattr(qa, "search", fake_search)
    return seen


@pytest.fixture(autouse=True)
def no_embedder(monkeypatch):
    monkeypatch.setattr(qa, "load_embedder", lambda name: None)


@pytest.fixture
def settings(tmp_path, queries):
    settings = Settings(data_dir=tmp_path, answer=AnswerConfig(provider="none", top_k=2))
    VideoLibrary(settings).save(VideoRecord(video_id="v", title="Demo", source_path="/x.mp4", duration=52.0, status="ready"))
    return settings


def test_answer_strips_reasoning_and_extracts_citations(settings):
    model = GenericFakeChatModel(messages=iter([AIMessage(content="<think>The forest excerpt says it.</think>\n\nThey are more than 200 years old [00:15].")]))
    events = list(qa.VideoQA(settings, "v", llm=model).stream("How old are the trees?"))

    kinds = [kind for kind, _ in events]
    assert kinds[0] == "sources" and kinds[-1] == "done" and "reasoning" in kinds
    answer = events[-1][1]
    assert answer.text == "They are more than 200 years old [00:15]."
    assert answer.reasoning == "The forest excerpt says it."
    assert answer.citations == [15.0]
    assert [d.metadata["chunk_id"] for d in answer.sources] == ["v-0001", "v-0000"]  # relevance order


def test_prompt_contains_timestamped_context_in_time_order(settings):
    answer = qa.VideoQA(settings, "v", llm=ParrotFakeChatModel()).ask("How old are the trees?")
    prompt = answer.text  # the parrot model echoes the human message
    assert "Video: Demo (length 00:52)" in prompt
    assert prompt.index("00:15–00:25") < prompt.index("00:40–00:46")
    assert 'Said: "Some of these trees are more than 200 years old."' in prompt
    assert "On screen: a forest with trees" in prompt
    assert prompt.rstrip().endswith("Question: How old are the trees?")


def test_follow_up_is_rewritten_before_retrieval(settings, queries):
    model = FakeListChatModel(responses=["<think>resolve 'they'</think>How old are the trees in the forest?", "Over 200 years [00:15]."])
    history = [{"role": "user", "content": "Tell me about the forest."}, {"role": "assistant", "content": "It is a pine forest [00:15]."}]
    answer = qa.VideoQA(settings, "v", llm=model).ask("How old are they?", history)
    assert queries == ["How old are the trees in the forest?"]
    assert answer.search_query == "How old are the trees in the forest?"
    assert answer.text == "Over 200 years [00:15]."


def test_history_is_ignored_when_disabled(settings, queries):
    settings = replace(settings, answer=replace(settings.answer, use_history=False))
    model = FakeListChatModel(responses=["Over 200 years [00:15]."])
    qa.VideoQA(settings, "v", llm=model).ask("How old are they?", [{"role": "user", "content": "hi"}])
    assert queries == ["How old are they?"]


def test_retrieval_only_mode_lists_moments(settings):
    answer = qa.VideoQA(settings, "v").ask("How old are the trees?")
    assert answer.model == "retrieval only"
    assert answer.text.startswith("No LLM is available")
    assert answer.citations == [40.0, 15.0]
