import json

import pytest

from vidsense.config import Settings
from vidsense.evaluate import (
    contains_match,
    format_summary,
    load_activitynet,
    load_jsonl,
    normalize,
    is_relevant,
    parse_judgement,
    random_hit_rate,
    summarize_rows,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("More than two hundred years!", "more than 200 years"),
        ("Fifty degrees Celsius.", "50 degrees celsius"),
        ("two thousand and twenty six", "2026"),
        ("The answer is 3.5 meters.", "answer is 3.5 meters"),
        ("A small sailboat", "small sailboat"),
    ],
)
def test_normalize(text, expected):
    assert normalize(text) == expected


def test_contains_match():
    assert contains_match("The trees are older than 200 years [00:15].", ["200 years", "two hundred years"])
    assert contains_match("It can reach fifty degrees at noon.", ["50 degrees"])
    assert not contains_match("It can reach 150 degrees.", ["50 degrees"])  # whole-token match only
    assert contains_match("Yes, the athlete wears trousers.", ["yes"])
    assert not contains_match("There is no doubt: yes.", ["no"])  # yes/no must lead the answer
    assert not contains_match("anything", [])


def test_is_relevant_needs_real_overlap():
    assert is_relevant(10, 20, 15, 30)  # 5 s inside the span
    assert not is_relevant(10, 20, 19.5, 30)  # touches the span at a scene boundary
    assert is_relevant(10, 11, 10.4, 30)  # short chunk: more than half of it overlaps
    assert is_relevant(10, 20, 14, 14)  # a moment inside the chunk
    assert not is_relevant(10, 20, 25, 25)


def test_random_hit_rate():
    assert random_hit_rate(7, 1, 1) == pytest.approx(1 / 7)
    assert random_hit_rate(7, 1, 5) == pytest.approx(5 / 7)
    assert random_hit_rate(3, 1, 5) == 1.0
    assert random_hit_rate(7, 0, 5) == 0.0


@pytest.mark.parametrize(
    ("reply", "expected"),
    [("yes 4", (True, 4)), ("No, 1", (False, 1)), ("**Yes** - score: 5", (True, 5)), ("unsure", (None, None))],
)
def test_parse_judgement(reply, expected):
    assert parse_judgement(reply) == expected


def test_load_jsonl_resolves_relative_paths(tmp_path):
    qa = tmp_path / "qa.jsonl"
    qa.write_text(
        json.dumps({"video": "clip.mp4", "question": "Q1?", "answer": "A", "start": 1, "end": 2}) + "\n\n"
        + json.dumps({"video": "/abs/other.mp4", "question": "Q2?", "answers": ["x", "y"]}) + "\n"
    )
    first, second = load_jsonl(qa)
    assert first.video == str(tmp_path / "clip.mp4") and first.answers == ["A"] and (first.start, first.end) == (1, 2)
    assert second.video == "/abs/other.mp4" and second.answers == ["x", "y"] and second.start is None


def test_load_activitynet_keeps_questions_with_videos(tmp_path):
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "v_abc.mp4").write_bytes(b"")
    (tmp_path / "q.json").write_text(json.dumps([
        {"video_name": "abc", "question": "is the man wearing a hat", "question_id": "v_abc_1"},
        {"video_name": "v_gone", "question": "what color is the car", "question_id": "v_gone_1"},
    ]))
    (tmp_path / "a.json").write_text(json.dumps([
        {"answer": "no", "type": 3, "question_id": "v_abc_1"},
        {"answer": "red", "type": 4, "question_id": "v_gone_1"},
    ]))
    items = load_activitynet(tmp_path / "q.json", tmp_path / "a.json", tmp_path / "videos")
    assert len(items) == 1
    assert items[0].question == "is the man wearing a hat?" and items[0].answers == ["no"] and items[0].qtype == "yes/no"
    with pytest.raises(ValueError, match="none of the ActivityNet-QA videos"):
        load_activitynet(tmp_path / "q.json", tmp_path / "a.json", tmp_path)


def test_summary_and_report(tmp_path):
    rows = [
        {"type": "forest", "retrieval_ms": 8.0, "gold": [0, 1], "hit_at_1": True, "hit_at_k": True, "reciprocal_rank": 1.0,
         "random_hit_at_1": 0.2, "random_hit_at_k": 0.8, "answer_s": 4.0, "contains_match": True, "judge_correct": True, "judge_score": 5, "model": "m"},
        {"type": "desert", "retrieval_ms": 12.0, "gold": [0, 1], "hit_at_1": False, "hit_at_k": True, "reciprocal_rank": 0.5,
         "random_hit_at_1": 0.2, "random_hit_at_k": 0.8, "answer_s": 6.0, "contains_match": False, "judge_correct": None, "judge_score": None, "model": "m"},
        {"type": "", "retrieval_ms": 10.0},
    ]
    summary = summarize_rows(rows, k=5, settings=Settings(data_dir=tmp_path), n_videos=1)
    assert summary["retrieval"] == {"questions": 2, "hit_at_1": 0.5, "hit_at_k": 1.0, "mrr": 0.75, "random_hit_at_1": 0.2, "random_hit_at_k": 0.8}
    assert summary["answers"]["contains_match"] == 0.5 and summary["answers"]["judge_accuracy"] == 1.0
    assert summary["latency"]["retrieval_ms"]["p50"] == 10.0
    text = format_summary(summary)
    assert "hit@5   1.00   (random ranking: 0.80)" in text and "contains-match  0.50" in text
