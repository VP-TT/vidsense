"""Evaluation: retrieval quality, answer accuracy and latency on a QA set.

Inputs
  * VidSense JSONL (what `vidsense demo` writes), one object per line:
      {"video": "path.mp4", "question": "...", "answer": "...", "start": 12.0, "end": 20.5}
    "answer" may also be "answers" (a list); answer and span are both optional.
  * ActivityNet-QA: test_q.json + test_a.json and a folder of the downloaded videos.

Metrics
  retrieval  hit@1, hit@k and MRR: a retrieved chunk counts if it really overlaps the gold
             time span (at least 1 s, or half the chunk if it is shorter; touching the span
             at a scene boundary doesn't count). Each comes with the score a random ranking
             would get, because on a short video with few chunks hit@5 is easy.
  answers    contains-match: the normalised reference answer appears in the prediction
             (yes/no answers must lead the prediction); and optionally an LLM judge that
             says yes/no plus a 0-5 score, the protocol Video-ChatGPT introduced for
             open-ended video QA (and that most ActivityNet-QA numbers use).
  latency    retrieval and end-to-end answer time, p50 and p95, after a warm-up query.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Settings

ACTIVITYNET_TYPES = {0: "motion", 1: "spatial", 2: "temporal", 3: "yes/no", 4: "color", 5: "object", 6: "location", 7: "number", 8: "other"}
MIN_OVERLAP = 1.0  # seconds of real overlap for a chunk to count as relevant


@dataclass
class QAItem:
    video: str
    question: str
    answers: list[str] = field(default_factory=list)
    start: float | None = None
    end: float | None = None
    qtype: str = ""
    qid: str = ""


# ---------------------------------------------------------------- loading


def load_jsonl(path: str | Path) -> list[QAItem]:
    path = Path(path)
    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        video = Path(row["video"]).expanduser()
        if not video.is_absolute():
            video = path.parent / video
        answers = row.get("answers") or ([row["answer"]] if row.get("answer") else [])
        items.append(
            QAItem(
                video=str(video),
                question=row["question"],
                answers=[str(a) for a in answers],
                start=row.get("start"),
                end=row.get("end"),
                qtype=str(row.get("type", row.get("scene", ""))),
                qid=str(row.get("id", line_no)),
            )
        )
    return items


def _find_video(video_dir: Path, name: str) -> Path | None:
    stem = name[2:] if name.startswith("v_") else name
    for candidate in (name, stem, f"v_{stem}"):
        for match in sorted(video_dir.glob(f"{candidate}.*")):
            if match.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"):
                return match
    return None


def load_activitynet(questions: str | Path, answers: str | Path, video_dir: str | Path, limit: int | None = None) -> list[QAItem]:
    """ActivityNet-QA questions whose videos exist in video_dir (many YouTube originals are gone)."""
    video_dir = Path(video_dir)
    by_id = {a["question_id"]: a for a in json.loads(Path(answers).read_text(encoding="utf-8"))}
    items, missing = [], set()
    for q in json.loads(Path(questions).read_text(encoding="utf-8")):
        video = _find_video(video_dir, q["video_name"])
        if video is None:
            missing.add(q["video_name"])
            continue
        answer = by_id.get(q["question_id"], {})
        items.append(
            QAItem(
                video=str(video),
                question=q["question"].rstrip("?") + "?",
                answers=[str(answer.get("answer", ""))] if answer.get("answer") else [],
                qtype=ACTIVITYNET_TYPES.get(answer.get("type"), "other"),
                qid=q["question_id"],
            )
        )
        if limit and len(items) >= limit:
            break
    if not items:
        raise ValueError(f"none of the ActivityNet-QA videos were found in {video_dir} ({len(missing)} missing)")
    return items


# ---------------------------------------------------------------- scoring

_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}


def _words_to_digits(tokens: list[str]) -> list[str]:
    """['more', 'than', 'two', 'hundred', 'years'] -> ['more', 'than', '200', 'years']."""
    out: list[str] = []
    total = current = 0
    active = False
    for token in tokens + [""]:
        if token in _UNITS or token in _TENS:
            current += _UNITS.get(token, 0) + _TENS.get(token, 0)
            active = True
        elif token in _SCALES and active:
            if _SCALES[token] == 100:
                current *= 100
            else:
                total += current * _SCALES[token]
                current = 0
        elif token == "and" and active:
            continue
        else:
            if active:
                out.append(str(total + current))
                total = current = 0
                active = False
            if token:
                out.append(token)
    return out


def normalize(text: str) -> str:
    tokens = re.sub(r"[^a-z0-9.]+", " ", text.lower()).replace(". ", " ").split()
    tokens = [t.strip(".") for t in tokens if t.strip(".")]
    return " ".join(t for t in _words_to_digits(tokens) if t not in ("a", "an", "the"))


def contains_match(prediction: str, answers: list[str]) -> bool:
    predicted = normalize(prediction)
    for answer in answers:
        gold = normalize(answer)
        if not gold:
            continue
        if gold in ("yes", "no"):
            if predicted.split()[:1] == [gold]:
                return True
        elif f" {gold} " in f" {predicted} ":
            return True
    return False


def is_relevant(start: float, end: float, gold_start: float, gold_end: float) -> bool:
    if gold_end - gold_start < MIN_OVERLAP:  # a moment rather than a span
        middle = (gold_start + gold_end) / 2
        return start - 0.5 <= middle <= end + 0.5
    overlap = min(end, gold_end) - max(start, gold_start)
    return overlap >= min(MIN_OVERLAP, 0.5 * (end - start))


def random_hit_rate(n_chunks: int, n_relevant: int, k: int) -> float:
    """Chance that k chunks picked at random include at least one relevant chunk."""
    k = min(k, n_chunks)
    if n_relevant <= 0 or n_chunks <= 0:
        return 0.0
    return 1.0 - math.comb(n_chunks - n_relevant, k) / math.comb(n_chunks, k)


JUDGE_PROMPT = """You are grading answers to questions about a video.

Question: {question}
Correct answer: {answer}
Predicted answer: {prediction}

Does the predicted answer mean the same as the correct answer? Small wording differences are fine.
Reply with "yes" or "no" followed by a score from 0 to 5 for how well it matches, for example: yes 4"""


def parse_judgement(text: str) -> tuple[bool | None, int | None]:
    verdict = re.search(r"\b(yes|no)\b", text.lower())
    score = re.search(r"\b([0-5])\b", text)
    return (verdict.group(1) == "yes" if verdict else None), (int(score.group(1)) if score else None)


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {"p50": round(float(np.percentile(values, 50)), 3), "p95": round(float(np.percentile(values, 95)), 3), "n": len(values)}


# ---------------------------------------------------------------- running


def evaluate(
    settings: Settings,
    items: list[QAItem],
    *,
    k: int = 5,
    answers: bool = False,
    judge: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict, list[dict]]:
    """Return (summary, per-question rows). Videos that aren't indexed yet get processed first."""
    from .llm import make_chat_model, message_text, split_reasoning
    from .pipeline import process_video
    from .qa import VideoQA
    from .retrieval import search
    from .store import VideoLibrary

    say = progress or (lambda message: None)
    library = VideoLibrary(settings)
    video_ids: dict[str, str] = {}
    for path in dict.fromkeys(item.video for item in items):
        say(f"indexing {Path(path).name}")
        video_ids[path] = process_video(path, settings).video_id

    first = next(iter(video_ids.values()))
    search(settings, first, "warm up", k=1)  # load the embedder before timing anything
    judge_llm = make_chat_model(settings.answer) if judge else None
    qa_cache: dict[str, VideoQA] = {}
    rows = []
    for number, item in enumerate(items, start=1):
        say(f"[{number}/{len(items)}] {item.question}")
        video_id = video_ids[item.video]
        chunks = library.chunks(video_id)
        row: dict = {"qid": item.qid, "type": item.qtype, "video_id": video_id, "question": item.question, "answers": item.answers}

        started = time.perf_counter()
        hits = search(settings, video_id, item.question, k=k)
        row["retrieval_ms"] = round((time.perf_counter() - started) * 1000, 2)
        row["retrieved"] = [[h.chunk.start, h.chunk.end] for h in hits]
        if item.start is not None and item.end is not None:
            relevant = [is_relevant(h.chunk.start, h.chunk.end, item.start, item.end) for h in hits]
            first_hit = relevant.index(True) + 1 if True in relevant else None
            n_relevant = sum(is_relevant(c.start, c.end, item.start, item.end) for c in chunks)
            row.update(
                gold=[item.start, item.end],
                hit_at_1=bool(relevant[:1] and relevant[0]),
                hit_at_k=first_hit is not None,
                reciprocal_rank=1.0 / first_hit if first_hit else 0.0,
                random_hit_at_1=random_hit_rate(len(chunks), n_relevant, 1),
                random_hit_at_k=random_hit_rate(len(chunks), n_relevant, k),
            )

        if answers:
            qa = qa_cache.setdefault(video_id, VideoQA(settings, video_id))
            answer = qa.ask(item.question)
            row.update(prediction=answer.text, answer_s=round(answer.total_ms / 1000, 2), citations=answer.citations, model=answer.model)
            if item.answers:
                row["contains_match"] = contains_match(answer.text, item.answers)
                if judge_llm is not None:
                    reply = judge_llm.invoke(JUDGE_PROMPT.format(question=item.question, answer=item.answers[0], prediction=answer.text))
                    verdict, score = parse_judgement(split_reasoning(message_text(reply.content))[1])
                    row.update(judge_correct=verdict, judge_score=score)
        rows.append(row)
    return summarize_rows(rows, k=k, settings=settings, n_videos=len(video_ids)), rows


def summarize_rows(rows: list[dict], *, k: int, settings: Settings, n_videos: int) -> dict:
    def mean(key: str, subset: list[dict]) -> float | None:
        values = [float(r[key]) for r in subset if r.get(key) is not None]
        return round(sum(values) / len(values), 4) if values else None

    spanned = [r for r in rows if "gold" in r]
    graded = [r for r in rows if "contains_match" in r]
    summary: dict = {
        "questions": len(rows),
        "videos": n_videos,
        "k": k,
        "retrieval": {
            "questions": len(spanned),
            "hit_at_1": mean("hit_at_1", spanned),
            "hit_at_k": mean("hit_at_k", spanned),
            "mrr": mean("reciprocal_rank", spanned),
            "random_hit_at_1": mean("random_hit_at_1", spanned),
            "random_hit_at_k": mean("random_hit_at_k", spanned),
        },
        "latency": {
            "retrieval_ms": _percentiles([r["retrieval_ms"] for r in rows]),
            "answer_s": _percentiles([r["answer_s"] for r in rows if "answer_s" in r]),
        },
        "settings": {"whisper_model": settings.processing.whisper_model, "embed_model": settings.processing.embed_model},
    }
    if graded:
        summary["answers"] = {
            "questions": len(graded),
            "model": graded[0].get("model"),
            "contains_match": mean("contains_match", graded),
            "judge_accuracy": mean("judge_correct", graded),
            "judge_score": mean("judge_score", graded),
        }
        by_type: dict[str, list[dict]] = defaultdict(list)
        for r in graded:
            by_type[r["type"] or "all"].append(r)
        if len(by_type) > 1:
            summary["answers"]["by_type"] = {t: {"n": len(rs), "contains_match": mean("contains_match", rs)} for t, rs in sorted(by_type.items())}
    return summary


def write_report(summary: dict, rows: list[dict], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rows_path = out_dir / f"eval-{stamp}.jsonl"
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out_dir / f"eval-{stamp}-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return rows_path


def format_summary(summary: dict) -> str:
    def num(value, pattern="{:.2f}"):
        return "-" if value is None else pattern.format(value)

    r, lat = summary["retrieval"], summary["latency"]
    lines = [f"Evaluated {summary['questions']} questions on {summary['videos']} video(s)", ""]
    if r["questions"]:
        lines += [
            f"Retrieval ({r['questions']} questions with a gold time span)",
            f"  hit@1   {num(r['hit_at_1'])}   (random ranking: {num(r['random_hit_at_1'])})",
            f"  hit@{summary['k']}   {num(r['hit_at_k'])}   (random ranking: {num(r['random_hit_at_k'])})",
            f"  MRR     {num(r['mrr'])}",
            "",
        ]
    if "answers" in summary:
        a = summary["answers"]
        lines += [f"Answers ({a['questions']} questions with a reference answer, {a['model']})", f"  contains-match  {num(a['contains_match'])}"]
        if a.get("judge_accuracy") is not None:
            lines.append(f"  LLM judge       {num(a['judge_accuracy'])}  (mean score {num(a['judge_score'], '{:.1f}')}/5)")
        for qtype, stats in (a.get("by_type") or {}).items():
            lines.append(f"    {qtype:<10} n={stats['n']:<4} contains-match {num(stats['contains_match'])}")
        lines.append("")
    lines.append("Latency")
    if lat["retrieval_ms"]:
        lines.append(f"  retrieval  p50 {lat['retrieval_ms']['p50']:.0f} ms   p95 {lat['retrieval_ms']['p95']:.0f} ms")
    if lat["answer_s"]:
        lines.append(f"  answer     p50 {lat['answer_s']['p50']:.1f} s    p95 {lat['answer_s']['p95']:.1f} s")
    return "\n".join(lines)
