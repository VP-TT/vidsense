"""Command-line interface: `vidsense <command> --help` for details."""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from dataclasses import replace

from .config import Settings, load_settings
from .schemas import VideoRecord
from .timeutil import format_range, format_ts


def _processing_overrides(args: argparse.Namespace, settings: Settings) -> Settings:
    changes = {
        "whisper_model": args.whisper_model,
        "language": args.language,
        "frame_fps": args.fps,
        "dtw_semantic_weight": args.semantic_weight,
    }
    changes = {k: v for k, v in changes.items() if v is not None}
    if args.translate:
        changes["translate"] = True
    if args.no_tags:
        changes["visual_tags"] = False
    return replace(settings, processing=replace(settings.processing, **changes))


def _answer_overrides(args: argparse.Namespace, settings: Settings) -> Settings:
    changes = {"provider": args.provider, "model": args.model, "ollama_url": args.ollama_url}
    changes = {k: v for k, v in changes.items() if v is not None}
    if "provider" in changes and "model" not in changes:
        changes["model"] = ""  # the new provider's default model
    if getattr(args, "k", None):
        changes["top_k"] = args.k
    return replace(settings, answer=replace(settings.answer, **changes))


def _ensure_llm(settings: Settings) -> Settings:
    """Fall back to retrieval-only mode, with a hint, when the LLM isn't reachable."""
    from .llm import check_llm

    ready, message = check_llm(settings.answer)
    if not ready and settings.answer.provider != "none":
        print(f"note: {message}\nShowing the best-matching moments instead.\n", file=sys.stderr)
        return replace(settings, answer=replace(settings.answer, provider="none"))
    return settings


def _progress_printer():
    from tqdm import tqdm

    bar = tqdm(total=100, bar_format="{bar:30} {n:3.0f}% {elapsed}  {desc}", file=sys.stderr)

    def update(fraction: float, stage: str, message: str) -> None:
        bar.n = round(100 * fraction, 1)
        bar.set_description_str(f"{stage}: {message}"[:60].ljust(60), refresh=False)
        bar.refresh()

    return bar, update


def _describe(record: VideoRecord) -> str:
    stats = record.stats
    lines = [
        f"{record.title}  [{record.video_id}]",
        f"  duration {format_ts(record.duration)}, language {record.language or '-'}, status {record.status}",
    ]
    if stats:
        cov = stats.get("keyframe_coverage") or {}
        timings = stats.get("timings", {})
        lines += [
            f"  {stats['segments']} transcript segments, {stats['sampled_frames']} sampled frames -> "
            f"{stats['keyframes']} keyframes, {stats['chunks']} chunks",
        ]
        if cov:
            lines.append(
                f"  keyframes cover {cov['coverage']:.0%} of sampled frames at CLIP cosine >= {cov['threshold']}"
                f" (mean best similarity {cov['mean_best_similarity']:.3f})"
            )
        dtw = stats.get("dtw", {})
        if dtw.get("full_cells"):
            lines.append(f"  DTW evaluated {dtw['cells']} of {dtw['full_cells']} cells (banded: {dtw['banded']})")
        if timings:
            lines.append("  timings: " + ", ".join(f"{k} {v}s" for k, v in timings.items()))
    if record.error:
        lines.append(f"  error: {record.error}")
    return "\n".join(lines)


def cmd_process(args: argparse.Namespace, settings: Settings) -> int:
    from .pipeline import process_video

    settings = _processing_overrides(args, settings)
    bar, update = _progress_printer()
    try:
        record = process_video(
            args.video,
            settings,
            title=args.title,
            placement="copy" if args.copy else "reference",
            force=args.force,
            progress=update,
        )
    finally:
        bar.close()
    print(_describe(record))
    return 0


def cmd_list(args: argparse.Namespace, settings: Settings) -> int:
    from .store import VideoLibrary

    records = VideoLibrary(settings).list(status=None)
    if not records:
        print("No videos yet. Try: vidsense demo && vidsense process demo/vidsense_demo.mp4")
        return 0
    for r in records:
        print(f"{r.video_id}  {format_ts(r.duration):>8}  {r.status:<10}  {r.title}")
    return 0


def cmd_show(args: argparse.Namespace, settings: Settings) -> int:
    from .store import VideoLibrary

    library = VideoLibrary(settings)
    record = library.resolve(args.video_ref)
    print(_describe(record))
    if record.status != "ready":
        return 0
    print("\nKeyframes:")
    for kf in library.keyframes(record.video_id):
        tags = ", ".join(f"{label} ({score:.2f})" for label, score in kf.tags) or "-"
        print(f"  {format_ts(kf.time)}  covers {format_range(kf.span_start, kf.span_end)}  {tags}")
    if args.chunks:
        print("\nChunks:")
        for chunk in library.chunks(record.video_id):
            print(f"  [{format_range(chunk.start, chunk.end)}] {textwrap.shorten(chunk.embedding_text(), 110)}")
    return 0


def cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    import time

    from .retrieval import search
    from .store import VideoLibrary

    record = VideoLibrary(settings).resolve(args.video_ref)
    search(settings, record.video_id, "warm up", k=1)  # load the embedder before timing
    start = time.perf_counter()
    hits = search(settings, record.video_id, args.query, k=args.k)
    elapsed = (time.perf_counter() - start) * 1000
    for hit in hits:
        chunk = hit.chunk
        print(f"{hit.rank}. [{format_range(chunk.start, chunk.end)}] score {hit.score:.3f}")
        print(textwrap.indent(textwrap.fill(chunk.embedding_text(), 96), "   "))
    print(f"({len(hits)} results in {elapsed:.0f} ms)")
    return 0


def cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    from .qa import VideoQA
    from .store import VideoLibrary

    settings = _ensure_llm(_answer_overrides(args, settings))
    record = VideoLibrary(settings).resolve(args.video_ref)
    answer = None
    thinking = False
    for kind, payload in VideoQA(settings, record.video_id).stream(args.question):
        if kind == "reasoning" and args.show_reasoning:
            if not thinking:
                print("[reasoning]", file=sys.stderr)
                thinking = True
            print(payload, end="", flush=True, file=sys.stderr)
        elif kind == "reasoning" and not thinking:
            print("(thinking...)", end="\r", flush=True, file=sys.stderr)
            thinking = True
        elif kind == "done":
            answer = payload
    if thinking:  # end the reasoning block, or erase the "(thinking...)" note
        print("\n" if args.show_reasoning else " " * 14 + "\r", end="", file=sys.stderr, flush=True)
    # Printed once complete, so reasoning that arrives without an opening tag never leaks into the answer.
    print(answer.text)
    print("\nSources:")
    for doc in answer.sources:
        meta = doc.metadata
        snippet = meta["transcript"] or "On screen: " + "; ".join(meta["tags"])
        print(f"  [{format_range(meta['start'], meta['end'])}] {textwrap.shorten(snippet, 90)}")
    cited = ", ".join(format_ts(t) for t in answer.citations) or "none"
    print(f"\nCited: {cited} | retrieval {answer.retrieval_ms:.0f} ms, total {answer.total_ms / 1000:.1f} s, {answer.model}")
    return 0


def cmd_summarize(args: argparse.Namespace, settings: Settings) -> int:
    from .store import VideoLibrary
    from .summarize import summarize

    settings = _answer_overrides(args, settings)
    if settings.answer.provider == "none":
        print("error: summaries need an LLM; use --provider ollama or --provider openai", file=sys.stderr)
        return 1
    if _ensure_llm(settings).answer.provider == "none":
        print("error: summaries need an LLM (see the note above)", file=sys.stderr)
        return 1
    record = VideoLibrary(settings).resolve(args.video_ref)
    summary = summarize(settings, record.video_id, force=args.force, progress=lambda f, m: print(f"  {m}...", file=sys.stderr))
    print(f"{record.title}\n\n{textwrap.fill(summary.overview, 96)}\n")
    for chapter in summary.chapters:
        print(f"  [{format_ts(chapter.start)}] {chapter.title}" + (f" - {chapter.description}" if chapter.description else ""))
    print(f"\n({summary.model}, {summary.created_at})")
    return 0


def cmd_delete(args: argparse.Namespace, settings: Settings) -> int:
    from .store import VideoLibrary

    library = VideoLibrary(settings)
    record = library.resolve(args.video_ref)
    library.delete(record.video_id)
    print(f"Deleted {record.title} [{record.video_id}] (the original video file was not touched)")
    return 0


def cmd_demo(args: argparse.Namespace, settings: Settings) -> int:
    from .demo import make_demo

    video, qa = make_demo(args.out)
    print(f"Wrote {video}\nWrote {qa}\nNext: vidsense process {video}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vidsense", description="Ask questions about a video and get answers with timestamps.")
    parser.add_argument("-v", "--verbose", action="store_true", help="show info-level logs")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="transcribe, pick keyframes, align and index a video")
    p.add_argument("video", help="path to a video (or audio) file")
    p.add_argument("--title", help="display title (default: file name)")
    p.add_argument("--whisper-model", help="tiny, base, small, medium, large-v3, large-v3-turbo, ...")
    p.add_argument("--language", help="spoken language code, e.g. en (default: auto-detect)")
    p.add_argument("--translate", action="store_true", help="translate speech to English while transcribing")
    p.add_argument("--fps", type=float, help="frames sampled per second (default 1)")
    p.add_argument("--semantic-weight", type=float, help="weight of CLIP similarity in the DTW cost, 0-1")
    p.add_argument("--no-tags", action="store_true", help="skip zero-shot visual tags")
    p.add_argument("--copy", action="store_true", help="copy the video into the data folder")
    p.add_argument("--force", action="store_true", help="re-process even if this file was processed before")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("list", help="list processed videos")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="show stats and keyframes of a video")
    p.add_argument("video_ref", help="video id, id prefix, title or 'latest'")
    p.add_argument("--chunks", action="store_true", help="also list the indexed chunks")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("search", help="find the moments that best match a query (no LLM)")
    p.add_argument("video_ref")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5, help="number of results")
    p.set_defaults(func=cmd_search)

    def add_llm_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--provider", choices=["ollama", "openai", "none"], help="LLM provider (default from settings)")
        p.add_argument("--model", help="model name, e.g. deepseek-r1:8b or gpt-4o-mini")
        p.add_argument("--ollama-url", help="Ollama server URL (default http://localhost:11434)")

    p = sub.add_parser("ask", help="answer a question about a video, with timestamps")
    p.add_argument("video_ref")
    p.add_argument("question")
    p.add_argument("-k", type=int, help="number of chunks to retrieve (default 5)")
    p.add_argument("--show-reasoning", action="store_true", help="print the model's reasoning (DeepSeek-R1)")
    add_llm_flags(p)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("summarize", help="summarize a video into an overview and chapters")
    p.add_argument("video_ref")
    p.add_argument("--force", action="store_true", help="regenerate instead of using the cached summary")
    add_llm_flags(p)
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("delete", help="remove a video's index and artifacts")
    p.add_argument("video_ref")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("demo", help="generate a synthetic demo video and QA file")
    p.add_argument("--out", default="demo", help="output folder (default: demo)")
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.func(args, load_settings())
    except (LookupError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
