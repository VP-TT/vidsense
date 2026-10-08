"""Streamlit UI. Run `streamlit run app.py` from the repo root, or `vidsense ui`.

The sidebar holds the library, uploads and settings. The main area shows the player,
its scenes and the processing details on the left, and tabs on the right (Ask, Summary,
Moments, Transcript). Every timestamp is a button that moves the player.

Streamlit re-runs the script on every interaction, and drawing the player re-reads the
video file, so the interactive tabs are fragments: asking a question or searching
re-runs only that tab. A jump re-runs the whole page, which is what moves the player.
"""

from __future__ import annotations

import textwrap
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitAPIException

from vidsense.config import DEFAULT_MODELS, PROVIDERS, AnswerConfig, ProcessingConfig, Settings, load_settings
from vidsense.jobs import JobManager
from vidsense.schemas import Chunk, VideoRecord
from vidsense.store import VideoLibrary
from vidsense.timeutil import format_range, format_ts

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"]
UPLOAD_TYPES = ["mp4", "mov", "m4v", "webm", "mkv", "avi", "mp3", "m4a", "wav"]
BROWSER_FORMATS = {".webm": "video/webm", ".ogv": "video/ogg"}


# ---------------------------------------------------------------- shared resources


@st.cache_resource
def base_settings() -> Settings:
    return load_settings()


@st.cache_resource
def job_manager() -> JobManager:
    return JobManager()


@st.cache_resource
def warm_up(embed_model: str) -> None:
    """Load the text embedder in the background so the first question doesn't wait for it."""
    from vidsense.embed import load_embedder

    threading.Thread(target=load_embedder, args=(embed_model,), daemon=True).start()


@st.cache_data(ttl=10, show_spinner=False)
def llm_status(provider: str, model: str, ollama_url: str) -> tuple[bool, str]:
    from vidsense.llm import check_llm

    return check_llm(AnswerConfig(provider=provider, model=model, ollama_url=ollama_url))


# ---------------------------------------------------------------- jumping


def seek(seconds: float) -> None:
    # The player takes whole seconds; rounding keeps it within half a second of the moment
    # (about Whisper's own timestamp accuracy). The exact time feeds the "current moment" card.
    target = max(0, round(seconds))
    if st.session_state.seek == target:  # the player only seeks when start_time changes
        target = target - 1 if target > 0 else target + 1
    st.session_state.seek = target
    st.session_state.seek_exact = max(0.0, float(seconds))


def rerun_fragment() -> None:
    """Re-run only the current fragment; during a full-page run that isn't allowed, so re-run the page."""
    try:
        st.rerun(scope="fragment")
    except StreamlitAPIException:
        st.rerun()


def jump_button(seconds: float, key: str, label: str | None = None, hint: str | None = None) -> None:
    tooltip = f"{hint} · play from {format_ts(seconds)}" if hint else f"Play from {format_ts(seconds)}"
    if st.button(label or format_ts(seconds), key=key, icon=":material/play_arrow:", help=tooltip):
        seek(seconds)
        st.rerun()  # inside a fragment this re-runs the whole page, so the player moves


# ---------------------------------------------------------------- sidebar


def processing_options(settings: Settings) -> ProcessingConfig:
    p = settings.processing
    with st.expander("Processing options", icon=":material/tune:"):
        model = st.selectbox(
            "Whisper model",
            WHISPER_MODELS,
            index=WHISPER_MODELS.index(p.whisper_model) if p.whisper_model in WHISPER_MODELS else 4,
            help="Larger is more accurate and slower. large-v3-turbo is the default; base is fine for quick tests.",
            key="opt_whisper",
        )
        language = st.text_input("Spoken language", value=p.language or "", placeholder="auto-detect, or a code like en", key="opt_language")
        translate = st.checkbox("Translate speech to English", value=p.translate, key="opt_translate")
        fps = st.slider("Frames sampled per second", 0.25, 2.0, float(p.frame_fps), 0.25, key="opt_fps")
        spacing = st.slider("Seconds per keyframe (before merging duplicates)", 2.0, 30.0, float(p.seconds_per_keyframe), 1.0, key="opt_spk")
        tags = st.toggle("Label keyframes with CLIP", value=p.visual_tags, key="opt_tags")
        semantic = st.slider(
            "CLIP weight in the DTW cost",
            0.0,
            0.6,
            float(p.dtw_semantic_weight),
            0.05,
            help="0 aligns keyframes and speech by timestamps only.",
            key="opt_semantic",
        )
        st.caption("These apply to the next video you process.")
    return replace(
        p,
        whisper_model=model,
        language=language.strip() or None,
        translate=translate,
        frame_fps=fps,
        seconds_per_keyframe=spacing,
        visual_tags=tags,
        dtw_semantic_weight=semantic,
    )


def answer_options(settings: Settings) -> AnswerConfig:
    a = settings.answer
    ready, message = llm_status(a.provider, a.resolved_model, a.ollama_url)
    with st.expander("Answer model", icon=":material/smart_toy:", expanded=not ready):
        provider = st.radio(
            "Provider",
            PROVIDERS,
            index=PROVIDERS.index(a.provider),
            horizontal=True,
            format_func={"ollama": "Ollama (local)", "openai": "OpenAI", "none": "None"}.get,
            key="llm_provider",
        )
        model = st.text_input(
            "Model",
            value=a.model if provider == a.provider else "",
            placeholder=DEFAULT_MODELS[provider] or "retrieval only",
            disabled=provider == "none",
            key=f"llm_model_{provider}",
        )
        top_k = st.slider("Excerpts per answer", 1, 10, a.top_k, key="llm_top_k")
        temperature = st.slider("Temperature", 0.0, 1.0, a.temperature, 0.05, key="llm_temperature")
        history = st.toggle("Use the conversation for follow-up questions", value=a.use_history, key="llm_history")
        cfg = replace(a, provider=provider, model=model.strip(), top_k=top_k, temperature=temperature, use_history=history)
        ready, message = llm_status(cfg.provider, cfg.resolved_model, cfg.ollama_url)
        if ready:
            st.success(message, icon=":material/check_circle:")
        else:
            st.warning(message, icon=":material/warning:")
    return cfg


def start_job(source: Path, title: str, settings: Settings, *, placement: str, force: bool = False) -> None:
    job = job_manager().submit(source, title, settings, placement=placement, force=force)
    st.session_state.job_id = job.id
    st.rerun()


def add_video(settings: Settings) -> None:
    st.subheader("Add a video", anchor=False)
    upload, disk, demo = st.tabs(["Upload", "From disk", "Demo"])
    with upload:
        file = st.file_uploader("Video or audio file", type=UPLOAD_TYPES, label_visibility="collapsed")
        if st.button("Process video", type="primary", disabled=file is None, width="stretch"):
            path = settings.uploads_dir / f"{uuid.uuid4().hex}{Path(file.name).suffix.lower()}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(file.getbuffer())
            start_job(path, Path(file.name).stem, settings, placement="move")
    with disk:
        raw = st.text_input("Path to a video file", placeholder="/Users/you/Movies/lecture.mp4")
        if st.button("Process file", disabled=not raw.strip(), width="stretch"):
            path = Path(raw.strip().strip("\"'")).expanduser()
            if path.is_file():
                start_job(path, path.stem, settings, placement="reference")
            else:
                st.error("There's no file at that path.")
        st.caption("Large files are better added this way: the video stays where it is.")
    with demo:
        st.caption("A one-minute synthetic tour with narration, generated on this machine. Good for a first try.")
        if st.button("Make and process the demo", width="stretch"):
            from vidsense.demo import make_demo

            try:
                with st.spinner("Generating the demo video..."):
                    video, _ = make_demo(settings.data_dir / "demo")
            except RuntimeError as exc:
                st.error(str(exc))
            else:
                start_job(video, "VidSense demo", settings, placement="reference")


def sidebar(settings: Settings, library: VideoLibrary) -> tuple[Settings, VideoRecord | None]:
    with st.sidebar:
        st.title(":material/movie: VidSense", anchor=False)
        st.caption("Ask a video anything. Answers point to the exact moment.")
        videos = library.list()
        record = None
        if videos:
            by_id = {v.video_id: v for v in videos}
            if st.session_state.get("pending_video") in by_id:
                st.session_state.video_select = st.session_state.pop("pending_video")
            if st.session_state.get("video_select") not in by_id:
                st.session_state.video_select = videos[0].video_id
            chosen = st.selectbox(
                "Your videos",
                list(by_id),
                format_func=lambda vid: f"{by_id[vid].title} · {format_ts(by_id[vid].duration)}",
                key="video_select",
            )
            record = by_id[chosen]
        add_video(replace(settings, processing=processing_from_state(settings)))
        processing = processing_options(settings)
        answer = answer_options(settings)
    return replace(settings, processing=processing, answer=answer), record


def processing_from_state(settings: Settings) -> ProcessingConfig:
    """The sidebar's processing options, read from session state so buttons above them can use them."""
    ss, p = st.session_state, settings.processing
    return replace(
        p,
        whisper_model=ss.get("opt_whisper", p.whisper_model),
        language=(ss.get("opt_language") or "").strip() or p.language,
        translate=ss.get("opt_translate", p.translate),
        frame_fps=ss.get("opt_fps", p.frame_fps),
        seconds_per_keyframe=ss.get("opt_spk", p.seconds_per_keyframe),
        visual_tags=ss.get("opt_tags", p.visual_tags),
        dtw_semantic_weight=ss.get("opt_semantic", p.dtw_semantic_weight),
    )


# ---------------------------------------------------------------- processing status


@st.fragment(run_every=1.0)
def job_panel() -> None:
    jobs = job_manager()
    job = jobs.get(st.session_state.job_id)
    if job is None:
        running = jobs.active()
        job = running[0] if running else None
        if job is None:
            return
        st.session_state.job_id = job.id  # a refreshed page picks up the job it started
    if job.status == "done":
        st.session_state.job_id = None
        st.session_state.pending_video = job.video_id
        st.session_state.seek = 0
        st.toast(f"{job.title} is ready", icon=":material/check_circle:")
        st.rerun()
    with st.container(border=True):
        if job.status == "error":
            st.error(f"Processing {job.title} failed: {job.error}", icon=":material/error:")
            if st.button("Dismiss", key=f"dismiss-{job.id}"):
                st.session_state.job_id = None
                st.rerun()
            return
        st.markdown(f"**Processing {job.title}**")
        label = f"{job.stage}: {job.message}" if job.stage else job.message
        st.progress(min(job.progress, 1.0), text=label)
        queued = len(job_manager().active()) - 1
        st.caption(
            "Whisper transcribes while CLIP picks keyframes; then DTW aligns them and the chunks are indexed."
            + (f" {queued} more video(s) queued." if queued > 0 else "")
        )


# ---------------------------------------------------------------- player


def keyframe_image(library: VideoLibrary, video_id: str, keyframe_ids: list[int]) -> str | None:
    keyframes = {k.id: k for k in library.keyframes(video_id)}
    for kid in keyframe_ids:
        if kid in keyframes:
            path = library.path(video_id, keyframes[kid].image)
            if path.is_file():
                return str(path)
    return None


def player(record: VideoRecord, library: VideoLibrary) -> None:
    source = Path(record.source_path)
    if not source.is_file():
        st.error(f"Can't find the video file at {source}. The index still works; process the file again to watch it here.")
        return
    start = st.session_state.seek
    if not record.has_video:
        st.audio(str(source), start_time=start)
        return
    if source.suffix.lower() in (".mkv", ".avi"):
        st.caption("Browsers can't play MKV or AVI files. Convert to MP4 to watch here; search and answers still work.")
    vtt = library.path(record.video_id, "transcript.vtt")
    st.video(
        str(source),
        format=BROWSER_FORMATS.get(source.suffix.lower(), "video/mp4"),
        start_time=start,
        subtitles={"Transcript": str(vtt)} if record.stats.get("segments") else None,
    )


def chunk_at(chunks: list[Chunk], seconds: float) -> Chunk | None:
    containing = [c for c in chunks if c.start <= seconds <= c.end]
    if containing:  # on a boundary, the chunk that starts there
        return max(containing, key=lambda c: c.start)
    return min(chunks, key=lambda c: min(abs(c.start - seconds), abs(c.end - seconds)), default=None)


def moment_card(record: VideoRecord, library: VideoLibrary) -> None:
    seconds = st.session_state.seek_exact
    chunk = chunk_at(library.chunks(record.video_id), seconds) if st.session_state.seek else None
    if chunk is None:
        st.caption("Click any timestamp to jump there. The transcript plays as captions.")
        return
    with st.container(border=True):
        st.caption(f"At {format_ts(seconds)} · {format_range(chunk.start, chunk.end)}")
        image = keyframe_image(library, record.video_id, chunk.keyframe_ids)
        text, tags = chunk.transcript or "(no speech)", ", ".join(chunk.tags[:3])
        if image:
            left, right = st.columns([1, 3])
            left.image(image, width="stretch")
            right.write(text)
            if tags:
                right.caption(f"On screen: {tags}")
        else:
            st.write(text)


# ---------------------------------------------------------------- tabs


def source_rows(sources: list[dict], record: VideoRecord, library: VideoLibrary, key: str, show_score: bool = False) -> None:
    for j, src in enumerate(sources):
        image = keyframe_image(library, record.video_id, src.get("keyframe_ids", []))
        left, right = st.columns([1, 3])
        if image:
            left.image(image, width="stretch")
        with right:
            jump_button(src["start"], key=f"{key}-{j}", label=format_range(src["start"], src["end"]))
            st.caption(textwrap.shorten(src["transcript"] or "(no speech)", 240))
            details = []
            if src.get("tags"):
                details.append("On screen: " + ", ".join(src["tags"][:3]))
            if show_score:
                details.append(f"similarity {src['score']:.2f}")
            if details:
                st.caption(" · ".join(details))


def render_message(message: dict, index: int, record: VideoRecord, library: VideoLibrary) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] != "assistant" or message.get("error"):
            return
        cited = message.get("citations") or []
        times = cited or [s["start"] for s in message.get("sources", [])[:3]]
        if times:
            if not cited:
                st.caption("Most relevant moments:")
            with st.container(horizontal=True, gap="small"):
                for j, seconds in enumerate(times[:8]):
                    jump_button(seconds, key=f"cite-{record.video_id}-{index}-{j}")
        if message.get("sources"):
            with st.expander(f"Sources ({len(message['sources'])})"):
                source_rows(message["sources"], record, library, key=f"src-{record.video_id}-{index}")
        if message.get("reasoning"):
            with st.expander("Model reasoning"):
                st.markdown(message["reasoning"])
        st.caption(f"{message['model']} · {message['seconds']:.1f} s")


def stream_answer(record: VideoRecord, settings: Settings, question: str, history: list[dict]) -> dict:
    from vidsense.qa import VideoQA

    status = st.status("Searching the video...", expanded=False)
    thinking = status.empty()
    output = st.empty()
    reasoning, text, final = "", "", None
    try:
        qa = VideoQA(settings, record.video_id)
        for kind, payload in qa.stream(question, history):
            if kind == "sources":
                status.update(label=f"Found {len(payload)} relevant moments" + (", writing the answer..." if qa.llm else ""))
            elif kind == "reasoning":
                reasoning += payload
                status.update(label="Reasoning...")
                thinking.markdown(reasoning[-1500:])
            elif kind == "answer":
                text += payload
                output.markdown(text + " ▌")
            else:
                final = payload
    except Exception as exc:  # e.g. the LLM server went away mid-answer
        status.update(label="Couldn't answer", state="error")
        return {"role": "assistant", "content": f"Couldn't answer: {exc}", "error": True}
    status.update(label=f"Answered in {final.total_ms / 1000:.1f} s", state="complete")
    return {
        "role": "assistant",
        "content": final.text,
        "reasoning": final.reasoning,
        "citations": final.citations,
        "sources": [doc.metadata for doc in final.sources],
        "model": final.model,
        "seconds": final.total_ms / 1000,
    }


def suggestions(library: VideoLibrary, record: VideoRecord) -> list[str]:
    questions = ["What is this video about?", "What are the key takeaways?"]
    labels = [tag for k in library.keyframes(record.video_id) for tag in k.tags[:1]]
    if labels:  # the label CLIP was most sure about makes a good "find the moment" question
        questions.append(f"When does it show {max(labels, key=lambda tag: tag[1])[0]}?")
    return questions


@st.fragment
def ask_tab(record: VideoRecord, settings: Settings) -> None:
    library = VideoLibrary(settings)
    history = st.session_state.chats.setdefault(record.video_id, [])
    a = settings.answer
    ready, message = llm_status(a.provider, a.resolved_model, a.ollama_url)
    if not ready:
        settings = replace(settings, answer=replace(a, provider="none"))
        st.info(f"{message} Until then, answers list the best-matching moments.", icon=":material/info:")

    box = st.container(height=520, border=False)
    picked = None
    with box:
        if not history:
            st.caption("Ask anything about the video, for example:")
            for i, suggestion in enumerate(suggestions(library, record)):
                if st.button(suggestion, key=f"suggest-{record.video_id}-{i}", type="tertiary"):
                    picked = suggestion
        for i, msg in enumerate(history):
            render_message(msg, i, record, library)

    question = st.chat_input("Ask about this video", key=f"chat-{record.video_id}") or picked
    if question:
        with box:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                reply = stream_answer(record, settings, question, history)
        history += [{"role": "user", "content": question}, reply]
        rerun_fragment()
    if history and st.button("Clear conversation", key=f"clear-{record.video_id}", type="tertiary", icon=":material/delete_sweep:"):
        history.clear()
        rerun_fragment()


@st.fragment
def summary_tab(record: VideoRecord, settings: Settings) -> None:
    from vidsense.summarize import summarize

    library = VideoLibrary(settings)
    a = settings.answer
    ready, message = llm_status(a.provider, a.resolved_model, a.ollama_url)
    cached = library.read_json(record.video_id, "summary.json", default={}).get(a.resolved_model)
    if cached is None:
        if not ready:
            st.info(f"Summaries need an LLM. {message}", icon=":material/info:")
            return
        st.write("Get an overview of the whole video, split into chapters you can jump to.")
        if st.button("Summarize this video", type="primary", icon=":material/summarize:"):
            bar = st.progress(0.0, text="Reading the transcript...")
            try:
                summarize(settings, record.video_id, progress=lambda f, m: bar.progress(f, text=m))
            except Exception as exc:
                st.error(f"Couldn't summarize: {exc}")
                return
            rerun_fragment()
        return

    st.markdown(cached["overview"])
    for i, chapter in enumerate(cached["chapters"]):
        left, right = st.columns([1, 4], vertical_alignment="center")
        with left:
            jump_button(chapter["start"], key=f"chapter-{record.video_id}-{i}")
        right.markdown(f"**{chapter['title']}**" + (f" — {chapter['description']}" if chapter["description"] else ""))
    if not cached["chapters"]:
        st.caption("The model didn't return chapters in the expected format, so this is its full summary.")
    st.caption(f"Generated by {cached['model']}")
    if ready and st.button("Regenerate", icon=":material/refresh:", type="tertiary"):
        with st.spinner("Summarizing..."):
            summarize(settings, record.video_id, force=True)
        rerun_fragment()


@st.fragment
def moments_tab(record: VideoRecord, settings: Settings) -> None:
    from vidsense.retrieval import search

    query = st.text_input("Find a moment", placeholder="e.g. the part about the desert", key=f"moments-{record.video_id}")
    if not query.strip():
        st.caption("Semantic search over what was said and what was on screen. No LLM needed.")
        return
    started = time.perf_counter()
    hits = search(settings, record.video_id, query, k=max(5, settings.answer.top_k))
    st.caption(f"{len(hits)} moments in {(time.perf_counter() - started) * 1000:.0f} ms")
    sources = [
        {"start": h.chunk.start, "end": h.chunk.end, "transcript": h.chunk.transcript, "tags": h.chunk.tags, "keyframe_ids": h.chunk.keyframe_ids, "score": h.score}
        for h in hits
    ]
    source_rows(sources, record, VideoLibrary(settings), key=f"moment-{record.video_id}", show_score=True)


@st.fragment
def transcript_tab(record: VideoRecord, library: VideoLibrary) -> None:
    import pandas as pd

    segments = library.segments(record.video_id)
    if not segments:
        st.info("No speech was detected in this video.", icon=":material/info:")
        return
    needle = st.text_input("Filter", placeholder="Filter the transcript", label_visibility="collapsed", key=f"tfilter-{record.video_id}")
    rows = [(s.start, format_ts(s.start), s.text) for s in segments if needle.lower() in s.text.lower()]
    table = pd.DataFrame(rows, columns=["start", "Time", "Said"])
    st.caption("Select a row to jump there.")
    event = st.dataframe(
        table[["Time", "Said"]],
        hide_index=True,
        height=440,
        on_select="rerun",
        selection_mode="single-row",
        key=f"transcript-{record.video_id}",
        column_config={"Time": st.column_config.TextColumn(width="small"), "Said": st.column_config.TextColumn(width="large")},
    )
    if event.selection.rows:
        row = event.selection.rows[0]
        marker = (record.video_id, needle, row)
        if st.session_state.get("transcript_jump") != marker:  # react to a new selection only
            st.session_state.transcript_jump = marker
            seek(table.iloc[row]["start"])
            st.rerun()
    vtt = library.path(record.video_id, "transcript.vtt")
    st.download_button("Download captions (.vtt)", vtt.read_bytes(), file_name=f"{record.title}.vtt", icon=":material/download:")


def scenes_strip(record: VideoRecord, library: VideoLibrary) -> None:
    keyframes = library.keyframes(record.video_id)
    if not keyframes:
        return
    st.markdown("**Scenes**")
    st.caption(
        f"{len(keyframes)} keyframes picked from {record.stats.get('sampled_frames', '?')} sampled frames by "
        "temporal K-means on CLIP embeddings. Hover a time to see what CLIP recognised."
    )
    holder = st.container(height=330, border=False) if len(keyframes) > 12 else st.container()
    row = holder.container(horizontal=True, wrap=True, gap="small")
    for i, kf in enumerate(keyframes):
        tile = row.container(width=124, gap="small")
        tile.image(str(library.path(record.video_id, kf.image)), width="stretch")
        with tile:
            jump_button(kf.time, key=f"kf-{record.video_id}-{i}", hint=", ".join(label for label, _ in kf.tags[:3]) or None)


def dtw_chart(record: VideoRecord, library: VideoLibrary) -> None:
    """The DTW warping path: each dot pairs a transcript segment with the keyframe it was matched to."""
    import altair as alt
    import pandas as pd

    alignment = library.read_json(record.video_id, "alignment.json", default={})
    if not alignment.get("path"):
        st.caption("No alignment: the video has no speech or no frames.")
        return
    segments, keyframes = library.segments(record.video_id), library.keyframes(record.video_id)
    rows = []
    for step, ((i, j), cost) in enumerate(zip(alignment["path"], alignment["path_costs"])):
        middle = (segments[j].start + segments[j].end) / 2
        rows.append(
            {
                "step": step,
                "speech": round(middle, 1),
                "keyframe": keyframes[i].time,
                "Speech at": format_ts(middle),
                "Keyframe at": format_ts(keyframes[i].time),
                "Cost": cost,
                "Said": textwrap.shorten(segments[j].text, 70),
            }
        )
    data = pd.DataFrame(rows)
    dark = (st.context.theme or {}).get("type") == "dark"
    series, surface, reference = ("#3987e5", "#0e1117", "#383835") if dark else ("#2a78d6", "#ffffff", "#c3c2b7")
    top = float(max(data["speech"].max(), data["keyframe"].max()))

    x = alt.X("speech:Q", title="Speech (seconds)", scale=alt.Scale(domain=[0, top]))
    y = alt.Y("keyframe:Q", title="Keyframe (seconds)", scale=alt.Scale(domain=[0, top]))
    hover = alt.selection_point(on="pointerover", nearest=True, fields=["step"], empty=False, clear="pointerout")
    tooltip = ["Said", "Speech at", "Keyframe at", alt.Tooltip("Cost:Q", format=".2f")]
    in_sync = alt.Chart(pd.DataFrame({"t": [0.0, top]})).mark_line(color=reference, strokeWidth=1).encode(x="t:Q", y="t:Q")
    base = alt.Chart(data).encode(x=x, y=y, order="step:Q")
    path = base.mark_line(color=series, strokeWidth=2, strokeJoin="round", strokeCap="round")
    dots = base.mark_circle(color=series, opacity=1, stroke=surface, strokeWidth=2).encode(
        size=alt.condition(hover, alt.value(160), alt.value(64))
    )
    targets = base.mark_circle(size=600, opacity=0).encode(tooltip=tooltip).add_params(hover)  # 24px hit areas
    st.markdown("**DTW alignment path**")
    st.altair_chart((in_sync + path + dots + targets).properties(height=300), width="stretch")
    st.caption(
        "Each dot pairs a transcript segment with the keyframe DTW matched it to; the grey diagonal is perfect sync. "
        + (
            f"The {record.processing.get('dtw_band_seconds', 60):.0f} s band cut the search to {alignment['cells']} of {alignment['full_cells']} cells."
            if alignment["cells"] < alignment["full_cells"]
            else f"DTW evaluated all {alignment['full_cells']} cells; the band only prunes on longer videos."
        )
    )
    with st.expander("Alignment table"):
        st.dataframe(data[["Speech at", "Keyframe at", "Cost", "Said"]], hide_index=True)


def details_panel(record: VideoRecord, library: VideoLibrary, settings: Settings) -> None:
    with st.expander("How this video was processed", icon=":material/analytics:"):
        details(record, library, settings)


def details(record: VideoRecord, library: VideoLibrary, settings: Settings) -> None:
    stats = record.stats
    coverage = stats.get("keyframe_coverage") or {}
    timings = stats.get("timings", {})
    a, b, c, d = st.columns(4)
    a.metric("Keyframes", stats.get("keyframes", 0), help=f"from {stats.get('sampled_frames', 0)} sampled frames")
    b.metric("Coverage", f"{coverage.get('coverage', 0):.0%}", help="Share of sampled frames within CLIP cosine 0.9 of a keyframe")
    c.metric("Chunks", stats.get("chunks", 0))
    d.metric("Processing", f"{timings.get('total', 0):.0f} s")
    dtw_chart(record, library)
    with st.expander("Timings and settings"):
        st.write({"timings (s)": timings, "devices": stats.get("devices", {}), "settings": record.processing})
    with st.expander("Re-process or delete"):
        st.caption("Re-processing uses the processing options in the sidebar.")
        if st.button("Re-process this video", icon=":material/refresh:"):
            start_job(Path(record.source_path), record.title, settings, placement="reference", force=True)
        sure = st.checkbox("Delete the index, keyframes and transcript for this video", key=f"delete-sure-{record.video_id}")
        if st.button("Delete", disabled=not sure, icon=":material/delete:"):
            library.delete(record.video_id)
            st.session_state.chats.pop(record.video_id, None)
            st.session_state.pop("video_select", None)
            st.rerun()


# ---------------------------------------------------------------- pages


def welcome(settings: Settings) -> None:
    st.title("Ask a video anything", anchor=False)
    st.markdown("Upload a video, ask questions in plain language, and get answers that cite the exact moments. Click a timestamp to jump there.")
    steps = [
        (":material/upload: **1. Add a video**", "Upload one in the sidebar, point to a file on disk, or make the one-minute demo."),
        (":material/hourglass_top: **2. Let it process**", "Whisper transcribes the audio while CLIP picks keyframes; DTW aligns them and ChromaDB indexes the result."),
        (":material/chat: **3. Ask and jump**", "Answers cite [mm:ss] timestamps. Summaries, moment search and the transcript all link into the video."),
    ]
    for column, (title, body) in zip(st.columns(3), steps):
        with column.container(border=True):
            st.markdown(title)
            st.caption(body)
    a = settings.answer
    ready, message = llm_status(a.provider, a.resolved_model, a.ollama_url)
    if not ready:
        st.info(f"{message} VidSense works without an LLM too: you'll get the matching moments instead of a written answer.", icon=":material/info:")


def video_page(record: VideoRecord, settings: Settings) -> None:
    library = VideoLibrary(settings)
    warm_up(record.processing.get("embed_model", settings.processing.embed_model))
    if st.session_state.get("last_video") != record.video_id:
        st.session_state.last_video = record.video_id
        st.session_state.seek, st.session_state.seek_exact = 0, 0.0
    stats = record.stats
    st.header(record.title, anchor=False)
    st.caption(
        " · ".join(
            [
                format_ts(record.duration),
                record.language.upper() if record.language else "no speech",
                f"{stats.get('segments', 0)} transcript segments",
                f"{stats.get('keyframes', 0)} keyframes",
                f"{stats.get('chunks', 0)} chunks",
            ]
        )
    )
    left, right = st.columns([6, 5], gap="large")
    with left:
        player(record, library)
        moment_card(record, library)
        scenes_strip(record, library)
        details_panel(record, library, settings)
    with right:
        ask, summary, moments, transcript = st.tabs(
            [":material/chat: Ask", ":material/summarize: Summary", ":material/search: Moments", ":material/subtitles: Transcript"]
        )
        with ask:
            ask_tab(record, settings)
        with summary:
            summary_tab(record, settings)
        with moments:
            moments_tab(record, settings)
        with transcript:
            transcript_tab(record, library)


def main() -> None:
    st.set_page_config(page_title="VidSense", page_icon=":material/movie:", layout="wide")
    st.session_state.setdefault("seek", 0)
    st.session_state.setdefault("seek_exact", 0.0)
    st.session_state.setdefault("chats", {})
    st.session_state.setdefault("job_id", None)

    settings = base_settings()
    library = VideoLibrary(settings)
    settings, record = sidebar(settings, library)
    job_panel()
    if record is None:
        welcome(settings)
    else:
        video_page(record, settings)


if __name__ == "__main__":
    main()
