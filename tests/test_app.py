"""Smoke tests for the Streamlit UI with Streamlit's AppTest (no models are loaded)."""

import json
from pathlib import Path

import av
import numpy as np
import pytest
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

import vidsense.app

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDSENSE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VIDSENSE_LLM_PROVIDER", "none")
    monkeypatch.setattr(vidsense.app, "warm_up", lambda model: None)  # don't load MiniLM
    st.cache_resource.clear()
    st.cache_data.clear()
    yield tmp_path
    st.cache_resource.clear()


def _tiny_video(path: Path) -> None:
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=2)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for i in range(4):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i * 60, dtype=np.uint8), format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)


def _processed_video(root: Path) -> None:
    folder = root / "videos" / "abc123"
    (folder / "keyframes").mkdir(parents=True)
    _tiny_video(folder / "source.mp4")
    Image.new("RGB", (64, 36), "navy").save(folder / "keyframes" / "kf_0000.jpg")
    files = {
        "manifest.json": {
            "video_id": "abc123", "title": "Tiny", "source_path": str(folder / "source.mp4"), "duration": 2.0,
            "language": "en", "created_at": "2026-01-01T00:00:00+00:00", "status": "ready",
            "processing": {"embed_model": "x", "dtw_band_seconds": 60},
            "stats": {"segments": 1, "keyframes": 1, "chunks": 1, "sampled_frames": 2,
                      "keyframe_coverage": {"coverage": 1.0}, "timings": {"total": 1.0}},
        },
        "transcript.json": {"language": "en", "segments": [{"id": 0, "start": 0.2, "end": 1.8, "text": "Hello there."}]},
        "keyframes.json": [{"id": 0, "time": 0.5, "span_start": 0.0, "span_end": 2.0, "frame_count": 2,
                            "image": "keyframes/kf_0000.jpg", "tags": [["a title card with large text", 0.5]]}],
        "chunks.json": [{"id": "abc123-0000", "index": 0, "start": 0.2, "end": 1.8, "transcript": "Hello there.",
                         "tags": ["a title card with large text"], "segment_ids": [0], "keyframe_ids": [0]}],
        "alignment.json": {"path": [[0, 0]], "path_costs": [0.0], "total_cost": 0.0, "cells": 1, "full_cells": 1, "banded": True},
    }
    for name, data in files.items():
        (folder / name).write_text(json.dumps(data))
    (folder / "transcript.vtt").write_text("WEBVTT\n\n00:00:00.200 --> 00:00:01.800\nHello there.\n")


def test_welcome_page_without_videos(data_dir):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "Ask a video anything" in [t.value for t in at.title]
    assert any("VidSense works without an LLM" in info.value for info in at.info)


def test_video_page_renders_and_jumps(data_dir):
    _processed_video(data_dir)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.header[0].value == "Tiny"
    assert [tab.label.split()[-1] for tab in at.tabs if tab.label.startswith(":material")] == ["Ask", "Summary", "Moments", "Transcript"]
    assert len(at.get("video")) == 1

    at.button(key="kf-abc123-0").click().run()  # the scene strip's jump button
    assert not at.exception
    assert at.session_state["seek_exact"] == 0.5
    assert any("Hello there." in md.value for md in at.markdown) or any("Hello there." in t.value for t in at.text)
