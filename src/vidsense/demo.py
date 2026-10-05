"""Generate a synthetic demo video: drawn scenes with text-to-speech narration.

The scene boundaries are known exactly, so the QA file written next to the video has
ground-truth answer spans for `vidsense eval`. It is a smoke test of the whole
pipeline, not a benchmark. Narration uses the macOS `say` command, or espeak-ng /
espeak on Linux.
"""

from __future__ import annotations

import json
import math
import random
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .media import load_audio

WIDTH, HEIGHT = 1280, 720
FPS = 12
SAMPLE_RATE = 44100
LEAD_IN, TAIL = 0.8, 1.4  # silence before and after each scene's narration


@dataclass
class Scene:
    key: str
    narration: str
    draw: Callable[[float], Image.Image]  # local time in seconds -> frame


# ---------------------------------------------------------------- drawing helpers


@lru_cache(maxsize=16)
def _gradient(top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    ramp = np.linspace(0.0, 1.0, HEIGHT)[:, None, None]
    rows = (1 - ramp) * np.array(top) + ramp * np.array(bottom)
    return Image.fromarray(np.repeat(rows, WIDTH, axis=1).astype(np.uint8), "RGB")


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has only the small bitmap font
        return ImageFont.load_default()


def _canvas(top, bottom) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = _gradient(top, bottom).copy()
    return image, ImageDraw.Draw(image)


def _title_card(title: str, subtitle: str) -> Callable[[float], Image.Image]:
    def draw(t: float) -> Image.Image:
        image, d = _canvas((14, 22, 58), (44, 66, 128))
        d.text((WIDTH // 2, HEIGHT // 2 - 50), title, font=_font(92), fill=(255, 255, 255), anchor="mm")
        d.text((WIDTH // 2, HEIGHT // 2 + 50), subtitle, font=_font(40), fill=(190, 205, 240), anchor="mm")
        for i in range(5):  # small animated dots so the frames are not identical
            radius = 9 + 5 * math.sin(t * 3 + i)
            x, y = WIDTH // 2 - 120 + i * 60, HEIGHT // 2 + 140
            d.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(120, 170, 255))
        return image

    return draw


def _ocean(t: float) -> Image.Image:
    image, d = _canvas((120, 180, 235), (215, 235, 250))
    horizon = int(HEIGHT * 0.55)
    d.ellipse((WIDTH - 270, 60, WIDTH - 150, 180), fill=(255, 220, 90))
    d.rectangle((0, horizon, WIDTH, HEIGHT), fill=(20, 90, 160))
    for row in range(8):
        y = horizon + 25 + row * 38
        points = [(x, y + 6 * math.sin(x / 45 + t * 1.5 + row)) for x in range(0, WIDTH + 20, 20)]
        d.line(points, fill=(200, 230, 255), width=3)
    bx, by = 160 + (t * 45) % (WIDTH - 320), horizon + 90 + 6 * math.sin(t * 2)
    d.polygon([(bx - 95, by), (bx + 95, by), (bx + 62, by + 42), (bx - 62, by + 42)], fill=(120, 70, 30))
    d.line([(bx - 5, by - 180), (bx - 5, by)], fill=(80, 50, 20), width=6)
    d.polygon([(bx, by - 175), (bx, by - 8), (bx + 110, by - 8)], fill=(250, 250, 250))
    for k in range(4):
        gx, gy = (220 + k * 240 + t * 60) % WIDTH, 130 + 30 * math.sin(t + k)
        d.line([(gx - 20, gy), (gx, gy + 11), (gx + 20, gy)], fill=(60, 60, 60), width=3)
    return image


def _forest(t: float) -> Image.Image:
    image, d = _canvas((150, 200, 240), (200, 230, 210))
    d.rectangle((0, int(HEIGHT * 0.7), WIDTH, HEIGHT), fill=(70, 110, 50))
    rng = random.Random(7)
    trees = sorted(((rng.randint(-40, WIDTH + 40), rng.randint(int(HEIGHT * 0.55), HEIGHT - 20), rng.uniform(0.6, 1.4)) for _ in range(26)), key=lambda tree: tree[1])
    for i, (x, base, scale) in enumerate(trees):
        sway = 4 * math.sin(t * 1.2 + i)
        trunk_w, height = 16 * scale, 260 * scale
        d.rectangle((x - trunk_w / 2, base - 50 * scale, x + trunk_w / 2, base), fill=(95, 60, 30))
        for level in range(3):
            top = base - height + level * height * 0.22
            half = (55 + level * 22) * scale
            d.polygon([(x + sway, top), (x - half, top + height * 0.38), (x + half, top + height * 0.38)], fill=(20, 80 + level * 15, 40))
    return image


def _city(t: float) -> Image.Image:
    image, d = _canvas((245, 150, 95), (70, 45, 100))
    road_top = int(HEIGHT * 0.78)
    rng = random.Random(3)
    x = -20
    while x < WIDTH:
        w, h = rng.randint(90, 170), rng.randint(220, 520)
        shade = rng.randint(60, 105)
        d.rectangle((x, road_top - h, x + w, road_top), fill=(shade, shade, shade + 20))
        for wy in range(road_top - h + 18, road_top - 20, 34):
            for wx in range(x + 12, x + w - 18, 26):
                lit = (wx * 7 + wy * 13 + int(t * 2)) % 5 != 0
                d.rectangle((wx, wy, wx + 12, wy + 18), fill=(255, 215, 110) if lit else (45, 45, 60))
        x += w + rng.randint(6, 20)
    d.rectangle((0, road_top, WIDTH, HEIGHT), fill=(50, 50, 55))
    for lane_x in range(-80, WIDTH, 120):
        lx = lane_x + (t * 120) % 120
        d.rectangle((lx, road_top + 70, lx + 60, road_top + 78), fill=(240, 240, 240))
    for k, color in enumerate([(200, 40, 40), (40, 90, 200), (230, 200, 40), (240, 240, 240)]):
        direction = 1 if k % 2 == 0 else -1
        cx = (k * 330 + direction * t * 150) % (WIDTH + 200) - 100
        cy = road_top + (25 if direction > 0 else 100)
        d.rounded_rectangle((cx, cy, cx + 150, cy + 45), radius=10, fill=color)
        for wheel in (cx + 30, cx + 120):
            d.ellipse((wheel - 14, cy + 32, wheel + 14, cy + 60), fill=(20, 20, 20))
    return image


def _desert(t: float) -> Image.Image:
    image, d = _canvas((150, 200, 245), (250, 225, 170))
    d.ellipse((180, 70, 360, 250), fill=(255, 235, 120))
    for layer, (color, base, amplitude) in enumerate([((235, 195, 125), 0.55, 40), ((220, 170, 100), 0.68, 55), ((200, 150, 80), 0.82, 45)]):
        y0 = int(HEIGHT * base)
        points = [(x, y0 + amplitude * math.sin(x / (180 + layer * 40) + layer + 0.05 * math.sin(t))) for x in range(0, WIDTH + 20, 20)]
        d.polygon(points + [(WIDTH, HEIGHT), (0, HEIGHT)], fill=color)
    for cx, scale in ((880, 1.0), (1080, 0.7), (420, 0.8)):
        base = int(HEIGHT * 0.8)
        h, w = 190 * scale, 36 * scale
        d.rounded_rectangle((cx - w / 2, base - h, cx + w / 2, base), radius=int(w / 2), fill=(60, 130, 60))
        d.rounded_rectangle((cx - 2.2 * w, base - 0.75 * h, cx - 1.2 * w, base - 0.35 * h), radius=int(w / 2), fill=(60, 130, 60))
        d.rectangle((cx - 1.6 * w, base - 0.45 * h, cx - w / 2, base - 0.35 * h), fill=(60, 130, 60))
    return image


def _night(t: float) -> Image.Image:
    image, d = _canvas((4, 8, 30), (30, 40, 85))
    rng = random.Random(11)
    for i in range(160):
        sx, sy = rng.randint(0, WIDTH), rng.randint(0, int(HEIGHT * 0.6))
        level = int(150 + 100 * math.sin(t * 3 + i))
        r = 1 if i % 4 else 2
        d.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(level, level, min(255, level + 20)))
    d.ellipse((960, 80, 1080, 200), fill=(245, 245, 230))
    d.ellipse((990, 70, 1100, 190), fill=(10, 15, 42))
    for peak_x, peak_y, half in ((250, 330, 330), (640, 260, 380), (1030, 350, 320)):
        d.polygon([(peak_x - half, HEIGHT), (peak_x, peak_y), (peak_x + half, HEIGHT)], fill=(22, 26, 48))
        d.polygon([(peak_x - 45, peak_y + 50), (peak_x, peak_y), (peak_x + 45, peak_y + 50)], fill=(210, 215, 235))
    return image


SCENES = [
    Scene(
        "intro",
        "Welcome to the VidSense demo. In about a minute, we will visit five very different places around the world.",
        _title_card("VidSense demo", "A one-minute trip around the world"),
    ),
    Scene("ocean", "Our first stop is the open ocean. A small sailboat drifts across the waves, while seagulls circle overhead.", _ocean),
    Scene("forest", "Next, we hike through a quiet pine forest. Some of these trees are more than two hundred years old.", _forest),
    Scene("city", "Now we arrive in a busy city. Traffic moves slowly past the tall office towers downtown.", _city),
    Scene("desert", "The desert is our fourth stop. At noon, the temperature here can reach fifty degrees Celsius.", _desert),
    Scene("night", "When the sun goes down, thousands of stars appear above the mountains.", _night),
    Scene("outro", "That is the end of our tour. Thanks for watching the VidSense demo.", _title_card("Thanks for watching", "VidSense demo")),
]

# (question, short answer, scene). An empty answer means the question is about finding a moment.
QUESTIONS = [
    ("How old are some of the trees in the forest?", "more than two hundred years", "forest"),
    ("How hot can it get in the desert?", "fifty degrees Celsius", "desert"),
    ("What drifts across the waves?", "a small sailboat", "ocean"),
    ("Where does traffic move slowly?", "past the tall office towers downtown", "city"),
    ("When do the stars appear?", "when the sun goes down", "night"),
    ("How many places does the tour visit?", "five", "intro"),
    ("Show me the mountains at night", "", "night"),
    ("When do we see tall buildings?", "", "city"),
    ("Which part shows sand dunes and cactus?", "", "desert"),
    ("When is the boat on the water shown?", "", "ocean"),
]


# ---------------------------------------------------------------- narration


def _say_voice() -> str | None:
    try:
        listing = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    voices = {}
    for line in listing.splitlines():
        match = re.match(r"^(.+?)\s+([a-z]{2}_[A-Z]{2})\s+#", line)
        if match:
            voices[match.group(1).strip()] = match.group(2)
    for preferred in ("Samantha", "Alex", "Daniel", "Karen", "Moira", "Tessa"):
        if preferred in voices:
            return preferred
    english = [name for name, locale in voices.items() if locale in ("en_US", "en_GB")]
    return english[0] if english else None


def _synthesize(text: str, out_dir: Path, index: int) -> np.ndarray:
    if shutil.which("say"):
        path = out_dir / f"line{index}.aiff"
        voice = _say_voice()
        cmd = ["say", "-o", str(path)] + (["-v", voice] if voice else []) + [text]
    elif exe := (shutil.which("espeak-ng") or shutil.which("espeak")):
        path = out_dir / f"line{index}.wav"
        cmd = [exe, "-w", str(path), text]
    else:
        raise RuntimeError("No text-to-speech found: needs `say` (macOS) or espeak-ng / espeak (Linux).")
    subprocess.run(cmd, check=True, capture_output=True)
    return load_audio(path, SAMPLE_RATE)


# ---------------------------------------------------------------- assembly


def make_demo(out_dir: str | Path, scenes: list[Scene] | None = None, questions=QUESTIONS) -> tuple[Path, Path]:
    """Write `vidsense_demo.mp4` and `demo_qa.jsonl` into out_dir and return both paths."""
    scenes = scenes or SCENES
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path, qa_path = out_dir / "vidsense_demo.mp4", out_dir / "demo_qa.jsonl"

    with tempfile.TemporaryDirectory() as tmp:
        speech = [_synthesize(scene.narration, Path(tmp), i) for i, scene in enumerate(scenes)]

    spans, cursor = {}, 0.0
    for scene, audio in zip(scenes, speech):
        length = max(6.0, LEAD_IN + len(audio) / SAMPLE_RATE + TAIL)
        spans[scene.key] = (round(cursor, 2), round(cursor + length, 2))
        cursor += length
    total = cursor

    track = np.zeros(int(total * SAMPLE_RATE) + SAMPLE_RATE, dtype=np.float32)
    for scene, audio in zip(scenes, speech):
        start = int((spans[scene.key][0] + LEAD_IN) * SAMPLE_RATE)
        track[start : start + len(audio)] = audio * 0.9

    n_frames = int(total * FPS)
    with av.open(str(video_path), "w", container_options={"movflags": "+faststart"}) as out:
        video = out.add_stream("libx264", rate=FPS, options={"preset": "veryfast", "crf": "23"})
        video.width, video.height, video.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        audio_stream = out.add_stream("aac", rate=SAMPLE_RATE)
        audio_stream.layout = "mono"
        audio_pos, scene_idx = 0, 0
        for i in range(n_frames):
            t = i / FPS
            while scene_idx < len(scenes) - 1 and t >= spans[scenes[scene_idx].key][1]:
                scene_idx += 1
            scene = scenes[scene_idx]
            frame = av.VideoFrame.from_image(scene.draw(t - spans[scene.key][0]))
            frame.pts = i
            for packet in video.encode(frame):
                out.mux(packet)
            target = min(int((i + 1) / FPS * SAMPLE_RATE), len(track))
            while audio_pos < target:  # interleave audio with video as we go
                block = track[audio_pos : min(audio_pos + 2048, target)]
                aframe = av.AudioFrame.from_ndarray(block[None, :], format="fltp", layout="mono")
                aframe.sample_rate, aframe.pts = SAMPLE_RATE, audio_pos
                for packet in audio_stream.encode(aframe):
                    out.mux(packet)
                audio_pos += len(block)
        for stream in (video, audio_stream):
            for packet in stream.encode(None):
                out.mux(packet)

    with qa_path.open("w", encoding="utf-8") as f:
        for question, answer, key in questions:
            if key in spans:
                start, end = spans[key]
                row = {"video": str(video_path), "question": question, "answer": answer, "start": start, "end": end, "scene": key}
                f.write(json.dumps(row) + "\n")
    return video_path, qa_path


if __name__ == "__main__":
    import sys

    video, qa = make_demo(sys.argv[1] if len(sys.argv) > 1 else "demo")
    print(video, qa, sep="\n")
