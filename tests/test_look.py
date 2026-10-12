import av
import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import ParrotFakeChatModel
from langchain_core.messages import AIMessage
from PIL import Image

from vidsense import qa
from vidsense.config import AnswerConfig, Settings
from vidsense.look import describe, frames_to_inspect
from vidsense.media import grab_frame
from vidsense.schemas import Chunk, Hit, Keyframe, VideoRecord
from vidsense.store import VideoLibrary


class RecordingVision:
    """Stands in for a vision chat model: records what it was shown, replies from a list."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def invoke(self, messages):
        self.calls.append(messages[0].content)
        return AIMessage(content=self.replies.pop(0))


def _keyframe(i, time):
    return Keyframe(id=i, time=time, span_start=time - 2, span_end=time + 2, frame_count=4, image=f"keyframes/kf_{i:04d}.jpg")


def test_frames_to_inspect_takes_each_chunks_first_keyframe_in_relevance_order():
    keyframes = {0: _keyframe(0, 5), 1: _keyframe(1, 27), 2: _keyframe(2, 42)}
    docs = [Document("a", metadata={"keyframe_ids": [1]}), Document("b", metadata={"keyframe_ids": []}),
            Document("c", metadata={"keyframe_ids": [2, 0]}), Document("d", metadata={"keyframe_ids": [0]})]
    picks = frames_to_inspect(docs, keyframes, limit=2)
    assert [(doc.page_content, kf.id) for doc, kf in picks] == [("a", 1), ("c", 2)]


def test_describe_sends_the_question_and_the_image():
    vision = RecordingVision(["The cars are red, blue and yellow.", "Nothing relevant."])
    image = Image.new("RGB", (64, 36), "red")
    assert describe(vision, image, "What colour are the cars?", 27.4) == "The cars are red, blue and yellow."
    text, picture = vision.calls[0]
    assert "00:27" in text["text"] and '"What colour are the cars?"' in text["text"]
    assert picture["type"] == "image_url" and picture["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert describe(vision, image, "What colour are the cars?", 3.0) == ""  # nothing relevant -> no observation


def test_grab_frame_returns_the_frame_at_that_time(tmp_path):
    path = tmp_path / "colours.mp4"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=4)
        stream.width, stream.height, stream.pix_fmt = 1280, 720, "yuv420p"
        for i in range(12):  # 3 s: red, then green, then blue, one second each
            colour = [(220, 30, 30), (30, 200, 30), (30, 30, 220)][i // 4]
            frame = av.VideoFrame.from_ndarray(np.full((720, 1280, 3), colour, dtype=np.uint8), format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    image = grab_frame(path, 1.5, max_side=768)
    assert max(image.size) == 768
    r, g, b = image.getpixel((100, 100))
    assert g > 150 and r < 80 and b < 80  # the green second


@pytest.fixture
def settings(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, answer=AnswerConfig(provider="none", top_k=2, vision_frames=2))
    library = VideoLibrary(settings)
    library.save(VideoRecord(video_id="v", title="Demo", source_path=str(tmp_path / "gone.mp4"), duration=52.0, status="ready"))
    (tmp_path / "videos" / "v" / "keyframes").mkdir(parents=True)
    for i in (0, 1):
        Image.new("RGB", (64, 36), "gray").save(tmp_path / "videos" / "v" / "keyframes" / f"kf_{i:04d}.jpg")
    library.write_json("v", "keyframes.json", [_keyframe(0, 27.0), _keyframe(1, 42.0)])
    chunks = [
        Chunk(id="v-0000", index=0, start=24.0, end=30.0, transcript="Traffic moves slowly downtown.", tags=["cars on a road"], keyframe_ids=[0]),
        Chunk(id="v-0001", index=1, start=40.0, end=46.0, transcript="Stars appear above the mountains.", tags=["mountains"], keyframe_ids=[1]),
    ]
    monkeypatch.setattr(qa, "search", lambda _s, _v, query, k=5: [Hit(chunk=c, score=0.9, rank=i + 1) for i, c in enumerate(chunks)])
    monkeypatch.setattr(qa, "load_embedder", lambda name: None)
    return settings


def test_observations_reach_the_answer_prompt(settings):
    vision = RecordingVision(["The cars are red, blue and yellow.", "Nothing relevant."])
    events = list(qa.VideoQA(settings, "v", llm=ParrotFakeChatModel(), vision=vision).stream("What colour are the cars?"))
    kinds = [kind for kind, _ in events]
    assert kinds.index("looking") < kinds.index("seen") < kinds.index("answer")
    answer = events[-1][1]
    assert answer.observations == [(27.0, "The cars are red, blue and yellow.")]
    assert "Looked at the frame (00:27): The cars are red, blue and yellow." in answer.text  # the parrot echoes the prompt
    assert len(vision.calls) == 2  # both retrieved moments were inspected (from the thumbnails: the video is gone)


def test_no_vision_model_means_no_looking(settings):
    events = list(qa.VideoQA(settings, "v", llm=ParrotFakeChatModel(), vision=None).stream("What colour are the cars?"))
    assert "looking" not in [kind for kind, _ in events]
    assert "Looked at the frame" not in events[-1][1].text


def test_a_failing_vision_model_does_not_cost_the_answer(settings):
    class Broken:
        def invoke(self, messages):
            raise ConnectionError("vision model went away")

    answer = qa.VideoQA(settings, "v", llm=ParrotFakeChatModel(), vision=Broken()).ask("What colour are the cars?")
    assert answer.observations == [] and "Traffic moves slowly" in answer.text


def test_an_empty_answer_falls_back_to_what_was_found(settings):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    vision = RecordingVision(["The cars are red and white.", "Nothing relevant."])
    silent = FakeListChatModel(responses=["<think>which scene is the third one? " + "hmm " * 50 + "</think>"])
    answer = qa.VideoQA(settings, "v", llm=silent, vision=vision).ask("What colour are the cars in the third scene?")
    assert answer.text.startswith("The model ran out of room before answering")
    assert "Seen at [00:27]: The cars are red and white." in answer.text
    assert answer.citations[:1] == [24.0]
