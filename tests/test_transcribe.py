from collections import namedtuple

from vidsense.transcribe import split_sentences

Word = namedtuple("Word", "start end word")


def _words(text, start=0.0, step=0.4):
    out = []
    for i, token in enumerate(text.split()):
        out.append(Word(start + i * step, start + i * step + 0.3, f" {token}"))
    return out


def test_sentences_cross_whisper_windows():
    # Whisper's windows ended at "...places around" and "...across the waves,".
    words = _words("Welcome to the demo. We will visit five places around the world. Our first stop is the ocean.")
    sentences = split_sentences(words)
    assert [s.text for s in sentences] == [
        "Welcome to the demo.",
        "We will visit five places around the world.",
        "Our first stop is the ocean.",
    ]
    assert sentences[1].start == words[4].start and sentences[1].end == words[11].end
    assert [s.id for s in sentences] == [0, 1, 2]


def test_long_pause_and_missing_punctuation_also_cut():
    words = _words("la la la", 0.0) + _words("still no punctuation here", 5.0)  # 3.8 s of silence between
    assert [s.text for s in split_sentences(words)] == ["la la la", "still no punctuation here"]
    endless = _words(" ".join(["word"] * 80), step=0.5)  # 40 s without punctuation or pauses
    assert all(s.end - s.start <= 20.0 for s in split_sentences(endless))


def test_quotes_after_the_period_still_end_a_sentence():
    words = _words('He said "stop." Then he left.')
    assert [s.text for s in split_sentences(words)] == ['He said "stop."', "Then he left."]
