import pytest

from vidsense.schemas import Segment
from vidsense.timeutil import format_range, format_ts, parse_ts, to_vtt


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "00:00"), (59.99, "00:59"), (135.7, "02:15"), (3600, "1:00:00"), (3725, "1:02:05"), (-4, "00:00")],
)
def test_format_ts(seconds, expected):
    assert format_ts(seconds) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("02:15", 135), ("1:02:05", 3725), ("0:07", 7), ("12.5", 12.5), (" 3:04 ", 184)],
)
def test_parse_ts(text, expected):
    assert parse_ts(text) == pytest.approx(expected)


def test_parse_ts_rejects_garbage():
    with pytest.raises(ValueError):
        parse_ts("soon")


def test_format_range():
    assert format_range(5, 75) == "00:05–01:15"


def test_to_vtt():
    vtt = to_vtt([Segment(0, 0.0, 2.5, " hello "), Segment(1, 3661.25, 3662.0, "world")])
    assert vtt.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:02.500\nhello\n" in vtt
    assert "01:01:01.250 --> 01:01:02.000\nworld\n" in vtt
