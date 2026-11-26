"""Offline tests for chapter parsing (no network, no API quota)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server  # noqa: E402
from mcp.server.fastmcp.exceptions import ToolError  # noqa: E402
from server import parse_chapters  # noqa: E402


def titles(chapters):
    return [c["title"] for c in chapters]


def starts(chapters):
    return [c["start_seconds"] for c in chapters]


def test_basic_list_with_prose_around_it():
    desc = """In this talk we cover agents.

0:00 Intro
1:30 What is an agent?
5:05 Keep it simple
12:40 Think like your agents

Follow us on Twitter!"""
    ch, ok = parse_chapters(desc, duration=900)
    assert titles(ch) == ["Intro", "What is an agent?", "Keep it simple", "Think like your agents"]
    assert starts(ch) == [0, 90, 305, 760]
    assert [c["end_seconds"] for c in ch] == [90, 305, 760, 900]
    assert ch[1]["start"] == "01:30"
    assert ok is True


@pytest.mark.parametrize(
    "line, title, start",
    [
        ("00:00 - Intro", "Intro", 0),
        ("(0:00) Intro", "Intro", 0),
        ("[0:00] Intro", "Intro", 0),
        ("1. 0:00 Intro", "Intro", 0),
        ("• 0:00 – Intro", "Intro", 0),
        ("Intro - 0:00", "Intro", 0),
        ("Intro (0:00)", "Intro", 0),
        ("0:00 | Intro", "Intro", 0),
        ("00:00:00 Intro", "Intro", 0),
        ("0:00 - 1:30 Intro", "Intro", 0),  # range: second timestamp dropped
    ],
)
def test_common_line_formats(line, title, start):
    desc = f"{line}\n2:00 Next\n4:00 Last"
    ch, _ = parse_chapters(desc, duration=600)
    assert ch[0]["title"] == title
    assert ch[0]["start_seconds"] == start


def test_hour_timestamps():
    ch, _ = parse_chapters("0:00 Start\n59:59 Almost\n1:02:03 Q&A", duration=4000)
    assert starts(ch) == [0, 3599, 3723]
    assert ch[2]["start"] == "1:02:03"


def test_mid_sentence_timestamps_ignored():
    desc = "At 3:20 he explains the trick, and again around 7:45 in more detail."
    assert parse_chapters(desc, duration=600) == ([], False)


def test_no_timestamps_returns_empty():
    assert parse_chapters("Just a description.", duration=600) == ([], False)
    assert parse_chapters("", duration=0) == ([], False)


def test_single_timestamp_is_not_a_chapter_list():
    assert parse_chapters("Skip to 0:45 for the demo", duration=600)[0] == []


def test_longest_increasing_run_wins():
    """A stray 'key moment' earlier in the description shouldn't break the list."""
    desc = "Best bit: 8:00 the reveal\n\n0:00 Intro\n2:00 Setup\n5:00 Demo\n9:00 Wrap-up"
    ch, _ = parse_chapters(desc, duration=700)
    assert titles(ch) == ["Intro", "Setup", "Demo", "Wrap-up"]


def test_timestamps_past_the_end_dropped():
    ch, _ = parse_chapters("0:00 Intro\n2:00 Body\n99:00 Bonus", duration=300)
    assert titles(ch) == ["Intro", "Body"]
    assert ch[-1]["end_seconds"] == 300


def test_empty_title_gets_a_placeholder():
    ch, _ = parse_chapters("0:00\n1:00 Second\n2:00 Third", duration=300)
    assert ch[0]["title"] == "Chapter 1"


@pytest.mark.parametrize(
    "desc, ok",
    [
        ("0:00 A\n1:00 B\n2:00 C", True),
        ("0:05 A\n1:00 B\n2:00 C", False),  # doesn't start at 0:00
        ("0:00 A\n1:00 B", False),  # fewer than 3
        ("0:00 A\n0:05 B\n2:00 C", False),  # a chapter shorter than 10s
    ],
)
def test_youtube_rules_flag(desc, ok):
    ch, meets = parse_chapters(desc, duration=300)
    assert ch and meets is ok


def test_unknown_duration_leaves_last_end_open():
    ch, _ = parse_chapters("0:00 A\n1:00 B\n2:00 C", duration=0)
    assert ch[-1]["end_seconds"] is None


# --- the tool ------------------------------------------------------------------

def test_tool_adds_links_and_wraps_result(monkeypatch):
    info = {
        "title": "Talk",
        "description": "0:00 Intro\n1:30 Main\n5:00 Outro",
        "duration_seconds": 600,
    }
    monkeypatch.setattr(server, "_video_info", lambda vid: info)
    r = server.get_chapters("https://youtu.be/D7_ipDqhtwk")
    assert r["video_id"] == "D7_ipDqhtwk" and r["count"] == 3
    assert r["youtube_chapter_rules_met"] is True
    assert r["chapters"][1]["url"] == "https://youtube.com/watch?v=D7_ipDqhtwk&t=90s"


def test_tool_no_chapters_is_not_an_error(monkeypatch):
    info = {"title": "Talk", "description": "No chapters here.", "duration_seconds": 600}
    monkeypatch.setattr(server, "_video_info", lambda vid: info)
    r = server.get_chapters("D7_ipDqhtwk")
    assert r["count"] == 0 and r["chapters"] == []


def test_tool_missing_video_is_an_error(monkeypatch):
    monkeypatch.setattr(server, "_video_info", lambda vid: None)
    with pytest.raises(ToolError, match="not found"):
        server.get_chapters("AAAAAAAAAAA")
