"""Offline tests for transcript search and rendering.

All tests run on fixture caption lines (no network, no API quota). The fixture
is the real caption split that exposed P1: "think like your agents" is spoken at
0:59 but the captions break it across two lines.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server  # noqa: E402
from mcp.server.fastmcp.exceptions import ToolError  # noqa: E402
from server import Captions, Segment  # noqa: E402

VID = "D7_ipDqhtwk"

FIXTURE = [
    {"text": "keep it simple. And", "start": 57.0, "duration": 2.0},
    {"text": "third, think like your", "start": 59.0, "duration": 1.5},
    {"text": "agents. Let's first start with a recap", "start": 60.5, "duration": 2.5},
]


def make_captions(rows=FIXTURE, generated=True) -> Captions:
    return Captions(
        segments=tuple(Segment(r["text"], r["start"], r["duration"]) for r in rows),
        language="English (auto-generated)" if generated else "English",
        language_code="en",
        is_generated=generated,
    )


@pytest.fixture
def captions(monkeypatch):
    """Patch the shared loader so tools read the fixture instead of YouTube."""
    caps = make_captions()
    monkeypatch.setattr(server, "load_segments", lambda video_id, languages: caps)
    return caps


def search(needle):
    return server.search_transcript(VID, needle)


# --- Solution 1: phrase search -------------------------------------------------

def test_phrase_across_two_lines(captions):
    """P1: the bug. Old per-line search returned nothing here."""
    r = search("think like your agents")
    assert r["count"] == 1
    hit = r["results"][0]
    assert hit["start_seconds"] == 59.0
    assert hit["end_seconds"] == 63.0  # end of the "agents. ..." line
    assert hit["text"] == "third, think like your agents."
    assert hit["url"] == f"https://youtube.com/watch?v={VID}&t=59s"


def test_match_inside_one_line(captions):
    r = search("recap")
    assert r["count"] == 1
    assert r["results"][0]["start_seconds"] == 60.5
    assert r["results"][0]["text"] == "agents. Let's first start with a recap"


def test_phrase_across_three_lines(captions):
    r = search("and third, think like your agents")
    assert r["count"] == 1
    hit = r["results"][0]
    assert hit["start_seconds"] == 57.0
    assert hit["text"] == "keep it simple. And third, think like your agents."
    assert hit["context_before"] == ""


def test_case_and_punctuation_ignored(captions):
    assert search("THINK like your agents!") == {
        **search("think like your agents"),
        "query": "THINK like your agents!",
    }


def test_curly_quotes_match_straight(captions):
    assert search("let’s first")["count"] == 1
    assert search("lets first")["count"] == 1


def test_no_match_is_not_an_error(captions):
    r = search("kubernetes")
    assert r["count"] == 0
    assert r["results"] == []


@pytest.mark.parametrize("phrase", ["", "   ", "?!"])
def test_empty_phrase_is_a_clear_error(captions, phrase):
    with pytest.raises(ToolError, match="empty"):
        search(phrase)


def test_context_lines(captions):
    hit = search("think like your agents")["results"][0]
    assert hit["context_before"] == "keep it simple. And"
    assert hit["context_after"] == "Let's first start with a recap"


def test_repeated_matches_all_returned_in_time_order(monkeypatch):
    rows = FIXTURE + [
        {"text": "so again, think like", "start": 486.0, "duration": 1.5},
        {"text": "your agents", "start": 487.5, "duration": 1.0},
    ]
    monkeypatch.setattr(server, "load_segments", lambda v, l: make_captions(rows))
    r = search("think like your agents")
    assert [h["start_seconds"] for h in r["results"]] == [59.0, 486.0]


def test_music_tags_kept_and_symbol_lines_skipped(monkeypatch):
    rows = [
        {"text": "[Music]", "start": 0.0, "duration": 2.0},
        {"text": "♪", "start": 2.0, "duration": 1.0},  # cleans to "" — must not break offsets
        {"text": "hello and", "start": 3.0, "duration": 1.0},
        {"text": "welcome", "start": 4.0, "duration": 1.0},
    ]
    monkeypatch.setattr(server, "load_segments", lambda v, l: make_captions(rows))
    assert search("and welcome")["results"][0]["start_seconds"] == 3.0
    assert search("music")["results"][0]["text"] == "[Music]"


def test_whole_words_only(monkeypatch):
    """Regression: substring search made "ai" match "said" and "cap" match "recap"."""
    rows = [{"text": "he said again that we maintain AI tools", "start": 0.0, "duration": 2.0}]
    monkeypatch.setattr(server, "load_segments", lambda v, l: make_captions(rows))
    r = search("ai")
    assert r["count"] == 1 and r["total_occurrences"] == 1  # only the real "AI"
    assert search("tool")["count"] == 0  # no partial-word match at the end either
    assert search("maintain ai")["count"] == 1


def test_whole_words_across_line_break(captions):
    assert search("cap")["count"] == 0  # inside "recap"
    assert search("your agents")["count"] == 1  # boundary is the joining space
    assert search("ur agents")["count"] == 0  # starts inside "your"


def test_repeats_in_one_line_are_counted(monkeypatch):
    rows = [
        {"text": "agents, agents, agents", "start": 10.0, "duration": 2.0},
        {"text": "and more agents", "start": 12.0, "duration": 2.0},
    ]
    monkeypatch.setattr(server, "load_segments", lambda v, l: make_captions(rows))
    r = search("agents")
    assert r["count"] == 2  # one result per caption line
    assert [h["occurrences"] for h in r["results"]] == [3, 1]
    assert r["total_occurrences"] == 4


def test_result_is_one_object_and_reports_caption_kind(captions):
    r = search("recap")
    assert set(r) == {
        "video_id", "query", "auto_generated_captions", "count", "total_occurrences", "results",
    }
    assert r["video_id"] == VID and r["auto_generated_captions"] is True
    json.dumps(r)  # serialisable as a single JSON document


# --- Solution 0: shared, cached loader ------------------------------------------

def test_search_and_transcript_share_one_cached_fetch(monkeypatch):
    fetched = []

    class FakeApi:
        def fetch(self, video_id, languages):
            fetched.append(video_id)

            class FT:  # minimal stand-in for youtube_transcript_api.FetchedTranscript
                language, language_code, is_generated = "English", "en", False

                def __iter__(self):
                    return iter(
                        type("S", (), {"text": r["text"], "start": r["start"], "duration": r["duration"]})
                        for r in FIXTURE
                    )

            return FT()

    server._load_segments_cached.cache_clear()
    monkeypatch.setattr(server, "yt_transcripts", lambda: FakeApi())
    try:
        server.search_transcript(VID, "recap")
        server.get_transcript(VID)
        assert fetched == [VID]
    finally:
        server._load_segments_cached.cache_clear()


# --- Solution 2: timestamped transcripts ------------------------------------------

def test_default_output_is_plain_text_with_header(captions):
    out = server.get_transcript(VID)
    header, body = out.split("\n\n", 1)
    assert header == "Captions: English (en), auto-generated"
    assert body == " ".join(r["text"] for r in FIXTURE)


def test_timestamped_output(captions):
    out = server.get_transcript(VID, timestamps=True)
    assert out.split("\n\n", 1)[1].splitlines() == [
        "[00:57] keep it simple. And",
        "[00:59] third, think like your",
        "[01:00] agents. Let's first start with a recap",
    ]


def test_time_range(captions):
    out = server.get_transcript(VID, timestamps=True, start="59", end="1:00")
    assert out.split("\n\n", 1)[1] == "[00:59] third, think like your"


def test_time_range_accepts_plain_numbers(captions):
    """Regression: numeric seconds were rejected by the tool schema (typed str)."""
    out = server.get_transcript(VID, timestamps=True, start=59, end=60.0)
    assert out.split("\n\n", 1)[1] == "[00:59] third, think like your"


def test_truncation_note_says_to_keep_options(captions):
    out = server.get_transcript(VID, timestamps=True, max_chars=50)
    assert "same timestamps/start/end options" in out


def test_time_range_with_no_lines(captions):
    out = server.get_transcript(VID, start="10:00")
    assert "No caption lines" in out


def test_end_before_start_is_an_error(captions):
    with pytest.raises(ToolError):
        server.get_transcript(VID, start="1:00", end="0:30")


@pytest.mark.parametrize("timestamps", [False, True])
def test_pages_end_on_line_boundaries_and_cover_everything(captions, timestamps):
    full = server.get_transcript(VID, timestamps=timestamps).split("\n\n", 1)[1]
    sep = "\n" if timestamps else " "
    lines = full.split(sep) if timestamps else None

    pieces, start_char = [], 0
    for _ in range(20):
        out = server.get_transcript(VID, timestamps=timestamps, max_chars=50, start_char=start_char)
        page = out.split("\n\n", 1)[1]
        if "[Truncated:" in page:
            page, note = page.split("\n\n[Truncated:")
            start_char = int(note.split("start_char=")[1].split(" ")[0])
            pieces.append(page)
        else:
            pieces.append(page)
            break
    else:
        pytest.fail("paging never finished")

    assert len(pieces) > 1
    for p in pieces:  # every page is whole caption lines
        if timestamps:
            assert all(line in lines for line in p.split("\n"))
        else:
            assert full.startswith(p) or f" {p}" in full
            assert any(p.endswith(r["text"]) for r in FIXTURE)
    assert sep.join(pieces) == full  # nothing lost or duplicated


def test_hard_cut_when_one_line_exceeds_page(monkeypatch):
    rows = [{"text": "x" * 100, "start": 0.0, "duration": 5.0}]
    monkeypatch.setattr(server, "load_segments", lambda v, l: make_captions(rows))
    out = server.get_transcript(VID, max_chars=40)
    assert "start_char=40" in out


def test_clock_format_adds_hours_only_when_needed():
    assert server.fmt_clock(59) == "00:59"
    assert server.fmt_clock(3723) == "1:02:03"
