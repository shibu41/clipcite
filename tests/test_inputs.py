"""Input parsing and MCP-level behaviour (schemas, threading). Offline."""

import asyncio
import json
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server  # noqa: E402
from mcp.server.fastmcp.exceptions import ToolError  # noqa: E402
from server import Captions, Segment, extract_video_id, parse_timestamp  # noqa: E402


# --- parse_timestamp -------------------------------------------------------------

@pytest.mark.parametrize(
    "value, seconds",
    [
        (90, 90.0),
        (90.5, 90.5),
        (0, 0.0),
        ("90", 90.0),
        ("90s", 90.0),
        (" 1:30 ", 90.0),
        ("01:30", 90.0),
        ("1:30.5", 90.5),
        ("1:02:03", 3723.0),
        ("90:00", 5400.0),  # leading part may exceed 59
    ],
)
def test_parse_timestamp_accepts(value, seconds):
    assert parse_timestamp(value) == seconds


@pytest.mark.parametrize(
    "value",
    ["nan", "inf", "-inf", float("nan"), float("inf"), "1:90", "1:-30", "1:60",
     "1.5:30", "abc", "", "1:2:3:4", "1e3", True],
)
def test_parse_timestamp_rejects(value):
    with pytest.raises(ToolError):
        parse_timestamp(value)


@pytest.mark.parametrize("value", [-1, "-5", -0.5])
def test_parse_timestamp_rejects_negative(value):
    with pytest.raises(ToolError):
        parse_timestamp(value)


# --- extract_video_id --------------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?feature=share&v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?t=43",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/live/dQw4w9WgXcQ",
    ],
)
def test_extract_video_id(url):
    assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_extract_video_id_ignores_other_params_ending_in_v():
    """Regression: `dev=XXXXXXXXXXX` was mistaken for the video id."""
    url = "https://www.youtube.com/watch?dev=AAAAAAAAAAA&v=dQw4w9WgXcQ"
    assert extract_video_id(url) == "dQw4w9WgXcQ"


# --- MCP level: schemas and threading ------------------------------------------------

def call(name, args):
    return asyncio.run(server.mcp.call_tool(name, args))


def test_tools_are_async_handlers_with_real_schemas():
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    props = tools["get_transcript"].inputSchema["properties"]
    assert {"video_id_or_url", "timestamps", "start", "end", "max_chars", "start_char"} <= set(props)
    assert "needle" in tools["search_transcript"].inputSchema["required"]
    assert tools["search_transcript"].description.startswith("Find every moment")


def test_numeric_start_accepted_through_mcp(monkeypatch):
    caps = Captions((Segment("hello", 59.0, 1.0), Segment("later", 70.0, 1.0)), "English", "en", False)
    monkeypatch.setattr(server, "load_segments", lambda v, l: caps)
    _, structured = call("get_transcript", {"video_id_or_url": "D7_ipDqhtwk", "start": 59, "end": 60})
    assert structured["result"].endswith("hello")


def test_resources_still_registered():
    templates = asyncio.run(server.mcp.list_resource_templates())
    assert {t.uriTemplate for t in templates} == {
        "clipcite://transcript/{video_id}",
        "clipcite://channel/{handle}/recent",
        "clipcite://video/{video_id}/info",
    }


def test_slow_tools_run_concurrently_off_the_event_loop(monkeypatch):
    """Regression: plain `def` tools blocked the event loop for the whole call."""
    loop_thread = []

    def slow_info(vid):
        loop_thread.append(threading.get_ident())
        time.sleep(0.5)
        return {"title": "t", "description": "", "duration_seconds": 0}

    monkeypatch.setattr(server, "_video_info", slow_info)

    async def main():
        loop_ident = threading.get_ident()
        t0 = time.monotonic()
        await asyncio.gather(*(
            server.mcp.call_tool("get_chapters", {"video_id_or_url": "D7_ipDqhtwk"})
            for _ in range(3)
        ))
        return loop_ident, time.monotonic() - t0

    loop_ident, elapsed = asyncio.run(main())
    assert elapsed < 1.2  # 3 x 0.5s in parallel, not 1.5s in series
    assert loop_ident not in loop_thread  # work happened off the event-loop thread


def test_each_thread_gets_its_own_api_client(monkeypatch):
    monkeypatch.setenv(server.API_KEY_ENV, "test-key")
    built = []
    monkeypatch.setattr(server, "build", lambda *a, **k: built.append(object()) or built[-1])
    monkeypatch.setattr(server, "_thread_local", threading.local())

    clients = []
    threads = [threading.Thread(target=lambda: clients.append((server.yt(), server.yt()))) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    (a1, a2), (b1, b2) = clients
    assert a1 is a2 and b1 is b2  # reused within a thread
    assert a1 is not b1  # never shared between threads
    assert len(built) == 2


FAKE_KEY = "not-a-real-key-clipcite-test-0000"  # deliberately not Google-key shaped (secret scanners)


@pytest.mark.parametrize(
    "content",
    [b"", b"<html>502 Bad Gateway</html>", b'{"error": {"message": "Backend Error"}}'],
)
def test_api_errors_never_echo_the_key(monkeypatch, content):
    """Regression: str(HttpError) includes the request URL, which contains key=..."""
    from googleapiclient.errors import HttpError

    monkeypatch.setenv(server.API_KEY_ENV, FAKE_KEY)

    class Resp:
        status, reason = 502, ""

        def get(self, k, d=None):
            return d

    uri = f"https://youtube.googleapis.com/youtube/v3/videos?id=x&key={FAKE_KEY}&alt=json"

    class Req:
        def execute(self):
            raise HttpError(Resp(), content, uri=uri)

    with pytest.raises(ToolError) as info:
        server.run(Req())
    assert FAKE_KEY not in str(info.value)
    assert "502" in str(info.value)
    assert info.value.__cause__ is None and info.value.__suppress_context__  # no traceback chain


def test_redact_key_scrubs_urls_and_the_raw_key(monkeypatch):
    monkeypatch.setenv(server.API_KEY_ENV, FAKE_KEY)
    assert server.redact_key(f"...?a=1&key={FAKE_KEY}&b=2") == "...?a=1&key=REDACTED&b=2"
    assert FAKE_KEY not in server.redact_key(f"bare {FAKE_KEY} here")


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv(server.API_KEY_ENV, raising=False)
    monkeypatch.setattr(server, "_thread_local", threading.local())
    with pytest.raises(ToolError, match=server.API_KEY_ENV):
        server.yt()
