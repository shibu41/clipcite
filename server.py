"""
ClipCite MCP Server — real YouTube API integration.

Demonstrates all three MCP primitives against the live YouTube Data API v3
plus youtube-transcript-api for captions.

Setup:
    pip install -r requirements.txt
    # put YOUTUBE_API_KEY=... in .env (or export it)

Run with the Inspector:
    npx @modelcontextprotocol/inspector python server.py

Three primitives demonstrated:
  TOOLS     — actions the AI takes ("verbs"): search_videos, get_transcript, ...
  RESOURCES — data the user/host loads ("nouns"): clipcite://transcript/{id}, ...
  PROMPTS   — templates the user invokes ("slash commands"): /study-from-video, ...

Quota note: search.list costs 100 units of the default 10,000/day quota;
videos.list and channels.list cost 1. Channel lookups and transcripts are
cached in-process to avoid paying for the same thing twice.

Frame extraction (get_video_frames / sample_video_frames) needs ffmpeg on PATH
(or FFMPEG_PATH) and uses yt-dlp to resolve a stream URL; ffmpeg then seeks
into that stream, so only a few seconds per frame are fetched. It uses no
YouTube Data API quota. WARNING: this violates YouTube's Terms of Service —
see the README disclaimer. Set CLIPCITE_ENABLE_FRAMES=false to turn the frame
tools off entirely: they are then not registered, and yt-dlp is never imported
(so it doesn't need to be installed).
"""

from __future__ import annotations

import functools
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import warnings
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal

import anyio

# MCP uses stdio — stdout must carry only JSON.
# Redirect all logging and warnings to stderr so they never corrupt the stream.
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
warnings.filterwarnings("default")

# Load .env if present (picks up YOUTUBE_API_KEY without manual export)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logging.getLogger("googleapiclient.discovery").setLevel(logging.ERROR)
logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)
from mcp.server.fastmcp import FastMCP, Image
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import Field
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

# ---------------------------------------------------------------------------
# YouTube API client (lazy + cached so the SDK is built once)
# ---------------------------------------------------------------------------

API_KEY_ENV = "YOUTUBE_API_KEY"
FRAMES_ENV = "CLIPCITE_ENABLE_FRAMES"


def frames_enabled() -> bool:
    """Frame tools are on unless CLIPCITE_ENABLE_FRAMES is false/0/no/off."""
    return os.environ.get(FRAMES_ENV, "true").strip().lower() not in {"false", "0", "no", "off"}


FRAMES_ENABLED = frames_enabled()

SearchOrder = Literal["relevance", "date", "rating", "viewCount", "title"]


# Tools run in worker threads (see `threaded` below). The Google client's HTTP
# layer (httplib2) and requests sessions aren't safe to share across threads,
# so each thread gets its own clients, built once and reused.
_thread_local = threading.local()


def yt():
    """Return this thread's YouTube Data API v3 client."""
    client = getattr(_thread_local, "yt", None)
    if client is None:
        key = os.environ.get(API_KEY_ENV)
        if not key:
            raise ToolError(
                f"Set {API_KEY_ENV} in the environment. "
                "Get a key from https://console.cloud.google.com/ "
                "(enable YouTube Data API v3)."
            )
        client = _thread_local.yt = build("youtube", "v3", developerKey=key, cache_discovery=False)
    return client


_KEY_PARAM_RE = re.compile(r"(key=)[^&\s\"'<>]+", re.IGNORECASE)


def redact_key(text: str) -> str:
    """Remove the API key from any text that might echo a request URL."""
    text = _KEY_PARAM_RE.sub(r"\1REDACTED", text)
    key = os.environ.get(API_KEY_ENV)
    return text.replace(key, "REDACTED") if key else text


def run(request) -> dict:
    """Execute a googleapiclient request, turning HttpError into a clean ToolError.

    Never uses str(e): HttpError's text includes the request URL, and the URL
    carries `key=<your API key>`. Errors are returned to the assistant, so
    that would leak the key into the conversation.
    """
    try:
        return request.execute()
    except HttpError as e:
        status = e.resp.status
        reason = redact_key(str(getattr(e, "reason", "") or "")).strip()
        if status == 403 and "quota" in reason.lower():
            raise ToolError(
                "YouTube API daily quota exceeded. It resets at midnight Pacific time."
            ) from None
        raise ToolError(f"YouTube API error ({status}): {reason or 'no details returned'}") from None


def yt_transcripts() -> YouTubeTranscriptApi:
    """Return this thread's youtube-transcript-api client (v1.0+ instance API)."""
    client = getattr(_thread_local, "transcripts", None)
    if client is None:
        client = _thread_local.transcripts = YouTubeTranscriptApi()
    return client


@dataclass(frozen=True)
class Segment:
    """One caption line. `text` has its internal whitespace/newlines collapsed."""

    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class Captions:
    """A video's caption lines plus which track they came from."""

    segments: tuple[Segment, ...]
    language: str
    language_code: str
    is_generated: bool

    @property
    def header(self) -> str:
        kind = "auto-generated" if self.is_generated else "human-written"
        # The library's name already says "(auto-generated)" for auto tracks.
        name = re.sub(r"\s*\(auto-generated\)", "", self.language)
        return f"Captions: {name} ({self.language_code}), {kind}"


@lru_cache(maxsize=64)
def _load_segments_cached(video_id: str, languages: tuple[str, ...]) -> Captions:
    ft = yt_transcripts().fetch(video_id, languages=list(languages))
    return Captions(
        segments=tuple(
            Segment(" ".join(s.text.split()), float(s.start), float(s.duration))
            for s in ft
        ),
        language=ft.language,
        language_code=ft.language_code,
        is_generated=ft.is_generated,
    )


def load_segments(video_id: str, languages: list[str]) -> Captions:
    """The one place captions are fetched. Cached by (video_id, languages).

    Raises the library's own exceptions. Failures are not cached (lru_cache
    never stores exceptions), so a retry after a transient block refetches.
    """
    return _load_segments_cached(video_id, tuple(languages))


def load_segments_or_raise(video_id: str, languages: list[str]) -> Captions:
    """load_segments for tools: turns any transcript failure into a ToolError."""
    try:
        return load_segments(video_id, languages)
    except CouldNotRetrieveTranscript as e:
        raise ToolError(transcript_error_message(e, video_id, languages)) from e


def transcript_error_message(e: CouldNotRetrieveTranscript, vid: str, langs) -> str:
    """Human-readable explanation for any transcript-fetch failure."""
    if isinstance(e, (NoTranscriptFound, TranscriptsDisabled)):
        return f"No transcript available for {vid} in languages {list(langs)}."
    if isinstance(e, VideoUnavailable):
        return f"Video {vid} is unavailable."
    if isinstance(e, (RequestBlocked, IpBlocked)):
        return (
            f"YouTube blocked the transcript request for {vid} from this IP "
            "(common on cloud/VPN networks). Try another network or configure "
            "a proxy for youtube-transcript-api."
        )
    return f"Could not retrieve transcript for {vid}: {type(e).__name__}."


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

_VIDEO_ID_RE = re.compile(
    # `v=` must be its own query parameter (not the tail of e.g. `dev=`).
    r"(?:[?&]v=|\/shorts\/|\/embed\/|\/live\/|youtu\.be\/)([A-Za-z0-9_-]{11})"
)


def extract_video_id(url_or_id: str) -> str:
    """Accept a raw 11-char ID or any standard YouTube URL form, return the ID."""
    url_or_id = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    m = _VIDEO_ID_RE.search(url_or_id)
    if not m:
        raise ToolError(f"Could not extract video id from: {url_or_id!r}")
    return m.group(1)


def parse_iso8601_duration(s: str) -> int:
    """Convert YouTube's P#DT#H#M#S duration to total seconds."""
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s or "")
    if not m:
        return 0
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se


def fmt_duration(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _is_channel_id(s: str) -> bool:
    return bool(re.fullmatch(r"UC[A-Za-z0-9_-]{22}", s))


@lru_cache(maxsize=128)
def _resolve_channel_id_cached(key: str) -> str | None:
    if _is_channel_id(key):
        return key

    # Handles can't contain spaces; skip the cheap lookup for display names.
    if " " not in key:
        handle = key.lstrip("@")
        try:
            items = yt().channels().list(part="id", forHandle=f"@{handle}").execute().get("items", [])
            if items:
                return items[0]["id"]
        except HttpError:
            pass

    # Fall back to search (costs 100 quota units — hence the cache).
    resp = run(yt().search().list(part="snippet", q=key, type="channel", maxResults=1))
    items = resp.get("items", [])
    return items[0]["snippet"]["channelId"] if items else None


def resolve_channel_id(handle_or_id: str) -> str | None:
    """Accept @handle, channel id (UC...), or channel name; return canonical channel ID."""
    return _resolve_channel_id_cached(handle_or_id.strip())


def require_channel_id(handle_or_id: str) -> str:
    ch_id = resolve_channel_id(handle_or_id)
    if not ch_id:
        raise ToolError(f"Channel not found: {handle_or_id}")
    return ch_id


def _video_info(vid: str) -> dict | None:
    """Fetch metadata for one video; None if it doesn't exist."""
    resp = run(yt().videos().list(part="snippet,contentDetails,statistics", id=vid))
    items = resp.get("items", [])
    if not items:
        return None
    it = items[0]
    sn, cd, st = it["snippet"], it["contentDetails"], it.get("statistics", {})
    return {
        "id": vid,
        "title": sn["title"],
        "channel": sn["channelTitle"],
        "channel_id": sn["channelId"],
        "published_at": sn["publishedAt"],
        "duration_seconds": parse_iso8601_duration(cd.get("duration", "")),
        "view_count": int(st.get("viewCount", 0)),
        "like_count": int(st.get("likeCount", 0)),
        "comment_count": int(st.get("commentCount", 0)),
        "description": sn["description"],
        "tags": sn.get("tags", []),
        "url": f"https://youtube.com/watch?v={vid}",
    }


def _channel_videos(ch_id: str, max_results: int, order: str) -> list[dict]:
    resp = run(
        yt().search().list(
            part="snippet",
            channelId=ch_id,
            type="video",
            maxResults=max(1, min(50, max_results)),
            order=order,
        )
    )
    return [
        {
            "id": item["id"]["videoId"],
            "title": item["snippet"]["title"],
            "published_at": item["snippet"]["publishedAt"],
        }
        for item in resp.get("items", [])
    ]


# ---------------------------------------------------------------------------
# Frame extraction (yt-dlp resolves the stream, ffmpeg seeks + grabs a frame)
# ---------------------------------------------------------------------------

MAX_FRAMES = 10
_STREAM_TTL_SECONDS = 3600  # YouTube stream URLs expire after ~6h; refresh well before.
_FRAME_WORKERS = 4
_FFMPEG_TIMEOUT = 60

_stream_cache: dict[str, tuple[float, dict]] = {}
_stream_lock = threading.Lock()


class _YtDlpLogger:
    """Route yt-dlp output to stderr logging — never stdout (the MCP channel)."""

    _log = logging.getLogger("yt_dlp")

    def debug(self, msg): self._log.debug(msg)
    def info(self, msg): self._log.info(msg)
    def warning(self, msg): self._log.warning(msg)
    def error(self, msg): self._log.error(msg)


def ffmpeg_path() -> str:
    path = os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg")
    if not path:
        raise ToolError(
            "ffmpeg not found. Install it (e.g. `winget install Gyan.FFmpeg`) "
            "or set FFMPEG_PATH."
        )
    return path


_TS_PART_RE = re.compile(r"\d+")
_TS_LAST_RE = re.compile(r"\d+(?:\.\d+)?")

# Time inputs a tool accepts: a number of seconds, or a "90" / "mm:ss" / "hh:mm:ss" string.
TimeValue = str | float


def parse_timestamp(ts: str | float | int) -> float:
    """Accept seconds (90, 90.5, "90", "90s"), "mm:ss" or "hh:mm:ss"; return seconds.

    Every part must be a plain non-negative number; parts after the first must
    be under 60 ("1:90" is rejected). nan, inf and negatives are rejected.
    """
    bad = ToolError(f"Invalid timestamp {ts!r}: use seconds, mm:ss or hh:mm:ss.")
    if isinstance(ts, bool):
        raise bad
    if isinstance(ts, (int, float)):
        if not math.isfinite(ts):
            raise bad
        if ts < 0:
            raise ToolError(f"Timestamp must not be negative: {ts!r}")
        return float(ts)

    parts = str(ts).strip().lower().removesuffix("s").split(":")
    if not 1 <= len(parts) <= 3:
        raise bad
    *leading, last = parts
    if not all(_TS_PART_RE.fullmatch(p) for p in leading) or not _TS_LAST_RE.fullmatch(last):
        raise bad
    values = [float(p) for p in parts]
    if any(v >= 60 for v in values[1:]):
        raise bad
    seconds = 0.0
    for v in values:
        seconds = seconds * 60 + v
    return seconds


def parse_optional_time(ts: str | float | None) -> float | None:
    """parse_timestamp, treating None / "" as not given."""
    if ts is None or (isinstance(ts, str) and not ts.strip()):
        return None
    return parse_timestamp(ts)


def _stream_info(vid: str) -> dict:
    """Resolve (and cache) a direct progressive/DASH stream URL for a video."""
    with _stream_lock:
        hit = _stream_cache.get(vid)
        if hit and hit[0] > time.monotonic():
            return hit[1]

    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "logger": _YtDlpLogger(),
        # Video-only is fine for stills; cap at 720p to keep seeks fast.
        "format": (
            "bestvideo[height<=720][protocol^=http]/"
            "best[height<=720][protocol^=http]/best[protocol^=http]"
        ),
        # YouTube extraction needs a JS runtime; allow Node as well as Deno.
        "js_runtimes": {"node": {}, "deno": {}},
    }
    # Imported here, not at module top, so the server runs without yt-dlp
    # installed when frame tools are disabled.
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
    except DownloadError as e:
        msg = re.sub(r"^ERROR:\s*(\[youtube\]\s*\S+:\s*)?", "", str(e))
        raise ToolError(f"Could not get a video stream for {vid}: {msg}") from e

    if info.get("is_live"):
        raise ToolError(f"{vid} is a live stream; frame extraction needs a finished video.")
    if not info.get("url"):
        raise ToolError(f"No directly seekable stream found for {vid}.")

    stream = {
        "url": info["url"],
        "headers": "".join(f"{k}: {v}\r\n" for k, v in (info.get("http_headers") or {}).items()),
        "duration": float(info.get("duration") or 0),
        "title": info.get("title", vid),
    }
    with _stream_lock:
        _stream_cache[vid] = (time.monotonic() + _STREAM_TTL_SECONDS, stream)
    return stream


def _grab_frame(stream: dict, seconds: float, width: int) -> bytes:
    """Seek into the stream and return one JPEG frame."""
    cmd = [
        ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-headers", stream["headers"],
        "-ss", f"{seconds:.3f}",  # before -i → fast HTTP-range seek
        "-i", stream["url"],
        "-frames:v", "1",
        "-vf", f"scale={width}:-2",
        "-q:v", "4",
        "-f", "image2pipe", "-c:v", "mjpeg", "pipe:1",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg timed out after {_FFMPEG_TIMEOUT}s") from None
    if p.returncode != 0 or not p.stdout:
        err = p.stderr.decode(errors="replace").strip().splitlines()
        raise RuntimeError(err[-1] if err else f"ffmpeg exited with code {p.returncode}")
    return p.stdout


def _extract_frames(vid: str, seconds_list: list[float], width: int) -> list:
    """Blocking core shared by the frame tools. Returns MCP content items."""
    stream = _stream_info(vid)
    dur = stream["duration"]
    if dur:
        too_late = [s for s in seconds_list if s >= dur]
        if too_late:
            raise ToolError(
                f"Timestamps beyond the video's length ({fmt_duration(int(dur))}): "
                + ", ".join(fmt_duration(int(s)) for s in too_late)
            )

    with ThreadPoolExecutor(max_workers=_FRAME_WORKERS) as pool:
        futures = [pool.submit(_grab_frame, stream, s, width) for s in seconds_list]
        results = []
        for s, f in zip(seconds_list, futures):
            try:
                results.append((s, f.result(), None))
            except Exception as e:  # one bad frame shouldn't sink the rest
                results.append((s, None, str(e)))

    if all(img is None for _, img, _ in results):
        # Most likely an expired/blocked stream URL — drop it so a retry re-resolves.
        with _stream_lock:
            _stream_cache.pop(vid, None)
        raise ToolError(f"Could not extract any frames from {vid}: {results[0][2]}")

    content: list = [f"{len(seconds_list)} frame(s) from \"{stream['title']}\""]
    for s, img, err in results:
        label = f"{fmt_duration(int(s))} — https://youtube.com/watch?v={vid}&t={int(s)}s"
        if img is None:
            content.append(f"Frame at {label}: failed ({err})")
        else:
            content.append(f"Frame at {label}")
            content.append(Image(data=img, format="jpeg"))
    return content


# ---------------------------------------------------------------------------
# Transcript search and rendering (pure functions over caption lines)
# ---------------------------------------------------------------------------

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def clean_with_map(text: str) -> tuple[str, list[int]]:
    """Normalise text for matching and remember where each char came from.

    Lowercases, straightens curly quotes, drops punctuation and collapses
    whitespace. Returns (cleaned, index_map) where index_map[i] is the
    position in `text` that produced cleaned[i].
    """
    out: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(text.translate(_QUOTES)):
        if ch.isspace():
            if out and out[-1] != " ":
                out.append(" ")
                idx.append(i)
        elif ch.isalnum():
            for c in ch.lower():  # lower() can yield >1 char (e.g. 'İ')
                out.append(c)
                idx.append(i)
        # anything else is punctuation: dropped
    if out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def clean(text: str) -> str:
    return clean_with_map(text)[0]


def fmt_clock(seconds: float) -> str:
    """[mm:ss] style label; adds hours only when needed."""
    t = int(seconds)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def find_phrase(segments: tuple[Segment, ...] | list[Segment], vid: str, phrase: str) -> list[dict]:
    """Find a phrase anywhere in the transcript, even across caption lines.

    Cleans each line, joins them into one string while recording where each
    line starts, searches that string, then maps each hit back to the lines
    it spans with bisect.
    """
    needle = clean(phrase)
    if not needle:
        raise ToolError("Search phrase is empty (after removing punctuation).")

    joined_parts: list[str] = []
    line_starts: list[int] = []  # char offset of each included line in `joined`
    line_index: list[int] = []  # which segment each included line is
    line_maps: list[list[int]] = []
    pos = 0
    for i, seg in enumerate(segments):
        cleaned, idx_map = clean_with_map(seg.text)
        if not cleaned:  # e.g. "♪" — nothing to match, keep offsets exact
            continue
        if joined_parts:
            pos += 1  # the single joining space
        line_starts.append(pos)
        line_index.append(i)
        line_maps.append(idx_map)
        joined_parts.append(cleaned)
        pos += len(cleaned)
    joined = " ".join(joined_parts)

    results: list[dict] = []
    result_for_span: dict[tuple[int, int], dict] = {}
    hit = joined.find(needle)
    while hit != -1:
        end = hit + len(needle)
        # Whole words only: "ai" must not match inside "said", "cap" not in "recap".
        if (hit > 0 and joined[hit - 1] != " ") or (end < len(joined) and joined[end] != " "):
            hit = joined.find(needle, hit + 1)
            continue

        last_char = end - 1
        a = bisect_right(line_starts, hit) - 1
        b = bisect_right(line_starts, last_char) - 1
        first, last = line_index[a], line_index[b]

        if (first, last) in result_for_span:
            # Same phrase again within the same caption line(s): one result with
            # the same timestamp, but count every occurrence.
            result_for_span[(first, last)]["occurrences"] += 1
        else:
            # Cut the last line just after the matched word (keeping trailing
            # punctuation like "agents."); the rest becomes context.
            last_text = segments[last].text
            cut = line_maps[b][last_char - line_starts[b]] + 1
            while cut < len(last_text) and not last_text[cut].isspace():
                cut += 1
            spanned = [segments[k].text for k in range(first, last)] + [last_text[:cut]]
            after = [last_text[cut:].strip()]
            if last + 1 < len(segments):
                after.append(segments[last + 1].text)

            start = segments[first].start
            result = {
                "start_seconds": round(start, 1),
                "end_seconds": round(segments[last].end, 1),
                "text": " ".join(spanned),
                "context_before": segments[first - 1].text if first > 0 else "",
                "context_after": " ".join(t for t in after if t),
                "occurrences": 1,
                "url": f"https://youtube.com/watch?v={vid}&t={int(start)}s",
            }
            result_for_span[(first, last)] = result
            results.append(result)
        hit = joined.find(needle, end)
    return results


def render_transcript(
    captions: Captions,
    *,
    timestamps: bool = False,
    start: float | None = None,
    end: float | None = None,
    max_chars: int = 50_000,
    start_char: int = 0,
) -> str:
    """Render captions as text, optionally timestamped and time-ranged.

    Pages by `max_chars` but always ends a page on a complete caption line
    (unless a single line is longer than the page).
    """
    segs = [
        s
        for s in captions.segments
        if (start is None or s.start >= start) and (end is None or s.start < end)
    ]
    sep = "\n" if timestamps else " "
    units = [f"[{fmt_clock(s.start)}] {s.text}" if timestamps else s.text for s in segs]
    units = [u for u in units if u]

    # Char offset where each unit ends in the joined body.
    unit_ends: list[int] = []
    pos = 0
    for i, u in enumerate(units):
        pos += (len(sep) if i else 0) + len(u)
        unit_ends.append(pos)
    body = sep.join(units)

    page_start = min(max(0, start_char), len(body))
    limit = page_start + max(1, max_chars)
    if limit >= len(body):
        page_end, next_start = len(body), None
    else:
        # Last complete line that fits; fall back to a hard cut for a giant line.
        k = bisect_right(unit_ends, limit) - 1
        page_end = unit_ends[k] if k >= 0 and unit_ends[k] > page_start else limit
        next_start = page_end + (len(sep) if page_end in unit_ends else 0)

    out = captions.header + "\n\n" + body[page_start:page_end].lstrip()
    if not units:
        out += "(No caption lines in that time range.)"
    if next_start is not None:
        out += (
            f"\n\n[Truncated: showing chars {page_start}–{page_end} of {len(body)}. "
            f"Call again with start_char={next_start} for more, keeping the same "
            f"timestamps/start/end options (offsets differ between them).]"
        )
    return out


# ---------------------------------------------------------------------------
# Chapters (parsed from the description — the Data API has no chapters field)
# ---------------------------------------------------------------------------

# h:mm:ss or m:ss / mm:ss, not glued to other digits or colons.
_TS = r"(?<![\d:])(?:(\d{1,2}):)?(\d{1,3}):([0-5]\d)(?![\d:])"
# Bullets, list numbers ("1.", "01)"), brackets and dashes around a timestamp.
_LEAD = r"^[\s\-–—•*·>|#\[\(]*(?:\d{1,3}[.)]\s+)?[\[\(]?"
_CHAPTER_START_RE = re.compile(_LEAD + _TS + r"[\]\)]?")
_CHAPTER_END_RE = re.compile(r"[\[\(]?" + _TS + r"[\]\)]?\s*$")
_TITLE_TRIM = " \t-–—:|•*·.,;>"


def _ts_seconds(h: str | None, m: str, s: str) -> int:
    return int(h or 0) * 3600 + int(m) * 60 + int(s)


def parse_chapters(description: str, duration: int = 0) -> tuple[list[dict], bool]:
    """Find a chapter list in a video description.

    A line counts if a timestamp starts it ("0:00 Intro", "1. 02:15 - Setup",
    "[1:02:03] Q&A") or ends it ("Intro - 0:00"); timestamps mid-sentence
    ("at 3:20 he says") are ignored. The longest run of strictly increasing
    timestamps is kept, so stray timestamps elsewhere in the text don't break
    the list.

    Returns (chapters, meets_youtube_rules). YouTube itself only shows chapters
    when the list starts at 0:00, has 3+ entries, each at least 10 seconds.
    """
    found: list[tuple[int, str]] = []
    for raw in (description or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _CHAPTER_START_RE.match(line)
        if m:
            title = line[m.end():]
        else:
            m = _CHAPTER_END_RE.search(line)
            if not m:
                continue
            title = line[: m.start()]
        # A range like "0:00 - 1:30 Intro": drop the second timestamp too.
        title = re.sub(r"^\s*[-–—]\s*" + _TS, "", title)
        t = _ts_seconds(*m.groups())
        if duration and t >= duration:
            continue
        found.append((t, title.strip(_TITLE_TRIM)))

    # Longest strictly increasing run of consecutive timestamped lines.
    best: list[tuple[int, str]] = []
    run: list[tuple[int, str]] = []
    for item in found:
        if run and item[0] <= run[-1][0]:
            run = []
        run.append(item)
        if len(run) > len(best):
            best = list(run)
    if len(best) < 2:
        return [], False

    chapters = []
    for i, (start, title) in enumerate(best):
        end = best[i + 1][0] if i + 1 < len(best) else (duration or None)
        chapters.append(
            {
                "index": i + 1,
                "title": title or f"Chapter {i + 1}",
                "start": fmt_clock(start),
                "start_seconds": start,
                "end_seconds": end,
            }
        )
    lengths = [b[0] - a[0] for a, b in zip(best, best[1:])]
    if duration:
        lengths.append(duration - best[-1][0])
    meets_rules = best[0][0] == 0 and len(best) >= 3 and all(n >= 10 for n in lengths)
    return chapters, meets_rules


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = FastMCP("clipcite")


def threaded(register):
    """Register a blocking function as an async MCP handler run in a worker thread.

    FastMCP 1.x calls plain `def` handlers directly on the event loop, so one
    slow YouTube request would stall every other request, pings and
    cancellations included. Usage: `@threaded(mcp.tool())`. The plain function
    stays importable under its own name (tests call it directly).
    """

    def deco(fn):
        @functools.wraps(fn)  # keeps name, docstring and signature for the schema
        async def handler(*args, **kwargs):
            return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))

        register(handler)
        return fn

    return deco


# ===========================================================================
# TOOLS — actions the AI decides to take mid-conversation
# ===========================================================================
# Errors are raised as ToolError so the client receives isError=true,
# rather than a "successful" result that happens to contain an error string.

@threaded(mcp.tool())
def search_videos(
    query: str,
    channel: str | None = None,
    max_results: int = 10,
    order: SearchOrder = "relevance",
) -> dict:
    """Search YouTube for videos matching a query.

    Args:
        query: Search terms.
        channel: Optional @handle or channel ID to scope the search.
        max_results: 1–50.
        order: relevance | date | rating | viewCount | title.
    """
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max(1, min(50, max_results)),
        "order": order,
    }
    if channel:
        params["channelId"] = require_channel_id(channel)

    resp = run(yt().search().list(**params))
    results = [
        {
            "id": item["id"]["videoId"],
            "title": item["snippet"]["title"],
            "channel": item["snippet"]["channelTitle"],
            "channel_id": item["snippet"]["channelId"],
            "published_at": item["snippet"]["publishedAt"],
            "description": item["snippet"]["description"],
        }
        for item in resp.get("items", [])
    ]
    return {"query": query, "count": len(results), "results": results}


@threaded(mcp.tool())
def get_video_info(video_id_or_url: str) -> dict:
    """Get full metadata for a YouTube video (title, channel, duration, stats)."""
    vid = extract_video_id(video_id_or_url)
    info = _video_info(vid)
    if info is None:
        raise ToolError(f"Video not found: {vid}")
    return info


@threaded(mcp.tool())
def get_transcript(
    video_id_or_url: str,
    languages: list[str] | None = None,
    timestamps: bool = False,
    start: TimeValue | None = None,
    end: TimeValue | None = None,
    max_chars: int = 50_000,
    start_char: int = 0,
) -> str:
    """Fetch the transcript text for a YouTube video.

    The first line says which caption track was used and whether it is
    auto-generated (auto captions often mis-hear names and terms).
    Long transcripts come in pages of up to `max_chars` characters, always
    ending on a complete caption line; a truncated page ends with a note
    giving the `start_char` for the next page.

    Args:
        video_id_or_url: 11-char video id or any YouTube URL.
        languages: Preferred caption languages, e.g. ["en", "en-US"]. Defaults to ["en"].
        timestamps: If true, one caption line per row, prefixed "[mm:ss]".
        start: Only lines starting at or after this time (seconds e.g. 90, or "mm:ss" / "hh:mm:ss").
        end: Only lines starting before this time.
        max_chars: Maximum characters per page (default 50,000).
        start_char: Character offset to start from, for paging. Only valid with
            the same timestamps/start/end values that produced it.
    """
    vid = extract_video_id(video_id_or_url)
    lo = parse_optional_time(start)
    hi = parse_optional_time(end)
    captions = load_segments_or_raise(vid, languages or ["en"])
    if lo is not None and hi is not None and hi <= lo:
        raise ToolError("`end` must be after `start`.")
    return render_transcript(
        captions,
        timestamps=timestamps,
        start=lo,
        end=hi,
        max_chars=max_chars,
        start_char=start_char,
    )


@threaded(mcp.tool())
def search_transcript(
    video_id_or_url: str,
    needle: str,
    languages: list[str] | None = None,
) -> dict:
    """Find every moment in a video's transcript where a word or phrase is said.

    Phrases are matched as whole words across caption-line breaks, ignoring
    case and punctuation ("agent" does not match "agents"; search each form).
    Each result has start/end times, the matching caption text, one line of
    context on each side, how many times the phrase occurs in those lines, and
    a timestamped link. `count` is the number of results; `total_occurrences`
    counts every time the phrase is said. No matches returns count 0, not an
    error.

    Args:
        video_id_or_url: 11-char video id or any YouTube URL.
        needle: Word or phrase to find, e.g. "think like your agents".
        languages: Preferred caption languages. Defaults to ["en"].
    """
    vid = extract_video_id(video_id_or_url)
    if not needle.strip():
        raise ToolError("Search phrase is empty.")
    captions = load_segments_or_raise(vid, languages or ["en"])
    results = find_phrase(captions.segments, vid, needle)
    return {
        "video_id": vid,
        "query": needle,
        "auto_generated_captions": captions.is_generated,
        "count": len(results),
        "total_occurrences": sum(r["occurrences"] for r in results),
        "results": results,
    }


@threaded(mcp.tool())
def get_channel_videos(
    channel: str, max_results: int = 10, order: SearchOrder = "date"
) -> dict:
    """List recent videos from a specific YouTube channel (@handle, UC... id, or name)."""
    ch_id = require_channel_id(channel)
    results = _channel_videos(ch_id, max_results, order)
    return {"channel": channel, "channel_id": ch_id, "count": len(results), "results": results}


@threaded(mcp.tool())
def get_chapters(video_id_or_url: str) -> dict:
    """Get a video's chapter list (title + start/end time + link for each section).

    Chapters are read from the timestamps the creator put in the description,
    the same source YouTube uses for its chapter markers. Costs 1 quota unit.
    If the description has no chapter list, returns count 0 (not an error);
    YouTube's auto-generated chapters are not available through the API.
    `youtube_chapter_rules_met` says whether the list also meets YouTube's
    own rules (starts at 0:00, 3+ chapters, each at least 10 seconds).
    """
    vid = extract_video_id(video_id_or_url)
    info = _video_info(vid)
    if info is None:
        raise ToolError(f"Video not found: {vid}")
    chapters, meets_rules = parse_chapters(info["description"], info["duration_seconds"])
    for c in chapters:
        c["url"] = f"https://youtube.com/watch?v={vid}&t={c['start_seconds']}s"
    return {
        "video_id": vid,
        "title": info["title"],
        "duration_seconds": info["duration_seconds"],
        "source": "description",
        "youtube_chapter_rules_met": meets_rules,
        "count": len(chapters),
        "chapters": chapters,
    }


async def get_video_frames(
    video_id_or_url: str,
    timestamps: list[TimeValue],
    width: int = 640,
) -> list:
    """Extract still frames from a YouTube video at specific timestamps, returned as images.

    Useful for seeing what's on screen (slides, code, diagrams) at moments found
    with search_transcript. Each frame takes a few seconds; the first call for a
    video also spends ~10s resolving the stream.

    Args:
        video_id_or_url: 11-char video id or any YouTube URL.
        timestamps: Up to 10 times, each as seconds (90 or "90"), "mm:ss" or "hh:mm:ss".
        width: Output width in pixels, 160–1280 (default 640). Height keeps aspect ratio.
    """
    vid = extract_video_id(video_id_or_url)
    if not timestamps:
        raise ToolError("Provide at least one timestamp.")
    if len(timestamps) > MAX_FRAMES:
        raise ToolError(f"At most {MAX_FRAMES} timestamps per call (got {len(timestamps)}).")
    seconds = [parse_timestamp(t) for t in timestamps]
    w = max(160, min(1280, width))
    return await anyio.to_thread.run_sync(_extract_frames, vid, seconds, w)


async def sample_video_frames(
    video_id_or_url: str,
    count: int = 6,
    start: TimeValue | None = None,
    end: TimeValue | None = None,
    width: int = 640,
) -> list:
    """Grab evenly spaced frames across a video (or a section of it) for a visual overview.

    Args:
        video_id_or_url: 11-char video id or any YouTube URL.
        count: Number of frames, 1–10 (default 6).
        start: Optional section start (seconds, "mm:ss" or "hh:mm:ss"). Defaults to 0.
        end: Optional section end. Defaults to the end of the video.
        width: Output width in pixels, 160–1280 (default 640).
    """
    vid = extract_video_id(video_id_or_url)
    n = max(1, min(MAX_FRAMES, count))
    w = max(160, min(1280, width))
    lo_given = parse_optional_time(start)  # validate before the slow stream lookup
    hi_given = parse_optional_time(end)

    def run_sampling():
        dur = _stream_info(vid)["duration"]
        lo = lo_given if lo_given is not None else 0.0
        hi = hi_given if hi_given is not None else dur
        if not hi:
            raise ToolError("Video duration unknown; pass an explicit `end`.")
        if dur:
            hi = min(hi, dur - 0.5)
        if hi <= lo:
            raise ToolError("`end` must be after `start` (and within the video).")
        step = (hi - lo) / n
        # Centre of each slice — avoids black first/last frames.
        return _extract_frames(vid, [lo + (i + 0.5) * step for i in range(n)], w)

    return await anyio.to_thread.run_sync(run_sampling)


# Frame tools are registered only when enabled, so with CLIPCITE_ENABLE_FRAMES
# off the assistant never sees them and cannot call them.
if FRAMES_ENABLED:
    mcp.tool(structured_output=False)(get_video_frames)
    mcp.tool(structured_output=False)(sample_video_frames)


# ===========================================================================
# RESOURCES — data the user/host attaches up-front
# ===========================================================================
# NOUNS. The user pins these to the conversation: "here, look at this."
# Same underlying data as some tools, but formatted for *reading* rather
# than for LLM consumption mid-call. Expected failures render as a short
# Markdown document instead of raising.

@threaded(mcp.resource("clipcite://transcript/{video_id}"))
def transcript_resource(video_id: str) -> str:
    """A video transcript as a readable document the user can pin to context."""
    try:
        vid = extract_video_id(video_id)
    except ToolError as e:
        return f"# Invalid video ID\n\n{e}"

    info = _video_info(vid)
    if info is None:
        return f"# Video not found\n\n{vid}"

    try:
        captions = load_segments(vid, ["en"])
    except CouldNotRetrieveTranscript as e:
        return (
            f"# {info['title']}\n\n"
            f"**Channel:** {info['channel']}  \n"
            f"**Duration:** {fmt_duration(info['duration_seconds'])}\n\n"
            f"_{transcript_error_message(e, vid, ['en'])}_"
        )

    out = [
        f"# {info['title']}",
        "",
        f"**Channel:** {info['channel']}  ",
        f"**Duration:** {fmt_duration(info['duration_seconds'])}  ",
        f"**Views:** {info['view_count']:,}  ",
        f"**URL:** {info['url']}  ",
        f"**{captions.header}**",
        "",
        "## Transcript",
        "",
    ]
    # Interleave a timestamp roughly every 30s of content
    last_ts = -30.0
    chunk: list[str] = []
    for seg in captions.segments:
        if seg.start - last_ts >= 30:
            if chunk:
                out.append(" ".join(chunk))
                chunk = []
            t = int(seg.start)
            out.append(f"\n**[{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}]**")
            last_ts = seg.start
        chunk.append(seg.text)
    if chunk:
        out.append(" ".join(chunk))
    return "\n".join(out)


@threaded(mcp.resource("clipcite://channel/{handle}/recent"))
def channel_recent_resource(handle: str) -> str:
    """A channel's recent uploads as a browseable document (@handle or UC... id)."""
    handle = handle.strip()
    # Only add '@' to bare handles — never to channel IDs.
    lookup = handle if _is_channel_id(handle) or handle.startswith("@") else f"@{handle}"
    ch_id = resolve_channel_id(lookup)
    if not ch_id:
        return f"# Channel not found\n\n{handle}"

    ch_items = run(yt().channels().list(part="snippet,statistics", id=ch_id)).get("items", [])
    if not ch_items:
        return f"# Channel not found\n\n{handle}"
    ch = ch_items[0]
    title = ch["snippet"]["title"]
    custom = ch["snippet"].get("customUrl", lookup)
    subs = ch["statistics"].get("subscriberCount")
    subs_line = f"**Subscribers:** {int(subs):,}" if subs and subs.isdigit() else "**Subscribers:** hidden"

    lines = [
        f"# {title} ({custom}) — Recent Uploads",
        subs_line,
        f"**Fetched:** {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    for v in _channel_videos(ch_id, max_results=15, order="date"):
        lines.append(
            f"- **{v['title']}** — `{v['id']}` "
            f"([watch](https://youtube.com/watch?v={v['id']}))"
        )
    return "\n".join(lines)


@threaded(mcp.resource("clipcite://video/{video_id}/info"))
def video_info_resource(video_id: str) -> str:
    """A video's full metadata formatted as a readable card."""
    try:
        vid = extract_video_id(video_id)
    except ToolError as e:
        return f"# Invalid video ID\n\n{e}"
    info = _video_info(vid)
    if info is None:
        return f"# Video not found\n\n{vid}"
    return (
        f"# {info['title']}\n"
        f"**Channel:** {info['channel']}  \n"
        f"**Published:** {info['published_at']}  \n"
        f"**Duration:** {fmt_duration(info['duration_seconds'])}  \n"
        f"**Views:** {info['view_count']:,}  \n"
        f"**Likes:** {info['like_count']:,}  \n"
        f"**Comments:** {info['comment_count']:,}  \n"
        f"**URL:** {info['url']}\n\n"
        f"## Description\n\n{info['description']}"
    )


# ===========================================================================
# PROMPTS — templates the user invokes (slash commands)
# ===========================================================================

@mcp.prompt()
def study_from_video(
    video_url: str = Field(description="YouTube URL or video ID to study from"),
) -> str:
    """Turn a video into a structured study guide."""
    return (
        f"Use get_video_info, get_chapters and get_transcript on `{video_url}`. "
        f"If the video has chapters, use them as the guide's outline.\n"
        f"Then produce a study guide containing:\n"
        f"1. The 3–5 main concepts taught, each with a one-sentence summary.\n"
        f"2. Timestamps for each concept (from chapters, or use search_transcript to locate).\n"
        f"3. Five practice questions of increasing difficulty.\n"
        f"4. Three suggested follow-up videos via search_videos.\n\n"
        f"Format as markdown with clear sections."
    )


@mcp.prompt()
def learning_path_from_channel(
    channel: str = Field(description="Channel handle, e.g. @mitocw"),
    topic: str = Field(description="Topic of interest, e.g. 'quantum physics'"),
    math_level: str = Field(
        default="unsure",
        description="Learner's math comfort: solid / rusty / unsure",
    ),
) -> str:
    """Generate a multi-phase learning path from a channel on a given topic."""
    warm_up = (
        "\n\nGiven math_level is not 'solid', include a Phase 0 diagnostic "
        "and a warm-up before Phase 1."
        if math_level != "solid"
        else ""
    )
    return (
        f"You are helping a learner build a learning path.\n\n"
        f"1. Use search_videos scoped to channel `{channel}` (order=viewCount) "
        f"to find the canonical lecture series on '{topic}'. Prefer complete "
        f"series with high view counts.\n"
        f"2. Use get_channel_videos to confirm the series structure.\n"
        f"3. Break the course into 3–5 phases by conceptual arc.\n"
        f"4. For each phase, output: (a) lecture titles with links, "
        f"(b) prerequisites (including math, if any) specific to that phase with "
        f"one free resource per prereq, (c) one hands-on project that applies "
        f"both the phase's new {topic} material and its prerequisites.\n"
        f"5. End with a single concrete 'start today' action and a 3-question "
        f"sanity check the learner can answer to verify they're ready for the "
        f"next phase.{warm_up}"
    )


@mcp.prompt()
def compare_videos(
    video_a: str = Field(description="First video URL or ID"),
    video_b: str = Field(description="Second video URL or ID"),
) -> str:
    """Compare two videos on the same topic side-by-side."""
    return (
        f"Use get_transcript and get_video_info on both `{video_a}` and `{video_b}`. "
        f"Then produce a comparison covering:\n"
        f"- What each video covers (3-bullet summary each)\n"
        f"- Where they agree and disagree\n"
        f"- Which suits a beginner vs. experienced learner, with reasoning\n"
        f"- A recommended order to watch them in\n\n"
        f"Format as a markdown table where possible."
    )


@mcp.prompt()
def find_the_good_bit(
    video_url: str = Field(description="YouTube URL or ID"),
    topic: str = Field(description="What you're looking for in the video"),
) -> str:
    """Find the timestamped clips in a video that discuss a specific topic."""
    return (
        f"Use search_transcript on `{video_url}` with a needle related to '{topic}'. "
        f"Return the top 3–5 matches with:\n"
        f"- Direct YouTube timestamp link\n"
        f"- 2-sentence summary of what's said at that moment\n"
        f"- Why it's relevant to '{topic}'\n\n"
        f"If matches are sparse, broaden the keyword and try again."
        + (
            " For the best match, call get_video_frames at its timestamp and "
            "describe what is on screen (slides, code, diagrams)."
            if FRAMES_ENABLED
            else ""
        )
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
