"""The CLIPCITE_ENABLE_FRAMES switch.

Value parsing is tested in-process. Registration is tested by importing the
server in a fresh interpreter, because the switch is read once at import time
and decides which tools get registered.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SERVER_DIR)

import server  # noqa: E402

FRAME_TOOLS = {"get_video_frames", "sample_video_frames"}


@pytest.mark.parametrize("value", [None, "", "true", "1", "yes", "TRUE"])
def test_switch_values_that_enable(monkeypatch, value):
    monkeypatch.delenv(server.FRAMES_ENV, raising=False)
    if value is not None:
        monkeypatch.setenv(server.FRAMES_ENV, value)
    assert server.frames_enabled() is True  # only an explicit "off" value disables


@pytest.mark.parametrize("value", ["false", "0", "no", "off", " FALSE ", "Off"])
def test_switch_values_that_disable(monkeypatch, value):
    monkeypatch.setenv(server.FRAMES_ENV, value)
    assert server.frames_enabled() is False


PROBE = r"""
import asyncio, json, sys
import server
tools = [t.name for t in asyncio.run(server.mcp.list_tools())]
prompt = server.find_the_good_bit(video_url="x", topic="y")
print(json.dumps({
    "tools": tools,
    "prompt_mentions_frames": "get_video_frames" in prompt,
    "yt_dlp_imported": "yt_dlp" in sys.modules,
}))
"""


def probe(value, tmp_path):
    """Import a copy of server.py in a fresh interpreter, isolated from the real .env.

    server.py calls load_dotenv(), which finds the nearest .env walking up from
    the file. Running a copy next to an empty .env means a developer's own
    CLIPCITE_ENABLE_FRAMES setting can't leak into the test.
    """
    shutil.copy(os.path.join(SERVER_DIR, "server.py"), tmp_path / "server.py")
    (tmp_path / ".env").write_text("")
    env = {k: v for k, v in os.environ.items() if k != server.FRAMES_ENV}
    if value is not None:
        env[server.FRAMES_ENV] = value
    out = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_enabled_by_default_registers_frame_tools(tmp_path):
    r = probe(None, tmp_path)
    assert FRAME_TOOLS <= set(r["tools"])
    assert r["prompt_mentions_frames"] is True
    assert r["yt_dlp_imported"] is False  # imported lazily, on first frame call


def test_default_ignores_the_projects_own_env_file(tmp_path):
    """Regression: a developer .env with frames off further up must not leak in."""
    (tmp_path / ".env").write_text(f"{server.FRAMES_ENV}=false\n")
    (tmp_path / "proj").mkdir()
    r = probe(None, tmp_path / "proj")  # probe's empty .env is nearer, so it wins
    assert FRAME_TOOLS <= set(r["tools"])


def test_disabled_hides_frame_tools_and_never_imports_yt_dlp(tmp_path):
    r = probe("false", tmp_path)
    assert not FRAME_TOOLS & set(r["tools"])
    assert "get_transcript" in r["tools"]  # everything else still there
    assert r["prompt_mentions_frames"] is False
    assert r["yt_dlp_imported"] is False
