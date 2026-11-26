# ClipCite

An MCP server that lets an AI assistant search YouTube, read and search video transcripts, and look at video frames. It demonstrates all three MCP primitives:

- **Tools:** `search_videos`, `get_video_info`, `get_chapters`, `get_transcript`, `search_transcript`, `get_channel_videos`, `get_video_frames`, `sample_video_frames`
- **Resources:** `clipcite://transcript/{video_id}`, `clipcite://video/{video_id}/info`, `clipcite://channel/{handle}/recent`
- **Prompts:** `study_from_video`, `learning_path_from_channel`, `compare_videos`, `find_the_good_bit`

> [!WARNING]
> **Disclaimer: frame fetching violates YouTube's Terms of Service.**
>
> The frame tools (`get_video_frames` and `sample_video_frames`) use [yt-dlp](https://github.com/yt-dlp/yt-dlp) to download parts of YouTube video streams outside the official API. [YouTube's Terms of Service](https://www.youtube.com/t/terms) do not allow downloading content, or accessing the service by automated means, without YouTube's permission. Using these tools breaks those terms.
>
> - **Your API key is at risk.** [YouTube's API Developer Policies](https://developers.google.com/youtube/terms/developer-policies) also forbid apps that use the API from downloading video outside it. Running the frame tools alongside your API key can get your Google Cloud project's API access suspended.
> - **Videos are copyrighted.** Frames belong to the video's creator. You are responsible for how you use them.
> - **Personal and educational use only.** Do not deploy the frame tools in a public or hosted service, and do not redistribute extracted frames.
> - **No affiliation, no warranty.** ClipCite is not affiliated with or endorsed by YouTube or Google. It is provided as-is, and you use it at your own risk.
>
> If you can't accept these risks, turn the frame tools off with `CLIPCITE_ENABLE_FRAMES=false` (see [Turning off frame fetching](#turning-off-frame-fetching)). Every other tool works without them.

## Setup

1. Install Python 3.12+ and create a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. **Get your own YouTube Data API key.** ClipCite doesn't include one, and every user needs their own. It's free:
   1. In the [Google Cloud console](https://console.cloud.google.com/), create a project (or pick an existing one).
   2. Under **APIs & Services → Library**, enable **YouTube Data API v3**.
   3. Under **APIs & Services → Credentials**, choose **Create credentials → API key**.
   4. Edit the key and, under **API restrictions**, restrict it to the YouTube Data API v3.

   Then copy `.env.example` to `.env` and paste the key after `YOUTUBE_API_KEY=`. All API calls are made with your key and count against your Google Cloud project.

   **Keep the key private.** `.env` is listed in `.gitignore`, so never commit it, paste it into an issue, or share it in a screenshot. If a key is ever exposed, delete it in the Cloud console and create a new one. ClipCite also removes the key from any API error message before passing it to the assistant.

3. For the frame tools only, install [ffmpeg](https://ffmpeg.org/) and a JavaScript runtime (Node or Deno), both on `PATH`. Skip this step if you turn the frame tools off.

Open the server in the MCP Inspector:

```powershell
npx -y @modelcontextprotocol/inspector .venv\Scripts\python.exe server.py
```

Run the tests (offline):

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest tests -q
```

## How transcripts are fetched

Transcripts do **not** come from the YouTube Data API. The official API can list a video's caption tracks, but its download endpoint (`captions.download`) requires OAuth sign-in as someone allowed to edit the video. In practice, only the video's owner can download its captions that way.

Instead, ClipCite uses the open-source [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) library. It reads the same caption tracks that the YouTube web player shows. What that means:

- **It's a Terms of Service grey area.** The library reads YouTube outside the official API, which is automated access that [YouTube's Terms of Service](https://www.youtube.com/t/terms) don't allow without permission. [YouTube's API Developer Policies](https://developers.google.com/youtube/terms/developer-policies) also expect apps that use the API to get YouTube data through the API, not by scraping. This is milder than frame fetching: no video is downloaded, and captions are text YouTube already sends to every viewer. Use it for personal and educational purposes only. If you deploy ClipCite for other people, or apply to Google for more API quota (which involves a compliance review), the transcript tools are a risk to weigh.
- **It's unofficial.** The library depends on how YouTube's player works, so a YouTube change can break it until the library is updated. If transcripts start failing, upgrade it first: `pip install -U youtube-transcript-api`.
- **It can be blocked.** YouTube often blocks these requests from cloud servers and VPNs. The tools then return a clear "blocked" error. Running from a home network, or configuring a proxy in the library, usually fixes it.
- **Auto-generated captions are imperfect.** Many videos only have speech-to-text captions, which mis-hear names and terms (for example "cloud" for "Claude"). `get_transcript` starts with a header saying which track was used, and `search_transcript` returns an `auto_generated_captions` flag, so the assistant knows when quotes may be inexact.
- **Captions are cached in memory** per video and language, so searching a video and then reading its transcript fetches it once. The cache clears when the server restarts.

## How chapters are found

The YouTube Data API has no chapters field, so `get_chapters` reads the timestamps the creator wrote in the video description. That's the same source YouTube uses for its own chapter markers.

- A line counts as a chapter if it starts or ends with a timestamp, such as `0:00 Intro`, `1. 02:15 - Setup`, `[1:02:03] Q&A` or `Intro - 0:00`. Timestamps in the middle of a sentence are ignored.
- The longest run of increasing timestamps is kept, so a stray "skip to 8:00" elsewhere in the description doesn't break the list.
- `youtube_chapter_rules_met` says whether the list also meets YouTube's own rules: it starts at 0:00, has at least 3 chapters, and each lasts at least 10 seconds.
- A video with no chapter list returns `count: 0`, not an error. Chapters that YouTube generates automatically aren't available through the API, so they're not included.

## How frames are fetched

`get_video_frames` and `sample_video_frames` use [yt-dlp](https://github.com/yt-dlp/yt-dlp) to find the video's stream, then ffmpeg reads only the few seconds needed for each frame.

**This violates YouTube's Terms of Service.** Fetching only a few seconds is still downloading. Read the [disclaimer](#clipcite) at the top before using these tools.

### Turning off frame fetching

Set `CLIPCITE_ENABLE_FRAMES=false` in `.env`, or in the `env` block of your MCP client's config, then restart the server:

```json
"clipcite": {
  "command": "C:\\path\\to\\clipcite\\.venv\\Scripts\\python.exe",
  "args": ["C:\\path\\to\\clipcite\\server.py"],
  "env": { "CLIPCITE_ENABLE_FRAMES": "false" }
}
```

When it's off:

- `get_video_frames` and `sample_video_frames` aren't registered at all, so the assistant never sees them and can't call them.
- yt-dlp is never imported, so you can uninstall it (`pip uninstall yt-dlp`). ffmpeg and a JavaScript runtime aren't needed either.
- The `find_the_good_bit` prompt stops suggesting frames.

`false`, `0`, `no` and `off` all turn it off. Leaving it unset, or any other value, keeps the frame tools on.
