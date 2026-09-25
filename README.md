# 🎙️ SCRIBE — video → transcript

**Paste any video link → get a clean transcript you can download and reuse anywhere.**

SCRIBE pulls the audio from almost any video URL (TikTok, YouTube, Instagram, X, Facebook, Vimeo, Reddit,
Twitch, news sites — 1000+ platforms via `yt-dlp`), transcribes it **locally** with Whisper, and hands back a
transcript you can export in a dozen formats or transform with AI.

The transcription runs entirely on the machine — **the audio never touches a cloud transcription service.**

---

## Why it exists

Creators, students, and researchers constantly need the *words* out of a video — for captions, blog posts,
show-notes, study notes, or feeding another tool. Existing options either upload your audio to a paid cloud
service or lock the transcript behind a subscription. SCRIBE keeps transcription **local, private, and free**,
and then adds the "so what do I do with it now?" layer on top.

## What it does

### 📝 Transcription
- Any `yt-dlp`-supported URL → audio → transcript with timestamps.
- Powered by [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) (`base.en` by default, CPU, int8).
- Only the **audio** is downloaded (`bestaudio/best`) — smaller and faster than pulling full video.

### 📦 Export formats (10)
| For | Formats |
|-----|---------|
| Writing / docs | **TXT**, **Markdown**, **DOCX** (Word) |
| Captions / subtitles | **SRT**, **VTT**, **SBV** (YouTube), **TTML** (broadcast / Final Cut) |
| Data / developers | **JSON**, **CSV** |
| Publishing | **HTML** — a styled transcript whose timestamps link back to the source video at that moment |

Most formats are generated client-side from the transcript JSON; DOCX is built server-side with `python-docx`.

### ✨ AI tools (optional)
Once you have a transcript, one click transforms it:
- **Summarize** — TL;DR + key points
- **Chapters** — timestamped section list (like YouTube chapters)
- **Repurpose** — turn it into a blog post, X/Twitter thread, LinkedIn post, or newsletter
- **Chat** — ask a question, answered *strictly from the transcript*

> **Privacy note:** transcription is always local. The AI features send the transcript **text** to an
> OpenAI-compatible LLM (Groq by default). With no key configured, the AI panel simply reports "not enabled" and
> the rest of the app works unchanged.

---

## Architecture

```
Browser (static/index.html)
        │  POST /api/transcribe {url}
        ▼
FastAPI (scribe.py, :8970)
   ├─ yt-dlp            → download audio
   ├─ faster-whisper    → transcript + segments
   ├─ /api/ai           → summary · chapters · repurpose · chat  (→ Groq, OpenAI-compatible)
   └─ /api/docx         → Word file (python-docx)
```

In production it runs behind **nginx** (reverse proxy on `:80`) with the FastAPI app under a systemd service,
fronted by a Cloudflare Tunnel.

### Guardrails for public hosting
- `SCRIBE_MAX_DURATION` — reject clips longer than N seconds (a friendly `413` *before* downloading).
- `SCRIBE_MAX_CONCURRENT` — cap simultaneous transcriptions so a small box isn't swamped.
- The LLM key is read from the environment only — never hardcoded, never committed.

---

## Run it locally

```bash
pip install -r requirements.txt
python scribe.py          # → http://127.0.0.1:8970
```

Optional — enable the AI features (any OpenAI-compatible endpoint; Groq shown):

```bash
export GROQ_API_KEY=...            # your key
export SCRIBE_LLM_MODEL=openai/gpt-oss-120b
python scribe.py
```

## Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `SCRIBE_MODEL` | `base.en` | Whisper model (`base`/`small` = multilingual) |
| `SCRIBE_MAX_DURATION` | `1800` | Max clip length in seconds (`0` = no cap) |
| `SCRIBE_MAX_CONCURRENT` | `1` | Concurrent transcriptions |
| `GROQ_API_KEY` / `SCRIBE_LLM_KEY` | — | Enables AI features |
| `SCRIBE_LLM_BASE` | `https://api.groq.com/openai/v1` | OpenAI-compatible base URL |
| `SCRIBE_LLM_MODEL` | `openai/gpt-oss-120b` | Chat model |

## Known limitations
- **Cloudflare's free edge cuts responses at ~100s.** Behind a tunnel, keep `SCRIBE_MAX_DURATION` low (e.g.
  120s) or move transcription to an async job model to lift it.
- **Some platforms (TikTok, Instagram, X) block headless/server fetches** or require login cookies. A logged-in
  desktop can supply them via `cookiesfrombrowser`; a headless server usually can't. Public sources like
  YouTube work server-side.
- Whisper is CPU-bound here — sized for personal / demo use, not heavy concurrent load.

## Stack
FastAPI · faster-whisper · yt-dlp · python-docx · vanilla JS front-end · nginx · Cloudflare Tunnel

---

*Transcription is local and private. AI features are opt-in and clearly labeled.*
