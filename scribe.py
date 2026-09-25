"""
SCRIBE — paste a video link, get a downloadable transcript.
===========================================================
Give it a URL (TikTok, YouTube, Instagram, X/Twitter, Facebook — anything yt-dlp supports); SCRIBE pulls the
audio, transcribes it locally with Whisper, and hands back a clean transcript you can download as TXT, SRT,
VTT, or Markdown to reuse in another platform (blog, captions, notes, show-notes).

Everything runs on THIS machine — the audio and transcript never leave it (no cloud transcription service).

Run:  pip install fastapi uvicorn yt-dlp faster-whisper
      python scribe.py            →  http://127.0.0.1:8970
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
import urllib.request

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_SIZE = os.getenv("SCRIBE_MODEL", "base.en")   # base.en=fast English; "base"/"small" = multilingual
MAX_DURATION = int(os.getenv("SCRIBE_MAX_DURATION", "1800"))   # reject clips longer than this (sec); 0=no cap
MAX_CONCURRENT = int(os.getenv("SCRIBE_MAX_CONCURRENT", "1"))  # transcriptions allowed to run at once (CPU guard)
_model = None   # lazy-loaded Whisper model, reused across requests
_sem = asyncio.Semaphore(MAX_CONCURRENT)   # protects a small box from being swamped by parallel jobs


class DurationError(RuntimeError):
    """Raised when a video is longer than the configured cap."""


# ── AI layer (optional) — routes transcript text to an OpenAI-compatible LLM (Groq by default) ───────
# Transcription stays 100% local; these features send the *text* to the configured LLM. Set GROQ_API_KEY
# (or SCRIBE_LLM_KEY) to enable — with no key, /api/ai returns a friendly "not configured" message.
LLM_KEY = os.getenv("GROQ_API_KEY") or os.getenv("SCRIBE_LLM_KEY") or ""
LLM_BASE = os.getenv("SCRIBE_LLM_BASE", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("SCRIBE_LLM_MODEL", "openai/gpt-oss-120b")

_AI_SYS = {
    "summary": ("You summarize video transcripts. Reply in Markdown: a one-paragraph TL;DR, then a "
                "'## Key points' list of 3–7 bullets. Be faithful to the transcript; never invent facts."),
    "chapters": ("You split a timestamped transcript into chapters. Return a Markdown list, one per line, "
                 "each `mm:ss — Chapter title`. Choose 3–8 natural sections. Use only timestamps present in "
                 "the input. Titles are short (2–6 words)."),
    "chat": ("You answer questions strictly from the provided video transcript. If the transcript does not "
             "contain the answer, say so plainly. Be concise and quote the transcript where useful."),
    "blog": ("You are a content writer. Turn this transcript into a structured blog post in Markdown: a "
             "compelling '# title', a short intro, 2–4 '## sections', and a brief conclusion. Preserve the "
             "speaker's meaning and improve clarity. Do not fabricate facts."),
    "x": ("You write social threads. Turn this transcript into an X/Twitter thread: 5–8 numbered tweets, each "
          "≤ 280 characters, a strong hook first, one idea per tweet, at most 2 hashtags at the very end."),
    "linkedin": ("You are a LinkedIn ghostwriter. Turn this transcript into ONE professional post: a strong "
                 "first-line hook, short punchy paragraphs, a clear takeaway, and 3 relevant hashtags at the end."),
    "newsletter": ("You are a newsletter editor. Turn this transcript into an email section: a subject line, a "
                   "warm intro, key insights as short paragraphs/bullets, and a one-line sign-off."),
    "clips": ("You find the most shareable short-video moments in a timestamped transcript. Return STRICT JSON "
              "ONLY — no prose, no code fences: an array of 3–6 objects, each with keys \"start\" (seconds, "
              "integer), \"end\" (seconds, integer), \"title\" (a hook of ≤ 8 words), \"reason\" (one line on "
              "why it grabs attention — emotion, insight, or a strong quote), \"caption\" (a ready-to-post "
              "caption), and \"hashtags\" (array of 3–5 tags, no # sign). Each clip is 15–30 seconds long and "
              "uses only timestamps within the transcript. Order the array best-first."),
}


def _extract_json(raw: str):
    """Pull a JSON array/object out of an LLM reply (tolerates code fences / stray prose)."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("\n") + 1:] if "\n" in s else s
    a, b = s.find("["), s.rfind("]")
    if a != -1 and b > a:
        try:
            return json.loads(s[a:b + 1])
        except Exception:  # noqa: BLE001
            pass
    return raw   # fall back to raw text if it isn't parseable JSON


def _llm(system: str, user: str, max_tokens: int = 900, temperature: float = 0.4) -> str:
    """Call the configured OpenAI-compatible chat endpoint. Blocking (run in a thread)."""
    if not LLM_KEY:
        raise RuntimeError("AI features aren't enabled on this instance (no LLM key configured).")
    payload = json.dumps({
        "model": LLM_MODEL, "temperature": temperature, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(
        LLM_BASE.rstrip("/") + "/chat/completions", data=payload,
        # a real User-Agent avoids Cloudflare's bot-fingerprint block (error 1010) in front of Groq
        headers={"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json",
                 "User-Agent": "curl/8.5.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        data = json.loads(r.read().decode())
    return data["choices"][0]["message"]["content"].strip()


def run_ai(body: dict) -> str:
    """Dispatch one AI task over transcript text. Blocking (run in a thread)."""
    task = (body.get("task") or "").lower()
    text = (body.get("text") or "").strip()[:16000]          # bound tokens
    if not text:
        raise ValueError("no transcript text provided")
    if task == "summary":
        return _llm(_AI_SYS["summary"], text, 700)
    if task == "chapters":
        segs = body.get("segments") or []
        timed = "\n".join(
            f"[{int(s['start'] // 60):02d}:{int(s['start'] % 60):02d}] {s['text']}" for s in segs)[:16000]
        return _llm(_AI_SYS["chapters"], timed or text, 700)
    if task == "chat":
        q = (body.get("question") or "").strip()
        if not q:
            raise ValueError("no question provided")
        return _llm(_AI_SYS["chat"], f"TRANSCRIPT:\n{text}\n\nQUESTION: {q}", 600)
    if task == "repurpose":
        target = (body.get("target") or "blog").lower()
        return _llm(_AI_SYS.get(target, _AI_SYS["blog"]), text, 1300)
    if task == "clips":
        segs = body.get("segments") or []
        timed = "\n".join(f"[{int(s['start'])}s] {s['text']}" for s in segs)[:16000]
        return _extract_json(_llm(_AI_SYS["clips"], timed or text, 1100, 0.5))
    raise ValueError(f"unknown AI task: {task!r}")


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        print(f"[scribe] loading Whisper '{MODEL_SIZE}' …")
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        print("[scribe] model ready")
    return _model


# ── download the audio/video for a URL (yt-dlp; falls back to browser cookies) ──────────────────────
def _download(url: str, workdir: str) -> dict:
    import yt_dlp
    out = os.path.join(workdir, "media.%(ext)s")
    # we only need audio to transcribe — "bestaudio" is smaller/faster and avoids YouTube's
    # video/audio split (where a single "best" progressive stream often isn't offered)
    base = {"outtmpl": out, "format": "bestaudio/best", "quiet": True, "no_warnings": True,
            "noplaylist": True, "restrictfilenames": True}
    attempts = [{}]                                            # public vids need no cookies
    # only reach for browser cookies where a Chrome profile actually exists (a logged-in desktop);
    # on a headless server it's pointless and its error would mask the real one
    if os.name == "nt" or os.path.isdir(os.path.expanduser("~/.config/google-chrome")):
        attempts.append({"cookiesfrombrowser": ("chrome",)})
    last = None
    for extra in attempts:
        try:
            with yt_dlp.YoutubeDL(dict(base, **extra)) as ydl:
                probe = ydl.extract_info(url, download=False)   # look before we leap
                dur = probe.get("duration")
                if MAX_DURATION and dur and dur > MAX_DURATION:
                    raise DurationError(
                        f"that clip is {int(dur // 60)} min long — this instance caps transcriptions at "
                        f"{MAX_DURATION // 60} min. Try a shorter video.")
                info = ydl.extract_info(url, download=True)
                fn = ydl.prepare_filename(info)
            if os.path.exists(fn):
                return {"path": fn, "title": info.get("title") or "",
                        "uploader": info.get("uploader") or info.get("channel") or "",
                        "duration": info.get("duration"), "caption": (info.get("description") or "")[:2000],
                        "webpage_url": info.get("webpage_url") or url}
        except DurationError:
            raise                     # a length cap is final — don't retry with cookies
        except Exception as e:  # noqa: BLE001
            if last is None:          # keep the first (bare) attempt's error — the informative one
                last = e
    raise RuntimeError(f"could not fetch the video ({type(last).__name__}: {str(last)[:160]})")


def _ts(sec: float, comma: bool = True) -> str:
    h, r = divmod(max(0.0, sec), 3600)
    m, s = divmod(r, 60)
    ms = int((s - int(s)) * 1000)
    sep = "," if comma else "."
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{ms:03d}"


def _srt(segs: list[dict]) -> str:
    return "\n".join(f"{i}\n{_ts(s['start'])} --> {_ts(s['end'])}\n{s['text'].strip()}\n"
                     for i, s in enumerate(segs, 1))


def _vtt(segs: list[dict]) -> str:
    body = "\n".join(f"{_ts(s['start'], False)} --> {_ts(s['end'], False)}\n{s['text'].strip()}\n" for s in segs)
    return "WEBVTT\n\n" + body


def transcribe_url(url: str) -> dict:
    """Download → transcribe → return transcript in several formats. Blocking (run in a thread)."""
    work = tempfile.mkdtemp(prefix="scribe_")
    try:
        meta = _download(url, work)
        segments, info = _get_model().transcribe(meta["path"], vad_filter=True)
        segs = [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in segments]
        text = " ".join(s["text"] for s in segs).strip()
        return {"ok": True, "title": meta["title"], "uploader": meta["uploader"],
                "duration": meta["duration"], "caption": meta["caption"], "source": meta["webpage_url"],
                "language": getattr(info, "language", "") or "", "text": text,
                "segments": segs, "srt": _srt(segs), "vtt": _vtt(segs), "words": len(text.split())}
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ── API ─────────────────────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="SCRIBE — video → transcript")


@app.post("/api/transcribe")
async def api_transcribe(req: Request):
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid JSON"}, status_code=400)
    url = (body.get("url") or "").strip()
    if not url.startswith("http"):
        return JSONResponse({"ok": False, "error": "paste a full video URL (starting with http)."}, status_code=400)
    t0 = time.time()
    try:
        async with _sem:                       # one heavy job at a time on a small box
            res = await asyncio.to_thread(transcribe_url, url)
    except DurationError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=413)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)
    res["elapsed_s"] = round(time.time() - t0, 1)
    return JSONResponse(res)


@app.post("/api/ai")
async def api_ai(req: Request):
    """Summary / chapters / repurpose / chat over an already-produced transcript."""
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid JSON"}, status_code=400)
    try:
        async with _sem:                       # share the CPU/rate budget with transcription
            out = await asyncio.to_thread(run_ai, body)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)
    return JSONResponse({"ok": True, "task": (body.get("task") or "").lower(), "result": out})


@app.post("/api/docx")
async def api_docx(req: Request):
    """Build a Word (.docx) file from transcript data supplied by the client."""
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid JSON"}, status_code=400)

    def build() -> bytes:
        import io
        from docx import Document
        doc = Document()
        doc.add_heading(body.get("title") or "Transcript", level=1)
        meta = " — ".join(x for x in (body.get("uploader"), body.get("source")) if x)
        if meta:
            doc.add_paragraph(meta)
        segs = body.get("segments") or []
        if body.get("timestamps") and segs:
            for s in segs:
                mm, ss = int(s["start"] // 60), int(s["start"] % 60)
                doc.add_paragraph(f"[{mm:02d}:{ss:02d}]  {s['text']}")
        else:
            doc.add_paragraph(body.get("text") or "")
        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()

    try:
        data = await asyncio.to_thread(build)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"could not build .docx ({e})"}, status_code=502)
    base = "".join(c for c in (body.get("title") or "transcript") if c.isalnum() or c in " -_").strip()[:60] or "transcript"
    return Response(
        data, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{base}.docx"'})


@app.get("/health")
def health():
    return {"ok": True, "service": "scribe", "model": MODEL_SIZE, "ai": bool(LLM_KEY)}


@app.get("/logo.svg")
def logo():
    from fastapi.responses import Response
    try:
        svg = open(os.path.join(_HERE, "static", "logo.svg"), encoding="utf-8").read()
        return Response(svg, media_type="image/svg+xml")
    except FileNotFoundError:
        return Response("", media_type="image/svg+xml", status_code=404)


@app.get("/", response_class=HTMLResponse)
def index():
    try:
        return open(os.path.join(_HERE, "static", "index.html"), encoding="utf-8").read()
    except FileNotFoundError:
        return "<h1>SCRIBE</h1><p>frontend missing</p>"


def main():
    import uvicorn
    print("🎙️  SCRIBE — http://127.0.0.1:8970")
    uvicorn.run("scribe:app", host="127.0.0.1", port=8970, log_level="warning")


if __name__ == "__main__":
    main()
