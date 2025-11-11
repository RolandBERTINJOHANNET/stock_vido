#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
service/app.py — tiny FastAPI front door for Lovable

Env (server-side)
-----------------
# Providers
OPENAI_API_KEY=...
ELEVEN_API_KEY=...
PIXABAY_KEY=...
KIE_API_KEY=...                  # only used if your pipeline triggers Grok fallback

# Supabase
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_BUCKET=outputs          # optional; defaults to 'outputs'
REQUIRE_SUPABASE=true            # if true: fail request when upload fails (no local fallback)

# API security & CORS
APP_API_KEY=supersecret123       # header: X-API-Key must match
ALLOWED_ORIGINS=https://*.lovable.dev,https://yourapp.lovable.dev

# Paths for dev (local fallback only when REQUIRE_SUPABASE != true)
FFMPEG=/usr/bin/ffmpeg           # optional override
FFPROBE=/usr/bin/ffprobe         # optional override
"""

from __future__ import annotations
import os, uuid, pathlib, tempfile
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# import your pipeline (uses PYTHONPATH=./service in dev; in Docker set ENV PYTHONPATH=/app/service)
from test_parts_1_2 import render_final_video

# -------------------- config --------------------

APP_API_KEY = os.getenv("APP_API_KEY")
ALLOWED = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
REQUIRE_SUPABASE = os.getenv("REQUIRE_SUPABASE", "false").lower() in {"1", "true", "yes"}
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "outputs")

# optional local static mount (used only if REQUIRE_SUPABASE is False)
FILES_DIR = pathlib.Path("./files")
FILES_DIR.mkdir(exist_ok=True)

# Supabase client (best-effort)
sb = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        sb = None  # we'll handle failure paths below

# -------------------- app --------------------

app = FastAPI(title="Stock Vido API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=(ALLOWED if ALLOWED != ["*"] else ["*"]),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# dev convenience: serve local files if not requiring Supabase
app.mount("/files", StaticFiles(directory=str(FILES_DIR), html=False), name="files")

class VideoReq(BaseModel):
    script: str

def _require_key(x_api_key: Optional[str]) -> None:
    # if APP_API_KEY is not set, don't block (dev mode)
    if APP_API_KEY and x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")

def _ffmpeg_ok() -> bool:
    return os.system(f"{os.getenv('FFMPEG', 'ffmpeg')} -version > /dev/null 2>&1") == 0

@app.get("/health")
def health():
    return {"status": "ok", "ffmpeg": _ffmpeg_ok()}


import os, uuid, tempfile
from fastapi import HTTPException
# --- replace in service/app.py ---
import os, uuid, tempfile
from fastapi import HTTPException

SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "outputs")

def _upload_supabase(mp4_bytes: bytes) -> str:
    """
    Upload to Supabase Storage and return a public URL.
    Guarantees the object has Content-Type: video/mp4.
    Raises HTTP 502 if REQUIRE_SUPABASE=true and upload/public URL fails.
    """
    if not sb:
        if REQUIRE_SUPABASE:
            raise HTTPException(status_code=502, detail="supabase client not initialized")
        return ""  # dev fallback

    # sanity: bucket name must be lowercase/alphanumeric/dashes only
    bucket = SUPABASE_BUCKET
    if not bucket or not bucket.replace("-", "").isalnum() or bucket.lower() != bucket:
        raise HTTPException(status_code=502, detail=f"invalid bucket name: {bucket!r} (use lowercase)")

    remote_key = f"videos/{uuid.uuid4().hex}.mp4"

    # write to a temp file; the SDK streams FileIO reliably
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(mp4_bytes)
        tmp_path = tmp.name

    try:
        # IMPORTANT: python client expects hyphenated header keys
        #   - 'content-type' (not 'content_type')
        #   - 'upsert' must be a STRING ("true") with this client
        #   - adding a cache policy is nice for CDN behavior
        with open(tmp_path, "rb") as fh:
            sb.storage.from_(bucket).upload(
                remote_key,
                fh,
                file_options={
                    "content-type": "video/mp4",
                    "cache-control": "public, max-age=3600",
                    "upsert": "true",
                },
            )

        public_url = sb.storage.from_(bucket).get_public_url(remote_key)
        if not public_url:
            if REQUIRE_SUPABASE:
                raise HTTPException(status_code=502, detail="failed to obtain public URL")
            return ""

        return public_url

    except Exception as e:
        if REQUIRE_SUPABASE:
            raise HTTPException(status_code=502, detail=f"supabase upload failed: {e}")
        return ""
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass





def _save_local(mp4_bytes: bytes) -> str:
    """Dev-only fallback: save under /files and return /files/<name> URL."""
    name = f"{uuid.uuid4().hex}.mp4"
    path = FILES_DIR / name
    path.write_bytes(mp4_bytes)
    return f"/files/{name}"

@app.post("/video/orchestrate")
def orchestrate(req: VideoReq, x_api_key: Optional[str] = Header(None)):
    # 1) auth
    _require_key(x_api_key)

    # 2) basic validation
    script = (req.script or "").strip()
    if not script:
        raise HTTPException(status_code=400, detail="script must be a non-empty string")

    # 3) run the pipeline (this may take a bit)
    try:
        final_mp4 = render_final_video(script)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"pipeline failed: {e}")

    # 4) upload to Supabase (or dev fallback)
    url = _upload_supabase(final_mp4)
    if not url:
        # only allowed when REQUIRE_SUPABASE is false
        url = _save_local(final_mp4)

    return {"status": "succeeded", "video_url": url}
