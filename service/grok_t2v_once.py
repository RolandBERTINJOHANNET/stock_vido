#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
text_to_video(prompt, *, api_key=None, aspect_ratio="3:2", mode="normal", poll_interval=5, timeout=600) -> bytes

Creates a Kie.ai Grok Imagine text-to-video job, polls until completion, and returns the MP4 bytes.

Setup:
  pip install requests
  export KIE_API_KEY="YOUR_API_KEY"

Import usage:
  from grok_text2video import text_to_video
  mp4_bytes = text_to_video("A tiny robot barista pours latte art in a cozy café; cinematic, warm lighting.")
  with open("grok_video.mp4", "wb") as f:
      f.write(mp4_bytes)

CLI usage:
  python grok_text2video.py "your prompt here"
"""

import os
import sys
import time
import json
import requests
from typing import Tuple, List, Optional

API_BASE = "https://api.kie.ai"
CREATE_URL = f"{API_BASE}/api/v1/jobs/createTask"
RECORD_URL = f"{API_BASE}/api/v1/jobs/recordInfo"

class GrokT2VError(RuntimeError):
    pass

def _create_task(prompt: str, api_key: str, *, aspect_ratio: str, mode: str) -> str:
    payload = {
        "model": "grok-imagine/text-to-video",
        "input": {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "mode": mode,
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    r = requests.post(CREATE_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    js = r.json()
    if js.get("code") != 200 or "data" not in js or "taskId" not in js["data"]:
        raise GrokT2VError(f"CreateTask failed: {js}")
    return js["data"]["taskId"]

def _record_info(task_id: str, api_key: str) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    r = requests.get(RECORD_URL, headers=headers, params={"taskId": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def _parse_state_and_urls(info: dict) -> Tuple[Optional[str], List[str]]:
    data = info.get("data") or {}
    # state can be 'waiting' | 'generating' | 'success' | 'fail' (strings may vary)
    state = data.get("state") or data.get("status")
    urls: List[str] = []
    # Prefer resultJson.resultUrls if present
    result_json = data.get("resultJson")
    if isinstance(result_json, str):
        try:
            parsed = json.loads(result_json)
            if isinstance(parsed, dict):
                urls = list(parsed.get("resultUrls") or [])
        except Exception:
            pass
    # Some variants place it under data.response
    if not urls and isinstance(data.get("response"), dict):
        resp = data["response"]
        urls = list(resp.get("resultUrls") or resp.get("originUrls") or [])
    return state, urls

def _download(url: str) -> bytes:
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        chunks = []
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                chunks.append(chunk)
        return b"".join(chunks)

def text_to_video(
    prompt: str,
    *,
    api_key: Optional[str] = None,
    aspect_ratio: str = "3:2",
    mode: str = "normal",
    poll_interval: int = 5,
    timeout: int = 10 * 60,
) -> bytes:
    """
    Submit a Grok Imagine text-to-video job and return the resulting MP4 bytes.

    Args:
        prompt: Text description for the video.
        api_key: Kie.ai API key. If None, reads KIE_API_KEY from the environment.
        aspect_ratio: "2:3", "3:2", or "1:1".
        mode: "normal" (or other modes supported by the API).
        poll_interval: Seconds between status polls.
        timeout: Max seconds to wait for completion.

    Returns:
        Raw MP4 bytes.

    Raises:
        GrokT2VError on API or job failure.
        TimeoutError if the job doesn’t complete in time.
    """
    key = api_key or os.getenv("KIE_API_KEY")
    if not key:
        raise GrokT2VError("Missing API key: set api_key=... or export KIE_API_KEY")

    task_id = _create_task(prompt, key, aspect_ratio=aspect_ratio, mode=mode)

    deadline = time.time() + timeout
    while True:
        info = _record_info(task_id, key)
        state, urls = _parse_state_and_urls(info)

        if state and str(state).lower() in {"success", "succeeded", "1"}:
            if not urls:
                raise GrokT2VError(f"Task succeeded but no result URLs found: {info}")
            # Return the first result (expected to be a video)
            return _download(urls[0])

        if state and str(state).lower() in {"fail", "failed", "error", "2"}:
            raise GrokT2VError(f"Task failed: {json.dumps(info, indent=2)}")

        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for task {task_id}: last_state={state}, urls={urls}")

        time.sleep(poll_interval)

# --------------------- CLI demo ---------------------
if __name__ == "__main__":
    user_prompt = (
        "A tiny robot barista pours latte art in a cozy café; slow cinematic pan, warm lighting, steam swirling."
        if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    )
    try:
        mp4 = text_to_video(user_prompt)
        out = "grok_video.mp4"
        with open(out, "wb") as f:
            f.write(mp4)
        print(f"Saved: {out}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
