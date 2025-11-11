#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expose a single function:
    speak_tagged_sentence(line: str) -> bytes

- Validates that `line` is exactly ONE sentence and starts with a bracketed tag like: [excited] Hello there.
- Calls ElevenLabs v3 exactly as in the provided script (voice picking via env, v3 model, stability=0.0, retries, streaming-safe).
- Returns the AUDIO BYTES (not a file path).

Env (required/optional)
-----------------------
  ELEVEN_API_KEY       (required)
  ELEVEN_VOICE_ID      (optional; exact voice id)
  ELEVEN_VOICE_NAME    (optional; fuzzy name match if you don’t know the ID)

Install
-------
  pip install elevenlabs python-dotenv
"""

from __future__ import annotations
import os
import re
import time
import json
from io import BytesIO
from typing import Optional

try:
    from dotenv import load_dotenv  # optional
except Exception:
    def load_dotenv() -> None:
        return None

from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError

# httpx is used under the hood by the SDK; we import common exceptions for robust retry handling.
try:
    import httpx
    HTTPX_ERRORS = (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError, httpx.NetworkError)
except Exception:
    HTTPX_ERRORS = tuple()

JSONDecodeError = json.JSONDecodeError

MODEL_ID = "eleven_v3"               # force v3 for tag responsiveness
DEFAULT_VOICE_ID = "gUABw7pXQjhjt0kNFBTF"  # Bella, decent default


def speak_tagged_sentence(line: str) -> bytes:
    """
    Convert ONE tagged sentence to speech and return raw audio bytes (MP3 44.1k/128kbps).

    Validation:
      - Must start with a bracketed tag: e.g., "[excited] Bonjour !"
      - Must be exactly ONE sentence (no additional sentence-ending punctuation after the first).

    Raises:
      ValueError on validation failures.
      SystemExit or ApiError on fatal API/transport errors after retries.
    """
    load_dotenv()

    # ---------------- Validation: bracketed tag + single sentence ----------------
    if not isinstance(line, str):
        raise ValueError("Input must be a string.")

    text = line.strip()
    if not text:
        raise ValueError("Input is empty.")

    # Must start with [tag]
    if not re.match(r'^\[[^\[\]]+\]\s*\S', text):
        raise ValueError("Sentence must start with a bracketed tag like: [excited] Hello there.")

    # Remove the leading [tag] for sentence counting
    body = re.sub(r'^\[[^\[\]]+\]\s*', '', text).strip()

    # Must be a single sentence (simple heuristic: exactly one segment before an end punctuation cluster)
    # Treat . ! ? … as sentence enders; allow commas/semicolons/colons.
    segments = [s for s in re.split(r'[.!?…]+', body) if s.strip()]
    if len(segments) != 1:
        raise ValueError("Provide exactly ONE sentence after the tag.")

    # Reject hard newlines which often signal multiple sentences/lines
    if "\n" in body or "\r" in body:
        raise ValueError("Sentence must be a single line without newlines.")

    # ---------------- Client & voice selection ----------------
    api_key = os.getenv("ELEVEN_API_KEY")
    if not api_key:
        raise SystemExit("Missing ELEVEN_API_KEY (set env or use a .env file).")

    voice_id_env: Optional[str] = os.getenv("ELEVEN_VOICE_ID")
    voice_name_env: Optional[str] = os.getenv("ELEVEN_VOICE_NAME")

    client = ElevenLabs(api_key=api_key)

    def _pick_voice_id(c: ElevenLabs, voice_id: Optional[str], voice_name: Optional[str]) -> str:
        if voice_id:
            return voice_id
        if voice_name:
            try:
                voices = c.voices.get_all().voices or []
                needle = voice_name.strip().lower()
                for v in voices:
                    if needle in (v.name or "").lower():
                        return v.voice_id
            except Exception:
                # fall through to default
                pass
        return DEFAULT_VOICE_ID

    voice_id = _pick_voice_id(client, voice_id_env, voice_name_env)

    # ---------------- Request configuration (exactly as shown) ----------------
    output_format = "mp3_44100_128"
    seed = 42
    stability = 0.0            # valid v3: {0.0, 0.5, 1.0}
    similarity_boost = 0.7
    request_options = {"timeout": 120.0}

    if stability not in (0.0, 0.5, 1.0):
        raise ValueError("stability must be one of {0.0, 0.5, 1.0}")

    voice_settings = {
        "stability": stability,
        "similarity_boost": similarity_boost,
        # "style": 0.8,
        # "use_speaker_boost": True,
    }

    # ---------------- Robust retries (same behavior as the provided script) ----------------
    max_attempts = 5
    base_delay = 1.0

    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            audio = client.text_to_speech.convert(
                voice_id=voice_id,
                model_id=MODEL_ID,
                text=text,
                output_format=output_format,
                seed=seed,
                voice_settings=voice_settings,
                request_options=request_options,
            )

            # The SDK may return an iterator (stream) or raw bytes.
            if hasattr(audio, "__iter__") and not isinstance(audio, (bytes, bytearray)):
                buf = BytesIO()
                for chunk in audio:
                    if not chunk:
                        continue
                    if isinstance(chunk, memoryview):
                        chunk = chunk.tobytes()
                    buf.write(chunk)
                return buf.getvalue()
            else:
                return audio if isinstance(audio, (bytes, bytearray)) else bytes(audio)

        except ApiError as e:
            last_err = e
            code = getattr(e, "status_code", None)
            # Retry on 429 and 5xx
            if code == 429 or (isinstance(code, int) and 500 <= code < 600):
                delay = base_delay * (2 ** (attempt - 1))
                time.sleep(delay)
                continue
            # Non-retryable ApiError → raise immediately
            raise

        except JSONDecodeError as e:
            last_err = e
            delay = base_delay * (2 ** (attempt - 1))
            time.sleep(delay)
            continue

        except Exception as e:
            last_err = e
            # Retry on httpx network-ish errors; otherwise bubble up
            if HTTPX_ERRORS and isinstance(e, HTTPX_ERRORS):
                delay = base_delay * (2 ** (attempt - 1))
                time.sleep(delay)
                continue
            raise

    # If we exhausted retries, surface the last error in a clear way.
    if last_err:
        raise SystemExit(f"Text-to-speech failed after {max_attempts} attempts: {last_err}")
    raise SystemExit(f"Text-to-speech failed after {max_attempts} attempts.")
