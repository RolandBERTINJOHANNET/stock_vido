#!/usr/bin/env python3
"""
Step 1 — Expressivity pass (LLM-powered):

- Keeps sentences EXACTLY the same.
- For each block of up to 5 sentences, calls OpenAI (gpt-4.1-mini) to assign
  one emotion tag (from an allowed set) to each sentence.
- Prepends the tag to each sentence, preserving original whitespace/newlines.
- Returns the updated script text.

Extra helper:
- randomly_append_pause_tags(text, p=0.5, seed=None) -> str
  Adds a trailing [pause] to ~50% of sentences, inserted *before* terminal punctuation
  to keep "exactly one sentence" for ElevenLabs validation.

Env:
  OPENAI_API_KEY must be set.

Public API:
  add_emotion_tags_to_script(raw_text: str,
                             model: str = "gpt-4.1-mini",
                             batch_size: int = 5,
                             temperature: float = 0.2,
                             max_retries: int = 2) -> str

  randomly_append_pause_tags(tagged_text: str, p: float = 0.5, seed: int | None = None) -> str
"""

from __future__ import annotations
import os
import re
import time
import random
from typing import List, Tuple, Optional

# --- OpenAI client (chat.completions) ---
try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("Please `pip install openai` (>=1.0).")


# ------------------------
# Hardcoded allowed tags
# ------------------------
ALLOWED_TAGS: List[str] = [
    "[laughs]", "[laughs harder]", "[starts laughing]", "[wheezing]", "[whispers]",
    "[sighs]", "[exhales]", "[sarcastic]", "[curious]", "[excited]", "[crying]",
    "[mischievously]", "[shout]", "[tired]", "[nervous]", "[frustrated]", "[sorrowful]",
    "[calm]", "[gasps]", "[pauses]", "[hesitates]", "[stammers]", "[resigned tone]",
    "[cheerfully]", "[flatly]", "[deadpan]", "[playfully]", "[quietly]"
]
DEFAULT_TAG = "[calm]"
PAUSE_TAG = "[pause]"

# ------------------------
# Hardcoded test script
# ------------------------
TEST_SCRIPT = """Welcome to today’s short training session. We’re making Pasta Puttanesca, Sicilian style—a bright, briny classic that’s fast, flavorful, and perfect for service. By the end, you’ll know exactly what to do and why each step matters, so every plate tastes consistent and delicious. 

Step 1 — Prepare the olive oil container.
Before we heat a pan, get the olive oil ready for smooth, controlled pouring. Make two quick slits in the top of the can. This gives you easy access and better control, helping you avoid messy glugs and oily countertops. 
"""


# =========================================================
# Sentence splitting while preserving original separators
# =========================================================
_SENTENCE_RE = re.compile(r'(.+?(?:[.!?]))(\s+|$)', re.DOTALL)

def _split_sentences_with_separators(text: str) -> List[Tuple[str, str]]:
    """
    Splits `text` into [(sentence, separator_after), ...].
    - Sentences include their final punctuation.
    - Separators are the exact whitespace/newlines that followed them originally.
    - If there is any trailing text without punctuation, it becomes its own "sentence".
    """
    parts: List[Tuple[str, str]] = []
    pos = 0
    for m in _SENTENCE_RE.finditer(text):
        sent = m.group(1)
        sep = m.group(2)
        parts.append((sent, sep))
        pos = m.end()

    if pos < len(text):
        # Trailing text without terminal punctuation
        tail = text[pos:]
        parts.append((tail, ""))

    return parts


# =========================================
# OpenAI call: assign tags for N sentences
# =========================================
def _openai_assign_tags_for_block(
    client: OpenAI,
    sentences: List[str],
    allowed_tags: List[str],
    model: str = "gpt-4.1-mini",
    temperature: float = 0.2,
    max_retries: int = 2,
) -> List[str]:
    """
    Ask the model to select ONE tag per sentence from allowed_tags.
    Returns a list of tags, same length/order as `sentences`.
    """
    assert 1 <= len(sentences) <= 5, "Blocks must be 1..5 sentences."

    allowed_str = ", ".join(allowed_tags)
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))

    system_msg = (
        "You are tagging sentences for a professional kitchen training voiceover. "
        "Choose exactly ONE emotion/delivery tag per sentence from the allowed list. "
        "Don't use the same tag twice in a row."
    )

    user_msg = (
        f"Allowed tags (use EXACT brackets as written): {allowed_str}\n\n"
        f"Sentences ({len(sentences)}):\n{numbered}\n\n"
        "Return EXACTLY a comma-separated list of tags, one per sentence, same order, "
        "like this:\n[tag_a], [tag_b], [tag_c], ...\n"
        "No extra words, no numbering, no quotes."
    )

    # Robust retry loop
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = resp.choices[0].message.content or ""
            tags = _parse_llm_tags(raw, n=len(sentences), allowed=allowed_tags)
            if len(tags) == len(sentences):
                return tags
        except Exception:
            if attempt >= max_retries:
                break
            time.sleep(0.6 * (attempt + 1))  # gentle backoff

    # Fallback: default tag for all
    return [DEFAULT_TAG] * len(sentences)


_TAG_RE = re.compile(r'\[[^\]]+\]')

def _parse_llm_tags(output: str, n: int, allowed: List[str]) -> List[str]:
    """
    Extract bracketed tags from LLM output, validate against allowed list,
    and return at most n. If fewer than n valid tags are found, fill with DEFAULT_TAG.
    """
    allowed_lower = {t.lower(): t for t in allowed}
    found = _TAG_RE.findall(output)
    result: List[str] = []

    for t in found:
        key = t.strip().lower()
        if key in allowed_lower:
            result.append(allowed_lower[key])
            if len(result) == n:
                break

    while len(result) < n:
        result.append(DEFAULT_TAG)

    return result


# =========================================================
# Public API: add tags across entire script in blocks of 5
# =========================================================
def add_emotion_tags_to_script(
    raw_text: str,
    model: str = "gpt-4.1-mini",
    batch_size: int = 5,
    temperature: float = 0.2,
    max_retries: int = 2,
    openai_api_key: Optional[str] = None,
) -> str:
    """
    Main entry: assign an emotion tag to each sentence (in chunks of up to 5),
    prepend the tag, and return the updated script text.

    - Keeps sentence text exactly the same.
    - Preserves original whitespace/newlines after each sentence.
    - Uses only tags from ALLOWED_TAGS.
    """
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Please export it or pass openai_api_key=...")

    client = OpenAI(api_key=api_key)

    # Split & preserve separators
    pairs = _split_sentences_with_separators(raw_text)

    # Extract just sentences for batching
    sentences_only = [s for (s, _sep) in pairs]

    # Process in blocks of up to `batch_size` sentences
    tagged_sentences: List[str] = []
    for i in range(0, len(sentences_only), batch_size):
        block = sentences_only[i:i + batch_size]

        # Request tags for this block (1..5 sentences)
        tags = _openai_assign_tags_for_block(
            client=client,
            sentences=block,
            allowed_tags=ALLOWED_TAGS,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
        )

        # Prepend tags to sentences
        for tag, sentence in zip(tags, block):
            tagged_sentences.append(f"{tag} {sentence}")

    # Re-stitch with original separators
    out_chunks: List[str] = []
    si = 0
    for (_orig_sentence, sep) in pairs:
        out_chunks.append(tagged_sentences[si])
        out_chunks.append(sep)
        si += 1

    return "".join(out_chunks)


# =========================================================
# Extra helper: randomly append [pause] before punctuation
# =========================================================
_TERM_PUNCT_RE = re.compile(r'([.!?…]+)$')

def randomly_append_pause_tags(
    tagged_text: str,
    p: float = 0.5,
    seed: Optional[int] = None,
) -> str:
    """
    Append a PAUSE_TAG to ~p fraction of sentences, but insert it *before* the
    terminal punctuation so ElevenLabs still sees exactly one sentence.

    - Preserves original whitespace/newlines after each sentence.
    - Skips sentences that already end with [pause] (case-insensitive).
    """
    if seed is not None:
        random.seed(seed)

    pairs = _split_sentences_with_separators(tagged_text)
    out_chunks: List[str] = []

    for sentence, sep in pairs:
        s = sentence
        # If it's all whitespace, or empty, just keep it
        if not s.strip():
            out_chunks.append(s)
            out_chunks.append(sep)
            continue

        # If already ends with [pause] (allow trailing punctuation/spaces), skip
        if re.search(r'\[pause\]\s*[.!?…]*\s*$', s, flags=re.IGNORECASE):
            out_chunks.append(s)
            out_chunks.append(sep)
            continue

        if random.random() < p:
            # Insert before terminal punctuation if present, else append
            m = _TERM_PUNCT_RE.search(s)
            if m:
                start, end = m.span()
                s = f"{s[:start]} {PAUSE_TAG}{s[start:]}"
            else:
                s = f"{s} {PAUSE_TAG}"

        out_chunks.append(s)
        out_chunks.append(sep)

    return "".join(out_chunks)


# ==========
# __main__
# ==========
if __name__ == "__main__":
    print("=== Step 1: Expressivity pass demo (gpt-4.1-mini) ===")
    print("Note: Set OPENAI_API_KEY in your environment.")
    print("\n--- ORIGINAL ---\n")
    print(TEST_SCRIPT)

    try:
        tagged = add_emotion_tags_to_script(TEST_SCRIPT)
    except Exception as e:
        raise SystemExit(f"\nError during tagging: {e}\n")

    print("\n--- TAGGED ---\n")
    print(tagged)

    # Optional: demonstrate the [pause] helper
    paused = randomly_append_pause_tags(tagged, p=0.5, seed=42)
    print("\n--- TAGGED + RANDOM [pause] ---\n")
    print(paused)
