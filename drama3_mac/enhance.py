"""LLM script augmenter (`POST /v1/enhance`): proposes delivery tags.

Never touches MLX: one HTTP call to an OpenAI-compatible endpoint (LM Studio
qwen3.8-27b by default; GLM-5.3 quality override via DRAMA3_ENHANCE_* env),
then regex validators. The author's words are the source of truth — the
augmenter may only add direction. Any line that fails validation falls back
to the original text with a warning.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter

from .guardrails import KNOWN_PAIRED_TAGS

ENHANCE_BASE_URL = os.environ.get("DRAMA3_ENHANCE_BASE_URL", "http://127.0.0.1:30000/v1")
ENHANCE_MODEL = os.environ.get("DRAMA3_ENHANCE_MODEL", "PoolModel")
ENHANCE_API_KEY = os.environ.get("DRAMA3_ENHANCE_API_KEY", "sk-nothing")
ENHANCE_TIMEOUT_S = float(os.environ.get("DRAMA3_ENHANCE_TIMEOUT", "600"))
# 0 (or negative) = no timeout at all: the LLM thinks as long as it needs.

_SPECIAL_TOKEN = re.compile(r"<\|[^|]*\|>")
_PAIRED_TAG = re.compile(r"</?([a-zA-Z][a-zA-Z-]*)>")
_BRACKETED = re.compile(r"\[([^\[\]]*)\]")
_MAX_PAIRED_TAGS = 2
_MAX_DIRECTION_CHARS = 80
_WORD = re.compile(r"[a-z0-9']+")
_OPEN_TAG = re.compile(r"<([a-zA-Z][a-zA-Z-]*)>")
# Line-leading cues with a sound-event stem explode into multi-second silent
# gaps (measured: [sighs] at line start → 8 s of silence). Briefs like [warm]
# are safe openers; these are not.
_SOUND_EVENT_STEM = re.compile(
    r"sigh|laugh|chuckl|gasp|breath|exhal|inhal|cough|sniff|gulp|throat|hum|"
    r"whisper|shout|scream|mutter|groan|moan|yawn|snort|hiccup|sneeze|kiss|slam",
    re.I,
)

# Measured pause ladder (whisper word-timestamps, seed-locked, q4 2026-10-15):
# ellipsis ≈ 520-560 ms — the ONLY reliable pause marker in dialogue renders.
# [break] is bimodal (renders 0 ms or explodes to 4.9-22.8 s); [long-break]
# likewise; bare comma = no pause at all. Both bracket breaks are normalized
# to "..." by the validator.
DIRECTOR_PROMPT = """\
You are a dialogue director for a text-to-speech engine. For each line, add \
delivery direction ONLY. Allowed: bracketed cues like [warm], [leaning in], \
[a dry chuckle] and the paired tags <whisper> <emphasis> <soft> <fast> <slow> \
<stress> <forceful>. Rules:
- Small sounds belong mid-sentence, never at the start of a line.
- NEVER use "grinning"/"grinned"/"smirking" in briefs: measured to combine
  with <emphasis> into manic laughing. Say "delighted", "warm", "buzzing"
  or "smiling" instead.
- At most 2 paired tags per line.
- Briefs are physical/behavioral, at most 10 words, at most one per line.
- Paired tags wrap SHORT phrases only (a few words), never whole sentences,
  and never two long tags in one line: whole-sentence <soft>/<slow> makes the
  voice lose its identity (measured).
- Lines shorter than 8 words: add at most ONE brief and NO paired tags — the
  author's own punctuation carries the delivery there (tags on tiny lines
  have twice produced noise fills or dead air, measured).
- Put "..." where a natural reader would beat, especially before a payoff
  phrase: "When your whole life is a routine... that matters." Nobody reads
  "routine that matters" straight through — the pause is part of the words.
- Never put a paired tag on the FIRST word of a line: opening on a pitch spike
  destabilizes the whole take (measured swing 176 Hz). Start plain and place
  tags from the second phrase onward.
- NEVER change, add, or drop words. NEVER use ALL CAPS. NEVER add speaker \
markers like <|speaker:0|>. NEVER invent new tag types.
Answer with a strict JSON array, one object per input line, same order:
[{"line_id": <as given>, "text": "<the directed line>", "rationale": "<= 12 words"}]\
"""


class EnhanceError(Exception):
    """LLM unreachable / bad reply — the caller returns 503 with original lines."""


def _plain_words(text: str) -> list[str]:
    """Words left after stripping tags and bracketed cues."""
    stripped = _BRACKETED.sub(" ", text)
    stripped = _PAIRED_TAG.sub(" ", stripped)
    return _WORD.findall(stripped.lower())


def sanitize_line(original: str, augmented: str) -> tuple[str, list[str]]:
    """Apply the safety layers to an LLM-proposed line.

    Returns (text, warnings). Never raises: the last-resort fallback is the
    original line untouched.
    """
    warnings: list[str] = []
    text = augmented.strip()
    if not text:
        return original, ["[WARN] enhance: empty proposal; kept original"]

    # 1. special tokens are never user-facing
    if _SPECIAL_TOKEN.search(text):
        text = _SPECIAL_TOKEN.sub("", text)
        warnings.append("[WARN] enhance: stripped <|special|> token")

    # 1b. [break]/[long-break] render unreliably in dialogue (vanish or
    # explode); the ellipsis is the only pause that renders true every time
    if "[long-break]" in text or "[break]" in text:
        text = re.sub(r"\[(long-)?break\]", "...", text)
        warnings.append("[WARN] enhance: [break] normalized to ellipsis")

    # 2. unknown paired tags are likely typos: keep inner text, drop the tag
    def _drop_unknown(m: re.Match) -> str:
        return "" if m.group(1).lower() not in KNOWN_PAIRED_TAGS else m.group(0)

    unknown = [
        t for t in _PAIRED_TAG.findall(text) if t.lower() not in KNOWN_PAIRED_TAGS
    ]
    if unknown:
        text = _PAIRED_TAG.sub(_drop_unknown, text)
        warnings.append(f"[WARN] enhance: dropped unknown tag(s) {unknown}")

    # 3. bracketed cues: line-start sound events, length, balance
    leading = re.match(r"\s*(\[[^\]]*\])", text)
    if leading and _SOUND_EVENT_STEM.search(leading.group(1)):
        text = text[leading.end() :].lstrip()
        warnings.append("[WARN] enhance: removed line-start sound event")
    for cue in _BRACKETED.findall(text):
        if len(cue.strip()) > _MAX_DIRECTION_CHARS:
            text = _BRACKETED.sub(" ", text, count=1)
            warnings.append(
                f"[WARN] enhance: direction over {_MAX_DIRECTION_CHARS} chars removed"
            )
    opens, closes = text.count("["), text.count("]")
    if opens != closes:
        text = re.sub(r"[\[\]]", "", text)
        warnings.append("[WARN] enhance: unbalanced brackets removed")

    # 4. paired-tag budget: count opening tags (one per pair)
    known_used = [t for t in _OPEN_TAG.findall(text) if t.lower() in KNOWN_PAIRED_TAGS]
    if len(known_used) > _MAX_PAIRED_TAGS:
        text = _PAIRED_TAG.sub("", text)
        warnings.append(
            f"[WARN] enhance: {_MAX_PAIRED_TAGS}-tag budget exceeded; tags stripped"
        )

    # 5. typography: shouting is a loop risk, not an emotion
    def _downcase(m: re.Match) -> str:
        return m.group(0).lower()

    if re.search(r"\b[A-Z]{3,}\b", text):
        text = re.sub(r"\b[A-Z]{3,}\b", _downcase, text)
        warnings.append("[WARN] enhance: downcased ALL-CAPS")

    # 7. word preservation: the author's words are the contract
    text = re.sub(r"\s+", " ", text).strip()
    if Counter(_plain_words(original)) != Counter(_plain_words(text)):
        return original, warnings + [
            "[WARN] enhance: word content changed; kept original line"
        ]

    return text, warnings


def _extract_json_array(reply: str) -> list[dict]:
    start, end = reply.find("["), reply.rfind("]")
    if start == -1 or end <= start:
        raise EnhanceError("LLM reply contained no JSON array")
    try:
        parsed = json.loads(reply[start : end + 1])
    except json.JSONDecodeError as exc:
        raise EnhanceError(f"bad JSON in reply: {exc}") from exc
    if not isinstance(parsed, list) or not all(isinstance(o, dict) for o in parsed):
        raise EnhanceError("LLM reply was not an array of objects")
    return parsed


def _chat(messages: list[dict], base_url: str, model: str, api_key: str,
          timeout: float) -> str:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        # OWNER RULE (2026-10-15): NEVER cap the LLM's output. No max_tokens,
        # no truncation — these are the owner's own local models; they think
        # for exactly as long as they need. An earlier 2048 cap burned the
        # whole budget on reasoning and returned null content.
    }).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise EnhanceError(f"enhance LLM unavailable: {exc}") from exc
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise EnhanceError(f"malformed LLM reply: {exc}") from exc
    if not content or not content.strip():
        # reasoning models can spend the entire token budget thinking
        raise EnhanceError("LLM returned empty content (reasoning exhausted the budget?)")
    return content


def enhance_lines(lines: list[dict], *, tone_hints: str | None = None,
                  base_url: str | None = None, model: str | None = None,
                  api_key: str | None = None, timeout: float | None = None) -> list[dict]:
    """Propose directed lines; validates everything; never drops a line."""
    base_url = base_url or ENHANCE_BASE_URL
    model = model or ENHANCE_MODEL
    api_key = api_key or ENHANCE_API_KEY
    if timeout is None:
        timeout = ENHANCE_TIMEOUT_S
    if timeout <= 0:
        timeout = None  # owner rule: unlimited think time when explicitly set
    if not lines or not all(isinstance(l, dict) and str(l.get("text", "")).strip()
                            for l in lines):
        raise ValueError("[FAIL] enhance: 'lines' must be non-empty {text, ...} objects")
    if len(lines) > 64:
        raise ValueError("[FAIL] enhance: at most 64 lines per request (KV growth)")

    user = json.dumps({
        "tone_hints": tone_hints or "",
        "lines": [{"line_id": l.get("line_id", i), "text": l["text"]}
                  for i, l in enumerate(lines)],
    }, ensure_ascii=False)
    reply = _chat(
        [{"role": "system", "content": DIRECTOR_PROMPT},
         {"role": "user", "content": user}],
        base_url, model, api_key, timeout,
    )
    proposals = _extract_json_array(reply)
    by_id = {str(p.get("line_id")): p for p in proposals}

    results = []
    for i, line in enumerate(lines):
        line_id = line.get("line_id", i)
        proposal = by_id.get(str(line_id))
        if proposal is None or not str(proposal.get("text", "")).strip():
            results.append({"line_id": line_id, "text": line["text"],
                            "warnings": ["[WARN] enhance: no proposal; kept original"]})
            continue
        text, warnings = sanitize_line(line["text"], str(proposal["text"]))
        results.append({
            "line_id": line_id,
            "text": text,
            "rationale": str(proposal.get("rationale", ""))[:200],
            "warnings": warnings,
        })
    return results
