"""Dud-job guardrails: preflight checks that refuse bad jobs early and
postflight checks that flag suspicious renders.

Findings carry a level, mirroring a real preflight system:

- ``FAIL`` — the job cannot succeed (or the output is verified bad); the
  caller must stop.  ``engine.tts`` raises :class:`PreflightError`.
- ``WARN`` — the job will probably run, but something looks off; the finding
  is reported and generation continues.

Everything measurable is measured, nothing guessed: capacity math uses the
same row accounting as ``prompt.build_prompt`` (1 row per text token, 1 row
per audio frame) and the loop detector reads the generated main-token ids.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .prompt import MAX_PROMPT_ROWS

# Paired delivery tags the model understands (docs: README "Directing the
# performance").  Anything else in <...>...</...> form is likely a typo.
KNOWN_PAIRED_TAGS = frozenset(
    {"whisper", "emphasis", "soft", "fast", "slow", "stress", "forceful"}
)
PAIRED_TAG_PATTERN = re.compile(r"</?([a-zA-Z][a-zA-Z-]*)>")

# Loop detector: a run of `min_repeats` consecutive repetitions of a 1-4
# token pattern inside the generated semantic codes.  Speech phonology
# repeats legitimately at small scales (a plosive pair, a stressed
# syllable), so the thresholds are deliberately high.
LOOP_MAX_PATTERN = 4
LOOP_MIN_REPEATS = 6

# Reference-clip limits.  ~21.5 frames/s means 30 s ≈ 645 rows of the 8192-row
# prompt budget.
REFERENCE_MIN_SECONDS = 1.0
REFERENCE_WARN_SECONDS = 30.0

VERIFY_DEFAULT_THRESHOLD = 0.8
VERIFY_MODEL = "mlx-community/whisper-medium.en-mlx"


class PreflightError(ValueError):
    """A FAIL-level preflight finding; the job was refused before generation."""


@dataclass
class Finding:
    level: str  # "FAIL" | "WARN"
    check: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.check}: {self.message}"


def _fail(check: str, message: str) -> Finding:
    return Finding("FAIL", check, message)


def _warn(check: str, message: str) -> Finding:
    return Finding("WARN", check, message)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def check_checkpoint(model_dir: str | Path, codec_path: str | Path | None) -> list[Finding]:
    """FAIL if the checkpoint directory is missing files the engine needs."""
    findings: list[Finding] = []
    d = Path(model_dir)
    if not d.is_dir():
        findings.append(_fail(
            "checkpoint",
            f"{d} does not exist; convert a checkpoint with "
            f"`python -m drama3_mac.convert --checkpoint model --out {d}` "
            "(see README Installation)",
        ))
        return findings
    shards = sorted(d.glob("*.safetensors"))
    if not shards:
        findings.append(_fail(
            "checkpoint",
            f"{d} contains no .safetensors shards",
        ))
    if not (d / "config.json").exists() and not list(d.glob("*.json")):
        findings.append(_fail("checkpoint", f"{d} has no config json"))
    codec = Path(codec_path) if codec_path else d / "codec.pth"
    if not codec.exists():
        findings.append(_fail(
            "checkpoint",
            f"codec weights not found at {codec}; download the original model "
            "(huggingface-cli download fishaudio/drama-3 --local-dir model) "
            "or pass --codec /path/to/codec.pth",
        ))
    return findings


def check_tag_balance(text: str) -> list[Finding]:
    """FAIL on unbalanced paired tags, WARN on unknown tag names.

    An unclosed ``<whisper>`` makes the whole render whisper; a stray ``</fast>``
    is noise.  Square-bracket cues (``[pause]``) are free-form and not checked.
    """
    findings: list[Finding] = []
    stack: list[tuple[str, int]] = []
    for match in PAIRED_TAG_PATTERN.finditer(text):
        name, closing = match.group(1).lower(), match.group(0).startswith("</")
        if closing:
            if stack and stack[-1][0] == name:
                stack.pop()
            else:
                findings.append(_fail(
                    "tag-balance",
                    f"stray </{name}> at offset {match.start()} with no matching opener",
                ))
        else:
            stack.append((name, match.start()))
            if name not in KNOWN_PAIRED_TAGS:
                findings.append(_warn(
                    "tag-balance",
                    f"<{name}> is not a known delivery tag "
                    f"({', '.join(sorted(KNOWN_PAIRED_TAGS))}); it will be read "
                    "as literal text",
                ))
    for name, pos in stack:
        findings.append(_fail(
            "tag-balance",
            f"<{name}> opened at offset {pos} is never closed; the delivery "
            "would leak into the rest of the render",
        ))
    return findings


def check_capacity(
    text: str,
    ref_frames: int,
    *,
    utf8_len,
    max_new_tokens: int,
    history_rows: int = 0,
    max_prompt_rows: int = MAX_PROMPT_ROWS,
) -> list[Finding]:
    """FAIL if reference + text rows + generation cannot fit the prompt budget.

    Row accounting matches the prompt builder: 1 row per text token (bounded
    above by UTF-8 bytes) and 1 row per audio frame (~21.5 frames/s).
    ``history_rows`` covers fed-back chunk audio on the chunked path.
    """
    findings: list[Finding] = []
    text_rows_upper = utf8_len(text) + 16  # + template rows
    projected = ref_frames + text_rows_upper + history_rows
    if projected + max(1, max_new_tokens) > max_prompt_rows:
        findings.append(_fail(
            "capacity",
            f"projected prompt ({projected} rows: reference {ref_frames} + text "
            f"~{text_rows_upper} + history {history_rows}) plus a generation "
            f"budget of {max_new_tokens} exceeds the {max_prompt_rows}-row "
            "prompt ceiling — this overflows decode positions and crashes. "
            "Shorten the reference (5-10 s is plenty) or the text, or let "
            "auto-chunking split it (do not force --single-chunk)",
        ))
    elif projected + max(1, max_new_tokens) > max_prompt_rows * 0.9:
        findings.append(_warn(
            "capacity",
            f"projected prompt uses {projected + max(1, max_new_tokens)} of "
            f"{max_prompt_rows} rows (>90%); little headroom for the generation",
        ))
    if ref_frames > REFERENCE_WARN_SECONDS * 21.5:
        findings.append(_warn(
            "capacity",
            f"reference is ~{ref_frames / 21.5:.0f} s ({ref_frames} rows); "
            "adherence gains flatten after ~30 s and long references eat the "
            "prompt budget",
        ))
    return findings


def check_reference(audio: np.ndarray, sample_rate: int) -> list[Finding]:
    """FAIL on unreadable/too-short reference clips, WARN on very long ones."""
    seconds = len(audio) / sample_rate if sample_rate else 0.0
    if seconds < REFERENCE_MIN_SECONDS:
        return [_fail(
            "reference",
            f"reference clip is {seconds:.2f} s; at least "
            f"{REFERENCE_MIN_SECONDS:.0f} s of clean speech is needed for cloning",
        )]
    if not np.isfinite(audio).all():
        return [_fail("reference", "reference audio contains NaN/inf samples")]
    findings: list[Finding] = []
    if seconds > REFERENCE_WARN_SECONDS:
        findings.append(_warn(
            "reference",
            f"reference clip is {seconds:.0f} s; 5-30 s works best, and every "
            "extra second costs prompt rows (see the capacity check)",
        ))
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak < 0.01:
        findings.append(_warn(
            "reference",
            f"reference peak amplitude is {peak:.4f} — the clip is near-silence",
        ))
    return findings


def run_preflight(
    *,
    model_dir: str | Path,
    codec_path: str | Path | None,
    text: str,
    reference_audio: np.ndarray | None,
    reference_sample_rate: int,
    ref_frames: int,
    utf8_len,
    max_new_tokens: int,
    history_rows: int = 0,
) -> list[Finding]:
    """Every pre-generation check; raises PreflightError if any FAIL."""
    findings: list[Finding] = []
    findings += check_checkpoint(model_dir, codec_path)
    findings += check_tag_balance(text)
    if reference_audio is not None:
        findings += check_reference(reference_audio, reference_sample_rate)
    findings += check_capacity(
        text, ref_frames, utf8_len=utf8_len,
        max_new_tokens=max_new_tokens, history_rows=history_rows,
    )
    failures = [f for f in findings if f.level == "FAIL"]
    if failures:
        raise PreflightError("; ".join(str(f) for f in failures))
    return findings


# ---------------------------------------------------------------------------
# Postflight
# ---------------------------------------------------------------------------


def check_finish_discipline(
    finish_reasons: list[str], frames_per_chunk: list[int]
) -> list[Finding]:
    """WARN when chunks end suspiciously (the model talked over its budget)."""
    findings: list[Finding] = []
    for i, (reason, frames) in enumerate(zip(finish_reasons, frames_per_chunk)):
        seconds = frames / 21.5
        if reason == "length" and frames < 600 * 0.25:
            findings.append(_warn(
                "finish-discipline",
                f"chunk {i} hit the token budget after only {seconds:.1f}s "
                "of audio; it was cut off mid-sentence",
            ))
        if reason == "eos" and seconds < 0.5:
            findings.append(_warn(
                "finish-discipline",
                f"chunk {i} emitted EOS after {seconds:.1f}s — possible "
                "immediate-EOS dud; re-render with a higher temperature",
            ))
    return findings


def check_loops(chunk_main: list[np.ndarray]) -> list[Finding]:
    """WARN on periodic runs in the generated semantic codes (the 1,2,1,2 loop).

    Scans each chunk's main-token sequence for any pattern of length 1-4 that
    repeats ``LOOP_MIN_REPEATS``+ times back to back.
    """
    findings: list[Finding] = []
    for i, seq in enumerate(chunk_main):
        n = len(seq)
        for pattern_len in range(1, LOOP_MAX_PATTERN + 1):
            limit = n - pattern_len * LOOP_MIN_REPEATS
            start = 0
            while start <= limit:
                pattern = seq[start : start + pattern_len]
                repeats = 1
                while (
                    start + (repeats + 1) * pattern_len <= n
                    and np.array_equal(
                        seq[start + repeats * pattern_len : start + (repeats + 1) * pattern_len],
                        pattern,
                    )
                ):
                    repeats += 1
                if repeats >= LOOP_MIN_REPEATS:
                    findings.append(_warn(
                        "loop-detector",
                        f"chunk {i}: pattern {pattern.tolist()} repeats "
                        f"{repeats}× starting at frame {start} — the sampler "
                        "likely looped; try a lower temperature or re-render",
                    ))
                    start += repeats * pattern_len
                else:
                    start += 1
    return findings


def run_postflight(
    finish_reasons: list[str],
    frames_per_chunk: list[int],
    chunk_main: list[np.ndarray],
) -> list[Finding]:
    findings = check_finish_discipline(finish_reasons, frames_per_chunk)
    findings += check_loops(chunk_main)
    return findings


# ---------------------------------------------------------------------------
# STT round-trip verification (--verify)
# ---------------------------------------------------------------------------


def normalize_words(text: str) -> list[str]:
    """Lowercase word tokens; punctuation and casing are ignored."""
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def word_f1(reference: str, hypothesis: str) -> float:
    """Token F1 between the prompt and the transcript (acceptance.py metric)."""
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    if not ref or not hyp:
        return 0.0
    from collections import Counter

    common = sum((Counter(ref) & Counter(hyp)).values())
    if common == 0:
        return 0.0
    precision = common / len(hyp)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)


def verify_roundtrip(
    wav_path: str | Path,
    prompt_text: str,
    *,
    threshold: float = VERIFY_DEFAULT_THRESHOLD,
    model: str = VERIFY_MODEL,
) -> Finding:
    """Transcribe the render with mlx-whisper and compare to the prompt.

    Returns a FAIL finding below ``threshold`` word-F1, a PASS finding above.
    Raises ImportError with setup instructions if mlx-whisper is missing.
    """
    try:
        import mlx_whisper
    except ImportError as exc:
        raise ImportError(
            "the --verify flag needs mlx-whisper: uv pip install mlx-whisper"
        ) from exc
    result = mlx_whisper.transcribe(str(wav_path), path_or_hf_repo=model)
    transcript = " ".join(seg.get("text", "") for seg in result.get("segments", []))
    f1 = word_f1(prompt_text, transcript)
    if f1 < threshold:
        return _fail(
            "roundtrip-verify",
            f"word-F1 {f1:.2f} < {threshold:.2f}; Whisper heard: {transcript[:200]!r}",
        )
    return Finding(
        "PASS", "roundtrip-verify", f"word-F1 {f1:.2f} ≥ {threshold:.2f}"
    )
