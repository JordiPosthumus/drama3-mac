"""Tests for long-text chunking: splitter semantics + reference parity.

The reference implementation lives in research/fish-speech (untracked); when
present, the splitter must produce byte-identical chunks to it.  The history
candidates and the protected-span logic are tested directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from drama3_mac.chunking import (
    CHUNK_BYTES,
    Drama3History,
    split_by_length,
    split_drama3_text,
)

REFERENCE = Path(__file__).parent.parent / "research" / "fish-speech"
if REFERENCE.exists():
    sys.path.insert(0, str(REFERENCE))
    from fish_speech.models.text2semantic.fish_omni.chunking import (  # noqa: E402
        split_drama3_text as ref_split,
    )
else:  # pragma: no cover - reference checkout is optional
    ref_split = None

LONG_TEXT = (
    "Every reinsurance horror story starts the same way. With a spreadsheet "
    "that looked fine. It was a Friday. Quarter close. The hurricane had a "
    "name, and the name was Kevin. We priced for hail! Hail! Anyway, "
    "renewals are in March. Nobody will ever know. "
) * 4


def test_short_text_single_chunk():
    assert split_by_length("Hello there.") == ["Hello there."]


def test_long_text_multiple_chunks_within_budget():
    chunks = split_by_length(LONG_TEXT)
    assert len(chunks) > 1
    assert all(len(c.encode()) <= CHUNK_BYTES + 40 for c in chunks)  # + tag slack
    joined = " ".join(chunks)
    for word in ("Kevin", "hail", "renewals"):
        assert word in joined


def test_sentence_boundaries_preferred():
    chunks = split_by_length("One sentence here. Another sentence there. " * 20)
    for chunk in chunks:
        # every chunk ends at a sentence boundary (modulo the final merge)
        assert chunk.rstrip()[-1] in ".!" or chunk is chunks[-1]


def test_protected_spans_not_split():
    text = (
        "[calm, measured voice] This is a direction cue that must stay intact. "
        "<whisper>And this paired tag also stays together.</whisper> "
        "More prose follows here to push past the chunk size limit easily. "
    ) * 6
    for chunk in split_drama3_text(text):
        assert chunk.count("[") == chunk.count("]")
        assert chunk.count("<whisper>") == chunk.count("</whisper>")


def test_speaker_segments_regroup():
    text = (
        "<|speaker:1|> First speaker says a line of reasonable length here. "
        "<|speaker:2|> Second speaker answers with another line of text. "
        "<|speaker:1|> First speaker returns for the final word. "
    ) * 3
    chunks = split_drama3_text(text)
    assert len(chunks) >= 2
    for chunk in chunks:
        speakers = set()
        for part in chunk.split("\n"):
            if part.startswith("<|speaker:"):
                speakers.add(part.split("|>")[0])
        assert len(speakers) <= 3


def test_reference_splitter_parity():
    if ref_split is None:
        pytest.skip("research/fish-speech not checked out")
    texts = [
        LONG_TEXT,
        "Short one.",
        "[calm] A direction. <soft>soft words</soft> more prose. " * 12,
        "<|speaker:2|> Alpha line. <|speaker:3|> Beta line. " * 8,
        "Decimals stay whole: 3.14 and 2.718 plus more text to wrap. " * 10,
    ]
    for text in texts:
        assert split_drama3_text(text) == ref_split(text)


def test_history_candidates_order():
    history = Drama3History(recent_turns=2)
    codes = np.zeros((10, 4), dtype=np.int64)
    history.push("first", codes)
    history.push("second", codes)
    history.push("third", codes)
    candidates = list(history.candidates())
    # longest first: pinned + all recent, then fewer recents, then empty
    assert [t for t, _ in candidates[0]] == ["first", "second", "third"]
    assert [t for t, _ in candidates[1]] == ["first", "third"]
    assert [t for t, _ in candidates[2]] == ["first"]
    assert candidates[-1] == []
    # default keeps only the most recent turn besides the pinned one
    default = Drama3History()
    default.push("a", codes)
    default.push("b", codes)
    default.push("c", codes)
    assert [t for t, _ in list(default.candidates())[0]] == ["a", "c"]


def test_speaker_regroup_chunks_are_strings_and_validate_clean():
    """Regression: speaker regrouping must yield strings, and validate_chunks
    must accept the engine's chunk list (blocker 2026-10-15: a list-wrapped
    chunk list from the engine crashed validate_text with .strip)."""
    from drama3_mac.chunking import validate_chunks

    text = (
        "<|speaker:0|> The keeper rose before dawn, as he had for thirty years.\n"
        "<|speaker:1|> The sea had not yet decided between grey and gold.\n"
        "<|speaker:0|> He logged the wind and watched the boats slip out."
    )
    for shape in (text, text * 4):  # single-chunk and multi-chunk shapes
        chunks = split_drama3_text(shape)
        assert chunks and all(isinstance(c, str) for c in chunks)
        validate_chunks(chunks)  # must not raise


def test_validate_chunks_normalizes_group_shaped_chunks():
    """Defence in depth: a producer handing back a list of segments per chunk
    is joined in place instead of crashing on .strip."""
    from drama3_mac.chunking import validate_chunks

    chunks = ["<|speaker:0|> plain text here", ["<|speaker:1|> part one", "more"]]
    validate_chunks(chunks)
    assert chunks[1] == "<|speaker:1|> part one\nmore"
