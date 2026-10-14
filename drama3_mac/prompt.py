"""Drama 3 prompt construction (port of ``fish_omni.generate``).

Rows are ``[main_token_id, code_0, ..., code_9]``: text rows carry the token id
in column 0 and zeros elsewhere; audio rows carry ``semantic_start + code_0``
and the ten codec codes.  Every literal segment is tokenized on its own.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from .config import ModelConfig

DRAMA3_TTS_INSTRUCTION = "convert the provided text to speech"
DRAMA3_REFERENCE_INSTRUCTION = (
    "convert the provided text to speech reference to the following:"
)
DRAMA3_MESSAGE_END = "<|im_end|>\n"
DRAMA3_GENERATION_START = "<|im_start|>assistant\n<|voice|>"
DRAMA3_STRUCTURAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|voice|>",
    "<|text|>",
    "<|interleave|>",
    "<|semantic:",
)
SPEAKER_TAG_PATTERN = re.compile(r"<\|speaker:([0-9]+)\|>")
MAX_SPEAKER_ID = 15
UNICODE_WHITE_SPACE = "".join(
    map(
        chr,
        (*range(0x9, 0xE), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B))
        + (0x2028, 0x2029, 0x202F, 0x205F, 0x3000),
    )
)
MAX_PROMPT_ROWS = 8192


class PromptTooLongError(ValueError):
    pass


def validate_text(text: str, location: str) -> None:
    if not text.strip(UNICODE_WHITE_SPACE):
        raise ValueError(f"{location}: text must not be empty")
    for token in DRAMA3_STRUCTURAL_TOKENS:
        if token in text:
            raise ValueError(f"{location}: {token} is written by the prompt builder")
    for marker in re.finditer(r"<\|speaker", text):
        tag = SPEAKER_TAG_PATTERN.match(text, marker.start())
        digits = tag.group(1) if tag else ""
        if not digits or str(int(digits)) != digits or int(digits) > MAX_SPEAKER_ID:
            raise ValueError(
                f"{location}: speaker tags must be <|speaker:0|> to "
                f"<|speaker:{MAX_SPEAKER_ID}|>; got {text!r}"
            )


def with_speaker_tag(text: str) -> str:
    if SPEAKER_TAG_PATTERN.match(text.lstrip(UNICODE_WHITE_SPACE)) is None:
        return f"<|speaker:0|> {text}"
    return text


def build_prompt(
    tokenizer,
    config: ModelConfig,
    text: str,
    prompt_texts: Sequence[str] = (),
    prompt_tokens: Sequence[np.ndarray] = (),
    history: Sequence[tuple[str, np.ndarray]] = (),
) -> np.ndarray:
    """CPU int32 rows [rows, 1 + num_codebooks] ending in the open assistant turn."""
    num_codebooks = config.num_codebooks
    codebook_size = config.audio_decoder.vocab_size
    for token, expected in (
        ("<|semantic:0|>", config.semantic_start_token_id),
        (f"<|semantic:{codebook_size - 1}|>", config.semantic_end_token_id),
        ("<|im_end|>", config.eos_token_id),
    ):
        if tokenizer.token_to_id(token) != expected:
            raise ValueError(f"tokenizer/config mismatch for {token}")
    if len(prompt_texts) != len(prompt_tokens):
        raise ValueError("every reference needs both a transcript and codes")
    validate_text(text, "text")

    def check_codes(codes: np.ndarray, location: str) -> None:
        if codes.ndim != 2 or codes.shape[0] != num_codebooks or codes.shape[1] == 0:
            raise ValueError(
                f"{location}: codes must be integer [{num_codebooks}, frames >= 1]; "
                f"got {codes.shape}"
            )
        if codes.min() < 0 or codes.max() >= codebook_size:
            raise ValueError(f"{location}: codes must be in [0, {codebook_size})")

    segments: list[str | np.ndarray] = [
        "<|im_start|>system\n",
        DRAMA3_REFERENCE_INSTRUCTION if len(prompt_tokens) else DRAMA3_TTS_INSTRUCTION,
    ]
    for index, (transcript, codes) in enumerate(zip(prompt_texts, prompt_tokens), 1):
        validate_text(transcript, f"reference {index}")
        check_codes(np.asarray(codes), f"reference {index}")
        tagged = with_speaker_tag(transcript)
        body = tagged.lstrip(UNICODE_WHITE_SPACE)
        tag = SPEAKER_TAG_PATTERN.match(body)
        leading = tagged[: len(tagged) - len(body)]
        transcript = tag.group(0) + leading + body[tag.end() :]
        segments += [f"\n\n[Reference {index}]\nText: {transcript}\nSpeech: ", np.asarray(codes)]
    segments.append(DRAMA3_MESSAGE_END)
    for index, (turn_text, codes) in enumerate(history, 1):
        validate_text(turn_text, f"history {index}")
        check_codes(np.asarray(codes), f"history {index}")
        segments += [
            "<|im_start|>user\n",
            with_speaker_tag(turn_text),
            DRAMA3_MESSAGE_END,
            DRAMA3_GENERATION_START,
            np.asarray(codes),
            DRAMA3_MESSAGE_END,
        ]
    segments += [
        "<|im_start|>user\n",
        with_speaker_tag(text),
        DRAMA3_MESSAGE_END,
        DRAMA3_GENERATION_START,
    ]

    rows = []
    for segment in segments:
        if isinstance(segment, str):
            ids = np.asarray(tokenizer.encode(segment).ids, dtype=np.int64)
            text_rows = np.zeros((len(ids), 1 + num_codebooks), dtype=np.int64)
            text_rows[:, 0] = ids
            rows.append(text_rows)
        else:
            codes = segment.astype(np.int64)
            main = codes[:1] + config.semantic_start_token_id
            rows.append(np.concatenate([main, codes]).T)
    result = np.concatenate(rows)
    if len(result) > MAX_PROMPT_ROWS:
        raise PromptTooLongError(
            f"prompt has {len(result)} rows; at most {MAX_PROMPT_ROWS}"
        )
    return result.astype(np.int32)
