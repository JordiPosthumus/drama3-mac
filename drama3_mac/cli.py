"""Command line interface for the Drama 3 Mac port.

Usage:
    drama3 synthesize "Text to speak" -o out.wav [--ref ref.wav --ref-text "..."]
    drama3 serve --host 127.0.0.1 --port 8843
"""

from __future__ import annotations

import argparse
import sys

from .engine import DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P, Drama3Engine

DEFAULT_MODEL = "checkpoints/drama3-q4"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL, help="converted checkpoint dir")
    parser.add_argument("--codec", default=None, help="path to codec.pth (default: ../model/codec.pth)")
    parser.add_argument("--codec-device", default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--codec-dtype", default="float32", choices=["float32", "bfloat16"])


def cmd_synthesize(args: argparse.Namespace) -> int:
    engine = Drama3Engine(
        args.model,
        codec_path=args.codec,
        codec_device=args.codec_device,
        codec_dtype=args.codec_dtype,
    )
    result = engine.tts(
        args.text,
        reference_audio=args.ref,
        reference_text=args.ref_text,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        single_chunk=args.single_chunk,
        chunk_gap=args.chunk_gap,
        repetition_penalty=args.repetition_penalty,
    )
    out = engine.save(result, args.output)
    gen = result.generation
    print(
        f"wrote {out}  ({len(result.audio) / result.sample_rate:.2f}s audio, "
        f"{gen.elapsed:.2f}s generate, {len(gen.frames)} frames, "
        f"finish={gen.finish_reason}, rtf={result.real_time_factor:.2f})"
    )
    for warning in result.warnings:
        print(warning, file=sys.stderr)
    if args.verify:
        from .guardrails import verify_roundtrip

        verdict = verify_roundtrip(
            out, args.text, threshold=args.verify_threshold
        )
        print(str(verdict))
        if verdict.level == "FAIL":
            return 1
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    serve(
        model_dir=args.model,
        codec_path=args.codec,
        codec_device=args.codec_device,
        codec_dtype=args.codec_dtype,
        host=args.host,
        port=args.port,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="drama3", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    syn = sub.add_parser("synthesize", help="text (+ optional reference clip) -> wav")
    syn.add_argument("text", help="text to speak")
    syn.add_argument("-o", "--output", default="output.wav", help="output wav path")
    syn.add_argument("--ref", default=None, help="reference wav for voice cloning")
    syn.add_argument("--ref-text", default=None, help="transcript of the reference wav")
    syn.add_argument("--max-new-tokens", type=int, default=0,
                     help="frames per generation pass; 0 = auto (2048 single chunk, "
                     "600 per chunk for long text)")
    syn.add_argument("--single-chunk", action="store_true",
                     help="generate the whole text in one pass instead of chunking")
    syn.add_argument("--chunk-gap", type=float, default=0.05,
                     help="silence in seconds between chunk audios")
    syn.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    syn.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    syn.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    syn.add_argument("--repetition-penalty", type=float, default=1.0,
                     help="opt-in CTRL-style penalty on recently drawn semantic "
                          "codes (1.0 = off = Fish-faithful; Fish ignores its own "
                          "repetition_penalty field for Drama 3)")
    syn.add_argument("--verify", action="store_true",
                     help="STT-round-trip the output with mlx-whisper medium.en "
                          "and fail below a word-similarity threshold")
    syn.add_argument("--verify-threshold", type=float, default=0.8,
                     help="minimum word-F1 for --verify (default 0.8)")
    _add_common(syn)
    syn.set_defaults(func=cmd_synthesize)

    srv = sub.add_parser("serve", help="run the local TTS HTTP server")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8843)
    _add_common(srv)
    srv.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
