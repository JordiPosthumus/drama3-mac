"""Named-voice registry for the production API.

A voice is a directory ``<root>/<name>/`` holding the reference clip(s) as
WAV plus a ``voice.json`` manifest (transcript + sampling recipe).  The
directory is created on POST /v1/voices and removed on DELETE; renders
reference voices by name.  Refs live on disk and are git-ignored — no voice
data ever enters the repository.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf

NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
RECIPE_KEYS = ("temperature", "top_p", "top_k", "repetition_penalty")


class VoiceError(ValueError):
    pass


def validate_recipe(recipe: dict) -> dict:
    """Fill defaults and range-check the sampling recipe."""
    out = {
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 30,
        "repetition_penalty": 1.0,
    }
    recipe = recipe or {}
    unknown = set(recipe) - set(RECIPE_KEYS)
    if unknown:
        raise VoiceError(f"unknown recipe keys: {sorted(unknown)}")
    for key in RECIPE_KEYS:
        if key in recipe:
            value = recipe[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise VoiceError(f"recipe.{key} must be a number")
            out[key] = float(value)
    if not 0.1 <= out["temperature"] <= 2.0:
        raise VoiceError("recipe.temperature must be within [0.1, 2.0]")
    if not 0.0 < out["top_p"] <= 1.0:
        raise VoiceError("recipe.top_p must be within (0, 1]")
    if int(out["top_k"]) < 1:
        raise VoiceError("recipe.top_k must be >= 1")
    if out["repetition_penalty"] < 1.0:
        raise VoiceError("recipe.repetition_penalty must be >= 1.0")
    out["top_k"] = int(out["top_k"])
    return out


class Voice:
    """One registered voice: reference wavs + transcript + sampling recipe."""

    def __init__(self, name: str, directory: Path, manifest: dict):
        self.name = name
        self.directory = directory
        self.manifest = manifest

    @property
    def text(self) -> str:
        return self.manifest["text"]

    @property
    def recipe(self) -> dict:
        return dict(self.manifest["recipe"])

    @property
    def files(self) -> list[str]:
        return list(self.manifest["files"])

    @property
    def seconds(self) -> float:
        return float(self.manifest.get("seconds", 0.0))

    def content_hash(self) -> str:
        """Hash over the manifest — changes when the voice is recreated."""
        return self.manifest.get("sha256", self.name)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "text": self.text,
            "recipe": self.recipe,
            "references": len(self.files),
            "seconds": round(self.seconds, 2),
            "created_at": self.manifest.get("created_at"),
        }


class VoiceRegistry:
    """Creates, lists, loads and deletes voices under ``root``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- create ---------------------------------------------------------------

    def create(
        self,
        name: str,
        clips: list[tuple[np.ndarray, int]],
        text: str,
        recipe: dict,
    ) -> Voice:
        """Persist a voice from decoded reference clips (float32 mono, sr)."""
        if not NAME_PATTERN.match(name or ""):
            raise VoiceError(
                "voice name must match [a-z0-9][a-z0-9_-]{0,63}"
            )
        if self.exists(name):
            raise VoiceError(f"voice {name!r} already exists (DELETE it first)")
        if not text or not text.strip():
            raise VoiceError("a voice needs a transcript of its reference clip(s)")
        recipe = validate_recipe(recipe)
        if not clips:
            raise VoiceError("a voice needs at least one reference clip")

        directory = self.root / name
        directory.mkdir()
        import hashlib

        hasher = hashlib.sha256()
        files: list[str] = []
        total = 0.0
        try:
            for i, (wav, sr) in enumerate(clips, 1):
                if wav.ndim > 1:  # safety: callers pass mono
                    wav = wav.mean(axis=1)
                rel = f"reference-{i}.wav"
                sf.write(directory / rel, wav, sr, subtype="PCM_16")
                raw = (directory / rel).read_bytes()
                hasher.update(raw)
                files.append(rel)
                total += len(wav) / sr
            manifest = {
                "name": name,
                "text": text,
                "recipe": recipe,
                "files": files,
                "seconds": total,
                "sha256": hasher.hexdigest(),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            (directory / "voice.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return Voice(name, directory, manifest)

    def create_from_bytes(
        self, name: str, audio_b64_items: list[bytes], text: str, recipe: dict
    ) -> Voice:
        """Decode uploaded audio bytes (wav/mp3/...) and persist the voice."""
        clips: list[tuple[np.ndarray, int]] = []
        for raw in audio_b64_items:
            try:
                wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
            except Exception as exc:
                raise VoiceError(f"cannot decode reference audio: {exc}") from exc
            clips.append((wav.mean(axis=1), sr))
        for wav, sr in clips:
            if len(wav) / sr < 1.0:
                raise VoiceError(
                    "reference clip is under 1 s; at least 1 s of clean speech "
                    "is needed (5-30 s works best)"
                )
        return self.create(name, clips, text, recipe)

    # -- load / list / delete -------------------------------------------------

    def exists(self, name: str) -> bool:
        return (self.root / name / "voice.json").exists()

    def get(self, name: str) -> Voice:
        manifest_path = self.root / name / "voice.json"
        if not NAME_PATTERN.match(name or "") or not manifest_path.exists():
            raise VoiceError(f"voice {name!r} not found (GET /v1/voices to list)")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return Voice(name, manifest_path.parent, manifest)

    def list(self) -> list[Voice]:
        voices = []
        for child in sorted(self.root.iterdir()):
            if (child / "voice.json").exists():
                voices.append(self.get(child.name))
        return voices

    def delete(self, name: str) -> None:
        directory = self.root / name
        if not NAME_PATTERN.match(name or "") or not (directory / "voice.json").exists():
            raise VoiceError(f"voice {name!r} not found (GET /v1/voices to list)")
        shutil.rmtree(directory)

    # -- reference access ------------------------------------------------------

    def load_references(self, voice: Voice) -> list[tuple[np.ndarray, int]]:
        """Decode the voice's reference wavs (mono float32 + sr)."""
        clips = []
        for rel in voice.files:
            wav, sr = sf.read(voice.directory / rel, dtype="float32", always_2d=True)
            clips.append((wav.mean(axis=1), sr))
        return clips
