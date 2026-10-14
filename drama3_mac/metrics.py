"""Cheap audio metrics for the JSON envelope: duration, RMS, median f0.

No new dependencies — everything is numpy.  The f0 estimator decimates the
signal to ~8 kHz, then measures per-frame autocorrelation peaks in a
60–400 Hz search band (voiced frames only, energy-gated); the median over
accepted frames is the reported value, ``None`` when nothing is voiced.
MakeAPod's gate measures with librosa.yin (60–220 Hz band); the numbers
agree to a few Hz on clean speech, which is all the envelope needs.
"""

from __future__ import annotations

import numpy as np

F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0
F0_VOICED_THRESHOLD = 0.45  # normalized autocorrelation peak to accept a frame
TARGET_RATE = 8000  # decimation target for the lag search


def duration_s(audio: np.ndarray, sample_rate: int) -> float:
    return len(audio) / sample_rate if sample_rate else 0.0


def rms(audio: np.ndarray) -> float:
    if not len(audio):
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def _decimate(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Low-pass + decimate to ~``TARGET_RATE`` Hz (anti-alias via hamming FIR)."""
    factor = max(1, sample_rate // TARGET_RATE)
    if factor == 1:
        return audio, sample_rate
    taps = np.hamming(2 * factor + 1)
    taps /= taps.sum()
    filtered = np.convolve(audio, taps, mode="same")
    return filtered[::factor].copy(), sample_rate // factor


def f0_median_hz(
    audio: np.ndarray,
    sample_rate: int,
    *,
    fmin: float = F0_MIN_HZ,
    fmax: float = F0_MAX_HZ,
    max_frames: int = 200,
) -> float | None:
    """Median f0 over voiced frames, or ``None`` if nothing is voiced.

    Frames are energy-gated (above 15% of the clip's 90th-percentile frame
    energy) and up to ``max_frames`` are spread evenly across the clip.
    """
    if len(audio) < sample_rate // 10:  # <100 ms: nothing to measure
        return None
    x, sr = _decimate(np.asarray(audio, np.float32), sample_rate)
    frame = int(sr * 0.04)
    hop = frame // 2
    n_frames = 1 + (len(x) - frame) // hop
    if n_frames < 1:
        return None

    energies = np.array(
        [float(np.mean(x[i * hop : i * hop + frame] ** 2)) for i in range(n_frames)]
    )
    floor = 0.15 * np.percentile(energies, 90)
    voiced_indices = np.nonzero(energies > max(floor, 1e-8))[0]
    if not len(voiced_indices):
        return None
    if len(voiced_indices) > max_frames:
        pick = np.linspace(0, len(voiced_indices) - 1, max_frames).astype(int)
        voiced_indices = voiced_indices[pick]

    lag_min = int(sr / fmax)
    lag_max = int(sr / fmin)
    estimates: list[float] = []
    for i in voiced_indices:
        seg = x[i * hop : i * hop + frame]
        seg = seg - seg.mean()
        energy = float(np.dot(seg, seg))
        if energy <= 0:
            continue
        ac = np.correlate(seg, seg, mode="full")[frame - 1 :]
        if lag_max >= len(ac):
            continue
        window = ac[lag_min : lag_max + 1]
        best = int(np.argmax(window)) + lag_min
        if ac[best] / energy >= F0_VOICED_THRESHOLD:
            estimates.append(sr / best)
    if not estimates:
        return None
    return float(np.median(estimates))
