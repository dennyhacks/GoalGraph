"""Referee-whistle detector.

A whistle is a narrow-band tone around 2.5-4.5 kHz.  We look for STFT frames
where (a) that band dominates the spectrum and (b) the spectrum is *peaky*
(tonal, not crowd noise or sibilant speech).  Whistles give a sharp acoustic
anchor (+-0.1 s) for fouls, kick-offs and half/full time — sharper than either
commentary or the scoreboard.
"""
from __future__ import annotations

import numpy as np

from ..schema import Candidate
from ..video_io import load_wav


def detect_whistles(wav: str, sr: int = 16000, band=(2500, 4500), min_dur=0.2) -> list[dict]:
    x = load_wav(wav, sr)
    if len(x) == 0:
        return []
    n_fft, hop = 1024, 160            # 10 ms hop
    win = np.hanning(n_fft).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * win
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    bmask = (freqs >= band[0]) & (freqs <= band[1])
    band_e = spec[:, bmask]
    total = spec[:, freqs > 150].sum(1) + 1e-9
    ratio = band_e.sum(1) / total
    peak = band_e.max(1) / (band_e.mean(1) + 1e-12)       # tonality inside the band
    loud = 10 * np.log10(band_e.sum(1) + 1e-12)
    floor = np.percentile(loud, 50)
    on = (ratio > 0.45) & (peak > 12) & (loud > floor + 12)
    # smooth: close 50 ms gaps
    on = _close(on, 5)
    out = []
    i = 0
    T = len(on)
    while i < T:
        if on[i]:
            j = i
            while j < T and on[j]:
                j += 1
            t0, t1 = i * hop / sr, j * hop / sr
            if t1 - t0 >= min_dur:
                f0 = float(freqs[bmask][band_e[i:j].mean(0).argmax()])
                out.append({"start": round(t0, 3), "end": round(t1, 3), "dur": round(t1 - t0, 3),
                            "freq": round(f0, 1), "strength": float(ratio[i:j].mean())})
            i = j
        else:
            i += 1
    return out


def _close(mask: np.ndarray, k: int) -> np.ndarray:
    m = mask.copy()
    idx = np.flatnonzero(mask)
    for a, b in zip(idx, idx[1:]):
        if 1 < b - a <= k:
            m[a:b] = True
    return m


def whistle_candidates(whistles: list[dict]) -> list[Candidate]:
    """Short blasts -> stoppage (foul) or kickoff; long blasts -> half/full time.

    The type is left ambiguous ("whistle_short"/"whistle_long"); the fusion
    engine attaches whistles to nearby events rather than creating events.
    """
    out = []
    for w in whistles:
        kind = "whistle_long" if w["dur"] >= 0.9 else "whistle_short"
        out.append(Candidate("audio", kind, w["start"], min(0.95, 0.5 + w["strength"] / 2),
                             {"whistle": w, "lag": 0.0}))
    return out
