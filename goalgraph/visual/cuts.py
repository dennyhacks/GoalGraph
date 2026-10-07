"""Camera cut / scene transition detection using histogram difference + PySceneDetect."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ..video_io import iter_frames

log = logging.getLogger(__name__)


def detect_cuts(video_path: str, threshold: float = 0.35, min_scene_len: float = 0.5) -> list[dict]:
    """Detect shot transitions / camera cuts in football broadcast.
    Returns list of cut points: [{"t": float, "frame_idx": int, "score": float}]
    """
    cuts = []
    prev_hist = None
    prev_t = 0.0

    # Fast histogram difference across sampled frames (e.g. 10 fps)
    for t, frame in iter_frames(video_path, fps=12.0):
        # Convert to HSV and compute 2D hue-saturation histogram
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        if prev_hist is not None:
            # Correlation / Chi-square difference
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > threshold and (t - prev_t) >= min_scene_len:
                cuts.append({"t": round(t, 2), "score": round(float(diff), 3)})
                prev_t = t
        else:
            prev_t = t
        prev_hist = hist

    return cuts
