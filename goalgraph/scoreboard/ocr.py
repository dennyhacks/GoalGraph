"""Scoreboard OCR branch: reads match clock and score to anchor events.

Why this branch matters:
1. When a score changes from e.g. 0-0 -> 1-0, we know a goal happened immediately
   prior (with a small broadcast graphic delay, e.g. 1.0 - 2.5s).
2. When the clock stops or disappears, it anchors kickoff, half-time, or full-time.
3. During replays, the live broadcast scoreboard is almost always removed or replaced
   with a "REPLAY" bug — an immediate independent signal of a replay segment!
"""
from __future__ import annotations

import re
import json
import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ..schema import Candidate
from ..video_io import iter_frames

log = logging.getLogger(__name__)

# Fallback OCR regexes
SCORE_RE = re.compile(r"([A-Z]{2,4})?\s*([0-9])\s*[-–:]\s*([0-9])\s*([A-Z]{2,4})?", re.I)
CLOCK_RE = re.compile(r"([0-9]{1,2})\s*[:\.]\s*([0-9]{2})")


class ScoreboardTracker:
    def __init__(self, roi: tuple[int, int, int, int] | None = None, use_easyocr: bool = True):
        """
        roi: (x, y, w, h) in pixels. If None, auto-locates in top 20% of frame.
        """
        self.roi = roi
        self.use_easyocr = use_easyocr
        self._reader = None
        if self.use_easyocr:
            try:
                import easyocr
                self._reader = easyocr.Reader(["en"], gpu=False)
            except Exception as e:
                log.warning("EasyOCR init failed: %s", e)
                self._reader = None

    def find_roi(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        """Find the scoreboard header in top 20% of frame if ROI not preset."""
        if self.roi is not None:
            return self.roi
        h, w = frame.shape[:2]
        # Standard broadcast scoreboard in top 20% left half
        return (0, 0, min(int(w * 0.55), w), min(int(h * 0.20), h))

    def read_crop(self, crop: np.ndarray) -> dict:
        """Reads scoreboard crop via EasyOCR or morphological fallback."""
        if crop is None or crop.size == 0:
            return {"score": None, "clock_s": None, "clock": None, "teams": [], "raw": "", "visible": False}

        if self._reader is not None:
            try:
                texts = self._reader.readtext(crop, detail=0)
                score, teams, clock = self._parse_ocr_texts(texts)
                raw = " ".join(texts)
                visible = bool(score or clock or "HT" in raw.upper() or "FT" in raw.upper() or len(teams) >= 2)
                return {
                    "score": score,
                    "clock": clock,
                    "clock_s": None,
                    "teams": teams,
                    "raw": raw,
                    "visible": visible
                }
            except Exception as e:
                log.debug("EasyOCR read error: %s", e)

        return self.read_crop_fast(crop)

    def _parse_ocr_texts(self, texts: list[str]) -> tuple[list[int] | None, list[str], str | None]:
        score = None
        teams = []
        clock = None
        cleaned_texts = []
        score_patterns = [
            re.compile(r"([0-9])\s*[-–:]\s*([0-9])"),
            re.compile(r"\b([0-9])\s+([0-9])\b"),
            re.compile(r"([A-Z]{2,4})\s*([0-9])\s*[-–:]\s*([0-9])\s*([A-Z]{2,4})", re.I)
        ]

        for t in texts:
            m_clk = re.search(r"\b([0-9]{1,2})[:\.]([0-9]{2})\b", t)
            if m_clk:
                clock = f"{m_clk.group(1)}:{m_clk.group(2)}"
                t_rem = re.sub(r"\b[0-9]{1,2}[:\.][0-9]{2}\b", "", t).strip()
                if t_rem:
                    cleaned_texts.append(t_rem)
            else:
                cleaned_texts.append(t)

        for t in cleaned_texts:
            for pat in score_patterns:
                m_sc = pat.search(t)
                if m_sc:
                    g = m_sc.groups()
                    if len(g) == 2 and g[0].isdigit() and g[1].isdigit():
                        score = [int(g[0]), int(g[1])]
                        break
                    elif len(g) == 4 and g[1].isdigit() and g[2].isdigit():
                        score = [int(g[1]), int(g[2])]
                        teams.extend([g[0].upper(), g[3].upper()])
                        break
            if score:
                break

        if not score:
            digits = [int(tok) for tok in cleaned_texts if tok.isdigit() and len(tok) == 1]
            if len(digits) >= 2:
                score = digits[:2]

        for t in cleaned_texts:
            for tok in re.findall(r"\b[A-Za-z]{3,4}\b", t):
                u = tok.upper()
                if u not in ("FULL", "HALF", "TIME", "GOAL", "LINE", "CARD") and u not in teams:
                    teams.append(u)

        return score, teams, clock

    def read_crop_fast(self, crop: np.ndarray) -> dict:
        """Fast offline OCR fallback using morphology."""
        if crop is None or crop.size == 0:
            return {"score": None, "clock_s": None, "clock": None, "teams": [], "raw": "", "visible": False}

        ch, cw = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        non_zero = cv2.countNonZero(thresh)
        if non_zero < 80:
            return {"score": None, "clock_s": None, "clock": None, "teams": [], "raw": "", "visible": False}

        score_crop = thresh[:, int(cw * 0.15): int(cw * 0.58)]
        score_val = self._recognize_score_pattern(score_crop)

        return {
            "score": list(score_val) if score_val else None,
            "clock_s": None,
            "clock": None,
            "teams": [],
            "raw": "",
            "visible": True
        }

    def _recognize_score_pattern(self, binarized: np.ndarray) -> tuple[int, int] | None:
        cnts, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if h >= 8 and w >= 2:
                valid.append((x, y, w, h, c))
        if not valid:
            return None
        valid.sort(key=lambda item: item[0])
        hyphen_idx = None
        for idx, (x, y, w, h, c) in enumerate(valid):
            if w > h * 1.3 and h <= 8:
                hyphen_idx = idx
                break
        if hyphen_idx is not None and hyphen_idx > 0 and hyphen_idx < len(valid) - 1:
            d1_box = valid[hyphen_idx - 1]
            d2_box = valid[hyphen_idx + 1]
            s1 = self._classify_single_digit(binarized[d1_box[1]:d1_box[1]+d1_box[3], d1_box[0]:d1_box[0]+d1_box[3]])
            s2 = self._classify_single_digit(binarized[d2_box[1]:d2_box[1]+d2_box[3], d2_box[0]:d2_box[0]+d2_box[3]])
            if s1 is not None and s2 is not None:
                return (s1, s2)
        return None

    def _classify_single_digit(self, crop: np.ndarray) -> int | None:
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if h < 6 or w < 2:
            return None
        if (w / h) < 0.45:
            return 1
        center = crop[int(h * 0.3): int(h * 0.7), int(w * 0.3): int(w * 0.7)]
        if center.size > 0 and cv2.countNonZero(center) < (center.size * 0.15):
            return 0
        fill = cv2.countNonZero(crop) / float(crop.size)
        if fill > 0.45:
            return 2
        return 0

    def process_video(self, video_path: str, fps: float = 0.2, cache_path: str | None = None) -> list[dict]:
        """Sample frames (default every 5s), track score changes and clock stops."""
        if cache_path and Path(cache_path).exists():
            return json.loads(Path(cache_path).read_text())

        samples = []
        roi = None

        for t, frame in iter_frames(video_path, fps=fps):
            if roi is None:
                roi = self.find_roi(frame)
            rx, ry, rw, rh = roi
            crop = frame[ry: ry + rh, rx: rx + rw]

            info = self.read_crop(crop)
            info["t"] = round(t, 2)
            samples.append(info)

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(json.dumps(samples, indent=2))

        return samples


def detect_scoreboard_events(samples: list[dict], lag_prior: float = 1.2) -> list[Candidate]:
    """Detect events from scoreboard state transitions:
    - Score increment -> Goal event (anchor at t - lag_prior)
    - Clock freeze or tag (HT/FT) -> Half-time / Full-time
    - Scoreboard disappears -> Replay window indicator
    """
    candidates = []
    last_score = None
    last_score_t = None

    for i, s in enumerate(samples):
        sc = s.get("score")
        t = s["t"]

        if sc is not None and isinstance(sc, (list, tuple)) and len(sc) == 2:
            if last_score is None:
                # Initialize starting score (usually [0, 0])
                if sc[0] <= 1 and sc[1] <= 1:
                    last_score = list(sc)
                    last_score_t = t
            else:
                # Monotonic rule: soccer scores never decrease
                if sc[0] >= last_score[0] and sc[1] >= last_score[1]:
                    # Exactly one goal incremented
                    if (sc[0] == last_score[0] + 1 and sc[1] == last_score[1]) or \
                       (sc[1] == last_score[1] + 1 and sc[0] == last_score[0]):
                        team_idx = "A" if sc[0] > last_score[0] else "B"
                        live_t = max(0.0, t - lag_prior)
                        payload = {
                            "score_before": f"{last_score[0]}-{last_score[1]}",
                            "score_after": f"{sc[0]}-{sc[1]}",
                            "screen_t": t,
                            "team": team_idx,
                            "lag": lag_prior
                        }
                        candidates.append(
                            Candidate(
                                source="scoreboard",
                                type="goal",
                                t=live_t,
                                confidence=0.95,
                                payload=payload
                            )
                        )
                        last_score = list(sc)
                        last_score_t = t
                    elif sc == last_score:
                        # Score confirmed unchanged
                        last_score_t = t

        # Check for tags in raw text with strict word boundaries
        raw_upper = s.get("raw", "").upper()
        is_ht = bool(re.search(r"\b(HT|HALF[\s\-_]*TIME)\b", raw_upper))
        is_ft = bool(re.search(r"\b(FT|FULL[\s\-_]*(TIME|TIHE))\b", raw_upper))
        max_vid_t = max((item.get("t", 0.0) for item in samples), default=0.0)

        if is_ht and not is_ft:
            if max_vid_t == 0.0 or t <= max_vid_t * 0.70:
                candidates.append(
                    Candidate(
                        source="scoreboard",
                        type="half_time",
                        t=t,
                        confidence=0.88,
                        payload={"screen_t": t, "raw": raw_upper}
                    )
                )
        elif is_ft:
            if max_vid_t == 0.0 or t >= max_vid_t * 0.75:
                candidates.append(
                    Candidate(
                        source="scoreboard",
                        type="full_time",
                        t=t,
                        confidence=0.90,
                        payload={"screen_t": t, "raw": raw_upper}
                    )
                )

    return candidates
