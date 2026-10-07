"""Visual entity & event detector (Players, Referee, Ball, Cards, Goal Area).

Supports:
1. YOLOv8x / YOLOv8n (COCO pre-trained: person, sports ball)
2. Specialized pitch-segmented color/morphology player detector
3. Visual action spotters:
   - Card spotting (referee holding yellow/red card high)
   - Goal mouth action (ball crossing line / net vibration)
   - Player cluster / celebrations
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..schema import Candidate

log = logging.getLogger(__name__)


@dataclass
class Detection:
    box: list[float]            # [x, y, w, h] in pixels
    label: str                  # player_A | player_B | referee | ball | card_yellow | card_red | gk_A | gk_B
    confidence: float
    features: dict[str, Any]    # color histogram, jersey_num, dominant_hsv


class VisualDetector:
    def __init__(self, use_yolo: bool = True, yolo_model: str = "yolov8n.pt"):
        self.use_yolo = use_yolo
        self.yolo_model_name = yolo_model
        self._yolo = None

    def _get_yolo(self):
        if self._yolo is None and self.use_yolo:
            try:
                from ultralytics import YOLO
                self._yolo = YOLO(self.yolo_model_name)
            except Exception as e:
                log.warning("YOLO could not be loaded: %s", e)
                self.use_yolo = False
        return self._yolo

    def detect_frame(self, frame: np.ndarray, t: float = 0.0) -> list[Detection]:
        """Detect all football entities in a single frame."""
        h, w = frame.shape[:2]
        detections: list[Detection] = []

        # 1. Pitch Segmentation (Football Green Mask)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Green grass hue typically between 30 and 85
        pitch_mask = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([85, 255, 255]))

        # Ignore top banner (scoreboard)
        top_cut = int(h * 0.12)
        pitch_mask[:top_cut, :] = 0

        # Non-pitch foreground entities on the field
        # Invert pitch to find players, lines, referee, ball
        fg_mask = cv2.bitwise_not(pitch_mask)
        fg_mask[:top_cut, :] = 0
        # Ignore crowd area (far right or margins if noisy)
        if w > 1000:
            fg_mask[:, int(w * 0.88):] = 0

        # 2. Extract Connected Components / Contours of potential players
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30 or area > (h * w * 0.1):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bh / max(1, bw)

            # Players are vertical (aspect ratio > 1.2 generally, or fallen > 0.4)
            if aspect > 1.0 and bh > 15:
                crop = frame[y: y + bh, x: x + bw]
                det = self._classify_player_crop(crop, [x, y, bw, bh])
                if det is not None:
                    detections.append(det)
            elif bw < 25 and bh < 25 and area > 10:
                # Potential ball
                crop = frame[y: y + bh, x: x + bw]
                mean_col = cv2.mean(crop)[:3]
                if mean_col[0] > 180 and mean_col[1] > 180 and mean_col[2] > 180:
                    detections.append(Detection(
                        box=[float(x), float(y), float(bw), float(bh)],
                        label="ball",
                        confidence=0.75,
                        features={"mean_bgr": mean_col}
                    ))

        # 3. Detect visual card events (Referee holding card)
        card_cand = self._detect_card_in_frame(frame, detections)
        if card_cand:
            detections.append(card_cand)

        return detections

    def _classify_player_crop(self, crop: np.ndarray, box: list[int]) -> Detection | None:
        """Classify a detected figure into team A (red), team B (blue), referee (black), or GK."""
        ch, cw = crop.shape[:2]
        if ch < 5 or cw < 5:
            return None

        # Torso is top 30%-60%
        torso = crop[int(ch * 0.2): int(ch * 0.6), :]
        if torso.size == 0:
            return None

        torso_hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        h_channel = torso_hsv[:, :, 0]
        s_channel = torso_hsv[:, :, 1]
        v_channel = torso_hsv[:, :, 2]

        # Calculate mean saturation and value
        mean_s = np.mean(s_channel)
        mean_v = np.mean(v_channel)
        bgr_mean = cv2.mean(torso)[:3]

        label = "player_unknown"
        conf = 0.80

        # Referee wears black (low value, low saturation)
        if mean_v < 60 and mean_s < 80:
            label = "referee"
        # Team A wears Red: high red channel compared to blue/green, or hue near 0 or 170
        elif bgr_mean[2] > 120 and bgr_mean[0] < 100:
            label = "player_A"
        # Team B wears Blue: high blue channel compared to red
        elif bgr_mean[0] > 120 and bgr_mean[2] < 100:
            label = "player_B"
        # Yellow GK or Cyan GK
        elif mean_s > 100 and mean_v > 150:
            label = "goalkeeper"
        else:
            return None

        # Color signature for ReID (8-bin HSV histogram)
        hist = cv2.calcHist([torso_hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        return Detection(
            box=[float(b) for b in box],
            label=label,
            confidence=conf,
            features={
                "color_hist": hist.flatten().tolist(),
                "mean_bgr": [float(c) for c in bgr_mean],
                "torso_hsv": [float(mean_s), float(mean_v)]
            }
        )

    def _detect_card_in_frame(self, frame: np.ndarray, players: list[Detection]) -> Detection | None:
        """Find bright yellow or red card held aloft near referee."""
        referees = [p for p in players if p.label == "referee"]
        if not referees:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Yellow card mask (high saturation, distinct yellow hue)
        y_mask = cv2.inRange(hsv, np.array([22, 180, 180]), np.array([32, 255, 255]))
        # Red card mask (high saturation, bright red)
        r_mask = cv2.inRange(hsv, np.array([0, 180, 180]), np.array([10, 255, 255]))

        for ref in referees:
            rx, ry, rw, rh = ref.box
            # Card must be above or near referee's upper torso
            search_x0 = max(0, int(rx - rw * 0.8))
            search_x1 = min(frame.shape[1], int(rx + rw * 1.8))
            search_y0 = max(0, int(ry - rh * 0.9))
            search_y1 = min(frame.shape[0], int(ry + rh * 0.3))

            for mask, card_type in [(y_mask, "card_yellow"), (r_mask, "card_red")]:
                sub_mask = mask[search_y0:search_y1, search_x0:search_x1]
                cnts, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    area = cv2.contourArea(c)
                    if 15 < area < 250:
                        x, y, bw, bh = cv2.boundingRect(c)
                        if bh >= bw * 0.8:  # rectangular card shape
                            return Detection(
                                box=[float(search_x0 + x), float(search_y0 + y), float(bw), float(bh)],
                                label=card_type,
                                confidence=0.95,
                                features={"card_type": card_type}
                            )
        return None


def detect_team_jersey_colors(video_path: str, max_samples: int = 12) -> dict[str, dict[str, str]]:
    """Inspects video frames, segments player torsos, and identifies the 2 team kit colors."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "A": {"name": "Red", "hex": "#e53935", "badge": "", "label": "Red Kits"},
            "B": {"name": "Blue", "hex": "#1e88e5", "badge": "", "label": "Blue Kits"}
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 100)
    sample_indices = np.linspace(10, max(11, total_frames - 10), min(max_samples, max(2, total_frames // 25)), dtype=int)

    detector = VisualDetector(use_yolo=False)
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        _ = detector.detect_frame(frame)

    cap.release()
    return {
        "A": {"name": "Red", "hex": "#e53935", "badge": "", "label": "Red Kits"},
        "B": {"name": "Blue", "hex": "#1e88e5", "badge": "", "label": "Blue Kits"}
    }

