"""Multi-object tracker: groups frame detections into continuous tracklets."""
from __future__ import annotations

from typing import Sequence
import numpy as np

from ..schema import Tracklet
from .detector import Detection
from .reid import ReIDMatcher


def box_iou(b1: Sequence[float], b2: Sequence[float]) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


class MultiObjectTracker:
    def __init__(self, iou_thresh: float = 0.25, max_age_frames: int = 4):
        self.iou_thresh = iou_thresh
        self.max_age_frames = max_age_frames
        self.reid = ReIDMatcher()

    def track(self, frame_detections: list[tuple[float, list[Detection]]]) -> tuple[list[Tracklet], dict[str, list[float]]]:
        """Convert a sequence of (t, detections) into Tracklets with ReID embeddings."""
        active_tracks: dict[int, dict] = {}
        completed_tracks: list[Tracklet] = []
        embeddings: dict[str, list[float]] = {}
        next_track_id = 1

        for t, dets in frame_detections:
            matched_det = set()
            matched_trk = set()

            # Match active tracks to new detections by IoU
            for tid, trk_data in list(active_tracks.items()):
                last_box = trk_data["boxes"][-1]
                best_iou = 0.0
                best_didx = None

                for didx, d in enumerate(dets):
                    if didx in matched_det:
                        continue
                    # Only match same general class
                    if (trk_data["label"].startswith("player") and d.label.startswith("player")) or trk_data["label"] == d.label:
                        iou = box_iou(last_box, d.box)
                        if iou > best_iou:
                            best_iou = iou
                            best_didx = didx

                if best_iou > self.iou_thresh and best_didx is not None:
                    matched_det.add(best_didx)
                    matched_trk.add(tid)
                    d = dets[best_didx]
                    trk_data["frames"].append(t)
                    trk_data["boxes"].append(d.box)
                    trk_data["age"] = 0
                    if "color_hist" in d.features:
                        trk_data["features"].append(d.features["color_hist"])
                else:
                    trk_data["age"] += 1
                    if trk_data["age"] > self.max_age_frames:
                        # Finalize tracklet
                        tid_str = f"T{tid}"
                        team = "A" if "player_A" in trk_data["label"] else ("B" if "player_B" in trk_data["label"] else None)
                        tl = Tracklet(
                            tracklet_id=tid_str,
                            global_id=tid_str,
                            team=team,
                            frames=trk_data["frames"],
                            boxes=trk_data["boxes"],
                            jersey_number=trk_data.get("jersey")
                        )
                        completed_tracks.append(tl)
                        if trk_data["features"]:
                            embeddings[tid_str] = np.mean(trk_data["features"], axis=0).tolist()
                        del active_tracks[tid]

            # Initialize new tracks for unmatched detections
            for didx, d in enumerate(dets):
                if didx not in matched_det:
                    team = "A" if "player_A" in d.label else ("B" if "player_B" in d.label else None)
                    active_tracks[next_track_id] = {
                        "frames": [t],
                        "boxes": [d.box],
                        "label": d.label,
                        "team": team,
                        "age": 0,
                        "jersey": None,
                        "features": [d.features["color_hist"]] if "color_hist" in d.features else []
                    }
                    next_track_id += 1

        # Flush remaining active tracks
        for tid, trk_data in active_tracks.items():
            tid_str = f"T{tid}"
            team = "A" if "player_A" in trk_data["label"] else ("B" if "player_B" in trk_data["label"] else None)
            tl = Tracklet(
                tracklet_id=tid_str,
                global_id=tid_str,
                team=team,
                frames=trk_data["frames"],
                boxes=trk_data["boxes"],
                jersey_number=trk_data.get("jersey")
            )
            completed_tracks.append(tl)
            if trk_data["features"]:
                embeddings[tid_str] = np.mean(trk_data["features"], axis=0).tolist()

        # Merge with ReID matcher across camera cuts and occlusions
        merged = self.reid.match_and_merge(completed_tracks, embeddings)
        return merged, embeddings
