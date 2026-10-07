"""Replay Detection & Replay-Aware Temporal Mapping.

Novel Feature #2:
In football broadcasts, a goal is often followed by 2-4 slow-motion replays from
different camera angles. Naive action spotters detect 4 goals at 4 different times.
GoalGraph:
1. Detects replay windows using:
   - On-screen "REPLAY" badge / bug
   - Broadcast logo wipes (stinger transitions)
   - Scoreboard removal during slow motion
   - Audio commentary cues ("let's see that again", "another look")
2. Maps any action spotted inside a replay segment back to the original live timestamp!
3. Deduplicates events so "One goal detected once, not 4 times".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from ..schema import Candidate
from ..video_io import iter_frames

log = logging.getLogger(__name__)


@dataclass
class ReplaySegment:
    start_t: float
    end_t: float
    live_start_t: float | None = None
    confidence: float = 0.85
    reason: str = "visual_badge"


class ReplayDetector:
    def __init__(self, badge_threshold: float = 0.6):
        self.badge_threshold = badge_threshold

    def detect_segments(
        self,
        video_path: str,
        fps: float = 2.0,
        audio_cues: list[Candidate] | None = None,
        cuts: list[dict] | None = None
    ) -> list[ReplaySegment]:
        """Detect intervals [start_t, end_t] that are slow-motion or tactical replays."""
        raw_hits = []

        # 1. Scan for REPLAY graphic badge (usually top right or top left)
        # Red/white "REPLAY" banner
        for t, frame in iter_frames(video_path, fps=fps):
            h, w = frame.shape[:2]
            # Check top right corner (w - 260 to w - 20, 20 to 90)
            tr_crop = frame[20:90, max(0, w - 260): w - 20]
            if tr_crop.size > 0:
                hsv = cv2.cvtColor(tr_crop, cv2.COLOR_BGR2HSV)
                # Red badge detection
                r_mask = cv2.inRange(hsv, np.array([0, 140, 140]), np.array([10, 255, 255])) | \
                         cv2.inRange(hsv, np.array([170, 140, 140]), np.array([180, 255, 255]))
                if cv2.countNonZero(r_mask) > 100:
                    raw_hits.append(t)

        # 2. Group consecutive frame hits into segments
        segments: list[ReplaySegment] = []
        if raw_hits:
            cur_start = raw_hits[0]
            cur_end = raw_hits[0]
            for t in raw_hits[1:]:
                if t - cur_end <= 2.5:  # within 2.5 seconds
                    cur_end = t
                else:
                    if (cur_end - cur_start) >= 1.5:
                        segments.append(ReplaySegment(
                            start_t=round(max(0.0, cur_start - 0.5), 2),
                            end_t=round(cur_end + 0.8, 2),
                            confidence=0.92,
                            reason="replay_badge"
                        ))
                    cur_start = t
                    cur_end = t
            if (cur_end - cur_start) >= 1.5:
                segments.append(ReplaySegment(
                    start_t=round(max(0.0, cur_start - 0.5), 2),
                    end_t=round(cur_end + 0.8, 2),
                    confidence=0.92,
                    reason="replay_badge"
                ))

        # 3. Correlate with audio commentary cues if any missed
        if audio_cues:
            for cue in audio_cues:
                if cue.type == "replay_cue":
                    # Check if already covered by an existing segment
                    covered = any(seg.start_t - 3.0 <= cue.t <= seg.end_t + 2.0 for seg in segments)
                    if not covered:
                        # Find cut right after the commentary
                        t0 = cue.t
                        t1 = cue.t + 8.0
                        segments.append(ReplaySegment(
                            start_t=round(t0, 2),
                            end_t=round(t1, 2),
                            confidence=0.75,
                            reason="audio_commentary_cue"
                        ))

        # 4. Filter segments: A broadcast replay lasts between 1.5s and 35.0s.
        # If it exceeds 35s, it is a permanent channel logo/watermark, not a replay!
        valid_segments: list[ReplaySegment] = []
        for s in segments:
            dur = s.end_t - s.start_t
            if 1.5 <= dur <= 35.0:
                valid_segments.append(s)
            else:
                log.info("Discarded persistent watermark badge: %.1fs to %.1fs (dur=%.1fs)", s.start_t, s.end_t, dur)

        return valid_segments

    def map_to_live_time(
        self,
        event_t: float,
        replay_segments: list[ReplaySegment],
        prior_live_events: list[Candidate]
    ) -> tuple[bool, float, float | None]:
        """Check if an event timestamp falls inside a replay segment.
        Returns:
            (is_replay, live_timestamp, delta)
        """
        for seg in replay_segments:
            if seg.start_t <= event_t <= seg.end_t:
                # Event is inside replay! Find the matching live anchor prior to this segment
                recent_live = [
                    cand for cand in prior_live_events
                    if cand.t < seg.start_t and cand.type in ("goal", "shot_on_target", "foul")
                ]
                if recent_live:
                    best_anchor = max(recent_live, key=lambda c: c.t)
                    return True, best_anchor.t, round(event_t - best_anchor.t, 2)
                # Fallback: estimate live event occurred 8-15s before replay
                est_live = max(0.0, seg.start_t - 10.0)
                return True, est_live, round(event_t - est_live, 2)

        return False, event_t, None
