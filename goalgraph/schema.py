"""Data model shared by every branch of the pipeline."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Candidate:
    """A single-source hint that an event happened around time ``t``.

    Produced independently by the audio, visual and scoreboard branches and
    consumed by the fusion engine.  ``payload`` keeps raw provenance
    (keyword, OCR string, tracklet id, bbox ...).
    """

    source: str                 # audio | visual | scoreboard
    type: str                   # event type from config.EVENT_TYPES
    t: float                    # video seconds
    confidence: float = 0.5
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Evidence:
    clip: str | None = None
    keyframe: str | None = None
    bbox: list[float] | None = None          # x, y, w, h in pixels
    commentary: str | None = None
    audio_timestamp: float | None = None
    visual_timestamp: float | None = None
    scoreboard_timestamp: float | None = None
    scoreboard_before: str | None = None
    scoreboard_after: str | None = None
    match_clock: str | None = None
    tracklet_id: str | None = None
    replay_segments: list[list[float]] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)   # raw candidates used
    vlm: dict | None = None


@dataclass
class Event:
    event_id: str
    type: str
    start_time: float
    end_time: float
    live_timestamp: float                     # fused point estimate
    time_interval: list[float]                # CI on live_timestamp
    time_sigma: float
    time_confidence: float
    confidence: float
    team: str | None = None
    player_id: str | None = None
    half: int = 1
    is_replay: bool = False
    replay_count: int = 0
    evidence: Evidence = field(default_factory=Evidence)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Event":
        d = dict(d)
        d["evidence"] = Evidence(**d.get("evidence", {}))
        return Event(**d)


@dataclass
class Tracklet:
    tracklet_id: str               # local track id (resets across cuts)
    global_id: str                 # ReID identity, stable across cuts/occlusion
    team: str | None
    frames: list[float] = field(default_factory=list)      # timestamps
    boxes: list[list[float]] = field(default_factory=list) # x, y, w, h
    jersey_number: str | None = None

    @property
    def start(self) -> float:
        return self.frames[0] if self.frames else 0.0

    @property
    def end(self) -> float:
        return self.frames[-1] if self.frames else 0.0

    def box_at(self, t: float) -> list[float] | None:
        if not self.frames:
            return None
        best = min(range(len(self.frames)), key=lambda i: abs(self.frames[i] - t))
        if abs(self.frames[best] - t) > 1.0:
            return None
        return self.boxes[best]


def dump_json(obj: Any, path) -> None:
    def default(o):
        if hasattr(o, "to_dict"):
            return o.to_dict()
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)
        if hasattr(o, "tolist"):
            return o.tolist()
        if hasattr(o, "item"):
            return o.item()
        raise TypeError(type(o))

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=default)


def load_json(path) -> Any:
    with open(path) as f:
        return json.load(f)
