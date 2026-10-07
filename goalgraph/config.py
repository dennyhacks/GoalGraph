"""Central configuration. Every knob lives here so results are reproducible."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

EVENT_TYPES = [
    "goal",
    "shot_on_target",
    "foul",
    "corner",
    "yellow_card",
    "red_card",
    "substitution",
    "kickoff",
    "half_time",
    "full_time",
]

# Per-source timing noise (seconds, 1-sigma) used by the fusion engine.
# Commentary lags the action; scoreboard graphics lag the goal; vision is the
# tightest when it fires.  Lags are *systematic* offsets that get removed
# before fusion; sigmas are the residual random error.
SOURCE_LAG = {"audio": 0.6, "visual": 0.0, "scoreboard": 1.2}
SOURCE_SIGMA = {"audio": 0.9, "visual": 0.35, "scoreboard": 0.8}


@dataclass
class PipelineConfig:
    video_path: str = ""
    out_dir: str = str(OUTPUT_DIR)

    # visual
    analysis_fps: float = 5.0          # frames/sec to analyse
    yolo_model: str = "yolov8n.pt"     # swap to yolov8x.pt on a GPU
    detector: str = "auto"             # auto | yolo | color
    use_reid_cnn: bool = False         # OSNet via torchreid if installed

    # audio
    whisper_model: str = "small.en"    # large-v3 on GPU
    whisper_compute: str = "int8"

    # scoreboard
    ocr_fps: float = 1.0
    scoreboard_roi: tuple | None = None  # (x, y, w, h); None = auto locate

    # fusion
    merge_window: float = 6.0          # seconds within which candidates merge
    ci_level: float = 0.95

    # replay
    replay_sim_threshold: float = 0.80

    # VLM verification (optional)
    vlm_provider: str = field(default_factory=lambda: os.getenv("GOALGRAPH_VLM", "none"))

    # misc
    use_cache: bool = True
