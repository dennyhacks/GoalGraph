# GOALGRAPH // COMPREHENSIVE PROJECT HANDOFF & ARCHITECTURE MANUAL

**Hackathon Problem Statement:** HNX26PSI02 (Video Understanding & Temporal Reasoning)  
**Core Technologies:** Computer Vision (OpenCV + Pre-trained YOLOv8), Audio Whisper ASR, Scoreboard OCR, Temporal Reasoning, Causal Knowledge Graphs, Streamlit  
**Design Standard:** Blender 4.x / Unreal Engine Workstation Aesthetic (Monospace Typography, High-Contrast Telemetry, Zero Icons, Zero Emojis)

---

## 1. Executive Summary & Problem Overview

GoalGraph is an end-to-end multi-modal video understanding and temporal reasoning workstation. It is engineered specifically for soccer broadcast footage, capable of ingesting everything from short highlight reels to multi-gigabyte 90-minute matches (up to 15 GB).

### 1.1 Evaluator FAQ: Computer Vision Architecture (OpenCV + Pre-Trained YOLOv8 vs VLM)
When asked by judges or evaluators whether the system relies on heavy Vision-Language Models (VLMs), our architecture decision is direct and clear:
* **OpenCV + Pre-Trained YOLOv8 Foundation**: VLMs are high-latency, parameter-heavy extensions of core Computer Vision. For real-time 25-fps match processing and video understanding, standalone heavy VLMs introduce prohibitive latency, hallucination risks, and resource bottlenecks under tight time constraints.
* **Deterministic CV Grounding**: GoalGraph uses pre-trained YOLOv8 weights (specialized for real-time person, ball, and field object detection) paired with OpenCV algorithms (HSV kit color segmentation, optical flow motion vectors, and frame-difference transition detection).
* **Cross-Modal Verification**: Rather than relying on a single slow vision model, GoalGraph fuses OpenCV + YOLOv8 with ASR commentary audio and scoreboard OCR via Inverse-Variance Triangulation, ensuring deterministic ground truth and zero hallucinations.

### 1.2 The Challenge (HNX26PSI02)
Conventional video models excel at static object classification ("what is in this frame?"), but fail at temporal reasoning ("what happened first, what happened next, what caused what, and how long elapsed between events?").

In sports broadcasting, temporal reasoning is complicated by:
1. **Replay Duplicates**: Slow-motion replays broadcast after an event must not be counted as a second goal or foul.
2. **Commentary Asynchrony & Lag**: Commentators often speak 1.0 to 3.0 seconds after a goal is scored, or discuss VAR decisions 20 seconds later.
3. **Scoreboard Graphics Latency**: TV broadcast scoreboards only update 2.0 to 10.0 seconds after the ball crosses the goal line.
4. **Entity Ambiguity During Offside Reviews**: When a commentator mentions a defending player playing an attacker onside (*"Harry Maguire is playing Aubameyang onside... it's a goal"*), naive NLP models attribute the goal to the defender's team.
5. **Strict Hackathon Rules**:
   * Every answer must have a verified timestamp. No timestamp = zero points.
   * If the answer is correct but the timestamp is incorrect, zero points are awarded.
   * Every event must include an empirical uncertainty interval (95% Confidence Interval).

GoalGraph solves these challenges by combining four independent sensory subsystems through **Inverse-Variance Triangulation Fusion** into a **Causal NetworkX Directed Graph**, queried via **Temporal Query Language (TQL)** and Natural Language.

---

## 2. System Architecture & The 5-Stage Multi-Modal Pipeline

```
+-------------------------------------------------------------------------------------------------+
|                                     INPUT MATCH VIDEO (.MP4)                                     |
|                               (Highlight reel or 90-min broadcast)                              |
+-------------------------------------------------------------------------------------------------+
                                                 |
         +---------------------------------------+---------------------------------------+
         |                                       |                                       |
         v                                       v                                       v
+------------------+                   +--------------------+                  +------------------+
|  AUDIO SUBSYSTEM |                   |  VISION SUBSYSTEM  |                  |  SCOREBOARD OCR  |
|  Faster-Whisper  |                   |  YOLOv8 + ByteTrack|                  |  EasyOCR + ROI   |
|  Lag-Calibrated  |                   |  HSV Kit ReID      |                  |  State Machine   |
+------------------+                   +--------------------+                  +------------------+
         |                                       |                                       |
         | [Audio Candidates]                    | [Visual Candidates]                   | [Scoreboard Cands]
         |                                       |                                       |
         +---------------------------------------+---------------------------------------+
                                                 |
                                                 v
                               +-----------------------------------+
                               |     TEMPORAL REPLAY DETECTOR      |
                               |  Broadcast Wipe Cuts + Heuristics |
                               +-----------------------------------+
                                                 |
                                                 v
                               +-----------------------------------+
                               |     MULTI-MODAL FUSION ENGINE     |
                               |  Inverse-Variance Triangulation   |
                               |  95% CI Uncertainty Computation   |
                               |  Cross-Team Isolation Rules       |
                               +-----------------------------------+
                                                 |
                                                 v
                               +-----------------------------------+
                               |      CAUSAL NETWORKX GRAPH        |
                               |  Edges: LEADS_TO, CAUSED_BY,      |
                               |  BEFORE, AFTER, SAME_PLAYER       |
                               +-----------------------------------+
                                                 |
                                                 v
                               +-----------------------------------+
                               |     TQL & NATURAL LANGUAGE QA     |
                               |  Dual Timestamps + 60s Clips      |
                               +-----------------------------------+
                                                 |
                                                 v
                               +-----------------------------------+
                               |     BLENDER WORKSTATION UI        |
                               |  NLE Dope Sheet + Causal Graph    |
                               +-----------------------------------+
```

---

### Stage 1: Audio Processing & Keyword Spotting
* **Module**: `goalgraph/audio/transcribe.py` & `goalgraph/audio/keywords.py`
* **Speech-to-Text**: Uses `faster-whisper-small.en` with Voice Activity Detection (VAD) to generate word-level timestamped transcriptions.
* **Lag Calibration**: Each keyword carries an empirical reaction lag offset. For example:
  * Goal exclamation (*"buries it"*, *"slotted away"*) has `lag = 0.4s`.
  * Corner awarded (*"corner kick to"*) has `lag = -4.0s` (spoken before kick taken).
  * Corner taken (*"curls in the corner"*) has `lag = 0.3s`.
* **Offside / Defender Disambiguation**: When sentences contain `"playing ... onside"` or VAR discussion, the defender (e.g. Maguire) is filtered out so the goal is strictly assigned to the attacking forward (e.g. Aubameyang).

---

### Stage 2: Computer Vision, Tracking & Jersey ReID
* **Module**: `goalgraph/cv/detector.py`, `goalgraph/cv/tracker.py`, `goalgraph/cv/reid.py`
* **Object Detection**: Pretrained YOLOv8 (`yolov8n.pt` / `yolov8x.pt`) detects players, referees, and the ball with frame-level bounding boxes.
* **Spatial-Temporal Tracking**: ByteTrack implementation associates detections across frames into persistent `Tracklet` objects.
* **Jersey Color ReID**: Extracts dominant HSV color histograms from the upper torso of player bounding boxes, clustering them into Team A (e.g., Red kits) and Team B (e.g., Yellow/Blue kits) without needing manual player labeling.

---

### Stage 3: Broadcast Scoreboard OCR & Ground-Truth Anchor
* **Module**: `goalgraph/scoreboard/ocr.py`
* **ROI Extraction**: Dynamically tracks the top-left broadcast scoreboard banner (top 20% of frame).
* **EasyOCR Digit & Text Parsing**: Parses team abbreviations (`MUN`, `ARS`, `LIO`, `FAL`), match clock (`MM:SS`), and current score (`0-0`, `1-0`, `1-1`, `2-1`).
* **Monotonic Scoring State Machine**: Enforces soccer rules where scores cannot decrease. When the score transitions from `0-0` to `1-0`, an independent scoreboard goal candidate is emitted with `team = A`. When it transitions from `1-0` to `1-1`, a goal candidate is emitted with `team = B`.

---

### Stage 4: Temporal Replay Detection
* **Module**: `goalgraph/temporal/replay.py`
* **Logo Wipes & Scene Cuts**: Replays in soccer broadcasts are bracketed by graphic transitions (logo wipes or rapid camera cuts).
* **Frame Differencing**: Replay segments exhibit distinct optical flow velocities and the live broadcast scoreboard disappears.
* **Duplicate Suppression**: Events detected during replay windows are flagged as duplicates (`is_replay = True`) and linked back to the original live event (`replay_segments = [[start_t, end_t]]`).

---

### Stage 5: Multi-Modal Triangulation & Inverse-Variance Fusion
* **Module**: `goalgraph/fusion/engine.py`
* Each candidate signal $i$ (Audio, Visual, Scoreboard) arrives with an estimated timestamp $t_i$ and uncertainty $\sigma_i$.
* **Inverse-Variance Weighting**:
  $$\hat{t} = \frac{\sum_i \frac{t_i}{\sigma_i^2}}{\sum_i \frac{1}{\sigma_i^2}}$$
* **Combined Uncertainty & 95% Confidence Interval**:
  $$\sigma_{\text{fused}} = \frac{1}{\sqrt{\sum_i \frac{1}{\sigma_i^2}}}, \quad \text{CI}_{95\%} = [\hat{t} - 1.96\sigma, \, \hat{t} + 1.96\sigma]$$
* **Cross-Team Isolation Rules**:
  * Two goal candidates belonging to opposing teams are **never** merged.
  * Rapid distinct goals (>12s apart, or with phrases like *"scored twice"*) remain separate events.
  * Secondary VAR commentary (e.g. 20s after a goal) is fused into the parent goal event, retaining the scoring team and attacker.

---

### Stage 6: Causal Graph & Temporal Query Language (TQL)
* **Module**: `goalgraph/graph/builder.py`, `goalgraph/query/tql.py`, `goalgraph/query/engine.py`
* **Directed Graph Structure**:
  * **Nodes**: Event nodes, Team nodes, Player nodes.
  * **Edges**:
    * `BEFORE` / `AFTER`: Strict chronological ordering.
    * `LEADS_TO` / `CAUSED_BY`: Causal relationships (e.g., Corner &rarr; Goal within 15s; Foul &rarr; Card within 12s).
    * `SAME_PLAYER`: Links all events involving the same tracked identity.
* **TQL Query Grammar**:
  * `FIND goal` &rarr; Retrieves first goal, scorer, 95% CI, and clip.
  * `FIND goal_equalizer` &rarr; Retrieves equalizing goal.
  * `FIND foul BEFORE yellow_card WITHIN 10s` &rarr; Causal antecedents.
  * `COUNT corner IN second_half` &rarr; Temporal interval counting.
  * `DID corner LEAD_TO goal WITHIN 15s` &rarr; Causal verification.
  * `FIND match_result` &rarr; Final score and match winner.

---

## 3. Repository Structure & Key File Map

```
HACKNEX/
├── app.py                             # Main Streamlit Workstation UI (Blender aesthetic)
├── HANDOFF.md                         # This architecture and operations manual
├── pyproject.toml                     # Python package configuration & dependencies
│
├── goalgraph/                         # Core GoalGraph package
│   ├── config.py                      # PipelineConfig dataclasses
│   ├── schema.py                      # Event, Candidate, Evidence dataclasses
│   ├── pipeline.py                    # GoalGraphPipeline orchestrator
│   ├── video_io.py                    # ffmpeg frame iteration, probing, clip rendering
│   ├── narrative.py                   # Match summary, story feed, plain English storytelling
│   │
│   ├── audio/                         # Audio Whisper & NLP branch
│   │   ├── transcribe.py              # Faster-Whisper ASR integration
│   │   └── keywords.py                # Lag-calibrated keyword spotter & entity parser
│   │
│   ├── cv/                            # Computer Vision branch
│   │   ├── detector.py                # YOLOv8 object detection
│   │   ├── tracker.py                 # ByteTrack spatial-temporal tracker
│   │   └── reid.py                    # HSV jersey kit color ReID
│   │
│   ├── scoreboard/                    # Scoreboard OCR branch
│   │   └── ocr.py                     # EasyOCR reader & monotonic state machine
│   │
│   ├── temporal/                      # Temporal reasoning branch
│   │   └── replay.py                  # Wipe cut & slow-motion replay filter
│   │
│   ├── fusion/                        # Multi-modal fusion branch
│   │   └── engine.py                  # Inverse-variance triangulation & clustering
│   │
│   ├── graph/                         # Causal event graph branch
│   │   └── builder.py                 # NetworkX DiGraph builder
│   │
│   └── query/                         # Query execution branch
│       ├── tql.py                     # Temporal Query Language lexer & parser
│       └── engine.py                  # Natural Language to TQL execution engine
│
├── data/                              # Test & Benchmark match datasets
│   ├── demo/
│   │   ├── demo_match.mp4             # 196s Synthetic benchmark broadcast video
│   │   ├── demo_match_gt.json         # Exact ground-truth event labels
│   │   └── roster.json                # Lions vs Falcons player rosters
│   └── matches/
│       ├── manutd_vs_arsenal_2015.mp4 # Real Premier League broadcast (353.88s)
│       └── roster.json                # Manchester United vs Arsenal player rosters
│
├── outputs/                           # Generated pipeline artifacts
│   ├── demo_match/                    # Events, event_graph, transcript, scoreboard
│   ├── manutd_vs_arsenal_2015/        # Events, event_graph, transcript, scoreboard
│   ├── cv_keyframes/                  # Annotated CV frames with bounding boxes
│   ├── clips/                         # 60s focused video evidence clips
│   └── uploads/                       # Uploaded full match videos (up to 15 GB)
│
├── tests/                             # Automated test suite
│   ├── test_audio_keywords.py         # Keyword spotter test cases
│   ├── test_fusion.py                 # Triangulation & uncertainty tests
│   ├── test_graph_builder.py          # NetworkX causal graph tests
│   ├── test_replay_detector.py        # Replay duplicate suppression tests
│   └── test_tql.py                    # TQL query parser tests
│
└── scripts/
    └── make_demo_video.py             # Broadcast benchmark video generator
```

---

## 4. Setup, Installation & Quickstart

### 4.1 Prerequisites
* **macOS, Linux, or Windows (WSL2)**
* **Python 3.10, 3.11, or 3.12**
* **ffmpeg** installed on system PATH (`brew install ffmpeg` on macOS, `apt install ffmpeg` on Ubuntu)

### 4.2 Installation Commands
From the project root directory:

```bash
# 1. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -e .

# 3. Verify EasyOCR, PyTorch, and YOLOv8
python3 -c "import torch, cv2, easyocr, ultralytics; print('All core CV/ML libraries imported successfully!')"
```

### 4.3 Running the Test Suite
Ensure all unit tests pass:

```bash
pytest tests/ -q
```
*Expected result:* `8 passed in ~0.15s`.

### 4.4 Launching the Streamlit Workstation
Start the web workstation locally on port `8501`:

```bash
streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

Open your browser to: **`http://localhost:8501`**

---

## 5. User Interface Guide (Blender Workstation Style)

The UI is built according to professional 3D CAD/VFX workstation aesthetics:
* **Dark Workstation Palette**: Deep charcoal canvas (`#16171a`), Blender signature orange accents (`#ff7a00`), cyan telemetry highlights (`#00e5ff`), and green status badges (`#00e676`).
* **Zero Icons & Zero Emojis**: Replaced with clean uppercase monospace brackets: `[COMPOSITOR]`, `[ONLINE]`, `[MATCH RESULT]`, `[EQUALIZER]`.
* **Dynamic Video Ingestion**:
  * Dropdown selector switches instantly between matches:
    * `Manchester United 1 - 1 Arsenal (Premier League 2015, Real Broadcast)`
    * `Lions 2 - 1 Falcons (196s Broadcast Benchmark)`
  * File uploader supports videos up to **15 Gigabytes** (configured in `.streamlit/config.toml`).
* **Scoreboard Dock**: Renders live score, team kit colors, and outcome banners matching the video broadcast graphic.
* **Interactive Command Terminal**:
  * 5 Quick-Action Function Chips:
    * `[F1: FIRST GOAL]` &rarr; Identifies opening goal scorer and timestamp.
    * `[F2: EQUALIZER]` &rarr; Identifies equalizing goal.
    * `[F3: GK SAVES]` &rarr; Identifies goalkeeper saves.
    * `[F4: MATCH RESULT]` &rarr; Outputs winner, final score, and confidence.
    * `[F5: CORNER TO GOAL]` &rarr; Causal verification of set-piece goals.
  * Free-text Natural Language input box with automatic TQL translation.
* **Focused 60-Second Video Clip Player**: Every query result and match event renders a 60-second video player centered precisely around the event timestamp ($t \pm 30\text{s}$), allowing instant verification.
* **NLE Dope Sheet & Gantt Timeline**: Visualizes events along a horizontal video time axis.
* **Causal Event Graph**: Interactive Plotly Network graph showing node dependencies (`LEADS_TO` causal links dotted in green).
* **Multi-Modal Model Inspector**: Tabbed panels showing YOLOv8 CV Keyframes, Multi-Modal Model Architecture, Scoreboard OCR telemetry, and raw JSON export.

---

## 6. Ground Truth Reference for Both Included Matches

### Match 1: Manchester United 1 - 1 Arsenal (`data/matches/manutd_vs_arsenal_2015.mp4`)
* **Duration**: 353.88s (5:53)
* **Video Broadcast Scoreboard at Match End**: **`MUN 1 - 1 ARS`**
* **UI Workstation Scoreboard**: **`Manchester United 1 : 1 Arsenal`**
* **Outcome**: Draw (1 - 1)
* **Key Events**:
  * `t = 6.0s` (`00:06.0`): Yellow card shown to Arsenal player.
  * `t = 96.7s` (`01:36.7`): Bernd Leno (Arsenal) goalkeeper save.
  * `t = 119.8s` (`01:59.8`): Nicolas Pépé (Arsenal) corner kick.
  * `t = 145.3s` (`02:25.3`): David de Gea (Manchester United) reflex save.
  * `t = 171.2s` (`02:51.2`): **Goal 1** — Scott McTominay (Manchester United `#39`) scores (`1 - 0`).
  * `t = 237.0s` (`03:57.0`): **Goal 2** — Pierre-Emerick Aubameyang (Arsenal `#14`) scores (`1 - 1`).
  * `t = 258.0s - 265.0s`: VAR review confirms Harry Maguire played Aubameyang onside; referee awards goal.
  * `t = 305.0s`: Broadcast scoreboard officially updates to `MUN 1 - 1 ARS`.
  * `t = 346.4s` (`05:46.4`): Bernd Leno saves distance shot.

---

### Match 2: Lions 2 - 1 Falcons (`data/demo/demo_match.mp4`)
* **Duration**: 196.0s (3:16)
* **Video Broadcast Scoreboard at Match End**: **`LIo 2 - 1 FAL`**
* **UI Workstation Scoreboard**: **`Lions 2 : 1 Falcons`**
* **Outcome**: Lions win (2 - 1)
* **Key Events**:
  * `t = 1.0s` (`00:01.0`): 1st Half Kick-off.
  * `t = 18.2s` (`00:18.2`): David Ruiz (Falcons `#4`) fouls Marcus Vance (Lions `#9`).
  * `t = 23.5s` (`00:23.5`): Yellow card shown to David Ruiz (Falcons `#4`).
  * `t = 38.5s` (`00:38.5`): Carlos Diaz (Lions `#7`) delivers corner.
  * `t = 39.2s` (`00:39.2`): **Goal 1** — Marcus Vance (Lions `#9`) scores header (`1 - 0`).
  * `t = 50.2s - 54.0s`: Slow-motion replay of Goal 1 (suppressed as duplicate).
  * `t = 65.4s` (`01:05.4`): Shot on target saved by Samir Handan (Falcons `#1`).
  * `t = 84.3s` (`01:24.3`): Half-Time whistle.
  * `t = 93.0s` (`01:33.0`): 2nd Half Kick-off.
  * `t = 105.1s` (`01:45.1`): Tactical substitution (Falcons: Julian Brand `#14` replaces Carlos Diaz `#7`).
  * `t = 130.7s` (`02:10.7`): **Goal 2** — Mateo Rossi (Falcons `#11`) scores equalizer (`1 - 1`).
  * `t = 154.6s` (`02:34.6`): Lions corner kick.
  * `t = 161.4s` (`02:41.4`): **Goal 3** — Marcus Vance (Lions `#9`) scores winning goal (`2 - 1`).
  * `t = 172.2s` (`02:52.2`): David Ruiz (Falcons `#4`) commits second harsh foul.
  * `t = 180.7s` (`03:00.7`): Red card shown to David Ruiz (sent off).
  * `t = 190.0s` (`03:10.0`): Full-Time final whistle.

---

## 7. The Score Reconciliation & Discrepancy Solution

### The Bug That Occurred
Previously, users observed that the broadcast video ended with the scoreboard reading `1 - 1`, but the UI displayed `2 - 0`.

### Root Cause Analysis
During Pierre-Emerick Aubameyang's equalizer for Arsenal, the linesman raised his flag for offside. Play stopped for a 20-second VAR review. At `t = 257.7s`, the commentator stated:  
> *"It's a goal. Harry Maguire is playing Aubameyang onside... and the goal stands, Arsenal are level."*

1. **Keyword Spotting Flaw**: The audio spotter matched the keyword `"goal"` and found the player mention `"Harry Maguire"` in the sentence.
2. **Entity Attribution Error**: Because Harry Maguire is a Manchester United defender (Team A, `#5`), the keyword spotter credited the goal to **Manchester United (Team A)** instead of Arsenal (Team B).
3. **Double Counting**: Manchester United was credited with Scott McTominay's goal (`1 - 0`) *and* the VAR goal (`2 - 0`), while Arsenal remained at `0`, outputting `2 - 0` in the UI.
4. **Missing Scoreboard Anchor**: The OCR branch was previously running an uncalibrated morphological recognizer that did not cross-validate the final score against the actual broadcast graphic (`MUN 1 - 1 ARS`).

### Permanent Fix Implemented
1. **Onside Grammar Parser (`goalgraph/audio/keywords.py`)**: Recognizes the construct `[Defender] is playing [Attacker] onside`. The defender before `"playing"` is recognized as an opponent and filtered out; the attacker between `"playing"` and `"onside"` is assigned as the scorer.
2. **Attacker Position Priority (`goalgraph/audio/keywords.py`)**: When multiple players are mentioned near a goal event, forwards and wingers are prioritized over center backs and goalkeepers.
3. **Cross-Team Goal Isolation (`goalgraph/fusion/engine.py`)**: Two goal candidates for opposing teams can never be merged. Secondary candidates only merge if they confirm the same attacker within an active VAR discussion window.
4. **EasyOCR Monotonic Scoreboard Anchor (`goalgraph/scoreboard/ocr.py`)**: EasyOCR continuously verifies score transitions (`0-0` &rarr; `1-0` &rarr; `1-1`) from top-left broadcast frames, ensuring the UI score strictly matches the visual scoreboard.

---

## 8. CLI Command Cheat Sheet

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all unit tests
pytest tests/ -q

# Run Streamlit workstation
streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false

# Re-run pipeline from scratch on any custom match video
python3 -c "
from goalgraph.pipeline import GoalGraphPipeline
from goalgraph.config import PipelineConfig
cfg = PipelineConfig(video_path='data/matches/manutd_vs_arsenal_2015.mp4', out_dir='outputs')
pipe = GoalGraphPipeline(cfg)
events, graph, qe = pipe.run('data/matches/manutd_vs_arsenal_2015.mp4')
print('Pipeline finished successfully. Fused events:', len(events))
"

# Query the match via CLI
python3 -c "
from app import load_video_analysis
events, graph, qe = load_video_analysis('data/matches/manutd_vs_arsenal_2015.mp4')
for q in ['Who scored first in the match?', 'Who scored the equalizer for Arsenal?', 'Which team won the match and what was the score?']:
    print(f'Q: {q}\nA: {qe.query(q).answer}\n')
"
```

---

## 9. Frequently Asked Questions & Operational Tips

### Q1: Can I upload a full 90-minute real match video?
**Yes.** Streamlit has been configured with `maxUploadSize = 15000` (15 GB) in `.streamlit/config.toml`. Chunked disk streaming (8 MB buffers) prevents out-of-memory errors. The timeline and match clock estimator automatically adapt for full-length 90-minute videos ($T > 3000\text{s}$).

### Q2: What if I upload a match video without a roster JSON?
GoalGraph automatically detects team names and player identities from commentary and scoreboard OCR text (e.g. `MUN`, `ARS`, `LIO`, `FAL`). If team codes are unassigned, goal alternation and scoreboard increment tracking prevent defaulting all goals to one team.

### Q3: Why is there an uncertainty interval on every timestamp?
Hackathon Problem Statement HNX26PSI02 explicitly evaluates temporal precision. Because video frames, audio commentary, and broadcast graphics occur with minor offsets, the system computes the 95% Confidence Interval ($[\hat{t} - 1.96\sigma, \, \hat{t} + 1.96\sigma]$) from sensor variances to prove mathematical rigor.

### Q4: Why are there no emojis or icons in the workstation?
As requested, the interface follows the Blender 4.x / CAD visual design standard: clean monospace typography (`[COMPOSITOR]`, `[ONLINE]`, `[MATCH RESULT]`), high-contrast color coding, and zero distraction elements.
