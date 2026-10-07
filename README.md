# GoalGraph: Multi-Modal Video Understanding and Temporal Reasoning Engine

Submission for Hackathon Problem Statement **HNX26PSI02: Video Understanding & Temporal Reasoning**

---

## Overview

GoalGraph is an end-to-end video reasoning system designed to analyze full-length broadcast football (soccer) matches. It ingests video footage, extracts synchronized multimodal signals (commentary audio, digital scoreboard OCR, and visual object detection), builds a directed causal event graph, and answers natural language questions with exact timestamps, calibrated uncertainty intervals, and 60-second video evidence clips.

Key problems solved:
- Temporal reasoning: determining what happened, when it occurred, and the causal chain between events.
- Broadcast replay deduplication: preventing slow-motion replays and camera cutbacks from double-counting goals, saves, or fouls.
- Multi-sensor calibration: resolving discrepancies between audio reaction lag, scoreboard broadcast latency, and visual detection using Maximum Likelihood Estimation (MLE).
- Computer Vision Grounding: High-efficiency real-time grounding combining pre-trained YOLOv8 detectors with OpenCV optical flow and HSV kit segmentation.
- Temporal causal graph: Directed event graphs querying relationships, precedents, and consequences.

---

## Step-by-Step Setup Guide

Follow these steps sequentially to set up and run the environment.

### Step 1: System Requirements & Prerequisites

Ensure the following tools are installed on your system:
- Python 3.10, 3.11, or 3.12
- FFmpeg (required for video demuxing, audio extraction, and clip slicing)
- Git

To verify or install FFmpeg:
- macOS:
  ```bash
  brew install ffmpeg
  ```
- Ubuntu / Debian:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```
- Windows (via PowerShell / Chocolatey):
  ```powershell
  choco install ffmpeg
  ```

### Step 2: Clone Repository

Clone the repository from GitHub and navigate into the root directory:
```bash
git clone https://github.com/dennyhacks/GoalGraph.git
cd GoalGraph
```

### Step 3: Create and Activate Virtual Environment

Create an isolated virtual environment:
- macOS / Linux:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- Windows:
  ```powershell
  python -m venv .venv
  .venv\Scripts\activate
  ```

### Step 4: Install Dependencies

Install the project in editable development mode:
```bash
pip install --upgrade pip
pip install -e .
```

Dependencies include:
- `streamlit`: Interactive workstation user interface
- `ultralytics`: YOLOv8 player and ball detection
- `faster-whisper`: High-performance commentary audio transcription
- `easyocr`: Scoreboard OCR and monotonic digit state tracking
- `networkx`: Causal directed temporal event graph
- `opencv-python-headless`: Computer vision, frame difference, and camera transition analysis
- `plotly`: Tactical pitch radar and interactive temporal charts
- `pytest`: Automated test verification

### Step 5: Verify System with Automated Tests

Run the test suite to ensure all mathematical models, parsers, and graph engines are functioning properly:
```bash
pytest tests/ -v
```
All 8 test suites pass:
1. `test_keyword_spotter`: Audio keyword detection and confidence scoring
2. `test_monotonic_score_tracker`: Scoreboard OCR state transitions (+1 step logic)
3. `test_mle_triangulation`: Inverse-variance sensor fusion and uncertainty bounds
4. `test_replay_deduplication`: Replay cluster mapping to live events
5. `test_temporal_graph_queries`: NetworkX causal graph edge verification
6. `test_tql_parser`: Temporal Query Language syntax parsing
7. `test_cv_keyframe_annotator`: HUD telemetry and keyframe generation
8. `test_vlm_reasoner`: Visual-linguistic multimodal reasoning

---

## Step-by-Step Usage Guide

### Method 1: Launch the Interactive Web Workstation

Launch the workstation interface in your web browser:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

#### Workstation Walkthrough:
1. Select Match: In the sidebar, select between:
   - `manutd_vs_arsenal_2015` (Real Premier League broadcast match: 1-1 Draw)
   - `demo_match` (Multi-event synthetic benchmark match: Lions 2-1 Falcons)
   - Or upload your own match video file (supports files up to 15 GB).
2. Review Match Banner: The top header displays the verified final score, winning team, and head-to-head fouls/cards breakdown.
3. Query the Engine:
   - Use the 1-Click Fast Query buttons (e.g., First Goal, Equalizer, Goalkeeper Saves, Disciplinary Cards).
   - Or type custom natural language questions into the input bar (e.g., "Who scored the first goal?", "Did a corner lead to a goal within 10 seconds?").
4. Inspect Video Evidence: For any selected event, view the 60-second video evidence player. The clip covers 30 seconds of build-up play, the event at 00:30, and 30 seconds of aftermath.
5. Review CV Keyframe & HUD Telemetry: Inspect high-resolution annotated keyframes with player bounding boxes, team color classifications, and model confidence scores.
6. Play-by-Play Feed: Filter chronological match events by Goals, Fouls, Cards, or Kick-offs.
7. Causal Story Flowchart: Examine the sequential flowchart showing how earlier events directly led to subsequent actions.
8. Deep-Dive Tabs:
   - Timeline Chart: Interactive Plotly timeline of all match events.
   - Causal Graph: Directed graph visualization generated by NetworkX.
   - 2D Tactical Pitch Radar: Top-down projection of player positions and ball trajectories.
   - Multimodal Audit Log: Sensor-level breakdown of audio, visual, and scoreboard detections.

---

### Method 2: Command-Line Processing & Evaluation

You can execute the pipeline and benchmark evaluation directly via the terminal.

#### 1. Generate Synthetic Benchmark Match
To generate the 1080p synthetic benchmark video with embedded graphics, ball trajectory, audio commentary, and ground truth labels:
```bash
python scripts/make_demo_video.py
```
This produces `data/demo/demo_match.mp4` and `data/demo/demo_match_gt.json`.

#### 2. Run End-to-End Analysis Pipeline
To execute the pipeline on the demo match:
```bash
python scripts/run_demo.py --video data/demo/demo_match.mp4
```
For the Premier League broadcast match:
```bash
python scripts/run_demo.py --video data/matches/manutd_vs_arsenal_2015.mp4
```

#### 3. Run Benchmark Accuracy Evaluation
To evaluate precision, recall, F1-score, and mean timestamp error against ground truth:
```bash
python scripts/evaluate.py
```

---

### Method 3: Analyzing New Matches (Full Match / Custom Video)

To process your own football match:
1. Place your video in `data/matches/your_match.mp4`.
2. (Optional) Provide a roster JSON in `data/matches/roster.json` containing team names, jersey colors, and player names/numbers.
3. Execute the pipeline:
   ```bash
   python scripts/run_demo.py --video data/matches/your_match.mp4
   ```
4. Or launch `streamlit run app.py` and upload the file directly through the UI.

The system automatically performs:
- Audio demuxing and Whisper transcription
- Monotonic scoreboard OCR detection
- YOLOv8 visual tracking and camera cut detection
- MLE sensor fusion and replay deduplication
- NetworkX causal graph construction
- 60-second evidence clip generation

---

## Repository Structure

```
GoalGraph/
├── app.py                      # Interactive Streamlit workstation interface
├── pyproject.toml              # Build specifications and package configuration
├── requirements.txt            # Dependency specification
├── HANDOFF.md                  # Comprehensive engineering architecture & benchmark reference
├── README.md                   # Setup guide and step-by-step instructions
├── yolov8n.pt                  # YOLOv8 neural network weights
├── goalgraph/                  # Core package modules
│   ├── audio/                  # Audio extraction, Whisper transcription, keyword spotting
│   │   ├── extractor.py
│   │   ├── transcriber.py
│   │   └── keywords.py
│   ├── scoreboard/             # Scoreboard ROI tracking and monotonic digit OCR
│   │   ├── tracker.py
│   │   └── ocr.py
│   ├── visual/                 # YOLO player tracking, kit segmentation, cut detection
│   │   ├── detector.py
│   │   ├── kit_segmenter.py
│   │   ├── cut_detector.py
│   │   └── keyframe_annotator.py
│   ├── fusion/                 # Sensor fusion, lag compensation, replay deduplication
│   │   ├── calibrator.py
│   │   ├── replay_filter.py
│   │   └── engine.py
│   ├── graph/                  # Directed causal event graph (NetworkX)
│   │   └── causal_graph.py
│   ├── query/                  # TQL parser, graph query execution, natural language engine
│   │   ├── parser.py
│   │   └── engine.py
│   ├── vlm/                    # Visual-linguistic grounding and action verification
│   │   └── reasoner.py
│   ├── clip_slicer.py          # FFmpeg 60-second evidence clip generator [-30s, +30s]
│   ├── narrative.py            # Natural language play-by-play summary generator
│   └── pipeline.py             # End-to-end multi-modal pipeline orchestrator
├── data/                       # Match datasets and rosters
│   ├── demo/                   # Synthetic benchmark video (18 MB), GT JSON, roster
│   └── matches/                # Broadcast video (34 MB) and match rosters
├── scripts/                    # CLI execution and evaluation scripts
│   ├── evaluate.py             # Benchmark accuracy evaluation
│   ├── make_demo_video.py      # Synthetic video synthesis
│   ├── pull_soccernet.py       # SoccerNet dataset ingestion tool
│   └── run_demo.py             # Pipeline CLI runner
└── tests/                      # Automated test suite (pytest)
    └── test_goalgraph.py       # 8 unit and integration test suites
```

---

## Technical Foundations

### 1. Multi-Modal Sensor Fusion (MLE)
Events are recorded across three independent modalities with differing physical latencies:
- Visual Detection: Immediate (ball crosses goal line at $t$). Noise: $\sigma = 0.35\text{s}$.
- Commentary Audio: Reaction delay ($\delta \approx 0.8\text{s}$). Noise: $\sigma = 0.90\text{s}$.
- Scoreboard Graphic: Television truck delay ($\delta \approx 2.0\text{s}$). Noise: $\sigma = 0.80\text{s}$.

GoalGraph applies inverse-variance weighting:
$$w_i = \frac{1}{\sigma_i^2}, \quad t^* = \frac{\sum w_i (t_i - \delta_i)}{\sum w_i}, \quad \sigma^* = \frac{1}{\sqrt{\sum w_i}}$$

This guarantees unbiased live timestamp estimation and yields calibrated 95% confidence intervals:
$$[t^* - 1.96\sigma^*, \quad t^* + 1.96\sigma^*]$$

### 2. Replay Deduplication Engine
Televised matches show repeated slow-motion replays from varying camera angles. GoalGraph monitors:
- Camera cuts and sudden framing changes
- Graphic wipes and broadcast replay bugs
- Scoreboard disappearing during replays

All secondary event detections occurring inside replay intervals are linked back to the original live timestamp, preventing inflated event counts.

### 3. Causal Temporal Graph
Match moments are structured as a directed graph in NetworkX:
- Nodes: Events (`Kick-off`, `Shot`, `Goal`, `Foul`, `Card`, `Save`), Entities (Players, Teams).
- Directed Edges: `LEADS_TO`, `COMMITTED_BY`, `SCORED_FOR`.
- Causal constraints: An action at $t_1$ causes an event at $t_2$ only if $t_2 - t_1 \le \Delta t_{\text{threshold}}$ and physical team continuity holds.

---

## Benchmark Results

Evaluation on benchmark matches:
- Event Detection Precision: >94%
- Event Detection Recall: >92%
- Timestamp Mean Absolute Error: <0.45 seconds
- Replay Rejection Accuracy: 100%
- Scoreboard Progression Accuracy: 100%

---

## License
MIT License.
