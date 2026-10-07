# GoalGraph: Multimodal Video Understanding and Temporal Reasoning Engine

Hackathon Submission for Problem Statement: HNX26PSI02 - Video Understanding and Temporal Reasoning

---

## What is GoalGraph? (In Plain Terms, Without Buzzwords)

When you watch a televised football (soccer) match, it lasts at least 90 minutes. If you want to find every goal, yellow card, or foul, you usually have two choices:
1. Manually fast-forward through 90 minutes of video, which is slow and tiring.
2. Feed the entire video into a giant artificial intelligence model. However, 90-minute video files are huge (often 5 to 15 gigabytes). Giant AI models either run out of memory, crash, or guess random timestamps that are off by 20 to 30 seconds.

GoalGraph is a lightweight, practical system that solves this problem. It watches and listens to the video stream, figures out the exact second important events happen, connects the chain of events together, and lets you search the match using plain English questions.

When you ask a question like "Who scored the first goal?", GoalGraph does not guess. It points you to the exact second, shows you a verified 60-second video evidence clip (30 seconds of build-up play, the event itself, and 30 seconds of aftermath), and displays an annotated frame showing where the players and ball were located.

---

## What the System Does (Step by Step in Simple Terms)

GoalGraph breaks the job down into specialized steps instead of relying on a single black-box model:

1. Listening for the Whistle (Acoustic Processing)
   The referee's whistle has a sharp, high-pitched frequency (around 2,500 to 4,500 Hz). The system uses audio frequency filters to detect whistle blasts within 0.1 seconds of real time.

2. Listening to the Commentators (Speech-to-Text)
   It transcribes the commentator's voice into text and records the exact microsecond each word was spoken. If the commentator shouts "Goal!", the system notes the time.

3. Reading the Scoreboard (Scorebug OCR)
   It crops the broadcast scoreboard on screen and uses text recognition to read the score numbers. It enforces a strict rule: scores can only stay the same or go up by 1 (+1). If the number glitches or reads a random blur, it rejects it.

4. Tracking Players and the Ball (Computer Vision)
   It uses an object detector (YOLOv8) to track where players, referees, and the ball are on the pitch.

5. Ignoring Replays (Replay Filtering)
   When a goal is scored, TV broadcasts immediately show the same goal two or three times from different camera angles in slow motion. If a computer system is not careful, it will think three goals were scored instead of one. GoalGraph watches for broadcast graphic wipes (the transition logo that sweeps across the screen) and marks any repeated clips as replays, so the score is never inflated.

6. Synchronizing the Clocks (Sensor Calibration)
   Different sources react at different speeds:
   - The ball hits the net: Immediate (Time = 0s)
   - The commentator shouts: About 0.8 seconds later (human reaction delay)
   - The TV scoreboard graphic updates: About 2.0 seconds later (production truck delay)
   GoalGraph uses a mathematical formula (Inverse-Variance Maximum Likelihood Estimation) to line up these delays, giving the true physical timestamp with a 95% confidence interval.

7. Building the Event Timeline (Causal Graph)
   All verified events are saved into a timeline graph. This allows the system to understand causal connections, such as whether a corner kick directly resulted in a goal.

8. Answering Questions (Query Engine)
   You can click pre-built query buttons or type normal questions (e.g., "Show me the equalizer", "Who scored for France?"). The engine parses the query, finds the matching event in the graph, and loads the video clip immediately.

---

## Step-by-Step Installation Guide (For Every Laptop)

GoalGraph runs locally on personal laptops and desktops. Select the instructions for your operating system below.

### Hardware and Software Requirements
- Operating System: macOS (Apple Silicon or Intel), Windows 10/11, or Linux (Ubuntu, Debian, Fedora)
- RAM: 8 GB minimum (16 GB recommended)
- Python Version: Python 3.10, 3.11, or 3.12 (Python 3.10 or 3.11 is recommended)
- Disk Space: At least 2 GB of free disk space

---

### Option A: Setup on Apple Mac (macOS - M1, M2, M3, M4 or Intel)

1. Open the Terminal application (Press Command + Space, type "Terminal", and press Enter).

2. Install Homebrew (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. Install Git, Python, and FFmpeg:
   ```bash
   brew install git python@3.11 ffmpeg
   ```

4. Verify installations:
   ```bash
   python3 --version
   ffmpeg -version
   git --version
   ```

5. Clone the repository and navigate into the folder:
   ```bash
   git clone https://github.com/dennyhacks/GoalGraph.git
   cd GoalGraph
   ```

6. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   (Your terminal prompt will now display `(.venv)` at the beginning).

7. Install the dependencies:
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

8. Verify the setup by running the automated tests:
   ```bash
   pytest tests/ -v
   ```

9. Start the application:
   ```bash
   streamlit run app.py
   ```
   Your web browser will automatically open to `http://localhost:8501`.

---

### Option B: Setup on Windows (Windows 10 or Windows 11)

1. Install Python:
   - Download Python 3.11 from the official website: https://www.python.org/downloads/
   - IMPORTANT: During installation, make sure to check the box that says "Add python.exe to PATH".

2. Install FFmpeg:
   - Open PowerShell as Administrator (Right-click Start menu, choose "Terminal (Admin)" or "PowerShell (Admin)").
   - Run:
     ```powershell
     winget install Gyan.FFmpeg
     ```
     (Alternative using Chocolatey: `choco install ffmpeg -y`)
   - Close and reopen your terminal, then verify:
     ```powershell
     ffmpeg -version
     ```

3. Clone the repository:
   - In PowerShell or Command Prompt:
     ```powershell
     git clone https://github.com/dennyhacks/GoalGraph.git
     cd GoalGraph
     ```

4. Enable script execution (Required on Windows if PowerShell blocks virtual environments):
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```

5. Create and activate the virtual environment:
   - If using PowerShell:
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - If using standard Command Prompt (cmd.exe):
     ```cmd
     python -m venv .venv
     .venv\Scripts\activate.bat
     ```

6. Install the dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -e .
   ```

7. Run tests to confirm installation:
   ```powershell
   pytest tests/ -v
   ```

8. Start the application:
   ```powershell
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

---

### Option C: Setup on Linux (Ubuntu / Debian / Linux Mint)

1. Open your terminal.

2. Update system packages and install prerequisites:
   ```bash
   sudo apt update
   sudo apt install -y python3 python3-pip python3-venv ffmpeg git
   ```

3. Clone the repository and enter the directory:
   ```bash
   git clone https://github.com/dennyhacks/GoalGraph.git
   cd GoalGraph
   ```

4. Create and activate the virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install the Python dependencies:
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

6. Run automated test suite:
   ```bash
   pytest tests/ -v
   ```

7. Launch the interface:
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

---

## How to Use the Application

Once `streamlit run app.py` is running:

1. Select a Match:
   In the left sidebar, choose a match:
   - `videoplayback` (France vs. Belgium broadcast)
   - `manutd_vs_arsenal_2015` (Premier League broadcast)
   - `demo_match` (Multi-event synthetic benchmark match)
   - Or upload your own video file directly via the file uploader.

2. Match Header:
   The top banner displays the verified score, team names, match status, and a summary breakdown.

3. Ask Questions (Query Section):
   - Click any 1-Click Fast Query button (such as "First Goal", "Equalizer", "Goalkeeper Saves", or "Disciplinary Cards").
   - Or type a custom question into the text box (for example: "When was the first goal scored?" or "Who scored for France?").

4. Inspect the Video Evidence:
   GoalGraph generates a synchronized 60-second video clip for the requested event. You can play, pause, and scrub through the video directly in your browser.

5. Review the Uncertainty Timeline:
   The timeline chart plots every detected event across the match. Each event shows its exact timestamp, its confidence score, and error margin.

6. Tactical Pitch Radar:
   Examine the 2D pitch view to see estimated player and ball positions on the pitch at key event moments.

7. Technical Topology Graph (Optional):
   Expand the technical causal graph section to inspect the chronological sequence connecting preceding actions to their outcomes.

---

## Technical Stack & Libraries Used

| Component | Library / Tool | What It Does |
| :--- | :--- | :--- |
| Programming Language | Python 3.10+ | Core language for all pipeline scripts and logic |
| Deep Learning Runtime | PyTorch (`torch`) | Neural network execution |
| Object Detection | `ultralytics` (YOLOv8) | Detects players, referees, and the ball |
| Player Tracking | `ByteTrack` / `lap` | Keeps player IDs consistent across camera cuts |
| Audio Demuxing | `ffmpeg` | Extracts clean 16 kHz audio tracks from video files |
| Acoustic Signal Filtering | `scipy.signal`, `numpy.fft` | Bandpass filter (2.5–4.5 kHz) and STFT for referee whistles |
| Speech-to-Text | `faster-whisper`, `ctranslate2` | Transcribes commentary audio with word-level timestamps |
| Scoreboard OCR | `easyocr` (CRAFT + ResNet/LSTM) | Reads score numbers with strict monotonic (+1) constraints |
| Computer Vision Utilities | `opencv-python` (`cv2`) | Video slicing, frame reading, and replay banner color detection |
| Causal Event Graph | `networkx` | Stores events as nodes and causal relationships as edges |
| Query Grammar Parser | `lark` | Parses search queries into structured graph lookups |
| Web Interface | `streamlit` | Runs the interactive browser dashboard |
| Visualizations | `plotly` | Renders interactive pitch radars, timelines, and graphs |
| Unit Testing | `pytest` | Validates that math, logic, and parsing remain correct |

---

## Running from the Command Line (Without UI)

If you want to run the pipeline or evaluate accuracy purely from the terminal:

1. Generate a synthetic benchmark video with known ground truth:
   ```bash
   python scripts/make_demo_video.py
   ```

2. Run the analysis pipeline on a video file:
   ```bash
   python scripts/run_demo.py --video data/demo/demo_match.mp4
   ```

3. Run benchmark evaluation against ground truth:
   ```bash
   python scripts/evaluate.py
   ```

---

## Repository Structure

```
GoalGraph/
├── app.py                      # Interactive Streamlit workstation dashboard
├── pyproject.toml              # Project dependencies and build configuration
├── requirements.txt            # Python package requirements
├── README.md                   # Setup guide and documentation
├── HANDOFF.md                  # Detailed architectural and benchmark reference
├── yolov8n.pt                  # YOLOv8 object detection weights
├── goalgraph/                  # Main Python package
│   ├── audio/                  # Audio extraction, whistle filter, Whisper transcription
│   │   ├── extractor.py
│   │   ├── transcriber.py
│   │   ├── whistle.py
│   │   └── keywords.py
│   ├── scoreboard/             # Scoreboard ROI tracking and OCR
│   │   ├── tracker.py
│   │   └── ocr.py
│   ├── visual/                 # YOLO tracking, kit colors, replay cut detection
│   │   ├── detector.py
│   │   ├── kit_segmenter.py
│   │   ├── cut_detector.py
│   │   └── keyframe_annotator.py
│   ├── fusion/                 # Maximum Likelihood Estimation and delay compensation
│   │   ├── calibrator.py
│   │   ├── replay_filter.py
│   │   ├── mle.py
│   │   └── engine.py
│   ├── graph/                  # NetworkX Directed Acyclic Causal Graph
│   │   ├── causal_graph.py
│   │   └── builder.py
│   ├── query/                  # Query grammar (Lark) and execution engine
│   │   ├── parser.py
│   │   ├── tql.py
│   │   └── engine.py
│   ├── clip_slicer.py          # FFmpeg 60-second evidence clip generator
│   ├── narrative.py            # Natural language match summary builder
│   └── pipeline.py             # End-to-end multimodal pipeline runner
├── data/                       # Match videos and roster JSON files
├── scripts/                    # Command-line utility and evaluation scripts
└── tests/                      # Automated test suite
```

---

## Troubleshooting Common Issues

1. "ffmpeg: command not found" or "FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'"
   - Cause: FFmpeg is not installed or not in your system's PATH.
   - Solution: Follow Step 2 or 3 of the installation guide for your OS (`brew install ffmpeg` on Mac, `winget install Gyan.FFmpeg` on Windows, `sudo apt install ffmpeg` on Linux). After installing, restart your terminal.

2. Windows: "cannot be loaded because running scripts is disabled on this system"
   - Cause: Windows PowerShell has a security policy that disables running unsigned scripts by default.
   - Solution: In PowerShell, run:
     ```powershell
     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
     ```
     Then re-run `.venv\Scripts\Activate.ps1`.

3. "Port 8501 is already in use"
   - Cause: Another instance of Streamlit is already running.
   - Solution: Run on a different port:
     ```bash
     streamlit run app.py --server.port 8502
     ```
     Or terminate the existing process.

4. "Python version incompatible"
   - Cause: Using Python 3.9 or older, or Python 3.13 (some deep learning packages have not yet released wheels for 3.13).
   - Solution: Use Python 3.10, 3.11, or 3.12.

---

## Verification and Testing

To verify that all components are functioning as expected, run:
```bash
pytest tests/ -v
```

All 9 test suites verify:
- Whistle acoustic detection and bandpass filtering
- Scoreboard OCR monotonic digit updates
- Sensor fusion (MLE) and timestamp confidence intervals
- Replay deduplication and slow-motion rejection
- Directed causal graph edge construction
- Query language (TQL) grammar compilation
- Event keyframe annotation and HUD overlays

---

## License

This project is licensed under the MIT License.
