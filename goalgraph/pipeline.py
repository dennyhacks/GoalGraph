"""End-to-End GoalGraph Pipeline.

Executes:
1. Audio branch (transcription + keyword spotting + whistle detection)
2. Scoreboard branch (OCR score changes + clock)
3. Visual branch (cut detection, player/action detection, ReID tracking, replay detection)
4. Multi-modal fusion engine (triangulation, replay mapping, uncertainty intervals)
5. Causal event graph builder (NetworkX)
6. Serializes outputs to outputs/<video_name>/
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .audio.keywords import spot
from .audio.transcribe import transcribe
from .audio.whistle import detect_whistles, whistle_candidates
from .config import PipelineConfig
from .fusion.engine import FusionEngine
from .graph.builder import EventGraphBuilder
from .query.engine import QueryEngine
from .schema import Candidate, Event, Tracklet, dump_json, load_json
from .scoreboard.ocr import ScoreboardTracker, detect_scoreboard_events
from .video_io import extract_audio, iter_frames, probe
from .visual.cuts import detect_cuts
from .visual.detector import VisualDetector
from .visual.reid import ReIDMatcher
from .visual.replay import ReplayDetector
from .visual.tracker import MultiObjectTracker

log = logging.getLogger("goalgraph")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class GoalGraphPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.out_dir = Path(self.config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        video_path: str,
        roster_path: str | None = None,
        progress_callback: Callable[[int, int, str, float], None] | None = None
    ) -> tuple[list[Event], Any, QueryEngine]:
        """Run full GoalGraph pipeline on video with live stage reporting."""
        def report(stage: int, msg: str, pct: float):
            if progress_callback:
                progress_callback(stage, 5, msg, pct)

        video_p = Path(video_path)
        v_name = video_p.stem
        run_out = self.out_dir / v_name
        run_out.mkdir(parents=True, exist_ok=True)

        roster = None
        if roster_path and Path(roster_path).exists():
            roster = json.loads(Path(roster_path).read_text())
        elif (video_p.parent / "roster.json").exists():
            roster = json.loads((video_p.parent / "roster.json").read_text())

        all_candidates: list[Candidate] = []

        # -------------------------------------------------------------
        # 1. AUDIO BRANCH
        # -------------------------------------------------------------
        report(1, "Demuxing 16kHz audio stream & initializing Whisper speech-to-text...", 0.05)
        log.info("==> [Audio Branch] Extracting and transcribing commentary...")
        wav_path = str(run_out / "audio.wav")
        transcript_cache = str(run_out / "transcript.json")
        try:
            extract_audio(video_path, wav_path)
            report(1, "Transcribing commentary audio with word-level timestamps (Whisper)...", 0.12)
            sentences = transcribe(wav_path, cache=transcript_cache, model_size=self.config.whisper_model)
            report(1, "Scanning commentary for lexical keywords and referee whistles...", 0.18)
            audio_cands = spot(sentences, roster=roster)
            whistles = detect_whistles(wav_path)
            wh_cands = whistle_candidates(whistles)
            all_candidates.extend(audio_cands)
            all_candidates.extend(wh_cands)
            report(1, f"Audio branch completed: {len(audio_cands)} keyword cues, {len(whistles)} whistle anchors", 0.22)
            log.info("Audio produced %d keyword candidates and %d whistle anchors", len(audio_cands), len(whistles))
        except Exception as e:
            log.warning("Audio branch failed: %s", e)
            audio_cands = []

        # -------------------------------------------------------------
        # 2. SCOREBOARD BRANCH
        # -------------------------------------------------------------
        report(2, "Tracking scoreboard ROI and running monotonic digit OCR (EasyOCR)...", 0.25)
        log.info("==> [Scoreboard Branch] Tracking score & match clock...")
        sb_cache = str(run_out / "scoreboard.json")
        try:
            tracker = ScoreboardTracker(roi=self.config.scoreboard_roi, use_easyocr=True)
            sb_samples = tracker.process_video(video_path, fps=self.config.ocr_fps, cache_path=sb_cache)
            report(2, "Parsing score transitions and verifying clock progression...", 0.40)
            sb_cands = detect_scoreboard_events(sb_samples)
            all_candidates.extend(sb_cands)
            report(2, f"Scoreboard OCR completed: {len(sb_cands)} verified score changes", 0.45)
            log.info("Scoreboard branch produced %d candidates", len(sb_cands))
        except Exception as e:
            log.warning("Scoreboard branch failed: %s", e)
            sb_cands = []

        # -------------------------------------------------------------
        # 3. VISUAL BRANCH
        # -------------------------------------------------------------
        report(3, "Detecting camera shot transitions and slow-motion replay graphics...", 0.48)
        log.info("==> [Visual Branch] Detecting camera cuts, replays, and tracking players...")
        cuts = detect_cuts(video_path)
        log.info("Detected %d camera cuts / shot transitions", len(cuts))

        report(3, "Mapping replay intervals back to live broadcast moments...", 0.58)
        replay_detector = ReplayDetector()
        replay_segs = replay_detector.detect_segments(video_path, fps=2.0, audio_cues=audio_cands, cuts=cuts)
        log.info("Detected %d replay segments (will map replay timestamps to live timestamps)", len(replay_segs))
        report(3, f"Visual cuts & replays completed: {len(cuts)} transitions, {len(replay_segs)} replay intervals", 0.65)

        # Entity detection & tracking
        report(4, "Tracking players, referees, and kit colors (YOLOv8 + ReID)...", 0.68)
        detector = VisualDetector(use_yolo=(self.config.detector == "yolo"))
        tracker = MultiObjectTracker()

        frame_dets = []
        for t, frame in iter_frames(video_path, fps=self.config.analysis_fps):
            dets = detector.detect_frame(frame, t=t)
            frame_dets.append((t, dets))
            # If visual card detected, add candidate
            for d in dets:
                if d.label in ("card_yellow", "card_red"):
                    all_candidates.append(Candidate(
                        source="visual",
                        type="yellow_card" if d.label == "card_yellow" else "red_card",
                        t=t,
                        confidence=d.confidence,
                        payload={"box": d.box, "card_type": d.label}
                    ))

        report(4, "Linking persistent player tracklets across camera transitions...", 0.80)
        tracklets, embeddings = tracker.track(frame_dets)
        report(4, f"Player tracking completed: {len(tracklets)} unique identities mapped", 0.85)
        log.info("Tracked %d unique player tracklets with persistent ReID identities", len(tracklets))

        # -------------------------------------------------------------
        # 4. FUSION ENGINE
        # -------------------------------------------------------------
        report(5, "Triangulating audio, OCR, and CV signals with Maximum Likelihood Estimation (MLE)...", 0.88)
        log.info("==> [Fusion Engine] Triangulating 3 signals & computing calibrated uncertainty...")
        fusion = FusionEngine(merge_window=self.config.merge_window, ci_level=self.config.ci_level)
        fused_events = fusion.fuse(
            candidates=all_candidates,
            replay_segments=replay_segs,
            tracklets=tracklets,
            cuts=cuts
        )
        log.info("Fused %d high-confidence provenance-verified events", len(fused_events))

        # Save fused events JSON
        events_json = run_out / "events.json"
        dump_json([e.to_dict() for e in fused_events], events_json)

        # -------------------------------------------------------------
        # 5. CAUSAL EVENT GRAPH BUILDER
        # -------------------------------------------------------------
        report(5, "Constructing NetworkX causal temporal event graph & index...", 0.94)
        log.info("==> [Graph Builder] Constructing NetworkX causal temporal event graph...")
        builder = EventGraphBuilder()
        team_dict = roster.get("teams", {}) if roster else {}
        graph = builder.build_graph(fused_events, team_names={k: v.get("name", k) for k, v in team_dict.items()})

        # Save graph summary
        graph_data = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "nodes": list(graph.nodes(data=True)),
            "edges": [(u, v, d) for u, v, d in graph.edges(data=True)]
        }
        dump_json(graph_data, run_out / "event_graph.json")

        # -------------------------------------------------------------
        # 6. QUERY ENGINE
        # -------------------------------------------------------------
        engine = QueryEngine(fused_events, graph, video_path=video_path, out_dir=str(run_out))

        report(5, "Pipeline analysis complete. Synchronizing evidence clips...", 1.0)
        log.info("GoalGraph pipeline execution finished successfully!")
        return fused_events, graph, engine
