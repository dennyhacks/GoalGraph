"""Unit & integration test suite for GoalGraph."""
from __future__ import annotations

import pytest
from pathlib import Path

from goalgraph.schema import Candidate, Event, Evidence, Tracklet
from goalgraph.audio.keywords import spot
from goalgraph.audio.whistle import whistle_candidates
from goalgraph.visual.replay import ReplayDetector, ReplaySegment
from goalgraph.visual.reid import ReIDMatcher, cosine_sim
from goalgraph.fusion.engine import FusionEngine
from goalgraph.graph.builder import EventGraphBuilder
from goalgraph.query.tql import parse_tql
from goalgraph.query.engine import QueryEngine


def test_tql_parser():
    q1 = parse_tql("FIND goal AFTER corner WITHIN 10s")
    assert q1.action == "FIND"
    assert q1.target_event == "goal"
    assert q1.temporal_op == "AFTER"
    assert q1.reference_event == "corner"
    assert q1.within_seconds == 10.0

    q2 = parse_tql("COUNT corner IN second_half")
    assert q2.action == "COUNT"
    assert q2.target_event == "corner"
    assert q2.half == 2

    q3 = parse_tql("WHEN yellow_card BY P#4")
    assert q3.action == "WHEN"
    assert q3.target_event == "yellow_card"
    assert q3.player == "P#4"

    q4 = parse_tql("DID corner LEAD_TO goal WITHIN 10s")
    assert q4.action == "DID_LEAD_TO"
    assert q4.target_event == "corner"
    assert q4.reference_event == "goal"
    assert q4.within_seconds == 10.0


def test_keyword_spotting():
    sentences = [
        {"start": 1.0, "end": 3.0, "text": "And we're underway with the kick off.", "words": []},
        {"start": 18.0, "end": 20.0, "text": "Oh that is a foul, number four brings him down.", "words": []},
        {"start": 23.0, "end": 25.0, "text": "The referee shows a yellow card to number four.", "words": []},
        {"start": 39.0, "end": 42.0, "text": "Goal! Number nine scores for the Lions!", "words": []},
        {"start": 51.0, "end": 54.0, "text": "Let's see that goal again.", "words": []},
    ]
    roster = {"teams": {"A": {"name": "Lions", "code": "LIO"}}}
    cands = spot(sentences, roster=roster)
    types = [c.type for c in cands]

    assert "kickoff" in types
    assert "foul" in types
    assert "yellow_card" in types
    assert "goal" in types
    assert "replay_cue" in types  # Replay phrase detected as replay cue, NOT a new goal!


def test_fusion_triangulation_and_uncertainty():
    # Triangulate 3 independent signals for a goal
    cands = [
        Candidate(source="visual", type="goal", t=39.8, confidence=0.85, payload={"lag": 0.0}),
        Candidate(source="audio", type="goal", t=40.8, confidence=0.80, payload={"lag": 1.0}),
        Candidate(source="scoreboard", type="goal", t=41.5, confidence=0.90, payload={"lag": 1.7, "score_after": "1-0"}),
    ]
    fe = FusionEngine(merge_window=5.0)
    events = fe.fuse(cands)
    assert len(events) == 1
    ev = events[0]
    assert ev.type == "goal"
    # Point estimate should be tightly centered around 39.8s
    assert 39.6 <= ev.live_timestamp <= 40.0
    # Uncertainty interval should bracket 39.8s with small sigma
    assert ev.time_interval[0] <= 39.8 <= ev.time_interval[1]
    assert ev.time_sigma < 0.6
    assert ev.confidence >= 0.90


def test_replay_deduplication():
    # 1 live goal at 39.8s and 1 replay candidate at 52.0s
    cands = [
        Candidate(source="audio", type="goal", t=40.8, payload={"lag": 1.0}),
        Candidate(source="visual", type="goal", t=52.0, payload={"box": [10, 10, 20, 20]}),
    ]
    replay_segs = [ReplaySegment(start_t=48.0, end_t=60.0, reason="replay_badge")]
    fe = FusionEngine()
    events = fe.fuse(cands, replay_segments=replay_segs)
    # Deduplicated: only 1 goal event should be created, NOT 2!
    assert len(events) == 1
    assert events[0].live_timestamp < 45.0


def test_causal_graph_construction():
    e1 = Event(
        event_id="E1", type="corner", start_time=35.0, end_time=38.0,
        live_timestamp=37.0, time_interval=[36.5, 37.5], time_sigma=0.2,
        time_confidence=0.9, confidence=0.9, team="A", player_id="A#7"
    )
    e2 = Event(
        event_id="E2", type="goal", start_time=39.0, end_time=42.0,
        live_timestamp=39.8, time_interval=[39.2, 40.4], time_sigma=0.3,
        time_confidence=0.9, confidence=0.95, team="A", player_id="A#9"
    )
    builder = EventGraphBuilder()
    G = builder.build_graph([e1, e2])
    # Check causal edge: corner -(LEADS_TO)-> goal
    assert G.has_edge("E1", "E2")
    edge_data = G.get_edge_data("E1", "E2")
    assert edge_data["relation"] == "LEADS_TO"
    assert edge_data["cause"] == "set_piece"


def test_occlusion_reid_matching():
    # Two tracklets separated by occlusion gap
    t1 = Tracklet(tracklet_id="T1", global_id="T1", team="A", frames=[30.0, 34.0], boxes=[[10, 10, 20, 20]], jersey_number="9")
    t2 = Tracklet(tracklet_id="T2", global_id="T2", team="A", frames=[38.0, 42.0], boxes=[[15, 12, 20, 20]], jersey_number="9")
    matcher = ReIDMatcher()
    emb = {"T1": [1.0, 0.0, 0.5], "T2": [0.95, 0.0, 0.52]}
    merged = matcher.match_and_merge([t1, t2], emb)
    # Same global ID preserved!
    assert merged[0].global_id == merged[1].global_id


def test_cv_keyframe_annotator():
    from goalgraph.visual.annotator import annotate_event_keyframe
    # Annotate a test frame from demo match
    res = annotate_event_keyframe(
        "data/demo/demo_match.mp4",
        timestamp=39.8,
        event_id="E_TEST",
        event_type="goal",
        player_label="A#9",
        confidence=0.95
    )
    assert res is not None
    assert Path(res).exists()


def test_vlm_reasoner():
    from goalgraph.vlm.reasoner import generate_vlm_audit
    audit = generate_vlm_audit(
        event_id="E007",
        event_type="goal",
        timestamp=171.24,
        player_name="Scott McTominay #39",
        team_name="Manchester United",
        opposing_team="Arsenal"
    )
    assert audit.confidence >= 0.90
    assert "Scott McTominay" in audit.visual_action_description
    assert len(audit.entities_detected) > 0
    assert audit.to_dict()["event_id"] == "E007"

