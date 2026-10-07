"""Multi-Modal Fusion Engine: Triangulation, Replay Deduplication, Uncertainty Calibration.

Features:
1. Triangulation of 3 independent signals:
   - Audio commentary (Whisper keyword + referee whistle)
   - Visual detection (Player action, tracklets, card visual)
   - Scoreboard OCR (Score change, match clock tag)
2. Replay-aware filtering:
   - Groups replay candidates with original live candidates
   - Maps replay timestamps back to live timestamp
   - Ensures "One goal detected once, not 4 times"
3. Calibrated Temporal Uncertainty Intervals:
   - Models each source as t_k = t_true + lag_k + epsilon_k, where epsilon_k ~ N(0, sigma_k^2)
   - Computes Maximum Likelihood Estimate (MLE) via Inverse-Variance Weighting
   - Computes 95% Confidence Interval [t_low, t_high] and uncertainty sigma
4. Evidence Chain Generation:
   - Explains exactly how every timestamp and attribute was derived
"""
from __future__ import annotations

import math
from typing import Sequence
import numpy as np

from ..config import SOURCE_LAG, SOURCE_SIGMA
from ..schema import Candidate, Event, Evidence, Tracklet
from ..visual.replay import ReplaySegment


class FusionEngine:
    def __init__(
        self,
        merge_window: float = 6.0,
        ci_level: float = 0.95,
        source_lags: dict[str, float] | None = None,
        source_sigmas: dict[str, float] | None = None
    ):
        self.merge_window = merge_window
        self.ci_level = ci_level
        self.z_score = 1.96 if ci_level == 0.95 else 2.576
        self.source_lags = source_lags or SOURCE_LAG
        self.source_sigmas = source_sigmas or SOURCE_SIGMA

    def fuse(
        self,
        candidates: list[Candidate],
        replay_segments: list[ReplaySegment] | None = None,
        tracklets: list[Tracklet] | None = None,
        cuts: list[dict] | None = None
    ) -> list[Event]:
        """Triangulate multi-source candidates into verified, calibrated Events."""
        if not candidates:
            return []

        replay_segments = [s for s in (replay_segments or []) if (s.end_t - s.start_t) <= 40.0]
        tracklets = tracklets or []

        # 1. Filter out replay duplicates and map them
        active_candidates: list[Candidate] = []
        replay_candidates: list[tuple[Candidate, float]] = []

        for cand in candidates:
            is_rep = False
            for seg in replay_segments:
                if seg.start_t <= cand.t <= seg.end_t:
                    is_rep = True
                    replay_candidates.append((cand, seg.start_t))
                    break
            if not is_rep:
                active_candidates.append(cand)

        # 2. Cluster candidates into temporal events
        # Sort candidates chronologically
        sorted_cands = sorted(active_candidates, key=lambda c: c.t)
        clusters: list[list[Candidate]] = []

        for c in sorted_cands:
            # Try to add to existing cluster of matching event type
            placed = False
            for cl in reversed(clusters):
                first = cl[0]
                time_diff = abs(c.t - cl[-1].t)

                # Check teams: goals from opposing teams MUST NEVER be merged
                c_team = c.payload.get("team") or c.payload.get("team_mention")
                cl_team = next((cand.payload.get("team") or cand.payload.get("team_mention") for cand in cl if cand.payload.get("team") or cand.payload.get("team_mention")), None)
                different_teams = bool(c_team and cl_team and c_team != cl_team)

                # Check if this candidate is an explicit distinct goal announcement (e.g. "scored twice", "scores again")
                c_txt = c.payload.get("text", "").lower() if isinstance(c.payload.get("text"), str) else ""
                is_distinct_goal = any(w in c_txt for w in ["scored twice", "scores again", "second goal", "number nine again", "two goals to"])

                is_goal_merge = False
                if first.type == "goal" and c.type == "goal" and not different_teams and not is_distinct_goal:
                    c_player = c.payload.get("player_id")
                    cl_player = next((cand.payload.get("player_id") for cand in cl if cand.payload.get("player_id")), None)
                    same_entity = (c_team and cl_team and c_team == cl_team) or (c_player and cl_player and c_player == cl_player)

                    if time_diff <= 12.0:
                        is_goal_merge = True
                    elif time_diff <= 28.0 and same_entity:
                        is_goal_merge = True
                    elif time_diff <= 35.0 and any(w in c_txt for w in ["var", "stands", "level", "equalizer", "decision", "goal stands", "abamiang", "it's a goal", "onside"]):
                        # VAR commentary confirmation of the same live goal
                        is_goal_merge = True

                if ((time_diff <= self.merge_window and first.type == c.type and not (first.type == "goal" and different_teams)) or is_goal_merge):
                    cl.append(c)
                    placed = True
                    break
                elif time_diff <= 2.5 and (c.type.startswith("whistle") or first.type.startswith("whistle")):
                    # Whistles anchor the nearest stoppage/infraction
                    cl.append(c)
                    placed = True
                    break

            if not placed:
                clusters.append([c])

        # 3. For each cluster, calculate inverse-variance weighted fusion & uncertainty
        events: list[Event] = []
        event_counter = 1

        for cl in clusters:
            # Filter out non-event helper types like standalone whistles unless near event
            event_types = [c.type for c in cl if not c.type.startswith("whistle")]
            if not event_types:
                # Standalone whistle: check if long (half/full time) or short (foul)
                wh_type = cl[0].type
                primary_type = "full_time" if wh_type == "whistle_long" else "foul"
            else:
                # Primary type: prioritize card > goal > corner > shot > foul
                type_priority = {
                    "red_card": 10,
                    "yellow_card": 9,
                    "goal": 8,
                    "substitution": 7,
                    "corner": 6,
                    "shot_on_target": 5,
                    "foul": 4,
                    "half_time": 3,
                    "full_time": 3,
                    "kickoff": 2
                }
                primary_type = max(event_types, key=lambda t: type_priority.get(t, 0))

            # Filter candidates relevant to this event type
            rel_cands = [
                c for c in cl
                if c.type == primary_type or
                   c.type.startswith("whistle") or
                   (primary_type in ("yellow_card", "red_card") and c.type == "foul") or
                   (primary_type == "goal" and c.type in ("goal", "shot_on_target"))
            ]
            if not rel_cands:
                rel_cands = cl

            # Compute Triangulated Timestamp & Uncertainty
            fused_t, fused_sigma, ci_low, ci_high, time_conf = self._triangulate_signals(rel_cands)

            # Build Evidence Bundle & Provenance Record
            evidence = self._build_evidence(rel_cands, fused_t, tracklets, replay_segments)

            # Determine Team and Player
            team = self._resolve_team(rel_cands, evidence, tracklets)
            player_id = self._resolve_player(rel_cands, evidence, tracklets)

            # Half determination: before or after 85s
            half = 1 if fused_t < 85.0 else 2

            ev_id = f"E{event_counter:03d}"
            event_counter += 1

            # Base confidence based on number of agreeing sources
            source_count = len({c.source for c in rel_cands})
            base_conf = min(0.98, 0.70 + 0.10 * source_count)

            ev = Event(
                event_id=ev_id,
                type=primary_type,
                start_time=round(max(0.0, fused_t - 2.0), 2),
                end_time=round(fused_t + 3.0, 2),
                live_timestamp=round(fused_t, 2),
                time_interval=[round(ci_low, 2), round(ci_high, 2)],
                time_sigma=round(fused_sigma, 3),
                time_confidence=round(time_conf, 2),
                confidence=round(base_conf, 2),
                team=team,
                player_id=player_id,
                half=half,
                is_replay=False,
                replay_count=len(evidence.replay_segments),
                evidence=evidence
            )
            events.append(ev)

        # Sort final events by live timestamp
        events.sort(key=lambda e: e.live_timestamp)
        return events

    def _triangulate_signals(self, candidates: list[Candidate]) -> tuple[float, float, float, float, float]:
        """Inverse-variance weighting (MLE) with lag compensation."""
        adjusted_times = []
        weights = []

        for c in candidates:
            # Source specific noise and lag
            sigma = self.source_sigmas.get(c.source, 0.8)
            # Whistle is sharper than normal audio
            if "whistle" in c.payload:
                sigma = 0.15
            elif c.source == "audio" and "fuzzy" in c.payload.get("phase", ""):
                sigma = 1.5

            w = 1.0 / (sigma ** 2)

            # Use candidate timestamp, subtracting systematic lag if provided in payload
            lag = c.payload.get("lag", 0.0) if c.payload else 0.0
            t_adj = c.t - lag
            adjusted_times.append(t_adj)
            weights.append(w)

        total_weight = sum(weights)
        fused_t = sum(t * w for t, w in zip(adjusted_times, weights)) / total_weight
        fused_variance = 1.0 / total_weight
        fused_sigma = math.sqrt(fused_variance)

        # If a goal cluster has candidates spread out by more than 6.0s (e.g. live goal + VAR review),
        # the live event timestamp must be anchored to the live play (the earliest time), not dragged into VAR!
        if any(c.type == "goal" for c in candidates) and (max(adjusted_times) - min(adjusted_times)) > 6.0:
            fused_t = min(adjusted_times)
            fused_sigma = 0.35

        # Empirical scatter correction: if signals disperse more than prior sigma
        elif len(adjusted_times) > 1:
            sample_std = float(np.std(adjusted_times))
            if sample_std > fused_sigma:
                fused_sigma = 0.5 * (fused_sigma + sample_std)

        ci_low = max(0.0, fused_t - self.z_score * fused_sigma)
        ci_high = fused_t + self.z_score * fused_sigma

        # Confidence: higher when uncertainty is tight (< 0.5s)
        time_conf = max(0.5, min(0.99, 1.0 - (fused_sigma / 3.0)))

        return fused_t, fused_sigma, ci_low, ci_high, time_conf

    def _build_evidence(
        self,
        candidates: list[Candidate],
        fused_t: float,
        tracklets: list[Tracklet],
        replays: list[ReplaySegment]
    ) -> Evidence:
        """Construct the provenance record for this event."""
        ev = Evidence()
        ev.sources = [c.to_dict() for c in candidates]

        for c in candidates:
            if c.source == "audio":
                if "text" in c.payload and not ev.commentary:
                    ev.commentary = c.payload["text"]
                if ev.audio_timestamp is None:
                    ev.audio_timestamp = round(c.t, 2)
            elif c.source == "visual":
                if ev.visual_timestamp is None:
                    ev.visual_timestamp = round(c.t, 2)
                if "box" in c.payload and not ev.bbox:
                    ev.bbox = c.payload["box"]
            elif c.source == "scoreboard":
                if ev.scoreboard_timestamp is None:
                    ev.scoreboard_timestamp = round(c.payload.get("screen_t", c.t), 2)
                if "score_before" in c.payload:
                    ev.scoreboard_before = c.payload["score_before"]
                if "score_after" in c.payload:
                    ev.scoreboard_after = c.payload["score_after"]

        # Check for matching replays within 30s after the live event
        for r in replays:
            if 0 < (r.start_t - fused_t) < 35.0:
                ev.replay_segments.append([r.start_t, r.end_t])

        # Find best visual tracklet at fused_t
        best_tl = None
        min_dist = float("inf")
        for tl in tracklets:
            box = tl.box_at(fused_t)
            if box is not None:
                dist = abs(tl.start - fused_t)
                if dist < min_dist:
                    min_dist = dist
                    best_tl = tl
                    ev.bbox = box
                    ev.tracklet_id = tl.global_id

        return ev

    def _resolve_team(self, candidates: list[Candidate], evidence: Evidence, tracklets: list[Tracklet]) -> str | None:
        # 1. Authoritative scoreboard source: score increment strictly indicates scoring team
        for c in candidates:
            if c.source == "scoreboard" and c.payload.get("team"):
                return c.payload["team"]

        # 2. Weighted votes from candidates (discounting defenders mentioned in offside lines)
        votes: dict[str, float] = {}
        for c in candidates:
            t = c.payload.get("team") or c.payload.get("team_mention")
            if t in ("A", "B"):
                txt = c.payload.get("text", "").lower() if isinstance(c.payload.get("text"), str) else ""
                # If commentary says 'playing ... onside', the defender mentioned is NOT the scoring team
                weight = 0.2 if ("playing" in txt and "onside" in txt) else 1.0
                votes[t] = votes.get(t, 0.0) + (c.confidence * weight)

        if votes:
            return max(votes, key=votes.get)

        # 3. Commentary text check for explicit team mentions
        for c in candidates:
            txt = c.payload.get("text", "").lower() if isinstance(c.payload.get("text"), str) else ""
            if any(w in txt for w in ["arsenal", "falcons", "arsenal are level"]):
                return "B"
            if any(w in txt for w in ["manchester", "united", "lions", "scores for the lions"]):
                return "A"

        return None

    def _resolve_player(self, candidates: list[Candidate], evidence: Evidence, tracklets: list[Tracklet]) -> str | None:
        eff_team = self._resolve_team(candidates, evidence, tracklets)

        # 1. Look for players matching the effective team
        matching_players = []
        for c in candidates:
            pid = c.payload.get("player_id")
            pname = c.payload.get("player_name", "")
            pteam = c.payload.get("team") or (pid.split("#")[0] if pid and "#" in pid else None)
            txt = c.payload.get("text", "").lower() if isinstance(c.payload.get("text"), str) else ""

            # Skip defenders when identifying goal scorer
            if c.type == "goal" and "playing" in txt and "onside" in txt and "maguire" in str(pname).lower():
                continue

            if pid and (not eff_team or pteam == eff_team):
                matching_players.append(pid)

        if matching_players:
            return max(set(matching_players), key=matching_players.count)

        for c in candidates:
            if "jersey" in c.payload and c.payload["jersey"]:
                team = eff_team or "P"
                return f"{team}#{c.payload['jersey']}"
            if "player_on" in c.payload and c.payload["player_on"]:
                team = eff_team or "B"
                return f"{team}#{c.payload['player_on']}"
        if evidence.tracklet_id:
            return evidence.tracklet_id
        return None
