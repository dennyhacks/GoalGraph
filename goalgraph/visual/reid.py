"""Occlusion-robust Re-Identification (ReID) across cuts, occlusions and camera zooms.

Features used for ReID:
1. Multi-level appearance embedding: HSV color histogram of torso + shorts
2. Team association + position / jersey number signature (OCR / template)
3. Smooth kinematic motion prior before and after cut / occlusion
4. Cosine similarity matching between tracklet embeddings to merge identities
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from ..schema import Tracklet


def cosine_sim(v1: Sequence[float], v2: Sequence[float]) -> float:
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class ReIDMatcher:
    def __init__(self, sim_threshold: float = 0.75, max_gap_seconds: float = 20.0):
        self.sim_threshold = sim_threshold
        self.max_gap_seconds = max_gap_seconds

    def match_and_merge(self, tracklets: list[Tracklet], embeddings: dict[str, list[float]]) -> list[Tracklet]:
        """Group tracklets that share the same global identity across camera cuts and occlusions."""
        if not tracklets:
            return []

        # Sort tracklets chronologically
        sorted_tracks = sorted(tracklets, key=lambda t: t.start)
        global_map: dict[str, str] = {}  # local tracklet_id -> global_id
        cluster_reps: dict[str, list[float]] = {}  # global_id -> running avg embedding
        cluster_teams: dict[str, str | None] = {}
        cluster_jerseys: dict[str, str | None] = {}
        cluster_last_seen: dict[str, float] = {}

        for tr in sorted_tracks:
            tid = tr.tracklet_id
            emb = embeddings.get(tid)
            jersey = tr.jersey_number
            team = tr.team

            best_gid = None
            best_sim = -1.0

            # Match against existing clusters
            for gid, rep_emb in cluster_reps.items():
                # Constraint 1: same team (or unknown)
                g_team = cluster_teams.get(gid)
                if team is not None and g_team is not None and team != g_team:
                    continue

                # Constraint 2: jersey number matches exactly if both known
                g_jersey = cluster_jerseys.get(gid)
                if jersey is not None and g_jersey is not None:
                    if jersey == g_jersey:
                        best_gid = gid
                        best_sim = 1.0
                        break
                    else:
                        continue

                # Constraint 3: temporal gap within max threshold
                last_t = cluster_last_seen.get(gid, 0.0)
                if (tr.start - last_t) > self.max_gap_seconds:
                    continue

                # Visual feature similarity
                if emb is not None and rep_emb is not None:
                    sim = cosine_sim(emb, rep_emb)
                    if sim > self.sim_threshold and sim > best_sim:
                        best_sim = sim
                        best_gid = gid

            if best_gid is None:
                # Create new global identity
                if jersey and team:
                    new_gid = f"P_{team}_{jersey}"
                elif jersey:
                    new_gid = f"P_{jersey}"
                else:
                    new_gid = f"P_{team or 'X'}_{len(cluster_reps) + 1:02d}"

                best_gid = new_gid
                if emb is not None:
                    cluster_reps[best_gid] = list(emb)
                cluster_teams[best_gid] = team
                cluster_jerseys[best_gid] = jersey
            else:
                # Update cluster representation
                if emb is not None and best_gid in cluster_reps:
                    old_emb = np.array(cluster_reps[best_gid])
                    new_emb = np.array(emb)
                    cluster_reps[best_gid] = (0.7 * old_emb + 0.3 * new_emb).tolist()
                if jersey and not cluster_jerseys.get(best_gid):
                    cluster_jerseys[best_gid] = jersey
                if team and not cluster_teams.get(best_gid):
                    cluster_teams[best_gid] = team

            cluster_last_seen[best_gid] = tr.end
            tr.global_id = best_gid

        return sorted_tracks
