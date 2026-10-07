"""Causal Temporal Event Graph (NetworkX).

Novel Feature #6:
Builds a directed provenance graph containing:
Nodes:
- Event nodes (goal, corner, foul, card, etc.)
- Player nodes (e.g. A#9, B#4)
- Team nodes (Lions, Falcons)

Edges:
- BEFORE / AFTER: Chronological sequence
- CAUSED_BY / LEADS_TO: Causal dependency (e.g. corner -> shot -> goal; foul -> yellow_card; foul -> red_card)
- SAME_PLAYER: Relates actions performed by the same tracked identity
- INVOLVES: Event -> Player / Team
"""
from __future__ import annotations

import networkx as nx
from ..schema import Event


class EventGraphBuilder:
    def __init__(self):
        pass

    def build_graph(self, events: list[Event], team_names: dict[str, str] | None = None) -> nx.DiGraph:
        """Construct the causal temporal event graph from fused events."""
        G = nx.DiGraph()
        team_names = team_names or {"A": "Lions", "B": "Falcons"}

        # 1. Add Team Nodes
        for t_code, t_name in team_names.items():
            G.add_node(f"team_{t_code}", node_type="team", code=t_code, name=t_name)

        # 2. Add Player & Event Nodes
        players_seen = set()

        for i, ev in enumerate(events):
            # Event node
            G.add_node(
                ev.event_id,
                node_type="event",
                event_type=ev.type,
                live_timestamp=ev.live_timestamp,
                start_time=ev.start_time,
                end_time=ev.end_time,
                time_interval=ev.time_interval,
                time_sigma=ev.time_sigma,
                confidence=ev.confidence,
                half=ev.half,
                team=ev.team,
                player_id=ev.player_id,
                is_replay=ev.is_replay,
                replay_count=ev.replay_count,
                commentary=ev.evidence.commentary or "",
                bbox=ev.evidence.bbox
            )

            # Team edge
            if ev.team:
                G.add_edge(ev.event_id, f"team_{ev.team}", relation="FOR_TEAM")

            # Player node & edge
            if ev.player_id:
                p_node = f"player_{ev.player_id}"
                if p_node not in players_seen:
                    G.add_node(p_node, node_type="player", player_id=ev.player_id, team=ev.team)
                    players_seen.add(p_node)
                    if ev.team:
                        G.add_edge(p_node, f"team_{ev.team}", relation="PLAYS_FOR")

                G.add_edge(ev.event_id, p_node, relation="BY_PLAYER")

            # 3. Temporal Ordering Edges: BEFORE / AFTER
            if i > 0:
                prev_ev = events[i - 1]
                delta_t = round(ev.live_timestamp - prev_ev.live_timestamp, 2)
                G.add_edge(prev_ev.event_id, ev.event_id, relation="BEFORE", delta_seconds=delta_t)
                G.add_edge(ev.event_id, prev_ev.event_id, relation="AFTER", delta_seconds=delta_t)

        # 4. Same Player Edges across events
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                e1, e2 = events[i], events[j]
                if e1.player_id and e2.player_id and e1.player_id == e2.player_id:
                    G.add_edge(e1.event_id, e2.event_id, relation="SAME_PLAYER", player=e1.player_id)

        # 5. Causal Edges: CAUSED_BY / LEADS_TO
        # Causal rules in football:
        # - Corner -(within 15s)-> Shot -(within 5s)-> Goal
        # - Corner -(within 15s)-> Goal
        # - Foul -(within 12s)-> Yellow Card / Red Card
        for i, ev in enumerate(events):
            # Check upcoming events within causal window
            for j in range(i + 1, len(events)):
                next_ev = events[j]
                delta = next_ev.live_timestamp - ev.live_timestamp
                if delta < 0:
                    continue
                if delta > 16.0:  # beyond causal horizon
                    break

                # Rule A: Foul -> Card
                if ev.type == "foul" and next_ev.type in ("yellow_card", "red_card") and delta <= 12.0:
                    G.add_edge(ev.event_id, next_ev.event_id, relation="LEADS_TO", cause="foul_sanction", delta_s=delta)
                    G.add_edge(next_ev.event_id, ev.event_id, relation="CAUSED_BY", cause="foul_sanction", delta_s=delta)

                # Rule B: Corner -> Goal / Shot
                if ev.type == "corner" and next_ev.type in ("goal", "shot_on_target") and delta <= 15.0:
                    G.add_edge(ev.event_id, next_ev.event_id, relation="LEADS_TO", cause="set_piece", delta_s=delta)
                    G.add_edge(next_ev.event_id, ev.event_id, relation="CAUSED_BY", cause="set_piece", delta_s=delta)

                # Rule C: Shot -> Goal
                if ev.type == "shot_on_target" and next_ev.type == "goal" and delta <= 4.0:
                    G.add_edge(ev.event_id, next_ev.event_id, relation="LEADS_TO", cause="shot_scored", delta_s=delta)
                    G.add_edge(next_ev.event_id, ev.event_id, relation="CAUSED_BY", cause="shot_scored", delta_s=delta)

        return G
