"""Temporal Query Language (TQL) formal grammar and parser using Lark.

Grammar supports:
FIND <event_type> [AFTER|BEFORE <event_type>] [WITHIN <num>s] [BY <player>] [IN <half>]
COUNT <event_type> [IN <half>]
WHEN <event_type> [BY <player>]
CAUSED_BY <event_type>
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from lark import Lark, Transformer, v_args

TQL_GRAMMAR = r"""
?start: query

query: find_query
     | count_query
     | when_query
     | causal_query

find_query: "FIND" event_type [temporal_clause] [within_clause] [player_clause] [half_clause]
count_query: "COUNT" event_type [half_clause]
when_query: "WHEN" event_type [player_clause]
causal_query: "DID" event_type ("LEAD_TO"|"LEAD TO"|"CAUSE") event_type [within_clause]

temporal_clause: ("AFTER" | "BEFORE") event_type
within_clause: "WITHIN" NUMBER ("s" | "sec" | "seconds")?
player_clause: "BY" (PLAYER | "same_player")
half_clause: "IN" HALF

event_type: CNAME
HALF: "first_half" | "second_half" | "1st_half" | "2nd_half" | "half_1" | "half_2"
PLAYER: /[A-Za-z0-9_#]+/

%import common.CNAME
%import common.NUMBER
%import common.WS
%ignore WS
"""


@dataclass
class TQLQuery:
    action: str                 # FIND | COUNT | WHEN | DID_LEAD_TO
    target_event: str
    temporal_op: str | None = None      # AFTER | BEFORE
    reference_event: str | None = None
    within_seconds: float | None = None
    player: str | None = None
    half: int | None = None     # 1 or 2


class TQLTransformer(Transformer):
    def event_type(self, items):
        return str(items[0]).lower()

    def NUMBER(self, n):
        return float(n)

    def PLAYER(self, p):
        return str(p)

    def HALF(self, h):
        h_str = str(h).lower()
        return 1 if ("first" in h_str or "1" in h_str) else 2

    def temporal_clause(self, items):
        return (str(items[0]).upper(), items[1])

    def within_clause(self, items):
        return float(items[0])

    def player_clause(self, items):
        return str(items[0])

    def half_clause(self, items):
        return int(items[0])

    def find_query(self, items):
        q = TQLQuery(action="FIND", target_event=items[0])
        for it in items[1:]:
            if isinstance(it, tuple) and it[0] in ("AFTER", "BEFORE"):
                q.temporal_op, q.reference_event = it
            elif isinstance(it, float):
                q.within_seconds = it
            elif isinstance(it, int):
                q.half = it
            elif isinstance(it, str):
                q.player = it
        return q

    def count_query(self, items):
        q = TQLQuery(action="COUNT", target_event=items[0])
        if len(items) > 1 and isinstance(items[1], int):
            q.half = items[1]
        return q

    def when_query(self, items):
        q = TQLQuery(action="WHEN", target_event=items[0])
        if len(items) > 1 and isinstance(items[1], str):
            q.player = items[1]
        return q

    def causal_query(self, items):
        q = TQLQuery(action="DID_LEAD_TO", target_event=items[0], reference_event=items[1])
        if len(items) > 2 and isinstance(items[2], float):
            q.within_seconds = items[2]
        return q

    def query(self, items):
        return items[0]


_PARSER = None


def parse_tql(text: str) -> TQLQuery:
    """Parse a TQL query string into structured TQLQuery object."""
    global _PARSER
    clean = text.strip()
    # Normalize common variations
    clean = re.sub(r"\s+", " ", clean)

    if _PARSER is None:
        _PARSER = Lark(TQL_GRAMMAR, parser="lalr", transformer=TQLTransformer())

    try:
        return _PARSER.parse(clean)
    except Exception as e:
        # Robust regex fallback for slightly off syntax
        return _fallback_parse(clean)


def _fallback_parse(text: str) -> TQLQuery:
    text_upper = text.upper()
    half = 1 if "FIRST_HALF" in text_upper or "FIRST HALF" in text_upper else (2 if "SECOND_HALF" in text_upper or "SECOND HALF" in text_upper else None)
    within_m = re.search(r"WITHIN\s+(\d+(?:\.\d+)?)\s*S?", text_upper)
    within_s = float(within_m.group(1)) if within_m else None
    by_m = re.search(r"BY\s+([A-Za-z0-9_#]+)", text, re.I)
    player = by_m.group(1) if by_m else None

    if text_upper.startswith("COUNT"):
        m = re.search(r"COUNT\s+([A-Za-z_]+)", text_upper)
        target = m.group(1).lower() if m else "event"
        return TQLQuery(action="COUNT", target_event=target, half=half)

    if text_upper.startswith("WHEN"):
        m = re.search(r"WHEN\s+([A-Za-z_]+)", text_upper)
        target = m.group(1).lower() if m else "goal"
        return TQLQuery(action="WHEN", target_event=target, player=player)

    if "LEAD" in text_upper or "CAUSE" in text_upper:
        m = re.search(r"(?:DID\s+)?([A-Za-z_]+)\s+(?:LEAD\s+TO|CAUSE)\s+(?:A\s+|AN\s+|THE\s+)?([A-Za-z_]+)", text_upper)
        if m:
            return TQLQuery(action="DID_LEAD_TO", target_event=m.group(1).lower(), reference_event=m.group(2).lower(), within_seconds=within_s)

    # Default FIND
    m = re.search(r"FIND\s+([A-Za-z_]+)(?:\s+(AFTER|BEFORE)\s+([A-Za-z_]+))?", text_upper)
    target = m.group(1).lower() if m else "goal"
    op = m.group(2) if m and m.group(2) else None
    ref = m.group(3).lower() if m and m.group(3) else None
    return TQLQuery(action="FIND", target_event=target, temporal_op=op, reference_event=ref, within_seconds=within_s, player=player, half=half)
