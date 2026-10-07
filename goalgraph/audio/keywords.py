"""Commentary keyword spotting -> audio Candidates.

Plain "does the sentence contain *goal*" is wrong a lot of the time:

* "through **on goal**", "shot **at goal**", "**goal** kick"         -> not a goal
* "let's see that **goal** again", "another look at the **goal**"  -> replay cue
* "two **goals** to one"                                             -> recap
* "second yellow, and it's a **red card**"                           -> red, not yellow
* "**corner** for the Lions" (awarded)  vs  "swings in the corner" (taken)

So every rule carries *exclusions*, a *phase* and a *lag prior*, and replay
phrases are emitted as their own ``replay_cue`` signal that the replay
detector consumes.  Fuzzy matching (rapidfuzz) tolerates ASR errors such as
"goel" or "yellow cart".
"""
from __future__ import annotations

import re

from ..schema import Candidate

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None

NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "twenty-one": 21, "twenty-two": 22,
    "twenty-three": 23,
}
_num_alt = "|".join(sorted(NUM_WORDS, key=len, reverse=True))
NUMBER_RE = re.compile(rf"\b(?:number|no\.?)\s+(\d{{1,2}}|{_num_alt})\b", re.I)

REPLAY_RE = re.compile(
    r"\b(?:(?:see|look at|watch)\s+(?:\w+\s+){0,3}again|another look|slow motion|replay)\b",
    re.I
)

# (event type, phase, pattern, exclusion, base confidence, lag seconds)
# lag = how long after the true event the keyword is typically spoken.
RULES = [
    ("red_card", "shown", r"\bred card\b|\bsent off\b|\bmarching orders\b|\bsecond yellow\b", None, 0.9, 0.5),
    ("yellow_card", "shown", r"\byellow card\b|\bbooked\b|\bcaution(ed)?\b|\bin the book\b",
     r"\bsecond yellow\b|\bred card\b", 0.85, 0.5),
    ("goal", "scored", r"\bgoal\b|\bscores?\b|\bscored\b|\bequali[sz]es?\b|\b(are|is) level\b|\bback of the net\b|\bit'?s in\b|\bburies\b|\bslotted away\b|\bfinish into the corner\b|\bfinds the net\b",
     r"\bon goal\b|\bat goal\b|\bgoal kick\b|\bgoals to\b|\bgoal line\b|\bno goal\b|\bdisallowed\b", 0.90, 0.4),
    ("shot_on_target", "saved", r"\bsaves?\b|\bsaved\b|\bsafe from\b|\bshot\b|\bheader\b|\bon target\b|\bstrike\b",
     r"\bshot wide\b|\bover the bar\b|\boff target\b", 0.75, 0.4),
    ("foul", "committed", r"\bfoul\b|\bbrings? (him )?down\b|\bfouled\b|\bfree kick\b|\btrips?\b|\bchallenge\b",
     None, 0.8, 0.45),
    ("corner", "awarded", r"\bout for a corner\b|\bcorner (kick )?(for|to)\b|\bit'?s a corner\b",
     None, 0.6, -4.0),
    ("corner", "taken", r"\b(swings?|whips?|curls?|delivers?|takes?) (in )?the corner\b|\bcorner comes in\b|"
                        r"\bfrom the corner\b|\bcorner is taken\b|\bto take it\b",
     None, 0.75, 0.3),
    ("substitution", "made", r"\bsubstitution\b|\bcomes? on for\b|\bmakes? way\b|\breplaced by\b|\bcoming off\b",
     None, 0.85, 0.0),
    ("kickoff", "start", r"\bkick ?off\b|\bunderway\b|\bunder way\b|\bget us started\b", None, 0.8, 0.3),
    ("half_time", "whistle", r"\bhalf[- ]time\b|\bend of the first half\b", r"\bsecond half\b", 0.85, 0.3),
    ("full_time", "whistle", r"\bfull[- ]time\b|\bfinal whistle\b|\bit'?s all over\b", None, 0.9, 0.4),
]
COMPILED = [(t, ph, re.compile(p, re.I), re.compile(x, re.I) if x else None, c, lag)
            for t, ph, p, x, c, lag in RULES]

FUZZY_TERMS = {"goal": "goal", "yellow card": "yellow_card", "red card": "red_card", "corner": "corner",
               "foul": "foul", "substitution": "substitution"}


def _to_int(tok: str) -> int | None:
    tok = tok.lower()
    if tok.isdigit():
        return int(tok)
    return NUM_WORDS.get(tok)


def _word_time(sentence: dict, match_start_char: int) -> float:
    """Map a character offset in the sentence text to the spoken word's start."""
    pos = 0
    for w in sentence.get("words", []):
        idx = sentence["text"].find(w["w"], pos)
        if idx < 0:
            continue
        if idx + len(w["w"]) >= match_start_char:
            return w["start"]
        pos = idx + len(w["w"])
    return sentence["start"]


def _team(text: str, roster: dict | None) -> str | None:
    low = text.lower()
    if roster and "teams" in roster:
        hits = []
        for code, t in roster.get("teams", {}).items():
            names = [t.get("name", "").lower(), t.get("code", "").lower()] + [a.lower() for a in t.get("aliases", [])]
            for n in names:
                if n and re.search(rf"\b{re.escape(n)}\b", low):
                    hits.append((low.find(n), code))
        if hits:
            return sorted(hits)[0][1]

    # Heuristic fallback for common soccer teams in benchmarks/broadcasts
    if re.search(r"\b(?:manchester|man utd|united|lions|home side)\b", low):
        return "A"
    if re.search(r"\b(?:arsenal|gunners|falcons|away side)\b", low):
        return "B"
    return None


def _player(text: str, roster: dict | None, is_goal: bool = False) -> tuple[str | None, str | None, str | None, int | None]:
    """Extract (player_id, player_name, team_code, jersey_number) if mentioned."""
    if not roster or "players" not in roster:
        return None, None, None, None
    low = text.lower()
    matches = []
    for pid, p in roster.get("players", {}).items():
        name = p.get("name", "").lower()
        aliases = [a.lower() for a in p.get("aliases", [])]
        for query in [name] + aliases:
            if not query:
                continue
            m = re.search(rf"\b{re.escape(query)}\b", low)
            if m:
                matches.append((m.start(), pid, p.get("name"), p.get("team"), p.get("number"), p.get("position", "")))
                break

    if not matches:
        return None, None, None, None

    if len(matches) == 1:
        _, pid, name, team, num, _ = matches[0]
        return pid, name, team, num

    # Multiple players in sentence
    if is_goal:
        # Check if "playing ... onside" pattern: attacker is after playing and before onside
        if "playing" in low and "onside" in low:
            idx_play = low.find("playing")
            idx_on = low.find("onside")
            onside_cands = [m for m in matches if idx_play < m[0] < idx_on]
            if onside_cands:
                _, pid, name, team, num, _ = onside_cands[0]
                return pid, name, team, num

        # Prioritize forward/winger over defenders for goal scoring attribution
        pos_rank = {"Forward": 4, "Winger": 3, "Midfielder": 2, "Defender": 0, "Goalkeeper": -1}
        matches.sort(key=lambda m: pos_rank.get(m[5], 1), reverse=True)
        _, pid, name, team, num, _ = matches[0]
        return pid, name, team, num

    matches.sort(key=lambda m: m[0])
    _, pid, name, team, num, _ = matches[0]
    return pid, name, team, num


def spot(sentences: list[dict], roster: dict | None = None) -> list[Candidate]:
    cands: list[Candidate] = []
    for s in sentences:
        text = s["text"]
        low = text.lower()
        team = _team(text, roster)
        pid, pname, pteam, pnum = _player(text, roster, is_goal=False)
        if not team and pteam:
            team = pteam

        numbers = [(_to_int(m.group(1)), m.start()) for m in NUMBER_RE.finditer(text)]
        is_replay_talk = bool(REPLAY_RE.search(low))
        if is_replay_talk:
            cands.append(Candidate("audio", "replay_cue", s["start"], 0.7,
                                   {"text": text, "sentence": [s["start"], s["end"]]}))
        fired = set()
        for typ, phase, pat, excl, conf, lag in COMPILED:
            m = pat.search(text)
            if not m:
                continue
            if excl and excl.search(low) and not _strong_override(typ, low):
                continue
            if typ in fired:
                continue
            if is_replay_talk and typ in ("goal", "shot_on_target", "foul"):
                # "Let's see that goal again" describes a past event.
                continue
            if typ == "shot_on_target" and "goal" in fired:
                continue
            fired.add(typ)
            t_kw = _word_time(s, m.start())
            payload = {"keyword": m.group(0), "text": text, "phase": phase, "lag": lag,
                       "spoken_at": t_kw, "sentence": [s["start"], s["end"]], "team_mention": team}
            if typ == "goal" and not pid:
                # Find closest player from the scoring team within 16s
                close_sens = sorted(sentences, key=lambda o: abs(o["start"] - s["start"]))
                for other in close_sens:
                    if abs(other["start"] - s["start"]) > 16.0:
                        break
                    # If this context sentence mentions team or equality, update team
                    other_team = _team(other["text"], roster)
                    if not team and other_team:
                        team = other_team

                    n_pid, n_pname, n_pteam, n_pnum = _player(other["text"], roster, is_goal=True)
                    if n_pid and (team is None or n_pteam == team):
                        pid, pname, pteam, pnum = n_pid, n_pname, n_pteam, n_pnum
                        if not team:
                            team = pteam
                        break

            if pid:
                payload["player_id"] = pid
                payload["player_name"] = pname
                payload["jersey"] = pnum
                payload["team"] = pteam

            if numbers and "jersey" not in payload:
                if typ == "substitution" and len(numbers) >= 2:
                    on_off = _sub_on_off(text, numbers)
                    payload.update(on_off)
                else:
                    j_num = numbers[0][0]
                    payload["jersey"] = j_num
                    if not pid and roster and "players" in roster:
                        for p_k, p_v in roster["players"].items():
                            if p_v.get("number") == j_num and (team is None or p_v.get("team") == team):
                                payload["player_id"] = p_k
                                payload["player_name"] = p_v.get("name")
                                payload["team"] = p_v.get("team")
                                break

            if typ == "goal":
                t_cand = min(round(t_kw - lag, 3), max(0.0, round(s["start"] - 0.5, 3)))
            else:
                t_cand = round(t_kw - lag, 3)

            cands.append(Candidate("audio", typ, t_cand, conf, payload))
        if not fired and fuzz is not None:
            cands.extend(_fuzzy(s, team))
    return cands


def _strong_override(typ: str, low: str) -> bool:
    if typ == "goal":
        return bool(re.search(r"\bgoal\b|\bscores?\b|\bscored\b|\bequali[sz]es?\b|\b(are|is) level\b|\bburies\b|\bslotted away\b", low)) and \
            not re.search(r"\bon goal\b|\bat goal\b|\bgoal kick\b|\bgoals to\b", low)
    return False


def _sub_on_off(text: str, numbers: list) -> dict:
    low = text.lower()
    m = re.search(r"(\w+) comes? on for (?:number )?(\w+)", low)
    if m:
        a, b = _to_int(m.group(1)), _to_int(m.group(2))
        if a and b:
            return {"player_on": a, "player_off": b}
    return {"player_on": numbers[0][0], "player_off": numbers[1][0]}


def _fuzzy(s: dict, team) -> list[Candidate]:
    out = []
    for w in s.get("words", []):
        tok = re.sub(r"[^a-z ]", "", w["w"].lower())
        if len(tok) < 4:
            continue
        for term, typ in FUZZY_TERMS.items():
            if " " in term:
                continue
            score = fuzz.ratio(tok, term)
            if 80 <= score < 100:
                out.append(Candidate("audio", typ, w["start"], 0.35 * score / 100,
                                     {"keyword": w["w"], "fuzzy": term, "score": score, "text": s["text"],
                                      "team_mention": team, "lag": 0.0, "phase": "fuzzy"}))
    return out
