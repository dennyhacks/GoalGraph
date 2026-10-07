#!/usr/bin/env python3
"""Synthetic football broadcast generator with exact ground truth.

Why this exists
---------------
SoccerNet videos need an NDA password and YouTube clips need manual labels.
This script renders a ~3-minute *broadcast-style* highlight package so the
whole GoalGraph pipeline can be run and **scored** end-to-end, reproducibly,
on any laptop.  It deliberately contains every hard case from PS02:

* camera cuts (wide / mid / close-up shots)            -> cut detection, ReID
* slow-motion replays bracketed by logo wipes           -> replay-aware timestamps
* a scorer who is occluded and later leaves the frame   -> occlusion-robust ReID
* scoreboard bug that updates *after* the goal          -> lag-calibrated fusion
* commentary that mentions "goal" when no goal happens  -> triangulation rejects it
* "second yellow ... red card" in one sentence          -> keyword disambiguation
* compressed match clock (highlights)                   -> match-time reasoning
* TTS commentary + whistles + crowd noise               -> Whisper + whistle detector

Output: ``data/demo/demo_match.mp4``, ``demo_match_gt.json``, ``roster.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 1280, 720, 25
DURATION = 196.0
PL, PW = 105.0, 68.0
TEAM_CODE = {"A": "LIO", "B": "FAL"}
TEAM_FULL = {"A": "Lions", "B": "Falcons"}
ZOOM = {"wide": 18.0, "mid": 26.0, "close": 38.0, "replay": 33.0}

# BGR colours, chosen to have well separated hues
C = {
    "A_shirt": (40, 40, 210), "A_shorts": (30, 30, 95),
    "B_shirt": (200, 90, 25), "B_shorts": (95, 45, 20),
    "A_gk": (210, 210, 30), "B_gk": (170, 40, 160),
    "ref": (25, 25, 25), "skin": (140, 170, 210), "num": (245, 245, 245),
    "grass1": (45, 140, 55), "grass2": (38, 126, 48), "line": (235, 235, 235),
    "flag": (0, 120, 255), "yellow": (0, 225, 255), "red": (20, 20, 235),
}

FORM_A = {1: (4, 34), 2: (25, 10), 5: (22, 27), 6: (22, 41), 3: (25, 58), 4: (40, 24),
          8: (40, 44), 7: (55, 10), 10: (55, 34), 11: (55, 58), 9: (68, 34)}
FORM_B = {n: (PL - x, PW - y) for n, (x, y) in FORM_A.items()}
FORM_B[14] = FORM_B[7]


# --------------------------------------------------------------------------
# Script
# --------------------------------------------------------------------------
class Scene:
    def __init__(self, name, t0, t1, half, clock, ball, over=None, shots=None,
                 focus=None, fallen=None, cards=None, run_from=None, stop_at=None,
                 tag=None, banner=None):
        self.name, self.t0, self.t1, self.half = name, t0, t1, half
        self.clock = clock                       # match seconds at t0
        self.run_from = t0 if run_from is None else run_from
        self.stop_at = stop_at
        self.ball = ball                         # [(t, x, y, z?)]
        self.over = over or {}                   # {("A", 9): [(t, x, y)]}
        self.shots = shots or [(t0, "wide")]
        self.focus = focus or []                 # [(t0, t1, ("A", 9))]
        self.fallen = fallen or []               # [(team, num, t0, t1)]
        self.cards = cards or []                 # [(t0, t1, colour)]
        self.tag = tag                           # (t_from, "HT")
        self.banner = banner                     # (t0, t1, line1, line2)

    def clock_at(self, t):
        if t < self.run_from:
            return self.clock
        end = t if self.stop_at is None else min(t, self.stop_at)
        return self.clock + (end - self.run_from)

    def mode_at(self, t):
        m = self.shots[0][1]
        for ts, mode in self.shots:
            if t >= ts:
                m = mode
        return m


def mmss(m, s):
    return 60 * m + s


SCENES = [
    Scene("kickoff_h1", 0, 10, 1, 0, run_from=1.0,
          ball=[(0, 52.5, 34), (1.0, 52.5, 34), (2.0, 45, 40), (4.5, 30, 30), (8.0, 70, 20), (10, 74, 19)],
          over={("A", 10): [(0, 52.5, 35.6), (1.0, 52.6, 35.0), (3, 54, 36)],
                ("A", 9): [(0, 52.5, 32.0), (3, 58, 30)],
                ("A", 8): [(0, 46, 41), (2.0, 45.5, 40.5), (4, 44, 38)],
                ("A", 5): [(0, 30, 30), (4.5, 30.5, 30.3), (10, 33, 30)]}),
    Scene("foul_yellow", 10, 28, 1, mmss(8, 10),
          ball=[(10, 60, 40), (17.9, 72, 36.5), (19.0, 74.5, 37), (28, 74.5, 37)],
          over={("A", 10): [(10, 59, 40), (17.9, 71, 36.5)],
                ("B", 4): [(10, 76, 28), (14, 75, 30), (18.0, 72.3, 35.6), (19.5, 73, 34.3)],
                ("R", 0): [(10, 64, 28), (18, 66, 30), (22, 71, 33.2)]},
          shots=[(10, "wide"), (21.8, "close")],
          focus=[(21.8, 28, ("R", 0))],
          fallen=[("A", 10, 18.0, 21.5)],
          cards=[(23.0, 25.5, "yellow")]),
    Scene("corner_goal", 28, 48, 1, mmss(21, 40),
          ball=[(28, 81, 12.5), (30.5, 95, 8), (31.0, 105, 3), (31.4, 106.5, 2), (33.9, 106.5, 2),
                (34.0, 104.6, 0.6), (37.0, 104.6, 0.6), (38.0, 99, 18, 4), (38.9, 95.4, 33.2, 2.2),
                (39.0, 95.5, 33.2, 2.0), (39.8, 105.0, 35, 1.0), (40.3, 106.3, 35.2, 0)],
          over={("A", 7): [(28, 80, 12.5), (30.5, 94, 8.2), (33.9, 97, 7), (34.0, 103.2, -0.4),
                           (36.4, 103.3, -0.4), (37.0, 104.2, 0.2), (39, 102, 3)],
                ("B", 3): [(28, 92, 10), (30.6, 96.5, 6.2), (34, 98, 12)],
                ("A", 9): [(28, 82, 30), (34.0, 87.5, 27), (35.4, 91.0, 24.6), (36.3, 91.6, 24.5),
                           (38.9, 95.3, 33.4), (40.5, 96, 30), (42, 100, 18), (44.5, 104.4, 2.0),
                           (48, 104.6, 1.6)],
                ("B", 5): [(28, 90, 30), (34.0, 91.3, 24.8), (36.3, 91.9, 24.8), (40, 96, 36)],
                ("B", 1): [(28, 103, 34), (39.4, 103.4, 34.6), (39.7, 104, 36.4)]},
          shots=[(28, "wide"), (34.0, "mid"), (44.0, "close")],
          focus=[(34.0, 37.3, ("BALLX", 0)), (44.0, 48, ("A", 9))],
          fallen=[("B", 1, 39.6, 42.5)]),
    Scene("shot_save", 60, 80, 1, mmss(33, 15),
          ball=[(60, 70, 40), (65.9, 83.5, 30.6), (66.8, 102.7, 33.1, 1.1), (72, 102.4, 33.1, 1.1),
                (73.6, 88, 52), (80, 80, 55)],
          over={("A", 10): [(60, 69, 41), (65.9, 82.8, 31.0), (68, 86, 31)],
                ("B", 1): [(60, 103, 34), (66.8, 103.0, 33.1), (72, 102.5, 33.1)]},
          shots=[(60, "wide"), (68, "close"), (74, "wide")],
          focus=[(68, 74, ("B", 1))]),
    Scene("half_time", 80, 89, 1, mmss(44, 56), stop_at=84.0,
          ball=[(80, 50, 30), (84, 55, 33), (89, 55, 33)], tag=(85.0, "HT")),
    Scene("kickoff_h2", 92, 102, 2, mmss(45, 0), run_from=93.0,
          ball=[(92, 52.5, 34), (93.0, 52.5, 34), (94.5, 60, 28), (98, 75, 40), (102, 70, 45)],
          over={("B", 10): [(92, 52.5, 32.4), (93.0, 52.4, 33.0), (95, 55, 30)]}),
    Scene("sub_corner", 102, 124, 2, mmss(52, 30),
          ball=[(102, 60, 45), (110, 40, 60), (112.6, 18.5, 62), (112.8, 15, 63), (113.3, -1, 66),
                (113.6, -2, 66.5), (114.9, -2, 66.5), (115.0, 0.5, 67.4), (117.0, 0.5, 67.4),
                (118.6, 6, 36, 2.0), (120, 30, 30), (124, 40, 32)],
          over={("B", 7): [(102, 55, 64), (105, 52.5, 69.5), (110, 52, 71)],
                ("B", 14): [(102, 52.5, 71), (105.5, 52.6, 69), (107, 53, 64), (110, 50, 62),
                            (114.9, 2, 68.6), (116.4, 0.0, 68.2), (117.0, 0.3, 67.6), (119, 3, 64)],
                ("B", 11): [(102, 62, 50), (110, 41, 60), (112.6, 19.5, 62), (114, 15, 58)],
                ("A", 3): [(102, 30, 58), (112.6, 16, 63.5), (114, 14, 62)],
                ("A", 5): [(102, 22, 30), (118.6, 6.3, 36.2), (121, 10, 34)]},
          shots=[(102, "close"), (110, "wide")],
          focus=[(102, 110, ("FIX", (53, 66)))],
          banner=(104.5, 109.5, "SUBSTITUTION  FAL", "OFF 7    ON 14")),
    Scene("goal_b", 124, 138, 2, mmss(61, 5),
          ball=[(124, 36, 30.5), (129.95, 18.3, 28.6), (130.6, 0.0, 33, 0.8), (131.2, -1.4, 33.2, 0),
                (138, -1.4, 33.2)],
          over={("B", 11): [(124, 35, 30), (129.9, 17.6, 28.3), (131, 16, 27), (134, 20, 10),
                            (138, 25, 4)],
                ("A", 1): [(124, 2, 34), (130.4, 1.8, 32), (130.6, 1.5, 33.5)]},
          shots=[(124, "wide"), (131.5, "close")],
          focus=[(131.5, 138, ("B", 11))],
          fallen=[("A", 1, 130.5, 133.5)]),
    Scene("corner_save", 146, 156, 2, mmss(74, 20),
          ball=[(146, 104.6, 0.6), (150.0, 104.6, 0.6), (151.0, 99, 16, 4), (151.9, 94.2, 30.1, 2.1),
                (152.0, 94.3, 30.2, 2.0), (152.5, 103.2, 35.4, 1.1), (156, 103, 35.4, 1.1)],
          over={("A", 7): [(146, 103.4, -0.5), (149.4, 103.3, -0.5), (150.0, 104.2, 0.2), (152, 102, 4)],
                ("A", 9): [(146, 90, 40), (151.9, 94.1, 30.3), (154, 95, 28)],
                ("B", 1): [(146, 103, 34), (152.5, 103.4, 35.4), (156, 103, 35)]},
          shots=[(146, "mid")],
          focus=[(146, 150.3, ("BALLX", 0))]),
    Scene("goal_a2", 156, 168, 2, mmss(78, 2),
          ball=[(156, 76, 40.5), (160.4, 88.6, 37.2), (161.0, 105.0, 31, 0.8), (161.4, 106.2, 30.8, 0),
                (168, 106.2, 30.8)],
          over={("A", 9): [(156, 75, 41), (160.4, 87.9, 37.4), (161.5, 92, 38), (164, 95, 48),
                           (168, 93, 52)],
                ("B", 1): [(156, 102, 34), (160.9, 103.4, 32.8), (161.1, 103.6, 31.6)]},
          shots=[(156, "wide"), (163.0, "close")],
          focus=[(163.0, 168, ("A", 9))],
          fallen=[("B", 1, 161.0, 163.5)]),
    Scene("foul_red", 168, 184, 2, mmss(86, 40),
          ball=[(168, 61, 30.5), (171.95, 68.8, 33.2), (172.8, 71, 35), (184, 71, 35)],
          over={("A", 10): [(168, 60, 30.5), (171.9, 67.9, 33.1)],
                ("B", 4): [(168, 72, 28), (172.0, 68.3, 33.4), (173.5, 69, 32.5), (179.6, 69, 32.5),
                           (184, 64, 50)],
                ("R", 0): [(168, 60, 40), (176, 67, 35.4)]},
          shots=[(168, "wide"), (176, "close")],
          focus=[(176, 184, ("R", 0))],
          fallen=[("A", 10, 172.0, 176.0)],
          cards=[(177.0, 179.5, "red")]),
    Scene("full_time", 184, 196, 2, mmss(93, 5), stop_at=190.0,
          ball=[(184, 45, 30), (190, 50, 34), (196, 50, 34)], tag=(191.0, "FT")),
]


class Replay:
    def __init__(self, t0, t1, src_from, speed=0.5, focus=None):
        self.t0, self.t1, self.src_from, self.speed = t0, t1, src_from, speed
        self.focus = focus
        self.wipe = 0.6

    def world_t(self, t):
        u = min(max(t - (self.t0 + self.wipe), 0.0), (self.t1 - self.t0 - 2 * self.wipe))
        return self.src_from + u * self.speed

    @property
    def gt(self):
        a, b = self.t0 + self.wipe, self.t1 - self.wipe
        return {"video": [a, b], "live": [self.world_t(a), self.world_t(b)], "speed": self.speed}


REPLAYS = [Replay(48, 60, 36.5, focus=("A", 9)), Replay(138, 146, 128.9, focus=("B", 11))]
GRAPHICS = [(89, 92, "HALF TIME")]

SCORE_CHANGES = [(41.5, (1, 0)), (132.2, (1, 1)), (162.8, (2, 1))]

COMMENTARY = [
    (1.3, "And we're underway! The referee blows for kick off."),
    (5.8, "The Lions knocking it around at the back."),
    (18.5, "Oh, that's a foul! Number four brings him down."),
    (23.4, "And the referee shows a yellow card to number four."),
    (28.6, "The Lions attacking down the right."),
    (31.4, "Deflected, and that's out for a corner."),
    (36.2, "Number seven swings in the corner."),
    (40.1, "Goal! Number nine scores for the Lions!"),
    (44.2, "What a moment for the home side."),
    (51.0, "Let's see that goal again. What a header from number nine."),
    (62.0, "The Lions pressing forward."),
    (66.3, "Shot from number ten! Great save by the keeper!"),
    (76.0, "The Falcons trying to build from the back."),
    (84.3, "And there's the whistle for half time."),
    (93.4, "The second half is underway."),
    (97.0, "The Falcons need a response here."),
    (105.0, "A substitution for the Falcons. Number fourteen comes on for number seven."),
    (113.2, "Blocked, and it's out for a corner to the Falcons."),
    (116.6, "The corner comes in, and it's cleared."),
    (126.0, "The Falcons break forward."),
    (130.8, "Goal! Number eleven equalises for the Falcons!"),
    (139.5, "Another look at the goal. Clinical finish from number eleven."),
    (146.8, "Corner for the Lions. Number seven to take it."),
    (152.4, "Header from number nine, but the keeper saves!"),
    (157.0, "Number nine is through on goal!"),
    (161.3, "Goal! Number nine again! He has scored twice!"),
    (166.0, "The Lions lead two one."),
    (172.4, "Another foul by number four!"),
    (177.3, "That's a second yellow, and it's a red card! Number four is sent off!"),
    (185.5, "We're deep into stoppage time."),
    (190.4, "And that's full time! The Lions win it, two goals to one."),
]
WHISTLES = [(1.0, 0.5), (18.25, 0.35), (84.0, 1.2), (93.0, 0.5), (172.25, 0.35), (190.0, 1.6)]
CHEERS = [(39.9, 5.0, 0.22), (130.7, 4.5, 0.18), (161.1, 4.5, 0.22), (66.8, 1.8, 0.10), (152.6, 1.8, 0.10)]

GT_EVENTS = [
    ("kickoff", 1.0, None, None, 1),
    ("foul", 18.0, "B", "B#4", 1),
    ("yellow_card", 23.0, "B", "B#4", 1),
    ("corner", 37.0, "A", "A#7", 1),
    ("shot_on_target", 39.0, "A", "A#9", 1),
    ("goal", 39.8, "A", "A#9", 1),
    ("shot_on_target", 66.0, "A", "A#10", 1),
    ("half_time", 84.0, None, None, 1),
    ("kickoff", 93.0, "B", None, 2),
    ("substitution", 105.0, "B", "B#14", 2),
    ("corner", 117.0, "B", "B#14", 2),
    ("shot_on_target", 130.0, "B", "B#11", 2),
    ("goal", 130.6, "B", "B#11", 2),
    ("corner", 150.0, "A", "A#7", 2),
    ("shot_on_target", 152.0, "A", "A#9", 2),
    ("shot_on_target", 160.4, "A", "A#9", 2),
    ("goal", 161.0, "A", "A#9", 2),
    ("foul", 172.0, "B", "B#4", 2),
    ("red_card", 177.0, "B", "B#4", 2),
    ("full_time", 190.0, None, None, 2),
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def interp(track, t):
    if t <= track[0][0]:
        p = track[0][1:]
    elif t >= track[-1][0]:
        p = track[-1][1:]
    else:
        p = None
        for a, b in zip(track, track[1:]):
            if a[0] <= t <= b[0]:
                u = (t - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 1.0
                pa, pb = list(a[1:]), list(b[1:])
                n = max(len(pa), len(pb))
                pa += [0.0] * (n - len(pa))
                pb += [0.0] * (n - len(pb))
                p = [x + (y - x) * u for x, y in zip(pa, pb)]
                break
    p = list(p)
    while len(p) < 3:
        p.append(0.0)
    return p


def scene_at_world(tw):
    for s in SCENES:
        if s.t0 <= tw < s.t1 or (s is SCENES[-1] and tw >= s.t0):
            return s
    return None


def active_roster(t):
    a = list(FORM_A.keys())
    b = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
    b.append(7 if t < 105.5 else 14)
    if 102 <= t < 110:          # both visible at the touchline during the sub
        b = sorted(set(b) | {7, 14})
    if t >= 184:                # sent off
        b.remove(4)
    return a, b


def entity_pos(scene, team, num, t):
    key = (team, num)
    if key in scene.over:
        return interp(scene.over[key], t)[:2]
    bx, by, _ = interp(scene.ball, t)
    if team == "R":
        return [bx - 8 + 2 * math.sin(0.3 * t), by + 7 + 2 * math.cos(0.25 * t)]
    base = FORM_A[num] if team == "A" else FORM_B[num]
    ph = num * 1.7 + (0.0 if team == "A" else 3.1)
    if num == 1:
        x = base[0] + 0.05 * (bx - 52.5)
        y = 34 + 0.15 * (by - 34)
    else:
        x = base[0] + 0.35 * (bx - 52.5) + 0.8 * math.sin(0.4 * t + ph)
        y = base[1] + 0.2 * (by - 34) + 1.2 * math.sin(0.5 * t + ph)
    return [min(max(x, 1.0), PL - 1.0), min(max(y, 1.0), PW - 1.0)]


def is_fallen(scene, team, num, t):
    return any(f[0] == team and f[1] == num and f[2] <= t < f[3] for f in scene.fallen)


# --------------------------------------------------------------------------
# Pitch texture (world-space, pre-rendered at two resolutions)
# --------------------------------------------------------------------------
X0, X1, Y0, Y1 = -12.0, 117.0, -16.0, 84.0


def build_texture(ppm):
    w, h = int((X1 - X0) * ppm), int((Y1 - Y0) * ppm)
    img = np.zeros((h, w, 3), np.uint8)
    rng = np.random.default_rng(7)

    def P(x, y):
        return int(round((x - X0) * ppm)), int(round((y - Y0) * ppm))

    # stands + crowd
    img[:] = (58, 50, 46)
    n = int(w * h / (ppm * ppm) * 3)
    xs, ys = rng.integers(0, w, n), rng.integers(0, h, n)
    cols = rng.integers(60, 230, (n, 3))
    for x, y, c in zip(xs, ys, cols):
        cv2.circle(img, (int(x), int(y)), max(1, int(ppm * 0.18)), tuple(int(v) for v in c), -1)
    # grass incl. margin
    cv2.rectangle(img, P(-5, -5), P(PL + 5, PW + 5), C["grass2"], -1)
    stripe = PL / 20
    for i in range(-1, 21):
        if i % 2 == 0:
            cv2.rectangle(img, P(max(-5, i * stripe), -5), P(min(PL + 5, (i + 1) * stripe), PW + 5),
                          C["grass1"], -1)
    # advertising boards
    for (xa, ya, xb, yb) in [(-5, -6.5, PL + 5, -5), (-5, PW + 5, PL + 5, PW + 6.5)]:
        cv2.rectangle(img, P(xa, ya), P(xb, yb), (40, 20, 10), -1)
        for k, xx in enumerate(np.arange(xa, xb, 12)):
            cv2.putText(img, "GOALGRAPH", P(xx + 1, yb - 0.35), cv2.FONT_HERSHEY_SIMPLEX,
                        ppm * 0.032, (0, 200, 255) if k % 2 else (255, 255, 255),
                        max(1, int(ppm * 0.06)), cv2.LINE_AA)
    lt = max(2, int(round(0.12 * ppm)))
    L = C["line"]
    cv2.rectangle(img, P(0, 0), P(PL, PW), L, lt)
    cv2.line(img, P(PL / 2, 0), P(PL / 2, PW), L, lt)
    cv2.circle(img, P(PL / 2, PW / 2), int(9.15 * ppm), L, lt)
    cv2.circle(img, P(PL / 2, PW / 2), max(2, int(0.25 * ppm)), L, -1)
    for side in (0, 1):
        sx = (lambda v: v) if side == 0 else (lambda v: PL - v)
        cv2.rectangle(img, P(sx(0), 13.84), P(sx(16.5), PW - 13.84), L, lt)
        cv2.rectangle(img, P(sx(0), 24.84), P(sx(5.5), PW - 24.84), L, lt)
        cv2.circle(img, P(sx(11), 34), max(2, int(0.2 * ppm)), L, -1)
        c = P(sx(11), 34)
        a0, a1 = (-53, 53) if side == 0 else (127, 233)
        cv2.ellipse(img, c, (int(9.15 * ppm), int(9.15 * ppm)), 0, a0, a1, L, lt)
        # repaint inside the box to hide the arc part inside it
        # goal + net (behind the line)
        gx0, gx1 = (sx(0), sx(-2.0)) if side == 0 else (sx(0), sx(-2.0))
        xa, xb = min(gx0, gx1), max(gx0, gx1)
        cv2.rectangle(img, P(xa, 30.34), P(xb, 37.66), (60, 150, 70), -1)
        step = 0.4
        nlt = max(1, int(round(0.05 * ppm)))
        for xx in np.arange(xa, xb + 1e-6, step):
            cv2.line(img, P(xx, 30.34), P(xx, 37.66), (240, 240, 240), nlt)
        for yy in np.arange(30.34, 37.66 + 1e-6, step):
            cv2.line(img, P(xa, yy), P(xb, yy), (240, 240, 240), nlt)
        cv2.rectangle(img, P(xa, 30.34), P(xb, 37.66), (255, 255, 255), lt + 1)
    for (cx, cy) in [(0, 0), (PL, 0), (0, PW), (PL, PW)]:
        cv2.ellipse(img, P(cx, cy), (int(1 * ppm), int(1 * ppm)), 0, 0, 360, L, lt)
        ox = -0.6 if cx == 0 else 0.6
        oy = -0.6 if cy == 0 else 0.6
        tri = np.array([P(cx + ox, cy + oy), P(cx + ox * 2.2, cy + oy * 1.2),
                        P(cx + ox * 1.2, cy + oy * 2.2)], np.int32)
        cv2.fillPoly(img, [tri], C["flag"])
    # repaint pitch area outside the margin-cut on the near grass margin
    return img


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
class Renderer:
    def __init__(self):
        self.tex = {20: build_texture(20), 40: build_texture(40)}

    def background(self, cx, cy, z):
        ppm = 20 if z <= 22 else 40
        s = z / ppm
        M = np.array([[s, 0, (X0 - cx) * z + W / 2], [0, s, (Y0 - cy) * z + H / 2]], np.float32)
        return cv2.warpAffine(self.tex[ppm], M, (W, H), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(58, 50, 46))

    @staticmethod
    def to_screen(x, y, cam):
        cx, cy, z = cam
        return (x - cx) * z + W / 2, (y - cy) * z + H / 2


def draw_player(img, sx, sy, z, shirt, shorts, num, fallen=False, card=None):
    sx, sy = int(sx), int(sy)
    cv2.ellipse(img, (sx, sy), (max(2, int(0.45 * z)), max(1, int(0.15 * z))), 0, 0, 360, (25, 80, 30), -1)
    if fallen:
        body = np.array([[sx - 0.95 * z, sy - 0.42 * z], [sx + 0.55 * z, sy - 0.42 * z],
                         [sx + 0.55 * z, sy - 0.02 * z], [sx - 0.95 * z, sy - 0.02 * z]], np.int32)
        cv2.fillPoly(img, [body], shirt)
        cv2.rectangle(img, (int(sx + 0.55 * z), int(sy - 0.38 * z)), (int(sx + 0.85 * z), int(sy - 0.06 * z)),
                      shorts, -1)
        cv2.circle(img, (int(sx - 1.1 * z), int(sy - 0.22 * z)), max(2, int(0.15 * z)), C["skin"], -1)
        return
    lw = max(1, int(0.09 * z))
    for dx in (-0.1, 0.1):
        cv2.line(img, (int(sx + dx * z), sy), (int(sx + dx * z), int(sy - 0.8 * z)), C["skin"], lw)
        cv2.line(img, (int(sx + dx * z), sy), (int(sx + dx * z), int(sy - 0.35 * z)), shirt, lw)
    cv2.rectangle(img, (int(sx - 0.24 * z), int(sy - 1.02 * z)), (int(sx + 0.24 * z), int(sy - 0.78 * z)),
                  shorts, -1)
    x0, y0, x1, y1 = int(sx - 0.29 * z), int(sy - 1.58 * z), int(sx + 0.29 * z), int(sy - 1.0 * z)
    cv2.rectangle(img, (x0, y0), (x1, y1), shirt, -1)
    cv2.circle(img, (sx, int(sy - 1.72 * z)), max(2, int(0.14 * z)), C["skin"], -1)
    if num is not None and z >= 24:
        txt = str(num)
        scale = 0.0175 * z
        th = max(1, int(z / 22))
        (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, scale, th)
        cv2.putText(img, txt, (sx - tw // 2, int(sy - 1.29 * z + tht / 2)), cv2.FONT_HERSHEY_DUPLEX,
                    scale, C["num"], th, cv2.LINE_AA)
    if card:
        cx, cy = int(sx + 0.42 * z), int(sy - 2.2 * z)
        cv2.line(img, (int(sx + 0.25 * z), int(sy - 1.5 * z)), (cx, cy + int(0.2 * z)), C["skin"], lw)
        cv2.rectangle(img, (int(cx - 0.21 * z), int(cy - 0.3 * z)), (int(cx + 0.21 * z), int(cy + 0.3 * z)),
                      C[card], -1)


def draw_ball(img, sx, sy, z, height):
    r = max(3, int(0.25 * z))
    cv2.ellipse(img, (int(sx), int(sy)), (r, max(1, r // 2)), 0, 0, 360, (25, 80, 30), -1)
    by = int(sy - height * z)
    cv2.circle(img, (int(sx), by), r, (250, 250, 250), -1, cv2.LINE_AA)
    cv2.circle(img, (int(sx), by), max(1, r // 3), (60, 60, 60), -1)


def focus_pos(scene, key, t):
    team, num = key
    if team == "FIX":
        return list(num)
    if team == "BALLX":
        bx, by, _ = interp(scene.ball, t)
        return [bx - 10 if bx > 52 else bx + 10, by + (14 if by < 34 else -14)]
    return entity_pos(scene, team, num, t)


def camera(scene, t, z, shot_start, focus_key=None):
    pts = []
    for k in range(9):
        tt = max(shot_start, t - 0.9 + k * 0.1125)
        if focus_key:
            pts.append(focus_pos(scene, focus_key, tt))
        else:
            pts.append(interp(scene.ball, tt)[:2])
    cx, cy = np.mean(pts, axis=0)
    hw, hh = W / 2 / z, H / 2 / z
    cx = min(max(cx, X0 + hw), X1 - hw)
    cy = min(max(cy, Y0 + hh), Y1 - hh)
    return cx, cy, z


def render_world(R, scene, tw, cam, tv):
    img = R.background(*cam)
    z = cam[2]
    a_nums, b_nums = active_roster(tw)
    items = []
    for team, nums in (("A", a_nums), ("B", b_nums)):
        for n in nums:
            x, y = entity_pos(scene, team, n, tw)
            if team == "A":
                shirt = C["A_gk"] if n == 1 else C["A_shirt"]
                shorts = C["A_shorts"]
            else:
                shirt = C["B_gk"] if n == 1 else C["B_shirt"]
                shorts = C["B_shorts"]
            items.append((y, "p", x, team, n, shirt, shorts))
    rx, ry = entity_pos(scene, "R", 0, tw)
    items.append((ry, "r", rx))
    bx, by, bz = interp(scene.ball, tw)
    items.append((by + 0.01, "b", bx, bz))
    # stable ordering: the occluder (B#5) is drawn after A#9 when they overlap
    items.sort(key=lambda it: (round(it[0], 1), 0 if (it[1] == "p" and it[3] == "A") else 1))
    for it in items:
        y = it[0]
        if it[1] == "p":
            _, _, x, team, n, shirt, shorts = it
            sx, sy = Renderer.to_screen(x, y, cam)
            if -60 < sx < W + 60 and -80 < sy < H + 120:
                draw_player(img, sx, sy, z, shirt, shorts, n, fallen=is_fallen(scene, team, n, tw))
        elif it[1] == "r":
            sx, sy = Renderer.to_screen(it[2], y, cam)
            card = None
            for c0, c1, col in scene.cards:
                if c0 <= tw < c1:
                    card = col
            draw_player(img, sx, sy, z, C["ref"], C["ref"], None, card=card)
        else:
            sx, sy = Renderer.to_screen(it[2], y, cam)
            draw_ball(img, sx, sy, z, it[3])
    return img


def draw_scoreboard(img, score, clock_s, tag=None):
    x, y = 36, 28
    cv2.rectangle(img, (x, y), (x + 300, y + 46), (60, 28, 18), -1)
    cv2.rectangle(img, (x, y), (x + 8, y + 46), C["A_shirt"], -1)
    cv2.rectangle(img, (x + 292, y), (x + 300, y + 46), C["B_shirt"], -1)
    txt = f"{TEAM_CODE['A']} {score[0]}-{score[1]} {TEAM_CODE['B']}"
    cv2.putText(img, txt, (x + 22, y + 33), cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    m, s = int(clock_s // 60), int(clock_s % 60)
    cv2.rectangle(img, (x + 300, y), (x + 400, y + 46), (235, 235, 235), -1)
    cv2.putText(img, f"{m:02d}:{s:02d}", (x + 312, y + 33), cv2.FONT_HERSHEY_DUPLEX, 0.95, (30, 30, 30), 2,
                cv2.LINE_AA)
    if tag:
        cv2.rectangle(img, (x + 400, y), (x + 460, y + 46), (0, 190, 255), -1)
        cv2.putText(img, tag, (x + 410, y + 33), cv2.FONT_HERSHEY_DUPLEX, 0.95, (20, 20, 20), 2, cv2.LINE_AA)


def score_at(t):
    s = (0, 0)
    for tc, sc in SCORE_CHANGES:
        if t >= tc:
            s = sc
    return s


def draw_banner(img, l1, l2):
    x, y = 200, H - 150
    cv2.rectangle(img, (x, y), (W - 200, y + 92), (60, 28, 18), -1)
    cv2.rectangle(img, (x, y), (x + 12, y + 92), (0, 190, 255), -1)
    cv2.putText(img, l1, (x + 36, y + 40), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, l2, (x + 36, y + 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA)


def draw_wipe(img, u):
    """Broadcast logo wipe; u in [0,1]."""
    band = 520
    cx = int(-band + u * (W + 2 * band))
    pts = np.array([[cx - band // 2, 0], [cx + band // 2 + 200, 0], [cx + band // 2, H],
                    [cx - band // 2 - 200, H]], np.int32)
    cv2.fillPoly(img, [pts], (0, 170, 255))
    cv2.fillPoly(img, [pts + np.array([60, 0])], (30, 20, 120))
    cv2.circle(img, (cx + 30, H // 2), 120, (255, 255, 255), -1)
    cv2.putText(img, "GG", (cx - 50, H // 2 + 40), cv2.FONT_HERSHEY_DUPLEX, 3.2, (30, 20, 120), 8, cv2.LINE_AA)


def draw_graphic(text, score):
    img = np.zeros((H, W, 3), np.uint8)
    for yy in range(H):
        img[yy] = (70 - yy * 40 // H, 30, 20)
    cv2.putText(img, text, (W // 2 - 260, H // 2 - 40), cv2.FONT_HERSHEY_DUPLEX, 2.6, (255, 255, 255), 5,
                cv2.LINE_AA)
    s = f"{TEAM_CODE['A']}  {score[0]} - {score[1]}  {TEAM_CODE['B']}"
    cv2.putText(img, s, (W // 2 - 270, H // 2 + 80), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 220, 255), 4,
                cv2.LINE_AA)
    return img


def render_frame(R, t, state):
    for g0, g1, text in GRAPHICS:
        if g0 <= t < g1:
            return draw_graphic(text, score_at(t)), "graphic"
    for rp in REPLAYS:
        if rp.t0 <= t < rp.t1:
            tw = rp.world_t(t)
            sc = scene_at_world(tw)
            z = ZOOM["replay"]
            cam = camera(sc, tw, z, rp.src_from, rp.focus if tw > rp.src_from + 3.5 else None)
            img = render_world(R, sc, tw, cam, t)
            cv2.rectangle(img, (W - 230, 30), (W - 40, 80), (20, 20, 200), -1)
            cv2.putText(img, "REPLAY", (W - 215, 67), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            if t < rp.t0 + rp.wipe:
                draw_wipe(img, (t - rp.t0) / rp.wipe)
            elif t >= rp.t1 - rp.wipe:
                draw_wipe(img, (t - (rp.t1 - rp.wipe)) / rp.wipe)
            return img, "replay"
    sc = scene_at_world(t)
    mode = sc.mode_at(t)
    shot_start = max(ts for ts, _ in sc.shots if ts <= t)
    fk = None
    for f0, f1, key in sc.focus:
        if f0 <= t < f1:
            fk = key
    cam = camera(sc, t, ZOOM[mode], shot_start, fk)
    img = render_world(R, sc, t, cam, t)
    tag = sc.tag[1] if sc.tag and t >= sc.tag[0] else None
    draw_scoreboard(img, score_at(t), sc.clock_at(t), tag)
    if sc.banner and sc.banner[0] <= t < sc.banner[1]:
        draw_banner(img, sc.banner[2], sc.banner[3])
    return img, f"{sc.name}:{mode}"


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------
SR = 22050


def tts(text, path):
    for voice in ("Daniel", "Alex", None):
        cmd = ["say", "-r", "185", "-o", str(path)]
        if voice:
            cmd[1:1] = ["-v", voice]
        cmd.append(text)
        if subprocess.run(cmd, capture_output=True).returncode == 0 and path.exists():
            return True
    return False


def read_audio(path):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def pink_noise(n, rng):
    white = rng.standard_normal(n)
    f = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    f /= np.maximum(np.sqrt(freqs), 1.0)
    f[freqs > 6000] *= 0.3
    x = np.fft.irfft(f, n)
    return x / (np.abs(x).max() + 1e-9)


def build_audio(path, tmp):
    rng = np.random.default_rng(3)
    n = int(DURATION * SR)
    t = np.arange(n) / SR
    mix = 0.045 * pink_noise(n, rng) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.07 * t))
    for t0, dur, amp in CHEERS:
        i0, i1 = int(t0 * SR), int(min(DURATION, t0 + dur) * SR)
        seg = pink_noise(i1 - i0, rng)
        u = np.linspace(0, 1, i1 - i0)
        env = np.minimum(u / 0.15, 1.0) * np.exp(-2.2 * u)
        mix[i0:i1] += amp * seg * env * 3
    for t0, dur in WHISTLES:
        i0, i1 = int(t0 * SR), int((t0 + dur) * SR)
        tt = np.arange(i1 - i0) / SR
        tone = (np.sin(2 * np.pi * 3150 * tt) + 0.5 * np.sin(2 * np.pi * 3300 * tt)) * \
               (0.75 + 0.25 * np.sin(2 * np.pi * 28 * tt))
        env = np.minimum(1, tt / 0.02) * np.minimum(1, (dur - tt) / 0.05)
        mix[i0:i1] += 0.28 * tone * env
    have_tts = shutil.which("say") is not None
    for k, (t0, text) in enumerate(COMMENTARY):
        if not have_tts:
            break
        p = tmp / f"line_{k}.aiff"
        if not tts(text, p):
            continue
        x = read_audio(p)
        x = 0.55 * x / (np.abs(x).max() + 1e-9)
        i0 = int(t0 * SR)
        i1 = min(n, i0 + len(x))
        mix[i0:i1] += x[: i1 - i0]
    mix = np.clip(mix, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())
    return have_tts


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------
def clock_at_video(t):
    for s in SCENES:
        if s.t0 <= t < s.t1:
            return s.clock_at(t)
    return None


def build_gt(cuts):
    events = []
    for i, (typ, t, team, player, half) in enumerate(GT_EVENTS):
        events.append({"event_id": f"GT{i + 1:02d}", "type": typ, "t": t, "team": team, "player": player,
                       "half": half, "match_clock_s": clock_at_video(t)})
    return {
        "video": "demo_match.mp4",
        "fps": FPS, "duration": DURATION,
        "teams": {"A": TEAM_FULL["A"], "B": TEAM_FULL["B"]},
        "events": events,
        "replays": [r.gt for r in REPLAYS],
        "cuts": cuts,
        "graphics": [[g0, g1, txt] for g0, g1, txt in GRAPHICS],
        "occlusions": [{"player": "A#9", "occluder": "B#5", "interval": [35.4, 36.3]},
                       {"player": "A#9", "out_of_frame": True, "interval": [41.5, 44.0]}],
        "commentary": [{"t": t, "text": x} for t, x in COMMENTARY],
        "whistles": [{"t": t, "dur": d} for t, d in WHISTLES],
        "score_changes": [{"t": t, "score": f"{a}-{b}"} for t, (a, b) in SCORE_CHANGES],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "demo"))
    ap.add_argument("--seconds", type=float, default=DURATION, help="render only the first N seconds")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R = Renderer()
    tmp = Path(tempfile.mkdtemp(prefix="gg_"))
    raw = tmp / "video.mp4"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n = int(min(args.seconds, DURATION) * FPS)
    cuts, prev = [], None
    for i in range(n):
        t = i / FPS
        img, shot_id = render_frame(R, t, None)
        if prev is not None and shot_id != prev:
            cuts.append({"t": round(t, 3), "from": prev, "to": shot_id})
        prev = shot_id
        vw.write(img)
        if i % (FPS * 20) == 0:
            print(f"  rendered {t:6.1f}s / {n / FPS:.0f}s", flush=True)
    vw.release()
    wav = tmp / "audio.wav"
    print("  building audio (TTS commentary + whistles + crowd)...", flush=True)
    had_tts = build_audio(wav, tmp)
    final = out / "demo_match.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-i", str(wav), "-t", str(n / FPS),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k", "-shortest", str(final)], check=True)
    gt = build_gt(cuts)
    gt["has_commentary_audio"] = had_tts
    (out / "demo_match_gt.json").write_text(json.dumps(gt, indent=2))
    roster = {"teams": {"A": {"name": "Lions", "code": "LIO", "aliases": ["lions", "home side", "home"]},
                        "B": {"name": "Falcons", "code": "FAL", "aliases": ["falcons", "away side", "visitors"]}},
              "players": {}}
    (out / "roster.json").write_text(json.dumps(roster, indent=2))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {final}  ({n / FPS:.0f}s, {len(gt['events'])} GT events, {len(cuts)} cuts)")


if __name__ == "__main__":
    main()
