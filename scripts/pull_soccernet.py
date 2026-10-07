#!/usr/bin/env python3
"""SoccerNet Dataset Ingestion & Converter Tool for GoalGraph.

Allows pulling:
1. Match annotations (Labels-v2.json: Goals, Fouls, Yellow/Red Cards, Kick-offs, Corners, Substitutions)
2. Camera transitions & replay tags (Labels-cameras.json: Real-time vs Replay)
3. Match videos (1_224p.mkv, 2_224p.mkv, 1_720p.mkv, 2_720p.mkv)
4. Automatic conversion into GoalGraph ground truth and roster format
5. Extraction of event clips (e.g., 2-minute highlight clips around goals and fouls)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from SoccerNet.Downloader import SoccerNetDownloader, getListGames

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("soccernet_pull")


def list_matches(league: str | None = None, split: str = "test", limit: int = 15) -> list[str]:
    """List available matches in SoccerNet."""
    games = getListGames(split=split)
    if league:
        games = [g for g in games if league.lower() in g.lower()]
    log.info("Found %d matches (showing first %d):", len(games), min(limit, len(games)))
    for idx, g in enumerate(games[:limit]):
        print(f"[{idx + 1:02d}] {g}")
    return games


def pull_match_data(
    game: str,
    output_dir: str = "data/soccernet",
    files: list[str] | None = None
) -> Path:
    """Download annotations or videos for a specific game."""
    files = files or ["Labels-v2.json", "Labels-cameras.json"]
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    log.info("Downloading %s for match: %s", files, game)
    dl = SoccerNetDownloader(LocalDirectory=str(out_p))
    dl.downloadGame(game, files=files)
    
    game_dir = out_p / game
    log.info("Downloaded successfully to: %s", game_dir)
    return game_dir


def convert_soccernet_to_goalgraph(
    game_dir: Path,
    out_gt_path: Path | None = None,
    out_roster_path: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert SoccerNet Labels-v2.json and Labels-cameras.json into GoalGraph format."""
    labels_file = game_dir / "Labels-v2.json"
    cameras_file = game_dir / "Labels-cameras.json"

    if not labels_file.exists():
        raise FileNotFoundError(f"Missing {labels_file}")

    data = json.loads(labels_file.read_text())
    home_team = data.get("gameHomeTeam", "Home Team")
    away_team = data.get("gameAwayTeam", "Away Team")

    # Map teams to GoalGraph roster
    roster = {
        "teams": {
            "A": {
                "name": home_team,
                "code": home_team[:3].upper(),
                "color": "Home Kit",
                "color_hex": "#e53935",
                "aliases": [home_team.lower(), "home", "home side"]
            },
            "B": {
                "name": away_team,
                "code": away_team[:3].upper(),
                "color": "Away Kit",
                "color_hex": "#1e88e5",
                "aliases": [away_team.lower(), "away", "visitors"]
            }
        },
        "players": {}
    }

    # Map labels to GoalGraph event types
    type_map = {
        "Goal": "goal",
        "Yellow card": "yellow_card",
        "Red card": "red_card",
        "Yellow->red card": "red_card",
        "Foul": "foul",
        "Corner": "corner",
        "Kick-off": "kickoff",
        "Substitution": "substitution",
        "Shots on target": "shot_on_target",
        "Shots off target": "shot_off_target",
        "Offside": "offside"
    }

    events = []
    ev_idx = 1
    for a in data.get("annotations", []):
        lbl = a.get("label")
        if lbl not in type_map:
            continue

        raw_pos = float(a.get("position", 0)) / 1000.0  # seconds in video
        game_time = a.get("gameTime", "1 - 00:00")
        
        # Parse half
        half = 1 if game_time.startswith("1") else 2

        # Team
        team_str = a.get("team")
        team_code = "A" if team_str == "home" else ("B" if team_str == "away" else None)

        events.append({
            "event_id": f"SN{ev_idx:03d}",
            "type": type_map[lbl],
            "t": round(raw_pos, 2),
            "team": team_code,
            "player": None,
            "half": half,
            "game_time": game_time,
            "visibility": a.get("visibility", "visible")
        })
        ev_idx += 1

    # Camera cuts and replays
    replays = []
    if cameras_file.exists():
        cam_data = json.loads(cameras_file.read_text())
        current_replay_start = None
        for c in cam_data.get("annotations", []):
            pos_s = float(c.get("position", 0)) / 1000.0
            is_replay = (c.get("replay") != "real-time")
            if is_replay and current_replay_start is None:
                current_replay_start = pos_s
            elif not is_replay and current_replay_start is not None:
                replays.append({"video": [round(current_replay_start, 2), round(pos_s, 2)]})
                current_replay_start = None

    gt = {
        "match": game_dir.name,
        "home_team": home_team,
        "away_team": away_team,
        "score": data.get("gameScore", "0 - 0"),
        "events": events,
        "replays": replays
    }

    if out_gt_path:
        out_gt_path.parent.mkdir(parents=True, exist_ok=True)
        out_gt_path.write_text(json.dumps(gt, indent=2))
        log.info("Saved GoalGraph GT to: %s", out_gt_path)

    if out_roster_path:
        out_roster_path.parent.mkdir(parents=True, exist_ok=True)
        out_roster_path.write_text(json.dumps(roster, indent=2))
        log.info("Saved GoalGraph Roster to: %s", out_roster_path)

    return gt, roster


def main():
    parser = argparse.ArgumentParser(description="SoccerNet Ingestion & Converter Tool")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # list
    p_list = sub.add_parser("list", help="List SoccerNet matches")
    p_list.add_argument("--league", type=str, default=None, help="Filter league (e.g. epl, laliga, uefa)")
    p_list.add_argument("--split", type=str, default="test", choices=["train", "valid", "test", "challenge"])
    p_list.add_argument("--limit", type=int, default=15)

    # pull
    p_pull = sub.add_parser("pull", help="Download labels or videos for a match")
    p_pull.add_argument("--game", type=str, required=True, help="Full match path (from list)")
    p_pull.add_argument("--video", action="store_true", help="Download 224p video (default: labels only)")
    p_pull.add_argument("--hd", action="store_true", help="Download 720p HD video")
    p_pull.add_argument("--out", type=str, default="data/soccernet")

    # convert
    p_conv = sub.add_parser("convert", help="Convert downloaded match to GoalGraph format")
    p_conv.add_argument("--game-dir", type=str, required=True, help="Path to downloaded match directory")
    p_conv.add_argument("--out-dir", type=str, default=None, help="Output directory for gt.json & roster.json")

    args = parser.parse_args()

    if args.cmd == "list":
        list_matches(league=args.league, split=args.split, limit=args.limit)

    elif args.cmd == "pull":
        files = ["Labels-v2.json", "Labels-cameras.json"]
        if args.hd:
            files.extend(["1_720p.mkv", "2_720p.mkv"])
        elif args.video:
            files.extend(["1_224p.mkv", "2_224p.mkv"])
        pull_match_data(args.game, output_dir=args.out, files=files)

    elif args.cmd == "convert":
        g_dir = Path(args.game_dir)
        out_d = Path(args.out_dir) if args.out_dir else g_dir
        convert_soccernet_to_goalgraph(
            game_dir=g_dir,
            out_gt_path=out_d / "gt.json",
            out_roster_path=out_d / "roster.json"
        )


if __name__ == "__main__":
    main()
