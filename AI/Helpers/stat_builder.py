from collections import defaultdict


def calc_home_cover(opening_spread, home_score, away_score, home_spread_is_positive_favorite=True):
    if opening_spread is None:
        return None

    mov = home_score - away_score

    if home_spread_is_positive_favorite:
        return mov > opening_spread

    return (mov + opening_spread) > 0


def init_team_stats():
    return {
        "games": 0,
        "wins": 0,
        "losses": 0,
        "pts_for": 0,
        "pts_against": 0,
        "point_diff_total": 0,

        "home_games": 0,
        "home_wins": 0,
        "home_pts_for": 0,
        "home_pts_against": 0,

        "away_games": 0,
        "away_wins": 0,
        "away_pts_for": 0,
        "away_pts_against": 0,

        "ats_games": 0,
        "ats_wins": 0,

        "recent_results": [],
        "recent_point_diff": [],
        "recent_pts_for": [],
        "recent_pts_against": [],
        "recent_ats": []
    }


def avg(nums):
    return round(sum(nums) / len(nums), 3) if nums else None


def pct(num, den):
    return round(num / den, 3) if den else None


def summarize_team(team_id, stats):
    recent_n = 5

    return {
        "team_id": team_id,
        "games": stats["games"],
        "wins": stats["wins"],
        "losses": stats["losses"],
        "win_pct": pct(stats["wins"], stats["games"]),
        "avg_pts_for": round(stats["pts_for"] / stats["games"], 3) if stats["games"] else None,
        "avg_pts_against": round(stats["pts_against"] / stats["games"], 3) if stats["games"] else None,
        "avg_point_diff": round(stats["point_diff_total"] / stats["games"], 3) if stats["games"] else None,

        "home_games": stats["home_games"],
        "home_win_pct": pct(stats["home_wins"], stats["home_games"]),
        "home_avg_pts_for": round(stats["home_pts_for"] / stats["home_games"], 3) if stats["home_games"] else None,
        "home_avg_pts_against": round(stats["home_pts_against"] / stats["home_games"], 3) if stats["home_games"] else None,

        "away_games": stats["away_games"],
        "away_win_pct": pct(stats["away_wins"], stats["away_games"]),
        "away_avg_pts_for": round(stats["away_pts_for"] / stats["away_games"], 3) if stats["away_games"] else None,
        "away_avg_pts_against": round(stats["away_pts_against"] / stats["away_games"], 3) if stats["away_games"] else None,

        "ats_games": stats["ats_games"],
        "ats_win_pct": pct(stats["ats_wins"], stats["ats_games"]),

        "last_5_win_pct": pct(sum(stats["recent_results"][-recent_n:]), len(stats["recent_results"][-recent_n:])),
        "last_5_avg_point_diff": avg(stats["recent_point_diff"][-recent_n:]),
        "last_5_avg_pts_for": avg(stats["recent_pts_for"][-recent_n:]),
        "last_5_avg_pts_against": avg(stats["recent_pts_against"][-recent_n:]),
        "last_5_ats_win_pct": pct(sum(stats["recent_ats"][-recent_n:]), len(stats["recent_ats"][-recent_n:])),
    }


def build_team_stats(historical_games, home_spread_is_positive_favorite=True):
    team_stats = defaultdict(init_team_stats)

    for g in historical_games:
        home = g["home_team_id"]
        away = g["away_team_id"]
        hs = g["home_score"]
        aws = g["away_score"]
        spread = g["opening_spread"]

        home_win = hs > aws
        away_win = aws > hs

        home_cover = calc_home_cover(
            spread,
            hs,
            aws,
            home_spread_is_positive_favorite=home_spread_is_positive_favorite
        )
        away_cover = None if home_cover is None else (not home_cover)

        home_stats = team_stats[home]
        home_stats["games"] += 1
        home_stats["wins"] += 1 if home_win else 0
        home_stats["losses"] += 0 if home_win else 1
        home_stats["pts_for"] += hs
        home_stats["pts_against"] += aws
        home_stats["point_diff_total"] += (hs - aws)
        home_stats["home_games"] += 1
        home_stats["home_wins"] += 1 if home_win else 0
        home_stats["home_pts_for"] += hs
        home_stats["home_pts_against"] += aws
        home_stats["recent_results"].append(1 if home_win else 0)
        home_stats["recent_point_diff"].append(hs - aws)
        home_stats["recent_pts_for"].append(hs)
        home_stats["recent_pts_against"].append(aws)

        if home_cover is not None:
            home_stats["ats_games"] += 1
            home_stats["ats_wins"] += 1 if home_cover else 0
            home_stats["recent_ats"].append(1 if home_cover else 0)

        away_stats = team_stats[away]
        away_stats["games"] += 1
        away_stats["wins"] += 1 if away_win else 0
        away_stats["losses"] += 0 if away_win else 1
        away_stats["pts_for"] += aws
        away_stats["pts_against"] += hs
        away_stats["point_diff_total"] += (aws - hs)
        away_stats["away_games"] += 1
        away_stats["away_wins"] += 1 if away_win else 0
        away_stats["away_pts_for"] += aws
        away_stats["away_pts_against"] += hs
        away_stats["recent_results"].append(1 if away_win else 0)
        away_stats["recent_point_diff"].append(aws - hs)
        away_stats["recent_pts_for"].append(aws)
        away_stats["recent_pts_against"].append(hs)

        if away_cover is not None:
            away_stats["ats_games"] += 1
            away_stats["ats_wins"] += 1 if away_cover else 0
            away_stats["recent_ats"].append(1 if away_cover else 0)

    return team_stats


def build_matchup_payload_from_api_games(api_games, team_stats, source_team_map):
    payload = []

    for g in api_games:
        home_source_id = g["home_source_team_id"]
        away_source_id = g["away_source_team_id"]

        home_team = source_team_map.get(home_source_id)
        away_team = source_team_map.get(away_source_id)

        if not home_team or not away_team:
            print(f"Skipping game because team mapping was not found: {g['away_team_name']} at {g['home_team_name']}")
            continue

        payload.append({
            "game_id": int(g["nba_game_id"]) if g["nba_game_id"] else None,
            "matchup": f"{away_team['team_name']} vs {home_team['team_name']}",
            "opening_spread": g.get("opening_spread"),
            "current_spread": g.get("current_spread"),
            "game_status": g["game_status"]
        })

    return payload