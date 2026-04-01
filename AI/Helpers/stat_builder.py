from collections import defaultdict

from Helpers.l1_live_features import (
    build_l1_model_features_subset,
    get_l1_allowlist_from_env,
)


def calc_home_cover(opening_spread, home_score, away_score):
    if opening_spread is None:
        return None

    mov = home_score - away_score
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


def nz(value, default=0.0):
    return default if value is None else value


def calc_site_point_diff(team_summary, is_home):
    if is_home:
        pf = nz(team_summary.get("home_avg_pts_for"))
        pa = nz(team_summary.get("home_avg_pts_against"))
    else:
        pf = nz(team_summary.get("away_avg_pts_for"))
        pa = nz(team_summary.get("away_avg_pts_against"))

    return pf - pa


def _get_strength_inputs(home_summary, away_summary):
    home_season_pd = nz(home_summary.get("avg_point_diff"))
    away_season_pd = nz(away_summary.get("avg_point_diff"))

    home_site_pd = calc_site_point_diff(home_summary, is_home=True)
    away_site_pd = calc_site_point_diff(away_summary, is_home=False)

    home_recent_pd = nz(home_summary.get("last_5_avg_point_diff"))
    away_recent_pd = nz(away_summary.get("last_5_avg_point_diff"))

    return {
        "home_season_pd": home_season_pd,
        "away_season_pd": away_season_pd,
        "home_site_pd": home_site_pd,
        "away_site_pd": away_site_pd,
        "home_recent_pd": home_recent_pd,
        "away_recent_pd": away_recent_pd
    }


def calculate_estimated_edge(home_summary, away_summary, market_home_spread):
    if market_home_spread is None:
        return {
            "projected_home_margin": None,
            "fair_home_spread": None,
            "estimated_edge_points": None,
            "edge_side": "PASS"
        }

    vals = _get_strength_inputs(home_summary, away_summary)

    home_strength = (
        0.50 * vals["home_season_pd"] +
        0.30 * vals["home_site_pd"] +
        0.20 * vals["home_recent_pd"]
    )

    away_strength = (
        0.50 * vals["away_season_pd"] +
        0.30 * vals["away_site_pd"] +
        0.20 * vals["away_recent_pd"]
    )

    home_court_advantage = 1.5

    projected_home_margin = round((home_strength - away_strength) + home_court_advantage, 3)
    fair_home_spread = round(-projected_home_margin, 3)

    spread_diff = round(fair_home_spread - market_home_spread, 3)
    estimated_edge_points = round(abs(spread_diff), 3)

    if -1.5 <= spread_diff <= 1.5:
        edge_side = "PASS"
    elif spread_diff < -1.5:
        edge_side = "HOME_SPREAD"
    elif spread_diff > 1.5:
        edge_side = "AWAY_SPREAD"
    else:
        edge_side = "PASS"

    return {
        "projected_home_margin": projected_home_margin,
        "fair_home_spread": fair_home_spread,
        "estimated_edge_points": estimated_edge_points,
        "edge_side": edge_side
    }


def calculate_estimated_edge_v2(home_summary, away_summary, market_home_spread):
    if market_home_spread is None:
        return {
            "projected_home_margin": None,
            "fair_home_spread": None,
            "spread_diff": None,
            "adjusted_spread_diff": None,
            "estimated_edge_points": None,
            "edge_side": "PASS"
        }

    vals = _get_strength_inputs(home_summary, away_summary)

    home_strength = (
        0.60 * vals["home_season_pd"] +
        0.25 * vals["home_site_pd"] +
        0.15 * vals["home_recent_pd"]
    )

    away_strength = (
        0.60 * vals["away_season_pd"] +
        0.25 * vals["away_site_pd"] +
        0.15 * vals["away_recent_pd"]
    )

    home_court_advantage = 1.5

    raw_projected_home_margin = (home_strength - away_strength) + home_court_advantage
    projected_home_margin = round(raw_projected_home_margin * 1.10, 3)
    fair_home_spread = round(-projected_home_margin, 3)

    spread_diff = round(fair_home_spread - market_home_spread, 3)

    adjusted_spread_diff = spread_diff
    if abs(market_home_spread) >= 10:
        adjusted_spread_diff *= 0.80
    elif abs(market_home_spread) >= 7:
        adjusted_spread_diff *= 0.90

    adjusted_spread_diff = round(adjusted_spread_diff, 3)
    estimated_edge_points = round(abs(adjusted_spread_diff), 3)

    threshold = 2.5
    if abs(market_home_spread) >= 10:
        threshold = 3.5

    if abs(adjusted_spread_diff) <= threshold:
        edge_side = "PASS"
    elif adjusted_spread_diff < 0:
        edge_side = "HOME_SPREAD"
    else:
        edge_side = "AWAY_SPREAD"

    return {
        "projected_home_margin": projected_home_margin,
        "fair_home_spread": fair_home_spread,
        "spread_diff": spread_diff,
        "adjusted_spread_diff": adjusted_spread_diff,
        "estimated_edge_points": estimated_edge_points,
        "edge_side": edge_side
    }

def calculate_estimated_edge_v3(home_summary, away_summary, market_home_spread):
    if market_home_spread is None:
        return {
            "projected_home_margin": None,
            "fair_home_spread": None,
            "spread_diff": None,
            "estimated_edge_points": None,
            "edge_side": "PASS"
        }

    vals = _get_strength_inputs(home_summary, away_summary)

    home_strength = (
        0.60 * vals["home_season_pd"] +
        0.25 * vals["home_site_pd"] +
        0.15 * vals["home_recent_pd"]
    )

    away_strength = (
        0.60 * vals["away_season_pd"] +
        0.25 * vals["away_site_pd"] +
        0.15 * vals["away_recent_pd"]
    )

    home_court_advantage = 1.5

    raw_projected_home_margin = (home_strength - away_strength) + home_court_advantage

    # calibration placeholder — tune from backtest
    calibrated_home_margin = round(raw_projected_home_margin * 1.10, 3)

    fair_home_spread = round(-calibrated_home_margin, 3)
    spread_diff = round(fair_home_spread - market_home_spread, 3)
    estimated_edge_points = round(abs(spread_diff), 3)

    if abs(spread_diff) <= 2.5:
        edge_side = "PASS"
    elif spread_diff < 0:
        edge_side = "HOME_SPREAD"
    else:
        edge_side = "AWAY_SPREAD"

    return {
        "projected_home_margin": calibrated_home_margin,
        "fair_home_spread": fair_home_spread,
        "spread_diff": spread_diff,
        "estimated_edge_points": estimated_edge_points,
        "edge_side": edge_side
    }


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


def build_team_stats(historical_games):
    team_stats = defaultdict(init_team_stats)

    for g in historical_games:
        home = g["home_team_id"]
        away = g["away_team_id"]
        hs = g["home_score"]
        aws = g["away_score"]
        spread = g["opening_spread"]

        home_win = hs > aws
        away_win = aws > hs

        home_cover = calc_home_cover(spread, hs, aws)
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


def build_head_to_head_stats(home_team_id, away_team_id, historical_games):
    h2h_games = []

    for g in historical_games:
        g_home = g["home_team_id"]
        g_away = g["away_team_id"]

        teams_match = (
            (g_home == home_team_id and g_away == away_team_id) or
            (g_home == away_team_id and g_away == home_team_id)
        )

        if teams_match:
            h2h_games.append(g)

    if not h2h_games:
        return {
            "games": 0,
            "home_team_wins": 0,
            "away_team_wins": 0,
            "home_team_win_pct": None,
            "away_team_win_pct": None,
            "home_team_avg_margin": None,
            "away_team_avg_margin": None,
            "home_team_ats_wins": 0,
            "away_team_ats_wins": 0,
            "home_team_ats_win_pct": None,
            "away_team_ats_win_pct": None,
            "last_5_matchups": []
        }

    home_team_wins = 0
    away_team_wins = 0
    home_team_ats_wins = 0
    away_team_ats_wins = 0
    margins_for_home_team = []
    last_5_matchups = []

    for g in h2h_games:
        g_home = g["home_team_id"]
        g_away = g["away_team_id"]
        hs = g["home_score"]
        aws = g["away_score"]
        spread = g["opening_spread"]

        if g_home == home_team_id and g_away == away_team_id:
            margin_for_home_team = hs - aws
            home_team_won = hs > aws

            home_cover = calc_home_cover(spread, hs, aws)  
            away_cover = None if home_cover is None else (not home_cover)

            h2h_home_cover = home_cover
            h2h_away_cover = away_cover

        else:
            margin_for_home_team = aws - hs
            home_team_won = aws > hs

            actual_home_cover = calc_home_cover(spread, hs, aws)
            actual_away_cover = None if actual_home_cover is None else (not actual_home_cover)

            h2h_home_cover = actual_away_cover
            h2h_away_cover = actual_home_cover

        margins_for_home_team.append(margin_for_home_team)

        if home_team_won:
            home_team_wins += 1
        else:
            away_team_wins += 1

        if h2h_home_cover is True:
            home_team_ats_wins += 1
        if h2h_away_cover is True:
            away_team_ats_wins += 1

        last_5_matchups.append({
            "margin_for_home_team": margin_for_home_team,
            "home_team_won": home_team_won,
            "home_team_covered": h2h_home_cover,
            "away_team_covered": h2h_away_cover
        })

    last_5_matchups = last_5_matchups[-5:]

    games = len(h2h_games)

    return {
        "games": games,
        "home_team_wins": home_team_wins,
        "away_team_wins": away_team_wins,
        "home_team_win_pct": pct(home_team_wins, games),
        "away_team_win_pct": pct(away_team_wins, games),
        "home_team_avg_margin": avg(margins_for_home_team),
        "away_team_avg_margin": round(-avg(margins_for_home_team), 3) if margins_for_home_team else None,
        "home_team_ats_wins": home_team_ats_wins,
        "away_team_ats_wins": away_team_ats_wins,
        "home_team_ats_win_pct": pct(home_team_ats_wins, games),
        "away_team_ats_win_pct": pct(away_team_ats_wins, games),
        "last_5_matchups": last_5_matchups
    }


def build_matchup_payload_from_api_games(api_games, team_stats, source_team_map, historical_games):
    payload = []

    for g in api_games:
        home_source_id = g["home_source_team_id"]
        away_source_id = g["away_source_team_id"]

        home_team = source_team_map.get(home_source_id)
        away_team = source_team_map.get(away_source_id)

        if not home_team or not away_team:
            print(f"Skipping game because team mapping was not found: {g['away_team_name']} at {g['home_team_name']}")
            continue

        home_team_id = home_team["team_id"]
        away_team_id = away_team["team_id"]

        home_summary = summarize_team(home_team_id, team_stats[home_team_id])
        away_summary = summarize_team(away_team_id, team_stats[away_team_id])

        edge_info_v1 = calculate_estimated_edge(
            home_summary,
            away_summary,
            g.get("home_current_spread")
        )

        edge_info_v2 = calculate_estimated_edge_v2(
            home_summary,
            away_summary,
            g.get("home_current_spread")
        )

        edge_info_v3 = calculate_estimated_edge_v3(
            home_summary,
            away_summary,
            g.get("home_current_spread")
        )

        #which one to use as primary edge info sent to AI
        active_edge_info = edge_info_v2

        h2h_stats = build_head_to_head_stats(
            home_team_id,
            away_team_id,
            historical_games
        )

        game_entry = {
            "game_id": int(g["nba_game_id"]) if g["nba_game_id"] else None,
            "matchup": f"{away_team['team_name']} @ {home_team['team_name']}",
            "game_status": g["game_status"],

            "home_team_name": home_team["team_name"],
            "away_team_name": away_team["team_name"],

            "home_opening_spread": g.get("home_opening_spread"),
            "away_opening_spread": g.get("away_opening_spread"),
            "home_current_spread": g.get("home_current_spread"),
            "away_current_spread": g.get("away_current_spread"),

            "opening_spread": g.get("opening_spread"),
            "current_spread": g.get("current_spread"),

            "home_team_id": home_team_id,
            "away_team_id": away_team_id,

            "home_stats": home_summary,
            "away_stats": away_summary,
            "head_to_head_stats": h2h_stats,

            # fields AI currently expects
            "projected_home_margin": active_edge_info["projected_home_margin"],
            "fair_home_spread": active_edge_info["fair_home_spread"],
            "estimated_edge_points": active_edge_info["estimated_edge_points"],
            "edge_side": active_edge_info["edge_side"],

            # keep both for testing
            "edge_model_v1": edge_info_v1,
            "edge_model_v2": edge_info_v2,
            "edge_model_v3": edge_info_v3,
        }

        # Optional: JSON allow-list from L1 (env AI_L1_FEATURES_JSON) — same names as training
        l1_allow = get_l1_allowlist_from_env()
        if l1_allow:
            game_entry["l1_model_features"] = build_l1_model_features_subset(
                l1_allow,
                home_team_id,
                away_team_id,
                historical_games,
                g,
            )

        payload.append(game_entry)

    return payload