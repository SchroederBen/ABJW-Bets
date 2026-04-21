from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from Helpers.l1_live_features import (
    build_l1_model_features_subset,
    build_full_feature_row,
    get_l1_allowlist_from_env,
    score_with_l1_model,
)
from Helpers.cover_model_live import (
    load_cover_model_and_scaler,
    score_with_cover_model,
)

# ======================================================================
# Trained stat_builder weights loader
# ======================================================================

_ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "src" / "models" / "artifacts"
_cached_trained_weights = None
_trained_weights_load_attempted = False


def _load_trained_weights() -> dict | None:
    """
    Load the latest trained stat_builder_weights artifact JSON.
    Returns the full bundle dict, or None if not found.
    Caches after first load.
    """
    global _cached_trained_weights, _trained_weights_load_attempted

    if _trained_weights_load_attempted:
        return _cached_trained_weights

    _trained_weights_load_attempted = True

    if not _ARTIFACTS_DIR.is_dir():
        return None

    candidates = sorted(_ARTIFACTS_DIR.glob("stat_builder_weights_struct_*.json"))
    if not candidates:
        print("  No trained stat_builder weights found — using hardcoded v2 weights.")
        return None

    path = candidates[-1]
    try:
        with open(path, encoding="utf-8") as f:
            _cached_trained_weights = json.load(f)
        print(f"  Loaded trained stat_builder weights: {path.name}")
        sb = _cached_trained_weights.get("stat_builder", {})
        print(
            f"    w_season={sb.get('w_season', '?'):.4f}  "
            f"w_site={sb.get('w_site', '?'):.4f}  "
            f"w_recent={sb.get('w_recent', '?'):.4f}  "
            f"HCA={sb.get('home_court_advantage', '?'):.4f}  "
            f"mult={sb.get('margin_mult', '?'):.4f}"
        )
    except Exception as e:
        print(f"  Warning: Could not load trained weights: {e}")
        _cached_trained_weights = None

    return _cached_trained_weights


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


def calculate_estimated_edge_learned(home_summary, away_summary, market_home_spread):
    """
    Edge calculation using trained weights from train_stat_builder_weights.

    Uses optimized blend weights, home court advantage, and margin multiplier
    learned via L-BFGS-B minimization of log-loss on historical ATS data.
    Includes a logistic head that maps projected margin + spread to P(home covers),
    plus a learned tau threshold for PASS decisions.

    Falls back to v2 (hardcoded) if no trained weights artifact is found.
    """
    trained = _load_trained_weights()

    if trained is None or market_home_spread is None:
        # Fallback: use v2 if no trained weights
        if market_home_spread is None:
            return {
                "projected_home_margin": None,
                "fair_home_spread": None,
                "spread_diff": None,
                "p_home_cover": None,
                "estimated_edge_points": None,
                "edge_side": "PASS",
                "model_version": "fallback_v2",
            }
        return {
            **calculate_estimated_edge_v2(home_summary, away_summary, market_home_spread),
            "p_home_cover": None,
            "model_version": "fallback_v2",
        }

    sb = trained.get("stat_builder", {})
    logit_head = trained.get("logit_head", {})
    mapping = trained.get("mapping", {})

    w_season = sb.get("w_season", 0.60)
    w_site = sb.get("w_site", 0.25)
    w_recent = sb.get("w_recent", 0.15)
    hca = sb.get("home_court_advantage", 1.5)
    margin_mult = sb.get("margin_mult", 1.10)
    logit_scale = logit_head.get("scale", 0.11)
    logit_bias = logit_head.get("bias", 0.0)
    tau = mapping.get("tau", 0.05)
    k_edge_points = mapping.get("k_edge_points", 20.0)

    vals = _get_strength_inputs(home_summary, away_summary)

    home_strength = (
        w_season * vals["home_season_pd"] +
        w_site * vals["home_site_pd"] +
        w_recent * vals["home_recent_pd"]
    )

    away_strength = (
        w_season * vals["away_season_pd"] +
        w_site * vals["away_site_pd"] +
        w_recent * vals["away_recent_pd"]
    )

    inner_diff = home_strength - away_strength
    projected_home_margin = round(margin_mult * (inner_diff + hca), 3)
    fair_home_spread = round(-projected_home_margin, 3)
    spread_diff = round(fair_home_spread - market_home_spread, 3)

    # Logistic head: p(home covers) = sigmoid(bias + scale * (margin + spread))
    logit = logit_bias + logit_scale * (projected_home_margin + market_home_spread)
    logit = max(-60.0, min(60.0, logit))
    p_home_cover = 1.0 / (1.0 + math.exp(-logit))

    # Edge points from probability
    edge_prob = abs(p_home_cover - 0.5)
    estimated_edge_points = round(edge_prob * k_edge_points, 3)

    # Decision using learned tau
    if p_home_cover >= (0.5 + tau):
        edge_side = "HOME_SPREAD"
    elif p_home_cover <= (0.5 - tau):
        edge_side = "AWAY_SPREAD"
    else:
        edge_side = "PASS"

    return {
        "projected_home_margin": projected_home_margin,
        "fair_home_spread": fair_home_spread,
        "spread_diff": spread_diff,
        "p_home_cover": round(p_home_cover, 4),
        "estimated_edge_points": estimated_edge_points,
        "edge_side": edge_side,
        "model_version": trained.get("model_version", "trained"),
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


def build_matchup_payload_from_api_games(
    api_games,
    team_stats,
    source_team_map,
    historical_games,
    team_game_rows=None,
):
    """
    Build the matchup payload sent to the AI.

    Parameters
    ----------
    api_games : list[dict]
        Today's games from the NBA API.
    team_stats : dict
        Aggregated season stats per team_id (from build_team_stats).
    source_team_map : dict
        Mapping from source_team_id -> {team_id, team_name, team_abbrev}.
    historical_games : list[dict]
        Score-only game rows (fallback for features).
    team_game_rows : list[dict] | None
        Box-score rows from fetch_historical_games_with_box_scores.
        When provided, enables full L1 feature computation (FG%, 3PT%, etc.).
    """
    payload = []

    # Load L1 allow-list and attempt to load the trained L1 model
    l1_allow = get_l1_allowlist_from_env()

    # Pre-load the L1 and cover models so we only print load messages once
    from Helpers.l1_live_features import load_l1_model_and_scaler
    l1_model, l1_scaler, l1_feature_cols = load_l1_model_and_scaler()
    cover_model, cover_scaler, cover_feature_cols = load_cover_model_and_scaler()

    has_box_scores = team_game_rows is not None and len(team_game_rows) > 0
    if has_box_scores:
        print(f"  Box-score rows available: {len(team_game_rows)} team-game rows")
    else:
        print("  No box-score rows — L1 features will use score-only fallback")

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

        # Edge calculations — all versions for comparison
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

        # Learned edge (uses trained weights if available, else falls back to v2)
        edge_info_learned = calculate_estimated_edge_learned(
            home_summary,
            away_summary,
            g.get("home_current_spread")
        )

        # Use learned edge as primary if trained weights loaded, else v2
        if edge_info_learned.get("model_version", "").startswith("stat_builder_weights"):
            active_edge_info = edge_info_learned
        else:
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

            # Primary edge fields sent to AI
            "projected_home_margin": active_edge_info["projected_home_margin"],
            "fair_home_spread": active_edge_info["fair_home_spread"],
            "estimated_edge_points": active_edge_info["estimated_edge_points"],
            "edge_side": active_edge_info["edge_side"],

            # Keep all versions for testing / comparison
            "edge_model_v1": edge_info_v1,
            "edge_model_v2": edge_info_v2,
            "edge_model_v3": edge_info_v3,
            "edge_model_learned": edge_info_learned,
        }

        # P(home covers) from the learned model
        if edge_info_learned.get("p_home_cover") is not None:
            game_entry["p_home_cover"] = edge_info_learned["p_home_cover"]
            game_entry["edge_model_version"] = edge_info_learned.get("model_version", "unknown")

        # ---- L1 model features (allow-list subset for AI context) ----
        if l1_allow:
            l1_features = build_l1_model_features_subset(
                l1_allow,
                home_team_id,
                away_team_id,
                team_game_rows if has_box_scores else None,
                historical_games,
                g,
            )

            null_feature_count = sum(1 for value in l1_features.values() if value is None)

            game_entry["l1_model_features"] = l1_features
            game_entry["l1_model_features_meta"] = {
                "enabled": True,
                "allowlist_size": len(l1_allow),
                "computed_feature_count": len(l1_features),
                "null_feature_count": null_feature_count,
                "data_source": "box_scores" if has_box_scores else "scores_only",
            }
        else:
            game_entry["l1_model_features_meta"] = {
                "enabled": False,
                "allowlist_size": 0,
                "computed_feature_count": 0,
                "null_feature_count": 0,
                "data_source": "none",
            }

        # ---- Model scoring (L1 win + cover models share the same pregame row) ----
        if l1_model is not None or cover_model is not None:
            full_row = build_full_feature_row(
                home_team_id,
                away_team_id,
                team_game_rows if has_box_scores else None,
                historical_games,
                g,
            )
        else:
            full_row = None

        if l1_model is not None and full_row is not None:
            l1_score = score_with_l1_model(full_row)
            game_entry["l1_model_score"] = (
                l1_score if l1_score is not None else {"l1_model_available": False}
            )
        else:
            game_entry["l1_model_score"] = {"l1_model_available": False}

        if cover_model is not None and full_row is not None:
            cover_score = score_with_cover_model(full_row)
            game_entry["cover_model_score"] = (
                cover_score if cover_score is not None else {"cover_model_available": False}
            )
        else:
            game_entry["cover_model_score"] = {"cover_model_available": False}

        payload.append(game_entry)

    return payload