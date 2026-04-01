from collections import defaultdict


# =========================================================
# BASIC HELPERS
# =========================================================

def calc_home_cover(opening_spread, home_score, away_score):
    """
    Returns True if the home team covered the opening spread,
    False if the away team covered, or None if no spread exists.

    Example:
    home_score = 110, away_score = 100, opening_spread = -7.5
    home margin = 10
    10 + (-7.5) = 2.5 > 0, so home covered
    """
    if opening_spread is None:
        return None

    mov = home_score - away_score
    return (mov + opening_spread) > 0


def avg(nums):
    """Safe average with rounding. Returns None for empty list."""
    return round(sum(nums) / len(nums), 3) if nums else None


def pct(num, den):
    """Safe percentage helper. Returns None when denominator is 0."""
    return round(num / den, 3) if den else None


def nz(value, default=0.0):
    """Converts None to a numeric default."""
    return default if value is None else value


# =========================================================
# TEAM STATS STRUCTURE
# =========================================================

def init_team_stats():
    """
    Creates the default stats container for a team.
    This stores season totals, home/away splits, ATS data,
    and rolling recent data used later for summaries.
    """
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


# =========================================================
# SUMMARY / FEATURE BUILDERS
# =========================================================

def calc_raw_site_point_diff(team_summary, is_home):
    """
    Calculates raw point differential for home or away games only.

    home version = home_avg_pts_for - home_avg_pts_against
    away version = away_avg_pts_for - away_avg_pts_against
    """
    if is_home:
        pf = nz(team_summary.get("home_avg_pts_for"))
        pa = nz(team_summary.get("home_avg_pts_against"))
    else:
        pf = nz(team_summary.get("away_avg_pts_for"))
        pa = nz(team_summary.get("away_avg_pts_against"))

    return pf - pa


def calc_shrunk_site_point_diff(team_summary, is_home, shrink_games=10):
    """
    Shrinks noisy home/away splits toward overall season point diff.

    Why:
    A team's home-only or away-only sample can be noisy, especially if the
    sample size is small. This function blends the site split with the full
    season average so we do not overreact.

    shrink_games:
    - bigger value = more conservative, more regression toward season average
    - smaller value = trusts site-specific split more quickly
    """
    season_pd = nz(team_summary.get("avg_point_diff"))
    raw_site_pd = calc_raw_site_point_diff(team_summary, is_home)

    games = team_summary.get("home_games", 0) if is_home else team_summary.get("away_games", 0)
    weight = games / (games + shrink_games) if (games + shrink_games) > 0 else 0

    # Blend site split with season average
    shrunk_site_pd = (weight * raw_site_pd) + ((1 - weight) * season_pd)
    return round(shrunk_site_pd, 3)


def get_recent_window(stats, preferred=8):
    """
    Chooses a recent window size for summary stats.

    Using 8 instead of 5 makes recent form less noisy, but still responsive.
    If fewer games exist, it uses what is available.
    """
    return min(preferred, len(stats["recent_results"])) if stats["recent_results"] else 0


def summarize_team(team_id, stats):
    """
    Converts the running team stats into a cleaner summary object
    used by the prediction model and payload output.
    """
    recent_n = get_recent_window(stats, preferred=8)

    recent_results = stats["recent_results"][-recent_n:] if recent_n else []
    recent_pd = stats["recent_point_diff"][-recent_n:] if recent_n else []
    recent_pf = stats["recent_pts_for"][-recent_n:] if recent_n else []
    recent_pa = stats["recent_pts_against"][-recent_n:] if recent_n else []
    recent_ats = stats["recent_ats"][-recent_n:] if recent_n else []

    summary = {
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

        "recent_sample_size": recent_n,
        "last_n_win_pct": pct(sum(recent_results), len(recent_results)) if recent_results else None,
        "last_n_avg_point_diff": avg(recent_pd),
        "last_n_avg_pts_for": avg(recent_pf),
        "last_n_avg_pts_against": avg(recent_pa),
        "last_n_ats_win_pct": pct(sum(recent_ats), len(recent_ats)) if recent_ats else None,
    }

    # Add shrunk site features directly into the summary so the model can use them
    summary["shrunk_home_site_point_diff"] = calc_shrunk_site_point_diff(summary, is_home=True)
    summary["shrunk_away_site_point_diff"] = calc_shrunk_site_point_diff(summary, is_home=False)

    return summary


def _get_strength_inputs(home_summary, away_summary):
    """
    Pulls the core strength inputs used by the betting model.

    Main idea:
    - season point diff = strongest base signal
    - shrunk site point diff = useful, but safer than raw split
    - recent point diff = small form adjustment only
    """
    return {
        "home_season_pd": nz(home_summary.get("avg_point_diff")),
        "away_season_pd": nz(away_summary.get("avg_point_diff")),

        "home_site_pd": nz(home_summary.get("shrunk_home_site_point_diff")),
        "away_site_pd": nz(away_summary.get("shrunk_away_site_point_diff")),

        "home_recent_pd": nz(home_summary.get("last_n_avg_point_diff")),
        "away_recent_pd": nz(away_summary.get("last_n_avg_point_diff")),
    }


# =========================================================
# EDGE MODEL
# =========================================================

def calculate_estimated_edge(home_summary, away_summary, market_home_spread):
    """
    =========================================================
    EDGE MODEL V1
    =========================================================

    PURPOSE:
    This is the simpler / baseline spread model.

    It tries to answer:
    "Based on each team's season strength, site-specific strength,
    and recent form, what should the home spread be?"

    Then it compares that model-implied spread to the sportsbook's
    current home spread to decide whether there is value on:
        - the home team spread
        - the away team spread
        - or no bet / PASS

    IMPORTANT SIGN CONVENTION:
    - A NEGATIVE home spread means the home team is favored
      Example: -5.5 means home is favored by 5.5
    - A POSITIVE home spread means the home team is the underdog
      Example: +5.5 means home is getting 5.5

    HOW THIS MODEL WORKS:
    1. Build a "strength score" for the home team
    2. Build a "strength score" for the away team
    3. Compare those strengths and add home-court advantage
    4. Convert projected scoring margin into a fair spread
    5. Compare fair spread to the market spread
    6. If the gap is large enough, pick a side; otherwise PASS

    V1 TRAITS:
    - More reactive / simpler
    - Uses one fixed PASS threshold
    - No large-spread caution
    - Good as a benchmark model
    """

    # If there is no market spread available, we cannot compare our model
    # to the sportsbook number, so we must PASS.
    if market_home_spread is None:
        return {
            "projected_home_margin": None,
            "fair_home_spread": None,
            "estimated_edge_points": None,
            "edge_side": "PASS"
        }

    # Pull the core strength ingredients for each team.
    #
    # These values typically include:
    # - season point differential
    # - site-specific point differential
    #   (home-only for home team, away-only for away team)
    # - recent point differential
    #
    # Example:
    # vals["home_season_pd"] might be +4.2
    # vals["away_season_pd"] might be -1.8
    vals = _get_strength_inputs(home_summary, away_summary)

    # -----------------------------------------------------
    # HOME TEAM STRENGTH SCORE
    # -----------------------------------------------------
    # This is a weighted blend of 3 ideas:
    #
    # 1. Season point differential (50%)
    #    The biggest base indicator of overall team quality.
    #
    # 2. Site-specific point differential (30%)
    #    How the home team performs specifically at home.
    #
    # 3. Recent point differential (20%)
    #    Last-5-ish form. Gives the model some recency awareness.
    #
    # Example:
    # If home season PD = 6
    #    home site PD   = 8
    #    home recent PD = 2
    #
    # home_strength = (0.50*6) + (0.30*8) + (0.20*2)
    #               = 3.0 + 2.4 + 0.4 = 5.8
    home_strength = (
        0.50 * vals["home_season_pd"] +
        0.30 * vals["home_site_pd"] +
        0.20 * vals["home_recent_pd"]
    )

    # -----------------------------------------------------
    # AWAY TEAM STRENGTH SCORE
    # -----------------------------------------------------
    # Same idea as the home team, but now:
    # - season PD is full-season average
    # - site PD is AWAY-only performance
    # - recent PD is recent form
    #
    # Example:
    # If away season PD = 3
    #    away site PD   = 1
    #    away recent PD = 5
    #
    # away_strength = (0.50*3) + (0.30*1) + (0.20*5)
    #               = 1.5 + 0.3 + 1.0 = 2.8
    away_strength = (
        0.50 * vals["away_season_pd"] +
        0.30 * vals["away_site_pd"] +
        0.20 * vals["away_recent_pd"]
    )

    # Flat home-court advantage.
    #
    # This is an extra points bonus given to the home team simply
    # for playing at home.
    #
    # A value of 1.5 means:
    # even if the teams were otherwise equal, the home team would be
    # projected about 1.5 points better in this game.
    home_court_advantage = 1.5

    # -----------------------------------------------------
    # PROJECTED HOME MARGIN
    # -----------------------------------------------------
    # This is the model's estimated scoring margin from the HOME TEAM'S
    # perspective.
    #
    # Formula:
    # projected_home_margin =
    #     (home_strength - away_strength) + home_court_advantage
    #
    # Interpretation:
    # - Positive result -> home team should win by that many
    # - Negative result -> away team should win by that many
    #
    # Example:
    # home_strength = 5.8
    # away_strength = 2.8
    # HCA = 1.5
    #
    # projected_home_margin = (5.8 - 2.8) + 1.5 = 4.5
    #
    # Meaning:
    # model thinks home should win by about 4.5 points.
    projected_home_margin = round((home_strength - away_strength) + home_court_advantage, 3)

    # -----------------------------------------------------
    # FAIR HOME SPREAD
    # -----------------------------------------------------
    # Sportsbooks express point spreads opposite the projected margin:
    #
    # If home should win by 4.5, fair home spread = -4.5
    # If home should lose by 2.0, fair home spread = +2.0
    #
    # So we convert projected margin to spread by flipping the sign.
    fair_home_spread = round(-projected_home_margin, 3)

    # -----------------------------------------------------
    # SPREAD DIFFERENCE
    # -----------------------------------------------------
    # This compares OUR model line to the MARKET line.
    #
    # spread_diff = fair_home_spread - market_home_spread
    #
    # How to interpret:
    #
    # If spread_diff is NEGATIVE:
    #   Our fair line is more favorable to the home team than the market.
    #   That means HOME SPREAD may have value.
    #
    # If spread_diff is POSITIVE:
    #   Market is asking too much from the home team.
    #   That means AWAY SPREAD may have value.
    #
    # Example 1:
    # fair_home_spread = -7
    # market_home_spread = -4
    # spread_diff = -7 - (-4) = -3
    # -> home spread value
    #
    # Example 2:
    # fair_home_spread = -7
    # market_home_spread = -11
    # spread_diff = -7 - (-11) = +4
    # -> away spread value
    spread_diff = round(fair_home_spread - market_home_spread, 3)

    # Absolute size of the disagreement between our model and the market.
    #
    # This ignores direction and only answers:
    # "How many points apart are we from the sportsbook?"
    #
    # Example:
    # spread_diff = -3.2 -> edge size is 3.2
    # spread_diff = +3.2 -> edge size is also 3.2
    estimated_edge_points = round(abs(spread_diff), 3)

    # -----------------------------------------------------
    # DECISION RULES
    # -----------------------------------------------------
    # V1 uses a simple fixed no-bet zone:
    #
    # - If spread_diff is between -1.5 and +1.5, there is not enough
    #   separation from the market -> PASS
    #
    # - If spread_diff < -1.5:
    #   We think the home team should be laying MORE points than the market
    #   says, so the home spread has value.
    #
    # - If spread_diff > +1.5:
    #   We think the market number is too high on the home side,
    #   so the away spread has value.
    if -1.5 <= spread_diff <= 1.5:
        edge_side = "PASS"
    elif spread_diff < -1.5:
        edge_side = "HOME_SPREAD"
    elif spread_diff > 1.5:
        edge_side = "AWAY_SPREAD"
    else:
        # Safety fallback; normally should never be reached.
        edge_side = "PASS"

    # Return the final core outputs from the model.
    return {
        # Model's predicted scoring margin for the home team
        "projected_home_margin": projected_home_margin,

        # The spread we think would be "fair" for the home team
        "fair_home_spread": fair_home_spread,

        # Absolute gap between our fair line and the market line
        "estimated_edge_points": estimated_edge_points,

        # Final directional result:
        # HOME_SPREAD / AWAY_SPREAD / PASS
        "edge_side": edge_side
    }


def calculate_estimated_edge_v2(home_summary, away_summary, market_home_spread):
    """
    =========================================================
    EDGE MODEL V2
    =========================================================

    PURPOSE:
    This is the more refined version of the model and the one you said
    backtested best.

    It still uses the same overall structure as v1:
    - build team strengths
    - project home margin
    - convert to fair spread
    - compare against market
    - decide side

    BUT it differs in important ways:
    1. It puts more weight on season performance
    2. It puts less weight on recent form
    3. It boosts projected margin by 10%
    4. It reduces edge size on large market spreads
    5. It uses a bigger PASS threshold on very large spreads

    WHY V2 MAY HAVE TESTED BETTER:
    - It is less reactive to short-term noise
    - It trusts the full-season sample more
    - It is more cautious on huge lines where underdogs often look
      tempting but the market may still be mostly correct
    """

    # If there is no market spread, we cannot evaluate edge.
    if market_home_spread is None:
        return {
            "projected_home_margin": None,
            "fair_home_spread": None,
            "spread_diff": None,
            "adjusted_spread_diff": None,
            "estimated_edge_points": None,
            "edge_side": "PASS"
        }

    # Pull shared strength inputs.
    vals = _get_strength_inputs(home_summary, away_summary)

    # -----------------------------------------------------
    # HOME TEAM STRENGTH SCORE (V2)
    # -----------------------------------------------------
    # Compared with v1, this version:
    # - increases season weight from 50% to 60%
    # - reduces site weight from 30% to 25%
    # - reduces recent weight from 20% to 15%
    #
    # So v2 is telling the model:
    # "Trust the larger full-season sample more, and trust short-term form less."
    #
    # That generally makes the model more stable and less likely to swing
    # too hard from recent hot/cold stretches.
    home_strength = (
        0.60 * vals["home_season_pd"] +
        0.25 * vals["home_site_pd"] +
        0.15 * vals["home_recent_pd"]
    )

    # Same weighting logic for the away team.
    away_strength = (
        0.60 * vals["away_season_pd"] +
        0.25 * vals["away_site_pd"] +
        0.15 * vals["away_recent_pd"]
    )

    # Same fixed home-court advantage as v1.
    home_court_advantage = 1.5

    # -----------------------------------------------------
    # RAW PROJECTED HOME MARGIN
    # -----------------------------------------------------
    # This is the same basic concept as v1 before the v2 adjustment.
    #
    # Example:
    # if home_strength = 6.0
    #    away_strength = 3.5
    # then raw difference = 2.5
    # plus home-court 1.5 = 4.0
    raw_projected_home_margin = (home_strength - away_strength) + home_court_advantage

    # -----------------------------------------------------
    # V2 CALIBRATION MULTIPLIER
    # -----------------------------------------------------
    # V2 multiplies the projected margin by 1.10.
    #
    # Why do this?
    # It slightly widens the model's projected result.
    #
    # Example:
    # raw_projected_home_margin = 4.0
    # projected_home_margin = 4.0 * 1.10 = 4.4
    #
    # So v2 becomes a little stronger / more aggressive in what it thinks
    # the true margin should be.
    #
    # Since you said v2 tested better, this multiplier may be helping
    # correct an underestimation bias in the raw projection.
    projected_home_margin = round(raw_projected_home_margin * 1.10, 3)

    # Convert projected scoring margin into a fair home spread.
    #
    # Same logic as v1:
    # win by 4.4 -> fair spread is -4.4
    # lose by 2.1 -> fair spread is +2.1
    fair_home_spread = round(-projected_home_margin, 3)

    # -----------------------------------------------------
    # RAW SPREAD DIFFERENCE
    # -----------------------------------------------------
    # This is the initial difference between our fair line and the market line.
    #
    # Same interpretation as v1:
    # - negative -> home spread value
    # - positive -> away spread value
    spread_diff = round(fair_home_spread - market_home_spread, 3)

    # Start with raw spread difference.
    # Then we may reduce it if the market spread is large.
    adjusted_spread_diff = spread_diff

    # -----------------------------------------------------
    # LARGE SPREAD PENALTY / EDGE SHRINK
    # -----------------------------------------------------
    # This is one of the biggest differences between v1 and v2.
    #
    # Idea:
    # Very large spreads are trickier and noisier.
    # Dogs can look attractive because "that's a lot of points,"
    # but these situations can also be unstable.
    #
    # So instead of trusting the raw edge fully, v2 shrinks it:
    #
    # If abs(spread) >= 10:
    #   keep only 80% of the raw edge
    #
    # If abs(spread) >= 7:
    #   keep only 90% of the raw edge
    #
    # Examples:
    #
    # raw spread_diff = +8.0, market spread = -12
    # adjusted_spread_diff = 8.0 * 0.80 = 6.4
    #
    # raw spread_diff = -4.0, market spread = -8
    # adjusted_spread_diff = -4.0 * 0.90 = -3.6
    #
    # This does NOT change the direction of the bet.
    # It only reduces confidence / edge size.
    if abs(market_home_spread) >= 10:
        adjusted_spread_diff *= 0.80
    elif abs(market_home_spread) >= 7:
        adjusted_spread_diff *= 0.90

    adjusted_spread_diff = round(adjusted_spread_diff, 3)

    # After the shrink adjustment, compute the final edge size.
    #
    # This is the number you will often use for confidence or filtering.
    estimated_edge_points = round(abs(adjusted_spread_diff), 3)

    # -----------------------------------------------------
    # DYNAMIC PASS THRESHOLD
    # -----------------------------------------------------
    # V2 does not use the same fixed no-bet zone as v1.
    #
    # Instead:
    # - normal games use threshold = 2.5
    # - very large spreads (10+) use threshold = 3.5
    #
    # That means v2 requires MORE separation from the market before
    # it will fire a bet on very large lines.
    #
    # This is another major reason it may have tested better:
    # fewer low-quality big-spread bets.
    threshold = 2.5
    if abs(market_home_spread) >= 10:
        threshold = 3.5

    # -----------------------------------------------------
    # DECISION RULES
    # -----------------------------------------------------
    # If the adjusted edge is too small, PASS.
    #
    # Otherwise:
    # - adjusted_spread_diff < 0 -> HOME_SPREAD
    # - adjusted_spread_diff > 0 -> AWAY_SPREAD
    #
    # Notice:
    # v2 uses adjusted_spread_diff here, not raw spread_diff.
    # So large-spread penalties can turn some borderline plays into PASS.
    if abs(adjusted_spread_diff) <= threshold:
        edge_side = "PASS"
    elif adjusted_spread_diff < 0:
        edge_side = "HOME_SPREAD"
    else:
        edge_side = "AWAY_SPREAD"

    # Return more debugging detail than v1.
    #
    # This is useful because v2 has extra logic beyond raw spread diff,
    # so seeing both raw and adjusted values helps you understand
    # what changed and why a play may have become PASS.
    return {
        # Final projected scoring margin for home team after 1.10 multiplier
        "projected_home_margin": projected_home_margin,

        # Model-implied fair home spread
        "fair_home_spread": fair_home_spread,

        # Raw difference between our fair line and market line
        "spread_diff": spread_diff,

        # Difference after large-spread penalties
        "adjusted_spread_diff": adjusted_spread_diff,

        # Final absolute edge size used for filtering and confidence
        "estimated_edge_points": estimated_edge_points,

        # Final directional result
        "edge_side": edge_side
    }


# =========================================================
# BUILD TEAM STATS FROM HISTORICAL GAMES
# =========================================================

def build_team_stats(historical_games):
    """
    Loops through historical games and accumulates all team-level stats.
    These totals are later summarized into per-team features.
    """
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

        # ---------------- HOME TEAM UPDATE ----------------
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

        # ---------------- AWAY TEAM UPDATE ----------------
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


# =========================================================
# HEAD-TO-HEAD
# =========================================================

def build_head_to_head_stats(home_team_id, away_team_id, historical_games):
    """
    Builds head-to-head info for display/debugging.

    Important:
    We are keeping this out of the core projection model because H2H tends
    to be noisy and often not very predictive across changing rosters/seasons.
    """
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

            # Flip the result because the historical game's home team is not
            # necessarily the same as the current matchup's home team.
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


# =========================================================
# PAYLOAD BUILDER
# =========================================================

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

        # Run both models so they can be compared side-by-side
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

        # Pick which model drives the main fields sent to AI/output
        # Switch this line to edge_info_v1 if you want v1 to be the active model
        active_edge_info = edge_info_v2

        h2h_stats = build_head_to_head_stats(
            home_team_id,
            away_team_id,
            historical_games
        )

        payload.append({
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

            # Primary fields based on whichever model is active
            "projected_home_margin": active_edge_info["projected_home_margin"],
            "fair_home_spread": active_edge_info["fair_home_spread"],
            "estimated_edge_points": active_edge_info["estimated_edge_points"],
            "edge_side": active_edge_info["edge_side"],

            # Side-by-side testing fields
            "edge_model_v1": edge_info_v1,
            "edge_model_v2": edge_info_v2
        })

    return payload