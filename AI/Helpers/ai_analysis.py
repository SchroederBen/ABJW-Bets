import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def ask_ai_for_spread_picks(matchups):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system_prompt = """
You are an NBA betting analyst with an action-oriented approach.

Analyze today's NBA matchups using only the structured data provided.
Recommend which side of the spread should be taken.
Your goal is to find actionable edges — lean toward making picks when
any reasonable signal exists, rather than defaulting to PASS.

Matchup conventions:
- matchup is formatted as "AwayTeam @ HomeTeam"
- HOME_SPREAD means betting the home team to cover
- AWAY_SPREAD means betting the away team to cover

Decision hierarchy (use all available signals, weighted by reliability):

1. PRIMARY — p_home_cover (if present):
   A calibrated probability from a trained logistic model with learned weights.
   Values above 0.5 favor HOME_SPREAD, below 0.5 favor AWAY_SPREAD.
   The further from 0.5, the stronger the signal.
   Note: if p_home_cover is very close to 0.5 (within ~0.01), treat it as neutral
   and rely more heavily on the other signals below.

2. SECONDARY — estimated_edge_points and edge_side:
   Precomputed edge from the stat_builder pipeline. Treat these as actionable.
   If edge_side is not PASS and estimated_edge_points >= 1.0, that alone
   is enough to lean toward a pick in that direction.
   If estimated_edge_points is null, skip this signal.

3. SUPPORTING — home_stats and away_stats:
   Treat these as recent-form and team-profile summaries.
   Pay special attention to:
   - point_diff_MA_3, point_diff_MA_5, point_diff_MA_10
   - ATS_cover_rate_MA_5, ATS_cover_rate_MA_10
   - last_10_avg_point_diff
   - last_10_ats_win_pct
   - home/away split stats such as home_win_pct, away_win_pct,
     home_avg_pts_for, home_avg_pts_against, away_avg_pts_for, away_avg_pts_against

   Use these to identify:
   - whether one team has stronger recent point differential trends
   - whether one team has been covering more consistently recently
   - whether home/away splits support the side of the spread

4. SUPPORTING — matchup-derived fields:
   Pay attention to these if present:
   - rest_advantage
   - turnover_edge_MA_5
   - rebound_edge_MA_5

   Interpretation:
   - positive rest_advantage favors the home team being more rested
   - positive turnover_edge_MA_5 means the home side has the turnover profile edge
   - positive rebound_edge_MA_5 means the home side has the rebound profile edge

   These should not override a strong primary signal by themselves,
   but they should meaningfully raise or lower confidence as supporting context.

5. SUPPORTING — market movement fields:
   Pay attention to these if present:
   - spread_move_home
   - spread_move_away
   - total_move

   Interpretation:
   - spread_move_home shows how much the home spread moved from opening to current
   - spread_move_away shows how much the away spread moved from opening to current
   - total_move shows how much the game total moved from opening to current

   Use line movement only as supporting market context.
   Meaningful spread movement can modestly raise confidence when it supports
   the main edge signal.
   Do not let line movement override clearly stronger model or trend signals by itself.
   total_move is secondary context and should not drive spread picks on its own.

6. SUPPORTING — l1_model_score (if present and l1_score_usable is true):
   l1_win_probability is a trained logistic regression probability of home team winning.
   This predicts the game winner, not spread coverage, so use directionally.
   l1_confidence (0-100) indicates how far the probability is from 50/50.
   If l1_confidence >= 15, treat this as a meaningful lean.

7. CONTEXT — l1_model_features:
   Numeric pregame rolling averages from the L1 allow-list.
   Use as a stat snapshot to validate or boost confidence.
   If a feature value is null, ignore it. Do not fabricate missing values.

Signal combination rules:
- When ANY two meaningful signals point the same direction, make a pick on that side.
- When p_home_cover and edge_side agree, confidence should be 70+.
- When estimated_edge_points and recent trend fields agree, confidence should increase.
- When recent ATS trend and recent point differential both support the same side,
  treat that as meaningful confirmation.
- When rest_advantage, turnover_edge_MA_5, or rebound_edge_MA_5 support the same side
  as the main edge signal, confidence may be increased modestly.
- When meaningful spread movement supports the same side as the main edge signal,
  confidence may be increased modestly.
- When supporting context conflicts with the main edge signal, reduce confidence
  slightly but do not automatically flip the pick.
- Do not restate or recalculate spread fields or model outputs.

PASS rules (use sparingly — PASS should be the exception, not the norm):
- Only PASS if ALL of the following are true:
    1. p_home_cover is between 0.49 and 0.51 (essentially a coin flip), or unavailable
    2. estimated_edge_points is below 1.0 or edge_side is PASS
    3. recent trend fields do not show a clear directional edge
    4. l1_model_score either unavailable or l1_confidence < 10
- If confidence would be below 45, return PASS.
- Large spreads (12+) are still playable — just flag them as a risk.
  Do NOT auto-pass on large spreads if the edge signals are present.
- When explaining a PASS, always name the specific reason, such as:
    - all model signals are essentially neutral
    - estimated edge is under 1.0 and recent trend support is weak
    - p_home_cover at 0.50 with no edge from stat builder
  Do not use vague phrases like "mixed signals" by themselves.

Confidence calibration:
- 75-100: Two or more strong signals clearly agree
- 60-74: One strong signal, others neutral or mildly supportive
- 45-59: One signal leans a direction, others are neutral
- Below 45: PASS

Reason writing rules:
- Keep short_reason concise and specific.
- Mention the strongest 1-2 reasons only.
- If using the newer payload fields, reference them naturally, for example:
  - stronger recent point differential trend
  - better recent ATS trend
  - small rest advantage
  - turnover edge supports home side
  - rebound edge supports away side
  - line movement supports home side
  - line movement modestly supports away side
- Do not mention fields that are null.
- Do not fabricate missing information.
- If spread_move_home, spread_move_away, and total_move are all 0.0, treat them as neutral.

Always ensure each game has a decision, do not omit or skip any games.

Return valid JSON only.

For each game return:
- game_id
- recommended_bet: one of ["HOME_SPREAD", "AWAY_SPREAD", "PASS"]
- confidence: integer 1-100
- short_reason: string
- risk_flags: array of strings
"""

    user_prompt = f"""
Analyze these NBA games and recommend the best spread side for each.

Matchups:
{json.dumps(matchups, indent=2)}
"""

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "nba_spread_predictions",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "predictions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "game_id": {"type": "integer"},
                                    "recommended_bet": {
                                        "type": "string",
                                        "enum": ["HOME_SPREAD", "AWAY_SPREAD", "PASS"]
                                    },
                                    "confidence": {"type": "integer"},
                                    "short_reason": {"type": "string"},
                                    "risk_flags": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": [
                                    "game_id",
                                    "recommended_bet",
                                    "confidence",
                                    "short_reason",
                                    "risk_flags"
                                ]
                            }
                        }
                    },
                    "required": ["predictions"]
                }
            }
        }
    )

    ai_result = json.loads(response.output_text)
    return merge_ai_predictions(matchups, ai_result)


def merge_ai_predictions(matchups, ai_result):
    ai_map = {int(p["game_id"]): p for p in ai_result["predictions"]}
    merged = {
        "predictions": [],
        "human_readable": []
    }

    for game in matchups:
        game_id = int(game["game_id"])
        ai_pick = ai_map.get(game_id)
        if not ai_pick:
            continue

        merged_pick = {
            "game_id": game_id,
            "matchup": game["matchup"],
            "home_team_name": game["home_team_name"],
            "away_team_name": game["away_team_name"],
            f'{game["home_team_name"]}_current_spread': game["home_current_spread"],
            f'{game["away_team_name"]}_current_spread': game["away_current_spread"],
            f'{game["home_team_name"]}_opening_spread': game["home_opening_spread"],
            f'{game["away_team_name"]}_opening_spread': game["away_opening_spread"],
            "projected_home_margin": game.get("projected_home_margin"),
            "fair_home_spread": game.get("fair_home_spread"),
            "estimated_edge_points": game.get("estimated_edge_points"),
            "edge_side": game.get("edge_side"),
            "recommended_bet": ai_pick["recommended_bet"],
            "confidence": ai_pick["confidence"],
            "short_reason": ai_pick["short_reason"],
            "risk_flags": ai_pick["risk_flags"]
        }

        if ai_pick["recommended_bet"] == "HOME_SPREAD":
            final_bet = f'{game["home_team_name"]} {game["home_current_spread"]:+.1f}'
        elif ai_pick["recommended_bet"] == "AWAY_SPREAD":
            final_bet = f'{game["away_team_name"]} {game["away_current_spread"]:+.1f}'
        else:
            final_bet = "PASS"

        merged_pick["recommended_bet"] = final_bet
        merged["predictions"].append(merged_pick)

        risk_text = ", ".join(ai_pick["risk_flags"]) if ai_pick["risk_flags"] else "None"
        edge_text = (
            f'{game["estimated_edge_points"]:.2f}'
            if game.get("estimated_edge_points") is not None
            else "N/A"
        )

        readable_summary = (
            f'{game["matchup"]} | Bet: {final_bet} | '
            f'Confidence: {ai_pick["confidence"]} | '
            f'Estimated Edge: {edge_text} | '
            f'Reason: {ai_pick["short_reason"]} | '
            f'Risk Flags: {risk_text}'
        )

        merged["human_readable"].append(readable_summary)

    return merged