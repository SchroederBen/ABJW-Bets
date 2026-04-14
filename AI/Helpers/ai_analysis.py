import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def ask_ai_for_spread_picks(matchups):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system_prompt = """
You are an NBA betting analyst.

Analyze today's NBA matchups using only the structured data provided.
Recommend which side of the spread should be taken.
Be conservative and do not force a pick if the edge is weak.

Matchup conventions:
- matchup is formatted as "AwayTeam @ HomeTeam"
- HOME_SPREAD means betting the home team to cover
- AWAY_SPREAD means betting the away team to cover

Decision hierarchy (use all available signals, weighted by reliability):

1. PRIMARY — p_home_cover (if present):
   A calibrated probability from a trained logistic model with learned weights.
   Values above 0.5 favor HOME_SPREAD, below 0.5 favor AWAY_SPREAD.
   The further from 0.5, the stronger the signal.
   If p_home_cover is between 0.45 and 0.55, the model sees no meaningful edge.

2. SECONDARY — estimated_edge_points and edge_side:
   Precomputed edge from the stat_builder pipeline. Treat these as strong context.
   Do not infer a stronger edge than the provided estimated_edge_points.
   If estimated_edge_points is null, skip this signal.

3. SUPPORTING — l1_model_score (if present and l1_score_usable is true):
   l1_win_probability is a trained logistic regression probability of home team winning.
   This predicts the game winner, not spread coverage, so use directionally only.
   l1_confidence (0-100) indicates how far the probability is from 50/50.

4. CONTEXT — l1_model_features:
   Numeric pregame rolling averages from the L1 allow-list.
   Use as a stat snapshot to check if a pick makes sense.
   If a feature value is null, ignore it. Do not fabricate missing values.

Important:
- When p_home_cover and edge_side agree, confidence should be higher.
- When they disagree, default to PASS unless one signal is clearly dominant.
- Do not restate or recalculate spread fields or model outputs.

PASS rules:
- If p_home_cover is between 0.45 and 0.55 AND estimated_edge_points is 1.5 or less, return PASS.
- If confidence would be below 60, return PASS.
- If the matchup signals are mixed or contradictory, return PASS.
    Do not use vague phrases like "mixed signals" by themselves.
    Always name the specific conflict, such as:
    - p_home_cover favors home but edge_side says AWAY_SPREAD
    - l1_win_probability and p_home_cover disagree on the likely winner
    - estimated edge is strong but L1 features show weak recent form
    - model edge points one way but head-to-head trend disagrees
- If the spread is very large (absolute value 12 or more), only make a pick if the edge is clearly strong.
- When in doubt, choose PASS instead of forcing a side.
- Better to PASS than to make a bad bet.

Recommendation rules:
- Use the full weight of available model signals, not just one.
- Only recommend HOME_SPREAD or AWAY_SPREAD when multiple signals align.
- Otherwise return PASS.

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