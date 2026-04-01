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

Important:
- Use the provided spread fields and precomputed edge fields only as context.
- If l1_model_features is present, use those numeric pregame fields as the model-aligned
  stat snapshot (names match offline training); nulls mean that stat was not available live.
- Treat estimated_edge_points and edge_side as the primary decision inputs.
- Do not infer a stronger edge than the provided estimated_edge_points.
- If estimated_edge_points is null, return PASS.
- Do not restate or recalculate spread fields.
- Do not recalculate estimated_edge_points.

PASS rules:
- If the provided estimated_edge_points is 1.5 or less, return PASS.
- If confidence would be below 60, return PASS.
- If the matchup signals are mixed or contradictory, return PASS.
    Do not use vague phrases like "mixed signals" or "conflicting indicators" by themselves.
    Always name the specific conflict, such as:
    - fair-line edge favors away team but recent form favors home team
    - estimated edge is strong but confidence is lowered by large spread
    - model edge and head-to-head trend disagree
- If the spread is very large (absolute value 12 or more), only make a pick if the edge is clearly strong.
- When in doubt, choose PASS instead of forcing a side.
- Better to PASS than to be make a bad bet.

Recommendation rules:
- Use edge_side as a strong signal, but not an automatic pick.
- Only recommend HOME_SPREAD or AWAY_SPREAD when the full context supports it.
- Otherwise return PASS.

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