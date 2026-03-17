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

Spread conventions:
- opening_spread is the opening spread from the home team's perspective.
- current_spread is the current spread from the home team's perspective.
- Example: -4.5 means the home team is favored by 4.5.
- Example: +4.5 means the home team is getting 4.5.

Use both the opening and current spread when relevant, especially to notice line movement.

Return valid JSON only.

For each game return:
- game_id
- matchup
- opening_spread
- current_spread
- recommended_bet: one of ["HOME_SPREAD", "AWAY_SPREAD", "PASS"]
- confidence: integer 1-100
- estimated_edge_points: number
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
                                    "matchup": {"type": "string"},
                                    "opening_spread": {
                                        "anyOf": [
                                            {"type": "number"},
                                            {"type": "null"}
                                        ]
                                    },
                                    "current_spread": {
                                        "anyOf": [
                                            {"type": "number"},
                                            {"type": "null"}
                                        ]
                                    },
                                    "recommended_bet": {
                                        "type": "string",
                                        "enum": ["HOME_SPREAD", "AWAY_SPREAD", "PASS"]
                                    },
                                    "confidence": {"type": "integer"},
                                    "estimated_edge_points": {"type": "number"},
                                    "short_reason": {"type": "string"},
                                    "risk_flags": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": [
                                    "game_id",
                                    "matchup",
                                    "opening_spread",
                                    "current_spread",
                                    "recommended_bet",
                                    "confidence",
                                    "estimated_edge_points",
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

    return json.loads(response.output_text)