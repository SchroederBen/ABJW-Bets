import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env")

client = OpenAI(api_key=OPENAI_API_KEY)

def summarize_teams_with_ai(teams):
    teams_text = "\n".join(
        f"{team['team_id']}: {team['team_city']} {team['team_name']} ({team['team_abbrev']}) | source_team_id={team['source_team_id']}"
        for team in teams
    )

    prompt = f"""
Here is the list of NBA teams from my database.

{teams_text}

Please do two things:
1. Confirm how many teams are in the list.
2. Return the teams as a clean bullet list in the format:
   - ABBR: City Name
"""

    response = client.responses.create(
    model="gpt-4.1-nano",
    temperature=0.1,
    input=prompt
)

    return response.output_text