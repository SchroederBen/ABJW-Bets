import os
import json
from dotenv import load_dotenv

from Helpers.game_queries import fetch_historical_games, fetch_team_id_map_by_source_id
from Helpers.stat_builder import build_team_stats, build_matchup_payload_from_api_games
from Helpers.ai_analysis import ask_ai_for_spread_picks
from NBA_APIs import get_today_nba_games

load_dotenv()

START_DATE = os.getenv("START_DATE", "2024-01-01")
HOME_SPREAD_IS_POSITIVE_FAVORITE = (
    os.getenv("HOME_SPREAD_IS_POSITIVE_FAVORITE", "true").lower() == "true"
)


def main():
    try:
        historical_games = fetch_historical_games(START_DATE)
        todays_api_games = get_today_nba_games()
        source_team_map = fetch_team_id_map_by_source_id()

        if not historical_games:
            print(f"No historical games found since {START_DATE}.")
            return

        if not todays_api_games:
            print("No NBA games found from the daily API.")
            return

        if not source_team_map:
            print("No team source ID mappings found in teams table.")
            return

        team_stats = build_team_stats(
            historical_games,
            home_spread_is_positive_favorite=HOME_SPREAD_IS_POSITIVE_FAVORITE
        )

        matchup_payload = build_matchup_payload_from_api_games(
            todays_api_games,
            team_stats,
            source_team_map
        )

        if not matchup_payload:
            print("No matchups could be built from API games and DB team mappings.")
            return

        print("=== Matchup payload sent to AI ===")
        print(json.dumps(matchup_payload, indent=2, default=str))

        ai_result = ask_ai_for_spread_picks(matchup_payload)

        print("\n=== AI Predictions ===")
        print(json.dumps(ai_result, indent=2))

    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()