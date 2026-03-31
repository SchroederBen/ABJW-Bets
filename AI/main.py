import os
import json
from dotenv import load_dotenv

from Helpers.game_queries import fetch_historical_games, fetch_team_id_map_by_source_id
from Helpers.stat_builder import build_team_stats, build_matchup_payload_from_api_games
from Helpers.ai_analysis import ask_ai_for_spread_picks
from NBA_APIs import get_today_nba_games

load_dotenv()

#year-month-day
START_DATE = os.getenv("START_DATE", "2023-10-01")


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
            historical_games
        )

        matchup_payload = build_matchup_payload_from_api_games(
            todays_api_games,
            team_stats,
            source_team_map,
            historical_games
        )

        if not matchup_payload:
            print("No matchups could be built from API games and DB team mappings.")
            return

        #print("=== Matchup payload sent to AI ===")
        #print(json.dumps(matchup_payload, indent=2, default=str))

        print("\n=== Head-to-Head Stats ===")
        for game in matchup_payload:
            print(f'{game["matchup"]}')
            print(json.dumps(game.get("head_to_head_stats", {}), indent=2))
            print()

        ai_result = ask_ai_for_spread_picks(matchup_payload)

        print("\n=== AI Predictions ===")
        print(json.dumps(ai_result, indent=2))

        print("\n=== Edge Debug ===")
        for game in matchup_payload:
            print(game["matchup"])
            print("top-level estimated_edge_points:", game.get("estimated_edge_points"))
            print("top-level edge_side:", game.get("edge_side"))
            print("v1:", game.get("edge_model_v1"))
            print("v2:", game.get("edge_model_v2"))
            print("v3:", game.get("edge_model_v3"))
            print()

        print("\n=== Human Readable Predictions ===")
        for line in ai_result.get("human_readable", []):
            print(line+"\n")

    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()