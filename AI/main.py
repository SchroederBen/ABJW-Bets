import os
import json
import csv
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from Helpers.game_queries import fetch_historical_games, fetch_team_id_map_by_source_id
from Helpers.stat_builder import build_team_stats, build_matchup_payload_from_api_games
from Helpers.ai_analysis import ask_ai_for_spread_picks
from NBA_APIs import get_today_nba_games

load_dotenv()

#year-month-day
START_DATE = os.getenv("START_DATE", "2023-10-01")
RUN_LOG_CSV = Path(__file__).resolve().parent / "run_results_log.csv"


def append_run_results_to_csv(ai_result):
    run_timestamp = datetime.now().isoformat(timespec="seconds")
    csv_headers = [
        "run_timestamp",
        "game_id",
        "matchup",
        "home_team_name",
        "away_team_name",
        "projected_home_margin",
        "fair_home_spread",
        "estimated_edge_points",
        "edge_side",
        "recommended_bet",
        "confidence",
        "short_reason",
        "risk_flags"
    ]

    rows = []
    for prediction in ai_result.get("predictions", []):
        rows.append(
            {
                "run_timestamp": run_timestamp,
                "game_id": prediction.get("game_id"),
                "matchup": prediction.get("matchup"),
                "home_team_name": prediction.get("home_team_name"),
                "away_team_name": prediction.get("away_team_name"),
                "projected_home_margin": prediction.get("projected_home_margin"),
                "fair_home_spread": prediction.get("fair_home_spread"),
                "estimated_edge_points": prediction.get("estimated_edge_points"),
                "edge_side": prediction.get("edge_side"),
                "recommended_bet": prediction.get("recommended_bet"),
                "confidence": prediction.get("confidence"),
                "short_reason": prediction.get("short_reason"),
                "risk_flags": " | ".join(prediction.get("risk_flags", []))
            }
        )

    if not rows:
        return

    file_exists = RUN_LOG_CSV.exists()
    with RUN_LOG_CSV.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


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
        append_run_results_to_csv(ai_result)
        print(f"\nSaved run results to CSV: {RUN_LOG_CSV}")

        print("\n=== V1 vs V2 Comparison ===")
        for game in matchup_payload:
            print(
                f"{game['matchup']} | "
                f"V1: {game['edge_model_v1']['edge_side']} ({game['edge_model_v1']['estimated_edge_points']}) | "
                f"V2: {game['edge_model_v2']['edge_side']} ({game['edge_model_v2']['estimated_edge_points']}) | "
                f"ACTIVE: {game['edge_side']}"
    )

        print("\n=== Human Readable Predictions ===")
        for line in ai_result.get("human_readable", []):
            print(line+"\n")

    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()