import os
import json
import csv
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from Helpers.game_queries import (
    fetch_historical_games,
    fetch_historical_games_with_box_scores,
    fetch_team_id_map_by_source_id,
)
from Helpers.stat_builder import build_team_stats, build_matchup_payload_from_api_games
from Helpers.ai_analysis import ask_ai_for_spread_picks
from NBA_APIs import get_today_nba_games

load_dotenv()

# year-month-day
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
        print("=== Fetching data ===")
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

        # Fetch box-score data (graceful fallback if team_game_stats table
        # doesn't exist yet or is empty)
        team_game_rows = None
        try:
            team_game_rows = fetch_historical_games_with_box_scores(START_DATE)
            if team_game_rows:
                print(f"Loaded {len(team_game_rows)} box-score rows from team_game_stats.")
            else:
                print("No box-score rows found — using score-only fallback.")
                team_game_rows = None
        except Exception as e:
            print(f"Could not fetch box-score data (falling back to scores only): {e}")
            team_game_rows = None

        team_stats = build_team_stats(historical_games)

        matchup_payload = build_matchup_payload_from_api_games(
            todays_api_games,
            team_stats,
            source_team_map,
            historical_games,
            team_game_rows=team_game_rows,
        )

        if not matchup_payload:
            print("No matchups could be built from API games and DB team mappings.")
            return

        print("=== Matchup payload sent to AI ===")
        print(json.dumps(matchup_payload, indent=2, default=str))

        print("\n=== Head-to-Head Stats ===")
        for game in matchup_payload:
            print(f'{game["matchup"]}')
            print(json.dumps(game.get("head_to_head_stats", {}), indent=2))
            print()

        print("\n=== L1 Feature Preview ===")
        for game in matchup_payload:
            l1_meta = game.get("l1_model_features_meta", {})

            if l1_meta.get("enabled"):
                print(
                    f"{game['matchup']} | "
                    f"L1 allow-list: {l1_meta.get('allowlist_size', 0)} features | "
                    f"computed: {l1_meta.get('computed_feature_count', 0)} | "
                    f"null: {l1_meta.get('null_feature_count', 0)} | "
                    f"source: {l1_meta.get('data_source', '?')}"
                )
            else:
                print(f"{game['matchup']} | L1 allow-list not loaded")

        ai_result = ask_ai_for_spread_picks(matchup_payload)

        print("\n=== SAMPLE L1 FEATURE BLOCK (FIRST GAME) ===")
        if matchup_payload:
            sample_game = matchup_payload[0]

            print(f"Matchup: {sample_game['matchup']}")

            l1_features = sample_game.get("l1_model_features", {})

            non_null_features = {
                k: v for k, v in l1_features.items() if v is not None
            }

            print(f"\n--- Non-null L1 features: {len(non_null_features)} ---")
            print(json.dumps(non_null_features, indent=2))

            print(f"\n--- Null features: {sum(1 for v in l1_features.values() if v is None)} ---")

        # print("\n=== AI Predictions ===")
        # print(json.dumps(ai_result, indent=2))
        # append_run_results_to_csv(ai_result)
        # print(f"\nSaved run results to CSV: {RUN_LOG_CSV}")

        print("\n=== Edge Model Comparison ===")
        for game in matchup_payload:
            v2 = game.get("edge_model_v2", {})
            learned = game.get("edge_model_learned", {})
            l1_score = game.get("l1_model_score", {})

            parts = [
                f"{game['matchup']}",
                f"V2: {v2.get('edge_side', '?')} ({v2.get('estimated_edge_points', '?')})",
            ]

            # Learned model info
            if learned.get("model_version", "").startswith("stat_builder_weights"):
                parts.append(
                    f"Learned: {learned.get('edge_side', '?')} "
                    f"({learned.get('estimated_edge_points', '?')}) "
                    f"P(cover)={learned.get('p_home_cover', '?')}"
                )
            else:
                parts.append("Learned: (not loaded)")

            # L1 model score
            if l1_score.get("l1_score_usable"):
                parts.append(
                    f"L1 P(win)={l1_score.get('l1_win_probability', '?')} "
                    f"conf={l1_score.get('l1_confidence', '?')} "
                    f"null%={l1_score.get('l1_null_feature_pct', '?')}"
                )
            elif l1_score.get("l1_model_available"):
                parts.append(
                    f"L1: too many nulls ({l1_score.get('l1_null_feature_pct', '?')})"
                )
            else:
                parts.append("L1 model: not loaded")

            parts.append(f"ACTIVE: {game.get('edge_side', '?')}")

            print(" | ".join(parts))

        print("\n=== Human Readable Predictions ===")
        for line in ai_result.get("human_readable", []):
            print(line + "\n")

    except Exception as e:
        import traceback
        print(f"Unexpected error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()