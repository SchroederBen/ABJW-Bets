import pandas as pd
from src.data.load_raw_team_game_history import load_raw_team_game_history
from tqdm import tqdm

"""
    Calculates a moving average for a given column using a sliding window.

    Important:
    - grouped by team_id by default
    - uses shift(1) so the current game's value is NOT included
      in its own rolling average

    """
def calculate_moving_average(df, column, window, group_col="team_id"):
    
    output_col = f"{column}_MA_{window}"

    df[output_col] = (
        df.groupby(group_col)[column]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
    )

    return df


#Calculates rest days between games for each team.
def calculate_days_rest(df, group_col="team_id", date_col="date"):
    
    df = df.copy()
    df["days_rest"] = (
        df.groupby(group_col)[date_col]
        .diff()
        .dt.days
    )
    return df

#Cleans up the data if needed
def clean_raw_team_game_history(df):
    
    df = df.copy()

    # Dates
    df["date"] = pd.to_datetime(df["date"])

    # Remove duplicate team-game rows if any exist
    df = df.drop_duplicates(subset=["game_id", "team_id"])

    # Sort for rolling calculations
    df = df.sort_values(["team_id", "date", "game_id"]).reset_index(drop=True)

    # Convert numeric columns
    numeric_cols = [
        "opening_spread",
        "opening_total",
        "opening_moneyline",
        "home_score",
        "away_score",
        "team_score",
        "opponent_score",
        "num_minutes",
        "points",
        "field_goals_made",
        "field_goals_attempted",
        "field_goal_pct",
        "three_pt_made",
        "three_pt_attempted",
        "three_pt_pct",
        "free_throws_made",
        "free_throws_attempted",
        "free_throw_pct",
        "offensive_rebounds",
        "defensive_rebounds",
        "total_rebounds",
        "assists",
        "steals",
        "blocks",
        "turnovers",
        "personal_fouls",
    ]

    # makes sure cols are there and if they are converts to numeric if not already
    existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
    if existing_numeric_cols:
        df[existing_numeric_cols] = df[existing_numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        )

    return df

"""
    Loads raw team-game history, cleans it, and creates reusable rolling features.

    windows:
        tuple of rolling window sizes to create, or None to skip rolling averages.
        example: (3, 5, 10)
        If None (the default) no rolling average columns will be added.
    """
def build_pregame_features(windows=None):
    
    df = load_raw_team_game_history()
    df = clean_raw_team_game_history(df)

    # Days rest
    df = calculate_days_rest(df)

    # Rolling columns to use
    rolling_cols = [
        "points",
        "team_score",
        "opponent_score",
        "field_goal_pct",
        "three_pt_pct",
        "free_throw_pct",
        "offensive_rebounds",
        "defensive_rebounds",
        "total_rebounds",
        "assists",
        "steals",
        "blocks",
        "turnovers",
        "personal_fouls",
        "team_win",
    ]

    # Build rolling averages for each chosen window. If `windows` is None or empty,
    # skip adding any rolling-average columns (this is the default behavior).
    if windows:
        for window in tqdm(windows, desc="rolling windows"):
            for col in tqdm(rolling_cols, desc=f"cols (w={window})", leave=False):
                if col in df.columns:
                    df = calculate_moving_average(df, col, window)

    # Fill only rolling/rest columns for now
    engineered_cols = [
        col for col in df.columns
        if "_MA_" in col or col == "days_rest"
    ]

    for col in engineered_cols:
        df[col] = df[col].fillna(0)

    return df


if __name__ == "__main__":
    df = build_pregame_features(windows=(3, 5, 10))

    print(df.tail())
    print()
    print("Shape:", df.shape)
    print()
    print("Example rolling columns:")
    print([col for col in df.columns if "_MA_" in col][:20])

    df.tail(200).to_csv("src/data/pregame_features_sample.csv", index=False)
    print("Saved sample CSV to src/data/pregame_features_sample.csv")