import pandas as pd
from src.utils.db_connection import get_engine

RAW_TEAM_GAME_HISTORY_QUERY = """
SELECT
    -- game identifiers
    g.game_id,
    g.date,
    g.source_game_id,
    g.mgm_game_id,

    -- matchup info from games
    g.home_team_id,
    g.away_team_id,
    home_team.team_name AS home_team_name,
    away_team.team_name AS away_team_name,

    -- lines / outcomes from games
    g.opening_spread,
    g.opening_total,
    g.opening_moneyline,
    g.home_score,
    g.away_score,

    -- team row info
    tgs.team_game_stat_id,
    tgs.team_id,
    team_lookup.team_name,
    team_lookup.team_abbrev,
    team_lookup.team_city,
    tgs.opponent_team_id,
    opponent_lookup.team_name AS opponent_team_name,
    opponent_lookup.team_abbrev AS opponent_team_abbrev,
    opponent_lookup.team_city AS opponent_team_city,
    tgs.game_datetime,
    tgs.source_system,
    tgs.created_at,

    -- role flags derived from games
    CASE
        WHEN tgs.team_id = g.home_team_id THEN 1
        ELSE 0
    END AS is_home,

    CASE
        WHEN tgs.team_id = g.away_team_id THEN 1
        ELSE 0
    END AS is_away,

    CASE
        WHEN tgs.team_id = g.home_team_id THEN g.away_team_id
        WHEN tgs.team_id = g.away_team_id THEN g.home_team_id
        ELSE NULL
    END AS derived_opponent_team_id,

    CASE
        WHEN tgs.team_id = g.home_team_id THEN away_team.team_name
        WHEN tgs.team_id = g.away_team_id THEN home_team.team_name
        ELSE NULL
    END AS derived_opponent_team_name,

    -- result labels at team-row level
    CASE
        WHEN tgs.team_id = g.home_team_id THEN g.home_score
        WHEN tgs.team_id = g.away_team_id THEN g.away_score
        ELSE NULL
    END AS team_score,

    CASE
        WHEN tgs.team_id = g.home_team_id THEN g.away_score
        WHEN tgs.team_id = g.away_team_id THEN g.home_score
        ELSE NULL
    END AS opponent_score,

    CASE
        WHEN tgs.team_id = g.home_team_id AND g.home_score > g.away_score THEN 1
        WHEN tgs.team_id = g.away_team_id AND g.away_score > g.home_score THEN 1
        ELSE 0
    END AS team_win,

    -- raw team game stats
    tgs.num_minutes,
    tgs.points,
    tgs.field_goals_made,
    tgs.field_goals_attempted,
    tgs.field_goal_pct,
    tgs.three_pt_made,
    tgs.three_pt_attempted,
    tgs.three_pt_pct,
    tgs.free_throws_made,
    tgs.free_throws_attempted,
    tgs.free_throw_pct,
    tgs.offensive_rebounds,
    tgs.defensive_rebounds,
    tgs.total_rebounds,
    tgs.assists,
    tgs.steals,
    tgs.blocks,
    tgs.turnovers,
    tgs.personal_fouls

FROM games g

JOIN team_game_stats tgs
    ON g.game_id = tgs.game_id

JOIN teams team_lookup
    ON tgs.team_id = team_lookup.team_id

LEFT JOIN teams opponent_lookup
    ON tgs.opponent_team_id = opponent_lookup.team_id

JOIN teams home_team
    ON g.home_team_id = home_team.team_id

JOIN teams away_team
    ON g.away_team_id = away_team.team_id

ORDER BY
    team_lookup.team_name,
    g.date,
    g.game_id;
"""


def load_query(query: str) -> pd.DataFrame:
    """
    Execute a SQL query and return the results as a pandas DataFrame.
    """
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df


def load_raw_team_game_history() -> pd.DataFrame:
    """
    Load one row per team per game, combining:
    - games
    - team_game_stats
    - teams

    This is the raw historical dataset used to build rolling pregame features.
    """
    return load_query(RAW_TEAM_GAME_HISTORY_QUERY)


if __name__ == "__main__":
    df = load_raw_team_game_history()

    print(df.tail())
    print()
    print("Shape:", df.shape)
    print()
    print("Columns:")
    print(df.columns.tolist())
    #df.tail(200).to_csv("src/data/raw_team_game_history_sample.csv", index=False)