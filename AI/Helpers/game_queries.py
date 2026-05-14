from datetime import date
from sqlalchemy import text
from db import engine


def fetch_historical_games(start_date: str):
    query = text("""
        SELECT
            game_id,
            date,
            home_team_id,
            away_team_id,
            opening_spread,
            opening_total,
            opening_moneyline,
            home_score,
            away_score
        FROM games
        WHERE date >= :start_date
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
        ORDER BY date ASC, game_id ASC;
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"start_date": start_date})
        rows = result.mappings().all()

    return [dict(row) for row in rows]


def fetch_historical_games_with_box_scores(start_date: str):
    """
    Fetch completed games joined with per-team box-score stats from
    team_game_stats.  Returns TWO rows per game (one per team), matching
    the shape used in training (load_raw_team_game_history).

    Each row carries the game-level columns PLUS the team's own box-score
    stats and an ``is_home`` flag so the caller can split home / away.
    """
    query = text("""
        SELECT
            g.game_id,
            g.date,
            g.home_team_id,
            g.away_team_id,
            g.opening_spread,
            g.opening_total,
            g.opening_moneyline,
            g.home_score,
            g.away_score,

            tgs.team_id,
            CASE WHEN tgs.team_id = g.home_team_id THEN 1 ELSE 0 END AS is_home,
            CASE WHEN tgs.team_id = g.away_team_id THEN 1 ELSE 0 END AS is_away,

            CASE
                WHEN tgs.team_id = g.home_team_id THEN g.home_score
                WHEN tgs.team_id = g.away_team_id THEN g.away_score
            END AS team_score,
            CASE
                WHEN tgs.team_id = g.home_team_id THEN g.away_score
                WHEN tgs.team_id = g.away_team_id THEN g.home_score
            END AS opponent_score,
            CASE
                WHEN tgs.team_id = g.home_team_id AND g.home_score > g.away_score THEN 1
                WHEN tgs.team_id = g.away_team_id AND g.away_score > g.home_score THEN 1
                ELSE 0
            END AS team_win,

            tgs.points,
            tgs.field_goal_pct,
            tgs.three_pt_pct,
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
        JOIN team_game_stats tgs ON g.game_id = tgs.game_id
        WHERE g.date >= :start_date
          AND g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
        ORDER BY g.date ASC, g.game_id ASC, tgs.team_id ASC;
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"start_date": start_date})
        rows = result.mappings().all()

    return [dict(row) for row in rows]


def fetch_team_names():
    query = text("""
        SELECT
            team_id,
            team_name
        FROM teams
        ORDER BY team_id;
    """)

    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.mappings().all()

    return {row["team_id"]: row["team_name"] for row in rows}


def fetch_team_id_map_by_source_id():
    query = text("""
        SELECT
            team_id,
            source_team_id,
            team_name,
            team_abbrev
        FROM teams
        WHERE source_team_id IS NOT NULL;
    """)

    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.mappings().all()

    return {
        str(row["source_team_id"]): {
            "team_id": row["team_id"],
            "team_name": row["team_name"],
            "team_abbrev": row["team_abbrev"]
        }
        for row in rows
    }