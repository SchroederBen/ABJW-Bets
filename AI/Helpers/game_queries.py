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