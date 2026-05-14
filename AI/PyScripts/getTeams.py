import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")

DATABASE_URL = (
    f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
    "?sslmode=require&channel_binding=require"
)

engine = create_engine(DATABASE_URL)

def get_all_teams():
    query = text("""
        SELECT
            team_id,
            team_name,
            source_team_id,
            team_abbrev,
            team_city
        FROM teams
        ORDER BY team_id;
    """)

    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()

    return [
        {
            "team_id": row.team_id,
            "team_name": row.team_name,
            "source_team_id": row.source_team_id,
            "team_abbrev": row.team_abbrev,
            "team_city": row.team_city
        }
        for row in rows
    ]


if __name__ == "__main__":
    teams = get_all_teams()

    print(f"{'ID':<6} {'NAME':<15} {'ABBREV':<8} {'CITY':<20} {'SOURCE_TEAM_ID':<12}")
    print("-" * 70)

    for team in teams:
        print(
            f"{team['team_id']:<6} "
            f"{team['team_name']:<15} "
            f"{team['team_abbrev']:<8} "
            f"{team['team_city']:<20} "
            f"{team['source_team_id']:<12}"
        )