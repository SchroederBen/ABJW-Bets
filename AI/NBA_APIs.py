import requests
from datetime import datetime
import json

SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
ODDS_URL = "https://cdn.nba.com/static/json/liveData/odds/odds_todaysGames.json"


def safe_get_json(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None
    except ValueError:
        print(f"Error: {url} did not return valid JSON.")
        return None


def parse_scoreboard_games(scoreboard_data):
    if not scoreboard_data:
        return []

    games = scoreboard_data.get("scoreboard", {}).get("games", [])
    if not games:
        return []

    parsed_games = []

    for game in games:
        away_team = game.get("awayTeam", {})
        home_team = game.get("homeTeam", {})

        parsed_games.append({
            "nba_game_id": str(game.get("gameId")) if game.get("gameId") is not None else None,
            "date": game.get("gameEt", datetime.now().strftime("%Y-%m-%d"))[:10],
            "game_status": game.get("gameStatusText", "Unknown Status"),

            "home_source_team_id": str(home_team.get("teamId")) if home_team.get("teamId") is not None else None,
            "home_team_name": home_team.get("teamName", "Unknown"),
            "home_team_tricode": home_team.get("teamTricode", ""),
            "home_score": home_team.get("score"),

            "away_source_team_id": str(away_team.get("teamId")) if away_team.get("teamId") is not None else None,
            "away_team_name": away_team.get("teamName", "Unknown"),
            "away_team_tricode": away_team.get("teamTricode", ""),
            "away_score": away_team.get("score"),

            "start_time_et": game.get("gameEt"),

            # old generic fields if you still want them
            "opening_spread": None,
            "current_spread": None,

            # explicit team spreads
            "home_opening_spread": None,
            "away_opening_spread": None,
            "home_current_spread": None,
            "away_current_spread": None,

            "opening_total": None,
            "current_total": None,

            "opening_moneyline_home": None,
            "opening_moneyline_away": None,
            "current_moneyline_home": None,
            "current_moneyline_away": None,

            "odds_provider": None
        })

    return parsed_games


def pick_preferred_book(books):
    """
    Prefer BetMGM US first.
    If not available, fall back to FanDuel US.
    Then first available.
    """
    if not books:
        return None

    for book in books:
        if book.get("name") == "BetMGM" and book.get("countryCode") == "US":
            return book

    for book in books:
        if book.get("name") == "FanDuel" and book.get("countryCode") == "US":
            return book

    return books[0]


def to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_odds_games(odds_data):
    if not odds_data:
        return {}

    games = odds_data.get("games", [])
    odds_map = {}

    for game in games:
        game_id = game.get("gameId")
        if not game_id:
            continue

        game_id = str(game_id)

        parsed = {
            "opening_spread": None,
            "current_spread": None,

            "home_opening_spread": None,
            "away_opening_spread": None,
            "home_current_spread": None,
            "away_current_spread": None,

            "opening_total": None,
            "current_total": None,
            "opening_moneyline_home": None,
            "opening_moneyline_away": None,
            "current_moneyline_home": None,
            "current_moneyline_away": None,
            "odds_provider": None
        }

        markets = game.get("markets", [])

        for market in markets:
            market_name = (market.get("name") or "").lower()
            books = market.get("books", [])
            book = pick_preferred_book(books)

            if not book:
                continue

            if parsed["odds_provider"] is None:
                parsed["odds_provider"] = book.get("name")

            outcomes = book.get("outcomes", [])

            if market_name == "2way":
                for outcome in outcomes:
                    outcome_type = (outcome.get("type") or "").lower()

                    if outcome_type == "home":
                        parsed["current_moneyline_home"] = to_float(outcome.get("odds"))
                        parsed["opening_moneyline_home"] = to_float(outcome.get("opening_odds"))

                    elif outcome_type == "away":
                        parsed["current_moneyline_away"] = to_float(outcome.get("odds"))
                        parsed["opening_moneyline_away"] = to_float(outcome.get("opening_odds"))

            elif market_name == "spread":
                home_spread_found = None
                home_opening_spread_found = None
                away_spread_found = None
                away_opening_spread_found = None

                for outcome in outcomes:
                    outcome_type = (outcome.get("type") or "").lower()
                    current_spread = to_float(outcome.get("spread"))
                    opening_spread = to_float(outcome.get("opening_spread"))

                    if outcome_type == "home":
                        home_spread_found = current_spread
                        home_opening_spread_found = opening_spread

                    elif outcome_type == "away":
                        away_spread_found = current_spread
                        away_opening_spread_found = opening_spread

                # If both sides exist in the feed, use them directly
                if home_spread_found is not None:
                    parsed["home_current_spread"] = home_spread_found
                if away_spread_found is not None:
                    parsed["away_current_spread"] = away_spread_found

                if home_opening_spread_found is not None:
                    parsed["home_opening_spread"] = home_opening_spread_found
                if away_opening_spread_found is not None:
                    parsed["away_opening_spread"] = away_opening_spread_found

                # If only home side exists, derive away side
                if parsed["home_current_spread"] is not None and parsed["away_current_spread"] is None:
                    parsed["away_current_spread"] = -parsed["home_current_spread"]

                if parsed["home_opening_spread"] is not None and parsed["away_opening_spread"] is None:
                    parsed["away_opening_spread"] = -parsed["home_opening_spread"]

                # If only away side exists, derive home side
                if parsed["away_current_spread"] is not None and parsed["home_current_spread"] is None:
                    parsed["home_current_spread"] = -parsed["away_current_spread"]

                if parsed["away_opening_spread"] is not None and parsed["home_opening_spread"] is None:
                    parsed["home_opening_spread"] = -parsed["away_opening_spread"]

                # Keep these as home-perspective generic values if you still want them
                parsed["current_spread"] = parsed["home_current_spread"]
                parsed["opening_spread"] = parsed["home_opening_spread"]

            elif market_name in ("total", "totals", "overunder", "over_under"):
                for outcome in outcomes:
                    current_total = to_float(outcome.get("total") or outcome.get("spread"))
                    opening_total = to_float(outcome.get("opening_total") or outcome.get("opening_spread"))

                    if current_total is not None and parsed["current_total"] is None:
                        parsed["current_total"] = current_total
                    if opening_total is not None and parsed["opening_total"] is None:
                        parsed["opening_total"] = opening_total

        odds_map[game_id] = parsed

    return odds_map


def print_condensed_odds_data(odds_data):
    if not odds_data:
        print("=== CONDENSED ODDS DATA ===")
        print("No odds data returned.")
        return

    condensed_games = []

    for game in odds_data.get("games", []):
        condensed_game = {
            "gameId": game.get("gameId"),
            "homeTeamId": game.get("homeTeamId"),
            "awayTeamId": game.get("awayTeamId"),
            "markets": []
        }

        for market in game.get("markets", []):
            books_summary = []

            for book in market.get("books", []):
                books_summary.append({
                    "name": book.get("name"),
                    "countryCode": book.get("countryCode"),
                    "outcomes": book.get("outcomes", [])
                })

            condensed_game["markets"].append({
                "name": market.get("name"),
                "books": books_summary
            })

        condensed_games.append(condensed_game)

    # print("=== CONDENSED ODDS DATA ===")
    # print(json.dumps({"games": condensed_games}, indent=2)[:12000])


def get_today_nba_games():
    scoreboard_data = safe_get_json(SCOREBOARD_URL)
    odds_data = safe_get_json(ODDS_URL)

    print_condensed_odds_data(odds_data)

    games = parse_scoreboard_games(scoreboard_data)
    odds_map = parse_odds_games(odds_data)

    for game in games:
        odds = odds_map.get(game["nba_game_id"])
        if odds:
            game["opening_spread"] = odds.get("opening_spread")
            game["current_spread"] = odds.get("current_spread")

            game["home_opening_spread"] = odds.get("home_opening_spread")
            game["away_opening_spread"] = odds.get("away_opening_spread")
            game["home_current_spread"] = odds.get("home_current_spread")
            game["away_current_spread"] = odds.get("away_current_spread")

            game["opening_total"] = odds.get("opening_total")
            game["current_total"] = odds.get("current_total")

            game["opening_moneyline_home"] = odds.get("opening_moneyline_home")
            game["opening_moneyline_away"] = odds.get("opening_moneyline_away")
            game["current_moneyline_home"] = odds.get("current_moneyline_home")
            game["current_moneyline_away"] = odds.get("current_moneyline_away")

            game["odds_provider"] = odds.get("odds_provider")

    return games


if __name__ == "__main__":
    games = get_today_nba_games()

    if not games:
        print("No NBA games found for today.")
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        print(f"NBA games for today ({today}):\n")

        for game in games:
            print(f"{game['away_team_name']} at {game['home_team_name']}")
            print(f"Score: {game['away_score']} - {game['home_score']}")
            print(f"Status: {game['game_status']}")
            print(f"Tipoff (ET): {game['start_time_et']}")
            print(f"{game['home_team_name']} Current Spread: {game['home_current_spread']}")
            print(f"{game['away_team_name']} Current Spread: {game['away_current_spread']}")
            print(f"{game['home_team_name']} Opening Spread: {game['home_opening_spread']}")
            print(f"{game['away_team_name']} Opening Spread: {game['away_opening_spread']}")
            print(f"Book: {game['odds_provider']}")
            print("-" * 40 + "\n")