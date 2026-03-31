Install dependencies
- From the project root:
  `pip install -r requirements.txt`

Database configuration
- The loader uses get_engine() from `src.utils.db_connection`. Ensure that function is configured to return a SQLAlchemy Engine connected to your database.
- Common setups:
  - Configure a DATABASE_URL environment variable if your db_connection reads it.
  - Or edit `src/utils/db_connection.py` to point to your local/dev DB.

Run the loader
- From the project root (Windows):
  `python -m src.data.load_raw_team_game_history`

- Or run interactively:
  python
  >>> from src.data.load_raw_team_game_history import load_raw_team_game_history
  >>> df = load_raw_team_game_history()
  >>> df.shape
  >>> df.columns.tolist()

Save output to CSV
- Two options:
  1) Uncomment the last line in `load_raw_team_game_history.py`:
     # df.tail(200).to_csv("src/data/raw_team_game_history_sample.csv", index=False)
  2) Save programmatically:
     >>> df.to_csv("src/data/raw_team_game_history_full.csv", index=False)

Build pregame features
- From the project root, run (quiet by default):
  `python -m src.data.build_pregame_features`

- Examples:
  - Build with rolling windows 3,5,10 and show progress/output:
    `python -m src.data.build_pregame_features --windows 3 5 10 --verbose`
  - Save a sample CSV:
    `python -m src.data.build_pregame_features --save-sample`

- Programmatic usage:
  ```python
  from src.data.build_pregame_features import build_pregame_features
  # no rolling averages
  df = build_pregame_features()
  # with rolling averages
  df = build_pregame_features(windows=(3,5,10))
  ```
