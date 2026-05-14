CREATE EXTENSION IF NOT EXISTS vector;

-- ===== TEAMS =====
CREATE TABLE IF NOT EXISTS teams (
  team_id     INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  team_name   TEXT NOT NULL UNIQUE
);

-- ===== GAMES =====
CREATE TABLE IF NOT EXISTS games (
  game_id           INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  date              DATE NOT NULL,

  home_team_id      INTEGER NOT NULL REFERENCES teams(team_id),
  away_team_id      INTEGER NOT NULL REFERENCES teams(team_id),

  closing_spread    DOUBLE PRECISION,
  closing_total     DOUBLE PRECISION,
  closing_moneyline DOUBLE PRECISION,

  CONSTRAINT games_home_away_different CHECK (home_team_id <> away_team_id)
);

-- (Optional but common) prevent duplicate matchups same day
CREATE UNIQUE INDEX IF NOT EXISTS ux_games_date_home_away
  ON games(date, home_team_id, away_team_id);

-- ===== CTG_STATS =====
CREATE TABLE IF NOT EXISTS ctg_stats (
  ctg_id             INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  game_id            INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
  team_id            INTEGER NOT NULL REFERENCES teams(team_id),

  efficiency_metrics DOUBLE PRECISION,
  four_factor_stats  DOUBLE PRECISION,
  pace               DOUBLE PRECISION,
  rest_days          INTEGER,
  travel_distance    INTEGER,

  -- usually at most one CTG row per (game, team)
  CONSTRAINT ux_ctg_game_team UNIQUE (game_id, team_id)
);

-- ===== BBR_DAYOF_STATS =====
CREATE TABLE IF NOT EXISTS bbr_dayof_stats (
  bbr_id              INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  game_id             INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
  team_id             INTEGER NOT NULL REFERENCES teams(team_id),

  season_averages     DOUBLE PRECISION,
  rolling_form        DOUBLE PRECISION,
  injuries            INTEGER,
  rotation_indicators INTEGER,

  CONSTRAINT ux_bbr_game_team UNIQUE (game_id, team_id)
);

-- ===== SPORTSBOOK_LINES =====
CREATE TABLE IF NOT EXISTS sportsbook_lines (
  line_id            INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  game_id            INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,

  opening_spread     DOUBLE PRECISION,
  current_spread     DOUBLE PRECISION,
  opening_total      DOUBLE PRECISION,
  current_total      DOUBLE PRECISION,
  opening_moneyline  DOUBLE PRECISION,
  current_moneyline  DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS ix_lines_game_id ON sportsbook_lines(game_id);

-- ===== LLM_CONTEXT_FEATURES =====
CREATE TABLE IF NOT EXISTS llm_context_features (
  context_id             INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  game_id                INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,

  market_efficiency_score DOUBLE PRECISION,
  public_bias_score       DOUBLE PRECISION,
  confidence              DOUBLE PRECISION,
  injury_uncertainty      DOUBLE PRECISION,
  spread_move             DOUBLE PRECISION,
  total_move              DOUBLE PRECISION,
  mispricing_score        DOUBLE PRECISION,
  volatility_flags        TEXT
);

CREATE INDEX IF NOT EXISTS ix_llm_game_id ON llm_context_features(game_id);

-- ===== MODEL_FEATURES =====
CREATE TABLE IF NOT EXISTS model_features (
  feature_id      INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  game_id         INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,

  feature_vector  vector,
  schema_version  TEXT NOT NULL,

  -- diagram says "has_one" per game
  CONSTRAINT ux_model_features_game UNIQUE (game_id)
);

/* Model Predictions */
CREATE TABLE IF NOT EXISTS model_predictions (
  prediction_id   INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  game_id         INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
  win_probability DOUBLE PRECISION,
  cover_probability DOUBLE PRECISION,
  total_probability DOUBLE PRECISION,
  fair_moneyline DOUBLE PRECISION, 
  fair_spread DOUBLE PRECISION,
  fair_total DOUBLE PRECISION, 
  model_version TEXT NOT NULL

);

/* Profitability Analysis */
CREATE TABLE IF NOT EXISTS profability_analysis ( 
    profit_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    prediction_id INTEGER NOT NULL REFERENCES model_predictions(prediction_id) ON DELETE CASCADE,
    expected_value DOUBLE PRECISION,
    stake_kelly DOUBLE PRECISION,
    is_profitable BOOLEAN


)

