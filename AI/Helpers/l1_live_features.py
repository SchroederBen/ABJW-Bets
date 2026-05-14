"""
Live L1 / model feature alignment for AI matchup payloads.

Loads a JSON allow-list (same shape as l1_feature_selection ``*_features_reduced_*.json``:
a list of feature name strings). Computes game-level values from ``team_game_rows``
using the full box-score stats returned by ``fetch_historical_games_with_box_scores``.

Now computes ALL training-aligned features:
  - Rolling MAs for box-score stats (field_goal_pct, three_pt_pct, assists, etc.)
  - Rolling MAs for score-derived stats (points, team_score, opponent_score, team_win)
  - days_rest
  - opp_* (opponent's rolling stats)
  - diff_* (home minus away for every parallel feature)

Also supports loading the trained L1 model + scaler for direct probability scoring.

Environment:
  AI_L1_FEATURES_JSON — optional path override to a JSON list of feature names.

Default allow-list file (if env unset and this file exists):
  AI/config/l1_allowlist/features_reduced.json — written by L1 training with
  ``--export-live`` (``train_logistic_model`` / ``run_l1_feature_selection``).

If neither env nor default file yields a list, no ``l1_model_features`` block is added.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

# Same path as src.models.l1_feature_selection.default_ai_live_allowlist_path()
_DEFAULT_LIVE_ALLOWLIST = (
    Path(__file__).resolve().parents[1] / "config" / "l1_allowlist" / "features_reduced.json"
)

# Default location for trained model artifacts
_ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "src" / "models" / "artifacts"

# Rolling windows used in training pipelines (keep in sync with train_logistic_cover_model)
DEFAULT_WINDOWS = (3, 5, 10)

# All columns we can derive from full box-score history
# (matches build_pregame_features rolling_cols)
ROLLING_BASE_COLS = (
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
)

# Subset that can be derived from score-only games (fallback)
SCORE_ONLY_COLS = ("points", "team_score", "opponent_score", "team_win")


def load_l1_allowlist(path: str | Path | None) -> list[str] | None:
    """Load allow-list from JSON file; expects a JSON array of feature name strings."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return None
    return [str(x) for x in data]


def default_live_allowlist_path() -> Path:
    """Stable JSON path under ``AI/config/l1_allowlist/`` (latest L1 export)."""
    return _DEFAULT_LIVE_ALLOWLIST


def get_l1_allowlist_from_env() -> list[str] | None:
    """
    Resolve allow-list: ``AI_L1_FEATURES_JSON`` if set, else default
    ``AI/config/l1_allowlist/features_reduced.json`` if the file exists.
    """
    env_path = os.getenv("AI_L1_FEATURES_JSON", "").strip()
    if env_path:
        return load_l1_allowlist(env_path)
    return load_l1_allowlist(_DEFAULT_LIVE_ALLOWLIST)


# ======================================================================
# Model / scaler loading for live inference
# ======================================================================

_cached_l1_model = None
_cached_l1_scaler = None
_cached_l1_feature_cols = None
_l1_model_load_attempted = False


def load_l1_model_and_scaler() -> tuple[Any, Any, list[str] | None]:
    """
    Load the trained L1 model, scaler, and feature column list from artifacts.
    Returns (model, scaler, feature_cols) or (None, None, None) if not found.
    Caches after first load.
    """
    global _cached_l1_model, _cached_l1_scaler, _cached_l1_feature_cols
    global _l1_model_load_attempted

    if _l1_model_load_attempted:
        return _cached_l1_model, _cached_l1_scaler, _cached_l1_feature_cols

    _l1_model_load_attempted = True

    if not _ARTIFACTS_DIR.is_dir():
        return None, None, None

    # Find actual model file (not scaler, not calibration png)
    model_candidates = sorted(
        p for p in _ARTIFACTS_DIR.glob("l1_selection_*.joblib")
        if "scaler" not in p.name and "calibration" not in p.name
    )
    scaler_candidates = sorted(_ARTIFACTS_DIR.glob("l1_selection_scaler_*.joblib"))
    features_all_candidates = sorted(_ARTIFACTS_DIR.glob("l1_selection_features_all_*.json"))

    model_path = model_candidates[-1] if model_candidates else None
    scaler_path = scaler_candidates[-1] if scaler_candidates else None
    features_all_path = features_all_candidates[-1] if features_all_candidates else None

    if not model_path or not scaler_path or not features_all_path:
        print("  L1 model artifacts not found — model scoring disabled.")
        return None, None, None

    try:
        _cached_l1_model = joblib.load(model_path)
        _cached_l1_scaler = joblib.load(scaler_path)
        with open(features_all_path, encoding="utf-8") as f:
            _cached_l1_feature_cols = json.load(f)
        print(f"  Loaded L1 model: {model_path.name}")
        print(f"  Loaded L1 scaler: {scaler_path.name}")
        print(f"  L1 feature columns: {len(_cached_l1_feature_cols)}")
    except Exception as e:
        print(f"  Warning: Could not load L1 model artifacts: {e}")
        _cached_l1_model = None
        _cached_l1_scaler = None
        _cached_l1_feature_cols = None

    return _cached_l1_model, _cached_l1_scaler, _cached_l1_feature_cols


def score_with_l1_model(feature_row: dict[str, Any]) -> dict[str, Any] | None:
    """
    Score a single game's feature row using the trained L1 model.

    Returns a dict with win_probability and confidence, or None if model
    is not available or features are insufficient.
    """
    model, scaler, feature_cols = load_l1_model_and_scaler()
    if model is None or feature_cols is None:
        return None

    # Build the feature vector in training column order
    values = []
    null_count = 0
    for col in feature_cols:
        v = feature_row.get(col)
        if v is None:
            values.append(0.0)  # fill missing with 0 (same as training fillna)
            null_count += 1
        else:
            values.append(float(v))

    # If too many features are missing, don't trust the score
    null_pct = null_count / len(feature_cols) if feature_cols else 1.0
    if null_pct > 0.5:
        return {
            "l1_win_probability": None,
            "l1_confidence": None,
            "l1_null_feature_pct": round(null_pct, 3),
            "l1_model_available": True,
            "l1_score_usable": False,
        }

    X = np.array(values, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)
    p_home_win = float(model.predict_proba(X_scaled)[0, 1])

    return {
        "l1_win_probability": round(p_home_win, 4),
        "l1_confidence": round(abs(p_home_win - 0.5) * 200, 1),  # 0-100 scale
        "l1_null_feature_pct": round(null_pct, 3),
        "l1_model_available": True,
        "l1_score_usable": True,
    }


# ======================================================================
# Feature computation from historical game rows
# ======================================================================

def _parse_date(d: Any) -> date | None:
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        try:
            return datetime.fromisoformat(d[:10]).date()
        except ValueError:
            return None
    return None


def _safe_float(v: Any) -> float | None:
    """Convert a value to float, returning None if not possible."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _team_game_sequence_from_box_scores(
    team_id: int,
    team_game_rows: list[dict],
) -> list[dict]:
    """
    Chronological list of completed games for team_id with ALL box-score fields.
    Expects rows from fetch_historical_games_with_box_scores (2 rows per game,
    one per team).
    """
    rows: list[dict] = []
    for r in team_game_rows:
        if r.get("team_id") != team_id:
            continue

        row = {
            "date": r["date"],
            "points": _safe_float(r.get("points")),
            "team_score": _safe_float(r.get("team_score")),
            "opponent_score": _safe_float(r.get("opponent_score")),
            "team_win": _safe_float(r.get("team_win")),
            "field_goal_pct": _safe_float(r.get("field_goal_pct")),
            "three_pt_pct": _safe_float(r.get("three_pt_pct")),
            "free_throw_pct": _safe_float(r.get("free_throw_pct")),
            "offensive_rebounds": _safe_float(r.get("offensive_rebounds")),
            "defensive_rebounds": _safe_float(r.get("defensive_rebounds")),
            "total_rebounds": _safe_float(r.get("total_rebounds")),
            "assists": _safe_float(r.get("assists")),
            "steals": _safe_float(r.get("steals")),
            "blocks": _safe_float(r.get("blocks")),
            "turnovers": _safe_float(r.get("turnovers")),
            "personal_fouls": _safe_float(r.get("personal_fouls")),
        }
        rows.append(row)

    return rows


def _team_game_sequence_from_scores(
    team_id: int,
    historical_games: list[dict],
) -> list[dict]:
    """
    Fallback: build sequence from score-only games table (original behavior).
    """
    rows: list[dict] = []
    for g in sorted(
        historical_games,
        key=lambda x: (x.get("date"), x.get("game_id", 0)),
    ):
        hid, aid = g.get("home_team_id"), g.get("away_team_id")
        hs, aws = g.get("home_score"), g.get("away_score")
        if hs is None or aws is None:
            continue
        if hid == team_id:
            rows.append({
                "date": g["date"],
                "points": float(hs),
                "team_score": float(hs),
                "opponent_score": float(aws),
                "team_win": 1.0 if hs > aws else 0.0,
            })
        elif aid == team_id:
            rows.append({
                "date": g["date"],
                "points": float(aws),
                "team_score": float(aws),
                "opponent_score": float(hs),
                "team_win": 1.0 if aws > hs else 0.0,
            })
    return rows


def _mean_tail(seq: list[dict], col: str, window: int) -> float | None:
    if not seq:
        return None
    tail = seq[-window:]
    vals = [row[col] for row in tail if row.get(col) is not None]
    return sum(vals) / len(vals) if vals else None


def _team_pregame_stats(
    team_id: int,
    team_game_rows: list[dict] | None,
    historical_games: list[dict] | None,
    next_game_date: Any,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> dict[str, float | None]:
    """
    Pregame rolling stats for a team's next game.

    Uses team_game_rows (box-score data) when available, falls back to
    historical_games (score-only).
    """
    if team_game_rows is not None:
        seq = _team_game_sequence_from_box_scores(team_id, team_game_rows)
        available_cols = ROLLING_BASE_COLS
    elif historical_games is not None:
        seq = _team_game_sequence_from_scores(team_id, historical_games)
        available_cols = SCORE_ONLY_COLS
    else:
        return {}

    out: dict[str, float | None] = {}
    for w in windows:
        for col in available_cols:
            key = f"{col}_MA_{w}"
            out[key] = _mean_tail(seq, col, w)

    next_d = _parse_date(next_game_date)
    if seq and next_d:
        last_d = _parse_date(seq[-1]["date"])
        if last_d:
            out["days_rest"] = float((next_d - last_d).days)
        else:
            out["days_rest"] = None
    else:
        out["days_rest"] = None

    return out


def _pick_home_perspective_spread(g: dict) -> float | None:
    """Prefer opening line, then home opening, then current."""
    for key in (
        "opening_spread",
        "home_opening_spread",
        "home_current_spread",
    ):
        v = g.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def build_game_level_pregame_row(
    home_team_id: int,
    away_team_id: int,
    team_game_rows: list[dict] | None,
    historical_games: list[dict] | None,
    g: dict,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> dict[str, Any]:
    """
    One game-level row: home stats, opp_* = away,
    diff_* = home - away for ALL MA and days_rest features, plus spread fields.
    """
    next_date = g.get("date")
    h = _team_pregame_stats(home_team_id, team_game_rows, historical_games, next_date, windows)
    a = _team_pregame_stats(away_team_id, team_game_rows, historical_games, next_date, windows)

    row: dict[str, Any] = {}

    # Home team stats (raw names)
    for k, v in h.items():
        row[k] = v

    # Away team stats (opp_ prefix)
    for k, v in a.items():
        row[f"opp_{k}"] = v

    # Diff features: home - away for every parallel column
    for k in list(h.keys()):
        if "_MA_" not in k and k != "days_rest":
            continue
        hk, ok = h.get(k), a.get(k)
        if hk is not None and ok is not None:
            row[f"diff_{k}"] = hk - ok
        else:
            row[f"diff_{k}"] = None

    # Spread features
    spread = _pick_home_perspective_spread(g)
    row["opening_spread"] = spread
    row["abs_opening_spread"] = abs(spread) if spread is not None else None

    return row


def build_l1_model_features_subset(
    allow_list: list[str],
    home_team_id: int,
    away_team_id: int,
    team_game_rows: list[dict] | None,
    historical_games: list[dict] | None,
    g: dict,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> dict[str, Any]:
    """Full pregame row projected onto allow_list (missing keys -> None)."""
    full = build_game_level_pregame_row(
        home_team_id, away_team_id, team_game_rows, historical_games, g, windows
    )
    return {name: full.get(name) for name in allow_list}


def build_full_feature_row(
    home_team_id: int,
    away_team_id: int,
    team_game_rows: list[dict] | None,
    historical_games: list[dict] | None,
    g: dict,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> dict[str, Any]:
    """
    Build the complete feature row (all features, not just allow-list) for
    L1 model scoring. This matches the full training feature set.
    """
    return build_game_level_pregame_row(
        home_team_id, away_team_id, team_game_rows, historical_games, g, windows
    )