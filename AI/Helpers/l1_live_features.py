"""
Live L1 / model feature alignment for AI matchup payloads.

Loads a JSON allow-list (same shape as l1_feature_selection ``*_features_reduced_*.json``:
a list of feature name strings). Computes game-level values from ``historical_games``
using only fields available there (scores + dates), matching the naming used in
``train_logistic_cover_model`` / ``build_pregame_features`` for:

  points, team_score, opponent_score, team_win, days_rest, opening_spread, abs_opening_spread,
  opp_*, diff_*

Stats that require box-score extras (e.g. ``field_goal_pct_MA_5``) are left missing (None)
if the allow-list asks for them — extend the historical query later to fill those.

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

# Same path as src.models.l1_feature_selection.default_ai_live_allowlist_path()
_DEFAULT_LIVE_ALLOWLIST = (
    Path(__file__).resolve().parents[1] / "config" / "l1_allowlist" / "features_reduced.json"
)

# Rolling windows used in training pipelines (keep in sync with train_logistic_cover_model)
DEFAULT_WINDOWS = (3, 5, 10)

# Columns we can derive from score-only history (matches build_pregame_features rolling_cols subset)
ROLLING_BASE_COLS = ("points", "team_score", "opponent_score", "team_win")


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


def _team_game_sequence(team_id: int, historical_games: list[dict]) -> list[dict]:
    """Chronological list of completed games for team_id with training-aligned keys."""
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
            rows.append(
                {
                    "date": g["date"],
                    "points": float(hs),
                    "team_score": float(hs),
                    "opponent_score": float(aws),
                    "team_win": 1.0 if hs > aws else 0.0,
                }
            )
        elif aid == team_id:
            rows.append(
                {
                    "date": g["date"],
                    "points": float(aws),
                    "team_score": float(aws),
                    "opponent_score": float(hs),
                    "team_win": 1.0 if aws > hs else 0.0,
                }
            )
    return rows


def _mean_tail(seq: list[dict], col: str, window: int) -> float | None:
    if not seq:
        return None
    tail = seq[-window:]
    vals = [row[col] for row in tail]
    return sum(vals) / len(vals) if vals else None


def _team_pregame_stats(
    team_id: int,
    historical_games: list[dict],
    next_game_date: Any,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> dict[str, float | None]:
    """
    Pregame rolling stats for the next game (not in DB): MAs over completed games only;
    days_rest = gap between last game and next_game_date.
    """
    seq = _team_game_sequence(team_id, historical_games)
    out: dict[str, float | None] = {}
    for w in windows:
        for col in ROLLING_BASE_COLS:
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
    """Prefer opening line, then home opening, then current (live API often only has current)."""
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
    historical_games: list[dict],
    g: dict,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> dict[str, Any]:
    """
    One game-level row in the same spirit as train merge: home stats, opp_* = away,
    diff_* = home - away for MA and days_rest, plus spread fields.
    """
    next_date = g.get("date")
    h = _team_pregame_stats(home_team_id, historical_games, next_date, windows)
    a = _team_pregame_stats(away_team_id, historical_games, next_date, windows)

    row: dict[str, Any] = {}
    for k, v in h.items():
        row[k] = v
    for k, v in a.items():
        row[f"opp_{k}"] = v

    for k in list(h.keys()):
        if k != "days_rest" and "_MA_" not in k:
            continue
        hk, ok = h.get(k), a.get(k)
        if hk is not None and ok is not None:
            row[f"diff_{k}"] = hk - ok
        else:
            row[f"diff_{k}"] = None

    spread = _pick_home_perspective_spread(g)
    row["opening_spread"] = spread
    row["abs_opening_spread"] = abs(spread) if spread is not None else None

    return row


def build_l1_model_features_subset(
    allow_list: list[str],
    home_team_id: int,
    away_team_id: int,
    historical_games: list[dict],
    g: dict,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> dict[str, Any]:
    """Full pregame row projected onto allow_list (missing keys -> None)."""
    full = build_game_level_pregame_row(
        home_team_id, away_team_id, historical_games, g, windows
    )
    return {name: full.get(name) for name in allow_list}
