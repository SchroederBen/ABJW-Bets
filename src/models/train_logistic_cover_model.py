"""
Minimal logistic regression for P(home covers the opening spread).

Problem (binary classification):
    margin     = home_score - away_score
    cover      = 1 if margin + spread > 0 else 0   (spread = home perspective, e.g. -5.5)
    Pushes     = dropped (margin + spread == 0)

Features:
    - Rolling pregame stats (home + opp_-prefixed away), same source as build_pregame_features
    - opening_spread, abs_opening_spread  (spread encodes market expectation — include it)
    - Matchup diffs: diff_<stat> = home_stat - away_stat for parallel MA / days_rest columns

Model:
    Basic LogisticRegression (L2, fixed C) + StandardScaler, fit on train only.

After prediction:
    Compare p_model to a flat implied cover prob for -110 (~52.4%) for a simple edge summary.

Usage:
    python -m src.models.train_logistic_cover_model [--recent]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data.build_pregame_features import build_pregame_features

# Saved model + metrics land here (timestamped filenames per run)
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
# Rolling window lengths for MA features from build_pregame_features
WINDOWS = (3, 5, 10)
# Hold out the last fraction for test (chronological; no shuffle)
TEST_FRACTION = 0.20
# --recent: drop older seasons for a quicker / more modern slice
RECENT_CUTOFF = "2015-01-01"

# Typical -110 American odds: risk 1.10 to win 1.00 → implied prob ≈ 1.10 / (1+1.10)
IMPLIED_P_COVER_NEG110 = 1.10 / 2.10


def _build_game_dataset_cover(df: pd.DataFrame) -> pd.DataFrame:
    """One row per game: labels + home/opp features."""
    # Team-game rows use _MA_* and days_rest; same stats for home and away (renamed below)
    stat_cols = sorted(c for c in df.columns if "_MA_" in c or c == "days_rest")

    home = df[df["is_home"] == 1][
        ["game_id", "date", "home_score", "away_score", "opening_spread"] + stat_cols
    ].copy()
    away = df[df["is_away"] == 1][["game_id"] + stat_cols].copy()
    # Prefix away team's rolling stats so we can merge one row per game_id
    away = away.rename(columns={c: f"opp_{c}" for c in stat_cols})

    game_df = home.merge(away, on="game_id", how="inner")
    game_df = game_df.sort_values("date").reset_index(drop=True)

    game_df["home_margin"] = (
        pd.to_numeric(game_df["home_score"], errors="coerce")
        - pd.to_numeric(game_df["away_score"], errors="coerce")
    )
    game_df["opening_spread"] = pd.to_numeric(game_df["opening_spread"], errors="coerce")
    # ATS: home covers if final margin beats the line (spread is home perspective)
    game_df["margin_plus_spread"] = game_df["home_margin"] + game_df["opening_spread"]

    game_df = game_df[game_df["opening_spread"].notna()].copy()
    game_df = game_df[game_df["margin_plus_spread"].notna()].copy()
    # Drop pushes so the label stays strictly binary
    game_df = game_df[game_df["margin_plus_spread"] != 0].copy()

    game_df["home_cover"] = (game_df["margin_plus_spread"] > 0).astype(int)
    game_df["abs_opening_spread"] = game_df["opening_spread"].abs()

    # Remove post-game helpers; model must not see them as features
    game_df = game_df.drop(
        columns=["home_margin", "margin_plus_spread"],
        errors="ignore",
    )
    return game_df


def _add_matchup_diffs(game_df: pd.DataFrame) -> pd.DataFrame:
    """home_stat - away_stat for each parallel rolling column (matchup-relative signal)."""
    out = game_df.copy()
    for c in list(out.columns):
        if c.startswith("opp_"):
            continue
        if "_MA_" not in c and c != "days_rest":
            continue
        opp_c = f"opp_{c}"
        if opp_c not in out.columns:
            continue
        # Relative strength: e.g. home rolling points MA minus away rolling points MA
        out[f"diff_{c}"] = pd.to_numeric(out[c], errors="coerce") - pd.to_numeric(
            out[opp_c], errors="coerce"
        )
    return out


def _temporal_split(
    game_df: pd.DataFrame, test_frac: float = TEST_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Past = train, future = test (index order after sort by date)
    split_idx = int(len(game_df) * (1 - test_frac))
    return game_df.iloc[:split_idx].copy(), game_df.iloc[split_idx:].copy()


def _feature_columns(game_df: pd.DataFrame) -> list[str]:
    # Everything except ids, dates, label, and scores (scores are post-game)
    exclude = {
        "game_id",
        "date",
        "home_cover",
        "home_score",
        "away_score",
    }
    return [c for c in game_df.columns if c not in exclude]


def _edge_summary(p_model: np.ndarray, edge_threshold: float = 0.03) -> dict:
    """
    Flat reference: implied P(cover) ≈ IMPLIED_P_COVER_NEG110 for -110 both sides.
    edge = p_model - p_implied. Count how often |edge| > threshold (illustrative).
    """
    implied = IMPLIED_P_COVER_NEG110
    edge = p_model - implied
    # Simple rule-of-thumb bands (not optimized; for reporting only)
    strong_home = edge > edge_threshold
    strong_away = edge < -edge_threshold
    return {
        "p_implied_cover_neg110": float(implied),
        "edge_threshold": float(edge_threshold),
        "n_test": int(len(p_model)),
        "n_strong_home_edge": int(np.sum(strong_home)),
        "n_strong_away_edge": int(np.sum(strong_away)),
        "n_no_bet_zone": int(np.sum(~strong_home & ~strong_away)),
        "mean_edge": float(np.mean(edge)),
    }


def train_cover_model(*, recent_only: bool = False) -> dict:
    print("Loading pregame features...")
    # Team-game table: two rows per game (home + away); rolling stats are pregame-only
    raw = build_pregame_features(windows=WINDOWS)

    print("Building game-level dataset (home_cover vs opening spread)...")
    game_df = _build_game_dataset_cover(raw)
    game_df = _add_matchup_diffs(game_df)

    if recent_only:
        before = len(game_df)
        game_df = game_df[game_df["date"] >= RECENT_CUTOFF].reset_index(drop=True)
        print(f"  --recent: kept {len(game_df)}/{before} games (>= {RECENT_CUTOFF})")

    feature_cols = _feature_columns(game_df)
    print(f"  Games: {len(game_df)}  |  Features: {len(feature_cols)}")
    print(f"  Home cover rate: {game_df['home_cover'].mean():.4f}")

    train_df, test_df = _temporal_split(game_df)
    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df["home_cover"].values.astype(int)
    X_test = test_df[feature_cols].values.astype(float)
    y_test = test_df["home_cover"].values.astype(int)

    # Scale using train stats only (avoids leaking test distribution into coefficients)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("\nFitting LogisticRegression (L2, C=1.0)...")
    # Plain L2 logistic; no CV here (keep the script easy to read and tweak)
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    # Column 1 = P(home covers | x)
    p_train = model.predict_proba(X_train_s)[:, 1]
    p_test = model.predict_proba(X_test_s)[:, 1]

    metrics = {
        "target": "home_cover_opening_spread",
        "line": "opening_spread (home perspective)",
        "model": "LogisticRegression L2 C=1.0",
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_features": len(feature_cols),
        "train_log_loss": float(log_loss(y_train, p_train)),
        "test_log_loss": float(log_loss(y_test, p_test)),
        "test_brier": float(brier_score_loss(y_test, p_test)),
        "test_roc_auc": float(roc_auc_score(y_test, p_test)),
        "train_date_range": [
            str(train_df["date"].min().date()),
            str(train_df["date"].max().date()),
        ],
        "test_date_range": [
            str(test_df["date"].min().date()),
            str(test_df["date"].max().date()),
        ],
        # Compare model prob to a flat -110 implied prob on the test set
        "edge_vs_neg110": _edge_summary(p_test),
    }

    print("\n--- Metrics ---")
    print(f"  Train log loss: {metrics['train_log_loss']:.4f}")
    print(f"  Test log loss:  {metrics['test_log_loss']:.4f}")
    print(f"  Test Brier:     {metrics['test_brier']:.4f}")
    print(f"  Test ROC-AUC:   {metrics['test_roc_auc']:.4f}")
    ev = metrics["edge_vs_neg110"]
    print(
        f"\n--- Edge vs -110 implied P ~ {ev['p_implied_cover_neg110']:.3f} "
        f"(threshold +/-{ev['edge_threshold']}) ---"
    )
    print(f"  Strong home edge: {ev['n_strong_home_edge']}  |  Strong away edge: {ev['n_strong_away_edge']}")
    print(f"  No-bet zone:       {ev['n_no_bet_zone']}  |  Mean edge: {ev['mean_edge']:.4f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"cover_logit_basic_{ts}"

    # Bundle needed for inference: model + scaler + feature list (order must match training)
    paths = {
        "model": ARTIFACTS_DIR / f"{prefix}.joblib",
        "scaler": ARTIFACTS_DIR / f"{prefix}_scaler.joblib",
        "metrics": ARTIFACTS_DIR / f"{prefix}_metrics.json",
        "features": ARTIFACTS_DIR / f"{prefix}_features.json",
        "coefficients": ARTIFACTS_DIR / f"{prefix}_coefficients.csv",
    }

    joblib.dump(model, paths["model"])
    joblib.dump(scaler, paths["scaler"])
    with open(paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(paths["features"], "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    # Signed weights on scaled features (interpret with care: features are correlated)
    coef_df = pd.DataFrame(
        {"feature": feature_cols, "coefficient": model.coef_[0]}
    ).sort_values("coefficient", key=np.abs, ascending=False)
    coef_df.to_csv(paths["coefficients"], index=False)

    print(f"\n--- Saved: {paths['model'].name} ---")
    metrics["artifact_prefix"] = prefix
    return metrics


if __name__ == "__main__":
    # Optional: python -m src.models.train_logistic_cover_model --recent
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    recent = "--recent" in flags
    if recent:
        print(f"*** --recent: games >= {RECENT_CUTOFF} ***\n")
    train_cover_model(recent_only=recent)
