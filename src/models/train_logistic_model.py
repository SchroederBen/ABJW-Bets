"""
Logistic Regression Training Pipeline for NBA Game Prediction

Phase 1 – L2 Baseline (Ridge): Train LogisticRegressionCV with L2 penalty
    using TimeSeriesSplit cross-validation. Keeps all features but shrinks
    correlated coefficients toward each other.

Phase 2 – L1 Feature Selection (Lasso) & Elastic Net: Retrain with L1 penalty
    to drive irrelevant coefficients to zero, identify droppable features, and
    compare against the L2 baseline. Also trains an Elastic Net blend.

Artifacts saved to src/models/artifacts/.

CLI: ``python -m src.models.train_logistic_model l1|all [--recent] [--export-live]``
writes ``AI/config/l1_allowlist/features_reduced.json`` (overwrites) for live AI payloads.
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.data.build_pregame_features import build_pregame_features
from src.models.l1_feature_selection import write_ai_live_allowlist_json

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
WINDOWS = (3, 5, 10)

# LogisticRegressionCV: Cs=int uses logspace(1e-4, 1e4) and often picks huge C → no L1 zeros.
# Smaller C = stronger L1 / more sparsity; cap max C so CV explores meaningful shrinkage.
L1_CS_GRID = np.logspace(-3, 1.5, 16)  # ~0.001 .. ~31.6
ENET_CS_GRID = np.logspace(-2, 1.5, 12)  # ~0.01 .. ~31.6 (saga ENET phase)
TEST_FRACTION = 0.20
RECENT_CUTOFF = "2015-01-01"


def _build_game_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Merge home and away team features into one row per game.

    Each game in the raw data has two rows (one per team).  This function
    pivots them so the home team's rolling stats become the main features
    and the away team's rolling stats are prefixed with ``opp_``.

    Target variable is ``team_win`` from the home team's row (1 = home win).
    """
    feature_cols = sorted(
        c for c in df.columns if "_MA_" in c or c == "days_rest"
    )

    home = df[df["is_home"] == 1][["game_id", "date", "team_win"] + feature_cols].copy()
    away = df[df["is_away"] == 1][["game_id"] + feature_cols].copy()

    away = away.rename(columns={c: f"opp_{c}" for c in feature_cols})

    game_df = home.merge(away, on="game_id", how="inner")
    game_df = game_df.sort_values("date").reset_index(drop=True)
    return game_df


def _temporal_split(
    game_df: pd.DataFrame, test_frac: float = TEST_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by chronological order, reserving the latest *test_frac* games."""
    split_idx = int(len(game_df) * (1 - test_frac))
    return game_df.iloc[:split_idx].copy(), game_df.iloc[split_idx:].copy()


def _feature_columns(game_df: pd.DataFrame) -> list[str]:
    """Return feature column names (everything except identifiers and target)."""
    exclude = {"game_id", "date", "team_win"}
    return [c for c in game_df.columns if c not in exclude]


# ======================================================================
# Shared data preparation
# ======================================================================

def _prepare_data(recent_only: bool = False) -> dict:
    """Load pregame features, build game dataset, temporal-split, and scale.

    Returns a dict consumed by both Phase 1 (L2) and Phase 2 (L1) trainers
    so the database is only hit once when running the full pipeline.

    If *recent_only* is True, filters to games on or after RECENT_CUTOFF
    to speed up development / testing iterations.
    """
    print("Loading pregame features...")
    df = build_pregame_features(windows=WINDOWS)

    print("Building game-level dataset (merging home / away)...")
    game_df = _build_game_dataset(df)

    if recent_only:
        before = len(game_df)
        game_df = game_df[game_df["date"] >= RECENT_CUTOFF].reset_index(drop=True)
        print(f"  --recent: kept {len(game_df)}/{before} games (>= {RECENT_CUTOFF})")

    feature_cols = _feature_columns(game_df)

    print(f"  Games: {len(game_df)}  |  Features: {len(feature_cols)}")

    train_df, test_df = _temporal_split(game_df)
    print(
        f"  Train: {len(train_df)} games  "
        f"({train_df['date'].min().date()} -> {train_df['date'].max().date()})"
    )
    print(
        f"  Test:  {len(test_df)} games  "
        f"({test_df['date'].min().date()} -> {test_df['date'].max().date()})"
    )

    X_train = train_df[feature_cols].values
    y_train = train_df["team_win"].values.astype(int)
    X_test = test_df[feature_cols].values
    y_test = test_df["team_win"].values.astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "feature_cols": feature_cols,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df,
        "scaler": scaler,
    }


def _load_latest_l2_metrics() -> dict | None:
    """Load the most recent L2 baseline metrics JSON from artifacts."""
    candidates = sorted(ARTIFACTS_DIR.glob("l2_baseline_metrics_*.json"))
    if not candidates:
        return None
    with open(candidates[-1]) as f:
        return json.load(f)


# ======================================================================
# Phase 1 – L2 Baseline (Ridge)
# ======================================================================

def train_l2_baseline(data: dict | None = None, *, recent_only: bool = False) -> tuple:
    """Train the Phase 1 L2 logistic regression baseline and persist artifacts."""

    if data is None:
        data = _prepare_data(recent_only=recent_only)

    feature_cols = data["feature_cols"]
    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    train_df = data["train_df"]
    test_df = data["test_df"]
    scaler = data["scaler"]

    # ------------------------------------------------------------------
    # Train LogisticRegressionCV  (L2 / Ridge, TimeSeriesSplit)
    # ------------------------------------------------------------------
    tscv = TimeSeriesSplit(n_splits=5)

    print("\nTraining LogisticRegressionCV (L2 / Ridge)...")
    model = LogisticRegressionCV(
        penalty="l2",
        Cs=20,
        cv=tscv,
        solver="lbfgs",
        max_iter=1000,
        scoring="neg_log_loss",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    best_C = float(model.C_[0])
    print(f"  Best C (inverse regularisation strength): {best_C:.6f}")

    # ------------------------------------------------------------------
    # Evaluate on held-out test set
    # ------------------------------------------------------------------
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    metrics: dict = {
        "phase": "L2_baseline",
        "log_loss": float(log_loss(y_test, y_prob)),
        "brier_score": float(brier_score_loss(y_test, y_prob)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "best_C": best_C,
        "n_features": len(feature_cols),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_date_range": [
            str(train_df["date"].min().date()),
            str(train_df["date"].max().date()),
        ],
        "test_date_range": [
            str(test_df["date"].min().date()),
            str(test_df["date"].max().date()),
        ],
    }

    print("\n--- Test-Set Metrics ---")
    print(f"  Log Loss:    {metrics['log_loss']:.4f}")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")

    # ------------------------------------------------------------------
    # Calibration assessment (& optional Platt scaling)
    # ------------------------------------------------------------------
    prob_true, prob_pred = calibration_curve(
        y_test, y_prob, n_bins=10, strategy="uniform"
    )
    max_cal_error = float(np.max(np.abs(prob_true - prob_pred)))
    metrics["max_calibration_error"] = max_cal_error
    print(f"  Max Cal Err: {max_cal_error:.4f}")

    calibrated_model = None
    prob_true_cal = prob_pred_cal = None

    if max_cal_error > 0.05:
        print(
            "\n  Calibration error > 0.05 -- "
            "applying CalibratedClassifierCV (sigmoid)..."
        )
        calibrated_model = CalibratedClassifierCV(
            model, method="sigmoid", cv=tscv
        )
        calibrated_model.fit(X_train_scaled, y_train)

        y_prob_cal = calibrated_model.predict_proba(X_test_scaled)[:, 1]
        metrics["calibrated_log_loss"] = float(log_loss(y_test, y_prob_cal))
        metrics["calibrated_brier_score"] = float(
            brier_score_loss(y_test, y_prob_cal)
        )
        print(f"  Calibrated Log Loss:    {metrics['calibrated_log_loss']:.4f}")
        print(
            f"  Calibrated Brier Score: {metrics['calibrated_brier_score']:.4f}"
        )

        prob_true_cal, prob_pred_cal = calibration_curve(
            y_test, y_prob_cal, n_bins=10, strategy="uniform"
        )
    else:
        print("\n  Calibration within tolerance (<= 0.05) -- no post-hoc fix needed.")

    # ------------------------------------------------------------------
    # Feature importance (coefficients)
    # ------------------------------------------------------------------
    coef_df = (
        pd.DataFrame({"feature": feature_cols, "coefficient": model.coef_[0]})
        .assign(abs_coefficient=lambda d: d["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .drop(columns="abs_coefficient")
        .reset_index(drop=True)
    )

    print("\n--- Top 15 Features by |Coefficient| ---")
    print(coef_df.head(15).to_string(index=False))

    # ------------------------------------------------------------------
    # Persist artifacts
    # ------------------------------------------------------------------
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    final_model = calibrated_model if calibrated_model else model

    paths = {
        "model": ARTIFACTS_DIR / f"l2_baseline_{ts}.joblib",
        "scaler": ARTIFACTS_DIR / f"l2_baseline_scaler_{ts}.joblib",
        "metrics": ARTIFACTS_DIR / f"l2_baseline_metrics_{ts}.json",
        "features": ARTIFACTS_DIR / f"l2_baseline_features_{ts}.json",
        "coefficients": ARTIFACTS_DIR / f"l2_baseline_coefficients_{ts}.csv",
        "calibration_plot": ARTIFACTS_DIR / f"l2_baseline_calibration_{ts}.png",
    }

    joblib.dump(final_model, paths["model"])
    joblib.dump(scaler, paths["scaler"])

    with open(paths["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    with open(paths["features"], "w") as f:
        json.dump(feature_cols, f, indent=2)

    coef_df.to_csv(paths["coefficients"], index=False)

    # Calibration plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, "s-", label="Uncalibrated")
    if prob_true_cal is not None:
        ax.plot(prob_pred_cal, prob_true_cal, "^-", label="Calibrated (sigmoid)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Plot – L2 Baseline")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(paths["calibration_plot"], dpi=150)
    plt.close(fig)

    print(f"\n--- Artifacts saved to {ARTIFACTS_DIR} ---")
    for label, p in paths.items():
        print(f"  {label:>18s}: {p.name}")

    return final_model, scaler, metrics


# ======================================================================
# Phase 2 – L1 Feature Selection (Lasso) & Elastic Net
# ======================================================================

def train_l1_selection(
    data: dict | None = None,
    l2_metrics: dict | None = None,
    *,
    recent_only: bool = False,
    export_live_allowlist: bool = False,
) -> tuple:
    """Train L1 (Lasso) and Elastic Net models, compare against L2 baseline.

    Workflow:
        1. Fit L1 logistic regression (liblinear solver) to identify features
           whose coefficients are driven to zero.
        2. Fit Elastic Net (blend of L1 + L2) across several l1_ratio values.
        3. Print a side-by-side comparison of L2, L1, and Elastic Net.
        4. Persist the L1 model, reduced feature list, and metrics.

    export_live_allowlist
        If True, also overwrite ``AI/config/l1_allowlist/features_reduced.json`` for
        ``l1_live_features`` (use ``--export-live`` on the CLI).
    """

    if data is None:
        data = _prepare_data(recent_only=recent_only)

    if l2_metrics is None:
        l2_metrics = _load_latest_l2_metrics()

    feature_cols = data["feature_cols"]
    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    train_df = data["train_df"]
    test_df = data["test_df"]
    scaler = data["scaler"]

    tscv = TimeSeriesSplit(n_splits=5)

    # ------------------------------------------------------------------
    # 1. L1 (Lasso) logistic regression
    # ------------------------------------------------------------------
    print("\nTraining LogisticRegressionCV (L1 / Lasso)...")
    l1_model = LogisticRegressionCV(
        penalty="l1",
        Cs=L1_CS_GRID,
        cv=tscv,
        solver="liblinear",
        max_iter=1000,
        scoring="neg_log_loss",
        random_state=42,
    )
    l1_model.fit(X_train_scaled, y_train)

    best_C_l1 = float(l1_model.C_[0])
    print(f"  Best C (L1): {best_C_l1:.6f}")

    # Surviving vs dropped features
    l1_coef = l1_model.coef_[0]
    nonzero_mask = l1_coef != 0
    surviving_features = [f for f, nz in zip(feature_cols, nonzero_mask) if nz]
    dropped_features = [f for f, nz in zip(feature_cols, nonzero_mask) if not nz]

    print(
        f"\n  Features surviving L1: {len(surviving_features)} / {len(feature_cols)}"
    )
    print(f"  Features dropped (zero coefficient): {len(dropped_features)}")

    if dropped_features:
        print("\n  Dropped features:")
        for f in sorted(dropped_features):
            print(f"    - {f}")

    # Evaluate L1 on test set
    y_prob_l1 = l1_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_l1 = l1_model.predict(X_test_scaled)

    l1_metrics: dict = {
        "phase": "L1_selection",
        "log_loss": float(log_loss(y_test, y_prob_l1)),
        "brier_score": float(brier_score_loss(y_test, y_prob_l1)),
        "roc_auc": float(roc_auc_score(y_test, y_prob_l1)),
        "accuracy": float(accuracy_score(y_test, y_pred_l1)),
        "best_C": best_C_l1,
        "n_features_total": len(feature_cols),
        "n_features_surviving": len(surviving_features),
        "n_features_dropped": len(dropped_features),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_date_range": [
            str(train_df["date"].min().date()),
            str(train_df["date"].max().date()),
        ],
        "test_date_range": [
            str(test_df["date"].min().date()),
            str(test_df["date"].max().date()),
        ],
    }

    print("\n--- L1 Test-Set Metrics ---")
    print(f"  Log Loss:    {l1_metrics['log_loss']:.4f}")
    print(f"  Brier Score: {l1_metrics['brier_score']:.4f}")
    print(f"  ROC-AUC:     {l1_metrics['roc_auc']:.4f}")
    print(f"  Accuracy:    {l1_metrics['accuracy']:.4f}")

    # ------------------------------------------------------------------
    # 2. Elastic Net (blend of L1 + L2)
    # ------------------------------------------------------------------
    print("\nTraining LogisticRegressionCV (Elastic Net)...")
    enet_model = LogisticRegressionCV(
        penalty="elasticnet",
        Cs=ENET_CS_GRID,
        cv=tscv,
        solver="saga",
        l1_ratios=[0.25, 0.5, 0.75],
        max_iter=3000,
        scoring="neg_log_loss",
        random_state=42,
    )
    enet_model.fit(X_train_scaled, y_train)

    best_C_enet = float(enet_model.C_[0])
    best_l1_ratio = float(enet_model.l1_ratio_[0])
    enet_coef = enet_model.coef_[0]
    enet_nonzero = int(np.count_nonzero(enet_coef))
    print(
        f"  Best C: {best_C_enet:.6f}  |  "
        f"Best l1_ratio: {best_l1_ratio:.2f}  |  "
        f"Non-zero features: {enet_nonzero}/{len(feature_cols)}"
    )

    y_prob_enet = enet_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_enet = enet_model.predict(X_test_scaled)

    enet_metrics: dict = {
        "log_loss": float(log_loss(y_test, y_prob_enet)),
        "brier_score": float(brier_score_loss(y_test, y_prob_enet)),
        "roc_auc": float(roc_auc_score(y_test, y_prob_enet)),
        "accuracy": float(accuracy_score(y_test, y_pred_enet)),
        "best_C": best_C_enet,
        "best_l1_ratio": best_l1_ratio,
        "n_features_nonzero": enet_nonzero,
    }

    # ------------------------------------------------------------------
    # 3. Side-by-side comparison
    # ------------------------------------------------------------------
    l2_n_feat = l2_metrics.get("n_features", len(feature_cols)) if l2_metrics else "?"

    print("\n" + "=" * 65)
    print("  MODEL COMPARISON (vs L2 Baseline)")
    print("=" * 65)
    header = (
        f"  {'Metric':<22s} {'L2 Baseline':>12s} "
        f"{'L1 (Lasso)':>12s} {'Elastic Net':>12s}"
    )
    print(header)
    print("  " + "-" * 61)

    rows = [
        ("Log Loss", "log_loss"),
        ("Brier Score", "brier_score"),
        ("ROC-AUC", "roc_auc"),
        ("Accuracy", "accuracy"),
    ]
    for label, key in rows:
        l2_val = l2_metrics.get(key, float("nan")) if l2_metrics else float("nan")
        print(
            f"  {label:<22s} {l2_val:>12.4f} "
            f"{l1_metrics[key]:>12.4f} {enet_metrics[key]:>12.4f}"
        )

    print(
        f"  {'Non-zero Features':<22s} {str(l2_n_feat):>12s} "
        f"{len(surviving_features):>12d} {enet_nonzero:>12d}"
    )
    print("=" * 65)

    l1_metrics["elastic_net"] = enet_metrics
    l1_metrics["surviving_features"] = sorted(surviving_features)
    l1_metrics["dropped_features"] = sorted(dropped_features)

    # ------------------------------------------------------------------
    # 4. Calibration assessment (& optional Platt scaling)
    # ------------------------------------------------------------------
    prob_true_l1, prob_pred_l1 = calibration_curve(
        y_test, y_prob_l1, n_bins=10, strategy="uniform"
    )
    max_cal_error_l1 = float(np.max(np.abs(prob_true_l1 - prob_pred_l1)))
    l1_metrics["max_calibration_error"] = max_cal_error_l1
    print(f"\n  L1 Max Calibration Error: {max_cal_error_l1:.4f}")

    calibrated_model = None
    prob_true_cal = prob_pred_cal = None

    if max_cal_error_l1 > 0.05:
        print(
            "  Calibration error > 0.05 -- "
            "applying CalibratedClassifierCV (sigmoid)..."
        )
        calibrated_model = CalibratedClassifierCV(
            l1_model, method="sigmoid", cv=tscv
        )
        calibrated_model.fit(X_train_scaled, y_train)

        y_prob_cal = calibrated_model.predict_proba(X_test_scaled)[:, 1]
        l1_metrics["calibrated_log_loss"] = float(log_loss(y_test, y_prob_cal))
        l1_metrics["calibrated_brier_score"] = float(
            brier_score_loss(y_test, y_prob_cal)
        )
        print(f"  Calibrated Log Loss:    {l1_metrics['calibrated_log_loss']:.4f}")
        print(
            f"  Calibrated Brier Score: {l1_metrics['calibrated_brier_score']:.4f}"
        )

        prob_true_cal, prob_pred_cal = calibration_curve(
            y_test, y_prob_cal, n_bins=10, strategy="uniform"
        )
    else:
        print("  Calibration within tolerance (<= 0.05) -- no post-hoc fix needed.")

    # ------------------------------------------------------------------
    # 5. L1 coefficient analysis
    # ------------------------------------------------------------------
    coef_df = (
        pd.DataFrame({"feature": feature_cols, "coefficient": l1_coef})
        .assign(abs_coefficient=lambda d: d["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .drop(columns="abs_coefficient")
        .reset_index(drop=True)
    )
    coef_nonzero_df = coef_df[coef_df["coefficient"] != 0].reset_index(drop=True)

    print("\n--- L1 Surviving Features by |Coefficient| ---")
    print(coef_nonzero_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 6. Persist artifacts
    # ------------------------------------------------------------------
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    final_model = calibrated_model if calibrated_model else l1_model

    paths = {
        "model": ARTIFACTS_DIR / f"l1_selection_{ts}.joblib",
        "scaler": ARTIFACTS_DIR / f"l1_selection_scaler_{ts}.joblib",
        "metrics": ARTIFACTS_DIR / f"l1_selection_metrics_{ts}.json",
        "features_all": ARTIFACTS_DIR / f"l1_selection_features_all_{ts}.json",
        "features_reduced": ARTIFACTS_DIR / f"l1_selection_features_reduced_{ts}.json",
        "coefficients": ARTIFACTS_DIR / f"l1_selection_coefficients_{ts}.csv",
        "calibration_plot": ARTIFACTS_DIR / f"l1_selection_calibration_{ts}.png",
        "enet_model": ARTIFACTS_DIR / f"elastic_net_{ts}.joblib",
    }

    joblib.dump(final_model, paths["model"])
    joblib.dump(scaler, paths["scaler"])
    joblib.dump(enet_model, paths["enet_model"])

    with open(paths["metrics"], "w") as f:
        json.dump(l1_metrics, f, indent=2)

    with open(paths["features_all"], "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(paths["features_reduced"], "w") as f:
        json.dump(sorted(surviving_features), f, indent=2)

    if export_live_allowlist:
        live_p = write_ai_live_allowlist_json(surviving_features)
        print(f"  Live AI allow-list (overwrite): {live_p}")

    coef_df.to_csv(paths["coefficients"], index=False)

    # Calibration plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(prob_pred_l1, prob_true_l1, "s-", label="L1 (uncalibrated)")
    if prob_true_cal is not None:
        ax.plot(prob_pred_cal, prob_true_cal, "^-", label="L1 calibrated (sigmoid)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Plot – L1 Selection")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(paths["calibration_plot"], dpi=150)
    plt.close(fig)

    print(f"\n--- Artifacts saved to {ARTIFACTS_DIR} ---")
    for label, p in paths.items():
        print(f"  {label:>18s}: {p.name}")

    return final_model, scaler, l1_metrics


# ======================================================================
# Full pipeline
# ======================================================================

def train_pipeline(
    *, recent_only: bool = False, export_live_allowlist: bool = False
) -> None:
    """Run Phase 1 (L2) then Phase 2 (L1) sharing a single data load."""
    data = _prepare_data(recent_only=recent_only)

    print("\n" + "=" * 65)
    print("  PHASE 1 -- L2 BASELINE (Ridge)")
    print("=" * 65)
    _, _, l2_metrics = train_l2_baseline(data=data)

    print("\n" + "=" * 65)
    print("  PHASE 2 -- L1 FEATURE SELECTION (Lasso) & ELASTIC NET")
    print("=" * 65)
    train_l1_selection(
        data=data,
        l2_metrics=l2_metrics,
        export_live_allowlist=export_live_allowlist,
    )


if __name__ == "__main__":
    import sys

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    phase = args[0] if args else "all"
    recent = "--recent" in flags
    export_live = "--export-live" in flags

    if recent:
        print(f"*** --recent mode: only games >= {RECENT_CUTOFF} ***\n")
    if export_live:
        print("*** --export-live: write AI/config/l1_allowlist/features_reduced.json ***\n")

    if phase == "l2":
        train_l2_baseline(recent_only=recent)
    elif phase == "l1":
        train_l1_selection(recent_only=recent, export_live_allowlist=export_live)
    elif phase == "all":
        train_pipeline(recent_only=recent, export_live_allowlist=export_live)
    else:
        print(
            f"Usage: {sys.argv[0]} [l2|l1|all] [--recent] [--export-live]"
        )
        sys.exit(1)
