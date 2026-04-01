"""
Reusable L1 (Lasso) logistic regression for feature selection.

Use this module from any training pipeline (cover, win, etc.) and from LLM tooling:
load the saved ``*_features_reduced_*.json`` to know which columns to compute
for prompts / llm_context_features / matchup payloads.

Typical artifacts (when ``artifacts_dir`` is set):
    - ``{prefix}_{ts}.joblib`` — fitted LogisticRegressionCV
    - ``{prefix}_scaler_{ts}.joblib`` — StandardScaler (fit on train)
    - ``{prefix}_metrics_{ts}.json``
    - ``{prefix}_features_all_{ts}.json`` — full column list (training order)
    - ``{prefix}_features_reduced_{ts}.json`` — surviving feature names (for LLM)
    - ``{prefix}_coefficients_{ts}.csv``
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


@dataclass
class L1FeatureSelectionResult:
    """Return value from :func:`run_l1_feature_selection`."""

    model: LogisticRegressionCV
    metrics: dict[str, Any]
    surviving_features: list[str]
    dropped_features: list[str]
    paths: dict[str, Path] | None
    feature_cols: list[str]


def load_surviving_feature_names(reduced_json_path: str | Path) -> list[str]:
    """Load feature names from a ``*_features_reduced_*.json`` artifact."""
    path = Path(reduced_json_path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list of feature names, got {type(data)}")
    return list(data)


def run_l1_feature_selection(
    *,
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler: StandardScaler,
    artifact_prefix: str = "l1_selection",
    feature_list_prefix: str | None = None,
    artifacts_dir: Path | None = DEFAULT_ARTIFACTS_DIR,
    target_name: str = "y",
    target_description: str | None = None,
    phase_name: str = "L1_selection",
    baseline_metrics: dict | None = None,
    print_header: str = "Training LogisticRegressionCV (L1 / Lasso)...",
    cs: int = 10,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> L1FeatureSelectionResult:
    """
    Fit L1 logistic regression, identify surviving vs dropped features, evaluate
    on the test set, and optionally persist artifacts.

    Parameters
    ----------
    target_name
        Label column name for metrics (e.g. ``home_cover``, ``team_win``).
    target_description
        Optional extra line in metrics (e.g. ``opening_spread``).
    baseline_metrics
        If provided, ``log_loss`` / ``roc_auc`` are copied in as
        ``l2_baseline_*`` when those keys exist (for comparison prints).
    save_artifacts
        If False, no files are written; ``paths`` in the result is None.
    artifact_prefix
        Prefix for model, scaler, and metrics files.
    feature_list_prefix
        If set, used for ``*_features_all_*``, ``*_features_reduced_*``, and
        ``*_coefficients_*`` filenames; otherwise *artifact_prefix* is used for all.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    print(f"\n{print_header}")

    model = LogisticRegressionCV(
        penalty="l1",
        Cs=cs,
        cv=tscv,
        solver="liblinear",
        max_iter=1000,
        scoring="neg_log_loss",
        random_state=random_state,
    )
    model.fit(X_train_scaled, y_train)

    best_C = float(model.C_[0])
    print(f"  Best C (L1): {best_C:.6f}")

    l1_coef = model.coef_[0]
    nonzero_mask = l1_coef != 0
    surviving = [f for f, nz in zip(feature_cols, nonzero_mask) if nz]
    dropped = [f for f, nz in zip(feature_cols, nonzero_mask) if not nz]

    print(f"  Surviving features: {len(surviving)} / {len(feature_cols)}")
    if dropped:
        preview = sorted(dropped)[:20]
        extra = " ..." if len(dropped) > 20 else ""
        print(f"  Dropped ({len(dropped)}): {preview}{extra}")

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    metrics: dict[str, Any] = {
        "phase": phase_name,
        "target": target_name,
        "log_loss": float(log_loss(y_test, y_prob)),
        "brier_score": float(brier_score_loss(y_test, y_prob)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "best_C": best_C,
        "n_features_total": len(feature_cols),
        "n_features_surviving": len(surviving),
        "n_features_dropped": len(dropped),
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
        "surviving_features": sorted(surviving),
        "dropped_features": sorted(dropped),
    }

    if target_description is not None:
        metrics["line"] = target_description

    if baseline_metrics is not None:
        if "log_loss" in baseline_metrics:
            metrics["l2_baseline_log_loss"] = baseline_metrics["log_loss"]
        if "roc_auc" in baseline_metrics:
            metrics["l2_baseline_roc_auc"] = baseline_metrics["roc_auc"]

    print("\n--- L1 Test-Set Metrics ---")
    print(f"  Log Loss:    {metrics['log_loss']:.4f}")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")

    paths: dict[str, Path] | None = None
    if save_artifacts and artifacts_dir is not None:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = feature_list_prefix if feature_list_prefix is not None else artifact_prefix

        paths = {
            "model": artifacts_dir / f"{artifact_prefix}_{ts}.joblib",
            "scaler": artifacts_dir / f"{artifact_prefix}_scaler_{ts}.joblib",
            "metrics": artifacts_dir / f"{artifact_prefix}_metrics_{ts}.json",
            "features_all": artifacts_dir / f"{fp}_features_all_{ts}.json",
            "features_reduced": artifacts_dir / f"{fp}_features_reduced_{ts}.json",
            "coefficients": artifacts_dir / f"{fp}_coefficients_{ts}.csv",
        }

        joblib.dump(model, paths["model"])
        joblib.dump(scaler, paths["scaler"])

        metrics_out = {k: v for k, v in metrics.items() if k not in ("surviving_features", "dropped_features")}
        metrics_out["surviving_features_count"] = len(surviving)
        metrics_out["dropped_features_count"] = len(dropped)

        with open(paths["metrics"], "w", encoding="utf-8") as f:
            json.dump(metrics_out, f, indent=2)

        with open(paths["features_all"], "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, indent=2)

        with open(paths["features_reduced"], "w", encoding="utf-8") as f:
            json.dump(sorted(surviving), f, indent=2)

        coef_df = (
            pd.DataFrame({"feature": feature_cols, "coefficient": l1_coef})
            .assign(abs_coefficient=lambda d: d["coefficient"].abs())
            .sort_values("abs_coefficient", ascending=False)
        )
        coef_df.to_csv(paths["coefficients"], index=False)

        print(f"\n--- Artifacts saved to {artifacts_dir} ---")
        for label, p in paths.items():
            print(f"  {label:>18s}: {p.name}")
        print(
            f"\n  LLM / context: use features_reduced file for column allow-list "
            f"({paths['features_reduced'].name})"
        )

    return L1FeatureSelectionResult(
        model=model,
        metrics=metrics,
        surviving_features=surviving,
        dropped_features=dropped,
        paths=paths,
        feature_cols=feature_cols,
    )
