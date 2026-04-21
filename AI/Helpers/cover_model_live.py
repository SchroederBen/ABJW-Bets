"""
Live cover-model scoring for AI matchup payloads.

Loads the trained logistic cover model (from ``train_logistic_cover_model.py``)
and produces a per-game ``cover_model_score`` block for the AI payload.

Artifacts expected in ``src/models/artifacts/`` (written by the training script):
    - ``cover_logit_basic_{ts}.joblib``          — fitted LogisticRegression
    - ``cover_logit_basic_{ts}_scaler.joblib``   — StandardScaler fit on train
    - ``cover_logit_basic_{ts}_features.json``   — training column order

Shares the same pregame feature row shape as the L1 win model, so we reuse
``build_full_feature_row`` from ``l1_live_features`` to construct the input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


_ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "src" / "models" / "artifacts"

_cached_model = None
_cached_scaler = None
_cached_feature_cols: list[str] | None = None
_load_attempted = False


def load_cover_model_and_scaler() -> tuple[Any, Any, list[str] | None]:
    """
    Load the latest trained cover model, scaler, and feature column list.
    Returns (model, scaler, feature_cols) or (None, None, None) if not found.
    Caches after the first call.
    """
    global _cached_model, _cached_scaler, _cached_feature_cols, _load_attempted

    if _load_attempted:
        return _cached_model, _cached_scaler, _cached_feature_cols

    _load_attempted = True

    if not _ARTIFACTS_DIR.is_dir():
        return None, None, None

    model_candidates = sorted(
        p for p in _ARTIFACTS_DIR.glob("cover_logit_basic_*.joblib")
        if "scaler" not in p.name
    )
    scaler_candidates = sorted(_ARTIFACTS_DIR.glob("cover_logit_basic_*_scaler.joblib"))
    features_candidates = sorted(_ARTIFACTS_DIR.glob("cover_logit_basic_*_features.json"))

    model_path = model_candidates[-1] if model_candidates else None
    scaler_path = scaler_candidates[-1] if scaler_candidates else None
    features_path = features_candidates[-1] if features_candidates else None

    if not model_path or not scaler_path or not features_path:
        print("  Cover model artifacts not found — cover scoring disabled.")
        return None, None, None

    try:
        _cached_model = joblib.load(model_path)
        _cached_scaler = joblib.load(scaler_path)
        with open(features_path, encoding="utf-8") as f:
            _cached_feature_cols = json.load(f)
        print(f"  Loaded cover model: {model_path.name}")
        print(f"  Loaded cover scaler: {scaler_path.name}")
        print(f"  Cover feature columns: {len(_cached_feature_cols)}")
    except Exception as e:
        print(f"  Warning: Could not load cover model artifacts: {e}")
        _cached_model = None
        _cached_scaler = None
        _cached_feature_cols = None

    return _cached_model, _cached_scaler, _cached_feature_cols


def score_with_cover_model(feature_row: dict[str, Any]) -> dict[str, Any] | None:
    """
    Score a single game's feature row using the trained cover model.

    Returns a ``cover_model_score`` dict, or None if the model is not available.
    If too many features are missing, the score is flagged as not usable.
    """
    model, scaler, feature_cols = load_cover_model_and_scaler()
    if model is None or feature_cols is None:
        return None

    values = []
    null_count = 0
    for col in feature_cols:
        v = feature_row.get(col)
        if v is None:
            values.append(0.0)
            null_count += 1
        else:
            values.append(float(v))

    null_pct = null_count / len(feature_cols) if feature_cols else 1.0
    if null_pct > 0.5:
        return {
            "p_home_cover": None,
            "cover_confidence": None,
            "cover_null_feature_pct": round(null_pct, 3),
            "cover_model_available": True,
            "cover_score_usable": False,
        }

    X = np.array(values, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)
    p_home_cover = float(model.predict_proba(X_scaled)[0, 1])

    return {
        "p_home_cover": round(p_home_cover, 4),
        "cover_confidence": round(abs(p_home_cover - 0.5) * 200, 1),
        "cover_null_feature_pct": round(null_pct, 3),
        "cover_model_available": True,
        "cover_score_usable": True,
    }
