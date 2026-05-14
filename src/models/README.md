## Models Overview

This directory contains the end-to-end logistic regression pipeline used to predict NBA game outcomes from pre-game rolling-window features.

- **Data source**: pre-game rolling statistics built by `build_pregame_features` from `src/data/build_pregame_features.py`
- **Target**: `team_win` from the home team’s perspective (1 = home win, 0 = loss)
- **Design goals**: well-calibrated probabilities (log loss / Brier score) and interpretability via model coefficients

**ATS / spread (LLM-aligned)**: see **`train_logistic_cover_model.py`** — same rolling features, but the target is **`home_cover`** vs **`opening_spread`** (pushes dropped). Artifacts use prefixes `l2_cover_*` and `l1_cover_*`. Run: `python -m src.models.train_logistic_cover_model [l2|l1|all] [--recent]`.

**Reusable L1 selection**: **`l1_feature_selection.py`** exposes `run_l1_feature_selection()` and `load_surviving_feature_names()` so any pipeline (cover, win, future LLM builders) can share the same Lasso CV + surviving-feature JSON. The cover trainer uses it with `artifact_prefix="l1_cover_selection"` and `feature_list_prefix="l1_cover"` for filenames.

### Files

- **`train_logistic_model.py`**
  - Phase 1: **L2 baseline (Ridge)**
    - Loads pregame features and builds a **game-level** dataset by merging home/away rows into a single row per game (home stats + `opp_`-prefixed away stats).
    - Performs a **temporal train/test split** (earliest games for training, most recent for test) to avoid leaking future information.
    - Scales features with `StandardScaler`.
    - Trains `LogisticRegressionCV` with:
      - `penalty="l2"`, `solver="lbfgs"`, `TimeSeriesSplit(n_splits=5)`
      - scoring on **negative log loss**.
    - Evaluates on the held-out test set with:
      - **Log loss** (primary)
      - **Brier score**
      - **ROC-AUC**
      - **Accuracy**
    - Checks calibration using `calibration_curve` and, if needed, fits a calibrated model with `CalibratedClassifierCV(method="sigmoid")`.
    - Saves artifacts under `src/models/artifacts/` with `l2_baseline_*` prefixes:
      - Trained model (`.joblib`)
      - Feature scaler (`.joblib`)
      - Metrics (`.json`)
      - Feature list in training order (`.json`)
      - Coefficients (`.csv`)
      - Calibration plot (`.png`)

  - Phase 2: **L1 feature selection (Lasso) + Elastic Net**
    - Reuses the same prepared dataset and scaling.
    - **L1 (Lasso)**:
      - Trains `LogisticRegressionCV` with `penalty="l1"` and `solver="liblinear"`.
      - Identifies features whose coefficients are driven to exactly zero and treats them as **droppable**.
      - Reports:
        - Number of **surviving** vs **dropped** features.
        - Full list of dropped features (zero coefficients).
        - Test-set log loss, Brier score, ROC-AUC, and accuracy.
    - **Elastic Net**:
      - Trains `LogisticRegressionCV` with `penalty="elasticnet"`, `solver="saga"`, and several `l1_ratio` values.
      - Reports the best C, best `l1_ratio`, and number of non-zero coefficients.
    - Prints a **side-by-side comparison table** of L2 vs L1 vs Elastic Net on:
      - Log loss
      - Brier score
      - ROC-AUC
      - Accuracy
      - Number of non-zero features
    - Performs a separate calibration check for the L1 model (with optional Platt scaling).
    - Saves artifacts under `src/models/artifacts/` with `l1_selection_*` and `elastic_net_*` prefixes:
      - L1 model (`.joblib`)
      - Elastic Net model (`.joblib`)
      - Scaler (`.joblib`)
      - Metrics JSON (includes both L1 and Elastic Net metrics)
      - **Full feature list** (`l1_selection_features_all_*.json`)
      - **Reduced feature list** of surviving features (`l1_selection_features_reduced_*.json`)
      - Coefficients (`.csv`)
      - Calibration plot (`.png`)

  - **Shared data prep**
    - `_build_game_dataset`:
      - Takes team-game-level rows and creates one row per game, joining home and away.
      - Home features stay as-is; away features are prefixed with `opp_`.
    - `_temporal_split`:
      - Chronological split into train/test, with the **most recent** games reserved for testing.
    - `_prepare_data(recent_only: bool = False)`:
      - Loads rolling-window features via `build_pregame_features(windows=(3, 5, 10))`.
      - Builds the game-level dataset.
      - If `recent_only=True`, filters to games on or after a configured cutoff date (see below).
      - Applies temporal split and standard scaling.

### Recent-only mode (for faster experimentation)

To avoid training on the entire historical database while you iterate, the training script supports a **recent-only** mode controlled by:

- `RECENT_CUTOFF` at the top of `train_logistic_model.py` (currently set to `"2015-01-01"`).
- A `--recent` command-line flag.

When `--recent` is enabled:

- `_prepare_data(recent_only=True)` filters the game-level dataset to rows with `date >= RECENT_CUTOFF`.
- The console prints how many games were kept vs original (e.g. `--recent: kept 15400/72865 games`).
- Train/test splits and metrics are computed only on this filtered subset.

This mode is intended for **development and hyperparameter experimentation** to save time and compute. For final models, rerun **without** `--recent` so the coefficients and metrics reflect the full history.

### How to run

From the project root (`ABJW-Bets`), you can run:

- **Phase 1 only (L2 baseline)**:

  ```bash
  python -m src.models.train_logistic_model l2          # full history
  python -m src.models.train_logistic_model l2 --recent # recent seasons only
  ```

- **Phase 2 only (L1 + Elastic Net)**:

  ```bash
  python -m src.models.train_logistic_model l1          # full history
  python -m src.models.train_logistic_model l1 --recent # recent seasons only
  ```

- **Full pipeline (Phase 1 then Phase 2, sharing one data load)**:

  ```bash
  python -m src.models.train_logistic_model all          # full history
  python -m src.models.train_logistic_model all --recent # recent seasons only
  ```

### Current results snapshot (recent-only run)

On games since 2015 (using `--recent`), the latest run produced roughly:

- **L2 baseline (Ridge)**:
  - Log loss ≈ 0.6497
  - Brier score ≈ 0.2288
  - ROC-AUC ≈ 0.66
  - Accuracy ≈ 0.622
  - Non-zero features: 92 (all features)

- **L1 selection (Lasso, liblinear)**:
  - Log loss ≈ 0.6414
  - Brier score ≈ 0.2255
  - ROC-AUC ≈ 0.679
  - Accuracy ≈ 0.621
  - Non-zero features: 51 (41 features dropped)

- **Elastic Net (saga)**:
  - Log loss ≈ 0.6421
  - Brier score ≈ 0.2258
  - ROC-AUC ≈ 0.678
  - Accuracy ≈ 0.622
  - Non-zero features: 53

So far, **L1 feature selection matches or beats the L2 baseline on probability-quality metrics while using nearly half as many features**, making the model more interpretable and efficient for downstream betting EV analysis.

