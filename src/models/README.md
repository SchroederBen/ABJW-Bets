# `src/models/` — Model Training Pipelines

This directory holds every training pipeline in the project. Each script
produces timestamped artifacts in `src/models/artifacts/` that the live AI
pipeline (`AI/main.py`) loads at runtime.

There are **three distinct models** and **one shared helper** here, each
predicting a different thing:

| File | Model family | Target | Role in AI payload |
|---|---|---|---|
| `train_logistic_model.py` | Logistic regression (L2 / L1 / ElasticNet) | `P(home team wins)` | `l1_model_score.l1_win_probability` (scored by ENet); `l1_model_features` allow-list |
| `train_logistic_cover_model.py` | Logistic regression (plain L2) | `P(home team covers the spread)` | `cover_model_score.p_home_cover` |
| `train_stat_builder_weights.py` | Structured stat-blend + logit head | `P(home team covers the spread)` | Top-level `p_home_cover`, `projected_home_margin`, `fair_home_spread`, `estimated_edge_points`, `edge_side` |
| `l1_feature_selection.py` | Shared helper (no standalone training) | — | Writes the live AI allow-list |

---

## Design principles (shared by all three)

Before diving into each file, a few conventions are identical across the
pipelines. Understanding them once saves explaining them three times.

### Pregame features from `build_pregame_features`

All three trainers start from `src/data/build_pregame_features.py`, which
produces one row **per team per game** with rolling-window stats computed
strictly from prior games (leakage-free). The windows used throughout are:

```python
WINDOWS = (3, 5, 10)   # last 3, 5, and 10 games
```

For each window `w` and each base stat, the pipeline emits columns like
`points_MA_w`, `field_goal_pct_MA_w`, `assists_MA_w`, etc. The per-team rows
are later merged into one row per game: home-team stats keep raw names,
away-team stats get an `opp_` prefix, and for every parallel column a
`diff_X = home_X − away_X` differential is added.

### Temporal train/test split — no shuffling

Betting data is time-ordered. Random train/test shuffling leaks future
information into training. All pipelines use the same approach:

```python
TEST_FRACTION = 0.20
split_idx = int(len(game_df) * (1 - TEST_FRACTION))
train_df = game_df.iloc[:split_idx]   # earlier 80%
test_df  = game_df.iloc[split_idx:]   # later 20%
```

Hyperparameter selection inside training uses `TimeSeriesSplit(n_splits=5)`
for the same reason: every CV fold trains on an earlier window and
validates on a later one.

### Standardization fit on train only

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # reuse train stats only
```

Fitting the scaler on train-only avoids leaking the test-set mean/std into
the model. The same `scaler.joblib` is persisted and reused for live
inference so new games are aligned to the exact same distribution.

### `--recent` flag

Every training script accepts `--recent`, which restricts training to games
on or after:

```python
RECENT_CUTOFF = "2015-01-01"
```

Intended for fast iteration while developing. Final models should still be
checked against the full history.

### Artifact directory

```
src/models/artifacts/    (gitignored; created on first run)
```

Each trainer writes versioned (timestamped) files here so you can see a
history of runs and roll back if a training run is worse than a previous
one. The AI always loads the **latest** matching file via `sorted(...)[-1]`
globs.

---

## `train_logistic_model.py` — Win Model (L2 / L1 / ElasticNet)

**Predicts**: `P(home team wins outright)`.
**Target column**: `team_win` (1 if the home team won the game, else 0).
**Runs**: `python -m src.models.train_logistic_model [l2|l1|all] [--recent] [--export-live]`

### What it trains

The script has three phases that share one data load (`_prepare_data`):

1. **Phase 1 — L2 baseline (Ridge)**
   `LogisticRegressionCV(penalty="l2", solver="lbfgs", cv=TimeSeriesSplit(5))`.
   Keeps every feature, but shrinks correlated coefficients together. This
   is the **"best we can do with every feature"** reference model.
2. **Phase 2a — L1 selection (Lasso)**
   `LogisticRegressionCV(penalty="l1", solver="liblinear", Cs=L1_CS_GRID)`.
   Drives weak coefficients to **exactly zero**, producing a sparse model.
   The surviving feature names become the AI's live allow-list.
3. **Phase 2b — Elastic Net**
   `LogisticRegressionCV(penalty="elasticnet", solver="saga",
   l1_ratios=[0.25, 0.5, 0.75])`. Combines L1 sparsity with L2 stability
   on correlated features. **Currently used by the AI for live win-probability
   scoring.**

### Why a custom `Cs` grid

`LogisticRegressionCV`'s default `Cs=10` explores `logspace(-4, 4)`,
which at the high end produces essentially no regularization — L1 never
zeros anything out. The project overrides this:

```python
L1_CS_GRID   = np.logspace(-3, 1.5, 16)   # ~0.001 .. ~31.6
ENET_CS_GRID = np.logspace(-2, 1.5, 12)   # ~0.01  .. ~31.6
```

Smaller C = stronger regularization = more sparsity. The grid is capped
above ~31.6 so CV actually explores meaningful shrinkage.

### Calibration assessment

After fitting, the script computes a 10-bin reliability curve on the test
set and records `max_calibration_error`. If that exceeds 0.05, the model
is wrapped in `CalibratedClassifierCV(method="sigmoid")` (Platt scaling)
and the calibrated version is saved. This matters because betting math
relies on probabilities being close to truth, not just directionally
correct.

### Outputs comparison table

At the end of Phase 2 the script prints a side-by-side table the slide
deck can screenshot verbatim:

```
MODEL COMPARISON (vs L2 Baseline)
  Metric                L2 Baseline   L1 (Lasso)   Elastic Net
  ------------------------------------------------------------
  Log Loss              ...           ...          ...
  Brier Score           ...           ...          ...
  ROC-AUC               ...           ...          ...
  Accuracy              ...           ...          ...
  Non-zero Features     ALL           ~55          ~51
```

### Artifacts produced

Timestamped with `{ts} = YYYYMMDD_HHMMSS`:

| File | What it is |
|---|---|
| `l2_baseline_{ts}.joblib` | L2 model (possibly wrapped in CalibratedClassifierCV) |
| `l2_baseline_scaler_{ts}.joblib` | StandardScaler from L2 training |
| `l2_baseline_metrics_{ts}.json` | Log loss, Brier, ROC-AUC, accuracy, best C, calibration error, feature counts, train/test date ranges |
| `l2_baseline_features_{ts}.json` | Column order used for training |
| `l2_baseline_coefficients_{ts}.csv` | Each feature's coefficient, sorted by \|value\| |
| `l2_baseline_calibration_{ts}.png` | Reliability plot |
| `l1_selection_{ts}.joblib` | L1 model (possibly calibrated) — **not currently loaded by AI** |
| `l1_selection_scaler_{ts}.joblib` | StandardScaler (shared with ENet) |
| `l1_selection_metrics_{ts}.json` | L1 + Elastic Net metrics |
| `l1_selection_features_all_{ts}.json` | **Full column order — used by AI for live ENet alignment** |
| `l1_selection_features_reduced_{ts}.json` | Surviving features — source of the live AI allow-list |
| `l1_selection_coefficients_{ts}.csv` | Signed L1 coefficients |
| `l1_selection_calibration_{ts}.png` | L1 reliability plot |
| `elastic_net_{ts}.joblib` | **Elastic Net model — what the AI actually loads for win-prob scoring** |

### `--export-live`

When this flag is passed, after training the L1 phase the script writes
the surviving feature list to:

```
AI/config/l1_allowlist/features_reduced.json
```

This is the **single stable file** the AI loads at runtime (see
`AI/Helpers/l1_live_features.py::get_l1_allowlist_from_env`). Every export
overwrites the previous one.

### CLI flags

| Flag | Effect |
|---|---|
| `l2` | Run only Phase 1 |
| `l1` | Run only Phase 2 (L1 + ENet) — **this is the one you normally use** |
| `all` | Run Phase 1 then Phase 2 sharing one data load |
| `--recent` | Filter to games ≥ `RECENT_CUTOFF` (2015-01-01) |
| `--export-live` | Overwrite `AI/config/l1_allowlist/features_reduced.json` |

---

## `train_logistic_cover_model.py` — Cover Model

**Predicts**: `P(home team covers the opening spread)`.
**Target**: `home_cover = 1 if (home_margin + opening_spread) > 0 else 0`, with pushes dropped.
**Runs**: `python -m src.models.train_logistic_cover_model [--recent]`

### Why a separate model

Covering the spread is a fundamentally different question than winning
outright — the sportsbook prices the line so outcomes are near 50/50 on
purpose. A model trained on `team_win` gives you "who is better" (a
question already priced in), but not "who beats the line." This script
retargets the same feature set at `home_cover` directly.

### What it does

1. Loads pregame features via `build_pregame_features(windows=(3, 5, 10))`.
2. Merges home/away rows into one row per game (home raw names, away
   prefixed `opp_`).
3. Computes the ATS label and drops pushes.
4. Adds matchup diffs (`diff_X = home_X − opp_X`) and `abs_opening_spread`.
5. Chronological 80/20 split, train-only StandardScaler.
6. Fits plain L2 logistic regression:

   ```python
   LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000)
   ```

   No cross-validation grid here — the cover problem is noisy enough
   that a simple fixed C keeps the script easy to read and tweak.
7. Evaluates on the held-out test set.
8. Computes an `edge_vs_neg110` summary — compares the model's probability
   to the flat implied probability for a -110 line (~52.4%) and counts
   how often the edge exceeds 3%. This is a sanity-check distribution, not
   a betting strategy.

### Why plain L2 instead of L1/ENet

The cover signal is so weak that further feature selection tends to just
add variance without improving log loss. Keeping all features with mild L2
shrinkage produces a smoother probability distribution that the LLM can
use as a corroboration signal.

### Artifacts produced

Prefix `cover_logit_basic_{ts}`:

| File | What it is |
|---|---|
| `cover_logit_basic_{ts}.joblib` | Trained logistic regression model |
| `cover_logit_basic_{ts}_scaler.joblib` | StandardScaler |
| `cover_logit_basic_{ts}_metrics.json` | Log loss, Brier, ROC-AUC, edge summary, date ranges |
| `cover_logit_basic_{ts}_features.json` | Training column order — required for live alignment |
| `cover_logit_basic_{ts}_coefficients.csv` | Signed coefficients sorted by \|value\| |

These are read by `AI/Helpers/cover_model_live.py` at runtime to produce
each game's `cover_model_score` block.


---

## `train_stat_builder_weights.py` — Stat-Blend + Logit Head

**Predicts**: `P(home team covers the opening spread)`, but via a structured
low-parameter model rather than a generic logistic regression.
**Runs**: `python -m src.models.train_stat_builder_weights [--recent]`

### What's different about this one

The other two trainers are black-box logistic regressions on ~100 features.
This one deliberately mirrors the **exact formula** used by
`AI/Helpers/stat_builder.py::calculate_estimated_edge_v2` and learns its
hand-tuned constants from data:

```
home_strength   = w_season * home_season_pd
                + w_site   * home_site_pd
                + w_recent * home_recent_pd

away_strength   = (same weights applied to away team)

projected_margin = margin_mult * ((home_strength - away_strength)
                                   + home_court_advantage)

logit            = logit_bias + logit_scale * (projected_margin + opening_spread)
P(home covers)   = sigmoid(logit)
```

**Only 7 free parameters** (`w_season`, `w_site`, `w_recent`,
`home_court_advantage`, `margin_mult`, `logit_scale`, `logit_bias`). Because
the structure is fixed, the output is interpretable and directly substitutes
into the existing stat_builder plumbing.

### Why a walk-forward dataset

Feature computation here is **state-based**, not rolling-window-based. For
each game in chronological order:

1. Compute features (`season_pd_diff`, `site_pd_diff`, `recent_pd_diff`)
   from the prior state of both teams.
2. Record the pregame row + actual outcome.
3. **Then** update each team's state with the observed point differential.

This guarantees zero future leakage without needing rolling window merges.
A self-test (`--self-test`) asserts no row's features depend on the current
game's outcome.

### Optimization

The 7 parameters are packed into an unconstrained 7-vector `eta` and
mapped onto valid ranges via:

- `eta[0:3]` → softmax → simplex weights (`w_season`, `w_site`, `w_recent`) sum to 1
- `eta[3]` → `home_court_advantage` (direct)
- `eta[4]` → logistic map to `(1.0, 1.35]` → `margin_mult`
- `eta[5]` → `0.01 + exp(eta[5]) > 0` → `logit_scale`
- `eta[6]` → `logit_bias` (direct)

L-BFGS-B minimizes mean negative log-likelihood on the training set with
bounds `[-10, 10]` on each `eta`. On convergence, `unpack_structured_params`
converts `eta_opt` back into the seven named parameters.

### Post-fit calibration and policy

After the 7 structural parameters are fit:

- `_fit_k_edge_points` derives a scaling constant so that
  `estimated_edge_points ≈ k * |p − 0.5|` has the right magnitude to
  display alongside spread numbers.
- `_choose_tau_for_pass_band` selects the decision threshold `tau` that
  maximizes ROI on the training set's -110 proxy backtest. Below `tau`
  the model returns PASS; above it it picks a side.

Both are saved in the `mapping` block of the artifact so the live pipeline
reproduces them exactly.

### Artifact produced

One JSON per run:

```
src/models/artifacts/stat_builder_weights_struct_{ts}.json
```

Structure:

```json
{
  "model_version": "stat_builder_weights_struct_20260421_140222",
  "stat_builder": { "w_season": ..., "w_site": ..., "w_recent": ...,
                    "home_court_advantage": ..., "margin_mult": ... },
  "logit_head": { "scale": ..., "bias": ... },
  "mapping":    { "tau": ..., "k_edge_points": ... },
  "metrics":    { "log_loss": ..., "brier_score": ..., "roc_auc": ..., ... },
  "formula_notes": "Shared blend: home_strength = ... ; p_cover = sigmoid(...)"
}
```

`AI/Helpers/stat_builder.py::_load_trained_weights` reads this directly and
plugs each named parameter into `calculate_estimated_edge_learned`. This
is the **primary** cover probability the AI sees.

### Diagnostics it prints

Each run prints:

- The seven learned parameters
- Train / test log loss, Brier, ROC-AUC
- A backtest comparing the learned strategy vs the hand-tuned v2 proxy at
  a -110 price, including `win_rate` and `roi_per_bet`

If you see `w_season` collapse near 0 alongside `logit_scale` near 0.01,
the optimizer likely found a degenerate solution (probabilities pinned near
0.5). Rerun with more data or constrained bounds; do not ship that artifact.

---

## `l1_feature_selection.py` — Shared L1 Helper

Not a standalone script — a reusable module importable from any training
pipeline. Two public entry points:

### `run_l1_feature_selection(...)` (lines 97+)

Takes already-prepared train/test matrices, fits
`LogisticRegressionCV(penalty="l1", solver="liblinear", Cs=...)`, separates
surviving from dropped features, evaluates on the held-out test set, and
optionally persists artifacts with configurable prefixes.

Used by:

- `train_logistic_model.py::train_l1_selection` — prefix `l1_selection`
- Any future trainer that wants L1-based feature pruning (potentially the
  cover model if you wanted to prune its ~140-column input)

### `write_ai_live_allowlist_json(surviving_features)` (lines 59–72)

Writes sorted feature names to `AI/config/l1_allowlist/features_reduced.json`
(relative to the repo root), overwriting any previous content. This is the
function behind the `--export-live` flag.

### `load_surviving_feature_names(reduced_json_path)` (lines 87–94)

Inverse operation: load a `*_features_reduced_*.json` file. Used by any
downstream tool that wants to read a specific L1 run's allow-list (not
just the latest live export).

### `default_ai_live_allowlist_path()` (lines 54–56)

Centralizes the live path as:

```
{repo_root}/AI/config/l1_allowlist/features_reduced.json
```

Changing this one constant moves where the live export lands for every
pipeline.

---

## `__init__.py`

Empty. Exists so Python treats `src/models/` as a package, which lets
`python -m src.models.train_logistic_model ...` work as a module entry
point.

---

## `artifacts/` directory

Gitignored. Created on first training run. Every model described above
writes there with timestamped filenames:

```
src/models/artifacts/
├── l2_baseline_20260421_140034.joblib
├── l2_baseline_scaler_20260421_140034.joblib
├── l2_baseline_metrics_20260421_140034.json
├── l2_baseline_features_20260421_140034.json
├── l2_baseline_coefficients_20260421_140034.csv
├── l2_baseline_calibration_20260421_140034.png
├── l1_selection_20260421_140034.joblib
├── l1_selection_scaler_20260421_140034.joblib
├── l1_selection_metrics_20260421_140034.json
├── l1_selection_features_all_20260421_140034.json
├── l1_selection_features_reduced_20260421_140034.json
├── l1_selection_coefficients_20260421_140034.csv
├── l1_selection_calibration_20260421_140034.png
├── elastic_net_20260421_140034.joblib
├── cover_logit_basic_20260421_140146.joblib
├── cover_logit_basic_20260421_140146_scaler.joblib
├── cover_logit_basic_20260421_140146_metrics.json
├── cover_logit_basic_20260421_140146_features.json
├── cover_logit_basic_20260421_140146_coefficients.csv
└── stat_builder_weights_struct_20260421_140222.json
```

The AI always loads the chronologically latest file for each model via
`sorted(glob(...))[-1]`, so older runs can stay around for comparison or
rollback without interfering.

---

## What feeds the live AI

Summary of the data flow from this directory into `AI/main.py`:

```
train_logistic_model.py (--export-live)
    ├── elastic_net_*.joblib          ──► win_probability (l1_win_probability key)
    ├── l1_selection_scaler_*.joblib  ──► shared scaler for ENet
    ├── l1_selection_features_all_*.json ─► ENet column order
    └── AI/config/l1_allowlist/features_reduced.json ─► l1_model_features allow-list

train_logistic_cover_model.py
    ├── cover_logit_basic_*.joblib    ──► cover_model_score.p_home_cover
    ├── cover_logit_basic_*_scaler.joblib
    └── cover_logit_basic_*_features.json

train_stat_builder_weights.py
    └── stat_builder_weights_struct_*.json ──► top-level p_home_cover,
                                                projected_home_margin,
                                                estimated_edge_points,
                                                edge_side
```

If any of these files are missing when `AI/main.py` runs, its corresponding
block will fall back to a default (for the cover/win models, the score is
marked `not available`; for stat_builder_weights, the AI falls back to the
hand-tuned v2 heuristic).

---

## Running these pipelines

See [`../../docs/howToRun.md`](../../docs/howToRun.md) for the end-to-end
command list. Quick reminder:

```bash
# From repo root
python -m src.models.train_logistic_model l1 --recent --export-live
python -m src.models.train_logistic_cover_model --recent
python -m src.models.train_stat_builder_weights --recent
```

Then `cd AI && python main.py` to produce picks using the freshly trained
models.
