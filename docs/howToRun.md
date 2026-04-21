# How to Run

Quick steps to train the models and run the AI.

---

## 1. Train the models

Run all three from the **repo root** (`ABJW-Bets/`). Order matters only for the first one (because it writes the live allow-list the AI reads).

### Win model (L1 + Elastic Net)

```bash
python -m src.models.train_logistic_model l1 --recent --export-live
```

What this does:

- Trains the L1 and Elastic Net logistic regression models
- Saves artifacts to `src/models/artifacts/`
- Overwrites `AI/config/l1_allowlist/features_reduced.json` — the allow-list the AI loads at runtime

Notes:

- `--recent` restricts training to games from 2015 onward (faster, more modern).
- `--export-live` is required if you want the AI to use today's L1 features.
- The AI scores with the **Elastic Net** model (`elastic_net_*.joblib`), but the allow-list still comes from the L1 phase.

### Cover model

```bash
python -m src.models.train_logistic_cover_model --recent
```

What this does:

- Trains a logistic regression model predicting `P(home covers the opening spread)`
- Saves `cover_logit_basic_*.joblib` + scaler + features JSON in `src/models/artifacts/`

### Stat-builder weights (learned edge model)

```bash
python -m src.models.train_stat_builder_weights --recent
```

What this does:

- Learns the stat_builder blend weights (`w_season`, `w_site`, `w_recent`, home court advantage, margin multiplier, logit head)
- Saves `stat_builder_weights_struct_*.json` in `src/models/artifacts/`
- Provides the **primary** `p_home_cover` signal the AI prompt relies on

---

## 2. Run the AI

From the **`AI/`** directory:

```bash
cd AI
python main.py
```

This will:

1. Fetch today's games and historical data from the DB
2. Auto-load the latest model artifacts
3. Build the matchup payload (stats, L1 allow-list features, win probability, cover probability, edge model outputs)
4. Send the payload to the LLM and print its recommendations

---
## Typical workflow


```bash
# From repo root — only when you want to refresh models
python -m src.models.train_logistic_model l1 --recent --export-live
python -m src.models.train_logistic_cover_model --recent
python -m src.models.train_stat_builder_weights --recent

# From AI/ — every time you want picks
cd AI
python main.py
```
