## L1 regression -> AI 

### What you get from L1

- **Output**: a JSON file containing a **list of feature names** (strings) that survived L1 (non-zero coefficients after balancing regression).

- **Purpose**: that list becomes an **allow-list** for what we compute and send to the LLM as `l1_model_features`.

The AI does **not** read the L1 model weights. It only uses the **surviving feature names** as a schema for what to include.

---

## Step 1 — Run L1 and export the live AI allow-list

From the repo root:

```bash
python -m src.models.train_logistic_model l1 --recent --export-live
```

- `--recent` limits training to recent seasons (based on that script’s cutoff):
- I would use --recent because I think it gets slightly more accurate since the old data is not as good



What `--export-live` does:

- Overwrites this file every run (so it’s always “latest”):
  - `AI/config/l1_allowlist/features_reduced.json`

That file is the allow-list consumed by the AI pipeline.

---

## Step 2 — Run the AI as normal

From the `AI` directory:

```bash
python main.py
```

## How it works inside the AI code

### A) `l1_live_features.py` (loads the allow-list + computes values)

File: `AI/Helpers/l1_live_features.py`

1. **Loads the allow-list**
   - If env var `AI_L1_FEATURES_JSON` is set, it loads that file.
   - Otherwise it tries the default stable file:
     - `AI/config/l1_allowlist/features_reduced.json`

2. **Computes per-game feature values**
   - Builds a “pregame row” for the upcoming matchup (home vs away perspective).
   - Produces fields like:
     - `points_MA_3`, `team_score_MA_10`, `opp_opponent_score_MA_5`, `team_win_MA_5`
     - `days_rest`, `opp_days_rest`
     - `opening_spread`, `abs_opening_spread`
     - `diff_*` (home − away) for MA/rest fields

Important limitation (current DB query):

- The AI DB history loader (`fetch_historical_games`) only has **scores + dates + teams + spreads**.
- So features requiring box-score stats (e.g. `assists_MA_5`, `field_goal_pct_MA_10`) will appear as **`null`** in `l1_model_features` until the live history query includes those fields.
- I just had those already used in my regression models so I figured allow them here too 

### B) `stat_builder.py` Change (adds `l1_model_features` to each game)

- Added imports from `Helpers.l1_live_features`:
  - `get_l1_allowlist_from_env`
  - `build_l1_model_features_subset`
- In `build_matchup_payload_from_api_games`, after creating `game_entry`, we now:
  - Load the allow-list (if present)
  - Add `game_entry["l1_model_features"] = ...` (a dict of `{feature_name: value_or_null}`)

That’s it — nothing else about the existing payload fields was removed. just adds an extra per-game block.

### C) `ai_analysis.py` changes

File: `AI/Helpers/ai_analysis.py`


- Updated the `system_prompt` with a new rule:
  - “If `l1_model_features` is present, use those numeric pregame fields as the model‑aligned stat snapshot…”

No other prompt formatting changed — the user prompt still includes the full matchup payload via `json.dumps(matchups, indent=2)`, so the new `l1_model_features` block is automatically included when present.

### High level
- pretty much just adds the stats that survived L1 regression and then gives them to the LLM to consider 
