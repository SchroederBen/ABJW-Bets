"""
Offline training pipeline to learn weights in the same shape as AI/Helpers/stat_builder.py.

Structured model (shared blend for home and away, like calculate_estimated_edge_v2):

    home_strength = w_season * home_season_pd + w_site * home_site_pd + w_recent * home_recent_pd
    away_strength = same weights on away team inputs
    inner_diff     = home_strength - away_strength   # equals w·(home_vec - away_vec) on walk-forward diffs
    projected_margin = margin_mult * (inner_diff + home_court_advantage)

Then a simple logistic head maps projected margin vs the opening line to P(home covers):

    logit = bias + scale * (projected_margin + opening_spread)
    p_cover = sigmoid(logit)

Weights (w_season, w_site, w_recent) are a simplex (softmax of 3 free parameters).

Features are computed leakage-free with a walk-forward state update.

Artifacts: versioned JSON under src/models/artifacts/ (STAT_BUILDER_MODEL_VERSION when wired in).
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.data.load_raw_team_game_history import load_raw_team_game_history


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
TEST_FRACTION = 0.20
RECENT_N = 5
RECENT_CUTOFF = "2015-01-01"


@dataclass
class TeamState:
    games: int = 0
    point_diff_total: float = 0.0

    home_games: int = 0
    home_point_diff_total: float = 0.0

    away_games: int = 0
    away_point_diff_total: float = 0.0

    recent_point_diffs: deque[float] = field(default_factory=lambda: deque(maxlen=RECENT_N))


def _avg(x_sum: float, n: int) -> float:
    return float(x_sum) / float(n) if n else 0.0


def _recent_avg(diffs: deque[float]) -> float:
    return float(np.mean(diffs)) if diffs else 0.0


def _build_game_table(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raw team-game history (2 rows per game) into one row per game
    with the fields needed for ATS label + walk-forward feature construction.
    """
    cols = [
        "game_id",
        "date",
        "home_team_id",
        "away_team_id",
        "opening_spread",
        "home_score",
        "away_score",
    ]
    missing = [c for c in cols if c not in raw.columns]
    if missing:
        raise KeyError(f"Missing required columns in raw history: {missing}")

    game_df = raw[cols].drop_duplicates(subset=["game_id"]).copy()
    game_df["date"] = pd.to_datetime(game_df["date"])
    game_df = game_df.sort_values(["date", "game_id"]).reset_index(drop=True)
    return game_df


def add_ats_home_cover_label(game_df: pd.DataFrame, *, drop_pushes: bool = True) -> pd.DataFrame:
    """
    Create ATS label vs opening spread:
        home_cover = 1 if (home_margin + opening_spread) > 0 else 0

    Push (== 0) rows are optionally dropped (recommended).
    """
    df = game_df.copy()
    df = df[df["opening_spread"].notna()].copy()
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()

    df["home_margin"] = df["home_score"].astype(float) - df["away_score"].astype(float)
    df["margin_plus_spread"] = df["home_margin"] + df["opening_spread"].astype(float)

    if drop_pushes:
        df = df[df["margin_plus_spread"] != 0].copy()

    df["home_cover"] = (df["margin_plus_spread"] > 0).astype(int)
    return df.reset_index(drop=True)


def build_walkforward_dataset(game_df_labeled: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per game with leakage-free features computed from prior games.

    Features are diffs that correspond to stat_builder strength inputs:
        season_pd_diff, site_pd_diff, recent_pd_diff, plus spread controls.
    """
    states: dict[int, TeamState] = {}
    rows: list[dict[str, Any]] = []

    for g in game_df_labeled.itertuples(index=False):
        home = int(g.home_team_id)
        away = int(g.away_team_id)

        hs = float(g.home_score)
        aws = float(g.away_score)
        spread = float(g.opening_spread)

        home_state = states.get(home, TeamState())
        away_state = states.get(away, TeamState())

        # --- pre-game features (from prior state only) ---
        home_season_pd = _avg(home_state.point_diff_total, home_state.games)
        away_season_pd = _avg(away_state.point_diff_total, away_state.games)

        home_site_pd = _avg(home_state.home_point_diff_total, home_state.home_games)
        away_site_pd = _avg(away_state.away_point_diff_total, away_state.away_games)

        home_recent_pd = _recent_avg(home_state.recent_point_diffs)
        away_recent_pd = _recent_avg(away_state.recent_point_diffs)

        rows.append(
            {
                "game_id": int(g.game_id),
                "date": pd.to_datetime(g.date),
                "home_team_id": home,
                "away_team_id": away,
                "opening_spread": spread,
                "abs_spread": abs(spread),
                "season_pd_diff": home_season_pd - away_season_pd,
                "site_pd_diff": home_site_pd - away_site_pd,
                "recent_pd_diff": home_recent_pd - away_recent_pd,
                "home_cover": int(g.home_cover),
                "home_margin": float(g.home_margin),
                "margin_plus_spread": float(g.margin_plus_spread),
            }
        )

        # --- update team states with the current game's outcome ---
        home_diff = hs - aws
        away_diff = aws - hs

        home_state.games += 1
        home_state.point_diff_total += home_diff
        home_state.home_games += 1
        home_state.home_point_diff_total += home_diff
        home_state.recent_point_diffs.append(home_diff)

        away_state.games += 1
        away_state.point_diff_total += away_diff
        away_state.away_games += 1
        away_state.away_point_diff_total += away_diff
        away_state.recent_point_diffs.append(away_diff)

        states[home] = home_state
        states[away] = away_state

    df = pd.DataFrame(rows).sort_values(["date", "game_id"]).reset_index(drop=True)
    return df


def _temporal_split(df: pd.DataFrame, test_frac: float = TEST_FRACTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(float) - float(np.max(x))
    e = np.exp(x)
    return e / float(np.sum(e))


def unpack_structured_params(eta: np.ndarray) -> dict[str, float]:
    """
    Map unconstrained vector eta (length 7) to stat_builder-shaped parameters.

    eta[0:3] -> simplex weights via softmax (w_season, w_site, w_recent)
    eta[3]   -> home_court_advantage (direct)
    eta[4]   -> margin_mult in (1.0, 1.35] via logistic map (matches ~1.10 tuning)
    eta[5]   -> logit_scale = 0.01 + exp(eta[5]) > 0
    eta[6]   -> logit_bias
    """
    w = _softmax(eta[0:3])
    hca = float(eta[3])
    margin_mult = 1.0 + 0.35 / (1.0 + np.exp(-float(eta[4])))
    scale = 0.01 + float(np.exp(float(eta[5])))
    bias = float(eta[6])
    return {
        "w_season": float(w[0]),
        "w_site": float(w[1]),
        "w_recent": float(w[2]),
        "home_court_advantage": hca,
        "margin_mult": float(margin_mult),
        "logit_scale": float(scale),
        "logit_bias": float(bias),
    }


def predict_p_cover_structured(
    d_season: np.ndarray,
    d_site: np.ndarray,
    d_recent: np.ndarray,
    opening_spread: np.ndarray,
    params: dict[str, float],
) -> np.ndarray:
    w_s = params["w_season"]
    w_i = params["w_site"]
    w_r = params["w_recent"]
    inner = w_s * d_season + w_i * d_site + w_r * d_recent
    proj = params["margin_mult"] * (inner + params["home_court_advantage"])
    logit = params["logit_bias"] + params["logit_scale"] * (proj + opening_spread)
    logit = np.clip(np.asarray(logit, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logit))


def _structured_mean_nll(
    eta: np.ndarray,
    d_s: np.ndarray,
    d_i: np.ndarray,
    d_r: np.ndarray,
    spread: np.ndarray,
    y: np.ndarray,
) -> float:
    p = predict_p_cover_structured(d_s, d_i, d_r, spread, unpack_structured_params(eta))
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _choose_tau_for_pass_band(y_true: np.ndarray, p_home_cover: np.ndarray) -> dict[str, float]:
    """
    Choose PASS threshold tau for converting probabilities into {HOME, AWAY, PASS}.

    We grid-search tau to maximize a simple -110 ROI proxy on the evaluation set:
        - bet HOME when p >= 0.5 + tau
        - bet AWAY when p <= 0.5 - tau
        - PASS otherwise

    This is intentionally simple and deterministic. It does not use line movement
    or book prices; it approximates spread vig as -110.
    """
    # -110: win earns 100/110 ≈ 0.9091, loss costs 1.0
    win_payout = 100.0 / 110.0

    best = {"tau": 0.05, "roi_per_bet": -1.0, "n_bets": 0.0, "win_rate": 0.0}
    for tau in np.linspace(0.0, 0.20, 41):
        take_home = p_home_cover >= (0.5 + tau)
        take_away = p_home_cover <= (0.5 - tau)
        take = take_home | take_away
        n_bets = int(np.sum(take))
        if n_bets == 0:
            continue

        # if we bet away, it's correct when home_cover == 0
        correct = (take_home & (y_true == 1)) | (take_away & (y_true == 0))
        wins = int(np.sum(correct))
        losses = n_bets - wins
        profit = wins * win_payout - losses * 1.0
        roi_per_bet = profit / float(n_bets)
        win_rate = wins / float(n_bets)

        # Prefer higher ROI; break ties by more bets.
        if (roi_per_bet > best["roi_per_bet"]) or (
            abs(roi_per_bet - best["roi_per_bet"]) < 1e-12 and n_bets > best["n_bets"]
        ):
            best = {
                "tau": float(tau),
                "roi_per_bet": float(roi_per_bet),
                "n_bets": float(n_bets),
                "win_rate": float(win_rate),
            }
    return best


def _fit_k_edge_points(train_df: pd.DataFrame, p_home_cover_train: np.ndarray) -> float:
    """
    Fit a simple scaling factor k so that:
        edge_points ~= k * abs(p - 0.5)

    We fit k on training data (not test) using median ratios for robustness.
    """
    edge_prob = np.abs(p_home_cover_train - 0.5)
    edge_prob = np.clip(edge_prob, 1e-6, None)
    edge_points_target = np.abs(train_df["margin_plus_spread"].values.astype(float))
    return float(np.median(edge_points_target) / np.median(edge_prob))


def _stat_builder_v2_proxy_side(df: pd.DataFrame) -> np.ndarray:
    """
    Proxy of AI/Helpers/stat_builder.calculate_estimated_edge_v2 using our diffs.

    Returns:
        side: +1 for HOME_SPREAD, -1 for AWAY_SPREAD, 0 for PASS
    """
    # weights from stat_builder v2
    home_court_advantage = 1.5
    raw_margin = (
        0.60 * df["season_pd_diff"].values
        + 0.25 * df["site_pd_diff"].values
        + 0.15 * df["recent_pd_diff"].values
        + home_court_advantage
    )
    projected_home_margin = raw_margin * 1.10
    fair_home_spread = -projected_home_margin
    spread_diff = fair_home_spread - df["opening_spread"].values

    adjusted = spread_diff.copy()
    abs_spread = np.abs(df["opening_spread"].values)
    adjusted = np.where(abs_spread >= 10, adjusted * 0.80, adjusted)
    adjusted = np.where((abs_spread >= 7) & (abs_spread < 10), adjusted * 0.90, adjusted)

    threshold = np.where(abs_spread >= 10, 3.5, 2.5)

    side = np.zeros(len(df), dtype=int)
    side[np.abs(adjusted) <= threshold] = 0
    side[(np.abs(adjusted) > threshold) & (adjusted < 0)] = +1
    side[(np.abs(adjusted) > threshold) & (adjusted > 0)] = -1
    return side


def _evaluate_side_strategy(y_true: np.ndarray, side: np.ndarray) -> dict[str, float]:
    """
    Evaluate HOME/AWAY/PASS strategy on ATS label y_true (home_cover).
    """
    take = side != 0
    n_bets = int(np.sum(take))
    if n_bets == 0:
        return {"n_bets": 0.0, "win_rate": 0.0, "roi_per_bet": 0.0}

    win_payout = 100.0 / 110.0
    take_home = side == +1
    take_away = side == -1
    correct = (take_home & (y_true == 1)) | (take_away & (y_true == 0))
    wins = int(np.sum(correct))
    losses = n_bets - wins
    profit = wins * win_payout - losses * 1.0
    return {
        "n_bets": float(n_bets),
        "win_rate": float(wins / float(n_bets)),
        "roi_per_bet": float(profit / float(n_bets)),
    }


def _self_test_walkforward_no_leakage() -> None:
    """
    Minimal sanity test:
    - After the first game for a team, season/site/recent diffs should be 0
      because there is no prior data.
    - After one prior game, recent should reflect that single outcome.
    """
    toy = pd.DataFrame(
        [
            # game 1: A(home) beats B by 10, spread -2
            dict(
                game_id=1,
                date="2020-01-01",
                home_team_id=100,
                away_team_id=200,
                opening_spread=-2.0,
                home_score=110,
                away_score=100,
            ),
            # game 2: A(home) loses to B by 5, spread -4
            dict(
                game_id=2,
                date="2020-01-02",
                home_team_id=100,
                away_team_id=200,
                opening_spread=-4.0,
                home_score=95,
                away_score=100,
            ),
        ]
    )
    toy = add_ats_home_cover_label(toy, drop_pushes=True)
    df = build_walkforward_dataset(toy)

    r1 = df.iloc[0]
    assert r1["season_pd_diff"] == 0.0
    assert r1["site_pd_diff"] == 0.0
    assert r1["recent_pd_diff"] == 0.0

    r2 = df.iloc[1]
    # After game 1: A has +10 recent, B has -10 recent => diff = 20
    assert abs(float(r2["recent_pd_diff"]) - 20.0) < 1e-9


def train_stat_builder_weights(*, recent_only: bool = False) -> dict[str, Any]:
    """
    Fit stat_builder-shaped weights (blend + HCA + margin_mult) plus a small
    logistic head, minimizing mean log loss on the training slice (time-ordered).
    """
    print("Loading raw team-game history...")
    raw = load_raw_team_game_history()
    raw["date"] = pd.to_datetime(raw["date"])

    if recent_only:
        raw = raw[raw["date"] >= pd.Timestamp(RECENT_CUTOFF)].copy()
        print(f"  --recent: rows kept = {len(raw)} (>= {RECENT_CUTOFF})")

    print("Building game table...")
    games = _build_game_table(raw)
    games = add_ats_home_cover_label(games, drop_pushes=True)
    print(f"  Games with label (pushes dropped): {len(games)}")

    print("Building walk-forward leakage-free features...")
    df = build_walkforward_dataset(games)

    diff_cols = ["season_pd_diff", "site_pd_diff", "recent_pd_diff"]
    feature_cols = [*diff_cols, "opening_spread"]

    train_df, test_df = _temporal_split(df)
    d_s_tr = train_df["season_pd_diff"].values.astype(float)
    d_i_tr = train_df["site_pd_diff"].values.astype(float)
    d_r_tr = train_df["recent_pd_diff"].values.astype(float)
    spr_tr = train_df["opening_spread"].values.astype(float)
    y_train = train_df["home_cover"].values.astype(int)

    d_s_te = test_df["season_pd_diff"].values.astype(float)
    d_i_te = test_df["site_pd_diff"].values.astype(float)
    d_r_te = test_df["recent_pd_diff"].values.astype(float)
    spr_te = test_df["opening_spread"].values.astype(float)
    y_test = test_df["home_cover"].values.astype(int)

    # Near hand-tuned v2: equal mix -> softmax(0), HCA 1.5, margin ~1.1, mild logit slope
    x0 = np.array(
        [0.0, 0.0, 0.0, 1.5, 0.0, np.log(0.11), 0.0],
        dtype=float,
    )

    print("Optimizing structured stat_builder parameters (L-BFGS-B)...")
    result = minimize(
        _structured_mean_nll,
        x0,
        args=(d_s_tr, d_i_tr, d_r_tr, spr_tr, y_train),
        method="L-BFGS-B",
        bounds=[(-10.0, 10.0)] * 7,
        options={"maxiter": 8000, "ftol": 1e-10},
    )
    eta_opt = np.asarray(result.x, dtype=float)
    params = unpack_structured_params(eta_opt)
    train_nll = float(_structured_mean_nll(eta_opt, d_s_tr, d_i_tr, d_r_tr, spr_tr, y_train))

    y_prob_train = predict_p_cover_structured(d_s_tr, d_i_tr, d_r_tr, spr_tr, params)
    y_prob = predict_p_cover_structured(d_s_te, d_i_te, d_r_te, spr_te, params)

    metrics: dict[str, Any] = {
        "task": "home_cover_opening_spread",
        "model_family": "structured_stat_builder_sigmoid",
        "n_games_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_date_range": [
            str(train_df["date"].min().date()),
            str(train_df["date"].max().date()),
        ],
        "test_date_range": [
            str(test_df["date"].min().date()),
            str(test_df["date"].max().date()),
        ],
        "train_log_loss": float(log_loss(y_train, y_prob_train)),
        "log_loss": float(log_loss(y_test, y_prob)),
        "brier_score": float(brier_score_loss(y_test, y_prob)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "structured_train_nll": train_nll,
        "optimization_success": bool(result.success),
        "optimization_message": str(result.message),
        "stat_builder": {
            "w_season": params["w_season"],
            "w_site": params["w_site"],
            "w_recent": params["w_recent"],
            "home_court_advantage": params["home_court_advantage"],
            "margin_mult": params["margin_mult"],
        },
        "logit_head": {
            "scale": params["logit_scale"],
            "bias": params["logit_bias"],
        },
    }

    tau_choice = _choose_tau_for_pass_band(y_test, y_prob)
    k_edge_points = _fit_k_edge_points(train_df, y_prob_train)
    metrics["mapping_tau"] = float(tau_choice["tau"])
    metrics["mapping_tau_eval"] = tau_choice
    metrics["k_edge_points"] = float(k_edge_points)

    tau = float(tau_choice["tau"])
    side_learned = np.zeros(len(test_df), dtype=int)
    side_learned[y_prob >= (0.5 + tau)] = +1
    side_learned[y_prob <= (0.5 - tau)] = -1
    learned_eval = _evaluate_side_strategy(y_test, side_learned)

    side_v2 = _stat_builder_v2_proxy_side(test_df)
    v2_eval = _evaluate_side_strategy(y_test, side_v2)

    metrics["strategy_eval"] = {
        "learned_structured": learned_eval,
        "stat_builder_v2_proxy": v2_eval,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version = f"stat_builder_weights_struct_{ts}"

    bundle: dict[str, Any] = {
        "model_version": model_version,
        "trained_at": ts,
        "feature_cols": feature_cols,
        "formula_notes": (
            "Shared blend: home_strength = w_season*h_season_pd + w_site*h_site_pd + "
            "w_recent*h_recent_pd; away same weights; "
            "projected_margin = margin_mult * ((home_strength - away_strength) + home_court_advantage); "
            "p_cover = sigmoid(logit_bias + logit_scale * (projected_margin + opening_spread))."
        ),
        "stat_builder": metrics["stat_builder"],
        "logit_head": metrics["logit_head"],
        "raw_optimizer_eta": eta_opt.astype(float).tolist(),
        "metrics": metrics,
        "mapping": {
            "tau": float(tau_choice["tau"]),
            "tau_selection": tau_choice,
            "k_edge_points": float(k_edge_points),
        },
    }

    out_path = ARTIFACTS_DIR / f"{model_version}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    metrics["model_version"] = model_version
    metrics["bundle_path"] = str(out_path)

    sb = metrics["stat_builder"]
    print("\n--- Learned stat_builder weights (use in stat_builder.py template) ---")
    print(
        f"  w_season={sb['w_season']:.4f}  w_site={sb['w_site']:.4f}  "
        f"w_recent={sb['w_recent']:.4f}"
    )
    print(
        f"  home_court_advantage={sb['home_court_advantage']:.4f}  "
        f"margin_mult={sb['margin_mult']:.4f}"
    )
    print(
        f"  logit_head: scale={params['logit_scale']:.6f}  bias={params['logit_bias']:.6f}"
    )
    print("\n--- Test Metrics (home_cover) ---")
    print(f"  Train Log Loss: {metrics['train_log_loss']:.4f}")
    print(f"  Test Log Loss:  {metrics['log_loss']:.4f}")
    print(f"  Brier Score:    {metrics['brier_score']:.4f}")
    print(f"  ROC-AUC:        {metrics['roc_auc']:.4f}")
    print("\n--- Strategy Backtest (test set, -110 proxy) ---")
    print(
        "  LearnedStructured:"
        f" bets={int(learned_eval['n_bets'])}"
        f" win_rate={learned_eval['win_rate']:.3f}"
        f" roi/bet={learned_eval['roi_per_bet']:.4f}"
        f" (tau={tau:.3f})"
    )
    print(
        "  StatBuilderV2Proxy:"
        f" bets={int(v2_eval['n_bets'])}"
        f" win_rate={v2_eval['win_rate']:.3f}"
        f" roi/bet={v2_eval['roi_per_bet']:.4f}"
    )
    print(f"  Saved:       {out_path.name}")

    return metrics


if __name__ == "__main__":
    import sys

    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if "--self-test" in flags:
        _self_test_walkforward_no_leakage()
        print("Self-test passed.")
        raise SystemExit(0)

    recent = "--recent" in flags
    train_stat_builder_weights(recent_only=recent)

