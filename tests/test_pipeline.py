"""Smoke + property tests for the pipeline (TZ outputs)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ru_liquidity_sentinel.aggregate import (
    detect_stress_episodes,
    fit_model_suite,
    lightgbm_lsi,
    module_columns,
    noise_breakdown,
    sensitivity_analysis,
)
from ru_liquidity_sentinel.config import LSI_THRESHOLDS, PipelineConfig
from ru_liquidity_sentinel.normalize import (
    robust_online_cusum_series,
    rolling_mad_zscore,
    smooth_ewma,
    winsorize_rolling,
    zscore_to_subindex,
)
from ru_liquidity_sentinel.pipeline import run_pipeline
from ru_liquidity_sentinel.reporting import write_all_reports


def _calendar(n: int = 600) -> pd.DatetimeIndex:
    return pd.date_range("2018-01-01", periods=n, freq="D")


# ---------------------------------------------------------------------------
# Unit tests for normalize.py
# ---------------------------------------------------------------------------


def test_mad_zscore_constant_zero():
    cal = _calendar(120)
    s = pd.Series(np.full(len(cal), 7.0), index=cal)
    z = rolling_mad_zscore(s, window_days=90, min_periods=20)
    assert (z == 0.0).all()


def test_mad_zscore_outlier():
    cal = _calendar(200)
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 1, len(cal)), index=cal)
    s.iloc[-1] = 10.0
    z = rolling_mad_zscore(s, window_days=180, direction="upper")
    assert z.iloc[-1] > 3.0
    assert (z >= 0).all()


def test_subindex_bounds():
    z = pd.Series([0.0, 1.0, 2.0, 4.0, 10.0])
    sub = zscore_to_subindex(z, scale=2.0)
    assert (sub >= 0).all() and (sub <= 100).all()
    assert sub.iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_winsorize_rolling_caps_extreme_values():
    cal = _calendar(500)
    rng = np.random.default_rng(11)
    base = rng.normal(loc=10, scale=1, size=len(cal))
    base[250] = 5000.0
    s = pd.Series(base, index=cal)
    capped = winsorize_rolling(s, window_days=180, upper_q=0.995, min_periods=30)
    assert capped.max() < 100
    untouched = (capped.iloc[:200] == s.iloc[:200]).mean()
    assert untouched > 0.95


def test_smooth_ewma_reduces_noise():
    cal = _calendar(400)
    rng = np.random.default_rng(42)
    s = pd.Series(rng.normal(0, 1, len(cal)), index=cal)
    smoothed = smooth_ewma(s, halflife_days=7.0)
    assert smoothed.diff().std() < s.diff().std()


def test_robust_online_cusum_burnin_and_response():
    """MIO emits 0 during burn-in then ramps up on a sustained shock."""

    cal = _calendar(400)
    rng = np.random.default_rng(7)
    base = rng.normal(0, 1, len(cal))
    s = pd.Series(base, index=cal)
    out = robust_online_cusum_series(s.rename("test"), window_size=200, threshold=10.0)
    assert (out.iloc[:21] == 0.0).all()
    # Inject a sustained 50-day shock.
    s.iloc[200:250] = s.iloc[200:250] + 8.0
    out2 = robust_online_cusum_series(s.rename("test"), window_size=200, threshold=10.0)
    # After the shock peak the value should reach ≈ 1 (capped) and decay
    # quickly when the signal returns to normal.
    assert out2.iloc[200:260].max() > 0.5
    assert (out2 >= 0.0).all() and (out2 <= 1.0).all()


# ---------------------------------------------------------------------------
# Module output shape tests (TZ: mad + flag + mio for every module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke_run():
    cfg = PipelineConfig(fit_models=True)
    return run_pipeline(cfg)


def test_module_outputs_follow_tz(smoke_run):
    feat = smoke_run.features
    for module in ("M1", "M2", "M3", "M4", "M5"):
        cols = module_columns(feat, module)
        assert cols["mad"], f"{module}: missing mad_score columns"
        assert cols["flag"], f"{module}: missing flag columns"
        assert cols["mio"], f"{module}: missing mio_cusum column"


def test_module_features_no_subindex(smoke_run):
    """The TZ refactor removes the legacy *_subindex output entirely."""

    feat = smoke_run.features
    legacy = [c for c in feat.columns if c.endswith("_subindex")]
    assert legacy == [], legacy


def test_mio_cusum_in_unit_interval(smoke_run):
    feat = smoke_run.features
    for module in ("M1", "M2", "M3", "M4", "M5"):
        col = f"{module.lower()}_mio_cusum"
        assert col in feat.columns
        s = feat[col].dropna()
        assert (s >= 0.0).all() and (s <= 1.0).all(), col


def test_m3_auction_silence_days_present_and_nonneg(smoke_run):
    feat = smoke_run.features
    assert "m3_auction_silence_days" in feat.columns
    assert (feat["m3_auction_silence_days"] >= 0).all()
    assert feat["m3_auction_silence_days"].max() > 30


# ---------------------------------------------------------------------------
# Aggregator tests
# ---------------------------------------------------------------------------


def test_run_pipeline_smoke(smoke_run):
    run = smoke_run
    assert (run.lsi["lsi"] >= 0).all() and (run.lsi["lsi"] <= 100).all()
    for module in ("M1", "M2", "M3", "M4", "M5"):
        s = run.lsi[f"score_{module}"]
        assert (s >= 0).all() and (s <= 100).all(), module
    sf = run.features["m4_seasonal_factor"]
    assert (sf >= 1.0).all() and (sf <= 1.4).all()
    assert len(run.backtest) == len(run.config.backtest_episodes)
    assert "mae" in run.holdout
    assert not run.sensitivity_summary.empty


def test_lightgbm_trained(smoke_run):
    """LightGBM is the sole aggregator — it must train to completion."""

    run = smoke_run
    assert run.models is not None
    names = set(run.models.models.keys())
    assert "lightgbm" in names
    # No other models in the LightGBM-only configuration.
    assert names == {"lightgbm"}


def test_lightgbm_predictions_in_unit_interval(smoke_run):
    """LightGBM predictions (after log calibration) must be in [0, 100]."""

    run = smoke_run
    lgb = run.models.models["lightgbm"]
    pred = lgb.predictions.dropna()
    assert not pred.empty
    assert (pred >= 0.0).all() and (pred <= 100.0).all()


def test_lightgbm_module_attribution_shape(smoke_run):
    """LightGBM must expose per-module SHAP attribution (TZ requirement:
    contribution of each module to the final signal)."""

    run = smoke_run
    lgb = run.models.models["lightgbm"]
    attr = lgb.module_attribution
    assert attr is not None and not attr.empty
    for module in ("M1", "M2", "M3", "M4", "M5"):
        col = f"contrib_{module}"
        assert col in attr.columns, f"missing {col} in LightGBM attribution"
        share_col = f"share_{module}"
        assert share_col in attr.columns
        valid = attr[share_col].dropna()
        assert (valid >= -1e-6).all() and (valid <= 1.0 + 1e-6).all(), share_col

    # On rows where the LSI is non-zero, shares must sum to ~1.
    share_cols = [f"share_{m}" for m in ("M1", "M2", "M3", "M4", "M5")]
    nonzero = attr[share_cols].sum(axis=1)
    nonzero = nonzero[nonzero > 1e-9]
    assert nonzero.between(0.95, 1.05).mean() > 0.95


def test_log_calibration_monotonic_and_bounded():
    """LogCalibration must be monotone and saturate to [0, 100]."""

    from ru_liquidity_sentinel.gbm_lsi import LogCalibration

    rng = np.random.default_rng(7)
    # Heavy-tailed synthetic train distribution (log-normal).
    train = rng.lognormal(mean=0.0, sigma=1.0, size=2000)
    cal = LogCalibration.fit(train, q_low=0.10, q_high=0.99)

    # Probe monotonicity on a finely spaced grid.
    grid = np.linspace(train.min(), train.max() * 2, 200)
    out = cal.transform(grid)
    assert (np.diff(out) >= -1e-9).all(), "calibration is not monotone"
    # Saturation: any value above q_high should clip at 100; below q_low at 0.
    assert cal.transform(np.array([train.max() * 100]))[0] == 100.0
    assert cal.transform(np.array([train.min() / 100]))[0] == 0.0
    # Anchor matches: at q_low we should be at 0, at q_high at 100.
    qlow_value = float(np.quantile(train, 0.10))
    qhigh_value = float(np.quantile(train, 0.99))
    assert cal.transform(np.array([qlow_value]))[0] == pytest.approx(0.0, abs=1e-6)
    assert cal.transform(np.array([qhigh_value]))[0] == pytest.approx(100.0, abs=1e-6)


def test_log_calibration_handles_negative_train():
    """If the model occasionally outputs negative predictions, the
    calibrator must still produce a valid LSI."""

    from ru_liquidity_sentinel.gbm_lsi import LogCalibration

    rng = np.random.default_rng(11)
    train = rng.normal(loc=-0.5, scale=1.0, size=500)  # ~ half negative
    cal = LogCalibration.fit(train)
    out = cal.transform(train)
    assert np.isfinite(out).all()
    assert (out >= 0).all() and (out <= 100).all()


def test_walk_forward_cv_per_fold_results(smoke_run):
    """Each tuned model must expose per-fold walk-forward CV metrics.

    The validation windows must be strictly chronological (val_start of
    fold k+1 strictly later than val_start of fold k), the train window
    must be expanding (n_train monotonic non-decreasing), and the train
    window must end strictly before the val window.
    """

    run = smoke_run
    cv = run.models.cv_table()
    assert not cv.empty
    for model in ("lightgbm",):
        sub = cv[cv["model"] == model]
        assert not sub.empty
        n_unique_folds = sub["fold"].nunique()
        assert n_unique_folds >= 3, f"{model}: too few folds ({n_unique_folds})"
        # Pick the *first* config to verify chronology — it's the same for
        # every config because TimeSeriesSplit indices only depend on n.
        first_params = sub["params"].iloc[0]
        ordered = sub[sub["params"] == first_params].sort_values("fold").reset_index(drop=True)
        train_starts = pd.to_datetime(ordered["train_start"])
        val_starts = pd.to_datetime(ordered["val_start"])
        train_ends = pd.to_datetime(ordered["train_end"])
        val_ends = pd.to_datetime(ordered["val_end"])
        # No leakage: train must end strictly before val starts.
        assert (train_ends < val_starts).all(), model
        # Walk-forward: val windows advance monotonically forward in time.
        assert val_starts.is_monotonic_increasing
        assert val_ends.is_monotonic_increasing
        # Expanding window: training set never shrinks.
        assert ordered["n_train"].is_monotonic_increasing
        # Folds never touch the held-out test window (>= 2024-01-01).
        assert val_ends.max() < pd.Timestamp("2024-01-01")


def test_walk_forward_best_params_recorded(smoke_run):
    """best_params + cv_summary must be populated for every tuned model."""

    run = smoke_run
    for name in ("lightgbm",):
        m = run.models.models[name]
        assert m.best_params, f"{name}: missing best_params"
        assert m.cv_summary, f"{name}: missing cv_summary"
        assert "cv_val_mae" in m.cv_summary
        assert m.cv_summary["cv_val_mae"] > 0


def test_no_smoothing_columns(smoke_run):
    """LSI must be raw — no `lsi_smoothed` / `lsi_raw` distinction anymore."""

    run = smoke_run
    # Per the refactor the LSI is computed once, no EWMA stage.
    assert "lsi_smoothed" not in run.lsi.columns


def test_noise_breakdown_sums_to_total(smoke_run):
    """Decomposition of Var(ΔLSI) into per-module + cross-cov terms must
    reconstruct the empirical Var(ΔLSI).

    With LSI = Σ contrib_M{i} by construction (share_M{i} sum to 1 →
    contrib_M{i} sum to LSI), the decomposition

        Var(ΔLSI) = Σ_i Var(Δcontrib_M{i}) + 2 Σ_{i<j} Cov(Δcontrib_M{i}, Δcontrib_M{j})

    holds exactly up to numerical noise.
    """

    run = smoke_run
    nb = run.noise_breakdown
    assert not nb.empty
    decomposed = nb["contribution"].sum()
    empirical = float(run.lsi["lsi"].diff().var(ddof=0))
    assert decomposed == pytest.approx(empirical, rel=0.05, abs=0.5)


def test_auto_episode_detection(smoke_run):
    episodes = smoke_run.auto_episodes
    assert len(episodes) >= 3
    assert episodes["max_lsi"].is_monotonic_decreasing


def test_status_thresholds(smoke_run):
    statuses = smoke_run.lsi["status"].unique().tolist()
    assert set(statuses).issubset({"green", "yellow", "red"})


def test_sensitivity_does_not_explode(smoke_run):
    """Sensitivity perturbs LightGBM hyper-parameters ±20% and re-runs
    training.  Deviations must stay bounded — a small perturbation of
    learning_rate / num_leaves / min_child_samples shouldn't reshape
    the whole LSI series.
    """

    run = smoke_run
    sens = run.sensitivity
    if sens is None or sens.empty:
        pytest.skip("sensitivity analysis produced no scenarios")
    assert "lsi_base" in sens.columns
    base = sens["lsi_base"]
    for col in sens.columns:
        if col == "lsi_base":
            continue
        diff = (sens[col] - base).abs()
        # Retrained LightGBM can shift the LSI more than a scalar weight
        # change; 30-point mean absolute deviation is the upper bound.
        assert diff.mean() < 30.0, col


def test_detect_stress_episodes_synthetic():
    cal = _calendar(200)
    s = pd.Series(np.zeros(len(cal)), index=cal)
    s.iloc[20:30] = 90.0
    s.iloc[80:95] = 80.0
    eps = detect_stress_episodes(s, threshold=70.0, min_episode_days=3)
    assert len(eps) == 2
    assert eps.iloc[0]["max_lsi"] >= eps.iloc[1]["max_lsi"]


def test_reporting_writes_files(tmp_path, smoke_run):
    """Reporting must serialise to disk without crashing.  Re-uses the
    smoke_run (which already trained LightGBM) and just re-points the
    output directories — fit_models=False is not supported by the
    LightGBM-based pipeline.
    """

    cfg = smoke_run.config
    cfg.artifacts_dir = tmp_path / "artifacts"
    cfg.reports_dir = tmp_path / "reports"
    cfg.figures_dir = tmp_path / "reports" / "figures"
    cfg.tables_dir = tmp_path / "reports" / "tables"
    cfg.ensure_dirs()
    paths = write_all_reports(smoke_run, cfg)
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
