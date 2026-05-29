"""Comprehensive evaluation metrics for the Liquidity Stress Index pipeline.

Computes four families of metrics for each of the three models in the
final aggregation layer (NSVM), plus the
weighted-sum LSI as a non-ML baseline:

1.  **Regression** — model predictions vs the proxy stress target on a
    chronological train (< 2024-01-01) / test (≥ 2024-01-01) split.
    Reports MAE, RMSE, MAPE, R², explained variance, bias, Pearson,
    Spearman.

2.  **Classification** — predictions used as a binary stress detector
    against the proxy ≥ p90 / p95 thresholds.  Reports ROC-AUC, PR-AUC,
    lift@top-5/10%, best-F1 threshold, and confusion-matrix metrics
    at the LSI status thresholds (40 / 70 / 90).

3.  **Temporal stability** — Day-to-day diff std on the LSI, max
    single-day jump, lag-1 autocorrelation, "ringing" rate (sign flips
    in the diff).  Computed only for the LSI, since predictions live
    in proxy-stress units.

4.  **Episode-level** — for the three TZ episodes plus auto-detected
    stress windows: lead-time (days from first ``LSI ≥ 40`` to start),
    coverage (fraction of episode days in red / yellow zones), max LSI.

All metrics are written to ``reports/tables/eval_metrics.csv`` and a
plain-text summary is printed.  ROC, PR and calibration curves are
emitted under ``reports/figures/`` for every model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ru_liquidity_sentinel.config import LSI_THRESHOLDS, PipelineConfig


MODEL_NAMES = ("nsvm",)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_artifacts(cfg: PipelineConfig) -> Dict:
    art = cfg.artifacts_dir
    out = {
        "features": pd.read_parquet(art / "features_daily.parquet"),
        "lsi": pd.read_parquet(art / "lsi_daily.parquet"),
        "proxy": pd.read_parquet(art / "proxy_target.parquet")["proxy_stress"],
        "metadata": json.loads((art / "metadata.json").read_text()),
        "auto_episodes": pd.read_csv(
            art / "auto_episodes.csv", parse_dates=["start", "end"]
        )
        if (art / "auto_episodes.csv").exists()
        else pd.DataFrame(),
    }
    p_pred = art / "model_predictions.parquet"
    if p_pred.exists() and p_pred.stat().st_size > 0:
        out["model_predictions"] = pd.read_parquet(p_pred)
    else:
        out["model_predictions"] = pd.DataFrame()
    p_metrics = art / "model_metrics.csv"
    if p_metrics.exists():
        out["model_metrics"] = pd.read_csv(p_metrics)
    else:
        out["model_metrics"] = pd.DataFrame()
    return out


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def regression_metrics(y_true: pd.Series, y_pred: pd.Series, label: str) -> Dict:
    y_true = y_true.dropna()
    y_pred = y_pred.reindex(y_true.index).dropna()
    y_true = y_true.reindex(y_pred.index)
    if len(y_true) == 0:
        return {"split": label, "n": 0}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    expvar = explained_variance_score(y_true, y_pred)
    nz = y_true > 1.0
    mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100) if nz.any() else float("nan")
    bias = float((y_pred - y_true).mean())
    spearman = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))
    pearson = float(pd.Series(y_true).corr(pd.Series(y_pred), method="pearson"))

    return {
        "split": label,
        "n": int(len(y_true)),
        "mae": float(mae),
        "rmse": rmse,
        "mape_pct": mape,
        "r2": float(r2),
        "explained_variance": float(expvar),
        "bias_mean_pred_minus_true": bias,
        "pearson_r": pearson,
        "spearman_r": spearman,
    }


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_metrics(score: pd.Series, label: pd.Series, name: str) -> Dict:
    df = pd.DataFrame({"score": score, "label": label}).dropna()
    if df["label"].nunique() < 2:
        return {"name": name, "n": int(len(df)), "n_pos": int(df["label"].sum())}

    y = df["label"].astype(int).values
    s = df["score"].astype(float).values
    auc = float(roc_auc_score(y, s))
    ap = float(average_precision_score(y, s))
    base_rate = float(y.mean())

    prec, rec, thr = precision_recall_curve(y, s)
    f1_curve = (2 * prec * rec) / (prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f1_curve))
    best_thr = float(thr[best_idx]) if best_idx < len(thr) else float(s.max())
    best_f1 = float(f1_curve[best_idx])

    return {
        "name": name,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "base_rate": base_rate,
        "roc_auc": auc,
        "pr_auc": ap,
        "lift_at_top10pct": _lift_at_top_k(y, s, 0.10),
        "lift_at_top5pct": _lift_at_top_k(y, s, 0.05),
        "best_f1": best_f1,
        "best_f1_threshold": best_thr,
    }


def _lift_at_top_k(y: np.ndarray, s: np.ndarray, k: float) -> float:
    n = len(y)
    if n == 0:
        return float("nan")
    order = np.argsort(-s)
    top_n = max(1, int(n * k))
    base = y.mean()
    if base == 0:
        return float("nan")
    top = y[order[:top_n]].mean()
    return float(top / base)


def threshold_metrics(score: pd.Series, label: pd.Series, threshold: float, name: str) -> Dict:
    df = pd.DataFrame({"score": score, "label": label}).dropna()
    y = df["label"].astype(int).values
    s = (df["score"].astype(float).values >= threshold).astype(int)
    if y.sum() == 0 or len(np.unique(y)) < 2:
        return {"name": name, "threshold": threshold, "n": int(len(y))}
    cm = confusion_matrix(y, s)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "name": name,
        "threshold": float(threshold),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(precision_score(y, s, zero_division=0)),
        "recall": float(recall_score(y, s, zero_division=0)),
        "f1": float(f1_score(y, s, zero_division=0)),
        "accuracy": float(accuracy_score(y, s)),
        "balanced_accuracy": float(balanced_accuracy_score(y, s)),
        "mcc": float(matthews_corrcoef(y, s)) if len(np.unique(s)) > 1 else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Temporal stability metrics
# ---------------------------------------------------------------------------

def temporal_stability(lsi: pd.DataFrame) -> Dict:
    """LSI is no longer smoothed — we report jitter directly on it."""

    s = lsi["lsi"].dropna()
    diff = s.diff().dropna()
    sign_flips = float(((diff.shift() * diff) < 0).sum()) / max(1, len(diff))
    return {
        "n": int(len(s)),
        "std_diff_lsi": float(diff.std()),
        "max_abs_jump_lsi": float(diff.abs().max()),
        "lag1_autocorr_lsi": float(s.autocorr(lag=1)),
        "sign_flip_rate_lsi": sign_flips,
    }


# ---------------------------------------------------------------------------
# Episode-level metrics
# ---------------------------------------------------------------------------

def episode_lead_time(
    lsi: pd.Series, start: pd.Timestamp, end: pd.Timestamp, threshold: float
) -> int:
    look_start = start - pd.Timedelta(days=30)
    pre = lsi.loc[(lsi.index >= look_start) & (lsi.index <= end)]
    if pre.empty:
        return None
    above = pre[pre >= threshold]
    if above.empty:
        return None
    first_cross = above.index.min()
    return int((start - first_cross).days)


def episode_metrics(
    lsi: pd.Series,
    episodes: List[Tuple[str, str, str]],
    red: float,
    yellow: float,
) -> pd.DataFrame:
    rows = []
    for name, s, e in episodes:
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        win = lsi.loc[(lsi.index >= s) & (lsi.index <= e)]
        if win.empty:
            continue
        rows.append(
            {
                "episode": name,
                "start": s.date(),
                "end": e.date(),
                "days": int(len(win)),
                "mean_lsi": float(win.mean()),
                "max_lsi": float(win.max()),
                "days_red": int((win >= red).sum()),
                "days_yellow": int(((win >= yellow) & (win < red)).sum()),
                "frac_red": float((win >= red).mean()),
                "frac_yellow_or_red": float((win >= yellow).mean()),
                "lead_time_yellow_days": episode_lead_time(lsi, s, e, yellow),
                "lead_time_red_days": episode_lead_time(lsi, s, e, red),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_roc(score: pd.Series, label: pd.Series, path: Path, title: str) -> Path:
    df = pd.DataFrame({"s": score, "y": label}).dropna()
    if df["y"].nunique() < 2:
        return path
    fpr, tpr, _ = roc_curve(df["y"].astype(int), df["s"].astype(float))
    auc = roc_auc_score(df["y"].astype(int), df["s"].astype(float))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=1.6, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pr(score: pd.Series, label: pd.Series, path: Path, title: str) -> Path:
    df = pd.DataFrame({"s": score, "y": label}).dropna()
    if df["y"].nunique() < 2:
        return path
    prec, rec, _ = precision_recall_curve(df["y"].astype(int), df["s"].astype(float))
    ap = average_precision_score(df["y"].astype(int), df["s"].astype(float))
    base_rate = float(df["y"].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, color="#d62728", lw=1.6, label=f"AP = {ap:.3f}")
    ax.axhline(base_rate, color="grey", ls="--", lw=0.5, label=f"base = {base_rate:.2f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_calibration(
    pred: pd.Series, target: pd.Series, path: Path, title: str, n_bins: int = 10
) -> Path:
    df = pd.DataFrame({"pred": pred, "target": target}).dropna()
    if df.empty or df["pred"].nunique() < 2:
        return path
    df["bin"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")
    grouped = df.groupby("bin", observed=True).agg(
        mean_pred=("pred", "mean"),
        mean_target=("target", "mean"),
        n=("pred", "size"),
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(grouped["mean_pred"], grouped["mean_target"], marker="o", color="#1f77b4", lw=1.4)
    lim = [
        min(df["pred"].min(), df["target"].min()),
        max(df["pred"].max(), df["target"].max()),
    ]
    ax.plot(lim, lim, "k--", lw=0.5)
    ax.set_xlabel("predicted")
    ax.set_ylabel("observed proxy stress")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    cfg = PipelineConfig()
    art = load_artifacts(cfg)

    lsi_df = art["lsi"]
    lsi = lsi_df["lsi"]
    proxy = art["proxy"]
    preds = art["model_predictions"]

    # Splits.
    test_start = pd.Timestamp("2024-01-01")
    train_idx = lsi.index < test_start
    test_idx = lsi.index >= test_start

    # ---------------------- 1. Regression ---------------------- #
    reg_rows: List[Dict] = []
    for model in MODEL_NAMES:
        if model not in preds.columns:
            continue
        pred = preds[model]
        reg_rows.append(regression_metrics(proxy.loc[train_idx], pred.loc[train_idx], f"{model}_train"))
        reg_rows.append(regression_metrics(proxy.loc[test_idx], pred.loc[test_idx], f"{model}_test"))

    # Log-calibrated NSVM LSI vs target — same series as the
    # model output above; the row is kept for backwards compatibility
    # of the output schema (downstream tools expect `lsi_train/test`).
    reg_rows.append(regression_metrics(proxy.loc[train_idx], lsi.loc[train_idx], "lsi_train"))
    reg_rows.append(regression_metrics(proxy.loc[test_idx], lsi.loc[test_idx], "lsi_test"))
    reg_df = pd.DataFrame(reg_rows)

    # ---------------------- 2. Classification ---------------------- #
    p90 = float(proxy.quantile(0.90))
    p95 = float(proxy.quantile(0.95))
    print(f"[eval] proxy p90 = {p90:.3f}, p95 = {p95:.3f}")
    stress_label_p90 = (proxy >= p90).astype(int)
    stress_label_p95 = (proxy >= p95).astype(int)

    cls_rows: List[Dict] = [
        classification_metrics(lsi.loc[test_idx], stress_label_p90.loc[test_idx], "lsi_vs_target_p90_test"),
        classification_metrics(lsi, stress_label_p90, "lsi_vs_target_p90_full"),
        classification_metrics(lsi.loc[test_idx], stress_label_p95.loc[test_idx], "lsi_vs_target_p95_test"),
        classification_metrics(lsi, stress_label_p95, "lsi_vs_target_p95_full"),
    ]
    for model in MODEL_NAMES:
        if model not in preds.columns:
            continue
        pred = preds[model]
        cls_rows.append(
            classification_metrics(pred.loc[test_idx], stress_label_p90.loc[test_idx], f"{model}_vs_p90_test")
        )
        cls_rows.append(
            classification_metrics(pred, stress_label_p90, f"{model}_vs_p90_full")
        )
        cls_rows.append(
            classification_metrics(pred.loc[test_idx], stress_label_p95.loc[test_idx], f"{model}_vs_p95_test")
        )
    cls_df = pd.DataFrame(cls_rows)

    # Threshold metrics on the LSI scale (40 / 70 / 90).
    thr_rows = []
    for thr, name in [(40.0, "lsi_ge40"), (70.0, "lsi_ge70"), (90.0, "lsi_ge90")]:
        thr_rows.append(threshold_metrics(lsi, stress_label_p90, thr, name + "_full_p90"))
        thr_rows.append(
            threshold_metrics(lsi.loc[test_idx], stress_label_p90.loc[test_idx], thr, name + "_test_p90")
        )
    thr_df = pd.DataFrame(thr_rows)

    # ---------------------- 3. Temporal stability ---------------------- #
    stab = temporal_stability(lsi_df)

    # ---------------------- 4. Episodes ---------------------- #
    tz_eps = episode_metrics(
        lsi,
        cfg.backtest_episodes,
        red=LSI_THRESHOLDS["yellow_max"],
        yellow=LSI_THRESHOLDS["green_max"],
    )

    auto_eps_df = art["auto_episodes"]
    if not auto_eps_df.empty:
        episodes = list(
            zip(
                [f"auto_{int(r):02d}" for r in auto_eps_df["rank"]],
                auto_eps_df["start"].dt.strftime("%Y-%m-%d"),
                auto_eps_df["end"].dt.strftime("%Y-%m-%d"),
            )
        )
        auto_eps = episode_metrics(
            lsi,
            episodes,
            red=LSI_THRESHOLDS["yellow_max"],
            yellow=LSI_THRESHOLDS["green_max"],
        )
    else:
        auto_eps = pd.DataFrame()

    # ---------------------- IO ---------------------- #
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    reg_df.to_csv(cfg.tables_dir / "eval_regression.csv", index=False)
    cls_df.to_csv(cfg.tables_dir / "eval_classification.csv", index=False)
    thr_df.to_csv(cfg.tables_dir / "eval_thresholds.csv", index=False)
    pd.DataFrame([stab]).T.rename(columns={0: "value"}).to_csv(
        cfg.tables_dir / "eval_temporal_stability.csv"
    )
    if not tz_eps.empty:
        tz_eps.to_csv(cfg.tables_dir / "eval_tz_episodes.csv", index=False)
    if not auto_eps.empty:
        auto_eps.to_csv(cfg.tables_dir / "eval_auto_episodes.csv", index=False)

    # Combined long-form summary.
    long_rows: List[Dict] = []
    for r in reg_rows:
        for k, v in r.items():
            if k == "split":
                continue
            long_rows.append({"category": "regression", "subset": r["split"], "metric": k, "value": v})
    for r in cls_rows:
        for k, v in r.items():
            if k == "name":
                continue
            long_rows.append({"category": "classification", "subset": r["name"], "metric": k, "value": v})
    for r in thr_rows:
        for k, v in r.items():
            if k == "name":
                continue
            long_rows.append({"category": "threshold", "subset": r["name"], "metric": k, "value": v})
    for k, v in stab.items():
        long_rows.append({"category": "temporal", "subset": "lsi", "metric": k, "value": v})
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(cfg.tables_dir / "eval_metrics.csv", index=False)

    # ---------------------- Figures ---------------------- #
    plot_roc(lsi, stress_label_p90, cfg.figures_dir / "roc_lsi_full.png",
             "ROC: LSI vs proxy ≥ p90 (full sample)")
    plot_pr(lsi, stress_label_p90, cfg.figures_dir / "pr_lsi_full.png",
            "PR: LSI vs proxy ≥ p90 (full sample)")
    if test_idx.sum() > 50 and stress_label_p90.loc[test_idx].sum() > 0:
        plot_roc(lsi.loc[test_idx], stress_label_p90.loc[test_idx],
                 cfg.figures_dir / "roc_lsi_test.png",
                 "ROC: LSI vs proxy ≥ p90 (test, ≥ 2024-01-01)")
        plot_pr(lsi.loc[test_idx], stress_label_p90.loc[test_idx],
                cfg.figures_dir / "pr_lsi_test.png",
                "PR: LSI vs proxy ≥ p90 (test, ≥ 2024-01-01)")
    for model in MODEL_NAMES:
        if model not in preds.columns:
            continue
        pred = preds[model]
        plot_roc(
            pred.loc[test_idx], stress_label_p90.loc[test_idx],
            cfg.figures_dir / f"roc_{model}_test.png",
            f"ROC: {model} vs proxy ≥ p90 (test)",
        )
        plot_pr(
            pred.loc[test_idx], stress_label_p90.loc[test_idx],
            cfg.figures_dir / f"pr_{model}_test.png",
            f"PR: {model} vs proxy ≥ p90 (test)",
        )
        plot_calibration(
            pred.loc[test_idx], proxy.loc[test_idx],
            cfg.figures_dir / f"calibration_{model}_test.png",
            f"Calibration: {model} (test)",
        )
    plot_calibration(lsi, proxy, cfg.figures_dir / "calibration_lsi_full.png",
                     "Calibration: weighted-sum LSI (full)")

    # ---------------------- Console summary ---------------------- #
    print()
    print("=" * 78)
    print("RU LIQUIDITY SENTINEL — model evaluation metrics (NSVM)")
    print("=" * 78)
    print()
    print("[1] Regression metrics (proxy stress target)")
    print("-" * 78)
    print(reg_df.round(3).to_string(index=False))
    print()
    print("[2] Classification metrics — predictions as stress detector (target ≥ p90/p95)")
    print("-" * 78)
    print(cls_df.round(3).to_string(index=False))
    print()
    print("[3] Threshold-specific alert metrics on LSI")
    print("-" * 78)
    print(thr_df.round(3).to_string(index=False))
    print()
    print("[4] Temporal stability")
    print("-" * 78)
    for k, v in stab.items():
        print(f"  {k:<32s}  {v: .4f}" if isinstance(v, float) else f"  {k:<32s}  {v}")
    print()
    print("[5] TZ episodes")
    print("-" * 78)
    if not tz_eps.empty:
        print(tz_eps.round(2).to_string(index=False))
    print()
    print("[6] Auto-detected stress episodes (top-K)")
    print("-" * 78)
    if not auto_eps.empty:
        print(auto_eps.round(2).to_string(index=False))
    print()
    print(f"Wrote tables → {cfg.tables_dir}")
    print(f"Wrote figures → {cfg.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
