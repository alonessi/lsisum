"""Static reporting: figures, tables & alerts.

These outputs back the Streamlit dashboard but can also be inspected
directly under ``reports/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .aggregate import all_module_scores, module_columns
from .config import LSI_THRESHOLDS, PipelineConfig
from .pipeline import PipelineRun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_fig(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# LSI core plots
# ---------------------------------------------------------------------------


def plot_lsi(run: PipelineRun, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    lsi = run.lsi["lsi"]
    ax.plot(lsi.index, lsi.values, color="#1f77b4", lw=1.0, label="LSI (no smoothing)")
    ax.axhspan(LSI_THRESHOLDS["yellow_max"], 100, color="#ff4d4d", alpha=0.10)
    ax.axhspan(
        LSI_THRESHOLDS["green_max"],
        LSI_THRESHOLDS["yellow_max"],
        color="#ffcc00",
        alpha=0.10,
    )
    ax.axhspan(0, LSI_THRESHOLDS["green_max"], color="#2ecc71", alpha=0.08)
    ax.axhline(LSI_THRESHOLDS["green_max"], color="#2ecc71", lw=0.5, ls="--")
    ax.axhline(LSI_THRESHOLDS["yellow_max"], color="#ff4d4d", lw=0.5, ls="--")

    for name, start, end in run.config.backtest_episodes:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.20)

    if not run.auto_episodes.empty:
        for _, row in run.auto_episodes.head(15).iterrows():
            ax.axvspan(
                pd.Timestamp(row["start"]),
                pd.Timestamp(row["end"]),
                color="#d62728",
                alpha=0.06,
            )

    ax.set_title("Liquidity Stress Index (raw, no smoothing)")
    ax.set_ylabel("LSI (0–100)")
    ax.set_xlabel("date")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    return _save_fig(fig, path)


def plot_module_scores(run: PipelineRun, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    score_cols = [f"score_{m}" for m in ("M1", "M2", "M3", "M4", "M5")]
    labels = ["M1 reserves", "M2 repo", "M3 OFZ", "M4 tax", "M5 treasury"]
    for col, label in zip(score_cols, labels):
        ax.plot(run.lsi.index, run.lsi[col].values, lw=0.9, label=label)
    ax.set_title("Per-module share of LSI (|Σ SHAP_module| / Σ |SHAP| × 100)")
    ax.set_ylim(0, 100)
    ax.set_xlabel("date")
    ax.legend(loc="upper left")
    return _save_fig(fig, path)


def plot_contributions(run: PipelineRun, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    contrib_cols = [c for c in run.lsi.columns if c.startswith("contrib_")]
    contrib = run.lsi[contrib_cols]
    ax.stackplot(
        contrib.index,
        contrib.T.values,
        labels=[c.replace("contrib_", "") for c in contrib_cols],
        alpha=0.75,
    )
    ax.set_title("Module contributions to LSI (NSVM SHAP attribution)")
    ax.set_xlabel("date")
    ax.set_ylabel("contribution")
    ax.legend(loc="upper left")
    return _save_fig(fig, path)


def plot_episode_zoom(run: PipelineRun, path: Path) -> Path:
    fig, axes = plt.subplots(
        len(run.config.backtest_episodes), 1, figsize=(12, 9), sharey=True
    )
    if len(run.config.backtest_episodes) == 1:
        axes = [axes]
    lsi = run.lsi["lsi"]
    for ax, (name, start, end) in zip(axes, run.config.backtest_episodes):
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        window = lsi.loc[(lsi.index >= s) & (lsi.index <= e)]
        ax.plot(window.index, window.values, color="#d62728", lw=1.5, label="LSI")
        ax.axhline(LSI_THRESHOLDS["green_max"], color="#2ecc71", lw=0.5, ls="--")
        ax.axhline(LSI_THRESHOLDS["yellow_max"], color="#ff4d4d", lw=0.5, ls="--")
        ax.set_title(f"Episode: {name} ({s.date()} → {e.date()})")
        ax.set_ylim(0, 100)
        ax.set_ylabel("LSI")
        ax.legend(loc="upper left")
    return _save_fig(fig, path)


def plot_auto_episodes(run: PipelineRun, path: Path) -> Optional[Path]:
    if run.auto_episodes.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 5))
    lsi = run.lsi["lsi"]
    ax.plot(lsi.index, lsi.values, color="#1f77b4", lw=1.0)
    ax.axhline(LSI_THRESHOLDS["yellow_max"], color="#ff4d4d", lw=0.4, ls="--")
    ax.axhline(LSI_THRESHOLDS["green_max"], color="#2ecc71", lw=0.4, ls="--")

    tz_episodes = {
        (pd.Timestamp(s), pd.Timestamp(e)) for _, s, e in run.config.backtest_episodes
    }
    for _, row in run.auto_episodes.iterrows():
        start = pd.Timestamp(row["start"])
        end = pd.Timestamp(row["end"])
        is_tz = any(s <= start <= e or s <= end <= e for s, e in tz_episodes)
        ax.axvspan(start, end, color="#d62728" if is_tz else "#ff7f0e", alpha=0.20)

    ax.set_title("Auto-detected stress episodes (LSI ≥ p90)")
    ax.set_ylabel("LSI")
    ax.set_xlabel("date")
    ax.set_ylim(0, 100)
    return _save_fig(fig, path)


# ---------------------------------------------------------------------------
# Three-model comparison plots & tables
# ---------------------------------------------------------------------------


_MODEL_COLORS = {
    "nsvm": "#9467bd",
}

def plot_model_predictions(run: PipelineRun, path: Path) -> Optional[Path]:
    if run.models is None:
        return None
    preds = run.models.predictions_frame()
    if preds.empty:
        return None

    available = [m for m in ("nsvm",) if m in preds.columns]
    if not available:
        return None

    train_max = run.models.train_idx.max() if run.models.train_idx is not None else None

    fig, axes = plt.subplots(
        len(available), 1, figsize=(13, 3.0 * len(available)),
        sharex=True, sharey=True,
    )
    if len(available) == 1:
        axes = [axes]

    green = LSI_THRESHOLDS["green_max"]
    yellow = LSI_THRESHOLDS["yellow_max"]

    for ax, name in zip(axes, available):
        s = preds[name].dropna()
        ax.plot(
            s.index,
            s.values,
            color=_MODEL_COLORS.get(name, "grey"),
            lw=0.9,
            alpha=0.95,
        )
        ax.axhline(green, color="#2ecc71", lw=0.5, ls="--", alpha=0.7)
        ax.axhline(yellow, color="#ff4d4d", lw=0.5, ls="--", alpha=0.7)
        if train_max is not None and pd.notna(train_max):
            ax.axvline(train_max, color="grey", lw=0.7, ls="--", alpha=0.7)
        ax.set_title(f"{name} — predicted LSI (full history)")
        ax.set_ylabel("LSI (0–100)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("date")
    fig.suptitle("Aggregator models — per-model LSI predictions", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save_fig(fig, path)


def plot_model_predictions_episode_zoom(
    run: PipelineRun, path: Path
) -> Optional[Path]:
    if run.models is None:
        return None
    preds = run.models.predictions_frame()
    if preds.empty:
        return None

    fig, axes = plt.subplots(
        len(run.config.backtest_episodes), 1, figsize=(12, 10), sharey=True
    )
    if len(run.config.backtest_episodes) == 1:
        axes = [axes]
    for ax, (name, start, end) in zip(axes, run.config.backtest_episodes):
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        for model_name in ("nsvm",):
            if model_name not in preds.columns:
                continue
            window = preds[model_name].loc[s:e].dropna()
            ax.plot(
                window.index,
                window.values,
                color=_MODEL_COLORS.get(model_name, "grey"),
                lw=1.2,
                label=model_name,
            )
        ax.axhline(LSI_THRESHOLDS["green_max"], color="#2ecc71", lw=0.4, ls="--")
        ax.axhline(LSI_THRESHOLDS["yellow_max"], color="#ff4d4d", lw=0.4, ls="--")
        ax.set_title(f"Episode: {name} ({s.date()} → {e.date()})")
        ax.set_ylim(0, 100)
        ax.set_ylabel("LSI (0–100)")
        ax.legend(loc="upper left", ncol=3)
    return _save_fig(fig, path)


def plot_model_metrics_table(run: PipelineRun, path: Path) -> Optional[Path]:
    if run.models is None:
        return None
    df = run.models.metrics_table()
    if df.empty:
        return None
    df = df.copy()
    metric_cols = [c for c in df.columns if c not in ("model", "split")]
    df = df.sort_values(["split", "model"]).reset_index(drop=True)

    fig, ax = plt.subplots(
        figsize=(min(1.4 * (len(metric_cols) + 2), 18), 0.5 * len(df) + 1.4)
    )
    ax.axis("off")
    cell_text = [
        [str(row["model"]), str(row["split"])]
        + [
            "" if pd.isna(row[c]) else f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c])
            for c in metric_cols
        ]
        for _, row in df.iterrows()
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=["model", "split"] + metric_cols,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    ax.set_title("Aggregator model metrics — NSVM (Unsupervised NLL)")
    return _save_fig(fig, path)


def plot_cv_folds(run: PipelineRun, path: Path) -> Optional[Path]:
    if run.models is None:
        return None
    cv = run.models.cv_table()
    if cv.empty:
        # Since we removed TimeSeriesSplit CV for the Unsupervised approach, this will be gracefully skipped.
        return None

    winners = []
    for (model, params), grp in cv.groupby(["model", "params"]):
        winners.append({"model": model, "params": params, "val_mean": grp["val_mae"].mean()})
    pick = (
        pd.DataFrame(winners)
        .sort_values("val_mean")
        .groupby("model", as_index=False)
        .first()
    )

    n_models = pick.shape[0]
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.2))
    if n_models == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, pick.iterrows()):
        sub = cv[(cv["model"] == row["model"]) & (cv["params"] == row["params"])]
        sub = sub.sort_values("fold")
        x = sub["fold"].astype(str) + "\n" + sub["val_start"].str[:7] + "→" + sub["val_end"].str[:7]
        ax.plot(x, sub["train_mae"], "-o", label="train MAE", color=_MODEL_COLORS.get(row["model"], "grey"))
        ax.plot(x, sub["val_mae"], "--s", label="val MAE", color=_MODEL_COLORS.get(row["model"], "grey"), alpha=0.6)
        ax.set_title(f"{row['model']} — walk-forward CV")
        ax.set_ylabel("MAE")
        ax.set_xlabel("fold (validation window)")
        ax.legend()
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle(
        "Time-aware (TimeSeriesSplit) walk-forward validation — chosen config per model",
        fontsize=12,
    )
    return _save_fig(fig, path)


def plot_feature_importance(run: PipelineRun, path: Path) -> Optional[Path]:
    if run.models is None:
        return None
    available = {
        n: m.feature_importance
        for n, m in run.models.models.items()
        if m.feature_importance is not None and not m.feature_importance.empty
    }
    if not available:
        return None

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (name, imp) in zip(axes, available.items()):
        top = imp.sort_values(ascending=True).tail(15)
        ax.barh(top.index, top.values, color=_MODEL_COLORS.get(name, "grey"))
        ax.set_title(f"{name}: top-15 feature importance")
    return _save_fig(fig, path)


# ---------------------------------------------------------------------------
# Noise breakdown
# ---------------------------------------------------------------------------


def plot_noise_breakdown(run: PipelineRun, path: Path) -> Path:
    nb = run.noise_breakdown.copy()
    if nb.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Noise breakdown — no data")
        return _save_fig(fig, path)

    var_part = nb[nb["component"] == "var"].sort_values("contribution", ascending=False)
    cov_part = nb[nb["component"] == "cov"].sort_values("contribution", key=abs, ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(
        var_part["term"],
        var_part["contribution"],
        color="#1f77b4",
    )
    axes[0].set_title("Per-module variance contribution to Var(ΔLSI)")
    axes[0].set_ylabel("variance points")
    for tick in axes[0].get_xticklabels():
        tick.set_rotation(0)

    axes[1].bar(
        cov_part["term"],
        cov_part["contribution"],
        color=["#d62728" if v < 0 else "#2ca02c" for v in cov_part["contribution"]],
    )
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_title("Cross-module covariance contribution (signed)")
    axes[1].set_ylabel("variance points")
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(45)
    fig.suptitle(
        "Noise breakdown — what creates day-to-day jitter in LSI"
        f"   (total Var(ΔLSI) = {nb['contribution'].sum():.2f})",
        y=1.02,
    )
    return _save_fig(fig, path)


# ---------------------------------------------------------------------------
# Correlation matrices
# ---------------------------------------------------------------------------


def _annotate_corr(ax, mat: pd.DataFrame, fmt: str = "{:.2f}") -> None:
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j]
            if pd.isna(v):
                continue
            ax.text(
                j,
                i,
                fmt.format(v),
                ha="center",
                va="center",
                color="black" if abs(v) < 0.6 else "white",
                fontsize=8,
            )


def plot_correlation_matrix(
    df: pd.DataFrame,
    title: str,
    path: Path,
    annotate: bool = True,
) -> Path:
    if df.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_title(f"{title} — too few columns")
        return _save_fig(fig, path)

    corr = df.corr(numeric_only=True)
    n = len(corr)
    fig, ax = plt.subplots(figsize=(min(0.7 * n + 2, 18), min(0.6 * n + 1.5, 16)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.index, fontsize=9)
    if annotate and n <= 24:
        _annotate_corr(ax, corr)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    return _save_fig(fig, path)


def plot_module_correlation_matrices(
    run: PipelineRun, fdir: Path
) -> List[Path]:
    out: List[Path] = []
    for module in ("M1", "M2", "M3", "M4", "M5"):
        cols = module_columns(run.features, module)
        all_cols = cols["mad"] + cols["flag"] + cols["mio"]
        if not all_cols:
            continue
        df = run.features[all_cols].copy()
        path = fdir / f"correlation_{module}.png"
        out.append(
            plot_correlation_matrix(
                df,
                title=f"Correlations within {module} (mad_scores, flags, mio_cusum)",
                path=path,
            )
        )
    return out


def plot_global_correlation_matrix(run: PipelineRun, path: Path) -> Path:
    parts: List[pd.DataFrame] = []
    for module in ("M1", "M2", "M3", "M4", "M5"):
        cols = module_columns(run.features, module)
        block = run.features[cols["mad"] + cols["flag"] + cols["mio"]]
        if not block.empty:
            parts.append(block)

    scores = all_module_scores(run.features)

    df = pd.concat(parts + [scores], axis=1)
    if "lsi" in run.lsi.columns:
        df = df.join(run.lsi["lsi"].rename("lsi"))
    return plot_correlation_matrix(
        df,
        title=(
            "Global correlation matrix — all mad_scores, flags, mio_cusum + module sums + LSI"
        ),
        path=path,
        annotate=False,
    )


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def export_alerts_table(run: PipelineRun, path: Path, top_n: int = 50) -> Path:
    df = run.lsi.copy()
    df = df.sort_values("lsi", ascending=False).head(top_n)
    df.to_csv(path)
    return path


def write_all_reports(run: PipelineRun, cfg: PipelineConfig | None = None) -> List[Path]:
    cfg = cfg or run.config
    cfg.ensure_dirs()
    fdir = cfg.figures_dir
    tdir = cfg.tables_dir

    out: List[Path] = []
    out.append(plot_lsi(run, fdir / "lsi_full.png"))
    out.append(plot_module_scores(run, fdir / "module_scores.png"))
    out.append(plot_contributions(run, fdir / "module_contributions.png"))
    out.append(plot_episode_zoom(run, fdir / "episode_zoom.png"))
    auto = plot_auto_episodes(run, fdir / "auto_episodes.png")
    if auto is not None:
        out.append(auto)

    # Three-model comparison
    mp = plot_model_predictions(run, fdir / "model_comparison.png")
    if mp is not None:
        out.append(mp)
    mz = plot_model_predictions_episode_zoom(run, fdir / "model_comparison_episodes.png")
    if mz is not None:
        out.append(mz)
    mt = plot_model_metrics_table(run, fdir / "model_metrics_table.png")
    if mt is not None:
        out.append(mt)
    fi = plot_feature_importance(run, fdir / "feature_importance.png")
    if fi is not None:
        out.append(fi)
    cvf = plot_cv_folds(run, fdir / "model_cv_folds.png")
    if cvf is not None:
        out.append(cvf)

    # Noise breakdown
    out.append(plot_noise_breakdown(run, fdir / "noise_breakdown.png"))

    # Correlation matrices
    out.extend(plot_module_correlation_matrices(run, fdir))
    out.append(
        plot_global_correlation_matrix(run, fdir / "correlation_global.png")
    )

    out.append(export_alerts_table(run, tdir / "top_alerts.csv"))

    # HERE IS THE FIX: Removed reference to supervised holdout metrics.
    summary_rows = [
        {"metric": "n_days", "value": int(run.features.shape[0])},
        {"metric": "lsi_q50", "value": run.metadata["lsi_quantiles"]["q50"]},
        {"metric": "lsi_q90", "value": run.metadata["lsi_quantiles"]["q90"]},
        {"metric": "lsi_q99", "value": run.metadata["lsi_quantiles"]["q99"]},
        {
            "metric": "lsi_std_day_diff",
            "value": run.metadata["lsi_smoothness"]["std_day_diff"],
        },
        {"metric": "n_auto_episodes", "value": run.metadata["n_auto_episodes"]},
    ]

    if run.models is not None:
        for model_name, m in run.models.models.items():
            for split, metrics in (
                ("train", m.train_metrics),
                ("test", m.test_metrics),
            ):
                for k, v in metrics.items():
                    summary_rows.append(
                        {
                            "metric": f"{model_name}_{split}_{k}",
                            "value": float(v) if isinstance(v, (int, float)) else v,
                        }
                    )

    summary = pd.DataFrame(summary_rows)
    summary_path = tdir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    out.append(summary_path)
    return out


__all__ = [
    "plot_lsi",
    "plot_module_scores",
    "plot_contributions",
    "plot_episode_zoom",
    "plot_auto_episodes",
    "plot_model_predictions",
    "plot_model_predictions_episode_zoom",
    "plot_model_metrics_table",
    "plot_feature_importance",
    "plot_cv_folds",
    "plot_noise_breakdown",
    "plot_correlation_matrix",
    "plot_module_correlation_matrices",
    "plot_global_correlation_matrix",
    "export_alerts_table",
    "write_all_reports",
]