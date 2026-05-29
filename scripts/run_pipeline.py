"""CLI entry point: run the full pipeline + write reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running ``python scripts/run_pipeline.py`` from a fresh checkout
# without having to install the package first.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ru_liquidity_sentinel.config import PipelineConfig  # noqa: E402
from ru_liquidity_sentinel.pipeline import run_pipeline  # noqa: E402
from ru_liquidity_sentinel.reporting import write_all_reports  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the RU Liquidity Sentinel pipeline")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=PipelineConfig().data_dir,
        help="Directory containing the parsed CSVs (default: data/raw)",
    )
    ap.add_argument(
        "--no-models",
        action="store_true",
        help="Skip fitting the NSVM aggregator",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    ap.add_argument(
        "--with-eda",
        action="store_true",
        help="Run scripts/eda.py and scripts/eval_metrics.py after the main pipeline",
    )
    args = ap.parse_args()

    fit_models = not args.no_models
    cfg = PipelineConfig(data_dir=args.data_dir, fit_models=fit_models)
    cfg.ensure_dirs()

    if not args.quiet:
        print(f"[run] using data_dir={cfg.data_dir}")
        print(f"[run] artefacts -> {cfg.artifacts_dir}")
        print(f"[run] reports   -> {cfg.reports_dir}")

    run = run_pipeline(cfg)

    if not args.quiet:
        print(
            "[run] LSI quantiles q50/q90/q99 = "
            f"{run.metadata['lsi_quantiles']['q50']:.1f}/"
            f"{run.metadata['lsi_quantiles']['q90']:.1f}/"
            f"{run.metadata['lsi_quantiles']['q99']:.1f}"
        )
        print(f"[run] hold-out MAE={run.holdout.get('mae'):.2f} AUC={run.holdout.get('auc')}"
              f" n={run.holdout.get('n')}")
        if run.models is not None:
            for name, mr in run.models.models.items():
                tm = mr.test_metrics
                def _fmt(v: object, spec: str = ".3f") -> str:
                    return format(v, spec) if isinstance(v, (int, float)) else "n/a"
                print(
                    f"[run] {name:<8s} test MAE={_fmt(tm.get('mae'), '.2f')} "
                    f"R²={_fmt(tm.get('r2'))} ROC-AUC={_fmt(tm.get('roc_auc_p90'))} "
                    f"n={tm.get('n')}"
                )
            lgb = run.models.models.get("nsvm")
            if lgb is not None and lgb.feature_importance is not None:
                top = lgb.feature_importance.head(8)
                print(f"[run] NSVM top-8 feature_importances: {dict(top.round(3))}")
        print(f"[run] auto-detected stress episodes: {run.metadata.get('n_auto_episodes', 0)}")

    paths = write_all_reports(run, cfg)
    if not args.quiet:
        for p in paths:
            print(f"[report] {p}")

    if args.with_eda:
        import subprocess

        eda_script = REPO_ROOT / "scripts" / "eda.py"
        eval_script = REPO_ROOT / "scripts" / "eval_metrics.py"
        env = {"PYTHONPATH": str(SRC)}
        for script in (eda_script, eval_script):
            if not args.quiet:
                print(f"[run] -> {script.name}")
            res = subprocess.run(
                [sys.executable, str(script)],
                env={**__import__("os").environ, **env},
                check=False,
            )
            if res.returncode != 0:
                print(f"[run] {script.name} returned {res.returncode}")
                return res.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
