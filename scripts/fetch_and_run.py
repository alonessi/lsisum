"""End-to-end orchestration: fetch fresh data → run pipeline → build reports.

This script chains the parser layer (``parsers/``) with the main
analytics pipeline (``src/ru_liquidity_sentinel/``).  It is the single
entry-point a user needs to refresh the system from scratch.

Steps (each optional via flags):

  1. Fetch raw CSVs from public sources (ЦБ, Минфин, Росказна, ФНС)
     using ``parsers/scripts/fetch_all.py``.  Writes to
     ``parsers/data/processed/``.
  2. Copy the freshly-parsed CSVs to ``data/raw/`` — the location
     ``loaders.RAW_FILES`` expects.
  3. Run the NSVM aggregation pipeline (``run_pipeline``).
  4. Generate the EDA + evaluation reports.

Usage:

    python scripts/fetch_and_run.py                       # everything
    python scripts/fetch_and_run.py --skip-fetch          # use existing CSVs
    python scripts/fetch_and_run.py --skip-reports        # only pipeline
    python scripts/fetch_and_run.py --skip-fetch --skip-reports
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PARSERS_ROOT = REPO_ROOT / "parsers"
PROCESSED = PARSERS_ROOT / "data" / "processed"
RAW = REPO_ROOT / "data" / "raw"


def run_parsers(extra_args: list[str]) -> None:
    """Invoke parsers/scripts/fetch_all.py as a subprocess.

    A subprocess (not a Python import) is used because parsers are a
    separate top-level package (``ru_liquidity_sentinel_parsers``) with
    its own ``data/`` layout that must remain isolated from the main
    package.
    """

    fetch_all = PARSERS_ROOT / "scripts" / "fetch_all.py"
    if not fetch_all.exists():
        raise FileNotFoundError(f"Parser entry-point not found: {fetch_all}")

    env_path = str(PARSERS_ROOT) + ":" + (str(SRC))
    cmd = [sys.executable, str(fetch_all), *extra_args]
    print(f"[fetch] cwd={PARSERS_ROOT}")
    print(f"[fetch] cmd: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(PARSERS_ROOT),
        env={**dict(__import__("os").environ), "PYTHONPATH": env_path},
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Parsers exited with code {result.returncode}.  "
            f"Re-run with `--skip-fetch` to use existing CSVs in {PROCESSED}."
        )


def copy_parsed_csvs() -> None:
    """Copy the freshly-parsed CSVs from parsers/data/processed → data/raw."""

    if not PROCESSED.exists():
        raise FileNotFoundError(
            f"No parsed CSVs found in {PROCESSED}.  "
            "Run parsers first (drop --skip-fetch) or place CSVs manually."
        )
    RAW.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for csv in PROCESSED.glob("*.csv"):
        dst = RAW / csv.name
        shutil.copy2(csv, dst)
        n_copied += 1
    print(f"[copy] {n_copied} CSVs → {RAW}")


def run_pipeline() -> None:
    """Run the NSVM aggregation pipeline end-to-end."""

    from ru_liquidity_sentinel.pipeline import run_pipeline as _run

    print(f"[pipeline] reading from {RAW}")
    run = _run()
    n_days = run.features.shape[0]
    lsi = run.lsi["lsi"]
    print(
        f"[pipeline] OK — {n_days} days, "
        f"LSI median={lsi.median():.1f}, q90={lsi.quantile(0.9):.1f}, "
        f"q99={lsi.quantile(0.99):.1f}"
    )


def run_reports() -> None:
    """Generate EDA + evaluation markdown reports."""

    for script in ("eda.py", "eval_metrics.py"):
        path = REPO_ROOT / "scripts" / script
        if not path.exists():
            print(f"[reports] {script} not found, skipping")
            continue
        print(f"[reports] running {script}")
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(REPO_ROOT),
        )
        if result.returncode != 0:
            print(f"[reports] {script} failed with code {result.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-fetch", action="store_true", help="Don't run parsers")
    ap.add_argument("--skip-pipeline", action="store_true", help="Don't run main pipeline")
    ap.add_argument("--skip-reports", action="store_true", help="Don't generate EDA/eval reports")
    ap.add_argument(
        "parser_args",
        nargs=argparse.REMAINDER,
        help=(
            "Extra args forwarded to parsers/scripts/fetch_all.py "
            "(e.g. `-- --from 2020-01-01 --skip-m4`)"
        ),
    )
    args = ap.parse_args()

    if not args.skip_fetch:
        forwarded = args.parser_args
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        run_parsers(forwarded)
        copy_parsed_csvs()
    else:
        if not RAW.exists() or not any(RAW.glob("*.csv")):
            # No raw CSVs and --skip-fetch given: try to grab them from
            # the parsers' processed folder anyway, since the user
            # might have run the parsers separately.
            copy_parsed_csvs()
        else:
            print(f"[copy] skipping — {RAW} already populated")

    if not args.skip_pipeline:
        run_pipeline()

    if not args.skip_reports:
        run_reports()

    print("\n[done] To open the dashboard:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
