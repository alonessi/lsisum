from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ru_liquidity_sentinel.config import PipelineConfig
from ru_liquidity_sentinel.incremental import incremental_update, offline_initialize


def main() -> int:
    ap = argparse.ArgumentParser(description="Manage offline/init and incremental NSVM LSI artifacts")
    ap.add_argument("--init", action="store_true", help="Run phase 1 offline initialization")
    ap.add_argument("--cutoff", default=None, help="Training cutoff date for --init, default from config")
    ap.add_argument("--no-fetch", action="store_true", help="Use existing data/raw CSVs during incremental update")
    args = ap.parse_args()

    cfg = PipelineConfig()
    cfg.ensure_dirs()

    if args.init:
        result = offline_initialize(cfg, cutoff_date=args.cutoff)
        print(
            f"[init] historical rows={result.new_rows}, "
            f"date_max={result.last_available_date.date()}, "
            f"path={result.historical_path}"
        )
        return 0

    result = incremental_update(cfg, fetch=not args.no_fetch)
    print(
        f"[update] new_rows={result.new_rows}, "
        f"date_max={result.last_available_date.date()}, "
        f"mean_nll={result.mean_nll}, threshold={result.threshold}, "
        f"regime_shift={result.regime_shift}, partial_fit={result.partial_fit_ran}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
