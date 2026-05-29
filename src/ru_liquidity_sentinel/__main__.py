"""Allow ``python -m ru_liquidity_sentinel`` to run the pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_pipeline.py"

if __name__ == "__main__":
    sys.argv[0] = str(SCRIPT)
    exec(SCRIPT.read_text(), {"__name__": "__main__", "__file__": str(SCRIPT)})


def main() -> int:  # pragma: no cover - CLI shim
    sys.argv[0] = str(SCRIPT)
    exec(SCRIPT.read_text(), {"__name__": "__main__", "__file__": str(SCRIPT)})
    return 0
