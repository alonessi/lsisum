from __future__ import annotations

from .config import PipelineConfig
from .incremental import incremental_update, offline_initialize


def main() -> int:
    """Run the production incremental NSVM workflow."""

    cfg = PipelineConfig()
    cfg.ensure_dirs()
    result = incremental_update(cfg, fetch=False)
    if result.new_rows == 0 and result.last_saved_date is None:
        offline_initialize(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
