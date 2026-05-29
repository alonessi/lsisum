"""Centralised configuration: paths, history window, HTTP defaults."""

from __future__ import annotations

import datetime as _dt
import logging
import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

DEFAULT_FROM_DATE: _dt.date = _dt.date(2014, 1, 1)
DEFAULT_TO_DATE: _dt.date = _dt.date.today()

USER_AGENT: str = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HTTP_TIMEOUT: float = 60.0
HTTP_RETRIES: int = 5
HTTP_BACKOFF: float = 1.5


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a compact format suitable for CLI usage."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_dirs() -> None:
    """Create raw / processed data directories if missing."""
    for d in (RAW_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def env_default(name: str, fallback: str) -> str:
    return os.environ.get(name, fallback)
