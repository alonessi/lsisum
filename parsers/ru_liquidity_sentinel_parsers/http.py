"""HTTP helpers: retrying session, raw-file downloader."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from . import config

logger = logging.getLogger(__name__)

_RETRYABLE_EXC = (
    requests.ConnectionError,
    requests.Timeout,
    requests.HTTPError,
)


def make_session(verify: bool = True) -> requests.Session:
    """Build a `requests.Session` preconfigured with realistic headers."""
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": config.USER_AGENT,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        }
    )
    s.verify = verify
    return s


@retry(
    reraise=True,
    retry=retry_if_exception_type(_RETRYABLE_EXC),
    stop=stop_after_attempt(config.HTTP_RETRIES),
    wait=wait_exponential(multiplier=config.HTTP_BACKOFF, min=1, max=30),
)
def http_get(
    url: str,
    *,
    session: requests.Session | None = None,
    params: dict[str, Any] | None = None,
    timeout: float | None = None,
    verify: bool | None = None,
) -> requests.Response:
    """GET with retries on connection/timeout/5xx errors."""
    sess = session if session is not None else make_session()
    kwargs: dict[str, Any] = {"timeout": timeout or config.HTTP_TIMEOUT}
    if params is not None:
        kwargs["params"] = params
    if verify is not None:
        kwargs["verify"] = verify
    logger.debug("GET %s params=%s", url, params)
    resp = sess.get(url, **kwargs)
    if resp.status_code >= 500:
        resp.raise_for_status()
    return resp


def download_to_file(
    url: str,
    dest: Path,
    *,
    session: requests.Session | None = None,
    params: dict[str, Any] | None = None,
    verify: bool | None = None,
) -> Path:
    """Stream a URL into ``dest`` (parent dirs auto-created)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = http_get(url, session=session, params=params, verify=verify)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    logger.info("downloaded %s -> %s (%d bytes)", url, dest, len(resp.content))
    return dest
