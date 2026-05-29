"""M4 — Налоговый календарь ФНС (open data XML) + посуточные флаги"""

from __future__ import annotations

import datetime as _dt
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from .. import config, http
from ._common import write_csv

logger = logging.getLogger(__name__)

OPENDATA_INDEX_URL = "https://www.nalog.gov.ru/opendata/7707329152-kalendar/"

_MONTHS_EN = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}
_STRONG_RE = re.compile(r"<strong[^>]*>(.*?)</strong>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NBSP_RE = re.compile(r"&nbsp;|\xa0")
_WS_RE = re.compile(r"\s+")


def _strip_html(html_text: str) -> str:
    text = _HTML_TAG_RE.sub(" ", html_text)
    text = _NBSP_RE.sub(" ", text)
    text = text.replace("&laquo;", "«").replace("&raquo;", "»")
    text = text.replace("&quot;", '"').replace("&amp;", "&")
    text = _WS_RE.sub(" ", text)
    return text.strip()

def _extract_title(html_text: str) -> str:
    m = _STRONG_RE.search(html_text)
    if m:
        title = _strip_html(m.group(1))
        if title.endswith(":"):
            title = title[:-1].strip()
        if title:
            return title
    flat = _strip_html(html_text)
    return flat[:150]


def discover_xml_urls(*, out_raw: Path = config.RAW_DIR) -> list[str]:
    """Crawl the FNS open-data page and return absolute XML URLs"""
    resp = http.http_get(OPENDATA_INDEX_URL)
    raw = out_raw / "fns" / "opendata_index.html"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(resp.content)
    soup = BeautifulSoup(resp.content, "lxml")
    urls: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".xml" not in href.lower():
            continue
        if "kalendar" not in href.lower() or "data-" not in href.lower():
            continue
        if href.startswith("//"):
            href = "https:" + href
        elif href.startswith("/"):
            href = "https://www.nalog.gov.ru" + href
        urls.add(href)
    out = sorted(urls)
    logger.info("FNS opendata: discovered %d XML snapshots", len(out))
    return out


def fetch_xml_files(
    urls: list[str], *, out_raw: Path = config.RAW_DIR
) -> list[Path]:
    target_dir = out_raw / "fns" / "opendata_xml"
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for url in urls:
        name = url.rsplit("/", 1)[-1]
        dest = target_dir / name
        if not dest.exists():
            try:
                http.download_to_file(url, dest)
            except Exception as exc:
                logger.warning("XML download failed %s: %s", url, exc)
                continue
        paths.append(dest)
    return paths

def _try_parse_xml_bytes(content: bytes) -> ET.Element | None:
    try:
        return ET.fromstring(content)
    except ET.ParseError:
        pass

    end = content.find(b"</calendar>")
    truncated = (
        content[: end + len(b"</calendar>")] if end != -1 else content
    )
    if truncated != content:
        try:
            return ET.fromstring(truncated)
        except ET.ParseError:
            pass
    try:
        text = truncated.decode("utf-8")
    except UnicodeDecodeError:
        return None
    text = re.sub(
        r"<\?xml[^>]*\?>",
        '<?xml version="1.0" encoding="utf-8"?>',
        text,
        count=1,
    )
    try:
        return ET.fromstring(text)
    except ET.ParseError:
        return None


def _parse_xml(path: Path) -> list[dict[str, object]]:
    """Parse one opendata XML and return a list of event dicts."""
    with path.open("rb") as f:
        content = f.read()
    root = _try_parse_xml_bytes(content)
    if root is None:
        logger.warning("XML parse failed %s", path.name)
        return []

    year_el = root.find(".//year")
    if year_el is None:
        return []
    year_str = year_el.get("index", "")
    try:
        year = int(year_str)
    except ValueError:
        return []

    events: list[dict[str, object]] = []
    for month_el in root.findall(".//month"):
        month_name = month_el.get("name", "").lower()
        month_num = _MONTHS_EN.get(month_name)
        if month_num is None:
            continue
        for day_el in month_el.findall("day"):
            if day_el.get("type") != "event":
                continue
            day_num_raw = day_el.get("num", "")
            try:
                day_num = int(day_num_raw)
            except ValueError:
                continue
            html_payload = day_el.text or ""
            if not html_payload.strip():
                continue
            try:
                date = _dt.date(year, month_num, day_num)
            except ValueError:
                continue
            title = _extract_title(html_payload)
            message = _strip_html(html_payload)
            if not message:
                continue
            events.append(
                {
                    "date": date.isoformat(),
                    "title": title,
                    "message": message,
                    "source": path.name,
                }
            )
    return events


def parse_xml_files(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for p in paths:
        rows.extend(_parse_xml(p))
    if not rows:
        return pd.DataFrame(columns=["date", "title", "message", "source"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    # Newer snapshots win on conflict — keep last by source name length tie
    df = (
        df.sort_values(["date", "title", "source"])
        .drop_duplicates(subset=["date", "message"], keep="last")
        .reset_index(drop=True)
    )
    return df


def build_daily_flags(
    events_df: pd.DataFrame,
    date_from: _dt.date,
    date_to: _dt.date,
) -> pd.DataFrame:
    """Daily indicator series: tax peaks, end-of-month / quarter flags."""
    idx = pd.date_range(start=date_from, end=date_to, freq="D")
    out = pd.DataFrame({"date": idx})
    out["dom"] = out["date"].dt.day
    out["month"] = out["date"].dt.month
    out["dow"] = out["date"].dt.dayofweek

    out["flag_tax_peak_15"] = ((out["dom"] >= 13) & (out["dom"] <= 16)).astype(int)
    out["flag_tax_peak_20"] = ((out["dom"] >= 18) & (out["dom"] <= 22)).astype(int)
    out["flag_tax_peak_25"] = ((out["dom"] >= 23) & (out["dom"] <= 25)).astype(int)
    out["flag_tax_peak_28"] = ((out["dom"] >= 25) & (out["dom"] <= 28)).astype(int)
    eom = pd.Series(idx).dt.is_month_end.values
    out["flag_end_of_month"] = (eom | (out["dom"] >= 28)).astype(int)
    out["flag_quarter_end"] = (
        (out["month"].isin([3, 6, 9, 12])) & (eom | (out["dom"] >= 28))
    ).astype(int)
    out["flag_year_end"] = ((out["month"] == 12) & (out["dom"] >= 25)).astype(int)

    if not events_df.empty:
        per_day = (
            events_df.assign(date=events_df["date"].dt.normalize())
            .groupby("date")
            .agg(
                events_count=("title", "size"),
                titles=(
                    "title",
                    lambda s: "; ".join(sorted(set(s)))[:500],
                ),
            )
            .reset_index()
        )
        out = out.merge(per_day, how="left", on="date")
        out["events_count"] = out["events_count"].fillna(0).astype(int)
        out["titles"] = out["titles"].fillna("")
        out["flag_has_event"] = (out["events_count"] > 0).astype(int)
    else:
        out["events_count"] = 0
        out["titles"] = ""
        out["flag_has_event"] = 0
    return out


def fetch_all(
    date_from: _dt.date = config.DEFAULT_FROM_DATE,
    date_to: _dt.date = config.DEFAULT_TO_DATE,
    *,
    out_raw: Path = config.RAW_DIR,
    out_processed: Path = config.PROCESSED_DIR,
    extra_xml_urls: list[str] | None = None,
) -> dict[str, Path]:
    """Download XML snapshots, build calendar + daily flags CSVs."""
    config.ensure_dirs()

    urls = discover_xml_urls(out_raw=out_raw)
    if extra_xml_urls:
        urls = sorted(set(urls) | set(extra_xml_urls))
    paths = fetch_xml_files(urls, out_raw=out_raw)
    events_df = parse_xml_files(paths)

    if not events_df.empty:
        mask = (
            (events_df["date"] >= pd.Timestamp(date_from))
            & (events_df["date"] <= pd.Timestamp(date_to))
        )
        events_df = events_df.loc[mask].reset_index(drop=True)

    events_csv = out_processed / "m4_tax_calendar.csv"
    write_csv(events_df, events_csv)
    logger.info(
        "M4 tax calendar: %d events -> %s", len(events_df), events_csv
    )

    flags_df = build_daily_flags(events_df, date_from, date_to)
    flags_csv = out_processed / "m4_tax_flags_daily.csv"
    write_csv(flags_df, flags_csv)
    logger.info(
        "M4 tax flags: %d daily rows -> %s", len(flags_df), flags_csv
    )
    return {"tax_calendar": events_csv, "tax_flags_daily": flags_csv}
