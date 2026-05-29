"""M1 — Усреднение обязательных резервов + RUONIA"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from .. import config, http
from ._common import parse_ru_date, parse_ru_number, write_csv

logger = logging.getLogger(__name__)

RRESERVES_XLSX_URL = (
    "https://www.cbr.ru/vfs/hd_base/RReserves/required_reserves_table.xlsx"
)
RUONIA_DYNAMICS_URL = "https://www.cbr.ru/hd_base/ruonia/dynamics/"

_RESERVES_COLS: tuple[tuple[str, str], ...] = (
    ("period_start", "первый день"),
    ("actual_balances_blnrub", "фактические среднедневные"),
    ("required_avg_blnrub", "подлежащие усреднению"),
    ("required_account_blnrub", "счетах для их учета"),
    ("banks_with_avg_right", "пользующихся правом"),
    ("banks_active", "действующих"),
    ("avg_period_anchor", "период усреднения обязательных резервов"),
    ("avg_period_calendar_days", "число календарных дней"),
    ("reporting_period", "отчетный период"),
    ("regulation_period", "период регулирования"),
)


def _match_column(df_cols: list[str], needle: str) -> str | None:
    needle_low = needle.lower()
    for c in df_cols:
        if needle_low in str(c).lower():
            return c
    return None

def download_rreserves_xlsx(out_dir: Path = config.RAW_DIR) -> Path:
    """Download the CBR RReserves XLSX into ``out_dir``"""
    dest = out_dir / "cbr" / "required_reserves_table.xlsx"
    return http.download_to_file(RRESERVES_XLSX_URL, dest)

def parse_rreserves(xlsx_path: Path) -> pd.DataFrame:
    """Read the 'Обязательные резервы' sheet and normalise column names"""
    df = pd.read_excel(xlsx_path, sheet_name="Обязательные резервы", header=2)
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
    rename: dict[str, str] = {}
    available = list(df.columns)
    for canonical, needle in _RESERVES_COLS:
        match = _match_column(available, needle)
        if match is not None:
            rename[match] = canonical
            available.remove(match)
    df = df.rename(columns=rename)
    df = df.dropna(subset=["period_start"]).reset_index(drop=True)

    df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce")
    if "avg_period_anchor" in df.columns:
        df["avg_period_anchor"] = pd.to_datetime(
            df["avg_period_anchor"], errors="coerce"
        )
    for c in (
        "actual_balances_blnrub",
        "required_avg_blnrub",
        "required_account_blnrub",
        "banks_with_avg_right",
        "banks_active",
        "avg_period_calendar_days",
    ):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if {
        "actual_balances_blnrub",
        "required_avg_blnrub",
    }.issubset(df.columns):
        df["spread_blnrub"] = (
            df["actual_balances_blnrub"] - df["required_avg_blnrub"]
        )

    keep = [c for c in (
        "period_start",
        "avg_period_anchor",
        "avg_period_calendar_days",
        "actual_balances_blnrub",
        "required_avg_blnrub",
        "required_account_blnrub",
        "spread_blnrub",
        "banks_with_avg_right",
        "banks_active",
        "reporting_period",
        "regulation_period",
    ) if c in df.columns]
    df = df[keep]
    df = df.sort_values("period_start").reset_index(drop=True)
    return df


def fetch_ruonia(
    date_from: _dt.date,
    date_to: _dt.date,
    *,
    out_raw: Path = config.RAW_DIR,
) -> pd.DataFrame:
    """Fetch RUONIA dynamics for a date range; saves raw HTML to ``out_raw``"""
    params = {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From": date_from.strftime("%d.%m.%Y"),
        "UniDbQuery.To": date_to.strftime("%d.%m.%Y"),
    }
    resp = http.http_get(RUONIA_DYNAMICS_URL, params=params)
    raw_path = out_raw / "cbr" / (
        f"ruonia_dynamics_{date_from:%Y%m%d}_{date_to:%Y%m%d}.html"
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(resp.content)

    return parse_ruonia_html(resp.content)


def parse_ruonia_html(html: bytes | str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="data") or soup.find("table")
    if table is None:
        raise ValueError("RUONIA dynamics: table not found in HTML")

    rows = []
    headers: list[str] | None = None
    for tr in table.find_all("tr"):
        cells = [
            c.get_text(" ", strip=True).replace("\u00a0", " ")
            for c in tr.find_all(["th", "td"])
        ]
        if not cells:
            continue
        if headers is None:
            headers = cells
            continue
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells, strict=False)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    rename = {
        "Дата ставки": "date",
        "Ставка RUONIA, % годовых": "ruonia_rate_pct",
        "Объем сделок RUONIA, млрд руб.": "volume_blnrub",
        "Количество сделок, ед.": "deals_count",
        "Количество участников RUONIA, совершавших сделки в данный день, ед.": (
            "participants_count"
        ),
        "Минимальная процентная ставка, % годовых": "min_rate_pct",
        "25-й процентиль по процентным ставкам, % годовых": "p25_rate_pct",
        "75-й процентиль по процентным ставкам, % годовых": "p75_rate_pct",
        "Максимальная процентная ставка, % годовых": "max_rate_pct",
        "Статус расчета": "status",
        "Дата публикации": "publish_date",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["date"] = df["date"].map(parse_ru_date)
    if "publish_date" in df.columns:
        df["publish_date"] = df["publish_date"].map(parse_ru_date)
    for c in (
        "ruonia_rate_pct",
        "volume_blnrub",
        "deals_count",
        "participants_count",
        "min_rate_pct",
        "p25_rate_pct",
        "p75_rate_pct",
        "max_rate_pct",
    ):
        if c in df.columns:
            df[c] = df[c].map(parse_ru_number)

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def fetch_all(
    date_from: _dt.date = config.DEFAULT_FROM_DATE,
    date_to: _dt.date = config.DEFAULT_TO_DATE,
    *,
    out_raw: Path = config.RAW_DIR,
    out_processed: Path = config.PROCESSED_DIR,
) -> dict[str, Path]:
    """High-level entrypoint: download + parse + save CSVs."""
    config.ensure_dirs()

    rr_xlsx = download_rreserves_xlsx(out_raw)
    rr_df = parse_rreserves(rr_xlsx)
    rr_df = rr_df[
        (rr_df["period_start"] >= pd.Timestamp(date_from))
        & (rr_df["period_start"] <= pd.Timestamp(date_to))
    ].reset_index(drop=True)
    rr_csv = out_processed / "m1_rreserves.csv"
    write_csv(rr_df, rr_csv)
    logger.info("M1 RReserves: %d rows -> %s", len(rr_df), rr_csv)

    ruonia_df = fetch_ruonia(date_from, date_to, out_raw=out_raw)
    ruonia_csv = out_processed / "m1_ruonia.csv"
    write_csv(ruonia_df, ruonia_csv)
    logger.info("M1 RUONIA: %d rows -> %s", len(ruonia_df), ruonia_csv)

    return {"rreserves": rr_csv, "ruonia": ruonia_csv}
