"""M5 — Дефицит/профицит ликвидности банковского сектора + остатки ФК.

Sources:
- ЦБ Дефицит/профицит ликвидности банковского сектора (с 2014 г.):
    https://www.cbr.ru/hd_base/bliquidity/
- ЦБ СОРС - сведения о привлечённых средствах (XLSX):
    https://www.cbr.ru/vfs/statistics/BankSector/Borrowings/02_01_Funds_all.xlsx
- Roskazna - размещение средств ЕКС на банковских депозитах:
    https://roskazna.gov.ru/...

Бliquidity HTML has a multi-row header (4 levels deep). We hard-code
the column mapping after the first 5 header rows.
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from .. import config, http
from ._common import parse_ru_date, parse_ru_number, write_csv

logger = logging.getLogger(__name__)

BLIQUIDITY_URL = "https://www.cbr.ru/hd_base/bliquidity/"
SORS_FUNDS_ALL_URL = (
    "https://www.cbr.ru/vfs/statistics/BankSector/Borrowings/02_01_Funds_all.xlsx"
)
ROSKAZNA_DEPOSITS_URL = (
    "https://roskazna.gov.ru/finansovye-operacii/"
    "razmeshchenie-sredstv-edinogo-kaznachejskogo-scheta/"
    "razmeshchenie-sredstv-edinogo-kaznachejskogo-scheta-na-bankovskih-depozitah"
)


_BLIQ_HEADERS_5_LEVEL = [
    "date",
    "deficit_total_blnrub",
    "deficit_excl_corr_blnrub",
    "cb_claims_total_blnrub",
    "cb_claims_repo_swap_auction_blnrub",
    "cb_claims_secured_loans_auction_blnrub",
    "cb_claims_repo_swap_standing_blnrub",
    "cb_claims_secured_loans_standing_blnrub",
    "cb_liabilities_total_blnrub",
    "cb_liabilities_deposits_auction_blnrub",
    "cb_liabilities_kobr_blnrub",
    "cb_liabilities_deposits_standing_blnrub",
    "cb_unstandard_blnrub",
    "bank_corraccounts_blnrub",
    "required_avg_blnrub",
]


def fetch_bliquidity(
        date_from: _dt.date,
        date_to: _dt.date,
        *,
        out_raw: Path = config.RAW_DIR,
) -> pd.DataFrame:
    """Daily liquidity deficit/surplus history from ЦБ."""
    params = {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From": date_from.strftime("%d.%m.%Y"),
        "UniDbQuery.To": date_to.strftime("%d.%m.%Y"),
    }
    resp = http.http_get(BLIQUIDITY_URL, params=params)
    raw_path = out_raw / "cbr" / (
        f"bliquidity_{date_from:%Y%m%d}_{date_to:%Y%m%d}.html"
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(resp.content)

    soup = BeautifulSoup(resp.content, "lxml")
    table = soup.find("table", class_="data")
    if table is None:
        raise ValueError("bliquidity: table not found")

    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [
            c.get_text(" ", strip=True).replace("\u00a0", " ")
            for c in tr.find_all(["th", "td"])
        ]
        rows.append(cells)
    data_rows = [r for r in rows if len(r) == len(_BLIQ_HEADERS_5_LEVEL)]
    body = []
    for r in data_rows:
        if not r[0]:
            continue
        if r[0].lower() in {"1", "дата"}:
            continue
        if "=" in r[1] or r[1].lower() in {"1", "2"}:
            continue
        if not _looks_like_date(r[0]):
            continue
        body.append(r)

    df = pd.DataFrame(body, columns=_BLIQ_HEADERS_5_LEVEL)
    df["date"] = df["date"].map(parse_ru_date)
    for c in _BLIQ_HEADERS_5_LEVEL[1:]:
        df[c] = df[c].map(parse_ru_number)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # --- Считаем дельты и флаги (col 14 = bank_corraccounts_blnrub), НИЧЕГО НЕ ДРОПАЕМ ---
    df["delta_1d_blnrub"] = df["bank_corraccounts_blnrub"].diff(1)
    df["delta_5d_blnrub"] = df["bank_corraccounts_blnrub"].diff(5)  # Недельная дельта (5 рабочих дней)
    df["delta_22d_blnrub"] = df["bank_corraccounts_blnrub"].diff(22)

    # Флаги оттока для агрегационного слоя
    df["flag_budget_drain"] = (df["delta_1d_blnrub"] <= -300).astype(int)
    df["flag_budget_drain_strong"] = (df["delta_1d_blnrub"] <= -500).astype(int)

    # Возвращаем полный датафрейм со всеми оригинальными колонками ЦБ + нашими расчетами
    return df


def _looks_like_date(s: str) -> bool:
    s = s.strip()
    return bool(s) and s[:2].isdigit() and "." in s


def fetch_sors_funds_all(out_raw: Path = config.RAW_DIR) -> Path:
    """Download the СОРС funds-all XLSX (wide format, time series)."""
    dest = out_raw / "cbr" / "sors_02_01_Funds_all.xlsx"
    return http.download_to_file(SORS_FUNDS_ALL_URL, dest)


def parse_sors_funds_all(xlsx_path: Path) -> pd.DataFrame:
    """Parse the СОРС wide table into a long DataFrame (date, indicator, value)."""
    out: list[pd.DataFrame] = []
    for sh in pd.ExcelFile(xlsx_path).sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sh, header=None)
        if df.shape[0] < 5:
            continue
        header_row: int | None = None
        for i in range(min(8, df.shape[0])):
            for v in df.iloc[i].dropna().astype(str):
                if "01.01" in v or "01.02" in v:
                    header_row = i
                    break
            if header_row is not None:
                break
        if header_row is None:
            continue

        date_row = df.iloc[header_row]
        body = df.iloc[header_row + 1 :].reset_index(drop=True)
        body.columns = date_row.tolist()
        # First column is the indicator label
        body = body.rename(columns={body.columns[0]: "indicator"})
        body = body.dropna(subset=["indicator"])
        long = body.melt(id_vars=["indicator"], var_name="date", value_name="value_mlnrub")
        long["date"] = pd.to_datetime(long["date"], errors="coerce")
        long["value_mlnrub"] = pd.to_numeric(long["value_mlnrub"], errors="coerce")
        long = long.dropna(subset=["date", "value_mlnrub"])
        long["sheet"] = sh
        out.append(long)
    if not out:
        return pd.DataFrame(
            columns=["date", "indicator", "value_mlnrub", "sheet"]
        )
    return (
        pd.concat(out, ignore_index=True)
        .sort_values(["sheet", "indicator", "date"])
        .reset_index(drop=True)
    )


def fetch_roskazna_index(
    *,
    out_raw: Path = config.RAW_DIR,
    pages: int = 1,
) -> pd.DataFrame:
    """Index of Roskazna ЕКС-deposit operation-day documents (date + URLs).

    Note: roskazna.gov.ru uses a self-signed Минцифры-issued certificate
    that is not in the system trust store — SSL verification is therefore
    disabled here. Document contents are not parsed (.docx/.XML formats).
    """
    sess = http.make_session()
    sess.verify = False
    raw_dir = out_raw / "roskazna"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for page in range(1, pages + 1):
        params = {"page": str(page)} if page > 1 else None
        resp = http.http_get(
            ROSKAZNA_DEPOSITS_URL,
            session=sess,
            params=params,
            verify=False,
        )
        (raw_dir / f"deposits_index_p{page}.html").write_bytes(resp.content)
        soup = BeautifulSoup(resp.content, "lxml")
        table = soup.find("table")
        if table is None:
            break
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if len(cells) < 2:
                continue
            date_text = cells[0].get_text(" ", strip=True)
            if not date_text or date_text.lower().startswith("дата"):
                continue
            urls = []
            for a in cells[1].find_all("a"):
                href = a.get("href", "")
                if href:
                    urls.append(
                        href if href.startswith("http")
                        else "https://roskazna.gov.ru" + href
                    )
            rows.append(
                {
                    "date_raw": date_text,
                    "n_documents": len(urls),
                    "documents": ";".join(urls),
                    "page": page,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = df["date_raw"].apply(_parse_roskazna_date)
        df = df.sort_values("date").reset_index(drop=True)
    return df


_RU_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}


def _parse_roskazna_date(s: str) -> pd.Timestamp | None:
    parts = s.split()
    if len(parts) < 3:
        return None
    try:
        day = int(parts[0])
        mon = _RU_MONTHS[parts[1].lower()]
        year = int(parts[2])
        return pd.Timestamp(year, mon, day)
    except (KeyError, ValueError):
        return None


def fetch_all(
    date_from: _dt.date = config.DEFAULT_FROM_DATE,
    date_to: _dt.date = config.DEFAULT_TO_DATE,
    *,
    skip_sors: bool = False,
    skip_roskazna: bool = False,
    roskazna_pages: int = 1,
    out_raw: Path = config.RAW_DIR,
    out_processed: Path = config.PROCESSED_DIR,
) -> dict[str, Path]:
    config.ensure_dirs()
    out: dict[str, Path] = {}

    bliq_df = fetch_bliquidity(date_from, date_to, out_raw=out_raw)
    bliq_csv = out_processed / "m5_bliquidity.csv"
    write_csv(bliq_df, bliq_csv)
    logger.info("M5 bliquidity: %d rows -> %s", len(bliq_df), bliq_csv)
    out["bliquidity"] = bliq_csv

    if not skip_sors:
        try:
            xlsx = fetch_sors_funds_all(out_raw=out_raw)
            sors_df = parse_sors_funds_all(xlsx)
            sors_csv = out_processed / "m5_sors_funds_all.csv"
            write_csv(sors_df, sors_csv)
            logger.info("M5 СОРС funds_all: %d rows -> %s", len(sors_df), sors_csv)
            out["sors_funds_all"] = sors_csv
        except Exception as exc:
            logger.warning("M5 СОРС fetch failed: %s", exc)

    if not skip_roskazna:
        try:
            rk_df = fetch_roskazna_index(out_raw=out_raw, pages=roskazna_pages)
            rk_csv = out_processed / "m5_roskazna_eks_deposits_index.csv"
            write_csv(rk_df, rk_csv)
            logger.info(
                "M5 Roskazna ЕКС deposits index: %d rows -> %s",
                len(rk_df),
                rk_csv,
            )
            out["roskazna_index"] = rk_csv
        except Exception as exc:
            logger.warning("M5 Roskazna fetch failed: %s", exc)

    return out
