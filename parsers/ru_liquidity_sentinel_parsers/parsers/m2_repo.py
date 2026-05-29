"""M2 — Аукционы репо ЦБ + ключевая ставка + кросс-чек из баланса ликвидности.

Sources:
- ЦБ репо: https://www.cbr.ru/hd_base/repo/
- ЦБ ключевая ставка: https://www.cbr.ru/hd_base/keyrate/
- ЦБ баланс ликвидности (кросс-чек): https://www.cbr.ru/hd_base/bliquidity/
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from .. import config, http
from ._common import parse_ru_date, parse_ru_number, write_csv

logger = logging.getLogger(__name__)

REPO_URL = "https://www.cbr.ru/hd_base/repo/"
KEYRATE_URL = "https://www.cbr.ru/hd_base/keyrate/"
BLIQUIDITY_URL = "https://www.cbr.ru/hd_base/bliquidity/"


def _read_history_table(html: bytes | str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="data")
    if table is None:
        raise ValueError("CBR page: table not found")
    headers: list[str] | None = None
    rows = []
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
    return pd.DataFrame(rows)


def fetch_repo_history(
    date_from: _dt.date,
    date_to: _dt.date,
    *,
    out_raw: Path = config.RAW_DIR,
) -> pd.DataFrame:
    params = {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From": date_from.strftime("%d.%m.%Y"),
        "UniDbQuery.To": date_to.strftime("%d.%m.%Y"),
    }
    resp = http.http_get(REPO_URL, params=params)
    raw_path = out_raw / "cbr" / f"repo_history_{date_from:%Y%m%d}_{date_to:%Y%m%d}.html"
    raw_path.write_bytes(resp.content)

    df = _read_history_table(resp.content)
    if df.empty:
        return df

    rename = {
        "Тип аукциона": "auction_type",
        "Срок, дни": "term_days",
        "Дата": "date",
        "Общий объем заключенных сделок, млн руб.": "allotment_mlnrub",
        "Средневзвешенная ставка, % годовых": "weighted_avg_rate_pct",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["date"] = df["date"].map(parse_ru_date)
    for c in ("term_days", "allotment_mlnrub", "weighted_avg_rate_pct"):
        if c in df.columns:
            df[c] = df[c].map(parse_ru_number)
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def fetch_bliq_m2_cols(
    date_from: _dt.date,
    date_to: _dt.date,
) -> pd.DataFrame:
    """Вытягиваем col 5 и col 8 из таблицы bliquidity для M2."""
    params = {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From": date_from.strftime("%d.%m.%Y"),
        "UniDbQuery.To": date_to.strftime("%d.%m.%Y"),
    }
    resp = http.http_get(BLIQUIDITY_URL, params=params)
    soup = BeautifulSoup(resp.content, "lxml")
    table = soup.find("table", class_="data")
    if not table:
        return pd.DataFrame()

    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(strip=True) for c in tr.find_all(["td"])]
        if len(cells) >= 15: # Всего в таблице 15 колонок
            rows.append({
                "date": cells[0],
                "bliq_repo_auction_vol": cells[4],    # col 5
                "bliq_standing_loans_vol": cells[7],  # col 8
            })

    df = pd.DataFrame(rows)
    df["date"] = df["date"].map(parse_ru_date)
    df["bliq_repo_auction_vol"] = df["bliq_repo_auction_vol"].map(parse_ru_number)
    df["bliq_standing_loans_vol"] = df["bliq_standing_loans_vol"].map(parse_ru_number)
    return df.dropna(subset=["date"])


def fetch_keyrate(date_from: _dt.date, date_to: _dt.date) -> pd.DataFrame:
    params = {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From": date_from.strftime("%d.%m.%Y"),
        "UniDbQuery.To": date_to.strftime("%d.%m.%Y"),
    }
    resp = http.http_get(KEYRATE_URL, params=params)
    df = _read_history_table(resp.content)
    if df.empty: return df
    df = df.rename(columns={"Дата": "date", "Ставка": "key_rate_pct"})
    df["date"] = df["date"].map(parse_ru_date)
    df["key_rate_pct"] = df["key_rate_pct"].map(parse_ru_number)
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def fetch_all(
    date_from: _dt.date = config.DEFAULT_FROM_DATE,
    date_to: _dt.date = config.DEFAULT_TO_DATE,
    *,
    detailed: bool = False,
    out_raw: Path = config.RAW_DIR,
    out_processed: Path = config.PROCESSED_DIR,
) -> dict[str, Path]:
    config.ensure_dirs()

    # 1. Тянем историю аукционов
    repo_df = fetch_repo_history(date_from, date_to, out_raw=out_raw)

    # 2. Тянем 2 колонки из баланса ликвидности и мерджим
    bliq_df = fetch_bliq_m2_cols(date_from, date_to)
    if not repo_df.empty and not bliq_df.empty:
        repo_df = pd.merge(repo_df, bliq_df, on="date", how="left")

    repo_csv = out_processed / "m2_repo_auctions.csv"
    write_csv(repo_df, repo_csv)

    # 3. Ключевая ставка
    key_df = fetch_keyrate(date_from, date_to)
    key_csv = out_processed / "m2_keyrate.csv"
    write_csv(key_df, key_csv)

    return {"repo_history": repo_csv, "keyrate": key_csv}