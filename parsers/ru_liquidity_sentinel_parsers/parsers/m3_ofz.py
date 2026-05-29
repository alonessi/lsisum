"""M3 — Размещение ОФЗ (Минфин XLSX)"""

from __future__ import annotations

import datetime as _dt
import logging
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from .. import config, http
from ._common import write_csv

logger = logging.getLogger(__name__)

MINFIN_BASE = "https://minfin.gov.ru"
MINFIN_OFZ_AUCTION_URL = (
    "https://minfin.gov.ru/ru/perfomance/public_debt/internal/operations/ofz/auction"
)
_XLSX_RE = re.compile(
    r"INTERNET_Auction_Results_rus_(?P<year>\d{4})_[^.]+\.(xlsx|xls)",
    re.IGNORECASE,
)
# Map canonical -> list of needles that may appear in the source header
_OFZ_COLS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("auction_date", ("дата аукциона", "дата")),
    ("auction_format", ("формат",)),
    ("issue_code", ("код выпуска", "код  выпуска", "код выпуск")),
    ("paper_type", ("тип бумаги",)),
    ("maturity_date", ("дата погашения",)),
    ("days_to_maturity", ("дней до погашения",)),
    ("offer_volume_mlnrub", ("объем предложения",)),
    ("cutoff_price_pct", ("цена отсечения",)),
    ("weighted_avg_price_pct", ("цена средневзвешенная", "цена средневзве")),
    ("cutoff_yield_pct", ("доходность по цене отсечения",)),
    ("weighted_avg_yield_pct", ("доходность по средневзвешенной", "по средневзве")),
    ("demand_volume_mlnrub", ("совокупный объем спроса",)),
    ("allotment_volume_mlnrub", ("объем размещения",)),
    ("revenue_mlnrub", ("объем выручки",)),
    ("activity_ratio", ("коэффициент активности",)),
    ("placement_ratio", ("коэффициент размещения на аукционе",)),
    ("demand_satisfaction_ratio", (
        "коэффициент удовлетворения спроса",
        "(13 / 12)",
        "(13/12)",
        "(12/11)",
        "(12 / 11)",
    )),
)


def _discover_yearly_xlsx_urls(
    *,
    out_raw: Path = config.RAW_DIR,
    max_pages: int = 6,
) -> dict[int, str]:
    """Crawl the OFZ auction listing pages to find yearly XLSX archive URLs."""
    sess = http.make_session()
    urls: dict[int, str] = {}
    raw_dir = out_raw / "minfin"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for page in range(1, max_pages + 1):
        page_url = MINFIN_OFZ_AUCTION_URL
        params = {"page_66": str(page)} if page > 1 else None
        resp = http.http_get(page_url, session=sess, params=params)
        (raw_dir / f"ofz_auction_index_p{page}.html").write_bytes(resp.content)
        soup = BeautifulSoup(resp.content, "lxml")
        h2 = next(
            (h for h in soup.find_all("h2")
             if "Таблицы по результатам" in h.get_text()),
            None,
        )
        page_added = 0
        if h2 is not None:
            el = h2
            while True:
                el = el.find_next()
                if el is None or (el.name == "h2"):
                    break
                if el.name != "a":
                    continue
                href = el.get("href", "") or ""
                m = _XLSX_RE.search(href)
                if not m:
                    continue
                year = int(m.group("year"))
                full = href if href.startswith("http") else MINFIN_BASE + href
                if year not in urls:
                    urls[year] = full
                    page_added += 1
        if page_added == 0 and page > 1:
            break
    logger.info("Minfin OFZ: discovered %d yearly XLSX URLs", len(urls))
    return urls


def _read_xlsx(path: Path) -> pd.DataFrame:
    """Read a Minfin OFZ XLSX/XLS into a normalised DataFrame."""
    suffix = path.suffix.lower()
    engine = "xlrd" if suffix == ".xls" else "openpyxl"
    df_raw = pd.read_excel(path, sheet_name=0, header=None, engine=engine)
    header_row: int | None = None
    for i in range(min(20, df_raw.shape[0])):
        row_text = " ".join(
            str(x) for x in df_raw.iloc[i].tolist() if pd.notna(x)
        ).lower()
        if "дата аукциона" in row_text:
            header_row = i
            break
        if (
            row_text.startswith("дата")
            and "код" in row_text
            and "тип бумаги" in row_text
        ):
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"OFZ XLSX header row not found in {path}")
    df = pd.read_excel(path, sheet_name=0, header=header_row, engine=engine)
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
    rename: dict[str, str] = {}
    available = list(df.columns)
    for canonical, needles in _OFZ_COLS:
        match: str | None = None
        for c in available:
            cl = c.lower()
            if any(n in cl for n in needles):
                match = c
                break
        if match is not None:
            rename[match] = canonical
            available.remove(match)
    df = df.rename(columns=rename)

    if "auction_date" not in df.columns:
        raise ValueError(f"auction_date column not detected in {path}")
    df["auction_date"] = pd.to_datetime(df["auction_date"], errors="coerce")
    df = df.dropna(subset=["auction_date"]).reset_index(drop=True)
    df = df[df["auction_date"] >= pd.Timestamp("2000-01-01")]

    if "maturity_date" in df.columns:
        df["maturity_date"] = pd.to_datetime(
            df["maturity_date"], errors="coerce"
        )
    numeric_cols = [
        "days_to_maturity",
        "offer_volume_mlnrub",
        "cutoff_price_pct",
        "weighted_avg_price_pct",
        "cutoff_yield_pct",
        "weighted_avg_yield_pct",
        "demand_volume_mlnrub",
        "allotment_volume_mlnrub",
        "revenue_mlnrub",
        "activity_ratio",
        "placement_ratio",
        "demand_satisfaction_ratio",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if {"demand_volume_mlnrub", "allotment_volume_mlnrub"}.issubset(df.columns):
        denom = df["allotment_volume_mlnrub"].where(
            df["allotment_volume_mlnrub"] > 0
        )
        df["cover_ratio"] = df["demand_volume_mlnrub"] / denom
        df["flag_undersubscribed"] = (df["cover_ratio"] < 1.2).astype("Int64")
        df["flag_oversubscribed"] = (df["cover_ratio"] > 2.0).astype("Int64")
    if {"demand_volume_mlnrub", "offer_volume_mlnrub"}.issubset(df.columns):
        denom = df["offer_volume_mlnrub"].where(df["offer_volume_mlnrub"] > 0)
        df["bid_to_offer_ratio"] = df["demand_volume_mlnrub"] / denom

    canonical_order = [c for (c, _n) in _OFZ_COLS]
    extra = (
        "cover_ratio",
        "bid_to_offer_ratio",
        "flag_undersubscribed",
        "flag_oversubscribed",
    )
    keep = [c for c in canonical_order if c in df.columns] + [
        c for c in extra if c in df.columns
    ]
    df = df[keep].reset_index(drop=True)
    return df


def fetch_all(
    date_from: _dt.date = config.DEFAULT_FROM_DATE,
    date_to: _dt.date = config.DEFAULT_TO_DATE,
    *,
    out_raw: Path = config.RAW_DIR,
    out_processed: Path = config.PROCESSED_DIR,
    extra_xlsx_urls: dict[int, str] | None = None,
) -> dict[str, Path]:
    """Discover, download and parse all available yearly Minfin XLSX archives."""
    config.ensure_dirs()
    raw_dir = out_raw / "minfin"
    raw_dir.mkdir(parents=True, exist_ok=True)

    urls = _discover_yearly_xlsx_urls(out_raw=out_raw)
    if extra_xlsx_urls:
        urls.update(extra_xlsx_urls)

    frames: list[pd.DataFrame] = []
    for year in sorted(urls):
        url = urls[year]
        suffix = ".xls" if url.lower().endswith(".xls") else ".xlsx"
        dest = raw_dir / f"INTERNET_Auction_Results_rus_{year}{suffix}"
        try:
            http.download_to_file(url, dest)
            df = _read_xlsx(dest)
            df["source_year"] = year
            frames.append(df)
        except Exception as exc:
            logger.warning("Minfin OFZ %d failed: %s", year, exc)

    if not frames:
        logger.error("Minfin OFZ: no data collected")
        return {}

    full = pd.concat(frames, ignore_index=True)
    full = full[
        (full["auction_date"] >= pd.Timestamp(date_from))
        & (full["auction_date"] <= pd.Timestamp(date_to))
    ]
    full = full.sort_values(
        ["auction_date", "issue_code"]
        if "issue_code" in full.columns else "auction_date"
    ).reset_index(drop=True)

    out_csv = out_processed / "m3_ofz_auctions.csv"
    write_csv(full, out_csv)
    logger.info("M3 OFZ auctions: %d rows -> %s", len(full), out_csv)
    return {"ofz_auctions": out_csv}
