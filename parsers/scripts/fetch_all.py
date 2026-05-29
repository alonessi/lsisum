"""CLI: fetch ALL modules (M1..M5)."""

from __future__ import annotations

import logging

import _cli_helpers  # noqa: F401
import click

from ru_liquidity_sentinel_parsers import config
from ru_liquidity_sentinel_parsers.parsers import (
    m1_rreserves,
    m2_repo,
    m3_ofz,
    m4_tax,
    m5_treasury,
)

logger = logging.getLogger(__name__)


@click.command()
@click.option("--from", "date_from", default="2014-01-01",
              help="Inclusive start date (YYYY-MM-DD).")
@click.option("--to", "date_to", default=None,
              help="Inclusive end date (YYYY-MM-DD); default = today.")
@click.option("--detailed-repo", is_flag=True, default=False)
@click.option("--skip-m1", is_flag=True, default=False)
@click.option("--skip-m2", is_flag=True, default=False)
@click.option("--skip-m3", is_flag=True, default=False)
@click.option("--skip-m4", is_flag=True, default=False)
@click.option("--skip-m5", is_flag=True, default=False)
@click.option("--m4-extra-xml", multiple=True,
              help="Extra XML snapshot URL(s) for M4 (merged with "
                   "auto-discovered set).")
@click.option("--roskazna-pages", default=10, type=int)
@click.option("--out-raw", default=None)
@click.option("--out-processed", default=None)
def main(
    date_from: str,
    date_to: str | None,
    detailed_repo: bool,
    skip_m1: bool,
    skip_m2: bool,
    skip_m3: bool,
    skip_m4: bool,
    skip_m5: bool,
    m4_extra_xml: tuple[str, ...],
    roskazna_pages: int,
    out_raw: str | None,
    out_processed: str | None,
) -> None:
    config.configure_logging()
    df = _cli_helpers.parse_date(date_from)
    dt = _cli_helpers.parse_date(date_to) or config.DEFAULT_TO_DATE
    from pathlib import Path
    raw = config.RAW_DIR if not out_raw else Path(out_raw)
    proc = config.PROCESSED_DIR if not out_processed else Path(out_processed)

    summary: dict[str, dict] = {}
    if not skip_m1:
        summary["M1"] = m1_rreserves.fetch_all(df, dt, out_raw=raw, out_processed=proc)
    if not skip_m2:
        summary["M2"] = m2_repo.fetch_all(
            df, dt, detailed=detailed_repo, out_raw=raw, out_processed=proc
        )
    if not skip_m3:
        summary["M3"] = m3_ofz.fetch_all(df, dt, out_raw=raw, out_processed=proc)
    if not skip_m4:
        summary["M4"] = m4_tax.fetch_all(
            df, dt,
            extra_xml_urls=list(m4_extra_xml),
            out_raw=raw, out_processed=proc,
        )
    if not skip_m5:
        summary["M5"] = m5_treasury.fetch_all(
            df, dt,
            roskazna_pages=roskazna_pages,
            out_raw=raw, out_processed=proc,
        )

    click.echo("\n=== Summary ===")
    for mod, files in summary.items():
        click.echo(f"  {mod}:")
        for k, v in files.items():
            click.echo(f"    {k}: {v}")


if __name__ == "__main__":
    main()
