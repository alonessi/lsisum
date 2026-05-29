"""CLI: fetch M2 data (REPO auctions + key rate)."""

from __future__ import annotations

import _cli_helpers  # noqa: F401
import click

from ru_liquidity_sentinel_parsers import config
from ru_liquidity_sentinel_parsers.parsers import m2_repo


@click.command()
@click.option("--from", "date_from", default="2014-01-01",
              help="Inclusive start date (YYYY-MM-DD).")
@click.option("--to", "date_to", default=None,
              help="Inclusive end date (YYYY-MM-DD); default = today.")
@click.option("--detailed", is_flag=True, default=False,
              help="Also fetch per-auction detail (slow; ~400 requests).")
@click.option("--out-raw", default=None)
@click.option("--out-processed", default=None)
def main(date_from: str, date_to: str | None, detailed: bool,
         out_raw: str | None, out_processed: str | None) -> None:
    config.configure_logging()
    df = _cli_helpers.parse_date(date_from)
    dt = _cli_helpers.parse_date(date_to) or config.DEFAULT_TO_DATE
    raw = config.RAW_DIR if not out_raw else __import__("pathlib").Path(out_raw)
    proc = (
        config.PROCESSED_DIR
        if not out_processed
        else __import__("pathlib").Path(out_processed)
    )
    files = m2_repo.fetch_all(
        date_from=df, date_to=dt,
        detailed=detailed,
        out_raw=raw, out_processed=proc,
    )
    click.echo(f"M2 done. Files: {files}")


if __name__ == "__main__":
    main()
