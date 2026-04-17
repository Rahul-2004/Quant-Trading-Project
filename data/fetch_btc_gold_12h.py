#!/usr/bin/env python3
"""
Fetch BTC and Gold prices from 2015 and align them on the same 12-hour timeline.

Data sources (Yahoo Finance chart API):
- BTC: BTC-USD
- Gold: GC=F (COMEX Gold Futures)

Note:
- Long-range historical data from Yahoo is daily for these symbols.
- This script maps daily closes onto a 12h UTC grid by carry-forward alignment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Iterable
from urllib.parse import urlencode

import polars as pl

YAHOO_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"


def parse_date(date_text: str) -> datetime:
    return datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _to_utc_midnight(epoch_seconds: int) -> datetime:
    dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)


def fetch_json_with_curl(url: str, max_retries: int = 5) -> dict:
    user_agent = "Mozilla/5.0"

    for attempt in range(max_retries):
        result = subprocess.run(
            ["curl", "-s", "-A", user_agent, url],
            capture_output=True,
            text=True,
            check=False,
        )
        body = result.stdout.strip()

        if result.returncode == 0 and body:
            if "Too Many Requests" in body:
                if attempt < max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
            try:
                payload = json.loads(body)
                return payload
            except json.JSONDecodeError:
                pass

        if attempt < max_retries - 1:
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Failed to fetch JSON with curl after retries: {url}")


def fetch_yahoo_daily_close(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    output_col: str,
) -> pl.DataFrame:
    params = {
        "period1": int(start_date.timestamp()),
        # Yahoo period2 is exclusive.
        "period2": int((end_date + timedelta(days=1)).timestamp()),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }

    url = f"{YAHOO_CHART_URL.format(symbol=symbol)}?{urlencode(params)}"
    payload = fetch_json_with_curl(url=url, max_retries=5)

    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise RuntimeError(f"Yahoo API error for {symbol}: {error}")

    result = (chart.get("result") or [None])[0]
    if not result:
        raise RuntimeError(f"No result returned for {symbol}")

    timestamps: Iterable[int] = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    closes: Iterable[float | None] = quote.get("close") or []

    rows: list[tuple[datetime, float]] = []
    for ts, close in zip(timestamps, closes):
        if close is None:
            continue
        rows.append((_to_utc_midnight(int(ts)), float(close)))

    if not rows:
        raise RuntimeError(f"No daily close rows parsed for {symbol}")

    df = pl.DataFrame(rows, schema=["timestamp", output_col], orient="row")
    # Keep one row per day.
    return (
        df.sort("timestamp")
        .group_by("timestamp")
        .agg(pl.col(output_col).last())
        .sort("timestamp")
    )


def build_12h_grid(start_date: datetime, end_date: datetime) -> pl.DataFrame:
    # Include two bars per day: 00:00 and 12:00 UTC.
    grid_end = end_date.replace(hour=12, minute=0, second=0, microsecond=0)
    timestamps = pl.datetime_range(
        start=start_date.replace(hour=0, minute=0, second=0, microsecond=0),
        end=grid_end,
        interval="12h",
        eager=True,
    )
    return pl.DataFrame({"timestamp": timestamps}).sort("timestamp")


def align_on_same_12h_timestamps(
    btc_daily: pl.DataFrame,
    gold_daily: pl.DataFrame,
    grid_12h: pl.DataFrame,
) -> pl.DataFrame:
    merged = (
        grid_12h.join_asof(btc_daily.sort("timestamp"), on="timestamp", strategy="backward")
        .join_asof(gold_daily.sort("timestamp"), on="timestamp", strategy="backward")
        .drop_nulls(["btc_close", "gold_close"])
        .sort("timestamp")
    )

    return merged.select(["timestamp", "btc_close", "gold_close"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch BTC and Gold prices from Yahoo and align on 12h timestamps."
    )
    parser.add_argument("--start", default="2015-01-01", help="Start date, format YYYY-MM-DD")
    parser.add_argument(
        "--end",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="End date, format YYYY-MM-DD",
    )
    parser.add_argument(
        "--out",
        default="data/btc_gold_12h_aligned.csv",
        help="Output CSV path for aligned 12h dataset",
    )
    args = parser.parse_args()

    start_date = parse_date(args.start)
    end_date = parse_date(args.end)
    if end_date < start_date:
        raise ValueError("--end must be on or after --start")

    btc_daily = fetch_yahoo_daily_close(
        symbol="BTC-USD",
        start_date=start_date,
        end_date=end_date,
        output_col="btc_close",
    )
    gold_daily = fetch_yahoo_daily_close(
        symbol="GC=F",
        start_date=start_date,
        end_date=end_date,
        output_col="gold_close",
    )
    grid_12h = build_12h_grid(start_date=start_date, end_date=end_date)

    aligned = align_on_same_12h_timestamps(
        btc_daily=btc_daily,
        gold_daily=gold_daily,
        grid_12h=grid_12h,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.write_csv(out_path)

    print(f"Wrote {aligned.height} aligned rows to {out_path}")
    print(f"Range: {aligned['timestamp'].min()} -> {aligned['timestamp'].max()}")


if __name__ == "__main__":
    main()
