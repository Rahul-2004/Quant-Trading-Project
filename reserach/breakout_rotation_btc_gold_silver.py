#!/usr/bin/env python3
"""Breakout regime switching: BTC <-> Gold/Silver.

Core strategy (Option A neutral handling):
- Bullish: BTC close > highest close of last 20 days (40 bars at 12h)
- Bearish: BTC close < lowest close of last 20 days
- Neutral: hold previous allocation

Allocation:
- Bullish: 100% BTC
- Bearish: 50% Gold + 50% Silver

Also tests benchmarks:
- BTC breakout only (Option A): BTC on breakout, cash on breakdown, hold previous in neutral
- BTC breakout->cash (strict): BTC only when breakout condition true, cash otherwise

Outputs:
- reserach/results/breakout_rotation_results.csv
- reserach/results/breakout_rotation_ranked.csv
- reserach/results/breakout_rotation_weights.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd

BARS_PER_DAY = 2
BARS_PER_YEAR = 365.25 * BARS_PER_DAY
YAHOO_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Breakout BTC<->Gold/Silver rotation test")
    p.add_argument("--data", default="data/btc_gold_12h_aligned.csv")
    p.add_argument("--cache-dir", default="reserach/cache")
    p.add_argument("--out-dir", default="reserach/results")
    p.add_argument("--fee-bps", type=float, default=4.0)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--refresh", action="store_true", help="Refresh cached external Yahoo data")
    return p.parse_args()


def fetch_json_with_curl(url: str, max_retries: int = 7) -> dict:
    for attempt in range(max_retries):
        res = subprocess.run(
            ["curl", "-s", "-A", "Mozilla/5.0", url],
            capture_output=True,
            text=True,
            check=False,
        )
        body = res.stdout.strip()
        if res.returncode == 0 and body:
            if "Too Many Requests" in body or body.startswith("Edge: Too Many Requests"):
                if attempt < max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                if attempt < max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
        if attempt < max_retries - 1:
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Failed to fetch JSON: {url}")


def fetch_yahoo_daily(symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    period1 = int(start_ts.timestamp())
    period2 = int((end_ts + pd.Timedelta(days=1)).timestamp())
    url = (
        YAHOO_CHART_URL.format(symbol=symbol)
        + f"?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
    )
    payload = fetch_json_with_curl(url)

    result = (payload.get("chart", {}).get("result") or [None])[0]
    if result is None:
        raise RuntimeError(f"No Yahoo result for {symbol}")

    ts = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    close = quote.get("close") or []
    volume = quote.get("volume") or [None] * len(close)

    rows = []
    for t, c, v in zip(ts, close, volume):
        if c is None:
            continue
        rows.append(
            {
                "timestamp": pd.to_datetime(int(t), unit="s", utc=True).floor("D"),
                "close": float(c),
                "volume": None if v is None else float(v),
            }
        )

    if not rows:
        raise RuntimeError(f"No parsed rows for {symbol}")

    return pd.DataFrame(rows).sort_values("timestamp").drop_duplicates("timestamp", keep="last")


def load_or_fetch(cache_file: Path, symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, refresh: bool) -> pd.DataFrame:
    if cache_file.exists() and not refresh:
        df = pd.read_csv(cache_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
        return df.sort_values("timestamp").reset_index(drop=True)

    df = fetch_yahoo_daily(symbol, start_ts, end_ts)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    return df.reset_index(drop=True)


def merge_asof(base: pd.DataFrame, feat: pd.DataFrame, col: str) -> pd.Series:
    b = base[["timestamp"]].sort_values("timestamp").reset_index(drop=True)
    f = feat[["timestamp", col]].sort_values("timestamp").reset_index(drop=True)
    out = pd.merge_asof(b, f, on="timestamp", direction="backward")
    return out[col]


def prepare_data(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    start_ts = df["timestamp"].min().floor("D")
    end_ts = df["timestamp"].max().ceil("D")

    cache_dir = Path(args.cache_dir)

    btc_daily = load_or_fetch(cache_dir / "yahoo_btc_daily.csv", "BTC-USD", start_ts, end_ts, args.refresh)
    # Silver futures
    silver_daily = load_or_fetch(cache_dir / "yahoo_silver_daily.csv", "SI=F", start_ts, end_ts, args.refresh)

    btc_daily = btc_daily.rename(columns={"close": "btc_yahoo_close", "volume": "btc_volume"})
    silver_daily = silver_daily.rename(columns={"close": "silver_close", "volume": "silver_volume"})

    df["silver_close"] = merge_asof(df, silver_daily, "silver_close")
    df["btc_volume"] = merge_asof(df, btc_daily, "btc_volume")

    df["silver_close"] = df["silver_close"].ffill().bfill()
    df["btc_volume"] = df["btc_volume"].ffill().bfill()

    for col in ["btc_close", "gold_close", "silver_close"]:
        df[f"{col}_ret"] = df[col].pct_change().fillna(0.0)

    # filters
    df["vol_ma"] = df["btc_volume"].rolling(40).mean()
    df["volume_confirm"] = df["btc_volume"] > df["vol_ma"]

    atr = df["btc_close"].pct_change().abs().rolling(28).mean()
    atr_rel = atr / atr.rolling(40).mean()
    df["vol_expansion"] = (atr_rel > 1.0) & (atr_rel.diff() > 0)

    return df


def breakout_breakdown_flags(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    lookback = 40  # 20 days on 12h bars
    hi = df["btc_close"].shift(1).rolling(lookback).max()
    lo = df["btc_close"].shift(1).rolling(lookback).min()
    breakout = df["btc_close"] > hi
    breakdown = df["btc_close"] < lo
    return breakout.fillna(False), breakdown.fillna(False)


def build_weights_rotation(
    df: pd.DataFrame,
    use_volume_filter: bool = False,
    use_vol_expansion_filter: bool = False,
) -> np.ndarray:
    breakout, breakdown = breakout_breakdown_flags(df)

    if use_volume_filter:
        breakout = breakout & df["volume_confirm"].fillna(False)
    if use_vol_expansion_filter:
        breakout = breakout & df["vol_expansion"].fillna(False)

    # state: 1 risk-on BTC, 0 defensive Gold/Silver
    state = 0
    w = np.zeros((len(df), 4), dtype=float)  # btc, gold, silver, cash

    for i in range(len(df)):
        if breakout.iloc[i]:
            state = 1
        elif breakdown.iloc[i]:
            state = 0

        if state == 1:
            w[i] = [1.0, 0.0, 0.0, 0.0]
        else:
            w[i] = [0.0, 0.5, 0.5, 0.0]

    return w


def build_weights_btc_breakout_option_a(df: pd.DataFrame) -> np.ndarray:
    breakout, breakdown = breakout_breakdown_flags(df)

    state = 0  # 1 BTC, 0 cash
    w = np.zeros((len(df), 4), dtype=float)

    for i in range(len(df)):
        if breakout.iloc[i]:
            state = 1
        elif breakdown.iloc[i]:
            state = 0

        if state == 1:
            w[i] = [1.0, 0.0, 0.0, 0.0]
        else:
            w[i] = [0.0, 0.0, 0.0, 1.0]

    return w


def build_weights_btc_breakout_strict_cash(df: pd.DataFrame) -> np.ndarray:
    breakout, _ = breakout_breakdown_flags(df)
    w = np.zeros((len(df), 4), dtype=float)
    for i in range(len(df)):
        if breakout.iloc[i]:
            w[i] = [1.0, 0.0, 0.0, 0.0]
        else:
            w[i] = [0.0, 0.0, 0.0, 1.0]
    return w


def build_weights_buy_hold_btc(df: pd.DataFrame) -> np.ndarray:
    w = np.zeros((len(df), 4), dtype=float)
    w[:, 0] = 1.0
    return w


def backtest(df: pd.DataFrame, weights: np.ndarray, fee_bps: float, slippage_bps: float) -> Dict[str, float | np.ndarray]:
    asset_ret = np.vstack(
        [
            df["btc_close_ret"].to_numpy(float),
            df["gold_close_ret"].to_numpy(float),
            df["silver_close_ret"].to_numpy(float),
            np.zeros(len(df), dtype=float),
        ]
    ).T

    prev_w = np.vstack([np.zeros((1, 4)), weights[:-1]])

    gross_ret = np.sum(prev_w * asset_ret, axis=1)
    turnover = np.sum(np.abs(weights - prev_w), axis=1)
    costs = turnover * (fee_bps + slippage_bps) / 10000.0
    net_ret = gross_ret - costs

    eq = np.cumprod(1.0 + net_ret)
    eq_gross = np.cumprod(1.0 + gross_ret)

    years = len(df) / BARS_PER_YEAR
    total_return = float(eq[-1] - 1.0)
    cagr = float((eq[-1] ** (1.0 / years) - 1.0) if years > 0 and eq[-1] > 0 else np.nan)

    run_max = np.maximum.accumulate(eq)
    drawdown = eq / run_max - 1.0
    max_dd = float(np.min(drawdown)) if len(drawdown) else 0.0

    sd = float(np.std(net_ret))
    sharpe = float((np.mean(net_ret) / sd) * np.sqrt(BARS_PER_YEAR)) if sd > EPS else 0.0

    return {
        "total_return": total_return,
        "return_pct": total_return * 100.0,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "gross_return": float(eq_gross[-1] - 1.0),
        "net_ret": net_ret,
        "equity": eq,
        "weights": weights,
        "prev_weights": prev_w,
    }


def phase_returns(df: pd.DataFrame, net_ret: np.ndarray) -> Tuple[float, float]:
    # Bull phase proxy: BTC above 200-day MA (400 bars)
    bull = (df["btc_close"] > df["btc_close"].rolling(400).mean()).fillna(False).to_numpy()

    # Drawdown phase: BTC >=20% below running high
    dd = df["btc_close"] / df["btc_close"].cummax() - 1.0
    drawdown_phase = (dd <= -0.20).fillna(False).to_numpy()

    bull_ret = float(np.prod(1.0 + net_ret[bull]) - 1.0) if bull.any() else np.nan
    dd_ret = float(np.prod(1.0 + net_ret[drawdown_phase]) - 1.0) if drawdown_phase.any() else np.nan
    return bull_ret, dd_ret


def compute_oos(df: pd.DataFrame, weights: np.ndarray, fee_bps: float, slippage_bps: float) -> Dict[str, float]:
    split = int(len(df) * 0.70)
    dfo = df.iloc[split:].reset_index(drop=True)
    wo = weights[split:]

    o = backtest(dfo, wo, fee_bps, slippage_bps)
    switches = int(np.sum(np.sum(np.abs(np.diff(wo, axis=0)), axis=1) > EPS))
    years = len(dfo) / BARS_PER_YEAR

    return {
        "oos_return": float(o["total_return"]),
        "oos_return_pct": float(o["return_pct"]),
        "oos_max_drawdown": float(o["max_drawdown"]),
        "oos_sharpe": float(o["sharpe"]),
        "oos_switches": switches,
        "oos_switches_per_year": float(switches / years) if years > 0 else np.nan,
    }


def summarize(name: str, df: pd.DataFrame, bt: Dict[str, float | np.ndarray], fee_bps: float, slippage_bps: float) -> Dict[str, float | str]:
    w = bt["weights"]
    pw = bt["prev_weights"]
    net_ret = bt["net_ret"]

    years = len(df) / BARS_PER_YEAR

    time_btc = float(np.mean(pw[:, 0] > EPS))
    time_def = float(np.mean((pw[:, 1] + pw[:, 2]) > EPS))
    time_cash = float(np.mean(pw[:, 3] > EPS))

    switches = int(np.sum(np.sum(np.abs(np.diff(w, axis=0)), axis=1) > EPS))
    switches_per_year = float(switches / years) if years > 0 else np.nan

    bull_ret, dd_ret = phase_returns(df, net_ret)
    oos = compute_oos(df, w, fee_bps, slippage_bps)

    return {
        "strategy": name,
        "total_return": float(bt["total_return"]),
        "return_pct": float(bt["return_pct"]),
        "cagr": float(bt["cagr"]),
        "max_drawdown": float(bt["max_drawdown"]),
        "sharpe": float(bt["sharpe"]),
        "gross_return": float(bt["gross_return"]),
        "time_in_btc_pct": time_btc * 100.0,
        "time_in_gold_silver_pct": time_def * 100.0,
        "time_in_cash_pct": time_cash * 100.0,
        "switches": switches,
        "switches_per_year": switches_per_year,
        "btc_bull_phase_return": bull_ret,
        "btc_drawdown_phase_return": dd_ret,
        **oos,
    }


def main() -> None:
    args = parse_args()
    df = prepare_data(args)

    strategies = {
        "rotation_base_optionA": build_weights_rotation(df, False, False),
        "rotation_volume_filter": build_weights_rotation(df, True, False),
        "rotation_vol_expansion_filter": build_weights_rotation(df, False, True),
        "rotation_volume_and_vol_expansion": build_weights_rotation(df, True, True),
        "benchmark_btc_breakout_only_optionA": build_weights_btc_breakout_option_a(df),
        "benchmark_btc_breakout_to_cash_strict": build_weights_btc_breakout_strict_cash(df),
        "benchmark_btc_buy_and_hold": build_weights_buy_hold_btc(df),
    }

    rows = []
    weight_dump = pd.DataFrame({"timestamp": df["timestamp"]})

    for name, w in strategies.items():
        bt = backtest(df, w, args.fee_bps, args.slippage_bps)
        rows.append(summarize(name, df, bt, args.fee_bps, args.slippage_bps))
        weight_dump[f"{name}_btc"] = w[:, 0]
        weight_dump[f"{name}_gold"] = w[:, 1]
        weight_dump[f"{name}_silver"] = w[:, 2]
        weight_dump[f"{name}_cash"] = w[:, 3]

    res = pd.DataFrame(rows)
    ranked = res.sort_values(["return_pct", "oos_return_pct", "sharpe"], ascending=[False, False, False]).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res.to_csv(out_dir / "breakout_rotation_results.csv", index=False)
    ranked.to_csv(out_dir / "breakout_rotation_ranked.csv", index=False)
    weight_dump.to_csv(out_dir / "breakout_rotation_weights.csv", index=False)

    cols = [
        "strategy",
        "return_pct",
        "oos_return_pct",
        "max_drawdown",
        "oos_max_drawdown",
        "sharpe",
        "switches",
        "switches_per_year",
        "time_in_btc_pct",
        "time_in_gold_silver_pct",
        "time_in_cash_pct",
    ]
    print(ranked[cols].to_string(index=False))


if __name__ == "__main__":
    main()
