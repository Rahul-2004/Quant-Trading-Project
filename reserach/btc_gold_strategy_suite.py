#!/usr/bin/env python3
"""Comprehensive BTC + Gold strategy suite with diagnostics.

Outputs (in --out-dir):
- ordered_metrics.csv                 # strategy-level metrics + validation summary
- equity_curves.csv                   # net equity curve for each strategy
- trade_logs.csv                      # trade-level logs (MAE/MFE, fees, slippage, exit reason)
- context_logs.csv                    # entry/exit regime/context snapshots
- validation_splits.csv               # in-sample / out-of-sample / holdout metrics
- walk_forward_windows.csv            # per-window OOS metrics
- robustness_checks.csv               # small parameter perturbation checks
- benchmark_comparison.csv            # vs buy-and-hold / simple baseline / no-trade
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

BARS_PER_DAY = 2  # 12h bars
BARS_PER_YEAR = 365.25 * BARS_PER_DAY
EPS = 1e-10

YAHOO_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest BTC+Gold strategies with diagnostics")
    parser.add_argument("--data", default="data/btc_gold_12h_aligned.csv", help="Aligned BTC/Gold CSV path")
    parser.add_argument("--out-dir", default="reserach/results", help="Output directory")
    parser.add_argument("--cache-dir", default="reserach/cache", help="External-data cache directory")
    parser.add_argument("--fee-bps", type=float, default=4.0, help="Fee bps per turnover")
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="Slippage bps per turnover")
    parser.add_argument("--refresh-external", action="store_true", help="Refresh cached external data")
    parser.add_argument("--long-short", action="store_true", help="Include optional long/short variants")
    return parser.parse_args()


def to_utc_midnight(ts_seconds: int) -> pd.Timestamp:
    return pd.to_datetime(ts_seconds, unit="s", utc=True).floor("D")


def fetch_json_with_curl(url: str, max_retries: int = 7) -> dict | list:
    for attempt in range(max_retries):
        result = subprocess.run(
            ["curl", "-s", "-A", "Mozilla/5.0", url],
            capture_output=True,
            text=True,
            check=False,
        )

        body = result.stdout.strip()
        if result.returncode == 0 and body:
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

    raise RuntimeError(f"Failed to fetch valid JSON after retries: {url}")


def fetch_yahoo_daily(symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    period1 = int(start_ts.timestamp())
    period2 = int((end_ts + pd.Timedelta(days=1)).timestamp())  # exclusive

    url = (
        YAHOO_CHART_URL.format(symbol=symbol)
        + f"?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
    )
    payload = fetch_json_with_curl(url)

    chart = payload.get("chart", {})
    err = chart.get("error")
    if err:
        raise RuntimeError(f"Yahoo chart error for {symbol}: {err}")

    result = (chart.get("result") or [None])[0]
    if not result:
        raise RuntimeError(f"No Yahoo result for {symbol}")

    ts_list = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    close = quote.get("close") or []
    volume = quote.get("volume") or [None] * len(close)

    rows: list[tuple[pd.Timestamp, float, float | None]] = []
    for ts, c, v in zip(ts_list, close, volume):
        if c is None:
            continue
        rows.append((to_utc_midnight(int(ts)), float(c), None if v is None else float(v)))

    if not rows:
        raise RuntimeError(f"No parsed rows for {symbol}")

    df = pd.DataFrame(rows, columns=["timestamp", "close", "volume"])
    return df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")


def fetch_binance_funding(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    cur = start_ms
    rows: list[tuple[pd.Timestamp, float]] = []

    for _ in range(300):
        url = f"{BINANCE_FUNDING_URL}?symbol=BTCUSDT&startTime={cur}&endTime={end_ms}&limit=1000"
        payload = fetch_json_with_curl(url)

        if not isinstance(payload, list) or len(payload) == 0:
            break

        for item in payload:
            ft = int(item["fundingTime"])
            fr = float(item["fundingRate"])
            rows.append((pd.to_datetime(ft, unit="ms", utc=True), fr))

        next_cur = int(payload[-1]["fundingTime"]) + 1
        if next_cur <= cur:
            break
        cur = next_cur

        if cur > end_ms:
            break

        time.sleep(0.05)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    return (
        pd.DataFrame(rows, columns=["timestamp", "funding_rate"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
    )


def load_or_fetch_external(
    cache_file: Path,
    fetch_fn,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    refresh: bool,
) -> pd.DataFrame:
    if cache_file.exists() and not refresh:
        df = pd.read_csv(cache_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    else:
        df = fetch_fn(start_ts, end_ts)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False)

    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp in {cache_file}")

    return df.sort_values("timestamp").reset_index(drop=True)


def merge_asof_feature(base: pd.DataFrame, feature: pd.DataFrame, col_name: str) -> pd.Series:
    b = base[["timestamp"]].sort_values("timestamp").reset_index(drop=True)
    f = feature[["timestamp", col_name]].sort_values("timestamp").reset_index(drop=True)
    merged = pd.merge_asof(b, f, on="timestamp", direction="backward")
    return merged[col_name]


def rolling_percentile_last(series: pd.Series, window: int) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)

    for i in range(window - 1, len(arr)):
        w = arr[i - window + 1 : i + 1]
        valid = w[~np.isnan(w)]
        if len(valid) == 0 or np.isnan(arr[i]):
            continue
        out[i] = float(np.mean(valid <= arr[i]))

    return pd.Series(out, index=series.index)


def atr_proxy(close: pd.Series, window_bars: int = 28) -> pd.Series:
    return close.pct_change().abs().rolling(window_bars).mean()


def trend_strength_proxy(close: pd.Series) -> pd.Series:
    ema_fast = close.ewm(span=24, adjust=False).mean()
    ema_slow = close.ewm(span=72, adjust=False).mean()
    vol = close.pct_change().abs().rolling(28).mean()
    return (ema_fast - ema_slow).abs() / (close * (vol + 1e-9))


def classify_regime(df: pd.DataFrame) -> pd.Series:
    trend_flag = df["trend_strength"] > df["trend_strength"].rolling(200).median()
    high_vol_flag = df["volatility_pctile"] > 0.7
    low_vol_flag = df["volatility_pctile"] < 0.3

    labels = np.full(len(df), "chop", dtype=object)
    labels[trend_flag.fillna(False).to_numpy()] = "trend"
    labels[(trend_flag & high_vol_flag).fillna(False).to_numpy()] = "trend_high_vol"
    labels[(~trend_flag & high_vol_flag).fillna(False).to_numpy()] = "chop_high_vol"
    labels[(~trend_flag & low_vol_flag).fillna(False).to_numpy()] = "chop_low_vol"
    return pd.Series(labels, index=df.index)


def load_data(path: str, cache_dir: str, refresh_external: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in ("btc_close", "gold_close"):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        if (df[col] <= 0).any():
            raise ValueError(f"Found non-positive values in {col}")

    start_ts = df["timestamp"].min().floor("D")
    end_ts = df["timestamp"].max().ceil("D")
    cache_path = Path(cache_dir)

    btc_daily = load_or_fetch_external(
        cache_file=cache_path / "yahoo_btc_daily.csv",
        fetch_fn=lambda s, e: fetch_yahoo_daily("BTC-USD", s, e).rename(
            columns={"close": "btc_yahoo_close", "volume": "btc_volume"}
        ),
        start_ts=start_ts,
        end_ts=end_ts,
        refresh=refresh_external,
    )

    spy_daily = load_or_fetch_external(
        cache_file=cache_path / "yahoo_spy_daily.csv",
        fetch_fn=lambda s, e: fetch_yahoo_daily("SPY", s, e).rename(columns={"close": "spy_close"}),
        start_ts=start_ts,
        end_ts=end_ts,
        refresh=refresh_external,
    )

    funding_8h = load_or_fetch_external(
        cache_file=cache_path / "binance_btc_funding_8h.csv",
        fetch_fn=fetch_binance_funding,
        start_ts=start_ts,
        end_ts=end_ts,
        refresh=refresh_external,
    )

    df["btc_volume"] = merge_asof_feature(df, btc_daily, "btc_volume")
    df["spy_close"] = merge_asof_feature(df, spy_daily, "spy_close")

    if funding_8h.empty:
        df["funding_rate"] = 0.0
    else:
        df["funding_rate"] = merge_asof_feature(df, funding_8h, "funding_rate")

    df["btc_volume"] = df["btc_volume"].ffill().bfill()
    df["spy_close"] = df["spy_close"].ffill().bfill()
    df["funding_rate"] = df["funding_rate"].fillna(0.0)

    df["btc_ret"] = df["btc_close"].pct_change().fillna(0.0)
    df["gold_ret"] = df["gold_close"].pct_change().fillna(0.0)
    df["spy_ret"] = df["spy_close"].pct_change().fillna(0.0)

    df["log_btc"] = np.log(df["btc_close"])
    df["log_gold"] = np.log(df["gold_close"])
    df["ratio_log"] = df["log_btc"] - df["log_gold"]

    df["atr"] = atr_proxy(df["btc_close"], 28)
    df["trend_strength"] = trend_strength_proxy(df["btc_close"])

    vol_lookback = 252
    df["realized_vol"] = df["btc_ret"].rolling(40).std()
    df["volatility_pctile"] = rolling_percentile_last(df["realized_vol"], vol_lookback)
    df["volume_pctile"] = rolling_percentile_last(df["btc_volume"], vol_lookback)

    df["higher_tf_trend"] = (
        df["btc_close"] > df["btc_close"].rolling(400).mean()
    ).astype(float)

    ratio_mu = df["ratio_log"].rolling(120).mean()
    ratio_sd = df["ratio_log"].rolling(120).std()
    df["btc_gold_ratio_z"] = (df["ratio_log"] - ratio_mu) / ratio_sd

    df["gold_shock_flag"] = (
        (df["gold_close"].pct_change(6) > 0.03) | (df["gold_close"].pct_change(10) > 0.04)
    ).astype(float)

    df["regime_label"] = classify_regime(df)
    return df


# -------- Signal helpers --------

def stateful_long_signal(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    e = entry.fillna(False).to_numpy(dtype=bool)
    x = exit_.fillna(False).to_numpy(dtype=bool)

    out = np.zeros(len(e), dtype=float)
    in_pos = False
    for i in range(len(e)):
        if not in_pos and e[i]:
            in_pos = True
        elif in_pos and x[i]:
            in_pos = False
        out[i] = 1.0 if in_pos else 0.0
    return pd.Series(out)


def stateful_long_short_signal(
    long_entry: pd.Series,
    long_exit: pd.Series,
    short_entry: pd.Series,
    short_exit: pd.Series,
) -> pd.Series:
    le = long_entry.fillna(False).to_numpy(dtype=bool)
    lx = long_exit.fillna(False).to_numpy(dtype=bool)
    se = short_entry.fillna(False).to_numpy(dtype=bool)
    sx = short_exit.fillna(False).to_numpy(dtype=bool)

    out = np.zeros(len(le), dtype=float)
    pos = 0.0
    for i in range(len(le)):
        if pos == 0.0:
            if le[i]:
                pos = 1.0
            elif se[i]:
                pos = -1.0
        elif pos > 0.0:
            if lx[i]:
                pos = 0.0
            elif se[i]:
                pos = -1.0
        else:
            if sx[i]:
                pos = 0.0
            elif le[i]:
                pos = 1.0
        out[i] = pos
    return pd.Series(out)


def apply_atr_sizing(signal: pd.Series, atr: pd.Series, floor: float = 0.25, cap: float = 1.0) -> pd.Series:
    med = atr.median(skipna=True)
    if pd.isna(med) or med <= 0:
        return signal.fillna(0.0)

    size = (med / atr).replace([np.inf, -np.inf], np.nan).clip(lower=floor, upper=cap)
    size = size.fillna(1.0)
    return signal.fillna(0.0) * size


def ma_trend_signal(df: pd.DataFrame, long_short: bool, w20: int = 40, w50: int = 100, w200: int = 400) -> pd.Series:
    close = df["btc_close"]
    ma20 = close.rolling(w20).mean()
    ma50 = close.rolling(w50).mean()
    ma200 = close.rolling(w200).mean()

    bull = (ma20 > ma50) & (ma50 > ma200)
    bear = (ma20 < ma50) & (ma50 < ma200)

    if long_short:
        return pd.Series(np.where(bull, 1.0, np.where(bear, -1.0, 0.0)))
    return pd.Series(np.where(bull, 1.0, 0.0))


def breakout_signal(
    df: pd.DataFrame,
    long_short: bool,
    lookback: int = 40,
    exit_ma: int = 20,
) -> pd.Series:
    close = df["btc_close"]
    hi = close.shift(1).rolling(lookback).max()
    lo = close.shift(1).rolling(lookback).min()

    if long_short:
        long_entry = close > hi
        long_exit = close < close.rolling(exit_ma).mean()
        short_entry = close < lo
        short_exit = close > close.rolling(exit_ma).mean()
        return stateful_long_short_signal(long_entry, long_exit, short_entry, short_exit)

    entry = close > hi
    exit_ = close < lo
    return stateful_long_signal(entry, exit_)


def zscore_reversal_signal(
    df: pd.DataFrame,
    long_short: bool,
    lookback: int = 40,
    z_thr: float = 2.0,
) -> pd.Series:
    close = df["btc_close"]
    mean = close.rolling(lookback).mean()
    std = close.rolling(lookback).std()
    z = (close - mean) / std
    dz = z.diff()

    if long_short:
        long_entry = (z < -z_thr) & (dz > 0)
        long_exit = (z > -0.25) | ((z > 1.25) & (dz < 0))

        short_entry = (z > z_thr) & (dz < 0)
        short_exit = (z < 0.25) | ((z < -1.25) & (dz > 0))
        return stateful_long_short_signal(long_entry, long_exit, short_entry, short_exit)

    entry = (z < -z_thr) & (dz > 0)
    exit_ = (z > -0.25) | ((z > 1.25) & (dz < 0))
    return stateful_long_signal(entry, exit_)


def ratio_mean_reversion_signal(
    df: pd.DataFrame,
    long_short: bool,
    lookback: int = 120,
    z_thr: float = 2.0,
) -> pd.Series:
    ratio = df["ratio_log"]
    mu = ratio.rolling(lookback).mean()
    sd = ratio.rolling(lookback).std()
    z = (ratio - mu) / sd
    dz = z.diff()

    if long_short:
        long_entry = (z < -z_thr) & (dz > 0)
        long_exit = (z > -0.15) | ((z > 1.25) & (dz < 0))

        short_entry = (z > z_thr) & (dz < 0)
        short_exit = (z < 0.15) | ((z < -1.25) & (dz > 0))
        return stateful_long_short_signal(long_entry, long_exit, short_entry, short_exit)

    entry = (z < -z_thr) & (dz > 0)
    exit_ = (z > -0.15) | ((z > 1.25) & (dz < 0))
    return stateful_long_signal(entry, exit_)


def gold_regime_filter_signal(df: pd.DataFrame, g_factor: float = 0.5) -> pd.Series:
    base = ma_trend_signal(df, long_short=False)
    g = df["gold_close"]
    defensive = (g > g.rolling(400).mean()) & (g.pct_change(40) > 0)
    factor = pd.Series(np.where(defensive, g_factor, 1.0))
    return base * factor


def gold_shock_filter_signal(df: pd.DataFrame, stress_factor: float = 0.35) -> pd.Series:
    base = ma_trend_signal(df, long_short=False)
    shock = (df["gold_close"].pct_change(6) > 0.03) | (df["gold_close"].pct_change(10) > 0.04)
    stress_active = shock.rolling(10, min_periods=1).max().astype(bool)
    factor = pd.Series(np.where(stress_active, stress_factor, 1.0))
    return base * factor


def beta_corr_filter_signal(df: pd.DataFrame, long_short: bool, unstable_factor: float = 0.4) -> pd.Series:
    base = ratio_mean_reversion_signal(df, long_short=long_short)

    corr = df["btc_ret"].rolling(120).corr(df["gold_ret"])
    cov = df["btc_ret"].rolling(120).cov(df["gold_ret"])
    var = df["gold_ret"].rolling(120).var()
    beta = cov / var

    corr_lag = corr.shift(20)
    beta_lag = beta.shift(20)

    unstable = (corr * corr_lag < 0) | ((corr - corr_lag).abs() > 0.35) | ((beta - beta_lag).abs() > 0.50)
    factor = pd.Series(np.where(unstable.fillna(False), unstable_factor, 1.0))
    return base * factor


def residual_stationary_signal(
    df: pd.DataFrame,
    long_short: bool,
    reg_window: int = 360,
    z_thr: float = 2.0,
) -> pd.Series:
    y = df["log_btc"].to_numpy(dtype=float)
    x = df["log_gold"].to_numpy(dtype=float)
    n = len(df)

    adf_step = 12
    residual = np.full(n, np.nan)
    stationary = np.zeros(n, dtype=bool)

    for i in range(reg_window, n):
        y_w = y[i - reg_window : i]
        x_w = x[i - reg_window : i]

        vx = np.var(x_w)
        if vx < 1e-12:
            continue

        beta = np.cov(x_w, y_w, ddof=0)[0, 1] / vx
        alpha = np.mean(y_w) - beta * np.mean(x_w)
        residual[i] = y[i] - (alpha + beta * x[i])

        if i % adf_step == 0:
            resid_w = y_w - (alpha + beta * x_w)
            try:
                pval = adfuller(resid_w, maxlag=1, regression="c", autolag=None)[1]
                stationary[i : min(n, i + adf_step)] = pval < 0.05
            except Exception:
                stationary[i : min(n, i + adf_step)] = False

    resid_s = pd.Series(residual)
    z = (resid_s - resid_s.rolling(120).mean()) / resid_s.rolling(120).std()
    dz = z.diff()
    stat = pd.Series(stationary)

    if long_short:
        long_entry = stat & (z < -z_thr) & (dz > 0)
        long_exit = (~stat) | (z > -0.1) | ((z > 1.25) & (dz < 0))

        short_entry = stat & (z > z_thr) & (dz < 0)
        short_exit = (~stat) | (z < 0.1) | ((z < -1.25) & (dz > 0))
        return stateful_long_short_signal(long_entry, long_exit, short_entry, short_exit)

    entry = stat & (z < -z_thr) & (dz > 0)
    exit_ = (~stat) | (z > -0.1) | ((z > 1.25) & (dz < 0))
    return stateful_long_signal(entry, exit_)


def clean_combined_signal(df: pd.DataFrame, rich_cut: float = 1.25) -> pd.Series:
    btc_momo = ma_trend_signal(df, long_short=False)
    regime_factor = pd.Series(
        np.where(
            ((df["gold_close"] > df["gold_close"].rolling(400).mean()) & (df["gold_close"].pct_change(40) > 0)).fillna(False),
            0.5,
            1.0,
        )
    )

    ratio = df["ratio_log"]
    rz = (ratio - ratio.rolling(120).mean()) / ratio.rolling(120).std()
    rdz = rz.diff()

    overlay = pd.Series(1.0, index=df.index)
    overlay[(rz > rich_cut) & (rdz < 0)] = 0.6

    return btc_momo * regime_factor * overlay


# New requested tests

def vol_expansion_breakout_signal(df: pd.DataFrame, vol_thr: float = 1.0, lookback: int = 40) -> pd.Series:
    close = df["btc_close"]
    hi = close.shift(1).rolling(lookback).max()
    lo = close.shift(1).rolling(lookback).min()

    atr = df["atr"]
    atr_rel = atr / atr.rolling(40).mean()
    vol_expanding = (atr_rel > vol_thr) & (atr_rel.diff() > 0)

    entry = (close > hi) & vol_expanding
    exit_ = (close < lo) | (atr_rel < 0.9)
    return stateful_long_signal(entry, exit_)


def squeeze_expansion_signal(df: pd.DataFrame, q: float = 0.2) -> pd.Series:
    close = df["btc_close"]
    vol = close.pct_change().rolling(20).std()
    low_q = vol.rolling(80).quantile(q)
    compression = vol <= low_q
    compression_recent = compression.rolling(10, min_periods=1).max().astype(bool)

    hi = close.shift(1).rolling(20).max()
    lo = close.shift(1).rolling(20).min()

    entry = compression_recent & (close > hi)
    exit_ = (close < lo) | (vol < vol.rolling(40).mean() * 0.9)
    return stateful_long_signal(entry, exit_)


def drawdown_recovery_momentum_signal(df: pd.DataFrame, dd_cut: float = -0.25) -> pd.Series:
    close = df["btc_close"]
    local_high = close.rolling(120).max()
    dd = close / local_high - 1.0

    ma20 = close.rolling(40).mean()
    ma50 = close.rolling(100).mean()

    deep_dd_recent = (dd < dd_cut).rolling(120, min_periods=1).max().astype(bool)
    momo_flip_up = (ma20 > ma50) & (ma20.shift(1) <= ma50.shift(1))

    entry = deep_dd_recent & momo_flip_up
    exit_ = ma20 < ma50
    return stateful_long_signal(entry, exit_)


def funding_sentiment_filter_signal(df: pd.DataFrame, z_cut: float = 1.5) -> pd.Series:
    base = breakout_signal(df, long_short=False)
    fr = df["funding_rate"].fillna(0.0)

    fr_mean = fr.rolling(90).mean()
    fr_std = fr.rolling(90).std().replace(0, np.nan)
    fr_z = (fr - fr_mean) / fr_std

    avoid_long = fr_z > z_cut
    scale = pd.Series(np.where(fr < 0, 1.0, 0.7), index=df.index)
    scale[avoid_long.fillna(False)] = 0.0

    return base * scale


def volume_confirmed_breakout_signal(df: pd.DataFrame, vol_mult: float = 1.2) -> pd.Series:
    close = df["btc_close"]
    vol = df["btc_volume"].ffill().bfill()

    hi = close.shift(1).rolling(40).max()
    lo = close.shift(1).rolling(40).min()

    vol_spike = vol > vol.rolling(40).mean() * vol_mult

    entry = (close > hi) & vol_spike
    exit_ = close < lo
    return stateful_long_signal(entry, exit_)


def session_effect_signal(df: pd.DataFrame) -> pd.Series:
    base = ma_trend_signal(df, long_short=False)
    us_session_bar = df["timestamp"].dt.hour == 12
    return base * us_session_bar.astype(float)


def regime_switching_signal(df: pd.DataFrame, strength_cut: float = 1.1) -> pd.Series:
    trending = df["trend_strength"] > strength_cut
    trend_leg = breakout_signal(df, long_short=False)
    range_leg = zscore_reversal_signal(df, long_short=False) * 0.5
    return pd.Series(np.where(trending.fillna(False), trend_leg, range_leg))


def trend_strength_filter_signal(df: pd.DataFrame, strength_cut: float = 1.1) -> pd.Series:
    base = ma_trend_signal(df, long_short=False)
    return base * (df["trend_strength"] > strength_cut).astype(float)


def momentum_decay_exit_signal(df: pd.DataFrame, lookback: int = 40) -> pd.Series:
    close = df["btc_close"]
    hi = close.shift(1).rolling(lookback).max()

    mom = close.pct_change(20)
    mom_ema = mom.ewm(span=10, adjust=False).mean()

    entry = close > hi
    exit_ = (mom < mom_ema) | (mom < 0)
    return stateful_long_signal(entry, exit_)


def cross_asset_risk_on_signal(df: pd.DataFrame) -> pd.Series:
    base = ma_trend_signal(df, long_short=False)

    btc_up = df["btc_close"] > df["btc_close"].rolling(100).mean()
    spy_up = df["spy_close"] > df["spy_close"].rolling(400).mean()
    gold_down = df["gold_close"] < df["gold_close"].rolling(100).mean()

    risk_on = btc_up & spy_up & gold_down
    return base * risk_on.astype(float)


# -------- Strategy catalog --------

def strategy_names(include_long_short: bool) -> List[str]:
    base = [
        "00_no_trade",
        "01_btc_buy_and_hold",
        "02_btc_ma_20_50_200",
        "03_btc_20d_breakout",
        "04_btc_20d_zscore_reversal",
        "05_btc_gold_ratio_mr",
        "06_btc_momentum_plus_gold_regime",
        "07_btc_momentum_plus_gold_shock_filter",
        "08_ratio_mr_plus_beta_corr_stability",
        "09_residual_spread_stationary_only",
        "10_clean_combo_momo_regime_ratio_overlay",
        "11_vol_expansion_breakout",
        "12_squeeze_compression_expansion",
        "13_drawdown_recovery_momentum",
        "14_funding_sentiment_filter",
        "15_volume_confirmed_breakout",
        "16_time_of_day_session_effect",
        "17_regime_switching_model",
        "18_trend_strength_filter",
        "19_momentum_decay_exit",
        "20_cross_asset_risk_on_filter",
    ]

    if include_long_short:
        base.extend(
            [
                "ls_01_btc_ma_20_50_200",
                "ls_02_btc_gold_ratio_mr",
                "ls_03_residual_spread_stationary_only",
            ]
        )
    return base


def build_signal(df: pd.DataFrame, name: str, perturb: float = 1.0) -> Tuple[pd.Series, bool]:
    atr = df["atr"]

    def w(v: int) -> int:
        return max(3, int(round(v * perturb)))

    def z(v: float) -> float:
        return float(v * perturb)

    long_short = False

    if name == "00_no_trade":
        sig = pd.Series(0.0, index=df.index)
    elif name == "01_btc_buy_and_hold":
        sig = pd.Series(1.0, index=df.index)
    elif name == "02_btc_ma_20_50_200":
        sig = apply_atr_sizing(ma_trend_signal(df, False, w20=w(40), w50=w(100), w200=w(400)), atr)
    elif name == "03_btc_20d_breakout":
        sig = apply_atr_sizing(breakout_signal(df, False, lookback=w(40), exit_ma=w(20)), atr)
    elif name == "04_btc_20d_zscore_reversal":
        sig = apply_atr_sizing(zscore_reversal_signal(df, False, lookback=w(40), z_thr=z(2.0)), atr)
    elif name == "05_btc_gold_ratio_mr":
        sig = apply_atr_sizing(ratio_mean_reversion_signal(df, False, lookback=w(120), z_thr=z(2.0)), atr)
    elif name == "06_btc_momentum_plus_gold_regime":
        sig = apply_atr_sizing(gold_regime_filter_signal(df, g_factor=max(0.2, min(0.8, 0.5 * perturb))), atr)
    elif name == "07_btc_momentum_plus_gold_shock_filter":
        sig = apply_atr_sizing(gold_shock_filter_signal(df, stress_factor=max(0.15, min(0.7, 0.35 * perturb))), atr)
    elif name == "08_ratio_mr_plus_beta_corr_stability":
        sig = apply_atr_sizing(beta_corr_filter_signal(df, False, unstable_factor=max(0.2, min(0.8, 0.4 * perturb))), atr)
    elif name == "09_residual_spread_stationary_only":
        sig = apply_atr_sizing(
            residual_stationary_signal(df, False, reg_window=w(360), z_thr=z(2.0)),
            atr,
        )
    elif name == "10_clean_combo_momo_regime_ratio_overlay":
        sig = apply_atr_sizing(clean_combined_signal(df, rich_cut=z(1.25)), atr)
    elif name == "11_vol_expansion_breakout":
        sig = apply_atr_sizing(vol_expansion_breakout_signal(df, vol_thr=z(1.0), lookback=w(40)), atr)
    elif name == "12_squeeze_compression_expansion":
        q = min(0.4, max(0.05, 0.2 * perturb))
        sig = apply_atr_sizing(squeeze_expansion_signal(df, q=q), atr)
    elif name == "13_drawdown_recovery_momentum":
        dd_cut = max(-0.5, min(-0.1, -0.25 * perturb))
        sig = apply_atr_sizing(drawdown_recovery_momentum_signal(df, dd_cut=dd_cut), atr)
    elif name == "14_funding_sentiment_filter":
        sig = apply_atr_sizing(funding_sentiment_filter_signal(df, z_cut=z(1.5)), atr)
    elif name == "15_volume_confirmed_breakout":
        sig = apply_atr_sizing(volume_confirmed_breakout_signal(df, vol_mult=z(1.2)), atr)
    elif name == "16_time_of_day_session_effect":
        sig = apply_atr_sizing(session_effect_signal(df), atr)
    elif name == "17_regime_switching_model":
        sig = apply_atr_sizing(regime_switching_signal(df, strength_cut=z(1.1)), atr)
    elif name == "18_trend_strength_filter":
        sig = apply_atr_sizing(trend_strength_filter_signal(df, strength_cut=z(1.1)), atr)
    elif name == "19_momentum_decay_exit":
        sig = apply_atr_sizing(momentum_decay_exit_signal(df, lookback=w(40)), atr)
    elif name == "20_cross_asset_risk_on_filter":
        sig = apply_atr_sizing(cross_asset_risk_on_signal(df), atr)
    elif name == "ls_01_btc_ma_20_50_200":
        long_short = True
        sig = apply_atr_sizing(ma_trend_signal(df, True, w20=w(40), w50=w(100), w200=w(400)), atr, cap=1.0)
    elif name == "ls_02_btc_gold_ratio_mr":
        long_short = True
        sig = apply_atr_sizing(ratio_mean_reversion_signal(df, True, lookback=w(120), z_thr=z(2.0)), atr, cap=1.0)
    elif name == "ls_03_residual_spread_stationary_only":
        long_short = True
        sig = apply_atr_sizing(
            residual_stationary_signal(df, True, reg_window=w(360), z_thr=z(2.0)),
            atr,
            cap=1.0,
        )
    else:
        raise ValueError(f"Unknown strategy: {name}")

    return sig.fillna(0.0), long_short


# -------- Backtest core --------

def simulate(
    df: pd.DataFrame,
    raw_position: pd.Series,
    fee_bps: float,
    slippage_bps: float,
    long_short: bool,
) -> Dict[str, Any]:
    pos = raw_position.fillna(0.0).to_numpy(dtype=float)
    pos = np.clip(pos, -1.0, 1.0) if long_short else np.clip(pos, 0.0, 1.0)

    prev_pos = np.r_[0.0, pos[:-1]]
    ret = df["btc_ret"].to_numpy(dtype=float)

    turnover = np.abs(pos - prev_pos)
    fee_cost = turnover * fee_bps / 10000.0
    slippage_cost = turnover * slippage_bps / 10000.0
    total_cost = fee_cost + slippage_cost

    gross_ret = prev_pos * ret
    net_ret = gross_ret - total_cost

    equity_gross = np.cumprod(1.0 + gross_ret)
    equity_net = np.cumprod(1.0 + net_ret)

    return {
        "pos": pos,
        "prev_pos": prev_pos,
        "turnover": turnover,
        "fee_cost": fee_cost,
        "slippage_cost": slippage_cost,
        "total_cost": total_cost,
        "gross_ret": gross_ret,
        "net_ret": net_ret,
        "equity_gross": equity_gross,
        "equity_net": equity_net,
    }


def compute_metrics_from_sim(sim: Dict[str, Any], trade_df: pd.DataFrame | None = None) -> Dict[str, float]:
    net_ret = sim["net_ret"]
    gross_ret = sim["gross_ret"]
    equity_net = sim["equity_net"]
    equity_gross = sim["equity_gross"]
    turnover = sim["turnover"]
    prev_pos = sim["prev_pos"]

    years = len(net_ret) / BARS_PER_YEAR
    ret_after = float(equity_net[-1] - 1.0) if len(equity_net) else 0.0
    ret_before = float(equity_gross[-1] - 1.0) if len(equity_gross) else 0.0

    cagr = (1.0 + ret_after) ** (1.0 / years) - 1.0 if years > 0 and (1.0 + ret_after) > 0 else np.nan

    running_max = np.maximum.accumulate(equity_net)
    drawdown = equity_net / running_max - 1.0
    max_drawdown = float(np.min(drawdown)) if len(drawdown) else 0.0

    sd = float(np.std(net_ret))
    sharpe = (float(np.mean(net_ret)) / sd) * np.sqrt(BARS_PER_YEAR) if sd > 1e-12 else 0.0

    downside = net_ret[net_ret < 0]
    downside_sd = float(np.std(downside)) if len(downside) else 0.0
    sortino = (float(np.mean(net_ret)) / downside_sd) * np.sqrt(BARS_PER_YEAR) if downside_sd > 1e-12 else 0.0

    exposure = float(np.mean(np.abs(prev_pos) > EPS))
    annual_turnover = float(np.mean(turnover) * BARS_PER_YEAR)

    if trade_df is None or trade_df.empty:
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        expectancy = 0.0
        profit_factor = 0.0
    else:
        wins = trade_df[trade_df["net_pnl"] > 0]["net_pnl"]
        losses = trade_df[trade_df["net_pnl"] < 0]["net_pnl"]

        win_rate = float(np.mean(trade_df["net_pnl"] > 0))
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        expectancy = float(trade_df["net_pnl"].mean())

        gp = float(wins.sum()) if len(wins) else 0.0
        gl = float(np.abs(losses.sum())) if len(losses) else 0.0
        profit_factor = gp / gl if gl > 1e-12 else (999.0 if gp > 0 else 0.0)

    return {
        "total_return": ret_after,
        "return_pct": ret_after * 100.0,
        "return_before_costs": ret_before,
        "return_after_costs": ret_after,
        "cagr": float(cagr) if not np.isnan(cagr) else np.nan,
        "max_drawdown": max_drawdown,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "profit_factor": float(profit_factor),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "expectancy_per_trade": float(expectancy),
        "exposure": exposure,
        "turnover": annual_turnover,
    }


def extract_trade_and_context_logs(
    strategy: str,
    df: pd.DataFrame,
    sim: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pos = sim["pos"]
    gross_ret = sim["gross_ret"]
    net_ret = sim["net_ret"]
    fee_cost = sim["fee_cost"]
    slippage_cost = sim["slippage_cost"]

    close = df["btc_close"].to_numpy(dtype=float)
    ts = df["timestamp"].to_numpy()

    trade_rows: list[dict] = []
    context_rows: list[dict] = []

    trade_id = 0
    i = 1
    n = len(df)

    while i < n:
        if abs(pos[i - 1]) <= EPS and abs(pos[i]) > EPS:
            side_sign = 1.0 if pos[i] > 0 else -1.0
            entry_idx = i

            k = i + 1
            while k < n and (side_sign * pos[k] > EPS):
                k += 1

            if k < n:
                exit_idx = k
                if side_sign * pos[k] < -EPS:
                    exit_reason = "signal_flip"
                else:
                    exit_reason = "signal_flat"
            else:
                exit_idx = n - 1
                exit_reason = "end_of_data"

            bars = np.arange(entry_idx, exit_idx + 1)
            trade_gross = float(np.prod(1.0 + gross_ret[bars]) - 1.0)
            trade_net = float(np.prod(1.0 + net_ret[bars]) - 1.0)
            trade_fee = float(np.sum(fee_cost[bars]))
            trade_slip = float(np.sum(slippage_cost[bars]))

            entry_price = float(close[entry_idx])
            exit_price = float(close[exit_idx])
            path = close[entry_idx : exit_idx + 1]

            if side_sign > 0:
                rel = path / entry_price - 1.0
            else:
                rel = entry_price / path - 1.0

            mfe = float(np.nanmax(rel)) if len(rel) else 0.0
            mae = float(np.nanmin(rel)) if len(rel) else 0.0

            entry_atr = float(df["atr"].iloc[entry_idx]) if not np.isnan(df["atr"].iloc[entry_idx]) else 0.0
            risk_unit = max(2.0 * entry_atr, 1e-6)
            r_multiple = float(trade_net / risk_unit)

            hold_bars = int(exit_idx - entry_idx + 1)
            hold_hours = hold_bars * 12

            trade_id += 1
            trade_row = {
                "strategy": strategy,
                "trade_id": trade_id,
                "entry_time": pd.Timestamp(ts[entry_idx]),
                "exit_time": pd.Timestamp(ts[exit_idx]),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "side": "LONG" if side_sign > 0 else "SHORT",
                "position_size": float(abs(pos[entry_idx])),
                "leverage": float(abs(pos[entry_idx])),
                "gross_pnl": trade_gross,
                "fees": trade_fee,
                "slippage": trade_slip,
                "net_pnl": trade_net,
                "r_multiple": r_multiple,
                "holding_bars": hold_bars,
                "holding_hours": hold_hours,
                "exit_reason": exit_reason,
                "mae": mae,
                "mfe": mfe,
            }
            trade_rows.append(trade_row)

            context_row = {
                "strategy": strategy,
                "trade_id": trade_id,
                "entry_time": pd.Timestamp(ts[entry_idx]),
                "exit_time": pd.Timestamp(ts[exit_idx]),
                "entry_trend_strength": float(df["trend_strength"].iloc[entry_idx]),
                "entry_volatility_pctile": float(df["volatility_pctile"].iloc[entry_idx]),
                "entry_volume_pctile": float(df["volume_pctile"].iloc[entry_idx]),
                "entry_funding_rate": float(df["funding_rate"].iloc[entry_idx]),
                "entry_higher_tf_trend": float(df["higher_tf_trend"].iloc[entry_idx]),
                "entry_regime_label": str(df["regime_label"].iloc[entry_idx]),
                "entry_btc_gold_ratio_z": float(df["btc_gold_ratio_z"].iloc[entry_idx]),
                "entry_gold_shock_flag": float(df["gold_shock_flag"].iloc[entry_idx]),
                "exit_trend_strength": float(df["trend_strength"].iloc[exit_idx]),
                "exit_volatility_pctile": float(df["volatility_pctile"].iloc[exit_idx]),
                "exit_volume_pctile": float(df["volume_pctile"].iloc[exit_idx]),
                "exit_funding_rate": float(df["funding_rate"].iloc[exit_idx]),
                "exit_higher_tf_trend": float(df["higher_tf_trend"].iloc[exit_idx]),
                "exit_regime_label": str(df["regime_label"].iloc[exit_idx]),
                "exit_btc_gold_ratio_z": float(df["btc_gold_ratio_z"].iloc[exit_idx]),
                "exit_gold_shock_flag": float(df["gold_shock_flag"].iloc[exit_idx]),
            }
            context_rows.append(context_row)

            i = exit_idx + 1
        else:
            i += 1

    trade_df = pd.DataFrame(trade_rows)
    context_df = pd.DataFrame(context_rows)
    return trade_df, context_df


# -------- Validation --------

def build_splits(n: int) -> dict[str, Tuple[int, int]]:
    holdout_start = int(0.85 * n)
    dev_end = holdout_start
    is_end = int(0.70 * dev_end)

    return {
        "in_sample": (0, is_end),
        "out_of_sample": (is_end, dev_end),
        "holdout": (holdout_start, n),
    }


def build_walkforward_windows(dev_n: int) -> List[Tuple[int, int, int, int, int]]:
    train_len = max(500, int(0.50 * dev_n))
    test_len = max(150, int(0.10 * dev_n))

    windows = []
    wid = 0
    start = 0
    while start + train_len + test_len <= dev_n:
        wid += 1
        is_s = start
        is_e = start + train_len
        oos_s = is_e
        oos_e = is_e + test_len
        windows.append((wid, is_s, is_e, oos_s, oos_e))
        start += test_len

    return windows


def evaluate_segment(
    df: pd.DataFrame,
    pos: np.ndarray,
    fee_bps: float,
    slippage_bps: float,
    long_short: bool,
    start: int,
    end: int,
) -> Dict[str, Any]:
    if end - start <= 2:
        return {
            "total_return": 0.0,
            "return_pct": 0.0,
            "cagr": np.nan,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy_per_trade": 0.0,
            "exposure": 0.0,
            "turnover": 0.0,
            "return_before_costs": 0.0,
            "return_after_costs": 0.0,
        }

    seg_df = df.iloc[start:end].reset_index(drop=True)
    seg_pos = pd.Series(pos[start:end])

    sim = simulate(seg_df, seg_pos, fee_bps, slippage_bps, long_short)
    trade_df, _ = extract_trade_and_context_logs("seg", seg_df, sim)
    return compute_metrics_from_sim(sim, trade_df)


def run_robustness(
    df: pd.DataFrame,
    strategy: str,
    fee_bps: float,
    slippage_bps: float,
    long_short: bool,
) -> Tuple[pd.DataFrame, float]:
    rows = []
    returns = []

    for label, perturb in [("down_10pct", 0.90), ("base", 1.00), ("up_10pct", 1.10)]:
        sig, ls = build_signal(df, strategy, perturb=perturb)
        ls = ls or long_short
        sim = simulate(df, sig, fee_bps, slippage_bps, ls)
        ret = float(sim["equity_net"][-1] - 1.0)
        rows.append(
            {
                "strategy": strategy,
                "variant": label,
                "perturb_factor": perturb,
                "total_return": ret,
                "return_pct": ret * 100.0,
            }
        )
        returns.append(ret)

    robust_score = float(np.mean(np.array(returns) > 0.0))
    return pd.DataFrame(rows), robust_score


# -------- Runner --------

def run_suite(df: pd.DataFrame, fee_bps: float, slippage_bps: float, include_long_short: bool) -> Dict[str, pd.DataFrame]:
    names = strategy_names(include_long_short)
    n = len(df)

    splits = build_splits(n)
    dev_n = splits["out_of_sample"][1]
    wf_windows = build_walkforward_windows(dev_n)

    metrics_rows: list[dict] = []
    validation_rows: list[dict] = []
    wf_rows: list[dict] = []
    robustness_tables: list[pd.DataFrame] = []
    trade_tables: list[pd.DataFrame] = []
    context_tables: list[pd.DataFrame] = []

    equity_df = pd.DataFrame({"timestamp": df["timestamp"]})

    for name in names:
        signal, long_short = build_signal(df, name, perturb=1.0)
        sim = simulate(df, signal, fee_bps, slippage_bps, long_short)

        trade_df, context_df = extract_trade_and_context_logs(name, df, sim)
        trade_tables.append(trade_df)
        context_tables.append(context_df)

        full_metrics = compute_metrics_from_sim(sim, trade_df)

        # IS / OOS / Holdout
        split_results: dict[str, dict] = {}
        for split_name, (s, e) in splits.items():
            seg = evaluate_segment(df, sim["pos"], fee_bps, slippage_bps, long_short, s, e)
            split_results[split_name] = seg
            validation_rows.append(
                {
                    "strategy": name,
                    "split": split_name,
                    "start_time": df["timestamp"].iloc[s],
                    "end_time": df["timestamp"].iloc[e - 1],
                    **seg,
                }
            )

        # Walk-forward OOS windows
        wf_ok = []
        wf_oos_returns = []
        for wid, is_s, is_e, oos_s, oos_e in wf_windows:
            oos_seg = evaluate_segment(df, sim["pos"], fee_bps, slippage_bps, long_short, oos_s, oos_e)
            wf_oos_returns.append(oos_seg["total_return"])
            wf_ok.append((oos_seg["total_return"] > 0.0) and (oos_seg["sharpe"] > 0.0))

            wf_rows.append(
                {
                    "strategy": name,
                    "window_id": wid,
                    "is_start_time": df["timestamp"].iloc[is_s],
                    "is_end_time": df["timestamp"].iloc[is_e - 1],
                    "oos_start_time": df["timestamp"].iloc[oos_s],
                    "oos_end_time": df["timestamp"].iloc[oos_e - 1],
                    "oos_total_return": oos_seg["total_return"],
                    "oos_return_pct": oos_seg["return_pct"],
                    "oos_sharpe": oos_seg["sharpe"],
                    "oos_max_drawdown": oos_seg["max_drawdown"],
                }
            )

        wf_consistency = float(np.mean(wf_ok)) if len(wf_ok) else np.nan
        wf_mean_oos_return = float(np.mean(wf_oos_returns)) if len(wf_oos_returns) else np.nan

        # Parameter perturbation robustness
        robust_df, robust_score = run_robustness(df, name, fee_bps, slippage_bps, long_short)
        robustness_tables.append(robust_df)

        metric_row = {
            "strategy": name,
            **full_metrics,
            "in_sample_return": split_results["in_sample"]["total_return"],
            "out_of_sample_return": split_results["out_of_sample"]["total_return"],
            "holdout_return": split_results["holdout"]["total_return"],
            "walk_forward_consistency": wf_consistency,
            "walk_forward_mean_oos_return": wf_mean_oos_return,
            "robustness_score": robust_score,
        }
        metrics_rows.append(metric_row)

        equity_df[name] = sim["equity_net"]

    metrics_df = pd.DataFrame(metrics_rows)
    validation_df = pd.DataFrame(validation_rows)
    wf_df = pd.DataFrame(wf_rows)
    robustness_df = pd.concat(robustness_tables, ignore_index=True) if robustness_tables else pd.DataFrame()

    trade_log_df = pd.concat([t for t in trade_tables if not t.empty], ignore_index=True) if trade_tables else pd.DataFrame()
    context_log_df = pd.concat([c for c in context_tables if not c.empty], ignore_index=True) if context_tables else pd.DataFrame()

    # Benchmark comparison
    bench = metrics_df.set_index("strategy")
    buy_hold = bench.loc["01_btc_buy_and_hold"]
    simple_base = bench.loc["02_btc_ma_20_50_200"]
    no_trade = bench.loc["00_no_trade"]

    b_rows = []
    for _, r in metrics_df.iterrows():
        b_rows.append(
            {
                "strategy": r["strategy"],
                "total_return": r["total_return"],
                "return_pct": r["return_pct"],
                "sharpe": r["sharpe"],
                "excess_return_vs_buy_hold_pct": (r["total_return"] - buy_hold["total_return"]) * 100.0,
                "excess_return_vs_simple_baseline_pct": (r["total_return"] - simple_base["total_return"]) * 100.0,
                "excess_return_vs_no_trade_pct": (r["total_return"] - no_trade["total_return"]) * 100.0,
                "sharpe_diff_vs_buy_hold": r["sharpe"] - buy_hold["sharpe"],
                "sharpe_diff_vs_simple_baseline": r["sharpe"] - simple_base["sharpe"],
            }
        )

    benchmark_df = pd.DataFrame(b_rows)

    return {
        "metrics": metrics_df,
        "equity": equity_df,
        "trade_logs": trade_log_df,
        "context_logs": context_log_df,
        "validation": validation_df,
        "walkforward": wf_df,
        "robustness": robustness_df,
        "benchmark": benchmark_df,
    }


def main() -> None:
    args = parse_args()

    df = load_data(args.data, args.cache_dir, args.refresh_external)
    outputs = run_suite(
        df=df,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        include_long_short=args.long_short,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_map = {
        "metrics": out_dir / "ordered_metrics.csv",
        "equity": out_dir / "equity_curves.csv",
        "trade_logs": out_dir / "trade_logs.csv",
        "context_logs": out_dir / "context_logs.csv",
        "validation": out_dir / "validation_splits.csv",
        "walkforward": out_dir / "walk_forward_windows.csv",
        "robustness": out_dir / "robustness_checks.csv",
        "benchmark": out_dir / "benchmark_comparison.csv",
    }

    for key, path in file_map.items():
        outputs[key].to_csv(path, index=False)

    show = outputs["metrics"].copy()
    print(f"Saved outputs to {out_dir}")
    print(show[["strategy", "return_pct", "cagr", "max_drawdown", "sharpe", "profit_factor", "win_rate", "out_of_sample_return", "walk_forward_consistency", "robustness_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
