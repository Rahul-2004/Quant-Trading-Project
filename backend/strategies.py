import csv
from datetime import datetime
import polars as pl
import numpy as np
import torch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List

from arch import arch_model

from .models_shared import LinearModel, NonLinearModel
from .utils import load_ohlc_timeseries, add_lags, to_tensor, timeseries_split, batch_train_reg, set_seed

ANNUALIZATION_FACTOR = np.sqrt(24 * 365)
RESULTS_DIR = Path(__file__).resolve().parents[1] / "reserach" / "results"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = Path(__file__).resolve().parents[1] / "reserach" / "cache"


def _to_1d(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array.reshape(-1)


def _equity_curve_from_returns(trade_returns: np.ndarray) -> np.ndarray:
    if trade_returns.size == 0:
        return np.array([], dtype=float)
    return np.cumsum(trade_returns)


def _compute_win_rate(signals: np.ndarray, realized_returns: np.ndarray) -> float:
    if signals.size == 0 or realized_returns.size == 0:
        return 0.0

    length = min(signals.size, realized_returns.size)
    aligned_signals = signals[:length]
    aligned_returns = realized_returns[:length]
    traded_mask = aligned_signals != 0
    if not np.any(traded_mask):
        return 0.0

    return float(np.mean(np.sign(aligned_signals[traded_mask]) == np.sign(aligned_returns[traded_mask])))


def _compute_sharpe(trade_returns: np.ndarray) -> float:
    if trade_returns.size == 0:
        return 0.0

    std = float(np.std(trade_returns))
    if std <= 1e-8:
        return 0.0
    return float(np.mean(trade_returns) / std * ANNUALIZATION_FACTOR)


def _compute_cagr_from_log_returns(log_returns: np.ndarray, periods_per_year: float = 24 * 365) -> float:
    if log_returns.size == 0:
        return 0.0

    total_log_return = float(np.sum(log_returns))
    periods = float(log_returns.size)
    if periods <= 0:
        return 0.0

    growth_multiple = float(np.exp(total_log_return))
    if growth_multiple <= 0:
        return 0.0

    return float(growth_multiple ** (periods_per_year / periods) - 1.0)


def _default_allocations(length: int, btc_pct: float = 100.0) -> list[dict[str, float]]:
    return [{"btc": btc_pct, "gold": 0.0, "silver": 0.0} for _ in range(max(length, 0))]


def _normalize_allocations(
    allocations: Optional[list[dict[str, float]]],
    length: int,
    default_btc_pct: float = 100.0,
) -> list[dict[str, float]]:
    normalized = list(allocations or [])
    if len(normalized) > length:
        normalized = normalized[:length]
    elif len(normalized) < length:
        normalized.extend(_default_allocations(length - len(normalized), btc_pct=default_btc_pct))
    return normalized


def _normalized_override_triplet(override: dict[str, float]) -> dict[str, float]:
    btc = float(override.get("btc", 0.0))
    gold = float(override.get("gold", 0.0))
    silver = float(override.get("silver", 0.0))
    total = btc + gold + silver
    if total <= 0:
        return {"btc": 100.0, "gold": 0.0, "silver": 0.0}
    scale = 100.0 / total
    return {"btc": btc * scale, "gold": gold * scale, "silver": silver * scale}


def _apply_year_overrides_to_allocations(
    allocations: list[dict[str, float]],
    timestamps: list[str],
    year_overrides: Optional[list[dict[str, float]]],
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    if not year_overrides:
        return allocations, []

    override_by_year: dict[int, dict[str, float]] = {}
    applied_overrides: list[dict[str, float]] = []
    for raw in year_overrides:
        try:
            year = int(raw.get("year"))
        except (TypeError, ValueError):
            continue
        normalized = _normalized_override_triplet(raw)
        override_by_year[year] = normalized
        applied_overrides.append({"year": year, **normalized})

    if not override_by_year:
        return allocations, []

    updated = []
    for index, allocation in enumerate(allocations):
        ts = timestamps[index] if index < len(timestamps) else ""
        year = None
        if ts:
            parsed = str(ts)[:4]
            if parsed.isdigit():
                year = int(parsed)

        if year is not None and year in override_by_year:
            updated.append(dict(override_by_year[year]))
        else:
            updated.append(dict(allocation))

    return updated, applied_overrides


_ASSET_RETURNS_CACHE: Optional[dict[str, dict[str, float]]] = None


def _timestamp_key(timestamp: str) -> str:
    value = str(timestamp or "").strip()
    if not value:
        return ""

    candidate = value.replace(" ", "T", 1)
    if candidate.endswith("Z"):
        candidate = f"{candidate[:-1]}+00:00"
    if len(candidate) >= 5 and candidate[-5] in ("+", "-") and candidate[-3] != ":":
        candidate = f"{candidate[:-2]}:{candidate[-2:]}"

    try:
        parsed = datetime.fromisoformat(candidate)
        return parsed.strftime("%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        return candidate[:19]


def _build_asset_returns_cache() -> dict[str, dict[str, float]]:
    btc_gold_path = DATA_DIR / "btc_gold_12h_aligned.csv"
    silver_path = CACHE_DIR / "yahoo_silver_daily.csv"
    if not btc_gold_path.exists() or not silver_path.exists():
        return {}

    with btc_gold_path.open("r", newline="", encoding="utf-8") as handle:
        btc_gold_rows = list(csv.DictReader(handle))
    with silver_path.open("r", newline="", encoding="utf-8") as handle:
        silver_rows = list(csv.DictReader(handle))

    if not btc_gold_rows or not silver_rows:
        return {}

    silver_dates = [row.get("timestamp", "")[:10] for row in silver_rows]
    silver_prices = [_safe_float(row.get("close"), 0.0) for row in silver_rows]
    silver_ptr = 0
    last_silver_price = silver_prices[0] if silver_prices else 0.0

    prev_btc = None
    prev_gold = None
    prev_silver = None
    returns_by_timestamp: dict[str, dict[str, float]] = {}

    for row in btc_gold_rows:
        timestamp = str(row.get("timestamp", ""))
        day = timestamp[:10]

        while silver_ptr < len(silver_dates) and silver_dates[silver_ptr] <= day:
            candidate = silver_prices[silver_ptr]
            if candidate > 0:
                last_silver_price = candidate
            silver_ptr += 1

        btc_price = _safe_float(row.get("btc_close"), 0.0)
        gold_price = _safe_float(row.get("gold_close"), 0.0)
        silver_price = last_silver_price

        btc_ret = (btc_price / prev_btc - 1.0) if prev_btc and btc_price > 0 else 0.0
        gold_ret = (gold_price / prev_gold - 1.0) if prev_gold and gold_price > 0 else 0.0
        silver_ret = (silver_price / prev_silver - 1.0) if prev_silver and silver_price > 0 else 0.0

        returns_by_timestamp[_timestamp_key(timestamp)] = {"btc": btc_ret, "gold": gold_ret, "silver": silver_ret}

        if btc_price > 0:
            prev_btc = btc_price
        if gold_price > 0:
            prev_gold = gold_price
        if silver_price > 0:
            prev_silver = silver_price

    return returns_by_timestamp


def _asset_returns_for_timestamps(timestamps: list[str]) -> list[dict[str, float]]:
    global _ASSET_RETURNS_CACHE
    if _ASSET_RETURNS_CACHE is None:
        _ASSET_RETURNS_CACHE = _build_asset_returns_cache()

    cache = _ASSET_RETURNS_CACHE or {}
    return [cache.get(_timestamp_key(str(ts)), {"btc": 0.0, "gold": 0.0, "silver": 0.0}) for ts in timestamps]


def _build_backtest_payload(
    strategy_name: str,
    in_sample_signals: np.ndarray,
    in_sample_realized_returns: np.ndarray,
    out_of_sample_signals: np.ndarray,
    out_of_sample_realized_returns: np.ndarray,
    in_sample_timestamps: Optional[List[str]] = None,
    out_of_sample_timestamps: Optional[List[str]] = None,
    in_sample_allocations: Optional[list[dict[str, float]]] = None,
    out_of_sample_allocations: Optional[list[dict[str, float]]] = None,
    applied_overrides: Optional[list[dict[str, float]]] = None,
) -> Dict[str, Any]:
    in_sample_trade_returns = in_sample_signals * in_sample_realized_returns
    out_of_sample_trade_returns = out_of_sample_signals * out_of_sample_realized_returns

    in_sample_curve = _equity_curve_from_returns(in_sample_trade_returns)
    out_of_sample_curve = _equity_curve_from_returns(out_of_sample_trade_returns)

    if in_sample_curve.size and out_of_sample_curve.size:
        full_curve = np.concatenate([in_sample_curve, in_sample_curve[-1] + out_of_sample_curve])
    elif in_sample_curve.size:
        full_curve = in_sample_curve
    else:
        full_curve = out_of_sample_curve

    in_sample_timestamps = in_sample_timestamps or []
    out_of_sample_timestamps = out_of_sample_timestamps or []
    full_timestamps = list(in_sample_timestamps) + list(out_of_sample_timestamps)

    in_sample_allocations = _normalize_allocations(in_sample_allocations, len(in_sample_timestamps))
    out_of_sample_allocations = _normalize_allocations(out_of_sample_allocations, len(out_of_sample_timestamps))
    full_allocations = list(in_sample_allocations) + list(out_of_sample_allocations)

    return {
        "strategy": strategy_name,
        "total_return": float(out_of_sample_trade_returns.sum()) if out_of_sample_trade_returns.size else 0.0,
        "win_rate": _compute_win_rate(out_of_sample_signals, out_of_sample_realized_returns),
        "sharpe": _compute_sharpe(out_of_sample_trade_returns),
        "cagr": _compute_cagr_from_log_returns(out_of_sample_trade_returns),
        "equity_curve": full_curve.tolist(),
        "equity_timestamps": full_timestamps,
        "equity_allocations": full_allocations,
        "in_sample_curve": in_sample_curve.tolist(),
        "in_sample_timestamps": list(in_sample_timestamps),
        "in_sample_allocations": list(in_sample_allocations),
        "out_of_sample_curve": out_of_sample_curve.tolist(),
        "out_of_sample_timestamps": list(out_of_sample_timestamps),
        "out_of_sample_allocations": list(out_of_sample_allocations),
        "in_sample_total_return": float(in_sample_trade_returns.sum()) if in_sample_trade_returns.size else 0.0,
        "out_of_sample_total_return": float(out_of_sample_trade_returns.sum()) if out_of_sample_trade_returns.size else 0.0,
        "out_of_sample_start_index": int(in_sample_curve.size),
        "applied_overrides": applied_overrides or [],
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _strategy_display_name(strategy_id: str) -> str:
    cleaned = strategy_id
    if "_" in cleaned and cleaned.split("_", 1)[0].isdigit():
        cleaned = cleaned.split("_", 1)[1]
    tokens = cleaned.split("_")
    pretty_tokens = []
    for token in tokens:
        if not token:
            continue
        if any(char.isupper() for char in token[1:]):
            pretty_tokens.append(token[0].upper() + token[1:])
        elif token.isupper():
            pretty_tokens.append(token)
        else:
            pretty_tokens.append(token.capitalize())
    return " ".join(pretty_tokens)


def _btc_only_allocations(length: int) -> list[dict[str, float]]:
    return _default_allocations(length, btc_pct=100.0)


class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run_backtest(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        year_overrides: Optional[list[dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Run backtest and return metrics + equity curve."""
        pass

    @abstractmethod
    def get_signal(self, ticker: str) -> int:
        """Get live signal (-1, 0, 1)."""
        pass


class LinearStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Linear Regression (AR)")
        self.model = None
        self.lags = 3

    def run_backtest(self, ticker: str, start_date=None, end_date=None, year_overrides=None) -> Dict[str, Any]:
        set_seed(42)
        df = load_ohlc_timeseries(ticker, "1h")
        df = df.with_columns((pl.col("close") / pl.col("close").shift(1)).log().alias("log_return")).drop_nulls()
        df = add_lags(df, "log_return", self.lags)
        timestamps = [str(value) for value in df["datetime"].to_list()] if "datetime" in df.columns else []

        if df.height < 50:
            raise ValueError(f"Not enough data to backtest {self.name} on {ticker}")

        features = [f"lag_{index + 1}" for index in range(self.lags)]
        X = to_tensor(df.select(features))
        y = to_tensor(df.select("target"))
        X_train, X_test = timeseries_split(X)
        y_train, y_test = timeseries_split(y)
        split_index = len(y_train)
        in_sample_timestamps = timestamps[:split_index]
        out_of_sample_timestamps = timestamps[split_index : split_index + len(y_test)]

        if len(X_test) == 0:
            raise ValueError(f"Out-of-sample split is empty for {self.name} on {ticker}")

        self.model = LinearModel(self.lags)
        batch_train_reg(self.model, X_train, X_test, y_train, y_test, no_epochs=50)

        self.model.eval()
        with torch.no_grad():
            train_preds = self.model(X_train).numpy().flatten()
            test_preds = self.model(X_test).numpy().flatten()

        in_sample_signals = np.sign(_to_1d(train_preds))
        out_of_sample_signals = np.sign(_to_1d(test_preds))
        in_sample_returns = _to_1d(y_train.numpy())
        out_of_sample_returns = _to_1d(y_test.numpy())
        in_sample_allocations = _btc_only_allocations(len(in_sample_timestamps))
        out_of_sample_allocations = _btc_only_allocations(len(out_of_sample_timestamps))

        return _build_backtest_payload(
            strategy_name=self.name,
            in_sample_signals=in_sample_signals,
            in_sample_realized_returns=in_sample_returns,
            out_of_sample_signals=out_of_sample_signals,
            out_of_sample_realized_returns=out_of_sample_returns,
            in_sample_timestamps=in_sample_timestamps,
            out_of_sample_timestamps=out_of_sample_timestamps,
            in_sample_allocations=in_sample_allocations,
            out_of_sample_allocations=out_of_sample_allocations,
        )

    def get_signal(self, ticker: str) -> int:
        return 1


class NonLinearStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Neural Network (Non-Linear)")
        self.model = None
        self.lags = 2

    def run_backtest(self, ticker: str, start_date=None, end_date=None, year_overrides=None) -> Dict[str, Any]:
        set_seed(42)
        df = load_ohlc_timeseries(ticker, "1h")
        df = df.with_columns((pl.col("close") / pl.col("close").shift(1)).log().alias("log_return")).drop_nulls()
        df = add_lags(df, "log_return", self.lags)
        timestamps = [str(value) for value in df["datetime"].to_list()] if "datetime" in df.columns else []

        if df.height < 50:
            raise ValueError(f"Not enough data to backtest {self.name} on {ticker}")

        features = [f"lag_{index + 1}" for index in range(self.lags)]
        X = to_tensor(df.select(features))
        y = to_tensor(df.select("target"))
        X_train, X_test = timeseries_split(X)
        y_train, y_test = timeseries_split(y)
        split_index = len(y_train)
        in_sample_timestamps = timestamps[:split_index]
        out_of_sample_timestamps = timestamps[split_index : split_index + len(y_test)]

        if len(X_test) == 0:
            raise ValueError(f"Out-of-sample split is empty for {self.name} on {ticker}")

        self.model = NonLinearModel(self.lags)
        batch_train_reg(self.model, X_train, X_test, y_train, y_test, no_epochs=50)

        self.model.eval()
        with torch.no_grad():
            train_preds = self.model(X_train).numpy().flatten()
            test_preds = self.model(X_test).numpy().flatten()

        in_sample_signals = np.sign(_to_1d(train_preds))
        out_of_sample_signals = np.sign(_to_1d(test_preds))
        in_sample_returns = _to_1d(y_train.numpy())
        out_of_sample_returns = _to_1d(y_test.numpy())
        in_sample_allocations = _btc_only_allocations(len(in_sample_timestamps))
        out_of_sample_allocations = _btc_only_allocations(len(out_of_sample_timestamps))

        return _build_backtest_payload(
            strategy_name=self.name,
            in_sample_signals=in_sample_signals,
            in_sample_realized_returns=in_sample_returns,
            out_of_sample_signals=out_of_sample_signals,
            out_of_sample_realized_returns=out_of_sample_returns,
            in_sample_timestamps=in_sample_timestamps,
            out_of_sample_timestamps=out_of_sample_timestamps,
            in_sample_allocations=in_sample_allocations,
            out_of_sample_allocations=out_of_sample_allocations,
        )

    def get_signal(self, ticker: str) -> int:
        return -1


class GARCHStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("GARCH Volatility Regime")

    def run_backtest(self, ticker: str, start_date=None, end_date=None, year_overrides=None) -> Dict[str, Any]:
        set_seed(42)
        df = load_ohlc_timeseries(ticker, "1h")
        timestamps_raw = [str(value) for value in df["datetime"].to_list()] if "datetime" in df.columns else []

        returns_pct = (df["close"] / df["close"].shift(1) - 1).drop_nulls().to_numpy() * 100
        log_returns = (df["close"] / df["close"].shift(1)).log().drop_nulls().to_numpy()
        timestamps = timestamps_raw[1 : 1 + len(log_returns)] if timestamps_raw else []

        if len(returns_pct) < 40:
            raise ValueError(f"Not enough data to backtest {self.name} on {ticker}")

        train_size = int(len(returns_pct) * 0.75)
        train_size = max(5, min(train_size, len(returns_pct) - 1))

        train_returns_pct = _to_1d(returns_pct[:train_size])
        test_returns_pct = _to_1d(returns_pct[train_size:])
        train_log_returns = _to_1d(log_returns[:train_size])
        test_log_returns = _to_1d(log_returns[train_size:])
        in_sample_timestamps = timestamps[:train_size]
        out_of_sample_timestamps = timestamps[train_size : train_size + len(test_log_returns)]

        if test_returns_pct.size == 0:
            raise ValueError(f"Out-of-sample split is empty for {self.name} on {ticker}")

        model = arch_model(train_returns_pct, vol="Garch", p=1, q=1, dist="t")
        fit_result = model.fit(disp="off")
        train_vol = _to_1d(fit_result.conditional_volatility)

        vol_threshold = float(np.median(train_vol)) if train_vol.size else 0.0

        in_sample_signals = np.zeros(train_returns_pct.size, dtype=float)
        for index in range(1, train_returns_pct.size):
            if train_vol[min(index, train_vol.size - 1)] < vol_threshold:
                in_sample_signals[index] = np.sign(train_returns_pct[index - 1])
            else:
                in_sample_signals[index] = -np.sign(train_returns_pct[index - 1])

        out_of_sample_vol = np.zeros(test_returns_pct.size, dtype=float)
        out_of_sample_vol[0] = train_vol[-1] if train_vol.size else float(np.std(train_returns_pct))
        alpha = 0.06
        beta = 0.93
        for index in range(1, test_returns_pct.size):
            out_of_sample_vol[index] = np.sqrt(
                alpha * (test_returns_pct[index - 1] ** 2) + beta * (out_of_sample_vol[index - 1] ** 2)
            )

        out_of_sample_signals = np.zeros(test_returns_pct.size, dtype=float)
        for index in range(1, test_returns_pct.size):
            if out_of_sample_vol[index] < vol_threshold:
                out_of_sample_signals[index] = np.sign(test_returns_pct[index - 1])
            else:
                out_of_sample_signals[index] = -np.sign(test_returns_pct[index - 1])
        in_sample_allocations = _btc_only_allocations(len(in_sample_timestamps))
        out_of_sample_allocations = _btc_only_allocations(len(out_of_sample_timestamps))

        return _build_backtest_payload(
            strategy_name=self.name,
            in_sample_signals=in_sample_signals,
            in_sample_realized_returns=train_log_returns,
            out_of_sample_signals=out_of_sample_signals,
            out_of_sample_realized_returns=test_log_returns,
            in_sample_timestamps=in_sample_timestamps,
            out_of_sample_timestamps=out_of_sample_timestamps,
            in_sample_allocations=in_sample_allocations,
            out_of_sample_allocations=out_of_sample_allocations,
        )

    def get_signal(self, ticker: str) -> int:
        return 0


class PrecomputedStrategy(BaseStrategy):
    def __init__(
        self,
        strategy_id: str,
        metrics: Dict[str, float],
        equity_curve: list[float],
        timestamps: Optional[list[str]] = None,
        allocations: Optional[list[dict[str, float]]] = None,
    ):
        super().__init__(_strategy_display_name(strategy_id))
        self.strategy_id = strategy_id
        self.metrics = metrics
        self.equity_curve = equity_curve
        self.timestamps = timestamps or []
        self.allocations = allocations or []

    def run_backtest(self, ticker: str, start_date=None, end_date=None, year_overrides=None) -> Dict[str, Any]:
        full_curve = list(self.equity_curve)
        if not full_curve:
            full_curve = [1.0]

        full_timestamps = list(self.timestamps)
        if len(full_timestamps) > len(full_curve):
            full_timestamps = full_timestamps[: len(full_curve)]
        elif len(full_timestamps) < len(full_curve):
            full_timestamps.extend([""] * (len(full_curve) - len(full_timestamps)))

        base_allocations = _normalize_allocations(self.allocations, len(full_curve))
        full_allocations, applied_overrides = _apply_year_overrides_to_allocations(
            base_allocations,
            full_timestamps,
            year_overrides,
        )

        has_override = len(applied_overrides) > 0
        if has_override:
            asset_returns = _asset_returns_for_timestamps(full_timestamps)
            if len(asset_returns) == len(full_allocations):
                equity = 1.0
                recalculated_curve: list[float] = []
                period_simple_returns: list[float] = []
                for allocation, asset_ret in zip(full_allocations, asset_returns):
                    period_ret = (
                        (allocation.get("btc", 0.0) / 100.0) * float(asset_ret.get("btc", 0.0))
                        + (allocation.get("gold", 0.0) / 100.0) * float(asset_ret.get("gold", 0.0))
                        + (allocation.get("silver", 0.0) / 100.0) * float(asset_ret.get("silver", 0.0))
                    )
                    period_simple_returns.append(period_ret)
                    equity *= (1.0 + period_ret)
                    recalculated_curve.append(equity)
                full_curve = recalculated_curve
            else:
                applied_overrides = []

        split_index = int(len(full_curve) * 0.8)
        split_index = min(max(split_index, 1), len(full_curve))
        in_sample_curve = full_curve[:split_index]
        out_of_sample_curve = full_curve[split_index:]
        in_sample_timestamps = full_timestamps[:split_index]
        out_of_sample_timestamps = full_timestamps[split_index:]
        in_sample_allocations = full_allocations[:split_index]
        out_of_sample_allocations = full_allocations[split_index:]

        if has_override and applied_overrides:
            if in_sample_curve:
                in_sample_total_return = float(in_sample_curve[-1] - 1.0)
            else:
                in_sample_total_return = 0.0

            if out_of_sample_curve and in_sample_curve:
                out_of_sample_total_return = float((out_of_sample_curve[-1] / in_sample_curve[-1]) - 1.0) if in_sample_curve[-1] != 0 else 0.0
            elif out_of_sample_curve:
                out_of_sample_total_return = float(out_of_sample_curve[-1] - 1.0)
            else:
                out_of_sample_total_return = 0.0

            total_return = float(full_curve[-1] - 1.0) if full_curve else 0.0
            oos_period_returns = []
            if out_of_sample_curve:
                prior = in_sample_curve[-1] if in_sample_curve else 1.0
                for point in out_of_sample_curve:
                    if prior != 0:
                        oos_period_returns.append((point / prior) - 1.0)
                    prior = point

            oos_period_returns_np = np.asarray(oos_period_returns, dtype=float)
            win_rate = float(np.mean(oos_period_returns_np > 0)) if oos_period_returns_np.size else 0.0
            oos_log_returns = np.log1p(np.clip(oos_period_returns_np, -0.999999, None))
            sharpe = _compute_sharpe(oos_log_returns)
            cagr = _compute_cagr_from_log_returns(oos_log_returns)
        else:
            total_return = self.metrics.get("total_return", 0.0)
            in_sample_total_return = self.metrics.get("in_sample_return", 0.0)
            out_of_sample_total_return = self.metrics.get("out_of_sample_return", 0.0)
            holdout_return = self.metrics.get("holdout_return", 0.0)
            if out_of_sample_total_return == 0.0 and holdout_return != 0.0:
                out_of_sample_total_return = holdout_return
            if out_of_sample_total_return == 0.0:
                out_of_sample_total_return = total_return

            win_rate = self.metrics.get("win_rate")
            if win_rate is None:
                win_rate = 0.0
            if float(win_rate) == 0.0 and self.equity_curve:
                inferred_win_rate = _equity_curve_win_rate(self.equity_curve)
                if inferred_win_rate is not None:
                    win_rate = inferred_win_rate

            sharpe = float(self.metrics.get("sharpe", 0.0))
            cagr = float(self.metrics.get("cagr", 0.0))

        return {
            "strategy": self.name,
            "total_return": float(total_return),
            "win_rate": float(win_rate),
            "sharpe": float(sharpe),
            "cagr": float(cagr),
            "equity_curve": full_curve,
            "equity_timestamps": full_timestamps,
            "equity_allocations": full_allocations,
            "in_sample_curve": in_sample_curve,
            "in_sample_timestamps": in_sample_timestamps,
            "in_sample_allocations": in_sample_allocations,
            "out_of_sample_curve": out_of_sample_curve,
            "out_of_sample_timestamps": out_of_sample_timestamps,
            "out_of_sample_allocations": out_of_sample_allocations,
            "in_sample_total_return": float(in_sample_total_return),
            "out_of_sample_total_return": float(out_of_sample_total_return),
            "out_of_sample_start_index": split_index,
            "applied_overrides": applied_overrides,
        }

    def get_signal(self, ticker: str) -> int:
        return 0


def _load_precomputed_strategy_curves() -> tuple[dict[str, list[float]], list[str]]:
    file_path = RESULTS_DIR / "equity_curves.csv"
    if not file_path.exists():
        return {}, []

    with file_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return {}, []

    strategy_columns = [column for column in rows[0].keys() if column != "timestamp"]
    curves: dict[str, list[float]] = {column: [] for column in strategy_columns}
    timestamps = [str(row.get("timestamp", "")) for row in rows]

    for row in rows:
        for strategy in strategy_columns:
            curves[strategy].append(_safe_float(row.get(strategy), 1.0))

    return curves, timestamps


def _default_precomputed_allocations(strategy_id: str, length: int) -> list[dict[str, float]]:
    if strategy_id == "00_no_trade":
        return _default_allocations(length, btc_pct=0.0)
    return _default_allocations(length, btc_pct=100.0)


def _load_rotation_allocations() -> tuple[dict[str, list[dict[str, float]]], list[str]]:
    weights_file = RESULTS_DIR / "breakout_rotation_weights.csv"
    if not weights_file.exists():
        return {}, []

    with weights_file.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return {}, []

    strategy_columns = [column for column in rows[0].keys() if column.endswith("_btc")]
    strategy_ids = sorted(column[: -len("_btc")] for column in strategy_columns)
    allocations: dict[str, list[dict[str, float]]] = {strategy_id: [] for strategy_id in strategy_ids}
    timestamps = [str(row.get("timestamp", "")) for row in rows]

    for row in rows:
        for strategy_id in strategy_ids:
            allocations[strategy_id].append(
                {
                    "btc": _safe_float(row.get(f"{strategy_id}_btc")) * 100.0,
                    "gold": _safe_float(row.get(f"{strategy_id}_gold")) * 100.0,
                    "silver": _safe_float(row.get(f"{strategy_id}_silver")) * 100.0,
                }
            )

    return allocations, timestamps


def _load_rotation_equity_curves() -> tuple[dict[str, list[float]], list[str]]:
    weights_file = RESULTS_DIR / "breakout_rotation_weights.csv"
    btc_gold_file = DATA_DIR / "btc_gold_12h_aligned.csv"
    silver_file = CACHE_DIR / "yahoo_silver_daily.csv"

    if not weights_file.exists() or not btc_gold_file.exists() or not silver_file.exists():
        return {}, []

    with weights_file.open("r", newline="", encoding="utf-8") as handle:
        weight_rows = list(csv.DictReader(handle))
    with btc_gold_file.open("r", newline="", encoding="utf-8") as handle:
        btc_gold_rows = list(csv.DictReader(handle))
    with silver_file.open("r", newline="", encoding="utf-8") as handle:
        silver_rows = list(csv.DictReader(handle))

    if not weight_rows or not btc_gold_rows or not silver_rows:
        return {}, []

    strategy_columns = [column for column in weight_rows[0].keys() if column.endswith("_btc")]
    strategy_ids = sorted(column[: -len("_btc")] for column in strategy_columns)
    curves: dict[str, list[float]] = {strategy_id: [] for strategy_id in strategy_ids}
    timestamps: list[str] = []

    silver_dates = [row.get("timestamp", "")[:10] for row in silver_rows]
    silver_prices = [_safe_float(row.get("close"), 0.0) for row in silver_rows]
    silver_ptr = 0
    last_silver_price = silver_prices[0] if silver_prices else 0.0

    equity = {strategy_id: 1.0 for strategy_id in strategy_ids}
    prev_btc = None
    prev_gold = None
    prev_silver = None
    prev_weight_row: dict[str, str] | None = None

    row_count = min(len(weight_rows), len(btc_gold_rows))
    for index in range(row_count):
        weight_row = weight_rows[index]
        price_row = btc_gold_rows[index]
        timestamps.append(str(weight_row.get("timestamp", "")))

        day = (weight_row.get("timestamp", "") or "")[:10]
        while silver_ptr < len(silver_dates) and silver_dates[silver_ptr] <= day:
            candidate = silver_prices[silver_ptr]
            if candidate > 0:
                last_silver_price = candidate
            silver_ptr += 1

        btc_price = _safe_float(price_row.get("btc_close"), 0.0)
        gold_price = _safe_float(price_row.get("gold_close"), 0.0)
        silver_price = last_silver_price

        btc_ret = (btc_price / prev_btc - 1.0) if prev_btc and btc_price > 0 else 0.0
        gold_ret = (gold_price / prev_gold - 1.0) if prev_gold and gold_price > 0 else 0.0
        silver_ret = (silver_price / prev_silver - 1.0) if prev_silver and silver_price > 0 else 0.0

        weight_source = prev_weight_row if prev_weight_row is not None else weight_row
        for strategy_id in strategy_ids:
            w_btc = _safe_float(weight_source.get(f"{strategy_id}_btc"), 0.0)
            w_gold = _safe_float(weight_source.get(f"{strategy_id}_gold"), 0.0)
            w_silver = _safe_float(weight_source.get(f"{strategy_id}_silver"), 0.0)
            period_ret = (w_btc * btc_ret) + (w_gold * gold_ret) + (w_silver * silver_ret)
            equity[strategy_id] *= 1.0 + period_ret
            curves[strategy_id].append(equity[strategy_id])

        if btc_price > 0:
            prev_btc = btc_price
        if gold_price > 0:
            prev_gold = gold_price
        if silver_price > 0:
            prev_silver = silver_price
        prev_weight_row = weight_row

    return curves, timestamps


def _load_precomputed_strategies() -> dict[str, BaseStrategy]:
    metrics_file = RESULTS_DIR / "ordered_metrics.csv"
    if not metrics_file.exists():
        return {}

    curves, timestamps = _load_precomputed_strategy_curves()
    strategies: dict[str, BaseStrategy] = {}

    with metrics_file.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            strategy_id = row.get("strategy", "").strip()
            if not strategy_id:
                continue

            metrics = {
                "total_return": _safe_float(row.get("total_return")),
                "win_rate": _safe_float(row.get("win_rate")),
                "sharpe": _safe_float(row.get("sharpe")),
                "cagr": _safe_float(row.get("cagr")),
                "in_sample_return": _safe_float(row.get("in_sample_return")),
                "out_of_sample_return": _safe_float(row.get("out_of_sample_return")),
                "holdout_return": _safe_float(row.get("holdout_return")),
            }

            strategies[strategy_id] = PrecomputedStrategy(
                strategy_id=strategy_id,
                metrics=metrics,
                equity_curve=curves.get(strategy_id, []),
                timestamps=timestamps,
                allocations=_default_precomputed_allocations(strategy_id, len(curves.get(strategy_id, []))),
            )

    return strategies


def _load_rotation_precomputed_strategies(existing_ids: set[str] | None = None) -> dict[str, BaseStrategy]:
    metrics_file = RESULTS_DIR / "breakout_rotation_ranked.csv"
    if not metrics_file.exists():
        return {}

    existing_ids = existing_ids or set()
    curves, timestamps = _load_rotation_equity_curves()
    rotation_allocations, allocation_timestamps = _load_rotation_allocations()
    selected_timestamps = allocation_timestamps if allocation_timestamps else timestamps
    strategies: dict[str, BaseStrategy] = {}

    with metrics_file.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            strategy_id = row.get("strategy", "").strip()
            if not strategy_id or strategy_id in existing_ids:
                continue

            metrics = {
                "total_return": _safe_float(row.get("total_return")),
                "win_rate": 0.0,
                "sharpe": _safe_float(row.get("sharpe")),
                "cagr": _safe_float(row.get("cagr")),
                "in_sample_return": _safe_float(row.get("total_return")),
                "out_of_sample_return": _safe_float(row.get("oos_return")),
                "holdout_return": 0.0,
            }

            strategies[strategy_id] = PrecomputedStrategy(
                strategy_id=strategy_id,
                metrics=metrics,
                equity_curve=curves.get(strategy_id, []),
                timestamps=selected_timestamps,
                allocations=rotation_allocations.get(strategy_id, _default_precomputed_allocations(strategy_id, len(curves.get(strategy_id, [])))),
            )

    return strategies


def _load_strategy_performance_meta() -> dict[str, dict[str, float]]:
    meta: dict[str, dict[str, float | None]] = {}

    ordered_metrics_file = RESULTS_DIR / "ordered_metrics.csv"
    if ordered_metrics_file.exists():
        with ordered_metrics_file.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                strategy_id = (row.get("strategy") or "").strip()
                if not strategy_id:
                    continue
                meta[strategy_id] = {
                    "return_pct": _safe_float(row.get("return_pct")),
                    "cagr": _safe_float(row.get("cagr")),
                    "win_rate": _safe_float(row.get("win_rate")),
                }

    rotation_metrics_file = RESULTS_DIR / "breakout_rotation_ranked.csv"
    if rotation_metrics_file.exists():
        with rotation_metrics_file.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                strategy_id = (row.get("strategy") or "").strip()
                if not strategy_id:
                    continue
                meta[strategy_id] = {
                    "return_pct": _safe_float(row.get("return_pct")),
                    "cagr": _safe_float(row.get("cagr")),
                    "win_rate": None,
                }

    return meta


def _equity_curve_win_rate(equity_curve: list[float]) -> float | None:
    if len(equity_curve) < 2:
        return None

    deltas = np.diff(np.asarray(equity_curve, dtype=float))
    traded = deltas[deltas != 0]
    if traded.size == 0:
        return 0.0
    return float(np.mean(traded > 0))


def get_ranked_strategy_items() -> list[dict[str, Any]]:
    meta = _load_strategy_performance_meta()

    items: list[dict[str, Any]] = []
    for strategy_id, strategy in STRATEGIES.items():
        row = meta.get(strategy_id, {})
        return_pct = row.get("return_pct")
        cagr = row.get("cagr")
        win_rate = row.get("win_rate")

        if win_rate is None and isinstance(strategy, PrecomputedStrategy):
            explicit_win_rate = strategy.metrics.get("win_rate")
            if explicit_win_rate is not None:
                win_rate = float(explicit_win_rate)
            if (win_rate is None or win_rate == 0.0) and strategy.equity_curve:
                win_rate = _equity_curve_win_rate(strategy.equity_curve)

        if return_pct is None:
            display_name = strategy.name
        else:
            display_name = f"{strategy.name} - {return_pct:.2f}%"

        items.append(
            {
                "id": strategy_id,
                "name": display_name,
                "return_pct": return_pct,
                "cagr": cagr,
                "win_rate": win_rate,
                "supports_allocation_override": isinstance(strategy, PrecomputedStrategy),
            }
        )

    items.sort(key=lambda item: item["return_pct"] if item["return_pct"] is not None else float("-inf"), reverse=True)
    return items


STRATEGIES: dict[str, BaseStrategy] = {
    "linear": LinearStrategy(),
    "nonlinear": NonLinearStrategy(),
    "garch": GARCHStrategy(),
}
STRATEGIES.update(_load_precomputed_strategies())
STRATEGIES.update(_load_rotation_precomputed_strategies(set(STRATEGIES.keys())))
