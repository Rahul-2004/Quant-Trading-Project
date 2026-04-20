import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api", tags=["findings"])

RESULTS_DIR = Path(__file__).resolve().parents[2] / "reserach" / "results"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_DIR = Path(__file__).resolve().parents[2] / "reserach" / "cache"
MAX_TIME_POINTS = 360


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_csv(filename: str) -> list[dict[str, str]]:
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required findings file: {filename}")

    with file_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _downsample_rows(rows: list[dict[str, str]], max_points: int = MAX_TIME_POINTS) -> list[dict[str, str]]:
    if len(rows) <= max_points:
        return rows

    step = math.ceil(len(rows) / max_points)
    sampled = rows[::step]

    # Always keep final point for continuity.
    if sampled[-1] != rows[-1]:
        sampled.append(rows[-1])

    return sampled


def _build_ordered_metrics() -> list[dict[str, Any]]:
    rows = _read_csv("ordered_metrics.csv")
    metrics = []
    for row in rows:
        metrics.append(
            {
                "strategy": row["strategy"],
                "return_pct": _to_float(row.get("return_pct")),
                "sharpe": _to_float(row.get("sharpe")),
                "max_drawdown": _to_float(row.get("max_drawdown")),
                "win_rate": _to_float(row.get("win_rate")),
                "cagr": _to_float(row.get("cagr")),
                "robustness_score": _to_float(row.get("robustness_score")),
                "walk_forward_consistency": _to_float(row.get("walk_forward_consistency")),
            }
        )

    metrics.sort(key=lambda item: item["return_pct"], reverse=True)
    return metrics


def _build_benchmark_comparison() -> list[dict[str, Any]]:
    rows = _read_csv("benchmark_comparison.csv")
    payload = []
    for row in rows:
        payload.append(
            {
                "strategy": row["strategy"],
                "return_pct": _to_float(row.get("return_pct")),
                "sharpe": _to_float(row.get("sharpe")),
                "excess_return_vs_buy_hold_pct": _to_float(row.get("excess_return_vs_buy_hold_pct")),
                "excess_return_vs_simple_baseline_pct": _to_float(row.get("excess_return_vs_simple_baseline_pct")),
            }
        )

    payload.sort(key=lambda item: item["return_pct"], reverse=True)
    return payload


def _build_equity_curves() -> dict[str, Any]:
    rows = _read_csv("equity_curves.csv")
    if not rows:
        return {"strategies": [], "series": []}

    columns = [column for column in rows[0].keys() if column != "timestamp"]
    sampled_rows = _downsample_rows(rows)

    series = []
    for row in sampled_rows:
        point = {"timestamp": row["timestamp"]}
        for column in columns:
            point[column] = _to_float(row.get(column))
        series.append(point)

    return {"strategies": columns, "series": series, "point_count": len(series), "full_point_count": len(rows)}


def _build_robustness_checks() -> list[dict[str, Any]]:
    rows = _read_csv("robustness_checks.csv")
    payload = []
    for row in rows:
        payload.append(
            {
                "strategy": row["strategy"],
                "variant": row["variant"],
                "perturb_factor": _to_float(row.get("perturb_factor")),
                "return_pct": _to_float(row.get("return_pct")),
            }
        )

    payload.sort(key=lambda item: (item["strategy"], item["perturb_factor"]))
    return payload


def _build_validation_splits() -> list[dict[str, Any]]:
    rows = _read_csv("validation_splits.csv")
    payload = []
    for row in rows:
        payload.append(
            {
                "strategy": row["strategy"],
                "split": row["split"],
                "return_pct": _to_float(row.get("return_pct")),
                "sharpe": _to_float(row.get("sharpe")),
                "max_drawdown": _to_float(row.get("max_drawdown")),
            }
        )

    payload.sort(key=lambda item: (item["strategy"], item["split"]))
    return payload


def _build_walk_forward_windows() -> list[dict[str, Any]]:
    rows = _read_csv("walk_forward_windows.csv")
    payload = []
    for row in rows:
        payload.append(
            {
                "strategy": row["strategy"],
                "window_id": int(_to_float(row.get("window_id"), 0.0)),
                "oos_return_pct": _to_float(row.get("oos_return_pct")),
                "oos_sharpe": _to_float(row.get("oos_sharpe")),
                "oos_max_drawdown": _to_float(row.get("oos_max_drawdown")),
            }
        )

    payload.sort(key=lambda item: (item["strategy"], item["window_id"]))
    return payload


def _build_trade_logs_summary() -> dict[str, Any]:
    rows = _read_csv("trade_logs.csv")

    pnl_values = []
    strategy_stats = defaultdict(lambda: {"trades": 0, "sum_net_pnl": 0.0, "wins": 0, "sum_holding_hours": 0.0})
    exit_reason_counts = defaultdict(int)

    for row in rows:
        strategy = row.get("strategy", "unknown")
        net_pnl = _to_float(row.get("net_pnl"))
        holding_hours = _to_float(row.get("holding_hours"))
        pnl_values.append(net_pnl)

        stats = strategy_stats[strategy]
        stats["trades"] += 1
        stats["sum_net_pnl"] += net_pnl
        stats["sum_holding_hours"] += holding_hours
        if net_pnl > 0:
            stats["wins"] += 1

        exit_reason = row.get("exit_reason") or "unknown"
        exit_reason_counts[exit_reason] += 1

    per_strategy = []
    for strategy, stats in strategy_stats.items():
        trades = stats["trades"]
        per_strategy.append(
            {
                "strategy": strategy,
                "trades": trades,
                "avg_net_pnl": (stats["sum_net_pnl"] / trades) if trades else 0.0,
                "total_net_pnl": stats["sum_net_pnl"],
                "win_rate": (stats["wins"] / trades) if trades else 0.0,
                "avg_holding_hours": (stats["sum_holding_hours"] / trades) if trades else 0.0,
            }
        )

    per_strategy.sort(key=lambda item: item["total_net_pnl"], reverse=True)

    histogram = []
    if pnl_values:
        min_pnl = min(pnl_values)
        max_pnl = max(pnl_values)
        if math.isclose(min_pnl, max_pnl):
            histogram = [{"bin_label": f"{min_pnl:.3f}", "count": len(pnl_values), "midpoint": min_pnl}]
        else:
            bin_count = 20
            bin_width = (max_pnl - min_pnl) / bin_count
            counts = [0 for _ in range(bin_count)]
            for pnl in pnl_values:
                index = min(int((pnl - min_pnl) / bin_width), bin_count - 1)
                counts[index] += 1

            for index, count in enumerate(counts):
                left = min_pnl + index * bin_width
                right = left + bin_width
                histogram.append(
                    {
                        "bin_label": f"{left:.3f}..{right:.3f}",
                        "count": count,
                        "midpoint": (left + right) / 2,
                    }
                )

    exit_reason_distribution = [
        {"exit_reason": reason, "count": count}
        for reason, count in sorted(exit_reason_counts.items(), key=lambda item: item[1], reverse=True)
    ]

    return {
        "strategy_stats": per_strategy,
        "pnl_histogram": histogram,
        "exit_reason_distribution": exit_reason_distribution,
    }


def _build_context_summary() -> dict[str, Any]:
    rows = _read_csv("context_logs.csv")

    regime_counts = defaultdict(int)
    strategy_regime_counts = defaultdict(lambda: defaultdict(int))
    strategy_context_sum = defaultdict(
        lambda: {
            "count": 0,
            "entry_trend_strength": 0.0,
            "entry_volatility_pctile": 0.0,
            "entry_volume_pctile": 0.0,
            "entry_btc_gold_ratio_z": 0.0,
        }
    )

    for row in rows:
        strategy = row.get("strategy", "unknown")
        regime = row.get("entry_regime_label") or "unknown"

        regime_counts[regime] += 1
        strategy_regime_counts[strategy][regime] += 1

        stats = strategy_context_sum[strategy]
        stats["count"] += 1
        stats["entry_trend_strength"] += _to_float(row.get("entry_trend_strength"))
        stats["entry_volatility_pctile"] += _to_float(row.get("entry_volatility_pctile"))
        stats["entry_volume_pctile"] += _to_float(row.get("entry_volume_pctile"))
        stats["entry_btc_gold_ratio_z"] += _to_float(row.get("entry_btc_gold_ratio_z"))

    regime_distribution = [
        {"regime": regime, "count": count}
        for regime, count in sorted(regime_counts.items(), key=lambda item: item[1], reverse=True)
    ]

    per_strategy_regimes = []
    for strategy, counts in strategy_regime_counts.items():
        payload = {"strategy": strategy}
        payload.update(counts)
        per_strategy_regimes.append(payload)

    per_strategy_context = []
    for strategy, sums in strategy_context_sum.items():
        count = sums["count"]
        if not count:
            continue
        per_strategy_context.append(
            {
                "strategy": strategy,
                "avg_entry_trend_strength": sums["entry_trend_strength"] / count,
                "avg_entry_volatility_pctile": sums["entry_volatility_pctile"] / count,
                "avg_entry_volume_pctile": sums["entry_volume_pctile"] / count,
                "avg_entry_btc_gold_ratio_z": sums["entry_btc_gold_ratio_z"] / count,
            }
        )

    per_strategy_context.sort(key=lambda item: item["avg_entry_trend_strength"], reverse=True)

    return {
        "regime_distribution": regime_distribution,
        "strategy_regime_distribution": per_strategy_regimes,
        "strategy_context_averages": per_strategy_context,
    }


def _build_breakout_rotation_ranked() -> list[dict[str, Any]]:
    rows = _read_csv("breakout_rotation_ranked.csv")
    payload = []
    for row in rows:
        payload.append(
            {
                "strategy": row["strategy"],
                "return_pct": _to_float(row.get("return_pct")),
                "sharpe": _to_float(row.get("sharpe")),
                "max_drawdown": _to_float(row.get("max_drawdown")),
                "oos_return_pct": _to_float(row.get("oos_return_pct")),
                "oos_sharpe": _to_float(row.get("oos_sharpe")),
                "switches_per_year": _to_float(row.get("switches_per_year")),
            }
        )

    payload.sort(key=lambda item: item["return_pct"], reverse=True)
    return payload


def _build_breakout_rotation_results() -> list[dict[str, Any]]:
    rows = _read_csv("breakout_rotation_results.csv")
    payload = []
    for row in rows:
        payload.append(
            {
                "strategy": row["strategy"],
                "cagr": _to_float(row.get("cagr")),
                "return_pct": _to_float(row.get("return_pct")),
                "max_drawdown": _to_float(row.get("max_drawdown")),
                "time_in_btc_pct": _to_float(row.get("time_in_btc_pct")),
                "time_in_gold_silver_pct": _to_float(row.get("time_in_gold_silver_pct")),
                "time_in_cash_pct": _to_float(row.get("time_in_cash_pct")),
            }
        )

    payload.sort(key=lambda item: item["return_pct"], reverse=True)
    return payload


def _build_breakout_rotation_weights() -> dict[str, Any]:
    rows = _read_csv("breakout_rotation_weights.csv")
    if not rows:
        return {"strategies": [], "series": []}

    columns = [column for column in rows[0].keys() if column != "timestamp"]
    sampled_rows = _downsample_rows(rows)

    strategies = set()
    for column in columns:
        if column.endswith("_btc"):
            strategies.add(column[: -len("_btc")])

    series = []
    for row in sampled_rows:
        point = {"timestamp": row["timestamp"]}
        for column in columns:
            point[column] = _to_float(row.get(column))
        series.append(point)

    return {
        "strategies": sorted(strategies),
        "series": series,
        "point_count": len(series),
        "full_point_count": len(rows),
    }


def _build_rotation_equity_series() -> dict[str, Any]:
    """Reconstruct rotation strategy equity curves from weights and underlying asset prices."""
    weight_rows = _read_csv("breakout_rotation_weights.csv")
    if not weight_rows:
        return {"strategies": [], "series": []}

    btc_gold_path = DATA_DIR / "btc_gold_12h_aligned.csv"
    silver_path = CACHE_DIR / "yahoo_silver_daily.csv"

    if not btc_gold_path.exists() or not silver_path.exists():
        return {"strategies": [], "series": []}

    with btc_gold_path.open("r", newline="", encoding="utf-8") as handle:
        btc_gold_rows = list(csv.DictReader(handle))

    with silver_path.open("r", newline="", encoding="utf-8") as handle:
        silver_rows = list(csv.DictReader(handle))

    if not btc_gold_rows or not silver_rows:
        return {"strategies": [], "series": []}

    strategy_columns = [column for column in weight_rows[0].keys() if column.endswith("_btc")]
    strategies = sorted(column[: -len("_btc")] for column in strategy_columns)

    silver_dates = [row.get("timestamp", "")[:10] for row in silver_rows]
    silver_prices = [_to_float(row.get("close"), 0.0) for row in silver_rows]
    silver_ptr = 0
    last_silver_price = silver_prices[0] if silver_prices else 0.0

    # Base index starts at 1 and compounds by period returns.
    equity = {strategy: 1.0 for strategy in strategies}
    series: list[dict[str, Any]] = []

    prev_btc = None
    prev_gold = None
    prev_silver = None
    prev_weight_row: dict[str, str] | None = None

    row_count = min(len(weight_rows), len(btc_gold_rows))
    for index in range(row_count):
        weight_row = weight_rows[index]
        price_row = btc_gold_rows[index]

        timestamp = weight_row.get("timestamp", "")
        day = timestamp[:10]

        while silver_ptr < len(silver_dates) and silver_dates[silver_ptr] <= day:
            candidate = silver_prices[silver_ptr]
            if candidate > 0:
                last_silver_price = candidate
            silver_ptr += 1

        btc_price = _to_float(price_row.get("btc_close"), 0.0)
        gold_price = _to_float(price_row.get("gold_close"), 0.0)
        silver_price = last_silver_price

        btc_ret = (btc_price / prev_btc - 1.0) if prev_btc and btc_price > 0 else 0.0
        gold_ret = (gold_price / prev_gold - 1.0) if prev_gold and gold_price > 0 else 0.0
        silver_ret = (silver_price / prev_silver - 1.0) if prev_silver and silver_price > 0 else 0.0

        point: dict[str, Any] = {"timestamp": timestamp}

        weight_source = prev_weight_row if prev_weight_row is not None else weight_row

        for strategy in strategies:
            w_btc = _to_float(weight_source.get(f"{strategy}_btc"), 0.0)
            w_gold = _to_float(weight_source.get(f"{strategy}_gold"), 0.0)
            w_silver = _to_float(weight_source.get(f"{strategy}_silver"), 0.0)
            # cash weight implied 0 return.
            period_ret = (w_btc * btc_ret) + (w_gold * gold_ret) + (w_silver * silver_ret)
            equity[strategy] *= 1.0 + period_ret
            point[strategy] = equity[strategy]

        series.append(point)

        if btc_price > 0:
            prev_btc = btc_price
        if gold_price > 0:
            prev_gold = gold_price
        if silver_price > 0:
            prev_silver = silver_price
        prev_weight_row = weight_row

    return {"strategies": strategies, "series": series}


def _extract_investment_reference_from_series(
    series: list[dict[str, Any]], strategies: list[str]
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    latest_equity_by_strategy: dict[str, float] = {}
    start_equity_by_year: dict[str, dict[str, float]] = {}

    if not series:
        return latest_equity_by_strategy, start_equity_by_year

    for row in series:
        timestamp = row.get("timestamp", "")
        if len(timestamp) < 4:
            continue
        year = timestamp[:4]
        if year not in start_equity_by_year:
            start_equity_by_year[year] = {}
            for strategy in strategies:
                start_equity_by_year[year][strategy] = _to_float(row.get(strategy), 1.0)

    last_row = series[-1]
    for strategy in strategies:
        latest_equity_by_strategy[strategy] = _to_float(last_row.get(strategy), 1.0)

    return latest_equity_by_strategy, start_equity_by_year


def _build_investment_reference() -> dict[str, Any]:
    rows = _read_csv("equity_curves.csv")
    if not rows:
        return {
            "years": [],
            "latest_timestamp": None,
            "latest_equity_by_strategy": {},
            "start_equity_by_year": {},
        }

    base_strategies = [column for column in rows[0].keys() if column != "timestamp"]
    base_latest, base_starts = _extract_investment_reference_from_series(rows, base_strategies)

    rotation = _build_rotation_equity_series()
    rotation_latest, rotation_starts = _extract_investment_reference_from_series(
        rotation.get("series", []), rotation.get("strategies", [])
    )

    latest_equity_by_strategy = dict(base_latest)
    latest_equity_by_strategy.update(rotation_latest)

    start_equity_by_year: dict[str, dict[str, float]] = {}
    for year, mapping in base_starts.items():
        start_equity_by_year[year] = dict(mapping)
    for year, mapping in rotation_starts.items():
        existing = start_equity_by_year.setdefault(year, {})
        existing.update(mapping)

    return {
        "years": sorted(start_equity_by_year.keys()),
        "latest_timestamp": rows[-1].get("timestamp"),
        "latest_equity_by_strategy": latest_equity_by_strategy,
        "start_equity_by_year": start_equity_by_year,
    }


@router.get("/findings")
def get_findings() -> dict[str, Any]:
    """Serve all research findings in chart-ready form for the frontend dashboard."""
    try:
        ordered_metrics = _build_ordered_metrics()

        return {
            "ordered_metrics": ordered_metrics,
            "benchmark_comparison": _build_benchmark_comparison(),
            "equity_curves": _build_equity_curves(),
            "robustness_checks": _build_robustness_checks(),
            "validation_splits": _build_validation_splits(),
            "walk_forward_windows": _build_walk_forward_windows(),
            "trade_logs_summary": _build_trade_logs_summary(),
            "context_summary": _build_context_summary(),
            "breakout_rotation_ranked": _build_breakout_rotation_ranked(),
            "breakout_rotation_results": _build_breakout_rotation_results(),
            "breakout_rotation_weights": _build_breakout_rotation_weights(),
            "investment_reference": _build_investment_reference(),
            "strategy_options": [item["strategy"] for item in ordered_metrics],
        }
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:  # pragma: no cover - defensive API guard
        raise HTTPException(status_code=500, detail=f"Failed to build findings payload: {error}") from error
