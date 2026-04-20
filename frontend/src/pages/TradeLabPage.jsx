import React, { useEffect, useMemo, useState } from 'react';
import { DollarSign, Play, TrendingUp } from 'lucide-react';

import BacktestChart from '../components/BacktestChart';
import MetricsTable from '../components/MetricsTable';
import { executeTrade, runBacktest } from '../services/api';

const TradeLabPage = ({ strategies, tickers, loading, error }) => {
    const [selectedStrategy, setSelectedStrategy] = useState('');
    const [selectedTicker, setSelectedTicker] = useState('BTCUSDT');
    const [backtestResults, setBacktestResults] = useState(null);
    const [loadingBacktest, setLoadingBacktest] = useState(false);
    const [tradeStatus, setTradeStatus] = useState(null);
    const [selectedYearPoint, setSelectedYearPoint] = useState(null);

    const strategyOptions = strategies || [];
    const activeStrategy = selectedStrategy || strategyOptions[0]?.id || '';
    const activeTicker = selectedTicker || tickers?.[0] || '';
    const activeStrategyMeta = useMemo(
        () => strategyOptions.find((item) => item.id === activeStrategy),
        [strategyOptions, activeStrategy],
    );
    const supportsAllocationOverride = Boolean(activeStrategyMeta?.supports_allocation_override);

    useEffect(() => {
        setSelectedYearPoint(null);
    }, [activeStrategy, supportsAllocationOverride]);

    const handleRunBacktest = async (yearOverrides = []) => {
        if (!activeStrategy || !activeTicker) return;

        setLoadingBacktest(true);
        setBacktestResults(null);

        try {
            const results = await runBacktest(activeStrategy, activeTicker, undefined, undefined, yearOverrides);
            setBacktestResults(results);
        } catch (requestError) {
            alert(`Backtest failed: ${requestError.message}`);
        } finally {
            setLoadingBacktest(false);
        }
    };

    const handleSelectYearPoint = (year, allocation) => {
        const alloc = allocation || { btc: 100, gold: 0, silver: 0 };
        setSelectedYearPoint({
            year,
            btc: Number(alloc.btc ?? 0),
            gold: Number(alloc.gold ?? 0),
            silver: Number(alloc.silver ?? 0),
        });
    };

    const updateSelectedAllocation = (field, value) => {
        setSelectedYearPoint((current) => {
            if (!current) return current;
            return { ...current, [field]: Number(value) };
        });
    };
    const selectedAllocationTotal = selectedYearPoint
        ? Number(selectedYearPoint.btc || 0) + Number(selectedYearPoint.gold || 0) + Number(selectedYearPoint.silver || 0)
        : 0;

    const handleApplyYearOverride = async () => {
        if (!selectedYearPoint) return;
        if (!supportsAllocationOverride) {
            alert('Allocation override is only available for precomputed multi-asset strategies.');
            return;
        }

        const rawBtc = Number(selectedYearPoint.btc || 0);
        const rawGold = Number(selectedYearPoint.gold || 0);
        const rawSilver = Number(selectedYearPoint.silver || 0);
        const total = rawBtc + rawGold + rawSilver;
        if (total <= 0) {
            alert('Allocation sum must be greater than 0.');
            return;
        }

        const scale = 100 / total;
        const override = {
            year: Number(selectedYearPoint.year),
            btc: rawBtc * scale,
            gold: rawGold * scale,
            silver: rawSilver * scale,
        };
        await handleRunBacktest([override]);
    };

    const handleExecuteTrade = async (side) => {
        if (!activeStrategy || !activeTicker) return;

        try {
            const result = await executeTrade({
                ticker: activeTicker,
                side,
                quantity: 0.01,
                strategy_id: activeStrategy,
            });

            setTradeStatus({
                success: true,
                message: `Trade Executed: ${result.message} (ID: ${result.order_id})`,
            });
        } catch (requestError) {
            setTradeStatus({ success: false, message: `Trade failed: ${requestError.message}` });
        }

        setTimeout(() => setTradeStatus(null), 5000);
    };

    if (loading) {
        return <div className="card p-6 text-gray-300">Loading trading controls...</div>;
    }

    if (error) {
        return (
            <div className="card p-6 border border-rose-800 bg-rose-950/30">
                <h2 className="text-lg font-semibold text-rose-300">Trading controls unavailable</h2>
                <p className="mt-2 text-sm text-rose-200/80">{error}</p>
            </div>
        );
    }

    return (
        <section className="space-y-8">
            <div className="card">
                <div className="p-6 border-b border-gray-700 bg-gray-800/40">
                    <h2 className="text-lg font-semibold text-gray-100">Trade Lab</h2>
                    <p className="mt-1 text-sm text-gray-400">Run ad-hoc backtests and simulate order execution.</p>
                </div>

                <div className="p-6 grid grid-cols-1 md:grid-cols-4 gap-6 items-end">
                    <div>
                        <label className="block text-sm font-medium text-gray-400 mb-1">Strategy</label>
                        <select value={activeStrategy} onChange={(event) => setSelectedStrategy(event.target.value)} className="input-field">
                            {(strategies || []).map((strategy) => (
                                <option key={strategy.id} value={strategy.id}>
                                    {strategy.name}
                                </option>
                            ))}
                        </select>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-400 mb-1">Ticker</label>
                        <select value={activeTicker} onChange={(event) => setSelectedTicker(event.target.value)} className="input-field">
                            {(tickers || []).map((ticker) => (
                                <option key={ticker} value={ticker}>
                                    {ticker}
                                </option>
                            ))}
                        </select>
                    </div>

                    <div>
                        <button
                            onClick={() => handleRunBacktest()}
                            disabled={loadingBacktest}
                            className={`w-full btn-primary ${loadingBacktest ? 'opacity-70 cursor-wait' : ''}`}
                        >
                            {loadingBacktest ? (
                                <span className="flex items-center">
                                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Running...
                                </span>
                            ) : (
                                <>
                                    <Play className="h-4 w-4 mr-2" />
                                    Run Backtest
                                </>
                            )}
                        </button>
                    </div>

                    <div className="flex gap-3">
                        <button
                            onClick={() => handleExecuteTrade('BUY')}
                            className="flex-1 inline-flex justify-center items-center px-4 py-2 rounded-lg text-sm font-medium text-white bg-emerald-600 hover:bg-emerald-700 transition-colors"
                        >
                            Buy
                        </button>
                        <button
                            onClick={() => handleExecuteTrade('SELL')}
                            className="flex-1 inline-flex justify-center items-center px-4 py-2 rounded-lg text-sm font-medium text-white bg-rose-600 hover:bg-rose-700 transition-colors"
                        >
                            Sell
                        </button>
                    </div>
                    <p className="text-xs text-gray-500 md:col-span-4">
                        Allocation override can be applied only on precomputed multi-asset strategies.
                    </p>
                </div>

                {tradeStatus && (
                    <div
                        className={`mx-6 mb-6 p-4 rounded-lg border ${
                            tradeStatus.success
                                ? 'bg-emerald-900/20 border-emerald-800 text-emerald-300'
                                : 'bg-rose-900/20 border-rose-800 text-rose-300'
                        }`}
                    >
                        <div className="flex items-center">
                            <DollarSign className="w-5 h-5 mr-2" />
                            {tradeStatus.message}
                        </div>
                    </div>
                )}
            </div>

            {backtestResults ? (
                <div className="space-y-8">
                    <MetricsTable metrics={backtestResults} />
                    <div className="card p-6">
                        <h3 className="text-lg font-medium text-gray-100 mb-2">Backtest Equity Curve (YoY CAGR)</h3>
                        <p className="text-xs text-gray-400 mb-4">
                            X axis is Year and Y axis is YoY CAGR % over the combined in-sample and out-of-sample range.
                        </p>
                        <BacktestChart
                            data={backtestResults.equity_curve}
                            timestamps={backtestResults.equity_timestamps}
                            allocations={backtestResults.equity_allocations}
                            onYearSelect={supportsAllocationOverride ? handleSelectYearPoint : undefined}
                            selectedYear={selectedYearPoint?.year}
                        />
                        {supportsAllocationOverride && selectedYearPoint && (
                            <div className="mt-5 rounded-lg border border-cyan-700/60 bg-cyan-950/20 p-4">
                                <h4 className="text-sm font-semibold text-cyan-300">Year Override: {selectedYearPoint.year}</h4>
                                <p className="mt-1 text-xs text-cyan-200/80">
                                    Use sliders to change BTC/Gold/Silver allocation for this year, then rerun backtest.
                                </p>
                                <div className="mt-4 space-y-3">
                                    <label className="block text-xs text-gray-200">
                                        <span className="flex justify-between">
                                            <span>BTC %</span>
                                            <span>{Number(selectedYearPoint.btc || 0).toFixed(1)}%</span>
                                        </span>
                                        <input
                                            type="range"
                                            min="0"
                                            max="100"
                                            step="0.1"
                                            value={selectedYearPoint.btc}
                                            onChange={(event) => updateSelectedAllocation('btc', event.target.value)}
                                            className="mt-1 w-full accent-cyan-400"
                                        />
                                    </label>
                                    <label className="block text-xs text-gray-200">
                                        <span className="flex justify-between">
                                            <span>Gold %</span>
                                            <span>{Number(selectedYearPoint.gold || 0).toFixed(1)}%</span>
                                        </span>
                                        <input
                                            type="range"
                                            min="0"
                                            max="100"
                                            step="0.1"
                                            value={selectedYearPoint.gold}
                                            onChange={(event) => updateSelectedAllocation('gold', event.target.value)}
                                            className="mt-1 w-full accent-amber-300"
                                        />
                                    </label>
                                    <label className="block text-xs text-gray-200">
                                        <span className="flex justify-between">
                                            <span>Silver %</span>
                                            <span>{Number(selectedYearPoint.silver || 0).toFixed(1)}%</span>
                                        </span>
                                        <input
                                            type="range"
                                            min="0"
                                            max="100"
                                            step="0.1"
                                            value={selectedYearPoint.silver}
                                            onChange={(event) => updateSelectedAllocation('silver', event.target.value)}
                                            className="mt-1 w-full accent-gray-300"
                                        />
                                    </label>
                                </div>
                                <p className="mt-3 text-xs text-cyan-100/80">
                                    Current total: {selectedAllocationTotal.toFixed(1)}%. Values are normalized to 100% when applied.
                                </p>
                                <div className="mt-4">
                                    <button
                                        onClick={handleApplyYearOverride}
                                        disabled={loadingBacktest}
                                        className={`btn-primary ${loadingBacktest ? 'opacity-70 cursor-wait' : ''}`}
                                    >
                                        Apply Year Allocation And Re-Backtest
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                    <div className="card p-6">
                        <h3 className="text-lg font-medium text-gray-100 mb-2">Out-of-Sample Testing Curve (Monthly Return %)</h3>
                        <p className="text-xs text-gray-400 mb-4">
                            X axis is Month and Y axis is Monthly Return % for unseen data only (includes 2026 where available). Split starts at index {backtestResults.out_of_sample_start_index ?? 0}.
                        </p>
                        {backtestResults.out_of_sample_curve?.length ? (
                            <BacktestChart
                                data={backtestResults.out_of_sample_curve}
                                timestamps={backtestResults.out_of_sample_timestamps}
                                allocations={backtestResults.out_of_sample_allocations}
                                onYearSelect={undefined}
                                selectedYear={selectedYearPoint?.year}
                                mode="monthly_return_pct"
                            />
                        ) : (
                            <p className="text-sm text-gray-500">No out-of-sample points available for this run.</p>
                        )}
                        {!!backtestResults.applied_overrides?.length && (
                            <p className="mt-3 text-xs text-emerald-300">
                                Applied override: {backtestResults.applied_overrides.map((item) => (
                                    `${item.year} (B${Number(item.btc).toFixed(0)} G${Number(item.gold).toFixed(0)} S${Number(item.silver).toFixed(0)})`
                                )).join(', ')}
                            </p>
                        )}
                    </div>
                </div>
            ) : (
                !loadingBacktest && (
                    <div className="text-center py-16 border-2 border-dashed border-gray-700 rounded-xl bg-gray-800/30">
                        <TrendingUp className="mx-auto h-14 w-14 text-gray-600 mb-4" />
                        <h3 className="text-lg font-medium text-gray-300">Run a simulation</h3>
                        <p className="mt-2 text-gray-500 max-w-md mx-auto">
                            Pick a strategy and ticker, then run the backtest to inspect performance before executing trades.
                        </p>
                    </div>
                )
            )}
        </section>
    );
};

export default TradeLabPage;
