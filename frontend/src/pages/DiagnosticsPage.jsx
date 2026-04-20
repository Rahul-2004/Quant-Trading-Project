import React, { useMemo, useState } from 'react';
import {
    Bar,
    BarChart,
    CartesianGrid,
    Legend,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';

import { round, shortStrategy } from '../utils/formatters';

const GRID_STROKE = '#374151';
const AXIS_STROKE = '#9ca3af';

const SPLIT_ORDER = {
    in_sample: 1,
    out_of_sample: 2,
    holdout: 3,
};

const DiagnosticsPage = ({ findings, loading, error }) => {
    const [selectedStrategy, setSelectedStrategy] = useState('');
    const activeStrategy = selectedStrategy || findings?.strategy_options?.[0] || '';

    const strategyHeadlineMetrics = useMemo(() => {
        if (!findings?.ordered_metrics || !activeStrategy) return null;
        return findings.ordered_metrics.find((item) => item.strategy === activeStrategy) || null;
    }, [findings, activeStrategy]);

    const robustnessData = useMemo(() => {
        if (!findings?.robustness_checks || !activeStrategy) return [];
        return findings.robustness_checks
            .filter((item) => item.strategy === activeStrategy)
            .sort((a, b) => a.perturb_factor - b.perturb_factor);
    }, [findings, activeStrategy]);

    const validationData = useMemo(() => {
        if (!findings?.validation_splits || !activeStrategy) return [];
        return findings.validation_splits
            .filter((item) => item.strategy === activeStrategy)
            .sort((a, b) => (SPLIT_ORDER[a.split] || 99) - (SPLIT_ORDER[b.split] || 99));
    }, [findings, activeStrategy]);

    const selectedOosSplit = useMemo(() => {
        if (!validationData.length) return null;
        return validationData.find((item) => item.split === 'out_of_sample') || validationData.find((item) => item.split === 'holdout') || null;
    }, [validationData]);

    const walkForwardData = useMemo(() => {
        if (!findings?.walk_forward_windows || !activeStrategy) return [];
        return findings.walk_forward_windows.filter((item) => item.strategy === activeStrategy);
    }, [findings, activeStrategy]);

    const tradePnlStats = useMemo(() => {
        if (!findings?.trade_logs_summary?.strategy_stats) return [];
        return findings.trade_logs_summary.strategy_stats.slice(0, 12);
    }, [findings]);

    const regimeData = useMemo(() => {
        if (!findings?.context_summary?.strategy_regime_distribution || !activeStrategy) return [];

        const found = findings.context_summary.strategy_regime_distribution.find((item) => item.strategy === activeStrategy);
        if (!found) return [];

        return Object.entries(found)
            .filter(([key]) => key !== 'strategy')
            .map(([regime, count]) => ({ regime, count }))
            .sort((a, b) => b.count - a.count);
    }, [findings, activeStrategy]);

    if (loading) {
        return <div className="card p-6 text-gray-300">Loading diagnostics...</div>;
    }

    if (error) {
        return (
            <div className="card p-6 border border-rose-800 bg-rose-950/30">
                <h2 className="text-lg font-semibold text-rose-300">Diagnostics unavailable</h2>
                <p className="mt-2 text-sm text-rose-200/80">{error}</p>
            </div>
        );
    }

    if (!findings) return null;

    return (
        <section className="space-y-8">
            <div className="card p-6">
                <h2 className="text-lg font-semibold text-gray-100">Diagnostics</h2>
                <p className="mt-1 text-sm text-gray-400">Robustness, split behavior, trade outcomes and regime exposure.</p>
                <div className="mt-4 max-w-sm">
                    <label className="text-sm text-gray-400">
                        Strategy focus
                        <select value={activeStrategy} onChange={(event) => setSelectedStrategy(event.target.value)} className="input-field">
                            {(findings.strategy_options || []).map((strategy) => (
                                <option key={strategy} value={strategy}>
                                    {shortStrategy(strategy)}
                                </option>
                            ))}
                        </select>
                    </label>
                </div>

                <div className="mt-5 grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="bg-gray-900/70 border border-gray-700 rounded-lg p-4">
                        <p className="text-xs uppercase tracking-wider text-gray-400">Return</p>
                        <p className="mt-1 text-lg text-emerald-300 font-semibold">{round(strategyHeadlineMetrics?.return_pct, 2)}%</p>
                    </div>
                    <div className="bg-gray-900/70 border border-gray-700 rounded-lg p-4">
                        <p className="text-xs uppercase tracking-wider text-gray-400">Win Rate</p>
                        <p className="mt-1 text-lg text-cyan-300 font-semibold">{round((strategyHeadlineMetrics?.win_rate || 0) * 100, 2)}%</p>
                    </div>
                    <div className="bg-gray-900/70 border border-gray-700 rounded-lg p-4">
                        <p className="text-xs uppercase tracking-wider text-gray-400">Sharpe</p>
                        <p className="mt-1 text-lg text-amber-300 font-semibold">{round(strategyHeadlineMetrics?.sharpe, 3)}</p>
                    </div>
                    <div className="bg-gray-900/70 border border-gray-700 rounded-lg p-4">
                        <p className="text-xs uppercase tracking-wider text-gray-400">OOS Return</p>
                        <p className="mt-1 text-lg text-indigo-300 font-semibold">{round(selectedOosSplit?.return_pct, 2)}%</p>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 gap-8 xl:grid-cols-2">
                <div className="card p-5">
                    <h3 className="text-lg font-medium text-gray-100">Robustness Sensitivity</h3>
                    <p className="text-xs text-gray-400 mt-1">Source: `robustness_checks.csv` ({shortStrategy(activeStrategy)})</p>
                    <div className="h-80 mt-4">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={robustnessData}>
                                <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
                                <XAxis dataKey="perturb_factor" stroke={AXIS_STROKE} />
                                <YAxis stroke={AXIS_STROKE} tickFormatter={(value) => `${round(value)}%`} />
                                <Tooltip formatter={(value) => round(value)} />
                                <Legend />
                                <Line type="monotone" dataKey="return_pct" stroke="#a78bfa" strokeWidth={2} name="Return %" />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="card p-5">
                    <h3 className="text-lg font-medium text-gray-100">Validation Splits</h3>
                    <p className="text-xs text-gray-400 mt-1">Source: `validation_splits.csv` ({shortStrategy(activeStrategy)})</p>
                    <div className="h-80 mt-4">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={validationData}>
                                <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
                                <XAxis dataKey="split" stroke={AXIS_STROKE} />
                                <YAxis yAxisId="left" stroke={AXIS_STROKE} />
                                <YAxis yAxisId="right" orientation="right" stroke={AXIS_STROKE} />
                                <Tooltip formatter={(value) => round(value)} />
                                <Legend />
                                <Bar yAxisId="left" dataKey="return_pct" fill="#22d3ee" name="Return %" />
                                <Bar yAxisId="right" dataKey="sharpe" fill="#f59e0b" name="Sharpe" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="card p-5">
                    <h3 className="text-lg font-medium text-gray-100">Walk-Forward OOS Stability</h3>
                    <p className="text-xs text-gray-400 mt-1">Source: `walk_forward_windows.csv` ({shortStrategy(activeStrategy)})</p>
                    <div className="h-80 mt-4">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={walkForwardData}>
                                <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
                                <XAxis dataKey="window_id" stroke={AXIS_STROKE} />
                                <YAxis yAxisId="left" stroke={AXIS_STROKE} tickFormatter={(value) => `${round(value)}%`} />
                                <YAxis yAxisId="right" orientation="right" stroke={AXIS_STROKE} />
                                <Tooltip formatter={(value) => round(value)} />
                                <Legend />
                                <Line yAxisId="left" type="monotone" dataKey="oos_return_pct" stroke="#34d399" strokeWidth={2} dot={false} name="OOS Return %" />
                                <Line yAxisId="right" type="monotone" dataKey="oos_sharpe" stroke="#f97316" strokeWidth={2} dot={false} name="OOS Sharpe" />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="card p-5">
                    <h3 className="text-lg font-medium text-gray-100">Trade PnL Distribution</h3>
                    <p className="text-xs text-gray-400 mt-1">Source: `trade_logs.csv` histogram</p>
                    <div className="h-80 mt-4">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={findings.trade_logs_summary?.pnl_histogram || []}>
                                <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
                                <XAxis dataKey="bin_label" stroke={AXIS_STROKE} hide />
                                <YAxis stroke={AXIS_STROKE} />
                                <Tooltip />
                                <Bar dataKey="count" fill="#60a5fa" name="Trades" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="card p-5">
                    <h3 className="text-lg font-medium text-gray-100">Trade Outcomes by Strategy</h3>
                    <p className="text-xs text-gray-400 mt-1">Source: `trade_logs.csv` aggregated</p>
                    <div className="h-80 mt-4">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={tradePnlStats}>
                                <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
                                <XAxis dataKey="strategy" tickFormatter={shortStrategy} stroke={AXIS_STROKE} interval={0} angle={-25} height={72} textAnchor="end" />
                                <YAxis yAxisId="left" stroke={AXIS_STROKE} />
                                <YAxis yAxisId="right" orientation="right" stroke={AXIS_STROKE} />
                                <Tooltip formatter={(value) => round(value, 4)} labelFormatter={(label) => shortStrategy(label)} />
                                <Legend />
                                <Bar yAxisId="left" dataKey="total_net_pnl" fill="#34d399" name="Total Net PnL" />
                                <Line yAxisId="right" type="monotone" dataKey="win_rate" stroke="#f43f5e" strokeWidth={2} dot={false} name="Win Rate" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="card p-5">
                    <h3 className="text-lg font-medium text-gray-100">Regime Mix at Entry</h3>
                    <p className="text-xs text-gray-400 mt-1">Source: `context_logs.csv` ({shortStrategy(activeStrategy)})</p>
                    <div className="h-80 mt-4">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={regimeData}>
                                <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
                                <XAxis dataKey="regime" stroke={AXIS_STROKE} interval={0} angle={-25} height={72} textAnchor="end" />
                                <YAxis stroke={AXIS_STROKE} />
                                <Tooltip />
                                <Bar dataKey="count" fill="#a78bfa" name="Trades" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default DiagnosticsPage;
