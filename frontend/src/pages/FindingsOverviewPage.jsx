import React, { useMemo } from 'react';
import {
    Bar,
    BarChart,
    CartesianGrid,
    ComposedChart,
    Legend,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';

import { percent, round, shortStrategy } from '../utils/formatters';

const COLORS = ['#22d3ee', '#60a5fa', '#a78bfa', '#34d399', '#f97316', '#f43f5e', '#facc15', '#2dd4bf'];
const GRID_STROKE = '#374151';
const AXIS_STROKE = '#9ca3af';

const FindingsOverviewPage = ({ findings, loading, error }) => {
    const topMetrics = useMemo(() => {
        if (!findings?.ordered_metrics) return [];
        return findings.ordered_metrics.filter((item) => item.strategy !== '00_no_trade').slice(0, 12);
    }, [findings]);

    const topEquityStrategies = useMemo(() => topMetrics.slice(0, 6).map((item) => item.strategy), [topMetrics]);

    const equityData = useMemo(() => {
        if (!findings?.equity_curves?.series || topEquityStrategies.length === 0) return [];

        return findings.equity_curves.series.map((row) => {
            const point = { timestamp: row.timestamp?.slice(0, 10) ?? '' };
            topEquityStrategies.forEach((strategy) => {
                point[strategy] = row[strategy];
            });
            return point;
        });
    }, [findings, topEquityStrategies]);

    const topPerformer = topMetrics[0];

    const oosSplitLeaderboard = useMemo(() => {
        if (!findings?.validation_splits) return [];

        const grouped = new Map();
        findings.validation_splits.forEach((item) => {
            if (item.split !== 'out_of_sample' && item.split !== 'holdout') return;

            const current = grouped.get(item.strategy) || {
                strategy: item.strategy,
                oos_return_pct: 0,
                oos_sharpe: 0,
                count: 0,
            };

            current.oos_return_pct += Number(item.return_pct || 0);
            current.oos_sharpe += Number(item.sharpe || 0);
            current.count += 1;
            grouped.set(item.strategy, current);
        });

        return Array.from(grouped.values())
            .filter((item) => item.count > 0)
            .map((item) => ({
                strategy: item.strategy,
                oos_return_pct: item.oos_return_pct / item.count,
                oos_sharpe: item.oos_sharpe / item.count,
            }))
            .sort((a, b) => b.oos_return_pct - a.oos_return_pct)
            .slice(0, 12);
    }, [findings]);

    if (loading) {
        return <div className="card p-6 text-gray-300">Loading findings overview...</div>;
    }

    if (error) {
        return (
            <div className="card p-6 border border-rose-800 bg-rose-950/30">
                <h2 className="text-lg font-semibold text-rose-300">Findings unavailable</h2>
                <p className="mt-2 text-sm text-rose-200/80">{error}</p>
            </div>
        );
    }

    if (!findings) return null;

    return (
        <section className="space-y-8">
            <div className="card p-6">
                <h2 className="text-lg font-semibold text-gray-100">Findings Overview</h2>
                <p className="mt-1 text-sm text-gray-400">Performance and benchmark snapshots across all strategies.</p>

                {topPerformer && (
                    <div className="mt-5 grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="bg-gray-900/70 border border-gray-700 rounded-lg p-4">
                            <p className="text-xs uppercase tracking-wider text-gray-400">Top Return Strategy</p>
                            <p className="mt-1 text-lg text-cyan-300 font-semibold">{shortStrategy(topPerformer.strategy)}</p>
                        </div>
                        <div className="bg-gray-900/70 border border-gray-700 rounded-lg p-4">
                            <p className="text-xs uppercase tracking-wider text-gray-400">Return</p>
                            <p className="mt-1 text-lg text-emerald-300 font-semibold">{percent(topPerformer.return_pct)}</p>
                        </div>
                        <div className="bg-gray-900/70 border border-gray-700 rounded-lg p-4">
                            <p className="text-xs uppercase tracking-wider text-gray-400">Sharpe</p>
                            <p className="mt-1 text-lg text-amber-300 font-semibold">{round(topPerformer.sharpe, 3)}</p>
                        </div>
                    </div>
                )}
            </div>

            <div className="grid grid-cols-1 gap-8 xl:grid-cols-2">
                <div className="card p-5">
                    <h3 className="text-lg font-medium text-gray-100">Ordered Metrics: Return vs Sharpe</h3>
                    <p className="text-xs text-gray-400 mt-1">Source: `ordered_metrics.csv`</p>
                    <div className="h-80 mt-4">
                        <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart data={topMetrics}>
                                <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
                                <XAxis dataKey="strategy" tickFormatter={shortStrategy} stroke={AXIS_STROKE} interval={0} angle={-25} height={72} textAnchor="end" />
                                <YAxis yAxisId="left" stroke={AXIS_STROKE} tickFormatter={(value) => `${round(value)}%`} />
                                <YAxis yAxisId="right" orientation="right" stroke={AXIS_STROKE} />
                                <Tooltip formatter={(value) => round(value)} labelFormatter={(label) => shortStrategy(label)} />
                                <Legend />
                                <Bar yAxisId="left" dataKey="return_pct" fill="#22d3ee" name="Return %" />
                                <Line yAxisId="right" dataKey="sharpe" stroke="#f59e0b" strokeWidth={2} dot={false} name="Sharpe" />
                            </ComposedChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="card p-5">
                    <h3 className="text-lg font-medium text-gray-100">Benchmark Comparison</h3>
                    <p className="text-xs text-gray-400 mt-1">Source: `benchmark_comparison.csv`</p>
                    <div className="h-80 mt-4">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={findings.benchmark_comparison || []}>
                                <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
                                <XAxis dataKey="strategy" tickFormatter={shortStrategy} stroke={AXIS_STROKE} interval={0} angle={-25} height={72} textAnchor="end" />
                                <YAxis stroke={AXIS_STROKE} tickFormatter={(value) => `${round(value)}%`} />
                                <Tooltip formatter={(value) => round(value)} labelFormatter={(label) => shortStrategy(label)} />
                                <Legend />
                                <Bar dataKey="return_pct" fill="#34d399" name="Return %" />
                                <Bar dataKey="excess_return_vs_buy_hold_pct" fill="#f97316" name="Excess vs Buy/Hold %" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="card p-5 xl:col-span-2">
                    <h3 className="text-lg font-medium text-gray-100">Out-of-Sample Validation Snapshot</h3>
                    <p className="text-xs text-gray-400 mt-1">Source: `validation_splits.csv` (out_of_sample + holdout)</p>
                    <div className="h-80 mt-4">
                        <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart data={oosSplitLeaderboard}>
                                <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
                                <XAxis dataKey="strategy" tickFormatter={shortStrategy} stroke={AXIS_STROKE} interval={0} angle={-25} height={72} textAnchor="end" />
                                <YAxis yAxisId="left" stroke={AXIS_STROKE} tickFormatter={(value) => `${round(value)}%`} />
                                <YAxis yAxisId="right" orientation="right" stroke={AXIS_STROKE} />
                                <Tooltip formatter={(value) => round(value)} labelFormatter={(label) => shortStrategy(label)} />
                                <Legend />
                                <Bar yAxisId="left" dataKey="oos_return_pct" fill="#34d399" name="OOS Return %" />
                                <Line yAxisId="right" dataKey="oos_sharpe" stroke="#f97316" strokeWidth={2} dot={false} name="OOS Sharpe" />
                            </ComposedChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="card p-5 xl:col-span-2">
                    <h3 className="text-lg font-medium text-gray-100">Equity Curves (Top Strategies)</h3>
                    <p className="text-xs text-gray-400 mt-1">
                        Source: `equity_curves.csv` downsampled to {findings.equity_curves?.point_count || 0} points.
                    </p>
                    <div className="h-[26rem] mt-4">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={equityData}>
                                <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
                                <XAxis dataKey="timestamp" stroke={AXIS_STROKE} minTickGap={40} />
                                <YAxis stroke={AXIS_STROKE} />
                                <Tooltip formatter={(value) => round(value, 4)} />
                                <Legend />
                                {topEquityStrategies.map((strategy, index) => (
                                    <Line
                                        key={strategy}
                                        type="monotone"
                                        dataKey={strategy}
                                        stroke={COLORS[index % COLORS.length]}
                                        strokeWidth={2}
                                        dot={false}
                                        name={shortStrategy(strategy)}
                                    />
                                ))}
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default FindingsOverviewPage;
