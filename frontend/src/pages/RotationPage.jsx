import React, { useMemo, useState } from 'react';
import {
    Area,
    AreaChart,
    Bar,
    BarChart,
    CartesianGrid,
    Legend,
    ResponsiveContainer,
    Scatter,
    ScatterChart,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';

import { round, shortStrategy } from '../utils/formatters';

const GRID_STROKE = '#374151';
const AXIS_STROKE = '#9ca3af';

const RotationPage = ({ findings, loading, error }) => {
    const [selectedRotation, setSelectedRotation] = useState('');
    const activeRotation = selectedRotation || findings?.breakout_rotation_weights?.strategies?.[0] || '';

    const selectedRotationStats = useMemo(() => {
        if (!findings?.breakout_rotation_ranked || !activeRotation) return null;
        return findings.breakout_rotation_ranked.find((item) => item.strategy === activeRotation) || null;
    }, [findings, activeRotation]);

    const breakoutWeightsData = useMemo(() => {
        if (!findings?.breakout_rotation_weights?.series || !activeRotation) return [];

        const btcKey = `${activeRotation}_btc`;
        const goldKey = `${activeRotation}_gold`;
        const silverKey = `${activeRotation}_silver`;
        const cashKey = `${activeRotation}_cash`;

        return findings.breakout_rotation_weights.series.map((row) => ({
            timestamp: row.timestamp?.slice(0, 10) ?? '',
            btc: row[btcKey] ?? 0,
            gold: row[goldKey] ?? 0,
            silver: row[silverKey] ?? 0,
            cash: row[cashKey] ?? 0,
        }));
    }, [findings, activeRotation]);

    if (loading) {
        return <div className="card p-6 text-gray-300">Loading rotation research...</div>;
    }

    if (error) {
        return (
            <div className="card p-6 border border-rose-800 bg-rose-950/30">
                <h2 className="text-lg font-semibold text-rose-300">Rotation view unavailable</h2>
                <p className="mt-2 text-sm text-rose-200/80">{error}</p>
            </div>
        );
    }

    if (!findings) return null;

    return (
        <section className="space-y-8">
            <div className="card p-6">
                <h2 className="text-lg font-semibold text-gray-100">Breakout Rotation</h2>
                <p className="mt-1 text-sm text-gray-400">Ranking, risk profile, and dynamic BTC/Gold/Silver/Cash allocations.</p>
                <div className="mt-4 max-w-sm">
                    <label className="text-sm text-gray-400">
                        Rotation strategy
                        <select value={activeRotation} onChange={(event) => setSelectedRotation(event.target.value)} className="input-field">
                            {(findings.breakout_rotation_weights?.strategies || []).map((strategy) => (
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
                        <p className="mt-1 text-lg text-emerald-300 font-semibold">{round(selectedRotationStats?.return_pct, 2)}%</p>
                    </div>
                    <div className="bg-gray-900/70 border border-gray-700 rounded-lg p-4">
                        <p className="text-xs uppercase tracking-wider text-gray-400">Sharpe</p>
                        <p className="mt-1 text-lg text-amber-300 font-semibold">{round(selectedRotationStats?.sharpe, 3)}</p>
                    </div>
                    <div className="bg-gray-900/70 border border-gray-700 rounded-lg p-4">
                        <p className="text-xs uppercase tracking-wider text-gray-400">OOS Return</p>
                        <p className="mt-1 text-lg text-cyan-300 font-semibold">{round(selectedRotationStats?.oos_return_pct, 2)}%</p>
                    </div>
                    <div className="bg-gray-900/70 border border-gray-700 rounded-lg p-4">
                        <p className="text-xs uppercase tracking-wider text-gray-400">OOS Sharpe</p>
                        <p className="mt-1 text-lg text-indigo-300 font-semibold">{round(selectedRotationStats?.oos_sharpe, 3)}</p>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 gap-8 xl:grid-cols-2">
                <div className="card p-5">
                    <h3 className="text-lg font-medium text-gray-100">Rotation Ranking</h3>
                    <p className="text-xs text-gray-400 mt-1">Source: `breakout_rotation_ranked.csv`</p>
                    <div className="h-80 mt-4">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={findings.breakout_rotation_ranked || []}>
                                <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
                                <XAxis dataKey="strategy" tickFormatter={shortStrategy} stroke={AXIS_STROKE} interval={0} angle={-25} height={72} textAnchor="end" />
                                <YAxis stroke={AXIS_STROKE} />
                                <Tooltip formatter={(value) => round(value)} labelFormatter={(label) => shortStrategy(label)} />
                                <Legend />
                                <Bar dataKey="return_pct" fill="#22d3ee" name="Return %" />
                                <Bar dataKey="oos_return_pct" fill="#f59e0b" name="OOS Return %" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="card p-5">
                    <h3 className="text-lg font-medium text-gray-100">Risk vs CAGR Map</h3>
                    <p className="text-xs text-gray-400 mt-1">Source: `breakout_rotation_results.csv`</p>
                    <div className="h-80 mt-4">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart>
                                <CartesianGrid stroke={GRID_STROKE} />
                                <XAxis type="number" dataKey="max_drawdown" name="Max Drawdown" stroke={AXIS_STROKE} />
                                <YAxis type="number" dataKey="cagr" name="CAGR" stroke={AXIS_STROKE} />
                                <Tooltip formatter={(value) => round(value, 4)} />
                                <Scatter data={findings.breakout_rotation_results || []} fill="#f97316" />
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="card p-5 xl:col-span-2">
                    <h3 className="text-lg font-medium text-gray-100">Out-of-Sample Comparison</h3>
                    <p className="text-xs text-gray-400 mt-1">Source: `breakout_rotation_ranked.csv`</p>
                    <div className="h-80 mt-4">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={(findings.breakout_rotation_ranked || []).slice(0, 12)}>
                                <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
                                <XAxis dataKey="strategy" tickFormatter={shortStrategy} stroke={AXIS_STROKE} interval={0} angle={-25} height={72} textAnchor="end" />
                                <YAxis stroke={AXIS_STROKE} />
                                <Tooltip formatter={(value) => round(value)} labelFormatter={(label) => shortStrategy(label)} />
                                <Legend />
                                <Bar dataKey="oos_return_pct" fill="#22d3ee" name="OOS Return %" />
                                <Bar dataKey="oos_sharpe" fill="#f97316" name="OOS Sharpe" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="card p-5 xl:col-span-2">
                    <h3 className="text-lg font-medium text-gray-100">Allocation Over Time</h3>
                    <p className="text-xs text-gray-400 mt-1">
                        Source: `breakout_rotation_weights.csv` downsampled to {findings.breakout_rotation_weights?.point_count || 0} points.
                    </p>
                    <div className="h-[26rem] mt-4">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={breakoutWeightsData}>
                                <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
                                <XAxis dataKey="timestamp" stroke={AXIS_STROKE} minTickGap={40} />
                                <YAxis stroke={AXIS_STROKE} domain={[0, 1]} />
                                <Tooltip formatter={(value) => round(value, 3)} />
                                <Legend />
                                <Area type="monotone" dataKey="btc" stackId="alloc" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.5} name="BTC" />
                                <Area type="monotone" dataKey="gold" stackId="alloc" stroke="#facc15" fill="#facc15" fillOpacity={0.4} name="Gold" />
                                <Area type="monotone" dataKey="silver" stackId="alloc" stroke="#cbd5e1" fill="#cbd5e1" fillOpacity={0.4} name="Silver" />
                                <Area type="monotone" dataKey="cash" stackId="alloc" stroke="#34d399" fill="#34d399" fillOpacity={0.4} name="Cash" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default RotationPage;
