import React, { useMemo, useState } from 'react';
import {
    Bar,
    BarChart,
    CartesianGrid,
    Legend,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';

import { currency, percent, shortStrategy } from '../utils/formatters';

const GRID_STROKE = '#374151';
const AXIS_STROKE = '#9ca3af';

const InvestmentProjectionPage = ({ findings, loading, error }) => {
    const [initialAmount, setInitialAmount] = useState(10000);
    const [selectedYear, setSelectedYear] = useState('');

    const investmentReference = findings?.investment_reference;
    const years = investmentReference?.years || [];
    const activeYear = selectedYear || years[0] || '';

    const projectionRows = useMemo(() => {
        if (!investmentReference || !activeYear) return [];

        const startMap = investmentReference.start_equity_by_year?.[activeYear] || {};
        const latestMap = investmentReference.latest_equity_by_strategy || {};

        return Object.keys(latestMap)
            .map((strategy) => {
                const startEquity = Number(startMap[strategy] ?? 0);
                const latestEquity = Number(latestMap[strategy] ?? 0);

                if (startEquity <= 0) {
                    return {
                        strategy,
                        multiple: 0,
                        currentValue: 0,
                        gain: -Number(initialAmount || 0),
                        returnPct: -100,
                    };
                }

                const multiple = latestEquity / startEquity;
                const currentValue = Number(initialAmount || 0) * multiple;

                return {
                    strategy,
                    multiple,
                    currentValue,
                    gain: currentValue - Number(initialAmount || 0),
                    returnPct: (multiple - 1) * 100,
                };
            })
            .sort((a, b) => b.currentValue - a.currentValue);
    }, [investmentReference, activeYear, initialAmount]);

    const best = projectionRows[0];
    const median = projectionRows.length ? projectionRows[Math.floor(projectionRows.length / 2)] : null;
    const worst = projectionRows.length ? projectionRows[projectionRows.length - 1] : null;
    const topBars = projectionRows.slice(0, 12);

    if (loading) {
        return <div className="card p-6 text-gray-300">Loading investment projections...</div>;
    }

    if (error) {
        return (
            <div className="card p-6 border border-rose-800 bg-rose-950/30">
                <h2 className="text-lg font-semibold text-rose-300">Investment projections unavailable</h2>
                <p className="mt-2 text-sm text-rose-200/80">{error}</p>
            </div>
        );
    }

    if (!findings) return null;

    return (
        <section className="space-y-8">
            <div className="card p-6">
                <h2 className="text-lg font-semibold text-gray-100">Investment Growth Simulator</h2>
                <p className="mt-1 text-sm text-gray-400">Pick a year and amount to estimate value now for each strategy.</p>
                <p className="mt-1 text-xs text-gray-500">Reference endpoint: `investment_reference` built from `equity_curves.csv`.</p>

                <div className="mt-5 grid grid-cols-1 md:grid-cols-2 gap-4">
                    <label className="text-sm text-gray-400">
                        Initial amount (USD)
                        <input
                            type="number"
                            min="0"
                            value={initialAmount}
                            onChange={(event) => setInitialAmount(Number(event.target.value || 0))}
                            className="input-field"
                        />
                    </label>

                    <label className="text-sm text-gray-400">
                        Start year
                        <select value={activeYear} onChange={(event) => setSelectedYear(event.target.value)} className="input-field">
                            {years.map((year) => (
                                <option key={year} value={year}>
                                    {year}
                                </option>
                            ))}
                        </select>
                    </label>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="card p-4">
                    <p className="text-xs uppercase tracking-wider text-gray-400">Best Outcome</p>
                    <p className="mt-1 text-lg text-emerald-300 font-semibold">{best ? shortStrategy(best.strategy) : '-'}</p>
                    <p className="text-sm text-gray-300">{best ? currency(best.currentValue) : '-'}</p>
                </div>
                <div className="card p-4">
                    <p className="text-xs uppercase tracking-wider text-gray-400">Median Outcome</p>
                    <p className="mt-1 text-lg text-cyan-300 font-semibold">{median ? shortStrategy(median.strategy) : '-'}</p>
                    <p className="text-sm text-gray-300">{median ? currency(median.currentValue) : '-'}</p>
                </div>
                <div className="card p-4">
                    <p className="text-xs uppercase tracking-wider text-gray-400">Worst Outcome</p>
                    <p className="mt-1 text-lg text-rose-300 font-semibold">{worst ? shortStrategy(worst.strategy) : '-'}</p>
                    <p className="text-sm text-gray-300">{worst ? currency(worst.currentValue) : '-'}</p>
                </div>
            </div>

            <div className="card p-5">
                <h3 className="text-lg font-medium text-gray-100">Projected Value (Top 12)</h3>
                <p className="text-xs text-gray-400 mt-1">Investing {currency(initialAmount)} in {activeYear}, valued at latest timestamp {investmentReference?.latest_timestamp?.slice(0, 10) || '-'}</p>
                <div className="h-96 mt-4">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={topBars}>
                            <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
                            <XAxis dataKey="strategy" tickFormatter={shortStrategy} stroke={AXIS_STROKE} interval={0} angle={-25} height={72} textAnchor="end" />
                            <YAxis stroke={AXIS_STROKE} />
                            <Tooltip
                                formatter={(value, name) => {
                                    if (name === 'currentValue') return currency(value);
                                    if (name === 'returnPct') return percent(value);
                                    return value;
                                }}
                                labelFormatter={(label) => shortStrategy(label)}
                            />
                            <Legend />
                            <Bar dataKey="currentValue" fill="#22d3ee" name="Projected Value" />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="card p-5 overflow-x-auto">
                <h3 className="text-lg font-medium text-gray-100">All Strategy Projections</h3>
                <table className="mt-4 w-full text-sm">
                    <thead>
                        <tr className="text-left text-gray-400 border-b border-gray-700">
                            <th className="py-2 pr-3">Strategy</th>
                            <th className="py-2 pr-3">Multiple</th>
                            <th className="py-2 pr-3">Return</th>
                            <th className="py-2 pr-3">Projected Value</th>
                            <th className="py-2 pr-3">Gain / Loss</th>
                        </tr>
                    </thead>
                    <tbody>
                        {projectionRows.map((row) => (
                            <tr key={row.strategy} className="border-b border-gray-800/80 text-gray-200">
                                <td className="py-2 pr-3">{shortStrategy(row.strategy)}</td>
                                <td className="py-2 pr-3">{row.multiple.toFixed(2)}x</td>
                                <td className={`py-2 pr-3 ${row.returnPct >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>{percent(row.returnPct)}</td>
                                <td className="py-2 pr-3">{currency(row.currentValue)}</td>
                                <td className={`py-2 pr-3 ${row.gain >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>{currency(row.gain)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </section>
    );
};

export default InvestmentProjectionPage;
