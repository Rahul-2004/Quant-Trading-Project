import React from 'react';

const MetricsTable = ({ metrics }) => {
    if (!metrics) return null;

    const totalReturn = Number(metrics.total_return ?? 0);
    const winRate = Number(metrics.win_rate ?? 0);
    const sharpe = Number(metrics.sharpe ?? 0);
    const cagr = Number(metrics.cagr ?? 0);
    const outOfSampleReturn = Number(metrics.out_of_sample_total_return ?? totalReturn);

    const data = [
        { label: 'Strategy', value: metrics.strategy },
        { label: 'Total Return', value: `${(totalReturn * 100).toFixed(2)}%` },
        { label: 'Win Rate', value: `${(winRate * 100).toFixed(2)}%` },
        { label: 'Sharpe Ratio', value: sharpe.toFixed(3) },
        { label: 'CAGR', value: `${(cagr * 100).toFixed(2)}%` },
        { label: 'OOS Return', value: `${(outOfSampleReturn * 100).toFixed(2)}%` },
    ];

    return (
        <div className="card p-6">
            <h3 className="text-lg font-medium text-gray-100 mb-4">Backtest Metrics</h3>
            <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-6">
                {data.map((item) => (
                    <div key={item.label} className="px-4 py-4 bg-gray-900/70 border border-gray-700 rounded-lg overflow-hidden">
                        <dt className="text-xs font-medium text-gray-400 uppercase tracking-wider">{item.label}</dt>
                        <dd className="mt-1 text-2xl font-semibold text-gray-100 break-words">{item.value}</dd>
                    </div>
                ))}
            </dl>
        </div>
    );
};

export default MetricsTable;
