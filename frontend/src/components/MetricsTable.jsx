
import React from 'react';

const MetricsTable = ({ metrics }) => {
    if (!metrics) return null;

    const data = [
        { label: 'Strategy', value: metrics.strategy },
        { label: 'Total Return', value: `${(metrics.total_return * 100).toFixed(2)}%` },
        { label: 'Win Rate', value: `${(metrics.win_rate * 100).toFixed(2)}%` },
        { label: 'Sharpe Ratio', value: metrics.sharpe.toFixed(3) },
    ];

    return (
        <div className="bg-white shadow rounded-lg p-6 mb-6">
            <h3 className="text-lg font-medium leading-6 text-gray-900 mb-4">Performance Metrics</h3>
            <dl className="grid grid-cols-1 gap-5 sm:grid-cols-4">
                {data.map((item) => (
                    <div key={item.label} className="px-4 py-5 bg-gray-50 shadow rounded-lg overflow-hidden sm:p-6">
                        <dt className="text-sm font-medium text-gray-500 truncate">{item.label}</dt>
                        <dd className="mt-1 text-3xl font-semibold text-gray-900">{item.value}</dd>
                    </div>
                ))}
            </dl>
        </div>
    );
};

export default MetricsTable;
