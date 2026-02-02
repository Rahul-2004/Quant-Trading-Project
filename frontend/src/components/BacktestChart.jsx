
import React from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
} from 'recharts';

const BacktestChart = ({ data }) => {
    if (!data || data.length === 0) return null;

    // Format data for Recharts (array of objects)
    const chartData = data.map((val, index) => ({
        time: index,
        equity: val,
    }));

    return (
        <div className="bg-white shadow rounded-lg p-6 mb-6 h-96">
            <h3 className="text-lg font-medium leading-6 text-gray-900 mb-4">Equity Curve</h3>
            <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                        data={chartData}
                        margin={{
                            top: 5,
                            right: 30,
                            left: 20,
                            bottom: 5,
                        }}
                    >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="time" />
                        <YAxis domain={['auto', 'auto']} />
                        <Tooltip />
                        <Legend />
                        <Line
                            type="monotone"
                            dataKey="equity"
                            stroke="#4F46E5"
                            activeDot={{ r: 8 }}
                            strokeWidth={2}
                            dot={false}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default BacktestChart;
