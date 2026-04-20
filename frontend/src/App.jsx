import React, { useEffect, useState } from 'react';
import { Activity, CandlestickChart } from 'lucide-react';

import TradeLabPage from './pages/TradeLabPage';
import { getStrategies, getTickers } from './services/api';

const App = () => {
    const [strategies, setStrategies] = useState([]);
    const [tickers, setTickers] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            try {
                const [strategyResult, tickerResult] = await Promise.allSettled([
                    getStrategies(),
                    getTickers(),
                ]);

                const errors = [];

                if (strategyResult.status === 'fulfilled') {
                    setStrategies(strategyResult.value);
                } else {
                    console.error('Failed to load strategies', strategyResult.reason);
                    errors.push('strategies');
                }

                if (tickerResult.status === 'fulfilled') {
                    setTickers(tickerResult.value);
                } else {
                    console.error('Failed to load tickers', tickerResult.reason);
                    errors.push('tickers');
                }

                if (errors.length > 0) {
                    setError(`Some datasets failed to load: ${errors.join(', ')}.`);
                } else {
                    setError('');
                }
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    return (
        <div className="min-h-screen bg-gray-900 text-gray-100">
            <header className="bg-gray-800/95 border-b border-gray-700 sticky top-0 z-50 backdrop-blur">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex flex-col gap-4 py-4 md:flex-row md:items-center md:justify-between">
                        <div className="flex items-center">
                            <Activity className="h-7 w-7 text-cyan-400" />
                            <div className="ml-3">
                                <h1 className="text-xl font-bold text-gray-100">QuantDash Research Suite</h1>
                                <p className="text-xs text-gray-400">Single workspace focused on Trade Lab backtesting and execution</p>
                            </div>
                        </div>

                        <div className="inline-flex items-center px-3 py-2 rounded-lg text-sm border bg-cyan-500/20 text-cyan-300 border-cyan-500/50">
                            <CandlestickChart className="w-4 h-4 mr-2" />
                            Trade Lab
                        </div>
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <TradeLabPage strategies={strategies} tickers={tickers} loading={loading} error={error} />
            </main>
        </div>
    );
};

export default App;
