
import React, { useState, useEffect } from 'react';
import { getStrategies, getTickers, runBacktest, executeTrade } from '../services/api';
import MetricsTable from './MetricsTable';
import BacktestChart from './BacktestChart';
import { Play, TrendingUp, Activity, DollarSign, BarChart2 } from 'lucide-react';

const Dashboard = () => {
    const [strategies, setStrategies] = useState([]);
    const [tickers, setTickers] = useState([]);
    const [selectedStrategy, setSelectedStrategy] = useState('');
    const [selectedTicker, setSelectedTicker] = useState('BTCUSDT');
    const [backtestResults, setBacktestResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [tradeStatus, setTradeStatus] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const strats = await getStrategies();
                const tcks = await getTickers();
                setStrategies(strats);
                setTickers(tcks);
                if (strats.length > 0) setSelectedStrategy(strats[0].id);
            } catch (err) {
                console.error("Failed to load initial data", err);
            }
        };
        fetchData();
    }, []);

    const handleRunBacktest = async () => {
        if (!selectedStrategy || !selectedTicker) return;
        setLoading(true);
        setBacktestResults(null);
        try {
            const results = await runBacktest(selectedStrategy, selectedTicker);
            setBacktestResults(results);
        } catch (err) {
            alert("Backtest failed: " + err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleExecuteTrade = async (side) => {
        if (!selectedStrategy || !selectedTicker) return;
        try {
            const result = await executeTrade({
                ticker: selectedTicker,
                side: side,
                quantity: 0.01,
                strategy_id: selectedStrategy
            });
            setTradeStatus({ success: true, message: `Trade Executed: ${result.message} (ID: ${result.order_id})` });
            setTimeout(() => setTradeStatus(null), 5000);
        } catch (err) {
            setTradeStatus({ success: false, message: "Trade failed: " + err.message });
            setTimeout(() => setTradeStatus(null), 5000);
        }
    };

    return (
        <div className="min-h-screen bg-gray-900 text-gray-100">
            <nav className="bg-gray-800 border-b border-gray-700 sticky top-0 z-50">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between h-16">
                        <div className="flex items-center">
                            <Activity className="h-8 w-8 text-indigo-500" />
                            <span className="ml-3 text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-cyan-400">
                                QuantDash
                            </span>
                        </div>
                    </div>
                </div>
            </nav>

            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                {/* Control Panel */}
                <div className="card mb-8">
                    <div className="p-6 border-b border-gray-700 bg-gray-800/50">
                        <h2 className="text-lg font-semibold text-gray-100 flex items-center">
                            <BarChart2 className="w-5 h-5 mr-2 text-indigo-400" />
                            Simulation & Trading
                        </h2>
                    </div>
                    <div className="p-6 grid grid-cols-1 md:grid-cols-4 gap-6 items-end">
                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-1">Strategy</label>
                            <select
                                value={selectedStrategy}
                                onChange={(e) => setSelectedStrategy(e.target.value)}
                                className="input-field"
                            >
                                {strategies.map(s => (
                                    <option key={s.id} value={s.id}>{s.name}</option>
                                ))}
                            </select>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-1">Ticker</label>
                            <select
                                value={selectedTicker}
                                onChange={(e) => setSelectedTicker(e.target.value)}
                                className="input-field"
                            >
                                {tickers.map(t => (
                                    <option key={t} value={t}>{t}</option>
                                ))}
                            </select>
                        </div>

                        <div>
                            <button
                                onClick={handleRunBacktest}
                                disabled={loading}
                                className={`w-full btn-primary ${loading ? 'opacity-75 cursor-wait' : ''}`}
                            >
                                {loading ? (
                                    <span className="flex items-center">
                                        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                        Processing...
                                    </span>
                                ) : (
                                    <>
                                        <Play className="h-4 w-4 mr-2" />
                                        Run Backtest
                                    </>
                                )}
                            </button>
                        </div>

                        <div className="flex space-x-3">
                            <button
                                onClick={() => handleExecuteTrade('BUY')}
                                className="flex-1 inline-flex justify-center items-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg text-white bg-emerald-600 hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-emerald-500 transition-colors"
                            >
                                Buy
                            </button>
                            <button
                                onClick={() => handleExecuteTrade('SELL')}
                                className="flex-1 inline-flex justify-center items-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg text-white bg-rose-600 hover:bg-rose-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-rose-500 transition-colors"
                            >
                                Sell
                            </button>
                        </div>
                    </div>

                    {tradeStatus && (
                        <div className={`mx-6 mb-6 p-4 rounded-lg border ${tradeStatus.success ? 'bg-emerald-900/20 border-emerald-800 text-emerald-400' : 'bg-rose-900/20 border-rose-800 text-rose-400'}`}>
                            <div className="flex items-center">
                                <DollarSign className="w-5 h-5 mr-2" />
                                {tradeStatus.message}
                            </div>
                        </div>
                    )}
                </div>

                {/* Results Section */}
                {backtestResults ? (
                    <div className="space-y-8 animate-fade-in">
                        <MetricsTable metrics={backtestResults} />
                        <div className="card p-6">
                            <h3 className="text-lg font-medium text-gray-100 mb-6">Equity Curve</h3>
                            <BacktestChart data={backtestResults.equity_curve} />
                        </div>
                    </div>
                ) : (
                    !loading && (
                        <div className="text-center py-20 border-2 border-dashed border-gray-700 rounded-xl bg-gray-800/30">
                            <TrendingUp className="mx-auto h-16 w-16 text-gray-600 mb-4" />
                            <h3 className="text-lg font-medium text-gray-300">Ready to Analyze</h3>
                            <p className="mt-2 text-gray-500 max-w-sm mx-auto">
                                Select a trading strategy and a ticker above, then click "Run Backtest" to see historical performance.
                            </p>
                        </div>
                    )
                )}
            </main>
        </div>
    );
};

export default Dashboard;
