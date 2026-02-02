import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const getStrategies = async () => {
    try {
        const response = await api.get('/strategies');
        return response.data;
    } catch (error) {
        console.error('Error fetching strategies:', error);
        throw error;
    }
};

export const getTickers = async () => {
    try {
        const response = await api.get('/tickers');
        return response.data;
    } catch (error) {
        console.error('Error fetching tickers:', error);
        throw error;
    }
};

export const runBacktest = async (strategyId, ticker, startDate, endDate) => {
    try {
        const response = await api.post('/backtest', {
            strategy_id: strategyId,
            ticker: ticker,
            start_date: startDate,
            end_date: endDate,
        });
        return response.data;
    } catch (error) {
        console.error('Error running backtest:', error);
        throw error;
    }
};

export const executeTrade = async (tradeParams) => {
    try {
        const response = await api.post('/trade', tradeParams);
        return response.data;
    } catch (error) {
        console.error('Error executing trade:', error);
        throw error;
    }
};

export default api;
