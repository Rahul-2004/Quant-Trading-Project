
import polars as pl
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from .models_shared import LinearModel, NonLinearModel
from .utils import load_ohlc_timeseries, add_lags, to_tensor, timeseries_split, batch_train_reg, set_seed
from arch import arch_model

class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def run_backtest(self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """Run backtest and return metrics + equity curve."""
        pass
        
    @abstractmethod
    def get_signal(self, ticker: str) -> int:
        """Get live signal (-1, 0, 1)."""
        pass

class LinearStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Linear Regression (AR)")
        self.model = None
        self.lags = 3
        
    def run_backtest(self, ticker: str, start_date=None, end_date=None) -> Dict[str, Any]:
        set_seed(42)
        # Load Data
        df = load_ohlc_timeseries(ticker, "1h") # Default to 1h for now
        
        # Feature Engineering (Log Returns & Lags)
        df = df.with_columns(
            (pl.col('close') / pl.col('close').shift(1)).log().alias('log_return')
        ).drop_nulls()
        
        df = add_lags(df, "log_return", self.lags)
        
        # Prepare Tensors
        features = [f"lag_{i+1}" for i in range(self.lags)]
        X = to_tensor(df.select(features))
        y = to_tensor(df.select("target"))
        
        # Split
        X_train, X_test = timeseries_split(X)
        y_train, y_test = timeseries_split(y)
        
        # Train
        self.model = LinearModel(self.lags)
        batch_train_reg(self.model, X_train, X_test, y_train, y_test, no_epochs=50)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_test).numpy().flatten()
            
        # Signal Generation (Directional)
        signals = np.sign(preds)
        actual_returns = y_test.numpy().flatten()
        
        # Calculate Performance
        trade_returns = signals * actual_returns
        equity_curve = np.cumsum(trade_returns)
        
        # Metrics
        total_return = equity_curve[-1] if len(equity_curve) > 0 else 0.0
        win_rate = np.mean(np.sign(signals) == np.sign(actual_returns))
        sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(24*365) if np.std(trade_returns) > 0 else 0
        
        return {
            "strategy": self.name,
            "total_return": float(total_return),
            "win_rate": float(win_rate),
            "sharpe": float(sharpe),
            "equity_curve": equity_curve.tolist()
        }

    def get_signal(self, ticker: str) -> int:
        # Mock live signal logic (would need latest data)
        return 1

class NonLinearStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Neural Network (Non-Linear)")
        self.model = None
        self.lags = 2
        
    def run_backtest(self, ticker: str, start_date=None, end_date=None) -> Dict[str, Any]:
        set_seed(42)
         # Load Data
        df = load_ohlc_timeseries(ticker, "1h") 
        
        # Feature Engineering
        df = df.with_columns(
            (pl.col('close') / pl.col('close').shift(1)).log().alias('log_return')
        ).drop_nulls()
        
        df = add_lags(df, "log_return", self.lags)
        
        # Prepare Tensors
        features = [f"lag_{i+1}" for i in range(self.lags)]
        X = to_tensor(df.select(features))
        y = to_tensor(df.select("target"))
        
        # Split
        X_train, X_test = timeseries_split(X)
        y_train, y_test = timeseries_split(y)
        
        # Train
        self.model = NonLinearModel(self.lags)
        batch_train_reg(self.model, X_train, X_test, y_train, y_test, no_epochs=50)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_test).numpy().flatten()
            
        # Signal Generation
        signals = np.sign(preds)
        actual_returns = y_test.numpy().flatten()
        
        # Calculate Performance
        trade_returns = signals * actual_returns
        equity_curve = np.cumsum(trade_returns)
        
        # Metrics
        total_return = equity_curve[-1] if len(equity_curve) > 0 else 0.0
        win_rate = np.mean(np.sign(signals) == np.sign(actual_returns))
        sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(24*365) if np.std(trade_returns) > 0 else 0
        
        return {
            "strategy": self.name,
            "total_return": float(total_return),
            "win_rate": float(win_rate),
            "sharpe": float(sharpe),
            "equity_curve": equity_curve.tolist()
        }
        
    def get_signal(self, ticker: str) -> int:
        return -1

class GARCHStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("GARCH Volatility Regime")
        
    def run_backtest(self, ticker: str, start_date=None, end_date=None) -> Dict[str, Any]:
        set_seed(42)
        # Load Data
        df = load_ohlc_timeseries(ticker, "1h")
        
        returns_pct = (df['close'] / df['close'].shift(1) - 1).drop_nulls().to_numpy() * 100
        log_returns = (df['close'] / df['close'].shift(1)).log().drop_nulls().to_numpy()
        
        # Determine train split
        train_size = int(len(returns_pct) * 0.75)
        train_returns = returns_pct[:train_size]
        test_returns = returns_pct[train_size:]
        test_log_returns = log_returns[train_size:]
        
        # Fit GARCH
        am = arch_model(train_returns, vol='Garch', p=1, q=1, dist='t')
        res = am.fit(disp='off')
        
        # Rolling Forecast (Simplified for performance)
        # In a real app, this would use the forecast method properly or re-fit
        # Here we mimic the notebook logic: EWMA approximation for speed in backtest
        
        n_test = len(test_returns)
        vol_forecasts = np.zeros(n_test)
        
        # Use last vol from training as start
        last_vol = np.sqrt(res.conditional_volatility[-1])
        vol_forecasts[0] = last_vol
        
        alpha = 0.06
        beta = 0.93
        
        for i in range(1, n_test):
            vol_forecasts[i] = np.sqrt(alpha * (test_returns[i-1]**2) + beta * (vol_forecasts[i-1]**2))
            
        # Generate Signals
        vol_median = np.median(vol_forecasts)
        signals = np.zeros(n_test)
        
        for i in range(1, n_test):
            if vol_forecasts[i] < vol_median:
                # Low vol: trend following
                signals[i] = np.sign(test_returns[i-1])
            else:
                # High vol: mean reversion
                signals[i] = -np.sign(test_returns[i-1])
                
        # Calculate Performance
        trade_returns = signals * test_log_returns
        equity_curve = np.cumsum(trade_returns)
        
        total_return = equity_curve[-1] if len(equity_curve) > 0 else 0.0
        # Fix division by zero
        std_ret = np.std(trade_returns)
        sharpe = np.mean(trade_returns) / std_ret * np.sqrt(24*365) if std_ret > 1e-6 else 0.0
        win_rate = np.mean(np.sign(signals[signals != 0]) == np.sign(test_log_returns[signals != 0]))
        
        return {
            "strategy": self.name,
            "total_return": float(total_return),
            "win_rate": float(win_rate),
            "sharpe": float(sharpe),
            "equity_curve": equity_curve.tolist()
        }
        
    def get_signal(self, ticker: str) -> int:
        return 0

# Strategy Registry
STRATEGIES = {
    "linear": LinearStrategy(),
    "nonlinear": NonLinearStrategy(),
    "garch": GARCHStrategy()
}
