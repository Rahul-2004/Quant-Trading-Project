
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Ensure determinstic behavior in torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define standard OHLC aggregations
OHLC_AGGS = [
    pl.col("price").first().alias("open"),
    pl.col("price").max().alias("high"),
    pl.col("price").min().alias("low"),
    pl.col("price").last().alias("close"),
]

def get_trade_files(directory: str, sym: str) -> List[Path]:
    """Get all files in directory that start with '{sym}-trades'."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    pattern = f"{sym}-trades*"
    return sorted(dir_path.glob(pattern))

def load_timeseries(
    sym: str, 
    time_interval: str,
    aggs: List[pl.Expr],
    data_path: str = './data'
) -> pl.DataFrame:
    """Load and aggregate trade data into timeseries."""
    files = get_trade_files(data_path, sym)
    
    if not files:
        # Try cache if data not found, or raise
        cache_path = Path('./cache')
        files = get_trade_files(str(cache_path), sym)
        if not files:
             raise FileNotFoundError(f"No files found for {sym} in {data_path} or cache")
    
    ts_list = []
    
    for file in files: # Removing tqdm for backend logs usually
        try:
             # Try parquet first
            if file.suffix == '.parquet':
                 trades = pl.read_parquet(file)
            elif file.suffix == '.csv':
                 trades = pl.read_csv(file, try_parse_dates=True)
            elif file.suffix == '.zip':
                import zipfile
                with zipfile.ZipFile(file) as z:
                    # Assume there's one CSV or take the first one
                    csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                    if not csv_files:
                        continue
                    with z.open(csv_files[0]) as f:
                        trades = pl.read_csv(f.read(), try_parse_dates=True)
                        print(f"DEBUG: Read {len(trades)} rows from {csv_files[0]}")
                        print(f"DEBUG: Columns: {trades.columns}")
            else:
                 continue

            if "datetime" not in trades.columns:
                 # Check if we need to convert time
                 if "time" in trades.columns:
                     trades = trades.with_columns(pl.from_epoch("time", time_unit="ms").alias("datetime"))
                 else:
                     print(f"DEBUG: Missing 'datetime' or 'time' column. Found: {trades.columns}")
                     continue
            
            trades = trades.sort("datetime")
            
            ts = trades.group_by_dynamic(
                "datetime",
                every=time_interval,
                offset="0m"
            ).agg(aggs)
            
            ts_list.append(ts)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
            
    if not ts_list:
        return pl.DataFrame()
        
    result = pl.concat(ts_list)
    return result.sort("datetime")

def load_ohlc_timeseries(sym: str, time_interval: str, data_path: str = './data'):
    if sym == 'BTCUSDT':
        # Prioritize the 12h OHLC file as requested
        file_path = Path(data_path) / "BTCUSDT_12h_ohlc.csv"
        if file_path.exists():
            return pl.read_csv(file_path, try_parse_dates=True)
    return load_timeseries(sym, time_interval, OHLC_AGGS, data_path)

def add_lags(df: pl.DataFrame, target_col: str, max_lags: int, forecast_horizon: int = 1) -> pl.DataFrame:
    """Add lagged features to DataFrame."""
    # Create target (future return)
    df = df.with_columns(
        pl.col(target_col).shift(-forecast_horizon).alias("target")
    )
    
    # Add lags
    lag_cols = []
    for i in range(max_lags):
        period = i + 1  # 1 to max_lags
        lag_cols.append(pl.col(target_col).shift(period).alias(f"lag_{period}"))
        
    df = df.with_columns(lag_cols)
    return df.drop_nulls()

def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, (pl.Series, pl.DataFrame)):
        x = x.to_numpy()
    return torch.tensor(x, dtype=torch.float32 if dtype is None else dtype)

def timeseries_split(data: torch.Tensor, test_size: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split data sequentially (no shuffling)."""
    split_idx = int(len(data) * (1 - test_size))
    return data[:split_idx], data[split_idx:]

def batch_train_reg(
    model: nn.Module,
    X_train,
    X_test,
    y_train,
    y_test,
    no_epochs: int = 100,
    lr: float = 0.001,
    logging: bool = False
):
    """Train a PyTorch regression model."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(no_epochs):
        model.train()
        optimizer.zero_grad()
        
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        if logging and (epoch + 1) % (no_epochs // 10) == 0:
            model.eval()
            with torch.no_grad():
                y_test_pred = model(X_test)
                test_loss = criterion(y_test_pred, y_test)
                print(f"Epoch {epoch+1}/{no_epochs}, Train Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}")
                
    return model

def calculate_equity_curve(signals: np.ndarray, returns: np.ndarray) -> pl.DataFrame:
    """Calculate equity curve from signals and returns."""
    # Align lengths
    min_len = min(len(signals), len(returns))
    signals = signals[:min_len]
    returns = returns[:min_len]
    
    trade_returns = signals * returns
    
    # Using Polars for efficient calculation
    df = pl.DataFrame({
        "signal": signals,
        "return": returns,
        "trade_return": trade_returns
    })
    
    df = df.with_columns(
        (pl.col("trade_return") + 1).cum_prod().alias("equity_curve")
    )
    
    return df
