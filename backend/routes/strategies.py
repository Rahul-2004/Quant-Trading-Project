from fastapi import APIRouter
from typing import List, Dict
from ..strategies import STRATEGIES

router = APIRouter(prefix="/api", tags=["strategies"])

@router.get("/strategies")
def get_strategies() -> List[Dict[str, str]]:
    """List available trading strategies."""
    return [
        {"id": key, "name": strategy.name} 
        for key, strategy in STRATEGIES.items()
    ]

@router.get("/tickers")
def get_tickers() -> List[str]:
    """List available tickers (mocked for now, or scanned from data dir)."""
    # In a real app, scan the data/ folder or fetch from exchange
    return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
