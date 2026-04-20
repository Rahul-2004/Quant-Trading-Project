from fastapi import APIRouter
from typing import List, Dict, Any

from ..strategies import get_ranked_strategy_items

router = APIRouter(prefix="/api", tags=["strategies"])

@router.get("/strategies")
def get_strategies() -> List[Dict[str, Any]]:
    """List available trading strategies."""
    return get_ranked_strategy_items()

@router.get("/tickers")
def get_tickers() -> List[str]:
    """List available tickers (mocked for now, or scanned from data dir)."""
    # In a real app, scan the data/ folder or fetch from exchange
    return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
