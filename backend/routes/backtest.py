from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from ..strategies import STRATEGIES

router = APIRouter(prefix="/api", tags=["backtest"])

class BacktestRequest(BaseModel):
    strategy_id: str
    ticker: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class BacktestResponse(BaseModel):
    strategy: str
    total_return: float
    win_rate: float
    sharpe: float
    equity_curve: List[float]

@router.post("/backtest", response_model=BacktestResponse)
def run_backtest(request: BacktestRequest):
    """Run a backtest for a specific strategy and ticker."""
    strategy = STRATEGIES.get(request.strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    try:
        results = strategy.run_backtest(
            ticker=request.ticker, 
            start_date=request.start_date, 
            end_date=request.end_date
        )
        return results
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Data not found for ticker {request.ticker}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
