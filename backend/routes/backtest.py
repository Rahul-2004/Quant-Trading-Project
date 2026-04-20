from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..strategies import PrecomputedStrategy, STRATEGIES

router = APIRouter(prefix="/api", tags=["backtest"])

class YearAllocationOverride(BaseModel):
    year: int
    btc: float
    gold: float
    silver: float


class BacktestRequest(BaseModel):
    strategy_id: str
    ticker: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    year_overrides: List[YearAllocationOverride] = Field(default_factory=list)

class BacktestResponse(BaseModel):
    strategy: str
    total_return: float
    win_rate: float
    sharpe: float
    cagr: float = 0.0
    equity_curve: List[float]
    equity_timestamps: List[str] = Field(default_factory=list)
    equity_allocations: List[dict[str, float]] = Field(default_factory=list)
    in_sample_curve: List[float] = Field(default_factory=list)
    in_sample_timestamps: List[str] = Field(default_factory=list)
    in_sample_allocations: List[dict[str, float]] = Field(default_factory=list)
    out_of_sample_curve: List[float] = Field(default_factory=list)
    out_of_sample_timestamps: List[str] = Field(default_factory=list)
    out_of_sample_allocations: List[dict[str, float]] = Field(default_factory=list)
    in_sample_total_return: float = 0.0
    out_of_sample_total_return: float = 0.0
    out_of_sample_start_index: int = 0
    applied_overrides: List[YearAllocationOverride] = Field(default_factory=list)

@router.post("/backtest", response_model=BacktestResponse)
def run_backtest(request: BacktestRequest):
    """Run a backtest for a specific strategy and ticker."""
    strategy = STRATEGIES.get(request.strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")

    if request.year_overrides and not isinstance(strategy, PrecomputedStrategy):
        raise HTTPException(
            status_code=400,
            detail="Year allocation overrides are only supported for precomputed multi-asset strategies.",
        )
    
    try:
        results = strategy.run_backtest(
            ticker=request.ticker, 
            start_date=request.start_date, 
            end_date=request.end_date,
            year_overrides=[override.model_dump() for override in request.year_overrides],
        )
        return results
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Data not found for ticker {request.ticker}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
