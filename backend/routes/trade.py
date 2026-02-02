from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal, Optional
from ..services.execution import execution_service

router = APIRouter(prefix="/api", tags=["trade"])

class TradeRequest(BaseModel):
    ticker: str
    side: Literal["BUY", "SELL"]
    quantity: float
    strategy_id: Optional[str] = None

class TradeResponse(BaseModel):
    status: str
    order_id: str
    message: str

@router.post("/trade", response_model=TradeResponse)
def execute_trade(request: TradeRequest):
    """Execute a trade."""
    try:
        result = execution_service.execute_order(
            ticker=request.ticker,
            side=request.side,
            quantity=request.quantity,
            strategy_id=request.strategy_id
        )
        
        return {
            "status": result["status"],
            "order_id": result["order_id"],
            "message": f"Successfully executed {request.side} {request.quantity} {request.ticker}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
