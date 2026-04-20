from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import backtest, findings, strategies, trade

app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(strategies.router)
app.include_router(backtest.router)
app.include_router(trade.router)
app.include_router(findings.router)

@app.get("/")
def read_root():
    return {"message": "Trading Dashboard API"}
