import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
app = FastAPI(title="Spike Market Signals", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE), name="static")

def features(close, volume):
    ret = close.pct_change()
    frame = pd.DataFrame(index=close.index)
    frame["return_1"] = ret
    frame["return_3"] = close.pct_change(3)
    frame["return_5"] = close.pct_change(5)
    frame["momentum_10"] = close / close.rolling(10).mean() - 1
    frame["momentum_20"] = close / close.rolling(20).mean() - 1
    frame["volatility_10"] = ret.rolling(10).std()
    frame["volume_change"] = volume / volume.rolling(20).mean() - 1
    frame["rsi"] = 100 - 100 / (1 + (-ret.clip(upper=0).rolling(14).mean() /
                                      ret.clip(lower=0).rolling(14).mean().replace(0, np.nan)))
    return frame.replace([np.inf, -np.inf], np.nan)

@app.get("/")
def home():
    return FileResponse(BASE / "index.html")

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Spike Market Signals"}

@app.get("/api/predict")
def predict(symbol: str = Query("SPY", min_length=1, max_length=12),
            horizon: int = Query(5, description="Forecast horizon in minutes; 5 or 10")):
    symbol = symbol.strip().upper()
    if horizon not in (5, 10):
        raise HTTPException(status_code=400, detail="Horizon must be 5 or 10 minutes.")
    try:
        data = yf.download(symbol, period="7d", interval="1m", progress=False, auto_adjust=True, threads=False)
        if data.empty:
            raise ValueError("No recent intraday data was returned. Check the ticker or try during market hours.")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        close, volume = data["Close"].dropna(), data["Volume"].reindex(data.index).fillna(0)
        aligned = pd.concat([close.rename("close"), volume.rename("volume")], axis=1).dropna()
        close, volume = aligned["close"], aligned["volume"]
        x = features(close, volume)
        future_return = close.shift(-horizon) / close - 1
        train = x.copy()
        train["target"] = (future_return > 0).astype(int)
        train = train.iloc[:-horizon].dropna()
        current_x = x.dropna().iloc[[-1]]
        if len(train) < 100 or current_x.empty or train["target"].nunique() < 2:
            raise ValueError("Not enough usable intraday history to form a signal. Try again later.")
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        model.fit(train.drop(columns="target"), train["target"])
        probability_up = float(model.predict_proba(current_x)[0][list(model.classes_).index(1)])
        probability_down = 1.0 - probability_up
        last_price = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        change = (last_price / prev - 1) * 100
        signal = "UP" if probability_up >= 0.55 else ("DOWN" if probability_up <= 0.45 else "NEUTRAL")
        recent_time = close.index[-1]
        timestamp = recent_time.isoformat() if hasattr(recent_time, "isoformat") else str(recent_time)
        return {
            "symbol": symbol, "horizon_minutes": horizon, "signal": signal,
            "probability_up": round(probability_up * 100, 1),
            "probability_down": round(probability_down * 100, 1),
            "last_price": round(last_price, 4), "last_bar_change_pct": round(change, 4),
            "data_time": timestamp, "generated_at": datetime.now(timezone.utc).isoformat(),
            "bars_used": int(len(train)),
            "method": "Logistic regression on recent 1-minute price, momentum, volatility and volume features",
            "notice": "Experimental estimate, not a reliable guarantee or financial advice. Intraday data can be delayed or unavailable."
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc) or "Market data could not be loaded.") from exc
