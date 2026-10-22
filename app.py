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
app = FastAPI(title="Spike Market Signals", version="1.1.0")
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
    loss = -ret.clip(upper=0).rolling(14).mean()
    gain = ret.clip(lower=0).rolling(14).mean()
    frame["rsi"] = 100 - 100 / (1 + loss / gain.replace(0, np.nan))
    return frame.replace([np.inf, -np.inf], np.nan)


def download_minutes(symbol):
    data = yf.download(symbol, period="7d", interval="1m", progress=False,
                       auto_adjust=True, threads=False)
    if data.empty:
        raise ValueError("No intraday data returned. Check the ticker and try during market hours.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if "Close" not in data or "Volume" not in data:
        raise ValueError("The market-data provider did not return required price and volume columns.")
    aligned = pd.concat([
        data["Close"].rename("close"),
        data["Volume"].rename("volume")
    ], axis=1).dropna(subset=["close"])
    aligned["volume"] = aligned["volume"].fillna(0)
    aligned = aligned[~aligned.index.duplicated(keep="last")].sort_index()
    if len(aligned) < 250:
        raise ValueError("Not enough one-minute bars for a useful walk-forward test.")
    return aligned


def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    )


def walk_forward_backtest(close, volume, horizon, max_predictions):
    x = features(close, volume)
    future_return = close.shift(-horizon) / close - 1
    # A target is usable only after its full future horizon has elapsed.
    target = (future_return > 0).astype(int)
    frame = x.copy()
    frame["target"] = target
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 220:
        raise ValueError("Not enough clean feature rows after indicator warm-up.")

    # Test only a recent, held-out segment. Each test prediction is generated
    # using an expanding training window whose labels are fully known at that time.
    stride = max(1, int(np.ceil(len(frame) / max_predictions)))
    first_test = max(160, int(len(frame) * 0.55))
    test_positions = list(range(first_test, len(frame) - horizon, stride))
    test_positions = test_positions[-max_predictions:]
    rows = []
    features_only = frame.drop(columns="target")
    targets = frame["target"]
    for pos in test_positions:
        train_end = pos - horizon
        if train_end < 120:
            continue
        train_x = features_only.iloc[:train_end]
        train_y = targets.iloc[:train_end]
        valid = train_x.notna().all(axis=1) & train_y.notna()
        train_x, train_y = train_x.loc[valid], train_y.loc[valid]
        if len(train_x) < 100 or train_y.nunique() < 2:
            continue
        model = make_model()
        model.fit(train_x, train_y)
        sample = features_only.iloc[[pos]]
        if sample.isna().any(axis=None):
            continue
        p_up = float(model.predict_proba(sample)[0][list(model.classes_).index(1)])
        actual_up = int(targets.iloc[pos])
        predicted = "UP" if p_up >= 0.55 else ("DOWN" if p_up <= 0.45 else "NEUTRAL")
        rows.append({
            "time": str(frame.index[pos]),
            "probability_up": p_up,
            "actual_up": actual_up,
            "predicted": predicted,
            "correct": int((p_up >= 0.5) == bool(actual_up)),
        })

    if not rows:
        raise ValueError("The walk-forward test could not produce predictions. Try another ticker.")
    total = len(rows)
    directional = [r for r in rows if r["predicted"] != "NEUTRAL"]
    correct = sum(r["correct"] for r in rows)
    signal_correct = sum(
        int((r["predicted"] == "UP" and r["actual_up"] == 1) or
            (r["predicted"] == "DOWN" and r["actual_up"] == 0))
        for r in directional
    )
    actual_up_count = sum(r["actual_up"] for r in rows)
    actual_down_count = total - actual_up_count
    up_calls = [r for r in directional if r["predicted"] == "UP"]
    down_calls = [r for r in directional if r["predicted"] == "DOWN"]
    return {
        "horizon_minutes": horizon,
        "predictions_tested": total,
        "directional_accuracy_pct": round(correct / total * 100, 1),
        "always_up_baseline_pct": round(actual_up_count / total * 100, 1),
        "signal_coverage_pct": round(len(directional) / total * 100, 1),
        "signal_accuracy_pct": round(signal_correct / len(directional) * 100, 1) if directional else None,
        "up_calls": len(up_calls),
        "up_call_accuracy_pct": round(sum(r["actual_up"] == 1 for r in up_calls) / len(up_calls) * 100, 1) if up_calls else None,
        "down_calls": len(down_calls),
        "down_call_accuracy_pct": round(sum(r["actual_up"] == 0 for r in down_calls) / len(down_calls) * 100, 1) if down_calls else None,
        "first_test_time": rows[0]["time"],
        "last_test_time": rows[-1]["time"],
        "caveat": "Historical walk-forward simulation only. Overlapping horizons, data quality, costs, slippage, and regime changes can affect results; accuracy does not guarantee future performance."
    }


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
        bars = download_minutes(symbol)
        close, volume = bars["close"], bars["volume"]
        x = features(close, volume)
        future_return = close.shift(-horizon) / close - 1
        train = x.copy()
        train["target"] = (future_return > 0).astype(int)
        train = train.iloc[:-horizon].dropna()
        current_x = x.dropna().iloc[[-1]]
        if len(train) < 100 or current_x.empty or train["target"].nunique() < 2:
            raise ValueError("Not enough usable intraday history to form a signal. Try again later.")
        model = make_model()
        model.fit(train.drop(columns="target"), train["target"])
        probability_up = float(model.predict_proba(current_x)[0][list(model.classes_).index(1)])
        probability_down = 1.0 - probability_up
        last_price = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        change = (last_price / prev - 1) * 100
        signal = "UP" if probability_up >= 0.55 else ("DOWN" if probability_up <= 0.45 else "NEUTRAL")
        return {
            "symbol": symbol, "horizon_minutes": horizon, "signal": signal,
            "probability_up": round(probability_up * 100, 1),
            "probability_down": round(probability_down * 100, 1),
            "last_price": round(last_price, 4), "last_bar_change_pct": round(change, 4),
            "data_time": close.index[-1].isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bars_used": int(len(train)),
            "method": "Logistic regression on recent 1-minute price, momentum, volatility and volume features",
            "notice": "Experimental estimate, not a reliable guarantee or financial advice. Intraday data can be delayed or unavailable."
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc) or "Market data could not be loaded.") from exc


@app.get("/api/backtest")
def backtest(symbol: str = Query("SPY", min_length=1, max_length=12),
             max_predictions: int = Query(80, ge=20, le=200)):
    """Run chronological expanding-window tests for both 5- and 10-minute horizons."""
    symbol = symbol.strip().upper()
    try:
        bars = download_minutes(symbol)
        results = [
            walk_forward_backtest(bars["close"], bars["volume"], horizon, max_predictions)
            for horizon in (5, 10)
        ]
        return {
            "symbol": symbol,
            "data_bars": int(len(bars)),
            "data_start": bars.index[0].isoformat(),
            "data_end": bars.index[-1].isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "method": "Expanding-window walk-forward validation; each prediction is trained only on observations whose full future outcome was already known at prediction time.",
            "results": results,
            "notice": "These results are historical estimates, not evidence of guaranteed or profitable future performance."
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc) or "Backtest failed.") from exc
