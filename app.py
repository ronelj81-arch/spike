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



def indicator_snapshot(bars):
    close = bars["close"].astype(float)
    volume = bars["volume"].astype(float)
    if len(close) < 60:
        raise ValueError("At least 60 one-minute bars are required to calculate indicators.")
    ema9 = close.ewm(span=9, adjust=False).mean()
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    std20 = close.rolling(20).std(ddof=0)
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(loss != 0, 100).where(~((gain == 0) & (loss == 0)), 50)
    vol_avg = volume.rolling(20).mean()
    latest = float(close.iloc[-1])

    def value(series):
        result = series.iloc[-1]
        return round(float(result), 4) if pd.notna(result) and np.isfinite(result) else None

    rsi_value = value(rsi)
    macd_value = value(macd)
    signal_value = value(macd_signal)
    if rsi_value is None:
        rsi_read = "Unavailable"
    elif rsi_value >= 70:
        rsi_read = "Potentially overbought"
    elif rsi_value <= 30:
        rsi_read = "Potentially oversold"
    else:
        rsi_read = "Middle range"
    return {
        "symbol": str(bars.attrs.get("symbol", "")),
        "last_price": round(latest, 4),
        "data_time": close.index[-1].isoformat(),
        "indicators": {
            "rsi_14": {"value": rsi_value, "interpretation": rsi_read},
            "macd": {"value": macd_value, "signal": signal_value,
                     "interpretation": ("Bullish momentum crossover" if macd_value is not None and signal_value is not None and macd_value > signal_value else "Bearish momentum crossover" if macd_value is not None and signal_value is not None and macd_value < signal_value else "Unavailable")},
            "sma_20": {"value": value(sma20), "position": "price above" if latest > float(sma20.iloc[-1]) else "price below"},
            "sma_50": {"value": value(sma50), "position": "price above" if latest > float(sma50.iloc[-1]) else "price below"},
            "ema_9": {"value": value(ema9)},
            "bollinger_upper": {"value": value(upper)},
            "bollinger_middle": {"value": value(sma20)},
            "bollinger_lower": {"value": value(lower)},
            "volume_vs_20_bar_avg_pct": round((float(volume.iloc[-1]) / float(vol_avg.iloc[-1]) - 1) * 100, 1) if pd.notna(vol_avg.iloc[-1]) and vol_avg.iloc[-1] > 0 else None
        },
        "method": "Indicators calculated from the latest available one-minute close and volume bars. SMA/EMA and Bollinger Bands are descriptive technical measures, not predictions.",
        "notice": "Indicators can lag, give conflicting signals, and behave differently across timeframes. They are not trading advice."
    }



def bullish_snapshot(bars, symbol):
    close = bars["close"].astype(float)
    volume = bars["volume"].astype(float)
    if len(close) < 60:
        raise ValueError("Not enough bars to score this ticker.")
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    ema9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).where(loss != 0, 100).where(~((gain == 0) & (loss == 0)), 50)
    rsi_now = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0
    avg_volume = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio = float(volume.iloc[-1] / avg_volume) if avg_volume > 0 else 0.0
    points = 0
    checks = []
    def check(name, passed, weight, detail):
        nonlocal points
        if passed:
            points += weight
        checks.append({"name": name, "bullish": bool(passed), "detail": detail})
    check("Price above SMA 20", last > sma20, 20, f"Last price {last:.2f}; SMA 20 {sma20:.2f}")
    check("Price above SMA 50", last > sma50, 20, f"Last price {last:.2f}; SMA 50 {sma50:.2f}")
    check("EMA 9 above SMA 20", ema9 > sma20, 15, f"EMA 9 {ema9:.2f}; SMA 20 {sma20:.2f}")
    check("MACD above signal", float(macd.iloc[-1]) > float(macd_signal.iloc[-1]), 20, "MACD momentum is above its signal line" if float(macd.iloc[-1]) > float(macd_signal.iloc[-1]) else "MACD is not above its signal line")
    check("RSI momentum", 50 <= rsi_now < 70, 15, f"RSI 14 is {rsi_now:.1f}; scanner rewards positive momentum without the >70 overbought zone")
    check("Volume confirmation", vol_ratio >= 1.2 and last > prev, 10, f"Latest volume is {vol_ratio:.2f}× its 20-bar average; latest bar {'rose' if last > prev else 'did not rise'}")
    label = "Strong bullish" if points >= 80 else "Bullish" if points >= 60 else "Mixed" if points >= 40 else "Weak / bearish"
    return {
        "symbol": symbol, "last_price": round(last, 4), "change_1m_pct": round((last / prev - 1) * 100, 3),
        "bullish_score": points, "classification": label, "rsi_14": round(rsi_now, 2),
        "volume_ratio": round(vol_ratio, 2), "data_time": close.index[-1].isoformat(), "checks": checks
    }

def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    )


def walk_forward_backtest(close, volume, horizon, max_predictions):
    x = features(close, volume)
    future_close = close.shift(-horizon)
    future_return = future_close / close - 1
    # Exclude session gaps: a next-session open is not a 5/10-minute outcome.
    timestamps = pd.Series(close.index, index=close.index)
    future_time = timestamps.shift(-horizon)
    elapsed_minutes = (future_time - timestamps).dt.total_seconds() / 60
    valid_horizon = elapsed_minutes.between(horizon * 0.8, horizon * 1.5)
    target = (future_return > 0).where(future_return.notna() & valid_horizon, np.nan)
    frame = x.copy()
    frame["target"] = target
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    # pandas can preserve boolean labels as object when missing labels were masked.
    # Convert the cleaned target to numeric 0/1 for current scikit-learn versions.
    frame["target"] = frame["target"].astype("int8")
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
        future_close = close.shift(-horizon)
        future_return = future_close / close - 1
        timestamps = pd.Series(close.index, index=close.index)
        future_time = timestamps.shift(-horizon)
        elapsed_minutes = (future_time - timestamps).dt.total_seconds() / 60
        valid_horizon = elapsed_minutes.between(horizon * 0.8, horizon * 1.5)
        train = x.copy()
        train["target"] = (future_return > 0).where(future_return.notna() & valid_horizon, np.nan)
        train = train.dropna()
        train["target"] = train["target"].astype("int8")
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



@app.get("/api/indicators")
def get_indicators(symbol: str = Query("SPY", min_length=1, max_length=12)):
    symbol = symbol.strip().upper()
    try:
        bars = download_minutes(symbol)
        bars.attrs["symbol"] = symbol
        return indicator_snapshot(bars)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc) or "Indicators could not be calculated.") from exc



@app.get("/api/scanner")
def bullish_scanner(symbols: str = Query("SPY,QQQ,AAPL,MSFT,NVDA,AMD,META,TSLA,AMZN", max_length=300)):
    requested = list(dict.fromkeys(s.strip().upper() for s in symbols.split(",") if s.strip()))
    if not requested or len(requested) > 12 or any(not all(c.isalnum() or c in ".^=-" for c in s) or len(s) > 12 for s in requested):
        raise HTTPException(status_code=400, detail="Provide 1–12 valid comma-separated ticker symbols.")
    results, errors = [], []
    for symbol in requested:
        try:
            bars = download_minutes(symbol)
            results.append(bullish_snapshot(bars, symbol))
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc) or "Data unavailable"})
    results.sort(key=lambda item: (item["bullish_score"], item["change_1m_pct"]), reverse=True)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "Rule-based technical score from six trend, momentum, and volume checks; not a probability or expected return.",
        "results": results, "errors": errors,
        "notice": "A score of 100 means all six configured bullish checks passed on the latest available one-minute bar. It does not mean a 100% chance of rising. Short-term indicators can fail and data may be delayed."
    }


@app.get("/api/chart")
def chart_data(symbol: str = Query("SPY", min_length=1, max_length=12),
               limit: int = Query(120, ge=40, le=300)):
    symbol = symbol.strip().upper()
    try:
        bars = download_minutes(symbol).tail(limit).copy()
        close = bars["close"].astype(float)
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema20 = close.ewm(span=20, adjust=False).mean()
        sma50 = close.rolling(50).mean()
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = (100 - 100 / (1 + rs)).where(loss != 0, 100).where(~((gain == 0) & (loss == 0)), 50)
        rows = []
        prior_bull = False
        for i, (ts, price) in enumerate(close.items()):
            e9, e20 = float(ema9.iloc[i]), float(ema20.iloc[i])
            s50 = float(sma50.iloc[i]) if pd.notna(sma50.iloc[i]) else None
            current_bull = e9 > e20
            previous_bull = bool(ema9.iloc[i-1] > ema20.iloc[i-1]) if i else current_bull
            signal = None
            # Mark only crossover bars with trend/RSI confirmation.
            rsi_i = float(rsi.iloc[i]) if pd.notna(rsi.iloc[i]) else None
            if i and current_bull and not previous_bull and (s50 is None or price > s50) and (rsi_i is None or 50 <= rsi_i < 70):
                signal = "BUY WATCH"
            elif i and not current_bull and previous_bull:
                signal = "SELL / EXIT WATCH"
            rows.append({
                "time": ts.isoformat(), "price": round(float(price), 4),
                "ema9": round(e9, 4), "ema20": round(e20, 4),
                "sma50": round(s50, 4) if s50 is not None else None,
                "rsi14": round(rsi_i, 2) if rsi_i is not None else None,
                "signal": signal
            })
        return {
            "symbol": symbol, "bars": rows,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "method": "Illustrative technical markers: BUY WATCH on EMA 9 crossing above EMA 20 with trend/RSI checks; SELL / EXIT WATCH on EMA 9 crossing below EMA 20.",
            "notice": "Markers are retrospective technical rule triggers, not personalized financial advice or guaranteed entry/exit prices. Confirm live data, spreads, liquidity, and risk. No stop-loss or position size is implied."
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc) or "Chart data could not be loaded.") from exc

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
