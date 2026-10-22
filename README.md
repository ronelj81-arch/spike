# Spike — Short-Term Market Signals

A small web app that estimates the probability of an upward or downward move over the next **5 or 10 minutes** for a stock or ETF ticker. It uses recent one-minute bars from Yahoo Finance via `yfinance` and a scikit-learn logistic regression model.

## Run locally

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000. API docs: http://127.0.0.1:8000/docs.

## Deploy

The included `render.yaml` is a starter configuration for deploying to Render. Connect this repository to Render and deploy the web service. Availability and limits depend on the hosting provider and its current plan.

## How it works

- Loads available one-minute bars for the selected symbol.
- Calculates recent returns, moving-average momentum, volatility, volume change, and RSI-like features.
- Trains a logistic regression on recent historical bars labeled by the chosen 5- or 10-minute forward direction.
- Displays model-derived UP/DOWN probabilities and a neutral band when neither side reaches 55%.

## Important limitations

This is an experimental educational prototype, **not a dependable trading system**. The probabilities are model estimates, not calibrated guarantees or proof of an edge. Minute data can be delayed, unavailable, or restricted, and this app does not include transaction costs, spread, slippage, market-regime validation, or rigorous walk-forward testing. It does not place trades. Verify data freshness and validate performance with out-of-sample backtests and paper trading before considering any financial use. Never invest money you cannot afford to lose.

## API

`GET /api/predict?symbol=SPY&horizon=5` (horizon must be 5 or 10 minutes)

`GET /api/health`


## Historical backtesting

The dashboard includes a **Run 5 & 10 min backtest** button. The API endpoint is:

`GET /api/backtest?symbol=SPY&max_predictions=80`

- Runs separate 5-minute and 10-minute walk-forward tests using recent one-minute bars.
- Uses an expanding training window and keeps training labels at least one forecast horizon behind each test prediction to reduce look-ahead leakage.
- Reports directional accuracy, an always-UP baseline, accuracy of non-neutral signals, signal coverage, and UP/DOWN call counts.
- `max_predictions` can be set from 20 to 200.

**Accuracy reports are generated only when the endpoint runs successfully against available market data.** No accuracy figures are hardcoded or fabricated in this repository. The provider may limit historical one-minute data. Treat results as exploratory: this basic test does not model spread, fees, slippage, or prove profitability, and its estimates may be unstable across market regimes.
