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

## Test the app

```bash
pip install -r requirements.txt
pytest -q
```

## Run with Docker

```bash
docker build -t spike-market-signals .
docker run --rm -p 8000:8000 spike-market-signals
```

Then open `http://localhost:8000`.

## Deploy online

The repository includes `render.yaml` for Render and a `Dockerfile` for Docker-compatible hosts. On Render, create a new Blueprint from this repository and review the service settings before deploying. After deployment, verify `/api/health`, then try `/api/predict?symbol=SPY&horizon=5` and `/api/backtest?symbol=SPY&max_predictions=80`. Data availability depends on Yahoo Finance and can vary by ticker, time, and provider restrictions.

## API endpoints

- `GET /api/health` — service status
- `GET /api/predict?symbol=SPY&horizon=5` — current model estimate (horizon: 5 or 10)
- `GET /api/backtest?symbol=SPY&max_predictions=80` — walk-forward report for both horizons (20–200 predictions requested)

The historical test is a useful first diagnostic, not a complete research-grade validation: the forecast windows can overlap, observations are not independent, the sample is limited by intraday-data availability, and transaction costs/spread/slippage are not modeled. Use additional out-of-sample periods and paper trading before making any financial decisions.
