import json
import os
from datetime import datetime, timezone
from pathlib import Path

from app import download_minutes, walk_forward_backtest


def main():
    symbol = os.getenv("BACKTEST_SYMBOL", "SPY").strip().upper()
    max_predictions = int(os.getenv("BACKTEST_MAX_PREDICTIONS", "80"))
    bars = download_minutes(symbol)
    results = [
        walk_forward_backtest(bars["close"], bars["volume"], horizon, max_predictions)
        for horizon in (5, 10)
    ]
    report = {
        "symbol": symbol,
        "data_source": "Yahoo Finance via yfinance",
        "data_bars": int(len(bars)),
        "data_start": bars.index[0].isoformat(),
        "data_end": bars.index[-1].isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "limitations": [
            "Intraday one-minute history is short-window data, not a long-term validation.",
            "Results are historical simulation metrics, not a promise of future performance.",
            "Transaction costs, slippage, overlapping horizons and market-regime changes may affect results."
        ],
    }
    output = Path("backtest-report.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
