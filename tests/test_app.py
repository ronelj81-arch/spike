import numpy as np
import pandas as pd

from app import features, walk_forward_backtest


def synthetic_bars(size=650):
    rng = np.random.default_rng(17)
    # A gently drifting noisy series gives the model both UP and DOWN outcomes.
    returns = rng.normal(0.00005, 0.002, size)
    close = pd.Series(100 * np.exp(np.cumsum(returns)),
                      index=pd.date_range("2026-01-05 09:30", periods=size, freq="min"))
    volume = pd.Series(rng.integers(100, 5000, size), index=close.index)
    return close, volume


def test_features_have_expected_columns_and_no_infinities():
    close, volume = synthetic_bars()
    result = features(close, volume)
    assert {"return_1", "momentum_10", "volatility_10", "volume_change", "rsi"} <= set(result.columns)
    assert not np.isinf(result.to_numpy()).any()


def test_walk_forward_backtest_returns_metrics_for_both_horizons():
    close, volume = synthetic_bars()
    for horizon in (5, 10):
        result = walk_forward_backtest(close, volume, horizon, 20)
        assert result["horizon_minutes"] == horizon
        assert result["predictions_tested"] > 0
        assert 0 <= result["directional_accuracy_pct"] <= 100
        assert 0 <= result["signal_coverage_pct"] <= 100
        assert result["first_test_time"] <= result["last_test_time"]
