import pandas as pd
import numpy as np
import pytest

from tradingo.analytics.returns import (
    value_returns,
    pct_returns,
    log_returns,
    compounded_returns,
    returns,
)


@pytest.fixture
def price_series():
    return pd.Series(
        [100, 105, 110, 120, 115], index=pd.date_range("2024-01-01", periods=5)
    )


@pytest.fixture
def fx_series():
    return pd.Series(
        [1.0, 0.95, 1.0, 1.05, 1.1], index=pd.date_range("2024-01-01", periods=5)
    )


def test_value_returns(price_series, fx_series):
    val = value_returns(price_series, fx_series)
    expected = price_series * fx_series
    expected = expected - expected.shift(1)
    expected = expected.fillna(0.0)
    pd.testing.assert_series_equal(val, expected)


def test_pct_returns(price_series, fx_series):
    val = pct_returns(price_series, fx_series)
    adjusted = price_series * fx_series
    expected = (adjusted - adjusted.shift(1)) / adjusted.shift(1)
    expected = expected.fillna(0.0)
    pd.testing.assert_series_equal(val, expected)


def test_log_returns(price_series, fx_series):
    val = log_returns(price_series, fx_series)
    adjusted = price_series * fx_series
    expected = np.log(adjusted / adjusted.shift(1)).fillna(0.0)
    pd.testing.assert_series_equal(val, expected)


def test_compounded_returns(price_series, fx_series):
    val = compounded_returns(price_series, fx_series, period=1)
    pct_ret = pct_returns(price_series, fx_series, period=1)
    compounded = (1 + pct_ret).cumprod() - 1
    expected = (1 + compounded) / (1 + compounded.shift(1)) - 1
    pd.testing.assert_series_equal(val, expected, rtol=1e-8)


@pytest.mark.parametrize(
    "kind,func",
    [
        ("value", value_returns),
        ("pct", pct_returns),
        ("log", log_returns),
        ("compounded", compounded_returns),
    ],
)
def test_returns_api_dispatch(price_series, kind, func):
    unified = returns(price_series, kind=kind)
    direct = func(price_series)
    pd.testing.assert_series_equal(unified, direct)


def test_returns_api_invalid_kind(price_series):
    with pytest.raises(ValueError, match="Unknown return kind"):
        returns(price_series, kind="unknown")


if __name__ == "__main__":
    pytest.main([__file__])
