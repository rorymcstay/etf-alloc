import numpy as np
import pandas as pd
import pytest

from tradingo.analytics.returns import (
    compounded_returns,
    log_returns,
    pct_returns,
    returns,
    value_returns,
    fx_adjusted_returns,
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


@pytest.fixture
def price_dataframe():
    data = {
        "AAPL": [150, 155, 160, 165, 170],
        "MSFT": [300, 310, 320, 330, 340],
    }
    return pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=5))


@pytest.fixture
def fx_dataframe():
    data = {
        "USD": [1.0, 0.95, 1.0, 1.05, 1.1],
        "EUR": [0.85, 0.83, 0.84, 0.86, 0.88],
    }
    return pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=5))


@pytest.fixture
def symbols_ccys():
    return {"AAPL": "USD", "MSFT": "EUR"}


def test_fx_adjusted_returns(price_dataframe, fx_dataframe, symbols_ccys):
    val = fx_adjusted_returns(price_dataframe, fx_dataframe, symbols_ccys, kind="pct")
    expected_aapl = pct_returns(price_dataframe["AAPL"], fx_dataframe["USD"])
    expected_msft = pct_returns(price_dataframe["MSFT"], fx_dataframe["EUR"])
    expected = pd.concat([expected_aapl, expected_msft], axis=1)
    expected.columns = ["AAPL", "MSFT"]
    pd.testing.assert_frame_equal(val, expected)


def test_fx_adjusted_returns_invalid_symbols(price_dataframe, fx_dataframe):
    invalid_symbols_ccys = {"AAPL": "USD", "GOOG": "EUR"}
    with pytest.raises(
        ValueError, match="prices columns .* do not match currency_map symbols .*"
    ):
        fx_adjusted_returns(price_dataframe, fx_dataframe, invalid_symbols_ccys)


def test_fx_adjusted_returns_missing_fx(price_dataframe, fx_dataframe, symbols_ccys):
    incomplete_fx_dataframe = fx_dataframe.drop(columns=["EUR"])
    with pytest.raises(ValueError, match="fx_series columns miss currencies: .*"):
        fx_adjusted_returns(price_dataframe, incomplete_fx_dataframe, symbols_ccys)


if __name__ == "__main__":
    pytest.main([__file__])
