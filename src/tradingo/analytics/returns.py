"""functions to calculate returns"""

__all__ = [
    "value_returns",
    "pct_returns",
    "log_returns",
    "compounded_returns",
    "returns",
]

from typing import Optional, Union

import numpy as np
import pandas as pd


def _align_series(
    series: pd.Series,
    other: Union[pd.Series, float, int, None] = None,
) -> tuple[pd.Series, pd.Series]:
    """
    align a series to another ahead of returns calculation

    :param series: reference series
    :param other: other series or value to align
    """

    if other is None:
        other = pd.Series(1.0, index=series.index, name=series.name)
    elif isinstance(other, (float, int)):
        other = pd.Series(other, index=series.index, name=series.name)
    elif isinstance(other, pd.Series):
        common_idx = series.index.intersection(other.index)
        series = series.reindex(common_idx).dropna()
        other = other.reindex(common_idx).dropna()
    else:
        raise TypeError(type(other))

    return series, other


def value_returns(
    prices: pd.Series,
    fx: Union[pd.Series, float] = 1.0,
    period: int = 1,
    freq: Optional[str] = None,
) -> pd.Series:
    """
    share value returns, adjusting prices by FX.

    :param prices: price series
    :param fx: FX series
    :param period: no. of periods to calculate returns on. for daily prices
        period=1 calculates daily returns, period=5 returns over one week.
    :param freq: to resample on a given frequency
    """

    prices, fx = _align_series(prices.ffill(), fx)
    share_value = prices.mul(fx)

    returns_ = ((share_value - share_value.shift(period))).fillna(0.0)

    if freq:
        returns_ = returns_.resample(freq).sum()

    return returns_


def pct_returns(
    prices: pd.Series,
    fx: Union[pd.Series, float] = 1.0,
    period: int = 1,
    freq: Optional[str] = None,
) -> pd.Series:
    """
    simple percentage returns.

    :param prices: price series
    :param fx: FX series
    :param period: no. of periods to calculate returns on. for daily prices
        period=1 calculates daily returns, period=5 returns over one week.
    :param freq: to resample on a given frequency
    """

    prices, fx = _align_series(prices.ffill(), fx)
    share_value = prices.mul(fx)

    returns_ = (
        (share_value - share_value.shift(period)) / share_value.shift(period)
    ).fillna(0.0)

    if freq:
        returns_ = (1.0 + returns_).resample(freq).prod() - 1.0

    return returns_


def log_returns(
    prices: pd.Series,
    fx: Union[pd.Series, float] = 1.0,
    period: int = 1,
    freq: Optional[str] = None,
) -> pd.Series:
    """
    log percentage returns.

    :param prices: price series
    :param fx: FX series
    :param period: no. of periods to calculate returns on. for daily prices
        period=1 calculates daily returns, period=5 returns over one week.
    :param freq: to resample on a given frequency
    """

    prices, fx = _align_series(prices.ffill(), fx)
    share_value = prices.mul(fx)

    returns_ = np.log(share_value / share_value.shift(period)).fillna(0.0)

    if freq:
        returns_ = returns_.resample(freq).sum()

    return returns_


def compounded_returns(
    prices: pd.Series,
    fx: Union[pd.Series, float] = 1.0,
    period: int = 1,
    freq: Optional[str] = None,
) -> pd.Series:
    """
    compounded returns.

    :param prices: price series
    :param fx: FX series
    :param period: no. of periods to calculate returns on. for daily prices
        period=1 calculates daily returns, period=5 returns over one week.
    :param freq: to resample on a given frequency
    """

    pct_returns_ = pct_returns(prices, fx, period=1, freq=freq)

    returns_ = (((pct_returns_ + 1).cumprod()) - 1)

    if period:
        if not returns_.shift(period).notna().any().all():
            raise ValueError(f"no data within the available window: {period}")
        returns_ = (1 + returns_) / (1 + returns_.shift(period)) - 1

    return returns_


def returns(
    prices: pd.Series,
    fx: Union[pd.Series, float] = 1.0,
    period: int = 1,
    freq: Optional[str] = None,
    kind: str = "pct",
) -> pd.Series:
    """
    Compute different types of returns in a unified API.

    :param prices: price series
    :param fx: FX series or scalar
    :param period: number of periods to calculate returns
    :param freq: optional frequency to resample
    :param kind: one of {'value', 'pct', 'log', 'compounded'}
    """
    kind = kind.lower()
    if kind == "value":
        return value_returns(prices, fx=fx, period=period, freq=freq)
    if kind == "pct":
        return pct_returns(prices, fx=fx, period=period, freq=freq)
    if kind == "log":
        return log_returns(prices, fx=fx, period=period, freq=freq)
    if kind == "compounded":
        return compounded_returns(prices, fx=fx, period=period, freq=freq)
    raise ValueError(f"Unknown return kind: {kind!r}")
