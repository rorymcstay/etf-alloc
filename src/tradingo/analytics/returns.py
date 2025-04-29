"""Analytics to calculate returns"""

from typing import Union

import pandas as pd


def align_series(series: pd.Series, other: pd.Series = None) -> tuple[pd.Series, pd.Series]:
    """align a series to another ahead of returns calculation"""

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


def returns(
    prices: pd.Series,
    fx: Union[pd.Series, float] = 1.0,
    period: int = 1,
    pct: bool = True,
    overlap: int = 1,
    freq: str = None,
) -> pd.Series:
    """
    simple returns.

    :param prices: price series
    :param fx: fx series
    :param period: no. of periods to calculate returns on. for daily prices
        period=1 calculates daily returns, period=5 returns over one week.
    :param pct: True for pct returns, False in share value (accounting fx)
    :param overlap: mean returns on a window to increase robustness.
    :param freq: to resample on a given frequency
    """

    prices, fx = align_series(prices.ffill(), fx)
    share_value = prices.mul(fx)

    pct_returns = (
        (share_value - share_value.shift(period)) / share_value.shift(period)
    ).fillna(0.0)

    if freq:
        pct_returns = (1.0 + pct_returns).resample(freq).prod() - 1.0

    # overlap returns on multiple days - more robust
    if overlap > 1:
        pct_returns = pct_returns.rolling(overlap).mean()

    if pct:
        return pct_returns

    return share_value.mul(pct_returns)


def compounded_returns(
    prices: pd.Series,
    fx: Union[pd.Series, float] = 1.0,
    period: int = 1,
    pct: bool = True,
    overlap: int = 1,
    freq: str = None,
) -> pd.Series:
    """
    compounded returns.
    """

    pct_returns = returns(prices, fx, period=1, pct=True, overlap=overlap, freq=freq)

    compounded = (((pct_returns + 1).cumprod()) - 1).fillna(0)

    if period:
        assert (
            compounded.shift(period).notna().any().all()
        ), f"no data within the available window: {period}"
        compounded = (1 + compounded) / (1 + compounded.shift(period)) - 1

    if pct:
        return compounded

    prices, fx = align_series(prices.ffill(), fx)
    share_value = prices.mul(fx)

    return share_value.mul(compounded)
