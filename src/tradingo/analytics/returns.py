"""functions to calculate returns"""

__all__ = [
    "value_returns",
    "pct_returns",
    "log_returns",
    "compounded_returns",
    "returns",
    "fx_adjusted_returns",
]

from typing import Optional

import numpy as np
import pandas as pd


def _align_series(
    series: pd.Series,
    other: pd.Series | float | int | None = None,
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
    fx: pd.Series | float = 1.0,
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
    fx: pd.Series | float = 1.0,
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
    fx: pd.Series | float = 1.0,
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
    fx: pd.Series | float = 1.0,
    period: int = 1,
    freq: Optional[str] = None,
) -> pd.Series:
    """
    compounded returns across n periods.

    :param prices: price series
    :param fx: FX series
    :param period: no. of periods to calculate returns on. for daily prices
        period=1 calculates daily returns, period=5 returns over one week.
    :param freq: to resample on a given frequency
    """

    pct_returns_ = pct_returns(prices, fx, period=1, freq=freq)

    returns_ = ((pct_returns_ + 1).cumprod()) - 1

    if period:
        if not returns_.shift(period).notna().any().all():
            raise ValueError(f"no data within the available window: {period}")
        returns_ = (1 + returns_) / (1 + returns_.shift(period)) - 1

    return returns_


_RETURNS_FUNCS = {
    "value": value_returns,
    "pct": pct_returns,
    "log": log_returns,
    "compounded": compounded_returns,
}


def returns(
    prices: pd.DataFrame | pd.Series,
    period: int = 1,
    freq: Optional[str] = None,
    kind: str = "pct",
) -> pd.DataFrame | pd.Series:
    """
    Compute different types of returns in a unified API.

    :param prices: price series or dataframe (if so, all must be in the same currency)
    :param period: number of periods to calculate returns
    :param freq: optional frequency to resample
    :param kind: one of {'value', 'pct', 'log', 'compounded'}
    """

    if isinstance(prices, pd.DataFrame):
        return pd.concat(
            (
                returns(prices[col], period=period, freq=freq, kind=kind)
                for col in prices.columns
            ),
            axis=1,
        )

    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a Series or DataFrame, not {type(prices)}")

    try:
        returns_func = _RETURNS_FUNCS[kind.lower()]
    except KeyError as ex:
        raise ValueError(f"Unknown return kind: {kind}") from ex

    return returns_func(prices, fx=1.0, period=period, freq=freq)


def fx_adjusted_returns(
    prices: pd.DataFrame,
    fx_series: pd.DataFrame,
    symbols_ccys: dict[str, str],
    period: int = 1,
    freq: Optional[str] = None,
    kind: str = "pct",
) -> pd.DataFrame:
    """
    Calculate returns accounting for prices in multiple currencies.

    :param prices: dataframe with prices
    :param fx_series: dataframe with FX series (all needed currencies)
    :param symbols_ccys: mapping of symbols to currencies
    :param period: number of periods to calculate returns
    :param freq: optional frequency to resample
    :param kind: one of {'value', 'pct', 'log', 'compounded'}
    """

    try:
        returns_func = _RETURNS_FUNCS[kind.lower()]
    except KeyError as ex:
        raise ValueError(f"Unknown return kind: {kind}") from ex

    if set(symbols_ccys) != set(prices.columns):
        raise ValueError(
            f"prices columns {set(prices.columns)} do not match currency_map symbols {set(symbols_ccys)}"
        )

    ccy_syms_map = {}
    for symbol, ccy in symbols_ccys.items():
        if ccy not in ccy_syms_map:
            ccy_syms_map[ccy] = []
        ccy_syms_map[ccy].append(symbol)
    if missing_ccys := set(ccy_syms_map).difference(set(fx_series.columns)):
        raise ValueError(f"fx_series columns miss currencies: {missing_ccys}")

    return pd.concat(
        (
            returns_func(prices[symbol], fx=fx_series[ccy], period=period, freq=freq)
            for ccy, symbols in ccy_syms_map.items()
            for symbol in symbols
        ),
        axis=1,
    )
