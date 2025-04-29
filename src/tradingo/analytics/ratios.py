"""Typical financial ratios."""

from typing import Optional

import numpy as np
import pandas as pd

from tradingo.analytics.expectation import expectation
from tradingo.analytics.vol import vol


def omega_ratio(returns: pd.Series, required_return: float = 0.0) -> float:
    """
    Calculate the Omega ratio of a strategy.

    :param returns: pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.

    :param required_return: float
        Minimum acceptance return of the investor. Threshold over which to
        consider positive vs negative returns. It will be converted to a
        value appropriate for the period of the returns. E.g. An annual minimum
        acceptable return of 100 will translate to a minimum acceptable
        return of 0.018.

    :returns omega_ratio : float
    """

    return_threshold = (1 + required_return) ** (1 / 252) - 1
    returns_less_thresh = returns - return_threshold
    numer = sum(returns_less_thresh[returns_less_thresh > 0.0])
    denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])
    if denom > 0.0:
        return numer / denom

    return np.nan


def sharpe_ratio(
    returns,
    annualisation=260,
    how="ewm",
    required_return: float = 0.0,
    volatility_floor: Optional[dict[str, float]] = None,
    **kwargs,
):
    """
    Calculate the sharpe ratio of a given dataframe of returns.

    :param dataframe: DataFrame containing the data.
    :param annualisation: The annualisation factor, by default 260.
    :param how: A DataFrame method which returns a BaseWindow [pandas.core.window.rolling]
        for calculating covariance: { ewm | rolling | expanding } or 'insample' for the full sample
    :param required_return: Minimum acceptance return of the investor
    :param volatility_floor: Minimum acceptance volatiltiy for each market
    :param **kwargs: keyword arguments
        Additional arguments for the ewm (exponentially weighted mean) method.
        For rolling expectation, the window size must be specified in kwargs.
        For example, `window=10` for a 10-period rolling expectation.
        For exponential expectation, the halflife can be specified in kwargs.
        For example, `halflife=10` for a 10-period halflife.

    :returns: The expectation of the data.
    """

    exp_returns = expectation(returns, annualisation=annualisation, how=how, **kwargs)
    volatility = vol(returns, annualisation=annualisation, how=how, **kwargs)

    # TODO: allow for non-constant volfloor, e.g. rolling quantile
    if volatility_floor is not None:
        volatility = volatility.apply(
            lambda col: (
                col.clip(lower=volatility_floor.get(col.name, 0.0))
                if col.name in volatility_floor
                else col
            )
        )

    sharpe = (exp_returns - required_return) / volatility

    return sharpe, volatility
