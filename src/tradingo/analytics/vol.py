"""Volatility calculation module."""

from typing import Callable

import numpy as np
import pandas as pd

from .expectation import safe_dataframe_window


def vol(
    dataframe: pd.DataFrame,
    annualisation: int = 260,
    how: str = "ewm",
    demean: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate the volatility of a given DataFrame.

    :param dataframe: DataFrame containing the data.
    :param annualisation: The annualisation factor, by default 260.
    :param how: A DataFrame method which returns a BaseWindow [pandas.core.window.rolling]
        for calculating volatility: { ewm | rolling | expanding }
    :param demean: Whether to demean the data, by default True.
            - True -> E[(X - E[X]) * (Y - E[Y])]
            - False -> E[X * Y]
        When True, it is the volatility.
        When False we calc the mean of the second moment, useful for raw signal magnitude,
        co-movement including mean shifts, or where you handle the mean explicitly.
        NOTE: because of returns being "somewhere around zero", this 'vol' accounts for
        the implicit bias within a timeseries (e.g. a typical long position)

    :param **kwargs: Additional arguments for rolling/ewm/... pandas methods.
        For rolling volatility, the window size must be specified in kwargs.
        For example, `window=10` for a 10-period rolling volatility.
        For exponential volatility, the halflife can be specified in kwargs.
        For example, `halflife=10` for a 10-period halflife.

    :returns: The vol
    """

    df = dataframe**2 if demean else dataframe

    agg_buffer = safe_dataframe_window(df, how, kwargs)

    agg_method = "mean" if demean else "std"
    std: Callable = getattr(agg_buffer, agg_method)

    return np.sqrt(annualisation * std()) if demean else np.sqrt(annualisation) * std()
