"""
Functions to compute moving window related quantities.
ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
"""

from typing import Optional

import numpy as np
import pandas as pd


def decay_from_halflife(halflife):
    """
    Compute the decay (alpha) from halflife.
    """
    assert halflife > 0, "Halflife must be positive"
    return 1 - np.exp(-np.log(2) / halflife)


def span_from_halflife(halflife):
    """
    Compute the span from halflife.
    """
    assert halflife > 0, "Halflife must be positive"
    alpha = 1 - np.exp(-np.log(2) / halflife)
    return (2 / alpha) - 1


def com_from_halflife(halflife):
    """
    Compute the center of mass from halflife.
    """
    assert halflife > 0, "Halflife must be positive"
    alpha = decay_from_halflife(halflife)
    return (1 - alpha) / alpha


def effective_sample_size(halflife: int, check_for_series_length: Optional[int] = None):
    """
    Compute the effective sample size for an EWMA from its halflife.
    The effective sample size is the number of independent samples that would give
    the same variance as the exponentially weighted moving average (EWMA) with the given halflife.

    NOTE: With short series with series_length < ess, the ess is not representative.
    Use warmup to remove the initial bias, or use the effective sample size as the number
    of observations.
    """
    alpha = decay_from_halflife(halflife)
    ess = 2.0 / alpha
    if check_for_series_length is not None:
        assert (
            check_for_series_length > ess
        ), "Series length must be greater than effective sample size"
    return ess


def safe_dataframe_sample_size(
    dataframe: pd.DataFrame, how: str, kwargs
) -> int | pd.DataFrame:
    """
    define the sample size of a dataframe with multiple aggregation window methods

    :param dataframe: DataFrame containing the data.
    :param how: A DataFrame method which returns a BaseWindow [pandas.core.window.rolling]
        for calculating covariance: { ewm | rolling | expanding } or 'insample' for the full sample
    :param kwargs: Additional arguments for the window method.
        For rolling expectation, the window size must be specified in kwargs.
        For example, `window=10` for a 10-period rolling expectation.
        For exponential expectation, the halflife can be specified in kwargs.
        For example, `halflife=10` for a 10-period halflife.
    :returns: The sample size of the data.
    """

    if how == "insample":
        if kwargs:
            raise ValueError(
                f"no kwargs should be passed when evaluating the full sample, got: {kwargs}"
            )
        return len(dataframe)

    _sample_size_method = {
        "ewm": lambda df: effective_sample_size(kwargs["halflife"], len(df)),
        "rolling": lambda df: kwargs["window"],
        "expanding": lambda df: pd.DataFrame(
            {c: range(1, len(df) + 1) for c in df.columns}
        ),
    }
    try:
        func = _sample_size_method[how]
    except KeyError as ex:
        raise NotImplementedError(f"how={how}") from ex
    return func(dataframe)
