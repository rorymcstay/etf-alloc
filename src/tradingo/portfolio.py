import logging
import random

import pandas as pd
import numpy as np

import arcticdb as adb

from tradingo.symbols import lib_provider, symbol_provider, symbol_publisher
from tradingo import signals
from tradingo import utils


logger = logging.getLogger(__name__)


DEFAULT_AUM = 100_000


@lib_provider(model_signals="signals")
@symbol_provider(
    close="prices/close",
    ivol="prices/ivol",
    vol="signals/vol_128",
    symbol_prefix="{provider}.{universe}.",
)
@symbol_provider(instruments="instruments/{universe}", no_date=True)
@symbol_publisher(
    "portfolio/raw.percent",
    "portfolio/raw.shares",
    "signals/raw.signal",
    symbol_prefix="{name}.{provider}.{universe}.",
)
def portfolio_construction(
    name: str,
    model_signals: adb.library.Library,
    close: pd.DataFrame,
    ivol: pd.DataFrame,
    vol: pd.DataFrame,
    config: dict,
    instruments: pd.DataFrame,
    provider: str,
    universe: str,
    **kwargs,
):

    ivol = ivol.iloc[-1].sort_values(ascending=False)

    strategy = config["portfolio"][name]
    signal_weights = strategy["signal_weights"]
    multiplier = strategy["multiplier"]
    default_weight = strategy["default_weight"]
    weights = pd.Series(multiplier, index=instruments.index)
    constraints = strategy["constraints"]
    for key, weights_config in strategy["instrument_weights"].items():

        if key not in instruments.columns:
            continue

        weights = weights * instruments.apply(
            lambda i: weights_config.get(i[key], default_weight),
            axis=1,
        )
        logger.info("Weights: %s", weights)

    signal_value = (
        pd.concat(
            (
                model_signals.read(
                    f"{provider}.{universe}.{signal}"
                    # , columns=symbols
                ).data
                * weights
                * signal_weight
                for signal, signal_weight in signal_weights.items()
            ),
            keys=signal_weights,
            axis=1,
        )
        .transpose()
        .groupby(level=[1])
        .sum()
        .transpose()
        .ffill()
        .fillna(0)
        .div(10 * vol)
    )

    logger.info("signal_data: %s", signal_value)

    positions = signal_value

    if constraints["long_only"]:
        positions[(positions < 0.0)] = 0.0

    pct_position = positions.div(positions.transpose().sum(), axis=0)
    for col in pct_position.columns:
        pct_position.loc[
            pct_position.index == pct_position.first_valid_index(), col
        ] = 0.0

    share_position = (pct_position * config["aum"]) / close

    return (pct_position, share_position, signal_value)


@symbol_provider(close="prices/adj_close", symbol_prefix="{provider}.{universe}.")
@symbol_publisher("prices/ivol", symbol_prefix="{provider}.{universe}.")
def instrument_ivol(close, provider, **kwargs):
    pct_returns = np.log(close / close.shift())

    ivols = []

    for symbol in pct_returns.columns:
        universe = pct_returns[
            random.sample([c for c in pct_returns.columns if c != symbol], 100)
        ]

        def vol(uni):
            return (1 - (1 + uni).prod(axis=1).pow(1 / 100)).ewm(10).std()

        ivol = vol(pd.concat((universe, pct_returns[symbol]), axis=1)) - vol(universe)
        ivols.append(ivol.rename(symbol))

    return (pd.concat(ivols, axis=1).rename_axis("Symbol"),)
