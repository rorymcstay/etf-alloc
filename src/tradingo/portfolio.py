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
@symbol_provider(close="prices/{provider}.close", ivol="prices/{provider}.ivol")
@symbol_publisher(
    "portfolio/raw.percent",
    "portfolio/raw.shares",
    "signals/raw.signal",
    symbol_prefix="{config_name}.{name}.",
)
def portfolio_construction(
    name: str, model_signals: adb.library.Library, close: pd.DataFrame, ivol, **kwargs
):

    ivol = ivol.iloc[-1].sort_values(ascending=False)

    symbols = [*ivol.iloc[0:50].index, *ivol.iloc[-50:].index]

    config = utils.get_config()
    strategy = config["portfolio"][name]
    signal_weights = strategy["signal_weights"]
    instruments = utils.get_instruments(config)
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
                    f'{kwargs["config_name"]}.{name}.{signal}'
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
    )

    logger.info("signal_data: %s", signal_value)

    positions = signal_value

    if constraints["long_only"]:
        positions[(positions < 0.0)] = 0.0

    pct_position = positions.div(positions.transpose().sum(), axis=0)

    share_position = (pct_position * config["aum"]) / close

    return (pct_position, share_position, signal_value)


@symbol_provider(close="prices/{provider}.adj_close")
@symbol_publisher("prices/{provider}.ivol")
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

    return (pd.concat(ivols, axis=1),)
