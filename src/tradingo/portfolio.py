import logging
import random

import pandas as pd
import numpy as np

import arcticdb as adb

from tradingo.symbols import lib_provider, symbol_provider, symbol_publisher
from tradingo import signals
from tradingo import utils


logger = logging.getLogger(__name__)


@lib_provider(model_signals="MODEL_SIGNALS")
@symbol_provider(close="ASSET_PRICES/CLOSE.{provider}")
@symbol_publisher(
    "PORTFOLIO/RAW.PERCENT",
    "PORTFOLIO/RAW.SHARES",
    symbol_prefix="{config_name}.{name}.",
)
def portfolio_construction(
    name: str, model_signals: adb.library.Library, close: pd.DataFrame, **kwargs
):

    config = utils.get_config()
    strategy = config["portfolio"][name]
    signal_weights = strategy["signal_weights"]
    asset_class_weights = strategy["instrument_weights"]["asset_class"]
    multiplier = strategy["instrument_weights"]["multiplier"]
    constraints = strategy["constraints"]
    instruments = utils.get_instruments(config)

    weights = multiplier * instruments.apply(
        lambda i: asset_class_weights.get(i["Asset Class"], 0), axis=1
    )
    logger.info("Weights: %s", weights)

    signal_data = pd.concat(
        (
            model_signals.read(f'{kwargs["config_name"]}.{name}.{signal}').data
            * weights
            * signal_weight
            for signal, signal_weight in signal_weights.items()
        ),
        keys=signal_weights,
        axis=1,
    )

    logger.info("signal_data: %s", signal_data)

    positions = (
        signal_data.transpose().groupby(level=[1]).prod().transpose().ffill().fillna(0)
    )

    if constraints["long_only"]:
        positions[(positions < 0.0)] = 0.0

    pct_position = positions.div(positions.transpose().sum(), axis=0)

    share_posiition = (pct_position * config["aum"]) / close

    return (pct_position, share_posiition)


@symbol_provider(close="ASSET_PRICES/ADJ_CLOSE.{provider}")
def instrument_ivol(close, provider, **kwargs):
    pct_returns = np.log(close / close.shift())

    ivols = []

    for symbol in pct_returns.columns:
        universe = pct_returns[
            random.sample([c for c in pct_returns.columns if c != symbol], 100)
        ]
        vol = lambda uni: (1 - ((1 + uni).prod(axis=1).pow(1 / 100))).ewm(10).std()
        ivol = vol(pd.concat((universe, pct_returns[symbol]), axis=1)) - vol(universe)
        ivols.append(ivol.rename(symbol))

    return pd.concat(ivols, axis=1)
