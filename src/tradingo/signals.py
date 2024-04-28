import logging
import numpy as np

import pandas as pd

from tradingo.symbols import symbol_provider, symbol_publisher


logger = logging.getLogger(__name__)


@symbol_provider(close="ASSET_PRICES/ADJ_CLOSE.{provider}")
@symbol_publisher(
    "MODEL_SIGNALS/{signal_name}",
    "MODEL_SIGNALS/vol_{speed1}",
    "MODEL_SIGNALS/vol_{speed2}",
    symbol_prefix="{config_name}.{model_name}.",
)
def ewmac_signal(
    close: pd.DataFrame,
    speed1: int,
    speed2: int,
    provider: str,
    config_name: str,
    model_name="ewmac",
    signal_name="ewmac_{speed1}_{speed2}",
    **kwargs,
):

    logger.info(
        "Running %s model=%s signal=%s with %s",
        config_name,
        model_name,
        signal_name,
        provider,
    )

    returns = np.log(close / close.shift())

    return (
        (
            close.ewm(halflife=speed2, min_periods=speed2).mean()
            - close.ewm(halflife=speed1, min_periods=speed1).mean()
        ),
        returns.ewm(halflife=speed1).std(),
        returns.ewm(halflife=speed2).std(),
    )


@symbol_publisher(
    "MODEL_SIGNALS/{signal}.scaled", symbol_prefix="{config_name}.{model_name}."
)
@symbol_provider(
    signal="MODEL_SIGNALS/{signal}", symbol_prefix="{config_name}.{model_name}."
)
def scaled(signal, scale: float, **kwargs):
    return ((signal / signal.abs().max()) * scale,)


@symbol_publisher(
    "MODEL_SIGNALS/{signal}.capped", symbol_prefix="{config_name}.{model_name}."
)
@symbol_provider(
    signal="MODEL_SIGNALS/{signal}", symbol_prefix="{config_name}.{model_name}."
)
def capped(signal: pd.Series, cap: float, **kwargs):
    signal[signal.abs() >= cap] = np.sign(signal) * cap
    return (signal,)
