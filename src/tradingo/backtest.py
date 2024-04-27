import logging
import copy
import enum
import dataclasses
from typing import Union

import pandas as pd

from tradingo.symbols import symbol_provider, symbol_publisher


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PnlSnapshot:

    date: pd.Timestamp | pd.NaT.__class__
    net_position = 0
    avg_open_price = 0
    net_investment = 0
    realised_pnl = 0
    unrealised_pnl = 0
    total_pnl = 0
    last_qty = 0
    last_trade_price = 0
    last_trade_date = None

    def __init__(self, opening_position, opening_price, date: pd.Timestamp):
        self.on_trade(opening_price, opening_position, date)

    def on_trade(
        self,
        trade_price: float,
        trade_quantity: float,
        trade_date: pd.Timestamp,
    ):

        self.date = trade_date
        # buy: positive position, sell: negative position
        is_still_open = (self.net_position * trade_quantity) >= 0
        # net investment
        self.net_investment = max(
            self.net_investment, abs(self.net_position * self.avg_open_price)
        )
        # realized pnl
        if not is_still_open:
            # Remember to keep the sign as the net position
            self.realised_pnl += (
                (trade_price - self.avg_open_price)
                * min(abs(trade_quantity), abs(self.net_position))
                * (abs(self.net_position) / self.net_position)
            )
        # total pnl
        self.total_pnl = self.realised_pnl + self.unrealised_pnl
        # avg open price
        if is_still_open:
            self.avg_open_price = (
                (self.avg_open_price * self.net_position)
                + (trade_price * trade_quantity)
            ) / (self.net_position + trade_quantity)
        else:
            # Check if it is close-and-open
            if trade_quantity > abs(self.net_position):
                self.avg_open_price = trade_price
        # net position
        self.net_position += trade_quantity
        self.last_qty = trade_quantity
        self.last_trade_price = trade_price
        self.last_trade_date = trade_date

    def on_market_data(self, last_price: float, date: pd.Timestamp):
        self.unrealised_pnl = (last_price - self.avg_open_price) * self.net_position
        self.total_pnl = self.realised_pnl + self.unrealised_pnl
        self.date = date


@symbol_provider(
    portfolio="PORTFOLIO/{name}.{stage}.PERCENT",
    prices="ASSET_PRICES/ADJ_CLOSE.{provider}",
)
@symbol_publisher(
    "BACKTEST/SUMMARY",
    "BACKTEST/INSTRUMENT_RETURNS",
    symbol_prefix="{config_name}.{name}.",
)
def backtest(
    portfolio: pd.DataFrame,
    prices: pd.DataFrame,
    name: str,
    stage: str = "RAW",
    **kwargs,
):

    logger.info("Running %s %s backtest", stage, name)

    returns = prices.pct_change() * portfolio

    sharpe = (
        returns.sum(axis=1).rolling(252).mean() / returns.sum(axis=1).rolling(252).std()
    )

    return (
        pd.concat(
            (
                returns.sum(axis=1),
                returns.sum(axis=1).cumsum(),
                sharpe,
            ),
            axis=1,
            keys=("RETURNS", "ACCOUNT", "SHARPE"),
        ),
        returns,
    )
