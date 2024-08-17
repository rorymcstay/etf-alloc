import logging
from typing import Optional
from typing import NamedTuple
import arcticdb as adb

import pandas as pd
import numpy as np

from tradingo.symbols import lib_provider, symbol_provider, symbol_publisher
from tradingo import _backtest


logger = logging.getLogger(__name__)


class PnlSnapshot(NamedTuple):

    date: pd.Timestamp
    net_position: float = 0
    avg_open_price: float = 0
    net_investment: float = 0
    realised_pnl: float = 0
    unrealised_pnl: float = 0
    total_pnl: float = 0
    last_qty: float = 0
    last_trade_price: float = 0
    last_trade_date: pd.Timestamp = None
    net_exposure: float = 0
    last_price: float = 0

    def on_trade(
        self,
        trade_price: float,
        trade_quantity: float,
        trade_date: pd.Timestamp,
    ):
        m_date = trade_date
        m_net_position = self.net_position
        m_realised_pnl = self.realised_pnl
        m_avg_open_price = self.avg_open_price
        # net investment
        m_net_investment = max(
            self.net_investment, abs(self.net_position * self.avg_open_price)
        )
        # realized pnl
        if abs(self.net_position + trade_quantity) < abs(self.net_position):
            m_realised_pnl += (
                (trade_price - self.avg_open_price)
                * abs(trade_quantity)
                * np.sign(self.net_position)
            )
        # total pnl
        m_total_pnl = m_realised_pnl + self.unrealised_pnl
        # avg open price
        if abs(self.net_position + trade_quantity) > abs(self.net_position):
            m_avg_open_price = (
                (m_avg_open_price * m_net_position) + (trade_price * trade_quantity)
            ) / (self.net_position + trade_quantity)
        else:
            # Check if it is close-and-open
            if trade_quantity > abs(self.net_position):
                m_avg_open_price = trade_price

        m_net_position += trade_quantity
        m_last_qty = trade_quantity
        m_last_trade_price = trade_price
        m_last_trade_date = trade_date
        return self._replace(
            date=m_date,
            net_position=m_net_position,
            realised_pnl=m_realised_pnl,
            net_investment=m_net_investment,
            avg_open_price=m_avg_open_price,
            last_qty=m_last_qty,
            last_trade_price=m_last_trade_price,
            last_trade_date=m_last_trade_date,
            total_pnl=m_total_pnl,
        )

    def on_market_data(self, last_price: float, date: pd.Timestamp):
        unrealised_pnl = (last_price - self.avg_open_price) * self.net_position
        return self._replace(
            unrealised_pnl=unrealised_pnl,
            total_pnl=self.realised_pnl + unrealised_pnl,
            date=date,
            net_exposure=self.net_position * last_price,
            last_price=last_price,
        )


BACKTEST_FIELDS = (
    "unrealised_pnl",
    "realised_pnl",
    "total_pnl",
    "net_investment",
    "net_position",
    "avg_open_price",
)


@symbol_provider(
    portfolio="portfolio/{name}.{stage}.shares",
    prices="prices/{provider}.close",
)
@symbol_publisher(
    "backtest/portfolio",
    *(f"backtest/instrument.{f}" for f in BACKTEST_FIELDS if f != "date"),
    symbol_prefix="{config_name}.{name}.{stage}.",
)
def backtest(
    *,
    portfolio: pd.DataFrame,
    prices: pd.DataFrame,
    name: str,
    stage: str = "raw",
    **kwargs,
):
    trades = portfolio.ffill().fillna(0.0).diff()
    prices = prices.ffill()

    logger.info("running backtest for %s on stage %s with %s", name, stage, kwargs)

    def compute_backtest(inst_trades: pd.Series):

        ticker: str = inst_trades.name
        logger.warning("Computing backtest for ticker=%s", ticker)
        inst_prices = prices[ticker].ffill()
        opening_position = portfolio.loc[portfolio.first_valid_index(), ticker]
        if opening_position != 0:
            opening_avg_price = prices.loc[portfolio.first_valid_index(), ticker]
        else:
            opening_avg_price = 0

        return pd.DataFrame(
            data=_backtest.compute_backtest(
                opening_position,
                opening_avg_price,
                inst_trades.fillna(0.0).to_numpy(),
                inst_prices.to_numpy(),
            ),
            index=inst_trades.index,
            columns=BACKTEST_FIELDS,
        )

    backtest = pd.concat(
        (compute_backtest(data) for _, data in trades.items()), keys=trades, axis=1
    ).reorder_levels([1, 0], axis=1)

    backtest_fields = (backtest.loc[:, f] for f in BACKTEST_FIELDS if f != "date")

    net_exposure = (backtest["net_position"] * prices.ffill()).sum(axis=1)

    summary = (
        backtest[
            [
                "net_investment",
                "unrealised_pnl",
                "realised_pnl",
                "total_pnl",
            ]
        ]
        .groupby(level=0, axis=1)
        .sum()
    )
    summary["net_exposure"] = net_exposure

    return (summary, *backtest_fields)


@lib_provider(backtests="backtest")
def get_instrument_backtest(*symbol, backtests: Optional[adb.library.Library] = None):
    return pd.concat(
        (
            i.data
            for i in backtests.read_batch(
                [
                    adb.ReadRequest(columns=symbol, symbol=f"ETFT.trend.INSTRUMENT.{f}")
                    for f in BACKTEST_FIELDS
                ],
            )
        ),
        keys=BACKTEST_FIELDS,
        axis=1,
    )
