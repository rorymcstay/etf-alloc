import logging
import copy
from typing import NamedTuple

import pandas as pd
import numpy as np

from tradingo.symbols import symbol_provider, symbol_publisher


logger = logging.getLogger(__name__)


class PnlSnapshot(NamedTuple):

    date: pd.Timestamp
    net_position = 0
    avg_open_price = 0
    net_investment = 0
    realised_pnl = 0
    unrealised_pnl = 0
    total_pnl = 0
    last_qty = 0
    last_trade_price = 0
    last_trade_date = None

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
        is_still_open = abs(self.net_position + trade_quantity) > 0
        # net investment
        m_net_investment = max(
            self.net_investment, abs(self.net_position * self.avg_open_price)
        )
        # realized pnl
        if not is_still_open and self.net_position:
            # Remember to keep the sign as the net position
            m_realised_pnl += (
                (trade_price - self.avg_open_price)
                * min(abs(trade_quantity), abs(self.net_position))
                * (abs(self.net_position) / self.net_position)
            )
        # total pnl
        m_total_pnl = m_realised_pnl + self.unrealised_pnl
        # avg open price
        if is_still_open:
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
        return self._replace(
            unrealised_pnl=(last_price - self.avg_open_price) * self.net_position,
            total_pnl=self.realised_pnl + self.unrealised_pnl,
            date=date,
        )


@symbol_provider(
    portfolio="PORTFOLIO/{name}.{stage}.SHARES",
    prices="ASSET_PRICES/ADJ_CLOSE.{provider}",
)
@symbol_publisher(
    "BACKTEST/SUMMARY",
    "BACKTEST/INSTRUMENT_RETURNS",
    # backtest="BACKTEST/BACKTEST.{}",
    symbol_prefix="{config_name}.{name}.",
)
def backtest(
    portfolio: pd.DataFrame,
    prices: pd.DataFrame,
    name: str,
    stage: str = "RAW",
    **kwargs,
):
    trades = portfolio.ffill().fillna(0.0).round().diff()
    prices = prices.ffill()

    def compute_backtest(inst_trades: pd.Series):

        ticker: str = inst_trades.name
        logger.warning("Computing backtest for ticker=%s", ticker)
        inst_prices = prices[ticker].ffill()

        current_pnl = PnlSnapshot(
            date=inst_trades.first_valid_index() - pd.offsets.BDay(1)
        )

        pnl_series = []

        pnl_series.append(current_pnl)

        data = pd.concat(
            (inst_trades.rename("trade"), inst_prices.rename("price")), axis=1
        )

        for date, (trade, last_price) in data.iterrows():
            current_pnl = current_pnl.on_market_data(last_price=last_price, date=date)
            if not np.isnan(trade) and trade:
                current_pnl = current_pnl.on_trade(
                    last_price, trade_quantity=trade, trade_date=date
                )

            pnl_series.append(current_pnl)

        return pd.DataFrame([i.to_dict() for i in pnl_series]).set_index(["date"])

    trades = pd.concat(
        (compute_backtest(data) for _, data in trades.items()), keys=trades, axis=1
    )

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
        trades,
    )
