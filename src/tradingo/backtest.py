import logging
import copy
import dataclasses

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

    def to_dict(self):
        return {
            "date": self.date,
            "net_position": self.net_position,
            "avg_open_price": self.avg_open_price,
            "net_investment": self.net_investment,
            "realised_pnl": self.realised_pnl,
            "unrealised_pnl": self.unrealised_pnl,
            "total_pnl": self.total_pnl,
            "last_qty": self.last_qty,
            "last_trade_price": self.last_trade_price,
            "last_trade_date": self.last_trade_date,
        }

    def __init__(self, date: pd.Timestamp, opening_position=0, opening_price=0):
        self.date = date
        if opening_position != 0:
            self.on_trade(opening_price, opening_position, date)

    def on_trade(
        self,
        trade_price: float,
        trade_quantity: float,
        trade_date: pd.Timestamp,
    ):
        logger.info(
            "%s: trade_price=%s trade_quantity=%s trade_date=%s",
            self,
            trade_price,
            trade_quantity,
            trade_date,
        )

        self = copy.copy(self)

        self.date = trade_date
        is_still_open = abs(self.net_position + trade_quantity) >= 0
        # net investment
        self.net_investment = max(
            self.net_investment, abs(self.net_position * self.avg_open_price)
        )
        # realized pnl
        if not is_still_open and self.net_position:
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
        return self

    def on_market_data(self, last_price: float, date: pd.Timestamp):
        self = copy.copy(self)
        self.unrealised_pnl = (last_price - self.avg_open_price) * self.net_position
        self.total_pnl = self.realised_pnl + self.unrealised_pnl
        self.date = date
        return self


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

    def compute_backtest(trds: pd.Series):

        ticker: str = trds.name
        logger.warning("Computing backtest for ticker=%s", ticker)
        inst_prices = prices[ticker].ffill()

        current_pnl = PnlSnapshot(date=trds.first_valid_index() - pd.offsets.BDay(1))

        pnl_series = []

        pnl_series.append(current_pnl)

        for date, (trade, last_price) in pd.concat(
            (trades, inst_prices.ffill()), axis=1
        ).items():
            current_pnl = current_pnl.on_market_data(last_price=last_price, date=date)
            if trade:
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
