import pytest
import pandas as pd
import arcticdb as adb

from tradingo import backtest
from tradingo.api import Tradingo
from tradingo.backtest import PnlSnapshot


@pytest.fixture
def arctic() -> adb.Arctic:
    return Tradingo("ETFT", "yfinance", "lmdb:///home/rory/dev/airflow/test/arctic.db")


@pytest.fixture
def prices(arctic: adb.Arctic):
    return arctic.prices.close()


@pytest.fixture
def portfolio(arctic: adb.Arctic):
    return arctic.portfolio.trend.raw.percent()


def test_pnl_snapshot():
    pnl = PnlSnapshot(pd.Timestamp("2024-04-26 00:00:00+00:00"))

    pnl_1 = pnl.on_trade(100, 1, pd.Timestamp("2024-04-29 00:00:00+00:00")).on_trade(
        100, 1, pd.Timestamp("2024-04-29 00:00:00+00:00")
    )

    assert pnl_1.net_investment == 100
    assert pnl_1.net_position == 2
    assert pnl_1.realised_pnl == 0
    assert pnl_1.unrealised_pnl == 0
    assert pnl_1.avg_open_price == 100

    pnl_2 = pnl_1.on_trade(101, -1, pd.Timestamp("2024-04-30 00:00:00+00:00"))
    assert pnl_2.net_investment == 200
    assert pnl_2.net_position == 1
    assert pnl_2.realised_pnl == 1
    assert pnl_2.unrealised_pnl == 0
    assert pnl_1.avg_open_price == 100
    assert pnl_1.avg_open_price == 100

    pnl_2 = pnl_2.on_trade(101, -1, pd.Timestamp("2024-05-01 00:00:00+00:00"))
    assert pnl_2.net_investment == 200
    assert pnl_2.net_position == 0
    assert pnl_2.realised_pnl == 2
    assert pnl_2.unrealised_pnl == 0
    pnl_2 = pnl_2.on_trade(101, 2, pd.Timestamp("2024-05-01 00:00:00+00:00"))
    assert pnl_2.net_investment == 200


def test_backtest(prices: pd.DataFrame, portfolio: pd.DataFrame):

    bt = backtest.backtest(
        prices=prices,
        portfolio=portfolio,
        start_date=pd.Timestamp("2018-01-01 00:00:00+00:00"),
        end_date=pd.Timestamp.now("utc"),
        name="trend",
        stage="RAW",
        config_name="ETFT",
        provider="yfinance",
        dry_run=True,
    )

    assert isinstance(bt[-1], pd.DataFrame)


if __name__ == "__main__":
    pytest.main()
