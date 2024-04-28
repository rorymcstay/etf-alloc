import pytest
import pandas as pd
import arcticdb as adb

from tradingo import backtest


@pytest.fixture
def arctic() -> adb.Arctic:
    return adb.Arctic("lmdb:///home/rory/dev/airflow/test/arctic.db")


@pytest.fixture
def prices(arctic: adb.Arctic):
    return arctic.get_library("ASSET_PRICES").read("ETFT.CLOSE.yfinane").data


@pytest.fixture
def portfolio(arctic: adb.Arctic):
    return arctic.get_library("PORTFOLIO").read("ETFT.trend.RAW.SHARES").data


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

    assert isinstance(bt, pd.DataFrame)


if __name__ == "__main__":
    pytest.main()
