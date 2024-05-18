import pytest
import pandas as pd

from tradingo import backtest
from tradingo.api import Tradingo
from tradingo.backtest import PnlSnapshot


@pytest.fixture
def arctic() -> Tradingo:
    return Tradingo("ETFT", "yfinance", "lmdb:///home/rory/dev/airflow/test/arctic.db")


@pytest.fixture
def prices(arctic: Tradingo):
    return arctic.prices.close()


@pytest.fixture
def constant_returns_prices():

    daily_return = 0.01
    idx = pd.bdate_range("2020-01-01", periods=260, freq="B")
    return pd.DataFrame(1 + daily_return, index=idx, columns=["ABCD"]).cumprod()


@pytest.fixture
def constant_position():

    position = 1
    idx = pd.bdate_range("2020-01-01", periods=260, freq="B")
    return pd.DataFrame(position, index=idx, columns=["ABCD"])


@pytest.fixture
def portfolio(arctic: Tradingo):
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


def test_backtest_constant_return(constant_returns_prices, constant_position):

    portfolio, *fields = backtest.backtest(
        portfolio=constant_position,
        prices=constant_returns_prices,
        name="test",
        config_name="test",
        dry_run=True,
        start_date=constant_returns_prices.index[0],
        end_date=constant_returns_prices.index[-1],
        stage="raw",
    )

    pd.testing.assert_series_equal(
        portfolio["unrealised_pnl"].diff() / 2,
        (constant_position * constant_returns_prices.diff()).squeeze(),
        check_names=False,
        check_freq=False,
    )


if __name__ == "__main__":
    pytest.main()
