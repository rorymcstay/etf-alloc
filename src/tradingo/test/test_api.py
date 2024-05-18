import pytest
import string
import numpy as np
import pandas as pd
from tradingo.api import Tradingo
from tradingo.utils import null_instruments


@pytest.fixture
def stock_data() -> pd.DataFrame:

    annual_std = 0.12

    returns = pd.DataFrame(
        np.random.normal(0, annual_std / np.sqrt(260), (100, 60)),
        index=pd.bdate_range(start="2024-01-03 00:00:00+00:00", periods=100),
        columns=[
            "".join(np.random.choice(list(string.ascii_uppercase), 4))
            for _ in range(60)
        ],
    )
    return (1 + returns).cumprod()


@pytest.fixture
def tradingo(stock_data: pd.DataFrame) -> Tradingo:
    t = Tradingo("test", "yfinance", "mem://tradingo")
    libraries = ["prices", "signals", "backtest", "portfolio", "instruments"]
    for library in libraries:
        t.create_library(library)
    t.instruments.etfs.update(null_instruments(stock_data.columns), upsert=True)
    t.prices.close.update(stock_data, upsert=True)
    return t


def test_tradingo_api(tradingo: Tradingo):

    df = tradingo.prices.close()
    return df


if __name__ == "__main__":
    pytest.main()
