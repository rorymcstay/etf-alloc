import pytest

from tradingo.api import Tradingo

from tradingo.test.fixtures import prices, tradingo, position


def test_tradingo_api(tradingo: Tradingo):

    df = tradingo.prices.close()
    return df


if __name__ == "__main__":
    pytest.main()
