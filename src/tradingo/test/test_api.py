import pytest

from tradingo.api import Tradingo


def test_tradingo_api(tradingo: Tradingo):

    df = tradingo.prices.close()
    return df


def test_tradingo_api_with(tradingo: Tradingo):

    t = tradingo

    return t.instruments.etfs.concat.prices.close.transpose(axis=1).transpose()


def test_tradingo_merge(tradingo: Tradingo):
    return tradingo.portfolio.model.raw.shares.concat.prices.close(
        axis=1, keys=("position", "close")
    )


if __name__ == "__main__":
    pytest.main()
