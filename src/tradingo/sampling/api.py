from tradingo.sampling.base import DataInterface
from tradingo.sampling.providers.ig import IGDataInterface


PROVIDERS: dict[str:DataInterface] = {
    "ig": IGDataInterface(),
    # "openbb": OpenBBProvider(),
    # "trading212": Trading212Provider(),
}


def list_instruments(provider: str, search: str):
    return PROVIDERS[provider].list_instruments(search)


def fetch_instruments(provider: str, symbols: list[str]):
    return PROVIDERS[provider].fetch_instruments(symbols)


def sample(provider: str, symbol: str, start: str, end: str, **kwargs):
    return PROVIDERS[provider].sample(symbol, start, end, **kwargs)
