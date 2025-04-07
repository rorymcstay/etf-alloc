from .providers.ig import IGProvider
from .providers.openbb import OpenBBProvider
from .providers.trading212 import Trading212Provider

PROVIDERS = {
    "ig": IGProvider(),
    "openbb": OpenBBProvider(),
    "trading212": Trading212Provider(),
}

def list_instruments(provider: str, instrument_type: str):
    return PROVIDERS[provider].list_instruments(instrument_type)

def fetch_data(provider: str, symbol: str, start: str, end: str):
    return PROVIDERS[provider].fetch_data(symbol, start, end)
