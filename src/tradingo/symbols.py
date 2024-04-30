import functools
import logging
from typing import NamedTuple
from urllib.parse import urlparse, parse_qsl

import pandas as pd

import arcticdb as adb


logger = logging.getLogger(__name__)

ENVIRONMENT = "test"
ARCTIC_URL = f"lmdb:///home/rory/dev/airflow/{ENVIRONMENT}/arctic.db"


class Symbol(NamedTuple):

    library: str
    symbol: str
    kwargs: dict


def lib_provider(**libs):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            arctic = adb.Arctic(ARCTIC_URL)
            libs_ = {
                k: arctic.get_library(v, create_if_missing=True)
                for k, v in libs.items()
            }
            return func(*args, **libs_, **kwargs)

        return wrapper

    return decorator


def parse_symbol(symbol, kwargs, symbol_prefix="", symbol_postfix=""):
    string_kwargs = {k: str(v) for k, v in kwargs.items()}
    symbol = symbol.format(**string_kwargs)
    parsed_symbol = urlparse(symbol)
    lib, sym = parsed_symbol.path.split("/")
    kwargs = dict(parse_qsl(parsed_symbol.query))
    symbol_prefix = symbol_prefix.format(**string_kwargs)
    symbol_postfix = symbol_postfix.format(**string_kwargs)
    return Symbol(lib, symbol_prefix + sym + symbol_postfix, kwargs)


def symbol_provider(
    symbol_prefix="{config_name}.",
    **symbols,
):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, start_date, end_date, **kwargs):

            arctic = adb.Arctic(ARCTIC_URL)
            orig_symbol_data = {}

            for symbol in symbols:
                if symbol in kwargs and isinstance(
                    kwargs[symbol], (pd.DataFrame, pd.Series)
                ):
                    orig_symbol_data[symbol] = kwargs.pop(symbol)

            logger.info("Providing %s symbols from %s", symbols, arctic)

            symbols_data = {
                k: (
                    arctic.get_library(
                        parse_symbol(v, kwargs, symbol_prefix=symbol_prefix).library,
                        create_if_missing=True,
                    )
                    .read(
                        parse_symbol(v, kwargs, symbol_prefix=symbol_prefix).symbol,
                        date_range=(pd.Timestamp(start_date), pd.Timestamp(end_date)),
                        **parse_symbol(v, kwargs, symbol_prefix=symbol_prefix).kwargs,
                    )
                    .data
                )
                for k, v in symbols.items()
                if k not in orig_symbol_data
            }
            kwargs.update(symbols_data)
            kwargs.update(orig_symbol_data)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def symbol_publisher(
    *symbols, symbol_prefix="{config_name}.", symbol_postfix="", astype=None
):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, dry_run=False, **kwargs):

            arctic = adb.Arctic(ARCTIC_URL)
            out = func(*args, arctic=arctic, **kwargs)
            logger.info("Publishing %s to %s", symbols, arctic)

            # if kws:
            #    kwdout = out.pop(-1)

            for data, symbol in zip(out, symbols):

                if astype:
                    data = data.astype(
                        astype
                        if not isinstance(astype, dict)
                        else {k: v for k, v in astype.items() if k in data.columns}
                    )

                logger.info(
                    "writing symbol=%s rows=%s",
                    parse_symbol(symbol, kwargs, symbol_prefix, symbol_postfix),
                    len(data.index),
                )

                lib, sym, params = parse_symbol(
                    symbol,
                    kwargs,
                    symbol_prefix=symbol_prefix,
                    symbol_postfix=symbol_postfix,
                )

                if not dry_run:
                    arctic.get_library(lib, create_if_missing=True).update(
                        sym, data, upsert=True, **params
                    )
                else:
                    return out

            return None

        return wrapper

    return decorator
