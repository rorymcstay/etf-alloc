from __future__ import annotations

import functools
import inspect
import logging
from collections import defaultdict
from typing import Any, Callable, DefaultDict, NamedTuple, Optional
from urllib.parse import parse_qsl, urlparse

import arcticdb as adb
from arcticdb.arctic import Library
import pandas as pd
from arcticdb_ext.exceptions import InternalException

logger = logging.getLogger(__name__)


class SymbolParseError(Exception):
    """raised when cant parse symbol"""


def add_params(function: Callable[...], *args: str):
    origsig = inspect.signature(function)
    orig_params = list(origsig.parameters.values())
    for arg in args:
        if arg in origsig.parameters:
            continue
        else:
            orig_params.insert(
                0,
                inspect.Parameter(
                    arg,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ),
            )

    function.__signature__ = origsig.replace(parameters=orig_params)


class Symbol(NamedTuple):
    library: str
    symbol: str
    kwargs: dict

    @classmethod
    def parse(
        cls, base: str, kwargs: dict, symbol_prefix: str = "", symbol_postfix: str = ""
    ) -> Symbol:
        """
        Parse a symbol string and return a Symbol object.
        """

        string_kwargs = {k: str(v) for k, v in kwargs.items()}

        try:
            symbol = base.format(**string_kwargs)
            parsed_symbol = urlparse(symbol)
            try:
                lib, sym = parsed_symbol.path.split("/")
            except ValueError as ex:
                raise SymbolParseError(f"symbol {symbol} is invalid.") from ex
            kwargs = dict(parse_qsl(parsed_symbol.query))
            symbol_prefix = symbol_prefix.format(**string_kwargs)
            symbol_postfix = symbol_postfix.format(**string_kwargs)
        except KeyError as ex:
            raise SymbolParseError(
                f"Missing parameter: {ex.args[0]},"
                f" {symbol_prefix=}, {symbol_postfix=}, {base=}, {string_kwargs=}"
            )

        for key, value in kwargs.items():
            if key == "as_of":
                try:
                    kwargs[key] = int(value)
                except TypeError:
                    continue

        return cls(lib, symbol_prefix + sym + symbol_postfix, kwargs)


def lib_provider(**libs: str):
    def decorator(func: Callable[...]):
        @functools.wraps(func)
        def wrapper(*args: str, arctic: adb.Arctic, **kwargs: str | Library):
            libs_ = {
                k: (
                    arctic.get_library(
                        str(kwargs.get(k, v)),
                        create_if_missing=True,
                    )
                    if not isinstance(kwargs.get(k), Library)
                    else k
                )
                for k, v in libs.items()
            }
            kwargs.update(libs_)
            return envoke_symbology_function(
                func,
                args,
                kwargs,
                arctic,
            )

        setattr(wrapper, "lib_provided", None)

        add_params(wrapper, "arctic")

        return wrapper

    return decorator


def symbol_provider(
    symbol_prefix: str = "",
    no_date: bool = False,
    **symbols: str,
):
    def decorator(func: Callable[..., tuple[pd.DataFrame | pd.Series]]):
        @functools.wraps(func)
        def wrapper(
            *args: Any,
            arctic: adb.Arctic,
            start_date: Optional[pd.Timestamp] = None,
            end_date: Optional[pd.Timestamp] = None,
            **kwargs: Any,
        ):
            orig_symbol_data = {}
            requested_symbols = symbols.copy()

            for symbol in symbols:
                if symbol in kwargs and isinstance(
                    kwargs[symbol], (pd.DataFrame, pd.Series)
                ):
                    orig_symbol_data[symbol] = kwargs.pop(symbol)
                if symbol in kwargs and isinstance(kwargs[symbol], str):
                    requested_symbols[symbol] = kwargs[symbol]
                if symbol in kwargs and kwargs[symbol] is None:
                    requested_symbols.pop(symbol)

            def get_symbol_data(v: str, with_no_date: bool = False):
                symbol = Symbol.parse(v, kwargs, symbol_prefix=symbol_prefix)
                try:
                    return (
                        arctic.get_library(
                            symbol.library,
                            create_if_missing=True,
                        )
                        .read(
                            symbol.symbol,
                            date_range=(
                                None
                                if with_no_date
                                else (
                                    pd.Timestamp(start_date) if start_date else None,
                                    pd.Timestamp(end_date) if end_date else None,
                                )
                            ),
                            **symbol.kwargs,
                        )
                        .data
                    )
                except InternalException as ex:
                    if "The data for this symbol is pickled" in ex.args[0]:
                        return get_symbol_data(v, with_no_date=True)
                    raise ex

            symbols_data = {
                k: get_symbol_data(v, with_no_date=no_date)
                for k, v in requested_symbols.items()
                if k not in orig_symbol_data
            }
            kwargs.update(symbols_data)
            kwargs.update(orig_symbol_data)

            logger.info("Providing %s symbols from %s", symbols_data.keys(), arctic)

            return envoke_symbology_function(
                func,
                args,
                kwargs,
                arctic=arctic,
                start_date=start_date,
                end_date=end_date,
            )

        setattr(wrapper, "is_provided", None)
        add_params(wrapper, "arctic", "start_date", "end_date")

        return wrapper

    return decorator


def envoke_symbology_function(
    function: Callable[..., tuple[pd.DataFrame | pd.Series]],
    args: Any,
    kwargs: Any,
    arctic: adb.Arctic,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
):
    sig = inspect.signature(function)

    if "start_date" in sig.parameters:
        kwargs.setdefault("start_date", start_date)
    if "end_date" in sig.parameters:
        kwargs.setdefault("end_date", end_date)
    if "arctic" in sig.parameters:
        kwargs.setdefault("arctic", arctic)

    return function(*args, **kwargs)


def symbol_publisher(
    *symbols: str,
    symbol_prefix: str = "",
    symbol_postfix: str = "",
    astype: Optional[dict[str, Any]] = None,
    template: Optional[str] = None,
    library_options: Optional[adb.LibraryOptions] = None,
    write_pickle: bool = False,
):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(
            *args: Any,
            arctic: adb.Arctic,
            dry_run: bool = True,
            snapshot: Optional[str] = None,
            clean: bool = False,
            **kwargs: Any,
        ):
            out = envoke_symbology_function(func, args, kwargs, arctic)
            logger.info("Publishing %s to %s", symbols or template, arctic)
            if template:
                out, symbols_ = tuple(zip(*out))
                formatted_symbols = tuple(template.format(*s) for s in symbols_)
            else:
                formatted_symbols = symbols

            libraries: DefaultDict[str, dict] = defaultdict(dict)

            for data, symbol in zip(out, formatted_symbols):
                if data.empty:
                    continue

                if astype:
                    data = data.astype(
                        astype
                        if not isinstance(astype, dict)
                        else {k: v for k, v in astype.items() if k in data.columns}
                    )

                if not dry_run:
                    parsed_symbol = Symbol.parse(
                        symbol,
                        kwargs,
                        symbol_prefix=symbol_prefix,
                        symbol_postfix=symbol_postfix,
                    )
                    logger.info(
                        "writing symbol=%s rows=%s",
                        parsed_symbol,
                        len(data.index),
                    )

                    lib_name, sym, params = parsed_symbol
                    lib = arctic.get_library(
                        lib_name,
                        create_if_missing=True,
                        library_options=library_options,
                    )
                    if isinstance(data.index, pd.DatetimeIndex):
                        if clean:
                            lib.delete(sym)
                        result = lib.update(
                            sym,
                            data,
                            upsert=True,
                            date_range=(data.index[0], data.index[-1]),
                            **params,
                        )
                    elif write_pickle:
                        result = lib.write_pickle(sym, data, **params)

                    else:
                        result = lib.write(sym, data, **params)

                    libraries[lib.name][result.symbol] = result.version

            if not dry_run and snapshot:
                for lib_name, versions in libraries.items():
                    logging.info(
                        "Snapshotting %s for %s %s", lib_name, snapshot, versions
                    )
                    lib = arctic.get_library(lib_name)
                    if snapshot in lib.list_snapshots():
                        lib.delete_snapshot(snapshot)
                    lib.snapshot(snapshot_name=snapshot, versions=versions)

            if dry_run:
                return pd.concat(out, keys=formatted_symbols, axis=1)

            return None

        setattr(wrapper, "is_publisher", None)
        add_params(wrapper, "arctic", "dry_run", "snapshot", "clean")
        return wrapper

    return decorator
