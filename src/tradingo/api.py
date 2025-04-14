from __future__ import annotations

import contextlib
import inspect
import re
from typing import Any, Callable, Generator, Optional

import arcticdb as adb
import pandas as pd

import tradingo.utils

READ_SIG = inspect.signature(adb.arctic.Library.read)


class _Read:
    def __init__(
        self,
        path_so_far: list[str],
        library: adb.library.Library,
        assets: list[str],
        common_args: Any,
        common_kwargs: Any,
        root: Tradingo,
    ):
        self._path_so_far = path_so_far
        self._library = library
        self._assets = assets
        self._path = ".".join(self._path_so_far)
        self._common_args = common_args
        self._common_kwargs = common_kwargs
        self._root = root

    def __dir__(self) -> list[str]:
        return [*self.list(), *super().__dir__()]

    def __repr__(self) -> str:
        return f'Namespace("{self._path}")'

    def __getattr__(self, symbol) -> _Read:
        return self.__class__(
            [*self._path_so_far, symbol],
            library=self._library,
            assets=self._assets,
            common_args=self._common_args,
            common_kwargs=self._common_kwargs,
            root=self._root,
        )

    def __getitem__(self, symbol) -> _Read:
        return self.__getattr__(symbol)

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        operations = ["merge", "transpose", "concat", "with_instrument_details"]
        operation, index = next(
            (
                (elem, i)
                for i, elem in enumerate(self._path_so_far)
                if elem in operations
            ),
            (None, len(self._path_so_far)),
        )
        part_one_path = self._path_so_far[0:index]
        part_two_path = self._path_so_far[index + 1 :]

        lib_kwargs = {
            k: v for k, v in kwargs.items() if k in READ_SIG.parameters.keys()
        }

        callback_kwargs = {}

        def get_callback(operation: str) -> Callable[...]:
            for i in (pd.DataFrame, tradingo.utils, pd):
                try:
                    return getattr(i, operation)
                except AttributeError:
                    continue
            raise AttributeError(operation)

        if operation:
            callback = get_callback(operation)
            callback_sig = inspect.signature(callback)
            callback_kwargs = {
                k: v for k, v in kwargs.items() if k in callback_sig.parameters.keys()
            }

        def get_data_at_path(path: list[str], kw: Any) -> pd.DataFrame:
            kw.setdefault(
                "columns",
                (
                    self._assets
                    if all(i in path for i in ("backtest", "portfolio"))
                    else None
                ),
            )

            for k in self._common_kwargs:
                kw.pop(k, None)
            return pd.DataFrame(
                self._library.read(
                    ".".join(path),
                    *args,
                    *self._common_args,
                    **kw,
                    **self._common_kwargs,
                ).data
            )

        lhs = get_data_at_path(part_one_path, lib_kwargs)

        if operation:
            callback_args: (
                tuple[pd.DataFrame]
                | tuple[pd.DataFrame, pd.DataFrame]
                | tuple[tuple[pd.DataFrame, pd.DataFrame]]
            ) = (lhs,)
            if part_two_path:
                callback_lib = part_two_path[0]
                callback_args = (
                    lhs,
                    self.__class__(
                        (
                            *self._root._get_path_so_far(callback_lib),
                            *part_two_path[1:],
                        ),
                        getattr(self._root, (callback_lib))._library,
                        self._assets,
                        self._common_args,
                        self._common_kwargs,
                        self._root,
                    )(*args, **kwargs),
                )

            if operation == "concat":
                callback_args = (callback_args,)  # type: ignore

            callback = get_callback(operation)
            callback_sig = inspect.signature(pd.DataFrame)
            return callback(*callback_args, **callback_kwargs)
        return lhs

    def update(self, *args: Any, **kwargs: Any):
        self._library.update(self._path, *args, **kwargs)

    def list(self, *args: Any, **kwargs: Any) -> list[str]:
        if self._path_so_far:
            regex = re.escape(".".join((self._path_so_far)) + ".")
            kwargs["regex"] = regex + kwargs.setdefault("regex", "")
        return list(
            dict.fromkeys(
                [
                    i.replace(
                        f"{'.'.join(self._path_so_far)}." if self._path_so_far else "",
                        "",
                    ).split(".")[0]
                    for i in self._library.list_symbols(*args, **kwargs)
                ]
            )
        )

    def head(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame(self._library.head(self._path, *args, **kwargs).data)

    def tail(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame(self._library.tail(self._path, *args, **kwargs).data)

    def exists(self) -> bool:
        """Return true if symbol exists"""
        return bool(self._library.list_symbols(regex=f"^{re.escape(self._path)}$"))


class Tradingo(adb.Arctic):
    def __init__(
        self,
        *args: str,
        provider: Optional[str] = None,
        universe: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._provider = provider
        self._universe = universe
        self._context_args = ()
        self._context_kwargs: dict[str, Any] = {}

    @contextlib.contextmanager
    def common_args(self, *args, **kwargs) -> Generator[Tradingo, None, None]:
        try:
            self._context_args = args
            self._context_kwargs = kwargs
            yield self
        finally:
            self._context_args = ()
            self._context_kwargs = {}

    def _get_path_so_far(self, library) -> list[str]:
        path_so_far = []
        if library == "instruments":
            return path_so_far
        if self._provider:
            path_so_far.append(self._provider)
        if self._universe:
            path_so_far.append(self._universe)
        return path_so_far

    def __getattr__(self, library) -> _Read:
        if library in self.list_libraries():
            path_so_far = self._get_path_so_far(library)
            if library == "instruments":
                return _Read(
                    library=self.get_library(library),
                    path_so_far=path_so_far,
                    assets=[],
                    common_args=(),
                    common_kwargs={},
                    root=self,
                )

            assets = []
            if self._universe:
                assets = getattr(self.instruments, self._universe)().index.to_list()

            return _Read(
                library=self.get_library(library),
                path_so_far=path_so_far,
                assets=assets,
                common_args=self._context_args,
                common_kwargs=self._context_kwargs,
                root=self,
            )

        return super().__getattr__(library)

    def __dir__(self) -> list[str]:
        return [*self.list_libraries(), *super().__dir__()]


class VolSurface(Tradingo):
    def get(
        self,
        symbol: str,
        start_date: pd.Timestamp = pd.Timestamp("1970-01-01 00:00+00:00"),
        end_date: pd.Timestamp = pd.Timestamp.now("utc"),
        **kwargs: Any,
    ) -> pd.DataFrame:
        with self.common_args(date_range=(start_date, end_date)):
            futures = (
                pd.concat(
                    (self.futures.cboe.VX.expiration(), self.futures.cboe.VX.price()),
                    axis=1,
                    keys=("expiration", "price"),
                )
                .stack()
                .astype({"expiration": "datetime64[ns]"})
                .reset_index()
                .set_index(["timestamp", "symbol"])
            )

            library = getattr(self.options.cboe, symbol)
            option_chains = pd.concat(
                (
                    library.expiration(),
                    library.option_type(),
                    library.strike(),
                    library.bid(),
                    library.ask(),
                    library.implied_volatility(),
                    library.delta(),
                    library.vega(),
                    library.gamma(),
                    library.theta(),
                    library.rho(),
                ),
                axis=1,
                keys=(
                    "expiration",
                    "option_type",
                    "strike",
                    "bid",
                    "ask",
                    "implied_volatility",
                    "delta",
                    "vega",
                    "gamma",
                    "theta",
                    "rho",
                ),
            ).stack(future_stack=True)

        return option_chains.merge(
            futures, on=["timestamp", "expiration"], how="left"
        ).set_index(["expiration", "option_type", "strike"], append=True)
