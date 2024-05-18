import contextlib
import pandas as pd
from datetime import datetime
import re
import arcticdb as adb


class _Read:

    def __init__(
        self,
        path_so_far,
        library: adb.library.Library,
        assets,
        common_args,
        common_kwargs,
    ):
        self.path_so_far = path_so_far
        self.library = library
        self.assets = assets
        self.path = ".".join(self.path_so_far)
        self.common_args = common_args
        self.common_kwargs = common_kwargs

    def __dir__(self):
        return [*self.list(), *super().__dir__()]

    def __repr__(self):
        return f'Namespace("{self.path}")'

    def __getattr__(self, symbol):
        return self.__class__(
            (*self.path_so_far, symbol),
            library=self.library,
            assets=self.assets,
            common_args=self.common_args,
            common_kwargs=self.common_kwargs,
        )

    def __call__(self, *args, **kwargs):
        kwargs.setdefault(
            "columns",
            (
                self.assets
                if all(i in self.path_so_far for i in ("backtest", "portfolio"))
                else []
            ),
        )
        return self.library.read(
            self.path, *args, *self.common_args, **kwargs, **self.common_kwargs
        ).data

    def update(self, *args, **kwargs):
        self.library.update(self.path, *args, **kwargs)

    def list(self, *args, **kwargs):
        regex = re.escape(".".join((self.path_so_far)) + ".")
        kwargs["regex"] = regex + kwargs.setdefault("regex", "")
        return list(
            dict.fromkeys(
                [
                    i.replace(f'{".".join(self.path_so_far)}.', "").split(".")[0]
                    for i in self.library.list_symbols(*args, **kwargs)
                ]
            )
        )

    def head(self, *args, **kwargs):
        return self.library.head(self.path, *args, **kwargs).data

    def tail(self, *args, **kwargs):
        return self.library.tail(self.path, *args, **kwargs).data

    def exists(self):
        pass


class Tradingo(adb.Arctic):

    def __init__(self, name, provider, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.provider = provider
        self._context_args = ()
        self._context_kwargs = {}

    @contextlib.contextmanager
    def common_args(self, *args, **kwargs):
        try:
            self._context_args = args
            self._context_kwargs = kwargs
            yield self
        finally:
            self._context_args = ()
            self._context_kwargs = {}

    def __getattr__(self, library):

        if library in self.list_libraries():
            path_so_far = [self.name]
            if library == "prices":
                path_so_far.append(self.provider)
            if library == "instruments":
                return _Read(
                    library=self.get_library(library),
                    path_so_far=path_so_far,
                    assets=[],
                    common_args=(),
                    common_kwargs={},
                )
            return _Read(
                library=self.get_library(library),
                path_so_far=path_so_far,
                assets=self.instruments.etfs().index.to_list(),
                common_args=self._context_args,
                common_kwargs=self._context_kwargs,
            )

        return super().__getattr__(library)

    def __dir__(self):
        return [*self.list_libraries(), *super().__dir__()]


class VolSurface(Tradingo):

    def get(
        self,
        symbol: str,
        start_date: pd.Timestamp = pd.Timestamp("1970-01-01 00:00+00:00"),
        end_date: pd.Timestamp = pd.Timestamp.now("utc"),
        **kwargs,
    ):

        with self.common_args(date_range=(start_date, end_date)):

            futures = (
                pd.concat(
                    (self.futures.cboe.VX.expiration(), t.futures.cboe.VX.price()),
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

        return option_chains.merge(futures, on=["timestamp", "expiration"], how="left")
