import re
import pandas as pd
from tradingo.utils import get_instruments
import arcticdb as adb


class _Read:

    def __init__(self, path_so_far, library: adb.library.Library, assets):
        self.path_so_far = path_so_far
        self.library = library
        self.assets = assets
        self.path = ".".join(self.path_so_far)

    def __dir__(self):
        return self.list()

    def __repr__(self):
        return f'Namespace("{self.path}")'

    def __getattr__(self, symbol):
        return self.__class__(
            (*self.path_so_far, symbol), library=self.library, assets=self.assets
        )

    def __call__(self, *args, **kwargs):
        assets = (
            self.assets
            if all(i in self.path_so_far for i in ("backtest", "portfolio"))
            else []
        )
        return self.library.read(self.path, *args, **kwargs, columns=assets).data

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

    def exists(self):
        pass


class Tradingo(adb.Arctic):

    def __init__(self, name, provider, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.provider = provider

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
                )
            return _Read(
                library=self.get_library(library),
                path_so_far=path_so_far,
                assets=self.instruments.etfs().index.to_list(),
            )

        return super().__getattr__(library)

    def __dir__(self):
        return [*self.list_libraries(), *super().__dir__()]
