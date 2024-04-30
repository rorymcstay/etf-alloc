import arcticdb as adb


class _Read:

    def __init__(self, path_so_far, library, assets):
        self.path_so_far = path_so_far
        self.library = library
        self.assets = assets

    def __getattr__(self, symbol):
        return self.__class__(
            (*self.path_so_far, symbol), library=self.library, assets=self.assets
        )

    def __call__(self, *args, **kwargs):
        return self.library.read(
            ".".join(self.path_so_far), *args, **kwargs, columns=self.assets
        ).data


class Tradingo(adb.Arctic):

    def __init__(self, name, provider, assets, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.provider = provider
        self.assets = assets

    def __getattr__(self, library):
        if library in self.list_libraries():
            return _Read(
                library=self.get_library(library),
                path_so_far=(self.name,),
                assets=self.assets,
            )
        return super().__getattr__(library)
