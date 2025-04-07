"""Base interface for data providers."""

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

import pandas as pd
from arcticdb.version_store.library import Library

from tradingo.sampling.types import InstrumentType


ServiceType = TypeVar("T")


class DataInterface(ABC, Generic[ServiceType]):
    """
    Base interface for data providers.
    """

    instrument_type: InstrumentType

    def __init__(self, service: Optional[ServiceType] = None, library: Library = None):
        self._service = service or self._get_service()
        self._library: Library = library

    @property
    def service(self) -> ServiceType:
        """the provider service instance."""
        if not self._service:
            raise ValueError("Service not set.")
        return self._service

    @property
    def library(self) -> Library:
        """the library instance."""
        if not self._library:
            raise ValueError("Library not set.")
        return self._library

    @abstractmethod
    def _get_service(self): ...

    @abstractmethod
    def list_instruments(self, search: str = None) -> list[str]: ...

    @abstractmethod
    def fetch_instruments(self, symbols: list[str]) -> list[str]: ...

    @abstractmethod
    def sample(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame: ...

    @abstractmethod
    def create_universe(
        instruments: pd.DataFrame,
        pricelib: Library,
        end_date: pd.Timestamp,
        start_date: pd.Timestamp,
    ): ...
