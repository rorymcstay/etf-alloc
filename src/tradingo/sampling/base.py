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

    Args:
        service: The service instance to use for data fetching.
        library: The ArcticDB library instance.
    """

    instrument_type: InstrumentType

    def __init__(self, service: Optional[ServiceType] = None, library: Library = None):
        self._service = service
        self._library: Library = library

    @property
    def service(self) -> ServiceType:
        """the provider service instance."""
        if not self._service:
            self._service = self._get_service()
        return self._service

    @property
    def library(self) -> Library:
        """the library instance."""
        if not self._library:
            raise ValueError("Library not set.")
        return self._library

    @abstractmethod
    def _get_service(self) -> ServiceType:
        """instatiate the service instance."""

    @abstractmethod
    def list_instruments(self, search: str = None) -> list[str]:
        """List available instruments."""

    @abstractmethod
    def fetch_instruments(self, symbols: list[str]) -> list[str]:
        """Fetch instruments' static data by symbols."""

    @abstractmethod
    def sample(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data for a specific instrument."""

    @abstractmethod
    def create_universe(
        self,
        instruments: pd.DataFrame,
        pricelib: Library,
        end_date: pd.Timestamp,
        start_date: pd.Timestamp,
    ):
        """Create a universe of instruments."""
