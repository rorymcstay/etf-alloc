import dateutil.tz
import logging
from typing import Optional

import pandas as pd
import numpy as np
from tenacity import Retrying, wait_exponential, retry_if_exception_type
from trading_ig.rest import IGService, ApiExceededException, IGException

from tradingo.config import IGTradingConfig
from tradingo.sampling.base import DataInterface
from tradingo.sampling.types import InstrumentType


logger = logging.getLogger(__name__)


class IGDataInterface(DataInterface[IGService]):
    """
    Data interface for IG data provider.
    """

    instrument_type: InstrumentType = InstrumentType.CFD

    @staticmethod
    def _get_service(
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        acc_type: Optional[str] = None,
    ) -> IGService:
        """Create an IGService instance with the provided credentials."""

        config = IGTradingConfig.from_env()
        retryer = Retrying(
            wait=wait_exponential(),
            retry=retry_if_exception_type(ApiExceededException),
        )
        service = IGService(
            username=username or config.username,
            password=password or config.password,
            api_key=api_key or config.api_key,
            acc_type=acc_type or config.acc_type,
            use_rate_limiter=True,
            retryer=retryer,
        )
        service.create_session()
        return service

    def list_instruments(self, search: str = None) -> pd.DataFrame:
        """
        List instruments based on a search string.
        """
        data = pd.DataFrame(self.service.search_markets(search))
        if data.empty and "epic" not in data.columns:
            return data
        return data.set_index("epic").rename_axis("Symbol", axis=0)

    def fetch_instruments(self, symbols: list[str]) -> pd.DataFrame:
        """
        Fetch instruments based on a list of symbols.
        """
        res = []
        exceptions = []
        for sym in symbols:
            try:
                res.append(self.service.fetch_market_by_epic(sym)["instrument"])
            except Exception as ex:  # IG raises generic exception
                exceptions.append(ex)
        if exceptions:
            raise IGException(exceptions)
        return pd.DataFrame(res).set_index("epic").rename_axis("Symbol", axis=0)

    def sample(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
        wait: int = 0,
    ) -> tuple[pd.DataFrame]:
        """
        Sample historical data for a given symbol between start and end dates.
        """

        try:
            result = (
                self.service.fetch_historical_prices_by_epic(
                    symbol,
                    end_date=pd.Timestamp(end_date)
                    .tz_convert(dateutil.tz.tzlocal())
                    .tz_localize(None)
                    .isoformat(),
                    start_date=(pd.Timestamp(start_date) + pd.Timedelta(seconds=1))
                    .tz_convert(dateutil.tz.tzlocal())
                    .tz_localize(None)
                    .isoformat(),
                    resolution=interval,
                    wait=wait,
                )["prices"]
                .tz_localize(dateutil.tz.tzlocal())
                .tz_convert("utc")
            )

        except Exception as ex:
            if ex.args and (
                ex.args[0] == "Historical price data not found"
                or ex.args[0] == "error.price-history.io-error"
            ):
                logger.warning("Historical price data not found %s", symbol)
                result = pd.DataFrame(
                    np.nan,
                    columns=pd.MultiIndex.from_tuples(
                        (
                            ("bid", "Open"),
                            ("bid", "High"),
                            ("bid", "Low"),
                            ("bid", "Close"),
                            ("ask", "Open"),
                            ("ask", "High"),
                            ("ask", "Low"),
                            ("ask", "Close"),
                        ),
                    ),
                    index=pd.DatetimeIndex([], name="DateTime", tz="utc"),
                )
                # raise SkipException after return
            else:
                raise ex

        return (
            result["bid"],
            result["ask"],
        )

    def create_universe(
        self,
        instruments: pd.DataFrame,
        end_date: pd.Timestamp,
        start_date: pd.Timestamp,
    ) -> tuple[pd.DataFrame]:
        """
        Create a universe of instruments based on the provided data.
        """

        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        def get_data(symbol: str):
            return pd.concat(
                (
                    self.library.read(
                        f"{symbol}.bid", date_range=(start_date, end_date)
                    ).data,
                    self.library.read(
                        f"{symbol}.ask", date_range=(start_date, end_date)
                    ).data,
                ),
                axis=1,
                keys=("bid", "ask"),
            )

        result = pd.concat(
            ((get_data(symbol) for symbol in instruments.index.to_list())),
            axis=1,
            keys=instruments.index.to_list(),
        ).reorder_levels([1, 2, 0], axis=1)

        return (
            result["bid"]["Open"],
            result["bid"]["High"],
            result["bid"]["Low"],
            result["bid"]["Close"],
            result["ask"]["Open"],
            result["ask"]["High"],
            result["ask"]["Low"],
            result["ask"]["Close"],
            ((result["ask"]["Open"] + result["bid"]["Open"]) / 2),
            ((result["ask"]["High"] + result["bid"]["High"]) / 2),
            ((result["ask"]["Low"] + result["bid"]["Low"]) / 2),
            ((result["ask"]["Close"] + result["bid"]["Close"]) / 2),
        )
