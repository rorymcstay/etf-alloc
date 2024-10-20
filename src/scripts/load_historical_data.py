# Historical IG data may be downloaded from the following site.
# https://www.dukascopy.com/swiss/english/marketwatch/historical/
#
import argparse
import logging
from typing import List
import dateutil.tz
from collections import defaultdict
import re
import os
from pathlib import Path

from arcticdb import Arctic

import pandas as pd

from tradingo.signals import symbol_provider, symbol_publisher
from tradingo import sampling


logger = logging.getLogger(__name__)


ASSET_MAPPING = {
    "USA500.IDXUSD": "IX.D.SPTRD.IFS.IP",
    "GAS.CMDUSD": "CC.D.NG.UMP.IP",
    "BRENT.CMDUSD": "CC.D.LCO.UMP.IP",
    "USTBOND.TRUSD": "IR.D.10YEAR100.FWM2.IP",
}


def cli_app():

    app = argparse.ArgumentParser("load-historical-data")

    app.add_argument("--path", type=Path, required=True, nargs="+")
    app.add_argument("--arctic-uri", required=True)
    app.add_argument("--dry-run", action="store_true")
    app.add_argument("--universe", required=True)
    app.add_argument("--provider", required=True)

    return app


# groups: symbol, frequency, field, start_date, end_date
FILE_REGEX = r"^([A-Z0-9\.]+)_Candlestick_([0-9]+_[MDHS])_(BID|ASK)_([0-9]{2}\.[0-9]{2}\.[0-9]{4})-([0-9]{2}\.[0-9]{2}\.[0-9]{4}).csv$"


def main():

    args = cli_app().parse_args()

    a = Arctic(args.arctic_uri)

    result = read_backfill(
        paths=args.path,
        arctic=a,
        dry_run=args.dry_run,
        universe=args.universe,
        provider=args.provider,
    )

    service = sampling.get_ig_service()

    sampling.download_instruments(
        index_col=None,
        epics=ASSET_MAPPING.values(),
        universe=args.universe,
        arctic=a,
    )

    if args.dry_run:
        print(result)
    print("finished")


@symbol_publisher(
    template="prices/{0}.{1}",
    symbol_prefix="{provider}.{universe}.",
)
def read_backfill(
    paths: List[Path],
    **kwargs,
):

    data_files = defaultdict(list)

    for path in paths:

        for file in os.listdir(path):

            if match := re.match(FILE_REGEX, file):

                symbol, frequency, field, start_date, end_date = match.groups()

                # need to group by symbol

                data_files[(field.lower(), symbol)].append(path / file)

    # symbol, ohlc, symbol
    #
    #
    def read_file(f):
        logger.warning("Reading %s", f)
        out = pd.read_csv(
            f,
            index_col=0,
            date_format="%d.%m.%Y %H:%M:%S.%f",
        )
        out.index = pd.to_datetime(out.index)
        if out.index.name == "Local time":
            out.index = out.index.tz_convert("utc")
        elif out.index.name == "Gmt time":
            out.index = out.index.tz_localize("GMT").tz_convert("utc")
        else:
            raise ValueError(out.index.name)
        return out.rename_axis("DateTime")

    result = pd.concat(
        (
            pd.concat(read_file(f) for f in files).query("~index.duplicated()")
            for files in data_files.values()
        ),
        axis=1,
        keys=data_files.keys(),
    ).reorder_levels([0, 2, 1], axis=1)

    result.rename(columns=ASSET_MAPPING, inplace=True)

    return (
        (result["bid"]["Open"], ("bid", "open")),
        (result["bid"]["High"], ("bid", "high")),
        (result["bid"]["Low"], ("bid", "low")),
        (result["bid"]["Close"], ("bid", "close")),
        (result["ask"]["Open"], ("ask", "open")),
        (result["ask"]["High"], ("ask", "high")),
        (result["ask"]["Low"], ("ask", "low")),
        (result["ask"]["Close"], ("ask", "close")),
        (((result["ask"]["Open"] + result["bid"]["Open"]) / 2), ("mid", "open")),
        (((result["ask"]["High"] + result["bid"]["High"]) / 2), ("mid", "high")),
        (((result["ask"]["Low"] + result["bid"]["Low"]) / 2), ("mid", "low")),
        (((result["ask"]["Close"] + result["bid"]["Close"]) / 2), ("mid", "close")),
    )


if __name__ == "__main__":

    import sys

    logging.getLogger(__name__).setLevel(logging.INFO)

    sys.argv.extend(
        [
            "--arctic-uri",
            "s3://s3.us-east-1.amazonaws.com:tradingo-store?aws_auth=true&path_prefix=uat",
            "--path",
            str(Path.home() / "dev" / "market-data" / "GAS"),
            str(Path.home() / "dev" / "market-data" / "USA500"),
            str(Path.home() / "dev" / "market-data" / "BRENT"),
            # str(Path.home() / "dev" / "market-data" / "USTBOND"),
            # "--dry-run",
            "--provider",
            "ig-trading",
            "--universe",
            "im-assets",
        ]
    )

    main()
