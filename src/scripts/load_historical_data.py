# Historical IG data may be downloaded from the following site.
# https://www.dukascopy.com/swiss/english/marketwatch/historical/
#
import argparse
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import pandas as pd
from arcticdb import Arctic

import tradingo.sampling as sampling

logger = logging.getLogger(__name__)


ASSET_MAPPING = {
    "USA500.IDXUSD": "IX.D.SPTRD.IFS.IP",
    "GAS.CMDUSD": "CC.D.NG.UMP.IP",
    "BRENT.CMDUSD": "CC.D.LCO.UMP.IP",
    "USTBOND.TRUSD": "IR.D.10YEAR100.FWM2.IP",
    "COCOA.CMDUSD": "CC.D.CC.UMP.IP",
}

MULTIPLIERS = {
    "USA500.IDXUSD": 1,
    "GAS.CMDUSD": 1000,
    "BRENT.CMDUSD": 100,
    "USTBOND.TRUSD": 100,
    "COCOA.CMDUSD": 1,
}


def cli_app() -> argparse.ArgumentParser:
    app = argparse.ArgumentParser("load-historical-data")

    app.add_argument("--path", type=Path, required=True, nargs="+")
    app.add_argument("--arctic-uri", required=True)
    app.add_argument("--dry-run", action="store_true")
    app.add_argument("--universe", required=True)
    app.add_argument("--provider", required=True)
    app.add_argument("--clean", action="store_true")
    app.add_argument("--end-date", type=pd.Timestamp, default=None)

    return app


# groups: symbol, frequency, field, start_date, end_date
FILE_REGEX = r"^([A-Z0-9\.]+)_Candlestick_([0-9]+_[MDHS])_(BID|ASK)_([0-9]{2}\.[0-9]{2}\.[0-9]{4})-([0-9]{2}\.[0-9]{2}\.[0-9]{4}).csv$"


def main() -> None:
    args = cli_app().parse_args()

    a = Arctic(args.arctic_uri)

    result = read_backfill(
        paths=args.path,
        arctic=a,
        dry_run=args.dry_run,
        universe=args.universe,
        provider=args.provider,
        clean=args.clean,
    )

    sampling.instruments.download_instruments(
        index_col=None,
        epics=ASSET_MAPPING.values(),
        universe=args.universe,
        arctic=a,
    )

    if args.dry_run:
        print(result)
    print("finished")


def read_backfill(
    paths: List[Path],
    end_date: Optional[pd.Timestamp] = None,
) -> tuple[pd.DataFrame]:
    data_files = defaultdict(list)

    for path in paths:
        for file in os.listdir(path):
            if match := re.match(FILE_REGEX, file):
                symbol, _, field, _, end_date = match.groups()

                data_files[(field.lower(), symbol)].append(path / file)

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

    result = result.mul(pd.Series(MULTIPLIERS), level=2, axis=1)

    result.rename(columns=ASSET_MAPPING, inplace=True)

    if end_date:
        result = result[result.index <= end_date]

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
