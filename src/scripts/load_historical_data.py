# Historical IG data may be downloaded from the following site.
# https://www.dukascopy.com/swiss/english/marketwatch/historical/
#
import argparse
import dateutil.tz
from collections import defaultdict
import re
import os
from pathlib import Path

import pandas as pd

from tradingo.signals import symbol_provider, symbol_publisher


def cli_app():

    app = argparse.ArgumentParser("load-historical-data")

    app.add_argument("--path", type=Path, required=True)
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
        path=args.path,
        arctic=a,
        dry_run=args.dry_run,
        universe=args.universe,
        provider=args.provider,
    )

    if args.dry_run:
        print(result)


@symbol_provider(instruments="instruments/{universe}", no_date=True)
@symbol_publisher(
    template="prices/{0}.{1}",
    symbol_prefix="{provider}.{universe}.",
)
def read_backfill(
    path: Path,
    **kwargs,
):

    data_files = defaultdict(list)

    for file in os.listdir(path):

        if match := re.match(FILE_REGEX, file):

            symbol, frequency, field, start_date, end_date = match.groups()

            # need to group by symbol

            data_files[(field.lower(), symbol)].append(file)

    # symbol, ohlc, symbol
    #
    #
    def read_file(f):
        out = pd.read_csv(
            path / f,
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
        return out.rename_axis("Datetime")

    result = pd.concat(
        (
            pd.concat(read_file(f) for f in files).query("~index.duplicated()")
            for files in data_files.values()
        ),
        axis=1,
        keys=data_files.keys(),
    ).reorder_levels([0, 2, 1], axis=1)

    return (
        (result["bid"]["Open"], ("bid", "open")),
        (result["bid"]["High"], ("bid", "high")),
        (result["bid"]["Low"], ("bid", "low")),
        (result["bid"]["Close"], ("bid", "close")),
        (result["ask"]["Open"], ("ask", "open")),
        (result["ask"]["High"], ("ask", "high")),
        (result["ask"]["Low"], ("ask", "low")),
        (result["ask"]["Close"], ("ask", "close")),
        # (result["last"]["Open"], ("last", "open")),
        # (result["last"]["High"], ("last", "high")),
        # (result["last"]["Low"], ("last", "low")),
        # (result["last"]["Close"], ("last", "close")),
        # (result["last"]["Volume"], ("last", "volume")),
    )


if __name__ == "__main__":

    import sys

    sys.argv.extend(
        [
            "--arctic-uri",
            "lmdb:///home/rory/dev/airflow/test/arctic.db",
            "--path",
            str(Path.home() / "Downloads/"),
            # "--dry-run",
            "--provider",
            "dukascopy",
            "--universe",
            "igtrading",
        ]
    )

    from arcticdb import Arctic

    main()
