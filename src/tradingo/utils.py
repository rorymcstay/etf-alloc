import json
import os
import pathlib
import pandas as pd


HOME_DIR = pathlib.Path("/home/rory/dev/airflow/") / "trading"


def get_config():
    return json.loads((HOME_DIR / "config.json").read_text())


def get_instruments(config) -> pd.DataFrame:
    return pd.read_csv(
        config["universe"]["file"],
        index_col=config["universe"]["index_col"],
        parse_dates=["Incept. Date"],
        date_format="%b %d, %Y",
    ).rename_axis("Symbol")


def with_instrument_details(
    dataframe: pd.DataFrame,
    instruments: pd.DataFrame,
    columns: list[str],
):
    """Add instrument details to column index"""
    return (
        dataframe.transpose()
        .rename_axis("Symbol")
        .merge(instruments[columns], left_index=True, right_index=True)
        .reset_index()
        .set_index([*columns, "Symbol"])
        .sort_index()
        .transpose()
    ).dropna()


def buffer(
    series: pd.Series,
):
    pass
