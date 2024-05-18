import json
import os
import pathlib
import pandas as pd


HOME_DIR = pathlib.Path("/home/rory/dev/airflow/") / "trading"


def get_config():
    return json.loads((HOME_DIR / "config.json").read_text())


def get_instruments(config, key="equity") -> pd.DataFrame:
    return pd.read_csv(
        config[key]["file"],
        index_col=config[key]["index_col"],
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


def null_instruments(symbols):
    return pd.DataFrame(
        data="",
        index=symbols,
        columns=[
            "Name",
            "SEDOL",
            "ISIN",
            "CUSIP",
            "Incept. Date",
            "Gross Expense Ratio (%)",
            "Net Expense Ratio (%)",
            "Net Assets (USD)",
            "Net Assets as of",
            "Asset Class",
            "Sub Asset Class",
            "Region",
            "Market",
            "Location",
            "Investment Style",
            "Key Facts",
            "Avg. Annual Return: NAV Quarterly",
            "Avg. Annual Return: Price Quarterly",
            "Avg. Annual Return: NAV Monthly",
            "Avg. Annual Return: Price Monthly",
            "Yield",
            "Fixed Income Characteristics",
            "Sustainability Characteristics (MSCI ESG Fund Ratings)",
        ],
    )


def buffer(
    series: pd.Series,
):
    pass
