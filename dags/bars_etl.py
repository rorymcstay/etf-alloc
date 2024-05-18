import logging
import os
import json
import pathlib
import re
from datetime import datetime
from airflow.exceptions import DuplicateTaskIdFound

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Operator

import arcticdb as adb
import pandas as pd

from tradingo.symbols import ARCTIC_URL, symbol_provider, symbol_publisher, lib_provider
from tradingo import signals
from tradingo.backtest import backtest
from tradingo.portfolio import instrument_ivol, portfolio_construction
from tradingo import sampling

logger = logging.getLogger(__name__)


HOME_DIR = pathlib.Path(os.environ["AIRFLOW_HOME"]) / "trading"
START_DATE = "2018-01-01 00:00:00+00:00"
PROVIDER = "yfinance"
AUM = 70_000
ARCTIC = adb.Arctic(ARCTIC_URL)


def get_config(config_path):
    return json.loads(
        (pathlib.Path(config_path) or (HOME_DIR / "config.json")).read_text()
    )


def get_instruments(config, key="equity") -> pd.DataFrame:
    return pd.read_csv(
        config[key]["file"],
        index_col=config[key]["index_col"],
    ).rename_axis("Symbol")


def make_task_id(task, symbol=None) -> str:
    if symbol:
        symbol = re.sub(r"\^|/", ".", str(symbol).strip())
        return f"{task}_{symbol}"
    return task


@symbol_provider(
    positions="portfolio/{name}.{stage}?as_of=-1",
    previous_positions="portfolio/{name}.{stage}?as_of=-2",
)
@symbol_publisher("trades/{stage}.{name}")
def calculate_trades(
    name: str,
    stage: str,
    positions: pd.DataFrame,
    previous_positions: pd.DataFrame,
    previous_refdate: pd.Timestamp,
    **kwargs,
):

    logger.info(
        "Calculating %s trades for %s previous_refdate=%s",
        stage,
        name,
        previous_refdate,
    )

    return (
        positions
        - previous_positions.reindex_like(positions, method="ffill").fillna(0.0),
    )


def get_or_create_signal(
    name, signal, function, config, depends_on, dag: DAG, prices
) -> Operator:
    print("getting ", name)
    signal = signal.copy()
    task_id = make_task_id("signal", name)
    try:
        signal = PythonOperator(
            task_id=task_id,
            python_callable=getattr(signals, function),
            op_kwargs={
                "model_name": "trend",
                "start_date": START_DATE,
                "end_date": "{{ data_interval_end }}",
                "signal_name": name,
                "config_name": dag.dag_id,
                "provider": PROVIDER,
                **signal,
            },
        )
        _ = prices >> signal

    except DuplicateTaskIdFound:
        signal = dag.get_task(task_id)

    for sig in depends_on:
        signal_config = config["signal_configs"][sig]
        _ = (
            get_or_create_signal(
                sig,
                signal_config["kwargs"],
                signal_config["function"],
                config,
                signal_config.get("depends_on", []),
                dag,
                prices,
            )
            >> signal
        )
    return signal


DAG_DEFAULT_ARGS = {
    "depends_on_past": True,
    "max_active_runs": 1,
}


def trading_dag(
    dag_id,
    config_path=str(HOME_DIR / "config.json"),
    start_date=datetime.fromisoformat("2024-04-25 00:00:00+00:00"),
    schedule="0 0 * * 1-5",
    catchup=True,
):
    with DAG(
        dag_id=dag_id,
        start_date=start_date,
        schedule=schedule,
        catchup=catchup,
        default_args=DAG_DEFAULT_ARGS,
    ) as dag:
        config = get_config(config_path)

        prices = PythonOperator(
            task_id=make_task_id("sample_equity"),
            python_callable=sampling.sample_equity,
            op_kwargs={
                "universe": get_instruments(config, "equity").index,
                "start_date": START_DATE,
                "end_date": "{{ data_interval_end }}",
                "provider": PROVIDER,
                "config_name": config["name"],
            },
        )

        if "futures" in config:

            _ = PythonOperator(
                task_id="sample_futures",
                python_callable=sampling.sample_futures,
                op_kwargs=dict(
                    universe=config["futures"],
                    provider="cboe",
                    start_date="{{ data_interval_start }}",
                    end_date="{{ data_interval_end }}",
                    config_name=config["name"],
                ),
            )

        if "options" in config:

            _ = PythonOperator(
                task_id="sample_options",
                python_callable=sampling.sample_options,
                op_kwargs=dict(
                    universe=config["options"],
                    provider="cboe",
                    start_date="{{ data_interval_start }}",
                    end_date="{{ data_interval_end }}",
                    config_name=config["name"],
                ),
            )
        ivol = PythonOperator(
            task_id=make_task_id("ivol"),
            python_callable=instrument_ivol,
            op_kwargs={
                "start_date": START_DATE,
                "end_date": "{{ data_interval_end }}",
                "provider": PROVIDER,
                "config_name": config["name"],
            },
        )

        _ = prices >> ivol

        for name, strategy in config["portfolio"].items():
            pos = PythonOperator(
                task_id=make_task_id("portfolio_construction", name),
                python_callable=portfolio_construction,
                op_kwargs={
                    "start_date": START_DATE,
                    "end_date": "{{ data_interval_end }}",
                    "name": name,
                    "config_name": config["name"],
                    "provider": PROVIDER,
                },
            )
            _ = pos >> PythonOperator(
                task_id=make_task_id("backtest", name),
                python_callable=backtest,
                op_kwargs={
                    "start_date": START_DATE,
                    "end_date": "{{ data_interval_end }}",
                    "name": name,
                    "stage": "raw",
                    "config_name": config["name"],
                    "provider": PROVIDER,
                },
            )
            # _ = pos >> PythonOperator(
            #    task_id=make_task_id("calculate_trades", name),
            #    python_callable=calculate_trades,
            #    op_kwargs={
            #        "start_date": START_DATE,
            #        "end_date": "{{ data_interval_end }}",
            #        "previous_refdate": "{{ data_interval_start }}",
            #        "name": name,
            #        "stage": "raw",
            #        "provider": PROVIDER,
            #        "config_name": config["name"],
            #    },
            # )

            for signal in strategy["signal_weights"]:
                signal_config = config["signal_configs"][signal].copy()
                _ = (
                    prices
                    >> get_or_create_signal(
                        signal,
                        signal_config["kwargs"],
                        signal_config["function"],
                        config,
                        signal_config.get("depends_on", []),
                        dag,
                        prices,
                    )
                    >> pos
                )

    return dag


etft = trading_dag(dag_id="ETFT", start_date=pd.Timestamp("2024-05-09 00:00:00+00:00"))
