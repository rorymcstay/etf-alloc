#!/usr/bin/env bash

export AIRFLOW_HOME=/home/rory/dev/airflow/
export AIRFLOW__CORE__LOAD_EXAMPLES=false

tmux new-session -d -s airflow airflow standalone
tmux new-session -d -s dtale dtale --arcticdb-uri lmdb:///home/rory/dev/airflow/test/arctic.db --arcticdb-use_store
tmux new-session -d -s jupyter jupyter notebook -d ./notebooks/
