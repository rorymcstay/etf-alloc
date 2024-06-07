import arcticdb as adb
import json
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import dtale

# from openbb import obb
from sklearn.decomposition import PCA
from tradingo import backtest
from tradingo.utils import get_instruments, get_config, with_instrument_details
from tradingo.api import Tradingo
import itertools
import seaborn as sns
from tradingo.plotting import plot_multi
import logging

from tradingo import sampling

t = Tradingo(
    name="ETFT", uri="lmdb:///home/rory/dev/airflow/test/arctic.db", provider="yfinance"
)
