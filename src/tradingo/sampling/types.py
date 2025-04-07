"""
Tradingo instrument types.
"""

from enum import Enum


class InstrumentType(str, Enum):
    FUTURE = "future"
    OPTION = "option"
    ETF = "etf"
    CFD = "cfd"
    STOCK = "stock"
