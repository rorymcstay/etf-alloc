from tradingo.api import Tradingo

from tradingo import signals


def test_buffer_signal(tradingo: Tradingo):

    return signals.buffered(
        signal=tradingo.portfolio.model.raw.shares(),
        thresholds=tradingo.portfolio.model.raw.shares() * 0.1,
        dry_run=True,
        start_date="2023-01-01",
        end_date="2024-05-31",
        config_name="test",
        model_name="model",
        library="portfolio",
        buffer_width=0.5,
        provider="yfinance",
        universe="etfs",
        arctic=tradingo,
    )
