# Tradingo

A functional strategy backtesting framework

Integrates with IG platform

```
uv sync
uv run tradingo-cli --help
```

## Universe
Defines the list of assets to trade. Create one as a list or csv of instruments or http

### Sampling
## Signals
Write a signal function and add it to the config like [here](https://github.com/rorymcstay/tradingo-plat/blob/main/config/trading/ig-currencies.json)


## Portfolio Construction
## Simulation
## Platform Integration
Fork [tradingo-plat](https://github.com/rorymcstay/tradingo-plat/tree/main) for airflow integration on docker-compose.
Fill in config/variables.json

## running the dag
```
tradingo-cli task list
```
