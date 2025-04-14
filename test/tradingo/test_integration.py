import pathlib


from tradingo import templates
from tradingo.cli import DAG


TEMPLATES = pathlib.Path(templates.__file__)


CONFIG = {
    "prices": {
        "include": f"file://{TEMPLATES/'instruments/ig-trading.yaml'}",
        "variables": {
            "epics": [
                "ABC",
                "DEF",
            ]
        },
    },
}


def test_config():

    dag = DAG.from_config(CONFIG)

    assert list(dag.keys()) == []
