import importlib
import json
import logging
import os
import sys
from dataclasses import fields
from typing import Callable, Tuple, Type

import click
from ruamel.yaml import YAML

from athena.core.debug import DebugOnException


@click.group()
def reranking() -> None:
    DebugOnException.start()
    os.environ["USE_VANILLA_DATACLASS"] = "0"
    FORMAT = (
        "%(levelname).1s%(asctime)s.%(msecs)03d %(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=FORMAT, datefmt="%m%d %H%M%S"
    )


def _load_runner_and_config_class(runner: str) -> Tuple[Callable, Type]:
    """Load base runner callable and its configuration template.

    Args:
        runner (str): full path to a runner (including runner name)

    Returns:
        _type_: _description_
    """
    module_name, runner_name = runner.rsplit(".", 1)

    module = importlib.import_module(module_name)
    runner_callable = getattr(module, runner_name)

    from athena.core.config import create_config_class

    @create_config_class(runner_callable)
    class ConfigClass:
        pass

    return runner_callable, ConfigClass


@reranking.command(short_help="Run reranking service with given config file")
@click.argument("runner")
@click.argument("config_file", type=click.File("r"))
@click.option("--extra-options", default=None,
              help="Additional options complement and override the config")
def run(runner: str, config_file, extra_options):
    runner_callable, ConfigClass = _load_runner_and_config_class(runner)
    yaml = YAML(typ="safe")
    config_dict = yaml.load(config_file.read())
    assert config_dict is not None, "failed to read yaml file"
    if extra_options is not None:
        config_dict.update(json.loads(extra_options))
    config_dict = {
        field.name: config_dict[field.name]
        for field in fields(ConfigClass)
        if field.name in config_dict
    }
    config = ConfigClass(**config_dict)
    runner_callable(**config.asdict())


@reranking.command(short_help="Print JSON-schema of the runner")
@click.argument("runner")
def schema(runner: str):
    _, ConfigClass = _load_runner_and_config_class(runner)
    print(ConfigClass.__pydantic_model__.schema_json())


if __name__ == '__main__':
    reranking()
