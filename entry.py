import importlib
from typing import Callable, Tuple, Type
from dataclasses import dataclass

import click

from source.core.debug import DebugOnException


@click.group()
def reranking() -> None:
    DebugOnException.start()


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

    from source.core.confg import create_config_class

    @create_config_class(runner_callable)
    class ConfigClass:
        pass

    return runner_callable, ConfigClass

@reranking.command(short_help="Run reranking service with given config file")
@click.argument("runner")
@click.argument("config", type=click.File("r"))
@click.option("--extra-options", default=None,
              help="Additional options complement and override the config")
def run(runner: str, config, extra_options):
    runner_callable, ConfigClass = _load_runner_and_config_class(runner)


if __name__ == '__main__':
    reranking()
