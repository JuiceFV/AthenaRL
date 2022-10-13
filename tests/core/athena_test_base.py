import logging
from typing import Callable
import unittest

import pytorch_lightning as pl
from athena.core.config import create_config_class
from athena.core.tensorboard import SummaryWriterContext

from ruamel.yaml import YAML

SEED = 0


class AthenaTestBase(unittest.TestCase):
    def setUp(self) -> None:
        SummaryWriterContext._reset_globals()
        logging.basicConfig(level=logging.INFO)
        pl.seed_everything(SEED)

    def tearDown(self) -> None:
        SummaryWriterContext._reset_globals()

    @classmethod
    def run_from_config(cls, test_runner: Callable, config_path: str, use_gpu: bool):
        yaml = YAML(typ="safe")
        with open(config_path, "r") as f:
            config_dict = yaml.load(f.read())
        config_dict["use_gpu"] = use_gpu

        @create_config_class(test_runner)
        class ConfigClass:
            pass

        config = ConfigClass(**config_dict)
        return test_runner(**config.asdict())
