import logging
import os
import shutil

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pyspark import SparkConf
from athena.data.fap.config import SparkConfig

from sparktestingbase.sqltestcase import SQLTestCase

from athena.data.fap.spark import SparkFapper

HIVE_METASTORE = "metastore_db"
TEST_CLASS_PTR = 0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class AthenaSQLTestBase(SQLTestCase):
    def getConf(self):
        config = SparkConf()
        for k, v in SparkConfig().asdict().items():
            config.set(k, v)
        return config

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        global TEST_CLASS_PTR
        cls.test_class_seed = TEST_CLASS_PTR
        logger.info(f"Allocating seed {cls.test_class_seed} to {cls.__name__}")
        TEST_CLASS_PTR += 1

    def setUp(self):
        super().setUp()
        assert not os.path.isdir(HIVE_METASTORE), f"Delete {HIVE_METASTORE} first!"

        pl.seed_everything(self.test_class_seed)
        logging.basicConfig()
        self.fapper = SparkFapper(config=SparkConfig())

    def assertEq(self, pd_series: pd.Series, arr: np.ndarray):
        series_as_arr = np.array(pd_series.tolist())
        np.testing.assert_equal(series_as_arr, arr)

    def assertAllClose(self, pd_series: pd.Series, arr: np.ndarray):
        series_as_arr = np.array(pd_series.tolist())
        np.testing.assert_allclose(series_as_arr, arr)

    def assertEqWithPresence(self, pd_series: pd.Series, presence: np.ndarray, arr: np.ndarray):
        series_as_arr = np.array(pd_series.tolist())
        present_sa = series_as_arr[presence]
        present_arr = arr[presence]
        np.testing.assert_equal(present_sa, present_arr)

    def tearDown(self) -> None:
        super().tearDown()

        if os.path.isdir(HIVE_METASTORE):
            shutil.rmtree(HIVE_METASTORE)
