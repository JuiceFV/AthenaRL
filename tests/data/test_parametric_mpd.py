import logging
import unittest
from typing import Optional

import numpy as np
import pytest
from athena.core.dtypes import Dataset
from athena.core.dtypes.rl.options import RewardOptions
from athena.data.data_extractor import DataExtractor
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import asc
from tests.data.data_ex import gen_parametric_mdp_pandas_df
from tests.data.athena_sql_test_base import AthenaSQLTestBase

logger = logging.getLogger(__name__)


def gen_parametric_data(sqlCtx: SparkSession, table_name: str):
    df = gen_parametric_mdp_pandas_df(False)
    df: DataFrame = sqlCtx.createDataFrame(df)
    logger.info("Created dataframe")
    df.show()
    df.createOrReplaceTempView(table_name)


class TestParametricMDP(AthenaSQLTestBase):
    def setUp(self):
        super().setUp()
        logging.getLogger(__name__).setLevel(logging.INFO)
        self.table_name = "test_table"
        logger.info(f"Table name is {self.table_name}")

    def gen_data(self):
        gen_parametric_data(self.sqlCtx, table_name=self.table_name)

    def read_parametric_data(self, custom_reward: Optional[str] = None, gamma: Optional[float] = None):
        data_extractor = DataExtractor(self.fapper)
        reward_options = RewardOptions(custom_reward=custom_reward, gamma=gamma)
        dataset: Dataset = data_extractor.query_data(
            table_name=self.table_name,
            discrete_actions=False,
            reward_options=reward_options
        )
        df = self.sqlCtx.read.parquet(dataset.parquet_url)
        df = df.orderBy(asc("sequence_number"))
        logger.info("Read parquet dataframe: ")
        df.show()
        return df

    @pytest.mark.serial
    def test_query_data_parametric(self):
        self.gen_data()
        df = self.read_parametric_data()
        df = df.toPandas()
        self.assert_all_except_reward(df)
        self.assertEq(df["reward"], np.array([0.0, 1.0, 4.0, 5.0], dtype=float))

        df = self.read_parametric_data(custom_reward="POWER(reward, 3) + 10")
        df = df.toPandas()
        self.assert_all_except_reward(df)
        self.assertEq(df["reward"], np.array([10.0, 11.0, 74.0, 135.0], dtype=float))

    def assert_all_except_reward(self, df: DataFrame):
        self.assertEq(df["sequence_number"], np.array([1, 2, 3, 4], dtype=int))

        state_features_presence = np.array(
            [
                [True, False, False, False, False],
                [False, True, False, False, False],
                [False, False, True, False, False],
                [False, False, False, True, False]
            ],
            dtype=bool
        )
        self.assertEq(df["state_features_presence"], state_features_presence)
        state_features = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            dtype=float
        )
        self.assertEqWithPresence(df["state_features"], state_features_presence, state_features)

        actions_presence = np.array(
            [
                [True, False, False, False],
                [False, False, False, True],
                [False, False, True, False],
                [False, True, False, False]
            ],
            dtype=bool
        )
        self.assertEq(df["actions_presence"], actions_presence)

        actions = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
            ]
        )
        self.assertEqWithPresence(df["actions"], actions_presence, actions)
        self.assertEq(df["not_terminal"], np.array([1, 1, 1, 0], dtype=bool))
        next_state_features_presence = np.array(
            [
                [False, True, False, False, False],
                [False, False, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
            ],
            dtype=bool,
        )
        self.assertEq(df["next_state_features_presence"], next_state_features_presence)
        next_state_features = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float
        )
        self.assertEqWithPresence(df["next_state_features"], next_state_features_presence, next_state_features)

        next_actions_presence = np.array(
            [
                [False, False, False, True],
                [False, False, True, False],
                [False, True, False, False],
                [False, False, False, False]
            ],
            dtype=bool
        )

        next_actions = np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]
            ]
        )
        self.assertEqWithPresence(df["next_actions"], next_actions_presence, next_actions)
        self.assertEq(df["time_diff"], np.array([1, 3, 1, 1]))
        self.assertEq(df["step"], np.array([1, 1, 1, 1]))


if __name__ == '__main__':
    unittest.main()
