import pprint
import random
import string
from athena.core.dtypes.rl.options import RewardOptions
import athena.data.fap.spark_utils.rl as srl
from typing import Dict, List, Optional, Tuple

from athena.data.fap.spark_utils.common import get_table_url, query_original_table
from athena.core.dataclasses import dataclass
from athena.data.fap.config import SparkConfig
from athena.data.fap.fapper import FAPper
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DataType

MAX_UINT32 = 4294967295
MAX_UPLOAD_PARQUET_TRIES = 10


@dataclass
class RLSparkFapper(FAPper):
    config: SparkConfig

    def __post_init_post_parse__(self) -> None:
        self._session = self._retrieve_session(self.config.asdict())

    def _retrieve_session(self, config_dict: Dict[str, str]) -> SparkSession:
        self.info(f"Retrieving (or creating) spark session with: \n{pprint.pformat(config_dict)}")
        spark_session = SparkSession.builder.enableHiveSupport()
        for k, v in config_dict.items():
            spark_session = spark_session.config(k, v)
        spark_session = spark_session.getOrCreate()
        spark_session.sparkContext.setLogLevel("ERROR")
        return spark_session

    @property
    def session(self) -> SparkSession:
        return self._session

    def fap(
        self,
        table_name: str,
        discrete_actions: bool,
        actions_space: Optional[List[str]] = None,
        sample_range: Optional[Tuple[float, float]] = None,
        reward_options: Optional[RewardOptions] = None,
        extra_columns: Dict[str, DataType] = {}
    ) -> str:
        reward_options = reward_options or RewardOptions()
        reward_col_name = reward_options.reward_col_name
        metrics_col_name = reward_options.metrics_col_name
        custom_reward = reward_options.custom_reward
        gamma = reward_options.gamma
        df = query_original_table(self._session, table_name)
        df = srl.reward_discount(self._session, df, reward_col_name, custom_reward, gamma)
        df = srl.hash_mdp_id_and_subsample(df, sample_range)
        df = srl.misc_column_preprocessing(df)
        df = srl.state_and_metrics_sparse2dense(
            df,
            states=srl.infer_states_names(df),
            metrics=srl.infer_metrics_names(df, metrics_col_name),
            metrics_col_name=metrics_col_name
        )
        if discrete_actions:
            if actions_space is None:
                raise ValueError("In discrete actions case, actions space must be given.")
            df = srl.discrete_actions_preprocessing(df, actions=actions_space)
        else:
            actions = srl.infer_actions_names(df)
            df = srl.continuous_actions_preprocessing(df, actions=actions)

        df = srl.select_relevant_columns(df, discrete_actions, reward_col_name, metrics_col_name, extra_columns)
        return self.upload_as_parquet(df)

    def upload_as_parquet(self, df: DataFrame) -> str:
        success = False
        for _ in range(MAX_UPLOAD_PARQUET_TRIES):
            suffix = "".join(random.choice(string.ascii_letters) for _ in range(10))
            rand_name = f"tmp_parquet_{suffix}"
            if not self._session.catalog._jcatalog.tableExists(rand_name):
                success = True
                break
        if not success:
            raise Exception(f"Failed to find name after {MAX_UPLOAD_PARQUET_TRIES} tries.")

        df.write.mode("errorifexists").format("parquet").saveAsTable(rand_name)
        parquet_url = get_table_url(self._session, rand_name)
        self.info(f"Saved parquet to {parquet_url}")
        return parquet_url
