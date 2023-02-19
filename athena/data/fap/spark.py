import pprint
import random
import string
from typing import Dict, List, Optional, Tuple

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql.functions import col, crc32, lit, udf
from pyspark.sql.types import (ArrayType, BooleanType, FloatType, LongType,
                               MapType)

import athena.data.fap.spark_utils.common as scm
import athena.data.fap.spark_utils.rl as srl
from athena.core.dataclasses import dataclass
from athena.core.dtypes.preprocessing.options import PreprocessingOptions
from athena.core.dtypes.rl.options import RewardOptions
from athena.core.parameters import NormalizationParams
from athena.data.fap.config import SparkConfig
from athena.data.fap.fapper import FAPper
from athena.preprocessing.normalization import infer_normalization

MAX_UINT32 = 4294967295
MAX_UPLOAD_PARQUET_TRIES = 10


@dataclass
class SparkFapper(FAPper):
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

    def identify_normalization_params(
        self,
        table_name: str,
        col_name: str,
        preprocessing_options: PreprocessingOptions,
        seed: Optional[int] = None
    ) -> Dict[int, NormalizationParams]:
        self._session = self._retrieve_session(self.config.asdict())
        df = scm.query_original_table(self._session, table_name)
        df = scm.stratified_sampling_norm_spec(df, col_name, preprocessing_options.nsamples, seed)
        rows = df.collect()

        normalization_processor = infer_normalization(
            max_unique_enum_values=preprocessing_options.max_unique_enum_values,
            qunatile_size=preprocessing_options.quantile_size,
            quantile_k2_threshold=preprocessing_options.quantile_k2_threshold,
            skip_box_cox=preprocessing_options.skip_boxcox,
            skip_quantiles=preprocessing_options.quantile_size,
            feature_overrides=preprocessing_options.feature_overrides,
            allowed_features=preprocessing_options.allowed_features,
            assert_allowlist_feature_coverage=preprocessing_options.assert_allowlist_feature_coverage
        )
        return normalization_processor(rows)

    def fap(
        self,
        table_name: str,
        discrete_actions: bool,
        actions_space: Optional[List[str]] = None,
        sample_range: Optional[Tuple[float, float]] = None,
        reward_options: Optional[RewardOptions] = None,
        extra_columns: List[str] = []
    ) -> str:
        reward_options = reward_options or RewardOptions()
        reward_col_name = reward_options.reward_col_name
        metrics_col_name = reward_options.metrics_col_name
        custom_reward = reward_options.custom_reward
        gamma = reward_options.gamma
        self._session = self._retrieve_session(self.config.asdict())
        self.info(f"Fetching and Processing {table_name} with {reward_options}")
        df = scm.query_original_table(self._session, table_name)
        df = self._reward_discount(self._session, df, reward_col_name, custom_reward, gamma)
        df = self._hash_mdp_id_and_subsample(df, sample_range)
        df = self._misc_column_preprocessing(df)
        df = self._state_and_metrics_sparse2dense(
            df,
            states=srl.infer_trajectory_entity_names(df, "state_features"),
            metrics=srl.infer_nontrajectory_entity_names(df, metrics_col_name),
            metrics_col_name=metrics_col_name
        )
        if discrete_actions:
            if actions_space is None:
                raise ValueError("In discrete actions case, actions space must be given.")
            df = self._discrete_actions_preprocessing(df, actions=actions_space)
        else:
            actions_space = srl.infer_trajectory_entity_names(df, "actions")
            df = self._parametric_actions_preprocessing(df, actions=actions_space)

        df = self._process_extra_columns(df, extra_columns)
        df = self._select_relevant_columns(df, discrete_actions, reward_col_name, metrics_col_name, extra_columns)
        return self.upload_as_parquet(df)

    @staticmethod
    def _reward_discount(
        sqlCtx: SparkSession,
        df: DataFrame,
        reward_col_name: str = "reward",
        custom_reward: Optional[str] = None,
        gamma: Optional[float] = None
    ) -> DataFrame:
        if custom_reward is not None and gamma is not None:
            raise RuntimeError("Custom reward expresion and gamma passed;")

        def calculate_custom_reward(df: DataFrame, custom_reward: str) -> DataFrame:
            temp_table_name = "_tmp_calc_reward_df"
            temp_reward_name = "_tmp_reward_col"
            df.createOrReplaceTempView(temp_table_name)
            df = sqlCtx.sql(
                f"SELECT *, CAST(COALESCE({custom_reward}, 0) AS FLOAT)"
                f" as {temp_reward_name} FROM {temp_table_name}"
            )
            return df.drop(reward_col_name).withColumnRenamed(temp_reward_name, reward_col_name)

        def calculate_gamma_discount_reward(df: DataFrame, gamma: float) -> DataFrame:
            expr = f"AGGREGATE(REVERSE({reward_col_name}), FLOAT(0), (s, x) -> FLOAT({gamma}) * s + x)"
            return calculate_custom_reward(df, expr)

        if custom_reward is not None:
            df = calculate_custom_reward(df, custom_reward)
        elif gamma is not None:
            df = calculate_gamma_discount_reward(df, gamma)

        if isinstance(df.schema[reward_col_name].dataType, ArrayType):
            max_seq_len = scm.get_max_sequence_length(df, reward_col_name)
            df = scm.vector_padding(df, reward_col_name, max_seq_len)

        return df

    @staticmethod
    def _hash_mdp_id_and_subsample(df: DataFrame, sample_range: Optional[Tuple[float, float]] = None) -> DataFrame:
        if sample_range:
            if not (0.0 <= sample_range[0] and sample_range[0] <= sample_range[1] and sample_range[1] <= 100.0):
                raise ValueError(f"Train and Eval samples must sum up to 100; Got {sample_range}")

        df = df.withColumn("mdp_id", crc32(col("mdp_id")))
        if sample_range:
            lower_bound = sample_range[0] / 100.0 * MAX_UINT32
            upper_bound = sample_range[1] / 100.0 * MAX_UINT32
            df = df.filter((lower_bound <= col("mdp_id")) & (col("mdp_id") <= upper_bound))
        return df

    @staticmethod
    def _misc_column_preprocessing(df: DataFrame) -> DataFrame:
        df = df.withColumn("step", lit(1))
        df = df.withColumn("sequence_number", col("sequence_number_ordinal"))
        return df

    @staticmethod
    def _state_and_metrics_sparse2dense(
        df: DataFrame, states: List[int], metrics: List[str], metrics_col_name: str = "metrics"
    ) -> DataFrame:
        type_udf = scm.make_type_udf(MapType(LongType(), FloatType()))
        df = df.withColumn("next_state_features", type_udf("next_state_features"))
        df = df.withColumn(metrics_col_name, type_udf(metrics_col_name))
        df = scm.make_sparse2dense(df, "state_features", states)
        df = scm.make_sparse2dense(df, "next_state_features", states)
        df = scm.make_sparse2dense(df, metrics_col_name, metrics)
        return df

    @staticmethod
    def _discrete_actions_preprocessing(df: DataFrame, actions: List[str]) -> DataFrame:
        where_udf = scm.make_where_udf(actions)
        df = df.withColumn("actions", where_udf("actions"))
        type_udf = scm.make_type_udf(LongType())
        df = df.withColumn("next_actions", where_udf(type_udf("next_actions")))

        def make_not_terminal_udf(actions: List[str]):
            return udf(lambda next_actions: next_actions < len(actions), BooleanType())

        not_terminal_udf = make_not_terminal_udf(actions)
        df = df.withColumn("not_terminal", not_terminal_udf("next_actions"))
        return df

    @staticmethod
    def _parametric_actions_preprocessing(df: DataFrame, actions: List[str]) -> DataFrame:
        type_udf = scm.make_type_udf(MapType(LongType(), FloatType()))
        df = df.withColumn("next_actions", type_udf("next_actions"))

        def make_not_terminal_udf():
            return udf(lambda next_actions: len(next_actions) > 0, BooleanType())

        not_terminal_udf = make_not_terminal_udf()
        df = df.withColumn("not_terminal", not_terminal_udf("next_actions"))

        df = scm.make_sparse2dense(df, "actions", actions)
        df = scm.make_sparse2dense(df, "next_actions", actions)
        return df

    def _process_extra_columns(self, df: DataFrame, extra_columns: List[str] = []) -> DataFrame:
        for col_name in extra_columns:
            processor = getattr(self, f"_process_{col_name}", None)
            if processor is not None:
                df = processor(df)
            else:
                self.warning(f"No processor found for the extra column {col_name}")
        return df

    @staticmethod
    def _process_state_sequence_features(df: DataFrame) -> DataFrame:
        keys = srl.infer_nontrajectory_entity_names(df, "state_sequence_features", is_sequence=True)
        max_seq_len = scm.get_max_sequence_length(df, "state_sequence_features")
        df = scm.make_sparse2dense(df, "state_sequence_features", keys, max_seq_len)
        return df

    @staticmethod
    def _select_state_sequence_features(col_name: str) -> List[Column]:
        columns = [
            col(col_name).cast(ArrayType(FloatType())),
            col(f"{col_name}_presence").cast(ArrayType(BooleanType())),
        ]
        return columns

    def _select_relevant_columns(
        self,
        df: DataFrame,
        discrete_actions: bool = True,
        reward_col_name: str = "reward",
        metrics_col_name: str = "metrics",
        extra_columns: List[str] = []
    ) -> DataFrame:
        select_columns = [
            col("state_features").cast(ArrayType(FloatType())),
            col("state_features_presence").cast(ArrayType(BooleanType())),
            col("next_state_features").cast(ArrayType(FloatType())),
            col("next_state_features_presence").cast(ArrayType(BooleanType())),
            col("not_terminal").cast(BooleanType()),
            col("actions_probability").cast(FloatType()),
            col("mdp_id").cast(LongType()),
            col("sequence_number").cast(LongType()),
            col("step").cast(LongType()),
            col("time_diff").cast(LongType()),
            col(metrics_col_name).cast(ArrayType(FloatType())),
            col(f"{metrics_col_name}_presence").cast(ArrayType(BooleanType()))
        ]

        if isinstance(df.schema[reward_col_name].dataType, ArrayType):
            select_columns.extend(
                [
                    col(reward_col_name).cast(ArrayType(FloatType())),
                    col(f"{reward_col_name}_presence").cast(ArrayType(BooleanType()))
                ]
            )
        else:
            select_columns.append(col(reward_col_name).cast(FloatType()))

        if discrete_actions:
            select_columns.extend(
                [
                    col("actions").cast(LongType()),
                    col("next_actions").cast(LongType())
                ]
            )
        else:
            select_columns.extend(
                [
                    col("actions").cast(ArrayType(FloatType())),
                    col("next_actions").cast(ArrayType(FloatType())),
                    col("actions_presence").cast(ArrayType(BooleanType())),
                    col("next_actions_presence").cast(ArrayType(BooleanType()))
                ]
            )

        for column in extra_columns:
            selector = getattr(self, f"_select_{column}", lambda cname: [col(cname)])
            select_columns.extend(selector(column))

        return df.select(*select_columns)

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
        parquet_url = scm.get_table_url(self._session, rand_name)
        self.info(f"Saved parquet to {parquet_url}")
        return parquet_url
