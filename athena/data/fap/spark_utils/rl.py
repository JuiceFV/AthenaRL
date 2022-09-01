from numbers import Number
from typing import Dict, Optional, Tuple, List, Union

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, crc32, explode, map_keys, udf, lit
from pyspark.sql.types import (DataType, FloatType, LongType, MapType,
                               StructType, StructField, ArrayType, BooleanType)

MAX_UINT32 = 4294967295


def make_get_step_udf():
    return udf(lambda _: 1, LongType())


def make_type_udf(return_type: DataType):
    return udf(lambda col: col, return_type)


def make_where_udf(arr: List[str]):
    def find(item: str) -> int:
        for i, arr_item in enumerate(arr):
            if arr_item == item:
                return i
        return len(arr)
    return udf(find, LongType())


def make_sparse2dense(df: DataFrame, col_name: str, possible_keys: List[Union[int, str]]) -> DataFrame:
    output_type = StructType(
        [
            StructField("presence", ArrayType(BooleanType()), False),
            StructField("dense", ArrayType(FloatType()), False)
        ]
    )

    def sparse2dense(map_col: Dict[Union[int, str], Number]) -> Tuple[List[bool], List[float]]:
        if not isinstance(map_col, dict):
            raise TypeError(f"{map_col} has type {type(map_col)} and is not a dict.")
        presence = []
        dense = []
        for key in possible_keys:
            val = map_col.get(key,  None)
            if val is not None:
                presence.append(True)
                dense.append(float(val))
            else:
                presence.append(False)
                dense.append(0.0)
        return presence, dense

    sparse2dense_udf = udf(sparse2dense, output_type)
    df = df.withColumn(col_name, sparse2dense_udf(col_name))
    df = df.withColumn(f"{col_name}_presence", col(f"{col_name}.presence"))
    df = df.withColumn(col_name, col(f"{col_name}.dense"))
    return df


def get_distinct_keys(df: DataFrame, col_name: str) -> List[Union[int, str]]:
    df = df.select(explode(map_keys(col_name)))
    return df.distinct().rdd.flatMap(lambda x: x).collect()


def infer_states_names(df: DataFrame) -> List[Union[int, str]]:
    state_keys = get_distinct_keys(df, "state_features")
    next_state_keys = get_distinct_keys(df, "next_state_features")
    return sorted(set(state_keys) | set(next_state_keys))


def infer_actions_names(df: DataFrame) -> List[Union[int, str]]:
    actions_keys = get_distinct_keys(df, "actions")
    next_actions_keys = get_distinct_keys(df, "next_actions")
    return sorted(set(actions_keys) | set(next_actions_keys))


def infer_metrics_names(df: DataFrame, metrics_col_name: str = "metrics") -> List[str]:
    return sorted(get_distinct_keys(df, metrics_col_name))


def reward_discount(
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
        expr = f"AGGREGATE(REVERSE({reward_col_name}), FLOAT(0), (s, x) -> FLOAT({repr(gamma)}) * s + x)"
        return calculate_custom_reward(df, expr)

    if custom_reward is not None:
        df = calculate_custom_reward(df, custom_reward)
    elif gamma is not None:
        df = calculate_gamma_discount_reward(df, gamma)
    return df


def hash_mdp_id_and_subsample(df: DataFrame, sample_range: Optional[Tuple[float, float]] = None) -> DataFrame:
    if sample_range:
        if not (0.0 <= sample_range[0] and sample_range[0] <= sample_range[1] and sample_range[1] <= 100.0):
            raise ValueError(f"Train and Eval samples must sum up to 100; Got {sample_range}")

    df = df.withColumn("mdp_id", crc32(col("mdp_id")))
    if sample_range:
        lower_bound = sample_range[0] / 100.0 * MAX_UINT32
        upper_bound = sample_range[1] / 100.0 * MAX_UINT32
        df = df.filter((lower_bound <= col("mdp_id")) & (col("mdp_id") <= upper_bound))
    return df


def misc_column_preprocessing(df: DataFrame) -> DataFrame:
    df = df.withColumn("step", lit(1))
    df = df.withColumn("sequence_number", col("sequence_number_ordinal"))
    return df


def state_and_metrics_sparse2dense(
    df: DataFrame, states: List[int], metrics: List[str], metrics_col_name: str = "metrics"
) -> DataFrame:
    type_udf = make_type_udf(MapType(LongType(), FloatType()))
    df = df.withColumn("next_state_features", type_udf("next_state_features"))
    df = df.withColumn(metrics_col_name, type_udf(metrics_col_name))
    df = make_sparse2dense(df, "state_features", states)
    df = make_sparse2dense(df, "next_state_features", states)
    df = make_sparse2dense(df, metrics_col_name, metrics)
    return df


def discrete_actions_preprocessing(df: DataFrame, actions: List[str]) -> DataFrame:
    where_udf = make_where_udf(actions)
    df = df.withColumn("actions", where_udf("actions"))
    type_udf = make_type_udf(LongType())
    df = df.withColumn("next_actions", where_udf(type_udf("next_actions")))

    def make_not_terminal_udf(actions: List[str]):
        return udf(lambda next_actions: next_actions < len(actions), BooleanType())

    not_terminal_udf = make_not_terminal_udf(actions)
    df = df.withColumn("not_terminal", not_terminal_udf("next_actions"))
    return df


def continuous_actions_preprocessing(df: DataFrame, actions: List[str]) -> DataFrame:
    type_udf = make_type_udf(MapType(LongType(), FloatType()))
    df = df.withColumn("next_actions", type_udf("next_actions"))

    def make_not_terminal_udf():
        return udf(lambda next_actions: len(next_actions) > 0, BooleanType())

    not_terminal_udf = make_not_terminal_udf()
    df = df.withColumn("not_terminal", not_terminal_udf("next_actions"))

    df = make_sparse2dense(df, "actions", actions)
    df = make_sparse2dense(df, "next_actions", actions)
    return df


def select_relevant_columns(
    df: DataFrame,
    discrete_actions: bool = True,
    reward_col_name: str = "reward",
    metrics_col_name: str = "metrics",
    extra_columns: Dict[str, DataType] = {}
) -> DataFrame:
    select_columns = [
        col(reward_col_name).cast(FloatType()),
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
        col(f"{metrics_col_name}_presence").cast(ArrayType(FloatType()))
    ]

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

    for column, data_type in extra_columns.items():
        select_columns.append(col(column).cast(data_type))

    return df.select(*select_columns)
