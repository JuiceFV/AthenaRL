from numbers import Number
from typing import List, Dict, Union, Tuple, Optional, TypeVar

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, explode, map_keys, udf, size, collect_list
from pyspark.sql.types import (ArrayType, BooleanType, DataType, FloatType,
                               LongType, StructField, StructType)

SparseKeyType = TypeVar("SparseKeyType", int, str)


def query_original_table(sqlCtx: SparkSession, table_name: str) -> DataFrame:
    return sqlCtx.sql(f"SELECT * FROM {table_name}")


def get_table_url(sqlCtx: SparkSession, table_name: str) -> str:
    url = sqlCtx.sql(f"DESCRIBE FORMATTED {table_name}") \
                .filter((col("col_name") == "Location")) \
                .select("data_type")                     \
                .toPandas()                              \
                .astype(str)["data_type"]                \
                .values[0]
    schema, path = str(url).split(":")
    return f"{schema}://{path}"


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


def get_distinct_keys(df: DataFrame, col_name: str) -> List[SparseKeyType]:
    df = df.select(explode(map_keys(col_name)))
    return df.distinct().rdd.flatMap(lambda x: x).collect()


def get_max_sequence_length(df: DataFrame, col_name: str) -> int:
    df = df.select(size(col_name))
    return df.rdd.flatMap(lambda x: x).max()


def get_sequence_keys(df: DataFrame, col_name: str) -> List[SparseKeyType]:
    # FIXME(ezarifov): Potentially it's bug. In case list elements have different
    # representation, for example [{0: 1.0, 1: 0.2}, {0: 1.0, 1: 0.2, 2: 0.7}]
    # only common features will be considered, i.e. `get_sequence_keys` will
    # return [0, 1]. To fix it delete the `limit(1)`.
    df = df.select(explode(col_name)).limit(1)
    return df.rdd.flatMap(lambda x: x[0]).collect()


def make_sparse2dense(
    df: DataFrame,
    col_name: str,
    possible_keys: List[Union[int, str]],
    max_seq_len: Optional[int] = None
) -> DataFrame:
    output_type = StructType(
        [
            StructField("presence", ArrayType(BooleanType()), False),
            StructField("dense", ArrayType(FloatType()), False)
        ]
    )

    def map_sparse2dense(map_col: Dict[Union[int, str], Number]) -> Tuple[List[bool], List[float]]:
        if not isinstance(map_col, dict):
            raise TypeError(f"{map_col} has type {type(map_col)} and is not a dict.")
        presence = [False] * len(possible_keys)
        dense = [0.0] * len(possible_keys)
        for i, key in enumerate(possible_keys):
            val = map_col.get(key,  None)
            if val is not None:
                presence[i] = True
                dense[i] = float(val)
        return presence, dense

    def sequence_sparse2dense(sequence_col: List[Dict[Union[int, str], Number]]) -> Tuple[List[bool], List[float]]:
        if not isinstance(sequence_col, list):
            raise TypeError(f"{sequence_col} has type {type(sequence_col)} and is not a list.")
        presence = [False] * (len(possible_keys) * max_seq_len)
        dense = [0.0] * (len(possible_keys) * max_seq_len)
        dense_pos = 0
        for item in sequence_col:
            for key in possible_keys:
                val = item.get(key,  None)
                if val is not None:
                    presence[dense_pos] = True
                    dense[dense_pos] = float(val)
                dense_pos += 1
        return presence, dense

    sparse2dense_udf = udf(sequence_sparse2dense if max_seq_len is not None else map_sparse2dense, output_type)
    df = df.withColumn(col_name, sparse2dense_udf(col_name))
    df = df.withColumn(f"{col_name}_presence", col(f"{col_name}.presence"))
    df = df.withColumn(col_name, col(f"{col_name}.dense"))
    return df


def vector_padding(df: DataFrame, vector_column: str, max_vec_len: int) -> DataFrame:
    output_type = StructType(
        [
            StructField("presence", ArrayType(BooleanType(), False)),
            StructField("dense", ArrayType(FloatType(), False))
        ]
    )

    def pad(vector: List[float]) -> Tuple[List[bool], List[float]]:
        if not isinstance(vector, list):
            raise TypeError(f"{vector} has type {type(vector)} and is not a vector.")
        presence = [False] * max_vec_len
        dense = [0.0] * max_vec_len
        for i, value in enumerate(vector):
            presence[i] = True
            dense[i] = float(value)
        return presence, dense

    sparse2dense_udf = udf(pad, output_type)
    df = df.withColumn(vector_column, sparse2dense_udf(vector_column))
    df = df.withColumn(f"{vector_column}_presence", col(f"{vector_column}.presence"))
    df = df.withColumn(vector_column, col(f"{vector_column}.dense"))
    return df


def stratified_sampling_norm_spec(
    df: DataFrame, col_name: str, nsamples: int, seed: Optional[int] = None
) -> DataFrame:
    if isinstance(df.schema[col_name].dataType, ArrayType):
        df = df.select(explode(col(col_name)).alias(col_name))
    df = df.select(explode(col(col_name).alias("features")).alias("fname", "fvalue"))
    counts_df: DataFrame = df.groupBy("fname").count()
    fracs = {}
    for row in counts_df.collect():
        if nsamples > row["count"]:
            raise RuntimeError(
                f"nsamples should be less than min(# of a feature); Got {nsamples} > {row['count']} for {row}"
            )
        fracs[row["fname"]] = nsamples / row["count"]

    df = df.sampleBy("fname", fractions=fracs, seed=seed)
    df = df.groupBy("fname").agg(collect_list("fvalue").alias("fvalues"))
    return df
