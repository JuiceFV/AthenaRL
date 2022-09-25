from typing import List
from pyspark.sql import DataFrame
from athena.data.fap.spark_utils.common import get_distinct_keys, get_sequence_keys, SparseKeyType


def infer_trajectory_entity_names(df: DataFrame, col_name: str, is_sequence: bool = False) -> List[SparseKeyType]:
    keys_getter = get_sequence_keys if is_sequence else get_distinct_keys
    entity_keys = keys_getter(df, col_name)
    next_entity_keys = keys_getter(df, f"next_{col_name}")
    return sorted(set(entity_keys) | set(next_entity_keys))


def infer_nontrajectory_entity_names(df: DataFrame, col_name: str, is_sequence: bool = False) -> List[SparseKeyType]:
    keys_getter = get_sequence_keys if is_sequence else get_distinct_keys
    return sorted(keys_getter(df, col_name))
