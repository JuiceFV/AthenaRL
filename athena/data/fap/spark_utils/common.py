from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col


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
