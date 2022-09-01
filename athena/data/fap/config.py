import abc
from typing import Dict
import athena
from os.path import abspath, dirname, join
from athena.core.dataclasses import dataclass
from dataclasses import fields


SPARK_JAR_REL_PATH = "preprocessing/target/preprocessing-0.1.jar"


class FSConfigBase(abc.ABC):
    @abc.abstractmethod
    def asdict(self) -> Dict[str, str]:
        pass


@dataclass
class SparkConfig(FSConfigBase):
    spark_app_name: str = "NastenkaSolnishko"
    spark_sql_session_timeZone: str = "UTC"
    spark_driver_host: str = "127.0.0.1"
    spark_master: str = "local[*]"
    spark_sql_warehouse_dir: str = abspath("spark-warehouse")
    spark_sql_shuffle_partitions: str = "12"
    spark_sql_execution_arrow_enabled: str = "true"
    spark_driver_extraClassPath: str = join(dirname(dirname(athena.__file__)), SPARK_JAR_REL_PATH)
    spark_sql_catalogImplementation: str = "hive"

    def asdict(self) -> Dict[str, str]:
        return {".".join(field.name.split("_")): getattr(self, field.name) for field in fields(self)}
