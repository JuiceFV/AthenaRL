from typing import Optional
from source.core.dataclasses import dataclass
from source.core.logger import LoggerMixin
from source.core.dtypes import TableSpec
from source.data.base import DataModule


@dataclass
class ModelManager(LoggerMixin):
    def __post_init_post_parse__(self):
        pass

    def get_data_module(
        self,
        input_table_spec: TableSpec
    ) -> Optional[DataModule]:
        return None
