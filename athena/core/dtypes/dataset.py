from typing import Optional

from athena.core.dataclasses import dataclass

@dataclass
class Dataset:
    parquet_url: str


@dataclass
class TableSpec:
    table_name: str
    table_sample: Optional[float] = None
    eval_table_sample: Optional[float] = None
    test_table_sample: Optional[float] = None