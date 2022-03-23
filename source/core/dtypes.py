"""Basic data types.
"""
import dataclasses

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Dataset:
    json_url: str


@dataclass
class TableSpec:
    table_name: str
    table_sample: Optional[float] = None
    val_table_sample: Optional[float] = None
    test_table_sample: Optional[float] = None


@dataclass
class ReaderOptions:
    minibatch_size: int = 1024
    reader_pool_type: str = "thread"
