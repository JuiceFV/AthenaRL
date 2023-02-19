from typing import Optional

from athena.core.dataclasses import dataclass


@dataclass
class Dataset:
    r"""
    Preprocessed dataset.
    """
    #: Parquet link to a dataset.
    parquet_url: str


@dataclass
class TableSpec:
    r"""
    Table specification used for training.

    .. note::

        Currently, ``test_table_sample`` is the same as ``eval_table_sample``.

    .. important::

        ``table_sample + eval_table_sample == 100``
    """
    #: Name of a table.
    table_name: str
    #: Portion of entire table used to learn.
    table_sample: Optional[float] = None
    #: Portion of entire table used to evaluate.
    eval_table_sample: Optional[float] = None
    #: Portion of entire table used to test.
    test_table_sample: Optional[float] = None
