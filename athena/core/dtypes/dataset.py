from typing import Optional

from athena.core.dataclasses import dataclass


@dataclass
class Dataset:
    r"""
    Preprocessed dataset.
    """
    #: Parquet link to the dataset.
    parquet_url: str


@dataclass
class TableSpec:
    r"""
    Table specification used for training.

    .. note::

        Currently, ``test_table_sample`` is the same as
        ``eval_table_sample``.

    .. important::

        ``able_sample + eval_table_sample == 100``
    """
    #: Name of a table.
    table_name: str
    #: Portion of entire table used to learn.
    #: The value should lie within interval :math:`[0, 100]`
    table_sample: Optional[float] = None
    #: Portion of entire table used to evaluate.
    #: The value should lie within interval :math:`[0, 100]`
    eval_table_sample: Optional[float] = None
    #: Portion of entire table used to test.
    #: The value should lie within interval :math:`[0, 100]`
    test_table_sample: Optional[float] = None
