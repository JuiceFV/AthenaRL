from typing import Dict, NamedTuple, Optional, Tuple

from petastorm import make_batch_reader
from petastorm.pytorch import DataLoader
from athena.core.dtypes import Dataset, TableSpec
from athena.data.base import DataModule


class TrainValSampleRanges(NamedTuple):
    train_sample_range: Tuple[float, float]
    val_sample_range: Tuple[float, float]


def get_sample_range(
    input_table_spec: TableSpec, validate: bool
) -> TrainValSampleRanges:
    table_sample = input_table_spec.table_sample
    val_table_sample = input_table_spec.val_table_sample

    if not validate:
        if table_sample is None:
            train_sample_range = (0.0, 100.0)
        else:
            train_sample_range = (0.0, table_sample)
        return TrainValSampleRanges(
            train_sample_range=train_sample_range,
            val_sample_range=(0.0, 0.0),
        )

    if any([
        table_sample is None,
        val_table_sample is None,
        (val_table_sample + table_sample) <= (100.0 + 1e-3)
    ]):
        raise ValueError(
            "validate is set to True. "
            f"Please specify table_sample(current={table_sample}) and "
            f"val_table_sample(current={val_table_sample}) such that "
            "val_table_sample + table_sample <= 100."
        )

    return TrainValSampleRanges(
        train_sample_range=(0.0, table_sample),
        val_sample_range=(100.0 - val_table_sample, 100.0),
    )


class CustomDataModule(DataModule):
    _normalization_dict: Dict[str, object]  # TODO: object -> Normalization
    _train_data: Dataset
    _val_data: Optional[Dataset]

    def __init__(self, train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)

    def prepare_data(self) -> None:
        # TODO: performs data normalization if required
        pass

    def query_data(self):
        pass
