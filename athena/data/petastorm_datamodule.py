from typing import Optional, Union
import pytorch_lightning as pl
from petastorm import make_batch_reader, TransformSpec
from petastorm.pytorch import DataLoader
from athena.core.dtypes.dataset import Dataset
from athena.core.dtypes.options import ReaderOptions
from athena.data.athena_datamodule import DATA_ITER_STEP, closing_iter, arbitrary_transform, collate_and_preprocess
from athena.preprocessing.batch_preprocessor import BatchPreprocessor
from athena.preprocessing.transforms.base import Compose, Transformation


class PetastormLightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data: Dataset,
        eval_data: Dataset,
        test_data: Dataset,
        transformation: Optional[Union[Compose, Transformation]],
        batch_preprocessor: BatchPreprocessor,
        reader_options: ReaderOptions
    ) -> None:
        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data
        self.transformation = transformation
        self.batch_preprocessor = batch_preprocessor
        self.reader_options = reader_options

    def get_dataloader(self, dataset: Dataset) -> DataLoader:
        data_reader = make_batch_reader(
            dataset.parquet_url,
            num_epochs=1,
            reader_pool_type=self.reader_options.petasorm_reader_pool_type,
            transform_spec=TransformSpec(arbitrary_transform(self.transformation))
        )
        dataloader = DataLoader(
            data_reader,
            batch_size=self.reader_options.minibatch_size,
            collate_fn=collate_and_preprocess(
                batch_preprocessor=self.batch_preprocessor,
                use_gpu=False
            )
        )
        return closing_iter(dataloader)

    def train_dataloader(self) -> DATA_ITER_STEP:
        return self.get_dataloader(self.train_data)

    def val_dataloader(self) -> Optional[DATA_ITER_STEP]:
        return self._get_eval_data()

    def test_dataloader(self) -> Optional[DATA_ITER_STEP]:
        return self._get_eval_data()

    def _get_eval_data(self) -> Optional[DATA_ITER_STEP]:
        eval_data = self.eval_data
        return None if not eval_data else self.get_dataloader(eval_data)
