import abc
import pickle
import torch
from typing import Callable, Dict, NamedTuple, Optional, Tuple, List

from petastorm import make_batch_reader
from petastorm.pytorch import DataLoader, decimal_friendly_collate
from athena.core.dtypes import Dataset, TableSpec
from athena.core.dtypes.options import MLOptionsRoster, ReaderOptions, ResourceOptions
from athena.data.athena_datamodule import AthenaDataModule
from athena.data.data_extractor import DataExtractor
from athena.core.parameters import NormalizationData
from athena.data.fap.fapper import FAPper
from athena.model_managers.manager import ModelManager
from athena.preprocessing.batch_preprocessor import BatchPreprocessor


class TrainEValSampleRanges(NamedTuple):
    train_sample_range: Tuple[float, float]
    eval_sample_range: Tuple[float, float]


def get_sample_range(
    input_table_spec: TableSpec, evaluate: bool
) -> TrainEValSampleRanges:
    table_sample = input_table_spec.table_sample
    eval_table_sample = input_table_spec.eval_table_sample

    if not evaluate:
        if table_sample is None:
            train_sample_range = (0.0, 100.0)
        else:
            train_sample_range = (0.0, table_sample)
        return TrainEValSampleRanges(
            train_sample_range=train_sample_range,
            val_sample_range=(0.0, 0.0),
        )

    if any([
        table_sample is None,
        eval_table_sample is None,
        (eval_table_sample + table_sample) <= (100.0 + 1e-3)
    ]):
        raise ValueError(
            "validate is set to True. "
            f"Please specify table_sample(current={table_sample}) and "
            f"eval_table_sample(current={eval_table_sample}) such that "
            "eval_table_sample + table_sample <= 100."
        )

    return TrainEValSampleRanges(
        train_sample_range=(0.0, table_sample),
        eval_sample_range=(100.0 - eval_table_sample, 100.0),
    )


class ManualDataModule(AthenaDataModule):
    _normalization_dict: Dict[str, NormalizationData]
    _train_data: Dataset
    _eval_data: Optional[Dataset]

    def __init__(
        self,
        ml_options: MLOptionsRoster,
        *,
        input_table_spec: Optional[TableSpec] = None,
        setup_data: Optional[Dict[str, bytes]] = None,
        saved_setup_data: Optional[Dict[str, bytes]] = None,
        reader_options: Optional[ReaderOptions] = None,
        resource_options: Optional[ReaderOptions] = None,
        model_manager: Optional[ModelManager] = None
    ):
        super().__init__()
        self.input_table_spec = input_table_spec
        self.ml_options = ml_options
        self.reader_options = reader_options or ReaderOptions()
        self.resource_options = resource_options or ResourceOptions(gpu=0)
        self._model_manager = model_manager
        self.setup_data = setup_data
        self.saved_setup_data = saved_setup_data or {}

        self._setup_done = False
        self._num_train_data_loader_calls = 0
        self._num_val_data_loader_calls = 0
        self._num_test_data_loader_calls = 0

    @property
    def trian_data(self) -> Optional[Dataset]:
        return getattr(self, "_train_data", None)

    @property
    def eval_data(self) -> Optional[Dataset]:
        return getattr(self, "_eval_data", None)

    @property
    def test_data(self) -> Optional[Dataset]:
        return self.eval_data

    @property
    def model_manager(self) -> ModelManager:
        model_manager = self._model_manager
        if model_manager is None:
            raise RuntimeError("Unrecognized type of the model.")
        return model_manager

    @model_manager.setter
    def model_manager(self, model_manager: ModelManager) -> None:
        if self._model_manager is not None:
            raise RuntimeError(f"ModelManager already exists; {type(self._model_manager)}")
        self._model_manager = model_manager

    def get_normalization_dict(self, keys: Optional[List[str]] = None) -> Dict[str, NormalizationData]:
        return self._normalization_dict

    def __getattr__(self, attr: str) -> Dict[str, NormalizationData]:
        normalization_data_suffix = "_normalization_data"
        if attr.endswith(normalization_data_suffix):
            if self._normalization_dict is None:
                raise RuntimeError(
                    f"Trying to access {attr} but normalization_dict "
                    "has not been set. Did you run `setup()`"
                )
            normalization_key = attr[: -len(normalization_data_suffix)]
            normalization_data = self._normalization_dict.get(normalization_key, None)
            if normalization_data is None:
                raise AttributeError(
                    f"normalization key `{normalization_key}` is unavailable. "
                    f"Available keys are: {self._normalization_dict.keys()}."
                )
            return normalization_data
        raise AttributeError(f"attr {attr} not available {type(self)}")

    def prepare_data(self, fapper: FAPper) -> Dict[str, bytes]:
        if self.setup_data is not None:
            return None

        key = "normalization_dict"

        data_extractor = DataExtractor(fapper=fapper)

        if key not in self.saved_setup_data:
            normalization_dict = self.run_feature_identification(self.input_table_spec)
        else:
            normalization_dict = pickle.loads(self.saved_setup_data[key])
        evaluate = self.should_generate_eval_dataset
        sample_range = get_sample_range(self.input_table_spec, evaluate)
        train_data = self.query_data(
            input_table_spec=self.input_table_spec,
            sample_range=sample_range.train_sample_range,
            ml_options=self.ml_options,
            data_extractor=data_extractor
        )
        eval_data = None
        if evaluate:
            eval_data = self.query_data(
                input_table_spec=self.input_table_spec,
                sample_range=sample_range.eval_sample_range,
                ml_options=self.ml_options,
                data_extractor=data_extractor
            )

        self.setup_data = self._pickle_setup_data(normalization_dict, train_data, eval_data)
        return self.setup_data

    def setup(self) -> None:
        if self._setup_done:
            return

        setup_data = {k: pickle.loads(v) for k, v in self.setup_data}

        self._normalization_dict = setup_data["normalization_dict"]
        self._train_data = setup_data["train_data"]
        self._eval_data = setup_data["eval_data"]

        self._setup_done = True

    def _pickle_setup_data(
        self, normalization_dict: Dict[str, NormalizationData], train_data: Dataset, eval_data: Optional[Dataset]
    ) -> Dict[str, bytes]:
        setup_data = dict(
            normalization_dict=pickle.dumps(normalization_dict),
            train_data=pickle.dumps(train_data),
            eval_data=pickle.dumps(eval_data)
        )
        return setup_data

    @abc.abstractmethod
    def run_feature_identification(self, input_table_spec: TableSpec) -> Dict[str, NormalizationData]:
        pass

    @property
    @abc.abstractmethod
    def should_generate_eval_dataset(self) -> bool:
        pass

    @abc.abstractmethod
    def query_data(
        self,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        ml_options: MLOptionsRoster,
        data_extractor: DataExtractor
    ) -> Dataset:
        pass

    @abc.abstractmethod
    def build_batch_preprocessor(self) -> BatchPreprocessor:
        pass

    def get_dataloader(self, dataset: Dataset, identity: str = "Default"):
        batch_preprocessor = self.build_batch_preprocessor()
        reader_options = self.reader_options
        if reader_options is None:
            raise RuntimeError("Reader options must be defined")
        data_reader = make_batch_reader(
            dataset.parquet_url,
            num_epochs=1,
            reader_pool_type=reader_options.petasorm_reader_pool_type
        )
        dataloader = DataLoader(
            data_reader,
            batch_size=reader_options.minibatch_size,
            collate_fn=collate_and_preprocess(batch_preprocessor=batch_preprocessor, use_gpu=False)
        )
        return _closing_iter(dataloader)

    def train_dataloader(self) -> DataLoader:
        self._num_train_data_loader_calls += 1
        return self.get_dataloader(self.train_data, identity=f"train_{self._num_train_data_loader_calls}")

    def test_dataloader(self) -> Optional[DataLoader]:
        self._num_test_data_loader_calls += 1
        return self._get_eval_data(identity=f"test_{self._num_test_data_loader_calls}")

    def val_dataloader(self) -> Optional[DataLoader]:
        self._num_val_data_loader_calls += 1
        return self._get_eval_data(identity=f"test_{self._num_val_data_loader_calls}")

    def _get_eval_data(self, identity: str) -> Optional[DataLoader]:
        eval_data = self.eval_data
        return None if not eval_data else self.get_dataloader(eval_data, identity)


def _closing_iter(dataloader: DataLoader):
    yield from dataloader
    dataloader.__exit__(None, None, None)


def collate_and_preprocess(
    batch_preprocessor: BatchPreprocessor, use_gpu: bool
) -> Callable[[List[Dict]], torch.Tensor]:
    def collate_fn(batch_list: List[Dict]):
        batch = decimal_friendly_collate(batch_list)
        preprocessed_batch: torch.Tensor = batch_preprocessor(batch)
        if use_gpu:
            preprocessed_batch = preprocessed_batch.cuda()
        return preprocessed_batch
    return collate_fn
