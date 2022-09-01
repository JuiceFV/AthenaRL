import torch
import logging
from typing import Dict, Optional

from athena.core.dtypes import TrainingOutput
from athena.core.dtypes.dataset import Dataset
from athena.core.dtypes.options import MLOptionsRoster, ReaderOptions, ResourceOptions
from athena.core.parameters import NormalizationData
from athena.data.data_extractor import DataExtractor
from athena.data.fap.roster import FAPperRoster
from athena.data.manual_datamodule import get_sample_range
from athena.model_managers.manager import ModelManager
from athena.model_managers.roster import ModelManagerRoster
from athena.core.dtypes import TableSpec, AthenaOptions
from athena.validators import ModelValidatorRoster
from athena.publisher import ModelPublisherRoster
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)


def build_and_train(
    input_table_spec: TableSpec,
    model: ModelManagerRoster,
    nepochs: int,
    options: AthenaOptions,
    fapper: FAPperRoster,
    warmstart_path: Optional[str] = None,
    validator: Optional[ModelValidatorRoster] = None,
    publisher: Optional[ModelPublisherRoster] = None
) -> TrainingOutput:
    resource_options = options.resource_options or ResourceOptions()
    use_gpu = torch.cuda.is_available() if resource_options.use_gpu else False

    if not use_gpu:
        logger.info("GPU is not in use")

    ml_options = options.ml_options or MLOptionsRoster()
    reader_options = options.reader_options or ReaderOptions()

    manager: ModelManager = model.value

    normalization_dict: Optional[Dict[int, NormalizationData]] = None
    setup_data: Optional[Dict[str, bytes]] = None

    data_module = manager.get_data_module(
        input_table_spec=input_table_spec,
        ml_options=ml_options,
        reader_options=reader_options,
        resource_options=resource_options
    )
    if data_module is not None:
        setup_data = data_module.prepare_data(fapper=fapper.value)
    else:
        normalization_dict = manager.run_feature_identification(input_table_spec)

    return query_and_train(
        input_table_spec,
        model,
        nepochs,
        use_gpu=use_gpu,
        fapper=fapper,
        setup_data=setup_data,
        normalization_dict=normalization_dict,
        ml_options=ml_options,
        reader_options=reader_options,
        resource_options=resource_options,
        warmstart_path=warmstart_path,
        validator=validator,
        publisher=publisher
    )


def query_and_train(
    input_table_spec: TableSpec,
    model: ModelManagerRoster,
    nepochs: int,
    use_gpu: bool,
    fapper: FAPperRoster,
    *,
    setup_data: Optional[Dict[str, bytes]] = None,
    saved_setup_data: Optional[Dict[str, bytes]] = None,
    normalization_dict: Optional[Dict[str, NormalizationData]] = None,
    ml_options: Optional[MLOptionsRoster] = None,
    reader_options: Optional[ReaderOptions] = None,
    resource_options: Optional[ResourceOptions] = None,
    warmstart_path: Optional[str] = None,
    validator: Optional[ModelValidatorRoster] = None,
    publisher: Optional[ModelPublisherRoster] = None
) -> TrainingOutput:
    logger.info("Query ...")

    ml_options = ml_options or MLOptionsRoster()
    reader_options = reader_options or ReaderOptions()
    resource_options = resource_options or ResourceOptions()
    manager: ModelManager = model.value

    resource_options.gpu = int(use_gpu)

    if setup_data is None:
        data_module = manager.get_data_module(
            input_table_spec=input_table_spec,
            ml_options=ml_options,
            reader_options=reader_options,
            resource_options=resource_options,
            saved_setup_data=saved_setup_data
        )
        if data_module is not None:
            setup_data = data_module.prepare_data(fapper=fapper.value)
            normalization_dict = None

    if all([setup_data is not None, normalization_dict is not None]):
        raise ValueError("setup_data and normalization_dict are mutually exclusive")

    train_data = None
    eval_data = None
    data_extractor = DataExtractor(fapper=fapper.value)
    if normalization_dict is not None:
        evaluate = manager.should_generate_eval_dataset
        sample_range = get_sample_range(input_table_spec, evaluate)
        train_data = manager.query_data(
            input_table_spec=input_table_spec,
            sample_range=sample_range.train_sample_range,
            ml_options=ml_options,
            data_extractor=data_extractor
        )
        eval_data = None
        if evaluate:
            eval_data = manager.query_data(
                input_table_spec=input_table_spec,
                sample_range=sample_range.eval_sample_range,
                ml_options=ml_options,
                data_extractor=data_extractor
            )

    logger.info("Training ...")


def train(
    model_manager: ModelManager,
    train_data: Optional[Dataset],
    eval_data: Optional[Dataset],
    *,
    nepochs: int,
    use_gpu: bool,
    setup_data: Optional[Dict[str, bytes]] = None,
    normalization_dict: Optional[Dict[str, NormalizationData]] = None,
    ml_options: Optional[MLOptionsRoster] = None,
    reader_options: Optional[ReaderOptions] = None,
    resource_options: Optional[ResourceOptions] = None,
    wamstart_path: Optional[str] = None
) -> TrainingOutput:
    writer = SummaryWriter()
    logger.info(f"TensorBoard logging loaction is: {writer.log_dir}")

    if setup_data is not None:
        data_module = model_manager.get_data_module(
            setup_data=setup_data,
            ml_options=ml_options,
            reader_options=reader_options,
            resource_options=resource_options
        )
        if data_module is None:
            raise RuntimeError(f"Implement `get_data_module()` for the {model_manager} first.")
        data_module.setup()
    else:
        data_module = None

    if normalization_dict is None:
        if data_module is None:
            raise RuntimeError(
                "Missing any data, implement either `get_data_module()` "
                "or `run_feature_identification()`"
            )
        normalization_dict = data_module.get_normalization_dict()

    warmstart_input_path = wamstart_path or None
    trainer_module = model_manager.build_trainer(
        use_gpu=use_gpu,
        ml_options=ml_options,
        normalization_dict=normalization_dict
    )
