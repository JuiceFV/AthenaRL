import dataclasses
import logging
import time
from typing import Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from athena.core.dtypes import AthenaOptions, TableSpec, TrainingOutput
from athena.core.dtypes.dataset import Dataset
from athena.core.dtypes.options import (ReaderOptions, ResourceOptions,
                                        RLOptions)
from athena.core.parameters import NormalizationData
from athena.core.tensorboard import summary_writer_context
from athena.data.data_extractor import DataExtractor
from athena.data.fap.fapper import FAPper
from athena.data.fap.roster import FAPperRoster
from athena.data.manual_datamodule import get_sample_range
from athena.model_managers.manager import ModelManager
from athena.model_managers.roster import ModelManagerRoster
from athena.publisher import ModelPublisherRoster
from athena.publisher.publisher_base import ModelPublisher
from athena.validators import ModelValidatorRoster
from athena.validators.validator_base import ModelValidator

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

    rl_options = options.rl_options or RLOptions()
    reader_options = options.reader_options or ReaderOptions()

    manager: ModelManager = model.value

    normalization_dict: Optional[Dict[int, NormalizationData]] = None
    setup_data: Optional[Dict[str, bytes]] = None

    data_module = manager.get_data_module(
        fapper=fapper.value,
        input_table_spec=input_table_spec,
        rl_options=rl_options,
        reader_options=reader_options,
        resource_options=resource_options
    )
    if data_module is not None:
        setup_data = data_module.prepare_data()
    else:
        normalization_dict = manager.run_feature_identification(input_table_spec, fapper.value)

    return query_and_train(
        input_table_spec,
        model,
        nepochs,
        use_gpu=use_gpu,
        fapper=fapper,
        setup_data=setup_data,
        normalization_dict=normalization_dict,
        rl_options=rl_options,
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
    rl_options: Optional[RLOptions] = None,
    reader_options: Optional[ReaderOptions] = None,
    resource_options: Optional[ResourceOptions] = None,
    warmstart_path: Optional[str] = None,
    validator: Optional[ModelValidatorRoster] = None,
    publisher: Optional[ModelPublisherRoster] = None
) -> TrainingOutput:
    logger.info("Query ...")

    rl_options = rl_options or RLOptions()
    reader_options = reader_options or ReaderOptions()
    resource_options = resource_options or ResourceOptions()
    manager: ModelManager = model.value

    resource_options.gpu = int(use_gpu)

    if setup_data is None:
        data_module = manager.get_data_module(
            fapper=fapper.value,
            input_table_spec=input_table_spec,
            rl_options=rl_options,
            reader_options=reader_options,
            resource_options=resource_options,
            saved_setup_data=saved_setup_data
        )
        if data_module is not None:
            setup_data = data_module.prepare_data()
            normalization_dict = None

    if all([setup_data is not None, normalization_dict is not None]):
        raise ValueError("setup_data and normalization_dict are mutually exclusive")

    train_data = None
    eval_data = None
    data_extractor = DataExtractor(fapper=fapper.value)
    if normalization_dict is not None:
        evaluate = manager.should_generate_eval_data
        sample_range = get_sample_range(input_table_spec, evaluate)
        train_data = manager.query_data(
            input_table_spec=input_table_spec,
            sample_range=sample_range.train_sample_range,
            rl_options=rl_options,
            data_extractor=data_extractor
        )
        eval_data = None
        if evaluate:
            eval_data = manager.query_data(
                input_table_spec=input_table_spec,
                sample_range=sample_range.eval_sample_range,
                rl_options=rl_options,
                data_extractor=data_extractor
            )

    logger.info("Training ...")

    results = train(
        manager,
        train_data,
        eval_data,
        fapper.value,
        nepochs=nepochs,
        use_gpu=use_gpu,
        input_table_spec=input_table_spec,
        setup_data=setup_data,
        normalization_dict=normalization_dict,
        rl_options=rl_options,
        reader_options=reader_options,
        resource_options=resource_options,
        warmstart_path=warmstart_path
    )

    if validator is not None:
        results = run_validator(validator, results)

    if publisher is not None:
        results = run_publisher(publisher, model, results)
    return results


def train(
    model_manager: ModelManager,
    train_data: Optional[Dataset],
    eval_data: Optional[Dataset],
    fapper: FAPper,
    *,
    nepochs: int,
    use_gpu: bool,
    input_table_spec: Optional[TableSpec] = None,
    setup_data: Optional[Dict[str, bytes]] = None,
    normalization_dict: Optional[Dict[str, NormalizationData]] = None,
    rl_options: Optional[RLOptions] = None,
    reader_options: Optional[ReaderOptions] = None,
    resource_options: Optional[ResourceOptions] = None,
    warmstart_path: Optional[str] = None
) -> TrainingOutput:
    writer = SummaryWriter()
    logger.info(f"TensorBoard logging loaction is: {writer.log_dir}")

    if setup_data is not None:
        data_module = model_manager.get_data_module(
            fapper=fapper,
            input_table_spec=input_table_spec,
            setup_data=setup_data,
            rl_options=rl_options,
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

    warmstart_input_path = warmstart_path or None
    trainer_module = model_manager.build_trainer(
        use_gpu=use_gpu,
        rl_options=rl_options,
        normalization_dict=normalization_dict
    )

    if not reader_options:
        reader_options = ReaderOptions()

    if not resource_options:
        resource_options = ResourceOptions()

    with summary_writer_context(writer):
        training_output, _ = model_manager.train(
            trainer_module=trainer_module,
            train_data=train_data,
            eval_data=eval_data,
            test_data=None,
            data_module=data_module,
            nepochs=nepochs,
            reader_options=reader_options,
            resource_options=resource_options,
            checkpoint_path=warmstart_input_path
        )

    output_paths = {}
    for module_name, serving_module in model_manager.build_serving_modules(
        trainer_module=trainer_module, normalization_dict=normalization_dict
    ).items():
        torchscript_output_path = f"{model_manager.__class__.__name__}_{module_name}_{round(time.time())}.torchscript"
        torch.jit.save(serving_module, torchscript_output_path)
        logger.info(f"Saed {module_name} to {torchscript_output_path}")
        output_paths[module_name] = torchscript_output_path
    return dataclasses.replace(training_output, output_paths=output_paths)


def run_validator(
    validator: ModelValidatorRoster, training_output: TrainingOutput
) -> TrainingOutput:
    model_validator: ModelValidator = validator.value
    validation_result = model_validator.validate(training_output)
    return dataclasses.replace(training_output, validation_result=validation_result)


def run_publisher(
    publisher: ModelPublisherRoster,
    model: ModelManagerRoster,
    training_output: TrainingOutput,
) -> TrainingOutput:
    model_publisher: ModelPublisher = publisher.value
    model_manager: ModelManager = model.value
    publishing_result = model_publisher.publish(training_output, model_manager)
    return dataclasses.replace(training_output, publishing_result)
