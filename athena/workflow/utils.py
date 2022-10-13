import logging
from typing import Union, Optional
import torch
import pytorch_lightning as pl
from athena.core.dtypes import Dataset
from athena.core.dtypes.options import ReaderOptions, ResourceOptions
from athena.core.logger import ManifoldTensorboardLogger
from athena.preprocessing.batch_preprocessor import BatchPreprocessor
from athena.preprocessing.transforms.base import Compose, Transformation
from athena.trainers.athena_lightening import AthenaLightening, StoppingEpochCallback
from athena.data.athena_datamodule import AthenaDataModule
from athena.data.petastorm_datamodule import PetastormLightningDataModule


logger = logging.getLogger(__name__)


def get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def train_eval(
    train_data: Dataset,
    eval_data: Optional[Dataset],
    test_data: Optional[Dataset],
    trainer_module: AthenaLightening,
    data_module: AthenaDataModule,
    nepochs: int,
    logger_name: str,
    transformation: Optional[Union[Compose, Transformation]] = None,
    batch_preprocessor: Optional[BatchPreprocessor] = None,
    reader_options: Optional[ReaderOptions] = None,
    checkpoint_path: Optional[str] = None,
    resource_options: Optional[ResourceOptions] = None
) -> pl.Trainer:
    resource_options = resource_options or ResourceOptions()
    reader_options = reader_options or ReaderOptions()
    datamodule = data_module or PetastormLightningDataModule(
        train_data, eval_data, test_data, transformation, batch_preprocessor, reader_options
    )
    trainer = pl.Trainer(
        logger=ManifoldTensorboardLogger(save_dir="log_tensorboard", name=logger_name),
        max_epochs=nepochs * 1000,
        gpus=resource_options.use_gpu,
        reload_dataloaders_every_n_epochs=1,
        resume_from_checkpoint=checkpoint_path,
        callbacks=[StoppingEpochCallback(nepochs)]
    )
    trainer.fit(trainer_module, datamodule=datamodule)

    if type(trainer_module).test_step != pl.LightningModule.test_step:
        trainer.test(ckpt_path=None, datamodule=datamodule)
    else:
        logger.warning(f"Module {type(trainer_module).__name__} doesn't implement test_step().")

    if checkpoint_path is not None:
        trainer_module.set_clean_stop(True)
        trainer.save_checkpoint(checkpoint_path)

    return trainer
