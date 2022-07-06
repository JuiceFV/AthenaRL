import inspect
from typing import Any, Generator, Optional, Union

import pytorch_lightning as pl
import torch
from athena.core.dtypes.base import TensorDataClass
from athena.core.logger import LoggerMixin
from pytorch_lightning.loggers.base import DummyExperiment, LoggerCollection
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing_extensions import final
from athena.core.tensorboard import SummaryWriterContext


class AthenaLightening(pl.LightningModule, LoggerMixin):
    def __init__(self, automatic_optimization: bool = True) -> None:
        super().__init__()
        self._automatic_optimization = automatic_optimization
        self._reporter = DummyExperiment()
        self._training_step_gen: Optional[Generator[STEP_OUTPUT, None, None]] = None
        self._verified_steps = False
        self._summary_writer_logger = None
        self._summary_writer = None
        self.register_buffer("_next_stopping_epoch", None)
        self.register_buffer("_cleanly_stopped", None)
        self._next_stopping_epoch = torch.tensor([-1]).int()
        self._cleanly_stopped = torch.ones(1)
        self.batches_processed_this_epoch = 0
        self.all_batches_processed = 0

    def set_reporter(self, reporter: Optional[Any]) -> "AthenaLightening":
        # TODO: reporter should be of some specific base class
        if reporter is None:
            reporter = DummyExperiment()
        self._reporter = reporter
        return self

    @property
    def reporter(self) -> DummyExperiment:
        # TODO: reporter should be of some specific base class
        return self._reporter

    def set_clean_stop(self, clean_stop: bool) -> None:
        self._cleanly_stopped[0] = int(clean_stop)

    def increase_next_stopping_epochs(self, num_epochs: int) -> "AthenaLightening":
        self._next_stopping_epoch += num_epochs
        self.set_clean_stop(False)
        return self

    def train_step_gen(
        self, 
        training_batch: TensorDataClass, 
        batch_idx: int
    ) -> Generator[STEP_OUTPUT, None, None]:
        raise NotImplementedError
    
    @property
    def summary_writer(self):
        if self._summary_writer_logger is self.logger:
            return self._summary_writer
        
        self._summary_writer = None
        self._summary_writer_logger = self.logger
        
        if isinstance(self.logger, LoggerCollection):
            for logger in self.logger:
                if isinstance(logger, TensorBoardLogger):
                    self._summary_writer = logger.experiment
                    break
        elif isinstance(self.logger, TensorBoardLogger):
            self._summary_writer = self.logger.experiment
            
        return self._summary_writer
    
    def training_step(
        self, 
        batch: TensorDataClass, 
        batch_idx: int, 
        optimizer_idx: int = 0
    ) -> STEP_OUTPUT:
        if optimizer_idx >= self._num_opt_steps:
            raise IndexError(f"Index {optimizer_idx} out of bound {self._num_opt_steps}")
        
        if self._training_step_gen is None:
            self._training_step_gen = self.train_step_gen(batch, batch_idx)
            
        output = next(self._training_step_gen)
        
        if optimizer_idx == self._num_opt_steps - 1:
            if not self._verified_steps:
                try:
                    next(self._training_step_gen)
                except StopIteration:
                    self._verified_steps = True
                if not self._verified_steps:
                    raise RuntimeError(
                        f"The number of training steps should match the number " 
                        f"of optimizers {self._num_opt_steps}"
                    )
            self._training_step_gen = None
            SummaryWriterContext.increase_global_step()
        return output
    
    def optimizers(self, use_pl_optimizer: bool = True):
        opt = super().optimizers(use_pl_optimizer)
        return opt if isinstance(opt, list) else [opt]
        
            
    @property
    def _num_opt_steps(self) -> int:
        # TODO: replace with lazy property https://stackoverflow.com/a/6849299/11748947
        return len(self.configure_optimizers())
    
    @final
    def on_epoch_end(self) -> None:
        self.info(
            f"Finished epoch with {self.batches_processed_this_epoch} batches processed"
        )
        self.batches_processed_this_epoch = 0
        self.reporter.flush(self.current_epoch)
        
        if self.current_epoch == self._next_stopping_epoch.item():
            self.trainer.should_stop = True
            
    @final
    def on_train_batch_end(self, *args, **kwargs) -> None:
        self.batches_processed_this_epoch += 1
        self.all_batches_processed += 1
        
    @final
    def on_validation_batch_end(self, *args, **kwargs) -> None:
        self.batches_processed_this_epoch += 1
        
    @final
    def on_test_batch_end(self, *args, **kwargs) -> None:
        self.batches_processed_this_epoch += 1