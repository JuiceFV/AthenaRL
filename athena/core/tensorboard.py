import contextlib
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from athena.core.logger import LoggerMixin


class SummaryWriterContextMeta(type, LoggerMixin):
    def __getattr__(cls, func: str):
        if func.startswith("__"):
            return super().__getattr__(func)

        if not cls._writer_stacks:

            def _pass(*args, **kwargs):
                return

            return _pass

        writer = cls._writer_stacks[-1]

        def call(*args, **kwargs) -> Callable[..., Optional[Any]]:
            if "global_step" not in kwargs:
                kwargs["global_step"] = cls._global_step
            try:
                return getattr(writer, func)(*args, **kwargs)
            except Exception as ex:
                if hasattr(writer, "ignoring_exeptions") and isinstance(ex, writer.ignoring_exeptions):
                    cls.warning(f"Exception {ex} is ignored.")
                    if hasattr(writer, "ignored_exceptions_handler"):
                        writer.ignored_exceptions_handler(ex)
                    return None
                raise
        return call


class SummaryWriterContext(metaclass=SummaryWriterContextMeta):
    _writer_stacks: List[SummaryWriter] = []
    _global_step = 0
    _custom_scalars: Dict[str, Any] = {}

    @classmethod
    def _reset_globals(cls) -> None:
        cls._global_step = 0
        cls._custom_scalars.clear()

    @classmethod
    def increase_global_step(cls) -> None:
        cls._global_step += 1

    @classmethod
    def add_histogram(cls, tag: str, values: Union[torch.Tensor, np.ndarray], *args, **kwargs) -> None:
        try:
            return cls.__getattr__("add_histogram")(tag, values, *args, **kwargs)
        except ValueError:
            cls.warning(f"Cannot create histogram for {tag}, got values: {values}")

    @classmethod
    def add_custom_scalars(cls, writer: SummaryWriter) -> None:
        writer.add_custom_scalars(cls._custom_scalars)

    @classmethod
    def add_custom_scalars_multichart(cls, tags: Any, category: str, title: str) -> None:
        if category not in cls._custom_scalars:
            cls._custom_scalars[category] = {}
        if title in cls._custom_scalars[category]:
            raise ValueError(f"Title ({title}) is already in category ({category})")
        cls._custom_scalars[category][title] = ["Multiline", tags]

    @classmethod
    def push(cls, writer: SummaryWriter) -> None:
        if not isinstance(writer, SummaryWriter):
            raise TypeError(f"Writer is not a SummaryWriter: {writer}")
        cls._writer_stacks.append(writer)

    @classmethod
    def pop(cls) -> SummaryWriter:
        return cls._writer_stacks.pop()


@contextlib.contextmanager
def summary_writer_context(writer: SummaryWriter):
    if writer is not None:
        SummaryWriterContext.push(writer)
    try:
        yield
    finally:
        if writer is not None:
            SummaryWriterContext.pop()
