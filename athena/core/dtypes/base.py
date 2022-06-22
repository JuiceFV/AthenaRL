from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass

import torch
from athena.core.base_dclass import BaseDataClass
from athena.core.logger import LoggerMixin


@dataclass
class Dataset:
    json_url: str


@dataclass
class TableSpec:
    table_name: str
    table_sample: Optional[float] = None
    val_table_sample: Optional[float] = None
    test_table_sample: Optional[float] = None


@dataclass
class ReaderOptions:
    minibatch_size: int = 1024
    reader_pool_type: str = "thread"


@dataclass
class TensorDataClass(BaseDataClass, LoggerMixin):
    def __getattr__(self, __name: str):
        if __name.startswith("__") and __name.endswith("__"):
            raise AttributeError(
                "We don't wanna call superprivate method of torch.Tensor"
            )
        tensor_attr = getattr(torch.Tensor, __name, None)

        if tensor_attr is None or not callable(tensor_attr):
            self.error(
                f"Attempting to call {self.__class__.__name__}.{__name} on "
                f"{type(self)} (instance of TensorDataClass)."
            )
            if tensor_attr is None:
                raise AttributeError(
                    f"{self.__class__.__name__} doesn't have {__name} attribute."
                )
            else:
                raise RuntimeError(
                    f"{self.__class__.__name__}.{__name} is not callable."
                )

        def tensor_attrs_call(*args, **kwargs):
            """The TensorDataClass is the base one, thus we wanna get 
            attribute (when we call `__getattr__`) at every single 
            child's `Callable` attribute where it possible (if 
            child's attribute has torch.Tensor instance).
            """
            def recursive_call(obj: Any):
                if isinstance(obj, (torch.Tensor, TensorDataClass)):
                    return getattr(obj, __name)(*args, **kwargs)
                if isinstance(obj, dict):
                    return {key: recursive_call(value) for key, value in obj.items()}
                if isinstance(obj, tuple):
                    return tuple(recursive_call(value) for value in obj)
                return obj
            return type(self)(*recursive_call(**self.__dict__))
        return tensor_attrs_call


@dataclass
class Feature(TensorDataClass):
    repr: torch.Tensor


class TransformerConstants(Enum):
    PADDING_SYMBOL: int = 0
    DECODER_START_SYMBOL: int = 1