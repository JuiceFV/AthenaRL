from enum import Enum
from typing import Any, Optional, Union
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
    r"""
    The base data structure represents n-dimensional tensor-based data.
    Generally, we don't need internal :class:`torch.Tensor` implementation 
    to represent tensor-based data, i.e. the explicit interface is enough. 
    If a structure has multiple :class:`torch.Tensor` fields then attribute
    call will be applied to each one.

    Example::

        @dataclass
        class DocSeq(TensorDataClass):
            repr: torch.Tensor
            mask: Optional[torch.Tensor] = None

        docs = DocSeq(torch.Tensor(1, 3), torch.ones(1,3, dtype=torch.bool))
        docs.is_shared() # DocSeq(repr=False, mask=False)
    """

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
            return type(self)(**recursive_call(self.__dict__))
        return tensor_attrs_call

    def cuda(self, *args, **kwargs) -> Union["TensorDataClass", torch.Tensor]:
        """Returns a copy of this object in CUDA memory.

        Args:
            *args: Arguments required by :func:`torch.Tensor.cuda`
            **kwargs: Keyword arguments required by :func:`torch.Tensor.cuda`

        Returns:
            Union["TensorDataClass", torch.Tensor]: Copy of the object
        """
        cuda_tensor = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                kwargs["non_blocking"] = kwargs.get("non_blocking", True)
                cuda_tensor[k] = v.cuda(*args, **kwargs)
            elif isinstance(v, TensorDataClass):
                cuda_tensor[k] = v.cuda(*args, **kwargs)
            else:
                cuda_tensor[k] = v
        return type(self)(**cuda_tensor)

    def cpu(self) -> Union["TensorDataClass", torch.Tensor]:
        r"""
        Returns a copy of this object in CPU memory.

        Returns:
            Union["TensorDataClass", torch.Tensor]: Copy of the object.
        """
        cpu_tensor = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (torch.Tensor, TensorDataClass)):
                cpu_tensor[k] = v.cpu()
            else:
                cpu_tensor[k] = v
        return type(self)(**cpu_tensor)


@dataclass
class Feature(TensorDataClass):
    repr: torch.Tensor


class TransformerConstants(Enum):
    PADDING_SYMBOL: int = 0
    DECODER_START_SYMBOL: int = 1
