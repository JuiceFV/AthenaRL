import abc
from typing import Any, Dict, Final, List, NamedTuple, Tuple

import numpy as np
import torch
from source.core.dataclasses import dataclass
from source.core.logger import LoggerMixin

STATIC_KEYS = ["action", "state", "score"]


@dataclass
class ElementMeta:
    @abc.abstractclassmethod
    def try_create_from_value(cls, name: str, value: Any):
        raise NotImplementedError()

    @abc.abstractmethod
    def validate(self, name: str, value: Any):
        raise NotImplementedError()

    @abc.abstractmethod
    def initialize_storage(self, capacity: int):
        raise NotImplementedError()


@dataclass
class DenseMeta(ElementMeta):
    shape: Tuple[int, ...]
    dtype: np.dtype

    @classmethod
    def try_create_from_value(cls, name: str, value: Any) -> ElementMeta:
        arr_value = np.array(value)
        dtype = arr_value.dtype
        if dtype == np.dtype("float32"):
            dtype = np.dtype("float64")
        obj = cls(arr_value.shape, dtype)
        obj.validate(name, value)
        return obj

    def validate(self, name: str, value: Any) -> None:
        if isinstance(value, (dict, torch.Tensor)):
            raise TypeError(
                f"{name} type ({type(value)}) shouldn't instatiate dict or Tensor."
            )
        arr_value = np.array(value)
        dtype = arr_value.dtype
        if dtype == np.dtype("float32"):
            dtype = np.dtype("float64")
        if (arr_value.shape != self.shape) or (arr_value.dtype != self.dtype):
            raise ValueError(
                f"For the {name} expected {self.shape} {self.dtype}, "
                f"but got {arr_value.shape} {arr_value.dtype}."
            )

    def initialize_storage(self, capacity: int) -> torch.Tensor:
        shape = [capacity, *self.shape]
        if self.dtype == bool:
            return torch.from_numpy(np.zeros(shape, dtype=torch.bool))
        return torch.from_numpy(np.empty(shape, dtype=self.dtype))


class BufferElement(NamedTuple):
    name: str
    meta: ElementMeta


class EpisodesBuffer(LoggerMixin):
    def __init__(
        self,
        sim_capacity: int = 10000,
        batch_size: int = 1
    ) -> None:
        self._initialized: bool = False
        self._batch_size: int = batch_size
        self._sim_capacity: Final[int] = sim_capacity

        self.add_calls = np.array(0)

        self._storage: Dict[str, torch.Tensor] = {}
        self._storage_signature: List[BufferElement] = []

        self._additional_keys: List[str] = []

    def _create_buffer_element(self, name: str, value: Any) -> BufferElement:
        if isinstance(value, torch.Tensor):
            raise TypeError("Input shouldn't be tensor")
        meta = None
        for meta_cls in [DenseMeta]:
            try:
                meta = meta_cls.try_create_from_value(name, value)
            except Exception as ex:
                self.info(
                    f"Unable to create {meta_cls} for the {name} ({value}): {ex}"
                )
        if meta is None:
            raise ValueError(
                f"Unable to deduce simulation type for {name}: {value}"
            )
        return BufferElement(name, meta)

    def _initialize_storage(self):
        for element in self.get_storage_signature():
            self._storage[element.name] = element.meta.initialize_storage(
                self._sim_capacity
            )

    def _add(self, **kwargs):
        pass

    def initialize_buffer(self, **kwargs):
        kwargs_keys = set(kwargs.keys())
        if not set(STATIC_KEYS).issubset(kwargs_keys):
            raise ValueError(
                f"{kwargs_keys} doesn't contain all of {STATIC_KEYS}"
            )
        self._additional_keys = list(kwargs_keys - set(STATIC_KEYS))

        for key in STATIC_KEYS + self._additional_keys:
            self._storage_signature.append(
                self._create_buffer_element(key, kwargs[key])
            )
        self._initialize_storage()
        self._initialized = True

        self.info(f"Initialized {self.__class__.__name__}")
        self.info(f"\t Buffer capacity is {self._sim_capacity}")
        self.info(f"\t Storage inner types are:")
        for element in self.get_storage_signature():
            self.info(f"\t\t {element}")

    def get_storage_signature(self) -> List[BufferElement]:
        return self._storage_signature

    def add(self, **kwargs):
        if not self._initialized:
            self.initialize_buffer(**kwargs)

    @property
    def size(self) -> int:
        pass

    @property
    def capacity(self) -> int:
        return self._sim_capacity

    def iter(self) -> int:
        return self.add_calls % self._sim_capacity
