import abc
from collections import namedtuple
from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np
import torch
from source.core.dataclasses import dataclass
from source.core.logger import LoggerMixin

try:
    from typing import Final
except ImportError:
    from typing_extensions import Final

STATIC_KEYS = ["action", "state", "score", "is_last"]


@dataclass
class ElementMeta:
    @classmethod
    @abc.abstractmethod
    def try_create_from_value(cls, name: str, value: Any):
        raise NotImplementedError()

    @abc.abstractmethod
    def validate(self, name: str, value: Any) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def initialize_storage(self, capacity: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def turn_into_storage_element(self, value: Any) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def empty_observation(self) -> Any:
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_to_output(self, sample: torch.Tensor) -> torch.Tensor:
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
            return torch.zeros(shape, dtype=torch.bool)
        return torch.from_numpy(np.empty(shape, dtype=self.dtype))

    def turn_into_storage_element(self, value: Any) -> torch.Tensor:
        return torch.from_numpy(np.array(value, dtype=self.dtype))

    def empty_observation(self) -> Any:
        return np.zeros(self.shape, dtype=self.dtype)

    def sample_to_output(self, sample: torch.Tensor) -> torch.Tensor:
        res: torch.Tensor = torch.einsum("ij...->i...j", sample)
        return res.squeeze(-1) if res.shape[-1] == 1 else res


class BufferElement(NamedTuple):
    name: str
    meta: ElementMeta


class EpisodesBuffer(LoggerMixin):
    def __init__(
        self,
        capacity: int = 10000,
        batch_size: int = 1,
        episode_capacity: int = 1,
        stack_size: int = 1
    ) -> None:
        self._initialized: bool = False
        self._batch_size: int = batch_size
        self._capacity: Final[int] = capacity
        self._episode_capacity: Final[int] = episode_capacity
        self._stack_size: Final[int] = stack_size

        self.add_calls = np.array(0)

        self._is_index_valid = torch.zeros(self._capacity, dtype=torch.bool)
        self._num_valid_indices: int = 0
        self._episodic_num_observation: int = 0

        self._storage: Dict[str, torch.Tensor] = {}
        self._storage_signature: List[BufferElement] = []
        self._batch_struct = namedtuple("filler", [])

        self._additional_keys: List[str] = []
        self._key_to_buffer_element: Dict[str, BufferElement] = {}
        self._empty_observation: Dict[str, Any] = {}
        self._observation_fields: List[str] = []

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
                f"Unable to deduce inner buffer type for {name}: {value}"
            )
        return BufferElement(name, meta)

    def _initialize_storage(self) -> None:
        for element in self.get_storage_signature():
            self._storage[element.name] = element.meta.initialize_storage(
                self._capacity
            )

    def _check_addable_quantity(self, **kwargs) -> None:
        if len(kwargs) != len(self.get_storage_signature()):
            raise ValueError(
                f"Expected: {self.get_storage_signature()}; received {kwargs}"
            )

    def _check_addable_signature(self, **kwargs) -> None:
        self._check_addable_quantity(**kwargs)
        for storage_element in self.get_storage_signature():
            storage_element.meta.validate(
                storage_element.name,
                kwargs[storage_element.name]
            )

    def _add(self, **kwargs) -> None:
        self._check_addable_quantity(**kwargs)
        for element in self.get_storage_signature():
            kwargs[element.name] = element.meta.turn_into_storage_element(
                kwargs[element.name]
            )
        self._add_observation(kwargs)

    def _add_observation(self, observation: Dict[str, torch.Tensor]) -> None:
        pos = self.iter()
        for name, value in observation.items():
            self._storage[name][pos] = value
        self.add_calls += 1

    @abc.abstractmethod
    def set_index_validity(self, index: int, is_valid: bool) -> None:
        raise NotImplementedError()

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
        for el in self.get_storage_signature():
            self._key_to_buffer_element[el.name] = el
            self._empty_observation[el.name] = el.meta.empty_observation()

        self._initialize_storage()
        self._observation_fields = self.get_observation_fields()
        self._batch_struct = namedtuple(
            "butch_struct", self._observation_fields
        )
        self._initialized = True

        self.info(f"Initialized {self.__class__.__name__}")
        self.info(f"\t Buffer capacity is {self._capacity}")
        self.info(f"\t Episdoe capacity is {self._episode_capacity}")
        self.info(f"\t Observation stack size is {self._stack_size}")
        self.info(f"\t Storage inner types are:")
        for element in self.get_storage_signature():
            self.info(f"\t\t {element}")

    def get_observation_fields(self) -> List[str]:
        return [
            "indices",
            "step",
            *STATIC_KEYS
        ] + self._additional_keys

    def get_storage_signature(self) -> List[BufferElement]:
        return self._storage_signature

    @abc.abstractmethod
    def add(self, **kwargs) -> None:
        raise NotImplementedError()

    @property
    def size(self) -> int:
        return self._num_valid_indices

    @property
    def episode_capacity(self) -> int:
        return self._episode_capacity

    @property
    def capacity(self) -> int:
        return self._capacity

    @abc.abstractmethod
    def iter(self) -> int:
        raise NotImplementedError()

    def is_empty(self) -> bool:
        return self.add_calls == 0

    def is_full(self) -> bool:
        return self.add_calls >= self._capacity

    def is_valid_observation(self, index: int) -> bool:
        return self._is_index_valid[index]

    def sample_index_batch(self, batch_size: int) -> torch.Tensor:
        if self._num_valid_indices == 0:
            raise RuntimeError(
                f"Cannot sample {batch_size} since there are no valid indices so far."
            )
        valid_idcs = self._is_index_valid.nonzero().squeeze(1)
        return valid_idcs[torch.randint(valid_idcs.shape[0], (batch_size,))]

    def sample_observation_batch(self, batch_size: int = None, indices: torch.Tensor = None):
        if batch_size is None:
            batch_size = self._batch_size
        if indices is None:
            indices = self.sample_index_batch(batch_size)
        else:
            if not isinstance(indices, torch.Tensor):
                raise TypeError(
                    f"Indices {indices} have type {type(indices)} but should be Tensor"
                )
            indices = indices.type(dtype=torch.int64)
        if len(indices) != batch_size:
            raise RuntimeError(
                f"Indices len {len(indices)} and batch size deffer but shouldn't"
            )
        forward_indicies = torch.arange(self._episode_capacity)
        episode_step_idcs = indices.unsqueeze(1) + forward_indicies
        episode_step_idcs %= self._capacity

        steps = self._get_steps(episode_step_idcs)
        margin_indices = (indices + steps - 1) % self._capacity
        margin_indices = margin_indices.unique()
        batch_arrays = []
        for field in self._observation_fields:
            if field == "indices":
                batch = indices
            elif field == "step":
                batch = steps
            elif field in self._storage:
                batch = self._get_batch_for_indices(field, margin_indices)
            else:
                batch = None

            if isinstance(batch, torch.Tensor) and batch.ndim == 1:
                batch = batch.unsqueeze(1)
            batch_arrays.append(batch)
        return self._batch_struct(*batch_arrays)

    def _get_steps(self, dense_indices: torch.Tensor) -> torch.Tensor:
        is_lasts = self._storage["is_last"][dense_indices].to(torch.bool)
        is_lasts[:, -1] = True
        is_lasts = is_lasts.float()
        position_mask = torch.arange(is_lasts.shape[1] + 1, 1, -1)
        is_lasts = torch.einsum("ij,j->ij", is_lasts, position_mask)
        return torch.argmax(is_lasts, dim=1) + 1

    def _get_batch_for_indices(self, key: str, indices: torch.Tensor):
        if len(indices.shape) != 1:
            raise ValueError(
                f"The indices tensor is not 1-dimensional"
            )
        return self._get_stack_for_indices(key, indices)

    def _get_stack_for_indices(self, key: str, indices: torch.Tensor):
        if len(indices.shape) != 1:
            raise ValueError(
                f"The indices tensor is not 1-dimensional"
            )
        backward_indices = torch.arange(-self._stack_size + 1, 1)
        stack_indices = indices.unsqueeze(1) + backward_indices
        stack_indices %= self._capacity
        values_stack = self._storage[key][stack_indices]
        return self._key_to_buffer_element[key].meta.sample_to_output(values_stack)
