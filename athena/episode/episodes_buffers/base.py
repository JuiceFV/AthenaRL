"""This is implementation base of out-of-graph buffer.
It also could be wrapped with in-graph wrapper.
"""
import abc
from collections import namedtuple
from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np
import torch
from athena.core.dataclasses import dataclass
from athena.core.logger import LoggerMixin

try:
    from typing import Final
except ImportError:
    from typing_extensions import Final

STATIC_KEYS = ["action", "state", "score", "is_last"]


@dataclass
class ElementMeta:
    """Base class for elements buffer consists from.
    """
    @classmethod
    @abc.abstractmethod
    def try_create_from_value(cls, name: str, value: Any):
        """Try to create an instance of a given class 
        to a specific `value` for `name` key. It's good
        practice to call self.validate after meta initialization.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def validate(self, name: str, value: Any) -> None:
        """Look for some inapropriate data/metadata in given `value`.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def initialize_storage(self, capacity: int) -> torch.Tensor:
        """Initialize the buffer with given `capacity`, for this data type.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def turn_into_storage_element(self, value: Any) -> torch.Tensor:
        """Convert given `value` to the correct reppresentation acceptable by buffer.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def empty_observation(self) -> Any:
        """How does empty observation look like? For padding especially. 
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_to_output(self, sample: torch.Tensor) -> torch.Tensor:
        """Adjust inner buffer representation to the `output`.
        """
        raise NotImplementedError()


@dataclass
class DenseMeta(ElementMeta):
    """Inner representation of Tensor.
    """
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
        # bool unable to handle all bit reprs
        if self.dtype == bool:
            return torch.zeros(shape, dtype=torch.bool)
        return torch.from_numpy(np.empty(shape, dtype=self.dtype))

    def turn_into_storage_element(self, value: Any) -> torch.Tensor:
        return torch.from_numpy(np.array(value, dtype=self.dtype))

    def empty_observation(self) -> Any:
        return np.zeros(self.shape, dtype=self.dtype)

    def sample_to_output(self, sample: torch.Tensor) -> torch.Tensor:
        # Permute so that torch.Size([batch_size, stack_size, observ_size])
        # becomes torch.Size([batch_size, observ_size, stack_size])
        res: torch.Tensor = torch.einsum("ij...->i...j", sample)
        # if stack_size isn't 1-D squeeze it
        return res.squeeze(-1) if res.shape[-1] == 1 else res


class BufferElement(NamedTuple):
    """Representation of buffer element content.
    """
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
        """Buffer initialization.

        Args:
            capacity (int, optional): Buffer capacity. Defaults to 10000.
            batch_size (int, optional): Batch size. Defaults to 1.
            episode_capacity (int, optional): The `n` in (n-szie sequence). Defaults to 1.
            stack_size (int, optional): The sample size when batching. Defaults to 1.
        """
        self._initialized: bool = False
        self._batch_size: int = batch_size
        self._capacity: Final[int] = capacity
        self._episode_capacity: Final[int] = episode_capacity
        self._stack_size: Final[int] = stack_size

        self.add_calls = np.array(0)  # The only public attr

        # Due to possible paddings and cyclic traits not all
        # indices are valid. Thus the indexing differs depends on
        # buffer type
        self._is_index_valid = torch.zeros(self._capacity, dtype=torch.bool)
        self._num_valid_indices: int = 0
        self._episodic_num_observation: int = 0

        # Everything requered for storage and batching
        self._storage: Dict[str, torch.Tensor] = {}
        self._storage_signature: List[BufferElement] = []
        self._batch_struct = namedtuple("filler", [])

        # Auxiliary attrs
        self._additional_keys: List[str] = []
        self._key_to_buffer_element: Dict[str, BufferElement] = {}
        self._empty_observation: Dict[str, Any] = {}
        self._observation_fields: List[str] = []

    def _create_buffer_element(self, name: str, value: Any) -> BufferElement:
        """Trying to create inner buffer representation for the given element
        by browsing all possible buffer types.

        Args:
            name (str): buffer field name.
            value (Any): Value to convert.

        Raises:
            TypeError: In case the value is instantiated from torch.Tensor
            ValueError: If there is no suitable inner buffer type to 
                represent the value.

        Returns:
            BufferElement: Inner buffer representation of the value.
        """
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
        """Private method which initialize the storage with
        empty values.
        """
        for element in self.get_storage_signature():
            self._storage[element.name] = element.meta.initialize_storage(
                self._capacity
            )

    def _check_addable_quantity(self, **kwargs) -> None:
        """Are the addable args enough and not overmuch?

        Raises:
            ValueError: If at least one condition is not met.
        """
        if len(kwargs) != len(self.get_storage_signature()):
            raise ValueError(
                f"Expected: {self.get_storage_signature()}; received {kwargs}"
            )

    def _check_addable_signature(self, **kwargs) -> None:
        """Check if at least one argument can't be converted 
        to the same inner type as all field's entities.
        """
        self._check_addable_quantity(**kwargs)
        for storage_element in self.get_storage_signature():
            storage_element.meta.validate(
                storage_element.name,
                kwargs[storage_element.name]
            )

    def _add(self, **kwargs) -> None:
        """Inner add method which turns arg value into inner buffer representation.
        """
        self._check_addable_quantity(**kwargs)
        for element in self.get_storage_signature():
            kwargs[element.name] = element.meta.turn_into_storage_element(
                kwargs[element.name]
            )
        self._add_observation(kwargs)

    def _add_observation(self, observation: Dict[str, torch.Tensor]) -> None:
        """Add observation to the buffer.

        Args:
            observation (Dict[str, torch.Tensor]): Observation in inner buffer format.
        """
        pos = self.iter()
        for name, value in observation.items():
            self._storage[name][pos] = value
        self.add_calls += 1

    def _get_steps(self, dense_indices: torch.Tensor) -> torch.Tensor:
        """Get number of steps required to reach the last record in
        episode or last record in the given sequence.

        Args:
            dense_indices (torch.Tensor): Random indicies for sampling.

        Returns:
            torch.Tensor: Steps for indicies
        """
        is_lasts = self._storage["is_last"][dense_indices].to(torch.bool)
        # if there is no last record in the sequence mark the episode_capacity
        is_lasts[:, -1] = True
        is_lasts = is_lasts.float()
        # Create positional mask in purpose to get first encountered
        # `is_last` via torch.argmax
        position_mask = torch.arange(is_lasts.shape[1] + 1, 1, -1)
        # Matrix (A_ij) times vector (v) where new matrix element
        # N_ij = A_ij x v_j. I.e. each new matrix row contains
        # element-wise product of old matrix row and vector
        is_lasts = torch.einsum("ij,j->ij", is_lasts, position_mask)
        return torch.argmax(is_lasts, dim=1) + 1

    def _get_batch_for_indices(self, key: str, indices: torch.Tensor):
        if len(indices.shape) != 1:
            raise ValueError(
                f"The indices tensor is not 1-dimensional"
            )
        return self._get_stack_for_indices(key, indices)

    def _get_stack_for_indices(self, key: str, indices: torch.Tensor):
        """Retrieve `stack_size` samples from each sampled index.

        Args:
            key (str): Field key.
            indices (torch.Tensor): Sampled indicies.

        Raises:
            ValueError: If indicies array isn't 1-D

        Returns:
            torch.Tensor: Sampled buffer values.
        """
        if len(indices.shape) != 1:
            raise ValueError(
                f"The indices tensor is not 1-dimensional"
            )
        backward_indices = torch.arange(-self._stack_size + 1, 1)
        stack_indices = indices.unsqueeze(1) + backward_indices
        stack_indices %= self._capacity
        values_stack = self._storage[key][stack_indices]
        return self._key_to_buffer_element[key].meta.sample_to_output(values_stack)

    @abc.abstractmethod
    def set_index_validity(self, index: int, is_valid: bool) -> None:
        """Set is the index valid for sampling. Depends on buffer type.

        Args:
            index (int): Checkable index.
            is_valid (bool): Index validity.
        """
        raise NotImplementedError()

    def initialize_buffer(self, **kwargs):
        """Call when first element is added to the buffer.

        Raises:
            ValueError: If passed args omit at least one required field.
        """
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
        """Get fields for the batching.

        Returns:
            List[str]: Resulted fields.
        """
        return [
            "indices",
            "step",
            *STATIC_KEYS
        ] + self._additional_keys

    def get_storage_signature(self) -> List[BufferElement]:
        """What is storage inner representation of all elements?

        Returns:
            List[BufferElement]: Storage elements prototypes.
        """
        return self._storage_signature

    @abc.abstractmethod
    def add(self, **kwargs) -> None:
        """Add method. Depends on storage type.
        """
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
        """Current position in a storage. Depends on storage type.

        Returns:
            int: Position in a storage.
        """
        raise NotImplementedError()

    def is_empty(self) -> bool:
        return self.add_calls == 0

    def is_full(self) -> bool:
        return self.add_calls >= self._capacity

    def is_valid_observation(self, index: int) -> bool:
        return self._is_index_valid[index]

    def sample_index_batch(self, batch_size: int) -> torch.Tensor:
        """Sample valid indices of `batch_size`.

        Raises:
            RuntimeError: If there are no valid indices to batch. 

        Returns:
            torch.Tensor: Sampled indices.
        """
        if self._num_valid_indices == 0:
            raise RuntimeError(
                f"Cannot sample {batch_size} since there are no valid indices so far."
            )
        valid_indcs = self._is_index_valid.nonzero().squeeze(1)
        return valid_indcs[torch.randint(valid_indcs.shape[0], (batch_size,))]

    def sample_observation_batch(self, batch_size: int = None, indices: torch.Tensor = None):
        """Returns a batch of observations (including any extra contents).
        If get_observation_fields has been overridden and defines elements not
        stored in self._store, None will be returned and it will be
        left to the child class to fill it. 
        NOTE: This observation contains the indices of the sampled elements. These
        are only valid during the call to sample_observation_batch, i.e. they may
        be used by subclasses of this buffer but may point to different data
        as soon as sampling is done.
        NOTE: Tensors are reshaped. I.e., state is 2-D unless stack_size > 1.
        Scalar values are returned as (batch_size, 1) instead of (batch_size,).

        Args:
            batch_size (int, optional): Number of sequence elements returned. Defaults to None.
            indices (torch.Tensor, optional): The indices of every observation in the batch. Defaults to None.

        Raises:
            TypeError: If custom indicies aren't instantiated from torch.Tensor.
            RuntimeError: If num of indicies and batch size differ.

        Returns:
            _type_: Smapled batch.
        """
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
                f"Indices len {len(indices)} and batch size differ but shouldn't"
            )
        # NOTE: The following implementation is made for the
        # further batch enhancement. As instance, sliding
        # window of `stack_size` from last to first record in an
        # episode.
        forward_indicies = torch.arange(self._episode_capacity)
        episode_step_indcs = indices.unsqueeze(1) + forward_indicies
        episode_step_indcs %= self._capacity

        steps = self._get_steps(episode_step_indcs)
        # NOTE: The following implementation of indices batching
        # could be changed depends on class and batching type.
        margin_indices = (indices + steps - 1) % self._capacity
        margin_indices = margin_indices.unique()
        batch_arrays = []
        # TODO: It's worth to split action-states of episodes
        # as lookup table which contains first record of each episode
        # stored as pd.DataFrame and adjucent records which stored
        # as buffer. So that we could reduce mem usage in contrast to
        # store all data in pd.DataFrame. To do that we need to implement
        # storage serialization/deserialization.
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
