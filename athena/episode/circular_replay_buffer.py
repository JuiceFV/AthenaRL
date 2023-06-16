
import os
import abc
import gzip
import pickle
import logging
import collections
from dataclasses import dataclass
from typing import Dict, List, Any, NamedTuple, Tuple, Iterable, Union, Optional
from numpy.typing import NDArray

try:
    from typing import Final
except ImportError:
    from typing_extensions import Final

import numpy as np
import torch

logger = logging.getLogger(__name__)

IndexT = Iterable[np.int64]
DenseT = Iterable[Union[np.float32, np.float64]]
IDListT = Dict[str, IndexT]
IDScoreListT = Dict[str, Tuple[IndexT, DenseT]]
ValueT = Union[DenseT, IDListT, IDScoreListT]
SparseOutputT = Dict[str, Tuple[torch.Tensor, ...]]

STORE_FILENAME_PREFIX = "$stores$_"
CHECKPOINT_DURATION = 4
STATIC_KEYS = ["observation", "actions", "reward", "terminal"]


@dataclass
class ElementMeta:
    """Base class for elements buffer consists from.
    """
    @classmethod
    @abc.abstractmethod
    def try_create_from_value(cls, name: str, value: ValueT) -> "ElementMeta":
        """Try to create an instance of a given class
        to a specific `value` for `name` key. It's good
        practice to call self.validate after meta initialization.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def validate(self, name: str, value: ValueT) -> None:
        """Look for some inapropriate data/metadata in given `value`.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def initialize_storage(self, capacity: int) -> Union[torch.Tensor, NDArray]:
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
    def try_create_from_value(cls, name: str, value: Iterable) -> "DenseMeta":
        arr_value = np.array(value)
        dtype = arr_value.dtype
        if dtype == np.dtype("float32"):
            dtype = np.dtype("float64")
        obj = cls(arr_value.shape, dtype)
        obj.validate(name, value)
        return obj

    def validate(self, name: str, value: Iterable) -> None:
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

    def empty_observation(self) -> NDArray:
        return np.zeros(self.shape, dtype=self.dtype)

    def sample_to_output(self, sample: torch.Tensor) -> torch.Tensor:
        # Permute so that torch.Size([batch_size, stack_size, observ_size])
        # becomes torch.Size([batch_size, observ_size, stack_size])
        res: torch.Tensor = torch.einsum("ij...->i...j", sample)
        # if stack_size isn't 1-D squeeze it
        return res.squeeze(-1) if res.shape[-1] == 1 else res


@dataclass
class IDListMeta(ElementMeta):

    keys: List[str]

    @classmethod
    def try_create_from_value(cls, name: str, value: IDListT) -> "IDListMeta":
        res = cls(list(value.keys()))
        res.validate(name, value)
        return res

    def validate(self, name: str, value: IDListT) -> None:
        if not isinstance(value, dict):
            raise TypeError(f"{name}: {type(value)} isn't dict")
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(f"{name}: {k} ({type(k)}) is not str")
            if k not in self.keys:
                raise RuntimeError(f"{name}: {k} not in {self.keys}")
            arr_value = np.array(value)
            if len(arr_value) > 0:
                if arr_value.dtype != np.int64:
                    f"{name}: {v} array has dtype {arr_value.dtype}, not np.int64"

    def initialize_storage(self, capacity: int) -> NDArray:
        shape = (capacity,)
        return np.empty(shape, dtype=np.object)

    def turn_into_storage_element(self, value: Any) -> Any:
        return value

    def empty_observation(self) -> IDListT:
        return {key: [] for key in self.keys}

    def sample_to_output(self, sample: torch.Tensor) -> SparseOutputT:
        sample = sample.squeeze(1)
        result: SparseOutputT = {}
        for key in self.keys:
            offsets = []
            ids = []
            for element in sample:
                if element is None:
                    current_ids = []
                else:
                    current_ids = element[key]
                offsets.append(len(ids))
                ids.extend(current_ids)
            result[key] = (
                torch.tensor(offsets, dtype=torch.int32),
                torch.tensor(ids, dtype=torch.int64)
            )
        return result


@dataclass
class IDScoreListMeta(ElementMeta):

    keys: List[str]

    @classmethod
    def try_create_from_value(cls, name: str, value: IDScoreListT) -> "IDScoreListMeta":
        res = cls(list(value.keys()))
        res.validate(name, value)
        return res

    def validate(self, name: str, value: IDScoreListT) -> None:
        if not isinstance(value, dict):
            raise TypeError(f"{name}: {type(value)} isn't dict")
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(f"{name}: {k} ({type(value)}) isn't str")
            if k not in self.keys:
                raise RuntimeError(f"{name}: {k} not in {self.keys}")
            if not isinstance(v, tuple) or len(v) != 2:
                raise RuntimeError(f"{name}: {v} ({type(v)}) isn't len 2 tuple")
            ids = np.array(v[0])
            scores = np.array(v[1])
            if len(ids) != len(scores):
                raise RuntimeError(f"{name}: {len(ids)} != {len(scores)}")
            if len(ids) > 0:
                if ids.dtype != np.int64:
                    raise TypeError(f"{name}: ids dtype {ids.dtype} isn't np.int64")
                if scores.dtype not in (np.float32, np.float64):
                    raise TypeError(f"{name}: scores dtype {scores.dtype} isn't np.float32 or np.float64")

    def initialize_storage(self, capacity: int) -> NDArray:
        shape = (capacity,)
        return np.empty(shape, dtype=np.object)

    def turn_into_storage_element(self, value: Any) -> Any:
        return value

    def empty_observation(self) -> IDScoreListT:
        return {key: ([], []) for key in self.keys}

    def sample_to_output(self, sample: torch.Tensor) -> SparseOutputT:
        sample = sample.squeeze(1)
        result: SparseOutputT = {}
        for key in self.keys:
            offset = []
            ids = []
            scores = []
            for element in sample:
                if element is None:
                    current_ids, current_scores = [], []
                else:
                    current_ids, current_scores = element[key]
                if len(current_ids) != len(current_scores):
                    raise RuntimeError(f"{len(current_ids)} != {len(current_scores)}")
                offset.append(len(ids))
                ids.extend(current_ids)
                scores.extend(current_scores)
            result[key] = (
                torch.tensor(offset, dtype=torch.int32),
                torch.tensor(ids, dtype=torch.int64),
                torch.tensor(scores, dtype=torch.float32)
            )
        return result


class ReplayElement(NamedTuple):
    name: str
    metadata: ElementMeta


def create_replay_element(name: str, value: ValueT) -> ReplayElement:
    if isinstance(value, torch.Tensor):
        raise TypeError("Input shouldn't be tensor")
    metadata = None
    for meta_cls in [DenseMeta, IDListMeta, IDScoreListMeta]:
        try:
            metadata: ElementMeta = meta_cls.try_create_from_value(name, value)
            break
        except Exception as e:
            logger.info(f"Failed attempt to create {meta_cls} from ({name}) {value}: {e}")
    if metadata is None:
        raise ValueError(f"Unable to deduce type for {name}: {value}")
    return ReplayElement(name, metadata)


class ReplayBuffer:
    def __init__(
        self,
        stack_size: int = 1,
        replay_capacity: int = 10000,
        batch_size: int = 1,
        need_as_stack: bool = False,
        need_as_timeline: bool = False,
        update_horizon: int = 1,
        gamma: float = 0.99
    ) -> None:
        if replay_capacity < update_horizon + stack_size:
            raise ValueError(
                "There is not enough capacity to cover update_horizon and stack_size."
            )

        if need_as_timeline:
            if update_horizon <= 1:
                logger.warn(
                    f"Pointless to set need_as_timeline when update_horizon ({update_horizon}) isn't > 1."
                )

        self._initialized: bool = False
        self._stack_size: int = stack_size
        self._need_as_stack: bool = need_as_stack
        self._need_as_timeline: bool = need_as_timeline
        self._replay_capacity: Final[int] = replay_capacity
        self._batch_size: Final[int] = batch_size
        self._update_horizon: Final[int] = update_horizon
        self._gamma: Final[float] = gamma

        self.add_calls: NDArray = np.array(0)

        self._decays: torch.Tensor = (self._gamma ** torch.arange(self._update_horizon)).unsqueeze(0)

        self._is_index_valid: torch.Tensor = torch.zeros(self._replay_capacity, dtype=torch.bool)
        self._num_valid_indices: int = 0
        self._episodic_num_transitions: int = 0

        self._storage: Dict[str, torch.Tensor] = {}
        self._storage_signature: List[ReplayElement] = []
        self._batch_type = collections.namedtuple("filler", [])

        self._additional_keys: List[str] = []
        self._key_to_replay_element: Dict[str, ReplayElement] = {}
        self._empty_transition: Dict[str, Any] = {}
        self._transition_fields: List[str] = []

    def _initialize_storage(self) -> None:
        for element in self.get_storage_signature():
            self._storage[element.name] = element.metadata.initialize_storage(self._replay_capacity)

    def _check_addable_quantity(self, **kwargs) -> None:
        if len(kwargs) != len(self.get_storage_signature()):
            raise ValueError(
                f"Expected: {self.get_storage_signature()}; received {kwargs}"
            )

    def _check_addable_signature(self, **kwargs) -> None:
        self._check_addable_quantity(**kwargs)
        for storage_element in self.get_storage_signature():
            storage_element.metadata.validate(storage_element.name, kwargs[storage_element.name])

    def _add(self, **kwargs) -> None:
        """Inner add method which turns arg value into inner buffer representation.
        """
        self._check_addable_quantity(**kwargs)
        for element in self.get_storage_signature():
            kwargs[element.name] = element.metadata.turn_into_storage_element(kwargs[element.name])
        self._add_transition(kwargs)

    def _add_transition(self, observation: Dict[str, torch.Tensor]) -> None:
        pos = self.iter()
        for name, value in observation.items():
            self._storage[name][pos] = value
        self.add_calls += 1

    def _get_steps(self, nstep_indices: torch.Tensor) -> torch.Tensor:
        terminals = self._storage["terminal"][nstep_indices].to(torch.bool)
        terminals[:, -1] = True
        terminals = terminals.float()
        position_maks = torch.arange(terminals.shape[1] + 1, 1, -1)
        terminals = torch.einsum("ij,j->ij", terminals, position_maks)
        return torch.argmax(terminals, dim=1) + 1

    def _get_batch_for_indices(
        self, key: str, indices: torch.Tensor, steps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if len(indices.shape) != 1:
            raise ValueError("The indices tensor is not 1-dimensional")
        if steps is not None:
            if indices.shape == steps.shape:
                raise ValueError("Indices and steps dims don't fit")
            return [
                self._get_stack_for_indices(key, torch.arange(start_idx, start_idx + step))
                for start_idx, step in zip(indices.tolist(), steps.tolist())
            ]
        else:
            return self._get_stack_for_indices(key, indices)

    def _get_stack_for_indices(self, key: str, indices: torch.Tensor) -> torch.Tensor:
        if len(indices.shape) != 1:
            raise ValueError("The indices tensor is not 1-dimensional")
        backward_indices = torch.arange(-self._stack_size + 1, 1)
        stack_indices = indices.unsqueeze(1) + backward_indices
        stack_indices %= self._replay_capacity
        values_stack = self._storage[key][stack_indices]
        return self._key_to_replay_element[key].metadata.sample_to_output(values_stack)

    def _reduce_nstep_reward(self, nstep_indices: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        masks = torch.arange(self._update_horizon) < steps.unsqueeze(1)
        rewards = self._storage["reward"][nstep_indices] * self._decays * masks
        return rewards.sum(dim=1)

    def iter(self) -> int:
        return self.add_calls % self._replay_capacity

    def initialize_buffer(self, **kwargs) -> None:
        kwargs_keys = set(kwargs.keys())
        if not set(STATIC_KEYS).issubset(kwargs_keys):
            raise ValueError(
                f"{kwargs_keys} doesn't contain all of {STATIC_KEYS}"
            )
        self._additional_keys = list(kwargs_keys - set(STATIC_KEYS))
        for key in STATIC_KEYS + self._additional_keys:
            self._storage_signature.append(create_replay_element(key, kwargs[key]))
        for el in self.get_storage_signature():
            self._key_to_replay_element[el.name] = el
            self._empty_transition[el.name] = el.metadata.empty_observation()

        self._initialize_storage()
        self._transition_fields = self.get_transition_fields()
        self._batch_type = collections.namedtuple("batch_type", self._transition_fields)
        self._initialized = True

    @property
    def size(self) -> int:
        return self._num_valid_indices

    def set_index_validity(self, index: int, is_valid: bool) -> None:
        # Due to the buffer is circular we check if the given index
        # is already valid.
        already_valid = self._is_index_valid[index]
        if not already_valid and is_valid:
            self._num_valid_indices += 1
        elif already_valid and not is_valid:
            self._num_valid_indices -= 1
        if self._num_valid_indices < 0:
            raise ValueError("Number of valid indices is negative")
        self._is_index_valid[index] = is_valid

    def add(self, **kwargs) -> None:
        if not self._initialized:
            self.initialize_buffer(**kwargs)

        self._check_addable_signature(**kwargs)
        prev_pos = (self.iter() - 1) % self._replay_capacity
        if self.is_empty() or self._storage["terminal"][prev_pos]:
            self._episodic_num_transitions = 0
            for _ in range(self._stack_size - 1):
                self._add(**self._empty_transition)

        curr_pos = self.iter()
        self.set_index_validity(index=curr_pos, is_valid=False)
        if self._episodic_num_transitions >= self._update_horizon:
            start_pos = (curr_pos - self._update_horizon) % self._replay_capacity
            self.set_index_validity(index=start_pos, is_valid=True)
        self._add(**kwargs)
        self._episodic_num_transitions += 1

        for i in range(self._stack_size - 1):
            idx = (self.iter() + i) % self._replay_capacity
            self.set_index_validity(index=idx, is_valid=False)

        if kwargs["terminal"]:
            num_valid_obs = min(
                self._episodic_num_transitions,
                self._update_horizon
            )
            for i in range(num_valid_obs):
                considarable_pos = (curr_pos - i) % self._replay_capacity
                self.set_index_validity(index=considarable_pos, is_valid=True)

    def get_storage_signature(self) -> List[ReplayElement]:
        return self._storage_signature

    def get_transition_fields(self) -> List[str]:
        """Get fields for the batching.

        Returns:
            List[str]: Resulted fields.
        """
        additional_keys = []
        for field in self._additional_keys:
            for prefix in ["", "next_"]:
                additional_keys.append(f"{prefix}{field}")
        return [
            "state",
            "actions",
            "reward",
            "next_state",
            "next_actions",
            "next_reward",
            "terminal",
            "indices",
            "step",
            *additional_keys
        ]

    def is_empty(self) -> bool:
        return self.add_calls == 0

    def is_full(self) -> bool:
        return self.add_calls >= self._replay_capacity

    def is_valid_transition(self, index: int) -> torch.Tensor:
        return self._is_index_valid[index]

    def sample_index_batch(self, batch_size: int) -> torch.Tensor:
        if self._num_valid_indices == 0:
            raise RuntimeError(f"Cannot sample {batch_size} since there are no valid indices so far.")
        valid_indcs = self._is_index_valid.nonzero().squeeze(1)
        return valid_indcs[torch.randint(valid_indcs.shape[0], (batch_size,))]

    def sample_all_valid_transitions(self) -> NamedTuple:
        valid_indices = self._is_index_valid.nonzero().squeeze(1)
        if valid_indices.ndim == 1:
            raise ValueError(f"Expecting 1D tensor since is_index_valid is 1D. Got {valid_indices}.")
        return self.sample_transition_batch(batch_size=len(valid_indices), indices=valid_indices)

    def sample_transition_batch(
        self, batch_size: Optional[int] = None, indices: Optional[torch.Tensor] = None
    ) -> NamedTuple:
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

        forward_indices = torch.arange(self._update_horizon)
        nstep_indices = indices.unsqueeze(1) + forward_indices
        nstep_indices %= self._replay_capacity

        steps = self._get_steps(nstep_indices)

        if self._need_as_timeline:
            next_indices = (indices + 1) % self._replay_capacity
            steps_for_timeline_format = steps
        else:
            next_indices = (indices + steps) % self._replay_capacity
            steps_for_timeline_format = None

        batch_arrays = []
        for field in self._transition_fields:
            if field == "state":
                batch = self._get_batch_for_indices("observation", indices)
            elif field == "next_state":
                batch = self._get_batch_for_indices("observation", next_indices, steps_for_timeline_format)
            elif field == "indices":
                batch = indices
            elif field == "terminal":
                terminal_indices = (indices + steps - 1) % self._replay_capacity
                batch = self._storage["terminal"][terminal_indices].to(torch.bool)
            elif field == "reward":
                if self._need_as_timeline or self._need_as_stack:
                    batch = self._get_batch_for_indices("reward", indices, steps_for_timeline_format)
                else:
                    batch = self._reduce_nstep_reward(nstep_indices, steps)
            elif field == "step":
                batch = steps
            elif field in self._storage:
                batch = self._get_batch_for_indices(field, indices)
            elif field.startswith("next_"):
                storage_name = field[len("next_"):]
                if storage_name not in self._storage:
                    raise KeyError(f"{storage_name} is not in {self._storage.keys()}")
                batch = self._get_batch_for_indices(storage_name, next_indices, steps_for_timeline_format)
            else:
                batch = None

            if isinstance(batch, torch.Tensor) and batch.ndim == 1:
                batch = batch.unsqueeze(1)
            batch_arrays.append(batch)
        return self._batch_type(*batch_arrays)

    def _generate_filename(self, checkpoint_dir: str, name: str, suffix: Union[int, str]) -> str:
        return os.path.join(checkpoint_dir, f"{name}_ckpt.{suffix}.gz")

    def _retrieve_checkpointable_members(self) -> Dict[str, Any]:
        checkpointable_members = {}
        for member_name, member in self.__dict__.items():
            if member_name == "_storage":
                for array_name, array in self._storage.items():
                    checkpointable_members[STORE_FILENAME_PREFIX + array_name] = array
            elif not member_name.startswith("_"):
                checkpointable_members[member_name] = member
        return checkpointable_members

    def save(self, checkpoint_dir: str, iter_num: int) -> None:
        if not os.path.exists(checkpoint_dir):
            return

        checkpointable_members = self._retrieve_checkpointable_members()

        for member_name in checkpointable_members:
            filename = self._generate_filename(checkpoint_dir, member_name, iter_num)
            with open(filename, "wb") as file:
                with gzip.GzipFile(fileobj=file) as outfile:
                    if member_name.startswith(STORE_FILENAME_PREFIX):
                        array_name = member_name[len(STORE_FILENAME_PREFIX):]
                        np.save(outfile, self._storage[array_name].numpy(), allow_pickle=False)
                    elif isinstance(self.__dict__[member_name], np.ndarray):
                        np.save(outfile, self.__dict__[member_name], allow_pickle=False)
                    else:
                        pickle.dump(self.__dict__[member_name], outfile)

                    stale_iteration = iter_num - CHECKPOINT_DURATION
                    if stale_iteration >= 0:
                        stale_filename = self._generate_filename(checkpoint_dir, member_name, stale_iteration)
                        try:
                            os.remove(stale_filename)
                        except FileNotFoundError:
                            pass

    def load(self, checkpoint_dir: str, suffix: Union[int, str]) -> None:
        required_members = self._retrieve_checkpointable_members()

        for member_name in required_members:
            filename = self._generate_filename(checkpoint_dir, member_name, suffix)
            if not os.path.exists(filename):
                raise FileNotFoundError(None, None, f"Missing File: {filename}")

        for member_name in required_members:
            filename = self._generate_filename(checkpoint_dir, member_name, suffix)
            with open(filename, "rb") as file:
                with gzip.GzipFile(fileobj=file) as infile:
                    if member_name.startswith(STORE_FILENAME_PREFIX):
                        array_name = member_name[len(STORE_FILENAME_PREFIX):]
                        self._storage[array_name] = torch.from_numpy(np.load(infile, allow_pickle=False))
                    elif isinstance(self.__dict__[member_name], np.ndarray):
                        self.__dict__[member_name] = np.load(infile, allow_pickle=False)
                    else:
                        self.__dict__[member_name] = pickle.load(infile)
