import numpy as np
import torch
from athena.episode import circular_replay_buffer, sum_tree
from typing import Optional, NamedTuple, List


class PrioritizedReplayBuffer(circular_replay_buffer.ReplayBuffer):

    def __init__(
        self,
        stack_size: int,
        replay_capacity: int,
        batch_size: int,
        update_horizon: int = 1,
        gamma: float = 0.99,
        max_sample_attempts: int = 1000
    ) -> None:
        super(PrioritizedReplayBuffer, self).__init__(
            stack_size=stack_size,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            update_horizon=update_horizon,
            gamma=gamma
        )
        self._max_sample_attempts = max_sample_attempts
        self.sum_tree = sum_tree.SumTree(replay_capacity)

    def _add(self, **kwargs) -> None:
        self._check_addable_quantity(**kwargs)
        transitions = {}
        for element in self.get_storage_signature():
            if element.name == "priority":
                priority = kwargs[element.name]
            else:
                transitions[element.name] = element.metadata.turn_into_storage_element(kwargs[element.name])

        self.sum_tree.set(self.iter(), priority)
        super(PrioritizedReplayBuffer, self)._add_transition(transitions)

    def sample_index_batch(self, batch_size: Optional[int] = None) -> torch.Tensor:
        indices = self.sum_tree.stratified_sample(batch_size)
        allowed_attempts = self._max_sample_attempts
        for i in range(len(indices)):
            if not self.is_valid_transition(indices[i]):
                if allowed_attempts == 0:
                    raise RuntimeError(
                        f"Max sample attempts: Tried {self._max_sample_attempts} times but only sampled {i}"
                        f" valid indices. Batch size is {batch_size}"
                    )
                index = indices[i]
                while not self.is_valid_transition(index) and allowed_attempts > 0:
                    index = self.sum_tree.sample()
                    allowed_attempts -= 1
                indices[i] = index
        return torch.tensor(indices, dtype=torch.int64)

    def sample_transition_batch(
        self, batch_size: Optional[int] = None, indices: Optional[torch.Tensor] = None
    ) -> NamedTuple:
        transition = super(PrioritizedReplayBuffer, self).sample_transition_batch(batch_size, indices)
        batch_arrays = []
        for field in self._transition_fields:
            if field == "sampling_probabilities":
                batch = torch.from_numpy(
                    self.get_priority(transition.indices.numpy().astype(np.int32))
                ).view(batch_size, 1)
            else:
                batch = getattr(transition, field)
            batch_arrays.append(batch)
        return self._batch_type(*batch_arrays)

    def set_priority(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        if indices.dtype != np.int32:
            TypeError(f"Indices must be integers, given {indices.dtype}")
        for index, priority in zip(indices, priorities):
            self.sum_tree.set(index, priority)

    def get_priority(self, indices: np.ndarray):
        if not hasattr(indices, "shape"):
            raise TypeError("Indices must be an array.")
        if indices.dtype != np.int32:
            raise TypeError(f"Indices must be int32s, given: {indices.dtype}")
        batch_size = len(indices)
        priority_batch = np.empty((batch_size), dtype=np.float32)
        for i, memory_index in enumerate(indices):
            priority_batch[i] = self.sum_tree.get(memory_index)
        return priority_batch

    def get_transition_fields(self) -> List[str]:
        parent_transition_fields = super(PrioritizedReplayBuffer, self).get_transition_fields()
        return parent_transition_fields + ["sampling_probabilities"]
