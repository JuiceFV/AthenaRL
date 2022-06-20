"""In-graph buffer implementation
"""
from athena.episode.episodes_buffers.base import EpisodesBuffer


class FiniteEpisodesBuffer(EpisodesBuffer):
    def __init__(
        self,
        capacity: int = 10000,
        batch_size: int = 1,
        episode_capacity: int = 1,
        stack_size: int = 1
    ) -> None:
        if (capacity < episode_capacity) or (stack_size > episode_capacity):
            raise ValueError(
                f"Expected capacity >= episode_capacity; "
                f"Got {capacity} >= {episode_capacity}. "
                f"Expected stack_size <= episode_capacity; "
                f"Got {stack_size} <= {episode_capacity}."
            )
        super().__init__(
            capacity=capacity,
            batch_size=batch_size,
            episode_capacity=episode_capacity,
            stack_size=stack_size
        )

    def set_index_validity(self, index: int, is_valid: bool) -> None:
        if is_valid:
            self._num_valid_indices += 1
        self._is_index_valid[index] = is_valid

    def add(self, **kwargs) -> None:
        if not self._initialized:
            self.initialize_buffer(**kwargs)

        self._check_addable_signature(**kwargs)
        prev_pos = self.iter() - 1
        if self.is_empty() or self._storage["is_last"][prev_pos]:
            self._episodic_num_observation = 0

        curr_pos = self.iter()
        # If the buffer capacity has been exceeded do not raise the exception.
        # Just let a user be aware about.
        try:
            self.set_index_validity(index=curr_pos, is_valid=False)
        except IndexError:
            self.warning(
                f"The storage capacity has been surpassed: "
                f" the element ({kwargs}) is omitted"
            )
            return

        self._add(**kwargs)
        self._episodic_num_observation += 1
        # For the further enhancement mark the previous n indicies
        # as valid for sampling.
        if kwargs["is_last"]:
            num_valid_obs = min(
                self._episodic_num_observation,
                self._episode_capacity
            )
            for i in range(num_valid_obs):
                considarable_pos = curr_pos - 1 - i
                self.set_index_validity(index=considarable_pos, is_valid=True)

    def iter(self) -> int:
        return self.add_calls
