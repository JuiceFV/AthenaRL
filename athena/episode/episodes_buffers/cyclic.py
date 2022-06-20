"""Out-of-graph implementation.
"""
from athena.episode.episodes_buffers.base import EpisodesBuffer


class CyclicEpisodesBuffer(EpisodesBuffer):
    def __init__(
        self,
        capacity: int = 10000,
        batch_size: int = 1,
        episode_capacity: int = 1,
        stack_size: int = 1
    ) -> None:
        if capacity < episode_capacity + stack_size:
            raise ValueError(
                "Storage capacity has to be greater than "
                "joint episode_capacity and stack_size"
            )
        super().__init__(
            capacity=capacity,
            batch_size=batch_size,
            episode_capacity=episode_capacity,
            stack_size=stack_size
        )

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
        prev_pos = (self.iter() - 1) % self._capacity
        # If the current iteration is the beginning of an episode or buffer
        # we add stack_size paddings.
        if self.is_empty() or self._storage["is_last"][prev_pos]:
            self._episodic_num_observation = 0
            for _ in range(self._stack_size - 1):
                self._add(**self._empty_observation)

        curr_pos = self.iter()
        self.set_index_validity(index=curr_pos, is_valid=False)
        if self._episodic_num_observation >= self._episode_capacity:
            start_pos = (curr_pos - self._episode_capacity) % self._capacity
            # Mark the index as valid if the episode capacity is surpassed
            self.set_index_validity(index=start_pos, is_valid=True)
        self._add(**kwargs)
        self._episodic_num_observation += 1

        # Mark next stack_size indicies as not valid (advanced iter by 1)
        for i in range(self._stack_size - 1):
            next_pos = (self.iter() + i) % self._capacity
            self.set_index_validity(index=next_pos, is_valid=False)

        # For the further enhancement mark the previous n indicies
        # as valid for sampling.
        if kwargs["is_last"]:
            num_valid_obs = min(
                self._episodic_num_observation,
                self._episode_capacity
            )
            for i in range(num_valid_obs):
                considarable_pos = (curr_pos - i) % self._capacity
                self.set_index_validity(index=considarable_pos, is_valid=True)

    def iter(self) -> int:
        return self.add_calls % self._capacity
